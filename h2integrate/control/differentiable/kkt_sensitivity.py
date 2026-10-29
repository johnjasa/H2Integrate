"""Implicit differentiation of parametric Pyomo optimization problems.

The :class:`KKTSensitivity` class solves a Pyomo model with Ipopt for given parameter values and
linearizes its Karush-Kuhn-Tucker (KKT) conditions at the solution. The resulting
:class:`KKTLinearization` provides Jacobian-vector products in forward mode (``jvp``) and
vector-Jacobian products in reverse mode (``vjp``) for the optimal decision variables with respect
to the parameters, which is what OpenMDAO needs for a matrix-free component.

The approach follows the interior-point KKT sensitivity method used in the NLR
Optimization-Layers-for-Pyomo (PyomoLayers) project (BSD-3-Clause): PyNumero evaluates the
Hessian of the Lagrangian and the constraint Jacobian, and Ipopt's bound multipliers are folded
into the primal Hessian block as ``Sigma = z_L / (x - x_L) + z_U / (x_U - x)``.

Model requirements:

- Exactly one active objective.
- Only equality constraints. Write inequalities with explicit nonnegative slack variables.
- Parameters are scalar Pyomo ``Var`` data objects that the caller passes in; they are fixed
  during each solve and unfixed only to build the PyNumero NLP.
- Variable bounds must not depend on parameters. Write parameter-dependent limits as
  constraints with slack variables instead.
- The PyNumero NLP is built once, so mutable Pyomo ``Param`` values are frozen into it. A
  mutable ``Param`` may change between solves only if it enters the objective linearly (for
  example a cost coefficient), because then it does not affect the Hessian or the constraints.

Setting ``barrier_mu > 0`` adds an explicit log-barrier term, ``-barrier_mu * sum(log(x - x_L))``
(and the matching upper-bound terms), to the objective for every bounded decision variable. The
optimal solution is then a smooth function of the parameters, which removes the kinks of linear
programs and makes the exact derivatives agree with finite differences. Because the barrier is
part of the model, Ipopt converges with its normal tolerances and PyNumero differentiates the
barrier terms exactly.
"""

import os
import shutil
import warnings
from typing import ClassVar
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import pyomo.environ as pyo
from scipy.sparse.linalg import splu


_IPOPT_SEARCHED = False


def _add_idaes_bin_to_path():
    """Prepend the IDAES binary directory (Ipopt, PyNumero ASL) to ``PATH`` if it exists.

    ``idaes get-extensions`` downloads Ipopt and the PyNumero ASL library to a per-user folder
    that is only added to ``PATH`` when ``idaes`` is imported. This helper does the same thing
    without importing IDAES.
    """
    candidates = []
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(Path(local_app_data) / "idaes" / "bin")
    candidates.append(Path.home() / ".idaes" / "bin")

    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    for candidate in candidates:
        if candidate.is_dir() and str(candidate) not in path_entries:
            os.environ["PATH"] = str(candidate) + os.pathsep + os.environ.get("PATH", "")
            from pyomo.common import Executable

            Executable("ipopt").rehash()
            return


def kkt_tools_available() -> bool:
    """Return True if Ipopt and the PyNumero ASL library are both usable.

    Searches the IDAES binary directory the first time it is called.

    Returns:
        bool: True when KKT sensitivities can be computed in this environment.
    """
    global _IPOPT_SEARCHED
    if not _IPOPT_SEARCHED:
        if shutil.which("ipopt") is None:
            _add_idaes_bin_to_path()
        _IPOPT_SEARCHED = True

    from pyomo.contrib.pynumero.asl import AmplInterface

    ipopt_ok = pyo.SolverFactory("ipopt").available(exception_flag=False)
    return bool(ipopt_ok) and bool(AmplInterface.available())


def _require_kkt_tools():
    if not kkt_tools_available():
        msg = (
            "KKT sensitivities need Ipopt and the PyNumero ASL library. Install the optional "
            'dependencies with `pip install -e ".[dol]"` and then run `idaes get-extensions`.'
        )
        raise RuntimeError(msg)


def _symmetrize(hess: sp.spmatrix) -> sp.csr_matrix:
    """Return a full symmetric matrix from a Hessian that may store only one triangle."""
    hess = sp.csr_matrix(hess)
    upper = sp.triu(hess, k=1)
    lower = sp.tril(hess, k=-1)
    if upper.nnz == 0 and lower.nnz > 0:
        return (hess + lower.T).tocsr()
    if lower.nnz == 0 and upper.nnz > 0:
        return (hess + upper.T).tocsr()
    return hess


class KKTLinearization:
    """Factorized KKT system at an optimal point.

    Holds a sparse LU factorization of the reduced KKT matrix

    ``K = [[H_xx + Sigma, J_x^T], [J_x, 0]]``

    and the parameter block ``B = [[H_xp], [J_p]]``, so that ``dz/dp = -K^{-1} B`` where ``z``
    stacks the decision variables and the constraint multipliers.

    Args:
        lu (scipy.sparse.linalg.SuperLU): LU factorization of ``K``.
        param_block (scipy.sparse.csc_matrix): The matrix ``B``.
        n_x (int): Number of decision variables.
        grad_lagrangian_params (np.ndarray): Gradient of the Lagrangian with respect to the
            parameters at the solution, used for the envelope theorem.
    """

    def __init__(self, lu, param_block, n_x, grad_lagrangian_params):
        self._lu = lu
        self._param_block = param_block
        self.n_x = n_x
        self.n_params = param_block.shape[1]
        self._grad_lagrangian_params = grad_lagrangian_params

    def jvp(self, d_params: np.ndarray) -> np.ndarray:
        """Forward-mode product ``(dx/dp) @ d_params``.

        Args:
            d_params (np.ndarray): Parameter perturbation of length ``n_params``.

        Returns:
            np.ndarray: Decision-variable perturbation of length ``n_x``.
        """
        rhs = -(self._param_block @ np.asarray(d_params, dtype=float))
        return self._lu.solve(rhs)[: self.n_x]

    def vjp(self, x_bar: np.ndarray) -> np.ndarray:
        """Reverse-mode product ``(dx/dp)^T @ x_bar``.

        Args:
            x_bar (np.ndarray): Seed on the decision variables, length ``n_x``.

        Returns:
            np.ndarray: Seed on the parameters, length ``n_params``.
        """
        rhs = np.zeros(self._lu.shape[0])
        rhs[: self.n_x] = x_bar
        w = self._lu.solve(rhs, trans="T")
        return -(self._param_block.T @ w)

    def jacobian(self) -> np.ndarray:
        """Dense Jacobian ``dx/dp`` with shape ``(n_x, n_params)``. Intended for small problems."""
        return -self._lu.solve(self._param_block.toarray())[: self.n_x]

    def envelope_gradient(self) -> np.ndarray:
        """Gradient of the optimal objective value with respect to the parameters.

        Uses the envelope theorem, ``dJ*/dp = dL/dp`` at the solution, so it needs no linear
        solve. It is exact when ``barrier_mu`` is zero and approximate (to order ``barrier_mu``)
        otherwise.

        Returns:
            np.ndarray: Gradient of length ``n_params``.
        """
        return self._grad_lagrangian_params.copy()


class KKTSolution:
    """Result of :meth:`KKTSensitivity.solve`.

    The KKT factorization is computed lazily the first time :attr:`linearization` is accessed,
    from the primal and dual values stored at solve time. This keeps analysis-only runs cheap and
    lets several solves share one Pyomo model.

    Attributes:
        x (np.ndarray): Optimal decision variables in the adapter's ordering.
        objective (float): Objective value at the solution.
    """

    def __init__(self, x, objective, sensitivity, primals, duals, z_lower, z_upper):
        self.x = x
        self.objective = objective
        self._sensitivity = sensitivity
        self._state = (primals, duals, z_lower, z_upper)
        self._linearization = None

    @property
    def linearization(self) -> KKTLinearization:
        """KKT factorization at this solution (computed on first access)."""
        if self._linearization is None:
            self._linearization = self._sensitivity._linearize(*self._state)
        return self._linearization


class KKTSensitivity:
    """Solve a parametric Pyomo problem with Ipopt and differentiate it through its KKT system.

    Args:
        model (pyo.ConcreteModel): Model with one active objective and only equality constraints.
        parameters (list): Scalar Pyomo ``VarData`` objects that act as parameters. Their order
            defines the parameter vector used by :meth:`solve`, ``jvp``, and ``vjp``.
        solver_options (dict | None): Extra Ipopt options that override the defaults.
        barrier_mu (float): Weight of the explicit log-barrier smoothing term. Zero solves the
            original problem.
        barrier_offset (float): Small shift inside each barrier logarithm,
            ``log(x - x_L + barrier_offset)``. It keeps the barrier finite when constraints force a
            variable exactly onto its bound (for example a flow limited by a zero input), much
            like Ipopt's own bound relaxation.
    """

    default_solver_options: ClassVar[dict] = {
        "tol": 1e-9,
        "max_iter": 3000,
        "bound_relax_factor": 1e-8,
        "honor_original_bounds": "no",
    }

    def __init__(self, model, parameters, solver_options=None, barrier_mu=0.0, barrier_offset=1e-6):
        _require_kkt_tools()
        from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP

        self.model = model
        self.parameters = list(parameters)
        self.barrier_mu = float(barrier_mu)
        self.barrier_offset = float(barrier_offset)

        objectives = list(model.component_data_objects(pyo.Objective, active=True))
        if len(objectives) != 1:
            msg = f"Model must have exactly one active objective, found {len(objectives)}"
            raise ValueError(msg)
        self._objective = objectives[0]
        if self.barrier_mu > 0.0:
            self._add_barrier_objective()

        for suffix_name in ("dual", "ipopt_zL_out", "ipopt_zU_out"):
            if model.component(suffix_name) is None:
                model.add_component(suffix_name, pyo.Suffix(direction=pyo.Suffix.IMPORT))

        # Build the PyNumero NLP once with the parameters free, so the NL file contains them as
        # variables. Their values are then set with set_primals(), which avoids rewriting the
        # NL file for every parameter change.
        for p in self.parameters:
            p.unfix()
        try:
            self._nlp = PyomoNLP(model)
        finally:
            for p in self.parameters:
                p.fix()

        if self._nlp.n_ineq_constraints() > 0:
            raise ValueError(
                "KKTSensitivity supports only equality constraints. Rewrite inequalities with "
                "nonnegative slack variables."
            )

        self._all_vars = self._nlp.get_pyomo_variables()
        self._constraints = self._nlp.get_pyomo_constraints()
        self._param_idx = np.asarray(self._nlp.get_primal_indices(self.parameters), dtype=int)
        is_param = np.zeros(self._nlp.n_primals(), dtype=bool)
        is_param[self._param_idx] = True
        self._x_idx = np.flatnonzero(~is_param)
        self._x_vars = [self._all_vars[i] for i in self._x_idx]
        self._position_in_x = np.full(self._nlp.n_primals(), -1, dtype=int)
        self._position_in_x[self._x_idx] = np.arange(self._x_idx.size)

        self._x_lb = self._nlp.primals_lb()[self._x_idx]
        self._x_ub = self._nlp.primals_ub()[self._x_idx]

        self.solver_options = dict(self.default_solver_options)
        if solver_options:
            self.solver_options.update(solver_options)
        self._solver = pyo.SolverFactory("ipopt")

    def _add_barrier_objective(self):
        """Replace the objective with ``f - mu * sum(log(bound gaps))`` for smoothing.

        Only decision variables that appear in the active constraints or objective get barrier
        terms, so a variable that is otherwise unused cannot drive the objective to minus
        infinity. The original objective stays on the model (deactivated) so that
        :attr:`KKTSolution.objective` reports its value.
        """
        from pyomo.core.expr.visitor import identify_variables

        param_ids = {id(p) for p in self.parameters}
        used = {}
        for con in self.model.component_data_objects(pyo.Constraint, active=True):
            for v in identify_variables(con.body, include_fixed=True):
                used[id(v)] = v
        for v in identify_variables(self._objective.expr, include_fixed=True):
            used[id(v)] = v

        terms = []
        offset = self.barrier_offset
        for v in used.values():
            if id(v) in param_ids:
                continue
            if v.lb is not None:
                terms.append(pyo.log(v - v.lb + offset))
            if v.ub is not None:
                terms.append(pyo.log(v.ub - v + offset))
        sign = 1.0 if self._objective.sense == pyo.minimize else -1.0
        barrier = -sign * self.barrier_mu * sum(terms)
        self._objective.deactivate()
        self.model.kkt_barrier_objective = pyo.Objective(
            expr=self._objective.expr + barrier, sense=self._objective.sense
        )

    @property
    def n_x(self) -> int:
        """Number of decision variables."""
        return self._x_idx.size

    @property
    def n_params(self) -> int:
        """Number of parameters."""
        return self._param_idx.size

    def variable_indices(self, variables) -> np.ndarray:
        """Positions of Pyomo variables in the decision-variable vector.

        Args:
            variables (list | pyo.Var): Scalar ``VarData`` objects or an indexed ``Var``
                (expanded in index order).

        Returns:
            np.ndarray: Integer positions into :attr:`KKTSolution.x`.
        """
        if isinstance(variables, pyo.Var) and variables.is_indexed():
            variables = list(variables.values())
        nlp_idx = np.asarray(self._nlp.get_primal_indices(list(variables)), dtype=int)
        positions = self._position_in_x[nlp_idx]
        if np.any(positions < 0):
            raise ValueError("Some requested variables are parameters, not decision variables.")
        return positions

    def solve(self, param_values, linearize=False) -> KKTSolution:
        """Solve the problem for the given parameter values.

        The current values of the decision variables are used as Ipopt's starting point, so
        repeated calls with nearby parameters are warm started.

        Args:
            param_values (array_like): Parameter values in the order given at construction.
            linearize (bool): If True, factorize the KKT system now instead of on first use.

        Returns:
            KKTSolution: Optimal decision variables and objective, with lazy linearization.

        Raises:
            RuntimeError: If Ipopt does not report an optimal solution.
        """
        param_values = np.asarray(param_values, dtype=float)
        if param_values.size != self.n_params:
            msg = f"Expected {self.n_params} parameter values, got {param_values.size}"
            raise ValueError(msg)
        for p, val in zip(self.parameters, param_values):
            p.fix(float(val))

        results = self._solver.solve(
            self.model, tee=False, options=self.solver_options, load_solutions=False
        )
        condition = results.solver.termination_condition
        message = str(results.solver.message or "")
        if condition not in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.locallyOptimal,
        ):
            raise RuntimeError(
                f"Ipopt did not converge (termination condition: {condition}; message: {message})"
            )
        self.model.solutions.load_from(results)
        model_dual = self.model.dual
        z_lower_suffix = self.model.ipopt_zL_out
        z_upper_suffix = self.model.ipopt_zU_out

        primals = np.array([v.value for v in self._all_vars], dtype=float)
        self._nlp.set_primals(primals)
        residual = np.abs(self._nlp.evaluate_eq_constraints())
        if residual.size and residual.max() > 1e-6 * max(1.0, np.abs(primals).max()):
            msg = f"Ipopt returned an infeasible point (max residual {residual.max():.3e})"
            raise RuntimeError(msg)
        objective = float(pyo.value(self._objective))
        # PyNumero uses L = f + lambda^T c, while Pyomo's Ipopt interface returns -lambda.
        duals = -np.array([model_dual.get(c, 0.0) for c in self._constraints], dtype=float)
        z_lower = np.abs(np.array([z_lower_suffix.get(v, 0.0) for v in self._x_vars]))
        z_upper = np.abs(np.array([z_upper_suffix.get(v, 0.0) for v in self._x_vars]))
        solution = KKTSolution(
            primals[self._x_idx], objective, self, primals, duals, z_lower, z_upper
        )
        if linearize:
            _ = solution.linearization
        return solution

    def _linearize(self, primals, duals, z_lower, z_upper) -> KKTLinearization:
        nlp = self._nlp
        nlp.set_primals(primals)
        nlp.set_duals(duals)
        nlp.set_obj_factor(1.0)

        hess = _symmetrize(nlp.evaluate_hessian_lag())
        jac = sp.csr_matrix(nlp.evaluate_jacobian())

        x = primals[self._x_idx]
        relax = float(self.solver_options.get("bound_relax_factor", 0.0))
        sigma = np.zeros(self.n_x)
        has_lb = np.isfinite(self._x_lb)
        has_ub = np.isfinite(self._x_ub)
        tiny = 1e-300
        lb = self._x_lb[has_lb]
        ub = self._x_ub[has_ub]
        gap_lb = x[has_lb] - lb + relax * np.maximum(1.0, np.abs(lb))
        gap_ub = ub - x[has_ub] + relax * np.maximum(1.0, np.abs(ub))
        sigma[has_lb] += z_lower[has_lb] / np.maximum(gap_lb, tiny)
        sigma[has_ub] += z_upper[has_ub] / np.maximum(gap_ub, tiny)

        hess_x = hess[self._x_idx, :]
        h_xx = hess_x[:, self._x_idx] + sp.diags(sigma)
        h_xp = hess_x[:, self._param_idx]
        j_x = jac[:, self._x_idx]
        j_p = jac[:, self._param_idx]

        kkt = sp.bmat([[h_xx, j_x.T], [j_x, None]], format="csc")
        param_block = sp.vstack([h_xp, j_p], format="csc")
        lu = self._factorize(kkt)

        grad_f = nlp.evaluate_grad_objective()
        grad_lagrangian_params = grad_f[self._param_idx] + j_p.T @ duals
        return KKTLinearization(lu, param_block, self.n_x, np.asarray(grad_lagrangian_params))

    def _factorize(self, kkt):
        try:
            return splu(kkt)
        except RuntimeError:
            # Singular KKT matrix (for example a degenerate LP without smoothing). Add a small
            # dual regularization, as Ipopt does, and warn that derivatives may be inaccurate.
            n_c = kkt.shape[0] - self.n_x
            scale = max(1.0, abs(kkt).max())
            delta = 1e-10 * scale
            reg = sp.block_diag([sp.identity(self.n_x) * delta, -sp.identity(n_c) * delta])
            warnings.warn(
                "KKT matrix is singular; using a regularized factorization. Consider enabling "
                "quadratic or barrier smoothing.",
                RuntimeWarning,
                stacklevel=3,
            )
            return splu((kkt + reg).tocsc())
