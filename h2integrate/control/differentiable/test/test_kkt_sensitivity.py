import numpy as np
import pytest
import pyomo.environ as pyo
from pyomo.opt import SolverResults

from h2integrate.control.differentiable.cached_ipopt import CachedIpopt, cached_ipopt_available
from h2integrate.control.differentiable.kkt_sensitivity import KKTSensitivity, kkt_tools_available


pytestmark = pytest.mark.skipif(
    not kkt_tools_available(),
    reason="Ipopt and the PyNumero ASL library are not available (install the dol extra)",
)

P0 = 0.3
requires_cached_ipopt = pytest.mark.skipif(
    not cached_ipopt_available(), reason="the Ipopt shared library is not available"
)


def _equality_qp():
    """min (x1 - p)^2 + (x2 - 2p)^2 s.t. x1 + x2 = 1, with x1 = (1 - p)/2 and x2 = (1 + p)/2."""
    m = pyo.ConcreteModel()
    m.p = pyo.Var(initialize=P0)
    m.p.fix()
    m.x1 = pyo.Var(initialize=0.0)
    m.x2 = pyo.Var(initialize=0.0)
    m.sum_one = pyo.Constraint(expr=m.x1 + m.x2 == 1)
    m.obj = pyo.Objective(expr=(m.x1 - m.p) ** 2 + (m.x2 - 2 * m.p) ** 2)
    return m


def _bounded_qp():
    """QP with an active bound, a slack-based inequality, and a bilinear parameter term."""
    m = pyo.ConcreteModel()
    m.p = pyo.Var(range(3), initialize=1.0)
    for i in range(3):
        m.p[i].fix()
    m.x = pyo.Var(range(3), bounds=(0, None), initialize=1.0)
    m.slack = pyo.Var(bounds=(0, None), initialize=1.0)
    m.total = pyo.Constraint(expr=m.x[0] + m.x[1] + m.x[2] == m.p[0])
    m.limit = pyo.Constraint(expr=m.p[1] - m.x[0] - m.slack == 0)
    m.coupling = pyo.Constraint(expr=m.x[2] * m.p[2] + m.x[1] == m.p[0])
    m.obj = pyo.Objective(
        expr=(m.x[0] - m.p[2]) ** 2 + 2 * m.x[1] ** 2 + 3 * m.x[2] ** 2 + m.p[2] * m.x[1]
    )
    return m


def _fd_jacobian(sens, idx, p0, step=1e-6, objective=False):
    cols = []
    for k in range(p0.size):
        plus = p0.copy()
        minus = p0.copy()
        plus[k] += step
        minus[k] -= step
        sol_plus = sens.solve(plus)
        sol_minus = sens.solve(minus)
        if objective:
            cols.append((sol_plus.objective - sol_minus.objective) / (2 * step))
        else:
            cols.append((sol_plus.x[idx] - sol_minus.x[idx]) / (2 * step))
    return np.array(cols).T


@pytest.mark.unit
def test_equality_qp_matches_analytic(subtests):
    m = _equality_qp()
    sens = KKTSensitivity(m, [m.p])
    idx = sens.variable_indices([m.x1, m.x2])
    solution = sens.solve([P0])
    lin = solution.linearization

    with subtests.test("solution"):
        assert solution.x[idx] == pytest.approx([(1 - P0) / 2, (1 + P0) / 2], abs=1e-8)
    with subtests.test("jacobian"):
        assert lin.jacobian()[idx, 0] == pytest.approx([-0.5, 0.5], abs=1e-10)
    with subtests.test("jvp"):
        assert lin.jvp([2.0])[idx] == pytest.approx([-1.0, 1.0], abs=1e-10)
    with subtests.test("vjp"):
        x_bar = np.zeros(sens.n_x)
        x_bar[idx] = [1.0, 3.0]
        assert lin.vjp(x_bar) == pytest.approx([1.0], abs=1e-10)


@pytest.mark.unit
@pytest.mark.parametrize("barrier_mu", [0.0, 1e-4])
def test_bounded_qp_matches_finite_differences(subtests, barrier_mu):
    m = _bounded_qp()
    sens = KKTSensitivity(m, [m.p[0], m.p[1], m.p[2]], barrier_mu=barrier_mu)
    idx = sens.variable_indices(m.x)
    p0 = np.array([3.0, 0.5, 1.2])
    lin = sens.solve(p0).linearization
    jac = lin.jacobian()[idx]
    jac_fd = _fd_jacobian(sens, idx, p0)

    with subtests.test("jacobian vs finite differences"):
        assert jac == pytest.approx(jac_fd, abs=1e-6)
    with subtests.test("jvp and vjp consistent with jacobian"):
        v = np.array([0.3, -1.0, 2.0])
        w = np.array([1.0, 2.0, -0.5])
        x_bar = np.zeros(sens.n_x)
        x_bar[idx] = w
        assert lin.jvp(v)[idx] == pytest.approx(jac @ v, abs=1e-8)
        assert lin.vjp(x_bar) == pytest.approx(jac.T @ w, abs=1e-8)


@pytest.mark.unit
def test_envelope_gradient_matches_finite_differences():
    m = _bounded_qp()
    sens = KKTSensitivity(m, [m.p[0], m.p[1], m.p[2]])
    p0 = np.array([3.0, 0.5, 1.2])
    gradient = sens.solve(p0).linearization.envelope_gradient()
    gradient_fd = _fd_jacobian(sens, None, p0, objective=True)
    assert gradient == pytest.approx(gradient_fd, abs=1e-5)


@pytest.mark.unit
def test_invalid_models_raise(subtests):
    with subtests.test("inequality constraint"):
        m = _equality_qp()
        m.extra = pyo.Constraint(expr=m.x1 <= 5)
        with pytest.raises(ValueError, match="only equality constraints"):
            KKTSensitivity(m, [m.p])
    with subtests.test("two objectives"):
        m = _equality_qp()
        m.obj2 = pyo.Objective(expr=m.x1)
        with pytest.raises(ValueError, match="exactly one active objective"):
            KKTSensitivity(m, [m.p])
    with subtests.test("wrong parameter count"):
        m = _equality_qp()
        sens = KKTSensitivity(m, [m.p])
        with pytest.raises(ValueError, match="Expected 1 parameter values"):
            sens.solve([1.0, 2.0])


@pytest.mark.unit
def test_infeasible_problem_raises():
    m = pyo.ConcreteModel()
    m.p = pyo.Var(initialize=-1.0)
    m.p.fix()
    m.x = pyo.Var(bounds=(0, None), initialize=1.0)
    m.c = pyo.Constraint(expr=m.x == m.p)
    m.obj = pyo.Objective(expr=m.x**2)
    sens = KKTSensitivity(m, [m.p])
    with pytest.raises(RuntimeError):
        sens.solve([-1.0])


@pytest.mark.unit
def test_stalled_solver_is_not_treated_as_optimal(monkeypatch):
    model = _equality_qp()
    sensitivity = KKTSensitivity(model, [model.p])
    results = pyo.SolverFactory("ipopt")

    def stalled(*args, **kwargs):
        response = SolverResults()
        response.solver.termination_condition = pyo.TerminationCondition.internalSolverError
        response.solver.message = "Search Direction becomes Too Small"
        return response

    monkeypatch.setattr(results, "solve", stalled)
    sensitivity._solver = results
    with pytest.raises(RuntimeError, match="Search Direction becomes Too Small"):
        sensitivity.solve([P0])


@pytest.mark.unit
@requires_cached_ipopt
@pytest.mark.parametrize("barrier_mu", [0.0, 1e-4])
def test_cached_solver_matches_shell_and_derivatives(barrier_mu, subtests):
    models = [_bounded_qp(), _bounded_qp()]
    shell, cached = [
        KKTSensitivity(
            model,
            [model.p[0], model.p[1], model.p[2]],
            barrier_mu=barrier_mu,
            solver_backend=backend,
        )
        for model, backend in zip(models, ("shell", "cached"))
    ]
    for values in (np.array([3.0, 0.5, 1.2]), np.array([3.1, 0.6, 1.1])):
        shell_solution = shell.solve(values)
        cached_solution = cached.solve(values)
        with subtests.test("primals and objective", values=values):
            assert cached_solution.x == pytest.approx(shell_solution.x, abs=1e-7)
            assert cached_solution.objective == pytest.approx(shell_solution.objective, abs=1e-8)
        with subtests.test("jacobian", values=values):
            assert cached_solution.linearization.jacobian() == pytest.approx(
                shell_solution.linearization.jacobian(), abs=1e-6
            )
        with subtests.test("model duals and bound multipliers", values=values):
            for con0, con1 in zip(shell._constraints, cached._constraints):
                assert models[1].dual[con1] == pytest.approx(models[0].dual[con0], abs=1e-7)
            for name in ("ipopt_zL_out", "ipopt_zU_out"):
                for v0, v1 in zip(shell._x_vars, cached._x_vars):
                    assert getattr(models[1], name).get(v1, 0.0) == pytest.approx(
                        getattr(models[0], name).get(v0, 0.0), abs=1e-7
                    )


@pytest.mark.unit
@requires_cached_ipopt
def test_cached_solver_refreshes_mutable_parameters():
    model = _equality_qp()
    model.coefficient = pyo.Param(initialize=1.0, mutable=True)
    model.obj.set_value(model.obj.expr + model.coefficient * model.x1 * model.p)
    sensitivity = KKTSensitivity(
        model, [model.p], solver_backend="cached", solver_options={"warm_start_init_point": "yes"}
    )
    first = sensitivity.solve([P0])
    old_solver = sensitivity._cached_solver
    model.coefficient.set_value(2.0)
    second = sensitivity.solve([P0])
    assert sensitivity._cached_solver is not old_solver
    assert first.x != pytest.approx(second.x, abs=1e-4)
    for coefficient, cached_solution in ((1.0, first), (2.0, second)):
        shell_model = _equality_qp()
        shell_model.coefficient = pyo.Param(initialize=coefficient, mutable=True)
        shell_model.obj.set_value(
            shell_model.obj.expr + shell_model.coefficient * shell_model.x1 * shell_model.p
        )
        shell = KKTSensitivity(shell_model, [shell_model.p])
        reference = shell.solve([P0])
        assert cached_solution.x == pytest.approx(reference.x, abs=1e-8)
        assert cached_solution.linearization.jacobian() == pytest.approx(
            reference.linearization.jacobian(), abs=1e-8
        )


@pytest.mark.unit
@requires_cached_ipopt
def test_cached_solver_rejects_nonoptimal_and_unsupported_cases():
    model = _equality_qp()
    model.obj.sense = pyo.maximize
    with pytest.raises(ValueError, match="minimization only"):
        KKTSensitivity(model, [model.p], solver_backend="cached")
    model = pyo.ConcreteModel()
    model.p = pyo.Var(initialize=-1.0)
    model.p.fix()
    model.x = pyo.Var(bounds=(0, None), initialize=1.0)
    model.c = pyo.Constraint(expr=model.x == model.p)
    model.obj = pyo.Objective(expr=model.x**2)
    sensitivity = KKTSensitivity(model, [model.p], solver_backend="cached")
    with pytest.raises(RuntimeError, match="did not converge"):
        sensitivity.solve([-1.0])


@pytest.mark.unit
@requires_cached_ipopt
def test_cached_solver_rejects_export_suffixes_and_relaxed_parameters():
    model = _equality_qp()
    model.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    model.scaling_factor[model.x1] = 2.0
    with pytest.raises(ValueError, match="does not support export suffix"):
        KKTSensitivity(model, [model.p], solver_backend="cached")
    model = _equality_qp()
    sensitivity = KKTSensitivity(model, [model.p], solver_backend="cached")
    model.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    model.scaling_factor[model.x1] = 2.0
    with pytest.raises(ValueError, match="does not support export suffix"):
        sensitivity.solve([P0])
    model.del_component(model.scaling_factor)
    sensitivity.solver_options["fixed_variable_treatment"] = "relax_bounds"
    with pytest.raises(ValueError, match="requires fixed_variable_treatment"):
        sensitivity.solve([P0])
    sensitivity.solver_options.pop("fixed_variable_treatment")
    model.x1.setlb(-1.0)
    with pytest.raises(RuntimeError, match="changing decision-variable bounds"):
        sensitivity.solve([P0])


@pytest.mark.unit
@requires_cached_ipopt
def test_cached_solver_rejects_problem_mutation(subtests):
    with subtests.test("objective"):
        model = _equality_qp()
        sensitivity = KKTSensitivity(model, [model.p], solver_backend="cached")
        model.obj.set_value(model.x1**2)
        with pytest.raises(RuntimeError, match="changing the objective"):
            sensitivity.solve([P0])
    with subtests.test("constraints"):
        model = _equality_qp()
        sensitivity = KKTSensitivity(model, [model.p], solver_backend="cached")
        model.sum_one.set_value(model.x1 + model.x2 == 2)
        with pytest.raises(RuntimeError, match="changing constraints"):
            sensitivity.solve([P0])
    with subtests.test("other fixed variables"):
        model = _equality_qp()
        model.q = pyo.Var(initialize=1.0)
        model.q.fix()
        model.obj.set_value(model.obj.expr + model.q * model.x1)
        sensitivity = KKTSensitivity(model, [model.p], solver_backend="cached")
        model.q.set_value(2.0)
        with pytest.raises(RuntimeError, match="changing other fixed variables"):
            sensitivity.solve([P0])
    with subtests.test("fixed decision variables"):
        model = _equality_qp()
        sensitivity = KKTSensitivity(model, [model.p], solver_backend="cached")
        model.x1.fix()
        with pytest.raises(RuntimeError, match="fixing decision variables"):
            sensitivity.solve([P0])


@pytest.mark.unit
@requires_cached_ipopt
def test_cached_solver_surfaces_callback_errors(monkeypatch):
    model = _equality_qp()
    sensitivity = KKTSensitivity(model, [model.p], solver_backend="cached")

    def fail():
        raise ValueError("objective evaluation failed")

    monkeypatch.setattr(sensitivity._nlp, "evaluate_objective", fail)
    with pytest.raises(RuntimeError, match="could not evaluate") as exc:
        sensitivity.solve([P0])
    assert str(exc.value.__cause__) == "objective evaluation failed"


@pytest.mark.unit
def test_cached_solver_reports_unavailable_library(monkeypatch):
    def missing(_self):
        raise RuntimeError("no shared Ipopt library")

    monkeypatch.setattr(CachedIpopt, "_load_library", missing)
    assert not cached_ipopt_available()
    model = _equality_qp()
    with pytest.raises(RuntimeError, match="no shared Ipopt library"):
        KKTSensitivity(model, [model.p], solver_backend="cached")


@pytest.mark.unit
@requires_cached_ipopt
def test_cached_solver_warm_starts_primal_dual(monkeypatch, subtests):
    models = [_bounded_qp(), _bounded_qp()]
    warm, cold = [
        KKTSensitivity(
            model,
            [model.p[0], model.p[1], model.p[2]],
            barrier_mu=1e-4,
            solver_backend="cached",
            solver_options={"warm_start_init_point": mode},
        )
        for model, mode in zip(models, ("yes", "no"))
    ]
    library = warm._cached_solver._lib
    original_solve = library.IpoptSolve
    original_option = library.AddIpoptStrOption
    starts = []
    option_values = []

    def capture_option(problem, name, value):
        if name == b"warm_start_init_point":
            option_values.append(value)
        return original_option(problem, name, value)

    def capture_solve(problem, x, g, objective, dual, lower, upper, user_data):
        starts.append(
            (
                np.ctypeslib.as_array(dual, shape=(warm._nlp.n_constraints(),)).copy(),
                np.ctypeslib.as_array(lower, shape=(warm._nlp.n_primals(),)).copy(),
                np.ctypeslib.as_array(upper, shape=(warm._nlp.n_primals(),)).copy(),
            )
        )
        return original_solve(problem, x, g, objective, dual, lower, upper, user_data)

    monkeypatch.setattr(library, "AddIpoptStrOption", capture_option)
    monkeypatch.setattr(library, "IpoptSolve", capture_solve)
    for values in (np.array([3.0, 0.5, 1.2]), np.array([3.1, 0.6, 1.1])):
        warm_solution = warm.solve(values)
        cold_solution = cold.solve(values)
        if len(starts) == 1:
            first_multipliers = tuple(x.copy() for x in warm._cached_solver._previous_multipliers)
        with subtests.test("solution and derivatives", values=values):
            assert warm_solution.x == pytest.approx(cold_solution.x, abs=1e-7)
            assert warm_solution.objective == pytest.approx(cold_solution.objective, abs=1e-8)
            assert warm_solution.linearization.jacobian() == pytest.approx(
                cold_solution.linearization.jacobian(), abs=1e-6
            )
    with subtests.test("warm-start inputs"):
        assert option_values == [b"no", b"yes"]
        assert all(np.count_nonzero(value) == 0 for value in starts[0])
        for actual, expected in zip(starts[1], first_multipliers):
            np.testing.assert_array_equal(actual, expected)

    with subtests.test("failed solve clears saved multipliers"):
        monkeypatch.setattr(library, "IpoptSolve", lambda *_args: 3)
        with pytest.raises(RuntimeError, match="did not converge"):
            warm.solve(np.array([3.2, 0.6, 1.1]))
        assert warm._cached_solver._previous_multipliers is None
        monkeypatch.setattr(library, "IpoptSolve", capture_solve)
        warm.solve(np.array([3.0, 0.5, 1.2]))
        assert option_values[-1] == b"no"
