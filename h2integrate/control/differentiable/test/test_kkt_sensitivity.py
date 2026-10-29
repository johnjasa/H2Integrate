import numpy as np
import pytest
import pyomo.environ as pyo
from pyomo.opt import SolverResults

from h2integrate.control.differentiable.kkt_sensitivity import KKTSensitivity, kkt_tools_available


pytestmark = pytest.mark.skipif(
    not kkt_tools_available(),
    reason="Ipopt and the PyNumero ASL library are not available (install the dol extra)",
)

P0 = 0.3


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
