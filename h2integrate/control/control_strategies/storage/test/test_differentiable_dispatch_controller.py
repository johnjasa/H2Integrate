import numpy as np
import pytest
import openmdao.api as om

from h2integrate.storage.storage_performance_model import StoragePerformanceModel
from h2integrate.control.differentiable.cached_ipopt import cached_ipopt_available
from h2integrate.control.differentiable.kkt_sensitivity import kkt_tools_available
from h2integrate.control.control_strategies.storage.demand_openloop_storage_controller import (
    DemandOpenLoopStorageController,
)
from h2integrate.control.control_strategies.storage.differentiable_dispatch_controller import (
    DifferentiableDispatchStorageController,
    DifferentiableDispatchStorageControllerConfig,
)


pytestmark = pytest.mark.skipif(
    not kkt_tools_available(),
    reason="Ipopt and the PyNumero ASL library are not available (install the dol extra)",
)

N_STEPS = 24


def _storage_parameters(**overrides):
    params = {
        "commodity": "electricity",
        "commodity_rate_units": "MW",
        "commodity_amount_units": "MW*h",
        "max_capacity": 40.0,
        "max_charge_rate": 10.0,
        "min_soc_fraction": 0.1,
        "max_soc_fraction": 0.9,
        "init_soc_fraction": 0.5,
        "charge_equals_discharge": True,
        "charge_efficiency": 0.95,
        "discharge_efficiency": 0.95,
        "demand_profile": 6.0,
    }
    params.update(overrides)
    return params


def _profiles(n=N_STEPS):
    t = np.arange(n)
    generation = 8.0 + 7.0 * np.sin(2 * np.pi * t / 24.0) + 2.0 * np.cos(2 * np.pi * t / 7.0)
    generation = np.clip(generation, 0.5, None)
    demand = 6.0 + 0.5 * np.sin(t)
    return generation, demand


def _plant_config(n=N_STEPS):
    return {"plant": {"plant_life": 30, "simulation": {"n_timesteps": n, "dt": 3600}}}


def _build_problem(n=N_STEPS, controller_cls=DifferentiableDispatchStorageController, **overrides):
    params = _storage_parameters(**overrides)
    tech_config = {"model_inputs": {"shared_parameters": params}}
    generation, demand = _profiles(n)

    prob = om.Problem()
    ivc = prob.model.add_subsystem("ivc", om.IndepVarComp(), promotes=["*"])
    ivc.add_output("electricity_in", generation, units="MW")
    ivc.add_output("electricity_set_point", demand, units="MW")
    ivc.add_output("storage_capacity", params["max_capacity"], units="MW*h")
    ivc.add_output("max_charge_rate", params["max_charge_rate"], units="MW")
    if not params["charge_equals_discharge"]:
        ivc.add_output("max_discharge_rate", params["max_discharge_rate"], units="MW")
    prob.model.add_subsystem(
        "controller",
        controller_cls(plant_config=_plant_config(n), tech_config=tech_config),
        promotes=["*"],
    )
    prob.setup()
    return prob


def _input_vector(prob):
    names = ["electricity_in", "electricity_set_point", "storage_capacity", "max_charge_rate"]
    return names, [np.atleast_1d(prob.get_val(name)).copy() for name in names]


def _output_vector(prob):
    names = [
        "electricity_command_value",
        "dispatch_electricity_stored",
        "dispatch_electricity_unmet_demand",
        "dispatch_cost",
    ]
    return np.concatenate([np.atleast_1d(prob.get_val(name)) for name in names])


def _directional_fd(prob, direction, step=1e-4):
    names, base = _input_vector(prob)
    results = []
    for sign in (1.0, -1.0):
        for name, value, delta in zip(names, base, direction):
            prob.set_val(name, value + sign * step * delta)
        prob.run_model()
        results.append(_output_vector(prob))
    for name, value in zip(names, base):
        prob.set_val(name, value)
    prob.run_model()
    return (results[0] - results[1]) / (2 * step)


def _jacvec(prob, direction):
    names, _ = _input_vector(prob)
    seeds = dict(zip([f"ivc.{name}" for name in names], direction))
    totals = prob.compute_jacvec_product(
        of=[
            "electricity_command_value",
            "dispatch_electricity_stored",
            "dispatch_electricity_unmet_demand",
            "dispatch_cost",
        ],
        wrt=[f"ivc.{name}" for name in names],
        mode="fwd",
        seed=seeds,
    )
    return np.concatenate([np.atleast_1d(v) for v in totals.values()])


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides",
    [{}, {"charge_equals_discharge": False, "max_discharge_rate": 6.0}],
    ids=["equal_rates", "separate_discharge_rate"],
)
def test_full_horizon_partials_match_finite_differences(overrides):
    prob = _build_problem(**overrides)
    prob.run_model()
    data = prob.check_partials(out_stream=None, method="fd", step=1e-5, form="central")
    for key, val in data["controller"].items():
        for direction in ("J_fwd", "J_rev"):
            scale = max(1.0, np.abs(val["J_fd"]).max())
            assert np.abs(val[direction] - val["J_fd"]).max() / scale < 1e-3, (key, direction)


@pytest.mark.unit
def test_rolling_derivatives(subtests):
    prob = _build_problem(dispatch_horizon_mode="rolling", n_horizon_window=8, n_control_window=4)
    prob.run_model()
    rng = np.random.default_rng(0)
    _, base = _input_vector(prob)

    with subtests.test("forward products match directional finite differences"):
        for _ in range(2):
            direction = [rng.standard_normal(value.shape) for value in base]
            assert _jacvec(prob, direction) == pytest.approx(
                _directional_fd(prob, direction), rel=1e-3, abs=1e-4
            )

    with subtests.test("forward and reverse products are transposes"):
        prob.run_model()
        comp = prob.model.controller
        n = N_STEPS
        v = [rng.standard_normal(value.shape) for value in base]
        w = {
            "electricity_command_value": rng.standard_normal(n),
            "dispatch_electricity_stored": rng.standard_normal(n),
            "dispatch_electricity_unmet_demand": rng.standard_normal(n),
            "dispatch_cost": rng.standard_normal(1),
        }
        d_vars = comp._forward(v[0], v[1], v[2][0], v[3][0], v[3][0])
        jv = {
            "electricity_command_value": d_vars["discharge"] - d_vars["charge"],
            "dispatch_electricity_stored": d_vars["stored"],
            "dispatch_electricity_unmet_demand": d_vars["unmet"],
            "dispatch_cost": np.atleast_1d(comp._dispatch_cost(d_vars)),
        }
        lhs = sum(float(np.dot(w[k], jv[k])) for k in w)
        bars = comp._output_seeds(w)
        bar_in, bar_demand, bar_cap, bar_charge, bar_discharge = comp._reverse(bars)
        rhs = (
            np.dot(bar_in, v[0])
            + np.dot(bar_demand, v[1])
            + bar_cap * v[2][0]
            + (bar_charge + bar_discharge) * v[3][0]
        )
        assert lhs == pytest.approx(rhs, rel=1e-10, abs=1e-10)


@pytest.mark.unit
def test_rolling_matches_full_horizon_when_window_is_full():
    full = _build_problem()
    full.run_model()
    rolling = _build_problem(
        dispatch_horizon_mode="rolling", n_horizon_window=N_STEPS, n_control_window=N_STEPS
    )
    rolling.run_model()
    assert rolling.get_val("electricity_command_value") == pytest.approx(
        full.get_val("electricity_command_value"), abs=1e-8
    )


@pytest.mark.unit
@pytest.mark.skipif(
    not cached_ipopt_available(), reason="the Ipopt shared library is not available"
)
def test_one_week_dispatch_directional_derivatives():
    problem = _build_problem(n=168, solver_backend="cached")
    problem.run_model()
    rng = np.random.default_rng(42)
    _, base = _input_vector(problem)
    direction = [rng.standard_normal(value.shape) for value in base]
    np.testing.assert_allclose(
        _jacvec(problem, direction),
        _directional_fd(problem, direction),
        rtol=2e-3,
        atol=1e-4,
    )
    problem.cleanup()


@pytest.mark.unit
@pytest.mark.skipif(
    not cached_ipopt_available(), reason="the Ipopt shared library is not available"
)
@pytest.mark.parametrize("horizon_mode", ["full_horizon", "rolling"])
@pytest.mark.parametrize("warm_start", ["no", "yes"])
def test_cached_solver_matches_shell_with_varying_penalties(horizon_mode, warm_start):
    settings = {
        "dispatch_horizon_mode": horizon_mode,
        "n_horizon_window": 8,
        "n_control_window": 4,
        "unmet_demand_penalty": np.linspace(20.0, 100.0, N_STEPS).tolist(),
    }
    shell = _build_problem(solver_backend="shell", **settings)
    cached = _build_problem(
        solver_backend="cached",
        solver_options={"warm_start_init_point": warm_start},
        **settings,
    )
    for rate in (10.0, 9.0):
        for problem in (shell, cached):
            problem.set_val("max_charge_rate", rate, units="MW")
            problem.run_model()
        assert _output_vector(cached) == pytest.approx(_output_vector(shell), abs=1e-6)
        direction = [np.ones_like(value) for value in _input_vector(shell)[1]]
        assert _jacvec(cached, direction) == pytest.approx(
            _jacvec(shell, direction), rel=1e-5, abs=1e-6
        )


@pytest.mark.unit
def test_dispatch_is_physically_consistent(subtests):
    prob = _build_problem()
    prob.run_model()
    generation, demand = _profiles()
    charge = prob.get_val("dispatch_electricity_charge")
    discharge = prob.get_val("dispatch_electricity_discharge")
    stored = prob.get_val("dispatch_electricity_stored")
    unmet = prob.get_val("dispatch_electricity_unmet_demand")
    curtailed = prob.get_val("dispatch_electricity_curtailed")
    delivered = prob.get_val("dispatch_electricity_delivered")

    with subtests.test("supply balance"):
        assert generation - curtailed - charge + discharge == pytest.approx(delivered, abs=1e-5)
    with subtests.test("demand balance"):
        assert delivered + unmet == pytest.approx(demand, abs=1e-5)
    with subtests.test("storage limits"):
        assert np.all(stored >= 0.1 * 40.0 - 1e-4)
        assert np.all(stored <= 0.9 * 40.0 + 1e-4)
        assert np.all(charge <= 10.0 + 1e-6)
        assert np.all(discharge <= 10.0 + 1e-6)
    with subtests.test("storage balance"):
        previous = np.concatenate([[0.5 * 40.0], stored[:-1]])
        assert stored == pytest.approx(previous + 0.95 * charge - discharge / 0.95, abs=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize(
    "barrier_mu,tolerance",
    [(0.0, 1e-4), (1e-6, 2e-3)],
    ids=["unsmoothed", "default_smoothing"],
)
def test_commands_are_reproduced_by_storage_performance_model(barrier_mu, tolerance):
    params = _storage_parameters(smoothing_barrier_mu=barrier_mu)
    tech_config = {"model_inputs": {"shared_parameters": params}}
    generation, demand = _profiles()

    prob = om.Problem()
    ivc = prob.model.add_subsystem("ivc", om.IndepVarComp(), promotes=["*"])
    ivc.add_output("electricity_in", generation, units="MW")
    ivc.add_output("electricity_set_point", demand, units="MW")
    prob.model.add_subsystem(
        "controller",
        DifferentiableDispatchStorageController(
            plant_config=_plant_config(), tech_config=tech_config
        ),
        promotes=["*"],
    )
    prob.model.add_subsystem(
        "storage",
        StoragePerformanceModel(plant_config=_plant_config(), tech_config=tech_config),
        promotes=["*"],
    )
    prob.setup()
    prob.run_model()

    # With smoothing, the barrier allows small simultaneous charge and discharge flows that the
    # storage model sees only as a net command, so the trajectories differ slightly.
    command = prob.get_val("electricity_command_value", units="MW")
    storage_out = prob.get_val("electricity_out", units="MW")
    soc = prob.get_val("SOC", units="unitless")
    stored = prob.get_val("dispatch_electricity_stored", units="MW*h")
    assert storage_out == pytest.approx(command, abs=tolerance * 10.0)
    assert soc == pytest.approx(stored / 40.0, abs=tolerance)


@pytest.mark.unit
def test_perfect_foresight_does_not_increase_unmet_demand():
    optimal = _build_problem(smoothing_barrier_mu=0.0)
    optimal.run_model()
    heuristic = _build_problem(controller_cls=DemandOpenLoopStorageController)
    heuristic.run_model()

    generation, demand = _profiles()
    command = heuristic.get_val("electricity_command_value")
    heuristic_unmet = np.maximum(demand - generation - np.maximum(command, 0.0), 0.0).sum()
    optimal_unmet = optimal.get_val("dispatch_electricity_unmet_demand").sum()
    assert optimal_unmet <= heuristic_unmet + 1e-4


@pytest.mark.unit
def test_zero_storage_falls_back_to_direct_delivery(subtests):
    prob = _build_problem(max_capacity=0.0)
    prob.set_val("storage_capacity", 0.0)
    prob.run_model()
    generation, demand = _profiles()

    with subtests.test("outputs"):
        assert prob.get_val("electricity_command_value") == pytest.approx(np.zeros(N_STEPS))
        assert prob.get_val("dispatch_electricity_delivered") == pytest.approx(
            np.minimum(generation, demand)
        )
    with subtests.test("partials with respect to profiles"):
        # Zero capacity is a nonsmooth boundary, so only the profile derivatives are defined.
        data = prob.check_partials(out_stream=None, method="fd", step=1e-6)
        for (of, wrt), val in data["controller"].items():
            if wrt not in ("electricity_in", "electricity_set_point"):
                continue
            assert np.abs(val["J_fwd"] - val["J_fd"]).max() < 1e-6, (of, wrt)
            assert np.abs(val["J_rev"] - val["J_fd"]).max() < 1e-6, (of, wrt)


@pytest.mark.unit
def test_derivative_options(subtests):
    with subtests.test("fd partials"):
        prob = _build_problem(derivatives="fd")
        assert not prob.model.controller.matrix_free
        prob.run_model()
        totals = prob.compute_totals(
            of=["dispatch_cost"], wrt=["ivc.storage_capacity"], return_format="array"
        )
        assert np.all(np.isfinite(totals))
    with subtests.test("no partials"):
        prob = _build_problem(derivatives="none")
        assert not prob.model.controller.matrix_free
        prob.run_model()
        assert prob.get_val("dispatch_cost")[0] > 0.0


@pytest.mark.unit
def test_config_validation(subtests):
    with subtests.test("control window longer than horizon"):
        with pytest.raises(ValueError, match="n_control_window"):
            DifferentiableDispatchStorageControllerConfig.from_dict(
                _storage_parameters(
                    dispatch_horizon_mode="rolling", n_horizon_window=4, n_control_window=8
                )
            )
    with subtests.test("unknown horizon mode"):
        with pytest.raises(ValueError):
            DifferentiableDispatchStorageControllerConfig.from_dict(
                _storage_parameters(dispatch_horizon_mode="weekly")
            )
