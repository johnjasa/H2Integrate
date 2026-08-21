import numpy as np
import pytest
import openmdao.api as om
from pytest import fixture

from h2integrate.converters.hydrogen.simple_electrolyzer import SimpleElectrolyzerPerformanceModel


N_TIMESTEPS = 8760
PLANT_LIFE = 30


@fixture
def plant_config():
    return {
        "plant": {
            "plant_life": PLANT_LIFE,
            "simulation": {
                "n_timesteps": N_TIMESTEPS,
                "dt": 3600,
            },
        },
    }


@fixture
def performance_params():
    return {
        "rating_MW": 50.0,
        "efficiency_kWh_per_kg": 55.0,
        "turndown_ratio": 0.1,
    }


def build_problem(plant_config, performance_params):
    comp = SimpleElectrolyzerPerformanceModel(
        plant_config=plant_config,
        tech_config={"model_inputs": {"performance_parameters": performance_params}},
        driver_config={},
    )
    prob = om.Problem()
    prob.model.add_subsystem("electrolyzer", comp, promotes=["*"])
    prob.setup()
    return prob


@pytest.mark.unit
def test_converts_electricity_at_the_given_efficiency(plant_config, performance_params, subtests):
    rating_kW = performance_params["rating_MW"] * 1e3
    specific_energy = performance_params["efficiency_kWh_per_kg"]

    prob = build_problem(plant_config, performance_params)
    prob.set_val("electricity_in", np.full(N_TIMESTEPS, rating_kW), units="kW")
    prob.run_model()

    rated_production = rating_kW / specific_energy

    with subtests.test("hydrogen out"):
        assert pytest.approx(prob.get_val("hydrogen_out", units="kg/h")) == np.full(
            N_TIMESTEPS, rated_production
        )

    with subtests.test("rated production"):
        assert (
            pytest.approx(prob.get_val("rated_hydrogen_production", units="kg/h")[0])
            == rated_production
        )

    with subtests.test("total produced"):
        assert (
            pytest.approx(prob.get_val("total_hydrogen_produced", units="kg")[0])
            == rated_production * N_TIMESTEPS
        )

    with subtests.test("capacity factor"):
        assert pytest.approx(prob.get_val("capacity_factor")) == np.ones(PLANT_LIFE)

    with subtests.test("electrolyzer size"):
        assert (
            pytest.approx(prob.get_val("electrolyzer_size_mw", units="MW")[0])
            == performance_params["rating_MW"]
        )


@pytest.mark.unit
def test_consumption_is_capped_at_the_rating(plant_config, performance_params, subtests):
    rating_kW = performance_params["rating_MW"] * 1e3
    specific_energy = performance_params["efficiency_kWh_per_kg"]

    prob = build_problem(plant_config, performance_params)
    prob.set_val("electricity_in", np.full(N_TIMESTEPS, 3.0 * rating_kW), units="kW")
    prob.run_model()

    with subtests.test("electricity consumed"):
        assert pytest.approx(prob.get_val("electricity_consumed", units="kW")) == np.full(
            N_TIMESTEPS, rating_kW
        )

    with subtests.test("hydrogen out"):
        assert pytest.approx(prob.get_val("hydrogen_out", units="kg/h")) == np.full(
            N_TIMESTEPS, rating_kW / specific_energy
        )


@pytest.mark.unit
def test_turndown_shuts_the_electrolyzer_off(plant_config, performance_params, subtests):
    rating_kW = performance_params["rating_MW"] * 1e3
    turndown = performance_params["turndown_ratio"]

    electricity_in = np.full(N_TIMESTEPS, 0.5 * turndown * rating_kW)
    electricity_in[:10] = turndown * rating_kW

    prob = build_problem(plant_config, performance_params)
    prob.set_val("electricity_in", electricity_in, units="kW")
    prob.run_model()

    hydrogen_out = prob.get_val("hydrogen_out", units="kg/h")

    with subtests.test("off below turndown"):
        assert np.all(hydrogen_out[10:] == 0.0)

    with subtests.test("on at turndown"):
        assert np.all(hydrogen_out[:10] > 0.0)


@pytest.mark.unit
def test_rejects_unsupported_size_modes(plant_config, performance_params):
    performance_params = performance_params | {
        "size_mode": "resize_by_max_feedstock",
        "flow_used_for_sizing": "electricity",
    }

    prob = build_problem(plant_config, performance_params)
    with pytest.raises(NotImplementedError, match="only supports the 'normal' size mode"):
        prob.run_model()
