import copy
from types import SimpleNamespace

import numpy as np
import pytest
import openmdao.api as om

from h2integrate import H2I_LIBRARY_DIR, load_yaml
from h2integrate.converters.wind import floris
from h2integrate.core.model_baseclass import CacheBaseClass
from h2integrate.converters.wind.floris import FlorisWindPlantPerformanceModel


@pytest.fixture
def wind_config(temp_dir):
    return {
        "num_turbines": 2,
        "floris_wake_config": load_yaml(H2I_LIBRARY_DIR / "floris_v4_default_template.yaml"),
        "floris_turbine_config": load_yaml(H2I_LIBRARY_DIR / "floris_turbine_Vestas_660kW.yaml"),
        "default_turbulence_intensity": 0.06,
        "hub_height": 65.0,
        "layout": {
            "layout_mode": "basicgrid",
            "layout_options": {"row_D_spacing": 5.0, "turbine_D_spacing": 5.0},
        },
        "operational_losses": 12.83,
        "enable_caching": True,
        "cache_dir": temp_dir,
    }


@pytest.mark.unit
def test_floris_cache_key_full_content(temp_dir):
    model = FlorisWindPlantPerformanceModel()
    model.config = SimpleNamespace(cache_dir=temp_dir)
    config = {"array": np.arange(10000.0), "options": {"loss": 1.0}}
    inputs = {"hub_height": np.array([65.0]), "command": np.ones(10000)}
    discrete = {"wind_resource_data": {"wind_speed_100m": np.arange(10000.0)}}

    def key():
        return model.make_cache_hash_filename(config, inputs, discrete)

    with np.printoptions(threshold=10):
        baseline = key()
        arrays = (
            config["array"],
            inputs["command"],
            discrete["wind_resource_data"]["wind_speed_100m"],
        )
        for array in arrays:
            array[5000] += 1
            assert key() != baseline
            array[5000] -= 1
            assert key() == baseline
        inputs["hub_height"][0] += 1e-10
        assert key() != baseline
        inputs["hub_height"][0] = 65.0
        config["options"]["loss"] += 1
        assert key() != baseline
        config["options"]["loss"] -= 1
        discrete["wind_resource_data"]["wind_speed_100m"] = np.arange(10000, dtype=np.int64)
        assert key() != baseline
        discrete["wind_resource_data"]["wind_speed_100m"] = np.arange(10000.0).reshape(100, 100)
        assert key() != baseline
        discrete["wind_resource_data"]["wind_speed_100m"] = np.arange(10000.0)
        discrete["wind_resource_data"]["other"] = "new field"
        assert key() != baseline
        del discrete["wind_resource_data"]["other"]
    with np.printoptions(threshold=100000):
        discrete["wind_resource_data"]["wind_speed_100m"] = np.arange(10000.0)
        assert key() == baseline


@pytest.mark.unit
def test_floris_cache_key_noncontiguous_and_object_arrays(temp_dir):
    model = FlorisWindPlantPerformanceModel()
    model.config = SimpleNamespace(cache_dir=temp_dir)
    values = np.arange(200.0)[::2]
    objects = np.array([{"items": [1, 2]}, {"items": [3, 4]}], dtype=object)
    discrete = {"values": values, "objects": objects}
    baseline = model.make_cache_hash_filename({}, {}, discrete)
    values[50] += 0.1
    assert model.make_cache_hash_filename({}, {}, discrete) != baseline
    values[50] -= 0.1
    assert model.make_cache_hash_filename({}, {}, discrete) == baseline
    objects[1]["items"][0] += 1
    assert model.make_cache_hash_filename({}, {}, discrete) != baseline


def make_problem(config, n=120, life=3, control=False):
    plant = {
        "plant": {
            "plant_life": life,
            "simulation": {"n_timesteps": n, "dt": 3600, "start_time": "01/01 00:30:00"},
        }
    }
    if control:
        plant["system_level_control"] = {}
    problem = om.Problem(reports=False)
    problem.model.add_subsystem(
        "wind",
        FlorisWindPlantPerformanceModel(
            plant_config=plant,
            tech_config={"model_inputs": {"performance_parameters": copy.deepcopy(config)}},
            driver_config={},
        ),
    )
    problem.setup()
    return problem


@pytest.mark.regression
def test_floris_cache_invalidation_and_results(wind_config, monkeypatch):
    n = 120
    resource = {
        "wind_speed_10m": np.full(n, 6.0),
        "wind_speed_100m": np.full(n, 9.0),
        "wind_direction_10m": np.full(n, 270.0),
        "wind_direction_100m": np.full(n, 270.0),
    }
    original_model = floris.FlorisModel
    calls = []

    def counting_model(*args, **kwargs):
        calls.append(1)
        return original_model(*args, **kwargs)

    monkeypatch.setattr(floris, "FlorisModel", counting_model)
    problem = make_problem(wind_config, n=n)
    problem.set_val("wind.wind_resource_data", resource)
    with np.printoptions(threshold=10):
        problem.run_model()
        initial = problem.get_val("wind.electricity_out").copy()
        problem.run_model()
        assert len(calls) == 1
        np.testing.assert_array_equal(problem.get_val("wind.electricity_out"), initial)

        resource["wind_speed_100m"][n // 2] += 4.0
        problem.run_model()
        modified = problem.get_val("wind.electricity_out").copy()
        assert len(calls) == 2
        assert modified[n // 2] != initial[n // 2]

        uncached_config = copy.deepcopy(wind_config)
        uncached_config["enable_caching"] = False
        uncached = make_problem(uncached_config, n=n)
        uncached.set_val("wind.wind_resource_data", resource)
        uncached.run_model()
        for name in (
            "electricity_out",
            "uncurtailed_electricity_out",
            "total_electricity_produced",
            "annual_electricity_produced",
            "rated_electricity_production",
            "capacity_factor",
            "replacement_schedule",
            "operational_life",
        ):
            np.testing.assert_allclose(
                problem.get_val(f"wind.{name}"), uncached.get_val(f"wind.{name}")
            )

        reloaded = make_problem(wind_config, n=n)
        reloaded.set_val("wind.wind_resource_data", resource)
        reloaded.run_model()
        assert len(calls) == 3
        np.testing.assert_array_equal(reloaded.get_val("wind.electricity_out"), modified)


@pytest.mark.regression
def test_floris_cache_runtime_and_configuration(wind_config, monkeypatch):
    n = 120
    resource = {
        "wind_speed_65m": np.full(n, 9.0),
        "wind_direction_65m": np.full(n, 270.0),
    }
    original_model = floris.FlorisModel
    calls = []

    def counting_model(*args, **kwargs):
        calls.append(1)
        return original_model(*args, **kwargs)

    monkeypatch.setattr(floris, "FlorisModel", counting_model)

    problem = make_problem(wind_config, n=n)
    problem.set_val("wind.wind_resource_data", resource)
    problem.final_setup()
    component = problem.model.wind
    legacy_config = component.config.as_dict()
    legacy_config["wind_turbine_size_kw"] = component.wind_turbine_rating_kW
    legacy = CacheBaseClass.make_cache_hash_filename(
        component, legacy_config, component._inputs, component._discrete_inputs
    )
    legacy.write_bytes(b"legacy")
    problem.run_model()
    initial = problem.get_val("wind.electricity_out").copy()
    assert len(calls) == 1

    # Runtime plant state affects cached output shape and curtailment independently.
    different_life = make_problem(wind_config, n=n, life=4)
    different_life.set_val("wind.wind_resource_data", resource)
    different_life.run_model()
    assert len(calls) == 2
    assert len(different_life.get_val("wind.capacity_factor")) == 4
    controlled = make_problem(wind_config, n=n, control=True)
    controlled.set_val("wind.wind_resource_data", resource)
    controlled.set_val("wind.electricity_command_value", np.full(n, 100.0))
    with np.printoptions(threshold=10):
        controlled.run_model()
        assert len(calls) == 3
        np.testing.assert_allclose(
            controlled.get_val("wind.electricity_out"), np.minimum(initial, 100)
        )
        command = np.full(n, 100.0)
        command[n // 2] = 50.0
        controlled.set_val("wind.electricity_command_value", command)
        controlled.run_model()
        assert len(calls) == 4
        assert controlled.get_val("wind.electricity_out")[n // 2] == pytest.approx(50.0)

    # Config and standard continuous inputs must still invalidate the cache.
    problem.set_val("wind.hub_height", 66.0)
    problem.run_model()
    assert len(calls) == 5
    problem.model.wind.config.operational_losses = 25.0
    problem.run_model()
    assert len(calls) == 6
    total = problem.get_val("wind.total_electricity_produced").copy()
    problem.model.wind.dt = 1800
    problem.run_model()
    assert len(calls) == 7
    np.testing.assert_allclose(problem.get_val("wind.total_electricity_produced"), total / 2)


@pytest.mark.unit
def test_disabled_floris_cache_does_not_hash(wind_config, monkeypatch):
    wind_config["enable_caching"] = False
    problem = make_problem(wind_config, n=4)
    problem.set_val(
        "wind.wind_resource_data",
        {"wind_speed_65m": np.full(4, 9.0), "wind_direction_65m": np.full(4, 270.0)},
    )

    def fail_hash(*args, **kwargs):
        raise AssertionError("disabled cache must not hash")

    monkeypatch.setattr(problem.model.wind, "make_cache_hash_filename", fail_hash)
    problem.run_model()
