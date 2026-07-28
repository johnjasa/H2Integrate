import numpy as np
import pytest
import openmdao.api as om

from h2integrate.converters.geothermal.geothermal_pysam import (
    PYSAMGeothermalPlantPerformanceModel,
    PYSAMGeothermalPlantPerformanceModelConfig,
)


@pytest.fixture
def geothermal_performance_params(weather_file):
    return {
        "nameplate_kW": 30000.0,
        "resource_temp_C": 200.0,
        "resource_depth_m": 2000.0,
        "resource_type": 0,
        "conversion_type": 0,
        "analysis_type": 0,
        "create_model_from": "default",
        "config_name": "GeothermalPowerSingleOwner",
        "weather_file": weather_file,
    }


@pytest.mark.unit
class TestGeothermalConfig:
    def test_valid_config(self, geothermal_performance_params):
        config = PYSAMGeothermalPlantPerformanceModelConfig.from_dict(geothermal_performance_params)
        assert config.nameplate_kW == 30000.0
        assert config.resource_temp_C == 200.0
        assert config.resource_type == 0

    def test_missing_weather_file(self, geothermal_performance_params):
        params = dict(geothermal_performance_params)
        params["weather_file"] = "does_not_exist.csv"
        with pytest.raises(FileNotFoundError):
            PYSAMGeothermalPlantPerformanceModelConfig.from_dict(params)

    def test_resource_temp_out_of_range(self, geothermal_performance_params):
        params = dict(geothermal_performance_params)
        params["resource_temp_C"] = 500.0
        with pytest.raises(ValueError):
            PYSAMGeothermalPlantPerformanceModelConfig.from_dict(params)

    def test_new_model_requires_pysam_options(self, geothermal_performance_params):
        params = dict(geothermal_performance_params)
        params["create_model_from"] = "new"
        with pytest.raises(ValueError):
            PYSAMGeothermalPlantPerformanceModelConfig.from_dict(params)

    def test_invalid_pysam_group(self, geothermal_performance_params):
        params = dict(geothermal_performance_params)
        params["pysam_options"] = {"NotAGroup": {"foo": 1}}
        with pytest.raises(ValueError):
            PYSAMGeothermalPlantPerformanceModelConfig.from_dict(params)

    def test_duplicated_managed_param(self, geothermal_performance_params):
        params = dict(geothermal_performance_params)
        params["pysam_options"] = {"GeoHourly": {"resource_temp": 210.0}}
        with pytest.raises(ValueError):
            PYSAMGeothermalPlantPerformanceModelConfig.from_dict(params)


@pytest.mark.unit
def test_geothermal_outputs(geothermal_performance_params, plant_config, subtests):
    tech_config_dict = {
        "model_inputs": {
            "performance_parameters": geothermal_performance_params,
        }
    }

    prob = om.Problem()
    comp = PYSAMGeothermalPlantPerformanceModel(
        plant_config=plant_config,
        tech_config=tech_config_dict,
        driver_config={},
    )
    prob.model.add_subsystem("comp", comp, promotes=["*"])
    prob.setup()
    prob.run_model()

    n_timesteps = int(plant_config["plant"]["simulation"]["n_timesteps"])
    plant_life = int(plant_config["plant"]["plant_life"])

    with subtests.test("electricity_out length"):
        assert len(prob.get_val("comp.electricity_out", units="kW")) == n_timesteps

    with subtests.test("electricity_out is non-negative"):
        assert np.all(prob.get_val("comp.electricity_out", units="kW") >= 0)

    with subtests.test("system_capacity_AC > 0"):
        assert prob.get_val("comp.system_capacity_AC", units="kW")[0] > 0

    with subtests.test("rated_electricity_production > 0"):
        assert prob.get_val("comp.rated_electricity_production", units="kW")[0] > 0

    with subtests.test("total_electricity_produced > 0"):
        assert prob.get_val("comp.total_electricity_produced", units="kW*h")[0] > 0

    with subtests.test("0 <= capacity_factor <= 1"):
        cf = prob.get_val("comp.capacity_factor", units="unitless")
        assert np.all(cf >= 0)
        assert np.all(cf <= 1)

    with subtests.test("capacity_factor length"):
        assert len(prob.get_val("comp.capacity_factor", units="unitless")) == plant_life

    with subtests.test("geothermal is high capacity factor baseload"):
        # Geothermal is a baseload resource and should have a high capacity factor.
        assert np.all(prob.get_val("comp.capacity_factor", units="unitless") > 0.5)

    with subtests.test("annual_electricity_produced > 0"):
        assert np.all(prob.get_val("comp.annual_electricity_produced", units="kW*h/year") > 0)
