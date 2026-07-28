import numpy as np
import pytest
import openmdao.api as om

from h2integrate.converters.geothermal.geothermal_cost import (
    GeothermalPlantCostModel,
    GeothermalPlantCostModelConfig,
)
from h2integrate.converters.geothermal.geothermal_pysam import PYSAMGeothermalPlantPerformanceModel


@pytest.fixture
def geothermal_cost_params():
    # Representative hydrothermal binary geothermal costs (order-of-magnitude, NREL ATB).
    return {
        "capex_per_kW": 5500.0,
        "opex_per_kW_per_year": 130.0,
        "cost_year": 2022,
    }


@pytest.mark.unit
class TestGeothermalCostConfig:
    def test_valid_config(self, geothermal_cost_params):
        config = GeothermalPlantCostModelConfig.from_dict(geothermal_cost_params)
        assert config.capex_per_kW == 5500.0
        assert config.opex_per_kW_per_year == 130.0
        assert config.cost_year == 2022

    def test_negative_capex_rejected(self, geothermal_cost_params):
        params = dict(geothermal_cost_params)
        params["capex_per_kW"] = -1.0
        with pytest.raises(ValueError):
            GeothermalPlantCostModelConfig.from_dict(params)


@pytest.mark.unit
def test_geothermal_cost_scaling(geothermal_cost_params, plant_config, subtests):
    tech_config_dict = {
        "model_inputs": {
            "cost_parameters": geothermal_cost_params,
        }
    }
    cost_comp = GeothermalPlantCostModel(
        plant_config=plant_config,
        tech_config=tech_config_dict,
        driver_config={},
    )

    prob = om.Problem()
    prob.model.add_subsystem("cost", cost_comp, promotes=["*"])
    prob.setup()

    capacity = 25000.0
    prob.set_val("cost.system_capacity_AC", capacity, units="kW")
    prob.run_model()

    with subtests.test("CapEx"):
        assert prob.get_val("cost.CapEx", units="USD")[0] == pytest.approx(
            geothermal_cost_params["capex_per_kW"] * capacity
        )

    with subtests.test("OpEx"):
        assert prob.get_val("cost.OpEx", units="USD/year")[0] == pytest.approx(
            geothermal_cost_params["opex_per_kW_per_year"] * capacity
        )

    with subtests.test("cost_year"):
        assert prob.get_val("cost.cost_year") == geothermal_cost_params["cost_year"]


@pytest.mark.unit
def test_geothermal_performance_and_cost(
    geothermal_cost_params, plant_config, solar_resource_data, subtests
):
    performance_params = {
        "nameplate_kW": 30000.0,
        "resource_temp_C": 200.0,
        "resource_depth_m": 2000.0,
        "create_model_from": "default",
    }
    tech_config_dict = {
        "model_inputs": {
            "performance_parameters": performance_params,
            "cost_parameters": geothermal_cost_params,
        }
    }

    prob = om.Problem()
    perf_comp = PYSAMGeothermalPlantPerformanceModel(
        plant_config=plant_config,
        tech_config=tech_config_dict,
        driver_config={},
    )
    cost_comp = GeothermalPlantCostModel(
        plant_config=plant_config,
        tech_config=tech_config_dict,
        driver_config={},
    )
    prob.model.add_subsystem("geo_perf", perf_comp, promotes=["*"])
    prob.model.add_subsystem("geo_cost", cost_comp, promotes=["*"])
    prob.setup()
    prob.set_val("solar_resource_data", solar_resource_data)
    prob.run_model()

    capacity = prob.get_val("geo_cost.system_capacity_AC", units="kW")[0]

    with subtests.test("capacity flows from performance to cost"):
        assert capacity > 0

    with subtests.test("CapEx > 0"):
        assert prob.get_val("geo_cost.CapEx", units="USD")[0] == pytest.approx(
            geothermal_cost_params["capex_per_kW"] * capacity
        )

    with subtests.test("OpEx > 0"):
        assert np.all(prob.get_val("geo_cost.OpEx", units="USD/year") > 0)
