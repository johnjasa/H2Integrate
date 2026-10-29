import pytest
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from h2integrate.converters.hydrogen.custom_electrolyzer_cost_model import (
    CustomElectrolyzerCostModel,
)


@pytest.mark.unit
def test_custom_electrolyzer_cost_model(subtests):
    capex_usd_per_kw = 10.0
    opex_usd_per_kw = 5.0

    tech_config_dict = {
        "model_inputs": {
            "cost_parameters": {
                "capex_USD_per_kW": capex_usd_per_kw,  # dummy number
                "fixed_om_USD_per_kW_per_year": opex_usd_per_kw,  # dummy number
                "cost_year": 2022,
            },
        }
    }

    plant_config = {
        "plant": {
            "plant_life": 30,
            "simulation": {
                "n_timesteps": 8760,  # Default number of timesteps for the simulation
                "dt": 3600,  # Default time step length in seconds (1 hour)
            },
        },
    }

    rated_power_kW = 1000.0
    prob = om.Problem()
    comp = CustomElectrolyzerCostModel(
        plant_config=plant_config,
        tech_config=tech_config_dict,
    )

    prob.model.add_subsystem("custom_elec_cost", comp)
    prob.setup()
    prob.set_val("custom_elec_cost.electrolyzer_size_mw", rated_power_kW, units="kW")

    prob.run_model()

    expected_outputs = {
        "CapEx": [rated_power_kW * capex_usd_per_kw],
        "OpEx": [rated_power_kW * opex_usd_per_kw],
    }

    for out, expected in expected_outputs.items():
        with subtests.test(out):
            units = "USD" if out == "CapEx" else "USD/year"
            val = prob.get_val(f"custom_elec_cost.{out}", units=units)
            assert pytest.approx(val, rel=1e-6) == expected[0]

    with subtests.test("analytic partials"):
        assert_check_partials(
            prob.check_partials(out_stream=None, method="fd", form="central", step=1e-4),
            atol=1e-6,
            rtol=1e-6,
        )
