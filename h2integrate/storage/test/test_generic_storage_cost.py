import pytest
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from h2integrate.storage.generic_storage_cost import GenericStorageCostModel


@pytest.mark.unit
@pytest.mark.parametrize(
    "rate_units,amount_units",
    [("MW", "MW*h"), ("kg/h", "kg"), ("kg/h", "t")],
)
def test_generic_storage_cost_partials(rate_units, amount_units):
    plant_config = {"plant": {"plant_life": 2, "simulation": {"n_timesteps": 3, "dt": 3600}}}
    tech_config = {
        "model_inputs": {
            "cost_parameters": {
                "commodity_rate_units": rate_units,
                "commodity_amount_units": amount_units,
                "max_capacity": 5.0,
                "max_charge_rate": 2.0,
                "capacity_capex": 3.0,
                "charge_capex": 4.0,
                "opex_fraction": 0.1,
                "cost_year": 2022,
            }
        }
    }
    prob = om.Problem(reports=False)
    prob.model.add_subsystem(
        "cost", GenericStorageCostModel(plant_config=plant_config, tech_config=tech_config)
    )
    prob.setup()
    prob.run_model()
    assert_check_partials(
        prob.check_partials(out_stream=None, method="fd", form="central", step=1e-5),
        atol=1e-6,
        rtol=1e-6,
    )

    prob.set_val("cost.max_charge_rate", 0.0)
    prob.run_model()
    with pytest.raises(ValueError, match="cost is discontinuous at zero"):
        prob.compute_totals(of=["cost.CapEx"], wrt=["cost.max_charge_rate"])
