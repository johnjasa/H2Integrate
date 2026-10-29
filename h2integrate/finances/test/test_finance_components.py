import numpy as np
import pytest
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from h2integrate.finances.finances import AdjustedCapexOpexComp, AdjustedCapacityFactorComp


@pytest.mark.unit
def test_adjusted_cost_partials():
    plant_config = {
        "plant": {"plant_life": 3},
        "finance_parameters": {
            "cost_adjustment_parameters": {
                "cost_year_adjustment_inflation": 0.025,
                "target_dollar_year": 2022,
            }
        },
    }
    prob = om.Problem(reports=False)
    prob.model.add_subsystem(
        "cost",
        AdjustedCapexOpexComp(
            plant_config=plant_config, tech_configs={"battery": {}, "solar": {}}, driver_config={}
        ),
    )
    prob.setup()
    for tech, year in (("battery", 2020), ("solar", 2024)):
        prob.set_val(f"cost.cost_year_{tech}", year)
        prob.set_val(f"cost.capex_{tech}", 100.0)
        prob.set_val(f"cost.opex_{tech}", 5.0)
        prob.set_val(f"cost.varopex_{tech}", [1.0, 2.0, 3.0])
    prob.run_model()
    assert_check_partials(
        prob.check_partials(out_stream=None, method="fd", form="central", step=1e-4),
        atol=1e-7,
        rtol=1e-7,
    )


@pytest.fixture
def plant_config(n_timesteps):
    plant = {
        "plant": {
            "plant_life": 30,
            "simulation": {
                "dt": 3600,
                "n_timesteps": n_timesteps,
            },
        },
    }
    return plant


@pytest.mark.unit
@pytest.mark.parametrize("n_timesteps", [8760])
def test_adjusted_cf_comp(plant_config, subtests):
    rng = np.random.default_rng(seed=0)
    electricity_input = rng.random(8760) * 1e3

    prob = om.Problem()
    comp = AdjustedCapacityFactorComp(plant_config=plant_config, commodity_type="electricity")
    prob.model.add_subsystem("comp", comp, promotes=["*"])

    ivc = om.IndepVarComp()
    ivc.add_output("electricity_produced", val=np.zeros(8760), units="kW")
    prob.model.add_subsystem("ivc", ivc, promotes=["*"])

    prob.setup()
    prob.set_val("ivc.electricity_produced", val=electricity_input, units="kW")
    prob.run_model()

    with subtests.test("Check rated_electricity_production calc"):
        assert (
            pytest.approx(
                prob.model.get_val("comp.rated_electricity_production", units="kW"), rel=1e-6
            )
            == electricity_input.mean()
        )

    with subtests.test("Check capacity factor"):
        assert np.all(prob.model.get_val("comp.capacity_factor", units="unitless")) == 1.0

    with subtests.test("Capacity factor length"):
        assert len(prob.model.get_val("comp.capacity_factor", units="unitless")) == 30
