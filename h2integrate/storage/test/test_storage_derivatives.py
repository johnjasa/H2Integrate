import pytest
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_totals

from h2integrate.storage.storage_performance_model import StoragePerformanceModel


@pytest.mark.unit
@pytest.mark.parametrize("separate_rates", [False, True])
@pytest.mark.parametrize("initial_soc", [0.5, 0.85])
@pytest.mark.parametrize("dt", [1800, 3600])
def test_storage_forward_totals(separate_rates, initial_soc, dt):
    plant_config = {"plant": {"plant_life": 2, "simulation": {"n_timesteps": 4, "dt": dt}}}
    shared = {
        "commodity": "electricity",
        "commodity_rate_units": "kW",
        "max_capacity": 10.0,
        "max_charge_rate": 2.0,
        "init_soc_fraction": initial_soc,
        "min_soc_fraction": 0.1,
        "max_soc_fraction": 0.9,
        "charge_efficiency": 0.9,
        "discharge_efficiency": 0.8,
        "charge_equals_discharge": not separate_rates,
        "demand_profile": 1.0,
    }
    if separate_rates:
        shared["max_discharge_rate"] = 1.5
    driver = {"driver": {"optimization": {"flag": True, "total_derivatives": "components"}}}
    prob = om.Problem(reports=False)
    prob.model.add_subsystem(
        "storage",
        StoragePerformanceModel(
            plant_config=plant_config,
            tech_config={"model_inputs": {"shared_parameters": shared}},
            driver_config=driver,
        ),
    )
    prob.setup(mode="fwd")
    prob.set_val("storage.electricity_in", [1.0, 3.0, 4.0, 2.0])
    prob.set_val("storage.electricity_command_value", [-1.4, 2.5, -0.5, 0.6])
    prob.run_model()
    outputs = [
        "storage.electricity_out",
        "storage.SOC",
        "storage.total_electricity_produced",
        "storage.annual_electricity_produced",
        "storage.capacity_factor",
        "storage.standard_capacity_factor",
        "storage.storage_duration",
        "storage.electricity_headroom",
        "storage.electricity_charge_demand",
    ]
    inputs = [
        "storage.electricity_in",
        "storage.electricity_command_value",
        "storage.storage_capacity",
        "storage.max_charge_rate",
    ]
    if separate_rates:
        inputs.append("storage.max_discharge_rate")
    totals = prob.check_totals(
        of=outputs, wrt=inputs, out_stream=None, method="fd", form="central", step=1e-5
    )
    assert_check_totals(totals, atol=1e-6, rtol=1e-6)


@pytest.mark.unit
def test_storage_reverse_mode_reports_unsupported():
    plant_config = {"plant": {"plant_life": 1, "simulation": {"n_timesteps": 1, "dt": 3600}}}
    shared = {
        "commodity": "electricity",
        "commodity_rate_units": "kW",
        "max_capacity": 10.0,
        "max_charge_rate": 2.0,
        "init_soc_fraction": 0.5,
        "min_soc_fraction": 0.1,
        "max_soc_fraction": 0.9,
        "round_trip_efficiency": 0.9,
        "demand_profile": 1.0,
    }
    driver = {"driver": {"optimization": {"flag": True, "total_derivatives": "components"}}}
    prob = om.Problem(reports=False)
    prob.model.add_subsystem(
        "storage",
        StoragePerformanceModel(
            plant_config=plant_config,
            tech_config={"model_inputs": {"shared_parameters": shared}},
            driver_config=driver,
        ),
    )
    prob.setup(mode="rev")
    prob.run_model()
    with pytest.raises(om.AnalysisError, match="derivative_mode: fwd"):
        prob.compute_totals(of=["storage.electricity_out"], wrt=["storage.storage_capacity"])
