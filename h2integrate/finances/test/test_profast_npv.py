import copy
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import openmdao.api as om
from pytest import fixture

from h2integrate.finances.profast_lco import ProFastLCO
from h2integrate.finances.profast_npv import ProFastNPV


@fixture
def profast_inputs_no1():
    params = {
        "analysis_start_year": 2032,
        "installation_time": 36,
        "inflation_rate": 0.0,
        "discount_rate": 0.0948,
        "debt_equity_ratio": 1.72,
        "property_tax_and_insurance": 0.015,
        "total_income_tax_rate": 0.2574,
        "capital_gains_tax_rate": 0.15,
        "sales_tax_rate": 0.00,
        "debt_interest_rate": 0.046,
        "debt_type": "Revolving debt",
        "loan_period_if_used": 0,
        "cash_onhand_months": 1,
        "admin_expense": 0.00,
    }
    cap_items = {"depr_type": "MACRS", "depr_period": 5, "refurb": [0.0]}
    model_inputs = {
        "commodity_sell_price": 0.04,  # USD/kWh for electricity
        "commodity_sell_price_units": "USD/(kW*h)",
        "params": params,
        "capital_items": cap_items,
    }

    return model_inputs


@fixture
def profast_inputs_no2():
    params = {
        "analysis_start_year": 2032,
        "installation_time": 36,
        "inflation_rate": 0.0,
        "discount_rate": 0.0615,
        "debt_equity_ratio": 2.82,
        "property_tax_and_insurance": 0.015,
        "total_income_tax_rate": 0.2574,
        "capital_gains_tax_rate": 0.15,
        "sales_tax_rate": 0.00,
        "debt_interest_rate": 0.0439,
        "debt_type": "Revolving debt",
        "loan_period_if_used": 0,
        "cash_onhand_months": 1,
        "admin_expense": 0.00,
    }
    cap_items = {"depr_type": "MACRS", "depr_period": 5, "refurb": [0.0]}

    model_inputs = {
        "commodity_sell_price": 0.07,  # USD/kWh for electricity
        "commodity_sell_price_units": "USD/(kW*h)",
        "params": params,
        "capital_items": cap_items,
    }

    return model_inputs


@fixture
def fake_filtered_tech_config():
    tech_config = {
        "wind": {"model_inputs": {}},
        "solar": {"model_inputs": {}},
        "battery": {"model_inputs": {}},
        "natural_gas": {"model_inputs": {}},
    }
    return tech_config


@fixture
def fake_cost_dict():
    fake_costs = {
        "capex_adjusted_wind": 950054634.1,
        "opex_adjusted_wind": 21093892.68,
        "varopex_adjusted_wind": [0.0] * 30,
        "capex_adjusted_solar": 6561339.6,
        "opex_adjusted_solar": 88372.77,
        "varopex_adjusted_solar": [0.0] * 30,
        "capex_adjusted_battery": 3402926,
        "opex_adjusted_battery": 779.27,
        "varopex_adjusted_battery": [0.0] * 30,
        "capex_adjusted_natural_gas": 1170731708.0,
        "opex_adjusted_natural_gas": 12783853.58,
        "varopex_adjusted_natural_gas": [65458026.9] * 30,
    }
    return fake_costs


@pytest.mark.unit
@pytest.mark.parametrize("finance_model", [ProFastNPV, ProFastLCO])
@pytest.mark.parametrize("credit_rate", [0.0, 0.4, 1.0])
@pytest.mark.parametrize("battery_name", ["battery", "storage_bank"])
def test_battery_investment_tax_credit(
    finance_model,
    credit_rate,
    battery_name,
    profast_inputs_no1,
    fake_filtered_tech_config,
    subtests,
):
    profast_inputs_no1["params"]["one_time_cap_inct"] = {
        "value": 500.0,
        "depr_type": "MACRS",
        "depr_period": 5,
        "depreciable": False,
    }
    fake_filtered_tech_config.pop("battery")
    capital_items = {
        "investment_tax_credit": credit_rate,
        "replacement_cost_percent": 1.0,
    }
    fake_filtered_tech_config[battery_name] = {
        "model_inputs": {"financial_parameters": {"capital_items": capital_items}}
    }
    plant_config = {
        "plant": {"plant_life": 30},
        "finance_parameters": {"model_inputs": profast_inputs_no1},
    }
    prob = om.Problem()
    component = finance_model(
        driver_config={},
        plant_config=plant_config,
        tech_config=fake_filtered_tech_config,
        commodity_type="electricity",
    )
    ivc = om.IndepVarComp()
    ivc.add_output("rated_electricity_production", 1000.0, units="kW")
    ivc.add_output("capacity_factor", np.ones(30), units="unitless")
    prob.model.add_subsystem("ivc", ivc, promotes=["*"])
    prob.model.add_subsystem("pf", component, promotes=["*"])
    prob.setup()
    component.price_units = "USD/(kW*h)"
    component.commodity_amount_units = "kW*h"
    inputs = {
        "rated_electricity_production": np.array([1000.0]),
        "capacity_factor": np.ones(30),
    }
    for technology in fake_filtered_tech_config:
        inputs[f"capex_adjusted_{technology}"] = np.array([1.0e6])
        inputs[f"opex_adjusted_{technology}"] = np.array([0.0])
        inputs[f"varopex_adjusted_{technology}"] = np.zeros(30)
        inputs[f"replacement_schedule_{technology}"] = np.ones(30)

    with patch("h2integrate.finances.profast_base.create_and_populate_profast") as populate:
        component.populate_profast(inputs)
        configuration = populate.call_args.args[0]
        with subtests.test("Credit applies only to battery initial CapEx"):
            assert configuration["params"]["one time cap inct"]["value"] == (
                500.0 + credit_rate * 1.0e6
            )
            assert configuration["capital_items"][battery_name]["cost"] == 1.0e6
            assert configuration["capital_items"][battery_name]["refurb"] == [1.0] * 30
            assert "investment_tax_credit" not in configuration["capital_items"][battery_name]

        inputs[f"capex_adjusted_{battery_name}"] = np.array([2.0e6])
        component.populate_profast(inputs)
        with subtests.test("Credit is recalculated without accumulating"):
            assert populate.call_args.args[0]["params"]["one time cap inct"]["value"] == (
                500.0 + credit_rate * 2.0e6
            )
            assert component.params.one_time_cap_inct["value"] == 500.0
            assert capital_items["investment_tax_credit"] == credit_rate
            assert "cost" not in capital_items

        with subtests.test("Credits from multiple technologies are summed"):
            fake_filtered_tech_config["solar"]["model_inputs"] = {
                "financial_parameters": {"capital_items": {"investment_tax_credit": 0.3}}
            }
            component.populate_profast(inputs)
            assert populate.call_args.args[0]["params"]["one time cap inct"]["value"] == (
                500.0 + credit_rate * 2.0e6 + 0.3 * 1.0e6
            )

        for invalid_rate in [-0.1, 1.1, 40, np.nan, np.inf, "0.4", None]:
            with subtests.test("Invalid credit rate", rate=invalid_rate):
                capital_items["investment_tax_credit"] = invalid_rate
                with pytest.raises(ValueError, match="must be a fraction between 0 and 1"):
                    component.populate_profast(inputs)
    prob.cleanup()


@pytest.mark.regression
@pytest.mark.parametrize("finance_model", [ProFastNPV, ProFastLCO])
def test_battery_itc_financial_results(
    finance_model, profast_inputs_no1, fake_filtered_tech_config, fake_cost_dict, subtests
):
    battery_credit = 0.4 * fake_cost_dict["capex_adjusted_battery"]
    prob = om.Problem()
    ivc = om.IndepVarComp()
    ivc.add_output("rated_electricity_production", 500000.0, units="kW")
    ivc.add_output("capacity_factor", np.ones(30), units="unitless")
    prob.model.add_subsystem("ivc", ivc, promotes=["*"])
    for name, credit_rate, incentive_value in [
        ("baseline", 0.0, 0.0),
        ("battery_itc", 0.4, 0.0),
        ("explicit_credit", 0.0, battery_credit),
    ]:
        model_inputs = copy.deepcopy(profast_inputs_no1)
        model_inputs["params"]["one_time_cap_inct"] = {
            "value": incentive_value,
            "depr_type": "MACRS",
            "depr_period": 5,
            "depreciable": False,
        }
        technology_config = copy.deepcopy(fake_filtered_tech_config)
        technology_config["battery"]["model_inputs"] = {
            "financial_parameters": {"capital_items": {"investment_tax_credit": credit_rate}}
        }
        component = finance_model(
            driver_config={},
            plant_config={
                "plant": {"plant_life": 30},
                "finance_parameters": {"model_inputs": model_inputs},
            },
            tech_config=technology_config,
            commodity_type="electricity",
        )
        prob.model.add_subsystem(
            name, component, promotes_inputs=["rated_electricity_production", "capacity_factor"]
        )
    prob.setup()
    for name in ["baseline", "battery_itc", "explicit_credit"]:
        for variable, cost in fake_cost_dict.items():
            prob.set_val(f"{name}.{variable}", cost)
    prob.run_model()

    output_name = "NPV_electricity" if finance_model is ProFastNPV else "LCOE"
    baseline = prob.get_val(f"baseline.{output_name}")[0]
    with_credit = prob.get_val(f"battery_itc.{output_name}")[0]
    explicit_credit = prob.get_val(f"explicit_credit.{output_name}")[0]
    with subtests.test("Credit matches an equivalent ProFAST dollar incentive"):
        assert with_credit == pytest.approx(explicit_credit)
    with subtests.test("Credit improves the financial metric"):
        if finance_model is ProFastNPV:
            assert with_credit > baseline
        else:
            assert with_credit < baseline
            breakdown = prob.get_val("battery_itc.LCOE_breakdown")
            assert breakdown["LCOE: Total ($/kW*h)"] == pytest.approx(with_credit)
    prob.cleanup()


@pytest.mark.regression
def test_profast_npv_no1(profast_inputs_no1, fake_filtered_tech_config, fake_cost_dict, subtests):
    mean_hourly_production = 500000.0
    prob = om.Problem()
    plant_config = {
        "plant": {
            "plant_life": 30,
        },
        "finance_parameters": {"model_inputs": profast_inputs_no1},
    }
    pf = ProFastNPV(
        driver_config={},
        plant_config=plant_config,
        tech_config=fake_filtered_tech_config,
        commodity_type="electricity",
        description="no1",
    )
    ivc = om.IndepVarComp()

    ivc.add_output("rated_electricity_production", mean_hourly_production, units="kW")
    ivc.add_output("capacity_factor", [1.0] * plant_config["plant"]["plant_life"], units="unitless")

    prob.model.add_subsystem("ivc", ivc, promotes=["*"])
    prob.model.add_subsystem("pf", pf, promotes=["rated_electricity_production", "capacity_factor"])
    prob.setup()
    for variable, cost in fake_cost_dict.items():
        units = "USD" if "capex" in variable else "USD/year"
        prob.set_val(f"pf.{variable}", cost, units=units)

    prob.run_model()

    with subtests.test("Sell price"):
        assert (
            pytest.approx(
                prob.get_val("pf.sell_price_electricity_no1", units="USD/(kW*h)"), rel=1e-6
            )
            == profast_inputs_no1["commodity_sell_price"]
        )

    with subtests.test("NPV"):
        assert (
            pytest.approx(prob.get_val("pf.NPV_electricity_no1", units="USD")[0], rel=1e-6)
            == -580179388.883
        )


@pytest.mark.regression
def test_profast_npv_no1_change_sell_price(
    profast_inputs_no1, fake_filtered_tech_config, fake_cost_dict, subtests
):
    mean_hourly_production = 500000.0
    prob = om.Problem()
    plant_config = {
        "plant": {
            "plant_life": 30,
        },
        "finance_parameters": {"model_inputs": profast_inputs_no1},
    }
    pf = ProFastNPV(
        driver_config={},
        plant_config=plant_config,
        tech_config=fake_filtered_tech_config,
        commodity_type="electricity",
        description="no1",
    )

    pf2 = ProFastNPV(
        driver_config={},
        plant_config=plant_config,
        tech_config=fake_filtered_tech_config,
        commodity_type="electricity",
        description="no1_expensive",
    )

    ivc = om.IndepVarComp()

    ivc.add_output("rated_electricity_production", mean_hourly_production, units="kW")
    ivc.add_output("capacity_factor", [1.0] * plant_config["plant"]["plant_life"], units="unitless")

    prob.model.add_subsystem("ivc", ivc, promotes=["*"])
    prob.model.add_subsystem("pf", pf, promotes=["rated_electricity_production", "capacity_factor"])
    prob.model.add_subsystem(
        "pf2", pf2, promotes=["rated_electricity_production", "capacity_factor"]
    )
    prob.setup()
    # set inputs for 'pf' with commodity sell price of 0.04 USD/(kW*h)
    for variable, cost in fake_cost_dict.items():
        units = "USD" if "capex" in variable else "USD/year"
        prob.set_val(f"pf.{variable}", cost, units=units)

    # set inputs for 'pf2' with commodity sell price of 0.07 USD/(kW*h)
    new_sell_price = 0.07
    prob.set_val("pf2.sell_price_electricity_no1_expensive", new_sell_price, units="USD/(kW*h)")

    for variable, cost in fake_cost_dict.items():
        units = "USD" if "capex" in variable else "USD/year"
        prob.set_val(f"pf2.{variable}", cost, units=units)

    prob.run_model()

    with subtests.test("Sell price for pf"):
        assert (
            pytest.approx(
                prob.get_val("pf.sell_price_electricity_no1", units="USD/(kW*h)"), rel=1e-6
            )
            == profast_inputs_no1["commodity_sell_price"]
        )

    with subtests.test("NPV with sell price of 0.04 USD/(kW*h)"):
        assert (
            pytest.approx(prob.get_val("pf.NPV_electricity_no1", units="USD")[0], rel=1e-6)
            == -580179388.883
        )

    with subtests.test("Sell price for pf2"):
        assert (
            pytest.approx(
                prob.get_val("pf2.sell_price_electricity_no1_expensive", units="USD/(kW*h)"),
                rel=1e-6,
            )
            == new_sell_price
        )

    with subtests.test("NPV is higher with higher commodity sell price"):
        assert (
            prob.get_val("pf2.NPV_electricity_no1_expensive", units="USD")
            > prob.get_val("pf.NPV_electricity_no1", units="USD")[0]
        )

    with subtests.test("NPV with sell price of 0.07 USD/(kW*h)"):
        assert (
            pytest.approx(
                prob.get_val("pf2.NPV_electricity_no1_expensive", units="USD")[0], rel=1e-6
            )
            == 150581030.887
        )


@pytest.mark.regression
def test_profast_npv_no2(profast_inputs_no2, fake_filtered_tech_config, fake_cost_dict, subtests):
    mean_hourly_production = 500000.0
    prob = om.Problem()
    plant_config = {
        "plant": {
            "plant_life": 30,
        },
        "finance_parameters": {"model_inputs": profast_inputs_no2},
    }
    pf = ProFastNPV(
        driver_config={},
        plant_config=plant_config,
        tech_config=fake_filtered_tech_config,
        commodity_type="electricity",
        description="no2",
    )

    ivc = om.IndepVarComp()
    ivc.add_output("rated_electricity_production", mean_hourly_production, units="kW")
    ivc.add_output("capacity_factor", [1.0] * plant_config["plant"]["plant_life"], units="unitless")

    prob.model.add_subsystem("ivc", ivc, promotes=["*"])
    prob.model.add_subsystem("pf", pf, promotes=["rated_electricity_production", "capacity_factor"])
    prob.setup()
    for variable, cost in fake_cost_dict.items():
        units = "USD" if "capex" in variable else "USD/year"
        prob.set_val(f"pf.{variable}", cost, units=units)

    prob.run_model()

    with subtests.test("Sell price"):
        assert (
            pytest.approx(
                prob.get_val("pf.sell_price_electricity_no2", units="USD/(kW*h)"), rel=1e-6
            )
            == profast_inputs_no2["commodity_sell_price"]
        )

    with subtests.test("NPV"):
        assert (
            pytest.approx(prob.get_val("pf.NPV_electricity_no2", units="USD")[0], rel=1e-6)
            == 611288384.412
        )


@pytest.mark.regression
def test_profast_npv_nonstandard_price_units(
    profast_inputs_no2, fake_filtered_tech_config, fake_cost_dict, subtests
):
    mean_hourly_production = 500000.0
    prob = om.Problem()

    profast_inputs_no2["commodity_sell_price_units"] = "USD/(kW*min)"

    plant_config = {
        "plant": {
            "plant_life": 30,
        },
        "finance_parameters": {"model_inputs": profast_inputs_no2},
    }
    pf = ProFastNPV(
        driver_config={},
        plant_config=plant_config,
        tech_config=fake_filtered_tech_config,
        commodity_type="electricity",
        description="no2",
    )

    ivc = om.IndepVarComp()
    ivc.add_output("rated_electricity_production", mean_hourly_production, units="kW")
    ivc.add_output("capacity_factor", [1.0] * plant_config["plant"]["plant_life"], units="unitless")

    prob.model.add_subsystem("ivc", ivc, promotes=["*"])
    prob.model.add_subsystem("pf", pf, promotes=["rated_electricity_production", "capacity_factor"])
    prob.setup()
    for variable, cost in fake_cost_dict.items():
        units = "USD" if "capex" in variable else "USD/year"
        prob.set_val(f"pf.{variable}", cost, units=units)

    prob.set_val(
        "pf.sell_price_electricity_no2",
        profast_inputs_no2["commodity_sell_price"],
        units="USD/(kW*h)",
    )
    prob.run_model()

    with subtests.test("Sell price"):
        assert (
            pytest.approx(
                prob.get_val("pf.sell_price_electricity_no2", units="USD/(kW*h)"), rel=1e-6
            )
            == profast_inputs_no2["commodity_sell_price"]
        )

    with subtests.test("NPV"):
        assert (
            pytest.approx(prob.get_val("pf.NPV_electricity_no2", units="USD")[0], rel=1e-6)
            == 611288384.412
        )


@pytest.mark.regression
def test_profast_npv_multi_year_sell_price(
    profast_inputs_no2, fake_filtered_tech_config, fake_cost_dict, subtests
):
    mean_hourly_production = 500000.0
    prob = om.Problem()
    profast_inputs_no2["commodity_sell_price"] = [0.07] * 30
    plant_config = {
        "plant": {
            "plant_life": 30,
        },
        "finance_parameters": {"model_inputs": profast_inputs_no2},
    }
    pf = ProFastNPV(
        driver_config={},
        plant_config=plant_config,
        tech_config=fake_filtered_tech_config,
        commodity_type="electricity",
        description="no2",
    )

    ivc = om.IndepVarComp()
    ivc.add_output("rated_electricity_production", mean_hourly_production, units="kW")
    ivc.add_output("capacity_factor", [1.0] * plant_config["plant"]["plant_life"], units="unitless")

    prob.model.add_subsystem("ivc", ivc, promotes=["*"])
    prob.model.add_subsystem("pf", pf, promotes=["rated_electricity_production", "capacity_factor"])
    prob.setup()
    for variable, cost in fake_cost_dict.items():
        units = "USD" if "capex" in variable else "USD/year"
        prob.set_val(f"pf.{variable}", cost, units=units)

    prob.run_model()

    with subtests.test("Sell price"):
        assert (
            pytest.approx(
                prob.get_val("pf.sell_price_electricity_no2", units="USD/(kW*h)"), rel=1e-6
            )
            == profast_inputs_no2["commodity_sell_price"]
        )

    with subtests.test("NPV"):
        assert (
            pytest.approx(prob.get_val("pf.NPV_electricity_no2", units="USD")[0], rel=1e-6)
            == 611288384.412
        )


@pytest.mark.regression
def test_profast_npv_multi_year_error(profast_inputs_no2, fake_filtered_tech_config, subtests):
    prob = om.Problem()
    profast_inputs_no2["commodity_sell_price"] = [0.07] * 10
    plant_config = {
        "plant": {
            "plant_life": 30,
        },
        "finance_parameters": {"model_inputs": profast_inputs_no2},
    }
    mean_hourly_production = 500000.0
    pf = ProFastNPV(
        driver_config={},
        plant_config=plant_config,
        tech_config=fake_filtered_tech_config,
        commodity_type="electricity",
        description="no2",
    )

    ivc = om.IndepVarComp()
    ivc.add_output("rated_electricity_production", mean_hourly_production, units="kW")
    ivc.add_output("capacity_factor", [1.0] * plant_config["plant"]["plant_life"], units="unitless")

    prob.model.add_subsystem("ivc", ivc, promotes=["*"])
    prob.model.add_subsystem("pf", pf, promotes=["rated_electricity_production", "capacity_factor"])

    expected_message = "`commodity_sell_price` has an invalid length of 10"
    with subtests.test("Incorrect sell price length"):
        with pytest.raises(ValueError) as excinfo:
            prob.setup()
        assert expected_message in str(excinfo.value)


@pytest.mark.regression
def test_profast_npv_missing_sell_price(profast_inputs_no2, fake_filtered_tech_config, subtests):
    prob = om.Problem()
    profast_inputs_no2["commodity_sell_price"] = None
    plant_config = {
        "plant": {
            "plant_life": 30,
        },
        "finance_parameters": {"model_inputs": profast_inputs_no2},
    }
    mean_hourly_production = 500000.0
    pf = ProFastNPV(
        driver_config={},
        plant_config=plant_config,
        tech_config=fake_filtered_tech_config,
        commodity_type="electricity",
        description="no2",
    )

    ivc = om.IndepVarComp()
    ivc.add_output("rated_electricity_production", mean_hourly_production, units="kW")
    ivc.add_output("capacity_factor", [1.0] * plant_config["plant"]["plant_life"], units="unitless")

    prob.model.add_subsystem("ivc", ivc, promotes=["*"])
    prob.model.add_subsystem("pf", pf, promotes=["rated_electricity_production", "capacity_factor"])

    expected_message = "commodity_sell_price is missing as an input"
    with subtests.test("Missing sell price"):
        with pytest.raises(ValueError) as excinfo:
            prob.setup()
        assert expected_message in str(excinfo.value)


@pytest.mark.regression
def test_profast_npv_with_inflation(
    profast_inputs_no2, fake_filtered_tech_config, fake_cost_dict, subtests
):
    price_escalation = 0.02  # 2% inflation in price
    pf_params = profast_inputs_no2["params"]
    pf_params["commodity"] = {"escalation": price_escalation}
    mean_hourly_production = 500000.0
    prob = om.Problem()

    years = np.arange(0, 34, 1)
    initial_price = 0.07
    # inflated_price = initial_price * ((1.0 + price_escalation) ** (years - 1))

    profast_inputs_no2["commodity_sell_price"] = [initial_price] * 30
    profast_inputs_no2["params"] = pf_params
    # profast_inputs_no2["params"]["inflation_rate"] = price_escalation  # 2% inflation
    plant_config = {
        "plant": {
            "plant_life": 30,
        },
        "finance_parameters": {"model_inputs": profast_inputs_no2},
    }
    pf = ProFastNPV(
        driver_config={},
        plant_config=plant_config,
        tech_config=fake_filtered_tech_config,
        commodity_type="electricity",
        description="no2",
    )

    ivc = om.IndepVarComp()
    ivc.add_output("rated_electricity_production", mean_hourly_production, units="kW")
    ivc.add_output("capacity_factor", [1.0] * plant_config["plant"]["plant_life"], units="unitless")

    prob.model.add_subsystem("ivc", ivc, promotes=["*"])
    prob.model.add_subsystem("pf", pf, promotes=["rated_electricity_production", "capacity_factor"])
    prob.setup()
    for variable, cost in fake_cost_dict.items():
        units = "USD" if "capex" in variable else "USD/year"
        prob.set_val(f"pf.{variable}", cost, units=units)

    prob.run_model()

    with subtests.test("Sell price (flat rate)"):
        assert (
            pytest.approx(
                prob.get_val("pf.sell_price_electricity_no2", units="USD/(kW*h)"), rel=1e-6
            )
            == profast_inputs_no2["commodity_sell_price"]
        )

    # This NPV is about 2.2x times the NPV when inflation is zero
    with subtests.test("NPV"):
        assert (
            pytest.approx(prob.get_val("pf.NPV_electricity_no2", units="USD")[0], rel=1e-6)
            == 1427542124.9970489
        )

    nominal_price = np.concatenate(
        [np.zeros(4), prob.get_val("pf.sell_price_electricity_no2", units="USD/(kW*h)")]
    )

    # Remove inflation from nominal price
    real_price = nominal_price / ((1.0 + price_escalation) ** (years - 1))

    prob.set_val("pf.sell_price_electricity_no2", real_price[4:], units="USD/(kW*h)")
    prob.run_model()
    # Removing inflation from nominal price gives NPV of 0.4 what is what before
    with subtests.test("NPV (real sell price)"):
        assert (
            pytest.approx(prob.get_val("pf.NPV_electricity_no2", units="USD")[0], rel=1e-6)
            == 611288384.4121004
        )


@pytest.mark.regression
def test_profast_npv_uses_first_year_price_for_construction_padding(
    profast_inputs_no2,
    subtests,
):
    mock_pf = MagicMock()
    mock_pf.cash_flow.return_value = 123.0

    # A non-divisible by 12 installation period exercises the construction-year boundary case.
    profast_inputs_no2["params"]["installation_time"] = 14
    profast_inputs_no2["commodity_sell_price"] = [0.07] * 30
    plant_life = 30
    tech_config = {"grid": {"model_inputs": {}}}

    plant_config = {
        "plant": {
            "plant_life": plant_life,
        },
        "finance_parameters": {"model_inputs": profast_inputs_no2},
    }

    mean_hourly_production = 500000.0
    prob = om.Problem()
    pf = ProFastNPV(
        driver_config={},
        plant_config=plant_config,
        tech_config=tech_config,
        commodity_type="electricity",
        description="no2",
    )

    ivc = om.IndepVarComp()
    ivc.add_output("rated_electricity_production", mean_hourly_production, units="kW")
    ivc.add_output("capacity_factor", [1.0] * plant_life, units="unitless")

    prob.model.add_subsystem("ivc", ivc, promotes=["*"])
    prob.model.add_subsystem("pf", pf, promotes=["rated_electricity_production", "capacity_factor"])
    prob.setup()

    prob.set_val("pf.capex_adjusted_grid", 1.0e6, units="USD")
    prob.set_val("pf.opex_adjusted_grid", 1.0e4, units="USD/year")
    prob.set_val("pf.varopex_adjusted_grid", [0.0] * plant_life, units="USD/year")
    prob.set_val("pf.replacement_schedule_grid", [0.0] * plant_life, units="unitless")

    with patch.object(ProFastNPV, "populate_profast", return_value=mock_pf):
        prob.run_model()

    expected_prefix_len = int(np.ceil(profast_inputs_no2["params"]["installation_time"] / 12) + 1)
    expected_price = profast_inputs_no2["commodity_sell_price"][0]
    calculated_price = np.asarray(mock_pf.cash_flow.call_args.kwargs["price"], dtype=float)

    with subtests.test("Construction-year padding uses first sell price"):
        assert np.allclose(calculated_price[:expected_prefix_len], expected_price)

    with subtests.test("First operating year keeps same sell price"):
        assert calculated_price[expected_prefix_len] == pytest.approx(expected_price)

    with subtests.test("NPV uses mocked cash_flow return"):
        assert prob.get_val("pf.NPV_electricity_no2", units="USD")[0] == pytest.approx(123.0)
