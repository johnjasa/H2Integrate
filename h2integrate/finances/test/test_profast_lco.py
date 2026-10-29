from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest
import ProFAST
import openmdao.api as om
from pytest import fixture
from openmdao.utils.assert_utils import assert_check_totals, assert_check_partials

from h2integrate.finances.profast_lco import ProFastLCO
from h2integrate.finances.profast_baseclass import BasicProFASTParameterConfig


def _derivative_problem(model_inputs, mode="fwd", component_totals=True, output_dir=None):
    finance_inputs = deepcopy(model_inputs)
    plant_life = 10 if output_dir is not None else 3
    if output_dir is not None:
        finance_inputs.update(save_profast_config=True, save_profast_results=True)
    plant_config = {
        "plant": {"plant_life": plant_life},
        "finance_parameters": {"model_inputs": finance_inputs},
    }
    driver = {
        "driver": {
            "optimization": {
                "flag": True,
                "total_derivatives": "components" if component_totals else "fd",
            }
        },
        "general": {"folder_output": str(output_dir)},
    }
    tech_config = {
        "wind": {
            "model_inputs": {
                "financial_parameters": {"capital_items": {"replacement_cost_percent": 0.2}}
            }
        },
        "solar": {"model_inputs": {}},
    }
    prob = om.Problem(reports=False)
    ivc = om.IndepVarComp()
    ivc.add_output("rated_electricity_production", 500000.0, units="kW")
    ivc.add_output("capacity_factor", [0.8, 0.75, 0.7] + [0.7] * (plant_life - 3), units="unitless")
    prob.model.add_subsystem("ivc", ivc, promotes=["*"])
    prob.model.add_subsystem(
        "pf",
        ProFastLCO(
            driver_config=driver,
            plant_config=plant_config,
            tech_config=tech_config,
            commodity_type="electricity",
        ),
        promotes=["rated_electricity_production", "capacity_factor"],
    )
    prob.setup(mode=mode)
    for name, value in {
        "capex_adjusted_wind": 950054634.1,
        "opex_adjusted_wind": 21093892.68,
        "varopex_adjusted_wind": [2000000.0, 2100000.0, 2200000.0],
        "replacement_schedule_wind": [0.0, 0.5, 0.25],
        "capex_adjusted_solar": 6561339.6,
        "opex_adjusted_solar": 88372.77,
        "varopex_adjusted_solar": [300000.0, 320000.0, 340000.0],
    }.items():
        if isinstance(value, list):
            extension = 0.0 if name.startswith("replacement_schedule") else value[-1]
            value = value + [extension] * (plant_life - 3)
        prob.set_val(f"pf.{name}", value)
    return prob


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("coproduct", [False, True])
def test_profast_light_partials_match_full_compute(profast_inputs_no1, mode, coproduct):
    if coproduct:
        profast_inputs_no1["params"]["inflation_rate"] = 0.02
        profast_inputs_no1["params"]["sales_tax_rate"] = 0.07
    prob = _derivative_problem(profast_inputs_no1, mode=mode)
    if coproduct:
        prob.set_val("pf.varopex_adjusted_wind", [-2000000.0, -2100000.0, -2200000.0])
    prob.run_model()
    for _ in range(2):
        partials = prob.check_partials(
            out_stream=None, method="fd", step=1e-5, step_calc="rel_avg", minimum_step=1e-6
        )
        lco_partials = {
            "pf": {key: value for key, value in partials["pf"].items() if key[0] == "LCOE"}
        }
        assert_check_partials(lco_partials, atol=1e-9, rtol=1e-6)
        totals = prob.check_totals(
            of=["pf.LCOE"],
            wrt=[
                "rated_electricity_production",
                "capacity_factor",
                "pf.capex_adjusted_wind",
                "pf.opex_adjusted_wind",
                "pf.varopex_adjusted_wind",
                "pf.replacement_schedule_wind",
            ],
            out_stream=None,
            method="fd",
            step=1e-5,
            step_calc="rel_avg",
        )
        assert_check_totals(totals, atol=1e-8, rtol=1e-6)
        prob.set_val("pf.capex_adjusted_wind", 700000000.0)
        prob.set_val("capacity_factor", [0.7, 0.65, 0.6])
        prob.set_val("pf.replacement_schedule_wind", [0.3, 0.2, 0.1])
        prob.model.pf.params.discount_rate = 0.12
        prob.run_model()


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_profast_partials_preserve_primal_and_exports(profast_inputs_no1, tmp_path, mode):
    prob = _derivative_problem(profast_inputs_no1, mode=mode, output_dir=tmp_path)
    reference = _derivative_problem(
        profast_inputs_no1, mode=mode, component_totals=False, output_dir=tmp_path
    )
    prob.run_model()
    exports = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    reference.run_model()
    names = [
        "LCOE",
        "price_electricity",
        "wacc_electricity",
        "crf_electricity",
        "irr_electricity",
        "profit_index_electricity",
        "investor_payback_period_electricity",
    ]
    for name in names:
        np.testing.assert_array_equal(prob.get_val(f"pf.{name}"), reference.get_val(f"pf.{name}"))
    breakdown = deepcopy(prob.get_val("pf.LCOE_breakdown"))
    assert breakdown == reference.get_val("pf.LCOE_breakdown")
    assert len(exports) == 3
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == exports

    with patch("h2integrate.finances.profast_lco.run_profast") as full_run:
        totals = prob.compute_totals(
            of=["pf.LCOE"],
            wrt=["pf.capex_adjusted_wind", "capacity_factor"],
            return_format="array",
        )
    full_run.assert_not_called()
    assert np.all(np.isfinite(totals))
    for name in names:
        np.testing.assert_array_equal(prob.get_val(f"pf.{name}"), reference.get_val(f"pf.{name}"))
    assert prob.get_val("pf.LCOE_breakdown") == breakdown
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == exports
    prob.run_model()
    assert prob.get_val("pf.LCOE_breakdown") == breakdown
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == exports


@pytest.mark.unit
@pytest.mark.parametrize("component_totals", [False, True])
def test_profast_whole_model_fd_runs_full_compute(profast_inputs_no1, component_totals, tmp_path):
    prob = _derivative_problem(
        profast_inputs_no1, component_totals=component_totals, output_dir=tmp_path
    )
    prob.model.approx_totals(method="fd", step=1000.0)
    prob.run_model()
    baseline = prob.get_val("pf.LCOE")[0]
    prob.set_val("pf.capex_adjusted_wind", 950055634.1)
    prob.run_model()
    expected = (prob.get_val("pf.LCOE")[0] - baseline) / 1000.0
    prob.set_val("pf.capex_adjusted_wind", 950054634.1)
    prob.run_model()
    from h2integrate.finances.profast_lco import (
        run_profast,
        format_profast_price_breakdown_per_year,
    )

    with (
        patch("h2integrate.finances.profast_lco.run_profast", wraps=run_profast) as full_run,
        patch(
            "h2integrate.finances.profast_lco.format_profast_price_breakdown_per_year",
            wraps=format_profast_price_breakdown_per_year,
        ) as export,
    ):
        total = prob.compute_totals(of=["pf.LCOE"], wrt=["pf.capex_adjusted_wind"])
    assert full_run.call_count > 0
    assert export.call_count == full_run.call_count
    assert total["pf.LCOE", "pf.capex_adjusted_wind"][0, 0] == pytest.approx(expected)


@pytest.mark.unit
def test_profast_partials_invalid_production_and_solver_errors(profast_inputs_no1):
    prob = _derivative_problem(profast_inputs_no1)
    prob.set_val("rated_electricity_production", -1.0)
    with pytest.warns(UserWarning, match="zero capacity"):
        prob.run_model()
    with pytest.warns(UserWarning, match="zero capacity"):
        total = prob.compute_totals(of=["pf.LCOE"], wrt=["pf.capex_adjusted_wind"])
    assert total["pf.LCOE", "pf.capex_adjusted_wind"][0, 0] == 0.0
    prob.set_val("rated_electricity_production", 500000.0)
    prob.set_val("capacity_factor", [0.0, 0.0, 0.0])
    with pytest.warns(UserWarning, match="zero capacity factor"):
        prob.run_model()
    with pytest.warns(UserWarning, match="zero capacity factor"):
        total = prob.compute_totals(of=["pf.LCOE"], wrt=["pf.capex_adjusted_wind"])
    assert total["pf.LCOE", "pf.capex_adjusted_wind"][0, 0] == 0.0
    prob.set_val("capacity_factor", [0.8, 0.75, 0.7])
    prob.run_model()
    with patch.object(prob.model.pf, "populate_profast", side_effect=RuntimeError("solver failed")):
        with pytest.raises(RuntimeError, match="solver failed"):
            prob.compute_totals(of=["pf.LCOE"], wrt=["pf.capex_adjusted_wind"])
    prob.run_model()
    assert np.isfinite(prob.get_val("pf.LCOE")[0])


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_profast_relevance_changes_between_totals(profast_inputs_no1, mode):
    prob = _derivative_problem(profast_inputs_no1, mode=mode)
    prob.run_model()
    original_solve = ProFAST.ProFAST.solve_price
    counts = []

    def count_solve(pf):
        counts.append(1)
        return original_solve(pf)

    for wrt, expected_calls in [
        ("pf.capex_adjusted_wind", 2),
        ("capacity_factor", 4),
        ("pf.varopex_adjusted_wind", 4),
        ("pf.capex_adjusted_solar", 2),
        ("pf.capex_adjusted_wind", 2),
    ]:
        counts.clear()
        with patch.object(ProFAST.ProFAST, "solve_price", count_solve):
            prob.compute_totals(of=["pf.LCOE"], wrt=[wrt])
        assert len(counts) == expected_calls
        totals = prob.check_totals(
            of=["pf.LCOE"],
            wrt=[wrt],
            out_stream=None,
            method="fd",
            step=1e-5,
            step_calc="rel_avg",
        )
        assert_check_totals(totals, atol=1e-8, rtol=1e-6)


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
    model_inputs = {"params": params, "capital_items": cap_items}

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
def test_profast_component_derivatives(profast_inputs_no1):
    plant_config = {
        "plant": {"plant_life": 3},
        "finance_parameters": {"model_inputs": profast_inputs_no1},
    }
    driver = {"driver": {"optimization": {"flag": True, "total_derivatives": "components"}}}
    prob = om.Problem(reports=False)
    ivc = om.IndepVarComp()
    ivc.add_output("rated_electricity_production", 500000.0, units="kW")
    ivc.add_output("capacity_factor", [0.8, 0.8, 0.8], units="unitless")
    prob.model.add_subsystem("ivc", ivc, promotes=["*"])
    prob.model.add_subsystem(
        "pf",
        ProFastLCO(
            driver_config=driver,
            plant_config=plant_config,
            tech_config={"wind": {"model_inputs": {}}},
            commodity_type="electricity",
        ),
        promotes=["rated_electricity_production", "capacity_factor"],
    )
    prob.setup()
    prob.set_val("pf.capex_adjusted_wind", 950054634.1, units="USD")
    prob.set_val("pf.opex_adjusted_wind", 21093892.68, units="USD/year")
    prob.run_model()
    derivative = prob.compute_totals(
        of=["pf.LCOE"], wrt=["pf.capex_adjusted_wind"], return_format="array"
    )[0, 0]
    samples = []
    for capex in (950154634.1, 949954634.1):
        prob.set_val("pf.capex_adjusted_wind", capex, units="USD")
        prob.run_model()
        samples.append(prob.get_val("pf.LCOE")[0])
    assert derivative > 0
    assert derivative == pytest.approx((samples[0] - samples[1]) / 200000.0, rel=1e-4)


@pytest.mark.unit
def test_profast_params_config(profast_inputs_no1, subtests):
    profast_params = profast_inputs_no1["params"]
    profast_params["plant_life"] = 30
    profast_params["inflation_rate"] = 0.02
    profast_params["incidental_revenue"] = {"value": 0.0, "escalation": 0.03}
    profast_params["road_tax"] = {"value": 0.0, "escalation": 0.0}
    pf_config = BasicProFASTParameterConfig.from_dict(profast_params)
    pf_params = pf_config.as_dict()
    with subtests.test("Incidental revenue escalation rate"):
        assert pf_params["incidental revenue"]["escalation"] == 0.03

    with subtests.test("Road tax escalation rate"):
        assert pf_params["road tax"]["escalation"] == 0.0

    with subtests.test("Commodity escalation rate"):
        assert pf_params["commodity"]["escalation"] == 0.02

    with subtests.test("General inflation rate"):
        assert pf_params["general inflation rate"] == 0.02

    with subtests.test("Labor escalation rate"):
        assert pf_params["labor"]["escalation"] == 0.02

    with subtests.test("Maintenance escalation rate"):
        assert pf_params["maintenance"]["escalation"] == 0.02


@pytest.mark.regression
def test_profast_comp(profast_inputs_no1, fake_filtered_tech_config, fake_cost_dict, subtests):
    mean_hourly_production = 500000.0
    prob = om.Problem()
    plant_config = {
        "plant": {
            "plant_life": 30,
        },
        "finance_parameters": {"model_inputs": profast_inputs_no1},
    }
    pf = ProFastLCO(
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

    lcoe = prob.get_val("pf.LCOE_no1", units="USD/(MW*h)")
    price = prob.get_val("pf.price_electricity_no1", units="USD/(MW*h)")

    wacc = prob.get_val("pf.wacc_electricity_no1", units="percent")
    crf = prob.get_val("pf.crf_electricity_no1", units="percent")
    profit_index = prob.get_val("pf.profit_index_electricity_no1", units="unitless")
    irr = prob.get_val("pf.irr_electricity_no1", units="percent")
    ipp = prob.get_val("pf.investor_payback_period_electricity_no1", units="yr")

    lcoe_breakdown = prob.get_val("pf.LCOE_no1_breakdown")

    with subtests.test("LCOE"):
        assert pytest.approx(lcoe[0], rel=1e-6) == 63.8181779

    with subtests.test("WACC"):
        assert pytest.approx(wacc[0], rel=1e-6) == 0.056453864

    with subtests.test("CRF"):
        assert pytest.approx(crf[0], rel=1e-6) == 0.0674704169

    with subtests.test("Profit Index"):
        assert pytest.approx(profit_index[0], rel=1e-6) == 2.12026237778

    with subtests.test("IRR"):
        assert pytest.approx(irr[0], rel=1e-6) == 0.0948

    with subtests.test("Investor payback period"):
        assert pytest.approx(ipp[0], rel=1e-6) == 8

    with subtests.test("LCOE == price"):
        assert pytest.approx(lcoe, rel=1e-6) == price

    with subtests.test("LCOE breakdown total"):
        assert pytest.approx(lcoe_breakdown["LCOE: Total ($/kW*h)"] * 1e3, rel=1e-6) == lcoe

    with subtests.test("Wind operating costs in LCOE breakdown"):
        assert lcoe_breakdown["LCOE: wind Variable OpEx ($/kW*h)"] == 0.0
        assert lcoe_breakdown["LCOE: wind OpEx ($/kW*h)"] > 0.0

    with subtests.test("Wind OpEx cost in LCOE breakdown"):
        assert (
            pytest.approx(lcoe_breakdown["LCOE: wind OpEx ($/kW*h)"], rel=1e-6)
            == 0.004815957232876712
        )

    with subtests.test("Natural gas OpEx and VarOpEx in breakdown"):
        assert (
            lcoe_breakdown["LCOE: natural_gas OpEx ($/kW*h)"]
            != lcoe_breakdown["LCOE: natural_gas Variable OpEx ($/kW*h)"]
        )

    with subtests.test("Natural Gas OpEx costs in LCOE breakdown"):
        assert (
            pytest.approx(lcoe_breakdown["LCOE: natural_gas Variable OpEx ($/kW*h)"], rel=1e-6)
            == 0.014944754999999997
        )
        assert (
            pytest.approx(lcoe_breakdown["LCOE: natural_gas OpEx ($/kW*h)"], rel=1e-6)
            == 0.0029186880319634705
        )


@pytest.mark.regression
def test_profast_comp_coproduct(
    profast_inputs_no1, fake_filtered_tech_config, fake_cost_dict, subtests
):
    mean_hourly_production = 500000.0  # kW*h
    grid_sell_price = 63.8181779 / 1e3  # USD/(kW*h)
    wind_sold_USD = [-1 * mean_hourly_production * 8760 * grid_sell_price] * 30
    fake_cost_dict.update({"varopex_adjusted_wind": wind_sold_USD})

    prob = om.Problem()
    plant_config = {
        "plant": {
            "plant_life": 30,
        },
        "finance_parameters": {"model_inputs": profast_inputs_no1},
    }
    pf = ProFastLCO(
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

    lcoe = prob.get_val("pf.LCOE_no1", units="USD/(MW*h)")
    price = prob.get_val("pf.price_electricity_no1", units="USD/(MW*h)")

    wacc = prob.get_val("pf.wacc_electricity_no1", units="percent")
    crf = prob.get_val("pf.crf_electricity_no1", units="percent")
    profit_index = prob.get_val("pf.profit_index_electricity_no1", units="unitless")
    irr = prob.get_val("pf.irr_electricity_no1", units="percent")
    ipp = prob.get_val("pf.investor_payback_period_electricity_no1", units="yr")

    lcoe_breakdown = prob.get_val("pf.LCOE_no1_breakdown")

    with subtests.test("LCOE"):
        assert pytest.approx(lcoe[0], abs=1e-6) == 0

    with subtests.test("WACC"):
        assert pytest.approx(wacc[0], rel=1e-6) == 0.056453864

    with subtests.test("CRF"):
        assert pytest.approx(crf[0], rel=1e-6) == 0.0674704169

    with subtests.test("Profit Index"):
        assert pytest.approx(profit_index[0], rel=1e-6) == 2.12026237778

    with subtests.test("IRR"):
        assert pytest.approx(irr[0], rel=1e-6) == 0.0948

    with subtests.test("Investor payback period"):
        assert pytest.approx(ipp[0], rel=1e-6) == 8

    with subtests.test("LCOE == price"):
        assert pytest.approx(lcoe, rel=1e-6) == price

    with subtests.test("LCOE breakdown total"):
        assert pytest.approx(lcoe_breakdown["LCOE: Total ($/kW*h)"] * 1e3, rel=1e-6) == lcoe


@pytest.mark.regression
def test_profast_comp_heat(profast_inputs_no1, fake_filtered_tech_config, fake_cost_dict, subtests):
    mean_hourly_production = 500000.0
    prob = om.Problem()
    plant_config = {
        "plant": {
            "plant_life": 30,
        },
        "finance_parameters": {"model_inputs": profast_inputs_no1},
    }
    pf = ProFastLCO(
        driver_config={},
        plant_config=plant_config,
        tech_config=fake_filtered_tech_config,
        commodity_type="heat",
        description="no1",
    )
    ivc = om.IndepVarComp()

    ivc.add_output("rated_heat_production", mean_hourly_production, units="MMBtu/h")
    ivc.add_output("capacity_factor", [1.0] * plant_config["plant"]["plant_life"], units="unitless")

    prob.model.add_subsystem("ivc", ivc, promotes=["*"])
    prob.model.add_subsystem("pf", pf, promotes=["rated_heat_production", "capacity_factor"])
    prob.setup()
    for variable, cost in fake_cost_dict.items():
        units = "USD" if "capex" in variable else "USD/year"
        prob.set_val(f"pf.{variable}", cost, units=units)

    prob.run_model()

    with subtests.test("LCO-Heat"):
        assert (
            pytest.approx(prob.get_val("pf.LCOH_no1", units="USD/(MMBtu)")[0], rel=1e-6)
            == 63.8181779 * 1e-3
        )

    with subtests.test("WACC"):
        assert (
            pytest.approx(prob.get_val("pf.wacc_heat_no1", units="percent")[0], rel=1e-6)
            == 0.056453864
        )

    with subtests.test("CRF"):
        assert (
            pytest.approx(prob.get_val("pf.crf_heat_no1", units="percent")[0], rel=1e-6)
            == 0.0674704169
        )

    with subtests.test("Profit Index"):
        assert (
            pytest.approx(prob.get_val("pf.profit_index_heat_no1", units="unitless")[0], rel=1e-6)
            == 2.12026237778
        )


@pytest.mark.regression
def test_profast_comp_sales_tax(
    profast_inputs_no1, fake_filtered_tech_config, fake_cost_dict, subtests
):
    profast_inputs_no1["params"]["sales_tax_rate"] = 0.07
    mean_hourly_production = 500000.0
    mean_production_per_sec = mean_hourly_production / 3600  #
    prob = om.Problem()
    plant_config = {
        "plant": {
            "plant_life": 30,
        },
        "finance_parameters": {"model_inputs": profast_inputs_no1},
    }
    pf = ProFastLCO(
        driver_config={},
        plant_config=plant_config,
        tech_config=fake_filtered_tech_config,
        commodity_type="hydrogen",
        description="no1",
    )
    ivc = om.IndepVarComp()

    ivc.add_output("rated_hydrogen_production", mean_production_per_sec, units="kg/s")
    ivc.add_output("capacity_factor", [0.5] * plant_config["plant"]["plant_life"], units="unitless")

    prob.model.add_subsystem("ivc", ivc, promotes=["*"])
    prob.model.add_subsystem("pf", pf, promotes=["rated_hydrogen_production", "capacity_factor"])
    prob.setup()
    for variable, cost in fake_cost_dict.items():
        units = "USD" if "capex" in variable else "USD/year"
        prob.set_val(f"pf.{variable}", cost, units=units)

    prob.run_model()

    lcoh = prob.get_val("pf.LCOH_no1", units="USD/kg")
    price = prob.get_val("pf.price_hydrogen_no1", units="USD/kg")
    lcoh_breakdown = prob.get_val("pf.LCOH_no1_breakdown")

    with subtests.test("LCOH"):
        assert pytest.approx(lcoh[0], rel=1e-6) == 0.13724339329249208

    with subtests.test("LCOH == price"):
        assert pytest.approx(lcoh, rel=1e-6) == price

    with subtests.test("LCOH breakdown total"):
        assert pytest.approx(lcoh_breakdown["LCOH: Total ($/kg*h/s)"] / 3600, rel=1e-6) == lcoh


@pytest.mark.regression
def test_profast_comp_sales_tax_different_units(
    profast_inputs_no1, fake_filtered_tech_config, fake_cost_dict, subtests
):
    # this is the same as test_profast_comp_sales_tax but sets the rated_hydrogen_production
    # in units of kg/h (instead of kg/s). This ensures that the ProFAST LCO model properly handles
    # different rate units (LCOH should not change for different rate units)
    profast_inputs_no1["params"]["sales_tax_rate"] = 0.07
    mean_hourly_production = 500000.0
    prob = om.Problem()
    plant_config = {
        "plant": {
            "plant_life": 30,
        },
        "finance_parameters": {"model_inputs": profast_inputs_no1},
    }
    pf = ProFastLCO(
        driver_config={},
        plant_config=plant_config,
        tech_config=fake_filtered_tech_config,
        commodity_type="hydrogen",
        description="no1",
    )
    ivc = om.IndepVarComp()

    ivc.add_output("rated_hydrogen_production", mean_hourly_production, units="kg/h")
    ivc.add_output("capacity_factor", [0.5] * plant_config["plant"]["plant_life"], units="unitless")

    prob.model.add_subsystem("ivc", ivc, promotes=["*"])
    prob.model.add_subsystem("pf", pf, promotes=["rated_hydrogen_production", "capacity_factor"])
    prob.setup()
    for variable, cost in fake_cost_dict.items():
        units = "USD" if "capex" in variable else "USD/year"
        prob.set_val(f"pf.{variable}", cost, units=units)

    prob.run_model()

    lcoh = prob.get_val("pf.LCOH_no1", units="USD/kg")
    price = prob.get_val("pf.price_hydrogen_no1", units="USD/kg")
    lcoh_breakdown = prob.get_val("pf.LCOH_no1_breakdown")

    with subtests.test("LCOH"):
        assert pytest.approx(lcoh[0], rel=1e-6) == 0.13724339329249208

    with subtests.test("LCOH == price"):
        assert pytest.approx(lcoh, rel=1e-6) == price

    with subtests.test("LCOH breakdown total"):
        assert pytest.approx(lcoh_breakdown["LCOH: Total ($/kg)"], rel=1e-6) == lcoh


@pytest.mark.regression
@pytest.mark.parametrize("use_slc", [True, False])
def test_profast_lco_warnings(
    profast_inputs_no1, fake_filtered_tech_config, fake_cost_dict, subtests, use_slc
):
    mean_hourly_production = 500000.0
    prob = om.Problem()
    plant_config = {
        "plant": {
            "plant_life": 30,
        },
        "finance_parameters": {"model_inputs": profast_inputs_no1},
    }
    if use_slc:
        plant_config["system_level_control"] = {}
    pf = ProFastLCO(
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

    # Set rated capacity to 0
    prob.set_val("pf.rated_electricity_production", 0.0, units="kW")

    with subtests.test("Test zero capacity warning"):
        expected_str = "has a zero capacity."
        with pytest.warns(UserWarning) as excinfo:
            prob.run_model()
        assert expected_str in str(excinfo.list[0].message)
        assert "default value of 1e12" in str(excinfo.list[0].message)

    with subtests.test("Test zero capacity value"):
        assert prob.get_val("pf.LCOE_no1", units="TUSD/(kW*h)")[0] == 1.0

    # Set capacity factor to 0
    prob.set_val("pf.rated_electricity_production", mean_hourly_production, units="kW")
    prob.set_val(
        "pf.capacity_factor", [0.0] * plant_config["plant"]["plant_life"], units="unitless"
    )

    with subtests.test("Test zero capacity factor warning"):
        expected_str = "has a zero capacity factor."
        with pytest.warns(UserWarning) as excinfo:
            prob.run_model()
        assert expected_str in str(excinfo.list[0].message)
        assert "default value of 1e12" in str(excinfo.list[0].message)

    with subtests.test("Test zero capacity factor value"):
        assert prob.get_val("pf.LCOE_no1", units="TUSD/(kW*h)")[0] == 1.0
