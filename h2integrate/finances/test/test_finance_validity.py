from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest
import openmdao.api as om
import numpy_financial as npf

from h2integrate import H2IntegrateModel, load_yaml
from h2integrate.tools.profast_tools import create_and_populate_profast
from h2integrate.finances.profast_lco import ProFastLCO
from h2integrate.finances.profast_npv import ProFastNPV
from h2integrate.postprocess.reporting import print_results
from h2integrate.core.h2integrate_model import _FinanceEvaluationGroup
from h2integrate.postprocess.sql_to_csv import convert_sql_to_csv_summary
from h2integrate.finances.numpy_financial_npv import NumpyFinancialNPV


@pytest.fixture
def finance_problem(tmp_path):
    def build(
        model_class,
        *,
        requirement=None,
        save=False,
        objective=None,
        connected=False,
        second=False,
        record=False,
        financial_parameters=None,
    ):
        model_inputs = {
            "params": {
                "analysis_start_year": 2032,
                "installation_time": 12,
                "inflation_rate": 0.02,
                "discount_rate": 0.09,
                "debt_equity_ratio": 1.72,
                "property_tax_and_insurance": 0.015,
                "total_income_tax_rate": 0.2574,
                "capital_gains_tax_rate": 0.15,
                "sales_tax_rate": 0.0,
                "debt_interest_rate": 0.046,
                "debt_type": "Revolving debt",
                "loan_period_if_used": 0,
                "cash_onhand_months": 1,
                "admin_expense": 0.0,
            },
            "capital_items": {"depr_type": "MACRS", "depr_period": 5, "refurb": [0.0]},
            "save_profast_results": save,
            "save_profast_config": save,
        }
        if model_class is ProFastNPV:
            model_inputs.update(
                {"commodity_sell_price": 0.04, "commodity_sell_price_units": "USD/(kW*h)"}
            )
        elif model_class is NumpyFinancialNPV:
            model_inputs = {
                "real_discount_rate": 0.1,
                "commodity_sell_price": 0.04,
                "commodity_sell_price_units": "USD/(kW*h)",
                "save_cost_breakdown": save,
                "save_npv_breakdown": save,
            }
        options = {
            "driver_config": {"general": {"folder_output": str(tmp_path)}},
            "plant_config": {
                "plant": {"plant_life": 3},
                "finance_parameters": {
                    "model_inputs": model_inputs,
                    "minimum_annual_production": requirement,
                },
            },
            "tech_config": {
                "wind": {"model_inputs": {"financial_parameters": financial_parameters or {}}}
            },
            "commodity_type": "electricity",
        }
        component = model_class(**deepcopy(options), description="trial")
        components = [component]
        prob = om.Problem(
            model=_FinanceEvaluationGroup(finance_components=components), reports=False
        )
        ivc = om.IndepVarComp()
        ivc.add_output("rated_electricity_production", val=100.0, units="kW")
        ivc.add_output("capacity_factor", val=np.ones(3), units="unitless")
        if second:
            ivc.add_output("report_capacity", val=100.0, units="kW")
        prob.model.add_subsystem("ivc", ivc, promotes=["*"])
        prob.model.add_subsystem(
            "finance",
            component,
            promotes_inputs=["rated_electricity_production", "capacity_factor"],
        )
        if second:
            reporting = model_class(**deepcopy(options), description="report")
            components.append(reporting)
            prob.model.add_subsystem(
                "reporting",
                reporting,
                promotes_inputs=[
                    ("rated_electricity_production", "report_capacity"),
                    "capacity_factor",
                ],
            )
        if connected:
            prob.model.add_subsystem(
                "consumer",
                om.ExecComp(
                    "result=price", price={"units": "USD/(kW*h)"}, result={"units": "USD/(kW*h)"}
                ),
            )
            prob.model.connect("finance.LCOE_trial", "consumer.price")
        if objective:
            prob.model.add_design_var("rated_electricity_production", lower=0.0, upper=100.0)
            response = (
                "finance.LCOE_trial" if objective == "lco" else "rated_electricity_production"
            )
            prob.model.add_objective(response)
            prob.model.approx_totals(method="fd")
            prob.driver = om.ScipyOptimizeDriver(optimizer="SLSQP", disp=False)
        if record:
            prob.model.add_recorder(om.SqliteRecorder(str(tmp_path / "finance.sql")))
        prob.setup()
        for name in ["finance", "reporting"] if second else ["finance"]:
            prob.set_val(f"{name}.capex_adjusted_wind", 1000.0, units="USD")
            prob.set_val(f"{name}.opex_adjusted_wind", 10.0, units="USD/year")
            prob.set_val(f"{name}.varopex_adjusted_wind", [10.0] * 3, units="USD/year")
        return prob

    return build


@pytest.mark.unit
def test_lco_validity_exports_recording_and_failure_recovery(
    finance_problem, tmp_path, monkeypatch
):
    prob = finance_problem(ProFastLCO, save=True, record=True)
    prob.run_model()
    assert prob.get_val("finance.LCOE_trial_valid")
    assert prob.get_val("finance.LCOE_trial_breakdown")
    prob.set_val("rated_electricity_production", 0.0)
    # Keep recorder case names distinct across repeated evaluations
    with pytest.warns(UserWarning, match="undefined"):
        prob.run_model(reset_iter_counts=False)
    assert np.isnan(prob.get_val("finance.LCOE_trial")[0])
    assert prob.get_val("finance.LCOE_trial_status") == "undefined_zero_production"
    assert not prob.get_val("finance.LCOE_trial_valid")
    assert prob.get_val("finance.LCOE_trial_breakdown") == {}
    assert np.isnan(prob.get_val("finance.price_electricity_trial")[0])
    assert load_yaml(tmp_path / "ProFastLCO_electricity_trial_config.yaml")["status"] == (
        "undefined_zero_production"
    )
    report = print_results(prob.model, includes=["*LCOE_trial*"])
    assert report["explicit_outputs"]["finance.LCOE_trial_status"]["mean"] == (
        "undefined_zero_production"
    )

    prob.set_val("finance.capex_adjusted_wind", 0.0)
    prob.set_val("finance.opex_adjusted_wind", 0.0)
    prob.set_val("finance.varopex_adjusted_wind", np.zeros(3))
    with pytest.warns(UserWarning, match="undefined"):
        prob.run_model(reset_iter_counts=False)
    assert np.isnan(prob.get_val("finance.LCOE_trial")[0])
    prob.set_val("finance.capex_adjusted_wind", 1000.0)
    prob.set_val("finance.opex_adjusted_wind", 10.0)
    prob.set_val("finance.varopex_adjusted_wind", [10.0] * 3)
    prob.set_val("rated_electricity_production", 100.0)
    prob.run_model(reset_iter_counts=False)
    assert prob.get_val("finance.LCOE_trial_valid")

    def fail():
        raise om.AnalysisError("operations failed")

    with monkeypatch.context() as patcher:
        patcher.setattr(prob.model.ivc, "_solve_nonlinear", fail)
        with pytest.raises(om.AnalysisError, match="operations failed"):
            prob.run_model(reset_iter_counts=False)
    assert np.isnan(prob.get_val("finance.LCOE_trial")[0])
    assert prob.get_val("finance.LCOE_trial_status") == "analysis_failed"
    assert load_yaml(tmp_path / "ProFastLCO_electricity_trial_config.yaml")["status"] == (
        "analysis_failed"
    )
    prob.run_model(reset_iter_counts=False)
    assert prob.get_val("finance.LCOE_trial_valid")
    prob.cleanup()
    summary = convert_sql_to_csv_summary(tmp_path / "finance.sql", save_to_file=False)
    undefined = summary["finance.LCOE_trial_status"] == "undefined_zero_production"
    assert undefined.any()
    assert summary.loc[undefined, "finance.LCOE_trial (USD/kW/h)"].isna().all()


@pytest.mark.unit
@pytest.mark.parametrize("model_class", [ProFastLCO, ProFastNPV, NumpyFinancialNPV])
def test_finance_input_and_production_requirements(finance_problem, model_class, subtests):
    prob = finance_problem(model_class, requirement={"value": 500.0, "units": "MW*h"})
    result = "LCOE_trial" if model_class is ProFastLCO else "NPV_electricity_trial"
    prob.run_model()
    assert prob.get_val(f"finance.{result}_valid")

    with subtests.test("minimum is checked for each year using converted units"):
        prob.set_val("capacity_factor", [1.0, 0.5, 1.0])
        with pytest.raises(om.AnalysisError, match=r"year 2.*438"):
            prob.run_model()
        assert prob.get_val(f"finance.{result}_status") == "production_requirement_violated"
        assert np.isnan(prob.get_val(f"finance.{result}")[0])
        prob.set_val("capacity_factor", np.ones(3))

    with subtests.test("negative capacity is invalid"):
        prob.set_val("rated_electricity_production", -1.0)
        with pytest.raises(om.AnalysisError, match="rated_electricity_production"):
            prob.run_model()
        assert prob.get_val(f"finance.{result}_status") == "analysis_failed"
        prob.set_val("rated_electricity_production", 100.0)

    with subtests.test("capacity factor and nonfinite costs are validated"):
        prob.set_val("capacity_factor", [1.0, 1.1, 1.0])
        with pytest.raises(om.AnalysisError, match="capacity_factor"):
            prob.run_model()
        prob.set_val("capacity_factor", np.ones(3))
        prob.set_val("finance.opex_adjusted_wind", np.nan)
        with pytest.raises(om.AnalysisError, match="opex_adjusted_wind"):
            prob.run_model()
        prob.set_val("finance.opex_adjusted_wind", 10.0)
        prob.run_model()
        assert prob.get_val(f"finance.{result}_valid")

    with subtests.test("invalid minimum and incompatible units are rejected"):
        prob.model.finance.production_requirement = {"value": np.inf, "units": "MW*h"}
        with pytest.raises(ValueError, match="positive and finite"):
            prob.run_model()
        prob.model.finance.production_requirement = {"value": 1.0, "units": "kg"}
        with pytest.raises(ValueError, match="incompatible"):
            prob.run_model()


@pytest.mark.unit
def test_zero_production_npv_and_replacement_cash_flows(finance_problem, subtests):
    with subtests.test("ProFAST rejects zero production and recovers"):
        prob = finance_problem(ProFastNPV)
        prob.run_model()
        prob.set_val("rated_electricity_production", 0.0)
        with pytest.raises(om.AnalysisError, match=r"zero production.*not supported"):
            prob.run_model()
        assert np.isnan(prob.get_val("finance.NPV_electricity_trial")[0])
        assert prob.get_val("finance.NPV_electricity_trial_status") == "unsupported_zero_production"
        prob.set_val("rated_electricity_production", 100.0)
        prob.run_model()
        assert prob.get_val("finance.NPV_electricity_trial_valid")

    with subtests.test("NumPy NPV retains idle and replacement costs"):
        prob = finance_problem(
            NumpyFinancialNPV,
            financial_parameters={"capital_items": {"replacement_cost_percent": 0.1}},
        )
        prob.set_val("rated_electricity_production", 0.0)
        prob.set_val("finance.varopex_adjusted_wind", np.zeros(3))
        prob.set_val("finance.replacement_schedule_wind", [0.0, 1.0, 0.0])
        prob.run_model()
        assert prob.get_val("finance.NPV_electricity_trial")[0] == pytest.approx(
            npf.npv(0.1, [-1000.0, -10.0, -110.0, -10.0])
        )
        prob.set_val("finance.capex_adjusted_wind", 0.0)
        prob.set_val("finance.opex_adjusted_wind", 0.0)
        prob.run_model()
        assert prob.get_val("finance.NPV_electricity_trial")[0] == 0.0
        assert prob.get_val("finance.NPV_electricity_trial_valid")


@pytest.mark.unit
@pytest.mark.parametrize("model_class", [ProFastLCO, ProFastNPV])
def test_mixed_year_costs_and_zero_production(finance_problem, model_class):
    prob = finance_problem(model_class)
    prob.set_val("capacity_factor", [1.0, 0.0, 1.0])
    prob.set_val("finance.varopex_adjusted_wind", [-8.0, 0.0, 6.0], units="USD/year")
    with patch(
        "h2integrate.finances.profast_baseclass.create_and_populate_profast",
        wraps=create_and_populate_profast,
    ) as create:
        prob.run_model()
    config = create.call_args.args[0]
    np.testing.assert_allclose(
        list(config["feedstocks"]["wind"]["cost"].values()), [0.0, 0.0, 6.0 / 876000.0]
    )
    np.testing.assert_allclose(
        list(config["coproducts"]["wind"]["cost"].values()), [8.0 / 876000.0, 0.0, 0.0]
    )
    prob.set_val("finance.varopex_adjusted_wind", [-8.0, 1.0, 6.0], units="USD/year")
    with pytest.raises(om.AnalysisError, match=r"wind.*zero production"):
        prob.run_model()


@pytest.mark.unit
def test_optimizer_and_downstream_require_valid_lco(finance_problem, subtests):
    with subtests.test("undefined objective"):
        prob = finance_problem(ProFastLCO, objective="lco")
        prob.set_val("rated_electricity_production", 0.0)
        with pytest.raises(om.AnalysisError, match=r"undefined.*zero production"):
            prob.run_driver()

    with subtests.test("undefined downstream input"):
        prob = finance_problem(ProFastLCO, connected=True)
        prob.set_val("rated_electricity_production", 0.0)
        with pytest.raises(om.AnalysisError, match=r"undefined.*zero production"):
            prob.run_model()

    with subtests.test("production requirement during optimizer trials"):
        prob = finance_problem(
            ProFastLCO, objective="production", requirement={"value": 500.0, "units": "MW*h"}
        )
        with pytest.raises(om.AnalysisError, match="minimum annual production requirement"):
            prob.run_driver()


@pytest.mark.unit
def test_optimizer_refreshes_reporting_only_finance(finance_problem):
    prob = finance_problem(ProFastLCO, objective="lco", second=True)
    prob.run_driver()
    assert prob.model.reporting.iter_count > 1
    assert prob.get_val("reporting.LCOE_report_valid")
    assert np.isfinite(prob.get_val("reporting.LCOE_report")[0])


@pytest.mark.integration
def test_h2i_finance_subgroup_requirements(finance_problem, tmp_path):
    reference = finance_problem(ProFastLCO)
    lco_inputs = deepcopy(
        reference.model.finance.options["plant_config"]["finance_parameters"]["model_inputs"]
    )
    reference.cleanup()
    config = {
        "name": "required_optional_finance",
        "driver_config": {"general": {"folder_output": str(tmp_path), "create_om_reports": False}},
        "technology_config": {
            "technologies": {
                "grid": {
                    "performance_model": {"model": "GridPerformanceModel"},
                    "cost_model": {"model": "GridCostModel"},
                    "model_inputs": {
                        "shared_parameters": {"interconnection_size": 100.0},
                        "cost_parameters": {
                            "cost_year": 2022,
                            "interconnection_capex_per_kw": 50.0,
                            "interconnection_opex_per_kw": 2.0,
                            "fixed_interconnection_cost": 1000.0,
                        },
                    },
                }
            }
        },
        "plant_config": {
            "plant": {"plant_life": 3, "simulation": {"n_timesteps": 8760, "dt": 3600}},
            "finance_parameters": {
                "finance_groups": {
                    "lco": {"finance_model": "ProFastLCO", "model_inputs": lco_inputs},
                    "npv": {
                        "finance_model": "NumpyFinancialNPV",
                        "model_inputs": {
                            "real_discount_rate": 0.1,
                            "commodity_sell_price": 0.04,
                            "commodity_sell_price_units": "USD/(kW*h)",
                        },
                    },
                },
                "finance_subgroups": {
                    "optional": {
                        "commodity": "electricity",
                        "commodity_stream": "grid",
                        "technologies": ["grid"],
                        "finance_groups": ["lco"],
                    },
                    "combined": {
                        "commodity": "electricity",
                        "commodity_stream": "grid",
                        "technologies": ["grid"],
                        "finance_groups": ["lco", "npv"],
                        "use_commodity_stream_timeseries": True,
                        "commodity_stream_output": "electricity_out",
                        "minimum_annual_production": {"value": 500.0, "units": "MW*h"},
                    },
                },
                "cost_adjustment_parameters": {
                    "target_dollar_year": 2022,
                    "cost_year_adjustment_inflation": 0.02,
                },
            },
        },
    }
    h2i = H2IntegrateModel(config)
    h2i.setup()
    h2i.prob.set_val("grid.electricity_set_point", np.zeros(8760), units="kW")
    with pytest.warns(UserWarning, match="undefined"):
        with pytest.raises(om.AnalysisError, match="minimum annual production requirement"):
            h2i.prob.run_model()
    assert h2i.prob.get_val("finance_subgroup_optional.LCOE_status") == (
        "undefined_zero_production"
    )
    h2i.prob.set_val("grid.electricity_set_point", np.full(8760, 80.0), units="kW")
    h2i.prob.run_model()
    assert h2i.prob.get_val("finance_subgroup_optional.LCOE_valid")
    assert h2i.prob.get_val("finance_subgroup_combined.LCOE_lco_valid")
    assert h2i.prob.get_val("finance_subgroup_combined.NPV_electricity__npv_valid")
