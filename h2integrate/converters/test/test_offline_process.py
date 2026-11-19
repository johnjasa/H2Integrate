from pathlib import Path

import yaml
import numpy as np
import pytest
import openmdao.api as om

from h2integrate import H2IntegrateModel, load_tech_yaml
from h2integrate.converters.offline_process import OfflineProcessPerformanceModel
from h2integrate.converters.water.desal.desalination import ReverseOsmosisPerformanceModel


@pytest.mark.unit
def test_offline_process_point(subtests) -> None:
    """Prove common availability gating and rate-to-amount conversion at a half-hour timestep."""
    parameters = {
        "source": "Synthetic process fixture, version 1",
        "commodity": "water",
        "product": {"rate": 0.5, "units": "kg/s"},
        "commodity_amount_units": "kg",
        "requirements": {
            "electricity": {"rate": 4.0, "units": "kW"},
            "feedwater": {"rate": 1.0, "units": "kg/s"},
        },
        "coproducts": {"concentrate": {"rate": 0.5, "units": "kg/s"}},
    }
    plant_config = {"plant": {"plant_life": 2, "simulation": {"dt": 1800, "n_timesteps": 4}}}
    tech_config = {"model_inputs": {"performance_parameters": parameters}}
    problem = om.Problem()
    problem.model.add_subsystem(
        "process",
        OfflineProcessPerformanceModel(plant_config=plant_config, tech_config=tech_config),
        promotes=["*"],
    )
    problem.setup()
    problem.set_val("electricity_in", [4000, 3999, 5000, np.inf], units="W")
    problem.set_val("feedwater_in", [1.0, 1.0, 0.9, 1.5], units="kg/s")
    problem.run_model()

    with subtests.test("one shared mask drives production, utilities, and coproducts"):
        assert problem.get_val("water_out") == pytest.approx([0.5, 0, 0, 0])
        assert problem.get_val("electricity_consumed") == pytest.approx([4, 0, 0, 0])
        assert problem.get_val("feedwater_consumed") == pytest.approx([1, 0, 0, 0])
        assert problem.get_val("concentrate_out") == pytest.approx([0.5, 0, 0, 0])
    with subtests.test("rated and integrated outputs follow the H2Integrate convention"):
        assert problem.get_val("rated_water_production", units="kg/h") == pytest.approx(1800)
        assert problem.get_val("total_water_produced", units="kg") == pytest.approx(900)
        assert problem.get_val("annual_water_produced", units="kg/year") == pytest.approx(
            [3_942_000, 3_942_000]
        )
        assert problem.get_val("capacity_factor") == pytest.approx([0.25, 0.25])


@pytest.mark.unit
@pytest.mark.parametrize(
    "case_name",
    ["watertap_pressure_exchanger", "watertap_no_recovery", "prommis_decrepitation"],
)
def test_solved_pilot_replay(case_name: str, subtests) -> None:
    """Replay exported upstream solves without importing the upstream packages."""
    points = Path(__file__).parent / "inputs" / "offline_pilot_points.yaml"
    case = yaml.safe_load(points.read_text(encoding="utf-8"))["cases"][case_name]
    params = case["performance_parameters"]
    requirements = params["requirements"]
    n_timesteps = 1 + len(requirements)
    plant_config = {
        "plant": {"plant_life": 2, "simulation": {"dt": 3600, "n_timesteps": n_timesteps}}
    }
    problem = om.Problem()
    problem.model.add_subsystem(
        "process",
        OfflineProcessPerformanceModel(
            plant_config=plant_config,
            tech_config={"model_inputs": {"performance_parameters": params}},
        ),
        promotes=["*"],
    )
    is_ro = params["commodity"] == "water"
    if is_ro:
        validation = case["validation"]
        ro_parameters = {
            "freshwater_capacity_kg_per_hr": validation["permeate_mass_kg_per_h"],
            "freshwater_density": validation["permeate_density_kg_per_m3"],
            "design_point": {
                "source": params["source"],
                "feed_mass_flow_kg_per_hr": validation["feed_mass_kg_per_h"],
                "feed_density_kg_per_m3": validation["feed_density_kg_per_m3"],
                "net_electricity_kw": requirements["electricity"]["rate"],
                "feed_salt_mass_fraction": validation["feed_salt_mass_fraction"],
                "permeate_salt_mass_fraction": validation["permeate_salt_mass_fraction"],
                "max_permeate_salt_mass_fraction": validation["permeate_limit_mass_fraction"],
            },
        }
        problem.model.add_subsystem(
            "ro",
            ReverseOsmosisPerformanceModel(
                plant_config=plant_config,
                tech_config={"model_inputs": {"performance_parameters": ro_parameters}},
            ),
            promotes_inputs=["electricity_in", "feedwater_in"],
        )
        for name, stream in requirements.items():
            problem.model.set_input_defaults(f"{name}_in", units=stream["units"])

    problem.setup()
    for off_step, (name, stream) in enumerate(requirements.items(), start=1):
        available = np.full(n_timesteps, stream["rate"])
        available[off_step] *= 0.99
        problem.set_val(f"{name}_in", available, units=stream["units"])
    problem.run_model()

    product = params["product"]
    output = f"{params['commodity']}_out"
    with subtests.test("the solved load is replayed only when all requirements are met"):
        assert problem.get_val(output, units=product["units"]) == pytest.approx(
            [product["rate"], *([0] * (n_timesteps - 1))]
        )
        for name, stream in requirements.items():
            assert problem.get_val(f"{name}_consumed", units=stream["units"]) == pytest.approx(
                [stream["rate"], *([0] * (n_timesteps - 1))]
            )
        for name, stream in params["coproducts"].items():
            assert problem.get_val(f"{name}_out", units=stream["units"]) == pytest.approx(
                [stream["rate"], *([0] * (n_timesteps - 1))]
            )
    if is_ro:
        with subtests.test("shared and dedicated models agree at a qualified RO point"):
            for name in ("water_out", "electricity_consumed", "feedwater_consumed"):
                assert problem.get_val(f"ro.{name}") == pytest.approx(problem.get_val(name))
            for shared, dedicated in (
                ("concentrate_out", "concentrate_mass_flow"),
                ("concentrate_salt_out", "concentrate_salt_mass_flow"),
            ):
                assert problem.get_val(f"ro.{dedicated}") == pytest.approx(problem.get_val(shared))


@pytest.mark.integration
def test_offline_process_cost_and_finance(tmp_path, monkeypatch, subtests) -> None:
    """Load the model pair from YAML and verify the cost-to-finance handoff."""
    tech_config = load_tech_yaml(Path(__file__).parent / "inputs" / "offline_process_tech.yaml")
    monkeypatch.chdir(tmp_path)
    n_timesteps = 8760
    plant_life = 10
    model = H2IntegrateModel(
        {
            "name": "offline_process_finance_fixture",
            "system_summary": "Test a synthetic offline process point with costs and finance",
            "driver_config": {"general": {"create_om_reports": False}},
            "technology_config": tech_config,
            "plant_config": {
                "plant": {
                    "plant_life": plant_life,
                    "simulation": {"n_timesteps": n_timesteps, "dt": 3600},
                },
                "finance_parameters": {
                    "cost_adjustment_parameters": {
                        "target_dollar_year": 2022,
                        "cost_year_adjustment_inflation": 0.0,
                    },
                    "finance_groups": {
                        "lco": {
                            "finance_model": "ProFastLCO",
                            "model_inputs": {
                                "params": {
                                    "analysis_start_year": 2032,
                                    "installation_time": 12,
                                    "inflation_rate": 0.0,
                                    "discount_rate": 0.08,
                                    "debt_equity_ratio": 1.0,
                                    "property_tax_and_insurance": 0.0,
                                    "total_income_tax_rate": 0.0,
                                    "capital_gains_tax_rate": 0.0,
                                    "sales_tax_rate": 0.0,
                                    "debt_interest_rate": 0.046,
                                    "debt_type": "Revolving debt",
                                    "loan_period_if_used": 0,
                                    "cash_onhand_months": 1,
                                    "admin_expense": 0.0,
                                },
                                "capital_items": {
                                    "depr_type": "MACRS",
                                    "depr_period": 5,
                                    "refurb": [0.0],
                                },
                            },
                        }
                    },
                    "finance_subgroups": {
                        "electricity": {
                            "commodity": "electricity",
                            "commodity_stream": "process",
                            "commodity_desc": "fixture_electricity",
                            "technologies": ["process"],
                            "finance_groups": ["lco"],
                        }
                    },
                },
            },
        }
    )
    model.setup()
    problem = model.prob
    lcoe_name = "finance_subgroup_electricity.LCOE_fixture_electricity"
    problem.set_val("process.fuel_in", np.full(n_timesteps, 10.0), units="kg/h")
    model.run()
    full_lcoe = problem.get_val(lcoe_name, units="USD/(kW*h)").item()

    fuel_available = np.zeros(n_timesteps)
    fuel_available[: n_timesteps // 2] = 10.0
    problem.set_val("process.fuel_in", fuel_available, units="kg/h")
    model.run()

    with subtests.test("the cost component receives actual annual output"):
        assert problem.get_val("process.CapEx", units="USD") == pytest.approx(10000.0)
        assert problem.get_val("process.OpEx", units="USD/year") == pytest.approx(500.0)
        assert problem.get_val("process.VarOpEx", units="USD/year") == pytest.approx(
            np.full(plant_life, 4380.0)
        )
        assert problem.get_val("process.cost_year") == 2022
    with subtests.test("finance reflects lower utilization at the same installed cost"):
        half_lcoe = problem.get_val(lcoe_name, units="USD/(kW*h)").item()
        assert np.isfinite(half_lcoe)
        assert half_lcoe > full_lcoe > 0.02
