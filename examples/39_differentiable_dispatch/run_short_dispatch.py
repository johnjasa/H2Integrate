"""Size storage for a fixed one-week electricity profile.

This is a fast dispatch and cost-proxy diagnostic, not the full example's hydrogen TEA.
It uses the H2I dispatch controller and storage cost model with a fixed synthetic profile.
An optional CSV can provide 168 hourly electricity values in MW instead.
"""

import time
import argparse
from pathlib import Path

import numpy as np
import openmdao.api as om

from h2integrate.storage.generic_storage_cost import GenericStorageCostModel
from h2integrate.control.control_strategies.storage.differentiable_dispatch_controller import (
    DifferentiableDispatchStorageController,
)


N_HOURS = 168


def _validate_generation_profile(generation: np.ndarray) -> np.ndarray:
    if generation.shape != (N_HOURS,) or not np.all(np.isfinite(generation)):
        raise ValueError(f"Generation profile must contain {N_HOURS} finite hourly values.")
    if np.any(generation < 0):
        raise ValueError("Generation profile values must be nonnegative.")
    return generation


def load_generation_profile(path: Path | None = None) -> np.ndarray:
    """Return a fixed one-week generation profile in MW.

    Args:
        path: Optional headerless, single-column CSV with 168 hourly MW values.

    Returns:
        The one-week profile.

    Raises:
        ValueError: If the profile does not contain 168 finite, nonnegative values.
    """
    if path is None:
        hours = np.arange(N_HOURS)
        generation = 8.0 + 7.0 * np.sin(2 * np.pi * hours / 24) + 2.0 * np.cos(hours / 7)
        generation = np.maximum(0.5, generation)
    else:
        data = np.loadtxt(path, delimiter=",", ndmin=2)
        if data.shape != (N_HOURS, 1):
            raise ValueError(f"Generation CSV must contain {N_HOURS} rows and one column.")
        generation = data[:, 0]
    return _validate_generation_profile(generation)


def build_problem(generation: np.ndarray) -> om.Problem:
    """Build the illustrative one-week storage-sizing problem.

    The objective combines weekly dispatch cost with a prorated 12.5% annual capital cost.
    It is not LCOH or a replacement for annual simulation and finance.

    Args:
        generation: One-week electricity generation profile in MW.

    Returns:
        The initialized OpenMDAO problem.
    """
    generation = _validate_generation_profile(generation)
    plant = {"plant": {"plant_life": 30, "simulation": {"n_timesteps": N_HOURS, "dt": 3600}}}
    tech = {
        "model_inputs": {
            "shared_parameters": {
                "commodity": "electricity",
                "commodity_rate_units": "MW",
                "commodity_amount_units": "MW*h",
                "max_capacity": 40.0,
                "max_charge_rate": 10.0,
                "min_soc_fraction": 0.1,
                "max_soc_fraction": 0.9,
                "init_soc_fraction": 0.5,
                "charge_equals_discharge": True,
                "charge_efficiency": 0.95,
                "discharge_efficiency": 0.95,
                "demand_profile": 6.0,
                "solver_backend": "cached",
                "unmet_demand_penalty": 1000.0,
                "charge_cost": 1.0,
                "discharge_cost": 1.0,
            },
            "cost_parameters": {
                "capacity_capex": 173000.0,
                "charge_capex": 750000.0,
                "opex_fraction": 0.025,
                "cost_year": 2022,
            },
        }
    }
    driver = {"driver": {"optimization": {"flag": True, "total_derivatives": "components"}}}
    problem = om.Problem(reports=False)
    ivc = problem.model.add_subsystem("inputs", om.IndepVarComp(), promotes=["*"])
    ivc.add_output("electricity_in", generation, units="MW")
    ivc.add_output("electricity_set_point", np.full(N_HOURS, 6.0), units="MW")
    ivc.add_output("storage_capacity", 40.0, units="MW*h")
    ivc.add_output("max_charge_rate", 10.0, units="MW")
    problem.model.add_subsystem(
        "controller",
        DifferentiableDispatchStorageController(plant_config=plant, tech_config=tech),
        promotes=[
            "electricity_in",
            "electricity_set_point",
            "storage_capacity",
            "max_charge_rate",
            "dispatch_cost",
        ],
    )
    problem.model.add_subsystem(
        "battery_cost",
        GenericStorageCostModel(plant_config=plant, tech_config=tech, driver_config=driver),
        promotes_inputs=["storage_capacity", "max_charge_rate"],
    )
    problem.model.add_subsystem(
        "score",
        om.ExecComp(
            "week_cost = dispatch_cost + 0.125 * capex * (168 / 8760)",
            dispatch_cost={"units": "USD"},
            capex={"units": "USD"},
            week_cost={"units": "USD"},
        ),
        promotes_inputs=["dispatch_cost"],
    )
    problem.model.connect("battery_cost.CapEx", "score.capex")
    problem.driver = om.ScipyOptimizeDriver(optimizer="SLSQP", tol=1e-6, disp=False)
    problem.model.add_design_var("storage_capacity", lower=1.0, upper=100.0, ref=50.0)
    problem.model.add_design_var("max_charge_rate", lower=0.1, upper=25.0, ref=10.0)
    problem.model.add_objective("score.week_cost", ref=1e5)
    problem.setup(mode="rev")
    return problem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generation-csv", type=Path, help="Headerless CSV of 168 hourly MW values"
    )
    args = parser.parse_args()
    problem = build_problem(load_generation_profile(args.generation_csv))
    try:
        start = time.perf_counter()
        problem.run_model()
        initial_cost = problem.get_val("score.week_cost", units="USD").item()
        if not np.isfinite(initial_cost):
            raise RuntimeError("The initial one-week sizing objective is nonfinite.")
        problem.run_driver()
        if not problem.driver.result.success:
            raise RuntimeError(f"SLSQP did not converge: {problem.driver.result.exit_status}")
        final_cost = problem.get_val("score.week_cost", units="USD").item()
        if not np.isfinite(final_cost) or final_cost > initial_cost:
            raise RuntimeError("The one-week sizing run returned an invalid or worse objective.")
        print("One-week dispatch cost proxy (not hydrogen TEA):")
        print(
            f"  Battery: {problem.get_val('storage_capacity').item():.2f} MWh / "
            f"{problem.get_val('max_charge_rate').item():.2f} MW"
        )
        print(f"  Proxy: {initial_cost:.2f} -> {final_cost:.2f} USD")
        print(
            f"  Model evaluations: {problem.driver.result.model_evals}; "
            f"elapsed: {time.perf_counter() - start:.1f} s"
        )
    finally:
        problem.cleanup()


if __name__ == "__main__":
    main()
