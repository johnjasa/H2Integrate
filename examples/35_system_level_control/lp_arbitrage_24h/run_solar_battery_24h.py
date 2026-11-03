"""Demonstrate simultaneous solar export and grid charging in one SLC window.

Run this example from its directory with ``python run_solar_battery_24h.py``.
It writes ``solar_grid_charging.png`` in the current directory.

H2Integrate requires a year-long plant configuration. This script constructs
that plant but solves only one 24-hour window of its SLC optimization model.
It first finds the unconstrained optimum, then applies demonstration-only bounds
to select a tied optimum with at least 10 MW of import and export in hour 0.
The bounds use private Pyomo objects and are not a controller configuration
feature. The plot shows the optimized window, not a plant performance run.
"""

from typing import TypedDict
from pathlib import Path

import numpy as np
import pyomo.environ as pyo
import matplotlib.pyplot as plt

from h2integrate.core.h2integrate_model import H2IntegrateModel


class ExampleResults(TypedDict):
    """Power and price series and objectives from the example run."""

    sell_price: np.ndarray
    buy_price: np.ndarray
    grid_dispatch_price: np.ndarray
    solar: np.ndarray
    grid_import: np.ndarray
    grid_export: np.ndarray
    charge: np.ndarray
    discharge: np.ndarray
    curtailed: np.ndarray
    baseline_objective: float
    selected_objective: float
    grid_import_cost: float


def run_example(output_path: Path) -> ExampleResults:
    """Solve the controller window and save a plot of the selected schedule.

    Args:
        output_path: Destination for the diagnostic PNG.

    Returns:
        The optimized power and price series and the two dispatch objectives.

    Raises:
        RuntimeError: If the selected schedule is not tied with the original
            optimum or the optimized flows do not satisfy the example.
    """
    model = H2IntegrateModel(Path(__file__).with_name("solar_battery_24h.yaml"))
    model.setup()
    try:
        model.prob.final_setup()
        controller = model.prob.model.plant.system_level_controller
        if controller.window_len != 24:
            raise ValueError("This example requires a 24-hour controller window.")
        hours = np.arange(controller.window_len)
        sell_price = np.where(hours < 12, 0.05, 0.20)
        buy_price = sell_price - 0.02
        solar = np.asarray(
            model.technology_config["technologies"]["solar"]["model_inputs"][
                "performance_parameters"
            ]["production_profile"],
            dtype=float,
        )
        if solar.shape != (controller.window_len,):
            raise ValueError("The solar profile must contain 24 hourly values.")
        lp = controller._build_lp_model()
        grid_config = model.technology_config["technologies"]["grid_buy"]["model_inputs"]
        for grid in lp.D:
            lp.rated[grid] = grid_config["shared_parameters"]["interconnection_size"]
        for storage in lp.S:
            storage_config = controller.storage_params[storage]
            lp.soc_init[storage] = storage_config["init_soc_fraction"] * storage_config["capacity"]
        for t in lp.T:
            lp.must_run[t] = float(solar[t])
            lp.demand[t] = 0.0
            lp.sell_price[t] = float(sell_price[t])
            lp.export_limit[t] = controller.export_limit
            for grid in lp.D:
                lp.marginal_cost[grid, t] = float(sell_price[t])

        controller._solve_window(lp, 0)
        baseline_objective = float(pyo.value(lp.objective))

        # Select one of the tied optima; these bounds are not a production dispatch rule.
        lp.dispatch["grid_buy", 0].setlb(10_000.0)
        lp.export[0].setlb(10_000.0)
        controller._solve_window(lp, 0)
        selected_objective = float(pyo.value(lp.objective))

        results: ExampleResults = {
            "sell_price": sell_price,
            "buy_price": buy_price,
            "grid_dispatch_price": np.array(
                [pyo.value(lp.marginal_cost["grid_buy", t]) for t in lp.T]
            ),
            "solar": solar,
            "grid_import": np.array([pyo.value(lp.dispatch["grid_buy", t]) for t in lp.T]),
            "grid_export": np.array([pyo.value(lp.export[t]) for t in lp.T]),
            "charge": np.array([pyo.value(lp.charge["battery", t]) for t in lp.T]),
            "discharge": np.array([pyo.value(lp.discharge["battery", t]) for t in lp.T]),
            "curtailed": np.array([pyo.value(lp.curtail[t]) for t in lp.T]),
            "baseline_objective": baseline_objective,
            "selected_objective": selected_objective,
            "grid_import_cost": float(
                sum(pyo.value(lp.dispatch["grid_buy", t]) * buy_price[t] for t in lp.T)
            ),
        }
    finally:
        model.prob.cleanup()

    solar = results["solar"]
    imported = results["grid_import"]
    exported = results["grid_export"]
    charge = results["charge"]
    discharge = results["discharge"]
    curtailed = results["curtailed"]
    power_tolerance = 0.01  # kW
    if abs(baseline_objective - selected_objective) > 0.01:
        raise RuntimeError("The selected dispatch is not tied with the original optimum.")
    if (
        imported[0] < 10_000.0 - power_tolerance
        or exported[0] < 10_000.0 - power_tolerance
        or discharge[0] > power_tolerance
        or np.any(imported > charge + power_tolerance)
        or np.any(exported > solar + discharge + power_tolerance)
        or np.any((charge > power_tolerance) & (discharge > power_tolerance))
        or not np.allclose(
            imported + solar + discharge,
            charge + exported + curtailed,
            rtol=0,
            atol=power_tolerance,
        )
    ):
        raise RuntimeError("The optimized window flows do not satisfy the grid resale checks.")

    fig, (ax_price, ax_flow) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, layout="constrained"
    )
    ax_price.step(hours, sell_price * 1000, where="mid", label="Sell price (used for dispatch)")
    ax_price.step(hours, buy_price * 1000, where="mid", linestyle="--", label="Actual buy price")
    ax_price.set_ylabel("Price (USD/MWh)")
    ax_price.legend(ncol=2)
    ax_price.grid(alpha=0.25)

    width = 0.36
    ax_flow.bar(hours - width / 2, solar / 1000, width, label="Solar supply", color="gold")
    ax_flow.bar(
        hours - width / 2,
        imported / 1000,
        width,
        bottom=solar / 1000,
        label="Grid import",
        color="tab:blue",
    )
    ax_flow.bar(
        hours - width / 2,
        discharge / 1000,
        width,
        bottom=(solar + imported) / 1000,
        label="Battery discharge",
        color="tab:orange",
    )
    ax_flow.bar(
        hours + width / 2, charge / 1000, width, label="Battery charging", color="tab:purple"
    )
    ax_flow.bar(
        hours + width / 2,
        exported / 1000,
        width,
        bottom=charge / 1000,
        label="Grid export",
        color="tab:green",
    )
    ax_flow.bar(
        hours + width / 2,
        curtailed / 1000,
        width,
        bottom=(charge + exported) / 1000,
        label="Curtailment",
        color="lightgray",
    )
    ax_flow.axvspan(-0.5, 0.5, color="tab:blue", alpha=0.1)
    max_flow = (
        max(np.max(solar + imported + discharge), np.max(charge + exported + curtailed)) / 1000
    )
    ax_flow.set_ylim(0, max_flow * 1.22)
    ax_flow.annotate(
        f"Hour 0: import {imported[0] / 1000:.1f} MW,\n"
        f"export {exported[0] / 1000:.1f} MW,\n"
        f"charge {charge[0] / 1000:.1f} MW",
        xy=(0, (charge[0] + exported[0]) / 1000),
        xytext=(3, max_flow * 0.85),
        arrowprops={"arrowstyle": "->"},
    )
    ax_flow.set_xlabel("Hour")
    ax_flow.set_ylabel("Power (MW)")
    ax_flow.set_title("Each hour: left bar = sources; right bar = destinations")
    ax_flow.legend(ncol=3, loc="upper right", fontsize=9)
    ax_flow.grid(alpha=0.25, axis="y")
    fig.suptitle(
        "Solar export during grid charging: one 24-hour SLC window\n"
        "Hour 0 is constrained to select a tied optimum under the dispatch objective"
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    print(f"Plot saved to {output_path}")
    print(
        f"Hour 0: grid import {imported[0] / 1000:.1f} MW, "
        f"solar {solar[0] / 1000:.1f} MW, battery charge {charge[0] / 1000:.1f} MW, "
        f"grid export {exported[0] / 1000:.1f} MW"
    )
    print(
        f"Change in the controller's dispatch objective: "
        f"{selected_objective - baseline_objective:.6f} USD"
    )
    print(
        f"Illustrative 24-hour purchase cost at the buy price: "
        f"{results['grid_import_cost']:.2f} USD"
    )
    return results


if __name__ == "__main__":
    run_example(Path("solar_grid_charging.png"))
