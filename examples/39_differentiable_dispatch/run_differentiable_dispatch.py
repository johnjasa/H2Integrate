"""Differentiable optimal battery dispatch in H2Integrate.

Part 1 runs the example 13 plant (wind and solar feeding a battery and an electrolyzer) with the
battery dispatched by ``DifferentiableDispatchStorageController``, a drop-in replacement for the
open-loop demand controller. The controller solves a perfect-foresight dispatch optimization with
Ipopt and provides exact derivatives through its optimality (KKT) conditions.

Part 2 optimizes battery and solar sizing in the actual H2I plant, including the FLORIS, PySAM,
PEM, and ProFAST models. The PEM cluster count and rating are fixed because the PEM model rounds
cluster capacity to whole megawatts. Component-level derivatives combine analytic partials and
the dispatch KKT sensitivity with local finite differences for black-box models.
Pass ``--compare-cobyla`` to also run a derivative-free comparison.

Requires the optional ``dol`` dependencies: ``pip install -e ".[dol]"`` then
``idaes get-extensions``.
"""

import time
import argparse
from pathlib import Path

from sizing_problem import set_fixed_demand, build_sizing_problem, set_initial_sizing_design

from h2integrate import H2IntegrateModel, load_tech_yaml, load_plant_yaml, load_driver_yaml


def run_plant(this_dir):
    """Run the H2I plant with the differentiable battery controller."""
    tech_config = load_tech_yaml(this_dir / "tech_config.yaml")
    plant_config = load_plant_yaml(this_dir / "plant_config.yaml")
    driver_config = load_driver_yaml(this_dir / "driver_config.yaml")

    driver_config["general"]["folder_output"] = str(Path(this_dir / "outputs").absolute())
    tech_config["technologies"]["distributed_wind_plant"]["model_inputs"]["performance_parameters"][
        "cache_dir"
    ] = this_dir / "cache"

    h2i = H2IntegrateModel(
        {
            "plant_config": plant_config,
            "technology_config": tech_config,
            "driver_config": driver_config,
        }
    )
    h2i.setup()

    electrolyzer_capacity_MW = 60
    h2i.prob.set_val("battery.electricity_set_point", 0.1 * electrolyzer_capacity_MW, units="MW")
    h2i.prob.set_val("elec_load_demand.electricity_demand", electrolyzer_capacity_MW, units="MW")
    h2i.run()
    h2i.post_process(print_results=False, summarize_sql=True)
    return h2i


def run_sizing_study(this_dir, compare_cobyla=False):
    """Optimize the complete H2I plant, optionally comparing COBYLA."""
    results = {}
    optimizers = ("SLSQP", "COBYLA") if compare_cobyla else ("SLSQP",)
    for optimizer in optimizers:
        h2i = build_sizing_problem(this_dir, optimizer=optimizer)
        h2i.setup()
        set_fixed_demand(h2i)
        set_initial_sizing_design(h2i)
        prob = h2i.prob
        start = time.time()
        h2i.run()
        if not prob.driver.result.success:
            raise RuntimeError(f"{optimizer} did not converge: {prob.driver.result.exit_status}")
        capacity_factor = prob.get_val("electrolyzer.capacity_factor")[0]
        minimum = h2i.driver_config["constraints"]["electrolyzer"]["capacity_factor"]["lower"]
        if capacity_factor < minimum - 1e-5:
            raise RuntimeError(
                f"{optimizer} returned an infeasible first-year capacity factor: "
                f"{capacity_factor:.6f} < {minimum:.6f}"
            )
        results[optimizer] = {
            "LCOH": prob.get_val("finance_subgroup_hydrogen.LCOH", units="USD/kg").item(),
            "battery_MWh": prob.get_val("battery.storage_capacity", units="MW*h").item(),
            "battery_MW": prob.get_val("battery.max_charge_rate", units="MW").item(),
            "solar_MW": prob.get_val("distributed_pv.system_capacity_DC", units="MW").item(),
            "capacity_factor": capacity_factor,
            "model_evaluations": prob.driver.result.model_evals,
            "gradient_evaluations": prob.driver.result.deriv_evals,
            "seconds": time.time() - start,
        }
        prob.cleanup()

    print("\nFull-year H2I sizing study (fixed electrolyzer design):")
    header = (
        f"{'optimizer':>9} {'LCOH':>8} {'batt MWh':>9} {'batt MW':>8} {'solar MW':>8} "
        f"{'CF':>6} {'evals':>6} {'grads':>6} {'time s':>7}"
    )
    print(header)
    for name, r in results.items():
        print(
            f"{name:>9} {r['LCOH']:8.4f} {r['battery_MWh']:9.2f} {r['battery_MW']:8.2f} "
            f"{r['solar_MW']:8.2f} {r['capacity_factor']:6.3f} {r['model_evaluations']:6d} "
            f"{r['gradient_evaluations']:6d} {r['seconds']:7.1f}"
        )
    return results


def main(compare_cobyla=False):
    this_dir = Path(__file__).parent
    h2i = run_plant(this_dir)

    lcoh = h2i.prob.get_val("finance_subgroup_hydrogen.LCOH", units="USD/kg")[0]
    unmet = h2i.prob.get_val("battery.dispatch_electricity_unmet_demand", units="MW").sum()
    print(f"Plant LCOH with differentiable battery dispatch: {lcoh:.3f} USD/kg")
    print(f"Battery set-point not met: {unmet:.1f} MWh over the year")

    run_sizing_study(this_dir, compare_cobyla=compare_cobyla)


# The guard is required on Windows: Pyomo's PyNumero loader starts a multiprocessing worker.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compare-cobyla",
        action="store_true",
        help="Also run the slower derivative-free optimizer.",
    )
    main(compare_cobyla=parser.parse_args().compare_cobyla)
