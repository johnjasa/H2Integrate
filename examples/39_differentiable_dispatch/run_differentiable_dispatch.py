"""Differentiable optimal battery dispatch in H2Integrate.

Part 1 runs the example 13 plant (wind and solar feeding a battery and an electrolyzer) with the
battery dispatched by ``DifferentiableDispatchStorageController``, a drop-in replacement for the
open-loop demand controller. The controller solves a perfect-foresight dispatch optimization with
Ipopt and provides exact derivatives through its optimality (KKT) conditions.

Part 2 uses the plant's generation profile to size the battery and the electrolyzer for the
lowest levelized cost of hydrogen, subject to a minimum electrolyzer capacity factor. Exact total
derivatives flow through the dispatch optimization, so the script first checks them against
finite differences and then compares gradient-based SLSQP with gradient-free COBYLA.

Requires the optional ``dol`` dependencies: ``pip install -e ".[dol]"`` then
``idaes get-extensions``.
"""

import time
from pathlib import Path

from sizing_problem import build_sizing_problem

from h2integrate import H2IntegrateModel, load_tech_yaml, load_plant_yaml, load_driver_yaml


# Days of the generation profile used for the sizing study. A full year (365) gives the most
# representative design and takes about 15 minutes for both optimizers; 28 days runs in about
# two minutes.
SIZING_DAYS = 28
MIN_ELECTROLYZER_CAPACITY_FACTOR = 0.6


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


def run_sizing_study(generation_MW):
    """Check total derivatives, then compare SLSQP and COBYLA on the sizing problem."""
    prob = build_sizing_problem(generation_MW, min_capacity_factor=MIN_ELECTROLYZER_CAPACITY_FACTOR)
    prob.run_model()
    print("\nTotal derivative check (exact adjoint vs central finite differences):")
    totals = prob.check_totals(out_stream=None, method="fd", step=1e-5, form="central")
    for (of, wrt), data in totals.items():
        exact = data.get("J_rev", data.get("J_fwd")).item()
        fd = data["J_fd"].item()
        print(f"  d({of})/d({wrt}): exact {exact: .6e}  fd {fd: .6e}")

    results = {}
    for optimizer in ("SLSQP", "COBYLA"):
        prob = build_sizing_problem(
            generation_MW,
            optimizer=optimizer,
            min_capacity_factor=MIN_ELECTROLYZER_CAPACITY_FACTOR,
        )
        start = time.time()
        prob.run_driver()
        lcoh = prob.model.lcoh
        results[optimizer] = {
            "LCOH": prob.get_val("LCOH", units="USD/kg").item(),
            "battery_MWh": prob.get_val("storage_capacity", units="MW*h").item(),
            "battery_MW": prob.get_val("max_charge_rate", units="MW").item(),
            "electrolyzer_MW": prob.get_val("electrolyzer_rating", units="MW").item(),
            "capacity_factor": prob.get_val("electrolyzer_capacity_factor").item(),
            "model_evaluations": lcoh.n_compute,
            "gradient_evaluations": lcoh.n_partials,
            "seconds": time.time() - start,
        }

    print(f"\nSizing study over {generation_MW.size} hours:")
    header = (
        f"{'optimizer':>9} {'LCOH':>8} {'batt MWh':>9} {'batt MW':>8} {'elec MW':>8} "
        f"{'CF':>6} {'evals':>6} {'grads':>6} {'time s':>7}"
    )
    print(header)
    for name, r in results.items():
        print(
            f"{name:>9} {r['LCOH']:8.4f} {r['battery_MWh']:9.2f} {r['battery_MW']:8.2f} "
            f"{r['electrolyzer_MW']:8.2f} {r['capacity_factor']:6.3f} {r['model_evaluations']:6d} "
            f"{r['gradient_evaluations']:6d} {r['seconds']:7.1f}"
        )
    return results


def main():
    this_dir = Path(__file__).parent
    h2i = run_plant(this_dir)

    lcoh = h2i.prob.get_val("finance_subgroup_hydrogen.LCOH", units="USD/kg")[0]
    unmet = h2i.prob.get_val("battery.dispatch_electricity_unmet_demand", units="MW").sum()
    print(f"Plant LCOH with differentiable battery dispatch: {lcoh:.3f} USD/kg")
    print(f"Battery set-point not met: {unmet:.1f} MWh over the year")

    generation = h2i.prob.get_val("gen_combiner.electricity_out", units="MW")
    run_sizing_study(generation[: 24 * SIZING_DAYS])


# The guard is required on Windows: Pyomo's PyNumero loader starts a multiprocessing worker.
if __name__ == "__main__":
    main()
