import pickle
from pathlib import Path

import yaml
import numpy as np
import matplotlib.pyplot as plt

from h2integrate.core.h2integrate_model import H2IntegrateModel


# Boolean to control whether to run simulations or just load saved data
RUN_SIMULATIONS = False  # Set to False to just plot using saved data

# Mapping from file names to nice display names (in desired order)
CASE_MAPPING = {
    "base.yaml": "Baseline Case",
    "increase_time.yaml": "+25% Repair Time",
    "increase_materials.yaml": "+25% Repair Cost",
    "increase_crew.yaml": "+25% Crew Costs",
    "decrease_time.yaml": "-25% Repair Time",
    "decrease_materials.yaml": "-25% Repair Cost",
    "decrease_crew.yaml": "-25% Crew Costs",
    "base_with_DI.yaml": "Biannual DI Water Failure",
    "base_with_DI_doubled.yaml": "+200% Failure Frequency",
    "base_with_DI_increase_time.yaml": "+400% Repair Time",
    "base_with_DI_increase_lead.yaml": "+350% Technician Lead Time",
    "base_with_DI_doubled_increase_time_increase_lead.yaml": "All of the above",
}

# Find all WOMBAT config files in the library
wombat_library_path = (
    Path(__file__).parent.parent.parent / "resource_files" / "wombat_library" / "project" / "config"
)

# Get only the files specified in CASE_MAPPING (preserving order) and exclude electrolyzer_test
config_files = [f for f in CASE_MAPPING.keys() if (wombat_library_path / f).exists()]

print(f"Found {len(config_files)} WOMBAT config files to process:")
for cf in config_files:
    print(f"  - {cf} -> {CASE_MAPPING[cf]}")

# Storage for results
results = {
    "config_name": [],
    "display_name": [],
    "LCOH": [],  # LCOH from hydrogen finance group
    "LCOH_breakdown": [],  # LCOH breakdown dictionary
    "capacity_factor": [],
    "electrolyzer_availability": [],
    "percent_hydrogen_lost": [],
    "OpEx": [],
    "CapEx": [],
    "total_hydrogen_produced": [],
    # Timeseries data
    "wind_electricity_out": [],  # kW timeseries
    "electrolyzer_availability_timeseries": [],  # availability timeseries
    "electrolyzer_hydrogen_out": [],  # kg/h timeseries
}

# Path to save/load results
results_file = Path(__file__).parent / "electrolyzer_om_results.pkl"

if RUN_SIMULATIONS:
    # Loop through each config file
    for config_file in config_files:
        print(f"\n{'='*60}")
        print(f"Running simulation with: {config_file}")
        print(f"Display name: {CASE_MAPPING[config_file]}")
        print(f"{'='*60}")

        # Load the tech config and modify the filepath
        with Path("tech_config.yaml").open() as f:
            tech_config = yaml.safe_load(f)

        # Update the WOMBAT filepath
        tech_config["technologies"]["electrolyzer"]["model_inputs"]["shared_parameters"][
            "filepath"
        ] = f"project/config/{config_file}"

        # Write the modified tech config to a temporary file
        temp_tech_config = Path("tech_config_temp.yaml")
        with temp_tech_config.open("w") as f:
            yaml.dump(tech_config, f)

        # Create a temporary main config that points to the temp tech config
        with Path("electrolyzer_om.yaml").open() as f:
            main_config = yaml.safe_load(f)

        main_config["technology_config"] = str(temp_tech_config)
        temp_main_config = Path("electrolyzer_om_temp.yaml")
        with temp_main_config.open("w") as f:
            yaml.dump(main_config, f)

        try:
            # Create and run the model
            h2i = H2IntegrateModel(temp_main_config)
            h2i.run()

            # Extract results
            results["config_name"].append(config_file.replace(".yaml", ""))
            results["display_name"].append(CASE_MAPPING[config_file])

            LCOH = h2i.prob.get_val(
                "finance_subgroup_hydrogen.price_hydrogen_lcoh_financials", units="USD/kg"
            )[0]
            results["LCOH"].append(LCOH)

            # Get LCOH breakdown
            LCOH_breakdown = h2i.prob.get_val(
                "finance_subgroup_hydrogen.LCOH_lcoh_financials_breakdown"
            )
            print("LCOH Breakdown:")
            for key, value in LCOH_breakdown.items():
                print(f"  {key}: ${value:.2f}/kg")
            results["LCOH_breakdown"].append(LCOH_breakdown)

            # Get electrolyzer performance metrics
            results["capacity_factor"].append(h2i.prob.get_val("electrolyzer.capacity_factor")[0])
            results["electrolyzer_availability"].append(
                h2i.prob.get_val("electrolyzer.electrolyzer_availability")[0]
            )
            results["percent_hydrogen_lost"].append(
                h2i.prob.get_val("electrolyzer.percent_hydrogen_lost")[0]
            )
            results["OpEx"].append(h2i.prob.get_val("electrolyzer.OpEx", units="USD/year")[0])
            results["CapEx"].append(h2i.prob.get_val("electrolyzer.CapEx", units="USD")[0])
            results["total_hydrogen_produced"].append(
                h2i.prob.get_val("electrolyzer.total_hydrogen_produced", units="kg/year")[0]
            )

            # Get timeseries data
            results["wind_electricity_out"].append(
                h2i.prob.get_val("wind.electricity_out", units="kW")
            )
            results["electrolyzer_hydrogen_out"].append(
                h2i.prob.get_val("electrolyzer.hydrogen_out", units="kg/h")
            )

            print(f"\nResults for {CASE_MAPPING[config_file]}:")
            print(f"  LCOH (from LCOH group): ${results['LCOH'][-1]:.3f}/kg")
            print(f"  Capacity Factor: {results['capacity_factor'][-1]:.2%}")
            print(f"  Availability: {results['electrolyzer_availability'][-1]:.2%}")
            print(f"  H2 Lost: {results['percent_hydrogen_lost'][-1]:.2f}%")

        finally:
            # Clean up temp files
            if temp_tech_config.exists():
                temp_tech_config.unlink()
            if temp_main_config.exists():
                temp_main_config.unlink()

    # Save results to pickle file
    with results_file.open("wb") as f:
        pickle.dump(results, f)
    print(f"\n{'='*60}")
    print(f"Results saved to {results_file}")
    print(f"{'='*60}")

else:
    # Load results from pickle file
    print(f"\n{'='*60}")
    print(f"Loading saved results from {results_file}")
    print(f"{'='*60}")

    if not results_file.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_file}. Please run with RUN_SIMULATIONS=True first."
        )

    with results_file.open("rb") as f:
        results = pickle.load(f)

    print(f"Loaded {len(results['config_name'])} results")

# Create comparison plots
print(f"\n{'='*60}")
print("Creating tornado sensitivity plots (±25% cases)...")
print(f"{'='*60}")

# Use display names for plotting
display_names = results["display_name"]

# Baseline metrics (assume first entry is baseline)
baseline_lcoh = results["LCOH"][0]
baseline_h2_lost = results["percent_hydrogen_lost"][0]
baseline_opex = results["OpEx"][0]
baseline_h2_prod = results["total_hydrogen_produced"][0]


# Helper: get value by config_name safely
def get_val(list_name, cfg_name, transform=lambda x: x):
    try:
        idx = results["config_name"].index(cfg_name)
        return transform(results[list_name][idx])
    except ValueError:
        return None


# Parameter pairs for ±25% scenarios
param_pairs = [
    ("increase_time", "decrease_time", "Repair Time ±25%"),
    ("increase_materials", "decrease_materials", "Repair Cost ±25%"),
    ("increase_crew", "decrease_crew", "Crew Costs ±25%"),
]

# Prepare figure: 1 row x 3 columns (adjusted figure size)
fig_tornado, axes_tornado = plt.subplots(1, 3, figsize=(14, 4))
plt.subplots_adjust(wspace=0.15)  # Reduce space between subplots


def tornado_subplot(
    ax, metric_list_name, baseline_value, title, value_transform=lambda x: x, unit_fmt=None
):
    labels = []
    inc_vals = []
    dec_vals = []
    for inc_key, dec_key, label in param_pairs:
        labels.append(label)
        inc_v = get_val(metric_list_name, inc_key, value_transform)
        dec_v = get_val(metric_list_name, dec_key, value_transform)
        # Handle no-effect crew cost case by adding tiny epsilon so bars render distinctly
        if (
            label == "Crew Costs ±25%" or label == "Repair Cost ±25%"
        ) and title == "% reduction in H2 produced due to O&M (downtime)":
            eps = (abs(baseline_value) * 1e-6) + 1e-6
            inc_plot_v = baseline_value + eps
            dec_plot_v = baseline_value - eps
        else:
            inc_plot_v = inc_v if inc_v is not None else baseline_value
            dec_plot_v = dec_v if dec_v is not None else baseline_value
        inc_vals.append(inc_plot_v)
        dec_vals.append(dec_plot_v)

    y_pos = np.arange(len(labels))

    # Single muted blue color for both directions
    bar_color = "#5b7aa3"  # muted desaturated blue

    # For tornado: draw bars extending from baseline to scenario value
    # Increase (right): start at baseline, width = inc - baseline
    for y, inc in zip(y_pos, inc_vals):
        if inc is not None and inc != baseline_value:
            width = inc - baseline_value
            ax.barh(y, width, left=baseline_value, color=bar_color, alpha=0.85)
    # Decrease (left): start at dec value, width = baseline - dec
    for y, dec in zip(y_pos, dec_vals):
        if dec is not None and dec != baseline_value:
            width = baseline_value - dec
            ax.barh(y, width, left=dec, color=bar_color, alpha=0.85)

    # Baseline line (gray color)
    ax.axvline(baseline_value, color="gray", linewidth=1.0)

    # Annotate values slightly outside bar ends; baseline label offset upward
    # Determine offset after setting limits

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=13)
    ax.set_title(title, fontsize=13)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    # Auto x-limits with margin
    all_vals = [v for pair in zip(inc_vals, dec_vals) for v in pair if v is not None]
    min_v = min([*all_vals, baseline_value])
    max_v = max([*all_vals, baseline_value])
    span = max_v - min_v if max_v > min_v else max_v * 0.1 + 1e-6
    ax.set_xlim(min_v - 0.05 * span, max_v + 0.05 * span)
    # Improve x-axis tick label formatting
    ax.tick_params(axis="x", labelsize=11)
    # Recompute offset now that limits fixed
    offset = 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]

    for y, inc, dec, label in zip(y_pos, inc_vals, dec_vals, labels):
        # Check if there's no effect for this parameter or specific no-effect cases
        no_effect = abs(inc - baseline_value) < 1e-5 and abs(dec - baseline_value) < 1e-5

        if label == "Repair Cost ±25%" and metric_list_name == "percent_hydrogen_lost":
            no_effect = True  # Force no-effect for crew costs in % reduction plot

        if no_effect:
            # For crew costs and % reduction cases, show "No effect" on the left side
            ax.text(
                baseline_value - offset * 15,
                y,
                "No effect",
                va="center",
                ha="right",
                fontsize=11,
                color="#333333",
            )
            # Put baseline value closer to center
            ax.text(
                baseline_value - offset * 1.5,
                y,
                f"{baseline_value:.2f}" if unit_fmt is None else unit_fmt(baseline_value),
                va="center",
                ha="right",
                fontsize=11,
                color="black",
            )
        else:
            # Place value labels, checking if they're at plot edges
            if inc != baseline_value:
                # Check if label would be at right edge of plot
                if (inc + offset) > (ax.get_xlim()[0] + 0.95 * x_range):
                    # Place inside bar
                    ax.text(
                        inc - offset,
                        y,
                        f"{inc:.2f}" if unit_fmt is None else unit_fmt(inc),
                        va="center",
                        ha="right",
                        fontsize=11,
                    )
                else:
                    # Place outside bar
                    ax.text(
                        inc + offset,
                        y,
                        f"{inc:.2f}" if unit_fmt is None else unit_fmt(inc),
                        va="center",
                        ha="left",
                        fontsize=11,
                    )

            if dec != baseline_value:
                # Check if label would be at left edge of plot
                if (dec - offset) < (ax.get_xlim()[0] + 0.05 * x_range):
                    # Place inside bar
                    ax.text(
                        dec + offset,
                        y,
                        f"{dec:.2f}" if unit_fmt is None else unit_fmt(dec),
                        va="center",
                        ha="left",
                        fontsize=11,
                    )
                else:
                    # Place outside bar
                    ax.text(
                        dec - offset,
                        y,
                        f"{dec:.2f}" if unit_fmt is None else unit_fmt(dec),
                        va="center",
                        ha="right",
                        fontsize=11,
                    )

            # Baseline label closer to center line
            ax.text(
                baseline_value - offset * 1.5,
                y,
                f"{baseline_value:.2f}" if unit_fmt is None else unit_fmt(baseline_value),
                va="center",
                ha="right",
                fontsize=11,
                color="black",
            )
    # Add annotation clarifying ±25% input variation - removed per request


# LCOH
tornado_subplot(axes_tornado[0], "LCOH", baseline_lcoh, "LCOH ($/kg)")
# % Reduction (original metric was percent_hydrogen_lost)
tornado_subplot(
    axes_tornado[1],
    "percent_hydrogen_lost",
    baseline_h2_lost,
    "% reduction in H2 produced due to O&M (downtime)",
)
# OpEx (convert to thousand $/yr)
tornado_subplot(
    axes_tornado[2],
    "OpEx",
    baseline_opex / 1e3,
    "OpEx (Thousand $/yr)",
    value_transform=lambda x: x / 1e3,
)

# Remove y-axis labels for the right two plots
axes_tornado[1].set_yticklabels([])
axes_tornado[2].set_yticklabels([])

# (Removed overall figure title per request)
fig_tornado.tight_layout(rect=[0, 0.03, 1, 0.95])
fig_tornado.savefig("tornado_sensitivity_25pct.png", dpi=400, bbox_inches="tight")
print("Tornado sensitivity figure saved as 'tornado_sensitivity_25pct.png'")

# ----------------------------------------------------------------------------
# Separate figure for DI water failure scenarios (include baseline for context)
# ----------------------------------------------------------------------------
print(f"\n{'='*60}")
print("Creating DI water failure scenario comparison plots...")
print(f"{'='*60}")

di_case_names = [
    "base",  # baseline
    "base_with_DI",
    "base_with_DI_doubled",
    "base_with_DI_increase_time",
    "base_with_DI_increase_lead",
    "base_with_DI_doubled_increase_time_increase_lead",
]

di_indices = []
di_display = []
for cname in di_case_names:
    if cname in results["config_name"]:
        idx = results["config_name"].index(cname)
        di_indices.append(idx)
        di_display.append(results["display_name"][idx])

if len(di_indices) >= 2:
    fig_di, axes_di = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(di_indices))

    # Extract series for DI subset (removed h2prod_vals)
    lcoh_vals = [results["LCOH"][i] for i in di_indices]
    h2lost_vals = [results["percent_hydrogen_lost"][i] for i in di_indices]
    opex_vals = [results["OpEx"][i] / 1e3 for i in di_indices]

    def bar_subplot(ax, values, title, ylabel):
        bars = ax.bar(x, values, color="#5b7aa3", alpha=0.85)  # Same muted blue as tornado plot
        ax.set_xticks(x)
        ax.set_xticklabels(di_display, rotation=45, ha="right", fontsize=9)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=10)
        # Put values inside the bars
        for xi, val, bar in zip(x, values, bars):
            bar_height = bar.get_height()
            # Place value label in axes coordinates (centered horizontally, near top of bar)
            ax.text(
                xi,
                bar_height - 0.02 * ax.get_ylim()[1],
                f"{val:.2f}",
                ha="center",
                va="top",
                fontsize=10,
                color="white",
            )
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    bar_subplot(axes_di[0], lcoh_vals, "LCOH", "$/kg")
    bar_subplot(axes_di[1], h2lost_vals, "% reduction in H2 produced due to O&M (downtime)", "%")
    bar_subplot(axes_di[2], opex_vals, "OpEx", "Thousand $/yr")

    fig_di.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_di.savefig("di_water_failure_cases.png", dpi=400, bbox_inches="tight")
    print("DI water failure scenario figure saved as 'di_water_failure_cases.png'")
else:
    print("DI water failure scenarios not found in results or insufficient cases to plot.")

# ============================================================================
# LCOH Breakdown Stacked Bar Chart
# ============================================================================
if "LCOH_breakdown" in results and results["LCOH_breakdown"][0] is not None:
    print("\nCreating LCOH breakdown stacked bar chart...")

    # Define positions for stacked breakdown using simple sequential spacing
    # Groups: Baseline | +25% (3) | -25% (3) | DI base | DI variants (4)
    display_names = results["display_name"]  # ensure available
    x_positions = np.arange(len(display_names))
    separator_positions = [0.5, 3.5, 6.5, 7.5]

    fig_breakdown, ax_breakdown = plt.subplots(figsize=(18, 10))

    # Function to clean up component names
    def clean_component_name(name):
        """Remove 'LCOH: ' prefix and ' ($/kg)' suffix, and create nice names"""
        # Remove prefixes and suffixes
        clean = name.replace("LCOH: ", "").replace(" ($/kg)", "")

        # Map to nice names
        name_mapping = {
            "wind CapEx": "Wind CapEx",
            "electrolyzer CapEx": "H2 CapEx",
            "wind OpEx": "Wind OpEx",
            "electrolyzer OpEx": "H2 OpEx",
            "battery CapEx": "Batt. CapEx",
            "battery OpEx": "Batt. OpEx",
            "solar CapEx": "Solar CapEx",
            "solar OpEx": "Solar OpEx",
            "Taxes": "Taxes & Finances",
            "Finances": "Taxes & Finances",
        }

        return name_mapping.get(clean, clean)

    # Extract all unique cost components across all cases, excluding "Total"
    all_components = set()
    total_values = []  # Store total values separately

    for breakdown in results["LCOH_breakdown"]:
        if breakdown is not None:
            # Extract total if it exists
            total_key = [k for k in breakdown.keys() if "total" in k.lower()]
            if total_key:
                total_values.append(breakdown[total_key[0]])
            else:
                total_values.append(sum(breakdown.values()))

            # Add all components except Total, and combine Taxes and Finances
            components = {}
            taxes_finances_sum = 0.0
            for k, v in breakdown.items():
                if "total" not in k.lower():
                    clean_k = k.replace("LCOH: ", "").replace(" ($/kg)", "")
                    if clean_k in ["Taxes", "Finances"]:
                        taxes_finances_sum += v
                    else:
                        components[k] = v

            # Add combined Taxes & Finances if non-zero
            if taxes_finances_sum > 0:
                components["LCOH: Taxes ($/kg)"] = (
                    taxes_finances_sum  # Use one key as representative
                )

            all_components.update(components.keys())
        else:
            total_values.append(0.0)

    # Sort components for consistent ordering
    component_list = sorted(all_components)

    # Create matrix of cost contributions (excluding Total)
    cost_matrix = []
    for breakdown in results["LCOH_breakdown"]:
        if breakdown is not None:
            # Filter out Total entry and combine Taxes and Finances
            components = {}
            taxes_finances_sum = 0.0
            for k, v in breakdown.items():
                if "total" not in k.lower():
                    clean_k = k.replace("LCOH: ", "").replace(" ($/kg)", "")
                    if clean_k in ["Taxes", "Finances"]:
                        taxes_finances_sum += v
                    else:
                        components[k] = v

            # Add combined Taxes & Finances if non-zero
            if taxes_finances_sum > 0:
                components["LCOH: Taxes ($/kg)"] = taxes_finances_sum

            row = [components.get(comp, 0.0) for comp in component_list]
        else:
            row = [0.0] * len(component_list)
        cost_matrix.append(row)

    cost_matrix = np.array(cost_matrix).T  # Transpose for stacking

    # Define color mapping by technology (using color families)
    def get_component_color(component_name):
        """Assign colors based on technology with CapEx/OpEx hue variations"""
        clean = component_name.replace("LCOH: ", "").replace(" ($/kg)", "")

        # Color families: darker for CapEx, lighter for OpEx
        if "wind" in clean.lower():
            return "#2d7a2d" if "CapEx" in clean else "#66b366"  # Dark green / Light green
        elif "electrolyzer" in clean.lower() or "h2" in clean.lower():
            return "#1f5fa8" if "CapEx" in clean else "#6ba3d4"  # Dark blue / Light blue
        elif "battery" in clean.lower():
            return "#b8860b" if "CapEx" in clean else "#daa520"  # Dark goldenrod / Goldenrod
        elif "solar" in clean.lower():
            return "#d97700" if "CapEx" in clean else "#ff9933"  # Dark orange / Light orange
        elif "taxes" in clean.lower() or "finances" in clean.lower():
            return "#8b4789"  # Purple for combined Taxes & Finances
        else:
            return "#808080"  # Gray for unknown

    # Create color list for each component
    colors = [get_component_color(comp) for comp in component_list]
    bottom = np.zeros(len(display_names))

    # Store small components for rightmost bar labeling
    small_components_rightmost = []

    for idx, (component, color) in enumerate(zip(component_list, colors)):
        values = cost_matrix[idx]
        bars = ax_breakdown.bar(
            x_positions,
            values,
            bottom=bottom,
            color=color,
            alpha=0.8,
            width=0.8,
            label=clean_component_name(component),
        )

        # Get clean component name
        clean_name = clean_component_name(component)

        # Add text labels for each component segment based on value thresholds
        for i, (x, val) in enumerate(zip(x_positions, values)):
            y_pos = bottom[i] + val / 2

            if val >= 0.25:
                # Value >= 0.25: show component name and dollar value
                label_text = f"{clean_name}\n${val:.2f}"
                ax_breakdown.text(
                    x,
                    y_pos,
                    label_text,
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="normal",
                    color="black",
                )
            elif val >= 0.05:
                # 0.05 <= value < 0.25: show only dollar value
                label_text = f"${val:.2f}"
                ax_breakdown.text(
                    x,
                    y_pos,
                    label_text,
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="normal",
                    color="black",
                )
            # else: value < 0.05, no label

        bottom += values

    # Add total LCOH text above each bar
    for x, total in zip(x_positions, total_values):
        ax_breakdown.text(x, total + 0.1, f"${total:.2f}", ha="center", va="bottom", fontsize=13)

    # Add vertical separators
    for pos in separator_positions:
        ax_breakdown.axvline(x=pos, color="gray", linestyle="-", linewidth=1, alpha=0.5)

    ax_breakdown.set_ylabel("LCOH ($/kg)", fontsize=14)
    ax_breakdown.set_xlabel("Case", fontsize=14)
    ax_breakdown.set_xticks(x_positions)
    ax_breakdown.set_xticklabels(display_names, rotation=45, ha="right", fontsize=11)
    ax_breakdown.tick_params(axis="both", labelsize=12)
    ax_breakdown.set_ylim(top=max(total_values) * 1.12)  # Add space for total labels

    # Add legend to the right side of the plot (reversed order)
    handles, labels = ax_breakdown.get_legend_handles_labels()
    ax_breakdown.legend(
        handles[::-1],
        labels[::-1],  # Reverse the order
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=11,
        frameon=True,
        framealpha=0.9,
    )

    plt.tight_layout()
    plt.savefig("lcoh_breakdown_stacked.png", dpi=400, bbox_inches="tight")
    print("LCOH breakdown stacked bar chart saved as 'lcoh_breakdown_stacked.png'")
else:
    print(
        "\nNote: LCOH breakdown not available. Set RUN_SIMULATIONS=True to generate breakdown data."
    )

plt.show()

print("\n" + "=" * 60)
print("All simulations complete!")
print("=" * 60)
