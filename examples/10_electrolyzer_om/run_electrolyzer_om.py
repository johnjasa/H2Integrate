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


# # ============================================================================
# # Timeseries Comparison Function and Plot
# # ============================================================================
# def plot_timeseries_comparison(
#     results, case_names_to_plot, save_filename="timeseries_comparison.png"
# ):
#     """
#     Plot timeseries comparison for specified cases showing a representative week in May.

#     Parameters:
#     -----------
#     results : dict
#         Dictionary containing all results including timeseries data
#     case_names_to_plot : list
#         List of display names to plot (e.g., ['Baseline Case', 'All of the above'])
#     save_filename : str
#         Filename to save the plot
#     """
#     print(f"\nCreating timeseries comparison plot for: {case_names_to_plot}")

#     # Find indices of cases to plot
#     case_indices = []
#     for name in case_names_to_plot:
#         try:
#             idx = results["display_name"].index(name)
#             case_indices.append(idx)
#         except ValueError:
#             print(f"Warning: Case '{name}' not found in results")

#     if not case_indices:
#         print("No valid cases found to plot")
#         return

#     # Create figure with 2 subplots
#     fig, axes = plt.subplots(2, 1, figsize=(14, 10))

#     # Define a week in December (middle of December)
#     # December starts around hour 7920 (31+28+31+30+31+30+31+31+30+31+30 = 334 days * 24 hours)
#     # Show mid-December: day 345-352 of year
#     start_hour = 345 * 24  # Day 345
#     end_hour = 352 * 24  # Day 352 (one week)
#     hours_in_week = np.arange(start_hour, end_hour)
#     days_in_week = (hours_in_week - start_hour) / 24  # 0-7 days

#     colors = plt.cm.tab10(np.linspace(0, 1, len(case_indices)))

#     for idx, case_idx in enumerate(case_indices):
#         case_name = results["display_name"][case_idx]
#         color = colors[idx]

#         # Plot 1: Wind Electricity Production
#         wind_elec = results["wind_electricity_out"][case_idx]
#         if wind_elec is not None and len(wind_elec) > 0:
#             wind_week = wind_elec[start_hour:end_hour]
#             axes[0].plot(
#                 days_in_week,
#                 wind_week / 1000,
#                 label=case_name,
#                 color=color,
#                 alpha=0.7,
#                 linewidth=1.5,
#             )

#         # Plot 2: Hydrogen Production
#         h2_prod = results["electrolyzer_hydrogen_out"][case_idx]
#         if h2_prod is not None and len(h2_prod) > 0:
#             h2_week = h2_prod[start_hour:end_hour]
#             axes[1].plot(
#                 days_in_week, h2_week, label=case_name, color=color, alpha=0.7, linewidth=1.5
#             )

#     # Configure subplot 1: Wind Electricity
#     axes[0].set_ylabel("Wind Power (MW)", fontsize=12)
#     axes[0].set_title("Wind Electricity Production", fontsize=13)
#     axes[0].legend(loc="best", fontsize=10)
#     axes[0].set_xlim(0, 7)
#     axes[0].set_xticks(range(8))
#     axes[0].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "Mon"])
#     axes[0].grid(True, alpha=0.3, linestyle="--")

#     # Configure subplot 2: Hydrogen Production
#     axes[1].set_ylabel("H₂ Production (kg/h)", fontsize=12)
#     axes[1].set_xlabel("Day of Week", fontsize=12)
#     axes[1].set_title("Hydrogen Production Rate", fontsize=13)
#     axes[1].legend(loc="best", fontsize=10)
#     axes[1].set_xlim(0, 7)
#     axes[1].set_xticks(range(8))
#     axes[1].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "Mon"])
#     axes[1].grid(True, alpha=0.3, linestyle="--")

#     plt.tight_layout()
#     plt.savefig(save_filename, dpi=400, bbox_inches="tight")
#     print(f"Timeseries plot saved as '{save_filename}'")

#     return fig


# # Create timeseries plot for Baseline and "All of the above" cases
# if "wind_electricity_out" in results and results["wind_electricity_out"][0] is not None:
#     plot_timeseries_comparison(results, ["Baseline Case", "All of the above"])
# else:
#     print(
#         "\nNote: Timeseries data not available. "
#         "Set RUN_SIMULATIONS=True to generate timeseries data."
#     )

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

    fig_breakdown, ax_breakdown = plt.subplots(figsize=(16, 10))

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
            "Taxes": "Taxes",
            "Finances": "Finances",
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

            # Add all components except Total
            components = {k: v for k, v in breakdown.items() if "total" not in k.lower()}
            all_components.update(components.keys())
        else:
            total_values.append(0.0)

    # Sort components for consistent ordering
    component_list = sorted(all_components)

    # Create matrix of cost contributions (excluding Total)
    cost_matrix = []
    for breakdown in results["LCOH_breakdown"]:
        if breakdown is not None:
            # Filter out Total entry
            components = {k: v for k, v in breakdown.items() if "total" not in k.lower()}
            row = [components.get(comp, 0.0) for comp in component_list]
        else:
            row = [0.0] * len(component_list)
        cost_matrix.append(row)

    cost_matrix = np.array(cost_matrix).T  # Transpose for stacking

    # Create stacked bar chart
    colors = plt.cm.tab20(np.linspace(0, 1, len(component_list)))
    bottom = np.zeros(len(display_names))

    # Store small components for rightmost bar labeling
    small_components_rightmost = []

    for idx, (component, color) in enumerate(zip(component_list, colors)):
        values = cost_matrix[idx]
        bars = ax_breakdown.bar(
            x_positions, values, bottom=bottom, color=color, alpha=0.8, width=0.8
        )

        # Get clean component name
        clean_name = clean_component_name(component)

        # Add text labels for each component segment (only if value is significant)
        for i, (x, val) in enumerate(zip(x_positions, values)):
            if val > 0.5:  # Only label if contribution is > $0.50/kg
                y_pos = bottom[i] + val / 2
                # Two lines: component name on top, dollar amount below
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
            elif (
                i == len(x_positions) - 1 and val > 0.01
            ):  # For rightmost bar, collect small components
                y_pos = bottom[i] + val / 2
                small_components_rightmost.append(
                    {"name": clean_name, "value": val, "y_pos": y_pos, "color": color}
                )

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
