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

        except (ValueError, KeyError, RuntimeError) as e:
            print(f"ERROR running {config_file}: {e}")
            # Append NaN values for failed runs
            results["config_name"].append(config_file.replace(".yaml", ""))
            results["display_name"].append(CASE_MAPPING[config_file])
            for key in results:
                if key not in ["config_name", "display_name"]:
                    if "timeseries" in key or key in [
                        "wind_electricity_out",
                        "electrolyzer_hydrogen_out",
                        "LCOH_breakdown",
                    ]:
                        results[key].append(None)
                    else:
                        results[key].append(np.nan)

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
print("Creating comparison plots...")
print(f"{'='*60}")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Use display names for plotting
display_names = results["display_name"]

# Get baseline values (first case)
baseline_lcoh = results["LCOH"][0]
baseline_cf = results["capacity_factor"][0]
baseline_avail = results["electrolyzer_availability"][0]
baseline_h2_lost = results["percent_hydrogen_lost"][0]
baseline_opex = results["OpEx"][0]
baseline_h2_prod = results["total_hydrogen_produced"][0]

# Create x-positions with spacing between groups
# Groups: baseline(0) | +25%(1,2,3) | -25%(4,5,6) | Biannual(7) | remaining(8,9,10,11)
gap = 0.5  # Gap size between groups
x_positions = np.array(
    [
        0,  # Baseline
        1 + gap,
        2 + gap,
        3 + gap,  # +25% group
        4 + 2 * gap,
        5 + 2 * gap,
        6 + 2 * gap,  # -25% group
        7 + 3 * gap,  # Biannual DI
        8 + 4 * gap,
        9 + 4 * gap,
        10 + 4 * gap,
        11 + 4 * gap,  # Extreme cases
    ]
)

# Define vertical line positions for separators (between groups)
separator_positions = [0.5 + gap / 2, 3.5 + 1.5 * gap, 6.5 + 2.5 * gap, 7.5 + 3.5 * gap]

# Plot 1: LCOH comparison
ax = axes[0, 0]
ax.bar(x_positions, results["LCOH"], alpha=0.8)
ax.axhline(y=baseline_lcoh, color="gray", linestyle="--", linewidth=1.5, zorder=0)
for pos in separator_positions:
    ax.axvline(x=pos, color="gray", linestyle="-", linewidth=1, alpha=0.5)

ax.set_ylabel("LCOH ($/kg)")
ax.set_title("Levelized Cost of Hydrogen")
ax.set_xticks(x_positions)
ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=8)

# Plot 2: Hydrogen Lost
ax = axes[0, 1]
ax.bar(x_positions, results["percent_hydrogen_lost"], color="red", alpha=0.7)
ax.axhline(y=baseline_h2_lost, color="gray", linestyle="--", linewidth=1.5, zorder=0)
for pos in separator_positions:
    ax.axvline(x=pos, color="gray", linestyle="-", linewidth=1, alpha=0.5)
ax.set_ylabel("Hydrogen Lost (%)")
ax.set_title("Percent Hydrogen Lost to O&M")
ax.set_xticks(x_positions)
ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=8)

# Plot 3: OpEx
ax = axes[1, 0]
ax.bar(x_positions, np.array(results["OpEx"]) / 1e6, color="orange", alpha=0.7)
ax.axhline(y=baseline_opex / 1e6, color="gray", linestyle="--", linewidth=1.5, zorder=0)
for pos in separator_positions:
    ax.axvline(x=pos, color="gray", linestyle="-", linewidth=1, alpha=0.5)
ax.set_ylabel("OpEx (Million $/year)")
ax.set_title("Annual Operational Expenditure")
ax.set_xticks(x_positions)
ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=8)

# Plot 4: Total H2 Produced
ax = axes[1, 1]
ax.bar(x_positions, np.array(results["total_hydrogen_produced"]) / 1e3, color="purple", alpha=0.7)
ax.axhline(y=baseline_h2_prod / 1e3, color="gray", linestyle="--", linewidth=1.5, zorder=0)
for pos in separator_positions:
    ax.axvline(x=pos, color="gray", linestyle="-", linewidth=1, alpha=0.5)
ax.set_ylabel("Total H2 Produced (tonnes/year)")
ax.set_title("Annual Hydrogen Production")
ax.set_xticks(x_positions)
ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=8)

plt.tight_layout()
plt.savefig("wombat_config_comparison.png", dpi=400, bbox_inches="tight")
print("\nPlot saved as 'wombat_config_comparison.png'")


# ============================================================================
# Timeseries Comparison Function and Plot
# ============================================================================
def plot_timeseries_comparison(
    results, case_names_to_plot, save_filename="timeseries_comparison.png"
):
    """
    Plot timeseries comparison for specified cases showing a representative week in May.

    Parameters:
    -----------
    results : dict
        Dictionary containing all results including timeseries data
    case_names_to_plot : list
        List of display names to plot (e.g., ['Baseline Case', 'All of the above'])
    save_filename : str
        Filename to save the plot
    """
    print(f"\nCreating timeseries comparison plot for: {case_names_to_plot}")

    # Find indices of cases to plot
    case_indices = []
    for name in case_names_to_plot:
        try:
            idx = results["display_name"].index(name)
            case_indices.append(idx)
        except ValueError:
            print(f"Warning: Case '{name}' not found in results")

    if not case_indices:
        print("No valid cases found to plot")
        return

    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Define a week in December (middle of December)
    # December starts around hour 7920 (31+28+31+30+31+30+31+31+30+31+30 = 334 days * 24 hours)
    # Show mid-December: day 345-352 of year
    start_hour = 345 * 24  # Day 345
    end_hour = 352 * 24  # Day 352 (one week)
    hours_in_week = np.arange(start_hour, end_hour)
    days_in_week = (hours_in_week - start_hour) / 24  # 0-7 days

    colors = plt.cm.tab10(np.linspace(0, 1, len(case_indices)))

    for idx, case_idx in enumerate(case_indices):
        case_name = results["display_name"][case_idx]
        color = colors[idx]

        # Plot 1: Wind Electricity Production
        wind_elec = results["wind_electricity_out"][case_idx]
        if wind_elec is not None and len(wind_elec) > 0:
            wind_week = wind_elec[start_hour:end_hour]
            axes[0].plot(
                days_in_week,
                wind_week / 1000,
                label=case_name,
                color=color,
                alpha=0.7,
                linewidth=1.5,
            )

        # Plot 2: Hydrogen Production
        h2_prod = results["electrolyzer_hydrogen_out"][case_idx]
        if h2_prod is not None and len(h2_prod) > 0:
            h2_week = h2_prod[start_hour:end_hour]
            axes[1].plot(
                days_in_week, h2_week, label=case_name, color=color, alpha=0.7, linewidth=1.5
            )

    # Configure subplot 1: Wind Electricity
    axes[0].set_ylabel("Wind Power (MW)", fontsize=12)
    axes[0].set_title("Wind Electricity Production", fontsize=13)
    axes[0].legend(loc="best", fontsize=10)
    axes[0].set_xlim(0, 7)
    axes[0].set_xticks(range(8))
    axes[0].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "Mon"])
    axes[0].grid(True, alpha=0.3, linestyle="--")

    # Configure subplot 2: Hydrogen Production
    axes[1].set_ylabel("H₂ Production (kg/h)", fontsize=12)
    axes[1].set_xlabel("Day of Week", fontsize=12)
    axes[1].set_title("Hydrogen Production Rate", fontsize=13)
    axes[1].legend(loc="best", fontsize=10)
    axes[1].set_xlim(0, 7)
    axes[1].set_xticks(range(8))
    axes[1].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "Mon"])
    axes[1].grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(save_filename, dpi=400, bbox_inches="tight")
    print(f"Timeseries plot saved as '{save_filename}'")

    return fig


# Create timeseries plot for Baseline and "All of the above" cases
if "wind_electricity_out" in results and results["wind_electricity_out"][0] is not None:
    plot_timeseries_comparison(results, ["Baseline Case", "All of the above"])
else:
    print(
        "\nNote: Timeseries data not available. "
        "Set RUN_SIMULATIONS=True to generate timeseries data."
    )

# ============================================================================
# LCOH Breakdown Stacked Bar Chart
# ============================================================================
if "LCOH_breakdown" in results and results["LCOH_breakdown"][0] is not None:
    print("\nCreating LCOH breakdown stacked bar chart...")

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

        bottom += values

    # Add total LCOH text above each bar
    for x, total in zip(x_positions, total_values):
        ax_breakdown.text(x, total + 0.2, f"${total:.2f}", ha="center", va="bottom", fontsize=13)

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
