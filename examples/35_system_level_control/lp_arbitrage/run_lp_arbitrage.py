"""LP-based energy arbitrage for a merchant solar plus battery plant.

A synthetic hourly locational marginal price (LMP) series drives both the
export price and the import price. The system-level controller solves a rolling
24-hour linear program that decides, simultaneously, when to charge the battery
(from solar or from the grid), when to discharge, and how much to export.
"""

import numpy as np
import matplotlib.pyplot as plt

from h2integrate.core.h2integrate_model import H2IntegrateModel


def make_lmp_profile(n_timesteps=8760, seed=0):
    """Build a synthetic hourly LMP series in $/kWh.

    Combines a diurnal shape (an evening peak and a midday solar-driven trough
    that dips negative), a seasonal summer premium, and lognormal noise with
    occasional scarcity spikes.

    Args:
        n_timesteps (int): Number of hourly timesteps.
        seed (int): Seed for the random generator.

    Returns:
        np.ndarray: Price series of shape ``(n_timesteps,)`` in ``$/kWh``.
    """
    rng = np.random.default_rng(seed)
    hours = np.arange(n_timesteps)
    hour_of_day = hours % 24
    day_of_year = hours // 24

    # Evening ramp peak around hour 19, midday solar trough around hour 13.
    diurnal = 0.030 + 0.026 * np.sin((hour_of_day - 9) * np.pi / 12)
    midday_trough = -0.022 * np.exp(-(((hour_of_day - 13) / 2.6) ** 2))
    evening_peak = 0.030 * np.exp(-(((hour_of_day - 19) / 1.9) ** 2))

    seasonal = 0.010 * np.sin((day_of_year - 100) * 2 * np.pi / 365)
    noise = rng.lognormal(mean=0.0, sigma=0.25, size=n_timesteps) - 1.0

    price = diurnal + midday_trough + evening_peak + seasonal + 0.012 * noise

    # Scarcity spikes on a small number of hours.
    spike_hours = rng.choice(n_timesteps, size=n_timesteps // 400, replace=False)
    price[spike_hours] += rng.uniform(0.15, 0.60, size=spike_hours.size)

    return price


# -- Create and set up the model --
h2i = H2IntegrateModel("solar_battery_arbitrage.yaml")
h2i.setup()

# -- Apply the LMP series to both grid connections --
# The export price feeds the controller (input-to-input) and the revenue term of
# the grid cost model. The import price feeds the controller's marginal cost for
# grid_buy and the purchase term of the cost model. A small adder on the import
# side represents transmission and ancillary charges.
n_timesteps = 8760
lmp = make_lmp_profile(n_timesteps)
h2i.prob.set_val("grid_sell.electricity_sell_price", lmp, units="USD/(kW*h)")
h2i.prob.set_val("grid_buy.electricity_buy_price", lmp + 0.004, units="USD/(kW*h)")

h2i.run()
h2i.post_process()

# -- Extract results --
solar_out = h2i.prob.get_val("plant.solar.electricity_out", units="kW")
battery_net = h2i.prob.get_val("plant.battery.electricity_out", units="kW")
battery_discharge = h2i.prob.get_val("plant.battery.storage_electricity_discharge", units="kW")
# The storage model reports charging as a negative rate; flip it to a magnitude.
battery_charge = -h2i.prob.get_val("plant.battery.storage_electricity_charge", units="kW")
battery_soc = h2i.prob.get_val("plant.battery.SOC", units="percent")
grid_import = h2i.prob.get_val("plant.grid_buy.electricity_out", units="kW")
grid_export = h2i.prob.get_val("plant.grid_sell.electricity_sold", units="kW")

dt_h = 1.0
export_revenue = float(np.sum(grid_export * lmp) * dt_h)
import_cost = float(np.sum(grid_import * (lmp + 0.004)) * dt_h)

print(f"Annual export:          {grid_export.sum() * dt_h / 1e3:>12,.0f} MWh")
print(f"Annual import:          {grid_import.sum() * dt_h / 1e3:>12,.0f} MWh")
print(f"Battery throughput:     {battery_discharge.sum() * dt_h / 1e3:>12,.0f} MWh discharged")
print(f"Equivalent full cycles: {battery_discharge.sum() * dt_h / 200000:>12,.1f}")
print(f"Gross export revenue:   {export_revenue:>12,.0f} USD")
print(f"Gross import cost:      {import_cost:>12,.0f} USD")
print(f"Gross energy margin:    {export_revenue - import_cost:>12,.0f} USD")
charge_in_cheap_hours = battery_charge[lmp < np.quantile(lmp, 0.2)].sum() / battery_charge.sum()
discharge_in_costly_hours = (
    battery_discharge[lmp > np.quantile(lmp, 0.8)].sum() / battery_discharge.sum()
)
print(f"Charging while price < 20th pct: {100 * charge_in_cheap_hours:.1f}%")
print(f"Discharging while price > 80th pct: {100 * discharge_in_costly_hours:.1f}%")

# -- Plot a representative week --
start = 24 * 180  # mid-summer
n_hours = 168
window = slice(start, start + n_hours)
hours = np.arange(n_hours)

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

axes[0].plot(hours, lmp[window] * 100, color="tab:red")
axes[0].axhline(0.0, color="k", linewidth=0.8, linestyle=":")
axes[0].set_ylabel("LMP (\u00a2/kWh)")
axes[0].set_title("LP Arbitrage: Representative Summer Week")

axes[1].bar(
    hours, solar_out[window] / 1e3, width=1.0, color="tab:orange", align="edge", label="Solar"
)
axes[1].bar(
    hours,
    grid_import[window] / 1e3,
    width=1.0,
    bottom=solar_out[window] / 1e3,
    color="tab:gray",
    align="edge",
    label="Grid import",
)
axes[1].set_ylabel("Supply (MW)")
axes[1].legend(loc="upper right")

axes[2].bar(
    hours,
    battery_discharge[window] / 1e3,
    width=1.0,
    color="tab:green",
    align="edge",
    label="Discharge",
)
axes[2].bar(
    hours,
    -battery_charge[window] / 1e3,
    width=1.0,
    color="tab:purple",
    align="edge",
    label="Charge",
)
axes[2].axhline(0.0, color="k", linewidth=0.8)
axes[2].set_ylabel("Battery (MW)")
axes[2].legend(loc="upper right")

ax_soc = axes[2].twinx()
ax_soc.plot(hours, battery_soc[window], color="k", linewidth=1.2, label="SOC")
ax_soc.set_ylabel("SOC (%)")

axes[3].bar(hours, grid_export[window] / 1e3, width=1.0, color="tab:blue", align="edge")
axes[3].set_ylabel("Export (MW)")
axes[3].set_xlabel("Hour of week")

plt.tight_layout()
plt.savefig("lp_arbitrage_results.png", dpi=150)
print("Plot saved to lp_arbitrage_results.png")
