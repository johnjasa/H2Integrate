import matplotlib.pyplot as plt

from h2integrate.core.h2integrate_model import H2IntegrateModel


# Create an H2I model
h2i = H2IntegrateModel("20_run_of_river.yaml")

# Run the model
h2i.run()

# Post-process the results
h2i.post_process()


# Battery dispatch plotting
model = h2i
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 5))

start_hour = 2300
end_hour = 2600
total_time_steps = model.prob.get_val("battery.electricity_soc").size
demand_profile = model.prob.get_val("battery.electricity_demand_profile", units="MW/h")

ax[0].plot(
    range(start_hour, end_hour),
    model.prob.get_val("battery.electricity_soc", units="percent")[start_hour:end_hour],
    label="SOC",
    linewidth=2,
)
ax[0].set_ylabel("State-of-charge for PSH (%)", fontsize=12)
ax[0].set_ylim([0, 110])

ax[1].plot(
    range(start_hour, end_hour),
    model.prob.get_val("battery.electricity_in", units="MW")[start_hour:end_hour],
    linestyle="-",
    label="Electricity In (MW)",
    linewidth=2,
)
ax[1].plot(
    range(start_hour, end_hour),
    model.prob.get_val("battery.electricity_excess_resource", units="MW")[start_hour:end_hour],
    linestyle=":",
    label="Excess Electricity Resource (MW)",
    linewidth=2,
)
ax[1].plot(
    range(start_hour, end_hour),
    model.prob.get_val("battery.electricity_unmet_demand", units="MW")[start_hour:end_hour],
    linestyle=":",
    label="Electricity Unmet Demand (MW)",
    linewidth=2,
)
ax[1].plot(
    range(start_hour, end_hour),
    model.prob.get_val("battery.electricity_out", units="MW")[start_hour:end_hour],
    linestyle="-",
    label="Electricity Out (MW)",
    linewidth=2,
)
ax[1].plot(
    range(start_hour, end_hour),
    demand_profile[start_hour:end_hour],
    linestyle="--",
    label="Electricity Demand (MW)",
    linewidth=2,
)
ax[1].set_ylabel("Electricity Hourly (MW)", fontsize=12)
ax[1].set_xlabel("Timestep (hr)", fontsize=12)

# Increase font size for tick labels
ax[0].tick_params(axis="both", which="major", labelsize=10)
ax[1].tick_params(axis="both", which="major", labelsize=10)

plt.legend(ncol=2, frameon=False, fontsize=10)
plt.tight_layout()
plt.savefig("battery_dispatch.png", dpi=300, bbox_inches="tight")
plt.show()
