"""Gradient-based battery and electrolyzer sizing through a differentiable dispatch.

This helper builds a small OpenMDAO problem around H2I's
``DifferentiableDispatchStorageController``. The plant's wind and solar generation profile is
an input. The battery is dispatched optimally for each candidate design, the electricity
delivered to the electrolyzer is converted to hydrogen, and a simple annualized LCOH is the
objective. Every component provides analytic derivatives, so OpenMDAO computes exact total
derivatives with one reverse (adjoint) solve per optimizer iteration.

The cost assumptions follow example 13 (distributed wind, solar, battery, and electrolyzer).
"""

import numpy as np
import openmdao.api as om

from h2integrate.control.control_strategies.storage.differentiable_dispatch_controller import (
    DifferentiableDispatchStorageController,
)


KWH_PER_KG_H2 = 55.0
DISCOUNT_RATE = 0.09
PLANT_LIFE_YEARS = 30

COSTS = {
    # Fixed generation: 66 MW wind at 2286 USD/kW and 50 MWdc solar at 1492 USD/kWdc
    "generation_capex": 66_000 * 2286.0 + 50_000 * 1492.0,
    "generation_opex": 66_000 * 38.0 + 50_000 * 16.0,
    "battery_energy_capex": 173_000.0,  # USD/MWh
    "battery_power_capex": 750_000.0,  # USD/MW
    "battery_opex_fraction": 0.025,
    "electrolyzer_capex": 1_000_000.0,  # USD/MW
    "electrolyzer_opex": 12_800.0,  # USD/MW/year
}

BATTERY_PARAMETERS = {
    "commodity": "electricity",
    "commodity_rate_units": "MW",
    "commodity_amount_units": "MW*h",
    "max_capacity": 48.0,
    "max_charge_rate": 12.0,
    "init_soc_fraction": 0.0,
    "charge_equals_discharge": True,
    "charge_efficiency": 0.95,
    "discharge_efficiency": 0.95,
    "min_soc_fraction": 0.0,
    "max_soc_fraction": 1.0,
    "demand_profile": 60.0,
    "unmet_demand_penalty": 100.0,
    "charge_cost": 1.0,
    "discharge_cost": 1.0,
}


class ElectrolyzerSetPoint(om.ExplicitComponent):
    """Constant electricity set-point equal to the electrolyzer rating."""

    def initialize(self):
        self.options.declare("n_timesteps", types=int)

    def setup(self):
        n = self.options["n_timesteps"]
        self.add_input("electrolyzer_rating", val=60.0, units="MW")
        self.add_output("electricity_set_point", val=60.0, shape=n, units="MW")
        self.declare_partials(
            "electricity_set_point",
            "electrolyzer_rating",
            rows=np.arange(n),
            cols=np.zeros(n, dtype=int),
            val=1.0,
        )

    def compute(self, inputs, outputs):
        outputs["electricity_set_point"] = inputs["electrolyzer_rating"][0]


class LinearElectrolyzer(om.ExplicitComponent):
    """Hydrogen from the electricity delivered to the electrolyzer at a fixed efficiency.

    Outputs the annual hydrogen production and the electrolyzer capacity factor.
    """

    def initialize(self):
        self.options.declare("n_timesteps", types=int)

    def setup(self):
        n = self.options["n_timesteps"]
        self.add_input("dispatch_electricity_delivered", val=0.0, shape=n, units="MW")
        self.add_input("electrolyzer_rating", val=60.0, units="MW")
        self.add_output("annual_hydrogen", val=0.0, units="kg/year")
        self.add_output("electrolyzer_capacity_factor", val=0.0, units="unitless")
        # MW over one hour is 1000 kWh; scale a partial year up to a full year.
        self._factor = 1000.0 / KWH_PER_KG_H2 * (8760.0 / n)
        self.declare_partials("annual_hydrogen", "dispatch_electricity_delivered", val=self._factor)
        self.declare_partials("electrolyzer_capacity_factor", "*")

    def compute(self, inputs, outputs):
        delivered = inputs["dispatch_electricity_delivered"]
        rating = inputs["electrolyzer_rating"][0]
        outputs["annual_hydrogen"] = self._factor * np.sum(delivered)
        outputs["electrolyzer_capacity_factor"] = np.mean(delivered) / rating

    def compute_partials(self, inputs, partials):
        delivered = inputs["dispatch_electricity_delivered"]
        rating = inputs["electrolyzer_rating"][0]
        partials["electrolyzer_capacity_factor", "dispatch_electricity_delivered"] = 1.0 / (
            delivered.size * rating
        )
        partials["electrolyzer_capacity_factor", "electrolyzer_rating"] = (
            -np.mean(delivered) / rating**2
        )


class SimpleLCOH(om.ExplicitComponent):
    """Annualized levelized cost of hydrogen with analytic partials.

    Also counts model and gradient evaluations for the optimizer comparison.
    """

    def setup(self):
        self.add_input("storage_capacity", val=48.0, units="MW*h")
        self.add_input("max_charge_rate", val=12.0, units="MW")
        self.add_input("electrolyzer_rating", val=60.0, units="MW")
        self.add_input("annual_hydrogen", val=1.0, units="kg/year")
        self.add_output("LCOH", val=0.0, units="USD/kg")
        self.declare_partials("LCOH", "*")
        r, n = DISCOUNT_RATE, PLANT_LIFE_YEARS
        self.crf = r * (1 + r) ** n / ((1 + r) ** n - 1)
        self.n_compute = 0
        self.n_partials = 0

    def _annual_cost_coefficients(self):
        c = COSTS
        battery_factor = self.crf + c["battery_opex_fraction"]
        return {
            "fixed": self.crf * c["generation_capex"] + c["generation_opex"],
            "storage_capacity": battery_factor * c["battery_energy_capex"],
            "max_charge_rate": battery_factor * c["battery_power_capex"],
            "electrolyzer_rating": self.crf * c["electrolyzer_capex"] + c["electrolyzer_opex"],
        }

    def compute(self, inputs, outputs):
        self.n_compute += 1
        coef = self._annual_cost_coefficients()
        annual_cost = coef["fixed"] + sum(
            coef[name] * inputs[name][0]
            for name in ("storage_capacity", "max_charge_rate", "electrolyzer_rating")
        )
        outputs["LCOH"] = annual_cost / inputs["annual_hydrogen"][0]

    def compute_partials(self, inputs, partials):
        self.n_partials += 1
        coef = self._annual_cost_coefficients()
        h2 = inputs["annual_hydrogen"][0]
        annual_cost = coef["fixed"] + sum(
            coef[name] * inputs[name][0]
            for name in ("storage_capacity", "max_charge_rate", "electrolyzer_rating")
        )
        for name in ("storage_capacity", "max_charge_rate", "electrolyzer_rating"):
            partials["LCOH", name] = coef[name] / h2
        partials["LCOH", "annual_hydrogen"] = -annual_cost / h2**2


def build_sizing_problem(generation_MW, optimizer="SLSQP", min_capacity_factor=0.6, maxiter=200):
    """Build the battery and electrolyzer sizing problem.

    Args:
        generation_MW (np.ndarray): Hourly wind and solar generation in MW.
        optimizer (str): ``"SLSQP"`` (exact gradients) or ``"COBYLA"`` (gradient-free).
        min_capacity_factor (float | None): Lower bound on the electrolyzer capacity factor,
            for example to serve a steady hydrogen offtaker. None removes the constraint.
        maxiter (int): Maximum optimizer iterations.

    Returns:
        om.Problem: The set-up problem, ready to run.
    """
    n = int(generation_MW.size)
    plant_config = {"plant": {"plant_life": 30, "simulation": {"n_timesteps": n, "dt": 3600}}}
    tech_config = {"model_inputs": {"shared_parameters": dict(BATTERY_PARAMETERS)}}

    prob = om.Problem(reports=False)
    model = prob.model
    ivc = model.add_subsystem("design", om.IndepVarComp(), promotes=["*"])
    ivc.add_output("storage_capacity", 48.0, units="MW*h")
    ivc.add_output("max_charge_rate", 12.0, units="MW")
    ivc.add_output("electrolyzer_rating", 60.0, units="MW")
    ivc.add_output("electricity_in", np.asarray(generation_MW, dtype=float), units="MW")

    model.add_subsystem("set_point", ElectrolyzerSetPoint(n_timesteps=n), promotes=["*"])
    model.add_subsystem(
        "battery_dispatch",
        DifferentiableDispatchStorageController(plant_config=plant_config, tech_config=tech_config),
        promotes=["*"],
    )
    model.add_subsystem("electrolyzer", LinearElectrolyzer(n_timesteps=n), promotes=["*"])
    model.add_subsystem("lcoh", SimpleLCOH(), promotes=["*"])

    model.add_design_var("storage_capacity", lower=1.0, upper=500.0, ref=100.0, units="MW*h")
    model.add_design_var("max_charge_rate", lower=1.0, upper=100.0, ref=25.0, units="MW")
    model.add_design_var("electrolyzer_rating", lower=5.0, upper=100.0, ref=50.0, units="MW")
    model.add_objective("LCOH", ref=5.0, units="USD/kg")
    if min_capacity_factor is not None:
        model.add_constraint("electrolyzer_capacity_factor", lower=min_capacity_factor)

    prob.driver = om.ScipyOptimizeDriver(optimizer=optimizer, maxiter=maxiter, disp=False)
    if optimizer == "COBYLA":
        prob.driver.opt_settings["rhobeg"] = 0.2
        prob.driver.options["tol"] = 1e-5
    else:
        prob.driver.options["tol"] = 1e-7

    prob.setup(mode="rev")
    return prob
