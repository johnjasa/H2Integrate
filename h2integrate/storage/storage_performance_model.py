import numpy as np
import openmdao.api as om
from attrs import field, define, validators

from h2integrate.core.utilities import merge_shared_inputs, component_totals_enabled
from h2integrate.core.supported_models import register
from h2integrate.storage.storage_baseclass import (
    StoragePerformanceBase,
    StoragePerformanceBaseConfig,
)


@define(kw_only=True)
class StoragePerformanceModelConfig(StoragePerformanceBaseConfig):
    """Configuration class for storage performance models.

    This class defines configuration parameters for simulating storage
    performance with the Pyomo controllers. It includes
    specifications such as capacity, charge rate, state-of-charge limits,
    and charge/discharge efficiencies.

    Attributes:
        commodity (str): name of commodity
        commodity_rate_units (str): Units of the commodity (e.g., "kg/h").
        demand_profile (int | float | list): Demand values for each timestep, in
            the same units as `commodity_rate_units`. May be a scalar for constant
            demand or a list/array for time-varying demand.
        max_capacity (float):  Maximum storage energy capacity in commodity_amount_units.
            Must be greater than zero.
        max_charge_rate (float): Rated commodity capacity of the storage  in commodity_rate_units.
            Must be greater than zero.
        min_soc_fraction (float): Minimum allowable state of charge as a fraction (0 to 1).
        max_soc_fraction (float): Maximum allowable state of charge as a fraction (0 to 1).
        init_soc_fraction (float): Initial state of charge as a fraction (0 to 1).
        commodity_amount_units (str | None, optional): Units of the commodity as an amount
            (i.e., kW*h or kg). If not provided, defaults to commodity_rate_units*h.
        max_discharge_rate (float | None, optional): Maximum rate at which the commodity can be
            discharged (in units per time step, e.g., "kg/time step"). This rate does not include
            the discharge_efficiency. Only required if `charge_equals_discharge` is False.
        charge_equals_discharge (bool, optional): If True, set the max_discharge_rate equal to the
            max_charge_rate. If False, specify the max_discharge_rate as a value different than
            the max_charge_rate. Defaults to True.
        charge_efficiency (float | None, optional): Efficiency of charging the storage, represented
            as a decimal between 0 and 1 (e.g., 0.9 for 90% efficiency). Optional if
            `round_trip_efficiency` is provided.
        discharge_efficiency (float | None, optional): Efficiency of discharging the storage,
            represented as a decimal between 0 and 1 (e.g., 0.9 for 90% efficiency). Optional if
            `round_trip_efficiency` is provided.
        round_trip_efficiency (float | None, optional): Combined efficiency of charging and
            discharging the storage, represented as a decimal between 0 and 1 (e.g., 0.81 for
            81% efficiency). Optional if `charge_efficiency` and `discharge_efficiency` are
            provided.

    """

    commodity: str = field()
    commodity_rate_units: str = field()

    max_capacity: float = field(validator=validators.gt(0))
    max_charge_rate: float = field(validator=validators.gt(0))

    init_soc_fraction: float = field(validator=(validators.ge(0), validators.le(1)))

    commodity_amount_units: str = field(default=None)
    max_discharge_rate: float | None = field(default=None)
    charge_equals_discharge: bool = field(default=True)

    charge_efficiency: float | None = field(
        default=None, validator=validators.optional((validators.ge(0), validators.le(1)))
    )
    discharge_efficiency: float | None = field(
        default=None, validator=validators.optional((validators.ge(0), validators.le(1)))
    )
    round_trip_efficiency: float | None = field(
        default=None, validator=validators.optional((validators.ge(0), validators.le(1)))
    )

    def __attrs_post_init__(self):
        """
        Post-initialization logic to validate and calculate efficiencies.

        Ensures that either `charge_efficiency` and `discharge_efficiency` are provided,
        or `round_trip_efficiency` is provided. If `round_trip_efficiency` is provided,
        it calculates `charge_efficiency` and `discharge_efficiency` as the square root
        of `round_trip_efficiency`.
        """
        if (self.round_trip_efficiency is not None) and (
            self.charge_efficiency is None and self.discharge_efficiency is None
        ):
            # Calculate charge and discharge efficiencies from round-trip efficiency
            self.charge_efficiency = np.sqrt(self.round_trip_efficiency)
            self.discharge_efficiency = np.sqrt(self.round_trip_efficiency)

        if self.charge_efficiency is None or self.discharge_efficiency is None:
            raise ValueError(
                "Exactly one of the following sets of parameters must be set: (a) "
                "`round_trip_efficiency`, or (b) both `charge_efficiency` "
                "and `discharge_efficiency`."
            )

        if self.charge_equals_discharge:
            if (
                self.max_discharge_rate is not None
                and self.max_discharge_rate != self.max_charge_rate
            ):
                msg = (
                    "Max discharge rate does not equal max charge rate but charge_equals_discharge "
                    f"is True. Discharge rate is {self.max_discharge_rate} and charge rate "
                    f"is {self.max_charge_rate}."
                )
                raise ValueError(msg)

            self.max_discharge_rate = self.max_charge_rate

        if not self.charge_equals_discharge and self.max_discharge_rate is None:
            msg = (
                "max_discharge_rate is required when charge_equals_discharge is False. "
                "Please input the discharge rate using the key `max_discharge_rate`."
            )
            raise ValueError(msg)

        if self.commodity_amount_units is None:
            self.commodity_amount_units = f"({self.commodity_rate_units})*h"


@register
class StoragePerformanceModel(StoragePerformanceBase):
    """OpenMDAO component for a storage component."""

    _time_step_bounds = (
        1,
        3600,
    )  # (min, max) time step lengths (in seconds) compatible with this model

    _is_steppable = True

    def setup(self):
        self.config = StoragePerformanceModelConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance"),
            strict=False,
            additional_cls_name=self.__class__.__name__,
        )

        self.commodity = self.config.commodity
        self.commodity_rate_units = self.config.commodity_rate_units
        self.commodity_amount_units = self.config.commodity_amount_units

        super().setup()

        self.add_output(
            f"{self.commodity}_charge_demand",
            val=0.0,
            shape=self.n_timesteps,
            units=self.commodity_rate_units,
        )  # the current demand to charge the system at maximum rate

        self.add_output(
            f"{self.commodity}_headroom",
            val=0.0,
            shape=self.n_timesteps,
            units=self.commodity_rate_units,
        )  # this represents the excess capacity that could have been dumped

        self.matrix_free = component_totals_enabled(self.options)

    def compute(self, inputs, outputs, discrete_inputs=[], discrete_outputs=[]):
        """Run the storage performance model."""

        # Unpack scalar inputs
        charge_rate = inputs["max_charge_rate"][0]

        if "max_discharge_rate" in inputs:
            discharge_rate = inputs["max_discharge_rate"][0]
        else:
            discharge_rate = inputs["max_charge_rate"][0]
        storage_capacity = inputs["storage_capacity"][0]

        outputs = self.run_storage(
            charge_rate, discharge_rate, storage_capacity, inputs, outputs, discrete_inputs
        )

        # add charge demand calculation
        headroom_capacity_charge = (
            (self.config.max_soc_fraction - outputs["SOC"] / 100.0) * storage_capacity / self.dt_hr
        )
        available_charge = np.maximum(
            0.0,  # at worst no charge is available
            np.minimum(
                charge_rate,  # the fundamental limit on charge rate is the max
                headroom_capacity_charge,  # but that could be limited by the available capacity
            ),
        )  # this is the max i could charge right now

        # add headroom calculation
        headroom_capacity_discharge = (
            (outputs["SOC"] / 100.0 - self.config.min_soc_fraction) * storage_capacity / self.dt_hr
        )  # i *could've* dumped the state of charge by this much

        available_discharge = np.maximum(
            0.0,  # at worst no discharge is available
            np.minimum(
                discharge_rate,  # the fundamental limit on discharge rate is the max
                headroom_capacity_discharge,  # but that could be limited by the available capacity
            ),
        )  # this is the max i could discharge right now

        # add the charge/discharge outputs
        outputs[f"{self.commodity}_charge_demand"] = (
            available_charge / self.config.charge_efficiency
        )
        outputs[f"{self.commodity}_headroom"] = (
            available_discharge
            * self.config.discharge_efficiency  # i could dump this much power out total
        )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode, discrete_inputs=None):
        """Propagate piecewise storage derivatives forward over a full simulation horizon."""
        if mode != "fwd":
            raise om.AnalysisError(
                "StoragePerformanceModel component derivatives currently require "
                "derivative_mode: fwd."
            )
        if self.using_feedback_control or self.n_timesteps_window != self.n_timesteps:
            raise om.AnalysisError(
                "StoragePerformanceModel component derivatives require open-loop control "
                "and a full-horizon simulation."
            )
        if "timestep_index" in d_inputs and np.any(d_inputs["timestep_index"]):
            raise om.AnalysisError(
                "Storage timestep_index is discrete and cannot be differentiated."
            )

        commodity = self.commodity

        def scalar_seed(name):
            return float(d_inputs[name][0]) if name in d_inputs else 0.0

        def vector_seed(name):
            return np.asarray(d_inputs[name]) if name in d_inputs else np.zeros(self.n_timesteps)

        capacity = inputs["storage_capacity"][0]
        charge_rate = inputs["max_charge_rate"][0]
        discharge_rate = (
            inputs["max_discharge_rate"][0] if "max_discharge_rate" in inputs else charge_rate
        )
        if capacity <= 0 or charge_rate <= 0 or discharge_rate <= 0:
            raise om.AnalysisError(
                "StoragePerformanceModel derivatives require positive capacity and rate limits."
            )
        d_capacity = scalar_seed("storage_capacity")
        d_charge_rate = scalar_seed("max_charge_rate")
        d_discharge_rate = (
            scalar_seed("max_discharge_rate") if "max_discharge_rate" in inputs else d_charge_rate
        )
        available = inputs[f"{commodity}_in"]
        command = inputs[f"{commodity}_command_value"]
        d_available = vector_seed(f"{commodity}_in")
        d_command = vector_seed(f"{commodity}_command_value")
        charge_eff = self.config.charge_efficiency
        discharge_eff = self.config.discharge_efficiency
        soc = self.soc_init
        d_soc = 0.0
        flow = np.zeros(self.n_timesteps)
        d_flow = np.zeros(self.n_timesteps)
        soc_values = np.zeros(self.n_timesteps)
        d_soc_values = np.zeros(self.n_timesteps)

        for t in range(self.n_timesteps):
            if command[t] < 0:
                headroom = (self.config.max_soc_fraction - soc) * capacity / self.dt_amount
                d_headroom = (
                    (self.config.max_soc_fraction - soc) * d_capacity - capacity * d_soc
                ) / self.dt_amount
                values = (headroom / charge_eff, charge_rate, -command[t], available[t])
                tangents = (d_headroom / charge_eff, d_charge_rate, -d_command[t], d_available[t])
                selected = int(np.argmin(values))
                charge = max(0.0, values[selected])
                d_charge = tangents[selected] if values[selected] > 0 else 0.0
                delta = charge * charge_eff * self.dt_amount / capacity
                d_delta = (
                    charge_eff
                    * self.dt_amount
                    * (d_charge / capacity - charge * d_capacity / capacity**2)
                )
                soc += delta
                d_soc += d_delta
                flow[t] = -charge
                d_flow[t] = -d_charge
            else:
                headroom = (soc - self.config.min_soc_fraction) * capacity / self.dt_amount
                d_headroom = (
                    (soc - self.config.min_soc_fraction) * d_capacity + capacity * d_soc
                ) / self.dt_amount
                values = (headroom, discharge_rate / discharge_eff, command[t] / discharge_eff)
                tangents = (
                    d_headroom,
                    d_discharge_rate / discharge_eff,
                    d_command[t] / discharge_eff,
                )
                selected = int(np.argmin(values))
                discharge = max(0.0, values[selected])
                d_discharge = tangents[selected] if values[selected] > 0 else 0.0
                delta = discharge * self.dt_amount / capacity
                d_delta = self.dt_amount * (
                    d_discharge / capacity - discharge * d_capacity / capacity**2
                )
                soc -= delta
                d_soc -= d_delta
                flow[t] = discharge * discharge_eff
                d_flow[t] = d_discharge * discharge_eff
            soc_values[t] = soc
            d_soc_values[t] = d_soc

        d_charge = np.where(flow < 0, d_flow, 0.0)
        d_discharge = np.where(flow > 0, d_flow, 0.0)
        amount = np.sum(flow) * self.dt_amount
        d_amount = np.sum(d_flow) * self.dt_amount
        discharged = np.maximum(flow, 0.0).sum() * self.dt_amount
        d_discharged = d_discharge.sum() * self.dt_amount
        denominator = discharge_rate * self.n_timesteps * self.dt_amount
        d_denominator = d_discharge_rate * self.n_timesteps * self.dt_amount

        charge_headroom = (self.config.max_soc_fraction - soc_values) * capacity / self.dt_hr
        d_charge_headroom = (
            (self.config.max_soc_fraction - soc_values) * d_capacity - capacity * d_soc_values
        ) / self.dt_hr
        discharge_headroom = (soc_values - self.config.min_soc_fraction) * capacity / self.dt_hr
        d_discharge_headroom = (
            (soc_values - self.config.min_soc_fraction) * d_capacity + capacity * d_soc_values
        ) / self.dt_hr
        d_charge_available = np.where(
            charge_headroom < charge_rate, d_charge_headroom, d_charge_rate
        )
        d_charge_available = np.where(
            np.minimum(charge_headroom, charge_rate) > 0, d_charge_available, 0.0
        )
        d_discharge_available = np.where(
            discharge_headroom < discharge_rate, d_discharge_headroom, d_discharge_rate
        )
        d_discharge_available = np.where(
            np.minimum(discharge_headroom, discharge_rate) > 0, d_discharge_available, 0.0
        )
        derivatives = {
            f"{commodity}_out": d_flow,
            f"storage_{commodity}_charge": d_charge,
            f"storage_{commodity}_discharge": d_discharge,
            "SOC": 100.0 * d_soc_values,
            "storage_duration": d_capacity / discharge_rate
            - capacity * d_discharge_rate / discharge_rate**2,
            f"rated_{commodity}_production": d_discharge_rate,
            f"total_{commodity}_produced": d_amount,
            f"annual_{commodity}_produced": d_amount / self.fraction_of_year_simulated,
            "capacity_factor": d_amount / denominator - amount * d_denominator / denominator**2,
            "standard_capacity_factor": d_discharged / denominator
            - discharged * d_denominator / denominator**2,
            f"{commodity}_charge_demand": d_charge_available / charge_eff,
            f"{commodity}_headroom": d_discharge_available * discharge_eff,
        }
        for name, value in derivatives.items():
            if name in d_outputs:
                d_outputs[name] += value
