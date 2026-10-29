from attrs import field, define, validators
from openmdao.utils import units

from h2integrate.core.utilities import merge_shared_inputs
from h2integrate.core.model_baseclass import CostModelBaseClass, CostModelBaseConfig
from h2integrate.core.supported_models import register


@define(kw_only=True)
class GenericStorageCostConfig(CostModelBaseConfig):
    """Configuration class for the GenericStorageCostModel with costs based on storage
    capacity and charge rate for any commodity.

    Note:
        This could be expanded to allow for different types of commodity units in the future.
        Currently only supports electrical, mass, and some thermal units.

    Fields include `capacity_capex`, `charge_capex`, `opex_fraction`, `max_capacity`,
    `max_charge_rate`, and `commodity_rate_units`. The `cost_year` field is inherited
    from `CostModelBaseConfig`.
    """

    capacity_capex: float | int = field(validator=validators.ge(0))
    charge_capex: float | int = field(validator=validators.ge(0))
    opex_fraction: float = field(validator=(validators.ge(0), validators.le(1)))
    max_capacity: float = field()
    max_charge_rate: float = field()
    commodity_rate_units: str = field(
        validator=validators.in_(["W", "kW", "MW", "GW", "TW", "g/h", "kg/h", "t/h", "MMBtu/h"])
    )

    commodity_amount_units: str = field(default=None)

    def __attrs_post_init__(self):
        if self.commodity_amount_units is None:
            self.commodity_amount_units = f"({self.commodity_rate_units})*h"


@register
class GenericStorageCostModel(CostModelBaseClass):
    """Generic storage cost model for any commodity (electricity, hydrogen, etc.).

    This model calculates costs based on storage capacity and charge/discharge rate.

    Total_CapEx = capacity_capex * Storage_Hours + charge_capex

    - Total_CapEx: Total System Cost (USD/charge_units)
    - Storage_Hours: Storage Duration (hr)
    - capacity_capex: Storage Capacity Cost (USD/capacity_units)
    - charge_capex: Storage Charge Cost (USD/charge_units)

    """

    _cost_year_is_constant = True
    _time_step_bounds = (
        3600,
        3600,
    )  # (min, max) time step lengths (in seconds) compatible with this model

    def setup(self):
        self.config = GenericStorageCostConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "cost"),
            strict=False,
            additional_cls_name=self.__class__.__name__,
        )

        super().setup()

        charge_units = self.config.commodity_rate_units

        capacity_units = self.config.commodity_amount_units

        self.add_input(
            "max_charge_rate",
            val=self.config.max_charge_rate,
            units=charge_units,
            desc="Storage charge/discharge rate",
        )
        self.add_input(
            "storage_capacity",
            val=self.config.max_capacity,
            units=capacity_units,
            desc="Storage storage capacity",
        )
        self.add_input(
            "capacity_capex",
            val=self.config.capacity_capex,
            units=f"USD/({capacity_units})",
            desc="Storage energy capital cost",
        )
        self.add_input(
            "charge_capex",
            val=self.config.charge_capex,
            units=f"USD/({charge_units})",
            desc="Storage power capital cost",
        )
        self.add_input(
            "opex_fraction",
            val=self.config.opex_fraction,
            units="unitless",
            desc="Annual operating cost as a fraction of total system cost",
        )
        for output in ("CapEx", "OpEx"):
            for input_name in (
                "max_charge_rate",
                "storage_capacity",
                "capacity_capex",
                "charge_capex",
                "opex_fraction",
            ):
                if output != "CapEx" or input_name != "opex_fraction":
                    self.declare_partials(output, input_name)

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        storage_duration_hrs = 0.0

        if inputs["max_charge_rate"] > 0:
            storage_duration_hrs = units.convert_units(
                inputs["storage_capacity"] / inputs["max_charge_rate"],
                f"({self.config.commodity_amount_units})/({self.config.commodity_rate_units})",
                "h",
            )
        if inputs["max_charge_rate"] < 0:
            msg = (
                f"max_charge_rate cannot be less than zero and has value of "
                f"{inputs['max_charge_rate']}"
            )
            raise UserWarning(msg)
        # Calculate total system cost based on capacity and charge components
        total_system_cost = (storage_duration_hrs * inputs["capacity_capex"]) + inputs[
            "charge_capex"
        ]
        capex = total_system_cost * inputs["max_charge_rate"]
        # Calculate operating expenses as a fraction of capital expenses
        opex = inputs["opex_fraction"] * capex
        outputs["CapEx"] = capex
        outputs["OpEx"] = opex

    def compute_partials(self, inputs, partials):
        charge_rate = inputs["max_charge_rate"][0]
        capacity = inputs["storage_capacity"][0]
        capacity_capex = inputs["capacity_capex"][0]
        charge_capex = inputs["charge_capex"][0]
        fraction = inputs["opex_fraction"][0]
        conversion = units.convert_units(
            1.0,
            f"({self.config.commodity_amount_units})/({self.config.commodity_rate_units})",
            "h",
        )
        if charge_rate <= 0:
            raise ValueError(
                "GenericStorageCostModel derivatives require a positive max_charge_rate; "
                "the cost is discontinuous at zero."
            )
        capex = capacity * conversion * capacity_capex + charge_rate * charge_capex
        derivatives = {
            "max_charge_rate": charge_capex,
            "storage_capacity": conversion * capacity_capex,
            "capacity_capex": capacity * conversion,
            "charge_capex": charge_rate,
        }
        for name, value in derivatives.items():
            partials["CapEx", name] = value
            partials["OpEx", name] = fraction * value
        partials["OpEx", "opex_fraction"] = capex
