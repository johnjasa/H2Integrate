import numpy as np
from attrs import field, define

from h2integrate.core.utilities import merge_shared_inputs
from h2integrate.core.validators import gt_zero, range_val
from h2integrate.core.model_baseclasses import ResizeablePerformanceModelBaseConfig
from h2integrate.converters.hydrogen.electrolyzer_baseclass import ElectrolyzerPerformanceBaseClass


@define(kw_only=True)
class SimpleElectrolyzerPerformanceModelConfig(ResizeablePerformanceModelBaseConfig):
    """Configuration class for the SimpleElectrolyzerPerformanceModel.

    Args:
        rating_MW (float): Nameplate electrical rating of the electrolyzer in MW.
        efficiency_kWh_per_kg (float): Electricity required per kilogram of hydrogen
            produced, in kWh/kg. Constant across the whole operating range.
        turndown_ratio (float): Minimum fraction of the nameplate rating at which the
            electrolyzer can operate. Below this fraction the electrolyzer shuts off
            and produces no hydrogen. Defaults to 0.1.
    """

    rating_MW: float = field(validator=gt_zero)
    efficiency_kWh_per_kg: float = field(validator=gt_zero)
    turndown_ratio: float = field(default=0.1, validator=range_val(0.0, 1.0))


class SimpleElectrolyzerPerformanceModel(ElectrolyzerPerformanceBaseClass):
    """A constant-efficiency electrolyzer performance model.

    Hydrogen production is the electricity consumed divided by a single specific energy
    consumption value in kWh/kg. Electricity consumption is capped at the nameplate
    rating, and the electrolyzer shuts off whenever the available power falls below the
    turndown threshold.

    There is no degradation, no cluster-level dispatch, and no temperature or pressure
    dependence. Use this model when the electrolyzer only needs to be represented as a
    fixed conversion efficiency, for example in early-stage sizing studies or when a
    smooth, inexpensive electrolyzer response is needed inside an optimization loop.

    This model does not include a cost model. Pair it with one of the existing
    electrolyzer cost models, such as ``CustomElectrolyzerCostModel``, which consumes the
    ``electrolyzer_size_mw`` output.
    """

    _time_step_bounds = (
        3600,
        3600,
    )  # (min, max) time step lengths (in seconds) compatible with this model
    _control_classifier = "dispatchable"

    def setup(self):
        self.config = SimpleElectrolyzerPerformanceModelConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance"),
            strict=False,
            additional_cls_name=self.__class__.__name__,
        )
        super().setup()

        self.add_input(
            "rating_MW",
            val=self.config.rating_MW,
            units="MW",
            desc="Nameplate electrical rating of the electrolyzer",
        )
        self.add_output(
            "electrolyzer_size_mw",
            val=self.config.rating_MW,
            units="MW",
            desc="Size of the electrolyzer in MW",
        )
        self.add_output(
            "electricity_consumed",
            val=0.0,
            shape=self.n_timesteps,
            units="kW",
            desc="Electricity actually consumed by the electrolyzer",
        )

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        size_mode = discrete_inputs["size_mode"]
        if size_mode != "normal":
            raise NotImplementedError(
                f"{self.__class__.__name__} only supports the 'normal' size mode, but "
                f"'{size_mode}' was requested. Set the electrolyzer size directly with "
                "'rating_MW'."
            )

        specific_energy = self.config.efficiency_kWh_per_kg
        rating_kW = float(inputs["rating_MW"][0]) * 1e3
        rated_hydrogen_production = rating_kW / specific_energy

        # Consume no more than the nameplate rating, and no more than is needed to meet
        # the hydrogen command value when a system-level controller is present.
        electricity_consumed = np.minimum(inputs["electricity_in"], rating_kW)
        if "system_level_control" in self.options["plant_config"]:
            command_value = inputs[f"{self.commodity}_command_value"]
            electricity_consumed = np.minimum(electricity_consumed, command_value * specific_energy)

        # Below the turndown threshold the electrolyzer shuts off entirely.
        minimum_power_kW = self.config.turndown_ratio * rating_kW
        electricity_consumed = np.where(
            electricity_consumed >= minimum_power_kW, electricity_consumed, 0.0
        )

        hydrogen_out = electricity_consumed / specific_energy
        total_hydrogen_produced = np.sum(hydrogen_out) * (self.dt / 3600.0)
        max_production = rated_hydrogen_production * self.n_timesteps * (self.dt / 3600.0)

        outputs["electricity_consumed"] = electricity_consumed
        outputs["hydrogen_out"] = hydrogen_out
        outputs["rated_hydrogen_production"] = rated_hydrogen_production
        outputs["total_hydrogen_produced"] = total_hydrogen_produced
        outputs["annual_hydrogen_produced"] = np.full(
            self.plant_life, total_hydrogen_produced / self.fraction_of_year_simulated
        )
        outputs["capacity_factor"] = np.full(
            self.plant_life, total_hydrogen_produced / max_production
        )
        outputs["electrolyzer_size_mw"] = rating_kW / 1e3
