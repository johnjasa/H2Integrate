from attrs import field, define

from h2integrate.core.utilities import merge_shared_inputs
from h2integrate.core.validators import gt_zero
from h2integrate.core.model_baseclasses import CostModelBaseClass, CostModelBaseConfig


@define(kw_only=True)
class GeothermalPlantCostModelConfig(CostModelBaseConfig):
    """Configuration class for the GeothermalPlantCostModel.

    Capital and operating costs are scaled by the plant net (AC) capacity. Representative
    cost values for geothermal power technologies can be found in the NREL Annual Technology
    Baseline (ATB), `here <https://atb.nrel.gov/electricity/2024/geothermal>`_.

    Attributes:
        capex_per_kW (float | int): capital cost of the geothermal plant in $/kW-AC.
        opex_per_kW_per_year (float | int): annual fixed operating cost of the geothermal
            plant in $/kW-AC/year.
        cost_year (int): dollar year corresponding to the input costs.
    """

    capex_per_kW: float | int = field(validator=gt_zero)
    opex_per_kW_per_year: float | int = field(validator=gt_zero)


class GeothermalPlantCostModel(CostModelBaseClass):
    """Capacity-based cost model for a geothermal power plant.

    CapEx and fixed OpEx are computed from the plant net (AC) capacity using
    per-kW cost factors.
    """

    _time_step_bounds = (
        3600,
        3600,
    )  # (min, max) time step lengths (in seconds) compatible with this model

    def setup(self):
        self.config = GeothermalPlantCostModelConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "cost"),
            additional_cls_name=self.__class__.__name__,
        )
        super().setup()

        self.add_input(
            "system_capacity_AC",
            val=0.0,
            units="kW",
            desc="Geothermal plant net rated capacity",
        )

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        capacity = inputs["system_capacity_AC"][0]
        outputs["CapEx"] = self.config.capex_per_kW * capacity
        outputs["OpEx"] = self.config.opex_per_kW_per_year * capacity
