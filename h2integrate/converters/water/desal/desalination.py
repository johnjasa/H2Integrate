import numpy as np
from attrs import field, define, converters, validators

from h2integrate.core.utilities import BaseConfig, merge_shared_inputs
from h2integrate.core.model_baseclass import (
    CostModelBaseClass,
    CostModelBaseConfig,
    PerformanceModelBaseClass,
)
from h2integrate.core.supported_models import register


# SWRO: Sea Water Reverse Osmosis, water >18,000 ppm
# BWRO: Brackish Water Reverse Osmosis, water <18,000 ppm
# recovery_ratio (freshwater out / feedwater in) source:
#   https://www.usbr.gov/research/dwpr/reportpdfs/report072.pdf
# specific_energy_consumption [kW*h/m**3 of freshwater] source:
#   https://www.sciencedirect.com/science/article/pii/S0011916417321057
#   SWRO range is 2.5 to 4.0 kWh/m^3 and BWRO range is 1.0 to 1.5 kWh/m^3
SALINITY_PROPERTIES = {
    "seawater": {"recovery_ratio": 0.5, "specific_energy_consumption": 4.0},
    "brackish": {"recovery_ratio": 0.75, "specific_energy_consumption": 1.5},
}


@define(kw_only=True)
class ReverseOsmosisDesignPointConfig(BaseConfig):
    """Fixed operating point imported from an offline reverse osmosis solve.

    Args:
        source (str): Reference identifying the solver version and run settings.
        feed_mass_flow_kg_per_hr (float): Total feed solution mass flow [kg/h].
        feed_density_kg_per_m3 (float): Feed solution density [kg/m**3].
        net_electricity_kw (float): Net electrical demand of the RO train [kW].
        feed_salt_mass_fraction (float): Dissolved salt mass fraction of the feed solution.
        permeate_salt_mass_fraction (float): Dissolved salt mass fraction of the permeate.
        max_permeate_salt_mass_fraction (float): Maximum acceptable permeate salt mass fraction.

    See :class:`ReverseOsmosisPerformanceModelConfig` and :doc:`/technology_models/desalination`.
    """

    source: str = field(validator=validators.instance_of(str))
    feed_mass_flow_kg_per_hr: float = field(validator=validators.gt(0))
    feed_density_kg_per_m3: float = field(validator=validators.gt(0))
    net_electricity_kw: float = field(validator=validators.gt(0))
    feed_salt_mass_fraction: float = field(
        validator=validators.and_(validators.ge(0), validators.lt(1))
    )
    permeate_salt_mass_fraction: float = field(
        validator=validators.and_(validators.ge(0), validators.lt(1))
    )
    max_permeate_salt_mass_fraction: float = field(
        validator=validators.and_(validators.ge(0), validators.lt(1))
    )


@define(kw_only=True)
class ReverseOsmosisPerformanceModelConfig(BaseConfig):
    """Configuration class for the ReverseOsmosisPerformanceModel.

    Args:
        freshwater_capacity_kg_per_hr (float): Desalination plant capacity, represented as
            the maximum freshwater production rate [kg/h].
        salinity (str, optional): "seawater" (>18,000 ppm) or "brackish" (<18,000 ppm). Sets the
            recovery ratio (0.5 for seawater, 0.75 for brackish) and specific energy
            consumption (4.0 kW*h/m**3 for seawater, 1.5 kW*h/m**3 for brackish). Specify
            either salinity or a design point, not both.
        freshwater_density (float, optional): Density of the output freshwater [kg/m**3].
            Defaults to 997.
        mass_per_capacity (float, optional): Mass of the desalination system per unit of
            freshwater capacity [kg/(m**3/h)]. Defaults to 346.7.
        footprint_per_capacity (float, optional): Footprint of the desalination system per unit
            of freshwater capacity [m**2/(m**3/h)]. Defaults to 0.467.
        design_point (ReverseOsmosisDesignPointConfig, optional): A fixed operating point.
            In this mode, the configured freshwater capacity and density are the solved
            permeate solution mass flow and density, respectively.

    See :class:`ReverseOsmosisPerformanceModel` and :doc:`/technology_models/desalination`.
    """

    freshwater_capacity_kg_per_hr: float = field(validator=validators.gt(0))
    salinity: str | None = field(
        default=None, validator=validators.optional(validators.in_(list(SALINITY_PROPERTIES)))
    )
    freshwater_density: float = field(validator=validators.gt(0), default=997)

    # Mass and footprint per unit of freshwater capacity.
    # Based on commercial industrial RO systems:
    # https://www.appliedmembranes.com/s-series-seawater-reverse-osmosis-systems-2000-to-100000-gpd.html
    # All mass and footprint estimates are scaled from the largest RO system, S-308F:
    #   - 436 m^3/day
    #   - 6330 kg
    #   - 762 cm (L) x 112 cm (D) x 183 cm (H)
    # 436 m^3/day = 18.17 m^3/hr = 8.5 m^2, 6330 kg
    # 1 m^3/hr = 0.467 m^2, 346.7 kg
    # Voltage codes: 460 or 480 V / 3 phase / 60 Hz
    mass_per_capacity: float = field(validator=validators.gt(0), default=346.7)
    footprint_per_capacity: float = field(validator=validators.gt(0), default=0.467)
    design_point: ReverseOsmosisDesignPointConfig | None = field(
        default=None, converter=converters.optional(ReverseOsmosisDesignPointConfig.from_dict)
    )

    def __attrs_post_init__(self) -> None:
        """Validate the choice of operating mode and the fixed point's mass balance."""
        if (self.salinity is None) == (self.design_point is None):
            raise ValueError("Specify exactly one of salinity or design_point.")
        point = self.design_point
        if point is None:
            return
        if not point.source.strip():
            raise ValueError("design_point source must identify the offline solve.")
        if not all(
            np.isfinite(value)
            for value in (
                self.freshwater_capacity_kg_per_hr,
                self.freshwater_density,
                point.feed_mass_flow_kg_per_hr,
                point.feed_density_kg_per_m3,
                point.net_electricity_kw,
            )
        ):
            raise ValueError("design_point mass flows, densities, and electricity must be finite.")
        if point.permeate_salt_mass_fraction > point.max_permeate_salt_mass_fraction:
            raise ValueError("design_point permeate exceeds the salt mass fraction limit.")
        if point.permeate_salt_mass_fraction > point.feed_salt_mass_fraction:
            raise ValueError("design_point permeate salt mass fraction exceeds that of the feed.")
        concentrate_mass = point.feed_mass_flow_kg_per_hr - self.freshwater_capacity_kg_per_hr
        concentrate_salt = (
            point.feed_mass_flow_kg_per_hr * point.feed_salt_mass_fraction
            - self.freshwater_capacity_kg_per_hr * point.permeate_salt_mass_fraction
        )
        if concentrate_mass <= 0 or concentrate_salt >= concentrate_mass:
            raise ValueError("design_point feed and permeate do not yield a valid concentrate.")


@register
class ReverseOsmosisPerformanceModel(PerformanceModelBaseClass):
    """Reverse osmosis desalination performance model.

    Freshwater production at each timestep is limited by the lowest of the plant capacity,
    the available electricity, and the available feedwater in salinity mode. With an imported
    design point, the train runs only when enough resources are available for the full point.

    Inputs:
        electricity_in (ndarray): Electricity available to the plant [kW] at each timestep.
        feedwater_in (ndarray): Feedwater available to the plant [m**3/h] at each timestep.
        freshwater_capacity (float): Maximum freshwater production rate [kg/h].
        freshwater_density (float): Density of the freshwater [kg/m**3].

    Outputs:
        water_out (ndarray): Freshwater produced [m**3/h] at each timestep.
        electricity_consumed (ndarray): Electricity consumed [kW] at each timestep.
        feedwater_consumed (ndarray): Feedwater consumed [m**3/h] at each timestep.
        concentrate_mass_flow (ndarray): Concentrate mass flow [kg/h], design-point mode only.
        concentrate_salt_mass_flow (ndarray): Concentrate salt mass flow [kg/h], design-point
            mode only.
        mass (float): Mass of the desalination system [kg].
        footprint (float): Footprint of the desalination system [m**2].

    See :class:`ReverseOsmosisPerformanceModelConfig` and :doc:`/technology_models/desalination`.
    """

    _time_step_bounds = (
        1,
        3600,
    )  # (min, max) time step lengths (in seconds) compatible with this model
    _control_classifier = "dispatchable"

    def initialize(self):
        super().initialize()
        self.commodity = "water"
        self.commodity_amount_units = "m**3"
        self.commodity_rate_units = "m**3/h"

    def setup(self):
        super().setup()
        self.config = ReverseOsmosisPerformanceModelConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance"),
            additional_cls_name=self.__class__.__name__,
        )

        self.add_input(
            "electricity_in",
            val=0.0,
            shape=self.n_timesteps,
            units="kW",
            desc="Electricity available to the desalination plant",
        )
        self.add_input(
            "feedwater_in",
            val=0.0,
            shape=self.n_timesteps,
            units="m**3/h",
            desc="Feedwater available to the desalination plant",
        )
        self.add_input(
            "freshwater_capacity",
            val=self.config.freshwater_capacity_kg_per_hr,
            units="kg/h",
            desc="Maximum freshwater production rate",
        )
        self.add_input(
            "freshwater_density",
            val=self.config.freshwater_density,
            units="kg/m**3",
            desc="Density of the freshwater",
        )

        self.add_output(
            "electricity_consumed",
            val=0.0,
            shape=self.n_timesteps,
            units="kW",
            desc="Electricity consumed by the desalination plant",
        )
        self.add_output(
            "feedwater_consumed",
            val=0.0,
            shape=self.n_timesteps,
            units="m**3/h",
            desc="Feedwater consumed by the desalination plant",
        )
        self.add_output("mass", val=0.0, units="kg", desc="Mass of desalination system")
        self.add_output("footprint", val=0.0, units="m**2", desc="Footprint of desalination system")
        if self.config.design_point is not None:
            self.add_output(
                "concentrate_mass_flow",
                val=0.0,
                shape=self.n_timesteps,
                units="kg/h",
                desc="Mass flow of concentrate solution leaving the desalination plant",
            )
            self.add_output(
                "concentrate_salt_mass_flow",
                val=0.0,
                shape=self.n_timesteps,
                units="kg/h",
                desc="Mass flow of dissolved salt leaving in the concentrate",
            )

    def compute(self, inputs, outputs):
        # Calculate the rated freshwater production capacity as a rate of volume.
        capacity = inputs["freshwater_capacity"] / inputs["freshwater_density"]  # m**3/h

        point = self.config.design_point
        if point is None:
            properties = SALINITY_PROPERTIES[self.config.salinity]
            recovery_ratio = properties["recovery_ratio"]
            specific_energy = properties["specific_energy_consumption"]
            water_from_electricity = inputs["electricity_in"] / specific_energy
            water_from_feedwater = inputs["feedwater_in"] * recovery_ratio
            water_out = np.minimum.reduce(
                np.broadcast_arrays(capacity, water_from_electricity, water_from_feedwater)
            ).clip(min=0.0)
            outputs["electricity_consumed"] = water_out * specific_energy
            outputs["feedwater_consumed"] = water_out / recovery_ratio
        else:
            permeate_mass = self.config.freshwater_capacity_kg_per_hr
            if not (
                np.isclose(inputs["freshwater_capacity"].item(), permeate_mass, rtol=1e-12, atol=0)
                and np.isclose(
                    inputs["freshwater_density"].item(),
                    self.config.freshwater_density,
                    rtol=1e-12,
                    atol=0,
                )
            ):
                raise ValueError(
                    "freshwater_capacity and freshwater_density are fixed at design_point."
                )
            capacity = permeate_mass / self.config.freshwater_density
            feed_mass = point.feed_mass_flow_kg_per_hr
            feed_volume = feed_mass / point.feed_density_kg_per_m3
            running = (inputs["electricity_in"] >= point.net_electricity_kw) & (
                inputs["feedwater_in"] >= feed_volume
            )
            water_out = np.where(running, capacity, 0.0)
            outputs["electricity_consumed"] = np.where(running, point.net_electricity_kw, 0.0)
            outputs["feedwater_consumed"] = np.where(running, feed_volume, 0.0)
            outputs["concentrate_mass_flow"] = np.where(running, feed_mass - permeate_mass, 0.0)
            outputs["concentrate_salt_mass_flow"] = np.where(
                running,
                feed_mass * point.feed_salt_mass_fraction
                - permeate_mass * point.permeate_salt_mass_fraction,
                0.0,
            )

        outputs["water_out"] = water_out

        hours_per_timestep = self.dt / 3600  # rates are per hour, so scale by timestep length
        outputs["total_water_produced"] = water_out.sum() * hours_per_timestep
        outputs["rated_water_production"] = capacity
        outputs["capacity_factor"] = outputs["total_water_produced"] / (
            capacity * self.n_timesteps * hours_per_timestep
        )
        outputs["annual_water_produced"] = outputs["total_water_produced"] / (
            self.fraction_of_year_simulated
        )

        outputs["mass"] = capacity * self.config.mass_per_capacity
        outputs["footprint"] = capacity * self.config.footprint_per_capacity


@define(kw_only=True)
class ReverseOsmosisCostModelConfig(CostModelBaseConfig):
    """Configuration class for the ReverseOsmosisCostModel.

    Args:
        freshwater_capacity_kg_per_hr (float): Desalination plant capacity, represented as
            the maximum freshwater production rate [kg/h].
        cost_year (int): Dollar year of the costs. Only 2013 is supported.

    See :class:`ReverseOsmosisCostModel` and :doc:`/technology_models/desalination`.
    """

    freshwater_capacity_kg_per_hr: float = field(validator=validators.gt(0))
    cost_year: int = field(default=2013, converter=int, validator=validators.in_([2013]))


@register
class ReverseOsmosisCostModel(CostModelBaseClass):
    """Reverse osmosis desalination cost model.

    Inputs:
        rated_freshwater_production_capacity (float): Maximum freshwater production rate [kg/s].

    Outputs:
        CapEx (float): Capital expenditure [USD].
        OpEx (float): Fixed operating expenditure [USD/year].

    See :class:`ReverseOsmosisCostModelConfig` and :doc:`/technology_models/desalination`.
    """

    _time_step_bounds = (
        1,
        3600,
    )  # (min, max) time step lengths (in seconds) compatible with this model

    def setup(self):
        self.config = ReverseOsmosisCostModelConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "cost"),
            additional_cls_name=self.__class__.__name__,
        )
        super().setup()
        self.add_input(
            "rated_freshwater_production_capacity",
            val=self.config.freshwater_capacity_kg_per_hr / 3600,
            units="kg/s",
            desc="Maximum freshwater production rate",
        )

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        """Cost reference: Table 3 of https://www.nlr.gov/docs/fy16osti/66073.pdf.
        CapEx includes 2.55% financing factor, numbers based on INL report
        https://doi.org/10.2172/1236837 (in Table 10) which came from this report
        https://www-pub.iaea.org/MTCD/Publications/PDF/te_1561_web.pdf
        (Table 6 says 2006 is the currency reference year)
        """
        capacity = inputs["rated_freshwater_production_capacity"]

        outputs["CapEx"] = 32894 * capacity  # USD
        outputs["OpEx"] = 4841 * capacity  # USD/year
