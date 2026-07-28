import numpy as np
import PySAM.Geothermal as Geothermal
from attrs import field, define

from h2integrate.core.utilities import BaseConfig, merge_shared_inputs
from h2integrate.core.validators import gt_zero, contains, range_val
from h2integrate.converters.geothermal.geothermal_baseclass import GeothermalPerformanceBaseClass


def estimate_wet_bulb_temperature(dry_bulb_temp_C, relative_humidity_pct):
    """Estimate the wet-bulb temperature from dry-bulb temperature and relative humidity.

    Uses the empirical approximation from Stull (2011), which is accurate to within about
    1 degree Celsius for typical atmospheric conditions.

    Args:
        dry_bulb_temp_C (float): Dry-bulb (ambient) temperature in degrees Celsius.
        relative_humidity_pct (float): Relative humidity as a percentage in the range (0, 100].

    Returns:
        float: Estimated wet-bulb temperature in degrees Celsius.
    """
    t = dry_bulb_temp_C
    rh = np.clip(relative_humidity_pct, 1.0, 100.0)
    tw = (
        t * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(t + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * (rh**1.5) * np.arctan(0.023101 * rh)
        - 4.686035
    )
    return float(tw)


@define(kw_only=True)
class PYSAMGeothermalPlantPerformanceModelConfig(BaseConfig):
    """Configuration class for the PYSAMGeothermalPlantPerformanceModel which uses the
    Geothermal module (GETEM, the Geothermal Electricity Technology Evaluation Model)
    available in PySAM. PySAM documentation can be found
    `here <https://nrel-pysam.readthedocs.io/en/main/modules/Geothermal.html>`__ and
    additional information on GETEM can be found
    `here <https://www.energy.gov/hgeo/geothermal/geothermal-electricity-technology-evaluation-model>`__.

    The surface ambient conditions used to set the power-cycle design point (dry-bulb
    temperature, humidity, and pressure) are taken from a connected weather resource model
    rather than from a weather file, following the same resource-handling pattern used by
    the solar and wind PySAM models.

    Attributes:
        nameplate_kW (float): Required, desired plant nameplate (net) output in kW.
        resource_temp_C (float): Required, geothermal resource temperature in degrees Celsius.
            Must be in the range [0, 373].
        resource_depth_m (float): Required, geothermal resource depth in meters.
        resource_type (int): Type of geothermal resource. 0 = hydrothermal (default),
            1 = enhanced geothermal system (EGS).
        conversion_type (int): Power conversion cycle type. 0 = binary (default), 1 = flash.
        analysis_type (int): Sizing basis. 0 = specify the plant nameplate output (default),
            1 = specify the number of production wells.
        num_wells (float): Number of production wells. Only used when ``analysis_type`` is 1.
        create_model_from (str):
            - 'default': instantiate Geothermal model from the default config 'config_name'
              (default).
            - 'new': instantiate a new Geothermal model. Requires pysam_options.
        config_name (str): PySAM.Geothermal configuration name. Defaults to
            'GeothermalPowerSingleOwner'. Only used if create_model_from='default'.
        pysam_options (dict): dictionary of Geothermal input parameters with top-level keys
            corresponding to the different Geothermal variable groups ('GeoHourly',
            'AdjustmentFactors'). Please refer to the Geothermal documentation
            `here <https://nrel-pysam.readthedocs.io/en/main/modules/Geothermal.html>`__.
    """

    nameplate_kW: float = field(validator=gt_zero)
    resource_temp_C: float = field(validator=range_val(0.0, 373.0))
    resource_depth_m: float = field(validator=gt_zero)

    resource_type: int = field(default=0, converter=int, validator=contains([0, 1]))
    conversion_type: int = field(default=0, converter=int, validator=contains([0, 1]))
    analysis_type: int = field(default=0, converter=int, validator=contains([0, 1]))
    num_wells: float = field(default=2.0, validator=gt_zero)

    create_model_from: str = field(
        default="default",
        validator=contains(["default", "new"]),
        converter=(str.strip, str.lower),
    )
    config_name: str = field(
        default="GeothermalPowerSingleOwner",
        validator=contains(
            [
                "GeothermalPowerAllEquityPartnershipFlip",
                "GeothermalPowerLCOECalculator",
                "GeothermalPowerLeveragedPartnershipFlip",
                "GeothermalPowerMerchantPlant",
                "GeothermalPowerNone",
                "GeothermalPowerSaleLeaseback",
                "GeothermalPowerSingleOwner",
            ]
        ),
    )
    pysam_options: dict = field(default={})

    def __attrs_post_init__(self):
        if self.create_model_from == "new" and not bool(self.pysam_options):
            msg = (
                "To create a new Geothermal object, please provide a dictionary "
                "of Geothermal design variables for the 'pysam_options' key."
            )
            raise ValueError(msg)

        self.check_pysam_options()

    def check_pysam_options(self):
        """Checks that the top-level keys of the pysam_options dictionary are valid groups
        and that parameters managed directly by this model are not duplicated in pysam_options.

        Raises:
            ValueError: if top-level keys of pysam_options are not valid groups.
            ValueError: if a parameter managed by this model is provided in
                pysam_options['GeoHourly'].
        """
        valid_groups = ["GeoHourly", "AdjustmentFactors"]
        managed_params = [
            "resource_temp",
            "resource_depth",
            "resource_type",
            "conversion_type",
            "analysis_type",
            "nameplate",
            "num_wells",
            "geothermal_analysis_period",
            "system_use_lifetime_output",
            "ui_calculations_only",
            "use_weather_file_conditions",
            "design_temp",
            "wet_bulb_temp",
            "ambient_pressure",
        ]
        if bool(self.pysam_options):
            invalid_groups = [k for k in self.pysam_options if k not in valid_groups]
            if len(invalid_groups) > 0:
                msg = (
                    f"Invalid group(s) found in pysam_options: {invalid_groups}. "
                    f"Valid groups are: {valid_groups}."
                )
                raise ValueError(msg)

            geo_options = self.pysam_options.get("GeoHourly", {})
            duplicated = [p for p in managed_params if p in geo_options]
            if duplicated:
                msg = (
                    f"The following parameters are managed by this model and should not be set "
                    f"in pysam_options['GeoHourly']: {duplicated}. Please set them using the "
                    "dedicated performance parameters or the connected resource model instead."
                )
                raise ValueError(msg)
        return


class PYSAMGeothermalPlantPerformanceModel(GeothermalPerformanceBaseClass):
    """
    An OpenMDAO component that wraps a GETEM geothermal power plant model via PySAM.

    It takes geothermal resource and plant parameters as input and outputs electricity
    generation data. Geothermal is modeled as a fixed (baseload) generator.
    """

    _time_step_bounds = (
        3600,
        3600,
    )  # (min, max) time step lengths (in seconds) compatible with this model

    def setup(self):
        super().setup()

        self.config = PYSAMGeothermalPlantPerformanceModelConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance"),
            additional_cls_name=self.__class__.__name__,
        )

        self.add_input(
            "nameplate",
            val=self.config.nameplate_kW,
            units="kW",
            desc="Desired geothermal plant nameplate (net) output",
        )
        self.add_output(
            "system_capacity_AC",
            val=0.0,
            units="kW",
            desc="Geothermal plant net rated capacity",
        )

        if self.config.create_model_from == "default":
            self.system_model = Geothermal.default(self.config.config_name)
        elif self.config.create_model_from == "new":
            self.system_model = Geothermal.new()

        # Apply any user-provided PySAM options first so that the parameters managed
        # by this model (assigned below) take precedence.
        if bool(self.config.pysam_options):
            self.system_model.assign(self.config.pysam_options)

    def ambient_conditions_from_resource(self, resource_data):
        """Derive the power-cycle design ambient conditions from the connected resource data.

        The annual-mean dry-bulb temperature is used as the power-block design temperature.
        The wet-bulb temperature is estimated from the mean relative humidity when available,
        and the ambient pressure is converted from the resource data when available.

        Args:
            resource_data (dict): Weather resource data dictionary, expected to contain a
                ``temperature`` array (degrees Celsius) and optionally ``relative_humidity``
                (percent) and ``pressure`` (millibar).

        Returns:
            3-element tuple containing

            - **design_temp_C** (*float*): mean dry-bulb temperature in degrees Celsius.
            - **wet_bulb_temp_C** (*float*): estimated wet-bulb temperature in degrees Celsius.
            - **ambient_pressure_psi** (*float*): mean ambient pressure in psi.
        """
        temperature = np.asarray(resource_data.get("temperature", []), dtype=float)
        if temperature.size == 0:
            msg = (
                "The geothermal performance model requires ambient 'temperature' data from a "
                "connected weather resource model. Please connect a resource model to this "
                "technology via 'resource_to_tech_connections' in the plant config."
            )
            raise ValueError(msg)
        design_temp_C = float(np.mean(temperature))

        relative_humidity = np.asarray(resource_data.get("relative_humidity", []), dtype=float)
        if relative_humidity.size > 0:
            wet_bulb_temp_C = estimate_wet_bulb_temperature(
                design_temp_C, float(np.mean(relative_humidity))
            )
        else:
            # Without humidity data, assume saturated air (wet bulb equals dry bulb).
            wet_bulb_temp_C = design_temp_C

        pressure_mbar = np.asarray(resource_data.get("pressure", []), dtype=float)
        if pressure_mbar.size > 0:
            # Convert from millibar to psi (1 mbar = 0.0145038 psi).
            ambient_pressure_psi = float(np.mean(pressure_mbar)) * 0.0145037738
        else:
            # Standard sea-level atmospheric pressure.
            ambient_pressure_psi = 14.6959

        return design_temp_C, wet_bulb_temp_C, ambient_pressure_psi

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        model = self.system_model

        # Derive the power-cycle ambient design conditions from the connected resource model.
        design_temp_C, wet_bulb_temp_C, ambient_pressure_psi = (
            self.ambient_conditions_from_resource(discrete_inputs["solar_resource_data"])
        )

        # Assign the subsurface resource and plant design parameters
        model.value("resource_temp", self.config.resource_temp_C)
        model.value("resource_depth", self.config.resource_depth_m)
        model.value("resource_type", self.config.resource_type)
        model.value("conversion_type", self.config.conversion_type)
        model.value("analysis_type", self.config.analysis_type)
        model.value("nameplate", inputs["nameplate"][0])
        model.value("num_wells", self.config.num_wells)

        # Set the surface ambient conditions from the connected resource data rather than
        # reading them from a weather file.
        model.value("use_weather_file_conditions", 0)
        model.value("design_temp", design_temp_C)
        model.value("wet_bulb_temp", wet_bulb_temp_C)
        model.value("ambient_pressure", ambient_pressure_psi)

        model.value("geothermal_analysis_period", self.plant_life)
        model.value("system_use_lifetime_output", 0)

        # Run GETEM's design (UI) calculations only. This sizes the plant and computes the
        # gross and parasitic pump power from the subsurface resource and ambient conditions
        # without requiring an hourly weather file. Geothermal is modeled as a firm baseload
        # generator, so the net design capacity is dispatched at a constant rate.
        model.value("ui_calculations_only", 1)
        model.execute(0)

        gross_output_MW = float(model.value("gross_output"))
        pump_work_MW = float(model.value("pump_work"))
        net_capacity_kW = max(gross_output_MW - pump_work_MW, 0.0) * 1e3

        # Baseload (firm) generation profile at the net design capacity.
        electricity_out = np.full(self.n_timesteps, net_capacity_kW, dtype=float)
        outputs["electricity_out"] = electricity_out

        outputs["system_capacity_AC"] = net_capacity_kW
        outputs["rated_electricity_production"] = net_capacity_kW

        outputs["total_electricity_produced"] = electricity_out.sum() * (self.dt / 3600)

        max_production = net_capacity_kW * self.n_timesteps * (self.dt / 3600)
        if max_production > 0:
            outputs["capacity_factor"] = outputs["total_electricity_produced"] / max_production
        else:
            outputs["capacity_factor"] = 0.0

        # Annual firm energy production (kWh/year) from the constant net capacity.
        hours_per_year = 8760.0
        outputs["annual_electricity_produced"] = net_capacity_kW * hours_per_year
