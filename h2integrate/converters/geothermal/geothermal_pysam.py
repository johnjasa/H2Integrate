from pathlib import Path

import numpy as np
import PySAM.Geothermal as Geothermal
from attrs import field, define

from h2integrate.core.utilities import BaseConfig, merge_shared_inputs
from h2integrate.core.validators import gt_zero, contains, range_val
from h2integrate.converters.geothermal.geothermal_baseclass import GeothermalPerformanceBaseClass


@define(kw_only=True)
class PYSAMGeothermalPlantPerformanceModelConfig(BaseConfig):
    """Configuration class for the PYSAMGeothermalPlantPerformanceModel which uses the
    Geothermal module (GETEM, the Geothermal Electricity Technology Evaluation Model)
    available in PySAM. PySAM documentation can be found
    `here <https://nrel-pysam.readthedocs.io/en/main/modules/Geothermal.html>`__ and
    additional information on GETEM can be found
    `here <https://www.energy.gov/hgeo/geothermal/geothermal-electricity-technology-evaluation-model>`__.

    Attributes:
        nameplate_kW (float): Required, desired plant nameplate (net) output in kW.
        resource_temp_C (float): Required, geothermal resource temperature in degrees Celsius.
            Must be in the range [0, 373].
        resource_depth_m (float): Required, geothermal resource depth in meters.
        weather_file (str): Required, path to the ambient weather file (TMY / PSM3 format) used
            to model the power cycle. The geothermal resource itself is defined by the subsurface
            parameters; the weather file provides the ambient conditions for the surface plant.
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
    weather_file: str = field()

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

        if not Path(self.weather_file).is_file():
            msg = (
                f"The geothermal 'weather_file' could not be found at: {self.weather_file}. "
                "Please provide a valid path to a TMY/PSM3 weather file."
            )
            raise FileNotFoundError(msg)

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
            "file_name",
            "geothermal_analysis_period",
            "system_use_lifetime_output",
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
                    "dedicated performance parameters instead."
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

    def compute(self, inputs, outputs):
        model = self.system_model

        # Assign the subsurface resource and plant design parameters
        model.value("resource_temp", self.config.resource_temp_C)
        model.value("resource_depth", self.config.resource_depth_m)
        model.value("resource_type", self.config.resource_type)
        model.value("conversion_type", self.config.conversion_type)
        model.value("analysis_type", self.config.analysis_type)
        model.value("nameplate", inputs["nameplate"][0])
        model.value("num_wells", self.config.num_wells)
        model.value("file_name", self.config.weather_file)

        # GETEM analysis period is limited to the plant life; request a single
        # representative year of hourly output (length == 8760) rather than the
        # full multi-year lifetime profile.
        model.value("geothermal_analysis_period", self.plant_life)
        model.value("system_use_lifetime_output", 0)

        # run the model
        model.execute(0)

        gen = np.asarray(model.Outputs.gen, dtype=float)

        # PySAM returns a full year (8760) of hourly generation; align it to the
        # simulation length by truncating or tiling as needed.
        if len(gen) >= self.n_timesteps:
            electricity_out = gen[: self.n_timesteps]
        else:
            repeats = int(np.ceil(self.n_timesteps / len(gen)))
            electricity_out = np.tile(gen, repeats)[: self.n_timesteps]

        outputs["electricity_out"] = electricity_out

        net_capacity_kW = float(np.max(gen)) if len(gen) > 0 else 0.0
        outputs["system_capacity_AC"] = net_capacity_kW
        outputs["rated_electricity_production"] = net_capacity_kW

        outputs["total_electricity_produced"] = electricity_out.sum() * (self.dt / 3600)

        max_production = net_capacity_kW * self.n_timesteps * (self.dt / 3600)
        if max_production > 0:
            outputs["capacity_factor"] = outputs["total_electricity_produced"] / max_production
        else:
            outputs["capacity_factor"] = 0.0

        outputs["annual_electricity_produced"] = model.value("annual_energy")
