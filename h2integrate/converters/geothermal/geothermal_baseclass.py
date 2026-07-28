from h2integrate.core.model_baseclasses import PerformanceModelBaseClass


class GeothermalPerformanceBaseClass(PerformanceModelBaseClass):
    """Base class for geothermal power plant performance models.

    Geothermal plants are treated as fixed (baseload) generators: they produce
    electricity based on the subsurface resource and cannot be dispatched or
    curtailed within H2Integrate's system-level control framework.
    """

    _time_step_bounds = (
        3600,
        3600,
    )  # (min, max) time step lengths (in seconds) compatible with this model
    _control_classifier = "fixed"

    def initialize(self):
        super().initialize()
        self.commodity = "electricity"
        self.commodity_rate_units = "kW"
        self.commodity_amount_units = "kW*h"

    def setup(self):
        super().setup()

        # Ambient weather resource used to set the power-cycle design conditions.
        # The geothermal (subsurface) resource is defined by the performance
        # parameters, while this dictionary supplies the surface ambient conditions
        # (dry-bulb temperature, humidity, pressure) provided by a connected
        # resource model in the same way the solar and wind PySAM models consume
        # their resource data.
        self.add_discrete_input(
            "solar_resource_data",
            val={},
            desc="Weather resource data dictionary providing ambient conditions",
        )

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        """
        Computation for the OM component.

        For a template class this is not implemented and raises an error.
        """

        raise NotImplementedError("This method should be implemented in a subclass.")
