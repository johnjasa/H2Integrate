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

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        """
        Computation for the OM component.

        For a template class this is not implemented and raises an error.
        """

        raise NotImplementedError("This method should be implemented in a subclass.")
