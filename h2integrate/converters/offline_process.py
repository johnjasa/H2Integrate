import re

import numpy as np
from attrs import field, define, validators
from openmdao.utils.units import convert_units
from openmdao.vectors.vector import Vector

from h2integrate.core.utilities import BaseConfig, merge_shared_inputs
from h2integrate.tools.constants import SECONDS_PER_HOUR
from h2integrate.core.model_baseclass import PerformanceModelBaseClass
from h2integrate.core.supported_models import register


PORT_NAME = re.compile(r"[a-z][a-z0-9_]*")


@define(kw_only=True)
class OfflineProcessStreamConfig(BaseConfig):
    """One fixed stream or utility rate from an offline process solve.

    Args:
        rate (float): Finite, nonnegative rate at the operating point.
        units (str): OpenMDAO rate units, such as ``kg/h`` or ``kW``.
    """

    rate: float = field(validator=validators.and_(validators.ge(0), validators.lt(float("inf"))))
    units: str = field(converter=str.strip, validator=validators.min_len(1))


def _stream_configs(
    streams: dict[str, dict | OfflineProcessStreamConfig],
) -> dict[str, OfflineProcessStreamConfig]:
    return {name: OfflineProcessStreamConfig.from_dict(stream) for name, stream in streams.items()}


@define(kw_only=True)
class OfflineProcessPerformanceModelConfig(BaseConfig):
    """Configuration for :class:`OfflineProcessPerformanceModel`.

    Args:
        source (str): Versioned reference to the offline run and its operating assumptions.
        commodity (str): Primary H2Integrate commodity produced by the process.
        product (OfflineProcessStreamConfig): Positive rate and units of the primary product.
        commodity_amount_units (str): Units of the accumulated product, such as ``kg``.
        requirements (dict): Named availability inputs and their positive operating rates.
        coproducts (dict, optional): Named secondary output streams and their operating rates.

    See :doc:`/technology_models/offline_process` for the data boundary and cost pairing.
    """

    source: str = field(converter=str.strip, validator=validators.min_len(1))
    commodity: str = field(validator=validators.matches_re(PORT_NAME))
    product: OfflineProcessStreamConfig = field(converter=OfflineProcessStreamConfig.from_dict)
    commodity_amount_units: str = field(converter=str.strip, validator=validators.min_len(1))
    requirements: dict[str, OfflineProcessStreamConfig] = field(
        converter=_stream_configs, validator=validators.min_len(1)
    )
    coproducts: dict[str, OfflineProcessStreamConfig] = field(
        factory=dict, converter=_stream_configs
    )

    def __attrs_post_init__(self) -> None:
        """Reject ambiguous port names and points that cannot produce the primary product."""
        if self.commodity in self.coproducts:
            raise ValueError("The primary commodity cannot also be a coproduct.")
        for name in self.requirements.keys() | self.coproducts.keys():
            if not isinstance(name, str) or not PORT_NAME.fullmatch(name):
                raise ValueError(f"Invalid offline process port name: {name!r}.")
        if self.product.rate <= 0 or any(stream.rate <= 0 for stream in self.requirements.values()):
            raise ValueError("The product and all requirement rates must be positive.")


@register
class OfflineProcessPerformanceModel(PerformanceModelBaseClass):
    """Run an offline process point at full rate or leave it off each timestep.

    Each requirement adds ``<name>_in`` and ``<name>_consumed`` ports. Each coproduct
    adds a ``<name>_out`` port. Rates are zero when any requirement is unavailable.
    No process solver is called. Pair with a separate cost model, such as
    :class:`~h2integrate.converters.generic_converter_cost.GenericConverterCostModel`.

    See :class:`OfflineProcessPerformanceModelConfig` and :doc:`/technology_models/offline_process`.
    """

    _time_step_bounds = (1, SECONDS_PER_HOUR)
    _control_classifier = "fixed"

    def setup(self) -> None:
        """Register fixed-point stream ports and standard H2Integrate production outputs."""
        self.config = OfflineProcessPerformanceModelConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance"),
            additional_cls_name=self.__class__.__name__,
        )
        self.commodity = self.config.commodity
        self.commodity_rate_units = self.config.product.units
        self.commodity_amount_units = self.config.commodity_amount_units
        self._product_per_hour = convert_units(
            self.config.product.rate,
            f"({self.commodity_rate_units})*h",
            self.commodity_amount_units,
        )
        super().setup()

        for name, stream in self.config.requirements.items():
            self.add_input(f"{name}_in", val=0.0, shape=self.n_timesteps, units=stream.units)
            self.add_output(f"{name}_consumed", val=0.0, shape=self.n_timesteps, units=stream.units)
        for name, stream in self.config.coproducts.items():
            self.add_output(f"{name}_out", val=0.0, shape=self.n_timesteps, units=stream.units)

    def compute(self, inputs: Vector, outputs: Vector) -> None:
        """Apply the availability mask and integrate primary production over the simulation."""
        running = np.ones(self.n_timesteps, dtype=bool)
        for name, stream in self.config.requirements.items():
            available = inputs[f"{name}_in"]
            running &= np.isfinite(available) & (available >= stream.rate)

        outputs[f"{self.commodity}_out"] = np.where(running, self.config.product.rate, 0.0)
        for name, stream in self.config.requirements.items():
            outputs[f"{name}_consumed"] = np.where(running, stream.rate, 0.0)
        for name, stream in self.config.coproducts.items():
            outputs[f"{name}_out"] = np.where(running, stream.rate, 0.0)

        total = running.sum() * (self.dt / SECONDS_PER_HOUR) * self._product_per_hour
        outputs[f"rated_{self.commodity}_production"] = self.config.product.rate
        outputs[f"total_{self.commodity}_produced"] = total
        outputs[f"annual_{self.commodity}_produced"] = total / self.fraction_of_year_simulated
        outputs["capacity_factor"] = running.mean()
