import warnings

import numpy as np
import openmdao.api as om

from h2integrate.control.control_strategies.system_level.system_level_control_baseclass import (
    price_src_indices,
)


class SLCRootGroup(om.Group):
    """Top-level model group that feeds technology prices to the system-level controller.

    The controller lists the prices it needs in ``price_inputs`` (see
    ``SystemLevelControlBase._add_price_input``). Each is connected to the named
    technology's price, which may be an input (such as a configured grid price)
    or an output (such as a price computed by a custom model). Which one exists
    is only known once the technologies are set up, so the connections are made
    in :meth:`configure`. They live on the top-level model because OpenMDAO only
    allows input-to-input connections there.
    """

    def configure(self):
        slc = getattr(self.plant, "system_level_controller", None)
        for input_name, (tech_name, price_names) in getattr(slc, "price_inputs", {}).items():
            self._connect_price(slc, input_name, tech_name, price_names)

    def _connect_price(self, slc, input_name, tech_name, price_names):
        """Connect the first of ``price_names`` found on ``tech_name`` to the controller.

        If the technology has none of them, the controller keeps its default
        price and a warning is raised.
        """
        tech = getattr(self.plant, tech_name, None)
        tech_vars = {}
        if tech is not None:
            for iotype in ("output", "input"):
                metadata = tech.get_io_metadata(
                    iotypes=iotype, metadata_keys=["val", "units", "shape"]
                )
                for meta in metadata.values():
                    tech_vars.setdefault(meta["prom_name"], (iotype, meta))

        price_name = next((name for name in price_names if name in tech_vars), None)
        if price_name is None:
            warnings.warn(
                f"The system level controller input '{input_name}' is not connected because "
                f"'{tech_name}' has no input or output named {' or '.join(price_names)}. "
                "The controller will use the price from the technology config, or zero.",
                UserWarning,
                stacklevel=2,
            )
            return

        iotype, meta = tech_vars[price_name]
        source = f"{tech_name}.{price_name}"
        if iotype == "input":
            # Both inputs share one source, so the tech's value must be the default.
            self.set_input_defaults(source, val=meta["val"], units=meta["units"])

        src_indices = None
        if meta["shape"] is not None:
            try:
                src_indices = price_src_indices(
                    int(np.prod(meta["shape"])), slc.n_timesteps, slc.plant_life
                )
            except ValueError as err:
                raise ValueError(
                    f"Cannot connect '{source}' to the system level controller: {err}"
                ) from err
        self.connect(source, f"system_level_controller.{input_name}", src_indices=src_indices)
