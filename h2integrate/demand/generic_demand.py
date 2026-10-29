import numpy as np

from h2integrate.core.utilities import merge_shared_inputs
from h2integrate.core.supported_models import register
from h2integrate.demand.demand_baseclass import DemandComponentBase, DemandComponentBaseConfig


@register
class GenericDemandComponent(DemandComponentBase):
    """Component for for converting input supply into met demand.

    This component computes unmet demand, unused (curtailed) production, and
    the resulting commodity output profile based on the incoming supply and an
    externally specified demand profile. It uses simple arithmetic rules:

    * If demand exceeds supplied commodity, the difference is unmet demand.
    * If supply exceeds demand, the excess is unused (curtailed) commodity.
    * Output equals supplied commodity minus curtailed commodity.

    This component relies on configuration provided through the
    ``tech_config`` dictionary, which must define the demand's
    ``performance_parameters``.
    """

    _is_steppable = True

    def setup(self):
        self.config = DemandComponentBaseConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance"),
            strict=True,
            additional_cls_name=self.__class__.__name__,
        )
        super().setup()
        commodity = self.commodity
        indices = np.arange(self.n_timesteps)
        vector_outputs = (
            f"{commodity}_out",
            f"unmet_{commodity}_demand_out",
            f"unused_{commodity}_out",
        )
        inputs = (f"{commodity}_in", f"{commodity}_demand")
        for output in vector_outputs:
            for input_name in inputs:
                self.declare_partials(output, input_name, rows=indices, cols=indices)
        for output in (f"total_{commodity}_produced", "percent_load_missed", "curtailment_percent"):
            for input_name in inputs:
                self.declare_partials(
                    output, input_name, rows=np.zeros(self.n_timesteps, dtype=int), cols=indices
                )
        for output in (f"annual_{commodity}_produced", "capacity_factor"):
            for input_name in inputs:
                self.declare_partials(
                    output,
                    input_name,
                    rows=np.repeat(np.arange(self.plant_life), self.n_timesteps),
                    cols=np.tile(indices, self.plant_life),
                )
        self.declare_partials(
            f"rated_{commodity}_production",
            f"{commodity}_demand",
            rows=np.zeros(self.n_timesteps, dtype=int),
            cols=indices,
            val=1.0 / self.n_timesteps,
        )

    def compute(self, inputs, outputs):
        """Compute unmet demand, unused commodity, and converter output.

        This method compares the demand profile to the supplied commodity for
        each timestep and assigns unmet demand, curtailed production, and
        actual delivered output.

        Args:
            inputs (dict-like): Mapping of input variable names to their
                current values, including:

                    * ``{commodity}_demand``: Demand profile.
                    * ``{commodity}_in``: Supplied commodity.
            outputs (dict-like): Mapping of output variable names where results
                will be written, including:

                    * ``unmet_{commodity}_demand_out``: Unmet demand.
                    * ``unused_{commodity}_out``: Curtailed production.
                    * ``{commodity}_out``: Actual output delivered.

        Notes:
            All variables operate on a per-timestep basis and typically have
            array shape ``(n_timesteps,)``.
        """

        commodity_in = inputs[f"{self.commodity}_in"]
        commodity_demand = inputs[f"{self.commodity}_demand"]
        outputs = self.calculate_outputs(commodity_in, commodity_demand, inputs, outputs)

    def compute_partials(self, inputs, partials):
        """Compute piecewise derivatives for a full-horizon demand calculation."""
        if self.n_timesteps_window != self.n_timesteps:
            raise RuntimeError(
                "GenericDemandComponent partials require a full-horizon simulation; "
                "time-marching derivatives are not supported."
            )

        commodity = self.commodity
        supply_name = f"{commodity}_in"
        demand_name = f"{commodity}_demand"
        supply = inputs[supply_name]
        demand = inputs[demand_name]
        short = (demand > supply).astype(float)
        excess = (supply > demand).astype(float)
        delivered_supply = 1.0 - excess
        delivered_demand = excess
        unmet = np.maximum(demand - supply, 0.0).sum()
        unused = np.maximum(supply - demand, 0.0).sum()
        delivered = np.minimum(supply, demand).sum()
        total_demand = demand.sum()
        total_supply = supply.sum()
        if total_demand == 0:
            raise ValueError("Demand component capacity-factor derivatives require nonzero demand.")

        for input_name, d_delivered, d_unmet, d_unused in (
            (supply_name, delivered_supply, -short, excess),
            (demand_name, delivered_demand, short, -excess),
        ):
            partials[f"{commodity}_out", input_name] = d_delivered
            partials[f"unmet_{commodity}_demand_out", input_name] = d_unmet
            partials[f"unused_{commodity}_out", input_name] = d_unused
            d_total = d_delivered * self.dt / 3600.0
            partials[f"total_{commodity}_produced", input_name] = d_total
            partials[f"annual_{commodity}_produced", input_name] = np.tile(
                d_total / self.fraction_of_year_simulated, self.plant_life
            )
            d_demand = 1.0 if input_name == demand_name else 0.0
            d_supply = 1.0 if input_name == supply_name else 0.0
            d_capacity_factor = d_delivered / total_demand - delivered * d_demand / total_demand**2
            partials["capacity_factor", input_name] = np.tile(d_capacity_factor, self.plant_life)
            partials["percent_load_missed", input_name] = (
                100.0 * (d_unmet / total_demand - unmet * d_demand / total_demand**2)
                if total_demand > 0
                else 0.0
            )
            partials["curtailment_percent", input_name] = (
                100.0 * (d_unused / total_supply - unused * d_supply / total_supply**2)
                if total_supply > 0
                else 0.0
            )
