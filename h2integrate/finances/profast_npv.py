from collections.abc import Iterable

import numpy as np
from openmdao.utils.units import convert_units, simplify_unit
from openmdao.core.analysis_error import AnalysisError

from h2integrate.finances.tools import (
    _compute_rate_units,
    validate_finance_inputs,
    check_minimum_annual_production,
)
from h2integrate.core.supported_models import register
from h2integrate.finances.profast_baseclass import ProFastBase


@register
class ProFastNPV(ProFastBase):
    """Calculates the Net Present Value (NPV) of a commodity using ProFAST.

    This component extends `ProFastBase` to compute the NPV based on the user-defined
    commodity and its sell price. The NPV output reflects the present value of future
    cash flows, given the financial configuration of the plant.

    See :class:`~h2integrate.finances.profast_baseclass.ProFastBase` and
    :doc:`/finance_models/ProFastNPV`.

    Attributes:
        output_txt (str): Label used for naming outputs based on commodity type.
        lco_units (str): Units for pricing inputs (USD/kg or USD/kWh, depending on commodity).

    Outputs:
        NPV_<commodity> (float): Net Present Value of the commodity in USD.
        NPV_<commodity>_valid (bool): Whether NPV is valid for the current evaluation.
        NPV_<commodity>_status (str): Validity status for the current model evaluation.
    """

    _has_financial_validity = True

    def add_model_specific_outputs(self):
        """Define NPV output variable for the model.

        Creates an output variable named `NPV_<commodity>` in USD.

        Returns:
            None
        """
        self.add_output(
            f"NPV_{self.output_txt}",
            val=0.0,
            units="USD",
        )
        self.add_discrete_output(f"NPV_{self.output_txt}_valid", val=False)
        self.add_discrete_output(f"NPV_{self.output_txt}_status", val="not_evaluated")

        return

    def setup(self):
        """Set up inputs for the NPV calculation.

        Retrieves the commodity sell price and its units from the plant configuration
        and registers it as an input for the component. Calls the base `setup()` method
        to initialize other ProFAST inputs and outputs. The commodity sell price can be
        either a scalar value or a list with the same length as plant life.

        Raises:
            ValueError: If `commodity_sell_price` or `commodity_sell_price_units` is not
                provided in the configuration.

        Returns:
            None
        """
        model_inputs = self.options["plant_config"]["finance_parameters"]["model_inputs"]
        self.commodity_sell_price = model_inputs.get("commodity_sell_price", None)
        self.commodity_sell_price_units = model_inputs.get("commodity_sell_price_units", None)

        if self.commodity_sell_price_units is None:
            raise ValueError(
                "commodity_sell_price_units is missing as an input. "
                "ProFastNPV requires the user to specify the units of "
                "commodity_sell_price explicitly in "
                "plant_config['finance_parameters']['model_inputs']."
            )

        super().setup()

        if self.commodity_sell_price is None:
            raise ValueError("commodity_sell_price is missing as an input")

        if isinstance(self.commodity_sell_price, Iterable):
            if len(self.commodity_sell_price) != int(self.params.plant_life):
                raise ValueError(
                    f"`commodity_sell_price` has an invalid length of "
                    f"{len(self.commodity_sell_price)}. `commodity_sell_price`"
                    f"must be the same length as the plant life ({self.params.plant_life}) "
                    "or a single value."
                )

        self.add_input(
            f"sell_price_{self.output_txt}",
            val=self.commodity_sell_price,
            shape=int(self.params.plant_life),
            units=self.commodity_sell_price_units,
        )

    def invalidate_financial_result(
        self, status: str = "not_evaluated", *, preserve_specific: bool = False
    ) -> None:
        """Mark the previous NPV unavailable for the current evaluation."""
        result_name = f"NPV_{self.output_txt}"
        if preserve_specific and self._discrete_outputs[f"{result_name}_status"] in (
            "unsupported_zero_production",
            "production_requirement_violated",
        ):
            return
        self._outputs[result_name] = np.nan
        self._discrete_outputs[f"{result_name}_valid"] = False
        self._discrete_outputs[f"{result_name}_status"] = status

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        """Compute the NPV of the commodity using ProFAST cash flows.

        Args:
            inputs (dict): Model inputs, including `sell_price_<commodity>`.
            outputs (dict): Model outputs to populate with NPV results.
            discrete_inputs (dict): Discrete component inputs.
            discrete_outputs (dict): NPV validity and status outputs.

        Returns:
            None
        """
        result_name = f"NPV_{self.output_txt}"
        # Clear the previous result before validating or running ProFAST
        self.invalidate_financial_result("analysis_failed")
        commodity = self.options["commodity_type"]
        has_production = validate_finance_inputs(inputs, commodity, self.pathname)
        io_meta_data = self.get_io_metadata()
        capacity_units = io_meta_data[f"rated_{commodity}_production"]["units"]
        violation = check_minimum_annual_production(
            inputs, commodity, capacity_units, self.production_requirement
        )
        if violation:
            discrete_outputs[f"{result_name}_status"] = "production_requirement_violated"
            raise AnalysisError(f"{self.pathname}: {violation}")
        if not has_production:
            discrete_outputs[f"{result_name}_status"] = "unsupported_zero_production"
            raise AnalysisError(
                f"{self.pathname}: ProFAST NPV for zero production is not supported; "
                "its cash flow cannot be calculated faithfully with zero capacity or utilization"
            )

        self.price_units = io_meta_data[f"sell_price_{self.output_txt}"]["units"]
        self.commodity_amount_units = simplify_unit(f"USD/({self.price_units})")

        # compute rate_units from the price
        rate_units_from_price = _compute_rate_units(
            self.commodity_sell_price_units, check_conversion=False
        )
        rate_units_capacity = io_meta_data[f"rated_{self.options['commodity_type']}_production"][
            "units"
        ]
        conversion_ratio = convert_units(1, rate_units_from_price, rate_units_capacity)

        # ensure that sell price units are compatible with the rate units
        if float(conversion_ratio) != 1.0:
            # convert rate units to units compatible with the price_units
            inputs_adjusted = dict(inputs.items())
            capacity_converted = convert_units(
                inputs[f"rated_{self.options['commodity_type']}_production"],
                rate_units_capacity,
                rate_units_from_price,
            )
            inputs_adjusted[f"rated_{self.options['commodity_type']}_production"] = (
                capacity_converted
            )
            pf = self.populate_profast(inputs_adjusted)
        else:
            pf = self.populate_profast(inputs)

        non_op_Nyears = int(np.ceil(self.params.installation_time / 12) + 1)
        sell_price = inputs[f"sell_price_{self.output_txt}"]
        # Use first operating-year price for pre-operation padding.
        # ProFAST needs the non-zero price padding when installation time is not divisible by 12.
        sell_profile = np.concatenate([np.full(non_op_Nyears, sell_price[0]), sell_price])

        npv = pf.cash_flow(price=sell_profile)
        if not np.isfinite(npv):
            raise AnalysisError(f"{self.pathname}: ProFAST returned a nonfinite NPV")
        outputs[result_name] = npv
        discrete_outputs[f"{result_name}_valid"] = True
        discrete_outputs[f"{result_name}_status"] = "valid"
