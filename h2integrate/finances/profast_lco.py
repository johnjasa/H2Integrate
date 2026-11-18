import csv
import warnings
from pathlib import Path

import numpy as np
from openmdao.core.group import Group
from openmdao.utils.units import simplify_unit
from openmdao.core.analysis_error import AnalysisError
from openmdao.solvers.nonlinear.nonlinear_runonce import NonlinearRunOnce

from h2integrate.finances.tools import (
    _compute_price_units,
    validate_finance_inputs,
    check_minimum_annual_production,
)
from h2integrate.core.dict_utils import dict_to_yaml_formatting
from h2integrate.tools.profast_tools import (
    run_profast,
    convert_pf_to_dict,
    make_price_breakdown,
    format_profast_price_breakdown_per_year,
)
from h2integrate.core.supported_models import register
from h2integrate.core.inputs.validation import write_yaml
from h2integrate.finances.profast_baseclass import ProFastBase


@register
class ProFastLCO(ProFastBase):
    """Calculate the Levelized Cost of Commodity (LCO) with ProFAST.

    This component estimates the levelized cost of user-defined commodities such as
    hydrogen (LCOH), electricity (LCOE), ammonia (LCOA), nitrogen (LCON), or CO₂ (LCOC).
    It can output scalar results (LCO, IRR, WACC) and detailed cost breakdowns. Optionally,
    ProFAST inputs, configurations, and results can be exported to YAML and CSV files.

    See :class:`~h2integrate.finances.profast_baseclass.ProFastBase` and
    :doc:`/finance_models/ProFastComp`.

    Attributes:
        LCO_str (str): Name of the primary output variable (e.g., "LCOH").
        output_txt (str): Text label used for output naming, typically based on the commodity type.
        outputs_to_units (dict): Mapping of output variable names to their physical units.
        lco_units (str): Units of the LCO output, dependent on the commodity type (e.g., USD/kg or
            USD/kWh).

    Outputs:
        LCOx (float): Levelized cost of the commodity, where ``x`` corresponds to the first letter
            of the commodity (e.g., LCOH for hydrogen). Units depend on commodity type.
        wacc_<commodity> (float): Weighted average cost of capital, as a fraction.
        crf_<commodity> (float): Capital recovery factor, as a fraction.
        irr_<commodity> (float): Internal rate of return, as a fraction.
        profit_index_<commodity> (float): Profitability index, dimensionless.
        investor_payback_period_<commodity> (float): Time until the initial investment is
            recovered (years).
        price_<commodity> (float): First-year selling price of the commodity in the same units
            as LCOx.
        ``<LCOx>_breakdown`` (dict): Annualized breakdown of LCO costs by category.
        ``<LCOx>_valid`` (bool): Whether the LCO and associated results are valid.
        ``<LCOx>_status`` (str): Validity status for the current model evaluation.

    Notes:

        - Outputs and file exports are governed by user-specified finance parameters in the plant
          configuration.
        - The computation relies on ``run_profast()`` for the core financial simulation and
          ``make_price_breakdown()`` for cost disaggregation.
    """

    _has_financial_validity = True

    def add_model_specific_outputs(self):
        """Define output variables specific to the selected commodity.

        Constructs standardized names for the Levelized Cost of Commodity (LCO) and its
        associated financial metrics (WACC, CRF, IRR, etc.), based on the commodity type
        and optional user-specified description.

        Returns:
            None
        """
        # Construct output name based on commodity and optional description
        # this is necessary to allow for financial subgroups
        desc = self.options["description"].strip().strip("_()-")
        base = f"LCO{self.options['commodity_type'][0].upper()}"
        self.LCO_str = f"{base}_{desc}" if desc else base

        self.add_output(self.LCO_str, val=0.0, compute_units=_compute_price_units)
        self.add_output(f"price_{self.output_txt}", val=0.0, compute_units=_compute_price_units)

        self.outputs_to_units = {
            "wacc": "percent",
            "crf": "percent",
            "irr": "percent",
            "profit_index": "unitless",
            "investor_payback_period": "yr",
        }
        for output_var, units in self.outputs_to_units.items():
            self.add_output(f"{output_var}_{self.output_txt}", val=0.0, units=units)

        self.add_discrete_output(f"{self.LCO_str}_breakdown", val={}, desc="LCO Breakdown of costs")
        self.add_discrete_output(f"{self.LCO_str}_valid", val=False)
        self.add_discrete_output(f"{self.LCO_str}_status", val="not_evaluated")
        return

    def invalidate_financial_result(
        self,
        status: str = "not_evaluated",
        *,
        preserve_specific: bool = False,
        reason: str = "Finance evaluation has not completed",
    ) -> None:
        """Invalidate the LCO and replace any exported results with its status."""
        outputs, discrete_outputs = self._outputs, self._discrete_outputs
        if preserve_specific and discrete_outputs[f"{self.LCO_str}_status"] in (
            "undefined_zero_production",
            "production_requirement_violated",
        ):
            return
        outputs[self.LCO_str] = np.nan
        outputs[f"price_{self.output_txt}"] = np.nan
        for output_var in self.outputs_to_units:
            outputs[f"{output_var}_{self.output_txt}"] = np.nan
        discrete_outputs[f"{self.LCO_str}_breakdown"] = {}
        discrete_outputs[f"{self.LCO_str}_valid"] = False
        discrete_outputs[f"{self.LCO_str}_status"] = status

        model_inputs = self.options["plant_config"]["finance_parameters"]["model_inputs"]
        save_results = model_inputs.get("save_profast_results", False)
        save_config = model_inputs.get("save_profast_config", False)
        if not (save_results or save_config):
            return

        output_dir = Path(self.options["driver_config"]["general"]["folder_output"])
        output_dir.mkdir(parents=True, exist_ok=True)
        description = model_inputs.get("profast_output_description", "ProFastLCO")
        basename = f"{description}_{self.output_txt}"
        unavailable = {"status": status, "reason": reason}
        if save_config:
            write_yaml(unavailable, output_dir / f"{basename}_config.yaml")
        if save_results:
            write_yaml(unavailable, output_dir / f"{basename}_LCO_breakdown.yaml")
            with (output_dir / f"{basename}_profast_price_breakdown.csv").open(
                "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(file, fieldnames=["status", "reason"])
                writer.writeheader()
                writer.writerow(unavailable)

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        """Run ProFAST simulation and populate outputs.

        Executes a ProFAST financial simulation using model inputs to calculate the
        Levelized Cost of Commodity (LCO) and related economic metrics. Generates cost
        breakdown dictionaries and optionally exports configuration and results to YAML
        or CSV files, depending on user options.

        Args:
            inputs (dict): Continuous model inputs.
            outputs (dict): Continuous model outputs to be populated.
            discrete_inputs (dict): Discrete model inputs (e.g., configuration flags).
            discrete_outputs (dict): Discrete outputs for structured results.

        Returns:
            None
        """

        # Clear prior outputs and exports before any calculation can fail
        self.invalidate_financial_result("analysis_failed")

        io_meta_data = self.get_io_metadata()
        self.price_units = io_meta_data[self.LCO_str]["units"]
        self.commodity_amount_units = simplify_unit(f"USD/({self.price_units})")

        commodity = self.options["commodity_type"]
        has_production = validate_finance_inputs(inputs, commodity, self.pathname)
        capacity_units = io_meta_data[f"rated_{commodity}_production"]["units"]
        violation = check_minimum_annual_production(
            inputs, commodity, capacity_units, self.production_requirement
        )
        if violation:
            message = f"{self.pathname}: {violation}"
            self.invalidate_financial_result("production_requirement_violated", reason=message)
            raise AnalysisError(message)

        if not has_production:
            status = "undefined_zero_production"
            message = f"{self.pathname}: {self.LCO_str} is undefined because of zero production"
            self.invalidate_financial_result(status, reason=message)
            model = self._problem_meta["model_ref"]()
            # Do not let an undefined LCO enter a solve, response, or downstream component
            iterative = any(
                (not group.pathname or self.pathname.startswith(f"{group.pathname}."))
                and not isinstance(group.nonlinear_solver, NonlinearRunOnce)
                for group in model.system_iter(include_self=True, recurse=True, typ=Group)
            )
            result_names = {
                f"{self.pathname}.{name}"
                for name in (
                    self.LCO_str,
                    f"price_{self.output_txt}",
                    f"{self.LCO_str}_breakdown",
                    *(f"{name}_{self.output_txt}" for name in self.outputs_to_units),
                )
            }
            connected = any(
                source in result_names for source in model._conn_global_abs_in2out.values()
            )
            response = any(
                item["source"] in result_names for item in model.get_responses().values()
            )
            if iterative or connected or response:
                raise AnalysisError(
                    f"{message}; a driver response or downstream component requires this result"
                )
            warnings.warn(message, UserWarning, stacklevel=2)
            return

        pf = self.populate_profast(inputs)
        sol, _summary, price_breakdown = run_profast(pf)
        if not np.isfinite(sol["lco"]) or not np.isfinite(sol["price"]):
            raise AnalysisError(f"{self.pathname}: ProFAST returned a nonfinite LCO or price")
        pf_config_dict = convert_pf_to_dict(pf)
        lco_breakdown, _lco_check = make_price_breakdown(price_breakdown, pf_config_dict)

        # Check whether to export profast object to .yaml file
        save_results = self.options["plant_config"]["finance_parameters"]["model_inputs"].get(
            "save_profast_results", False
        )
        save_config = self.options["plant_config"]["finance_parameters"]["model_inputs"].get(
            "save_profast_config", False
        )

        if save_results or save_config:
            output_dir = self.options["driver_config"]["general"]["folder_output"]
            fdesc = self.options["plant_config"]["finance_parameters"]["model_inputs"].get(
                "profast_output_description", "ProFastLCO"
            )

            fbasename = f"{fdesc}_{self.output_txt}"

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            pf_config_dict = dict_to_yaml_formatting(pf_config_dict)

            if save_config:
                config_fpath = Path(output_dir) / f"{fbasename}_config.yaml"
                write_yaml(pf_config_dict, config_fpath)

            if save_results:
                price_breakdown_formatted = format_profast_price_breakdown_per_year(price_breakdown)
                pf_breakdown_fpath = Path(output_dir) / f"{fbasename}_profast_price_breakdown.csv"
                lco_breakdown_fpath = Path(output_dir) / f"{fbasename}_LCO_breakdown.yaml"
                price_breakdown_formatted.to_csv(pf_breakdown_fpath)
                lco_breakdown = dict_to_yaml_formatting(lco_breakdown)
                write_yaml(lco_breakdown, lco_breakdown_fpath)

        # Publish results only after calculation and requested exports succeed
        outputs[self.LCO_str] = sol["lco"]
        outputs[f"price_{self.output_txt}"] = sol["price"]
        for output_var in self.outputs_to_units:
            val = sol[output_var.replace("_", " ")]
            if isinstance(val, np.ndarray | list | tuple):
                val = val[-1]
            outputs[f"{output_var}_{self.output_txt}"] = val
        discrete_outputs[f"{self.LCO_str}_breakdown"] = lco_breakdown
        discrete_outputs[f"{self.LCO_str}_valid"] = True
        discrete_outputs[f"{self.LCO_str}_status"] = "valid"
