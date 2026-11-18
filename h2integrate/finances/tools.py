import re
from typing import Any

import numpy as np
from openmdao.utils.units import _find_unit, convert_units, is_compatible, simplify_unit
from openmdao.core.analysis_error import AnalysisError


def check_minimum_annual_production(
    inputs: Any,
    commodity_type: str,
    capacity_units: str | None,
    requirement: dict | None,
) -> str | None:
    """Return a message if annual production is below the configured minimum.

    Args:
        inputs: Continuous finance inputs.
        commodity_type: Commodity whose production is checked.
        capacity_units: Units of the rated production rate.
        requirement: Minimum annual amount and units, or None.

    Returns:
        str | None: A description of the first violation, or None.

    Raises:
        ValueError: If the requirement or its units are invalid.
        AnalysisError: If annual production is nonfinite.
    """
    if requirement is None:
        return None
    value, units = requirement["value"], requirement["units"]
    if not np.isfinite(value) or value <= 0:
        raise ValueError("minimum_annual_production value must be positive and finite")
    if not isinstance(units, str) or not units.strip():
        raise ValueError("minimum_annual_production units must not be empty")
    if not capacity_units:
        raise ValueError("rated production units are required")
    try:
        capacity = convert_units(
            inputs[f"rated_{commodity_type}_production"][0], capacity_units, f"({units})/h"
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"minimum_annual_production units '{units}' are "
            f"incompatible with rated production units '{capacity_units}'"
        ) from error
    with np.errstate(over="ignore", invalid="ignore"):
        annual_production = capacity * inputs["capacity_factor"] * 8760
    if not np.all(np.isfinite(annual_production)):
        raise AnalysisError("annual production must be finite")
    deficient_years = np.flatnonzero(annual_production < value)
    if deficient_years.size:
        year = int(deficient_years[0])
        return (
            f"minimum annual production requirement is violated in operating year {year + 1}: "
            f"{annual_production[year]} {units} is below {value} {units}"
        )
    return None


def validate_finance_inputs(inputs: Any, commodity_type: str, component_path: str) -> bool:
    """Validate finance inputs and report whether any production is available.

    Args:
        inputs: Continuous OpenMDAO inputs for the financial component.
        commodity_type: Commodity whose rated production is being evaluated.
        component_path: OpenMDAO component path for error messages.

    Returns:
        bool: Whether the rated capacity and at least one capacity factor are positive.

    Raises:
        AnalysisError: If an input is nonfinite, capacity is negative, or a capacity factor
            is outside the physical range from zero to one.
    """
    for name, value in inputs.items():
        if not np.all(np.isfinite(value)):
            raise AnalysisError(f"{component_path}: finance input '{name}' must be finite")

    capacity_name = f"rated_{commodity_type}_production"
    capacity = inputs[capacity_name][0]
    if capacity < 0:
        raise AnalysisError(f"{component_path}: '{capacity_name}' must be nonnegative")

    capacity_factor = inputs["capacity_factor"]
    if np.any((capacity_factor < 0) | (capacity_factor > 1)):
        raise AnalysisError(f"{component_path}: 'capacity_factor' must be between 0 and 1")
    return bool(capacity > 0 and np.any(capacity_factor > 0))


def check_plant_config_and_profast_params(
    plant_config_dict: dict, pf_param_dict: dict, plant_config_key: str, pf_config_key: str
):
    """
    Checks for consistency between values in the plant configuration dictionary and the
    ProFAST parameters dictionary.

    This function compares the value associated with `plant_config_key` in `plant_config_dict`
    to the value associated with `pf_config_key` in `pf_param_dict`. If `pf_config_key` is not
    present in `pf_param_dict`, the value from `plant_config_dict` is used as the default.
    If the values are inconsistent, a ValueError is raised with a descriptive message.

    Args:
        plant_config_dict (dict): Dictionary containing plant configuration parameters.
        pf_param_dict (dict): Dictionary containing ProFAST parameter values.
        plant_config_key (str): Key to look up in `plant_config_dict`.
        pf_config_key (str): Key to look up in `pf_param_dict`.

    Raises:
        ValueError: If the values for the specified keys in the two dictionaries are inconsistent.
    """

    if (
        pf_param_dict.get(pf_config_key, plant_config_dict[plant_config_key])
        != plant_config_dict[plant_config_key]
    ):
        msg = (
            f"Inconsistent values provided for {pf_config_key} and {plant_config_key}, "
            f"{pf_config_key} is {pf_param_dict.get(pf_config_key)} but "
            f"{plant_config_key} is {plant_config_dict[plant_config_key]}."
            f"Please check that {pf_config_key} is the same as {plant_config_key} or remove "
            f"{pf_config_key} from pf_params input."
        )
        raise ValueError(msg)


def _compute_price_units(outputs):
    """Derive the OpenMDAO unit for a commodity price from an output dictionary.

    This helper is intended to be passed as the ``compute_units`` callback of
    ``om.ExplicitComponent.add_output``. It inspects the component's already-declared
    outputs to find the unit of the rated commodity production, then constructs the
    matching price unit as ``USD / (rate_unit * h)``.

    The function searches ``outputs`` for the first key matching the regex
    ``rated_\\w+_production`` (e.g. ``rated_hydrogen_production``,
    ``rated_electricity_production``). The unit string of the matched output is
    then used as ``rate_unit``, and the resulting price unit
    ``USD/(rate_unit*h)`` is resolved into an OpenMDAO unit object via
    ``openmdao.utils.units._find_unit``.

    Example:
        If ``outputs`` contains ``rated_hydrogen_production`` declared with units
        ``kg/h``, this function returns the OpenMDAO unit for ``USD/((kg/h)*h)``,
        which simplifies to ``USD/kg``.

    Args:
        outputs (dict): Mapping of output name to OpenMDAO output metadata object
            (typically the ``outputs`` dictionary passed to a ``compute_units``
            callback). Each value must expose a ``name()`` method returning the
            unit string of that output.

    Returns:
        The OpenMDAO unit object corresponding to ``USD/(rate_unit*h)``, suitable
        for use as the unit of a price-valued output.

    Raises:
        ValueError: If no ``rated_*_production`` output is present in ``outputs``
            to derive the rate unit from.
    """
    rate_units = [v.name() for k, v in outputs.items() if re.fullmatch(r"rated_\w+_production", k)]
    if len(rate_units) == 0:
        raise ValueError("Cannot find rate units")
    commodity_amount_units = f"({rate_units[0]})*h"
    price_units = f"USD/({commodity_amount_units})"
    return _find_unit(price_units)


def _compute_rate_units(price_units: str, check_conversion: bool):
    """Estimate rate units from price units.

    Args:
        price_units (str): Price units. Should be defined as USD per some commodity amount.
        check_conversion (bool): Whether to check for a conversion factor of 1.0 from price_units
            to price units re-calculated from the estimated rate_units

    Raises:
        ValueError: if the `rate_units` cannot be easily estimated from the `price_units`

    Returns:
        str: rate units extrapolated from `price_units`.
    """

    # A price has units of USD/amount, and an amount equals a rate times a time.
    # Therefore rate_units = USD / (price_units * h), which OpenMDAO can simplify
    # directly.
    rate_units = simplify_unit(f"USD/(({price_units})*h)")

    if not check_conversion:
        return rate_units
    equivalent_price_units = simplify_unit(f"USD/(({rate_units})*h)")

    conversion = convert_units(1, price_units, equivalent_price_units)
    if is_compatible(price_units, equivalent_price_units) and float(conversion) == 1.0:
        return rate_units

    msg = (
        f"Estimated rate units of '{rate_units}' from price units of '{price_units}'."
        f"Try to use price units of '{equivalent_price_units}' instead."
    )
    raise ValueError(msg)
