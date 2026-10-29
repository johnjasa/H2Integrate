"""Build the example's H2I sizing problem using its actual plant models."""

from pathlib import Path

from h2integrate import H2IntegrateModel, load_tech_yaml, load_plant_yaml, load_driver_yaml


def build_sizing_problem(example_dir: Path, optimizer: str = "SLSQP") -> H2IntegrateModel:
    """Build the full H2I optimization with a fixed, discrete electrolyzer design.

    Args:
        example_dir: Directory containing the example configurations.
        optimizer: ``SLSQP`` or ``COBYLA``.

    Returns:
        The initialized H2I model. Call ``setup`` before setting inputs or evaluating totals.
    """
    if optimizer not in ("SLSQP", "COBYLA"):
        raise ValueError(f"Unsupported optimizer {optimizer!r}")
    tech = load_tech_yaml(example_dir / "tech_config.yaml")
    plant = load_plant_yaml(example_dir / "plant_config.yaml")
    driver = load_driver_yaml(example_dir / "driver_config_optimization.yaml")
    driver["driver"]["optimization"]["solver"] = optimizer
    if optimizer == "COBYLA":
        driver["driver"]["optimization"]["rhobeg"] = 0.2
        driver["driver"]["optimization"]["catol"] = 1e-5
    driver["general"]["folder_output"] = str(example_dir / "outputs")
    tech["technologies"]["distributed_wind_plant"]["model_inputs"]["performance_parameters"][
        "cache_dir"
    ] = example_dir / "cache"
    return H2IntegrateModel(
        {"plant_config": plant, "technology_config": tech, "driver_config": driver}
    )


def set_fixed_demand(model: H2IntegrateModel) -> None:
    """Set the full-horizon load and battery setpoint for the fixed 60 MW electrolyzer."""
    model.prob.set_val("battery.electricity_set_point", 6.0, units="MW")
    model.prob.set_val("elec_load_demand.electricity_demand", 60.0, units="MW")


def set_initial_sizing_design(model: H2IntegrateModel) -> None:
    """Start the sizing optimization at a feasible solar and battery design."""
    model.prob.set_val("distributed_pv.system_capacity_DC", 90000.0, units="kW")
    model.prob.set_val("battery.storage_capacity", 20.0, units="MW*h")
    model.prob.set_val("battery.max_charge_rate", 2.1, units="MW")
