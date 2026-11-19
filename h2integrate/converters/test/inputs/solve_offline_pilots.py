"""Generate pilot test data in an isolated IDAES environment, not the H2Integrate runtime."""

import math
import argparse
from pathlib import Path
from importlib.metadata import version

import yaml
from pyomo.environ import SolverFactory, units, value, assert_optimal_termination
from prommis.hydrogen_decrepitation import hydrogen_decrepitation_flowsheet as hd
from idaes.core.util.model_statistics import degrees_of_freedom
from watertap.flowsheets.RO_with_energy_recovery import RO_with_energy_recovery as ro


PINNED_PACKAGES = {
    "idaes-pse": "2.13.0",
    "pyomo": "6.10.1",
    "watertap": "1.8.0",
    "watertap-solvers": "24.12.9",
    "prommis": "1.1.0",
}
PERMEATE_LIMIT = 500e-6


def _rate(rate: float, rate_units: str) -> dict:
    return {"rate": rate, "units": rate_units}


def _quantity(expression, target_units) -> float:
    return float(value(units.convert(expression, to_units=target_units)))


def _solution_masses(unit) -> tuple[float, float]:
    return tuple(
        _quantity(unit.flow_mass_phase_comp[0, "Liq", component], units.kg / units.h)
        for component in ("H2O", "NaCl")
    )


def _solve_ro(erd_type: ro.ERDtype) -> dict:
    solver = ro.get_solver()
    model = ro.build(erd_type=erd_type)
    ro.set_operating_conditions(model, solver=solver)
    ro.initialize_system(model, solver=solver)
    assert_optimal_termination(ro.solve(model, solver=solver))
    fs = model.fs
    feed, product, brine = map(_solution_masses, (fs.feed, fs.product, fs.disposal))
    residuals = [
        inlet - permeate - concentrate
        for inlet, permeate, concentrate in zip(feed, product, brine, strict=True)
    ]
    if max(map(abs, residuals)) > 1e-6:
        raise ValueError(f"RO water or salt mass balance did not close: {residuals} kg/h")
    salinity = product[1] / sum(product)
    recovery = product[0] / feed[0]
    if not math.isclose(recovery, 0.5, abs_tol=1e-7) or not 0 <= salinity <= PERMEATE_LIMIT:
        raise ValueError(f"RO point does not meet the service target: {recovery=}, {salinity=}")
    feed_volume = _quantity(fs.feed.properties[0].flow_vol, units.m**3 / units.h)
    product_volume = _quantity(fs.product.properties[0].flow_vol, units.m**3 / units.h)
    pumps = (fs.P1, fs.P2) if erd_type == ro.ERDtype.pressure_exchanger else (fs.P1,)
    power = sum(_quantity(pump.work_mechanical[0], units.kW) for pump in pumps)
    return {
        "performance_parameters": {
            "source": f"WaterTAP 1.8.0 RO_with_energy_recovery 0D simulation ({erd_type.value})",
            "commodity": "water",
            "product": _rate(product_volume, "m**3/h"),
            "commodity_amount_units": "m**3",
            "requirements": {
                "feedwater": _rate(feed_volume, "m**3/h"),
                "electricity": _rate(power, "kW"),
            },
            "coproducts": {
                "concentrate": _rate(sum(brine), "kg/h"),
                "concentrate_salt": _rate(brine[1], "kg/h"),
            },
        },
        "validation": {
            "termination": "optimal",
            "feed_mass_kg_per_h": sum(feed),
            "feed_density_kg_per_m3": sum(feed) / feed_volume,
            "feed_temperature_k": value(fs.feed.properties[0].temperature),
            "feed_pressure_pa": value(fs.feed.properties[0].pressure),
            "feed_salt_mass_fraction": feed[1] / sum(feed),
            "permeate_mass_kg_per_h": sum(product),
            "permeate_density_kg_per_m3": sum(product) / product_volume,
            "permeate_salt_mass_fraction": salinity,
            "permeate_limit_mass_fraction": PERMEATE_LIMIT,
            "water_mass_recovery": recovery,
            "membrane_area_m2": _quantity(fs.RO.area, units.m**2),
            "water_residual_kg_per_h": residuals[0],
            "salt_residual_kg_per_h": residuals[1],
            "specific_pump_energy_kwh_per_m3": power / product_volume,
        },
    }


def _solve_furnace(solver) -> dict:
    model = hd.build_flowsheet()
    if degrees_of_freedom(model) != 0:
        raise ValueError("The published furnace case must be fully specified.")
    assert_optimal_termination(solver.solve(model))
    furnace = model.fs.hydrogen_decrepitation_furnace
    feed = _quantity(furnace.solid_in[0].flow_mass, units.kg / units.h)
    product = _quantity(furnace.solid_out[0].flow_mass, units.kg / units.h)
    gas_in, gas_out = (
        _quantity(state.flow_mol_comp["H2"] * state.mw, units.kg / units.h)
        for state in (furnace.gas_in[0], furnace.gas_out[0])
    )
    if not math.isclose(gas_in, gas_out, abs_tol=1e-9) or not math.isclose(
        feed + gas_in, product + gas_out, rel_tol=1e-5
    ):
        raise ValueError("The furnace boundary differs from the published reference balance.")
    return {
        "performance_parameters": {
            "source": "PrOMMiS 1.1.0 hydrogen_decrepitation_flowsheet reference furnace",
            "commodity": "decrepitated_magnet",
            "product": _rate(product, "kg/h"),
            "commodity_amount_units": "kg",
            "requirements": {
                "shredded_magnet": _rate(feed, "kg/h"),
                "hydrogen": _rate(gas_in, "kg/h"),
                "heat": _rate(_quantity(furnace.total_heat_duty[0], units.kW), "kW"),
            },
            "coproducts": {"hydrogen": _rate(gas_out, "kg/h")},
        },
        "validation": {
            "termination": "optimal",
            "operating_temperature_k": _quantity(furnace.operating_temperature, units.K),
            "feed_temperature_k": _quantity(furnace.temp_feed, units.K),
            "hydrogen_inlet_pressure_pa": _quantity(furnace.gas_in[0].pressure, units.Pa),
            "feed_nd2fe14b_mass_fraction": value(model.fs.shredder.mass_frac_comp[0, "Nd2Fe14B"]),
            "decrepitation_duration_h": _quantity(furnace.decrepitation_duration, units.h),
            "hydrogen_in_minus_out_kg_per_h": gas_in - gas_out,
            "inlet_minus_outlet_mass_kg_per_h": feed + gas_in - product - gas_out,
        },
    }


def main() -> None:
    """Solve all reference cases and export a single provenance and replay YAML file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Path to a new YAML output file")
    args = parser.parse_args()
    installed = {name: version(name) for name in PINNED_PACKAGES}
    if installed != PINNED_PACKAGES:
        raise ValueError(f"Expected pinned package versions {PINNED_PACKAGES}; found {installed}")
    solver = SolverFactory("ipopt")
    pressure_exchanger = _solve_ro(ro.ERDtype.pressure_exchanger)
    no_recovery = _solve_ro(ro.ERDtype.no_ERD)
    if not math.isclose(
        pressure_exchanger["performance_parameters"]["product"]["rate"],
        no_recovery["performance_parameters"]["product"]["rate"],
        rel_tol=1e-5,
    ):
        raise ValueError("The RO cases do not meet the same product volume target.")
    data = {
        "packages": installed,
        "ipopt_version": ".".join(map(str, solver.version())),
        "cases": {
            "watertap_pressure_exchanger": pressure_exchanger,
            "watertap_no_recovery": no_recovery,
            "prommis_decrepitation": _solve_furnace(solver),
        },
    }
    with args.output.open("x", encoding="utf-8") as output:
        yaml.safe_dump(data, output, sort_keys=False)


if __name__ == "__main__":
    main()
