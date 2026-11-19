# Offline Process Operating Point

{py:class}`~h2integrate.converters.offline_process.OfflineProcessPerformanceModel`
replays one prequalified operating point from a process model. Solve and validate the point
outside H2Integrate, then supply its product rate, required stream and utility rates, and optional
coproduct rates in the technology configuration. H2Integrate does not import or call IDAES,
WaterTAP, PrOMMiS, or Aspen. This is a common interface for fixed operating points, not a
general-purpose flowsheet or dispatch model.

## Runtime Contract

- Pick one primary commodity. `product` supplies its rated rate and units. The corresponding
  `<commodity>_out`, `rated_<commodity>_production`, `total_<commodity>_produced`,
  `annual_<commodity>_produced`, and `capacity_factor` use standard H2Integrate conventions.
- Each `requirements` key adds `<name>_in` and `<name>_consumed` in the specified rate units.
  Connect the available resource, not its price or the process's requested rate. An unconnected
  input defaults to zero. If any input is below its required rate or is nonfinite, all output
  and consumption rates are zero for that step. Otherwise they equal the specified rates.
- Each `coproducts` key adds `<name>_out` in the specified rate units. The model does not give
  coproducts an automatic revenue credit or assume downstream compatibility.
- The performance component supports time steps from one second to one hour. Total output
  integrates the rate over the simulated hours; annual output extrapolates that period's
  utilization and repeats it over the plant life. It does not model degradation.

Rates must be nonnegative and finite. The primary product and required input rates must be
positive. All streams are on the same solved equipment and operating basis. Screen product
quality and check material and energy balances before importing a point. Use a specialized
model when composition, quality, waste treatment, ramping, startup, variable load, or control
commands must affect operation. The fixed-point switch is not a continuous design variable.

## Cost And Finance

Pair the performance model with
{py:class}`~h2integrate.converters.generic_converter_cost.GenericConverterCostModel` for
hourly simulations. Set its `commodity` to the same primary commodity and give it compatible
rate and amount units. For a fixed equipment design, use `additional_capex_USD` and
`additional_opex_USD_per_year` for the installed cost and annual fixed O&M, with `unit_capex`
and `unit_opex` set to zero. Use `unit_varopex` for non-utility variable O&M per unit of primary
output. Give costs an explicit dollar year. Utility quantities are reported by performance;
price them once in the H2Integrate system rather than importing a process-model levelized cost
or counting purchased utilities again in operating costs.

This test fixture uses synthetic numbers to demonstrate the pairing, not to represent an
equipment cost or an upstream solve:

```{literalinclude} ../../h2integrate/converters/test/inputs/offline_process_tech.yaml
:language: yaml
```

## Source Data Handoff

Keep the solver input and export reproducible. For a single operating point, the YAML above is
the runnable artifact. Use `source` to reference a versioned upstream run record that includes
the flowsheet and package versions, solver and settings, equipment design, operating conditions,
stream bases (including density or reference state for volumetric rates), balance residuals,
product specifications, and the cost scope and exclusions. Assign an owner to each real source
case. Unknown or invalid points are not interchangeable with zero output: exclude them from
export, and record the reason in the source run record.

For a future operating map, use one metadata YAML per fixed equipment design and one CSV with
one row per attempted point. Define the axis names, units and valid ranges in metadata, and the
basis and sign convention of every output. Include a point ID, axis values, solve and quality
status, and a fit or held-out designation. Include utility, product and waste rates in accepted
rows; leave failed results out of the interpolation set. Do not mix designs or extrapolate
outside validated points. The component above **does not read or interpolate this CSV**. Add
a bounded evaluator only after the pilots confirm how to handle invalid regions and which
constraints can be shared.

## Solved Pilot Replays

H2Integrate replays two WaterTAP 0D RO simulations and one PrOMMiS furnace simulation from
[`offline_pilot_points.yaml`](../../h2integrate/converters/test/inputs/offline_pilot_points.yaml).
The shared component reproduces the reported rates when all inputs are available and shuts off
when any input is short. Both RO points also agree with the dedicated
[RO design-point model](desalination.md), which enforces the salt limit itself.

| Source case | Primary product | Utility at the modeled boundary |
| --- | ---: | --- |
| WaterTAP, pressure exchanger | 1.7833 m^3/h permeate | 4.9550 kW pump load (2.7785 kWh/m^3) |
| WaterTAP, no energy recovery | 1.7833 m^3/h permeate | 9.2395 kW pump load (5.1811 kWh/m^3) |
| PrOMMiS, hydrogen decrepitation | 20.6521 kg/h decrepitated magnet | 3.7617 kW thermal duty, 0.04101 kg/h gross H2 inlet |

The RO runs both use 3.6 m^3/h of feed at 298.15 K and 3.5% NaCl by mass and recover 50% of
the feed water by mass. Both produce about 240 ppm NaCl by mass, below the pilot limit of
500 ppm. The pressure exchanger reduces modeled pump energy per unit of permeate by 46.4%.
These are matched designs at one service target, not a fixed-area operating sweep. The
comparison excludes pretreatment and polishing; 500 ppm is not an electrolyzer quality claim.

The furnace is a reference throughput calculation for a 3-hour decrepitation phase, not a
simulation of batch timing or heat storage. The published gas balance sets H2 outlet equal to
inlet. H2Integrate replays gross inlet and outlet; it does not infer a purchase rate, recycle, or
zero physical uptake. The recorded inlet-minus-outlet total mass residual is about
`-7.85e-5 kg/h`. Heat is a thermal duty, not an electrical draw; heater efficiency, upstream
hydrogen preheating, and other balance-of-plant loads are outside the replay.

**Reproduce offline.** Use a separate Python 3.11 environment with WaterTAP 1.8.0, PrOMMiS 1.1.0,
IDAES 2.13.0, Pyomo 6.10.1, and watertap-solvers 24.12.9. The reported run used IPOPT 3.13.2
from the IDAES 3.4.2 Windows extension bundle. With IPOPT on the isolated environment's `PATH`,
run from the H2Integrate root:

```text
python h2integrate/converters/test/inputs/solve_offline_pilots.py offline_pilot_points.yaml
```

The generator checks package versions and solver termination before writing a new file. RO
exports are screened for water and salt closure, recovery, and permeate salinity. The furnace
export checks the stated gas balance and records the small mass residual rather than adjusting
the solved streams. Neither case exports costs: the RO initializer briefly warns about a pump
cost variable, and no cost boundary for either case has been qualified. Run the replay tests in
the normal H2Integrate environment; they need only the exported YAML, not the upstream solvers.

```{eval-rst}
.. autoclass:: h2integrate.converters.offline_process.OfflineProcessPerformanceModelConfig
   :no-members:
   :no-index:

.. autoclass:: h2integrate.converters.offline_process.OfflineProcessStreamConfig
   :no-members:
   :no-index:
```
