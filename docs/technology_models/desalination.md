# Reverse Osmosis Desalination

{py:class}`~h2integrate.converters.water.desal.desalination.ReverseOsmosisPerformanceModel`
represents a reverse osmosis plant that converts feedwater and
electricity into freshwater. At each timestep, freshwater production (`water_out`, m^3/h) is the
minimum of three limits:

- the plant capacity, `freshwater_capacity_kg_per_hr` (kg/h) divided by `freshwater_density`
   (kg/m^3)
- the available electricity, `electricity_in` (kW) divided by `specific_energy_consumption` (kWh/m^3)
- the available feedwater, `feedwater_in` (m^3/h) multiplied by `recovery_ratio`

`electricity_in` and `feedwater_in` are inputs with one value per timestep, so they may vary over
the simulation. The model reports the resources actually used as `electricity_consumed` and
`feedwater_consumed`, along with the system `mass` and `footprint`.

Choose `salinity: seawater` or `salinity: brackish` in the configuration. This sets the recovery
ratio and specific energy consumption:

| Salinity   | Recovery ratio | Specific energy consumption |
| ---------- | -------------- | --------------------------- |
| `seawater` | 0.5            | 4.0 kWh/m^3                 |
| `brackish` | 0.75           | 1.5 kWh/m^3                 |

The capacity and freshwater density are also OpenMDAO inputs, so they can be varied by a design of
experiments or optimization.

{py:class}`~h2integrate.converters.water.desal.desalination.ReverseOsmosisCostModel`
estimates capital and operating costs from
`freshwater_capacity_kg_per_hr` using the model's reference costs. Its cost configuration is fixed
to 2013 USD.
Both models support time steps from 1 second to 1 hour. Rates (`water_out`, `electricity_in`,
`feedwater_in`, and the consumed quantities) are per hour regardless of time step, while
`total_water_produced` and `annual_water_produced` account for the time step length.

The performance settings are described by
{py:class}`~h2integrate.converters.water.desal.desalination.ReverseOsmosisPerformanceModelConfig`;
the cost settings are described by
{py:class}`~h2integrate.converters.water.desal.desalination.ReverseOsmosisCostModelConfig`.
The cost model exposes `rated_freshwater_production_capacity` in kg/s as an OpenMDAO input,
initialized from the configured capacity, and reports `CapEx` in USD and `OpEx` in USD/year.
The inherited `capacity_factor`, `rated_water_production`, `total_water_produced`, and
`annual_water_produced` outputs summarize performance over the simulation.

## Offline Design Point

To use a fixed operating point from a process model such as
[WaterTAP's RO flowsheet](https://watertap.readthedocs.io/en/stable/technical_reference/flowsheets/RO_with_energy_recovery.html),
omit `salinity` and supply `design_point` in `model_inputs.performance_parameters` for
`ReverseOsmosisPerformanceModel`. Set `freshwater_capacity_kg_per_hr` and `freshwater_density` to
the solved total permeate solution mass flow and density. In `design_point`, supply the total
feed solution mass flow and density, net electrical demand, and feed and permeate salt mass
fractions from the same solve. Record the version and run settings in `source`, and set
`max_permeate_salt_mass_fraction` from the downstream water requirement. The model rejects a
point that exceeds this limit or cannot close the solution mass and salt balances.

The train runs at this point only when both `electricity_in` and `feedwater_in` meet the point's
requirements. Otherwise it produces no water and consumes neither resource. In this mode,
`concentrate_mass_flow` and `concentrate_salt_mass_flow` report total solution and dissolved salt
leaving in the concentrate, both in kg/h. Use an electrical demand that already accounts for any
energy recovery included in the solve; do not subtract that recovery again in H2Integrate.

H2Integrate does not call the upstream solver or include a WaterTAP dataset. This mode assumes
the same feed composition and temperature as the solved point. It does not model part load,
changing salinity, or polishing. Choose another solved point rather than varying `freshwater_capacity`
or `freshwater_density` as design variables. The existing cost model is unchanged and does not
estimate WaterTAP-specific equipment costs.

The offline settings are described by
{py:class}`~h2integrate.converters.water.desal.desalination.ReverseOsmosisDesignPointConfig`.

```{eval-rst}
.. autoclass:: h2integrate.converters.water.desal.desalination.ReverseOsmosisPerformanceModelConfig
   :no-members:
   :no-index:

.. autoclass:: h2integrate.converters.water.desal.desalination.ReverseOsmosisDesignPointConfig
   :no-members:
   :no-index:

.. autoclass:: h2integrate.converters.water.desal.desalination.ReverseOsmosisCostModelConfig
   :no-members:
   :no-index:
```
