
# Geothermal power model using the Geothermal (GETEM) module in PySAM

This model uses the [Geothermal module](https://nrel-pysam.readthedocs.io/en/main/modules/Geothermal.html) available in PySAM to simulate the performance of a geothermal power plant. The Geothermal module is a wrapper around the Geothermal Electricity Technology Evaluation Model (GETEM). More information on GETEM can be found [here](https://www.energy.gov/hgeo/geothermal/geothermal-electricity-technology-evaluation-model).

Geothermal plants are modeled as fixed (baseload) generators of electricity. Their output is driven by the subsurface resource and is not curtailed or dispatched within H2Integrate's system-level control framework.

The surface ambient conditions used to size the power cycle (dry-bulb temperature, humidity, and pressure) are taken from a connected weather resource model rather than a weather file, following the same resource-handling pattern used by the solar and wind PySAM models. Connect a weather/solar resource to the geothermal technology through `resource_to_tech_connections` in the plant config so that its `solar_resource_data` output is provided to the model.

## Performance model

To use the performance model, specify `"PYSAMGeothermalPlantPerformanceModel"` as the performance model. An example of how this may look in the `tech_config` file is shown below, and details on the performance parameter inputs can be found [here](#performance-parameters).

```yaml
technologies:
    geothermal:
        performance_model: "PYSAMGeothermalPlantPerformanceModel"
        cost_model: "GeothermalPlantCostModel"
        model_inputs:
            performance_parameters:
                nameplate_kW: 30000.0
                resource_temp_C: 200.0
                resource_depth_m: 2000.0
                resource_type: 0 # 0 = hydrothermal, 1 = EGS
                conversion_type: 0 # 0 = binary, 1 = flash
                analysis_type: 0 # 0 = specify nameplate, 1 = specify number of wells
                create_model_from: "default" # options are "default" and "new"
                config_name: "GeothermalPowerSingleOwner"
                pysam_options: # optional, additional PySAM inputs
                    GeoHourly:
                    AdjustmentFactors:
            cost_parameters:
                capex_per_kW: 5500.0
                opex_per_kW_per_year: 130.0
                cost_year: 2022
```

(performance-parameters)=
### Performance Parameters
- `nameplate_kW` (required): desired plant nameplate (net) output in kW.
- `resource_temp_C` (required): geothermal resource temperature in degrees Celsius. Must be in the range [0, 373].
- `resource_depth_m` (required): geothermal resource depth in meters.
- `resource_type` (optional): type of geothermal resource. `0` for hydrothermal (default) or `1` for an enhanced geothermal system (EGS).
- `conversion_type` (optional): power conversion cycle type. `0` for a binary cycle (default) or `1` for a flash cycle.
- `analysis_type` (optional): sizing basis. `0` to specify the plant nameplate output (default) or `1` to specify the number of production wells.
- `num_wells` (optional): number of production wells. Only used when `analysis_type` is `1`.
- `create_model_from` (optional): either `"default"` or `"new"`, defaults to `"default"`. If `"default"`, the model is initialized using `Geothermal.default(config_name)` and then updated with the parameters above. If `"new"`, the model is initialized using `Geothermal.new()` and must be populated with parameters specified in `pysam_options`.
- `config_name` (optional): only used if `create_model_from` is `"default"`. Defaults to `"GeothermalPowerSingleOwner"`.
- `pysam_options` (optional): dictionary of additional PySAM Geothermal inputs with top-level keys corresponding to the Geothermal variable groups (`GeoHourly`, `AdjustmentFactors`). Parameters managed directly by this model (for example `resource_temp`, `nameplate`, `design_temp`) must not be duplicated here.

### Resource connection

The model reads the ambient dry-bulb temperature, relative humidity, and pressure from the connected resource model's `solar_resource_data` output. The annual-mean dry-bulb temperature sets the power-block design temperature, the wet-bulb temperature is estimated from the mean relative humidity, and the mean pressure sets the ambient pressure. GETEM is run in its design-calculation mode using these conditions, and the resulting net capacity is dispatched as a constant baseload profile.

### Outputs
- `electricity_out`: hourly electricity generation profile in kW.
- `system_capacity_AC`: net rated (AC) capacity of the plant in kW.
- `total_electricity_produced`, `annual_electricity_produced`, `capacity_factor`, and the other standard performance outputs.

## Cost model

To use the cost model, specify `"GeothermalPlantCostModel"` as the cost model. Capital and fixed operating costs are scaled by the plant net (AC) capacity computed by the performance model. Representative cost values for geothermal power technologies can be found in the [NREL Annual Technology Baseline (ATB)](https://atb.nrel.gov/electricity/2024/geothermal).

### Cost Parameters
- `capex_per_kW` (required): capital cost of the geothermal plant in $/kW-AC.
- `opex_per_kW_per_year` (required): annual fixed operating cost of the geothermal plant in $/kW-AC/year.
- `cost_year` (required): dollar year corresponding to the input costs.

The CapEx and OpEx are computed as:

$$
\text{CapEx} = \text{capex\_per\_kW} \times \text{system\_capacity\_AC}
$$

$$
\text{OpEx} = \text{opex\_per\_kW\_per\_year} \times \text{system\_capacity\_AC}
$$
