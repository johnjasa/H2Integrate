(slc-base)=
# System Level Control Base Class

The system-level control base class provides a common framework that all controllers (advanced control strategies) can use to configure required inputs and outputs for both the controllers and the components they control or track. This generalization is necessary to implement system-level control in H2I. If the technologies and controllers in a given system were fully specified, this base class would not be needed.

```{important}
SLC demand is supplied by a demand component. When SLC is enabled, only one demand component is currently supported.
```

The base class also abstracts logic that may be shared across different controller types. It includes methods that could be useful, but not all methods will be relevant to every controller you implement.

There are several methods that are already used in the simple controllers that inherit these system.

Setup I/O for SLC controllers.
- `initialize()`
- `setup()`
- `_setup_commodity()`
- `_setup_tech_category()`
- `_setup_feedstock_category()`
- `find_converter_techs()`
    - Note: this method is currently is not used but will be used for heterogeneous commodity systems.

Functions for controlling components based on assigned control classifier.
- `_subtract_curtailable()`
- `_dispatch_storage()`
- `get_upstream_techs_for_commodity()`

Helper functions for cost-aware controllers.
- `_add_price_input()`
- `_setup_marginal_costs()`
- `_compute_marginal_costs()`
- `_buy_price_marginal_cost()`
- `_varopex_marginal_cost()`
- `_find_feedstock_techs()`
- `_feedstock_marginal_cost()`

### Buy and sell prices

`_add_price_input(tech_name, side)` adds a per-timestep `{tech_name}_{side}_price` input, where `side` is `"buy"` or `"sell"`. `H2IntegrateModel` connects it to the technology's `{commodity}_{side}_price`, where `commodity` is the controlled commodity. A buy price may also come from a feedstock `price`. The technology can expose the price as an input (for example a configured grid price) or as an output (for example a price computed by a custom model), and it can be scalar, per-timestep, or per-year. Scalar and per-year prices are held at their first value across the simulation. If the technology has no such variable, a warning is raised and the controller uses the price from the technology config, or zero.

`LPArbitrageControl` reads its sell price this way from the technology named by
`export_component`. It uses that sell price to dispatch grid imports, even when
their financial buy price differs. The grid cost model still uses the actual buy
price to calculate costs. Other dispatchable technologies use their configured
`cost_per_tech` marginal costs.

The controller detects grid imports using `GridPerformanceModel`. For custom grid
performance models, set `grid_import_techs` to the complete list of controlled
grid import technology names. Grid imports may supply demand or charge storage,
but cannot be sold back or curtailed in the same timestep. Local production may
still be sold while the grid charges storage. The battery charge/discharge mode
is a binary decision, so the configured solver must support mixed-integer linear
programs. The controller assumes all supply, storage and demand share one bus;
the plant's physical connections must support the scheduled flows.

`examples/35_system_level_control/lp_arbitrage_24h/` provides a short,
reproducible 24-hour controller-window demonstration, including an explicitly
constrained tied optimum where solar export coincides with grid charging. It
solves the controller's window model, rather than running a full plant
simulation. The neighboring `lp_arbitrage` example runs the full-year plant.
That example can export recovered spill from the existing plant up to the grid
interconnection limit; its financial results include that spill revenue but
exclude the existing plant's costs.

## Base Class and Methods

```{eval-rst}
.. autoclass:: h2integrate.control.control_strategies.system_level.system_level_control_baseclass.SystemLevelControlBase
   :members:
   :undoc-members:
   :show-inheritance:
```
