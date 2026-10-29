(differentiable-dispatch-control)=
# Differentiable Dispatch Controller

The `DifferentiableDispatchStorageController` dispatches a storage technology by solving an
optimization problem, and it also provides **exact derivatives** of its outputs with respect to
its inputs. Those derivatives come from implicit differentiation of the problem's optimality
(Karush-Kuhn-Tucker, or KKT) conditions, the same idea used by NLR's Differentiable Optimization
Layers (DOL) work. This lets gradient-based optimizers size storage and the surrounding plant
while the dispatch is re-optimized for every design, without finite differencing through the
dispatch problem.

```{note}
This controller needs the optional `dol` dependencies. Install them with
`pip install -e ".[dol]"` and then run `idaes get-extensions`, which downloads Ipopt and the Pyomo
PyNumero library. On Windows, scripts that run this controller need an
`if __name__ == "__main__":` guard, because Pyomo starts a `multiprocessing` worker the first time
it loads PyNumero.
```

## Drop-in interface

The controller uses the same interface as the
[`DemandOpenLoopStorageController`](#demand-open-loop-storage-controller), so it can replace it in
any `tech_config` by changing only the `control_strategy` model name and adding optional
`control_parameters`.

| Direction | Name | Description |
|---|---|---|
| Input | `<commodity>_in` | Commodity available to the storage (for example generation). |
| Input | `<commodity>_set_point` | Demand the storage helps meet. |
| Input | `storage_capacity`, `max_charge_rate`, `max_discharge_rate` | Storage design. `max_discharge_rate` exists only when `charge_equals_discharge` is false. |
| Output | `<commodity>_command_value` | Dispatch command for the storage performance model (positive discharges, negative charges). |
| Output | `dispatch_<commodity>_charge`, `dispatch_<commodity>_discharge` | Charge and discharge flows. |
| Output | `dispatch_<commodity>_stored` | Stored amount at the end of each time step. |
| Output | `dispatch_<commodity>_delivered`, `dispatch_<commodity>_unmet_demand`, `dispatch_<commodity>_curtailed` | Flow toward the set-point, the shortfall, and unused input. |
| Output | `dispatch_cost` | Realized dispatch cost in USD. |

## Optimization problem

In each dispatch window, with flows normalized by `flow_scale`, the controller solves

```{math}
\begin{aligned}
\min \quad & \sum_t w_t\,\Delta t\,\big(p_t\,u_t + c_c\,x^c_t + c_d\,x^d_t + c_u\,x^u_t\big)
  + q \sum_t \big((x^c_t)^2 + (x^d_t)^2\big) \\
\text{s.t.} \quad & \text{in}_t - x^u_t - x^c_t + x^d_t = \text{delivered}_t, \qquad
  \text{delivered}_t + u_t = \text{set point}_t \\
& s_t = s_{t-1} + \big(\eta_c\,x^c_t - x^d_t/\eta_d\big)\,\Delta t \\
& \text{soc}_{\min} E \le s_t \le \text{soc}_{\max} E, \quad x^c_t \le P_c, \quad x^d_t \le P_d,
  \quad x^c_t \le \text{in}_t, \quad x^u_t \le \text{in}_t
\end{aligned}
```

where {math}`x^c, x^d, x^u, u` are charge, discharge, curtailment, and unmet demand (all
nonnegative), {math}`s` is the stored amount, {math}`E` is the storage capacity, and {math}`P_c,
P_d` are the rate limits. The problem is solved with Ipopt.

## Smoothing and derivatives

Linear dispatch problems have kinks: the optimal dispatch changes slope whenever a limit becomes
active or inactive, so finite differences and gradient-based optimizers struggle. Two smoothing
terms make the dispatch a smooth function of the inputs:

- `smoothing_quadratic_weight` adds a small quadratic penalty on charge and discharge, which makes
  the optimum unique.
- `smoothing_barrier_mu` adds an explicit log-barrier term for every bounded flow, which turns the
  kinks into smooth transitions.

The barrier keeps each flow slightly away from its bound, by roughly
`smoothing_barrier_mu / cost * flow_scale`. Expect small simultaneous charge and discharge flows
and a small residual unmet demand of that size. Increasing `charge_cost` and `discharge_cost`
relative to `unmet_demand_penalty`, or decreasing `smoothing_barrier_mu`, reduces these effects.
Setting `smoothing_barrier_mu: 0` recovers the unsmoothed optimum.

The `derivatives` option selects how OpenMDAO gets partial derivatives:

- `adjoint` (default): matrix-free Jacobian-vector products from the factorized KKT system. In
  reverse mode, one back-solve per window gives the gradient of any scalar output with respect to
  all inputs, including full time series.
- `fd`: finite-difference partials on this component.
- `none`: no partials, for analysis-only runs.

H2I's optimization driver defaults to whole-model finite differences. To use these derivatives
in a YAML-driven optimization, select `total_derivatives: components`. Every component on the
objective and constraint paths must supply valid partials; missing partials are treated as zero.
Example 39 uses a mixture of analytic, KKT, and component-level finite-difference derivatives.

## Dispatch horizon

`dispatch_horizon_mode` selects the dispatch fidelity:

- `full_horizon` (default): one perfect-foresight problem over the whole simulation.
- `rolling`: a receding horizon that solves windows of `n_horizon_window` time steps and applies
  the first `n_control_window` steps of each. The stored amount is carried between windows, and
  reverse-mode derivatives sweep backward through the chain of windows.

Both modes give identical results when the window covers the whole simulation. Rolling mode
changes the control problem when the look-ahead is shorter and may be slower because it solves
many windows. H2I's separate time-marching solver does not support this controller or its
full-plant derivative chain.

Example 39 has no system-level controller or plant feedback loop: it uses the battery's
technology-level controller, and the plant runs once per optimizer evaluation. The controller
still re-solves its dispatch problem when the design or generation inputs change.

## Ipopt backend

The default `solver_backend: shell` writes a new NL file and launches Ipopt for each dispatch
solve. Set `solver_backend: cached` to reuse the PyNumero NLP and call the Ipopt shared library
in-process. This preserves the full-horizon formulation and KKT derivatives. Example 39 selects
the cached backend.

The cached backend requires a loadable Ipopt shared library with the standard 32-bit-index C
interface, in addition to PyNumero. The IDAES solver extensions provide one on supported
platforms. It currently supports minimization, requires
`fixed_variable_treatment: make_parameter`, and rejects Pyomo export suffixes. A changed mutable
Pyomo parameter triggers an NLP rebuild. Changing the objective, constraints, decision-variable
bounds, or other fixed variables is unsupported; build a new adapter for those changes.
Use `shell` if these restrictions or the shared-library requirement do not suit the model.

For the cached backend, `solver_options: {warm_start_init_point: "yes"}` reuses the previous
solution's constraint and bound multipliers along with the primal starting point. The first
solve starts without multipliers, and a failed solve or NLP rebuild discards the saved
multipliers. Warm starts can be slower or fail for some changes in inputs; they do not
automatically relax the convergence requirements. They are not enabled in example 39 because,
in a paired test at three nearby designs, primal-dual starts took longer than primal-only starts
while producing matching outputs and totals.

On one Windows run of example 39, the full-year SLSQP sizing study took 162 seconds with the
cached backend versus 282 seconds with the shell backend, with the same reported design and
evaluation counts. After the FLORIS cache and ProFAST partial improvements, a subsequent run
took about 140 seconds with the same design and evaluation counts. These individual timings
depend on the machine and solver behavior.

A short fixed generation profile can also be used with the controller directly. For example,
a synthetic 168-hour dispatch was much faster than the corresponding 8760-hour dispatch in a
local timing test. The separate `run_short_dispatch.py` example sizes storage using a fixed
synthetic 168-hour electricity profile and an illustrative weekly cost proxy. It also accepts
`--generation-csv` with 168 headerless, single-column hourly MW values. It uses the H2I storage
controller and cost model, but not the complete plant, electrolyzer, or ProFAST.

```console
python run_short_dispatch.py --generation-csv generation.csv
```

This is not a one-week version of example 39's full TEA: in particular, the PEM lifetime
calculations assume that their simulation period is close to a year. A short profile must not
be interpreted as a representative annual finance case without addressing that assumption
and the storage's inter-period state of charge.

## Configuration

These `control_parameters` are specific to this controller. The storage parameters
(`max_capacity`, `max_charge_rate`, state-of-charge limits, efficiencies, and `demand_profile`)
are the same as for the open-loop storage controllers.

| Parameter | Default | Description |
|---|---|---|
| `dispatch_horizon_mode` | `full_horizon` | `full_horizon` or `rolling`. |
| `n_horizon_window` | 48 | Look-ahead length in time steps (`rolling` only). |
| `n_control_window` | 24 | Time steps applied per window (`rolling` only). |
| `unmet_demand_penalty` | 1.0 | Cost of unmet set-point in USD per amount unit; scalar or one value per time step. |
| `charge_cost`, `discharge_cost` | 0.001 | Cycling costs in USD per amount unit. |
| `curtailment_cost` | 0.0 | Cost of unused input in USD per amount unit. |
| `time_weighting_factor` | 1.0 | Per-step discount factor within each window. |
| `smoothing_quadratic_weight` | 1e-4 | Quadratic regularization weight (normalized). |
| `smoothing_barrier_mu` | 1e-6 | Log-barrier smoothing weight (normalized). |
| `derivatives` | `adjoint` | `adjoint`, `fd`, or `none`. |
| `flow_scale` | computed | Flow normalization; defaults to the largest of the rate limits and the mean set-point. |
| `solver_options` | `{}` | Extra Ipopt options. |
| `solver_backend` | `shell` | `shell` or `cached`; see the backend requirements above. |

The battery in `examples/39_differentiable_dispatch` uses this controller:

```{literalinclude} ../../../examples/39_differentiable_dispatch/tech_config.yaml
:language: yaml
:lines: 79-115
```

## Example: gradient-based sizing through the dispatch

`examples/39_differentiable_dispatch/run_differentiable_dispatch.py` first runs the example 13
plant (wind and solar, battery, electrolyzer) with this controller. It then optimizes the
battery energy, battery power, and solar DC rating in the actual H2I plant using
`driver_config_optimization.yaml`. The objective is ProFAST LCOH, with a 0.5 first-year PEM
capacity-factor constraint. This differs from the old simplified demo's 0.6 constraint: the
full H2I plant reaches about 0.54 even at the 120 MW solar and 500 MWh battery bounds. The
script starts from a feasible design with 90 MW of solar, runs SLSQP by default, and accepts
`--compare-cobyla` to also run gradient-free COBYLA.

This is a **mixed-derivative**, not an entirely exact, chain. The cable, combiners, demand,
storage performance, cost models, and cost-year adjustment provide analytic or piecewise analytic
partials; dispatch uses KKT sensitivities; and PySAM, FLORIS, PEM, and ProFAST use local finite
differences. PySAM's supported input is DC capacity and FLORIS's is hub height; other wind and
solar design inputs are not covered. The PEM directional finite difference and storage recurrence
currently support forward mode only. These partials have only been implemented for the example's
paths and modes, not every H2I component.

The PEM model rounds cluster capacity to whole megawatts and uses discrete operating decisions,
so its cluster count and rating are held fixed in this optimization. Local finite differences
can change substantially across PEM on/off thresholds or storage limits; agreement at one point
does not imply a globally smooth model. For example, at 90 MW solar and 20 MWh battery energy,
the component and independent finite-difference battery-power sensitivities disagreed at 2.0 MW
but agreed at the chosen 2.1 MW starting point. Near the rounded sizing result, a 0.01 MW
decrease in battery power changed LCOH by about 0.14 USD/kg and coincided with the PEM
replacement schedule changing from every five years to every four years. The schedule's
individual contribution to the jump has not been isolated. Check totals at multiple steps
and designs, and compare against derivative-free solutions before relying on an optimization
result.

## Developer notes

The KKT machinery lives in `h2integrate.control.differentiable.KKTSensitivity`, a general adapter
for any parametric Pyomo model with equality constraints and bounded variables. It solves the
model with Ipopt and, on first use, factorizes the KKT system to provide `jvp`, `vjp`, a dense
`jacobian`, and an `envelope_gradient` of the optimal objective. Other controllers can reuse it
by expressing their parameters as fixed Pyomo variables and writing inequalities with slack
variables.
