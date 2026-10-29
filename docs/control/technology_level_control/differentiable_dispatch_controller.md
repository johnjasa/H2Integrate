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

H2I's optimization driver currently approximates total derivatives of the whole model with
finite differences, so the exact partials are used when the controller is part of a problem
whose other components also provide partials, as in example 39's sizing study. Extending exact
derivatives to the rest of the H2I model chain is planned follow-on work.

## Dispatch horizon

`dispatch_horizon_mode` selects the dispatch fidelity:

- `full_horizon` (default): one perfect-foresight problem over the whole simulation.
- `rolling`: a receding horizon that solves windows of `n_horizon_window` time steps and applies
  the first `n_control_window` steps of each. The stored amount is carried between windows, and
  reverse-mode derivatives sweep backward through the chain of windows.

Both modes give identical results when the window covers the whole simulation. A full-year
(8760-hour) `full_horizon` solve takes roughly 10 seconds, and an exact reverse-mode gradient
takes well under a second more. Rolling mode solves one small problem per window, so it is slower
in total.

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

The battery in `examples/39_differentiable_dispatch` uses this controller:

```{literalinclude} ../../../examples/39_differentiable_dispatch/tech_config.yaml
:language: yaml
:lines: 79-114
```

## Example: gradient-based sizing through the dispatch

`examples/39_differentiable_dispatch/run_differentiable_dispatch.py` first runs the example 13
plant (wind and solar, battery, electrolyzer) with this controller. It then sizes the battery
energy, battery power, and electrolyzer rating for the lowest levelized cost of hydrogen, subject
to a minimum electrolyzer capacity factor of 0.6. A linear electrolyzer and a simple annualized
LCOH provide analytic partials, so the problem has exact total derivatives end to end.

The script checks the exact total derivatives against finite differences, which agree to about
six significant figures. It then compares gradient-based SLSQP with gradient-free COBYLA:

| Horizon | Optimizer | LCOH (USD/kg) | Battery (MWh / MW) | Electrolyzer (MW) | Model evaluations | Gradient evaluations | Wall time |
|---|---|---|---|---|---|---|---|
| 28 days (script default) | SLSQP | 6.517 | 27.6 / 4.8 | 51.3 | 50 | 20 | about 35 s |
| 28 days (script default) | COBYLA | 6.517 | 27.9 / 4.9 | 51.3 | 123 | 0 | about 95 s |
| Full year (`SIZING_DAYS = 365`) | SLSQP | 8.425 | 57.1 / 9.4 | 39.3 | 12 | 9 | about 2.5 min |
| Full year (`SIZING_DAYS = 365`) | COBYLA | 8.425 | 57.1 / 9.4 | 39.3 | 66 | 0 | about 13 min |

Both optimizers reach the same design, with the capacity factor constraint active, but the
gradient-based optimizer needs far fewer dispatch solves. Each gradient costs one reverse
(adjoint) solve, regardless of the number of design variables.

## Developer notes

The KKT machinery lives in `h2integrate.control.differentiable.KKTSensitivity`, a general adapter
for any parametric Pyomo model with equality constraints and bounded variables. It solves the
model with Ipopt and, on first use, factorizes the KKT system to provide `jvp`, `vjp`, a dense
`jacobian`, and an `envelope_gradient` of the optimal objective. Other controllers can reuse it
by expressing their parameters as fixed Pyomo variables and writing inequalities with slack
variables.
