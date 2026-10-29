import numpy as np
import openmdao.api as om
import pyomo.environ as pyo
from attrs import field, define, validators
from openmdao.utils import units as om_units

from h2integrate.core.utilities import merge_shared_inputs
from h2integrate.core.supported_models import register
from h2integrate.control.control_strategies.openloop_control_baseclass import (
    OpenLoopControlBase,
    OpenLoopControlBaseConfig,
)


# Pyomo decision variables reported by each dispatch window, in output order.
_WINDOW_OUTPUT_VARS = ("charge", "discharge", "stored", "unmet", "curtail", "delivered")


@define(kw_only=True)
class DifferentiableDispatchStorageControllerConfig(OpenLoopControlBaseConfig):
    """Configuration for :class:`DifferentiableDispatchStorageController`.

    Storage sizing and efficiency parameters come from :class:`OpenLoopControlBaseConfig`
    (``max_capacity``, ``max_charge_rate``, ``min_soc_fraction``, ``max_soc_fraction``,
    ``init_soc_fraction``, ``charge_efficiency``, ``discharge_efficiency`` or
    ``round_trip_efficiency``, ``charge_equals_discharge``, ``max_discharge_rate``, and
    ``demand_profile``).

    Attributes:
        dispatch_horizon_mode (str): ``"full_horizon"`` solves one perfect-foresight problem over
            the whole simulation. ``"rolling"`` solves a sequence of look-ahead windows and applies
            the first ``n_control_window`` steps of each (receding horizon).
        n_horizon_window (int): Look-ahead length in time steps for ``"rolling"`` mode.
        n_control_window (int): Number of time steps applied from each window in ``"rolling"``
            mode. Must not exceed ``n_horizon_window``.
        unmet_demand_penalty (float | list[float]): Cost of unmet demand in USD per
            ``commodity_amount_units``, as a scalar or one value per time step.
        charge_cost (float): Cost of charging in USD per ``commodity_amount_units``. A small
            positive value discourages unnecessary cycling.
        discharge_cost (float): Cost of discharging in USD per ``commodity_amount_units``.
        curtailment_cost (float): Cost of curtailing available commodity in USD per
            ``commodity_amount_units``.
        time_weighting_factor (float): Per-step discount factor applied within each window,
            in (0, 1]. 1.0 weights all steps equally.
        smoothing_quadratic_weight (float): Weight of the quadratic regularization on charge and
            discharge, relative to the largest cost coefficient with flows normalized by
            ``flow_scale``. It makes the optimal dispatch unique.
        smoothing_barrier_mu (float): Weight of an explicit log-barrier term on every bounded
            flow, in the same normalized units. Positive values make the dispatch a smooth
            function of the inputs so derivatives agree with finite differences. The barrier
            keeps each flow at least about ``smoothing_barrier_mu / cost`` (times
            ``flow_scale``) away from zero, so small simultaneous charge and discharge flows and
            a small residual unmet demand of that size are expected. Set it to 0 to recover the
            unsmoothed optimum.
        derivatives (str): ``"adjoint"`` provides exact KKT-based Jacobian-vector products
            (matrix-free), ``"fd"`` declares finite-difference partials on this component, and
            ``"none"`` declares no partials (analysis only).
        flow_scale (float | None): Rate used to normalize flows in the optimization problem, in
            ``commodity_rate_units``. Defaults to the largest of ``max_charge_rate``,
            ``max_discharge_rate``, and the mean absolute ``demand_profile``.
        solver_options (dict): Extra Ipopt options.
        solver_backend (str): ``"shell"`` writes an NL file for each Ipopt solve. ``"cached"``
            reuses the PyNumero NLP and calls a shared Ipopt library in-process.
    """

    require_storage_parameters = True

    dispatch_horizon_mode: str = field(
        default="full_horizon", validator=validators.in_(["full_horizon", "rolling"])
    )
    n_horizon_window: int = field(default=48, validator=validators.gt(0))
    n_control_window: int = field(default=24, validator=validators.gt(0))
    unmet_demand_penalty: float | list = field(default=1.0)
    charge_cost: float = field(default=1.0e-3, validator=validators.ge(0))
    discharge_cost: float = field(default=1.0e-3, validator=validators.ge(0))
    curtailment_cost: float = field(default=0.0, validator=validators.ge(0))
    time_weighting_factor: float = field(
        default=1.0, validator=(validators.gt(0), validators.le(1))
    )
    smoothing_quadratic_weight: float = field(default=1.0e-4, validator=validators.ge(0))
    smoothing_barrier_mu: float = field(default=1.0e-6, validator=validators.ge(0))
    derivatives: str = field(default="adjoint", validator=validators.in_(["adjoint", "fd", "none"]))
    flow_scale: float | None = field(default=None)
    solver_options: dict = field(factory=dict)
    solver_backend: str = field(default="shell", validator=validators.in_(["shell", "cached"]))

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.common_post_init_operations()
        if (
            self.dispatch_horizon_mode == "rolling"
            and self.n_control_window > self.n_horizon_window
        ):
            msg = (
                f"n_control_window ({self.n_control_window}) must not exceed n_horizon_window "
                f"({self.n_horizon_window})."
            )
            raise ValueError(msg)


class _DispatchWindow:
    """Pyomo model and KKT sensitivity object for one window length."""

    def __init__(self, sensitivity, model, n_steps):
        self.sensitivity = sensitivity
        self.model = model
        self.n_steps = n_steps
        self.index = {
            name: sensitivity.variable_indices(getattr(model, name))
            for name in (_WINDOW_OUTPUT_VARS)
        }


@register
class DifferentiableDispatchStorageController(OpenLoopControlBase):
    """Optimal storage dispatch with exact derivatives through the optimization (KKT) problem.

    A drop-in alternative to :class:`DemandOpenLoopStorageController`: it reads the same inputs
    (``<commodity>_in``, ``<commodity>_set_point``, ``storage_capacity``, ``max_charge_rate``) and
    writes the same ``<commodity>_command_value`` output for the storage performance model. Instead
    of a heuristic, it solves a smooth quadratic program with Ipopt that minimizes unmet demand,
    cycling, and curtailment costs over a perfect-foresight horizon (or a rolling look-ahead).

    Derivatives of all outputs with respect to the inputs come from implicit differentiation of
    the optimality (KKT) conditions, through the H2I-owned adapter in
    :mod:`h2integrate.control.differentiable`. In ``"rolling"`` mode the windows are chained
    through the stored energy, and reverse-mode products sweep backward through the windows.

    The optimization problem in each window, with flows normalized by ``flow_scale``, is::

        min  sum_t w_t * dt * (p_t * unmet_t + c_c * charge_t + c_d * discharge_t
                               + c_u * curtail_t) + q * sum_t (charge_t^2 + discharge_t^2)
        s.t. in_t - curtail_t - charge_t + discharge_t = delivered_t
             delivered_t + unmet_t = demand_t
             stored_t = stored_{t-1} + (eta_c * charge_t - discharge_t / eta_d) * dt
             soc_min * capacity <= stored_t <= soc_max * capacity
             charge_t <= max_charge_rate, charge_t <= in_t
             discharge_t <= max_discharge_rate, curtail_t <= in_t
             all flows >= 0

    Inequalities are written with nonnegative slack variables so that all parameter-dependent
    limits are constraints, as the KKT adapter requires.

    Requires the optional ``dol`` dependencies (``pip install -e ".[dol]"`` and
    ``idaes get-extensions``). On Windows, scripts that run this controller need an
    ``if __name__ == "__main__":`` guard because Pyomo's PyNumero loader starts a
    ``multiprocessing`` worker the first time it is used.
    """

    _time_step_bounds = (1, 36000)
    _is_steppable = False

    def setup(self):
        self.config = DifferentiableDispatchStorageControllerConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "control"),
            strict=False,
            additional_cls_name=self.__class__.__name__,
        )
        super().setup()

        if self.n_timesteps_window != self.n_timesteps:
            msg = (
                f"{self.__class__.__name__} is not steppable; it requires n_timesteps_window "
                "to equal n_timesteps."
            )
            raise ValueError(msg)

        commodity = self.config.commodity
        rate_units = self.config.commodity_rate_units
        amount_units = self.config.commodity_amount_units
        n = self.n_timesteps

        self.add_input(
            "max_charge_rate",
            val=self.config.max_charge_rate,
            units=rate_units,
            desc="Storage charge rate",
        )
        self.add_input(
            "storage_capacity",
            val=self.config.max_capacity,
            units=amount_units,
            desc="Maximum storage capacity",
        )
        if not self.config.charge_equals_discharge:
            self.add_input(
                "max_discharge_rate",
                val=self.config.max_discharge_rate,
                units=rate_units,
                desc="Storage discharge rate",
            )

        self.output_names = {
            "charge": f"dispatch_{commodity}_charge",
            "discharge": f"dispatch_{commodity}_discharge",
            "stored": f"dispatch_{commodity}_stored",
            "unmet": f"dispatch_{commodity}_unmet_demand",
            "curtail": f"dispatch_{commodity}_curtailed",
            "delivered": f"dispatch_{commodity}_delivered",
        }
        descriptions = {
            "charge": f"{commodity} drawn from the input to charge storage",
            "discharge": f"{commodity} supplied by storage",
            "stored": f"{commodity} stored at the end of each time step",
            "unmet": f"{commodity} set-point not met",
            "curtail": f"available {commodity} that was not used",
            "delivered": f"{commodity} delivered toward the set-point",
        }
        for key, name in self.output_names.items():
            self.add_output(
                name,
                val=0.0,
                shape=n,
                units=amount_units if key == "stored" else rate_units,
                desc=descriptions[key],
            )
        self.add_output(
            "dispatch_cost",
            val=0.0,
            units="USD",
            desc="Realized dispatch cost (unmet demand, cycling, and curtailment)",
        )

        dt_seconds = float(self.options["plant_config"]["plant"]["simulation"].get("dt", 3600))
        self.dt_amount = float(
            om_units.convert_units(dt_seconds, "s", f"({amount_units})/({rate_units})")
        )

        penalty = np.asarray(self.config.unmet_demand_penalty, dtype=float)
        if penalty.ndim == 0:
            penalty = np.full(n, float(penalty))
        if penalty.size != n:
            msg = f"unmet_demand_penalty must be a scalar or have length {n}, got {penalty.size}"
            raise ValueError(msg)
        self.unmet_penalty = penalty
        self.cost_scale = max(
            float(np.max(np.abs(penalty))),
            self.config.charge_cost,
            self.config.discharge_cost,
            self.config.curtailment_cost,
            1e-12,
        )
        self.flow_scale = self._default_flow_scale()

        self._input_names = [f"{commodity}_in", f"{commodity}_set_point"]
        self._input_names += ["storage_capacity", "max_charge_rate"]
        if not self.config.charge_equals_discharge:
            self._input_names.append("max_discharge_rate")
        self._continuous_outputs = [f"{commodity}_command_value"]
        self._continuous_outputs += [*self.output_names.values(), "dispatch_cost"]

        if self.config.derivatives == "adjoint":
            self.matrix_free = True
        else:
            self.matrix_free = False
            if self.config.derivatives == "fd":
                self.declare_partials(self._continuous_outputs, self._input_names, method="fd")

        self._windows = {}
        self._solutions = []
        self._no_storage_masks = None
        self._input_active = (np.ones(n, dtype=bool), np.ones(n, dtype=bool))

    def _default_flow_scale(self):
        if self.config.flow_scale is not None:
            return float(self.config.flow_scale)
        demand = self.config.demand_profile
        if isinstance(demand, dict):
            demand = demand["demand"]
        candidates = [
            float(self.config.max_charge_rate or 0.0),
            float(self.config.max_discharge_rate or 0.0),
            float(np.mean(np.abs(np.asarray(demand, dtype=float)))),
        ]
        scale = max(candidates)
        return scale if scale > 0 else 1.0

    def _window(self, n_steps):
        """Return the cached Pyomo model and KKT adapter for a window of ``n_steps``."""
        if n_steps not in self._windows:
            from h2integrate.control.differentiable.kkt_sensitivity import KKTSensitivity

            model = self._build_window_model(n_steps)
            params = [model.commodity_in[t] for t in model.T]
            params += [model.demand[t] for t in model.T]
            params += [model.capacity, model.charge_rate, model.discharge_rate, model.stored_init]
            sensitivity = KKTSensitivity(
                model,
                params,
                solver_options=self.config.solver_options,
                barrier_mu=self.config.smoothing_barrier_mu,
                solver_backend=self.config.solver_backend,
            )
            self._windows[n_steps] = _DispatchWindow(sensitivity, model, n_steps)
        return self._windows[n_steps]

    def _build_window_model(self, n_steps):
        """Build the normalized dispatch quadratic program for one window."""
        cfg = self.config
        dt = self.dt_amount
        eta_c = float(cfg.charge_efficiency)
        eta_d = float(cfg.discharge_efficiency)
        soc_min = float(cfg.min_soc_fraction)
        soc_max = float(cfg.max_soc_fraction)
        weights = cfg.time_weighting_factor ** np.arange(n_steps)
        charge_coef = cfg.charge_cost / self.cost_scale
        discharge_coef = cfg.discharge_cost / self.cost_scale
        curtail_coef = cfg.curtailment_cost / self.cost_scale
        quad = cfg.smoothing_quadratic_weight

        m = pyo.ConcreteModel(name="differentiable_dispatch")
        m.T = pyo.Set(initialize=range(n_steps), ordered=True)

        # Parameters are fixed variables so the KKT adapter can differentiate with respect to
        # them. Values are set before every solve.
        m.commodity_in = pyo.Var(m.T, initialize=0.0)
        m.demand = pyo.Var(m.T, initialize=0.0)
        m.capacity = pyo.Var(initialize=1.0)
        m.charge_rate = pyo.Var(initialize=1.0)
        m.discharge_rate = pyo.Var(initialize=1.0)
        m.stored_init = pyo.Var(initialize=0.0)
        m.unmet_penalty = pyo.Param(m.T, mutable=True, initialize=1.0)

        nonneg = {"within": pyo.NonNegativeReals, "initialize": 0.0}
        for name in (
            "charge",
            "discharge",
            "curtail",
            "unmet",
            "delivered",
            "slack_soc_min",
            "slack_soc_max",
            "slack_charge_rate",
            "slack_discharge_rate",
            "slack_charge_available",
            "slack_curtail",
        ):
            m.add_component(name, pyo.Var(m.T, **nonneg))
        m.stored = pyo.Var(m.T, initialize=0.0)

        m.supply_balance = pyo.Constraint(
            m.T,
            rule=lambda m, t: (
                m.commodity_in[t] - m.curtail[t] - m.charge[t] + m.discharge[t] - m.delivered[t]
                == 0
            ),
        )
        m.demand_balance = pyo.Constraint(
            m.T, rule=lambda m, t: m.delivered[t] + m.unmet[t] - m.demand[t] == 0
        )

        def storage_balance(m, t):
            previous = m.stored_init if t == 0 else m.stored[t - 1]
            return m.stored[t] - previous - (eta_c * m.charge[t] - m.discharge[t] / eta_d) * dt == 0

        m.storage_balance = pyo.Constraint(m.T, rule=storage_balance)
        m.soc_min_limit = pyo.Constraint(
            m.T, rule=lambda m, t: m.stored[t] - soc_min * m.capacity - m.slack_soc_min[t] == 0
        )
        m.soc_max_limit = pyo.Constraint(
            m.T, rule=lambda m, t: soc_max * m.capacity - m.stored[t] - m.slack_soc_max[t] == 0
        )
        m.charge_rate_limit = pyo.Constraint(
            m.T, rule=lambda m, t: m.charge_rate - m.charge[t] - m.slack_charge_rate[t] == 0
        )
        m.discharge_rate_limit = pyo.Constraint(
            m.T,
            rule=lambda m, t: m.discharge_rate - m.discharge[t] - m.slack_discharge_rate[t] == 0,
        )
        m.charge_available_limit = pyo.Constraint(
            m.T,
            rule=lambda m, t: m.commodity_in[t] - m.charge[t] - m.slack_charge_available[t] == 0,
        )
        m.curtail_limit = pyo.Constraint(
            m.T, rule=lambda m, t: m.commodity_in[t] - m.curtail[t] - m.slack_curtail[t] == 0
        )

        m.objective = pyo.Objective(
            expr=sum(
                weights[t]
                * dt
                * (
                    m.unmet_penalty[t] * m.unmet[t]
                    + charge_coef * m.charge[t]
                    + discharge_coef * m.discharge[t]
                    + curtail_coef * m.curtail[t]
                )
                + quad * (m.charge[t] ** 2 + m.discharge[t] ** 2)
                for t in m.T
            ),
            sense=pyo.minimize,
        )
        return m

    def _read_inputs(self, inputs):
        commodity = self.config.commodity
        commodity_in = np.asarray(inputs[f"{commodity}_in"], dtype=float)
        demand = np.asarray(inputs[f"{commodity}_set_point"], dtype=float)
        capacity = float(inputs["storage_capacity"][0])
        charge_rate = float(inputs["max_charge_rate"][0])
        if self.config.charge_equals_discharge:
            discharge_rate = charge_rate
        else:
            discharge_rate = float(inputs["max_discharge_rate"][0])
        return commodity_in, demand, capacity, charge_rate, discharge_rate

    def _window_starts(self):
        n = self.n_timesteps
        if self.config.dispatch_horizon_mode == "full_horizon":
            return [(0, n, n)]
        horizon = self.config.n_horizon_window
        control = self.config.n_control_window
        return [(s, min(horizon, n - s), min(control, n - s)) for s in range(0, n, control)]

    def compute(self, inputs, outputs):
        """Solve the dispatch problem and write the dispatch outputs.

        Args:
            inputs (om.vectors.Vector): Input values.
            outputs (om.vectors.Vector): Output values.
        """
        commodity = self.config.commodity
        commodity_in, demand, capacity, charge_rate, discharge_rate = self._read_inputs(inputs)
        # Negative availability or demand is not physical; clip it and zero its derivatives.
        self._input_active = (commodity_in >= 0.0, demand >= 0.0)
        commodity_in = np.maximum(commodity_in, 0.0)
        demand = np.maximum(demand, 0.0)
        if min(capacity, charge_rate, discharge_rate) < 0:
            raise om.AnalysisError("storage_capacity and charge rates must be >= 0")

        n = self.n_timesteps
        results = {name: np.zeros(n) for name in _WINDOW_OUTPUT_VARS}
        self._solutions = []
        self._no_storage_masks = None

        if capacity <= 0.0 or max(charge_rate, discharge_rate) <= 0.0:
            # Storage cannot operate; deliver what is available without an optimization.
            delivered = np.minimum(commodity_in, demand)
            results["delivered"] = delivered
            results["unmet"] = demand - delivered
            results["curtail"] = commodity_in - delivered
            results["stored"] = np.zeros(n)
            self._no_storage_masks = commodity_in <= demand
        else:
            scale = self.flow_scale
            stored_init = self.config.init_soc_fraction * capacity
            for start, n_steps, n_apply in self._window_starts():
                window = self._window(n_steps)
                model = window.model
                stop = start + n_steps
                for t in range(n_steps):
                    model.unmet_penalty[t] = self.unmet_penalty[start + t] / self.cost_scale
                params = np.concatenate(
                    [
                        commodity_in[start:stop],
                        demand[start:stop],
                        [capacity, charge_rate, discharge_rate, stored_init],
                    ]
                )
                try:
                    solution = window.sensitivity.solve(params / scale)
                except RuntimeError as err:
                    raise om.AnalysisError(f"Dispatch optimization failed: {err}") from err
                x = solution.x * scale
                for name in _WINDOW_OUTPUT_VARS:
                    results[name][start : start + n_apply] = x[window.index[name][:n_apply]]
                stored_init = results["stored"][start + n_apply - 1]
                self._solutions.append((start, n_steps, n_apply, window, solution))

        outputs[f"{commodity}_command_value"] = results["discharge"] - results["charge"]
        for key, name in self.output_names.items():
            outputs[name] = results[key]
        outputs["dispatch_cost"] = self._dispatch_cost(results)

    def _dispatch_cost(self, results):
        cfg = self.config
        return self.dt_amount * (
            np.dot(self.unmet_penalty, results["unmet"])
            + cfg.charge_cost * np.sum(results["charge"])
            + cfg.discharge_cost * np.sum(results["discharge"])
            + cfg.curtailment_cost * np.sum(results["curtail"])
        )

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """Matrix-free derivatives from the KKT conditions of each dispatch window.

        Args:
            inputs (om.vectors.Vector): Input values.
            d_inputs (om.vectors.Vector): Input seeds (fwd) or accumulated sensitivities (rev).
            d_outputs (om.vectors.Vector): Output sensitivities (fwd) or seeds (rev).
            mode (str): ``"fwd"`` or ``"rev"``.
        """
        if mode == "fwd":
            seeds = self._input_seeds(d_inputs)
            d_vars = self._forward(*seeds)
            self._write_output_perturbations(d_vars, d_outputs)
        else:
            bars = self._output_seeds(d_outputs)
            sensitivities = self._reverse(bars)
            self._accumulate_input_sensitivities(sensitivities, d_inputs)

    def _input_seeds(self, d_inputs):
        commodity = self.config.commodity
        n = self.n_timesteps

        def seed(name, size):
            if name in d_inputs:
                return np.asarray(d_inputs[name], dtype=float).copy()
            return np.zeros(size)

        d_in = seed(f"{commodity}_in", n) * self._input_active[0]
        d_demand = seed(f"{commodity}_set_point", n) * self._input_active[1]
        d_capacity = float(seed("storage_capacity", 1)[0])
        d_charge_rate = float(seed("max_charge_rate", 1)[0])
        if self.config.charge_equals_discharge:
            d_discharge_rate = d_charge_rate
        else:
            d_discharge_rate = float(seed("max_discharge_rate", 1)[0])
        return d_in, d_demand, d_capacity, d_charge_rate, d_discharge_rate

    def _forward(self, d_in, d_demand, d_capacity, d_charge_rate, d_discharge_rate):
        n = self.n_timesteps
        d_vars = {name: np.zeros(n) for name in _WINDOW_OUTPUT_VARS}
        if self._no_storage_masks is not None:
            short = self._no_storage_masks
            d_vars["delivered"] = np.where(short, d_in, d_demand)
            d_vars["unmet"] = d_demand - d_vars["delivered"]
            d_vars["curtail"] = d_in - d_vars["delivered"]
            return d_vars

        d_stored_init = self.config.init_soc_fraction * d_capacity
        for start, n_steps, n_apply, window, solution in self._solutions:
            stop = start + n_steps
            d_params = np.concatenate(
                [
                    d_in[start:stop],
                    d_demand[start:stop],
                    [d_capacity, d_charge_rate, d_discharge_rate, d_stored_init],
                ]
            )
            dx = solution.linearization.jvp(d_params)
            for name in _WINDOW_OUTPUT_VARS:
                d_vars[name][start : start + n_apply] = dx[window.index[name][:n_apply]]
            d_stored_init = d_vars["stored"][start + n_apply - 1]
        return d_vars

    def _write_output_perturbations(self, d_vars, d_outputs):
        commodity = self.config.commodity
        command = f"{commodity}_command_value"
        if command in d_outputs:
            d_outputs[command] += d_vars["discharge"] - d_vars["charge"]
        for key, name in self.output_names.items():
            if name in d_outputs:
                d_outputs[name] += d_vars[key]
        if "dispatch_cost" in d_outputs:
            d_outputs["dispatch_cost"] += self._dispatch_cost(d_vars)

    def _output_seeds(self, d_outputs):
        cfg = self.config
        n = self.n_timesteps
        bars = {name: np.zeros(n) for name in _WINDOW_OUTPUT_VARS}
        command = f"{cfg.commodity}_command_value"
        if command in d_outputs:
            seed = np.asarray(d_outputs[command], dtype=float)
            bars["discharge"] += seed
            bars["charge"] -= seed
        for key, name in self.output_names.items():
            if name in d_outputs:
                bars[key] += np.asarray(d_outputs[name], dtype=float)
        if "dispatch_cost" in d_outputs:
            cost_seed = float(np.asarray(d_outputs["dispatch_cost"]).ravel()[0]) * self.dt_amount
            bars["unmet"] += cost_seed * self.unmet_penalty
            bars["charge"] += cost_seed * cfg.charge_cost
            bars["discharge"] += cost_seed * cfg.discharge_cost
            bars["curtail"] += cost_seed * cfg.curtailment_cost
        return bars

    def _reverse(self, bars):
        n = self.n_timesteps
        bar_in = np.zeros(n)
        bar_demand = np.zeros(n)
        bar_capacity = 0.0
        bar_charge_rate = 0.0
        bar_discharge_rate = 0.0

        if self._no_storage_masks is not None:
            short = self._no_storage_masks
            bar_delivered = bars["delivered"] - bars["unmet"] - bars["curtail"]
            bar_in = bars["curtail"] + np.where(short, bar_delivered, 0.0)
            bar_demand = bars["unmet"] + np.where(short, 0.0, bar_delivered)
            return bar_in, bar_demand, bar_capacity, bar_charge_rate, bar_discharge_rate

        bar_stored_init = 0.0
        for start, n_steps, n_apply, window, solution in reversed(self._solutions):
            stop = start + n_steps
            x_bar = np.zeros(window.sensitivity.n_x)
            for name in _WINDOW_OUTPUT_VARS:
                x_bar[window.index[name][:n_apply]] += bars[name][start : start + n_apply]
            x_bar[window.index["stored"][n_apply - 1]] += bar_stored_init
            p_bar = solution.linearization.vjp(x_bar)
            bar_in[start:stop] += p_bar[:n_steps]
            bar_demand[start:stop] += p_bar[n_steps : 2 * n_steps]
            bar_capacity += p_bar[2 * n_steps]
            bar_charge_rate += p_bar[2 * n_steps + 1]
            bar_discharge_rate += p_bar[2 * n_steps + 2]
            bar_stored_init = p_bar[2 * n_steps + 3]
        bar_capacity += self.config.init_soc_fraction * bar_stored_init
        return bar_in, bar_demand, bar_capacity, bar_charge_rate, bar_discharge_rate

    def _accumulate_input_sensitivities(self, sensitivities, d_inputs):
        commodity = self.config.commodity
        bar_in, bar_demand, bar_capacity, bar_charge_rate, bar_discharge_rate = sensitivities
        bar_in = bar_in * self._input_active[0]
        bar_demand = bar_demand * self._input_active[1]
        if f"{commodity}_in" in d_inputs:
            d_inputs[f"{commodity}_in"] += bar_in
        if f"{commodity}_set_point" in d_inputs:
            d_inputs[f"{commodity}_set_point"] += bar_demand
        if "storage_capacity" in d_inputs:
            d_inputs["storage_capacity"] += bar_capacity
        if self.config.charge_equals_discharge:
            if "max_charge_rate" in d_inputs:
                d_inputs["max_charge_rate"] += bar_charge_rate + bar_discharge_rate
        else:
            if "max_charge_rate" in d_inputs:
                d_inputs["max_charge_rate"] += bar_charge_rate
            if "max_discharge_rate" in d_inputs:
                d_inputs["max_discharge_rate"] += bar_discharge_rate
