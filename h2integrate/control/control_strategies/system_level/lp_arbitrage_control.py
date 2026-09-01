import numpy as np
import pyomo.environ as pyo
from attrs import field, define
from pyomo.common.errors import ApplicationError

from h2integrate.core.utilities import BaseConfig, merge_shared_inputs
from h2integrate.control.control_strategies.system_level.system_level_control_base import (
    SystemLevelControlBase,
    _get_sell_price_default_and_shape,
)


@define(kw_only=True)
class LPArbitrageControlConfig(BaseConfig):
    """Configuration for :class:`LPArbitrageControl`.

    Attributes:
        n_control_window_hours (int): Length of each rolling optimization window
            in hours. The state of charge at the end of one window becomes the
            initial state of charge of the next. Defaults to 24.
        cost_per_tech (dict): Marginal-cost specification for each dispatchable
            technology, using the same syntax as ``CostMinimizationControl``.
            Each entry may be a numeric value, ``"buy_price"``, ``"VarOpEx"``,
            or ``"feedstock"``. Defaults to an empty dict.
        value_of_lost_load (float): Penalty applied to unmet demand, in
            ``USD/(commodity_rate_unit*h)``. Acts as the price of the unmet-demand
            slack variable that keeps the linear program feasible. Defaults to
            10.0 (that is, ``$10/kWh`` for electricity).
        storage_cycle_cost (float): Cost applied to storage throughput (charge
            plus discharge), in ``USD/(commodity_rate_unit*h)``. Represents
            degradation and breaks ties between equivalent dispatch schedules.
            Defaults to 0.0.
        terminal_soc_price_factor (float): Scaling applied to the end-of-window
            state-of-charge valuation. The stored commodity remaining at the end
            of a window is valued at ``factor * mean_window_price *
            discharge_efficiency``, which prevents the optimizer from emptying
            storage at every window boundary. Set to 0.0 to disable. Defaults
            to 1.0.
        solver_name (str): Pyomo solver used to solve each window. Defaults to
            ``"glpk"``.
        solver_options (dict): Additional options passed through to the solver.
            Defaults to an empty dict.
    """

    n_control_window_hours: int = field(default=24, converter=int)
    cost_per_tech: dict = field(default={})
    value_of_lost_load: float = field(default=10.0, converter=float)
    storage_cycle_cost: float = field(default=0.0, converter=float)
    terminal_soc_price_factor: float = field(default=1.0, converter=float)
    solver_name: str = field(default="glpk")
    solver_options: dict = field(default={})


class LPArbitrageControl(SystemLevelControlBase):
    """System-level controller that co-optimizes dispatch with a linear program.

    Solves a rolling-horizon linear program that maximizes plant profit against
    a time-varying sale price (for example, a locational marginal price series).
    It is aimed at solar-plus-storage arbitrage but is written generically over
    the technology classifiers, so any mix of fixed, flexible, dispatchable, and
    storage technologies on a single commodity bus is supported.

    Unlike the heuristic controllers, which apply a fixed
    fixed -> flexible -> storage -> dispatchable priority order, this controller
    decides storage charge/discharge, dispatchable output, and grid export
    *simultaneously*. That matters for arbitrage, because whether it is worth
    charging in a given hour depends on what the dispatchable technologies would
    otherwise cost in the hours the storage would later discharge into.

    **Decision variables** (per timestep, per rolling window):

    - ``charge[s]`` / ``discharge[s]`` for each storage technology, in
      commodity rate units on the *bus* side of the storage (this matches the
      sign convention of the storage performance model, whose command value is
      also bus-side).
    - ``soc[s]``, the stored quantity in commodity amount units.
    - ``dispatch[d]`` for each dispatchable technology, bounded by its rated
      production.
    - ``export``, the quantity sold through the export technology, bounded by
      that technology's interconnection size.
    - ``curtail``, surplus that cannot be exported because the interconnection
      is saturated.
    - ``unmet``, a demand shortfall slack penalized at ``value_of_lost_load``.
      This exists so the program is always feasible; a well-posed plant should
      converge with ``unmet`` at zero.

    **Objective** (maximized)::

        sum_t dt * (  sell_price[t] * export[t]
                    - sum_d marginal_cost[d, t] * dispatch[d, t]
                    - value_of_lost_load * unmet[t]
                    - storage_cycle_cost * sum_s (charge[s, t] + discharge[s, t]) )
        + sum_s terminal_price[s] * soc[s, last]

    **Constraints**:

    1. Commodity balance at each timestep.
    2. Storage state-of-charge dynamics, including charge/discharge efficiency.
    3. Dispatchable output bounded by rated production.
    4. Total charging bounded by the commodity actually available on the bus.
       This mirrors the ``charge_available`` clip inside the storage performance
       model, so the schedule the controller commands is one the storage model
       can actually follow.

    Fixed and flexible technologies are treated as must-run parameters rather
    than decision variables. Flexible performance models (wind, solar) are
    resource-driven and do not respond to a set-point, so their output is
    availability, not a choice. Economic curtailment is therefore represented by
    the ``curtail`` variable rather than by lowering a set-point.

    Charging from the grid is permitted: the ``dispatch`` of an import
    technology raises the right-hand side of the charge-availability
    constraint. For this to be physically realizable the plant topology must
    actually route the import technology into the storage technology's
    commodity input.

    Configuration is read from
    ``plant_config["system_level_control"]["control_parameters"]``. The export
    technology is named by ``plant_config["system_level_control"]["export_component"]``
    and its ``electricity_sell_price`` supplies the price series.
    """

    # Retries for transient solver-interface faults (see _solve_window).
    _solve_attempts = 3

    def setup(self):
        super().setup()

        plant_config = self.options["plant_config"]
        slc_config = plant_config["system_level_control"]

        self.config = LPArbitrageControlConfig.from_dict(
            slc_config.get("control_parameters", {}),
            additional_cls_name=self.__class__.__name__,
        )

        self.plant_life = int(plant_config["plant"]["plant_life"])
        self.dt_h = plant_config["plant"]["simulation"]["dt"] / 3600.0

        # --- Export technology: supplies both the price series and the limit ---
        self.export_tech = self.options["slc_topology"].get("export_tech", None)
        if self.export_tech is None:
            raise ValueError(
                f"{self.__class__.__name__} requires an export technology. Set "
                "``system_level_control['export_component']`` in the plant "
                "configuration to the name of the technology that sells the "
                "controlled commodity (for example a GridPerformanceModel "
                "instance connected to the demand component's "
                "``unused_{commodity}_out``)."
            )

        default_price, price_shape = _get_sell_price_default_and_shape(
            self.options["tech_config"],
            self.export_tech,
            self.n_timesteps,
            self.plant_life,
        )
        self.add_input(
            f"{self.export_tech}_sell_price",
            val=default_price,
            shape=price_shape,
            units=f"USD/({self.commodity_rate_units}*h)",
            desc=f"Sale price of {self.commodity} exported through {self.export_tech}",
        )

        self.export_limit = self._read_export_limit()

        # Marginal-cost inputs for dispatchable techs (shared with the
        # cost-aware heuristic controllers).
        self._setup_marginal_costs()

        # Storage design parameters are read from the technology config rather
        # than from connected inputs. Connected values are zero until the
        # storage model has executed once, which would make the first solver
        # iteration degenerate.
        self.storage_params = self._read_storage_parameters()

        # Only technologies producing the controlled commodity participate.
        self.lp_storage_techs = list(self.storage_params.keys())
        self.lp_dispatchable_techs = [
            tech
            for tech in self.dispatchable_techs
            if self.commodity in self._get_commodity_for_tech(tech)
        ]
        self.lp_must_run_techs = [
            tech
            for tech in (self.fixed_techs + self.flexible_techs)
            if self.commodity in self._get_commodity_for_tech(tech)
        ]

        self.window_len = self._resolve_window_length()

        # Built lazily on the first compute() so setup stays cheap and the
        # solver is only required when the model is actually run.
        self._lp_model = None
        self._solver = None
        self._cached_key = None
        self._cached_set_points = None

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def _read_export_limit(self):
        """Return the export technology's interconnection size.

        Returns:
            float: Maximum export rate in ``commodity_rate_units``.

        Raises:
            ValueError: If the export technology declares no interconnection size.
        """
        tech_def = self.options["tech_config"]["technologies"][self.export_tech]
        model_inputs = tech_def.get("model_inputs", {})
        all_params = {
            **model_inputs.get("shared_parameters", {}),
            **model_inputs.get("performance_parameters", {}),
            **model_inputs.get("cost_parameters", {}),
        }
        if "interconnection_size" not in all_params:
            raise ValueError(
                f"Export technology '{self.export_tech}' does not define "
                "``interconnection_size``, which is required to bound the "
                "export decision variable."
            )
        return float(all_params["interconnection_size"])

    def _read_storage_parameters(self):
        """Read storage sizing, efficiency, and state-of-charge bounds from config.

        Returns:
            dict[str, dict]: Per-technology parameter dictionaries for every
            storage technology producing the controlled commodity.
        """
        storage_params = {}
        technologies = self.options["tech_config"]["technologies"]

        for tech_name in self.storage_techs:
            if self.commodity not in self._get_commodity_for_tech(tech_name):
                continue

            params = merge_shared_inputs(technologies[tech_name]["model_inputs"], "performance")

            charge_efficiency = params.get("charge_efficiency", None)
            discharge_efficiency = params.get("discharge_efficiency", None)
            if charge_efficiency is None or discharge_efficiency is None:
                round_trip_efficiency = params.get("round_trip_efficiency", None)
                if round_trip_efficiency is None:
                    raise ValueError(
                        f"Storage technology '{tech_name}' must define either "
                        "``round_trip_efficiency`` or both ``charge_efficiency`` "
                        f"and ``discharge_efficiency`` for {self.__class__.__name__}."
                    )
                charge_efficiency = float(np.sqrt(round_trip_efficiency))
                discharge_efficiency = float(np.sqrt(round_trip_efficiency))

            max_charge_rate = float(params["max_charge_rate"])
            max_discharge_rate = params.get("max_discharge_rate", None)
            if max_discharge_rate is None:
                max_discharge_rate = max_charge_rate

            storage_params[tech_name] = {
                "capacity": float(params["max_capacity"]),
                "max_charge_rate": max_charge_rate,
                "max_discharge_rate": float(max_discharge_rate),
                "charge_efficiency": float(charge_efficiency),
                "discharge_efficiency": float(discharge_efficiency),
                "min_soc_fraction": float(params.get("min_soc_fraction", 0.0)),
                "max_soc_fraction": float(params.get("max_soc_fraction", 1.0)),
                "init_soc_fraction": float(params.get("init_soc_fraction", 0.0)),
            }

        return storage_params

    def _resolve_window_length(self):
        """Convert the configured window length in hours to a number of timesteps.

        Returns:
            int: Window length in timesteps, clipped to the simulation length.
        """
        window_len = round(self.config.n_control_window_hours / self.dt_h)
        if window_len < 1:
            raise ValueError(
                f"n_control_window_hours ({self.config.n_control_window_hours}) is "
                f"shorter than a single timestep ({self.dt_h} h)."
            )
        return min(window_len, self.n_timesteps)

    # ------------------------------------------------------------------
    # Linear program construction
    # ------------------------------------------------------------------

    def _build_lp_model(self):
        """Build the rolling-window linear program.

        The model is constructed once and re-solved for every window with
        updated mutable parameters, which avoids rebuilding Pyomo objects
        thousands of times over the course of a nonlinear solver loop.

        Returns:
            pyo.ConcreteModel: The window-scale optimization model.
        """
        storage_params = self.storage_params
        window_len = self.window_len
        dt_h = self.dt_h

        model = pyo.ConcreteModel(name="system_level_arbitrage")

        model.T = pyo.RangeSet(0, window_len - 1)
        model.S = pyo.Set(initialize=self.lp_storage_techs, ordered=True)
        model.D = pyo.Set(initialize=self.lp_dispatchable_techs, ordered=True)

        # --- Mutable parameters, refreshed for every window ---------------
        model.must_run = pyo.Param(model.T, initialize=0.0, mutable=True)
        model.demand = pyo.Param(model.T, initialize=0.0, mutable=True)
        model.sell_price = pyo.Param(model.T, initialize=0.0, mutable=True)
        model.marginal_cost = pyo.Param(model.D, model.T, initialize=0.0, mutable=True)
        model.rated = pyo.Param(model.D, initialize=0.0, mutable=True)
        model.soc_init = pyo.Param(model.S, initialize=0.0, mutable=True)
        model.terminal_price = pyo.Param(model.S, initialize=0.0, mutable=True)

        # --- Decision variables -------------------------------------------
        def _charge_bounds(_, tech_name, __):
            return 0.0, storage_params[tech_name]["max_charge_rate"]

        def _discharge_bounds(_, tech_name, __):
            return 0.0, storage_params[tech_name]["max_discharge_rate"]

        def _soc_bounds(_, tech_name, __):
            params = storage_params[tech_name]
            return (
                params["min_soc_fraction"] * params["capacity"],
                params["max_soc_fraction"] * params["capacity"],
            )

        model.charge = pyo.Var(model.S, model.T, bounds=_charge_bounds)
        model.discharge = pyo.Var(model.S, model.T, bounds=_discharge_bounds)
        model.soc = pyo.Var(model.S, model.T, bounds=_soc_bounds)
        model.dispatch = pyo.Var(model.D, model.T, domain=pyo.NonNegativeReals)
        model.export = pyo.Var(model.T, bounds=(0.0, self.export_limit))
        model.curtail = pyo.Var(model.T, domain=pyo.NonNegativeReals)
        model.unmet = pyo.Var(model.T, domain=pyo.NonNegativeReals)

        # --- Constraints ---------------------------------------------------
        def _balance_rule(m, t):
            supply = (
                m.must_run[t]
                + sum(m.discharge[s, t] for s in m.S)
                + sum(m.dispatch[d, t] for d in m.D)
                + m.unmet[t]
            )
            use = m.demand[t] + sum(m.charge[s, t] for s in m.S) + m.export[t] + m.curtail[t]
            return supply == use

        model.balance = pyo.Constraint(model.T, rule=_balance_rule)

        def _soc_rule(m, s, t):
            previous = m.soc_init[s] if t == 0 else m.soc[s, t - 1]
            charged = m.charge[s, t] * storage_params[s]["charge_efficiency"]
            discharged = m.discharge[s, t] / storage_params[s]["discharge_efficiency"]
            return m.soc[s, t] == previous + dt_h * (charged - discharged)

        model.soc_balance = pyo.Constraint(model.S, model.T, rule=_soc_rule)

        def _rated_rule(m, d, t):
            return m.dispatch[d, t] <= m.rated[d]

        model.rated_limit = pyo.Constraint(model.D, model.T, rule=_rated_rule)

        def _charge_availability_rule(m, t):
            # Mirrors the ``charge_available`` clip in the storage performance
            # model: storage can only absorb commodity that is present on the
            # bus this timestep. Import technologies count toward availability,
            # which is what makes grid charging possible.
            return sum(m.charge[s, t] for s in m.S) <= m.must_run[t] + sum(
                m.dispatch[d, t] for d in m.D
            )

        model.charge_availability = pyo.Constraint(model.T, rule=_charge_availability_rule)

        # --- Objective -----------------------------------------------------
        def _objective_rule(m):
            revenue = sum(m.sell_price[t] * m.export[t] for t in m.T)
            generation_cost = sum(
                m.marginal_cost[d, t] * m.dispatch[d, t] for d in m.D for t in m.T
            )
            unmet_cost = self.config.value_of_lost_load * sum(m.unmet[t] for t in m.T)
            cycle_cost = self.config.storage_cycle_cost * sum(
                m.charge[s, t] + m.discharge[s, t] for s in m.S for t in m.T
            )
            terminal_value = sum(m.terminal_price[s] * m.soc[s, window_len - 1] for s in m.S)
            return dt_h * (revenue - generation_cost - unmet_cost - cycle_cost) + terminal_value

        model.objective = pyo.Objective(rule=_objective_rule, sense=pyo.maximize)

        return model

    def _get_solver(self):
        """Return the configured Pyomo solver, validating availability once.

        Returns:
            The Pyomo solver object.

        Raises:
            RuntimeError: If the configured solver is not available.
        """
        if self._solver is None:
            solver = pyo.SolverFactory(self.config.solver_name)
            if not solver.available(exception_flag=False):
                raise RuntimeError(
                    f"Pyomo solver '{self.config.solver_name}' is required by "
                    f"{self.__class__.__name__} but is not available. Install it "
                    "(for example ``conda install -c conda-forge glpk``) or set "
                    "``solver_name`` in the controller's control_parameters."
                )
            self._solver = solver
        return self._solver

    def _solve_window(self, model, start):
        """Solve one dispatch window, retrying transient solver-interface faults.

        Args:
            model (pyo.ConcreteModel): The window model, already populated.
            start (int): Index of the first timestep in the window, used for
                error reporting.

        Raises:
            RuntimeError: If the window cannot be solved to optimality, or if
                the solver interface keeps failing after ``_solve_attempts``.
        """
        last_exc = None
        for _ in range(self._solve_attempts):
            try:
                results = self._get_solver().solve(model, options=self.config.solver_options)
            except (OSError, ApplicationError) as exc:
                # Shell-based solver interfaces are flaky on Windows under rapid
                # repeated invocation: the temporary files they exchange with the
                # solver process are sometimes still locked when Pyomo reads or
                # deletes them. The formulation is unaffected, so retry.
                last_exc = exc
                continue

            termination = results.solver.termination_condition
            if termination != pyo.TerminationCondition.optimal:
                raise RuntimeError(
                    f"{self.__class__.__name__} failed to solve the dispatch window "
                    f"starting at timestep {start}. Solver '{self.config.solver_name}' "
                    f"reported termination condition '{termination}'."
                )
            return

        raise RuntimeError(
            f"{self.__class__.__name__} could not run solver "
            f"'{self.config.solver_name}' for the dispatch window starting at "
            f"timestep {start} after {self._solve_attempts} attempts. Last error: "
            f"{last_exc!r}"
        )

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------

    def _broadcast_price(self, price):
        """Expand a price input of any supported shape to one value per timestep.

        Args:
            price (np.ndarray): Raw price input, of shape ``(n_timesteps,)``,
                ``(plant_life,)``, or ``(1,)``.

        Returns:
            np.ndarray: Price array of shape ``(n_timesteps,)``.
        """
        price = np.asarray(price, dtype=float)
        if price.shape == (self.n_timesteps,):
            return price
        if price.shape == (self.plant_life,):
            # Per-year price: the first year represents operating conditions.
            return np.full(self.n_timesteps, price[0])
        return np.broadcast_to(price, self.n_timesteps).copy()

    def compute(self, inputs, outputs):
        commodity = self.commodity
        n_timesteps = self.n_timesteps
        window_len = self.window_len

        demand = np.asarray(inputs[self.demand_input_name], dtype=float)
        sell_price = self._broadcast_price(inputs[f"{self.export_tech}_sell_price"])

        # Must-run production: fixed techs plus resource-driven flexible techs.
        must_run = np.zeros(n_timesteps)
        for tech_name in self.lp_must_run_techs:
            must_run += np.asarray(inputs[f"{tech_name}_{commodity}_out"], dtype=float)

        # Flexible techs are commanded at rated production; their performance
        # models are resource-limited, so this is a pass-through rather than a
        # dispatch decision.
        for tech_name in self.flexible_techs:
            for tech_commodity in self._get_commodity_for_tech(tech_name):
                rated_name = f"{tech_name}_rated_{tech_commodity}_production"
                set_point_name = f"{tech_name}_{tech_commodity}_set_point"
                if rated_name in inputs and set_point_name in outputs:
                    outputs[set_point_name] = inputs[rated_name] * np.ones(n_timesteps)

        # Marginal costs are returned aligned with self.dispatchable_techs.
        marginal_costs = dict(
            zip(self.dispatchable_techs, self._compute_marginal_costs(inputs), strict=True)
        )

        # Initialize every controlled set-point so techs excluded from the LP
        # (for example dispatchables on another commodity) hold a defined value.
        for set_point_name in self.dispatchable_set_point_names:
            outputs[set_point_name] = np.zeros(n_timesteps)
        for set_point_name in self.storage_set_point_names:
            outputs[set_point_name] = np.zeros(n_timesteps)

        rated = np.array(
            [
                float(np.asarray(inputs[f"{tech_name}_rated_{commodity}_production"]).item())
                for tech_name in self.lp_dispatchable_techs
            ]
        )
        marginal_cost_stack = (
            np.stack([marginal_costs[tech_name] for tech_name in self.lp_dispatchable_techs])
            if self.lp_dispatchable_techs
            else np.zeros((0, n_timesteps))
        )

        # The schedule is a pure function of these arrays. This controller sits
        # in a fixed-point loop, and because flexible technologies are
        # resource-driven rather than set-point-driven, the inputs stop changing
        # after the first pass. Re-solving every window on every iteration would
        # repeat identical work.
        cache_key = (must_run, demand, sell_price, marginal_cost_stack, rated)
        if self._cached_key is not None and all(
            np.array_equal(cached, current)
            for cached, current in zip(self._cached_key, cache_key, strict=True)
        ):
            for name, values in self._cached_set_points.items():
                outputs[name] = values
            return

        if self._lp_model is None:
            self._lp_model = self._build_lp_model()
        model = self._lp_model

        # Static per-window parameters
        for tech_name, rated_value in zip(self.lp_dispatchable_techs, rated, strict=True):
            model.rated[tech_name] = float(rated_value)

        soc_state = {
            tech_name: params["init_soc_fraction"] * params["capacity"]
            for tech_name, params in self.storage_params.items()
        }

        for start in range(0, n_timesteps, window_len):
            end = min(start + window_len, n_timesteps)
            actual_len = end - start

            # A trailing partial window is padded by holding the final value so
            # the fixed-size Pyomo model can be reused; only the real timesteps
            # are written back out.
            window_index = np.arange(start, start + window_len)
            window_index = np.clip(window_index, start, end - 1)

            window_price = sell_price[window_index]

            for t in range(window_len):
                idx = window_index[t]
                model.must_run[t] = float(must_run[idx])
                model.demand[t] = float(demand[idx])
                model.sell_price[t] = float(sell_price[idx])
                for tech_name in self.lp_dispatchable_techs:
                    model.marginal_cost[tech_name, t] = float(marginal_costs[tech_name][idx])

            mean_window_price = float(window_price.mean())
            for tech_name, params in self.storage_params.items():
                model.soc_init[tech_name] = float(soc_state[tech_name])
                model.terminal_price[tech_name] = (
                    self.config.terminal_soc_price_factor
                    * mean_window_price
                    * params["discharge_efficiency"]
                )

            self._solve_window(model, start)

            for tech_name in self.lp_storage_techs:
                schedule = np.array(
                    [
                        pyo.value(model.discharge[tech_name, t])
                        - pyo.value(model.charge[tech_name, t])
                        for t in range(actual_len)
                    ]
                )
                outputs[f"{tech_name}_{commodity}_set_point"][start:end] = schedule
                soc_state[tech_name] = pyo.value(model.soc[tech_name, actual_len - 1])

            for tech_name in self.lp_dispatchable_techs:
                schedule = np.array(
                    [pyo.value(model.dispatch[tech_name, t]) for t in range(actual_len)]
                )
                outputs[f"{tech_name}_{commodity}_set_point"][start:end] = schedule

        self._cached_key = tuple(np.array(item, copy=True) for item in cache_key)
        self._cached_set_points = {
            name: np.array(outputs[name], copy=True) for name in self._var_rel_names["output"]
        }
