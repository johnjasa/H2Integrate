import shutil

import numpy as np
import pytest
import pyomo.environ as pyo

from h2integrate.core.h2integrate_model import H2IntegrateModel


EXAMPLE = "35_system_level_control/lp_arbitrage"

requires_glpk = pytest.mark.skipif(
    shutil.which("glpsol") is None,
    reason="GLPK executable 'glpsol' is not available in PATH",
)


def _make_controller(example_folder):
    """Set up the example and return its system-level controller."""
    model = H2IntegrateModel(example_folder / "solar_battery_arbitrage.yaml")
    model.setup()
    model.prob.final_setup()
    controller = model.prob.model.plant.system_level_controller
    model.prob.cleanup()
    return controller


def _fill_window(
    lp,
    controller,
    sell_price,
    must_run=0.0,
    demand=0.0,
    marginal_cost=0.02,
    rated=50_000.0,
    soc_init=None,
    export_limit=None,
):
    """Populate the mutable parameters of a built LP window model."""
    sell_price = np.broadcast_to(np.asarray(sell_price, dtype=float), (controller.window_len,))
    must_run = np.broadcast_to(np.asarray(must_run, dtype=float), (controller.window_len,))
    demand = np.broadcast_to(np.asarray(demand, dtype=float), (controller.window_len,))
    marginal_cost = np.broadcast_to(
        np.asarray(marginal_cost, dtype=float), (controller.window_len,)
    )
    if export_limit is None:
        export_limit = controller.export_limit
    export_limit = np.broadcast_to(np.asarray(export_limit, dtype=float), (controller.window_len,))

    for t in lp.T:
        lp.must_run[t] = float(must_run[t])
        lp.demand[t] = float(demand[t])
        lp.sell_price[t] = float(sell_price[t])
        lp.export_limit[t] = float(export_limit[t])
        for d in lp.D:
            lp.marginal_cost[d, t] = float(marginal_cost[t])
    for d in lp.D:
        lp.rated[d] = rated
    for s in lp.S:
        params = controller.storage_params[s]
        lp.soc_init[s] = (
            params["min_soc_fraction"] * params["capacity"] if soc_init is None else soc_init
        )


def _series(var, index, window_len):
    return np.array([pyo.value(var[index, t]) for t in range(window_len)])


@pytest.mark.unit
@pytest.mark.parametrize("example_folder,resource_example_folder", [(EXAMPLE, None)])
def test_lp_arbitrage_setup(subtests, temp_copy_of_example):
    """The controller reads its topology and storage sizing from configuration."""
    controller = _make_controller(temp_copy_of_example)

    with subtests.test("Export technology resolved from export_component"):
        assert controller.export_tech == "grid_sell"

    with subtests.test("Export limit is the interconnection size"):
        assert controller.export_limit == pytest.approx(100_000.0)

    with subtests.test("Window length from n_control_window_hours"):
        assert controller.window_len == 24

    with subtests.test("Technology classification"):
        assert controller.lp_storage_techs == ["battery"]
        assert controller.lp_dispatchable_techs == ["grid_buy"]
        assert controller.lp_grid_import_techs == ["grid_buy"]
        assert controller.lp_must_run_techs == ["added_solar"]

    with subtests.test("Sell price input is added"):
        assert "grid_sell_sell_price" in controller._var_rel_names["input"]

    with subtests.test("Default storage sizing comes from the tech config"):
        params = controller.storage_params["battery"]
        assert params["capacity"] == pytest.approx(200_000.0)
        assert params["max_charge_rate"] == pytest.approx(50_000.0)
        assert params["max_discharge_rate"] == pytest.approx(50_000.0)
        # Round-trip efficiency of 0.88 is split evenly across charge and discharge.
        assert params["charge_efficiency"] == pytest.approx(np.sqrt(0.88))
        assert params["discharge_efficiency"] == pytest.approx(np.sqrt(0.88))


CUSTOM_GRID_MODELS = """
from h2integrate.converters.grid.grid import GridCostModel, GridPerformanceModel


class CustomGridPerformanceModel(GridPerformanceModel):
    pass


class GridSellPriceOutput(GridCostModel):
    def setup(self):
        super().setup()
        self.add_output(
            "electricity_sell_price", val=0.07, shape=self.n_timesteps, units="USD/(kW*h)"
        )


class GridSellPriceInput(GridCostModel):
    def setup(self):
        super().setup()
        self.add_input(
            "electricity_sell_price", val=0.07, shape=self.n_timesteps, units="USD/(kW*h)"
        )


class GridPriceOutputs(GridCostModel):
    def setup(self):
        super().setup()
        self.add_output(
            "electricity_buy_price", val=0.05, shape=self.n_timesteps, units="USD/(MW*h)"
        )
        self.add_output("electricity_sell_price", val=0.07, units="USD/(kW*h)")
"""


def _use_unpriced_grid_sell(example_folder, cost_model="GridCostModel"):
    """Drop the configured sell price and optionally swap in a custom grid cost model."""
    (example_folder / "custom_grid.py").write_text(CUSTOM_GRID_MODELS)
    config_path = example_folder / "tech_config.yaml"
    text = config_path.read_text()
    text = text.replace(
        "        electricity_sell_price: 0.03  # $/kWh; overridden with an LMP series in the run "
        "script\n",
        "",
    )
    if cost_model != "GridCostModel":
        text = text.replace(
            "  grid_sell:\n    performance_model:\n      model: GridPerformanceModel\n"
            "    cost_model:\n      model: GridCostModel\n",
            "  grid_sell:\n    performance_model:\n      model: GridPerformanceModel\n"
            f"    cost_model:\n      model: {cost_model}\n      model_location: custom_grid.py\n",
        )
    config_path.write_text(text)


@pytest.mark.unit
@pytest.mark.parametrize("example_folder,resource_example_folder", [(EXAMPLE, None)])
@pytest.mark.parametrize("cost_model", ["GridSellPriceOutput", "GridSellPriceInput"])
def test_lp_arbitrage_sell_price_from_custom_model(temp_copy_of_example, cost_model):
    """A custom export model's sell price reaches the controller without a configured price."""
    _use_unpriced_grid_sell(temp_copy_of_example, cost_model)
    model = H2IntegrateModel(temp_copy_of_example / "solar_battery_arbitrage.yaml")
    model.setup()
    model.prob.final_setup()

    source = model.prob.model.get_source("system_level_controller.grid_sell_sell_price")
    assert source == model.prob.model.get_source("grid_sell.electricity_sell_price")


@pytest.mark.unit
@pytest.mark.parametrize("example_folder,resource_example_folder", [(EXAMPLE, None)])
def test_lp_arbitrage_sell_price_missing_warns(temp_copy_of_example):
    """An export technology without any sell price leaves the controller default in place."""
    _use_unpriced_grid_sell(temp_copy_of_example)
    model = H2IntegrateModel(temp_copy_of_example / "solar_battery_arbitrage.yaml")
    with pytest.warns(UserWarning, match="'grid_sell_sell_price' is not connected"):
        model.setup()


@pytest.mark.unit
@pytest.mark.parametrize("example_folder,resource_example_folder", [(EXAMPLE, None)])
def test_lp_arbitrage_prices_from_custom_grid(subtests, temp_copy_of_example):
    """Computed buy and sell prices of any supported shape reach the controller."""
    _use_unpriced_grid_sell(temp_copy_of_example, "GridPriceOutputs")
    config_path = temp_copy_of_example / "tech_config.yaml"
    text = config_path.read_text()
    text = text.replace(
        "        electricity_buy_price: 0.03  # $/kWh; overridden with an LMP series in the run "
        "script\n",
        "",
    )
    text = text.replace(
        "  grid_buy:\n    performance_model:\n      model: GridPerformanceModel\n"
        "    cost_model:\n      model: GridCostModel\n",
        "  grid_buy:\n    performance_model:\n      model: GridPerformanceModel\n"
        "    cost_model:\n      model: GridPriceOutputs\n      model_location: custom_grid.py\n",
    )
    config_path.write_text(text)

    model = H2IntegrateModel(temp_copy_of_example / "solar_battery_arbitrage.yaml")
    model.setup()
    model.prob.final_setup()
    om_model = model.prob.model

    with subtests.test("Buy price is connected"):
        assert om_model.get_source("system_level_controller.grid_buy_buy_price") == (
            om_model.get_source("grid_buy.electricity_buy_price")
        )

    with subtests.test("Scalar sell price is connected"):
        assert om_model.get_source("system_level_controller.grid_sell_sell_price") == (
            om_model.get_source("grid_sell.electricity_sell_price")
        )


@requires_glpk
@pytest.mark.unit
@pytest.mark.parametrize("example_folder,resource_example_folder", [(EXAMPLE, None)])
def test_lp_arbitrage_buys_low_sells_high(subtests, temp_copy_of_example):
    """The LP charges during the cheap half of the window and discharges in the expensive half."""
    controller = _make_controller(temp_copy_of_example)
    lp = controller._build_lp_model()

    window_len = controller.window_len
    half = window_len // 2
    price = np.concatenate([np.full(half, 0.01), np.full(window_len - half, 0.10)])
    _fill_window(lp, controller, sell_price=price, marginal_cost=price)

    controller._solve_window(lp, 0)

    charge = _series(lp.charge, "battery", window_len)
    discharge = _series(lp.discharge, "battery", window_len)
    soc = _series(lp.soc, "battery", window_len)

    with subtests.test("Charging happens only while the price is low"):
        assert charge[:half].sum() > 0.0
        assert charge[half:].sum() == pytest.approx(0.0, abs=1e-6)

    with subtests.test("Discharging happens only while the price is high"):
        assert discharge[half:].sum() > 0.0
        assert discharge[:half].sum() == pytest.approx(0.0, abs=1e-6)

    with subtests.test("Never charges and discharges simultaneously"):
        assert np.all((charge < 1e-6) | (discharge < 1e-6))

    with subtests.test("Rate limits respected"):
        params = controller.storage_params["battery"]
        assert charge.max() <= params["max_charge_rate"] + 1e-6
        assert discharge.max() <= params["max_discharge_rate"] + 1e-6

    with subtests.test("State of charge stays within its bounds"):
        params = controller.storage_params["battery"]
        assert soc.min() >= params["min_soc_fraction"] * params["capacity"] - 1e-6
        assert soc.max() <= params["max_soc_fraction"] * params["capacity"] + 1e-6


@requires_glpk
@pytest.mark.unit
@pytest.mark.parametrize("example_folder,resource_example_folder", [(EXAMPLE, None)])
def test_lp_arbitrage_does_not_resell_grid_imports(subtests, temp_copy_of_example):
    """A buy/sell price spread must not create instantaneous grid arbitrage."""
    controller = _make_controller(temp_copy_of_example)
    lp = controller._build_lp_model()
    _fill_window(lp, controller, sell_price=0.10, marginal_cost=0.01)

    controller._solve_window(lp, 0)

    imported = _series(lp.dispatch, "grid_buy", controller.window_len)
    exported = np.array([pyo.value(lp.export[t]) for t in lp.T])
    charge = _series(lp.charge, "battery", controller.window_len)
    discharge = _series(lp.discharge, "battery", controller.window_len)
    with subtests.test("Imports go to storage, not directly to exports"):
        assert np.all(imported <= charge + 1e-6)
        assert np.all(exported <= discharge + 1e-6)
    with subtests.test("Storage cannot manufacture export while charging"):
        assert np.all((charge < 1e-6) | (discharge < 1e-6))


@requires_glpk
@pytest.mark.unit
@pytest.mark.parametrize("example_folder,resource_example_folder", [(EXAMPLE, None)])
def test_lp_arbitrage_solar_export_during_grid_charge(subtests, temp_copy_of_example):
    """A local source can export while grid imports charge the battery."""
    controller = _make_controller(temp_copy_of_example)
    lp = controller._build_lp_model()
    half = controller.window_len // 2
    prices = np.r_[np.full(half, 0.05), np.full(controller.window_len - half, 0.20)]
    solar = np.r_[np.full(half, 20_000.0), np.zeros(controller.window_len - half)]
    _fill_window(lp, controller, sell_price=prices, must_run=solar, marginal_cost=0.01)

    controller._solve_window(lp, 0)

    imported = _series(lp.dispatch, "grid_buy", controller.window_len)
    charge = _series(lp.charge, "battery", controller.window_len)
    exported = np.array([pyo.value(lp.export[t]) for t in lp.T])
    with subtests.test("Solar exports while grid imports charge storage"):
        importing = imported > 1e-6
        assert np.any(importing[:half] & (exported[:half] > 1e-6))
        assert np.all(exported[importing] <= solar[importing] + 1e-6)
        assert np.all(imported <= charge + 1e-6)


@requires_glpk
@pytest.mark.unit
@pytest.mark.parametrize("example_folder,resource_example_folder", [(EXAMPLE, None)])
def test_lp_arbitrage_grid_dispatch_uses_sell_price(subtests, temp_copy_of_example):
    """The actual buy price does not alter dispatch, including on cached solves."""
    controller = _make_controller(temp_copy_of_example)
    controller.n_timesteps = controller.window_len = controller.commit_len = 4
    price = np.array([0.01, 0.01, 0.20, 0.20])
    inputs = {
        controller.demand_input_name: np.zeros(4),
        "grid_sell_sell_price": price,
        "grid_buy_buy_price": np.full(4, 1.0),
        "added_solar_electricity_out": np.zeros(4),
        "added_solar_rated_electricity_production": np.array([50_000.0]),
        "existing_load_demand_surplus": np.zeros(4),
        "grid_buy_rated_electricity_production": np.array([50_000.0]),
        "battery_storage_capacity": np.array([200_000.0]),
        "battery_max_charge_rate": np.array([50_000.0]),
    }
    outputs = {name: np.zeros(4) for name in controller._var_rel_names["output"]}
    controller.compute(inputs, outputs)
    first_schedule = outputs["grid_buy_electricity_set_point"].copy()

    with subtests.test("Grid imports use the sell price"):
        assert first_schedule.sum() > 0.0
        marginal_cost = [
            pyo.value(controller._lp_model.marginal_cost["grid_buy", t]) for t in range(4)
        ]
        assert np.allclose(marginal_cost, price)

    inputs["grid_buy_buy_price"] = np.full(4, -1.0)
    controller.compute(inputs, outputs)
    with subtests.test("A changed buy price leaves the dispatch unchanged"):
        assert np.allclose(outputs["grid_buy_electricity_set_point"], first_schedule)


@requires_glpk
@pytest.mark.unit
@pytest.mark.parametrize("example_folder,resource_example_folder", [(EXAMPLE, None)])
def test_lp_arbitrage_imports_can_serve_demand(temp_copy_of_example):
    """Grid purchases are allowed to supply demand when there is no storage charging."""
    controller = _make_controller(temp_copy_of_example)
    lp = controller._build_lp_model()
    _fill_window(lp, controller, sell_price=0.10, demand=10_000.0, marginal_cost=0.01)
    for s in lp.S:
        lp.max_charge[s] = 0.0
        lp.max_discharge[s] = 0.0

    controller._solve_window(lp, 0)

    imported = _series(lp.dispatch, "grid_buy", controller.window_len)
    exported = np.array([pyo.value(lp.export[t]) for t in lp.T])
    assert np.allclose(imported, 10_000.0)
    assert np.allclose(exported, 0.0)


@requires_glpk
@pytest.mark.unit
@pytest.mark.parametrize("example_folder,resource_example_folder", [(EXAMPLE, None)])
def test_lp_arbitrage_multiple_grid_imports(temp_copy_of_example):
    """The anti-resale constraint applies to the sum of all grid connections."""
    controller = _make_controller(temp_copy_of_example)
    controller.lp_dispatchable_techs.append("second_grid")
    controller.lp_grid_import_techs.append("second_grid")
    lp = controller._build_lp_model()
    _fill_window(lp, controller, sell_price=0.10, marginal_cost=0.01)

    controller._solve_window(lp, 0)

    imported = sum(_series(lp.dispatch, grid, controller.window_len) for grid in lp.G)
    charge = _series(lp.charge, "battery", controller.window_len)
    assert imported.sum() > 0.0
    assert np.all(imported <= charge + 1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("example_folder,resource_example_folder", [(EXAMPLE, None)])
@pytest.mark.parametrize("grid_import_techs", ["[not_a_tech]", "[grid_buy, grid_buy]", "grid_buy"])
def test_lp_arbitrage_rejects_invalid_grid_import_techs(temp_copy_of_example, grid_import_techs):
    """Explicit grid import names must uniquely identify controlled dispatchables."""
    config_path = temp_copy_of_example / "plant_config.yaml"
    text = config_path.read_text()
    config_path.write_text(
        text.replace(
            "    n_control_window_hours: 24\n",
            f"    n_control_window_hours: 24\n    grid_import_techs: {grid_import_techs}\n",
        )
    )
    with pytest.raises(ValueError, match="grid_import_techs"):
        _make_controller(temp_copy_of_example)


@pytest.mark.unit
@pytest.mark.parametrize("example_folder,resource_example_folder", [(EXAMPLE, None)])
def test_lp_arbitrage_custom_grid_import(temp_copy_of_example):
    """Explicit grid import names support custom performance models."""
    (temp_copy_of_example / "custom_grid.py").write_text(CUSTOM_GRID_MODELS)
    tech_path = temp_copy_of_example / "tech_config.yaml"
    tech_path.write_text(
        tech_path.read_text().replace(
            "  grid_buy:\n    performance_model:\n      model: GridPerformanceModel\n",
            "  grid_buy:\n    performance_model:\n"
            "      model: CustomGridPerformanceModel\n      model_location: custom_grid.py\n",
        )
    )
    plant_path = temp_copy_of_example / "plant_config.yaml"
    plant_path.write_text(
        plant_path.read_text().replace(
            "    n_control_window_hours: 24\n",
            "    n_control_window_hours: 24\n    grid_import_techs: [grid_buy]\n",
        )
    )

    controller = _make_controller(temp_copy_of_example)
    assert controller.lp_grid_import_techs == ["grid_buy"]


@requires_glpk
@pytest.mark.unit
@pytest.mark.parametrize("example_folder,resource_example_folder", [(EXAMPLE, None)])
def test_lp_arbitrage_respects_export_limit(subtests, temp_copy_of_example):
    """Production beyond the interconnection limit is curtailed rather than exported."""
    controller = _make_controller(temp_copy_of_example)
    lp = controller._build_lp_model()

    window_len = controller.window_len
    oversupply = 5.0 * controller.export_limit
    _fill_window(lp, controller, sell_price=0.05, must_run=oversupply)

    controller._solve_window(lp, 0)

    export = np.array([pyo.value(lp.export[t]) for t in range(window_len)])
    curtail = np.array([pyo.value(lp.curtail[t]) for t in range(window_len)])

    with subtests.test("Export never exceeds the interconnection size"):
        assert export.max() <= controller.export_limit + 1e-6

    with subtests.test("Export is saturated"):
        assert export.min() == pytest.approx(controller.export_limit, rel=1e-6)

    with subtests.test("Surplus is curtailed"):
        assert curtail.min() > 0.0


@requires_glpk
@pytest.mark.unit
@pytest.mark.parametrize("example_folder,resource_example_folder", [(EXAMPLE, None)])
def test_lp_arbitrage_charge_limited_by_availability(temp_copy_of_example):
    """Storage cannot charge from commodity that is not physically on the bus.

    This mirrors the ``charge_available`` clip inside the storage performance
    model. With no must-run production and no import capacity there is nothing
    to charge from, so the commanded charge must be zero even though the price
    spread would otherwise make charging attractive.
    """
    controller = _make_controller(temp_copy_of_example)
    lp = controller._build_lp_model()

    window_len = controller.window_len
    half = window_len // 2
    price = np.concatenate([np.full(half, 0.01), np.full(window_len - half, 0.10)])
    _fill_window(lp, controller, sell_price=price, must_run=0.0, rated=0.0)

    controller._solve_window(lp, 0)

    charge = _series(lp.charge, "battery", window_len)
    assert charge.sum() == pytest.approx(0.0, abs=1e-6)


@requires_glpk
@pytest.mark.unit
@pytest.mark.parametrize("example_folder,resource_example_folder", [(EXAMPLE, None)])
def test_lp_arbitrage_time_varying_export_limit(subtests, temp_copy_of_example):
    """A per-timestep export ceiling is honored, and is re-read on every solve.

    The window model is built once and re-solved with updated mutable
    parameters, so this also guards the assumption that Pyomo re-evaluates a
    mutable ``Param`` used in a variable bound rather than freezing it at
    construction time.
    """
    controller = _make_controller(temp_copy_of_example)
    lp = controller._build_lp_model()

    window_len = controller.window_len
    half = window_len // 2
    oversupply = 5.0 * controller.export_limit
    ceiling = np.concatenate([np.zeros(half), np.full(window_len - half, 20_000.0)])
    _fill_window(lp, controller, sell_price=0.05, must_run=oversupply, export_limit=ceiling)

    controller._solve_window(lp, 0)
    export = np.array([pyo.value(lp.export[t]) for t in range(window_len)])

    with subtests.test("Export tracks the time-varying ceiling"):
        assert np.allclose(export, ceiling, atol=1e-6)

    with subtests.test("Blocked production is curtailed"):
        curtail = np.array([pyo.value(lp.curtail[t]) for t in range(window_len)])
        assert curtail[:half].min() > 0.0

    # Re-solve the same model object with a different ceiling.
    relaxed = np.full(window_len, 60_000.0)
    _fill_window(lp, controller, sell_price=0.05, must_run=oversupply, export_limit=relaxed)
    controller._solve_window(lp, 0)
    export = np.array([pyo.value(lp.export[t]) for t in range(window_len)])

    with subtests.test("A rebuilt ceiling takes effect without rebuilding the model"):
        assert np.allclose(export, relaxed, atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("example_folder,resource_example_folder", [(EXAMPLE, None)])
def test_lp_arbitrage_headroom_components(subtests, temp_copy_of_example):
    """Headroom keys add controller inputs and resolve to the named component."""
    controller = _make_controller(temp_copy_of_example)

    with subtests.test("The example uses spill but has no unmet-demand export ceiling"):
        assert "existing_load_demand_unmet_demand" not in controller._var_rel_names["input"]
        assert "existing_load_demand_surplus" in controller._var_rel_names["input"]
        assert controller.export_limit_component is None
        assert controller.surplus_source_component == "existing_load_demand"

    with subtests.test("The uncontrolled existing plant is not dispatched"):
        assert "existing_solar" not in controller.lp_must_run_techs
        assert "existing_solar" not in controller.lp_dispatchable_techs
        assert "existing_wind" not in controller.lp_must_run_techs
        assert "existing_wind" not in controller.lp_dispatchable_techs

    config_path = temp_copy_of_example / "plant_config.yaml"
    text = config_path.read_text()
    config_path.write_text(
        text.replace(
            "    surplus_source_component: existing_load_demand\n",
            "    export_limit_component: existing_load_demand\n"
            "    surplus_source_component: existing_load_demand\n",
        )
    )
    controller = _make_controller(temp_copy_of_example)
    with subtests.test("An explicitly configured unmet-demand ceiling is still supported"):
        assert "existing_load_demand_unmet_demand" in controller._var_rel_names["input"]
        assert controller.export_limit_component == "existing_load_demand"


@requires_glpk
@pytest.mark.unit
@pytest.mark.parametrize("example_folder,resource_example_folder", [(EXAMPLE, None)])
def test_lp_arbitrage_example_exports_existing_spill(temp_copy_of_example):
    """The example can sell existing spill without an unmet-demand ceiling."""
    controller = _make_controller(temp_copy_of_example)
    assert controller.export_limit_component is None
    controller.n_timesteps = controller.window_len = controller.commit_len = 4
    inputs = {
        controller.demand_input_name: np.zeros(4),
        "grid_sell_sell_price": np.full(4, 0.05),
        "grid_buy_buy_price": np.full(4, 0.01),
        "added_solar_electricity_out": np.zeros(4),
        "added_solar_rated_electricity_production": np.array([50_000.0]),
        "existing_load_demand_surplus": np.full(4, 20_000.0),
        "grid_buy_rated_electricity_production": np.array([50_000.0]),
        "battery_storage_capacity": np.array([200_000.0]),
        "battery_max_charge_rate": np.array([0.0]),
    }
    outputs = {name: np.zeros(4) for name in controller._var_rel_names["output"]}
    controller.compute(inputs, outputs)

    exported = np.array([pyo.value(controller._lp_model.export[t]) for t in range(4)])
    assert np.allclose(exported, 20_000.0, atol=1e-6)
    assert np.allclose(outputs["grid_buy_electricity_set_point"], 0.0, atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("example_folder,resource_example_folder", [(EXAMPLE, None)])
def test_lp_arbitrage_rejects_unknown_headroom_component(temp_copy_of_example):
    """A headroom key naming a missing technology is reported as a config error."""
    config_path = temp_copy_of_example / "plant_config.yaml"
    text = config_path.read_text()
    config_path.write_text(
        text.replace(
            "    surplus_source_component: existing_load_demand\n",
            "    export_limit_component: not_a_tech\n"
            "    surplus_source_component: existing_load_demand\n",
        )
    )

    # The controller is wired up during construction, so this fails before setup().
    with pytest.raises(ValueError, match="not a configured technology"):
        H2IntegrateModel(temp_copy_of_example / "solar_battery_arbitrage.yaml")


@requires_glpk
@pytest.mark.unit
@pytest.mark.parametrize("example_folder,resource_example_folder", [(EXAMPLE, None)])
def test_lp_arbitrage_requires_export_component(temp_copy_of_example):
    """A missing ``export_component`` is reported as a configuration error."""
    config_path = temp_copy_of_example / "plant_config.yaml"
    text = config_path.read_text()
    config_path.write_text(text.replace("  export_component: grid_sell\n", ""))

    model = H2IntegrateModel(temp_copy_of_example / "solar_battery_arbitrage.yaml")
    with pytest.raises(ValueError, match="requires an export technology"):
        model.setup()


@pytest.mark.unit
@pytest.mark.parametrize("example_folder,resource_example_folder", [(EXAMPLE, None)])
def test_lp_arbitrage_commit_window(subtests, temp_copy_of_example):
    """``n_commit_window_hours`` decouples the committed block from the lookahead."""
    config_path = temp_copy_of_example / "plant_config.yaml"
    text = config_path.read_text()

    with subtests.test("Commit length defaults to a quarter of the window"):
        controller = _make_controller(temp_copy_of_example)
        assert controller.window_len == 24
        assert controller.commit_len == 6

    with subtests.test("A shorter commit length is honored"):
        config_path.write_text(
            text.replace(
                "    n_control_window_hours: 24\n",
                "    n_control_window_hours: 24\n    n_commit_window_hours: 4\n",
            )
        )
        controller = _make_controller(temp_copy_of_example)
        assert controller.window_len == 24
        assert controller.commit_len == 4

    with subtests.test("Committing more than the lookahead is rejected"):
        config_path.write_text(
            text.replace(
                "    n_control_window_hours: 24\n",
                "    n_control_window_hours: 24\n    n_commit_window_hours: 48\n",
            )
        )
        model = H2IntegrateModel(temp_copy_of_example / "solar_battery_arbitrage.yaml")
        with pytest.raises(ValueError, match="cannot commit more timesteps"):
            model.setup()


@requires_glpk
@pytest.mark.unit
@pytest.mark.parametrize("example_folder,resource_example_folder", [(EXAMPLE, None)])
def test_lp_arbitrage_storage_sizing_from_inputs(subtests, temp_copy_of_example):
    """Storage sizing is taken from the storage model's inputs so sweeps stay consistent."""
    model = H2IntegrateModel(temp_copy_of_example / "solar_battery_arbitrage.yaml")
    model.setup()
    model.prob.final_setup()
    controller = model.prob.model.plant.system_level_controller

    with subtests.test("Sizing inputs are declared and connected"):
        assert "battery_storage_capacity" in controller._var_rel_names["input"]
        assert "battery_max_charge_rate" in controller._var_rel_names["input"]
        # Charge and discharge rates are equal here, so the storage model
        # declares no discharge-rate input and neither does the controller.
        assert "battery_max_discharge_rate" not in controller._var_rel_names["input"]

    with subtests.test("Resizing the battery moves the controller with it"):
        model.prob.set_val("plant.battery.storage_capacity", 400_000.0, units="kW*h")
        model.prob.set_val("plant.battery.max_charge_rate", 10_000.0, units="kW")
        model.prob.final_setup()
        capacity = model.prob.get_val(
            "plant.system_level_controller.battery_storage_capacity", units="kW*h"
        ).item()
        charge_rate = model.prob.get_val(
            "plant.system_level_controller.battery_max_charge_rate", units="kW"
        ).item()
        assert capacity == pytest.approx(400_000.0)
        assert charge_rate == pytest.approx(10_000.0)

    with subtests.test("The linear program is bounded by the swept sizing"):
        lp = controller._build_lp_model()
        lp.max_charge["battery"] = charge_rate
        lp.soc_max["battery"] = controller.storage_params["battery"]["max_soc_fraction"] * capacity
        # Cheap first half, expensive second half, so charging to the rate limit pays.
        hours = np.arange(controller.window_len)
        _fill_window(
            lp,
            controller,
            sell_price=np.where(hours < controller.window_len // 2, 0.01, 0.10),
            must_run=20_000.0,
        )
        controller._solve_window(lp, 0)
        charge = _series(lp.charge, "battery", controller.window_len)
        assert charge.max() == pytest.approx(10_000.0)

    with subtests.test("The storage mode constraints honor a later rate change"):
        lp.max_charge["battery"] = 3_000.0
        controller._solve_window(lp, 0)
        charge = _series(lp.charge, "battery", controller.window_len)
        assert charge.max() == pytest.approx(3_000.0)


@requires_glpk
@pytest.mark.integration
@pytest.mark.parametrize("example_folder,resource_example_folder", [(EXAMPLE, None)])
def test_lp_arbitrage_example(subtests, temp_copy_of_example):
    """The full example follows the optimized schedule within solver tolerance."""
    model = H2IntegrateModel(temp_copy_of_example / "solar_battery_arbitrage.yaml")
    model.setup()

    # A deterministic diurnal price: cheap overnight, expensive in the evening.
    hour_of_day = np.arange(8760) % 24
    price = 0.03 + 0.025 * np.sin((hour_of_day - 9) * np.pi / 12)
    model.prob.set_val("grid_sell.electricity_sell_price", price, units="USD/(kW*h)")
    model.prob.set_val("grid_buy.electricity_buy_price", price - 0.020, units="USD/(kW*h)")

    # A single evaluation, not the example's battery sizing sweep. This
    # test is about whether the dispatch the controller plans is realizable.
    model.prob.run_model()

    get = model.prob.get_val
    commanded = get("system_level_controller.battery_electricity_set_point", units="kW")
    actual = get("battery.electricity_out", units="kW")
    charge = -get("battery.storage_electricity_charge", units="kW")
    discharge = get("battery.storage_electricity_discharge", units="kW")
    soc = get("battery.SOC", units="percent")
    imported = get("grid_buy.electricity_out", units="kW")
    exported = get("grid_sell.electricity_sold", units="kW")
    curtailed = get("grid_sell.electricity_excess", units="kW")
    spill = get("existing_load_demand.unused_electricity_out", units="kW")
    added_solar = get("added_solar.electricity_out", units="kW")
    existing_solar = get("existing_solar.electricity_out", units="kW")
    existing_wind = get("existing_wind.electricity_out", units="kW")
    existing_combined = get("existing_combiner.electricity_out", units="kW")
    headroom = get("existing_load_demand.unmet_electricity_demand_out", units="kW")
    unmet = get("electrical_load_demand.unmet_electricity_demand_out", units="kW")

    with subtests.test("Storage follows the commanded schedule"):
        # Nothing is clipped by the storage model's charge-availability limit.
        max_error = np.max(np.abs(commanded - actual))
        assert max_error <= 0.01, f"Maximum storage dispatch error: {max_error} kW"

    with subtests.test("State of charge stays within its bounds"):
        assert soc.min() >= 10.0 - 1e-6
        assert soc.max() <= 100.0 + 1e-6

    with subtests.test("Export respects the interconnection limit"):
        assert exported.max() <= 100_000.0 + 1e-6

    with subtests.test("Existing spill can be exported despite zero unmet demand"):
        assert np.any((spill > 1.0) & (exported > headroom + 1.0))

    with subtests.test("No unmet demand"):
        assert unmet.sum() == pytest.approx(0.0, abs=1e-6)

    with subtests.test("Commodity balance closes"):
        supply = imported + spill + added_solar + discharge - charge
        assert np.allclose(supply, exported + curtailed, rtol=1e-6, atol=1e-6)

    with subtests.test("Existing wind and solar feed the existing load"):
        assert existing_wind.sum() > 0.0
        assert existing_solar.sum() > 0.0
        assert np.allclose(existing_combined, existing_solar + existing_wind)

    with subtests.test("The solar addition produces electricity"):
        assert added_solar.sum() > 0.0

    with subtests.test("Grid imports charge storage rather than being resold"):
        max_import_excess = np.max(imported - charge)
        max_export_excess = np.max(exported - added_solar - spill - discharge)
        assert max_import_excess <= 0.01, f"Maximum import excess: {max_import_excess} kW"
        assert max_export_excess <= 0.01, f"Maximum export excess: {max_export_excess} kW"
        assert np.all((charge < 1e-6) | (discharge < 1e-6))

    with subtests.test("Round-trip efficiency is applied"):
        # Not exactly the round-trip efficiency because the battery does not end
        # the year at its starting state of charge.
        assert discharge.sum() / charge.sum() == pytest.approx(0.88, rel=5e-3)

    with subtests.test("Charges cheaper than it discharges"):
        assert np.average(price, weights=charge) < np.average(price, weights=discharge)
