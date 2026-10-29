import pytest
import openmdao.api as om
from openmdao.core.driver import Driver

from h2integrate.core.utilities import component_totals_enabled
from h2integrate.core.pose_optimization import PoseOptimization


class Quadratic(om.ExplicitComponent):
    def setup(self):
        self.add_input("x", 2.0)
        self.add_output("y", 0.0)
        self.declare_partials("y", "x")

    def compute(self, inputs, outputs):
        outputs["y"] = inputs["x"] ** 2

    def compute_partials(self, inputs, partials):
        partials["y", "x"] = 2 * inputs["x"]


def _problem(options):
    config = {
        "general": {"folder_output": "."},
        "driver": {
            "optimization": {"flag": True, "solver": "SLSQP", "debug_print": False} | options
        },
    }
    prob = om.Problem(reports=False)
    prob.model.add_subsystem("quadratic", Quadratic(), promotes=["*"])
    prob.model.add_design_var("x")
    prob.model.add_objective("y")
    PoseOptimization(config).set_driver(prob)
    prob.setup()
    prob.run_model()
    return prob


@pytest.mark.unit
@pytest.mark.parametrize(
    "options,expected",
    [({"step_size": 0.1}, 4.1), ({"total_derivatives": "components"}, 4.0)],
)
def test_total_derivative_mode(options, expected):
    prob = _problem(options)
    total = prob.compute_totals(of=["y"], wrt=["x"], return_format="array")
    assert total[0, 0] == pytest.approx(expected)


@pytest.mark.unit
def test_unknown_total_derivative_mode():
    with pytest.raises(ValueError, match="Unknown total_derivatives"):
        _problem({"total_derivatives": "incorrect"})


@pytest.mark.unit
def test_coloring_is_opt_in(monkeypatch):
    calls = []

    def record_coloring(driver):
        calls.append(driver)

    monkeypatch.setattr(om.ScipyOptimizeDriver, "declare_coloring", record_coloring)
    _problem({"total_derivatives": "components"})
    assert calls == []
    _problem({"total_derivatives": "components", "coloring": True})
    assert len(calls) == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    "driver_config,expected",
    [
        (None, False),
        ({}, False),
        ({"driver": {"optimization": {"flag": False, "total_derivatives": "components"}}}, False),
        ({"driver": {"optimization": {"flag": True, "total_derivatives": "components"}}}, True),
    ],
)
def test_component_totals_option(driver_config, expected):
    from openmdao.utils.options_dictionary import OptionsDictionary

    options = OptionsDictionary()
    options.declare("driver_config", types=dict)
    if driver_config is not None:
        options["driver_config"] = driver_config
    assert component_totals_enabled(options) is expected


@pytest.mark.unit
def test_ipopt_driver_configuration(monkeypatch):
    class MockPyOptSparseDriver(Driver):
        def __init__(self, gradient_method):
            super().__init__()
            self.gradient_method = gradient_method
            self.opt_settings = {}
            self.options.declare("optimizer", default="")

    monkeypatch.setattr(om, "pyOptSparseDriver", MockPyOptSparseDriver)
    config = {
        "general": {"folder_output": "."},
        "driver": {
            "optimization": {
                "flag": True,
                "solver": "IPOPT",
                "debug_print": False,
                "step_size": 1e-6,
                "tol": 1e-5,
                "max_iter": 42,
            }
        },
    }
    prob = om.Problem(reports=False)
    PoseOptimization(config).set_driver(prob)
    assert prob.driver.options["optimizer"] == "IPOPT"
    assert prob.driver.gradient_method == "openmdao"
    assert prob.driver.opt_settings == {"tol": 1e-5, "max_iter": 42}
