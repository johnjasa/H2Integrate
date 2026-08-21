import numpy as np
import pytest
import openmdao.api as om
from pytest import fixture

from h2integrate.converters.hydrogen.pem_electrolyzer import ECOElectrolyzerPerformanceModel
from h2integrate.converters.hydrogen.pem_model.run_PEM_main import run_PEM_clusters


@fixture
def plant_config():
    return {
        "plant": {
            "plant_life": 30,
            "simulation": {
                "n_timesteps": 8760,
                "dt": 3600,
            },
        },
    }


def make_tech_config(continuous_clusters, n_clusters):
    return {
        "model_inputs": {
            "performance_parameters": {
                "n_clusters": n_clusters,
                "continuous_clusters": continuous_clusters,
                "location": "onshore",
                "cluster_rating_MW": 10,
                "eol_eff_percent_loss": 10.0,
                "uptime_hours_until_eol": 8000,
                "include_degradation_penalty": True,
                "turndown_ratio": 0.1,
                "electrolyzer_capex": 10.0,
                "use_fatigue_deg": False,
            }
        }
    }


def run_case(plant_config, continuous_clusters, n_clusters):
    prob = om.Problem()
    prob.model.add_subsystem(
        "comp",
        ECOElectrolyzerPerformanceModel(
            plant_config=plant_config,
            tech_config=make_tech_config(continuous_clusters, n_clusters),
            driver_config={},
        ),
        promotes=["*"],
    )
    prob.setup()
    prob.set_val("comp.electricity_in", np.ones(8760) * 32.0, units="MW")
    prob.run_model()
    return {
        "size_mw": float(prob.get_val("comp.electrolyzer_size_mw", units="MW")[0]),
        "total_hydrogen": float(prob.get_val("comp.total_hydrogen_produced", units="kg")[0]),
        "hydrogen_out": np.array(prob.get_val("comp.hydrogen_out", units="kg/h")),
    }


@pytest.mark.unit
def test_cluster_weights_are_whole_for_integer_counts():
    """An integer cluster count produces the same unweighted model as before."""
    pem = run_PEM_clusters(
        np.ones(8760) * 32000.0, 40.0, 4, 1295.0, 30, {"turndown_ratio": 0.1}, verbose=False
    )

    assert pem.n_clusters_run == 4
    assert pem.cluster_weights == pytest.approx(np.ones(4))


@pytest.mark.unit
def test_cluster_weights_split_the_marginal_cluster(subtests):
    """A fractional cluster count adds one partial cluster carrying the remainder."""
    pem = run_PEM_clusters(
        np.ones(8760) * 32000.0, 45.0, 4.5, 1295.0, 30, {"turndown_ratio": 0.1}, verbose=False
    )

    with subtests.test("one extra cluster is simulated"):
        assert pem.n_clusters_run == 5

    with subtests.test("weights sum to the requested cluster count"):
        assert np.sum(pem.cluster_weights) == pytest.approx(4.5)

    with subtests.test("marginal cluster carries the fractional weight"):
        assert pem.cluster_weights == pytest.approx([1.0, 1.0, 1.0, 1.0, 0.5])


@pytest.mark.unit
def test_continuous_clusters_matches_default_at_integer_counts(plant_config, subtests):
    """Turning the relaxation on does not change results for a whole number of clusters."""
    default = run_case(plant_config, continuous_clusters=False, n_clusters=4.0)
    continuous = run_case(plant_config, continuous_clusters=True, n_clusters=4.0)

    with subtests.test("electrolyzer size"):
        assert continuous["size_mw"] == pytest.approx(default["size_mw"])

    with subtests.test("total hydrogen produced"):
        assert continuous["total_hydrogen"] == pytest.approx(default["total_hydrogen"], rel=1e-12)

    with subtests.test("hydrogen time series"):
        assert continuous["hydrogen_out"] == pytest.approx(default["hydrogen_out"], rel=1e-12)


@pytest.mark.unit
def test_continuous_clusters_is_smooth_across_a_cluster_boundary(plant_config, subtests):
    """A fractional cluster count sizes and produces between the neighboring integers."""
    lower = run_case(plant_config, continuous_clusters=True, n_clusters=4.0)
    middle = run_case(plant_config, continuous_clusters=True, n_clusters=4.5)
    upper = run_case(plant_config, continuous_clusters=True, n_clusters=5.0)

    with subtests.test("size is continuous"):
        assert middle["size_mw"] == pytest.approx(45.0)

    with subtests.test("default sizing would round up instead"):
        rounded = run_case(plant_config, continuous_clusters=False, n_clusters=4.5)
        assert rounded["size_mw"] == pytest.approx(50.0)

    with subtests.test("production is strictly between the neighboring integer designs"):
        assert lower["total_hydrogen"] < middle["total_hydrogen"] < upper["total_hydrogen"]
