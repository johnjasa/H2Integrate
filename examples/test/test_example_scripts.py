"""Run every example script the same way a user would from the command line.

The value checks for each example live in ``test_all_examples.py``. These tests only confirm that
each ``run_*.py`` script, including its plotting and post-processing code, finishes without error.
"""

import os
import sys
import shutil
import subprocess
import importlib.util

import numpy as np
import pytest

from h2integrate import EXAMPLE_DIR


# Python modules under examples/ that are imported by a run script or referenced from a config
HELPER_MODULES = {
    "03_methanol/co2_hydrogenation_doc/plot_co2h_methanol.py",
    "06_custom_tech/user_defined_model/paper_mill.py",
    "08_wind_electrolyzer/user_finance_model/simple_lco.py",
    "39_differentiable_dispatch/sizing_problem.py",
}

# Scripts that need an optional dependency, mapped to the module that must be importable
OPTIONAL_DEPENDENCIES = {
    "21_iron_examples/iron_mapping/run_iron.py": "geopandas",
    "29_wind_ard/run_windard_pv_battery.py": "ard",
    "39_differentiable_dispatch/run_differentiable_dispatch.py": "idaes",
    "39_differentiable_dispatch/run_short_dispatch.py": "idaes",
}

SCRIPT_TIMEOUT_SECONDS = 3600


def _relative(path):
    return path.relative_to(EXAMPLE_DIR).as_posix()


def _example_python_files(pattern):
    return {
        _relative(path)
        for path in EXAMPLE_DIR.rglob(pattern)
        if path.relative_to(EXAMPLE_DIR).parts[0] != "test" and "__pycache__" not in path.parts
    }


EXAMPLE_SCRIPTS = sorted(_example_python_files("run_*.py"))


def _cached_dol_available():
    if importlib.util.find_spec("idaes") is None:
        return False
    from h2integrate.control.differentiable import kkt_tools_available
    from h2integrate.control.differentiable.cached_ipopt import cached_ipopt_available

    return kkt_tools_available() and cached_ipopt_available()


def _script_param(script):
    module = OPTIONAL_DEPENDENCIES.get(script)
    marks = []
    if module is not None:
        marks.append(
            pytest.mark.skipif(
                importlib.util.find_spec(module) is None, reason=f"{module} is not installed"
            )
        )
    if script in {
        "39_differentiable_dispatch/run_differentiable_dispatch.py",
        "39_differentiable_dispatch/run_short_dispatch.py",
    }:
        marks.append(
            pytest.mark.skipif(
                not _cached_dol_available(),
                reason="Ipopt, its shared library, and PyNumero ASL are not available",
            )
        )
    return pytest.param(script, marks=marks, id=script)


@pytest.mark.unit
def test_example_scripts_are_discoverable():
    """Every example Python file is either a ``run_*.py`` script or a known helper module."""
    python_files = _example_python_files("*.py")
    unexpected = sorted(python_files - set(EXAMPLE_SCRIPTS) - HELPER_MODULES)
    assert not unexpected, (
        "Name example scripts `run_*.py` so they are tested, or add helper modules to "
        f"HELPER_MODULES in {__file__}: {unexpected}"
    )
    missing_helpers = sorted(HELPER_MODULES - python_files)
    assert not missing_helpers, f"Remove deleted files from HELPER_MODULES: {missing_helpers}"


@pytest.mark.unit
def test_short_dispatch_profile_validation(tmp_path):
    """The one-week diagnostic accepts exactly one finite, nonnegative hourly column."""
    path = EXAMPLE_DIR / "39_differentiable_dispatch" / "run_short_dispatch.py"
    spec = importlib.util.spec_from_file_location("short_dispatch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    profile = np.arange(168, dtype=float)
    csv = tmp_path / "generation.csv"
    np.savetxt(csv, profile, delimiter=",")
    np.testing.assert_array_equal(module.load_generation_profile(csv), profile)
    np.savetxt(csv, np.ones((168, 2)), delimiter=",")
    with pytest.raises(ValueError, match="168 rows and one column"):
        module.load_generation_profile(csv)
    profile[1] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        module.build_problem(profile)
    profile[1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        module.build_problem(profile)


@pytest.mark.examples
@pytest.mark.skipif(
    not _cached_dol_available(), reason="Ipopt, its shared library, and PyNumero ASL are missing"
)
def test_short_dispatch_objective_totals(tmp_path, monkeypatch):
    """The one-week cost proxy has correct reverse-mode totals at a representative design."""
    monkeypatch.chdir(tmp_path)
    path = EXAMPLE_DIR / "39_differentiable_dispatch" / "run_short_dispatch.py"
    spec = importlib.util.spec_from_file_location("short_dispatch_totals", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    problem = module.build_problem(module.load_generation_profile())
    try:
        problem.run_model()
        names = ("storage_capacity", "max_charge_rate")
        totals = problem.compute_totals(of=["score.week_cost"], wrt=names)
        for name in names:
            baseline = problem.get_val(name).copy()
            values = []
            for delta in (1e-4, -1e-4):
                problem.set_val(name, baseline + delta)
                problem.run_model()
                values.append(problem.get_val("score.week_cost").item())
            problem.set_val(name, baseline)
            finite_difference = (values[0] - values[1]) / 2e-4
            assert totals["score.week_cost", name][0, 0] == pytest.approx(
                finite_difference, rel=2e-3, abs=1e-4
            ), name
    finally:
        problem.cleanup()


@pytest.mark.examples
@pytest.mark.parametrize("script", [_script_param(script) for script in EXAMPLE_SCRIPTS])
def test_example_script_runs(script, tmp_path):
    """Run the script from a temporary copy of its top-level example folder."""
    example_root = script.split("/")[0]
    shutil.copytree(
        EXAMPLE_DIR / example_root,
        tmp_path / example_root,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    script_path = tmp_path / script

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["OPENMDAO_REPORTS"] = "none"
    env["PYTHONIOENCODING"] = "utf-8"
    # Import the h2integrate source tree under test, even when the package is installed elsewhere
    python_path = [str(EXAMPLE_DIR.parent), env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = os.pathsep.join(path for path in python_path if path)

    result = subprocess.run(
        [sys.executable, script_path.name],
        cwd=script_path.parent,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=SCRIPT_TIMEOUT_SECONDS,
        check=False,
    )
    assert result.returncode == 0, (
        f"{script} exited with code {result.returncode}\n"
        f"--- stdout (last 3000 characters) ---\n{result.stdout[-3000:]}\n"
        f"--- stderr (last 6000 characters) ---\n{result.stderr[-6000:]}"
    )
