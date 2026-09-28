import sys

import pytest
import openmdao.api as om

from h2integrate import H2IntegrateModel, load_driver_yaml
from h2integrate.core import pose_optimization
from h2integrate.core.pose_optimization import PoseOptimization


TEST_RECORDER_OUTPUT_FILE0 = "testingtesting_filename.sql"
TEST_RECORDER_OUTPUT_FILE1 = "testingtesting_filename0.sql"
TEST_RECORDER_OUTPUT_FILE2 = "testingtesting_filename1.sql"


class FakeMPIComm:
    """Stand-in for an MPI communicator on one rank of a multi-process run."""

    def __init__(self, rank, size, value_from_root=None):
        self.rank = rank
        self.size = size
        self.value_from_root = value_from_root

    def bcast(self, obj, root=0):
        return obj if self.rank == root else self.value_from_root


class FakeProblem:
    """Minimal stand-in for an ``om.Problem`` that exposes what ``set_recorders`` uses."""

    def __init__(self, comm, run_parallel=True):
        self.comm = comm
        self.model = om.Group()
        self.driver = om.DOEDriver(om.UniformGenerator(num_samples=2))
        self.driver.options["run_parallel"] = run_parallel


def make_recorder_config(folder_output, **recorder_options):
    return {
        "general": {"folder_output": str(folder_output)},
        "recorder": {"flag": True, "file": "cases.sql", **recorder_options},
    }


@pytest.mark.unit
def test_parallel_recorder_path(temp_dir, monkeypatch, subtests):
    """Test that all MPI ranks use the recorder path chosen on rank 0."""
    monkeypatch.setattr(pose_optimization, "MPI", object())
    output_folder = temp_dir / "outputs"
    output_folder.mkdir()
    for fname in ["cases.sql_0", "cases.sql_1", "cases.sql_meta"]:
        (output_folder / fname).touch()

    root_path = PoseOptimization(make_recorder_config(output_folder)).set_recorders(
        FakeProblem(FakeMPIComm(rank=0, size=2))
    )
    with subtests.test("rank 0 does not reuse the name of a previous parallel run"):
        assert root_path == (output_folder / "cases0.sql").absolute()

    other_folder = temp_dir / "not_created"
    other_path = PoseOptimization(make_recorder_config(other_folder)).set_recorders(
        FakeProblem(FakeMPIComm(rank=1, size=2, value_from_root=root_path))
    )
    with subtests.test("other ranks use the path broadcast from rank 0"):
        assert other_path == root_path
    with subtests.test("other ranks do not touch the file system"):
        assert not other_folder.exists()


@pytest.mark.unit
def test_parallel_recorder_overwrite(temp_dir, monkeypatch, subtests):
    """Test that overwriting a parallel recording removes every file from the previous run."""
    monkeypatch.setattr(pose_optimization, "MPI", object())
    stale_files = ["cases.sql", "cases.sql_0", "cases.sql_1", "cases.sql_2", "cases.sql_meta"]
    kept_files = ["cases0.sql_0", "cases.sql.csv", "other_cases.sql"]
    for fname in stale_files + kept_files:
        (temp_dir / fname).touch()

    config = make_recorder_config(temp_dir, overwrite_recorder=True)
    recorder_path = PoseOptimization(config).set_recorders(FakeProblem(FakeMPIComm(0, 2)))

    with subtests.test("recorder path is the requested file"):
        assert recorder_path == (temp_dir / "cases.sql").absolute()
    with subtests.test("files from the previous run are removed"):
        assert not any((temp_dir / fname).exists() for fname in stale_files)
    with subtests.test("unrelated files are kept"):
        assert all((temp_dir / fname).exists() for fname in kept_files)


@pytest.mark.unit
def test_recorder_file_in_subfolder(temp_dir):
    """Test that the recorder file can be placed in a subfolder of the output folder."""
    config = make_recorder_config(temp_dir)
    config["recorder"]["file"] = "sweep_1/cases.sql"
    recorder_path = PoseOptimization(config).set_recorders(FakeProblem(FakeMPIComm(0, 1)))

    assert recorder_path == (temp_dir / "sweep_1" / "cases.sql").absolute()
    assert recorder_path.parent.is_dir()


@pytest.mark.unit
def test_parallel_model_recorder_raises(temp_dir, monkeypatch):
    """Test that model recording is rejected when cases run in parallel under MPI."""
    monkeypatch.setattr(pose_optimization, "MPI", object())
    config = make_recorder_config(temp_dir, recorder_attachment="model")

    with pytest.raises(ValueError, match="cannot record cases that are run in parallel"):
        PoseOptimization(config).set_recorders(FakeProblem(FakeMPIComm(0, 2)))


@pytest.mark.unit
@pytest.mark.parametrize("example_folder,resource_example_folder", [("05_wind_h2_opt", None)])
def test_output_folder_creation_first_run(temp_copy_of_example_module_scope, subtests):
    """Test that the sql file is written to the output folder with the specified name."""

    # initialize H2I using non-optimization config
    example_folder = temp_copy_of_example_module_scope
    input_file = example_folder / "wind_plant_electrolyzer0.yaml"
    h2i = H2IntegrateModel(input_file)

    # load driver config for optimization run
    driver_config = load_driver_yaml(example_folder / "driver_config.yaml")

    # update driver config params with test variables
    filename_initial = TEST_RECORDER_OUTPUT_FILE0
    output_folder = example_folder / driver_config["general"]["folder_output"]
    driver_config["recorder"]["file"] = filename_initial
    driver_config["driver"]["optimization"]["max_iter"] = 5  # to prevent tests taking too long

    # reset the driver config in H2I
    h2i.driver_config = driver_config

    # reinitialize the driver model
    h2i.create_driver_model()

    # check if output folder and output files exist
    output_folder_exists = output_folder.exists()
    output_file_exists_prerun = (output_folder / filename_initial).exists()

    with subtests.test("Run 0: output folder exists"):
        assert output_folder_exists is True
    with subtests.test("Run 0: recorder output file does not exist yet"):
        assert output_file_exists_prerun is False

    # run the model
    h2i.run()

    # check that recorder file was created
    output_file_exists_postrun = (output_folder / filename_initial).exists()
    with subtests.test("Run 0: recorder output file exists after run"):
        assert output_file_exists_postrun is True


@pytest.mark.unit
@pytest.mark.parametrize("example_folder,resource_example_folder", [("05_wind_h2_opt", None)])
def test_output_new_recorder_filename_second_run(temp_copy_of_example_module_scope, subtests):
    """Test that the sql file is written to the output folder with the specified base name and
    an appended 0.
    """

    # initialize H2I using non-optimization config
    example_folder = temp_copy_of_example_module_scope
    input_file = example_folder / "wind_plant_electrolyzer0.yaml"
    h2i = H2IntegrateModel(input_file)

    # load driver config for optimization run
    driver_config = load_driver_yaml(example_folder / "driver_config.yaml")

    # update driver config params with test variables
    filename_initial = TEST_RECORDER_OUTPUT_FILE0
    filename_expected = TEST_RECORDER_OUTPUT_FILE1

    output_folder = example_folder / driver_config["general"]["folder_output"]
    driver_config["recorder"]["file"] = filename_initial
    driver_config["driver"]["optimization"]["max_iter"] = 5  # to prevent tests taking too long

    # reset the driver config in H2I
    h2i.driver_config = driver_config

    # reinitialize the driver model
    h2i.create_driver_model()

    # check if output folder and output files exist
    with subtests.test("Run 1: output folder exists"):
        assert output_folder.exists()
    with subtests.test("Run 1: initial recorder output file exists"):
        assert (output_folder / filename_initial).exists()

    # run the model
    h2i.run()

    # check that the new recorder file was created
    with subtests.test("Run 1: new recorder output file was made"):
        assert (output_folder / filename_expected).exists()


@pytest.mark.unit
@pytest.mark.parametrize("example_folder,resource_example_folder", [("05_wind_h2_opt", None)])
@pytest.mark.xfail(sys.platform == "win32", reason="OpenMDAO incorrectly ends SQL processes")
def test_output_new_recorder_overwrite_first_run(temp_copy_of_example_module_scope, subtests):
    # initialize H2I using non-optimization config
    example_folder = temp_copy_of_example_module_scope
    input_file = example_folder / "wind_plant_electrolyzer0.yaml"
    h2i = H2IntegrateModel(input_file)

    # load driver config for optimization run
    driver_config = load_driver_yaml(example_folder / "driver_config.yaml")

    # update driver config params with test variables
    filename_initial = TEST_RECORDER_OUTPUT_FILE0
    filename_exists_if_failed = TEST_RECORDER_OUTPUT_FILE2
    output_folder = example_folder / driver_config["general"]["folder_output"]
    driver_config["recorder"]["file"] = filename_initial

    # specify that we want the previous file overwritten rather
    # than create a new file
    driver_config["recorder"].update({"overwrite_recorder": True})
    driver_config["driver"]["optimization"]["max_iter"] = 5  # to prevent tests taking too long

    # reset the driver config in H2I
    h2i.driver_config = driver_config

    # reinitialize the driver model
    h2i.create_driver_model()

    # check if output folder and output files exist
    with subtests.test("Run 2: output folder exists"):
        assert output_folder.exists()
    with subtests.test("Run 2: initial recorder output file exists"):
        assert (output_folder / filename_initial).exists()

    # run the model
    h2i.run()

    # check that recorder file was overwritten
    with subtests.test("Run 2: initial output file was overwritten"):
        assert not (output_folder / filename_exists_if_failed).exists()


@pytest.mark.unit
@pytest.mark.parametrize("example_folder,resource_example_folder", [("05_wind_h2_opt", None)])
def test_output_new_recorder_filename_third_run(temp_copy_of_example_module_scope, subtests):
    # initialize H2I using non-optimization config
    example_folder = temp_copy_of_example_module_scope
    input_file = example_folder / "wind_plant_electrolyzer0.yaml"
    h2i = H2IntegrateModel(input_file)

    # load driver config for optimization run
    driver_config = load_driver_yaml(example_folder / "driver_config.yaml")

    # update driver config params with test variables
    filename_initial = TEST_RECORDER_OUTPUT_FILE0
    filename_second = TEST_RECORDER_OUTPUT_FILE1
    filename_expected = TEST_RECORDER_OUTPUT_FILE2
    output_folder = example_folder / driver_config["general"]["folder_output"]
    driver_config["recorder"]["file"] = filename_initial
    driver_config["driver"]["optimization"]["max_iter"] = 5  # to prevent tests taking too long

    # reset the driver config in H2I
    h2i.driver_config = driver_config

    # reinitialize the driver model
    h2i.create_driver_model()

    # check if output folder and output files exist
    with subtests.test("Run 3: output folder exists"):
        assert output_folder.exists()
    with subtests.test("Run 3: initial recorder output file exists"):
        assert (output_folder / filename_initial).exists()
    with subtests.test("Run 3: second recorder output file exists"):
        assert (output_folder / filename_second).exists()

    # run the model
    h2i.run()

    # check that the new recorder file was created
    with subtests.test("Run 3: new recorder output file was made"):
        assert (output_folder / filename_expected).exists()
