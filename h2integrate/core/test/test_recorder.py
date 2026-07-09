import sys

import pytest
import openmdao.api as om

from h2integrate import H2IntegrateModel, load_driver_yaml


TEST_RECORDER_OUTPUT_FILE0 = "testingtesting_filename.sql"
TEST_RECORDER_OUTPUT_FILE1 = "testingtesting_filename0.sql"
TEST_RECORDER_OUTPUT_FILE2 = "testingtesting_filename1.sql"


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


@pytest.mark.unit
@pytest.mark.parametrize("example_folder,resource_example_folder", [("05_wind_h2_opt", None)])
def test_multiple_recorders_with_final_iteration(temp_copy_of_example_module_scope, subtests):
    """Test configuring multiple recorders at once, including a problem-level
    recorder that stores only the final design point of an optimization case.
    """
    # initialize H2I using non-optimization config
    example_folder = temp_copy_of_example_module_scope
    input_file = example_folder / "wind_plant_electrolyzer0.yaml"
    h2i = H2IntegrateModel(input_file)

    # load driver config for optimization run
    driver_config = load_driver_yaml(example_folder / "driver_config.yaml")

    driver_filename = "multi_recorder_driver.sql"
    final_filename = "multi_recorder_final.sql"
    output_folder = example_folder / driver_config["general"]["folder_output"]

    # configure two recorders: a driver recorder (every iteration) and a
    # problem recorder (only the final design point)
    driver_config["recorder"] = [
        {
            "flag": True,
            "file": driver_filename,
            "recorder_attachment": "driver",
            "overwrite_recorder": True,
            "includes": ["*"],
            "excludes": ["*resource_data"],
        },
        {
            "flag": True,
            "file": final_filename,
            "recorder_attachment": "problem",
            "overwrite_recorder": True,
            "includes": ["*"],
            "excludes": ["*resource_data"],
        },
    ]
    driver_config["driver"]["optimization"]["max_iter"] = 5  # to prevent tests taking too long

    # reset the driver config in H2I
    h2i.driver_config = driver_config

    # reinitialize the driver model
    h2i.create_driver_model()

    with subtests.test("both recorder paths are registered"):
        assert len(h2i.recorder_paths) == 2
    with subtests.test("primary recorder_path is the first recorder"):
        assert h2i.recorder_path == h2i.recorder_paths[0]
        assert h2i.recorder_path.name == driver_filename

    # run the model
    h2i.run()

    # close recorders so the sql files can be read below
    h2i.prob.cleanup()

    driver_path = output_folder / driver_filename
    final_path = output_folder / final_filename

    with subtests.test("driver recorder output file exists"):
        assert driver_path.exists()
    with subtests.test("final-iteration recorder output file exists"):
        assert final_path.exists()

    # the problem-level recorder should contain exactly one case: the final point
    final_cr = om.CaseReader(str(final_path))
    with subtests.test("final-iteration recorder stores a single case"):
        assert len(final_cr.list_cases("problem", out_stream=None)) == 1

    # the driver recorder should contain more than one case
    driver_cr = om.CaseReader(str(driver_path))
    with subtests.test("driver recorder stores multiple cases"):
        assert len(driver_cr.list_cases("driver", out_stream=None)) > 1
