"""
This file is based on the WISDEM file of the same name: https://github.com/NLRWindSystems/WISDEM
and also based off of the H2Integrate file of the same name originally adapted by Jared Thomas.
"""

import warnings
from pathlib import Path

import openmdao.api as om

from h2integrate.core.file_utils import make_unique_case_name, check_file_format_for_csv_generator


class PoseOptimization:
    """
    This class contains a collection of methods for setting up an OpenMDAO
    optimization problem for a H2Integrate simulation.

    Args:
        config: instance of a H2Integrate config containing all desired simulation set up
    """

    def __init__(self, config):
        """
        This method primarily establishes lists of optimization methods
        available through different optimization drivers"""

        self.config = config

        self.scipy_methods = [
            "SLSQP",
            "Nelder-Mead",
            "COBYLA",
        ]

        self.pyoptsparse_methods = [
            "SNOPT",
            "CONMIN",
            "NSGA2",
        ]

    def _get_step_size(self):
        """
        If a step size for the driver-level finite differencing is provided,
        use that step size. Otherwise use a default value.

        Returns:
            step size (float): step size for optimization
        """

        if "step_size" not in self.config["driver"]["optimization"]:
            step_size = 1.0e-6
            warnings.warn(
                f"Step size was not specified, setting step size to {step_size}. \
                Step size may be set in the h2integrate \
                config file under opt_options/driver/optimization/step_size \
                and should be of type float",
                UserWarning,
            )
        else:
            step_size = self.config["driver"]["optimization"]["step_size"]

        return step_size

    def _set_optimizer_properties(
        self, opt_prob, options_keys=[], opt_settings_keys=[], mapped_keys={}
    ):
        """Set optimizer properties on ``driver.options`` and ``driver.opt_settings``.

        Refer to OpenMDAO driver documentation to determine which settings belong
        in ``driver.options`` versus ``driver.opt_settings``.

        Args:
            opt_prob (OpenMDAO problem object):  The hybrid plant OpenMDAO problem object.
            options_keys (list, optional): List of keys for driver opt_settings
                to be set. Defaults to [].
            opt_settings_keys (list, optional): List of keys for driver options
                to be set. Defaults to [].
            mapped_keys (dict, optional): Key pairs where the YAML name differs
                from what is expected by the driver. Specifically, the key is
                what is given in the YAML and the value is what is expected by
                the driver. Defaults to {}.

        Returns:
            opt_prob (OpenMDAO problem object): The updated openmdao problem
                object with driver settings applied.
        """

        opt_options = self.config["driver"]["optimization"]

        # Loop through all of the options provided and set them in the OM driver object
        for key in options_keys:
            if key in opt_options:
                if key in mapped_keys:
                    opt_prob.driver.options[mapped_keys[key]] = opt_options[key]
                else:
                    opt_prob.driver.options[key] = opt_options[key]

        # Loop through all of the opt_settings provided and set them in the OM driver object
        for key in opt_settings_keys:
            if key in opt_options:
                if key in mapped_keys:
                    opt_prob.driver.opt_settings[mapped_keys[key]] = opt_options[key]
                else:
                    opt_prob.driver.opt_settings[key] = opt_options[key]

        return opt_prob

    def _get_autoscaler(self, autoscaler_name):
        """Return an OpenMDAO autoscaler instance for the requested name.

        Autoscalers are set on the driver and therefore work with any optimization
        driver (Scipy, pyOptSparse, GA, pymoo, etc.), not just gradient-based ones.

        Args:
            autoscaler_name (str): Name of the autoscaler requested in the driver
                config. Supported values are ``"bounds"`` (normalize each design
                variable to the interval [0, 1] using its bounds) and
                ``"none"``/``"default"`` (use the driver's default, user-declared
                ``ref``/``ref0``/``scaler``/``adder`` scaling).

        Raises:
            ValueError: The requested autoscaler is not supported.

        Returns:
            Autoscaler: An OpenMDAO autoscaler instance.
        """
        autoscalers = {
            "bounds": om.BoundsAutoscaler,
            "none": om.Autoscaler,
            "default": om.Autoscaler,
        }
        key = str(autoscaler_name).lower()
        if key not in autoscalers:
            raise ValueError(
                f"Autoscaler '{autoscaler_name}' is not supported. "
                f"Supported options are {sorted(autoscalers)}."
            )
        return autoscalers[key]()

    def set_driver(self, opt_prob):
        """set which optimization driver to use and set options

        Args:
            opt_prob (openmdao problem instance): openmdao problem class instance
                for current optimization problem

        Raises:
            ImportError: An optimization algorithm from pyoptsparse was selected,
                but pyoptsparse is not installed
            ImportError: An optimization algorithm from pyoptsparse was selected,
                but the algorithm code is not currently installed within pyoptsparse
            ImportError: An optimization algorithm was requested from NLopt, but
                NLopt is not currently installed.
            ValueError: The selected optimizer is not yet supported.
            Exception: The specified generator type for the OpenMDAO design
                of experiments is unsupported.

        Returns:
            opt_prob (openmdao problem instance): openmdao problem class instance,
                edited from input with desired driver and driver options
        """

        folder_output = self.config["general"]["folder_output"]

        if self.config["driver"].get("optimization", {}).get("flag", False):
            opt_options = self.config["driver"]["optimization"]
            step_size = self._get_step_size()

            if "step_calc" in opt_options.keys():
                if opt_options["step_calc"] == "None":
                    step_calc = None
                else:
                    step_calc = opt_options["step_calc"]
            else:
                step_calc = None

            if "form" in opt_options.keys():
                if opt_options["form"] == "None":
                    form = None
                else:
                    form = opt_options["form"]
            else:
                form = None

            opt_prob.model.approx_totals(
                method="fd", step=step_size, form=form, step_calc=step_calc
            )

            # Set optimization solver and options. First, Scipy's SLSQP and COBYLA
            if opt_options["solver"] in self.scipy_methods:
                opt_prob.driver = om.ScipyOptimizeDriver()
                opt_prob.driver.options["optimizer"] = opt_options["solver"]

                options_keys = ["tol", "max_iter", "disp"]
                opt_settings_keys = ["rhobeg", "catol", "adaptive"]
                mapped_keys = {"max_iter": "maxiter"}
                opt_prob = self._set_optimizer_properties(
                    opt_prob, options_keys, opt_settings_keys, mapped_keys
                )

            # The next two optimization methods require pyOptSparse.
            elif opt_options["solver"] in self.pyoptsparse_methods:
                try:
                    from openmdao.api import pyOptSparseDriver
                except RuntimeError:
                    raise ImportError(
                        f"You requested the optimization solver {opt_options['solver']}, \
                        but you have not installed pyOptSparse. \
                        Please do so and rerun."
                    ) from None
                opt_prob.driver = pyOptSparseDriver(gradient_method=opt_options["gradient_method"])

                try:
                    opt_prob.driver.options["optimizer"] = opt_options["solver"]
                except ImportError:
                    raise ImportError(
                        f"You requested the optimization solver {opt_options['solver']}, \
                        but you have not installed it within pyOptSparse. \
                        Please build {opt_options['solver']} and rerun."
                    ) from None

                # Most of the pyOptSparse options have special syntax when setting them,
                # so here we set them by hand instead of using
                # `_set_optimizer_properties` for SNOPT and CONMIN.
                if opt_options["solver"] == "CONMIN":
                    opt_prob.driver.opt_settings["ITMAX"] = opt_options["max_iter"]

                if opt_options["solver"] == "NSGA2":
                    opt_settings_keys = [
                        "PopSize",
                        "maxGen",
                        "pCross_real",
                        "pMut_real",
                        "eta_c",
                        "eta_m",
                        "pCross_bin",
                        "pMut_bin",
                        "PrintOut",
                        "seed",
                        "xinit",
                    ]
                    opt_prob = self._set_optimizer_properties(
                        opt_prob, opt_settings_keys=opt_settings_keys
                    )

                elif opt_options["solver"] == "SNOPT":
                    opt_prob.driver.opt_settings["Major optimality tolerance"] = float(
                        opt_options["tol"]
                    )
                    opt_prob.driver.opt_settings["Major iterations limit"] = int(
                        opt_options["max_major_iter"]
                    )
                    opt_prob.driver.opt_settings["Iterations limit"] = int(
                        opt_options["max_minor_iter"]
                    )
                    opt_prob.driver.opt_settings["Major feasibility tolerance"] = float(
                        opt_options["tol"]
                    )
                    if "time_limit" in opt_options:
                        opt_prob.driver.opt_settings["Time limit"] = int(opt_options["time_limit"])
                    opt_prob.driver.opt_settings["Summary file"] = (
                        Path(folder_output) / "SNOPT_Summary_file.txt"
                    )
                    opt_prob.driver.opt_settings["Print file"] = (
                        Path(folder_output) / "SNOPT_Print_file.txt"
                    )
                    if "hist_file_name" in opt_options:
                        opt_prob.driver.hist_file = opt_options["hist_file_name"]
                    if "verify_level" in opt_options:
                        opt_prob.driver.opt_settings["Verify level"] = opt_options["verify_level"]
                    else:
                        opt_prob.driver.opt_settings["Verify level"] = -1
                if "hotstart_file" in opt_options:
                    opt_prob.driver.hotstart_file = opt_options["hotstart_file"]

            elif opt_options["solver"] == "GA":
                opt_prob.driver = om.SimpleGADriver()
                options_keys = [
                    "Pc",
                    "Pm",
                    "bits",
                    "compute_pareto",
                    "cross_bits",
                    "elitism",
                    "gray",
                    "max_gen",
                    "multi_obj_exponent",
                    "multi_obj_weights",
                    "penalty_exponent",
                    "penalty_parameter",
                    "pop_size",
                    "procs_per_model",
                    "run_parallel",
                ]
                opt_prob = self._set_optimizer_properties(opt_prob, options_keys)

            elif opt_options["solver"] == "pymoo":
                # Genetic / evolutionary optimizers from the pymoo library,
                # exposed through OpenMDAO's pymooDriver. The specific algorithm
                # (e.g. "GA", "DE", "NSGA2") is selected via the "optimizer" key.
                try:
                    from openmdao.drivers.pymoo_driver import pymooDriver
                except ImportError as err:
                    raise ImportError(
                        "You requested the 'pymoo' solver, but pymoo is not "
                        "installed. Install it with `pip install pymoo` and rerun."
                    ) from err

                opt_prob.driver = pymooDriver()
                opt_prob.driver.options["optimizer"] = opt_options.get("optimizer", "GA")
                if "disp" in opt_options:
                    opt_prob.driver.options["disp"] = opt_options["disp"]
                if "procs_per_model" in opt_options:
                    opt_prob.driver.options["procs_per_model"] = opt_options["procs_per_model"]

                # Algorithm hyperparameters (population size, operators, ...) are
                # passed straight through to the pymoo algorithm constructor.
                alg_settings = dict(opt_options.get("alg_settings", {}))
                if "pop_size" in opt_options:
                    alg_settings.setdefault("pop_size", opt_options["pop_size"])
                opt_prob.driver.alg_settings.update(alg_settings)

                # Run-level settings passed to pymoo's minimize()/setup() (seed,
                # verbose, termination, ...). "n_gen" is a convenience shortcut
                # for the ("n_gen", N) termination tuple.
                run_settings = dict(opt_options.get("run_settings", {}))
                if "seed" in opt_options:
                    run_settings.setdefault("seed", opt_options["seed"])
                if "n_gen" in opt_options:
                    run_settings.setdefault("termination", ("n_gen", int(opt_options["n_gen"])))
                opt_prob.driver.run_settings.update(run_settings)

            else:
                raise ValueError(f"Optimizer {opt_options['solver']} is not yet supported.")

            # Optionally override how the driver scales design variables. Because
            # the autoscaler lives on the base Driver, this works for every
            # optimization driver, not just the Scipy/SLSQP-style ones.
            if opt_options.get("autoscaler") is not None:
                opt_prob.driver.autoscaler = self._get_autoscaler(opt_options["autoscaler"])

            if opt_options["debug_print"]:
                opt_prob.driver.options["debug_print"] = [
                    "desvars",
                    "ln_cons",
                    "nl_cons",
                    "objs",
                    "totals",
                ]

        elif self.config["driver"].get("parameter_sweep", False) or self.config["driver"].get(
            "design_of_experiments", False
        ):
            # Support both "parameter_sweep" (preferred) and legacy "design_of_experiments" key
            sweep_options = self.config["driver"].get(
                "parameter_sweep", self.config["driver"].get("design_of_experiments", {})
            )
            if sweep_options["flag"]:
                if sweep_options["generator"].lower() == "uniform":
                    generator = om.UniformGenerator(
                        num_samples=int(sweep_options["num_samples"]),
                        seed=sweep_options["seed"],
                    )
                elif sweep_options["generator"].lower() == "fullfact":
                    generator = om.FullFactorialGenerator(levels=int(sweep_options["levels"]))
                elif sweep_options["generator"].lower() == "plackettburman":
                    generator = om.PlackettBurmanGenerator()
                elif sweep_options["generator"].lower() == "boxbehnken":
                    generator = om.BoxBehnkenGenerator()
                elif sweep_options["generator"].lower() == "latinhypercube":
                    generator = om.LatinHypercubeGenerator(
                        samples=int(sweep_options["num_samples"]),
                        criterion=sweep_options["criterion"],
                        seed=sweep_options["seed"],
                    )
                elif sweep_options["generator"].lower() == "csvgen":
                    valid_file = check_file_format_for_csv_generator(
                        sweep_options["filename"], self.config, check_only=True
                    )
                    if not valid_file:
                        raise UserWarning(
                            f"There may be issues with the csv file {sweep_options['filename']}, "
                            f"which may cause errors within OpenMDAO. "
                            "To check this csv file or create a new one, run the function "
                            "h2integrate.core.utilities.check_file_format_for_csv_generator()."
                        )
                    generator = om.CSVGenerator(
                        filename=sweep_options["filename"],
                    )
                else:
                    raise Exception(
                        "The generator type {} is unsupported.".format(sweep_options["generator"])
                    )

                # Initialize driver (OpenMDAO calls this DOEDriver / "Design of Experiments")
                opt_prob.driver = om.DOEDriver(generator)

                if sweep_options["debug_print"]:
                    opt_prob.driver.options["debug_print"] = [
                        "desvars",
                        "ln_cons",
                        "nl_cons",
                        "objs",
                    ]

                # options
                if "run_parallel" in sweep_options:
                    opt_prob.driver.options["run_parallel"] = sweep_options["run_parallel"]

        else:
            warnings.warn(
                "Design variables are set to be optimized or studied, but no driver is selected. "
                "If you want to run an optimization, please enable a driver.",
                UserWarning,
            )

        return opt_prob

    def set_objective(self, opt_prob):
        """Set merit figure. Each objective has its own scaling.  Check first for user override.

        The optimization is always minimizing the objective. If you wish to maximize the objective,
        use a negative ref or scaler value in the config.

        Args:
            opt_prob (openmdao problem instance): openmdao problem instance for
                current optimization problem

        Returns:
            opt_prob (openmdao problem instance): openmdao problem instance for
                current optimization problem with objective set
        """
        if self.config.get("objective", False):
            if "ref" in self.config["objective"]:
                ref = self.config["objective"]["ref"]
            else:
                ref = None
            opt_prob.model.add_objective(
                self.config["objective"]["name"],
                ref=ref,
            )

        return opt_prob

    def set_design_variables(self, opt_prob):
        """Set optimization design variables.

        Args:
            opt_prob (openmdao problem instance): openmdao problem instance for
                current optimization problem

        Returns:
            opt_prob (openmdao problem instance): openmdao problem instance for
                current optimization problem with design variables set
        """

        for technology, variables in self.config["design_variables"].items():
            for key, value in variables.items():
                if value["flag"]:
                    value.pop("flag")
                    opt_prob.model.add_design_var(f"{technology}.{key}", **value)

        return opt_prob

    def set_constraints(self, opt_prob):
        """sets up optimization constraints for the h2integrate optimization problem

        Args:
            opt_prob (openmdao problem instance): openmdao problem instance for
                current optimization problem

        Raises:
            Exception: all design variables must have at least one of an upper
                and lower bound specified

        Returns:
            opt_prob (openmdao problem instance): openmdao problem instance for
                current optimization problem edited to include constraint setup
        """
        if self.config.get("constraints", False):
            for technology, variables in self.config["constraints"].items():
                for key, value in variables.items():
                    if value["flag"]:
                        value.pop("flag")
                        opt_prob.model.add_constraint(f"{technology}.{key}", **value)

    def set_recorders(self, opt_prob):
        """sets up one or more recorders for the openmdao problem as desired in the input yaml

        The ``recorder`` entry in the driver config may be either a single mapping
        (one recorder) or a list of mappings (multiple recorders). Each recorder may
        write to its own sql file and record different variables via its own
        ``includes``/``excludes`` statements. In addition to the ``driver`` and
        ``model`` attachments, a recorder may be attached to the ``problem``, which
        records only the final design point of an optimization case (the case is
        written by ``H2IntegrateModel.run`` after the driver has finished).

        Args:
            opt_prob (openmdao problem instance): openmdao problem instance
                for current optimization problem

        Returns:
            recorder_paths (list of Path): Paths to each enabled recorder file.
                Empty list if no recorders are enabled.
        """
        recorder_config = self.config.get("recorder")
        if recorder_config is None:
            return []

        # Normalize to a list so one or many recorders are handled the same way
        if isinstance(recorder_config, dict):
            recorder_configs = [recorder_config]
        else:
            recorder_configs = recorder_config

        folder_output = self.config["general"]["folder_output"]

        recorder_paths = []
        for recorder_cfg in recorder_configs:
            recorder_path = self._set_single_recorder(opt_prob, recorder_cfg, folder_output)
            if recorder_path is not None:
                recorder_paths.append(recorder_path)

        return recorder_paths

    def _set_single_recorder(self, opt_prob, recorder_cfg, folder_output):
        """Set up a single recorder from its config mapping.

        Args:
            opt_prob (openmdao problem instance): openmdao problem instance for
                current optimization problem
            recorder_cfg (dict): configuration for a single recorder
            folder_output (str or Path): output folder for recorder files

        Raises:
            ValueError: The requested recorder attachment is not supported.

        Returns:
            recorder_path (Path or None): Path to the recorder file if the recorder
                is enabled, None otherwise.
        """
        if not recorder_cfg.get("flag", False):
            return None

        # Set recorder on the OpenMDAO driver level using the `optimization_log`
        # filename supplied in the optimization yaml
        recorder_options = ["record_inputs", "record_outputs", "record_residuals"]

        # Check that the output folder exists and create it if needed
        if not Path(folder_output).exists():
            Path(folder_output).mkdir(parents=True, exist_ok=True)

        overwrite_recorder = recorder_cfg.get("overwrite_recorder", False)
        recorder_path = Path(folder_output) / recorder_cfg["file"]

        if not overwrite_recorder:
            # make a unique filename with the same base as recorder_cfg["file"]
            # separate out the filename without the extension
            file_base = recorder_cfg["file"].split(".sql")[0]

            recorder_fname = make_unique_case_name(Path(folder_output), f"{file_base}.sql", ".sql")
            recorder_path = Path(folder_output) / recorder_fname

        recorder_attachment = recorder_cfg.get("recorder_attachment", "driver").lower()
        allowed_attachments = ["driver", "model", "problem"]
        if recorder_attachment not in allowed_attachments:
            msg = (
                f"Invalid recorder attachment '{recorder_attachment}'. "
                f"Currently supported options are {allowed_attachments}. "
                "We recommend using 'driver' if running an optimization "
                "or parameter sweep in parallel, and 'problem' to record only "
                "the final design point of an optimization case."
            )
            raise ValueError(msg)

        # Create recorder
        recorder = om.SqliteRecorder(recorder_path)

        if recorder_attachment == "model":
            # add the recorder to the model
            recorder_options += ["options_excludes"]

            opt_prob.model.add_recorder(recorder)

            for recorder_opt in recorder_options:
                if recorder_opt in recorder_cfg:
                    opt_prob.model.recording_options[recorder_opt] = recorder_cfg.get(recorder_opt)

            opt_prob.model.recording_options["includes"] = recorder_cfg.get("includes", ["*"])
            opt_prob.model.recording_options["excludes"] = recorder_cfg.get(
                "excludes", ["*resource_data"]
            )

        elif recorder_attachment == "problem":
            # add the recorder to the problem. Problem-level recorders only write a
            # case when opt_prob.record() is called, which H2IntegrateModel does after
            # run_driver, so only the final design point is stored.
            opt_prob.add_recorder(recorder)

            for recorder_opt in recorder_options:
                if recorder_opt in recorder_cfg:
                    opt_prob.recording_options[recorder_opt] = recorder_cfg.get(recorder_opt)

            opt_prob.recording_options["includes"] = recorder_cfg.get("includes", ["*"])
            opt_prob.recording_options["excludes"] = recorder_cfg.get(
                "excludes", ["*resource_data"]
            )

        else:  # recorder_attachment == "driver"
            recorder_options += [
                "record_constraints",
                "record_derivative",
                "record_desvars",
                "record_objectives",
            ]
            # add the recorder to the driver
            opt_prob.driver.add_recorder(recorder)

            for recorder_opt in recorder_options:
                if recorder_opt in recorder_cfg:
                    opt_prob.driver.recording_options[recorder_opt] = recorder_cfg.get(recorder_opt)

            opt_prob.driver.recording_options["includes"] = recorder_cfg.get("includes", ["*"])
            opt_prob.driver.recording_options["excludes"] = recorder_cfg.get(
                "excludes", ["*resource_data"]
            )

        return recorder_path

    def set_restart(self, opt_prob):
        """
        Prepares to restart from last recorded iteration if the original
        problem was set up for warm start

        Args:
            opt_prob (openmdao problem instance): openmdao problem instance for
            current optimization problem

        Returns:
            opt_prob (openmdao problem instance): openmdao problem instance
                for current optimization problem set up for warm start
        """

        if "warmstart_file" in self.config["driver"]["optimization"]:
            # Directly read the pyoptsparse sqlite db file
            from pyoptsparse import SqliteDict

            db = SqliteDict(self.config["driver"]["optimization"]["warmstart_file"])

            # Grab the last iteration's design variables
            last_key = db["last"]
            desvars = db[last_key]["xuser"]

            # Obtain the already-setup OM problem's design variables
            if opt_prob.model._static_mode:
                design_vars = opt_prob.model._static_design_vars
            else:
                design_vars = opt_prob.model._design_vars

            # Get the absolute names from the promoted names within the OM model.
            # We need this because the pyoptsparse db has the absolute names for
            # variables but the OM model uses the promoted names.
            prom2abs = opt_prob.model._var_allprocs_prom2abs_list["output"]
            abs2prom = {}
            for key in design_vars:
                abs2prom[prom2abs[key][0]] = key

            # Loop through each design variable
            for key in desvars:
                prom_key = abs2prom[key]

                # Scale each DV based on the OM scaling from the problem.
                # This assumes we're running the same problem with the same scaling
                scaler = design_vars[prom_key]["scaler"]
                adder = design_vars[prom_key]["adder"]

                if scaler is None:
                    scaler = 1.0
                if adder is None:
                    adder = 0.0

                desvars[key] / scaler - adder

        return opt_prob
