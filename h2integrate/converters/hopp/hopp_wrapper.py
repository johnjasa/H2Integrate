import hashlib
from pathlib import Path

import dill
import numpy as np
import openmdao.api as om

from h2integrate.converters.hopp.hopp_mgmt import run_hopp, setup_hopp


n_timesteps = 8760


class HOPPSubGroup(om.Group):
    def initialize(self):
        self.options.declare("tech_config", types=dict)
        self.options.declare("plant_config", types=dict)

    def setup(self):
        tech_config = self.options["tech_config"]
        plant_config = self.options["plant_config"]
        hopp_config = self.options["tech_config"]["performance_model"]["config"]
        design_variables = hopp_config["config"]["simulation_options"]["design_variables"]

        self.add_subsystem(
            "hopp_component",
            HOPPComponent(tech_config=tech_config, plant_config=plant_config),
            promotes=["*"],
        )

        c2i_str = "power_capacity_to_interconnect_ratio = (0"
        for v in design_variables:
            if "h" not in v:
                if "wind_turbine" in v:
                    c2i_str += f" + {v}*{hopp_config['technologies']['wind']['num_turbines']}"
                else:
                    c2i_str += f" + {v}"
        c2i_str += f")/{hopp_config['technologies']['grid']['interconnect_kw']}"

        self.add_subsystem(
            "hopp_c2i_component",
            om.ExecComp(c2i_str, power_capacity_to_interconnect_ratio={"units": "unitless"}),
            promotes=["*"],
        )

        for v in design_variables:
            if "h" not in v:
                if "wind_turbine" in v:
                    rating = hopp_config["technologies"]["wind"]["turbine_rating_kw"]
                    self.set_input_defaults(v, units="kW", val=rating)
                else:
                    self.set_input_defaults(v, val=0.0, units="kW")

        if np.any(["battery" in v for v in design_variables]):
            batt_str = (
                "battery_duration = battery_capacity_kwh/battery_capacity_kw "
                "if battery_capacity_kw > 0.001 else 0.0"
            )
            batt_kw = hopp_config["technologies"]["battery"]["system_capacity_kw"]
            batt_kwh = hopp_config["technologies"]["battery"]["system_capacity_kwh"]
            self.add_subsystem(
                "hopp_batt_duration_component",
                om.ExecComp(
                    batt_str,
                    battery_duration={"val": batt_kwh / batt_kw, "units": "h"},
                    battery_capacity_kw={"val": batt_kw, "units": "kW"},
                    battery_capacity_kwh={"val": batt_kwh, "units": "kW*h"},
                ),
                promotes=["*"],
            )

        uphours = np.count_nonzero(hopp_config["site"]["desired_schedule"])
        interconnect_kw = hopp_config["technologies"]["grid"]["interconnect_kw"]
        interconnect_kwh = interconnect_kw * uphours
        e2i_str = f"annual_energy_to_interconnect_potential_ratio = aep/{interconnect_kwh}"
        self.add_subsystem(
            "hopp_e2i_component",
            om.ExecComp(
                e2i_str,
                annual_energy_to_interconnect_potential_ratio={"units": "unitless"},
                aep={"units": "kW*h"},
            ),
            promotes=["*"],
        )

    def configure(self):
        self.promotes("hopp_component", any=["*"])
        self.promotes("hopp_c2i_component", any=["power_capacity_to_interconnect_ratio"])
        self.promotes("hopp_batt_duration_component", any=["battery_duration"])
        self.promotes("hopp_e2i_component", any=["annual_energy_to_interconnect_potential_ratio"])


class HOPPComponent(om.ExplicitComponent):
    """
    A simple OpenMDAO component that represents a HOPP model.

    This component uses caching to store and retrieve results of the HOPP model
    based on the configuration and project lifetime. The caching mechanism helps
    to avoid redundant computations and speeds up the execution by reusing previously
    computed results when the same configuration is encountered.
    """

    def initialize(self):
        self.options.declare("tech_config", types=dict)
        self.options.declare("plant_config", types=dict)

    def setup(self):
        self.hopp_config = self.options["tech_config"]["performance_model"]["config"]

        if "cache" in self.hopp_config["config"]["simulation_options"]:
            self.cache = self.hopp_config["config"]["simulation_options"]["cache"]
        else:
            self.cache = True

        if self.hopp_config["technologies"]["wind"]["turbine_rating_kw"]:
            wind_turbine_rating_kw_init = self.hopp_config["technologies"]["wind"][
                "turbine_rating_kw"
            ]
        else:
            wind_turbine_rating_kw_init = 0.0
        self.add_input("wind_turbine_rating_kw", val=wind_turbine_rating_kw_init, units="kW")

        if self.hopp_config["technologies"]["pv"]["system_capacity_kw"]:
            pv_capacity_kw_init = self.hopp_config["technologies"]["pv"]["system_capacity_kw"]
        else:
            pv_capacity_kw_init = 0.0
        self.add_input("pv_capacity_kw", val=pv_capacity_kw_init, units="kW")

        if self.hopp_config["technologies"]["battery"]["system_capacity_kw"]:
            battery_capacity_kw_init = self.hopp_config["technologies"]["battery"][
                "system_capacity_kw"
            ]
        else:
            battery_capacity_kw_init = 0.0
        self.add_input("battery_capacity_kw", val=battery_capacity_kw_init, units="kW")

        if self.hopp_config["technologies"]["battery"]["system_capacity_kwh"]:
            battery_capacity_kwh_init = self.hopp_config["technologies"]["battery"][
                "system_capacity_kwh"
            ]
        else:
            battery_capacity_kwh_init = 0.0
        self.add_input("battery_capacity_kwh", val=battery_capacity_kwh_init, units="kW*h")

        # Outputs
        self.add_output("percent_load_missed", units="percent", val=0.0)
        self.add_output("curtailment_percent", units="percent", val=0.0)
        self.add_output("aep", units="kW*h", val=0.0)
        self.add_output(
            "electricity_out", val=np.zeros(n_timesteps), units="kW", desc="Power output"
        )
        self.add_output("CapEx", val=0.0, units="USD", desc="Total capital expenditures")
        self.add_output("OpEx", val=0.0, units="USD/year", desc="Total fixed operating costs")

    def compute(self, inputs, outputs):
        # Define the keys of interest from the HOPP results that we want to cache
        keys_of_interest = [
            "percent_load_missed",
            "curtailment_percent",
            "combined_hybrid_power_production_hopp",
            "annual_energies",
            "capex",
            "opex",
        ]

        if self.cache:
            # Create a unique hash for the current configuration to use as a cache key
            config_hash = hashlib.md5(
                str(self.options["tech_config"]["performance_model"]["config"]).encode("utf-8")
                + str(self.options["plant_config"]["plant"]["plant_life"]).encode("utf-8")
            ).hexdigest()

            # Create a cache directory if it doesn't exist
            cache_dir = Path("cache")
            if not cache_dir.exists():
                cache_dir.mkdir(parents=True)
            cache_file = f"cache/{config_hash}.pkl"

        # Check if the results for the current configuration are already cached
        if self.cache and Path(cache_file).exists():
            # Load the cached results
            cache_path = Path(cache_file)
            with cache_path.open("rb") as f:
                subset_of_hopp_results = dill.load(f)
        else:
            electrolyzer_rating = None
            if "electrolyzer_rating" in self.options["tech_config"]:
                electrolyzer_rating = self.options["tech_config"]["electrolyzer_rating"]

            self.hybrid_interface = setup_hopp(
                hopp_config=self.options["tech_config"]["performance_model"]["config"],
                wind_turbine_rating_kw=float(inputs["wind_turbine_rating_kw"]),
                pv_rating_kw=float(inputs["pv_capacity_kw"]),
                battery_rating_kw=float(inputs["battery_capacity_kw"]),
                battery_rating_kwh=float(inputs["battery_capacity_kwh"]),
                electrolyzer_rating=electrolyzer_rating,
            )

            # Run the HOPP model and get the results
            hopp_results = run_hopp(
                self.hybrid_interface, self.options["plant_config"]["plant"]["plant_life"]
            )
            # Extract the subset of results we are interested in
            subset_of_hopp_results = {key: hopp_results[key] for key in keys_of_interest}
            # Cache the results for future use
            if self.cache:
                cache_path = Path(cache_file)
                with cache_path.open("wb") as f:
                    dill.dump(subset_of_hopp_results, f)

        # Set the outputs from the cached or newly computed results
        outputs["percent_load_missed"] = subset_of_hopp_results["percent_load_missed"]
        outputs["curtailment_percent"] = subset_of_hopp_results["curtailment_percent"]
        outputs["aep"] = subset_of_hopp_results["annual_energies"]["hybrid"]
        outputs["electricity_out"] = subset_of_hopp_results["combined_hybrid_power_production_hopp"]
        outputs["CapEx"] = subset_of_hopp_results["capex"]
        outputs["OpEx"] = subset_of_hopp_results["opex"]
