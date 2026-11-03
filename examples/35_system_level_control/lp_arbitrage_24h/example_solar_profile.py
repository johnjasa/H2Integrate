"""Synthetic solar profile for the 24-hour arbitrage example."""

import numpy as np
from attrs import field, define

from h2integrate.core.utilities import BaseConfig, merge_shared_inputs
from h2integrate.core.model_baseclass import PerformanceModelBaseClass


@define(kw_only=True)
class ExampleSolarProfileConfig(BaseConfig):
    """Provide a repeating available solar power profile."""

    production_profile: list[float] = field()


class ExampleSolarProfilePerformanceModel(PerformanceModelBaseClass):
    """Emit the configured solar profile, subject to the SLC command."""

    _time_step_bounds = (3600, 3600)
    _control_classifier = "flexible"

    def initialize(self):
        super().initialize()
        self.commodity = "electricity"
        self.commodity_rate_units = "kW"
        self.commodity_amount_units = "kW*h"

    def setup(self):
        super().setup()
        self.config = ExampleSolarProfileConfig.from_dict(
            merge_shared_inputs(self.options["tech_config"]["model_inputs"], "performance")
        )
        profile = np.asarray(self.config.production_profile, dtype=float)
        if (
            profile.ndim != 1
            or profile.size == 0
            or self.n_timesteps % profile.size != 0
            or not np.all(np.isfinite(profile))
            or np.any(profile < 0)
        ):
            raise ValueError(
                "The nonnegative finite solar profile must evenly divide the simulation length."
            )
        self.profile = np.tile(profile, self.n_timesteps // profile.size)

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        outputs["electricity_out"] = self.profile
        self.apply_curtailment(outputs)
        total = np.sum(outputs["electricity_out"]) * self.dt / 3600
        rated = np.max(self.profile)
        outputs["rated_electricity_production"] = rated
        outputs["total_electricity_produced"] = total
        outputs["annual_electricity_produced"] = total / self.fraction_of_year_simulated
        outputs["capacity_factor"] = (
            total / (rated * self.n_timesteps * self.dt / 3600) if rated else 0
        )
