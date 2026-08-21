import time
import warnings

import numpy as np
import pandas as pd
from pyomo.environ import *  # FIXME: no * imports, delete whole comment when fixed # noqa: F403

from h2integrate.converters.hydrogen.pem_model.PEM_H2_LT_electrolyzer_Clusters import (
    PEM_H2_Clusters as PEMClusters,
)


# from PyOMO import ipOpt !! FOR SANJANA!!
warnings.filterwarnings("ignore", category=RuntimeWarning)
"""
Perform a LCOH analysis for an offshore wind + Hydrogen PEM system

1. Offshore wind site locations and cost details (4 sites, $1300/kw capex + BOS cost which will come
   from Orbit Runs)~
2. Cost Scaling Based on Year (Have Weiser et. al report with cost scaling for fixed and floating
   tech, will implement)
3. Cost Scaling Based on Plant Size (Shields et. Al report)
4. Future Model Development Required:
- Floating Electrolyzer Platform
"""


#
# ---------------------------
#
class run_PEM_clusters:
    """Inputs:
    `electrical_power_signal`: plant power signal in kWh
    `system_size_mw`: total installed electrolyzer capacity (for green steel this is 1000 MW)
    `num_clusters`: number of PEM clusters that can be run independently. May be fractional,
    in which case the plant is modelled as ``floor(num_clusters)`` full clusters plus one
    marginal cluster whose contribution is weighted by the remaining fraction. This keeps
    plant output continuous with respect to a continuous cluster count, which matters when
    the cluster count is an optimization design variable.
    ->ESG note: I have been using num_clusters = 8 for centralized cases
    Nomenclature:
    `cluster`: cluster is built up of 1MW stacks
    `stack`: must be 1MW (because of current PEM model)
    """

    def __init__(
        self,
        electrical_power_signal,
        system_size_mw,
        num_clusters,
        electrolyzer_direct_cost_kw,
        useful_life,
        user_defined_electrolyzer_params,
        verbose=True,
    ):
        # nomen
        self.cluster_cap_mw = np.round(system_size_mw / num_clusters)
        # capacity of each cluster, must be a multiple of 1 MW

        self.num_clusters = num_clusters

        # Split a (possibly fractional) cluster count into whole clusters plus a marginal
        # cluster. ``cluster_weights`` is how much of each simulated cluster actually
        # exists, and is used to weight the per-cluster results when they are aggregated.
        n_full_clusters = int(np.floor(num_clusters + 1e-9))
        marginal_cluster_weight = float(num_clusters) - n_full_clusters
        if marginal_cluster_weight > 1e-9:
            self.cluster_weights = np.array([1.0] * n_full_clusters + [marginal_cluster_weight])
        else:
            self.cluster_weights = np.ones(n_full_clusters)
        self.n_clusters_run = len(self.cluster_weights)

        self.user_params = user_defined_electrolyzer_params
        self.plant_life_yrs = useful_life
        # Do not modify stack_rating_kw or stack_min_power_kw
        # these represent the hard-coded and unmodifiable
        # PEM model basecode
        turndown_ratio = user_defined_electrolyzer_params["turndown_ratio"]
        self.stack_rating_kw = 1000  # single stack rating - DO NOT CHANGE
        self.stack_min_power_kw = turndown_ratio * self.stack_rating_kw
        # self.stack_min_power_kw = 0.1 * self.stack_rating_kw
        self.input_power_kw = electrical_power_signal
        self.cluster_min_power = self.stack_min_power_kw * self.cluster_cap_mw
        self.cluster_max_power = self.stack_rating_kw * self.cluster_cap_mw

        # For the optimization problem:
        self.T = len(self.input_power_kw)
        self.farm_power = 1e9
        self.switching_cost = (
            (electrolyzer_direct_cost_kw * 0.15 * self.cluster_cap_mw * 1000)
            * (1.48e-4)
            / (0.26586)
        )
        self.verbose = verbose

    def run_grid_connected_pem(self, system_size_mw, hydrogen_production_capacity_required_kgphr):
        pem = PEMClusters(
            system_size_mw,
            self.plant_life_yrs,
            **self.user_params,
        )

        power_timeseries, stack_current = pem.grid_connected_func(
            hydrogen_production_capacity_required_kgphr
        )
        h2_ts, h2_tot = pem.run_grid_connected_workaround(power_timeseries, stack_current)
        # h2_ts, h2_tot = pem.run(power_timeseries)
        h2_df_ts = pd.Series(h2_ts, name="Cluster #0")
        h2_df_tot = pd.Series(h2_tot, name="Cluster #0")
        # h2_df_ts = pd.DataFrame(h2_ts, index=list(h2_ts.keys()), columns=['Cluster #0'])
        # h2_df_tot = pd.DataFrame(h2_tot, index=list(h2_tot.keys()), columns=['Cluster #0'])
        return pd.DataFrame(h2_df_ts), pd.DataFrame(h2_df_tot)

    def run(self):
        # TODO: add control type as input!
        clusters = self.create_clusters()  # initialize clusters
        power_to_clusters = self.even_split_power()
        h2_df_ts = pd.DataFrame()
        h2_df_tot = pd.DataFrame()

        col_names = []
        start = time.perf_counter()
        for ci in range(len(clusters)):
            cl_name = f"Cluster #{ci}"
            col_names.append(cl_name)
            h2_ts, h2_tot = clusters[ci].run(power_to_clusters[ci])
            # h2_dict_ts['Cluster #{}'.format(ci)] = h2_ts

            h2_ts_temp = pd.Series(h2_ts, name=cl_name)
            h2_tot_temp = pd.Series(h2_tot, name=cl_name)
            if len(h2_df_tot) == 0:
                # h2_df_ts=pd.concat([h2_df_ts,h2_ts_temp],axis=0,ignore_index=False)
                h2_df_tot = pd.concat([h2_df_tot, h2_tot_temp], axis=0, ignore_index=False)
                h2_df_tot.columns = col_names

                h2_df_ts = pd.concat([h2_df_ts, h2_ts_temp], axis=0, ignore_index=False)
                h2_df_ts.columns = col_names
            else:
                # h2_df_ts = h2_df_ts.join(h2_ts_temp)
                h2_df_tot = h2_df_tot.join(h2_tot_temp)
                h2_df_tot.columns = col_names

                h2_df_ts = h2_df_ts.join(h2_ts_temp)
                h2_df_ts.columns = col_names

        end = time.perf_counter()
        self.clusters = clusters
        if self.verbose:
            print(f"Took {round(end - start, 3)} sec to run the RUN function")
        return h2_df_ts, h2_df_tot
        # return h2_dict_ts, h2_df_tot

    def even_split_power(self):
        """Distribute the input power signal evenly across active PEM clusters.

        At each timestep, the number of clusters turned on is set so that each one
        operates at or above its minimum stable power. The available power is then
        split evenly across the active clusters; inactive clusters receive 0 kW.

        The implementation is defensive against upstream resource-profile issues:
        NaN, +/-inf, and negative input powers are coerced to 0 kW, and a
        non-physical ``cluster_min_power`` (zero, negative, or non-finite) shuts
        every cluster off rather than producing NaNs downstream.

        Returns:
            np.ndarray: Power dispatched to each cluster, shape
            ``(n_clusters_run, n_timesteps)`` in kW.
        """
        start = time.perf_counter()

        # Sanitize the input power signal. Upstream resource profiles can produce
        # NaN, +/-inf, or small negative values; any of these would otherwise
        # propagate through np.floor(...) and blow up the later int() cast.
        input_power_kw = np.asarray(self.input_power_kw, dtype=float)
        input_power_kw = np.nan_to_num(input_power_kw, nan=0.0, posinf=0.0, neginf=0.0)
        np.maximum(input_power_kw, 0.0, out=input_power_kw)

        n_timesteps = input_power_kw.size

        # Guard against a degenerate cluster_min_power. If it isn't a positive
        # finite number, no cluster can be turned on at any timestep.
        if not np.isfinite(self.cluster_min_power) or self.cluster_min_power <= 0:
            num_clusters_on = np.zeros(n_timesteps, dtype=int)
        else:
            num_clusters_on = np.floor(input_power_kw / self.cluster_min_power).astype(int)

        # Clamp the cluster count to [0, n_clusters_run].
        np.clip(num_clusters_on, 0, self.n_clusters_run, out=num_clusters_on)

        # Number of cluster-equivalents operating. The marginal cluster of a fractional
        # cluster count only counts for its weight, so the power split (and therefore the
        # plant output) stays continuous as the cluster count crosses an integer.
        equivalent_clusters_on = np.minimum(num_clusters_on, self.num_clusters)

        # Safe division: where no clusters are on, per-cluster power is 0 kW.
        power_per_cluster = np.divide(
            input_power_kw,
            equivalent_clusters_on,
            out=np.zeros_like(input_power_kw),
            where=equivalent_clusters_on > 0,
        )

        # Build the (n_timesteps, n_clusters_run) dispatch matrix by broadcasting:
        # the first num_clusters_on[t] columns get power_per_cluster[t], rest 0.
        cluster_idx = np.arange(self.n_clusters_run)[np.newaxis, :]
        active_mask = cluster_idx < num_clusters_on[:, np.newaxis]
        power_to_clusters = np.where(active_mask, power_per_cluster[:, np.newaxis], 0.0)

        end = time.perf_counter()
        if self.verbose:
            print(f"Took {round(end - start, 3)} sec to run even_split_power function")

        # Rows are clusters, columns are timesteps: shape (num_clusters, n_timesteps).
        return np.transpose(power_to_clusters)

    def max_h2_cntrl(self):
        # run as many at lower power as possible
        ...

    def min_deg_cntrl(self):
        # run as few as possible
        ...

    def create_clusters(self):
        start = time.perf_counter()
        # TODO fix the power input - don't make it required!
        # in_dict={'dt':3600}
        clusters = PEMClusters(self.cluster_cap_mw, self.plant_life_yrs, **self.user_params)
        stacks = [clusters] * self.n_clusters_run
        end = time.perf_counter()
        if self.verbose:
            print(f"Took {round(end - start, 3)} sec to run the create clusters")
        return stacks
