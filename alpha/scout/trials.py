"""Trial recording and the effective number of independent trials per family.

A trial is one configuration evaluated on real (not permuted, not resampled)
data. Permutation, bootstrap and cross-validation resamples are NOT trials; each
such test is recorded once as a `test_result` row.

Effective number of trials = number of correlation clusters among the trials'
return series:

    distance d_ij = sqrt( (1 − ρ_ij) / 2 )
    average-linkage hierarchical clustering, cut at d = 0.5  (i.e. ρ = 0.5)
    N_eff = number of clusters

N identical trials give 1, N uncorrelated trials give N, and 90 + 10 tight trials
give 2 (a participation-ratio estimate gives 1.23 there, which under-deflates the
lopsided grids that real searches produce). The single cut at ρ = 0.5 is the only
tuning choice; P2 calibration may revisit it.

Family N for the DSR = N_eff of the family's recorded trials
                       + prior_trials_int declared by retro registrations.
Prior trials are counted as independent, which is the conservative choice.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from alpha.scout.ledger import REPO_ROOT_PATH, Ledger
from alpha.scout.registration import registration_rows

CLUSTER_CUT_CORRELATION_FLOAT = 0.5
MODE_TUPLE = ("parity", "truth")


def effective_trial_count(trial_return_df: pd.DataFrame, min_overlap_int: int = 60) -> float:
    """Number of correlation clusters among the trials (see module docstring)."""
    trial_count_int = trial_return_df.shape[1]
    if trial_count_int == 0:
        raise ValueError("No trials.")
    if trial_count_int == 1:
        return 1.0
    constant_column_list = [name for name in trial_return_df.columns if not trial_return_df[name].std(skipna=True) > 0.0]
    if constant_column_list:
        raise ValueError(f"Trials with constant or missing returns have no correlation: {constant_column_list}.")
    correlation_df = trial_return_df.corr(min_periods=min_overlap_int)
    if correlation_df.isna().to_numpy().any():
        raise ValueError(f"Some trial pairs overlap on fewer than {min_overlap_int} observations.")

    distance_mat = np.sqrt(np.clip((1.0 - correlation_df.to_numpy()) / 2.0, 0.0, None))
    np.fill_diagonal(distance_mat, 0.0)
    linkage_mat = linkage(squareform(distance_mat, checks=False), method="average")
    cut_distance_float = np.sqrt((1.0 - CLUSTER_CUT_CORRELATION_FLOAT) / 2.0)
    cluster_label_vec = fcluster(linkage_mat, t=cut_distance_float, criterion="distance")
    return float(np.unique(cluster_label_vec).size)


def current_code_commit_str() -> str | None:
    """HEAD commit of this checkout, or None when git is unavailable."""
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT_PATH, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def record_trial(
    ledger: Ledger,
    registration_id_str: str,
    config_dict: dict,
    per_period_sharpe_float: float,
    observation_count_int: int,
    window_str: str,
    station_str: str,
    mode_str: str = "parity",
    data_snapshot_id_str: str | None = None,
    metrics_dict: dict | None = None,
    return_series_path_str: str | None = None,
    vault_touched_bool: bool = False,
) -> dict:
    """Write one trial row. The registration must exist and the config must be in its frozen grid."""
    if mode_str not in MODE_TUPLE:
        raise ValueError(f"mode_str must be one of {MODE_TUPLE}.")

    def precondition(existing_ledger: Ledger) -> dict:
        registration_row_dict = registration_rows(existing_ledger).get(registration_id_str)
        if registration_row_dict is None:
            raise ValueError(f"Registration {registration_id_str!r} does not exist; register before running trials.")
        grid_dict = registration_row_dict["param_grid_dict"]
        if grid_dict:
            in_grid_bool = set(config_dict) == set(grid_dict) and all(
                config_dict[name_str] in value_list for name_str, value_list in grid_dict.items()
            )
            if not in_grid_bool:
                raise ValueError(
                    f"Config {config_dict} is outside the frozen grid of {registration_id_str!r}; "
                    "a new search space needs a new registration."
                )
        return {"family_id_str": registration_row_dict["family_id_str"]}

    return ledger.append(
        "trial",
        {
            "registration_id_str": registration_id_str,
            "config_dict": config_dict,
            "per_period_sharpe_float": float(per_period_sharpe_float),
            "observation_count_int": int(observation_count_int),
            "window_str": window_str,
            "station_str": station_str,
            "mode_str": mode_str,
            "code_commit_str": current_code_commit_str(),
            "data_snapshot_id_str": data_snapshot_id_str,
            "metrics_dict": metrics_dict or {},
            "return_series_path_str": return_series_path_str,
            "vault_touched_bool": bool(vault_touched_bool),
        },
        precondition_fn=precondition,
    )


@dataclass(frozen=True)
class FamilyTrialSummary:
    family_id_str: str
    recorded_trial_count_int: int
    prior_trial_count_int: int
    effective_trial_count_float: float
    # None when fewer than two trials are recorded: the family's Sharpe dispersion is unknown.
    # `deflated_sharpe_ratio` then falls back to the null sampling variance 1 / (T − 1).
    trial_sharpe_variance_float: float | None


def family_trial_summary(ledger: Ledger, family_id_str: str, trial_return_df: pd.DataFrame | None = None) -> FamilyTrialSummary:
    """Inputs for the DSR of any selection made inside this family.

    Without `trial_return_df` every recorded trial counts as independent (the
    conservative fallback). With it (one column per recorded trial, in ledger
    order), N_eff comes from the correlation clusters of their returns.
    """
    trial_row_list = [row_dict for row_dict in ledger.rows("trial") if row_dict["family_id_str"] == family_id_str]
    prior_trial_count_int = sum(
        int(row_dict.get("prior_trials_int") or 0)
        for row_dict in registration_rows(ledger).values()
        if row_dict["family_id_str"] == family_id_str
    )
    sharpe_vec = np.array([row_dict["per_period_sharpe_float"] for row_dict in trial_row_list], dtype=float)
    recorded_trial_count_int = int(sharpe_vec.size)

    if trial_return_df is not None:
        if trial_return_df.shape[1] != recorded_trial_count_int:
            raise ValueError("trial_return_df must have one column per recorded trial of the family.")
        recorded_effective_float = effective_trial_count(trial_return_df)
    else:
        recorded_effective_float = float(recorded_trial_count_int)

    return FamilyTrialSummary(
        family_id_str=family_id_str,
        recorded_trial_count_int=recorded_trial_count_int,
        prior_trial_count_int=prior_trial_count_int,
        effective_trial_count_float=max(1.0, recorded_effective_float + prior_trial_count_int),
        trial_sharpe_variance_float=float(sharpe_vec.var(ddof=1)) if recorded_trial_count_int >= 2 else None,
    )
