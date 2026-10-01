"""Station S2: indicator quality (Masters, Statistically Sound Indicators; diagnostic, WARN only).

- Stability over time: quantiles of the indicator inside the eligible regime per 5-year block, and threshold drift
  (the share of eligible observations the registered threshold selects in each block). A threshold that selects 5%
  of observations in one era and 20% in another is not the same rule in both.
- Novelty: Spearman correlation with known features on a random sample of eligible observations. Above 0.8 in
  absolute value: WARN "this is probably <feature> in disguise".
- Masters' per-indicator battery (P4b; Masters, Statistically Sound Indicators, ch. 2-3), on eligible values:
  - tails: (max − min) / IQR, and the share of values beyond 3 IQR from the quartiles (WARN above 1%): a few
    extreme values compress everything else into a narrow band;
  - relative entropy: the entropy of a 20-bin histogram over [min, max] divided by log(20) (WARN below 0.5): an
    indicator that sits in a few bins carries little usable resolution;
  - mutual information (bits) between indicator deciles and forward-excess deciles, against the same statistic
    with the forward excess shuffled within each date (20 shuffles; WARN when not above the shuffles' maximum):
    it catches non-linear information that a decile table can miss;
  - a mean break: the sup-Wald statistic of a single break in the monthly median of the indicator, splits in the
    middle 70%, Newey-West variance with Andrews' (1991) AR(1) automatic lag (Andrews 1993 5% critical value 8.68;
    WARN above; P4b simulation: 0-3.8% false warnings on break-free AR(1) series with phi 0 to 0.95, 96% power for
    a half-standard-deviation shift at mid-sample): the indicator's level moved, so a fixed threshold means
    different things in different eras.
  Threshold optimisation is deliberately NOT here: choosing a threshold is a search, so it belongs to S4/S5 where
  every trial is counted.
All of S2 is diagnostic (WARN only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from alpha.scout.features import Feature
from alpha.scout.panel import Panel

BLOCK_TUPLE = (("1998-2002", "1998", "2002"), ("2003-2007", "2003", "2007"), ("2008-2012", "2008", "2012"),
               ("2013-2017", "2013", "2017"), ("2018-2022", "2018", "2022"))
NOVELTY_WARN_FLOAT = 0.8
TAIL_SHARE_WARN_FLOAT = 0.01
ENTROPY_WARN_FLOAT = 0.5
ANDREWS_SUP_WALD_5PCT_FLOAT = 8.68  # one parameter, 15% trimming (Andrews 1993, table 1)


def _entropy_ratio(value_vec: np.ndarray, bin_count_int: int = 20) -> float:
    count_vec, _ = np.histogram(value_vec, bins=bin_count_int)
    share_vec = count_vec[count_vec > 0] / count_vec.sum()
    return float(-(share_vec * np.log(share_vec)).sum() / np.log(bin_count_int))


def _decile_codes(value_vec: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.quantile(value_vec, np.linspace(0.1, 0.9, 9)), value_vec, side="right")


def _mutual_information_bits(first_code_vec: np.ndarray, second_code_vec: np.ndarray) -> float:
    joint_mat = np.zeros((10, 10))
    np.add.at(joint_mat, (first_code_vec, second_code_vec), 1.0)
    joint_mat /= joint_mat.sum()
    outer_mat = joint_mat.sum(axis=1, keepdims=True) @ joint_mat.sum(axis=0, keepdims=True)
    mask_mat = joint_mat > 0
    return float((joint_mat[mask_mat] * np.log2(joint_mat[mask_mat] / outer_mat[mask_mat])).sum())


def sup_wald_break(series_vec, trim_float: float = 0.15) -> dict:
    """Largest squared t of a mean shift over every split in the middle of the series (HAC variance under the null)."""
    from alpha.stats.newey_west import newey_west_mean_t_stat

    value_vec = np.asarray(series_vec, dtype=float)
    value_vec = value_vec[np.isfinite(value_vec)]
    count_int = value_vec.size
    if count_int < 40:
        return {"sup_wald_float": float("nan"), "break_index_int": -1, "warn_bool": False}
    # Andrews (1991) AR(1) plug-in bandwidth for the Bartlett kernel: persistent series get a long lag (a fixed
    # 4(n/100)^(2/9) rule warned on 51% of break-free AR(1) series with phi = 0.9, review 2026-10-01).
    demeaned_vec = value_vec - value_vec.mean()
    rho_float = float(np.clip(demeaned_vec[1:] @ demeaned_vec[:-1] / (demeaned_vec[:-1] @ demeaned_vec[:-1]), -0.97, 0.97))
    alpha_float = 4 * rho_float**2 / ((1 - rho_float) ** 2 * (1 + rho_float) ** 2)
    lag_int = int(min(np.floor(1.1447 * (alpha_float * count_int) ** (1 / 3)), count_int // 4))
    long_run_variance_float = newey_west_mean_t_stat(value_vec, lag_int).standard_error_float ** 2 * count_int
    cumulative_vec = np.cumsum(value_vec)
    best_float, best_int = 0.0, -1
    for split_int in range(int(trim_float * count_int), int((1 - trim_float) * count_int)):
        first_mean_float = cumulative_vec[split_int - 1] / split_int
        second_mean_float = (cumulative_vec[-1] - cumulative_vec[split_int - 1]) / (count_int - split_int)
        wald_float = (second_mean_float - first_mean_float) ** 2 / (long_run_variance_float * (1 / split_int + 1 / (count_int - split_int)))
        if wald_float > best_float:
            best_float, best_int = wald_float, split_int
    return {"sup_wald_float": best_float, "break_index_int": best_int, "warn_bool": best_float > ANDREWS_SUP_WALD_5PCT_FLOAT}


def masters_battery(
    indicator_df: pd.DataFrame,
    eligible_mask_df: pd.DataFrame,
    forward_excess_df: pd.DataFrame | None = None,
    sample_size_int: int = 300_000,
    shuffle_count_int: int = 20,
    random_seed_int: int = 0,
) -> dict:
    masked_df = indicator_df.where(eligible_mask_df)
    value_vec = masked_df.stack().to_numpy(dtype=float)
    value_vec = value_vec[np.isfinite(value_vec)]
    q25_float, q75_float = np.quantile(value_vec, [0.25, 0.75])
    iqr_float = q75_float - q25_float
    result_dict = {
        "range_over_iqr_float": float((value_vec.max() - value_vec.min()) / iqr_float) if iqr_float > 0 else float("inf"),
        "tail_share_float": float(np.mean((value_vec < q25_float - 3 * iqr_float) | (value_vec > q75_float + 3 * iqr_float))),
        "relative_entropy_float": _entropy_ratio(value_vec),
    }
    warn_list = []
    if result_dict["tail_share_float"] > TAIL_SHARE_WARN_FLOAT:
        warn_list.append(f"heavy tails: {result_dict['tail_share_float']:.1%} of values beyond 3 IQR")
    if result_dict["relative_entropy_float"] < ENTROPY_WARN_FLOAT:
        warn_list.append(f"low relative entropy {result_dict['relative_entropy_float']:.2f}")

    monthly_median_ser = masked_df.median(axis=1).resample("ME").mean().dropna()
    break_dict = sup_wald_break(monthly_median_ser.to_numpy())
    result_dict["mean_break"] = {
        "sup_wald_float": break_dict["sup_wald_float"],
        "break_month_str": str(monthly_median_ser.index[break_dict["break_index_int"]].date()) if break_dict["break_index_int"] >= 0 else "",
        "critical_5pct_float": ANDREWS_SUP_WALD_5PCT_FLOAT,
    }
    if break_dict["warn_bool"]:
        warn_list.append(f"level break around {result_dict['mean_break']['break_month_str']} (sup-Wald {break_dict['sup_wald_float']:.1f})")

    if forward_excess_df is not None:
        paired_df = pd.DataFrame({"indicator": masked_df.stack(), "excess": forward_excess_df.where(eligible_mask_df).stack()}).dropna()
        rng_obj = np.random.default_rng(random_seed_int)
        if len(paired_df) > sample_size_int:
            paired_df = paired_df.iloc[np.sort(rng_obj.choice(len(paired_df), sample_size_int, replace=False))]
        indicator_code_vec = _decile_codes(paired_df["indicator"].to_numpy())
        excess_vec = paired_df["excess"].to_numpy()
        date_code_vec = pd.factorize(paired_df.index.get_level_values(0))[0]
        observed_float = _mutual_information_bits(indicator_code_vec, _decile_codes(excess_vec))
        shuffled_list = []
        for _ in range(shuffle_count_int):
            # Within-date shuffle: sort by (date, random key) and take the excess in that order within each date.
            order_vec = np.lexsort((rng_obj.random(excess_vec.size), date_code_vec))
            shuffled_vec = np.empty_like(excess_vec)
            shuffled_vec[np.lexsort((np.arange(excess_vec.size), date_code_vec))] = excess_vec[order_vec]
            shuffled_list.append(_mutual_information_bits(indicator_code_vec, _decile_codes(shuffled_vec)))
        result_dict["mutual_information"] = {
            "bits_float": observed_float,
            "shuffled_median_float": float(np.median(shuffled_list)),
            "shuffled_max_float": float(np.max(shuffled_list)),
            "pairs_int": int(excess_vec.size),
        }
        if observed_float <= max(shuffled_list):
            warn_list.append("no mutual information with the forward excess beyond within-date shuffles")
    result_dict["warn_list"] = warn_list
    return result_dict


def run_s2(
    indicator_df: pd.DataFrame,
    eligible_mask_df: pd.DataFrame,
    threshold_fn,
    panel: Panel,
    reference_feature_list: list[Feature],
    sample_size_int: int = 200_000,
    random_seed_int: int = 0,
    forward_excess_df: pd.DataFrame | None = None,
) -> dict:
    masked_df = indicator_df.where(eligible_mask_df)
    block_row_list = []
    for label_str, start_str, end_str in BLOCK_TUPLE:
        block_vec = masked_df.loc[start_str:end_str].stack().to_numpy()
        if block_vec.size == 0:
            continue
        quantile_vec = np.quantile(block_vec, [0.05, 0.25, 0.5, 0.75, 0.95])
        block_row_list.append(
            {
                "block_str": label_str,
                "observations_int": int(block_vec.size),
                "q05_float": quantile_vec[0], "q25_float": quantile_vec[1], "median_float": quantile_vec[2],
                "q75_float": quantile_vec[3], "q95_float": quantile_vec[4],
                "threshold_selects_share_float": float(np.mean(threshold_fn(block_vec))),
            }
        )

    stacked_ser = masked_df.stack()
    rng_obj = np.random.default_rng(random_seed_int)
    sample_index = stacked_ser.index[rng_obj.choice(stacked_ser.size, size=min(sample_size_int, stacked_ser.size), replace=False)]
    novelty_row_list = []
    for feature in reference_feature_list:
        reference_df = feature.compute_fn(panel)
        reference_ser = reference_df.stack().reindex(sample_index)
        paired_df = pd.DataFrame({"indicator": stacked_ser.reindex(sample_index), "reference": reference_ser}).dropna()
        rho_float = float(stats.spearmanr(paired_df["indicator"], paired_df["reference"]).statistic) if len(paired_df) > 100 else float("nan")
        novelty_row_list.append({"feature_str": feature.name_str, "spearman_float": rho_float, "warn_bool": abs(rho_float) > NOVELTY_WARN_FLOAT})

    share_vec = np.array([row["threshold_selects_share_float"] for row in block_row_list])
    battery_dict = masters_battery(indicator_df, eligible_mask_df, forward_excess_df, random_seed_int=random_seed_int)
    return {
        "blocks": block_row_list,
        "threshold_share_range_float": float(share_vec.max() - share_vec.min()) if share_vec.size else float("nan"),
        "novelty": novelty_row_list,
        "masters": battery_dict,
        "warn_list": [f"looks like {row['feature_str']} (Spearman {row['spearman_float']:.2f})" for row in novelty_row_list if row["warn_bool"]]
        + battery_dict["warn_list"],
    }
