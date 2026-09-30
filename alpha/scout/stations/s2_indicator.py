"""Station S2: indicator quality (Masters, Statistically Sound Indicators; diagnostic, WARN only).

- Stability over time: quantiles of the indicator inside the eligible regime per 5-year block, and threshold drift
  (the share of eligible observations the registered threshold selects in each block). A threshold that selects 5%
  of observations in one era and 20% in another is not the same rule in both.
- Novelty: Spearman correlation with known features on a random sample of eligible observations. Above 0.8 in
  absolute value: WARN "this is probably <feature> in disguise".
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


def run_s2(
    indicator_df: pd.DataFrame,
    eligible_mask_df: pd.DataFrame,
    threshold_fn,
    panel: Panel,
    reference_feature_list: list[Feature],
    sample_size_int: int = 200_000,
    random_seed_int: int = 0,
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
    return {
        "blocks": block_row_list,
        "threshold_share_range_float": float(share_vec.max() - share_vec.min()) if share_vec.size else float("nan"),
        "novelty": novelty_row_list,
        "warn_list": [f"looks like {row['feature_str']} (Spearman {row['spearman_float']:.2f})" for row in novelty_row_list if row["warn_bool"]],
    }
