"""Station S1: data and causality checks for every feature a study uses (design section 9, S1).

1. Prefix invariance. The feature computed on the panel truncated at date c must equal, at date c, the feature
   computed on the full panel. Any use of rows after c (a negative shift, a centred window, a full-sample
   normalisation) breaks this.
2. Future corporate-action invariance. A split after date d rescales every earlier adjusted price. A 2:1 split is
   simulated on a sample of symbols (OHLC before d halved, Volume doubled, raw close and turnover unchanged); a
   feature declared "scale_invariant" must not change at any date before d. "level" features are reported as not
   future-split safe (WARN): their history is revised by every later split.
3. Point-in-time membership integrity (a property of the panel, run once per panel). S3 counts only member
   (date, symbol) pairs, so a leak can only come from a membership mask built with hindsight. The known form is a
   tail trim (dropping a member's last sessions before it leaves the index, which skips failures such as Lehman):
   a stock that stopped trading while it was still in the index must be a member on its last bar.

Checks 1-2 run on a random symbol sample so a full S1 pass takes seconds, not minutes. A check that compares fewer
than MIN_COMPARISON_INT finite values is not a pass: it tested nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from alpha.scout.features import Feature
from alpha.scout.panel import Panel

RELATIVE_TOLERANCE_FLOAT = 1e-12
CUTOFF_COUNT_INT = 20
MIN_COMPARISON_INT = 200


@dataclass(frozen=True)
class CausalityResult:
    feature_name_str: str
    basis_str: str
    prefix_pass_bool: bool
    prefix_detail_str: str
    split_pass_bool: bool | None  # None: not required ("level" feature)
    split_detail_str: str

    @property
    def hard_fail_bool(self) -> bool:
        return (not self.prefix_pass_bool) or (self.basis_str == "scale_invariant" and self.split_pass_bool is False)


def _compare(first_ser: pd.Series, second_ser: pd.Series) -> tuple[int, int, float]:
    """(NaN-pattern mismatches, finite comparisons, worst relative gap)."""
    first_arr, second_arr = first_ser.to_numpy(dtype=float), second_ser.to_numpy(dtype=float)
    nan_mismatch_int = int(np.sum(np.isnan(first_arr) != np.isnan(second_arr)))
    both_mask = np.isfinite(first_arr) & np.isfinite(second_arr)
    scale_arr = np.maximum(np.abs(first_arr[both_mask]), 1.0)
    worst_float = float(np.max(np.abs(first_arr[both_mask] - second_arr[both_mask]) / scale_arr)) if both_mask.any() else 0.0
    return nan_mismatch_int, int(both_mask.sum()), worst_float


def _verdict(nan_mismatch_int: int, comparison_int: int, worst_float: float) -> tuple[bool, str]:
    detail_str = f"{comparison_int} finite comparisons, NaN pattern differs on {nan_mismatch_int}, worst relative gap {worst_float:.1e}"
    if comparison_int < MIN_COMPARISON_INT:
        return False, f"not tested (fewer than {MIN_COMPARISON_INT} comparisons): {detail_str}"
    return nan_mismatch_int == 0 and worst_float <= RELATIVE_TOLERANCE_FLOAT, detail_str


def prefix_invariance(feature: Feature, panel: Panel, cutoff_count_int: int = CUTOFF_COUNT_INT, random_seed_int: int = 0) -> tuple[bool, str]:
    full_df = feature.compute_fn(panel)
    rng_obj = np.random.default_rng(random_seed_int)
    candidate_index = panel.date_index[feature.lookback_int + 5 : -5]
    cutoff_index = pd.DatetimeIndex(sorted(rng_obj.choice(candidate_index, size=min(cutoff_count_int, len(candidate_index)), replace=False)))
    comparison_int, worst_float = 0, 0.0
    for cutoff_ts in cutoff_index:
        truncated_df = feature.compute_fn(panel.truncated(cutoff_ts))
        mismatch_int, count_int, gap_float = _compare(full_df.loc[cutoff_ts], truncated_df.loc[cutoff_ts])
        if mismatch_int or gap_float > RELATIVE_TOLERANCE_FLOAT:
            return False, f"differs at cutoff {cutoff_ts.date()}: NaN pattern differs on {mismatch_int}, worst relative gap {gap_float:.1e}"
        comparison_int, worst_float = comparison_int + count_int, max(worst_float, gap_float)
    pass_bool, detail_str = _verdict(0, comparison_int, worst_float)
    return pass_bool, f"{len(cutoff_index)} cutoffs: {detail_str}"


def split_invariance(feature: Feature, panel: Panel, split_ratio_float: float = 2.0, random_seed_int: int = 0) -> tuple[bool, str]:
    rng_obj = np.random.default_rng(random_seed_int)
    split_ts = panel.date_index[int(len(panel.date_index) * 0.7)]
    split_symbol_list = list(rng_obj.choice(panel.symbol_list, size=max(1, len(panel.symbol_list) // 2), replace=False))
    before_mask = panel.date_index < split_ts
    field_dict = {}
    for field_str, frame in panel.field_dict.items():
        adjusted_df = frame.copy()
        if field_str in ("Open", "High", "Low", "Close", "Dividend"):
            adjusted_df.loc[before_mask, split_symbol_list] = frame.loc[before_mask, split_symbol_list] / split_ratio_float
        elif field_str == "Volume":
            adjusted_df.loc[before_mask, split_symbol_list] = frame.loc[before_mask, split_symbol_list] * split_ratio_float
        field_dict[field_str] = adjusted_df
    split_panel = replace(panel, field_dict=field_dict)
    original_df = feature.compute_fn(panel).loc[before_mask]
    split_df = feature.compute_fn(split_panel).loc[before_mask]
    equal_bool, detail_str = _verdict(*_compare(original_df.stack(future_stack=True), split_df.stack(future_stack=True)))
    return equal_bool, f"simulated {split_ratio_float:g}:1 split on {len(split_symbol_list)} symbols at {split_ts.date()}: {detail_str}"


def run_s1(feature_list: list[Feature], panel: Panel, sample_size_int: int = 40, random_seed_int: int = 0) -> list[CausalityResult]:
    rng_obj = np.random.default_rng(random_seed_int)
    sample_list = sorted(rng_obj.choice(panel.symbol_list, size=min(sample_size_int, len(panel.symbol_list)), replace=False))
    sample_panel = panel.subset(sample_list)
    result_list = []
    for feature in feature_list:
        prefix_pass_bool, prefix_detail_str = prefix_invariance(feature, sample_panel, random_seed_int=random_seed_int)
        split_pass_bool, split_detail_str = split_invariance(feature, sample_panel, random_seed_int=random_seed_int)
        if feature.basis_str != "scale_invariant":
            split_detail_str = "level feature, not future-split safe (WARN): " + split_detail_str
            split_pass_bool = None
        result_list.append(
            CausalityResult(feature.name_str, feature.basis_str, prefix_pass_bool, prefix_detail_str, split_pass_bool, split_detail_str)
        )
    return result_list


def non_member_share(event_mask_df: pd.DataFrame, panel: Panel) -> float:
    """Share of events on sessions where the symbol was not an index member (S3 drops them; reported, not a check)."""
    event_df = event_mask_df.fillna(False).astype(bool)
    member_df = panel.member_df.reindex(index=event_df.index, columns=event_df.columns).fillna(0)
    event_count_int = int(event_df.to_numpy().sum())
    return int((event_df & (member_df != 1)).to_numpy().sum()) / event_count_int if event_count_int else float("nan")


def membership_integrity(panel: Panel, recent_window_int: int = 20, max_trimmed_share_float: float = 0.2) -> dict:
    """Stocks that stopped trading while in the index must be members on their last bar (no tail trim).

    Looks at symbols whose last bar is before the panel's last date and that were members at some point in their
    last `recent_window_int` bars (they left by delisting or acquisition, or shortly before). Among those, the
    share whose membership ends before their last bar is the trimmed share; a tail-trimmed mask puts it near 1.
    """
    close_df, member_df = panel.field("Close"), panel.member_df == 1
    last_date_ts = panel.date_index[-1]
    checked_int, trimmed_int, example_list = 0, 0, []
    for symbol_str in panel.symbol_list:
        valid_index = close_df.index[close_df[symbol_str].notna()]
        if valid_index.empty or valid_index[-1] >= last_date_ts:
            continue
        tail_member_ser = member_df.loc[valid_index[-recent_window_int:], symbol_str]
        if not tail_member_ser.any():
            continue
        checked_int += 1
        if not tail_member_ser.iloc[-1]:
            trimmed_int += 1
            if len(example_list) < 5:
                example_list.append(f"{symbol_str} (last bar {valid_index[-1].date()}, last member {tail_member_ser[tail_member_ser].index[-1].date()})")
    trimmed_share_float = trimmed_int / checked_int if checked_int else float("nan")
    return {
        "checked_int": checked_int,
        "trimmed_int": trimmed_int,
        "trimmed_share_float": trimmed_share_float,
        "pass_bool": bool(checked_int >= 5 and trimmed_share_float <= max_trimmed_share_float),
        "example_list": example_list,
    }
