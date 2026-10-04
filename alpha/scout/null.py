"""The per-asset permutation null for point-in-time panels (P4b; the S5 MCPT on stock families).

A permuted panel keeps every structural fact of the real one and destroys only the order of each stock's bars:
- every symbol keeps its listed dates (its NaN pattern) and its first bar, which anchors the price level;
- its later bars move among its own listed dates, following one global date permutation wherever they can
  (`alpha.stats.mcpt.live_span_source_index_mat`), so stocks live on the same dates move together;
- bars stay in their membership state: member-period bars move among member dates, the others among non-member
  dates (a stock's calmer index years are not mixed with its volatile pre-inclusion years);
- a bar moves whole (Masters' bar permutation): the gap from the previous close log(O/C_prev) and the intraday
  shape log(H/O), log(L/O), log(C/O), with its Dividend as a fraction of the close;
- the membership mask, Volume, Turnover and the Unadjusted/adjusted Close ratio stay on their real dates:
  liquidity is structural, like membership (moving it with the bars scrambled eras: 1998 volumes landed in 2008,
  and the real-vs-null Spearman of the 63-day turnover rank fell to 0.5-0.6 in review).
- Scores on these panels must be the Sharpe of the daily active return over the baseline (see alpha.stats.mcpt).

Prices are rebuilt by chaining the moved bars from the anchor, so every indicator a search computes (DV2's High/Low,
NATR, rolling windows) sees a coherent history with no temporal structure beyond each bar's own shape.

*** CRITICAL*** Permuted panels exist only inside the MCPT. Nothing computed from them may be written back as a
signal, feature or parameter.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pandas as pd

from alpha.scout.panel import Panel
from alpha.stats.mcpt import McptResult, live_span_source_index_mat, permute_live_spans
from alpha.stats.permutation import permutation_p_value


def _relative_bar_mat_dict(panel: Panel) -> tuple[dict, np.ndarray, np.ndarray]:
    """Per-bar relative quantities on every listed, non-anchor row (NaN elsewhere), the movable-row mask, and the
    close matrix with each symbol's anchor close."""
    close_mat = panel.field("Close").to_numpy(dtype=float)
    open_mat = np.where(np.isfinite(panel.field("Open").to_numpy(dtype=float)), panel.field("Open").to_numpy(dtype=float), close_mat)
    high_mat = np.fmax(np.where(np.isfinite(panel.field("High").to_numpy(dtype=float)), panel.field("High").to_numpy(dtype=float), close_mat), np.fmax(open_mat, close_mat))
    low_mat = np.fmin(np.where(np.isfinite(panel.field("Low").to_numpy(dtype=float)), panel.field("Low").to_numpy(dtype=float), close_mat), np.fmin(open_mat, close_mat))
    listed_mask_mat = np.isfinite(close_mat) & (close_mat > 0)

    previous_close_mat = np.full_like(close_mat, np.nan)
    movable_mask_mat = np.zeros_like(listed_mask_mat)
    for column_int in range(close_mat.shape[1]):
        row_vec = np.flatnonzero(listed_mask_mat[:, column_int])
        if row_vec.size > 1:
            previous_close_mat[row_vec[1:], column_int] = close_mat[row_vec[:-1], column_int]
            movable_mask_mat[row_vec[1:], column_int] = True

    with np.errstate(divide="ignore", invalid="ignore"):
        bar_dict = {
            "gap": np.log(open_mat / previous_close_mat),
            "high": np.log(high_mat / open_mat),
            "low": np.log(low_mat / open_mat),
            "body": np.log(close_mat / open_mat),
            "dividend_ratio": panel.field("Dividend").to_numpy(dtype=float) / close_mat,
        }
    bar_dict = {name_str: np.where(movable_mask_mat, value_mat, np.nan) for name_str, value_mat in bar_dict.items()}
    return bar_dict, movable_mask_mat, close_mat


def permuted_panel(panel: Panel, rng_obj: np.random.Generator, _cache: dict | None = None) -> Panel:
    """One draw of the per-asset null. `_cache` (a dict the caller keeps) avoids recomputing the relative bars."""
    if _cache is not None and "bar_dict" in _cache:
        bar_dict, movable_mask_mat, close_mat = _cache["bar_dict"], _cache["movable_mask_mat"], _cache["close_mat"]
    else:
        bar_dict, movable_mask_mat, close_mat = _relative_bar_mat_dict(panel)
        if _cache is not None:
            _cache.update(bar_dict=bar_dict, movable_mask_mat=movable_mask_mat, close_mat=close_mat)

    source_index_mat = live_span_source_index_mat(movable_mask_mat, rng_obj, (panel.member_df == 1).to_numpy())
    moved_dict = {name_str: permute_live_spans(value_mat, source_index_mat) for name_str, value_mat in bar_dict.items()}

    # Chain the moved bars from each symbol's anchor close: log C_new = log C_anchor + cumulative (gap + body).
    step_mat = np.where(movable_mask_mat, np.nan_to_num(moved_dict["gap"]) + np.nan_to_num(moved_dict["body"]), 0.0)
    anchor_mask_mat = np.isfinite(close_mat) & (close_mat > 0) & ~movable_mask_mat
    log_anchor_mat = np.where(anchor_mask_mat, np.log(np.where(anchor_mask_mat, close_mat, 1.0)), 0.0)
    new_log_close_mat = np.cumsum(log_anchor_mat + step_mat, axis=0)
    # A symbol whose listing restarts after a gap keeps chaining; cells outside the listed rows are NaN.
    listed_mask_mat = anchor_mask_mat | movable_mask_mat
    new_close_mat = np.where(listed_mask_mat, np.exp(new_log_close_mat), np.nan)
    previous_new_close_mat = np.where(movable_mask_mat, np.exp(new_log_close_mat - step_mat), np.nan)
    new_open_mat = np.where(movable_mask_mat, previous_new_close_mat * np.exp(np.nan_to_num(moved_dict["gap"])), new_close_mat)

    original_open_mat, original_high_mat, original_low_mat = (panel.field(name_str).to_numpy(dtype=float) for name_str in ("Open", "High", "Low"))
    # Rebuilt High/Low bracket Open and Close exactly (float error could otherwise leave High 1e-14 below Close).
    new_high_mat = np.where(movable_mask_mat, np.fmax(new_open_mat * np.exp(np.nan_to_num(moved_dict["high"])), np.fmax(new_open_mat, new_close_mat)), original_high_mat)
    new_low_mat = np.where(movable_mask_mat, np.fmin(new_open_mat * np.exp(np.nan_to_num(moved_dict["low"])), np.fmin(new_open_mat, new_close_mat)), original_low_mat)
    new_open_mat = np.where(movable_mask_mat, new_open_mat, original_open_mat)

    with np.errstate(divide="ignore", invalid="ignore"):
        unadjusted_ratio_mat = panel.field("Unadjusted Close").to_numpy(dtype=float) / close_mat
    field_array_dict = {
        "Open": new_open_mat,
        "High": new_high_mat,
        "Low": new_low_mat,
        "Close": new_close_mat,
        "Volume": panel.field("Volume").to_numpy(dtype=float),
        "Turnover": panel.field("Turnover").to_numpy(dtype=float),
        "Unadjusted Close": unadjusted_ratio_mat * new_close_mat,
        "Dividend": np.where(movable_mask_mat, moved_dict["dividend_ratio"] * new_close_mat, panel.field("Dividend").to_numpy(dtype=float)),
    }
    field_dict = {
        name_str: pd.DataFrame(value_mat, index=panel.date_index, columns=panel.symbol_list)
        for name_str, value_mat in field_array_dict.items()
    }
    return replace(panel, field_dict=field_dict, snapshot_id_str=panel.snapshot_id_str + "|permuted")


def mcpt_panel(
    search_fn: Callable[[Panel], float],
    panel: Panel,
    permutation_count_int: int,
    random_seed_int: int,
) -> McptResult:
    """MCPT of a whole search on a point-in-time panel under the per-asset null (S5 gate for stock families).

    `search_fn(panel)` must run the FULL search (every configuration of the family's grid union, plateau selection)
    and return the selected result's Sharpe of the daily active return over the registered baseline on the same
    panel (not a difference of two Sharpe ratios; see alpha.stats.mcpt).
    """
    if permutation_count_int < 1:
        raise ValueError("permutation_count_int must be >= 1.")
    observed_score_float = float(search_fn(panel))
    rng_obj = np.random.default_rng(int(random_seed_int))
    cache_dict: dict = {}
    null_score_vec = np.empty(permutation_count_int)
    for permutation_idx_int in range(permutation_count_int):
        null_score_vec[permutation_idx_int] = float(search_fn(permuted_panel(panel, rng_obj, cache_dict)))
    return McptResult(
        observed_score_float=observed_score_float,
        null_score_vec=null_score_vec,
        p_value_float=permutation_p_value(observed_score_float, null_score_vec, "greater"),
        permutation_count_int=int(permutation_count_int),
        stratified_bool=False,
    )
