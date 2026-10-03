"""Limit-price variants for the DV2 limit-anchor study (registration dv2_limit_anchor_20261003).

The book, the fill rules, the slot logic and the costs are those of
`scripts/research/scout_dv2_limit_entry_20261002/limit_book.py` (imported, unchanged). Only the ENTRY LIMIT PRICE of an
order decided at Close_T and worked on session T+1 changes. Everything here builds that price as a (date x symbol) matrix
whose row T is the limit for session T+1, the form `limit_book(entry_limit=...)` reads.

Offset measures (fractions of price; row T reads rows <= T only)
    natr_fraction        NATR14_T / 100 (the replica's NATR is in percent)
    close_return_std     std (ddof 1) of the 21 close-to-close returns C_t / C_(t-1) - 1 for t = T-20 .. T
    open_excursion       per bar: (Open_t - Low_t) / Open_t, how far the session fell below its open (>= 0)
    close_excursion      per bar: (Close_(t-1) - Low_t) / Close_(t-1), how far it fell below the previous close (can be < 0)
    rolling_mean         mean over rows T-20 .. T (a full window of finite values, else NaN)
    ExcursionQuantile    the stock's own (1 - p) quantile (numpy "linear") of its last 63 excursions, rows T-62 .. T (a full
                         window, else NaN), floored at 0: the drop below the anchor it exceeded with probability p. The
                         sorted windows are built once, only on the cells that can carry an order (the DV2 events), so a
                         calibration over p costs one interpolation per step.

Limit prices
    close anchor   L_T = Close_T x (1 - offset_T), rounded DOWN to the nominal tick (limit_book.entry_limit_mat).
    open anchor    the order is placed right after the opening print of T+1:
                   L_T = min(floor_tick(nominal Open_(T+1) x (1 - offset_T)), nominal Open_(T+1) - one tick), in adjusted
                   units with the adjustment ratio of row T. L < Open by construction, so limit_book's "Open <= L" branch
                   never fires: it fills only by trade-through, Low_(T+1) <= L x (1 - m_T), at L, passive (no spread).

*** CRITICAL*** the open-anchored row T is the ONLY place a T+1 value enters an order price, and only the open print,
which is known when the order is placed. The decision (which names, how many slots) is made at Close_T by the book from
rows <= T. Fills read the bar of T+1 only (limit_book). Calibration (`book_fill_rate`) runs the book on rows dated <= the
calibration end only; the order decided at the last calibration row is never worked, so its open-anchored price (which
reads the first bar after the end) cannot affect the calibrated parameter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from scripts.research.scout_dv2_limit_entry_20261002.limit_book import (
    clean_nominal_mat,
    entry_limit_mat,
    limit_book,
    nominal_tick_mat,
)

STD_WINDOW_INT = 21
MEAN_WINDOW_INT = 21
QUANTILE_WINDOW_INT = 63


# ---------------------------------------------------------------- offset measures (row T reads rows <= T)
def natr_fraction_mat(natr_percent_mat: np.ndarray) -> np.ndarray:
    return np.asarray(natr_percent_mat, dtype=float) / 100.0


def close_return_std_mat(close_mat: np.ndarray, window_int: int = STD_WINDOW_INT) -> np.ndarray:
    """Std (ddof 1) of the last `window_int` close-to-close returns, rows T-window+1 .. T (needs Close_(T-window))."""
    close_df = pd.DataFrame(np.asarray(close_mat, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        return_df = close_df / close_df.shift(1) - 1.0
    return return_df.rolling(window_int, min_periods=window_int).std(ddof=1).to_numpy()


def open_excursion_mat(open_mat: np.ndarray, low_mat: np.ndarray) -> np.ndarray:
    """(Open_t - Low_t) / Open_t per bar; NaN where a price is missing or not positive."""
    open_mat = np.asarray(open_mat, dtype=float)
    low_mat = np.asarray(low_mat, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out_mat = (open_mat - low_mat) / open_mat
    out_mat[~(np.isfinite(out_mat) & (open_mat > 0) & (low_mat > 0))] = np.nan
    return out_mat


def close_excursion_mat(close_mat: np.ndarray, low_mat: np.ndarray) -> np.ndarray:
    """(Close_(t-1) - Low_t) / Close_(t-1) per bar; row 0 and missing prices NaN."""
    close_mat = np.asarray(close_mat, dtype=float)
    low_mat = np.asarray(low_mat, dtype=float)
    out_mat = np.full(close_mat.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        out_mat[1:] = (close_mat[:-1] - low_mat[1:]) / close_mat[:-1]
        valid_mat = np.zeros(close_mat.shape, dtype=bool)
        valid_mat[1:] = (close_mat[:-1] > 0) & (low_mat[1:] > 0)
    out_mat[~(np.isfinite(out_mat) & valid_mat)] = np.nan
    return out_mat


def rolling_mean_mat(value_mat: np.ndarray, window_int: int = MEAN_WINDOW_INT) -> np.ndarray:
    """Mean over rows T-window+1 .. T; NaN unless every value in the window is finite."""
    return pd.DataFrame(np.asarray(value_mat, dtype=float)).rolling(window_int, min_periods=window_int).mean().to_numpy()


@njit(cache=False)
def _sorted_windows(value_mat, row_vec, column_vec, window_int):
    out_mat = np.full((row_vec.size, window_int), np.nan)
    for i_int in range(row_vec.size):
        r_int, c_int = row_vec[i_int], column_vec[i_int]
        if r_int < window_int - 1:
            continue
        window_vec = value_mat[r_int - window_int + 1: r_int + 1, c_int].copy()
        if np.all(np.isfinite(window_vec)):
            out_mat[i_int] = np.sort(window_vec)
    return out_mat


class ExcursionQuantile:
    """The (1 - p) quantile of each cell's last `window_int` excursions (rows T-window+1 .. T), on the cells of `mask_mat`.

    *** CRITICAL*** a cell's window ends at its own row: no bar after T."""

    def __init__(self, excursion_mat: np.ndarray, mask_mat: np.ndarray, window_int: int = QUANTILE_WINDOW_INT):
        self.shape_tuple = excursion_mat.shape
        self.row_vec, self.column_vec = np.nonzero(np.asarray(mask_mat, dtype=bool))
        self.sorted_mat = _sorted_windows(np.ascontiguousarray(excursion_mat, dtype=float), self.row_vec.astype(np.int64),
                                          self.column_vec.astype(np.int64), int(window_int))

    def quantile_vec(self, level_float: float) -> np.ndarray:
        """numpy's default ("linear") quantile at `level_float` of every stored window."""
        position_float = level_float * (self.sorted_mat.shape[1] - 1)
        low_int = int(np.floor(position_float))
        high_int = min(low_int + 1, self.sorted_mat.shape[1] - 1)
        weight_float = position_float - low_int
        return self.sorted_mat[:, low_int] * (1.0 - weight_float) + self.sorted_mat[:, high_int] * weight_float

    def offset_mat(self, probability_float: float) -> np.ndarray:
        """Offset that the stock's own excursion exceeded with probability p in its window, floored at 0; NaN off the mask."""
        out_mat = np.full(self.shape_tuple, np.nan)
        out_mat[self.row_vec, self.column_vec] = np.maximum(self.quantile_vec(1.0 - probability_float), 0.0)
        return out_mat


# ---------------------------------------------------------------- limit prices
def close_anchor_limit_mat(close_mat: np.ndarray, unadjusted_close_mat: np.ndarray, offset_fraction_mat: np.ndarray) -> np.ndarray:
    """Row T: Close_T x (1 - offset_T), rounded DOWN to the nominal tick (limit_book.entry_limit_mat with k = 1)."""
    return entry_limit_mat(close_mat, unadjusted_close_mat, 100.0 * np.asarray(offset_fraction_mat, dtype=float), 1.0)


def open_anchor_limit_mat(open_mat: np.ndarray, close_mat: np.ndarray, unadjusted_close_mat: np.ndarray,
                          offset_fraction_mat: np.ndarray) -> np.ndarray:
    """Row T: the buy limit placed right after the opening print of T+1, in adjusted units:
    min(floor_tick(nominal Open_(T+1) x (1 - offset_T)), nominal Open_(T+1) - one tick), strictly below the open.
    Nominal = adjusted x (Unadjusted Close_T / Close_T), the ratio of row T. Last row and missing inputs: NaN (no order).

    *** CRITICAL*** row T reads Open_(T+1) (the open print, known when the order is placed), Close_T, Unadjusted Close_T
    and offset_T; never High, Low or Close of T+1."""
    open_mat = np.asarray(open_mat, dtype=float)
    close_mat = np.asarray(close_mat, dtype=float)
    next_open_mat = np.full(open_mat.shape, np.nan)
    next_open_mat[:-1] = open_mat[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio_mat = clean_nominal_mat(unadjusted_close_mat) / close_mat  # nominal per adjusted unit at T
        nominal_open_mat = np.round(next_open_mat * ratio_mat, 4)
        tick_mat = nominal_tick_mat(nominal_open_mat)
        raw_nominal_mat = nominal_open_mat * (1.0 - np.asarray(offset_fraction_mat, dtype=float))
        rounded_nominal_mat = np.fmin(np.floor(raw_nominal_mat / tick_mat + 1e-6) * tick_mat, nominal_open_mat - tick_mat)
        out_mat = rounded_nominal_mat / ratio_mat
    out_mat[~np.isfinite(out_mat) | ~(rounded_nominal_mat > 0) | ~np.isfinite(raw_nominal_mat)] = np.nan
    return out_mat


# ---------------------------------------------------------------- calibration (rows <= the calibration end only)
def truncate_mats(mats: dict, row_count_int: int) -> dict:
    """The rule matrices cut to the first `row_count_int` rows (candidate CSR included)."""
    pointer_vec = mats["pointer_vec"][: row_count_int + 1]
    return {"open": mats["open"][:row_count_int], "close": mats["close"][:row_count_int], "pointer_vec": pointer_vec,
            "candidate_vec": mats["candidate_vec"][: pointer_vec[-1]], "exit_signal": mats["exit_signal"][:row_count_int]}


def book_fill_rate(date_index: pd.DatetimeIndex, symbol_list: list, mats: dict, high_mat: np.ndarray, low_mat: np.ndarray,
                   entry_limit: np.ndarray, margin_mat: np.ndarray, exit_limit: np.ndarray, max_positions_int: int, start_str: str,
                   end_str: str, exit_str: str = "limit") -> tuple[float, int]:
    """(entries / entry orders, orders) of the book run on rows dated start .. end ONLY (no cost: fills do not depend on
    costs). Every input is cut at the end row before the book sees it.

    *** CRITICAL*** nothing dated after `end_str` reaches the book, and the order decided at the last kept row is never
    worked (the book stops there)."""
    row_count_int = int(np.searchsorted(date_index.to_numpy(), np.datetime64(pd.Timestamp(end_str)), side="right"))
    cut = lambda m: None if m is None else np.asarray(m)[:row_count_int]
    shape_tuple = (row_count_int, len(symbol_list))
    result = limit_book(date_index[:row_count_int], symbol_list, truncate_mats(mats, row_count_int), cut(high_mat), cut(low_mat),
                        max_positions_int, start_str, cut(entry_limit), exit_str, cut(margin_mat), cut(exit_limit), 0.0,
                        np.ones(shape_tuple), 0.0, 0.0, 0.0)
    log_df = result.log_df[result.log_df["date"] >= pd.Timestamp(start_str)]
    order_count_int = int(log_df["kind_int"].isin([0, 1]).sum())
    entry_count_int = int((log_df["kind_int"] == 1).sum())
    return (entry_count_int / order_count_int if order_count_int else float("nan")), order_count_int


def calibrate(fill_rate_fn, target_float: float, low_float: float, high_float: float, decreasing_bool: bool,
              iteration_int: int = 16, max_expand_int: int = 6) -> dict:
    """Bisection for the parameter whose fill rate is closest to the target. `fill_rate_fn(param) -> fill rate` is
    monotone up to path noise: decreasing in a multiplier k, increasing in a probability p. For a decreasing function the
    upper bound doubles (at most `max_expand_int` times) until it brackets the target. Returns the evaluated point closest
    to the target."""
    evaluated_list = []

    def rate(param_float: float) -> float:
        value_float = float(fill_rate_fn(param_float))
        evaluated_list.append((param_float, value_float))
        return value_float

    above = lambda value_float: value_float > target_float if decreasing_bool else value_float < target_float
    rate(low_float)
    high_rate_float = rate(high_float)
    expand_int = 0
    while decreasing_bool and above(high_rate_float) and expand_int < max_expand_int:
        low_float, high_float = high_float, 2.0 * high_float
        high_rate_float = rate(high_float)
        expand_int += 1
    for _ in range(iteration_int):
        middle_float = 0.5 * (low_float + high_float)
        if above(rate(middle_float)):
            low_float = middle_float
        else:
            high_float = middle_float
    best_param_float, best_rate_float = min(evaluated_list, key=lambda pair: (abs(pair[1] - target_float), pair[0]))
    return {"param_float": best_param_float, "fill_rate_float": best_rate_float, "target_float": target_float,
            "evaluations_int": len(evaluated_list)}
