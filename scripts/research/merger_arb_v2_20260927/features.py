"""Event and pin features for merger arbitrage v2 (research only). One code path for streaming (S = 1) and panels.

Event at session d (data through d; PIT Russell 3000 member at d):
    jump_d         = Close_d / Close_{d-1} - 1 >= J
    volume shock   = Turnover_d >= 5 x median(Turnover_{d-60..d-1})          (60 valid positive values)
    liquidity      = ADV20_{d-1} = median(Turnover_{d-20..d-1}) >= q25_d,  q25_d = 25th percentile of ADV20_{d-1}
                     among the Russell 3000 members at d (computed exactly from the streaming pass)
Pin window s = d+1..d+W, r_s = Close_s / Close_{s-1} - 1:
    every Turnover_s > 0;  median |r_s| <= theta;  max |r_s| <= 3%;  min Close_s >= 0.97 x Close_d
    pin_ref_d = median Close_s  (fixed for the position's life);  pin_stat_d = median |r_s| (queue tie-break)
Confirmation at t = d + W; the decision row carries pin_stat and pin_ref.
Every feature is a ratio of prices of one symbol or uses native Turnover, so it is invariant to a constant rescaling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from merger_arb_v2_20260927.cells import HOLD_FLOOR_FLOAT, MAX_MOVE_FLOAT, VOLUME_MEDIAN_WINDOW_INT, VOLUME_SHOCK_MULT_FLOAT
from trend_breakout_20260927.features import DailyFeatureBook

ADV_WINDOW_INT = 20


def clean_turnover(turnover_arr: np.ndarray) -> np.ndarray:
    turnover_arr = turnover_arr.astype(np.float64)
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(turnover_arr) & (turnover_arr > 0), turnover_arr, np.nan)


def _rolling_median_prev(value_arr: np.ndarray, window_int: int) -> np.ndarray:
    """Trailing median over the window ending at d-1 (shift 1); same shape as the input (1-D streaming or 2-D panel)."""
    # *** CRITICAL *** the window ends at d-1: the event day's own value is excluded.
    out_arr = pd.DataFrame(value_arr).rolling(window=window_int, min_periods=window_int).median().shift(1).to_numpy()
    return out_arr.reshape(value_arr.shape)


def adv20_prev_arr(turnover_arr: np.ndarray) -> np.ndarray:
    """median(Turnover_{d-20..d-1}) on row d; NaN unless all 20 are valid and positive."""
    return _rolling_median_prev(clean_turnover(turnover_arr), ADV_WINDOW_INT)


def median60_prev_arr(turnover_arr: np.ndarray) -> np.ndarray:
    return _rolling_median_prev(clean_turnover(turnover_arr), VOLUME_MEDIAN_WINDOW_INT)


def jump_ret_arr(close_arr: np.ndarray) -> np.ndarray:
    out_arr = np.full(close_arr.shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out_arr[1:] = close_arr[1:] / close_arr[:-1] - 1.0
    return out_arr


def volume_shock_arr(turnover_arr: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return clean_turnover(turnover_arr) >= VOLUME_SHOCK_MULT_FLOAT * median60_prev_arr(turnover_arr)


def candidate_event_arr(close_arr: np.ndarray, turnover_arr: np.ndarray, member_arr: np.ndarray, jump_float: float) -> np.ndarray:
    """Jump + volume shock + membership (the liquidity test needs the cross-sectional quantile and is applied later)."""
    with np.errstate(invalid="ignore"):
        return (member_arr == 1) & (jump_ret_arr(close_arr) >= jump_float) & volume_shock_arr(turnover_arr)


PIN_ROW_CHUNK_INT = 256  # rows per chunk: bounds the W-stacked transient to W x 256 x S x 8 bytes (memory only)


def pin_feature_dict(close_arr: np.ndarray, turnover_arr: np.ndarray, window_int: int) -> dict[str, np.ndarray]:
    """At the event row d: median |r|, max |r|, min-close ratio, all-traded flag, pin_ref, over d+1..d+W.
    Computed in row chunks so the W-stacked transient stays small; the arithmetic per cell is unchanged."""
    import warnings

    date_count_int = close_arr.shape[0]
    ret_arr = jump_ret_arr(close_arr)
    turnover_clean_arr = clean_turnover(turnover_arr)
    median_abs_arr = np.full(close_arr.shape, np.nan)
    max_abs_arr = np.full(close_arr.shape, np.nan)
    pin_ref_arr = np.full(close_arr.shape, np.nan)
    traded_arr = np.ones(close_arr.shape, dtype=bool)
    min_close_arr = np.full(close_arr.shape, np.inf)
    for start_int in range(0, date_count_int, PIN_ROW_CHUNK_INT):
        stop_int = min(start_int + PIN_ROW_CHUNK_INT, date_count_int)
        chunk_shape = (stop_int - start_int,) + close_arr.shape[1:]
        abs_ret_list, close_list = [], []
        for k_int in range(1, window_int + 1):
            # r, close and turnover at d+k on row d (rows beyond the end of the series stay NaN)
            src_start_int, src_stop_int = start_int + k_int, min(stop_int + k_int, date_count_int)
            count_int = max(src_stop_int - src_start_int, 0)
            shifted_ret_arr = np.full(chunk_shape, np.nan)
            shifted_close_arr = np.full(chunk_shape, np.nan)
            shifted_turnover_arr = np.full(chunk_shape, np.nan)
            if count_int > 0:
                shifted_ret_arr[:count_int] = ret_arr[src_start_int:src_stop_int]
                shifted_close_arr[:count_int] = close_arr[src_start_int:src_stop_int]
                shifted_turnover_arr[:count_int] = turnover_clean_arr[src_start_int:src_stop_int]
            abs_ret_list.append(np.abs(shifted_ret_arr))
            close_list.append(shifted_close_arr)
            with np.errstate(invalid="ignore"):
                traded_arr[start_int:stop_int] &= np.isfinite(shifted_turnover_arr) & (shifted_turnover_arr > 0)
                min_close_arr[start_int:stop_int] = np.fmin(min_close_arr[start_int:stop_int], np.where(np.isfinite(shifted_close_arr), shifted_close_arr, np.inf))
        abs_stack_arr = np.stack(abs_ret_list, axis=0)
        close_stack_arr = np.stack(close_list, axis=0)
        all_finite_arr = np.isfinite(abs_stack_arr).all(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            median_abs_arr[start_int:stop_int] = np.where(all_finite_arr, np.nanmedian(abs_stack_arr, axis=0), np.nan)
            max_abs_arr[start_int:stop_int] = np.where(all_finite_arr, np.nanmax(abs_stack_arr, axis=0), np.nan)
            pin_ref_arr[start_int:stop_int] = np.where(np.isfinite(close_stack_arr).all(axis=0), np.nanmedian(close_stack_arr, axis=0), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        hold_arr = min_close_arr / close_arr >= HOLD_FLOOR_FLOAT
        pin_ok_arr = traded_arr & np.isfinite(median_abs_arr) & (max_abs_arr <= MAX_MOVE_FLOAT) & hold_arr & np.isfinite(pin_ref_arr)
    return {"median_abs": median_abs_arr, "max_abs": max_abs_arr, "hold": hold_arr, "traded": traded_arr, "pin_ref": pin_ref_arr, "pin_ok": pin_ok_arr}


def confirmation_dict(event_arr: np.ndarray, pin_dict: dict[str, np.ndarray], theta_float: float, window_int: int) -> dict:
    """Confirmation at the decision row t = d + W. Dense: the bool conf panel. Sparse (one row per confirmation, sorted
    by decision row then symbol): t_vec, symbol_vec, pin_stat_vec (median |r|), pin_ref_vec, event_pos_vec."""
    with np.errstate(invalid="ignore"):
        confirmed_at_event_arr = event_arr & pin_dict["pin_ok"] & (pin_dict["median_abs"] <= theta_float)
    conf_arr = np.zeros(event_arr.shape, dtype=bool)
    # *** CRITICAL *** an event at d is confirmed at the close of d + W and can be bought at the open of d + W + 1.
    conf_arr[window_int:] = confirmed_at_event_arr[:-window_int]
    if event_arr.ndim == 2:
        event_pos_vec, symbol_vec = np.nonzero(confirmed_at_event_arr)
    else:
        event_pos_vec = np.flatnonzero(confirmed_at_event_arr)
        symbol_vec = np.zeros(len(event_pos_vec), dtype=np.int64)
    t_vec = event_pos_vec + window_int
    keep_vec = t_vec < event_arr.shape[0]
    event_pos_vec, symbol_vec, t_vec = event_pos_vec[keep_vec], symbol_vec[keep_vec], t_vec[keep_vec]
    index_tuple = (event_pos_vec, symbol_vec) if event_arr.ndim == 2 else (event_pos_vec,)
    return {"conf": conf_arr, "t_vec": t_vec.astype(np.int64), "symbol_vec": symbol_vec.astype(np.int64), "pin_stat_vec": pin_dict["median_abs"][index_tuple].astype(np.float64),
            "pin_ref_vec": pin_dict["pin_ref"][index_tuple].astype(np.float64), "event_pos_vec": event_pos_vec.astype(np.int64), "confirmed_at_event": confirmed_at_event_arr}


class V2FeatureBook(DailyFeatureBook):
    """Panel feature book for the confirmed-symbol subset. The universe dict carries the exact daily q25 thresholds
    and the half-membership panels from the streaming pass. Float64 panels are used in place (no copies)."""

    def __init__(self, universe_dict: dict):
        super().__init__(universe_dict)
        if universe_dict["close_arr"].dtype == np.float64:
            self.close_arr = universe_dict["close_arr"]

    def _no_copy(self, key_str: str) -> np.ndarray:
        arr = self.u[key_str]
        return arr if arr.dtype == np.float64 else arr.astype(np.float64)

    def open64(self) -> np.ndarray:
        return self._no_copy("open_arr")

    def unadjusted_close(self) -> np.ndarray:
        return self._no_copy("unadjusted_close_arr")

    def dividend64(self) -> np.ndarray:
        return self._no_copy("dividend_arr")

    def turnover64(self) -> np.ndarray:
        return self._get(("turnover64",), lambda: clean_turnover(self.u["turnover_arr"]))

    def adv20_prev(self) -> np.ndarray:
        return self._get(("adv20_prev",), lambda: adv20_prev_arr(self.u["turnover_arr"]))

    def rel25_prev_pass(self) -> np.ndarray:
        def build():
            adv_arr = self.adv20_prev()
            q25_vec = self.u["q25_adv20_prev_vec"].astype(np.float64)
            with np.errstate(invalid="ignore"):
                return np.isfinite(adv_arr) & np.isfinite(q25_vec)[:, None] & (adv_arr >= q25_vec[:, None])

        return self._get(("rel25_prev",), build)

    def event(self, jump_float: float, half_str: str = "ALL") -> np.ndarray:
        def build():
            member_arr = self.u["member_arr"] if half_str == "ALL" else self.u[{"R1000": "r1000_member_arr", "R2000": "r2000_only_member_arr"}[half_str]]
            return candidate_event_arr(self.close_arr, self.u["turnover_arr"], member_arr, jump_float) & self.rel25_prev_pass()

        return self._get(("event", jump_float, half_str), build)

    def pin(self, window_int: int) -> dict[str, np.ndarray]:
        """Single-slot cache: only the most recently used W is kept (three float64 panels), the grid visits W in runs."""
        cached_tuple = getattr(self, "_pin_cache_tuple", None)
        if cached_tuple is None or cached_tuple[0] != window_int:
            cached_tuple = (window_int, pin_feature_dict(self.close_arr, self.u["turnover_arr"], window_int))
            self._pin_cache_tuple = cached_tuple
        return cached_tuple[1]

    def confirmation(self, jump_float: float, theta_float: float, window_int: int, half_str: str = "ALL") -> dict:
        """Cached sparsely: the dense bool panel (14 MB) plus one row per confirmation."""
        return self._get(("confirmation", jump_float, theta_float, window_int, half_str), lambda: confirmation_dict(self.event(jump_float, half_str), self.pin(window_int), theta_float, window_int))
