"""Shared VIX stress gate and SPMO parking weight for the MR capsule (research-only).

Research record: docs/research/MR_CAPSULE_20261003.md ("Final capsule specification"), gate studies
docs/research/MR_GATE_*_20261003.md. Both capsule pods (DV2 and HPI 2/3/5 vote) read the SAME gate.

Formulas (all evaluated after Close_t; orders that use them fill at Open_(t+1)):

    threshold_t   = mean(VIX_1990-01-02 .. VIX_t)                      (expanding mean, >= 500 closes)
    gate opens    at the first close with VIX_t > threshold_t while the gate is closed
    gate closes   at the first close with VIX_t <= threshold_t that comes at least
                  GATE_MEMORY_SESSION_INT sessions after the opening close
    spmo_weight_t = min(1, 0.08 / (std(r_(t-19..t)) * sqrt(252))),     r = SPMO close-to-close return

The research version is scripts/research/mr_gate_selfcal_20261003/run_selfcal.py (selfcal_params, gate_mem);
tests/test_strategy_mr_capsule_gate.py pins this module to it on a synthetic path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VIX_SYMBOL_STR = "$VIX"
VIX_HISTORY_START_DATE_STR = "1990-01-01"
GATE_MIN_HISTORY_SESSION_INT = 500
GATE_MEMORY_SESSION_INT = 15
SPMO_SYMBOL_STR = "SPMO"
BIL_SYMBOL_STR = "BIL"
SPMO_VOL_TARGET_FLOAT = 0.08
SPMO_VOL_WINDOW_SESSION_INT = 20
TRADING_DAY_PER_YEAR_INT = 252


def vix_threshold_ser(vix_close_ser: pd.Series, min_history_session_int: int = GATE_MIN_HISTORY_SESSION_INT) -> pd.Series:
    """Expanding mean of all VIX closes up to and including each close."""
    clean_vix_ser = pd.Series(vix_close_ser, copy=True).astype(float).sort_index().dropna()
    # *** CRITICAL*** expanding window ends at Close_t: the threshold for the decision after Close_t uses
    # VIX closes <= t only. Never centre or back-fill it.
    threshold_ser = clean_vix_ser.expanding(min_periods=min_history_session_int).mean()
    threshold_ser.name = "vix_threshold_ser"
    return threshold_ser.reindex(pd.Series(vix_close_ser).index)


def stress_gate_open_ser(
    vix_close_ser: pd.Series,
    memory_session_int: int = GATE_MEMORY_SESSION_INT,
    min_history_session_int: int = GATE_MIN_HISTORY_SESSION_INT,
) -> pd.Series:
    """Gate state after each close: True = open (pods may enter), False = closed."""
    if memory_session_int < 0:
        raise ValueError("memory_session_int must be non-negative.")
    vix_ser = pd.Series(vix_close_ser, copy=True).astype(float).sort_index()
    threshold_ser = vix_threshold_ser(vix_ser, min_history_session_int)
    vix_arr, threshold_arr = vix_ser.to_numpy(), threshold_ser.to_numpy()
    gate_arr = np.zeros(len(vix_arr), dtype=bool)
    open_bool, held_session_int = False, 0
    # *** CRITICAL*** a sequential state machine over closes <= t; the state at t never reads t+1. A missing
    # VIX or threshold carries the previous state forward (no new information, no transition).
    for row_int in range(len(vix_arr)):
        if np.isfinite(vix_arr[row_int]) and np.isfinite(threshold_arr[row_int]):
            above_bool = bool(vix_arr[row_int] > threshold_arr[row_int])
            if not open_bool and above_bool:
                open_bool, held_session_int = True, 0
            elif open_bool:
                held_session_int += 1
                if not above_bool and held_session_int >= memory_session_int:
                    open_bool = False
        gate_arr[row_int] = open_bool
    return pd.Series(gate_arr, index=vix_ser.index, name="stress_gate_open_bool")


def spmo_target_weight_ser(
    spmo_close_ser: pd.Series,
    vol_target_float: float = SPMO_VOL_TARGET_FLOAT,
    window_session_int: int = SPMO_VOL_WINDOW_SESSION_INT,
    *,
    spmo_volume_ser: pd.Series | None = None,
) -> pd.Series:
    """SPMO share of the idle-cash sleeve: min(1, vol target / 20-session realised volatility); 0 when unknown.

    Tradability guard (build amendment B1, 2026-10-04), applied when spmo_volume_ser is given: the weight is 0 unless
    SPMO traded (Volume > 0) on each of the last window_session_int sessions. SPMO had 151 sessions without a trade
    in 2016 and a median daily turnover of USD 0-2K in 2015-2017; its padded closes there are stale prices, not
    tradable quotes. From 2018 SPMO trades every session and the guard is inactive.
    """
    close_ser = pd.Series(spmo_close_ser, copy=True).astype(float).sort_index()
    # *** CRITICAL*** pct_change and the rolling std end at Close_t; the weight decided after Close_t is traded
    # at Open_(t+1). fill_method=None keeps pre-inception NaNs from becoming fake zero returns.
    realised_vol_ser = close_ser.pct_change(fill_method=None).rolling(window_session_int, min_periods=window_session_int).std() * np.sqrt(TRADING_DAY_PER_YEAR_INT)
    weight_ser = (vol_target_float / realised_vol_ser).clip(upper=1.0)
    weight_ser = weight_ser.where(np.isfinite(weight_ser) & (weight_ser > 0.0), 0.0)
    if spmo_volume_ser is not None:
        traded_ser = pd.Series(spmo_volume_ser, copy=True).astype(float).reindex(close_ser.index).fillna(0.0).gt(0.0)
        # *** CRITICAL*** the volume window also ends at Close_t: sessions <= t only.
        fully_traded_bool_ser = traded_ser.astype(float).rolling(window_session_int, min_periods=window_session_int).min().eq(1.0)
        weight_ser = weight_ser.where(fully_traded_bool_ser, 0.0)
    weight_ser.name = "spmo_target_weight_ser"
    return weight_ser


def spmo_weight_from_history(
    spmo_close_ser: pd.Series,
    spmo_volume_ser: pd.Series | None = None,
    vol_target_float: float = SPMO_VOL_TARGET_FLOAT,
    window_session_int: int = SPMO_VOL_WINDOW_SESSION_INT,
) -> float:
    """The SPMO weight decided after the LAST close of the given history (spmo_target_weight_ser on its tail).

    The pods call this inside iterate on the engine's data, which already ends at Close(previous_bar), so the weight
    cannot see later prices; it also keeps no precomputed state that a truncated signal-audit recompute could replace.
    """
    tail_int = window_session_int + 1
    close_tail_ser = pd.Series(spmo_close_ser).iloc[-tail_int:]
    if len(close_tail_ser) < tail_int:
        return 0.0
    volume_tail_ser = None if spmo_volume_ser is None else pd.Series(spmo_volume_ser).iloc[-tail_int:]
    weight_ser = spmo_target_weight_ser(close_tail_ser, vol_target_float, window_session_int, spmo_volume_ser=volume_tail_ser)
    return float(weight_ser.iloc[-1])


def load_vix_close_ser(end_date_str: str | None = None) -> pd.Series:
    """$VIX closes from 1990 (the gate's expanding mean needs the whole history, not the 1998 price window)."""
    from data.norgate_loader import load_price_timeseries

    vix_df = load_price_timeseries(VIX_SYMBOL_STR, start_date_str=VIX_HISTORY_START_DATE_STR, end_date_str=end_date_str)
    vix_close_ser = vix_df["Close"].astype(float)
    vix_close_ser.index = pd.to_datetime(vix_close_ser.index)
    vix_close_ser.name = "vix_close_ser"
    return vix_close_ser.dropna()


def gate_state_at(gate_open_ser: pd.Series, decision_date_ts: pd.Timestamp) -> bool:
    """Gate state after the decision close: the latest gate row on or before the decision date."""
    # *** CRITICAL*** never use a gate row dated after the decision close (asof, not nearest).
    position_int = int(gate_open_ser.index.searchsorted(pd.Timestamp(decision_date_ts), side="right")) - 1
    return bool(gate_open_ser.iloc[position_int]) if position_int >= 0 else False
