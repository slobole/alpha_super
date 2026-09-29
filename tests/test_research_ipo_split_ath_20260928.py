"""Unit tests for the IPO / split all-time-high study (PREREG section 7)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

STUDY_PATH = Path(__file__).resolve().parents[1] / "scripts" / "research" / "ipo_split_ath_20260928"
if str(STUDY_PATH) not in sys.path:
    sys.path.insert(0, str(STUDY_PATH))

import features  # noqa: E402
import simulate  # noqa: E402

CALENDAR_IDX = pd.bdate_range("2000-01-03", periods=400)


def _bars_df(close_list, start_int=0, unadjusted_list=None, turnover_list=None, dividend_list=None,
             open_list=None, high_list=None, low_list=None) -> pd.DataFrame:
    n_int = len(close_list)
    close_arr = np.asarray(close_list, dtype=float)
    index = CALENDAR_IDX[start_int:start_int + n_int]
    return pd.DataFrame({
        "Open": np.asarray(open_list if open_list is not None else close_arr, dtype=float),
        "High": np.asarray(high_list if high_list is not None else close_arr * 1.01, dtype=float),
        "Low": np.asarray(low_list if low_list is not None else close_arr * 0.99, dtype=float),
        "Close": close_arr,
        "Volume": np.full(n_int, 1000.0),
        "Turnover": np.asarray(turnover_list if turnover_list is not None else np.full(n_int, 1e6), dtype=float),
        "Unadjusted Close": np.asarray(unadjusted_list if unadjusted_list is not None else close_arr, dtype=float),
        "Dividend": np.asarray(dividend_list if dividend_list is not None else np.zeros(n_int), dtype=float),
    }, index=index)


# ---------------------------------------------------------------------------------------------------------------------
# split detector
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("ratio_float,expected_bool", [
    (2.0, True), (1.5, True), (1.25, True), (4 / 3, True), (3.0, True), (7.0, True), (20.0, True), (50.0, True),
    (2.004, True), (2.02, False), (0.1, False), (0.5, False), (1.07, False), (1.0, False), (1.2, False),
    (np.nan, False), (51.0, False),
])
def test_split_ratio(ratio_float, expected_bool):
    assert features.is_forward_split_ratio_bool(ratio_float) is expected_bool


def test_split_detection_and_window():
    # 2:1 split on bar 5 (adjusted history: before the split k = 2, after k = 1); reverse 1:10 on bar 150;
    # a spin-off style factor change of 1.07 on bar 200 and a special dividend 1.03 on bar 250.
    n_int = 300
    close_arr = np.linspace(10, 20, n_int)
    k_arr = np.ones(n_int)
    k_arr[:5] = 2.0 * 0.1 * 1.07 * 1.03
    k_arr[5:150] = 0.1 * 1.07 * 1.03
    k_arr[150:200] = 1.07 * 1.03
    k_arr[200:250] = 1.03
    bars_df = _bars_df(close_arr, unadjusted_list=close_arr * k_arr)
    feat_df = features.symbol_feature_df(bars_df, CALENDAR_IDX, bars_df.index[0], False)
    assert list(np.nonzero(feat_df["is_split"].to_numpy())[0]) == [5]
    assert feat_df["sessions_since_split"].iloc[5] == 0
    assert feat_df["in_split_window"].iloc[5 + 89]
    assert not feat_df["in_split_window"].iloc[5 + 90]
    assert not feat_df["in_split_window"].iloc[4]


# ---------------------------------------------------------------------------------------------------------------------
# ATH, age, ADV
# ---------------------------------------------------------------------------------------------------------------------
def test_ath_ties_and_first_bar():
    close_list = [10, 11, 11, 10.5, 12, 12, 11.9]
    bars_df = _bars_df(close_list)
    feat_df = features.symbol_feature_df(bars_df, CALENDAR_IDX, bars_df.index[0], True)
    assert feat_df["ath"].tolist() == [False, True, True, False, True, True, False]


def test_age_and_ipo_window_use_market_sessions():
    close_list = list(np.linspace(10, 30, 120))
    bars_df = _bars_df(close_list, start_int=10)
    bars_df = bars_df.drop(bars_df.index[3])  # a halted session: age still counts market sessions
    feat_df = features.symbol_feature_df(bars_df, CALENDAR_IDX, CALENDAR_IDX[10], True)
    assert feat_df["age"].iloc[0] == 0
    assert feat_df["age"].iloc[3] == 4
    assert not feat_df["in_ipo_window"].iloc[0]
    window_age_arr = feat_df.loc[feat_df["in_ipo_window"], "age"].to_numpy()
    assert window_age_arr.min() == 1 and window_age_arr.max() == 89
    not_new_df = features.symbol_feature_df(bars_df, CALENDAR_IDX, CALENDAR_IDX[10], False)
    assert not not_new_df["in_ipo_window"].any()


def test_adv_excludes_listing_bar_and_is_trailing():
    turnover_list = [1e9, 1.0, 3.0, 2.0] + [10.0] * 30
    bars_df = _bars_df(np.full(34, 10.0), turnover_list=turnover_list)
    feat_df = features.symbol_feature_df(bars_df, CALENDAR_IDX, bars_df.index[0], True)
    adv_arr = feat_df["adv"].to_numpy()
    assert np.isnan(adv_arr[0])
    assert adv_arr[1] == 1.0
    assert adv_arr[2] == 2.0  # median(1, 3)
    assert adv_arr[3] == 2.0  # median(1, 3, 2)
    assert adv_arr[33] == 10.0
    # causality: changing a future turnover leaves earlier ADV unchanged
    turnover_list[20] = 1e12
    feat2_df = features.symbol_feature_df(_bars_df(np.full(34, 10.0), turnover_list=turnover_list), CALENDAR_IDX,
                                          CALENDAR_IDX[0], True)
    assert np.allclose(feat2_df["adv"].to_numpy()[:20], adv_arr[:20], equal_nan=True)


def test_rescaling_invariance_of_flags():
    rng = np.random.default_rng(1)
    close_arr = 20 * np.exp(np.cumsum(rng.normal(0, 0.02, 300)))
    k_arr = np.where(np.arange(300) < 120, 3.0, 1.0)
    bars_df = _bars_df(close_arr, unadjusted_list=close_arr * k_arr, turnover_list=rng.uniform(1e5, 1e7, 300))
    base_df = features.symbol_feature_df(bars_df, CALENDAR_IDX, bars_df.index[0], True)
    scaled_df = bars_df.copy()
    scaled_df[["Open", "High", "Low", "Close"]] *= 0.137  # a later corporate action rescales adjusted OHLC only
    scaled_feat_df = features.symbol_feature_df(scaled_df, CALENDAR_IDX, bars_df.index[0], True)
    for col_str in ["ath", "is_split", "in_split_window", "in_ipo_window", "age"]:
        assert (base_df[col_str] == scaled_feat_df[col_str]).all()
    assert np.allclose(base_df["adv"], scaled_feat_df["adv"], equal_nan=True)


def test_forward_returns_with_dividend_and_delisting():
    open_list = [10, 11, 12, 13, 14, 15]
    close_list = [10.5, 11.5, 12.5, 13.5, 14.5, 15.5]
    dividend_list = [0, 0, 0.5, 0, 0, 0]
    bars_df = _bars_df(close_list, open_list=open_list, dividend_list=dividend_list)
    cal_pos_arr = np.arange(6)
    fwd_df = features.forward_return_df(bars_df, np.array([0, 3]), cal_pos_arr)
    # row 0, h=1: entry Open_1 = 11, exit Open_2 = 12, dividends on bar 1 only = 0
    assert fwd_df["R1"].iloc[0] == pytest.approx(12 / 11 - 1)
    # row 0, h=5 -> exit bar 6 missing: last close 15.5; dividends bars 1..5 = 0.5
    assert fwd_df["R5"].iloc[0] == pytest.approx((15.5 + 0.5) / 11 - 1)
    assert bool(fwd_df["exit_is_close5"].iloc[0])
    # row 0, h=5: dividend on bar 2 included for h >= 2
    assert fwd_df["R1"].iloc[1] == pytest.approx(15 / 14 - 1)
    assert fwd_df["CC1"].iloc[0] == pytest.approx(11.5 / 10.5 - 1)


# ---------------------------------------------------------------------------------------------------------------------
# simulator
# ---------------------------------------------------------------------------------------------------------------------
def _store_one(symbol_str, bars_df, start_int=0):
    frame_df = bars_df[["Open", "High", "Low", "Close", "Unadjusted Close", "Dividend"]].reset_index(drop=True)
    frame_df.insert(0, "cal_pos", np.arange(start_int, start_int + len(bars_df)))
    return simulate.BarStore({symbol_str: frame_df})


def _run(bars_df, config, candidate_pos_int=0, end_int=None):
    store = _store_one("AAA", bars_df)
    end_int = len(bars_df) - 1 if end_int is None else end_int
    return simulate.run_pod({candidate_pos_int: [("AAA", 1e8)]}, store, CALENDAR_IDX, 0, end_int, config)


def test_entry_sizing_and_target_fill():
    # decision at close 0 (Unadjusted 50) -> budget = 100k/20 = 5000 -> 100 shares; fills at Open_1 = 50
    bars_df = _bars_df([50, 50, 55, 70], open_list=[50, 50, 56, 65], high_list=[50, 51, 57, 75], low_list=[50, 49, 54, 64])
    config = simulate.SimConfig(slippage_float=0.0)
    out = _run(bars_df, config)
    trade = out["trade_df"].iloc[0]
    # target = 50 x 1.2 = 60; bar 3 opens at 65 >= 60 -> fills at max(Open, G) = 65 (gap up)
    assert trade["reason"] == "target"
    assert trade["exit_pos"] == 3
    assert trade["entry_cost"] == pytest.approx(100 * 50 + 1.0)
    assert trade["exit_proceeds"] == pytest.approx(100 * 65 - 1.0)


def test_trailing_stop_uses_closes_through_previous_session_and_gap_down():
    # entry at Open_1 = 100; closes 100, 110 -> H = 110 at close 2; stop for bar 3 = 99.
    # bar 3 opens at 95 (gap through) -> fills at min(Open, S) = 95. Bar 3's own high 130 must not raise the stop.
    bars_df = _bars_df([100, 100, 110, 96], open_list=[100, 100, 105, 95], high_list=[100, 101, 111, 130],
                       low_list=[100, 99.5, 104, 90], unadjusted_list=[100, 100, 110, 96])
    out = _run(bars_df, simulate.SimConfig(slippage_float=0.0, profit_target_float=0.5))
    trade = out["trade_df"].iloc[0]
    assert trade["reason"] == "stop"
    assert trade["exit_pos"] == 3
    assert trade["exit_proceeds"] == pytest.approx(50 * 95 - 1.0)


def test_both_touched_assumes_stop_and_no_exit_on_entry_session():
    # entry session 1 touches both levels -> ignored (orders start the next session). Session 2 touches both -> stop.
    bars_df = _bars_df([100, 100, 100, 100], open_list=[100, 100, 100, 100], high_list=[100, 130, 130, 100],
                       low_list=[100, 80, 80, 100])
    out = _run(bars_df, simulate.SimConfig(slippage_float=0.0))
    trade = out["trade_df"].iloc[0]
    assert trade["exit_pos"] == 2 and trade["reason"] == "stop"
    assert trade["exit_proceeds"] == pytest.approx(50 * 90 - 1.0)


def test_e2_exits_next_open_after_close_signal():
    bars_df = _bars_df([100, 100, 89, 95, 95], open_list=[100, 100, 99, 92, 95], high_list=[100, 101, 99.5, 96, 96],
                       low_list=[100, 99, 80, 91, 94])
    out = _run(bars_df, simulate.SimConfig(slippage_float=0.0, exit_mode_str="E2"))
    trade = out["trade_df"].iloc[0]
    assert trade["exit_pos"] == 3 and trade["reason"] == "e2"
    assert trade["exit_proceeds"] == pytest.approx(50 * 92 - 1.0)


def test_terminal_liquidation_and_dividend_units():
    # 2:1 split history: adjusted closes 50, Unadjusted 100 before bar 2 -> 50 nominal shares = 100 adjusted shares
    bars_df = _bars_df([50, 50, 50], unadjusted_list=[100, 100, 50], dividend_list=[0, 0, 0.4])
    out = _run(bars_df, simulate.SimConfig(slippage_float=0.0, profit_target_float=5.0, trailing_stop_float=0.9), end_int=4)
    trade = out["trade_df"].iloc[0]
    assert trade["reason"] == "terminal"
    assert trade["exit_pos"] == 3
    # entry: 50 nominal x 100 = 5000 plus $1; dividend 0.75 x 0.4 x 100 adjusted shares = 30 credited at close 2
    final_value_float = 100_000 * (1 + out["return_ser"]).prod()
    assert final_value_float == pytest.approx(100_000 - 1 - 1 + 30)


def test_slot_limit_and_rank_order():
    frame_list = {}
    for symbol_str in ["AAA", "BBB", "CCC"]:
        bars_df = _bars_df([10, 10, 10, 10])
        frame_df = bars_df[["Open", "High", "Low", "Close", "Unadjusted Close", "Dividend"]].reset_index(drop=True)
        frame_df.insert(0, "cal_pos", np.arange(4))
        frame_list[symbol_str] = frame_df
    store = simulate.BarStore(frame_list)
    candidates = {0: [("CCC", 3e8), ("AAA", 2e8), ("BBB", 1e8)]}
    out = simulate.run_pod(candidates, store, CALENDAR_IDX, 0, 3, simulate.SimConfig(slot_count_int=2, slippage_float=0.0))
    assert out["held_count_ser"].iloc[1] == 2
    # budget = min(V/N, cash/free) = 50k each -> 5000 shares; both slots used, BBB skipped
    assert out["cash_weight_ser"].iloc[1] < 0.01


def test_halt_is_padded_not_liquidated():
    # bar 3 missing (halt) between bars 2 and 4: the position is held and marked at the last close; no terminal exit.
    bars_df = _bars_df([50, 50, 52, 53, 54], open_list=[50, 50, 51, 53, 54])
    frame_df = bars_df[["Open", "High", "Low", "Close", "Unadjusted Close", "Dividend"]].reset_index(drop=True)
    frame_df.insert(0, "cal_pos", np.arange(5))
    frame_df = frame_df.drop(index=3).reset_index(drop=True)
    store = simulate.BarStore({"AAA": frame_df})
    assert store.bar("AAA", 3) == (52.0, 52.0, 52.0, 52.0, 52.0, 0.0)
    assert store.bar("AAA", 5) is None
    out = simulate.run_pod({0: [("AAA", 1e8)]}, store, CALENDAR_IDX, 0, 4, simulate.SimConfig(slippage_float=0.0))
    assert len(out["trade_df"]) == 0
    assert out["held_count_ser"].tolist() == [0, 1, 1, 1, 1]
