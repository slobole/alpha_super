"""Tests for merger arbitrage v2 (scripts/research/merger_arb_v2_20260927). Synthetic data only, except the control-book
test, which reads the stored sleeves and Norgate BIL/SPY when available."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "research"))
from merger_arb_v2_20260927 import cells as cells_module  # noqa: E402
from merger_arb_v2_20260927 import common, data as data_module  # noqa: E402
from merger_arb_v2_20260927.features import V2FeatureBook, adv20_prev_arr, candidate_event_arr, confirmation_dict, pin_feature_dict  # noqa: E402
from merger_arb_v2_20260927.policies import PolicyV  # noqa: E402
from merger_arb_v2_20260927.simulate import simulate  # noqa: E402
from trend_breakout_20260927.policies import State  # noqa: E402


def make_panel(symbol_count_int: int = 6, seed_int: int = 3) -> dict:
    rng = np.random.default_rng(seed_int)
    date_index = pd.bdate_range("1998-06-01", "2002-12-31")
    n = len(date_index)
    close_arr = 50.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, (n, symbol_count_int)), axis=0))
    open_arr = close_arr * np.exp(rng.normal(0, 0.002, close_arr.shape))
    turnover_arr = np.full((n, symbol_count_int), 5e7)
    member_arr = np.ones(close_arr.shape, dtype=np.int8)
    return {"universe_str": "TEST", "date_index": date_index, "symbol_list": [f"S{i:02d}" for i in range(symbol_count_int)], "open_arr": open_arr, "close_arr": close_arr,
            "unadjusted_close_arr": close_arr.copy(), "dividend_arr": np.zeros(close_arr.shape), "turnover_arr": turnover_arr, "member_arr": member_arr,
            "r1000_member_arr": member_arr.copy(), "r2000_only_member_arr": np.zeros(close_arr.shape, dtype=np.int8), "q25_adv20_prev_vec": np.full(n, 1e6),
            "spy_close_ser": pd.Series(100.0, index=date_index), "month_end_index": pd.DatetimeIndex(pd.Series(date_index, index=date_index.to_period("M")).groupby(level=0).max().to_numpy())}


def plant_deal(panel: dict, s: int, d: int, jump: float = 0.25, moves: tuple = (0.002, -0.001, 0.0015, -0.002, 0.001, 0.0, 0.001, -0.001, 0.002, 0.0), turnover_zero_day: int | None = None, drop_after: int | None = None) -> None:
    close, open_, to = panel["close_arr"], panel["open_arr"], panel["turnover_arr"]
    level = close[d - 1, s] * (1.0 + jump)
    close[d, s] = level
    to[d, s] = to[d - 1, s] * 8.0
    for k in range(1, 70):
        m = moves[(k - 1) % len(moves)]
        close[d + k, s] = close[d + k - 1, s] * (1.0 + m)
    if drop_after is not None:
        close[d + drop_after :, s] = close[d + drop_after - 1, s] * 0.90
    open_[d:, s] = close[d:, s]
    panel["unadjusted_close_arr"][:, s] = close[:, s]
    if turnover_zero_day is not None:
        to[turnover_zero_day, s] = 0.0


def test_median_pin_max_move_hold_and_turnover_rules():
    p = make_panel()
    d = 400
    plant_deal(p, 0, d)                                                              # clean: median |r| 0.1%, max 0.2%
    plant_deal(p, 1, d, moves=(0.002, -0.001, 0.035, -0.002, 0.001))                # one 3.5% move: max rule fails
    plant_deal(p, 2, d, moves=(-0.011, -0.011, -0.011, 0.002, 0.001))               # drifts below 0.97 x Close_d
    plant_deal(p, 3, d, turnover_zero_day=d + 2)                                    # padded / halted bar
    plant_deal(p, 4, d, moves=(0.009, -0.009, 0.009, -0.009, 0.009))                # median 0.9% > theta 0.5%
    plant_deal(p, 5, d, jump=0.05)                                                  # jump too small
    f = V2FeatureBook(p)
    cell = cells_module.V0_CELL
    conf = f.confirmation(cell.jump_float, cell.theta_float, cell.window_int)
    t = d + cell.window_int
    assert conf["conf"][t].tolist() == [True, False, False, False, False, False]
    pin = f.pin(5)
    assert pin["median_abs"][d, 0] == pytest.approx(np.median(np.abs(p["close_arr"][d + 1 : d + 6, 0] / p["close_arr"][d : d + 5, 0] - 1)))
    assert pin["max_abs"][d, 1] > cells_module.MAX_MOVE_FLOAT and not pin["pin_ok"][d, 1]
    assert not pin["hold"][d, 2] and not pin["traded"][d, 3]
    assert pin["pin_ok"][d, 4] and pin["median_abs"][d, 4] > cell.theta_float
    row_vec = np.flatnonzero((conf["t_vec"] == t) & (conf["symbol_vec"] == 0))  # sparse confirmation rows
    assert len(row_vec) == 1 and conf["pin_ref_vec"][row_vec[0]] == pytest.approx(np.median(p["close_arr"][d + 1 : d + 6, 0]))
    assert conf["event_pos_vec"][row_vec[0]] == d and conf["pin_stat_vec"][row_vec[0]] == pytest.approx(pin["median_abs"][d, 0])
    # theta 0.8% admits the 0.9%-median name? no; 1.0% would. Stage-P loosest theta keeps symbol 4 out.
    assert not f.confirmation(cell.jump_float, 0.008, 5)["conf"][t, 4]


def test_queue_drop_rules_and_order():
    p = make_panel(symbol_count_int=6)
    d = 400
    for s in range(4):
        plant_deal(p, s, d, moves=(0.001 * (s + 1), -0.001 * (s + 1)))
    f = V2FeatureBook(p)
    cell = cells_module.VCell(slots_int=2)
    policy = PolicyV(f, cell)
    state = State(6, 100_000.0)
    state.cash_float, state.total_value_float = 40_000.0, 100_000.0
    t = d + cell.window_int
    intents = policy.decide(t, state)
    entries = [i for i in intents if i.kind_str == "value"]
    assert [i.symbol_idx for i in entries] == [0, 1]  # same confirmation date -> lowest median |r| first
    assert all(i.amount_float == pytest.approx(min(100_000.0 / 2, 40_000.0 / 2)) for i in entries)
    assert [q[3] for q in policy.queue_list] == [2, 3]
    # drop on a close <= 0.95 x pin_ref while queued
    state.shares_vec[[0, 1]] = 10.0
    state.entry_pos_vec[[0, 1]] = t + 1
    f.close_arr[t + 1, 2] = 0.9 * policy.queue_list[0][4]
    policy.decide(t + 1, state)
    assert [q[3] for q in policy.queue_list] == [3] and policy.dropped_dict["break_while_queued"] == 1
    # drop after 60 sessions since confirmation
    policy.decide(t + 60, state)
    assert policy.queue_list == [] and policy.dropped_dict["aged_out"] == 1
    # drop on delisting while queued
    plant_deal(p, 4, d + 100)
    f2 = V2FeatureBook(p)
    policy2 = PolicyV(f2, cells_module.VCell(slots_int=0))
    policy2.decide(d + 105, State(6, 100_000.0))
    assert [q[3] for q in policy2.queue_list] == [4]
    f2.close_arr[d + 106, 4] = np.nan
    policy2.decide(d + 106, State(6, 100_000.0))
    assert policy2.queue_list == [] and policy2.dropped_dict["delisted"] == 1


def test_break_and_time_stops():
    p = make_panel(symbol_count_int=2)
    f = V2FeatureBook(p)
    policy = PolicyV(f, cells_module.V0_CELL)
    state = State(2, 100_000.0)
    state.shares_vec[0] = 10.0
    state.entry_pos_vec[0] = 500
    policy.pin_ref_by_symbol_dict[0] = 100.0
    f.close_arr[520, 0] = 94.0
    assert [(i.reason_str, round(i.stop_level_float, 6)) for i in policy.decide(520, state)] == [("break", 95.0)]
    f.close_arr[520, 0] = 99.0
    assert policy.decide(520, state) == []
    f.close_arr[750:752, 0] = 99.0  # above the break level, so only the time stop can fire
    assert [i.reason_str for i in policy.decide(500 + 251, state)] == ["time"]
    assert policy.decide(500 + 250, state) == []


def test_streaming_detection_equals_panel_detection():
    p = make_panel(symbol_count_int=5, seed_int=11)
    d = 400
    plant_deal(p, 0, d)
    plant_deal(p, 2, d + 30, moves=(0.004, -0.004, 0.003))
    plant_deal(p, 4, d + 60, jump=0.12)
    f = V2FeatureBook(p)
    for j, th, w in ((0.10, 0.008, 3), (0.15, 0.005, 5), (0.15, 0.003, 10)):
        panel_conf = f.confirmation(j, th, w)
        for s in range(5):
            close_vec, to_vec, m_vec = p["close_arr"][:, s], p["turnover_arr"][:, s], p["member_arr"][:, s]
            adv_vec = adv20_prev_arr(to_vec)
            rel25_vec = np.isfinite(adv_vec) & (adv_vec >= p["q25_adv20_prev_vec"])
            ev = candidate_event_arr(close_vec, to_vec, m_vec, j) & rel25_vec
            stream_conf = confirmation_dict(ev, pin_feature_dict(close_vec, to_vec, w), th, w)
            np.testing.assert_array_equal(stream_conf["conf"], panel_conf["conf"][:, s])
            panel_rows = panel_conf["symbol_vec"] == s
            np.testing.assert_array_equal(stream_conf["t_vec"], panel_conf["t_vec"][panel_rows])
            np.testing.assert_array_equal(stream_conf["event_pos_vec"], panel_conf["event_pos_vec"][panel_rows])
            np.testing.assert_allclose(stream_conf["pin_ref_vec"], panel_conf["pin_ref_vec"][panel_rows])
            np.testing.assert_allclose(stream_conf["pin_stat_vec"], panel_conf["pin_stat_vec"][panel_rows])
    assert f.confirmation(0.10, 0.008, 3)["conf"].sum() >= 3


def test_russell_3000_membership_union_and_repo_convention():
    date_index = pd.bdate_range("2000-01-03", periods=30)
    last_ts = date_index[-1]
    r1000 = pd.Series(0, index=date_index)
    r1000.iloc[:10] = 1  # member for the first 10 sessions, then removed while still trading -> last 5 member rows dropped
    r2000 = pd.Series(0, index=date_index)
    r2000.iloc[10:] = 1  # member through the calendar's last session -> no drop
    v1 = data_module.member_vec_from_series(r1000, date_index, last_ts)
    v2 = data_module.member_vec_from_series(r2000, date_index, last_ts)
    assert v1.tolist() == [1] * 5 + [0] * 25
    assert v2.tolist() == [0] * 10 + [1] * 20
    union = data_module.union_member_vec(v1, v2)
    assert union.tolist() == [1] * 5 + [0] * 5 + [1] * 20
    assert data_module.member_vec_from_series(pd.Series(dtype=float), date_index, last_ts).sum() == 0
    # as-of: a flag on a date absent from the calendar is not back-filled to an earlier session
    off = pd.Series(1, index=pd.DatetimeIndex([date_index[3] + pd.Timedelta(days=1)]))
    assert data_module.member_vec_from_series(off, date_index, last_ts).sum() == 0


def test_terminal_liquidation_and_sweep():
    p = make_panel(symbol_count_int=3)
    d = int(pd.DatetimeIndex(p["date_index"]).searchsorted(pd.Timestamp("2000-01-01"))) + 60
    plant_deal(p, 0, d)
    for k in ("open_arr", "close_arr", "unadjusted_close_arr"):
        p[k][d + 40 :, 0] = np.nan
    f = V2FeatureBook(p)
    sim = simulate(f, PolicyV(f, cells_module.V0_CELL))
    assert len(sim["trade_df"]) == 1 and sim["trade_df"].iloc[0]["reason"] == "terminal" and sim["trade_df"].iloc[0]["exit_pos"] == d + 40
    sim99 = simulate(f, PolicyV(f, cells_module.V0_CELL), terminal_factor_float=0.99)
    assert sim99["trade_df"].iloc[0]["proceeds"] == pytest.approx(sim["trade_df"].iloc[0]["proceeds"] * 0.99)
    # cash weight and sweep
    idx = pd.bdate_range("2010-01-04", periods=4)
    swept = common.sweep_return_ser(pd.Series([0.01, 0.0, 0.0, 0.0], index=idx), pd.Series([1.0, 0.9, 0.0, 0.5], index=idx), pd.Series(0.001, index=idx))
    np.testing.assert_allclose(swept.to_numpy(), [0.01, 0.001, 0.0009, 0.0])
    sim_small = simulate(f, PolicyV(f, cells_module.V0_CELL), capital_float=25_000.0)
    assert sim_small["total_ser"].iloc[0] == pytest.approx(25_000.0)


@pytest.mark.skipif(not common.trend_common.SLEEVE_SERIES_PATH.exists(), reason="stored sleeves not available")
def test_control_books_reproduce_the_prereg_table():
    taa, l_ser, bil, spy = common.load_taa_ser(), common.trend_common.load_stored_l_ser(), common.load_bil_ret_ser(), common.load_spy_tr_ret_ser()
    controls = common.control_books(taa, l_ser, bil, spy)
    for name, (p1, p2, p3, full, full_dd, long_, long_dd) in {"G3": (0.689, 1.303, 1.257, 1.288, -0.141, 1.167, -0.153), "C_BIL": (0.728, 1.334, 1.437, 1.365, -0.117, 1.226, -0.143)}.items():
        m = controls[name]
        assert (m["G-P1"]["sharpe"], m["G-P2"]["sharpe"], m["G-P3"]["sharpe"]) == pytest.approx((p1, p2, p3), abs=1e-3)
        assert (m["G-FULL"]["sharpe"], m["G-FULL"]["max_dd"], m["G-LONG"]["sharpe"], m["G-LONG"]["max_dd"]) == pytest.approx((full, full_dd, long_, long_dd), abs=1e-3)
