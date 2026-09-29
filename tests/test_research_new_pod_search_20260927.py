"""Tests for the new-pod search replica (scripts/research/new_pod_search_20260927). Synthetic data only, except the
control-book test, which reads the stored sleeves and Norgate BIL/SPY when available."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "research"))
from new_pod_search_20260927 import cells as cells_module  # noqa: E402
from new_pod_search_20260927 import common  # noqa: E402
from new_pod_search_20260927.features import PodFeatureBook, seasonality_score_tables  # noqa: E402
from new_pod_search_20260927.policies import PolicyM, PolicyS  # noqa: E402
from new_pod_search_20260927.simulate import simulate  # noqa: E402
from trend_breakout_20260927.policies import Intent, State  # noqa: E402


def make_universe(symbol_count_int: int = 6, seed_int: int = 3, start_str: str = "1998-06-01", end_str: str = "2002-12-31") -> dict:
    rng = np.random.default_rng(seed_int)
    date_index = pd.bdate_range(start_str, end_str)
    n = len(date_index)
    close_arr = 50.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, (n, symbol_count_int)), axis=0))
    open_arr = close_arr * np.exp(rng.normal(0, 0.002, close_arr.shape))
    high_arr = np.maximum(open_arr, close_arr) * (1 + np.abs(rng.normal(0, 0.004, close_arr.shape)))
    low_arr = np.minimum(open_arr, close_arr) * (1 - np.abs(rng.normal(0, 0.004, close_arr.shape)))
    turnover_arr = np.full((n, symbol_count_int), 5e7)  # equal liquidity: every name passes REL25 (the filter itself is tested in the trend suite)
    spy_ser = pd.Series(100.0 * np.exp(np.cumsum(np.full(n, 0.0004))), index=date_index)
    month_end_index = pd.DatetimeIndex(pd.Series(date_index, index=date_index.to_period("M")).groupby(level=0).max().to_numpy())
    return {"universe_str": "TEST", "date_index": date_index, "symbol_list": [f"S{i:02d}" for i in range(symbol_count_int)], "open_arr": open_arr, "high_arr": high_arr,
            "low_arr": low_arr, "close_arr": close_arr, "volume_arr": np.full(close_arr.shape, 1e6), "turnover_arr": turnover_arr, "unadjusted_close_arr": close_arr.copy(),
            "dividend_arr": np.zeros(close_arr.shape), "member_arr": np.ones(close_arr.shape, dtype=np.int8), "spy_close_ser": spy_ser, "qqq_close_ser": spy_ser,
            "vxn_close_ser": pd.Series(20.0, index=date_index), "month_end_index": month_end_index}


def plant_event(universe_dict: dict, symbol_idx: int, d_int: int, jump_float: float = 0.25, window_int: int = 5, pin_range_float: float = 0.003, hold_ok_bool: bool = True, turnover_zero_day: int | None = None) -> None:
    """A cash-deal-like path: jump at d on a volume shock, then a flat pinned window."""
    close_arr, open_arr, high_arr, low_arr, turnover_arr = (universe_dict[k] for k in ("close_arr", "open_arr", "high_arr", "low_arr", "turnover_arr"))
    base_float = close_arr[d_int - 1, symbol_idx]
    level_float = base_float * (1.0 + jump_float)
    close_arr[d_int, symbol_idx] = level_float
    open_arr[d_int, symbol_idx] = level_float
    high_arr[d_int, symbol_idx] = level_float * 1.01
    low_arr[d_int, symbol_idx] = level_float * 0.99
    turnover_arr[d_int, symbol_idx] = turnover_arr[d_int - 1, symbol_idx] * 8.0
    for k_int in range(1, 60):
        close_arr[d_int + k_int, symbol_idx] = level_float * (1.0 - (0.0 if hold_ok_bool else 0.05) + 0.0005 * (k_int % 2))
        open_arr[d_int + k_int, symbol_idx] = close_arr[d_int + k_int, symbol_idx]
        high_arr[d_int + k_int, symbol_idx] = close_arr[d_int + k_int, symbol_idx] * (1 + pin_range_float / 2)
        low_arr[d_int + k_int, symbol_idx] = close_arr[d_int + k_int, symbol_idx] * (1 - pin_range_float / 2)
    universe_dict["unadjusted_close_arr"][:, symbol_idx] = close_arr[:, symbol_idx]
    if turnover_zero_day is not None:
        turnover_arr[turnover_zero_day, symbol_idx] = 0.0


# ----------------------------------------------------------------------------------------------------------------------
# family M features and policy
# ----------------------------------------------------------------------------------------------------------------------
def test_event_and_pin_detection_and_rejections():
    u = make_universe()
    d_int = 400
    plant_event(u, 0, d_int)                                   # clean event
    plant_event(u, 1, d_int, jump_float=0.05)                  # jump too small
    plant_event(u, 2, d_int, pin_range_float=0.03)             # pin window too volatile
    plant_event(u, 3, d_int, hold_ok_bool=False)               # falls below 0.97 x Close_d
    plant_event(u, 4, d_int, turnover_zero_day=d_int + 3)      # a halted / padded bar inside the window
    plant_event(u, 5, d_int)
    u["turnover_arr"][d_int, 5] = u["turnover_arr"][d_int - 1, 5] * 2.0  # no volume shock
    f = PodFeatureBook(u)
    cell = cells_module.M0_CELL
    conf = f.confirmation(cell.jump_float, cell.theta_float, cell.window_int)
    t_int = d_int + cell.window_int
    assert conf["conf"][t_int].tolist() == [True, False, False, False, False, False]
    assert not conf["conf"][t_int - 1].any() and not conf["conf"][t_int + 1].any()
    assert conf["event_pos"][t_int, 0] == d_int
    window_close = u["close_arr"][d_int + 1 : d_int + 1 + cell.window_int, 0]
    assert conf["pin_ref"][t_int, 0] == pytest.approx(np.median(window_close))
    assert f.event(cell.jump_float)[d_int, 0] and not f.event(cell.jump_float)[d_int, 1]
    assert f.pin_features(5)["traded"][d_int, 4] is np.False_ or not f.pin_features(5)["traded"][d_int, 4]


def test_policy_m_queue_order_budget_and_stops():
    u = make_universe(symbol_count_int=6)
    d_int = 400
    for s in range(4):
        plant_event(u, s, d_int, pin_range_float=0.002 + 0.001 * s)
    f = PodFeatureBook(u)
    cell = cells_module.MCell(slots_int=2)
    policy = PolicyM(f, cell)
    state = State(6, 100_000.0)
    state.cash_float = 30_000.0
    state.total_value_float = 100_000.0
    t_int = d_int + cell.window_int
    intents = policy.decide(t_int, state)
    entries = [i for i in intents if i.kind_str == "value"]
    assert [i.symbol_idx for i in entries] == [0, 1]  # same confirmation date: lowest pin_vol first
    assert all(i.amount_float == pytest.approx(min(100_000.0 / 2, 30_000.0 / 2)) for i in entries)
    assert [q[3] for q in policy.queue_list] == [2, 3]  # the rest waits FIFO
    # break stop: close at 94% of pin_ref fires; time stop after 252 sessions
    state.shares_vec[0] = 10.0
    state.entry_pos_vec[0] = t_int + 1
    policy.pin_ref_by_symbol_dict[0] = 100.0
    f.close_arr[t_int + 5, 0] = 93.0
    intents = policy.decide(t_int + 5, state)
    assert any(i.kind_str == "exit" and i.reason_str == "break" and i.symbol_idx == 0 and i.stop_level_float == pytest.approx(95.0) for i in intents)
    f.close_arr[t_int + 5, 0] = 99.0
    state.entry_pos_vec[0] = t_int + 5 - 251
    intents = policy.decide(t_int + 5, state)
    assert any(i.kind_str == "exit" and i.reason_str == "time" and i.symbol_idx == 0 for i in intents)
    state.entry_pos_vec[0] = t_int + 5 - 250
    assert not any(i.kind_str == "exit" for i in policy.decide(t_int + 5, state))


def test_simulated_deal_terminal_liquidation_and_exit_classes():
    u = make_universe(symbol_count_int=3)
    d_int = int(pd.DatetimeIndex(u["date_index"]).searchsorted(pd.Timestamp("2000-01-01"))) + 60  # inside the trading calendar
    plant_event(u, 0, d_int)
    close_end_int = d_int + 40
    for key in ("open_arr", "high_arr", "low_arr", "close_arr", "unadjusted_close_arr"):
        u[key][close_end_int:, 0] = np.nan  # the deal closes: the series ends
    f = PodFeatureBook(u)
    sim = simulate(f, PolicyM(f, cells_module.M0_CELL))
    trade_df = sim["trade_df"]
    assert len(trade_df) == 1
    row = trade_df.iloc[0]
    assert row["reason"] == "terminal" and row["entry_pos"] == d_int + 6 and row["exit_pos"] == close_end_int
    assert row["proceeds"] == pytest.approx(u["close_arr"][close_end_int - 1, 0] * (row["cost"] / (u["open_arr"][d_int + 6, 0] * 1.00025)))
    sim99 = simulate(f, PolicyM(f, cells_module.M_SENSITIVITY_DICT["terminal_x0.99"]), terminal_factor_float=0.99)
    assert sim99["trade_df"].iloc[0]["proceeds"] == pytest.approx(row["proceeds"] * 0.99)


# ----------------------------------------------------------------------------------------------------------------------
# family S scores, gate, hedge
# ----------------------------------------------------------------------------------------------------------------------
def test_seasonality_scores_match_hand_computation_and_use_only_past_years():
    u = make_universe(symbol_count_int=2, start_str="1998-06-01", end_str="2002-12-31")
    rng = np.random.default_rng(5)
    ext_index = pd.bdate_range("1990-01-01", "2002-12-31", freq="BME")
    month_end_index = pd.DatetimeIndex(u["month_end_index"])
    early = ext_index[ext_index.to_period("M") < month_end_index[0].to_period("M")]
    full_index = early.append(month_end_index)
    close_df = pd.DataFrame(np.exp(np.cumsum(rng.normal(0, 0.05, (len(full_index), 3)), axis=0)), index=full_index, columns=["SPY", "S00", "S01"])
    tables = seasonality_score_tables(u, close_df, ["SE_1", "SE_2_5", "SE_1_10", "SE_1_20"])
    # decision at the 2001-01 month-end (row for January 2001): target February; past Februaries 2000, 1999, ...
    row_int = list(month_end_index.to_period("M")).index(pd.Period("2001-01", "M"))
    ret = close_df / close_df.shift(1) - 1
    feb = [float(ret.loc[ret.index[ret.index.to_period("M") == pd.Period(f"{y}-02", "M")][0], "S00"]) for y in range(1990, 2001)]  # 11 Februaries 1990..2000
    assert tables["SE_1"][row_int, 0] == pytest.approx(feb[-1])
    assert tables["SE_2_5"][row_int, 0] == pytest.approx(np.mean(feb[-5:-1]))
    assert tables["SE_1_10"][row_int, 0] == pytest.approx(np.mean(feb[-10:]))
    # 11 valid Februaries (1990..2000) satisfy "at least 10 valid": SE_1_20 is their mean; two years earlier only 9 exist -> NaN
    assert tables["SE_1_20"][row_int, 0] == pytest.approx(np.mean(feb))
    assert tables["SE_1_20"][row_int - 12, 0] == pytest.approx(np.mean(feb[:-1]))
    assert np.isnan(tables["SE_1_20"][row_int - 24, 0])
    # causality: perturb the target month and every later month; no score at this row may change
    perturbed_df = close_df.copy()
    perturbed_df.loc[perturbed_df.index.to_period("M") >= pd.Period("2001-02", "M"), "S00"] *= 3.0
    tables_p = seasonality_score_tables(u, perturbed_df, ["SE_1", "SE_2_5", "SE_1_10"])
    for h in ("SE_1", "SE_2_5", "SE_1_10"):
        np.testing.assert_allclose(tables_p[h][row_int, 0], tables[h][row_int, 0])
    # a missing month-end close removes that year
    holed_df = close_df.copy()
    holed_df.loc[holed_df.index[holed_df.index.to_period("M") == pd.Period("2000-01", "M")], "S00"] = np.nan
    tables_h = seasonality_score_tables(u, holed_df, ["SE_1", "SE_2_5"])
    assert np.isnan(tables_h["SE_1"][row_int, 0])  # Feb 2000 needs the Jan 2000 close
    assert tables_h["SE_2_5"][row_int, 0] == pytest.approx(np.mean(feb[-5:-1]))


def test_policy_s_gate_and_hedge_weights():
    u = make_universe(symbol_count_int=5)
    u_sh = dict(u)
    for key in ("open_arr", "high_arr", "low_arr", "close_arr", "unadjusted_close_arr", "volume_arr", "turnover_arr", "dividend_arr"):
        u_sh[key] = np.column_stack([u[key], u[key][:, 0]])
    u_sh["member_arr"] = np.column_stack([u["member_arr"], np.zeros(len(u["date_index"]), dtype=np.int8)])
    u_sh["symbol_list"] = u["symbol_list"] + ["SH"]
    f = PodFeatureBook(u_sh)
    score_arr = np.tile(np.array([0.05, 0.04, 0.03, 0.02, 0.01, np.nan]), (len(u["month_end_index"]), 1))
    gated = PolicyS(f, cells_module.SCell("SE_1_10", 2, "GATED"), score_arr, sh_idx=5)
    hedged = PolicyS(f, cells_module.SCell("SE_1_10", 2, "HEDGED"), score_arr, sh_idx=5)
    pos_int = next(iter(gated.row_by_pos_dict))
    state = State(6, 100_000.0)
    gated.regime_vec[pos_int] = True
    intents = gated.decide(pos_int, state)
    assert sorted((i.symbol_idx, i.amount_float) for i in intents if i.kind_str == "target_pct") == [(0, 0.5), (1, 0.5)]
    gated.regime_vec[pos_int] = False
    state.shares_vec[0] = 5.0
    intents = gated.decide(pos_int, state)
    assert [(i.symbol_idx, i.kind_str, i.reason_str) for i in intents] == [(0, "exit", "gate")]
    intents = hedged.decide(pos_int, State(6, 100_000.0))
    targets = sorted((i.symbol_idx, i.amount_float) for i in intents if i.kind_str == "target_pct")
    assert targets == [(0, 0.25), (1, 0.25), (5, 0.5)]
    f.close_arr[pos_int, 5] = np.nan  # before SH exists nothing is traded
    assert hedged.decide(pos_int, State(6, 100_000.0)) == []


# ----------------------------------------------------------------------------------------------------------------------
# sweep and the control books
# ----------------------------------------------------------------------------------------------------------------------
def test_sweep_arithmetic():
    index = pd.bdate_range("2010-01-04", periods=5)
    engine_ret = pd.Series([0.01, -0.02, 0.0, 0.005, 0.0], index=index)
    cash_weight = pd.Series([1.0, 0.5, 0.0, -0.1, 0.3], index=index).clip(lower=0.0)
    bil = pd.Series([0.0001, 0.0002, 0.0003, 0.0004, 0.0005], index=index)
    swept = common.sweep_return_ser(engine_ret, cash_weight, bil)
    expected = engine_ret.to_numpy() + np.array([0.0, 1.0 * 0.0002, 0.5 * 0.0003, 0.0 * 0.0004, 0.0 * 0.0005])
    np.testing.assert_allclose(swept.to_numpy(), expected)
    swept_missing = common.sweep_return_ser(engine_ret, cash_weight, bil.iloc[:2])
    np.testing.assert_allclose(swept_missing.to_numpy(), engine_ret.to_numpy() + np.array([0.0, 0.0002, 0.0, 0.0, 0.0]))


@pytest.mark.skipif(not common.trend_common.SLEEVE_SERIES_PATH.exists(), reason="stored sleeves not available")
def test_control_books_reproduce_the_prereg_table():
    taa = common.load_taa_ser()
    l_ser = common.trend_common.load_stored_l_ser()
    bil = common.load_bil_ret_ser()
    spy = common.load_spy_tr_ret_ser()
    controls = common.control_books(taa, l_ser, bil, spy)
    expected = {"G3": (0.689, 1.303, 1.257, 1.288, -0.141, 1.167, -0.153), "C_CASH0": (0.722, 1.325, 1.357, 1.334, -0.117, 1.201, -0.145),
                "C_BIL": (0.728, 1.334, 1.437, 1.365, -0.117, 1.226, -0.143), "C_SPY": (0.628, 1.368, 1.325, 1.354, -0.133, 1.198, -0.206)}
    for name, (p1, p2, p3, full, full_dd, long_, long_dd) in expected.items():
        m = controls[name]
        assert m["G-P1"]["sharpe"] == pytest.approx(p1, abs=1e-3) and m["G-P2"]["sharpe"] == pytest.approx(p2, abs=1e-3) and m["G-P3"]["sharpe"] == pytest.approx(p3, abs=1e-3)
        assert m["G-FULL"]["sharpe"] == pytest.approx(full, abs=1e-3) and m["G-FULL"]["max_dd"] == pytest.approx(full_dd, abs=1e-3)
        assert m["G-LONG"]["sharpe"] == pytest.approx(long_, abs=1e-3) and m["G-LONG"]["max_dd"] == pytest.approx(long_dd, abs=1e-3)
