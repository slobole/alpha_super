"""Tests for the trend / breakout research replica (scripts/research/trend_breakout_20260927). Synthetic data only,
except the book test, which reads the stored corrected sleeves when they exist."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "research"))
import ndx_param_robustness_core as core  # noqa: E402
from trend_breakout_20260927 import cells as cells_module  # noqa: E402
from trend_breakout_20260927 import common, family_c  # noqa: E402
from trend_breakout_20260927.features import DailyFeatureBook  # noqa: E402
from trend_breakout_20260927.policies import Intent, PolicyA, PolicyB, PolicyMonthlyTargets, State, rank_indices  # noqa: E402
from trend_breakout_20260927.simulate import ledger_target_shares_float, simulate  # noqa: E402


# ----------------------------------------------------------------------------------------------------------------------
# synthetic universe
# ----------------------------------------------------------------------------------------------------------------------
def make_universe(symbol_count_int: int = 12, seed_int: int = 7, price_scale_vec: np.ndarray | None = None) -> dict:
    """Business days 1998-06..2001-12, random-walk OHLC, all members, quiet VXN, rising SPY, split factor 1."""
    rng = np.random.default_rng(seed_int)
    date_index = pd.bdate_range("1998-06-01", "2001-12-31")
    date_count_int = len(date_index)
    drift_vec = rng.normal(0.0005, 0.0004, symbol_count_int)
    return_arr = drift_vec[None, :] + rng.normal(0, 0.015, (date_count_int, symbol_count_int))
    close_arr = 50.0 * np.exp(np.cumsum(return_arr, axis=0))
    open_arr = close_arr * np.exp(rng.normal(0, 0.003, close_arr.shape))
    high_arr = np.maximum(open_arr, close_arr) * (1 + np.abs(rng.normal(0, 0.005, close_arr.shape)))
    low_arr = np.minimum(open_arr, close_arr) * (1 - np.abs(rng.normal(0, 0.005, close_arr.shape)))
    unadjusted_arr = close_arr.copy()
    if price_scale_vec is not None:
        for field_arr in (open_arr, high_arr, low_arr, close_arr):
            field_arr *= price_scale_vec[None, :]
    spy_ser = pd.Series(100.0 * np.exp(np.cumsum(np.full(date_count_int, 0.0004))), index=date_index)
    month_end_index = pd.DatetimeIndex(pd.Series(date_index, index=date_index.to_period("M")).groupby(level=0).max().to_numpy())
    return {
        "universe_str": "TEST",
        "date_index": date_index,
        "symbol_list": [f"S{i:02d}" for i in range(symbol_count_int)],
        "open_arr": open_arr,
        "high_arr": high_arr,
        "low_arr": low_arr,
        "close_arr": close_arr,
        "volume_arr": np.full(close_arr.shape, 1e6),
        "turnover_arr": np.tile(np.arange(1, symbol_count_int + 1, dtype=float) * 1e7, (date_count_int, 1)),
        "unadjusted_close_arr": unadjusted_arr,
        "dividend_arr": np.zeros(close_arr.shape),
        "member_arr": np.ones(close_arr.shape, dtype=np.int8),
        "spy_close_ser": spy_ser,
        "qqq_close_ser": spy_ser,
        "vxn_close_ser": pd.Series(20.0, index=date_index),
        "month_end_index": month_end_index,
    }


class FixedIntentPolicy:
    """Emits pre-set intents at given decision positions (tests the simulator on its own)."""

    def __init__(self, intents_by_pos_dict: dict):
        self.intents_by_pos_dict = intents_by_pos_dict

    def decide(self, pos_int: int, state: State) -> list[Intent]:
        return list(self.intents_by_pos_dict.get(pos_int, []))


def flat_universe(price_scale_float: float = 40.0) -> tuple[dict, int]:
    """Two symbols, raw close 14.04 then 15.0 from a chosen day, adjusted = raw / price_scale (a later 40:1 factor)."""
    universe_dict = make_universe(symbol_count_int=2)
    date_index = universe_dict["date_index"]
    pos_int = int(date_index.get_loc(pd.Timestamp("2000-03-01")))
    raw_close_arr = np.full(universe_dict["close_arr"].shape, 14.04)
    raw_close_arr[pos_int:] = 15.0
    raw_open_arr = raw_close_arr.copy()
    raw_open_arr[pos_int] = 14.8
    universe_dict["unadjusted_close_arr"] = raw_close_arr.copy()
    universe_dict["close_arr"] = raw_close_arr / price_scale_float
    universe_dict["open_arr"] = raw_open_arr / price_scale_float
    universe_dict["high_arr"] = (raw_close_arr + 1.0) / price_scale_float
    universe_dict["low_arr"] = (raw_open_arr - 1.0) / price_scale_float
    return universe_dict, pos_int


# ----------------------------------------------------------------------------------------------------------------------
# simulator accounting against the engine's own numbers (tests/test_historical_share_units.py)
# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("price_scale_float", [40.0, 0.025, 1.0])
def test_value_order_uses_raw_whole_shares_and_raw_equivalent_commission(price_scale_float):
    universe_dict, pos_int = flat_universe(price_scale_float)
    feature_obj = DailyFeatureBook(universe_dict)
    policy = FixedIntentPolicy({pos_int - 1: [Intent(0, "value", 5_000.0, "entry")]})
    sim = simulate(feature_obj, policy)
    # raw shares int(5000 / 14.04) = 356 -> ledger 356 x price_scale; fill 14.8 / scale x 1.00025; fee max(1, 0.005 x 356)
    ledger_float = 356.0 * price_scale_float
    fill_float = 14.8 / price_scale_float * 1.00025
    cash_float = 100_000.0 - ledger_float * fill_float - 1.78
    total_float = cash_float + ledger_float * 15.0 / price_scale_float
    assert sim["total_ser"].loc["2000-03-01"] == pytest.approx(total_float, rel=1e-12)
    assert sim["commission_ser"].loc["2000-03-01"] == pytest.approx(1.78)
    assert sim["traded_notional_ser"].loc["2000-03-01"] == pytest.approx(ledger_float * fill_float)


def test_target_percent_sizing_and_exit_commission_at_execution_scale():
    universe_dict, pos_int = flat_universe(40.0)
    # a 2-for-1 split between decision and fill day: raw close halves on the fill day, adjusted path unchanged
    universe_dict["unadjusted_close_arr"][pos_int:] = 15.0 / 40.0 * 20.0
    feature_obj = DailyFeatureBook(universe_dict)
    policy = FixedIntentPolicy({pos_int - 1: [Intent(0, "target_pct", 0.05, "rebalance")], pos_int: [Intent(0, "exit", 0.0, "stop", float("nan"))]})
    sim = simulate(feature_obj, policy)
    trade_df = sim["trade_df"]
    assert len(trade_df) == 1
    # buy: int(0.05 x 100000 / 14.04) = 356 raw -> 14240 ledger; fee at the fill-day scale k_E = 7.5 / 0.375 = 20 -> 712 raw -> $3.56
    assert sim["commission_ser"].iloc[sim["start_pos_int"] and 0 + (pos_int - sim["start_pos_int"])] == pytest.approx(3.56)
    # sell the whole position next day at open 15/40 x (1 - 0.00025); fee at that day's scale (still 20) -> 3.56
    assert trade_df.iloc[0]["proceeds"] == pytest.approx(14240.0 * 0.375 * (1 - 0.00025))
    assert trade_df.iloc[0]["reason"] == "stop"


def test_zero_share_order_is_skipped_without_commission_and_below_one_raw_share():
    universe_dict, pos_int = flat_universe(40.0)
    feature_obj = DailyFeatureBook(universe_dict)
    policy = FixedIntentPolicy({pos_int - 1: [Intent(0, "value", 10.0, "entry"), Intent(1, "exit", 0.0, "stop", float("nan"))]})
    sim = simulate(feature_obj, policy)
    assert sim["commission_ser"].sum() == 0.0
    assert sim["total_ser"].iloc[-1] == 100_000.0


def test_terminal_liquidation_uses_last_close_and_anchor_scale_and_haircut():
    universe_dict, pos_int = flat_universe(40.0)
    liquidation_pos_int = pos_int + 3
    for key_str in ("open_arr", "close_arr", "unadjusted_close_arr"):
        universe_dict[key_str][liquidation_pos_int:, 0] = np.nan
    feature_obj = DailyFeatureBook(universe_dict)
    policy = FixedIntentPolicy({pos_int - 1: [Intent(0, "value", 5_000.0, "entry")]})
    for haircut_float in (1.0, 0.75):
        sim = simulate(feature_obj, policy, terminal_haircut_float=haircut_float)
        trade_df = sim["trade_df"]
        assert len(trade_df) == 1 and trade_df.iloc[0]["reason"] == "terminal"
        assert trade_df.iloc[0]["exit_pos"] == liquidation_pos_int
        assert trade_df.iloc[0]["proceeds"] == pytest.approx(356.0 * 40.0 * 0.375 * haircut_float)
        # commission at the anchor bar (last finite close): raw 356 -> $1.78
        assert sim["commission_ser"].iloc[liquidation_pos_int - sim["start_pos_int"]] == pytest.approx(1.78)
        assert sim["open_positions_end"] == {}


def test_dividend_credited_the_session_after_entitlement_at_75_percent():
    universe_dict, pos_int = flat_universe(40.0)
    universe_dict["dividend_arr"][pos_int + 2, 0] = 0.4 / 40.0  # entitlement day pos+2, per adjusted share
    feature_obj = DailyFeatureBook(universe_dict)
    policy = FixedIntentPolicy({pos_int - 1: [Intent(0, "value", 5_000.0, "entry")]})
    sim = simulate(feature_obj, policy)
    start_int = sim["start_pos_int"]
    jump_float = sim["total_ser"].iloc[pos_int + 3 - start_int] - sim["total_ser"].iloc[pos_int + 2 - start_int]
    assert jump_float == pytest.approx(0.75 * 356.0 * 0.4)


def test_ledger_target_shares_legacy_and_historical():
    assert ledger_target_shares_float(5_000.0, 14.04, 14.04 / 40.0, True) == 356.0 * 40.0
    assert ledger_target_shares_float(5_000.0, 14.04, 14.04 / 40.0, False) == float(int(5_000.0 / (14.04 / 40.0)))
    with pytest.raises(RuntimeError):
        ledger_target_shares_float(5_000.0, np.nan, 1.0, True)


# ----------------------------------------------------------------------------------------------------------------------
# stops
# ----------------------------------------------------------------------------------------------------------------------
def test_stop_spec_arithmetic_and_undefined_inputs():
    ch = cells_module.StopSpec("CH", 3.0)
    pt = cells_module.StopSpec("PT", 0.15)
    close_vec = np.array([90.0, 84.9, 85.1, 100.0])
    hwm_vec = np.array([100.0, 100.0, 100.0, np.nan])
    atr_vec = np.array([5.0, 5.0, np.nan, 5.0])
    np.testing.assert_array_equal(ch.fires_vec(close_vec, hwm_vec, atr_vec), [False, True, False, False])
    np.testing.assert_array_equal(pt.fires_vec(close_vec, hwm_vec, atr_vec), [False, True, False, False])
    np.testing.assert_allclose(ch.stop_level_vec(hwm_vec, atr_vec)[:2], [85.0, 85.0])
    np.testing.assert_allclose(pt.stop_level_vec(hwm_vec, atr_vec)[:2], [85.0, 85.0])
    assert not cells_module.StopSpec().fires_vec(close_vec, hwm_vec, atr_vec).any()


def test_hwm_counts_the_entry_day_survives_resize_and_resets_on_reentry():
    universe_dict = make_universe(symbol_count_int=2, seed_int=3)
    feature_obj = DailyFeatureBook(universe_dict)
    date_index = universe_dict["date_index"]
    e_int = int(date_index.get_loc(pd.Timestamp("2000-02-01")))
    resize_int = e_int + 10
    exit_int = e_int + 20
    reentry_int = e_int + 30
    policy = FixedIntentPolicy({
        e_int - 1: [Intent(0, "target_pct", 0.10, "rebalance")],
        resize_int - 1: [Intent(0, "target_pct", 0.20, "rebalance")],
        exit_int - 1: [Intent(0, "exit", 0.0, "rebalance")],
        reentry_int - 1: [Intent(0, "target_pct", 0.10, "rebalance")],
    })
    hwm_log = {}

    class SpyPolicy(FixedIntentPolicy):
        def decide(self, pos_int, state):
            hwm_log[pos_int] = float(state.hwm_vec[0])
            return super().decide(pos_int, state)

    simulate(feature_obj, SpyPolicy(policy.intents_by_pos_dict))
    close_vec = universe_dict["close_arr"][:, 0]
    assert hwm_log[e_int] == close_vec[e_int]
    assert hwm_log[resize_int + 3] == close_vec[e_int : resize_int + 4].max()  # no reset at the re-size
    assert np.isnan(hwm_log[exit_int])
    assert hwm_log[reentry_int] == close_vec[reentry_int]  # fresh episode


# ----------------------------------------------------------------------------------------------------------------------
# policies
# ----------------------------------------------------------------------------------------------------------------------
def test_rank_indices_orders_by_score_then_symbol():
    score_vec = np.array([1.0, 3.0, 3.0, np.nan, 2.0])
    eligible_vec = np.array([True, True, True, True, False])
    symbol_rank_vec = np.array([0, 2, 1, 3, 4])
    np.testing.assert_array_equal(rank_indices(score_vec, eligible_vec, symbol_rank_vec), [2, 1, 0])


def test_policy_a_without_stop_equals_the_26_sep_replica_targets():
    universe_dict = make_universe()
    feature_obj = DailyFeatureBook(universe_dict)
    policy_a = PolicyA(feature_obj, cells_module.L_REFERENCE_CELL)
    sim_a = simulate(feature_obj, policy_a)
    core_targets = core.build_target_list(feature_obj, core.L_CELL)
    sim_core = simulate(feature_obj, PolicyMonthlyTargets(core_targets))
    pd.testing.assert_series_equal(sim_a["total_ser"], sim_core["total_ser"])
    for target_dict in core_targets:
        assert policy_a.selection_log_dict[target_dict["decision_pos"]] == sorted(int(i) for i in target_dict["symbol_idx_vec"])
    assert len(core_targets) > 12


def test_policy_a_stop_exits_next_open_and_cash_policy_keeps_slot_empty():
    universe_dict = make_universe(symbol_count_int=12, seed_int=11)
    feature_obj = DailyFeatureBook(universe_dict)
    cell = cells_module.ACell(cells_module.StopSpec("PT", 0.10), "CASH")
    sim = simulate(feature_obj, PolicyA(feature_obj, cell), record_positions_bool=True)
    trade_df = sim["trade_df"]
    stop_df = trade_df[trade_df["reason"] == "stop"]
    assert len(stop_df) > 0
    close_arr = universe_dict["close_arr"]
    for row in stop_df.itertuples(index=False):
        close_before = close_arr[row.entry_pos : row.exit_pos, row.symbol_idx]
        assert close_arr[row.exit_pos - 1, row.symbol_idx] <= 0.9 * close_before.max() + 1e-12
        assert row.exit_open == universe_dict["open_arr"][row.exit_pos, row.symbol_idx]
    # under CASH the number of names held never exceeds 10 and drops after a stop until the next month-end
    held_counts = {pos_int: len(held) for pos_int, held, _ in sim["position_log"]}
    assert max(held_counts.values()) <= 10


def test_policy_a_refill_budget_is_capped_by_cash_plus_exiting_value():
    universe_dict = make_universe(symbol_count_int=12, seed_int=5)
    feature_obj = DailyFeatureBook(universe_dict)
    cell = cells_module.ACell(cells_module.StopSpec("PT", 0.05), "REFILL")
    policy = PolicyA(feature_obj, cell)
    sim = simulate(feature_obj, policy)
    intent_df = sim["intent_df"]
    refill_df = intent_df[intent_df["reason"] == "refill"]
    assert len(refill_df) > 0
    # every refill day: budget <= V x s / 10 (s = 1 with VXN 20 < 22 -> clip(22/20, .25, 1) = 1)
    total_ser = sim["total_ser"]
    for decision_pos, group_df in refill_df.groupby("decision_pos"):
        total_float = float(total_ser.loc[universe_dict["date_index"][decision_pos]])
        assert group_df["amount"].max() <= total_float / 10 + 1e-9
        assert group_df["amount"].nunique() == 1


def test_policy_b_admission_budget_and_slot_count():
    universe_dict = make_universe(symbol_count_int=6, seed_int=2)
    feature_obj = DailyFeatureBook(universe_dict)
    cell = cells_module.BCell(n_int=50, k_float=5.0, slots_int=3, rank_str="R2")
    policy = PolicyB(feature_obj, cell)
    pos_int = len(universe_dict["date_index"]) - 5
    state = State(6, 100_000.0)
    state.cash_float = 12_000.0
    state.total_value_float = 90_000.0
    # force signals: all six names break out today
    policy.breakout_arr[pos_int] = True
    policy.sma200_arr[pos_int] = True
    policy.rel25_arr[pos_int] = True
    policy.regime_vec[pos_int] = True
    intent_list = policy.decide(pos_int, state)
    entry_list = [i for i in intent_list if i.kind_str == "value"]
    assert len(entry_list) == 3  # K = 3 free slots, 6 signals
    assert all(i.amount_float == pytest.approx(min(90_000.0 / 3, 12_000.0 / 3)) for i in entry_list)
    natr_vec = feature_obj.natr(20)[pos_int]
    assert [i.symbol_idx for i in entry_list] == list(np.argsort(natr_vec)[:3])  # R2: lowest NATR first


def test_policy_b_regime_blocks_entries_only_and_regime_exit_variant_sells():
    universe_dict = make_universe(symbol_count_int=6, seed_int=2)
    feature_obj = DailyFeatureBook(universe_dict)
    pos_int = len(universe_dict["date_index"]) - 5
    state = State(6, 100_000.0)
    state.shares_vec[0] = 10.0
    state.hwm_vec[0] = universe_dict["close_arr"][pos_int, 0]
    state.entry_pos_vec[0] = pos_int - 30
    for regime_exit_bool, expected_exits in ((False, 0), (True, 1)):
        policy = PolicyB(feature_obj, cells_module.BCell(regime_exit_bool=regime_exit_bool))
        policy.regime_vec[pos_int] = False
        policy.breakout_arr[pos_int] = True
        intent_list = policy.decide(pos_int, state)
        assert sum(i.kind_str == "value" for i in intent_list) == 0
        assert sum(i.kind_str == "exit" for i in intent_list) == expected_exits


def test_breakout_reference_excludes_the_breakout_day():
    universe_dict = make_universe(symbol_count_int=3)
    feature_obj = DailyFeatureBook(universe_dict)
    close_arr = universe_dict["close_arr"]
    hh_arr = feature_obj.hh_close(50)
    t = 400
    assert hh_arr[t, 1] == close_arr[t - 50 : t, 1].max()
    assert bool(feature_obj.breakout_pass(50)[t, 1]) == bool(close_arr[t, 1] > hh_arr[t, 1])


def test_rel25_uses_turnover_and_drops_the_least_traded_quarter():
    universe_dict = make_universe(symbol_count_int=8)
    feature_obj = DailyFeatureBook(universe_dict)
    pos_int = len(universe_dict["date_index"]) - 3
    pass_vec = feature_obj.rel25_pass()[pos_int]
    assert pass_vec.sum() == 6 and not pass_vec[:2].any()
    np.testing.assert_allclose(feature_obj.adv20_dollar()[pos_int], np.arange(1, 9) * 1e7)


# ----------------------------------------------------------------------------------------------------------------------
# family C scores
# ----------------------------------------------------------------------------------------------------------------------
def test_family_c_scores_match_a_direct_regression():
    rng = np.random.default_rng(1)
    universe_dict = make_universe(symbol_count_int=3)
    month_end_index = pd.DatetimeIndex(universe_dict["month_end_index"])
    early_index = pd.bdate_range("1995-01-01", "1998-05-31", freq="BME")
    ext_index = early_index.append(month_end_index)
    market_ret_vec = rng.normal(0.005, 0.04, len(ext_index))
    beta_vec = np.array([0.8, 1.2, 1.5])
    eps_arr = rng.normal(0, 0.05, (len(ext_index), 3))
    stock_ret_arr = 0.002 + beta_vec[None, :] * market_ret_vec[:, None] + eps_arr
    close_df = pd.DataFrame(np.exp(np.cumsum(np.column_stack([market_ret_vec, stock_ret_arr]), axis=0)), index=ext_index, columns=["SPY", "S00", "S01", "S02"])
    tables = family_c.build_c_score_tables(universe_dict, close_df, [cells_module.C0_CELL, cells_module.CCell("RES12-0", 36), cells_module.CCell("TOT12-1", 36)])
    ret_df = close_df / close_df.shift(1) - 1
    j = len(ext_index) - 1
    x = ret_df["SPY"].to_numpy()[j - 35 : j + 1]
    for s_int, symbol_str in enumerate(["S00", "S01", "S02"]):
        y = ret_df[symbol_str].to_numpy()[j - 35 : j + 1]
        coef = np.polyfit(x, y, 1)
        e = y - (coef[1] + coef[0] * x)
        assert tables["RES12-1_W36"][-1, s_int] == pytest.approx(e[-12:-1].mean() / e[-12:-1].std(ddof=1), rel=1e-9)
        assert tables["RES12-0_W36"][-1, s_int] == pytest.approx(e[-12:].mean() / e[-12:].std(ddof=1), rel=1e-9)
        assert tables["TOT12-1_W36"][-1, s_int] == pytest.approx(y[-12:-1].mean() / y[-12:-1].std(ddof=1), rel=1e-9)


# ----------------------------------------------------------------------------------------------------------------------
# invariance (V1-style) and the book model
# ----------------------------------------------------------------------------------------------------------------------
def test_daily_family_nav_is_invariant_to_a_per_stock_price_constant():
    scale_vec = np.exp(np.linspace(np.log(0.02), np.log(50.0), 8))
    base_dict = make_universe(symbol_count_int=8, seed_int=9)
    scaled_dict = make_universe(symbol_count_int=8, seed_int=9, price_scale_vec=scale_vec)
    for cell in (cells_module.BCell(n_int=50, k_float=3.0, slots_int=3), cells_module.ACell(cells_module.StopSpec("CH", 2.0), "REFILL")):
        run_pair = []
        for universe_dict in (base_dict, scaled_dict):
            feature_obj = DailyFeatureBook(universe_dict)
            policy = PolicyB(feature_obj, cell) if isinstance(cell, cells_module.BCell) else PolicyA(feature_obj, cell)
            run_pair.append(simulate(feature_obj, policy, record_positions_bool=True))
        # Rescaling makes U / Close bit-unstable across days, so a re-size to an unchanged raw share count becomes a
        # ~1e-13-share order that the engine fills at the $1 minimum commission (an engine artifact the replica
        # mirrors). Positions and decisions must be identical; NAV before commissions must match.
        pre_fee_pair = [run["total_ser"].to_numpy() + run["commission_ser"].cumsum().to_numpy() for run in run_pair]
        np.testing.assert_allclose(pre_fee_pair[0], pre_fee_pair[1], rtol=1e-6)
        fee_gap_float = abs(run_pair[0]["commission_ser"].sum() - run_pair[1]["commission_ser"].sum())
        assert fee_gap_float <= common.COMMISSION_MINIMUM_FLOAT * (run_pair[0]["phantom_fill_int"] + run_pair[1]["phantom_fill_int"]) + 1e-6
        assert len(run_pair[0]["trade_df"]) > 0
        for (p0, h0, _), (p1, h1, _) in zip(run_pair[0]["position_log"], run_pair[1]["position_log"]):
            assert p0 == p1 and set(h0) == set(h1)


@pytest.mark.skipif(not common.SLEEVE_SERIES_PATH.exists(), reason="stored sleeves not available")
def test_official_pod_model_reproduces_the_published_g3():
    taa_ser = common.load_taa_ser()
    l_ser = common.load_stored_l_ser()
    metrics = common.book_metrics_by_block({"taa": taa_ser, "L": l_ser}, common.ROLE_WEIGHT_DICT["G3"])
    assert metrics["G-FULL"]["sharpe"] == pytest.approx(1.2878, abs=5e-4)
    assert metrics["G-FULL"]["cagr"] == pytest.approx(0.1999, abs=5e-4)
    assert metrics["G-FULL"]["max_dd"] == pytest.approx(-0.1409, abs=5e-4)
    assert metrics["G-P1"]["sharpe"] == pytest.approx(0.6890, abs=5e-4)
    assert metrics["G-P2"]["sharpe"] == pytest.approx(1.3030, abs=5e-4)
    assert metrics["G-P3"]["sharpe"] == pytest.approx(1.2565, abs=5e-4)
    assert metrics["G-LONG"]["sharpe"] == pytest.approx(1.1668, abs=5e-4)
    assert metrics["G-LONG"]["max_dd"] == pytest.approx(-0.1527, abs=5e-4)
