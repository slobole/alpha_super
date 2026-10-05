"""MR capsule pods: parking plan, gate on entries, parking outside slots/exits, parity with the parent pods."""

from __future__ import annotations

import importlib
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from alpha.bench import catalog  # noqa: E402
from alpha.engine import portfolio_manager  # noqa: E402
from alpha.engine.backtest import run_daily  # noqa: E402
from alpha.engine.metrics import generate_trades_metrics  # noqa: E402
from alpha.engine.portfolio_manager import PortfolioManager  # noqa: E402
from alpha.live import release_manifest, strategy_host  # noqa: E402
from alpha.strategy_registry import MaturityTier, tier_for  # noqa: E402
from strategies.dv2.strategy_mr_dv2 import DVO2Strategy, default_trade_id_int  # noqa: E402
from strategies.hpi.stateful_long import ENTRY_HORIZON_VOTE_STR, TURNOVER_FIELD_STR, HPIStatefulLongStrategy  # noqa: E402
from strategies.mr_capsule import hpi_vote_vix_gated as hpi_capsule_mod  # noqa: E402
from strategies.mr_capsule.capsule_pod import PARKING_TRADE_ID_BASE_INT  # noqa: E402
from strategies.mr_capsule.parking import plan_parking_orders, require_parking_dividends  # noqa: E402
from strategies.mr_capsule.dv2_vix_gated import DV2VixGatedStrategy  # noqa: E402
from strategies.mr_capsule.hpi_vote_vix_gated import HPIVoteVixGatedStrategy  # noqa: E402

PARKING_LIST = ["SPMO", "BIL"]


# ----------------------------------------------------------------------------------------------- parking plan
def _plan(**overrides):
    kwargs = dict(pod_value_float=100_000.0, stock_value_after_orders_float=40_000.0, spmo_held_share_int=0, bil_held_share_int=0,
                  spmo_close_float=100.0, bil_close_float=91.0, gate_open_bool=False, gate_switched_bool=False,
                  week_end_bool=True, spmo_weight_float=0.5)
    kwargs.update(overrides)
    return plan_parking_orders(**kwargs)


def test_gate_open_parks_everything_in_bil_and_sells_spmo():
    plan = _plan(gate_open_bool=True, spmo_held_share_int=120)
    assert plan.spmo_target_share_int == 0
    assert plan.idle_value_float == pytest.approx(100_000 - 40_000 - 1_000)
    assert plan.bil_target_share_int == int(59_000 // 91.0)


def test_gate_closed_week_end_sets_spmo_to_vol_target_and_bil_to_the_rest():
    plan = _plan()
    assert plan.retarget_bool
    assert plan.spmo_target_share_int == int(0.5 * 59_000 // 100.0)
    assert plan.bil_target_share_int == int((59_000 - plan.spmo_target_share_int * 100.0) // 91.0)


def test_mid_week_leaves_spmo_untouched_and_does_not_buy_bil():
    plan = _plan(week_end_bool=False, spmo_held_share_int=200, bil_held_share_int=0)
    assert plan.spmo_target_share_int is None and not plan.retarget_bool
    assert plan.bil_target_share_int is None  # exit cash waits for the weekly sweep


def test_mid_week_sells_bil_to_fund_the_day_s_orders():
    # gate open, BIL holds 59K of idle value; today's entries leave only 29K idle: BIL is sold down the same day
    plan = _plan(gate_open_bool=True, week_end_bool=False, stock_value_after_orders_float=70_000.0, bil_held_share_int=int(59_000 // 91.0))
    assert plan.bil_target_share_int == int(29_000 // 91.0)
    # a shortfall inside the band is left to the cash buffer
    assert _plan(gate_open_bool=True, week_end_bool=False, stock_value_after_orders_float=40_500.0,
                 bil_held_share_int=int(59_000 // 91.0)).bil_target_share_int is None


def test_gate_switch_retargets_immediately():
    closing = _plan(week_end_bool=False, gate_switched_bool=True)
    assert closing.retarget_bool and closing.spmo_target_share_int == int(0.5 * 59_000 // 100.0)
    opening = _plan(gate_open_bool=True, week_end_bool=False, gate_switched_bool=True, spmo_held_share_int=300)
    assert opening.spmo_target_share_int == 0 and opening.bil_target_share_int == int(59_000 // 91.0)


def test_bil_band_skips_small_adjustments():
    wanted_int = int(59_000 // 91.0)
    assert _plan(gate_open_bool=True, bil_held_share_int=wanted_int - 5).bil_target_share_int is None
    assert _plan(gate_open_bool=True, bil_held_share_int=wanted_int - 20).bil_target_share_int == wanted_int


def test_missing_spmo_or_bil_prices():
    pre_spmo = _plan(spmo_close_float=np.nan)
    assert pre_spmo.spmo_target_share_int is None and pre_spmo.bil_target_share_int == int(59_000 // 91.0)
    assert _plan(gate_open_bool=True, bil_close_float=np.nan).bil_target_share_int is None  # 0% cash before BIL
    assert _plan(gate_open_bool=True, bil_close_float=np.nan, bil_held_share_int=3).bil_target_share_int == 0
    assert _plan(spmo_weight_float=0.0, spmo_held_share_int=40, week_end_bool=False).spmo_target_share_int == 0


def test_unknown_stock_value_leaves_parking_untouched():
    plan = _plan(stock_value_after_orders_float=np.nan, bil_held_share_int=500, spmo_held_share_int=100)
    assert plan.spmo_target_share_int is None and plan.bil_target_share_int is None


def test_fully_invested_pod_sells_parking_even_mid_week():
    plan = _plan(gate_open_bool=True, week_end_bool=False, stock_value_after_orders_float=99_500.0, bil_held_share_int=5)
    assert plan.bil_target_share_int == 0


def test_invalid_pod_value_fails_loud():
    with pytest.raises(ValueError):
        _plan(pod_value_float=np.nan)


# ----------------------------------------------------------------------------------------------- synthetic market
def _synthetic_market(symbol_count_int: int, session_count_int: int, seed_int: int, with_parking_bool: bool,
                      drift_float: float = 0.0004, market_vol_float: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed_int)
    date_index = pd.bdate_range("2010-01-04", periods=session_count_int)
    # optional common market factor, so that many stocks dip on the same day (fills the 10 slots)
    market_arr = rng.normal(0.0, market_vol_float, session_count_int) if market_vol_float > 0 else np.zeros(session_count_int)
    frame_dict = {}
    symbol_list = [f"S{i:03d}" for i in range(symbol_count_int)] + (PARKING_LIST if with_parking_bool else [])
    for symbol_str in symbol_list:
        if symbol_str == "BIL":
            close_arr = 91.0 * np.cumprod(1 + rng.normal(0.00008, 0.0002, session_count_int))
        else:
            mean_float = 0.0006 if symbol_str == "SPMO" else drift_float
            close_arr = 50.0 * np.cumprod(1 + market_arr + rng.normal(mean_float, 0.018, session_count_int))
        open_arr = close_arr * (1 + rng.normal(0, 0.004, session_count_int))
        high_arr = np.maximum(open_arr, close_arr) * (1 + np.abs(rng.normal(0, 0.006, session_count_int)))
        low_arr = np.minimum(open_arr, close_arr) * (1 - np.abs(rng.normal(0, 0.006, session_count_int)))
        volume_arr = rng.integers(1_000_000, 5_000_000, session_count_int).astype(float)
        frame_dict.update({(symbol_str, "Open"): open_arr, (symbol_str, "High"): high_arr, (symbol_str, "Low"): low_arr,
                           (symbol_str, "Close"): close_arr, (symbol_str, "Volume"): volume_arr,
                           (symbol_str, "Turnover"): close_arr * volume_arr, (symbol_str, "Unadjusted Close"): close_arr,
                           (symbol_str, "Dividend"): np.zeros(session_count_int)})
    pricing_df = pd.DataFrame(frame_dict, index=date_index)
    pricing_df.columns = pd.MultiIndex.from_tuples(pricing_df.columns)
    return pricing_df


def _stock_universe(pricing_df: pd.DataFrame) -> pd.DataFrame:
    stock_list = [s for s in pricing_df.columns.get_level_values(0).unique() if s not in PARKING_LIST]
    return pd.DataFrame(1, index=pricing_df.index, columns=stock_list)


def _dv2(cls, pricing_df, gate_ser=None, parking_bool=False):
    strategy = cls(name="dv2_test", benchmarks=[], capital_base=100_000.0, slippage=0.00025, commission_per_share=0.005, commission_minimum=1.0)
    strategy.universe_df = _stock_universe(pricing_df)
    strategy.trade_id = 0
    strategy.current_trade = defaultdict(default_trade_id_int)
    if gate_ser is not None:
        strategy.gate_override_ser = gate_ser
        strategy.parking_enabled_bool = parking_bool
    return strategy


def _hpi(cls, pricing_df, gate_ser=None, parking_bool=False):
    strategy = cls(name="hpi_test", benchmarks=[], ranking_field_str=TURNOVER_FIELD_STR, capital_base=100_000.0, slippage=0.00025,
                   entry_mode_str=ENTRY_HORIZON_VOTE_STR)
    strategy.universe_df = _stock_universe(pricing_df)
    if gate_ser is not None:
        strategy.gate_override_ser = gate_ser
        strategy.parking_enabled_bool = parking_bool
    return strategy


def _run(strategy, pricing_df, calendar):
    run_daily(strategy, pricing_df, calendar, show_progress=False, show_signal_progress_bool=False)
    return strategy


def _trades(strategy) -> pd.DataFrame:
    return strategy.get_transactions()[["bar", "asset", "amount", "price"]].reset_index(drop=True)


def _positions(strategy, calendar) -> pd.DataFrame:
    tx = strategy.get_transactions()
    return tx.pivot_table(index="bar", columns="asset", values="amount", aggfunc="sum").reindex(calendar).fillna(0.0).cumsum()


# ----------------------------------------------------------------------------------------------- parity with the parents
@pytest.mark.parametrize("seed_int", [11, 12])
def test_dv2_gated_with_gate_open_and_no_parking_equals_dvo2(seed_int):
    pricing_df = _synthetic_market(8, 420, seed_int, with_parking_bool=False)
    calendar = pricing_df.index[260:]
    parent = _run(_dv2(DVO2Strategy, pricing_df), pricing_df, calendar)
    child = _run(_dv2(DV2VixGatedStrategy, pricing_df, gate_ser=pd.Series(True, index=pricing_df.index)), pricing_df, calendar)
    assert len(_trades(parent)) > 10
    pd.testing.assert_frame_equal(_trades(parent), _trades(child))
    np.testing.assert_allclose(parent.results["total_value"].astype(float), child.results["total_value"].astype(float))


def test_dv2_parity_with_full_slots_and_parking_symbols_in_the_frame():
    pricing_df = _synthetic_market(90, 400, 13, with_parking_bool=True, drift_float=0.0012)
    calendar = pricing_df.index[260:]
    parent = _run(_dv2(DVO2Strategy, pricing_df), pricing_df, calendar)
    child = _run(_dv2(DV2VixGatedStrategy, pricing_df, gate_ser=pd.Series(True, index=pricing_df.index)), pricing_df, calendar)
    assert (_positions(parent, calendar) > 0).sum(axis=1).max() == 10  # slots fill
    pd.testing.assert_frame_equal(_trades(parent), _trades(child))
    np.testing.assert_allclose(parent.results["total_value"].astype(float), child.results["total_value"].astype(float))


def test_hpi_gated_with_gate_open_and_no_parking_equals_hpi():
    pricing_df = _synthetic_market(8, 1500, 21, with_parking_bool=False)
    calendar = pricing_df.index[1330:]
    parent = _run(_hpi(HPIStatefulLongStrategy, pricing_df), pricing_df, calendar)
    child = _run(_hpi(HPIVoteVixGatedStrategy, pricing_df, gate_ser=pd.Series(True, index=pricing_df.index)), pricing_df, calendar)
    assert len(_trades(parent)) > 5
    pd.testing.assert_frame_equal(_trades(parent), _trades(child))
    np.testing.assert_allclose(parent.results["total_value"].astype(float), child.results["total_value"].astype(float))


def test_hpi_parity_with_full_slots_and_parking_symbols_in_the_frame():
    pricing_df = _synthetic_market(150, 1400, 22, with_parking_bool=True, drift_float=0.0010, market_vol_float=0.012)
    calendar = pricing_df.index[1290:]
    parent = _run(_hpi(HPIStatefulLongStrategy, pricing_df), pricing_df, calendar)
    child = _run(_hpi(HPIVoteVixGatedStrategy, pricing_df, gate_ser=pd.Series(True, index=pricing_df.index)), pricing_df, calendar)
    assert (_positions(parent, calendar) > 0).sum(axis=1).max() == 10
    pd.testing.assert_frame_equal(_trades(parent), _trades(child))
    np.testing.assert_allclose(parent.results["total_value"].astype(float), child.results["total_value"].astype(float))


# ----------------------------------------------------------------------------------------------- gate and parking in the engine
def _alternating_gate(index: pd.DatetimeIndex, block_int: int = 40) -> pd.Series:
    return pd.Series((np.arange(len(index)) // block_int) % 2 == 0, index=index)


@pytest.mark.parametrize("pod", ["dv2", "hpi"])
def test_gate_blocks_entries_and_parking_follows_the_gate(pod):
    session_int, start_int = (420, 260) if pod == "dv2" else (1500, 1330)
    pricing_df = _synthetic_market(8, session_int, 31, with_parking_bool=True)
    gate_ser = _alternating_gate(pricing_df.index, 40 if pod == "dv2" else 4)  # HPI holds ~2-4 sessions
    strategy = (_dv2(DV2VixGatedStrategy, pricing_df, gate_ser, True) if pod == "dv2"
                else _hpi(HPIVoteVixGatedStrategy, pricing_df, gate_ser, True))
    calendar = pricing_df.index[start_int:]
    _run(strategy, pricing_df, calendar)
    tx = strategy.get_transactions()
    position_by_bar_df = _positions(strategy, calendar)
    prior_bar = pd.Series(pricing_df.index[pricing_df.index.get_indexer(calendar) - 1], index=calendar)
    gate_at_decision = gate_ser.reindex(prior_bar.values).to_numpy()
    stock_tx = tx[~tx["asset"].isin(PARKING_LIST)]

    def retarget_close(bar_ts) -> bool:
        decision_ts = prior_bar.loc[bar_ts]
        before_ts = pricing_df.index[pricing_df.index.get_loc(decision_ts) - 1]
        week_end_bool = decision_ts.isocalendar()[:2] != bar_ts.isocalendar()[:2]
        return week_end_bool or bool(gate_ser.loc[before_ts]) != bool(gate_ser.loc[decision_ts])

    # *** every stock entry fills after a decision close at which the gate was open; exits still happen when closed
    entry_bar_ser = stock_tx.loc[stock_tx["amount"] > 0, "bar"]
    assert len(entry_bar_ser) > 0
    assert all(bool(gate_ser.loc[prior_bar.loc[b]]) for b in entry_bar_ser)
    assert any(not bool(gate_ser.loc[prior_bar.loc[b]]) for b in stock_tx.loc[stock_tx["amount"] < 0, "bar"])

    # *** SPMO is held only after gate-closed decisions and is zero after every gate-open decision
    assert position_by_bar_df["SPMO"].max() > 0
    assert (position_by_bar_df["SPMO"].to_numpy()[gate_at_decision] == 0).all()
    assert position_by_bar_df["BIL"].max() > 0

    # *** SPMO changes only on re-target closes (or to zero when the gate is open); BIL is bought only on re-target
    # closes; a pure weekly re-weight (gate closed, no switch) does happen
    spmo_bar_list = sorted(set(tx.loc[tx["asset"] == "SPMO", "bar"]))
    assert all(bool(gate_ser.loc[prior_bar.loc[b]]) or retarget_close(b) for b in spmo_bar_list)
    bil_buy_bar_list = sorted(set(tx.loc[(tx["asset"] == "BIL") & (tx["amount"] > 0), "bar"]))
    assert len(bil_buy_bar_list) > 0 and all(retarget_close(b) for b in bil_buy_bar_list)
    assert any(not bool(gate_ser.loc[prior_bar.loc[b]]) and not bool(gate_ser.loc[pricing_df.index[pricing_df.index.get_loc(prior_bar.loc[b]) - 1]])
               for b in spmo_bar_list)

    # *** no short positions, at most 10 stock slots, cash inside the buffer (no material financing)
    assert (position_by_bar_df >= -1e-9).all().all()
    stock_column_list = [c for c in position_by_bar_df.columns if c not in PARKING_LIST]
    assert (position_by_bar_df[stock_column_list] > 0).sum(axis=1).max() <= 10
    total_value_ser = strategy.results["total_value"].astype(float)
    assert total_value_ser.notna().all()
    assert (strategy.results["cash"].astype(float) > -0.02 * total_value_ser).all()
    diagnostic_dict = strategy._accounting_policy_dict
    assert diagnostic_dict["parking_order_count_dict"]["SPMO"] == len(spmo_bar_list)
    assert diagnostic_dict["gate_closed_free_slot_session_count_int"] > 0
    assert diagnostic_dict["parking_one_way_turnover_per_year_dict"]["BIL"] > 0

    # *** trade statistics exclude the parking trades; NAV keeps them
    trade_df = strategy._trades
    stock_trade_df = trade_df[np.asarray(trade_df.index, dtype=float) < PARKING_TRADE_ID_BASE_INT]
    assert len(stock_trade_df) < len(trade_df)
    pd.testing.assert_frame_equal(strategy.summary_trades, generate_trades_metrics(stock_trade_df, strategy.results.index))


# ----------------------------------------------------------------------------------------------- slots and exits
def _hold_parking(strategy, held_ts):
    for trade_id_int, symbol_str in enumerate(PARKING_LIST, start=1):
        strategy.add_transaction(trade_id_int, held_ts, symbol_str, 100, 50.0, 5_000.0, trade_id_int, 0.0)


def _decision_rows(pricing_df, decision_int):
    decision_ts, execution_ts = pricing_df.index[decision_int], pricing_df.index[decision_int + 1]
    return decision_ts, execution_ts, pricing_df.loc[execution_ts].xs("Open", level=1)


def _close_row(row_dict):
    close_row_ser = pd.Series(row_dict, dtype=float)
    close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
    return close_row_ser


def test_dv2_parking_takes_no_slot_and_is_never_exited():
    pricing_df = _synthetic_market(12, 300, 41, with_parking_bool=True)
    strategy = _dv2(DV2VixGatedStrategy, pricing_df, gate_ser=pd.Series(True, index=pricing_df.index), parking_bool=False)
    strategy._prepare_capsule_state()
    decision_ts, execution_ts, open_ser = _decision_rows(pricing_df, 280)
    strategy.previous_bar, strategy.current_bar = decision_ts, execution_ts
    _hold_parking(strategy, pricing_df.index[279])
    row_dict = {}
    for i, symbol_str in enumerate([f"S{i:03d}" for i in range(12)] + PARKING_LIST):
        # every symbol qualifies on the signal fields; SPMO / BIL also close above yesterday's High (stock exit rule)
        close_float = float(pricing_df[(symbol_str, "High")].iloc[279]) * 1.05
        row_dict.update({(symbol_str, "Close"): close_float, (symbol_str, "dv2"): 5.0, (symbol_str, "sma_200"): close_float * 0.9,
                         (symbol_str, "p126d_return"): 0.2, (symbol_str, "natr"): 1.0 + i})
    strategy.iterate(pricing_df.loc[:decision_ts], _close_row(row_dict), open_ser)
    order_asset_list = [o.asset for o in strategy.get_orders()]
    assert not set(order_asset_list) & set(PARKING_LIST)
    assert len(order_asset_list) == 10  # SPMO / BIL held: still 10 stock slots


def test_hpi_parking_takes_no_slot_and_is_never_exited():
    pricing_df = _synthetic_market(12, 300, 42, with_parking_bool=True)
    strategy = _hpi(HPIVoteVixGatedStrategy, pricing_df, gate_ser=pd.Series(True, index=pricing_df.index), parking_bool=False)
    strategy._prepare_capsule_state()
    decision_ts, execution_ts, open_ser = _decision_rows(pricing_df, 280)
    strategy.previous_bar, strategy.current_bar = decision_ts, execution_ts
    _hold_parking(strategy, pricing_df.index[279])
    row_dict = {}
    for i, symbol_str in enumerate([f"S{i:03d}" for i in range(12)] + PARKING_LIST):
        # every symbol qualifies on the vote; SPMO / BIL are not index members (membership exit would fire on a stock)
        row_dict.update({(symbol_str, "Close"): 100.0, (symbol_str, TURNOVER_FIELD_STR): 1e6 * (i + 1),
                         (symbol_str, "sma_200_price_ser"): 90.0, (symbol_str, "ibs_value_ser"): 0.05,
                         (symbol_str, "rsi2_value_ser"): 10.0, (symbol_str, "return_2d_ser"): -0.02,
                         (symbol_str, "return_3d_ser"): -0.03, (symbol_str, "return_5d_ser"): -0.04,
                         (symbol_str, "hpi_2d_ser"): 10.0, (symbol_str, "hpi_value_ser"): 10.0, (symbol_str, "hpi_5d_ser"): 10.0})
    strategy.iterate(pricing_df.loc[:decision_ts], _close_row(row_dict), open_ser)
    order_asset_list = [o.asset for o in strategy.get_orders()]
    assert not set(order_asset_list) & set(PARKING_LIST)
    assert len(order_asset_list) == 10


def test_gate_closed_places_no_entry_orders():
    pricing_df = _synthetic_market(12, 300, 43, with_parking_bool=True)
    strategy = _dv2(DV2VixGatedStrategy, pricing_df, gate_ser=pd.Series(False, index=pricing_df.index), parking_bool=False)
    strategy._prepare_capsule_state()
    decision_ts, execution_ts, open_ser = _decision_rows(pricing_df, 280)
    strategy.previous_bar, strategy.current_bar = decision_ts, execution_ts
    row_dict = {}
    for i in range(12):
        row_dict.update({(f"S{i:03d}", "Close"): 100.0, (f"S{i:03d}", "dv2"): 5.0, (f"S{i:03d}", "sma_200"): 90.0,
                         (f"S{i:03d}", "p126d_return"): 0.2, (f"S{i:03d}", "natr"): 1.0 + i})
    strategy.iterate(pricing_df.loc[:decision_ts], _close_row(row_dict), open_ser)
    assert len(strategy.get_orders()) == 0
    assert strategy._gate_closed_free_slot_session_int == 1


def _fresh_parked_dv2(pricing_df, decision_int):
    """A new strategy object for one decision, as a live host builds it, holding SPMO and BIL at the plan's levels."""
    strategy = _dv2(DV2VixGatedStrategy, pricing_df, gate_ser=pd.Series(False, index=pricing_df.index), parking_bool=True)
    strategy._prepare_capsule_state()
    decision_ts, execution_ts, open_ser = _decision_rows(pricing_df, decision_int)
    strategy.previous_bar, strategy.current_bar = decision_ts, execution_ts
    spmo_close, bil_close = float(pricing_df.loc[decision_ts, ("SPMO", "Close")]), float(pricing_df.loc[decision_ts, ("BIL", "Close")])
    bil_share_int = int((99_000.0 - 100 * spmo_close) // bil_close)  # BIL already holds the rest of the idle value
    strategy.add_transaction(1, pricing_df.index[decision_int - 1], "SPMO", 100, spmo_close, 100 * spmo_close, 1, 0.0)
    strategy.add_transaction(2, pricing_df.index[decision_int - 1], "BIL", bil_share_int, bil_close, bil_share_int * bil_close, 2, 0.0)
    strategy.iterate(pricing_df.loc[:decision_ts], pricing_df.loc[decision_ts], open_ser)
    return [o.asset for o in strategy.get_orders()]


def test_a_fresh_object_retargets_only_on_the_calendar_and_gate_triggers():
    pricing_df = _synthetic_market(3, 300, 44, with_parking_bool=True)
    index = pricing_df.index
    mid_week_int = next(i for i in range(262, 290) if index[i].dayofweek == 1 and index[i + 1].dayofweek == 2)
    week_end_int = next(i for i in range(262, 290) if index[i].dayofweek == 4)
    assert _fresh_parked_dv2(pricing_df, mid_week_int) == []  # no stored "first decision": nothing to do mid-week
    assert "SPMO" in _fresh_parked_dv2(pricing_df, week_end_int)  # 100 shares is off the vol target: re-weighted


def test_dividend_ledger_is_enabled_and_fails_loud_without_dividends():
    pricing_df = _synthetic_market(3, 300, 45, with_parking_bool=True)
    gate_ser = pd.Series(False, index=pricing_df.index)
    strategy = _dv2(DV2VixGatedStrategy, pricing_df, gate_ser, True)
    assert strategy._accounting_policy_dict["dividend_cash_ledger_mode_str"] == "enabled"
    assert _hpi(HPIVoteVixGatedStrategy, pricing_df, gate_ser, True)._accounting_policy_dict["dividend_cash_ledger_mode_str"] == "enabled"
    no_dividend_df = pricing_df.drop(columns=[c for c in pricing_df.columns if c[1] == "Dividend"])
    with pytest.raises(RuntimeError, match="Dividend"):
        _run(_dv2(DV2VixGatedStrategy, no_dividend_df, gate_ser, True), no_dividend_df, no_dividend_df.index[260:])


def test_gate_switch_is_read_from_the_gate_series():
    index = pd.bdate_range("2020-01-01", periods=8)
    gate_ser = pd.Series([False, False, True, True, True, False, False, False], index=index)
    strategy = _dv2(DV2VixGatedStrategy, _synthetic_market(2, 8, 46, True), gate_ser=gate_ser, parking_bool=True)
    strategy._prepare_capsule_state()
    expected_list = [False, False, True, False, False, True, False, False]  # opening close, then closing close
    for decision_ts, expected_bool in zip(index, expected_list):
        strategy.previous_bar = decision_ts
        assert strategy._gate_switched_at_decision(strategy._gate_open_at_decision(), pd.DataFrame(index=index)) is expected_bool
    strategy.previous_bar = pd.Timestamp("2019-12-20")  # before the first gate row
    assert strategy._gate_switched_at_decision(False, pd.DataFrame(index=index)) is False


def test_gate_switch_uses_previous_pricing_session_when_vix_is_stale():
    pricing_df = _synthetic_market(2, 8, 48, with_parking_bool=True)
    calendar_idx = pricing_df.index
    gate_ser = pd.Series([False, True], index=calendar_idx[:2])
    strategy_obj = _dv2(DV2VixGatedStrategy, pricing_df, gate_ser, True)
    strategy_obj._prepare_capsule_state()
    strategy_obj.previous_bar = calendar_idx[2]
    # *** CRITICAL *** the opening close was yesterday, not the stale current decision close.
    assert strategy_obj._gate_open_at_decision() is True
    assert strategy_obj._gate_switched_at_decision(True, pricing_df.loc[:calendar_idx[2]]) is False
    strategy_obj.require_current_gate_observation_bool = True
    with pytest.raises(RuntimeError, match="no observation for decision close"):
        strategy_obj._gate_open_at_decision()


def test_gate_switch_calendar_matches_complete_history():
    pricing_df = _synthetic_market(2, 80, 49, with_parking_bool=True)
    gate_ser = _alternating_gate(pricing_df.index, block_int=7)
    strategy_obj = _dv2(DV2VixGatedStrategy, pricing_df, gate_ser, True)
    strategy_obj._prepare_capsule_state()
    for session_int, decision_date_ts in enumerate(pricing_df.index):
        strategy_obj.previous_bar = decision_date_ts
        gate_open_bool = strategy_obj._gate_open_at_decision()
        expected_bool = session_int > 0 and bool(gate_ser.iloc[session_int - 1]) != gate_open_bool
        assert strategy_obj._gate_switched_at_decision(gate_open_bool, pricing_df.loc[:decision_date_ts]) == expected_bool


@pytest.mark.parametrize("pod_str", ["dv2", "hpi"])
def test_missing_parking_open_preserves_holding_and_cancels_order(pod_str):
    pricing_df = _synthetic_market(1, 3, 50, with_parking_bool=True)
    gate_ser = pd.Series(False, index=pricing_df.index)
    strategy_obj = (_dv2(DV2VixGatedStrategy, pricing_df, gate_ser, True) if pod_str == "dv2"
                    else _hpi(HPIVoteVixGatedStrategy, pricing_df, gate_ser, True))
    strategy_obj._prepare_capsule_state()
    decision_date_ts, execution_date_ts = pricing_df.index[:2]
    strategy_obj.previous_bar, strategy_obj.current_bar = decision_date_ts, execution_date_ts
    strategy_obj.universe_df.loc[execution_date_ts:, "S000"] = 0
    for trade_id_int, symbol_str in enumerate(["S000", "BIL", "SPMO"], start=1):
        close_float = float(pricing_df.loc[decision_date_ts, (symbol_str, "Close")])
        strategy_obj.add_transaction(trade_id_int, decision_date_ts, symbol_str, 10, close_float, 10 * close_float, trade_id_int, 0.0)
        pricing_df.loc[execution_date_ts, [(symbol_str, field_str) for field_str in ("Open", "High", "Low")]] = np.nan
        pricing_df.loc[execution_date_ts, (symbol_str, "Close")] = close_float  # disclosed prior-close valuation
        strategy_obj.order_target(symbol_str, 0, trade_id=trade_id_int)
    original_pricing_df = pricing_df.copy(deep=True)
    strategy_obj.process_orders(pricing_df)
    assert strategy_obj.get_position("S000") == 0  # ordinary removed-stock semantics are unchanged
    assert strategy_obj.get_position("BIL") == 10
    assert strategy_obj.get_position("SPMO") == 10
    assert strategy_obj.get_orders() == []
    transaction_df = strategy_obj.get_transactions()
    assert transaction_df.loc[transaction_df["amount"] < 0, "asset"].tolist() == ["S000"]
    pd.testing.assert_frame_equal(pricing_df, original_pricing_df)


@pytest.mark.parametrize("pod_str", ["dv2", "hpi"])
def test_parent_missing_price_liquidation_still_includes_nonmember_etfs(pod_str):
    pricing_df = _synthetic_market(1, 3, 51, with_parking_bool=True)
    strategy_obj = _dv2(DVO2Strategy, pricing_df) if pod_str == "dv2" else _hpi(HPIStatefulLongStrategy, pricing_df)
    decision_date_ts, execution_date_ts = pricing_df.index[:2]
    strategy_obj.previous_bar, strategy_obj.current_bar = decision_date_ts, execution_date_ts
    close_float = float(pricing_df.loc[decision_date_ts, ("BIL", "Close")])
    strategy_obj.add_transaction(1, decision_date_ts, "BIL", 10, close_float, 10 * close_float, 1, 0.0)
    pricing_df.loc[execution_date_ts, ("BIL", "Open")] = np.nan
    strategy_obj.process_orders(pricing_df)
    assert strategy_obj.get_position("BIL") == 0
    assert strategy_obj.get_transactions().iloc[-1]["order_id"] == -1


def test_missing_parking_close_fails_instead_of_inventing_a_valuation():
    pricing_df = _synthetic_market(1, 3, 52, with_parking_bool=True)
    strategy_obj = _dv2(DV2VixGatedStrategy, pricing_df, pd.Series(False, index=pricing_df.index), True)
    strategy_obj._prepare_capsule_state()
    decision_date_ts, execution_date_ts = pricing_df.index[:2]
    strategy_obj.previous_bar, strategy_obj.current_bar = decision_date_ts, execution_date_ts
    close_float = float(pricing_df.loc[decision_date_ts, ("BIL", "Close")])
    strategy_obj.add_transaction(1, decision_date_ts, "BIL", 10, close_float, 10 * close_float, 1, 0.0)
    pricing_df.loc[execution_date_ts, ("BIL", "Close")] = np.nan
    with pytest.raises(RuntimeError, match="Active positions still contain missing close prices"):
        strategy_obj.process_orders(pricing_df)


def test_parking_etfs_must_carry_dividends():
    pricing_df = _synthetic_market(2, 30, 47, with_parking_bool=True)
    require_parking_dividends(pricing_df)
    with pytest.raises(RuntimeError, match="BIL"):
        require_parking_dividends(pricing_df.drop(columns=[("BIL", "Dividend")]))


# ----------------------------------------------------------------------------------------------- data plumbing
def test_append_parking_prices_keeps_padding_dividends_and_attrs(monkeypatch):
    calendar = pd.bdate_range("2020-01-01", periods=8)
    base_df = pd.DataFrame({("AAA", "Close"): np.arange(8, dtype=float) + 10.0}, index=calendar)
    base_df.columns = pd.MultiIndex.from_tuples(base_df.columns)
    base_df.attrs = {"norgate_adjustment_by_symbol_dict": {"AAA": "CAPITALSPECIAL"}, "benchmark_data_symbol_dict": {}}

    def fake_loader(symbol_str, **_kwargs):
        listed_idx = calendar[3:]  # listed on the 4th session; one padded no-trade session after listing
        frame = pd.DataFrame({"Open": 50.0, "High": 51.0, "Low": 49.0, "Close": 50.5, "Volume": 1_000.0, "Dividend": 0.0}, index=listed_idx)
        frame.loc[calendar[5], ["Open", "High", "Low", "Close", "Volume"]] = [50.5, 50.5, 50.5, 50.5, 0.0]
        frame.loc[calendar[6], "Dividend"] = 0.2
        return frame

    monkeypatch.setattr(hpi_capsule_mod, "load_price_timeseries", fake_loader)
    out_df = hpi_capsule_mod.append_parking_prices(base_df, "2020-01-01", None)
    assert out_df.attrs == base_df.attrs
    for symbol_str in PARKING_LIST:
        assert out_df[(symbol_str, "Close")].iloc[:3].isna().all()  # no price before listing (no back-fill)
        assert (out_df[(symbol_str, "Dividend")].iloc[:3] == 0.0).all()  # unobserved rows carry no dividend
        assert out_df[(symbol_str, "Open")].iloc[5] == 50.5  # padded no-trade session keeps a tradable Open
        assert out_df[(symbol_str, "Dividend")].iloc[6] == 0.2


# ----------------------------------------------------------------------------------------------- Bench entry points
ENTRY_POINT_LIST = [
    ("strategy_mr_dv2_vix_gated_spmo", "dv2", True, True),
    ("strategy_mr_dv2_vix_gated_bil", "dv2", True, False),
    ("strategy_mr_dv2_vix_gated_cash", "dv2", False, False),
    ("strategy_mr_hpi_vote_vix_gated_spmo", "hpi", True, True),
    ("strategy_mr_hpi_vote_vix_gated_bil", "hpi", True, False),
    ("strategy_mr_hpi_vote_vix_gated_cash", "hpi", False, False),
]


@pytest.mark.parametrize("module_name_str,pod_str,parking_bool,spmo_bool", ENTRY_POINT_LIST)
def test_bench_entry_points_fix_the_parking_and_the_name(monkeypatch, module_name_str, pod_str, parking_bool, spmo_bool):
    module_obj = importlib.import_module(f"strategies.mr_capsule.{module_name_str}")
    captured_dict = {}
    monkeypatch.setattr(module_obj, f"run_{pod_str}_capsule_pod", lambda **kwarg_dict: captured_dict.update(kwarg_dict) or "ran")
    assert module_obj.run_variant(show_display_bool=False, save_results_bool=False, end_date_str="2026-01-02") == "ran"
    assert captured_dict["parking_enabled_bool"] is parking_bool and captured_dict["spmo_parking_enabled_bool"] is spmo_bool
    assert captured_dict["strategy_name_str"] == module_name_str == module_obj.STRATEGY_NAME_STR
    assert captured_dict["end_date_str"] == "2026-01-02" and captured_dict["save_results_bool"] is False


@pytest.mark.parametrize("module_name_str,pod_str,parking_bool,spmo_bool", ENTRY_POINT_LIST)
def test_entry_point_analysis_hooks_keep_the_fixed_parking_and_name(monkeypatch, module_name_str, pod_str, parking_bool, spmo_bool):
    """WIRED modules expose the capacity and timing hooks with the same identity as run_variant."""
    module_obj = importlib.import_module(f"strategies.mr_capsule.{module_name_str}")
    capacity_dict, timing_dict = {}, {}
    monkeypatch.setattr(module_obj, f"build_{pod_str}_capsule_capacity_analysis_inputs",
                        lambda **kwarg_dict: capacity_dict.update(kwarg_dict) or "capacity")
    monkeypatch.setattr(module_obj, f"build_{pod_str}_capsule_execution_timing_analysis_inputs",
                        lambda **kwarg_dict: timing_dict.update(kwarg_dict) or "timing")
    assert module_obj.build_capacity_analysis_inputs(capital_base_float=250_000.0, end_date_str="2026-01-02") == "capacity"
    assert module_obj.build_execution_timing_analysis_inputs() == "timing"
    for captured_dict in (capacity_dict, timing_dict):
        assert captured_dict["strategy_name_str"] == module_obj.STRATEGY_NAME_STR
        assert captured_dict["parking_enabled_bool"] is parking_bool and captured_dict["spmo_parking_enabled_bool"] is spmo_bool
    assert capacity_dict["capital_base_float"] == 250_000.0 and capacity_dict["end_date_str"] == "2026-01-02"
    assert capacity_dict["backtest_start_date_str"] == "2004-01-01" and capacity_dict["show_display_bool"] is False


@pytest.mark.parametrize("pod_str", ["dv2", "hpi"])
def test_live_adapter_and_backtest_share_one_strategy_builder(pod_str):
    """The live adapter must build the pod with the backtest's builder, not its own constructor call."""
    import inspect

    from alpha.live import mr_capsule_adapter

    source_str = inspect.getsource(mr_capsule_adapter.build_mr_capsule_decision_plan)
    builder_str = f"build_{pod_str}_capsule_strategy"
    assert builder_str in source_str
    assert "VixGatedStrategy(" not in source_str
    pod_module_obj = importlib.import_module(
        "strategies.mr_capsule.dv2_vix_gated" if pod_str == "dv2" else "strategies.mr_capsule.hpi_vote_vix_gated"
    )
    assert builder_str in inspect.getsource(getattr(pod_module_obj, f"run_{pod_str}_capsule_pod"))


@pytest.mark.parametrize("module_name_str", [entry[0] for entry in ENTRY_POINT_LIST])
def test_capsule_pods_have_matching_wired_routes(module_name_str):
    """A WIRED capsule identity must have a real host and release route."""
    module_str = f"strategies.mr_capsule.{module_name_str}"
    entry_obj = catalog.get_strategy_by_module(module_str)
    assert entry_obj is not None and entry_obj.has_run_variant_bool and entry_obj.is_wired_bool
    # Bench offers no parking switch: the parking is fixed by the module (and so is the results name)
    assert not {p.name_str for p in entry_obj.run_variant_param_tuple} & {"parking_enabled_bool", "spmo_parking_enabled_bool"}
    assert tier_for(module_str) is MaturityTier.WIRED
    assert module_str in portfolio_manager.SUPPORTED_STRATEGY_IMPORT_TUPLE
    assert module_str in release_manifest.SUPPORTED_STRATEGY_IMPORT_TUPLE
    assert module_str in strategy_host.INCREMENTAL_DECISION_STRATEGY_IMPORT_SET


@pytest.mark.parametrize("suffix_str", ["spmo", "bil"])
def test_capsule_books_load_in_the_portfolio_manager(suffix_str):
    config_obj = PortfolioManager.from_yaml(Path(__file__).resolve().parents[1] / "portfolios" / f"mr_capsule_{suffix_str}.yaml").config
    assert [pod_obj.strategy_import_str for pod_obj in config_obj.pod_config_list] == [
        f"strategies.mr_capsule.strategy_mr_dv2_vix_gated_{suffix_str}",
        f"strategies.mr_capsule.strategy_mr_hpi_vote_vix_gated_{suffix_str}",
    ]
    assert config_obj.weight_list == [0.5, 0.5]
    assert config_obj.rebalance.frequency_str == "annually" and config_obj.max_workers_int == 1


def test_shared_pod_modules_are_not_bench_strategies():
    for module_str in ("strategies.mr_capsule.dv2_vix_gated", "strategies.mr_capsule.hpi_vote_vix_gated"):
        assert catalog.get_strategy_by_module(module_str) is None
