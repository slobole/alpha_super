"""Synthetic regression checks for the frozen comparison and scalar oracle."""
import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from alpha.engine.backtest import run_daily
from scripts.research.run_qqq_golden_cross_baseline import (
    QqqBuyHoldComparison, independent_account, performance_metrics, verify_account,
)
from strategies.momentum.strategy_mo_qqq_golden_cross import QqqGoldenCrossStrategy


def make_prices():
    close_vec = np.array([100.0] * 200 + [120, 100, 80, 80, 100, 120, 120, 100, 100])
    session_idx = xcals.get_calendar("XNYS", start="2022-01-01", end="2023-01-01").sessions[:len(close_vec)]
    pricing_df = pd.DataFrame({
        ("QQQ", "Open"): close_vec, ("QQQ", "High"): close_vec + 1,
        ("QQQ", "Low"): close_vec - 1, ("QQQ", "Close"): close_vec,
        ("QQQ", "Volume"): 1_000_000.0, ("QQQ", "Dividend"): 0.0,
    }, index=session_idx)
    pricing_df.loc[session_idx[201], ("QQQ", "Open")] = 135.0
    pricing_df.loc[session_idx[201], ("QQQ", "High")] = 136.0
    pricing_df.loc[session_idx[202], ("QQQ", "Dividend")] = 1.0
    pricing_df.attrs["norgate_adjustment_by_symbol_dict"] = {"QQQ": "CAPITALSPECIAL"}
    pricing_df.attrs["price_padding_policy_str"] = "NONE"
    return pricing_df


@pytest.mark.parametrize("buy_hold_bool", [False, True])
def test_independent_account_matches_engine_with_gap_costs_and_dividends(buy_hold_bool):
    pricing_df = make_prices()
    strategy_obj = QqqBuyHoldComparison(10_000) if buy_hold_bool else QqqGoldenCrossStrategy(10_000)
    run_daily(
        strategy_obj, pricing_df, calendar=pricing_df.index[200:],
        show_progress=False, show_signal_progress_bool=False,
    )
    oracle_df, verdict_dict = verify_account(strategy_obj, pricing_df, 200, buy_hold_bool)
    assert verdict_dict["passed"]
    assert oracle_df.iloc[0]["total_value"] == 10_000
    assert oracle_df.iloc[1]["cash"] < 0
    assert oracle_df.iloc[3]["cash"] - oracle_df.iloc[2]["cash"] == pytest.approx(83 * 0.75)
    fill_df = strategy_obj.get_transactions()
    assert fill_df.iloc[0]["bar"] == pricing_df.index[201]
    assert fill_df.iloc[0]["amount"] == 83
    if buy_hold_bool:
        assert len(fill_df) == 1
        assert strategy_obj.get_position("QQQ") == 83
    else:
        assert list(fill_df["amount"].map(np.sign)) == [1, -1, 1]


def test_subperiod_includes_return_from_previous_boundary_nav():
    daily_df = pd.DataFrame({
        "total_value": [99.0, 108.9], "portfolio_value": [99.0, 108.9],
    }, index=pd.to_datetime(["2020-01-02", "2020-01-03"]))
    result_dict = performance_metrics(daily_df, 110.0, pd.Timestamp("2019-12-31"))
    assert result_dict["total_return_pct"] == pytest.approx(-1.0)
    assert result_dict["max_drawdown_pct"] == pytest.approx(-10.0)
    assert result_dict["volatility_pct"] == pytest.approx(np.std([-0.1, 0.1], ddof=1) * np.sqrt(252) * 100)
    assert result_dict["sharpe"] == pytest.approx(0.0, abs=1e-12)


def test_scalar_oracle_preserves_flat_start_in_existing_bull_trend():
    pricing_df = make_prices()
    for field_str in ("Open", "High", "Low", "Close"):
        pricing_df[("QQQ", field_str)] = np.arange(len(pricing_df), dtype=float) + 100
    oracle_df, fill_df = independent_account(pricing_df, 200, 10_000, False)
    assert fill_df.empty
    assert oracle_df["total_value"].eq(10_000).all()
