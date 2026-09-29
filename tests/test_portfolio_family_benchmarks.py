from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.research.portfolio_family_20260923 import benchmarks as benchmark_module


def synthetic_price_df() -> pd.DataFrame:
    date_idx = pd.bdate_range("2024-01-29", "2024-03-04")
    column_dict = {}
    for asset_str in ("BIL", "SPY"):
        for field_str in ("Open", "High", "Low", "Close"):
            column_dict[(asset_str, field_str)] = np.full(len(date_idx), 100.0)
        column_dict[(asset_str, "Dividend")] = np.zeros(len(date_idx))
    column_dict[("$SPX", "Close")] = np.linspace(1000.0, 1030.0, len(date_idx))
    frame_df = pd.DataFrame(column_dict, index=date_idx)
    frame_df.attrs["norgate_adjustment_by_symbol_dict"] = {
        "BIL": "CAPITALSPECIAL", "SPY": "CAPITALSPECIAL", "$SPX": "TOTALRETURN",
    }
    frame_df.attrs["benchmark_data_symbol_dict"] = {"$SPX": "$SPXTR"}
    return frame_df


def test_gap_does_not_resize_prior_close_shares_and_first_fill_cost_counts():
    stable_df = synthetic_price_df()
    gap_df = stable_df.copy()
    gap_df.loc[gap_df.index[1], ("SPY", "Open")] = 110.0
    stable_obj = benchmark_module.run_control(stable_df, "SPY", 1000.0)
    gap_obj = benchmark_module.run_control(gap_df, "SPY", 1000.0)
    assert stable_obj._transactions.iloc[0]["amount"] == 9
    assert gap_obj._transactions.iloc[0]["amount"] == 9
    assert gap_obj._transactions.iloc[0]["price"] == pytest.approx(110 * 1.00025)
    assert pd.Timestamp(gap_obj._transactions.iloc[0]["bar"]) == gap_df.index[1]
    assert gap_obj.results.iloc[0]["total_value"] == 1000
    expected_nav_float = 1000 - 9 * 110 * 1.00025 - 1 + 9 * 100
    assert gap_obj.results.iloc[1]["total_value"] == pytest.approx(expected_nav_float)
    assert gap_obj.results.iloc[1]["daily_returns"] == pytest.approx(expected_nav_float / 1000 - 1)
    benchmark_module.validate_account(gap_obj)


def test_dividend_on_month_boundary_waits_until_next_month_to_reinvest():
    price_df = synthetic_price_df()
    # *** CRITICAL *** Dividend at Jan31 entitlement close is paid before
    # Feb1 open after Jan31 sizing; it must not increase that order's cash.
    price_df.loc[pd.Timestamp("2024-01-31"), ("SPY", "Dividend")] = 20.0
    strategy_obj = benchmark_module.run_control(price_df, "SPY", 1000.0)
    transaction_df = strategy_obj._transactions
    assert list(pd.to_datetime(transaction_df["bar"])) == [
        pd.Timestamp("2024-01-30"), pd.Timestamp("2024-03-01"),
    ]
    assert list(transaction_df["amount"]) == [9, 2]
    dividend_df = pd.DataFrame(strategy_obj._dividend_ledger_row_dict_list)
    assert len(dividend_df) == 1
    assert dividend_df.iloc[0]["gross_dividend_cash_float"] == pytest.approx(180)
    assert dividend_df.iloc[0]["withholding_cash_float"] == pytest.approx(45)
    assert dividend_df.iloc[0]["net_dividend_cash_float"] == pytest.approx(135)
    feb_first_dict = next(row_dict for row_dict in strategy_obj.decision_row_list
                          if row_dict["fill_date"] == pd.Timestamp("2024-02-01"))
    assert feb_first_dict["decision_cash_float"] == pytest.approx(98.775)
    assert feb_first_dict["buy_shares_int"] == 0
    benchmark_module.validate_account(strategy_obj)


def test_native_negative_cash_is_preserved_after_unexpected_open_gap():
    price_df = synthetic_price_df()
    price_df.loc[price_df.index[1], ("BIL", "Open")] = 120.0
    strategy_obj = benchmark_module.run_control(price_df, "BIL", 1000.0)
    assert strategy_obj.results.iloc[1]["cash"] < 0
    assert strategy_obj._accounting_policy_dict["negative_cash_financing_policy_str"] == "not_modeled"
    assert strategy_obj._transactions["amount"].gt(0).all()
    assert strategy_obj._accounting_policy_dict["negative_cash_day_count_int"] > 0
    benchmark_module.validate_account(strategy_obj)


@pytest.mark.parametrize("cash_float", [0, -10, 100, 1000, 1000000])
def test_affordability_reserves_slippage_and_minimum_or_per_share_commission(cash_float):
    share_count_int = benchmark_module.affordable_shares_int(cash_float, 100.0)
    cost_fn = lambda quantity_int: quantity_int * 100 * 1.00025 + max(1.0, .005 * quantity_int)
    if share_count_int:
        assert cost_fn(share_count_int) <= cash_float
    assert cost_fn(share_count_int + 1) > cash_float


def test_common_frame_retains_anchor_rejects_holes_and_requires_index_provenance():
    price_df = synthetic_price_df()
    price_df.loc[price_df.index[0], ("BIL", "Open")] = np.nan
    common_df = benchmark_module.common_price_frame_df(price_df)
    assert common_df.index[0] == price_df.index[1]
    damaged_df = common_df.copy()
    damaged_df.loc[damaged_df.index[2], ("BIL", "Close")] = np.nan
    with pytest.raises(ValueError, match="inside common"):
        benchmark_module.common_price_frame_df(damaged_df)
    wrong_basis_df = common_df.copy()
    wrong_basis_df.attrs["benchmark_data_symbol_dict"] = {"$SPX": "$SPX"}
    with pytest.raises(ValueError, match="TOTALRETURN"):
        benchmark_module.common_price_frame_df(wrong_basis_df)
