from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from scripts.research.portfolio_family_20260923 import core_replay as replay_module


def synthetic_core_price_df() -> pd.DataFrame:
    date_idx = pd.bdate_range("2020-01-02", periods=330)
    bar_vec = np.arange(len(date_idx), dtype=float)
    column_dict = {}
    for asset_number_int, asset_str in enumerate(("SPY", "IEF", "GLD", "DBC", "UUP", "BIL")):
        close_vec = 100 + .03*bar_vec + (6*np.sin(bar_vec/13+asset_number_int) if asset_str != "BIL" else 0)
        for field_str in ("Open", "High", "Low", "Close"):
            column_dict[(asset_str, field_str)] = close_vec.copy()
        column_dict[(asset_str, "Dividend")] = np.zeros(len(date_idx))
        if asset_str != "BIL":
            column_dict[("ADAPTIVE_TR_"+asset_str, "Close")] = close_vec*2
    for field_str in ("Open", "High", "Low", "Close"):
        column_dict[("$SPX", field_str)] = 1000+bar_vec
    frame_df = pd.DataFrame(column_dict, index=date_idx)
    frame_df.attrs["norgate_adjustment_by_symbol_dict"] = {
        **{asset_str:"CAPITALSPECIAL" for asset_str in ("SPY","IEF","GLD","DBC","UUP","BIL")},
        "$SPX":"TOTALRETURN",
    }
    frame_df.attrs["benchmark_data_symbol_dict"] = {"$SPX":"$SPXTR"}
    return frame_df


def test_collateral_rounding_offsets_cash_and_includes_negative_cash():
    position_ser = pd.Series({"DBC": -100.0, "BIL": 200.0})
    price_ser = pd.Series({"DBC": 100.01, "BIL": 100.0})
    # ceil(102.0102)=103 per share, not rounding only after aggregation.
    assert replay_module.funding_amount_tuple(10000, position_ser, price_ser) == (300, 10300)
    assert replay_module.funding_amount_tuple(-50, pd.Series(dtype=float), pd.Series(dtype=float)) == (50, 0)
    assert replay_module.funding_amount_tuple(11000, position_ser, price_ser) == (0, 10300)


def test_weekend_funding_debits_cash_nav_and_terminal_has_no_interval():
    config_obj = replace(replay_module.core_module.DEFAULT_CONFIG, capital_base_float=100000.0)
    strategy_obj = replay_module.FinancedCore5Strategy(config_obj, .05)
    date_idx = pd.DatetimeIndex(["2024-01-05", "2024-01-08"])
    strategy_obj.configure_run_calendar(date_idx)
    strategy_obj.current_bar = date_idx[0]
    strategy_obj.cash = 10000
    strategy_obj.total_value = 100000
    strategy_obj._position_amount_map = {"DBC": -100.0}
    strategy_obj._latest_close_price_ser = pd.Series({"DBC":100.01})
    strategy_obj.apply_financing()
    fee_float = 300*.05*3/360
    assert strategy_obj.cash == pytest.approx(10000-fee_float)
    assert strategy_obj.total_value == pytest.approx(100000-fee_float)
    assert strategy_obj.financing_row_list[-1]["funding_fee_float"] == pytest.approx(fee_float)
    strategy_obj.current_bar = date_idx[1]
    cash_before_float = strategy_obj.cash
    strategy_obj.apply_financing()
    assert strategy_obj.cash == cash_before_float
    assert strategy_obj.financing_row_list[-1]["calendar_days_int"] == 0


def test_zero_funding_matches_native_engine_and_borrow_exactly():
    price_df = synthetic_core_price_df()
    case_dict = dict(replay_module.CASE_LIST[4], capital_float=100000.0)
    start_str, end_str = str(price_df.index[250].date()), str(price_df.index[-1].date())
    financed_obj = replay_module.run_case(price_df, case_dict, start_str=start_str, end_str=end_str)
    native_obj = replay_module.run_case(price_df, case_dict, plain_native_bool=True, start_str=start_str, end_str=end_str)
    replay_module.assert_native_parity(financed_obj, native_obj)
    assert len(financed_obj.borrow_fee_df) > 0
    assert financed_obj.borrow_fee_total_float == native_obj.borrow_fee_total_float
    nav_df = replay_module.anchored_nav_df(financed_obj, str(price_df.index[249].date()))
    assert nav_df.iloc[0]["cash"] == 100000
    assert nav_df.iloc[0]["portfolio_value"] == 0
    assert nav_df.iloc[1]["daily_returns"] == pytest.approx(nav_df.iloc[1]["total_value"]/100000-1)


def test_open_gap_never_changes_first_close_fixed_share_decision():
    price_df = synthetic_core_price_df()
    gapped_df = price_df.copy()
    first_ts = price_df.index[250]
    for asset_str in ("SPY","IEF","GLD","DBC","UUP","BIL"):
        gapped_df.loc[first_ts, (asset_str,"Open")] *= 1.1
    case_dict = dict(replay_module.CASE_LIST[0], capital_float=100000.0)
    start_str, end_str = str(first_ts.date()), str(price_df.index[258].date())
    base_obj = replay_module.run_case(price_df, case_dict, start_str=start_str, end_str=end_str)
    gap_obj = replay_module.run_case(gapped_df, case_dict, start_str=start_str, end_str=end_str)
    base_transaction_df = base_obj._transactions.loc[pd.to_datetime(base_obj._transactions["bar"]) == first_ts]
    gap_transaction_df = gap_obj._transactions.loc[pd.to_datetime(gap_obj._transactions["bar"]) == first_ts]
    assert list(base_transaction_df["amount"]) == list(gap_transaction_df["amount"])
    assert base_transaction_df["price"].tolist() != gap_transaction_df["price"].tolist()
    assert gap_obj.financing_row_list[0]["funding_fee_float"] > 0


def test_funding_changes_next_sizing_nav_without_changing_signal_targets():
    price_df = synthetic_core_price_df()
    zero_dict = dict(replay_module.CASE_LIST[4], capital_float=100000.0)
    high_dict = dict(zero_dict, funding_rate_float=5.0)  # Synthetic magnification crosses whole-share rounding.
    start_str, end_str = str(price_df.index[250].date()), str(price_df.index[-1].date())
    zero_obj = replay_module.run_case(price_df, zero_dict, start_str=start_str, end_str=end_str)
    high_obj = replay_module.run_case(price_df, high_dict, start_str=start_str, end_str=end_str)
    pd.testing.assert_frame_equal(zero_obj.daily_target_weights, high_obj.daily_target_weights)
    assert sum(row_dict["funding_fee_float"] for row_dict in high_obj.financing_row_list) > 0
    assert high_obj._transactions["amount"].tolist() != zero_obj._transactions["amount"].tolist()
    ledger_df = pd.DataFrame(high_obj.financing_row_list)
    assert np.allclose(ledger_df["nav_before_funding_float"]-ledger_df["funding_fee_float"],
                       ledger_df["nav_after_funding_float"])
