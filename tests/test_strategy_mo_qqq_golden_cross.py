"""Synthetic-only checks for the fixed QQQ Golden Cross contract."""

from types import SimpleNamespace
from unittest.mock import Mock

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from strategies.momentum import strategy_mo_qqq_golden_cross as strategy_module


def make_prices(close_vec=None):
    if close_vec is None:
        close_vec = np.array([100.0] * 200 + [120, 100, 80, 80, 100, 120, 120, 100, 100])
    close_vec = np.asarray(close_vec, dtype=float)
    calendar_obj = xcals.get_calendar("XNYS", start="2022-01-01", end="2025-01-01")
    session_idx = calendar_obj.sessions[:len(close_vec)]
    pricing_df = pd.DataFrame({
        ("QQQ", "Open"): close_vec,
        ("QQQ", "High"): close_vec + 1,
        ("QQQ", "Low"): close_vec - 1,
        ("QQQ", "Close"): close_vec,
        ("QQQ", "Volume"): np.full(len(close_vec), 1_000_000.0),
        ("QQQ", "Dividend"): np.zeros(len(close_vec)),
    }, index=session_idx)
    # Explicit synthetic fixture metadata; these are not historical observations.
    pricing_df.attrs["norgate_adjustment_by_symbol_dict"] = {"QQQ": "CAPITALSPECIAL"}
    pricing_df.attrs["price_padding_policy_str"] = "NONE"
    return pricing_df


def run_synthetic(pricing_df, start_position_int=200):
    return strategy_module.run_variant(
        history_start_date_str=str(pricing_df.index[0].date()),
        backtest_start_date_str=str(pricing_df.index[start_position_int].date()),
        end_date_str=str(pricing_df.index[-1].date()),
        capital_base_float=10_000.0, pricing_data_df=pricing_df,
        show_display_bool=False, save_results_bool=False,
    )


def test_rolling_means_match_arithmetic_and_first_cross_requires_201_closes():
    pricing_df = make_prices()
    signal_df = strategy_module.QqqGoldenCrossStrategy(10_000).compute_signals(pricing_df)
    assert signal_df[("QQQ", "sma_200")].iloc[:199].isna().all()
    assert not signal_df[("QQQ", "entry_cross")].iloc[:200].any()
    assert signal_df[("QQQ", "entry_cross")].iloc[200]
    for position_int in range(199, len(pricing_df)):
        price_vec = pricing_df[("QQQ", "Close")].to_numpy()
        assert signal_df[("QQQ", "sma_50")].iloc[position_int] == pytest.approx(
            sum(price_vec[position_int - 49:position_int + 1]) / 50
        )
        assert signal_df[("QQQ", "sma_200")].iloc[position_int] == pytest.approx(
            sum(price_vec[position_int - 199:position_int + 1]) / 200
        )


def test_equality_does_not_trade_and_next_open_fills_share_orders():
    pricing_df = make_prices()
    pricing_df.loc[pricing_df.index[201], ("QQQ", "Open")] = 135.0
    pricing_df.loc[pricing_df.index[201], ("QQQ", "High")] = 136.0
    pricing_df.loc[pricing_df.index[204], ("QQQ", "Open")] = 77.0
    pricing_df.loc[pricing_df.index[204], ("QQQ", "Low")] = 76.0
    strategy_obj = run_synthetic(pricing_df)
    transaction_df = strategy_obj.get_transactions()
    assert list(transaction_df["bar"]) == list(pricing_df.index[[201, 204, 207]])
    assert list(transaction_df["trade_id"]) == [1, 1, 2]
    first_order_ser = transaction_df.iloc[0]
    assert first_order_ser["amount"] == 83  # floor(10000 / Close_T=120), not Open=135.
    assert first_order_ser["price"] == pytest.approx(135 * 1.00025)
    assert first_order_ser["commission"] == 1.0
    assert transaction_df.iloc[1]["amount"] == -83
    assert transaction_df.iloc[1]["price"] == pytest.approx(77 * 0.99975)
    # The 100% target is deliberately NOT a hard no-margin guarantee.
    assert strategy_obj.results.loc[pricing_df.index[201], "cash"] < 0


def test_pre_start_cross_is_not_carried_into_the_run():
    pricing_df = make_prices()
    strategy_obj = run_synthetic(pricing_df, start_position_int=201)
    assert list(strategy_obj.get_transactions()["bar"]) == [pricing_df.index[207]]


def test_initially_bullish_history_stays_in_cash_until_a_new_cross():
    pricing_df = make_prices(np.arange(1.0, 220.0) + 100.0)
    strategy_obj = run_synthetic(pricing_df)
    assert strategy_obj.get_transactions().empty
    assert strategy_obj.cash == 10_000.0


@pytest.mark.parametrize("position_float,entry_bool,exit_bool,expected_orders_int", [
    (0, False, True, 0), (10, True, False, 0),
    (0, False, False, 0), (10, False, False, 0),
    (0, True, False, 1), (10, False, True, 1),
])
def test_position_guards_prevent_shorting_and_duplicate_entries(
    position_float, entry_bool, exit_bool, expected_orders_int,
):
    strategy_obj = strategy_module.QqqGoldenCrossStrategy(10_000)
    strategy_obj.previous_bar = pd.Timestamp("2022-11-01")
    strategy_obj.current_bar = pd.Timestamp("2022-11-02")
    if position_float:
        strategy_obj.trade_id_int = 1
        strategy_obj.add_transaction(1, strategy_obj.previous_bar, "QQQ", position_float, 100, 1000, 1)
    close_ser = pd.Series({
        ("QQQ", "Close"): 100.0, ("QQQ", "entry_cross"): entry_bool,
        ("QQQ", "exit_cross"): exit_bool,
    })
    strategy_obj.iterate(pd.DataFrame(), close_ser, pd.Series({"QQQ": 9999.0}))
    assert len(strategy_obj.get_orders()) == expected_orders_int


def test_future_prices_do_not_change_past_signals_and_prefix_audit_passes():
    pricing_df = make_prices()
    strategy_obj = strategy_module.QqqGoldenCrossStrategy(10_000)
    signal_df = strategy_obj.compute_signals(pricing_df)
    altered_df = pricing_df.copy()
    altered_df.loc[altered_df.index[204]:, ("QQQ", "Close")] *= 10
    altered_signal_df = strategy_obj.compute_signals(altered_df)
    pd.testing.assert_frame_equal(signal_df.iloc[:204], altered_signal_df.iloc[:204])
    strategy_obj.audit_signals(pricing_df, signal_df, sample_size=len(pricing_df))


@pytest.mark.parametrize("field_str,bad_value_float", [
    ("Close", np.nan), ("Close", np.inf), ("Close", 0.0),
    ("Open", np.nan), ("Volume", 0.0), ("Dividend", np.nan),
])
def test_bad_observations_are_rejected_without_filling(field_str, bad_value_float):
    pricing_df = make_prices()
    pricing_df.loc[pricing_df.index[100], ("QQQ", field_str)] = bad_value_float
    with pytest.raises(ValueError, match="observations|positive"):
        strategy_module.QqqGoldenCrossStrategy(10_000).compute_signals(pricing_df)


@pytest.mark.parametrize("damage_str", ["missing_day", "duplicate", "unsorted", "adjustment", "padding", "dividend"])
def test_missing_sessions_and_wrong_data_contract_are_rejected(damage_str):
    pricing_df = make_prices()
    if damage_str == "missing_day":
        pricing_df = pricing_df.drop(pricing_df.index[100])
    elif damage_str == "duplicate":
        pricing_df = pd.concat([pricing_df, pricing_df.iloc[-1:]])
    elif damage_str == "unsorted":
        pricing_df = pricing_df.iloc[::-1]
    elif damage_str == "adjustment":
        pricing_df.attrs["norgate_adjustment_by_symbol_dict"] = {"QQQ": "TOTALRETURN"}
    elif damage_str == "padding":
        pricing_df.attrs["price_padding_policy_str"] = "ALLMARKETDAYS"
    else:
        pricing_df = pricing_df.drop(columns=[("QQQ", "Dividend")])
    with pytest.raises(ValueError):
        strategy_module.QqqGoldenCrossStrategy(10_000).compute_signals(pricing_df)


def test_dividend_ledger_credits_net_cash_without_reinvesting():
    pricing_df = make_prices()
    pricing_df.loc[pricing_df.index[202], ("QQQ", "Dividend")] = 1.0
    strategy_obj = run_synthetic(pricing_df)
    dividend_ledger_df = strategy_obj.get_dividend_ledger()
    assert len(dividend_ledger_df) == 1
    assert dividend_ledger_df.iloc[0]["entitlement_date"] == pricing_df.index[202]
    assert dividend_ledger_df.iloc[0]["ex_date"] == pricing_df.index[203]
    assert dividend_ledger_df.iloc[0]["net_dividend_cash_float"] == 83 * 0.75
    assert strategy_obj.dividend_withholding_rate_float == 0.25
    assert strategy_obj.get_transactions().iloc[1]["amount"] == -83
    comparison_obj = run_synthetic(make_prices())
    assert strategy_obj.results.loc[pricing_df.index[203], "cash"] == pytest.approx(
        comparison_obj.results.loc[pricing_df.index[203], "cash"] + 83 * 0.75
    )


def test_loader_requests_unpadded_capitalspecial_and_never_fetches_a_benchmark(monkeypatch):
    raw_df = make_prices().xs("QQQ", axis=1, level=0)
    price_timeseries_mock = Mock(return_value=raw_df)
    monkeypatch.setattr(strategy_module, "is_snapshot_mode_enabled_bool", lambda: False)
    monkeypatch.setattr(strategy_module, "norgatedata", SimpleNamespace(
        price_timeseries=price_timeseries_mock,
        StockPriceAdjustmentType=SimpleNamespace(CAPITALSPECIAL="capitalspecial-test"),
        PaddingType=SimpleNamespace(NONE="none-test"),
    ))
    pricing_df = strategy_module.get_prices("2022-01-01", "2023-01-01")
    price_timeseries_mock.assert_called_once_with(
        "QQQ", stock_price_adjustment_setting="capitalspecial-test",
        padding_setting="none-test", start_date="2022-01-01", end_date="2023-01-01",
        timeseriesformat="pandas-dataframe",
    )
    assert pricing_df.attrs["price_padding_policy_str"] == "NONE"
    monkeypatch.setattr(strategy_module, "is_snapshot_mode_enabled_bool", lambda: True)
    with pytest.raises(RuntimeError, match="direct Norgate"):
        strategy_module.get_prices("2022-01-01", "2023-01-01")


def test_short_history_is_rejected_before_any_run():
    with pytest.raises(ValueError, match="200 observed warmup"):
        run_synthetic(make_prices(), start_position_int=199)


def test_default_save_path_creates_complete_report_without_a_benchmark(tmp_path):
    pricing_df = make_prices()
    strategy_module.run_variant(
        history_start_date_str=str(pricing_df.index[0].date()),
        backtest_start_date_str=str(pricing_df.index[200].date()),
        end_date_str=str(pricing_df.index[-1].date()),
        capital_base_float=10_000.0, pricing_data_df=pricing_df,
        show_display_bool=False, output_dir_str=str(tmp_path),
    )
    report_path_list = list(tmp_path.rglob("report.html"))
    assert len(report_path_list) == 1
    report_dir_path = report_path_list[0].parent
    assert "data:image/png;base64," in report_path_list[0].read_text(encoding="utf-8")
    assert (report_dir_path / "metadata.json").is_file()
    assert (report_dir_path / "dividend_ledger.csv").is_file()
    assert (report_dir_path / (strategy_module.STRATEGY_NAME_STR + ".pkl")).is_file()
