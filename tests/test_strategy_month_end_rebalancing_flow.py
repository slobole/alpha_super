"""Frozen-source target parity and causal MOC account-ledger regression tests."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from alpha.engine.backtest import run_daily
from alpha.engine.execution_timing import ExecutionTimingAnalyzer
from alpha.strategy_registry import MaturityTier, tier_for
from strategies.taa_beyond_6040 import strategy_taa_month_end_rebalancing_flow as flow_module


FIXTURE_PATH = Path(__file__).parent / "fixtures/month_end_rebalancing_flow"


def frozen_close_df() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_PATH / "total_return_closes.csv", index_col=0, parse_dates=True)


def synthetic_pricing_df(end_date_str: str = "2003-02-10") -> pd.DataFrame:
    session_idx = flow_module.exchange_session_idx(pd.Timestamp(end_date_str))
    session_idx = session_idx[(session_idx >= "2002-07-26") & (session_idx <= end_date_str)]
    price_field_dict = {}
    for asset_str in flow_module.TRADED_ASSET_TUPLE:
        for field_str, value_float in (("Open", 80.), ("High", 120.), ("Low", 70.), ("Close", 100.), ("Dividend", 0.)):
            price_field_dict[(asset_str, field_str)] = np.full(len(session_idx), value_float)
    for asset_str in flow_module.SIGNAL_ASSET_TUPLE:
        price_field_dict[(f"FLOW_TR_{asset_str}", "Close")] = np.full(len(session_idx), 100.)
    for field_str in ("Open", "High", "Low", "Close"):
        price_field_dict[("$SPX", field_str)] = 100. + np.arange(len(session_idx)) * .01
    pricing_data_df = pd.DataFrame(price_field_dict, index=session_idx)
    pricing_data_df.attrs["norgate_adjustment_by_symbol_dict"] = {
        "SPY": "CAPITALSPECIAL", "TLT": "CAPITALSPECIAL", "$SPX": "TOTALRETURN",
        "FLOW_TR_SPY": "TOTALRETURN", "FLOW_TR_IEF": "TOTALRETURN",
    }
    pricing_data_df.attrs["benchmark_data_symbol_dict"] = {"$SPX": "$SPXTR"}
    pricing_data_df.attrs["price_padding_policy_str"] = "NONE"
    return pricing_data_df


def run_synthetic(pricing_data_df, capital_float=100_000.):
    strategy_obj = flow_module.MonthEndRebalancingFlowStrategy(
        replace(flow_module.DEFAULT_CONFIG, capital_base_float=capital_float)
    )
    run_daily(strategy_obj, pricing_data_df, calendar=pricing_data_df.index[pricing_data_df.index >= "2003-01-02"],
              show_progress=False, show_signal_progress_bool=False)
    return strategy_obj


def test_all_frozen_months_and_hold_targets_match():
    close_df = frozen_close_df()
    actual_month_df = flow_module.build_month_table_df(close_df).set_index("month_period")
    expected_month_df = pd.read_csv(FIXTURE_PATH / "monthly_reference.csv", index_col=0)
    for field_str in ("pressure_ief_measure_bps_float", "bucket_ief_measure_causal_int"):
        np.testing.assert_allclose(actual_month_df.loc[expected_month_df.index, field_str],
                                   expected_month_df[field_str], rtol=0, atol=1e-10, equal_nan=True)
    assert actual_month_df.loc["2004-07", "prior_month_count_int"] == 23
    assert pd.isna(actual_month_df.loc["2004-07", "bucket_ief_measure_causal_int"])
    assert actual_month_df.loc["2004-08", "prior_month_count_int"] == 24
    assert actual_month_df.loc["2026-08", "prior_month_count_int"] == 288
    expected_hold_df = pd.read_csv(FIXTURE_PATH / "hold_targets.csv", index_col=0, parse_dates=True)
    actual_hold_df = flow_module.build_reference_hold_weights_df(actual_month_df.reset_index(), expected_hold_df.index)
    pd.testing.assert_frame_equal(actual_hold_df, expected_hold_df, check_freq=False)
    # Independent source accounting check. This is a diagnostic TR ledger,
    # deliberately separate from whole-share CASH + mark account accounting.
    # *** CRITICAL*** Return on d uses closes d and d-1; h_d was known before
    # its prior-close fill. Source costs are booked on the next hold day.
    return_df = close_df.pct_change(fill_method=None).reindex(expected_hold_df.index)
    full_hold_df = flow_module.build_reference_hold_weights_df(actual_month_df.reset_index(), close_df.index)
    turnover_ser = full_hold_df.diff().abs().sum(axis=1).reindex(actual_hold_df.index)
    actual_return_ser = (actual_hold_df * return_df).sum(axis=1) - .0005 * turnover_ser
    expected_return_ser = pd.read_csv(FIXTURE_PATH / "daily_reference.csv", index_col=0, parse_dates=True).iloc[:, 0]
    np.testing.assert_allclose(actual_return_ser.loc[expected_return_ser.index], expected_return_ser,
                               rtol=0, atol=1e-12)


@pytest.mark.parametrize("cutoff_str", ["2002-07-31", "2004-08-20", "2020-03-23", "2026-08-21", "2026-09-10"])
def test_prefix_and_future_price_perturbation_are_causal(cutoff_str):
    close_df = frozen_close_df()
    expected_df = flow_module.build_month_table_df(close_df)
    expected_df = expected_df.loc[expected_df.measure_date <= cutoff_str].reset_index(drop=True)
    actual_df = flow_module.build_month_table_df(close_df.loc[:cutoff_str])
    pd.testing.assert_frame_equal(actual_df, expected_df, check_dtype=False)
    changed_df = close_df.copy()
    changed_df.loc[changed_df.index > cutoff_str, "SPY"] *= 3
    actual_df = flow_module.build_month_table_df(changed_df)
    pd.testing.assert_frame_equal(actual_df.loc[actual_df.measure_date <= cutoff_str].reset_index(drop=True), expected_df,
                                  check_dtype=False)


def test_ties_warmup_and_all_targets():
    assert np.isnan(flow_module.causal_bucket_float(0., [0.] * 23))
    assert flow_module.causal_bucket_float(0., [0.] * 24) == 5
    assert flow_module.causal_bucket_float(-1., [0.] * 24) == 1
    assert flow_module.causal_bucket_float(0., [-1.] * 12 + [1.] * 12) == 3
    for bucket_float in (1, 2, 3, 4, 5, np.nan):
        assert flow_module.target_weight_tuple(bucket_float, "final") == ((1., 0.) if bucket_float == 1 else (0., 1.))
        expected_tuple = (0., 0.) if bucket_float == 1 else ((.5, -.5) if bucket_float in (4, 5) else (0., -1.))
        assert flow_module.target_weight_tuple(bucket_float, "early") == expected_tuple


def test_known_calendar_dates_and_partial_month_are_not_price_truncated():
    month_df = flow_module.build_month_table_df(frozen_close_df()).set_index("month_period")
    assert month_df.loc["2020-03", "measure_date"] == pd.Timestamp("2020-03-23")
    assert month_df.loc["2020-03", "final_fill_date"] == pd.Timestamp("2020-03-24")
    assert month_df.loc["2026-08", "exit_fill_date"] == pd.Timestamp("2026-09-08")
    assert "2026-09" not in month_df.index
    partial_df = flow_module.build_month_table_df(frozen_close_df().loc[:"2026-08-21"])
    assert partial_df.iloc[-1].early_fill_date == pd.Timestamp("2026-08-31")
    assert pd.Timestamp("2012-10-29") not in flow_module.exchange_session_idx(pd.Timestamp("2012-10-31"))
    assert pd.Timestamp("2025-11-28") in flow_module.exchange_session_idx(pd.Timestamp("2025-11-30"))


@pytest.mark.parametrize("invalid_float", [np.nan, np.inf, 0., -1.])
def test_missing_or_invalid_endpoint_is_not_filled(invalid_float):
    close_df = frozen_close_df()
    close_df.loc["2020-03-23", "IEF"] = invalid_float
    with pytest.raises(ValueError, match="signal endpoint"):
        flow_module.build_month_table_df(close_df)


def test_missing_warmup_or_duplicate_dates_fail():
    close_df = frozen_close_df()
    with pytest.raises(ValueError, match="Full pressure history"):
        flow_module.build_month_table_df(close_df.loc["2003-01-01":])
    with pytest.raises(ValueError, match="unique and increasing"):
        flow_module.build_month_table_df(pd.concat([close_df, close_df.iloc[-1:]]))


def test_moc_sizing_fills_reversal_costs_and_fixed_shares():
    pricing_data_df = synthetic_pricing_df()
    pricing_data_df.loc["2003-01-24", ("TLT", "Close")] = 110.
    original_df = pricing_data_df.copy(deep=True)
    strategy_obj = run_synthetic(pricing_data_df)
    transaction_df = strategy_obj.get_transactions()
    first_row = transaction_df.iloc[0]
    assert first_row.bar == pd.Timestamp("2003-01-24")
    assert first_row.amount == 1000  # fixed from Jan23 close100, not fill110/open80
    assert first_row.price == pytest.approx(110 * 1.00025)
    assert first_row.commission == 5
    reversal_df = transaction_df.loc[transaction_df.bar == pd.Timestamp("2003-01-31")]
    assert len(reversal_df) == 2
    assert (reversal_df.amount < 0).all()
    assert reversal_df.trade_id.nunique() == 2
    assert (strategy_obj.held_share_df.loc["2003-01-24":"2003-01-30", "TLT"] == 1000).all()
    assert strategy_obj.held_share_df.loc["2003-02-07", "TLT"] == 0
    assert strategy_obj._slippage == .00025
    assert strategy_obj._compute_commission(1) == 1
    pd.testing.assert_frame_equal(pricing_data_df, original_df)
    assert tier_for(flow_module.__name__) == MaturityTier.PM_READY


def test_borrow_includes_weekend_and_stops_on_moc_cover():
    strategy_obj = run_synthetic(synthetic_pricing_df())
    borrow_df = strategy_obj.borrow_fee_df.set_index("accrual_start_date")
    first_row = borrow_df.loc["2003-01-31"]
    assert first_row.calendar_day_count_int == 3
    assert first_row.borrow_fee_float == pytest.approx(abs(first_row.short_share_float) * 102 * .01 * 3 / 360)
    assert pd.Timestamp("2003-02-07") not in borrow_df.index
    assert borrow_df.borrow_fee_float.sum() == pytest.approx(strategy_obj.borrow_fee_total_float)


def test_short_dividend_full_gross_and_long_withholding_before_moc():
    pricing_data_df = synthetic_pricing_df()
    # Norgate Dividend is stamped at entitlement close (the day before ex-date).
    pricing_data_df.loc["2003-01-27", ("TLT", "Dividend")] = 1.
    pricing_data_df.loc["2003-01-31", ("TLT", "Dividend")] = 2.
    pricing_data_df.loc["2003-02-06", ("TLT", "Dividend")] = 3.
    strategy_obj = run_synthetic(pricing_data_df)
    dividend_df = strategy_obj.get_dividend_ledger().set_index("ex_date")
    long_row = dividend_df.loc["2003-01-28"]
    assert long_row.net_dividend_cash_float == pytest.approx(1000 * .75)
    for date_str, dividend_float in (("2003-02-03", 2.), ("2003-02-07", 3.)):
        short_row = dividend_df.loc[date_str]
        assert short_row.position_share_float < 0
        assert short_row.withholding_cash_float == 0
        assert short_row.net_dividend_cash_float == pytest.approx(short_row.position_share_float * dividend_float)


def test_missing_auction_or_session_fails_instead_of_synthetic_liquidation():
    pricing_data_df = synthetic_pricing_df()
    pricing_data_df.loc["2003-01-24", ("TLT", "Close")] = np.nan
    with pytest.raises(RuntimeError, match="valid MOC close"):
        run_synthetic(pricing_data_df)
    with pytest.raises(RuntimeError, match="Missing XNYS session"):
        run_synthetic(synthetic_pricing_df().drop(pd.Timestamp("2003-01-24")))
    with pytest.raises(RuntimeError, match="Dividend field"):
        run_synthetic(synthetic_pricing_df().drop(columns=[("TLT", "Dividend")]))


def test_deterministic_rerun_and_capital_are_honored():
    pricing_data_df = synthetic_pricing_df()
    first_obj = run_synthetic(pricing_data_df)
    second_obj = run_synthetic(pricing_data_df)
    pd.testing.assert_frame_equal(first_obj.results, second_obj.results)
    pd.testing.assert_frame_equal(first_obj.borrow_fee_df, second_obj.borrow_fee_df)
    larger_obj = run_synthetic(pricing_data_df, capital_float=200_000.)
    assert larger_obj.get_transactions().iloc[0].amount == 2 * first_obj.get_transactions().iloc[0].amount


@pytest.mark.parametrize("asset_str,adjustment_str", [(None, None), ("SPY", "TOTALRETURN"), ("FLOW_TR_SPY", "CAPITALSPECIAL"), ("FLOW_TR_IEF", "CAPITALSPECIAL")])
def test_adjustment_provenance_is_mandatory(asset_str, adjustment_str):
    pricing_data_df = synthetic_pricing_df()
    if asset_str is None:
        pricing_data_df.attrs.pop("norgate_adjustment_by_symbol_dict")
    else:
        pricing_data_df.attrs["norgate_adjustment_by_symbol_dict"][asset_str] = adjustment_str
    with pytest.raises(ValueError, match="Adjustment provenance"):
        run_synthetic(pricing_data_df)


def test_terminal_short_does_not_prepay_unscored_borrow():
    strategy_obj = run_synthetic(synthetic_pricing_df("2003-01-31"))
    assert strategy_obj.get_position("TLT") < 0
    assert strategy_obj.borrow_fee_total_float == 0


def test_compute_signals_schedule_is_prefix_stable():
    pricing_data_df = synthetic_pricing_df()
    strategy_obj = flow_module.MonthEndRebalancingFlowStrategy()
    expected_df = strategy_obj.compute_signals(pricing_data_df)
    for cutoff_str in ("2002-07-31", "2003-01-23", "2003-01-30", "2003-02-06"):
        prefix_df = pricing_data_df.loc[:cutoff_str]
        actual_df = flow_module.MonthEndRebalancingFlowStrategy().compute_signals(prefix_df)
        pd.testing.assert_frame_equal(actual_df, expected_df.loc[:cutoff_str])


@pytest.mark.parametrize("padding_str", [None, "ALLMARKETDAYS"])
def test_finite_padded_prices_are_rejected(padding_str):
    pricing_data_df = synthetic_pricing_df()
    pricing_data_df.attrs["price_padding_policy_str"] = padding_str
    with pytest.raises(ValueError, match="Observed-price provenance"):
        run_synthetic(pricing_data_df)


def test_loader_uses_observed_prices_and_separate_adjustment_roles(monkeypatch):
    call_row_list = []
    source_df = synthetic_pricing_df()

    def fake_prices_df(symbol_str, **kwargs_dict):
        call_row_list.append((symbol_str, kwargs_dict))
        source_asset_str = "SPY" if symbol_str == "IEF" else ("$SPX" if symbol_str == "$SPXTR" else symbol_str)
        return source_df[source_asset_str].copy()

    monkeypatch.setattr(flow_module, "is_snapshot_mode_enabled_bool", lambda: False)
    monkeypatch.setattr(flow_module, "norgatedata", SimpleNamespace(
        price_timeseries=fake_prices_df,
        StockPriceAdjustmentType=SimpleNamespace(CAPITALSPECIAL="CAPITALSPECIAL", TOTALRETURN="TOTALRETURN"),
        PaddingType=SimpleNamespace(NONE="NONE"),
    ))
    pricing_data_df = flow_module.get_month_end_flow_data(flow_module.DEFAULT_CONFIG)
    assert all(kwargs_dict["padding_setting"] == "NONE" for _, kwargs_dict in call_row_list)
    assert [symbol_str for symbol_str, _ in call_row_list] == ["SPY", "TLT", "SPY", "IEF", "$SPXTR"]
    assert ("FLOW_TR_IEF", "Close") in pricing_data_df.columns
    assert pricing_data_df.attrs["norgate_adjustment_by_symbol_dict"]["SPY"] == "CAPITALSPECIAL"
    assert pricing_data_df.attrs["norgate_adjustment_by_symbol_dict"]["FLOW_TR_SPY"] == "TOTALRETURN"
    assert pricing_data_df.attrs["benchmark_data_symbol_dict"] == {"$SPX": "$SPXTR"}
    monkeypatch.setattr(flow_module, "is_snapshot_mode_enabled_bool", lambda: True)
    with pytest.raises(RuntimeError, match="snapshot observation provenance"):
        flow_module.get_month_end_flow_data(flow_module.DEFAULT_CONFIG)


def timing_result_obj(monkeypatch, pricing_data_df, entry_tuple=("same_close_moc",), exit_tuple=("same_close_moc",)):
    monkeypatch.setattr(flow_module, "get_month_end_flow_data", lambda config_obj: pricing_data_df.copy(deep=True))
    input_dict = flow_module.build_execution_timing_analysis_inputs()
    input_dict["entry_timing_str_tuple"] = entry_tuple
    input_dict["exit_timing_str_tuple"] = exit_tuple
    return ExecutionTimingAnalyzer(**input_dict, save_output_bool=False).run()


def test_timing_default_matches_vanilla_nav_shares_costs_and_dividends(monkeypatch):
    pricing_data_df = synthetic_pricing_df()
    pricing_data_df.loc["2003-01-30", ("TLT", "Dividend")] = 1.
    pricing_data_df.loc["2003-02-06", ("TLT", "Dividend")] = 2.
    vanilla_obj = run_synthetic(pricing_data_df)
    result_obj = timing_result_obj(monkeypatch, pricing_data_df)
    timing_obj = result_obj.strategy_map[("same_close_moc", "same_close_moc")]
    pd.testing.assert_frame_equal(timing_obj.results[["cash", "portfolio_value", "total_value", "daily_returns"]],
                                  vanilla_obj.results[["cash", "portfolio_value", "total_value", "daily_returns"]],
                                  check_exact=False, check_freq=False, atol=1e-8, rtol=1e-12)
    transaction_fields = ["bar", "asset", "amount", "price", "commission", "trade_id"]
    pd.testing.assert_frame_equal(timing_obj.get_transactions()[transaction_fields], vanilla_obj.get_transactions()[transaction_fields])
    pd.testing.assert_frame_equal(timing_obj.borrow_fee_df, vanilla_obj.borrow_fee_df)
    pd.testing.assert_frame_equal(timing_obj.get_dividend_ledger(), vanilla_obj.get_dividend_ledger())
    pd.testing.assert_frame_equal(timing_obj.held_share_df, vanilla_obj.held_share_df)


@pytest.mark.parametrize("entry_str,exit_str,expected_entry_str,expected_exit_str", [
    ("next_open", "same_close_moc", "2003-02-03", "2003-01-31"),
    ("same_close_moc", "next_open", "2003-01-31", "2003-02-03"),
])
def test_timing_reversal_short_entry_and_long_exit_move_independently(
    monkeypatch, entry_str, exit_str, expected_entry_str, expected_exit_str,
):
    result_obj = timing_result_obj(monkeypatch, synthetic_pricing_df(), (entry_str,), (exit_str,))
    strategy_obj = result_obj.strategy_map[(entry_str, exit_str)]
    transaction_df = strategy_obj.get_transactions()
    long_exit_df = transaction_df.loc[(transaction_df.trade_id == 1) & (transaction_df.amount < 0)]
    short_entry_df = transaction_df.loc[(transaction_df.trade_id == 2) & (transaction_df.amount < 0)]
    assert long_exit_df.iloc[0].bar == pd.Timestamp(expected_exit_str)
    assert short_entry_df.iloc[0].bar == pd.Timestamp(expected_entry_str)
    entry_price_float = 80. if entry_str == "next_open" else 100.
    assert short_entry_df.iloc[0].price == pytest.approx(entry_price_float * .99975)


def test_timing_declared_order_kind_rejects_invalid_value(monkeypatch):
    class InvalidRoleStrategy(flow_module.MonthEndRebalancingFlowTimingStrategy):
        def iterate(self, data_df, close_row_ser, open_price_ser):
            super().iterate(data_df, close_row_ser, open_price_ser)
            for order_obj in self.get_orders():
                order_obj.timing_order_kind_str = "invalid"

    pricing_data_df = synthetic_pricing_df()

    def factory_fn():
        strategy_obj = InvalidRoleStrategy(flow_module.DEFAULT_CONFIG)
        strategy_obj.compute_signals(pricing_data_df)
        return strategy_obj

    with pytest.raises(ValueError, match="timing_order_kind_str"):
        ExecutionTimingAnalyzer(
            strategy_factory_fn=factory_fn, pricing_data_df=pricing_data_df,
            calendar_idx=pricing_data_df.index[pricing_data_df.index >= "2003-01-02"],
            entry_timing_str_tuple=("same_close_moc",), exit_timing_str_tuple=("same_close_moc",),
            order_generation_mode_str="vanilla_current_bar", risk_model_str="taa_rebalance",
            save_output_bool=False,
        ).run()


def test_all_analyzer_hooks_and_capacity_preserve_vanilla_contract(monkeypatch):
    from scripts.research.run_strategy_analysis import _missing_hook_detail_str
    from alpha.engine.crisis import SUPPORTED_CRISIS_STRATEGY_SPEC_MAP

    pricing_data_df = synthetic_pricing_df()
    monkeypatch.setattr(flow_module, "_capacity_pricing_df", lambda end_date_str: pricing_data_df.copy(deep=True))
    for analysis_str in ("vanilla", "capacity", "timing", "risk", "stress"):
        assert _missing_hook_detail_str(flow_module, analysis_str) is None
    assert SUPPORTED_CRISIS_STRATEGY_SPEC_MAP[flow_module.STRATEGY_NAME_STR].full_history_replay_bool
    input_dict = flow_module.build_capacity_analysis_inputs(capital_base_float=200_000.)
    assert input_dict["execution_policy_str"] == "MOC"
    assert input_dict["pricing_data_df"].index[0] == pd.Timestamp("2002-07-26")
    pd.testing.assert_frame_equal(input_dict["strategy_obj"].results, run_synthetic(pricing_data_df, 200_000.).results)
    recent_dict = flow_module.build_capacity_analysis_inputs(capital_base_float=100_000., backtest_start_date_str="2003-01-27")
    assert recent_dict["strategy_obj"].results.index[0] == pd.Timestamp("2003-01-27")
    assert recent_dict["pricing_data_df"].index[0] == pd.Timestamp("2002-07-26")


def test_stress_retains_full_pressure_history_and_moc_accounting(monkeypatch):
    from alpha.engine.crisis import CrisisPeriodConfig
    from alpha.engine.stress_test import StressTestAnalyzer

    pricing_data_df = synthetic_pricing_df("2026-08-07")
    source_close_df = frozen_close_df()
    for asset_str in flow_module.SIGNAL_ASSET_TUPLE:
        pricing_data_df[(f"FLOW_TR_{asset_str}", "Close")] = source_close_df[asset_str].reindex(pricing_data_df.index)
    pricing_data_df.loc["2026-07-31", ("TLT", "Dividend")] = 1.
    monkeypatch.setattr(flow_module, "get_month_end_flow_data", lambda config_obj: pricing_data_df.copy(deep=True))
    result_obj = StressTestAnalyzer(
        strategy_key_str=flow_module.STRATEGY_NAME_STR,
        crisis_period_list=[CrisisPeriodConfig("Early August", "2026-08-03", "2026-08-07"),
                            CrisisPeriodConfig("Unavailable", "2001-09-10", "2001-09-28")],
        launch_offset_tuple=(5,), save_output_bool=False,
    ).run()
    assert result_obj.skipped_window_list
    strategy_obj = next(iter(result_obj.stress_strategy_map.values()))
    month_df = strategy_obj.month_table_df.set_index("month_period")
    assert month_df.loc["2026-07", "prior_month_count_int"] == 287
    assert month_df.loc["2026-07", "bucket_ief_measure_causal_int"] == 2
    transaction_df = strategy_obj.get_transactions()
    short_row = transaction_df.loc[(transaction_df.bar == pd.Timestamp("2026-07-31")) & (transaction_df.amount < 0)].iloc[0]
    assert short_row.price == pytest.approx(100. * .99975)
    assert strategy_obj.get_dividend_ledger().iloc[0].net_dividend_cash_float < 0
    assert strategy_obj.borrow_fee_df.calendar_day_count_int.max() == 3
    direct_obj = flow_module.MonthEndRebalancingFlowStrategy()
    run_daily(direct_obj, pricing_data_df, calendar=strategy_obj.run_calendar_idx,
              show_progress=False, show_signal_progress_bool=False)
    pd.testing.assert_frame_equal(strategy_obj.results, direct_obj.results)
    pd.testing.assert_frame_equal(strategy_obj.borrow_fee_df, direct_obj.borrow_fee_df)


@pytest.mark.parametrize("invalid_field_str", ["missing_session", "Open", "Close", "Dividend"])
def test_analysis_context_rejects_incomplete_executable_prices(invalid_field_str):
    pricing_data_df = synthetic_pricing_df()
    if invalid_field_str == "missing_session":
        pricing_data_df = pricing_data_df.drop(pd.Timestamp("2003-01-24"))
    else:
        pricing_data_df.loc["2003-01-24", ("TLT", invalid_field_str)] = np.nan
    with pytest.raises(ValueError, match="missing an XNYS|Invalid analysis"):
        flow_module._build_analysis_context_dict(pricing_data_df=pricing_data_df)
