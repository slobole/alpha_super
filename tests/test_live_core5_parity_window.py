"""A reporting start must preserve the strategy's canonical event history."""
import pandas as pd
import pytest

from scripts.review import verify_core5_adapter_parity as parity_module
from strategies.taa_beyond_6040 import strategy_taa_adaptive_macro_core5 as core5_module
import test_live_core5_adapter as fixture_module


def test_late_reporting_start_keeps_prior_dbc_weight_and_actual_share_drift(monkeypatch):
    pricing_df = fixture_module.price_df.__wrapped__().iloc[:165].copy()
    pricing_df[("$SPX", "Close")] = pricing_df[("SPY", "Close")] * 50
    signal_df = fixture_module._controlled_signals(monkeypatch, pricing_df)
    namespace_str = core5_module.signal_namespace_str("DBC")
    signal_df[(namespace_str, "long_state_ser")] = 0.0
    signal_df[(namespace_str, "short_state_ser")] = 1.0
    signal_df[(namespace_str, "annualized_volatility_ser")] = .5
    # With no new strategy event, increasing current volatility must not replace
    # the prior event's DBC weight: -0.025 / 0.5 = -0.05, not -0.025 / 2 = -0.0125.
    signal_df.loc[pricing_df.index[-4]:, (namespace_str, "annualized_volatility_ser")] = 2.0
    start_date_str = str(pricing_df.index[-3].date())
    assert not signal_df.loc[pricing_df.index[-4],
        (core5_module.PORTFOLIO_NAMESPACE_STR, core5_module.MONTH_END_REBALANCE_FIELD_STR)]

    full_report_dict = parity_module.compare_adapter_to_engine(pricing_df, "2024-01-02")
    late_report_dict = parity_module.compare_adapter_to_engine(pricing_df, start_date_str)
    expected_row_list = [row_dict for row_dict in full_report_dict["decision_row_list"]
        if row_dict["execution_date_str"] >= start_date_str]
    assert late_report_dict["decision_row_list"] == expected_row_list
    assert late_report_dict["decision_count_int"] == 3
    assert late_report_dict["warmup_decision_count_int"] > 0
    assert late_report_dict["oracle_start_date_str"] == full_report_dict["oracle_start_date_str"]
    assert late_report_dict["oracle_borrow_fee_float"] == full_report_dict["oracle_borrow_fee_float"]
    assert late_report_dict["decision_row_list"][0]["no_order_bool"]
    assert not late_report_dict["decision_row_list"][0]["rebalance_bool"]


def test_reporting_start_after_available_history_cannot_pass_empty_comparison():
    pricing_df = fixture_module.price_df.__wrapped__().iloc[:165].copy()
    start_date_str = str((pricing_df.index[-1] + pd.Timedelta(days=1)).date())
    with pytest.raises(ValueError, match="no execution sessions"):
        parity_module.compare_adapter_to_engine(pricing_df, start_date_str)
