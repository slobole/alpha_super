"""Adversarial data-vintage tests: future actions cannot rewrite decisions."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from strategies.momentum.strategy_mo_atr_normalized_ndx import (
    DEFAULT_CONFIG, compute_atr_normalized_signal_tables, get_unadjusted_close_df,
    get_monthly_decision_close_df,
)
from strategies.momentum.strategy_mo_atr_normalized_ndx_corr_penalty import (
    CorrPenaltyAtrNormalizedNdxStrategy,
)


def make_price_tables():
    date_idx = pd.bdate_range("2011-01-03", "2013-06-28")
    elapsed_vec = np.arange(len(date_idx), dtype=float)
    close_df = pd.DataFrame({
        "NVDA": 10.0 * np.exp(elapsed_vec * 0.0006),
        "CSCO": 20.0 * np.exp(elapsed_vec * 0.0005),
    }, index=date_idx)
    return close_df, close_df * 1.01, close_df * 0.99


def compute_tables(close_df, high_df, low_df, raw_close_df):
    return compute_atr_normalized_signal_tables(
        close_df, high_df, low_df, close_df["CSCO"],
        replace(DEFAULT_CONFIG, index_trend_window_int=20),
        price_unadjusted_close_df=raw_close_df,
    )


@pytest.mark.parametrize("factor_float", [40.0, 0.1, 1.5])
def test_future_action_per_symbol_leaves_historical_atr_and_score_unchanged(factor_float):
    close_df, high_df, low_df = make_price_tables()
    raw_close_df = close_df.copy()
    reference_tuple = compute_tables(close_df, high_df, low_df, raw_close_df)
    factor_ser = pd.Series({"NVDA": factor_float, "CSCO": 1.0})
    adjusted_tuple = compute_tables(
        close_df / factor_ser, high_df / factor_ser, low_df / factor_ser, raw_close_df,
    )
    pd.testing.assert_frame_equal(reference_tuple[2], adjusted_tuple[2])
    pd.testing.assert_frame_equal(reference_tuple[6], adjusted_tuple[6])


def test_split_inside_atr_window_keeps_split_adjusted_continuity():
    close_df, high_df, low_df = make_price_tables()
    # Known 4:1 action within the trailing window: history in current units
    # is continuous, while the nominal historical close changes units.
    raw_close_df = close_df.copy()
    raw_close_df.loc[:"2013-06-19", "NVDA"] *= 4.0
    result_tuple = compute_tables(close_df, high_df, low_df, raw_close_df)
    reference_tuple = compute_tables(close_df, high_df, low_df, close_df)
    pd.testing.assert_series_equal(result_tuple[2].loc["2013-06-28"], reference_tuple[2].loc["2013-06-28"])


def test_decision_prefix_does_not_read_later_adjustment_anchor():
    close_df, high_df, low_df = make_price_tables()
    reference_tuple = compute_tables(close_df, high_df, low_df, close_df)
    cutoff_ts = pd.Timestamp("2012-12-31")
    prefix_tuple = compute_tables(
        close_df.loc[:cutoff_ts], high_df.loc[:cutoff_ts], low_df.loc[:cutoff_ts], close_df.loc[:cutoff_ts],
    )
    pd.testing.assert_frame_equal(reference_tuple[6].loc[:cutoff_ts], prefix_tuple[6])


@pytest.mark.parametrize("invalid_float", [np.nan, 0.0, -1.0, np.inf])
def test_bad_decision_anchor_fails_closed(invalid_float):
    close_df, high_df, low_df = make_price_tables()
    raw_close_df = close_df.copy()
    raw_close_df.loc["2013-06-28", "NVDA"] = invalid_float
    with pytest.raises(ValueError, match="anchor"):
        compute_tables(close_df, high_df, low_df, raw_close_df)


def test_missing_raw_field_never_falls_back_to_adjusted_close():
    pricing_df = pd.DataFrame({("NVDA", "Close"): [0.351]})
    with pytest.raises(ValueError, match="Unadjusted Close"):
        get_unadjusted_close_df(pricing_df, ["NVDA"])


@pytest.mark.parametrize("month_end_str", [
    "2002-03-28", "2004-05-28", "2010-05-28", "2013-03-28",
    "2018-03-29", "2021-05-28", "2024-03-28",
])
def test_holiday_month_end_prefix_matches_full_history(month_end_str):
    decision_ts = pd.Timestamp(month_end_str)
    close_df = pd.DataFrame({"AAA": [100.0, 101.0]}, index=pd.to_datetime([decision_ts, decision_ts + pd.offsets.MonthBegin(1)]))
    full_df = get_monthly_decision_close_df(close_df)
    prefix_df = get_monthly_decision_close_df(close_df.loc[:decision_ts])
    assert decision_ts in prefix_df.index
    pd.testing.assert_frame_equal(full_df.loc[:decision_ts], prefix_df)


def test_mosaic_uses_native_dollar_turnover_not_split_adjusted_volume():
    close_df, high_df, low_df = make_price_tables()
    pricing_dict = {}
    for symbol_str in close_df:
        for field_str, field_df in [("Close", close_df), ("Open", close_df), ("High", high_df), ("Low", low_df)]:
            pricing_dict[(symbol_str, field_str)] = field_df[symbol_str]
        pricing_dict[(symbol_str, "Unadjusted Close")] = close_df[symbol_str]
        pricing_dict[(symbol_str, "Volume")] = 100_000.0
        pricing_dict[(symbol_str, "Turnover")] = 3_000_000.0
    pricing_dict[("SPY", "Close")] = close_df["CSCO"]
    pricing_df = pd.DataFrame(pricing_dict, index=close_df.index)
    strategy_obj = CorrPenaltyAtrNormalizedNdxStrategy(
        name="unit_test", benchmarks=[], rebalance_schedule_df=pd.DataFrame(
            {"decision_date_ts": [pd.Timestamp("2013-05-31")]},
            index=pd.to_datetime(["2013-06-03"]),
        ),
        min_dollar_adv_float=5_000_000.0,
    )
    strategy_obj.compute_signals(pricing_df)
    reference_df = strategy_obj.dollar_adv_df.copy()
    pricing_df[("NVDA", "Volume")] *= 40.0
    for field_str in ["Open", "High", "Low", "Close"]:
        pricing_df[("NVDA", field_str)] /= 40.0
    strategy_obj.compute_signals(pricing_df)
    pd.testing.assert_frame_equal(reference_df, strategy_obj.dollar_adv_df)
    assert strategy_obj.dollar_adv_df.loc["2013-06-28", "NVDA"] == 3_000_000.0
