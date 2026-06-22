import unittest
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd

from strategies.momentum.strategy_mo_gappers_iwc_close_to_open import (
    DEFAULT_CONFIG,
    IwcGappersCloseToOpenStrategy,
    build_base_gappers_config,
)
from strategies.momentum.strategy_mo_gappers_russell3000_close_to_open import (
    run_gappers_close_to_open_backtest,
)


class IwcGappersCloseToOpenTests(unittest.TestCase):
    def make_pricing_data_df(
        self,
        price_map_dict: dict[str, dict[str, list[float]]],
        date_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        column_map: dict[tuple[str, str], pd.Series] = {}
        for symbol_str, field_map_dict in price_map_dict.items():
            for field_str, value_list in field_map_dict.items():
                column_map[(symbol_str, field_str)] = pd.Series(
                    value_list,
                    index=date_index,
                    dtype=float,
                )
        pricing_data_df = pd.DataFrame(column_map, index=date_index)
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        return pricing_data_df

    def test_default_config_is_single_iwc_gap_z_variant(self):
        self.assertEqual(DEFAULT_CONFIG.symbol_str, "IWC")
        self.assertEqual(DEFAULT_CONFIG.benchmark_symbol_str, "SPY")
        self.assertEqual(DEFAULT_CONFIG.gap_vol_lookback_day_int, 252)
        self.assertAlmostEqual(DEFAULT_CONFIG.gap_z_min_float, 2.0)
        self.assertEqual(DEFAULT_CONFIG.trend_sma_day_int, 200)

        base_config_obj = build_base_gappers_config(config_obj=DEFAULT_CONFIG)
        self.assertEqual(base_config_obj.max_positions_int, 1)
        self.assertAlmostEqual(base_config_obj.min_entry_price_float, 0.01)
        self.assertGreater(base_config_obj.max_entry_price_float, 100_000.0)

    def test_compute_signals_uses_iwc_gap_without_mean_subtraction(self):
        date_index = pd.bdate_range("2024-01-02", periods=6)
        pricing_data_df = self.make_pricing_data_df(
            {
                "IWC": {
                    "Open": [100.0, 101.0, 99.0, 102.0, 105.0, 101.0],
                    "High": [106.0] * 6,
                    "Low": [94.0] * 6,
                    "Close": [100.0] * 6,
                },
                "SPY": {
                    "Open": [500.0] * 6,
                    "High": [501.0] * 6,
                    "Low": [499.0] * 6,
                    "Close": [500.0, 501.0, 502.0, 503.0, 504.0, 505.0],
                },
            },
            date_index=date_index,
        )
        config_obj = replace(DEFAULT_CONFIG, gap_vol_lookback_day_int=3)
        strategy_obj = IwcGappersCloseToOpenStrategy(
            name="IwcGappersCloseToOpenSignalTest",
            config_obj=config_obj,
        )

        signal_data_df = strategy_obj.compute_signals(pricing_data_df)

        # *** CRITICAL *** lookahead-sensitive: expected gap_z_t uses
        # Open_t / Close_{t-1} in the numerator and prior rolling gap
        # volatility ending at t-1 in the denominator.
        overnight_gap_ser = pricing_data_df[("IWC", "Open")] / pricing_data_df[
            ("IWC", "Close")
        ].shift(1) - 1.0
        expected_trailing_vol_ser = overnight_gap_ser.rolling(3, min_periods=3).std(ddof=1).shift(1)
        expected_gap_z_ser = overnight_gap_ser / expected_trailing_vol_ser
        unshifted_gap_z_ser = overnight_gap_ser / overnight_gap_ser.rolling(3, min_periods=3).std(ddof=1)

        check_ts = date_index[4]
        self.assertAlmostEqual(
            float(signal_data_df.loc[check_ts, ("IWC", "overnight_gap_ser")]),
            float(overnight_gap_ser.loc[check_ts]),
        )
        self.assertAlmostEqual(
            float(signal_data_df.loc[check_ts, ("IWC", "trailing_gap_vol_3_ser")]),
            float(expected_trailing_vol_ser.loc[check_ts]),
        )
        self.assertAlmostEqual(
            float(signal_data_df.loc[check_ts, ("IWC", "gap_z_ser")]),
            float(expected_gap_z_ser.loc[check_ts]),
        )
        self.assertNotAlmostEqual(
            float(signal_data_df.loc[check_ts, ("IWC", "gap_z_ser")]),
            float(unshifted_gap_z_ser.loc[check_ts]),
        )

    def test_compute_signals_requires_iwc_close_above_trailing_sma(self):
        date_index = pd.bdate_range("2024-01-02", periods=5)
        pricing_data_df = self.make_pricing_data_df(
            {
                "IWC": {
                    "Open": [10.0, 10.0, 10.0, 11.0, 9.0],
                    "High": [11.5] * 5,
                    "Low": [8.5] * 5,
                    "Close": [10.0, 10.0, 10.0, 11.0, 9.0],
                },
                "SPY": {
                    "Open": [500.0] * 5,
                    "High": [501.0] * 5,
                    "Low": [499.0] * 5,
                    "Close": [500.0, 501.0, 502.0, 503.0, 504.0],
                },
            },
            date_index=date_index,
        )
        config_obj = replace(
            DEFAULT_CONFIG,
            gap_vol_lookback_day_int=3,
            trend_sma_day_int=3,
        )
        strategy_obj = IwcGappersCloseToOpenStrategy(
            name="IwcGappersCloseToOpenTrendFilterTest",
            config_obj=config_obj,
        )

        signal_data_df = strategy_obj.compute_signals(pricing_data_df)

        # *** CRITICAL *** rolling-window timing: expected_sma_ser includes
        # Close_T because this research strategy also decides and enters at
        # Close_T. Shifting this SMA would test a different rule.
        expected_sma_ser = pricing_data_df[("IWC", "Close")].rolling(3, min_periods=3).mean()
        pass_ts = date_index[3]
        fail_ts = date_index[4]

        self.assertAlmostEqual(
            float(signal_data_df.loc[pass_ts, ("IWC", "sma_3_ser")]),
            float(expected_sma_ser.loc[pass_ts]),
        )
        self.assertTrue(bool(signal_data_df.loc[pass_ts, ("IWC", "above_sma_3_bool")]))
        self.assertTrue(bool(signal_data_df.loc[pass_ts, ("IWC", "price_filter_pass_bool")]))
        self.assertFalse(bool(signal_data_df.loc[fail_ts, ("IWC", "above_sma_3_bool")]))
        self.assertFalse(bool(signal_data_df.loc[fail_ts, ("IWC", "price_filter_pass_bool")]))

    def test_backtest_buys_iwc_signal_close_and_sells_next_open(self):
        date_index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
        signal_data_df = self.make_pricing_data_df(
            {
                "IWC": {
                    "Open": [100.0, 105.0, 113.0, 110.0],
                    "High": [101.0, 111.0, 114.0, 111.0],
                    "Low": [99.0, 104.0, 112.0, 109.0],
                    "Close": [100.0, 110.0, 112.0, 109.0],
                },
                "SPY": {
                    "Open": [500.0, 501.0, 502.0, 503.0],
                    "High": [501.0, 502.0, 503.0, 504.0],
                    "Low": [499.0, 500.0, 501.0, 502.0],
                    "Close": [500.0, 501.0, 502.0, 503.0],
                },
            },
            date_index=date_index,
        )
        feature_map = {
            ("IWC", "overnight_gap_ser"): [np.nan, 0.05, 0.01, 0.01],
            ("IWC", "trailing_gap_vol_3_ser"): [np.nan, 0.01, 0.01, 0.01],
            ("IWC", "gap_z_ser"): [np.nan, 3.0, 1.0, 1.0],
            ("IWC", "price_filter_pass_bool"): [True, True, True, True],
        }
        for column_tuple, value_list in feature_map.items():
            signal_data_df[column_tuple] = value_list
        signal_data_df.columns = pd.MultiIndex.from_tuples(signal_data_df.columns)

        config_obj = replace(
            DEFAULT_CONFIG,
            gap_vol_lookback_day_int=3,
            backtest_start_date_str="2024-01-02",
            capital_base_float=1_000.0,
            slippage_float=0.0,
            commission_per_share_float=0.0,
            commission_minimum_float=0.0,
        )
        strategy_obj = IwcGappersCloseToOpenStrategy(
            name="IwcGappersCloseToOpenTest",
            config_obj=config_obj,
        )
        strategy_obj.universe_df = pd.DataFrame(
            {"IWC": [1, 1, 1, 1]},
            index=date_index,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            run_gappers_close_to_open_backtest(
                strategy=strategy_obj,
                pricing_data_df=signal_data_df,
                signal_data_df=signal_data_df,
                backtest_start_date_str=None,
            )

        transaction_df = strategy_obj.get_transactions().reset_index(drop=True)
        self.assertEqual(len(transaction_df), 2)

        entry_row_ser = transaction_df.iloc[0]
        self.assertEqual(entry_row_ser["asset"], "IWC")
        self.assertEqual(pd.Timestamp(entry_row_ser["bar"]), pd.Timestamp("2024-01-03"))
        self.assertEqual(float(entry_row_ser["price"]), 110.0)
        self.assertEqual(float(entry_row_ser["amount"]), 9.0)

        exit_row_ser = transaction_df.iloc[1]
        self.assertEqual(exit_row_ser["asset"], "IWC")
        self.assertEqual(pd.Timestamp(exit_row_ser["bar"]), pd.Timestamp("2024-01-04"))
        self.assertEqual(float(exit_row_ser["price"]), 113.0)
        self.assertEqual(float(exit_row_ser["amount"]), -9.0)

        self.assertEqual(strategy_obj.daily_selection_df["symbol_str"].tolist(), ["IWC"])

    def test_backtest_does_not_trade_when_gap_passes_but_sma_filter_fails(self):
        date_index = pd.bdate_range("2024-01-02", periods=6)
        pricing_data_df = self.make_pricing_data_df(
            {
                "IWC": {
                    "Open": [10.0, 10.1, 9.9, 10.2, 10.5, 9.1],
                    "High": [10.6, 10.3, 10.1, 10.4, 10.7, 9.3],
                    "Low": [9.8, 9.9, 9.7, 9.9, 8.8, 8.9],
                    "Close": [10.0, 10.0, 10.0, 10.0, 9.0, 9.0],
                },
                "SPY": {
                    "Open": [500.0] * 6,
                    "High": [501.0] * 6,
                    "Low": [499.0] * 6,
                    "Close": [500.0, 501.0, 502.0, 503.0, 504.0, 505.0],
                },
            },
            date_index=date_index,
        )
        config_obj = replace(
            DEFAULT_CONFIG,
            gap_vol_lookback_day_int=3,
            trend_sma_day_int=3,
            backtest_start_date_str="2024-01-02",
            capital_base_float=1_000.0,
            slippage_float=0.0,
            commission_per_share_float=0.0,
            commission_minimum_float=0.0,
        )
        strategy_obj = IwcGappersCloseToOpenStrategy(
            name="IwcGappersCloseToOpenNoTradeTrendFilterTest",
            config_obj=config_obj,
        )
        strategy_obj.universe_df = pd.DataFrame(
            {"IWC": [1, 1, 1, 1, 1, 1]},
            index=date_index,
        )
        signal_data_df = strategy_obj.compute_signals(pricing_data_df)
        fail_ts = date_index[4]

        self.assertGreater(float(signal_data_df.loc[fail_ts, ("IWC", "gap_z_ser")]), 2.0)
        self.assertFalse(bool(signal_data_df.loc[fail_ts, ("IWC", "above_sma_3_bool")]))
        self.assertFalse(bool(signal_data_df.loc[fail_ts, ("IWC", "price_filter_pass_bool")]))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            run_gappers_close_to_open_backtest(
                strategy=strategy_obj,
                pricing_data_df=pricing_data_df,
                signal_data_df=signal_data_df,
                backtest_start_date_str=None,
            )

        self.assertEqual(len(strategy_obj.get_transactions()), 0)
        self.assertEqual(len(strategy_obj.daily_selection_df), 0)


if __name__ == "__main__":
    unittest.main()
