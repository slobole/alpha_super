import unittest
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd

from strategies.momentum.strategy_mo_gappers_russell3000_close_to_open import (
    DEFAULT_CONFIG,
    GappersRussell3000CloseToOpenStrategy,
    build_gappers_signal_data_df,
    get_gappers_selection_df,
    run_gappers_close_to_open_backtest,
)


class GappersRussell3000CloseToOpenTests(unittest.TestCase):
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

    def make_close_row_ser(self, row_map: dict[tuple[str, str], float | bool]) -> pd.Series:
        close_row_ser = pd.Series(row_map)
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        return close_row_ser

    def make_strategy(self, **kwargs) -> GappersRussell3000CloseToOpenStrategy:
        config_obj = kwargs.pop(
            "config",
            replace(
                DEFAULT_CONFIG,
                gap_vol_lookback_day_int=3,
                max_positions_int=1,
                capital_base_float=1_000.0,
                slippage_float=0.0,
                commission_per_share_float=0.0,
                commission_minimum_float=0.0,
            ),
        )
        return GappersRussell3000CloseToOpenStrategy(
            name="GappersRussell3000CloseToOpenTest",
            benchmarks=[config_obj.benchmark_symbol_str],
            config=config_obj,
        )

    def test_default_config_is_russell3000_top10_gappers(self):
        self.assertEqual(DEFAULT_CONFIG.indexname_str, "Russell 3000")
        self.assertEqual(DEFAULT_CONFIG.benchmark_symbol_str, "$RUA")
        self.assertEqual(DEFAULT_CONFIG.gap_vol_lookback_day_int, 252)
        self.assertEqual(DEFAULT_CONFIG.max_positions_int, 10)
        self.assertAlmostEqual(DEFAULT_CONFIG.gap_z_min_float, 2.0)
        self.assertAlmostEqual(DEFAULT_CONFIG.min_entry_price_float, 2.0)
        self.assertAlmostEqual(DEFAULT_CONFIG.max_entry_price_float, 10.0)

    def test_signal_data_uses_prior_rolling_gap_vol_without_mean_subtraction(self):
        date_index = pd.bdate_range("2024-01-02", periods=6)
        pricing_data_df = self.make_pricing_data_df(
            {
                "AAA": {
                    "Open": [10.0, 10.1, 9.9, 10.2, 10.5, 10.1],
                    "High": [10.6] * 6,
                    "Low": [9.4] * 6,
                    "Close": [10.0] * 6,
                },
                "$RUA": {
                    "Open": [100.0] * 6,
                    "High": [101.0] * 6,
                    "Low": [99.0] * 6,
                    "Close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
                },
            },
            date_index=date_index,
        )

        signal_data_df = build_gappers_signal_data_df(
            pricing_data_df=pricing_data_df,
            benchmark_list=["$RUA"],
            gap_vol_lookback_day_int=3,
            min_entry_price_float=2.0,
            max_entry_price_float=10.0,
        )

        overnight_gap_ser = pricing_data_df[("AAA", "Open")] / pricing_data_df[("AAA", "Close")].shift(1) - 1.0
        expected_trailing_vol_ser = overnight_gap_ser.rolling(3, min_periods=3).std(ddof=1).shift(1)
        expected_gap_z_ser = overnight_gap_ser / expected_trailing_vol_ser
        unshifted_gap_z_ser = overnight_gap_ser / overnight_gap_ser.rolling(3, min_periods=3).std(ddof=1)

        check_ts = date_index[4]
        self.assertAlmostEqual(
            float(signal_data_df.loc[check_ts, ("AAA", "overnight_gap_ser")]),
            float(overnight_gap_ser.loc[check_ts]),
        )
        self.assertAlmostEqual(
            float(signal_data_df.loc[check_ts, ("AAA", "trailing_gap_vol_3_ser")]),
            float(expected_trailing_vol_ser.loc[check_ts]),
        )
        self.assertAlmostEqual(
            float(signal_data_df.loc[check_ts, ("AAA", "gap_z_ser")]),
            float(expected_gap_z_ser.loc[check_ts]),
        )
        self.assertNotAlmostEqual(
            float(signal_data_df.loc[check_ts, ("AAA", "gap_z_ser")]),
            float(unshifted_gap_z_ser.loc[check_ts]),
        )

    def test_selection_filters_price_gap_and_pit_membership_then_takes_top_10(self):
        row_map: dict[tuple[str, str], float | bool] = {}
        universe_map: dict[str, int] = {}
        for symbol_int in range(1, 15):
            symbol_str = f"S{symbol_int:02d}"
            row_map[(symbol_str, "Close")] = 5.0
            row_map[(symbol_str, "overnight_gap_ser")] = 0.05
            row_map[(symbol_str, "gap_z_ser")] = float(symbol_int)
            row_map[(symbol_str, "price_filter_pass_bool")] = True
            universe_map[symbol_str] = 1

        row_map[("S14", "Close")] = 11.0
        row_map[("S14", "price_filter_pass_bool")] = False
        universe_map["S13"] = 0
        row_map[("LOW", "Close")] = 5.0
        row_map[("LOW", "overnight_gap_ser")] = 0.05
        row_map[("LOW", "gap_z_ser")] = 2.0
        row_map[("LOW", "price_filter_pass_bool")] = True
        universe_map["LOW"] = 1

        selected_df = get_gappers_selection_df(
            close_row_ser=self.make_close_row_ser(row_map),
            universe_row_ser=pd.Series(universe_map, dtype=int),
            gap_z_min_float=2.0,
            max_positions_int=10,
        )

        self.assertEqual(len(selected_df), 10)
        self.assertEqual(selected_df["symbol_str"].tolist(), [f"S{i:02d}" for i in range(12, 2, -1)])
        self.assertNotIn("S13", selected_df["symbol_str"].tolist())
        self.assertNotIn("S14", selected_df["symbol_str"].tolist())
        self.assertNotIn("LOW", selected_df["symbol_str"].tolist())

    def test_backtest_buys_signal_close_and_sells_next_open(self):
        date_index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
        signal_data_df = self.make_pricing_data_df(
            {
                "AAA": {
                    "Open": [4.8, 5.0, 6.2, 5.9],
                    "High": [5.2, 5.3, 6.3, 6.1],
                    "Low": [4.7, 4.9, 6.0, 5.8],
                    "Close": [5.0, 5.5, 6.0, 5.8],
                },
                "BBB": {
                    "Open": [4.0, 4.1, 4.2, 4.3],
                    "High": [4.2, 4.3, 4.4, 4.5],
                    "Low": [3.9, 4.0, 4.1, 4.2],
                    "Close": [4.1, 4.2, 4.3, 4.4],
                },
                "$RUA": {
                    "Open": [100.0, 101.0, 102.0, 103.0],
                    "High": [101.0, 102.0, 103.0, 104.0],
                    "Low": [99.0, 100.0, 101.0, 102.0],
                    "Close": [100.0, 101.0, 102.0, 103.0],
                },
            },
            date_index=date_index,
        )
        feature_map = {
            ("AAA", "overnight_gap_ser"): [np.nan, 0.05, 0.01, 0.01],
            ("AAA", "trailing_gap_vol_3_ser"): [np.nan, 0.01, 0.01, 0.01],
            ("AAA", "gap_z_ser"): [np.nan, 3.0, 1.0, 1.0],
            ("AAA", "price_filter_pass_bool"): [True, True, True, True],
            ("BBB", "overnight_gap_ser"): [np.nan, 0.04, 0.01, 0.01],
            ("BBB", "trailing_gap_vol_3_ser"): [np.nan, 0.01, 0.01, 0.01],
            ("BBB", "gap_z_ser"): [np.nan, 2.5, 1.0, 1.0],
            ("BBB", "price_filter_pass_bool"): [True, True, True, True],
        }
        for column_tuple, value_list in feature_map.items():
            signal_data_df[column_tuple] = value_list
        signal_data_df.columns = pd.MultiIndex.from_tuples(signal_data_df.columns)

        strategy_obj = self.make_strategy()
        strategy_obj.universe_df = pd.DataFrame(
            {"AAA": [1, 1, 1, 1], "BBB": [1, 1, 1, 1]},
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
        self.assertEqual(entry_row_ser["asset"], "AAA")
        self.assertEqual(pd.Timestamp(entry_row_ser["bar"]), pd.Timestamp("2024-01-03"))
        self.assertEqual(float(entry_row_ser["price"]), 5.5)
        self.assertEqual(float(entry_row_ser["amount"]), 181.0)

        exit_row_ser = transaction_df.iloc[1]
        self.assertEqual(exit_row_ser["asset"], "AAA")
        self.assertEqual(pd.Timestamp(exit_row_ser["bar"]), pd.Timestamp("2024-01-04"))
        self.assertEqual(float(exit_row_ser["price"]), 6.2)
        self.assertEqual(float(exit_row_ser["amount"]), -181.0)

        self.assertIsNotNone(strategy_obj.summary)
        self.assertIn("Strategy", strategy_obj.summary.columns)
        self.assertEqual(strategy_obj.daily_selection_df["symbol_str"].tolist(), ["AAA"])

    def test_backtest_fails_loud_when_next_open_exit_is_missing(self):
        date_index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        signal_data_df = self.make_pricing_data_df(
            {
                "AAA": {
                    "Open": [4.8, 5.0, np.nan],
                    "High": [5.2, 5.3, 6.3],
                    "Low": [4.7, 4.9, 6.0],
                    "Close": [5.0, 5.5, 6.0],
                },
                "$RUA": {
                    "Open": [100.0, 101.0, 102.0],
                    "High": [101.0, 102.0, 103.0],
                    "Low": [99.0, 100.0, 101.0],
                    "Close": [100.0, 101.0, 102.0],
                },
            },
            date_index=date_index,
        )
        feature_map = {
            ("AAA", "overnight_gap_ser"): [np.nan, 0.05, 0.01],
            ("AAA", "trailing_gap_vol_3_ser"): [np.nan, 0.01, 0.01],
            ("AAA", "gap_z_ser"): [np.nan, 3.0, 1.0],
            ("AAA", "price_filter_pass_bool"): [True, True, True],
        }
        for column_tuple, value_list in feature_map.items():
            signal_data_df[column_tuple] = value_list
        signal_data_df.columns = pd.MultiIndex.from_tuples(signal_data_df.columns)

        strategy_obj = self.make_strategy()
        strategy_obj.universe_df = pd.DataFrame({"AAA": [1, 1, 1]}, index=date_index)

        with self.assertRaisesRegex(RuntimeError, "Missing valid Open_\\{T\\+1\\} exit price"):
            run_gappers_close_to_open_backtest(
                strategy=strategy_obj,
                pricing_data_df=signal_data_df,
                signal_data_df=signal_data_df,
                backtest_start_date_str=None,
            )


if __name__ == "__main__":
    unittest.main()
