import unittest
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from scripts.research.run_mu_overnight_backtest import (
    MuOvernightBacktestConfig,
    build_cost_sensitivity_df,
    build_equity_ser,
    build_overnight_trade_detail_df,
    run_mu_overnight_backtest,
)


class MuOvernightBacktestTests(unittest.TestCase):
    def make_price_df(self) -> pd.DataFrame:
        date_idx = pd.date_range("2024-01-02", periods=4, freq="B")
        return pd.DataFrame(
            {
                "Open": [99.0, 110.0, 90.0, 120.0],
                "High": [101.0, 111.0, 91.0, 121.0],
                "Low": [98.0, 109.0, 89.0, 119.0],
                "Close": [100.0, 100.0, 100.0, 100.0],
                "Dividend": [0.0, 0.0, 0.0, 0.0],
            },
            index=date_idx,
            dtype=float,
        )

    def test_overnight_trade_detail_pairs_today_close_with_next_open(self):
        trade_detail_df = build_overnight_trade_detail_df(
            self.make_price_df(),
            symbol_str="MU",
            cost_bps_per_side_float=0.0,
        )

        self.assertEqual(len(trade_detail_df), 3)
        self.assertEqual(trade_detail_df.index[0], pd.Timestamp("2024-01-02"))
        self.assertEqual(trade_detail_df["exit_date"].iloc[0], pd.Timestamp("2024-01-03"))
        self.assertAlmostEqual(
            float(trade_detail_df["gross_overnight_return_float"].iloc[0]),
            0.10,
            places=12,
        )
        self.assertAlmostEqual(
            float(trade_detail_df["price_only_overnight_return_float"].iloc[0]),
            0.10,
            places=12,
        )
        self.assertAlmostEqual(
            float(trade_detail_df["gross_overnight_return_float"].iloc[1]),
            -0.10,
            places=12,
        )
        self.assertAlmostEqual(
            float(trade_detail_df["gross_overnight_return_float"].iloc[2]),
            0.20,
            places=12,
        )

    def test_net_return_uses_per_side_cost_on_entry_and_exit(self):
        trade_detail_df = build_overnight_trade_detail_df(
            self.make_price_df(),
            symbol_str="MU",
            cost_bps_per_side_float=10.0,
        )

        cost_rate_float = 10.0 * 0.0001
        expected_return_float = (110.0 * (1.0 - cost_rate_float)) / (
            100.0 * (1.0 + cost_rate_float)
        ) - 1.0
        self.assertAlmostEqual(
            float(trade_detail_df["net_overnight_return_float"].iloc[0]),
            expected_return_float,
            places=12,
        )
        self.assertGreater(
            float(trade_detail_df["gross_overnight_return_float"].iloc[0]),
            float(trade_detail_df["net_overnight_return_float"].iloc[0]),
        )

    def test_dividend_cash_on_exit_date_is_included_in_overnight_return(self):
        price_df = self.make_price_df()
        price_df.loc[pd.Timestamp("2024-01-03"), "Dividend"] = 1.0

        trade_detail_df = build_overnight_trade_detail_df(
            price_df,
            symbol_str="MU",
            cost_bps_per_side_float=0.0,
        )

        self.assertAlmostEqual(
            float(trade_detail_df["price_only_overnight_return_float"].iloc[0]),
            0.10,
            places=12,
        )
        self.assertAlmostEqual(
            float(trade_detail_df["dividend_return_float"].iloc[0]),
            0.01,
            places=12,
        )
        self.assertAlmostEqual(
            float(trade_detail_df["gross_overnight_return_float"].iloc[0]),
            0.11,
            places=12,
        )

    def test_missing_open_or_close_fails_loud(self):
        price_df = self.make_price_df()
        price_df.loc[pd.Timestamp("2024-01-03"), "Open"] = np.nan

        with self.assertRaisesRegex(RuntimeError, "must not contain missing"):
            build_overnight_trade_detail_df(price_df, symbol_str="MU")

    def test_missing_whole_trading_session_fails_loud_when_calendar_is_provided(self):
        full_price_df = self.make_price_df()
        missing_session_price_df = full_price_df.drop(pd.Timestamp("2024-01-04"))

        with self.assertRaisesRegex(RuntimeError, "missing expected trading"):
            build_overnight_trade_detail_df(
                missing_session_price_df,
                symbol_str="MU",
                expected_trading_date_idx=pd.DatetimeIndex(full_price_df.index),
            )

    def test_unsorted_or_duplicate_dates_fail_loud(self):
        unsorted_price_df = self.make_price_df().iloc[[1, 0, 2, 3]]
        with self.assertRaisesRegex(RuntimeError, "sorted"):
            build_overnight_trade_detail_df(unsorted_price_df, symbol_str="MU")

        duplicate_price_df = pd.concat(
            [self.make_price_df(), self.make_price_df().iloc[[0]]],
            axis=0,
        ).sort_index()
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            build_overnight_trade_detail_df(duplicate_price_df, symbol_str="MU")

    def test_multiindex_input_uses_symbol_columns(self):
        price_df = self.make_price_df()
        price_df.columns = pd.MultiIndex.from_tuples([("MU", column_str) for column_str in price_df.columns])

        trade_detail_df = build_overnight_trade_detail_df(
            price_df,
            symbol_str="MU",
            cost_bps_per_side_float=0.0,
        )

        self.assertEqual(len(trade_detail_df), 3)
        self.assertAlmostEqual(
            float(trade_detail_df["gross_overnight_return_float"].iloc[0]),
            0.10,
            places=12,
        )

    def test_equity_compounds_only_completed_overnight_trades(self):
        trade_detail_df = build_overnight_trade_detail_df(
            self.make_price_df(),
            symbol_str="MU",
            cost_bps_per_side_float=0.0,
        )
        equity_ser = build_equity_ser(
            trade_detail_df=trade_detail_df,
            return_column_str="gross_overnight_return_float",
            capital_base_float=100_000.0,
        )

        expected_final_equity_float = 100_000.0 * 1.10 * 0.90 * 1.20
        self.assertEqual(len(equity_ser), 4)
        self.assertAlmostEqual(float(equity_ser.iloc[-1]), expected_final_equity_float, places=8)

    def test_cost_sensitivity_worsens_final_equity_as_cost_increases(self):
        config = MuOvernightBacktestConfig(
            symbol_str="MU",
            cost_bps_per_side_tuple=(0.0, 10.0),
        )
        cost_sensitivity_df = build_cost_sensitivity_df(
            self.make_price_df(),
            config=config,
        )

        gross_final_equity_float = float(cost_sensitivity_df.iloc[0]["final_equity_float"])
        costed_final_equity_float = float(cost_sensitivity_df.iloc[1]["final_equity_float"])
        self.assertGreater(gross_final_equity_float, costed_final_equity_float)

    def test_run_backtest_returns_summary_without_saving(self):
        config = MuOvernightBacktestConfig(
            symbol_str="MU",
            capital_base_float=100_000.0,
            cost_bps_per_side_tuple=(0.0,),
        )
        result_obj = run_mu_overnight_backtest(
            config=config,
            ohlcv_price_df=self.make_price_df(),
            save_results_bool=False,
        )

        self.assertIsNone(result_obj.output_dir_path)
        self.assertEqual(int(result_obj.primary_summary_df.iloc[0]["trade_count_int"]), 3)
        self.assertTrue(np.isfinite(float(result_obj.primary_summary_df.iloc[0]["sharpe_float"])))

    def test_run_backtest_saves_artifacts_without_optional_markdown_dependencies(self):
        config = MuOvernightBacktestConfig(
            symbol_str="MU",
            capital_base_float=100_000.0,
            cost_bps_per_side_tuple=(0.0,),
        )
        with TemporaryDirectory() as output_dir_str:
            result_obj = run_mu_overnight_backtest(
                config=config,
                ohlcv_price_df=self.make_price_df(),
                output_dir_str=output_dir_str,
                save_results_bool=True,
            )

            self.assertIsNotNone(result_obj.output_dir_path)
            self.assertTrue((result_obj.output_dir_path / "README.md").exists())
            self.assertTrue((result_obj.output_dir_path / "summary.csv").exists())
            self.assertTrue((result_obj.output_dir_path / "trades.csv").exists())


if __name__ == "__main__":
    unittest.main()
