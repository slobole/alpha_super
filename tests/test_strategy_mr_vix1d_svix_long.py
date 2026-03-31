import os
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from alpha.engine.backtest import run_daily
from alpha.engine.order import MarketOrder
from strategies.strategy_mr_vix1d_svix_long import (
    Vix1dSvixLongStrategy,
    compute_vix1d_svix_long_signal_df,
)


class Vix1dSvixLongStrategyTests(unittest.TestCase):
    def make_strategy(self, **kwargs) -> Vix1dSvixLongStrategy:
        base_kwargs = dict(
            name="Vix1dSvixLongTest",
            benchmarks=["SPY"],
            trade_symbol_str="SVIX",
            vix1d_symbol_str="$VIX1D",
            vix_symbol_str="$VIX",
            vix3m_symbol_str="$VIX3M",
            short_threshold_float=15.0,
            target_weight_float=1.00,
            capital_base=10_000.0,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )
        base_kwargs.update(kwargs)
        return Vix1dSvixLongStrategy(**base_kwargs)

    def make_pricing_data_df(self) -> pd.DataFrame:
        date_index = pd.date_range("2024-01-02", periods=6, freq="B")

        pricing_data_df = pd.DataFrame(
            {
                ("SVIX", "Open"): [20.0, 21.0, 19.0, 18.0, 22.0, 21.0],
                ("SVIX", "High"): [21.0, 22.0, 20.0, 19.0, 23.0, 22.0],
                ("SVIX", "Low"): [19.0, 20.0, 18.0, 17.0, 21.0, 20.0],
                ("SVIX", "Close"): [20.5, 20.0, 18.5, 22.0, 21.5, 20.5],
                ("$VIX1D", "Open"): [12.0, 16.0, 18.0, 14.0, 17.0, 12.0],
                ("$VIX1D", "High"): [12.5, 16.5, 18.5, 14.5, 17.5, 12.5],
                ("$VIX1D", "Low"): [11.5, 15.5, 17.5, 13.5, 16.5, 11.5],
                ("$VIX1D", "Close"): [12.0, 16.0, 18.0, 14.0, 17.0, 12.0],
                ("$VIX", "Open"): [20.0, 18.0, 19.0, 17.0, 19.0, 20.0],
                ("$VIX", "High"): [20.5, 18.5, 19.5, 17.5, 19.5, 20.5],
                ("$VIX", "Low"): [19.5, 17.5, 18.5, 16.5, 18.5, 19.5],
                ("$VIX", "Close"): [20.0, 18.0, 19.0, 17.0, 19.0, 20.0],
                ("$VIX3M", "Open"): [21.0, 19.0, 18.0, 18.0, 20.0, 19.0],
                ("$VIX3M", "High"): [21.5, 19.5, 18.5, 18.5, 20.5, 19.5],
                ("$VIX3M", "Low"): [20.5, 18.5, 17.5, 17.5, 19.5, 18.5],
                ("$VIX3M", "Close"): [21.0, 19.0, 18.0, 18.0, 20.0, 19.0],
                ("SPY", "Open"): [470.0, 471.0, 472.0, 473.0, 474.0, 475.0],
                ("SPY", "High"): [471.0, 472.0, 473.0, 474.0, 475.0, 476.0],
                ("SPY", "Low"): [469.0, 470.0, 471.0, 472.0, 473.0, 474.0],
                ("SPY", "Close"): [470.5, 471.5, 472.5, 473.5, 474.5, 475.5],
            },
            index=date_index,
            dtype=float,
        )
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        return pricing_data_df

    def test_compute_vix1d_svix_long_signal_df_matches_threshold_and_term_filter_logic(self):
        date_index = pd.date_range("2024-01-02", periods=4, freq="B")
        vix1d_close_ser = pd.Series([12.0, 16.0, 18.0, 18.0], index=date_index, dtype=float)
        vix_close_ser = pd.Series([20.0, 18.0, 19.0, 22.0], index=date_index, dtype=float)
        vix3m_close_ser = pd.Series([21.0, 19.0, 18.0, 21.0], index=date_index, dtype=float)

        signal_df = compute_vix1d_svix_long_signal_df(
            vix1d_close_ser=vix1d_close_ser,
            vix_close_ser=vix_close_ser,
            vix3m_close_ser=vix3m_close_ser,
            short_threshold_float=15.0,
            target_weight_float=1.00,
        )

        self.assertAlmostEqual(float(signal_df.loc[date_index[0], "term_spread_ser"]), -1.0, places=12)
        self.assertFalse(bool(signal_df.loc[date_index[0], "short_vol_signal_bool"]))
        self.assertAlmostEqual(float(signal_df.loc[date_index[0], "target_weight_ser"]), 0.0, places=12)

        self.assertTrue(bool(signal_df.loc[date_index[1], "trade_filter_ok_bool"]))
        self.assertTrue(bool(signal_df.loc[date_index[1], "short_vol_signal_bool"]))
        self.assertAlmostEqual(float(signal_df.loc[date_index[1], "target_weight_ser"]), 1.00, places=12)

        self.assertFalse(bool(signal_df.loc[date_index[2], "trade_filter_ok_bool"]))
        self.assertFalse(bool(signal_df.loc[date_index[2], "short_vol_signal_bool"]))
        self.assertAlmostEqual(float(signal_df.loc[date_index[2], "target_weight_ser"]), 0.0, places=12)

        self.assertFalse(bool(signal_df.loc[date_index[3], "trade_filter_ok_bool"]))
        self.assertFalse(bool(signal_df.loc[date_index[3], "short_vol_signal_bool"]))
        self.assertAlmostEqual(float(signal_df.loc[date_index[3], "target_weight_ser"]), 0.0, places=12)

    def test_compute_signals_adds_expected_features_and_passes_signal_audit(self):
        strategy = self.make_strategy()
        pricing_data_df = self.make_pricing_data_df()

        signal_data_df = strategy.compute_signals(pricing_data_df)

        self.assertIn(("SVIX", "vix1d_close_ser"), signal_data_df.columns)
        self.assertIn(("SVIX", "vix_close_ser"), signal_data_df.columns)
        self.assertIn(("SVIX", "vix3m_close_ser"), signal_data_df.columns)
        self.assertIn(("SVIX", "term_spread_ser"), signal_data_df.columns)
        self.assertIn(("SVIX", "trade_filter_ok_bool"), signal_data_df.columns)
        self.assertIn(("SVIX", "short_vol_signal_bool"), signal_data_df.columns)
        self.assertIn(("SVIX", "target_weight_ser"), signal_data_df.columns)

        strategy.audit_signals(pricing_data_df, signal_data_df)

    def test_iterate_submits_entry_order_when_previous_close_signal_is_on(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-01-03")
        strategy.current_bar = pd.Timestamp("2024-01-04")

        close_row_ser = pd.Series(
            {
                ("SVIX", "target_weight_ser"): 1.00,
            }
        )
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        open_price_ser = pd.Series({"SVIX": 20.0}, dtype=float)

        strategy.iterate(pd.DataFrame(index=[strategy.previous_bar]), close_row_ser, open_price_ser)

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)
        entry_order = order_list[0]
        self.assertIsInstance(entry_order, MarketOrder)
        self.assertEqual(entry_order.asset, "SVIX")
        self.assertEqual(entry_order.unit, "shares")
        self.assertTrue(entry_order.target)
        self.assertEqual(entry_order.amount, 500)
        self.assertEqual(entry_order.trade_id, 1)

    def test_iterate_submits_exit_when_signal_turns_off(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-01-03")
        strategy.current_bar = pd.Timestamp("2024-01-04")
        strategy.trade_id_int = 4
        strategy.current_trade_id_int = 4
        strategy.add_transaction(4, strategy.previous_bar, "SVIX", 100, 20.0, 2_000.0, 1, 0.0)

        close_row_ser = pd.Series(
            {
                ("SVIX", "target_weight_ser"): 0.0,
            }
        )
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        open_price_ser = pd.Series({"SVIX": 20.0}, dtype=float)

        strategy.iterate(pd.DataFrame(index=[strategy.previous_bar]), close_row_ser, open_price_ser)

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)
        exit_order = order_list[0]
        self.assertIsInstance(exit_order, MarketOrder)
        self.assertEqual(exit_order.amount, 0)
        self.assertEqual(exit_order.trade_id, 4)
        self.assertEqual(strategy.current_trade_id_int, -1)

    def test_iterate_skips_no_op_when_target_share_count_matches_current_position(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-01-03")
        strategy.current_bar = pd.Timestamp("2024-01-04")
        strategy.trade_id_int = 8
        strategy.current_trade_id_int = 8
        strategy.add_transaction(8, strategy.previous_bar, "SVIX", 500, 20.0, 10_000.0, 1, 0.0)

        close_row_ser = pd.Series(
            {
                ("SVIX", "target_weight_ser"): 1.00,
            }
        )
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        open_price_ser = pd.Series({"SVIX": 20.0}, dtype=float)

        strategy.iterate(pd.DataFrame(index=[strategy.previous_bar]), close_row_ser, open_price_ser)

        self.assertEqual(len(strategy.get_orders()), 0)

    def test_run_daily_smoke_generates_summary(self):
        strategy = self.make_strategy()
        pricing_data_df = self.make_pricing_data_df()

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="divide by zero encountered in scalar divide",
                category=RuntimeWarning,
            )
            run_daily(
                strategy,
                pricing_data_df,
                calendar=pricing_data_df.index,
                show_progress=False,
                show_signal_progress_bool=False,
                audit_override_bool=None,
            )

        self.assertIsNotNone(strategy.summary)
        self.assertIn("Strategy", strategy.summary.columns)
        self.assertGreater(len(strategy.results), 0)


if __name__ == "__main__":
    unittest.main()
