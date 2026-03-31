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
from strategies.strategy_mo_spy_vol_adj_ema import (
    SpyVolAdjustedEmaStrategy,
    compute_spy_vol_adjusted_signal_df,
)


class SpyVolAdjustedEmaStrategyTests(unittest.TestCase):
    def make_strategy(self, **kwargs) -> SpyVolAdjustedEmaStrategy:
        base_kwargs = dict(
            name="SpyVolAdjustedEmaTest",
            benchmarks=["$SPX"],
            symbol_str="SPY",
            eta_float=0.25,
            max_weight_float=1.5,
            capital_base=100_000.0,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )
        base_kwargs.update(kwargs)
        return SpyVolAdjustedEmaStrategy(**base_kwargs)

    def make_pricing_data_df(self, close_vec: np.ndarray | None = None) -> pd.DataFrame:
        date_index = pd.date_range("2024-01-02", periods=8, freq="B")
        if close_vec is None:
            close_vec = np.array([100.0, 110.0, 99.0, 138.6, 140.0, 141.0, 142.0, 143.0], dtype=float)

        open_vec = close_vec - 1.0
        benchmark_close_vec = np.linspace(4_000.0, 4_070.0, len(date_index))

        pricing_data_df = pd.DataFrame(
            {
                ("SPY", "Open"): open_vec,
                ("SPY", "High"): close_vec + 1.5,
                ("SPY", "Low"): close_vec - 1.5,
                ("SPY", "Close"): close_vec,
                ("$SPX", "Open"): benchmark_close_vec - 5.0,
                ("$SPX", "High"): benchmark_close_vec + 10.0,
                ("$SPX", "Low"): benchmark_close_vec - 10.0,
                ("$SPX", "Close"): benchmark_close_vec,
            },
            index=date_index,
            dtype=float,
        )
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        return pricing_data_df

    def test_compute_spy_vol_adjusted_signal_df_matches_recursive_formula(self):
        date_index = pd.date_range("2024-01-02", periods=4, freq="B")
        price_close_ser = pd.Series([100.0, 110.0, 99.0, 138.6], index=date_index, dtype=float)

        signal_df = compute_spy_vol_adjusted_signal_df(
            price_close_ser=price_close_ser,
            eta_float=0.25,
            max_weight_float=1.5,
        )

        self.assertTrue(np.isnan(signal_df.loc[date_index[0], "return_ser"]))
        self.assertAlmostEqual(float(signal_df.loc[date_index[1], "return_ser"]), 0.10, places=12)
        self.assertAlmostEqual(float(signal_df.loc[date_index[1], "volatility_ser"]), 0.10, places=12)
        self.assertTrue(np.isnan(signal_df.loc[date_index[1], "normalized_return_ser"]))

        self.assertAlmostEqual(float(signal_df.loc[date_index[2], "normalized_return_ser"]), -1.0, places=12)
        self.assertAlmostEqual(float(signal_df.loc[date_index[2], "trend_signal_ser"]), -0.5, places=12)
        self.assertAlmostEqual(float(signal_df.loc[date_index[2], "target_weight_ser"]), 0.0, places=12)
        self.assertTrue(np.isnan(signal_df.loc[date_index[2], "turnover_ser"]))

        expected_sigma_t_float = float(np.sqrt(0.75 * 0.01 + 0.25 * 0.16))
        self.assertAlmostEqual(float(signal_df.loc[date_index[3], "volatility_ser"]), expected_sigma_t_float, places=12)
        self.assertAlmostEqual(float(signal_df.loc[date_index[3], "normalized_return_ser"]), 4.0, places=12)
        self.assertAlmostEqual(float(signal_df.loc[date_index[3], "trend_signal_ser"]), 1.625, places=12)
        self.assertAlmostEqual(float(signal_df.loc[date_index[3], "target_weight_ser"]), 1.5, places=12)
        self.assertAlmostEqual(float(signal_df.loc[date_index[3], "turnover_ser"]), 1.5, places=12)

    def test_compute_signals_adds_expected_features_and_passes_signal_audit(self):
        strategy = self.make_strategy(eta_float=0.25)
        pricing_data_df = self.make_pricing_data_df()

        signal_data_df = strategy.compute_signals(pricing_data_df)

        self.assertIn(("SPY", "return_ser"), signal_data_df.columns)
        self.assertIn(("SPY", "volatility_ser"), signal_data_df.columns)
        self.assertIn(("SPY", "normalized_return_ser"), signal_data_df.columns)
        self.assertIn(("SPY", "trend_signal_ser"), signal_data_df.columns)
        self.assertIn(("SPY", "target_weight_ser"), signal_data_df.columns)
        self.assertIn(("SPY", "turnover_ser"), signal_data_df.columns)

        strategy.audit_signals(pricing_data_df, signal_data_df)

    def test_compute_spy_vol_adjusted_signal_df_clips_to_configured_leverage_cap(self):
        date_index = pd.date_range("2024-01-02", periods=4, freq="B")
        price_close_ser = pd.Series([100.0, 110.0, 99.0, 138.6], index=date_index, dtype=float)

        signal_df = compute_spy_vol_adjusted_signal_df(
            price_close_ser=price_close_ser,
            eta_float=0.25,
            max_weight_float=1.2,
        )

        self.assertAlmostEqual(float(signal_df.loc[date_index[3], "trend_signal_ser"]), 1.625, places=12)
        self.assertAlmostEqual(float(signal_df.loc[date_index[3], "target_weight_ser"]), 1.2, places=12)

    def test_iterate_submits_entry_order_when_target_shares_turn_positive(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-01-03")
        strategy.current_bar = pd.Timestamp("2024-01-04")

        close_row_ser = pd.Series(
            {
                ("SPY", "target_weight_ser"): 0.50,
            }
        )
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        open_price_ser = pd.Series({"SPY": 100.0}, dtype=float)

        strategy.iterate(pd.DataFrame(index=[strategy.previous_bar]), close_row_ser, open_price_ser)

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)
        entry_order = order_list[0]
        self.assertIsInstance(entry_order, MarketOrder)
        self.assertEqual(entry_order.asset, "SPY")
        self.assertEqual(entry_order.unit, "shares")
        self.assertTrue(entry_order.target)
        self.assertEqual(entry_order.amount, 500)
        self.assertEqual(entry_order.trade_id, 1)

    def test_iterate_supports_leveraged_entry_up_to_one_point_five_x(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-01-03")
        strategy.current_bar = pd.Timestamp("2024-01-04")

        close_row_ser = pd.Series(
            {
                ("SPY", "target_weight_ser"): 1.50,
            }
        )
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        open_price_ser = pd.Series({"SPY": 100.0}, dtype=float)

        strategy.iterate(pd.DataFrame(index=[strategy.previous_bar]), close_row_ser, open_price_ser)

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)
        entry_order = order_list[0]
        self.assertEqual(entry_order.amount, 1500)
        self.assertEqual(entry_order.trade_id, 1)

    def test_iterate_skips_no_op_when_target_share_count_matches_current_position(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-01-03")
        strategy.current_bar = pd.Timestamp("2024-01-04")
        strategy.trade_id_int = 7
        strategy.current_trade_id_int = 7
        strategy.add_transaction(7, strategy.previous_bar, "SPY", 500, 100.0, 50_000.0, 1, 0.0)

        close_row_ser = pd.Series(
            {
                ("SPY", "target_weight_ser"): 0.50,
            }
        )
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        open_price_ser = pd.Series({"SPY": 100.0}, dtype=float)

        strategy.iterate(pd.DataFrame(index=[strategy.previous_bar]), close_row_ser, open_price_ser)

        self.assertEqual(len(strategy.get_orders()), 0)

    def test_iterate_submits_resize_with_same_trade_id(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-01-03")
        strategy.current_bar = pd.Timestamp("2024-01-04")
        strategy.trade_id_int = 3
        strategy.current_trade_id_int = 3
        strategy.add_transaction(3, strategy.previous_bar, "SPY", 200, 100.0, 20_000.0, 1, 0.0)

        close_row_ser = pd.Series(
            {
                ("SPY", "target_weight_ser"): 0.50,
            }
        )
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        open_price_ser = pd.Series({"SPY": 100.0}, dtype=float)

        strategy.iterate(pd.DataFrame(index=[strategy.previous_bar]), close_row_ser, open_price_ser)

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)
        resize_order = order_list[0]
        self.assertEqual(resize_order.amount, 500)
        self.assertEqual(resize_order.trade_id, 3)

    def test_iterate_submits_exit_with_existing_trade_id(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-01-03")
        strategy.current_bar = pd.Timestamp("2024-01-04")
        strategy.trade_id_int = 5
        strategy.current_trade_id_int = 5
        strategy.add_transaction(5, strategy.previous_bar, "SPY", 250, 100.0, 25_000.0, 1, 0.0)

        close_row_ser = pd.Series(
            {
                ("SPY", "target_weight_ser"): 0.0,
            }
        )
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        open_price_ser = pd.Series({"SPY": 100.0}, dtype=float)

        strategy.iterate(pd.DataFrame(index=[strategy.previous_bar]), close_row_ser, open_price_ser)

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)
        exit_order = order_list[0]
        self.assertEqual(exit_order.amount, 0)
        self.assertEqual(exit_order.trade_id, 5)
        self.assertEqual(strategy.current_trade_id_int, -1)

    def test_run_daily_smoke_generates_summary(self):
        strategy = self.make_strategy(
            eta_float=0.25,
            capital_base=100_000.0,
        )
        pricing_data_df = self.make_pricing_data_df(
            close_vec=np.array([100.0, 110.0, 99.0, 138.6, 120.0, 118.0, 90.0, 89.0], dtype=float)
        )
        calendar_idx = pricing_data_df.index[2:]

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="divide by zero encountered in scalar divide",
                category=RuntimeWarning,
            )
            run_daily(
                strategy,
                pricing_data_df,
                calendar=calendar_idx,
                show_progress=False,
                show_signal_progress_bool=False,
                audit_override_bool=None,
            )

        self.assertIsNotNone(strategy.summary)
        self.assertIn("Strategy", strategy.summary.columns)
        self.assertGreater(len(strategy.results), 0)


if __name__ == "__main__":
    unittest.main()
