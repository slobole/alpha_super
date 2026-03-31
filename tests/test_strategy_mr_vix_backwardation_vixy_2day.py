import os
import unittest
import warnings
from pathlib import Path

import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from alpha.engine.backtest import run_daily
from alpha.engine.order import MarketOrder
from strategies.strategy_mr_vix_backwardation_vixy_2day import (
    SIGNAL_NAMESPACE_STR,
    VixBackwardationVixy2DayStrategy,
    compute_vix_backwardation_vixy_2day_signal_df,
)


class VixBackwardationVixy2DayStrategyTests(unittest.TestCase):
    def make_strategy(self, **kwargs) -> VixBackwardationVixy2DayStrategy:
        base_kwargs = dict(
            name="VixBackwardationVixy2DayTest",
            benchmarks=["SPY"],
            trade_symbol_str="VIXY",
            vix_symbol_str="$VIX",
            vix3m_symbol_str="$VIX3M",
            hold_day_count_int=2,
            target_weight_float=1.00,
            capital_base=10_000.0,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )
        base_kwargs.update(kwargs)
        return VixBackwardationVixy2DayStrategy(**base_kwargs)

    def make_pricing_data_df(self) -> pd.DataFrame:
        date_index = pd.date_range("2024-01-02", periods=7, freq="B")

        pricing_data_df = pd.DataFrame(
            {
                ("VIXY", "Open"): [20.0, 21.0, 19.0, 18.0, 22.0, 21.0, 20.0],
                ("VIXY", "High"): [21.0, 22.0, 20.0, 19.0, 23.0, 22.0, 21.0],
                ("VIXY", "Low"): [19.0, 20.0, 18.0, 17.0, 21.0, 20.0, 19.0],
                ("VIXY", "Close"): [20.5, 20.0, 18.5, 22.0, 21.5, 20.5, 21.0],
                ("$VIX", "Open"): [20.0, 21.0, 22.0, 18.0, 17.0, 21.0, 22.0],
                ("$VIX", "High"): [20.5, 21.5, 22.5, 18.5, 17.5, 21.5, 22.5],
                ("$VIX", "Low"): [19.5, 20.5, 21.5, 17.5, 16.5, 20.5, 21.5],
                ("$VIX", "Close"): [20.0, 21.0, 22.0, 18.0, 17.0, 21.0, 22.0],
                ("$VIX3M", "Open"): [19.0, 20.0, 21.0, 19.0, 18.0, 20.0, 21.0],
                ("$VIX3M", "High"): [19.5, 20.5, 21.5, 19.5, 18.5, 20.5, 21.5],
                ("$VIX3M", "Low"): [18.5, 19.5, 20.5, 18.5, 17.5, 19.5, 20.5],
                ("$VIX3M", "Close"): [19.0, 20.0, 21.0, 19.0, 18.0, 20.0, 21.0],
                ("SPY", "Open"): [470.0, 471.0, 472.0, 473.0, 474.0, 475.0, 476.0],
                ("SPY", "High"): [471.0, 472.0, 473.0, 474.0, 475.0, 476.0, 477.0],
                ("SPY", "Low"): [469.0, 470.0, 471.0, 472.0, 473.0, 474.0, 475.0],
                ("SPY", "Close"): [470.5, 471.5, 472.5, 473.5, 474.5, 475.5, 476.5],
            },
            index=date_index,
            dtype=float,
        )
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        return pricing_data_df

    def test_compute_vix_backwardation_vixy_2day_signal_df_creates_two_day_hold_windows(self):
        date_index = pd.date_range("2024-01-02", periods=6, freq="B")

        signal_df = compute_vix_backwardation_vixy_2day_signal_df(
            vix_close_ser=pd.Series([20.0, 21.0, 22.0, 18.0, 17.0, 21.0], index=date_index, dtype=float),
            vix3m_close_ser=pd.Series([19.0, 20.0, 21.0, 19.0, 18.0, 20.0], index=date_index, dtype=float),
            hold_day_count_int=2,
            target_weight_float=1.0,
        )

        expected_target_weight_vec = [1.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.assertEqual(signal_df["target_weight_ser"].tolist(), expected_target_weight_vec)
        self.assertEqual(signal_df["entry_signal_bool_ser"].tolist(), [True, False, False, False, False, True])
        self.assertEqual(signal_df["flat_reset_bool_ser"].tolist(), [False, False, True, False, False, False])

    def test_compute_signals_adds_expected_features_and_passes_signal_audit(self):
        strategy = self.make_strategy()
        pricing_data_df = self.make_pricing_data_df()

        signal_data_df = strategy.compute_signals(pricing_data_df)

        self.assertIn((SIGNAL_NAMESPACE_STR, "term_spread_ser"), signal_data_df.columns)
        self.assertIn((SIGNAL_NAMESPACE_STR, "backwardation_bool"), signal_data_df.columns)
        self.assertIn((SIGNAL_NAMESPACE_STR, "entry_signal_bool_ser"), signal_data_df.columns)
        self.assertIn((SIGNAL_NAMESPACE_STR, "flat_reset_bool_ser"), signal_data_df.columns)
        self.assertIn((SIGNAL_NAMESPACE_STR, "close_bars_remaining_int_ser"), signal_data_df.columns)
        self.assertIn((SIGNAL_NAMESPACE_STR, "target_weight_ser"), signal_data_df.columns)

        strategy.audit_signals(pricing_data_df, signal_data_df)

    def test_iterate_submits_entry_order_when_target_weight_turns_on(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-01-03")
        strategy.current_bar = pd.Timestamp("2024-01-04")

        close_row_ser = pd.Series(
            {
                (SIGNAL_NAMESPACE_STR, "target_weight_ser"): 1.00,
            }
        )
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        open_price_ser = pd.Series({"VIXY": 20.0}, dtype=float)

        strategy.iterate(pd.DataFrame(index=[strategy.previous_bar]), close_row_ser, open_price_ser)

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)
        entry_order = order_list[0]
        self.assertIsInstance(entry_order, MarketOrder)
        self.assertEqual(entry_order.asset, "VIXY")
        self.assertEqual(entry_order.amount, 500)
        self.assertEqual(entry_order.trade_id, 1)

    def test_iterate_submits_exit_order_when_two_day_window_ends(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-01-03")
        strategy.current_bar = pd.Timestamp("2024-01-04")
        strategy.trade_id_int = 4
        strategy.current_trade_id_int = 4
        strategy.add_transaction(4, strategy.previous_bar, "VIXY", 500, 20.0, 10_000.0, 1, 0.0)

        close_row_ser = pd.Series(
            {
                (SIGNAL_NAMESPACE_STR, "target_weight_ser"): 0.0,
            }
        )
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        open_price_ser = pd.Series({"VIXY": 20.0}, dtype=float)

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
        strategy.add_transaction(8, strategy.previous_bar, "VIXY", 500, 20.0, 10_000.0, 1, 0.0)

        close_row_ser = pd.Series(
            {
                (SIGNAL_NAMESPACE_STR, "target_weight_ser"): 1.00,
            }
        )
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        open_price_ser = pd.Series({"VIXY": 20.0}, dtype=float)

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
