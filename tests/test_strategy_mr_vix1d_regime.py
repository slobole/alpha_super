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
from strategies.vix_stuff.strategy_mr_vix1d_regime import (
    SIGNAL_NAMESPACE_STR,
    Vix1dRegimeStrategy,
    compute_vix1d_regime_signal_df,
)


class Vix1dRegimeStrategyTests(unittest.TestCase):
    def make_strategy(self, **kwargs) -> Vix1dRegimeStrategy:
        base_kwargs = dict(
            name="Vix1dRegimeTest",
            benchmarks=["SPY"],
            long_vol_trade_symbol_str="VIXY",
            short_vol_trade_symbol_str="SVIX",
            vix1d_symbol_str="$VIX1D",
            vix_symbol_str="$VIX",
            vix3m_symbol_str="$VIX3M",
            long_threshold_float=10.0,
            short_threshold_float=15.0,
            target_weight_float=1.00,
            capital_base=10_000.0,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )
        base_kwargs.update(kwargs)
        return Vix1dRegimeStrategy(**base_kwargs)

    def make_pricing_data_df(self) -> pd.DataFrame:
        date_index = pd.date_range("2024-01-02", periods=6, freq="B")

        pricing_data_df = pd.DataFrame(
            {
                ("VIXY", "Open"): [20.0, 21.0, 19.0, 18.0, 22.0, 21.0],
                ("VIXY", "High"): [21.0, 22.0, 20.0, 19.0, 23.0, 22.0],
                ("VIXY", "Low"): [19.0, 20.0, 18.0, 17.0, 21.0, 20.0],
                ("VIXY", "Close"): [20.5, 20.0, 18.5, 22.0, 21.5, 20.5],
                ("SVIX", "Open"): [20.0, 21.0, 19.0, 18.0, 22.0, 21.0],
                ("SVIX", "High"): [21.0, 22.0, 20.0, 19.0, 23.0, 22.0],
                ("SVIX", "Low"): [19.0, 20.0, 18.0, 17.0, 21.0, 20.0],
                ("SVIX", "Close"): [20.5, 20.0, 18.5, 22.0, 21.5, 20.5],
                ("$VIX1D", "Open"): [12.0, 9.0, 16.0, 14.0, 18.0, 8.0],
                ("$VIX1D", "High"): [12.5, 9.5, 16.5, 14.5, 18.5, 8.5],
                ("$VIX1D", "Low"): [11.5, 8.5, 15.5, 13.5, 17.5, 7.5],
                ("$VIX1D", "Close"): [12.0, 9.0, 16.0, 14.0, 18.0, 8.0],
                ("$VIX", "Open"): [20.0, 18.0, 18.0, 17.0, 19.0, 18.0],
                ("$VIX", "High"): [20.5, 18.5, 18.5, 17.5, 19.5, 18.5],
                ("$VIX", "Low"): [19.5, 17.5, 17.5, 16.5, 18.5, 17.5],
                ("$VIX", "Close"): [20.0, 18.0, 18.0, 17.0, 19.0, 18.0],
                ("$VIX3M", "Open"): [21.0, 19.0, 19.0, 18.0, 18.0, 19.0],
                ("$VIX3M", "High"): [21.5, 19.5, 19.5, 18.5, 18.5, 19.5],
                ("$VIX3M", "Low"): [20.5, 18.5, 18.5, 17.5, 17.5, 18.5],
                ("$VIX3M", "Close"): [21.0, 19.0, 19.0, 18.0, 18.0, 19.0],
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

    def test_compute_vix1d_regime_signal_df_matches_spec(self):
        date_index = pd.date_range("2024-01-02", periods=4, freq="B")
        signal_df = compute_vix1d_regime_signal_df(
            vix1d_close_ser=pd.Series([12.0, 9.0, 16.0, 18.0], index=date_index, dtype=float),
            vix_close_ser=pd.Series([20.0, 18.0, 18.0, 19.0], index=date_index, dtype=float),
            vix3m_close_ser=pd.Series([21.0, 19.0, 19.0, 18.0], index=date_index, dtype=float),
            long_threshold_float=10.0,
            short_threshold_float=15.0,
            target_weight_float=1.0,
            long_vol_trade_symbol_str="VIXY",
            short_vol_trade_symbol_str="SVIX",
        )

        self.assertTrue(pd.isna(signal_df.loc[date_index[0], "target_symbol_obj_ser"]))
        self.assertEqual(signal_df.loc[date_index[1], "target_symbol_obj_ser"], "VIXY")
        self.assertEqual(signal_df.loc[date_index[2], "target_symbol_obj_ser"], "SVIX")
        self.assertTrue(pd.isna(signal_df.loc[date_index[3], "target_symbol_obj_ser"]))

        self.assertAlmostEqual(float(signal_df.loc[date_index[1], "long_vol_target_weight_ser"]), 1.0, places=12)
        self.assertAlmostEqual(float(signal_df.loc[date_index[1], "short_vol_target_weight_ser"]), 0.0, places=12)
        self.assertAlmostEqual(float(signal_df.loc[date_index[2], "long_vol_target_weight_ser"]), 0.0, places=12)
        self.assertAlmostEqual(float(signal_df.loc[date_index[2], "short_vol_target_weight_ser"]), 1.0, places=12)

    def test_compute_signals_adds_expected_features_and_passes_signal_audit(self):
        strategy = self.make_strategy()
        pricing_data_df = self.make_pricing_data_df()

        signal_data_df = strategy.compute_signals(pricing_data_df)

        self.assertIn((SIGNAL_NAMESPACE_STR, "trade_filter_ok_bool"), signal_data_df.columns)
        self.assertIn((SIGNAL_NAMESPACE_STR, "long_vol_signal_bool"), signal_data_df.columns)
        self.assertIn((SIGNAL_NAMESPACE_STR, "short_vol_signal_bool"), signal_data_df.columns)
        self.assertIn((SIGNAL_NAMESPACE_STR, "long_vol_target_weight_ser"), signal_data_df.columns)
        self.assertIn((SIGNAL_NAMESPACE_STR, "short_vol_target_weight_ser"), signal_data_df.columns)
        self.assertIn((SIGNAL_NAMESPACE_STR, "target_symbol_obj_ser"), signal_data_df.columns)

        strategy.audit_signals(pricing_data_df, signal_data_df)

    def test_iterate_enters_vixy_when_long_vol_signal_is_on(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-01-03")
        strategy.current_bar = pd.Timestamp("2024-01-04")

        close_row_ser = pd.Series(
            {
                (SIGNAL_NAMESPACE_STR, "long_vol_target_weight_ser"): 1.00,
                (SIGNAL_NAMESPACE_STR, "short_vol_target_weight_ser"): 0.00,
            }
        )
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)

        strategy.iterate(pd.DataFrame(index=[strategy.previous_bar]), close_row_ser, pd.Series({"VIXY": 20.0, "SVIX": 20.0}))

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)
        entry_order = order_list[0]
        self.assertIsInstance(entry_order, MarketOrder)
        self.assertEqual(entry_order.asset, "VIXY")
        self.assertEqual(entry_order.amount, 500)
        self.assertEqual(entry_order.trade_id, 1)

    def test_iterate_enters_svix_when_short_vol_signal_is_on(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-01-03")
        strategy.current_bar = pd.Timestamp("2024-01-04")

        close_row_ser = pd.Series(
            {
                (SIGNAL_NAMESPACE_STR, "long_vol_target_weight_ser"): 0.00,
                (SIGNAL_NAMESPACE_STR, "short_vol_target_weight_ser"): 1.00,
            }
        )
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)

        strategy.iterate(pd.DataFrame(index=[strategy.previous_bar]), close_row_ser, pd.Series({"VIXY": 20.0, "SVIX": 20.0}))

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)
        entry_order = order_list[0]
        self.assertIsInstance(entry_order, MarketOrder)
        self.assertEqual(entry_order.asset, "SVIX")
        self.assertEqual(entry_order.amount, 500)
        self.assertEqual(entry_order.trade_id, 1)

    def test_iterate_switches_from_vixy_to_svix(self):
        strategy = self.make_strategy()
        strategy.trade_id_int = 5
        strategy.current_trade_id_map["VIXY"] = 5
        strategy.previous_bar = pd.Timestamp("2024-01-03")
        strategy.current_bar = pd.Timestamp("2024-01-04")
        strategy.add_transaction(5, strategy.previous_bar, "VIXY", 500, 20.0, 10_000.0, 1, 0.0)

        close_row_ser = pd.Series(
            {
                (SIGNAL_NAMESPACE_STR, "long_vol_target_weight_ser"): 0.00,
                (SIGNAL_NAMESPACE_STR, "short_vol_target_weight_ser"): 1.00,
            }
        )
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)

        strategy.iterate(pd.DataFrame(index=[strategy.previous_bar]), close_row_ser, pd.Series({"VIXY": 20.0, "SVIX": 20.0}))

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 2)

        exit_order = order_list[0]
        self.assertEqual(exit_order.asset, "VIXY")
        self.assertEqual(exit_order.amount, 0)
        self.assertEqual(exit_order.trade_id, 5)

        entry_order = order_list[1]
        self.assertEqual(entry_order.asset, "SVIX")
        self.assertEqual(entry_order.amount, 500)
        self.assertEqual(entry_order.trade_id, 6)

    def test_iterate_closes_open_position_when_filter_forces_flat(self):
        strategy = self.make_strategy()
        strategy.trade_id_int = 7
        strategy.current_trade_id_map["SVIX"] = 7
        strategy.previous_bar = pd.Timestamp("2024-01-03")
        strategy.current_bar = pd.Timestamp("2024-01-04")
        strategy.add_transaction(7, strategy.previous_bar, "SVIX", 500, 20.0, 10_000.0, 1, 0.0)

        close_row_ser = pd.Series(
            {
                (SIGNAL_NAMESPACE_STR, "long_vol_target_weight_ser"): 0.00,
                (SIGNAL_NAMESPACE_STR, "short_vol_target_weight_ser"): 0.00,
            }
        )
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)

        strategy.iterate(pd.DataFrame(index=[strategy.previous_bar]), close_row_ser, pd.Series({"VIXY": 20.0, "SVIX": 20.0}))

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)
        exit_order = order_list[0]
        self.assertEqual(exit_order.asset, "SVIX")
        self.assertEqual(exit_order.amount, 0)
        self.assertEqual(exit_order.trade_id, 7)

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
