import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from alpha.engine.order import MarketOrder
from strategies.strategy_mo_relative_strength import MonthlyRelativeStrengthStrategy


class MonthlyRelativeStrengthStrategyTests(unittest.TestCase):
    def make_strategy(self) -> MonthlyRelativeStrengthStrategy:
        return MonthlyRelativeStrengthStrategy(
            name="MonthlyRelativeStrengthRotationTest",
            benchmarks=["$NDX"],
            capital_base=100_000,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )

    def make_pricing_data_df(self) -> pd.DataFrame:
        date_index = pd.date_range("2023-01-02", periods=260, freq="B")
        step_vec = np.arange(len(date_index), dtype=float)

        aaa_close_vec = 100.0 + 0.40 * step_vec + 2.0 * np.sin(step_vec * 0.05)
        bbb_close_vec = 90.0 + 0.30 * step_vec + 1.5 * np.cos(step_vec * 0.04)
        ndx_close_vec = 1_000.0 + 1.20 * step_vec + 8.0 * np.sin(step_vec * 0.03)

        pricing_data_df = pd.DataFrame(
            {
                ("AAA", "Open"): aaa_close_vec - 0.5,
                ("AAA", "High"): aaa_close_vec + 1.0,
                ("AAA", "Low"): aaa_close_vec - 1.0,
                ("AAA", "Close"): aaa_close_vec,
                ("AAA", "Turnover"): 1_000_000.0 + 5_000.0 * step_vec,
                ("BBB", "Open"): bbb_close_vec - 0.5,
                ("BBB", "High"): bbb_close_vec + 1.0,
                ("BBB", "Low"): bbb_close_vec - 1.0,
                ("BBB", "Close"): bbb_close_vec,
                ("BBB", "Turnover"): 900_000.0 + 4_000.0 * step_vec,
                ("$NDX", "Open"): ndx_close_vec - 1.0,
                ("$NDX", "High"): ndx_close_vec + 2.0,
                ("$NDX", "Low"): ndx_close_vec - 2.0,
                ("$NDX", "Close"): ndx_close_vec,
                ("$NDX", "Turnover"): 10_000_000.0 + 20_000.0 * step_vec,
            },
            index=date_index,
        )
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        return pricing_data_df

    def make_close_row_ser(self, row_map: dict[tuple[str, str], float | bool]) -> pd.Series:
        close_row_ser = pd.Series(row_map)
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        return close_row_ser

    def test_compute_signals_adds_expected_features_and_passes_signal_audit(self):
        strategy = self.make_strategy()
        pricing_data_df = self.make_pricing_data_df()

        signal_data_df = strategy.compute_signals(pricing_data_df)

        self.assertIn(("AAA", "momentum_score"), signal_data_df.columns)
        self.assertIn(("AAA", "realized_vol"), signal_data_df.columns)
        self.assertIn(("AAA", "stock_trend_bool"), signal_data_df.columns)
        self.assertIn(("$NDX", "market_trend_bool"), signal_data_df.columns)

        strategy.audit_signals(pricing_data_df, signal_data_df)

    def test_iterate_submits_no_orders_on_non_rebalance_dates(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-02-01")
        strategy.current_bar = pd.Timestamp("2024-02-02")
        strategy.universe_df = pd.DataFrame({"AAA": [1]}, index=[strategy.previous_bar])

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "momentum_score"): 1.00,
                ("AAA", "realized_vol"): 0.20,
                ("AAA", "stock_trend_bool"): True,
                ("$NDX", "market_trend_bool"): True,
            }
        )

        strategy.iterate(pd.DataFrame(), close_row_ser, pd.Series(dtype=float))

        self.assertEqual(len(strategy.get_orders()), 0)

    def test_iterate_submits_target_percent_orders_on_first_trading_day_of_new_month(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-01-31")
        strategy.current_bar = pd.Timestamp("2024-02-01")
        strategy.universe_df = pd.DataFrame({"AAA": [1]}, index=[strategy.previous_bar])

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "momentum_score"): 1.00,
                ("AAA", "realized_vol"): 0.20,
                ("AAA", "stock_trend_bool"): True,
                ("$NDX", "market_trend_bool"): True,
            }
        )

        strategy.iterate(pd.DataFrame(), close_row_ser, pd.Series(dtype=float))

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)
        entry_order = order_list[0]
        self.assertIsInstance(entry_order, MarketOrder)
        self.assertEqual(entry_order.asset, "AAA")
        self.assertEqual(entry_order.unit, "percent")
        self.assertTrue(entry_order.target)
        self.assertAlmostEqual(entry_order.amount, 0.10)

    def test_get_target_weight_ser_enforces_pit_universe_trend_and_volatility_filters(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-01-31")
        strategy.universe_df = pd.DataFrame(
            {
                "AAA": [1],
                "BBB": [1],
                "CCC": [1],
                "DDD": [1],
                "EEE": [1],
                "HIGHVOL": [1],
                "DOWNTREND": [1],
                "OUT": [0],
            },
            index=[strategy.previous_bar],
        )

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "momentum_score"): 1.00,
                ("AAA", "realized_vol"): 0.20,
                ("AAA", "stock_trend_bool"): True,
                ("BBB", "momentum_score"): 0.90,
                ("BBB", "realized_vol"): 0.25,
                ("BBB", "stock_trend_bool"): True,
                ("CCC", "momentum_score"): 0.80,
                ("CCC", "realized_vol"): 0.30,
                ("CCC", "stock_trend_bool"): True,
                ("DDD", "momentum_score"): 0.70,
                ("DDD", "realized_vol"): 0.35,
                ("DDD", "stock_trend_bool"): True,
                ("EEE", "momentum_score"): 0.60,
                ("EEE", "realized_vol"): 0.40,
                ("EEE", "stock_trend_bool"): True,
                ("HIGHVOL", "momentum_score"): 1.50,
                ("HIGHVOL", "realized_vol"): 1.00,
                ("HIGHVOL", "stock_trend_bool"): True,
                ("DOWNTREND", "momentum_score"): 2.00,
                ("DOWNTREND", "realized_vol"): 0.10,
                ("DOWNTREND", "stock_trend_bool"): False,
                ("OUT", "momentum_score"): 3.00,
                ("OUT", "realized_vol"): 0.05,
                ("OUT", "stock_trend_bool"): True,
                ("$NDX", "market_trend_bool"): True,
            }
        )

        target_weight_ser = strategy.get_target_weight_ser(close_row_ser)

        self.assertEqual(list(target_weight_ser.index), ["AAA", "BBB", "CCC"])
        self.assertNotIn("HIGHVOL", target_weight_ser.index)
        self.assertNotIn("DOWNTREND", target_weight_ser.index)
        self.assertNotIn("OUT", target_weight_ser.index)

    def test_get_target_weight_ser_keeps_only_top_ten_by_momentum_score(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-01-31")

        universe_map = {f"S{rank_int:02d}": 1 for rank_int in range(12)}
        strategy.universe_df = pd.DataFrame(universe_map, index=[strategy.previous_bar])

        row_map: dict[tuple[str, str], float | bool] = {("$NDX", "market_trend_bool"): True}
        for rank_int in range(12):
            symbol_str = f"S{rank_int:02d}"
            row_map[(symbol_str, "momentum_score")] = 12.0 - float(rank_int)
            row_map[(symbol_str, "realized_vol")] = 0.20
            row_map[(symbol_str, "stock_trend_bool")] = True

        close_row_ser = self.make_close_row_ser(row_map)
        target_weight_ser = strategy.get_target_weight_ser(close_row_ser)

        expected_symbol_list = [f"S{rank_int:02d}" for rank_int in range(10)]
        self.assertEqual(list(target_weight_ser.index), expected_symbol_list)
        self.assertEqual(len(target_weight_ser), 10)

    def test_iterate_liquidates_all_current_positions_when_market_trend_is_false(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-01-31")
        strategy.current_bar = pd.Timestamp("2024-02-01")
        strategy.universe_df = pd.DataFrame({"AAA": [1]}, index=[strategy.previous_bar])
        strategy.add_transaction(7, strategy.previous_bar, "AAA", 10, 100.0, 1_000.0, 1, 0.0)
        strategy.current_trade_map["AAA"] = 7

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "momentum_score"): 1.00,
                ("AAA", "realized_vol"): 0.20,
                ("AAA", "stock_trend_bool"): True,
                ("$NDX", "market_trend_bool"): False,
            }
        )

        strategy.iterate(pd.DataFrame(), close_row_ser, pd.Series(dtype=float))

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)
        liquidation_order = order_list[0]
        self.assertIsInstance(liquidation_order, MarketOrder)
        self.assertEqual(liquidation_order.asset, "AAA")
        self.assertEqual(liquidation_order.unit, "shares")
        self.assertEqual(liquidation_order.amount, 0)
        self.assertTrue(liquidation_order.target)
        self.assertEqual(liquidation_order.trade_id, 7)

    def test_iterate_preserves_existing_trade_ids_adds_new_trade_ids_and_keeps_residual_cash(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-01-31")
        strategy.current_bar = pd.Timestamp("2024-02-01")
        strategy.trade_id_int = 12
        strategy.universe_df = pd.DataFrame({"AAA": [1], "BBB": [1], "CCC": [1]}, index=[strategy.previous_bar])

        strategy.add_transaction(11, strategy.previous_bar, "AAA", 10, 100.0, 1_000.0, 1, 0.0)
        strategy.add_transaction(12, strategy.previous_bar, "CCC", 5, 80.0, 400.0, 2, 0.0)
        strategy.current_trade_map["AAA"] = 11
        strategy.current_trade_map["CCC"] = 12

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "momentum_score"): 1.00,
                ("AAA", "realized_vol"): 0.20,
                ("AAA", "stock_trend_bool"): True,
                ("BBB", "momentum_score"): 0.90,
                ("BBB", "realized_vol"): 0.60,
                ("BBB", "stock_trend_bool"): True,
                ("CCC", "momentum_score"): 0.10,
                ("CCC", "realized_vol"): 0.90,
                ("CCC", "stock_trend_bool"): True,
                ("$NDX", "market_trend_bool"): True,
            }
        )

        strategy.iterate(pd.DataFrame(), close_row_ser, pd.Series(dtype=float))

        order_list = strategy.get_orders()
        self.assertEqual([order.asset for order in order_list], ["CCC", "AAA", "BBB"])

        liquidation_order = order_list[0]
        self.assertEqual(liquidation_order.trade_id, 12)
        self.assertEqual(liquidation_order.unit, "shares")
        self.assertEqual(liquidation_order.amount, 0)

        survivor_order = order_list[1]
        self.assertEqual(survivor_order.trade_id, 11)
        self.assertEqual(survivor_order.unit, "percent")
        self.assertAlmostEqual(survivor_order.amount, 0.10)

        new_entry_order = order_list[2]
        self.assertEqual(new_entry_order.trade_id, 13)
        self.assertEqual(new_entry_order.unit, "percent")
        self.assertAlmostEqual(new_entry_order.amount, 0.05)

        target_weight_sum_float = survivor_order.amount + new_entry_order.amount
        self.assertAlmostEqual(target_weight_sum_float, 0.15)
        self.assertLess(target_weight_sum_float, 1.0)
        self.assertEqual(strategy.current_trade_map["BBB"], 13)
        self.assertEqual(strategy.trade_id_int, 13)


if __name__ == "__main__":
    unittest.main()
