import os
import unittest
from pathlib import Path

import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from alpha.engine.order import MarketOrder
from strategies.bom_tlt.strategy_bom_tlt_short import (
    BomTltShortStrategy,
    build_bom_target_weight_ser,
)


class BomTltShortStrategyTests(unittest.TestCase):
    def make_strategy(self, daily_target_weight_ser: pd.Series) -> BomTltShortStrategy:
        return BomTltShortStrategy(
            name="BomTltShortTest",
            benchmarks=[],
            trade_symbol_str="TLT",
            daily_target_weight_ser=daily_target_weight_ser,
            capital_base=100_000,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )

    def test_build_bom_target_weight_ser_marks_first_five_trading_days(self):
        trading_index = pd.to_datetime(
            [
                "2024-01-30",
                "2024-01-31",
                "2024-02-01",
                "2024-02-02",
                "2024-02-05",
                "2024-02-06",
                "2024-02-07",
                "2024-02-08",
                "2024-02-09",
                "2024-02-12",
                "2024-03-01",
                "2024-03-04",
            ]
        )

        target_weight_ser = build_bom_target_weight_ser(
            trading_index=trading_index,
            hold_day_count_int=5,
            target_weight_float=-1.0,
        )

        expected_weight_ser = pd.Series(
            [
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                -1.0,
            ],
            index=trading_index,
            dtype=float,
        )
        pd.testing.assert_series_equal(target_weight_ser, expected_weight_ser)

    def test_iterate_enters_short_on_first_month_open(self):
        trading_index = pd.bdate_range("2024-01-02", "2024-02-07")
        daily_target_weight_ser = build_bom_target_weight_ser(
            trading_index=trading_index,
            hold_day_count_int=5,
            target_weight_float=-1.0,
        )
        strategy = self.make_strategy(daily_target_weight_ser)
        strategy.previous_bar = pd.Timestamp("2024-01-31")
        strategy.current_bar = pd.Timestamp("2024-02-01")

        strategy.iterate(pd.DataFrame(), pd.Series(dtype=float), pd.Series({"TLT": 100.0}))

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)

        entry_order = order_list[0]
        self.assertIsInstance(entry_order, MarketOrder)
        self.assertEqual(entry_order.asset, "TLT")
        self.assertEqual(entry_order.unit, "percent")
        self.assertTrue(entry_order.target)
        self.assertAlmostEqual(entry_order.amount, -1.0)
        self.assertEqual(entry_order.trade_id, 1)

    def test_iterate_exits_short_on_first_open_after_window(self):
        trading_index = pd.to_datetime(
            [
                "2024-02-01",
                "2024-02-02",
                "2024-02-05",
                "2024-02-06",
                "2024-02-07",
                "2024-02-08",
                "2024-02-09",
            ]
        )
        daily_target_weight_ser = build_bom_target_weight_ser(
            trading_index=trading_index,
            hold_day_count_int=5,
            target_weight_float=-1.0,
        )
        strategy = self.make_strategy(daily_target_weight_ser)
        strategy.trade_id_int = 1
        strategy.active_trade_id_int = 1
        strategy.previous_bar = pd.Timestamp("2024-02-07")
        strategy.current_bar = pd.Timestamp("2024-02-08")
        strategy.add_transaction(
            1,
            pd.Timestamp("2024-02-01"),
            "TLT",
            -100,
            100.0,
            -10_000.0,
            1,
            0.0,
        )

        strategy.iterate(pd.DataFrame(), pd.Series(dtype=float), pd.Series({"TLT": 99.0}))

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)

        exit_order = order_list[0]
        self.assertIsInstance(exit_order, MarketOrder)
        self.assertEqual(exit_order.asset, "TLT")
        self.assertEqual(exit_order.unit, "shares")
        self.assertTrue(exit_order.target)
        self.assertEqual(exit_order.amount, 0)
        self.assertEqual(exit_order.trade_id, 1)

    def test_iterate_submits_no_order_mid_window(self):
        trading_index = pd.to_datetime(
            [
                "2024-02-01",
                "2024-02-02",
                "2024-02-05",
                "2024-02-06",
                "2024-02-07",
            ]
        )
        daily_target_weight_ser = build_bom_target_weight_ser(
            trading_index=trading_index,
            hold_day_count_int=5,
            target_weight_float=-1.0,
        )
        strategy = self.make_strategy(daily_target_weight_ser)
        strategy.trade_id_int = 1
        strategy.active_trade_id_int = 1
        strategy.previous_bar = pd.Timestamp("2024-02-05")
        strategy.current_bar = pd.Timestamp("2024-02-06")
        strategy.add_transaction(
            1,
            pd.Timestamp("2024-02-01"),
            "TLT",
            -100,
            100.0,
            -10_000.0,
            1,
            0.0,
        )

        strategy.iterate(pd.DataFrame(), pd.Series(dtype=float), pd.Series({"TLT": 98.0}))

        self.assertEqual(len(strategy.get_orders()), 0)


if __name__ == "__main__":
    unittest.main()
