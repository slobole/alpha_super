import os
import unittest
from pathlib import Path

import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from alpha.engine.order import MarketOrder
from strategies.strategy_bom_tlt_long_short import (
    BomTltLongShortStrategy,
    build_bom_long_short_target_weight_ser,
)


class BomTltLongShortStrategyTests(unittest.TestCase):
    def make_strategy(self, daily_target_weight_ser: pd.Series) -> BomTltLongShortStrategy:
        return BomTltLongShortStrategy(
            name="BomTltLongShortTest",
            benchmarks=[],
            trade_symbol_str="TLT",
            daily_target_weight_ser=daily_target_weight_ser,
            capital_base=100_000,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )

    def test_build_bom_long_short_target_weight_ser_marks_first_five_short_then_midmonth_flat_then_rest_long(self):
        trading_day_index = pd.bdate_range("2024-01-02", "2024-02-07")

        target_weight_ser = build_bom_long_short_target_weight_ser(
            trading_day_index=trading_day_index,
            short_hold_day_count_int=5,
            long_start_trading_day_int=16,
            short_target_weight_float=-1.0,
            long_target_weight_float=1.0,
        )

        january_day_index = trading_day_index[trading_day_index.month == 1]
        february_day_index = trading_day_index[trading_day_index.month == 2]

        self.assertTrue((target_weight_ser.loc[january_day_index[:5]] == -1.0).all())
        self.assertTrue((target_weight_ser.loc[january_day_index[5:15]] == 0.0).all())
        self.assertTrue((target_weight_ser.loc[january_day_index[15:]] == 1.0).all())
        self.assertTrue((target_weight_ser.loc[february_day_index] == -1.0).all())

    def test_iterate_reverses_from_long_to_short_on_first_month_open(self):
        trading_day_index = pd.bdate_range("2024-01-02", "2024-02-07")
        daily_target_weight_ser = build_bom_long_short_target_weight_ser(
            trading_day_index=trading_day_index,
            short_hold_day_count_int=5,
            long_start_trading_day_int=16,
            short_target_weight_float=-1.0,
            long_target_weight_float=1.0,
        )
        strategy = self.make_strategy(daily_target_weight_ser)
        strategy.trade_id_int = 7
        strategy.active_trade_id_int = 7
        strategy.previous_bar = pd.Timestamp("2024-01-31")
        strategy.current_bar = pd.Timestamp("2024-02-01")
        strategy.add_transaction(
            7,
            pd.Timestamp("2024-01-24"),
            "TLT",
            100,
            100.0,
            10_000.0,
            1,
            0.0,
        )

        strategy.iterate(pd.DataFrame(), pd.Series(dtype=float), pd.Series({"TLT": 100.0}))

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 2)

        exit_order = order_list[0]
        self.assertIsInstance(exit_order, MarketOrder)
        self.assertEqual(exit_order.asset, "TLT")
        self.assertEqual(exit_order.unit, "shares")
        self.assertTrue(exit_order.target)
        self.assertEqual(exit_order.amount, 0)
        self.assertEqual(exit_order.trade_id, 7)

        entry_order = order_list[1]
        self.assertIsInstance(entry_order, MarketOrder)
        self.assertEqual(entry_order.asset, "TLT")
        self.assertEqual(entry_order.unit, "percent")
        self.assertTrue(entry_order.target)
        self.assertAlmostEqual(entry_order.amount, -1.0)
        self.assertEqual(entry_order.trade_id, 8)

    def test_iterate_exits_short_on_sixth_trading_day_open(self):
        trading_day_index = pd.bdate_range("2024-02-01", "2024-02-29")
        daily_target_weight_ser = build_bom_long_short_target_weight_ser(
            trading_day_index=trading_day_index,
            short_hold_day_count_int=5,
            long_start_trading_day_int=16,
            short_target_weight_float=-1.0,
            long_target_weight_float=1.0,
        )
        strategy = self.make_strategy(daily_target_weight_ser)
        strategy.trade_id_int = 12
        strategy.active_trade_id_int = 12

        fifth_bar_ts = pd.Timestamp(trading_day_index[4])
        sixth_bar_ts = pd.Timestamp(trading_day_index[5])
        strategy.previous_bar = fifth_bar_ts
        strategy.current_bar = sixth_bar_ts
        strategy.add_transaction(
            12,
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
        self.assertEqual(exit_order.trade_id, 12)

    def test_iterate_submits_no_order_mid_short_window(self):
        trading_day_index = pd.bdate_range("2024-02-01", "2024-02-29")
        daily_target_weight_ser = build_bom_long_short_target_weight_ser(
            trading_day_index=trading_day_index,
            short_hold_day_count_int=5,
            long_start_trading_day_int=16,
            short_target_weight_float=-1.0,
            long_target_weight_float=1.0,
        )
        strategy = self.make_strategy(daily_target_weight_ser)
        strategy.trade_id_int = 1
        strategy.active_trade_id_int = 1
        strategy.previous_bar = pd.Timestamp(trading_day_index[2])
        strategy.current_bar = pd.Timestamp(trading_day_index[3])
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

    def test_iterate_enters_long_on_sixteenth_trading_day_open(self):
        trading_day_index = pd.bdate_range("2024-02-01", "2024-02-29")
        daily_target_weight_ser = build_bom_long_short_target_weight_ser(
            trading_day_index=trading_day_index,
            short_hold_day_count_int=5,
            long_start_trading_day_int=16,
            short_target_weight_float=-1.0,
            long_target_weight_float=1.0,
        )
        strategy = self.make_strategy(daily_target_weight_ser)
        strategy.trade_id_int = 2
        strategy.active_trade_id_int = -1
        strategy.previous_bar = pd.Timestamp(trading_day_index[14])
        strategy.current_bar = pd.Timestamp(trading_day_index[15])

        strategy.iterate(pd.DataFrame(), pd.Series(dtype=float), pd.Series({"TLT": 101.0}))

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)

        entry_order = order_list[0]
        self.assertIsInstance(entry_order, MarketOrder)
        self.assertEqual(entry_order.asset, "TLT")
        self.assertEqual(entry_order.unit, "percent")
        self.assertTrue(entry_order.target)
        self.assertAlmostEqual(entry_order.amount, 1.0)
        self.assertEqual(entry_order.trade_id, 3)

    def test_iterate_submits_no_order_mid_flat_window(self):
        trading_day_index = pd.bdate_range("2024-02-01", "2024-02-29")
        daily_target_weight_ser = build_bom_long_short_target_weight_ser(
            trading_day_index=trading_day_index,
            short_hold_day_count_int=5,
            long_start_trading_day_int=16,
            short_target_weight_float=-1.0,
            long_target_weight_float=1.0,
        )
        strategy = self.make_strategy(daily_target_weight_ser)
        strategy.previous_bar = pd.Timestamp(trading_day_index[8])
        strategy.current_bar = pd.Timestamp(trading_day_index[9])

        strategy.iterate(pd.DataFrame(), pd.Series(dtype=float), pd.Series({"TLT": 100.0}))

        self.assertEqual(len(strategy.get_orders()), 0)

    def test_iterate_submits_no_order_mid_long_window(self):
        trading_day_index = pd.bdate_range("2024-02-01", "2024-02-29")
        daily_target_weight_ser = build_bom_long_short_target_weight_ser(
            trading_day_index=trading_day_index,
            short_hold_day_count_int=5,
            long_start_trading_day_int=16,
            short_target_weight_float=-1.0,
            long_target_weight_float=1.0,
        )
        strategy = self.make_strategy(daily_target_weight_ser)
        strategy.trade_id_int = 4
        strategy.active_trade_id_int = 4
        strategy.previous_bar = pd.Timestamp(trading_day_index[16])
        strategy.current_bar = pd.Timestamp(trading_day_index[17])
        strategy.add_transaction(
            4,
            pd.Timestamp(trading_day_index[15]),
            "TLT",
            100,
            100.0,
            10_000.0,
            1,
            0.0,
        )

        strategy.iterate(pd.DataFrame(), pd.Series(dtype=float), pd.Series({"TLT": 101.0}))

        self.assertEqual(len(strategy.get_orders()), 0)


if __name__ == "__main__":
    unittest.main()
