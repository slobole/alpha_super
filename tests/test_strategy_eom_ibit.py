import unittest

import pandas as pd

from alpha.engine.order import MarketOrder
from strategies.eom_tlt_vs_spy.strategy_eom_ibit import (
    EomIbitStrategy,
    build_eom_target_weight_ser,
)


class EomIbitStrategyTests(unittest.TestCase):
    def make_strategy(self, daily_target_weight_ser: pd.Series) -> EomIbitStrategy:
        return EomIbitStrategy(
            name="EomIbitTest",
            benchmarks=[],
            trade_symbol_str="IBIT",
            daily_target_weight_ser=daily_target_weight_ser,
            capital_base=100_000,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )

    def test_build_eom_target_weight_ser_marks_last_five_trading_days(self):
        trading_index = pd.to_datetime(
            [
                "2024-01-22",
                "2024-01-23",
                "2024-01-24",
                "2024-01-25",
                "2024-01-26",
                "2024-01-29",
                "2024-01-30",
                "2024-01-31",
                "2024-02-01",
                "2024-02-02",
            ]
        )

        target_weight_ser = build_eom_target_weight_ser(trading_index, hold_day_count_int=5)

        expected_weight_ser = pd.Series(
            [
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            index=trading_index,
            dtype=float,
        )
        pd.testing.assert_series_equal(target_weight_ser, expected_weight_ser)

    def test_iterate_enters_on_first_open_inside_last_five_day_window(self):
        trading_index = pd.to_datetime(
            [
                "2024-01-24",
                "2024-01-25",
                "2024-01-26",
                "2024-01-29",
                "2024-01-30",
                "2024-01-31",
                "2024-02-01",
            ]
        )
        daily_target_weight_ser = build_eom_target_weight_ser(trading_index, hold_day_count_int=5)
        strategy = self.make_strategy(daily_target_weight_ser)
        strategy.previous_bar = pd.Timestamp("2024-01-24")
        strategy.current_bar = pd.Timestamp("2024-01-25")

        strategy.iterate(pd.DataFrame(), pd.Series(dtype=float), pd.Series({"IBIT": 25.0}))

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)

        entry_order = order_list[0]
        self.assertIsInstance(entry_order, MarketOrder)
        self.assertEqual(entry_order.asset, "IBIT")
        self.assertEqual(entry_order.unit, "percent")
        self.assertTrue(entry_order.target)
        self.assertAlmostEqual(entry_order.amount, 1.0)
        self.assertEqual(entry_order.trade_id, 1)

    def test_iterate_exits_on_first_open_of_next_month(self):
        trading_index = pd.to_datetime(
            [
                "2024-01-25",
                "2024-01-26",
                "2024-01-29",
                "2024-01-30",
                "2024-01-31",
                "2024-02-01",
                "2024-02-02",
                "2024-02-05",
                "2024-02-06",
                "2024-02-07",
                "2024-02-08",
            ]
        )
        daily_target_weight_ser = build_eom_target_weight_ser(trading_index, hold_day_count_int=5)
        strategy = self.make_strategy(daily_target_weight_ser)
        strategy.trade_id_int = 1
        strategy.active_trade_id_int = 1
        strategy.previous_bar = pd.Timestamp("2024-01-31")
        strategy.current_bar = pd.Timestamp("2024-02-01")
        strategy.add_transaction(
            1,
            pd.Timestamp("2024-01-25"),
            "IBIT",
            100,
            25.0,
            2_500.0,
            1,
            0.0,
        )

        strategy.iterate(pd.DataFrame(), pd.Series(dtype=float), pd.Series({"IBIT": 26.0}))

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)

        exit_order = order_list[0]
        self.assertIsInstance(exit_order, MarketOrder)
        self.assertEqual(exit_order.asset, "IBIT")
        self.assertEqual(exit_order.unit, "shares")
        self.assertTrue(exit_order.target)
        self.assertEqual(exit_order.amount, 0)
        self.assertEqual(exit_order.trade_id, 1)

    def test_iterate_submits_no_order_mid_window(self):
        trading_index = pd.to_datetime(
            [
                "2024-01-24",
                "2024-01-25",
                "2024-01-26",
                "2024-01-29",
                "2024-01-30",
                "2024-01-31",
            ]
        )
        daily_target_weight_ser = build_eom_target_weight_ser(trading_index, hold_day_count_int=5)
        strategy = self.make_strategy(daily_target_weight_ser)
        strategy.trade_id_int = 1
        strategy.active_trade_id_int = 1
        strategy.previous_bar = pd.Timestamp("2024-01-29")
        strategy.current_bar = pd.Timestamp("2024-01-30")
        strategy.add_transaction(
            1,
            pd.Timestamp("2024-01-25"),
            "IBIT",
            100,
            25.0,
            2_500.0,
            1,
            0.0,
        )

        strategy.iterate(pd.DataFrame(), pd.Series(dtype=float), pd.Series({"IBIT": 25.5}))

        self.assertEqual(len(strategy.get_orders()), 0)

if __name__ == "__main__":
    unittest.main()
