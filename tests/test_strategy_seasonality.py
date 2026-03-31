import os
import unittest
from pathlib import Path

import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from alpha.engine.order import MarketOrder
from strategies.strategy_seasonality import (
    DEFAULT_MONTH_TARGET_WEIGHT_MAP,
    SeasonalityStrategy,
)


class SeasonalityStrategyTests(unittest.TestCase):
    def make_strategy(self) -> SeasonalityStrategy:
        return SeasonalityStrategy(
            name="SeasonalityTest",
            benchmarks=[],
            month_target_weight_map=DEFAULT_MONTH_TARGET_WEIGHT_MAP,
            capital_base=100_000,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )

    def test_iterate_submits_no_orders_outside_july_boundary_dates(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-06-13")
        strategy.current_bar = pd.Timestamp("2024-06-14")

        strategy.iterate(pd.DataFrame(), pd.Series(dtype=float), pd.Series({"XLU": 70.0}))

        self.assertEqual(len(strategy.get_orders()), 0)

    def test_iterate_enters_gld_on_first_tradable_january_open(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-12-31")
        strategy.current_bar = pd.Timestamp("2025-01-02")

        strategy.iterate(pd.DataFrame(), pd.Series(dtype=float), pd.Series({"GLD": 190.0}))

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)

        entry_order = order_list[0]
        self.assertIsInstance(entry_order, MarketOrder)
        self.assertEqual(entry_order.asset, "GLD")
        self.assertEqual(entry_order.unit, "percent")
        self.assertTrue(entry_order.target)
        self.assertAlmostEqual(entry_order.amount, 1.0)
        self.assertEqual(entry_order.trade_id, 1)

    def test_iterate_rotates_from_gld_to_dba_on_first_tradable_february_open(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2025-01-31")
        strategy.current_bar = pd.Timestamp("2025-02-03")
        strategy.trade_id_int = 2
        strategy.current_trade_map["GLD"] = 2
        strategy.add_transaction(
            2,
            pd.Timestamp("2025-01-02"),
            "GLD",
            100,
            190.0,
            19_000.0,
            1,
            0.0,
        )

        strategy.iterate(pd.DataFrame(), pd.Series(dtype=float), pd.Series({"GLD": 192.0, "DBA": 24.0}))

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 2)

        exit_order = order_list[0]
        self.assertIsInstance(exit_order, MarketOrder)
        self.assertEqual(exit_order.asset, "GLD")
        self.assertEqual(exit_order.unit, "shares")
        self.assertTrue(exit_order.target)
        self.assertEqual(exit_order.amount, 0)
        self.assertEqual(exit_order.trade_id, 2)

        entry_order = order_list[1]
        self.assertIsInstance(entry_order, MarketOrder)
        self.assertEqual(entry_order.asset, "DBA")
        self.assertEqual(entry_order.unit, "percent")
        self.assertTrue(entry_order.target)
        self.assertAlmostEqual(entry_order.amount, 1.0)
        self.assertEqual(entry_order.trade_id, 3)

    def test_iterate_rotates_from_dba_to_xlu_on_first_tradable_march_open(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-02-29")
        strategy.current_bar = pd.Timestamp("2024-03-01")
        strategy.trade_id_int = 4
        strategy.current_trade_map["DBA"] = 4
        strategy.add_transaction(
            4,
            pd.Timestamp("2024-02-03"),
            "DBA",
            100,
            24.0,
            2_400.0,
            1,
            0.0,
        )

        strategy.iterate(pd.DataFrame(), pd.Series(dtype=float), pd.Series({"DBA": 24.5, "XLU": 68.0}))

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 2)

        exit_order = order_list[0]
        self.assertIsInstance(exit_order, MarketOrder)
        self.assertEqual(exit_order.asset, "DBA")
        self.assertEqual(exit_order.unit, "shares")
        self.assertTrue(exit_order.target)
        self.assertEqual(exit_order.amount, 0)
        self.assertEqual(exit_order.trade_id, 4)

        entry_order = order_list[1]
        self.assertIsInstance(entry_order, MarketOrder)
        self.assertEqual(entry_order.asset, "XLU")
        self.assertEqual(entry_order.unit, "percent")
        self.assertTrue(entry_order.target)
        self.assertAlmostEqual(entry_order.amount, 1.0)
        self.assertEqual(entry_order.trade_id, 5)

    def test_iterate_submits_no_order_when_same_asset_continues_from_march_to_april(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-03-28")
        strategy.current_bar = pd.Timestamp("2024-04-01")
        strategy.trade_id_int = 3
        strategy.current_trade_map["XLU"] = 3
        strategy.add_transaction(
            3,
            pd.Timestamp("2024-03-01"),
            "XLU",
            100,
            68.0,
            6_800.0,
            1,
            0.0,
        )

        strategy.iterate(pd.DataFrame(), pd.Series(dtype=float), pd.Series({"XLU": 69.0}))

        self.assertEqual(len(strategy.get_orders()), 0)

    def test_iterate_exits_on_first_tradable_may_open(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-04-30")
        strategy.current_bar = pd.Timestamp("2024-05-01")
        strategy.trade_id_int = 5
        strategy.current_trade_map["XLU"] = 5
        strategy.add_transaction(
            5,
            pd.Timestamp("2024-03-01"),
            "XLU",
            100,
            68.0,
            6_800.0,
            1,
            0.0,
        )

        strategy.iterate(pd.DataFrame(), pd.Series(dtype=float), pd.Series({"XLU": 70.0}))

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)

        exit_order = order_list[0]
        self.assertIsInstance(exit_order, MarketOrder)
        self.assertEqual(exit_order.asset, "XLU")
        self.assertEqual(exit_order.unit, "shares")
        self.assertTrue(exit_order.target)
        self.assertEqual(exit_order.amount, 0)
        self.assertEqual(exit_order.trade_id, 5)

    def test_iterate_enters_on_first_tradable_july_open(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-06-28")
        strategy.current_bar = pd.Timestamp("2024-07-01")

        strategy.iterate(pd.DataFrame(), pd.Series(dtype=float), pd.Series({"XLU": 71.0}))

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)

        entry_order = order_list[0]
        self.assertIsInstance(entry_order, MarketOrder)
        self.assertEqual(entry_order.asset, "XLU")
        self.assertEqual(entry_order.unit, "percent")
        self.assertTrue(entry_order.target)
        self.assertAlmostEqual(entry_order.amount, 1.0)
        self.assertEqual(entry_order.trade_id, 1)

    def test_iterate_submits_no_duplicate_order_mid_july_when_already_long(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2024-07-15")
        strategy.current_bar = pd.Timestamp("2024-07-16")
        strategy.trade_id_int = 1
        strategy.current_trade_map["XLU"] = 1
        strategy.add_transaction(
            1,
            pd.Timestamp("2024-07-01"),
            "XLU",
            100,
            70.0,
            7_000.0,
            1,
            0.0,
        )

        strategy.iterate(pd.DataFrame(), pd.Series(dtype=float), pd.Series({"XLU": 72.0}))

        self.assertEqual(len(strategy.get_orders()), 0)

    def test_iterate_rotates_from_xlu_to_ief_on_first_tradable_august_open(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2022-07-29")
        strategy.current_bar = pd.Timestamp("2022-08-01")
        strategy.trade_id_int = 7
        strategy.current_trade_map["XLU"] = 7
        strategy.add_transaction(
            7,
            pd.Timestamp("2022-07-01"),
            "XLU",
            100,
            65.0,
            6_500.0,
            1,
            0.0,
        )

        strategy.iterate(pd.DataFrame(), pd.Series(dtype=float), pd.Series({"XLU": 66.0, "IEF": 102.0}))

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 2)

        exit_order = order_list[0]
        self.assertIsInstance(exit_order, MarketOrder)
        self.assertEqual(exit_order.asset, "XLU")
        self.assertEqual(exit_order.unit, "shares")
        self.assertTrue(exit_order.target)
        self.assertEqual(exit_order.amount, 0)
        self.assertEqual(exit_order.trade_id, 7)

        entry_order = order_list[1]
        self.assertIsInstance(entry_order, MarketOrder)
        self.assertEqual(entry_order.asset, "IEF")
        self.assertEqual(entry_order.unit, "percent")
        self.assertTrue(entry_order.target)
        self.assertAlmostEqual(entry_order.amount, 1.0)
        self.assertEqual(entry_order.trade_id, 8)

    def test_iterate_exits_ief_on_first_tradable_september_open(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2022-08-31")
        strategy.current_bar = pd.Timestamp("2022-09-01")
        strategy.trade_id_int = 9
        strategy.current_trade_map["IEF"] = 9
        strategy.add_transaction(
            9,
            pd.Timestamp("2022-08-01"),
            "IEF",
            100,
            102.0,
            10_200.0,
            1,
            0.0,
        )

        strategy.iterate(pd.DataFrame(), pd.Series(dtype=float), pd.Series({"IEF": 101.0}))

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)

        exit_order = order_list[0]
        self.assertIsInstance(exit_order, MarketOrder)
        self.assertEqual(exit_order.asset, "IEF")
        self.assertEqual(exit_order.unit, "shares")
        self.assertTrue(exit_order.target)
        self.assertEqual(exit_order.amount, 0)
        self.assertEqual(exit_order.trade_id, 9)

    def test_iterate_uses_month_boundary_not_calendar_adjacency(self):
        strategy = self.make_strategy()
        strategy.previous_bar = pd.Timestamp("2023-06-30")
        strategy.current_bar = pd.Timestamp("2023-07-03")

        strategy.iterate(pd.DataFrame(), pd.Series(dtype=float), pd.Series({"XLU": 69.0}))

        order_list = strategy.get_orders()
        self.assertEqual(len(order_list), 1)

        entry_order = order_list[0]
        self.assertIsInstance(entry_order, MarketOrder)
        self.assertEqual(entry_order.asset, "XLU")
        self.assertEqual(entry_order.unit, "percent")
        self.assertTrue(entry_order.target)
        self.assertAlmostEqual(entry_order.amount, 1.0)


if __name__ == "__main__":
    unittest.main()
