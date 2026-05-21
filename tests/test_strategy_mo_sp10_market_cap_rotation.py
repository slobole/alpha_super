import unittest
from dataclasses import replace

import pandas as pd

from strategies.momentum.strategy_mo_sp10_market_cap_rotation import (
    DEFAULT_CONFIG,
    TrueSp10RotationStrategy,
    compute_month_end_sp10_weight_df,
    get_true_sp10_rotation_data,
    map_month_end_sp10_weights_to_rebalance_open_df,
)


class TrueSp10RotationTests(unittest.TestCase):
    def make_month_end_market_cap_df(self) -> pd.DataFrame:
        market_cap_df = pd.DataFrame(
            {
                "AAA": [100.0, 300.0],
                "BBB": [200.0, 50.0],
                "CCC": [1000.0, 400.0],
                "DDD": [150.0, 100.0],
            },
            index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
        )
        market_cap_df.index.name = "decision_date_ts"
        return market_cap_df

    def make_universe_df(self) -> pd.DataFrame:
        universe_df = pd.DataFrame(
            {
                "AAA": [1, 1],
                "BBB": [1, 1],
                "CCC": [0, 1],
                "DDD": [1, 1],
            },
            index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
        )
        return universe_df

    def test_month_end_weights_use_pit_membership_and_top_market_caps(self):
        month_end_weight_df = compute_month_end_sp10_weight_df(
            market_cap_df=self.make_month_end_market_cap_df(),
            universe_df=self.make_universe_df(),
            max_positions_int=2,
        )

        self.assertAlmostEqual(float(month_end_weight_df.loc["2024-01-31", "BBB"]), 0.5)
        self.assertAlmostEqual(float(month_end_weight_df.loc["2024-01-31", "DDD"]), 0.5)
        self.assertAlmostEqual(float(month_end_weight_df.loc["2024-01-31", "CCC"]), 0.0)
        self.assertAlmostEqual(float(month_end_weight_df.loc["2024-02-29", "CCC"]), 0.5)
        self.assertAlmostEqual(float(month_end_weight_df.loc["2024-02-29", "AAA"]), 0.5)

    def test_map_month_end_weights_uses_next_tradable_open(self):
        month_end_weight_df = compute_month_end_sp10_weight_df(
            market_cap_df=self.make_month_end_market_cap_df(),
            universe_df=self.make_universe_df(),
            max_positions_int=2,
        )
        execution_index = pd.to_datetime(["2024-01-31", "2024-02-01", "2024-02-29", "2024-03-01"])

        rebalance_weight_df, rebalance_schedule_df = map_month_end_sp10_weights_to_rebalance_open_df(
            month_end_weight_df=month_end_weight_df,
            execution_index=pd.DatetimeIndex(execution_index),
        )

        self.assertEqual(pd.Timestamp("2024-02-01"), rebalance_weight_df.index[0])
        self.assertEqual(pd.Timestamp("2024-03-01"), rebalance_weight_df.index[1])
        self.assertEqual(
            pd.Timestamp("2024-01-31"),
            pd.Timestamp(rebalance_schedule_df.loc["2024-02-01", "decision_date_ts"]),
        )
        self.assertAlmostEqual(float(rebalance_weight_df.loc["2024-02-01", "BBB"]), 0.5)

    def make_strategy(self) -> TrueSp10RotationStrategy:
        rebalance_weight_df = pd.DataFrame(
            {"AAA": [0.0], "BBB": [0.5], "DDD": [0.5]},
            index=pd.to_datetime(["2024-02-01"]),
        )
        rebalance_schedule_df = pd.DataFrame(
            {"decision_date_ts": [pd.Timestamp("2024-01-31")]},
            index=pd.to_datetime(["2024-02-01"]),
        )
        return TrueSp10RotationStrategy(
            name="TrueSp10Test",
            benchmarks=["$SPX"],
            rebalance_weight_df=rebalance_weight_df,
            rebalance_schedule_df=rebalance_schedule_df,
            capital_base=100_000,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
            max_positions_int=2,
        )

    def make_close_row_ser(self) -> pd.Series:
        close_row_ser = pd.Series(
            {
                ("AAA", "Close"): 10.0,
                ("BBB", "Close"): 20.0,
                ("DDD", "Close"): 40.0,
            }
        )
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        return close_row_ser

    def test_iterate_liquidates_exits_and_submits_target_percent_orders(self):
        strategy_obj = self.make_strategy()
        strategy_obj.current_bar = pd.Timestamp("2024-02-01")
        strategy_obj.previous_bar = pd.Timestamp("2024-01-31")
        strategy_obj.add_transaction(7, pd.Timestamp("2024-01-15"), "AAA", 10, 10.0, 100.0, 1, 0.0)

        strategy_obj.iterate(
            data=pd.DataFrame(index=pd.to_datetime(["2024-01-31"])),
            close=self.make_close_row_ser(),
            open_prices=pd.Series(dtype=float),
        )

        order_list = strategy_obj.get_orders()
        self.assertEqual(len(order_list), 3)
        exit_order = [order_obj for order_obj in order_list if order_obj.asset == "AAA"][0]
        self.assertTrue(exit_order.target)
        self.assertEqual(exit_order.unit, "shares")
        self.assertEqual(float(exit_order.amount), 0.0)

        entry_order_map = {order_obj.asset: order_obj for order_obj in order_list if order_obj.asset != "AAA"}
        self.assertEqual(set(entry_order_map), {"BBB", "DDD"})
        for order_obj in entry_order_map.values():
            self.assertTrue(order_obj.target)
            self.assertEqual(order_obj.unit, "percent")
            self.assertAlmostEqual(float(order_obj.amount), 0.5)

    def test_iterate_raises_when_schedule_decision_is_not_previous_bar(self):
        strategy_obj = self.make_strategy()
        strategy_obj.current_bar = pd.Timestamp("2024-02-01")
        strategy_obj.previous_bar = pd.Timestamp("2024-01-30")

        with self.assertRaisesRegex(RuntimeError, "Schedule misalignment"):
            strategy_obj.iterate(
                data=pd.DataFrame(index=pd.to_datetime(["2024-01-30"])),
                close=self.make_close_row_ser(),
                open_prices=pd.Series(dtype=float),
            )

    def test_data_loader_requires_explicit_pit_market_cap_csv(self):
        config_obj = replace(DEFAULT_CONFIG, market_cap_csv_path_str=None)

        with self.assertRaisesRegex(RuntimeError, "point-in-time historical market-cap CSV"):
            get_true_sp10_rotation_data(config_obj)


if __name__ == "__main__":
    unittest.main()
