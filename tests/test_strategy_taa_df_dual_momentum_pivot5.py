import unittest

import numpy as np
import pandas as pd

from alpha.engine.order import MarketOrder
from strategies.taa_df.strategy_taa_df_dual_momentum_pivot5 import (
    DualMomentumPivot5Config,
    DualMomentumPivot5Strategy,
    compute_month_end_pivot5_weight_df,
    map_month_end_pivot5_weights_to_rebalance_open_df,
)


class DualMomentumPivot5Tests(unittest.TestCase):
    def make_config(self) -> DualMomentumPivot5Config:
        return DualMomentumPivot5Config(
            asset_list=("AAA", "BBB", "CCC", "DDD"),
            benchmark_list=("VT",),
            momentum_lookback_month_tuple=(1,),
            selected_asset_count_int=2,
            absolute_momentum_threshold_float=0.0,
            start_date_str="2024-01-01",
            end_date_str="2024-03-31",
            capital_base_float=100_000.0,
            slippage_float=0.0,
            commission_per_share_float=0.0,
            commission_minimum_float=0.0,
        )

    def test_month_end_weights_rank_top_assets_and_leave_failed_slots_in_cash(self):
        config = self.make_config()
        signal_close_df = pd.DataFrame(
            {
                "AAA": [100.0, 110.0],
                "BBB": [100.0, 99.0],
                "CCC": [100.0, 95.0],
                "DDD": [100.0, 90.0],
            },
            index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
            dtype=float,
        )

        momentum_score_df, month_end_weight_df = compute_month_end_pivot5_weight_df(
            signal_close_df=signal_close_df,
            config=config,
        )

        decision_date_ts = pd.Timestamp("2024-02-29")
        self.assertAlmostEqual(float(momentum_score_df.loc[decision_date_ts, "AAA"]), 0.10)
        self.assertAlmostEqual(float(momentum_score_df.loc[decision_date_ts, "BBB"]), -0.01)
        self.assertAlmostEqual(float(month_end_weight_df.loc[decision_date_ts, "AAA"]), 0.5)
        self.assertAlmostEqual(float(month_end_weight_df.loc[decision_date_ts, "BBB"]), 0.0)
        self.assertAlmostEqual(float(month_end_weight_df.loc[decision_date_ts].sum()), 0.5)

    def test_month_end_mapping_uses_actual_decision_close_and_next_month_open(self):
        month_end_weight_df = pd.DataFrame(
            {"AAA": [0.5], "BBB": [0.0]},
            index=pd.to_datetime(["2024-03-31"]),
            dtype=float,
        )
        execution_index = pd.to_datetime(["2024-03-28", "2024-04-01", "2024-04-02"])

        rebalance_weight_df, rebalance_schedule_df = map_month_end_pivot5_weights_to_rebalance_open_df(
            month_end_weight_df=month_end_weight_df,
            execution_index=pd.DatetimeIndex(execution_index),
        )

        self.assertEqual(pd.Timestamp("2024-04-01"), pd.Timestamp(rebalance_weight_df.index[0]))
        self.assertEqual(
            pd.Timestamp("2024-03-28"),
            pd.Timestamp(rebalance_schedule_df.loc[pd.Timestamp("2024-04-01"), "decision_date_ts"]),
        )
        self.assertAlmostEqual(float(rebalance_weight_df.loc[pd.Timestamp("2024-04-01"), "AAA"]), 0.5)

    def make_strategy(self) -> DualMomentumPivot5Strategy:
        rebalance_weight_df = pd.DataFrame(
            {"AAA": [0.5], "BBB": [0.0], "CCC": [0.0]},
            index=pd.to_datetime(["2024-04-01"]),
            dtype=float,
        )
        rebalance_schedule_df = pd.DataFrame(
            {"decision_date_ts": [pd.Timestamp("2024-03-29")]},
            index=pd.to_datetime(["2024-04-01"]),
        )
        return DualMomentumPivot5Strategy(
            name="Pivot5Test",
            benchmarks=[],
            rebalance_weight_df=rebalance_weight_df,
            rebalance_schedule_df=rebalance_schedule_df,
            asset_list=("AAA", "BBB", "CCC"),
            capital_base=100_000.0,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )

    def test_iterate_liquidates_failed_slot_and_buys_selected_asset(self):
        strategy_obj = self.make_strategy()
        strategy_obj.current_bar = pd.Timestamp("2024-04-01")
        strategy_obj.previous_bar = pd.Timestamp("2024-03-29")
        strategy_obj.trade_id_int = 4
        strategy_obj.add_transaction(3, strategy_obj.previous_bar, "BBB", 10, 50.0, 500.0, 1, 0.0)
        strategy_obj.current_trade_map["BBB"] = 3

        close_row_ser = pd.Series(
            {
                ("AAA", "Close"): 100.0,
                ("BBB", "Close"): 50.0,
                ("CCC", "Close"): 25.0,
            },
            dtype=float,
        )
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)

        strategy_obj.iterate(
            data=pd.DataFrame(index=pd.to_datetime(["2024-03-29"])),
            close=close_row_ser,
            open_prices=pd.Series({"AAA": 101.0, "BBB": 49.0, "CCC": 26.0}, dtype=float),
        )

        order_list = strategy_obj.get_orders()
        self.assertEqual([order_obj.asset for order_obj in order_list], ["BBB", "AAA"])

        exit_order_obj = order_list[0]
        self.assertIsInstance(exit_order_obj, MarketOrder)
        self.assertTrue(exit_order_obj.target)
        self.assertEqual(exit_order_obj.unit, "shares")
        self.assertEqual(float(exit_order_obj.amount), 0.0)
        self.assertEqual(exit_order_obj.trade_id, 3)

        entry_order_obj = order_list[1]
        self.assertIsInstance(entry_order_obj, MarketOrder)
        self.assertTrue(entry_order_obj.target)
        self.assertEqual(entry_order_obj.unit, "percent")
        self.assertAlmostEqual(float(entry_order_obj.amount), 0.5)
        self.assertEqual(entry_order_obj.trade_id, 5)
        self.assertTrue(np.isclose(float(strategy_obj.rebalance_weight_df.iloc[0].sum()), 0.5))

    def test_iterate_raises_when_decision_close_is_not_previous_bar(self):
        strategy_obj = self.make_strategy()
        strategy_obj.current_bar = pd.Timestamp("2024-04-01")
        strategy_obj.previous_bar = pd.Timestamp("2024-03-28")

        close_row_ser = pd.Series({("AAA", "Close"): 100.0}, dtype=float)
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)

        with self.assertRaisesRegex(RuntimeError, "Schedule misalignment"):
            strategy_obj.iterate(
                data=pd.DataFrame(index=pd.to_datetime(["2024-03-28"])),
                close=close_row_ser,
                open_prices=pd.Series({"AAA": 101.0}, dtype=float),
            )


if __name__ == "__main__":
    unittest.main()
