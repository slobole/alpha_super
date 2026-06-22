from __future__ import annotations

import os
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from alpha.engine.order import MarketOrder
from strategies.dv2.strategy_mr_dv2_weekly_oversold_regime import (
    DEFAULT_CONFIG,
    NASDAQ100_CONFIG,
    RUSSELL1000_CONFIG,
    SP500_CONFIG,
    WeeklyDv2OversoldRegimeStrategy,
    map_weekly_decision_dates_to_rebalance_schedule_df,
)


class WeeklyDv2OversoldRegimeTests(unittest.TestCase):
    def make_test_config(self):
        return replace(
            DEFAULT_CONFIG,
            variant_key_str="test",
            regime_symbol_str="SPY",
            dv2_window_int=3,
            regime_sma_window_int=3,
            selection_fraction_float=0.40,
        )

    def make_rebalance_schedule_df(
        self,
        execution_date_str: str = "2024-01-08",
        decision_date_str: str = "2024-01-05",
    ) -> pd.DataFrame:
        rebalance_schedule_df = pd.DataFrame(
            {"decision_date_ts": [pd.Timestamp(decision_date_str)]},
            index=pd.to_datetime([execution_date_str]),
        )
        rebalance_schedule_df.index.name = "execution_date_ts"
        return rebalance_schedule_df

    def make_strategy(self, **kwargs) -> WeeklyDv2OversoldRegimeStrategy:
        config_obj = kwargs.pop("config", self.make_test_config())
        base_kwargs = dict(
            name="WeeklyDv2OversoldRegimeTest",
            benchmarks=[config_obj.regime_symbol_str],
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            config=config_obj,
        )
        base_kwargs.update(kwargs)
        return WeeklyDv2OversoldRegimeStrategy(**base_kwargs)

    def make_close_row_ser(self, row_map: dict[tuple[str, str], float | bool]) -> pd.Series:
        close_row_ser = pd.Series(row_map)
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        return close_row_ser

    def test_default_configs_use_matching_regime_indexes(self):
        self.assertEqual(SP500_CONFIG.indexname_str, "S&P 500")
        self.assertEqual(SP500_CONFIG.regime_symbol_str, "$SPX")
        self.assertEqual(NASDAQ100_CONFIG.indexname_str, "Nasdaq 100")
        self.assertEqual(NASDAQ100_CONFIG.regime_symbol_str, "$NDX")
        self.assertEqual(RUSSELL1000_CONFIG.indexname_str, "Russell 1000")
        self.assertEqual(RUSSELL1000_CONFIG.regime_symbol_str, "$RUI")

    def test_map_weekly_rebalance_schedule_uses_next_tradable_open(self):
        execution_index = pd.to_datetime(
            [
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-08",
                "2024-01-09",
                "2024-01-10",
                "2024-01-11",
                "2024-01-12",
                "2024-01-16",
            ]
        )
        decision_date_index = execution_index[:-1]

        rebalance_schedule_df = map_weekly_decision_dates_to_rebalance_schedule_df(
            decision_date_index=decision_date_index,
            execution_index=execution_index,
        )

        expected_schedule_df = pd.DataFrame(
            {"decision_date_ts": pd.to_datetime(["2024-01-05", "2024-01-12"])},
            index=pd.to_datetime(["2024-01-08", "2024-01-16"]),
        )
        expected_schedule_df.index.name = "execution_date_ts"
        pd.testing.assert_frame_equal(rebalance_schedule_df, expected_schedule_df)

    def test_selection_uses_lowest_dv2_top_fraction_and_pit_universe(self):
        strategy_obj = self.make_strategy()
        strategy_obj.previous_bar = pd.Timestamp("2024-01-05")
        strategy_obj.universe_df = pd.DataFrame(
            {
                "AAA": [1],
                "BBB": [1],
                "CCC": [1],
                "DDD": [1],
                "EEE": [1],
                "OUT": [0],
            },
            index=[strategy_obj.previous_bar],
        )
        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", strategy_obj.dv2_field_str): 20.0,
                ("BBB", strategy_obj.dv2_field_str): 5.0,
                ("CCC", strategy_obj.dv2_field_str): 15.0,
                ("DDD", strategy_obj.dv2_field_str): 40.0,
                ("EEE", strategy_obj.dv2_field_str): 10.0,
                ("OUT", strategy_obj.dv2_field_str): 1.0,
            }
        )

        selected_symbol_list = strategy_obj.get_selected_symbol_list(close_row_ser=close_row_ser)

        self.assertEqual(selected_symbol_list, ["BBB", "EEE"])

    def test_regime_failure_exits_all_positions_and_places_no_buys(self):
        strategy_obj = self.make_strategy()
        strategy_obj.previous_bar = pd.Timestamp("2024-01-05")
        strategy_obj.current_bar = pd.Timestamp("2024-01-08")
        strategy_obj.universe_df = pd.DataFrame(
            {
                "AAA": [1],
                "BBB": [1],
            },
            index=[strategy_obj.previous_bar],
        )
        strategy_obj.add_transaction(7, strategy_obj.previous_bar, "AAA", 10, 100.0, 1_000.0, 1, 0.0)
        strategy_obj.add_transaction(8, strategy_obj.previous_bar, "BBB", 10, 100.0, 1_000.0, 1, 0.0)
        strategy_obj.current_trade_map["AAA"] = 7
        strategy_obj.current_trade_map["BBB"] = 8

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", strategy_obj.dv2_field_str): 5.0,
                ("BBB", strategy_obj.dv2_field_str): 10.0,
                ("SPY", "regime_pass_bool"): False,
            }
        )

        strategy_obj.iterate(
            pd.DataFrame(index=[strategy_obj.previous_bar]),
            close_row_ser,
            pd.Series({"AAA": 100.0, "BBB": 100.0}, dtype=float),
        )

        order_list = strategy_obj.get_orders()
        self.assertEqual([order_obj.asset for order_obj in order_list], ["AAA", "BBB"])
        self.assertTrue(all(isinstance(order_obj, MarketOrder) for order_obj in order_list))
        self.assertTrue(all(order_obj.target for order_obj in order_list))
        self.assertTrue(all(order_obj.unit == "shares" for order_obj in order_list))
        self.assertTrue(all(float(order_obj.amount) == 0.0 for order_obj in order_list))

    def test_regime_pass_exits_non_selected_and_targets_equal_weight_selected(self):
        strategy_obj = self.make_strategy()
        strategy_obj.previous_bar = pd.Timestamp("2024-01-05")
        strategy_obj.current_bar = pd.Timestamp("2024-01-08")
        strategy_obj.universe_df = pd.DataFrame(
            {
                "AAA": [1],
                "BBB": [1],
                "CCC": [1],
                "DDD": [1],
                "EEE": [1],
            },
            index=[strategy_obj.previous_bar],
        )
        strategy_obj.add_transaction(9, strategy_obj.previous_bar, "AAA", 10, 100.0, 1_000.0, 1, 0.0)
        strategy_obj.current_trade_map["AAA"] = 9

        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", strategy_obj.dv2_field_str): 90.0,
                ("BBB", strategy_obj.dv2_field_str): 5.0,
                ("CCC", strategy_obj.dv2_field_str): 10.0,
                ("DDD", strategy_obj.dv2_field_str): 50.0,
                ("EEE", strategy_obj.dv2_field_str): 70.0,
                ("SPY", "regime_pass_bool"): True,
            }
        )

        strategy_obj.iterate(
            pd.DataFrame(index=[strategy_obj.previous_bar]),
            close_row_ser,
            pd.Series({"AAA": 100.0, "BBB": 100.0, "CCC": 100.0}, dtype=float),
        )

        order_list = strategy_obj.get_orders()
        self.assertEqual([order_obj.asset for order_obj in order_list], ["AAA", "BBB", "CCC"])
        self.assertEqual(float(order_list[0].amount), 0.0)
        self.assertEqual(order_list[0].unit, "shares")
        self.assertTrue(order_list[0].target)
        self.assertEqual([order_obj.unit for order_obj in order_list[1:]], ["percent", "percent"])
        self.assertTrue(all(order_obj.target for order_obj in order_list[1:]))
        self.assertTrue(all(float(order_obj.amount) == 0.5 for order_obj in order_list[1:]))

    def test_schedule_misalignment_fails_loud(self):
        strategy_obj = self.make_strategy()
        strategy_obj.previous_bar = pd.Timestamp("2024-01-04")
        strategy_obj.current_bar = pd.Timestamp("2024-01-08")
        strategy_obj.universe_df = pd.DataFrame({"AAA": [1]}, index=[pd.Timestamp("2024-01-04")])
        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", strategy_obj.dv2_field_str): 5.0,
                ("SPY", "regime_pass_bool"): True,
            }
        )

        with self.assertRaises(RuntimeError):
            strategy_obj.iterate(
                pd.DataFrame(index=[strategy_obj.previous_bar]),
                close_row_ser,
                pd.Series({"AAA": 100.0}, dtype=float),
            )


if __name__ == "__main__":
    unittest.main()
