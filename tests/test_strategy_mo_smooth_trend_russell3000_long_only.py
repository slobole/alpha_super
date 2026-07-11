import os
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from alpha.engine.order import MarketOrder
from strategies.momentum.strategy_mo_smooth_trend_russell3000_long_only import (
    DEFAULT_CONFIG,
    SmoothTrendRussell3000LongOnlyStrategy,
)


class SmoothTrendRussell3000LongOnlyTests(unittest.TestCase):
    def make_config(self):
        return replace(
            DEFAULT_CONFIG,
            lookback_trading_day_int=20,
            skip_trading_day_int=2,
            max_long_positions_int=2,
            capital_base_float=10_000.0,
            slippage_float=0.0,
            commission_per_share_float=0.0,
            commission_minimum_float=0.0,
        )

    def make_rebalance_schedule_df(
        self,
        execution_date_str: str = "2024-04-01",
        decision_date_str: str = "2024-03-29",
    ) -> pd.DataFrame:
        rebalance_schedule_df = pd.DataFrame(
            {"decision_date_ts": [pd.Timestamp(decision_date_str)]},
            index=pd.to_datetime([execution_date_str]),
        )
        rebalance_schedule_df.index.name = "execution_date_ts"
        return rebalance_schedule_df

    def make_strategy(self, **kwargs) -> SmoothTrendRussell3000LongOnlyStrategy:
        config_obj = kwargs.pop("config", self.make_config())
        base_kwargs = dict(
            name="SmoothTrendRussell3000LongOnlyTest",
            benchmarks=["SPY"],
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            config=config_obj,
        )
        base_kwargs.update(kwargs)
        return SmoothTrendRussell3000LongOnlyStrategy(**base_kwargs)

    def make_close_row_ser(self, row_map: dict[tuple[str, str], float]) -> pd.Series:
        close_row_ser = pd.Series(row_map)
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        return close_row_ser

    def make_rank_row_map(self) -> dict[tuple[str, str], float]:
        row_map: dict[tuple[str, str], float] = {}
        for symbol_int in range(1, 51):
            symbol_str = f"S{symbol_int:03d}"
            row_map[(symbol_str, "Close")] = 10.0
            row_map[(symbol_str, "trend_r2_20_2_ser")] = float(symbol_int)
            row_map[(symbol_str, "trend_slope_20_2_ser")] = float(symbol_int)
        return row_map

    def test_default_config_disables_short_side(self):
        self.assertEqual(DEFAULT_CONFIG.variant_key_str, "russell3000_option_a_n20_long_only_close_gt_1")
        self.assertEqual(DEFAULT_CONFIG.indexname_str, "Russell 3000")
        self.assertEqual(DEFAULT_CONFIG.max_long_positions_int, 20)
        self.assertEqual(DEFAULT_CONFIG.max_short_positions_int, 0)
        self.assertAlmostEqual(DEFAULT_CONFIG.long_gross_exposure_float, 1.0)
        self.assertAlmostEqual(DEFAULT_CONFIG.short_gross_exposure_float, 0.0)
        self.assertAlmostEqual(DEFAULT_CONFIG.minimum_close_price_float, 1.0)

    def test_get_selection_df_returns_only_long_targets(self):
        strategy_obj = self.make_strategy()
        strategy_obj.previous_bar = pd.Timestamp("2024-03-29")
        symbol_list = [f"S{i:03d}" for i in range(1, 51)]
        strategy_obj.universe_df = pd.DataFrame(
            {symbol_str: [1] for symbol_str in symbol_list},
            index=[strategy_obj.previous_bar],
        )

        selection_df = strategy_obj.get_selection_df(
            close_row_ser=self.make_close_row_ser(self.make_rank_row_map())
        )

        self.assertEqual(selection_df["side_str"].tolist(), ["long", "long"])
        self.assertEqual(selection_df["symbol_str"].tolist(), ["S050", "S049"])
        self.assertEqual(selection_df["target_weight_float"].tolist(), [0.5, 0.5])
        self.assertAlmostEqual(float(selection_df["target_weight_float"].sum()), 1.0)

    def test_iterate_liquidates_old_short_and_submits_only_long_targets(self):
        strategy_obj = self.make_strategy()
        strategy_obj.previous_bar = pd.Timestamp("2024-03-29")
        strategy_obj.current_bar = pd.Timestamp("2024-04-01")
        symbol_list = [f"S{i:03d}" for i in range(1, 51)]
        strategy_obj.universe_df = pd.DataFrame(
            {symbol_str: [1] for symbol_str in symbol_list},
            index=[strategy_obj.previous_bar],
        )
        strategy_obj.add_transaction(7, strategy_obj.previous_bar, "OLDL", 10, 100.0, 1_000.0, 1, 0.0)
        strategy_obj.add_transaction(8, strategy_obj.previous_bar, "OLDS", -10, 100.0, -1_000.0, 2, 0.0)
        strategy_obj.current_trade_map["OLDL"] = 7
        strategy_obj.current_trade_map["OLDS"] = 8

        strategy_obj.iterate(
            pd.DataFrame(index=[strategy_obj.previous_bar]),
            self.make_close_row_ser(self.make_rank_row_map()),
            pd.Series({symbol_str: 100.0 for symbol_str in symbol_list}, dtype=float),
        )

        order_list = strategy_obj.get_orders()
        self.assertTrue(all(isinstance(order_obj, MarketOrder) for order_obj in order_list))
        liquidation_asset_set = {
            order_obj.asset
            for order_obj in order_list
            if order_obj.unit == "shares" and order_obj.target
        }
        target_order_map = {
            order_obj.asset: order_obj
            for order_obj in order_list
            if order_obj.unit == "percent" and order_obj.target
        }

        self.assertEqual(liquidation_asset_set, {"OLDL", "OLDS"})
        self.assertEqual(set(target_order_map), {"S050", "S049"})
        self.assertAlmostEqual(float(target_order_map["S050"].amount), 0.5)
        self.assertAlmostEqual(float(target_order_map["S049"].amount), 0.5)
        self.assertEqual(len(strategy_obj.rebalance_selection_row_list), 2)


if __name__ == "__main__":
    unittest.main()
