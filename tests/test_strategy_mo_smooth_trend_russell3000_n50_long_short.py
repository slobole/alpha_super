import os
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from strategies.momentum.strategy_mo_smooth_trend_russell3000_n50_long_short import (
    DEFAULT_CONFIG,
    SmoothTrendRussell3000N50LongShortStrategy,
)


class SmoothTrendRussell3000N50LongShortTests(unittest.TestCase):
    def make_rebalance_schedule_df(self) -> pd.DataFrame:
        rebalance_schedule_df = pd.DataFrame(
            {"decision_date_ts": [pd.Timestamp("2024-03-29")]},
            index=pd.to_datetime(["2024-04-01"]),
        )
        rebalance_schedule_df.index.name = "execution_date_ts"
        return rebalance_schedule_df

    def make_strategy(self) -> SmoothTrendRussell3000N50LongShortStrategy:
        config_obj = replace(
            DEFAULT_CONFIG,
            lookback_trading_day_int=20,
            skip_trading_day_int=2,
            quintile_count_int=5,
            capital_base_float=10_000.0,
            slippage_float=0.0,
            commission_per_share_float=0.0,
            commission_minimum_float=0.0,
        )
        return SmoothTrendRussell3000N50LongShortStrategy(
            name="SmoothTrendRussell3000N50LongShortTest",
            benchmarks=["SPY"],
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            config=config_obj,
        )

    def make_close_row_ser(self, row_map: dict[tuple[str, str], float]) -> pd.Series:
        close_row_ser = pd.Series(row_map)
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        return close_row_ser

    def test_default_config_is_n50_long_short(self):
        self.assertEqual(DEFAULT_CONFIG.variant_key_str, "russell3000_option_a_n50_long_short_close_gt_1")
        self.assertEqual(DEFAULT_CONFIG.indexname_str, "Russell 3000")
        self.assertEqual(DEFAULT_CONFIG.max_long_positions_int, 50)
        self.assertEqual(DEFAULT_CONFIG.max_short_positions_int, 50)
        self.assertAlmostEqual(DEFAULT_CONFIG.long_gross_exposure_float, 1.0)
        self.assertAlmostEqual(DEFAULT_CONFIG.short_gross_exposure_float, 1.0)
        self.assertAlmostEqual(DEFAULT_CONFIG.minimum_close_price_float, 1.0)

    def test_get_selection_df_selects_fifty_per_side_when_corner_has_capacity(self):
        strategy_obj = self.make_strategy()
        strategy_obj.previous_bar = pd.Timestamp("2024-03-29")
        symbol_list = [f"S{i:04d}" for i in range(1, 1251)]
        strategy_obj.universe_df = pd.DataFrame(
            {symbol_str: [1] for symbol_str in symbol_list},
            index=[strategy_obj.previous_bar],
        )

        row_map: dict[tuple[str, str], float] = {}
        for symbol_int, symbol_str in enumerate(symbol_list, start=1):
            row_map[(symbol_str, "Close")] = 10.0
            row_map[(symbol_str, "trend_r2_20_2_ser")] = float(symbol_int)
            row_map[(symbol_str, "trend_slope_20_2_ser")] = float(symbol_int)

        selection_df = strategy_obj.get_selection_df(close_row_ser=self.make_close_row_ser(row_map))
        long_df = selection_df.loc[selection_df["side_str"] == "long"]
        short_df = selection_df.loc[selection_df["side_str"] == "short"]

        self.assertEqual(len(long_df), 50)
        self.assertEqual(len(short_df), 50)
        self.assertEqual(long_df["symbol_str"].head(2).tolist(), ["S1250", "S1249"])
        self.assertEqual(short_df["symbol_str"].head(2).tolist(), ["S0001", "S0002"])
        self.assertAlmostEqual(float(long_df["target_weight_float"].sum()), 1.0)
        self.assertAlmostEqual(float(short_df["target_weight_float"].sum()), -1.0)
        self.assertAlmostEqual(float(selection_df["target_weight_float"].abs().sum()), 2.0)


if __name__ == "__main__":
    unittest.main()
