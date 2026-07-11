import os
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from strategies.momentum.strategy_mo_smooth_trend_russell3000_n5_long_short import (
    DEFAULT_CONFIG as N5_DEFAULT_CONFIG,
    SmoothTrendRussell3000N5LongShortStrategy,
)
from strategies.momentum.strategy_mo_smooth_trend_russell3000_n10_long_short import (
    DEFAULT_CONFIG as N10_DEFAULT_CONFIG,
    SmoothTrendRussell3000N10LongShortStrategy,
)


class SmoothTrendRussell3000N5N10LongShortTests(unittest.TestCase):
    def make_rebalance_schedule_df(self) -> pd.DataFrame:
        rebalance_schedule_df = pd.DataFrame(
            {"decision_date_ts": [pd.Timestamp("2024-03-29")]},
            index=pd.to_datetime(["2024-04-01"]),
        )
        rebalance_schedule_df.index.name = "execution_date_ts"
        return rebalance_schedule_df

    def make_close_row_ser(self, row_map: dict[tuple[str, str], float]) -> pd.Series:
        close_row_ser = pd.Series(row_map)
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        return close_row_ser

    def make_row_map(self, symbol_list: list[str]) -> dict[tuple[str, str], float]:
        row_map: dict[tuple[str, str], float] = {}
        for symbol_int, symbol_str in enumerate(symbol_list, start=1):
            row_map[(symbol_str, "Close")] = 10.0
            row_map[(symbol_str, "trend_r2_20_2_ser")] = float(symbol_int)
            row_map[(symbol_str, "trend_slope_20_2_ser")] = float(symbol_int)
        return row_map

    def assert_variant_selects_expected_breadth(
        self,
        strategy_class,
        default_config_obj,
        expected_count_int: int,
        expected_variant_key_str: str,
    ) -> None:
        config_obj = replace(
            default_config_obj,
            lookback_trading_day_int=20,
            skip_trading_day_int=2,
            quintile_count_int=5,
            capital_base_float=10_000.0,
            slippage_float=0.0,
            commission_per_share_float=0.0,
            commission_minimum_float=0.0,
        )
        strategy_obj = strategy_class(
            name="SmoothTrendRussell3000BreadthTest",
            benchmarks=["SPY"],
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            config=config_obj,
        )
        strategy_obj.previous_bar = pd.Timestamp("2024-03-29")
        symbol_list = [f"S{i:04d}" for i in range(1, 251)]
        strategy_obj.universe_df = pd.DataFrame(
            {symbol_str: [1] for symbol_str in symbol_list},
            index=[strategy_obj.previous_bar],
        )

        selection_df = strategy_obj.get_selection_df(
            close_row_ser=self.make_close_row_ser(self.make_row_map(symbol_list=symbol_list))
        )
        long_df = selection_df.loc[selection_df["side_str"] == "long"]
        short_df = selection_df.loc[selection_df["side_str"] == "short"]

        self.assertEqual(default_config_obj.variant_key_str, expected_variant_key_str)
        self.assertEqual(default_config_obj.indexname_str, "Russell 3000")
        self.assertEqual(default_config_obj.max_long_positions_int, expected_count_int)
        self.assertEqual(default_config_obj.max_short_positions_int, expected_count_int)
        self.assertAlmostEqual(default_config_obj.long_gross_exposure_float, 1.0)
        self.assertAlmostEqual(default_config_obj.short_gross_exposure_float, 1.0)
        self.assertAlmostEqual(default_config_obj.minimum_close_price_float, 1.0)
        self.assertEqual(len(long_df), expected_count_int)
        self.assertEqual(len(short_df), expected_count_int)
        self.assertEqual(long_df["symbol_str"].iloc[0], "S0250")
        self.assertEqual(short_df["symbol_str"].iloc[0], "S0001")
        self.assertAlmostEqual(float(long_df["target_weight_float"].sum()), 1.0)
        self.assertAlmostEqual(float(short_df["target_weight_float"].sum()), -1.0)
        self.assertAlmostEqual(float(selection_df["target_weight_float"].abs().sum()), 2.0)

    def test_n5_selects_five_per_side_when_corner_has_capacity(self):
        self.assert_variant_selects_expected_breadth(
            strategy_class=SmoothTrendRussell3000N5LongShortStrategy,
            default_config_obj=N5_DEFAULT_CONFIG,
            expected_count_int=5,
            expected_variant_key_str="russell3000_option_a_n5_long_short_close_gt_1",
        )

    def test_n10_selects_ten_per_side_when_corner_has_capacity(self):
        self.assert_variant_selects_expected_breadth(
            strategy_class=SmoothTrendRussell3000N10LongShortStrategy,
            default_config_obj=N10_DEFAULT_CONFIG,
            expected_count_int=10,
            expected_variant_key_str="russell3000_option_a_n10_long_short_close_gt_1",
        )


if __name__ == "__main__":
    unittest.main()
