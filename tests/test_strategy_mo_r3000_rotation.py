import os
import unittest
from pathlib import Path

import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from strategies.strategy_mo_ndx_rotation import NdxTrendRotationStrategy
from strategies.strategy_mo_r3000_rotation import (
    DEFAULT_CONFIG,
    R3000TrendRotationStrategy,
)


class R3000TrendRotationWrapperTests(unittest.TestCase):
    def make_rebalance_schedule_df(self) -> pd.DataFrame:
        rebalance_schedule_df = pd.DataFrame(
            {"decision_date_ts": [pd.Timestamp("2024-03-08")]},
            index=pd.to_datetime(["2024-03-11"]),
        )
        rebalance_schedule_df.index.name = "execution_date_ts"
        return rebalance_schedule_df

    def test_default_config_points_to_r3000_universe_and_spx_regime(self):
        self.assertEqual(DEFAULT_CONFIG.indexname_str, "Russell 3000")
        self.assertEqual(DEFAULT_CONFIG.regime_symbol_str, "$SPX")
        self.assertEqual(DEFAULT_CONFIG.benchmark_symbol_list, ("$SPX",))
        self.assertEqual(DEFAULT_CONFIG.max_positions_int, 10)
        self.assertEqual(DEFAULT_CONFIG.liquidity_filter_mode_str, "adv20")
        self.assertEqual(DEFAULT_CONFIG.regime_filter_mode_str, "sma")
        self.assertEqual(DEFAULT_CONFIG.position_sizing_mode_str, "equal_slot")
        self.assertEqual(DEFAULT_CONFIG.slope_window_int, 126)
        self.assertFalse(DEFAULT_CONFIG.require_scaled_rs_positive_bool)
        self.assertFalse(DEFAULT_CONFIG.require_scaled_rs_above_benchmark_bool)

    def test_strategy_alias_reuses_weekly_rotation_engine_with_spx_regime(self):
        strategy = R3000TrendRotationStrategy(
            name="R3000RotationTest",
            benchmarks=["$SPX"],
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            regime_symbol_str="$SPX",
            capital_base=100_000,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )

        self.assertIsInstance(strategy, NdxTrendRotationStrategy)
        self.assertEqual(strategy.regime_symbol_str, "$SPX")


if __name__ == "__main__":
    unittest.main()
