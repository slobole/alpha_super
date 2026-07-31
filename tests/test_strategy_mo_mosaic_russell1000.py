import os
import unittest
from pathlib import Path

import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from alpha import strategy_registry
from strategies.momentum.strategy_mo_atr_normalized_ndx_corr_penalty import (
    CorrPenaltyAtrNormalizedNdxStrategy,
)
from strategies.momentum.strategy_mo_mosaic_russell1000 import (
    DEFAULT_CONFIG,
    MosaicRussell1000Strategy,
    STRATEGY_NAME_STR,
    build_mosaic_strategy,
)


class MosaicContractTests(unittest.TestCase):
    """The MOSAIC config IS the validated contract — lock every number."""

    def test_locked_configuration(self):
        self.assertEqual(DEFAULT_CONFIG.indexname_str, "Russell 1000")
        self.assertEqual(DEFAULT_CONFIG.regime_symbol_str, "$RUI")
        self.assertEqual(DEFAULT_CONFIG.max_positions_int, 20)
        self.assertEqual(DEFAULT_CONFIG.corr_penalty_lambda_float, 0.75)
        self.assertEqual(DEFAULT_CONFIG.corr_window_int, 126)
        self.assertEqual(DEFAULT_CONFIG.corr_min_overlap_int, 63)
        self.assertEqual(DEFAULT_CONFIG.min_dollar_adv_float, 5_000_000.0)
        self.assertEqual(DEFAULT_CONFIG.adv_window_int, 20)
        self.assertEqual(DEFAULT_CONFIG.lookback_month_int, 12)
        self.assertEqual(DEFAULT_CONFIG.index_trend_window_int, 200)
        self.assertEqual(DEFAULT_CONFIG.stock_trend_window_int, 100)
        self.assertEqual(DEFAULT_CONFIG.performance_benchmark_symbol_str, "$SPX")
        self.assertEqual(DEFAULT_CONFIG.performance_benchmark_data_symbol_str, "$SPXTR")

    def test_builder_wires_config_into_strategy(self):
        rebalance_schedule_df = pd.DataFrame(
            {"decision_date_ts": [pd.Timestamp("2024-03-28")]},
            index=pd.to_datetime(["2024-04-01"]),
        )
        rebalance_schedule_df.index.name = "execution_date_ts"
        strategy_obj = build_mosaic_strategy(
            config=DEFAULT_CONFIG,
            rebalance_schedule_df=rebalance_schedule_df,
        )
        self.assertIsInstance(strategy_obj, MosaicRussell1000Strategy)
        self.assertIsInstance(strategy_obj, CorrPenaltyAtrNormalizedNdxStrategy)
        self.assertEqual(strategy_obj.name, STRATEGY_NAME_STR)
        self.assertEqual(strategy_obj.regime_symbol_str, "$RUI")
        self.assertEqual(strategy_obj.max_positions_int, 20)
        self.assertEqual(strategy_obj.corr_penalty_lambda_float, 0.75)
        self.assertEqual(strategy_obj.corr_window_int, 126)
        self.assertEqual(strategy_obj.min_dollar_adv_float, 5_000_000.0)

    def test_subclass_adds_no_logic(self):
        """MOSAIC must stay a pure naming wrapper: any behavior change belongs
        in the corr-penalty engine where the research sweeps exercise it."""
        self.assertEqual(
            [name for name in vars(MosaicRussell1000Strategy) if not name.startswith("_")],
            [],
        )

    def test_registered_pm_ready(self):
        tier_obj = strategy_registry.tier_for(
            "strategies.momentum.strategy_mo_mosaic_russell1000:MosaicRussell1000Strategy"
        )
        self.assertIs(tier_obj, strategy_registry.MaturityTier.PM_READY)


if __name__ == "__main__":
    unittest.main()
