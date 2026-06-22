import unittest

from strategies.taa_df.strategy_taa_df_dual_momentum_pivot5 import DEFAULT_CONFIG
from strategies.taa_df.strategy_taa_df_dual_momentum_pivot5_no_bndx import (
    NO_BNDX_CONFIG,
    STRATEGY_NAME_STR,
)


class DualMomentumPivot5NoBndxTests(unittest.TestCase):
    def test_no_bndx_config_only_removes_bndx(self):
        expected_asset_tuple = tuple(asset_str for asset_str in DEFAULT_CONFIG.asset_list if asset_str != "BNDX")

        self.assertEqual("strategy_taa_df_dual_momentum_pivot5_no_bndx", STRATEGY_NAME_STR)
        self.assertEqual(expected_asset_tuple, NO_BNDX_CONFIG.asset_list)
        self.assertNotIn("BNDX", NO_BNDX_CONFIG.asset_list)
        self.assertEqual(DEFAULT_CONFIG.selected_asset_count_int, NO_BNDX_CONFIG.selected_asset_count_int)
        self.assertEqual(DEFAULT_CONFIG.momentum_lookback_month_tuple, NO_BNDX_CONFIG.momentum_lookback_month_tuple)
        self.assertAlmostEqual(DEFAULT_CONFIG.slot_weight_float, NO_BNDX_CONFIG.slot_weight_float)


if __name__ == "__main__":
    unittest.main()
