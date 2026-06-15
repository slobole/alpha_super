import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from strategies.momentum.strategy_mo_atr_normalized_index_vix_scaled import (
    InverseVol63VixScaledAtrNormalizedIndexStrategy,
    NASDAQ100_CONFIG,
    NASDAQ_BIOTECHNOLOGY_CONFIG,
    NYSE_COMPOSITE_CONFIG,
    RUSSELL1000_CONFIG,
    RUSSELL3000_CONFIG,
    RiskParity63AtrNormalizedIndexStrategy,
    SELECTION_SCORE_MODE_ATR20_STR,
    SELECTION_SCORE_MODE_NATR20_STR,
    SP500_CONFIG,
    TRAILING_VOL_FIELD_TEMPLATE_STR,
    VixScaledAtrNormalizedIndexStrategy,
    build_inverse_vol_63_strategy,
    build_risk_parity_63_strategy,
    build_strategy,
)
from strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled import (
    VxnScaledAtrNormalizedNdxStrategy,
)


class VixScaledAtrNormalizedIndexTests(unittest.TestCase):
    def make_rebalance_schedule_df(self) -> pd.DataFrame:
        rebalance_schedule_df = pd.DataFrame(
            {"decision_date_ts": [pd.Timestamp("2024-03-28")]},
            index=pd.to_datetime(["2024-04-01"]),
        )
        rebalance_schedule_df.index.name = "execution_date_ts"
        return rebalance_schedule_df

    def make_close_row_ser(self) -> pd.Series:
        close_row_ser = pd.Series(
            {
                ("AAA", "risk_adj_score_ser"): 2.0,
                ("AAA", "stock_trend_pass_bool"): True,
                ("AAA", "trailing_vol_63_float"): 0.10,
                ("BBB", "risk_adj_score_ser"): 1.0,
                ("BBB", "stock_trend_pass_bool"): True,
                ("BBB", "trailing_vol_63_float"): 0.20,
                ("CCC", "risk_adj_score_ser"): 3.0,
                ("CCC", "stock_trend_pass_bool"): False,
                ("CCC", "trailing_vol_63_float"): 0.05,
                ("SPY", "regime_pass_bool"): True,
            }
        )
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        return close_row_ser

    def test_default_configs_use_requested_broad_universes_and_vix(self):
        self.assertEqual(NASDAQ100_CONFIG.indexname_str, "Nasdaq 100")
        self.assertEqual(SP500_CONFIG.indexname_str, "S&P 500")
        self.assertEqual(RUSSELL1000_CONFIG.indexname_str, "Russell 1000")
        self.assertEqual(RUSSELL1000_CONFIG.regime_symbol_str, "$RUI")
        self.assertEqual(RUSSELL3000_CONFIG.indexname_str, "Russell 3000")
        self.assertEqual(NYSE_COMPOSITE_CONFIG.indexname_str, "NYSE Composite")
        self.assertEqual(NASDAQ_BIOTECHNOLOGY_CONFIG.indexname_str, "NASDAQ Biotechnology")
        self.assertEqual(NASDAQ_BIOTECHNOLOGY_CONFIG.history_start_date_str, "2001-05-21")
        self.assertEqual(NASDAQ_BIOTECHNOLOGY_CONFIG.backtest_start_date_str, "2002-06-01")
        self.assertEqual(NASDAQ100_CONFIG.vxn_symbol_str, "$VXN")
        self.assertEqual(SP500_CONFIG.vxn_symbol_str, "$VIX")
        self.assertEqual(RUSSELL1000_CONFIG.vxn_symbol_str, "$VIX")
        self.assertEqual(RUSSELL3000_CONFIG.vxn_symbol_str, "$VIX")
        self.assertEqual(NYSE_COMPOSITE_CONFIG.vxn_symbol_str, "$VIX")
        self.assertEqual(NASDAQ_BIOTECHNOLOGY_CONFIG.vxn_symbol_str, "$VIX")
        self.assertAlmostEqual(SP500_CONFIG.target_vxn_pct_float, 22.0)
        self.assertAlmostEqual(SP500_CONFIG.min_exposure_scale_float, 0.25)
        self.assertAlmostEqual(SP500_CONFIG.max_exposure_scale_float, 1.0)
        self.assertEqual(SP500_CONFIG.inverse_vol_window_int, 63)
        self.assertEqual(RUSSELL1000_CONFIG.selection_score_mode_str, SELECTION_SCORE_MODE_ATR20_STR)

    def test_strategy_reuses_original_vxn_scaled_mechanics(self):
        config_obj = replace(SP500_CONFIG, max_positions_int=2)
        strategy_obj = build_strategy(
            config=config_obj,
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            vix_scale_signal_df=pd.DataFrame(
                {"vxn_exposure_scale_float": [0.5]},
                index=pd.to_datetime(["2024-03-28"]),
            ),
        )

        self.assertIsInstance(strategy_obj, VixScaledAtrNormalizedIndexStrategy)
        self.assertIsInstance(strategy_obj, VxnScaledAtrNormalizedNdxStrategy)
        self.assertEqual(strategy_obj.max_positions_int, 2)

    def test_vix_scaled_target_weights_preserve_selection_then_scale(self):
        config_obj = replace(SP500_CONFIG, max_positions_int=2)
        strategy_obj = build_strategy(
            config=config_obj,
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            vix_scale_signal_df=pd.DataFrame(
                {"vxn_exposure_scale_float": [0.5]},
                index=pd.to_datetime(["2024-03-28"]),
            ),
        )
        strategy_obj.previous_bar = pd.Timestamp("2024-03-28")
        strategy_obj.universe_df = pd.DataFrame(
            {
                "AAA": [1],
                "BBB": [1],
                "CCC": [1],
            },
            index=[strategy_obj.previous_bar],
        )

        target_weight_ser = strategy_obj.get_target_weight_ser(
            close_row_ser=self.make_close_row_ser()
        )

        self.assertEqual(target_weight_ser.index.tolist(), ["AAA", "BBB"])
        self.assertTrue(np.allclose(target_weight_ser.to_numpy(dtype=float), 0.25))

    def test_inverse_vol_63_strategy_sets_inverse_vol_window(self):
        config_obj = replace(SP500_CONFIG, max_positions_int=2)
        strategy_obj = build_inverse_vol_63_strategy(
            config=config_obj,
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            vix_scale_signal_df=pd.DataFrame(
                {"vxn_exposure_scale_float": [0.5]},
                index=pd.to_datetime(["2024-03-28"]),
            ),
        )

        self.assertIsInstance(strategy_obj, InverseVol63VixScaledAtrNormalizedIndexStrategy)
        self.assertEqual(strategy_obj.inverse_vol_window_int, 63)
        self.assertEqual(
            strategy_obj.inverse_vol_field_str,
            TRAILING_VOL_FIELD_TEMPLATE_STR.format(window_int=63),
        )

    def test_inverse_vol_63_target_weights_normalize_selected_names_then_scale(self):
        config_obj = replace(SP500_CONFIG, max_positions_int=2)
        strategy_obj = build_inverse_vol_63_strategy(
            config=config_obj,
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            vix_scale_signal_df=pd.DataFrame(
                {"vxn_exposure_scale_float": [0.5]},
                index=pd.to_datetime(["2024-03-28"]),
            ),
        )
        strategy_obj.previous_bar = pd.Timestamp("2024-03-28")
        strategy_obj.universe_df = pd.DataFrame(
            {
                "AAA": [1],
                "BBB": [1],
                "CCC": [1],
            },
            index=[strategy_obj.previous_bar],
        )

        target_weight_ser = strategy_obj.get_target_weight_ser(
            close_row_ser=self.make_close_row_ser()
        )

        self.assertEqual(target_weight_ser.index.tolist(), ["AAA", "BBB"])
        self.assertAlmostEqual(float(target_weight_ser.loc["AAA"]), 1.0 / 3.0)
        self.assertAlmostEqual(float(target_weight_ser.loc["BBB"]), 1.0 / 6.0)

    def test_risk_parity_63_target_weights_normalize_selected_names_without_scale(self):
        config_obj = replace(SP500_CONFIG, max_positions_int=2)
        strategy_obj = build_risk_parity_63_strategy(
            config=config_obj,
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
        )
        strategy_obj.previous_bar = pd.Timestamp("2024-03-28")
        strategy_obj.universe_df = pd.DataFrame(
            {
                "AAA": [1],
                "BBB": [1],
                "CCC": [1],
            },
            index=[strategy_obj.previous_bar],
        )

        target_weight_ser = strategy_obj.get_target_weight_ser(
            close_row_ser=self.make_close_row_ser()
        )

        self.assertIsInstance(strategy_obj, RiskParity63AtrNormalizedIndexStrategy)
        self.assertEqual(target_weight_ser.index.tolist(), ["AAA", "BBB"])
        self.assertAlmostEqual(float(target_weight_ser.loc["AAA"]), 2.0 / 3.0)
        self.assertAlmostEqual(float(target_weight_ser.loc["BBB"]), 1.0 / 3.0)
        self.assertAlmostEqual(float(target_weight_ser.sum()), 1.0)

    def test_risk_parity_63_strategy_accepts_selection_score_mode(self):
        config_obj = replace(
            SP500_CONFIG,
            max_positions_int=2,
            selection_score_mode_str=SELECTION_SCORE_MODE_NATR20_STR,
        )

        strategy_obj = build_risk_parity_63_strategy(
            config=config_obj,
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
        )

        self.assertIsInstance(strategy_obj, RiskParity63AtrNormalizedIndexStrategy)
        self.assertEqual(strategy_obj.selection_score_mode_str, SELECTION_SCORE_MODE_NATR20_STR)
        self.assertTrue(strategy_obj.name.endswith("_risk_parity_63_natr20"))


if __name__ == "__main__":
    unittest.main()
