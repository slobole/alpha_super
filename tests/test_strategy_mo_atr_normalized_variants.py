import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from strategies.momentum.strategy_mo_radge_ndx import RadgeMomentumNdxStrategy
from strategies.momentum.strategy_mo_atr_normalized_ndx import (
    AtrNormalizedNdxStrategy,
    DEFAULT_CONFIG as NDX_DEFAULT_CONFIG,
)
from strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled import (
    VxnScaledAtrNormalizedNdxStrategy,
    DEFAULT_CONFIG as VXN_SCALED_NDX_DEFAULT_CONFIG,
    compute_vxn_scale_signal_df,
    get_asof_vxn_scale_float,
)
from strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled_roc_variants import (
    ROC_MODE_ANTI_REVERSAL_SKIP_BLEND_STR,
    ROC_MODE_CONSISTENCY_SKIP_BLEND_STR,
    ROC_MODE_EQUAL_SKIP_BLEND_STR,
    ROC_MODE_LAST_12M_STR,
    ROC_MODE_LAST_1M_STR,
    ROC_MODE_LAST_3M_STR,
    ROC_MODE_PRIOR_1M_STR,
    ROC_MODE_SKIP_12_1_STR,
    ROC_MODE_SKIP_3_1_STR,
    ROC_MODE_SKIP_6_1_STR,
    ROC_MODE_WEIGHTED_SKIP_BLEND_STR,
    VxnScaledAtrNormalizedNdxRocVariantStrategy,
    build_roc_variant_config,
    compute_monthly_roc_variant_df,
)
from strategies.momentum.strategy_mo_atr_normalized_ndx_weekly import (
    DEFAULT_CONFIG as WEEKLY_NDX_DEFAULT_CONFIG,
    WeeklyAtrNormalizedNdxConfig,
    WeeklyAtrNormalizedNdxStrategy,
    compute_weekly_atr_normalized_signal_tables,
    get_weekly_decision_close_df,
    map_week_end_decision_dates_to_rebalance_schedule_df,
)
from strategies.momentum.strategy_mo_atr_normalized_ndx_weekly_kama import (
    DEFAULT_CONFIG as WEEKLY_KAMA_NDX_DEFAULT_CONFIG,
    WeeklyKamaAtrNormalizedNdxConfig,
    WeeklyKamaAtrNormalizedNdxStrategy,
    compute_kama_df,
    compute_weekly_kama_atr_normalized_signal_tables,
)
from strategies.momentum.strategy_mo_atr_normalized_sp500 import (
    AtrNormalizedSp500Strategy,
    DEFAULT_CONFIG as SP500_DEFAULT_CONFIG,
)
from strategies.momentum.strategy_mo_atr_normalized_russell1000 import (
    AtrNormalizedRussell1000Strategy,
    DEFAULT_CONFIG as RUSSELL1000_DEFAULT_CONFIG,
)


class AtrNormalizedVariantConstructionTests(unittest.TestCase):
    def make_rebalance_schedule_df(self) -> pd.DataFrame:
        rebalance_schedule_df = pd.DataFrame(
            {"decision_date_ts": [pd.Timestamp("2024-03-28")]},
            index=pd.to_datetime(["2024-04-01"]),
        )
        rebalance_schedule_df.index.name = "execution_date_ts"
        return rebalance_schedule_df

    def make_close_row_ser(self, row_map: dict[tuple[str, str], float | bool]) -> pd.Series:
        close_row_ser = pd.Series(row_map)
        close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
        return close_row_ser

    def test_ndx_default_config_points_to_nasdaq_100(self):
        self.assertEqual(NDX_DEFAULT_CONFIG.indexname_str, "Nasdaq 100")
        self.assertEqual(NDX_DEFAULT_CONFIG.regime_symbol_str, "SPY")
        self.assertEqual(NDX_DEFAULT_CONFIG.max_positions_int, 10)

    def test_weekly_ndx_default_config_points_to_nasdaq_100_with_52_week_lookback(self):
        self.assertEqual(WEEKLY_NDX_DEFAULT_CONFIG.indexname_str, "Nasdaq 100")
        self.assertEqual(WEEKLY_NDX_DEFAULT_CONFIG.regime_symbol_str, "SPY")
        self.assertEqual(WEEKLY_NDX_DEFAULT_CONFIG.max_positions_int, 10)
        self.assertEqual(WEEKLY_NDX_DEFAULT_CONFIG.lookback_week_int, 52)

    def test_weekly_kama_ndx_default_config_uses_standard_kama_parameters(self):
        self.assertEqual(WEEKLY_KAMA_NDX_DEFAULT_CONFIG.indexname_str, "Nasdaq 100")
        self.assertEqual(WEEKLY_KAMA_NDX_DEFAULT_CONFIG.lookback_week_int, 52)
        self.assertEqual(WEEKLY_KAMA_NDX_DEFAULT_CONFIG.kama_er_window_int, 10)
        self.assertEqual(WEEKLY_KAMA_NDX_DEFAULT_CONFIG.kama_fast_period_int, 2)
        self.assertEqual(WEEKLY_KAMA_NDX_DEFAULT_CONFIG.kama_slow_period_int, 30)

    def test_kama_formula_uses_causal_efficiency_ratio_recursion(self):
        price_close_df = pd.DataFrame(
            {"AAA": [1.0, 2.0, 3.0, 4.0, 5.0]},
            index=pd.bdate_range("2024-01-01", periods=5),
        )

        kama_df = compute_kama_df(
            price_close_df=price_close_df,
            er_window_int=2,
            fast_period_int=2,
            slow_period_int=4,
        )

        smoothing_constant_float = (2.0 / 3.0) ** 2.0
        expected_first_float = 3.0
        expected_second_float = expected_first_float + smoothing_constant_float * (4.0 - expected_first_float)
        expected_third_float = expected_second_float + smoothing_constant_float * (5.0 - expected_second_float)

        self.assertTrue(pd.isna(kama_df.iloc[0]["AAA"]))
        self.assertTrue(pd.isna(kama_df.iloc[1]["AAA"]))
        self.assertAlmostEqual(float(kama_df.iloc[2]["AAA"]), expected_first_float)
        self.assertAlmostEqual(float(kama_df.iloc[3]["AAA"]), expected_second_float)
        self.assertAlmostEqual(float(kama_df.iloc[4]["AAA"]), expected_third_float)

    def test_weekly_decision_close_uses_completed_week_last_tradable_close(self):
        price_close_df = pd.DataFrame(
            {"AAA": [100.0, 101.0, 102.0, 103.0, 104.0]},
            index=pd.to_datetime(
                [
                    "2024-03-25",
                    "2024-03-26",
                    "2024-03-27",
                    "2024-03-28",
                    "2024-04-01",
                ]
            ),
        )

        weekly_decision_close_df = get_weekly_decision_close_df(price_close_df)

        self.assertEqual(weekly_decision_close_df.index.tolist(), [pd.Timestamp("2024-03-28")])
        self.assertAlmostEqual(float(weekly_decision_close_df.loc["2024-03-28", "AAA"]), 103.0)

    def test_weekly_decision_close_keeps_last_friday_close(self):
        price_close_df = pd.DataFrame(
            {"AAA": [100.0, 101.0, 102.0]},
            index=pd.to_datetime(["2024-04-03", "2024-04-04", "2024-04-05"]),
        )

        weekly_decision_close_df = get_weekly_decision_close_df(price_close_df)

        self.assertEqual(weekly_decision_close_df.index.tolist(), [pd.Timestamp("2024-04-05")])

    def test_map_week_end_decision_dates_uses_next_tradable_open(self):
        decision_date_index = pd.to_datetime(["2024-03-28", "2024-04-05"])
        execution_index = pd.to_datetime(["2024-03-28", "2024-04-01", "2024-04-05", "2024-04-08"])

        rebalance_schedule_df = map_week_end_decision_dates_to_rebalance_schedule_df(
            decision_date_index=decision_date_index,
            execution_index=execution_index,
        )

        expected_schedule_df = pd.DataFrame(
            {"decision_date_ts": pd.to_datetime(["2024-03-28", "2024-04-05"])},
            index=pd.to_datetime(["2024-04-01", "2024-04-08"]),
        )
        expected_schedule_df.index.name = "execution_date_ts"
        pd.testing.assert_frame_equal(rebalance_schedule_df, expected_schedule_df)

    def test_weekly_signal_table_uses_trailing_weekly_roc_and_daily_atr(self):
        date_index = pd.bdate_range("2024-01-01", periods=30)
        price_close_df = pd.DataFrame(
            {"AAA": np.arange(100.0, 130.0)},
            index=date_index,
        )
        price_high_df = price_close_df + 1.0
        price_low_df = price_close_df - 1.0
        regime_close_ser = pd.Series(np.arange(200.0, 230.0), index=date_index)
        config = WeeklyAtrNormalizedNdxConfig(
            lookback_week_int=1,
            index_trend_window_int=2,
            stock_trend_window_int=2,
        )

        (
            weekly_decision_close_df,
            weekly_roc_df,
            atr_decision_df,
            stock_trend_pass_df,
            _regime_sma_ser,
            regime_pass_ser,
            risk_adj_score_df,
        ) = compute_weekly_atr_normalized_signal_tables(
            price_close_df=price_close_df,
            price_high_df=price_high_df,
            price_low_df=price_low_df,
            regime_close_ser=regime_close_ser,
            config=config,
        )

        latest_decision_ts = weekly_decision_close_df.index[-1]
        previous_week_decision_ts = weekly_decision_close_df.index[-2]
        expected_roc_float = (
            float(price_close_df.loc[latest_decision_ts, "AAA"])
            / float(price_close_df.loc[previous_week_decision_ts, "AAA"])
        ) - 1.0

        self.assertAlmostEqual(float(weekly_roc_df.loc[latest_decision_ts, "AAA"]), expected_roc_float)
        self.assertAlmostEqual(float(atr_decision_df.loc[latest_decision_ts, "AAA"]), 2.0)
        self.assertTrue(bool(stock_trend_pass_df.loc[latest_decision_ts, "AAA"]))
        self.assertTrue(bool(regime_pass_ser.loc[latest_decision_ts]))
        self.assertAlmostEqual(
            float(risk_adj_score_df.loc[latest_decision_ts, "AAA"]),
            expected_roc_float / 2.0,
        )

    def test_weekly_kama_signal_table_adds_close_above_kama_filter(self):
        date_index = pd.bdate_range("2024-01-01", periods=30)
        price_close_df = pd.DataFrame(
            {
                "UP": np.arange(100.0, 130.0),
                "DOWN": np.arange(130.0, 100.0, -1.0),
            },
            index=date_index,
        )
        price_high_df = price_close_df + 1.0
        price_low_df = price_close_df - 1.0
        regime_close_ser = pd.Series(np.arange(200.0, 230.0), index=date_index)
        config = WeeklyKamaAtrNormalizedNdxConfig(
            lookback_week_int=1,
            index_trend_window_int=2,
            stock_trend_window_int=2,
            kama_er_window_int=2,
            kama_fast_period_int=2,
            kama_slow_period_int=4,
        )

        (
            weekly_decision_close_df,
            _weekly_roc_df,
            _atr_decision_df,
            _stock_trend_pass_df,
            kama_decision_df,
            kama_pass_df,
            _regime_sma_ser,
            _regime_pass_ser,
            _risk_adj_score_df,
        ) = compute_weekly_kama_atr_normalized_signal_tables(
            price_close_df=price_close_df,
            price_high_df=price_high_df,
            price_low_df=price_low_df,
            regime_close_ser=regime_close_ser,
            config=config,
        )

        latest_decision_ts = weekly_decision_close_df.index[-1]
        self.assertGreater(
            float(weekly_decision_close_df.loc[latest_decision_ts, "UP"]),
            float(kama_decision_df.loc[latest_decision_ts, "UP"]),
        )
        self.assertTrue(bool(kama_pass_df.loc[latest_decision_ts, "UP"]))
        self.assertFalse(bool(kama_pass_df.loc[latest_decision_ts, "DOWN"]))

    def test_vxn_scaled_ndx_default_config_uses_conservative_no_leverage_scaler(self):
        self.assertEqual(VXN_SCALED_NDX_DEFAULT_CONFIG.indexname_str, "Nasdaq 100")
        self.assertEqual(VXN_SCALED_NDX_DEFAULT_CONFIG.vxn_symbol_str, "$VXN")
        self.assertAlmostEqual(VXN_SCALED_NDX_DEFAULT_CONFIG.target_vxn_pct_float, 22.0)
        self.assertAlmostEqual(VXN_SCALED_NDX_DEFAULT_CONFIG.min_exposure_scale_float, 0.25)
        self.assertAlmostEqual(VXN_SCALED_NDX_DEFAULT_CONFIG.max_exposure_scale_float, 1.0)

    def test_sp500_default_config_points_to_sp500(self):
        self.assertEqual(SP500_DEFAULT_CONFIG.indexname_str, "S&P 500")
        self.assertEqual(SP500_DEFAULT_CONFIG.regime_symbol_str, "SPY")
        self.assertEqual(SP500_DEFAULT_CONFIG.max_positions_int, 10)

    def test_russell1000_default_config_points_to_russell1000(self):
        self.assertEqual(RUSSELL1000_DEFAULT_CONFIG.indexname_str, "Russell 1000")
        self.assertEqual(RUSSELL1000_DEFAULT_CONFIG.regime_symbol_str, "SPY")
        self.assertEqual(RUSSELL1000_DEFAULT_CONFIG.max_positions_int, 10)

    def test_ndx_strategy_can_be_built_without_radge_wrapper_inheritance(self):
        strategy = AtrNormalizedNdxStrategy(
            name="AtrNormalizedNdxTest",
            benchmarks=[NDX_DEFAULT_CONFIG.regime_symbol_str],
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            regime_symbol_str=NDX_DEFAULT_CONFIG.regime_symbol_str,
            capital_base=NDX_DEFAULT_CONFIG.capital_base_float,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
            lookback_month_int=NDX_DEFAULT_CONFIG.lookback_month_int,
            index_trend_window_int=NDX_DEFAULT_CONFIG.index_trend_window_int,
            stock_trend_window_int=NDX_DEFAULT_CONFIG.stock_trend_window_int,
            max_positions_int=NDX_DEFAULT_CONFIG.max_positions_int,
        )

        self.assertIsInstance(strategy, AtrNormalizedNdxStrategy)
        self.assertNotIsInstance(strategy, RadgeMomentumNdxStrategy)
        self.assertEqual(strategy.max_positions_int, 10)

    def test_weekly_ndx_strategy_can_be_built_without_changing_monthly_base(self):
        strategy = WeeklyAtrNormalizedNdxStrategy(
            name="WeeklyAtrNormalizedNdxTest",
            benchmarks=[WEEKLY_NDX_DEFAULT_CONFIG.regime_symbol_str],
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            regime_symbol_str=WEEKLY_NDX_DEFAULT_CONFIG.regime_symbol_str,
            capital_base=WEEKLY_NDX_DEFAULT_CONFIG.capital_base_float,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
            lookback_week_int=WEEKLY_NDX_DEFAULT_CONFIG.lookback_week_int,
            index_trend_window_int=WEEKLY_NDX_DEFAULT_CONFIG.index_trend_window_int,
            stock_trend_window_int=WEEKLY_NDX_DEFAULT_CONFIG.stock_trend_window_int,
            max_positions_int=WEEKLY_NDX_DEFAULT_CONFIG.max_positions_int,
        )

        self.assertIsInstance(strategy, WeeklyAtrNormalizedNdxStrategy)
        self.assertIsInstance(strategy, AtrNormalizedNdxStrategy)
        self.assertEqual(strategy.lookback_week_int, 52)
        self.assertEqual(strategy.max_positions_int, 10)

    def test_weekly_kama_strategy_excludes_high_score_symbol_below_kama(self):
        strategy = WeeklyKamaAtrNormalizedNdxStrategy(
            name="WeeklyKamaAtrNormalizedNdxTest",
            benchmarks=[WEEKLY_KAMA_NDX_DEFAULT_CONFIG.regime_symbol_str],
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            regime_symbol_str=WEEKLY_KAMA_NDX_DEFAULT_CONFIG.regime_symbol_str,
            capital_base=WEEKLY_KAMA_NDX_DEFAULT_CONFIG.capital_base_float,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
            lookback_week_int=WEEKLY_KAMA_NDX_DEFAULT_CONFIG.lookback_week_int,
            index_trend_window_int=WEEKLY_KAMA_NDX_DEFAULT_CONFIG.index_trend_window_int,
            stock_trend_window_int=WEEKLY_KAMA_NDX_DEFAULT_CONFIG.stock_trend_window_int,
            max_positions_int=2,
            kama_er_window_int=WEEKLY_KAMA_NDX_DEFAULT_CONFIG.kama_er_window_int,
            kama_fast_period_int=WEEKLY_KAMA_NDX_DEFAULT_CONFIG.kama_fast_period_int,
            kama_slow_period_int=WEEKLY_KAMA_NDX_DEFAULT_CONFIG.kama_slow_period_int,
        )
        strategy.previous_bar = pd.Timestamp("2024-03-28")
        strategy.universe_df = pd.DataFrame(
            {
                "AAA": [1],
                "BBB": [1],
                "CCC": [1],
            },
            index=[strategy.previous_bar],
        )
        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "risk_adj_score_ser"): 0.90,
                ("AAA", "stock_trend_pass_bool"): True,
                ("AAA", "kama_pass_bool"): True,
                ("BBB", "risk_adj_score_ser"): 2.00,
                ("BBB", "stock_trend_pass_bool"): True,
                ("BBB", "kama_pass_bool"): False,
                ("CCC", "risk_adj_score_ser"): 0.80,
                ("CCC", "stock_trend_pass_bool"): True,
                ("CCC", "kama_pass_bool"): True,
                ("SPY", "regime_pass_bool"): True,
            }
        )

        target_weight_ser = strategy.get_target_weight_ser(close_row_ser=close_row_ser)

        self.assertEqual(target_weight_ser.index.tolist(), ["AAA", "CCC"])
        self.assertTrue(np.allclose(target_weight_ser.to_numpy(dtype=float), 0.50))

    def test_vxn_scale_signal_clips_target_vxn_over_current_vxn(self):
        vxn_close_ser = pd.Series(
            [11.0, 22.0, 44.0, 110.0],
            index=pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-28", "2024-04-30"]),
            dtype=float,
        )

        vxn_scale_signal_df = compute_vxn_scale_signal_df(
            vxn_close_ser=vxn_close_ser,
            target_vxn_pct_float=22.0,
            min_exposure_scale_float=0.25,
            max_exposure_scale_float=1.0,
        )

        self.assertTrue(
            np.allclose(
                vxn_scale_signal_df["vxn_exposure_scale_float"].to_numpy(dtype=float),
                [1.0, 1.0, 0.5, 0.25],
            )
        )

    def test_vxn_scale_asof_lookup_uses_latest_prior_observation(self):
        vxn_scale_signal_df = pd.DataFrame(
            {"vxn_exposure_scale_float": [1.0, 0.5]},
            index=pd.to_datetime(["2024-03-27", "2024-04-30"]),
        )

        exposure_scale_float = get_asof_vxn_scale_float(
            vxn_scale_signal_df=vxn_scale_signal_df,
            decision_date_ts=pd.Timestamp("2024-03-28"),
        )

        self.assertAlmostEqual(exposure_scale_float, 1.0)

    def test_vxn_scaled_strategy_preserves_selection_and_scales_total_exposure(self):
        strategy = VxnScaledAtrNormalizedNdxStrategy(
            name="VxnScaledAtrNormalizedNdxTest",
            benchmarks=[VXN_SCALED_NDX_DEFAULT_CONFIG.regime_symbol_str],
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            vxn_scale_signal_df=pd.DataFrame(
                {"vxn_exposure_scale_float": [0.5]},
                index=pd.to_datetime(["2024-03-28"]),
            ),
            regime_symbol_str=VXN_SCALED_NDX_DEFAULT_CONFIG.regime_symbol_str,
            capital_base=VXN_SCALED_NDX_DEFAULT_CONFIG.capital_base_float,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
            lookback_month_int=VXN_SCALED_NDX_DEFAULT_CONFIG.lookback_month_int,
            index_trend_window_int=VXN_SCALED_NDX_DEFAULT_CONFIG.index_trend_window_int,
            stock_trend_window_int=VXN_SCALED_NDX_DEFAULT_CONFIG.stock_trend_window_int,
            max_positions_int=5,
        )
        strategy.previous_bar = pd.Timestamp("2024-03-28")
        strategy.universe_df = pd.DataFrame(
            {
                "AAA": [1],
                "BBB": [1],
                "CCC": [1],
                "DDD": [1],
                "EEE": [1],
                "FFF": [1],
            },
            index=[strategy.previous_bar],
        )
        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "risk_adj_score_ser"): 1.00,
                ("AAA", "stock_trend_pass_bool"): True,
                ("BBB", "risk_adj_score_ser"): 0.90,
                ("BBB", "stock_trend_pass_bool"): True,
                ("CCC", "risk_adj_score_ser"): 0.80,
                ("CCC", "stock_trend_pass_bool"): True,
                ("DDD", "risk_adj_score_ser"): 0.70,
                ("DDD", "stock_trend_pass_bool"): True,
                ("EEE", "risk_adj_score_ser"): 0.60,
                ("EEE", "stock_trend_pass_bool"): True,
                ("FFF", "risk_adj_score_ser"): 2.00,
                ("FFF", "stock_trend_pass_bool"): False,
                ("SPY", "regime_pass_bool"): True,
            }
        )

        target_weight_ser = strategy.get_target_weight_ser(close_row_ser=close_row_ser)

        self.assertEqual(target_weight_ser.index.tolist(), ["AAA", "BBB", "CCC", "DDD", "EEE"])
        self.assertTrue(np.allclose(target_weight_ser.to_numpy(dtype=float), 0.10))
        self.assertAlmostEqual(float(target_weight_ser.sum()), 0.50)

    def test_vxn_scaled_roc_variant_formulas_use_expected_months(self):
        monthly_decision_close_df = pd.DataFrame(
            {
                "AAA": [100.0, 110.0, 121.0, 115.5],
                "BBB": [50.0, 45.0, 54.0, 60.0],
            },
            index=pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29", "2024-04-30"]),
        )

        last_1m_roc_df = compute_monthly_roc_variant_df(
            monthly_decision_close_df=monthly_decision_close_df,
            roc_mode_str=ROC_MODE_LAST_1M_STR,
        )
        prior_1m_roc_df = compute_monthly_roc_variant_df(
            monthly_decision_close_df=monthly_decision_close_df,
            roc_mode_str=ROC_MODE_PRIOR_1M_STR,
        )
        last_3m_roc_df = compute_monthly_roc_variant_df(
            monthly_decision_close_df=monthly_decision_close_df,
            roc_mode_str=ROC_MODE_LAST_3M_STR,
        )

        self.assertAlmostEqual(float(last_1m_roc_df.loc["2024-04-30", "AAA"]), 115.5 / 121.0 - 1.0)
        self.assertAlmostEqual(float(prior_1m_roc_df.loc["2024-04-30", "AAA"]), 121.0 / 110.0 - 1.0)
        self.assertAlmostEqual(float(last_3m_roc_df.loc["2024-04-30", "AAA"]), 115.5 / 100.0 - 1.0)
        self.assertTrue(pd.isna(prior_1m_roc_df.loc["2024-02-29", "AAA"]))

    def test_vxn_scaled_skip_month_roc_formulas_use_prior_endpoint(self):
        monthly_decision_close_df = pd.DataFrame(
            {"AAA": [100.0, 110.0, 121.0, 133.1, 146.41, 160.0, 176.0]},
            index=pd.to_datetime(
                [
                    "2024-01-31",
                    "2024-02-29",
                    "2024-03-29",
                    "2024-04-30",
                    "2024-05-31",
                    "2024-06-28",
                    "2024-07-31",
                ]
            ),
        )

        skip_3_1_roc_df = compute_monthly_roc_variant_df(
            monthly_decision_close_df=monthly_decision_close_df,
            roc_mode_str=ROC_MODE_SKIP_3_1_STR,
        )
        skip_6_1_roc_df = compute_monthly_roc_variant_df(
            monthly_decision_close_df=monthly_decision_close_df,
            roc_mode_str=ROC_MODE_SKIP_6_1_STR,
        )

        self.assertAlmostEqual(float(skip_3_1_roc_df.loc["2024-04-30", "AAA"]), 121.0 / 100.0 - 1.0)
        self.assertAlmostEqual(float(skip_6_1_roc_df.loc["2024-07-31", "AAA"]), 160.0 / 100.0 - 1.0)

    def test_vxn_scaled_12_1_and_blend_formulas_are_explicit(self):
        monthly_decision_close_df = pd.DataFrame(
            {"AAA": [float(100 + month_int * 10) for month_int in range(13)]},
            index=pd.date_range("2024-01-31", periods=13, freq="ME"),
        )

        skip_3_1_roc_df = compute_monthly_roc_variant_df(
            monthly_decision_close_df=monthly_decision_close_df,
            roc_mode_str=ROC_MODE_SKIP_3_1_STR,
        )
        skip_6_1_roc_df = compute_monthly_roc_variant_df(
            monthly_decision_close_df=monthly_decision_close_df,
            roc_mode_str=ROC_MODE_SKIP_6_1_STR,
        )
        skip_12_1_roc_df = compute_monthly_roc_variant_df(
            monthly_decision_close_df=monthly_decision_close_df,
            roc_mode_str=ROC_MODE_SKIP_12_1_STR,
        )
        last_12m_roc_df = compute_monthly_roc_variant_df(
            monthly_decision_close_df=monthly_decision_close_df,
            roc_mode_str=ROC_MODE_LAST_12M_STR,
        )
        equal_blend_df = compute_monthly_roc_variant_df(
            monthly_decision_close_df=monthly_decision_close_df,
            roc_mode_str=ROC_MODE_EQUAL_SKIP_BLEND_STR,
        )
        weighted_blend_df = compute_monthly_roc_variant_df(
            monthly_decision_close_df=monthly_decision_close_df,
            roc_mode_str=ROC_MODE_WEIGHTED_SKIP_BLEND_STR,
        )
        consistency_blend_df = compute_monthly_roc_variant_df(
            monthly_decision_close_df=monthly_decision_close_df,
            roc_mode_str=ROC_MODE_CONSISTENCY_SKIP_BLEND_STR,
        )
        anti_reversal_blend_df = compute_monthly_roc_variant_df(
            monthly_decision_close_df=monthly_decision_close_df,
            roc_mode_str=ROC_MODE_ANTI_REVERSAL_SKIP_BLEND_STR,
        )
        last_1m_roc_df = compute_monthly_roc_variant_df(
            monthly_decision_close_df=monthly_decision_close_df,
            roc_mode_str=ROC_MODE_LAST_1M_STR,
        )

        latest_date_ts = monthly_decision_close_df.index[-1]
        skip_3_1_float = float(skip_3_1_roc_df.loc[latest_date_ts, "AAA"])
        skip_6_1_float = float(skip_6_1_roc_df.loc[latest_date_ts, "AAA"])
        skip_12_1_float = float(skip_12_1_roc_df.loc[latest_date_ts, "AAA"])
        last_12m_float = float(last_12m_roc_df.loc[latest_date_ts, "AAA"])
        expected_equal_float = (skip_3_1_float + skip_6_1_float + skip_12_1_float) / 3.0
        expected_weighted_float = (
            0.20 * skip_3_1_float
            + 0.30 * skip_6_1_float
            + 0.50 * skip_12_1_float
        )
        expected_anti_reversal_float = expected_weighted_float - (
            0.25 * float(last_1m_roc_df.loc[latest_date_ts, "AAA"])
        )

        self.assertAlmostEqual(skip_12_1_float, 210.0 / 100.0 - 1.0)
        self.assertAlmostEqual(last_12m_float, 220.0 / 100.0 - 1.0)
        self.assertAlmostEqual(float(equal_blend_df.loc[latest_date_ts, "AAA"]), expected_equal_float)
        self.assertAlmostEqual(float(weighted_blend_df.loc[latest_date_ts, "AAA"]), expected_weighted_float)
        self.assertAlmostEqual(float(consistency_blend_df.loc[latest_date_ts, "AAA"]), expected_equal_float)
        self.assertAlmostEqual(
            float(anti_reversal_blend_df.loc[latest_date_ts, "AAA"]),
            expected_anti_reversal_float,
        )

    def test_vxn_scaled_roc_variant_rejects_invalid_mode(self):
        monthly_decision_close_df = pd.DataFrame(
            {"AAA": [100.0, 110.0]},
            index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
        )

        with self.assertRaises(ValueError):
            compute_monthly_roc_variant_df(
                monthly_decision_close_df=monthly_decision_close_df,
                roc_mode_str="bad_mode",
            )

    def test_vxn_scaled_roc_variant_constructor_sets_required_history(self):
        strategy = VxnScaledAtrNormalizedNdxRocVariantStrategy(
            name="VxnScaledAtrNormalizedNdxPrior1mTest",
            benchmarks=[VXN_SCALED_NDX_DEFAULT_CONFIG.regime_symbol_str],
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            vxn_scale_signal_df=pd.DataFrame(
                {"vxn_exposure_scale_float": [0.5]},
                index=pd.to_datetime(["2024-03-28"]),
            ),
            roc_mode_str=ROC_MODE_PRIOR_1M_STR,
            regime_symbol_str=VXN_SCALED_NDX_DEFAULT_CONFIG.regime_symbol_str,
            capital_base=VXN_SCALED_NDX_DEFAULT_CONFIG.capital_base_float,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
            index_trend_window_int=VXN_SCALED_NDX_DEFAULT_CONFIG.index_trend_window_int,
            stock_trend_window_int=VXN_SCALED_NDX_DEFAULT_CONFIG.stock_trend_window_int,
            max_positions_int=5,
        )

        self.assertEqual(strategy.roc_mode_str, ROC_MODE_PRIOR_1M_STR)
        self.assertEqual(strategy.lookback_month_int, 2)

    def test_vxn_scaled_roc_variant_accepts_explicit_atr_window(self):
        config_obj = build_roc_variant_config(
            roc_mode_str=ROC_MODE_LAST_12M_STR,
            atr_window_int=63,
        )
        strategy = VxnScaledAtrNormalizedNdxRocVariantStrategy(
            name="VxnScaledAtrNormalizedNdxAtr63Test",
            benchmarks=[VXN_SCALED_NDX_DEFAULT_CONFIG.regime_symbol_str],
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            vxn_scale_signal_df=pd.DataFrame(
                {"vxn_exposure_scale_float": [0.5]},
                index=pd.to_datetime(["2024-03-28"]),
            ),
            roc_mode_str=config_obj.roc_mode_str,
            atr_window_int=config_obj.atr_window_int,
            regime_symbol_str=VXN_SCALED_NDX_DEFAULT_CONFIG.regime_symbol_str,
            capital_base=VXN_SCALED_NDX_DEFAULT_CONFIG.capital_base_float,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
            index_trend_window_int=VXN_SCALED_NDX_DEFAULT_CONFIG.index_trend_window_int,
            stock_trend_window_int=VXN_SCALED_NDX_DEFAULT_CONFIG.stock_trend_window_int,
            max_positions_int=5,
        )

        self.assertEqual(config_obj.atr_window_int, 63)
        self.assertEqual(strategy.roc_mode_str, ROC_MODE_LAST_12M_STR)
        self.assertEqual(strategy.lookback_month_int, 12)
        self.assertEqual(strategy.atr_window_int, 63)

    def test_sp500_wrapper_strategy_can_be_built(self):
        strategy = AtrNormalizedSp500Strategy(
            name="AtrNormalizedSp500Test",
            benchmarks=[SP500_DEFAULT_CONFIG.regime_symbol_str],
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            regime_symbol_str=SP500_DEFAULT_CONFIG.regime_symbol_str,
            capital_base=SP500_DEFAULT_CONFIG.capital_base_float,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
            lookback_month_int=SP500_DEFAULT_CONFIG.lookback_month_int,
            index_trend_window_int=SP500_DEFAULT_CONFIG.index_trend_window_int,
            stock_trend_window_int=SP500_DEFAULT_CONFIG.stock_trend_window_int,
            max_positions_int=SP500_DEFAULT_CONFIG.max_positions_int,
        )

        self.assertIsInstance(strategy, RadgeMomentumNdxStrategy)
        self.assertEqual(strategy.max_positions_int, 10)

    def test_russell1000_wrapper_strategy_can_be_built(self):
        strategy = AtrNormalizedRussell1000Strategy(
            name="AtrNormalizedRussell1000Test",
            benchmarks=[RUSSELL1000_DEFAULT_CONFIG.regime_symbol_str],
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            regime_symbol_str=RUSSELL1000_DEFAULT_CONFIG.regime_symbol_str,
            capital_base=RUSSELL1000_DEFAULT_CONFIG.capital_base_float,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
            lookback_month_int=RUSSELL1000_DEFAULT_CONFIG.lookback_month_int,
            index_trend_window_int=RUSSELL1000_DEFAULT_CONFIG.index_trend_window_int,
            stock_trend_window_int=RUSSELL1000_DEFAULT_CONFIG.stock_trend_window_int,
            max_positions_int=RUSSELL1000_DEFAULT_CONFIG.max_positions_int,
        )

        self.assertIsInstance(strategy, RadgeMomentumNdxStrategy)
        self.assertEqual(strategy.max_positions_int, 10)


if __name__ == "__main__":
    unittest.main()
