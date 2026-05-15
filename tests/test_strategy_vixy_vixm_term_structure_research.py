import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from strategies.vix_stuff.strategy_vixy_vixm_term_structure_research import (
    VixyVixmTermStructureResearchConfig,
    VixyVixmTermStructureResearchStrategy,
    build_allocation_summary_df,
    compute_close_to_close_return_df,
    compute_fixed_beta_weight_ser,
    compute_spread_daily_return_ser,
    run_article_research_backtest,
)


class VixyVixmTermStructureResearchTests(unittest.TestCase):
    def make_pricing_data_df(self) -> pd.DataFrame:
        date_index = pd.date_range("2024-01-02", periods=5, freq="B")
        pricing_data_df = pd.DataFrame(
            {
                ("VIXY", "Open"): [100.0, 110.0, 99.0, 108.9, 103.455],
                ("VIXY", "High"): [100.0, 110.0, 99.0, 108.9, 103.455],
                ("VIXY", "Low"): [100.0, 110.0, 99.0, 108.9, 103.455],
                ("VIXY", "Close"): [100.0, 110.0, 99.0, 108.9, 103.455],
                ("VIXM", "Open"): [100.0, 105.0, 102.9, 110.103, 107.900],
                ("VIXM", "High"): [100.0, 105.0, 102.9, 110.103, 107.900],
                ("VIXM", "Low"): [100.0, 105.0, 102.9, 110.103, 107.900],
                ("VIXM", "Close"): [100.0, 105.0, 102.9, 110.103, 107.900],
                ("SPY", "Open"): [400.0, 404.0, 402.0, 410.0, 408.0],
                ("SPY", "High"): [400.0, 404.0, 402.0, 410.0, 408.0],
                ("SPY", "Low"): [400.0, 404.0, 402.0, 410.0, 408.0],
                ("SPY", "Close"): [400.0, 404.0, 402.0, 410.0, 408.0],
            },
            index=date_index,
            dtype=float,
        )
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        return pricing_data_df

    def test_compute_fixed_beta_weight_ser_uses_gross_allocation_and_beta_ratio(self):
        weight_ser = compute_fixed_beta_weight_ser(
            allocation_float=0.50,
            fixed_beta_float=2.0,
            short_symbol_str="VIXY",
            hedge_symbol_str="VIXM",
        )

        self.assertAlmostEqual(float(weight_ser.loc["VIXY"]), -0.50, places=12)
        self.assertAlmostEqual(float(weight_ser.loc["VIXM"]), 1.00, places=12)
        self.assertAlmostEqual(float(weight_ser.loc["CashReserve"]), 0.50, places=12)
        self.assertAlmostEqual(abs(float(weight_ser.loc["VIXY"])) + abs(float(weight_ser.loc["VIXM"])), 1.50, places=12)

    def test_compute_spread_daily_return_ser_matches_article_weight_formula(self):
        pricing_data_df = self.make_pricing_data_df()
        close_to_close_return_df = compute_close_to_close_return_df(
            pricing_data_df=pricing_data_df,
            short_symbol_str="VIXY",
            hedge_symbol_str="VIXM",
        )
        weight_ser = compute_fixed_beta_weight_ser(
            allocation_float=0.50,
            fixed_beta_float=2.0,
            short_symbol_str="VIXY",
            hedge_symbol_str="VIXM",
        )

        spread_return_ser = compute_spread_daily_return_ser(
            close_to_close_return_df=close_to_close_return_df,
            target_weight_ser=weight_ser,
            short_symbol_str="VIXY",
            hedge_symbol_str="VIXM",
        )

        expected_day_2_return_float = -0.50 * 0.10 + 1.00 * 0.05
        expected_day_3_return_float = -0.50 * -0.10 + 1.00 * -0.02

        self.assertAlmostEqual(float(spread_return_ser.iloc[0]), 0.0, places=12)
        self.assertAlmostEqual(float(spread_return_ser.iloc[1]), expected_day_2_return_float, places=12)
        self.assertAlmostEqual(float(spread_return_ser.iloc[2]), expected_day_3_return_float, places=12)

    def test_build_allocation_summary_df_contains_article_sweep_metrics(self):
        pricing_data_df = self.make_pricing_data_df()
        config = VixyVixmTermStructureResearchConfig(
            allocation_tuple=(0.20, 0.50),
            primary_allocation_float=0.50,
            start_date_str="2024-01-02",
            end_date_str="2024-01-08",
            capital_base_float=100_000.0,
        )
        close_to_close_return_df = compute_close_to_close_return_df(
            pricing_data_df=pricing_data_df,
            short_symbol_str=config.short_symbol_str,
            hedge_symbol_str=config.hedge_symbol_str,
        )

        allocation_summary_df = build_allocation_summary_df(
            close_to_close_return_df=close_to_close_return_df,
            config=config,
        )

        self.assertEqual(allocation_summary_df.index.tolist(), [0.20, 0.50])
        self.assertIn("cagr_float", allocation_summary_df.columns)
        self.assertIn("annual_volatility_float", allocation_summary_df.columns)
        self.assertIn("sharpe_float", allocation_summary_df.columns)
        self.assertIn("gross_exposure_float", allocation_summary_df.columns)
        self.assertIn("full_sample_log_beta_diagnostic_float", allocation_summary_df.columns)
        self.assertAlmostEqual(float(allocation_summary_df.loc[0.50, "vixy_weight_float"]), -0.50, places=12)
        self.assertAlmostEqual(float(allocation_summary_df.loc[0.50, "gross_exposure_float"]), 1.50, places=12)

    def test_run_article_research_backtest_generates_strategy_summary(self):
        pricing_data_df = self.make_pricing_data_df()
        config = VixyVixmTermStructureResearchConfig(
            allocation_tuple=(0.50,),
            primary_allocation_float=0.50,
            start_date_str="2024-01-02",
            end_date_str="2024-01-08",
            capital_base_float=100_000.0,
        )
        strategy = VixyVixmTermStructureResearchStrategy(
            name="VixyVixmTermStructureResearchTest",
            benchmarks=config.benchmark_list,
            config=config,
            capital_base=config.capital_base_float,
        )

        run_article_research_backtest(
            strategy=strategy,
            pricing_data_df=pricing_data_df,
        )

        self.assertIsNotNone(strategy.summary)
        self.assertIn("Strategy", strategy.summary.columns)
        self.assertGreater(len(strategy.results), 0)
        self.assertFalse(strategy.allocation_summary_df.empty)
        self.assertAlmostEqual(float(strategy.daily_target_weight_df.iloc[-1]["CashReserve"]), 0.50, places=12)
        self.assertAlmostEqual(float(strategy.summary.loc["Exposure Time [%]", "Strategy"]), 100.0, places=12)
        self.assertTrue(np.isfinite(float(strategy.results["total_value"].iloc[-1])))


if __name__ == "__main__":
    unittest.main()
