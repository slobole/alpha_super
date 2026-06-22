import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from strategies.vix_stuff.strategy_evrp_vix_etn_research import (
    EvrpVixEtnResearchConfig,
    EvrpVixEtnResearchStrategy,
    build_rebalanced_weight_df,
    compute_evrp_signal_df,
    run_evrp_research_backtest,
)


class EvrpVixEtnResearchTests(unittest.TestCase):
    def make_config(self, **kwargs) -> EvrpVixEtnResearchConfig:
        base_kwargs = dict(
            short_vol_symbol_str="SVXY",
            long_vol_symbol_str="VXX",
            spy_symbol_str="SPY",
            vix_symbol_str="$VIX",
            vix3m_symbol_str="$VIX3M",
            benchmark_symbol_str="SPY",
            realized_vol_lookback_int=3,
            transaction_cost_bps_float=5.0,
            rebalance_threshold_float=0.02,
            max_asset_weight_float=1.0,
            capital_base_float=100_000.0,
        )
        base_kwargs.update(kwargs)
        return EvrpVixEtnResearchConfig(**base_kwargs)

    def make_pricing_data_df(self) -> pd.DataFrame:
        date_index = pd.date_range("2024-01-02", periods=8, freq="B")
        pricing_data_df = pd.DataFrame(
            {
                ("SVXY", "Open"): [20.0, 20.0, 20.0, 20.0, 22.0, 24.2, 21.78, 22.869],
                ("SVXY", "High"): [20.0, 20.0, 20.0, 20.0, 22.0, 24.2, 21.78, 22.869],
                ("SVXY", "Low"): [20.0, 20.0, 20.0, 20.0, 22.0, 24.2, 21.78, 22.869],
                ("SVXY", "Close"): [20.0, 20.0, 20.0, 20.0, 22.0, 24.2, 21.78, 22.869],
                ("VXX", "Open"): [30.0, 30.0, 30.0, 30.0, 28.5, 27.075, 31.136, 29.579],
                ("VXX", "High"): [30.0, 30.0, 30.0, 30.0, 28.5, 27.075, 31.136, 29.579],
                ("VXX", "Low"): [30.0, 30.0, 30.0, 30.0, 28.5, 27.075, 31.136, 29.579],
                ("VXX", "Close"): [30.0, 30.0, 30.0, 30.0, 28.5, 27.075, 31.136, 29.579],
                ("SPY", "Open"): [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
                ("SPY", "High"): [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
                ("SPY", "Low"): [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
                ("SPY", "Close"): [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
                ("$VIX", "Open"): [20.0, 20.0, 20.0, 20.0, 20.0, 30.0, 30.0, 20.0],
                ("$VIX", "High"): [20.0, 20.0, 20.0, 20.0, 20.0, 30.0, 30.0, 20.0],
                ("$VIX", "Low"): [20.0, 20.0, 20.0, 20.0, 20.0, 30.0, 30.0, 20.0],
                ("$VIX", "Close"): [20.0, 20.0, 20.0, 20.0, 20.0, 30.0, 30.0, 20.0],
                ("$VIX3M", "Open"): [25.0, 25.0, 25.0, 25.0, 25.0, 20.0, 20.0, 25.0],
                ("$VIX3M", "High"): [25.0, 25.0, 25.0, 25.0, 25.0, 20.0, 20.0, 25.0],
                ("$VIX3M", "Low"): [25.0, 25.0, 25.0, 25.0, 25.0, 20.0, 20.0, 25.0],
                ("$VIX3M", "Close"): [25.0, 25.0, 25.0, 25.0, 25.0, 20.0, 20.0, 25.0],
            },
            index=date_index,
            dtype=float,
        )
        pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
        return pricing_data_df

    def test_compute_evrp_signal_df_matches_formula_and_weight_rules(self):
        config = self.make_config(realized_vol_lookback_int=3)
        date_index = pd.date_range("2024-01-02", periods=5, freq="B")
        spy_close_ser = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=date_index, dtype=float)
        vix_close_ser = pd.Series([20.0, 20.0, 20.0, 20.0, 30.0], index=date_index, dtype=float)
        vix3m_close_ser = pd.Series([25.0, 25.0, 25.0, 25.0, 20.0], index=date_index, dtype=float)

        signal_df = compute_evrp_signal_df(
            spy_close_ser=spy_close_ser,
            vix_close_ser=vix_close_ser,
            vix3m_close_ser=vix3m_close_ser,
            config=config,
        )

        expected_return_ser = spy_close_ser.pct_change(fill_method=None)
        expected_erv_float = float(expected_return_ser.iloc[1:4].std(ddof=1) * np.sqrt(252.0) * 100.0)
        self.assertAlmostEqual(float(signal_df.loc[date_index[3], "expected_realized_vol_ser"]), expected_erv_float, places=12)
        self.assertAlmostEqual(float(signal_df.loc[date_index[3], "evrp_ser"]), 20.0 - expected_erv_float, places=12)
        self.assertEqual(str(signal_df.loc[date_index[3], "regime_label_ser"]), "short_vol_high_conviction")
        self.assertAlmostEqual(float(signal_df.loc[date_index[3], "SVXY"]), 0.40, places=12)
        self.assertAlmostEqual(float(signal_df.loc[date_index[3], "VXX"]), 0.0, places=12)

        self.assertEqual(str(signal_df.loc[date_index[4], "regime_label_ser"]), "cash_mixed_signal")
        self.assertAlmostEqual(float(signal_df.loc[date_index[4], "SVXY"]), 0.0, places=12)
        self.assertAlmostEqual(float(signal_df.loc[date_index[4], "VXX"]), 0.0, places=12)

    def test_build_rebalanced_weight_df_applies_two_percent_band(self):
        date_index = pd.date_range("2024-01-02", periods=4, freq="B")
        raw_target_weight_df = pd.DataFrame(
            {
                "SVXY": [0.40, 0.41, 0.43, 0.00],
                "VXX": [0.00, 0.00, 0.00, 0.30],
            },
            index=date_index,
            dtype=float,
        )

        executed_weight_df, turnover_ser = build_rebalanced_weight_df(
            raw_target_weight_df=raw_target_weight_df,
            rebalance_threshold_float=0.02,
        )

        self.assertAlmostEqual(float(executed_weight_df.loc[date_index[0], "SVXY"]), 0.40, places=12)
        self.assertAlmostEqual(float(turnover_ser.loc[date_index[0]]), 0.40, places=12)
        self.assertAlmostEqual(float(executed_weight_df.loc[date_index[1], "SVXY"]), 0.40, places=12)
        self.assertAlmostEqual(float(turnover_ser.loc[date_index[1]]), 0.00, places=12)
        self.assertAlmostEqual(float(executed_weight_df.loc[date_index[2], "SVXY"]), 0.43, places=12)
        self.assertAlmostEqual(float(turnover_ser.loc[date_index[2]]), 0.03, places=12)
        self.assertAlmostEqual(float(executed_weight_df.loc[date_index[3], "SVXY"]), 0.00, places=12)
        self.assertAlmostEqual(float(executed_weight_df.loc[date_index[3], "VXX"]), 0.30, places=12)
        self.assertAlmostEqual(float(turnover_ser.loc[date_index[3]]), 0.73, places=12)

    def test_run_evrp_research_backtest_delays_exposure_until_after_modeled_close_fill(self):
        config = self.make_config(realized_vol_lookback_int=3, transaction_cost_bps_float=5.0)
        strategy = EvrpVixEtnResearchStrategy(
            name="EvrpVixEtnResearchTest",
            benchmarks=[config.benchmark_symbol_str],
            config=config,
            capital_base=config.capital_base_float,
        )

        run_evrp_research_backtest(
            strategy=strategy,
            pricing_data_df=self.make_pricing_data_df(),
        )

        date_index = strategy.results.index
        self.assertAlmostEqual(float(strategy.realized_weight_df.loc[date_index[3], "SVXY"]), 0.0, places=12)
        self.assertAlmostEqual(float(strategy.realized_weight_df.loc[date_index[4], "SVXY"]), 0.40, places=12)

        expected_day_5_return_float = 0.40 * 0.10 - 0.40 * 0.0005
        self.assertAlmostEqual(float(strategy.results.loc[date_index[4], "daily_returns"]), expected_day_5_return_float, places=12)
        self.assertAlmostEqual(float(strategy.signal_state_df.loc[date_index[4], "realized_turnover_ser"]), 0.40, places=12)
        self.assertFalse(strategy.metric_summary_df.empty)
        self.assertFalse(strategy.annual_return_df.empty)
        self.assertIn("Strategy", strategy.summary.columns)

    def test_one_day_delay_mode_uses_extra_close_lag(self):
        config = self.make_config(
            realized_vol_lookback_int=3,
            timing_mode_str="one_day_delayed_close",
            transaction_cost_bps_float=0.0,
        )
        strategy = EvrpVixEtnResearchStrategy(
            name="EvrpVixEtnDelayedResearchTest",
            benchmarks=[config.benchmark_symbol_str],
            config=config,
            capital_base=config.capital_base_float,
        )

        run_evrp_research_backtest(
            strategy=strategy,
            pricing_data_df=self.make_pricing_data_df(),
        )

        date_index = strategy.results.index
        self.assertAlmostEqual(float(strategy.realized_weight_df.loc[date_index[4], "SVXY"]), 0.0, places=12)
        self.assertAlmostEqual(float(strategy.realized_weight_df.loc[date_index[5], "SVXY"]), 0.40, places=12)


if __name__ == "__main__":
    unittest.main()
