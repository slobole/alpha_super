import unittest

import numpy as np
import pandas as pd

from scripts.research.run_sector_dispersion_marginal_universe_study import (
    ACCEPTANCE_RULE_DICT,
    STRESS_RULE_DICT,
    build_candidate_manifest_df,
    compute_tail_stress_metric_dict,
    compute_period_metric_dict,
    evaluate_acceptance_rule,
    evaluate_stress_rule,
)
from strategies.mean_reversion.strategy_mr_sector_dispersion_ibs import (
    ORIGINAL_SYMBOL_TUPLE,
    UNIVERSE_C_SYMBOL_TUPLE,
)


class SectorDispersionMarginalUniverseStudyTests(unittest.TestCase):
    def test_candidate_manifest_is_frozen_universe_c_minus_originals(self):
        candidate_manifest_df = build_candidate_manifest_df()

        self.assertEqual(len(candidate_manifest_df), len(UNIVERSE_C_SYMBOL_TUPLE) - len(ORIGINAL_SYMBOL_TUPLE))
        self.assertEqual(candidate_manifest_df["symbol_str"].tolist()[:3], ["XLF", "XLE", "XLI"])
        self.assertEqual(candidate_manifest_df["symbol_str"].tolist()[-3:], ["XHB", "XAR", "XES"])
        self.assertEqual(len(candidate_manifest_df["symbol_str"].tolist()), len(set(candidate_manifest_df["symbol_str"])))
        self.assertFalse(set(ORIGINAL_SYMBOL_TUPLE).intersection(set(candidate_manifest_df["symbol_str"])))
        self.assertTrue(candidate_manifest_df["marginal_symbol_tuple_str"].str.startswith("SOXX,IGV,IBB,").all())

    def test_acceptance_rule_requires_oos_improvement_and_low_correlation(self):
        passing_row_dict = {
            "delta_oos_sharpe_float": ACCEPTANCE_RULE_DICT["min_delta_oos_sharpe_float"] + 0.01,
            "delta_oos_max_drawdown_pct_float": ACCEPTANCE_RULE_DICT["min_delta_oos_max_drawdown_pct_float"] + 0.01,
            "standalone_corr_to_baseline_float": ACCEPTANCE_RULE_DICT["max_corr_to_baseline_float"] - 0.01,
            "delta_full_sharpe_float": ACCEPTANCE_RULE_DICT["min_delta_full_sharpe_float"] + 0.01,
            "delta_cost_drag_ann_pct_float": ACCEPTANCE_RULE_DICT["max_delta_cost_drag_ann_pct_float"] - 0.01,
        }

        accept_bool, reject_reason_str = evaluate_acceptance_rule(passing_row_dict)

        self.assertTrue(accept_bool)
        self.assertEqual(reject_reason_str, "")

        failing_row_dict = dict(passing_row_dict)
        failing_row_dict["standalone_corr_to_baseline_float"] = (
            ACCEPTANCE_RULE_DICT["max_corr_to_baseline_float"] + 0.01
        )
        failing_row_dict["delta_oos_sharpe_float"] = -0.01

        accept_bool, reject_reason_str = evaluate_acceptance_rule(failing_row_dict)

        self.assertFalse(accept_bool)
        self.assertIn("oos_sharpe_not_better", reject_reason_str)
        self.assertIn("standalone_corr_too_high_or_missing", reject_reason_str)

    def test_acceptance_rule_rejects_missing_metrics(self):
        accept_bool, reject_reason_str = evaluate_acceptance_rule(
            {
                "delta_oos_sharpe_float": np.nan,
                "delta_oos_max_drawdown_pct_float": 0.0,
                "standalone_corr_to_baseline_float": 0.20,
                "delta_full_sharpe_float": 0.0,
                "delta_cost_drag_ann_pct_float": 0.0,
            }
        )

        self.assertFalse(accept_bool)
        self.assertIn("oos_sharpe_not_better", reject_reason_str)

    def test_stress_rule_requires_tail_help_and_tail_activity(self):
        passing_row_dict = {
            "accept_bool": True,
            "base_tail_delta_mean_return_pct_float": 0.01,
            "market_tail_delta_mean_return_pct_float": 0.01,
            "base_tail_standalone_corr_to_baseline_float": (
                STRESS_RULE_DICT["max_base_tail_corr_to_baseline_float"] - 0.01
            ),
            "market_tail_standalone_corr_to_baseline_float": (
                STRESS_RULE_DICT["max_market_tail_corr_to_baseline_float"] - 0.01
            ),
            "base_tail_candidate_active_pct_float": (
                STRESS_RULE_DICT["min_base_tail_candidate_active_pct_float"] + 1.0
            ),
            "market_tail_candidate_active_pct_float": (
                STRESS_RULE_DICT["min_market_tail_candidate_active_pct_float"] + 1.0
            ),
        }

        stress_pass_bool, reject_reason_str = evaluate_stress_rule(passing_row_dict)

        self.assertTrue(stress_pass_bool)
        self.assertEqual(reject_reason_str, "")

        failing_row_dict = dict(passing_row_dict)
        failing_row_dict["market_tail_delta_mean_return_pct_float"] = -0.01
        failing_row_dict["base_tail_candidate_active_pct_float"] = 0.0

        stress_pass_bool, reject_reason_str = evaluate_stress_rule(failing_row_dict)

        self.assertFalse(stress_pass_bool)
        self.assertIn("market_tail_not_helpful", reject_reason_str)
        self.assertIn("base_tail_candidate_inactive", reject_reason_str)

    def test_compute_tail_stress_metric_dict_measures_tail_improvement(self):
        date_index = pd.bdate_range("2024-01-02", periods=100)
        baseline_return_ser = pd.Series(0.001, index=date_index, dtype=float)
        standalone_return_ser = pd.Series(0.0, index=date_index, dtype=float)
        marginal_return_ser = baseline_return_ser.copy()
        benchmark_return_ser = pd.Series(0.001, index=date_index, dtype=float)
        standalone_active_bool_ser = pd.Series(False, index=date_index)

        base_tail_idx = date_index[1:6]
        market_tail_idx = date_index[1:11]
        baseline_return_ser.loc[base_tail_idx] = -0.05
        benchmark_return_ser.loc[market_tail_idx] = -0.04
        standalone_return_ser.loc[market_tail_idx] = [-0.02, -0.01, -0.015, -0.005, 0.0, 0.005, -0.01, 0.0, 0.005, 0.01]
        marginal_return_ser.loc[base_tail_idx] = -0.04
        marginal_return_ser.loc[market_tail_idx[5:]] = 0.005
        standalone_active_bool_ser.loc[market_tail_idx] = True

        metric_dict = compute_tail_stress_metric_dict(
            baseline_return_ser=baseline_return_ser,
            standalone_return_ser=standalone_return_ser,
            marginal_return_ser=marginal_return_ser,
            benchmark_return_ser=benchmark_return_ser,
            standalone_active_bool_ser=standalone_active_bool_ser,
        )

        self.assertEqual(metric_dict["base_tail_day_count_int"], 5)
        self.assertEqual(metric_dict["market_tail_day_count_int"], 10)
        self.assertAlmostEqual(metric_dict["base_tail_delta_mean_return_pct_float"], 1.0)
        self.assertGreater(metric_dict["market_tail_delta_mean_return_pct_float"], 0.0)
        self.assertEqual(metric_dict["base_tail_candidate_active_pct_float"], 100.0)
        self.assertEqual(metric_dict["market_tail_candidate_active_pct_float"], 100.0)

    def test_compute_period_metric_dict_uses_only_requested_period(self):
        date_index = pd.bdate_range("2021-12-29", periods=6)
        total_value_ser = pd.Series(
            [100.0, 200.0, 100.0, 110.0, 121.0, 133.1],
            index=date_index,
            dtype=float,
        )

        metric_dict = compute_period_metric_dict(
            total_value_ser=total_value_ser,
            start_ts=pd.Timestamp("2022-01-03"),
            end_ts=None,
            prefix_str="oos",
        )

        period_total_value_ser = total_value_ser.loc[total_value_ser.index >= pd.Timestamp("2022-01-03")]
        expected_daily_return_ser = period_total_value_ser.pct_change(fill_method=None).dropna()
        expected_running_peak_ser = period_total_value_ser.cummax()
        expected_drawdown_ser = period_total_value_ser / expected_running_peak_ser - 1.0

        self.assertEqual(metric_dict["oos_start_date_str"], "2022-01-03")
        self.assertEqual(metric_dict["oos_end_date_str"], "2022-01-05")
        self.assertEqual(metric_dict["oos_day_count_int"], 3)
        self.assertAlmostEqual(
            metric_dict["oos_volatility_ann_pct_float"],
            float(expected_daily_return_ser.std() * np.sqrt(252.0) * 100.0),
        )
        self.assertAlmostEqual(
            metric_dict["oos_max_drawdown_pct_float"],
            float(expected_drawdown_ser.min() * 100.0),
        )


if __name__ == "__main__":
    unittest.main()
