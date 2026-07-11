import unittest

import pandas as pd

from scripts.research.run_sector_dispersion_combination_universe_study import (
    DEFAULT_TRIPLE_POOL_SYMBOL_TUPLE,
    add_composite_score_df,
    build_asset_recommendation_df,
    build_combination_manifest_df,
    evaluate_combination_acceptance_rule,
    evaluate_combination_stress_rule,
)
from scripts.research.run_sector_dispersion_marginal_universe_study import build_candidate_manifest_df


class SectorDispersionCombinationUniverseStudyTests(unittest.TestCase):
    def test_combination_manifest_has_expected_search_space(self):
        candidate_manifest_df = build_candidate_manifest_df()
        candidate_count_int = len(candidate_manifest_df)
        expected_pair_count_int = (
            len(DEFAULT_TRIPLE_POOL_SYMBOL_TUPLE)
            * (len(DEFAULT_TRIPLE_POOL_SYMBOL_TUPLE) - 1)
            // 2
        )
        expected_triple_count_int = (
            len(DEFAULT_TRIPLE_POOL_SYMBOL_TUPLE)
            * (len(DEFAULT_TRIPLE_POOL_SYMBOL_TUPLE) - 1)
            * (len(DEFAULT_TRIPLE_POOL_SYMBOL_TUPLE) - 2)
            // 6
        )

        manifest_df = build_combination_manifest_df()

        self.assertEqual(
            len(manifest_df),
            1 + candidate_count_int + expected_pair_count_int + expected_triple_count_int + 3,
        )
        self.assertEqual(manifest_df["variant_kind_str"].iloc[0], "baseline")
        normalized_addition_set = {
            ",".join(sorted(filter(None, str(addition_tuple_str).split(","))))
            for addition_tuple_str in manifest_df["addition_tuple_str"]
        }
        self.assertIn("KIE,XLRE", normalized_addition_set)
        self.assertIn("full_universe_c", set(manifest_df["variant_kind_str"]))
        self.assertEqual(len(manifest_df["addition_tuple_str"]), len(set(manifest_df["addition_tuple_str"])))

    def test_combination_manifest_can_run_full_pair_grid(self):
        candidate_manifest_df = build_candidate_manifest_df()
        candidate_count_int = len(candidate_manifest_df)
        expected_pair_count_int = candidate_count_int * (candidate_count_int - 1) // 2

        manifest_df = build_combination_manifest_df(
            include_triples_bool=False,
            include_full_universes_bool=False,
            pair_scope_str="all",
        )

        self.assertEqual(len(manifest_df), 1 + candidate_count_int + expected_pair_count_int)
        self.assertIn("XLF,XES", set(manifest_df["addition_tuple_str"]))

    def test_combination_acceptance_rule_requires_oos_and_tail_help(self):
        passing_row_dict = {
            "delta_oos_sharpe_float": 0.01,
            "delta_oos_max_drawdown_pct_float": 0.0,
            "delta_full_sharpe_float": 0.0,
            "delta_cost_drag_ann_pct_float": 0.0,
            "base_tail_delta_mean_return_pct_float": 0.01,
            "market_tail_delta_mean_return_pct_float": 0.01,
        }

        accept_bool, reject_reason_str = evaluate_combination_acceptance_rule(passing_row_dict)

        self.assertTrue(accept_bool)
        self.assertEqual(reject_reason_str, "")

        failing_row_dict = dict(passing_row_dict)
        failing_row_dict["delta_oos_sharpe_float"] = -0.01
        failing_row_dict["market_tail_delta_mean_return_pct_float"] = -0.01

        accept_bool, reject_reason_str = evaluate_combination_acceptance_rule(failing_row_dict)

        self.assertFalse(accept_bool)
        self.assertIn("oos_sharpe_not_better", reject_reason_str)
        self.assertIn("market_tail_not_helpful", reject_reason_str)

    def test_combination_stress_rule_does_not_require_low_baseline_correlation(self):
        row_dict = {
            "accept_bool": True,
            "base_tail_delta_mean_return_pct_float": 0.01,
            "market_tail_delta_mean_return_pct_float": 0.01,
            "base_tail_candidate_active_pct_float": 10.0,
            "market_tail_candidate_active_pct_float": 10.0,
            "return_corr_to_baseline_float": 0.99,
        }

        stress_pass_bool, reject_reason_str = evaluate_combination_stress_rule(row_dict)

        self.assertTrue(stress_pass_bool)
        self.assertEqual(reject_reason_str, "")

    def test_asset_recommendation_prefers_top_combo_members(self):
        candidate_manifest_df = build_candidate_manifest_df().loc[
            lambda source_df: source_df["symbol_str"].isin(["KIE", "XLRE", "XLF"])
        ]
        diagnostic_df = pd.DataFrame(
            [
                {
                    "variant_label_str": "Base+KIE+XLRE",
                    "variant_kind_str": "pair_add",
                    "addition_count_int": 2,
                    "addition_tuple_str": "KIE,XLRE",
                    "accept_bool": True,
                    "stress_pass_bool": True,
                    "delta_oos_sharpe_float": 0.30,
                    "base_tail_delta_mean_return_pct_float": 0.45,
                    "market_tail_delta_mean_return_pct_float": 0.10,
                    "delta_full_sharpe_float": 0.01,
                    "delta_oos_max_drawdown_pct_float": 1.0,
                    "delta_cost_drag_ann_pct_float": 0.05,
                },
                {
                    "variant_label_str": "Base+KIE",
                    "variant_kind_str": "single_add",
                    "addition_count_int": 1,
                    "addition_tuple_str": "KIE",
                    "accept_bool": True,
                    "stress_pass_bool": True,
                    "delta_oos_sharpe_float": 0.25,
                    "base_tail_delta_mean_return_pct_float": 0.40,
                    "market_tail_delta_mean_return_pct_float": 0.09,
                    "delta_full_sharpe_float": 0.00,
                    "delta_oos_max_drawdown_pct_float": 0.8,
                    "delta_cost_drag_ann_pct_float": 0.04,
                },
                {
                    "variant_label_str": "Base+XLF",
                    "variant_kind_str": "single_add",
                    "addition_count_int": 1,
                    "addition_tuple_str": "XLF",
                    "accept_bool": False,
                    "stress_pass_bool": False,
                    "delta_oos_sharpe_float": -0.05,
                    "base_tail_delta_mean_return_pct_float": 0.01,
                    "market_tail_delta_mean_return_pct_float": -0.01,
                    "delta_full_sharpe_float": -0.20,
                    "delta_oos_max_drawdown_pct_float": -5.0,
                    "delta_cost_drag_ann_pct_float": 0.01,
                },
            ]
        )
        leaderboard_df = add_composite_score_df(diagnostic_df)

        asset_df = build_asset_recommendation_df(
            candidate_manifest_df=candidate_manifest_df,
            leaderboard_df=leaderboard_df,
        )
        asset_by_symbol_dict = asset_df.set_index("symbol_str").to_dict("index")

        self.assertEqual(asset_by_symbol_dict["KIE"]["recommendation_tier_str"], "core_candidate")
        self.assertIn(
            asset_by_symbol_dict["XLRE"]["recommendation_tier_str"],
            {"combo_candidate", "watchlist"},
        )
        self.assertEqual(asset_by_symbol_dict["KIE"]["best_variant_label_str"], "Base+KIE+XLRE")

    def test_composite_score_rewards_larger_positive_metrics(self):
        diagnostic_df = pd.DataFrame(
            [
                {
                    "variant_label_str": "weak",
                    "addition_count_int": 1,
                    "delta_oos_sharpe_float": 0.01,
                    "base_tail_delta_mean_return_pct_float": 0.01,
                    "market_tail_delta_mean_return_pct_float": 0.01,
                    "delta_full_sharpe_float": 0.01,
                    "delta_oos_max_drawdown_pct_float": 0.01,
                    "delta_cost_drag_ann_pct_float": 0.10,
                    "accept_bool": True,
                    "stress_pass_bool": True,
                },
                {
                    "variant_label_str": "strong",
                    "addition_count_int": 1,
                    "delta_oos_sharpe_float": 0.50,
                    "base_tail_delta_mean_return_pct_float": 0.50,
                    "market_tail_delta_mean_return_pct_float": 0.50,
                    "delta_full_sharpe_float": 0.50,
                    "delta_oos_max_drawdown_pct_float": 0.50,
                    "delta_cost_drag_ann_pct_float": 0.10,
                    "accept_bool": True,
                    "stress_pass_bool": True,
                },
            ]
        )

        leaderboard_df = add_composite_score_df(diagnostic_df)

        self.assertEqual(leaderboard_df.iloc[0]["variant_label_str"], "strong")


if __name__ == "__main__":
    unittest.main()
