import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from strategies.momentum.strategy_mo_atr_normalized_ndx import AtrNormalizedNdxStrategy
from strategies.momentum.strategy_mo_atr_normalized_ndx_corr_penalty import (
    CorrPenaltyAtrNormalizedNdxConfig,
    CorrPenaltyAtrNormalizedNdxStrategy,
    select_corr_penalized_symbol_list,
)


def make_corr_df(symbol_list: list[str], corr_map: dict[tuple[str, str], float]) -> pd.DataFrame:
    corr_df = pd.DataFrame(np.eye(len(symbol_list)), index=symbol_list, columns=symbol_list)
    for (symbol_a_str, symbol_b_str), corr_float in corr_map.items():
        corr_df.loc[symbol_a_str, symbol_b_str] = corr_float
        corr_df.loc[symbol_b_str, symbol_a_str] = corr_float
    return corr_df


class SelectCorrPenalizedSymbolListTests(unittest.TestCase):
    def test_lambda_zero_reproduces_score_ranked_top_n(self):
        candidate_score_ser = pd.Series({"AAA": 2.0, "BBB": 1.9, "CCC": 1.2, "DDD": 0.5})
        candidate_corr_df = make_corr_df(
            ["AAA", "BBB", "CCC", "DDD"],
            {("AAA", "BBB"): 0.95, ("AAA", "CCC"): 0.9, ("AAA", "DDD"): 0.9},
        )
        selected_symbol_list = select_corr_penalized_symbol_list(
            candidate_score_ser=candidate_score_ser,
            candidate_corr_df=candidate_corr_df,
            max_positions_int=3,
            corr_penalty_lambda_float=0.0,
        )
        self.assertEqual(selected_symbol_list, ["AAA", "BBB", "CCC"])

    def test_high_corr_duplicate_is_demoted_below_diversifier(self):
        # AMD-style near-duplicate of the top pick loses its slot to a
        # lower-score diversifier.
        candidate_score_ser = pd.Series({"NVDA": 2.0, "AMD": 1.9, "COST": 1.2})
        candidate_corr_df = make_corr_df(
            ["NVDA", "AMD", "COST"],
            {("NVDA", "AMD"): 0.9, ("NVDA", "COST"): 0.1, ("AMD", "COST"): 0.2},
        )
        selected_symbol_list = select_corr_penalized_symbol_list(
            candidate_score_ser=candidate_score_ser,
            candidate_corr_df=candidate_corr_df,
            max_positions_int=2,
            corr_penalty_lambda_float=1.0,
        )
        # AMD adjusted: 1.9 - 1.0 * 0.9 * 1.9 = 0.19; COST: 1.2 - 0.1 * 1.2 = 1.08.
        self.assertEqual(selected_symbol_list, ["NVDA", "COST"])

    def test_penalty_is_sign_safe_for_negative_scores(self):
        # A multiplicative penalty score * (1 - lambda * corr) would flip the
        # ranking for negative scores: high correlation would *shrink* a
        # negative score toward zero and improve its rank. The sign-safe form
        # must rank the correlated negative-score name lower.
        candidate_score_ser = pd.Series({"AAA": 2.0, "BBB": -0.5, "CCC": -0.1})
        candidate_corr_df = make_corr_df(
            ["AAA", "BBB", "CCC"],
            {("AAA", "BBB"): 0.9, ("AAA", "CCC"): 0.0, ("BBB", "CCC"): 0.0},
        )
        selected_symbol_list = select_corr_penalized_symbol_list(
            candidate_score_ser=candidate_score_ser,
            candidate_corr_df=candidate_corr_df,
            max_positions_int=2,
            corr_penalty_lambda_float=1.0,
        )
        # BBB adjusted: -0.5 - 0.9 * 0.5 = -0.95; CCC adjusted: -0.1.
        self.assertEqual(selected_symbol_list, ["AAA", "CCC"])

    def test_nan_corr_falls_back_to_median_valid_correlation(self):
        # BBB has no correlation history versus AAA. The fallback must be the
        # median valid off-diagonal correlation (0.8 here), so BBB gets no
        # free diversification credit and CCC wins the second slot.
        candidate_score_ser = pd.Series({"AAA": 2.0, "BBB": 1.5, "CCC": 1.45})
        candidate_corr_df = make_corr_df(
            ["AAA", "BBB", "CCC"],
            {("AAA", "BBB"): np.nan, ("AAA", "CCC"): 0.8, ("BBB", "CCC"): 0.8},
        )
        selected_symbol_list = select_corr_penalized_symbol_list(
            candidate_score_ser=candidate_score_ser,
            candidate_corr_df=candidate_corr_df,
            max_positions_int=2,
            corr_penalty_lambda_float=1.0,
        )
        # BBB adjusted: 1.5 - 0.8 * 1.5 = 0.30; CCC adjusted: 1.45 - 0.8 * 1.45 = 0.29.
        # BBB still wins here, but only because its raw score is higher under
        # the *same* fallback correlation as CCC's real one.
        self.assertEqual(selected_symbol_list, ["AAA", "BBB"])

        # Lower BBB's raw score below CCC and the fallback must not save it.
        candidate_score_ser = pd.Series({"AAA": 2.0, "BBB": 1.40, "CCC": 1.45})
        selected_symbol_list = select_corr_penalized_symbol_list(
            candidate_score_ser=candidate_score_ser,
            candidate_corr_df=candidate_corr_df,
            max_positions_int=2,
            corr_penalty_lambda_float=1.0,
        )
        self.assertEqual(selected_symbol_list, ["AAA", "CCC"])

    def test_fewer_candidates_than_slots_returns_all(self):
        candidate_score_ser = pd.Series({"AAA": 2.0, "BBB": 1.0})
        candidate_corr_df = make_corr_df(["AAA", "BBB"], {("AAA", "BBB"): 0.5})
        selected_symbol_list = select_corr_penalized_symbol_list(
            candidate_score_ser=candidate_score_ser,
            candidate_corr_df=candidate_corr_df,
            max_positions_int=10,
            corr_penalty_lambda_float=1.0,
        )
        self.assertEqual(selected_symbol_list, ["AAA", "BBB"])

    def test_ties_break_by_symbol_ascending(self):
        candidate_score_ser = pd.Series({"BBB": 1.0, "AAA": 1.0, "CCC": 1.0})
        candidate_corr_df = make_corr_df(["AAA", "BBB", "CCC"], {})
        selected_symbol_list = select_corr_penalized_symbol_list(
            candidate_score_ser=candidate_score_ser,
            candidate_corr_df=candidate_corr_df,
            max_positions_int=2,
            corr_penalty_lambda_float=1.0,
        )
        self.assertEqual(selected_symbol_list, ["AAA", "BBB"])

    def test_invalid_inputs_raise(self):
        candidate_score_ser = pd.Series({"AAA": 1.0})
        candidate_corr_df = make_corr_df(["AAA"], {})
        with self.assertRaises(ValueError):
            select_corr_penalized_symbol_list(
                candidate_score_ser=candidate_score_ser,
                candidate_corr_df=candidate_corr_df,
                max_positions_int=0,
                corr_penalty_lambda_float=0.5,
            )
        with self.assertRaises(ValueError):
            select_corr_penalized_symbol_list(
                candidate_score_ser=candidate_score_ser,
                candidate_corr_df=candidate_corr_df,
                max_positions_int=1,
                corr_penalty_lambda_float=-0.1,
            )
        with self.assertRaises(ValueError):
            select_corr_penalized_symbol_list(
                candidate_score_ser=pd.Series({"AAA": np.nan}),
                candidate_corr_df=candidate_corr_df,
                max_positions_int=1,
                corr_penalty_lambda_float=0.5,
            )
        with self.assertRaises(ValueError):
            select_corr_penalized_symbol_list(
                candidate_score_ser=pd.Series({"AAA": 1.0, "ZZZ": 1.0}),
                candidate_corr_df=candidate_corr_df,
                max_positions_int=1,
                corr_penalty_lambda_float=0.5,
            )


class CorrPenaltyConfigTests(unittest.TestCase):
    def test_default_config_values(self):
        config_obj = CorrPenaltyAtrNormalizedNdxConfig()
        self.assertEqual(config_obj.corr_window_int, 126)
        self.assertEqual(config_obj.corr_min_overlap_int, 63)
        self.assertEqual(config_obj.corr_penalty_lambda_float, 0.5)
        self.assertEqual(config_obj.max_positions_int, 10)

    def test_invalid_config_raises(self):
        with self.assertRaises(ValueError):
            CorrPenaltyAtrNormalizedNdxConfig(corr_window_int=1)
        with self.assertRaises(ValueError):
            CorrPenaltyAtrNormalizedNdxConfig(corr_min_overlap_int=200, corr_window_int=126)
        with self.assertRaises(ValueError):
            CorrPenaltyAtrNormalizedNdxConfig(corr_penalty_lambda_float=-0.5)


class CorrPenaltyStrategySelectionTests(unittest.TestCase):
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

    def make_strategy_obj(
        self,
        corr_penalty_lambda_float: float,
        max_positions_int: int = 2,
    ) -> CorrPenaltyAtrNormalizedNdxStrategy:
        strategy_obj = CorrPenaltyAtrNormalizedNdxStrategy(
            name="CorrPenaltyTest",
            benchmarks=["$SPX"],
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            regime_symbol_str="SPY",
            max_positions_int=max_positions_int,
            corr_window_int=10,
            corr_min_overlap_int=5,
            corr_penalty_lambda_float=corr_penalty_lambda_float,
        )
        strategy_obj.previous_bar = pd.Timestamp("2024-03-28")
        strategy_obj.universe_df = pd.DataFrame(
            {"AAA": [1], "BBB": [1], "CCC": [1]},
            index=[strategy_obj.previous_bar],
        )
        return strategy_obj

    def make_price_return_df(self) -> pd.DataFrame:
        date_index = pd.bdate_range("2024-03-01", periods=15)
        rng = np.random.default_rng(7)
        base_return_vec = rng.normal(0.0, 0.01, len(date_index))
        # AAA and BBB nearly duplicate each other; CCC is independent.
        price_return_df = pd.DataFrame(
            {
                "AAA": base_return_vec,
                "BBB": base_return_vec + rng.normal(0.0, 0.001, len(date_index)),
                "CCC": rng.normal(0.0, 0.01, len(date_index)),
            },
            index=date_index,
        )
        return price_return_df

    def make_close_row(self) -> pd.Series:
        return self.make_close_row_ser(
            {
                ("AAA", "risk_adj_score_ser"): 2.0,
                ("AAA", "stock_trend_pass_bool"): True,
                ("BBB", "risk_adj_score_ser"): 1.9,
                ("BBB", "stock_trend_pass_bool"): True,
                ("CCC", "risk_adj_score_ser"): 1.2,
                ("CCC", "stock_trend_pass_bool"): True,
                ("SPY", "regime_pass_bool"): True,
            }
        )

    def test_lambda_zero_matches_base_strategy_selection(self):
        base_strategy_obj = AtrNormalizedNdxStrategy(
            name="BaseTest",
            benchmarks=["$SPX"],
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            regime_symbol_str="SPY",
            max_positions_int=2,
        )
        base_strategy_obj.previous_bar = pd.Timestamp("2024-03-28")
        base_strategy_obj.universe_df = pd.DataFrame(
            {"AAA": [1], "BBB": [1], "CCC": [1]},
            index=[base_strategy_obj.previous_bar],
        )
        base_target_weight_ser = base_strategy_obj.get_target_weight_ser(self.make_close_row())

        corr_strategy_obj = self.make_strategy_obj(corr_penalty_lambda_float=0.0)
        corr_strategy_obj.price_return_df = self.make_price_return_df()
        corr_target_weight_ser = corr_strategy_obj.get_target_weight_ser(self.make_close_row())

        self.assertEqual(
            sorted(base_target_weight_ser.index.tolist()),
            sorted(corr_target_weight_ser.index.tolist()),
        )
        pd.testing.assert_series_equal(
            base_target_weight_ser.sort_index(),
            corr_target_weight_ser.sort_index(),
            check_names=False,
        )

    def test_penalty_prefers_uncorrelated_candidate(self):
        strategy_obj = self.make_strategy_obj(corr_penalty_lambda_float=1.0)
        strategy_obj.price_return_df = self.make_price_return_df()
        target_weight_ser = strategy_obj.get_target_weight_ser(self.make_close_row())
        # AAA picked first; BBB (near-duplicate) must lose slot 2 to CCC.
        self.assertEqual(sorted(target_weight_ser.index.tolist()), ["AAA", "CCC"])
        self.assertTrue(np.allclose(target_weight_ser.to_numpy(), 0.5))

    def test_regime_fail_returns_empty_weights(self):
        strategy_obj = self.make_strategy_obj(corr_penalty_lambda_float=1.0)
        strategy_obj.price_return_df = self.make_price_return_df()
        close_row_ser = self.make_close_row_ser(
            {
                ("AAA", "risk_adj_score_ser"): 2.0,
                ("AAA", "stock_trend_pass_bool"): True,
                ("SPY", "regime_pass_bool"): False,
            }
        )
        target_weight_ser = strategy_obj.get_target_weight_ser(close_row_ser)
        self.assertEqual(len(target_weight_ser), 0)

    def test_correlation_window_is_causal(self):
        # Poison every return row strictly after the decision date; selection
        # must be unchanged because those rows may not enter the window.
        strategy_obj = self.make_strategy_obj(corr_penalty_lambda_float=1.0)
        clean_price_return_df = self.make_price_return_df()
        strategy_obj.price_return_df = clean_price_return_df
        clean_target_weight_ser = strategy_obj.get_target_weight_ser(self.make_close_row())

        poisoned_price_return_df = clean_price_return_df.copy()
        future_date_index = pd.bdate_range("2024-03-29", periods=10)
        poison_df = pd.DataFrame(
            np.full((len(future_date_index), 3), 0.5),
            index=future_date_index,
            columns=["AAA", "BBB", "CCC"],
        )
        strategy_obj_poisoned = self.make_strategy_obj(corr_penalty_lambda_float=1.0)
        strategy_obj_poisoned.price_return_df = pd.concat(
            [poisoned_price_return_df, poison_df]
        )
        poisoned_target_weight_ser = strategy_obj_poisoned.get_target_weight_ser(self.make_close_row())

        pd.testing.assert_series_equal(
            clean_target_weight_ser.sort_index(),
            poisoned_target_weight_ser.sort_index(),
            check_names=False,
        )

    def test_adv_gate_excludes_illiquid_candidate(self):
        strategy_obj = CorrPenaltyAtrNormalizedNdxStrategy(
            name="AdvGateTest",
            benchmarks=["$SPX"],
            rebalance_schedule_df=self.make_rebalance_schedule_df(),
            regime_symbol_str="SPY",
            max_positions_int=2,
            corr_window_int=10,
            corr_min_overlap_int=5,
            corr_penalty_lambda_float=0.0,
            min_dollar_adv_float=5_000_000.0,
        )
        strategy_obj.previous_bar = pd.Timestamp("2024-03-28")
        strategy_obj.universe_df = pd.DataFrame(
            {"AAA": [1], "BBB": [1], "CCC": [1]},
            index=[strategy_obj.previous_bar],
        )
        strategy_obj.price_return_df = self.make_price_return_df()
        adv_index = pd.bdate_range("2024-03-01", periods=15)
        # AAA is the top scorer but fails the gate; BBB has NaN ADV (short
        # history) and must also fail; CCC passes.
        strategy_obj.dollar_adv_df = pd.DataFrame(
            {"AAA": 1_000_000.0, "BBB": np.nan, "CCC": 50_000_000.0},
            index=adv_index,
        )
        target_weight_ser = strategy_obj.get_target_weight_ser(self.make_close_row())
        self.assertEqual(target_weight_ser.index.tolist(), ["CCC"])
        audit_df = strategy_obj.get_selection_audit_df()
        self.assertEqual(audit_df.iloc[0]["adv_excluded_count_int"], 2)

    def test_adv_gate_disabled_by_default(self):
        strategy_obj = self.make_strategy_obj(corr_penalty_lambda_float=0.0)
        strategy_obj.price_return_df = self.make_price_return_df()
        # dollar_adv_df deliberately left None: with min_dollar_adv 0 the gate
        # must not be consulted at all.
        target_weight_ser = strategy_obj.get_target_weight_ser(self.make_close_row())
        self.assertEqual(sorted(target_weight_ser.index.tolist()), ["AAA", "BBB"])

    def test_selection_audit_records_avg_pairwise_corr(self):
        strategy_obj = self.make_strategy_obj(corr_penalty_lambda_float=1.0)
        strategy_obj.price_return_df = self.make_price_return_df()
        strategy_obj.get_target_weight_ser(self.make_close_row())
        selection_audit_df = strategy_obj.get_selection_audit_df()
        self.assertEqual(len(selection_audit_df), 1)
        audit_row_ser = selection_audit_df.iloc[0]
        self.assertEqual(audit_row_ser["candidate_count_int"], 3)
        self.assertEqual(sorted(audit_row_ser["selected_symbol_list"]), ["AAA", "CCC"])
        self.assertTrue(np.isfinite(audit_row_ser["avg_selected_pairwise_corr_float"]))
        # AAA and CCC are independent series: realized corr must be far below
        # the AAA/BBB near-duplicate level.
        self.assertLess(abs(audit_row_ser["avg_selected_pairwise_corr_float"]), 0.9)


if __name__ == "__main__":
    unittest.main()
