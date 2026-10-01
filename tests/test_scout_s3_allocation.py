"""Scout S3 for allocation (W) and monthly ranking (X) families (synthetic)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha.scout.stations.s3_allocation import (
    gate_split,
    predictive_tests,
    ranking_tests,
)

MONTH_INDEX = pd.date_range("2000-01-31", periods=240, freq="ME")


def _panel(beta_float: float, seed_int: int, asset_count_int: int = 6):
    rng_obj = np.random.default_rng(seed_int)
    score_df = pd.DataFrame(rng_obj.normal(0, 1, (len(MONTH_INDEX), asset_count_int)), index=MONTH_INDEX)
    next_return_df = beta_float * 0.01 * score_df + pd.DataFrame(rng_obj.normal(0, 0.04, score_df.shape), index=MONTH_INDEX)
    return score_df, next_return_df


def test_predictive_tests_find_a_planted_slope_and_not_noise():
    planted = predictive_tests(*_panel(1.0, 0))
    assert planted["fama_macbeth_slope"]["t_float"] > 4 and planted["on_off_spread"]["t_float"] > 3
    assert planted["fama_macbeth_slope"]["mean_float"] == pytest.approx(0.01, abs=0.003)
    noise_t_list = [predictive_tests(*_panel(0.0, seed_int))["fama_macbeth_slope"]["t_float"] for seed_int in range(1, 6)]
    assert all(abs(t_float) < 3 for t_float in noise_t_list)


def test_gate_split_sees_a_risk_gate():
    rng_obj = np.random.default_rng(1)
    gate_ser = pd.Series(rng_obj.random(len(MONTH_INDEX)) < 0.8, index=MONTH_INDEX)
    return_ser = pd.Series(np.where(gate_ser, rng_obj.normal(0.01, 0.04, len(MONTH_INDEX)), rng_obj.normal(0.0, 0.10, len(MONTH_INDEX))), index=MONTH_INDEX)
    result = gate_split(return_ser, gate_ser)
    assert result["check"][1] == "PASS" and result["vol_ratio_off_over_on_float"] > 2


def test_ranking_tests_detect_information_among_eligible_members_only():
    score_df, next_return_df = _panel(0.5, 2, asset_count_int=40)
    eligible_df = pd.DataFrame(True, index=MONTH_INDEX, columns=score_df.columns)
    assert ranking_tests(score_df, next_return_df, eligible_df, top_int=5)["rank_ic"]["t_float"] > 4
    eligible_df.iloc[:, :] = False
    assert ranking_tests(score_df, next_return_df, eligible_df, top_int=5)["rank_ic"]["months_int"] == 0

