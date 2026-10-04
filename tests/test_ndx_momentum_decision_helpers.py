"""Helpers of the momentum decision pack (scripts/research/scout_robustness_20261002/ndx_momentum_decision.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

STUDY_DIR_PATH = Path(__file__).resolve().parents[1] / "scripts" / "research" / "scout_robustness_20261002"
sys.path.insert(0, str(STUDY_DIR_PATH))

import ndx_momentum_decision as decision  # noqa: E402


def _eligible_frame(month_int: int = 14, name_int: int = 30, seed_int: int = 0):
    rng_obj = np.random.default_rng(seed_int)
    execution_index = pd.bdate_range("2023-01-02", periods=month_int * 21)[::21]
    eligible_df = pd.DataFrame(rng_obj.random((month_int, name_int)) > 0.3, index=execution_index,
                               columns=[f"S{i:02d}" for i in range(name_int)])
    eligible_df.iloc[3] = False  # a regime-off month: nothing is eligible
    slot_weight_ser = pd.Series(0.1 * rng_obj.uniform(0.25, 1.0, month_int), index=execution_index)
    slot_weight_ser.iloc[3] = 0.0
    return eligible_df, slot_weight_ser


def test_random_book_holds_ten_eligible_names_at_the_slot_weight_and_is_reproducible():
    eligible_df, slot_weight_ser = _eligible_frame()
    weight_df = decision.random_weight_df(eligible_df, slot_weight_ser, 0.7, draw_int=5)
    held_df = weight_df > 0
    assert list(weight_df.index) == list(eligible_df.index)
    assert (held_df.sum(axis=1).drop(eligible_df.index[3]) == 10).all()
    assert not held_df.loc[eligible_df.index[3]].any()  # regime off -> all cash
    assert not (held_df & ~eligible_df.reindex(columns=weight_df.columns)).to_numpy().any()  # never an ineligible name
    for execution_ts in eligible_df.index.drop(eligible_df.index[3]):
        assert weight_df.loc[execution_ts][held_df.loc[execution_ts]].to_numpy() == pytest.approx(slot_weight_ser.loc[execution_ts])
    assert weight_df.equals(decision.random_weight_df(eligible_df, slot_weight_ser, 0.7, draw_int=5))
    assert not weight_df.equals(decision.random_weight_df(eligible_df, slot_weight_ser, 0.7, draw_int=6))


def test_random_book_with_fewer_eligible_names_than_slots_holds_them_all():
    eligible_df, slot_weight_ser = _eligible_frame(name_int=30)
    eligible_df.iloc[5] = False
    eligible_df.iloc[5, :4] = True
    weight_df = decision.random_weight_df(eligible_df, slot_weight_ser, 0.7, draw_int=1)
    assert set(weight_df.columns[weight_df.iloc[5] > 0]) == set(eligible_df.columns[:4])


def test_retention_is_one_for_a_book_that_never_drops_an_eligible_name_and_tracks_the_parameter():
    eligible_df, slot_weight_ser = _eligible_frame(month_int=120, name_int=60, seed_int=3)
    sticky_df = decision.random_weight_df(eligible_df, slot_weight_ser, 1.0, draw_int=0)
    churn_df = decision.random_weight_df(eligible_df, slot_weight_ser, 0.0, draw_int=0)
    middle_df = decision.random_weight_df(eligible_df, slot_weight_ser, 0.7, draw_int=0)
    assert decision.retention_float(sticky_df, eligible_df) == pytest.approx(1.0)
    # With retention 0 a held name returns only by chance (10 slots over about 42 eligible names).
    assert decision.retention_float(churn_df, eligible_df) < 0.35
    assert 0.65 < decision.retention_float(middle_df, eligible_df) < 0.85


def test_book_series_resets_to_the_fixed_weights_each_month():
    rng_obj = np.random.default_rng(1)
    date_index = pd.bdate_range("2020-01-01", periods=130)
    sleeve_df = pd.DataFrame({"taa": rng_obj.normal(0.0005, 0.01, 130), "ndx": rng_obj.normal(0.0004, 0.012, 130)}, index=date_index)
    book_ser = decision.book_ser(sleeve_df, {"taa": 0.6, "ndx": 0.4})
    assert len(book_ser) == len(sleeve_df)
    for _, month_df in sleeve_df.groupby(pd.Grouper(freq="MS")):
        expected_float = 0.6 * (1 + month_df["taa"]).prod() + 0.4 * (1 + month_df["ndx"]).prod()
        assert (1 + book_ser.loc[month_df.index]).prod() == pytest.approx(expected_float)


def test_whole_share_check_counts_positions_below_one_share_and_the_exposure_left_in_cash():
    date_index = pd.bdate_range("2023-01-02", periods=100)
    raw_close_df = pd.DataFrame(100.0, index=date_index, columns=["BIG", "A", "B"])
    raw_close_df["BIG"] = 5000.0
    weight_df = pd.DataFrame(0.1, index=date_index[[10, 40, 70]], columns=["BIG", "A", "B"])
    small = decision.whole_share_dict(weight_df, raw_close_df, 12_000.0)  # 0.1 x 12,000 = 1,200 < one BIG share
    large = decision.whole_share_dict(weight_df, raw_close_df, 1_000_000.0)
    assert small["sub_share_position_share"] == pytest.approx(1 / 3)
    assert small["exposure_lost_share"] == pytest.approx(1 / 3)  # A and B buy exactly 12 shares; BIG buys none
    assert large == {"sub_share_position_share": 0.0, "exposure_lost_share": 0.0, "names_mean": 3.0}


def test_verdict_uses_the_a15_thresholds():
    assert decision.verdict_str(0.90, 0.10) == "EARNS ITS PLACE"
    assert decision.verdict_str(0.90, 0.02) == "SMALL"
    assert decision.verdict_str(0.60, 0.10) == "UNCLEAR"
    assert decision.verdict_str(0.40, -0.10) == "NO EVIDENCE"


def test_newey_west_recovers_alpha_and_beta():
    rng_obj = np.random.default_rng(2)
    factor_df = pd.DataFrame({"QQQ": rng_obj.normal(0.0, 0.02, 2000)})
    y_ser = 0.001 + 0.5 * factor_df["QQQ"] + rng_obj.normal(0.0, 0.005, 2000)
    fit = decision.newey_west_ols(y_ser, factor_df)
    assert fit["b_QQQ"] == pytest.approx(0.5, abs=0.02)
    assert fit["alpha_ann"] == pytest.approx(0.052, abs=0.015) and fit["alpha_t"] > 5 and fit["weeks"] == 2000
