"""Identity gate comparator (alpha/scout/gate/identity.py) on synthetic series."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha.scout.gate.identity import compare, compare_exact

DATE_INDEX = pd.bdate_range("2010-01-04", periods=2520)


def _returns(seed_int: int = 0) -> pd.Series:
    return pd.Series(np.random.default_rng(seed_int).normal(0.0004, 0.01, len(DATE_INDEX)), index=DATE_INDEX)


def _weights() -> pd.DataFrame:
    month_end_index = DATE_INDEX.to_series().groupby(DATE_INDEX.to_period("M")).last().values
    rng_obj = np.random.default_rng(1)
    weight_mat = rng_obj.dirichlet(np.ones(5), len(month_end_index))
    return pd.DataFrame(weight_mat, index=pd.DatetimeIndex(month_end_index), columns=list("ABCDE"))


def test_identical_outputs_pass():
    report = compare(_returns(), _returns(), _weights(), _weights())
    assert report.passed_bool, report.summary_str()
    assert report.mismatch_df.empty


def test_tiny_numerical_noise_still_passes():
    noisy_ser = _returns() + np.random.default_rng(5).normal(0, 1e-6, len(DATE_INDEX))
    assert compare(_returns(), noisy_ser, _weights(), _weights() + 1e-4).passed_bool


def test_a_deliberately_broken_strategy_fails():
    engine_ser, engine_weight_df = _returns(), _weights()
    # Broken spec: one rebalance in ten holds the wrong asset, and returns carry an extra 1 bp a day.
    broken_weight_df = engine_weight_df.copy()
    broken_weight_df.iloc[::10] = broken_weight_df.iloc[::10, ::-1].to_numpy()
    broken_ser = engine_ser + 0.0001
    report = compare(engine_ser, broken_ser, engine_weight_df, broken_weight_df)
    assert not report.passed_bool
    assert not report.check_dict["annualised return difference"]["pass_bool"]
    assert not report.check_dict["decision cells within 0.5 pp"]["pass_bool"]
    assert len(report.mismatch_df) > 0 and "FAIL" in report.summary_str()


def test_each_return_check_can_fail_alone():
    engine_ser = _returns()
    uncorrelated_ser = _returns(9)
    assert not compare(engine_ser, uncorrelated_ser).check_dict["daily return correlation"]["pass_bool"]
    crash_ser = engine_ser.copy()
    crash_ser.iloc[1000] -= 0.05  # one extra 5% loss: drawdown differs, correlation barely moves
    report = compare(engine_ser, crash_ser)
    assert not report.check_dict["max drawdown difference"]["pass_bool"]


def test_misaligned_or_short_inputs():
    with pytest.raises(ValueError, match="one year"):
        compare(_returns().iloc[:100], _returns().iloc[:100])
    report = compare(_returns(), _returns().iloc[10:])
    assert any("outside the common range" in note_str for note_str in report.note_list)
    assert any("only the return checks" in note_str for note_str in report.note_list)


def _daily_weights() -> pd.DataFrame:
    return pd.DataFrame(np.random.default_rng(2).dirichlet(np.ones(3), len(DATE_INDEX)), index=DATE_INDEX, columns=list("ABC"))


TRADE_DATES = DATE_INDEX[::21]


def test_exact_tier_passes_identical_runs_and_fails_tiny_differences():
    engine_ser, weight_df = _returns(), _daily_weights()
    assert compare_exact(engine_ser, engine_ser.copy(), weight_df, weight_df.copy(), TRADE_DATES, TRADE_DATES).passed_bool
    # 1e-6 on one day: invisible to the tolerance tier, a failure for the exact tier.
    nudged_ser = engine_ser.copy()
    nudged_ser.iloc[500] += 1e-6
    assert compare(engine_ser, nudged_ser).passed_bool
    assert not compare_exact(engine_ser, nudged_ser, weight_df, weight_df, TRADE_DATES, TRADE_DATES).passed_bool
    nudged_weight_df = weight_df.copy()
    nudged_weight_df.iloc[10, 0] += 1e-6
    assert not compare_exact(engine_ser, engine_ser, weight_df, nudged_weight_df, TRADE_DATES, TRADE_DATES).passed_bool


def test_exact_tier_requires_full_coverage_and_the_same_trade_dates():
    engine_ser, weight_df = _returns(), _daily_weights()
    late_report = compare_exact(engine_ser, engine_ser.iloc[504:], weight_df, weight_df, TRADE_DATES, TRADE_DATES)
    assert not late_report.check_dict["same first date"]["pass_bool"]
    early_report = compare_exact(engine_ser, engine_ser.iloc[:-756], weight_df, weight_df, TRADE_DATES, TRADE_DATES)
    assert not early_report.check_dict["coverage"]["pass_bool"]
    holed_report = compare_exact(engine_ser, engine_ser.drop(DATE_INDEX[900]), weight_df, weight_df, TRADE_DATES, TRADE_DATES)
    assert not holed_report.check_dict["coverage"]["pass_bool"]
    trailing_ok = compare_exact(engine_ser, engine_ser.iloc[:-3], weight_df, weight_df, TRADE_DATES, TRADE_DATES)
    assert trailing_ok.check_dict["coverage"]["pass_bool"]
    skipped_report = compare_exact(engine_ser, engine_ser, weight_df, weight_df, TRADE_DATES, TRADE_DATES[1:])
    assert not skipped_report.check_dict["same trade dates"]["pass_bool"]
