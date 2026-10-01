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


# ---------------------------------------------------------------- NDX momentum siblings (alpha/scout/specs/ndx_vxn.py)
def _synthetic_ndx_inputs():
    """A ($10, 5% daily range) and B ($100, 2% range) rise alike: dollar ATR ranks A first, NATR ranks B first."""
    from alpha.scout.specs.ndx_vxn import NdxInputs

    date_index = pd.bdate_range("2018-01-01", "2020-12-31")
    growth_vec = 1.001 ** np.arange(len(date_index))
    close_df = pd.DataFrame({"A": 10.0 * growth_vec, "B": 100.0 * growth_vec, "SPY": 50.0 * growth_vec}, index=date_index)
    range_df = close_df * pd.Series({"A": 0.05, "B": 0.02, "SPY": 0.01})
    return NdxInputs(
        open_df=close_df, high_df=close_df + range_df / 2, low_df=close_df - range_df / 2, close_df=close_df,
        raw_close_df=close_df, dividend_df=close_df * 0.0,
        member_df=pd.DataFrame(1, index=date_index, columns=["A", "B"]),
        vxn_close_ser=pd.Series(44.0, index=date_index),  # scale = clip(22 / 44, 0.25, 1) = 0.5
    )


def test_ndx_variant_switches_move_only_ranking_units_and_the_vxn_scale():
    from alpha.scout.gate.run import GATED_SPEC_DICT
    from alpha.scout.specs.ndx_vxn import (
        NDX_VARIANT_DICT,
        NdxConfig,
        rebalance_weight_df,
    )

    assert NDX_VARIANT_DICT["ndx_vxn"].config == NdxConfig()  # the LIVE pod stays the default
    assert set(NDX_VARIANT_DICT) <= set(GATED_SPEC_DICT)
    with pytest.raises(ValueError, match="atr_unit_str"):
        NdxConfig(atr_unit_str="natr")
    inputs = _synthetic_ndx_inputs()
    small_dict = {"roc_month_int": 1, "stock_sma_int": 5, "atr_window_int": 3, "regime_sma_int": 5, "top_count_int": 1}
    expected_dict = {("dollar", True): ("A", 0.5), ("dollar", False): ("A", 1.0), ("percent", True): ("B", 0.5), ("percent", False): ("B", 1.0)}
    for (unit_str, vxn_bool), (symbol_str, weight_float) in expected_dict.items():
        weight_df = rebalance_weight_df(inputs, NdxConfig(atr_unit_str=unit_str, vxn_scaled_bool=vxn_bool, **small_dict))
        assert len(weight_df) > 20
        assert (weight_df[symbol_str] == weight_float).all() and (weight_df.drop(columns=symbol_str) == 0.0).all().all()



# ---------------------------------------------------------------------------------------------- TAA family spec
def test_taa_linearity_score_matches_a_per_window_regression():
    """The vectorised linearity score equals R2adj x OLS slope window by window, with NaN and flat windows."""
    from alpha.scout.specs.taa_3x import _linearity_lookback_df

    log_close_ser = pd.Series(np.cumsum(np.random.default_rng(3).normal(0.0003, 0.01, 120)), index=DATE_INDEX[:120])
    log_close_ser.iloc[40] = np.nan  # windows covering row 40 are NaN
    log_close_ser.iloc[70:95] = 1.5  # a flat stretch: windows inside it score 0
    lookback_int = 21
    score_ser = _linearity_lookback_df(log_close_ser.to_frame("A"), lookback_int)["A"]
    for end_int in range(len(log_close_ser)):
        window_vec = log_close_ser.iloc[max(0, end_int - lookback_int + 1): end_int + 1].to_numpy()
        if len(window_vec) < lookback_int or np.isnan(window_vec).any():
            assert np.isnan(score_ser.iloc[end_int])
        elif np.ptp(window_vec) == 0.0:
            assert score_ser.iloc[end_int] == 0.0
        else:
            slope_float, _intercept_float = np.polyfit(np.arange(lookback_int), window_vec, 1)
            r2_float = np.corrcoef(np.arange(lookback_int), window_vec)[0, 1] ** 2
            expected_float = (1.0 - (1.0 - r2_float) * (lookback_int - 1) / (lookback_int - 2)) * slope_float
            assert score_ser.iloc[end_int] == pytest.approx(expected_float, rel=1e-9, abs=1e-15)


def test_taa_variants_are_gated_and_the_live_default_is_unchanged():
    from alpha.scout.family import TAA_FAMILY_NAME_DICT
    from alpha.scout.gate.run import GATED_SPEC_DICT
    from alpha.scout.specs.taa_3x import VARIANT_DICT, TaaConfig

    assert VARIANT_DICT["taa_3x"].config == TaaConfig()
    assert TaaConfig().traded_tuple == ("GLD", "UUP", "TLT", "DBC", "BTAL", "TQQQ") and TaaConfig().slot_weight_str == "rank"
    for name_str, variant in VARIANT_DICT.items():
        assert GATED_SPEC_DICT[name_str].strategy_import_str == variant.strategy_import_str
        assert name_str in TAA_FAMILY_NAME_DICT


def _norgate_ready() -> bool:
    try:
        import norgatedata

        return bool(norgatedata.status())
    except Exception:  # noqa: BLE001
        return False


def _norgate_running_bool() -> bool:
    return _norgate_ready()


EXACT_MEMBERSHIP_DEFAULT_TS = pd.Timestamp("2026-09-29 10:31")  # 9afc293: older engine runs used the member tail trim


@pytest.mark.skipif(not _norgate_running_bool(), reason="Norgate data not available")
@pytest.mark.parametrize("spec_name_str", ["ndx_atr", "ndx_natr20", "ndx_natr20_vxn"])
def test_ndx_sibling_gate_passes_against_the_saved_engine_run(spec_name_str):
    from pathlib import Path

    from alpha.scout.gate.run import GATED_SPEC_DICT, run_gate
    from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH

    path_list = sorted(Path(MAIN_CHECKOUT_ROOT_PATH).glob(GATED_SPEC_DICT[spec_name_str].pickle_glob_str))
    if not path_list or pd.Timestamp.fromtimestamp(path_list[-1].stat().st_mtime) < EXACT_MEMBERSHIP_DEFAULT_TS:
        pytest.skip("No saved engine run with exact membership; run `python -m alpha.scout gate <name> --fresh`.")
    report = run_gate(spec_name_str)
    assert report.passed_bool, report.summary_str()


@pytest.mark.skipif("not _norgate_ready()")
@pytest.mark.parametrize("spec_name_str", ["taa_3x_1n", "taa_lin_1n_qqq", "taa_2x_1n_qld", "taa_nobtal_2x_1n_qld", "taa_nobtal_2x_1n_sso"])
def test_taa_variant_gate_passes_against_the_saved_engine_run(spec_name_str):
    from pathlib import Path

    from alpha.scout.gate.run import GATED_SPEC_DICT, run_gate
    from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH

    if not any(Path(MAIN_CHECKOUT_ROOT_PATH).glob(GATED_SPEC_DICT[spec_name_str].pickle_glob_str)):
        pytest.skip("No saved engine run")
    report = run_gate(spec_name_str)
    assert report.passed_bool, report.summary_str()
