"""Identity gate comparator (alpha/scout/gate/identity.py) on synthetic series."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha.scout.gate.identity import compare, compare_exact

DATE_INDEX = pd.bdate_range("2010-01-04", periods=2520)
import dataclasses


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


# ---------------------------------------------------------------------------------------------- Inflation Compass spec
def test_compass_t5yie_alignment_and_slope_match_the_engine_helpers():
    """Published = dated strictly before T, observation-dated = on or before T, both NaN past 7 days; OLS slope as polyfit."""
    from alpha.scout.specs.compass import asof_value_ser, rolling_slope_vec
    from strategies.taa_df.strategy_taa_inflation_compass import (
        align_fred_to_session_ser,
        compute_rolling_ols_slope_ser,
    )

    session_index = DATE_INDEX[:200]
    observation_index = session_index.delete(list(range(50, 62)) + [100, 101, 140])  # a 12-session gap (> 7 days) and holidays
    value_ser = pd.Series(np.round(2.0 + np.random.default_rng(4).normal(0, 0.1, len(observation_index)), 2), index=observation_index)
    for same_date_bool in (False, True):
        expected_ser, _age_ser = align_fred_to_session_ser(value_ser, session_index, include_same_date_bool=same_date_bool)
        spec_ser = asof_value_ser(value_ser, session_index, include_same_date_bool=same_date_bool)
        pd.testing.assert_series_equal(spec_ser, expected_ser, check_names=False, check_freq=False, check_index_type=False)
    assert np.isnan(asof_value_ser(value_ser, session_index, False).iloc[61])  # the newest observation is 12 sessions old
    ratio_ser = pd.Series(np.cumprod(1.0 + np.random.default_rng(5).normal(0, 0.01, 150)))
    ratio_ser.iloc[30] = np.nan
    np.testing.assert_allclose(rolling_slope_vec(ratio_ser.to_numpy(), 20), compute_rolling_ols_slope_ser(ratio_ser, 20).to_numpy(), rtol=1e-9, atol=1e-15)


def test_compass_variants_family_and_regime_map():
    from alpha.scout.family import COMPASS_GRID_DICT, compass_family
    from alpha.scout.gate.run import GATED_SPEC_DICT
    from alpha.scout.specs.compass import (
        VARIANT_DICT,
        CompassConfig,
        config_from_dict,
        regime_weight_mat,
    )

    assert VARIANT_DICT["compass"].config == CompassConfig()
    assert CompassConfig().traded_tuple == ("XLE", "XLK", "XLU", "XLP", "IEF")
    assert VARIANT_DICT["compass_qqq"].config.traded_tuple == ("XLE", "QQQ", "XLU", "XLP", "IEF")
    for name_str, variant in VARIANT_DICT.items():
        assert GATED_SPEC_DICT[name_str].strategy_import_str == variant.strategy_import_str
    with pytest.raises(ValueError, match="goldilocks"):
        CompassConfig(goldilocks_str="XLE")
    tied = config_from_dict(CompassConfig(), {"trend_lookback_int": 40, "growth_sma_int": 150})
    assert (tied.breakeven_lookback_int, tied.asset_slope_lookback_int, tied.growth_sma_int) == (40, 40, 150)
    family = compass_family("compass", inputs=object())  # inputs are only read when a configuration runs
    assert family.family_id_str == "macro_regime_allocation" and len(family.config_list()) == 27
    assert family.live_config_dict in family.config_list() and set(family.param_grid_dict) == set(COMPASS_GRID_DICT)
    weight_mat = regime_weight_mat(np.array([True, True, False, False]), np.array([True, False, True, False]))
    np.testing.assert_array_equal(weight_mat, [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0.5, 0.5]])


def test_compass_fast_replica_holds_the_planted_regime_and_runs_on_a_shuffle():
    """SPY rising and T5YIE high and rising: growth up + inflation on -> XLE, held from the close after each decision."""
    from alpha.scout.specs.compass import fast_daily_list, mcpt_column_list

    rng_obj = np.random.default_rng(6)
    date_index = DATE_INDEX[:600]
    column_list = mcpt_column_list()
    matrix = np.column_stack([rng_obj.normal(0.0002, 0.01, len(date_index)) for _ in column_list])
    matrix[:, column_list.index("SPY")] = 0.001
    matrix[:, column_list.index("T5YIE_published")] = 3.0 + 0.001 * np.arange(len(date_index))
    matrix[:, column_list.index("T5YIE_dated")] = 3.0 + 0.001 * (np.arange(len(date_index)) + 1)
    small_dict = {"growth_sma_int": 50, "trend_lookback_int": 20, "inflation_threshold_float": 2.0}
    (daily_vec,) = fast_daily_list(matrix, date_index, [small_dict])
    first_decision_int = date_index.get_loc(date_index[date_index.to_period("M") == date_index[49].to_period("M")][-1])
    np.testing.assert_allclose(daily_vec[first_decision_int + 2:], matrix[first_decision_int + 2:, 0])
    assert (daily_vec[: first_decision_int + 2] == 0.0).all()
    shuffled_list = fast_daily_list(matrix[rng_obj.permutation(len(date_index))], date_index, [small_dict, {**small_dict, "growth_sma_int": 100}])
    assert all(np.isfinite(v).all() for v in shuffled_list)


@pytest.mark.skipif("not _norgate_ready()")
@pytest.mark.parametrize("spec_name_str", ["compass", "compass_qqq"])
def test_compass_gate_passes_against_the_saved_engine_run(spec_name_str):
    from pathlib import Path

    from alpha.scout.gate.run import GATED_SPEC_DICT, run_gate
    from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH

    if not any(Path(MAIN_CHECKOUT_ROOT_PATH).glob(GATED_SPEC_DICT[spec_name_str].pickle_glob_str)):
        pytest.skip("No saved engine run")
    report = run_gate(spec_name_str)
    assert report.passed_bool, report.summary_str()


@pytest.mark.skipif("not _norgate_ready()")
def test_compass_gate_fails_the_old_same_date_t5yie_leak(monkeypatch):
    """Near-miss mutant: reading the T5YIE value dated T at the T close (the leak fixed on 2026-09-28) must fail."""
    from pathlib import Path

    from alpha.scout.gate.run import GATED_SPEC_DICT, run_gate
    from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH
    from alpha.scout.specs import compass

    if not any(Path(MAIN_CHECKOUT_ROOT_PATH).glob(GATED_SPEC_DICT["compass"].pickle_glob_str)):
        pytest.skip("No saved engine run")
    honest_fn = compass.asof_value_ser
    monkeypatch.setattr(compass, "asof_value_ser", lambda value_ser, session_index, include_same_date_bool: honest_fn(value_ser, session_index, True))
    assert not run_gate("compass").passed_bool


@pytest.mark.skipif("not _norgate_ready()")
def test_compass_fast_replica_tracks_the_engine():
    from alpha.scout.family import compass_family
    from alpha.scout.specs import compass

    inputs = compass.load_inputs()
    family = compass_family("compass", inputs)
    date_index, matrix = compass.mcpt_matrix(inputs, end_date_str="2022-12-30")
    (fast_vec,) = compass.fast_daily_list(matrix, date_index, [family.live_config_dict])
    engine_ser = family.run_config(family.live_config_dict, compass.ENGINE_COST_MODEL).daily_return_ser.loc[:"2022-12-30"]
    fast_ser = pd.Series(fast_vec, index=date_index).reindex(engine_ser.index)
    start_ts = max(engine_ser.index[0], fast_ser.ne(0).idxmax())
    assert np.corrcoef(engine_ser.loc[start_ts:], fast_ser.loc[start_ts:])[0, 1] > 0.98


# ---------------------------------------------------------------------------------------------- CORE5 (alpha/scout/specs/core5.py)
def test_core5_spec_signal_targets_and_rebalance_dates():
    """Mid-rank percentile with ties; long/BIL/short targets; rebalances only at the start and XNYS month ends."""
    import exchange_calendars

    from alpha.scout.specs import core5

    severity_vec = np.array([0.0, 0.0, 0.1, 0.3, 0.1, 0.0, 0.2, 0.1])
    percentile_vec = core5._midrank_percentile_vec(severity_vec, 4)
    for end_int in range(3, len(severity_vec)):
        window_vec = severity_vec[end_int - 3 : end_int + 1]
        expected_float = ((window_vec < window_vec[-1]).sum() + ((window_vec == window_vec[-1]).sum() + 1) / 2) / 4
        assert percentile_vec[end_int] == expected_float
    assert np.isnan(percentile_vec[:3]).all()

    weight_vec = core5.target_weight_vec(np.array([1.0, 0.0, 1.0, 0.0, 1.0]), True, 0.5, core5.LIVE_CONFIG)
    np.testing.assert_allclose(weight_vec, [0.2, 0.0, 0.2, -0.05, 0.2, 0.4])  # DBC -min(10%, 2.5% / 50%)
    assert core5.target_weight_vec(np.ones(5), False, np.nan, core5.LIVE_CONFIG)[-1] == pytest.approx(0.0)

    # Four steady risers and a falling, choppy DBC: states never change after warm-up, so only month ends rebalance.
    session_index = exchange_calendars.get_calendar("XNYS", start="2015-01-02", end="2016-12-30").sessions.tz_localize(None)
    step_vec = np.arange(len(session_index))
    rising_vec = 100.0 * 1.001**step_vec
    dbc_vec = 100.0 * np.cumprod(1.0 - 0.003 + 0.005 * (-1.0) ** step_vec)
    close_df = pd.DataFrame({s: rising_vec for s in core5.RISK_ASSET_TUPLE}, index=session_index).assign(DBC=dbc_vec, BIL=50.0)
    inputs = core5.Core5Inputs(
        signal_close_df=close_df[list(core5.RISK_ASSET_TUPLE)], reserve_total_return_close_ser=close_df["BIL"],
        open_df=close_df, close_df=close_df, dividend_df=close_df * 0.0,
    )
    weight_df = core5.rebalance_weight_df(inputs)
    assert weight_df.index[0] == session_index[core5.LIVE_CONFIG.percentile_lookback_int]  # the first defined decision + 1
    # The AMA starts at the close, so the risers' states settle in the first weeks; from August 2015 nothing changes.
    settled_df = weight_df.loc["2015-08-01":]
    month_end_index = session_index[core5.month_end_flag_vec(session_index)][:-1]  # the last one has no next session
    expected_index = session_index[session_index.get_indexer(month_end_index) + 1]
    assert settled_df.index.equals(expected_index[expected_index >= "2015-08-01"])
    assert (settled_df[["SPY", "IEF", "GLD", "UUP"]] == 0.2).all().all() and np.allclose(settled_df["BIL"], 0.2)
    assert (settled_df["DBC"] < 0).all() and (settled_df["DBC"] >= -0.10).all()
    offset_df = core5.rebalance_weight_df(inputs, core5.Core5Config(decision_offset_int=3)).loc["2015-08-01":]
    offset_decision_index = session_index[core5.month_end_flag_vec(session_index, 3)]
    assert offset_decision_index[:-1].equals(session_index[session_index.get_indexer(month_end_index) - 3])
    offset_expected_index = session_index[session_index.get_indexer(offset_decision_index) + 1]
    assert offset_df.index.equals(offset_expected_index[offset_expected_index >= "2015-08-01"])  # 3 sessions earlier

@pytest.mark.skipif("not _norgate_ready()")
def test_core5_gate_passes_and_blocks_a_spec_without_the_borrow_fee(monkeypatch):
    from dataclasses import replace
    from pathlib import Path

    from alpha.scout.gate.run import GATED_SPEC_DICT, run_gate
    from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH
    from alpha.scout.specs import core5

    assert GATED_SPEC_DICT["core5"].strategy_import_str == core5.STRATEGY_IMPORT_STR
    if not any(Path(MAIN_CHECKOUT_ROOT_PATH).glob(GATED_SPEC_DICT["core5"].pickle_glob_str)):
        pytest.skip("No saved engine run; run `python -m alpha.scout gate core5 --fresh`.")
    report = run_gate("core5")
    assert report.passed_bool, report.summary_str()
    monkeypatch.setattr(core5, "LIVE_CONFIG", replace(core5.LIVE_CONFIG, annual_borrow_rate_float=0.0))
    assert not run_gate("core5").passed_bool  # the DBC borrow fee is about 6e-5 of NAV on a short day

@pytest.mark.skipif("not _norgate_ready()")
def test_core5_mcpt_replica_tracks_the_engine():
    from alpha.scout.family import core5_family
    from alpha.scout.specs import core5

    inputs = core5.load_inputs()
    family = core5_family(inputs)
    date_index, matrix = core5.mcpt_matrix(inputs)
    assert matrix.shape == (len(date_index), len(core5.TRADED_TUPLE)) and date_index[-1] <= pd.Timestamp("2022-12-30")
    fast_ser = pd.Series(core5.fast_daily_list(matrix, date_index, [family.live_config_dict])[0], index=date_index)
    engine_ser = family.run_config(family.live_config_dict).daily_return_ser.loc["2008-01-01":"2022-12-30"]
    assert np.corrcoef(engine_ser, fast_ser.reindex(engine_ser.index))[0, 1] > 0.97  # 0.983 on 2026-10-01


# ---------------------------------------------------------------------------------------------- TFI and Trinity specs
def _synthetic_tfi_inputs():
    from alpha.scout.specs.tfi import TfiInputs

    session_index = pd.bdate_range("2001-06-01", "2003-12-31")
    rng_obj = np.random.default_rng(4)
    close_df = pd.DataFrame(100.0 * np.cumprod(1.0 + rng_obj.normal(0.0002, 0.003, (len(session_index), 3)), axis=0),
                            index=session_index, columns=["IEF", "LQD", "BIL"])
    close_df.loc[:"2002-09-30", "BIL"] = np.nan  # the cash vehicle lists later
    yield_index = pd.bdate_range("1999-01-01", "2003-12-31")
    step_vec = np.arange(len(yield_index), dtype=float)
    yield_df = pd.DataFrame({"DGS10": 5.0 + np.sin(step_vec / 40.0), "DGS3MO": 2.0, "DAAA": 6.0 + np.cos(step_vec / 55.0), "DBAA": 7.0},
                            index=yield_index)
    return TfiInputs(open_df=close_df, close_df=close_df, dividend_df=close_df * 0.0, total_return_close_df=close_df, yield_df=yield_df)

def test_tfi_signal_reads_t_minus_one_includes_the_current_spread_and_maps_cash_to_bil():
    from alpha.scout.specs import tfi

    inputs = _synthetic_tfi_inputs()
    frame = tfi.signal_df(inputs)
    session_index = inputs.open_df.index
    for decision_ts, row in frame.iterrows():
        assert row["observation_date"] == session_index[session_index.get_loc(decision_ts) - 1]  # never the decision day
    # A planted extreme spread on one observation day enters its own decision's median (current spread included).
    planted_ts = frame["observation_date"].iloc[20]
    planted_dgs10_ser = inputs.yield_df["DGS10"].where(inputs.yield_df.index != planted_ts, 50.0)
    planted_inputs = dataclasses.replace(inputs, yield_df=inputs.yield_df.assign(DGS10=planted_dgs10_ser))
    planted_frame = tfi.signal_df(planted_inputs)
    prehistory_list = tfi._prehistory_list(planted_inputs.yield_df, tfi.TERM_SERIES_TUPLE, tfi.term_spread, frame.index[0])
    assert planted_frame["term_float"].iloc[20] == 48.0 and planted_frame["term_state_float"].iloc[20] == 1.0
    expected_float = np.median(prehistory_list + planted_frame["term_float"].iloc[:21].tolist())
    assert planted_frame["term_threshold_float"].iloc[20] == expected_float
    assert planted_frame["term_threshold_float"].iloc[21:].equals(
        pd.Series([np.median(prehistory_list + planted_frame["term_float"].iloc[: i + 1].tolist()) for i in range(21, len(frame))],
                  index=frame.index[21:], name="term_threshold_float")
    )
    weight_df = tfi.rebalance_weight_df(inputs)
    assert np.allclose(weight_df.loc["2002-11-01":].sum(axis=1), 1.0) and (weight_df.loc[:"2002-09-30", "BIL"] == 0.0).all()
    assert set(weight_df["IEF"].unique()) <= {0.0, 0.5}

def _synthetic_trinity_inputs():
    from alpha.scout.specs.trinity import TrinityInputs

    session_index = pd.bdate_range("2015-01-01", "2018-12-31")
    rng_obj = np.random.default_rng(6)
    return_mat = rng_obj.normal(0.0003, [0.012, 0.009, 0.010, 0.0002], (len(session_index), 4))
    return_mat[500:560, 0] *= 4.0  # a volatile spell: the overlay must scale down and trade inside the month
    close_df = pd.DataFrame(50.0 * np.cumprod(1.0 + return_mat, axis=0), index=session_index, columns=["VTI", "GLD", "TLT", "BIL"])
    return TrinityInputs(open_df=close_df, close_df=close_df, dividend_df=close_df * 0.0, total_return_close_df=close_df)

def test_trinity_decisions_replay_exactly_and_the_band_trades_between_months():
    from alpha.scout.engines.weights import simulate
    from alpha.scout.specs import trinity

    inputs = _synthetic_trinity_inputs()
    result = trinity.simulate_config(inputs)
    weight_df = result.decided_weight_df
    replay = simulate(inputs.open_df, inputs.close_df, inputs.dividend_df, weight_df, start_date=weight_df.index[0],
                      cost_model=trinity.ENGINE_COST_MODEL)
    assert np.array_equal(replay.total_value_ser.to_numpy(), result.total_value_ser.to_numpy())
    assert np.allclose(weight_df.sum(axis=1), 1.0) and (weight_df.to_numpy() >= 0.0).all() and (weight_df["BIL"] > 0.0).any()
    first_session_set = set(pd.Series(inputs.open_df.index).groupby(inputs.open_df.index.to_period("M")).min())
    assert any(date not in first_session_set for date in weight_df.index)  # the band trades inside a month
    assert len(weight_df) < 0.5 * len(result.total_value_ser)  # but not every day

def test_tfi_and_trinity_defaults_match_their_engines_and_are_gated():
    from alpha.scout.families import validate_family_id
    from alpha.scout.family import tfi_family, trinity_family
    from alpha.scout.gate.run import GATED_SPEC_DICT
    from alpha.scout.specs import tfi, trinity
    from strategies.taa_beyond_6040 import (
        strategy_taa_tactical_fixed_income_ief_lqd as tfi_engine,
    )
    from strategies.taa_beyond_6040 import (
        strategy_taa_trinity_vol_control_8_bil as trinity_engine,
    )

    assert tfi.FRED_SHA256_DICT == tfi_engine.FROZEN_FRED_SHA256_BY_SERIES_DICT
    assert tfi.ENGINE_COST_MODEL.slippage_float == tfi_engine.SLIPPAGE_PER_SIDE_FLOAT
    assert tfi.ENGINE_COST_MODEL.fee_per_share_float == tfi_engine.COMMISSION_PER_SHARE_FLOAT == 0.0
    engine_config = trinity_engine.DEFAULT_CONFIG
    live_config = trinity.LIVE_CONFIG
    assert live_config.vol_target_tuple == (engine_config.target_portfolio_vol_float, engine_config.trigger_portfolio_vol_float)
    assert (live_config.asset_vol_lookback_int, live_config.portfolio_vol_lookback_int) == (engine_config.asset_vol_lookback_int, engine_config.portfolio_vol_lookback_int)
    assert live_config.exposure_band_float == trinity_engine.EXPOSURE_REBALANCE_BAND_FLOAT
    assert trinity.ENGINE_COST_MODEL.slippage_float == engine_config.slippage_float
    assert trinity.ENGINE_COST_MODEL.fee_per_share_float == engine_config.commission_per_share_float
    for name_str, spec_module, family_fn, size_int in (("tfi", tfi, tfi_family, 12), ("trinity", trinity, trinity_family, 27)):
        assert GATED_SPEC_DICT[name_str].strategy_import_str == spec_module.STRATEGY_IMPORT_STR
        family = family_fn(inputs=object())  # the factory loads data only when no inputs are given
        validate_family_id(family.family_id_str)
        assert len(family.config_list()) == size_int and family.live_config_dict in family.config_list()

def test_tfi_and_trinity_replicas_run_on_a_shuffled_matrix():
    from alpha.scout.specs import tfi, trinity

    rng_obj = np.random.default_rng(8)
    date_index = pd.bdate_range("2010-01-01", "2013-12-31")
    row_count_int = len(date_index)
    tfi_matrix = np.column_stack([
        rng_obj.normal(0.0002, 0.004, (row_count_int, 3)), 4.0 + rng_obj.normal(0, 0.5, row_count_int),
        np.full(row_count_int, 1.0), 6.0 + rng_obj.normal(0, 0.5, (row_count_int, 2)),
    ])
    trinity_matrix = rng_obj.normal(0.0003, 0.01, (row_count_int, 7))
    for spec_module, matrix, config_list in (
        (tfi, tfi_matrix, [{"history_month_int": 0, "threshold_quantile_float": 0.5}]),
        (trinity, trinity_matrix, [{"asset_vol_lookback_int": 63, "portfolio_vol_lookback_int": 63}]),
    ):
        for candidate_mat in (matrix, matrix[rng_obj.permutation(row_count_int)]):
            daily_vec = spec_module.fast_daily_list(candidate_mat, date_index, config_list)[0]
            assert daily_vec.shape == (row_count_int,) and np.isfinite(daily_vec).all() and (daily_vec != 0.0).sum() > 200

@pytest.mark.skipif("not _norgate_ready()")
@pytest.mark.parametrize("spec_name_str", ["tfi", "trinity"])
def test_beyond_6040_gate_passes_against_the_saved_engine_run(spec_name_str):
    from pathlib import Path

    from alpha.scout.gate.run import GATED_SPEC_DICT, run_gate
    from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH

    if not any(Path(MAIN_CHECKOUT_ROOT_PATH).glob(GATED_SPEC_DICT[spec_name_str].pickle_glob_str)):
        pytest.skip("No saved engine run")
    report = run_gate(spec_name_str)
    assert report.passed_bool, report.summary_str()


# ---------------------------------------------------------------------------------------------- month-end rebalancing flow
def _synthetic_eom_inputs():
    from alpha.scout.specs.eom import EomInputs, xnys_session_index

    session_index = xnys_session_index(pd.Timestamp("2006-06-30"))
    session_index = session_index[(session_index >= "2002-07-26") & (session_index <= "2006-06-30")]
    rng_obj = np.random.default_rng(11)
    close_df = pd.DataFrame(100.0 * np.cumprod(1.0 + rng_obj.normal(0.0003, [0.012, 0.009, 0.004], (len(session_index), 3)), axis=0),
                            index=session_index, columns=["SPY", "TLT", "IEF"])
    return EomInputs(open_df=close_df[["SPY", "TLT"]], close_df=close_df[["SPY", "TLT"]], dividend_df=close_df[["SPY", "TLT"]] * 0.0,
                     total_return_close_df=close_df, session_index=xnys_session_index(session_index[-1]))


def test_eom_month_table_and_schedule_match_the_engine():
    from alpha.scout.specs import eom
    from strategies.taa_beyond_6040 import (
        strategy_taa_month_end_rebalancing_flow as engine,
    )

    inputs = _synthetic_eom_inputs()
    spec_df = eom.month_table_df(inputs)
    engine_df = engine.build_month_table_df(inputs.total_return_close_df[["SPY", "IEF"]])
    assert np.array_equal(spec_df["pressure_bps_float"].to_numpy(), engine_df["pressure_ief_measure_bps_float"].to_numpy())
    for column_str in ("measure_date", "final_fill_date", "early_fill_date", "exit_fill_date"):
        assert (spec_df[column_str].to_numpy() == engine_df[column_str].to_numpy()).all()
    # The (0.2, 0.6) cuts on F are the engine's quintile buckets: 1 -> low, 4-5 -> high, 2-3 -> mid, NaN -> missing.
    bucket_vec = engine_df["bucket_ief_measure_causal_int"].to_numpy()
    expected_list = ["missing" if np.isnan(b) else "low" if b == 1 else "high" if b >= 4 else "mid" for b in bucket_vec]
    assert spec_df["state_str"].tolist() == expected_list and spec_df["state_str"].iloc[24:].nunique() == 3
    weight_df = eom.rebalance_weight_df(inputs)
    session_index = inputs.session_index
    for month_row in spec_df.iloc[1:-1].itertuples():
        # dtme 6 entry, month-end reversal, session-5 exit; the measure is one full session before the first fill.
        assert session_index[session_index.get_loc(month_row.final_fill_date) - 1] == month_row.measure_date
        assert tuple(weight_df.loc[month_row.early_fill_date]) == eom.target_weight_tuple(month_row.state_str, "early")
        assert tuple(weight_df.loc[month_row.exit_fill_date]) == (0.0, 0.0)


def test_eom_family_grid_and_replica_on_a_shuffle():
    from alpha.scout.families import validate_family_id
    from alpha.scout.family import eom_family
    from alpha.scout.gate.run import GATED_SPEC_DICT
    from alpha.scout.specs import eom

    family = eom_family(inputs=object())  # the factory loads data only when no inputs are given
    validate_family_id(family.family_id_str)
    assert family.family_id_str == "calendar_and_flow" and family.offset_count_int == 1
    assert len(family.config_list()) == 27 and family.live_config_dict in family.config_list()
    assert GATED_SPEC_DICT["eom"].strategy_import_str == eom.STRATEGY_IMPORT_STR
    with pytest.raises(ValueError, match="luck band"):
        eom.EomConfig(decision_offset_int=1)
    inputs = _synthetic_eom_inputs()
    date_index, matrix = eom.mcpt_matrix(inputs, end_date_str=None)
    assert date_index[0] == pd.Timestamp("2002-08-01") and matrix.shape == (len(date_index), 3)
    rng_obj = np.random.default_rng(3)
    for candidate_mat in (matrix, matrix[rng_obj.permutation(len(matrix))]):
        daily_vec = eom.fast_daily_list(candidate_mat, date_index, [family.live_config_dict])[0]
        assert np.isfinite(daily_vec).all() and 300 < (daily_vec != 0.0).sum() < 0.7 * len(daily_vec)
    # Unshuffled, the replica tracks the engine path (gross, constant weights vs fixed shares with costs).
    engine_ser = eom.simulate_config(inputs).daily_return_ser
    fast_ser = pd.Series(eom.fast_daily_list(matrix, date_index, [eom.LIVE_CONFIG])[0], index=date_index).reindex(engine_ser.index)
    assert np.corrcoef(engine_ser, fast_ser)[0, 1] > 0.99


@pytest.mark.skipif("not _norgate_ready()")
def test_eom_gate_passes_and_blocks_the_hindsight_free_calendar(monkeypatch):
    from dataclasses import replace
    from pathlib import Path

    from alpha.scout.gate.run import GATED_SPEC_DICT, run_gate
    from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH
    from alpha.scout.specs import eom

    if not any(Path(MAIN_CHECKOUT_ROOT_PATH).glob(GATED_SPEC_DICT["eom"].pickle_glob_str)):
        pytest.skip("No saved engine run; run `python -m alpha.scout gate eom --fresh`.")
    report = run_gate("eom")
    assert report.passed_bool, report.summary_str()
    # Truth mode moves the October 2012 entry by two sessions (deviation xnys_closure_hindsight): the gate must see it.
    monkeypatch.setattr(eom, "LIVE_CONFIG", replace(eom.LIVE_CONFIG, notice_time_calendar_bool=True))
    assert not run_gate("eom").passed_bool


# ---------------------------------------------------------------------------------------------- Sector ETF IBS event pods
def _synthetic_sector_inputs(symbol_tuple, seed_int: int = 10, row_count_int: int = 700):
    from alpha.scout.specs.sector_ibs import SectorEtfInputs

    session_index = pd.bdate_range("2015-01-01", periods=row_count_int)
    rng_obj = np.random.default_rng(seed_int)
    shape_tuple = (row_count_int, len(symbol_tuple))
    market_vec = rng_obj.normal(0.0, 0.010, (row_count_int, 1))
    close_mat = 50.0 * np.exp(np.cumsum(market_vec + rng_obj.normal(0.0003, 0.008, shape_tuple), axis=0))
    open_mat = np.vstack([close_mat[:1], close_mat[:-1]]) * np.exp(rng_obj.normal(0.0, 0.004, shape_tuple))
    high_mat = np.maximum(open_mat, close_mat) * np.exp(np.abs(rng_obj.normal(0.0, 0.006, shape_tuple)))
    low_mat = np.minimum(open_mat, close_mat) * np.exp(-np.abs(rng_obj.normal(0.0, 0.006, shape_tuple)))
    frame = lambda mat: pd.DataFrame(mat, index=session_index, columns=list(symbol_tuple))
    return SectorEtfInputs(open_df=frame(open_mat), high_df=frame(high_mat), low_df=frame(low_mat), close_df=frame(close_mat),
                           dividend_df=frame(close_mat * 0.0), total_return_close_df=frame(close_mat), backtest_start_str="2015-01-01")

def test_sector_ibs_atr_matches_talib_and_features_are_causal():
    import talib

    from alpha.scout.specs import sector_ibs

    inputs = _synthetic_sector_inputs(sector_ibs.TRADED_TUPLE)
    high_vec, low_vec, close_vec = (frame["XLB"].to_numpy() for frame in (inputs.high_df, inputs.low_df, inputs.close_df))
    np.testing.assert_array_equal(sector_ibs.wilder_atr_vec(high_vec, low_vec, close_vec, 14), talib.ATR(high_vec, low_vec, close_vec, timeperiod=14))
    full_dict = sector_ibs.feature_dict(inputs)
    cut_ts = inputs.close_df.index[400]
    cut_inputs = dataclasses.replace(inputs, **{name: getattr(inputs, name).loc[:cut_ts] for name in ("open_df", "high_df", "low_df", "close_df")})
    for name_str, cut_df in sector_ibs.feature_dict(cut_inputs).items():  # a feature at T never changes when later bars arrive
        pd.testing.assert_frame_equal(cut_df, full_dict[name_str].loc[:cut_ts])
    assert full_dict["prior_natr_df"].notna().sum().min() > 600

def test_sector_ibs_event_rule_slots_ranking_and_untouched_holdings():
    from alpha.scout.engines.weights import CostModel, simulate
    from alpha.scout.specs import sector_ibs

    inputs = _synthetic_sector_inputs(sector_ibs.TRADED_TUPLE)
    config = dataclasses.replace(sector_ibs.LIVE_CONFIG, entry_ibs_max_float=0.25, downshock_atr_max_float=-0.2, exit_ibs_min_float=0.75)
    result = sector_ibs.simulate_config(inputs, config, CostModel())
    position_df = result.daily_position_df
    assert len(result.trade_df) > 100 and (position_df > 0).sum(axis=1).max() == config.max_positions_int
    # A held ETF is never resized: its trades alternate one entry and one full exit.
    for _asset_str, frame in result.trade_df.groupby("asset"):
        delta_vec = frame["delta_float"].to_numpy()
        exit_count_int = len(delta_vec) // 2
        assert (delta_vec[::2] > 0).all() and np.array_equal(delta_vec[1::2], -delta_vec[: 2 * exit_count_int: 2])
    # Entries on a day are the highest prior-NATR flat candidates.
    entry_mat, _, rank_mat = sector_ibs.signal_mats(sector_ibs.feature_dict(inputs, config), config)
    session_index = inputs.close_df.index
    held_before_df = position_df.shift(1).fillna(0.0)
    checked_int = 0
    for date, row in result.decided_weight_df.iterrows():
        if date == position_df.index[0]:
            continue
        p_int = session_index.get_loc(date) - 1
        entered_vec = np.flatnonzero(row.to_numpy() > 0.0)
        flat_vec = np.flatnonzero((held_before_df.loc[date].to_numpy() == 0.0) & entry_mat[p_int])
        if entered_vec.size and entered_vec.size < flat_vec.size:
            assert rank_mat[p_int, entered_vec].min() >= rank_mat[p_int, np.setdiff1d(flat_vec, entered_vec)].max()
            checked_int += 1
    assert checked_int > 5
    replay = simulate(inputs.open_df, inputs.close_df, inputs.dividend_df, result.decided_weight_df, start_date=result.total_value_ser.index[0],
                      cost_model=CostModel(), fractional_shares_bool=True, hold_nan_bool=True)
    assert np.array_equal(replay.total_value_ser.to_numpy(), result.total_value_ser.to_numpy())

def test_sector_ibs_specs_match_their_engines_and_are_gated():
    import importlib

    from alpha.scout.families import validate_family_id
    from alpha.scout.family import dispersion_ibs_family, sector_ibs_family
    from alpha.scout.gate.run import GATED_SPEC_DICT
    from alpha.scout.specs import sector_dispersion_ibs, sector_ibs
    from strategies.mean_reversion import (
        strategy_mr_sector_dispersion_ibs_kie_ihi_xlc_asset_sma200 as sma_engine,
    )
    from strategies.mean_reversion import (
        strategy_mr_us_sector_etf_ibs_downshock_vox_iyr as downshock_engine,
    )

    engine_config = downshock_engine.DEFAULT_CONFIG
    assert sector_ibs.TRADED_TUPLE == engine_config.symbol_tuple and sector_ibs.HISTORY_START_STR == engine_config.history_start_date_str
    for spec_str, engine_str in (("entry_ibs_max_float", "entry_ibs_max_float"), ("downshock_atr_max_float", "downshock_atr_max_float"),
                                 ("exit_ibs_min_float", "exit_ibs_min_float"), ("atr_lookback_int", "atr_lookback_day_int"),
                                 ("range_median_lookback_int", "range_median_lookback_day_int"), ("max_positions_int", "max_positions_int"),
                                 ("sizing_multiplier_float", "sizing_multiplier_float"), ("sizing_universe_count_int", "sizing_universe_count_int")):
        assert getattr(sector_ibs.LIVE_CONFIG, spec_str) == getattr(engine_config, engine_str)
    assert sector_ibs.ENGINE_COST_MODEL.slippage_float == 0.00025 and sector_ibs.ENGINE_COST_MODEL.min_fee_float == 1.0
    assert sma_engine.ASSET_SMA_LOOKBACK_DAY_INT == sector_dispersion_ibs.VARIANT_DICT["dispersion_ibs_kie_ihi_xlc_sma200"].config.asset_sma_int
    for name_str, variant in sector_dispersion_ibs.VARIANT_DICT.items():
        engine_config = importlib.import_module(variant.strategy_import_str).DEFAULT_CONFIG
        assert variant.symbol_tuple == engine_config.symbol_tuple and GATED_SPEC_DICT[name_str].strategy_import_str == variant.strategy_import_str
        assert (variant.config.entry_ibs_max_float, variant.config.exit_ibs_min_float, variant.config.min_relative_range_float,
                variant.config.range_vol_lookback_int, variant.config.portfolio_leverage_float) == (
            engine_config.entry_ibs_max_float, engine_config.exit_ibs_min_float, engine_config.min_relative_range_float,
            engine_config.range_vol_lookback_day_int, engine_config.portfolio_leverage_float)
        cost_model = sector_dispersion_ibs.ENGINE_COST_MODEL
        assert (cost_model.slippage_float, cost_model.fee_per_share_float, cost_model.min_fee_float) == (
            engine_config.slippage_float, engine_config.commission_per_share_float, engine_config.commission_minimum_float)
    assert GATED_SPEC_DICT["sector_ibs_vox_iyr"].strategy_import_str == sector_ibs.STRATEGY_IMPORT_STR
    for family in (sector_ibs_family(inputs=object()), *(dispersion_ibs_family(n, inputs=object()) for n in sector_dispersion_ibs.VARIANT_DICT)):
        validate_family_id(family.family_id_str)
        assert family.family_id_str == "etf_short_term_reversal" and family.offset_count_int == 1
        assert len(family.config_list()) == 27 and family.live_config_dict in family.config_list()
    with pytest.raises(ValueError, match="offset"):
        dataclasses.replace(sector_ibs.LIVE_CONFIG, decision_offset_int=1)

def test_sector_ibs_replicas_track_the_spec_and_run_on_a_shuffle():
    from alpha.scout.engines.weights import CostModel
    from alpha.scout.specs import sector_dispersion_ibs, sector_ibs

    gross_cost = CostModel(slippage_float=0.0, fee_per_share_float=0.0, min_fee_float=0.0)
    rng_obj = np.random.default_rng(12)
    dispersion_config = sector_dispersion_ibs.DispersionIbsConfig(asset_sma_int=200, entry_ibs_max_float=0.2)
    for spec_module, symbol_tuple, config, config_dict, kwarg_dict in (
        (sector_ibs, sector_ibs.TRADED_TUPLE, dataclasses.replace(sector_ibs.LIVE_CONFIG, entry_ibs_max_float=0.25, downshock_atr_max_float=-0.2),
         {"entry_ibs_max_float": 0.25, "downshock_atr_max_float": -0.2}, {}),
        (sector_dispersion_ibs, sector_dispersion_ibs.KIE_IHI_TUPLE, dispersion_config, {}, {"base_config": dispersion_config}),
    ):
        inputs = _synthetic_sector_inputs(symbol_tuple, row_count_int=900)
        date_index, matrix = spec_module.mcpt_matrix(inputs, end_date_str="2030-01-01")
        assert matrix.shape == (len(date_index), 5 * len(symbol_tuple))
        fast_vec = spec_module.fast_daily_list(matrix, date_index, [config_dict], **kwarg_dict)[0]
        engine_ser = spec_module.simulate_config(inputs, config, gross_cost).daily_return_ser
        assert np.corrcoef(engine_ser, pd.Series(fast_vec, index=date_index).reindex(engine_ser.index))[0, 1] > 0.99
        shuffled_vec = spec_module.fast_daily_list(matrix[rng_obj.permutation(len(date_index))], date_index, [config_dict], **kwarg_dict)[0]
        assert np.isfinite(shuffled_vec).all() and (shuffled_vec != 0.0).sum() > 100

def test_sector_ibs_s3_inputs_feed_the_class_e_station():
    from alpha.scout.specs import sector_ibs

    inputs = _synthetic_sector_inputs(sector_ibs.TRADED_TUPLE, row_count_int=900)
    input_dict = sector_ibs.s3_inputs(inputs, end_date_str="2030-01-01")
    assert input_dict["horizon_int"] in sector_ibs.S3_HORIZON_TUPLE and input_dict["event_mask_df"].to_numpy().sum() > 50
    result_dict = sector_ibs.s3_result(input_dict)
    assert len(result_dict["check_list"]) == 8 and {v for _, v, _ in result_dict["check_list"]} <= {"PASS", "WARN", "FAIL"}

@pytest.mark.skipif("not _norgate_ready()")
@pytest.mark.parametrize("spec_name_str", ["sector_ibs_vox_iyr","dispersion_ibs_kie_ihi_xlc", "dispersion_ibs_kie_ihi_xlc_sma200", "dispersion_ibs_kie_ihi_sma200"])
def test_sector_ibs_gate_passes_against_the_saved_engine_run(spec_name_str):
    from pathlib import Path

    from alpha.scout.gate.run import GATED_SPEC_DICT, run_gate
    from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH

    if not any(Path(MAIN_CHECKOUT_ROOT_PATH).glob(GATED_SPEC_DICT[spec_name_str].pickle_glob_str)):
        pytest.skip("No saved engine run")
    report = run_gate(spec_name_str)
    assert report.passed_bool, report.summary_str()

@pytest.mark.skipif("not _norgate_ready()")
def test_sector_ibs_mcpt_replica_tracks_the_engine():
    from alpha.scout.family import sector_ibs_family
    from alpha.scout.specs import sector_ibs

    inputs = sector_ibs.load_inputs()
    family = sector_ibs_family(inputs)
    date_index, matrix = sector_ibs.mcpt_matrix(inputs)
    assert matrix.shape == (len(date_index), 5 * len(sector_ibs.TRADED_TUPLE)) and date_index[-1] <= pd.Timestamp("2022-12-30")
    fast_ser = pd.Series(sector_ibs.fast_daily_list(matrix, date_index, [family.live_config_dict])[0], index=date_index)
    engine_ser = family.run_config(family.live_config_dict).daily_return_ser.loc[:"2022-12-30"]
    assert np.corrcoef(engine_ser, fast_ser.reindex(engine_ser.index))[0, 1] > 0.99  # 0.999 on 2026-10-02 (27-config min 0.998)
