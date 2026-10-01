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
