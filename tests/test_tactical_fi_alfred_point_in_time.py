"""Tactical FI ALFRED point-in-time path and stale-input rule.

The synthetic tests run offline. The tests marked ``real data`` use the
governed ALFRED snapshot and Norgate prices, like the existing Tactical FI
tests.
"""

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from alpha.data.alfred_snapshot import AlfredVintageSnapshot, encode_vintage_run_df
from strategies.taa_beyond_6040 import strategy_taa_tactical_fixed_income_ief_lqd as tfi


SERIES_TUPLE = tfi.FRED_SERIES_ID_TUPLE


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _session_index() -> pd.DatetimeIndex:
    # Runs into August so the July decision has a next-open fill session.
    return pd.bdate_range("2014-01-02", "2014-08-05")


def _current_yield_df() -> pd.DataFrame:
    date_index = pd.bdate_range("2013-06-03", "2014-07-31")
    rng = np.random.default_rng(20260928)
    step_arr = rng.normal(0.0, 0.03, size=(len(date_index), 4)).cumsum(axis=0)
    base_arr = np.array([2.5, 0.1, 4.2, 5.0])
    return pd.DataFrame(
        np.round(base_arr + step_arr, 2),
        index=date_index,
        columns=list(SERIES_TUPLE),
    )


REVISED_OBSERVATION_TS = pd.Timestamp("2014-05-29")
INITIAL_DAAA_OFFSET_FLOAT = -1.0
OUTAGE_START_TS = pd.Timestamp("2014-06-10")


OUTAGE_END_TS = pd.Timestamp("2014-06-30")
HISTORY_REVISION_OFFSET_FLOAT = 10.0


def _published_vintage_df(
    current_yield_df: pd.DataFrame,
    vintage_date_ts: pd.Timestamp,
    scenario_str: str = "default",
) -> pd.DataFrame:
    """What FRED would have shown on the vintage date in this synthetic history.

    Every scenario: observations are published the next business day.

    ``default``:
    - DAAA for 2014-05-29 was first published 1.00 lower and revised to its
      current value by the 2014-06-30 vintage.
    - DAAA/DBAA stopped publishing from 2014-06-10 until a backfill that is
      present in the 2014-07-31 vintage.

    ``never_backfilled``: the same outage, but publication resumes in July and
    the 2014-06-10..06-30 gap is never filled.

    ``history_revision``: every DAAA observation before 2014-04-01 was first
    published 10.00 higher and revised to its current value on 2014-06-01.
    Nothing else differs from the current vintage.
    """
    vintage_df = current_yield_df.loc[current_yield_df.index < vintage_date_ts].copy()
    if scenario_str == "history_revision":
        if vintage_date_ts < pd.Timestamp("2014-06-01"):
            vintage_df.loc[vintage_df.index < pd.Timestamp("2014-04-01"), "DAAA"] += (
                HISTORY_REVISION_OFFSET_FLOAT
            )
        return vintage_df
    if vintage_date_ts < pd.Timestamp("2014-06-30") and REVISED_OBSERVATION_TS in vintage_df.index:
        vintage_df.loc[REVISED_OBSERVATION_TS, "DAAA"] += INITIAL_DAAA_OFFSET_FLOAT
    if scenario_str == "never_backfilled":
        gap_bool_arr = (vintage_df.index >= OUTAGE_START_TS) & (vintage_df.index <= OUTAGE_END_TS)
        vintage_df.loc[gap_bool_arr, ["DAAA", "DBAA"]] = np.nan
    elif pd.Timestamp("2014-06-10") <= vintage_date_ts < pd.Timestamp("2014-07-15"):
        vintage_df.loc[vintage_df.index >= OUTAGE_START_TS, ["DAAA", "DBAA"]] = np.nan
    return vintage_df


def _snapshot_dict(
    current_yield_df: pd.DataFrame,
    session_index: pd.DatetimeIndex,
    frozen_signal_df: pd.DataFrame,
    later_value_noise_after_ts: pd.Timestamp | None = None,
    scenario_str: str = "default",
) -> dict[str, AlfredVintageSnapshot]:
    first_ts = pd.Timestamp(tfi.FIRST_ALFRED_DECISION_DATE_STR)
    decision_index = frozen_signal_df.index[frozen_signal_df.index >= first_ts]
    vintage_date_list = sorted(
        set(decision_index)
        | {tfi.previous_session(date_ts, session_index) for date_ts in decision_index}
    )
    rng = np.random.default_rng(7)
    snapshot_dict: dict[str, AlfredVintageSnapshot] = {}
    for series_id_str in SERIES_TUPLE:
        vintage_value_dict = {}
        for vintage_date_ts in vintage_date_list:
            value_ser = _published_vintage_df(current_yield_df, vintage_date_ts, scenario_str)[
                series_id_str
            ].dropna()
            if later_value_noise_after_ts is not None and vintage_date_ts > later_value_noise_after_ts:
                # Everything published after the cutoff becomes noise.
                value_ser = value_ser + rng.normal(0.0, 3.0, size=len(value_ser))
            value_ser.name = series_id_str
            vintage_value_dict[pd.Timestamp(vintage_date_ts)] = value_ser
        snapshot_dict[series_id_str] = AlfredVintageSnapshot(
            series_id_str=series_id_str,
            run_df=encode_vintage_run_df(vintage_value_dict),
            vintage_date_index=pd.DatetimeIndex(vintage_date_list),
            source_path_str="synthetic",
            sha256_str="synthetic",
        )
    return snapshot_dict


def _synthetic_pit(
    stale_input_policy_str: str = tfi.STALE_INPUT_POLICY_BLOCK_AND_HOLD_STR,
    later_value_noise_after_ts: pd.Timestamp | None = None,
    alfred_vintage_policy_str: str = tfi.ALFRED_VINTAGE_DECISION_DATE_STR,
    scenario_str: str = "default",
):
    session_index = _session_index()
    current_yield_df = _current_yield_df()
    frozen_signal_df, frozen_weight_df = tfi.build_month_end_signal_and_weight_df(
        current_yield_df,
        session_index,
        "2014-07",
    )
    snapshot_dict = _snapshot_dict(
        current_yield_df,
        session_index,
        frozen_signal_df,
        later_value_noise_after_ts,
        scenario_str,
    )
    signal_df, weight_df = tfi.build_point_in_time_signal_and_weight_df(
        frozen_signal_df,
        frozen_weight_df,
        snapshot_dict,
        session_index,
        alfred_vintage_policy_str,
    )
    signal_df, weight_df = tfi.apply_stale_input_rule(
        signal_df,
        weight_df,
        session_index,
        stale_input_policy_str,
    )
    return current_yield_df, frozen_signal_df, frozen_weight_df, snapshot_dict, signal_df, weight_df


# ---------------------------------------------------------------------------
# Stale-input rule
# ---------------------------------------------------------------------------


def test_observation_age_counts_sessions_after_observation_up_to_t_minus_one() -> None:
    session_index = _session_index()
    decision_ts = pd.Timestamp("2014-05-30")

    assert tfi.observation_age_sessions_int(pd.Timestamp("2014-05-29"), decision_ts, session_index) == 0
    assert tfi.observation_age_sessions_int(pd.Timestamp("2014-05-27"), decision_ts, session_index) == 2
    assert tfi.observation_age_sessions_int(pd.Timestamp("2014-05-26"), decision_ts, session_index) == 3
    # A bond-market date that is not an equity session still counts from the next session.
    assert tfi.observation_age_sessions_int(pd.Timestamp("2014-05-24"), decision_ts, session_index) == 4


def _one_decision_frames(observation_date_str: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    decision_ts = pd.Timestamp("2014-05-30")
    signal_df = pd.DataFrame(
        {"observation_date": [pd.Timestamp(observation_date_str)]},
        index=pd.DatetimeIndex([decision_ts], name="decision_date"),
    )
    weight_df = pd.DataFrame(
        {
            "decision_date": [decision_ts],
            "observation_date": [pd.Timestamp(observation_date_str)],
            "IEF": [0.5],
            "LQD": [0.5],
            "Cash": [0.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2014-06-02")], name="rebalance_date"),
    )
    return signal_df, weight_df


def test_stale_rule_boundary_is_two_sessions() -> None:
    session_index = _session_index()

    signal_df, weight_df = tfi.apply_stale_input_rule(
        *_one_decision_frames("2014-05-27"), session_index, tfi.STALE_INPUT_POLICY_RAISE_STR
    )
    assert not bool(signal_df["stale_input_blocked_bool"].iloc[0])
    assert len(weight_df) == 1

    signal_df, weight_df = tfi.apply_stale_input_rule(
        *_one_decision_frames("2014-05-26"),
        session_index,
        tfi.STALE_INPUT_POLICY_BLOCK_AND_HOLD_STR,
    )
    assert bool(signal_df["stale_input_blocked_bool"].iloc[0])
    assert signal_df["observation_age_sessions_int"].iloc[0] == 3
    assert weight_df.empty


def test_synthetic_outage_blocks_the_decision_and_places_no_order() -> None:
    (_current_df, _frozen_sig, _frozen_w, _snap, signal_df, weight_df) = _synthetic_pit()

    outage_row_ser = signal_df.loc[pd.Timestamp("2014-06-30")]
    assert bool(outage_row_ser["stale_input_blocked_bool"])
    assert outage_row_ser["observation_date"] == pd.Timestamp("2014-06-09")
    assert outage_row_ser["observation_age_sessions_int"] > tfi.MAX_OBSERVATION_AGE_SESSIONS_INT
    assert pd.Timestamp("2014-06-30") not in set(weight_df["decision_date"])
    # The other decisions still trade; after the backfill the rule clears.
    assert signal_df["stale_input_blocked_bool"].sum() == 1
    assert signal_df.loc[pd.Timestamp("2014-07-31"), "observation_date"] == pd.Timestamp("2014-07-30")
    assert pd.Timestamp("2014-07-31") in set(weight_df["decision_date"])


def test_synthetic_outage_raises_under_the_raise_policy() -> None:
    with pytest.raises(tfi.StaleMacroInputError, match="2014-06-30"):
        _synthetic_pit(stale_input_policy_str=tfi.STALE_INPUT_POLICY_RAISE_STR)


# ---------------------------------------------------------------------------
# Vintage correctness
# ---------------------------------------------------------------------------


def test_decision_uses_the_value_published_by_t_not_the_later_revision() -> None:
    (current_df, frozen_signal_df, _frozen_w, _snap, signal_df, _w) = _synthetic_pit()
    decision_ts = pd.Timestamp("2014-05-30")
    row_ser = signal_df.loc[decision_ts]

    assert row_ser["observation_date"] == REVISED_OBSERVATION_TS
    expected_credit_float = (
        0.5 * (current_df.loc[REVISED_OBSERVATION_TS, "DAAA"] + INITIAL_DAAA_OFFSET_FLOAT
               + current_df.loc[REVISED_OBSERVATION_TS, "DBAA"])
        - current_df.loc[REVISED_OBSERVATION_TS, "DGS3MO"]
    )
    assert row_ser["credit_spread_float"] == pytest.approx(expected_credit_float)
    # The frozen (current-vintage) row used the later revision.
    assert frozen_signal_df.loc[decision_ts, "credit_spread_float"] == pytest.approx(
        expected_credit_float - 0.5 * INITIAL_DAAA_OFFSET_FLOAT
    )


def test_values_published_after_t_can_never_change_decision_t() -> None:
    cutoff_ts = pd.Timestamp("2014-05-30")
    base_signal_df = _synthetic_pit()[4]
    noisy_signal_df = _synthetic_pit(later_value_noise_after_ts=cutoff_ts)[4]

    compare_column_list = [
        "observation_date",
        "term_spread_float",
        "credit_spread_float",
        "term_threshold_float",
        "credit_threshold_float",
        "term_state_float",
        "credit_state_float",
    ]
    pd.testing.assert_frame_equal(
        base_signal_df.loc[base_signal_df.index <= cutoff_ts, compare_column_list],
        noisy_signal_df.loc[noisy_signal_df.index <= cutoff_ts, compare_column_list],
    )
    # The noise does reach later decisions, so the test can detect leakage.
    assert not base_signal_df.loc[pd.Timestamp("2014-07-31"), compare_column_list].equals(
        noisy_signal_df.loc[pd.Timestamp("2014-07-31"), compare_column_list]
    )


def test_every_pit_row_is_rebuilt_from_its_own_vintage() -> None:
    (_current, frozen_signal_df, _fw, snapshot_dict, signal_df, _w) = _synthetic_pit()
    first_ts = pd.Timestamp(tfi.FIRST_ALFRED_DECISION_DATE_STR)

    for decision_ts, row_ser in signal_df.iterrows():
        if decision_ts < first_ts:
            assert row_ser["fred_data_source_str"] == "frozen_current_vintage_before_alfred_archive"
            assert row_ser["term_spread_float"] == frozen_signal_df.loc[decision_ts, "term_spread_float"]
            continue
        vintage_ts = pd.Timestamp(row_ser["vintage_date"])
        observation_ts = pd.Timestamp(row_ser["observation_date"])
        assert vintage_ts == decision_ts
        assert observation_ts < vintage_ts
        vintage_value_dict = {
            series_id_str: snapshot_dict[series_id_str].value_ser_as_of(vintage_ts).loc[observation_ts]
            for series_id_str in SERIES_TUPLE
        }
        assert row_ser["term_spread_float"] == pytest.approx(
            vintage_value_dict["DGS10"] - vintage_value_dict["DGS3MO"]
        )
        assert row_ser["credit_spread_float"] == pytest.approx(
            0.5 * (vintage_value_dict["DAAA"] + vintage_value_dict["DBAA"])
            - vintage_value_dict["DGS3MO"]
        )


def test_previous_session_policy_uses_the_prior_session_vintage() -> None:
    signal_df = _synthetic_pit(
        alfred_vintage_policy_str=tfi.ALFRED_VINTAGE_PREVIOUS_SESSION_STR
    )[4]
    row_ser = signal_df.loc[pd.Timestamp("2014-05-30")]

    assert row_ser["vintage_date"] == pd.Timestamp("2014-05-29")
    # Observation T-1 is not yet published in the T-1 vintage.
    assert row_ser["observation_date"] == pd.Timestamp("2014-05-28")


def test_older_history_is_rebuilt_from_the_vintage_not_the_current_file() -> None:
    (_current, frozen_signal_df, _fw, _snap, signal_df, _w) = _synthetic_pit(
        scenario_str="history_revision"
    )
    before_revision_ts = pd.Timestamp("2014-05-30")
    after_revision_ts = pd.Timestamp("2014-06-30")

    # Row T's own inputs are unrevised, so only the median history differs.
    assert signal_df.loc[before_revision_ts, "credit_spread_float"] == pytest.approx(
        frozen_signal_df.loc[before_revision_ts, "credit_spread_float"]
    )
    assert signal_df.loc[before_revision_ts, "credit_threshold_float"] > (
        frozen_signal_df.loc[before_revision_ts, "credit_threshold_float"] + 1.0
    )
    # Once the revision is published, the vintage history equals the current file.
    assert signal_df.loc[after_revision_ts, "credit_threshold_float"] == pytest.approx(
        frozen_signal_df.loc[after_revision_ts, "credit_threshold_float"]
    )


def test_a_usable_decision_with_a_stale_month_in_its_median_fails_loud() -> None:
    # July is fresh again but June's gap is never filled, so July's median
    # would contain a stale June month: that rule is not approved.
    with pytest.raises(AssertionError, match="median-composition rule"):
        _synthetic_pit(scenario_str="never_backfilled")


def test_panel_rejects_an_observation_after_its_vintage_date() -> None:
    vintage_ts = pd.Timestamp("2014-05-30")
    snapshot_dict = {}
    for series_id_str in SERIES_TUPLE:
        run_df = pd.DataFrame(
            {
                "observation_date": pd.to_datetime(["2014-05-29", "2014-06-02"]),
                "value": [1.0, 2.0],
                "first_vintage_date": [vintage_ts, vintage_ts],
                "last_vintage_date": [vintage_ts, vintage_ts],
            }
        )
        snapshot_dict[series_id_str] = AlfredVintageSnapshot(
            series_id_str=series_id_str,
            run_df=run_df,
            vintage_date_index=pd.DatetimeIndex([vintage_ts]),
            source_path_str="synthetic",
            sha256_str="synthetic",
        )

    with pytest.raises(AssertionError, match="after its vintage date"):
        tfi.alfred_yield_panel_as_of(snapshot_dict, vintage_ts)


def test_run_variant_keeps_config_modes_unless_a_keyword_overrides(monkeypatch) -> None:
    seen_config_list = []

    class _Stop(Exception):
        pass

    def fake_get_data(config_obj):
        seen_config_list.append(config_obj)
        raise _Stop()

    monkeypatch.setattr(tfi, "get_tactical_yield_data", fake_get_data)
    pit_config_obj = replace(
        tfi.DEFAULT_CONFIG,
        fred_data_mode_str=tfi.FRED_DATA_MODE_ALFRED_PIT_STR,
        alfred_vintage_policy_str=tfi.ALFRED_VINTAGE_PREVIOUS_SESSION_STR,
    )
    for kwarg_dict in (
        {"config_obj": pit_config_obj},
        {"config_obj": pit_config_obj, "stale_input_policy_str": "raise"},
        {"fred_data_mode_str": "alfred_point_in_time"},
        {},
    ):
        with pytest.raises(_Stop):
            tfi.run_variant(show_display_bool=False, save_results_bool=False, **kwarg_dict)

    assert [
        (c.fred_data_mode_str, c.alfred_vintage_policy_str, c.stale_input_policy_str)
        for c in seen_config_list
    ] == [
        ("alfred_point_in_time", "previous_session", "block_and_hold"),
        ("alfred_point_in_time", "previous_session", "raise"),
        ("alfred_point_in_time", "decision_date", "block_and_hold"),
        ("frozen_current_vintage", "decision_date", "block_and_hold"),
    ]


def test_config_rejects_unknown_modes() -> None:
    with pytest.raises(ValueError, match="fred_data_mode_str"):
        tfi.TacticalYieldConfig(fred_data_mode_str="latest")
    with pytest.raises(ValueError, match="alfred_vintage_policy_str"):
        tfi.TacticalYieldConfig(alfred_vintage_policy_str="same_day")
    with pytest.raises(ValueError, match="stale_input_policy_str"):
        tfi.TacticalYieldConfig(stale_input_policy_str="go_to_cash")


def test_run_info_records_the_fred_data_mode(tmp_path) -> None:
    (tmp_path / "run_info.json").write_text(
        json.dumps({"entity_type": "strategy", "parameters": {"capital": 100000.0}}),
        encoding="utf-8",
    )
    signal_df = pd.DataFrame(
        {"stale_input_blocked_bool": [False, True]},
        index=pd.to_datetime(["2016-09-30", "2016-10-31"]),
    )
    config_obj = replace(
        tfi.DEFAULT_CONFIG,
        fred_data_mode_str=tfi.FRED_DATA_MODE_ALFRED_PIT_STR,
    )

    record_dict = tfi.fred_data_mode_record_dict(config_obj, signal_df)
    tfi._record_fred_data_mode_in_run_info(tmp_path, record_dict)

    parameter_dict = json.loads((tmp_path / "run_info.json").read_text(encoding="utf-8"))[
        "parameters"
    ]
    assert parameter_dict["capital"] == 100000.0
    assert parameter_dict["fred_data_mode_str"] == "alfred_point_in_time"
    assert parameter_dict["alfred_vintage_policy_str"] == "decision_date"
    assert parameter_dict["stale_input_blocked_decision_count_int"] == 1
    assert record_dict["stale_input_blocked_decision_date_list"] == ["2016-10-31"]
    assert tfi.strategy_name_for_config_str(config_obj) == (
        "strategy_taa_tactical_fixed_income_ief_lqd__alfred_pit_decision_date"
    )
    assert tfi.strategy_name_for_config_str(tfi.DEFAULT_CONFIG) == tfi.STRATEGY_NAME_STR
    assert "stale_input_blocked_decision_date_list" not in parameter_dict
    frozen_record_dict = tfi.fred_data_mode_record_dict(tfi.DEFAULT_CONFIG, signal_df.iloc[:1])
    assert frozen_record_dict["fred_data_mode_str"] == "frozen_current_vintage"
    assert not any(key_str.startswith("alfred_") for key_str in frozen_record_dict)


# ---------------------------------------------------------------------------
# Real data: governed snapshot, Norgate prices
# ---------------------------------------------------------------------------


PIT_CONFIG = replace(tfi.DEFAULT_CONFIG, fred_data_mode_str=tfi.FRED_DATA_MODE_ALFRED_PIT_STR)
OUTAGE_BLOCKED_DATE_LIST = ["2016-10-31", "2016-11-30", "2016-12-30", "2017-01-31", "2017-02-28"]


@pytest.fixture(scope="module")
def real_pit_data():
    frozen_tuple = tfi.get_tactical_yield_data(tfi.DEFAULT_CONFIG)
    pit_tuple = tfi.get_tactical_yield_data(PIT_CONFIG)
    _manifest_dict, snapshot_dict = tfi.load_alfred_point_in_time_snapshots(PIT_CONFIG)
    return frozen_tuple, pit_tuple, snapshot_dict


def test_real_frozen_mode_is_unchanged_and_stale_free(real_pit_data) -> None:
    frozen_tuple = real_pit_data[0]
    signal_df, weight_df = frozen_tuple[2], frozen_tuple[3]

    assert len(signal_df) == 289
    assert len(weight_df) == 289
    assert not signal_df["stale_input_blocked_bool"].any()
    assert (signal_df["observation_age_sessions_int"] == 0).all()
    assert tfi.canonical_dataframe_sha256_str(
        tfi.build_canonical_signal_contract_df(signal_df, weight_df)
    ) == tfi.FROZEN_SIGNAL_CONTRACT_SHA256_STR


def test_real_pit_blocks_exactly_the_2016_17_moodys_outage(real_pit_data) -> None:
    signal_df, weight_df = real_pit_data[1][2], real_pit_data[1][3]

    blocked_list = [
        date_ts.date().isoformat()
        for date_ts in signal_df.index[signal_df["stale_input_blocked_bool"]]
    ]
    assert blocked_list == OUTAGE_BLOCKED_DATE_LIST
    assert len(weight_df) == 289 - len(OUTAGE_BLOCKED_DATE_LIST)
    assert (signal_df.loc[~signal_df["stale_input_blocked_bool"], "observation_age_sessions_int"]
            <= tfi.MAX_OBSERVATION_AGE_SESSIONS_INT).all()
    # On 2016-12-30 FRED's latest published Moody's observation was 2016-10-07.
    assert signal_df.loc[pd.Timestamp("2016-12-30"), "observation_date"] == pd.Timestamp("2016-10-07")


def test_real_pit_rows_use_only_values_published_by_their_vintage(real_pit_data) -> None:
    signal_df = real_pit_data[1][2]
    snapshot_dict = real_pit_data[2]
    pit_signal_df = signal_df.loc[signal_df["fred_data_source_str"] == "alfred_vintage_decision_date"]

    assert pit_signal_df.index[0] == pd.Timestamp(tfi.FIRST_ALFRED_DECISION_DATE_STR)
    assert len(pit_signal_df) == 148
    for decision_ts, row_ser in pit_signal_df.iterrows():
        vintage_ts = pd.Timestamp(row_ser["vintage_date"])
        observation_ts = pd.Timestamp(row_ser["observation_date"])
        assert vintage_ts == decision_ts
        assert observation_ts < vintage_ts
        value_dict = {
            series_id_str: snapshot_dict[series_id_str].value_ser_as_of(vintage_ts).loc[observation_ts]
            for series_id_str in SERIES_TUPLE
        }
        assert row_ser["term_spread_float"] == pytest.approx(value_dict["DGS10"] - value_dict["DGS3MO"])
        assert row_ser["credit_spread_float"] == pytest.approx(
            0.5 * (value_dict["DAAA"] + value_dict["DBAA"]) - value_dict["DGS3MO"]
        )


def test_real_pit_raise_policy_stops_at_the_first_outage_decision() -> None:
    with pytest.raises(tfi.StaleMacroInputError, match="2016-10-31"):
        tfi.get_tactical_yield_data(
            replace(PIT_CONFIG, stale_input_policy_str=tfi.STALE_INPUT_POLICY_RAISE_STR)
        )


def test_real_snapshot_reproduces_the_leakage_hunt_flips_and_metrics(real_pit_data) -> None:
    from scripts.research.run_tactical_fi_alfred_replay import (
        flipped_decision_df,
        leakage_hunt_reproduction_weight_df,
        return_metric_dict,
        run_backtest,
    )

    (price_df, yield_df, frozen_signal_df, frozen_weight_df, cash_ser, fred_tuple) = real_pit_data[0]
    snapshot_dict = real_pit_data[2]
    reproduction_signal_df, reproduction_weight_df = leakage_hunt_reproduction_weight_df(
        yield_df,
        frozen_signal_df,
        frozen_weight_df,
        snapshot_dict,
        pd.DatetimeIndex(price_df.index),
    )

    flip_df = flipped_decision_df(frozen_signal_df, reproduction_signal_df)
    assert [date_ts.date().isoformat() for date_ts in flip_df.index] == ["2016-12-30", "2017-01-31"]
    assert flip_df["observation_date_candidate"].tolist() == [
        pd.Timestamp("2016-11-15"),
        pd.Timestamp("2016-12-16"),
    ]
    assert flip_df["term_state_frozen"].tolist() == [1.0, 0.0]
    assert flip_df["term_state_candidate"].tolist() == [0.0, 1.0]

    # Numbers from results/research/leakage_hunt_20260927/def/tfi_vintage_metrics.csv.
    frozen_strategy_obj = run_backtest(
        tfi.DEFAULT_CONFIG, price_df, frozen_signal_df, frozen_weight_df, cash_ser, fred_tuple
    )
    reproduction_strategy_obj = run_backtest(
        tfi.DEFAULT_CONFIG, price_df, frozen_signal_df, reproduction_weight_df, cash_ser, fred_tuple
    )
    frozen_book_dict = return_metric_dict(
        frozen_strategy_obj.results["daily_returns"], "2012-10-02", "2026-08-19"
    )
    reproduction_book_dict = return_metric_dict(
        reproduction_strategy_obj.results["daily_returns"], "2012-10-02", "2026-08-19"
    )
    reproduction_2014_dict = return_metric_dict(
        reproduction_strategy_obj.results["daily_returns"], "2014-05-01", "2026-08-19"
    )
    assert frozen_book_dict["cagr"] == pytest.approx(0.027071000400353817, abs=1e-12)
    assert frozen_book_dict["sharpe"] == pytest.approx(1.0636845500553298, abs=1e-10)
    assert reproduction_book_dict["cagr"] == pytest.approx(0.02736146583785537, abs=1e-12)
    assert reproduction_book_dict["sharpe"] == pytest.approx(1.0788066976661674, abs=1e-10)
    assert reproduction_2014_dict["cagr"] == pytest.approx(0.02990554166720205, abs=1e-12)
    assert reproduction_2014_dict["sharpe"] == pytest.approx(1.4427459900739439, abs=1e-10)


def test_real_previous_session_policy_matches_its_pinned_contract() -> None:
    # get_tactical_yield_data verifies the pinned previous-session contract hash.
    (_px, _y, frozen_signal_df, *_rest) = tfi.get_tactical_yield_data(tfi.DEFAULT_CONFIG)
    signal_df = tfi.get_tactical_yield_data(
        replace(PIT_CONFIG, alfred_vintage_policy_str=tfi.ALFRED_VINTAGE_PREVIOUS_SESSION_STR)
    )[2]

    blocked_list = [
        date_ts.date().isoformat()
        for date_ts in signal_df.index[signal_df["stale_input_blocked_bool"]]
    ]
    assert blocked_list == OUTAGE_BLOCKED_DATE_LIST
    usable_df = signal_df.loc[
        (signal_df.index >= pd.Timestamp(tfi.FIRST_ALFRED_DECISION_DATE_STR))
        & ~signal_df["stale_input_blocked_bool"]
    ]
    assert (usable_df["vintage_date"] < usable_df.index).all()
    frozen_df = frozen_signal_df.loc[usable_df.index]
    flip_bool_ser = (usable_df["term_state_float"] != frozen_df["term_state_float"]) | (
        usable_df["credit_state_float"] != frozen_df["credit_state_float"]
    )
    assert [date_ts.date().isoformat() for date_ts in usable_df.index[flip_bool_ser]] == [
        "2015-04-30",
        "2015-09-30",
        "2015-11-30",
        "2016-04-29",
        "2020-03-31",
    ]


def test_real_run_variant_labels_the_pit_run_and_holds_through_the_block() -> None:
    strategy_obj = tfi.run_variant(
        show_display_bool=False,
        save_results_bool=False,
        fred_data_mode_str="alfred_point_in_time",
    )
    frozen_strategy_obj = tfi.run_variant(show_display_bool=False, save_results_bool=False)

    def fills_in_block_df(run_obj):
        transaction_df = run_obj.get_transactions()
        bar_ser = pd.to_datetime(transaction_df["bar"])
        return transaction_df.loc[
            (bar_ser > pd.Timestamp("2016-10-31")) & (bar_ser <= pd.Timestamp("2017-03-31"))
        ]

    # Blocked months place no order: positions are held until the 2017-03-31
    # decision fills on 2017-04-03. The frozen run did trade in that window
    # (into IEF/LQD on 2017-01-03 and back out on 2017-02-01).
    assert fills_in_block_df(strategy_obj).empty
    assert not fills_in_block_df(frozen_strategy_obj).empty
    assert frozen_strategy_obj.name == tfi.STRATEGY_NAME_STR

    assert strategy_obj.name == "strategy_taa_tactical_fixed_income_ief_lqd__alfred_pit_decision_date"
    policy_dict = strategy_obj._data_adjustment_policy_dict
    assert policy_dict["fred_data_mode_str"] == "alfred_point_in_time"
    assert policy_dict["alfred_manifest_sha256_str"] == tfi.FROZEN_ALFRED_MANIFEST_SHA256_STR
    assert policy_dict["stale_input_blocked_decision_date_list"] == OUTAGE_BLOCKED_DATE_LIST
    assert strategy_obj._accounting_policy_dict["paper_live_authorized_bool"] is False
