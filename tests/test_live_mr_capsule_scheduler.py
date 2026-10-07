"""Offline scheduler integration: real calendar, release parser, SQL and broker stub."""
from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from alpha.live import runner, scheduler_service, scheduler_utils
from alpha.live.models import DecisionPlan, PodState
from alpha.live.order_clerk import StubBrokerAdapter
from alpha.live.release_manifest import load_release_list
from alpha.live.state_store_v2 import LiveStateStore
from data.norgate_snapshot_store import MR_CAPSULE_DV2_PROFILE_STR, MR_CAPSULE_HPI_PROFILE_STR
from test_live_scheduler_service import (
    _insert_planned_decision_plan, _insert_ready_vplan, _insert_submitted_vplan,
    _insert_submitting_vplan, _write_manifest,
)


MARKET_TIMEZONE_OBJ = ZoneInfo("America/New_York")


def _time(date_str="2024-01-31", hour_int=16, minute_int=10):
    return datetime.fromisoformat(date_str).replace(hour=hour_int, minute=minute_int, tzinfo=MARKET_TIMEZONE_OBJ)


@pytest.fixture(params=["dv2_vix_gated", "hpi_vote_vix_gated"])
def capsule_case(request, tmp_path, monkeypatch):
    _write_manifest(tmp_path)
    manifest_path_obj = tmp_path / "releases" / "user_001" / "pod_test.yaml"
    profile_str = MR_CAPSULE_HPI_PROFILE_STR if request.param.startswith("hpi") else MR_CAPSULE_DV2_PROFILE_STR
    content_str = manifest_path_obj.read_text(encoding="utf-8")
    content_str = content_str.replace("strategies.dv2.strategy_mr_dv2:DVO2Strategy", f"strategies.mr_capsule.strategy_mr_{request.param}_bil")
    content_str = content_str.replace("norgate_eod_sp500_pit", profile_str)
    content_str = content_str.replace("params: {}", "params: {margin_account_confirmed_bool: true}")
    content_str = content_str.replace("pod_budget_fraction_float: 0.5", "pod_budget_fraction_float: 1.0")
    manifest_path_obj.write_text(content_str, encoding="utf-8")
    release_obj = load_release_list(str(tmp_path / "releases"))[0]
    store_obj = LiveStateStore(str(tmp_path / "state.sqlite3"))
    store_obj.upsert_release(release_obj)
    readiness_dict = {"date": pd.Timestamp("2024-01-31"), "calls": []}

    def read_snapshot(profile_str):
        readiness_dict["calls"].append(profile_str)
        return readiness_dict["date"]

    monkeypatch.setattr(scheduler_utils, "is_snapshot_mode_enabled_bool", lambda: True)
    monkeypatch.setattr(scheduler_utils, "load_latest_snapshot_session_label_ts", read_snapshot)
    return store_obj, release_obj, tmp_path, readiness_dict


def _state(release_obj, timestamp_ts=None, **change_dict):
    return replace(PodState(
        pod_id_str=release_obj.pod_id_str, user_id_str=release_obj.user_id_str,
        account_route_str=release_obj.account_route_str, position_amount_map={},
        cash_float=10000.0, total_value_float=10000.0, strategy_state_dict={},
        updated_timestamp_ts=timestamp_ts or _time(), snapshot_stage_str="eod", snapshot_source_str="broker",
    ), **change_dict)


def _decision(case_tuple, as_of_ts=None):
    store_obj, _, tmp_path, _ = case_tuple
    return scheduler_service.get_scheduler_decision(store_obj, as_of_ts or _time(), str(tmp_path / "releases"), "paper")


def _capture(case_tuple, broker_obj, as_of_ts=None):
    store_obj, _, tmp_path, _ = case_tuple
    return runner.eod_snapshot(
        store_obj, broker_obj, as_of_ts or _time(), env_mode_str="paper",
        releases_root_path_str=str(tmp_path / "releases"),
        log_path_str=str(tmp_path / "events.jsonl"), trace_enabled_bool=False,
    )


def _broker(release_obj, timestamp_ts=None):
    broker_obj = StubBrokerAdapter()
    broker_obj.seed_account_snapshot(
        account_route_str=release_obj.account_route_str, cash_float=10000.0,
        total_value_float=10000.0, position_amount_map={},
        snapshot_timestamp_ts=timestamp_ts or _time(), session_mode_str="paper",
    )
    return broker_obj


def test_capsule_profiles_reach_real_snapshot_gate(capsule_case):
    _, release_obj, _, readiness_dict = capsule_case
    gate_dict = scheduler_utils.evaluate_build_gate_dict(release_obj, _time())
    assert gate_dict["due_bool"]
    assert readiness_dict["calls"] == [release_obj.data_profile_str]


@pytest.mark.parametrize("data_current_bool,eod_kind_str,expected_build_bool", [
    (True, "missing", False), (True, "stale", False),
    (False, "current", False), (True, "current", True),
])
def test_new_decision_requires_both_current_sources(capsule_case, data_current_bool, eod_kind_str, expected_build_bool):
    store_obj, release_obj, _, readiness_dict = capsule_case
    readiness_dict["date"] = pd.Timestamp("2024-01-31" if data_current_bool else "2024-01-30")
    if eod_kind_str != "missing":
        store_obj.upsert_pod_state(_state(release_obj, _time("2024-01-30") if eod_kind_str == "stale" else _time()))
    decision_obj = _decision(capsule_case)
    assert (decision_obj.next_phase_str == "build_decision_plan") is expected_build_bool
    if eod_kind_str in {"missing", "stale"}:
        assert decision_obj.next_phase_str == "eod_snapshot"


def test_first_capture_failure_retries_capture_then_allows_decision(capsule_case):
    store_obj, release_obj, _, _ = capsule_case
    assert _decision(capsule_case).next_phase_str == "eod_snapshot"
    assert _capture(capsule_case, StubBrokerAdapter())["eod_snapshot_count_int"] == 0
    assert store_obj.get_pod_state(release_obj.pod_id_str) is None
    assert _decision(capsule_case).next_phase_str == "eod_snapshot"
    assert _capture(capsule_case, _broker(release_obj))["eod_snapshot_count_int"] == 1
    assert _decision(capsule_case).next_phase_str == "build_decision_plan"
    assert _capture(capsule_case, _broker(release_obj))["eod_snapshot_count_int"] == 0


@pytest.mark.parametrize("date_str,close_hour_int", [("2024-01-31", 16), ("2024-11-29", 13)])
def test_current_heartbeat_cannot_bypass_close_buffer(capsule_case, date_str, close_hour_int):
    store_obj, release_obj, _, readiness_dict = capsule_case
    readiness_dict["date"] = pd.Timestamp(date_str)
    before_ts = _time(date_str, close_hour_int, 9)
    prior_date_ts = scheduler_utils.get_latest_completed_session_label_ts(before_ts, "XNYS")
    store_obj.upsert_pod_state(_state(release_obj, _time(prior_date_ts.date().isoformat())))
    assert _decision(capsule_case, before_ts).next_phase_str != "build_decision_plan"
    after_ts = _time(date_str, close_hour_int, 10)
    assert _decision(capsule_case, after_ts).next_phase_str == "eod_snapshot"
    assert _capture(capsule_case, _broker(release_obj, after_ts), after_ts)["eod_snapshot_count_int"] == 1
    assert _decision(capsule_case, after_ts).next_phase_str == "build_decision_plan"


@pytest.mark.parametrize("change_dict", [
    {"snapshot_source_str": "pod_state"}, {"snapshot_stage_str": "post_execution"},
    {"updated_timestamp_ts": _time(hour_int=15)}, {"updated_timestamp_ts": _time(minute_int=11)},
    {"account_route_str": "DU_OTHER"},
])
def test_invalid_eod_metadata_does_not_allow_new_decision(capsule_case, change_dict):
    store_obj, release_obj, _, _ = capsule_case
    store_obj.upsert_pod_state(_state(release_obj, **change_dict))
    assert _decision(capsule_case).next_phase_str != "build_decision_plan"


@pytest.mark.parametrize("change_dict", [
    {"snapshot_timestamp_ts": _time("2024-01-30")}, {"snapshot_timestamp_ts": _time(hour_int=15)},
    {"snapshot_timestamp_ts": _time("2099-01-01")}, {"account_route_str": "DU_OTHER"},
    {"open_order_id_list": ["working_order"]},
])
def test_capture_does_not_relabel_untrusted_broker_state_as_eod(capsule_case, change_dict):
    store_obj, release_obj, _, _ = capsule_case
    broker_obj = _broker(release_obj)
    broker_obj._snapshot_map[release_obj.account_route_str] = replace(broker_obj.get_account_snapshot(release_obj.account_route_str), **change_dict)
    detail_dict = _capture(capsule_case, broker_obj)
    assert detail_dict["eod_snapshot_count_int"] == 0
    assert detail_dict["reason_count_map_dict"]["mr_capsule_eod_source_untrusted"] == 1
    assert store_obj.get_pod_state(release_obj.pod_id_str) is None


def test_capture_preserves_broker_response_time(capsule_case):
    store_obj, release_obj, _, _ = capsule_case
    response_ts = _time(minute_int=11)
    assert _capture(capsule_case, _broker(release_obj, response_ts))["eod_snapshot_count_int"] == 1
    assert store_obj.get_pod_state(release_obj.pod_id_str).updated_timestamp_ts == response_ts
    assert _decision(capsule_case, _time(minute_int=12)).next_phase_str == "build_decision_plan"


@pytest.mark.parametrize("phase_str,expected_str", [
    ("planned", "build_vplan"), ("ready", "submit_vplan"),
    ("submitting", "post_execution_reconcile"), ("submitted", "post_execution_reconcile"),
    ("expired_ready", "post_execution_reconcile"),
])
def test_saved_cycles_continue_without_new_eod(capsule_case, phase_str, expected_str):
    store_obj, _, _, readiness_dict = capsule_case
    readiness_dict["date"] = None
    if phase_str == "planned":
        _insert_planned_decision_plan(store_obj)
    elif phase_str in {"ready", "expired_ready"}:
        _insert_ready_vplan(store_obj)
    elif phase_str == "submitting":
        _insert_submitting_vplan(store_obj)
    else:
        _insert_submitted_vplan(store_obj)
    as_of_ts = _time("2024-02-01", 9, 35 if phase_str in {"submitting", "submitted", "expired_ready"} else 24)
    assert _decision(capsule_case, as_of_ts).next_phase_str == expected_str


@pytest.mark.parametrize("status_str,abandoned_bool,prior_date_str,expected_build_bool", [
    ("completed", False, "2024-01-31", False), ("blocked", True, "2024-01-31", False),
    ("expired", True, "2024-01-31", False), ("completed", False, "2024-01-30", True),
    ("blocked", True, "2024-01-30", False), ("expired", True, "2024-01-30", False),
    ("blocked", False, "2024-01-30", False), ("expired", False, "2024-01-30", False),
    ("completed_with_exceptions", False, "2024-01-30", True),
    ("completed_with_exceptions", False, "2024-01-31", False),
])
def test_new_cycle_requires_strictly_older_resolved_plan(capsule_case, status_str, abandoned_bool, prior_date_str, expected_build_bool):
    store_obj, release_obj, _, _ = capsule_case
    store_obj.upsert_pod_state(_state(release_obj))
    store_obj.insert_decision_plan(DecisionPlan(
        release_id_str=release_obj.release_id_str, user_id_str=release_obj.user_id_str,
        pod_id_str=release_obj.pod_id_str, account_route_str=release_obj.account_route_str,
        signal_timestamp_ts=_time(prior_date_str, 16, 0), submission_timestamp_ts=_time("2024-02-01", 9, 24),
        target_execution_timestamp_ts=_time("2024-02-01", 9, 30), execution_policy_str="next_open_moo",
        decision_base_position_map={}, strategy_state_dict={},
        status_str=status_str, snapshot_metadata_dict={"mr_capsule_unsubmitted_cycle_abandoned_bool": abandoned_bool},
    ))
    decision_obj = _decision(capsule_case)
    assert (decision_obj.next_phase_str == "build_decision_plan") is expected_build_bool
    if status_str in {"blocked", "expired"}:
        # Old abandonment flags cannot bypass fresh post-close settlement.
        assert decision_obj.next_phase_str == "post_execution_reconcile"
        assert decision_obj.reason_code_str == "waiting_for_post_execution_reconcile"


@pytest.mark.parametrize("legacy_profile_str", ["norgate_eod_ndx_pit_plus_vxn_helper", "norgate_eod_etf_plus_vix_helper"])
def test_monthly_build_keeps_priority_over_waiting_capsule(capsule_case, legacy_profile_str, monkeypatch):
    store_obj, release_obj, tmp_path, _ = capsule_case
    legacy_release_obj = replace(
        release_obj, release_id_str="monthly.v1", pod_id_str="monthly", account_route_str="DU_MONTHLY",
        strategy_import_str="strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled:VxnScaledAtrNormalizedNdxStrategy" if "ndx" in legacy_profile_str else "strategies.taa_df.strategy_taa_df_btal_1n_fallback_tqqq_vix_cash",
        data_profile_str=legacy_profile_str, signal_clock_str="month_end_snapshot_ready", execution_policy_str="next_month_first_open",
    )
    store_obj.upsert_release(legacy_release_obj)
    monkeypatch.setattr(scheduler_service, "_load_release_list_and_sync", lambda *argument_tuple, **keyword_dict: [release_obj, legacy_release_obj])
    decision_obj = _decision(capsule_case)
    assert decision_obj.next_phase_str == "build_decision_plan"
    assert decision_obj.related_pod_id_list == ["monthly"]


@pytest.mark.parametrize("change_dict", [{"enabled_bool": False}, {"mode_str": "live"}])
def test_inactive_capsule_has_no_scheduler_data_calls(capsule_case, change_dict, monkeypatch):
    _, release_obj, _, readiness_dict = capsule_case
    monkeypatch.setattr(scheduler_service, "_load_release_list_and_sync", lambda *argument_tuple, **keyword_dict: [replace(release_obj, **change_dict)])
    decision_obj = _decision(capsule_case)
    assert not decision_obj.due_now_bool
    assert readiness_dict["calls"] == []


@pytest.mark.parametrize("minute_int,expected_build_bool", [(22, True), (24, False)])
def test_next_morning_uses_prior_eod_only_before_build_deadline(capsule_case, minute_int, expected_build_bool):
    store_obj, release_obj, _, _ = capsule_case
    store_obj.upsert_pod_state(_state(release_obj))
    assert (_decision(capsule_case, _time("2024-02-01", 9, minute_int)).next_phase_str == "build_decision_plan") is expected_build_bool


def test_failed_capture_preserves_previous_eod(capsule_case):
    store_obj, release_obj, _, _ = capsule_case
    previous_state_obj = _state(release_obj, _time("2024-01-30"))
    store_obj.upsert_pod_state(previous_state_obj)
    assert _capture(capsule_case, _broker(release_obj, _time("2024-01-30")))["eod_snapshot_count_int"] == 0
    assert store_obj.get_pod_state(release_obj.pod_id_str) == previous_state_obj
    assert _decision(capsule_case).next_phase_str == "eod_snapshot"


def test_run_once_dispatches_capture_before_build(capsule_case, monkeypatch):
    store_obj, release_obj, tmp_path, _ = capsule_case
    tick_call_list = []
    monkeypatch.setattr(scheduler_service, "ensure_norgate_snapshots_for_live_tick", lambda **keyword_dict: {"all_profiles_ready_bool": True})

    def record_tick(**keyword_dict):
        assert keyword_dict["state_store_obj"].get_pod_state(release_obj.pod_id_str).snapshot_stage_str == "eod"
        tick_call_list.append(keyword_dict["as_of_ts"])
        return {}

    monkeypatch.setattr(runner, "tick", record_tick)
    argument_dict = dict(
        state_store_obj=store_obj, broker_adapter_obj=_broker(release_obj), as_of_ts=_time(),
        releases_root_path_str=str(tmp_path / "releases"), env_mode_str="paper",
        log_path_str=str(tmp_path / "events.jsonl"), trace_enabled_bool=False,
    )
    first_dict = scheduler_service.run_once(**argument_dict)
    assert first_dict["next_phase_str"] == "eod_snapshot"
    assert first_dict["tick_detail_dict"]["eod_snapshot_count_int"] == 1
    assert tick_call_list == []
    second_dict = scheduler_service.run_once(**argument_dict)
    assert second_dict["next_phase_str"] == "build_decision_plan"
    assert tick_call_list == [_time()]


@pytest.mark.parametrize("change_dict", [
    {"snapshot_source_str": "pod_state"}, {"account_route_str": "DU_OTHER"},
    {"updated_timestamp_ts": _time(hour_int=15)},
])
def test_existing_untrusted_eod_requests_review_without_overwrite(capsule_case, change_dict):
    store_obj, release_obj, _, _ = capsule_case
    invalid_state_obj = _state(release_obj, **change_dict)
    store_obj.upsert_pod_state(invalid_state_obj)
    decision_obj = _decision(capsule_case)
    assert not decision_obj.due_now_bool
    assert decision_obj.next_phase_str == "manual_review_pending"
    assert decision_obj.reason_code_str == "mr_capsule_eod_snapshot_untrusted"
    assert store_obj.get_pod_state(release_obj.pod_id_str) == invalid_state_obj
