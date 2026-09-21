"""Activity reads bounded saved evidence; it cannot cross a LIVE ownership scope."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from alpha.live.dashboard_v4 import activity_data
from alpha.live.logging_utils import build_structured_event_record_dict


BASE_TS = datetime(2026, 9, 21, 17, 0, tzinfo=timezone.utc)


def _target_obj(pod_str="pod_one", account_str="U111", owner_str="owner_one", **field_dict):
    return SimpleNamespace(release_obj=SimpleNamespace(pod_id_str=pod_str, account_route_str=account_str,
        user_id_str=owner_str, release_id_str=f"release_{pod_str}", mode_str="live", enabled_bool=True,
        **field_dict))


def _provider_obj(tmp_path, *target_list):
    return SimpleNamespace(event_log_path_str=str(tmp_path / "live_events.jsonl"),
        get_target_list=lambda: list(target_list) if target_list else [_target_obj()])


def _record_dict(code_str="submit_vplan_completed", *, event_ts=BASE_TS, **field_dict):
    return build_structured_event_record_dict(code_str, {"pod_id_str": "pod_one", "mode_str": "live",
        "account_route_str": "U111", "user_id_str": "owner_one", "release_id_str": "release_pod_one",
        "decision_plan_id_int": 7, "vplan_id_int": 9, **field_dict}, timestamp_obj=event_ts)


def _write_records(path_obj, *record_list):
    path_obj.write_text("".join(json.dumps(record_dict) + "\n" for record_dict in record_list), encoding="utf-8")


def _load_dict(provider_obj, **argument_dict):
    return activity_data.load_activity_source_dict(provider_obj, as_of_ts=BASE_TS, days_int=7, **argument_dict)


def test_material_event_is_normalized_redacted_and_keeps_exact_cycle_evidence(tmp_path):
    provider_obj = _provider_obj(tmp_path)
    _write_records(Path(provider_obj.event_log_path_str), _record_dict(broker_order_count_int=4,
        fill_count_int=4, reason_code_str="submitted", asset_str="BKR", severity_str="critical",
        error_str="password=secret-password at C:/private/state.db", account_number_str="SECRETACCOUNT",
        token_str="private-token", arbitrary_dict={"private": "anything"},
        broker_snapshot_timestamp_str="2026-09-21T12:55:00-04:00"))
    result_dict = _load_dict(provider_obj)
    assert result_dict["feed_available_bool"] is True
    assert result_dict["warning_list"] == []
    event_dict, = result_dict["event_list"]
    assert event_dict["timestamp_str"] == BASE_TS.isoformat()
    assert event_dict["level_str"] == "CRITICAL"
    assert event_dict["decision_plan_id_int"] == 7 and event_dict["vplan_id_int"] == 9
    assert event_dict["payload_dict"] == {"release_id_str": "release_pod_one", "decision_plan_id_int": 7,
        "vplan_id_int": 9, "broker_order_count_int": 4, "fill_count_int": 4, "reason_code_str": "submitted",
        "asset_str": "BKR", "broker_snapshot_timestamp_str": "2026-09-21T16:55:00+00:00"}
    text_str = json.dumps(result_dict)
    for private_str in ("secret-password", "C:/private", "SECRETACCOUNT", "private-token", "arbitrary", "U111", "owner_one"):
        assert private_str not in text_str


@pytest.mark.parametrize("field_str,value_obj", [
    ("mode_str", "paper"), ("env_mode_str", "incubation"), ("session_mode_str", "sim"),
    ("pod_id_str", "other_pod"), ("pod_str", "other_pod"), ("account_route_str", "U999"),
    ("account_id_str", "U999"), ("account_str", "U999"), ("user_id_str", "other_owner"),
    ("related_pod_id_list", ["other_pod"]), ("release_id_str", "other_release"),
])
def test_conflicting_nested_identity_is_excluded(tmp_path, field_str, value_obj):
    provider_obj = _provider_obj(tmp_path)
    record_dict = _record_dict()
    record_dict["payload_dict"] = {"payload_dict": {**record_dict["payload_dict"], field_str: value_obj}}
    _write_records(Path(provider_obj.event_log_path_str), record_dict)
    assert _load_dict(provider_obj)["event_list"] == []


def test_disabled_and_other_modes_are_filtered_before_identity_or_file_access(tmp_path, monkeypatch):
    ignored_target_list = [SimpleNamespace(release_obj=SimpleNamespace(mode_str=mode_str, enabled_bool=enabled_bool))
        for mode_str, enabled_bool in [("paper", True), ("incubation", True), ("live", False)]]
    provider_obj = _provider_obj(tmp_path, *ignored_target_list)
    monkeypatch.setattr(Path, "open", lambda *argument_list, **argument_dict: pytest.fail("No scoped targets must not open logs"))
    result_dict = _load_dict(provider_obj)
    assert result_dict["event_list"] == [] and result_dict["feed_available_bool"] is False


@pytest.mark.parametrize("target_list", [
    [_target_obj(), _target_obj()],
    [_target_obj(), _target_obj("pod_two", "U111")],
    [_target_obj(), _target_obj("pod_two", "U222", "other_owner")],
    [_target_obj("../bad")],
    [_target_obj(f"pod_{index_int}", f"U{index_int}") for index_int in range(33)],
])
def test_ambiguous_invalid_or_oversized_scope_fails_before_file_access(tmp_path, monkeypatch, target_list):
    provider_obj = _provider_obj(tmp_path, *target_list)
    monkeypatch.setattr(Path, "open", lambda *argument_list, **argument_dict: pytest.fail("Invalid scope must not open logs"))
    result_dict = _load_dict(provider_obj)
    assert result_dict["warning_list"] == ["Activity scope could not be verified."]
    assert result_dict["event_list"] == [] and result_dict["feed_available_bool"] is False


def test_provider_app_fallback_and_scope_hash_include_identity_and_log_source(tmp_path):
    provider_obj = _provider_obj(tmp_path)
    _write_records(Path(provider_obj.event_log_path_str))
    wrapper_obj = SimpleNamespace(app_obj=lambda: provider_obj)
    result_dict = _load_dict(wrapper_obj)
    assert result_dict["feed_available_bool"] is True
    assert result_dict["scope_key_str"] == _load_dict(provider_obj)["scope_key_str"]
    provider_obj.get_target_list = lambda: [_target_obj("pod_two", "U222")]
    assert result_dict["scope_key_str"] != _load_dict(provider_obj)["scope_key_str"]
    provider_obj.get_target_list = lambda: [_target_obj()]
    provider_obj.event_log_path_str = str(tmp_path / "another.jsonl")
    assert result_dict["scope_key_str"] != _load_dict(provider_obj)["scope_key_str"]


def test_global_events_and_multi_pod_events_need_no_unowned_identity(tmp_path):
    provider_obj = _provider_obj(tmp_path, _target_obj(), _target_obj("pod_two", "U222"))
    event_list = [build_structured_event_record_dict("system_started", {}, BASE_TS),
        build_structured_event_record_dict("scheduler_started", {"mode_str": "live"}, BASE_TS),
        build_structured_event_record_dict("system_changed", {"mode_str": "live",
            "related_pod_id_list": ["pod_two", "pod_one"]}, BASE_TS)]
    rejected_payload_list = [{"account_id_str": "U111"}, {"user_id_str": "owner_one"},
        {"release_id_str": "release_pod_one"}, {"pod_id_str": "pod_one"},
        {"related_pod_id_list": ["pod_one"]}, {"mode_str": "paper"},
        {"mode_str": "live", "related_pod_id_list": ["pod_one", "foreign_pod"]}]
    event_list.extend(build_structured_event_record_dict("private_event", payload_dict, BASE_TS) for payload_dict in rejected_payload_list)
    _write_records(Path(provider_obj.event_log_path_str), *event_list)
    result_dict = _load_dict(provider_obj)
    assert {event_dict["event_type_str"] for event_dict in result_dict["event_list"]} == {
        "system_started", "scheduler_started", "system_changed"}
    multi_dict = next(event_dict for event_dict in result_dict["event_list"] if event_dict["event_type_str"] == "system_changed")
    assert multi_dict["pod_id_str"] == ""
    assert multi_dict["payload_dict"]["related_pod_id_list"] == ["pod_one", "pod_two"]


def test_old_release_kept_for_same_owner_but_not_rewritten_as_current_cycle(tmp_path):
    provider_obj = _provider_obj(tmp_path)
    _write_records(Path(provider_obj.event_log_path_str), _record_dict(release_id_str="prior_release"))
    event_dict, = _load_dict(provider_obj)["event_list"]
    assert event_dict["payload_dict"]["release_id_str"] == "prior_release"


def test_occurrence_time_controls_et_day_boundary_and_not_as_of_or_plan_time(tmp_path):
    provider_obj = _provider_obj(tmp_path)
    _write_records(Path(provider_obj.event_log_path_str),
        _record_dict("too_old", event_ts=datetime(2026, 9, 15, 3, 59, 59, tzinfo=timezone.utc)),
        _record_dict("first_day", event_ts=datetime(2026, 9, 15, 4, 0, tzinfo=timezone.utc),
            as_of_timestamp_str="2000-01-01T00:00:00Z", target_execution_timestamp_str="2030-01-01T00:00:00Z"),
        _record_dict("latest", event_ts=BASE_TS))
    assert [event_dict["event_type_str"] for event_dict in _load_dict(provider_obj)["event_list"]] == ["latest", "first_day"]


@pytest.mark.parametrize("change_dict", [
    {"event_timestamp_str": "2026-09-21T17:00:01Z", "ts_utc": "2026-09-21T17:00:01Z"},
    {"event_timestamp_str": "2026-09-21T17:00:00", "ts_utc": None},
    {"ts_utc": "2026-09-21T16:59:00Z"}, {"event_timestamp_str": "not-a-date"},
    {"vplan_id_int": True}, {"decision_plan_id_int": "7"},
    {"vplan_id_int": 10},
])
def test_invalid_future_or_conflicting_cycle_time_cannot_be_evidence(tmp_path, change_dict):
    provider_obj = _provider_obj(tmp_path)
    record_dict = _record_dict()
    record_dict.update(change_dict)
    _write_records(Path(provider_obj.event_log_path_str), record_dict)
    result_dict = _load_dict(provider_obj)
    assert result_dict["event_list"] == [] and result_dict["feed_available_bool"] is False
    assert result_dict["warning_list"]


def test_critical_mirror_deduplicates_identical_events_but_not_distinct_occurrences(tmp_path):
    provider_obj = _provider_obj(tmp_path)
    record_dict = _record_dict("runner_failed", severity_str="critical")
    earlier_dict = _record_dict("runner_failed", severity_str="critical", event_ts=BASE_TS - timedelta(seconds=1))
    _write_records(Path(provider_obj.event_log_path_str), record_dict)
    _write_records(tmp_path / "live_critical_events.jsonl", record_dict, earlier_dict)
    event_list = _load_dict(provider_obj)["event_list"]
    assert len(event_list) == 2
    assert event_list[0]["source_str"] == "Event log"
    assert event_list[1]["timestamp_str"] == earlier_dict["event_timestamp_str"]


def test_journal_records_request_and_never_invents_completion_or_delivery(tmp_path):
    provider_obj = _provider_obj(tmp_path)
    _write_records(Path(provider_obj.event_log_path_str))
    _write_records(tmp_path / "operator_journal.jsonl", {"timestamp_str": BASE_TS.isoformat(), "pod_id_str": "pod_one",
        "mode_str": "live", "action_name_str": "eod_snapshot", "job_id_str": "job_one",
        "initial_status_str": "queued", "actor_str": "private@example.com"})
    event_dict, = _load_dict(provider_obj)["event_list"]
    assert event_dict["event_type_str"] == "operator_action_requested"
    assert event_dict["notification_delivery_str"] == ""
    assert event_dict["payload_dict"]["initial_status_str"] == "queued"
    assert "actor" not in json.dumps(event_dict)


def test_only_explicit_notification_events_can_claim_delivery(tmp_path):
    provider_obj = _provider_obj(tmp_path)
    _write_records(Path(provider_obj.event_log_path_str),
        _record_dict("notification_sent", delivered_bool=True),
        _record_dict("notification_retry", notification_delivery_str="queued"),
        _record_dict("notification_failed", delivered_bool=False),
        _record_dict("notification_unknown", delivery_status_str="unknown"),
        _record_dict("runner_failed", delivered_bool=True))
    (tmp_path / "notification_state.json").write_text('{"pod_severity_map_dict":{"pod_one":"red"}}', encoding="utf-8")
    event_dict_by_code = {event_dict["event_type_str"]: event_dict for event_dict in _load_dict(provider_obj)["event_list"]}
    assert {code_str: event_dict["notification_delivery_str"] for code_str, event_dict in event_dict_by_code.items()} == {
        "notification_sent": "delivered", "notification_retry": "queued", "notification_failed": "failed",
        "notification_unknown": "", "runner_failed": ""}


def test_scheduler_polling_is_quiet_but_failure_severity_is_not_suppressed(tmp_path):
    provider_obj = _provider_obj(tmp_path)
    event_list = [_record_dict(code_str) for code_str in activity_data.QUIET_EVENT_SET]
    failed_dict = _record_dict("scheduler.tick_result")
    failed_dict["payload_dict"]["severity_str"] = "critical"
    _write_records(Path(provider_obj.event_log_path_str), *event_list, failed_dict)
    event_dict, = _load_dict(provider_obj)["event_list"]
    assert event_dict["event_type_str"] == "scheduler.tick_result" and event_dict["level_str"] == "CRITICAL"


def test_missing_main_is_incomplete_optional_absence_is_normal_and_rotation_recovers_rows(tmp_path):
    provider_obj = _provider_obj(tmp_path)
    result_dict = _load_dict(provider_obj)
    assert result_dict["warning_list"] == ["The Activity event log is unavailable."]
    assert result_dict["feed_available_bool"] is False
    _write_records(tmp_path / "live_events.jsonl.10", _record_dict())
    result_dict = _load_dict(provider_obj)
    assert len(result_dict["event_list"]) == 1 and result_dict["feed_available_bool"] is False
    _write_records(Path(provider_obj.event_log_path_str))
    result_dict = _load_dict(provider_obj)
    assert result_dict["warning_list"] == [] and result_dict["feed_available_bool"] is True


@pytest.mark.parametrize("bad_bytes", [b"not json\n", b"[]\n", b"\xff\n", b'{"unfinished":',
    b'{"nested":' + b"[" * 1500 + b"0" + b"]" * 1500 + b"}\n", b"a" * (32 * 1024 + 1) + b"\n"],
    ids=["not-json", "array", "bad-unicode", "partial", "deep-json", "oversized"])
def test_corrupt_partial_oversized_or_deep_records_do_not_hide_valid_rows(tmp_path, bad_bytes):
    provider_obj = _provider_obj(tmp_path)
    path_obj = Path(provider_obj.event_log_path_str)
    _write_records(path_obj, _record_dict())
    path_obj.write_bytes(path_obj.read_bytes() + bad_bytes)
    result_dict = _load_dict(provider_obj)
    assert len(result_dict["event_list"]) == 1 and result_dict["feed_available_bool"] is False
    assert "Some Activity records are incomplete or invalid." in result_dict["warning_list"]


def test_locked_optional_source_is_visible_without_exception_or_private_path(tmp_path, monkeypatch):
    provider_obj = _provider_obj(tmp_path)
    _write_records(Path(provider_obj.event_log_path_str), _record_dict())
    original_open_func = Path.open
    def restricted_open(path_obj, *argument_list, **argument_dict):
        if path_obj.name == "operator_journal.jsonl":
            raise PermissionError("locked C:/private/private-log.jsonl")
        return original_open_func(path_obj, *argument_list, **argument_dict)
    monkeypatch.setattr(Path, "open", restricted_open)
    result_dict = _load_dict(provider_obj)
    assert len(result_dict["event_list"]) == 1 and result_dict["feed_available_bool"] is False
    assert result_dict["warning_list"] == ["An Activity source could not be read safely."]
    assert "private" not in json.dumps(result_dict)


def test_resolved_optional_path_outside_log_root_is_never_read(tmp_path, monkeypatch):
    provider_obj = _provider_obj(tmp_path)
    _write_records(Path(provider_obj.event_log_path_str), _record_dict())
    original_resolve_func = Path.resolve
    def redirected_resolve(path_obj, *argument_list, **argument_dict):
        if path_obj.name == "operator_journal.jsonl":
            return tmp_path.parent / "private_foreign.jsonl"
        return original_resolve_func(path_obj, *argument_list, **argument_dict)
    monkeypatch.setattr(Path, "resolve", redirected_resolve)
    result_dict = _load_dict(provider_obj)
    assert len(result_dict["event_list"]) == 1 and result_dict["feed_available_bool"] is False
    assert result_dict["warning_list"] == ["An Activity source could not be read safely."]


def test_file_tail_limit_keeps_newest_rows_and_flags_older_coverage(tmp_path, monkeypatch):
    provider_obj = _provider_obj(tmp_path)
    _write_records(Path(provider_obj.event_log_path_str), *[_record_dict(f"event_{index_int}",
        event_ts=BASE_TS - timedelta(seconds=9 - index_int)) for index_int in range(10)])
    monkeypatch.setattr(activity_data, "FILE_BYTES_INT", 2500)
    result_dict = _load_dict(provider_obj)
    assert result_dict["event_list"][0]["event_type_str"] == "event_9"
    assert 0 < len(result_dict["event_list"]) < 10
    assert result_dict["feed_available_bool"] is True
    assert result_dict["warning_list"] == ["Activity history reached its scan limit."]


def test_global_byte_budget_and_file_count_are_bounded(tmp_path, monkeypatch):
    provider_obj = _provider_obj(tmp_path)
    record_dict = _record_dict()
    for name_str in ("live_events.jsonl", "live_critical_events.jsonl", "operator_journal.jsonl", "live_events.jsonl.1"):
        _write_records(tmp_path / name_str, record_dict)
    monkeypatch.setattr(activity_data, "TOTAL_BYTES_INT", 2000)
    original_open_func, opened_list = Path.open, []
    def counted_open(path_obj, *argument_list, **argument_dict):
        opened_list.append(path_obj.name)
        return original_open_func(path_obj, *argument_list, **argument_dict)
    monkeypatch.setattr(Path, "open", counted_open)
    result_dict = _load_dict(provider_obj)
    assert len(opened_list) <= 3 and "live_events.jsonl.1" not in opened_list
    assert result_dict["feed_available_bool"] is True
    assert "Activity history reached its scan limit." in result_dict["warning_list"]


def test_line_and_output_caps_keep_newest_matching_rows(tmp_path, monkeypatch):
    provider_obj = _provider_obj(tmp_path)
    _write_records(Path(provider_obj.event_log_path_str), *[_record_dict(f"event_{index_int}",
        event_ts=BASE_TS - timedelta(seconds=5 - index_int)) for index_int in range(6)])
    monkeypatch.setattr(activity_data, "LINE_LIMIT_INT", 4)
    monkeypatch.setattr(activity_data, "EVENT_LIMIT_INT", 2)
    result_dict = _load_dict(provider_obj)
    assert [event_dict["event_type_str"] for event_dict in result_dict["event_list"]] == ["event_5", "event_4"]
    assert result_dict["feed_available_bool"] is True
    assert len(result_dict["warning_list"]) == 2


def test_truncated_window_without_any_complete_record_cannot_mark_feed_available(tmp_path, monkeypatch):
    provider_obj = _provider_obj(tmp_path)
    _write_records(Path(provider_obj.event_log_path_str), _record_dict())
    monkeypatch.setattr(activity_data, "FILE_BYTES_INT", 10)
    result_dict = _load_dict(provider_obj)
    assert result_dict["event_list"] == [] and result_dict["feed_available_bool"] is False
    assert "Some Activity records are incomplete or invalid." in result_dict["warning_list"]


def test_numeric_and_date_payload_fields_cannot_turn_invalid_into_zero(tmp_path):
    provider_obj = _provider_obj(tmp_path)
    _write_records(Path(provider_obj.event_log_path_str), _record_dict(fill_count_int=True,
        broker_order_count_int=-1, ack_coverage_ratio_float=float("nan"), residual_share_float=float("inf"),
        eod_market_date_str="2026-02-30", reason_code_str="secret with arbitrary text"))
    payload_dict = _load_dict(provider_obj)["event_list"][0]["payload_dict"]
    assert set(payload_dict) == {"release_id_str", "decision_plan_id_int", "vplan_id_int"}


@pytest.mark.parametrize("days_int", [0, 366, True, 7.0])
def test_input_range_is_bounded(tmp_path, days_int):
    with pytest.raises(ValueError):
        activity_data.load_activity_source_dict(_provider_obj(tmp_path), as_of_ts=BASE_TS, days_int=days_int)


def test_clock_must_be_aware(tmp_path):
    with pytest.raises(ValueError):
        activity_data.load_activity_source_dict(_provider_obj(tmp_path), as_of_ts=BASE_TS.replace(tzinfo=None), days_int=7)
