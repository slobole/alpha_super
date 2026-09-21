"""Activity reads bounded saved evidence; it cannot cross a LIVE ownership scope."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from alpha.live.dashboard_v4 import activity_data
from alpha.live.logging_utils import build_structured_event_record_dict


BASE_TS = datetime(2026, 9, 21, 17, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def clear_activity_cache():
    activity_data.FILE_CACHE_DICT.clear()
    yield
    activity_data.FILE_CACHE_DICT.clear()


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


@pytest.mark.parametrize("field_dict", [{"status_str": "failed"}, {"vplan_status_str": "blocked"},
    {"submit_ack_status_str": "missing_critical"}, {"status_str": "late"}, {"missing_ack_count_int": 1}])
def test_info_polling_with_explicit_problem_status_remains_material(tmp_path, field_dict):
    provider_obj = _provider_obj(tmp_path)
    _write_records(Path(provider_obj.event_log_path_str), _record_dict("scheduler.tick_result", **field_dict))
    event_dict, = _load_dict(provider_obj)["event_list"]
    assert event_dict["event_type_str"] == "scheduler.tick_result"
    assert all(event_dict["payload_dict"][field_str] == value_obj for field_str, value_obj in field_dict.items())


@pytest.mark.parametrize("code_str", ["new_failure_code", "scheduler.tick_result"])
def test_fatal_is_critical_and_never_quiet_or_neutral(tmp_path, code_str):
    provider_obj = _provider_obj(tmp_path)
    _write_records(Path(provider_obj.event_log_path_str), _record_dict(code_str, level_str="FATAL"))
    event_dict, = _load_dict(provider_obj)["event_list"]
    assert event_dict["event_type_str"] == code_str and event_dict["level_str"] == "CRITICAL"


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
    assert "rotation_gap" in result_dict["coverage_dict"]["reason_list"]
    assert result_dict["feed_available_bool"] is False


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
    _write_records(tmp_path / "operator_journal.jsonl")
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


def test_byte_limit_keeps_newest_rows_and_flags_older_coverage(tmp_path, monkeypatch):
    provider_obj = _provider_obj(tmp_path)
    _write_records(Path(provider_obj.event_log_path_str), *[_record_dict(f"event_{index_int}",
        event_ts=BASE_TS - timedelta(seconds=9 - index_int)) for index_int in range(10)])
    monkeypatch.setattr(activity_data, "TOTAL_BYTES_INT", 2500)
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
    monkeypatch.setattr(activity_data, "TOTAL_BYTES_INT", 10)
    result_dict = _load_dict(provider_obj)
    assert result_dict["event_list"] == [] and result_dict["feed_available_bool"] is False
    assert "byte_limit" in result_dict["coverage_dict"]["reason_list"]


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


def test_realistic_quiet_history_does_not_hide_earlier_failure_or_consume_material_budget(tmp_path, monkeypatch):
    provider_obj = _provider_obj(tmp_path)
    path_obj = Path(provider_obj.event_log_path_str)
    failure_ts = BASE_TS - timedelta(days=4)
    with path_obj.open("w", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(_record_dict("runner_failed", event_ts=failure_ts, severity_str="critical")) + "\n")
        for index_int in range(4000):
            file_obj.write(json.dumps(_record_dict("scheduler_sleeping", event_ts=BASE_TS - timedelta(minutes=3999 - index_int),
                diagnostic_str="x" * 5000)) + "\n")
    assert path_obj.stat().st_size > 40 * 1024 * 1024
    monkeypatch.setattr(activity_data, "MATERIAL_LIMIT_INT", 2)
    monkeypatch.setattr(activity_data, "SCAN_SECONDS_FLOAT", 20.0)
    result_dict = _load_dict(provider_obj)
    event_dict, = result_dict["event_list"]
    assert event_dict["event_type_str"] == "runner_failed"
    assert event_dict["timestamp_str"] == failure_ts.isoformat()
    assert result_dict["coverage_dict"]["read_bytes_int"] == path_obj.stat().st_size
    assert result_dict["feed_available_bool"] is True
    assert result_dict["warning_list"] == []


def test_seven_and_ninety_days_expand_actual_history_and_boundary_is_et_midnight(tmp_path):
    provider_obj = _provider_obj(tmp_path)
    _write_records(Path(provider_obj.event_log_path_str),
        _record_dict("oldest", event_ts=BASE_TS - timedelta(days=91)),
        _record_dict("sixty_days", event_ts=BASE_TS - timedelta(days=60)),
        _record_dict("twenty_days", event_ts=BASE_TS - timedelta(days=20)),
        _record_dict("recent", event_ts=BASE_TS - timedelta(days=2)))
    seven_dict = _load_dict(provider_obj)
    ninety_dict = activity_data.load_activity_source_dict(provider_obj, as_of_ts=BASE_TS, days_int=90)
    assert [event_dict["event_type_str"] for event_dict in seven_dict["event_list"]] == ["recent"]
    assert [event_dict["event_type_str"] for event_dict in ninety_dict["event_list"]] == ["recent", "twenty_days", "sixty_days"]
    assert seven_dict["coverage_dict"]["requested_from_timestamp_str"] == "2026-09-15T00:00:00-04:00"
    assert seven_dict["coverage_dict"]["complete_bool"] is True
    assert ninety_dict["coverage_dict"]["complete_bool"] is True


def test_boundary_stops_older_rotations_only_after_requested_history_is_read(tmp_path, monkeypatch):
    provider_obj = _provider_obj(tmp_path)
    for rotation_int in range(11):
        path_obj = tmp_path / ("live_events.jsonl" + (f".{rotation_int}" if rotation_int else ""))
        _write_records(path_obj, _record_dict(f"saved_{rotation_int}", event_ts=BASE_TS - timedelta(days=rotation_int * 8)))
    older_calls_list = []
    original_open_func = Path.open
    def counted_open(path_obj, *argument_list, **argument_dict):
        older_calls_list.append(path_obj.name)
        return original_open_func(path_obj, *argument_list, **argument_dict)
    monkeypatch.setattr(Path, "open", counted_open)
    seven_dict = _load_dict(provider_obj)
    assert "live_events.jsonl.2" not in older_calls_list
    assert seven_dict["coverage_dict"]["complete_bool"] is True
    older_calls_list.clear()
    ninety_dict = activity_data.load_activity_source_dict(provider_obj, as_of_ts=BASE_TS, days_int=90)
    assert "live_events.jsonl.10" in older_calls_list
    assert len(ninety_dict["event_list"]) == 11
    assert ninety_dict["coverage_dict"]["complete_bool"] is False
    assert ninety_dict["coverage_dict"]["reason_list"] == ["retained_history"]


def test_older_critical_log_cannot_extend_main_contiguous_coverage(tmp_path, monkeypatch):
    provider_obj = _provider_obj(tmp_path)
    _write_records(Path(provider_obj.event_log_path_str), _record_dict())
    older_ts = BASE_TS - timedelta(days=50)
    _write_records(tmp_path / "live_critical_events.jsonl", _record_dict("runner_failed", event_ts=older_ts))
    _write_records(tmp_path / "live_events.jsonl.2", _record_dict("old_rotation", event_ts=older_ts))
    result_dict = activity_data.load_activity_source_dict(provider_obj, as_of_ts=BASE_TS, days_int=90)
    assert result_dict["coverage_dict"]["scanned_from_timestamp_str"] == BASE_TS.isoformat()
    assert "rotation_gap" in result_dict["coverage_dict"]["reason_list"]
    assert result_dict["coverage_dict"]["complete_bool"] is False


def test_unchanged_cache_avoids_rescan_but_append_replace_clock_and_scope_invalidate(tmp_path):
    provider_obj = _provider_obj(tmp_path)
    path_obj = Path(provider_obj.event_log_path_str)
    _write_records(path_obj, _record_dict("first", event_ts=BASE_TS - timedelta(minutes=1)))
    first_dict = _load_dict(provider_obj)
    cached_dict = _load_dict(provider_obj)
    assert cached_dict["event_list"] == first_dict["event_list"]
    assert cached_dict["coverage_dict"]["read_bytes_int"] == 0
    assert cached_dict["coverage_dict"]["cache_hit_count_int"] == 1
    with path_obj.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(_record_dict("second")) + "\n")
    appended_dict = _load_dict(provider_obj)
    assert len(appended_dict["event_list"]) == 2 and appended_dict["coverage_dict"]["read_bytes_int"] > 0
    _write_records(path_obj, _record_dict("replacement"))
    assert _load_dict(provider_obj)["event_list"][0]["event_type_str"] == "replacement"
    backwards_dict = activity_data.load_activity_source_dict(provider_obj, as_of_ts=BASE_TS - timedelta(seconds=1), days_int=7)
    assert backwards_dict["event_list"] == [] and backwards_dict["coverage_dict"]["read_bytes_int"] > 0
    provider_obj.get_target_list = lambda: [_target_obj("pod_two", "U222")]
    changed_scope_dict = _load_dict(provider_obj)
    assert changed_scope_dict["event_list"] == [] and changed_scope_dict["coverage_dict"]["read_bytes_int"] > 0


def test_future_record_is_not_frozen_out_by_cache_when_clock_reaches_it(tmp_path):
    provider_obj = _provider_obj(tmp_path)
    future_ts = BASE_TS + timedelta(seconds=30)
    _write_records(Path(provider_obj.event_log_path_str), _record_dict(event_ts=future_ts))
    result_dict = _load_dict(provider_obj)
    assert result_dict["event_list"] == [] and result_dict["feed_available_bool"] is False
    result_dict = activity_data.load_activity_source_dict(provider_obj, as_of_ts=future_ts, days_int=7)
    assert len(result_dict["event_list"]) == 1
    assert result_dict["coverage_dict"]["cache_hit_count_int"] == 0


def test_oversized_line_across_many_chunks_is_skipped_without_losing_older_failure(tmp_path, monkeypatch):
    provider_obj = _provider_obj(tmp_path)
    path_obj = Path(provider_obj.event_log_path_str)
    _write_records(path_obj, _record_dict("runner_failed", event_ts=BASE_TS - timedelta(minutes=1)))
    with path_obj.open("ab") as file_obj:
        file_obj.write(b"z" * 200000 + b"\n")
        file_obj.write(json.dumps(_record_dict("scheduler_sleeping")).encode() + b"\n")
    monkeypatch.setattr(activity_data, "CHUNK_BYTES_INT", 4096)
    result_dict = _load_dict(provider_obj)
    assert [event_dict["event_type_str"] for event_dict in result_dict["event_list"]] == ["runner_failed"]
    assert result_dict["feed_available_bool"] is False
    assert "invalid_records" in result_dict["coverage_dict"]["reason_list"]


def test_hard_deadline_does_not_claim_full_range(tmp_path, monkeypatch):
    provider_obj = _provider_obj(tmp_path)
    _write_records(Path(provider_obj.event_log_path_str), _record_dict())
    monkeypatch.setattr(activity_data, "SCAN_SECONDS_FLOAT", 0.0)
    result_dict = _load_dict(provider_obj)
    assert result_dict["event_list"] == [] and result_dict["feed_available_bool"] is False
    assert result_dict["coverage_dict"]["reason_list"] == ["time_limit"]


def test_out_of_order_old_line_does_not_hide_newer_record_in_same_chunk(tmp_path):
    provider_obj = _provider_obj(tmp_path)
    _write_records(Path(provider_obj.event_log_path_str), _record_dict("runner_failed"),
        _record_dict("scheduler_sleeping", event_ts=BASE_TS - timedelta(days=100)))
    result_dict = _load_dict(provider_obj)
    assert len(result_dict["event_list"]) == 1
    assert "unordered_records" in result_dict["coverage_dict"]["reason_list"]
    assert result_dict["coverage_dict"]["complete_bool"] is False


@pytest.mark.parametrize("status_str,reason_str,quiet_bool", [
    ("ready", "local_snapshot_ready", True), ("ready", "no_enabled_releases", True),
    ("direct", "direct_norgate_mode", True), ("waiting", "sync_failure_cooldown", False),
    ("waiting", "sync_waiting_for_newer_snapshot", False), ("waiting", "sync_lock_busy", False),
    ("local_snapshot_only", "api_config_missing", False), ("failed", "local_snapshot_ready", False),
])
def test_sync_skips_are_quiet_only_for_verified_benign_reason_status_pairs(tmp_path, status_str, reason_str, quiet_bool):
    provider_obj = _provider_obj(tmp_path)
    _write_records(Path(provider_obj.event_log_path_str), _record_dict("norgate_snapshot_sync_skipped",
        status_str=status_str, reason_code_str=reason_str))
    assert bool(_load_dict(provider_obj)["event_list"]) is not quiet_bool


@pytest.mark.parametrize("release_list,expected_int", [(["release_pod_one"], 1), (["paper_release"], 0),
    (["prior_release"], 0), (["release_pod_one", "extra_release"], 0)])
def test_mode_less_sync_requires_exact_current_live_release_list(tmp_path, release_list, expected_int):
    provider_obj = _provider_obj(tmp_path)
    _write_records(Path(provider_obj.event_log_path_str), build_structured_event_record_dict("norgate_snapshot_sync_skipped",
        {"pod_id_list": ["pod_one"], "release_id_list": release_list, "status_str": "waiting",
         "reason_code_str": "sync_failure_cooldown"}, timestamp_obj=BASE_TS))
    event_list = _load_dict(provider_obj)["event_list"]
    assert len(event_list) == expected_int
    if event_list:
        assert event_list[0]["mode_str"] == "live"
        assert event_list[0]["payload_dict"]["reason_code_str"] == "sync_failure_cooldown"


def test_mode_less_sync_cannot_use_current_release_list_to_hide_conflicting_scalar_release(tmp_path):
    provider_obj = _provider_obj(tmp_path)
    _write_records(Path(provider_obj.event_log_path_str), build_structured_event_record_dict("norgate_snapshot_sync_failed",
        {"pod_id_list": ["pod_one"], "release_id_list": ["release_pod_one"], "release_id_str": "paper_release",
         "status_str": "failed"}, timestamp_obj=BASE_TS))
    assert _load_dict(provider_obj)["event_list"] == []


def test_manual_ticket_uses_actual_safe_payload_fields_and_hides_operator_reason(tmp_path):
    provider_obj = _provider_obj(tmp_path)
    _write_records(Path(provider_obj.event_log_path_str), _record_dict("manual_order_submit_requested",
        source_str="manual_broker_ticket", ticket_id_str="manual_20260921", severity_str="warning",
        operator_id_str="private-person", asset_str="BKR", side_str="BUY", broker_order_type_str="LMT",
        quantity_int=19, limit_price_float=42.5, time_in_force_str="DAY", reason_str="private freeform context"))
    event_dict, = _load_dict(provider_obj)["event_list"]
    payload_dict = event_dict["payload_dict"]
    assert {field_str: payload_dict[field_str] for field_str in ("ticket_id_str", "asset_str", "side_str", "broker_order_type_str", "quantity_int")} == {
        "ticket_id_str": "manual_20260921", "asset_str": "BKR", "side_str": "BUY", "broker_order_type_str": "LMT", "quantity_int": 19}
    assert "private" not in json.dumps(event_dict)
