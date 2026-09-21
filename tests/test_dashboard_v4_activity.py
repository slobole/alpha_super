from copy import deepcopy
from datetime import UTC, datetime

from alpha.live.dashboard_v4.activity import build_activity_page_dict


NOW_TS = datetime(2026, 9, 8, 15, tzinfo=UTC)
OVERVIEW_DICT = {"pod_list": [{"pod_id_str": "own", "name_str": "Our Pod"}]}


def _event_dict(code_str="build_vplan_created", **override_dict):
    return {"timestamp_str": "2026-09-08T13:20:00+00:00", "event_type_str": code_str,
        "pod_id_str": "own", "mode_str": "live", "level_str": "info", "decision_plan_id_int": 2, "vplan_id_int": 3,
        "payload_dict": {"release_id_str": "release-1"}, "source_str": "Events", **override_dict}


def _group_dict():
    return {"id_str": "group-1", "timestamp_str": "2026-09-08T13:35:00+00:00", "pod_id_str": "own",
        "release_id_str": "release-1", "decision_plan_id_int": 2, "vplan_id_int": 3,
        "child_list": [{"title_str": "Positions checked."}], "state_str": "done"}


def _page_dict(event_list, row_list=None):
    return build_activity_page_dict(OVERVIEW_DICT, {"event_list": event_list, "scope_key_str": "test"},
        {"row_list": row_list or [], "warning_list": []}, as_of_ts=NOW_TS, days_int=7)


def test_only_exact_healthy_cycle_records_fold():
    original_dict = _event_dict()
    saved_dict = deepcopy(original_dict)
    group_dict = _group_dict()
    result_dict = _page_dict([original_dict], [group_dict])
    assert result_dict["row_list"] == [group_dict]
    assert original_dict == saved_dict
    for field_str, value_obj in (("vplan_id_int", 9), ("decision_plan_id_int", 8)):
        assert len(_page_dict([_event_dict(**{field_str: value_obj})], [group_dict])["row_list"]) == 2
    assert len(_page_dict([_event_dict(payload_dict={"release_id_str": "other"})], [group_dict])["row_list"]) == 2
    assert len(_page_dict([_event_dict(payload_dict={})], [group_dict])["row_list"]) == 2


def test_failure_and_warning_keep_cycle_steps_unfolded():
    for code_str, level_str in (("submit_vplan_missing_broker_ack", "critical"), ("decision_plan_expired", "info"),
            ("build_vplan_position_warning", "warning")):
        result_dict = _page_dict([_event_dict(), _event_dict(code_str, level_str=level_str)], [_group_dict()])
        assert len(result_dict["row_list"]) == 2
        assert not any(row_dict["child_list"] for row_dict in result_dict["row_list"])
        assert any(row_dict["code_str"] == code_str for row_dict in result_dict["row_list"])


def test_operator_request_stays_separate_and_is_not_completion():
    result_dict = _page_dict([_event_dict("operator_action_requested", payload_dict={"release_id_str": "release-1", "action_name_str": "tick", "status_str": "requested"})], [_group_dict()])
    request_dict = next(row_dict for row_dict in result_dict["row_list"] if not row_dict["child_list"])
    assert request_dict["type_str"] == "operator"
    assert request_dict["state_str"] == "now"
    assert request_dict["title_str"] == "Trading check requested."


def test_display_withholds_raw_errors_accounts_and_paths_and_keeps_evidence_link():
    result_dict = _page_dict([_event_dict("submit_vplan_missing_broker_ack", payload_dict={"release_id_str": "release-1", "error_str": "private-token-value", "account_route_str": "U_PRIVATE", "path_str": "C:/secret", "order_count_int": 3})], [_group_dict()])
    row_dict = result_dict["row_list"][0]
    assert "private-token" not in str(result_dict) and "U_PRIVATE" not in str(result_dict) and "C:/secret" not in str(result_dict)
    assert "cycle=vplan%3A3" in row_dict["evidence_url_str"] and "tab=orders" in row_dict["evidence_url_str"]
    assert row_dict["detail_str"] == "3 orders"


def test_reused_or_unverified_numeric_cycle_id_cannot_link_to_different_evidence():
    for group_list in ([], [{**_group_dict(), "release_id_str": "replacement-release"}],
            [{**_group_dict(), "decision_plan_id_int": 99}]):
        result_dict = _page_dict([_event_dict("submit_vplan_missing_broker_ack")], group_list)
        row_dict = next(row_dict for row_dict in result_dict["row_list"] if not row_dict["child_list"])
        assert row_dict["evidence_url_str"] == ""
        assert row_dict["evidence_list"]


def test_no_future_foreign_or_out_of_period_rows_and_newest_first():
    result_dict = _page_dict([_event_dict(timestamp_str="2026-09-09T10:00:00Z"), _event_dict(pod_id_str="foreign"),
        _event_dict(timestamp_str="2026-08-01T10:00:00Z"), _event_dict(timestamp_str="invalid"),
        _event_dict(timestamp_str="2026-09-08T13:23:00Z"), _event_dict(timestamp_str="2026-09-08T13:20:00Z")])
    assert [row_dict["time_str"] for row_dict in result_dict["row_list"]] == ["09:23:00", "09:20:00"]


def test_unknown_event_is_not_reported_as_success():
    row_dict = _page_dict([_event_dict("new_future_code")])["row_list"][0]
    assert row_dict["title_str"] == "Other events (1)."
    assert row_dict["state_str"] == "now"
    assert row_dict["child_list"][0]["state_str"] == "unk"
    assert row_dict["child_list"][0]["code_str"] == "new_future_code"


def test_unmapped_routine_groups_preserve_pod_day_and_child_evidence():
    event_list = [_event_dict("new_one"), _event_dict("new_two", timestamp_str="2026-09-08T14:00:00Z"),
        _event_dict("new_one", pod_id_str=""), _event_dict("new_one", timestamp_str="2026-09-07T14:00:00Z")]
    saved_list = deepcopy(event_list)
    row_list = _page_dict(event_list)["row_list"]
    assert len(row_list) == 3 and sum(len(row_dict["child_list"]) for row_dict in row_list) == 4
    group_dict = next(row_dict for row_dict in row_list if len(row_dict["child_list"]) == 2)
    assert group_dict["pod_id_str"] == "own" and group_dict["day_str"] == "2026-09-08"
    assert group_dict["child_label_str"] == "events"
    assert all(row_dict["evidence_list"] for row_dict in group_dict["child_list"])
    assert event_list == saved_list


def test_unknown_warnings_failures_and_alert_operator_families_stay_individual():
    for code_str, level_str, payload_dict in (("future_warning", "warning", {}), ("future_error", "error", {}),
            ("future_late", "info", {"status_str": "late"}), ("notification_new_code", "info", {}),
            ("operator_new_code", "info", {}), ("manual_order_new_code", "info", {})):
        row_dict = _page_dict([_event_dict(code_str, level_str=level_str, payload_dict=payload_dict)])["row_list"][0]
        assert not row_dict["child_list"] and row_dict["code_str"] == code_str
        assert row_dict["state_str"] != "done" and row_dict["type_str"] in {"alerts", "operator"}


def test_manual_order_messages_are_operator_records_not_fill_claims():
    for code_str, expected_str in (("manual_order_submit_requested", "Manual order requested."),
            ("manual_order_submit_completed", "Manual order submission recorded."),
            ("manual_order_submit_failed", "Manual order submission failed.")):
        row_dict = _page_dict([_event_dict(code_str, level_str="warning", payload_dict={
            "asset_str": "GIS", "side_str": "SELL", "quantity_int": 2, "ticket_id_str": "ticket-1",
            "broker_order_type_str": "MKT", "operator_id_str": "private-operator", "reason_str": "private free text"})])["row_list"][0]
        assert row_dict["title_str"] == expected_str and row_dict["type_str"] == "operator"
        assert row_dict["state_str"] == ("fail" if code_str.endswith("failed") else "now")
        assert "private" not in str(row_dict) and "GIS" in str(row_dict["evidence_list"])
    row_dict = _page_dict([_event_dict("manual_order_submit_completed", payload_dict={"submit_ack_status_str": "missing_critical"})])["row_list"][0]
    assert row_dict["state_str"] == "fail"


def test_failed_sync_cooldown_is_not_presented_as_routine_skip():
    row_dict = _page_dict([_event_dict("norgate_snapshot_sync_skipped", pod_id_str="",
        payload_dict={"status_str": "waiting", "reason_code_str": "sync_failure_cooldown"})])["row_list"][0]
    assert row_dict["state_str"] == "fail" and row_dict["title_str"] == "Data sync waiting after a failure."


def test_partial_log_history_exposes_actual_scanned_et_span():
    page_dict = build_activity_page_dict(OVERVIEW_DICT, {"coverage_dict": {"complete_bool": False,
        "scanned_from_timestamp_str": "2026-09-08T13:38:00+00:00"}}, {}, as_of_ts=NOW_TS, days_int=90)
    assert page_dict["coverage_label_str"] == "Partial log history · scanned to 09-08 09:38 ET"
    assert page_dict["history_complete_bool"] is False


def test_global_events_have_inline_evidence_without_dead_system_link():
    row_dict = _page_dict([_event_dict("scheduler_started", pod_id_str="", decision_plan_id_int=None, vplan_id_int=None)])["row_list"][0]
    assert row_dict["type_str"] == "system" and row_dict["pod_name_str"] == "System"
    assert row_dict["evidence_url_str"] == "" and row_dict["evidence_list"]
