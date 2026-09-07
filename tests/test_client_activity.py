from datetime import UTC, datetime
import json

import pytest

from alpha.live.dashboard_v3.app import create_app
from alpha.live.dashboard_v3.client_operations import load_client_activity_dict, safe_client_event_list
from alpha.live.dashboard_v3.demo import DemoOperationsProvider, build_demo_fixture_tuple
from test_dashboard_operator_access import ForbiddenProvider


AS_OF_TS = datetime(2026, 9, 5, 12, tzinfo=UTC)


def client_dict():
    return build_demo_fixture_tuple()[0]["clients"][0]


def event_dict(**override_dict):
    return dict({"pod_id_str": "demo_0_0", "account_id_str": "DEMO_0_0", "mode_str": "live",
        "event_timestamp_str": "2026-07-01T03:59:00Z", "event_name_str": "post_execution_reconcile_completed",
        "level_str": "info", "status_str": "completed", "reason_code_str": "positions_match"}, **override_dict)


def project_list(raw_list):
    return safe_client_event_list(raw_list, {"pod_id_str": "demo_0_0", "account_route_str": "DEMO_0_0",
        "display_name_str": "Retired allocation", "effective_from_str": "2026-06-01", "effective_to_str": "2026-06-30"},
        from_date_str="2026-06-01", to_date_str="2026-09-04", as_of_ts=AS_OF_TS)


@pytest.mark.parametrize("override_dict", [
    {"mode_str": "paper"}, {"mode_str": None}, {"env_mode_str": "paper"}, {"session_mode_str": "paper"},
    {"account_route_str": "OTHER"}, {"account_str": "OTHER"}, {"account_id_str": None},
    {"pod_str": "OTHER"}, {"pod_id_str_list": ["OTHER"]}, {"related_pod_id_list": ["OTHER"]},
    {"event_timestamp_str": "2026-07-01T04:00:00Z"}, {"event_timestamp_str": "2026-06-01T03:59:00Z"},
    {"event_timestamp_str": "2026-06-30T12:00:00"}, {"event_timestamp_str": "invalid"},
    {"event_timestamp_str": "2026-09-05T13:00:00Z"}, {"ts_utc": "2026-06-01T12:00:00Z"},
    {"ts_utc": "2026-07-01T03:59:00"}, {"account_route_str": ["DEMO_0_0"]},
])
def test_ambiguous_unowned_wrong_mode_or_future_events_are_omitted(override_dict):
    assert project_list([event_dict(**override_dict)]) == []


def test_canonical_time_account_and_both_inclusive_et_boundaries():
    result_list = project_list([
        event_dict(timestamp_str="2026-08-01T00:00:00Z", ts_utc="2026-06-30T23:59:00-04:00", message_str="token=PRIVATE"),
        event_dict(event_timestamp_str="2026-06-01T04:00:00Z"),
    ])
    assert len(result_list) == 2
    assert result_list[0]["timestamp_str"] == "2026-07-01T03:59:00+00:00"
    assert result_list[0]["market_timestamp_str"] == "2026-06-30 23:59:00 EDT"
    assert result_list[0]["label_str"] == "Reconciliation recorded"
    assert result_list[0]["reason_code_str"] == "positions_match"
    assert "PRIVATE" not in json.dumps(result_list)
    assert "account_id_str" not in json.dumps(result_list)


def test_retired_reassigned_pod_reads_once_without_current_release_or_health():
    config_dict = client_dict()
    config_dict["accounts"] = [dict(config_dict["accounts"][0], effective_to="2026-06-30"),
        dict(config_dict["accounts"][0], account_route="LATER", display_name="Later mandate", effective_from="2026-07-01", effective_to="2026-08-31")]

    class HistoryProvider(ForbiddenProvider):
        call_list = []

        def get_pod_event_dict_list(self, pod_id_str, limit_int):
            self.call_list.append((pod_id_str, limit_int))
            return [event_dict(), event_dict(account_id_str="LATER", event_timestamp_str="2026-07-01T04:00:00Z"),
                    event_dict(event_timestamp_str="2026-07-01T04:00:00Z")]

    provider_obj = HistoryProvider()
    result_dict = load_client_activity_dict(config_dict, provider_obj, from_date_str="2026-06-01", to_date_str="2026-08-31", as_of_ts=AS_OF_TS)
    assert provider_obj.call_list == [("demo_0_0", 500)]
    assert [row_dict["display_name_str"] for row_dict in result_dict["event_list"]] == ["Later mandate", "Tactical allocation"]
    assert result_dict["coverage_str"] == "partial"


def snapshot_client_dict(tmp_path, event_list, **summary_override_dict):
    config_dict = client_dict()
    config_dict["accounts"] = [dict(config_dict["accounts"][0], effective_to="2026-06-30")]
    summary_dict = dict({"as_of_timestamp_str": AS_OF_TS.isoformat(), "pod_row_dict_list": [], "event_dict_list": event_list}, **summary_override_dict)
    path_obj = tmp_path / "operations.json"
    path_obj.write_text(json.dumps({"client_id_str": config_dict["client_id"], "schema_version_int": 1, "summary_dict": summary_dict}), encoding="utf-8")
    config_dict.update(operations_source="snapshot", operations_snapshot_path=str(path_obj))
    return config_dict


def test_remote_history_retired_only_client_works_without_local_provider(tmp_path):
    config_dict = snapshot_client_dict(tmp_path, [event_dict(), event_dict(account_id_str="OTHER", message_str="OTHER_CLIENT_PRIVATE")])
    app_obj = create_app(ForbiddenProvider(), read_only_bool=True,
        client_registry_dict={"schema_version": 1, "clients": [config_dict]},
        client_reporting_snapshot_fn=lambda client_id_str: pytest.fail("Activity must not read finances"))
    response_obj = app_obj.test_client().get("/clients/demo-owner/activity?from=2026-06-01&to=2026-06-30")
    html_str = response_obj.get_data(as_text=True)
    assert response_obj.status_code == 200
    assert "Reconciliation recorded" in html_str and "Tactical allocation" in html_str
    assert "OTHER_CLIENT_PRIVATE" not in html_str
    assert "2026-06-01 → 2026-06-30" in html_str
    assert "including retired history" not in html_str  # Retired events remain; redundant prose does not.
    assert "partial" in html_str


@pytest.mark.parametrize("summary_override_dict", [{"event_dict_list": None}, {"event_dict_list": {}},
    {"as_of_timestamp_str": "2026-09-05T12:00:00"}, {"as_of_timestamp_str": "2026-09-06T12:00:00Z"}])
def test_invalid_snapshot_events_never_fall_back_to_local(tmp_path, summary_override_dict):
    config_dict = snapshot_client_dict(tmp_path, [event_dict()], **summary_override_dict)
    result_dict = load_client_activity_dict(config_dict, ForbiddenProvider(), from_date_str="2026-06-01", to_date_str="2026-09-04", as_of_ts=AS_OF_TS)
    assert result_dict["event_list"] == []
    assert result_dict["coverage_str"] == "unavailable"
    assert result_dict["issue_list"]


def test_snapshot_event_cannot_postdate_export(tmp_path):
    config_dict = snapshot_client_dict(tmp_path, [event_dict()], as_of_timestamp_str="2026-06-30T12:00:00Z")
    assert load_client_activity_dict(config_dict, ForbiddenProvider(), from_date_str="2026-06-01", to_date_str="2026-09-04", as_of_ts=AS_OF_TS)["event_list"] == []


def test_snapshot_wrong_client_does_not_expose_events(tmp_path):
    config_dict = snapshot_client_dict(tmp_path, [event_dict(message_str="PRIVATE_EVENT")])
    config_dict["client_id"] = "different-client"
    result_dict = load_client_activity_dict(config_dict, ForbiddenProvider(), from_date_str="2026-06-01", to_date_str="2026-09-04", as_of_ts=AS_OF_TS)
    assert result_dict["coverage_str"] == "unavailable"
    assert "PRIVATE_EVENT" not in json.dumps(result_dict)


def test_unconfigured_or_no_ownership_never_reads_logs():
    config_dict = client_dict()
    config_dict["operations_source"] = "unconfigured"
    for from_str, to_str in [("2026-06-01", "2026-09-04"), ("2026-01-01", "2026-01-02")]:
        result_dict = load_client_activity_dict(config_dict, ForbiddenProvider(), from_date_str=from_str, to_date_str=to_str, as_of_ts=AS_OF_TS)
        assert result_dict["event_list"] == []


def test_overview_shows_three_material_events_with_same_activity_dates(monkeypatch):
    registry_dict, snapshot_dict = build_demo_fixture_tuple()
    provider_obj = DemoOperationsProvider()
    original_fn = provider_obj.get_pod_event_dict_list
    def history_list(pod_id_str, limit_int):
        row_dict = original_fn(pod_id_str)[0]
        return [dict(row_dict, event_timestamp_str=f"2026-09-0{day_int}T13:40:00Z") for day_int in (1, 2, 3)] + [dict(row_dict, event_name_str="heartbeat", message_str="HEARTBEAT_NOISE")]
    monkeypatch.setattr(provider_obj, "get_pod_event_dict_list", history_list)
    app_obj = create_app(provider_obj, read_only_bool=True,
        client_registry_dict=registry_dict, client_reporting_snapshot_fn=lambda client_id_str: snapshot_dict[client_id_str])
    client_obj = app_obj.test_client()
    html_str = client_obj.get("/clients/demo-owner/overview?from=2026-09-01&to=2026-09-02").get_data(as_text=True)
    assert html_str.count('class="client-event"') == 3
    assert "HEARTBEAT_NOISE" not in html_str
    assert "/clients/demo-owner/activity?from=2026-09-01&amp;to=2026-09-02" in html_str
    assert "Recorded changes" in html_str
    activity_html_str = client_obj.get("/clients/demo-owner/activity?from=2026-09-01&to=2026-09-02").get_data(as_text=True)
    assert activity_html_str.count('class="client-event"') == 6
    assert "HEARTBEAT_NOISE" in activity_html_str


def test_partial_source_failure_does_not_hide_other_strategy(monkeypatch):
    provider_obj = DemoOperationsProvider()
    original_fn = provider_obj.get_pod_event_dict_list
    def history_list(pod_id_str, limit_int):
        if pod_id_str == "demo_0_0":
            raise OSError("SECRET_PATH")
        return original_fn(pod_id_str, limit_int)
    monkeypatch.setattr(provider_obj, "get_pod_event_dict_list", history_list)
    result_dict = load_client_activity_dict(client_dict(), provider_obj, from_date_str="2026-06-01", to_date_str="2026-09-04", as_of_ts=AS_OF_TS)
    assert len(result_dict["event_list"]) == 1
    assert result_dict["coverage_str"] == "partial" and result_dict["issue_list"]
    assert "SECRET_PATH" not in json.dumps(result_dict)


def test_future_occurrence_on_same_owned_day_is_rejected():
    result_list = safe_client_event_list([event_dict(event_timestamp_str="2026-06-30T13:00:00Z")],
        {"pod_id_str": "demo_0_0", "account_route_str": "DEMO_0_0", "display_name_str": "A", "effective_from_str": "2026-06-01"},
        from_date_str="2026-06-01", to_date_str="2026-06-30", as_of_ts=datetime(2026, 6, 30, 12, tzinfo=UTC))
    assert result_list == []


def test_remote_display_cap_is_explicit_and_material_excerpt_remains_bounded(tmp_path):
    config_dict = snapshot_client_dict(tmp_path, [event_dict() for _ in range(1001)])
    result_dict = load_client_activity_dict(config_dict, ForbiddenProvider(), from_date_str="2026-06-01", to_date_str="2026-09-04", as_of_ts=AS_OF_TS)
    assert len(result_dict["event_list"]) == 1000
    assert result_dict["display_truncated_bool"] is True
    assert len(result_dict["material_event_list"]) == 3
    assert result_dict["coverage_str"] == "partial"


@pytest.mark.parametrize("level_str,expanded_bool", [("info", False), ("WARN", True), ("warning", True), ("error", True), ("critical", True), ("fatal", True)])
def test_compact_activity_keeps_warning_and_error_messages_expanded(tmp_path, level_str, expanded_bool):
    config_dict = snapshot_client_dict(tmp_path, [event_dict(level_str=level_str, message_str="Review execution evidence")])
    app_obj = create_app(ForbiddenProvider(), read_only_bool=True,
        client_registry_dict={"schema_version": 1, "clients": [config_dict]})
    html_str = app_obj.test_client().get("/clients/demo-owner/activity?from=2026-06-01&to=2026-06-30").get_data(as_text=True)
    event_tag_str = html_str.split('<details class="client-event"', 1)[1].split(">", 1)[0]
    assert ("open" in event_tag_str) is expanded_bool
    assert "Review execution evidence" in html_str
