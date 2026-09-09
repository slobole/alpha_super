from copy import deepcopy
from datetime import UTC, datetime
import json

import pytest
from flask import template_rendered

from alpha.live.dashboard_v3.app import create_app
from alpha.live.dashboard_v3.client_operations import (
    build_client_operations_dict, build_reference_exposure_list,
    load_operations_summary_dict, safe_client_event_list,
)
from alpha.live.dashboard_v3.demo import DemoOperationsProvider, build_demo_fixture_tuple
from alpha.live.dashboard_v3.schedule import TradingWindow, build_trading_window_list
from test_dashboard_operator_access import ForbiddenProvider, ForbiddenProvider


AS_OF_TS = datetime(2026, 9, 5, 12, tzinfo=UTC)


def fixture_tuple():
    registry_dict, snapshot_dict = build_demo_fixture_tuple()
    provider_obj = DemoOperationsProvider()
    summary_dict = provider_obj.get_summary_dict()
    summary_dict["as_of_timestamp_str"] = AS_OF_TS.isoformat()
    return registry_dict, snapshot_dict, provider_obj, summary_dict


def test_scoping_does_not_mutate_shared_summary_or_pool_clients():
    registry_dict, _, _, summary_dict = fixture_tuple()
    original_dict = deepcopy(summary_dict)
    owner_dict = build_client_operations_dict(registry_dict["clients"][0], summary_dict, as_of_ts=AS_OF_TS)
    second_dict = build_client_operations_dict(registry_dict["clients"][1], summary_dict, as_of_ts=AS_OF_TS)
    assert len(owner_dict["strategy_list"]) == 2
    assert owner_dict["severity_str"] == "green"
    assert len(second_dict["strategy_list"]) == 4
    assert second_dict["severity_str"] == "yellow"
    assert [row_dict["severity_str"] for row_dict in second_dict["strategy_list"]] == ["green", "green", "green", "yellow"]
    assert "DEMO_1_" not in json.dumps(owner_dict)
    assert "DEMO_0_" not in json.dumps(second_dict)
    assert summary_dict == original_dict


@pytest.mark.parametrize("field_str,value_obj", [("account_route_str", "WRONG"), ("pod_id_str", "WRONG"), ("mode_str", "paper"), ("account_route_str", None)])
def test_identity_mismatch_never_verifies(field_str, value_obj):
    registry_dict, _, _, summary_dict = fixture_tuple()
    summary_dict["pod_row_dict_list"][0][field_str] = value_obj
    result_dict = build_client_operations_dict(registry_dict["clients"][0], summary_dict, as_of_ts=AS_OF_TS)
    assert result_dict["strategy_list"][0]["matched_bool"] is False
    assert result_dict["severity_str"] == "gray"
    assert result_dict["strategy_list"][0]["evidence_dict"] == {}


def test_duplicate_pod_id_is_ambiguous_even_if_only_one_account_matches():
    registry_dict, _, _, summary_dict = fixture_tuple()
    duplicate_dict = deepcopy(summary_dict["pod_row_dict_list"][0])
    duplicate_dict["account_route_str"] = "OTHER"
    summary_dict["pod_row_dict_list"].append(duplicate_dict)
    assert not build_client_operations_dict(registry_dict["clients"][0], summary_dict, as_of_ts=AS_OF_TS)["strategy_list"][0]["matched_bool"]


@pytest.mark.parametrize("source_str", [None, "2026-09-05T11:57:59+00:00", "2026-09-05T12:00:01+00:00"])
def test_old_missing_future_assessment_is_not_green(source_str):
    registry_dict, _, _, summary_dict = fixture_tuple()
    summary_dict["as_of_timestamp_str"] = source_str
    assert build_client_operations_dict(registry_dict["clients"][0], summary_dict, as_of_ts=AS_OF_TS)["severity_str"] == "gray"


def test_pre_mandate_and_future_state_never_expose_holdings():
    registry_dict, _, _, summary_dict = fixture_tuple()
    for timestamp_str in ("2026-05-29T20:10:00+00:00", "2026-09-06T20:10:00+00:00"):
        summary_dict["pod_row_dict_list"][0]["latest_pod_state_timestamp_str"] = timestamp_str
        result_dict = build_client_operations_dict(registry_dict["clients"][0], summary_dict, as_of_ts=AS_OF_TS)
        assert result_dict["strategy_list"][0]["evidence_dict"] == {}


@pytest.mark.parametrize("timestamp_str", [None, "invalid timestamp"])
def test_local_missing_or_invalid_state_keeps_flow_not_holdings(timestamp_str):
    registry_dict, _, _, summary_dict = fixture_tuple()
    summary_dict["pod_row_dict_list"][0]["latest_pod_state_timestamp_str"] = timestamp_str
    result_dict = build_client_operations_dict(registry_dict["clients"][0], summary_dict, as_of_ts=AS_OF_TS,
        local_account_list=registry_dict["clients"][0]["accounts"][:1])
    strategy_dict = result_dict["strategy_list"][0]
    assert strategy_dict["matched_bool"] is True
    assert strategy_dict["severity_str"] != "green"
    assert strategy_dict["evidence_dict"]["position_exposure_dict_list"] == []
    assert strategy_dict["evidence_dict"]["lifecycle_step_dict_list"]
    assert strategy_dict["issue_list"].count("Pod state unavailable.") == 1
    assert "Pod state: Latest persisted" not in str(strategy_dict["issue_list"])


def test_issue_dedup_preserves_distinct_causes_and_raw_evidence():
    from alpha.live.dashboard_v3.client_operations import _deduplicated_issue_list
    issue_list = ["Review Norgate data: Snapshot missing", "Norgate: Snapshot missing",
        "Snapshot missing", "EOD Snapshot: Row missing", "Broker ACK missing", "Broker ACK missing"]
    assert _deduplicated_issue_list(issue_list) == ["Review Norgate data: Snapshot missing",
        "EOD Snapshot: Row missing", "Broker ACK missing"]
    assert len(issue_list) == 6


def test_compact_summary_prioritizes_red_broker_failure_over_yellow_data():
    registry_dict, _, _, summary_dict = fixture_tuple()
    row_dict = summary_dict["pod_row_dict_list"][0]
    norgate_dict = next(item_dict for item_dict in row_dict["data_freshness_dict"]["item_dict_list"] if item_dict["label_str"] == "Norgate")
    norgate_dict.update(severity_str="yellow", detail_str="Snapshot missing")
    row_dict["required_action_dict"] = {"severity_str": "red", "detail_str": "Broker ACK missing"}
    original_dict = deepcopy(summary_dict)
    result_dict = build_client_operations_dict(registry_dict["clients"][0], summary_dict, as_of_ts=AS_OF_TS)
    strategy_dict = result_dict["strategy_list"][0]
    assert strategy_dict["severity_str"] == "red" and strategy_dict["summary_str"] == "Broker ACK missing"
    assert any("Snapshot missing" in issue_str for issue_str in strategy_dict["issue_list"])
    assert summary_dict == original_dict


def test_retired_and_future_periods_do_not_enter_current_scope():
    registry_dict, _, _, summary_dict = fixture_tuple()
    registry_dict["clients"][0]["accounts"][0]["effective_to"] = "2026-09-04"
    registry_dict["clients"][0]["accounts"][1]["effective_from"] = "2026-09-06"
    assert build_client_operations_dict(registry_dict["clients"][0], summary_dict, as_of_ts=AS_OF_TS)["strategy_list"] == []


@pytest.mark.parametrize("field_str", ["required_action_dict", "debug_summary_dict", "data_freshness_dict"])
def test_malformed_nested_status_is_unknown(field_str):
    registry_dict, _, _, summary_dict = fixture_tuple()
    summary_dict["pod_row_dict_list"][0][field_str] = ["bad"]
    result_dict = build_client_operations_dict(registry_dict["clients"][0], summary_dict, as_of_ts=AS_OF_TS)
    assert result_dict["severity_str"] != "green"


def test_projection_redacts_nested_credentials_and_preserves_reference_dates():
    registry_dict, _, _, summary_dict = fixture_tuple()
    summary_dict["pod_row_dict_list"][0]["debug_summary_dict"]["detail_str"] = "https://local/?token=PRIVATE&ok=1"
    result_dict = build_client_operations_dict(registry_dict["clients"][0], summary_dict, as_of_ts=AS_OF_TS)
    assert "PRIVATE" not in json.dumps(result_dict)
    position_dict = build_reference_exposure_list(result_dict)[0]
    assert position_dict["reference_timestamp_str"].startswith("2026-09-01")
    assert position_dict["position_timestamp_str"].startswith("2026-09-04")
    assert position_dict["reference_value_float"] == 1400
    result_dict["strategy_list"][0]["evidence_dict"]["position_exposure_dict_list"][0].update(share_float=1e308, price_float=1e308)
    assert build_reference_exposure_list(result_dict)[0]["reference_value_float"] is None


def test_schedule_preserves_current_vs_previous_cycle_selection():
    registry_dict, _, _, summary_dict = fixture_tuple()
    row_dict = summary_dict["pod_row_dict_list"][0]
    row_dict.update(latest_decision_plan_id_int=2, latest_decision_plan_status_str="ready", next_action_str="build_vplan",
                    latest_decision_plan_target_execution_timestamp_str="2026-10-01T13:30:00+00:00",
                    latest_decision_plan_submission_timestamp_str="2026-10-01T13:23:30+00:00",
                    latest_vplan_is_for_latest_decision_bool=False, latest_vplan_cycle_role_str="previous_cycle",
                    latest_vplan_target_execution_timestamp_str="2026-09-01T13:30:00+00:00", latest_vplan_status_str="completed")
    client_dict = deepcopy(registry_dict["clients"][0])
    client_dict["accounts"] = client_dict["accounts"][:1]
    full_list = [window_obj.as_dict() for window_obj in build_trading_window_list({"pod_row_dict_list": [row_dict]}, mode_str="live", now_dt=AS_OF_TS)]
    assert build_client_operations_dict(client_dict, summary_dict, as_of_ts=AS_OF_TS)["trading_window_list"] == full_list


def test_missed_calendar_cycle_overrides_green_saved_labels():
    registry_dict, _, _, summary_dict = fixture_tuple()
    summary_dict["pod_row_dict_list"][0]["latest_decision_signal_timestamp_str"] = None
    result_dict = build_client_operations_dict(registry_dict["clients"][0], summary_dict, as_of_ts=AS_OF_TS)
    assert result_dict["severity_str"] == "red"
    assert result_dict["strategy_list"][0]["status_label_str"] == "Action required"


def test_unknown_calendar_is_unknown_without_hiding_healthy_strategy():
    registry_dict, _, _, summary_dict = fixture_tuple()
    summary_dict["pod_row_dict_list"][0]["session_calendar_id_str"] = "UNKNOWN"
    result_dict = build_client_operations_dict(registry_dict["clients"][0], summary_dict, as_of_ts=AS_OF_TS)
    assert result_dict["severity_str"] == "gray"
    assert [row_dict["severity_str"] for row_dict in result_dict["strategy_list"]] == ["gray", "green"]
    assert any(not window_dict["has_data_bool"] for window_dict in result_dict["trading_window_list"])
    assert any(window_dict["has_data_bool"] for window_dict in result_dict["trading_window_list"])


@pytest.mark.parametrize("failure_str", ["exception", "empty", "no_data", "idle"])
def test_calendar_failure_is_scoped_and_valid_gray_idle_stays_green(monkeypatch, failure_str):
    registry_dict, _, _, summary_dict = fixture_tuple()
    first_pod_str = summary_dict["pod_row_dict_list"][0]["pod_id_str"]

    def build_window_list(summary_dict, **kwargs_dict):
        if any(row_dict["pod_id_str"] == first_pod_str for row_dict in summary_dict["pod_row_dict_list"]):
            if failure_str == "exception":
                raise ValueError("Malformed calendar")
            if failure_str == "empty":
                return []
            return [TradingWindow(has_data_bool=failure_str == "idle", status_label_str="No action" if failure_str == "idle" else "Cannot verify", pod_id_str_list=[first_pod_str])]
        return build_trading_window_list(summary_dict, **kwargs_dict)

    monkeypatch.setattr("alpha.live.dashboard_v3.client_operations.build_trading_window_list", build_window_list)
    result_dict = build_client_operations_dict(registry_dict["clients"][0], summary_dict, as_of_ts=AS_OF_TS)
    assert result_dict["strategy_list"][0]["severity_str"] == ("green" if failure_str == "idle" else "gray")
    assert result_dict["strategy_list"][1]["severity_str"] == "green"


def test_existing_red_action_is_not_downgraded_by_unknown_calendar():
    registry_dict, _, _, summary_dict = fixture_tuple()
    summary_dict["pod_row_dict_list"][0].update(session_calendar_id_str="UNKNOWN", required_action_dict={"severity_str": "red", "detail_str": "Rejected order"})
    result_dict = build_client_operations_dict(registry_dict["clients"][0], summary_dict, as_of_ts=AS_OF_TS)
    assert result_dict["severity_str"] == "red"
    assert result_dict["strategy_list"][0]["status_label_str"] == "Action required"


def test_compact_status_shows_red_reason_even_after_an_unranked_calendar_issue(monkeypatch):
    registry_dict, _, provider_obj, summary_dict = fixture_tuple()
    summary_dict["pod_row_dict_list"][0].update(session_calendar_id_str="UNKNOWN", required_action_dict={"severity_str": "red", "detail_str": "Rejected order"})
    summary_dict["as_of_timestamp_str"] = datetime.now(UTC).isoformat()
    monkeypatch.setattr(provider_obj, "get_summary_dict", lambda: summary_dict)
    app_obj = create_app(provider_obj, read_only_bool=True, client_registry_dict={**registry_dict, "clients": registry_dict["clients"][:1]})
    html_str = app_obj.test_client().get("/clients/demo-owner/strategies").get_data(as_text=True)
    banner_str = html_str.split('aria-label="Current client operations">', 1)[1].split('</section>', 1)[0]
    assert "Rejected order" in banner_str.split('<details', 1)[0]
    assert "Saved check" in banner_str


def test_persisted_target_without_submission_evidence_is_unknown():
    registry_dict, _, _, summary_dict = fixture_tuple()
    summary_dict["pod_row_dict_list"][0].update(reason_code_str="awaiting_execution", latest_vplan_status_str="ready",
        latest_vplan_target_execution_timestamp_str="2026-10-01T13:30:00+00:00", latest_vplan_submission_timestamp_str=None)
    result_dict = build_client_operations_dict(registry_dict["clients"][0], summary_dict, as_of_ts=AS_OF_TS)
    assert [row_dict["severity_str"] for row_dict in result_dict["strategy_list"]] == ["gray", "green"]
    assert "missing signal or submission evidence" in result_dict["strategy_list"][0]["issue_list"][0]


def test_invalid_calendar_target_preserves_red_action():
    registry_dict, _, _, summary_dict = fixture_tuple()
    summary_dict["pod_row_dict_list"][0]["latest_vplan_target_execution_timestamp_str"] = "BAD"
    result_dict = build_client_operations_dict(registry_dict["clients"][0], summary_dict, as_of_ts=AS_OF_TS)
    assert result_dict["severity_str"] == "red"
    assert [row_dict["severity_str"] for row_dict in result_dict["strategy_list"]] == ["red", "green"]


def test_shared_unattributed_and_pre_mandate_events_are_omitted():
    registry_dict, _, provider_obj, summary_dict = fixture_tuple()
    strategy_dict = build_client_operations_dict(registry_dict["clients"][0], summary_dict, as_of_ts=AS_OF_TS)["strategy_list"][0]
    event_dict = provider_obj.get_pod_event_dict_list(strategy_dict["pod_id_str"])[0]
    event_list = [event_dict, dict(event_dict, related_pod_id_list=["OTHER"]), dict(event_dict, account_route_str="OTHER"), dict(event_dict, event_timestamp_str="2026-05-01T13:40:00+00:00")]
    result_list = safe_client_event_list(event_list, strategy_dict, from_date_str="2026-01-01", to_date_str="2026-09-05")
    assert len(result_list) == 1
    assert "account_route_str" not in result_list[0]


@pytest.mark.parametrize("envelope_dict", [{}, {"schema_version_int": True, "client_id_str": "demo-owner", "summary_dict": {}}, {"schema_version_int": 1, "client_id_str": "other", "summary_dict": {}}])
def test_snapshot_requires_exact_schema_and_client(tmp_path, envelope_dict):
    registry_dict, _, _, _ = fixture_tuple()
    path_obj = tmp_path / "saved.json"
    path_obj.write_text(json.dumps(envelope_dict), encoding="utf-8")
    client_dict = dict(registry_dict["clients"][0], operations_source="snapshot", operations_snapshot_path=str(path_obj))
    with pytest.raises(ValueError):
        load_operations_summary_dict(client_dict, ForbiddenProvider())


@pytest.mark.parametrize("view_str", ["overview", "strategies", "exposure", "activity", "diagnostics"])
def test_client_routes_keep_scope_and_dates_without_login(view_str):
    registry_dict, snapshot_dict, provider_obj, _ = fixture_tuple()
    app_obj = create_app(provider_obj, read_only_bool=True,
                         client_registry_dict={**registry_dict, "clients": registry_dict["clients"][:1]}, client_reporting_snapshot_fn=lambda client_id_str: snapshot_dict[client_id_str])
    app_obj.config["TESTING"] = True
    client_obj = app_obj.test_client()
    path_str = f"/clients/demo-owner/{view_str}?from=2026-06-01&to=2026-09-04"
    response_obj = client_obj.get(path_str)
    assert response_obj.status_code == 200
    html_str = response_obj.get_data(as_text=True)
    assert "DEMO_1_" not in html_str
    assert "Cannot verify operations" not in html_str
    assert 'to=2026-09-04' in html_str
    assert "cdn.tailwindcss.com" not in html_str
    assert response_obj.headers["Cache-Control"] == "no-store"
    assert client_obj.get("/").location == "/clients"


def test_financial_filter_never_changes_current_operations():
    registry_dict, snapshot_dict, provider_obj, _ = fixture_tuple()
    app_obj = create_app(provider_obj, read_only_bool=True, client_registry_dict={**registry_dict, "clients": registry_dict["clients"][:1]},
                         client_reporting_snapshot_fn=lambda client_id_str: snapshot_dict[client_id_str])
    client_obj = app_obj.test_client()
    result_list = [client_obj.get(f"/clients/demo-owner/diagnostics?from={from_str}&to=2026-09-04&download=json").get_json() for from_str in ("2026-06-01", "2026-09-01")]
    assert [row_dict["pod_id_str"] for row_dict in result_list[0]["strategy_list"]] == [row_dict["pod_id_str"] for row_dict in result_list[1]["strategy_list"]]
    assert all(result_dict["severity_str"] == "green" for result_dict in result_list)


def test_unconfigured_ops_never_calls_provider_or_financial_loader():
    registry_dict, _, _, _ = fixture_tuple()
    registry_dict["clients"][0]["operations_source"] = "unconfigured"
    app_obj = create_app(ForbiddenProvider(), read_only_bool=True, client_registry_dict={**registry_dict, "clients": registry_dict["clients"][:1]},
                         client_reporting_snapshot_fn=lambda client_id_str: pytest.fail("Operations must not read financial data"))
    response_obj = app_obj.test_client().get("/clients/demo-owner/diagnostics?from=2026-06-01&to=2026-09-04")
    assert response_obj.status_code == 200
    assert "Cannot verify operations" in response_obj.get_data(as_text=True)


@pytest.mark.parametrize("view_str", ["strategies", "exposure", "activity", "diagnostics"])
@pytest.mark.parametrize("query_str", ["", "?window=mtd"])
def test_operations_accessible_on_mandate_day_without_financial_read(monkeypatch, view_str, query_str):
    registry_dict, _, provider_obj, _ = fixture_tuple()
    for client_dict in registry_dict["clients"]:
        client_dict["mandate_start_date"] = "2026-09-05"
        for account_dict in client_dict["accounts"]:
            account_dict["effective_from"] = "2026-09-05"

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return AS_OF_TS

    monkeypatch.setattr("alpha.live.dashboard_v3.client_views.datetime", FixedDatetime)
    app_obj = create_app(provider_obj, read_only_bool=True, client_registry_dict={**registry_dict, "clients": registry_dict["clients"][:1]},
                         client_reporting_snapshot_fn=lambda client_id_str: pytest.fail("Operations must not read financial data"))
    rendered_period_list = []
    def capture_period(sender_obj, template, context, **extra_dict):
        rendered_period_list.append(context["report_dict"])
    with template_rendered.connected_to(capture_period, app_obj):
        response_obj = app_obj.test_client().get(f"/clients/demo-owner/{view_str}{query_str}")
    assert response_obj.status_code == 200
    html_str = response_obj.get_data(as_text=True)
    assert rendered_period_list == [{"requested_from_date_str": "2026-09-05", "requested_to_date_str": "2026-09-05"}]
    # Operational defaults include the mandate day, but rail links must not
    # turn that default into an explicitly selected financial date.
    assert f'href="/clients/demo-owner/performance{query_str}"' in html_str


def test_diagnostics_preserves_nonblocking_next_cycle_source_warning(monkeypatch):
    registry_dict, _, provider_obj, summary_dict = fixture_tuple()
    norgate_dict = next(item_dict for item_dict in summary_dict["pod_row_dict_list"][0]["data_freshness_dict"]["item_dict_list"] if item_dict["label_str"] == "Norgate")
    norgate_dict["sub_detail_str_list"] = ["Sync failed: next DecisionPlan needs review; token=PRIVATE"]
    summary_dict["as_of_timestamp_str"] = datetime.now(UTC).isoformat()
    monkeypatch.setattr(provider_obj, "get_summary_dict", lambda: summary_dict)
    app_obj = create_app(provider_obj, read_only_bool=True, client_registry_dict={**registry_dict, "clients": registry_dict["clients"][:1]})
    html_str = app_obj.test_client().get("/clients/demo-owner/diagnostics").get_data(as_text=True)
    assert "No action required" in html_str
    assert "Sync failed: next DecisionPlan needs review" in html_str
    assert html_str.count("Sync failed: next DecisionPlan needs review") == 1
    assert "Sync failed: next DecisionPlan needs review" in html_str.split("Saved checks and lifecycle", 1)[1]
    assert "PRIVATE" not in html_str


@pytest.mark.parametrize("previous_role_str", ["current", "previous", "previous_cycle"])
def test_visible_pod_flow_preserves_recorded_not_complete_fill_and_cycle_role(monkeypatch, previous_role_str):
    from alpha.live.dashboard import _build_lifecycle_step_dict_list

    registry_dict, _, provider_obj, summary_dict = fixture_tuple()
    row_dict = summary_dict["pod_row_dict_list"][0]
    row_dict.update(latest_vplan_cycle_role_str=previous_role_str,
        latest_vplan_is_for_latest_decision_bool=previous_role_str == "current",
        latest_vplan_status_str="submitted", latest_submit_ack_status_str="complete",
        fill_count_int=1, next_action_str="post_execution_reconcile")
    row_dict["lifecycle_step_dict_list"] = _build_lifecycle_step_dict_list(row_dict)
    summary_dict["as_of_timestamp_str"] = datetime.now(UTC).isoformat()
    monkeypatch.setattr(provider_obj, "get_summary_dict", lambda: summary_dict)
    app_obj = create_app(provider_obj, read_only_bool=True, client_registry_dict={**registry_dict, "clients": registry_dict["clients"][:1]})
    html_str = app_obj.test_client().get("/clients/demo-owner/strategies").get_data(as_text=True)
    card_str = html_str.split('<h2>Tactical allocation</h2>', 1)[1].split('</section>', 1)[0]
    flow_str = card_str.split('<ol class="client-pod-flow"', 1)[1].split('</ol>', 1)[0]
    assert card_str.index('class="client-pod-flow"') < card_str.index('<details')
    assert '<strong>Fill</strong><small>Recorded</small>' in flow_str
    assert '<strong>ACK</strong><small>Complete</small>' in flow_str
    assert '<strong>VPlan</strong><small>Submitted</small>' in flow_str
    assert '<time>' in flow_str
    assert 'Stage details' in card_str
    assert 'data-focus="true"' in flow_str
    assert 'Live vs Backtest' not in flow_str
    assert 'Live vs Backtest' in card_str.split('<details', 1)[1]
    assert flow_str.count('<li ') == 7
    assert 'Saved next action' in card_str
    assert 'Data freshness' in card_str
    assert ('Execution steps: previous cycle' in card_str) is (previous_role_str != "current")
    assert html_str.count('class="client-pod-flow"') == 2


@pytest.mark.parametrize("stage_list", [[], [{"label_str": "ACK"}], [{"step_key_str": "diff", "label_str": "Live vs Backtest", "status_str": "available", "severity_str": "green"}]])
def test_visible_pod_flow_does_not_invent_missing_stage_evidence(monkeypatch, stage_list):
    registry_dict, _, provider_obj, summary_dict = fixture_tuple()
    summary_dict["pod_row_dict_list"][0]["lifecycle_step_dict_list"] = stage_list
    summary_dict["as_of_timestamp_str"] = datetime.now(UTC).isoformat()
    monkeypatch.setattr(provider_obj, "get_summary_dict", lambda: summary_dict)
    app_obj = create_app(provider_obj, read_only_bool=True, client_registry_dict={**registry_dict, "clients": registry_dict["clients"][:1]})
    html_str = app_obj.test_client().get("/clients/demo-owner/strategies").get_data(as_text=True)
    flow_str = html_str.split('<ol class="client-pod-flow"', 1)[1].split('</ol>', 1)[0]
    assert ('Unverified' if stage_list and stage_list[0]["label_str"] == "ACK" else 'Flow unavailable') in flow_str
    assert 'data-severity="gray"' in flow_str
    assert 'data-severity="green"' not in flow_str
