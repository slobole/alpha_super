"""V4 uses saved LIVE evidence and cannot expose executable V3 routes."""

from copy import deepcopy
import re
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from alpha.live.client_reporting import BrokerReportingSnapshot
from alpha.live.dashboard_v4.app import create_app
from alpha.live.dashboard_v4.data import LiveDataProvider, LiveReadOnlyApp
from alpha.live.dashboard_v4.demo import DEMO_NOW_TS, build_demo_workspace_tuple, create_demo_app
from alpha.live.dashboard_v4.overview import build_overview_dict
from test_dashboard_local_workspace import build_fixture_app, file_snapshot_dict


@pytest.fixture
def fixture_tuple():
    return build_demo_workspace_tuple()


def _view_dict(fixture_tuple):
    workspace_dict, snapshot_obj, provider_obj = fixture_tuple
    return build_overview_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS)


def test_demo_renders_native_d_shell_and_seven_step_rows(fixture_tuple):
    app_obj = create_demo_app()
    html_str = app_obj.test_client().get("/").get_data(as_text=True)
    for label_str in ("ALPHA / OPS", "Overview", "Positions", "Performance", "Activity", "System health", "Tools",
                      "Account value", "Allocation", "Sample data. Not a real account."):
        assert label_str in html_str
    assert "READ-ONLY" not in html_str
    assert html_str.count('class="srow"') == 4
    assert 'aria-label="Portfolio return · 3M"' in html_str
    assert 'Calculated return · End-of-day cash flows' in html_str
    assert html_str.count('class="tl"') == 28
    assert 'title="PAPER is not available in V4 yet"' in html_str
    assert 'title="INCUBATION is not available in V4 yet"' in html_str
    assert "2026-09-08T" not in re.sub(r"<[^>]+>", "", html_str)
    assert '09:41:07 ET' in html_str
    assert 'hx-history="false"' in html_str and '"historyCacheSize":0' in html_str
    assert "hx-push-url" not in html_str
    assert html_str.count('hx-get="') == 1  # One acquisition, no competing period polls.
    partial_obj = app_obj.test_client().get("/overview/refresh?period=All")
    assert partial_obj.status_code == 200
    assert '<html' not in partial_obj.get_data(as_text=True)
    assert 'data-period="All" aria-current="true"' in partial_obj.get_data(as_text=True)
    hx_obj = app_obj.test_client().get("/?period=1M", headers={"HX-Request": "true"})
    assert '<html' not in hx_obj.get_data(as_text=True)


def test_healthy_monthly_idle_is_known_not_unknown(fixture_tuple, monkeypatch):
    workspace_dict = fixture_tuple[0]
    workspace_dict["operations_account_list"] = workspace_dict["operations_account_list"][2:]
    monkeypatch.setattr("alpha.live.dashboard_v4.overview.build_health_rollup", lambda *args, **kwargs: SimpleNamespace(severity_str="green"))
    view_dict = _view_dict(fixture_tuple)
    assert {pod_dict["state_str"] for pod_dict in view_dict["pod_list"]} == {"skip"}
    assert view_dict["live_state_str"] == "done"
    assert view_dict["attention_list"] == []
    assert view_dict["verdict_str"] == "No action needed."


def test_planned_wait_is_not_an_attention_item(fixture_tuple):
    workspace_dict = fixture_tuple[0]
    workspace_dict["operations_account_list"] = workspace_dict["operations_account_list"][:1]
    row_dict = workspace_dict["summary_dict"]["pod_row_dict_list"][0]
    row_dict.update(latest_vplan_status_str="submitted", latest_reconciliation_status_str="", latest_reconciliation_timestamp_str=None,
                    latest_vplan_target_execution_timestamp_str="2026-09-08T13:40:00+00:00",
                    required_action_dict={"severity_str": "yellow", "label_str": "Waiting reconcile", "reason_str": "waiting_for_post_execution_reconcile"})
    view_dict = _view_dict(fixture_tuple)
    assert view_dict["attention_list"] == []
    assert view_dict["pod_list"][0]["state_str"] == "now"


def test_completed_daily_pod_with_quantity_proof_needs_no_action(fixture_tuple):
    fixture_tuple[0]["operations_account_list"] = fixture_tuple[0]["operations_account_list"][:1]
    view_dict = _view_dict(fixture_tuple)
    assert view_dict["pod_list"][0]["pill_str"] == "On track"
    assert view_dict["pod_list"][0]["now_detail_str"] == "3 of 3 filled"
    assert view_dict["verdict_str"] == "No action needed."


@pytest.mark.parametrize("stage_str,due_str,allowance_int", [
    ("Submit", "2026-09-08T13:23:30+00:00", 60),
    ("Plan", "2026-09-08T13:23:30+00:00", 60),
    ("Reconcile", "2026-09-08T13:35:00+00:00", 30),
    ("EOD", "2026-09-08T20:10:00+00:00", 30),
])
@pytest.mark.parametrize("late_bool", [False, True])
def test_overview_attention_obeys_scheduler_allowance(fixture_tuple, stage_str, due_str, allowance_int, late_bool):
    workspace_dict, snapshot_obj, provider_obj = fixture_tuple
    workspace_dict["operations_account_list"] = workspace_dict["operations_account_list"][:1]
    summary_dict = workspace_dict["summary_dict"]
    row_dict = summary_dict["pod_row_dict_list"][0]
    now_dt = datetime.fromisoformat(due_str) + timedelta(seconds=allowance_int + 1 if late_bool else 5)
    summary_dict["as_of_timestamp_str"] = row_dict["as_of_timestamp_str"] = now_dt.isoformat()
    if stage_str in {"Submit", "Plan"}:
        row_dict.update(latest_vplan_status_str="ready", latest_decision_plan_status_str="vplan_ready",
                        latest_submit_ack_status_str="not_checked", broker_order_count_int=0, broker_ack_count_int=0, fill_count_int=0,
                        latest_reconciliation_timestamp_str=None, latest_reconciliation_status_str="",
                        next_action_str="submit_vplan", required_action_dict={"severity_str": "yellow", "label_str": "VPlan ready"})
        if stage_str == "Plan":
            row_dict.update(latest_vplan_id_int=None, latest_decision_plan_status_str="planned", next_action_str="build_vplan",
                            latest_decision_plan_submission_timestamp_str=due_str,
                            latest_decision_plan_target_execution_timestamp_str="2026-09-08T13:30:00+00:00",
                            required_action_dict={"severity_str": "yellow", "label_str": "Build VPlan"})
    elif stage_str == "Reconcile":
        row_dict.update(latest_vplan_status_str="submitted", latest_reconciliation_timestamp_str=None, latest_reconciliation_status_str="",
                        next_action_str="post_execution_reconcile", required_action_dict={"severity_str": "yellow", "label_str": "Waiting reconcile"})
        row_dict["cycle_evidence_dict"]["vplan_status_str"] = "submitted"
    else:
        row_dict["eod_snapshot_dict"].update(status_str="due_missing", last_required_market_date_str="2026-09-08", last_required_eod_present_bool=False)
        for item_dict in row_dict["data_freshness_dict"]["item_dict_list"]:
            if item_dict["label_str"] == "EOD Snapshot":
                item_dict["severity_str"] = "yellow"
    original_dict = deepcopy(workspace_dict)
    view_dict = build_overview_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=now_dt)
    assert bool(view_dict["attention_list"]) is late_bool
    assert view_dict["pod_list"][0]["state_str"] == ("late" if late_bool else "now")
    assert view_dict["verdict_str"] == ("1 pod needs action." if late_bool else "No action needed.")
    if late_bool:
        assert view_dict["attention_list"][0]["title_str"] not in {"No action", "Waiting reconcile"}
    assert workspace_dict == original_dict


@pytest.mark.parametrize("action_str,label_str,severity_str", [
    ("review_vplan", "Review VPlan", "yellow"), ("submit_vplan", "VPlan ready", "red"),
])
def test_submit_allowance_does_not_hide_manual_or_red_actions(fixture_tuple, action_str, label_str, severity_str):
    workspace_dict, snapshot_obj, provider_obj = fixture_tuple
    workspace_dict["operations_account_list"] = workspace_dict["operations_account_list"][:1]
    row_dict = workspace_dict["summary_dict"]["pod_row_dict_list"][0]
    row_dict.update(latest_vplan_status_str="ready", latest_submit_ack_status_str="not_checked",
                    latest_vplan_submission_timestamp_str=DEMO_NOW_TS.isoformat(), next_action_str=action_str,
                    required_action_dict={"label_str": label_str, "severity_str": severity_str})
    view_dict = build_overview_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS)
    assert view_dict["attention_list"][0]["title_str"] == label_str


@pytest.mark.parametrize("invalid_str", ["stale", "account", "mode"])
def test_evidence_reader_never_receives_stale_or_foreign_rows(fixture_tuple, monkeypatch, invalid_str):
    workspace_dict, _, provider_obj = fixture_tuple
    workspace_dict["operations_account_list"] = workspace_dict["operations_account_list"][:1]
    row_dict = workspace_dict["summary_dict"]["pod_row_dict_list"][0]
    if invalid_str == "stale":
        workspace_dict["summary_dict"]["as_of_timestamp_str"] = (DEMO_NOW_TS - timedelta(seconds=121)).isoformat()
    else:
        row_dict["account_route_str" if invalid_str == "account" else "mode_str"] = "foreign"
    def unexpected_read(*args, **kwargs):
        raise AssertionError("must scope and freshness-check before evidence acquisition")
    monkeypatch.setattr(provider_obj, "get_cycle_evidence_dict", unexpected_read, raising=False)
    _view_dict(fixture_tuple)


def test_evidence_reader_uses_summary_cutoff_not_response_clock(fixture_tuple, monkeypatch):
    workspace_dict, _, provider_obj = fixture_tuple
    workspace_dict["operations_account_list"] = workspace_dict["operations_account_list"][:1]
    cutoff_dt = DEMO_NOW_TS - timedelta(seconds=7)
    workspace_dict["summary_dict"]["as_of_timestamp_str"] = cutoff_dt.isoformat()
    call_list = []
    def capture_read(row_dict, *, as_of_ts):
        call_list.append((row_dict["pod_id_str"], as_of_ts))
        return {}
    monkeypatch.setattr(provider_obj, "get_cycle_evidence_dict", capture_read, raising=False)
    _view_dict(fixture_tuple)
    assert call_list == [(workspace_dict["operations_account_list"][0]["pod_id"], cutoff_dt)]


@pytest.mark.parametrize("error_type", [ValueError, OSError])
def test_optional_evidence_lookup_failure_keeps_overview_available(fixture_tuple, monkeypatch, error_type):
    workspace_dict, snapshot_obj, _ = fixture_tuple
    workspace_dict["operations_account_list"] = workspace_dict["operations_account_list"][:1]
    provider_obj = LiveDataProvider()
    def failed_lookup(pod_id_str):
        raise error_type("Configuration changed")
    monkeypatch.setattr(provider_obj, "get_target_for_pod", failed_lookup)
    monkeypatch.setattr("alpha.live.dashboard_v4.overview.build_financial_overview_dict", lambda *args, **kwargs: {})
    view_dict = build_overview_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS)
    assert view_dict["pod_list"][0]["pill_str"] == "Unknown"
    assert view_dict["verdict_str"] == "Status needs review."


@pytest.mark.parametrize("database_str", ["missing", "error"])
def test_missing_db_is_red_attention_even_with_gray_v3_action(fixture_tuple, database_str):
    workspace_dict = fixture_tuple[0]
    workspace_dict["operations_account_list"] = workspace_dict["operations_account_list"][:1]
    row_dict = workspace_dict["summary_dict"]["pod_row_dict_list"][0]
    row_dict.update(db_status_str=database_str, required_action_dict={"severity_str": "gray", "label_str": "Setup DB"})
    view_dict = _view_dict(fixture_tuple)
    assert view_dict["attention_list"][0]["state_str"] == "fail"
    assert view_dict["system_dict"]["state_str"] == "fail"
    assert "needs action" in view_dict["verdict_str"]
    assert {step_dict["state_str"] for step_dict in view_dict["pod_list"][0]["step_list"]} == {"unk"}


def test_attention_preserves_real_v3_reason_field(fixture_tuple):
    row_dict = fixture_tuple[0]["summary_dict"]["pod_row_dict_list"][1]
    row_dict["required_action_dict"] = {"severity_str": "red", "label_str": "Review broker ACK", "reason_str": "Missing ACK count: 1."}
    assert _view_dict(fixture_tuple)["attention_list"][0]["detail_str"] == "Missing ACK count: 1."


def test_header_formats_norgate_timestamp_fallback_to_et_seconds(fixture_tuple):
    for row_dict in fixture_tuple[0]["summary_dict"]["pod_row_dict_list"]:
        row_dict["data_freshness_dict"]["item_dict_list"][0]["value_str"] = "2026-09-08T12:10:00.053574+00:00"
    assert _view_dict(fixture_tuple)["system_dict"]["detail_str"] == "Data 09-08 08:10:00"


def test_source_expiry_budget_uses_remaining_assessment_age(fixture_tuple):
    fixture_tuple[0]["summary_dict"]["as_of_timestamp_str"] = (DEMO_NOW_TS - timedelta(seconds=119)).isoformat()
    assert _view_dict(fixture_tuple)["source_valid_ms_int"] == 1000


@pytest.mark.parametrize("age_int", [121, -1, None])
def test_stale_future_missing_source_never_reuses_green(fixture_tuple, age_int):
    summary_dict = fixture_tuple[0]["summary_dict"]
    summary_dict["as_of_timestamp_str"] = (DEMO_NOW_TS - timedelta(seconds=age_int)).isoformat() if age_int is not None else None
    view_dict = _view_dict(fixture_tuple)
    assert view_dict["verdict_str"] == "Status unknown."
    assert view_dict["live_state_str"] == "unk"
    assert {step_dict["state_str"] for pod_dict in view_dict["pod_list"] for step_dict in pod_dict["step_list"]} == {"unk"}
    assert view_dict["attention_list"] == []


@pytest.mark.parametrize("change_str", ["account", "duplicate", "mode"])
def test_ambiguous_or_foreign_pod_evidence_is_unknown(fixture_tuple, change_str):
    summary_dict = fixture_tuple[0]["summary_dict"]
    row_dict = summary_dict["pod_row_dict_list"][0]
    if change_str == "account":
        row_dict["account_route_str"] = "FOREIGN"
    elif change_str == "mode":
        row_dict["mode_str"] = "paper"
    else:
        summary_dict["pod_row_dict_list"].append(deepcopy(row_dict))
    assert _view_dict(fixture_tuple)["pod_list"][0]["state_str"] == "unk"


def test_finance_failure_keeps_operations_and_no_demo_fallback(fixture_tuple):
    workspace_dict, _, provider_obj = fixture_tuple
    view_dict = build_overview_dict(workspace_dict, BrokerReportingSnapshot(unavailable_reason_str="bad source"), provider_obj, as_of_ts=DEMO_NOW_TS)
    assert len(view_dict["pod_list"]) == 4
    assert view_dict["attention_list"]
    assert all(tile_dict["value_str"] == "—" for tile_dict in view_dict["tile_list"])


@pytest.mark.parametrize("path_str", ["/", "/overview/refresh"])
@pytest.mark.parametrize("query_str", ["mode=paper", "mode=incubation", "period=bad", "period=1M&period=All", "client=foreign"])
def test_invalid_scope_rejected_before_acquisition(path_str, query_str):
    def forbidden_fn():
        raise AssertionError("Provider must not be called")
    client_obj = create_app(workspace_snapshot_fn=forbidden_fn).test_client()
    assert client_obj.get(path_str + "?" + query_str).status_code == 400


@pytest.mark.parametrize("method_str", ["POST", "PUT", "PATCH", "DELETE"])
@pytest.mark.parametrize("path_str", ["/", "/actions/run", "/api/actions", "/tools/execute", "/advanced/pods/test/action", "/overview/refresh"])
def test_all_mutations_denied_before_any_reader(method_str, path_str):
    client_obj = create_app(workspace_snapshot_fn=lambda: pytest.fail("Unexpected read")).test_client()
    assert client_obj.open(path_str, method=method_str).status_code == 403


def test_read_only_routes_assets_and_remote_boundary():
    client_obj = create_app(workspace_snapshot_fn=lambda: pytest.fail("Unexpected read")).test_client()
    assert client_obj.get("/healthz").json == {"service": "dashboard_v4", "scope": "live", "read_only": True}
    for path_str in ("/actions", "/actions/token", "/api/actions", "/export", "/advanced", "/assets/../app.py", "/assets/ops.css"):
        assert client_obj.get(path_str).status_code == 404
    response_obj = client_obj.get("/assets/fonts/IBMPlexSans-latin.woff2")
    assert response_obj.status_code == 200
    assert response_obj.headers["Cache-Control"] == "no-store"
    assert "default-src 'self'" in response_obj.headers["Content-Security-Policy"]
    assert client_obj.get("/", environ_overrides={"REMOTE_ADDR": "203.0.113.8"}, headers={"X-Forwarded-Proto": "https"}).status_code == 426


def test_live_filter_before_summary_state_acquisition(monkeypatch):
    target_list = [SimpleNamespace(release_obj=SimpleNamespace(mode_str=mode_str)) for mode_str in ("live", "paper", "incubation")]
    monkeypatch.setattr("alpha.live.dashboard.DashboardApp.get_target_list", lambda self: target_list)
    app_obj = LiveReadOnlyApp()
    assert app_obj.get_target_list() == target_list[:1]
    assert app_obj.diff_job_manager_obj is None
    assert app_obj.action_job_manager_obj is None
    assert app_obj.pod_job_gate_obj is None


@pytest.mark.parametrize("finance_bool", [True, False])
def test_real_saved_source_routes_never_modify_files_or_read_non_live(tmp_path, monkeypatch, finance_bool):
    v3_app_obj = build_fixture_app(tmp_path, monkeypatch, finance_bool=finance_bool)
    source_obj = v3_app_obj.config["data_provider_obj"]
    provider_obj = LiveDataProvider(releases_root_path_str=source_obj.releases_root_path_str,
        config_path_str=source_obj.config_path_str, results_root_path_str=source_obj.results_root_path_str,
        event_log_path_str=source_obj.event_log_path_str)
    from alpha.live import dashboard
    original_fn = dashboard.build_pod_row_dict
    seen_list = []
    def live_row_fn(pod_target_obj, *args, **kwargs):
        assert pod_target_obj.release_obj.mode_str == "live"
        seen_list.append(pod_target_obj.release_obj.pod_id_str)
        return original_fn(pod_target_obj, *args, **kwargs)
    monkeypatch.setattr(dashboard, "build_pod_row_dict", live_row_fn)
    app_obj = create_app(provider_obj, performance_db_path_str=v3_app_obj.config["performance_db_path_str"])
    before_dict = file_snapshot_dict(tmp_path)
    client_obj = app_obj.test_client()
    for path_str in ("/", "/overview/refresh?period=All"):
        response_obj = client_obj.get(path_str)
        assert response_obj.status_code == 200
        html_str = response_obj.get_data(as_text=True)
        assert 'data-pod-id="pod_a"' in html_str and 'data-pod-id="pod_new"' in html_str
        assert "pod_sim" not in html_str and "demo_" not in html_str
        assert "Sample data" not in html_str
    assert set(seen_list) == {"pod_a", "pod_b", "pod_new"}
    assert client_obj.post("/actions/run").status_code == 403
    assert file_snapshot_dict(tmp_path) == before_dict


def test_empty_installation_renders_unknown_without_creating_files(tmp_path):
    provider_obj = LiveDataProvider(releases_root_path_str=str(tmp_path / "releases"),
        config_path_str=str(tmp_path / "config.yaml"), results_root_path_str=str(tmp_path / "results"),
        event_log_path_str=str(tmp_path / "events.jsonl"))
    response_obj = create_app(provider_obj, performance_db_path_str=str(tmp_path / "missing.sqlite3")).test_client().get("/")
    assert response_obj.status_code == 200
    assert "No LIVE pods" in response_obj.get_data(as_text=True)
    assert list(tmp_path.iterdir()) == []
