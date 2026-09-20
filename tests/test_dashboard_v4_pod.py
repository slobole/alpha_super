"""Selected cycle joins and route boundaries must preserve saved evidence."""

from copy import deepcopy
from datetime import timedelta

import pytest

from alpha.live.dashboard_v4.demo import DEMO_NOW_TS, build_demo_workspace_tuple, create_demo_app
from alpha.live.dashboard_v4.app import create_app
from alpha.live.dashboard_v4.pod import build_evidence_tables_dict, build_pod_page_dict
from alpha.live.dashboard_v4.overview import build_overview_dict


@pytest.fixture
def pod_fixture_tuple():
    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple()
    pod_id_str = workspace_dict["operations_account_list"][0]["pod_id"]
    source_dict = provider_obj.get_pod_cycles_dict(pod_id_str, as_of_ts=DEMO_NOW_TS)
    return workspace_dict, snapshot_obj, provider_obj, source_dict, pod_id_str


def test_request_join_preserves_same_symbol_entry_and_exit(pod_fixture_tuple):
    source_dict = pod_fixture_tuple[3]
    for list_key_str in ("plan_row_list", "order_list", "ack_list", "fill_list"):
        for row_dict in source_dict[list_key_str]:
            row_dict["asset_str"] = "SPY"
    source_dict["vplan_dict"]["current_broker_position_map_dict"] = {"SPY": 100}
    source_dict["reconciliation_dict"].update(model_position_map_dict={"SPY": 104}, broker_position_map_dict={"SPY": 104})
    table_dict = build_evidence_tables_dict(source_dict, as_of_ts=DEMO_NOW_TS, fresh_bool=True)["plan"]
    assert len(table_dict["row_list"]) == 3
    assert [row_dict["cell_list"][2] for row_dict in table_dict["row_list"]] == ["BUY 31", "BUY 17", "SELL 44"]
    assert [row_dict["cell_list"][4] for row_dict in table_dict["row_list"]] == ["158.42", "291.05", "112.80"]
    assert {row_dict["cell_list"][1] for row_dict in table_dict["row_list"]} == {"100"}
    assert {row_dict["cell_list"][5] for row_dict in table_dict["row_list"]} == {"104"}


def test_partial_execution_prices_are_weighted_within_order_only(pod_fixture_tuple):
    source_dict = pod_fixture_tuple[3]
    first_dict = source_dict["fill_list"][0]
    first_dict.update(fill_amount_float=10, fill_price_float=100)
    source_dict["fill_list"].append({**first_dict, "fill_amount_float": 21, "fill_price_float": 200})
    table_dict = build_evidence_tables_dict(source_dict, as_of_ts=DEMO_NOW_TS, fresh_bool=True)["plan"]
    assert table_dict["row_list"][0]["cell_list"][4] == "167.74"
    assert "Slip bps" not in table_dict["column_list"]  # No verified comparison source.


@pytest.mark.parametrize("missing_str", ["reconcile", "quantities", "before", "duplicate_order"])
def test_missing_or_ambiguous_evidence_never_manufactures_values(pod_fixture_tuple, missing_str):
    source_dict = pod_fixture_tuple[3]
    if missing_str == "reconcile":
        source_dict["reconciliation_dict"] = {}
    elif missing_str == "quantities":
        source_dict["cycle_evidence_dict"] = {}
    elif missing_str == "before":
        source_dict["vplan_dict"] = {}
    else:
        source_dict["order_list"].append(deepcopy(source_dict["order_list"][0]))
    table_dict = build_evidence_tables_dict(source_dict, as_of_ts=DEMO_NOW_TS, fresh_bool=True)["plan"]
    if missing_str == "reconcile":
        assert {row_dict["match_str"] for row_dict in table_dict["row_list"]} == {"unk"}
        assert {row_dict["cell_list"][5] for row_dict in table_dict["row_list"]} == {"—"}
    else:
        index_int = 1 if missing_str == "before" else 3
        assert table_dict["row_list"][0]["cell_list"][index_int] == "—"


def test_stale_cycle_never_keeps_green_states(pod_fixture_tuple):
    workspace_dict, snapshot_obj, provider_obj, source_dict, pod_id_str = pod_fixture_tuple
    overview_dict = build_overview_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS + timedelta(seconds=121))
    view_dict = build_pod_page_dict(overview_dict, source_dict, {}, pod_id_str=pod_id_str, as_of_ts=DEMO_NOW_TS + timedelta(seconds=121))
    assert view_dict["pill_str"] == "Unknown"
    assert {step_dict["state_str"] for step_dict in view_dict["step_list"]} == {"Unknown"}
    assert {row_dict["match_str"] for row_dict in view_dict["tables_dict"]["plan"]["row_list"]} == {"unk"}
    assert {row_dict["cell_list"][2] for row_dict in view_dict["tables_dict"]["orders"]["row_list"]} == {"Unknown"}


@pytest.mark.parametrize("tab_str", ["plan", "decision", "orders", "fills", "reconcile", "events", "files"])
def test_pod_tabs_and_refresh_are_real_read_only_routes(pod_fixture_tuple, tab_str):
    pod_id_str = pod_fixture_tuple[4]
    app_obj = create_demo_app()
    response_obj = app_obj.test_client().get(f"/pods/{pod_id_str}?tab={tab_str}")
    html_str = response_obj.get_data(as_text=True)
    assert response_obj.status_code == 200
    assert 'aria-label="Seven cycle steps"' in html_str and html_str.count('class="step ') == 7
    assert html_str.count('hx-get="') == 1
    assert 'href="/pods/' in html_str
    assert pod_fixture_tuple[3]["pod_row_dict"]["account_route_str"] not in html_str
    assert '09:30:02' in html_str
    assert app_obj.test_client().get(f"/pods/{pod_id_str}/refresh?tab={tab_str}").status_code == 200
    assert app_obj.test_client().post(f"/pods/{pod_id_str}?tab={tab_str}").status_code == 403


def test_history_navigation_changes_evidence_and_survives_poll(pod_fixture_tuple):
    pod_id_str = pod_fixture_tuple[4]
    app_obj = create_demo_app()
    html_str = app_obj.test_client().get(f"/pods/{pod_id_str}?cycle=vplan:1&tab=fills").get_data(as_text=True)
    assert "2026-09-04 · Open" in html_str
    assert "09-04 09:30:02" in html_str
    assert "cycle=vplan:1" in html_str and "tab=fills" in html_str
    assert 'aria-label="Next cycle"' in html_str
    assert app_obj.test_client().get(f"/pods/{pod_id_str}?cycle=vplan:999").status_code == 404


def test_issue_defaults_to_orders_and_preserves_ack_warning(pod_fixture_tuple):
    pod_id_str = pod_fixture_tuple[0]["operations_account_list"][1]["pod_id"]
    html_str = create_demo_app().test_client().get(f"/pods/{pod_id_str}").get_data(as_text=True)
    assert 'aria-label="Pod needs attention"' in html_str
    assert "No ack" in html_str and "Do not resubmit blindly." in html_str
    assert "Next: Review broker ACK" in html_str and "Next: Submit" not in html_str
    assert "Tools for this step" not in html_str
    assert 'class="step is-fail is-sel"' in html_str


@pytest.mark.parametrize("query_str", ["mode=paper", "tab=raw", "cycle=1", "cycle=vplan:0", "cycle=vplan:-1", "cycle=vplan:2&cycle=vplan:1", "tab=plan&tab=fills", "period=3M&period=All", "path=C:/secret"])
def test_invalid_pod_queries_rejected_before_acquisition(pod_fixture_tuple, query_str):
    response_obj = create_demo_app().test_client().get(f"/pods/{pod_fixture_tuple[4]}?{query_str}")
    assert response_obj.status_code == 400


def test_unknown_pod_is_not_a_path_or_non_live_fallback():
    app_obj = create_demo_app()
    assert app_obj.test_client().get("/pods/foreign_account").status_code == 404
    assert app_obj.test_client().get("/pods/..%2fconfig.env").status_code == 404


def test_overview_pod_links_work(pod_fixture_tuple):
    html_str = create_demo_app().test_client().get("/").get_data(as_text=True)
    assert f'/pods/{pod_fixture_tuple[4]}' in html_str
    assert 'Pod page is not available yet' not in html_str


@pytest.mark.parametrize("field_str,value_obj,expected_str", [("ack_status_str", "missing_critical", "No ack"), ("asset_str", "OTHER", "Unknown"), ("broker_order_id_str", "other-order", "Unknown")])
def test_conflicting_ack_is_not_acked(pod_fixture_tuple, field_str, value_obj, expected_str):
    source_dict = pod_fixture_tuple[3]
    source_dict["ack_list"][0][field_str] = value_obj
    table_dict = build_evidence_tables_dict(source_dict, as_of_ts=DEMO_NOW_TS, fresh_bool=True)["orders"]
    assert table_dict["row_list"][0]["cell_list"][2] == expected_str


def test_old_selected_assessment_clears_table_status_too(pod_fixture_tuple):
    workspace_dict, snapshot_obj, provider_obj, source_dict, pod_id_str = pod_fixture_tuple
    source_dict["pod_row_dict"]["as_of_timestamp_str"] = (DEMO_NOW_TS - timedelta(seconds=121)).isoformat()
    overview_dict = build_overview_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS)
    view_dict = build_pod_page_dict(overview_dict, source_dict, {}, pod_id_str=pod_id_str, as_of_ts=DEMO_NOW_TS)
    assert not view_dict["fresh_bool"]
    assert view_dict["tables_dict"]["orders"]["row_list"][0]["cell_list"][2] == "Unknown"
    assert view_dict["tables_dict"]["plan"]["row_list"][0]["match_str"] == "unk"


@pytest.mark.parametrize("query_str", ["", "?cycle=vplan:2&tab=orders", "?cycle=vplan:2&period=All"])
def test_current_cycle_failure_survives_tab_and_period_navigation(pod_fixture_tuple, query_str):
    workspace_dict, snapshot_obj, provider_obj, _, pod_id_str = pod_fixture_tuple
    workspace_dict["summary_dict"]["pod_row_dict_list"][0]["reconcile_read_failure_dict"] = {"timestamp_str": DEMO_NOW_TS.isoformat(), "error_str": "Read failed"}
    app_obj = create_app(provider_obj, demo_bool=True, workspace_snapshot_fn=lambda: (deepcopy(workspace_dict), snapshot_obj), now_fn=lambda: DEMO_NOW_TS)
    html_str = app_obj.test_client().get(f"/pods/{pod_id_str}{query_str}").get_data(as_text=True)
    assert "Broker read failed" in html_str


def test_old_cycle_does_not_borrow_current_failure(pod_fixture_tuple):
    workspace_dict, snapshot_obj, provider_obj, _, pod_id_str = pod_fixture_tuple
    workspace_dict["summary_dict"]["pod_row_dict_list"][0]["reconcile_read_failure_dict"] = {"timestamp_str": DEMO_NOW_TS.isoformat(), "error_str": "Read failed"}
    app_obj = create_app(provider_obj, demo_bool=True, workspace_snapshot_fn=lambda: (deepcopy(workspace_dict), snapshot_obj), now_fn=lambda: DEMO_NOW_TS)
    html_str = app_obj.test_client().get(f"/pods/{pod_id_str}?cycle=vplan:1").get_data(as_text=True)
    assert "Broker read failed" in html_str  # The current Pod warning stays visible.
    cycle_html_str = html_str.split('aria-label="Selected cycle"')[1].split('</section>')[0]
    assert "Broker read failed" not in cycle_html_str and "Saved cycle" in cycle_html_str


def test_slow_detail_read_cannot_extend_fresh_source_lifetime(pod_fixture_tuple, monkeypatch):
    workspace_dict, snapshot_obj, provider_obj, _, pod_id_str = pod_fixture_tuple
    clock_list = [DEMO_NOW_TS]
    original_fn = provider_obj.get_pod_cycles_dict
    def delayed_source(pod_id_str, **options_dict):
        source_dict = original_fn(pod_id_str, **options_dict)
        clock_list[0] += timedelta(seconds=121)
        return source_dict
    monkeypatch.setattr(provider_obj, "get_pod_cycles_dict", delayed_source)
    app_obj = create_app(provider_obj, demo_bool=True, workspace_snapshot_fn=lambda: (deepcopy(workspace_dict), snapshot_obj), now_fn=lambda: clock_list[0])
    response_obj = app_obj.test_client().get(f"/pods/{pod_id_str}")
    html_str = response_obj.get_data(as_text=True)
    assert response_obj.status_code == 200
    assert 'data-source-valid-ms="0"' in html_str and "Status out of date" in html_str
    assert "Acked" not in html_str


def test_malformed_unrelated_summary_row_does_not_break_pod(pod_fixture_tuple):
    workspace_dict, snapshot_obj, provider_obj, _, pod_id_str = pod_fixture_tuple
    workspace_dict["summary_dict"]["pod_row_dict_list"].append(None)
    app_obj = create_app(provider_obj, demo_bool=True, workspace_snapshot_fn=lambda: (deepcopy(workspace_dict), snapshot_obj), now_fn=lambda: DEMO_NOW_TS)
    response_obj = app_obj.test_client().get(f"/pods/{pod_id_str}")
    assert response_obj.status_code == 200
    assert "Financial data unavailable" in response_obj.get_data(as_text=True)


@pytest.mark.parametrize("conflict_str", ["asset", "order_id", "request", "response", "timestamp", "duplicate_ack", "duplicate_order", "missing_ack", "missing_order", "extra_ack", "order_asset", "shared_order_id", "order_type"])
def test_inconsistent_detailed_ack_cannot_leave_cycle_green(pod_fixture_tuple, conflict_str):
    workspace_dict, snapshot_obj, provider_obj, source_dict, pod_id_str = pod_fixture_tuple
    ack_dict, order_dict = source_dict["ack_list"][0], source_dict["order_list"][0]
    if conflict_str == "asset":
        ack_dict["asset_str"] = "OTHER"
    elif conflict_str == "order_id":
        ack_dict["broker_order_id_str"] = "OTHER"
    elif conflict_str == "request":
        ack_dict["order_request_key_str"] = "OTHER"
    elif conflict_str == "response":
        ack_dict["broker_response_ack_bool"] = False
    elif conflict_str == "timestamp":
        ack_dict["response_timestamp_str"] = (DEMO_NOW_TS + timedelta(seconds=1)).isoformat()
    elif conflict_str == "duplicate_ack":
        source_dict["ack_list"].append(deepcopy(ack_dict))
    elif conflict_str == "duplicate_order":
        source_dict["order_list"].append(deepcopy(order_dict))
    elif conflict_str == "missing_ack":
        source_dict["ack_list"] = source_dict["ack_list"][1:]
    elif conflict_str == "missing_order":
        source_dict["order_list"] = source_dict["order_list"][1:]
    elif conflict_str == "extra_ack":
        source_dict["ack_list"].append({**ack_dict, "order_request_key_str": "OTHER"})
    elif conflict_str == "order_asset":
        order_dict["asset_str"] = "OTHER"
    elif conflict_str == "shared_order_id":
        source_dict["order_list"][1]["broker_order_id_str"] = order_dict["broker_order_id_str"]
    else:
        source_dict["plan_row_list"][0]["broker_order_type_str"] = "MOO"
        order_dict["broker_order_type_str"] = ack_dict["broker_order_type_str"] = "MOC"
    before_dict = deepcopy(source_dict)
    overview_dict = build_overview_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS)
    view_dict = build_pod_page_dict(overview_dict, source_dict, {}, pod_id_str=pod_id_str, as_of_ts=DEMO_NOW_TS)
    assert view_dict["step_list"][3]["state_str"] == "Unknown"
    assert view_dict["pill_str"] == "Unknown"
    assert source_dict == before_dict


def test_detailed_critical_ack_overrides_complete_summary_and_no_order_claim(pod_fixture_tuple):
    workspace_dict, snapshot_obj, provider_obj, source_dict, pod_id_str = pod_fixture_tuple
    source_dict["ack_list"][0]["ack_status_str"] = "missing_critical"
    source_dict["cycle_evidence_dict"]["state_str"] = "no_orders"
    overview_dict = build_overview_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS)
    view_dict = build_pod_page_dict(overview_dict, source_dict, {}, pod_id_str=pod_id_str, as_of_ts=DEMO_NOW_TS)
    assert view_dict["step_list"][3]["state_str"] == "Failed"
    assert view_dict["pill_str"] == "Action needed"
    assert view_dict["issue_title_str"] == "Review broker ACK"
    assert view_dict["tables_dict"]["orders"]["row_list"][0]["cell_list"][2] == "No ack"


def test_upcoming_unsubmitted_plan_keeps_waiting_without_ack_alarm(pod_fixture_tuple):
    workspace_dict, snapshot_obj, provider_obj, source_dict, pod_id_str = pod_fixture_tuple
    row_dict = source_dict["pod_row_dict"]
    row_dict.update(latest_vplan_status_str="ready", latest_submit_ack_status_str="not_checked",
        broker_order_count_int=0, broker_ack_count_int=0, missing_ack_count_int=0,
        latest_vplan_submission_timestamp_str=(DEMO_NOW_TS + timedelta(minutes=5)).isoformat(),
        latest_vplan_target_execution_timestamp_str=(DEMO_NOW_TS + timedelta(minutes=10)).isoformat())
    source_dict["order_list"], source_dict["ack_list"], source_dict["cycle_evidence_dict"] = [], [], {}
    overview_dict = build_overview_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS)
    view_dict = build_pod_page_dict(overview_dict, source_dict, {}, pod_id_str=pod_id_str, as_of_ts=DEMO_NOW_TS)
    assert view_dict["step_list"][3]["state_str"] == "Planned"
    assert view_dict["step_list"][3]["fact_str"] == "Waiting to submit"
    assert not view_dict["issue_bool"]


@pytest.mark.parametrize("explicit_bool", [True, False])
def test_historical_failed_cycle_wording_preserves_default_unresolved_warning(pod_fixture_tuple, explicit_bool):
    workspace_dict, snapshot_obj, provider_obj, source_dict, pod_id_str = pod_fixture_tuple
    source_dict["selected_cycle_dict"].update(current_bool=False, unresolved_bool=True, session_date_str="2026-09-04")
    source_dict["selected_explicit_bool"] = explicit_bool
    source_dict["ack_list"][0]["ack_status_str"] = "missing_critical"
    overview_dict = build_overview_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS)
    view_dict = build_pod_page_dict(overview_dict, source_dict, {}, pod_id_str=pod_id_str, as_of_ts=DEMO_NOW_TS)
    assert view_dict["issue_bool"]
    if explicit_bool:
        assert view_dict["verdict_str"].startswith("Saved cycle:")
        assert "Next:" not in view_dict["verdict_detail_str"]
        assert view_dict["issue_title_str"] == "Saved cycle · Missing broker ACK"
        assert view_dict["issue_detail_str"] == "Review the saved records for 2026-09-04."
    else:
        assert not view_dict["verdict_str"].startswith("Saved cycle:")
        assert view_dict["issue_title_str"] == "Review broker ACK"
        assert "Do not resubmit blindly." in view_dict["issue_detail_str"]
