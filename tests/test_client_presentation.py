"""UI facts stay scoped and honest when sources are incomplete or change."""
from copy import deepcopy
from decimal import Decimal
from types import SimpleNamespace
import sqlite3

import pytest

from alpha.live.dashboard_v3.client_charts import nav_chart_dict
from alpha.live.dashboard_v3.client_presentation import (
    allocation_dict, portfolio_allocation_dict, holdings_allocation_dict, flow_dict, saved_stage_table_dict,
)


def test_nav_only_scope_still_has_full_end_date_allocation():
    report_dict = {"closing_date_str": "2026-09-03", "closing_nav_float": 200, "scope_complete_bool": False,
        "valuation_account_list": [{"account_route": route_str, "display_name": "Same name"} for route_str in ("A", "B")]}
    snapshot_obj = SimpleNamespace(row_tuple=tuple(SimpleNamespace(account_route_str=route_str,
        market_date_str="2026-09-03", closing_nav_decimal=Decimal(100)) for route_str in ("A", "B")))
    result_dict = portfolio_allocation_dict(report_dict, snapshot_obj)
    assert [item_dict["weight_float"] for item_dict in result_dict["item_list"]] == [.5, .5]
    assert [item_dict["detail_str"] for item_dict in result_dict["item_list"]] == ["A", "B"]
    for bad_tuple in (snapshot_obj.row_tuple[:1], snapshot_obj.row_tuple + snapshot_obj.row_tuple[:1]):
        assert not portfolio_allocation_dict(report_dict, SimpleNamespace(row_tuple=bad_tuple))["item_list"]


def test_exited_strategy_does_not_contribute_earlier_nav_to_closing_pie():
    report_dict = {"closing_date_str": "2026-09-03", "closing_nav_float": 100, "scope_complete_bool": True,
        "strategy_list": [{"account_route_str": "A", "display_name_str": "Active", "to_date_str": "2026-09-03"},
                          {"account_route_str": "B", "display_name_str": "Exited", "to_date_str": "2026-09-02"}]}
    snapshot_obj = SimpleNamespace(row_tuple=(SimpleNamespace(account_route_str="A", market_date_str="2026-09-03", closing_nav_decimal=Decimal(100)),
        SimpleNamespace(account_route_str="B", market_date_str="2026-09-02", closing_nav_decimal=Decimal(300))))
    assert [item_dict["label_str"] for item_dict in portfolio_allocation_dict(report_dict, snapshot_obj)["item_list"]] == ["Active"]
    report_dict["closing_nav_float"] = 400
    assert not portfolio_allocation_dict(report_dict, snapshot_obj)["item_list"]


@pytest.mark.parametrize("value_obj", [None, True, float("nan"), float("inf"), -1])
def test_invalid_pie_value_never_becomes_an_excluded_zero(value_obj):
    assert not allocation_dict([{"value_float": 100}, {"value_float": value_obj}])["item_list"]


def test_holdings_keep_shorts_unpriced_and_reference_dates_separate():
    evidence_dict = {"position_exposure_dict_list": [
        {"asset_str": "LONG", "share_float": 4, "price_float": 10},
        {"asset_str": "SHORT", "share_float": -1, "price_float": 10},
        {"asset_str": "MISSING", "share_float": 1, "price_float": None},
        {"asset_str": "BOOL", "share_float": True, "price_float": 10},
        {"asset_str": "OVERFLOW", "share_float": 1e308, "price_float": 1e308}],
        "latest_pod_state_timestamp_str": "2026-09-03T20:00:00Z", "latest_live_reference_snapshot_timestamp_str": "2026-09-01T20:00:00Z"}
    result_dict = holdings_allocation_dict(evidence_dict)
    assert [item_dict["weight_float"] for item_dict in result_dict["item_list"]] == [.8, .2]
    assert result_dict["item_list"][1]["signed_value_float"] == -10
    assert result_dict["missing_list"] == ["MISSING", "BOOL", "OVERFLOW"]
    assert result_dict["position_timestamp_str"] != result_dict["price_timestamp_str"]


def test_flow_blocker_wins_and_unknown_or_stale_never_gets_focus():
    evidence_dict = {"next_action_str": "post_execution_reconcile", "required_action_dict": {"label_str": "Review broker ACK"},
        "lifecycle_step_dict_list": [{"step_key_str": "ack", "severity_str": "red"}, {"step_key_str": "reconcile", "severity_str": "yellow"}]}
    assert [step_dict["step_key_str"] for step_dict in flow_dict(evidence_dict, source_fresh_bool=True)["step_list"] if step_dict["focus_bool"]] == ["ack"]
    assert not any(step_dict["focus_bool"] for step_dict in flow_dict(evidence_dict, source_fresh_bool=False)["step_list"])
    assert not flow_dict({"lifecycle_step_dict_list": [{"label_str": "ACK"}]}, source_fresh_bool=True)["step_list"][0]["focus_bool"]


@pytest.mark.parametrize("action_str,step_str", [("expire_stale", "decision"), ("missed_decision_cycle", "decision"), ("build_vplan", "vplan")])
def test_pending_action_is_located_without_relabeling_previous_evidence(action_str, step_str):
    result_dict = flow_dict({"next_action_str": action_str, "latest_vplan_cycle_role_str": "previous",
        "lifecycle_step_dict_list": [{"step_key_str": step_str, "status_str": "complete", "severity_str": "green"}]}, source_fresh_bool=True)
    assert result_dict["step_list"][0]["focus_bool"]
    assert result_dict["step_list"][0]["previous_bool"] is (step_str == "vplan")
    assert result_dict["step_list"][0]["status_str"] == "complete"


def test_chart_daily_dollars_are_exact_date_and_sod_does_not_get_future_pnl():
    chart_dict = nav_chart_dict([{"market_date_str": "2026-09-01 SOD", "nav_float": 100},
        {"market_date_str": "2026-09-01", "nav_float": 171}, {"market_date_str": "2026-09-02", "nav_float": 180}],
        daily_fact_list=[{"market_date_str": "2026-09-01", "pnl_float": -29}])
    assert [point_dict["pnl_label_str"] for point_dict in chart_dict["point_list"]] == ["Unavailable", "-$29.00", "Unavailable"]


def detail_fixture_tuple():
    evidence_dict = {"pod_id_str": "pod", "account_route_str": "A", "release_id_str": "r", "mode_str": "live",
        "latest_pod_state_timestamp_str": "2026-09-01T20:00:00Z", "latest_decision_plan_id_int": 1, "latest_vplan_id_int": 2,
        "broker_ack_count_int": 2, "fill_count_int": 0}
    identity_dict = {key_str: evidence_dict[key_str] for key_str in ("pod_id_str", "account_route_str", "release_id_str")}
    detail_dict = {"pod_row_dict": deepcopy(evidence_dict), "latest_decision_plan_dict": {**identity_dict,
        "decision_plan_id_int": 1, "signal_timestamp_str": "2026-09-01T20:00:00Z", "display_target_weight_map_dict": {"ABC": .5}},
        "latest_vplan_dict": {**identity_dict, "vplan_id_int": 2, "signal_timestamp_str": "2026-09-01T20:00:00Z",
            "broker_ack_row_dict_list": [{"vplan_id_int": 2, "account_route_str": "A", "asset_str": "ABC", "ack_status_str": "complete", "raw_json_str": "SECRET"},
                {"vplan_id_int": 2, "account_route_str": "OTHER", "asset_str": "PRIVATE"}]}}
    return detail_dict, {"evidence_dict": evidence_dict, "effective_from_str": "2026-09-01"}


def test_detail_projection_excludes_other_accounts_and_raw_payloads():
    detail_dict, strategy_dict = detail_fixture_tuple()
    result_dict = saved_stage_table_dict(detail_dict, strategy_dict)
    assert result_dict["decision"][0]["row_list"] == [["ABC", 50]]
    assert result_dict["ack"][0]["row_list"] == [["ABC", "complete", None, None]]
    assert "SECRET" not in str(result_dict) and "PRIVATE" not in str(result_dict)
    detail_dict["latest_vplan_dict"]["account_route_str"] = "OTHER"
    assert "ack" not in saved_stage_table_dict(detail_dict, strategy_dict)


@pytest.mark.parametrize("field_str,value_obj", [("account_route_str", "OTHER"), ("latest_vplan_id_int", 4), ("latest_vplan_status_str", "changed"), ("fill_count_int", 5)])
def test_detail_assessment_race_does_not_mix_new_tables_with_old_flow(field_str, value_obj):
    detail_dict, strategy_dict = detail_fixture_tuple()
    detail_dict["pod_row_dict"][field_str] = value_obj
    assert saved_stage_table_dict(detail_dict, strategy_dict) == {}


def test_detail_before_current_mandate_does_not_leak_targets():
    detail_dict, strategy_dict = detail_fixture_tuple()
    strategy_dict["effective_from_str"] = "2026-09-02"
    assert saved_stage_table_dict(detail_dict, strategy_dict) == {}


def test_actual_detail_plan_and_child_read_must_match_saved_assessment():
    detail_dict, strategy_dict = detail_fixture_tuple()
    detail_dict["latest_vplan_dict"]["status_str"] = "submitted"
    assert "ack" not in saved_stage_table_dict(detail_dict, strategy_dict)
    detail_dict["latest_vplan_dict"].pop("status_str")
    detail_dict["latest_vplan_dict"]["fill_row_dict_list"] = [{"account_route_str": "A", "vplan_id_int": 2}]
    assert "ack" not in saved_stage_table_dict(detail_dict, strategy_dict)


def test_reconciliation_uses_bound_persisted_snapshot_never_latest_cache():
    detail_dict, strategy_dict = detail_fixture_tuple()
    for evidence_dict in (strategy_dict["evidence_dict"], detail_dict["pod_row_dict"]):
        evidence_dict.update(latest_reconciliation_timestamp_str="2026-09-02T15:00:00Z", latest_reconciliation_status_str="passed")
    detail_dict["latest_execution_report_dict"] = {"pod_id_str": "pod", "latest_vplan_id_int": 2,
        "execution_row_dict_list": [{"asset_str": "WRONG_CACHE", "broker_share_float": 500}]}
    assert "reconcile" not in saved_stage_table_dict(detail_dict, strategy_dict)
    detail_dict["latest_reconciliation_dict"] = {"pod_id_str": "pod", "vplan_id_int": 2, "stage_str": "post_execution", "status_str": "passed",
        "created_timestamp_str": "2026-09-02T15:00:00Z", "model_position_json_str": '{"ABC": 10}', "broker_position_json_str": '{"ABC": 10}'}
    result_dict = saved_stage_table_dict(detail_dict, strategy_dict)
    assert result_dict["reconcile"][0]["row_list"] == [["ABC", 10, 10]]
    assert "WRONG_CACHE" not in str(result_dict)
    detail_dict["latest_reconciliation_dict"]["created_timestamp_str"] = "2026-09-03T15:00:00Z"
    assert "reconcile" not in saved_stage_table_dict(detail_dict, strategy_dict)


def test_previous_cycle_review_is_not_labeled_new_cycle():
    from flask import render_template
    from alpha.live.dashboard_v3.app import create_app
    from test_dashboard_operator_access import ForbiddenProvider
    evidence_dict = {"next_action_str": "wait", "required_action_dict": {"label_str": "Review previous ACK"},
        "latest_vplan_cycle_role_str": "previous", "lifecycle_step_dict_list": [{"step_key_str": "ack", "label_str": "ACK", "severity_str": "red", "status_str": "missing"}]}
    strategy_dict = {"pod_id_str": "pod", "account_route_str": "A", "display_name_str": "Example", "severity_str": "red", "evidence_dict": evidence_dict,
        "flow_dict": flow_dict(evidence_dict, source_fresh_bool=True), "allocation_dict": holdings_allocation_dict(evidence_dict), "stage_table_dict": {}}
    with create_app(ForbiddenProvider(), read_only_bool=True).test_request_context():
        html_str = render_template('_client_strategy.html', strategy_dict=strategy_dict)
    assert "Action here" in html_str and "Previous cycle evidence" in html_str
    assert "New cycle action" not in html_str


def test_optional_sqlite_detail_failure_preserves_strategy_page(monkeypatch):
    from alpha.live.dashboard_v3.app import create_app
    from test_client_operations import fixture_tuple
    registry_dict, _, provider_obj, _ = fixture_tuple()
    def fail_fn(*arg_tuple):
        raise sqlite3.OperationalError("locked")
    monkeypatch.setattr(provider_obj, "get_pod_detail_dict", fail_fn)
    response_obj = create_app(provider_obj, read_only_bool=True, client_registry_dict={**registry_dict, "clients": registry_dict["clients"][:1]}).test_client().get("/clients/demo-owner/strategies")
    assert response_obj.status_code == 200
    assert b"Data freshness" in response_obj.data and b"Execution flow" in response_obj.data


def test_snapshot_client_never_reads_local_detail(monkeypatch):
    from alpha.live.dashboard_v3.app import create_app
    from alpha.live.dashboard_v3.client_views import _strategy_display_list
    from test_client_operations import fixture_tuple
    registry_dict, _, provider_obj, _ = fixture_tuple()
    def forbidden_fn(*arg_tuple):
        raise AssertionError("Snapshot client must not read local detail")
    monkeypatch.setattr(provider_obj, "get_pod_detail_dict", forbidden_fn)
    app_obj = create_app(provider_obj, read_only_bool=True, client_registry_dict={**registry_dict, "clients": registry_dict["clients"][:1]})
    with app_obj.test_request_context():
        result_list = _strategy_display_list({"source_str": "snapshot", "source_fresh_bool": True,
            "strategy_list": [{"pod_id_str": "pod", "matched_bool": True, "evidence_dict": {}}]})
    assert result_list[0]["stage_table_dict"] == {}


def test_financial_source_failure_keeps_operational_schedule(monkeypatch):
    from alpha.live.client_reporting import ClientReportingError
    from alpha.live.dashboard_v3.app import create_app
    from test_client_operations import fixture_tuple
    registry_dict, _, provider_obj, _ = fixture_tuple()
    def fail_fn(*arg_tuple):
        raise ClientReportingError("Synthetic unavailable finance source")
    monkeypatch.setattr("alpha.live.dashboard_v3.client_views._snapshot_obj", fail_fn)
    response_obj = create_app(provider_obj, read_only_bool=True, client_registry_dict={**registry_dict, "clients": registry_dict["clients"][:1]}).test_client().get("/clients/demo-owner/overview")
    assert response_obj.status_code == 200
    assert b"Financial evidence unavailable" in response_obj.data
    assert response_obj.data.count(b"client-schedule-card") == 2
