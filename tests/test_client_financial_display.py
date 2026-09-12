"""Synthetic presentation regressions; financial formulas and raw facts stay fixed."""

from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import re

import pytest

from alpha.live.dashboard_v3.app import create_app
from alpha.live.dashboard_v3.client_charts import daily_history_list, nav_chart_dict
from alpha.live.dashboard_v3.client_financial_display import capital_day_key_set, financial_dates_dict, summarized_issue_list
from alpha.live.dashboard_v3.client_views import _period_tuple
from test_client_reporting import snapshot_obj, report_dict
from test_dashboard_operator_access import ForbiddenProvider
from test_dashboard_local_workspace import build_fixture_app, file_snapshot_dict
from test_ibkr_nav_profile import expanded_config_dict, expanded_nav_attributes_dict


def row_list(date_str, **override_dict):
    return [expanded_nav_attributes_dict(account_str, date_str=date_str, **override_dict)
        for account_str in ("U_TEST_A", "U_TEST_B")]


def dates_dict(attribute_list, *, as_of_str="2026-09-04T15:00:00+00:00", local_bool=True):
    config_dict = expanded_config_dict(True)
    return financial_dates_dict(config_dict, snapshot_obj(attribute_list), as_of_ts=datetime.fromisoformat(as_of_str),
        valuation_account_list=config_dict["accounts"] if local_bool else None)


def freeze_clock(monkeypatch, as_of_str="2026-09-04T15:00:00+00:00"):
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.fromisoformat(as_of_str)
    monkeypatch.setattr("alpha.live.dashboard_v3.client_views.datetime", FixedDatetime)


def test_latest_complete_date_and_newer_partial_day_are_distinct():
    result_dict = dates_dict(row_list("2026-09-02") + row_list("2026-09-03")[:1])
    assert result_dict == {"latest_complete_str": "2026-09-02", "expected_str": "2026-09-03", "delayed_bool": True}


@pytest.mark.parametrize("override_dict", [{"billPay": ""}, {"brokerFees": "1"}, {"linkingAdjustments": "1"},
    {"internalCashTransfers": "-100"}, {"opening_str": "999"}])
def test_endpoint_never_selects_by_accounting_success(override_dict):
    result_dict = dates_dict(row_list("2026-09-02") + row_list("2026-09-03", **override_dict))
    assert result_dict["latest_complete_str"] == "2026-09-03"
    assert not result_dict["delayed_bool"]


@pytest.mark.parametrize("complete_bool", [True, False])
def test_weekend_source_activity_is_not_lost_behind_exchange_calendar(complete_bool):
    attribute_list = row_list("2026-09-04") + row_list("2026-09-06")[:2 if complete_bool else 1]
    result_dict = dates_dict(attribute_list, as_of_str="2026-09-08T15:00:00+00:00")
    assert result_dict["latest_complete_str"] == ("2026-09-06" if complete_bool else "2026-09-04")
    assert result_dict["expected_str"] == "2026-09-06"
    assert result_dict["delayed_bool"] is not complete_bool


def test_holiday_today_future_and_new_york_midnight_boundaries():
    attribute_list = row_list("2026-09-04") + row_list("2026-09-08") + row_list("2026-09-09")
    assert dates_dict(attribute_list, as_of_str="2026-09-08T15:00:00+00:00") == {
        "latest_complete_str": "2026-09-04", "expected_str": "2026-09-04", "delayed_bool": False}
    assert dates_dict(row_list("2026-09-02") + row_list("2026-09-03"), as_of_str="2026-09-04T02:00:00+00:00")["latest_complete_str"] == "2026-09-02"


def test_local_all_identities_are_required_but_explicit_ownership_is_respected():
    config_dict = expanded_config_dict(True)
    config_dict["accounts"][1]["effective_from"] = "2026-09-02"
    source_obj = snapshot_obj(row_list("2026-09-01")[:1])
    as_of_ts = datetime(2026, 9, 4, tzinfo=UTC)
    assert financial_dates_dict(config_dict, source_obj, as_of_ts=as_of_ts)["latest_complete_str"] == "2026-09-01"
    assert financial_dates_dict(config_dict, source_obj, as_of_ts=as_of_ts, valuation_account_list=config_dict["accounts"])["latest_complete_str"] is None
    assert financial_dates_dict(config_dict, replace(source_obj, unavailable_reason_str="Invalid import"), as_of_ts=as_of_ts)["latest_complete_str"] is None
    assert dates_dict([])["latest_complete_str"] is None


@pytest.mark.parametrize("window_str,as_of_str,expected_start_str", [
    ("mtd", "2026-10-05T15:00:00+00:00", "2026-10-01"),
    ("ytd", "2027-01-05T15:00:00+00:00", "2027-01-01"),
    ("1w", "2026-10-05T15:00:00+00:00", "2026-09-28"),
])
def test_stale_endpoint_does_not_turn_current_preset_into_an_old_period(window_str, as_of_str, expected_start_str):
    app_obj = create_app(ForbiddenProvider())
    with app_obj.test_request_context("/?window=" + window_str):
        assert _period_tuple(expanded_config_dict(), datetime.fromisoformat(as_of_str), default_end_str="2026-09-03") == (expected_start_str, (datetime.fromisoformat(as_of_str).date() - timedelta(days=1)).isoformat())
    with app_obj.test_request_context("/?from=2026-09-01&to=2026-09-08"):
        assert _period_tuple(expanded_config_dict(), datetime.fromisoformat(as_of_str), default_end_str="2026-09-03") == ("2026-09-01", "2026-09-08")


def test_default_route_uses_common_date_explicit_dates_do_not_and_hash_is_shared(monkeypatch):
    freeze_clock(monkeypatch)
    config_dict = expanded_config_dict(True)
    source_obj = snapshot_obj(row_list("2026-09-01") + row_list("2026-09-02", opening_str="1010", closing_str="1020") + row_list("2026-09-03", opening_str="1020", closing_str="1030")[:1])
    app_obj = create_app(ForbiddenProvider(), read_only_bool=True, client_registry_dict={"schema_version": 1, "clients": [config_dict]}, client_reporting_snapshot_fn=lambda _: source_obj)
    client_obj = app_obj.test_client()
    result_list = [client_obj.get(f"/clients/sample/{view_str}?download=json").get_json() for view_str in ("overview", "performance", "report")]
    assert all(result_dict["requested_to_date_str"] == "2026-09-02" for result_dict in result_list)
    assert len({result_dict["report_hash_str"] for result_dict in result_list}) == 1
    explicit_dict = client_obj.get("/clients/sample/overview?from=2026-09-01&to=2026-09-02&download=json").get_json()
    assert explicit_dict["report_hash_str"] == result_list[0]["report_hash_str"]
    html_str = client_obj.get("/clients/sample/overview").get_data(as_text=True)
    assert "IBKR update pending" not in html_str
    assert 'value="2026-09-02"' in html_str
    incomplete_dict = client_obj.get("/clients/sample/overview?from=2026-09-01&to=2026-09-03&download=json").get_json()
    assert incomplete_dict["requested_to_date_str"] == "2026-09-03" and incomplete_dict["pnl_float"] is None


@pytest.mark.parametrize("missing_middle_bool", [True, False])
def test_good_endpoint_never_erases_prior_gap_or_bad_current_bridge(monkeypatch, missing_middle_bool):
    freeze_clock(monkeypatch)
    attribute_list = row_list("2026-09-01")
    if not missing_middle_bool:
        attribute_list += row_list("2026-09-02", opening_str="1010", closing_str="1020")
    attribute_list += row_list("2026-09-03", opening_str="1020", closing_str="1030", **({} if missing_middle_bool else {"billPay": ""}))
    app_obj = create_app(ForbiddenProvider(), read_only_bool=True, client_registry_dict={"schema_version": 1, "clients": [expanded_config_dict(True)]}, client_reporting_snapshot_fn=lambda _: snapshot_obj(attribute_list))
    result_dict = app_obj.test_client().get("/clients/sample/overview?download=json").get_json()
    assert result_dict["requested_from_date_str"] == "2026-09-01" and result_dict["requested_to_date_str"] == "2026-09-03"
    assert result_dict["pnl_float"] is None and result_dict["twr_float"] is None


@pytest.mark.parametrize("new_pod_bool", [True, False])
def test_real_local_entry_preserves_legacy_history_and_new_pod_unknowns(tmp_path, monkeypatch, new_pod_bool):
    freeze_clock(monkeypatch)
    app_obj = build_fixture_app(tmp_path, monkeypatch, expanded_bool=True, new_pod_bool=new_pod_bool)
    before_dict = file_snapshot_dict(tmp_path)
    client_obj = app_obj.test_client()
    result_dict = client_obj.get("/clients/local/overview?download=json").get_json()
    assert result_dict["requested_from_date_str"] == "2026-09-01" and result_dict["requested_to_date_str"] == "2026-09-03"
    assert result_dict["closing_nav_float"] == (None if new_pod_bool else 11135)
    assert result_dict["pnl_float"] is None and result_dict["twr_float"] is None  # Sep1 is still legacy.
    assert len(client_obj.get("/clients/local/diagnostics?download=json").get_json()["strategy_list"]) == (3 if new_pod_bool else 2)
    assert file_snapshot_dict(tmp_path) == before_dict


def test_repeated_diagnostics_group_by_account_reason_with_dates_and_escape_html(monkeypatch):
    freeze_clock(monkeypatch)
    attribute_list = row_list("2026-09-01", billPay="") + row_list("2026-09-02", opening_str="1010", closing_str="1020") + row_list("2026-09-03", opening_str="1020", closing_str="1030", billPay="")
    config_dict = expanded_config_dict(True)
    config_dict["accounts"][0]["display_name"] = "<Same name>"
    config_dict["accounts"][1]["display_name"] = "<Same name>"
    source_obj = snapshot_obj(attribute_list)
    result_dict = report_dict(attribute_list, config_dict=config_dict, to_str="2026-09-03")
    original_dict = deepcopy(result_dict)
    summary_list = summarized_issue_list(result_dict, [])
    assert len([line_str for line_str in summary_list if "billPay" in line_str]) == 2
    assert all("2 days · 2026-09-01 → 2026-09-03" in line_str for line_str in summary_list if "billPay" in line_str)
    assert any("U_TEST_A" in line_str for line_str in summary_list) and any("U_TEST_B" in line_str for line_str in summary_list)
    assert result_dict == original_dict
    app_obj = create_app(ForbiddenProvider(), read_only_bool=True, client_registry_dict={"schema_version": 1, "clients": [config_dict]}, client_reporting_snapshot_fn=lambda _: source_obj)
    html_str = app_obj.test_client().get("/clients/sample/overview").get_data(as_text=True)
    issues_str = re.search(r'<details class="client-data-issues">(.*?)</details>', html_str, re.S)[1]
    assert issues_str.count("billPay") == 2
    assert "&lt;Same name&gt;" in issues_str and "<Same name>" not in issues_str


@pytest.mark.parametrize("override_dict,expected_bool", [
    ({"depositsWithdrawals": "50", "billPay": "-50"}, True), ({"depositsWithdrawals": "0"}, False),
    ({"depositsWithdrawals": ""}, False), ({"linkingAdjustments": "10"}, False), ({"assetTransfers": "1"}, True),
])
def test_movement_marker_uses_explicit_components_not_nav_or_net_amount(override_dict, expected_bool):
    attribute_list = [expanded_nav_attributes_dict(**override_dict)]
    result_dict = report_dict(attribute_list, config_dict=expanded_config_dict())
    before_dict = deepcopy(result_dict)
    scope_list = daily_history_list(result_dict, movement_key_set=capital_day_key_set(snapshot_obj(attribute_list)))
    assert all(scope_dict["daily_list"][0]["capital_movement_bool"] is expected_bool for scope_dict in scope_list)
    assert result_dict == before_dict


def test_balanced_internal_transfers_mark_portfolio_and_both_accounts():
    attribute_list = [expanded_nav_attributes_dict(closing_str="960", internalCashTransfers="-50"),
        expanded_nav_attributes_dict("U_TEST_B", closing_str="1060", internalCashTransfers="50")]
    result_dict = report_dict(attribute_list, config_dict=expanded_config_dict(True))
    assert result_dict["pnl_float"] == 20 and result_dict["capital_movement_float"] == 0
    scope_list = daily_history_list(result_dict, movement_key_set=capital_day_key_set(snapshot_obj(attribute_list)))
    assert all(scope_dict["daily_list"][0]["capital_movement_bool"] for scope_dict in scope_list)
    app_obj = create_app(ForbiddenProvider(), read_only_bool=True, client_registry_dict={"schema_version": 1, "clients": [expanded_config_dict(True)]}, client_reporting_snapshot_fn=lambda _: snapshot_obj(attribute_list))
    html_str = app_obj.test_client().get("/clients/sample/overview?from=2026-09-01&to=2026-09-01").get_data(as_text=True)
    assert html_str.count('class="client-movement"') == 3


def test_grouping_preserves_distinct_unknown_reasons_and_compacts_missing_nav():
    result_dict = report_dict([expanded_nav_attributes_dict()], config_dict=expanded_config_dict())
    result_dict["strategy_list"][0]["issue_list"] = ["2026-09-01: Missing mtm", "2026-09-03: Missing mtm", "2026-09-02: Missing billPay"]
    result_dict["issue_list"] = ["Strategy A: " + issue_str for issue_str in result_dict["strategy_list"][0]["issue_list"]] + ["Unknown failure: preserve me"]
    notice_list = ["pod_a / U_TEST_A: missing IBKR NAV for 2026-09-01.", "pod_a / U_TEST_A: missing IBKR NAV for 2026-09-03."]
    summary_list = summarized_issue_list(result_dict, notice_list)
    assert len(summary_list) == 4
    assert any("Missing mtm · 2 days · 2026-09-01 → 2026-09-03" in issue_str for issue_str in summary_list)
    assert any("Missing billPay · 1 day · 2026-09-02" in issue_str for issue_str in summary_list)
    assert any("Missing IBKR NAV · 2 days" in issue_str for issue_str in summary_list)
    assert "Unknown failure: preserve me" in summary_list


def test_start_label_does_not_change_canonical_date_or_geometry():
    daily_list = [{"market_date_str": "2026-09-01 SOD", "nav_float": 0}, {"market_date_str": "2026-09-01", "nav_float": .01}]
    before_list = deepcopy(daily_list)
    chart_dict = nav_chart_dict(daily_list, unit_str="pct")
    assert chart_dict["from_str"] == "2026-09-01 SOD" and chart_dict["from_label_str"] == "2026-09-01 · Start"
    assert chart_dict["point_list"][0]["market_date_str"] == "2026-09-01 SOD"
    assert chart_dict["point_list"][0]["x_float"] == 3 and daily_list == before_list
