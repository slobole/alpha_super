"""Monthly client report: the Performance page's numbers for one month, read-only."""

from contextlib import closing
from copy import deepcopy
from datetime import UTC, datetime
from decimal import Decimal
import hashlib
import json
from html import unescape
import re
import sqlite3
from types import SimpleNamespace

import pytest

from alpha.live.client_reporting import BrokerReportingSnapshot, parse_broker_nav_import
from alpha.live.dashboard_v4 import monthly_report
from alpha.live.dashboard_v4.app import create_app
from alpha.live.dashboard_v4.demo import DEMO_NOW_TS, build_demo_workspace_tuple
from alpha.live.dashboard_v4.monthly_report import (
    ASSET_CLASS_TUPLE, _chart_dict, _flow_dict, _money_str, _month_option_list, _page_height_tuple, _percent_str, _year_list, _year_row_list,
    build_monthly_report_dict, load_holder_name_str,
)
from alpha.live.dashboard_v4.performance import build_performance_page_dict
from alpha.live.ibkr_nav_profile import local_ibkr_reporting_profile_dict
from alpha.live.scheduler_utils import get_exchange_calendar_obj
from test_client_reporting import xml_text_str
from test_ibkr_nav_profile import expanded_nav_attributes_dict


SAMPLE_AS_OF_TS = datetime(2026, 9, 4, 13, tzinfo=UTC)


@pytest.fixture
def demo_tuple():
    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple(include_holdings_bool=True)
    try:
        yield workspace_dict, snapshot_obj, provider_obj
    finally:
        provider_obj.close()


def _visible_text_str(html_str):
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", html_str))).strip()


def _money_int(value_str):
    return int(value_str.replace("$", "").replace(",", "").replace("−", "-").replace("+", ""))


def _money_value_dict(report_dict):
    return {row_dict["label_str"]: (row_dict["value_str"], row_dict["pct_str"]) for row_dict in report_dict["money_row_list"]}


def _sample_tuple(*, operations_source_str="local", effective_to_dict=None, last_row_dict=None, extra_event_dict=None):
    """Two Pods and a vault in the production Flex field schema, Aug 3 to Sep 3, 2026.

    effective_to_dict ends an account's period; last_row_dict stops its saved rows
    after a date; extra_event_dict adds raw Flex fields (a non-finite one is kept
    as text and left out of the NAV arithmetic).
    """
    session_list = [session_obj.date().isoformat() for session_obj in
        get_exchange_calendar_obj("XNYS").sessions_in_range("2026-08-03", "2026-09-03")]
    nav_dict = {"U100": Decimal("20000"), "U200": Decimal("10000"), "U900": Decimal("50000")}
    event_dict = {("U100", "2026-08-17"): {"depositsWithdrawals": "1000"},
        ("U100", "2026-08-24"): {"depositsWithdrawals": "-500"}, ("U200", "2026-08-24"): {"depositsWithdrawals": "500"},
        ("U200", "2026-08-10"): {"commissions": "-1.25"},
        ("U100", "2026-08-20"): {"dividends": "10", "withholdingTax": "-2.5"},
        ("U200", "2026-08-31"): {"interest": "0.3"}, ("U900", "2026-08-31"): {"mtm": "100"}}
    for key_tuple, field_dict in (extra_event_dict or {}).items():
        event_dict[key_tuple] = {**event_dict.get(key_tuple, {}), **field_dict}
    attribute_list = []
    for index_int, date_str in enumerate(session_list):
        for route_str in ("U100", "U200", "U900"):
            if date_str > (last_row_dict or {}).get(route_str, "9999-12-31"):
                continue
            override_dict = {"mtm": "0" if route_str == "U900" else str((index_int % 5 - 1) * (12 if route_str == "U100" else -9))}
            override_dict.update(event_dict.get((route_str, date_str), {}))
            capital_decimal = Decimal(override_dict.get("depositsWithdrawals", "0"))
            economic_decimal = sum((value_decimal for value_decimal in (Decimal(override_dict.get(field_str, "0"))
                for field_str in ("mtm", "commissions", "otherFees", "dividends", "withholdingTax", "interest")) if value_decimal.is_finite()), Decimal(0))
            opening_decimal = nav_dict[route_str]
            nav_dict[route_str] = opening_decimal + capital_decimal + economic_decimal
            attribute_list.append(expanded_nav_attributes_dict(route_str, date_str=date_str, opening_str=str(opening_decimal),
                closing_str=str(nav_dict[route_str]), twr_str=str(economic_decimal / opening_decimal * 100), **override_dict))
    raw_xml_str = xml_text_str(attribute_list)
    checksum_str = hashlib.sha256(raw_xml_str.encode()).hexdigest()
    row_list = parse_broker_nav_import(raw_xml_str, allowed_account_set={"U100", "U200", "U900"},
        query_name_str="TEST_NAV", source_import_id_int=1, source_checksum_str=checksum_str)
    snapshot_obj = BrokerReportingSnapshot(tuple(row_list), ({"import_id_int": 1, "checksum_str": checksum_str,
        "query_name_str": "TEST_NAV", "imported_timestamp_str": "2026-09-04T06:00:00+00:00"},))
    account_list = [
        {"account_route": "U100", "pod_id": "pod_taa", "display_name": "TAA 3x", "effective_from": "2026-08-03", "effective_to": None},
        {"account_route": "U200", "pod_id": "pod_ndx", "display_name": "NDX Momentum", "effective_from": "2026-08-03", "effective_to": None},
        {"account_route": "U900", "display_name": "Vault", "role": "reserve", "effective_from": "2026-08-03", "effective_to": None}]
    for account_dict in account_list:
        account_dict["effective_to"] = (effective_to_dict or {}).get(account_dict["account_route"])
    client_dict = {"client_id": "local", "display_name": "sample_user", "base_currency": "USD", "fee_basis": "Synthetic",
        "query_name": "TEST_NAV", "mandate_start_date": "2026-08-03", "operations_source": operations_source_str,
        "accounts": deepcopy(account_list), **local_ibkr_reporting_profile_dict()}
    workspace_dict = {"client_dict": client_dict, "financial_scope_complete_bool": True, "financial_error_str": None,
        "valuation_account_list": [{key_str: account_dict[key_str] for key_str in ("account_route", "display_name", "pod_id", "role")
            if key_str in account_dict} for account_dict in account_list],
        "operations_account_list": account_list[:2], "summary_dict": {"pod_list": [], "pod_row_dict_list": []}}
    return workspace_dict, snapshot_obj


def test_month_and_since_start_match_the_performance_page(demo_tuple):
    workspace_dict, snapshot_obj, provider_obj = demo_tuple
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS, month_str="2026-08")
    month_dict = build_performance_page_dict(workspace_dict, snapshot_obj, as_of_ts=DEMO_NOW_TS,
        from_date_str="2026-08-01", to_date_str="2026-08-31")["report_dict"]
    since_dict = build_performance_page_dict(workspace_dict, snapshot_obj, as_of_ts=DEMO_NOW_TS,
        from_date_str="2026-06-01", to_date_str="2026-08-31")["report_dict"]
    assert report_dict["available_bool"] and not report_dict["draft_bool"]
    assert _money_value_dict(report_dict) == {
        "Value on August 31": (_money_str(since_dict["closing_nav_float"]), ""),
        "August": (_money_str(month_dict["pnl_float"], signed_bool=True), _percent_str(month_dict["twr_float"])),
        "Since Jun 1, 2026": (_money_str(since_dict["pnl_float"], signed_bool=True), _percent_str(since_dict["twr_float"])),
        "Net invested": (_money_str(since_dict["closing_nav_float"] - since_dict["pnl_float"]), ""),
    }
    assert _money_value_dict(report_dict)["August"] == ("+$1,090", "+1.1%")
    # The months row links the same daily returns as the Performance grid.
    grid_dict = build_performance_page_dict(workspace_dict, snapshot_obj, as_of_ts=DEMO_NOW_TS, from_date_str="2026-06-01",
        to_date_str="2026-08-31")["monthly_row_list"][0]["cell_list"]
    program_row_dict = report_dict["month_table_dict"]["row_list"][0]
    assert report_dict["month_table_dict"]["label_list"] == ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    assert program_row_dict["name_str"] == "Portfolio 2026"
    assert [cell_dict["value_str"] for cell_dict in program_row_dict["cell_list"]] == ["", "", "", "", ""] + [
        _percent_str(grid_dict[month_int - 1]["return_float"], suffix_str="") for month_int in (6, 7, 8)] + ["", "", "", ""]
    # The first year's return is the return since start.
    assert program_row_dict["year_return_str"] == _percent_str(since_dict["twr_float"], suffix_str="")


def test_default_is_the_current_month_to_date_and_a_month_in_progress_is_a_draft(demo_tuple):
    workspace_dict, snapshot_obj, provider_obj = demo_tuple
    default_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS)
    assert default_dict["month_str"] == "2026-09"
    assert [option_dict["label_str"] for option_dict in default_dict["month_option_list"]] == [
        "September 2026 · to date", "August 2026", "July 2026", "June 2026"]
    assert [option_dict["selected_bool"] for option_dict in default_dict["month_option_list"]] == [True, False, False, False]
    current_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS, month_str="2026-09")
    assert current_dict == default_dict
    assert current_dict["draft_bool"] and current_dict["draft_reason_str"] == "Month in progress, through Sep 4"
    assert current_dict["period_label_str"] == "September 2026 · to Sep 4"
    assert current_dict["money_row_list"][0]["label_str"] == "Value on September 4"
    assert build_monthly_report_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS, month_str="2026-05") is None


def test_route_is_read_only_validates_the_month_and_renders_both_pages(demo_tuple, tmp_path):
    workspace_dict, snapshot_obj, provider_obj = demo_tuple
    app_obj = create_app(provider_obj, demo_bool=True, now_fn=lambda: DEMO_NOW_TS, performance_db_path_str=str(tmp_path / "none.sqlite3"),
        workspace_snapshot_fn=lambda: (workspace_dict, snapshot_obj))
    client_obj = app_obj.test_client()
    for query_str, status_int in (("", 200), ("?month=2026-09", 200), ("?month=2026-05", 404), ("?month=2026-13", 400),
            ("?month=26-08", 400), ("?month=2026-08&month=2026-07", 400), ("?period=All", 400)):
        assert client_obj.get("/monthly-report" + query_str).status_code == status_int, query_str
    response_obj = client_obj.get("/monthly-report?month=2026-08")
    text_str = _visible_text_str(response_obj.get_data(as_text=True))
    for expected_str in ("Monthly report", "Value on August 31 · USD", "+$1,090", "+1.1%", "Since Jun 1, 2026", "Holdings",
            "Return by month", "Return (%)", "TWR", "Details", "Net invested", "Capital account", "Money in and out",
            "August 2026 · Summary", "August 2026 · Details", "Quantitative Strategies"):
        assert expected_str in text_str
    # TWR is a legend on the chart, and nothing else repeats it or dates the holdings.
    assert response_obj.get_data(as_text=True).count(">TWR</text>") == 1 and "Time-weighted" not in text_str and "Weights on" not in text_str
    assert "About these numbers" not in text_str
    assert "Draft" not in text_str
    assert response_obj.headers["Content-Security-Policy"].startswith("default-src 'self'")
    assert "<script>" not in response_obj.get_data(as_text=True)
    assert "Draft · Month in progress, through Sep 4" in _visible_text_str(client_obj.get("/monthly-report?month=2026-09").get_data(as_text=True))
    assert 'href="/monthly-report"' in client_obj.get("/performance").get_data(as_text=True)
    assert '<svg class="mono"' in response_obj.get_data(as_text=True) and ">LS</text>" in response_obj.get_data(as_text=True)
    assert re.search(r'<img class="art" [^>]*style="height: \d+px"', response_obj.get_data(as_text=True))


@pytest.mark.parametrize("method_str", ["POST", "PUT", "PATCH", "DELETE"])
def test_writes_are_rejected_before_any_financial_source_access(method_str):
    app_obj = create_app(object(), now_fn=lambda: DEMO_NOW_TS,
        workspace_snapshot_fn=lambda: pytest.fail("Write request reached financial source"))
    assert app_obj.test_client().open("/monthly-report", method=method_str).status_code == 403


def test_programs_group_pods_without_changing_any_total(demo_tuple):
    workspace_dict, snapshot_obj, provider_obj = demo_tuple
    program_dict = {"demo_1_0": "Steady", "demo_1_1": "Steady", "demo_1_2": "Growth", "demo_1_3": "Growth"}
    single_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS, month_str="2026-08")
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS,
        month_str="2026-08", program_by_pod_dict=program_dict)
    assert report_dict["money_row_list"] == single_dict["money_row_list"]
    assert not report_dict["single_program_bool"] and [program["name_str"] for program in report_dict["program_list"]] == ["Steady", "Growth"]
    assert [program["color_str"] for program in report_dict["program_list"]] == ["#1D4F91", "#13875C"]
    pod_page_dict = build_performance_page_dict(workspace_dict, snapshot_obj, as_of_ts=DEMO_NOW_TS,
        from_date_str="2026-06-01", to_date_str="2026-08-31", level_str="pods")
    closing_dict = {row_dict["pod_id_str"]: row_dict["end_str"] for row_dict in pod_page_dict["pod_row_list"]}
    strategy_dict = {strategy_dict["pod_id_str"]: strategy_dict["closing_nav_float"] for strategy_dict in pod_page_dict["report_dict"]["strategy_list"]}
    for program in report_dict["program_list"]:
        member_list = [pod_id_str for pod_id_str, name_str in program_dict.items() if name_str == program["name_str"]]
        assert program["value_str"] == _money_str(sum(strategy_dict[pod_id_str] for pod_id_str in member_list))
        assert program["chart_available_bool"]
        assert [stat_dict["label_str"] for stat_dict in program["stat_list"]] == ["August", "Since start"]
        assert len(program["pod_holding_list"]) == 2 and all(closing_dict[pod_id_str] for pod_id_str in member_list)
    assert [(row_dict["class_str"], row_dict["name_str"]) for row_dict in report_dict["month_table_dict"]["row_list"]] == [
        ("prog", "Steady 2026"), ("pod", "Pod 1 · D···1_0"), ("pod", "Pod 2 · D···1_1"),
        ("prog", "Growth 2026"), ("pod", "Pod 3 · D···1_2"), ("pod", "Pod 4 · D···1_3")]
    # The table under the summary and the chart both go one level down: the programs.
    breakdown_dict = report_dict["breakdown_dict"]
    assert breakdown_dict["title_str"] == "Program" and [row_dict["name_str"] for row_dict in breakdown_dict["row_list"]] == ["Steady", "Growth"]
    assert abs(sum(float(row_dict["share_str"].rstrip("%")) for row_dict in breakdown_dict["row_list"]) - 100) < .15
    assert [row_dict["value_str"] for row_dict in breakdown_dict["row_list"]] == [program["value_str"] for program in report_dict["program_list"]]
    assert [(line_dict["name_str"], line_dict["color_str"]) for line_dict in report_dict["chart_dict"]["line_list"]] == [("Steady", "#1D4F91"), ("Growth", "#13875C")]
    assert report_dict["details_bool"] and "Steady" not in report_dict["sub_str"]


def test_holdings_use_the_close_and_fixed_class_order(demo_tuple):
    workspace_dict, snapshot_obj, provider_obj = demo_tuple
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS, month_str="2026-08")
    holding_dict = report_dict["program_list"][0]["holding_dict"]
    assert holding_dict["available_bool"]
    class_order_list = [monthly_report.CASH_LABEL_STR if name_str == "Cash" else name_str for name_str, _ in ASSET_CLASS_TUPLE]
    legend_list = holding_dict["legend_list"]
    assert [item_dict["name_str"] for item_dict in legend_list] == sorted((item_dict["name_str"] for item_dict in legend_list), key=class_order_list.index)
    assert abs(sum(float(item_dict["pct_str"].rstrip("%")) for item_dict in legend_list) - 100) < .35
    assert {item_dict["name_str"]: item_dict["detail_str"] for item_dict in legend_list}["Stocks"] == "10 names"
    stock_pod_dict, fund_pod_dict = report_dict["program_list"][0]["pod_holding_list"][0], report_dict["program_list"][0]["pod_holding_list"][2]
    assert not stock_pod_dict["name_bool"] and fund_pod_dict["name_bool"]
    assert stock_pod_dict["row_list"][-1]["symbol_str"] == "Cash and accruals" and stock_pod_dict["row_list"][-1]["muted_bool"]
    assert {row_dict["symbol_str"]: row_dict["name_str"] for row_dict in fund_pod_dict["row_list"]}["TQQQ"] == "Nasdaq 100"
    # Each weight is rounded on its own (equal holdings keep equal weights), so a
    # column may miss 100.0% by up to 0.05 points a row.
    for pod_dict in report_dict["program_list"][0]["pod_holding_list"]:
        assert abs(sum(float(row_dict["weight_str"].rstrip("%")) for row_dict in pod_dict["row_list"]) - 100) <= .05 * len(pod_dict["row_list"])


def test_flows_add_up_to_money_put_in_and_withhold_any_gap(demo_tuple):
    workspace_dict, snapshot_obj, provider_obj = demo_tuple
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS, month_str="2026-08")
    flow_dict = report_dict["flow_dict"]
    assert flow_dict["row_list"] == []  # No August capital movements; prior months are not repeated.
    assert flow_dict["total_str"] == _money_value_dict(report_dict)["Net invested"][0] == "$92,000"
    account_list = [account_dict for account_dict in workspace_dict["client_dict"]["accounts"]]
    pod_list = [{"pod_id_str": account_dict["pod_id"], "name_str": account_dict["display_name"]} for account_dict in account_list]
    wrong_dict = _flow_dict(account_list, snapshot_obj, pod_list, from_str="2026-06-01", to_str="2026-08-31", put_in_float=92100.)
    assert wrong_dict["row_list"] == [] and wrong_dict["reason_str"]


def test_other_accounts_stay_apart_from_programs_and_activity_reads_the_month(tmp_path):
    workspace_dict, snapshot_obj = _sample_tuple()
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, SimpleNamespace(), as_of_ts=SAMPLE_AS_OF_TS, month_str="2026-08")
    page_dict = build_performance_page_dict(workspace_dict, snapshot_obj, as_of_ts=SAMPLE_AS_OF_TS,
        from_date_str="2026-08-03", to_date_str="2026-08-31")
    assert not report_dict["draft_bool"], report_dict["draft_reason_str"]
    value_dict = _money_value_dict(report_dict)
    assert value_dict["Value on August 31"][0] == _money_str(page_dict["report_dict"]["closing_nav_float"])
    assert report_dict["program_list"][0]["value_str"] == _money_str(page_dict["report_dict"]["closing_nav_float"])
    # The vault is named, never counted: its USD change would mostly be the exchange rate.
    assert value_dict["Since Aug 3, 2026"][0] == _money_str(page_dict["report_dict"]["pnl_float"], signed_bool=True)
    assert report_dict["scope_note_str"] == "Vault is held separately and is not included."
    # One program, two Pods: the table goes down to the Pods, and each Pod's month adds up to the whole.
    breakdown_dict = report_dict["breakdown_dict"]
    assert breakdown_dict["title_str"] == "Pod" and [row_dict["name_str"] for row_dict in breakdown_dict["row_list"]] == ["Pod 1 · U···100", "Pod 2 · U···200"]
    assert sum(_money_int(row_dict["pnl_str"]) for row_dict in breakdown_dict["row_list"]) == _money_int(value_dict["August"][0])
    assert value_dict["Added in August"] == ("+$1,000", "")
    assert [row_dict["kind_str"] for row_dict in report_dict["flow_dict"]["row_list"]] == ["Opening capital", "Net inflow", "Offsetting movements"]
    assert report_dict["flow_dict"]["pod_total_list"] == ["$20,500", "$10,500"] and report_dict["flow_dict"]["total_str"] == "$31,000"
    assert report_dict["activity_list"] == [{"label_str": "Commissions", "value_str": "$1.25"},
        {"label_str": "Dividends", "value_str": "$10.00"}, {"label_str": "Tax withheld", "value_str": "$2.50"},
        {"label_str": "Interest", "value_str": "$0.30"}]
    # Each movement is one line, whatever the number of Pods.
    assert [(line_dict["kind_str"], line_dict["pod_str"], line_dict["value_str"]) for line_dict in report_dict["flow_dict"]["line_list"]] == [
        ("Opening capital", "Pod 1 · U···100", "$20,000.00"), ("Opening capital", "Pod 2 · U···200", "$10,000.00"),
        ("Net inflow", "Pod 1 · U···100", "+$1,000.00"), ("Offsetting movements", "Pod 1 · U···100", "−$500.00"), ("Offsetting movements", "Pod 2 · U···200", "+$500.00")]
    # Production client names come from the account holder; a technical configured id is never a title.
    assert report_dict["client_name_str"] == ""
    # Missing optional holdings remain absent, with their reason also on the printed page.
    program_dict = report_dict["program_list"][0]
    assert program_dict["holding_dict"]["reason_str"] == "Holdings on Aug 31 were not saved."
    assert program_dict["pod_holding_list"] == [] and report_dict["chart_dict"]["available_bool"]
    assert report_dict["status_note_str"] == "Holdings on Aug 31 were not saved."
    app_obj = create_app(object(), now_fn=lambda: SAMPLE_AS_OF_TS, performance_db_path_str=str(tmp_path / "none.sqlite3"),
        workspace_snapshot_fn=lambda: (workspace_dict, snapshot_obj))
    html_str = app_obj.test_client().get("/monthly-report?month=2026-08").get_data(as_text=True)
    page_str = html_str[html_str.index('<main class="mr-pages"'):]
    assert '<h1 class="h1">August 2026</h1>' in page_str and "sample_user" not in html_str
    assert "$50,100" not in page_str and "Vault is held separately and is not included." in page_str
    assert "<b>Pod 1 · U···100</b>" in page_str and "Since Aug 3, 2026" in page_str
    # The client's page says it plainly; the technical reason stays on the operator's screen bar.
    assert "Holdings are not included in this report." in page_str and "Holdings on Aug 31" not in page_str
    assert "Holdings on Aug 31 were not saved." in html_str[:html_str.index('<main class="mr-pages"')]


def test_flex_positions_leave_cash_and_accruals_as_the_rest_of_nav(monkeypatch):
    workspace_dict, snapshot_obj = _sample_tuple()
    value_by_route_dict = {"U100": ("TQQQ", 8000.), "U200": ("MU", 900.)}

    def close_holdings_dict(database_path_str, nav_row_obj, *, query_name_str, as_of_ts):
        symbol_str, value_float = value_by_route_dict[nav_row_obj.account_route_str]
        return {"available_bool": True, "position_list": [{"symbol_str": symbol_str, "shares_float": 10., "value_float": value_float}]}

    monkeypatch.setattr(monthly_report, "load_close_holdings_dict", close_holdings_dict)

    def holding_dict():
        return build_monthly_report_dict(workspace_dict, snapshot_obj, SimpleNamespace(), as_of_ts=SAMPLE_AS_OF_TS,
            month_str="2026-08", database_path_str="unused.sqlite3")["program_list"][0]["holding_dict"]

    closing_float = sum(float(row_obj.closing_nav_decimal) for row_obj in snapshot_obj.row_tuple
        if row_obj.account_route_str in value_by_route_dict and row_obj.market_date_str == "2026-08-31")
    assert {item_dict["name_str"]: item_dict["pct_str"] for item_dict in holding_dict()["legend_list"]} == {
        name_str: _percent_str(value_float / closing_float, signed_bool=False) for name_str, value_float in
        (("Nasdaq 100", 8000.), ("Stocks", 900.), ("Cash and accruals", closing_float - 8900.))}
    # Positions worth more than the account mean borrowing: stated, never drawn as a share of the donut.
    value_by_route_dict["U100"] = ("TQQQ", closing_float * 2)
    short_dict = holding_dict()
    assert not short_dict["available_bool"]
    assert short_dict["paper_note_str"] == "Holdings include borrowing or short positions. Details lists them."


def test_saved_portfolio_counts_only_when_captured_after_that_close(monkeypatch):
    workspace_dict, snapshot_obj = _sample_tuple()
    target_dict = {pod_id_str: SimpleNamespace(release_obj=SimpleNamespace(pod_id_str=pod_id_str, account_route_str=route_str))
        for pod_id_str, route_str in (("pod_taa", "U100"), ("pod_ndx", "U200"))}
    provider_obj = SimpleNamespace(get_target_for_pod=target_dict.get)
    observed_dict = {"timestamp_str": "2026-08-31T20:10:00+00:00"}

    def saved_portfolio_dict(target_obj, *, as_of_ts):
        position_list = ([{"symbol_str": "TQQQ", "value_float": 8000.}, {"symbol_str": "GLD", "value_float": 6000.},
            {"symbol_str": "UUP", "value_float": 4000.}] if target_obj.release_obj.pod_id_str == "pod_taa"
            else [{"symbol_str": symbol_str, "value_float": 900.} for symbol_str in ("MU", "CSCO", "FTNT", "WBD", "SNDK", "AMD")])
        nav_float = float(next(row_obj.closing_nav_decimal for row_obj in snapshot_obj.row_tuple
            if row_obj.account_route_str == target_obj.release_obj.account_route_str and row_obj.market_date_str == "2026-08-31"))
        return {"available_bool": True, "observed_timestamp_str": observed_dict["timestamp_str"],
            "position_list": position_list, "cash_float": nav_float - sum(row_dict["value_float"] for row_dict in position_list)}

    monkeypatch.setattr(monthly_report, "load_close_holdings_dict", lambda *args, **kwargs: {"available_bool": False})
    monkeypatch.setattr(monthly_report, "load_broker_holdings_dict", saved_portfolio_dict)
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=SAMPLE_AS_OF_TS,
        month_str="2026-08", database_path_str="unused.sqlite3")
    holding_dict = report_dict["program_list"][0]["holding_dict"]
    assert holding_dict["available_bool"]
    closing_float = sum(float(row_obj.closing_nav_decimal) for row_obj in snapshot_obj.row_tuple
        if row_obj.account_route_str in {"U100", "U200"} and row_obj.market_date_str == "2026-08-31")
    assert {item_dict["name_str"]: item_dict["pct_str"] for item_dict in holding_dict["legend_list"]} == {
        name_str: _percent_str(value_float / closing_float, signed_bool=False) for name_str, value_float in
        (("Nasdaq 100", 8000.), ("Stocks", 5400.), ("Gold", 6000.), ("US dollar", 4000.), ("Cash and accruals", closing_float - 23400.))}
    assert {item_dict["name_str"]: item_dict["detail_str"] for item_dict in holding_dict["legend_list"]}["Stocks"] == "6 names"
    # A capture from another day, or before the close, never stands in for this month end.
    for timestamp_str in ("2026-09-01T20:10:00+00:00", "2026-08-31T19:30:00+00:00"):
        observed_dict["timestamp_str"] = timestamp_str
        report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=SAMPLE_AS_OF_TS,
            month_str="2026-08", database_path_str="unused.sqlite3")
        assert report_dict["program_list"][0]["holding_dict"]["reason_str"] == "Holdings on Aug 31 were not saved."
        assert report_dict["program_list"][0]["pod_holding_list"] == []


def _one_pod_tuple(**keyword_dict):
    """The sample client with its second Pod and vault removed from the accounts."""
    workspace_dict, snapshot_obj = _sample_tuple(**keyword_dict)
    for key_list in (workspace_dict["client_dict"]["accounts"], workspace_dict["valuation_account_list"], workspace_dict["operations_account_list"]):
        key_list[:] = [account_dict for account_dict in key_list if account_dict["account_route"] == "U100"]
    return workspace_dict, snapshot_obj


def test_one_pod_is_one_page_with_everything_on_it(tmp_path):
    workspace_dict, snapshot_obj = _one_pod_tuple(extra_event_dict={("U100", "2026-08-05"): {"commissions": "-2"}})
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, SimpleNamespace(), as_of_ts=SAMPLE_AS_OF_TS, month_str="2026-08")
    assert not report_dict["details_bool"] and report_dict["breakdown_dict"] is None
    assert report_dict["sub_str"] == "Monthly report · Account U···100"
    assert [row_dict["name_str"] for row_dict in report_dict["month_table_dict"]["row_list"]] == ["2026"]
    assert report_dict["month_table_dict"]["wide_bool"]
    assert report_dict["chart_dict"]["height_int"] == 208
    app_obj = create_app(object(), now_fn=lambda: SAMPLE_AS_OF_TS, performance_db_path_str=str(tmp_path / "none.sqlite3"),
        workspace_snapshot_fn=lambda: (workspace_dict, snapshot_obj))
    html_str = app_obj.test_client().get("/monthly-report?month=2026-08").get_data(as_text=True)
    page_str = html_str[html_str.index('<main class="mr-pages"'):]
    assert page_str.count('class="pg"') == 1 and "Return by month" in page_str and "Return (%)" in page_str and "August activity" in page_str
    assert page_str.count(">TWR</text>") == 1 and "Time-weighted" not in page_str
    assert page_str.count("Commissions and fees charged by Interactive Brokers") == 1
    assert "Capital account" not in page_str and "· Details" not in page_str and "· Summary" not in page_str


def test_one_pod_with_shorts_gains_a_page_to_list_them(monkeypatch):
    workspace_dict, snapshot_obj = _one_pod_tuple()
    provider_obj = _patch_saved_portfolio(monkeypatch, {"pod_taa": [("TQQQ", 9000.), ("AMD", -3000.)], "pod_ndx": []}, {"pod_taa": 5000., "pod_ndx": 1.})
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=SAMPLE_AS_OF_TS,
        month_str="2026-08", database_path_str="unused.sqlite3")
    assert report_dict["details_bool"], "The sentence on page 1 points at the list on page 2."
    assert report_dict["program_list"][0]["holding_dict"]["paper_note_str"] == "Holdings include borrowing or short positions. Details lists them."


def test_a_named_single_program_shows_in_the_sub_line_and_a_year_fits_with_one_pod():
    workspace_dict, snapshot_obj = _one_pod_tuple()
    workspace_dict["client_dict"]["accounts"][0]["effective_from"] = "2026-08-03"
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, SimpleNamespace(), as_of_ts=SAMPLE_AS_OF_TS, month_str="2026-08",
        program_by_pod_dict={"pod_taa": "Growth"})
    assert report_dict["sub_str"] == "Monthly report · Growth · Account U···100"
    assert report_dict["chart_dict"]["line_list"][0]["color_str"] == monthly_report.LONE_LINE_COLOR_STR


def test_a_withdrawal_is_one_row_and_a_transfer_between_pods_is_none():
    workspace_dict, snapshot_obj = _sample_tuple(extra_event_dict={
        ("U100", "2026-08-17"): {"depositsWithdrawals": "-400"}, ("U100", "2026-08-24"): {"depositsWithdrawals": "0"}, ("U200", "2026-08-24"): {"depositsWithdrawals": "0"}})
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, SimpleNamespace(), as_of_ts=SAMPLE_AS_OF_TS, month_str="2026-08")
    assert _money_value_dict(report_dict)["Withdrawn in August"] == ("−$400", "")
    assert report_dict["money_row_list"][-1]["tone_str"] == "neg"
    workspace_dict, snapshot_obj = _sample_tuple(extra_event_dict={("U100", "2026-08-17"): {"depositsWithdrawals": "0"}})
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, SimpleNamespace(), as_of_ts=SAMPLE_AS_OF_TS, month_str="2026-08")
    assert [row_dict["label_str"] for row_dict in report_dict["money_row_list"]] == ["Value on August 31", "August", "Since Aug 3, 2026", "Net invested"]


def test_empty_activity_lines_stay_off_the_page():
    workspace_dict, snapshot_obj = _sample_tuple(extra_event_dict={
        ("U200", "2026-08-10"): {"commissions": "0"}, ("U100", "2026-08-20"): {"dividends": "0", "withholdingTax": "0"}, ("U200", "2026-08-31"): {"interest": "0"}})
    assert build_monthly_report_dict(workspace_dict, snapshot_obj, SimpleNamespace(), as_of_ts=SAMPLE_AS_OF_TS, month_str="2026-08")["activity_list"] == []


def test_money_moved_at_a_month_boundary_still_gets_its_row():
    # A Pod that starts on the month's first session is in the month's opening value.
    workspace_dict, snapshot_obj = _sample_tuple()
    for key_list in (workspace_dict["client_dict"]["accounts"],):
        next(account_dict for account_dict in key_list if account_dict["account_route"] == "U200")["effective_from"] = "2026-09-01"

    def report_dict(month_str):
        return build_monthly_report_dict(workspace_dict, snapshot_obj, SimpleNamespace(), as_of_ts=SAMPLE_AS_OF_TS, month_str=month_str)

    def net_invested_int(report_dict_obj):
        return _money_int(_money_value_dict(report_dict_obj)["Net invested"][0])

    opening_decimal = next(row_obj.opening_nav_decimal for row_obj in snapshot_obj.row_tuple if row_obj.account_route_str == "U200" and row_obj.market_date_str == "2026-09-01")
    assert _money_value_dict(report_dict("2026-09"))["Added in September"] == (_money_str(opening_decimal, signed_bool=True), "")
    assert abs(net_invested_int(report_dict("2026-09")) - net_invested_int(report_dict("2026-08")) - _money_int(_money_str(opening_decimal))) <= 1
    # A Pod that closes on the month's last day counted in that month's value, so it leaves in the next.
    workspace_dict, snapshot_obj = _sample_tuple(effective_to_dict={"U200": "2026-08-31"})
    closing_decimal = next(row_obj.closing_nav_decimal for row_obj in snapshot_obj.row_tuple if row_obj.account_route_str == "U200" and row_obj.market_date_str == "2026-08-31")
    assert "Withdrawn in August" not in _money_value_dict(report_dict("2026-08"))
    assert _money_value_dict(report_dict("2026-09"))["Withdrawn in September"] == (_money_str(-closing_decimal, signed_bool=True), "")
    # The month's list of movements says the same as the row above it.
    assert [(line_dict["date_str"], line_dict["kind_str"], line_dict["pod_str"], line_dict["value_str"]) for line_dict in report_dict("2026-09")["flow_dict"]["line_list"]] == [
        ("Aug 31, 2026", "Pod closed, money out", "Pod 2 · U···200", _money_str(-closing_decimal, signed_bool=True, cents_bool=True))]
    assert report_dict("2026-08")["flow_dict"]["line_list"][-1]["kind_str"] != "Pod closed, money out"
    assert abs(net_invested_int(report_dict("2026-08")) - net_invested_int(report_dict("2026-09")) - _money_int(_money_str(closing_decimal))) <= 1
    # The client's very first day is where net invested starts, not money moved.
    workspace_dict, snapshot_obj = _one_pod_tuple(extra_event_dict={("U100", "2026-08-17"): {"depositsWithdrawals": "0"}, ("U100", "2026-08-24"): {"depositsWithdrawals": "0"}})
    assert [row_dict["label_str"] for row_dict in report_dict("2026-08")["money_row_list"]] == ["Value on August 31", "August", "Since Aug 3, 2026", "Net invested"]


def test_pod_rows_add_up_to_the_month_when_a_pod_retires():
    workspace_dict, snapshot_obj = _sample_tuple(effective_to_dict={"U200": "2026-08-13"})
    # The vault here is a third Pod, so one Pod closes and two go on.
    for key_list in (workspace_dict["client_dict"]["accounts"], workspace_dict["valuation_account_list"]):
        third_dict = next(account_dict for account_dict in key_list if account_dict["account_route"] == "U900")
        third_dict.pop("role")
        third_dict["pod_id"] = "pod_third"
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, SimpleNamespace(), as_of_ts=SAMPLE_AS_OF_TS, month_str="2026-08")
    row_list = report_dict["breakdown_dict"]["row_list"]
    assert len(row_list) == 3 and row_list[1]["note_str"] == "closed Aug 13, 2026" and row_list[1]["value_str"] == "$0"
    assert abs(sum(_money_int(row_dict["pnl_str"]) for row_dict in row_list) - _money_int(_money_value_dict(report_dict)["August"][0])) <= 1


def test_each_year_gets_a_row_newest_first_and_the_reports_year_lists_the_pods():
    assert _year_list("2023-05-01", "2026-08") == ["2026", "2025", "2024", "2023"] and _year_list("2026-08-03", "2026-08") == ["2026"]
    return_dict = {"2025-11": .01, "2025-12": .02, "2026-01": -.01}
    pod_list = [{"head_str": "Pod 1 · U···100", "month_return_dict": {"2025-12": .03, "2026-01": .01}}, {"head_str": "Pod 2 · U···200", "month_return_dict": {"2025-11": .02}}]
    row_list = _year_row_list("Growth", return_dict, pod_list, ["2026", "2025", "2024"], report_year_str="2026", year_known_bool=True, one_pod_bool=False)
    # 2024 has nothing to show; Pod 2 had no month in the report's year.
    assert [(row_dict["class_str"], row_dict["name_str"], row_dict["year_return_str"]) for row_dict in row_list] == [
        ("prog", "Growth 2026", "−1.0"), ("pod", "Pod 1 · U···100", "+1.0"), ("prog", "Growth 2025", "+3.0")]
    assert [cell_dict["value_str"] for cell_dict in row_list[2]["cell_list"]] == [""] * 10 + ["+1.0", "+2.0"]
    assert [cell_dict["value_str"] for cell_dict in row_list[0]["cell_list"]][:2] == ["−1.0", ""]
    assert row_list[0]["year_return_tone_str"] == "neg" and row_list[0]["cell_list"][0]["tone_str"] == "neg"
    # One Pod: the year alone names the row. The report's year always has a row, even empty.
    row_list = _year_row_list("Growth", {}, [], ["2026"], report_year_str="2026", year_known_bool=True, one_pod_bool=True)
    assert [(row_dict["name_str"], row_dict["year_return_str"]) for row_dict in row_list] == [("2026", "—")]
    # A program that closed last year has no empty row for this one.
    row_list = _year_row_list("Steady", {"2025-11": .01}, [], ["2026", "2025"], report_year_str="2026", year_known_bool=True, one_pod_bool=False)
    assert [row_dict["name_str"] for row_dict in row_list] == ["Steady 2025"]


def test_many_years_move_the_table_of_one_pod_to_a_details_page(monkeypatch):
    workspace_dict, snapshot_obj = _one_pod_tuple(extra_event_dict={("U100", "2026-08-17"): {"depositsWithdrawals": "0"}, ("U100", "2026-08-24"): {"depositsWithdrawals": "0"}})

    def report_dict():
        return build_monthly_report_dict(workspace_dict, snapshot_obj, SimpleNamespace(), as_of_ts=SAMPLE_AS_OF_TS, month_str="2026-08")

    assert not report_dict()["details_bool"] and report_dict()["chart_dict"]["height_int"] == 208
    monkeypatch.setattr(monthly_report, "_year_list", lambda start_str, month_str: ["2026", "2025", "2024"])
    assert not report_dict()["details_bool"]
    monkeypatch.setattr(monthly_report, "_year_list", lambda start_str, month_str: ["2026", "2025", "2024", "2023"])
    # On a Details page the months leave page 1, and the chart takes the room.
    assert report_dict()["details_bool"] and report_dict()["chart_dict"]["height_int"] == 280


def _page_heights_tuple(legend_row_list=(4,), **keyword_dict):
    return _page_height_tuple(**{"art_bool": True, "banner_int": 0, "movement_bool": False, "sub_str": "Monthly report", "client_name_str": "Dana Levi",
        "scope_note_bool": False, "breakdown_row_int": 0, "legend_list_list": [[{"name_str": "Gold", "detail_str": "GLD"}] * row_int for row_int in legend_row_list],
        "details_bool": False, "year_row_int": 1, "activity_list": [{"label_str": "Commissions", "value_str": "$12.65"}], "flow_note_bool": False, **keyword_dict})


def _chart_height_of_page_int(*position_list, **keyword_dict):
    return _page_heights_tuple(*position_list, **keyword_dict)[0]


def _cost_list(value_str):
    return [{"label_str": label_str, "value_str": value_str} for label_str in ("Commissions", "Other fees", "Dividends", "Tax withheld", "Interest")]


def test_the_chart_takes_the_room_the_rest_of_the_page_leaves():
    height_int = _chart_height_of_page_int
    # One Pod, one page: the chart stops at a calm size, with air below it.
    assert height_int() == 208
    # More Pods: the months are on Details, so the chart can use the page.
    assert height_int(breakdown_row_int=2, legend_row_list=[5], details_bool=True) > 250
    assert height_int(breakdown_row_int=4, legend_row_list=[5], details_bool=True) < height_int(breakdown_row_int=2, legend_row_list=[5], details_bool=True)
    # Whatever else grows, the chart gives way, and never below a size that can be read.
    crowded_dict = {"year_row_int": 3, "movement_bool": True, "banner_int": 1, "scope_note_bool": True}
    assert 120 <= height_int(**crowded_dict) < 208
    # A long name or sub line wraps beside the monogram, and the chart makes room for the extra line.
    two_year_dict = {"year_row_int": 2}
    assert height_int(**two_year_dict, client_name_str="N" * 37) == height_int(**two_year_dict, client_name_str="N" * 36) - 26
    assert height_int(**two_year_dict, sub_str="S" * 63) == height_int(**two_year_dict, sub_str="S" * 62) - 20
    # So does a costs row long enough to wrap, and the line that says money could not be matched.
    assert height_int(**two_year_dict, activity_list=_cost_list("$1,234.56")) <= height_int(**two_year_dict, activity_list=_cost_list("$1.25")) - 24
    assert height_int(**two_year_dict, flow_note_bool=True) <= height_int(**two_year_dict) - 20
    assert height_int(breakdown_row_int=9, legend_row_list=[8, 8, 8, 8], details_bool=True, movement_bool=True, banner_int=2) == 120
    # Six or more classes on one program sit in two columns, so they take no more room than four.
    assert height_int(legend_row_list=[8]) == height_int(legend_row_list=[4])


FOUR_PODS_DICT = {"details_bool": True, "breakdown_row_int": 4, "legend_row_list": [5]}


@pytest.mark.parametrize("override_dict,height_int", [
    (FOUR_PODS_DICT, 210),
    ({**FOUR_PODS_DICT, "banner_int": 1}, 181),
    ({**FOUR_PODS_DICT, "movement_bool": True}, 179),
    ({**FOUR_PODS_DICT, "scope_note_bool": True}, 185),
    ({**FOUR_PODS_DICT, "legend_row_list": [2]}, 240),
    ({**FOUR_PODS_DICT, "legend_row_list": [0]}, 280),
    ({"art_bool": False, "details_bool": True, "breakdown_row_int": 3, "legend_row_list": [5, 3, 2]}, 240),
    ({"details_bool": True, "breakdown_row_int": 2, "legend_row_list": [5, 3]}, 258),
    ({"year_row_int": 3, "activity_list": []}, 208),
])
def test_chart_height_for_each_page_shape(override_dict, height_int):
    # The page's heights, from monthly_report.css. A browser checks them against the real print; see MONTHLY_CLIENT_REPORT.md.
    assert _chart_height_of_page_int(**override_dict) == height_int


def test_a_crowded_page_takes_room_from_the_picture_after_the_chart():
    # An ordinary page keeps the whole picture.
    assert _page_heights_tuple() == (208, 176)
    # One Pod, two years, money moved, a Vault note, five classes and five costs that wrap.
    assert _page_heights_tuple(legend_row_list=[5], year_row_int=2, movement_bool=True, scope_note_bool=True, activity_list=_cost_list("$1,234.56")) == (120, 141)
    # Six Pods with a Vault note and money moved.
    assert _page_heights_tuple(legend_row_list=[5], details_bool=True, breakdown_row_int=6, movement_bool=True, scope_note_bool=True) == (120, 143)
    # Past that the picture stops at a thin band, and the page continues on a further sheet.
    assert _page_heights_tuple(legend_row_list=[8, 8, 8, 8], details_bool=True, breakdown_row_int=9, movement_bool=True, banner_int=2) == (120, 96)


def test_rows_that_wrap_beside_another_program_take_room_from_the_chart():
    def height_int(detail_str):
        return _chart_height_of_page_int(details_bool=True, breakdown_row_int=2, legend_list_list=[
            [{"name_str": "Nasdaq 100", "detail_str": detail_str}] * 2 + [{"name_str": "Gold", "detail_str": "GLD"}] * 3, [{"name_str": "Gold", "detail_str": "GLD"}] * 3])

    assert height_int("TQQQ") == 258
    assert height_int("TQQQ, QQQ") == 258 - 35
    assert height_int("TQQQ, QQQ, QLD, X") == 258 - 70
    # In two columns under one program a row has more room.
    many_list = [{"name_str": "Nasdaq 100", "detail_str": "TQQQ, QQQ"}] * 8
    assert _chart_height_of_page_int(details_bool=True, breakdown_row_int=2, legend_list_list=[many_list]) == _chart_height_of_page_int(details_bool=True, breakdown_row_int=2, legend_row_list=[8])


def test_the_chart_budget_is_fed_by_what_the_page_prints(monkeypatch):
    seen_dict, real_fn = {}, monthly_report._page_height_tuple

    def spy_tuple(**keyword_dict):
        seen_dict.clear()
        seen_dict.update(keyword_dict)
        return real_fn(**keyword_dict)

    monkeypatch.setattr(monthly_report, "_page_height_tuple", spy_tuple)
    workspace_dict, snapshot_obj = _sample_tuple()
    workspace_dict["client_dict"]["display_name"] = "Dana Levi"
    provider_obj = _patch_saved_portfolio(monkeypatch, {"pod_taa": [("TQQQ", 9000.), ("GLD", 6000.)], "pod_ndx": [("MU", 9000.)]}, {"pod_taa": 5000., "pod_ndx": 4000.})
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=SAMPLE_AS_OF_TS, month_str="2026-08",
        database_path_str="unused.sqlite3", program_by_pod_dict={"pod_taa": "Growth", "pod_ndx": "Steady"}, art_url_fn=lambda month_str: "/art.jpg")
    assert seen_dict == {"art_bool": True, "banner_int": 0, "movement_bool": True, "sub_str": report_dict["sub_str"], "client_name_str": "Dana Levi",
        "scope_note_bool": True, "breakdown_row_int": 2, "legend_list_list": [program_dict["holding_dict"]["legend_list"] for program_dict in report_dict["program_list"]],
        "details_bool": True, "year_row_int": 2, "activity_list": report_dict["activity_list"], "flow_note_bool": False}
    assert report_dict["sub_str"] and all(seen_dict["legend_list_list"]) and seen_dict["activity_list"]
    assert report_dict["art_height_int"] == 176 and report_dict["chart_dict"]["height_int"] < 280
    # A month in progress prints no bar, and no picture means none is charged.
    build_monthly_report_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=SAMPLE_AS_OF_TS, month_str="2026-09", database_path_str="unused.sqlite3")
    assert seen_dict["banner_int"] == 0 and seen_dict["art_bool"] is False
    # Data that does not tie prints the bar, and the line that says money could not be matched.
    workspace_dict, snapshot_obj = _one_pod_tuple(extra_event_dict={("U100", "2026-08-12"): {"linkingAdjustments": "NaN"}})
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, SimpleNamespace(), as_of_ts=SAMPLE_AS_OF_TS, month_str="2026-08")
    assert report_dict["flow_dict"]["reason_str"] and seen_dict["flow_note_bool"] is True and seen_dict["details_bool"] is False and seen_dict["banner_int"] == 1


def test_the_chart_legend_names_the_measure_in_the_colour_of_a_lone_line():
    def series_dict(name_str, color_str):
        return {"name_str": name_str, "color_str": color_str, "path_list": [
            {"market_date_str": "2026-08-03", "cumulative_return_float": 0.}, {"market_date_str": "2026-08-04", "cumulative_return_float": .01}]}

    chart_dict = _chart_dict([series_dict("Growth", "#123456")], height_int=120)
    assert chart_dict["legend_color_str"] == "#123456"
    # The plot starts under the legend's strip, at the smallest chart too.
    assert min(tick_dict["y"] for tick_dict in chart_dict["tick_list"]) > 16
    assert _chart_dict([series_dict("Growth", "#123456"), series_dict("Steady", "#654321")], height_int=208)["legend_color_str"] == "#94A3B8"


def test_a_month_not_final_says_so_in_its_date_and_only_a_data_problem_prints_the_bar(tmp_path):
    def report_dict(month_str, extra_event_dict=None):
        workspace_dict, snapshot_obj = _sample_tuple(extra_event_dict=extra_event_dict)
        return build_monthly_report_dict(workspace_dict, snapshot_obj, SimpleNamespace(), as_of_ts=SAMPLE_AS_OF_TS, month_str=month_str)

    progress_dict = report_dict("2026-09")
    assert progress_dict["draft_bool"] and not progress_dict["draft_bar_bool"] and progress_dict["period_label_str"].endswith(" · to Sep 3")
    # The operator still sees why, and the report still cannot be saved as final.
    assert progress_dict["status_note_str"].startswith("Draft · Month in progress, through Sep 3")
    final_dict = report_dict("2026-08")
    assert not final_dict["draft_bool"] and not final_dict["draft_bar_bool"]
    for problem_dict in (report_dict("2026-08", {("U100", "2026-08-12"): {"linkingAdjustments": "NaN"}}), report_dict("2026-09", {("U100", "2026-09-01"): {"linkingAdjustments": "NaN"}})):
        assert problem_dict["draft_bool"] and problem_dict["draft_bar_bool"] and problem_dict["draft_reason_str"] != "Month in progress, through Sep 3"
    workspace_dict, snapshot_obj = _sample_tuple()
    client_obj = create_app(object(), now_fn=lambda: SAMPLE_AS_OF_TS, performance_db_path_str=str(tmp_path / "none.sqlite3"),
        workspace_snapshot_fn=lambda: (workspace_dict, snapshot_obj)).test_client()
    html_str = client_obj.get("/monthly-report?month=2026-09").get_data(as_text=True)
    assert 'class="mr-draft"' not in html_str and "Print draft" in html_str and "September 2026 · to Sep 3" in html_str


def test_other_fees_are_listed_beside_commissions():
    def report_dict(extra_event_dict=None):
        workspace_dict, snapshot_obj = _sample_tuple(extra_event_dict=extra_event_dict)
        return build_monthly_report_dict(workspace_dict, snapshot_obj, SimpleNamespace(), as_of_ts=SAMPLE_AS_OF_TS, month_str="2026-08")

    def gain_str(result_dict):
        return {row_dict["label_str"]: row_dict["value_str"] for row_dict in result_dict["bridge_row_list"]}["Investment gain/loss"]

    charged_dict, plain_dict = report_dict({("U100", "2026-08-05"): {"otherFees": "-14.5"}}), report_dict()
    assert charged_dict["activity_list"][:2] == [{"label_str": "Commissions", "value_str": "$1.25"}, {"label_str": "Other fees", "value_str": "$14.50"}]
    # The fee is already inside the gain: the line shows it, it does not take it again.
    assert not charged_dict["draft_bool"] and not plain_dict["draft_bool"]
    assert (gain_str(plain_dict), gain_str(charged_dict)) == ("+$63.55", "+$49.05")
    # The caption says who charges them, and only when a cost is printed.
    assert charged_dict["activity_note_str"] == "Commissions and fees charged by Interactive Brokers"
    income_dict = report_dict({("U200", "2026-08-10"): {"commissions": "0"}})
    assert income_dict["activity_list"] and income_dict["activity_note_str"] == ""
    # A refund reads as a negative cost; a charge and a refund that cancel stay off the page.
    assert {item_dict["label_str"]: item_dict["value_str"] for item_dict in report_dict({("U100", "2026-08-05"): {"otherFees": "3"}})["activity_list"]}["Other fees"] == "−$3.00"
    assert "Other fees" not in {item_dict["label_str"] for item_dict in report_dict({("U100", "2026-08-05"): {"otherFees": "-3"}, ("U200", "2026-08-06"): {"otherFees": "3"}})["activity_list"]}


def test_a_missing_since_start_never_shows_the_month_as_since_start(demo_tuple, monkeypatch):
    workspace_dict, snapshot_obj, provider_obj = demo_tuple
    real_fn = monthly_report.build_performance_page_dict

    def since_start_missing_dict(workspace_dict, snapshot_obj, **kwargs_dict):
        page_dict = real_fn(workspace_dict, snapshot_obj, **kwargs_dict)
        return {**page_dict, "report_dict": None, "client_report_dict": None} if kwargs_dict["from_date_str"] == "2026-06-01" else page_dict

    monkeypatch.setattr(monthly_report, "build_performance_page_dict", since_start_missing_dict)
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS, month_str="2026-08",
        program_by_pod_dict={"demo_1_0": "Steady", "demo_1_1": "Steady", "demo_1_2": "Growth", "demo_1_3": "Growth"})
    assert report_dict["breakdown_dict"]["title_str"] == "Program"
    assert all(row_dict["since_str"] == "—" for row_dict in report_dict["breakdown_dict"]["row_list"])
    assert all(row_dict["year_return_str"] == "—" for row_dict in report_dict["month_table_dict"]["row_list"])
    assert not report_dict["chart_dict"]["available_bool"]


@pytest.mark.parametrize("name_str,expected_str", [("ERAN SLOBODINSKY", "Eran Slobodinsky"), ("Loren Slobodinsky", "Loren Slobodinsky"),
    ("McDonald Levi", "McDonald Levi"), ("O'BRIEN", "O'Brien"), ("", ""), ("SMITH'S FUND LLC", "Smith's Fund LLC"),
    ("JOHN SMITH III", "John Smith III"), ("MARY-ANN O'BRIEN", "Mary-Ann O'Brien"), ("שלום", "שלום"),
    ("ANNE MCDONALD", "Anne McDonald"), ("SMITH’S", "Smith’s"), ("MC", "Mc")])
def test_names_stored_in_capitals_read_as_names(name_str, expected_str):
    assert monthly_report._display_name_str(name_str) == expected_str


def test_picture_rule_never_leaves_a_month_without_one_once_any_exists(tmp_path):
    from alpha.live.dashboard_v4.monthly_report import report_art_name_str
    assert report_art_name_str(tmp_path / "missing", "2026-10") == ""
    assert report_art_name_str(tmp_path, "2026-10") == ""
    for name_str in ("2026-10.jpg", "2026-12.webp", "notes.txt", "2026-13.jpg", "x2026-11.jpg"):
        (tmp_path / name_str).write_bytes(b"x")
    (tmp_path / "2026-11.jpg").mkdir()
    assert [report_art_name_str(tmp_path, month_str) for month_str in ("2026-09", "2026-10", "2026-11", "2026-12", "2027-03")] == [
        "2026-10.jpg", "2026-10.jpg", "2026-10.jpg", "2026-12.webp", "2026-12.webp"]
    # A file that failed to copy is no picture; the good older one stands in.
    (tmp_path / "2027-01.jpg").write_bytes(b"")
    assert report_art_name_str(tmp_path, "2027-03") == "2026-12.webp"


def test_holder_name_comes_from_the_client_query_and_is_escaped(tmp_path):
    db_path_str = str(tmp_path / "performance.sqlite3")

    def save_xml(xml_str, query_name_str="TEST_NAV"):
        with closing(sqlite3.connect(db_path_str)) as connection_obj, connection_obj:
            connection_obj.execute("CREATE TABLE IF NOT EXISTS flex_import (import_id_int INTEGER PRIMARY KEY, query_name_str TEXT, raw_xml_str TEXT)")
            connection_obj.execute("INSERT INTO flex_import (query_name_str, raw_xml_str) VALUES (?, ?)", (query_name_str, xml_str))

    def statement_str(account_str, name_str):
        return (f'<FlexStatement accountId="{account_str}"><AccountInformation accountId="{account_str}" name="{name_str}"/>'
            "</FlexStatement>")

    save_xml(f"<FlexQueryResponse><FlexStatements>{statement_str('U100', 'Old Name')}</FlexStatements></FlexQueryResponse>")
    save_xml("<FlexQueryResponse><FlexStatements>" + statement_str("U100", "Dana  Levi") + statement_str("U200", "Dana Levi")
        + statement_str("U300", "Someone Else") + "</FlexStatements></FlexQueryResponse>")
    # A newer import of another query (no account names) never replaces the client's own.
    save_xml("<FlexQueryResponse><FlexStatements/></FlexQueryResponse>", query_name_str="OTHER_QUERY")
    assert load_holder_name_str(db_path_str, {"U100", "U200"}, query_name_str="TEST_NAV") == "Dana Levi"
    assert load_holder_name_str(db_path_str, {"U100", "U300"}, query_name_str="TEST_NAV") == ""
    assert load_holder_name_str(str(tmp_path / "missing.sqlite3"), {"U100"}, query_name_str="TEST_NAV") == ""
    save_xml('<!DOCTYPE x [<!ENTITY a "Dana">]><FlexQueryResponse/>')
    assert load_holder_name_str(db_path_str, {"U100"}, query_name_str="TEST_NAV") == ""
    # The holder name is page text: markup in it stays text.
    save_xml("<FlexQueryResponse><FlexStatements>" + statement_str("U100", "&lt;b&gt;Dana&lt;/b&gt;")
        + statement_str("U200", "&lt;b&gt;Dana&lt;/b&gt;") + "</FlexStatements></FlexQueryResponse>")
    workspace_dict, snapshot_obj = _sample_tuple()
    app_obj = create_app(object(), now_fn=lambda: SAMPLE_AS_OF_TS, performance_db_path_str=db_path_str,
        workspace_snapshot_fn=lambda: (workspace_dict, snapshot_obj))
    html_str = app_obj.test_client().get("/monthly-report?month=2026-08").get_data(as_text=True)
    assert "<b>Dana</b>" not in html_str and html_str.count("&lt;b&gt;Dana&lt;/b&gt;") >= 3


def test_formats_round_half_up_without_a_negative_zero():
    assert [_money_str(value_float) for value_float in (1234.5, -0.4, 0.)] == ["$1,235", "$0", "$0"]
    assert _money_str(-2356.2, signed_bool=True) == "−$2,356" and _money_str(6.205, cents_bool=True) == "$6.21"
    assert [_percent_str(value_float) for value_float in (.0025, -.0004, -.071, None)] == ["+0.3%", "0.0%", "−7.1%", "—"]


def test_chart_stays_inside_its_plot_and_labels_zero():
    path_list = [{"market_date_str": "2026-04-13 SOD", "cumulative_return_float": 0.}]
    path_list += [{"market_date_str": f"2026-{month_int:02d}-15", "cumulative_return_float": value_float}
        for month_int, value_float in ((4, -.024), (5, .03), (6, .081), (7, .052), (8, .095), (9, .067))]
    chart_dict = _chart_dict([{"name_str": "Portfolio", "color_str": "#1F5F6B", "path_list": path_list}], height_int=208)
    line_dict = chart_dict["line_list"][0]
    coordinate_list = [tuple(map(float, pair_str.split())) for pair_str in line_dict["path_str"][1:].split(" L")]
    assert all(chart_dict["plot_left"] <= x_float <= chart_dict["plot_right"] and 10 <= y_float <= chart_dict["height_int"] - 24
        for x_float, y_float in coordinate_list)
    assert [tick_dict["label_str"] for tick_dict in chart_dict["tick_list"]] == ["0%", "+5%", "+10%"]
    assert line_dict["label_str"] == "+6.7%" and chart_dict["month_list"][0]["label_str"] == "Apr" and not line_dict["leader_bool"]
    # The shading closes on the 0% gridline at both ends, so it fills between the line and zero.
    zero_y = next(tick_dict["y"] for tick_dict in chart_dict["tick_list"] if tick_dict["zero_bool"])
    area_point_list = [tuple(map(float, part_str.split())) for part_str in line_dict["area_str"][1:-2].split(" L")]
    assert area_point_list[1:-1] == coordinate_list and line_dict["area_str"].endswith(" Z")
    assert area_point_list[0] == (coordinate_list[0][0], zero_y) and area_point_list[-1] == (coordinate_list[-1][0], zero_y)
    assert not _chart_dict([{"name_str": "Portfolio", "color_str": "#1F5F6B", "path_list": path_list[:1]}], height_int=208)["available_bool"]
    assert not _chart_dict([], height_int=208)["available_bool"]


def test_chart_end_labels_that_would_touch_move_apart_and_stay_on_the_page():
    def path_list(value_float):
        return [{"market_date_str": "2026-08-03", "cumulative_return_float": 0.}, {"market_date_str": "2026-09-30", "cumulative_return_float": value_float}]

    chart_dict = _chart_dict([{"name_str": "Steady", "color_str": "#1D4F91", "path_list": path_list(.024)},
        {"name_str": "Growth", "color_str": "#13875C", "path_list": path_list(.025)}], height_int=150)
    steady_dict, growth_dict = chart_dict["line_list"]
    assert abs(steady_dict["label_y"] - growth_dict["label_y"]) >= 18 and growth_dict["label_y"] < steady_dict["label_y"]
    assert steady_dict["leader_bool"] and growth_dict["leader_bool"] and steady_dict["area_str"] == growth_dict["area_str"] == ""
    # The longest allowed program name is cut in the label only, so its return stays on the chart.
    long_name_str = "Aggressive Long Term Growth Plus"[:30]
    chart_dict = _chart_dict([{"name_str": long_name_str, "color_str": "#1D4F91", "path_list": path_list(.5)},
        {"name_str": "Growth", "color_str": "#13875C", "path_list": path_list(-.5)}], height_int=150)
    assert chart_dict["plot_right"] > chart_dict["plot_left"] + 300
    long_dict = chart_dict["line_list"][0]
    assert long_dict["name_str"] == long_name_str and long_dict["label_name_str"].endswith("…") and len(long_dict["label_name_str"]) <= monthly_report.CHART_LABEL_NAME_LIMIT_INT
    assert long_dict["label_x"] + 15 + len(long_dict["label_name_str"] + " +50.0%") * 8.6 <= chart_dict["width_int"]
    # Ends at the very top and bottom keep their labels inside the plot.
    for line_dict in chart_dict["line_list"]:
        assert 10 <= line_dict["label_y"] <= chart_dict["height_int"] - 24 + 1


def test_unverified_source_shows_a_message_instead_of_numbers(demo_tuple, tmp_path):
    workspace_dict, snapshot_obj, provider_obj = demo_tuple
    broken_obj = BrokerReportingSnapshot(unavailable_reason_str="Saved account report could not be verified.")
    report_dict = build_monthly_report_dict(workspace_dict, broken_obj, provider_obj, as_of_ts=DEMO_NOW_TS, month_str="2026-08")
    assert not report_dict["available_bool"] and report_dict["error_str"] == "Saved financial data could not be verified."
    app_obj = create_app(provider_obj, demo_bool=True, now_fn=lambda: DEMO_NOW_TS, performance_db_path_str=str(tmp_path / "none.sqlite3"),
        workspace_snapshot_fn=lambda: (workspace_dict, broken_obj))
    text_str = _visible_text_str(app_obj.test_client().get("/monthly-report").get_data(as_text=True))
    assert "Saved financial data could not be verified." in text_str and "$" not in text_str


def _patch_saved_portfolio(monkeypatch, position_by_pod_dict, cash_by_pod_dict):
    """Saved IBKR end-of-day portfolios captured after the Aug 31 close (no Flex closing positions)."""
    def saved_portfolio_dict(target_obj, *, as_of_ts):
        pod_id_str = target_obj.release_obj.pod_id_str
        _, snapshot_obj = _sample_tuple()
        nav_float = float(next(row_obj.closing_nav_decimal for row_obj in snapshot_obj.row_tuple
            if row_obj.account_route_str == target_obj.release_obj.account_route_str and row_obj.market_date_str == "2026-08-31"))
        scale_float = nav_float / (cash_by_pod_dict[pod_id_str] + sum(value_float for _, value_float in position_by_pod_dict[pod_id_str]))
        return {"available_bool": True, "observed_timestamp_str": "2026-08-31T20:10:00+00:00", "cash_float": cash_by_pod_dict[pod_id_str] * scale_float,
            "position_list": [{"symbol_str": symbol_str, "value_float": value_float * scale_float} for symbol_str, value_float in position_by_pod_dict[pod_id_str]]}

    monkeypatch.setattr(monthly_report, "load_close_holdings_dict", lambda *args, **kwargs: {"available_bool": False})
    monkeypatch.setattr(monthly_report, "load_broker_holdings_dict", saved_portfolio_dict)
    return SimpleNamespace(get_target_for_pod={pod_id_str: SimpleNamespace(release_obj=SimpleNamespace(pod_id_str=pod_id_str,
        account_route_str=route_str)) for pod_id_str, route_str in (("pod_taa", "U100"), ("pod_ndx", "U200"))}.get)


def test_a_month_is_final_only_once_its_last_calendar_day_is_finalized():
    # October 2026 ends on a Saturday: its last session (Fri Oct 30) is not the month's last statement.
    october_dict = {option_dict["month_str"]: option_dict for option_dict in _month_option_list("2026-09-01", "2026-10-30")}["2026-10"]
    assert not october_dict["complete_bool"] and october_dict["sessions_done_bool"]
    assert october_dict["label_str"] == "October 2026 · to date"
    option_dict = {option_dict["month_str"]: option_dict for option_dict in _month_option_list("2026-09-01", "2026-11-02")}
    assert option_dict["2026-10"]["complete_bool"] and option_dict["2026-09"]["complete_bool"]


def test_a_missing_other_account_does_not_hold_back_the_programs_report():
    workspace_dict, snapshot_obj = _sample_tuple(last_row_dict={"U900": "2026-08-28"})
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, SimpleNamespace(), as_of_ts=SAMPLE_AS_OF_TS, month_str="2026-08")
    page_dict = build_performance_page_dict(workspace_dict, snapshot_obj, as_of_ts=SAMPLE_AS_OF_TS,
        from_date_str="2026-08-03", to_date_str="2026-08-31")
    assert report_dict["month_str"] == "2026-08" and not report_dict["draft_bool"], report_dict["draft_reason_str"]
    value_dict = _money_value_dict(report_dict)
    assert report_dict["program_list"][0]["value_str"] == _money_str(page_dict["report_dict"]["closing_nav_float"])
    assert value_dict["Value on August 31"][0] == _money_str(page_dict["report_dict"]["closing_nav_float"])
    assert build_monthly_report_dict(workspace_dict, snapshot_obj, SimpleNamespace(), as_of_ts=SAMPLE_AS_OF_TS,
        month_str="2026-09")["draft_reason_str"] == "Month in progress, through Sep 3"


def test_other_account_lines_need_an_other_account_at_the_close():
    workspace_dict, snapshot_obj = _sample_tuple(effective_to_dict={"U900": "2026-08-13"})
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, SimpleNamespace(), as_of_ts=SAMPLE_AS_OF_TS, month_str="2026-08")
    assert [row_dict["label_str"] for row_dict in report_dict["money_row_list"]] == [
        "Value on August 31", "August", "Since Aug 3, 2026", "Net invested", "Added in August"]


def test_a_retired_pod_keeps_its_history_but_leaves_the_split():
    workspace_dict, snapshot_obj = _sample_tuple(effective_to_dict={"U200": "2026-08-13"})
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, SimpleNamespace(), as_of_ts=SAMPLE_AS_OF_TS, month_str="2026-08")
    # A Pod that closed this month keeps its row until the month is over, then only its history stays.
    assert [(row_dict["name_str"], row_dict["value_str"], row_dict["share_str"], row_dict["note_str"]) for row_dict in report_dict["breakdown_dict"]["row_list"]] == [
        ("Pod 1 · U···100", report_dict["money_row_list"][0]["value_str"], "100.0%", ""), ("Pod 2 · U···200", "$0", "0.0%", "closed Aug 13, 2026")]
    assert build_monthly_report_dict(workspace_dict, snapshot_obj, SimpleNamespace(), as_of_ts=SAMPLE_AS_OF_TS, month_str="2026-09")["breakdown_dict"] is None
    assert report_dict["details_bool"]
    assert [row_dict["name_str"] for row_dict in report_dict["month_table_dict"]["row_list"]] == ["Portfolio 2026", "Pod 1 · U···100", "Pod 2 · U···200"]
    flow_dict = report_dict["flow_dict"]
    assert "Pod closed, money out" in [row_dict["kind_str"] for row_dict in flow_dict["row_list"]]
    assert flow_dict["total_str"] != "—" and flow_dict["total_str"] != _money_value_dict(report_dict)["Net invested"][0]
    assert build_monthly_report_dict(workspace_dict, snapshot_obj, SimpleNamespace(), as_of_ts=SAMPLE_AS_OF_TS,
        month_str="2026-09")["available_bool"]


def test_non_finite_flex_values_never_break_the_page(tmp_path):
    workspace_dict, snapshot_obj = _sample_tuple(extra_event_dict={("U200", "2026-08-12"): {"commissions": "NaN"}})
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, SimpleNamespace(), as_of_ts=SAMPLE_AS_OF_TS, month_str="2026-08")
    assert report_dict["available_bool"] and report_dict["draft_bool"]
    assert "Commissions" not in [item_dict["label_str"] for item_dict in report_dict["activity_list"]]
    app_obj = create_app(object(), now_fn=lambda: SAMPLE_AS_OF_TS, performance_db_path_str=str(tmp_path / "none.sqlite3"),
        workspace_snapshot_fn=lambda: (workspace_dict, snapshot_obj))
    assert app_obj.test_client().get("/monthly-report?month=2026-08").status_code == 200


def test_a_missing_since_start_report_keeps_the_month_and_says_why(demo_tuple, monkeypatch):
    workspace_dict, snapshot_obj, provider_obj = demo_tuple
    real_fn = monthly_report.build_performance_page_dict

    def since_start_missing_dict(workspace_dict, snapshot_obj, **kwargs_dict):
        page_dict = real_fn(workspace_dict, snapshot_obj, **kwargs_dict)
        return {**page_dict, "report_dict": None, "client_report_dict": None} if kwargs_dict["from_date_str"] == "2026-06-01" else page_dict

    monkeypatch.setattr(monthly_report, "build_performance_page_dict", since_start_missing_dict)
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS, month_str="2026-08")
    value_dict = _money_value_dict(report_dict)
    assert value_dict["August"] == ("+$1,090", "+1.1%")
    assert value_dict["Since Jun 1, 2026"] == ("—", "—") and value_dict["Net invested"] == ("—", "")
    assert report_dict["draft_reason_str"] == "Return since start unavailable"
    assert "Money in and out appears once the account history since start is complete." in report_dict["status_note_str"]
    assert not report_dict["chart_dict"]["available_bool"] and not report_dict["program_list"][0]["chart_available_bool"]
    assert all(row_dict["year_return_str"] == "—" for row_dict in report_dict["month_table_dict"]["row_list"])


def test_shorts_and_borrowing_get_no_donut_and_long_lists_fold(monkeypatch):
    workspace_dict, snapshot_obj = _sample_tuple()
    provider_obj = _patch_saved_portfolio(monkeypatch, {"pod_taa": [("TQQQ", 9000.), ("GLD", 6000.)],
        "pod_ndx": [("MU", 9000.), ("AMD", -3000.)]}, {"pod_taa": 5000., "pod_ndx": 4000.})
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=SAMPLE_AS_OF_TS,
        month_str="2026-08", database_path_str="unused.sqlite3")
    program_dict = report_dict["program_list"][0]
    assert program_dict["holding_dict"]["reason_str"] == "Holdings include borrowing or short positions. Details lists them."
    assert {row_dict["symbol_str"]: row_dict["weight_str"] for row_dict in program_dict["pod_holding_list"][1]["row_list"]}["AMD"] == "−30.0%"
    provider_obj = _patch_saved_portfolio(monkeypatch, {"pod_taa": [("TQQQ", 25000.)], "pod_ndx": [("MU", 6000.)]},
        {"pod_taa": -5000., "pod_ndx": 4000.})
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=SAMPLE_AS_OF_TS,
        month_str="2026-08", database_path_str="unused.sqlite3")
    assert report_dict["program_list"][0]["holding_dict"]["reason_str"] == "Holdings include borrowing or short positions. Details lists them."
    provider_obj = _patch_saved_portfolio(monkeypatch, {"pod_taa": [("TQQQ", 9000.)],
        "pod_ndx": [(f"S{index_int:02d}", 500. + index_int) for index_int in range(14)]}, {"pod_taa": 1000., "pod_ndx": 2000.})
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=SAMPLE_AS_OF_TS,
        month_str="2026-08", database_path_str="unused.sqlite3")
    row_list = report_dict["program_list"][0]["pod_holding_list"][1]["row_list"]
    assert len(row_list) == 13 and row_list[-2]["symbol_str"] == "3 more" and row_list[-2]["muted_bool"] and row_list[-1]["symbol_str"] == "Cash and accruals"
    legend_dict = {item_dict["name_str"]: item_dict["detail_str"] for item_dict in report_dict["program_list"][0]["holding_dict"]["legend_list"]}
    assert legend_dict["Nasdaq 100"] == "TQQQ" and legend_dict["Stocks"] == "14 names"
    # A fund among the stocks: four holdings in the class are four names, not four funds.
    provider_obj = _patch_saved_portfolio(monkeypatch, {"pod_taa": [("TQQQ", 9000.)],
        "pod_ndx": [("AAPL", 2000.), ("MSFT", 2000.), ("NVDA", 2000.), ("SPY", 2000.)]}, {"pod_taa": 1000., "pod_ndx": 2000.})
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=SAMPLE_AS_OF_TS,
        month_str="2026-08", database_path_str="unused.sqlite3")
    legend_dict = {item_dict["name_str"]: item_dict["detail_str"] for item_dict in report_dict["program_list"][0]["holding_dict"]["legend_list"]}
    assert legend_dict["Stocks"] == "4 names"


def test_four_programs_reuse_colour_families(demo_tuple):
    workspace_dict, snapshot_obj, provider_obj = demo_tuple
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS, month_str="2026-08",
        program_by_pod_dict={"demo_1_0": "One", "demo_1_1": "Two", "demo_1_2": "Three", "demo_1_3": "Four"})
    assert [program_dict["color_str"] for program_dict in report_dict["program_list"]] == ["#1D4F91", "#13875C", "#5B4BA8", "#1D4F91"]
    assert not report_dict["draft_bool"]


def test_guards_name_the_problem():
    workspace_dict, snapshot_obj = _sample_tuple()
    with pytest.raises(ValueError):
        build_monthly_report_dict(workspace_dict, snapshot_obj, SimpleNamespace(), as_of_ts=datetime(2026, 9, 4, 13))
    reserve_only_dict = deepcopy(workspace_dict)
    reserve_only_dict["client_dict"]["accounts"] = [account_dict for account_dict in reserve_only_dict["client_dict"]["accounts"]
        if account_dict.get("role") == "reserve"]
    report_dict = build_monthly_report_dict(reserve_only_dict, snapshot_obj, SimpleNamespace(), as_of_ts=SAMPLE_AS_OF_TS)
    assert not report_dict["available_bool"] and report_dict["error_str"] == "No program has started yet."
    early_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, SimpleNamespace(), as_of_ts=datetime(2026, 8, 3, 13, tzinfo=UTC))
    assert early_dict["error_str"] == "No finalized account value is available yet."


def test_pods_are_named_by_number_masked_account_and_ibkr_alias():
    workspace_dict, snapshot_obj = _sample_tuple()
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, SimpleNamespace(), as_of_ts=SAMPLE_AS_OF_TS, month_str="2026-08",
        alias_by_pod_dict={"pod_ndx": "growth_mom_natr20"})
    # The alias helps the operator in the Programs panel; the client's page names a Pod by number and account.
    assert [(row_dict["name_str"], row_dict["note_str"]) for row_dict in report_dict["breakdown_dict"]["row_list"]] == [("Pod 1 · U···100", ""), ("Pod 2 · U···200", "")]
    assert [choice_dict["name_str"] for choice_dict in report_dict["pod_choice_list"]] == ["Pod 1 · U···100", "Pod 2 · U···200 · growth_mom_natr20"]
    assert report_dict["flow_dict"]["pod_name_list"] == [{"head_str": "Pod 1 · U···100", "alias_str": ""},
        {"head_str": "Pod 2 · U···200", "alias_str": "growth_mom_natr20"}]
    # Numbers follow the first measured day: a Pod that starts later is numbered after the others.
    workspace_dict["client_dict"]["accounts"][0]["effective_from"] = "2026-08-04"
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, SimpleNamespace(), as_of_ts=SAMPLE_AS_OF_TS, month_str="2026-08")
    # They also print in that order, whatever the order of the accounts.
    assert [row_dict["name_str"] for row_dict in report_dict["breakdown_dict"]["row_list"]] == ["Pod 1 · U···200", "Pod 2 · U···100"]
    assert [row_dict["name_str"] for row_dict in report_dict["month_table_dict"]["row_list"]][1:] == ["Pod 1 · U···200", "Pod 2 · U···100"]


def test_alias_prints_only_in_the_programs_panel(monkeypatch, tmp_path):
    monkeypatch.setattr("alpha.live.dashboard_v4.app.load_alias_by_pod_dict", lambda _: {"pod_ndx": "growth_mom_natr20"})
    workspace_dict, snapshot_obj = _sample_tuple()
    # A short position lists the Pods' holdings on Details, one more place a Pod is named.
    provider_obj = _patch_saved_portfolio(monkeypatch, {"pod_taa": [("TQQQ", 9000.)], "pod_ndx": [("MU", 9000.), ("AMD", -3000.)]}, {"pod_taa": 5000., "pod_ndx": 4000.})
    provider_obj.event_log_path_str = str(tmp_path / "events.jsonl")
    client_obj = create_app(provider_obj, now_fn=lambda: SAMPLE_AS_OF_TS, performance_db_path_str=str(tmp_path / "none.sqlite3"),
        workspace_snapshot_fn=lambda: (workspace_dict, snapshot_obj)).test_client()
    panel_str, page_str = client_obj.get("/monthly-report?month=2026-08").get_data(as_text=True).split('<main class="mr-pages"', 1)
    assert '<span class="mr-pname">Pod 2 · U···200 · growth_mom_natr20</span>' in panel_str
    assert "signed holdings by Pod" in page_str and "Pod 2 · U···200" in page_str and "growth_mom_natr20" not in page_str


def test_every_table_follows_the_pod_numbers_when_accounts_arrive_out_of_order(monkeypatch):
    workspace_dict, snapshot_obj = _sample_tuple()
    workspace_dict["client_dict"]["accounts"][0]["effective_from"] = "2026-08-04"
    provider_obj = _patch_saved_portfolio(monkeypatch, {"pod_taa": [("TQQQ", 9000.), ("GLD", 6000.)], "pod_ndx": [("MU", 9000.)]}, {"pod_taa": 5000., "pod_ndx": 4000.})
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=SAMPLE_AS_OF_TS,
        month_str="2026-08", database_path_str="unused.sqlite3")
    expected_list = ["Pod 1 · U···200", "Pod 2 · U···100"]
    assert [row_dict["name_str"] for row_dict in report_dict["breakdown_dict"]["row_list"]] == expected_list
    assert [row_dict["name_str"] for row_dict in report_dict["month_table_dict"]["row_list"]][1:] == expected_list
    assert [item_dict["name_str"] for item_dict in report_dict["pod_choice_list"]] == expected_list
    assert [item_dict["name_str"] for item_dict in report_dict["program_list"][0]["pod_holding_list"]] == expected_list
    # Both Pods move money on Aug 24, so these lines show the column order.
    assert [line_dict["pod_str"] for line_dict in report_dict["flow_dict"]["line_list"] if line_dict["kind_str"] == "Offsetting movements"] == expected_list
    # Two programs: the program holding Pod 1 comes first.
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=SAMPLE_AS_OF_TS, month_str="2026-08",
        database_path_str="unused.sqlite3", program_by_pod_dict={"pod_taa": "Growth", "pod_ndx": "Steady"})
    assert [program_dict["name_str"] for program_dict in report_dict["program_list"]] == ["Steady", "Growth"]


def _program_token_str(client_obj):
    html_str = client_obj.get("/monthly-report?month=2026-08").get_data(as_text=True)
    return re.search(r'data-programs data-url="/monthly-report/programs" data-token="([^"]+)"', html_str).group(1)


def _post_programs(client_obj, body_obj, *, token_str, origin_str="http://localhost"):
    return client_obj.post("/monthly-report/programs", json=body_obj,
        headers={"Origin": origin_str, "X-Alpha-Action-Token": token_str})


def test_programs_panel_saves_choices_and_the_report_follows(demo_tuple, tmp_path):
    workspace_dict, snapshot_obj, provider_obj = demo_tuple
    app_obj = create_app(provider_obj, demo_bool=True, now_fn=lambda: DEMO_NOW_TS, performance_db_path_str=str(tmp_path / "none.sqlite3"),
        workspace_snapshot_fn=lambda: (workspace_dict, snapshot_obj))
    client_obj = app_obj.test_client()
    html_str = client_obj.get("/monthly-report?month=2026-08").get_data(as_text=True)
    assert html_str.count('class="mr-prow"') == 4 and "Other name…" in html_str
    token_str = _program_token_str(client_obj)
    response_obj = _post_programs(client_obj, {"confirmed_bool": True, "program_by_pod_dict": {
        "demo_1_0": "Steady", "demo_1_1": "Steady", "demo_1_2": "Growth", "demo_1_3": "  Growth   Plus "}}, token_str=token_str)
    assert response_obj.status_code == 200, response_obj.get_json()
    text_str = _visible_text_str(client_obj.get("/monthly-report?month=2026-08").get_data(as_text=True))
    for name_str in ("Steady", "Growth", "Growth Plus"):
        assert f"{name_str} program" in client_obj.get("/monthly-report?month=2026-08").get_data(as_text=True)
    assert "Portfolio since" not in text_str
    # Clearing a saved assignment moves that Pod back to Portfolio; the others keep theirs.
    assert _post_programs(client_obj, {"confirmed_bool": True, "program_by_pod_dict": {"demo_1_3": ""}}, token_str=token_str).status_code == 200
    html_str = client_obj.get("/monthly-report?month=2026-08").get_data(as_text=True)
    assert 'aria-label="Portfolio program"' in html_str and "Growth Plus program" not in html_str
    assert 'aria-label="Steady program"' in html_str


def test_programs_route_refuses_anything_but_a_same_origin_tokened_choice(demo_tuple, tmp_path):
    workspace_dict, snapshot_obj, provider_obj = demo_tuple
    app_obj = create_app(provider_obj, demo_bool=True, now_fn=lambda: DEMO_NOW_TS, performance_db_path_str=str(tmp_path / "none.sqlite3"),
        workspace_snapshot_fn=lambda: (workspace_dict, snapshot_obj))
    client_obj = app_obj.test_client()
    token_str = _program_token_str(client_obj)
    good_dict = {"confirmed_bool": True, "program_by_pod_dict": {"demo_1_0": "Steady"}}
    assert _post_programs(client_obj, good_dict, token_str="wrong").status_code == 403
    assert _post_programs(client_obj, good_dict, token_str=token_str, origin_str="http://evil.example").status_code == 403
    assert _post_programs(client_obj, {**good_dict, "confirmed_bool": False}, token_str=token_str).status_code == 400
    assert _post_programs(client_obj, {**good_dict, "extra": 1}, token_str=token_str).status_code == 400
    assert _post_programs(client_obj, {"confirmed_bool": True, "program_by_pod_dict": {}}, token_str=token_str).status_code == 400
    assert _post_programs(client_obj, {"confirmed_bool": True, "program_by_pod_dict": {"other_pod": "Steady"}}, token_str=token_str).status_code == 409
    for name_obj in ("<b>Growth</b>", "x" * 31, 7, " -Growth"):
        response_obj = _post_programs(client_obj, {"confirmed_bool": True, "program_by_pod_dict": {"demo_1_0": name_obj}}, token_str=token_str)
        assert response_obj.status_code == 400, name_obj
    assert client_obj.post("/monthly-report/programs", data="program=Steady", headers={"Origin": "http://localhost",
        "X-Alpha-Action-Token": token_str, "Content-Type": "application/x-www-form-urlencoded"}).status_code == 400
    assert client_obj.get("/monthly-report/programs").status_code == 405
    # Nothing was saved by the refused requests.
    assert 'aria-label="Portfolio program"' in client_obj.get("/monthly-report?month=2026-08").get_data(as_text=True)


def test_program_store_keeps_choices_in_a_file_next_to_the_event_log(tmp_path):
    from alpha.live.dashboard_v4.report_programs import PROGRAM_BYTE_LIMIT_INT, ReportProgramStore, clean_program_str, ReportProgramReadError
    provider_obj = SimpleNamespace(event_log_path_str=str(tmp_path / "events.jsonl"))
    ReportProgramStore(provider_obj).save({"pod_taa": "Growth", "pod_ndx": "Growth"})
    assert ReportProgramStore(provider_obj).program_by_pod_dict() == {"pod_taa": "Growth", "pod_ndx": "Growth"}
    ReportProgramStore(provider_obj).save({"pod_ndx": ""})
    assert ReportProgramStore(provider_obj).program_by_pod_dict() == {"pod_taa": "Growth"}
    path_obj = tmp_path / "report_programs.json"
    path_obj.write_text('{"program_by_pod_dict": {"pod_taa": "Steady", "pod_bad": "<script>"}}', encoding="utf-8")
    with pytest.raises(ReportProgramReadError):
        ReportProgramStore(provider_obj).program_by_pod_dict()
    path_obj.write_text("x" * (PROGRAM_BYTE_LIMIT_INT + 1), encoding="utf-8")
    with pytest.raises(ReportProgramReadError):
        ReportProgramStore(provider_obj).program_by_pod_dict()
    assert [clean_program_str(value_obj) for value_obj in ("  Growth  Plus ", "", "Q&A 1/2", None)] == ["Growth Plus", "", "Q&A 1/2", None]


def _sample_program_app(tmp_path, *, workspace_dict=None, snapshot_obj=None):
    """A production-mode app over the sample accounts; programs are saved next to its event log."""
    if workspace_dict is None:
        workspace_dict, snapshot_obj = _sample_tuple()
    (tmp_path / "logs").mkdir(exist_ok=True)
    provider_obj = SimpleNamespace(event_log_path_str=str(tmp_path / "logs" / "events.jsonl"))
    app_obj = create_app(provider_obj, now_fn=lambda: SAMPLE_AS_OF_TS, performance_db_path_str=str(tmp_path / "none.sqlite3"),
        workspace_snapshot_fn=lambda: (workspace_dict, snapshot_obj))
    return app_obj.test_client()


def test_unassigned_retired_program_keeps_its_own_history():
    workspace_dict, snapshot_obj = _sample_tuple(effective_to_dict={"U200": "2026-08-13"})
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, SimpleNamespace(), as_of_ts=SAMPLE_AS_OF_TS,
        month_str="2026-08", program_by_pod_dict={"pod_taa": "Growth"})
    assert [program_dict["name_str"] for program_dict in report_dict["program_list"]] == ["Growth", "Portfolio"]
    assert [row_dict["name_str"] for row_dict in report_dict["month_table_dict"]["row_list"]] == ["Growth 2026", "Pod 1 · U···100", "Portfolio 2026", "Pod 2 · U···200"]
    assert report_dict["program_list"][1]["closed_bool"]


def test_program_names_never_split_by_case(demo_tuple, tmp_path):
    workspace_dict, snapshot_obj, provider_obj = demo_tuple
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS, month_str="2026-08",
        program_by_pod_dict={"demo_1_0": "Growth Plus", "demo_1_1": "growth plus", "demo_1_2": "Steady", "demo_1_3": "Steady"})
    assert [program_dict["name_str"] for program_dict in report_dict["program_list"]] == ["Growth Plus", "Steady"]
    client_obj = _sample_program_app(tmp_path)
    token_str = _program_token_str(client_obj)
    assert _post_programs(client_obj, {"confirmed_bool": True, "program_by_pod_dict": {"pod_taa": "gROWTH", "pod_ndx": "PORTFOLIO"}},
        token_str=token_str).status_code == 200
    stored_dict = json.loads((tmp_path / "logs" / "report_programs.json").read_text(encoding="utf-8"))
    assert stored_dict == {"program_by_pod_dict": {"pod_taa": "Growth"}}


def test_program_store_survives_a_foreign_file(tmp_path):
    client_obj = _sample_program_app(tmp_path)
    path_obj = tmp_path / "logs" / "report_programs.json"
    for payload_str in ('{"program_by_pod_dict": ["Growth"]}', '["Growth"]', '{"program_by_pod_dict": "Growth"}'):
        path_obj.write_text(payload_str, encoding="utf-8")
        assert client_obj.get("/monthly-report?month=2026-08").status_code == 503, payload_str
    path_obj.write_text('{"program_by_pod_dict": {"pod_taa": "Growth ", "pod_ndx": "growth"}}', encoding="utf-8")
    html_str = client_obj.get("/monthly-report?month=2026-08").get_data(as_text=True)
    assert html_str.count('aria-label="Growth program"') == 1 and 'aria-label="Growth  program"' not in html_str


def test_programs_route_checks_origin_and_size(tmp_path):
    client_obj = _sample_program_app(tmp_path)
    token_str = _program_token_str(client_obj)
    body_dict = {"confirmed_bool": True, "program_by_pod_dict": {"pod_taa": "Growth"}}
    for header_dict in ({}, {"Origin": "null"}, {"Referer": "http://evil.example/page"}, {"Origin": "https://localhost"}):
        response_obj = client_obj.post("/monthly-report/programs", json=body_dict, headers={"X-Alpha-Action-Token": token_str, **header_dict})
        assert response_obj.status_code == 403, header_dict
    big_dict = {"confirmed_bool": True, "program_by_pod_dict": {f"pod_{index_int}": "Growth" for index_int in range(400)}}
    assert _post_programs(client_obj, big_dict, token_str=token_str).status_code == 400
    assert client_obj.post("/monthly-report/programs?x=1", json=body_dict,
        headers={"Origin": "http://localhost", "X-Alpha-Action-Token": token_str}).status_code == 400


def test_programs_route_reports_unreadable_accounts(tmp_path):
    workspace_dict, snapshot_obj = _sample_tuple()
    holder_dict = {"workspace_dict": workspace_dict}
    (tmp_path / "logs").mkdir()
    provider_obj = SimpleNamespace(event_log_path_str=str(tmp_path / "logs" / "events.jsonl"))
    client_obj = create_app(provider_obj, now_fn=lambda: SAMPLE_AS_OF_TS, performance_db_path_str=str(tmp_path / "none.sqlite3"),
        workspace_snapshot_fn=lambda: (holder_dict["workspace_dict"], snapshot_obj)).test_client()
    token_str = _program_token_str(client_obj)
    holder_dict["workspace_dict"] = {**workspace_dict, "financial_error_str": "Saved account report could not be verified."}
    response_obj = _post_programs(client_obj, {"confirmed_bool": True, "program_by_pod_dict": {"pod_taa": "Growth"}}, token_str=token_str)
    assert response_obj.status_code == 503 and response_obj.get_json()["error"] == "accounts_unreadable"
    assert not (tmp_path / "logs" / "report_programs.json").exists()


def test_program_saves_fail_cleanly(tmp_path, monkeypatch):
    from alpha.live.dashboard_v4 import report_programs
    client_obj = _sample_program_app(tmp_path)
    token_str = _program_token_str(client_obj)
    body_dict = {"confirmed_bool": True, "program_by_pod_dict": {"pod_taa": "Growth"}}
    assert _post_programs(client_obj, body_dict, token_str=token_str).status_code == 200
    path_obj = tmp_path / "logs" / "report_programs.json"
    before_str = path_obj.read_text(encoding="utf-8")
    calls_list = []

    def locked_replace(*args):
        calls_list.append(args)
        raise PermissionError(13, "The file is in use")

    monkeypatch.setattr(report_programs.os, "replace", locked_replace)
    monkeypatch.setattr(report_programs.time, "sleep", lambda seconds_float: None)
    response_obj = _post_programs(client_obj, {"confirmed_bool": True, "program_by_pod_dict": {"pod_ndx": "Steady"}}, token_str=token_str)
    assert response_obj.status_code == 500 and response_obj.get_json()["error"] == "write_failed"
    assert len(calls_list) == report_programs.REPLACE_ATTEMPT_INT and path_obj.read_text(encoding="utf-8") == before_str
    assert not (tmp_path / "logs" / "report_programs.json.tmp").exists()
    monkeypatch.undo()
    workspace_dict, snapshot_obj = _sample_tuple()
    provider_obj = SimpleNamespace(event_log_path_str=str(tmp_path / "gone" / "events.jsonl"))
    gone_client_obj = create_app(provider_obj, now_fn=lambda: SAMPLE_AS_OF_TS, performance_db_path_str=str(tmp_path / "none.sqlite3"),
        workspace_snapshot_fn=lambda: (workspace_dict, snapshot_obj)).test_client()
    response_obj = _post_programs(gone_client_obj, body_dict, token_str=_program_token_str(gone_client_obj))
    assert response_obj.status_code == 500 and response_obj.get_json()["error"] == "folder_missing"


def test_programs_panel_shows_the_saved_choice(tmp_path):
    client_obj = _sample_program_app(tmp_path)
    token_str = _program_token_str(client_obj)
    assert _post_programs(client_obj, {"confirmed_bool": True, "program_by_pod_dict": {"pod_taa": "Growth", "pod_ndx": "Q&A 1/2"}},
        token_str=token_str).status_code == 200
    html_str = client_obj.get("/monthly-report?month=2026-08").get_data(as_text=True)
    taa_str = html_str[html_str.index('data-pod-id="pod_taa"'):html_str.index('data-pod-id="pod_ndx"')]
    ndx_str = html_str[html_str.index('data-pod-id="pod_ndx"'):html_str.index("data-programs-save")]
    assert '<option value="Growth" selected>' in taa_str and 'data-program-name maxlength="30" placeholder="Program name" value=""' in taa_str
    assert '<option value="__other__" selected>' in ndx_str and 'value="Q&amp;A 1/2"' in ndx_str and " hidden>" not in ndx_str
    assert 'aria-label="Q&amp;A 1/2 program"' in html_str
    # A saved choice stays editable: renaming or clearing it is a normal save.
    assert " disabled" not in taa_str + ndx_str and " readonly" not in taa_str + ndx_str
    assert _post_programs(client_obj, {"confirmed_bool": True, "program_by_pod_dict": {"pod_taa": "Steady", "pod_ndx": ""}},
        token_str=token_str).status_code == 200
    html_str = client_obj.get("/monthly-report?month=2026-08").get_data(as_text=True)
    assert 'aria-label="Steady program"' in html_str and 'aria-label="Portfolio program"' in html_str
    assert 'aria-label="Q&amp;A 1/2 program"' not in html_str
