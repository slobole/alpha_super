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
    ASSET_CLASS_TUPLE, _chart_dict, _flow_dict, _money_str, _month_column_list, _month_option_list, _percent_str,
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
                for field_str in ("mtm", "commissions", "dividends", "withholdingTax", "interest")) if value_decimal.is_finite()), Decimal(0))
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
        "Change in August": (_money_str(month_dict["pnl_float"], signed_bool=True), _percent_str(month_dict["twr_float"])),
        "Gain since start": (_money_str(since_dict["pnl_float"], signed_bool=True), _percent_str(since_dict["twr_float"])),
        "Money you put in, net": (_money_str(since_dict["closing_nav_float"] - since_dict["pnl_float"]), ""),
    }
    assert _money_value_dict(report_dict)["Change in August"] == ("+$1,090", "+1.1%")
    # The months row links the same daily returns as the Performance grid.
    grid_dict = build_performance_page_dict(workspace_dict, snapshot_obj, as_of_ts=DEMO_NOW_TS, from_date_str="2026-06-01",
        to_date_str="2026-08-31")["monthly_row_list"][0]["cell_list"]
    program_row_dict = report_dict["month_table_dict"]["row_list"][0]
    assert report_dict["month_table_dict"]["label_list"] == ["Jun", "Jul", "Aug"]
    assert [cell_dict["value_str"] for cell_dict in program_row_dict["cell_list"]] == [
        _percent_str(grid_dict[month_int - 1]["return_float"], suffix_str="") for month_int in (6, 7, 8)]
    assert program_row_dict["since_str"] == _percent_str(since_dict["twr_float"], suffix_str="")


def test_default_is_the_latest_full_month_and_a_month_in_progress_is_a_draft(demo_tuple):
    workspace_dict, snapshot_obj, provider_obj = demo_tuple
    default_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS)
    assert default_dict["month_str"] == "2026-08"
    assert [option_dict["label_str"] for option_dict in default_dict["month_option_list"]] == [
        "September 2026 · to date", "August 2026", "July 2026", "June 2026"]
    assert [option_dict["selected_bool"] for option_dict in default_dict["month_option_list"]] == [False, True, False, False]
    current_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS, month_str="2026-09")
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
    for expected_str in ("Monthly report", "Value on August 31", "+$1,090", "+1.1%", "Gain since start", "Holdings",
            "Returns by month, %", "Details", "Money in and out, since start", "Money you put in, net", "Page 1 of 2", "Page 2 of 2"):
        assert expected_str in text_str
    assert "Draft" not in text_str
    assert response_obj.headers["Content-Security-Policy"].startswith("default-src 'self'")
    assert "<script>" not in response_obj.get_data(as_text=True)
    assert "Draft · Month in progress, through Sep 4" in _visible_text_str(client_obj.get("/monthly-report?month=2026-09").get_data(as_text=True))
    assert 'href="/monthly-report"' in client_obj.get("/performance").get_data(as_text=True)


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
        assert program["chart_dict"]["available_bool"] and program["chart_dict"]["width_int"] == 300
        assert [stat_dict["label_str"] for stat_dict in program["stat_list"]] == ["August", "Since start"]
        assert len(program["pod_holding_list"]) == 2 and all(closing_dict[pod_id_str] for pod_id_str in member_list)
    assert [(row_dict["class_str"], row_dict["name_str"]) for row_dict in report_dict["month_table_dict"]["row_list"]] == [
        ("prog", "Steady"), ("pod", "Pod 1 · D···1_0"), ("pod", "Pod 2 · D···1_1"),
        ("prog", "Growth"), ("pod", "Pod 3 · D···1_2"), ("pod", "Pod 4 · D···1_3")]
    share_list = [float(pod_dict["pct_str"].rstrip("%")) for group_dict in report_dict["split_dict"]["group_list"] for pod_dict in group_dict["pod_list"]]
    assert abs(sum(share_list) - 100) < .25 and len(report_dict["split_dict"]["segment_list"]) == 4


def test_holdings_use_the_close_and_fixed_class_order(demo_tuple):
    workspace_dict, snapshot_obj, provider_obj = demo_tuple
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS, month_str="2026-08")
    holding_dict = report_dict["program_list"][0]["holding_dict"]
    assert holding_dict["available_bool"]
    class_order_list = [name_str for name_str, _ in ASSET_CLASS_TUPLE]
    legend_list = holding_dict["legend_list"]
    assert [item_dict["name_str"] for item_dict in legend_list] == sorted((item_dict["name_str"] for item_dict in legend_list), key=class_order_list.index)
    assert abs(sum(float(item_dict["pct_str"].rstrip("%")) for item_dict in legend_list) - 100) < .35
    assert {item_dict["name_str"]: item_dict["detail_str"] for item_dict in legend_list}["Stocks"] == "10 stocks"
    stock_pod_dict, fund_pod_dict = report_dict["program_list"][0]["pod_holding_list"][0], report_dict["program_list"][0]["pod_holding_list"][2]
    assert not stock_pod_dict["name_bool"] and fund_pod_dict["name_bool"]
    assert stock_pod_dict["row_list"][-1]["symbol_str"] == "Cash" and stock_pod_dict["row_list"][-1]["muted_bool"]
    assert {row_dict["symbol_str"]: row_dict["name_str"] for row_dict in fund_pod_dict["row_list"]}["TQQQ"] == "Nasdaq 100 3×"
    # Each weight is rounded on its own (equal holdings keep equal weights), so a
    # column may miss 100.0% by up to 0.05 points a row.
    for pod_dict in report_dict["program_list"][0]["pod_holding_list"]:
        assert abs(sum(float(row_dict["weight_str"].rstrip("%")) for row_dict in pod_dict["row_list"]) - 100) <= .05 * len(pod_dict["row_list"])


def test_flows_add_up_to_money_put_in_and_withhold_any_gap(demo_tuple):
    workspace_dict, snapshot_obj, provider_obj = demo_tuple
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS, month_str="2026-08")
    flow_dict = report_dict["flow_dict"]
    assert [(row_dict["date_str"], row_dict["kind_str"], row_dict["total_str"]) for row_dict in flow_dict["row_list"]] == [
        ("Jun 1, 2026", "Start value", "$90,000"), ("Jul 15, 2026", "Deposit", "+$2,000")]
    assert flow_dict["total_str"] == _money_value_dict(report_dict)["Money you put in, net"][0] == "$92,000"
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
    assert value_dict["Programs, August 31"][0] == _money_str(page_dict["report_dict"]["closing_nav_float"])
    assert value_dict["Total, all accounts"][0] == _money_str(page_dict["client_report_dict"]["closing_nav_float"])
    assert value_dict["Other accounts"][0] == "$50,100"
    # The vault's gain is client money, not program profit.
    assert value_dict["Gain since start"][0] == _money_str(page_dict["report_dict"]["pnl_float"], signed_bool=True)
    assert [row_dict["kind_str"] for row_dict in report_dict["flow_dict"]["row_list"]] == ["Start value", "Deposit", "Between pods"]
    assert report_dict["flow_dict"]["pod_total_list"] == ["$20,500", "$10,500"] and report_dict["flow_dict"]["total_str"] == "$31,000"
    assert report_dict["activity_list"] == [{"label_str": "Commissions", "value_str": "$1.25"},
        {"label_str": "Dividends, after tax", "value_str": "$7.50"}, {"label_str": "Interest", "value_str": "$0.30"},
        {"label_str": "Added or withdrawn", "value_str": "+$1,000"}]
    # Production client names come from the account holder; a technical configured id is never a title.
    assert report_dict["client_name_str"] == ""
    # No saved holdings: the paper leaves the section out and the chart takes the card; the reason stays on screen.
    program_dict = report_dict["program_list"][0]
    assert program_dict["holding_dict"]["reason_str"] == "Holdings on Aug 31 were not saved."
    assert program_dict["pod_holding_list"] == [] and program_dict["chart_dict"]["width_int"] == 644
    assert report_dict["status_note_str"] == "Holdings on Aug 31 were not saved."
    app_obj = create_app(object(), now_fn=lambda: SAMPLE_AS_OF_TS, performance_db_path_str=str(tmp_path / "none.sqlite3"),
        workspace_snapshot_fn=lambda: (workspace_dict, snapshot_obj))
    html_str = app_obj.test_client().get("/monthly-report?month=2026-08").get_data(as_text=True)
    page_str = html_str[html_str.index('<main class="mr-pages">'):]
    assert '<h1 class="h1">August 2026</h1>' in page_str and "sample_user" not in html_str
    assert "not saved" not in page_str and "Holdings" not in _visible_text_str(page_str)
    assert "Holdings on Aug 31 were not saved." in html_str[:html_str.index('<main class="mr-pages">')]


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
        return {"available_bool": True, "observed_timestamp_str": observed_dict["timestamp_str"],
            "position_list": position_list, "cash_float": 2400. if target_obj.release_obj.pod_id_str == "pod_taa" else 5100.}

    monkeypatch.setattr(monthly_report, "load_close_holdings_dict", lambda *args, **kwargs: {"available_bool": False})
    monkeypatch.setattr(monthly_report, "load_broker_holdings_dict", saved_portfolio_dict)
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=SAMPLE_AS_OF_TS,
        month_str="2026-08", database_path_str="unused.sqlite3")
    holding_dict = report_dict["program_list"][0]["holding_dict"]
    assert holding_dict["available_bool"]
    # Shares of the two Pods' saved value together: 18,000 + 2,400 cash and 5,400 + 5,100 cash.
    assert {item_dict["name_str"]: item_dict["pct_str"] for item_dict in holding_dict["legend_list"]} == {
        "Nasdaq 100": "25.9%", "Stocks": "17.5%", "Gold": "19.4%", "US dollar": "12.9%", "Cash": "24.3%"}
    assert {item_dict["name_str"]: item_dict["detail_str"] for item_dict in holding_dict["legend_list"]}["Stocks"] == "6 stocks"
    # A capture from another day, or before the close, never stands in for this month end.
    for timestamp_str in ("2026-09-01T20:10:00+00:00", "2026-08-31T19:30:00+00:00"):
        observed_dict["timestamp_str"] = timestamp_str
        report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=SAMPLE_AS_OF_TS,
            month_str="2026-08", database_path_str="unused.sqlite3")
        assert report_dict["program_list"][0]["holding_dict"]["reason_str"] == "Holdings on Aug 31 were not saved."
        assert report_dict["program_list"][0]["pod_holding_list"] == []


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
    chart_dict = _chart_dict(path_list, single_bool=True)
    coordinate_list = [tuple(map(float, pair_str.split())) for pair_str in chart_dict["path_str"][1:].split(" L")]
    assert all(chart_dict["plot_left"] <= x_float <= chart_dict["plot_right"] and 10 <= y_float <= chart_dict["height_int"] - 22
        for x_float, y_float in coordinate_list)
    assert [tick_dict["label_str"] for tick_dict in chart_dict["tick_list"]] == ["0%", "+5%", "+10%"]
    assert chart_dict["end_label_str"] == "+6.7%" and chart_dict["month_list"][0]["label_str"] == "Apr"
    assert not _chart_dict(path_list[:1], single_bool=False)["available_bool"]


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
        return {"available_bool": True, "observed_timestamp_str": "2026-08-31T20:10:00+00:00", "cash_float": cash_by_pod_dict[pod_id_str],
            "position_list": [{"symbol_str": symbol_str, "value_float": value_float} for symbol_str, value_float in position_by_pod_dict[pod_id_str]]}

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
    assert _month_column_list("2025-06-10", "2026-02") == ["2025-06", "2025-07", "2025-08", "2025-09", "2025-10",
        "2025-11", "2025-12", "2026-01", "2026-02"]


def test_the_pods_close_sets_the_month_when_other_accounts_lag():
    workspace_dict, snapshot_obj = _sample_tuple(last_row_dict={"U900": "2026-08-28"})
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, SimpleNamespace(), as_of_ts=SAMPLE_AS_OF_TS)
    page_dict = build_performance_page_dict(workspace_dict, snapshot_obj, as_of_ts=SAMPLE_AS_OF_TS,
        from_date_str="2026-08-03", to_date_str="2026-08-31")
    assert report_dict["month_str"] == "2026-08"
    value_dict = _money_value_dict(report_dict)
    assert value_dict["Programs, August 31"][0] == _money_str(page_dict["report_dict"]["closing_nav_float"])
    assert value_dict["Other accounts"][0] == value_dict["Total, all accounts"][0] == "—"
    assert report_dict["draft_reason_str"] == "Other account data incomplete"
    assert build_monthly_report_dict(workspace_dict, snapshot_obj, SimpleNamespace(), as_of_ts=SAMPLE_AS_OF_TS,
        month_str="2026-09")["draft_reason_str"] == "Month in progress, through Sep 3"


def test_other_account_lines_need_an_other_account_at_the_close():
    workspace_dict, snapshot_obj = _sample_tuple(effective_to_dict={"U900": "2026-08-13"})
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, SimpleNamespace(), as_of_ts=SAMPLE_AS_OF_TS, month_str="2026-08")
    assert [row_dict["label_str"] for row_dict in report_dict["money_row_list"]] == [
        "Value on August 31", "Change in August", "Gain since start", "Money you put in, net"]


def test_a_retired_pod_keeps_its_history_but_leaves_the_split():
    workspace_dict, snapshot_obj = _sample_tuple(effective_to_dict={"U200": "2026-08-13"})
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, SimpleNamespace(), as_of_ts=SAMPLE_AS_OF_TS, month_str="2026-08")
    assert [pod_dict["name_str"] for group_dict in report_dict["split_dict"]["group_list"] for pod_dict in group_dict["pod_list"]] == ["Pod 1 · U···100"]
    assert [row_dict["name_str"] for row_dict in report_dict["month_table_dict"]["row_list"]] == ["Portfolio", "Pod 1 · U···100", "Pod 2 · U···200"]
    flow_dict = report_dict["flow_dict"]
    assert "Pod closed, money out" in [row_dict["kind_str"] for row_dict in flow_dict["row_list"]]
    assert flow_dict["total_str"] == _money_value_dict(report_dict)["Money you put in, net"][0] != "—"
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
    assert value_dict["Change in August"] == ("+$1,090", "+1.1%")
    assert value_dict["Gain since start"] == ("—", "—") and value_dict["Money you put in, net"] == ("—", "")
    assert report_dict["draft_reason_str"] == "Return since start unavailable"
    assert "Money in and out appears once the account history since start is complete." in report_dict["status_note_str"]
    assert not report_dict["program_list"][0]["chart_dict"]["available_bool"]
    assert all(row_dict["since_str"] == "—" for row_dict in report_dict["month_table_dict"]["row_list"])


def test_shorts_and_borrowing_get_no_donut_and_long_lists_fold(monkeypatch):
    workspace_dict, snapshot_obj = _sample_tuple()
    provider_obj = _patch_saved_portfolio(monkeypatch, {"pod_taa": [("TQQQ", 9000.), ("GLD", 6000.)],
        "pod_ndx": [("MU", 9000.), ("AMD", -3000.)]}, {"pod_taa": 5000., "pod_ndx": 4000.})
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=SAMPLE_AS_OF_TS,
        month_str="2026-08", database_path_str="unused.sqlite3")
    program_dict = report_dict["program_list"][0]
    assert program_dict["holding_dict"]["reason_str"] == "Holdings include borrowing or short positions. Page 2 lists them."
    assert {row_dict["symbol_str"]: row_dict["weight_str"] for row_dict in program_dict["pod_holding_list"][1]["row_list"]}["AMD"] == "−30.0%"
    provider_obj = _patch_saved_portfolio(monkeypatch, {"pod_taa": [("TQQQ", 25000.)], "pod_ndx": [("MU", 6000.)]},
        {"pod_taa": -5000., "pod_ndx": 4000.})
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=SAMPLE_AS_OF_TS,
        month_str="2026-08", database_path_str="unused.sqlite3")
    assert report_dict["program_list"][0]["holding_dict"]["reason_str"] == "Holdings include borrowing or short positions. Page 2 lists them."
    provider_obj = _patch_saved_portfolio(monkeypatch, {"pod_taa": [("TQQQ", 9000.)],
        "pod_ndx": [(f"S{index_int:02d}", 500. + index_int) for index_int in range(14)]}, {"pod_taa": 1000., "pod_ndx": 2000.})
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=SAMPLE_AS_OF_TS,
        month_str="2026-08", database_path_str="unused.sqlite3")
    row_list = report_dict["program_list"][0]["pod_holding_list"][1]["row_list"]
    assert len(row_list) == 13 and row_list[-2]["symbol_str"] == "3 more" and row_list[-2]["muted_bool"] and row_list[-1]["symbol_str"] == "Cash"
    legend_dict = {item_dict["name_str"]: item_dict["detail_str"] for item_dict in report_dict["program_list"][0]["holding_dict"]["legend_list"]}
    assert legend_dict["Nasdaq 100"] == "TQQQ 3×" and legend_dict["Stocks"] == "14 stocks"


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
    assert not report_dict["available_bool"] and report_dict["error_str"] == "No strategy Pod has saved performance yet."
    early_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, SimpleNamespace(), as_of_ts=datetime(2026, 8, 3, 13, tzinfo=UTC))
    assert early_dict["error_str"] == "No finalized account value is available yet."


def test_pods_are_named_by_number_masked_account_and_ibkr_alias():
    workspace_dict, snapshot_obj = _sample_tuple()
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, SimpleNamespace(), as_of_ts=SAMPLE_AS_OF_TS, month_str="2026-08",
        alias_by_pod_dict={"pod_ndx": "growth_mom_natr20"})
    name_list = ["Pod 1 · U···100", "Pod 2 · U···200 · growth_mom_natr20"]
    assert [pod_dict["name_str"] for pod_dict in report_dict["split_dict"]["group_list"][0]["pod_list"]] == name_list
    assert report_dict["flow_dict"]["pod_name_list"] == [{"head_str": "Pod 1 · U···100", "alias_str": ""},
        {"head_str": "Pod 2 · U···200", "alias_str": "growth_mom_natr20"}]
    # Numbers follow the first measured day: a Pod that starts later is numbered after the others.
    workspace_dict["client_dict"]["accounts"][0]["effective_from"] = "2026-08-04"
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, SimpleNamespace(), as_of_ts=SAMPLE_AS_OF_TS, month_str="2026-08")
    assert {pod_dict["name_str"] for pod_dict in report_dict["split_dict"]["group_list"][0]["pod_list"]} == {"Pod 1 · U···200", "Pod 2 · U···100"}


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
    # A cleared choice goes back under Portfolio; the other Pods keep theirs.
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
    from alpha.live.dashboard_v4.report_programs import PROGRAM_BYTE_LIMIT_INT, ReportProgramStore, clean_program_str
    provider_obj = SimpleNamespace(event_log_path_str=str(tmp_path / "events.jsonl"))
    ReportProgramStore(provider_obj).save({"pod_taa": "Growth", "pod_ndx": "Growth"})
    assert ReportProgramStore(provider_obj).program_by_pod_dict() == {"pod_taa": "Growth", "pod_ndx": "Growth"}
    ReportProgramStore(provider_obj).save({"pod_ndx": ""})
    assert ReportProgramStore(provider_obj).program_by_pod_dict() == {"pod_taa": "Growth"}
    path_obj = tmp_path / "report_programs.json"
    path_obj.write_text('{"program_by_pod_dict": {"pod_taa": "Steady", "pod_bad": "<script>"}}', encoding="utf-8")
    assert ReportProgramStore(provider_obj).program_by_pod_dict() == {"pod_taa": "Steady"}
    path_obj.write_text("x" * (PROGRAM_BYTE_LIMIT_INT + 1), encoding="utf-8")
    assert ReportProgramStore(provider_obj).program_by_pod_dict() == {}
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


def test_one_program_owns_every_pod_including_a_retired_one():
    workspace_dict, snapshot_obj = _sample_tuple(effective_to_dict={"U200": "2026-08-13"})
    report_dict = build_monthly_report_dict(workspace_dict, snapshot_obj, SimpleNamespace(), as_of_ts=SAMPLE_AS_OF_TS,
        month_str="2026-08", program_by_pod_dict={"pod_taa": "Growth"})
    assert [program_dict["name_str"] for program_dict in report_dict["program_list"]] == ["Growth"]
    # The card's return includes the retired Pod's history, so its row stays under the program.
    assert [row_dict["name_str"] for row_dict in report_dict["month_table_dict"]["row_list"]] == ["Growth", "Pod 1 · U···100", "Pod 2 · U···200"]


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
        assert client_obj.get("/monthly-report?month=2026-08").status_code == 200, payload_str
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
    response_obj = _post_programs(client_obj, {"confirmed_bool": True, "program_by_pod_dict": {"pod_taa": "Steady"}}, token_str=token_str)
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
