"""Primary strategy performance and full client capital have distinct scopes."""

from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
import csv
import hashlib
from io import BytesIO, StringIO
from urllib.parse import parse_qs, urlsplit

from flask import template_rendered
from pypdf import PdfReader
import pytest

from alpha.live.client_reporting import BrokerReportingSnapshot, parse_broker_nav_import
from alpha.live.dashboard_v4.app import create_app
from alpha.live.dashboard_v4.finance import build_financial_overview_dict
from alpha.live.dashboard_v4.performance import build_performance_page_dict
from alpha.live.dashboard_v4.performance_exports import export_performance_csv_str, export_performance_pdf_bytes
from alpha.live.ibkr_nav_profile import local_ibkr_reporting_profile_dict
from test_client_reporting import xml_text_str
from test_dashboard_operator_access import ForbiddenProvider
from test_ibkr_nav_profile import expanded_nav_attributes_dict


AS_OF_TS = datetime(2026, 9, 4, 13, tzinfo=UTC)


def _fixture_tuple(account_list, attribute_list, *, as_of_ts=AS_OF_TS):
    raw_xml_str = xml_text_str(attribute_list)
    checksum_str = hashlib.sha256(raw_xml_str.encode()).hexdigest()
    row_list = parse_broker_nav_import(raw_xml_str, allowed_account_set={account_dict["account_route"] for account_dict in account_list},
        query_name_str="TEST_NAV", source_import_id_int=1, source_checksum_str=checksum_str)
    snapshot_obj = BrokerReportingSnapshot(tuple(row_list), ({"import_id_int": 1,
        "checksum_str": checksum_str, "query_name_str": "TEST_NAV", "imported_timestamp_str": as_of_ts.isoformat()},))
    identity_dict = {account_dict["account_route"]: {field_str: account_dict[field_str] for field_str in
        ("account_route", "display_name", "pod_id", "role") if field_str in account_dict} for account_dict in account_list}
    # A real strategy identity owns valuation even when the same account had an
    # earlier reserve period. No operational positions/status decides ownership.
    for account_dict in account_list:
        if account_dict.get("role", "strategy") == "strategy":
            identity_dict[account_dict["account_route"]] = {field_str: account_dict[field_str] for field_str in
                ("account_route", "display_name", "pod_id")}
    client_dict = {"client_id": "sample", "display_name": "Sample client", "base_currency": "USD",
        "fee_basis": "Synthetic saved IBKR facts", "query_name": "TEST_NAV",
        "mandate_start_date": min(account_dict["effective_from"] for account_dict in account_list),
        "accounts": deepcopy(account_list), **local_ibkr_reporting_profile_dict()}
    strategy_list = [account_dict for account_dict in account_list if account_dict.get("role", "strategy") == "strategy"]
    workspace_dict = {"client_dict": client_dict, "financial_scope_complete_bool": True,
        "financial_error_str": None, "valuation_account_list": list(identity_dict.values()),
        "operations_account_list": [account_dict for account_dict in strategy_list if not account_dict.get("effective_to")],
        "summary_dict": {"pod_list": [{"pod_id_str": account_dict["pod_id"], "account_route_str": account_dict["account_route"],
            "enabled_bool": not bool(account_dict.get("effective_to")), "status_str": "Action needed",
            "positions_list": []} for account_dict in strategy_list]}}
    return workspace_dict, snapshot_obj


def _strategy_dict(route_str, pod_str, *, from_str="2026-09-01", to_str=None):
    return {"account_route": route_str, "pod_id": pod_str, "display_name": pod_str,
        "effective_from": from_str, "effective_to": to_str}


def _vault_fixture_tuple(*, internal_transfer_bool=False):
    account_list = [_strategy_dict("U100", "pod_invested"), _strategy_dict("U200", "pod_cash"),
        {"account_route": "U900", "display_name": "Vault", "role": "reserve",
            "effective_from": "2026-09-01", "effective_to": None}]
    attribute_list = []
    for day_int in (1, 2, 3):
        date_str = f"2026-09-0{day_int}"
        opening_int = 1000 + (day_int - 1) * 10
        transfer_int = 500 if internal_transfer_bool and day_int == 3 else 0
        attribute_list.append(expanded_nav_attributes_dict("U100", date_str=date_str,
            opening_str=str(opening_int), closing_str=str(opening_int + 10 + transfer_int), mtm="10",
            internalCashTransfers=str(transfer_int), twr_str=str(10 / opening_int * 100)))
        attribute_list.append(expanded_nav_attributes_dict("U200", date_str=date_str,
            opening_str="10000", closing_str="10000", mtm="0", twr_str="0"))
        gain_int = 4000 if day_int == 3 else 0
        attribute_list.append(expanded_nav_attributes_dict("U900", date_str=date_str,
            opening_str="5000", closing_str=str(5000 + gain_int), mtm=str(gain_int), twr_str=str(gain_int / 5000 * 100)))
    return _fixture_tuple(account_list, attribute_list)


def _eran_fixture_tuple():
    account_list = [{"account_route": "U100", "display_name": "Before launch", "role": "reserve",
        "effective_from": "2026-09-18", "effective_to": "2026-09-23"},
        _strategy_dict("U100", "pod_eran", from_str="2026-09-24")]
    attribute_list = [expanded_nav_attributes_dict("U100", date_str="2026-09-18", opening_str="0",
        closing_str="10000", depositsWithdrawals="10000", mtm="0", twr_str="0")]
    attribute_list.extend(expanded_nav_attributes_dict("U100", date_str=f"2026-09-{day_int}",
        opening_str="10000", closing_str="10000", mtm="0", twr_str="0") for day_int in (21, 22, 23))
    attribute_list.extend(expanded_nav_attributes_dict("U100", date_str=f"2026-09-{day_int}",
        opening_str=str(opening_int), closing_str=str(opening_int + 100), mtm="100",
        twr_str=str(100 / opening_int * 100)) for day_int, opening_int in ((24, 10000), (25, 10100)))
    return _fixture_tuple(account_list, attribute_list, as_of_ts=datetime(2026, 9, 26, 13, tzinfo=UTC))


def test_eran_all_performance_starts_at_pod_measurement_not_zero_nav_deposit():
    workspace_dict, snapshot_obj = _eran_fixture_tuple()
    before_dict = deepcopy(workspace_dict)
    page_dict = build_performance_page_dict(workspace_dict, snapshot_obj,
        as_of_ts=datetime(2026, 9, 26, 13, tzinfo=UTC), period_str="All")
    assert page_dict["from_date_str"] == "2026-09-24"
    assert page_dict["report_dict"]["opening_nav_float"] == 10000
    assert page_dict["report_dict"]["closing_nav_float"] == 10200
    assert page_dict["report_dict"]["pnl_float"] == 200
    assert page_dict["report_dict"]["twr_float"] == pytest.approx(.02)
    assert page_dict["client_report_dict"]["closing_nav_float"] == 10200
    assert workspace_dict == before_dict


def test_eran_overview_all_chart_starts_at_valid_strategy_date_and_keeps_account_value():
    workspace_dict, snapshot_obj = _eran_fixture_tuple()
    finance_dict = build_financial_overview_dict(workspace_dict, snapshot_obj, ForbiddenProvider(),
        as_of_ts=datetime(2026, 9, 26, 13, tzinfo=UTC), period_str="All")
    assert finance_dict["tile_list"][0]["value_str"] == "$10,200.00"
    assert finance_dict["tile_list"][2]["value_str"] == "+2.00%"
    assert finance_dict["tile_list"][3]["value_str"] == "+2.00%"
    assert finance_dict["chart_dict"]["available_bool"]
    assert finance_dict["chart_dict"]["x_tick_list"][0]["label_str"] == "Sep 24"


def test_vault_mtm_does_not_change_primary_strategy_day_return_or_remove_cash_pod():
    workspace_dict, snapshot_obj = _vault_fixture_tuple()
    finance_dict = build_financial_overview_dict(workspace_dict, snapshot_obj, ForbiddenProvider(), as_of_ts=AS_OF_TS)
    assert finance_dict["tile_list"][0]["value_str"] == "$20,030.00"
    assert finance_dict["tile_list"][1]["value_str"] == "+$10.00"
    assert finance_dict["tile_list"][1]["detail_str"] == "+0.09%"
    assert finance_dict["tile_list"][2]["value_str"] == "+0.27%"
    page_dict = build_performance_page_dict(workspace_dict, snapshot_obj, as_of_ts=AS_OF_TS)
    report_dict = page_dict["report_dict"]
    assert report_dict["opening_nav_float"] == 11000
    assert report_dict["closing_nav_float"] == 11030
    assert report_dict["pnl_float"] == 30
    assert report_dict["twr_float"] == pytest.approx(30 / 11000)
    assert {strategy_dict["pod_id_str"] for strategy_dict in report_dict["strategy_list"]} == {"pod_invested", "pod_cash"}


def test_full_client_pdf_and_csv_keep_vault_capital_and_its_mtm():
    workspace_dict, snapshot_obj = _vault_fixture_tuple()
    page_dict = build_performance_page_dict(workspace_dict, snapshot_obj, as_of_ts=AS_OF_TS)
    client_report_dict = page_dict["client_report_dict"]
    assert client_report_dict["closing_nav_float"] == 20030
    assert client_report_dict["pnl_float"] == 4030
    assert client_report_dict["twr_float"] == pytest.approx(4030 / 16000)
    csv_row_dict, = list(csv.DictReader(StringIO(export_performance_csv_str(client_report_dict))))
    assert csv_row_dict["End value"] == "20030.00"
    assert csv_row_dict["Profit / loss"] == "4030.00"
    pdf_text_str = "\n".join(page_obj.extract_text() for page_obj in PdfReader(BytesIO(export_performance_pdf_bytes(client_report_dict))).pages)
    assert "20,030" in pdf_text_str and "4,030" in pdf_text_str and "Vault" in pdf_text_str


def test_missing_vault_day_does_not_block_parsable_strategy_performance():
    workspace_dict, snapshot_obj = _vault_fixture_tuple()
    snapshot_obj = replace(snapshot_obj, row_tuple=tuple(row_obj for row_obj in snapshot_obj.row_tuple
        if not (row_obj.account_route_str == "U900" and row_obj.market_date_str == "2026-09-02")))
    page_dict = build_performance_page_dict(workspace_dict, snapshot_obj, as_of_ts=AS_OF_TS)
    assert page_dict["report_dict"]["twr_float"] == pytest.approx(30 / 11000)
    assert page_dict["report_dict"]["coverage_complete_bool"]
    assert page_dict["client_report_dict"]["twr_float"] is None
    finance_dict = build_financial_overview_dict(workspace_dict, snapshot_obj, ForbiddenProvider(), as_of_ts=AS_OF_TS)
    assert finance_dict["tile_list"][1]["value_str"] == "+$10.00"
    assert finance_dict["chart_dict"]["available_bool"]


def test_missing_strategy_session_still_blocks_all_strategy_return():
    workspace_dict, snapshot_obj = _vault_fixture_tuple()
    snapshot_obj = replace(snapshot_obj, row_tuple=tuple(row_obj for row_obj in snapshot_obj.row_tuple
        if not (row_obj.account_route_str == "U100" and row_obj.market_date_str == "2026-09-02")))
    page_dict = build_performance_page_dict(workspace_dict, snapshot_obj, as_of_ts=AS_OF_TS)
    assert page_dict["report_dict"]["twr_float"] is None
    assert not page_dict["report_dict"]["coverage_complete_bool"]
    finance_dict = build_financial_overview_dict(workspace_dict, snapshot_obj, ForbiddenProvider(), as_of_ts=AS_OF_TS)
    assert not finance_dict["chart_dict"]["available_bool"]


def test_retired_strategy_keeps_its_dated_history_even_if_not_currently_enabled():
    account_list = [_strategy_dict("U100", "pod_current"), _strategy_dict("U200", "pod_retired", to_str="2026-09-02")]
    attribute_list = [expanded_nav_attributes_dict("U100", date_str=f"2026-09-0{day_int}",
        opening_str=str(1000 + (day_int - 1) * 10), closing_str=str(1000 + day_int * 10), mtm="10",
        twr_str=str(10 / (1000 + (day_int - 1) * 10) * 100)) for day_int in (1, 2, 3)]
    attribute_list.extend(expanded_nav_attributes_dict("U200", date_str=f"2026-09-0{day_int}",
        opening_str=str(200 + (day_int - 1) * 50), closing_str=str(200 + day_int * 50), mtm="50",
        twr_str=str(50 / (200 + (day_int - 1) * 50) * 100)) for day_int in (1, 2))
    workspace_dict, snapshot_obj = _fixture_tuple(account_list, attribute_list)
    page_dict = build_performance_page_dict(workspace_dict, snapshot_obj, as_of_ts=AS_OF_TS)
    report_dict = page_dict["report_dict"]
    assert {strategy_dict["pod_id_str"] for strategy_dict in report_dict["strategy_list"]} == {"pod_current", "pod_retired"}
    assert report_dict["pnl_float"] == 130
    assert report_dict["twr_float"] == pytest.approx((1 + 60 / 1200) * (1 + 60 / 1260) * (1 + 10 / 1020) - 1)


def test_zero_opening_capital_inside_a_pod_period_still_blocks_return():
    account_list = [_strategy_dict("U100", "pod_zero")]
    attribute_list = [expanded_nav_attributes_dict("U100", date_str="2026-09-01", opening_str="0",
        closing_str="1000", mtm="0", depositsWithdrawals="1000", twr_str="0"),
        expanded_nav_attributes_dict("U100", date_str="2026-09-02", opening_str="1000",
            closing_str="1010", mtm="10", twr_str="1")]
    workspace_dict, snapshot_obj = _fixture_tuple(account_list, attribute_list)
    page_dict = build_performance_page_dict(workspace_dict, snapshot_obj, as_of_ts=AS_OF_TS)
    assert page_dict["report_dict"]["twr_float"] is None
    assert "Positive opening" in page_dict["report_dict"]["twr_reason_str"]


def test_unmatched_internal_transfer_still_blocks_strategy_return():
    workspace_dict, snapshot_obj = _vault_fixture_tuple(internal_transfer_bool=True)
    page_dict = build_performance_page_dict(workspace_dict, snapshot_obj, as_of_ts=AS_OF_TS)
    assert page_dict["report_dict"]["twr_float"] is None
    assert "transit" in page_dict["report_dict"]["twr_reason_str"]


def test_client_download_uses_full_client_hash_and_exact_selected_dates():
    workspace_dict, snapshot_obj = _vault_fixture_tuple()
    app_obj = create_app(ForbiddenProvider(), now_fn=lambda: AS_OF_TS,
        workspace_snapshot_fn=lambda: (workspace_dict, snapshot_obj))
    context_list = []

    def capture_context(sender_obj, **event_dict):
        context_list.append(event_dict["context"])

    client_obj = app_obj.test_client()
    with template_rendered.connected_to(capture_context, app_obj):
        response_obj = client_obj.get("/performance?level=pods")
    assert response_obj.status_code == 200
    page_dict = context_list[-1]["performance_page_dict"]
    client_hash_str = page_dict["client_report_dict"]["report_hash_str"]
    strategy_hash_str = page_dict["report_dict"]["report_hash_str"]
    assert client_hash_str != strategy_hash_str
    query_dict = parse_qs(urlsplit(page_dict["csv_url_str"]).query)
    assert (query_dict["from"], query_dict["to"], query_dict["expected"]) == (
        ["2026-09-01"], ["2026-09-03"], [client_hash_str])
    csv_response_obj = client_obj.get(page_dict["csv_url_str"])
    assert csv_response_obj.status_code == 200
    csv_row_list = list(csv.DictReader(StringIO(csv_response_obj.get_data(as_text=True))))
    assert sum(float(row_dict["Profit / loss"]) for row_dict in csv_row_list) == 4030
    assert csv_row_list[-1]["Name"] == "Vault"
    assert client_obj.get(page_dict["pdf_url_str"]).data.startswith(b"%PDF")
    assert client_obj.get(page_dict["csv_url_str"].replace(client_hash_str, strategy_hash_str)).status_code == 409
    html_str = response_obj.get_data(as_text=True)
    assert "Client CSV" in html_str and "Client PDF" in html_str
    assert 'performance-other-accounts' in html_str


def test_only_other_accounts_with_stale_data_still_have_matching_client_download_dates():
    workspace_dict, snapshot_obj = _vault_fixture_tuple()
    workspace_dict["client_dict"]["accounts"] = [account_dict for account_dict in workspace_dict["client_dict"]["accounts"]
        if account_dict.get("role") == "reserve"]
    workspace_dict["valuation_account_list"] = [account_dict for account_dict in workspace_dict["valuation_account_list"]
        if account_dict.get("role") == "reserve"]
    workspace_dict["operations_account_list"] = []
    workspace_dict["summary_dict"]["pod_list"] = []
    stale_as_of_ts = datetime(2026, 9, 10, 13, tzinfo=UTC)
    app_obj = create_app(ForbiddenProvider(), now_fn=lambda: stale_as_of_ts,
        workspace_snapshot_fn=lambda: (workspace_dict, snapshot_obj))
    context_list = []

    def capture_context(sender_obj, **event_dict):
        context_list.append(event_dict["context"])

    client_obj = app_obj.test_client()
    with template_rendered.connected_to(capture_context, app_obj):
        response_obj = client_obj.get("/performance")
    assert response_obj.status_code == 200
    page_dict = context_list[-1]["performance_page_dict"]
    assert page_dict["report_dict"] is None
    assert page_dict["client_report_dict"]["pnl_float"] == 4000
    assert (page_dict["from_date_str"], page_dict["to_date_str"]) == ("2026-09-01", "2026-09-03")
    assert client_obj.get(page_dict["csv_url_str"]).status_code == 200
    assert client_obj.get(page_dict["pdf_url_str"]).data.startswith(b"%PDF")
