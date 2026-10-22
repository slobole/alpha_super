"""The client's other accounts are found without config and counted in totals, never as a Pod."""
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
import pytest

from alpha.live.dashboard_v3.local_workspace import (
    LocalReportingError, build_local_workspace_dict, validate_local_bindings_unchanged,
)
from alpha.live.dashboard_v3.app import create_app as create_v3_app
from alpha.live.dashboard_v3 import client_views
from alpha.live.dashboard_v3.client_financial_display import financial_dates_dict
from alpha.live.dashboard_v3.client_presentation import portfolio_account_list
from alpha.live.dashboard_v4.app import create_app
from alpha.live.dashboard_v4.data import LiveDataProvider, load_workspace_snapshot_tuple
from alpha.live.dashboard_v4.demo import build_demo_workspace_tuple, DEMO_NOW_TS
from alpha.live.dashboard_v4.finance import build_financial_overview_dict
from alpha.live.dashboard_v4.performance import build_performance_page_dict
from alpha.live.dashboard_v4.pod_finance import build_pod_finance_dict
from alpha.live.dashboard_v4.positions import build_positions_page_dict
from alpha.live.client_reporting import BrokerNavRow
from alpha.live.ibkr_performance import PerformanceStore
from alpha.live.ibkr_performance_sync import build_live_binding_obj_list
from test_client_reporting import client_config_dict, nav_attributes_dict, snapshot_obj as fixture_snapshot_obj, xml_text_str
from test_dashboard_local_workspace import build_fixture_app, file_snapshot_dict
from test_ibkr_nav_profile import expanded_nav_attributes_dict


AS_OF_TS = datetime(2026, 9, 4, 13, 0, tzinfo=UTC)


def _reserve_fixture(tmp_path, monkeypatch, *, interest_float=0.0):
    app_obj = build_fixture_app(tmp_path, monkeypatch, finance_bool=False,
        new_pod_bool=False, expanded_bool=True)
    provider_obj = app_obj.config["data_provider_obj"]
    config_path_obj = tmp_path / "dashboard.yaml"
    attribute_list = []
    for day_int in (1, 2, 3):
        date_str = f"2026-09-0{day_int}"
        for route_str, starting_float, gain_float in (("U100", 1000.0, 10.0), ("U200", 10000.0, 50.0)):
            opening_float = starting_float + (day_int - 1) * gain_float
            attribute_list.append(expanded_nav_attributes_dict(route_str, date_str=date_str,
                opening_str=str(opening_float), closing_str=str(opening_float + gain_float),
                mtm=str(gain_float), twr_str=str(gain_float / opening_float * 100)))
        gain_float = interest_float if day_int == 3 else 0.0
        attribute_list.append(expanded_nav_attributes_dict("URESERVE", date_str=date_str,
            opening_str="5000", closing_str=str(5000 + gain_float), mtm="0",
            interest=str(gain_float), twr_str=str(gain_float / 5000 * 100)))
        # An unused sub-account that never held money: left out, never shown.
        attribute_list.append(expanded_nav_attributes_dict("UUNUSED", date_str=date_str,
            opening_str="0", closing_str="0", mtm="0", twr_str="0"))
    performance_path_str = str(tmp_path / "performance.sqlite3")
    binding_list = build_live_binding_obj_list(releases_root_path_str=str(tmp_path / "releases"),
        config_path_str=str(config_path_obj))
    PerformanceStore(performance_path_str).replace_range(
        xml_text_str=xml_text_str(attribute_list).replace('queryName="TEST_NAV"', 'queryName="ALPHA_DAILY_TWR"').replace(
            '<AccountInformation accountId="URESERVE" currency="USD" />',
            '<AccountInformation accountId="URESERVE" currency="USD" acctAlias="Private reserve" />'),
        query_name_str="ALPHA_DAILY_TWR", request_from_date_str="2026-09-01",
        request_to_date_str="2026-09-03", binding_obj_list=binding_list,
        imported_timestamp_str=AS_OF_TS.isoformat())
    return provider_obj, performance_path_str, config_path_obj


def test_other_account_is_found_financial_only_and_reads_do_not_write(tmp_path, monkeypatch):
    provider_obj, database_path_str, _ = _reserve_fixture(tmp_path, monkeypatch)
    before_dict = file_snapshot_dict(tmp_path)
    workspace_dict, snapshot_obj = load_workspace_snapshot_tuple(provider_obj, database_path_str, as_of_ts=AS_OF_TS)
    assert workspace_dict["financial_error_str"] is None
    assert workspace_dict["financial_scope_complete_bool"]
    assert {account_dict["account_route"] for account_dict in workspace_dict["operations_account_list"]} == {"U100", "U200"}
    assert {account_dict["account_route"] for account_dict in workspace_dict["valuation_account_list"]} == {"U100", "U200", "URESERVE"}
    reserve_dict, = [account_dict for account_dict in workspace_dict["client_dict"]["accounts"] if account_dict.get("role") == "reserve"]
    assert "pod_id" not in reserve_dict
    assert (reserve_dict["display_name"], reserve_dict["effective_from"]) == ("Private reserve", "2026-09-01")
    assert {row_obj.account_route_str for row_obj in snapshot_obj.row_tuple} == {"U100", "U200", "URESERVE"}
    assert file_snapshot_dict(tmp_path) == before_dict


@pytest.mark.parametrize("interest_float", [0.0, 2.0])
def test_other_account_enters_v4_totals_and_is_listed_apart_from_pods(tmp_path, monkeypatch, interest_float):
    provider_obj, database_path_str, _ = _reserve_fixture(tmp_path, monkeypatch, interest_float=interest_float)
    workspace_dict, snapshot_obj = load_workspace_snapshot_tuple(provider_obj, database_path_str, as_of_ts=AS_OF_TS)
    before_dict = deepcopy(workspace_dict)
    finance_dict = build_financial_overview_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=AS_OF_TS)
    performance_dict = build_performance_page_dict(workspace_dict, snapshot_obj, as_of_ts=AS_OF_TS)
    report_dict = performance_dict["report_dict"]
    assert report_dict["closing_nav_float"] == 16180 + interest_float
    assert report_dict["pnl_float"] == 180 + interest_float
    assert report_dict["twr_float"] == pytest.approx((180 + interest_float) / 16000)
    assert finance_dict["tile_list"][0]["value_str"] == f"${16180 + interest_float:,.2f}"
    assert finance_dict["chart_dict"]["available_bool"]
    assert {row_dict["pod_id_str"] for row_dict in performance_dict["pod_row_list"]} == {"pod_a", "pod_b"}
    assert sorted(row_dict["twr_float"] for row_dict in report_dict["strategy_list"]) == pytest.approx([.015, .03])
    assert [row_dict["name_str"] for row_dict in finance_dict["allocation_dict"]["row_list"]][-1] == "Private reserve (not traded)"
    assert len(finance_dict["allocation_dict"]["row_list"]) == 3
    assert finance_dict["allocation_dict"]["total_value_str"] == f"${16180 + interest_float:,.2f}"
    assert finance_dict["allocation_dict"]["cash_complete_bool"] is False
    assert finance_dict["allocation_dict"]["cash_percent_str"] == "—"
    assert len(finance_dict["allocation_dict"]["slice_list"]) == 3
    assert any(slice_dict["label_str"].startswith("Private reserve (not traded) value")
        for slice_dict in finance_dict["allocation_dict"]["slice_list"])
    other_row_dict, = performance_dict["other_row_list"]
    assert (other_row_dict["name_str"], other_row_dict["end_str"]) == ("Private reserve", f"${5000 + interest_float:,.2f}")
    assert performance_dict["contribution_dict"]["available_bool"]
    assert {row_dict["name_str"] for row_dict in performance_dict["contribution_dict"]["row_list"]} == {"Private reserve", *(
        row_dict["name_str"] for row_dict in performance_dict["pod_row_list"])}
    assert "UUNUSED" not in str(performance_dict) and "UUNUSED" not in str(finance_dict)
    pod_view_dict = build_performance_page_dict(workspace_dict, snapshot_obj, as_of_ts=AS_OF_TS, level_str="pods")
    assert len(pod_view_dict["monthly_row_list"]) == 2
    assert workspace_dict == before_dict


def test_new_other_account_during_a_read_requires_a_refresh(tmp_path, monkeypatch):
    provider_obj, database_path_str, _ = _reserve_fixture(tmp_path, monkeypatch)
    workspace_dict = build_local_workspace_dict(provider_obj, database_path_str, today_str="2026-09-04")
    validate_local_bindings_unchanged(workspace_dict, database_path_str)
    workspace_dict["other_account_list"] = []
    with pytest.raises(LocalReportingError, match="changed during this read"):
        validate_local_bindings_unchanged(workspace_dict, database_path_str)


def test_strategy_account_is_never_an_other_account(tmp_path, monkeypatch):
    provider_obj, database_path_str, _ = _reserve_fixture(tmp_path, monkeypatch)
    workspace_dict = build_local_workspace_dict(provider_obj, database_path_str, today_str="2026-09-04")
    assert [account_dict["account_route"] for account_dict in workspace_dict["other_account_list"]] == ["URESERVE"]
    assert {account_dict["account_route"] for account_dict in workspace_dict["client_dict"]["accounts"]
        if account_dict.get("role") != "reserve"} == {"U100", "U200"}


@pytest.mark.parametrize("missing_date_str", ["2026-09-02", "2026-09-03"])
def test_missing_reserve_day_does_not_unlock_strategy_only_client_return(tmp_path, monkeypatch, missing_date_str):
    provider_obj, database_path_str, _ = _reserve_fixture(tmp_path, monkeypatch)
    workspace_dict, snapshot_obj = load_workspace_snapshot_tuple(provider_obj, database_path_str, as_of_ts=AS_OF_TS)
    snapshot_obj = replace(snapshot_obj, row_tuple=tuple(row_obj for row_obj in snapshot_obj.row_tuple
        if not (row_obj.account_route_str == "URESERVE" and row_obj.market_date_str == missing_date_str)))
    performance_dict = build_performance_page_dict(workspace_dict, snapshot_obj, as_of_ts=AS_OF_TS,
        from_date_str="2026-09-01", to_date_str="2026-09-03")
    report_dict = performance_dict["report_dict"]
    assert report_dict["twr_float"] is None
    assert report_dict["pnl_float"] is None
    assert len(report_dict["strategy_list"]) == 2
    assert all(strategy_dict["twr_float"] is not None for strategy_dict in report_dict["strategy_list"])


def test_v4_render_has_only_strategy_cards_and_other_account_is_not_order_addressable(tmp_path, monkeypatch):
    source_provider_obj, database_path_str, _ = _reserve_fixture(tmp_path, monkeypatch)
    source_app_obj = source_provider_obj.app_obj()
    provider_obj = LiveDataProvider(releases_root_path_str=source_app_obj.releases_root_path_str,
        config_path_str=source_app_obj.config_path_str, results_root_path_str=source_app_obj.results_root_path_str,
        event_log_path_str=source_app_obj.event_log_path_str)
    app_obj = create_app(provider_obj, performance_db_path_str=database_path_str, now_fn=lambda: AS_OF_TS)
    before_dict = file_snapshot_dict(tmp_path)
    client_obj = app_obj.test_client()
    response_obj = client_obj.get("/")
    assert response_obj.status_code == 200
    html_str = response_obj.get_data(as_text=True)
    assert html_str.count('data-pod-id="') == 2
    assert 'data-pod-id="pod_a"' in html_str and 'data-pod-id="pod_b"' in html_str
    assert "$16,180.00" in html_str
    assert "URESERVE" not in html_str and "UUNUSED" not in html_str
    performance_html_str = client_obj.get("/performance?level=pods&from=2026-09-01&to=2026-09-03").get_data(as_text=True)
    assert "Private reserve <span class=\"muted\">· not traded</span>" in performance_html_str
    assert "URESERVE" not in performance_html_str and "UUNUSED" not in performance_html_str
    assert client_obj.get("/pods/URESERVE").status_code == 404
    assert client_obj.get("/tools?pod=URESERVE").status_code == 404
    assert file_snapshot_dict(tmp_path) == before_dict


def test_reserve_first_does_not_hide_pod_money_or_positions(tmp_path, monkeypatch):
    provider_obj, database_path_str, _ = _reserve_fixture(tmp_path, monkeypatch)
    workspace_dict, snapshot_obj = load_workspace_snapshot_tuple(provider_obj, database_path_str, as_of_ts=AS_OF_TS)
    for account_list in (workspace_dict["client_dict"]["accounts"], workspace_dict["valuation_account_list"]):
        account_list.sort(key=lambda account_dict: account_dict.get("role") != "reserve")
    performance_dict = build_performance_page_dict(workspace_dict, snapshot_obj, as_of_ts=AS_OF_TS, level_str="pods")
    pod_dict = build_pod_finance_dict(workspace_dict, snapshot_obj, provider_obj, pod_id_str="pod_a", as_of_ts=AS_OF_TS)
    positions_dict = build_positions_page_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=AS_OF_TS)
    assert len(performance_dict["pod_row_list"]) == 2
    assert pod_dict["tile_list"][0]["value_str"] == "$1,030.00"
    assert len(positions_dict["pod_filter_list"]) == 2
    assert {row_dict["pod_id_str"] for row_dict in positions_dict["pod_filter_list"]} == {"pod_a", "pod_b"}


@pytest.mark.parametrize("scope_complete_bool", [True, False])
def test_v3_reserve_first_pages_keep_operations_and_partial_history_accessible(tmp_path, monkeypatch, scope_complete_bool):
    provider_obj, database_path_str, _ = _reserve_fixture(tmp_path, monkeypatch)
    workspace_dict, _ = load_workspace_snapshot_tuple(provider_obj, database_path_str, as_of_ts=AS_OF_TS)
    workspace_dict["financial_scope_complete_bool"] = scope_complete_bool
    if not scope_complete_bool:
        workspace_dict["client_dict"]["accounts"] = [account_dict for account_dict in workspace_dict["client_dict"]["accounts"]
            if account_dict["account_route"] != "U200"]
    for account_list in (workspace_dict["client_dict"]["accounts"], workspace_dict["valuation_account_list"]):
        account_list.sort(key=lambda account_dict: account_dict.get("role") != "reserve")
    monkeypatch.setattr(client_views, "build_local_workspace_dict", lambda *_args, **_kwargs: deepcopy(workspace_dict))
    app_obj = create_v3_app(provider_obj, read_only_bool=True, performance_db_path_str=database_path_str,
        journal_path_str=str(tmp_path / "journal.jsonl"), notification_state_path_str=str(tmp_path / "notifications.json"))
    client_obj = app_obj.test_client()
    for view_str in ("overview", "performance", "strategies", "activity", "diagnostics", "report"):
        response_obj = client_obj.get(f"/clients/local/{view_str}?from=2026-09-01&to=2026-09-03")
        assert response_obj.status_code == 200, (view_str, response_obj.status_code)


def test_dated_reserve_does_not_block_financial_dates_or_allocation_after_exit():
    client_dict = client_config_dict()
    reserve_dict = {"account_route": "U_TEST_B", "role": "reserve", "display_name": "Reserve",
        "effective_from": "2026-09-02", "effective_to": "2026-09-02"}
    client_dict["accounts"].append(reserve_dict)
    valuation_list = [{field_str: client_dict["accounts"][0][field_str]
        for field_str in ("account_route", "pod_id", "display_name")}, reserve_dict]
    source_obj = fixture_snapshot_obj([
        nav_attributes_dict(date_str=f"2026-09-0{day_int}") for day_int in (1, 2, 3)
    ] + [nav_attributes_dict("U_TEST_B", date_str="2026-09-02")])
    assert financial_dates_dict(client_dict, source_obj, as_of_ts=AS_OF_TS,
        valuation_account_list=valuation_list)["latest_complete_str"] == "2026-09-03"
    for close_str, expected_int in (("2026-09-01", 1), ("2026-09-02", 2), ("2026-09-03", 1)):
        assert len(portfolio_account_list({"closing_date_str": close_str,
            "valuation_account_list": valuation_list})) == expected_int


def test_month_positions_reserve_interest_uses_client_total_and_existing_other_bucket():
    workspace_dict, source_obj, provider_obj = build_demo_workspace_tuple(include_holdings_bool=True)
    try:
        workspace_dict["operations_account_list"] = deepcopy(workspace_dict["operations_account_list"])
        workspace_dict["valuation_account_list"] = deepcopy(workspace_dict["valuation_account_list"])
        workspace_dict["client_dict"]["nav_bridge"]["economic_fields"].append("interest")
        reserve_dict = {"role": "reserve", "account_route": "URESERVE", "display_name": "Private reserve",
            "effective_from": "2026-09-01", "effective_to": None}
        workspace_dict["client_dict"]["accounts"].insert(0, reserve_dict)
        workspace_dict["valuation_account_list"].insert(0, deepcopy(reserve_dict))
        row_list = [replace(row_obj, attribute_dict={**row_obj.attribute_dict, "interest": "0"}) for row_obj in source_obj.row_tuple]
        for date_str in sorted({row_obj.market_date_str for row_obj in row_list if row_obj.market_date_str >= "2026-09-01"}):
            gain_decimal = Decimal("2") if date_str == "2026-09-04" else Decimal(0)
            attribute_dict = expanded_nav_attributes_dict("URESERVE", date_str=date_str,
                opening_str="5000", closing_str=str(5000 + gain_decimal), mtm="0", interest=str(gain_decimal), twr_str="0")
            row_list.append(BrokerNavRow("URESERVE", date_str, Decimal(5000), 5000 + gain_decimal,
                gain_decimal / 5000, attribute_dict, source_obj.row_tuple[0].source_import_id_int,
                source_obj.row_tuple[0].source_checksum_str))
        source_obj = replace(source_obj, row_tuple=tuple(row_list))
        performance_dict = build_performance_page_dict(workspace_dict, source_obj, as_of_ts=DEMO_NOW_TS,
            from_date_str="2026-09-01", to_date_str="2026-09-04")
        month_dict = build_positions_page_dict(workspace_dict, source_obj, provider_obj,
            as_of_ts=DEMO_NOW_TS, month_str="2026-09")["month_dict"]
        assert month_dict["available_bool"], month_dict["reason_str"]
        assert month_dict["total_dict"]["pnl_str"] == "+$" + f"{performance_dict['report_dict']['pnl_float']:,.2f}"
        row_pnl_float = sum(row_dict["sort_dict"]["pnl"] for row_dict in month_dict["row_list"])
        other_float = sum(bar_dict["value_float"] for bar_dict in month_dict["side_dict"]["pod_bar_list"] if bar_dict.get("other_bool"))
        assert row_pnl_float + other_float == pytest.approx(performance_dict["report_dict"]["pnl_float"], abs=.05)
        assert other_float == pytest.approx(2, abs=.05)
        book_day_list = performance_dict["report_dict"]["daily_book_list"]
        average_nav_float = sum(day_dict["nav_float"] for day_dict in book_day_list) / len(book_day_list)
        assert month_dict["map_dict"]["book_value_float"] == pytest.approx(average_nav_float)
        assert month_dict["total_dict"]["nav_str"] == "$" + f"{average_nav_float:,.2f}"
        other_dict = next(bar_dict for bar_dict in month_dict["side_dict"]["pod_bar_list"] if bar_dict.get("other_bool"))
        assert other_dict["title_str"].startswith("Account P&L not attributed to displayed positions.")
        assert "Private reserve" not in str(month_dict)
        assert len(month_dict["pod_row_list"]) == 4
    finally:
        provider_obj.close()


def test_client_report_pdf_and_csv_list_the_other_account(tmp_path, monkeypatch):
    from alpha.live.dashboard_v4.performance_exports import export_performance_csv_str, export_performance_pdf_bytes
    from alpha.live.investor_report import build_investor_snapshot_dict

    provider_obj, database_path_str, _ = _reserve_fixture(tmp_path, monkeypatch, interest_float=2.0)
    workspace_dict, snapshot_obj = load_workspace_snapshot_tuple(provider_obj, database_path_str, as_of_ts=AS_OF_TS)
    report_dict = build_performance_page_dict(workspace_dict, snapshot_obj, as_of_ts=AS_OF_TS)["report_dict"]
    other_dict, = build_investor_snapshot_dict(report_dict)["other_account_list"]
    assert (other_dict["display_name_str"], other_dict["closing_nav_float"], other_dict["pnl_float"]) == ("Private reserve", 5002.0, 2.0)
    assert "URESERVE" not in str(build_investor_snapshot_dict(report_dict))
    assert export_performance_pdf_bytes(report_dict).startswith(b"%PDF")
    csv_line_list = export_performance_csv_str(report_dict, level_str="pods").splitlines()
    assert [line_str.split(",")[0] for line_str in csv_line_list[1:]][-1] == "Private reserve"
    assert len(csv_line_list) == 4


def test_correction_dropping_the_other_account_withholds_totals_instead_of_shrinking_them(tmp_path, monkeypatch):
    provider_obj, database_path_str, _ = _reserve_fixture(tmp_path, monkeypatch)
    _, snapshot_obj = load_workspace_snapshot_tuple(provider_obj, database_path_str, as_of_ts=AS_OF_TS)
    attribute_list = [dict(row_obj.attribute_dict) for row_obj in snapshot_obj.row_tuple if row_obj.account_route_str != "URESERVE"]
    binding_list = build_live_binding_obj_list(releases_root_path_str=str(tmp_path / "releases"),
        config_path_str=str(tmp_path / "dashboard.yaml"))
    # A full-range Flex correction that omits the reserve.
    PerformanceStore(database_path_str).replace_range(
        xml_text_str=xml_text_str(attribute_list).replace('queryName="TEST_NAV"', 'queryName="ALPHA_DAILY_TWR"'),
        query_name_str="ALPHA_DAILY_TWR", request_from_date_str="2026-09-01", request_to_date_str="2026-09-03",
        binding_obj_list=binding_list, imported_timestamp_str=AS_OF_TS.isoformat())
    workspace_dict, snapshot_obj = load_workspace_snapshot_tuple(provider_obj, database_path_str, as_of_ts=AS_OF_TS)
    assert [account_dict["account_route"] for account_dict in workspace_dict["other_account_list"]] == ["URESERVE"]
    report_dict = build_performance_page_dict(workspace_dict, snapshot_obj, as_of_ts=AS_OF_TS,
        from_date_str="2026-09-01", to_date_str="2026-09-03")["report_dict"]
    assert (report_dict["closing_nav_float"], report_dict["pnl_float"], report_dict["twr_float"], report_dict["status_str"]) == (
        None, None, None, "draft")


def test_other_account_handing_over_to_a_pod_keeps_client_history(tmp_path, monkeypatch):
    provider_obj, database_path_str, _ = _reserve_fixture(tmp_path, monkeypatch, interest_float=2.0)
    workspace_dict, snapshot_obj = load_workspace_snapshot_tuple(provider_obj, database_path_str, as_of_ts=AS_OF_TS)
    before_dict = build_performance_page_dict(workspace_dict, snapshot_obj, as_of_ts=AS_OF_TS,
        from_date_str="2026-09-01", to_date_str="2026-09-03")["report_dict"]
    # The same money, but URESERVE becomes Pod pod_c from 2026-09-02: an other
    # account on 09-01, the Pod after. No Flex fact changes.
    handover_dict = deepcopy(workspace_dict)
    handover_dict["client_dict"]["accounts"] = [account_dict for account_dict in handover_dict["client_dict"]["accounts"]
        if account_dict["account_route"] != "URESERVE"] + [
        {"role": "reserve", "account_route": "URESERVE", "display_name": "Private reserve",
            "effective_from": "2026-09-01", "effective_to": "2026-09-01"},
        {"pod_id": "pod_c", "account_route": "URESERVE", "display_name": "Pod C",
            "effective_from": "2026-09-02", "effective_to": None}]
    handover_dict["valuation_account_list"] = [account_dict for account_dict in handover_dict["valuation_account_list"]
        if account_dict["account_route"] != "URESERVE"] + [{"pod_id": "pod_c", "account_route": "URESERVE", "display_name": "Pod C"}]
    after_dict = build_performance_page_dict(handover_dict, snapshot_obj, as_of_ts=AS_OF_TS,
        from_date_str="2026-09-01", to_date_str="2026-09-03")["report_dict"]
    for field_str in ("opening_nav_float", "closing_nav_float", "pnl_float", "twr_float", "scope_movement_float"):
        assert after_dict[field_str] == pytest.approx(before_dict[field_str]), field_str
    assert after_dict["status_str"] == before_dict["status_str"] == "ready"
    assert "pod_c" in {strategy_dict["pod_id_str"] for strategy_dict in after_dict["strategy_list"]}
