"""Independent hand-calculated cases for the explicit daily EOD convention."""
from datetime import UTC, datetime
from io import BytesIO

import pytest
from pypdf import PdfReader

from alpha.live.client_reporting import ClientReportingError, build_client_report_dict
from test_client_reporting import client_config_dict, nav_attributes_dict, report_dict, snapshot_obj
from alpha.live.investor_report import build_investor_snapshot_dict, render_investor_pdf_bytes


def twr_config_dict(second_bool=True):
    config_dict = client_config_dict(second_bool=second_bool)
    config_dict["client_twr"] = {"method": "daily_nav_eod_v1", "reviewed_by": "Synthetic fixture",
                                "evidence_ref": "Hand-calculated EOD tests; no real accounts"}
    return config_dict


def test_unequal_capital_uses_client_nav_not_average_account_returns():
    result_dict = report_dict([nav_attributes_dict(),
        nav_attributes_dict("U_TEST_B", opening_str="10000", closing_str="10050", twr_str=".5", mtm="50")],
        config_dict=twr_config_dict())
    assert result_dict["twr_float"] == pytest.approx(60 / 11000)
    assert result_dict["twr_float"] != pytest.approx((.01 + .005) / 2)
    assert result_dict["return_path_list"][0]["cumulative_return_float"] == 0
    assert result_dict["return_path_list"][-1]["cumulative_return_float"] == result_dict["twr_float"]
    assert result_dict["twr_daily_list"][0]["opening_nav_float"] == 11000
    assert result_dict["twr_method_id_str"] == "daily_nav_eod_v1"


def test_linked_returns_not_sum_and_selected_period_restarts_at_sod():
    row_list = [nav_attributes_dict(opening_str="100", closing_str="110", twr_str="10"),
        nav_attributes_dict(date_str="2026-09-02", opening_str="110", closing_str="99", twr_str="-10", mtm="-11")]
    config_dict = twr_config_dict(False)
    result_dict = report_dict(row_list, config_dict=config_dict, to_str="2026-09-02")
    assert result_dict["twr_float"] == pytest.approx(-.01)
    selected_dict = build_client_report_dict(config_dict, snapshot_obj(row_list), from_date_str="2026-09-02",
        to_date_str="2026-09-02", as_of_ts=datetime(2026, 9, 8, tzinfo=UTC))
    assert selected_dict["twr_float"] == pytest.approx(-.1)


@pytest.mark.parametrize("flow_str,close_str", [("100", "210"), ("-50", "60")])
def test_eod_flow_not_profit_and_official_account_return_stays_independent(flow_str, close_str):
    result_dict = report_dict([nav_attributes_dict(opening_str="100", closing_str=close_str,
        depositsWithdrawals=flow_str, twr_str="5")], config_dict=twr_config_dict(False))
    assert result_dict["twr_float"] == pytest.approx(.10)
    assert result_dict["strategy_list"][0]["twr_float"] == pytest.approx(.05)
    assert result_dict["pnl_float"] == 10


def test_same_day_internal_transfer_cancels_without_claiming_event_matching():
    result_dict = report_dict([
        nav_attributes_dict(opening_str="100", closing_str="60", internalCashTransfers="-50", twr_str="10"),
        nav_attributes_dict("U_TEST_B", opening_str="100", closing_str="150", internalCashTransfers="50", mtm="0", twr_str="0")],
        config_dict=twr_config_dict())
    assert result_dict["twr_float"] == pytest.approx(.05)
    assert result_dict["capital_movement_float"] == 0
    assert result_dict["pnl_float"] == 10


def test_cross_day_transfer_does_not_shrink_denominator_and_inflate_return():
    result_dict = report_dict([
        nav_attributes_dict(opening_str="100", closing_str="50", internalCashTransfers="-50", mtm="0", twr_str="0"),
        nav_attributes_dict("U_TEST_B", opening_str="100", closing_str="100", mtm="0", twr_str="0"),
        nav_attributes_dict(date_str="2026-09-02", opening_str="50", closing_str="55", mtm="5", twr_str="10"),
        nav_attributes_dict("U_TEST_B", date_str="2026-09-02", opening_str="100", closing_str="150", internalCashTransfers="50", mtm="0", twr_str="0")],
        config_dict=twr_config_dict(), to_str="2026-09-02")
    assert result_dict["twr_float"] is None  # Not the misleading 5/150.
    assert result_dict["return_path_list"] == result_dict["twr_daily_list"] == []
    assert "transit" in result_dict["twr_reason_str"]
    assert result_dict["pnl_float"] == 5
    assert result_dict["status_str"] == "draft"


def test_entrant_capital_in_denominator_without_losing_earlier_history():
    config_dict = twr_config_dict()
    config_dict["accounts"][1]["effective_from"] = "2026-09-02"
    result_dict = report_dict([nav_attributes_dict(),
        nav_attributes_dict(date_str="2026-09-02", opening_str="1010", closing_str="1020"),
        nav_attributes_dict("U_TEST_B", date_str="2026-09-02", opening_str="500", closing_str="505", mtm="5")],
        config_dict=config_dict, to_str="2026-09-02")
    assert result_dict["twr_float"] == pytest.approx(1.01 * (1 + 15 / 1510) - 1)
    assert result_dict["scope_movement_float"] == 500
    assert result_dict["pnl_float"] == 25


def test_exit_preserves_last_owned_profit_but_removes_next_day_capital():
    config_dict = twr_config_dict()
    config_dict["accounts"][1]["effective_to"] = "2026-09-01"
    result_dict = report_dict([nav_attributes_dict(), nav_attributes_dict("U_TEST_B"),
        nav_attributes_dict(date_str="2026-09-02", opening_str="1010", closing_str="1020")],
        config_dict=config_dict, to_str="2026-09-02")
    assert result_dict["twr_float"] == pytest.approx(1.01 * (1 + 10 / 1010) - 1)
    assert result_dict["scope_movement_float"] == -1010
    assert result_dict["pnl_float"] == 30


def test_weekend_exit_without_boundary_nav_remains_fail_closed():
    config_dict = twr_config_dict()
    config_dict["accounts"][1]["effective_to"] = "2026-09-04"
    row_list = [nav_attributes_dict(account_str, date_str=f"2026-09-0{day_int}",
        opening_str=str(1000 + (day_int - 1) * 10), closing_str=str(1000 + day_int * 10))
        for account_str in ("U_TEST_A", "U_TEST_B") for day_int in (1, 2, 3, 4)]
    row_list.append(nav_attributes_dict(date_str="2026-09-08", opening_str="1040", closing_str="1050"))
    result_dict = report_dict(row_list, config_dict=config_dict, to_str="2026-09-08")
    # The existing calendar-boundary contract requires Saturday NAV for the
    # continuing account. This fix must not invent it or conceal the gap.
    assert result_dict["twr_float"] is None and result_dict["pnl_float"] is None
    assert result_dict["return_path_list"] == []
    assert next(row_dict for row_dict in result_dict["daily_book_list"]
        if row_dict["market_date_str"] == "2026-09-05")["nav_float"] is None


@pytest.mark.parametrize("case_str", ["missing_row", "missing_field", "bad_bridge", "zero_base", "linking", "opposite_linking", "continuity"])
def test_invalid_daily_evidence_never_leaves_partial_twr(case_str):
    row_list = [nav_attributes_dict(), nav_attributes_dict("U_TEST_B")]
    to_str = "2026-09-01"
    if case_str == "missing_row":
        row_list.pop()
    elif case_str == "missing_field":
        del row_list[0]["billPay"]
    elif case_str == "bad_bridge":
        row_list[0]["mtm"] = "11"
    elif case_str == "zero_base":
        row_list = [nav_attributes_dict(account_str, opening_str="0", closing_str="10") for account_str in ("U_TEST_A", "U_TEST_B")]
    elif case_str in {"linking", "opposite_linking"}:
        row_list[0].update(endingValue="1110", linkingAdjustments="100")
        if case_str == "opposite_linking":
            row_list[1].update(endingValue="910", linkingAdjustments="-100")
    else:
        to_str = "2026-09-02"
        row_list += [nav_attributes_dict(date_str=to_str, opening_str="1011", closing_str="1021"),
                     nav_attributes_dict("U_TEST_B", date_str=to_str, opening_str="1010", closing_str="1020")]
    result_dict = report_dict(row_list, config_dict=twr_config_dict(), to_str=to_str)
    assert result_dict["twr_float"] is None
    assert result_dict["return_path_list"] == result_dict["twr_daily_list"] == []
    assert result_dict["twr_reason_str"]


def test_dplus1_withholds_configured_return():
    result_dict = build_client_report_dict(twr_config_dict(False), snapshot_obj([nav_attributes_dict()]),
        from_date_str="2026-09-01", to_date_str="2026-09-01", as_of_ts=datetime(2026, 9, 1, 23, tzinfo=UTC))
    assert result_dict["twr_float"] is None
    assert not result_dict["return_path_list"]


@pytest.mark.parametrize("closing_str,pnl_str", [("100", "-100"), ("50", "-150")])
def test_calculated_total_loss_never_uses_different_official_return_as_fallback(closing_str, pnl_str):
    result_dict = report_dict([nav_attributes_dict(opening_str="100", closing_str=closing_str,
        depositsWithdrawals="100", mtm=pnl_str, twr_str="0")], config_dict=twr_config_dict(False))
    assert result_dict["strategy_list"][0]["twr_float"] == 0
    assert result_dict["twr_float"] is None
    assert result_dict["return_path_list"] == []
    assert build_investor_snapshot_dict(result_dict)["document_status_str"] == "draft"


def test_unfunded_gap_does_not_silently_link_two_separate_books():
    config_dict = twr_config_dict()
    config_dict["accounts"][0]["effective_to"] = "2026-09-01"
    config_dict["accounts"][1]["effective_from"] = "2026-09-03"
    result_dict = report_dict([nav_attributes_dict(), nav_attributes_dict("U_TEST_B", date_str="2026-09-03")],
        config_dict=config_dict, to_str="2026-09-03")
    assert result_dict["twr_float"] is None
    assert "2026-09-02" in result_dict["twr_reason_str"]
    assert result_dict["pnl_float"] == 20


def test_opt_in_changes_hash_not_official_accounts_and_legacy_stays_unchanged():
    row_list = [nav_attributes_dict(), nav_attributes_dict("U_TEST_B")]
    legacy_dict = report_dict(row_list, config_dict=client_config_dict(second_bool=True))
    current_dict = report_dict(row_list, config_dict=twr_config_dict())
    assert legacy_dict["twr_float"] is None
    assert current_dict["twr_float"] == pytest.approx(.01)
    assert legacy_dict["strategy_list"] == current_dict["strategy_list"]
    assert legacy_dict["report_hash_str"] != current_dict["report_hash_str"]
    assert report_dict(row_list, config_dict=twr_config_dict())["report_hash_str"] == current_dict["report_hash_str"]


@pytest.mark.parametrize("field_str,value_obj", [("method", "automatic"), ("reviewed_by", ""), ("evidence_ref", "")])
def test_method_configuration_fails_closed(field_str, value_obj):
    config_dict = twr_config_dict(False)
    config_dict["client_twr"][field_str] = value_obj
    with pytest.raises(ClientReportingError):
        report_dict([nav_attributes_dict()], config_dict=config_dict)


def test_configured_method_requires_flow_contract():
    config_dict = twr_config_dict(False)
    del config_dict["nav_bridge"]
    with pytest.raises(ClientReportingError, match="NAV bridge"):
        report_dict([nav_attributes_dict()], config_dict=config_dict)


def test_non_session_activity_requires_all_active_accounts():
    config_dict = twr_config_dict()
    row_list = []
    for account_str in ("U_TEST_A", "U_TEST_B"):
        row_list.extend(nav_attributes_dict(account_str, date_str=f"2026-09-{day_int:02d}",
            opening_str=str(1000 + 10 * (day_int - 1)), closing_str=str(1000 + 10 * day_int)) for day_int in range(1, 5))
    row_list.append(nav_attributes_dict(date_str="2026-09-05", opening_str="1040", closing_str="1039", mtm="-1", twr_str="-.1"))
    result_dict = report_dict(row_list, config_dict=config_dict, to_str="2026-09-05")
    assert result_dict["twr_float"] is None
    row_list.append(nav_attributes_dict("U_TEST_B", date_str="2026-09-05", opening_str="1040", closing_str="1040", mtm="0", twr_str="0"))
    result_dict = report_dict(row_list, config_dict=config_dict, to_str="2026-09-05")
    assert result_dict["twr_float"] == pytest.approx(2079 / 2000 - 1)


def test_pdf_uses_same_client_return_and_keeps_private_contract_out():
    result_dict = report_dict([nav_attributes_dict(), nav_attributes_dict("U_TEST_B")], config_dict=twr_config_dict())
    investor_dict = build_investor_snapshot_dict(result_dict)
    assert investor_dict["twr_float"] == result_dict["twr_float"] == .01
    assert investor_dict["document_status_str"] == "final"
    assert investor_dict["twr_method_id_str"] == "daily_nav_eod_v1"
    text_str = "\n".join(page_obj.extract_text() for page_obj in PdfReader(BytesIO(render_investor_pdf_bytes(investor_dict))).pages)
    assert "Return (TWR)" in text_str and "1.00%" in text_str
    assert "end-of-day cash-flow convention" in text_str
    assert "U_TEST_" not in text_str and "Synthetic fixture" not in text_str


def test_configured_unavailable_return_keeps_dollar_facts_but_report_draft():
    result_dict = report_dict([nav_attributes_dict(closing_str="910", internalCashTransfers="-100")], config_dict=twr_config_dict(False))
    assert result_dict["flows_complete_bool"] and result_dict["pnl_float"] == 10
    assert result_dict["strategy_list"][0]["twr_float"] == .01
    investor_dict = build_investor_snapshot_dict(result_dict)
    assert investor_dict["document_status_str"] == "draft"
    assert investor_dict["twr_float"] is None


def test_demo_pdf_does_not_claim_calculated_return_uses_real_ibkr_data():
    config_dict = twr_config_dict(False)
    config_dict["is_demo"] = True
    investor_dict = build_investor_snapshot_dict(report_dict([nav_attributes_dict()], config_dict=config_dict))
    text_str = "\n".join(page_obj.extract_text() for page_obj in PdfReader(BytesIO(render_investor_pdf_bytes(investor_dict))).pages)
    assert "calculated from synthetic data" in text_str
    assert "calculated from IBKR data" not in text_str


def test_configured_client_headline_is_identical_across_views_and_export():
    from alpha.live.dashboard_v3.app import create_app
    from test_dashboard_operator_access import ForbiddenProvider

    config_dict = twr_config_dict()
    source_obj = snapshot_obj([nav_attributes_dict(),
        nav_attributes_dict("U_TEST_B", opening_str="10000", closing_str="10050", mtm="50", twr_str=".5")])
    app_obj = create_app(ForbiddenProvider(), read_only_bool=True,
        client_registry_dict={"schema_version": 1, "clients": [config_dict]},
        client_reporting_snapshot_fn=lambda client_id_str: source_obj)
    client_obj = app_obj.test_client()
    result_list = []
    for view_str in ("overview", "performance", "report"):
        path_str = f"/clients/sample/{view_str}?from=2026-09-01&to=2026-09-01"
        response_obj = client_obj.get(path_str)
        assert response_obj.status_code == 200
        html_str = response_obj.get_data(as_text=True)
        assert 'Return · TWR</p><strong data-sign="positive">0.55%' in html_str.split('data-client-twr>', 1)[1]
        chart_label_str = "Portfolio cumulative return (%)" if view_str == "overview" else "Calculated daily client TWR including opening zero baseline"
        assert f'aria-label="{chart_label_str}"' in html_str
        result_list.append(client_obj.get(path_str + "&download=json").get_json())
    assert len({result_dict["report_hash_str"] for result_dict in result_list}) == 1
    assert all(result_dict["twr_float"] == pytest.approx(60 / 11000) for result_dict in result_list)
    response_obj = client_obj.get("/clients/sample/report?from=2026-09-01&to=2026-09-01&download=pdf&expected=" + result_list[0]["report_hash_str"])
    assert response_obj.status_code == 200
    text_str = "\n".join(page_obj.extract_text() for page_obj in PdfReader(BytesIO(response_obj.data)).pages)
    assert "0.55%" in text_str
