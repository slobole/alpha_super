"""Exports preserve canonical facts without leaking operator source fields."""

from copy import deepcopy
import csv
from io import BytesIO, StringIO

import pytest
from pypdf import PdfReader

from alpha.live.dashboard_v4.performance_exports import export_performance_csv_str, export_performance_pdf_bytes
from test_client_reporting import client_config_dict, nav_attributes_dict, report_dict


def _rows_list(report_obj, level_str="portfolio"):
    return list(csv.DictReader(StringIO(export_performance_csv_str(report_obj, level_str=level_str))))


def test_portfolio_export_preserves_verified_bridge_and_percent_units():
    source_dict = report_dict([nav_attributes_dict(closing_str="1110", depositsWithdrawals="100")])
    before_dict = deepcopy(source_dict)
    row_dict = _rows_list(source_dict)[0]
    assert row_dict["Name"] == "Portfolio"
    assert float(row_dict["Start value"]) == 1000
    assert float(row_dict["End value"]) == 1110
    assert float(row_dict["Capital movements"]) == 100
    assert float(row_dict["Profit / loss"]) == 10
    assert float(row_dict["Return TWR (%)"]) == 1
    assert row_dict["Selected from"] == row_dict["Measured from"] == "2026-09-01"
    assert row_dict["Source"] == "Saved IBKR account facts"
    assert row_dict["Report hash"] == source_dict["report_hash_str"]
    assert source_dict == before_dict


def test_pod_export_retains_retired_membership_dates_and_independent_returns():
    config_dict = client_config_dict(second_bool=True)
    config_dict["accounts"][1]["effective_to"] = "2026-09-01"
    source_dict = report_dict([
        nav_attributes_dict(), nav_attributes_dict(date_str="2026-09-02", opening_str="1010", closing_str="1020"),
        nav_attributes_dict("U_TEST_B", "2026-09-01", "500", "505", "1", mtm="5"),
    ], config_dict=config_dict, to_str="2026-09-02")
    row_list = _rows_list(source_dict, "pods")
    assert len(row_list) == 2
    assert [row_dict["Measured to"] for row_dict in row_list] == ["2026-09-02", "2026-09-01"]
    assert all(row_dict["Selected to"] == "2026-09-02" for row_dict in row_list)
    assert [float(row_dict["Profit / loss"]) for row_dict in row_list] == [20, 5]
    assert float(row_list[1]["Return TWR (%)"]) == 1
    assert float(row_list[1]["Max drawdown (%)"]) == 0


def test_missing_return_and_pnl_are_blank_while_actual_zero_stays_zero():
    source_dict = report_dict([nav_attributes_dict(closing_str="1000", mtm="0", twr="0")])
    assert float(_rows_list(source_dict)[0]["Profit / loss"]) == 0
    source_dict["pnl_float"] = source_dict["twr_float"] = None
    source_dict["flows_complete_bool"] = False
    row_dict = _rows_list(source_dict)[0]
    assert row_dict["Profit / loss"] == row_dict["Return TWR (%)"] == ""
    assert row_dict["Flow coverage complete"] == "No"
    assert float(row_dict["End value"]) == 1000


@pytest.mark.parametrize("name_str", ["=SUM(A1:A9)", "+cmd", "-cmd", "@SUM(1)", "  =1+1", "\ufeff=1+1", "\tHello", "\rHello", "\nHello"])
def test_user_text_is_spreadsheet_formula_safe(name_str):
    source_dict = report_dict([nav_attributes_dict()])
    source_dict["strategy_list"][0]["display_name_str"] = name_str
    assert _rows_list(source_dict, "pods")[0]["Name"] == "'" + name_str


def test_csv_quotes_commas_and_newlines_without_changing_negative_numbers():
    source_dict = report_dict([nav_attributes_dict(closing_str="990", mtm="-10", twr="-1")])
    source_dict["strategy_list"][0]["display_name_str"] = 'A, "quoted"\nPod'
    row_dict = _rows_list(source_dict, "pods")[0]
    assert row_dict["Name"] == 'A, "quoted"\nPod'
    assert row_dict["Profit / loss"] == "-10.0"
    assert float(row_dict["Return TWR (%)"]) == -1
    assert float(row_dict["Max drawdown (%)"]) == -1


@pytest.mark.parametrize("level_str", ["portfolio", "pods"])
def test_csv_allowlist_excludes_account_ids_paths_and_arbitrary_fields(level_str):
    source_dict = report_dict([nav_attributes_dict()])
    source_dict["private_token"] = "do-not-export"
    source_dict["strategy_list"][0]["raw_log_path"] = "C:/private/operator.log"
    csv_str = export_performance_csv_str(source_dict, level_str=level_str)
    for private_str in ("U_TEST_A", "strategy_a", "TEST_NAV", "do-not-export", "operator.log"):
        assert private_str not in csv_str


@pytest.mark.parametrize("number_obj", [True, "10", float("nan"), float("inf"), -float("inf")])
def test_invalid_numbers_are_not_exported_as_verified_facts(number_obj):
    source_dict = report_dict([nav_attributes_dict()])
    source_dict["pnl_float"] = number_obj
    with pytest.raises(ValueError):
        export_performance_csv_str(source_dict)


def test_invalid_level_rejected_and_empty_pods_export_has_only_header():
    source_dict = report_dict([nav_attributes_dict()])
    with pytest.raises(ValueError):
        export_performance_csv_str(source_dict, level_str="paper")
    source_dict["strategy_list"] = []
    assert _rows_list(source_dict, "pods") == []


@pytest.mark.parametrize("demo_bool,draft_bool,label_str", [(False, False, "FINAL"), (False, True, "DRAFT"), (True, False, "DEMONSTRATION")])
def test_pdf_reuses_public_renderer_and_existing_finality(demo_bool, draft_bool, label_str):
    source_dict = report_dict([nav_attributes_dict()])
    source_dict["is_demo_bool"] = demo_bool
    if draft_bool:
        source_dict["flows_complete_bool"] = False
        source_dict["pnl_float"] = None
    before_dict = deepcopy(source_dict)
    source_dict["private_token"] = "do-not-export"
    pdf_bytes = export_performance_pdf_bytes(source_dict)
    assert pdf_bytes.startswith(b"%PDF-")
    text_str = "\n".join(page_obj.extract_text() for page_obj in PdfReader(BytesIO(pdf_bytes)).pages)
    assert label_str in text_str
    for private_str in ("U_TEST_A", "strategy_a", "TEST_NAV", "do-not-export"):
        assert private_str not in text_str
    source_dict.pop("private_token")
    assert source_dict == before_dict
