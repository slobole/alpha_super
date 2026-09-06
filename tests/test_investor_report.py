from io import BytesIO
import json
from zipfile import ZipFile

import pytest
from pypdf import PdfReader

from alpha.live.investor_report import build_investor_snapshot_dict, render_investor_pdf_bytes
from test_client_reporting import client_config_dict, nav_attributes_dict, report_dict, snapshot_obj
from test_client_views import financial_client_obj
from test_dashboard_operator_access import auth_headers_dict


def test_pdf_contains_same_financial_values_and_no_operator_identifiers():
    result_dict = report_dict([nav_attributes_dict()])
    result_dict["private_key"] = "secret=do-not-export"
    result_dict["strategy_list"][0]["raw_log_path"] = "C:/private/operator.log"
    investor_dict = build_investor_snapshot_dict(result_dict)
    pdf_bytes = render_investor_pdf_bytes(investor_dict)
    reader_obj = PdfReader(BytesIO(pdf_bytes))
    text_str = "\n".join(page_obj.extract_text() for page_obj in reader_obj.pages)
    assert len(reader_obj.pages) == 2
    for expected_str in ("$1,010.00", "$10.00", "1.00%", "Strategy A", "2026-09-01", investor_dict["document_hash_str"]):
        assert expected_str in text_str
    for private_str in ("U_TEST_A", "strategy_a", "TEST_NAV", "do-not-export", "operator.log"):
        assert private_str not in text_str
        assert private_str not in json.dumps(investor_dict)


def test_losing_report_uses_standard_negative_money_and_readable_issued_time(tmp_path):
    result_dict = report_dict([nav_attributes_dict(closing_str="990", mtm="-10", twr="-1")])
    investor_dict = build_investor_snapshot_dict(result_dict)
    pdf_bytes = render_investor_pdf_bytes(investor_dict)
    (tmp_path / "loss-report.pdf").write_bytes(pdf_bytes)
    text_str = "\n".join(page_obj.extract_text() for page_obj in PdfReader(BytesIO(pdf_bytes)).pages)
    assert "-$10.00" in text_str and "$-" not in text_str
    assert "Prepared: 2026-09-08 23:00 UTC" in text_str
    assert investor_dict["renderer_version_str"] == "investor_pdf_v5"
    assert investor_dict["pnl_float"] == -10


def test_multi_account_document_final_without_inventing_client_return():
    result_dict = report_dict([nav_attributes_dict(), nav_attributes_dict("U_TEST_B")], config_dict=client_config_dict(second_bool=True))
    investor_dict = build_investor_snapshot_dict(result_dict)
    assert investor_dict["document_status_str"] == "final"
    assert investor_dict["twr_float"] is None
    assert [item_dict["twr_float"] for item_dict in investor_dict["strategy_list"]] == [0.01, 0.01]
    reader_obj = PdfReader(BytesIO(render_investor_pdf_bytes(investor_dict)))
    assert "DRAFT" not in reader_obj.pages[0].extract_text()
    assert "Not reported" in reader_obj.pages[0].extract_text()
    assert "a combined return is not available" in reader_obj.pages[0].extract_text()


def test_report_finality_requires_complete_account_evidence():
    from copy import deepcopy

    complete_dict = report_dict([nav_attributes_dict()])
    for field_str in ("coverage_complete_bool", "flows_complete_bool"):
        for scope_str in ("report", "strategy"):
            incomplete_dict = deepcopy(complete_dict)
            target_dict = incomplete_dict if scope_str == "report" else incomplete_dict["strategy_list"][0]
            target_dict[field_str] = False
            assert build_investor_snapshot_dict(incomplete_dict)["document_status_str"] == "draft"
    complete_dict["strategy_list"][0]["twr_float"] = None
    assert build_investor_snapshot_dict(complete_dict)["document_status_str"] == "draft"
    complete_dict["strategy_list"] = []
    assert build_investor_snapshot_dict(complete_dict)["document_status_str"] == "draft"


def test_zero_return_is_valid_and_demonstration_always_wins():
    result_dict = report_dict([nav_attributes_dict(closing_str="1000", mtm="0", twr="0")])
    assert build_investor_snapshot_dict(result_dict)["document_status_str"] == "final"
    result_dict["is_demo_bool"] = True
    assert build_investor_snapshot_dict(result_dict)["document_status_str"] == "demonstration"


def test_final_multi_account_report_keeps_offsetting_transfers_out_of_profit():
    result_dict = report_dict([
        nav_attributes_dict(closing_str="910", internalCashTransfers="-100"),
        nav_attributes_dict("U_TEST_B", closing_str="1110", internalCashTransfers="100"),
    ], config_dict=client_config_dict(second_bool=True))
    investor_dict = build_investor_snapshot_dict(result_dict)
    assert investor_dict["document_status_str"] == "final"
    assert investor_dict["capital_movement_float"] == 0
    assert investor_dict["pnl_float"] == 20
    assert investor_dict["twr_float"] is None


def test_final_report_preserves_retirement_accounting_and_dates():
    config_dict = client_config_dict(second_bool=True)
    config_dict["accounts"][1]["effective_to"] = "2026-09-01"
    result_dict = report_dict([
        nav_attributes_dict(), nav_attributes_dict(date_str="2026-09-02", opening_str="1010", closing_str="1020"),
        nav_attributes_dict("U_TEST_B", "2026-09-01", "500", "505", "1", mtm="5"),
    ], config_dict=config_dict, to_str="2026-09-02")
    investor_dict = build_investor_snapshot_dict(result_dict)
    assert investor_dict["document_status_str"] == "final"
    assert investor_dict["scope_movement_float"] == -505
    assert investor_dict["pnl_float"] == 25
    assert investor_dict["strategy_list"][1]["to_date_str"] == "2026-09-01"


def test_missing_flow_or_unfinalized_source_cannot_issue_final_report():
    from alpha.live.client_reporting import build_client_report_dict
    from test_client_reporting import AS_OF_TS

    incomplete_dict = nav_attributes_dict()
    incomplete_dict.pop("billPay")
    assert build_investor_snapshot_dict(report_dict([incomplete_dict]))["document_status_str"] == "draft"
    current_dict = build_client_report_dict(client_config_dict(), snapshot_obj([nav_attributes_dict()]),
        from_date_str="2026-09-01", to_date_str="2026-09-01", as_of_ts=AS_OF_TS.replace(day=1, hour=14))
    assert build_investor_snapshot_dict(current_dict)["document_status_str"] == "draft"


def test_export_rejects_stale_or_missing_preview_confirmation(financial_client_obj):
    for expected_str in ("", "&expected=old-hash"):
        response_obj = financial_client_obj.get("/clients/sample/report?from=2026-09-01&to=2026-09-01&download=pdf" + expected_str, headers=auth_headers_dict())
        assert response_obj.status_code == 409


def test_pdf_bundle_freezes_public_snapshot_and_matching_pdf(financial_client_obj):
    prefix_str = "/clients/sample/report?from=2026-09-01&to=2026-09-01"
    result_dict = financial_client_obj.get(prefix_str + "&download=json", headers=auth_headers_dict()).get_json()
    response_obj = financial_client_obj.get(prefix_str + "&download=bundle&expected=" + result_dict["report_hash_str"], headers=auth_headers_dict())
    assert response_obj.status_code == 200
    assert response_obj.mimetype == "application/zip"
    with ZipFile(BytesIO(response_obj.data)) as zip_obj:
        assert len(zip_obj.namelist()) == 2
        json_name_str = next(name_str for name_str in zip_obj.namelist() if name_str.endswith(".json"))
        pdf_name_str = next(name_str for name_str in zip_obj.namelist() if name_str.endswith(".pdf"))
        investor_dict = json.loads(zip_obj.read(json_name_str))
        text_str = "\n".join(page_obj.extract_text() for page_obj in PdfReader(BytesIO(zip_obj.read(pdf_name_str))).pages)
    assert investor_dict["pnl_float"] == result_dict["pnl_float"]
    assert investor_dict["document_hash_str"] in text_str
    assert "U_TEST_A" not in json.dumps(investor_dict)


def test_pdf_html_markup_in_client_names_is_literal_text():
    config_dict = client_config_dict()
    config_dict["display_name"] = "Example <b>not markup</b> & partner"
    investor_dict = build_investor_snapshot_dict(report_dict([nav_attributes_dict()], config_dict=config_dict))
    text_str = PdfReader(BytesIO(render_investor_pdf_bytes(investor_dict))).pages[0].extract_text()
    assert "<b>not markup</b>" in text_str


def test_actual_source_revision_after_preview_prevents_export(financial_client_obj, monkeypatch):
    prefix_str = "/clients/sample/report?from=2026-09-01&to=2026-09-01"
    prior_dict = financial_client_obj.get(prefix_str + "&download=json", headers=auth_headers_dict()).get_json()
    financial_client_obj.application.config["client_reporting_snapshot_fn"] = lambda client_id_str: snapshot_obj([nav_attributes_dict(closing_str="1011", mtm="11")], import_id_int=2)
    monkeypatch.setattr("alpha.live.dashboard_v3.client_views.render_investor_pdf_bytes", lambda report_obj: (_ for _ in ()).throw(AssertionError("Stale preview reached PDF renderer")))
    response_obj = financial_client_obj.get(prefix_str + "&download=pdf&expected=" + prior_dict["report_hash_str"], headers=auth_headers_dict())
    assert response_obj.status_code == 409


def test_issued_document_identity_includes_issue_time_but_accounting_hash_is_stable():
    result_dict = report_dict([nav_attributes_dict()])
    first_dict = build_investor_snapshot_dict(result_dict)
    result_dict["generated_at_str"] = "2026-09-09T23:01:00+00:00"
    second_dict = build_investor_snapshot_dict(result_dict)
    assert first_dict["report_hash_str"] == second_dict["report_hash_str"]
    assert first_dict["document_hash_str"] != second_dict["document_hash_str"]


def test_pdf_market_comparison_matches_shared_result_and_excludes_nested_private_fields():
    from datetime import UTC, datetime
    from alpha.live.client_reporting import build_client_report_dict
    from alpha.live.dashboard_v3.demo import build_demo_benchmark_snapshot, build_demo_fixture_tuple

    registry_dict, source_dict = build_demo_fixture_tuple()
    result_dict = build_client_report_dict(registry_dict["clients"][1], source_dict["demo-client"],
        from_date_str="2026-06-01", to_date_str="2026-09-04", as_of_ts=datetime(2026, 9, 5, 12, tzinfo=UTC),
        benchmark_snapshot_obj=build_demo_benchmark_snapshot())
    result_dict["strategy_list"][0]["benchmark_dict"]["private_source_path"] = "C:/private/operator.log"
    investor_dict = build_investor_snapshot_dict(result_dict)
    text_str = "\n".join(page_obj.extract_text() for page_obj in PdfReader(BytesIO(render_investor_pdf_bytes(investor_dict))).pages)
    assert "How each strategy compared with the market" in text_str
    assert "SPY total return (synthetic)" in text_str
    assert "Benchmark closes: 2026-05-29 to 2026-09-04" in text_str
    assert "matching account dates and the market closing dates shown" in " ".join(text_str.split())
    assert "percentage points" in text_str
    for strategy_dict in result_dict["strategy_list"]:
        assert f"{strategy_dict['benchmark_dict']['difference_pp_float']:+.2f}" in text_str
        assert strategy_dict["benchmark_dict"]["price_hash_str"] in json.dumps(investor_dict)
    assert "operator.log" not in text_str
    assert "operator.log" not in json.dumps(investor_dict)


@pytest.mark.parametrize("is_demo_bool", [False, True])
def test_friendly_pdf_source_copy_preserves_frozen_evidence(is_demo_bool):
    from copy import deepcopy
    from alpha.live.client_reporting import content_hash_str

    result_dict = report_dict([nav_attributes_dict()])
    result_dict["is_demo_bool"] = is_demo_bool
    investor_dict = build_investor_snapshot_dict(result_dict)
    original_dict = deepcopy(investor_dict)
    text_str = "\n".join(page_obj.extract_text() for page_obj in PdfReader(BytesIO(render_investor_pdf_bytes(investor_dict))).pages)
    source_str = "Account data is taken from IBKR statements."
    demo_str = "Example data only - not actual IBKR results."
    assert (source_str in text_str) is (not is_demo_bool)
    assert (demo_str in text_str) is is_demo_bool
    assert ("DEMONSTRATION ONLY" in text_str) is is_demo_bool
    assert "Account return" in text_str
    assert "Profit / loss" in text_str
    for removed_str in ("Method and important qualifications", "geometrically linked",
                        "FINAL identifies", "Fee basis:", "Source SHA-256", "Source and version"):
        assert removed_str not in text_str
    assert investor_dict == original_dict
    for field_str in ("fee_basis_str", "twr_method_str", "limitations_list", "report_hash_str", "scope_hash_str"):
        assert investor_dict[field_str] == result_dict[field_str]
    assert investor_dict["source_checksum_list"] == sorted({item_dict["checksum_str"] for item_dict in result_dict["source_list"]})
    assert investor_dict["renderer_version_str"] == "investor_pdf_v5"
    prior_renderer_dict = {key_str: value_obj for key_str, value_obj in investor_dict.items() if key_str != "document_hash_str"}
    prior_renderer_dict["renderer_version_str"] = "investor_pdf_v3"
    assert content_hash_str(prior_renderer_dict) != investor_dict["document_hash_str"]
    assert investor_dict["document_hash_str"] in text_str


def test_friendly_pdf_many_strategies_repeat_headers_and_keep_all_values():
    from copy import deepcopy

    investor_dict = build_investor_snapshot_dict(report_dict([nav_attributes_dict()]))
    strategy_dict = investor_dict["strategy_list"][0]
    investor_dict["strategy_list"] = []
    for strategy_int in range(24):
        item_dict = deepcopy(strategy_dict)
        item_dict["display_name_str"] = f"Strategy {strategy_int + 1:02d} - long investment strategy name for layout verification"
        investor_dict["strategy_list"].append(item_dict)
    reader_obj = PdfReader(BytesIO(render_investor_pdf_bytes(investor_dict)))
    page_text_list = [page_obj.extract_text() for page_obj in reader_obj.pages]
    text_str = "\n".join(page_text_list)
    for strategy_int in range(24):
        assert f"Strategy {strategy_int + 1:02d}" in text_str
    for page_text_str in page_text_list[1:]:
        if "layout verification" in page_text_str:
            assert "Account return" in page_text_str
    assert text_str.count("$1,010.00") == 26
    assert investor_dict["document_hash_str"] in text_str
