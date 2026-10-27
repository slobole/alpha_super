"""Issue only a reviewed final report and serve its frozen, private snapshot."""

from html import unescape
import importlib
import json
import re
from types import SimpleNamespace

import pytest

from alpha.live.dashboard_v4.app import create_app
from alpha.live.dashboard_v4.report_archive import ReportArchiveError
from alpha.live.dashboard_v4.report_programs import ReportProgramStore
from test_dashboard_v4_monthly_report import SAMPLE_AS_OF_TS, _sample_tuple


@pytest.fixture
def issue_context_dict(tmp_path):
    workspace_dict, snapshot_obj = _sample_tuple()
    provider_obj = SimpleNamespace(event_log_path_str=str(tmp_path / "events.jsonl"))
    source_dict = {"workspace_dict": workspace_dict, "snapshot_obj": snapshot_obj, "read_count_int": 0, "block_reads_bool": False}

    def workspace_snapshot_tuple():
        assert not source_dict["block_reads_bool"], "Rejected or archived request reached the financial source"
        source_dict["read_count_int"] += 1
        return source_dict["workspace_dict"], source_dict["snapshot_obj"]

    app_obj = create_app(provider_obj, now_fn=lambda: SAMPLE_AS_OF_TS,
        performance_db_path_str=str(tmp_path / "none.sqlite3"), workspace_snapshot_fn=workspace_snapshot_tuple)
    return {"client_obj": app_obj.test_client(), "source_dict": source_dict, "provider_obj": provider_obj, "root_obj": tmp_path}


def _preview_dict(context_dict, month_str="2026-08"):
    response_obj = context_dict["client_obj"].get("/monthly-report?month=" + month_str)
    assert response_obj.status_code == 200
    html_str = response_obj.get_data(as_text=True)
    token_str = re.search(r'data-token="([^"]+)"', html_str).group(1)
    report_id_str = re.search(r'data-report-id="([0-9a-f]{16})"', html_str).group(1)
    return {"html_str": html_str, "header_dict": {"Origin": "http://localhost", "X-Alpha-Action-Token": unescape(token_str)},
        "body_dict": {"confirmed_bool": True, "month_str": month_str, "report_id_str": report_id_str}}


def _issue_obj(context_dict, preview_dict):
    return context_dict["client_obj"].post("/monthly-report/archive", json=preview_dict["body_dict"], headers=preview_dict["header_dict"])


def test_get_preview_and_missing_archive_never_write(issue_context_dict):
    context_dict = issue_context_dict
    before_list = list(context_dict["root_obj"].rglob("*"))
    _preview_dict(context_dict)
    assert list(context_dict["root_obj"].rglob("*")) == before_list
    context_dict["source_dict"]["block_reads_bool"] = True
    assert context_dict["client_obj"].get("/monthly-report/saved/" + "0" * 64).status_code == 404
    assert list(context_dict["root_obj"].rglob("*")) == before_list


def test_issue_and_saved_read_preserve_the_reviewed_report_after_source_and_map_changes(issue_context_dict):
    context_dict = issue_context_dict
    preview_dict = _preview_dict(context_dict)
    response_obj = _issue_obj(context_dict, preview_dict)
    assert response_obj.status_code == 200, response_obj.get_json()
    archive_url_str = response_obj.get_json()["archive_url_str"]
    assert re.fullmatch(r"/monthly-report/saved/[0-9a-f]{64}", archive_url_str)
    saved_obj = context_dict["client_obj"].get(archive_url_str)
    assert saved_obj.status_code == 200 and saved_obj.mimetype == "text/html"
    frozen_html_str = saved_obj.get_data(as_text=True)
    assert preview_dict["body_dict"]["report_id_str"] in frozen_html_str
    archive_path_obj = next((context_dict["root_obj"] / "monthly_reports").glob("*.json"))
    payload_dict = json.loads(archive_path_obj.read_text(encoding="utf-8"))
    assert payload_dict["html_str"] == frozen_html_str
    assert payload_dict["evidence_dict"]["report_id_str"] == preview_dict["body_dict"]["report_id_str"]
    assert payload_dict["evidence_dict"]["month_str"] == "2026-08"
    assert payload_dict["evidence_dict"]["source_checksum_list"]
    assert payload_dict["evidence_dict"]["generated_at_str"] == SAMPLE_AS_OF_TS.isoformat()
    context_dict["source_dict"]["workspace_dict"], context_dict["source_dict"]["snapshot_obj"] = _sample_tuple(
        extra_event_dict={("U100", "2026-08-31"): {"mtm": "500"}})
    ReportProgramStore(context_dict["provider_obj"]).save({"pod_taa": "Steady", "pod_ndx": "Growth"})
    assert _preview_dict(context_dict)["body_dict"]["report_id_str"] != preview_dict["body_dict"]["report_id_str"]
    context_dict["source_dict"]["block_reads_bool"] = True
    (context_dict["root_obj"] / "report_programs.json").write_text("corrupted current mapping", encoding="utf-8")
    assert context_dict["client_obj"].get(archive_url_str).get_data(as_text=True) == frozen_html_str
    assert len(list((context_dict["root_obj"] / "monthly_reports").iterdir())) == 1


def test_saved_html_is_self_contained_without_controls_tokens_or_internal_identifiers(issue_context_dict):
    context_dict = issue_context_dict
    preview_dict = _preview_dict(context_dict)
    response_obj = _issue_obj(context_dict, preview_dict)
    assert response_obj.status_code == 200
    html_str = context_dict["client_obj"].get(response_obj.get_json()["archive_url_str"]).get_data(as_text=True)
    assert "data:font/woff2;base64," in html_str and "@media print" in html_str
    assert "About these numbers" not in html_str
    for forbidden_str in ("<script", "<link", "<nav", "data-token=", "data-programs", "data-print", "/assets/", "/static/",
            "U100", "U200", "U900", "pod_taa", "pod_ndx", str(context_dict["root_obj"]),
            preview_dict["header_dict"]["X-Alpha-Action-Token"]):
        assert forbidden_str not in html_str, forbidden_str


def test_only_saved_report_allows_same_origin_print_frame_and_embedded_font(issue_context_dict):
    context_dict = issue_context_dict
    preview_dict = _preview_dict(context_dict)
    preview_obj = context_dict["client_obj"].get("/monthly-report?month=2026-08")
    assert preview_obj.headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in preview_obj.headers["Content-Security-Policy"]
    response_obj = _issue_obj(context_dict, preview_dict)
    saved_obj = context_dict["client_obj"].get(response_obj.get_json()["archive_url_str"])
    assert saved_obj.headers["X-Frame-Options"] == "SAMEORIGIN"
    assert "frame-ancestors 'self'" in saved_obj.headers["Content-Security-Policy"]
    assert "font-src 'self' data:" in saved_obj.headers["Content-Security-Policy"]
    assert "script-src 'self'" in saved_obj.headers["Content-Security-Policy"]


@pytest.mark.parametrize("header_change_dict", [
    {"X-Alpha-Action-Token": "wrong"}, {"Origin": "http://evil.example"}, {"Origin": "https://localhost"},
    {"Origin": "null"}, {"Referer": "http://evil.example/page"}, {"Origin": None},
])
def test_issue_authentication_rejects_before_financial_read(issue_context_dict, header_change_dict):
    context_dict = issue_context_dict
    preview_dict = _preview_dict(context_dict)
    header_dict = {**preview_dict["header_dict"], **header_change_dict}
    header_dict = {key_str: value_str for key_str, value_str in header_dict.items() if value_str is not None}
    context_dict["source_dict"]["block_reads_bool"] = True
    response_obj = context_dict["client_obj"].post("/monthly-report/archive", json=preview_dict["body_dict"], headers=header_dict)
    assert response_obj.status_code == 403
    assert not (context_dict["root_obj"] / "monthly_reports").exists()


@pytest.mark.parametrize("method_str,expected_int", [("GET", 405), ("HEAD", 405), ("PUT", 403), ("PATCH", 403), ("DELETE", 403)])
def test_only_issue_post_can_reach_financial_data(issue_context_dict, method_str, expected_int):
    context_dict = issue_context_dict
    context_dict["source_dict"]["block_reads_bool"] = True
    response_obj = context_dict["client_obj"].open("/monthly-report/archive", method=method_str)
    assert response_obj.status_code == expected_int
    assert not (context_dict["root_obj"] / "monthly_reports").exists()


@pytest.mark.parametrize("case_str", ["query", "oversized", "extra_field", "missing_field", "bad_month", "wrong_type", "not_json", "unconfirmed", "no_length"])
def test_bad_issue_request_is_rejected_before_financial_read(issue_context_dict, case_str):
    context_dict = issue_context_dict
    preview_dict = _preview_dict(context_dict)
    body_dict = dict(preview_dict["body_dict"])
    path_str = "/monthly-report/archive"
    request_dict = {"json": body_dict, "headers": preview_dict["header_dict"]}
    if case_str == "query":
        path_str += "?month=2026-08"
    elif case_str == "oversized":
        body_dict["report_id_str"] = "a" * 2100
    elif case_str == "extra_field":
        body_dict["extra"] = 1
    elif case_str == "missing_field":
        body_dict.pop("report_id_str")
    elif case_str == "bad_month":
        body_dict["month_str"] = "2026-13"
    elif case_str == "wrong_type":
        body_dict["month_str"] = ["2026-08"]
    elif case_str == "not_json":
        request_dict = {"data": "not json", "headers": preview_dict["header_dict"], "content_type": "text/plain"}
    elif case_str == "unconfirmed":
        body_dict["confirmed_bool"] = False
    elif case_str == "no_length":
        request_dict["environ_overrides"] = {"CONTENT_LENGTH": ""}
    context_dict["source_dict"]["block_reads_bool"] = True
    assert context_dict["client_obj"].post(path_str, **request_dict).status_code == 400
    assert not (context_dict["root_obj"] / "monthly_reports").exists()


@pytest.mark.parametrize("change_str", ["source", "mapping", "identity"])
def test_changed_report_requires_a_new_review_before_issuance(issue_context_dict, change_str):
    context_dict = issue_context_dict
    preview_dict = _preview_dict(context_dict)
    if change_str == "source":
        context_dict["source_dict"]["workspace_dict"], context_dict["source_dict"]["snapshot_obj"] = _sample_tuple(
            extra_event_dict={("U100", "2026-08-31"): {"mtm": "500"}})
    elif change_str == "mapping":
        ReportProgramStore(context_dict["provider_obj"]).save({"pod_taa": "Steady"})
    else:
        preview_dict["body_dict"]["report_id_str"] = "0" * 16
    response_obj = _issue_obj(context_dict, preview_dict)
    assert response_obj.status_code == 409 and "changed" in response_obj.get_json()["message"]
    assert not (context_dict["root_obj"] / "monthly_reports").exists()


def test_draft_report_can_be_viewed_but_not_issued(issue_context_dict):
    context_dict = issue_context_dict
    preview_dict = _preview_dict(context_dict, "2026-09")
    assert "Print draft" in preview_dict["html_str"] and "data-archive-url=" not in preview_dict["html_str"]
    response_obj = _issue_obj(context_dict, preview_dict)
    assert response_obj.status_code == 409 and "draft" in response_obj.get_json()["message"]
    assert not (context_dict["root_obj"] / "monthly_reports").exists()


def test_corrupt_program_mapping_withholds_preview_and_issue_before_financial_reads(issue_context_dict):
    context_dict = issue_context_dict
    preview_dict = _preview_dict(context_dict)
    (context_dict["root_obj"] / "report_programs.json").write_text("broken mapping", encoding="utf-8")
    context_dict["source_dict"]["block_reads_bool"] = True
    response_obj = context_dict["client_obj"].get("/monthly-report?month=2026-08")
    assert response_obj.status_code == 503
    assert "Saved programs could not be verified" in response_obj.get_data(as_text=True)
    assert "data-print" not in response_obj.get_data(as_text=True)
    response_obj = _issue_obj(context_dict, preview_dict)
    assert response_obj.status_code == 503 and "Report not saved" in response_obj.get_json()["message"]
    assert not (context_dict["root_obj"] / "monthly_reports").exists()


def test_archive_failure_returns_no_printable_saved_url(issue_context_dict, monkeypatch):
    context_dict = issue_context_dict
    preview_dict = _preview_dict(context_dict)
    app_module = importlib.import_module("alpha.live.dashboard_v4.app")

    def failed_save_str(*argument_tuple, **keyword_dict):
        raise ReportArchiveError("Storage unavailable")

    monkeypatch.setattr(app_module.ReportArchiveStore, "save", failed_save_str)
    response_obj = _issue_obj(context_dict, preview_dict)
    assert response_obj.status_code == 503 and "Report not saved" in response_obj.get_json()["message"]
    assert "archive_url_str" not in response_obj.get_json()
    assert not (context_dict["root_obj"] / "monthly_reports").exists()


def test_corrupt_saved_archive_returns_no_html_and_does_not_rebuild(issue_context_dict):
    context_dict = issue_context_dict
    response_obj = _issue_obj(context_dict, _preview_dict(context_dict))
    assert response_obj.status_code == 200
    archive_url_str = response_obj.get_json()["archive_url_str"]
    path_obj = next((context_dict["root_obj"] / "monthly_reports").glob("*.json"))
    path_obj.write_text("altered", encoding="utf-8")
    context_dict["source_dict"]["block_reads_bool"] = True
    response_obj = context_dict["client_obj"].get(archive_url_str)
    assert response_obj.status_code == 503
    assert response_obj.get_data(as_text=True) == "The saved report could not be verified."
    assert path_obj.read_text(encoding="utf-8") == "altered"
