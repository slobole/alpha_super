"""Performance routes stay read-only and bind exports to the displayed report."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from html import unescape
from io import BytesIO, StringIO
import csv
import re
from urllib.parse import parse_qs, urlencode, urlsplit

from flask import template_rendered
from pypdf import PdfReader
import pytest

from alpha.live.dashboard_v4.app import create_app
from alpha.live.dashboard_v4.demo import DEMO_NOW_TS, build_demo_workspace_tuple
from alpha.live.investor_report import build_investor_snapshot_dict


@pytest.fixture
def performance_fixture_tuple():
    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple()
    clock_dict = {"now_ts": DEMO_NOW_TS}
    source_dict = {"snapshot_obj": snapshot_obj}
    app_obj = create_app(provider_obj, demo_bool=True, now_fn=lambda: clock_dict["now_ts"],
        workspace_snapshot_fn=lambda: (workspace_dict, source_dict["snapshot_obj"]))
    app_obj.config["TESTING"] = True
    try:
        yield workspace_dict, source_dict, provider_obj, app_obj, clock_dict
    finally:
        provider_obj.close()


def _render_tuple(app_obj, path_str, *, headers_dict=None):
    context_list = []

    def capture_context(sender_obj, **signal_dict):
        context_list.append(signal_dict["context"])

    with template_rendered.connected_to(capture_context, app_obj):
        response_obj = app_obj.test_client().get(path_str, headers=headers_dict)
    assert response_obj.status_code == 200
    return response_obj, context_list[-1]


def _visible_text_str(html_str):
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", html_str))).strip()


def _query_dict(url_str):
    return parse_qs(urlsplit(unescape(url_str)).query)


@pytest.mark.parametrize("method_str", ["POST", "PUT", "PATCH", "DELETE"])
@pytest.mark.parametrize("path_str", ["/performance", "/performance/refresh"])
def test_writes_are_rejected_before_any_financial_source_access(method_str, path_str):
    app_obj = create_app(object(), now_fn=lambda: DEMO_NOW_TS,
        workspace_snapshot_fn=lambda: pytest.fail("Write request reached financial source"))
    response_obj = app_obj.test_client().open(path_str, method=method_str, json={"action": "submit"})
    assert response_obj.status_code == 403
    assert response_obj.get_json()["error"] == "read_only"


@pytest.mark.parametrize("query_str", [
    "mode=paper", "mode=incubation", "pod=demo_1_0", "action=submit", "q=SPY",
    "level=", "level=account", "level=portfolio&level=pods",
    "period=", "period=3M", "period=Custom", "period=All&period=All",
    "unit=", "unit=nav", "unit=pct&unit=usd",
    "from=2026-09-01", "to=2026-09-04", "from=&to=",
    "from=20260901&to=2026-09-04", "from=2026-W36-1&to=2026-09-04",
    "from=2026-9-1&to=2026-09-04", "from=2026-02-29&to=2026-09-04",
    "from=2026-09-01T00:00:00&to=2026-09-04",
    "from=2026-09-05&to=2026-09-04", "from=2026-09-01&to=2026-09-09",
    "from=2026-09-01&from=2026-09-01&to=2026-09-04",
    "from=2026-09-01&to=2026-09-04&to=2026-09-04",
    "unknown=", "expected=" + "a" * 64,
])
@pytest.mark.parametrize("path_str", ["/performance", "/performance/refresh"])
def test_invalid_queries_fail_before_source_access(query_str, path_str):
    app_obj = create_app(object(), now_fn=lambda: DEMO_NOW_TS,
        workspace_snapshot_fn=lambda: pytest.fail("Invalid query reached financial source"))
    assert app_obj.test_client().get(path_str + "?" + query_str).status_code == 400


@pytest.mark.parametrize("query_str", [
    "download=csv", "download=pdf", "download=xlsx&expected=" + "a" * 64,
    "download=&expected=" + "a" * 64, "download=csv&expected=",
    "download=csv&expected=not-a-report-hash", "download=csv&expected=" + "a" * 63,
    "download=csv&expected=" + "g" * 64,
    "download=csv&download=pdf&expected=" + "a" * 64,
    "download=csv&expected=" + "a" * 64 + "&expected=" + "a" * 64,
])
def test_invalid_download_requests_fail_before_source_access(query_str):
    app_obj = create_app(object(), now_fn=lambda: DEMO_NOW_TS,
        workspace_snapshot_fn=lambda: pytest.fail("Invalid download reached financial source"))
    assert app_obj.test_client().get("/performance?" + query_str).status_code == 400


@pytest.mark.parametrize("query_str", [
    "download=csv&expected=" + "a" * 64, "download=pdf&expected=" + "a" * 64,
    "download=csv", "expected=" + "a" * 64,
])
def test_refresh_cannot_be_used_for_downloads(query_str):
    app_obj = create_app(object(), now_fn=lambda: DEMO_NOW_TS,
        workspace_snapshot_fn=lambda: pytest.fail("Refresh download reached source"))
    assert app_obj.test_client().get("/performance/refresh?" + query_str).status_code == 400


def test_custom_date_limit_uses_et_day_before_source_access():
    # It is already September 9 in UTC, but still September 8 in New York.
    clock_ts = datetime(2026, 9, 9, 0, 30, tzinfo=UTC)
    app_obj = create_app(object(), now_fn=lambda: clock_ts,
        workspace_snapshot_fn=lambda: pytest.fail("Future ET date reached source"))
    assert app_obj.test_client().get("/performance?from=2026-09-09&to=2026-09-09").status_code == 400


def test_default_page_has_live_shell_active_navigation_and_financial_provenance(performance_fixture_tuple):
    workspace_dict, _, _, app_obj, _ = performance_fixture_tuple
    response_obj, context_dict = _render_tuple(app_obj, "/performance")
    page_dict = context_dict["performance_page_dict"]
    report_dict = page_dict["report_dict"]
    html_str = response_obj.get_data(as_text=True)
    assert page_dict["level_str"] == "portfolio" and page_dict["period_str"] == "All" and page_dict["unit_str"] == "pct"
    assert page_dict["from_date_str"] == workspace_dict["client_dict"]["mandate_start_date"]
    assert page_dict["to_date_str"] == "2026-09-04"
    assert report_dict["requested_from_date_str"] == page_dict["from_date_str"]
    assert report_dict["requested_to_date_str"] == page_dict["to_date_str"]
    assert re.fullmatch(r"[0-9a-f]{64}", report_dict["report_hash_str"])
    assert page_dict["error_str"] == "" and page_dict["is_demo_bool"] is True
    assert page_dict["source_str"].startswith("Demo")
    assert page_dict["source_str"] in _visible_text_str(html_str)
    assert "<html" in html_str and "ALPHA / OPS V4 · Performance" in html_str
    assert 'class="page calm performance-page"' in html_str
    assert html_str.count('hx-get="') == 1
    assert context_dict["overview_dict"]["refresh_url_str"] == "/performance/status"
    assert 'hx-swap="none"' in html_str
    assert _query_dict(page_dict["refresh_report_url_str"]) == {
        "level": ["portfolio"], "period": ["All"], "unit": ["pct"]}
    nav_str = re.search(r'<nav\b[^>]*aria-label="Main"[^>]*>(.*?)</nav>', html_str, re.S)[1]
    performance_anchor_str = re.search(r'<a\b[^>]*href="/performance"[^>]*>(.*?)</a>', nav_str, re.S)[0]
    assert 'aria-current="page"' in performance_anchor_str and "Performance" in performance_anchor_str
    assert "Performance is not available yet" not in html_str
    assert 'title="PAPER is not available in V4 yet"' in html_str
    assert 'title="INCUBATION is not available in V4 yet"' in html_str
    assert response_obj.headers["Cache-Control"] == "no-store"
    assert response_obj.headers["X-Content-Type-Options"] == "nosniff"
    assert response_obj.headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in response_obj.headers["Content-Security-Policy"]
    assert "script-src 'self'" in response_obj.headers["Content-Security-Policy"]


@pytest.mark.parametrize("path_str,headers_dict", [
    ("/performance", {"HX-Request": "true"}), ("/performance/refresh", None),
])
def test_custom_dates_override_preset_and_survive_tabs_units_and_refresh(performance_fixture_tuple, path_str, headers_dict):
    _, _, _, app_obj, _ = performance_fixture_tuple
    selection_dict = {"level": "pods", "period": "MTD", "unit": "usd",
        "from": "2026-08-20", "to": "2026-09-04"}
    response_obj, context_dict = _render_tuple(app_obj, path_str + "?" + urlencode(selection_dict), headers_dict=headers_dict)
    page_dict = context_dict["performance_page_dict"]
    html_str = response_obj.get_data(as_text=True)
    assert "<html" not in html_str
    assert page_dict["level_str"] == "pods" and page_dict["unit_str"] == "usd"
    assert page_dict["from_date_str"] == "2026-08-20" and page_dict["to_date_str"] == "2026-09-04"
    assert page_dict["report_dict"]["requested_from_date_str"] == "2026-08-20"
    assert page_dict["report_dict"]["requested_to_date_str"] == "2026-09-04"
    assert not any(option_dict["selected_bool"] for option_dict in page_dict["period_option_list"])
    assert context_dict["overview_dict"]["refresh_url_str"] == "/performance/status"
    assert _query_dict(page_dict["refresh_report_url_str"]) == {
        key_str: [value_str] for key_str, value_str in selection_dict.items()}
    assert 'data-selection-scope="performance:pods:MTD:2026-08-20:2026-09-04:usd"' in html_str
    for option_dict in page_dict["level_option_list"] + page_dict["unit_option_list"]:
        option_query_dict = _query_dict(option_dict["url_str"])
        assert option_query_dict["from"] == ["2026-08-20"] and option_query_dict["to"] == ["2026-09-04"]
    for option_dict in page_dict["period_option_list"]:
        option_query_dict = _query_dict(option_dict["url_str"])
        assert "from" not in option_query_dict and "to" not in option_query_dict
        assert option_query_dict["period"] == [option_dict["label_str"]]
        assert option_query_dict["level"] == ["pods"] and option_query_dict["unit"] == ["usd"]


@pytest.mark.parametrize("period_str,from_date_str", [
    ("1W", "2026-09-01"), ("MTD", "2026-09-01"), ("YTD", "2026-06-01"), ("All", "2026-06-01"),
])
def test_each_preset_has_explicit_measured_range(performance_fixture_tuple, period_str, from_date_str):
    _, _, _, app_obj, _ = performance_fixture_tuple
    _, context_dict = _render_tuple(app_obj, "/performance?level=pods&unit=usd&period=" + period_str)
    page_dict = context_dict["performance_page_dict"]
    assert page_dict["period_str"] == period_str and page_dict["from_date_str"] == from_date_str
    assert page_dict["to_date_str"] == "2026-09-04"
    assert [option_dict["label_str"] for option_dict in page_dict["period_option_list"] if option_dict["selected_bool"]] == [period_str]
    assert page_dict["report_dict"]["requested_from_date_str"] == from_date_str
    assert page_dict["pod_row_list"]
    assert all(row_dict["url_str"].startswith("/pods/") for row_dict in page_dict["pod_row_list"])


def test_unit_and_level_changes_preserve_same_accounting_report(performance_fixture_tuple):
    _, _, _, app_obj, _ = performance_fixture_tuple
    _, initial_context_dict = _render_tuple(app_obj, "/performance?level=portfolio&unit=pct")
    initial_dict = initial_context_dict["performance_page_dict"]
    _, selected_context_dict = _render_tuple(app_obj, "/performance?level=pods&unit=usd")
    selected_dict = selected_context_dict["performance_page_dict"]
    assert selected_dict["report_dict"] == initial_dict["report_dict"]
    assert selected_dict["tile_list"] == initial_dict["tile_list"]
    assert initial_dict["chart_title_str"] == "Portfolio return"
    assert selected_dict["chart_title_str"] == "Account value"
    assert selected_dict["chart_basis_str"] == "Includes capital movements"


def test_retired_owned_pod_remains_in_history_and_export_without_live_link(performance_fixture_tuple):
    workspace_dict, _, _, app_obj, _ = performance_fixture_tuple
    retired_dict = workspace_dict["client_dict"]["accounts"][-1]
    retired_dict["effective_to"] = "2026-09-02"
    retired_pod_str = retired_dict["pod_id"]
    workspace_dict["operations_account_list"] = [
        account_dict for account_dict in workspace_dict["operations_account_list"]
        if account_dict["pod_id"] != retired_pod_str]
    response_obj, context_dict = _render_tuple(app_obj,
        "/performance?level=pods&from=2026-09-01&to=2026-09-04")
    page_dict = context_dict["performance_page_dict"]
    retired_row_dict = next(row_dict for row_dict in page_dict["pod_row_list"] if row_dict["pod_id_str"] == retired_pod_str)
    assert retired_row_dict["from_date_str"] == "2026-09-01"
    assert retired_row_dict["to_date_str"] == "2026-09-02"
    assert retired_row_dict["pnl_str"] != "—" and retired_row_dict["return_str"] != "—"
    assert retired_row_dict["url_str"] == ""
    assert retired_dict["display_name"] in _visible_text_str(response_obj.get_data(as_text=True))
    assert f'href="/pods/{retired_pod_str}"' not in response_obj.get_data(as_text=True)
    assert len(page_dict["pod_row_list"]) == 4
    response_obj = app_obj.test_client().get(page_dict["csv_url_str"])
    assert response_obj.status_code == 200
    csv_row_list = list(csv.DictReader(StringIO(response_obj.get_data(as_text=True))))
    retired_csv_dict = next(row_dict for row_dict in csv_row_list if row_dict["Name"] == retired_dict["display_name"])
    assert retired_csv_dict["Selected to"] == "2026-09-04"
    assert retired_csv_dict["Measured to"] == "2026-09-02"
    assert retired_csv_dict["Profit / loss"] != "" and retired_csv_dict["Return TWR (%)"] != ""


def test_incomplete_report_retains_nav_but_withholds_pnl_and_return_in_page_and_csv(performance_fixture_tuple):
    workspace_dict, _, _, app_obj, _ = performance_fixture_tuple
    workspace_dict["financial_scope_complete_bool"] = False
    response_obj, context_dict = _render_tuple(app_obj,
        "/performance?unit=usd&from=2026-09-01&to=2026-09-04")
    page_dict = context_dict["performance_page_dict"]
    report_dict = page_dict["report_dict"]
    assert report_dict["closing_nav_float"] is not None
    assert report_dict["pnl_float"] is report_dict["twr_float"] is None
    assert page_dict["tile_list"][1]["value_str"] != "—"
    assert page_dict["tile_list"][2]["value_str"] == page_dict["tile_list"][3]["value_str"] == "—"
    assert page_dict["contribution_dict"]["available_bool"] is False
    assert "Return unavailable" in _visible_text_str(response_obj.get_data(as_text=True))
    response_obj = app_obj.test_client().get(page_dict["csv_url_str"])
    assert response_obj.status_code == 200
    csv_dict = list(csv.DictReader(StringIO(response_obj.get_data(as_text=True))))[0]
    assert float(csv_dict["End value"]) == report_dict["closing_nav_float"]
    assert csv_dict["Profit / loss"] == csv_dict["Return TWR (%)"] == ""
    assert csv_dict["Coverage complete"] == csv_dict["Flow coverage complete"] == "No"


@pytest.mark.parametrize("source_str", ["workspace_error", "snapshot_error", "missing_rows"])
def test_financial_failure_renders_unavailable_without_private_error_or_exports(performance_fixture_tuple, source_str):
    workspace_dict, source_dict, _, app_obj, _ = performance_fixture_tuple
    private_str = "private-account-credential C:/private/performance.sqlite3"
    if source_str == "workspace_error":
        workspace_dict["financial_error_str"] = private_str
    elif source_str == "snapshot_error":
        source_dict["snapshot_obj"] = replace(source_dict["snapshot_obj"], unavailable_reason_str=private_str)
    else:
        source_dict["snapshot_obj"] = replace(source_dict["snapshot_obj"], row_tuple=())
    response_obj, context_dict = _render_tuple(app_obj, "/performance")
    page_dict = context_dict["performance_page_dict"]
    html_str = response_obj.get_data(as_text=True)
    assert page_dict["report_dict"] is None
    assert page_dict["error_str"]
    assert "unavailable" in _visible_text_str(html_str).lower()
    assert private_str not in html_str
    assert page_dict["csv_url_str"] == page_dict["pdf_url_str"] == ""
    assert "download=csv" not in html_str and "download=pdf" not in html_str
    assert all(tile_dict["value_str"] == "—" for tile_dict in page_dict["tile_list"])


@pytest.mark.parametrize("path_str", ["/performance", "/performance/refresh"])
def test_slow_financial_read_expires_operational_header(performance_fixture_tuple, monkeypatch, path_str):
    _, _, _, app_obj, clock_dict = performance_fixture_tuple
    from alpha.live.dashboard_v4.app import build_performance_page_dict

    def slow_page_dict(*args, **kwargs):
        result_dict = build_performance_page_dict(*args, **kwargs)
        clock_dict["now_ts"] += timedelta(seconds=121)
        return result_dict

    monkeypatch.setattr("alpha.live.dashboard_v4.app.build_performance_page_dict", slow_page_dict)
    response_obj, context_dict = _render_tuple(app_obj, path_str + "?level=pods&unit=usd&period=MTD")
    overview_dict = context_dict["overview_dict"]
    assert overview_dict["source_fresh_bool"] is False
    assert overview_dict["source_valid_ms_int"] == 0
    assert all(pod_dict["state_str"] == "unk" for pod_dict in overview_dict["pod_list"])
    assert 'data-source-valid-ms="0"' in response_obj.get_data(as_text=True)
    assert overview_dict["refresh_url_str"] == "/performance/status"
    assert _query_dict(context_dict["performance_page_dict"]["refresh_report_url_str"]) == {
        "level": ["pods"], "period": ["MTD"], "unit": ["usd"]}
    assert context_dict["performance_page_dict"]["report_dict"]["requested_to_date_str"] == "2026-09-04"


@pytest.mark.parametrize("level_str", ["portfolio", "pods"])
def test_csv_download_matches_selected_canonical_report(performance_fixture_tuple, level_str):
    workspace_dict, _, _, app_obj, _ = performance_fixture_tuple
    selection_dict = {"level": level_str, "period": "All", "unit": "usd",
        "from": "2026-09-01", "to": "2026-09-04"}
    _, context_dict = _render_tuple(app_obj, "/performance?" + urlencode(selection_dict))
    page_dict = context_dict["performance_page_dict"]
    report_dict = page_dict["report_dict"]
    download_query_dict = _query_dict(page_dict["csv_url_str"])
    assert download_query_dict == {**{key_str: [value_str] for key_str, value_str in selection_dict.items()},
        "download": ["csv"], "expected": [report_dict["report_hash_str"]]}
    response_obj = app_obj.test_client().get(page_dict["csv_url_str"])
    assert response_obj.status_code == 200 and response_obj.mimetype == "text/csv"
    assert response_obj.headers["Content-Disposition"] == (
        f'attachment; filename="performance-{level_str}-2026-09-01-2026-09-04.csv"')
    assert response_obj.headers["Cache-Control"] == "no-store"
    csv_str = response_obj.get_data(as_text=True)
    record_list = list(csv.DictReader(StringIO(csv_str)))
    expected_list = [report_dict] if level_str == "portfolio" else report_dict["strategy_list"]
    assert len(record_list) == len(expected_list)
    for record_dict, expected_dict in zip(record_list, expected_list):
        assert record_dict["Name"] == ("Portfolio" if level_str == "portfolio" else expected_dict["display_name_str"])
        assert record_dict["Selected from"] == "2026-09-01" and record_dict["Selected to"] == "2026-09-04"
        assert record_dict["Report hash"] == report_dict["report_hash_str"]
        assert record_dict["Source"] == "Saved IBKR account facts"
        assert record_dict["Currency"] == "USD" and record_dict["Demo"] == "Yes"
        for column_str, field_str in (("Start value", "opening_nav_float"), ("End value", "closing_nav_float"),
                                     ("Profit / loss", "pnl_float")):
            if expected_dict[field_str] is None:
                assert record_dict[column_str] == ""
            else:
                assert float(record_dict[column_str]) == expected_dict[field_str]
        if expected_dict["twr_float"] is None:
            assert record_dict["Return TWR (%)"] == ""
        else:
            assert float(record_dict["Return TWR (%)"]) == pytest.approx(100 * expected_dict["twr_float"], abs=0.0000005)
    for account_dict in workspace_dict["operations_account_list"]:
        assert account_dict["account_route"] not in csv_str
        assert account_dict["pod_id"] not in csv_str


def test_pdf_download_is_dated_and_bound_to_same_report(performance_fixture_tuple):
    workspace_dict, _, _, app_obj, _ = performance_fixture_tuple
    _, context_dict = _render_tuple(app_obj, "/performance?level=pods&from=2026-09-01&to=2026-09-04")
    page_dict = context_dict["performance_page_dict"]
    report_dict = page_dict["report_dict"]
    assert _query_dict(page_dict["pdf_url_str"])["expected"] == [report_dict["report_hash_str"]]
    response_obj = app_obj.test_client().get(page_dict["pdf_url_str"])
    assert response_obj.status_code == 200 and response_obj.mimetype == "application/pdf"
    assert response_obj.headers["Content-Disposition"] == 'attachment; filename="performance-portfolio-2026-09-01-2026-09-04.pdf"'
    assert response_obj.data.startswith(b"%PDF-")
    pdf_reader_obj = PdfReader(BytesIO(response_obj.data))
    text_str = "\n".join(page_obj.extract_text() for page_obj in pdf_reader_obj.pages)
    assert "2026-09-01" in text_str and "2026-09-04" in text_str
    # The PDF prints its issued-document identity, which also includes time.
    assert build_investor_snapshot_dict(report_dict)["document_hash_str"] in text_str
    assert "DEMONSTRATION" in text_str
    assert report_dict["client_name_str"] in text_str
    for account_dict in workspace_dict["operations_account_list"]:
        assert account_dict["account_route"] not in text_str
        assert account_dict["pod_id"] not in text_str


@pytest.mark.parametrize("download_str", ["csv", "pdf"])
def test_changed_report_blocks_export_before_renderer(performance_fixture_tuple, monkeypatch, download_str):
    workspace_dict, _, _, app_obj, _ = performance_fixture_tuple
    _, context_dict = _render_tuple(app_obj, "/performance?from=2026-09-01&to=2026-09-04")
    page_dict = context_dict["performance_page_dict"]
    workspace_dict["client_dict"]["display_name"] += " corrected"
    renderer_str = "export_performance_csv_str" if download_str == "csv" else "export_performance_pdf_bytes"
    monkeypatch.setattr("alpha.live.dashboard_v4.app." + renderer_str,
        lambda *args, **kwargs: pytest.fail("Changed report reached export renderer"))
    response_obj = app_obj.test_client().get(page_dict[download_str + "_url_str"])
    assert response_obj.status_code == 409
    assert "report changed" in response_obj.get_data(as_text=True).lower()
    assert "Content-Disposition" not in response_obj.headers
