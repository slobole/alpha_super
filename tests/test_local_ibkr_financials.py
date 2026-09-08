"""Actual single-VPS entrypoint, temp ledgers/imports, no reporting registry."""

from io import BytesIO

import pytest
from pypdf import PdfReader

from alpha.live.ibkr_performance import PerformanceStore
from alpha.live.dashboard_v3.local_workspace import _saved_binding_list
from test_dashboard_local_workspace import build_fixture_app, file_snapshot_dict
from test_ibkr_performance import _xml_str


def test_expanded_default_local_workspace_shows_identical_money_and_exports(tmp_path, monkeypatch):
    app_obj = build_fixture_app(tmp_path, monkeypatch, new_pod_bool=False, expanded_bool=True)
    client_obj = app_obj.test_client()
    before_dict = file_snapshot_dict(tmp_path)
    result_list = []
    expected_return_float = (1 + 60 / 11000) * (1 - 25 / 11160) - 1
    for view_str in ("overview", "performance", "report"):
        path_str = f"/clients/local/{view_str}?from=2026-09-02&to=2026-09-03"
        result_dict = client_obj.get(path_str + "&download=json").get_json()
        result_list.append(result_dict)
        assert result_dict["status_str"] == "ready" and result_dict["scope_complete_bool"]
        assert result_dict["opening_nav_float"] == 11000 and result_dict["closing_nav_float"] == 11135
        assert result_dict["capital_movement_float"] == 100 and result_dict["pnl_float"] == 35
        assert result_dict["twr_float"] == pytest.approx(expected_return_float)
        assert result_dict["client_twr_configured_bool"] and result_dict["twr_method_id_str"] == "daily_nav_eod_v1"
        assert [row_dict["pnl_float"] for row_dict in result_dict["daily_book_list"]] == [60, -25]
        assert [row_dict["pnl_float"] for row_dict in result_dict["strategy_list"]] == [5, 30]
        assert result_dict["strategy_list"][0]["daily_list"][0]["return_float"] == .008
        assert result_dict["strategy_list"][0]["twr_float"] == pytest.approx(1.008 * (1 - 5/1110) - 1)
        html_str = client_obj.get(path_str).get_data(as_text=True)
        assert "$35.00" in html_str and "0.32%" in html_str
        assert "IBKR cash-flow setup required" not in html_str
        if view_str == "overview":
            assert 'data-account-unit="pct" aria-pressed="true"' in html_str
    assert len({result_dict["report_hash_str"] for result_dict in result_list}) == 1
    response_obj = client_obj.get("/clients/local/report?from=2026-09-02&to=2026-09-03&download=pdf&expected=" + result_list[0]["report_hash_str"])
    assert response_obj.status_code == 200
    pdf_text_str = "\n".join(page_obj.extract_text() for page_obj in PdfReader(BytesIO(response_obj.data)).pages)
    assert "$35.00" in pdf_text_str and "0.32%" in pdf_text_str
    assert "FINAL" in pdf_text_str
    assert file_snapshot_dict(tmp_path) == before_dict


def test_older_incomplete_period_never_borrows_new_expanded_fields(tmp_path, monkeypatch):
    app_obj = build_fixture_app(tmp_path, monkeypatch, new_pod_bool=False, expanded_bool=True)
    client_obj = app_obj.test_client()
    result_dict = client_obj.get("/clients/local/overview?from=2026-09-01&to=2026-09-03&download=json").get_json()
    assert result_dict["coverage_complete_bool"] and not result_dict["flows_complete_bool"]
    assert result_dict["twr_float"] is None and result_dict["pnl_float"] is None
    assert result_dict["closing_nav_float"] == 11135
    assert all(row_dict["twr_float"] is not None for row_dict in result_dict["strategy_list"])
    assert all(row_dict["pnl_float"] is None for row_dict in result_dict["daily_book_list"])
    assert result_dict["return_path_list"] == []
    html_str = client_obj.get("/clients/local/overview?from=2026-09-01&to=2026-09-03").get_data(as_text=True)
    assert 'data-account-unit="pct" aria-pressed="false" disabled' in html_str
    assert "Incomplete IBKR data" in html_str and "Portfolio return setup required" not in html_str
    # Choosing complete dates does not mutate the source or shrink ALL implicitly.
    selected_dict = client_obj.get("/clients/local/overview?from=2026-09-02&to=2026-09-03&download=json").get_json()
    assert selected_dict["pnl_float"] == 35


def test_complete_existing_accounts_do_not_hide_new_strategy_missing_history(tmp_path, monkeypatch):
    app_obj = build_fixture_app(tmp_path, monkeypatch, new_pod_bool=True, expanded_bool=True)
    client_obj = app_obj.test_client()
    path_str = "/clients/local/overview?from=2026-09-02&to=2026-09-03"
    result_dict = client_obj.get(path_str + "&download=json").get_json()
    assert not result_dict["scope_complete_bool"]
    assert result_dict["closing_nav_float"] is None
    assert result_dict["pnl_float"] is None and result_dict["twr_float"] is None
    assert [row_dict["pnl_float"] for row_dict in result_dict["strategy_list"]] == [5, 30]
    assert all(row_dict["twr_float"] is not None for row_dict in result_dict["strategy_list"])
    html_str = client_obj.get(path_str).get_data(as_text=True)
    assert 'data-account-unit="pct" aria-pressed="false" disabled' in html_str
    operations_dict = client_obj.get("/clients/local/diagnostics?download=json").get_json()
    assert {row_dict["pod_id_str"] for row_dict in operations_dict["strategy_list"]} == {"pod_a", "pod_b", "pod_new"}


def test_source_revision_removing_fields_invalidates_export_and_never_resurrects_money(tmp_path, monkeypatch):
    app_obj = build_fixture_app(tmp_path, monkeypatch, new_pod_bool=False, expanded_bool=True)
    client_obj = app_obj.test_client()
    path_str = "/clients/local/report?from=2026-09-02&to=2026-09-03"
    before_dict = client_obj.get(path_str + "&download=json").get_json()
    database_path_str = app_obj.config["performance_db_path_str"]
    # Only the isolated fixture is revised, through the real existing importer.
    PerformanceStore(database_path_str).replace_range(
        xml_text_str=_xml_str([("U100", "2026-09-02", 1000, 1110, .8), ("U200", "2026-09-02", 10000, 10050, .5),
            ("U100", "2026-09-03", 1110, 1105, -.45045045045045), ("U200", "2026-09-03", 10050, 10030, -.199004975124378)]),
        query_name_str="ALPHA_DAILY_TWR", request_from_date_str="2026-09-02", request_to_date_str="2026-09-03",
        binding_obj_list=_saved_binding_list(database_path_str), imported_timestamp_str="2026-09-05T12:00:00+00:00")
    after_dict = client_obj.get(path_str + "&download=json").get_json()
    assert after_dict["pnl_float"] is None and after_dict["twr_float"] is None
    assert before_dict["report_hash_str"] != after_dict["report_hash_str"]
    assert client_obj.get(path_str + "&download=pdf&expected=" + before_dict["report_hash_str"]).status_code == 409
    assert client_obj.get(path_str + "&download=bundle&expected=" + before_dict["report_hash_str"]).status_code == 409
