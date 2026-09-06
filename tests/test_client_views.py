from datetime import UTC, datetime
from html import unescape
import re

import pytest

from alpha.live.dashboard_v3.app import create_app
from alpha.live.dashboard_v3.client_views import nav_chart_dict
from alpha.live.dashboard_v3.demo import DemoOperationsProvider, build_demo_fixture_tuple
from test_client_reporting import client_config_dict, nav_attributes_dict, snapshot_obj
from test_dashboard_operator_access import TEST_ACCESS_STR, ForbiddenProvider, auth_headers_dict


@pytest.fixture
def financial_client_obj():
    config_dict = client_config_dict()
    source_obj = snapshot_obj([nav_attributes_dict()])
    app_obj = create_app(
        ForbiddenProvider(), read_only_bool=True, operator_access_token_str=TEST_ACCESS_STR,
        client_registry_dict={"schema_version": 1, "clients": [config_dict]},
        client_reporting_snapshot_fn=lambda client_id_str: source_obj,
    )
    app_obj.config["TESTING"] = True
    return app_obj.test_client()


@pytest.mark.parametrize("view_str", ["overview", "performance", "report"])
def test_authenticated_financial_views_share_dates_numbers_and_local_style(financial_client_obj, view_str):
    response_obj = financial_client_obj.get(f"/clients/sample/{view_str}?from=2026-09-01&to=2026-09-01", headers=auth_headers_dict())
    assert response_obj.status_code == 200
    html_str = response_obj.get_data(as_text=True)
    assert "$1,010.00" in html_str
    assert "$10.00" in html_str
    assert "1.00%" in html_str
    assert 'value="2026-09-01"' in html_str
    assert "Strategy A" in html_str
    assert "cdn.tailwindcss.com" not in html_str
    assert '<span class="client-badge">Read-only</span>' in html_str
    assert "Investors do not access this workspace" not in html_str
    assert "Broker value, capital movements and investment results" not in html_str
    assert '<details class="client-details client-source-details">' in html_str
    assert response_obj.headers["Cache-Control"] == "no-store"


def test_all_views_export_same_period_result_hash(financial_client_obj):
    report_list = [financial_client_obj.get(f"/clients/sample/{view_str}?from=2026-09-01&to=2026-09-01&download=json", headers=auth_headers_dict()).get_json() for view_str in ("overview", "performance", "report")]
    assert len({report_dict["report_hash_str"] for report_dict in report_list}) == 1
    assert all(report_dict["pnl_float"] == 10 for report_dict in report_list)


@pytest.mark.parametrize("view_str", ["overview", "performance", "report"])
def test_negative_dollars_are_readable_in_web_views(financial_client_obj, view_str):
    financial_client_obj.application.config["client_reporting_snapshot_fn"] = lambda client_id_str: snapshot_obj([
        nav_attributes_dict(closing_str="990", twr_str="-1", mtm="-10")])
    html_str = financial_client_obj.get(f"/clients/sample/{view_str}?from=2026-09-01&to=2026-09-01", headers=auth_headers_dict()).get_data(as_text=True)
    assert "-$10.00" in html_str
    assert "$-10.00" not in html_str


@pytest.mark.parametrize("view_str", ["overview", "report"])
def test_compact_data_badge_never_calls_missing_capital_data_complete(financial_client_obj, view_str):
    source_row_dict = nav_attributes_dict()
    source_row_dict.pop("billPay")
    financial_client_obj.application.config["client_reporting_snapshot_fn"] = lambda client_id_str: snapshot_obj([source_row_dict])
    path_str = f"/clients/sample/{view_str}?from=2026-09-01&to=2026-09-01"
    report_dict = financial_client_obj.get(path_str + "&download=json", headers=auth_headers_dict()).get_json()
    assert report_dict["strategy_list"][0]["coverage_complete_bool"] is True
    assert report_dict["strategy_list"][0]["flows_complete_bool"] is False
    html_str = financial_client_obj.get(path_str, headers=auth_headers_dict()).get_data(as_text=True)
    assert 'data-label="Data">Incomplete' in html_str
    assert "1/1 observed days · incomplete capital data" in html_str


@pytest.mark.parametrize("view_str", ["overview", "performance", "report"])
def test_compact_chart_labels_extrema_as_range_not_endpoint_return(financial_client_obj, view_str):
    html_str = financial_client_obj.get(f"/clients/sample/{view_str}?from=2026-09-01&to=2026-09-01", headers=auth_headers_dict()).get_data(as_text=True)
    assert "Range:" in html_str
    if view_str == "performance":
        assert "own scale" in html_str
        assert "Period-end drawdown" in html_str


@pytest.mark.parametrize("query_str", ["from=2026-09-01", "to=2026-09-01", "from=2026-09-01&to=2026-09-01&window=all", "from=2026-09-02&to=2026-09-01", "db_path=C:/private.sqlite3", "from=2026-09-01&from=2026-09-02&to=2026-09-02", "view_str=report"])
def test_ambiguous_dates_or_browser_paths_rejected(financial_client_obj, query_str):
    assert financial_client_obj.get(f"/clients/sample/overview?{query_str}", headers=auth_headers_dict()).status_code == 400


def test_unknown_client_or_view_never_falls_back_to_another(financial_client_obj):
    assert financial_client_obj.get("/clients/other/overview", headers=auth_headers_dict()).status_code == 404
    assert financial_client_obj.get("/clients/sample/shell", headers=auth_headers_dict()).status_code == 404


@pytest.mark.parametrize("from_str,to_str", [("2026-09-02", "2026-09-01"), ("2026-08-01", "2026-09-01"), ("20260901", "2026-09-01")])
@pytest.mark.parametrize("view_str", ["overview", "activity", "report"])
def test_invalid_period_preserves_client_dates_and_editable_form(financial_client_obj, from_str, to_str, view_str):
    response_obj = financial_client_obj.get(f"/clients/sample/{view_str}?from={from_str}&to={to_str}", headers=auth_headers_dict())
    html_str = response_obj.get_data(as_text=True)
    assert response_obj.status_code == 400
    assert 'role="alert"' in html_str and "Correct the selected period" in html_str
    assert f'value="{from_str}"' in html_str and f'value="{to_str}"' in html_str
    assert 'name="from" type="date"' in html_str
    assert "/clients/sample/" in html_str
    assert "No dates or results have been silently substituted" in html_str
    assert financial_client_obj.get(f"/clients/sample/{view_str}?from=2026-09-01&to=2026-09-01", headers=auth_headers_dict()).status_code == 200


def test_owned_weekend_without_valuation_rows_does_not_claim_no_ownership(financial_client_obj):
    config_dict = financial_client_obj.application.config["client_registry_dict"]["clients"][0]
    config_dict["mandate_start_date"] = "2026-08-01"
    config_dict["accounts"][0]["effective_from"] = "2026-08-01"
    response_obj = financial_client_obj.get("/clients/sample/performance?from=2026-08-29&to=2026-08-30", headers=auth_headers_dict())
    assert response_obj.status_code == 200
    html_str = response_obj.get_data(as_text=True)
    assert "No strategy valuation rows are available" in html_str
    assert "No strategy period overlaps" not in html_str


def test_unauthenticated_report_and_export_are_not_accessible(financial_client_obj):
    for path_str in ("/clients", "/clients/sample/overview", "/clients/sample/report?download=json"):
        assert financial_client_obj.get(path_str).status_code == 401


def test_report_route_does_not_write_missing_database(tmp_path):
    database_path_obj = tmp_path / "missing.sqlite3"
    app_obj = create_app(ForbiddenProvider(), read_only_bool=True, operator_access_token_str=TEST_ACCESS_STR,
        performance_db_path_str=str(database_path_obj), client_registry_dict={"schema_version": 1, "clients": [client_config_dict()]})
    response_obj = app_obj.test_client().get("/clients/sample/report?from=2026-09-01&to=2026-09-01&download=json", headers=auth_headers_dict())
    assert response_obj.status_code == 200
    assert response_obj.get_json()["status_str"] == "draft"
    assert not database_path_obj.exists()


def test_nav_chart_keeps_missing_intervals_as_separate_segments():
    chart_dict = nav_chart_dict([{"nav_float": 100}, {"nav_float": 110}, {"nav_float": None}, {"nav_float": 120}])
    assert len(chart_dict["segment_list"]) == 2


def test_launcher_refuses_to_serve_without_operator_credential(monkeypatch):
    from alpha.live.dashboard_v3 import __main__ as cli_module
    monkeypatch.delenv("ALPHA_OPS_OPERATOR_ACCESS_TOKEN_STR", raising=False)
    monkeypatch.setattr("sys.argv", ["dashboard", "--skip-env-file"])
    with pytest.raises(SystemExit) as error_obj:
        cli_module.main()
    assert error_obj.value.code == 2


@pytest.mark.parametrize("as_of_str,window_str", [("2026-10-01T12:00:00+00:00", "mtd"), ("2027-01-01T12:00:00+00:00", "ytd")])
def test_empty_current_month_year_preset_retains_pending_period(as_of_str, window_str):
    from datetime import datetime
    from alpha.live.dashboard_v3.client_views import _period_tuple
    app_obj = create_app(ForbiddenProvider())
    with app_obj.test_request_context(f"/?window={window_str}"):
        assert _period_tuple(client_config_dict(), datetime.fromisoformat(as_of_str)) == (as_of_str[:10], as_of_str[:10])


def test_pending_current_period_keeps_operations_and_withholds_financials(monkeypatch, financial_client_obj):
    from datetime import datetime

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.fromisoformat("2026-10-01T12:00:00+00:00")

    monkeypatch.setattr("alpha.live.dashboard_v3.client_views.datetime", FixedDatetime)
    response_obj = financial_client_obj.get("/clients/sample/overview?window=mtd&download=json", headers=auth_headers_dict())
    assert response_obj.status_code == 200
    report_dict = response_obj.get_json()
    assert report_dict["requested_from_date_str"] == report_dict["requested_to_date_str"] == "2026-10-01"
    assert report_dict["status_str"] == "draft"
    assert report_dict["twr_float"] is None
    assert report_dict["pnl_float"] is None
    html_str = financial_client_obj.get("/clients/sample/overview?window=mtd", headers=auth_headers_dict()).get_data(as_text=True)
    assert "Current client operations" in html_str


@pytest.mark.parametrize("error_obj", [ValueError("SECRET SOURCE PATH"), OSError("SECRET SOURCE PATH")])
def test_financial_failure_preserves_client_context_and_operations(financial_client_obj, error_obj):
    def fail_snapshot_fn(client_id_str):
        raise error_obj

    financial_client_obj.application.config["client_reporting_snapshot_fn"] = fail_snapshot_fn
    response_obj = financial_client_obj.get("/clients/sample/overview?from=2026-09-01&to=2026-09-01", headers=auth_headers_dict())
    html_str = response_obj.get_data(as_text=True)
    assert response_obj.status_code == 200
    assert "Financial evidence unavailable" in html_str
    assert "Current client operations" in html_str
    assert "/clients/sample/diagnostics?" in html_str
    assert "SECRET SOURCE PATH" not in html_str
    assert "$1,010.00" not in html_str
    assert financial_client_obj.get("/clients/sample/report?download=json", headers=auth_headers_dict()).status_code == 503


@pytest.mark.parametrize("as_of_str", ["2026-09-04T16:00:00+00:00", "2026-09-06T16:00:00+00:00"])
@pytest.mark.parametrize("query_str", ["", "?window=mtd", "?from=2026-09-01&to=2026-09-02"])
def test_operation_navigation_keeps_selection_not_its_today_default(monkeypatch, as_of_str, query_str):
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.fromisoformat(as_of_str)

    monkeypatch.setattr("alpha.live.dashboard_v3.client_views.datetime", FixedDatetime)
    registry_dict, snapshot_dict = build_demo_fixture_tuple()
    app_obj = create_app(DemoOperationsProvider(), read_only_bool=True, demo_mode_bool=True,
        operator_access_token_str=TEST_ACCESS_STR, client_registry_dict=registry_dict,
        client_reporting_snapshot_fn=lambda client_id_str: snapshot_dict[client_id_str])
    client_obj = app_obj.test_client()
    for view_str in ("strategies", "diagnostics", "exposure", "activity"):
        html_str = client_obj.get(f"/clients/demo-client/{view_str}{query_str}", headers=auth_headers_dict()).get_data(as_text=True)
        href_str = unescape(re.search(r'href="([^"]+/performance[^\"]*)"', html_str).group(1))
        assert href_str == "/clients/demo-client/performance" + query_str
        separator_str = "&" if query_str else "?"
        via_dict = client_obj.get(href_str + separator_str + "download=json", headers=auth_headers_dict()).get_json()
        direct_dict = client_obj.get("/clients/demo-client/performance" + query_str + separator_str + "download=json", headers=auth_headers_dict()).get_json()
        assert via_dict["report_hash_str"] == direct_dict["report_hash_str"]
        assert via_dict["pnl_float"] is not None


@pytest.mark.parametrize("read_only_bool,label_str", [(True, "Read-only"), (False, "Actions enabled")])
def test_client_header_reflects_actual_operator_mode(financial_client_obj, read_only_bool, label_str):
    financial_client_obj.application.config["read_only_bool"] = read_only_bool
    html_str = financial_client_obj.get("/clients/sample/overview?from=2026-09-01&to=2026-09-01", headers=auth_headers_dict()).get_data(as_text=True)
    assert f'<span class="client-badge">{label_str}</span>' in html_str
