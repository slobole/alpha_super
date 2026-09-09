from datetime import UTC, datetime
from html import unescape
import re

import pytest

from alpha.live.dashboard_v3.app import create_app
from alpha.live.dashboard_v3.client_views import nav_chart_dict
from alpha.live.dashboard_v3.demo import DemoOperationsProvider, build_demo_fixture_tuple
from test_client_reporting import client_config_dict, nav_attributes_dict, snapshot_obj
from test_dashboard_operator_access import ForbiddenProvider


@pytest.fixture
def financial_client_obj():
    config_dict = client_config_dict()
    source_obj = snapshot_obj([nav_attributes_dict()])
    app_obj = create_app(
        ForbiddenProvider(), read_only_bool=True,
        client_registry_dict={"schema_version": 1, "clients": [config_dict]},
        client_reporting_snapshot_fn=lambda client_id_str: source_obj,
    )
    app_obj.config["TESTING"] = True
    return app_obj.test_client()


@pytest.mark.parametrize("view_str", ["overview", "performance", "report"])
def test_financial_views_share_dates_numbers_and_local_style(financial_client_obj, view_str):
    response_obj = financial_client_obj.get(f"/clients/sample/{view_str}?from=2026-09-01&to=2026-09-01")
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
    assert '<footer class="client-evidence-strip">' in html_str
    assert 'Source: IBKR' in html_str and 'Data · JSON' in html_str
    assert 'Net capital movements do not establish' not in html_str
    assert response_obj.headers["Cache-Control"] == "no-store"


def test_all_views_export_same_period_result_hash(financial_client_obj):
    report_list = [financial_client_obj.get(f"/clients/sample/{view_str}?from=2026-09-01&to=2026-09-01&download=json").get_json() for view_str in ("overview", "performance", "report")]
    assert len({report_dict["report_hash_str"] for report_dict in report_list}) == 1
    assert all(report_dict["pnl_float"] == 10 for report_dict in report_list)


@pytest.mark.parametrize("view_str", ["overview", "performance", "report"])
def test_negative_dollars_are_readable_in_web_views(financial_client_obj, view_str):
    financial_client_obj.application.config["client_reporting_snapshot_fn"] = lambda client_id_str: snapshot_obj([
        nav_attributes_dict(closing_str="990", twr_str="-1", mtm="-10")])
    html_str = financial_client_obj.get(f"/clients/sample/{view_str}?from=2026-09-01&to=2026-09-01").get_data(as_text=True)
    assert "-$10.00" in html_str
    assert "$-10.00" not in html_str


@pytest.mark.parametrize("view_str", ["overview", "report"])
def test_compact_data_badge_never_calls_missing_capital_data_complete(financial_client_obj, view_str):
    source_row_dict = nav_attributes_dict()
    source_row_dict.pop("billPay")
    financial_client_obj.application.config["client_reporting_snapshot_fn"] = lambda client_id_str: snapshot_obj([source_row_dict])
    path_str = f"/clients/sample/{view_str}?from=2026-09-01&to=2026-09-01"
    report_dict = financial_client_obj.get(path_str + "&download=json").get_json()
    assert report_dict["strategy_list"][0]["coverage_complete_bool"] is True
    assert report_dict["strategy_list"][0]["flows_complete_bool"] is False
    html_str = financial_client_obj.get(path_str).get_data(as_text=True)
    assert 'data-label="Data">Incomplete' in html_str
    assert report_dict["strategy_list"][0]["observed_day_count_int"] == 1
    assert 'Data issues' in html_str and 'billPay' in html_str


@pytest.mark.parametrize("view_str", ["overview", "performance", "report"])
def test_charts_have_numbered_y_axes_instead_of_range_paragraph(financial_client_obj, view_str):
    html_str = financial_client_obj.get(f"/clients/sample/{view_str}?from=2026-09-01&to=2026-09-01").get_data(as_text=True)
    assert "Range:" not in html_str
    axis_list = re.findall(r'<div class="client-y-axis"[^>]*>(.*?)</div>', html_str, re.S)
    assert axis_list and all(axis_str.count('data-value=') == 3 for axis_str in axis_list)
    assert any('%' in axis_str for axis_str in axis_list) if view_str == "performance" else any('$' in axis_str for axis_str in axis_list)
    if view_str == "performance":
        assert "own scale" in html_str
        assert "Period-end drawdown" in html_str


@pytest.mark.parametrize("query_str", ["from=2026-09-01", "to=2026-09-01", "from=2026-09-01&to=2026-09-01&window=all", "from=2026-09-02&to=2026-09-01", "db_path=C:/private.sqlite3", "from=2026-09-01&from=2026-09-02&to=2026-09-02", "view_str=report"])
def test_ambiguous_dates_or_browser_paths_rejected(financial_client_obj, query_str):
    assert financial_client_obj.get(f"/clients/sample/overview?{query_str}").status_code == 400


def test_unknown_client_or_view_never_falls_back_to_another(financial_client_obj):
    assert financial_client_obj.get("/clients/other/overview").status_code == 404
    assert financial_client_obj.get("/clients/sample/shell").status_code == 404


def test_single_client_entry_opens_overview_without_client_selection(financial_client_obj):
    response_obj = financial_client_obj.get("/clients?from=2026-09-01&to=2026-09-01")
    assert response_obj.status_code == 302
    assert response_obj.location == "/clients/sample/overview?from=2026-09-01&to=2026-09-01"
    response_obj = financial_client_obj.get("/", follow_redirects=True)
    assert response_obj.status_code == 200
    assert response_obj.request.path == "/clients/sample/overview"
    html_str = response_obj.get_data(as_text=True)
    assert ">Clients<" not in html_str and "Switch client" not in html_str


@pytest.mark.parametrize("source_str", ["in_memory", "config_file"])
def test_multiple_client_setup_is_rejected_before_evidence_reads(tmp_path, source_str):
    import json

    registry_dict, _ = build_demo_fixture_tuple()
    argument_dict = {"client_registry_dict": registry_dict}
    if source_str == "config_file":
        config_path_obj = tmp_path / "clients.json"
        config_path_obj.write_text(json.dumps(registry_dict), encoding="utf-8")
        argument_dict = {"client_reporting_config_path_str": str(config_path_obj)}
    client_obj = create_app(ForbiddenProvider(), read_only_bool=True,
        client_reporting_snapshot_fn=lambda _: pytest.fail("Ambiguous client must not load financial evidence"),
        **argument_dict).test_client()
    for path_str in ("/clients", "/clients/demo-owner/overview", "/clients/demo-client/strategies",
                     "/clients/demo-client/report?download=json"):
        response_obj = client_obj.get(path_str)
        assert response_obj.status_code == (200 if path_str == "/clients" else 503)
        html_str = response_obj.get_data(as_text=True)
        assert "exactly one configured client" in html_str
        assert "Switch client" not in html_str and ">Clients<" not in html_str


@pytest.mark.parametrize("invalid_str", ["encoding", "nonfinite"])
def test_invalid_config_shows_setup_without_loading_evidence(tmp_path, invalid_str):
    config_path_obj = tmp_path / "private-client-config.json"
    argument_dict = {"client_reporting_config_path_str": str(config_path_obj)}
    if invalid_str == "encoding":
        config_path_obj.write_bytes(b"\xff")
    else:
        argument_dict = {"client_registry_dict": {"schema_version": 1, "clients": [client_config_dict()], "unused_float": float("nan")}}
    client_obj = create_app(ForbiddenProvider(), read_only_bool=True,
        client_reporting_snapshot_fn=lambda _: pytest.fail("Invalid config must not load evidence"),
        **argument_dict).test_client()
    for path_str in ("/clients", "/clients/sample/overview", "/clients/sample/report?download=json"):
        response_obj = client_obj.get(path_str)
        assert response_obj.status_code == (200 if path_str == "/clients" else 503)
        html_str = response_obj.get_data(as_text=True)
        assert "configuration could not be read or validated" in html_str
        assert str(config_path_obj) not in html_str and "codec" not in html_str


@pytest.mark.parametrize("from_str,to_str", [("2026-09-02", "2026-09-01"), ("2026-08-01", "2026-09-01"), ("20260901", "2026-09-01")])
@pytest.mark.parametrize("view_str", ["overview", "activity", "report"])
def test_invalid_period_preserves_client_dates_and_editable_form(financial_client_obj, from_str, to_str, view_str):
    response_obj = financial_client_obj.get(f"/clients/sample/{view_str}?from={from_str}&to={to_str}")
    html_str = response_obj.get_data(as_text=True)
    assert response_obj.status_code == 400
    assert 'role="alert"' in html_str and "Correct the selected period" in html_str
    assert f'value="{from_str}"' in html_str and f'value="{to_str}"' in html_str
    assert 'name="from" type="date"' in html_str
    assert "/clients/sample/" in html_str
    assert "No dates or results have been silently substituted" not in html_str
    assert financial_client_obj.get(f"/clients/sample/{view_str}?from=2026-09-01&to=2026-09-01").status_code == 200


def test_owned_weekend_without_valuation_rows_does_not_claim_no_ownership(financial_client_obj):
    config_dict = financial_client_obj.application.config["client_registry_dict"]["clients"][0]
    config_dict["mandate_start_date"] = "2026-08-01"
    config_dict["accounts"][0]["effective_from"] = "2026-08-01"
    response_obj = financial_client_obj.get("/clients/sample/performance?from=2026-08-29&to=2026-08-30")
    assert response_obj.status_code == 200
    html_str = response_obj.get_data(as_text=True)
    assert "No strategy valuation rows are available" in html_str
    assert "No strategy period overlaps" not in html_str


def test_client_pages_and_report_exports_do_not_require_login(financial_client_obj):
    for path_str in ("/clients", "/clients/sample/overview?from=2026-09-01&to=2026-09-01",
                     "/clients/sample/report?from=2026-09-01&to=2026-09-01&download=json"):
        response_obj = financial_client_obj.get(path_str, follow_redirects=True)
        assert response_obj.status_code == 200
        assert "WWW-Authenticate" not in response_obj.headers
        assert response_obj.headers["Cache-Control"] == "no-store"
    report_path_str = "/clients/sample/report?from=2026-09-01&to=2026-09-01"
    report_dict = financial_client_obj.get(report_path_str + "&download=json").get_json()
    assert financial_client_obj.get(report_path_str + "&download=pdf").status_code == 409
    response_obj = financial_client_obj.get(
        report_path_str + "&download=pdf&expected=" + report_dict["report_hash_str"]
    )
    assert response_obj.status_code == 200
    assert response_obj.mimetype == "application/pdf"
    assert "WWW-Authenticate" not in response_obj.headers
    assert response_obj.headers["Cache-Control"] == "no-store"


def test_report_route_does_not_write_missing_database(tmp_path):
    database_path_obj = tmp_path / "missing.sqlite3"
    app_obj = create_app(ForbiddenProvider(), read_only_bool=True,
        performance_db_path_str=str(database_path_obj), client_registry_dict={"schema_version": 1, "clients": [client_config_dict()]})
    response_obj = app_obj.test_client().get("/clients/sample/report?from=2026-09-01&to=2026-09-01&download=json")
    assert response_obj.status_code == 200
    assert response_obj.get_json()["status_str"] == "draft"
    assert not database_path_obj.exists()


def test_nav_chart_keeps_missing_intervals_as_separate_segments():
    chart_dict = nav_chart_dict([{"nav_float": 100}, {"nav_float": 110}, {"nav_float": None}, {"nav_float": 120}])
    assert len(chart_dict["segment_list"]) == 2


@pytest.mark.parametrize('return_str,closing_str,profit_str', [('1', '1010', '10'), ('0', '1000', '0')])
def test_account_chart_offers_verified_official_single_account_return(financial_client_obj, return_str, closing_str, profit_str):
    financial_client_obj.application.config['client_reporting_snapshot_fn'] = lambda client_id_str: snapshot_obj([
        nav_attributes_dict(twr_str=return_str, closing_str=closing_str, mtm=profit_str)])
    html_str = financial_client_obj.get('/clients/sample/overview?from=2026-09-01&to=2026-09-01').get_data(as_text=True)
    assert 'data-account-unit="pct" aria-pressed="true"' in html_str
    assert 'data-account-unit="usd"' in html_str
    assert 'Cumulative TWR' in html_str
    assert 'USD · includes transfers' in html_str
    assert 'aria-label="Portfolio cumulative return (%)"' in html_str


def test_future_strategy_does_not_turn_missing_selected_history_into_setup_warning(financial_client_obj):
    config_dict = financial_client_obj.application.config['client_registry_dict']['clients'][0]
    config_dict['accounts'].append(dict(config_dict['accounts'][0], account_route='U_TEST_B',
        pod_id='pod_b', display_name='Strategy B', effective_from='2026-09-03'))
    path_str = '/clients/sample/overview?from=2026-09-01&to=2026-09-02'
    report_dict = financial_client_obj.get(path_str + '&download=json').get_json()
    assert len(report_dict['strategy_list']) == 1 and report_dict['twr_float'] is None
    html_str = financial_client_obj.get(path_str).get_data(as_text=True)
    assert 'data-account-unit="pct" aria-pressed="false" disabled' in html_str
    assert 'Incomplete IBKR return data' in html_str
    assert 'Portfolio return setup required' not in html_str


def test_configured_failed_return_cannot_fall_back_to_valid_account_twr(financial_client_obj):
    from test_client_twr import twr_config_dict

    config_dict = financial_client_obj.application.config['client_registry_dict']['clients'][0]
    config_dict['client_twr'] = twr_config_dict(False)['client_twr']
    row_dict = nav_attributes_dict()
    row_dict.pop('billPay')
    financial_client_obj.application.config['client_reporting_snapshot_fn'] = lambda client_id_str: snapshot_obj([row_dict])
    path_str = '/clients/sample/overview?from=2026-09-01&to=2026-09-01'
    report_dict = financial_client_obj.get(path_str + '&download=json').get_json()
    assert report_dict['twr_float'] is None and report_dict['pnl_float'] is None
    assert report_dict['strategy_list'][0]['twr_float'] == pytest.approx(.01)
    html_str = financial_client_obj.get(path_str).get_data(as_text=True)
    assert 'data-account-unit="pct" aria-pressed="false" disabled' in html_str
    assert 'aria-label="Portfolio cumulative return (%)"' not in html_str
    assert 'Incomplete IBKR return data' in html_str and 'Incomplete IBKR data' in html_str
    assert 'IBKR cash-flow setup required' not in html_str


def test_missing_mapping_never_turns_nav_change_into_return_or_profit(financial_client_obj):
    config_dict = financial_client_obj.application.config['client_registry_dict']['clients'][0]
    config_dict.pop('nav_bridge')
    config_dict['accounts'].append(dict(config_dict['accounts'][0], account_route='U_TEST_B', pod_id='pod_b', display_name='Strategy B'))
    financial_client_obj.application.config['client_reporting_snapshot_fn'] = lambda client_id_str: snapshot_obj([
        nav_attributes_dict(), nav_attributes_dict('U_TEST_B')])
    path_str = '/clients/sample/overview?from=2026-09-01&to=2026-09-01'
    report_dict = financial_client_obj.get(path_str + '&download=json').get_json()
    html_str = financial_client_obj.get(path_str).get_data(as_text=True)
    assert report_dict['pnl_float'] is None and report_dict['twr_float'] is None
    assert 'data-account-unit="pct" aria-pressed="false" disabled' in html_str
    assert 'IBKR cash-flow setup required' in html_str
    assert 'Portfolio return setup required' in html_str
    assert 'a verified breakdown of IBKR capital movements is not available' not in html_str


@pytest.mark.parametrize("access_token_str", [None, "short", "obsolete-operator-credential-123456"])
@pytest.mark.parametrize("demo_bool", [False, True])
def test_launcher_serves_without_operator_credential(monkeypatch, access_token_str, demo_bool):
    from types import SimpleNamespace
    from alpha.live.dashboard_v3 import __main__ as cli_module
    if access_token_str is None:
        monkeypatch.delenv("ALPHA_OPS_OPERATOR_ACCESS_TOKEN_STR", raising=False)
    else:
        monkeypatch.setenv("ALPHA_OPS_OPERATOR_ACCESS_TOKEN_STR", access_token_str)
    captured_options_dict = {}
    captured_run_dict = {}

    def create_app_fn(*args, **options_dict):
        captured_options_dict.update(options_dict)
        return SimpleNamespace(run=lambda **run_dict: captured_run_dict.update(run_dict))

    monkeypatch.setattr(cli_module, "create_app", create_app_fn)
    monkeypatch.setattr("sys.argv", ["dashboard", "--skip-env-file"] + (["--demo"] if demo_bool else []))
    assert cli_module.main() == 0
    assert captured_options_dict["read_only_bool"] is True
    assert "operator_access_token_str" not in captured_options_dict
    assert captured_run_dict["host"] == "127.0.0.1"
    assert captured_run_dict["use_reloader"] is False


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
    response_obj = financial_client_obj.get("/clients/sample/overview?window=mtd&download=json")
    assert response_obj.status_code == 200
    report_dict = response_obj.get_json()
    assert report_dict["requested_from_date_str"] == report_dict["requested_to_date_str"] == "2026-10-01"
    assert report_dict["status_str"] == "draft"
    assert report_dict["twr_float"] is None
    assert report_dict["pnl_float"] is None
    html_str = financial_client_obj.get("/clients/sample/overview?window=mtd").get_data(as_text=True)
    assert "Current client operations" in html_str


@pytest.mark.parametrize("error_obj", [ValueError("SECRET SOURCE PATH"), OSError("SECRET SOURCE PATH")])
def test_financial_failure_preserves_client_context_and_operations(financial_client_obj, error_obj):
    def fail_snapshot_fn(client_id_str):
        raise error_obj

    financial_client_obj.application.config["client_reporting_snapshot_fn"] = fail_snapshot_fn
    response_obj = financial_client_obj.get("/clients/sample/overview?from=2026-09-01&to=2026-09-01")
    html_str = response_obj.get_data(as_text=True)
    assert response_obj.status_code == 200
    assert "Financial evidence unavailable" in html_str
    assert "Current client operations" in html_str
    assert "/clients/sample/diagnostics?" in html_str
    assert "SECRET SOURCE PATH" not in html_str
    assert "$1,010.00" not in html_str
    assert financial_client_obj.get("/clients/sample/report?download=json").status_code == 503


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
        client_registry_dict={**registry_dict, "clients": registry_dict["clients"][1:]},
        client_reporting_snapshot_fn=lambda client_id_str: snapshot_dict[client_id_str])
    client_obj = app_obj.test_client()
    for view_str in ("strategies", "diagnostics", "exposure", "activity"):
        html_str = client_obj.get(f"/clients/demo-client/{view_str}{query_str}").get_data(as_text=True)
        href_str = unescape(re.search(r'href="([^"]+/performance[^\"]*)"', html_str).group(1))
        assert href_str == "/clients/demo-client/performance" + query_str
        separator_str = "&" if query_str else "?"
        via_dict = client_obj.get(href_str + separator_str + "download=json").get_json()
        direct_dict = client_obj.get("/clients/demo-client/performance" + query_str + separator_str + "download=json").get_json()
        assert via_dict["report_hash_str"] == direct_dict["report_hash_str"]
        assert via_dict["pnl_float"] is not None


@pytest.mark.parametrize("read_only_bool,label_str", [(True, "Read-only"), (False, "Actions enabled")])
def test_client_header_reflects_actual_operator_mode(financial_client_obj, read_only_bool, label_str):
    financial_client_obj.application.config["read_only_bool"] = read_only_bool
    html_str = financial_client_obj.get("/clients/sample/overview?from=2026-09-01&to=2026-09-01").get_data(as_text=True)
    assert f'<span class="client-badge">{label_str}</span>' in html_str
