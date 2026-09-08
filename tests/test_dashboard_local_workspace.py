"""Normal startup against isolated production-format files, not the demo UI."""

from datetime import UTC, datetime
from io import BytesIO
import sqlite3

import pytest
from pypdf import PdfReader

from alpha.live.dashboard_v3.app import create_app
from alpha.live.dashboard_v3.data import DashboardDataProvider
from alpha.live.dashboard_v3.local_workspace import build_local_workspace_dict
from alpha.live.dashboard_v3.local_workspace import _saved_binding_list
from alpha.live.ibkr_performance import PerformanceStore, PodPerformanceBinding
from alpha.live.release_manifest import load_release_list
from test_ibkr_performance import _xml_str
from test_live_dashboard import (
    _write_release_manifest, _write_config, _seed_eod_pod_state,
    _seed_decision_vplan_and_broker_rows, _mark_latest_vplan_completed_and_reconciled,
)


VIEW_TUPLE = ("overview", "performance", "strategies", "exposure", "activity", "diagnostics", "report")


def build_fixture_app(tmp_path, monkeypatch, *, finance_bool=True, new_pod_bool=True, expanded_bool=False):
    monkeypatch.delenv("ALPHA_CLIENT_REPORTING_CONFIG_PATH_STR", raising=False)
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "true")
    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(tmp_path / "snapshots"))
    monkeypatch.setenv("IBKR_FLEX_QUERY_NAME_STR", "ALPHA_DAILY_TWR")
    flow_path_obj = tmp_path / "flows.yaml"
    flow_path_obj.write_text("flows: []\n", encoding="utf-8")
    monkeypatch.setenv("ALPHA_POD_CASH_FLOWS_PATH_STR", str(flow_path_obj))
    release_root_obj, config_path_obj = tmp_path / "releases", tmp_path / "dashboard.yaml"
    release_tuple_list = [("pod_a", "U100", "live", True), ("pod_b", "U200", "live", True),
        ("pod_retired", "U400", "live", False), ("pod_sim", "SIM_fixture", "incubation", True)]
    if expanded_bool:
        release_tuple_list = release_tuple_list[:2]
    if new_pod_bool:
        release_tuple_list.append(("pod_new", "U300", "live", True))
    for pod_str, account_str, mode_str, enabled_bool in release_tuple_list:
        _write_release_manifest(release_root_obj, user_id_str="local_owner", pod_id_str=pod_str,
            mode_str=mode_str, account_route_str=account_str, enabled_bool=enabled_bool)
    _write_config(config_path_obj, {pod_str: str(tmp_path / (pod_str + ".sqlite3")) for pod_str, _, _, _ in release_tuple_list})
    for release_obj in load_release_list(str(release_root_obj)):
        if release_obj.pod_id_str == "pod_new":
            continue
        database_path_obj = tmp_path / (release_obj.pod_id_str + ".sqlite3")
        _seed_decision_vplan_and_broker_rows(database_path_obj, release_obj)
        _mark_latest_vplan_completed_and_reconciled(database_path_obj, release_obj)
        for day_int, month_int in ((31, 8), (1, 9)):
            _seed_eod_pod_state(database_path_obj, release_obj, total_value_float=1010,
                updated_timestamp_ts=datetime(2026, month_int, day_int, 20, 10, tzinfo=UTC))
    performance_path_obj = tmp_path / "performance.sqlite3"
    if finance_bool:
        binding_list = [PodPerformanceBinding(account_route_str=account_str, pod_id_str=pod_str,
            return_start_date_str="2026-09-01", return_end_date_str=None if enabled_bool else "2026-09-01",
            enabled_bool=enabled_bool) for pod_str, account_str, mode_str, enabled_bool in release_tuple_list
            if mode_str == "live" and pod_str != "pod_new"]
        PerformanceStore(str(performance_path_obj)).replace_range(
            xml_text_str=_xml_str([(binding_obj.account_route_str, "2026-09-01",
                (990 if binding_obj.account_route_str == "U100" else 9990) if expanded_bool else 1000,
                (1000 if binding_obj.account_route_str == "U100" else 10000) if expanded_bool else 1010, 1)
                for binding_obj in binding_list]),
            query_name_str="ALPHA_DAILY_TWR", request_from_date_str="2026-09-01", request_to_date_str="2026-09-01",
            binding_obj_list=binding_list, imported_timestamp_str="2026-09-02T12:00:00+00:00",
        )
        if expanded_bool:
            from test_ibkr_nav_profile import expanded_nav_attributes_dict
            from test_client_reporting import xml_text_str
            expanded_row_list = [
                expanded_nav_attributes_dict("U100", date_str="2026-09-02", opening_str="1000", closing_str="1110",
                    depositsWithdrawals="100", mtm="12", commissions="-2", twr_str=".8"),
                expanded_nav_attributes_dict("U200", date_str="2026-09-02", opening_str="10000", closing_str="10050",
                    mtm="50", twr_str=".5"),
                expanded_nav_attributes_dict("U100", date_str="2026-09-03", opening_str="1110", closing_str="1105",
                    mtm="-5", twr_str="-.45045045045045"),
                expanded_nav_attributes_dict("U200", date_str="2026-09-03", opening_str="10050", closing_str="10030",
                    mtm="-20", twr_str="-.199004975124378"),
            ]
            PerformanceStore(str(performance_path_obj)).replace_range(
                xml_text_str=xml_text_str(expanded_row_list).replace('queryName="TEST_NAV"', 'queryName="ALPHA_DAILY_TWR"'),
                query_name_str="ALPHA_DAILY_TWR", request_from_date_str="2026-09-02", request_to_date_str="2026-09-03",
                binding_obj_list=binding_list, imported_timestamp_str="2026-09-04T12:00:00+00:00")
    provider_obj = DashboardDataProvider(releases_root_path_str=str(release_root_obj), config_path_str=str(config_path_obj),
        results_root_path_str=str(tmp_path / "results"), event_log_path_str=str(tmp_path / "events.jsonl"))
    return create_app(provider_obj, read_only_bool=True, performance_db_path_str=str(performance_path_obj),
        journal_path_str=str(tmp_path / "journal.jsonl"), notification_state_path_str=str(tmp_path / "notifications.json"))


def file_snapshot_dict(root_path_obj):
    return {str(path_obj.relative_to(root_path_obj)): (path_obj.read_bytes(), path_obj.stat().st_mtime_ns)
        for path_obj in root_path_obj.rglob("*") if path_obj.is_file()}


@pytest.mark.parametrize("finance_bool", [True, False])
def test_normal_local_entry_all_pages_no_registry_or_writes(tmp_path, monkeypatch, finance_bool):
    app_obj = build_fixture_app(tmp_path, monkeypatch, finance_bool=finance_bool)
    client_obj = app_obj.test_client()
    before_dict = file_snapshot_dict(tmp_path)
    response_obj = client_obj.get("/", follow_redirects=True)
    assert response_obj.status_code == 200
    assert response_obj.request.path == "/clients/local/overview"
    for view_str in VIEW_TUPLE:
        response_obj = client_obj.get(f"/clients/local/{view_str}")
        assert response_obj.status_code == 200
        html_str = response_obj.get_data(as_text=True)
        assert 'class="client-workspace"' in html_str
        assert "Switch client" not in html_str and ">Clients<" not in html_str
        assert "Set up" not in html_str and "Reporting setup needed" not in html_str
        assert len([label_str for label_str in VIEW_TUPLE if f"/clients/local/{label_str}" in html_str]) == 7
    status_dict = client_obj.get("/clients/local/diagnostics?download=json").json
    assert {row_dict["pod_id_str"] for row_dict in status_dict["strategy_list"]} == {"pod_a", "pod_b", "pod_new"}
    new_dict = next(row_dict for row_dict in status_dict["strategy_list"] if row_dict["pod_id_str"] == "pod_new")
    assert new_dict["matched_bool"] and new_dict["severity_str"] != "green"
    assert new_dict["evidence_dict"]["db_status_str"] != "ok"
    assert next(row_dict for row_dict in status_dict["strategy_list"] if row_dict["pod_id_str"] == "pod_a")["evidence_dict"]["lifecycle_step_dict_list"]
    assert client_obj.head("/clients/local/strategies").status_code == 200
    assert file_snapshot_dict(tmp_path) == before_dict


def test_known_account_twr_survives_new_pod_but_partial_book_never_becomes_total(tmp_path, monkeypatch):
    app_obj = build_fixture_app(tmp_path, monkeypatch)
    report_dict = app_obj.test_client().get("/clients/local/performance?from=2026-09-01&to=2026-09-01&download=json").json
    assert report_dict is not None
    assert {row_dict["pod_id_str"] for row_dict in report_dict["strategy_list"]} == {"pod_a", "pod_b", "pod_retired"}
    assert all(row_dict["twr_float"] == pytest.approx(.01) for row_dict in report_dict["strategy_list"])
    assert report_dict["scope_complete_bool"] is False
    assert report_dict["closing_nav_float"] is None and report_dict["opening_nav_float"] is None
    assert report_dict["twr_float"] is None and report_dict["pnl_float"] is None
    assert all(row_dict["nav_float"] is None for row_dict in report_dict["daily_book_list"])


def test_existing_old_flex_needs_no_new_financial_mapping(tmp_path, monkeypatch):
    app_obj = build_fixture_app(tmp_path, monkeypatch, new_pod_bool=False)
    report_dict = app_obj.test_client().get("/clients/local/performance?from=2026-09-01&to=2026-09-01&download=json").json
    assert report_dict["closing_nav_float"] == 3030
    assert report_dict["opening_nav_float"] == 3000
    assert report_dict["coverage_complete_bool"] is True
    assert report_dict["pnl_float"] is None and report_dict["twr_float"] is None
    assert all(row_dict["twr_float"] == pytest.approx(.01) for row_dict in report_dict["strategy_list"])


def test_corrupt_finance_does_not_hide_operations(tmp_path, monkeypatch):
    app_obj = build_fixture_app(tmp_path, monkeypatch, finance_bool=False)
    (tmp_path / "performance.sqlite3").write_bytes(b"not sqlite")
    before_dict = file_snapshot_dict(tmp_path)
    client_obj = app_obj.test_client()
    assert client_obj.get("/", follow_redirects=True).status_code == 200
    assert len(client_obj.get("/clients/local/diagnostics?download=json").json["strategy_list"]) == 3
    assert file_snapshot_dict(tmp_path) == before_dict


def test_empty_local_installation_is_navigable_and_not_green(tmp_path, monkeypatch):
    monkeypatch.delenv("ALPHA_CLIENT_REPORTING_CONFIG_PATH_STR", raising=False)
    provider_obj = DashboardDataProvider(releases_root_path_str=str(tmp_path / "releases"),
        config_path_str=str(tmp_path / "config.yaml"), results_root_path_str=str(tmp_path / "results"),
        event_log_path_str=str(tmp_path / "events.jsonl"))
    app_obj = create_app(provider_obj, read_only_bool=True, performance_db_path_str=str(tmp_path / "missing.sqlite3"))
    client_obj = app_obj.test_client()
    for view_str in VIEW_TUPLE:
        assert client_obj.get(f"/clients/local/{view_str}").status_code == 200
    status_dict = client_obj.get("/clients/local/diagnostics?download=json").json
    assert status_dict["strategy_list"] == [] and status_dict["severity_str"] != "green"
    assert list(tmp_path.iterdir()) == []


def test_local_operations_historical_dates_do_not_depend_on_finance(tmp_path, monkeypatch):
    app_obj = build_fixture_app(tmp_path, monkeypatch, finance_bool=False)
    # With no trusted financial dates at all, historical local logs still work.
    provider_obj = app_obj.config["data_provider_obj"]
    monkeypatch.setattr("alpha.live.dashboard_v3.local_workspace.build_live_binding_obj_list", lambda **kwargs: [])
    (tmp_path / "events.jsonl").write_text(
        '{"pod_id_str":"pod_new","account_route_str":"U300","mode_str":"live",'
        '"event_timestamp_str":"2026-09-01T15:00:00+00:00","event_name_str":"eod_snapshot_completed"}\n', encoding="utf-8")
    client_obj = app_obj.test_client()
    for view_str in ("strategies", "activity", "diagnostics", "exposure"):
        response_obj = client_obj.get(f"/clients/local/{view_str}?from=2026-09-01&to=2026-09-01")
        assert response_obj.status_code == 200
        if view_str == "activity":
            assert 'min="' not in response_obj.get_data(as_text=True)
            assert "End-of-day snapshot recorded" in response_obj.get_data(as_text=True)


def test_disabled_foreign_release_never_hides_local_pods_or_pools_money(tmp_path, monkeypatch):
    app_obj = build_fixture_app(tmp_path, monkeypatch, new_pod_bool=False)
    _write_release_manifest(tmp_path / "releases", user_id_str="other_owner", pod_id_str="foreign_disabled",
        account_route_str="U900", mode_str="live", enabled_bool=False)
    client_obj = app_obj.test_client()
    status_dict = client_obj.get("/clients/local/diagnostics?download=json").json
    assert {row_dict["pod_id_str"] for row_dict in status_dict["strategy_list"]} == {"pod_a", "pod_b"}
    report_dict = client_obj.get("/clients/local/performance?from=2026-09-01&to=2026-09-01&download=json").json
    assert report_dict["closing_nav_float"] == 3030
    assert "U900" not in str(report_dict)


def test_retired_reporting_endpoint_does_not_claim_client_cash_withdrawal(tmp_path, monkeypatch):
    app_obj = build_fixture_app(tmp_path, monkeypatch, new_pod_bool=False)
    report_dict = app_obj.test_client().get("/clients/local/performance?from=2026-09-01&to=2026-09-02&download=json").json
    assert report_dict["scope_complete_bool"] is True
    assert report_dict["coverage_complete_bool"] is False  # Active Sep2 rows are absent.
    assert report_dict["closing_nav_float"] is None
    assert report_dict["scope_movement_float"] is None
    retired_dict = next(row_dict for row_dict in report_dict["strategy_list"] if row_dict["pod_id_str"] == "pod_retired")
    assert retired_dict["twr_float"] == pytest.approx(.01)


def test_saved_only_retired_binding_is_not_lost(tmp_path, monkeypatch):
    app_obj = build_fixture_app(tmp_path, monkeypatch, new_pod_bool=False)
    (tmp_path / "releases" / "local_owner" / "pod_retired.yaml").unlink()
    workspace_dict = build_local_workspace_dict(app_obj.config["data_provider_obj"], app_obj.config["performance_db_path_str"], today_str="2026-09-06")
    assert {account_dict["pod_id"] for account_dict in workspace_dict["client_dict"]["accounts"]} == {"pod_a", "pod_b", "pod_retired"}


def test_conflicting_saved_identity_blocks_finance_not_current_operations(tmp_path, monkeypatch):
    app_obj = build_fixture_app(tmp_path, monkeypatch)
    with sqlite3.connect(tmp_path / "performance.sqlite3") as connection_obj:
        connection_obj.execute("UPDATE pod_binding SET pod_id_str='wrong_pod' WHERE account_route_str='U100'")
    client_obj = app_obj.test_client()
    assert len(client_obj.get("/clients/local/diagnostics?download=json").json["strategy_list"]) == 3
    assert client_obj.get("/clients/local/performance?download=json").status_code == 503


def test_remap_during_snapshot_read_rejects_export(tmp_path, monkeypatch):
    app_obj = build_fixture_app(tmp_path, monkeypatch, new_pod_bool=False)
    from alpha.live.dashboard_v3 import client_views
    original_fn = client_views.load_broker_reporting_snapshot
    def changed_snapshot_fn(*args, **kwargs):
        snapshot_obj = original_fn(*args, **kwargs)
        with sqlite3.connect(tmp_path / "performance.sqlite3") as connection_obj:
            connection_obj.execute("UPDATE pod_binding SET pod_id_str='concurrent_remap' WHERE account_route_str='U100'")
        return snapshot_obj
    monkeypatch.setattr(client_views, "load_broker_reporting_snapshot", changed_snapshot_fn)
    assert app_obj.test_client().get("/clients/local/performance?download=json").status_code == 503


def test_known_current_account_activity_precedes_first_performance_return(tmp_path, monkeypatch):
    app_obj = build_fixture_app(tmp_path, monkeypatch)
    (tmp_path / "events.jsonl").write_text(
        '{"pod_id_str":"pod_a","account_route_str":"U100","mode_str":"live",'
        '"event_timestamp_str":"2026-08-31T15:00:00+00:00","event_name_str":"eod_snapshot_completed"}\n', encoding="utf-8")
    response_obj = app_obj.test_client().get("/clients/local/activity?from=2026-08-31&to=2026-08-31")
    assert response_obj.status_code == 200
    assert "End-of-day snapshot recorded" in response_obj.get_data(as_text=True)


def test_foreign_disabled_identity_cannot_hide_account_remap(tmp_path, monkeypatch):
    app_obj = build_fixture_app(tmp_path, monkeypatch)
    with sqlite3.connect(tmp_path / "performance.sqlite3") as connection_obj:
        connection_obj.execute("UPDATE pod_binding SET pod_id_str='old_foreign_pod' WHERE account_route_str='U100'")
    _write_release_manifest(tmp_path / "releases", user_id_str="foreign_owner", pod_id_str="old_foreign_pod",
        account_route_str="U100", mode_str="live", enabled_bool=False)
    client_obj = app_obj.test_client()
    assert len(client_obj.get("/clients/local/diagnostics?download=json").json["strategy_list"]) == 3
    assert client_obj.get("/clients/local/performance?download=json").status_code == 503


def import_nav_day(app_obj, date_str, *, retired_closing_int=1020):
    database_path_str = app_obj.config["performance_db_path_str"]
    binding_list = _saved_binding_list(database_path_str)
    nav_tuple_list = [(binding_obj.account_route_str, date_str, 1010,
        retired_closing_int if binding_obj.account_route_str == "U400" else 1020, 1) for binding_obj in binding_list]
    # An ordinary import spanning the strategy baseline can contain earlier
    # raw NAV too; keep the existing importer contract unchanged.
    if date_str < "2026-09-01":
        nav_tuple_list += [(binding_obj.account_route_str, "2026-09-01", 1000, 1010, 1) for binding_obj in binding_list]
    PerformanceStore(database_path_str).replace_range(
        xml_text_str=_xml_str(nav_tuple_list),
        query_name_str="ALPHA_DAILY_TWR", request_from_date_str=date_str, request_to_date_str=max(date_str, "2026-09-01"),
        binding_obj_list=binding_list, imported_timestamp_str="2026-09-04T12:00:00+00:00")


def test_local_raw_nav_before_measured_start_is_not_a_portfolio_period(tmp_path, monkeypatch):
    app_obj = build_fixture_app(tmp_path, monkeypatch, new_pod_bool=False)
    import_nav_day(app_obj, "2026-08-31")
    before_dict = file_snapshot_dict(tmp_path)
    client_obj = app_obj.test_client()
    assert client_obj.get("/clients/local/report?from=2026-08-31&to=2026-08-31&download=json").status_code == 400
    assert client_obj.get("/clients/local/activity?from=2026-08-31&to=2026-08-31").status_code == 200
    assert file_snapshot_dict(tmp_path) == before_dict


def test_local_measured_nav_excludes_retired_account_on_screen_and_pdf(tmp_path, monkeypatch):
    date_str = "2026-09-02"
    app_obj = build_fixture_app(tmp_path, monkeypatch, new_pod_bool=False)
    import_nav_day(app_obj, date_str)
    client_obj = app_obj.test_client()
    route_str = f"/clients/local/report?from={date_str}&to={date_str}"
    before_dict = file_snapshot_dict(tmp_path)
    report_dict = client_obj.get(route_str + "&download=json").json
    assert report_dict["opening_nav_float"] == 2020 and report_dict["closing_nav_float"] == 2040
    assert report_dict["scope_complete_bool"] is True and report_dict["twr_float"] is None
    assert report_dict["pnl_float"] is None
    html_str = client_obj.get(route_str).get_data(as_text=True)
    assert "$2,040.00" in html_str and "$3,060.00" not in html_str
    pdf_response_obj = client_obj.get(route_str + "&download=pdf&expected=" + report_dict["report_hash_str"])
    assert pdf_response_obj.status_code == 200
    pdf_text_str = " ".join(page_obj.extract_text() for page_obj in PdfReader(BytesIO(pdf_response_obj.data)).pages)
    assert "2,040" in pdf_text_str and "DRAFT" in pdf_text_str
    assert file_snapshot_dict(tmp_path) == before_dict


def test_local_all_unknown_windows_still_show_raw_nav_and_earlier_dates(tmp_path, monkeypatch):
    app_obj = build_fixture_app(tmp_path, monkeypatch, new_pod_bool=False)
    import_nav_day(app_obj, "2026-08-31")
    with sqlite3.connect(tmp_path / "performance.sqlite3") as connection_obj:
        connection_obj.execute("UPDATE pod_binding SET return_start_date_str=NULL, return_end_date_str=NULL")
    monkeypatch.setattr("alpha.live.dashboard_v3.local_workspace.build_live_binding_obj_list", lambda **kwargs: [])
    client_obj = app_obj.test_client()
    route_str = "/clients/local/overview?from=2026-08-31&to=2026-08-31"
    report_dict = client_obj.get(route_str + "&download=json").json
    assert report_dict["closing_nav_float"] == 3060 and report_dict["strategy_list"] == []
    assert report_dict["status_str"] == "draft" and report_dict["twr_float"] is None
    html_str = client_obj.get(route_str).get_data(as_text=True)
    assert "$3,060.00" in html_str and 'min="2026-08-31"' in html_str
    assert "no verified strategy performance window" in html_str


@pytest.mark.parametrize("failure_str", ["missing", "corrupt", "mapping", "window"])
def test_local_source_errors_keep_historical_overview_operations(tmp_path, monkeypatch, failure_str):
    app_obj = build_fixture_app(tmp_path, monkeypatch, new_pod_bool=False)
    import_nav_day(app_obj, "2026-08-31")
    database_path_obj = tmp_path / "performance.sqlite3"
    if failure_str == "missing":
        database_path_obj.unlink()
        expected_str = "missing"
    elif failure_str == "corrupt":
        database_path_obj.write_bytes(b"invalid sqlite")
        expected_str = "Check the performance database and local ledger files"
    else:
        with sqlite3.connect(database_path_obj) as connection_obj:
            connection_obj.execute("UPDATE pod_binding SET " + ("pod_id_str='wrong_pod'" if failure_str == "mapping" else "return_start_date_str='2026-08-30'") + " WHERE account_route_str='U100'")
        expected_str = "Account/Pod mapping changed" if failure_str == "mapping" else "saved history start"
    before_dict = file_snapshot_dict(tmp_path)
    client_obj = app_obj.test_client()
    route_str = "/clients/local/overview?from=2026-08-31&to=2026-08-31"
    response_obj = client_obj.get(route_str)
    assert response_obj.status_code == 200
    html_str = response_obj.get_data(as_text=True)
    assert expected_str.lower() in html_str.lower()
    assert 'aria-label="Current client operations"' in html_str
    assert client_obj.get(route_str + "&download=json").status_code == 503
    assert str(tmp_path) not in html_str and "invalid sqlite" not in html_str
    assert file_snapshot_dict(tmp_path) == before_dict


def test_nav_only_revision_invalidates_previously_viewed_export(tmp_path, monkeypatch):
    app_obj = build_fixture_app(tmp_path, monkeypatch, new_pod_bool=False)
    import_nav_day(app_obj, "2026-09-02")
    client_obj = app_obj.test_client()
    route_str = "/clients/local/report?from=2026-09-02&to=2026-09-02"
    original_dict = client_obj.get(route_str + "&download=json").json
    import_nav_day(app_obj, "2026-09-02", retired_closing_int=1030)
    corrected_dict = client_obj.get(route_str + "&download=json").json
    assert corrected_dict["closing_nav_float"] == original_dict["closing_nav_float"] == 2040
    assert corrected_dict["report_hash_str"] != original_dict["report_hash_str"]
    for export_str in ("pdf", "bundle"):
        assert client_obj.get(route_str + f"&download={export_str}&expected=" + original_dict["report_hash_str"]).status_code == 409
