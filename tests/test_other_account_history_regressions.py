"""Real release/ledger transitions must preserve already owned client money."""

from contextlib import closing
from datetime import UTC, datetime
import sqlite3
from xml.etree import ElementTree

import pytest

from alpha.live.client_reporting import ClientReportingError, load_broker_reporting_snapshot
from alpha.live.dashboard_v4.data import load_workspace_snapshot_tuple
from alpha.live.dashboard_v4.performance import build_performance_page_dict
from alpha.live.ibkr_performance import PerformanceStore
from alpha.live.ibkr_performance_sync import build_live_binding_obj_list
from alpha.live.release_manifest import load_release_list
from alpha.live.reporting_accounts import discover_other_account_list
from test_live_dashboard import (
    _mark_latest_vplan_completed_and_reconciled, _seed_decision_vplan_and_broker_rows,
    _seed_eod_pod_state, _write_config, _write_release_manifest,
)
from test_other_accounts_discovery import _statement_str, _store_path_str
from test_other_accounts_workspace import AS_OF_TS, _reserve_fixture


def _latest_xml_str(database_path_str):
    with closing(sqlite3.connect(database_path_str)) as connection_obj:
        return connection_obj.execute("SELECT raw_xml_str FROM flex_import ORDER BY import_id_int DESC LIMIT 1").fetchone()[0]


def _legacy_without_import_roles(database_path_str):
    """A pre-feature DB needs no manual migration before normal reads/sync."""
    with closing(sqlite3.connect(database_path_str)) as connection_obj, connection_obj:
        column_set = {row_tuple[1] for row_tuple in connection_obj.execute("PRAGMA table_info(flex_import)")}
        if "strategy_account_routes_json_str" in column_set:
            connection_obj.execute("ALTER TABLE flex_import DROP COLUMN strategy_account_routes_json_str")


def _install_strategy_on_reserve(tmp_path, config_path_obj):
    """Use actual YAML and broker EOD ledger, not assembled reporting periods."""
    _write_release_manifest(tmp_path / "releases", user_id_str="local_owner",
        pod_id_str="pod_reserve", mode_str="live", account_route_str="URESERVE", enabled_bool=True)
    _write_config(config_path_obj, {pod_str: str(tmp_path / (pod_str + ".sqlite3"))
        for pod_str in ("pod_a", "pod_b", "pod_reserve")})
    release_obj = next(release_obj for release_obj in load_release_list(str(tmp_path / "releases"))
        if release_obj.pod_id_str == "pod_reserve")
    pod_path_obj = tmp_path / "pod_reserve.sqlite3"
    _seed_decision_vplan_and_broker_rows(pod_path_obj, release_obj)
    _mark_latest_vplan_completed_and_reconciled(pod_path_obj, release_obj)
    _seed_eod_pod_state(pod_path_obj, release_obj, total_value_float=5000,
        updated_timestamp_ts=datetime(2026, 9, 1, 20, 10, tzinfo=UTC))
    return build_live_binding_obj_list(releases_root_path_str=str(tmp_path / "releases"),
        config_path_str=str(config_path_obj))


def _correct_import(database_path_str, raw_xml_str, binding_list):
    # A genuine replacement source, while every financial fact stays identical.
    root_obj = ElementTree.fromstring(raw_xml_str)
    for statement_obj in root_obj.findall(".//FlexStatement"):
        if statement_obj.get("accountId") == "URESERVE":
            statement_obj.find("AccountInformation").set("acctAlias", "Now a strategy")
    PerformanceStore(database_path_str).replace_range(
        xml_text_str=ElementTree.tostring(root_obj, encoding="unicode"), query_name_str="ALPHA_DAILY_TWR",
        request_from_date_str="2026-09-01", request_to_date_str="2026-09-03", binding_obj_list=binding_list)


def _workspace_report_tuple(provider_obj, database_path_str):
    workspace_dict, snapshot_obj = load_workspace_snapshot_tuple(provider_obj, database_path_str, as_of_ts=AS_OF_TS)
    report_dict = build_performance_page_dict(workspace_dict, snapshot_obj, as_of_ts=AS_OF_TS,
        from_date_str="2026-09-01", to_date_str="2026-09-03")["report_dict"]
    return workspace_dict, report_dict


@pytest.mark.parametrize("legacy_bool", [False, True])
def test_actual_other_to_pod_handover_keeps_baseline_day_client_capital(tmp_path, monkeypatch, legacy_bool):
    provider_obj, database_path_str, config_path_obj = _reserve_fixture(tmp_path, monkeypatch, interest_float=2.0)
    before_workspace_dict, before_report_dict = _workspace_report_tuple(provider_obj, database_path_str)
    assert before_report_dict["opening_nav_float"] == 16000
    assert before_report_dict["closing_nav_float"] == 16182
    assert before_report_dict["pnl_float"] == 182
    assert before_report_dict["twr_float"] == pytest.approx(182 / 16000)
    assert [account_dict["account_route"] for account_dict in before_workspace_dict["other_account_list"]] == ["URESERVE"]
    raw_xml_str = _latest_xml_str(database_path_str)
    if legacy_bool:
        _legacy_without_import_roles(database_path_str)
    binding_list = _install_strategy_on_reserve(tmp_path, config_path_obj)
    for sync_bool in (False, True):
        if sync_bool:
            _correct_import(database_path_str, raw_xml_str, binding_list)
        workspace_dict, report_dict = _workspace_report_tuple(provider_obj, database_path_str)
        prior_period_list = [account_dict for account_dict in workspace_dict["other_account_list"]
            if account_dict["account_route"] == "URESERVE"]
        assert [(account_dict["effective_from"], account_dict["effective_to"]) for account_dict in prior_period_list] == [
            ("2026-09-01", "2026-09-01")]
        strategy_dict, = [account_dict for account_dict in workspace_dict["client_dict"]["accounts"]
            if account_dict.get("pod_id") == "pod_reserve"]
        assert strategy_dict["effective_from"] == "2026-09-02"
        for field_str in ("opening_nav_float", "closing_nav_float", "pnl_float", "twr_float", "scope_movement_float"):
            assert report_dict[field_str] == pytest.approx(before_report_dict[field_str]), (sync_bool, field_str)
        assert report_dict["status_str"] == "ready"
        assert report_dict["coverage_complete_bool"]


@pytest.mark.parametrize("legacy_bool", [False, True])
def test_account_already_a_strategy_at_first_import_keeps_its_normal_baseline(tmp_path, monkeypatch, legacy_bool):
    provider_obj, source_path_str, config_path_obj = _reserve_fixture(tmp_path, monkeypatch)
    raw_xml_str = _latest_xml_str(source_path_str)
    binding_list = _install_strategy_on_reserve(tmp_path, config_path_obj)
    # This separate store has the reserve account mapped as a strategy from its
    # very first import. There is no earlier Other-account admission in this DB.
    database_path_str = str(tmp_path / "already_strategy.sqlite3")
    PerformanceStore(database_path_str).replace_range(xml_text_str=raw_xml_str,
        query_name_str="ALPHA_DAILY_TWR", request_from_date_str="2026-09-01",
        request_to_date_str="2026-09-03", binding_obj_list=binding_list)
    workspace_dict, _ = _workspace_report_tuple(provider_obj, database_path_str)
    assert workspace_dict["other_account_list"] == []
    assert next(account_dict for account_dict in workspace_dict["client_dict"]["accounts"]
        if account_dict.get("pod_id") == "pod_reserve")["effective_from"] == "2026-09-02"
    if legacy_bool:
        _legacy_without_import_roles(database_path_str)
    _correct_import(database_path_str, raw_xml_str, binding_list)
    workspace_dict, _ = _workspace_report_tuple(provider_obj, database_path_str)
    assert workspace_dict["other_account_list"] == []
    with closing(sqlite3.connect(database_path_str)) as connection_obj:
        assert connection_obj.execute("SELECT MIN(market_date_str) FROM daily_performance WHERE account_route_str = 'URESERVE'").fetchone()[0] == "2026-09-02"


@pytest.mark.parametrize("missing_field_tuple", [
    ("startingValue",), ("endingValue",), ("startingValue", "endingValue"), ("twr",),
])
def test_missing_required_facts_cannot_prove_a_new_other_account_was_empty(tmp_path, missing_field_tuple):
    statement_str = _statement_str("U800", [("20260807", 0, 0), ("20260810", 0, 0)])
    for field_str in missing_field_tuple:
        statement_str = statement_str.replace(f'{field_str}="0"', "")
    database_path_str = _store_path_str(tmp_path / "performance.sqlite3", statement_str)
    with pytest.raises(ClientReportingError):
        other_account_list = discover_other_account_list(database_path_str,
            query_name_str="ALPHA_DAILY_TWR", strategy_start_by_route_dict={"U100": "2026-08-07"})
        load_broker_reporting_snapshot(database_path_str, query_name_str="ALPHA_DAILY_TWR",
            allowed_account_set={"U100", *(account_dict["account_route"] for account_dict in other_account_list)})
