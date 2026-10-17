"""IBKR account aliases on Live Ops pages, read from the saved Flex report."""

from contextlib import closing
import sqlite3

from alpha.live.dashboard_v4.account_alias import (
    account_label_str, load_alias_by_pod_dict, parse_alias_by_account_dict)
from alpha.live.dashboard_v4.demo import create_demo_app
from alpha.live.ibkr_performance import PerformanceStore


def _flex_xml_str(statement_list):
    statement_xml_str = "".join(
        f'<FlexStatement accountId="{account_str}"><AccountInformation accountId="{account_str}" '
        f'currency="USD" {attribute_str} /></FlexStatement>'
        for account_str, attribute_str in statement_list)
    return (f'<FlexQueryResponse queryName="q"><FlexStatements count="{len(statement_list)}">'
        f'{statement_xml_str}</FlexStatements></FlexQueryResponse>')


def _write_performance_db(path_obj, xml_text_str, binding_list):
    PerformanceStore(str(path_obj)).initialize()
    with closing(sqlite3.connect(path_obj)) as connection_obj, connection_obj:
        connection_obj.execute(
            "INSERT INTO flex_import (imported_timestamp_str, request_from_date_str, request_to_date_str,"
            " query_name_str, checksum_str, raw_xml_str) VALUES ('t', 'a', 'b', 'q', ?, ?)",
            (str(hash(xml_text_str)), xml_text_str))
        connection_obj.executemany(
            "INSERT INTO pod_binding (account_route_str, pod_id_str, return_start_date_str, return_end_date_str,"
            " enabled_bool_int, session_calendar_id_str, updated_timestamp_str) VALUES (?, ?, NULL, NULL, 1, 'XNYS', 't')",
            binding_list)


def test_parse_reads_account_information_alias_and_skips_accounts_without_one():
    xml_text_str = _flex_xml_str([
        ("U1", 'acctAlias="  growth_mr_dv2 "'),
        ("U2", 'accountAlias="defensive_core5"'),
        ("U3", 'acctAlias=""'),
        ("U4", ""),
    ])
    assert parse_alias_by_account_dict(xml_text_str) == {"U1": "growth_mr_dv2", "U2": "defensive_core5"}


def test_parse_falls_back_to_change_in_nav_and_bounds_length():
    long_alias_str = "x" * 80
    xml_text_str = ('<FlexQueryResponse><FlexStatements count="1"><FlexStatement accountId="U1">'
        '<AccountInformation accountId="U1" currency="USD" />'
        f'<ChangeInNAV accountId="U1" acctAlias="{long_alias_str}" /></FlexStatement></FlexStatements></FlexQueryResponse>')
    assert parse_alias_by_account_dict(xml_text_str) == {"U1": "x" * 40}


def test_load_maps_alias_to_pod_and_never_writes(tmp_path):
    db_path_obj = tmp_path / "performance.sqlite3"
    _write_performance_db(db_path_obj, _flex_xml_str([("U1", 'acctAlias="growth_mr_dv2"'), ("U2", "")]),
        [("U1", "pod_growth_mr_dv2_01"), ("U2", "pod_defensive_core5_01")])
    before_bytes, before_mtime_int = db_path_obj.read_bytes(), db_path_obj.stat().st_mtime_ns
    assert load_alias_by_pod_dict(str(db_path_obj)) == {"pod_growth_mr_dv2_01": "growth_mr_dv2"}
    assert db_path_obj.read_bytes() == before_bytes
    assert db_path_obj.stat().st_mtime_ns == before_mtime_int


def test_load_is_empty_without_database_import_or_valid_xml(tmp_path):
    assert load_alias_by_pod_dict(None) == {}
    assert load_alias_by_pod_dict(str(tmp_path / "missing.sqlite3")) == {}
    empty_path_obj = tmp_path / "empty.sqlite3"
    PerformanceStore(str(empty_path_obj)).initialize()
    assert load_alias_by_pod_dict(str(empty_path_obj)) == {}
    broken_path_obj = tmp_path / "broken.sqlite3"
    _write_performance_db(broken_path_obj, "<FlexQueryResponse>", [("U1", "pod")])
    assert load_alias_by_pod_dict(str(broken_path_obj)) == {}


def test_label_shows_alias_before_masked_account():
    assert account_label_str("growth_mr_dv2", "U···795") == "growth_mr_dv2 · U···795"
    assert account_label_str("", "U···795") == "U···795"


def test_pod_system_and_tools_pages_show_the_alias(tmp_path, monkeypatch):
    aliased_pod_str, plain_pod_str = "demo_1_0", "demo_1_1"
    db_path_obj = tmp_path / "performance.sqlite3"
    _write_performance_db(db_path_obj, _flex_xml_str([("DEMO_1_0", 'acctAlias="growth_mr_dv2"'), ("DEMO_1_1", "")]),
        [("DEMO_1_0", aliased_pod_str), ("DEMO_1_1", plain_pod_str)])
    monkeypatch.setenv("ALPHA_IBKR_PERFORMANCE_DB_PATH_STR", str(db_path_obj))
    app_obj = create_demo_app()
    client_obj = app_obj.test_client()
    aliased_html_str = client_obj.get(f"/pods/{aliased_pod_str}").get_data(as_text=True)
    assert "growth_mr_dv2 · D···1_0" in aliased_html_str
    plain_html_str = client_obj.get(f"/pods/{plain_pod_str}").get_data(as_text=True)
    assert "growth_mr_dv2" not in plain_html_str and "D···1_1" in plain_html_str
    assert "growth_mr_dv2 · " in client_obj.get(f"/tools?pod={aliased_pod_str}").get_data(as_text=True)
    # The demo System page has no verified release table; check the shared label and its use there.
    assert app_obj.jinja_env.from_string("{{ pod_account_label(p, '•••1_0') }}").render(
        p=aliased_pod_str) == "growth_mr_dv2 · •••1_0"
    system_html_str = client_obj.get("/system").get_data(as_text=True)
    assert '<span class="system-account-alias">growth_mr_dv2</span>D···1_0' in system_html_str
    assert 'data-label="Account">D···1_1</td>' in system_html_str


def test_pages_are_unchanged_when_the_alias_source_is_missing_or_corrupt(tmp_path, monkeypatch):
    corrupt_path_obj = tmp_path / "corrupt.sqlite3"
    corrupt_path_obj.write_bytes(b"not a database")
    html_by_source_dict = {}
    for name_str, db_path_str in (("missing", str(tmp_path / "missing.sqlite3")), ("corrupt", str(corrupt_path_obj))):
        monkeypatch.setenv("ALPHA_IBKR_PERFORMANCE_DB_PATH_STR", db_path_str)
        client_obj = create_demo_app().test_client()
        for path_str in ("/pods/demo_1_0", "/system", "/tools?pod=demo_1_0"):
            response_obj = client_obj.get(path_str)
            assert response_obj.status_code == 200
            html_by_source_dict[(name_str, path_str)] = response_obj.get_data(as_text=True)
        assert "demo_1_0 · D···1_0" in html_by_source_dict[(name_str, "/pods/demo_1_0")]
        assert 'data-label="Account">D···1_0</td>' in html_by_source_dict[(name_str, "/system")]
        assert "system-account-alias" not in html_by_source_dict[(name_str, "/system")]
