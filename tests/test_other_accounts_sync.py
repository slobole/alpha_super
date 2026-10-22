"""A Flex account without a strategy never blocks the Pod sync or gets a Pod row."""

import hashlib
import sqlite3

import pytest

from alpha.live.ibkr_performance import PerformanceContractError, PerformanceStore, PodPerformanceBinding, parse_flex_change_in_nav_xml
from alpha.live.ibkr_performance_sync import bootstrap_remote, import_xml_bool, sync_range_bool


def _binding_list():
    return [PodPerformanceBinding("U100", "pod_strategy", "2026-08-07", None, True)]


def _statement_str(route_str, row_list):
    return f'<FlexStatement accountId="{route_str}"><AccountInformation accountId="{route_str}" currency="USD" />' + "".join(
        f'<ChangeInNAV accountId="{route_str}" currency="USD" fromDate="{date_str}" toDate="{date_str}" startingValue="{opening_int}" endingValue="{closing_int}" twr="{return_float}" />'
        for date_str, opening_int, closing_int, return_float in row_list) + "</FlexStatement>"


def _xml_str(*, other_bool=True, pod_bool=True):
    statement_list = []
    if pod_bool:
        statement_list.append(_statement_str("U100", [("20260807", 100, 110, 10)]))
    if other_bool:
        # The vault, an unused empty sub-account, and one in another currency:
        # none of them is the Pod sync's business.
        statement_list.append(_statement_str("U900", [("20260807", 900, 900, 0)]))
        statement_list.append(_statement_str("U800", [("20260807", 0, 0, 0)]))
        statement_list.append(_statement_str("U700", [("20260807", 5, 5, 0)]).replace('currency="USD"', 'currency="EUR"'))
    return ('<FlexQueryResponse queryName="ALPHA_DAILY_TWR"><FlexStatements count="' + str(len(statement_list)) + '">'
        + "".join(statement_list) + "</FlexStatements></FlexQueryResponse>")


def _parse_list(xml_str):
    return parse_flex_change_in_nav_xml(xml_str, {binding_obj.account_route_str: binding_obj for binding_obj in _binding_list()},
        request_from_date_str="2026-08-07", request_to_date_str="2026-08-07", expected_query_name_str="ALPHA_DAILY_TWR")


def test_other_accounts_leave_pod_rows_unchanged():
    row_list = _parse_list(_xml_str())
    assert row_list == _parse_list(_xml_str(other_bool=False))
    assert len(row_list) == 1 and row_list[0].pod_id_str == "pod_strategy"


def test_missing_strategy_account_still_fails():
    with pytest.raises(PerformanceContractError, match="missing configured account"):
        _parse_list(_xml_str(pod_bool=False))


def test_import_keeps_raw_xml_but_only_strategy_rows(tmp_path):
    store_obj = PerformanceStore(str(tmp_path / "performance.sqlite3"))
    xml_str = _xml_str()
    assert store_obj.replace_range(xml_text_str=xml_str, query_name_str="ALPHA_DAILY_TWR",
        request_from_date_str="2026-08-07", request_to_date_str="2026-08-07", binding_obj_list=_binding_list())
    with sqlite3.connect(store_obj.db_path_obj) as connection_obj:
        assert connection_obj.execute("SELECT account_route_str, pod_id_str FROM pod_binding").fetchall() == [("U100", "pod_strategy")]
        assert connection_obj.execute("SELECT account_route_str FROM daily_performance").fetchall() == [("U100",)]
        assert connection_obj.execute("SELECT raw_xml_str, checksum_str FROM flex_import").fetchone() == (
            xml_str, hashlib.sha256(xml_str.encode()).hexdigest())


def test_bootstrap_sync_and_manual_import_accept_other_accounts(tmp_path, monkeypatch):
    monkeypatch.setattr("alpha.live.ibkr_performance_sync.fetch_flex_statement_xml_str", lambda **kwargs: _xml_str())
    command_kwargs = dict(binding_obj_list=_binding_list(), from_date_str="2026-08-07", to_date_str="2026-08-07",
        query_name_str="ALPHA_DAILY_TWR", token_str="test-token", query_id_str="test-query")
    bootstrap_remote(db_path_str=str(tmp_path / "bootstrap.sqlite3"), replace_bool=False, **command_kwargs)
    assert sync_range_bool(db_path_str=str(tmp_path / "sync.sqlite3"), **command_kwargs)
    xml_path_obj = tmp_path / "source.xml"
    xml_path_obj.write_text(_xml_str(), encoding="utf-8")
    assert import_xml_bool(db_path_str=str(tmp_path / "import.sqlite3"), xml_path_str=str(xml_path_obj),
        binding_obj_list=_binding_list(), query_name_str="ALPHA_DAILY_TWR")
    for name_str in ("bootstrap", "sync", "import"):
        with sqlite3.connect(tmp_path / f"{name_str}.sqlite3") as connection_obj:
            assert connection_obj.execute("SELECT account_route_str FROM daily_performance").fetchall() == [("U100",)]
