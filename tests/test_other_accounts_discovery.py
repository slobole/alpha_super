"""The client's other accounts are found in the saved Flex, never configured."""

from pathlib import Path

import pytest

from alpha.live.client_reporting import ClientReportingError
from alpha.live.ibkr_performance import PerformanceStore, PodPerformanceBinding
from alpha.live.reporting_accounts import discover_other_account_list


def _binding_list():
    return [PodPerformanceBinding("U100", "pod_strategy", "2026-08-07", None, True)]


def _statement_str(route_str, row_list, *, alias_str="", currency_str="USD"):
    alias_attribute_str = f' acctAlias="{alias_str}"' if alias_str else ""
    return (f'<FlexStatement accountId="{route_str}"><AccountInformation accountId="{route_str}" currency="{currency_str}"{alias_attribute_str} />'
        + "".join(f'<ChangeInNAV accountId="{route_str}" currency="{currency_str}" fromDate="{date_str}" toDate="{date_str}" '
            f'startingValue="{opening_int}" endingValue="{closing_int}" twr="0" />' for date_str, opening_int, closing_int in row_list)
        + "</FlexStatement>")


def _store_path_str(db_path_obj, *other_statement_str_tuple):
    statement_list = [_statement_str("U100", [("20260807", 100, 100), ("20260810", 100, 100)]), *other_statement_str_tuple]
    xml_str = ('<FlexQueryResponse queryName="ALPHA_DAILY_TWR"><FlexStatements count="' + str(len(statement_list)) + '">'
        + "".join(statement_list) + "</FlexStatements></FlexQueryResponse>")
    PerformanceStore(str(db_path_obj)).replace_range(xml_text_str=xml_str, query_name_str="ALPHA_DAILY_TWR",
        request_from_date_str="2026-08-07", request_to_date_str="2026-08-10", binding_obj_list=_binding_list())
    return str(db_path_obj)


def _discover_list(db_path_str):
    return discover_other_account_list(db_path_str, query_name_str="ALPHA_DAILY_TWR", strategy_account_route_set={"U100"})


def test_vault_is_found_with_its_alias_and_first_day(tmp_path):
    db_path_str = _store_path_str(tmp_path / "performance.sqlite3", _statement_str("U28444262", [("20260807", 500, 500), ("20260810", 500, 501)], alias_str="VAULT"))
    assert _discover_list(db_path_str) == [{"role": "reserve", "account_route": "U28444262", "display_name": "VAULT",
        "effective_from": "2026-08-07", "effective_to": None}]


def test_unnamed_account_shows_only_its_last_four_characters(tmp_path):
    db_path_str = _store_path_str(tmp_path / "performance.sqlite3", _statement_str("U28444262", [("20260807", 500, 500), ("20260810", 500, 500)]))
    assert _discover_list(db_path_str)[0]["display_name"] == "Other account …4262"


def test_empty_account_is_left_out_until_it_holds_money(tmp_path):
    db_path_str = _store_path_str(tmp_path / "performance.sqlite3", _statement_str("U800", [("20260807", 0, 0), ("20260810", 0, 0)]))
    assert _discover_list(db_path_str) == []
    funded_path_str = _store_path_str(tmp_path / "funded.sqlite3",
        _statement_str("U800", [("20260807", 0, 0), ("20260810", 0, 250)]))
    assert [(account_dict["account_route"], account_dict["effective_from"]) for account_dict in _discover_list(funded_path_str)] == [
        ("U800", "2026-08-10")]


def test_strategy_accounts_are_never_other_accounts(tmp_path):
    db_path_str = _store_path_str(tmp_path / "performance.sqlite3", _statement_str("U900", [("20260807", 5, 5), ("20260810", 5, 5)]))
    assert discover_other_account_list(db_path_str, query_name_str="ALPHA_DAILY_TWR",
        strategy_account_route_set={"U100", "U900"}) == []


def test_non_usd_account_is_reported_by_name_not_dropped(tmp_path):
    db_path_str = _store_path_str(tmp_path / "performance.sqlite3", _statement_str("U28444262", [("20260807", 5, 5), ("20260810", 5, 5)],
        alias_str="VAULT", currency_str="EUR"))
    with pytest.raises(ClientReportingError, match="VAULT: only USD accounts"):
        _discover_list(db_path_str)


def test_missing_database_and_reads_never_write(tmp_path):
    assert _discover_list(str(tmp_path / "missing.sqlite3")) == []
    assert not (tmp_path / "missing.sqlite3").exists()
    db_path_str = _store_path_str(tmp_path / "performance.sqlite3", _statement_str("U900", [("20260807", 5, 5), ("20260810", 5, 5)]))
    before_tuple = (Path(db_path_str).read_bytes(), Path(db_path_str).stat().st_mtime_ns)
    assert len(_discover_list(db_path_str)) == 1
    assert (Path(db_path_str).read_bytes(), Path(db_path_str).stat().st_mtime_ns) == before_tuple
