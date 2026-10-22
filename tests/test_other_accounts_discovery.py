"""The client's other accounts are found in the saved Flex, never configured."""

from pathlib import Path

import pytest

from alpha.live.client_reporting import ClientReportingError
from alpha.live.ibkr_performance import PerformanceStore, PodPerformanceBinding
from alpha.live.reporting_accounts import discover_other_account_list


def _binding_list():
    return [PodPerformanceBinding("U100", "pod_strategy", "2026-08-07", None, True)]


def _statement_str(route_str, row_list, *, alias_str="", currency_str="USD", extra_str=""):
    alias_attribute_str = f' acctAlias="{alias_str}"' if alias_str else ""
    return (f'<FlexStatement accountId="{route_str}"><AccountInformation accountId="{route_str}" currency="{currency_str}"{alias_attribute_str} />'
        + "".join(f'<ChangeInNAV accountId="{route_str}" currency="{currency_str}" fromDate="{date_str}" toDate="{date_str}" '
            f'startingValue="{opening_int}" endingValue="{closing_int}" twr="0"{extra_str} />' for date_str, opening_int, closing_int in row_list)
        + "</FlexStatement>")


def _store_path_str(db_path_obj, *other_statement_str_tuple):
    statement_list = [_statement_str("U100", [("20260807", 100, 100), ("20260810", 100, 100)]), *other_statement_str_tuple]
    xml_str = ('<FlexQueryResponse queryName="ALPHA_DAILY_TWR"><FlexStatements count="' + str(len(statement_list)) + '">'
        + "".join(statement_list) + "</FlexStatements></FlexQueryResponse>")
    PerformanceStore(str(db_path_obj)).replace_range(xml_text_str=xml_str, query_name_str="ALPHA_DAILY_TWR",
        request_from_date_str="2026-08-07", request_to_date_str="2026-08-10", binding_obj_list=_binding_list())
    return str(db_path_obj)


def _discover_list(db_path_str, **scope_kwargs):
    return discover_other_account_list(db_path_str, query_name_str="ALPHA_DAILY_TWR",
        **({"strategy_start_by_route_dict": {"U100": "2026-08-07"}} | scope_kwargs))


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


@pytest.mark.parametrize("extra_str", [' depositsWithdrawals="100" commissions="-100"', ' interest="0.01"', ' mtm="abc"'])
def test_money_moving_through_a_zero_balance_account_is_activity(tmp_path, extra_str):
    db_path_str = _store_path_str(tmp_path / "performance.sqlite3",
        _statement_str("U800", [("20260807", 0, 0)], extra_str=extra_str), _statement_str("U801", [("20260810", 0, 0)]))
    assert [(account_dict["account_route"], account_dict["effective_from"]) for account_dict in _discover_list(db_path_str)] == [
        ("U800", "2026-08-07")]


def test_a_correction_that_drops_the_account_keeps_its_membership(tmp_path):
    db_path_obj = tmp_path / "performance.sqlite3"
    _store_path_str(db_path_obj, _statement_str("U900", [("20260807", 5000, 5000), ("20260810", 5000, 5000)]))
    # The replacement covers the whole range and omits the account entirely:
    # the account stays a member and its days become missing, never zero.
    _store_path_str(db_path_obj)
    assert [(account_dict["account_route"], account_dict["effective_from"]) for account_dict in _discover_list(str(db_path_obj))] == [
        ("U900", "2026-08-07")]


def test_strategy_account_is_an_other_account_only_before_its_first_pod_day(tmp_path):
    db_path_str = _store_path_str(tmp_path / "performance.sqlite3", _statement_str("U900", [("20260807", 5, 5), ("20260810", 5, 5)]))
    start_dict = {"U100": "2026-08-07"}
    # Active on 08-07, before the baseline (08-10) of a Pod starting 08-11.
    assert _discover_list(db_path_str, strategy_start_by_route_dict=start_dict | {"U900": "2026-08-11"}) == [
        {"role": "reserve", "account_route": "U900", "display_name": "Other account …U900",
            "effective_from": "2026-08-07", "effective_to": "2026-08-10"}]
    # A Pod starting 08-10 has its baseline on 08-07: an ordinary Pod, as before.
    assert _discover_list(db_path_str, strategy_start_by_route_dict=start_dict | {"U900": "2026-08-10"}) == []
    assert _discover_list(db_path_str, strategy_start_by_route_dict=start_dict | {"U900": "2026-08-07"}) == []
    assert _discover_list(db_path_str, strategy_start_by_route_dict=start_dict | {"U900": None}) == []
    assert _discover_list(db_path_str, excluded_route_set={"U900"}) == []


def test_active_non_usd_account_is_reported_by_name_not_dropped(tmp_path):
    db_path_str = _store_path_str(tmp_path / "performance.sqlite3", _statement_str("U28444262", [("20260807", 5, 5), ("20260810", 5, 5)],
        alias_str="VAULT", currency_str="EUR"))
    with pytest.raises(ClientReportingError, match="VAULT: only USD accounts"):
        _discover_list(db_path_str)


def test_empty_non_usd_account_is_ignored(tmp_path):
    db_path_str = _store_path_str(tmp_path / "performance.sqlite3",
        _statement_str("U700", [("20260807", 0, 0), ("20260810", 0, 0)], currency_str="EUR"))
    assert _discover_list(db_path_str) == []


def test_missing_database_and_reads_never_write(tmp_path):
    assert _discover_list(str(tmp_path / "missing.sqlite3")) == []
    assert not (tmp_path / "missing.sqlite3").exists()
    db_path_str = _store_path_str(tmp_path / "performance.sqlite3", _statement_str("U900", [("20260807", 5, 5), ("20260810", 5, 5)]))
    before_tuple = (Path(db_path_str).read_bytes(), Path(db_path_str).stat().st_mtime_ns)
    assert len(_discover_list(db_path_str)) == 1
    assert (Path(db_path_str).read_bytes(), Path(db_path_str).stat().st_mtime_ns) == before_tuple
