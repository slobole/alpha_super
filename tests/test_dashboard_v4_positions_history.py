"""Month view: daily IBKR MTM rows are verified, reconciled and folded per Pod and symbol."""

from contextlib import closing
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
import hashlib
import sqlite3
import xml.etree.ElementTree as ElementTree

import pytest

from alpha.live.client_reporting import build_client_report_dict, parse_broker_nav_import
from alpha.live.dashboard_v4.demo import DEMO_NOW_TS, build_demo_workspace_tuple
from alpha.live.dashboard_v4.performance import build_performance_page_dict
from alpha.live.dashboard_v4.positions import build_positions_page_dict
from alpha.live.dashboard_v4.positions_history import (
    MISSING_SECTION_STR, ROW_STR, SECTION_STR, day_mtm_dict, load_day_mtm_map_dict, month_option_list,
    parse_day_mtm_dict, parse_import_dict,
)


ACCOUNT_STR = "U123"
QUERY_STR = "ALPHA_DAILY_TWR"
AS_OF_TS = datetime(2026, 9, 9, 14, 0, tzinfo=UTC)


def _row_dict(symbol_str, conid_int, **field_dict):
    """A row whose prior-open MTM matches its own quantity and prices by default."""
    row_dict = {"accountId": ACCOUNT_STR, "assetCategory": "STK", "currency": "USD", "symbol": symbol_str, "conid": str(conid_int),
        "prevCloseQuantity": "10", "prevClosePrice": "100", "closeQuantity": "10", "transactionMtm": "0", "priorOpenMtm": "10",
        "commissions": "0"}
    row_dict.update({key_str: str(value_obj) for key_str, value_obj in field_dict.items()})
    if "closePrice" not in field_dict:
        quantity_decimal = Decimal(row_dict["prevCloseQuantity"])
        row_dict["closePrice"] = str(Decimal(row_dict["prevClosePrice"]) + (Decimal(row_dict["priorOpenMtm"]) / quantity_decimal
            if quantity_decimal else 0))
    return row_dict


def _nav_dict(date_str, mtm_str, commissions_str="0", **extra_dict):
    compact_str = date_str.replace("-", "")
    return {"accountId": ACCOUNT_STR, "currency": "USD", "fromDate": compact_str, "toDate": compact_str,
        "startingValue": "10000", "endingValue": "10010", "twr": "0.1", "mtm": mtm_str, "commissions": commissions_str, **extra_dict}


def _xml_str(statement_list, query_str=QUERY_STR):
    """statement_list: [(nav_dict_list, row_list or None, statement_attribute_dict)]."""
    root_obj = ElementTree.Element("FlexQueryResponse", queryName=query_str)
    statements_obj = ElementTree.SubElement(root_obj, "FlexStatements", count=str(len(statement_list)))
    for nav_dict_list, row_list, statement_dict in statement_list:
        statement_obj = ElementTree.SubElement(statements_obj, "FlexStatement", accountId=ACCOUNT_STR, **statement_dict)
        ElementTree.SubElement(statement_obj, "AccountInformation", accountId=ACCOUNT_STR, currency="USD")
        for nav_dict in nav_dict_list:
            ElementTree.SubElement(statement_obj, "ChangeInNAV", **nav_dict)
        if row_list is not None:
            section_obj = ElementTree.SubElement(statement_obj, SECTION_STR)
            for row_dict in row_list:
                ElementTree.SubElement(section_obj, ROW_STR, **row_dict)
    return ElementTree.tostring(root_obj, encoding="unicode")


def _nav_rows(raw_xml_str, import_id_int=7):
    return parse_broker_nav_import(raw_xml_str, allowed_account_set={ACCOUNT_STR}, query_name_str=QUERY_STR,
        source_import_id_int=import_id_int, source_checksum_str=hashlib.sha256(raw_xml_str.encode()).hexdigest())


def _day_tuple(row_list, *, date_str="2026-09-08", mtm_str="10", commissions_str="0", section_bool=True, import_id_int=7):
    raw_xml_str = _xml_str([([_nav_dict(date_str, mtm_str, commissions_str)], row_list if section_bool else None, {})])
    return raw_xml_str, _nav_rows(raw_xml_str, import_id_int)[0]


def _parse(row_list, **kwargs_dict):
    raw_xml_str, nav_row_obj = _day_tuple(row_list, **kwargs_dict)
    return parse_day_mtm_dict(raw_xml_str, nav_row_obj, query_name_str=QUERY_STR, as_of_ts=AS_OF_TS)


# ---------- one day ----------

def test_reconciled_day_returns_one_row_per_instrument_and_skips_subtotals():
    result_dict = _parse([_row_dict("AAA", 1, priorOpenMtm="12.5"), _row_dict("BBB", 2, priorOpenMtm="-4.5", transactionMtm="3"),
        # IBKR subtotal rows have no conid; even with empty account/date fields they are skipped.
        {"accountId": "", "reportDate": "", "assetCategory": "", "symbol": "", "conid": "", "priorOpenMtm": "11", "transactionMtm": "3"}],
        mtm_str="11")
    assert result_dict["available_bool"] is True
    assert [row_dict["symbol_str"] for row_dict in result_dict["row_list"]] == ["AAA", "BBB"]
    assert result_dict["row_list"][1]["transaction_mtm_float"] == 3.0
    assert not any(key_str.endswith("_decimal") for key_str in result_dict["row_list"][0])


@pytest.mark.parametrize(("row_list", "kwargs_dict", "reason_str"), [
    ([_row_dict("AAA", 1, priorOpenMtm="12")], {"mtm_str": "11"}, "does not add up"),
    ([_row_dict("AAA", 1, commissions="-1")], {"commissions_str": "-2.5"}, "commissions do not add up"),
    ([_row_dict("AAA", 1, assetCategory="OPT")], {}, "could not be verified"),
    ([_row_dict("AAA", 1, currency="EUR")], {}, "could not be verified"),
    ([{key_str: value_str for key_str, value_str in _row_dict("AAA", 1).items() if key_str != "currency"}], {}, "could not be verified"),
    ([{key_str: value_str for key_str, value_str in _row_dict("AAA", 1).items() if key_str != "accountId"}], {}, "could not be verified"),
    ([_row_dict("AAA", 1, multiplier="100")], {}, "could not be verified"),
    ([_row_dict("AAA", 1, fxRateToBase="1.1")], {}, "could not be verified"),
    ([_row_dict("AAA", 1, priorOpenMtm="5"), _row_dict("AAA", 2, priorOpenMtm="5")], {}, "could not be verified"),
    ([_row_dict("AAA", 1, priorOpenMtm="5"), _row_dict("BBB", 1, priorOpenMtm="5")], {}, "could not be verified"),
    ([_row_dict("AAA", 1, reportDate="20260905")], {}, "could not be verified"),
    ([_row_dict("AAA", 1, closePrice="nan")], {}, "could not be verified"),
    ([_row_dict("AAA", 1, accountId="U999")], {}, "could not be verified"),
    # Quantity x10 with the same MTM: the row contradicts its own prices.
    ([_row_dict("AAA", 1, prevCloseQuantity="100", closeQuantity="100", closePrice="101")], {}, "does not match its own quantity and prices"),
    ([_row_dict("AAA", 1, closePrice="105")], {}, "does not match its own quantity and prices"),
    # Same MTM, absurd close quantity: gross value far above NAV.
    ([_row_dict("AAA", 1, closeQuantity="100000")], {}, "exceed four times"),
])
def test_unverifiable_days_fail_closed(row_list, kwargs_dict, reason_str):
    result_dict = _parse(row_list, **kwargs_dict)
    assert result_dict["available_bool"] is False and result_dict["row_list"] == []
    assert reason_str in result_dict["reason_str"]


@pytest.mark.parametrize(("prior_str", "available_bool"), [("10.05", True), ("9.95", True), ("10.06", False), ("9.94", False)])
def test_reconciliation_tolerance_is_five_cents(prior_str, available_bool):
    assert _parse([_row_dict("AAA", 1, priorOpenMtm=prior_str)], mtm_str="10")["available_bool"] is available_bool


def test_rows_without_commissions_pass_and_leave_commissions_to_other():
    row_dict = {key_str: value_str for key_str, value_str in _row_dict("AAA", 1).items() if key_str != "commissions"}
    result_dict = _parse([row_dict], commissions_str="-1.5")
    assert result_dict["available_bool"] is True and result_dict["row_list"][0]["commission_float"] == 0.0


def test_missing_section_is_named_and_today_is_never_final():
    missing_dict = _parse([], section_bool=False)
    assert missing_dict["missing_section_bool"] is True and missing_dict["reason_str"] == MISSING_SECTION_STR
    raw_xml_str, nav_row_obj = _day_tuple([_row_dict("AAA", 1)], date_str="2026-09-09")
    # D+1: the current market date is not a finalized close.
    assert parse_day_mtm_dict(raw_xml_str, nav_row_obj, query_name_str=QUERY_STR, as_of_ts=AS_OF_TS)["available_bool"] is False


def test_checksum_and_query_identity_are_required():
    raw_xml_str, nav_row_obj = _day_tuple([_row_dict("AAA", 1)])
    assert parse_day_mtm_dict(raw_xml_str.replace('closeQuantity="10"', 'closeQuantity="11"'), nav_row_obj,
        query_name_str=QUERY_STR, as_of_ts=AS_OF_TS)["available_bool"] is False
    assert parse_day_mtm_dict(raw_xml_str, nav_row_obj, query_name_str="OTHER", as_of_ts=AS_OF_TS)["available_bool"] is False


def test_nav_without_mtm_cannot_be_reconciled():
    raw_xml_str, _ = _day_tuple([_row_dict("AAA", 1)])
    stripped_str = raw_xml_str.replace(' mtm="10"', "")
    assert stripped_str != raw_xml_str
    stripped_row_obj = _nav_rows(stripped_str)[0]
    assert parse_day_mtm_dict(stripped_str, stripped_row_obj, query_name_str=QUERY_STR, as_of_ts=AS_OF_TS)["available_bool"] is False


def test_multi_day_statement_rows_must_name_their_day():
    row_list = [_row_dict("AAA", 1, priorOpenMtm="5", reportDate="20260903"), _row_dict("AAA", 1, priorOpenMtm="7", reportDate="2026-09-04")]
    raw_xml_str = _xml_str([([_nav_dict("2026-09-03", "5"), _nav_dict("2026-09-04", "7")], row_list, {})])
    for nav_row_obj in _nav_rows(raw_xml_str):
        day_dict = parse_day_mtm_dict(raw_xml_str, nav_row_obj, query_name_str=QUERY_STR, as_of_ts=AS_OF_TS)
        assert day_dict["available_bool"] is True and len(day_dict["row_list"]) == 1
    row_list[1].pop("reportDate")
    raw_xml_str = _xml_str([([_nav_dict("2026-09-03", "5"), _nav_dict("2026-09-04", "7")], row_list, {})])
    assert not any(parse_day_mtm_dict(raw_xml_str, nav_row_obj, query_name_str=QUERY_STR, as_of_ts=AS_OF_TS)["available_bool"]
        for nav_row_obj in _nav_rows(raw_xml_str))


def test_two_statements_for_one_account_day_are_ambiguous():
    statement_tuple = ([_nav_dict("2026-09-08", "10")], [_row_dict("AAA", 1)], {})
    raw_xml_str = _xml_str([statement_tuple, statement_tuple])
    block_dict = parse_import_dict(raw_xml_str, query_name_str=QUERY_STR)
    assert block_dict[(ACCOUNT_STR, "2026-09-08")]["error_str"] == "Two saved statements cover this day."


def test_mtm_only_statement_uses_its_own_date_and_reconciles_to_the_nav_query():
    _, nav_row_obj = _day_tuple([_row_dict("AAA", 1)])
    mtm_xml_str = _xml_str([([], [_row_dict("AAA", 1)], {"fromDate": "20260908", "toDate": "20260908"})], query_str="ALPHA_DAILY_MTM")
    block_dict = parse_import_dict(mtm_xml_str, query_name_str="ALPHA_DAILY_MTM")
    assert block_dict[(ACCOUNT_STR, "2026-09-08")]["nav_attribute_dict"] is None
    assert day_mtm_dict(block_dict[(ACCOUNT_STR, "2026-09-08")], nav_row_obj)["available_bool"] is True


# ---------- saved database ----------

def _database_path_str(path_obj, import_list):
    path_obj.mkdir(parents=True, exist_ok=True)
    database_path_obj = path_obj / "performance.sqlite3"
    with closing(sqlite3.connect(database_path_obj)) as connection_obj, connection_obj:
        connection_obj.execute("CREATE TABLE flex_import (import_id_int INTEGER PRIMARY KEY, imported_timestamp_str TEXT, "
            "request_from_date_str TEXT, request_to_date_str TEXT, query_name_str TEXT, checksum_str TEXT, raw_xml_str TEXT)")
        for import_tuple in import_list:
            connection_obj.execute("INSERT INTO flex_import VALUES (?, ?, ?, ?, ?, ?, ?)", import_tuple)
    return str(database_path_obj)


def _import_tuple(import_id_int, date_str, row_list, *, query_str=QUERY_STR, from_str=None, to_str=None, timestamp_str="2026-09-09T06:00:00+00:00", mtm_str="10"):
    raw_xml_str = _xml_str([([_nav_dict(date_str, mtm_str)], row_list, {})], query_str=query_str)
    return (import_id_int, timestamp_str, from_str or date_str, to_str or date_str, query_str,
        hashlib.sha256(raw_xml_str.encode()).hexdigest(), raw_xml_str)


def _load(database_str, row_list, **kwargs_dict):
    return load_day_mtm_map_dict(database_str, row_list, query_name_str=QUERY_STR, as_of_ts=AS_OF_TS, **kwargs_dict)


def test_database_reader_takes_each_days_newest_import(tmp_path):
    first_tuple = _import_tuple(7, "2026-09-03", [_row_dict("AAA", 1)])
    second_tuple = _import_tuple(8, "2026-09-04", [_row_dict("AAA", 1)])
    first_row_obj, second_row_obj = _nav_rows(first_tuple[6], 7)[0], _nav_rows(second_tuple[6], 8)[0]
    result_dict = _load(_database_path_str(tmp_path / "a", [first_tuple, second_tuple]), [first_row_obj, second_row_obj])
    assert [result_dict[(ACCOUNT_STR, date_str)]["available_bool"] for date_str in ("2026-09-03", "2026-09-04")] == [True, True]
    # A newer import covering 09-03 again supersedes the NAV row's own import.
    third_tuple = _import_tuple(9, "2026-09-03", [_row_dict("AAA", 1, priorOpenMtm="5")], mtm_str="5")
    result_dict = _load(_database_path_str(tmp_path / "b", [first_tuple, second_tuple, third_tuple]), [first_row_obj, second_row_obj])
    assert result_dict[(ACCOUNT_STR, "2026-09-03")]["available_bool"] is False
    assert result_dict[(ACCOUNT_STR, "2026-09-04")]["available_bool"] is True


@pytest.mark.parametrize("case_str", ["checksum", "range", "naive", "inverted", "missing", "future"])
def test_database_reader_rejects_unprovable_sources(tmp_path, case_str):
    import_tuple = _import_tuple(7, "2026-09-08", [_row_dict("AAA", 1)])
    nav_row_obj = _nav_rows(import_tuple[6], 7)[0]
    import_list = [import_tuple]
    if case_str == "checksum":
        import_list = [(*import_tuple[:5], "0" * 64, import_tuple[6])]
    elif case_str == "range":
        import_list = [(import_tuple[0], import_tuple[1], "2026-09-01", "2026-09-02", *import_tuple[4:])]
    elif case_str == "naive":
        import_list = [(import_tuple[0], "2026-09-09T06:00:00", *import_tuple[2:])]
    elif case_str == "inverted":
        import_list = [import_tuple, (8, "2026-09-09T07:00:00+00:00", "2026-09-10", "2026-09-01", QUERY_STR, "x" * 64, "<x/>")]
    elif case_str == "missing":
        import_list = [(6, *import_tuple[1:])]
    elif case_str == "future":
        import_list = [(import_tuple[0], "2026-09-10T06:00:00+00:00", *import_tuple[2:])]
    result_dict = _load(_database_path_str(tmp_path, import_list), [nav_row_obj])
    assert result_dict[(ACCOUNT_STR, "2026-09-08")]["available_bool"] is False


def test_database_reader_names_a_missing_database(tmp_path):
    _, nav_row_obj = _day_tuple([_row_dict("AAA", 1)])
    result_dict = _load(str(tmp_path / "missing.sqlite3"), [nav_row_obj])
    assert result_dict[(ACCOUNT_STR, "2026-09-08")]["reason_str"] == "No saved IBKR performance database."


def test_database_reader_supports_a_separate_mtm_query(tmp_path):
    _, nav_row_obj = _day_tuple([_row_dict("AAA", 1)])
    mtm_xml_str = _xml_str([([], [_row_dict("AAA", 1)], {"fromDate": "20260908", "toDate": "20260908"})], query_str="ALPHA_DAILY_MTM")
    import_tuple = (3, "2026-09-09T06:00:00+00:00", "2026-09-08", "2026-09-08", "ALPHA_DAILY_MTM",
        hashlib.sha256(mtm_xml_str.encode()).hexdigest(), mtm_xml_str)
    result_dict = _load(_database_path_str(tmp_path, [import_tuple]), [nav_row_obj], mtm_query_name_str="ALPHA_DAILY_MTM")
    assert result_dict[(ACCOUNT_STR, "2026-09-08")]["available_bool"] is True


# ---------- months on the demo book ----------

@pytest.fixture(scope="module")
def demo_tuple():
    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple(include_holdings_bool=True)
    try:
        yield workspace_dict, snapshot_obj, provider_obj
    finally:
        provider_obj.close()


def _month_page(demo_tuple, month_str, pod_str="all", *, workspace_dict=None, snapshot_obj=None, as_of_ts=DEMO_NOW_TS):
    base_workspace_dict, base_snapshot_obj, provider_obj = demo_tuple
    return build_positions_page_dict(workspace_dict or base_workspace_dict, snapshot_obj or base_snapshot_obj, provider_obj,
        as_of_ts=as_of_ts, month_str=month_str, pod_str=pod_str)


class _getter:
    """Temporarily replace the demo's per-day source."""

    def __init__(self, provider_obj, getter_fn):
        self.provider_obj, self.getter_fn = provider_obj, getter_fn

    def __enter__(self):
        self.original_fn = self.provider_obj.get_day_mtm_dict
        self.provider_obj.get_day_mtm_dict = lambda nav_row_obj, **kwargs_dict: self.getter_fn(nav_row_obj, self.original_fn, **kwargs_dict)
        return self

    def __exit__(self, *_):
        self.provider_obj.get_day_mtm_dict = self.original_fn


def _missing_dict():
    return {"available_bool": False, "reason_str": MISSING_SECTION_STR, "missing_section_bool": True, "row_list": []}


def test_month_options_are_newest_first_and_mark_the_open_month(demo_tuple):
    workspace_dict, snapshot_obj, provider_obj = demo_tuple
    now_dict = build_positions_page_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS)
    assert [option_dict["month_str"] for option_dict in now_dict["month_option_list"]] == ["2026-09", "2026-08", "2026-07", "2026-06"]
    assert now_dict["month_option_list"][0]["to_date_bool"] is True and "(to date)" in now_dict["month_option_list"][0]["label_str"]
    assert all(option_dict["to_date_bool"] is False for option_dict in now_dict["month_option_list"][1:])
    # Only owned accounts' rows make months.
    assert month_option_list(workspace_dict, snapshot_obj, [], as_of_ts=DEMO_NOW_TS) == []


def test_month_totals_are_the_performance_report_and_positions_plus_other_add_up(demo_tuple):
    workspace_dict, snapshot_obj, provider_obj = demo_tuple
    month_dict = _month_page(demo_tuple, "2026-08")["month_dict"]
    assert month_dict["available_bool"] is True
    report_dict = build_client_report_dict(workspace_dict["client_dict"], snapshot_obj, from_date_str="2026-08-01",
        to_date_str="2026-08-31", as_of_ts=DEMO_NOW_TS, scope_complete_bool=True)
    performance_dict = build_performance_page_dict(workspace_dict, snapshot_obj, as_of_ts=DEMO_NOW_TS,
        from_date_str="2026-08-01", to_date_str="2026-08-31")
    assert performance_dict["report_dict"]["pnl_float"] == report_dict["pnl_float"]
    assert month_dict["total_dict"]["pnl_str"] == "+$" + f"{report_dict['pnl_float']:,.2f}"
    row_pnl_float = sum(float(row_dict["sort_dict"]["pnl"]) for row_dict in month_dict["row_list"])
    other_float = sum(bar_dict["value_float"] for bar_dict in month_dict["side_dict"]["pod_bar_list"] if bar_dict.get("other_bool"))
    assert row_pnl_float + other_float == pytest.approx(report_dict["pnl_float"], abs=0.05)
    assert other_float == pytest.approx(0, abs=0.05)


def test_month_rows_state_held_dates_status_and_time_weighted_size(demo_tuple):
    month_dict = _month_page(demo_tuple, "2026-08")["month_dict"]
    sold_list = [row_dict for row_dict in month_dict["row_list"] if row_dict["status_str"].startswith("Sold")]
    kept_list = [row_dict for row_dict in month_dict["row_list"] if row_dict["status_str"] == "Held to month end"]
    assert sold_list and kept_list and all(row_dict["status_tag_bool"] for row_dict in sold_list)
    assert len({row_dict["key_str"] for row_dict in month_dict["row_list"]}) == len(month_dict["row_list"])
    map_item_list = [item_dict for pod_dict in month_dict["map_dict"]["pod_list"] for item_dict in pod_dict["item_list"]]
    assert {item_dict["status_str"] for item_dict in map_item_list} >= {"sold", "kept"}
    assert all(item_dict["status_note_str"].startswith("sold ") for item_dict in map_item_list if item_dict["status_str"] == "sold")
    # Size counts only the sessions held: a short holding is smaller than its value while held.
    assert all(row_dict["sort_dict"]["weight"] <= row_dict["sort_dict"]["value"] + 0.01 for row_dict in month_dict["row_list"])
    assert any(row_dict["sort_dict"]["weight"] < row_dict["sort_dict"]["value"] - 0.01 for row_dict in month_dict["row_list"])
    open_dict = _month_page(demo_tuple, "2026-09")["month_dict"]
    assert open_dict["to_date_bool"] is True
    assert any(row_dict["status_str"] == "Open now" for row_dict in open_dict["row_list"])
    assert not any(row_dict["status_str"] == "Held to month end" for row_dict in open_dict["row_list"])


def test_a_month_ending_on_a_weekend_row_keeps_open_positions_open(demo_tuple):
    workspace_dict, snapshot_obj, provider_obj = demo_tuple

    def cut_snapshot(last_str):
        return replace(snapshot_obj, row_tuple=tuple(row_obj for row_obj in snapshot_obj.row_tuple if row_obj.market_date_str <= last_str))
    as_of_ts = datetime(2026, 9, 2, 14, 0, tzinfo=UTC)
    friday_dict = _month_page(demo_tuple, "2026-08", snapshot_obj=cut_snapshot("2026-08-28"), as_of_ts=as_of_ts)["month_dict"]
    sunday_dict = _month_page(demo_tuple, "2026-08", snapshot_obj=cut_snapshot("2026-08-30"), as_of_ts=as_of_ts)["month_dict"]
    assert friday_dict["available_bool"] is sunday_dict["available_bool"] is True
    status_dict = {row_dict["key_str"]: row_dict["status_str"] for row_dict in friday_dict["row_list"]}
    assert {row_dict["key_str"]: row_dict["status_str"] for row_dict in sunday_dict["row_list"]} == status_dict
    assert sum(status_str == "Open now" for status_str in status_dict.values()) >= 5


@pytest.mark.parametrize(("pod_id_str", "field_str", "date_str", "month_list"), [
    ("demo_1_3", "effective_from", "2026-07-15", ["2026-06", "2026-07", "2026-08"]),
    ("demo_1_3", "effective_to", "2026-08-20", ["2026-08", "2026-09"]),
])
def test_pods_starting_or_retiring_mid_month_keep_months_reportable(demo_tuple, pod_id_str, field_str, date_str, month_list):
    workspace_dict, snapshot_obj, provider_obj = demo_tuple
    changed_dict = deepcopy(workspace_dict)
    for account_list in (changed_dict["client_dict"]["accounts"], changed_dict["valuation_account_list"], changed_dict["operations_account_list"]):
        for account_dict in account_list:
            if account_dict["pod_id"] == pod_id_str:
                account_dict[field_str] = date_str
    for month_str in month_list:
        month_dict = _month_page(demo_tuple, month_str, workspace_dict=changed_dict)["month_dict"]
        first_str = month_str + "-01"
        last_str = month_dict.get("to_date_str") or first_str
        performance_dict = build_performance_page_dict(changed_dict, snapshot_obj, as_of_ts=DEMO_NOW_TS,
            from_date_str=first_str, to_date_str=last_str)
        assert month_dict["available_bool"] is True, month_dict["reason_str"]
        assert month_dict["total_dict"]["pnl_str"].replace("−", "-") == f"{performance_dict['report_dict']['pnl_float']:+,.2f}".replace("+", "+$").replace("-", "-$")


def test_pod_filter_scopes_map_and_rows_but_not_tiles_or_by_pod(demo_tuple):
    all_dict = _month_page(demo_tuple, "2026-07")["month_dict"]
    one_dict = _month_page(demo_tuple, "2026-07", "demo_1_3")["month_dict"]
    assert {row_dict["pod_dict"]["pod_id_str"] for row_dict in one_dict["row_list"]} == {"demo_1_3"}
    assert [pod_dict["pod_id_str"] for pod_dict in one_dict["map_dict"]["pod_list"]] == ["demo_1_3"]
    assert one_dict["tile_list"] == all_dict["tile_list"]
    assert one_dict["pod_row_list"] == all_dict["pod_row_list"]
    tqqq_dict = next(item_dict for item_dict in one_dict["map_dict"]["pod_list"][0]["item_list"] if item_dict["symbol_str"] == "TQQQ")
    assert tqqq_dict["multiplier_float"] == 3.0 and any(row_list[0] == "Exposure" for row_list in tqqq_dict["tip_list"])


def test_missing_mtm_section_explains_the_flex_change_and_shows_no_numbers(demo_tuple):
    with _getter(demo_tuple[2], lambda nav_row_obj, original_fn, **_: _missing_dict()):
        page_dict = _month_page(demo_tuple, "2026-08")
    month_dict = page_dict["month_dict"]
    assert month_dict["available_bool"] is False and "no per-symbol P&L" in month_dict["reason_str"]
    assert "Flex setup guide" in month_dict["reason_str"]
    assert month_dict["row_list"] == [] and month_dict["tile_list"] == []
    assert page_dict["map_dict"]["available_bool"] is False and page_dict["map_dict"]["reason_str"] == month_dict["reason_str"]


def test_partial_and_late_section_coverage_are_named(demo_tuple):
    def early_missing(nav_row_obj, original_fn, **kwargs_dict):
        return _missing_dict() if nav_row_obj.market_date_str < "2026-08-17" else original_fn(nav_row_obj, **kwargs_dict)

    def late_missing(nav_row_obj, original_fn, **kwargs_dict):
        return _missing_dict() if nav_row_obj.market_date_str >= "2026-08-20" else original_fn(nav_row_obj, **kwargs_dict)

    def weekend_missing(nav_row_obj, original_fn, **kwargs_dict):
        return _missing_dict() if nav_row_obj.market_date_str in {"2026-08-01", "2026-08-02", "2026-08-08"} else original_fn(nav_row_obj, **kwargs_dict)
    with _getter(demo_tuple[2], early_missing):
        assert "starts Aug 17" in _month_page(demo_tuple, "2026-08")["month_dict"]["reason_str"]
    with _getter(demo_tuple[2], late_missing):
        assert "missing from Aug 20" in _month_page(demo_tuple, "2026-08")["month_dict"]["reason_str"]
    with _getter(demo_tuple[2], weekend_missing):
        assert _month_page(demo_tuple, "2026-08")["month_dict"]["available_bool"] is True


def test_one_bad_session_day_withholds_the_month_with_its_day_and_pod(demo_tuple):
    def bad_day(nav_row_obj, original_fn, **kwargs_dict):
        if (nav_row_obj.account_route_str, nav_row_obj.market_date_str) == ("DEMO_1_0", "2026-08-05"):
            return {"available_bool": False, "reason_str": "Per-symbol P&L does not add up to the day's IBKR mark-to-market.",
                "missing_section_bool": False, "row_list": []}
        return original_fn(nav_row_obj, **kwargs_dict)
    with _getter(demo_tuple[2], bad_day):
        page_dict = _month_page(demo_tuple, "2026-08")
    assert page_dict["month_dict"]["reason_str"] == "Aug 5 · DVO2: Per-symbol P&L does not add up to the day's IBKR mark-to-market."
    assert page_dict["map_dict"]["reason_str"] == page_dict["month_dict"]["reason_str"]


def test_quantities_must_carry_over_between_sessions(demo_tuple):
    def broken_day(nav_row_obj, original_fn, **kwargs_dict):
        day_dict = original_fn(nav_row_obj, **kwargs_dict)
        if (nav_row_obj.account_route_str, nav_row_obj.market_date_str) == ("DEMO_1_2", "2026-08-12") and day_dict["available_bool"]:
            day_dict = deepcopy(day_dict)
            day_dict["row_list"] = day_dict["row_list"][1:]
        return day_dict
    with _getter(demo_tuple[2], broken_day):
        month_dict = _month_page(demo_tuple, "2026-08")["month_dict"]
    assert month_dict["available_bool"] is False and "NDX Momentum: Quantities do not carry over" in month_dict["reason_str"]


def _row(symbol_str, conid_int, previous_float, close_float, *, price_float=100.0, prior_float=0.0, transaction_float=0.0):
    return {"symbol_str": symbol_str, "conid_int": conid_int, "previous_quantity_float": previous_float, "previous_price_float": price_float,
        "close_quantity_float": close_float, "close_price_float": price_float + (prior_float / previous_float if previous_float else 0.0),
        "transaction_mtm_float": transaction_float, "prior_open_mtm_float": prior_float, "commission_float": 0.0, "commission_seen_bool": False}


def test_exact_month_aggregation_for_sold_short_day_trade_and_weekend_rows(demo_tuple):
    def fake_day(nav_row_obj, original_fn, **_):
        date_str = nav_row_obj.market_date_str
        row_list = []
        if nav_row_obj.account_route_str == "DEMO_1_0":
            if date_str == "2026-08-03":
                row_list = [_row("XXX", 1, 0, 10), _row("YYY", 2, 0, -5, price_float=50.0)]
            elif date_str == "2026-08-04":
                row_list = [_row("XXX", 1, 10, 0, prior_float=10.0), _row("YYY", 2, -5, -5, price_float=50.0)]
            elif date_str == "2026-08-05":
                row_list = [_row("YYY", 2, -5, -5, price_float=50.0), _row("ZZZ", 3, 0, 0, transaction_float=3.0)]
            elif date_str == "2026-08-08":
                row_list = [_row("YYY", 2, -5, -5, price_float=50.0, prior_float=2.0)]
            elif "2026-08-05" < date_str <= "2026-08-31":
                row_list = [_row("YYY", 2, -5, -5, price_float=50.0)]
        return {"available_bool": True, "reason_str": "", "missing_section_bool": False, "row_list": row_list}
    with _getter(demo_tuple[2], fake_day):
        month_dict = _month_page(demo_tuple, "2026-08", "demo_1_0")["month_dict"]
    assert month_dict["available_bool"] is True, month_dict["reason_str"]
    row_by_symbol_dict = {row_dict["symbol_str"]: row_dict for row_dict in month_dict["row_list"]}
    x_dict, y_dict, z_dict = row_by_symbol_dict["XXX"], row_by_symbol_dict["YYY"], row_by_symbol_dict["ZZZ"]
    assert (x_dict["held_str"], x_dict["session_str"], x_dict["value_str"], x_dict["status_str"]) == ("Aug 3 – Aug 4", "2 of 21 sessions", "$1,000.00", "Sold Aug 4")
    assert x_dict["sort_dict"]["weight"] == pytest.approx(2000 / 21, abs=0.01) and x_dict["return_str"] == "+1.0% while held"
    # The short is sized by its absolute value; the Saturday P&L counts but adds no session.
    assert y_dict["short_bool"] is True and y_dict["status_str"] == "Held to month end"
    assert y_dict["sort_dict"]["weight"] == pytest.approx(250.0, abs=0.01) and y_dict["sort_dict"]["pnl"] == 2.0
    assert y_dict["session_str"] == "21 of 21 sessions"
    # A same-day round trip is a table row with P&L but no capital, so no map tile.
    assert z_dict["sort_dict"]["pnl"] == 3.0 and z_dict["value_str"] == "$0.00" and z_dict["return_str"] == ""
    tile_symbol_set = {item_dict["symbol_str"] for pod_dict in month_dict["map_dict"]["pod_list"] for item_dict in pod_dict["item_list"]}
    assert tile_symbol_set == {"XXX", "YYY"}


def test_unverified_scope_or_unreadable_sources_withhold_the_month(demo_tuple):
    workspace_dict, snapshot_obj, provider_obj = demo_tuple
    draft_dict = {**workspace_dict, "financial_scope_complete_bool": False}
    month_dict = _month_page(demo_tuple, "2026-08", workspace_dict=draft_dict)["month_dict"]
    performance_dict = build_performance_page_dict(draft_dict, snapshot_obj, as_of_ts=DEMO_NOW_TS,
        from_date_str="2026-08-01", to_date_str="2026-08-31")
    # Same answer as Performance: an unproven scope has no month P&L.
    assert performance_dict["report_dict"]["pnl_float"] is None
    assert month_dict["available_bool"] is False and month_dict["reason_str"].startswith("Month P&L could not be verified:")
    assert month_dict["tile_list"] == [] and month_dict["row_list"] == []
    production_dict = deepcopy(workspace_dict)
    production_dict["client_dict"]["is_demo"] = False
    month_dict = _month_page(demo_tuple, "2026-08", workspace_dict=production_dict)["month_dict"]
    assert month_dict["available_bool"] is False and month_dict["reason_str"] == "No saved IBKR performance database."


def test_unknown_month_is_reported_for_a_404_and_unreadable_reports_are_not(demo_tuple):
    workspace_dict, snapshot_obj, provider_obj = demo_tuple
    assert _month_page(demo_tuple, "2026-01")["month_known_bool"] is False
    error_dict = {**workspace_dict, "financial_error_str": "Saved financial data could not be verified."}
    page_dict = _month_page(demo_tuple, "2026-01", workspace_dict=error_dict)
    assert page_dict["month_known_bool"] is True and page_dict["month_dict"]["reason_str"] == "Saved IBKR reports unavailable."
    assert page_dict["month_dict"]["name_str"] == "January 2026"
