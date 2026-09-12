"""Cash display must preserve the canonical NAV scope, date and report."""

from copy import deepcopy
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
import sqlite3

import pytest

from alpha.live.dashboard_v3.client_cash import load_portfolio_cash_list, _is_selected_eod_bool
from alpha.live.dashboard_v3.client_presentation import portfolio_allocation_dict
from test_dashboard_operator_access import ForbiddenProvider


AS_OF_TS = datetime(2026, 9, 5, 12, tzinfo=UTC)


def allocation_fixture_tuple():
    report_dict = {"closing_date_str": "2026-09-03", "closing_nav_float": 200, "scope_complete_bool": False,
        "valuation_account_list": [{"account_route": route_str, "pod_id": "pod_" + route_str, "display_name": route_str} for route_str in ("A", "B")]}
    snapshot_obj = SimpleNamespace(row_tuple=tuple(SimpleNamespace(account_route_str=route_str,
        market_date_str="2026-09-03", closing_nav_decimal=Decimal(100)) for route_str in ("A", "B")))
    cash_list = [{"account_route_str": route_str, "pod_id_str": "pod_" + route_str,
        "market_date_str": "2026-09-03", "cash_float": 30, "equity_float": 100} for route_str in ("A", "B")]
    return report_dict, snapshot_obj, cash_list


def test_each_fifteen_percent_cash_contribution_sums_to_thirty_percent():
    report_dict, snapshot_obj, cash_list = allocation_fixture_tuple()
    before_dict = deepcopy(report_dict)
    result_dict = portfolio_allocation_dict(report_dict, snapshot_obj, cash_snapshot_list=cash_list)
    assert result_dict["cash_complete_bool"] is True
    assert result_dict["cash_weight_float"] == .3
    assert [item_dict["cash_weight_float"] for item_dict in result_dict["item_list"]] == [.15, .15]
    assert [item_dict["invested_weight_float"] for item_dict in result_dict["item_list"]] == [.35, .35]
    assert sum(item_dict["weight_float"] for item_dict in result_dict["item_list"]) == 1
    assert report_dict == before_dict


def test_real_broker_flex_difference_uses_one_eod_basis_for_the_whole_ring():
    report_dict, snapshot_obj, cash_list = allocation_fixture_tuple()
    flex_nav_list = [20603.785969295, 12111.235842935]
    broker_nav_list = [20608.33, 12104.40]
    broker_cash_list = [287.08, 1969.23]
    report_dict["closing_nav_float"] = sum(flex_nav_list)
    for row_obj, flex_nav_float, cash_dict, equity_float, cash_float in zip(
            snapshot_obj.row_tuple, flex_nav_list, cash_list, broker_nav_list, broker_cash_list):
        row_obj.closing_nav_decimal = Decimal(str(flex_nav_float))
        cash_dict.update(equity_float=equity_float, cash_float=cash_float)
    before_dict = deepcopy(report_dict)
    result_dict = portfolio_allocation_dict(report_dict, snapshot_obj, cash_snapshot_list=cash_list)
    assert result_dict["cash_complete_bool"] is True
    assert result_dict["source_str"] == "broker_eod"
    assert result_dict["total_float"] == pytest.approx(32712.73)
    assert result_dict["cash_weight_float"] == pytest.approx(2256.31 / 32712.73)
    for item_dict, equity_float, cash_float in zip(result_dict["item_list"], broker_nav_list, broker_cash_list):
        assert item_dict["value_float"] == equity_float
        assert item_dict["weight_float"] == pytest.approx(equity_float / 32712.73)
        assert item_dict["cash_weight_float"] == pytest.approx(cash_float / 32712.73)
        assert item_dict["invested_weight_float"] == pytest.approx((equity_float - cash_float) / 32712.73)
    assert sum(item_dict["weight_float"] for item_dict in result_dict["item_list"]) == pytest.approx(1)
    assert report_dict == before_dict
    assert [float(row_obj.closing_nav_decimal) for row_obj in snapshot_obj.row_tuple] == flex_nav_list


@pytest.mark.parametrize("override_dict", [
    {"cash_float": None}, {"cash_float": -1}, {"cash_float": 101}, {"cash_float": True},
    {"cash_float": float("nan")}, {"cash_float": float("inf")},
])
def test_invalid_cash_keeps_eod_equity_and_never_becomes_zero(override_dict):
    report_dict, snapshot_obj, cash_list = allocation_fixture_tuple()
    cash_list[0]["equity_float"] = 80
    cash_list[1]["equity_float"] = 120
    cash_list[0].update(override_dict)
    result_dict = portfolio_allocation_dict(report_dict, snapshot_obj, cash_snapshot_list=cash_list)
    assert result_dict["source_str"] == "broker_eod"
    assert result_dict["cash_weight_float"] is None and not result_dict["cash_complete_bool"]
    assert [item_dict["value_float"] for item_dict in result_dict["item_list"]] == [80, 120]
    assert [item_dict["weight_float"] for item_dict in result_dict["item_list"]] == [.4, .6]
    assert result_dict["item_list"][0]["cash_float"] is None
    assert result_dict["item_list"][1]["cash_float"] == 30


@pytest.mark.parametrize("override_dict", [
    {"equity_float": None}, {"equity_float": -1}, {"equity_float": float("inf")},
    {"equity_float": float("nan")}, {"equity_float": True}, {"account_route_str": "OTHER"},
    {"pod_id_str": "OTHER"}, {"market_date_str": "2026-09-02"},
])
def test_incomplete_eod_equity_keeps_entire_flex_ring_without_mixing_cash(override_dict):
    report_dict, snapshot_obj, cash_list = allocation_fixture_tuple()
    cash_list[0].update(override_dict)
    cash_list[1]["equity_float"] = 150
    result_dict = portfolio_allocation_dict(report_dict, snapshot_obj, cash_snapshot_list=cash_list)
    assert result_dict["source_str"] == "flex_nav"
    assert [item_dict["value_float"] for item_dict in result_dict["item_list"]] == [100, 100]
    assert all(item_dict["cash_float"] is None for item_dict in result_dict["item_list"])
    assert not result_dict["cash_complete_bool"]


def test_cash_limits_use_its_own_eod_equity_even_when_above_flex_nav():
    report_dict, snapshot_obj, cash_list = allocation_fixture_tuple()
    cash_list[0].update(equity_float=200, cash_float=150)
    result_dict = portfolio_allocation_dict(report_dict, snapshot_obj, cash_snapshot_list=cash_list)
    assert result_dict["cash_complete_bool"]
    assert result_dict["total_float"] == 300
    assert result_dict["cash_weight_float"] == pytest.approx(.6)


@pytest.mark.parametrize("cash_float", [0, 100])
def test_zero_or_all_cash_is_valid(cash_float):
    report_dict, snapshot_obj, cash_list = allocation_fixture_tuple()
    for cash_dict in cash_list:
        cash_dict["cash_float"] = cash_float
    result_dict = portfolio_allocation_dict(report_dict, snapshot_obj, cash_snapshot_list=cash_list)
    assert result_dict["cash_complete_bool"]
    assert result_dict["cash_weight_float"] == cash_float / 100


@pytest.mark.parametrize("cash_float,cash_count_int,invested_count_int,center_str", [
    (0, 0, 2, "0.0%"), (100, 2, 0, "100.0%"), (None, 0, 0, "—"),
])
def test_cash_ring_renders_zero_all_and_unknown_states(cash_float, cash_count_int, invested_count_int, center_str):
    from flask import Flask, render_template_string
    from pathlib import Path

    report_dict, snapshot_obj, cash_list = allocation_fixture_tuple()
    for cash_dict in cash_list:
        cash_dict["cash_float"] = cash_float
    result_dict = portfolio_allocation_dict(report_dict, snapshot_obj, cash_snapshot_list=cash_list)
    app_obj = Flask(__name__, template_folder=str(Path("alpha/live/dashboard_v3/templates").resolve()))
    with app_obj.app_context():
        html_str = render_template_string("{% from '_client_allocation.html' import allocation_chart %}{{ allocation_chart(allocation_dict, 'Portfolio allocation') }}", allocation_dict=result_dict)
    assert html_str.count('data-allocation-kind="cash"') == cash_count_int
    assert html_str.count('data-allocation-kind="invested"') == invested_count_int
    assert 'class="client-donut-count">' + center_str + '</text>' in html_str


def test_duplicate_names_keep_account_identity():
    report_dict, snapshot_obj, cash_list = allocation_fixture_tuple()
    for account_dict in report_dict["valuation_account_list"]:
        account_dict["display_name"] = "Same strategy"
    result_dict = portfolio_allocation_dict(report_dict, snapshot_obj, cash_snapshot_list=cash_list)
    assert all(item_dict["show_identity_bool"] for item_dict in result_dict["item_list"])
    assert [item_dict["detail_str"] for item_dict in result_dict["item_list"]] == ["A", "B"]


def test_duplicate_cash_evidence_is_unknown():
    report_dict, snapshot_obj, cash_list = allocation_fixture_tuple()
    cash_list.append(dict(cash_list[0]))
    result_dict = portfolio_allocation_dict(report_dict, snapshot_obj, cash_snapshot_list=cash_list)
    assert result_dict["item_list"][0]["cash_float"] is None


def saved_cash_fixture_tuple(tmp_path):
    database_path_obj = tmp_path / "cash.sqlite3"
    with sqlite3.connect(database_path_obj) as connection_obj:
        connection_obj.execute("CREATE TABLE pod_state_history (pod_id_str TEXT, account_route_str TEXT, user_id_str TEXT, snapshot_stage_str TEXT, snapshot_source_str TEXT, updated_timestamp_str TEXT, cash_float REAL, total_value_float REAL)")
        for date_str, cash_float in [("2026-09-02", 10), ("2026-09-03", 30), ("2026-09-04", 70)]:
            connection_obj.execute("INSERT INTO pod_state_history VALUES (?,?,?,?,?,?,?,?)",
                ("pod_A", "A", "owner", "eod", "broker", date_str + "T20:10:00+00:00", cash_float, 100))
    release_obj = SimpleNamespace(mode_str="live", pod_id_str="pod_A", account_route_str="A", user_id_str="owner", session_calendar_id_str="XNYS")
    target_obj = SimpleNamespace(db_path_str=str(database_path_obj), release_obj=release_obj)
    provider_obj = SimpleNamespace(get_target_for_pod=lambda pod_str: target_obj)
    report_dict, _, _ = allocation_fixture_tuple()
    report_dict["valuation_account_list"] = report_dict["valuation_account_list"][:1]
    return database_path_obj, target_obj, provider_obj, report_dict


def test_historical_cash_uses_exact_date_and_read_only_database(tmp_path):
    database_path_obj, _, provider_obj, report_dict = saved_cash_fixture_tuple(tmp_path)
    before_tuple = (database_path_obj.read_bytes(), database_path_obj.stat().st_mtime_ns)
    cash_list = load_portfolio_cash_list({"operations_source": "local"}, report_dict, provider_obj, {}, as_of_ts=AS_OF_TS)
    assert cash_list == [{"pod_id_str": "pod_A", "account_route_str": "A", "market_date_str": "2026-09-03", "cash_float": 30, "equity_float": 100}]
    assert (database_path_obj.read_bytes(), database_path_obj.stat().st_mtime_ns) == before_tuple


@pytest.mark.parametrize("field_str,value_obj", [
    ("account_route_str", "OTHER"), ("pod_id_str", "OTHER"), ("user_id_str", "OTHER"),
    ("snapshot_stage_str", "intraday"), ("snapshot_source_str", "model"),
    ("updated_timestamp_str", "2026-09-03T19:00:00+00:00"),
    ("updated_timestamp_str", "2026-09-03T20:09:59+00:00"),
    ("updated_timestamp_str", "2026-09-03T20:10:00"),
    ("updated_timestamp_str", "2026-09-06T20:10:00+00:00"),
])
def test_cash_history_rejects_wrong_owner_source_and_timing(tmp_path, field_str, value_obj):
    database_path_obj, _, provider_obj, report_dict = saved_cash_fixture_tuple(tmp_path)
    with sqlite3.connect(database_path_obj) as connection_obj:
        connection_obj.execute(f"UPDATE pod_state_history SET {field_str}=? WHERE updated_timestamp_str LIKE '2026-09-03%'", (value_obj,))
    assert load_portfolio_cash_list({"operations_source": "local"}, report_dict, provider_obj, {}, as_of_ts=AS_OF_TS) == []


def test_duplicate_latest_capture_is_not_silently_selected(tmp_path):
    database_path_obj, _, provider_obj, report_dict = saved_cash_fixture_tuple(tmp_path)
    with sqlite3.connect(database_path_obj) as connection_obj:
        connection_obj.execute("INSERT INTO pod_state_history SELECT * FROM pod_state_history WHERE updated_timestamp_str LIKE '2026-09-03%'")
    operations_dict = {"strategy_list": [{"pod_id_str": "pod_A", "account_route_str": "A", "matched_bool": True,
        "evidence_dict": {"session_calendar_id_str": "XNYS", "eod_snapshot_dict": {"source_str": "broker",
            "latest_market_date_str": "2026-09-03", "latest_timestamp_str": "2026-09-03T20:10:00Z", "cash_float": 30, "equity_float": 100}}}]}
    assert load_portfolio_cash_list({"operations_source": "local"}, report_dict, provider_obj, operations_dict, as_of_ts=AS_OF_TS) == []


def test_malformed_old_timestamp_does_not_hide_valid_selected_date(tmp_path):
    database_path_obj, _, provider_obj, report_dict = saved_cash_fixture_tuple(tmp_path)
    with sqlite3.connect(database_path_obj) as connection_obj:
        connection_obj.execute("UPDATE pod_state_history SET updated_timestamp_str='broken' WHERE updated_timestamp_str LIKE '2026-09-02%'")
    assert load_portfolio_cash_list({"operations_source": "local"}, report_dict, provider_obj, {}, as_of_ts=AS_OF_TS)[0]["cash_float"] == 30


@pytest.mark.parametrize("date_str,timestamp_str,expected_bool", [
    ("2026-11-27", "2026-11-27T18:10:00Z", True),
    ("2026-11-27", "2026-11-27T18:09:59Z", False),
    ("2026-11-26", "2026-11-26T22:00:00Z", False),
    ("2026-11-28", "2026-11-28T22:00:00Z", False),
])
def test_half_day_holiday_and_weekend_boundaries(date_str, timestamp_str, expected_bool):
    assert _is_selected_eod_bool(timestamp_str, date_str, "XNYS", datetime(2026, 12, 1, tzinfo=UTC)) is expected_bool


def test_snapshot_cash_uses_owned_date_and_never_calls_local_provider():
    report_dict, _, _ = allocation_fixture_tuple()
    report_dict["valuation_account_list"] = report_dict["valuation_account_list"][:1]
    operations_dict = {"strategy_list": [{"pod_id_str": "pod_A", "account_route_str": "A", "matched_bool": True,
        "evidence_dict": {"session_calendar_id_str": "XNYS", "eod_snapshot_dict": {"source_str": "broker",
            "latest_market_date_str": "2026-09-03", "latest_timestamp_str": "2026-09-03T20:10:00Z", "cash_float": 30, "equity_float": 100}}}]}
    assert load_portfolio_cash_list({"operations_source": "snapshot"}, report_dict, ForbiddenProvider(), operations_dict, as_of_ts=AS_OF_TS)[0]["cash_float"] == 30
    operations_dict["strategy_list"][0]["matched_bool"] = False
    assert load_portfolio_cash_list({"operations_source": "snapshot"}, report_dict, ForbiddenProvider(), operations_dict, as_of_ts=AS_OF_TS) == []


@pytest.mark.parametrize("broker_difference_float", [0, 4.54])
def test_real_local_cash_uses_own_equity_without_changing_report(tmp_path, monkeypatch, broker_difference_float):
    from test_dashboard_local_workspace import build_fixture_app, file_snapshot_dict
    from test_live_dashboard import _seed_eod_pod_state

    app_obj = build_fixture_app(tmp_path, monkeypatch, expanded_bool=True, new_pod_bool=False)
    client_obj = app_obj.test_client()
    path_str = "/clients/local/overview?from=2026-09-02&to=2026-09-03"
    before_report_dict = client_obj.get(path_str + "&download=json").get_json()
    for pod_str, nav_float in [("pod_a", 1105 + broker_difference_float), ("pod_b", 10030)]:
        target_obj = app_obj.config["data_provider_obj"].get_target_for_pod(pod_str)
        _seed_eod_pod_state(tmp_path / (pod_str + ".sqlite3"), target_obj.release_obj,
            total_value_float=nav_float, cash_float=nav_float * .3, updated_timestamp_ts=datetime(2026, 9, 3, 20, 10, tzinfo=UTC))
    before_files_dict = file_snapshot_dict(tmp_path)
    html_str = client_obj.get(path_str).get_data(as_text=True)
    assert 'class="client-donut-count">30.0%</text>' in html_str
    assert html_str.count('data-allocation-kind="cash"') == 2
    assert 'data-allocation-source="broker_eod"' in html_str
    assert 'Saved broker end-of-day snapshot; financial totals use finalized Flex · 2026-09-03' in html_str
    assert '${:,.2f}'.format(1105 + broker_difference_float) in html_str
    assert 'IBKR update pending' not in html_str and 'Source: IBKR' not in html_str
    after_report_dict = client_obj.get(path_str + "&download=json").get_json()
    for key_str in ["report_hash_str", "closing_nav_float", "pnl_float", "twr_float"]:
        assert after_report_dict[key_str] == before_report_dict[key_str]
    assert file_snapshot_dict(tmp_path) == before_files_dict


def test_missing_database_is_not_created_and_snapshot_source_never_reads_local(tmp_path):
    database_path_obj, target_obj, provider_obj, report_dict = saved_cash_fixture_tuple(tmp_path)
    target_obj.db_path_str = str(tmp_path / "missing.sqlite3")
    assert load_portfolio_cash_list({"operations_source": "local"}, report_dict, provider_obj, {}, as_of_ts=AS_OF_TS) == []
    assert not (tmp_path / "missing.sqlite3").exists()
    assert load_portfolio_cash_list({"operations_source": "snapshot"}, report_dict, ForbiddenProvider(), {}, as_of_ts=AS_OF_TS) == []
