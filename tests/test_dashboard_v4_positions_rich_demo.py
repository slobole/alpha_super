"""Rich demo observations are synthetic, dated, and read by production readers."""

from collections import defaultdict
from contextlib import closing
from datetime import timedelta
import json
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

import pytest

from alpha.live.dashboard_v4.demo import DEMO_NOW_TS, build_demo_workspace_tuple
from alpha.live.dashboard_v4.pod_broker_holdings import load_broker_holdings_dict
from alpha.live.dashboard_v4.pod_finance import build_pod_finance_dict
from alpha.live.dashboard_v4.positions_data import load_positions_dict


@pytest.fixture(scope="module")
def rich_demo_tuple():
    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple(include_holdings_bool=True)
    try:
        yield workspace_dict, snapshot_obj, provider_obj
    finally:
        provider_obj.close()


def _target_list(fixture_tuple):
    return fixture_tuple[2].get_target_list()


def test_rich_demo_has_complete_same_day_broker_values_and_quantity_samples(rich_demo_tuple):
    expected_timestamp_str = (DEMO_NOW_TS - timedelta(seconds=2)).isoformat()
    for target_obj in _target_list(rich_demo_tuple):
        holdings_dict = load_broker_holdings_dict(target_obj, as_of_ts=DEMO_NOW_TS)
        positions_dict = load_positions_dict(target_obj, as_of_ts=DEMO_NOW_TS)
        assert holdings_dict["available_bool"] and positions_dict["available_bool"]
        assert holdings_dict["observed_timestamp_str"] == positions_dict["position_timestamp_str"] == expected_timestamp_str
        assert holdings_dict["source_str"] == "IBKR portfolio"
        assert holdings_dict["currency_str"] == "USD"
        assert positions_dict["source_str"] == "broker_snapshot"
        assert {row_dict["symbol_str"]: row_dict["shares_float"] for row_dict in holdings_dict["position_list"]} == positions_dict["position_map_dict"]
        with closing(sqlite3.connect(target_obj.db_path_str)) as connection_obj:
            payload_dict = json.loads(connection_obj.execute(
                "SELECT portfolio_valuation_json_str FROM broker_snapshot_cache").fetchone()[0])
        assert payload_dict["schema_version_int"] == 1
        assert payload_dict["owner_dict"] == {field_str: getattr(target_obj.release_obj, field_str)
            for field_str in ("release_id_str", "user_id_str", "pod_id_str", "account_route_str", "mode_str")}
    assert (DEMO_NOW_TS - timedelta(seconds=2)).astimezone(ZoneInfo("America/New_York")).isoformat().startswith("2026-09-08T09:41:05")


def test_values_costs_and_pnl_are_coherent_with_shared_symbols_and_mixed_results(rich_demo_tuple):
    shared_dict, pnl_list = defaultdict(list), []
    for target_obj in _target_list(rich_demo_tuple):
        holdings_dict = load_broker_holdings_dict(target_obj, as_of_ts=DEMO_NOW_TS)
        for row_dict in holdings_dict["position_list"]:
            assert row_dict["value_float"] == pytest.approx(round(row_dict["shares_float"] * row_dict["market_price_float"], 2))
            assert row_dict["unrealized_pnl_float"] == pytest.approx(round(row_dict["shares_float"] *
                (row_dict["market_price_float"] - row_dict["average_cost_float"]), 2))
            assert row_dict["average_cost_float"] > 0
            pnl_list.append(row_dict["unrealized_pnl_float"])
            shared_dict[row_dict["symbol_str"]].append(row_dict)
        assert sum(row_dict["value_float"] for row_dict in holdings_dict["position_list"]) + holdings_dict["cash_float"] == pytest.approx(holdings_dict["broker_nav_float"], abs=.001)
    assert len(shared_dict["SGOV"]) == 4
    assert len({row_dict["conid_int"] for row_dict in shared_dict["SGOV"]}) == 1
    assert any(pnl_float > 0 for pnl_float in pnl_list)
    assert any(pnl_float < 0 for pnl_float in pnl_list)


def test_demo_pending_order_is_not_invented_as_a_position_and_closed_symbols_are_absent(rich_demo_tuple):
    provider_obj = rich_demo_tuple[2]
    for pod_id_str, closed_symbol_str in (("demo_1_0", "DIS"), ("demo_1_1", "NVDA")):
        target_obj = provider_obj.get_target_for_pod(pod_id_str)
        source_dict = provider_obj.get_pod_cycles_dict(pod_id_str, as_of_ts=DEMO_NOW_TS)
        holdings_dict = load_broker_holdings_dict(target_obj, as_of_ts=DEMO_NOW_TS)
        position_map_dict = {row_dict["symbol_str"]: row_dict["shares_float"] for row_dict in holdings_dict["position_list"]}
        assert closed_symbol_str not in position_map_dict
        assert source_dict["vplan_dict"]["target_execution_timestamp_str"].startswith("2026-09-08")
        before_dict = source_dict["vplan_dict"]["current_broker_position_map_dict"]
        expected_dict = dict(before_dict)
        for order_dict in source_dict["cycle_evidence_dict"]["order_list"]:
            symbol_str = order_dict["asset_str"]
            expected_dict[symbol_str] = expected_dict.get(symbol_str, 0.0) + order_dict["filled_share_float"]
        assert position_map_dict == {symbol_str: shares_float for symbol_str, shares_float in expected_dict.items() if shares_float}
        if pod_id_str == "demo_1_1":
            assert source_dict["cycle_evidence_dict"]["state_str"] == "partial"
            assert "MSFT" not in position_map_dict
            pending_dict = next(row_dict for row_dict in source_dict["cycle_evidence_dict"]["order_list"] if row_dict["asset_str"] == "MSFT")
            assert pending_dict["requested_share_float"] == 17 and pending_dict["filled_share_float"] == 0


def test_live_synthetic_snapshot_does_not_rewrite_historical_eod_or_official_close_panels(rich_demo_tuple):
    workspace_dict, snapshot_obj, provider_obj = rich_demo_tuple
    for index_int, account_dict in enumerate(workspace_dict["operations_account_list"]):
        target_obj = provider_obj.get_target_for_pod(account_dict["pod_id"])
        summary_dict = next(row_dict for row_dict in provider_obj.row_list if row_dict["pod_id_str"] == account_dict["pod_id"])
        with closing(sqlite3.connect(target_obj.db_path_str)) as connection_obj:
            connection_obj.row_factory = sqlite3.Row
            eod_list = connection_obj.execute(
                "SELECT cash_float,total_value_float,updated_timestamp_str FROM pod_state_history "
                "WHERE snapshot_stage_str='eod' ORDER BY updated_timestamp_str DESC").fetchall()
        assert eod_list[0]["updated_timestamp_str"].startswith("2026-09-04")
        assert eod_list[0]["cash_float"] == pytest.approx(summary_dict["eod_snapshot_dict"]["cash_float"], abs=.000001)
        assert eod_list[0]["total_value_float"] == summary_dict["eod_snapshot_dict"]["equity_float"]
        finance_dict = build_pod_finance_dict(workspace_dict, snapshot_obj, provider_obj,
            pod_id_str=account_dict["pod_id"], as_of_ts=DEMO_NOW_TS)
        assert finance_dict["money_asof_str"].startswith("Demo · Close 2026-09-04")
        allocation_dict = finance_dict["holdings_allocation_dict"]
        assert allocation_dict["available_bool"] and allocation_dict["close_date_str"] == "2026-09-04"
        assert len(allocation_dict["row_list"]) == (10 if index_int < 2 else 4)
        assert not allocation_dict.get("broker_observation_bool")


def test_default_helper_still_models_legacy_quantities_only():
    _, _, provider_obj = build_demo_workspace_tuple()
    try:
        for target_obj in provider_obj.get_target_list():
            assert not load_broker_holdings_dict(target_obj, as_of_ts=DEMO_NOW_TS)["available_bool"]
            assert load_positions_dict(target_obj, as_of_ts=DEMO_NOW_TS)["available_bool"]
    finally:
        provider_obj.close()


def test_synthetic_capture_cannot_be_read_before_its_observation(rich_demo_tuple):
    for target_obj in _target_list(rich_demo_tuple):
        assert not load_broker_holdings_dict(target_obj, as_of_ts=DEMO_NOW_TS - timedelta(seconds=3))["available_bool"]


def test_display_reads_do_not_mutate_generated_database(rich_demo_tuple):
    for target_obj in _target_list(rich_demo_tuple):
        database_path_obj = Path(target_obj.db_path_str)
        before_bytes, before_mtime_int = database_path_obj.read_bytes(), database_path_obj.stat().st_mtime_ns
        load_broker_holdings_dict(target_obj, as_of_ts=DEMO_NOW_TS)
        load_positions_dict(target_obj, as_of_ts=DEMO_NOW_TS)
        assert database_path_obj.read_bytes() == before_bytes
        assert database_path_obj.stat().st_mtime_ns == before_mtime_int
