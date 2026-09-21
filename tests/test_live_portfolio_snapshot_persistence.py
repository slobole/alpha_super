from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
from unittest.mock import Mock
from types import SimpleNamespace

import pytest

from alpha.live.models import BrokerSnapshot, LiveRelease
from alpha.live.order_clerk import IBKRGatewayBrokerAdapter, StubBrokerAdapter
from alpha.live.state_store_v2 import LiveStateStore
from alpha.live.dashboard_v4.pod_broker_holdings import load_broker_holdings_dict


@pytest.fixture
def snapshot_fixture_tuple(tmp_path):
    store_obj = LiveStateStore(str(tmp_path / "live.sqlite3"))
    release_obj = LiveRelease(
        release_id_str="release_1", user_id_str="user_1", pod_id_str="pod_1",
        account_route_str="U1", strategy_import_str="unused:Strategy", mode_str="live",
        session_calendar_id_str="XNYS", signal_clock_str="eod_snapshot_ready",
        execution_policy_str="next_open_moo", data_profile_str="unused", params_dict={},
        risk_profile_str="standard", enabled_bool=True, source_path_str="unused.yaml")
    snapshot_ts = datetime(2026, 9, 18, 20, 10, tzinfo=UTC)
    valuation_dict = {
        "available_bool": True, "reason_str": "", "source_str": "IBKR portfolio",
        "account_route_str": "U1", "observed_timestamp_str": snapshot_ts.isoformat(),
        "currency_str": "USD", "cash_float": 100.0, "broker_nav_float": 1000.0,
        "position_list": [{"symbol_str": "AAPL", "conid_int": 265598,
            "currency_str": "USD", "shares_float": 9.0, "market_price_float": 100.0,
            "value_float": 900.0}]}
    snapshot_obj = BrokerSnapshot(account_route_str="U1", snapshot_timestamp_ts=snapshot_ts,
        cash_float=100.0, total_value_float=1000.0, net_liq_float=1000.0,
        position_amount_map={"AAPL": 9.0}, portfolio_valuation_dict=valuation_dict)
    return store_obj, release_obj, snapshot_obj


def _saved_payload_dict(store_obj):
    with store_obj._connect() as connection_obj:
        payload_str = connection_obj.execute(
            "SELECT portfolio_valuation_json_str FROM broker_snapshot_cache WHERE account_route_str='U1'").fetchone()[0]
    return json.loads(payload_str) if payload_str else None


def test_owned_sample_survives_new_execution_snapshot_without_redating(snapshot_fixture_tuple):
    store_obj, release_obj, snapshot_obj = snapshot_fixture_tuple
    store_obj.upsert_broker_snapshot_cache(snapshot_obj, release_obj=release_obj)
    saved_dict = _saved_payload_dict(store_obj)
    assert saved_dict["owner_dict"] == {field_str: getattr(release_obj, field_str) for field_str in
        ("release_id_str", "user_id_str", "pod_id_str", "account_route_str", "mode_str")}
    assert saved_dict["schema_version_int"] == 1
    assert "owner_dict" not in snapshot_obj.portfolio_valuation_dict

    # VPlan/reconcile updates current cash/shares, never the display sample's time or values.
    newer_obj = replace(snapshot_obj, snapshot_timestamp_ts=snapshot_obj.snapshot_timestamp_ts + timedelta(days=3),
        cash_float=200.0, position_amount_map={"MSFT": 5.0}, portfolio_valuation_dict=None)
    store_obj.upsert_broker_snapshot_cache(newer_obj)
    assert _saved_payload_dict(store_obj) == saved_dict
    execution_obj = store_obj.get_latest_broker_snapshot_for_account("U1")
    assert execution_obj.position_amount_map == {"MSFT": 5.0}
    assert execution_obj.cash_float == 200.0
    assert execution_obj.snapshot_timestamp_ts == newer_obj.snapshot_timestamp_ts
    assert execution_obj.portfolio_valuation_dict is None


@pytest.mark.parametrize("valuation_obj", [None, {"available_bool": False, "reason_str": "timeout"},
    {"cash_float": float("nan")}, {"position_list": object()}, {"padding_str": "x" * 262145}])
def test_failed_new_capture_does_not_revive_older_success_or_block_eod(snapshot_fixture_tuple, valuation_obj):
    store_obj, release_obj, snapshot_obj = snapshot_fixture_tuple
    store_obj.upsert_broker_snapshot_cache(snapshot_obj, release_obj=release_obj)
    store_obj.upsert_broker_snapshot_cache(replace(snapshot_obj, cash_float=150.0,
        portfolio_valuation_dict=valuation_obj), release_obj=release_obj)
    assert _saved_payload_dict(store_obj)["available_bool"] is False
    assert store_obj.get_latest_broker_snapshot_for_account("U1").cash_float == 150.0


def test_wrong_release_account_cannot_attest_sample(snapshot_fixture_tuple):
    store_obj, release_obj, snapshot_obj = snapshot_fixture_tuple
    store_obj.upsert_broker_snapshot_cache(snapshot_obj, release_obj=replace(release_obj, account_route_str="U2"))
    assert _saved_payload_dict(store_obj)["available_bool"] is False


def test_stored_capture_round_trips_through_real_dashboard_reader(snapshot_fixture_tuple, tmp_path):
    store_obj, release_obj, snapshot_obj = snapshot_fixture_tuple
    store_obj.upsert_release(release_obj)
    store_obj.upsert_broker_snapshot_cache(snapshot_obj, release_obj=release_obj)
    target_obj = SimpleNamespace(release_obj=release_obj, db_path_str=str(tmp_path / "live.sqlite3"))
    holdings_dict = load_broker_holdings_dict(target_obj,
        as_of_ts=snapshot_obj.snapshot_timestamp_ts + timedelta(seconds=1))
    assert holdings_dict["available_bool"] is True
    assert holdings_dict["position_list"] == snapshot_obj.portfolio_valuation_dict["position_list"]
    assert holdings_dict["cash_float"] == 100.0
    assert holdings_dict["observed_timestamp_str"] == snapshot_obj.snapshot_timestamp_ts.isoformat()


def test_additive_migration_preserves_legacy_cache_and_is_repeatable(snapshot_fixture_tuple):
    store_obj, _, snapshot_obj = snapshot_fixture_tuple
    store_obj.upsert_broker_snapshot_cache(replace(snapshot_obj, portfolio_valuation_dict=None))
    with store_obj._connect() as connection_obj:
        connection_obj.execute("ALTER TABLE broker_snapshot_cache DROP COLUMN portfolio_valuation_json_str")
    for _ in range(2):
        store_obj._initialize_v2_schema()
    assert _saved_payload_dict(store_obj) is None
    assert store_obj.get_latest_broker_snapshot_for_account("U1").position_amount_map == {"AAPL": 9.0}


def test_only_eod_adapter_method_requests_valuation():
    adapter_obj = IBKRGatewayBrokerAdapter()
    adapter_obj.socket_client_obj = Mock()
    adapter_obj.get_account_snapshot("U1")
    adapter_obj.socket_client_obj.get_account_snapshot.assert_called_once_with("U1")
    adapter_obj.socket_client_obj.reset_mock()
    adapter_obj.get_eod_account_snapshot("U1")
    adapter_obj.socket_client_obj.get_account_snapshot.assert_called_once_with("U1", include_portfolio_valuation_bool=True)


def test_adapters_without_valuation_keep_existing_eod_behavior():
    adapter_obj = StubBrokerAdapter()
    adapter_obj.get_account_snapshot = Mock(return_value="existing snapshot")
    assert adapter_obj.get_eod_account_snapshot("U1") == "existing snapshot"
    adapter_obj.get_account_snapshot.assert_called_once_with("U1")
