"""Daily CORE5 receipts commit actual achievement; missed targets replay next day."""
from dataclasses import replace
import sqlite3

import pytest

from alpha.live.models import BrokerSnapshot
from alpha.live.state_store_v2 import LiveStateStore
import test_live_core5_adapter as fixture_module


@pytest.fixture
def receipt_case(tmp_path, monkeypatch):
    release_obj = fixture_module.release_obj.__wrapped__()
    price_df = fixture_module.price_df.__wrapped__()
    fixture_module._controlled_signals(monkeypatch, price_df, {
        ("2026-09-11", "DBC", "long_state_ser"): 0.0,
        ("2026-09-11", "DBC", "short_state_ser"): 1.0,
        ("2026-09-14", "DBC", "long_state_ser"): 0.0,
        ("2026-09-14", "DBC", "short_state_ser"): 1.0,
    })
    prior_obj = fixture_module._build(release_obj, price_df, "2026-09-10", fixture_module._state(release_obj, "2026-09-10"))
    state_obj = fixture_module._state(release_obj, "2026-09-11",
        prior_obj.snapshot_metadata_dict["fixed_target_share_map_dict"], fixture_module._committed_state_dict(prior_obj))
    store_obj = LiveStateStore(str(tmp_path / "receipt.sqlite3"))
    store_obj.upsert_release(release_obj)
    store_obj.upsert_pod_state(state_obj)
    decision_obj = store_obj.insert_decision_plan(fixture_module._build(release_obj, price_df, "2026-09-11", state_obj))
    return store_obj, release_obj, price_df, state_obj, decision_obj


@pytest.mark.parametrize("achieved_bool", [False, True])
def test_end_of_day_actual_targets_control_receipt_and_next_decision(receipt_case, achieved_bool):
    store_obj, release_obj, price_df, state_obj, decision_obj = receipt_case
    target_dict = dict(decision_obj.snapshot_metadata_dict["fixed_target_share_map_dict"])
    actual_dict = dict(target_dict)
    if not achieved_bool:
        actual_dict["DBC"] = state_obj.position_amount_map["DBC"]
    close_ts = fixture_module._time("2026-09-14", 16)
    snapshot_obj = BrokerSnapshot(release_obj.account_route_str, close_ts, 100_000.0, 100_000.0,
        position_amount_map=actual_dict, net_liq_float=100_000.0)
    exception_list = [] if achieved_bool else [{"asset_str": "DBC", "reason_str": "actual_holding_differs_from_vplan",
        "side_str": "SELL", "quantity_float": actual_dict["DBC"] - target_dict["DBC"]}]
    status_str = "completed" if achieved_bool else "completed_with_exceptions"
    assert store_obj.complete_daily_cycle(decision_obj.decision_plan_id_int, None, snapshot_obj, status_str, exception_list, close_ts)
    saved_state_obj = store_obj.get_pod_state(release_obj.pod_id_str)
    expected_receipt_dict = (decision_obj.snapshot_metadata_dict["core5_candidate_execution_receipt_dict"] if achieved_bool
        else state_obj.strategy_state_dict["core5_execution_receipt_dict"])
    assert saved_state_obj.strategy_state_dict["core5_execution_receipt_dict"] == expected_receipt_dict
    assert saved_state_obj.position_amount_map == actual_dict
    assert saved_state_obj.strategy_state_dict["last_signal_date_str"] == "2026-09-11"
    # The next normal EOD decision recovers a partial event without a resume API.
    next_state_obj = replace(saved_state_obj, snapshot_stage_str="eod", updated_timestamp_ts=fixture_module._time("2026-09-14", 17))
    next_obj = fixture_module._build(release_obj, price_df, "2026-09-14", next_state_obj)
    assert next_obj.snapshot_metadata_dict["rebalance_bool"] == (not achieved_bool)
    assert next_obj.snapshot_metadata_dict["core5_catch_up_bool"] == (not achieved_bool)
    history_count_int = len(store_obj.get_pod_state_history_row_dict_list(release_obj.pod_id_str))
    assert not store_obj.complete_daily_cycle(decision_obj.decision_plan_id_int, None, snapshot_obj, status_str, exception_list, close_ts)
    assert len(store_obj.get_pod_state_history_row_dict_list(release_obj.pod_id_str)) == history_count_int


def test_daily_receipt_and_actual_holdings_rollback_together(receipt_case):
    store_obj, release_obj, _, state_obj, decision_obj = receipt_case
    close_ts = fixture_module._time("2026-09-14", 16)
    snapshot_obj = BrokerSnapshot(release_obj.account_route_str, close_ts, 100_000.0, 100_000.0,
        position_amount_map=decision_obj.snapshot_metadata_dict["fixed_target_share_map_dict"], net_liq_float=100_000.0)
    with store_obj._connect() as connection_obj:
        connection_obj.execute("CREATE TRIGGER fail_daily_receipt BEFORE UPDATE OF status_str ON decision_plan BEGIN SELECT RAISE(ABORT, 'injected crash'); END")
    with pytest.raises(sqlite3.IntegrityError, match="injected crash"):
        store_obj.complete_daily_cycle(decision_obj.decision_plan_id_int, None, snapshot_obj, "completed", [], close_ts)
    assert store_obj.get_pod_state(release_obj.pod_id_str) == state_obj
    assert store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int).status_str == "planned"


def test_intraday_receipt_cannot_be_committed(receipt_case):
    store_obj, release_obj, _, state_obj, decision_obj = receipt_case
    intraday_ts = fixture_module._time("2026-09-14", 12)
    snapshot_obj = BrokerSnapshot(release_obj.account_route_str, intraday_ts, 100_000.0, 100_000.0,
        position_amount_map=decision_obj.snapshot_metadata_dict["fixed_target_share_map_dict"], net_liq_float=100_000.0)
    with pytest.raises(ValueError, match="after target close"):
        store_obj.complete_daily_cycle(decision_obj.decision_plan_id_int, None, snapshot_obj, "completed", [], intraday_ts)
    assert store_obj.get_pod_state(release_obj.pod_id_str).strategy_state_dict == state_obj.strategy_state_dict
