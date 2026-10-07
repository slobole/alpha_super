"""Daily completion is atomic actual-account settlement, independent of fills."""
from dataclasses import replace
from datetime import timedelta
import json

import pytest

from alpha.live.models import BrokerSnapshot
from alpha.live import mr_capsule_notifications
from test_live_daily_reconcile import CLOSE_TS, daily_case


def _snapshot(release_obj, position_dict, timestamp_ts=CLOSE_TS):
    return BrokerSnapshot(release_obj.account_route_str, timestamp_ts,
        cash_float=500.0, total_value_float=100000.0, net_liq_float=100000.0, position_amount_map=position_dict)


def _alert_list(store_obj):
    with store_obj._connect() as connection_obj:
        return connection_obj.execute("SELECT * FROM mr_capsule_execution_alert WHERE alert_kind_str='daily_exception'").fetchall()


@pytest.mark.parametrize("exception_bool", [False, True])
def test_actual_holdings_finish_without_fills_and_are_idempotent(daily_case, monkeypatch, exception_bool):
    store_obj, release_obj, decision_obj, plan_obj, _ = daily_case
    actual_dict = {"MSFT": 3.0, "BIL": 100.0} if exception_bool else {**plan_obj.current_broker_position_map, **plan_obj.target_share_map}
    exception_list = [{"asset_str": "MSFT", "quantity_float": 3.0, "side_str": "SELL", "reason_str": "unfilled"}] if exception_bool else []
    status_str = "completed_with_exceptions" if exception_bool else "completed"
    monkeypatch.setattr(store_obj, "get_fill_row_dict_list_for_vplan", lambda *args, **kwargs: pytest.fail("Fills must not gate completion"))
    assert store_obj.complete_daily_cycle(decision_obj.decision_plan_id_int, plan_obj.vplan_id_int,
        _snapshot(release_obj, actual_dict), status_str, exception_list, CLOSE_TS)
    assert not store_obj.complete_daily_cycle(decision_obj.decision_plan_id_int, plan_obj.vplan_id_int,
        _snapshot(release_obj, actual_dict), status_str, exception_list, CLOSE_TS)
    assert store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int).status_str == status_str
    assert store_obj.get_vplan_by_id(plan_obj.vplan_id_int).status_str == status_str
    assert store_obj.get_pod_state(release_obj.pod_id_str).position_amount_map == actual_dict
    assert len(_alert_list(store_obj)) == int(exception_bool)
    assert len(store_obj.get_pod_state_history_row_dict_list(release_obj.pod_id_str)) == 1


def test_decision_only_missed_batch_closes_without_a_vplan(daily_case):
    store_obj, release_obj, decision_obj, _, _ = daily_case
    next_decision_obj = store_obj.insert_decision_plan(replace(decision_obj, decision_plan_id_int=None,
        signal_timestamp_ts=decision_obj.signal_timestamp_ts + timedelta(days=1),
        target_execution_timestamp_ts=decision_obj.target_execution_timestamp_ts + timedelta(days=1), status_str="blocked"))
    next_close_ts = CLOSE_TS + timedelta(days=1)
    exception_list = [{"asset_str": "AAPL", "quantity_float": None, "side_str": "BUY", "reason_str": "decision_intent_not_dispatched"}]
    assert store_obj.complete_daily_cycle(next_decision_obj.decision_plan_id_int, None,
        _snapshot(release_obj, {"BIL": 100.0}, next_close_ts), "completed_with_exceptions", exception_list, next_close_ts)
    assert _alert_list(store_obj)[0]["vplan_id_int"] is None
    assert _alert_list(store_obj)[0]["decision_plan_id_int"] == next_decision_obj.decision_plan_id_int


def test_alert_failure_rolls_back_state_and_both_statuses(daily_case, monkeypatch):
    store_obj, release_obj, decision_obj, plan_obj, _ = daily_case
    def fail_alert(*args, **kwargs):
        raise RuntimeError("simulated outbox failure")
    monkeypatch.setattr(mr_capsule_notifications, "enqueue_daily_exception_alert", fail_alert)
    with pytest.raises(RuntimeError, match="outbox failure"):
        store_obj.complete_daily_cycle(decision_obj.decision_plan_id_int, plan_obj.vplan_id_int,
            _snapshot(release_obj, {"MSFT": 5.0}), "completed_with_exceptions",
            [{"asset_str": "MSFT", "quantity_float": 5.0, "side_str": "SELL", "reason_str": "unfilled"}], CLOSE_TS)
    assert store_obj.get_pod_state(release_obj.pod_id_str) is None
    assert store_obj.get_vplan_by_id(plan_obj.vplan_id_int).status_str == "submitted"
    assert store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int).status_str == "submitted"
    assert _alert_list(store_obj) == []


@pytest.mark.parametrize("failure_str", ["before_close", "stale", "wrong_account", "nonfinite", "changed_vplan"])
def test_completion_rejects_invalid_observation_or_changed_cycle(daily_case, failure_str):
    store_obj, release_obj, decision_obj, plan_obj, _ = daily_case
    snapshot_obj = _snapshot(release_obj, {})
    as_of_ts, vplan_id_int = CLOSE_TS, plan_obj.vplan_id_int
    if failure_str == "before_close":
        as_of_ts = CLOSE_TS - timedelta(seconds=1)
    elif failure_str == "stale":
        snapshot_obj = replace(snapshot_obj, snapshot_timestamp_ts=CLOSE_TS - timedelta(seconds=1))
    elif failure_str == "wrong_account":
        snapshot_obj = replace(snapshot_obj, account_route_str="DU_OTHER")
    elif failure_str == "nonfinite":
        snapshot_obj = replace(snapshot_obj, cash_float=float("nan"))
    elif failure_str == "changed_vplan":
        vplan_id_int = None
    with pytest.raises(ValueError):
        store_obj.complete_daily_cycle(decision_obj.decision_plan_id_int, vplan_id_int,
            snapshot_obj, "completed", [], as_of_ts)
    assert store_obj.get_pod_state(release_obj.pod_id_str) is None
    assert store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int).status_str == "submitted"


@pytest.mark.parametrize("achieved_bool", [False, True])
def test_receipt_commits_only_when_actual_frozen_targets_are_held(daily_case, achieved_bool):
    store_obj, release_obj, decision_obj, plan_obj, _ = daily_case
    prior_dict = {"last_applied_rebalance_date_str": "prior"}
    candidate_dict = {"last_applied_rebalance_date_str": "new"}
    target_dict = dict(plan_obj.target_share_map)
    metadata_dict = {**decision_obj.snapshot_metadata_dict, "fixed_target_share_map_dict": target_dict,
        "base_strategy_state_dict": {"core5_execution_receipt_dict": prior_dict},
        "core5_candidate_execution_receipt_dict": candidate_dict}
    with store_obj._connect() as connection_obj:
        connection_obj.execute("UPDATE decision_plan SET snapshot_metadata_json_str=? WHERE decision_plan_id_int=?",
            (json.dumps(metadata_dict), decision_obj.decision_plan_id_int))
    actual_dict = target_dict if achieved_bool else {"MSFT": 3.0}
    exception_list = [] if achieved_bool else [{"asset_str": "MSFT", "quantity_float": 3.0, "side_str": "SELL", "reason_str": "unfilled"}]
    store_obj.complete_daily_cycle(decision_obj.decision_plan_id_int, plan_obj.vplan_id_int,
        _snapshot(release_obj, actual_dict), "completed" if achieved_bool else "completed_with_exceptions", exception_list, CLOSE_TS)
    assert store_obj.get_pod_state(release_obj.pod_id_str).strategy_state_dict["core5_execution_receipt_dict"] == (candidate_dict if achieved_bool else prior_dict)


def test_older_cycle_closes_without_overwriting_newer_pod_state(daily_case):
    store_obj, release_obj, decision_obj, plan_obj, _ = daily_case
    newer_state_dict = {"core5_execution_receipt_dict": {"last_applied_rebalance_date_str": "newer"}}
    newer_decision_obj = store_obj.insert_decision_plan(replace(decision_obj, decision_plan_id_int=None,
        signal_timestamp_ts=decision_obj.signal_timestamp_ts + timedelta(days=1),
        target_execution_timestamp_ts=decision_obj.target_execution_timestamp_ts + timedelta(days=1),
        strategy_state_dict=newer_state_dict, snapshot_metadata_dict={}, status_str="blocked"))
    next_close_ts = CLOSE_TS + timedelta(days=1)
    newer_snapshot_obj = _snapshot(release_obj, {"BIL": 50.0}, next_close_ts)
    store_obj.complete_daily_cycle(newer_decision_obj.decision_plan_id_int, None,
        newer_snapshot_obj, "completed", [], next_close_ts)
    expected_state_obj = store_obj.get_pod_state(release_obj.pod_id_str)
    # This is also the upgrade case: an abandoned old decision is still pending.
    store_obj.complete_daily_cycle(decision_obj.decision_plan_id_int, plan_obj.vplan_id_int,
        _snapshot(release_obj, {"MSFT": 3.0}, next_close_ts + timedelta(seconds=1)),
        "completed_with_exceptions", [{"asset_str": "MSFT", "quantity_float": 3.0,
            "side_str": "SELL", "reason_str": "unfilled"}], next_close_ts)
    assert store_obj.get_pod_state(release_obj.pod_id_str) == expected_state_obj
    assert len(store_obj.get_pod_state_history_row_dict_list(release_obj.pod_id_str)) == 1
    closed_obj = store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int)
    assert closed_obj.status_str == "completed_with_exceptions"
    assert closed_obj.snapshot_metadata_dict["daily_execution_result_dict"]["pod_state_preserved_for_newer_decision_id_int"] == newer_decision_obj.decision_plan_id_int
    assert len(_alert_list(store_obj)) == 1


@pytest.mark.parametrize("identity_field_str", ["pod_id_str", "user_id_str", "account_route_str"])
def test_release_identity_change_cannot_redirect_cycle_state(daily_case, identity_field_str):
    store_obj, release_obj, decision_obj, plan_obj, _ = daily_case
    changed_release_obj = replace(release_obj, **{identity_field_str: "CHANGED_IDENTITY"})
    store_obj.upsert_release(changed_release_obj)
    with pytest.raises(ValueError):
        store_obj.complete_daily_cycle(decision_obj.decision_plan_id_int, plan_obj.vplan_id_int,
            _snapshot(changed_release_obj, {}), "completed", [], CLOSE_TS)
    assert store_obj.get_pod_state(release_obj.pod_id_str) is None
    assert store_obj.get_pod_state(changed_release_obj.pod_id_str) is None
    assert store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int).status_str == "submitted"
    assert store_obj.get_vplan_by_id(plan_obj.vplan_id_int).status_str == "submitted"
    assert not _alert_list(store_obj)
