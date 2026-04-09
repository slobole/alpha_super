from __future__ import annotations

from datetime import UTC, datetime

from alpha.live.models import BrokerPositionSnapshot
from alpha.live.reconcile import reconcile_account_state


def test_reconcile_account_state_passes_on_clean_match():
    broker_snapshot_obj = BrokerPositionSnapshot(
        account_route_str="U1",
        snapshot_timestamp_ts=datetime.now(UTC),
        cash_float=1000.0,
        total_value_float=1500.0,
        position_amount_map={"AAPL": 5.0},
    )

    reconciliation_result_obj = reconcile_account_state(
        model_position_map={"AAPL": 5.0},
        model_cash_float=1000.0,
        broker_snapshot_obj=broker_snapshot_obj,
    )

    assert reconciliation_result_obj.passed_bool is True
    assert reconciliation_result_obj.mismatch_dict == {}


def test_reconcile_account_state_blocks_on_position_mismatch():
    broker_snapshot_obj = BrokerPositionSnapshot(
        account_route_str="U1",
        snapshot_timestamp_ts=datetime.now(UTC),
        cash_float=1000.0,
        total_value_float=1500.0,
        position_amount_map={"AAPL": 3.0},
    )

    reconciliation_result_obj = reconcile_account_state(
        model_position_map={"AAPL": 5.0},
        model_cash_float=1000.0,
        broker_snapshot_obj=broker_snapshot_obj,
    )

    assert reconciliation_result_obj.passed_bool is False
    assert "AAPL" in reconciliation_result_obj.mismatch_dict


def test_reconcile_account_state_ignores_cash_mismatch_in_v1():
    broker_snapshot_obj = BrokerPositionSnapshot(
        account_route_str="U1",
        snapshot_timestamp_ts=datetime.now(UTC),
        cash_float=850.0,
        total_value_float=1350.0,
        position_amount_map={"AAPL": 5.0},
    )

    reconciliation_result_obj = reconcile_account_state(
        model_position_map={"AAPL": 5.0},
        model_cash_float=1000.0,
        broker_snapshot_obj=broker_snapshot_obj,
    )

    assert reconciliation_result_obj.passed_bool is True
    assert reconciliation_result_obj.mismatch_dict == {}
    assert reconciliation_result_obj.model_cash_float == 1000.0
    assert reconciliation_result_obj.broker_cash_float == 850.0
