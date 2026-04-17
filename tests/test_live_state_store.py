from __future__ import annotations

from datetime import UTC, datetime, timedelta

from alpha.live.models import LiveRelease, PodState
from alpha.live.state_store import LiveStateStore, SHARED_CORE_TABLE_NAME_TUPLE, V1_EXECUTION_TABLE_NAME_TUPLE


def test_state_store_bootstraps_shared_core_tables_and_release_roundtrip(tmp_path):
    db_path_str = str((tmp_path / "live.sqlite3").resolve())
    state_store_obj = LiveStateStore(db_path_str)
    release_obj = LiveRelease(
        release_id_str="release_001",
        user_id_str="user_001",
        pod_id_str="pod_001",
        account_route_str="DU1",
        strategy_import_str="strategies.dv2.strategy_mr_dv2:DVO2Strategy",
        mode_str="paper",
        session_calendar_id_str="XNYS",
        signal_clock_str="eod_snapshot_ready",
        execution_policy_str="next_open_moo",
        data_profile_str="norgate_eod_sp500_pit",
        params_dict={"max_positions_int": 10},
        risk_profile_str="standard",
        enabled_bool=True,
        source_path_str="manifest.yaml",
        pod_budget_fraction_float=0.15,
        auto_submit_enabled_bool=False,
    )

    state_store_obj.upsert_release(release_obj)
    roundtrip_release_obj = state_store_obj.get_release_by_id(release_obj.release_id_str)
    existing_table_name_set = set(state_store_obj.get_existing_table_name_list())

    assert roundtrip_release_obj.release_id_str == release_obj.release_id_str
    assert roundtrip_release_obj.params_dict == {"max_positions_int": 10}
    assert roundtrip_release_obj.pod_budget_fraction_float == 0.15
    assert roundtrip_release_obj.auto_submit_enabled_bool is False
    assert set(SHARED_CORE_TABLE_NAME_TUPLE).issubset(existing_table_name_set)
    assert set(V1_EXECUTION_TABLE_NAME_TUPLE).isdisjoint(existing_table_name_set)


def test_state_store_claims_scheduler_lease(tmp_path):
    db_path_str = str((tmp_path / "live.sqlite3").resolve())
    state_store_obj = LiveStateStore(db_path_str)
    expires_timestamp_ts = datetime.now(tz=UTC) + timedelta(minutes=5)

    assert state_store_obj.acquire_scheduler_lease(
        lease_name_str="tick",
        owner_token_str="owner_a",
        expires_timestamp_ts=expires_timestamp_ts,
    ) is True
    assert state_store_obj.acquire_scheduler_lease(
        lease_name_str="tick",
        owner_token_str="owner_b",
        expires_timestamp_ts=expires_timestamp_ts,
    ) is False


def test_state_store_roundtrips_pod_state(tmp_path):
    db_path_str = str((tmp_path / "live.sqlite3").resolve())
    state_store_obj = LiveStateStore(db_path_str)
    pod_state_obj = PodState(
        pod_id_str="pod_001",
        user_id_str="user_001",
        account_route_str="DU1",
        position_amount_map={"AAPL": 5.0},
        cash_float=2500.0,
        total_value_float=3000.0,
        strategy_state_dict={"trade_id_int": 3},
        updated_timestamp_ts=datetime(2024, 2, 1, 14, 25, tzinfo=UTC),
    )

    state_store_obj.upsert_pod_state(pod_state_obj)
    roundtrip_pod_state_obj = state_store_obj.get_pod_state("pod_001")

    assert roundtrip_pod_state_obj is not None
    assert roundtrip_pod_state_obj.position_amount_map == {"AAPL": 5.0}
    assert roundtrip_pod_state_obj.cash_float == 2500.0
    assert roundtrip_pod_state_obj.total_value_float == 3000.0
    assert roundtrip_pod_state_obj.strategy_state_dict == {"trade_id_int": 3}
