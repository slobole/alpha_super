from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from alpha.live.models import FrozenOrderIntent, FrozenOrderPlan, LiveRelease
from alpha.live.state_store import LiveStateStore


MARKET_TIMEZONE_OBJ = ZoneInfo("America/New_York")


def test_state_store_bootstraps_and_roundtrips_order_plan(tmp_path):
    db_path_str = str((tmp_path / "live.sqlite3").resolve())
    state_store_obj = LiveStateStore(db_path_str)
    release_obj = LiveRelease(
        release_id_str="release_001",
        user_id_str="user_001",
        pod_id_str="pod_001",
        account_route_str="U1",
        strategy_import_str="strategies.dv2.strategy_mr_dv2:DVO2Strategy",
        mode_str="paper",
        session_calendar_id_str="XNYS",
        signal_clock_str="eod_snapshot_ready",
        execution_policy_str="next_open_moo",
        data_profile_str="norgate_eod_sp500_pit",
        params_dict={},
        risk_profile_str="standard",
        enabled_bool=True,
        source_path_str="manifest.yaml",
    )
    state_store_obj.upsert_release(release_obj)
    order_plan_obj = FrozenOrderPlan(
        release_id_str=release_obj.release_id_str,
        user_id_str=release_obj.user_id_str,
        pod_id_str=release_obj.pod_id_str,
        account_route_str=release_obj.account_route_str,
        signal_timestamp_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        submission_timestamp_ts=datetime(2024, 2, 1, 9, 20, tzinfo=MARKET_TIMEZONE_OBJ),
        target_execution_timestamp_ts=datetime(2024, 2, 1, 9, 30, tzinfo=MARKET_TIMEZONE_OBJ),
        execution_policy_str="next_open_moo",
        snapshot_metadata_dict={"strategy_family_str": "test"},
        strategy_state_dict={"trade_id_int": 1},
        order_intent_list=[
            FrozenOrderIntent(
                asset_str="TEST",
                order_class_str="MarketOrder",
                unit_str="shares",
                amount_float=5.0,
                target_bool=False,
                trade_id_int=1,
                broker_order_type_str="MOO",
                sizing_reference_price_float=10.0,
                portfolio_value_float=1000.0,
            )
        ],
    )

    inserted_plan_obj = state_store_obj.insert_order_plan(order_plan_obj)
    submittable_plan_list = state_store_obj.get_submittable_order_plan_list(
        datetime(2024, 2, 1, 9, 20, tzinfo=MARKET_TIMEZONE_OBJ)
    )

    assert inserted_plan_obj.order_plan_id_int is not None
    assert inserted_plan_obj.submission_key_str == f"order_plan:{inserted_plan_obj.order_plan_id_int}"
    assert len(submittable_plan_list) == 1
    assert submittable_plan_list[0].order_intent_list[0].asset_str == "TEST"


def test_state_store_claims_order_plan_and_scheduler_lease(tmp_path):
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


def test_state_store_submittable_plan_filter_uses_real_timestamps_not_text_order(tmp_path):
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
        params_dict={},
        risk_profile_str="standard",
        enabled_bool=True,
        source_path_str="manifest.yaml",
    )
    state_store_obj.upsert_release(release_obj)
    order_plan_obj = FrozenOrderPlan(
        release_id_str=release_obj.release_id_str,
        user_id_str=release_obj.user_id_str,
        pod_id_str=release_obj.pod_id_str,
        account_route_str=release_obj.account_route_str,
        signal_timestamp_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        submission_timestamp_ts=datetime(2024, 2, 1, 9, 20, tzinfo=MARKET_TIMEZONE_OBJ),
        target_execution_timestamp_ts=datetime(2024, 2, 1, 9, 30, tzinfo=MARKET_TIMEZONE_OBJ),
        execution_policy_str="next_open_moo",
        snapshot_metadata_dict={"strategy_family_str": "test"},
        strategy_state_dict={"trade_id_int": 1},
        order_intent_list=[
            FrozenOrderIntent(
                asset_str="TEST",
                order_class_str="MarketOrder",
                unit_str="shares",
                amount_float=5.0,
                target_bool=False,
                trade_id_int=1,
                broker_order_type_str="MOO",
                sizing_reference_price_float=10.0,
                portfolio_value_float=1000.0,
            )
        ],
    )

    state_store_obj.insert_order_plan(order_plan_obj)
    as_of_ts = datetime(2024, 2, 1, 13, 6, tzinfo=UTC)
    submittable_plan_list = state_store_obj.get_submittable_order_plan_list(as_of_ts)

    assert submittable_plan_list == []
