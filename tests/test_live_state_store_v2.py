from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from alpha.live.models import DecisionPlan, LiveRelease, VPlan, VPlanRow
from alpha.live.models import BrokerOrderEvent, BrokerOrderFill, BrokerOrderRecord, SessionOpenPrice
from alpha.live.state_store import V1_EXECUTION_TABLE_NAME_TUPLE
from alpha.live.state_store_v2 import LiveStateStore


MARKET_TIMEZONE_OBJ = ZoneInfo("America/New_York")


def test_state_store_v2_does_not_bootstrap_v1_execution_tables(tmp_path):
    db_path_str = str((tmp_path / "live.sqlite3").resolve())
    state_store_obj = LiveStateStore(db_path_str)

    existing_table_name_set = set(state_store_obj.get_existing_table_name_list())

    assert set(V1_EXECUTION_TABLE_NAME_TUPLE).isdisjoint(existing_table_name_set)


def test_state_store_v2_roundtrips_decision_plan_and_vplan(tmp_path):
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
        pod_budget_fraction_float=0.25,
        auto_submit_enabled_bool=False,
    )
    state_store_obj.upsert_release(release_obj)
    decision_plan_obj = state_store_obj.insert_decision_plan(
        DecisionPlan(
            release_id_str=release_obj.release_id_str,
            user_id_str=release_obj.user_id_str,
            pod_id_str=release_obj.pod_id_str,
            account_route_str=release_obj.account_route_str,
            signal_timestamp_ts=datetime(2024, 1, 31, 16, 0, tzinfo=MARKET_TIMEZONE_OBJ),
            submission_timestamp_ts=datetime(2024, 2, 1, 9, 20, tzinfo=MARKET_TIMEZONE_OBJ),
            target_execution_timestamp_ts=datetime(2024, 2, 1, 9, 30, tzinfo=MARKET_TIMEZONE_OBJ),
            execution_policy_str="next_open_moo",
            decision_base_position_map={},
            snapshot_metadata_dict={"strategy_family_str": "stub"},
            strategy_state_dict={"trade_id_int": 1},
            decision_book_type_str="incremental_entry_exit_book",
            entry_target_weight_map_dict={"AAPL": 0.2},
            target_weight_map={"AAPL": 0.2},
            exit_asset_set=set(),
            entry_priority_list=["AAPL"],
            cash_reserve_weight_float=0.0,
        )
    )
    inserted_vplan_obj = state_store_obj.insert_vplan(
        VPlan(
            release_id_str=release_obj.release_id_str,
            user_id_str=release_obj.user_id_str,
            pod_id_str=release_obj.pod_id_str,
            account_route_str=release_obj.account_route_str,
            decision_plan_id_int=int(decision_plan_obj.decision_plan_id_int or 0),
            signal_timestamp_ts=decision_plan_obj.signal_timestamp_ts,
            submission_timestamp_ts=decision_plan_obj.submission_timestamp_ts,
            target_execution_timestamp_ts=decision_plan_obj.target_execution_timestamp_ts,
            execution_policy_str="next_open_moo",
            broker_snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
            live_reference_snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
            live_price_source_str="stub",
            net_liq_float=10000.0,
            available_funds_float=8000.0,
            excess_liquidity_float=7000.0,
            pod_budget_fraction_float=0.25,
            pod_budget_float=2500.0,
            current_broker_position_map={},
            live_reference_price_map={"AAPL": 100.0},
            target_share_map={"AAPL": 5.0},
            order_delta_map={"AAPL": 5.0},
            vplan_row_list=[
                VPlanRow(
                    asset_str="AAPL",
                    current_share_float=0.0,
                    target_share_float=5.0,
                    order_delta_share_float=5.0,
                    live_reference_price_float=100.0,
                    estimated_target_notional_float=500.0,
                    broker_order_type_str="MOO",
                )
            ],
        )
    )

    latest_decision_plan_obj = state_store_obj.get_latest_decision_plan_for_pod("pod_001")
    latest_vplan_obj = state_store_obj.get_latest_vplan_for_pod("pod_001")

    assert latest_decision_plan_obj is not None
    assert latest_decision_plan_obj.decision_book_type_str == "incremental_entry_exit_book"
    assert latest_decision_plan_obj.entry_target_weight_map_dict == {"AAPL": 0.2}
    assert latest_decision_plan_obj.target_weight_map == {"AAPL": 0.2}
    assert inserted_vplan_obj.vplan_id_int is not None
    assert latest_vplan_obj is not None
    assert latest_vplan_obj.vplan_row_list[0].asset_str == "AAPL"


def test_state_store_v2_roundtrips_full_target_weight_decision_plan(tmp_path):
    db_path_str = str((tmp_path / "live_full_target.sqlite3").resolve())
    state_store_obj = LiveStateStore(db_path_str)
    decision_plan_obj = state_store_obj.insert_decision_plan(
        DecisionPlan(
            release_id_str="release_full_target_001",
            user_id_str="user_001",
            pod_id_str="pod_full_target_001",
            account_route_str="DU1",
            signal_timestamp_ts=datetime(2024, 1, 31, 16, 0, tzinfo=MARKET_TIMEZONE_OBJ),
            submission_timestamp_ts=datetime(2024, 2, 1, 9, 20, tzinfo=MARKET_TIMEZONE_OBJ),
            target_execution_timestamp_ts=datetime(2024, 2, 1, 9, 30, tzinfo=MARKET_TIMEZONE_OBJ),
            execution_policy_str="next_open_moo",
            decision_base_position_map={"AAPL": 2.0},
            snapshot_metadata_dict={"strategy_family_str": "taa_stub"},
            strategy_state_dict={},
            decision_book_type_str="full_target_weight_book",
            full_target_weight_map_dict={"AAPL": 0.6, "TLT": 0.3},
            target_weight_map={"AAPL": 0.6, "TLT": 0.3},
            cash_reserve_weight_float=0.1,
            preserve_untouched_positions_bool=False,
            rebalance_omitted_assets_to_zero_bool=True,
        )
    )

    roundtrip_decision_plan_obj = state_store_obj.get_decision_plan_by_id(
        int(decision_plan_obj.decision_plan_id_int or 0)
    )

    assert roundtrip_decision_plan_obj.decision_book_type_str == "full_target_weight_book"
    assert roundtrip_decision_plan_obj.full_target_weight_map_dict == {"AAPL": 0.6, "TLT": 0.3}
    assert roundtrip_decision_plan_obj.target_weight_map == {"AAPL": 0.6, "TLT": 0.3}
    assert roundtrip_decision_plan_obj.rebalance_omitted_assets_to_zero_bool is True


def test_state_store_v2_roundtrips_order_events_session_open_and_fill_enrichment(tmp_path):
    db_path_str = str((tmp_path / "live_events.sqlite3").resolve())
    state_store_obj = LiveStateStore(db_path_str)

    state_store_obj.upsert_vplan_broker_order_record_list(
        [
            BrokerOrderRecord(
                broker_order_id_str="broker_001",
                decision_plan_id_int=11,
                vplan_id_int=13,
                account_route_str="DU1",
                asset_str="AAPL",
                broker_order_type_str="MOO",
                unit_str="shares",
                amount_float=10.0,
                filled_amount_float=0.0,
                remaining_amount_float=10.0,
                avg_fill_price_float=None,
                status_str="PendingSubmit",
                last_status_timestamp_ts=datetime(2024, 2, 1, 9, 20, tzinfo=MARKET_TIMEZONE_OBJ),
                submitted_timestamp_ts=datetime(2024, 2, 1, 9, 20, tzinfo=MARKET_TIMEZONE_OBJ),
                raw_payload_dict={},
            )
        ]
    )
    state_store_obj.insert_vplan_broker_order_event_list(
        [
            BrokerOrderEvent(
                broker_order_id_str="broker_001",
                decision_plan_id_int=11,
                vplan_id_int=13,
                account_route_str="DU1",
                asset_str="AAPL",
                status_str="PendingSubmit",
                filled_amount_float=0.0,
                remaining_amount_float=10.0,
                avg_fill_price_float=None,
                event_timestamp_ts=datetime(2024, 2, 1, 9, 20, tzinfo=MARKET_TIMEZONE_OBJ),
                event_source_str="test",
                message_str="submitted",
                raw_payload_dict={},
            ),
            BrokerOrderEvent(
                broker_order_id_str="broker_001",
                decision_plan_id_int=11,
                vplan_id_int=13,
                account_route_str="DU1",
                asset_str="AAPL",
                status_str="PendingSubmit",
                filled_amount_float=0.0,
                remaining_amount_float=10.0,
                avg_fill_price_float=None,
                event_timestamp_ts=datetime(2024, 2, 1, 9, 20, tzinfo=MARKET_TIMEZONE_OBJ),
                event_source_str="test",
                message_str="submitted",
                raw_payload_dict={},
            ),
        ]
    )
    state_store_obj.upsert_session_open_price_list(
        [
            SessionOpenPrice(
                session_date_str="2024-02-01",
                account_route_str="DU1",
                asset_str="AAPL",
                official_open_price_float=101.0,
                open_price_source_str="ibkr.tick_open",
                snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 30, tzinfo=MARKET_TIMEZONE_OBJ),
                raw_payload_dict={},
            )
        ]
    )
    state_store_obj.upsert_vplan_fill_list(
        [
            BrokerOrderFill(
                broker_order_id_str="broker_001",
                decision_plan_id_int=11,
                vplan_id_int=13,
                account_route_str="DU1",
                asset_str="AAPL",
                fill_amount_float=10.0,
                fill_price_float=101.5,
                official_open_price_float=101.0,
                open_price_source_str="ibkr.tick_open",
                fill_timestamp_ts=datetime(2024, 2, 1, 9, 30, tzinfo=MARKET_TIMEZONE_OBJ),
                raw_payload_dict={},
            )
        ]
    )

    broker_order_row_dict_list = state_store_obj.get_broker_order_row_dict_list_for_vplan(13)
    fill_row_dict_list = state_store_obj.get_fill_row_dict_list_for_vplan(13)
    session_open_price_map_dict = state_store_obj.get_session_open_price_map_dict("DU1", "2024-02-01")

    assert broker_order_row_dict_list == [
        {
            "broker_order_id_str": "broker_001",
            "asset_str": "AAPL",
            "broker_order_type_str": "MOO",
            "unit_str": "shares",
            "amount_float": 10.0,
            "filled_amount_float": 0.0,
            "remaining_amount_float": 10.0,
            "avg_fill_price_float": None,
            "status_str": "PendingSubmit",
            "last_status_timestamp_str": "2024-02-01T09:20:00-05:00",
            "submitted_timestamp_str": "2024-02-01T09:20:00-05:00",
        }
    ]
    assert fill_row_dict_list == [
        {
            "asset_str": "AAPL",
            "fill_amount_float": 10.0,
            "fill_price_float": 101.5,
            "official_open_price_float": 101.0,
            "open_price_source_str": "ibkr.tick_open",
            "fill_timestamp_str": "2024-02-01T09:30:00-05:00",
        }
    ]
    assert session_open_price_map_dict["AAPL"].official_open_price_float == 101.0

    with state_store_obj._connect() as connection_obj:
        row_count_int = int(
            connection_obj.execute(
                "SELECT COUNT(1) AS row_count_int FROM vplan_broker_order_event"
            ).fetchone()["row_count_int"]
        )

    assert row_count_int == 1
