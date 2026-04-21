from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

import pandas as pd

from alpha.data import FredSeriesSnapshot, FredSeriesStaleError
import alpha.live.logging_utils as logging_utils
from alpha.live.models import (
    BrokerOrderAck,
    BrokerOrderEvent,
    BrokerOrderFill,
    BrokerOrderRecord,
    DecisionPlan,
    LiveRelease,
    SubmitBatchResult,
    VPlan,
    VPlanRow,
)
from alpha.live.order_clerk import StubBrokerAdapter
from alpha.live.release_manifest import load_release_list
from alpha.live.runner import (
    BrokerAdapterResolver,
    _build_execution_row_dict_list,
    _enrich_broker_order_row_dict_list_for_display,
    _enrich_fill_row_dict_list,
    _render_command_output_str,
    _render_status_detail_str,
    build_decision_plans,
    build_vplans,
    cutover_v1_schema,
    get_execution_report_summary,
    preflight_contract_summary,
    get_status_summary,
    post_execution_reconcile,
    show_vplan_summary,
    submit_ready_vplans,
    tick,
)
from alpha.live.state_store_v2 import LiveStateStore


MARKET_TIMEZONE_OBJ = ZoneInfo("America/New_York")


def _write_manifest_file(
    root_path_obj: Path,
    file_name_str: str,
    pod_id_str: str,
    release_id_str: str,
    strategy_import_str: str,
    auto_submit_enabled_bool: bool,
    pod_budget_fraction_float: float = 0.5,
    account_route_str: str = "DU1",
    broker_host_str: str = "127.0.0.1",
    broker_port_int: int = 7497,
    broker_client_id_int: int = 31,
    broker_timeout_seconds_float: float = 4.0,
) -> None:
    releases_root_path_obj = root_path_obj / "releases" / "user_001"
    releases_root_path_obj.mkdir(parents=True, exist_ok=True)
    (releases_root_path_obj / file_name_str).write_text(
        "\n".join(
            [
                "identity:",
                f"  release_id: {release_id_str}",
                "  user_id: user_001",
                f"  pod_id: {pod_id_str}",
                "deployment:",
                "  mode: paper",
                "  enabled_bool: true",
                "broker:",
                f"  account_route: {account_route_str}",
                f"  host_str: {broker_host_str}",
                f"  port_int: {int(broker_port_int)}",
                f"  client_id_int: {int(broker_client_id_int)}",
                f"  timeout_seconds_float: {float(broker_timeout_seconds_float)}",
                "strategy:",
                f"  strategy_import_str: {strategy_import_str}",
                "  data_profile_str: norgate_eod_sp500_pit",
                "  params: {}",
                "market:",
                "  session_calendar_id_str: XNYS",
                "schedule:",
                "  signal_clock_str: eod_snapshot_ready",
                "  execution_policy_str: next_open_moo",
                "execution:",
                f"  pod_budget_fraction_float: {pod_budget_fraction_float}",
                f"  auto_submit_enabled_bool: {'true' if auto_submit_enabled_bool else 'false'}",
                "bootstrap:",
                "  initial_cash_float: 10000.0",
                "risk:",
                "  risk_profile_str: standard",
            ]
        ),
        encoding="utf-8",
    )


def _write_manifest(
    root_path_obj: Path,
    auto_submit_enabled_bool: bool,
    pod_budget_fraction_float: float = 0.5,
    account_route_str: str = "DU1",
    broker_host_str: str = "127.0.0.1",
    broker_port_int: int = 7497,
    broker_client_id_int: int = 31,
    broker_timeout_seconds_float: float = 4.0,
) -> None:
    _write_manifest_file(
        root_path_obj=root_path_obj,
        file_name_str="pod_test.yaml",
        pod_id_str="pod_test_01",
        release_id_str="user_001.pod_test.daily.v2",
        strategy_import_str="strategies.dv2.strategy_mr_dv2:DVO2Strategy",
        auto_submit_enabled_bool=auto_submit_enabled_bool,
        pod_budget_fraction_float=pod_budget_fraction_float,
        account_route_str=account_route_str,
        broker_host_str=broker_host_str,
        broker_port_int=broker_port_int,
        broker_client_id_int=broker_client_id_int,
        broker_timeout_seconds_float=broker_timeout_seconds_float,
    )


def _build_decision_plan_stub(release_obj, as_of_ts, pod_state_obj):
    return DecisionPlan(
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
        preserve_untouched_positions_bool=True,
    )


def _build_decision_plan_with_position_drift_stub(release_obj, as_of_ts, pod_state_obj):
    return DecisionPlan(
        release_id_str=release_obj.release_id_str,
        user_id_str=release_obj.user_id_str,
        pod_id_str=release_obj.pod_id_str,
        account_route_str=release_obj.account_route_str,
        signal_timestamp_ts=datetime(2024, 1, 31, 16, 0, tzinfo=MARKET_TIMEZONE_OBJ),
        submission_timestamp_ts=datetime(2024, 2, 1, 9, 20, tzinfo=MARKET_TIMEZONE_OBJ),
        target_execution_timestamp_ts=datetime(2024, 2, 1, 9, 30, tzinfo=MARKET_TIMEZONE_OBJ),
        execution_policy_str="next_open_moo",
        decision_base_position_map={"AAPL": 5.0},
        snapshot_metadata_dict={"strategy_family_str": "stub"},
        strategy_state_dict={"trade_id_int": 1},
        decision_book_type_str="incremental_entry_exit_book",
        entry_target_weight_map_dict={"AAPL": 0.2},
        target_weight_map={"AAPL": 0.2},
        exit_asset_set=set(),
        entry_priority_list=["AAPL"],
        cash_reserve_weight_float=0.0,
        preserve_untouched_positions_bool=True,
    )


def _build_full_target_weight_decision_plan_stub(release_obj, as_of_ts, pod_state_obj):
    return DecisionPlan(
        release_id_str=release_obj.release_id_str,
        user_id_str=release_obj.user_id_str,
        pod_id_str=release_obj.pod_id_str,
        account_route_str=release_obj.account_route_str,
        signal_timestamp_ts=datetime(2024, 1, 31, 16, 0, tzinfo=MARKET_TIMEZONE_OBJ),
        submission_timestamp_ts=datetime(2024, 2, 1, 9, 20, tzinfo=MARKET_TIMEZONE_OBJ),
        target_execution_timestamp_ts=datetime(2024, 2, 1, 9, 30, tzinfo=MARKET_TIMEZONE_OBJ),
        execution_policy_str="next_open_moo",
        decision_base_position_map={"AAPL": 4.0},
        snapshot_metadata_dict={"strategy_family_str": "full_target_stub"},
        strategy_state_dict={},
        decision_book_type_str="full_target_weight_book",
        full_target_weight_map_dict={"AAPL": 0.3, "TLT": 0.2},
        target_weight_map={"AAPL": 0.3, "TLT": 0.2},
        cash_reserve_weight_float=0.5,
        preserve_untouched_positions_bool=False,
        rebalance_omitted_assets_to_zero_bool=True,
    )


def _build_two_asset_decision_plan_stub(release_obj, as_of_ts, pod_state_obj):
    return DecisionPlan(
        release_id_str=release_obj.release_id_str,
        user_id_str=release_obj.user_id_str,
        pod_id_str=release_obj.pod_id_str,
        account_route_str=release_obj.account_route_str,
        signal_timestamp_ts=datetime(2024, 1, 31, 16, 0, tzinfo=MARKET_TIMEZONE_OBJ),
        submission_timestamp_ts=datetime(2024, 2, 1, 9, 20, tzinfo=MARKET_TIMEZONE_OBJ),
        target_execution_timestamp_ts=datetime(2024, 2, 1, 9, 30, tzinfo=MARKET_TIMEZONE_OBJ),
        execution_policy_str="next_open_moo",
        decision_base_position_map={},
        snapshot_metadata_dict={"strategy_family_str": "two_asset_stub"},
        strategy_state_dict={"trade_id_int": 1},
        decision_book_type_str="incremental_entry_exit_book",
        entry_target_weight_map_dict={"AAPL": 0.2, "MSFT": 0.2},
        target_weight_map={"AAPL": 0.2, "MSFT": 0.2},
        exit_asset_set=set(),
        entry_priority_list=["AAPL", "MSFT"],
        cash_reserve_weight_float=0.0,
        preserve_untouched_positions_bool=True,
    )


def _build_three_asset_decision_plan_stub(release_obj, as_of_ts, pod_state_obj):
    return DecisionPlan(
        release_id_str=release_obj.release_id_str,
        user_id_str=release_obj.user_id_str,
        pod_id_str=release_obj.pod_id_str,
        account_route_str=release_obj.account_route_str,
        signal_timestamp_ts=datetime(2024, 1, 31, 16, 0, tzinfo=MARKET_TIMEZONE_OBJ),
        submission_timestamp_ts=datetime(2024, 2, 1, 9, 20, tzinfo=MARKET_TIMEZONE_OBJ),
        target_execution_timestamp_ts=datetime(2024, 2, 1, 9, 30, tzinfo=MARKET_TIMEZONE_OBJ),
        execution_policy_str="next_open_moo",
        decision_base_position_map={},
        snapshot_metadata_dict={"strategy_family_str": "three_asset_stub"},
        strategy_state_dict={"trade_id_int": 1},
        decision_book_type_str="incremental_entry_exit_book",
        entry_target_weight_map_dict={"AAPL": 0.2, "MSFT": 0.2, "NVDA": 0.2},
        target_weight_map={"AAPL": 0.2, "MSFT": 0.2, "NVDA": 0.2},
        exit_asset_set=set(),
        entry_priority_list=["AAPL", "MSFT", "NVDA"],
        cash_reserve_weight_float=0.0,
        preserve_untouched_positions_bool=True,
    )


def _build_exit_only_decision_plan_stub(release_obj, as_of_ts, pod_state_obj):
    return DecisionPlan(
        release_id_str=release_obj.release_id_str,
        user_id_str=release_obj.user_id_str,
        pod_id_str=release_obj.pod_id_str,
        account_route_str=release_obj.account_route_str,
        signal_timestamp_ts=datetime(2024, 1, 31, 16, 0, tzinfo=MARKET_TIMEZONE_OBJ),
        submission_timestamp_ts=datetime(2024, 2, 1, 9, 20, tzinfo=MARKET_TIMEZONE_OBJ),
        target_execution_timestamp_ts=datetime(2024, 2, 1, 9, 30, tzinfo=MARKET_TIMEZONE_OBJ),
        execution_policy_str="next_open_moo",
        decision_base_position_map={"AAPL": 5.0},
        snapshot_metadata_dict={"strategy_family_str": "exit_only_stub"},
        strategy_state_dict={"trade_id_int": 2},
        decision_book_type_str="incremental_entry_exit_book",
        entry_target_weight_map_dict={},
        target_weight_map={},
        exit_asset_set={"AAPL"},
        entry_priority_list=[],
        cash_reserve_weight_float=1.0,
        preserve_untouched_positions_bool=True,
    )


def _install_pending_submit_truth_stub(
    broker_adapter_obj: StubBrokerAdapter,
    final_truth_by_asset_map_dict: dict[str, dict[str, float | str | None]],
) -> None:
    broker_order_index_int = 0

    def _submit_order_request_list(account_route_str, broker_order_request_list, submitted_timestamp_ts):
        nonlocal broker_order_index_int
        broker_order_record_list: list[BrokerOrderRecord] = []
        broker_order_event_list: list[BrokerOrderEvent] = []
        for broker_order_request_obj in broker_order_request_list:
            broker_order_index_int += 1
            broker_order_id_str = f"stub_order_{broker_order_index_int}"
            requested_share_float = abs(float(broker_order_request_obj.amount_float))
            pending_broker_order_record_obj = BrokerOrderRecord(
                broker_order_id_str=broker_order_id_str,
                decision_plan_id_int=broker_order_request_obj.decision_plan_id_int,
                vplan_id_int=broker_order_request_obj.vplan_id_int,
                account_route_str=account_route_str,
                asset_str=broker_order_request_obj.asset_str,
                order_request_key_str=broker_order_request_obj.order_request_key_str,
                broker_order_type_str=broker_order_request_obj.broker_order_type_str,
                unit_str=broker_order_request_obj.unit_str,
                amount_float=float(broker_order_request_obj.amount_float),
                filled_amount_float=0.0,
                remaining_amount_float=requested_share_float,
                avg_fill_price_float=None,
                status_str="PendingSubmit",
                submitted_timestamp_ts=submitted_timestamp_ts,
                last_status_timestamp_ts=submitted_timestamp_ts,
                submission_key_str=broker_order_request_obj.submission_key_str,
                raw_payload_dict={
                    "submission_key_str": broker_order_request_obj.submission_key_str,
                    "order_request_key_str": broker_order_request_obj.order_request_key_str,
                },
            )
            pending_broker_order_event_obj = BrokerOrderEvent(
                broker_order_id_str=broker_order_id_str,
                decision_plan_id_int=broker_order_request_obj.decision_plan_id_int,
                vplan_id_int=broker_order_request_obj.vplan_id_int,
                account_route_str=account_route_str,
                asset_str=broker_order_request_obj.asset_str,
                order_request_key_str=broker_order_request_obj.order_request_key_str,
                status_str="PendingSubmit",
                filled_amount_float=0.0,
                remaining_amount_float=requested_share_float,
                avg_fill_price_float=None,
                event_timestamp_ts=submitted_timestamp_ts,
                event_source_str="test.submit",
                message_str="submitted",
                submission_key_str=broker_order_request_obj.submission_key_str,
                raw_payload_dict={
                    "order_request_key_str": broker_order_request_obj.order_request_key_str,
                },
            )
            final_truth_dict = final_truth_by_asset_map_dict[broker_order_request_obj.asset_str]
            final_filled_amount_float = float(final_truth_dict.get("filled_amount_float", 0.0) or 0.0)
            final_remaining_amount_float = final_truth_dict.get("remaining_amount_float")
            if final_remaining_amount_float is None:
                final_remaining_amount_float = max(requested_share_float - abs(final_filled_amount_float), 0.0)
            final_fill_price_float = final_truth_dict.get("fill_price_float")
            final_broker_order_record_obj = BrokerOrderRecord(
                broker_order_id_str=broker_order_id_str,
                decision_plan_id_int=broker_order_request_obj.decision_plan_id_int,
                vplan_id_int=broker_order_request_obj.vplan_id_int,
                account_route_str=account_route_str,
                asset_str=broker_order_request_obj.asset_str,
                order_request_key_str=broker_order_request_obj.order_request_key_str,
                broker_order_type_str=broker_order_request_obj.broker_order_type_str,
                unit_str=broker_order_request_obj.unit_str,
                amount_float=float(broker_order_request_obj.amount_float),
                filled_amount_float=final_filled_amount_float,
                remaining_amount_float=float(final_remaining_amount_float),
                avg_fill_price_float=None if final_fill_price_float is None else float(final_fill_price_float),
                status_str=str(final_truth_dict["status_str"]),
                submitted_timestamp_ts=submitted_timestamp_ts,
                last_status_timestamp_ts=submitted_timestamp_ts,
                submission_key_str=broker_order_request_obj.submission_key_str,
                raw_payload_dict={
                    "submission_key_str": broker_order_request_obj.submission_key_str,
                    "order_request_key_str": broker_order_request_obj.order_request_key_str,
                },
            )
            final_broker_order_fill_list: list[BrokerOrderFill] = []
            if abs(final_filled_amount_float) > 0.0 and final_fill_price_float is not None:
                final_broker_order_fill_list.append(
                    BrokerOrderFill(
                        broker_order_id_str=broker_order_id_str,
                        decision_plan_id_int=broker_order_request_obj.decision_plan_id_int,
                        vplan_id_int=broker_order_request_obj.vplan_id_int,
                        account_route_str=account_route_str,
                        asset_str=broker_order_request_obj.asset_str,
                        fill_amount_float=(
                            final_filled_amount_float
                            if float(broker_order_request_obj.amount_float) >= 0.0
                            else -abs(final_filled_amount_float)
                        ),
                        fill_price_float=float(final_fill_price_float),
                        fill_timestamp_ts=submitted_timestamp_ts,
                        raw_payload_dict={},
                    )
                )
            broker_adapter_obj.seed_broker_order_state(
                final_broker_order_record_obj,
                broker_order_event_list=[],
                broker_order_fill_list=final_broker_order_fill_list,
            )
            broker_order_record_list.append(pending_broker_order_record_obj)
            broker_order_event_list.append(pending_broker_order_event_obj)
        broker_order_ack_list = [
            BrokerOrderAck(
                decision_plan_id_int=broker_order_request_obj.decision_plan_id_int,
                vplan_id_int=broker_order_request_obj.vplan_id_int,
                account_route_str=account_route_str,
                order_request_key_str=broker_order_request_obj.order_request_key_str,
                asset_str=broker_order_request_obj.asset_str,
                broker_order_type_str=broker_order_request_obj.broker_order_type_str,
                local_submit_ack_bool=True,
                broker_response_ack_bool=(
                    broker_order_request_obj.asset_str in final_truth_by_asset_map_dict
                ),
                ack_status_str="broker_acked",
                ack_source_str="test.submit_stub",
                broker_order_id_str=f"stub_order_{idx_int}",
                response_timestamp_ts=submitted_timestamp_ts,
                raw_payload_dict={},
            )
            for idx_int, broker_order_request_obj in enumerate(broker_order_request_list, start=1)
        ]
        return SubmitBatchResult(
            broker_order_record_list=broker_order_record_list,
            broker_order_event_list=broker_order_event_list,
            broker_order_fill_list=[],
            broker_order_ack_list=broker_order_ack_list,
            ack_coverage_ratio_float=1.0 if len(broker_order_ack_list) > 0 else 1.0,
            missing_ack_asset_list=[],
            submit_ack_status_str="complete",
        )

    broker_adapter_obj.submit_order_request_list = _submit_order_request_list


def test_broker_adapter_resolver_uses_release_broker_fields_and_reuses_cache():
    factory_call_list: list[tuple[str, int, int, float]] = []

    def _adapter_factory(host_str, port_int, client_id_int, timeout_seconds_float):
        factory_call_list.append((host_str, port_int, client_id_int, timeout_seconds_float))
        return StubBrokerAdapter()

    release_a_obj = LiveRelease(
        release_id_str="release_a",
        user_id_str="user_001",
        pod_id_str="pod_a",
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
        source_path_str="manifest_a.yaml",
        broker_host_str="127.0.0.1",
        broker_port_int=7496,
        broker_client_id_int=31,
        broker_timeout_seconds_float=4.0,
    )
    release_b_obj = LiveRelease(
        release_id_str="release_b",
        user_id_str="user_001",
        pod_id_str="pod_b",
        account_route_str="DU2",
        strategy_import_str="strategies.dv2.strategy_mr_dv2:DVO2Strategy",
        mode_str="paper",
        session_calendar_id_str="XNYS",
        signal_clock_str="eod_snapshot_ready",
        execution_policy_str="next_open_moo",
        data_profile_str="norgate_eod_sp500_pit",
        params_dict={},
        risk_profile_str="standard",
        enabled_bool=True,
        source_path_str="manifest_b.yaml",
        broker_host_str="127.0.0.1",
        broker_port_int=7496,
        broker_client_id_int=31,
        broker_timeout_seconds_float=4.0,
    )

    resolver_obj = BrokerAdapterResolver(adapter_factory_func=_adapter_factory)

    assert resolver_obj.get_adapter(release_a_obj) is resolver_obj.get_adapter(release_b_obj)
    assert factory_call_list == [("127.0.0.1", 7496, 31, 4.0)]


def test_broker_adapter_resolver_cli_override_wins_over_manifest_fields():
    release_obj = LiveRelease(
        release_id_str="release_a",
        user_id_str="user_001",
        pod_id_str="pod_a",
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
        source_path_str="manifest_a.yaml",
        broker_host_str="127.0.0.1",
        broker_port_int=7496,
        broker_client_id_int=31,
        broker_timeout_seconds_float=4.0,
    )

    resolver_obj = BrokerAdapterResolver(
        broker_host_str="10.0.0.5",
        broker_port_int=4001,
        broker_client_id_int=55,
        broker_timeout_seconds_float=9.0,
    )

    assert resolver_obj.get_connection_field_map_dict(release_obj) == {
        "broker_host_str": "10.0.0.5",
        "broker_port_int": 4001,
        "broker_client_id_int": 55,
        "broker_timeout_seconds_float": 9.0,
    }


def test_build_vplans_uses_per_pod_broker_config_without_global_port_leakage(tmp_path: Path):
    _write_manifest_file(
        root_path_obj=tmp_path,
        file_name_str="pod_a.yaml",
        pod_id_str="pod_a",
        release_id_str="user_001.pod_a.daily.v1",
        strategy_import_str="strategies.dv2.strategy_mr_dv2:DVO2Strategy",
        auto_submit_enabled_bool=False,
        account_route_str="DU1",
        broker_port_int=7496,
        broker_client_id_int=31,
    )
    _write_manifest_file(
        root_path_obj=tmp_path,
        file_name_str="pod_b.yaml",
        pod_id_str="pod_b",
        release_id_str="user_001.pod_b.daily.v1",
        strategy_import_str="strategies.dv2.strategy_mr_dv2:DVO2Strategy",
        auto_submit_enabled_bool=False,
        account_route_str="DU2",
        broker_port_int=7498,
        broker_client_id_int=41,
    )
    state_store_obj = LiveStateStore(str((tmp_path / "live.sqlite3").resolve()))
    release_list = load_release_list(str(tmp_path / "releases"))
    state_store_obj.upsert_release_list(release_list)
    release_by_pod_id_map_dict = {
        str(release_obj.pod_id_str): release_obj for release_obj in release_list
    }
    state_store_obj.insert_decision_plan(
        _build_decision_plan_stub(
            release_by_pod_id_map_dict["pod_a"],
            datetime(2024, 2, 1, 9, 0, tzinfo=MARKET_TIMEZONE_OBJ),
            None,
        )
    )
    state_store_obj.insert_decision_plan(
        _build_decision_plan_stub(
            release_by_pod_id_map_dict["pod_b"],
            datetime(2024, 2, 1, 9, 0, tzinfo=MARKET_TIMEZONE_OBJ),
            None,
        )
    )
    adapter_by_key_tup_dict: dict[tuple[str, int, int, float], StubBrokerAdapter] = {}

    def _adapter_factory(host_str, port_int, client_id_int, timeout_seconds_float):
        adapter_obj = StubBrokerAdapter()
        adapter_obj.seed_account_snapshot(
            account_route_str="DU1" if int(port_int) == 7496 else "DU2",
            cash_float=10000.0,
            total_value_float=10000.0,
            session_mode_str="paper",
        )
        adapter_obj.seed_live_price_snapshot(
            account_route_str="DU1" if int(port_int) == 7496 else "DU2",
            asset_reference_price_map={"AAPL": 100.0},
            price_source_str=f"stub:{int(port_int)}",
        )
        adapter_by_key_tup_dict[(host_str, int(port_int), int(client_id_int), float(timeout_seconds_float))] = adapter_obj
        return adapter_obj

    detail_dict = build_vplans(
        state_store_obj=state_store_obj,
        broker_adapter_obj=None,
        as_of_ts=datetime(2024, 2, 1, 9, 25, tzinfo=MARKET_TIMEZONE_OBJ),
        env_mode_str="paper",
        adapter_factory_func=_adapter_factory,
    )

    assert detail_dict["created_vplan_count_int"] == 2
    assert sorted(adapter_by_key_tup_dict) == [
        ("127.0.0.1", 7496, 31, 4.0),
        ("127.0.0.1", 7498, 41, 4.0),
    ]
    latest_vplan_a_obj = state_store_obj.get_latest_vplan_for_pod("pod_a")
    latest_vplan_b_obj = state_store_obj.get_latest_vplan_for_pod("pod_b")
    assert latest_vplan_a_obj is not None
    assert latest_vplan_b_obj is not None
    assert latest_vplan_a_obj.live_price_source_str == "stub:7496"
    assert latest_vplan_b_obj.live_price_source_str == "stub:7498"


def _install_partial_submit_ack_stub(
    broker_adapter_obj: StubBrokerAdapter,
    broker_ack_asset_set: set[str],
) -> None:
    broker_order_index_int = 0

    def _submit_order_request_list(account_route_str, broker_order_request_list, submitted_timestamp_ts):
        nonlocal broker_order_index_int
        broker_order_record_list: list[BrokerOrderRecord] = []
        broker_order_event_list: list[BrokerOrderEvent] = []
        broker_order_ack_list: list[BrokerOrderAck] = []
        for broker_order_request_obj in broker_order_request_list:
            broker_order_index_int += 1
            broker_order_id_str = f"partial_ack_order_{broker_order_index_int}"
            requested_share_float = abs(float(broker_order_request_obj.amount_float))
            broker_order_record_list.append(
                BrokerOrderRecord(
                    broker_order_id_str=broker_order_id_str,
                    decision_plan_id_int=broker_order_request_obj.decision_plan_id_int,
                    vplan_id_int=broker_order_request_obj.vplan_id_int,
                    account_route_str=account_route_str,
                    asset_str=broker_order_request_obj.asset_str,
                    order_request_key_str=broker_order_request_obj.order_request_key_str,
                    broker_order_type_str=broker_order_request_obj.broker_order_type_str,
                    unit_str=broker_order_request_obj.unit_str,
                    amount_float=float(broker_order_request_obj.amount_float),
                    filled_amount_float=0.0,
                    remaining_amount_float=requested_share_float,
                    avg_fill_price_float=None,
                    status_str="PendingSubmit",
                    submitted_timestamp_ts=submitted_timestamp_ts,
                    last_status_timestamp_ts=submitted_timestamp_ts,
                    submission_key_str=broker_order_request_obj.submission_key_str,
                    raw_payload_dict={
                        "submission_key_str": broker_order_request_obj.submission_key_str,
                        "order_request_key_str": broker_order_request_obj.order_request_key_str,
                    },
                )
            )
            broker_order_event_list.append(
                BrokerOrderEvent(
                    broker_order_id_str=broker_order_id_str,
                    decision_plan_id_int=broker_order_request_obj.decision_plan_id_int,
                    vplan_id_int=broker_order_request_obj.vplan_id_int,
                    account_route_str=account_route_str,
                    asset_str=broker_order_request_obj.asset_str,
                    order_request_key_str=broker_order_request_obj.order_request_key_str,
                    status_str="PendingSubmit",
                    filled_amount_float=0.0,
                    remaining_amount_float=requested_share_float,
                    avg_fill_price_float=None,
                    event_timestamp_ts=submitted_timestamp_ts,
                    event_source_str="test.submit",
                    message_str="submitted",
                    submission_key_str=broker_order_request_obj.submission_key_str,
                    raw_payload_dict={
                        "order_request_key_str": broker_order_request_obj.order_request_key_str,
                    },
                )
            )
            broker_order_ack_bool = broker_order_request_obj.asset_str in broker_ack_asset_set
            broker_order_ack_list.append(
                BrokerOrderAck(
                    decision_plan_id_int=broker_order_request_obj.decision_plan_id_int,
                    vplan_id_int=broker_order_request_obj.vplan_id_int,
                    account_route_str=account_route_str,
                    order_request_key_str=broker_order_request_obj.order_request_key_str,
                    asset_str=broker_order_request_obj.asset_str,
                    broker_order_type_str=broker_order_request_obj.broker_order_type_str,
                    local_submit_ack_bool=True,
                    broker_response_ack_bool=broker_order_ack_bool,
                    ack_status_str=(
                        "broker_acked" if broker_order_ack_bool else "missing_critical"
                    ),
                    ack_source_str=("test.stub" if broker_order_ack_bool else "missing"),
                    broker_order_id_str=(broker_order_id_str if broker_order_ack_bool else None),
                    response_timestamp_ts=(submitted_timestamp_ts if broker_order_ack_bool else None),
                    raw_payload_dict={},
                )
            )
        return SubmitBatchResult(
            broker_order_record_list=broker_order_record_list,
            broker_order_event_list=broker_order_event_list,
            broker_order_fill_list=[],
            broker_order_ack_list=broker_order_ack_list,
            ack_coverage_ratio_float=float(len(broker_ack_asset_set)) / float(len(broker_order_request_list)),
            missing_ack_asset_list=sorted(
                {
                    broker_order_request_obj.asset_str
                    for broker_order_request_obj in broker_order_request_list
                    if broker_order_request_obj.asset_str not in broker_ack_asset_set
                }
            ),
            submit_ack_status_str=(
                "complete"
                if len(broker_ack_asset_set) == len(broker_order_request_list)
                else "missing_critical"
            ),
        )

    broker_adapter_obj.submit_order_request_list = _submit_order_request_list


def test_enrich_fill_row_dict_list_computes_cost_bps_signs():
    enriched_fill_row_dict_list = _enrich_fill_row_dict_list(
        [
            {
                "asset_str": "BUY_WORSE",
                "fill_amount_float": 1.0,
                "fill_price_float": 101.0,
                "official_open_price_float": 100.0,
                "open_price_source_str": "test",
                "fill_timestamp_str": "2024-02-01T09:30:00-05:00",
            },
            {
                "asset_str": "BUY_BETTER",
                "fill_amount_float": 1.0,
                "fill_price_float": 99.0,
                "official_open_price_float": 100.0,
                "open_price_source_str": "test",
                "fill_timestamp_str": "2024-02-01T09:30:00-05:00",
            },
            {
                "asset_str": "SELL_WORSE",
                "fill_amount_float": -1.0,
                "fill_price_float": 99.0,
                "official_open_price_float": 100.0,
                "open_price_source_str": "test",
                "fill_timestamp_str": "2024-02-01T09:30:00-05:00",
            },
            {
                "asset_str": "SELL_BETTER",
                "fill_amount_float": -1.0,
                "fill_price_float": 101.0,
                "official_open_price_float": 100.0,
                "open_price_source_str": "test",
                "fill_timestamp_str": "2024-02-01T09:30:00-05:00",
            },
        ]
    )

    fill_slippage_bps_by_asset_map_dict = {
        fill_row_dict["asset_str"]: fill_row_dict["fill_slippage_bps_float"]
        for fill_row_dict in enriched_fill_row_dict_list
    }

    assert fill_slippage_bps_by_asset_map_dict == {
        "BUY_WORSE": 100.0,
        "BUY_BETTER": -100.0,
        "SELL_WORSE": 100.0,
        "SELL_BETTER": -100.0,
    }


def test_execution_quality_falls_back_to_weighted_fill_average_when_broker_avg_missing():
    vplan_obj = VPlan(
        release_id_str="release_001",
        user_id_str="user_001",
        pod_id_str="pod_001",
        account_route_str="DU1",
        decision_plan_id_int=7,
        signal_timestamp_ts=datetime(2024, 1, 31, 16, 0, tzinfo=MARKET_TIMEZONE_OBJ),
        submission_timestamp_ts=datetime(2024, 2, 1, 9, 20, tzinfo=MARKET_TIMEZONE_OBJ),
        target_execution_timestamp_ts=datetime(2024, 2, 1, 9, 30, tzinfo=MARKET_TIMEZONE_OBJ),
        execution_policy_str="next_open_moo",
        broker_snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 20, tzinfo=MARKET_TIMEZONE_OBJ),
        live_reference_snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 20, tzinfo=MARKET_TIMEZONE_OBJ),
        live_price_source_str="stub",
        net_liq_float=10000.0,
        available_funds_float=9000.0,
        excess_liquidity_float=9000.0,
        pod_budget_fraction_float=0.5,
        pod_budget_float=5000.0,
        current_broker_position_map={},
        live_reference_price_map={"AAPL": 100.0},
        target_share_map={"AAPL": 40.0},
        order_delta_map={"AAPL": 40.0},
        vplan_row_list=[
            VPlanRow(
                asset_str="AAPL",
                current_share_float=0.0,
                target_share_float=40.0,
                order_delta_share_float=40.0,
                live_reference_price_float=100.0,
                estimated_target_notional_float=4000.0,
                broker_order_type_str="MOO",
            )
        ],
        vplan_id_int=9,
    )
    fill_row_dict_list = _enrich_fill_row_dict_list(
        [
            {
                "asset_str": "AAPL",
                "fill_amount_float": 10.0,
                "fill_price_float": 101.0,
                "official_open_price_float": 100.0,
                "open_price_source_str": "test",
                "fill_timestamp_str": "2024-02-01T09:30:00-05:00",
            },
            {
                "asset_str": "AAPL",
                "fill_amount_float": 30.0,
                "fill_price_float": 103.0,
                "official_open_price_float": 100.0,
                "open_price_source_str": "test",
                "fill_timestamp_str": "2024-02-01T09:31:00-05:00",
            },
        ]
    )
    broker_order_row_dict_list = [
        {
            "broker_order_id_str": "broker_001",
            "asset_str": "AAPL",
            "broker_order_type_str": "MOO",
            "unit_str": "shares",
            "amount_float": 40.0,
            "filled_amount_float": 40.0,
            "remaining_amount_float": 0.0,
            "avg_fill_price_float": None,
            "status_str": "Filled",
            "last_status_timestamp_str": "2024-02-01T09:31:00-05:00",
            "submitted_timestamp_str": "2024-02-01T09:20:00-05:00",
            "submission_key_str": "vplan:9",
        }
    ]

    enriched_broker_order_row_dict_list = _enrich_broker_order_row_dict_list_for_display(
        broker_order_row_dict_list=broker_order_row_dict_list,
        fill_row_dict_list=fill_row_dict_list,
    )
    execution_row_dict_list = _build_execution_row_dict_list(
        vplan_obj=vplan_obj,
        broker_position_map_dict={"AAPL": 40.0},
        broker_order_row_dict_list=enriched_broker_order_row_dict_list,
        fill_row_dict_list=fill_row_dict_list,
    )

    assert enriched_broker_order_row_dict_list[0]["avg_fill_price_float"] == 102.5
    assert enriched_broker_order_row_dict_list[0]["official_open_price_float"] == 100.0
    assert enriched_broker_order_row_dict_list[0]["avg_slippage_bps_float"] == 250.0
    assert execution_row_dict_list[0]["avg_fill_price_float"] == 102.5
    assert execution_row_dict_list[0]["official_open_price_float"] == 100.0
    assert execution_row_dict_list[0]["avg_slippage_bps_float"] == 250.0


def test_live_runner_manual_vplan_flow(tmp_path: Path, monkeypatch):
    db_path_str = str((tmp_path / "live.sqlite3").resolve())
    _write_manifest(tmp_path, auto_submit_enabled_bool=False)
    monkeypatch.setattr("alpha.live.strategy_host.build_decision_plan_for_release", _build_decision_plan_stub)
    monkeypatch.setattr(
        "alpha.live.scheduler_utils.load_latest_norgate_heartbeat_session_label_ts",
        lambda data_profile_str: pd.Timestamp("2024-01-31"),
    )

    state_store_obj = LiveStateStore(db_path_str)
    broker_adapter_obj = StubBrokerAdapter()
    broker_adapter_obj.seed_account_snapshot(
        account_route_str="DU1",
        cash_float=10000.0,
        total_value_float=10000.0,
        net_liq_float=10000.0,
        available_funds_float=8000.0,
        excess_liquidity_float=7000.0,
        position_amount_map={},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 0, tzinfo=MARKET_TIMEZONE_OBJ),
        session_mode_str="paper",
    )
    broker_adapter_obj.seed_live_price_snapshot(
        account_route_str="DU1",
        asset_reference_price_map={"AAPL": 100.0},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
    )

    build_detail_dict = build_decision_plans(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
    )
    vplan_detail_dict = build_vplans(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        env_mode_str="paper",
    )
    show_vplan_detail_dict = show_vplan_summary(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
        pod_id_str="pod_test_01",
    )
    status_summary_dict = get_status_summary(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
    )

    assert build_detail_dict["created_decision_plan_count_int"] == 1
    assert vplan_detail_dict["created_vplan_count_int"] == 1
    assert show_vplan_detail_dict["vplan_dict_list"][0]["vplan_row_dict_list"] == [
        {
            "asset_str": "AAPL",
            "decision_base_share_float": 0.0,
            "current_share_float": 0.0,
            "drift_share_float": 0.0,
            "target_share_float": 10.0,
            "order_delta_share_float": 10.0,
            "live_reference_price_float": 100.0,
            "live_reference_source_str": "stub",
            "estimated_target_notional_float": 1000.0,
            "warning_bool": False,
        }
    ]
    assert show_vplan_detail_dict["vplan_dict_list"][0]["warning_row_dict_list"] == []
    assert status_summary_dict["pod_status_dict_list"][0]["next_action_str"] == "review_vplan"
    assert status_summary_dict["pod_status_dict_list"][0]["reason_code_str"] == "vplan_ready"


def test_submit_ready_vplans_persists_missing_broker_ack_and_surfaces_in_status(
    tmp_path: Path,
    monkeypatch,
):
    db_path_str = str((tmp_path / "live_missing_ack.sqlite3").resolve())
    log_path_obj = tmp_path / "live_events.jsonl"
    critical_log_path_obj = tmp_path / "live_critical_events.jsonl"
    _write_manifest(tmp_path, auto_submit_enabled_bool=False)
    monkeypatch.setattr("alpha.live.strategy_host.build_decision_plan_for_release", _build_two_asset_decision_plan_stub)
    monkeypatch.setattr(
        "alpha.live.scheduler_utils.load_latest_norgate_heartbeat_session_label_ts",
        lambda data_profile_str: pd.Timestamp("2024-01-31"),
    )
    monkeypatch.setattr(logging_utils, "DEFAULT_CRITICAL_LOG_PATH_STR", str(critical_log_path_obj))

    state_store_obj = LiveStateStore(db_path_str)
    broker_adapter_obj = StubBrokerAdapter()
    broker_adapter_obj.seed_account_snapshot(
        account_route_str="DU1",
        cash_float=10000.0,
        total_value_float=10000.0,
        net_liq_float=10000.0,
        available_funds_float=8000.0,
        excess_liquidity_float=7000.0,
        position_amount_map={},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 0, tzinfo=MARKET_TIMEZONE_OBJ),
        session_mode_str="paper",
    )
    broker_adapter_obj.seed_live_price_snapshot(
        account_route_str="DU1",
        asset_reference_price_map={"AAPL": 100.0, "MSFT": 50.0},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
    )
    _install_partial_submit_ack_stub(
        broker_adapter_obj=broker_adapter_obj,
        broker_ack_asset_set={"AAPL"},
    )

    build_decision_plans(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
        log_path_str=str(log_path_obj),
    )
    build_vplans(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        env_mode_str="paper",
        log_path_str=str(log_path_obj),
    )
    latest_vplan_obj = state_store_obj.get_latest_vplan_for_pod("pod_test_01")
    assert latest_vplan_obj is not None

    submit_detail_dict = submit_ready_vplans(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        env_mode_str="paper",
        manual_only_bool=False,
        vplan_id_int=int(latest_vplan_obj.vplan_id_int or 0),
        log_path_str=str(log_path_obj),
    )
    latest_vplan_obj = state_store_obj.get_latest_vplan_for_pod("pod_test_01")
    broker_ack_row_dict_list = state_store_obj.get_broker_ack_row_dict_list_for_vplan(
        int(latest_vplan_obj.vplan_id_int or 0)
    )
    status_summary_dict = get_status_summary(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
    )
    status_output_str = _render_status_detail_str(status_summary_dict)
    second_submit_detail_dict = submit_ready_vplans(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 23, tzinfo=MARKET_TIMEZONE_OBJ),
        env_mode_str="paper",
        manual_only_bool=False,
        vplan_id_int=int(latest_vplan_obj.vplan_id_int or 0),
        log_path_str=str(log_path_obj),
    )

    assert submit_detail_dict["submitted_vplan_count_int"] == 1
    assert latest_vplan_obj is not None
    assert latest_vplan_obj.status_str == "submitted"
    assert latest_vplan_obj.submit_ack_status_str == "missing_critical"
    assert latest_vplan_obj.ack_coverage_ratio_float == 0.5
    assert latest_vplan_obj.missing_ack_count_int == 1
    assert [broker_ack_row_dict["asset_str"] for broker_ack_row_dict in broker_ack_row_dict_list] == [
        "AAPL",
        "MSFT",
    ]
    assert [
        broker_ack_row_dict["asset_str"]
        for broker_ack_row_dict in broker_ack_row_dict_list
        if not broker_ack_row_dict["broker_response_ack_bool"]
    ] == ["MSFT"]
    assert status_summary_dict["pod_status_dict_list"][0]["submit_ack_status_str"] == "missing_critical"
    assert status_summary_dict["pod_status_dict_list"][0]["missing_ack_count_int"] == 1
    assert "CRITICAL ACK" in status_output_str
    assert "MSFT" in status_output_str
    assert second_submit_detail_dict["submitted_vplan_count_int"] == 0
    assert second_submit_detail_dict["blocked_action_count_int"] == 1

    critical_event_name_str_list = [
        json.loads(line_str)["event_name_str"]
        for line_str in critical_log_path_obj.read_text(encoding="utf-8").splitlines()
        if line_str.strip() != ""
    ]
    assert "submit_vplan_missing_broker_ack" in critical_event_name_str_list


def test_build_decision_plans_skips_pod_local_dtb3_stale_failure_and_continues(tmp_path: Path, monkeypatch):
    db_path_str = str((tmp_path / "live_dtb3_stale.sqlite3").resolve())
    log_path_obj = tmp_path / "dtb3_stale_events.jsonl"
    _write_manifest(tmp_path, auto_submit_enabled_bool=False)
    _write_manifest_file(
        root_path_obj=tmp_path,
        file_name_str="pod_ok.yaml",
        pod_id_str="pod_ok_02",
        release_id_str="user_001.pod_ok.daily.v2",
        strategy_import_str="strategies.dv2.strategy_mr_dv2:DVO2Strategy",
        auto_submit_enabled_bool=False,
        pod_budget_fraction_float=0.5,
    )
    monkeypatch.setattr(
        "alpha.live.scheduler_utils.load_latest_norgate_heartbeat_session_label_ts",
        lambda data_profile_str: pd.Timestamp("2024-01-31"),
    )

    stale_snapshot_obj = FredSeriesSnapshot(
        value_ser=pd.Series(
            [5.20],
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-29")]),
            name="DTB3",
        ),
        source_name_str="FRED",
        series_id_str="DTB3",
        download_attempt_timestamp_ts=datetime(2024, 1, 31, 16, 5, tzinfo=MARKET_TIMEZONE_OBJ),
        download_status_str="cache_fallback_after_download_error",
        latest_observation_date_ts=pd.Timestamp("2024-01-29"),
        used_cache_bool=True,
        freshness_business_days_int=2,
    )

    def _build_decision_plan_or_raise(release_obj, as_of_ts, pod_state_obj):
        if release_obj.pod_id_str == "pod_test_01":
            raise FredSeriesStaleError(
                message_str=(
                    "FRED series 'DTB3' is too stale for live use. "
                    "latest_observation_date=2024-01-29 freshness_business_days_int=2."
                ),
                series_id_str="DTB3",
                reason_code_str="dtb3_stale",
                series_snapshot_obj=stale_snapshot_obj,
            )
        return _build_decision_plan_stub(release_obj, as_of_ts, pod_state_obj)

    monkeypatch.setattr(
        "alpha.live.strategy_host.build_decision_plan_for_release",
        _build_decision_plan_or_raise,
    )

    state_store_obj = LiveStateStore(db_path_str)
    build_detail_dict = build_decision_plans(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
        log_path_str=str(log_path_obj),
    )

    skipped_decision_plan_obj = state_store_obj.get_latest_decision_plan_for_pod("pod_test_01")
    created_decision_plan_obj = state_store_obj.get_latest_decision_plan_for_pod("pod_ok_02")
    log_record_dict_list = [
        json.loads(log_line_str)
        for log_line_str in log_path_obj.read_text(encoding="utf-8").splitlines()
        if log_line_str.strip() != ""
    ]

    assert build_detail_dict["created_decision_plan_count_int"] == 1
    assert build_detail_dict["skipped_decision_plan_count_int"] == 1
    assert build_detail_dict["reason_count_map_dict"] == {"dtb3_stale": 1}
    assert skipped_decision_plan_obj is None
    assert created_decision_plan_obj is not None
    assert created_decision_plan_obj.pod_id_str == "pod_ok_02"
    assert any(
        log_record_dict["event_name_str"] == "build_decision_plan_data_dependency_error"
        and log_record_dict["pod_id_str"] == "pod_test_01"
        and log_record_dict["reason_code_str"] == "dtb3_stale"
        for log_record_dict in log_record_dict_list
    )


def test_build_vplan_warns_on_position_mismatch_but_continues(tmp_path: Path, monkeypatch):
    db_path_str = str((tmp_path / "live.sqlite3").resolve())
    _write_manifest(tmp_path, auto_submit_enabled_bool=False)
    monkeypatch.setattr(
        "alpha.live.strategy_host.build_decision_plan_for_release",
        _build_decision_plan_with_position_drift_stub,
    )
    monkeypatch.setattr(
        "alpha.live.scheduler_utils.load_latest_norgate_heartbeat_session_label_ts",
        lambda data_profile_str: pd.Timestamp("2024-01-31"),
    )

    state_store_obj = LiveStateStore(db_path_str)
    broker_adapter_obj = StubBrokerAdapter()
    broker_adapter_obj.seed_account_snapshot(
        account_route_str="DU1",
        cash_float=10000.0,
        total_value_float=10000.0,
        net_liq_float=10000.0,
        available_funds_float=8000.0,
        excess_liquidity_float=7000.0,
        position_amount_map={"AAPL": 3.0},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 0, tzinfo=MARKET_TIMEZONE_OBJ),
        session_mode_str="paper",
    )
    broker_adapter_obj.seed_live_price_snapshot(
        account_route_str="DU1",
        asset_reference_price_map={"AAPL": 100.0},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
    )

    build_decision_plans(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
    )
    vplan_detail_dict = build_vplans(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        env_mode_str="paper",
    )
    show_vplan_detail_dict = show_vplan_summary(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
        pod_id_str="pod_test_01",
    )

    assert vplan_detail_dict["created_vplan_count_int"] == 1
    assert vplan_detail_dict["blocked_action_count_int"] == 0
    assert vplan_detail_dict["warning_count_map_dict"] == {"position_reconciliation_warning": 1}
    assert show_vplan_detail_dict["vplan_dict_list"][0]["vplan_row_dict_list"] == [
        {
            "asset_str": "AAPL",
            "decision_base_share_float": 5.0,
            "current_share_float": 3.0,
            "drift_share_float": -2.0,
            "target_share_float": 10.0,
            "order_delta_share_float": 7.0,
            "live_reference_price_float": 100.0,
            "live_reference_source_str": "stub",
            "estimated_target_notional_float": 1000.0,
            "warning_bool": True,
        }
    ]
    assert show_vplan_detail_dict["vplan_dict_list"][0]["warning_row_dict_list"] == [
        {
            "asset_str": "AAPL",
            "decision_base_share_float": 5.0,
            "current_share_float": 3.0,
            "drift_share_float": -2.0,
        }
    ]


def test_live_runner_auto_tick_builds_submits_and_reconciles(tmp_path: Path, monkeypatch):
    db_path_str = str((tmp_path / "live.sqlite3").resolve())
    _write_manifest(tmp_path, auto_submit_enabled_bool=True)
    monkeypatch.setattr("alpha.live.strategy_host.build_decision_plan_for_release", _build_decision_plan_stub)
    monkeypatch.setattr(
        "alpha.live.scheduler_utils.load_latest_norgate_heartbeat_session_label_ts",
        lambda data_profile_str: pd.Timestamp("2024-01-31"),
    )

    state_store_obj = LiveStateStore(db_path_str)
    broker_adapter_obj = StubBrokerAdapter()
    broker_adapter_obj.seed_account_snapshot(
        account_route_str="DU1",
        cash_float=10000.0,
        total_value_float=10000.0,
        net_liq_float=10000.0,
        available_funds_float=8000.0,
        excess_liquidity_float=7000.0,
        position_amount_map={},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 0, tzinfo=MARKET_TIMEZONE_OBJ),
        session_mode_str="paper",
    )
    broker_adapter_obj.seed_live_price_snapshot(
        account_route_str="DU1",
        asset_reference_price_map={"AAPL": 100.0},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
    )

    first_tick_detail_dict = tick(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
        env_mode_str="paper",
    )
    second_tick_detail_dict = tick(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
        env_mode_str="paper",
    )
    third_tick_detail_dict = tick(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 35, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
        env_mode_str="paper",
    )
    status_summary_dict = get_status_summary(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 35, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
    )
    execution_report_summary_dict = get_execution_report_summary(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 35, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
    )

    assert first_tick_detail_dict["created_decision_plan_count_int"] == 1
    assert second_tick_detail_dict["created_vplan_count_int"] == 1
    assert second_tick_detail_dict["submitted_vplan_count_int"] == 1
    assert second_tick_detail_dict["completed_vplan_count_int"] == 0
    assert third_tick_detail_dict["completed_vplan_count_int"] == 1
    assert status_summary_dict["pod_status_dict_list"][0]["latest_decision_plan_status_str"] == "completed"
    assert status_summary_dict["pod_status_dict_list"][0]["latest_vplan_status_str"] == "completed"
    assert status_summary_dict["pod_status_dict_list"][0]["latest_broker_order_row_dict_list"] == [
        {
            "broker_order_id_str": status_summary_dict["pod_status_dict_list"][0]["latest_broker_order_row_dict_list"][0]["broker_order_id_str"],
            "asset_str": "AAPL",
            "order_request_key_str": "vplan:1:AAPL:1",
            "broker_order_type_str": "MOO",
            "unit_str": "shares",
            "amount_float": 10.0,
            "filled_amount_float": 10.0,
            "remaining_amount_float": 0.0,
            "avg_fill_price_float": 100.0,
            "status_str": "Filled",
            "last_status_timestamp_str": "2024-02-01T09:22:00-05:00",
            "submitted_timestamp_str": "2024-02-01T09:22:00-05:00",
            "submission_key_str": "vplan:1",
        }
    ]
    assert status_summary_dict["pod_status_dict_list"][0]["exception_row_dict_list"] == []
    assert execution_report_summary_dict["execution_report_dict_list"][0]["fill_row_dict_list"] == [
        {
            "asset_str": "AAPL",
            "fill_amount_float": 10.0,
            "fill_price_float": 100.0,
            "official_open_price_float": 100.0,
            "open_price_source_str": "stub.live_price",
            "fill_timestamp_str": "2024-02-01T09:22:00-05:00",
            "signed_fill_direction_float": 1.0,
            "fill_slippage_bps_float": 0.0,
            "slippage_share_float": 0.0,
            "slippage_notional_float": 0.0,
        }
    ]
    assert execution_report_summary_dict["execution_report_dict_list"][0]["fill_with_open_count_int"] == 1
    assert execution_report_summary_dict["execution_report_dict_list"][0]["aggregate_slippage_notional_float"] == 0.0


def test_submit_vplan_explicit_manual_path(tmp_path: Path, monkeypatch):
    db_path_str = str((tmp_path / "live.sqlite3").resolve())
    _write_manifest(tmp_path, auto_submit_enabled_bool=False)
    monkeypatch.setattr("alpha.live.strategy_host.build_decision_plan_for_release", _build_decision_plan_stub)
    monkeypatch.setattr(
        "alpha.live.scheduler_utils.load_latest_norgate_heartbeat_session_label_ts",
        lambda data_profile_str: pd.Timestamp("2024-01-31"),
    )

    state_store_obj = LiveStateStore(db_path_str)
    broker_adapter_obj = StubBrokerAdapter()
    broker_adapter_obj.seed_account_snapshot(
        account_route_str="DU1",
        cash_float=10000.0,
        total_value_float=10000.0,
        net_liq_float=10000.0,
        available_funds_float=8000.0,
        excess_liquidity_float=7000.0,
        position_amount_map={},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 0, tzinfo=MARKET_TIMEZONE_OBJ),
        session_mode_str="paper",
    )
    broker_adapter_obj.seed_live_price_snapshot(
        account_route_str="DU1",
        asset_reference_price_map={"AAPL": 100.0},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
    )

    build_decision_plans(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
    )
    build_vplans(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        env_mode_str="paper",
    )
    latest_vplan_obj = state_store_obj.get_latest_vplan_for_pod("pod_test_01")
    assert latest_vplan_obj is not None

    submit_detail_dict = submit_ready_vplans(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        env_mode_str="paper",
        manual_only_bool=False,
        vplan_id_int=int(latest_vplan_obj.vplan_id_int or 0),
    )
    reconcile_detail_dict = post_execution_reconcile(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 35, tzinfo=MARKET_TIMEZONE_OBJ),
    )

    assert submit_detail_dict["submitted_vplan_count_int"] == 1
    assert reconcile_detail_dict["completed_vplan_count_int"] == 1


def test_post_execution_reconcile_keeps_partial_entry_unresolved_and_reports_human_readably(
    tmp_path: Path,
    monkeypatch,
):
    db_path_str = str((tmp_path / "live_partial.sqlite3").resolve())
    log_path_obj = tmp_path / "partial_events.jsonl"
    _write_manifest(tmp_path, auto_submit_enabled_bool=False)
    monkeypatch.setattr("alpha.live.strategy_host.build_decision_plan_for_release", _build_decision_plan_stub)
    monkeypatch.setattr(
        "alpha.live.scheduler_utils.load_latest_norgate_heartbeat_session_label_ts",
        lambda data_profile_str: pd.Timestamp("2024-01-31"),
    )

    state_store_obj = LiveStateStore(db_path_str)
    broker_adapter_obj = StubBrokerAdapter()
    broker_adapter_obj.seed_account_snapshot(
        account_route_str="DU1",
        cash_float=10000.0,
        total_value_float=10000.0,
        net_liq_float=10000.0,
        available_funds_float=8000.0,
        excess_liquidity_float=7000.0,
        position_amount_map={},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 0, tzinfo=MARKET_TIMEZONE_OBJ),
        session_mode_str="paper",
    )
    broker_adapter_obj.seed_live_price_snapshot(
        account_route_str="DU1",
        asset_reference_price_map={"AAPL": 100.0},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
    )
    _install_pending_submit_truth_stub(
        broker_adapter_obj,
        {
            "AAPL": {
                "status_str": "Cancelled",
                "filled_amount_float": 4.0,
                "remaining_amount_float": 6.0,
                "fill_price_float": 101.0,
            }
        },
    )

    build_decision_plans(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
    )
    build_vplans(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        env_mode_str="paper",
    )
    latest_vplan_obj = state_store_obj.get_latest_vplan_for_pod("pod_test_01")
    assert latest_vplan_obj is not None

    submit_ready_vplans(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        env_mode_str="paper",
        manual_only_bool=False,
        vplan_id_int=int(latest_vplan_obj.vplan_id_int or 0),
    )
    broker_adapter_obj.seed_account_snapshot(
        account_route_str="DU1",
        cash_float=9596.0,
        total_value_float=9996.0,
        net_liq_float=9996.0,
        available_funds_float=9596.0,
        excess_liquidity_float=9596.0,
        position_amount_map={"AAPL": 4.0},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 35, tzinfo=MARKET_TIMEZONE_OBJ),
        session_mode_str="paper",
    )

    reconcile_detail_dict = post_execution_reconcile(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 35, tzinfo=MARKET_TIMEZONE_OBJ),
        log_path_str=str(log_path_obj),
    )
    latest_vplan_obj = state_store_obj.get_latest_vplan_for_pod("pod_test_01")
    latest_decision_plan_obj = state_store_obj.get_latest_decision_plan_for_pod("pod_test_01")
    status_summary_dict = get_status_summary(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 35, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
    )
    show_vplan_detail_dict = show_vplan_summary(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 35, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
        pod_id_str="pod_test_01",
    )
    execution_report_summary_dict = get_execution_report_summary(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 35, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
    )

    assert reconcile_detail_dict["completed_vplan_count_int"] == 0
    assert latest_vplan_obj is not None
    assert latest_vplan_obj.status_str == "submitted"
    assert latest_decision_plan_obj is not None
    assert latest_decision_plan_obj.status_str == "submitted"
    assert status_summary_dict["pod_status_dict_list"][0]["next_action_str"] == "manual_review"
    assert status_summary_dict["pod_status_dict_list"][0]["reason_code_str"] == "execution_exception_parked"
    assert status_summary_dict["pod_status_dict_list"][0]["exception_row_dict_list"] == [
        {
            "asset_str": "AAPL",
            "current_share_float": 0.0,
            "target_share_float": 10.0,
            "broker_share_float": 4.0,
            "residual_share_float": 6.0,
            "residual_notional_float": 600.0,
            "open_share_float": 6.0,
            "planned_order_delta_share_float": 10.0,
            "filled_share_float": 4.0,
            "latest_fill_timestamp_str": "2024-02-01T09:22:00-05:00",
            "latest_fill_price_float": 101.0,
            "latest_fill_slippage_bps_float": 100.0,
            "latest_broker_order_status_str": "Cancelled",
            "latest_broker_order_timestamp_str": "2024-02-01T09:22:00-05:00",
            "live_reference_price_float": 100.0,
            "unresolved_bool": True,
            "exit_breach_bool": False,
            "problem_bool": True,
            "avg_fill_price_float": 101.0,
            "official_open_price_float": 100.0,
            "avg_slippage_bps_float": 100.0,
        }
    ]
    assert show_vplan_detail_dict["vplan_dict_list"][0]["exception_row_dict_list"] == (
        status_summary_dict["pod_status_dict_list"][0]["exception_row_dict_list"]
    )
    assert show_vplan_detail_dict["vplan_dict_list"][0]["broker_order_row_dict_list"] == [
        {
            "broker_order_id_str": "stub_order_1",
            "asset_str": "AAPL",
            "order_request_key_str": "vplan:1:AAPL:1",
            "broker_order_type_str": "MOO",
            "unit_str": "shares",
            "amount_float": 10.0,
            "filled_amount_float": 4.0,
            "remaining_amount_float": 6.0,
            "avg_fill_price_float": 101.0,
            "status_str": "Cancelled",
            "last_status_timestamp_str": "2024-02-01T09:22:00-05:00",
            "submitted_timestamp_str": "2024-02-01T09:22:00-05:00",
            "submission_key_str": "vplan:1",
            "official_open_price_float": 100.0,
            "avg_slippage_bps_float": 100.0,
        }
    ]
    assert execution_report_summary_dict["execution_report_dict_list"][0]["fill_row_dict_list"] == [
        {
            "asset_str": "AAPL",
            "fill_amount_float": 4.0,
            "fill_price_float": 101.0,
            "official_open_price_float": 100.0,
            "open_price_source_str": "stub.live_price",
            "fill_timestamp_str": "2024-02-01T09:22:00-05:00",
            "signed_fill_direction_float": 1.0,
            "fill_slippage_bps_float": 100.0,
            "slippage_share_float": 1.0,
            "slippage_notional_float": 4.0,
        }
    ]
    log_record_dict_list = [
        json.loads(log_line_str)
        for log_line_str in log_path_obj.read_text(encoding="utf-8").splitlines()
        if log_line_str.strip() != ""
    ]
    parked_event_record_dict_list = [
        log_record_dict
        for log_record_dict in log_record_dict_list
        if log_record_dict["event_name_str"] == "execution_exception_parked"
    ]
    assert len(parked_event_record_dict_list) == 1
    assert parked_event_record_dict_list[0]["severity_str"] == "critical"
    assert parked_event_record_dict_list[0]["terminal_unresolved_asset_list"] == ["AAPL"]
    assert parked_event_record_dict_list[0]["terminal_unresolved_status_map_dict"] == {"AAPL": "Cancelled"}
    assert parked_event_record_dict_list[0]["terminal_unresolved_residual_share_map_dict"] == {"AAPL": 6.0}

    status_output_str = _render_command_output_str("status", status_summary_dict)
    vplan_output_str = _render_command_output_str("show_vplan", show_vplan_detail_dict)
    execution_output_str = _render_command_output_str("execution_report", execution_report_summary_dict)

    assert "Exceptions: 1" in status_output_str
    assert "Issue: AAPL | target=10.0000 | broker=4.0000 | residual=6.0000" in status_output_str
    assert "Row: AAPL | target=10.0000 | broker=4.0000 | residual=6.0000" in vplan_output_str
    assert "avg_bps=100.0 bps" in vplan_output_str
    assert "official_open=100.0000" in execution_output_str
    assert "slippage_bps=100.0 bps" in execution_output_str
    assert "slippage/share=1.0000" in execution_output_str


def test_post_execution_reconcile_parks_mixed_terminal_and_nonterminal_unresolved_vplan(
    tmp_path: Path,
    monkeypatch,
):
    db_path_str = str((tmp_path / "live_mixed.sqlite3").resolve())
    log_path_obj = tmp_path / "mixed_events.jsonl"
    _write_manifest(tmp_path, auto_submit_enabled_bool=False)
    monkeypatch.setattr(
        "alpha.live.strategy_host.build_decision_plan_for_release",
        _build_three_asset_decision_plan_stub,
    )
    monkeypatch.setattr(
        "alpha.live.scheduler_utils.load_latest_norgate_heartbeat_session_label_ts",
        lambda data_profile_str: pd.Timestamp("2024-01-31"),
    )

    state_store_obj = LiveStateStore(db_path_str)
    broker_adapter_obj = StubBrokerAdapter()
    broker_adapter_obj.seed_account_snapshot(
        account_route_str="DU1",
        cash_float=10000.0,
        total_value_float=10000.0,
        net_liq_float=10000.0,
        available_funds_float=8000.0,
        excess_liquidity_float=7000.0,
        position_amount_map={},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 0, tzinfo=MARKET_TIMEZONE_OBJ),
        session_mode_str="paper",
    )
    broker_adapter_obj.seed_live_price_snapshot(
        account_route_str="DU1",
        asset_reference_price_map={"AAPL": 100.0, "MSFT": 100.0, "NVDA": 100.0},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
    )

    build_decision_plans(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
    )
    build_vplans(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        env_mode_str="paper",
    )
    latest_vplan_obj = state_store_obj.get_latest_vplan_for_pod("pod_test_01")
    assert latest_vplan_obj is not None
    _install_pending_submit_truth_stub(
        broker_adapter_obj,
        {
            "AAPL": {
                "status_str": "Filled",
                "filled_amount_float": abs(float(latest_vplan_obj.order_delta_map["AAPL"])),
                "remaining_amount_float": 0.0,
                "fill_price_float": 101.0,
            },
            "MSFT": {
                "status_str": "Cancelled",
                "filled_amount_float": 0.0,
                "fill_price_float": None,
            },
            "NVDA": {
                "status_str": "PendingSubmit",
                "filled_amount_float": 0.0,
                "fill_price_float": None,
            },
        },
    )

    submit_ready_vplans(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        env_mode_str="paper",
        manual_only_bool=False,
        vplan_id_int=int(latest_vplan_obj.vplan_id_int or 0),
    )
    broker_adapter_obj.seed_account_snapshot(
        account_route_str="DU1",
        cash_float=8000.0,
        total_value_float=10000.0,
        net_liq_float=10000.0,
        available_funds_float=8000.0,
        excess_liquidity_float=7000.0,
        position_amount_map={"AAPL": float(latest_vplan_obj.target_share_map["AAPL"])},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 35, tzinfo=MARKET_TIMEZONE_OBJ),
        session_mode_str="paper",
    )

    reconcile_detail_dict = post_execution_reconcile(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 35, tzinfo=MARKET_TIMEZONE_OBJ),
        log_path_str=str(log_path_obj),
    )
    status_summary_dict = get_status_summary(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 35, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
    )
    log_record_dict_list = [
        json.loads(log_line_str)
        for log_line_str in log_path_obj.read_text(encoding="utf-8").splitlines()
        if log_line_str.strip() != ""
    ]

    assert reconcile_detail_dict["completed_vplan_count_int"] == 0
    assert status_summary_dict["pod_status_dict_list"][0]["next_action_str"] == "manual_review"
    assert status_summary_dict["pod_status_dict_list"][0]["reason_code_str"] == "execution_exception_parked"
    assert [
        exception_row_dict["asset_str"]
        for exception_row_dict in status_summary_dict["pod_status_dict_list"][0]["exception_row_dict_list"]
    ] == ["MSFT", "NVDA"]
    assert any(
        exception_row_dict["asset_str"] == "MSFT"
        and exception_row_dict["latest_broker_order_status_str"] == "Cancelled"
        for exception_row_dict in status_summary_dict["pod_status_dict_list"][0]["exception_row_dict_list"]
    )
    assert any(
        exception_row_dict["asset_str"] == "NVDA"
        and exception_row_dict["latest_broker_order_status_str"] == "PendingSubmit"
        for exception_row_dict in status_summary_dict["pod_status_dict_list"][0]["exception_row_dict_list"]
    )
    parked_event_record_dict_list = [
        log_record_dict
        for log_record_dict in log_record_dict_list
        if log_record_dict["event_name_str"] == "execution_exception_parked"
    ]
    assert len(parked_event_record_dict_list) == 1
    assert parked_event_record_dict_list[0]["terminal_unresolved_asset_list"] == ["MSFT"]
    assert parked_event_record_dict_list[0]["terminal_unresolved_status_map_dict"] == {"MSFT": "Cancelled"}


def test_post_execution_reconcile_keeps_nonterminal_unresolved_vplan_reconcile_eligible(
    tmp_path: Path,
    monkeypatch,
):
    db_path_str = str((tmp_path / "live_pending.sqlite3").resolve())
    log_path_obj = tmp_path / "pending_events.jsonl"
    _write_manifest(tmp_path, auto_submit_enabled_bool=False)
    monkeypatch.setattr("alpha.live.strategy_host.build_decision_plan_for_release", _build_decision_plan_stub)
    monkeypatch.setattr(
        "alpha.live.scheduler_utils.load_latest_norgate_heartbeat_session_label_ts",
        lambda data_profile_str: pd.Timestamp("2024-01-31"),
    )

    state_store_obj = LiveStateStore(db_path_str)
    broker_adapter_obj = StubBrokerAdapter()
    broker_adapter_obj.seed_account_snapshot(
        account_route_str="DU1",
        cash_float=10000.0,
        total_value_float=10000.0,
        net_liq_float=10000.0,
        available_funds_float=8000.0,
        excess_liquidity_float=7000.0,
        position_amount_map={},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 0, tzinfo=MARKET_TIMEZONE_OBJ),
        session_mode_str="paper",
    )
    broker_adapter_obj.seed_live_price_snapshot(
        account_route_str="DU1",
        asset_reference_price_map={"AAPL": 100.0},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
    )
    _install_pending_submit_truth_stub(
        broker_adapter_obj,
        {
            "AAPL": {
                "status_str": "PendingSubmit",
                "filled_amount_float": 0.0,
                "fill_price_float": None,
            }
        },
    )

    build_decision_plans(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
    )
    build_vplans(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        env_mode_str="paper",
    )
    latest_vplan_obj = state_store_obj.get_latest_vplan_for_pod("pod_test_01")
    assert latest_vplan_obj is not None

    submit_ready_vplans(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        env_mode_str="paper",
        manual_only_bool=False,
        vplan_id_int=int(latest_vplan_obj.vplan_id_int or 0),
    )
    broker_adapter_obj.seed_account_snapshot(
        account_route_str="DU1",
        cash_float=10000.0,
        total_value_float=10000.0,
        net_liq_float=10000.0,
        available_funds_float=8000.0,
        excess_liquidity_float=7000.0,
        position_amount_map={},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 35, tzinfo=MARKET_TIMEZONE_OBJ),
        session_mode_str="paper",
    )

    reconcile_detail_dict = post_execution_reconcile(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 35, tzinfo=MARKET_TIMEZONE_OBJ),
        log_path_str=str(log_path_obj),
    )
    status_summary_dict = get_status_summary(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 35, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
    )
    log_record_dict_list = [
        json.loads(log_line_str)
        for log_line_str in log_path_obj.read_text(encoding="utf-8").splitlines()
        if log_line_str.strip() != ""
    ]

    assert reconcile_detail_dict["completed_vplan_count_int"] == 0
    assert status_summary_dict["pod_status_dict_list"][0]["next_action_str"] == "post_execution_reconcile"
    assert status_summary_dict["pod_status_dict_list"][0]["reason_code_str"] == "waiting_for_post_execution_reconcile"
    assert any(
        exception_row_dict["asset_str"] == "AAPL"
        and exception_row_dict["latest_broker_order_status_str"] == "PendingSubmit"
        for exception_row_dict in status_summary_dict["pod_status_dict_list"][0]["exception_row_dict_list"]
    )
    assert not any(
        log_record_dict["event_name_str"] == "execution_exception_parked"
        for log_record_dict in log_record_dict_list
    )


def test_post_execution_reconcile_logs_critical_exit_residual_without_completing(
    tmp_path: Path,
    monkeypatch,
):
    db_path_str = str((tmp_path / "live_exit.sqlite3").resolve())
    log_path_obj = tmp_path / "exit_events.jsonl"
    _write_manifest(tmp_path, auto_submit_enabled_bool=False)
    monkeypatch.setattr(
        "alpha.live.strategy_host.build_decision_plan_for_release",
        _build_exit_only_decision_plan_stub,
    )
    monkeypatch.setattr(
        "alpha.live.scheduler_utils.load_latest_norgate_heartbeat_session_label_ts",
        lambda data_profile_str: pd.Timestamp("2024-01-31"),
    )

    state_store_obj = LiveStateStore(db_path_str)
    broker_adapter_obj = StubBrokerAdapter()
    broker_adapter_obj.seed_account_snapshot(
        account_route_str="DU1",
        cash_float=9500.0,
        total_value_float=10000.0,
        net_liq_float=10000.0,
        available_funds_float=9500.0,
        excess_liquidity_float=9500.0,
        position_amount_map={"AAPL": 5.0},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 0, tzinfo=MARKET_TIMEZONE_OBJ),
        session_mode_str="paper",
    )
    broker_adapter_obj.seed_live_price_snapshot(
        account_route_str="DU1",
        asset_reference_price_map={"AAPL": 100.0},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
    )
    _install_pending_submit_truth_stub(
        broker_adapter_obj,
        {
            "AAPL": {
                "status_str": "Cancelled",
                "filled_amount_float": 0.0,
                "remaining_amount_float": 5.0,
                "fill_price_float": None,
            }
        },
    )

    build_decision_plans(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
    )
    build_vplans(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        env_mode_str="paper",
    )
    latest_vplan_obj = state_store_obj.get_latest_vplan_for_pod("pod_test_01")
    assert latest_vplan_obj is not None

    submit_ready_vplans(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        env_mode_str="paper",
        manual_only_bool=False,
        vplan_id_int=int(latest_vplan_obj.vplan_id_int or 0),
    )
    broker_adapter_obj.seed_account_snapshot(
        account_route_str="DU1",
        cash_float=9500.0,
        total_value_float=10000.0,
        net_liq_float=10000.0,
        available_funds_float=9500.0,
        excess_liquidity_float=9500.0,
        position_amount_map={"AAPL": 5.0},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 35, tzinfo=MARKET_TIMEZONE_OBJ),
        session_mode_str="paper",
    )

    reconcile_detail_dict = post_execution_reconcile(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 35, tzinfo=MARKET_TIMEZONE_OBJ),
        log_path_str=str(log_path_obj),
    )
    status_summary_dict = get_status_summary(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 35, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
    )
    show_vplan_detail_dict = show_vplan_summary(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 35, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
        pod_id_str="pod_test_01",
    )
    log_record_dict_list = [
        json.loads(log_line_str)
        for log_line_str in log_path_obj.read_text(encoding="utf-8").splitlines()
        if log_line_str.strip() != ""
    ]

    assert reconcile_detail_dict["completed_vplan_count_int"] == 0
    assert status_summary_dict["pod_status_dict_list"][0]["exception_row_dict_list"][0]["exit_breach_bool"] is True
    assert show_vplan_detail_dict["vplan_dict_list"][0]["exception_row_dict_list"][0]["exit_breach_bool"] is True
    assert any(
        log_record_dict["event_name_str"] == "exit_residual_detected"
        and log_record_dict["severity_str"] == "critical"
        and log_record_dict["asset_str"] == "AAPL"
        for log_record_dict in log_record_dict_list
    )
    status_output_str = _render_command_output_str("status", status_summary_dict)
    assert "CRITICAL EXIT: AAPL" in status_output_str


def test_build_vplan_blocks_when_live_price_is_missing(tmp_path: Path, monkeypatch):
    db_path_str = str((tmp_path / "live.sqlite3").resolve())
    _write_manifest(tmp_path, auto_submit_enabled_bool=False)
    monkeypatch.setattr("alpha.live.strategy_host.build_decision_plan_for_release", _build_decision_plan_stub)
    monkeypatch.setattr(
        "alpha.live.scheduler_utils.load_latest_norgate_heartbeat_session_label_ts",
        lambda data_profile_str: pd.Timestamp("2024-01-31"),
    )

    state_store_obj = LiveStateStore(db_path_str)
    broker_adapter_obj = StubBrokerAdapter()
    broker_adapter_obj.seed_account_snapshot(
        account_route_str="DU1",
        cash_float=10000.0,
        total_value_float=10000.0,
        net_liq_float=10000.0,
        position_amount_map={},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 0, tzinfo=MARKET_TIMEZONE_OBJ),
        session_mode_str="paper",
    )
    broker_adapter_obj.seed_live_price_snapshot(
        account_route_str="DU1",
        asset_reference_price_map={},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
    )

    build_decision_plans(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
    )
    vplan_detail_dict = build_vplans(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        env_mode_str="paper",
    )

    assert vplan_detail_dict["created_vplan_count_int"] == 0
    assert vplan_detail_dict["blocked_action_count_int"] == 1
    assert vplan_detail_dict["reason_count_map_dict"] == {"missing_live_price": 1}


def test_build_vplan_market_price_fallback_keeps_share_sizing_and_logs_warning(tmp_path: Path, monkeypatch):
    db_path_str = str((tmp_path / "live_market_price_fallback.sqlite3").resolve())
    log_path_obj = tmp_path / "market_price_fallback_events.jsonl"
    _write_manifest(tmp_path, auto_submit_enabled_bool=False)
    monkeypatch.setattr("alpha.live.strategy_host.build_decision_plan_for_release", _build_decision_plan_stub)
    monkeypatch.setattr(
        "alpha.live.scheduler_utils.load_latest_norgate_heartbeat_session_label_ts",
        lambda data_profile_str: pd.Timestamp("2024-01-31"),
    )

    state_store_obj = LiveStateStore(db_path_str)
    broker_adapter_obj = StubBrokerAdapter()
    broker_adapter_obj.seed_account_snapshot(
        account_route_str="DU1",
        cash_float=10000.0,
        total_value_float=10000.0,
        net_liq_float=10000.0,
        position_amount_map={},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 0, tzinfo=MARKET_TIMEZONE_OBJ),
        session_mode_str="paper",
    )
    broker_adapter_obj.seed_live_price_snapshot(
        account_route_str="DU1",
        asset_reference_price_map={"AAPL": 100.0},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        price_source_str="marketPrice_fallback",
        asset_reference_source_map_dict={"AAPL": "ib_async.reqTickers.marketPrice.fallback"},
    )

    build_decision_plans(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
    )
    vplan_detail_dict = build_vplans(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        env_mode_str="paper",
        log_path_str=str(log_path_obj),
    )
    show_vplan_detail_dict = show_vplan_summary(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
        pod_id_str="pod_test_01",
    )
    log_record_dict_list = [
        json.loads(log_line_str)
        for log_line_str in log_path_obj.read_text(encoding="utf-8").splitlines()
        if log_line_str.strip() != ""
    ]

    assert vplan_detail_dict["created_vplan_count_int"] == 1
    assert show_vplan_detail_dict["vplan_dict_list"][0]["live_price_source_str"] == "marketPrice_fallback"
    assert show_vplan_detail_dict["vplan_dict_list"][0]["live_reference_source_map_dict"] == {
        "AAPL": "ib_async.reqTickers.marketPrice.fallback"
    }
    assert show_vplan_detail_dict["vplan_dict_list"][0]["vplan_row_dict_list"] == [
        {
            "asset_str": "AAPL",
            "decision_base_share_float": 0.0,
            "current_share_float": 0.0,
            "drift_share_float": 0.0,
            "target_share_float": 10.0,
            "order_delta_share_float": 10.0,
            "live_reference_price_float": 100.0,
            "live_reference_source_str": "ib_async.reqTickers.marketPrice.fallback",
            "estimated_target_notional_float": 1000.0,
            "warning_bool": False,
        }
    ]
    assert any(
        log_record_dict["event_name_str"] == "build_vplan_live_reference_fallback_warning"
        and log_record_dict["asset_str"] == "AAPL"
        and log_record_dict["fallback_source_str"] == "ib_async.reqTickers.marketPrice.fallback"
        for log_record_dict in log_record_dict_list
    )


def test_build_vplan_mixed_live_reference_sources_roundtrip_and_log_warnings(tmp_path: Path, monkeypatch):
    db_path_str = str((tmp_path / "live_mixed_reference_sources.sqlite3").resolve())
    log_path_obj = tmp_path / "mixed_reference_sources_events.jsonl"
    _write_manifest(tmp_path, auto_submit_enabled_bool=False)
    monkeypatch.setattr(
        "alpha.live.strategy_host.build_decision_plan_for_release",
        _build_full_target_weight_decision_plan_stub,
    )
    monkeypatch.setattr(
        "alpha.live.scheduler_utils.load_latest_norgate_heartbeat_session_label_ts",
        lambda data_profile_str: pd.Timestamp("2024-01-31"),
    )

    state_store_obj = LiveStateStore(db_path_str)
    broker_adapter_obj = StubBrokerAdapter()
    broker_adapter_obj.seed_account_snapshot(
        account_route_str="DU1",
        cash_float=10000.0,
        total_value_float=10000.0,
        net_liq_float=10000.0,
        available_funds_float=8000.0,
        excess_liquidity_float=7000.0,
        position_amount_map={"AAPL": 4.0, "MSFT": 2.0},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 0, tzinfo=MARKET_TIMEZONE_OBJ),
        session_mode_str="paper",
    )
    broker_adapter_obj.seed_live_price_snapshot(
        account_route_str="DU1",
        asset_reference_price_map={"AAPL": 100.0, "TLT": 50.0, "MSFT": 200.0},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        price_source_str="mixed",
        asset_reference_source_map_dict={
            "AAPL": "ib_async.reqMktData.225.auctionPrice",
            "TLT": "ib_async.reqMktData.marketPrice.fallback",
            "MSFT": "ib_async.reqTickers.marketPrice.fallback",
        },
    )

    build_decision_plans(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
    )
    vplan_detail_dict = build_vplans(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        env_mode_str="paper",
        log_path_str=str(log_path_obj),
    )
    show_vplan_detail_dict = show_vplan_summary(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
        pod_id_str="pod_test_01",
    )
    log_record_dict_list = [
        json.loads(log_line_str)
        for log_line_str in log_path_obj.read_text(encoding="utf-8").splitlines()
        if log_line_str.strip() != ""
    ]

    assert vplan_detail_dict["created_vplan_count_int"] == 1
    assert show_vplan_detail_dict["vplan_dict_list"][0]["live_price_source_str"] == "mixed"
    assert show_vplan_detail_dict["vplan_dict_list"][0]["live_reference_source_map_dict"] == {
        "AAPL": "ib_async.reqMktData.225.auctionPrice",
        "TLT": "ib_async.reqMktData.marketPrice.fallback",
        "MSFT": "ib_async.reqTickers.marketPrice.fallback",
    }
    assert [
        (
            vplan_row_dict["asset_str"],
            vplan_row_dict["live_reference_source_str"],
        )
        for vplan_row_dict in show_vplan_detail_dict["vplan_dict_list"][0]["vplan_row_dict_list"]
    ] == [
        ("AAPL", "ib_async.reqMktData.225.auctionPrice"),
        ("MSFT", "ib_async.reqTickers.marketPrice.fallback"),
        ("TLT", "ib_async.reqMktData.marketPrice.fallback"),
    ]
    assert {
        (
            log_record_dict["asset_str"],
            log_record_dict["fallback_source_str"],
        )
        for log_record_dict in log_record_dict_list
        if log_record_dict["event_name_str"] == "build_vplan_live_reference_fallback_warning"
    } == {
        ("MSFT", "ib_async.reqTickers.marketPrice.fallback"),
        ("TLT", "ib_async.reqMktData.marketPrice.fallback"),
    }


def test_build_vplan_full_target_weight_book_rebalances_omitted_holdings(tmp_path: Path, monkeypatch):
    db_path_str = str((tmp_path / "live_full_target.sqlite3").resolve())
    _write_manifest(tmp_path, auto_submit_enabled_bool=False)
    monkeypatch.setattr(
        "alpha.live.strategy_host.build_decision_plan_for_release",
        _build_full_target_weight_decision_plan_stub,
    )
    monkeypatch.setattr(
        "alpha.live.scheduler_utils.load_latest_norgate_heartbeat_session_label_ts",
        lambda data_profile_str: pd.Timestamp("2024-01-31"),
    )

    state_store_obj = LiveStateStore(db_path_str)
    broker_adapter_obj = StubBrokerAdapter()
    broker_adapter_obj.seed_account_snapshot(
        account_route_str="DU1",
        cash_float=10000.0,
        total_value_float=10000.0,
        net_liq_float=10000.0,
        available_funds_float=8000.0,
        excess_liquidity_float=7000.0,
        position_amount_map={"AAPL": 4.0, "MSFT": 2.0},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 0, tzinfo=MARKET_TIMEZONE_OBJ),
        session_mode_str="paper",
    )
    broker_adapter_obj.seed_live_price_snapshot(
        account_route_str="DU1",
        asset_reference_price_map={"AAPL": 100.0, "TLT": 50.0, "MSFT": 200.0},
        snapshot_timestamp_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
    )

    build_decision_plans(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 1, 31, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
    )
    vplan_detail_dict = build_vplans(
        state_store_obj=state_store_obj,
        broker_adapter_obj=broker_adapter_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        env_mode_str="paper",
    )
    show_vplan_detail_dict = show_vplan_summary(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
        pod_id_str="pod_test_01",
    )

    assert vplan_detail_dict["created_vplan_count_int"] == 1
    assert show_vplan_detail_dict["vplan_dict_list"][0]["vplan_row_dict_list"] == [
        {
            "asset_str": "AAPL",
            "decision_base_share_float": 4.0,
            "current_share_float": 4.0,
            "drift_share_float": 0.0,
            "target_share_float": 15.0,
            "order_delta_share_float": 11.0,
            "live_reference_price_float": 100.0,
            "live_reference_source_str": "stub",
            "estimated_target_notional_float": 1500.0,
            "warning_bool": False,
        },
        {
            "asset_str": "MSFT",
            "decision_base_share_float": 0.0,
            "current_share_float": 2.0,
            "drift_share_float": 2.0,
            "target_share_float": 0.0,
            "order_delta_share_float": -2.0,
            "live_reference_price_float": 200.0,
            "live_reference_source_str": "stub",
            "estimated_target_notional_float": 0.0,
            "warning_bool": True,
        },
        {
            "asset_str": "TLT",
            "decision_base_share_float": 0.0,
            "current_share_float": 0.0,
            "drift_share_float": 0.0,
            "target_share_float": 20.0,
            "order_delta_share_float": 20.0,
            "live_reference_price_float": 50.0,
            "live_reference_source_str": "stub",
            "estimated_target_notional_float": 1000.0,
            "warning_bool": False,
        },
    ]


def test_preflight_contract_summary_reports_pass(tmp_path: Path, monkeypatch):
    db_path_str = str((tmp_path / "live.sqlite3").resolve())
    _write_manifest(tmp_path, auto_submit_enabled_bool=False)

    monkeypatch.setattr(
        "alpha.live.strategy_host.preflight_decision_contract_for_release",
        lambda release_obj, as_of_ts, pod_state_obj: {
            "release_id_str": release_obj.release_id_str,
            "pod_id_str": release_obj.pod_id_str,
            "strategy_import_str": release_obj.strategy_import_str,
            "decision_book_type_str": "incremental_entry_exit_book",
            "contract_status_str": "pass",
            "accepted_shape_count_int": 1,
            "unsupported_shape_count_int": 0,
            "unsupported_shape_example_dict_list": [],
            "error_str": None,
        },
    )

    state_store_obj = LiveStateStore(db_path_str)
    detail_dict = preflight_contract_summary(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
    )

    assert detail_dict["enabled_release_count_int"] == 1
    assert detail_dict["passed_release_count_int"] == 1
    assert detail_dict["failed_release_count_int"] == 0
    assert detail_dict["contract_report_dict_list"][0]["contract_status_str"] == "pass"


def test_preflight_contract_summary_reports_failure(tmp_path: Path, monkeypatch):
    db_path_str = str((tmp_path / "live.sqlite3").resolve())
    _write_manifest(tmp_path, auto_submit_enabled_bool=False)

    monkeypatch.setattr(
        "alpha.live.strategy_host.preflight_decision_contract_for_release",
        lambda release_obj, as_of_ts, pod_state_obj: {
            "release_id_str": release_obj.release_id_str,
            "pod_id_str": release_obj.pod_id_str,
            "strategy_import_str": release_obj.strategy_import_str,
            "decision_book_type_str": "incremental_entry_exit_book",
            "contract_status_str": "fail",
            "accepted_shape_count_int": 0,
            "unsupported_shape_count_int": 1,
            "unsupported_shape_example_dict_list": [
                {
                    "asset_str": "AAPL",
                    "order_class_str": "LimitOrder",
                    "unit_str": "value",
                    "target_bool": False,
                    "amount_float": 1000.0,
                    "trade_id_int": 1,
                }
            ],
            "error_str": "Unsupported incremental research order shape.",
        },
    )

    state_store_obj = LiveStateStore(db_path_str)
    detail_dict = preflight_contract_summary(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        releases_root_path_str=str(tmp_path / "releases"),
    )

    assert detail_dict["enabled_release_count_int"] == 1
    assert detail_dict["passed_release_count_int"] == 0
    assert detail_dict["failed_release_count_int"] == 1
    assert detail_dict["contract_report_dict_list"][0]["contract_status_str"] == "fail"
    assert detail_dict["contract_report_dict_list"][0]["unsupported_shape_count_int"] == 1


def test_cutover_v1_schema_exports_and_drops_legacy_tables(tmp_path: Path):
    db_path_str = str((tmp_path / "live.sqlite3").resolve())
    state_store_obj = LiveStateStore(db_path_str)

    with sqlite3.connect(db_path_str) as connection_obj:
        connection_obj.executescript(
            """
            CREATE TABLE order_plan (
                order_plan_id_int INTEGER PRIMARY KEY,
                status_str TEXT NOT NULL
            );
            CREATE TABLE order_intent (
                order_intent_id_int INTEGER PRIMARY KEY,
                order_plan_id_int INTEGER NOT NULL,
                asset_str TEXT NOT NULL
            );
            CREATE TABLE broker_order (
                broker_order_id_str TEXT PRIMARY KEY,
                order_plan_id_int INTEGER NOT NULL,
                status_str TEXT NOT NULL
            );
            CREATE TABLE fill (
                fill_id_int INTEGER PRIMARY KEY,
                order_plan_id_int INTEGER NOT NULL,
                asset_str TEXT NOT NULL
            );
            CREATE TABLE reconciliation_snapshot (
                reconciliation_snapshot_id_int INTEGER PRIMARY KEY,
                order_plan_id_int INTEGER NOT NULL,
                status_str TEXT NOT NULL
            );
            CREATE TABLE execution_quality_snapshot (
                execution_quality_snapshot_id_int INTEGER PRIMARY KEY,
                order_plan_id_int INTEGER NOT NULL,
                metric_float REAL NOT NULL
            );
            """
        )
        connection_obj.execute(
            "INSERT INTO order_plan(order_plan_id_int, status_str) VALUES (?, ?)",
            (1, "ready"),
        )
        connection_obj.execute(
            "INSERT INTO order_intent(order_intent_id_int, order_plan_id_int, asset_str) VALUES (?, ?, ?)",
            (1, 1, "AAPL"),
        )
        connection_obj.execute(
            "INSERT INTO broker_order(broker_order_id_str, order_plan_id_int, status_str) VALUES (?, ?, ?)",
            ("broker_001", 1, "submitted"),
        )
        connection_obj.execute(
            "INSERT INTO fill(fill_id_int, order_plan_id_int, asset_str) VALUES (?, ?, ?)",
            (1, 1, "AAPL"),
        )
        connection_obj.execute(
            "INSERT INTO reconciliation_snapshot(reconciliation_snapshot_id_int, order_plan_id_int, status_str) VALUES (?, ?, ?)",
            (1, 1, "pass"),
        )
        connection_obj.execute(
            "INSERT INTO execution_quality_snapshot(execution_quality_snapshot_id_int, order_plan_id_int, metric_float) VALUES (?, ?, ?)",
            (1, 1, 0.25),
        )

    detail_dict = cutover_v1_schema(
        state_store_obj=state_store_obj,
        as_of_ts=datetime(2024, 2, 1, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
        db_path_str=db_path_str,
        archive_root_path_str=str(tmp_path / "schema_archives"),
    )

    assert detail_dict["exported_table_count_int"] == 6
    assert detail_dict["dropped_table_count_int"] == 6
    assert detail_dict["remaining_v1_table_name_list"] == []
    assert Path(detail_dict["manifest_path_str"]).exists()

    exported_table_name_set = {
        table_export_dict["table_name_str"] for table_export_dict in detail_dict["table_export_dict_list"]
    }
    assert exported_table_name_set == {
        "order_intent",
        "broker_order",
        "fill",
        "reconciliation_snapshot",
        "execution_quality_snapshot",
        "order_plan",
    }
    for table_export_dict in detail_dict["table_export_dict_list"]:
        archive_file_path_obj = Path(table_export_dict["archive_file_path_str"])
        assert archive_file_path_obj.exists()
        assert table_export_dict["source_row_count_int"] == 1
        assert table_export_dict["exported_row_count_int"] == 1

    with sqlite3.connect(db_path_str) as connection_obj:
        remaining_table_name_set = {
            row_obj[0]
            for row_obj in connection_obj.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }

    assert "order_plan" not in remaining_table_name_set
    assert "order_intent" not in remaining_table_name_set
    assert "broker_order" not in remaining_table_name_set
    assert "fill" not in remaining_table_name_set
    assert "reconciliation_snapshot" not in remaining_table_name_set
    assert "execution_quality_snapshot" not in remaining_table_name_set
    assert "decision_plan" in remaining_table_name_set
    assert "vplan" in remaining_table_name_set
