from __future__ import annotations

import json
import sqlite3
from typing import Iterable

from alpha.live.models import (
    BrokerOrderAck,
    BrokerOrderFill,
    BrokerOrderEvent,
    BrokerOrderRecord,
    BrokerSnapshot,
    CashLedgerEntry,
    DecisionPlan,
    LiveRelease,
    ReconciliationResult,
    SessionOpenPrice,
    VPlan,
    VPlanRow,
)
from alpha.live.state_store import (
    LiveStateStore as CoreLiveStateStore,
    _deserialize_timestamp_ts,
    _serialize_timestamp_str,
    _utc_now_ts,
)

class LiveStateStore(CoreLiveStateStore):
    def __init__(self, db_path_str: str):
        super().__init__(db_path_str)
        self._initialize_v2_schema()

    def _initialize_v2_schema(self) -> None:
        with self._connect() as connection_obj:
            connection_obj.executescript(
                """
                CREATE TABLE IF NOT EXISTS decision_plan (
                    decision_plan_id_int INTEGER PRIMARY KEY AUTOINCREMENT,
                    release_id_str TEXT NOT NULL,
                    user_id_str TEXT NOT NULL,
                    pod_id_str TEXT NOT NULL,
                    account_route_str TEXT NOT NULL,
                    signal_timestamp_str TEXT NOT NULL,
                    submission_timestamp_str TEXT NOT NULL,
                    target_execution_timestamp_str TEXT NOT NULL,
                    execution_policy_str TEXT NOT NULL,
                    decision_base_position_json_str TEXT NOT NULL,
                    decision_book_type_str TEXT NOT NULL DEFAULT 'incremental_entry_exit_book',
                    entry_target_weight_json_str TEXT NOT NULL DEFAULT '{}',
                    full_target_weight_json_str TEXT NOT NULL DEFAULT '{}',
                    target_weight_json_str TEXT NOT NULL,
                    exit_asset_json_str TEXT NOT NULL,
                    entry_priority_json_str TEXT NOT NULL,
                    cash_reserve_weight_float REAL NOT NULL,
                    preserve_untouched_positions_bool INTEGER NOT NULL,
                    rebalance_omitted_assets_to_zero_bool INTEGER NOT NULL DEFAULT 0,
                    snapshot_metadata_json_str TEXT NOT NULL,
                    strategy_state_json_str TEXT NOT NULL,
                    status_str TEXT NOT NULL,
                    created_timestamp_str TEXT NOT NULL,
                    updated_timestamp_str TEXT NOT NULL,
                    UNIQUE(pod_id_str, signal_timestamp_str, execution_policy_str)
                );

                CREATE TABLE IF NOT EXISTS vplan (
                    vplan_id_int INTEGER PRIMARY KEY AUTOINCREMENT,
                    release_id_str TEXT NOT NULL,
                    user_id_str TEXT NOT NULL,
                    pod_id_str TEXT NOT NULL,
                    account_route_str TEXT NOT NULL,
                    decision_plan_id_int INTEGER NOT NULL UNIQUE,
                    signal_timestamp_str TEXT NOT NULL,
                    submission_timestamp_str TEXT NOT NULL,
                    target_execution_timestamp_str TEXT NOT NULL,
                    execution_policy_str TEXT NOT NULL,
                    broker_snapshot_timestamp_str TEXT NOT NULL,
                    live_reference_snapshot_timestamp_str TEXT NOT NULL,
                    live_price_source_str TEXT NOT NULL,
                    net_liq_float REAL NOT NULL,
                    available_funds_float REAL,
                    excess_liquidity_float REAL,
                    pod_budget_fraction_float REAL NOT NULL,
                    pod_budget_float REAL NOT NULL,
                    current_broker_position_json_str TEXT NOT NULL,
                    live_reference_price_json_str TEXT NOT NULL,
                    live_reference_source_json_str TEXT NOT NULL DEFAULT '{}',
                    target_share_json_str TEXT NOT NULL,
                    order_delta_json_str TEXT NOT NULL,
                    submission_key_str TEXT,
                    status_str TEXT NOT NULL,
                    submit_ack_status_str TEXT NOT NULL DEFAULT 'not_checked',
                    ack_coverage_ratio_float REAL,
                    missing_ack_count_int INTEGER NOT NULL DEFAULT 0,
                    submit_ack_checked_timestamp_str TEXT,
                    created_timestamp_str TEXT NOT NULL,
                    updated_timestamp_str TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS vplan_row (
                    vplan_row_id_int INTEGER PRIMARY KEY AUTOINCREMENT,
                    vplan_id_int INTEGER NOT NULL,
                    asset_str TEXT NOT NULL,
                    current_share_float REAL NOT NULL,
                    target_share_float REAL NOT NULL,
                    order_delta_share_float REAL NOT NULL,
                    live_reference_price_float REAL NOT NULL,
                    estimated_target_notional_float REAL NOT NULL,
                    broker_order_type_str TEXT NOT NULL,
                    live_reference_source_str TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY(vplan_id_int) REFERENCES vplan(vplan_id_int)
                );

                CREATE TABLE IF NOT EXISTS broker_snapshot_cache (
                    account_route_str TEXT PRIMARY KEY,
                    snapshot_timestamp_str TEXT NOT NULL,
                    cash_float REAL NOT NULL,
                    total_value_float REAL NOT NULL,
                    net_liq_float REAL NOT NULL,
                    available_funds_float REAL,
                    excess_liquidity_float REAL,
                    cushion_float REAL,
                    position_json_str TEXT NOT NULL,
                    open_order_id_json_str TEXT NOT NULL,
                    updated_timestamp_str TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS vplan_reconciliation_snapshot (
                    vplan_reconciliation_snapshot_id_int INTEGER PRIMARY KEY AUTOINCREMENT,
                    pod_id_str TEXT NOT NULL,
                    decision_plan_id_int INTEGER,
                    vplan_id_int INTEGER,
                    stage_str TEXT NOT NULL,
                    status_str TEXT NOT NULL,
                    mismatch_json_str TEXT NOT NULL,
                    model_position_json_str TEXT NOT NULL,
                    broker_position_json_str TEXT NOT NULL,
                    model_cash_float REAL NOT NULL,
                    broker_cash_float REAL NOT NULL,
                    created_timestamp_str TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS vplan_broker_order (
                    broker_order_record_id_int INTEGER PRIMARY KEY AUTOINCREMENT,
                    broker_order_id_str TEXT NOT NULL,
                    decision_plan_id_int INTEGER,
                    vplan_id_int INTEGER NOT NULL,
                    account_route_str TEXT NOT NULL,
                    asset_str TEXT NOT NULL,
                    broker_order_type_str TEXT NOT NULL,
                    unit_str TEXT NOT NULL,
                    amount_float REAL NOT NULL,
                    filled_amount_float REAL NOT NULL,
                    remaining_amount_float REAL,
                    avg_fill_price_float REAL,
                    status_str TEXT NOT NULL,
                    last_status_timestamp_str TEXT,
                    submitted_timestamp_str TEXT NOT NULL,
                    submission_key_str TEXT,
                    order_request_key_str TEXT,
                    raw_payload_json_str TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS vplan_broker_order_event (
                    broker_order_event_id_int INTEGER PRIMARY KEY AUTOINCREMENT,
                    broker_order_id_str TEXT NOT NULL,
                    decision_plan_id_int INTEGER,
                    vplan_id_int INTEGER NOT NULL,
                    account_route_str TEXT NOT NULL,
                    asset_str TEXT NOT NULL,
                    status_str TEXT NOT NULL,
                    filled_amount_float REAL NOT NULL,
                    remaining_amount_float REAL,
                    avg_fill_price_float REAL,
                    event_timestamp_str TEXT NOT NULL,
                    event_source_str TEXT NOT NULL,
                    message_str TEXT NOT NULL,
                    submission_key_str TEXT,
                    order_request_key_str TEXT,
                    raw_payload_json_str TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS vplan_broker_ack (
                    broker_order_ack_id_int INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_plan_id_int INTEGER,
                    vplan_id_int INTEGER NOT NULL,
                    account_route_str TEXT NOT NULL,
                    order_request_key_str TEXT NOT NULL,
                    asset_str TEXT NOT NULL,
                    broker_order_type_str TEXT NOT NULL,
                    local_submit_ack_bool INTEGER NOT NULL,
                    broker_response_ack_bool INTEGER NOT NULL,
                    ack_status_str TEXT NOT NULL,
                    ack_source_str TEXT NOT NULL,
                    broker_order_id_str TEXT,
                    perm_id_int INTEGER,
                    response_timestamp_str TEXT,
                    raw_payload_json_str TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS vplan_fill (
                    fill_record_id_int INTEGER PRIMARY KEY AUTOINCREMENT,
                    broker_order_id_str TEXT NOT NULL,
                    decision_plan_id_int INTEGER,
                    vplan_id_int INTEGER NOT NULL,
                    account_route_str TEXT NOT NULL,
                    asset_str TEXT NOT NULL,
                    fill_amount_float REAL NOT NULL,
                    fill_price_float REAL NOT NULL,
                    official_open_price_float REAL,
                    open_price_source_str TEXT,
                    fill_timestamp_str TEXT NOT NULL,
                    raw_payload_json_str TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS session_open_price (
                    session_open_price_id_int INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_date_str TEXT NOT NULL,
                    account_route_str TEXT NOT NULL,
                    asset_str TEXT NOT NULL,
                    official_open_price_float REAL,
                    open_price_source_str TEXT,
                    snapshot_timestamp_str TEXT NOT NULL,
                    raw_payload_json_str TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS cash_ledger_entry (
                    cash_ledger_entry_id_int INTEGER PRIMARY KEY AUTOINCREMENT,
                    pod_id_str TEXT NOT NULL,
                    account_route_str TEXT NOT NULL,
                    vplan_id_int INTEGER NOT NULL,
                    broker_order_id_str TEXT NOT NULL,
                    asset_str TEXT NOT NULL,
                    entry_type_str TEXT NOT NULL,
                    cash_delta_float REAL NOT NULL,
                    entry_timestamp_str TEXT NOT NULL,
                    raw_payload_json_str TEXT NOT NULL
                );
                """
            )
            connection_obj.execute("DROP INDEX IF EXISTS vplan_broker_order_unique_order_idx")
            connection_obj.execute("DROP INDEX IF EXISTS vplan_broker_order_event_unique_idx")
            connection_obj.execute("DROP INDEX IF EXISTS vplan_fill_unique_event_idx")
            connection_obj.execute("DROP INDEX IF EXISTS vplan_broker_ack_unique_idx")
            connection_obj.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS vplan_broker_order_unique_order_idx
                ON vplan_broker_order (vplan_id_int, broker_order_id_str)
                """
            )
            connection_obj.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS vplan_broker_order_event_unique_idx
                ON vplan_broker_order_event (
                    vplan_id_int,
                    broker_order_id_str,
                    status_str,
                    event_timestamp_str,
                    message_str
                )
                """
            )
            connection_obj.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS vplan_fill_unique_event_idx
                ON vplan_fill (
                    vplan_id_int,
                    broker_order_id_str,
                    fill_timestamp_str,
                    fill_amount_float,
                    fill_price_float
                )
                """
            )
            connection_obj.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS vplan_broker_ack_unique_idx
                ON vplan_broker_ack (
                    vplan_id_int,
                    order_request_key_str
                )
                """
            )
            connection_obj.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS session_open_price_unique_idx
                ON session_open_price (
                    session_date_str,
                    account_route_str,
                    asset_str
                )
                """
            )
            connection_obj.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS cash_ledger_entry_unique_idx
                ON cash_ledger_entry (
                    vplan_id_int,
                    broker_order_id_str,
                    entry_type_str
                )
                """
            )

            live_release_column_name_list = [
                row_obj["name"]
                for row_obj in connection_obj.execute("PRAGMA table_info(live_release)").fetchall()
            ]
            if "pod_budget_fraction_float" not in live_release_column_name_list:
                connection_obj.execute(
                    """
                    ALTER TABLE live_release
                    ADD COLUMN pod_budget_fraction_float REAL NOT NULL DEFAULT 0.03
                    """
                )
            if "auto_submit_enabled_bool" not in live_release_column_name_list:
                connection_obj.execute(
                    """
                    ALTER TABLE live_release
                    ADD COLUMN auto_submit_enabled_bool INTEGER NOT NULL DEFAULT 0
                    """
                )
            if "broker_host_str" not in live_release_column_name_list:
                connection_obj.execute(
                    """
                    ALTER TABLE live_release
                    ADD COLUMN broker_host_str TEXT NOT NULL DEFAULT '127.0.0.1'
                    """
                )
            if "broker_port_int" not in live_release_column_name_list:
                connection_obj.execute(
                    """
                    ALTER TABLE live_release
                    ADD COLUMN broker_port_int INTEGER NOT NULL DEFAULT 7497
                    """
                )
            if "broker_client_id_int" not in live_release_column_name_list:
                connection_obj.execute(
                    """
                    ALTER TABLE live_release
                    ADD COLUMN broker_client_id_int INTEGER NOT NULL DEFAULT 31
                    """
                )
            if "broker_timeout_seconds_float" not in live_release_column_name_list:
                connection_obj.execute(
                    """
                    ALTER TABLE live_release
                    ADD COLUMN broker_timeout_seconds_float REAL NOT NULL DEFAULT 4.0
                    """
                )

            decision_plan_column_name_list = [
                row_obj["name"]
                for row_obj in connection_obj.execute("PRAGMA table_info(decision_plan)").fetchall()
            ]
            if "decision_book_type_str" not in decision_plan_column_name_list:
                connection_obj.execute(
                    """
                    ALTER TABLE decision_plan
                    ADD COLUMN decision_book_type_str TEXT NOT NULL DEFAULT 'incremental_entry_exit_book'
                    """
                )
            if "entry_target_weight_json_str" not in decision_plan_column_name_list:
                connection_obj.execute(
                    """
                    ALTER TABLE decision_plan
                    ADD COLUMN entry_target_weight_json_str TEXT NOT NULL DEFAULT '{}'
                    """
                )
            if "full_target_weight_json_str" not in decision_plan_column_name_list:
                connection_obj.execute(
                    """
                    ALTER TABLE decision_plan
                    ADD COLUMN full_target_weight_json_str TEXT NOT NULL DEFAULT '{}'
                    """
                )
            if "rebalance_omitted_assets_to_zero_bool" not in decision_plan_column_name_list:
                connection_obj.execute(
                    """
                    ALTER TABLE decision_plan
                    ADD COLUMN rebalance_omitted_assets_to_zero_bool INTEGER NOT NULL DEFAULT 0
                    """
                )

            vplan_column_name_list = [
                row_obj["name"]
                for row_obj in connection_obj.execute("PRAGMA table_info(vplan)").fetchall()
            ]
            if "live_reference_source_json_str" not in vplan_column_name_list:
                connection_obj.execute(
                    """
                    ALTER TABLE vplan
                    ADD COLUMN live_reference_source_json_str TEXT NOT NULL DEFAULT '{}'
                    """
                )
            if "submit_ack_status_str" not in vplan_column_name_list:
                connection_obj.execute(
                    """
                    ALTER TABLE vplan
                    ADD COLUMN submit_ack_status_str TEXT NOT NULL DEFAULT 'not_checked'
                    """
                )
            if "ack_coverage_ratio_float" not in vplan_column_name_list:
                connection_obj.execute(
                    """
                    ALTER TABLE vplan
                    ADD COLUMN ack_coverage_ratio_float REAL
                    """
                )
            if "missing_ack_count_int" not in vplan_column_name_list:
                connection_obj.execute(
                    """
                    ALTER TABLE vplan
                    ADD COLUMN missing_ack_count_int INTEGER NOT NULL DEFAULT 0
                    """
                )
            if "submit_ack_checked_timestamp_str" not in vplan_column_name_list:
                connection_obj.execute(
                    """
                    ALTER TABLE vplan
                    ADD COLUMN submit_ack_checked_timestamp_str TEXT
                    """
                )

            vplan_row_column_name_list = [
                row_obj["name"]
                for row_obj in connection_obj.execute("PRAGMA table_info(vplan_row)").fetchall()
            ]
            if "live_reference_source_str" not in vplan_row_column_name_list:
                connection_obj.execute(
                    """
                    ALTER TABLE vplan_row
                    ADD COLUMN live_reference_source_str TEXT NOT NULL DEFAULT ''
                    """
                )

            vplan_broker_order_column_name_list = [
                row_obj["name"]
                for row_obj in connection_obj.execute("PRAGMA table_info(vplan_broker_order)").fetchall()
            ]
            if "remaining_amount_float" not in vplan_broker_order_column_name_list:
                connection_obj.execute(
                    """
                    ALTER TABLE vplan_broker_order
                    ADD COLUMN remaining_amount_float REAL
                    """
                )
            if "avg_fill_price_float" not in vplan_broker_order_column_name_list:
                connection_obj.execute(
                    """
                    ALTER TABLE vplan_broker_order
                    ADD COLUMN avg_fill_price_float REAL
                    """
                )
            if "last_status_timestamp_str" not in vplan_broker_order_column_name_list:
                connection_obj.execute(
                    """
                    ALTER TABLE vplan_broker_order
                    ADD COLUMN last_status_timestamp_str TEXT
                    """
                )
            if "submission_key_str" not in vplan_broker_order_column_name_list:
                connection_obj.execute(
                    """
                    ALTER TABLE vplan_broker_order
                    ADD COLUMN submission_key_str TEXT
                    """
                )
            if "order_request_key_str" not in vplan_broker_order_column_name_list:
                connection_obj.execute(
                    """
                    ALTER TABLE vplan_broker_order
                    ADD COLUMN order_request_key_str TEXT
                    """
                )

            vplan_broker_order_event_column_name_list = [
                row_obj["name"]
                for row_obj in connection_obj.execute(
                    "PRAGMA table_info(vplan_broker_order_event)"
                ).fetchall()
            ]
            if "submission_key_str" not in vplan_broker_order_event_column_name_list:
                connection_obj.execute(
                    """
                    ALTER TABLE vplan_broker_order_event
                    ADD COLUMN submission_key_str TEXT
                    """
                )
            if "order_request_key_str" not in vplan_broker_order_event_column_name_list:
                connection_obj.execute(
                    """
                    ALTER TABLE vplan_broker_order_event
                    ADD COLUMN order_request_key_str TEXT
                    """
                )

            vplan_fill_column_name_list = [
                row_obj["name"]
                for row_obj in connection_obj.execute("PRAGMA table_info(vplan_fill)").fetchall()
            ]
            if "official_open_price_float" not in vplan_fill_column_name_list:
                connection_obj.execute(
                    """
                    ALTER TABLE vplan_fill
                    ADD COLUMN official_open_price_float REAL
                    """
                )
            if "open_price_source_str" not in vplan_fill_column_name_list:
                connection_obj.execute(
                    """
                    ALTER TABLE vplan_fill
                    ADD COLUMN open_price_source_str TEXT
                    """
                )

    def upsert_release(self, release_obj: LiveRelease) -> None:
        with self._connect() as connection_obj:
            connection_obj.execute(
                """
                INSERT INTO live_release (
                    release_id_str,
                    pod_id_str,
                    user_id_str,
                    account_route_str,
                    broker_host_str,
                    broker_port_int,
                    broker_client_id_int,
                    broker_timeout_seconds_float,
                    source_path_str,
                    strategy_import_str,
                    mode_str,
                    session_calendar_id_str,
                    signal_clock_str,
                    execution_policy_str,
                    data_profile_str,
                    risk_profile_str,
                    enabled_bool,
                    params_json_str,
                    pod_budget_fraction_float,
                    auto_submit_enabled_bool,
                    updated_timestamp_str
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(release_id_str) DO UPDATE SET
                    pod_id_str = excluded.pod_id_str,
                    user_id_str = excluded.user_id_str,
                    account_route_str = excluded.account_route_str,
                    broker_host_str = excluded.broker_host_str,
                    broker_port_int = excluded.broker_port_int,
                    broker_client_id_int = excluded.broker_client_id_int,
                    broker_timeout_seconds_float = excluded.broker_timeout_seconds_float,
                    source_path_str = excluded.source_path_str,
                    strategy_import_str = excluded.strategy_import_str,
                    mode_str = excluded.mode_str,
                    session_calendar_id_str = excluded.session_calendar_id_str,
                    signal_clock_str = excluded.signal_clock_str,
                    execution_policy_str = excluded.execution_policy_str,
                    data_profile_str = excluded.data_profile_str,
                    risk_profile_str = excluded.risk_profile_str,
                    enabled_bool = excluded.enabled_bool,
                    params_json_str = excluded.params_json_str,
                    pod_budget_fraction_float = excluded.pod_budget_fraction_float,
                    auto_submit_enabled_bool = excluded.auto_submit_enabled_bool,
                    updated_timestamp_str = excluded.updated_timestamp_str
                """,
                (
                    release_obj.release_id_str,
                    release_obj.pod_id_str,
                    release_obj.user_id_str,
                    release_obj.account_route_str,
                    release_obj.broker_host_str,
                    int(release_obj.broker_port_int),
                    int(release_obj.broker_client_id_int),
                    float(release_obj.broker_timeout_seconds_float),
                    release_obj.source_path_str,
                    release_obj.strategy_import_str,
                    release_obj.mode_str,
                    release_obj.session_calendar_id_str,
                    release_obj.signal_clock_str,
                    release_obj.execution_policy_str,
                    release_obj.data_profile_str,
                    release_obj.risk_profile_str,
                    int(release_obj.enabled_bool),
                    json.dumps(release_obj.params_dict, sort_keys=True),
                    float(release_obj.pod_budget_fraction_float),
                    int(release_obj.auto_submit_enabled_bool),
                    _serialize_timestamp_str(_utc_now_ts()),
                ),
            )

    def _row_to_release(self, row_obj: sqlite3.Row) -> LiveRelease:
        return LiveRelease(
            release_id_str=row_obj["release_id_str"],
            user_id_str=row_obj["user_id_str"],
            pod_id_str=row_obj["pod_id_str"],
            account_route_str=row_obj["account_route_str"],
            broker_host_str=row_obj["broker_host_str"],
            broker_port_int=int(row_obj["broker_port_int"]),
            broker_client_id_int=int(row_obj["broker_client_id_int"]),
            broker_timeout_seconds_float=float(row_obj["broker_timeout_seconds_float"]),
            strategy_import_str=row_obj["strategy_import_str"],
            mode_str=row_obj["mode_str"],
            session_calendar_id_str=row_obj["session_calendar_id_str"],
            signal_clock_str=row_obj["signal_clock_str"],
            execution_policy_str=row_obj["execution_policy_str"],
            data_profile_str=row_obj["data_profile_str"],
            params_dict=json.loads(row_obj["params_json_str"]),
            risk_profile_str=row_obj["risk_profile_str"],
            enabled_bool=bool(row_obj["enabled_bool"]),
            source_path_str=row_obj["source_path_str"],
            pod_budget_fraction_float=float(row_obj["pod_budget_fraction_float"]),
            auto_submit_enabled_bool=bool(row_obj["auto_submit_enabled_bool"]),
        )

    def get_enabled_release_list(self) -> list[LiveRelease]:
        with self._connect() as connection_obj:
            row_list = connection_obj.execute(
                """
                SELECT *
                FROM live_release
                WHERE enabled_bool = 1
                ORDER BY user_id_str ASC, pod_id_str ASC
                """
            ).fetchall()
        return [self._row_to_release(row_obj) for row_obj in row_list]

    def get_release_by_id(self, release_id_str: str) -> LiveRelease:
        with self._connect() as connection_obj:
            row_obj = connection_obj.execute(
                "SELECT * FROM live_release WHERE release_id_str = ?",
                (release_id_str,),
            ).fetchone()
        if row_obj is None:
            raise KeyError(f"Unknown release_id_str '{release_id_str}'.")
        return self._row_to_release(row_obj)

    def has_active_decision_plan(
        self,
        pod_id_str: str,
        signal_timestamp_ts,
        execution_policy_str: str,
    ) -> bool:
        with self._connect() as connection_obj:
            row_obj = connection_obj.execute(
                """
                SELECT COUNT(1) AS active_count_int
                FROM decision_plan
                WHERE pod_id_str = ?
                  AND signal_timestamp_str = ?
                  AND execution_policy_str = ?
                  AND status_str IN ('planned', 'vplan_ready', 'submitted')
                """,
                (
                    pod_id_str,
                    _serialize_timestamp_str(signal_timestamp_ts),
                    execution_policy_str,
                ),
            ).fetchone()
        return int(row_obj["active_count_int"]) > 0

    def insert_decision_plan(self, decision_plan_obj: DecisionPlan) -> DecisionPlan:
        created_timestamp_str = _serialize_timestamp_str(_utc_now_ts())
        with self._connect() as connection_obj:
            cursor_obj = connection_obj.execute(
                """
                INSERT INTO decision_plan (
                    release_id_str,
                    user_id_str,
                    pod_id_str,
                    account_route_str,
                    signal_timestamp_str,
                    submission_timestamp_str,
                    target_execution_timestamp_str,
                    execution_policy_str,
                    decision_base_position_json_str,
                    decision_book_type_str,
                    entry_target_weight_json_str,
                    full_target_weight_json_str,
                    target_weight_json_str,
                    exit_asset_json_str,
                    entry_priority_json_str,
                    cash_reserve_weight_float,
                    preserve_untouched_positions_bool,
                    rebalance_omitted_assets_to_zero_bool,
                    snapshot_metadata_json_str,
                    strategy_state_json_str,
                    status_str,
                    created_timestamp_str,
                    updated_timestamp_str
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_plan_obj.release_id_str,
                    decision_plan_obj.user_id_str,
                    decision_plan_obj.pod_id_str,
                    decision_plan_obj.account_route_str,
                    _serialize_timestamp_str(decision_plan_obj.signal_timestamp_ts),
                    _serialize_timestamp_str(decision_plan_obj.submission_timestamp_ts),
                    _serialize_timestamp_str(decision_plan_obj.target_execution_timestamp_ts),
                    decision_plan_obj.execution_policy_str,
                    json.dumps(decision_plan_obj.decision_base_position_map, sort_keys=True),
                    decision_plan_obj.decision_book_type_str,
                    json.dumps(decision_plan_obj.entry_target_weight_map_dict, sort_keys=True),
                    json.dumps(decision_plan_obj.full_target_weight_map_dict, sort_keys=True),
                    json.dumps(decision_plan_obj.target_weight_map, sort_keys=True),
                    json.dumps(sorted(decision_plan_obj.exit_asset_set)),
                    json.dumps(decision_plan_obj.entry_priority_list),
                    float(decision_plan_obj.cash_reserve_weight_float),
                    int(decision_plan_obj.preserve_untouched_positions_bool),
                    int(decision_plan_obj.rebalance_omitted_assets_to_zero_bool),
                    json.dumps(decision_plan_obj.snapshot_metadata_dict, sort_keys=True),
                    json.dumps(decision_plan_obj.strategy_state_dict, sort_keys=True),
                    decision_plan_obj.status_str,
                    created_timestamp_str,
                    created_timestamp_str,
                ),
            )
            decision_plan_id_int = int(cursor_obj.lastrowid)
        return DecisionPlan(
            **{
                **decision_plan_obj.__dict__,
                "decision_plan_id_int": decision_plan_id_int,
            }
        )

    def mark_decision_plan_status(self, decision_plan_id_int: int, status_str: str) -> None:
        with self._connect() as connection_obj:
            connection_obj.execute(
                """
                UPDATE decision_plan
                SET status_str = ?, updated_timestamp_str = ?
                WHERE decision_plan_id_int = ?
                """,
                (status_str, _serialize_timestamp_str(_utc_now_ts()), int(decision_plan_id_int)),
            )

    def get_latest_decision_plan_for_pod(self, pod_id_str: str) -> DecisionPlan | None:
        with self._connect() as connection_obj:
            row_obj = connection_obj.execute(
                """
                SELECT *
                FROM decision_plan
                WHERE pod_id_str = ?
                ORDER BY signal_timestamp_str DESC, decision_plan_id_int DESC
                LIMIT 1
                """,
                (pod_id_str,),
            ).fetchone()
        if row_obj is None:
            return None
        return self._row_to_decision_plan(row_obj)

    def get_first_vplan_for_pod(self, pod_id_str: str) -> VPlan | None:
        with self._connect() as connection_obj:
            row_obj = connection_obj.execute(
                """
                SELECT *
                FROM vplan
                WHERE pod_id_str = ?
                ORDER BY target_execution_timestamp_str ASC, vplan_id_int ASC
                LIMIT 1
                """,
                (pod_id_str,),
            ).fetchone()
        if row_obj is None:
            return None
        return self._row_to_vplan(row_obj)

    def get_decision_plan_by_id(self, decision_plan_id_int: int) -> DecisionPlan:
        with self._connect() as connection_obj:
            row_obj = connection_obj.execute(
                """
                SELECT *
                FROM decision_plan
                WHERE decision_plan_id_int = ?
                """,
                (int(decision_plan_id_int),),
            ).fetchone()
        if row_obj is None:
            raise KeyError(f"Unknown decision_plan_id_int '{decision_plan_id_int}'.")
        return self._row_to_decision_plan(row_obj)

    def get_due_decision_plan_list(self, as_of_ts) -> list[DecisionPlan]:
        with self._connect() as connection_obj:
            row_list = connection_obj.execute(
                """
                SELECT *
                FROM decision_plan
                WHERE status_str = 'planned'
                ORDER BY submission_timestamp_str ASC
                """
            ).fetchall()
        due_decision_plan_list: list[DecisionPlan] = []
        for row_obj in row_list:
            decision_plan_obj = self._row_to_decision_plan(row_obj)
            if decision_plan_obj.submission_timestamp_ts <= as_of_ts:
                due_decision_plan_list.append(decision_plan_obj)
        return due_decision_plan_list

    def get_expirable_decision_plan_list(self, as_of_ts) -> list[DecisionPlan]:
        with self._connect() as connection_obj:
            row_list = connection_obj.execute(
                """
                SELECT *
                FROM decision_plan
                WHERE status_str IN ('planned', 'vplan_ready')
                ORDER BY target_execution_timestamp_str ASC
                """
            ).fetchall()
        expirable_decision_plan_list: list[DecisionPlan] = []
        for row_obj in row_list:
            decision_plan_obj = self._row_to_decision_plan(row_obj)
            if decision_plan_obj.target_execution_timestamp_ts <= as_of_ts:
                expirable_decision_plan_list.append(decision_plan_obj)
        return expirable_decision_plan_list

    def insert_vplan(self, vplan_obj: VPlan) -> VPlan:
        created_timestamp_str = _serialize_timestamp_str(_utc_now_ts())
        live_reference_source_map_dict = (
            {
                str(asset_str): str(source_str)
                for asset_str, source_str in vplan_obj.live_reference_source_map_dict.items()
                if str(source_str) != ""
            }
            if len(vplan_obj.live_reference_source_map_dict) > 0
            else {
                str(vplan_row_obj.asset_str): str(
                    vplan_row_obj.live_reference_source_str or vplan_obj.live_price_source_str
                )
                for vplan_row_obj in vplan_obj.vplan_row_list
            }
        )
        if len(live_reference_source_map_dict) == 0:
            live_reference_source_map_dict = {
                str(asset_str): str(vplan_obj.live_price_source_str)
                for asset_str in vplan_obj.live_reference_price_map
            }
        with self._connect() as connection_obj:
            cursor_obj = connection_obj.execute(
                """
                INSERT INTO vplan (
                    release_id_str,
                    user_id_str,
                    pod_id_str,
                    account_route_str,
                    decision_plan_id_int,
                    signal_timestamp_str,
                    submission_timestamp_str,
                    target_execution_timestamp_str,
                    execution_policy_str,
                    broker_snapshot_timestamp_str,
                    live_reference_snapshot_timestamp_str,
                    live_price_source_str,
                    net_liq_float,
                    available_funds_float,
                    excess_liquidity_float,
                    pod_budget_fraction_float,
                    pod_budget_float,
                    current_broker_position_json_str,
                    live_reference_price_json_str,
                    live_reference_source_json_str,
                    target_share_json_str,
                    order_delta_json_str,
                    submission_key_str,
                    status_str,
                    submit_ack_status_str,
                    ack_coverage_ratio_float,
                    missing_ack_count_int,
                    submit_ack_checked_timestamp_str,
                    created_timestamp_str,
                    updated_timestamp_str
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    vplan_obj.release_id_str,
                    vplan_obj.user_id_str,
                    vplan_obj.pod_id_str,
                    vplan_obj.account_route_str,
                    int(vplan_obj.decision_plan_id_int),
                    _serialize_timestamp_str(vplan_obj.signal_timestamp_ts),
                    _serialize_timestamp_str(vplan_obj.submission_timestamp_ts),
                    _serialize_timestamp_str(vplan_obj.target_execution_timestamp_ts),
                    vplan_obj.execution_policy_str,
                    _serialize_timestamp_str(vplan_obj.broker_snapshot_timestamp_ts),
                    _serialize_timestamp_str(vplan_obj.live_reference_snapshot_timestamp_ts),
                    vplan_obj.live_price_source_str,
                    float(vplan_obj.net_liq_float),
                    None if vplan_obj.available_funds_float is None else float(vplan_obj.available_funds_float),
                    None if vplan_obj.excess_liquidity_float is None else float(vplan_obj.excess_liquidity_float),
                    float(vplan_obj.pod_budget_fraction_float),
                    float(vplan_obj.pod_budget_float),
                    json.dumps(vplan_obj.current_broker_position_map, sort_keys=True),
                    json.dumps(vplan_obj.live_reference_price_map, sort_keys=True),
                    json.dumps(live_reference_source_map_dict, sort_keys=True),
                    json.dumps(vplan_obj.target_share_map, sort_keys=True),
                    json.dumps(vplan_obj.order_delta_map, sort_keys=True),
                    vplan_obj.submission_key_str,
                    vplan_obj.status_str,
                    vplan_obj.submit_ack_status_str,
                    None
                    if vplan_obj.ack_coverage_ratio_float is None
                    else float(vplan_obj.ack_coverage_ratio_float),
                    int(vplan_obj.missing_ack_count_int),
                    None
                    if vplan_obj.submit_ack_checked_timestamp_ts is None
                    else _serialize_timestamp_str(vplan_obj.submit_ack_checked_timestamp_ts),
                    created_timestamp_str,
                    created_timestamp_str,
                ),
            )
            vplan_id_int = int(cursor_obj.lastrowid)
            for vplan_row_obj in vplan_obj.vplan_row_list:
                connection_obj.execute(
                    """
                    INSERT INTO vplan_row (
                        vplan_id_int,
                        asset_str,
                        current_share_float,
                        target_share_float,
                        order_delta_share_float,
                        live_reference_price_float,
                        estimated_target_notional_float,
                        broker_order_type_str,
                        live_reference_source_str
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        vplan_id_int,
                        vplan_row_obj.asset_str,
                        float(vplan_row_obj.current_share_float),
                        float(vplan_row_obj.target_share_float),
                        float(vplan_row_obj.order_delta_share_float),
                        float(vplan_row_obj.live_reference_price_float),
                        float(vplan_row_obj.estimated_target_notional_float),
                        vplan_row_obj.broker_order_type_str,
                        str(vplan_row_obj.live_reference_source_str or vplan_obj.live_price_source_str),
                    ),
                )
        return VPlan(
            **{
                **vplan_obj.__dict__,
                "vplan_id_int": vplan_id_int,
            }
        )

    def get_vplan_by_id(self, vplan_id_int: int) -> VPlan:
        with self._connect() as connection_obj:
            row_obj = connection_obj.execute(
                """
                SELECT *
                FROM vplan
                WHERE vplan_id_int = ?
                """,
                (int(vplan_id_int),),
            ).fetchone()
        if row_obj is None:
            raise KeyError(f"Unknown vplan_id_int '{vplan_id_int}'.")
        return self._row_to_vplan(row_obj)

    def get_latest_vplan_for_pod(self, pod_id_str: str) -> VPlan | None:
        with self._connect() as connection_obj:
            row_obj = connection_obj.execute(
                """
                SELECT *
                FROM vplan
                WHERE pod_id_str = ?
                ORDER BY signal_timestamp_str DESC, vplan_id_int DESC
                LIMIT 1
                """,
                (pod_id_str,),
            ).fetchone()
        if row_obj is None:
            return None
        return self._row_to_vplan(row_obj)

    def get_latest_vplan_for_decision(self, decision_plan_id_int: int) -> VPlan | None:
        with self._connect() as connection_obj:
            row_obj = connection_obj.execute(
                """
                SELECT *
                FROM vplan
                WHERE decision_plan_id_int = ?
                ORDER BY vplan_id_int DESC
                LIMIT 1
                """,
                (int(decision_plan_id_int),),
            ).fetchone()
        if row_obj is None:
            return None
        return self._row_to_vplan(row_obj)

    def claim_vplan_for_submission(self, vplan_id_int: int) -> bool:
        with self._connect() as connection_obj:
            cursor_obj = connection_obj.execute(
                """
                UPDATE vplan
                SET status_str = ?, updated_timestamp_str = ?
                WHERE vplan_id_int = ?
                  AND status_str = 'ready'
                """,
                ("submitting", _serialize_timestamp_str(_utc_now_ts()), int(vplan_id_int)),
            )
        return int(cursor_obj.rowcount) == 1

    def count_broker_orders_for_vplan(self, vplan_id_int: int) -> int:
        with self._connect() as connection_obj:
            row_obj = connection_obj.execute(
                """
                SELECT COUNT(1) AS broker_order_count_int
                FROM vplan_broker_order
                WHERE vplan_id_int = ?
                """,
                (int(vplan_id_int),),
            ).fetchone()
        return int(row_obj["broker_order_count_int"])

    def mark_vplan_status(self, vplan_id_int: int, status_str: str) -> None:
        with self._connect() as connection_obj:
            connection_obj.execute(
                """
                UPDATE vplan
                SET status_str = ?, updated_timestamp_str = ?
                WHERE vplan_id_int = ?
                """,
                (status_str, _serialize_timestamp_str(_utc_now_ts()), int(vplan_id_int)),
            )

    def update_vplan_submit_ack_summary(
        self,
        vplan_id_int: int,
        submit_ack_status_str: str,
        ack_coverage_ratio_float: float | None,
        missing_ack_count_int: int,
        submit_ack_checked_timestamp_ts,
    ) -> None:
        with self._connect() as connection_obj:
            connection_obj.execute(
                """
                UPDATE vplan
                SET
                    submit_ack_status_str = ?,
                    ack_coverage_ratio_float = ?,
                    missing_ack_count_int = ?,
                    submit_ack_checked_timestamp_str = ?,
                    updated_timestamp_str = ?
                WHERE vplan_id_int = ?
                """,
                (
                    str(submit_ack_status_str),
                    None
                    if ack_coverage_ratio_float is None
                    else float(ack_coverage_ratio_float),
                    int(missing_ack_count_int),
                    None
                    if submit_ack_checked_timestamp_ts is None
                    else _serialize_timestamp_str(submit_ack_checked_timestamp_ts),
                    _serialize_timestamp_str(_utc_now_ts()),
                    int(vplan_id_int),
                ),
            )

    def get_submitted_vplan_list(self) -> list[VPlan]:
        with self._connect() as connection_obj:
            row_list = connection_obj.execute(
                """
                SELECT *
                FROM vplan
                WHERE status_str IN ('submitted', 'submitting')
                ORDER BY submission_timestamp_str ASC
                """
            ).fetchall()
        return [self._row_to_vplan(row_obj) for row_obj in row_list]

    def upsert_broker_snapshot_cache(self, broker_snapshot_obj: BrokerSnapshot) -> None:
        with self._connect() as connection_obj:
            connection_obj.execute(
                """
                INSERT INTO broker_snapshot_cache (
                    account_route_str,
                    snapshot_timestamp_str,
                    cash_float,
                    total_value_float,
                    net_liq_float,
                    available_funds_float,
                    excess_liquidity_float,
                    cushion_float,
                    position_json_str,
                    open_order_id_json_str,
                    updated_timestamp_str
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(account_route_str) DO UPDATE SET
                    snapshot_timestamp_str = excluded.snapshot_timestamp_str,
                    cash_float = excluded.cash_float,
                    total_value_float = excluded.total_value_float,
                    net_liq_float = excluded.net_liq_float,
                    available_funds_float = excluded.available_funds_float,
                    excess_liquidity_float = excluded.excess_liquidity_float,
                    cushion_float = excluded.cushion_float,
                    position_json_str = excluded.position_json_str,
                    open_order_id_json_str = excluded.open_order_id_json_str,
                    updated_timestamp_str = excluded.updated_timestamp_str
                """,
                (
                    broker_snapshot_obj.account_route_str,
                    _serialize_timestamp_str(broker_snapshot_obj.snapshot_timestamp_ts),
                    float(broker_snapshot_obj.cash_float),
                    float(broker_snapshot_obj.total_value_float),
                    float(broker_snapshot_obj.net_liq_float),
                    None if broker_snapshot_obj.available_funds_float is None else float(broker_snapshot_obj.available_funds_float),
                    None if broker_snapshot_obj.excess_liquidity_float is None else float(broker_snapshot_obj.excess_liquidity_float),
                    None if broker_snapshot_obj.cushion_float is None else float(broker_snapshot_obj.cushion_float),
                    json.dumps(broker_snapshot_obj.position_amount_map, sort_keys=True),
                    json.dumps(list(broker_snapshot_obj.open_order_id_list)),
                    _serialize_timestamp_str(_utc_now_ts()),
                ),
            )

    def get_latest_broker_snapshot_for_account(self, account_route_str: str) -> BrokerSnapshot | None:
        with self._connect() as connection_obj:
            row_obj = connection_obj.execute(
                """
                SELECT *
                FROM broker_snapshot_cache
                WHERE account_route_str = ?
                """,
                (account_route_str,),
            ).fetchone()
        if row_obj is None:
            return None
        return BrokerSnapshot(
            account_route_str=row_obj["account_route_str"],
            snapshot_timestamp_ts=_deserialize_timestamp_ts(row_obj["snapshot_timestamp_str"]),
            cash_float=float(row_obj["cash_float"]),
            total_value_float=float(row_obj["total_value_float"]),
            net_liq_float=float(row_obj["net_liq_float"]),
            available_funds_float=None
            if row_obj["available_funds_float"] is None
            else float(row_obj["available_funds_float"]),
            excess_liquidity_float=None
            if row_obj["excess_liquidity_float"] is None
            else float(row_obj["excess_liquidity_float"]),
            cushion_float=None if row_obj["cushion_float"] is None else float(row_obj["cushion_float"]),
            position_amount_map=json.loads(row_obj["position_json_str"]),
            open_order_id_list=json.loads(row_obj["open_order_id_json_str"]),
        )

    def insert_vplan_reconciliation_snapshot(
        self,
        pod_id_str: str,
        decision_plan_id_int: int | None,
        vplan_id_int: int | None,
        stage_str: str,
        reconciliation_result_obj: ReconciliationResult,
    ) -> None:
        with self._connect() as connection_obj:
            connection_obj.execute(
                """
                INSERT INTO vplan_reconciliation_snapshot (
                    pod_id_str,
                    decision_plan_id_int,
                    vplan_id_int,
                    stage_str,
                    status_str,
                    mismatch_json_str,
                    model_position_json_str,
                    broker_position_json_str,
                    model_cash_float,
                    broker_cash_float,
                    created_timestamp_str
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pod_id_str,
                    decision_plan_id_int,
                    vplan_id_int,
                    stage_str,
                    reconciliation_result_obj.status_str,
                    json.dumps(reconciliation_result_obj.mismatch_dict, sort_keys=True),
                    json.dumps(reconciliation_result_obj.model_position_map, sort_keys=True),
                    json.dumps(reconciliation_result_obj.broker_position_map, sort_keys=True),
                    float(reconciliation_result_obj.model_cash_float),
                    float(reconciliation_result_obj.broker_cash_float),
                    _serialize_timestamp_str(_utc_now_ts()),
                ),
            )

    def has_post_execution_reconciliation_snapshot(self, vplan_id_int: int) -> bool:
        with self._connect() as connection_obj:
            row_obj = connection_obj.execute(
                """
                SELECT 1
                FROM vplan_reconciliation_snapshot
                WHERE vplan_id_int = ?
                  AND stage_str = 'post_execution'
                LIMIT 1
                """,
                (int(vplan_id_int),),
            ).fetchone()
        return row_obj is not None

    def upsert_vplan_broker_order_record_list(
        self,
        broker_order_record_list: Iterable[BrokerOrderRecord],
    ) -> None:
        with self._connect() as connection_obj:
            for broker_order_record_obj in broker_order_record_list:
                submission_key_str = (
                    broker_order_record_obj.submission_key_str
                    if broker_order_record_obj.submission_key_str is not None
                    else broker_order_record_obj.raw_payload_dict.get("submission_key_str")
                )
                order_request_key_str = (
                    broker_order_record_obj.order_request_key_str
                    if broker_order_record_obj.order_request_key_str is not None
                    else broker_order_record_obj.raw_payload_dict.get("order_request_key_str")
                )
                if order_request_key_str is not None:
                    existing_row_obj = connection_obj.execute(
                        """
                        SELECT broker_order_record_id_int
                        FROM vplan_broker_order
                        WHERE vplan_id_int = ?
                          AND order_request_key_str = ?
                        ORDER BY broker_order_record_id_int DESC
                        LIMIT 1
                        """,
                        (
                            broker_order_record_obj.vplan_id_int,
                            str(order_request_key_str),
                        ),
                    ).fetchone()
                elif submission_key_str is not None:
                    existing_row_obj = connection_obj.execute(
                        """
                        SELECT
                            broker_order_record_id_int,
                            broker_order_id_str
                        FROM vplan_broker_order
                        WHERE vplan_id_int = ?
                          AND submission_key_str = ?
                          AND asset_str = ?
                        ORDER BY broker_order_record_id_int DESC
                        LIMIT 1
                        """,
                        (
                            broker_order_record_obj.vplan_id_int,
                            str(submission_key_str),
                            broker_order_record_obj.asset_str,
                        ),
                    ).fetchone()
                else:
                    existing_row_obj = None
                if existing_row_obj is not None:
                    connection_obj.execute(
                        """
                        DELETE FROM vplan_broker_order
                        WHERE vplan_id_int = ?
                          AND broker_order_id_str = ?
                          AND broker_order_record_id_int != ?
                        """,
                        (
                            broker_order_record_obj.vplan_id_int,
                            broker_order_record_obj.broker_order_id_str,
                            existing_row_obj["broker_order_record_id_int"],
                        ),
                    )
                    connection_obj.execute(
                        """
                        UPDATE vplan_broker_order
                        SET
                            broker_order_id_str = ?,
                            decision_plan_id_int = ?,
                            vplan_id_int = ?,
                            account_route_str = ?,
                            asset_str = ?,
                            broker_order_type_str = ?,
                            unit_str = ?,
                            amount_float = ?,
                            filled_amount_float = ?,
                            remaining_amount_float = ?,
                            avg_fill_price_float = ?,
                            status_str = ?,
                            last_status_timestamp_str = ?,
                            submitted_timestamp_str = ?,
                            submission_key_str = ?,
                            order_request_key_str = ?,
                            raw_payload_json_str = ?
                        WHERE broker_order_record_id_int = ?
                        """,
                        (
                            broker_order_record_obj.broker_order_id_str,
                            broker_order_record_obj.decision_plan_id_int,
                            broker_order_record_obj.vplan_id_int,
                            broker_order_record_obj.account_route_str,
                            broker_order_record_obj.asset_str,
                            broker_order_record_obj.broker_order_type_str,
                            broker_order_record_obj.unit_str,
                            float(broker_order_record_obj.amount_float),
                            float(broker_order_record_obj.filled_amount_float),
                            None
                            if broker_order_record_obj.remaining_amount_float is None
                            else float(broker_order_record_obj.remaining_amount_float),
                            None
                            if broker_order_record_obj.avg_fill_price_float is None
                            else float(broker_order_record_obj.avg_fill_price_float),
                            broker_order_record_obj.status_str,
                            None
                            if broker_order_record_obj.last_status_timestamp_ts is None
                            else _serialize_timestamp_str(
                                broker_order_record_obj.last_status_timestamp_ts
                            ),
                            _serialize_timestamp_str(broker_order_record_obj.submitted_timestamp_ts),
                            None if submission_key_str is None else str(submission_key_str),
                            None if order_request_key_str is None else str(order_request_key_str),
                            json.dumps(broker_order_record_obj.raw_payload_dict, sort_keys=True),
                            existing_row_obj["broker_order_record_id_int"],
                        ),
                    )
                    continue
                connection_obj.execute(
                    """
                    INSERT INTO vplan_broker_order (
                        broker_order_id_str,
                        decision_plan_id_int,
                        vplan_id_int,
                        account_route_str,
                        asset_str,
                        broker_order_type_str,
                        unit_str,
                        amount_float,
                        filled_amount_float,
                        remaining_amount_float,
                        avg_fill_price_float,
                        status_str,
                        last_status_timestamp_str,
                        submitted_timestamp_str,
                        submission_key_str,
                        order_request_key_str,
                        raw_payload_json_str
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(vplan_id_int, broker_order_id_str) DO UPDATE SET
                        decision_plan_id_int = excluded.decision_plan_id_int,
                        vplan_id_int = excluded.vplan_id_int,
                        account_route_str = excluded.account_route_str,
                        asset_str = excluded.asset_str,
                        broker_order_type_str = excluded.broker_order_type_str,
                        unit_str = excluded.unit_str,
                        amount_float = excluded.amount_float,
                        filled_amount_float = excluded.filled_amount_float,
                        remaining_amount_float = excluded.remaining_amount_float,
                        avg_fill_price_float = excluded.avg_fill_price_float,
                        status_str = excluded.status_str,
                        last_status_timestamp_str = excluded.last_status_timestamp_str,
                        submitted_timestamp_str = excluded.submitted_timestamp_str,
                        submission_key_str = excluded.submission_key_str,
                        order_request_key_str = excluded.order_request_key_str,
                        raw_payload_json_str = excluded.raw_payload_json_str
                    """,
                    (
                        broker_order_record_obj.broker_order_id_str,
                        broker_order_record_obj.decision_plan_id_int,
                        broker_order_record_obj.vplan_id_int,
                        broker_order_record_obj.account_route_str,
                        broker_order_record_obj.asset_str,
                        broker_order_record_obj.broker_order_type_str,
                        broker_order_record_obj.unit_str,
                        float(broker_order_record_obj.amount_float),
                        float(broker_order_record_obj.filled_amount_float),
                        None
                        if broker_order_record_obj.remaining_amount_float is None
                        else float(broker_order_record_obj.remaining_amount_float),
                        None
                        if broker_order_record_obj.avg_fill_price_float is None
                        else float(broker_order_record_obj.avg_fill_price_float),
                        broker_order_record_obj.status_str,
                        None
                        if broker_order_record_obj.last_status_timestamp_ts is None
                        else _serialize_timestamp_str(broker_order_record_obj.last_status_timestamp_ts),
                        _serialize_timestamp_str(broker_order_record_obj.submitted_timestamp_ts),
                        None if submission_key_str is None else str(submission_key_str),
                        None if order_request_key_str is None else str(order_request_key_str),
                        json.dumps(broker_order_record_obj.raw_payload_dict, sort_keys=True),
                    ),
                )

    def insert_vplan_broker_order_event_list(
        self,
        broker_order_event_list: Iterable[BrokerOrderEvent],
    ) -> None:
        with self._connect() as connection_obj:
            for broker_order_event_obj in broker_order_event_list:
                event_timestamp_ts = (
                    broker_order_event_obj.event_timestamp_ts
                    if broker_order_event_obj.event_timestamp_ts is not None
                    else _utc_now_ts()
                )
                connection_obj.execute(
                    """
                    INSERT OR IGNORE INTO vplan_broker_order_event (
                        broker_order_id_str,
                        decision_plan_id_int,
                        vplan_id_int,
                        account_route_str,
                        asset_str,
                        status_str,
                        filled_amount_float,
                        remaining_amount_float,
                        avg_fill_price_float,
                        event_timestamp_str,
                        event_source_str,
                        message_str,
                        submission_key_str,
                        order_request_key_str,
                        raw_payload_json_str
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        broker_order_event_obj.broker_order_id_str,
                        broker_order_event_obj.decision_plan_id_int,
                        broker_order_event_obj.vplan_id_int,
                        broker_order_event_obj.account_route_str,
                        broker_order_event_obj.asset_str,
                        broker_order_event_obj.status_str,
                        float(broker_order_event_obj.filled_amount_float),
                        None
                        if broker_order_event_obj.remaining_amount_float is None
                        else float(broker_order_event_obj.remaining_amount_float),
                        None
                        if broker_order_event_obj.avg_fill_price_float is None
                        else float(broker_order_event_obj.avg_fill_price_float),
                        _serialize_timestamp_str(event_timestamp_ts),
                        broker_order_event_obj.event_source_str,
                        broker_order_event_obj.message_str,
                        broker_order_event_obj.submission_key_str,
                        broker_order_event_obj.order_request_key_str,
                        json.dumps(broker_order_event_obj.raw_payload_dict, sort_keys=True),
                    ),
                )

    def upsert_vplan_broker_ack_list(
        self,
        broker_order_ack_list: Iterable[BrokerOrderAck],
    ) -> None:
        with self._connect() as connection_obj:
            for broker_order_ack_obj in broker_order_ack_list:
                connection_obj.execute(
                    """
                    INSERT INTO vplan_broker_ack (
                        decision_plan_id_int,
                        vplan_id_int,
                        account_route_str,
                        order_request_key_str,
                        asset_str,
                        broker_order_type_str,
                        local_submit_ack_bool,
                        broker_response_ack_bool,
                        ack_status_str,
                        ack_source_str,
                        broker_order_id_str,
                        perm_id_int,
                        response_timestamp_str,
                        raw_payload_json_str
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(vplan_id_int, order_request_key_str) DO UPDATE SET
                        decision_plan_id_int = excluded.decision_plan_id_int,
                        account_route_str = excluded.account_route_str,
                        asset_str = excluded.asset_str,
                        broker_order_type_str = excluded.broker_order_type_str,
                        local_submit_ack_bool = excluded.local_submit_ack_bool,
                        broker_response_ack_bool = excluded.broker_response_ack_bool,
                        ack_status_str = excluded.ack_status_str,
                        ack_source_str = excluded.ack_source_str,
                        broker_order_id_str = excluded.broker_order_id_str,
                        perm_id_int = excluded.perm_id_int,
                        response_timestamp_str = excluded.response_timestamp_str,
                        raw_payload_json_str = excluded.raw_payload_json_str
                    """,
                    (
                        broker_order_ack_obj.decision_plan_id_int,
                        broker_order_ack_obj.vplan_id_int,
                        broker_order_ack_obj.account_route_str,
                        broker_order_ack_obj.order_request_key_str,
                        broker_order_ack_obj.asset_str,
                        broker_order_ack_obj.broker_order_type_str,
                        int(broker_order_ack_obj.local_submit_ack_bool),
                        int(broker_order_ack_obj.broker_response_ack_bool),
                        broker_order_ack_obj.ack_status_str,
                        broker_order_ack_obj.ack_source_str,
                        broker_order_ack_obj.broker_order_id_str,
                        broker_order_ack_obj.perm_id_int,
                        None
                        if broker_order_ack_obj.response_timestamp_ts is None
                        else _serialize_timestamp_str(broker_order_ack_obj.response_timestamp_ts),
                        json.dumps(broker_order_ack_obj.raw_payload_dict, sort_keys=True),
                    ),
                )

    def upsert_session_open_price_list(
        self,
        session_open_price_list: Iterable[SessionOpenPrice],
    ) -> None:
        with self._connect() as connection_obj:
            for session_open_price_obj in session_open_price_list:
                connection_obj.execute(
                    """
                    INSERT INTO session_open_price (
                        session_date_str,
                        account_route_str,
                        asset_str,
                        official_open_price_float,
                        open_price_source_str,
                        snapshot_timestamp_str,
                        raw_payload_json_str
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_date_str, account_route_str, asset_str) DO UPDATE SET
                        official_open_price_float = excluded.official_open_price_float,
                        open_price_source_str = excluded.open_price_source_str,
                        snapshot_timestamp_str = excluded.snapshot_timestamp_str,
                        raw_payload_json_str = excluded.raw_payload_json_str
                    """,
                    (
                        session_open_price_obj.session_date_str,
                        session_open_price_obj.account_route_str,
                        session_open_price_obj.asset_str,
                        None
                        if session_open_price_obj.official_open_price_float is None
                        else float(session_open_price_obj.official_open_price_float),
                        session_open_price_obj.open_price_source_str,
                        _serialize_timestamp_str(session_open_price_obj.snapshot_timestamp_ts),
                        json.dumps(session_open_price_obj.raw_payload_dict, sort_keys=True),
                    ),
                )

    def upsert_vplan_fill_list(self, fill_list: Iterable[BrokerOrderFill]) -> None:
        with self._connect() as connection_obj:
            for fill_obj in fill_list:
                connection_obj.execute(
                    """
                    INSERT INTO vplan_fill (
                        broker_order_id_str,
                        decision_plan_id_int,
                        vplan_id_int,
                        account_route_str,
                        asset_str,
                        fill_amount_float,
                        fill_price_float,
                        official_open_price_float,
                        open_price_source_str,
                        fill_timestamp_str,
                        raw_payload_json_str
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(
                        vplan_id_int,
                        broker_order_id_str,
                        fill_timestamp_str,
                        fill_amount_float,
                        fill_price_float
                    ) DO UPDATE SET
                        decision_plan_id_int = excluded.decision_plan_id_int,
                        vplan_id_int = excluded.vplan_id_int,
                        account_route_str = excluded.account_route_str,
                        asset_str = excluded.asset_str,
                        official_open_price_float = COALESCE(
                            excluded.official_open_price_float,
                            vplan_fill.official_open_price_float
                        ),
                        open_price_source_str = COALESCE(
                            excluded.open_price_source_str,
                            vplan_fill.open_price_source_str
                        ),
                        raw_payload_json_str = excluded.raw_payload_json_str
                    """,
                    (
                        fill_obj.broker_order_id_str,
                        fill_obj.decision_plan_id_int,
                        fill_obj.vplan_id_int,
                        fill_obj.account_route_str,
                        fill_obj.asset_str,
                        float(fill_obj.fill_amount_float),
                        float(fill_obj.fill_price_float),
                        None
                        if fill_obj.official_open_price_float is None
                        else float(fill_obj.official_open_price_float),
                        fill_obj.open_price_source_str,
                        _serialize_timestamp_str(fill_obj.fill_timestamp_ts),
                        json.dumps(fill_obj.raw_payload_dict, sort_keys=True),
                    ),
                )

    def insert_cash_ledger_entry_list(
        self,
        cash_ledger_entry_list: Iterable[CashLedgerEntry],
    ) -> None:
        with self._connect() as connection_obj:
            for cash_ledger_entry_obj in cash_ledger_entry_list:
                connection_obj.execute(
                    """
                    INSERT OR IGNORE INTO cash_ledger_entry (
                        pod_id_str,
                        account_route_str,
                        vplan_id_int,
                        broker_order_id_str,
                        asset_str,
                        entry_type_str,
                        cash_delta_float,
                        entry_timestamp_str,
                        raw_payload_json_str
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cash_ledger_entry_obj.pod_id_str,
                        cash_ledger_entry_obj.account_route_str,
                        int(cash_ledger_entry_obj.vplan_id_int),
                        cash_ledger_entry_obj.broker_order_id_str,
                        cash_ledger_entry_obj.asset_str,
                        cash_ledger_entry_obj.entry_type_str,
                        float(cash_ledger_entry_obj.cash_delta_float),
                        _serialize_timestamp_str(cash_ledger_entry_obj.entry_timestamp_ts),
                        json.dumps(cash_ledger_entry_obj.raw_payload_dict, sort_keys=True),
                    ),
                )

    def get_cash_ledger_row_dict_list_for_vplan(self, vplan_id_int: int) -> list[dict]:
        with self._connect() as connection_obj:
            row_list = connection_obj.execute(
                """
                SELECT
                    pod_id_str,
                    account_route_str,
                    vplan_id_int,
                    broker_order_id_str,
                    asset_str,
                    entry_type_str,
                    cash_delta_float,
                    entry_timestamp_str,
                    raw_payload_json_str
                FROM cash_ledger_entry
                WHERE vplan_id_int = ?
                ORDER BY cash_ledger_entry_id_int ASC
                """,
                (int(vplan_id_int),),
            ).fetchall()
        return [
            {
                "pod_id_str": str(row_obj["pod_id_str"]),
                "account_route_str": str(row_obj["account_route_str"]),
                "vplan_id_int": int(row_obj["vplan_id_int"]),
                "broker_order_id_str": str(row_obj["broker_order_id_str"]),
                "asset_str": str(row_obj["asset_str"]),
                "entry_type_str": str(row_obj["entry_type_str"]),
                "cash_delta_float": float(row_obj["cash_delta_float"]),
                "entry_timestamp_str": str(row_obj["entry_timestamp_str"]),
                "raw_payload_dict": json.loads(row_obj["raw_payload_json_str"]),
            }
            for row_obj in row_list
        ]

    def get_cash_ledger_delta_sum_float_for_pod(self, pod_id_str: str) -> float:
        with self._connect() as connection_obj:
            row_obj = connection_obj.execute(
                """
                SELECT COALESCE(SUM(cash_delta_float), 0.0) AS cash_delta_sum_float
                FROM cash_ledger_entry
                WHERE pod_id_str = ?
                """,
                (str(pod_id_str),),
            ).fetchone()
        return float(row_obj["cash_delta_sum_float"])

    def get_fill_row_dict_list_for_vplan(self, vplan_id_int: int) -> list[dict]:
        with self._connect() as connection_obj:
            row_list = connection_obj.execute(
                """
                SELECT
                    asset_str,
                    fill_amount_float,
                    fill_price_float,
                    official_open_price_float,
                    open_price_source_str,
                    fill_timestamp_str
                FROM vplan_fill
                WHERE vplan_id_int = ?
                ORDER BY fill_record_id_int ASC
                """,
                (int(vplan_id_int),),
            ).fetchall()
        return [
            {
                "asset_str": row_obj["asset_str"],
                "fill_amount_float": float(row_obj["fill_amount_float"]),
                "fill_price_float": float(row_obj["fill_price_float"]),
                "official_open_price_float": None
                if row_obj["official_open_price_float"] is None
                else float(row_obj["official_open_price_float"]),
                "open_price_source_str": row_obj["open_price_source_str"],
                "fill_timestamp_str": row_obj["fill_timestamp_str"],
            }
            for row_obj in row_list
        ]

    def get_broker_order_row_dict_list_for_vplan(self, vplan_id_int: int) -> list[dict]:
        with self._connect() as connection_obj:
            row_list = connection_obj.execute(
                """
                SELECT
                    broker_order_id_str,
                    asset_str,
                    order_request_key_str,
                    broker_order_type_str,
                    unit_str,
                    amount_float,
                    filled_amount_float,
                    remaining_amount_float,
                    avg_fill_price_float,
                    status_str,
                    last_status_timestamp_str,
                    submitted_timestamp_str,
                    submission_key_str
                FROM vplan_broker_order
                WHERE vplan_id_int = ?
                ORDER BY broker_order_record_id_int ASC
                """,
                (int(vplan_id_int),),
            ).fetchall()
        return [
            {
                "broker_order_id_str": str(row_obj["broker_order_id_str"]),
                "asset_str": str(row_obj["asset_str"]),
                "order_request_key_str": row_obj["order_request_key_str"],
                "broker_order_type_str": str(row_obj["broker_order_type_str"]),
                "unit_str": str(row_obj["unit_str"]),
                "amount_float": float(row_obj["amount_float"]),
                "filled_amount_float": float(row_obj["filled_amount_float"]),
                "remaining_amount_float": None
                if row_obj["remaining_amount_float"] is None
                else float(row_obj["remaining_amount_float"]),
                "avg_fill_price_float": None
                if row_obj["avg_fill_price_float"] is None
                else float(row_obj["avg_fill_price_float"]),
                "status_str": str(row_obj["status_str"]),
                "last_status_timestamp_str": row_obj["last_status_timestamp_str"],
                "submitted_timestamp_str": str(row_obj["submitted_timestamp_str"]),
                "submission_key_str": row_obj["submission_key_str"],
            }
            for row_obj in row_list
        ]

    def get_broker_order_event_row_dict_list_for_vplan(self, vplan_id_int: int) -> list[dict]:
        with self._connect() as connection_obj:
            row_list = connection_obj.execute(
                """
                SELECT
                    broker_order_id_str,
                    asset_str,
                    order_request_key_str,
                    status_str,
                    filled_amount_float,
                    remaining_amount_float,
                    avg_fill_price_float,
                    event_timestamp_str,
                    event_source_str,
                    message_str,
                    submission_key_str
                FROM vplan_broker_order_event
                WHERE vplan_id_int = ?
                ORDER BY broker_order_event_id_int ASC
                """,
                (int(vplan_id_int),),
            ).fetchall()
        return [
            {
                "broker_order_id_str": str(row_obj["broker_order_id_str"]),
                "asset_str": str(row_obj["asset_str"]),
                "order_request_key_str": row_obj["order_request_key_str"],
                "status_str": str(row_obj["status_str"]),
                "filled_amount_float": float(row_obj["filled_amount_float"]),
                "remaining_amount_float": None
                if row_obj["remaining_amount_float"] is None
                else float(row_obj["remaining_amount_float"]),
                "avg_fill_price_float": None
                if row_obj["avg_fill_price_float"] is None
                else float(row_obj["avg_fill_price_float"]),
                "event_timestamp_str": str(row_obj["event_timestamp_str"]),
                "event_source_str": str(row_obj["event_source_str"]),
                "message_str": row_obj["message_str"],
                "submission_key_str": row_obj["submission_key_str"],
            }
            for row_obj in row_list
        ]

    def get_broker_ack_row_dict_list_for_vplan(self, vplan_id_int: int) -> list[dict]:
        with self._connect() as connection_obj:
            row_list = connection_obj.execute(
                """
                SELECT
                    order_request_key_str,
                    asset_str,
                    broker_order_type_str,
                    local_submit_ack_bool,
                    broker_response_ack_bool,
                    ack_status_str,
                    ack_source_str,
                    broker_order_id_str,
                    perm_id_int,
                    response_timestamp_str
                FROM vplan_broker_ack
                WHERE vplan_id_int = ?
                ORDER BY asset_str ASC, order_request_key_str ASC
                """,
                (int(vplan_id_int),),
            ).fetchall()
        return [
            {
                "order_request_key_str": str(row_obj["order_request_key_str"]),
                "asset_str": str(row_obj["asset_str"]),
                "broker_order_type_str": str(row_obj["broker_order_type_str"]),
                "local_submit_ack_bool": bool(row_obj["local_submit_ack_bool"]),
                "broker_response_ack_bool": bool(row_obj["broker_response_ack_bool"]),
                "ack_status_str": str(row_obj["ack_status_str"]),
                "ack_source_str": str(row_obj["ack_source_str"]),
                "broker_order_id_str": row_obj["broker_order_id_str"],
                "perm_id_int": None
                if row_obj["perm_id_int"] is None
                else int(row_obj["perm_id_int"]),
                "response_timestamp_str": row_obj["response_timestamp_str"],
            }
            for row_obj in row_list
        ]

    def get_session_open_price_map_dict(
        self,
        account_route_str: str,
        session_date_str: str,
    ) -> dict[str, SessionOpenPrice]:
        with self._connect() as connection_obj:
            row_list = connection_obj.execute(
                """
                SELECT *
                FROM session_open_price
                WHERE account_route_str = ?
                  AND session_date_str = ?
                ORDER BY asset_str ASC
                """,
                (account_route_str, session_date_str),
            ).fetchall()
        return {
            str(row_obj["asset_str"]): SessionOpenPrice(
                session_date_str=str(row_obj["session_date_str"]),
                account_route_str=str(row_obj["account_route_str"]),
                asset_str=str(row_obj["asset_str"]),
                official_open_price_float=None
                if row_obj["official_open_price_float"] is None
                else float(row_obj["official_open_price_float"]),
                open_price_source_str=row_obj["open_price_source_str"],
                snapshot_timestamp_ts=_deserialize_timestamp_ts(row_obj["snapshot_timestamp_str"]),
                raw_payload_dict=json.loads(row_obj["raw_payload_json_str"]),
            )
            for row_obj in row_list
        }

    def _row_to_decision_plan(self, row_obj: sqlite3.Row) -> DecisionPlan:
        return DecisionPlan(
            release_id_str=row_obj["release_id_str"],
            user_id_str=row_obj["user_id_str"],
            pod_id_str=row_obj["pod_id_str"],
            account_route_str=row_obj["account_route_str"],
            signal_timestamp_ts=_deserialize_timestamp_ts(row_obj["signal_timestamp_str"]),
            submission_timestamp_ts=_deserialize_timestamp_ts(row_obj["submission_timestamp_str"]),
            target_execution_timestamp_ts=_deserialize_timestamp_ts(row_obj["target_execution_timestamp_str"]),
            execution_policy_str=row_obj["execution_policy_str"],
            decision_base_position_map=json.loads(row_obj["decision_base_position_json_str"]),
            decision_book_type_str=(
                row_obj["decision_book_type_str"]
                if "decision_book_type_str" in row_obj.keys()
                else "incremental_entry_exit_book"
            ),
            entry_target_weight_map_dict=(
                json.loads(row_obj["entry_target_weight_json_str"])
                if "entry_target_weight_json_str" in row_obj.keys()
                else json.loads(row_obj["target_weight_json_str"])
            ),
            full_target_weight_map_dict=(
                json.loads(row_obj["full_target_weight_json_str"])
                if "full_target_weight_json_str" in row_obj.keys()
                else {}
            ),
            target_weight_map=json.loads(row_obj["target_weight_json_str"]),
            exit_asset_set=set(json.loads(row_obj["exit_asset_json_str"])),
            entry_priority_list=list(json.loads(row_obj["entry_priority_json_str"])),
            cash_reserve_weight_float=float(row_obj["cash_reserve_weight_float"]),
            snapshot_metadata_dict=json.loads(row_obj["snapshot_metadata_json_str"]),
            strategy_state_dict=json.loads(row_obj["strategy_state_json_str"]),
            preserve_untouched_positions_bool=bool(row_obj["preserve_untouched_positions_bool"]),
            rebalance_omitted_assets_to_zero_bool=bool(
                row_obj["rebalance_omitted_assets_to_zero_bool"]
            )
            if "rebalance_omitted_assets_to_zero_bool" in row_obj.keys()
            else False,
            status_str=row_obj["status_str"],
            decision_plan_id_int=int(row_obj["decision_plan_id_int"]),
        )

    def _row_to_vplan(self, row_obj: sqlite3.Row) -> VPlan:
        with self._connect() as connection_obj:
            vplan_row_row_list = connection_obj.execute(
                """
                SELECT *
                FROM vplan_row
                WHERE vplan_id_int = ?
                ORDER BY asset_str ASC, vplan_row_id_int ASC
                """,
                (int(row_obj["vplan_id_int"]),),
            ).fetchall()
        vplan_row_list = [
            VPlanRow(
                asset_str=vplan_row_obj["asset_str"],
                current_share_float=float(vplan_row_obj["current_share_float"]),
                target_share_float=float(vplan_row_obj["target_share_float"]),
                order_delta_share_float=float(vplan_row_obj["order_delta_share_float"]),
                live_reference_price_float=float(vplan_row_obj["live_reference_price_float"]),
                estimated_target_notional_float=float(vplan_row_obj["estimated_target_notional_float"]),
                broker_order_type_str=vplan_row_obj["broker_order_type_str"],
                live_reference_source_str=str(
                    vplan_row_obj["live_reference_source_str"] or row_obj["live_price_source_str"]
                ),
            )
            for vplan_row_obj in vplan_row_row_list
        ]
        live_reference_source_map_dict = (
            {}
            if row_obj["live_reference_source_json_str"] is None
            else json.loads(row_obj["live_reference_source_json_str"])
        )
        if len(live_reference_source_map_dict) == 0:
            live_reference_source_map_dict = {
                str(vplan_row_obj.asset_str): str(vplan_row_obj.live_reference_source_str)
                for vplan_row_obj in vplan_row_list
                if str(vplan_row_obj.live_reference_source_str) != ""
            }
        if len(live_reference_source_map_dict) == 0:
            live_reference_source_map_dict = {
                str(asset_str): str(row_obj["live_price_source_str"])
                for asset_str in json.loads(row_obj["live_reference_price_json_str"])
            }
        return VPlan(
            release_id_str=row_obj["release_id_str"],
            user_id_str=row_obj["user_id_str"],
            pod_id_str=row_obj["pod_id_str"],
            account_route_str=row_obj["account_route_str"],
            decision_plan_id_int=int(row_obj["decision_plan_id_int"]),
            signal_timestamp_ts=_deserialize_timestamp_ts(row_obj["signal_timestamp_str"]),
            submission_timestamp_ts=_deserialize_timestamp_ts(row_obj["submission_timestamp_str"]),
            target_execution_timestamp_ts=_deserialize_timestamp_ts(row_obj["target_execution_timestamp_str"]),
            execution_policy_str=row_obj["execution_policy_str"],
            broker_snapshot_timestamp_ts=_deserialize_timestamp_ts(row_obj["broker_snapshot_timestamp_str"]),
            live_reference_snapshot_timestamp_ts=_deserialize_timestamp_ts(
                row_obj["live_reference_snapshot_timestamp_str"]
            ),
            live_price_source_str=row_obj["live_price_source_str"],
            net_liq_float=float(row_obj["net_liq_float"]),
            available_funds_float=None
            if row_obj["available_funds_float"] is None
            else float(row_obj["available_funds_float"]),
            excess_liquidity_float=None
            if row_obj["excess_liquidity_float"] is None
            else float(row_obj["excess_liquidity_float"]),
            pod_budget_fraction_float=float(row_obj["pod_budget_fraction_float"]),
            pod_budget_float=float(row_obj["pod_budget_float"]),
            current_broker_position_map=json.loads(row_obj["current_broker_position_json_str"]),
            live_reference_price_map=json.loads(row_obj["live_reference_price_json_str"]),
            target_share_map=json.loads(row_obj["target_share_json_str"]),
            order_delta_map=json.loads(row_obj["order_delta_json_str"]),
            vplan_row_list=vplan_row_list,
            live_reference_source_map_dict=live_reference_source_map_dict,
            submission_key_str=row_obj["submission_key_str"],
            status_str=row_obj["status_str"],
            submit_ack_status_str=(
                str(row_obj["submit_ack_status_str"])
                if "submit_ack_status_str" in row_obj.keys()
                else "not_checked"
            ),
            ack_coverage_ratio_float=(
                None
                if "ack_coverage_ratio_float" not in row_obj.keys()
                or row_obj["ack_coverage_ratio_float"] is None
                else float(row_obj["ack_coverage_ratio_float"])
            ),
            missing_ack_count_int=(
                0
                if "missing_ack_count_int" not in row_obj.keys()
                else int(row_obj["missing_ack_count_int"])
            ),
            submit_ack_checked_timestamp_ts=(
                None
                if "submit_ack_checked_timestamp_str" not in row_obj.keys()
                or row_obj["submit_ack_checked_timestamp_str"] is None
                else _deserialize_timestamp_ts(row_obj["submit_ack_checked_timestamp_str"])
            ),
            vplan_id_int=int(row_obj["vplan_id_int"]),
        )
