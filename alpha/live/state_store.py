from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from alpha.live.models import (
    BrokerOrderFill,
    BrokerOrderRecord,
    ExecutionQualitySnapshot,
    FrozenOrderIntent,
    FrozenOrderPlan,
    LiveRelease,
    PodState,
    ReconciliationResult,
)


def _utc_now_ts() -> datetime:
    return datetime.now(timezone.utc)


def _serialize_timestamp_str(timestamp_ts: datetime) -> str:
    if timestamp_ts.tzinfo is None:
        return timestamp_ts.replace(tzinfo=timezone.utc).isoformat()
    return timestamp_ts.isoformat()


def _deserialize_timestamp_ts(timestamp_str: str) -> datetime:
    return datetime.fromisoformat(timestamp_str)


class LiveStateStore:
    def __init__(self, db_path_str: str):
        self.db_path_str = db_path_str
        Path(db_path_str).parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        connection_obj = sqlite3.connect(self.db_path_str)
        connection_obj.row_factory = sqlite3.Row
        return connection_obj

    def _initialize_schema(self) -> None:
        with self._connect() as connection_obj:
            connection_obj.executescript(
                """
                CREATE TABLE IF NOT EXISTS live_release (
                    release_id_str TEXT PRIMARY KEY,
                    pod_id_str TEXT NOT NULL,
                    user_id_str TEXT NOT NULL,
                    account_route_str TEXT NOT NULL,
                    source_path_str TEXT NOT NULL,
                    strategy_import_str TEXT NOT NULL,
                    mode_str TEXT NOT NULL,
                    session_calendar_id_str TEXT NOT NULL,
                    signal_clock_str TEXT NOT NULL,
                    execution_policy_str TEXT NOT NULL,
                    data_profile_str TEXT NOT NULL,
                    risk_profile_str TEXT NOT NULL,
                    enabled_bool INTEGER NOT NULL,
                    params_json_str TEXT NOT NULL,
                    updated_timestamp_str TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS pod_state (
                    pod_id_str TEXT PRIMARY KEY,
                    user_id_str TEXT NOT NULL,
                    account_route_str TEXT NOT NULL,
                    position_json_str TEXT NOT NULL,
                    cash_float REAL NOT NULL,
                    total_value_float REAL NOT NULL,
                    strategy_state_json_str TEXT NOT NULL,
                    updated_timestamp_str TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS order_plan (
                    order_plan_id_int INTEGER PRIMARY KEY AUTOINCREMENT,
                    release_id_str TEXT NOT NULL,
                    user_id_str TEXT NOT NULL,
                    pod_id_str TEXT NOT NULL,
                    account_route_str TEXT NOT NULL,
                    signal_timestamp_str TEXT NOT NULL,
                    submission_timestamp_str TEXT NOT NULL,
                    target_execution_timestamp_str TEXT NOT NULL,
                    submission_key_str TEXT,
                    execution_policy_str TEXT NOT NULL,
                    status_str TEXT NOT NULL,
                    snapshot_metadata_json_str TEXT NOT NULL,
                    strategy_state_json_str TEXT NOT NULL,
                    created_timestamp_str TEXT NOT NULL,
                    updated_timestamp_str TEXT NOT NULL,
                    UNIQUE(pod_id_str, signal_timestamp_str, execution_policy_str)
                );

                CREATE TABLE IF NOT EXISTS order_intent (
                    order_intent_id_int INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_plan_id_int INTEGER NOT NULL,
                    asset_str TEXT NOT NULL,
                    order_class_str TEXT NOT NULL,
                    unit_str TEXT NOT NULL,
                    amount_float REAL NOT NULL,
                    target_bool INTEGER NOT NULL,
                    trade_id_int INTEGER,
                    broker_order_type_str TEXT NOT NULL,
                    sizing_reference_price_float REAL NOT NULL,
                    portfolio_value_float REAL NOT NULL,
                    FOREIGN KEY(order_plan_id_int) REFERENCES order_plan(order_plan_id_int)
                );

                CREATE TABLE IF NOT EXISTS broker_order (
                    broker_order_record_id_int INTEGER PRIMARY KEY AUTOINCREMENT,
                    broker_order_id_str TEXT NOT NULL,
                    order_plan_id_int INTEGER NOT NULL,
                    order_intent_id_int INTEGER NOT NULL,
                    account_route_str TEXT NOT NULL,
                    asset_str TEXT NOT NULL,
                    broker_order_type_str TEXT NOT NULL,
                    unit_str TEXT NOT NULL,
                    amount_float REAL NOT NULL,
                    filled_amount_float REAL NOT NULL,
                    status_str TEXT NOT NULL,
                    submitted_timestamp_str TEXT NOT NULL,
                    raw_payload_json_str TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS fill (
                    fill_record_id_int INTEGER PRIMARY KEY AUTOINCREMENT,
                    broker_order_id_str TEXT NOT NULL,
                    order_plan_id_int INTEGER NOT NULL,
                    account_route_str TEXT NOT NULL,
                    asset_str TEXT NOT NULL,
                    fill_amount_float REAL NOT NULL,
                    fill_price_float REAL NOT NULL,
                    fill_timestamp_str TEXT NOT NULL,
                    raw_payload_json_str TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS reconciliation_snapshot (
                    reconciliation_snapshot_id_int INTEGER PRIMARY KEY AUTOINCREMENT,
                    pod_id_str TEXT NOT NULL,
                    order_plan_id_int INTEGER,
                    stage_str TEXT NOT NULL,
                    status_str TEXT NOT NULL,
                    mismatch_json_str TEXT NOT NULL,
                    model_position_json_str TEXT NOT NULL,
                    broker_position_json_str TEXT NOT NULL,
                    model_cash_float REAL NOT NULL,
                    broker_cash_float REAL NOT NULL,
                    created_timestamp_str TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS job_run (
                    job_run_id_int INTEGER PRIMARY KEY AUTOINCREMENT,
                    command_name_str TEXT NOT NULL,
                    started_timestamp_str TEXT NOT NULL,
                    completed_timestamp_str TEXT,
                    status_str TEXT NOT NULL,
                    detail_json_str TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS scheduler_lease (
                    lease_name_str TEXT PRIMARY KEY,
                    owner_token_str TEXT NOT NULL,
                    acquired_timestamp_str TEXT NOT NULL,
                    expires_timestamp_str TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS execution_quality_snapshot (
                    order_plan_id_int INTEGER PRIMARY KEY,
                    pod_id_str TEXT NOT NULL,
                    reference_notional_float REAL NOT NULL,
                    actual_notional_float REAL NOT NULL,
                    slippage_cash_float REAL NOT NULL,
                    slippage_bps_float REAL NOT NULL,
                    fill_count_int INTEGER NOT NULL,
                    computed_timestamp_str TEXT NOT NULL
                );
                """
            )
            connection_obj.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS fill_unique_event_idx
                ON fill (
                    broker_order_id_str,
                    fill_timestamp_str,
                    fill_amount_float,
                    fill_price_float
                )
                """
            )
            column_name_list = [
                row_obj["name"]
                for row_obj in connection_obj.execute("PRAGMA table_info(live_release)").fetchall()
            ]
            if "session_calendar_id_str" not in column_name_list:
                connection_obj.execute(
                    "ALTER TABLE live_release ADD COLUMN session_calendar_id_str TEXT NOT NULL DEFAULT 'XNYS'"
                )

            order_plan_column_name_list = [
                row_obj["name"]
                for row_obj in connection_obj.execute("PRAGMA table_info(order_plan)").fetchall()
            ]
            if "submission_key_str" not in order_plan_column_name_list:
                connection_obj.execute(
                    "ALTER TABLE order_plan ADD COLUMN submission_key_str TEXT"
                )

    def record_job_start(self, command_name_str: str) -> int:
        started_timestamp_str = _serialize_timestamp_str(_utc_now_ts())
        with self._connect() as connection_obj:
            cursor_obj = connection_obj.execute(
                """
                INSERT INTO job_run (
                    command_name_str,
                    started_timestamp_str,
                    status_str,
                    detail_json_str
                ) VALUES (?, ?, ?, ?)
                """,
                (command_name_str, started_timestamp_str, "running", "{}"),
            )
            return int(cursor_obj.lastrowid)

    def record_job_finish(
        self,
        job_run_id_int: int,
        status_str: str,
        detail_dict: dict,
    ) -> None:
        with self._connect() as connection_obj:
            connection_obj.execute(
                """
                UPDATE job_run
                SET completed_timestamp_str = ?, status_str = ?, detail_json_str = ?
                WHERE job_run_id_int = ?
                """,
                (
                    _serialize_timestamp_str(_utc_now_ts()),
                    status_str,
                    json.dumps(detail_dict, sort_keys=True),
                    int(job_run_id_int),
                ),
            )

    def acquire_scheduler_lease(
        self,
        lease_name_str: str,
        owner_token_str: str,
        expires_timestamp_ts: datetime,
    ) -> bool:
        current_timestamp_ts = _utc_now_ts()
        current_timestamp_str = _serialize_timestamp_str(current_timestamp_ts)
        expires_timestamp_str = _serialize_timestamp_str(expires_timestamp_ts)

        with self._connect() as connection_obj:
            connection_obj.execute("BEGIN IMMEDIATE")
            row_obj = connection_obj.execute(
                """
                SELECT owner_token_str, expires_timestamp_str
                FROM scheduler_lease
                WHERE lease_name_str = ?
                """,
                (lease_name_str,),
            ).fetchone()
            if row_obj is not None:
                existing_expires_timestamp_ts = _deserialize_timestamp_ts(row_obj["expires_timestamp_str"])
                if existing_expires_timestamp_ts > current_timestamp_ts and row_obj["owner_token_str"] != owner_token_str:
                    return False

            connection_obj.execute(
                """
                INSERT INTO scheduler_lease (
                    lease_name_str,
                    owner_token_str,
                    acquired_timestamp_str,
                    expires_timestamp_str
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(lease_name_str) DO UPDATE SET
                    owner_token_str = excluded.owner_token_str,
                    acquired_timestamp_str = excluded.acquired_timestamp_str,
                    expires_timestamp_str = excluded.expires_timestamp_str
                """,
                (
                    lease_name_str,
                    owner_token_str,
                    current_timestamp_str,
                    expires_timestamp_str,
                ),
            )
        return True

    def release_scheduler_lease(
        self,
        lease_name_str: str,
        owner_token_str: str,
    ) -> None:
        with self._connect() as connection_obj:
            connection_obj.execute(
                """
                DELETE FROM scheduler_lease
                WHERE lease_name_str = ?
                  AND owner_token_str = ?
                """,
                (lease_name_str, owner_token_str),
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
                    updated_timestamp_str
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(release_id_str) DO UPDATE SET
                    pod_id_str = excluded.pod_id_str,
                    user_id_str = excluded.user_id_str,
                    account_route_str = excluded.account_route_str,
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
                    updated_timestamp_str = excluded.updated_timestamp_str
                """,
                (
                    release_obj.release_id_str,
                    release_obj.pod_id_str,
                    release_obj.user_id_str,
                    release_obj.account_route_str,
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
                    _serialize_timestamp_str(_utc_now_ts()),
                ),
            )

    def upsert_release_list(self, release_list: list[LiveRelease]) -> None:
        for release_obj in release_list:
            self.upsert_release(release_obj)

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
        return [
            LiveRelease(
                release_id_str=row_obj["release_id_str"],
                user_id_str=row_obj["user_id_str"],
                pod_id_str=row_obj["pod_id_str"],
                account_route_str=row_obj["account_route_str"],
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
            )
            for row_obj in row_list
        ]

    def get_release_by_id(self, release_id_str: str) -> LiveRelease:
        with self._connect() as connection_obj:
            row_obj = connection_obj.execute(
                "SELECT * FROM live_release WHERE release_id_str = ?",
                (release_id_str,),
            ).fetchone()
        if row_obj is None:
            raise KeyError(f"Unknown release_id_str '{release_id_str}'.")
        return LiveRelease(
            release_id_str=row_obj["release_id_str"],
            user_id_str=row_obj["user_id_str"],
            pod_id_str=row_obj["pod_id_str"],
            account_route_str=row_obj["account_route_str"],
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
        )

    def get_pod_state(self, pod_id_str: str) -> PodState | None:
        with self._connect() as connection_obj:
            row_obj = connection_obj.execute(
                "SELECT * FROM pod_state WHERE pod_id_str = ?",
                (pod_id_str,),
            ).fetchone()
        if row_obj is None:
            return None
        return PodState(
            pod_id_str=row_obj["pod_id_str"],
            user_id_str=row_obj["user_id_str"],
            account_route_str=row_obj["account_route_str"],
            position_amount_map=json.loads(row_obj["position_json_str"]),
            cash_float=float(row_obj["cash_float"]),
            total_value_float=float(row_obj["total_value_float"]),
            strategy_state_dict=json.loads(row_obj["strategy_state_json_str"]),
            updated_timestamp_ts=_deserialize_timestamp_ts(row_obj["updated_timestamp_str"]),
        )

    def upsert_pod_state(self, pod_state_obj: PodState) -> None:
        with self._connect() as connection_obj:
            connection_obj.execute(
                """
                INSERT INTO pod_state (
                    pod_id_str,
                    user_id_str,
                    account_route_str,
                    position_json_str,
                    cash_float,
                    total_value_float,
                    strategy_state_json_str,
                    updated_timestamp_str
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pod_id_str) DO UPDATE SET
                    user_id_str = excluded.user_id_str,
                    account_route_str = excluded.account_route_str,
                    position_json_str = excluded.position_json_str,
                    cash_float = excluded.cash_float,
                    total_value_float = excluded.total_value_float,
                    strategy_state_json_str = excluded.strategy_state_json_str,
                    updated_timestamp_str = excluded.updated_timestamp_str
                """,
                (
                    pod_state_obj.pod_id_str,
                    pod_state_obj.user_id_str,
                    pod_state_obj.account_route_str,
                    json.dumps(pod_state_obj.position_amount_map, sort_keys=True),
                    float(pod_state_obj.cash_float),
                    float(pod_state_obj.total_value_float),
                    json.dumps(pod_state_obj.strategy_state_dict, sort_keys=True),
                    _serialize_timestamp_str(pod_state_obj.updated_timestamp_ts),
                ),
            )

    def has_active_order_plan(
        self,
        pod_id_str: str,
        signal_timestamp_ts: datetime,
        execution_policy_str: str,
    ) -> bool:
        with self._connect() as connection_obj:
            row_obj = connection_obj.execute(
                """
                SELECT COUNT(1) AS active_count_int
                FROM order_plan
                WHERE pod_id_str = ?
                  AND signal_timestamp_str = ?
                  AND execution_policy_str = ?
                  AND status_str IN ('frozen', 'submitting', 'submitted')
                """,
                (
                    pod_id_str,
                    _serialize_timestamp_str(signal_timestamp_ts),
                    execution_policy_str,
                ),
            ).fetchone()
        return int(row_obj["active_count_int"]) > 0

    def insert_order_plan(self, order_plan_obj: FrozenOrderPlan) -> FrozenOrderPlan:
        created_timestamp_str = _serialize_timestamp_str(_utc_now_ts())
        with self._connect() as connection_obj:
            cursor_obj = connection_obj.execute(
                """
                INSERT INTO order_plan (
                    release_id_str,
                    user_id_str,
                    pod_id_str,
                    account_route_str,
                    signal_timestamp_str,
                    submission_timestamp_str,
                    target_execution_timestamp_str,
                    submission_key_str,
                    execution_policy_str,
                    status_str,
                    snapshot_metadata_json_str,
                    strategy_state_json_str,
                    created_timestamp_str,
                    updated_timestamp_str
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order_plan_obj.release_id_str,
                    order_plan_obj.user_id_str,
                    order_plan_obj.pod_id_str,
                    order_plan_obj.account_route_str,
                    _serialize_timestamp_str(order_plan_obj.signal_timestamp_ts),
                    _serialize_timestamp_str(order_plan_obj.submission_timestamp_ts),
                    _serialize_timestamp_str(order_plan_obj.target_execution_timestamp_ts),
                    None,
                    order_plan_obj.execution_policy_str,
                    order_plan_obj.status_str,
                    json.dumps(order_plan_obj.snapshot_metadata_dict, sort_keys=True),
                    json.dumps(order_plan_obj.strategy_state_dict, sort_keys=True),
                    created_timestamp_str,
                    created_timestamp_str,
                ),
            )
            order_plan_id_int = int(cursor_obj.lastrowid)
            submission_key_str = f"order_plan:{order_plan_id_int}"
            connection_obj.execute(
                """
                UPDATE order_plan
                SET submission_key_str = ?, updated_timestamp_str = ?
                WHERE order_plan_id_int = ?
                """,
                (
                    submission_key_str,
                    created_timestamp_str,
                    order_plan_id_int,
                ),
            )
            for order_intent_obj in order_plan_obj.order_intent_list:
                connection_obj.execute(
                    """
                    INSERT INTO order_intent (
                        order_plan_id_int,
                        asset_str,
                        order_class_str,
                        unit_str,
                        amount_float,
                        target_bool,
                        trade_id_int,
                        broker_order_type_str,
                        sizing_reference_price_float,
                        portfolio_value_float
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        order_plan_id_int,
                        order_intent_obj.asset_str,
                        order_intent_obj.order_class_str,
                        order_intent_obj.unit_str,
                        float(order_intent_obj.amount_float),
                        int(order_intent_obj.target_bool),
                        order_intent_obj.trade_id_int,
                        order_intent_obj.broker_order_type_str,
                        float(order_intent_obj.sizing_reference_price_float),
                        float(order_intent_obj.portfolio_value_float),
                    ),
                )

        return FrozenOrderPlan(
            release_id_str=order_plan_obj.release_id_str,
            user_id_str=order_plan_obj.user_id_str,
            pod_id_str=order_plan_obj.pod_id_str,
            account_route_str=order_plan_obj.account_route_str,
            signal_timestamp_ts=order_plan_obj.signal_timestamp_ts,
            submission_timestamp_ts=order_plan_obj.submission_timestamp_ts,
            target_execution_timestamp_ts=order_plan_obj.target_execution_timestamp_ts,
            execution_policy_str=order_plan_obj.execution_policy_str,
            snapshot_metadata_dict=dict(order_plan_obj.snapshot_metadata_dict),
            strategy_state_dict=dict(order_plan_obj.strategy_state_dict),
            order_intent_list=list(order_plan_obj.order_intent_list),
            submission_key_str=submission_key_str,
            status_str=order_plan_obj.status_str,
            order_plan_id_int=order_plan_id_int,
        )

    def get_order_intent_row_list(self, order_plan_id_int: int) -> list[sqlite3.Row]:
        with self._connect() as connection_obj:
            row_list = connection_obj.execute(
                """
                SELECT *
                FROM order_intent
                WHERE order_plan_id_int = ?
                ORDER BY order_intent_id_int ASC
                """,
                (int(order_plan_id_int),),
            ).fetchall()
        return list(row_list)

    def get_submittable_order_plan_list(self, as_of_ts: datetime) -> list[FrozenOrderPlan]:
        with self._connect() as connection_obj:
            row_list = connection_obj.execute(
                """
                SELECT *
                FROM order_plan
                WHERE status_str = 'frozen'
                ORDER BY submission_timestamp_str ASC
                """,
            ).fetchall()
        submittable_plan_list: list[FrozenOrderPlan] = []
        for row_obj in row_list:
            order_plan_obj = self._row_to_order_plan(row_obj)
            if order_plan_obj.submission_timestamp_ts <= as_of_ts:
                submittable_plan_list.append(order_plan_obj)
        return submittable_plan_list

    def get_submitted_order_plan_list(self) -> list[FrozenOrderPlan]:
        with self._connect() as connection_obj:
            row_list = connection_obj.execute(
                """
                SELECT *
                FROM order_plan
                WHERE status_str = 'submitted'
                ORDER BY submission_timestamp_str ASC
                """
            ).fetchall()
        return [self._row_to_order_plan(row_obj) for row_obj in row_list]

    def get_execution_fill_input_row_list(
        self,
        order_plan_id_int: int,
    ) -> list[dict]:
        with self._connect() as connection_obj:
            row_list = connection_obj.execute(
                """
                SELECT
                    fill.fill_amount_float AS fill_amount_float,
                    fill.fill_price_float AS fill_price_float,
                    order_intent.sizing_reference_price_float AS reference_price_float
                FROM fill
                INNER JOIN broker_order
                    ON broker_order.broker_order_id_str = fill.broker_order_id_str
                INNER JOIN order_intent
                    ON order_intent.order_intent_id_int = broker_order.order_intent_id_int
                WHERE fill.order_plan_id_int = ?
                ORDER BY fill.fill_record_id_int ASC
                """,
                (int(order_plan_id_int),),
            ).fetchall()
        return [
            {
                "fill_amount_float": float(row_obj["fill_amount_float"]),
                "fill_price_float": float(row_obj["fill_price_float"]),
                "reference_price_float": float(row_obj["reference_price_float"]),
            }
            for row_obj in row_list
        ]

    def get_fill_row_dict_list(
        self,
        order_plan_id_int: int,
    ) -> list[dict]:
        with self._connect() as connection_obj:
            row_list = connection_obj.execute(
                """
                SELECT
                    asset_str,
                    fill_amount_float,
                    fill_price_float,
                    fill_timestamp_str
                FROM fill
                WHERE order_plan_id_int = ?
                ORDER BY fill_record_id_int ASC
                """,
                (int(order_plan_id_int),),
            ).fetchall()
        return [
            {
                "asset_str": row_obj["asset_str"],
                "fill_amount_float": float(row_obj["fill_amount_float"]),
                "fill_price_float": float(row_obj["fill_price_float"]),
                "fill_timestamp_str": row_obj["fill_timestamp_str"],
            }
            for row_obj in row_list
        ]

    def upsert_execution_quality_snapshot(
        self,
        execution_quality_snapshot_obj: ExecutionQualitySnapshot,
    ) -> None:
        with self._connect() as connection_obj:
            connection_obj.execute(
                """
                INSERT INTO execution_quality_snapshot (
                    order_plan_id_int,
                    pod_id_str,
                    reference_notional_float,
                    actual_notional_float,
                    slippage_cash_float,
                    slippage_bps_float,
                    fill_count_int,
                    computed_timestamp_str
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(order_plan_id_int) DO UPDATE SET
                    pod_id_str = excluded.pod_id_str,
                    reference_notional_float = excluded.reference_notional_float,
                    actual_notional_float = excluded.actual_notional_float,
                    slippage_cash_float = excluded.slippage_cash_float,
                    slippage_bps_float = excluded.slippage_bps_float,
                    fill_count_int = excluded.fill_count_int,
                    computed_timestamp_str = excluded.computed_timestamp_str
                """,
                (
                    int(execution_quality_snapshot_obj.order_plan_id_int),
                    execution_quality_snapshot_obj.pod_id_str,
                    float(execution_quality_snapshot_obj.reference_notional_float),
                    float(execution_quality_snapshot_obj.actual_notional_float),
                    float(execution_quality_snapshot_obj.slippage_cash_float),
                    float(execution_quality_snapshot_obj.slippage_bps_float),
                    int(execution_quality_snapshot_obj.fill_count_int),
                    _serialize_timestamp_str(execution_quality_snapshot_obj.computed_timestamp_ts),
                ),
            )

    def get_execution_quality_snapshot_by_plan(
        self,
        order_plan_id_int: int,
    ) -> ExecutionQualitySnapshot | None:
        with self._connect() as connection_obj:
            row_obj = connection_obj.execute(
                """
                SELECT *
                FROM execution_quality_snapshot
                WHERE order_plan_id_int = ?
                """,
                (int(order_plan_id_int),),
            ).fetchone()
        if row_obj is None:
            return None
        return ExecutionQualitySnapshot(
            order_plan_id_int=int(row_obj["order_plan_id_int"]),
            pod_id_str=row_obj["pod_id_str"],
            reference_notional_float=float(row_obj["reference_notional_float"]),
            actual_notional_float=float(row_obj["actual_notional_float"]),
            slippage_cash_float=float(row_obj["slippage_cash_float"]),
            slippage_bps_float=float(row_obj["slippage_bps_float"]),
            fill_count_int=int(row_obj["fill_count_int"]),
            computed_timestamp_ts=_deserialize_timestamp_ts(row_obj["computed_timestamp_str"]),
        )

    def get_latest_execution_quality_snapshot_for_pod(
        self,
        pod_id_str: str,
    ) -> ExecutionQualitySnapshot | None:
        with self._connect() as connection_obj:
            row_obj = connection_obj.execute(
                """
                SELECT execution_quality_snapshot.*
                FROM execution_quality_snapshot
                INNER JOIN order_plan
                    ON order_plan.order_plan_id_int = execution_quality_snapshot.order_plan_id_int
                WHERE execution_quality_snapshot.pod_id_str = ?
                ORDER BY order_plan.signal_timestamp_str DESC, execution_quality_snapshot.order_plan_id_int DESC
                LIMIT 1
                """,
                (pod_id_str,),
            ).fetchone()
        if row_obj is None:
            return None
        return ExecutionQualitySnapshot(
            order_plan_id_int=int(row_obj["order_plan_id_int"]),
            pod_id_str=row_obj["pod_id_str"],
            reference_notional_float=float(row_obj["reference_notional_float"]),
            actual_notional_float=float(row_obj["actual_notional_float"]),
            slippage_cash_float=float(row_obj["slippage_cash_float"]),
            slippage_bps_float=float(row_obj["slippage_bps_float"]),
            fill_count_int=int(row_obj["fill_count_int"]),
            computed_timestamp_ts=_deserialize_timestamp_ts(row_obj["computed_timestamp_str"]),
        )

    def get_latest_order_plan_for_pod(self, pod_id_str: str) -> FrozenOrderPlan | None:
        with self._connect() as connection_obj:
            row_obj = connection_obj.execute(
                """
                SELECT *
                FROM order_plan
                WHERE pod_id_str = ?
                ORDER BY signal_timestamp_str DESC, order_plan_id_int DESC
                LIMIT 1
                """,
                (pod_id_str,),
            ).fetchone()
        if row_obj is None:
            return None
        return self._row_to_order_plan(row_obj)

    def count_active_order_plans_for_window(
        self,
        pod_id_str: str,
        submission_timestamp_ts: datetime,
    ) -> int:
        with self._connect() as connection_obj:
            row_obj = connection_obj.execute(
                """
                SELECT COUNT(1) AS active_count_int
                FROM order_plan
                WHERE pod_id_str = ?
                  AND submission_timestamp_str = ?
                  AND status_str IN ('frozen', 'submitting', 'submitted')
                """,
                (pod_id_str, _serialize_timestamp_str(submission_timestamp_ts)),
            ).fetchone()
        return int(row_obj["active_count_int"])

    def claim_order_plan_for_submission(
        self,
        order_plan_id_int: int,
    ) -> bool:
        with self._connect() as connection_obj:
            cursor_obj = connection_obj.execute(
                """
                UPDATE order_plan
                SET status_str = ?, updated_timestamp_str = ?
                WHERE order_plan_id_int = ?
                  AND status_str = 'frozen'
                """,
                (
                    "submitting",
                    _serialize_timestamp_str(_utc_now_ts()),
                    int(order_plan_id_int),
                ),
            )
        return int(cursor_obj.rowcount) == 1

    def count_broker_orders_for_plan(
        self,
        order_plan_id_int: int,
    ) -> int:
        with self._connect() as connection_obj:
            row_obj = connection_obj.execute(
                """
                SELECT COUNT(1) AS broker_order_count_int
                FROM broker_order
                WHERE order_plan_id_int = ?
                """,
                (int(order_plan_id_int),),
            ).fetchone()
        return int(row_obj["broker_order_count_int"])

    def mark_order_plan_status(
        self,
        order_plan_id_int: int,
        status_str: str,
    ) -> None:
        with self._connect() as connection_obj:
            connection_obj.execute(
                """
                UPDATE order_plan
                SET status_str = ?, updated_timestamp_str = ?
                WHERE order_plan_id_int = ?
                """,
                (
                    status_str,
                    _serialize_timestamp_str(_utc_now_ts()),
                    int(order_plan_id_int),
                ),
            )

    def insert_reconciliation_snapshot(
        self,
        pod_id_str: str,
        order_plan_id_int: int | None,
        stage_str: str,
        reconciliation_result_obj: ReconciliationResult,
    ) -> None:
        with self._connect() as connection_obj:
            connection_obj.execute(
                """
                INSERT INTO reconciliation_snapshot (
                    pod_id_str,
                    order_plan_id_int,
                    stage_str,
                    status_str,
                    mismatch_json_str,
                    model_position_json_str,
                    broker_position_json_str,
                    model_cash_float,
                    broker_cash_float,
                    created_timestamp_str
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pod_id_str,
                    order_plan_id_int,
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

    def insert_broker_order_record_list(
        self,
        broker_order_record_list: list[BrokerOrderRecord],
    ) -> None:
        with self._connect() as connection_obj:
            for broker_order_record_obj in broker_order_record_list:
                connection_obj.execute(
                    """
                    INSERT INTO broker_order (
                        broker_order_id_str,
                        order_plan_id_int,
                        order_intent_id_int,
                        account_route_str,
                        asset_str,
                        broker_order_type_str,
                        unit_str,
                        amount_float,
                        filled_amount_float,
                        status_str,
                        submitted_timestamp_str,
                        raw_payload_json_str
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        broker_order_record_obj.broker_order_id_str,
                        broker_order_record_obj.order_plan_id_int,
                        broker_order_record_obj.order_intent_id_int,
                        broker_order_record_obj.account_route_str,
                        broker_order_record_obj.asset_str,
                        broker_order_record_obj.broker_order_type_str,
                        broker_order_record_obj.unit_str,
                        float(broker_order_record_obj.amount_float),
                        float(broker_order_record_obj.filled_amount_float),
                        broker_order_record_obj.status_str,
                        _serialize_timestamp_str(broker_order_record_obj.submitted_timestamp_ts),
                        json.dumps(broker_order_record_obj.raw_payload_dict, sort_keys=True),
                    ),
                )

    def insert_fill_list(
        self,
        fill_list: list[BrokerOrderFill],
    ) -> None:
        with self._connect() as connection_obj:
            for fill_obj in fill_list:
                connection_obj.execute(
                    """
                    INSERT OR IGNORE INTO fill (
                        broker_order_id_str,
                        order_plan_id_int,
                        account_route_str,
                        asset_str,
                        fill_amount_float,
                        fill_price_float,
                        fill_timestamp_str,
                        raw_payload_json_str
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        fill_obj.broker_order_id_str,
                        fill_obj.order_plan_id_int,
                        fill_obj.account_route_str,
                        fill_obj.asset_str,
                        float(fill_obj.fill_amount_float),
                        float(fill_obj.fill_price_float),
                        _serialize_timestamp_str(fill_obj.fill_timestamp_ts),
                        json.dumps(fill_obj.raw_payload_dict, sort_keys=True),
                    ),
                )

    def _row_to_order_plan(self, row_obj: sqlite3.Row) -> FrozenOrderPlan:
        order_intent_row_list = self.get_order_intent_row_list(int(row_obj["order_plan_id_int"]))
        order_intent_list = [
            FrozenOrderIntent(
                asset_str=order_intent_row_obj["asset_str"],
                order_class_str=order_intent_row_obj["order_class_str"],
                unit_str=order_intent_row_obj["unit_str"],
                amount_float=float(order_intent_row_obj["amount_float"]),
                target_bool=bool(order_intent_row_obj["target_bool"]),
                trade_id_int=order_intent_row_obj["trade_id_int"],
                broker_order_type_str=order_intent_row_obj["broker_order_type_str"],
                sizing_reference_price_float=float(order_intent_row_obj["sizing_reference_price_float"]),
                portfolio_value_float=float(order_intent_row_obj["portfolio_value_float"]),
            )
            for order_intent_row_obj in order_intent_row_list
        ]
        return FrozenOrderPlan(
            order_plan_id_int=int(row_obj["order_plan_id_int"]),
            release_id_str=row_obj["release_id_str"],
            user_id_str=row_obj["user_id_str"],
            pod_id_str=row_obj["pod_id_str"],
            account_route_str=row_obj["account_route_str"],
            signal_timestamp_ts=_deserialize_timestamp_ts(row_obj["signal_timestamp_str"]),
            submission_timestamp_ts=_deserialize_timestamp_ts(row_obj["submission_timestamp_str"]),
            target_execution_timestamp_ts=_deserialize_timestamp_ts(row_obj["target_execution_timestamp_str"]),
            execution_policy_str=row_obj["execution_policy_str"],
            snapshot_metadata_dict=json.loads(row_obj["snapshot_metadata_json_str"]),
            strategy_state_dict=json.loads(row_obj["strategy_state_json_str"]),
            order_intent_list=order_intent_list,
            submission_key_str=row_obj["submission_key_str"],
            status_str=row_obj["status_str"],
        )
