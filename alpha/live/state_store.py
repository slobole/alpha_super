from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from alpha.live.models import LiveRelease, PodState


SHARED_CORE_TABLE_NAME_TUPLE: tuple[str, ...] = (
    "live_release",
    "pod_state",
    "job_run",
    "scheduler_lease",
)

V1_EXECUTION_TABLE_NAME_TUPLE: tuple[str, ...] = (
    "order_intent",
    "broker_order",
    "fill",
    "reconciliation_snapshot",
    "execution_quality_snapshot",
    "order_plan",
)

_TABLE_NAME_PATTERN_OBJ = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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
                    broker_host_str TEXT NOT NULL DEFAULT '127.0.0.1',
                    broker_port_int INTEGER NOT NULL DEFAULT 7497,
                    broker_client_id_int INTEGER NOT NULL DEFAULT 31,
                    broker_timeout_seconds_float REAL NOT NULL DEFAULT 4.0,
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
                    pod_budget_fraction_float REAL NOT NULL DEFAULT 0.03,
                    auto_submit_enabled_bool INTEGER NOT NULL DEFAULT 0,
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
                """
            )

            live_release_column_name_list = [
                row_obj["name"]
                for row_obj in connection_obj.execute("PRAGMA table_info(live_release)").fetchall()
            ]
            if "session_calendar_id_str" not in live_release_column_name_list:
                connection_obj.execute(
                    "ALTER TABLE live_release ADD COLUMN session_calendar_id_str TEXT NOT NULL DEFAULT 'XNYS'"
                )
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
                if (
                    existing_expires_timestamp_ts > current_timestamp_ts
                    and row_obj["owner_token_str"] != owner_token_str
                ):
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

    def upsert_release_list(self, release_list: list[LiveRelease]) -> None:
        for release_obj in release_list:
            self.upsert_release(release_obj)

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

    def get_existing_table_name_list(self) -> list[str]:
        with self._connect() as connection_obj:
            row_list = connection_obj.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                  AND name NOT LIKE 'sqlite_%'
                ORDER BY name ASC
                """
            ).fetchall()
        return [str(row_obj["name"]) for row_obj in row_list]

    def get_table_row_count_int(self, table_name_str: str) -> int:
        self._validate_table_name_str(table_name_str)
        with self._connect() as connection_obj:
            row_obj = connection_obj.execute(
                f"SELECT COUNT(1) AS row_count_int FROM {table_name_str}"
            ).fetchone()
        return int(row_obj["row_count_int"])

    def get_table_row_dict_list(self, table_name_str: str) -> list[dict[str, object]]:
        self._validate_table_name_str(table_name_str)
        with self._connect() as connection_obj:
            row_list = connection_obj.execute(
                f"SELECT * FROM {table_name_str}"
            ).fetchall()
        return [dict(row_obj) for row_obj in row_list]

    def drop_table_if_exists(self, table_name_str: str) -> None:
        self._validate_table_name_str(table_name_str)
        with self._connect() as connection_obj:
            connection_obj.execute(f"DROP TABLE IF EXISTS {table_name_str}")

    def _validate_table_name_str(self, table_name_str: str) -> None:
        if _TABLE_NAME_PATTERN_OBJ.fullmatch(str(table_name_str)) is None:
            raise ValueError(f"Unsafe table_name_str '{table_name_str}'.")
