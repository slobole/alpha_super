from __future__ import annotations

import os
import json
import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from alpha.live import scheduler_utils, strategy_host
from alpha.live.execution_engine import (
    build_broker_order_request_list_from_vplan,
    build_vplan,
    get_touched_asset_list_for_decision_plan,
)
from alpha.live.logging_utils import (
    DEFAULT_POD_TRACE_LOG_ROOT_PATH_STR,
    build_pod_trace_context_dict,
    log_pod_trace_event,
)
from alpha.live.models import DecisionPlan, LiveRelease, PodState, VPlan
from alpha.live.norgate_snapshot_sync import ensure_norgate_snapshots_for_live_tick
from alpha.live.release_manifest import (
    account_route_placeholder_bool,
    load_release_list,
    select_enabled_release_list_for_mode,
    validate_enabled_deployment_for_mode,
)
from alpha.live.reconcile import reconcile_account_state
from data.norgate_snapshot_store import (
    ALPHA_USE_NORGATE_SNAPSHOT_ENV_STR,
    NORGATE_SNAPSHOT_ROOT_ENV_STR,
    is_snapshot_mode_enabled_bool,
)
from scripts.norgate_config_env import NORGATE_RELEASES_ROOT_ENV_STR


DOCTOR_PASS_STR = "PASS"
DOCTOR_WAIT_STR = "WAIT"
DOCTOR_BLOCK_STR = "BLOCK"
DOCTOR_SKIP_STR = "SKIP"
DOCTOR_FEATURE_NAME_STR = "doctor"

WAIT_GATE_REASON_CODE_SET: set[str] = {
    "snapshot_not_ready",
    "not_month_end_session",
    "snapshot_not_ready_for_session",
}


PodStateLoader_Fn = Callable[[LiveRelease, datetime], PodState | None]
NorgateSnapshotSyncChecker_Fn = Callable[..., dict[str, Any]]


def _component_result_dict(
    component_name_str: str,
    status_str: str,
    reason_code_str: str,
    detail_str: str,
    extra_detail_dict: dict[str, object] | None = None,
) -> dict[str, object]:
    result_dict: dict[str, object] = {
        "component_name_str": str(component_name_str),
        "status_str": str(status_str),
        "reason_code_str": str(reason_code_str),
        "detail_str": str(detail_str),
    }
    if extra_detail_dict:
        result_dict.update(extra_detail_dict)
    return result_dict


def _overall_verdict_str(component_result_dict_list: list[dict[str, object]]) -> str:
    status_list = [str(result_dict["status_str"]) for result_dict in component_result_dict_list]
    if DOCTOR_BLOCK_STR in status_list:
        return DOCTOR_BLOCK_STR
    if DOCTOR_WAIT_STR in status_list:
        return DOCTOR_WAIT_STR
    return DOCTOR_PASS_STR


def _timestamp_str(timestamp_obj: datetime | None) -> str | None:
    if timestamp_obj is None:
        return None
    return timestamp_obj.isoformat()


def _release_dict(release_obj: LiveRelease) -> dict[str, object]:
    release_dict = asdict(release_obj)
    release_dict["source_path_str"] = str(release_obj.source_path_str)
    return release_dict


def _manifest_qualification_dict(release_obj: LiveRelease) -> dict[str, object]:
    return {
        "qualification_status_str": DOCTOR_PASS_STR,
        "qualification_reason_code_str": "manifest_qualified",
        "qualified_bool": True,
        "source_path_str": str(release_obj.source_path_str),
        "release_id_str": release_obj.release_id_str,
        "user_id_str": release_obj.user_id_str,
        "pod_id_str": release_obj.pod_id_str,
        "enabled_bool": bool(release_obj.enabled_bool),
        "mode_str": release_obj.mode_str,
        "account_route_str": release_obj.account_route_str,
        "account_route_placeholder_bool": account_route_placeholder_bool(
            release_obj.account_route_str
        ),
        "broker_host_str": release_obj.broker_host_str,
        "broker_port_int": int(release_obj.broker_port_int),
        "broker_client_id_int": int(release_obj.broker_client_id_int),
        "broker_timeout_seconds_float": float(release_obj.broker_timeout_seconds_float),
        "strategy_import_str": release_obj.strategy_import_str,
        "data_profile_str": release_obj.data_profile_str,
        "signal_clock_str": release_obj.signal_clock_str,
        "execution_policy_str": release_obj.execution_policy_str,
        "session_calendar_id_str": release_obj.session_calendar_id_str,
        "risk_profile_str": release_obj.risk_profile_str,
        "pod_budget_fraction_float": float(release_obj.pod_budget_fraction_float),
        "auto_submit_enabled_bool": bool(release_obj.auto_submit_enabled_bool),
    }


def _decision_plan_dict(decision_plan_obj: DecisionPlan) -> dict[str, object]:
    return {
        "release_id_str": decision_plan_obj.release_id_str,
        "pod_id_str": decision_plan_obj.pod_id_str,
        "account_route_str": decision_plan_obj.account_route_str,
        "status_str": decision_plan_obj.status_str,
        "signal_timestamp_str": _timestamp_str(decision_plan_obj.signal_timestamp_ts),
        "submission_timestamp_str": _timestamp_str(decision_plan_obj.submission_timestamp_ts),
        "target_execution_timestamp_str": _timestamp_str(
            decision_plan_obj.target_execution_timestamp_ts
        ),
        "execution_policy_str": decision_plan_obj.execution_policy_str,
        "decision_book_type_str": decision_plan_obj.decision_book_type_str,
        "entry_target_weight_map_dict": dict(decision_plan_obj.entry_target_weight_map_dict),
        "full_target_weight_map_dict": dict(decision_plan_obj.full_target_weight_map_dict),
        "target_weight_map": dict(decision_plan_obj.target_weight_map),
        "exit_asset_list": sorted(decision_plan_obj.exit_asset_set),
        "entry_priority_list": list(decision_plan_obj.entry_priority_list),
        "cash_reserve_weight_float": float(decision_plan_obj.cash_reserve_weight_float),
        "preserve_untouched_positions_bool": bool(
            decision_plan_obj.preserve_untouched_positions_bool
        ),
        "rebalance_omitted_assets_to_zero_bool": bool(
            decision_plan_obj.rebalance_omitted_assets_to_zero_bool
        ),
        "snapshot_metadata_dict": dict(decision_plan_obj.snapshot_metadata_dict),
    }


def _reconciliation_dict(reconciliation_result_obj: Any) -> dict[str, object]:
    return {
        "passed_bool": bool(reconciliation_result_obj.passed_bool),
        "status_str": str(reconciliation_result_obj.status_str),
        "mismatch_dict": dict(reconciliation_result_obj.mismatch_dict),
        "model_position_map": dict(reconciliation_result_obj.model_position_map),
        "broker_position_map": dict(reconciliation_result_obj.broker_position_map),
        "model_cash_float": float(reconciliation_result_obj.model_cash_float),
        "broker_cash_float": float(reconciliation_result_obj.broker_cash_float),
    }


def _vplan_preview_dict(vplan_obj: VPlan) -> dict[str, object]:
    broker_order_request_list = build_broker_order_request_list_from_vplan(vplan_obj)
    return {
        "status_str": vplan_obj.status_str,
        "account_route_str": vplan_obj.account_route_str,
        "execution_policy_str": vplan_obj.execution_policy_str,
        "submission_key_str": str(vplan_obj.submission_key_str or ""),
        "broker_snapshot_timestamp_str": _timestamp_str(vplan_obj.broker_snapshot_timestamp_ts),
        "live_reference_snapshot_timestamp_str": _timestamp_str(
            vplan_obj.live_reference_snapshot_timestamp_ts
        ),
        "live_price_source_str": vplan_obj.live_price_source_str,
        "net_liq_float": float(vplan_obj.net_liq_float),
        "available_funds_float": vplan_obj.available_funds_float,
        "pod_budget_fraction_float": float(vplan_obj.pod_budget_fraction_float),
        "pod_budget_float": float(vplan_obj.pod_budget_float),
        "target_share_map": dict(vplan_obj.target_share_map),
        "order_delta_map": dict(vplan_obj.order_delta_map),
        "live_reference_price_map": dict(vplan_obj.live_reference_price_map),
        "live_reference_source_map_dict": dict(vplan_obj.live_reference_source_map_dict),
        "vplan_row_dict_list": [
            {
                "asset_str": row_obj.asset_str,
                "current_share_float": float(row_obj.current_share_float),
                "target_share_float": float(row_obj.target_share_float),
                "order_delta_share_float": float(row_obj.order_delta_share_float),
                "live_reference_price_float": float(row_obj.live_reference_price_float),
                "estimated_target_notional_float": float(row_obj.estimated_target_notional_float),
                "broker_order_type_str": row_obj.broker_order_type_str,
                "live_reference_source_str": row_obj.live_reference_source_str,
            }
            for row_obj in vplan_obj.vplan_row_list
        ],
        "broker_order_request_count_int": len(broker_order_request_list),
        "broker_order_request_dict_list": [
            {
                "asset_str": request_obj.asset_str,
                "broker_order_type_str": request_obj.broker_order_type_str,
                "unit_str": request_obj.unit_str,
                "amount_float": float(request_obj.amount_float),
                "target_bool": bool(request_obj.target_bool),
                "sizing_reference_price_float": float(
                    request_obj.sizing_reference_price_float
                ),
                "portfolio_value_float": float(request_obj.portfolio_value_float),
                "order_request_key_str": request_obj.order_request_key_str,
            }
            for request_obj in broker_order_request_list
        ],
    }


def _default_pod_state_obj(release_obj: LiveRelease, as_of_ts: datetime) -> PodState:
    capital_base_float = float(release_obj.params_dict.get("capital_base_float", 100_000.0))
    return PodState(
        pod_id_str=release_obj.pod_id_str,
        user_id_str=release_obj.user_id_str,
        account_route_str=release_obj.account_route_str,
        position_amount_map={},
        cash_float=capital_base_float,
        total_value_float=capital_base_float,
        strategy_state_dict={},
        updated_timestamp_ts=as_of_ts,
    )


def _load_pod_state_or_default_obj(
    state_store_obj: Any | None,
    release_obj: LiveRelease,
    as_of_ts: datetime,
) -> PodState:
    if state_store_obj is not None:
        pod_state_obj = state_store_obj.get_pod_state(release_obj.pod_id_str)
        if pod_state_obj is not None:
            return pod_state_obj
    return _default_pod_state_obj(release_obj, as_of_ts)


def load_pod_state_from_sqlite_read_only_obj(
    *,
    db_path_str: str,
    release_obj: LiveRelease,
) -> PodState | None:
    db_path_obj = Path(db_path_str)
    if not db_path_obj.exists():
        return None

    db_uri_str = f"{db_path_obj.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(db_uri_str, uri=True) as connection_obj:
        connection_obj.row_factory = sqlite3.Row
        table_row_obj = connection_obj.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name = 'pod_state'
            """
        ).fetchone()
        if table_row_obj is None:
            return None
        row_obj = connection_obj.execute(
            "SELECT * FROM pod_state WHERE pod_id_str = ?",
            (release_obj.pod_id_str,),
        ).fetchone()

    if row_obj is None:
        return None

    row_key_set = set(row_obj.keys())
    snapshot_stage_str = (
        str(row_obj["snapshot_stage_str"])
        if "snapshot_stage_str" in row_key_set
        else "unknown"
    )
    snapshot_source_str = (
        str(row_obj["snapshot_source_str"])
        if "snapshot_source_str" in row_key_set
        else "pod_state"
    )
    return PodState(
        pod_id_str=str(row_obj["pod_id_str"]),
        user_id_str=str(row_obj["user_id_str"]),
        account_route_str=str(row_obj["account_route_str"]),
        position_amount_map=json.loads(row_obj["position_json_str"]),
        cash_float=float(row_obj["cash_float"]),
        total_value_float=float(row_obj["total_value_float"]),
        strategy_state_dict=json.loads(row_obj["strategy_state_json_str"]),
        updated_timestamp_ts=datetime.fromisoformat(str(row_obj["updated_timestamp_str"])),
        snapshot_stage_str=snapshot_stage_str,
        snapshot_source_str=snapshot_source_str,
    )


def inspect_persisted_lifecycle_read_only_dict(
    *,
    db_path_str: str | None,
    pod_id_str: str,
    as_of_ts: datetime,
) -> dict[str, object]:
    if not db_path_str:
        return {
            "db_path_str": "",
            "db_exists_bool": False,
            "active_lifecycle_bool": False,
            "reason_code_str": "state_db_path_missing",
        }

    db_path_obj = Path(db_path_str)
    if not db_path_obj.exists():
        return {
            "db_path_str": str(db_path_obj),
            "db_exists_bool": False,
            "active_lifecycle_bool": False,
            "reason_code_str": "state_db_not_found",
        }

    db_uri_str = f"{db_path_obj.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(db_uri_str, uri=True) as connection_obj:
        connection_obj.row_factory = sqlite3.Row
        table_name_set = {
            str(row_obj["name"])
            for row_obj in connection_obj.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                  AND name IN ('decision_plan', 'vplan')
                """
            ).fetchall()
        }
        latest_decision_plan_row_obj = None
        latest_vplan_row_obj = None
        if "decision_plan" in table_name_set:
            latest_decision_plan_row_obj = connection_obj.execute(
                """
                SELECT
                    decision_plan_id_int,
                    status_str,
                    signal_timestamp_str,
                    submission_timestamp_str,
                    target_execution_timestamp_str,
                    execution_policy_str
                FROM decision_plan
                WHERE pod_id_str = ?
                ORDER BY decision_plan_id_int DESC
                LIMIT 1
                """,
                (pod_id_str,),
            ).fetchone()
        if "vplan" in table_name_set:
            latest_vplan_row_obj = connection_obj.execute(
                """
                SELECT
                    vplan_id_int,
                    decision_plan_id_int,
                    status_str,
                    signal_timestamp_str,
                    submission_timestamp_str,
                    target_execution_timestamp_str,
                    execution_policy_str
                FROM vplan
                WHERE pod_id_str = ?
                ORDER BY vplan_id_int DESC
                LIMIT 1
                """,
                (pod_id_str,),
            ).fetchone()

    latest_decision_plan_dict: dict[str, object] = (
        {} if latest_decision_plan_row_obj is None else dict(latest_decision_plan_row_obj)
    )
    latest_vplan_dict: dict[str, object] = (
        {} if latest_vplan_row_obj is None else dict(latest_vplan_row_obj)
    )
    active_decision_status_set = {"planned", "vplan_ready", "submitted"}
    active_vplan_status_set = {"ready", "submitting", "submitted"}
    active_decision_bool = (
        str(latest_decision_plan_dict.get("status_str") or "") in active_decision_status_set
    )
    active_vplan_bool = (
        str(latest_vplan_dict.get("status_str") or "") in active_vplan_status_set
    )
    active_lifecycle_bool = active_decision_bool or active_vplan_bool

    expired_lifecycle_bool = False
    lifecycle_row_dict = latest_vplan_dict if active_vplan_bool else latest_decision_plan_dict
    if active_lifecycle_bool:
        try:
            target_execution_timestamp_ts = datetime.fromisoformat(
                str(lifecycle_row_dict["target_execution_timestamp_str"])
            )
            expired_lifecycle_bool = scheduler_utils.is_execution_window_expired_bool(
                str(lifecycle_row_dict["execution_policy_str"]),
                target_execution_timestamp_ts,
                as_of_ts,
            )
        except Exception:
            expired_lifecycle_bool = True

    if expired_lifecycle_bool:
        reason_code_str = "persisted_lifecycle_expired"
    elif active_lifecycle_bool:
        reason_code_str = "persisted_lifecycle_active"
    else:
        reason_code_str = "no_active_persisted_lifecycle"

    return {
        "db_path_str": str(db_path_obj),
        "db_exists_bool": True,
        "active_lifecycle_bool": bool(active_lifecycle_bool),
        "expired_lifecycle_bool": bool(expired_lifecycle_bool),
        "latest_decision_plan_dict": latest_decision_plan_dict,
        "latest_vplan_dict": latest_vplan_dict,
        "reason_code_str": reason_code_str,
    }


def _environment_result_dict(loaded_config_env_dict: dict[str, str] | None) -> dict[str, object]:
    snapshot_mode_value_str = os.getenv(ALPHA_USE_NORGATE_SNAPSHOT_ENV_STR, "").strip()
    snapshot_root_str = os.getenv(NORGATE_SNAPSHOT_ROOT_ENV_STR, "").strip()
    config_env_key_list = sorted((loaded_config_env_dict or {}).keys())

    if is_snapshot_mode_enabled_bool() and not snapshot_root_str:
        return _component_result_dict(
            component_name_str="environment",
            status_str=DOCTOR_BLOCK_STR,
            reason_code_str="snapshot_root_missing",
            detail_str=(
                f"{NORGATE_SNAPSHOT_ROOT_ENV_STR} must be set when "
                f"{ALPHA_USE_NORGATE_SNAPSHOT_ENV_STR}=true."
            ),
            extra_detail_dict={
                "snapshot_mode_env_str": snapshot_mode_value_str,
                "snapshot_root_path_str": snapshot_root_str,
                "loaded_config_env_key_list": config_env_key_list,
            },
        )

    return _component_result_dict(
        component_name_str="environment",
        status_str=DOCTOR_PASS_STR,
        reason_code_str="environment_loaded",
        detail_str="Runtime environment is coherent for doctor.",
        extra_detail_dict={
            "snapshot_mode_env_str": snapshot_mode_value_str,
            "snapshot_root_path_str": snapshot_root_str,
            "loaded_config_env_key_list": config_env_key_list,
        },
    )


def _resolve_relative_to_config_path_obj(
    raw_path_str: str,
    config_env_path_str: str | None,
) -> Path:
    raw_path_obj = Path(raw_path_str).expanduser()
    if raw_path_obj.is_absolute():
        return raw_path_obj.resolve()
    if config_env_path_str:
        return (Path(config_env_path_str).expanduser().resolve().parent / raw_path_obj).resolve()
    return raw_path_obj.resolve()


def _resolve_cli_path_obj(raw_path_str: str) -> Path:
    return Path(raw_path_str).expanduser().resolve()


def _path_contains_release_source_bool(
    *,
    release_root_path_obj: Path,
    source_path_str: str,
) -> bool:
    try:
        Path(source_path_str).expanduser().resolve().relative_to(release_root_path_obj)
        return True
    except ValueError:
        return False


def _config_release_root_result_dict(
    *,
    releases_root_path_str: str | None,
    releases_root_explicit_bool: bool,
    config_env_path_str: str | None,
    loaded_config_env_dict: dict[str, str] | None,
) -> tuple[str | None, dict[str, object], dict[str, object]]:
    config_env_path_obj = (
        None if config_env_path_str is None else Path(config_env_path_str).expanduser().resolve()
    )
    config_env_exists_bool = config_env_path_obj is not None and config_env_path_obj.exists()
    env_release_root_raw_str = str(
        (loaded_config_env_dict or {}).get(NORGATE_RELEASES_ROOT_ENV_STR) or ""
    ).strip()
    env_release_root_resolved_str = ""
    if env_release_root_raw_str:
        env_release_root_resolved_str = str(
            _resolve_relative_to_config_path_obj(
                env_release_root_raw_str,
                str(config_env_path_obj) if config_env_path_obj is not None else None,
            )
        )

    effective_root_path_obj: Path | None = None
    release_root_source_str = ""
    if releases_root_explicit_bool and releases_root_path_str:
        effective_root_path_obj = _resolve_cli_path_obj(releases_root_path_str)
        release_root_source_str = "explicit_arg"
    elif env_release_root_raw_str:
        effective_root_path_obj = Path(env_release_root_resolved_str)
        release_root_source_str = "config_env"

    effective_release_root_resolved_str = (
        "" if effective_root_path_obj is None else str(effective_root_path_obj)
    )
    release_root_match_bool = (
        bool(env_release_root_resolved_str)
        and bool(effective_release_root_resolved_str)
        and env_release_root_resolved_str == effective_release_root_resolved_str
    )
    root_detail_dict: dict[str, object] = {
        "config_env_path_str": "" if config_env_path_obj is None else str(config_env_path_obj),
        "config_env_exists_bool": bool(config_env_exists_bool),
        "env_release_root_raw_str": env_release_root_raw_str,
        "env_release_root_resolved_str": env_release_root_resolved_str,
        "effective_release_root_resolved_str": effective_release_root_resolved_str,
        "release_root_source_str": release_root_source_str,
        "release_root_match_bool": bool(release_root_match_bool),
        "selected_release_source_path_str": "",
    }

    if not config_env_exists_bool:
        return None, root_detail_dict, _component_result_dict(
            component_name_str="config_release_root",
            status_str=DOCTOR_BLOCK_STR,
            reason_code_str="config_env_missing",
            detail_str="config.env must exist for PAPER/LIVE doctor.",
            extra_detail_dict={"config_release_root_dict": dict(root_detail_dict)},
        )
    if not env_release_root_raw_str:
        return None, root_detail_dict, _component_result_dict(
            component_name_str="config_release_root",
            status_str=DOCTOR_BLOCK_STR,
            reason_code_str="norgate_releases_root_missing",
            detail_str=f"{NORGATE_RELEASES_ROOT_ENV_STR} must be set in config.env.",
            extra_detail_dict={"config_release_root_dict": dict(root_detail_dict)},
        )
    if effective_root_path_obj is None:
        return None, root_detail_dict, _component_result_dict(
            component_name_str="config_release_root",
            status_str=DOCTOR_BLOCK_STR,
            reason_code_str="effective_release_root_missing",
            detail_str="No effective release root could be resolved for doctor.",
            extra_detail_dict={"config_release_root_dict": dict(root_detail_dict)},
        )
    if releases_root_explicit_bool and not release_root_match_bool:
        return None, root_detail_dict, _component_result_dict(
            component_name_str="config_release_root",
            status_str=DOCTOR_BLOCK_STR,
            reason_code_str="release_root_mismatch",
            detail_str=(
                f"Explicit --releases-root does not match {NORGATE_RELEASES_ROOT_ENV_STR}."
            ),
            extra_detail_dict={"config_release_root_dict": dict(root_detail_dict)},
        )
    if not effective_root_path_obj.exists():
        return None, root_detail_dict, _component_result_dict(
            component_name_str="config_release_root",
            status_str=DOCTOR_BLOCK_STR,
            reason_code_str="release_root_path_missing",
            detail_str="Effective release root path does not exist.",
            extra_detail_dict={"config_release_root_dict": dict(root_detail_dict)},
        )

    return str(effective_root_path_obj), root_detail_dict, _component_result_dict(
        component_name_str="config_release_root",
        status_str=DOCTOR_PASS_STR,
        reason_code_str="release_root_ready",
        detail_str="config.env release root matches the effective doctor release root.",
        extra_detail_dict={"config_release_root_dict": dict(root_detail_dict)},
    )


def _select_single_enabled_release_obj(
    releases_root_path_str: str,
    env_mode_str: str,
    pod_id_str: str | None,
) -> tuple[LiveRelease | None, dict[str, object]]:
    release_list = load_release_list(releases_root_path_str)
    validate_enabled_deployment_for_mode(release_list, env_mode_str)

    if pod_id_str is not None:
        release_list = [
            release_obj
            for release_obj in release_list
            if release_obj.pod_id_str == pod_id_str
        ]
        if len(release_list) == 0:
            return None, _component_result_dict(
                component_name_str="manifest",
                status_str=DOCTOR_BLOCK_STR,
                reason_code_str="pod_not_found",
                detail_str=f"No release found for pod_id_str '{pod_id_str}'.",
            )

    enabled_release_list = select_enabled_release_list_for_mode(release_list, env_mode_str)
    if len(enabled_release_list) == 0:
        return None, _component_result_dict(
            component_name_str="manifest",
            status_str=DOCTOR_BLOCK_STR,
            reason_code_str="enabled_release_not_found",
            detail_str=(
                f"No enabled release found for mode '{env_mode_str}'"
                + ("" if pod_id_str is None else f" and pod_id_str '{pod_id_str}'")
                + "."
            ),
        )
    if len(enabled_release_list) > 1:
        return None, _component_result_dict(
            component_name_str="manifest",
            status_str=DOCTOR_BLOCK_STR,
            reason_code_str="multiple_enabled_pods",
            detail_str=(
                "Doctor is pod-scoped. Pass --pod-id when more than one enabled "
                f"release exists in mode '{env_mode_str}'."
            ),
            extra_detail_dict={
                "enabled_pod_id_list": sorted(
                    release_obj.pod_id_str for release_obj in enabled_release_list
                ),
            },
        )

    release_obj = enabled_release_list[0]
    return release_obj, _component_result_dict(
        component_name_str="manifest",
        status_str=DOCTOR_PASS_STR,
        reason_code_str="release_selected",
        detail_str=f"Selected enabled release '{release_obj.release_id_str}'.",
    )


def _persisted_lifecycle_result_dict(lifecycle_dict: dict[str, object]) -> dict[str, object]:
    reason_code_str = str(lifecycle_dict.get("reason_code_str") or "")
    if reason_code_str == "persisted_lifecycle_expired":
        return _component_result_dict(
            component_name_str="persisted_lifecycle",
            status_str=DOCTOR_BLOCK_STR,
            reason_code_str=reason_code_str,
            detail_str="A persisted DecisionPlan/VPlan is active but already stale.",
            extra_detail_dict={"persisted_lifecycle_dict": dict(lifecycle_dict)},
        )
    if reason_code_str == "persisted_lifecycle_active":
        return _component_result_dict(
            component_name_str="persisted_lifecycle",
            status_str=DOCTOR_WAIT_STR,
            reason_code_str=reason_code_str,
            detail_str=(
                "A persisted DecisionPlan/VPlan is already active; doctor will not "
                "preview a fresh replacement plan."
            ),
            extra_detail_dict={"persisted_lifecycle_dict": dict(lifecycle_dict)},
        )
    if reason_code_str in {"state_db_path_missing", "state_db_not_found"}:
        return _component_result_dict(
            component_name_str="persisted_lifecycle",
            status_str=DOCTOR_SKIP_STR,
            reason_code_str=reason_code_str,
            detail_str="No existing state DB was available for persisted lifecycle inspection.",
            extra_detail_dict={"persisted_lifecycle_dict": dict(lifecycle_dict)},
        )
    return _component_result_dict(
        component_name_str="persisted_lifecycle",
        status_str=DOCTOR_PASS_STR,
        reason_code_str=reason_code_str,
        detail_str="No active persisted DecisionPlan/VPlan blocks fresh doctor.",
        extra_detail_dict={"persisted_lifecycle_dict": dict(lifecycle_dict)},
    )


def _gate_component_result_dict(gate_dict: dict[str, object]) -> dict[str, object]:
    reason_code_str = str(gate_dict.get("reason_code_str") or "")
    due_bool = bool(gate_dict.get("due_bool"))
    if due_bool:
        return _component_result_dict(
            component_name_str="scheduler_gate",
            status_str=DOCTOR_PASS_STR,
            reason_code_str=reason_code_str,
            detail_str="Scheduler/data gate is ready to build a DecisionPlan.",
            extra_detail_dict={"scheduler_gate_dict": dict(gate_dict)},
        )
    if reason_code_str in WAIT_GATE_REASON_CODE_SET:
        return _component_result_dict(
            component_name_str="scheduler_gate",
            status_str=DOCTOR_WAIT_STR,
            reason_code_str=reason_code_str,
            detail_str="Scheduler/data gate is coherent, but not ready yet.",
            extra_detail_dict={"scheduler_gate_dict": dict(gate_dict)},
        )
    return _component_result_dict(
        component_name_str="scheduler_gate",
        status_str=DOCTOR_BLOCK_STR,
        reason_code_str=reason_code_str or "scheduler_gate_blocked",
        detail_str="Scheduler/data gate is not safe to proceed.",
        extra_detail_dict={"scheduler_gate_dict": dict(gate_dict)},
    )


def _release_requires_norgate_snapshot_sync_bool(release_obj: LiveRelease) -> bool:
    normalized_signal_clock_str = scheduler_utils.normalize_signal_clock_str(
        release_obj.signal_clock_str
    )
    return normalized_signal_clock_str in {"eod_snapshot_ready", "month_end_snapshot_ready"}


def _norgate_snapshot_sync_result_dict(sync_detail_dict: dict[str, object]) -> dict[str, object]:
    status_str = str(sync_detail_dict.get("status_str") or "")
    reason_code_str = str(sync_detail_dict.get("reason_code_str") or status_str)
    if status_str in {"direct", "ready"}:
        return _component_result_dict(
            component_name_str="norgate_snapshot_sync",
            status_str=DOCTOR_PASS_STR,
            reason_code_str=reason_code_str,
            detail_str="Norgate snapshot sync state is compatible with DecisionPlan build.",
            extra_detail_dict={"norgate_snapshot_sync_detail_dict": dict(sync_detail_dict)},
        )
    if status_str == "waiting":
        return _component_result_dict(
            component_name_str="norgate_snapshot_sync",
            status_str=DOCTOR_WAIT_STR,
            reason_code_str=reason_code_str,
            detail_str="Norgate snapshot sync is still waiting.",
            extra_detail_dict={"norgate_snapshot_sync_detail_dict": dict(sync_detail_dict)},
        )
    return _component_result_dict(
        component_name_str="norgate_snapshot_sync",
        status_str=DOCTOR_BLOCK_STR,
        reason_code_str=reason_code_str or "norgate_snapshot_sync_blocked",
        detail_str="Norgate snapshot sync blocks live DecisionPlan build.",
        extra_detail_dict={"norgate_snapshot_sync_detail_dict": dict(sync_detail_dict)},
    )


def _build_broker_adapter_resolver_obj(
    broker_host_str: str | None,
    broker_port_int: int | None,
    broker_client_id_int: int | None,
    broker_timeout_seconds_float: float | None,
    adapter_factory_func: Callable[[str, int, int, float], Any] | None,
) -> Any:
    from alpha.live.runner import BrokerAdapterResolver

    return BrokerAdapterResolver(
        broker_host_str=broker_host_str,
        broker_port_int=broker_port_int,
        broker_client_id_int=broker_client_id_int,
        broker_timeout_seconds_float=broker_timeout_seconds_float,
        adapter_factory_func=adapter_factory_func,
    )


def _decision_plan_norgate_provenance_block_reason_str(
    release_obj: LiveRelease,
    decision_plan_obj: DecisionPlan,
) -> str | None:
    from alpha.live.runner import _decision_plan_norgate_provenance_block_reason_str

    return _decision_plan_norgate_provenance_block_reason_str(
        release_obj,
        decision_plan_obj,
    )


def _resolve_db_path_for_release_str(
    *,
    db_path_str: str | None,
    env_mode_str: str,
    pod_id_str: str,
) -> str:
    from alpha.live.runner import _resolve_db_path_for_mode_str

    return _resolve_db_path_for_mode_str(
        db_path_str=db_path_str,
        env_mode_str=env_mode_str,
        pod_id_str=pod_id_str,
    )


def _compute_doctor_verdict_impl(
    *,
    releases_root_path_str: str | None,
    env_mode_str: str,
    as_of_ts: datetime,
    command_name_str: str = DOCTOR_FEATURE_NAME_STR,
    pod_id_str: str | None = None,
    releases_root_explicit_bool: bool = True,
    config_env_path_str: str | None = None,
    db_path_str: str | None = None,
    state_store_obj: Any | None = None,
    pod_state_loader_fn: PodStateLoader_Fn | None = None,
    norgate_snapshot_sync_checker_fn: NorgateSnapshotSyncChecker_Fn | None = None,
    broker_adapter_resolver_obj: Any | None = None,
    broker_host_str: str | None = None,
    broker_port_int: int | None = None,
    broker_client_id_int: int | None = None,
    broker_timeout_seconds_float: float | None = None,
    adapter_factory_func: Callable[[str, int, int, float], Any] | None = None,
    loaded_config_env_dict: dict[str, str] | None = None,
) -> dict[str, object]:
    component_result_dict_list: list[dict[str, object]] = []
    detail_dict: dict[str, object] = {
        "feature_name_str": DOCTOR_FEATURE_NAME_STR,
        "command_name_str": str(command_name_str),
        "mode_str": str(env_mode_str),
        "pod_id_str": pod_id_str,
        "as_of_timestamp_str": as_of_ts.isoformat(),
        "component_result_dict_list": component_result_dict_list,
        "config_release_root_dict": {},
        "manifest_qualification_dict": {},
        "release_dict": {},
        "scheduler_gate_dict": {},
        "persisted_lifecycle_dict": {},
        "norgate_snapshot_sync_detail_dict": {},
        "decision_plan_dict": {},
        "broker_dict": {},
        "reconciliation_dict": {},
        "vplan_preview_dict": {},
    }

    environment_result_dict = _environment_result_dict(loaded_config_env_dict)
    component_result_dict_list.append(environment_result_dict)
    if environment_result_dict["status_str"] == DOCTOR_BLOCK_STR:
        detail_dict["overall_verdict_str"] = _overall_verdict_str(component_result_dict_list)
        return detail_dict

    if env_mode_str == "incubation":
        component_result_dict_list.append(
            _component_result_dict(
                component_name_str="mode",
                status_str=DOCTOR_BLOCK_STR,
                reason_code_str="incubation_doctor_unsupported",
                detail_str="Doctor v1 supports PAPER/LIVE only; incubation uses rehearsal status.",
            )
        )
        detail_dict["overall_verdict_str"] = _overall_verdict_str(component_result_dict_list)
        return detail_dict

    effective_releases_root_path_str = releases_root_path_str
    if env_mode_str in {"paper", "live"}:
        (
            effective_releases_root_path_str,
            config_release_root_dict,
            config_release_root_result_dict,
        ) = _config_release_root_result_dict(
            releases_root_path_str=releases_root_path_str,
            releases_root_explicit_bool=releases_root_explicit_bool,
            config_env_path_str=config_env_path_str,
            loaded_config_env_dict=loaded_config_env_dict,
        )
        detail_dict["config_release_root_dict"] = dict(config_release_root_dict)
        component_result_dict_list.append(config_release_root_result_dict)
        if config_release_root_result_dict["status_str"] == DOCTOR_BLOCK_STR:
            detail_dict["overall_verdict_str"] = _overall_verdict_str(component_result_dict_list)
            return detail_dict

    try:
        release_obj, manifest_result_dict = _select_single_enabled_release_obj(
            releases_root_path_str=str(effective_releases_root_path_str),
            env_mode_str=env_mode_str,
            pod_id_str=pod_id_str,
        )
    except Exception as exc:
        detail_dict["manifest_qualification_dict"] = {
            "qualification_status_str": DOCTOR_BLOCK_STR,
            "qualification_reason_code_str": "manifest_error",
            "qualified_bool": False,
            "error_str": str(exc),
        }
        component_result_dict_list.append(
            _component_result_dict(
                component_name_str="manifest_qualification",
                status_str=DOCTOR_BLOCK_STR,
                reason_code_str="manifest_error",
                detail_str=str(exc),
            )
        )
        detail_dict["overall_verdict_str"] = _overall_verdict_str(component_result_dict_list)
        return detail_dict

    component_result_dict_list.append(manifest_result_dict)
    if release_obj is None:
        detail_dict["overall_verdict_str"] = _overall_verdict_str(component_result_dict_list)
        return detail_dict

    detail_dict["pod_id_str"] = release_obj.pod_id_str
    detail_dict["release_dict"] = _release_dict(release_obj)
    manifest_qualification_dict = _manifest_qualification_dict(release_obj)
    detail_dict["manifest_qualification_dict"] = manifest_qualification_dict
    component_result_dict_list.append(
        _component_result_dict(
            component_name_str="manifest_qualification",
            status_str=DOCTOR_PASS_STR,
            reason_code_str="manifest_qualified",
            detail_str=(
                "Manifest is enabled for the requested mode and passed release "
                "qualification checks."
            ),
            extra_detail_dict={
                "manifest_qualification_dict": dict(manifest_qualification_dict)
            },
        )
    )
    config_release_root_dict = dict(detail_dict.get("config_release_root_dict") or {})
    if config_release_root_dict:
        config_release_root_dict["selected_release_source_path_str"] = str(
            release_obj.source_path_str
        )
        detail_dict["config_release_root_dict"] = config_release_root_dict
        release_root_path_str = str(config_release_root_dict.get("effective_release_root_resolved_str") or "")
        if release_root_path_str and not _path_contains_release_source_bool(
            release_root_path_obj=Path(release_root_path_str),
            source_path_str=str(release_obj.source_path_str),
        ):
            component_result_dict_list.append(
                _component_result_dict(
                    component_name_str="config_release_root",
                    status_str=DOCTOR_BLOCK_STR,
                    reason_code_str="selected_release_outside_root",
                    detail_str="Selected release YAML is not inside the effective release root.",
                    extra_detail_dict={
                        "config_release_root_dict": dict(config_release_root_dict)
                    },
                )
            )
            detail_dict["overall_verdict_str"] = _overall_verdict_str(component_result_dict_list)
            return detail_dict

    try:
        effective_db_path_str = _resolve_db_path_for_release_str(
            db_path_str=db_path_str,
            env_mode_str=env_mode_str,
            pod_id_str=release_obj.pod_id_str,
        )
    except Exception as exc:
        component_result_dict_list.append(
            _component_result_dict(
                component_name_str="state_db",
                status_str=DOCTOR_BLOCK_STR,
                reason_code_str="state_db_path_error",
                detail_str=str(exc),
            )
        )
        detail_dict["overall_verdict_str"] = _overall_verdict_str(component_result_dict_list)
        return detail_dict

    lifecycle_dict = inspect_persisted_lifecycle_read_only_dict(
        db_path_str=effective_db_path_str,
        pod_id_str=release_obj.pod_id_str,
        as_of_ts=as_of_ts,
    )
    detail_dict["persisted_lifecycle_dict"] = dict(lifecycle_dict)
    lifecycle_result_dict = _persisted_lifecycle_result_dict(lifecycle_dict)
    component_result_dict_list.append(lifecycle_result_dict)
    if lifecycle_result_dict["status_str"] in {DOCTOR_BLOCK_STR, DOCTOR_WAIT_STR}:
        detail_dict["overall_verdict_str"] = _overall_verdict_str(component_result_dict_list)
        return detail_dict

    gate_dict = scheduler_utils.evaluate_build_gate_dict(release_obj, as_of_ts)
    detail_dict["scheduler_gate_dict"] = dict(gate_dict)
    gate_result_dict = _gate_component_result_dict(gate_dict)
    component_result_dict_list.append(gate_result_dict)
    if gate_result_dict["status_str"] != DOCTOR_PASS_STR:
        detail_dict["overall_verdict_str"] = _overall_verdict_str(component_result_dict_list)
        return detail_dict

    if _release_requires_norgate_snapshot_sync_bool(release_obj):
        if norgate_snapshot_sync_checker_fn is None:
            norgate_snapshot_sync_checker_fn = ensure_norgate_snapshots_for_live_tick
        sync_detail_dict = norgate_snapshot_sync_checker_fn(
            releases_root_path_str=str(effective_releases_root_path_str),
            env_mode_str=env_mode_str,
            as_of_ts=as_of_ts,
            pod_id_str=release_obj.pod_id_str,
            print_operator_bool=False,
        )
        detail_dict["norgate_snapshot_sync_detail_dict"] = dict(sync_detail_dict)
        sync_result_dict = _norgate_snapshot_sync_result_dict(sync_detail_dict)
        component_result_dict_list.append(sync_result_dict)
        if sync_result_dict["status_str"] != DOCTOR_PASS_STR:
            detail_dict["overall_verdict_str"] = _overall_verdict_str(component_result_dict_list)
            return detail_dict
    else:
        component_result_dict_list.append(
            _component_result_dict(
                component_name_str="norgate_snapshot_sync",
                status_str=DOCTOR_SKIP_STR,
                reason_code_str="snapshot_sync_not_required",
                detail_str="Release signal clock does not use the Norgate snapshot gate.",
            )
        )

    try:
        if pod_state_loader_fn is not None:
            pod_state_obj = pod_state_loader_fn(release_obj, as_of_ts)
            if pod_state_obj is None:
                component_result_dict_list.append(
                    _component_result_dict(
                        component_name_str="pod_state",
                        status_str=DOCTOR_SKIP_STR,
                        reason_code_str="pod_state_not_found",
                        detail_str="No stored pod state was found; using release bootstrap defaults.",
                    )
                )
                pod_state_obj = _default_pod_state_obj(release_obj, as_of_ts)
            else:
                component_result_dict_list.append(
                    _component_result_dict(
                        component_name_str="pod_state",
                        status_str=DOCTOR_PASS_STR,
                        reason_code_str="pod_state_loaded",
                        detail_str="Stored pod state loaded read-only.",
                    )
                )
        else:
            pod_state_obj = _load_pod_state_or_default_obj(state_store_obj, release_obj, as_of_ts)
    except Exception as exc:
        component_result_dict_list.append(
            _component_result_dict(
                component_name_str="pod_state",
                status_str=DOCTOR_BLOCK_STR,
                reason_code_str="pod_state_read_error",
                detail_str=str(exc),
            )
        )
        detail_dict["overall_verdict_str"] = _overall_verdict_str(component_result_dict_list)
        return detail_dict

    try:
        decision_plan_obj = strategy_host.build_decision_plan_for_release(
            release_obj=release_obj,
            as_of_ts=as_of_ts,
            pod_state_obj=pod_state_obj,
        )
    except Exception as exc:
        component_result_dict_list.append(
            _component_result_dict(
                component_name_str="decision_plan",
                status_str=DOCTOR_BLOCK_STR,
                reason_code_str="decision_plan_build_error",
                detail_str=str(exc),
            )
        )
        detail_dict["overall_verdict_str"] = _overall_verdict_str(component_result_dict_list)
        return detail_dict

    detail_dict["decision_plan_dict"] = _decision_plan_dict(decision_plan_obj)
    if scheduler_utils.is_execution_window_expired_bool(
        decision_plan_obj.execution_policy_str,
        decision_plan_obj.target_execution_timestamp_ts,
        as_of_ts,
    ):
        component_result_dict_list.append(
            _component_result_dict(
                component_name_str="decision_plan",
                status_str=DOCTOR_BLOCK_STR,
                reason_code_str="submission_window_expired",
                detail_str="DecisionPlan target execution timestamp is already stale.",
            )
        )
        detail_dict["overall_verdict_str"] = _overall_verdict_str(component_result_dict_list)
        return detail_dict

    provenance_block_reason_str = _decision_plan_norgate_provenance_block_reason_str(
        release_obj,
        decision_plan_obj,
    )
    if provenance_block_reason_str is not None:
        component_result_dict_list.append(
            _component_result_dict(
                component_name_str="decision_plan",
                status_str=DOCTOR_BLOCK_STR,
                reason_code_str=provenance_block_reason_str,
                detail_str="DecisionPlan Norgate provenance is not compatible with the release.",
            )
        )
        detail_dict["overall_verdict_str"] = _overall_verdict_str(component_result_dict_list)
        return detail_dict

    component_result_dict_list.append(
        _component_result_dict(
            component_name_str="decision_plan",
            status_str=DOCTOR_PASS_STR,
            reason_code_str="decision_plan_built",
            detail_str="DecisionPlan built in memory.",
        )
    )

    if broker_adapter_resolver_obj is None:
        broker_adapter_resolver_obj = _build_broker_adapter_resolver_obj(
            broker_host_str=broker_host_str,
            broker_port_int=broker_port_int,
            broker_client_id_int=broker_client_id_int,
            broker_timeout_seconds_float=broker_timeout_seconds_float,
            adapter_factory_func=adapter_factory_func,
        )

    connection_field_map_dict = broker_adapter_resolver_obj.get_connection_field_map_dict(
        release_obj
    )
    broker_dict: dict[str, object] = {
        **connection_field_map_dict,
        "account_route_str": release_obj.account_route_str,
    }
    detail_dict["broker_dict"] = broker_dict

    try:
        broker_adapter_obj = broker_adapter_resolver_obj.get_adapter(release_obj)
        visible_account_route_set = broker_adapter_obj.get_visible_account_route_set()
    except Exception as exc:
        broker_dict["error_str"] = str(exc)
        component_result_dict_list.append(
            _component_result_dict(
                component_name_str="broker",
                status_str=DOCTOR_BLOCK_STR,
                reason_code_str="broker_connection_error",
                detail_str=str(exc),
            )
        )
        detail_dict["overall_verdict_str"] = _overall_verdict_str(component_result_dict_list)
        return detail_dict

    visible_account_route_list = (
        None if visible_account_route_set is None else sorted(visible_account_route_set)
    )
    broker_dict["visible_account_route_list"] = visible_account_route_list
    broker_dict["expected_account_visible_bool"] = (
        visible_account_route_set is not None
        and release_obj.account_route_str in visible_account_route_set
    )
    if visible_account_route_set is None:
        component_result_dict_list.append(
            _component_result_dict(
                component_name_str="broker",
                status_str=DOCTOR_BLOCK_STR,
                reason_code_str="broker_accounts_unavailable",
                detail_str="Broker managedAccounts() was not available.",
            )
        )
        detail_dict["overall_verdict_str"] = _overall_verdict_str(component_result_dict_list)
        return detail_dict
    if release_obj.account_route_str not in visible_account_route_set:
        component_result_dict_list.append(
            _component_result_dict(
                component_name_str="broker",
                status_str=DOCTOR_BLOCK_STR,
                reason_code_str="account_not_visible",
                detail_str=f"Expected account '{release_obj.account_route_str}' is not visible.",
            )
        )
        detail_dict["overall_verdict_str"] = _overall_verdict_str(component_result_dict_list)
        return detail_dict
    if not broker_adapter_obj.is_session_ready(release_obj.account_route_str):
        component_result_dict_list.append(
            _component_result_dict(
                component_name_str="broker",
                status_str=DOCTOR_BLOCK_STR,
                reason_code_str="broker_not_ready",
                detail_str="Broker session is not ready for the expected account.",
            )
        )
        detail_dict["overall_verdict_str"] = _overall_verdict_str(component_result_dict_list)
        return detail_dict
    session_mode_str = broker_adapter_obj.get_session_mode_str(release_obj.account_route_str)
    broker_dict["session_mode_str"] = session_mode_str
    if session_mode_str is not None and session_mode_str != release_obj.mode_str:
        component_result_dict_list.append(
            _component_result_dict(
                component_name_str="broker",
                status_str=DOCTOR_BLOCK_STR,
                reason_code_str="session_mode_mismatch",
                detail_str=(
                    f"Broker session mode '{session_mode_str}' does not match release mode "
                    f"'{release_obj.mode_str}'."
                ),
            )
        )
        detail_dict["overall_verdict_str"] = _overall_verdict_str(component_result_dict_list)
        return detail_dict

    try:
        broker_snapshot_obj = broker_adapter_obj.get_account_snapshot(release_obj.account_route_str)
    except Exception as exc:
        broker_dict["error_str"] = str(exc)
        component_result_dict_list.append(
            _component_result_dict(
                component_name_str="broker",
                status_str=DOCTOR_BLOCK_STR,
                reason_code_str="broker_snapshot_error",
                detail_str=str(exc),
            )
        )
        detail_dict["overall_verdict_str"] = _overall_verdict_str(component_result_dict_list)
        return detail_dict

    broker_dict.update(
        {
            "snapshot_timestamp_str": _timestamp_str(broker_snapshot_obj.snapshot_timestamp_ts),
            "cash_float": float(broker_snapshot_obj.cash_float),
            "total_value_float": float(broker_snapshot_obj.total_value_float),
            "net_liq_float": float(broker_snapshot_obj.net_liq_float),
            "available_funds_float": broker_snapshot_obj.available_funds_float,
            "excess_liquidity_float": broker_snapshot_obj.excess_liquidity_float,
            "position_amount_map": dict(broker_snapshot_obj.position_amount_map),
            "open_order_id_list": list(broker_snapshot_obj.open_order_id_list),
        }
    )
    if broker_snapshot_obj.net_liq_float <= 0.0:
        component_result_dict_list.append(
            _component_result_dict(
                component_name_str="broker",
                status_str=DOCTOR_BLOCK_STR,
                reason_code_str="non_positive_net_liq",
                detail_str="Broker NetLiquidation must be positive before order sizing.",
            )
        )
        detail_dict["overall_verdict_str"] = _overall_verdict_str(component_result_dict_list)
        return detail_dict

    open_order_id_list = list(broker_snapshot_obj.open_order_id_list)
    if len(open_order_id_list) > 0:
        component_result_dict_list.append(
            _component_result_dict(
                component_name_str="broker",
                status_str=DOCTOR_BLOCK_STR,
                reason_code_str="open_orders_present",
                detail_str=(
                    "Broker account has open orders. Doctor cannot certify "
                    "duplicate-order safety while unmanaged orders are live."
                ),
                extra_detail_dict={"open_order_id_list": open_order_id_list},
            )
        )
        detail_dict["overall_verdict_str"] = _overall_verdict_str(component_result_dict_list)
        return detail_dict

    component_result_dict_list.append(
        _component_result_dict(
            component_name_str="broker",
            status_str=DOCTOR_PASS_STR,
            reason_code_str="broker_account_ready",
            detail_str="Broker account is visible and snapshot is available.",
        )
    )

    reconciliation_result_obj = reconcile_account_state(
        model_position_map=decision_plan_obj.decision_base_position_map,
        model_cash_float=0.0,
        broker_snapshot_obj=broker_snapshot_obj,
    )
    reconciliation_dict = _reconciliation_dict(reconciliation_result_obj)
    detail_dict["reconciliation_dict"] = reconciliation_dict
    if not reconciliation_result_obj.passed_bool:
        component_result_dict_list.append(
            _component_result_dict(
                component_name_str="position_reconciliation",
                status_str=DOCTOR_BLOCK_STR,
                reason_code_str="position_reconciliation_mismatch",
                detail_str="DecisionPlan base positions do not match broker positions.",
                extra_detail_dict={"reconciliation_dict": reconciliation_dict},
            )
        )
        detail_dict["overall_verdict_str"] = _overall_verdict_str(component_result_dict_list)
        return detail_dict

    component_result_dict_list.append(
        _component_result_dict(
            component_name_str="position_reconciliation",
            status_str=DOCTOR_PASS_STR,
            reason_code_str="position_reconciliation_passed",
            detail_str="DecisionPlan base positions match broker positions.",
            extra_detail_dict={"reconciliation_dict": reconciliation_dict},
        )
    )

    touched_asset_list = get_touched_asset_list_for_decision_plan(
        decision_plan_obj,
        broker_position_map_dict=broker_snapshot_obj.position_amount_map,
    )
    try:
        live_price_snapshot_obj = broker_adapter_obj.get_live_price_snapshot(
            release_obj.account_route_str,
            touched_asset_list,
            execution_policy_str=decision_plan_obj.execution_policy_str,
        )
    except Exception as exc:
        component_result_dict_list.append(
            _component_result_dict(
                component_name_str="vplan_preview",
                status_str=DOCTOR_BLOCK_STR,
                reason_code_str="live_price_snapshot_error",
                detail_str=str(exc),
            )
        )
        detail_dict["overall_verdict_str"] = _overall_verdict_str(component_result_dict_list)
        return detail_dict

    missing_asset_list = sorted(
        asset_str
        for asset_str in touched_asset_list
        if asset_str not in live_price_snapshot_obj.asset_reference_price_map
    )
    if len(missing_asset_list) > 0:
        component_result_dict_list.append(
            _component_result_dict(
                component_name_str="vplan_preview",
                status_str=DOCTOR_BLOCK_STR,
                reason_code_str="missing_live_price",
                detail_str=f"Missing live reference prices for assets: {missing_asset_list}.",
                extra_detail_dict={"missing_asset_list": missing_asset_list},
            )
        )
        detail_dict["overall_verdict_str"] = _overall_verdict_str(component_result_dict_list)
        return detail_dict

    try:
        vplan_obj = build_vplan(
            release_obj=release_obj,
            decision_plan_obj=decision_plan_obj,
            broker_snapshot_obj=broker_snapshot_obj,
            live_price_snapshot_obj=live_price_snapshot_obj,
        )
    except Exception as exc:
        component_result_dict_list.append(
            _component_result_dict(
                component_name_str="vplan_preview",
                status_str=DOCTOR_BLOCK_STR,
                reason_code_str="vplan_preview_error",
                detail_str=str(exc),
            )
        )
        detail_dict["overall_verdict_str"] = _overall_verdict_str(component_result_dict_list)
        return detail_dict

    detail_dict["vplan_preview_dict"] = _vplan_preview_dict(vplan_obj)
    component_result_dict_list.append(
        _component_result_dict(
            component_name_str="vplan_preview",
            status_str=DOCTOR_PASS_STR,
            reason_code_str="vplan_preview_built",
            detail_str="VPlan and broker order requests built in memory.",
        )
    )

    detail_dict["overall_verdict_str"] = _overall_verdict_str(component_result_dict_list)
    return detail_dict


def _doctor_trace_event_name_str(component_name_str: str) -> str:
    return "doctor." + str(component_name_str).replace(" ", "_")


def _emit_doctor_trace_event_list(
    *,
    detail_dict: dict[str, object],
    trace_enabled_bool: bool,
    trace_log_root_path_str: str,
) -> None:
    if not trace_enabled_bool:
        return
    mode_str = str(detail_dict.get("mode_str") or "")
    if mode_str not in {"paper", "live"}:
        return
    as_of_timestamp_str = str(detail_dict.get("as_of_timestamp_str") or "")
    release_dict = dict(detail_dict.get("release_dict") or {})
    broker_dict = dict(detail_dict.get("broker_dict") or {})
    trace_context_dict = build_pod_trace_context_dict(
        mode_str=mode_str,
        pod_id_str=str(detail_dict.get("pod_id_str") or release_dict.get("pod_id_str") or "unknown"),
        account_route_str=str(
            release_dict.get("account_route_str")
            or broker_dict.get("account_route_str")
            or ""
        ),
        release_id_str=str(release_dict.get("release_id_str") or ""),
        as_of_timestamp_str=as_of_timestamp_str,
    )
    component_result_dict_list = list(detail_dict.get("component_result_dict_list") or [])
    for component_result_dict in component_result_dict_list:
        component_name_str = str(component_result_dict.get("component_name_str") or "component")
        log_pod_trace_event(
            _doctor_trace_event_name_str(component_name_str),
            trace_context_dict=trace_context_dict,
            status_str=str(component_result_dict.get("status_str") or "UNKNOWN"),
            reason_code_str=str(component_result_dict.get("reason_code_str") or "unknown"),
            payload_dict={
                "component_result_dict": dict(component_result_dict),
                "config_release_root_dict": dict(
                    detail_dict.get("config_release_root_dict") or {}
                )
                if component_name_str == "config_release_root"
                else {},
                "manifest_qualification_dict": dict(
                    detail_dict.get("manifest_qualification_dict") or {}
                )
                if component_name_str == "manifest_qualification"
                else {},
                "scheduler_gate_dict": dict(detail_dict.get("scheduler_gate_dict") or {})
                if component_name_str == "scheduler_gate"
                else {},
                "decision_plan_dict": dict(detail_dict.get("decision_plan_dict") or {})
                if component_name_str == "decision_plan"
                else {},
                "broker_dict": dict(detail_dict.get("broker_dict") or {})
                if component_name_str == "broker"
                else {},
                "reconciliation_dict": dict(detail_dict.get("reconciliation_dict") or {})
                if component_name_str == "position_reconciliation"
                else {},
                "vplan_preview_dict": dict(detail_dict.get("vplan_preview_dict") or {})
                if component_name_str == "vplan_preview"
                else {},
                "norgate_snapshot_sync_detail_dict": dict(
                    detail_dict.get("norgate_snapshot_sync_detail_dict") or {}
                )
                if component_name_str == "norgate_snapshot_sync"
                else {},
            },
            trace_enabled_bool=True,
            trace_log_root_path_str=trace_log_root_path_str,
        )
    log_pod_trace_event(
        "doctor.final_verdict",
        trace_context_dict=trace_context_dict,
        status_str=str(detail_dict.get("overall_verdict_str") or "UNKNOWN"),
        reason_code_str=str(detail_dict.get("overall_verdict_str") or "unknown").lower(),
        payload_dict={
            "overall_verdict_str": detail_dict.get("overall_verdict_str"),
            "component_result_dict_list": component_result_dict_list,
        },
        trace_enabled_bool=True,
        trace_log_root_path_str=trace_log_root_path_str,
    )


def compute_doctor_verdict(
    *,
    releases_root_path_str: str | None,
    env_mode_str: str,
    as_of_ts: datetime,
    command_name_str: str = DOCTOR_FEATURE_NAME_STR,
    pod_id_str: str | None = None,
    releases_root_explicit_bool: bool = True,
    config_env_path_str: str | None = None,
    db_path_str: str | None = None,
    state_store_obj: Any | None = None,
    pod_state_loader_fn: PodStateLoader_Fn | None = None,
    norgate_snapshot_sync_checker_fn: NorgateSnapshotSyncChecker_Fn | None = None,
    broker_adapter_resolver_obj: Any | None = None,
    broker_host_str: str | None = None,
    broker_port_int: int | None = None,
    broker_client_id_int: int | None = None,
    broker_timeout_seconds_float: float | None = None,
    adapter_factory_func: Callable[[str, int, int, float], Any] | None = None,
    loaded_config_env_dict: dict[str, str] | None = None,
    trace_enabled_bool: bool = True,
    trace_log_root_path_str: str = DEFAULT_POD_TRACE_LOG_ROOT_PATH_STR,
) -> dict[str, object]:
    detail_dict = _compute_doctor_verdict_impl(
        releases_root_path_str=releases_root_path_str,
        env_mode_str=env_mode_str,
        as_of_ts=as_of_ts,
        command_name_str=command_name_str,
        pod_id_str=pod_id_str,
        releases_root_explicit_bool=releases_root_explicit_bool,
        config_env_path_str=config_env_path_str,
        db_path_str=db_path_str,
        state_store_obj=state_store_obj,
        pod_state_loader_fn=pod_state_loader_fn,
        norgate_snapshot_sync_checker_fn=norgate_snapshot_sync_checker_fn,
        broker_adapter_resolver_obj=broker_adapter_resolver_obj,
        broker_host_str=broker_host_str,
        broker_port_int=broker_port_int,
        broker_client_id_int=broker_client_id_int,
        broker_timeout_seconds_float=broker_timeout_seconds_float,
        adapter_factory_func=adapter_factory_func,
        loaded_config_env_dict=loaded_config_env_dict,
    )
    try:
        _emit_doctor_trace_event_list(
            detail_dict=detail_dict,
            trace_enabled_bool=trace_enabled_bool,
            trace_log_root_path_str=trace_log_root_path_str,
        )
    except Exception:
        pass
    return detail_dict


def render_doctor_detail_str(detail_dict: dict[str, object]) -> str:
    release_dict = dict(detail_dict.get("release_dict") or {})
    manifest_qualification_dict = dict(detail_dict.get("manifest_qualification_dict") or {})
    decision_plan_dict = dict(detail_dict.get("decision_plan_dict") or {})
    broker_dict = dict(detail_dict.get("broker_dict") or {})
    vplan_preview_dict = dict(detail_dict.get("vplan_preview_dict") or {})
    config_release_root_dict = dict(detail_dict.get("config_release_root_dict") or {})
    scheduler_gate_dict = dict(detail_dict.get("scheduler_gate_dict") or {})
    lifecycle_dict = dict(detail_dict.get("persisted_lifecycle_dict") or {})
    sync_detail_dict = dict(detail_dict.get("norgate_snapshot_sync_detail_dict") or {})
    reconciliation_dict = dict(detail_dict.get("reconciliation_dict") or {})
    component_result_dict_list = list(detail_dict.get("component_result_dict_list") or [])
    selected_manifest_path_str = str(
        config_release_root_dict.get("selected_release_source_path_str")
        or manifest_qualification_dict.get("source_path_str")
        or release_dict.get("source_path_str")
        or ""
    )

    line_list = [
        "Live Doctor",
        f"- VERDICT: {detail_dict.get('overall_verdict_str', 'UNKNOWN')}",
        f"- Manifest: {selected_manifest_path_str or 'none'}",
        f"- Mode: {detail_dict.get('mode_str', '')}",
        f"- Pod: {detail_dict.get('pod_id_str') or release_dict.get('pod_id_str') or 'none'}",
        f"- As-of: {detail_dict.get('as_of_timestamp_str', '')}",
    ]
    if release_dict:
        line_list.extend(
            [
                f"- Release: {release_dict.get('release_id_str')}",
                f"- Strategy: {release_dict.get('strategy_import_str')}",
                f"- Account: {release_dict.get('account_route_str')}",
                f"- Auto-submit: {'on' if bool(release_dict.get('auto_submit_enabled_bool')) else 'off'}",
            ]
        )

    if manifest_qualification_dict:
        line_list.extend(
            [
                "",
                "Manifest Qualification",
                f"- Qualified: {bool(manifest_qualification_dict.get('qualified_bool'))}",
            ]
        )
        if manifest_qualification_dict.get("error_str"):
            line_list.append(f"- Error: {manifest_qualification_dict.get('error_str')}")
        else:
            line_list.extend(
                [
                    f"- Source YAML: {manifest_qualification_dict.get('source_path_str')}",
                    f"- Release: {manifest_qualification_dict.get('release_id_str')}",
                    f"- Strategy: {manifest_qualification_dict.get('strategy_import_str')}",
                    f"- Data profile: {manifest_qualification_dict.get('data_profile_str')}",
                    f"- Signal clock: {manifest_qualification_dict.get('signal_clock_str')}",
                    f"- Execution policy: {manifest_qualification_dict.get('execution_policy_str')}",
                    f"- Calendar: {manifest_qualification_dict.get('session_calendar_id_str')}",
                    f"- Account route: {manifest_qualification_dict.get('account_route_str')}",
                    f"- Account placeholder: {bool(manifest_qualification_dict.get('account_route_placeholder_bool'))}",
                    (
                        "- Broker route: "
                        f"{manifest_qualification_dict.get('broker_host_str')}:"
                        f"{manifest_qualification_dict.get('broker_port_int')} "
                        f"client_id={manifest_qualification_dict.get('broker_client_id_int')}"
                    ),
                    f"- Broker timeout seconds: {manifest_qualification_dict.get('broker_timeout_seconds_float')}",
                    f"- Pod budget fraction: {manifest_qualification_dict.get('pod_budget_fraction_float')}",
                    f"- Auto-submit: {'on' if bool(manifest_qualification_dict.get('auto_submit_enabled_bool')) else 'off'}",
                ]
            )

    line_list.append("")
    line_list.append("Components")
    for result_dict in component_result_dict_list:
        line_list.append(
            "- "
            f"{result_dict.get('component_name_str')}: "
            f"{result_dict.get('status_str')} "
            f"({result_dict.get('reason_code_str')}) - "
            f"{result_dict.get('detail_str')}"
        )

    if config_release_root_dict:
        line_list.extend(
            [
                "",
                "VPS Config / Release Root",
                f"- config.env: {config_release_root_dict.get('config_env_path_str')}",
                f"- Env NORGATE_RELEASES_ROOT: {config_release_root_dict.get('env_release_root_raw_str')}",
                f"- Env root resolved: {config_release_root_dict.get('env_release_root_resolved_str')}",
                f"- Effective root: {config_release_root_dict.get('effective_release_root_resolved_str')}",
                f"- Root source: {config_release_root_dict.get('release_root_source_str')}",
                f"- Env/effective match: {bool(config_release_root_dict.get('release_root_match_bool'))}",
                f"- Selected YAML: {config_release_root_dict.get('selected_release_source_path_str')}",
            ]
        )

    if scheduler_gate_dict:
        line_list.extend(
            [
                "",
                "Scheduler/Data",
                f"- Reason: {scheduler_gate_dict.get('reason_code_str')}",
                f"- Due: {bool(scheduler_gate_dict.get('due_bool'))}",
                f"- Latest heartbeat session: {scheduler_gate_dict.get('latest_heartbeat_session_date_str')}",
            ]
        )
    if lifecycle_dict:
        line_list.extend(
            [
                "",
                "Persisted Lifecycle",
                f"- Reason: {lifecycle_dict.get('reason_code_str')}",
                f"- Active: {bool(lifecycle_dict.get('active_lifecycle_bool'))}",
                f"- Expired: {bool(lifecycle_dict.get('expired_lifecycle_bool'))}",
            ]
        )
    if sync_detail_dict:
        line_list.extend(
            [
                "",
                "Norgate Snapshot Sync",
                f"- Status: {sync_detail_dict.get('status_str')}",
                f"- Reason: {sync_detail_dict.get('reason_code_str')}",
                f"- Required profiles: {sync_detail_dict.get('required_profile_list')}",
            ]
        )

    if decision_plan_dict:
        metadata_dict = dict(decision_plan_dict.get("snapshot_metadata_dict") or {})
        line_list.extend(
            [
                "",
                "DecisionPlan",
                f"- Signal: {decision_plan_dict.get('signal_timestamp_str')}",
                f"- Submit: {decision_plan_dict.get('submission_timestamp_str')}",
                f"- Execute: {decision_plan_dict.get('target_execution_timestamp_str')}",
                f"- Book: {decision_plan_dict.get('decision_book_type_str')}",
                f"- Target weights: {decision_plan_dict.get('target_weight_map')}",
            ]
        )
        if metadata_dict:
            line_list.append(f"- Metadata: {metadata_dict}")

    if broker_dict:
        line_list.extend(
            [
                "",
                "Broker",
                f"- Host: {broker_dict.get('broker_host_str')}:{broker_dict.get('broker_port_int')}",
                f"- Client ID: {broker_dict.get('broker_client_id_int')}",
                f"- Visible accounts: {broker_dict.get('visible_account_route_list')}",
                f"- Expected account visible: {broker_dict.get('expected_account_visible_bool')}",
                f"- Session mode: {broker_dict.get('session_mode_str')}",
                f"- NetLiq: {broker_dict.get('net_liq_float')}",
                f"- AvailableFunds: {broker_dict.get('available_funds_float')}",
                f"- Open orders: {broker_dict.get('open_order_id_list')}",
            ]
        )
    if reconciliation_dict:
        line_list.extend(
            [
                "",
                "Position Reconciliation",
                f"- Passed: {bool(reconciliation_dict.get('passed_bool'))}",
                f"- Mismatches: {reconciliation_dict.get('mismatch_dict')}",
            ]
        )

    if vplan_preview_dict:
        order_request_dict_list = list(vplan_preview_dict.get("broker_order_request_dict_list") or [])
        line_list.extend(
            [
                "",
                "Order Intent Preview",
                "- Dry-run only: true",
                f"- Order count: {len(order_request_dict_list)}",
                f"- Pod budget: {vplan_preview_dict.get('pod_budget_float')}",
            ]
        )
        for order_request_dict in order_request_dict_list:
            amount_float = float(order_request_dict.get("amount_float", 0.0))
            side_str = "BUY" if amount_float > 0 else "SELL"
            line_list.append(
                "- "
                f"{order_request_dict.get('asset_str')} {side_str} "
                f"{abs(amount_float):.4f} shares "
                f"{order_request_dict.get('broker_order_type_str')} "
                f"ref={order_request_dict.get('sizing_reference_price_float')}"
            )

    return "\n".join(line_list)


