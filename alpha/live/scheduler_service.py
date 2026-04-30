from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from alpha.live import runner, scheduler_utils
from alpha.live.logging_utils import (
    DEFAULT_LOG_PATH_STR,
    log_event,
    log_operator_message,
)
from alpha.live.models import DecisionPlan, LiveRelease, VPlan
from alpha.live.release_manifest import load_release_list, validate_enabled_deployment_for_mode
from alpha.live.state_store_v2 import LiveStateStore


DEFAULT_ACTIVE_POLL_SECONDS_INT = 30
DEFAULT_IDLE_MAX_SLEEP_SECONDS_INT = 3600
DEFAULT_RECONCILE_GRACE_SECONDS_INT = 300
DEFAULT_ERROR_RETRY_SECONDS_INT = 60
DEFAULT_OPERATOR_HEARTBEAT_SECONDS_INT = 900
DEFAULT_ACTIVE_OPERATOR_HEARTBEAT_SECONDS_INT = 30
DEFAULT_SUBMIT_STUCK_SECONDS_INT = 60
DEFAULT_EOD_SNAPSHOT_BUFFER_MINUTES_INT = runner.DEFAULT_EOD_SNAPSHOT_BUFFER_MINUTES_INT
DEFAULT_RELEASES_ROOT_PATH_STR = str(Path(__file__).resolve().parent / "releases")
DEFAULT_DB_PATH_STR = runner.DEFAULT_DB_PATH_STR
DEFAULT_INCUBATION_DB_PATH_STR = runner.DEFAULT_INCUBATION_DB_PATH_STR


@dataclass(frozen=True)
class SchedulerDecision:
    as_of_timestamp_ts: datetime
    env_mode_str: str
    due_now_bool: bool
    active_poll_bool: bool
    next_phase_str: str
    reason_code_str: str
    next_due_timestamp_ts: datetime
    related_pod_id_list: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "as_of_timestamp_str": self.as_of_timestamp_ts.isoformat(),
            "env_mode_str": self.env_mode_str,
            "due_now_bool": bool(self.due_now_bool),
            "active_poll_bool": bool(self.active_poll_bool),
            "next_phase_str": self.next_phase_str,
            "reason_code_str": self.reason_code_str,
            "next_due_timestamp_str": self.next_due_timestamp_ts.isoformat(),
            "related_pod_id_list": list(self.related_pod_id_list),
        }


def _parse_as_of_timestamp_ts(as_of_timestamp_str: str | None) -> datetime:
    if as_of_timestamp_str is None:
        return datetime.now(tz=UTC)
    return datetime.fromisoformat(as_of_timestamp_str)


def _resolve_db_path_for_mode_str(
    db_path_str: str | None,
    env_mode_str: str,
) -> str:
    return runner._resolve_db_path_for_mode_str(
        db_path_str=db_path_str,
        env_mode_str=env_mode_str,
    )


def _load_release_list_and_sync(
    releases_root_path_str: str,
    state_store_obj: LiveStateStore,
) -> list[LiveRelease]:
    release_list = load_release_list(releases_root_path_str)
    state_store_obj.upsert_release_list(release_list)
    return release_list


def _validate_release_root_for_mutation(
    releases_root_path_str: str,
    env_mode_str: str,
) -> None:
    release_list = load_release_list(releases_root_path_str)
    validate_enabled_deployment_for_mode(release_list, env_mode_str)


def _get_current_cycle_vplan_obj(
    state_store_obj: LiveStateStore,
    decision_plan_obj: DecisionPlan | None,
) -> VPlan | None:
    if decision_plan_obj is None or decision_plan_obj.decision_plan_id_int is None:
        return None
    return state_store_obj.get_latest_vplan_for_decision(int(decision_plan_obj.decision_plan_id_int))


def _build_candidate_dict(
    *,
    priority_int: int,
    due_timestamp_ts: datetime,
    next_phase_str: str,
    reason_code_str: str,
    pod_id_str: str,
    active_poll_bool: bool,
) -> dict[str, object]:
    return {
        "priority_int": int(priority_int),
        "due_timestamp_ts": due_timestamp_ts,
        "next_phase_str": next_phase_str,
        "reason_code_str": reason_code_str,
        "pod_id_str": pod_id_str,
        "active_poll_bool": bool(active_poll_bool),
    }


def get_scheduler_decision(
    state_store_obj: LiveStateStore,
    as_of_ts: datetime,
    releases_root_path_str: str,
    env_mode_str: str,
    reconcile_grace_seconds_int: int = DEFAULT_RECONCILE_GRACE_SECONDS_INT,
    idle_max_sleep_seconds_int: int = DEFAULT_IDLE_MAX_SLEEP_SECONDS_INT,
) -> SchedulerDecision:
    release_list = _load_release_list_and_sync(releases_root_path_str, state_store_obj)
    enabled_release_list = [
        release_obj
        for release_obj in release_list
        if release_obj.enabled_bool and release_obj.mode_str == env_mode_str
    ]

    immediate_candidate_list: list[dict[str, object]] = []
    future_candidate_list: list[dict[str, object]] = []
    manual_review_pod_id_list: list[str] = []
    parked_manual_review_pod_id_list: list[str] = []
    idle_probe_pod_id_list: list[str] = []

    for release_obj in enabled_release_list:
        latest_decision_plan_obj = state_store_obj.get_latest_decision_plan_for_pod(release_obj.pod_id_str)
        current_vplan_obj = _get_current_cycle_vplan_obj(state_store_obj, latest_decision_plan_obj)
        build_gate_dict = scheduler_utils.evaluate_build_gate_dict(release_obj, as_of_ts)
        eod_due_timestamp_ts = runner._eod_snapshot_due_timestamp_ts(
            release_obj=release_obj,
            as_of_ts=as_of_ts,
            eod_snapshot_buffer_minutes_int=DEFAULT_EOD_SNAPSHOT_BUFFER_MINUTES_INT,
        )
        if eod_due_timestamp_ts is not None:
            eod_market_date_str = runner._eod_snapshot_market_date_str(
                release_obj=release_obj,
                as_of_ts=as_of_ts,
            )
            eod_snapshot_exists_bool = runner._pod_has_stage_snapshot_for_market_date_bool(
                state_store_obj=state_store_obj,
                release_obj=release_obj,
                snapshot_stage_str="eod",
                market_date_str=eod_market_date_str,
            )
            active_vplan_blocks_eod_bool = current_vplan_obj is not None and current_vplan_obj.status_str in (
                "ready",
                "submitted",
                "submitting",
            )
            # *** CRITICAL*** EOD snapshot is never scheduled ahead of an
            # active order plan. Submit/reconcile/manual review must settle
            # first so close-state sampling cannot overwrite operational truth.
            if not eod_snapshot_exists_bool and not active_vplan_blocks_eod_bool:
                if eod_due_timestamp_ts <= as_of_ts:
                    immediate_candidate_list.append(
                        _build_candidate_dict(
                            priority_int=5,
                            due_timestamp_ts=as_of_ts,
                            next_phase_str="eod_snapshot",
                            reason_code_str="eod_snapshot_due",
                            pod_id_str=release_obj.pod_id_str,
                            active_poll_bool=False,
                        )
                    )
                else:
                    future_candidate_list.append(
                        _build_candidate_dict(
                            priority_int=5,
                            due_timestamp_ts=eod_due_timestamp_ts,
                            next_phase_str="eod_snapshot",
                            reason_code_str="waiting_for_eod_snapshot",
                            pod_id_str=release_obj.pod_id_str,
                            active_poll_bool=False,
                        )
                    )

        if latest_decision_plan_obj is None:
            if bool(build_gate_dict["due_bool"]):
                immediate_candidate_list.append(
                    _build_candidate_dict(
                        priority_int=4,
                        due_timestamp_ts=as_of_ts,
                        next_phase_str="build_decision_plan",
                        reason_code_str="ready_to_build_decision_plan",
                        pod_id_str=release_obj.pod_id_str,
                        active_poll_bool=False,
                    )
                )
            else:
                idle_probe_pod_id_list.append(release_obj.pod_id_str)
            continue

        if (
            latest_decision_plan_obj.status_str in ("planned", "vplan_ready")
            and scheduler_utils.is_execution_window_expired_bool(
                latest_decision_plan_obj.execution_policy_str,
                latest_decision_plan_obj.target_execution_timestamp_ts,
                as_of_ts,
            )
        ):
            immediate_candidate_list.append(
                _build_candidate_dict(
                    priority_int=0,
                    due_timestamp_ts=as_of_ts,
                    next_phase_str="expire_stale",
                    reason_code_str="submission_window_expired",
                    pod_id_str=release_obj.pod_id_str,
                    active_poll_bool=False,
                )
            )
            continue

        if current_vplan_obj is not None and current_vplan_obj.status_str in ("submitted", "submitting"):
            if runner.is_vplan_execution_exception_parked(state_store_obj, current_vplan_obj):
                parked_manual_review_pod_id_list.append(release_obj.pod_id_str)
                continue
            reconcile_due_timestamp_ts = current_vplan_obj.target_execution_timestamp_ts + timedelta(
                seconds=reconcile_grace_seconds_int
            )
            if reconcile_due_timestamp_ts <= as_of_ts:
                immediate_candidate_list.append(
                    _build_candidate_dict(
                        priority_int=1,
                        due_timestamp_ts=as_of_ts,
                        next_phase_str="post_execution_reconcile",
                        reason_code_str="ready_to_reconcile",
                        pod_id_str=release_obj.pod_id_str,
                        active_poll_bool=True,
                    )
                )
            else:
                future_candidate_list.append(
                    _build_candidate_dict(
                        priority_int=1,
                        due_timestamp_ts=reconcile_due_timestamp_ts,
                        next_phase_str="post_execution_reconcile",
                        reason_code_str="waiting_for_post_execution_reconcile",
                        pod_id_str=release_obj.pod_id_str,
                        active_poll_bool=True,
                    )
                )
            continue

        if current_vplan_obj is not None and current_vplan_obj.status_str == "ready":
            if release_obj.auto_submit_enabled_bool:
                immediate_candidate_list.append(
                    _build_candidate_dict(
                        priority_int=2,
                        due_timestamp_ts=as_of_ts,
                        next_phase_str="submit_vplan",
                        reason_code_str="vplan_ready",
                        pod_id_str=release_obj.pod_id_str,
                        active_poll_bool=False,
                    )
                )
            else:
                manual_review_pod_id_list.append(release_obj.pod_id_str)
            continue

        if latest_decision_plan_obj.status_str == "planned":
            if latest_decision_plan_obj.submission_timestamp_ts <= as_of_ts:
                immediate_candidate_list.append(
                    _build_candidate_dict(
                        priority_int=3,
                        due_timestamp_ts=as_of_ts,
                        next_phase_str="build_vplan",
                        reason_code_str="ready_to_build_vplan",
                        pod_id_str=release_obj.pod_id_str,
                        active_poll_bool=False,
                    )
                )
            else:
                future_candidate_list.append(
                    _build_candidate_dict(
                        priority_int=3,
                        due_timestamp_ts=latest_decision_plan_obj.submission_timestamp_ts,
                        next_phase_str="build_vplan",
                        reason_code_str="waiting_for_submission_window",
                        pod_id_str=release_obj.pod_id_str,
                        active_poll_bool=False,
                    )
                )
            continue

        if latest_decision_plan_obj.status_str in ("completed", "expired", "blocked"):
            if bool(build_gate_dict["due_bool"]):
                immediate_candidate_list.append(
                    _build_candidate_dict(
                        priority_int=4,
                        due_timestamp_ts=as_of_ts,
                        next_phase_str="build_decision_plan",
                        reason_code_str="ready_to_build_decision_plan",
                        pod_id_str=release_obj.pod_id_str,
                        active_poll_bool=False,
                    )
                )
            else:
                idle_probe_pod_id_list.append(release_obj.pod_id_str)
            continue

    if len(immediate_candidate_list) > 0:
        top_priority_int = min(int(candidate_dict["priority_int"]) for candidate_dict in immediate_candidate_list)
        selected_candidate_list = [
            candidate_dict
            for candidate_dict in immediate_candidate_list
            if int(candidate_dict["priority_int"]) == top_priority_int
        ]
        selected_candidate_dict = selected_candidate_list[0]
        related_pod_id_list = sorted(
            {str(candidate_dict["pod_id_str"]) for candidate_dict in selected_candidate_list}
        )
        return SchedulerDecision(
            as_of_timestamp_ts=as_of_ts,
            env_mode_str=env_mode_str,
            due_now_bool=True,
            active_poll_bool=any(bool(candidate_dict["active_poll_bool"]) for candidate_dict in selected_candidate_list),
            next_phase_str=str(selected_candidate_dict["next_phase_str"]),
            reason_code_str=str(selected_candidate_dict["reason_code_str"]),
            next_due_timestamp_ts=as_of_ts,
            related_pod_id_list=related_pod_id_list,
        )

    if len(future_candidate_list) > 0:
        selected_candidate_dict = min(
            future_candidate_list,
            key=lambda candidate_dict: (
                candidate_dict["due_timestamp_ts"],
                int(candidate_dict["priority_int"]),
                str(candidate_dict["pod_id_str"]),
            ),
        )
        selected_due_timestamp_ts = selected_candidate_dict["due_timestamp_ts"]
        related_pod_id_list = sorted(
            {
                str(candidate_dict["pod_id_str"])
                for candidate_dict in future_candidate_list
                if candidate_dict["due_timestamp_ts"] == selected_due_timestamp_ts
                and str(candidate_dict["next_phase_str"]) == str(selected_candidate_dict["next_phase_str"])
            }
        )
        return SchedulerDecision(
            as_of_timestamp_ts=as_of_ts,
            env_mode_str=env_mode_str,
            due_now_bool=False,
            active_poll_bool=bool(selected_candidate_dict["active_poll_bool"]),
            next_phase_str=str(selected_candidate_dict["next_phase_str"]),
            reason_code_str=str(selected_candidate_dict["reason_code_str"]),
            next_due_timestamp_ts=selected_due_timestamp_ts,
            related_pod_id_list=related_pod_id_list,
        )

    if len(parked_manual_review_pod_id_list) > 0:
        return SchedulerDecision(
            as_of_timestamp_ts=as_of_ts,
            env_mode_str=env_mode_str,
            due_now_bool=False,
            active_poll_bool=False,
            next_phase_str="manual_review_pending",
            reason_code_str="execution_exception_parked",
            next_due_timestamp_ts=as_of_ts + timedelta(seconds=idle_max_sleep_seconds_int),
            related_pod_id_list=sorted(set(parked_manual_review_pod_id_list)),
        )

    if len(manual_review_pod_id_list) > 0:
        return SchedulerDecision(
            as_of_timestamp_ts=as_of_ts,
            env_mode_str=env_mode_str,
            due_now_bool=False,
            active_poll_bool=False,
            next_phase_str="manual_review_pending",
            reason_code_str="manual_review_required",
            next_due_timestamp_ts=as_of_ts + timedelta(seconds=idle_max_sleep_seconds_int),
            related_pod_id_list=sorted(set(manual_review_pod_id_list)),
        )

    return SchedulerDecision(
        as_of_timestamp_ts=as_of_ts,
        env_mode_str=env_mode_str,
        due_now_bool=False,
        active_poll_bool=False,
        next_phase_str="idle_probe",
        reason_code_str="no_due_work",
        next_due_timestamp_ts=as_of_ts + timedelta(seconds=idle_max_sleep_seconds_int),
        related_pod_id_list=sorted(set(idle_probe_pod_id_list)),
    )


def _build_sleep_seconds_float(
    scheduler_decision_obj: SchedulerDecision,
    as_of_ts: datetime,
    active_poll_seconds_int: int,
    idle_max_sleep_seconds_int: int,
) -> float:
    delta_seconds_float = max(
        0.0,
        (scheduler_decision_obj.next_due_timestamp_ts - as_of_ts).total_seconds(),
    )
    if scheduler_decision_obj.active_poll_bool:
        return min(float(active_poll_seconds_int), delta_seconds_float or float(active_poll_seconds_int))
    return min(float(idle_max_sleep_seconds_int), delta_seconds_float or float(idle_max_sleep_seconds_int))


def _build_related_pod_status_dict_list(
    state_store_obj: LiveStateStore,
    as_of_ts: datetime,
    releases_root_path_str: str,
    env_mode_str: str,
    related_pod_id_list: list[str],
) -> list[dict[str, object]]:
    if len(related_pod_id_list) == 0:
        return []
    status_summary_dict = runner.get_status_summary(
        state_store_obj=state_store_obj,
        as_of_ts=as_of_ts,
        releases_root_path_str=releases_root_path_str,
        env_mode_str=env_mode_str,
    )
    related_pod_id_set = set(related_pod_id_list)
    return [
        dict(pod_status_dict)
        for pod_status_dict in status_summary_dict["pod_status_dict_list"]
        if str(pod_status_dict["pod_id_str"]) in related_pod_id_set
    ]


def _build_related_execution_report_dict_list(
    state_store_obj: LiveStateStore,
    as_of_ts: datetime,
    releases_root_path_str: str,
    related_pod_id_list: list[str],
) -> list[dict[str, object]]:
    if len(related_pod_id_list) == 0:
        return []
    execution_report_summary_dict = runner.get_execution_report_summary(
        state_store_obj=state_store_obj,
        as_of_ts=as_of_ts,
        releases_root_path_str=releases_root_path_str,
    )
    related_pod_id_set = set(related_pod_id_list)
    deduped_execution_report_dict_list: list[dict[str, object]] = []
    seen_execution_key_set: set[tuple[str, int]] = set()
    for execution_report_dict in execution_report_summary_dict["execution_report_dict_list"]:
        if str(execution_report_dict["pod_id_str"]) not in related_pod_id_set:
            continue
        execution_key_tup = (
            str(execution_report_dict["pod_id_str"]),
            int(execution_report_dict["latest_vplan_id_int"]),
        )
        if execution_key_tup in seen_execution_key_set:
            continue
        seen_execution_key_set.add(execution_key_tup)
        deduped_execution_report_dict_list.append(dict(execution_report_dict))
    return deduped_execution_report_dict_list


def _build_related_vplan_dict_list(
    state_store_obj: LiveStateStore,
    as_of_ts: datetime,
    releases_root_path_str: str,
    related_pod_id_list: list[str],
) -> list[dict[str, object]]:
    if len(related_pod_id_list) == 0:
        return []
    related_pod_id_set = set(related_pod_id_list)
    vplan_summary_dict = runner.show_vplan_summary(
        state_store_obj=state_store_obj,
        as_of_ts=as_of_ts,
        releases_root_path_str=releases_root_path_str,
    )
    deduped_vplan_dict_list: list[dict[str, object]] = []
    seen_vplan_key_set: set[tuple[str, int]] = set()
    for vplan_dict in vplan_summary_dict["vplan_dict_list"]:
        if str(vplan_dict["pod_id_str"]) not in related_pod_id_set:
            continue
        vplan_id_int = int(vplan_dict["vplan_id_int"])
        vplan_key_tup = (str(vplan_dict["pod_id_str"]), vplan_id_int)
        if vplan_key_tup in seen_vplan_key_set:
            continue
        seen_vplan_key_set.add(vplan_key_tup)
        deduped_vplan_dict_list.append(dict(vplan_dict))
    return deduped_vplan_dict_list


def _build_related_broker_order_snapshot_dict_list(
    state_store_obj: LiveStateStore,
    related_pod_id_list: list[str],
) -> list[dict[str, object]]:
    broker_order_snapshot_dict_list: list[dict[str, object]] = []
    for pod_id_str in related_pod_id_list:
        latest_vplan_obj = state_store_obj.get_latest_vplan_for_pod(pod_id_str)
        if latest_vplan_obj is None or latest_vplan_obj.vplan_id_int is None:
            continue
        broker_order_row_dict_list = state_store_obj.get_broker_order_row_dict_list_for_vplan(
            int(latest_vplan_obj.vplan_id_int)
        )
        if len(broker_order_row_dict_list) == 0:
            continue
        broker_order_snapshot_dict_list.append(
            {
                "pod_id_str": str(pod_id_str),
                "latest_vplan_id_int": int(latest_vplan_obj.vplan_id_int),
                "submit_ack_status_str": str(latest_vplan_obj.submit_ack_status_str),
                "ack_coverage_ratio_float": latest_vplan_obj.ack_coverage_ratio_float,
                "missing_ack_count_int": int(latest_vplan_obj.missing_ack_count_int),
                "broker_order_count_int": len(broker_order_row_dict_list),
                "broker_order_row_dict_list": broker_order_row_dict_list,
            }
        )
    return broker_order_snapshot_dict_list


def _format_operator_field_timestamp_str(timestamp_obj: datetime | str | None) -> str | None:
    if timestamp_obj is None:
        return None
    normalized_timestamp_ts = (
        datetime.fromisoformat(timestamp_obj) if isinstance(timestamp_obj, str) else timestamp_obj
    )
    if normalized_timestamp_ts.tzinfo is None:
        normalized_timestamp_ts = normalized_timestamp_ts.replace(tzinfo=UTC)
    return normalized_timestamp_ts.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def _format_ratio_str(numerator_int: int, denominator_int: int) -> str:
    return f"{int(numerator_int)}/{int(denominator_int)}"


def _count_actionable_vplan_order_count_int(
    vplan_obj: VPlan | None,
    tolerance_float: float = 1e-9,
) -> int:
    if vplan_obj is None:
        return 0
    return sum(
        1
        for vplan_row_obj in vplan_obj.vplan_row_list
        if abs(float(vplan_row_obj.order_delta_share_float)) > tolerance_float
    )


def _compute_gross_order_notional_float(
    vplan_obj: VPlan | None,
    tolerance_float: float = 1e-9,
) -> float:
    if vplan_obj is None:
        return 0.0
    return sum(
        abs(float(vplan_row_obj.order_delta_share_float)) * float(vplan_row_obj.live_reference_price_float)
        for vplan_row_obj in vplan_obj.vplan_row_list
        if abs(float(vplan_row_obj.order_delta_share_float)) > tolerance_float
    )


def _build_submit_progress_summary_dict(
    broker_order_row_dict_list: list[dict[str, object]],
    broker_ack_row_dict_list: list[dict[str, object]],
    tolerance_float: float = 1e-9,
) -> dict[str, int]:
    order_count_int = 0
    filled_order_count_int = 0
    partial_fill_order_count_int = 0
    for broker_order_row_dict in broker_order_row_dict_list:
        order_amount_float = abs(float(broker_order_row_dict.get("amount_float", 0.0)))
        if order_amount_float <= tolerance_float:
            continue
        order_count_int += 1
        filled_amount_float = abs(float(broker_order_row_dict.get("filled_amount_float", 0.0)))
        if filled_amount_float <= tolerance_float:
            continue
        if abs(order_amount_float - filled_amount_float) <= tolerance_float or str(
            broker_order_row_dict.get("status_str", "")
        ).lower() == "filled":
            filled_order_count_int += 1
            continue
        partial_fill_order_count_int += 1
    broker_ack_count_int = sum(
        1
        for broker_ack_row_dict in broker_ack_row_dict_list
        if bool(broker_ack_row_dict.get("broker_response_ack_bool", False))
    )
    ack_request_count_int = max(len(broker_ack_row_dict_list), order_count_int)
    remaining_order_count_int = max(order_count_int - filled_order_count_int, 0)
    return {
        "order_count_int": int(order_count_int),
        "filled_order_count_int": int(filled_order_count_int),
        "partial_fill_order_count_int": int(partial_fill_order_count_int),
        "broker_ack_count_int": int(broker_ack_count_int),
        "ack_request_count_int": int(ack_request_count_int),
        "remaining_order_count_int": int(remaining_order_count_int),
    }


def _build_submit_ack_operator_message_spec_dict(
    *,
    timestamp_obj: datetime,
    base_field_map_dict: dict[str, object],
    vplan_id_int: int | None,
    submit_ack_status_str: str | None,
    missing_ack_count_int: int,
    submit_progress_summary_dict: dict[str, int],
) -> dict[str, object] | None:
    ack_request_count_int = int(submit_progress_summary_dict["ack_request_count_int"])
    if ack_request_count_int <= 0 and str(submit_ack_status_str or "not_checked") == "not_checked":
        return None
    phase_action_str = "submit_ack.ok"
    level_str = "INFO"
    if str(submit_ack_status_str) == "missing_critical":
        phase_action_str = "submit_ack.warn"
        level_str = "WARN"
    return _build_operator_message_spec_dict(
        level_str=level_str,
        phase_action_str=phase_action_str,
        timestamp_obj=timestamp_obj,
        field_map_dict={
            **base_field_map_dict,
            "vplan": vplan_id_int,
            "acked": _format_ratio_str(
                int(submit_progress_summary_dict["broker_ack_count_int"]),
                ack_request_count_int,
            ),
            "missing": int(missing_ack_count_int),
        },
    )


def _build_fill_progress_operator_message_spec_dict(
    *,
    timestamp_obj: datetime,
    base_field_map_dict: dict[str, object],
    vplan_id_int: int | None,
    submit_progress_summary_dict: dict[str, int],
) -> dict[str, object] | None:
    order_count_int = int(submit_progress_summary_dict["order_count_int"])
    if order_count_int <= 0:
        return None
    filled_order_count_int = int(submit_progress_summary_dict["filled_order_count_int"])
    partial_fill_order_count_int = int(submit_progress_summary_dict["partial_fill_order_count_int"])
    phase_action_str = "fill.none"
    if filled_order_count_int >= order_count_int:
        phase_action_str = "fill.complete"
    elif partial_fill_order_count_int > 0:
        phase_action_str = "fill.partial"
    return _build_operator_message_spec_dict(
        level_str="INFO",
        phase_action_str=phase_action_str,
        timestamp_obj=timestamp_obj,
        field_map_dict={
            **base_field_map_dict,
            "vplan": vplan_id_int,
            "filled": _format_ratio_str(filled_order_count_int, order_count_int),
            "partial": (
                int(partial_fill_order_count_int) if partial_fill_order_count_int > 0 else None
            ),
            "remaining": int(submit_progress_summary_dict["remaining_order_count_int"]),
        },
    )


def _humanize_reason_code_str(reason_code_str: str | None) -> str:
    reason_map_dict = {
        "active_decision_plan_exists": "active decision plan already exists",
        "account_not_visible": "target account is not visible in the broker session",
        "broker_not_ready": "broker session is not ready for the target account",
        "carry_forward_snapshot_ready": "signal is ready and waiting for the next submission window",
        "completed": "cycle completed",
        "dtb3_stale": "macro data is stale and decision creation was skipped",
        "duplicate_submission_guard": "submission skipped because broker orders already exist for this VPlan",
        "eod_snapshot_completed": "EOD broker snapshot recorded",
        "eod_snapshot_due": "EOD broker snapshot is due",
        "eod_snapshot_already_exists": "EOD snapshot already exists for this market date",
        "env_mode_mismatch": "release mode does not match the current runtime mode",
        "execution_exception_parked": "execution exception is parked for manual review",
        "live_reference_fallback": "using fallback market prices instead of auction prices",
        "missing_broker_response_ack": "broker did not acknowledge all submitted orders",
        "missing_live_price": "missing live reference price for at least one asset",
        "no_due_work": "no due work right now",
        "non_positive_net_liq": "broker net liquidation value is non-positive",
        "not_month_end_session": "waiting for the last session of the month",
        "position_reconciliation_warning": "model and broker positions differ before sizing",
        "ready_to_build_decision_plan": "ready to build the next decision plan",
        "ready_to_build_vplan": "ready to build the execution plan",
        "ready_to_reconcile": "ready to reconcile post-execution broker state",
        "session_mode_mismatch": "broker session mode does not match the release mode",
        "snapshot_not_ready_for_session": "snapshot is not ready for this session yet",
        "snapshot_ready": "snapshot is ready for this cycle",
        "snapshot_window_expired": "snapshot submit window already passed for this session",
        "submission_claim_failed": "submit claim failed because the VPlan is no longer ready",
        "submission_window_expired": "submit window already passed for this session",
        "submitted": "orders submitted to the broker",
        "unresolved_execution": "submitted execution is not reconciled yet",
        "vplan_ready": "execution plan is ready to submit",
        "waiting_for_eod_snapshot": "waiting for the EOD broker snapshot window",
        "waiting_for_post_execution_reconcile": "waiting for the reconcile grace window",
        "waiting_for_submission_window": "waiting for the submission window to open",
    }
    normalized_reason_code_str = str(reason_code_str or "").strip()
    if normalized_reason_code_str in reason_map_dict:
        return reason_map_dict[normalized_reason_code_str]
    if normalized_reason_code_str == "":
        return "unknown"
    return normalized_reason_code_str.replace("_", " ")


def _build_operator_message_spec_dict(
    *,
    level_str: str,
    phase_action_str: str,
    timestamp_obj: datetime | str | None,
    field_map_dict: dict[str, object],
) -> dict[str, object]:
    return {
        "level_str": str(level_str),
        "phase_action_str": str(phase_action_str),
        "timestamp_obj": timestamp_obj,
        "field_map_dict": dict(field_map_dict),
    }


def _emit_operator_message_spec_list(
    operator_message_spec_dict_list: list[dict[str, object]],
    *,
    log_path_str: str,
    print_message_bool: bool,
) -> None:
    for operator_message_spec_dict in operator_message_spec_dict_list:
        log_operator_message(
            level_str=str(operator_message_spec_dict["level_str"]),
            phase_action_str=str(operator_message_spec_dict["phase_action_str"]),
            timestamp_obj=operator_message_spec_dict["timestamp_obj"],
            field_map_dict=dict(operator_message_spec_dict["field_map_dict"]),
            audit_log_path_str=log_path_str,
            print_message_bool=print_message_bool,
        )


def _build_enabled_release_by_pod_id_map(
    state_store_obj: LiveStateStore,
) -> dict[str, LiveRelease]:
    return {
        str(release_obj.pod_id_str): release_obj
        for release_obj in state_store_obj.get_enabled_release_list()
    }


def _build_wait_operator_message_spec_list(
    scheduler_decision_obj: SchedulerDecision,
    *,
    pod_status_dict_list: list[dict[str, object]],
    execution_report_dict_list: list[dict[str, object]],
    broker_order_snapshot_dict_list: list[dict[str, object]],
) -> list[dict[str, object]]:
    next_due_timestamp_str = _format_operator_field_timestamp_str(
        scheduler_decision_obj.next_due_timestamp_ts
    )
    if len(pod_status_dict_list) == 0:
        return [
            _build_operator_message_spec_dict(
                level_str="INFO",
                phase_action_str="service.wait",
                timestamp_obj=scheduler_decision_obj.as_of_timestamp_ts,
                field_map_dict={
                    "reason": _humanize_reason_code_str(scheduler_decision_obj.reason_code_str),
                    "next": next_due_timestamp_str,
                },
            )
        ]
    execution_report_by_pod_id_map_dict = {
        str(execution_report_dict["pod_id_str"]): dict(execution_report_dict)
        for execution_report_dict in execution_report_dict_list
    }
    broker_order_snapshot_by_pod_id_map_dict = {
        str(broker_order_snapshot_dict["pod_id_str"]): dict(broker_order_snapshot_dict)
        for broker_order_snapshot_dict in broker_order_snapshot_dict_list
    }
    operator_message_spec_dict_list: list[dict[str, object]] = []
    for pod_status_dict in pod_status_dict_list:
        base_field_map_dict = {
            "pod": str(pod_status_dict["pod_id_str"]),
            "account": str(pod_status_dict["account_route_str"]),
        }
        broker_order_snapshot_dict = broker_order_snapshot_by_pod_id_map_dict.get(
            str(pod_status_dict["pod_id_str"]),
            {},
        )
        execution_report_dict = execution_report_by_pod_id_map_dict.get(
            str(pod_status_dict["pod_id_str"]),
            {},
        )
        broker_order_row_dict_list = list(broker_order_snapshot_dict.get("broker_order_row_dict_list", []))
        broker_ack_row_dict_list = list(pod_status_dict.get("broker_ack_row_dict_list", []))
        submit_progress_summary_dict = _build_submit_progress_summary_dict(
            broker_order_row_dict_list=broker_order_row_dict_list,
            broker_ack_row_dict_list=broker_ack_row_dict_list,
        )
        latest_vplan_id_int = broker_order_snapshot_dict.get("latest_vplan_id_int")
        if latest_vplan_id_int is None and execution_report_dict.get("latest_vplan_id_int") is not None:
            latest_vplan_id_int = int(execution_report_dict["latest_vplan_id_int"])
        active_reconcile_wait_bool = bool(scheduler_decision_obj.active_poll_bool) and (
            str(pod_status_dict.get("next_action_str")) == "post_execution_reconcile"
            or str(pod_status_dict.get("latest_vplan_status_str")) in ("submitted", "submitting")
        )
        if active_reconcile_wait_bool:
            fill_progress_operator_message_spec_dict = _build_fill_progress_operator_message_spec_dict(
                timestamp_obj=scheduler_decision_obj.as_of_timestamp_ts,
                base_field_map_dict=base_field_map_dict,
                vplan_id_int=None if latest_vplan_id_int is None else int(latest_vplan_id_int),
                submit_progress_summary_dict=submit_progress_summary_dict,
            )
            if fill_progress_operator_message_spec_dict is not None:
                operator_message_spec_dict_list.append(fill_progress_operator_message_spec_dict)
            latest_submission_timestamp_str = pod_status_dict.get("latest_submission_timestamp_str")
            elapsed_seconds_int = None
            if latest_submission_timestamp_str is not None:
                latest_submission_timestamp_ts = datetime.fromisoformat(str(latest_submission_timestamp_str))
                if latest_submission_timestamp_ts.tzinfo is None:
                    latest_submission_timestamp_ts = latest_submission_timestamp_ts.replace(tzinfo=UTC)
                elapsed_seconds_int = max(
                    0,
                    int(
                        (
                            scheduler_decision_obj.as_of_timestamp_ts
                            - latest_submission_timestamp_ts.astimezone(UTC)
                        ).total_seconds()
                    ),
                )
            operator_message_spec_dict_list.append(
                _build_operator_message_spec_dict(
                    level_str="INFO",
                    phase_action_str="reconcile.wait",
                    timestamp_obj=scheduler_decision_obj.as_of_timestamp_ts,
                    field_map_dict={
                        **base_field_map_dict,
                        "vplan": None if latest_vplan_id_int is None else int(latest_vplan_id_int),
                        "elapsed": None if elapsed_seconds_int is None else f"{int(elapsed_seconds_int)}s",
                        "acked": (
                            _format_ratio_str(
                                int(submit_progress_summary_dict["broker_ack_count_int"]),
                                int(submit_progress_summary_dict["ack_request_count_int"]),
                            )
                            if int(submit_progress_summary_dict["ack_request_count_int"]) > 0
                            else None
                        ),
                        "fills": (
                            _format_ratio_str(
                                int(submit_progress_summary_dict["filled_order_count_int"]),
                                int(submit_progress_summary_dict["order_count_int"]),
                            )
                            if int(submit_progress_summary_dict["order_count_int"]) > 0
                            else None
                        ),
                        "partial": (
                            int(submit_progress_summary_dict["partial_fill_order_count_int"])
                            if int(submit_progress_summary_dict["partial_fill_order_count_int"]) > 0
                            else None
                        ),
                        "fill_events": (
                            int(execution_report_dict["fill_count_int"])
                            if execution_report_dict.get("fill_count_int") is not None
                            else None
                        ),
                        "residuals": int(pod_status_dict.get("exception_count_int", 0)),
                        "next": next_due_timestamp_str,
                    },
                )
            )
            continue
        operator_message_spec_dict_list.append(
            _build_operator_message_spec_dict(
                level_str="INFO",
                phase_action_str="cycle.wait",
                timestamp_obj=scheduler_decision_obj.as_of_timestamp_ts,
                field_map_dict={
                    **base_field_map_dict,
                    "decision": str(pod_status_dict["latest_decision_plan_status_str"] or "none"),
                    "vplan": str(pod_status_dict["latest_vplan_status_str"] or "none"),
                    "reason": _humanize_reason_code_str(str(pod_status_dict["reason_code_str"])),
                    "next": next_due_timestamp_str,
                },
            )
        )
    return operator_message_spec_dict_list


def _build_phase_start_operator_message_spec_list(
    *,
    state_store_obj: LiveStateStore,
    scheduler_decision_obj: SchedulerDecision,
    broker_adapter_resolver_obj: runner.BrokerAdapterResolver,
) -> list[dict[str, object]]:
    release_by_pod_id_map_dict = _build_enabled_release_by_pod_id_map(state_store_obj)
    operator_message_spec_dict_list: list[dict[str, object]] = []
    for pod_id_str in scheduler_decision_obj.related_pod_id_list:
        release_obj = release_by_pod_id_map_dict.get(str(pod_id_str))
        if release_obj is None:
            continue
        base_field_map_dict = {
            "pod": str(release_obj.pod_id_str),
            "account": str(release_obj.account_route_str),
        }
        latest_vplan_obj = state_store_obj.get_latest_vplan_for_pod(str(pod_id_str))
        if scheduler_decision_obj.next_phase_str == "expire_stale":
            operator_message_spec_dict_list.append(
                _build_operator_message_spec_dict(
                    level_str="WARN",
                    phase_action_str="cycle.skip",
                    timestamp_obj=scheduler_decision_obj.as_of_timestamp_ts,
                    field_map_dict={
                        **base_field_map_dict,
                        "reason": _humanize_reason_code_str(scheduler_decision_obj.reason_code_str),
                    },
                )
            )
            continue
        if scheduler_decision_obj.next_phase_str in (
            "build_vplan",
            "submit_vplan",
            "eod_snapshot",
            "post_execution_reconcile",
        ):
            broker_connection_field_map_dict = broker_adapter_resolver_obj.get_connection_field_map_dict(
                release_obj
            )
            operator_message_spec_dict_list.append(
                _build_operator_message_spec_dict(
                    level_str="INFO",
                    phase_action_str="broker_connect.start",
                    timestamp_obj=scheduler_decision_obj.as_of_timestamp_ts,
                    field_map_dict={
                        **base_field_map_dict,
                        "broker": (
                            f"{broker_connection_field_map_dict['broker_host_str']}:"
                            f"{int(broker_connection_field_map_dict['broker_port_int'])}"
                        ),
                        "client_id": int(broker_connection_field_map_dict["broker_client_id_int"]),
                    },
                )
            )
        operator_message_spec_dict_list.append(
            _build_operator_message_spec_dict(
                level_str="INFO",
                phase_action_str=f"{scheduler_decision_obj.next_phase_str}.start",
                timestamp_obj=scheduler_decision_obj.as_of_timestamp_ts,
                field_map_dict={
                    **base_field_map_dict,
                    "vplan": (
                        None
                        if latest_vplan_obj is None or latest_vplan_obj.vplan_id_int is None
                        else int(latest_vplan_obj.vplan_id_int)
                    ),
                    "orders": (
                        _count_actionable_vplan_order_count_int(latest_vplan_obj)
                        if scheduler_decision_obj.next_phase_str == "submit_vplan"
                        else None
                    ),
                    "gross_order_notional": (
                        _format_float_str(_compute_gross_order_notional_float(latest_vplan_obj))
                        if scheduler_decision_obj.next_phase_str == "submit_vplan"
                        and latest_vplan_obj is not None
                        else None
                    ),
                    "reason": _humanize_reason_code_str(scheduler_decision_obj.reason_code_str),
                },
            )
        )
    return operator_message_spec_dict_list


def _build_phase_result_operator_message_spec_list(
    *,
    state_store_obj: LiveStateStore,
    scheduler_decision_obj: SchedulerDecision,
    tick_detail_dict: dict[str, object],
    pod_status_dict_list: list[dict[str, object]],
    execution_report_dict_list: list[dict[str, object]],
    broker_adapter_resolver_obj: runner.BrokerAdapterResolver,
) -> list[dict[str, object]]:
    release_by_pod_id_map_dict = _build_enabled_release_by_pod_id_map(state_store_obj)
    pod_status_by_pod_id_map_dict = {
        str(pod_status_dict["pod_id_str"]): dict(pod_status_dict) for pod_status_dict in pod_status_dict_list
    }
    execution_report_by_pod_id_map_dict = {
        str(execution_report_dict["pod_id_str"]): dict(execution_report_dict)
        for execution_report_dict in execution_report_dict_list
    }
    warning_count_map_dict = dict(tick_detail_dict.get("warning_count_map_dict", {}))
    reason_count_map_dict = dict(tick_detail_dict.get("reason_count_map_dict", {}))
    first_reason_code_str = (
        sorted(reason_count_map_dict.keys())[0] if len(reason_count_map_dict) > 0 else "unknown"
    )
    operator_message_spec_dict_list: list[dict[str, object]] = []

    for pod_id_str in scheduler_decision_obj.related_pod_id_list:
        release_obj = release_by_pod_id_map_dict.get(str(pod_id_str))
        if release_obj is None:
            continue
        base_field_map_dict = {
            "pod": str(release_obj.pod_id_str),
            "account": str(release_obj.account_route_str),
        }
        latest_decision_plan_obj = state_store_obj.get_latest_decision_plan_for_pod(str(pod_id_str))
        latest_vplan_obj = state_store_obj.get_latest_vplan_for_pod(str(pod_id_str))
        pod_status_dict = pod_status_by_pod_id_map_dict.get(str(pod_id_str), {})
        execution_report_dict = execution_report_by_pod_id_map_dict.get(str(pod_id_str), {})

        if scheduler_decision_obj.next_phase_str in (
            "build_vplan",
            "submit_vplan",
            "post_execution_reconcile",
        ):
            broker_connection_field_map_dict = broker_adapter_resolver_obj.get_connection_field_map_dict(
                release_obj
            )
            operator_message_spec_dict_list.append(
                _build_operator_message_spec_dict(
                    level_str="INFO",
                    phase_action_str="broker_connect.ok",
                    timestamp_obj=scheduler_decision_obj.as_of_timestamp_ts,
                    field_map_dict={
                        **base_field_map_dict,
                        "broker": (
                            f"{broker_connection_field_map_dict['broker_host_str']}:"
                            f"{int(broker_connection_field_map_dict['broker_port_int'])}"
                        ),
                        "client_id": int(broker_connection_field_map_dict["broker_client_id_int"]),
                    },
                )
            )

        if scheduler_decision_obj.next_phase_str == "eod_snapshot":
            if int(tick_detail_dict.get("eod_snapshot_count_int", 0)) > 0:
                operator_message_spec_dict_list.append(
                    _build_operator_message_spec_dict(
                        level_str="INFO",
                        phase_action_str="eod_snapshot.ok",
                        timestamp_obj=scheduler_decision_obj.as_of_timestamp_ts,
                        field_map_dict={
                            **base_field_map_dict,
                            "snapshots": int(tick_detail_dict.get("eod_snapshot_count_int", 0)),
                        },
                    )
                )
            elif int(tick_detail_dict.get("blocked_action_count_int", 0)) > 0:
                operator_message_spec_dict_list.append(
                    _build_operator_message_spec_dict(
                        level_str="ERROR",
                        phase_action_str="eod_snapshot.fail",
                        timestamp_obj=scheduler_decision_obj.as_of_timestamp_ts,
                        field_map_dict={
                            **base_field_map_dict,
                            "reason": _humanize_reason_code_str(first_reason_code_str),
                        },
                    )
                )
            continue

        if scheduler_decision_obj.next_phase_str == "build_decision_plan":
            if int(tick_detail_dict.get("created_decision_plan_count_int", 0)) > 0 and latest_decision_plan_obj is not None:
                operator_message_spec_dict_list.append(
                    _build_operator_message_spec_dict(
                        level_str="INFO",
                        phase_action_str="cycle.start",
                        timestamp_obj=scheduler_decision_obj.as_of_timestamp_ts,
                        field_map_dict={
                            **base_field_map_dict,
                            "signal": _format_operator_field_timestamp_str(latest_decision_plan_obj.signal_timestamp_ts),
                            "submit": _format_operator_field_timestamp_str(latest_decision_plan_obj.submission_timestamp_ts),
                            "target": _format_operator_field_timestamp_str(latest_decision_plan_obj.target_execution_timestamp_ts),
                        },
                    )
                )
                operator_message_spec_dict_list.append(
                    _build_operator_message_spec_dict(
                        level_str="INFO",
                        phase_action_str="build_decision_plan.ok",
                        timestamp_obj=scheduler_decision_obj.as_of_timestamp_ts,
                        field_map_dict={
                            **base_field_map_dict,
                            "plan_id": int(latest_decision_plan_obj.decision_plan_id_int or 0),
                        },
                    )
                )
            elif int(tick_detail_dict.get("skipped_decision_plan_count_int", 0)) > 0:
                operator_message_spec_dict_list.append(
                    _build_operator_message_spec_dict(
                        level_str="WARN",
                        phase_action_str="build_decision_plan.skip",
                        timestamp_obj=scheduler_decision_obj.as_of_timestamp_ts,
                        field_map_dict={
                            **base_field_map_dict,
                            "reason": _humanize_reason_code_str(first_reason_code_str),
                        },
                    )
                )
            continue

        if scheduler_decision_obj.next_phase_str == "build_vplan":
            if int(tick_detail_dict.get("created_vplan_count_int", 0)) > 0 and latest_vplan_obj is not None:
                actionable_order_count_int = _count_actionable_vplan_order_count_int(latest_vplan_obj)
                fallback_asset_count_int = sum(
                    1
                    for vplan_row_obj in latest_vplan_obj.vplan_row_list
                    if "fallback" in str(vplan_row_obj.live_reference_source_str)
                )
                if int(warning_count_map_dict.get("position_reconciliation_warning", 0)) > 0:
                    operator_message_spec_dict_list.append(
                        _build_operator_message_spec_dict(
                            level_str="WARN",
                            phase_action_str="build_vplan.warn",
                            timestamp_obj=scheduler_decision_obj.as_of_timestamp_ts,
                            field_map_dict={
                                **base_field_map_dict,
                                "reason": _humanize_reason_code_str("position_reconciliation_warning"),
                            },
                        )
                    )
                if fallback_asset_count_int > 0:
                    operator_message_spec_dict_list.append(
                        _build_operator_message_spec_dict(
                            level_str="WARN",
                            phase_action_str="build_vplan.warn",
                            timestamp_obj=scheduler_decision_obj.as_of_timestamp_ts,
                            field_map_dict={
                                **base_field_map_dict,
                                "reason": _humanize_reason_code_str("live_reference_fallback"),
                                "fallback_assets": int(fallback_asset_count_int),
                            },
                        )
                    )
                operator_message_spec_dict_list.append(
                    _build_operator_message_spec_dict(
                        level_str="INFO",
                        phase_action_str="build_vplan.ok",
                        timestamp_obj=scheduler_decision_obj.as_of_timestamp_ts,
                        field_map_dict={
                            **base_field_map_dict,
                            "plan_id": int(latest_vplan_obj.decision_plan_id_int),
                            "vplan": int(latest_vplan_obj.vplan_id_int or 0),
                            "asset_count": len(latest_vplan_obj.vplan_row_list),
                            "budget": _format_float_str(latest_vplan_obj.pod_budget_float),
                        },
                    )
                )
                operator_message_spec_dict_list.append(
                    _build_operator_message_spec_dict(
                        level_str="INFO",
                        phase_action_str="order_plan.ok",
                        timestamp_obj=scheduler_decision_obj.as_of_timestamp_ts,
                        field_map_dict={
                            **base_field_map_dict,
                            "vplan": int(latest_vplan_obj.vplan_id_int or 0),
                            "orders": int(actionable_order_count_int),
                            "gross_order_notional": _format_float_str(
                                _compute_gross_order_notional_float(latest_vplan_obj)
                            ),
                        },
                    )
                )
            elif int(tick_detail_dict.get("blocked_action_count_int", 0)) > 0:
                operator_message_spec_dict_list.append(
                    _build_operator_message_spec_dict(
                        level_str="ERROR",
                        phase_action_str="build_vplan.fail",
                        timestamp_obj=scheduler_decision_obj.as_of_timestamp_ts,
                        field_map_dict={
                            **base_field_map_dict,
                            "reason": _humanize_reason_code_str(first_reason_code_str),
                        },
                    )
                )
            continue

        if scheduler_decision_obj.next_phase_str == "submit_vplan":
            if int(tick_detail_dict.get("submitted_vplan_count_int", 0)) > 0 and latest_vplan_obj is not None:
                broker_order_row_dict_list = state_store_obj.get_broker_order_row_dict_list_for_vplan(
                    int(latest_vplan_obj.vplan_id_int or 0)
                )
                broker_ack_row_dict_list = state_store_obj.get_broker_ack_row_dict_list_for_vplan(
                    int(latest_vplan_obj.vplan_id_int or 0)
                )
                submit_progress_summary_dict = _build_submit_progress_summary_dict(
                    broker_order_row_dict_list=broker_order_row_dict_list,
                    broker_ack_row_dict_list=broker_ack_row_dict_list,
                )
                operator_message_spec_dict_list.append(
                    _build_operator_message_spec_dict(
                        level_str="INFO",
                        phase_action_str="submit_vplan.ok",
                        timestamp_obj=scheduler_decision_obj.as_of_timestamp_ts,
                        field_map_dict={
                            **base_field_map_dict,
                            "vplan": int(latest_vplan_obj.vplan_id_int or 0),
                            "orders": int(submit_progress_summary_dict["order_count_int"]),
                        },
                    )
                )
                submit_ack_operator_message_spec_dict = _build_submit_ack_operator_message_spec_dict(
                    timestamp_obj=scheduler_decision_obj.as_of_timestamp_ts,
                    base_field_map_dict=base_field_map_dict,
                    vplan_id_int=int(latest_vplan_obj.vplan_id_int or 0),
                    submit_ack_status_str=latest_vplan_obj.submit_ack_status_str,
                    missing_ack_count_int=int(latest_vplan_obj.missing_ack_count_int),
                    submit_progress_summary_dict=submit_progress_summary_dict,
                )
                if submit_ack_operator_message_spec_dict is not None:
                    operator_message_spec_dict_list.append(submit_ack_operator_message_spec_dict)
                fill_progress_operator_message_spec_dict = _build_fill_progress_operator_message_spec_dict(
                    timestamp_obj=scheduler_decision_obj.as_of_timestamp_ts,
                    base_field_map_dict=base_field_map_dict,
                    vplan_id_int=int(latest_vplan_obj.vplan_id_int or 0),
                    submit_progress_summary_dict=submit_progress_summary_dict,
                )
                if fill_progress_operator_message_spec_dict is not None:
                    operator_message_spec_dict_list.append(fill_progress_operator_message_spec_dict)
                if int(warning_count_map_dict.get("missing_broker_response_ack", 0)) > 0:
                    operator_message_spec_dict_list.append(
                        _build_operator_message_spec_dict(
                            level_str="CRITICAL",
                            phase_action_str="submit_vplan.warn",
                            timestamp_obj=scheduler_decision_obj.as_of_timestamp_ts,
                            field_map_dict={
                                **base_field_map_dict,
                                "vplan": int(latest_vplan_obj.vplan_id_int or 0),
                                "reason": _humanize_reason_code_str("missing_broker_response_ack"),
                                "missing_ack": int(latest_vplan_obj.missing_ack_count_int),
                            },
                        )
                    )
            elif int(tick_detail_dict.get("blocked_action_count_int", 0)) > 0:
                phase_action_str = "submit_vplan.skip"
                if first_reason_code_str not in ("duplicate_submission_guard", "submission_claim_failed"):
                    phase_action_str = "submit_vplan.fail"
                operator_message_spec_dict_list.append(
                    _build_operator_message_spec_dict(
                        level_str="ERROR" if phase_action_str.endswith(".fail") else "WARN",
                        phase_action_str=phase_action_str,
                        timestamp_obj=scheduler_decision_obj.as_of_timestamp_ts,
                        field_map_dict={
                            **base_field_map_dict,
                            "reason": _humanize_reason_code_str(first_reason_code_str),
                        },
                    )
                )
            continue

        if scheduler_decision_obj.next_phase_str == "post_execution_reconcile":
            if int(tick_detail_dict.get("completed_vplan_count_int", 0)) > 0 and latest_vplan_obj is not None:
                fill_count_int = int(execution_report_dict.get("fill_count_int", 0))
                residual_count_int = int(pod_status_dict.get("exception_count_int", 0))
                operator_message_spec_dict_list.append(
                    _build_operator_message_spec_dict(
                        level_str="INFO",
                        phase_action_str="reconcile.ok",
                        timestamp_obj=scheduler_decision_obj.as_of_timestamp_ts,
                        field_map_dict={
                            **base_field_map_dict,
                            "vplan": int(latest_vplan_obj.vplan_id_int or 0),
                            "fills": int(fill_count_int),
                            "residuals": int(residual_count_int),
                        },
                    )
                )
                operator_message_spec_dict_list.append(
                    _build_operator_message_spec_dict(
                        level_str="INFO",
                        phase_action_str="cycle.done",
                        timestamp_obj=scheduler_decision_obj.as_of_timestamp_ts,
                        field_map_dict={
                            **base_field_map_dict,
                            "fills": int(fill_count_int),
                            "residuals": int(residual_count_int),
                            "status": str(latest_vplan_obj.status_str),
                        },
                    )
                )
            elif str(pod_status_dict.get("reason_code_str")) == "execution_exception_parked":
                operator_message_spec_dict_list.append(
                    _build_operator_message_spec_dict(
                        level_str="CRITICAL",
                        phase_action_str="reconcile.fail",
                        timestamp_obj=scheduler_decision_obj.as_of_timestamp_ts,
                        field_map_dict={
                            **base_field_map_dict,
                            "reason": _humanize_reason_code_str("execution_exception_parked"),
                            "residuals": int(pod_status_dict.get("exception_count_int", 0)),
                        },
                    )
                )
            continue

    return operator_message_spec_dict_list


def _build_phase_failure_operator_message_spec_list(
    *,
    state_store_obj: LiveStateStore,
    scheduler_decision_obj: SchedulerDecision,
    as_of_ts: datetime,
    error_str: str,
    error_retry_seconds_int: int,
    broker_adapter_resolver_obj: runner.BrokerAdapterResolver,
) -> list[dict[str, object]]:
    release_by_pod_id_map_dict = _build_enabled_release_by_pod_id_map(state_store_obj)
    operator_message_spec_dict_list: list[dict[str, object]] = []
    broker_connection_failure_bool = "refused the network connection" in str(error_str).lower()
    for pod_id_str in scheduler_decision_obj.related_pod_id_list:
        release_obj = release_by_pod_id_map_dict.get(str(pod_id_str))
        if release_obj is None:
            continue
        base_field_map_dict = {
            "pod": str(release_obj.pod_id_str),
            "account": str(release_obj.account_route_str),
            "reason": str(error_str),
            "retry_in": f"{int(error_retry_seconds_int)}s",
        }
        if broker_connection_failure_bool and scheduler_decision_obj.next_phase_str in (
            "build_vplan",
            "submit_vplan",
            "eod_snapshot",
            "post_execution_reconcile",
        ):
            broker_connection_field_map_dict = broker_adapter_resolver_obj.get_connection_field_map_dict(
                release_obj
            )
            operator_message_spec_dict_list.append(
                _build_operator_message_spec_dict(
                    level_str="ERROR",
                    phase_action_str="broker_connect.fail",
                    timestamp_obj=as_of_ts,
                    field_map_dict={
                        **base_field_map_dict,
                        "broker": (
                            f"{broker_connection_field_map_dict['broker_host_str']}:"
                            f"{int(broker_connection_field_map_dict['broker_port_int'])}"
                        ),
                        "client_id": int(broker_connection_field_map_dict["broker_client_id_int"]),
                    },
                )
            )
        operator_message_spec_dict_list.append(
            _build_operator_message_spec_dict(
                level_str="ERROR",
                phase_action_str=f"{scheduler_decision_obj.next_phase_str}.fail",
                timestamp_obj=as_of_ts,
                field_map_dict=base_field_map_dict,
            )
        )
        operator_message_spec_dict_list.append(
            _build_operator_message_spec_dict(
                level_str="ERROR",
                phase_action_str="cycle.fail",
                timestamp_obj=as_of_ts,
                field_map_dict=base_field_map_dict,
            )
        )
    return operator_message_spec_dict_list


def _build_stuck_operator_message_spec_list(
    *,
    state_store_obj: LiveStateStore,
    as_of_ts: datetime,
    reconcile_grace_seconds_int: int,
) -> list[dict[str, object]]:
    operator_message_spec_dict_list: list[dict[str, object]] = []
    for release_obj in state_store_obj.get_enabled_release_list():
        latest_vplan_obj = state_store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str)
        if latest_vplan_obj is None or latest_vplan_obj.vplan_id_int is None:
            continue
        if (
            latest_vplan_obj.status_str == "submitting"
            and state_store_obj.count_broker_orders_for_vplan(int(latest_vplan_obj.vplan_id_int)) == 0
            and (
                as_of_ts - latest_vplan_obj.submission_timestamp_ts.astimezone(UTC)
            ).total_seconds()
            >= DEFAULT_SUBMIT_STUCK_SECONDS_INT
        ):
            operator_message_spec_dict_list.append(
                _build_operator_message_spec_dict(
                    level_str="CRITICAL",
                    phase_action_str="submit_vplan.stuck",
                    timestamp_obj=as_of_ts,
                    field_map_dict={
                        "pod": str(release_obj.pod_id_str),
                        "account": str(release_obj.account_route_str),
                        "vplan": int(latest_vplan_obj.vplan_id_int),
                        "reason": "no broker orders recorded after submit claim",
                    },
                )
            )
        if (
            latest_vplan_obj.status_str == "submitted"
            and not state_store_obj.has_post_execution_reconciliation_snapshot(int(latest_vplan_obj.vplan_id_int))
            and (
                as_of_ts - latest_vplan_obj.target_execution_timestamp_ts.astimezone(UTC)
            ).total_seconds()
            >= float(reconcile_grace_seconds_int)
        ):
            operator_message_spec_dict_list.append(
                _build_operator_message_spec_dict(
                    level_str="CRITICAL",
                    phase_action_str="reconcile.stuck",
                    timestamp_obj=as_of_ts,
                    field_map_dict={
                        "pod": str(release_obj.pod_id_str),
                        "account": str(release_obj.account_route_str),
                        "vplan": int(latest_vplan_obj.vplan_id_int),
                        "reason": "no post-execution reconcile snapshot recorded",
                    },
                )
            )
    return operator_message_spec_dict_list


def _should_emit_wait_operator_message(
    *,
    last_printed_signature_tup: tuple[object, ...] | None,
    current_signature_tup: tuple[object, ...],
    last_printed_timestamp_ts: datetime | None,
    as_of_ts: datetime,
    heartbeat_seconds_int: int = DEFAULT_OPERATOR_HEARTBEAT_SECONDS_INT,
) -> bool:
    if last_printed_signature_tup != current_signature_tup:
        return True
    if last_printed_timestamp_ts is None:
        return True
    return (as_of_ts - last_printed_timestamp_ts).total_seconds() >= float(heartbeat_seconds_int)


def _emit_scheduler_event(
    event_name_str: str,
    event_payload_dict: dict[str, object],
    log_path_str: str,
    print_events_bool: bool,
) -> None:
    log_event(
        event_name_str,
        event_payload_dict,
        log_path_str=log_path_str,
    )
    if print_events_bool:
        print(
            _render_scheduler_event_output_str(
                event_name_str=event_name_str,
                event_payload_dict=event_payload_dict,
            ),
            flush=True,
        )


def _format_float_str(value_obj: object, decimals_int: int = 2) -> str:
    if value_obj is None:
        return "none"
    return f"{float(value_obj):.{decimals_int}f}"


def _format_signed_float_str(value_obj: object, decimals_int: int = 2) -> str:
    if value_obj is None:
        return "none"
    return f"{float(value_obj):+.{decimals_int}f}"


def _format_count_map_str(count_map_dict: dict[str, object]) -> str:
    if len(count_map_dict) == 0:
        return "none"
    return ", ".join(
        f"{key_str}={int(count_map_dict[key_str])}"
        for key_str in sorted(count_map_dict)
    )


def _render_pod_status_line_list(pod_status_dict_list: list[dict[str, object]]) -> list[str]:
    line_list: list[str] = []
    for pod_status_dict in pod_status_dict_list:
        line_list.append(
            "  pod "
            f"{pod_status_dict['pod_id_str']}: "
            f"decision={pod_status_dict['latest_decision_plan_status_str'] or 'none'} "
            f"vplan={pod_status_dict['latest_vplan_status_str'] or 'none'} "
            f"ack={pod_status_dict.get('submit_ack_status_str') or 'none'} "
            f"ack_cov={_format_float_str(pod_status_dict.get('ack_coverage_ratio_float'))} "
            f"ack_missing={int(pod_status_dict.get('missing_ack_count_int', 0))} "
            f"next={pod_status_dict['next_action_str']} "
            f"reason={pod_status_dict['reason_code_str']} "
            f"latest_fill={pod_status_dict['latest_fill_timestamp_str'] or 'none'}"
        )
    return line_list


def _render_vplan_line_list(vplan_dict_list: list[dict[str, object]]) -> list[str]:
    line_list: list[str] = []
    for vplan_dict in vplan_dict_list:
        warning_row_count_int = len(vplan_dict.get("warning_row_dict_list", []))
        line_list.append(
            "  vplan "
            f"pod={vplan_dict['pod_id_str']} "
            f"id={vplan_dict['vplan_id_int']} "
            f"status={vplan_dict['status_str']} "
            f"net_liq={_format_float_str(vplan_dict['net_liq_float'])} "
            f"budget={_format_float_str(vplan_dict['pod_budget_float'])} "
            f"price_source={vplan_dict['live_price_source_str']} "
            f"warnings={warning_row_count_int}"
        )
        for vplan_row_dict in vplan_dict.get("vplan_row_dict_list", []):
            line_list.append(
                "    row "
                f"{vplan_row_dict['asset_str']}: "
                f"current={_format_float_str(vplan_row_dict['current_share_float'])} "
                f"target={_format_float_str(vplan_row_dict['target_share_float'])} "
                f"delta={_format_signed_float_str(vplan_row_dict['order_delta_share_float'])} "
                f"px={_format_float_str(vplan_row_dict['live_reference_price_float'])} "
                f"source={vplan_row_dict.get('live_reference_source_str', 'unknown')} "
                f"notional={_format_float_str(vplan_row_dict['estimated_target_notional_float'])} "
                f"warn={'yes' if bool(vplan_row_dict['warning_bool']) else 'no'}"
            )
    return line_list


def _render_broker_order_line_list(
    broker_order_snapshot_dict_list: list[dict[str, object]],
) -> list[str]:
    line_list: list[str] = []
    for broker_order_snapshot_dict in broker_order_snapshot_dict_list:
        line_list.append(
            "  orders "
            f"pod={broker_order_snapshot_dict['pod_id_str']} "
            f"vplan={broker_order_snapshot_dict['latest_vplan_id_int']} "
            f"ack={broker_order_snapshot_dict.get('submit_ack_status_str', 'none')} "
            f"ack_cov={_format_float_str(broker_order_snapshot_dict.get('ack_coverage_ratio_float'))} "
            f"ack_missing={int(broker_order_snapshot_dict.get('missing_ack_count_int', 0))} "
            f"count={broker_order_snapshot_dict['broker_order_count_int']}"
        )
        for broker_order_row_dict in broker_order_snapshot_dict.get("broker_order_row_dict_list", []):
            line_list.append(
                "    order "
                f"{broker_order_row_dict['asset_str']}: "
                f"type={broker_order_row_dict['broker_order_type_str']} "
                f"qty={_format_signed_float_str(broker_order_row_dict['amount_float'])} "
                f"filled={_format_float_str(broker_order_row_dict['filled_amount_float'])} "
                f"remaining={_format_float_str(broker_order_row_dict['remaining_amount_float'])} "
                f"status={broker_order_row_dict['status_str']} "
                f"last_status={broker_order_row_dict['last_status_timestamp_str'] or broker_order_row_dict['submitted_timestamp_str']}"
            )
    return line_list


def _render_fill_line_list(execution_report_dict_list: list[dict[str, object]]) -> list[str]:
    line_list: list[str] = []
    for execution_report_dict in execution_report_dict_list:
        line_list.append(
            "  fills "
            f"pod={execution_report_dict['pod_id_str']} "
            f"vplan={execution_report_dict['latest_vplan_id_int']} "
            f"count={execution_report_dict['fill_count_int']}"
        )
        for fill_row_dict in execution_report_dict.get("fill_row_dict_list", []):
            line_list.append(
                "    fill "
                f"{fill_row_dict['asset_str']}: "
                f"qty={_format_signed_float_str(fill_row_dict['fill_amount_float'])} "
                f"px={_format_float_str(fill_row_dict['fill_price_float'])} "
                f"open={_format_float_str(fill_row_dict['official_open_price_float'])} "
                f"open_source={fill_row_dict['open_price_source_str'] or 'none'} "
                f"ts={fill_row_dict['fill_timestamp_str']}"
            )
    return line_list


def _render_scheduler_event_output_str(
    event_name_str: str,
    event_payload_dict: dict[str, object],
) -> str:
    event_timestamp_str = str(
        event_payload_dict.get("as_of_timestamp_str")
        or event_payload_dict.get("scheduled_wake_timestamp_str")
        or "now"
    )
    header_fragment_list: list[str] = [f"[{event_timestamp_str}] {event_name_str}"]
    if "env_mode_str" in event_payload_dict:
        header_fragment_list.append(f"mode={event_payload_dict['env_mode_str']}")
    if "next_phase_str" in event_payload_dict:
        header_fragment_list.append(f"phase={event_payload_dict['next_phase_str']}")
    if "reason_code_str" in event_payload_dict:
        header_fragment_list.append(f"reason={event_payload_dict['reason_code_str']}")
    line_list = [" | ".join(header_fragment_list)]

    related_pod_id_list = list(event_payload_dict.get("related_pod_id_list", []))
    if len(related_pod_id_list) > 0:
        line_list.append(f"  pods={', '.join(related_pod_id_list)}")
    if "next_due_timestamp_str" in event_payload_dict:
        next_due_timestamp_str = str(event_payload_dict["next_due_timestamp_str"])
        if "sleep_seconds_float" in event_payload_dict:
            line_list.append(
                f"  next_due={next_due_timestamp_str} sleep={_format_float_str(event_payload_dict['sleep_seconds_float'])}s"
            )
        else:
            line_list.append(f"  next_due={next_due_timestamp_str}")
    if event_name_str == "scheduler_started":
        line_list.append(
            "  config "
            f"active_poll={int(event_payload_dict['active_poll_seconds_int'])}s "
            f"idle_max_sleep={int(event_payload_dict['idle_max_sleep_seconds_int'])}s "
            f"reconcile_grace={int(event_payload_dict['reconcile_grace_seconds_int'])}s"
        )
    tick_detail_dict = event_payload_dict.get("tick_detail_dict")
    if isinstance(tick_detail_dict, dict):
        line_list.append(
            "  tick "
            f"eod={int(tick_detail_dict.get('eod_snapshot_count_int', 0))} "
            f"created_decision={int(tick_detail_dict.get('created_decision_plan_count_int', 0))} "
            f"created_vplan={int(tick_detail_dict.get('created_vplan_count_int', 0))} "
            f"submitted={int(tick_detail_dict.get('submitted_vplan_count_int', 0))} "
            f"completed={int(tick_detail_dict.get('completed_vplan_count_int', 0))} "
            f"blocked={int(tick_detail_dict.get('blocked_action_count_int', 0))}"
        )
        warning_count_map_dict = dict(tick_detail_dict.get("warning_count_map_dict", {}))
        reason_count_map_dict = dict(tick_detail_dict.get("reason_count_map_dict", {}))
        line_list.append(f"  warnings={_format_count_map_str(warning_count_map_dict)}")
        line_list.append(f"  reasons={_format_count_map_str(reason_count_map_dict)}")
    pod_status_dict_list = event_payload_dict.get("post_tick_pod_status_dict_list")
    if isinstance(pod_status_dict_list, list):
        line_list.extend(_render_pod_status_line_list(pod_status_dict_list))
    pod_status_dict_list = event_payload_dict.get("pod_status_dict_list")
    if isinstance(pod_status_dict_list, list):
        line_list.extend(_render_pod_status_line_list(pod_status_dict_list))
    vplan_dict_list = event_payload_dict.get("post_tick_vplan_dict_list")
    if isinstance(vplan_dict_list, list):
        line_list.extend(_render_vplan_line_list(vplan_dict_list))
    broker_order_snapshot_dict_list = event_payload_dict.get("post_tick_broker_order_snapshot_dict_list")
    if isinstance(broker_order_snapshot_dict_list, list):
        line_list.extend(_render_broker_order_line_list(broker_order_snapshot_dict_list))
    broker_order_snapshot_dict_list = event_payload_dict.get("broker_order_snapshot_dict_list")
    if isinstance(broker_order_snapshot_dict_list, list):
        line_list.extend(_render_broker_order_line_list(broker_order_snapshot_dict_list))
    execution_report_dict_list = event_payload_dict.get("post_tick_execution_report_dict_list")
    if isinstance(execution_report_dict_list, list):
        line_list.extend(_render_fill_line_list(execution_report_dict_list))
    execution_report_dict_list = event_payload_dict.get("execution_report_dict_list")
    if isinstance(execution_report_dict_list, list):
        line_list.extend(_render_fill_line_list(execution_report_dict_list))
    if event_name_str == "scheduler_error_retry":
        line_list.append(
            f"  error={event_payload_dict['error_str']} retry={int(event_payload_dict['error_retry_seconds_int'])}s"
        )
    return "\n".join(line_list)


def _build_pod_status_signature_tup(
    pod_status_dict_list: list[dict[str, object]],
) -> tuple[tuple[str, str, str, str, str, str], ...]:
    return tuple(
        (
            str(pod_status_dict["pod_id_str"]),
            str(pod_status_dict.get("latest_decision_plan_status_str")),
            str(pod_status_dict.get("latest_vplan_status_str")),
            str(pod_status_dict.get("next_action_str")),
            str(pod_status_dict.get("reason_code_str")),
            str(pod_status_dict.get("latest_fill_timestamp_str")),
        )
        for pod_status_dict in pod_status_dict_list
    )


def _build_execution_report_signature_tup(
    execution_report_dict_list: list[dict[str, object]],
) -> tuple[tuple[str, int, int, str], ...]:
    signature_row_list: list[tuple[str, int, int, str]] = []
    for execution_report_dict in execution_report_dict_list:
        fill_row_dict_list = list(execution_report_dict.get("fill_row_dict_list", []))
        latest_fill_timestamp_str = (
            "none" if len(fill_row_dict_list) == 0 else str(fill_row_dict_list[-1]["fill_timestamp_str"])
        )
        signature_row_list.append(
            (
                str(execution_report_dict["pod_id_str"]),
                int(execution_report_dict["latest_vplan_id_int"]),
                int(execution_report_dict["fill_count_int"]),
                latest_fill_timestamp_str,
            )
        )
    return tuple(signature_row_list)


def _build_broker_order_signature_tup(
    broker_order_snapshot_dict_list: list[dict[str, object]],
) -> tuple[tuple[str, int, str, float, float, str, str], ...]:
    signature_row_list: list[tuple[str, int, str, float, float, str, str]] = []
    for broker_order_snapshot_dict in broker_order_snapshot_dict_list:
        latest_vplan_id_int = int(broker_order_snapshot_dict["latest_vplan_id_int"])
        for broker_order_row_dict in broker_order_snapshot_dict.get("broker_order_row_dict_list", []):
            signature_row_list.append(
                (
                    str(broker_order_snapshot_dict["pod_id_str"]),
                    latest_vplan_id_int,
                    str(broker_order_row_dict["asset_str"]),
                    float(broker_order_row_dict["amount_float"]),
                    float(broker_order_row_dict["filled_amount_float"]),
                    str(broker_order_row_dict["status_str"]),
                    str(
                        broker_order_row_dict.get("last_status_timestamp_str")
                        or broker_order_row_dict["submitted_timestamp_str"]
                    ),
                )
            )
    return tuple(signature_row_list)


def _build_scheduler_sleep_signature_tup(
    event_payload_dict: dict[str, object],
) -> tuple[object, ...]:
    pod_status_dict_list = list(event_payload_dict.get("pod_status_dict_list", []))
    execution_report_dict_list = list(event_payload_dict.get("execution_report_dict_list", []))
    broker_order_snapshot_dict_list = list(event_payload_dict.get("broker_order_snapshot_dict_list", []))
    return (
        str(event_payload_dict.get("env_mode_str")),
        str(event_payload_dict.get("next_phase_str")),
        str(event_payload_dict.get("reason_code_str")),
        tuple(str(pod_id_str) for pod_id_str in event_payload_dict.get("related_pod_id_list", [])),
        _build_pod_status_signature_tup(pod_status_dict_list),
        _build_execution_report_signature_tup(execution_report_dict_list),
        _build_broker_order_signature_tup(broker_order_snapshot_dict_list),
    )


def _render_scheduler_output_str(detail_dict: dict[str, object]) -> str:
    line_list = [
        "Scheduler",
        f"- Mode: {detail_dict['env_mode_str']}",
        f"- Now UTC: {detail_dict['as_of_timestamp_str']}",
        f"- Due now: {str(bool(detail_dict['due_now_bool'])).lower()}",
        f"- Active poll: {str(bool(detail_dict['active_poll_bool'])).lower()}",
        f"- Phase: {detail_dict['next_phase_str']}",
        f"- Reason: {detail_dict['reason_code_str']}",
        f"- Next due UTC: {detail_dict['next_due_timestamp_str']}",
        f"- Pods: {', '.join(detail_dict['related_pod_id_list']) if len(detail_dict['related_pod_id_list']) > 0 else 'none'}",
    ]
    if "tick_invoked_bool" in detail_dict:
        line_list.append(f"- Tick invoked: {str(bool(detail_dict['tick_invoked_bool'])).lower()}")
    if "tick_detail_dict" in detail_dict:
        tick_detail_dict = dict(detail_dict["tick_detail_dict"])
        line_list.append(
            "- Tick summary: "
            f"eod_snapshot_count_int={int(tick_detail_dict.get('eod_snapshot_count_int', 0))}, "
            f"created_decision_plan_count_int={int(tick_detail_dict.get('created_decision_plan_count_int', 0))}, "
            f"created_vplan_count_int={int(tick_detail_dict.get('created_vplan_count_int', 0))}, "
            f"submitted_vplan_count_int={int(tick_detail_dict.get('submitted_vplan_count_int', 0))}, "
            f"completed_vplan_count_int={int(tick_detail_dict.get('completed_vplan_count_int', 0))}"
        )
    if "post_tick_pod_status_dict_list" in detail_dict:
        for pod_status_dict in detail_dict["post_tick_pod_status_dict_list"]:
            line_list.append(
                "- Pod status: "
                f"pod_id_str={pod_status_dict['pod_id_str']}, "
                f"decision={pod_status_dict['latest_decision_plan_status_str']}, "
                f"vplan={pod_status_dict['latest_vplan_status_str']}, "
                f"next_action={pod_status_dict['next_action_str']}, "
                f"reason={pod_status_dict['reason_code_str']}, "
                f"latest_fill={pod_status_dict['latest_fill_timestamp_str']}"
            )
    if "post_tick_broker_order_snapshot_dict_list" in detail_dict:
        for broker_order_snapshot_dict in detail_dict["post_tick_broker_order_snapshot_dict_list"]:
            line_list.append(
                "- Broker orders: "
                f"pod_id_str={broker_order_snapshot_dict['pod_id_str']}, "
                f"latest_vplan_id_int={broker_order_snapshot_dict['latest_vplan_id_int']}, "
                f"broker_order_count_int={broker_order_snapshot_dict['broker_order_count_int']}"
            )
    if "post_tick_execution_report_dict_list" in detail_dict:
        for execution_report_dict in detail_dict["post_tick_execution_report_dict_list"]:
            line_list.append(
                "- Fills: "
                f"pod_id_str={execution_report_dict['pod_id_str']}, "
                f"latest_vplan_id_int={execution_report_dict['latest_vplan_id_int']}, "
                f"fill_count_int={execution_report_dict['fill_count_int']}"
            )
    return "\n".join(line_list)


def _render_serve_tick_summary_str(tick_detail_dict: dict[str, object]) -> str:
    return runner._render_tick_detail_str(tick_detail_dict)


def next_due(
    state_store_obj: LiveStateStore,
    as_of_ts: datetime,
    releases_root_path_str: str,
    env_mode_str: str,
    reconcile_grace_seconds_int: int = DEFAULT_RECONCILE_GRACE_SECONDS_INT,
    idle_max_sleep_seconds_int: int = DEFAULT_IDLE_MAX_SLEEP_SECONDS_INT,
) -> dict[str, object]:
    scheduler_decision_obj = get_scheduler_decision(
        state_store_obj=state_store_obj,
        as_of_ts=as_of_ts,
        releases_root_path_str=releases_root_path_str,
        env_mode_str=env_mode_str,
        reconcile_grace_seconds_int=reconcile_grace_seconds_int,
        idle_max_sleep_seconds_int=idle_max_sleep_seconds_int,
    )
    return scheduler_decision_obj.to_dict()


def run_once(
    state_store_obj: LiveStateStore,
    broker_adapter_obj,
    as_of_ts: datetime,
    releases_root_path_str: str,
    env_mode_str: str,
    reconcile_grace_seconds_int: int = DEFAULT_RECONCILE_GRACE_SECONDS_INT,
    idle_max_sleep_seconds_int: int = DEFAULT_IDLE_MAX_SLEEP_SECONDS_INT,
    log_path_str: str = DEFAULT_LOG_PATH_STR,
    debug_snapshots_bool: bool = False,
    broker_host_str: str | None = None,
    broker_port_int: int | None = None,
    broker_client_id_int: int | None = None,
    broker_timeout_seconds_float: float | None = None,
) -> dict[str, object]:
    _validate_release_root_for_mutation(releases_root_path_str, env_mode_str)
    broker_adapter_resolver_obj = runner.BrokerAdapterResolver(
        broker_adapter_obj=broker_adapter_obj,
        broker_host_str=broker_host_str,
        broker_port_int=broker_port_int,
        broker_client_id_int=broker_client_id_int,
        broker_timeout_seconds_float=broker_timeout_seconds_float,
        state_store_obj=state_store_obj,
        as_of_ts=as_of_ts,
    )
    scheduler_decision_obj = get_scheduler_decision(
        state_store_obj=state_store_obj,
        as_of_ts=as_of_ts,
        releases_root_path_str=releases_root_path_str,
        env_mode_str=env_mode_str,
        reconcile_grace_seconds_int=reconcile_grace_seconds_int,
        idle_max_sleep_seconds_int=idle_max_sleep_seconds_int,
    )
    detail_dict = scheduler_decision_obj.to_dict()
    detail_dict["tick_invoked_bool"] = False

    if scheduler_decision_obj.due_now_bool:
        _emit_scheduler_event(
            "scheduler_tick_invoked",
            {
                "as_of_timestamp_str": as_of_ts.isoformat(),
                "env_mode_str": env_mode_str,
                "next_phase_str": scheduler_decision_obj.next_phase_str,
                "reason_code_str": scheduler_decision_obj.reason_code_str,
                "related_pod_id_list": scheduler_decision_obj.related_pod_id_list,
            },
            log_path_str=log_path_str,
            print_events_bool=False,
        )
        if scheduler_decision_obj.next_phase_str == "eod_snapshot":
            tick_detail_dict = runner.eod_snapshot(
                state_store_obj=state_store_obj,
                broker_adapter_obj=broker_adapter_obj,
                as_of_ts=as_of_ts,
                env_mode_str=env_mode_str,
                releases_root_path_str=releases_root_path_str,
                log_path_str=log_path_str,
                broker_adapter_resolver_obj=broker_adapter_resolver_obj,
                require_due_bool=True,
            )
        else:
            tick_detail_dict = runner.tick(
                state_store_obj=state_store_obj,
                broker_adapter_obj=broker_adapter_obj,
                as_of_ts=as_of_ts,
                releases_root_path_str=releases_root_path_str,
                env_mode_str=env_mode_str,
                log_path_str=log_path_str,
                broker_adapter_resolver_obj=broker_adapter_resolver_obj,
            )
        detail_dict["tick_invoked_bool"] = True
        detail_dict["tick_detail_dict"] = tick_detail_dict
        detail_dict["post_tick_pod_status_dict_list"] = _build_related_pod_status_dict_list(
            state_store_obj=state_store_obj,
            as_of_ts=as_of_ts,
            releases_root_path_str=releases_root_path_str,
            env_mode_str=env_mode_str,
            related_pod_id_list=scheduler_decision_obj.related_pod_id_list,
        )
        detail_dict["post_tick_execution_report_dict_list"] = _build_related_execution_report_dict_list(
            state_store_obj=state_store_obj,
            as_of_ts=as_of_ts,
            releases_root_path_str=releases_root_path_str,
            related_pod_id_list=scheduler_decision_obj.related_pod_id_list,
        )
        if debug_snapshots_bool:
            detail_dict["post_tick_vplan_dict_list"] = _build_related_vplan_dict_list(
                state_store_obj=state_store_obj,
                as_of_ts=as_of_ts,
                releases_root_path_str=releases_root_path_str,
                related_pod_id_list=scheduler_decision_obj.related_pod_id_list,
            )
            detail_dict["post_tick_broker_order_snapshot_dict_list"] = (
                _build_related_broker_order_snapshot_dict_list(
                    state_store_obj=state_store_obj,
                    related_pod_id_list=scheduler_decision_obj.related_pod_id_list,
                )
            )
    return detail_dict


def serve(
    state_store_obj: LiveStateStore,
    broker_adapter_obj,
    releases_root_path_str: str,
    env_mode_str: str,
    broker_host_str: str | None,
    broker_port_int: int | None,
    broker_client_id_int: int | None,
    active_poll_seconds_int: int = DEFAULT_ACTIVE_POLL_SECONDS_INT,
    idle_max_sleep_seconds_int: int = DEFAULT_IDLE_MAX_SLEEP_SECONDS_INT,
    reconcile_grace_seconds_int: int = DEFAULT_RECONCILE_GRACE_SECONDS_INT,
    error_retry_seconds_int: int = DEFAULT_ERROR_RETRY_SECONDS_INT,
    log_path_str: str = DEFAULT_LOG_PATH_STR,
    broker_timeout_seconds_float: float | None = None,
) -> None:
    _validate_release_root_for_mutation(releases_root_path_str, env_mode_str)
    last_printed_sleep_signature_tup: tuple[object, ...] | None = None
    last_printed_sleep_timestamp_ts: datetime | None = None
    current_scheduler_decision_obj: SchedulerDecision | None = None
    broker_adapter_resolver_obj = runner.BrokerAdapterResolver(
        broker_adapter_obj=broker_adapter_obj,
        broker_host_str=broker_host_str,
        broker_port_int=broker_port_int,
        broker_client_id_int=broker_client_id_int,
        broker_timeout_seconds_float=broker_timeout_seconds_float,
    )
    _emit_scheduler_event(
        "scheduler_started",
        {
            "env_mode_str": env_mode_str,
            "active_poll_seconds_int": int(active_poll_seconds_int),
            "idle_max_sleep_seconds_int": int(idle_max_sleep_seconds_int),
            "reconcile_grace_seconds_int": int(reconcile_grace_seconds_int),
        },
        log_path_str=log_path_str,
        print_events_bool=False,
    )
    _emit_operator_message_spec_list(
        [
            _build_operator_message_spec_dict(
                level_str="INFO",
                phase_action_str="service.start",
                timestamp_obj=datetime.now(tz=UTC),
                field_map_dict={
                    "mode": env_mode_str,
                    "active_poll": f"{int(active_poll_seconds_int)}s",
                    "idle_sleep": f"{int(idle_max_sleep_seconds_int)}s",
                    "reconcile_grace": f"{int(reconcile_grace_seconds_int)}s",
                },
            )
        ],
        log_path_str=log_path_str,
        print_message_bool=True,
    )

    while True:
        as_of_ts = datetime.now(tz=UTC)
        broker_adapter_resolver_obj.set_incubation_context(
            state_store_obj=state_store_obj,
            as_of_ts=as_of_ts,
        )
        try:
            current_scheduler_decision_obj = get_scheduler_decision(
                state_store_obj=state_store_obj,
                as_of_ts=as_of_ts,
                releases_root_path_str=releases_root_path_str,
                env_mode_str=env_mode_str,
                reconcile_grace_seconds_int=reconcile_grace_seconds_int,
                idle_max_sleep_seconds_int=idle_max_sleep_seconds_int,
            )
            if current_scheduler_decision_obj.due_now_bool:
                _emit_scheduler_event(
                    "scheduler_due_now",
                    {
                        "as_of_timestamp_str": as_of_ts.isoformat(),
                        "env_mode_str": env_mode_str,
                        "next_phase_str": current_scheduler_decision_obj.next_phase_str,
                        "reason_code_str": current_scheduler_decision_obj.reason_code_str,
                        "related_pod_id_list": current_scheduler_decision_obj.related_pod_id_list,
                    },
                    log_path_str=log_path_str,
                    print_events_bool=False,
                )
                _emit_operator_message_spec_list(
                    _build_phase_start_operator_message_spec_list(
                        state_store_obj=state_store_obj,
                        scheduler_decision_obj=current_scheduler_decision_obj,
                        broker_adapter_resolver_obj=broker_adapter_resolver_obj,
                    ),
                    log_path_str=log_path_str,
                    print_message_bool=True,
                )
                last_printed_sleep_signature_tup = None
                last_printed_sleep_timestamp_ts = None
                if current_scheduler_decision_obj.next_phase_str == "eod_snapshot":
                    tick_detail_dict = runner.eod_snapshot(
                        state_store_obj=state_store_obj,
                        broker_adapter_obj=broker_adapter_obj,
                        as_of_ts=as_of_ts,
                        env_mode_str=env_mode_str,
                        releases_root_path_str=releases_root_path_str,
                        log_path_str=log_path_str,
                        broker_adapter_resolver_obj=broker_adapter_resolver_obj,
                        require_due_bool=True,
                    )
                else:
                    tick_detail_dict = runner.tick(
                        state_store_obj=state_store_obj,
                        broker_adapter_obj=broker_adapter_obj,
                        as_of_ts=as_of_ts,
                        releases_root_path_str=releases_root_path_str,
                        env_mode_str=env_mode_str,
                        log_path_str=log_path_str,
                        broker_adapter_resolver_obj=broker_adapter_resolver_obj,
                    )
                post_tick_pod_status_dict_list = _build_related_pod_status_dict_list(
                    state_store_obj=state_store_obj,
                    as_of_ts=as_of_ts,
                    releases_root_path_str=releases_root_path_str,
                    env_mode_str=env_mode_str,
                    related_pod_id_list=current_scheduler_decision_obj.related_pod_id_list,
                )
                post_tick_execution_report_dict_list = _build_related_execution_report_dict_list(
                    state_store_obj=state_store_obj,
                    as_of_ts=as_of_ts,
                    releases_root_path_str=releases_root_path_str,
                    related_pod_id_list=current_scheduler_decision_obj.related_pod_id_list,
                )
                post_tick_vplan_dict_list = _build_related_vplan_dict_list(
                    state_store_obj=state_store_obj,
                    as_of_ts=as_of_ts,
                    releases_root_path_str=releases_root_path_str,
                    related_pod_id_list=current_scheduler_decision_obj.related_pod_id_list,
                )
                post_tick_broker_order_snapshot_dict_list = _build_related_broker_order_snapshot_dict_list(
                    state_store_obj=state_store_obj,
                    related_pod_id_list=current_scheduler_decision_obj.related_pod_id_list,
                )
                _emit_scheduler_event(
                    "scheduler_phase_idle",
                    {
                        "as_of_timestamp_str": as_of_ts.isoformat(),
                        "env_mode_str": env_mode_str,
                        "next_phase_str": current_scheduler_decision_obj.next_phase_str,
                        "tick_detail_dict": tick_detail_dict,
                        "post_tick_pod_status_dict_list": post_tick_pod_status_dict_list,
                        "post_tick_execution_report_dict_list": post_tick_execution_report_dict_list,
                        "post_tick_vplan_dict_list": post_tick_vplan_dict_list,
                        "post_tick_broker_order_snapshot_dict_list": post_tick_broker_order_snapshot_dict_list,
                    },
                    log_path_str=log_path_str,
                    print_events_bool=False,
                )
                _emit_operator_message_spec_list(
                    _build_phase_result_operator_message_spec_list(
                        state_store_obj=state_store_obj,
                        scheduler_decision_obj=current_scheduler_decision_obj,
                        tick_detail_dict=tick_detail_dict,
                        pod_status_dict_list=post_tick_pod_status_dict_list,
                        execution_report_dict_list=post_tick_execution_report_dict_list,
                        broker_adapter_resolver_obj=broker_adapter_resolver_obj,
                    ),
                    log_path_str=log_path_str,
                    print_message_bool=True,
                )
                print(_render_serve_tick_summary_str(tick_detail_dict), flush=True)
                last_printed_sleep_signature_tup = None
                last_printed_sleep_timestamp_ts = None
                continue

            sleep_seconds_float = _build_sleep_seconds_float(
                scheduler_decision_obj=current_scheduler_decision_obj,
                as_of_ts=as_of_ts,
                active_poll_seconds_int=active_poll_seconds_int,
                idle_max_sleep_seconds_int=idle_max_sleep_seconds_int,
            )
            sleep_event_payload_dict = {
                "as_of_timestamp_str": as_of_ts.isoformat(),
                "env_mode_str": env_mode_str,
                "next_phase_str": current_scheduler_decision_obj.next_phase_str,
                "reason_code_str": current_scheduler_decision_obj.reason_code_str,
                "next_due_timestamp_str": current_scheduler_decision_obj.next_due_timestamp_ts.isoformat(),
                "sleep_seconds_float": sleep_seconds_float,
                "related_pod_id_list": current_scheduler_decision_obj.related_pod_id_list,
                "pod_status_dict_list": _build_related_pod_status_dict_list(
                    state_store_obj=state_store_obj,
                    as_of_ts=as_of_ts,
                    releases_root_path_str=releases_root_path_str,
                    env_mode_str=env_mode_str,
                    related_pod_id_list=current_scheduler_decision_obj.related_pod_id_list,
                ),
                "execution_report_dict_list": _build_related_execution_report_dict_list(
                    state_store_obj=state_store_obj,
                    as_of_ts=as_of_ts,
                    releases_root_path_str=releases_root_path_str,
                    related_pod_id_list=current_scheduler_decision_obj.related_pod_id_list,
                ),
                "broker_order_snapshot_dict_list": _build_related_broker_order_snapshot_dict_list(
                    state_store_obj=state_store_obj,
                    related_pod_id_list=current_scheduler_decision_obj.related_pod_id_list,
                ),
            }
            sleep_signature_tup = _build_scheduler_sleep_signature_tup(sleep_event_payload_dict)
            print_sleep_event_bool = _should_emit_wait_operator_message(
                last_printed_signature_tup=last_printed_sleep_signature_tup,
                current_signature_tup=sleep_signature_tup,
                last_printed_timestamp_ts=last_printed_sleep_timestamp_ts,
                as_of_ts=as_of_ts,
                heartbeat_seconds_int=(
                    int(active_poll_seconds_int)
                    if current_scheduler_decision_obj.active_poll_bool
                    else DEFAULT_OPERATOR_HEARTBEAT_SECONDS_INT
                ),
            )
            _emit_scheduler_event(
                "scheduler_sleeping",
                sleep_event_payload_dict,
                log_path_str=log_path_str,
                print_events_bool=False,
            )
            if print_sleep_event_bool:
                _emit_operator_message_spec_list(
                    _build_wait_operator_message_spec_list(
                        current_scheduler_decision_obj,
                        pod_status_dict_list=list(sleep_event_payload_dict["pod_status_dict_list"]),
                        execution_report_dict_list=list(
                            sleep_event_payload_dict["execution_report_dict_list"]
                        ),
                        broker_order_snapshot_dict_list=list(
                            sleep_event_payload_dict["broker_order_snapshot_dict_list"]
                        ),
                    )
                    + _build_stuck_operator_message_spec_list(
                        state_store_obj=state_store_obj,
                        as_of_ts=as_of_ts,
                        reconcile_grace_seconds_int=reconcile_grace_seconds_int,
                    ),
                    log_path_str=log_path_str,
                    print_message_bool=True,
                )
            last_printed_sleep_signature_tup = sleep_signature_tup
            last_printed_sleep_timestamp_ts = as_of_ts
            time.sleep(sleep_seconds_float)
            _emit_scheduler_event(
                "scheduler_woke",
                {
                    "env_mode_str": env_mode_str,
                    "scheduled_wake_timestamp_str": datetime.now(tz=UTC).isoformat(),
                },
                log_path_str=log_path_str,
                print_events_bool=False,
            )
        except Exception as exc:  # pragma: no cover - daemon safety
            _emit_scheduler_event(
                "scheduler_error_retry",
                {
                    "as_of_timestamp_str": as_of_ts.isoformat(),
                    "env_mode_str": env_mode_str,
                    "error_str": str(exc),
                    "error_retry_seconds_int": int(error_retry_seconds_int),
                },
                log_path_str=log_path_str,
                print_events_bool=False,
            )
            if current_scheduler_decision_obj is not None:
                _emit_operator_message_spec_list(
                    _build_phase_failure_operator_message_spec_list(
                        state_store_obj=state_store_obj,
                        scheduler_decision_obj=current_scheduler_decision_obj,
                        as_of_ts=as_of_ts,
                        error_str=str(exc),
                        error_retry_seconds_int=error_retry_seconds_int,
                        broker_adapter_resolver_obj=broker_adapter_resolver_obj,
                    ),
                    log_path_str=log_path_str,
                    print_message_bool=True,
                )
            last_printed_sleep_signature_tup = None
            last_printed_sleep_timestamp_ts = None
            time.sleep(float(error_retry_seconds_int))


def main(argv_list: list[str] | None = None) -> int:
    parser_obj = argparse.ArgumentParser(description="alpha.live scheduler service")
    parser_obj.add_argument(
        "command_name_str",
        choices=("serve", "next_due", "run_once", "compare_reference", "eod_snapshot"),
    )
    parser_obj.add_argument("--db-path", dest="db_path_str", default=None)
    parser_obj.add_argument("--releases-root", dest="releases_root_path_str", default=DEFAULT_RELEASES_ROOT_PATH_STR)
    parser_obj.add_argument("--as-of-ts", dest="as_of_timestamp_str", default=None)
    parser_obj.add_argument("--mode", dest="env_mode_str", default="paper")
    parser_obj.add_argument("--log-path", dest="log_path_str", default=DEFAULT_LOG_PATH_STR)
    parser_obj.add_argument("--json", dest="json_output_bool", action="store_true")
    parser_obj.add_argument("--broker-host", dest="broker_host_str", default=None)
    parser_obj.add_argument("--broker-port", dest="broker_port_int", type=int, default=None)
    parser_obj.add_argument("--broker-client-id", dest="broker_client_id_int", type=int, default=None)
    parser_obj.add_argument("--broker-timeout-seconds", dest="broker_timeout_seconds_float", type=float, default=None)
    parser_obj.add_argument("--pod-id", dest="pod_id_str", default=None)
    parser_obj.add_argument("--reference-strategy-pickle", dest="reference_strategy_pickle_path_str", default=None)
    parser_obj.add_argument("--html", dest="html_output_bool", action="store_true")
    parser_obj.add_argument("--output-dir", dest="output_dir_str", default="results")
    parser_obj.add_argument(
        "--active-poll-seconds",
        dest="active_poll_seconds_int",
        type=int,
        default=DEFAULT_ACTIVE_POLL_SECONDS_INT,
    )
    parser_obj.add_argument(
        "--idle-max-sleep-seconds",
        dest="idle_max_sleep_seconds_int",
        type=int,
        default=DEFAULT_IDLE_MAX_SLEEP_SECONDS_INT,
    )
    parser_obj.add_argument(
        "--reconcile-grace-seconds",
        dest="reconcile_grace_seconds_int",
        type=int,
        default=DEFAULT_RECONCILE_GRACE_SECONDS_INT,
    )
    parser_obj.add_argument(
        "--error-retry-seconds",
        dest="error_retry_seconds_int",
        type=int,
        default=DEFAULT_ERROR_RETRY_SECONDS_INT,
    )
    parsed_args_obj = parser_obj.parse_args(argv_list)

    db_path_str = _resolve_db_path_for_mode_str(
        db_path_str=parsed_args_obj.db_path_str,
        env_mode_str=parsed_args_obj.env_mode_str,
    )
    state_store_obj = LiveStateStore(db_path_str)
    broker_adapter_obj = None
    as_of_ts = _parse_as_of_timestamp_ts(parsed_args_obj.as_of_timestamp_str)

    if parsed_args_obj.command_name_str == "serve":
        serve(
            state_store_obj=state_store_obj,
            broker_adapter_obj=broker_adapter_obj,
            releases_root_path_str=parsed_args_obj.releases_root_path_str,
            env_mode_str=parsed_args_obj.env_mode_str,
            broker_host_str=parsed_args_obj.broker_host_str,
            broker_port_int=parsed_args_obj.broker_port_int,
            broker_client_id_int=parsed_args_obj.broker_client_id_int,
            active_poll_seconds_int=parsed_args_obj.active_poll_seconds_int,
            idle_max_sleep_seconds_int=parsed_args_obj.idle_max_sleep_seconds_int,
            reconcile_grace_seconds_int=parsed_args_obj.reconcile_grace_seconds_int,
            error_retry_seconds_int=parsed_args_obj.error_retry_seconds_int,
            log_path_str=parsed_args_obj.log_path_str,
            broker_timeout_seconds_float=parsed_args_obj.broker_timeout_seconds_float,
        )
        return 0

    if parsed_args_obj.command_name_str == "run_once":
        detail_dict = run_once(
            state_store_obj=state_store_obj,
            broker_adapter_obj=broker_adapter_obj,
            as_of_ts=as_of_ts,
            releases_root_path_str=parsed_args_obj.releases_root_path_str,
            env_mode_str=parsed_args_obj.env_mode_str,
            reconcile_grace_seconds_int=parsed_args_obj.reconcile_grace_seconds_int,
            idle_max_sleep_seconds_int=parsed_args_obj.idle_max_sleep_seconds_int,
            log_path_str=parsed_args_obj.log_path_str,
            broker_host_str=parsed_args_obj.broker_host_str,
            broker_port_int=parsed_args_obj.broker_port_int,
            broker_client_id_int=parsed_args_obj.broker_client_id_int,
            broker_timeout_seconds_float=parsed_args_obj.broker_timeout_seconds_float,
        )
    elif parsed_args_obj.command_name_str == "compare_reference":
        detail_dict = runner.get_compare_reference_summary(
            state_store_obj=state_store_obj,
            as_of_ts=as_of_ts,
            releases_root_path_str=parsed_args_obj.releases_root_path_str,
            env_mode_str=parsed_args_obj.env_mode_str,
            pod_id_str=parsed_args_obj.pod_id_str,
            reference_strategy_pickle_path_str=parsed_args_obj.reference_strategy_pickle_path_str,
            html_output_bool=parsed_args_obj.html_output_bool,
            output_dir_str=parsed_args_obj.output_dir_str,
        )
    elif parsed_args_obj.command_name_str == "eod_snapshot":
        detail_dict = runner.eod_snapshot(
            state_store_obj=state_store_obj,
            broker_adapter_obj=broker_adapter_obj,
            as_of_ts=as_of_ts,
            env_mode_str=parsed_args_obj.env_mode_str,
            releases_root_path_str=parsed_args_obj.releases_root_path_str,
            log_path_str=parsed_args_obj.log_path_str,
            broker_host_str=parsed_args_obj.broker_host_str,
            broker_port_int=parsed_args_obj.broker_port_int,
            broker_client_id_int=parsed_args_obj.broker_client_id_int,
            broker_timeout_seconds_float=parsed_args_obj.broker_timeout_seconds_float,
        )
    else:
        detail_dict = next_due(
            state_store_obj=state_store_obj,
            as_of_ts=as_of_ts,
            releases_root_path_str=parsed_args_obj.releases_root_path_str,
            env_mode_str=parsed_args_obj.env_mode_str,
            reconcile_grace_seconds_int=parsed_args_obj.reconcile_grace_seconds_int,
            idle_max_sleep_seconds_int=parsed_args_obj.idle_max_sleep_seconds_int,
        )

    if parsed_args_obj.json_output_bool:
        print(json.dumps(detail_dict, indent=2, sort_keys=True))
    elif parsed_args_obj.command_name_str in ("compare_reference", "eod_snapshot"):
        print(runner._render_command_output_str(parsed_args_obj.command_name_str, detail_dict))
    else:
        print(_render_scheduler_output_str(detail_dict))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
