from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from alpha.live import runner, scheduler_utils
from alpha.live.logging_utils import DEFAULT_LOG_PATH_STR, log_event
from alpha.live.models import DecisionPlan, LiveRelease, VPlan
from alpha.live.order_clerk import IBKRGatewayBrokerAdapter
from alpha.live.release_manifest import load_release_list
from alpha.live.state_store_v2 import LiveStateStore


DEFAULT_ACTIVE_POLL_SECONDS_INT = 30
DEFAULT_IDLE_MAX_SLEEP_SECONDS_INT = 900
DEFAULT_RECONCILE_GRACE_SECONDS_INT = 300
DEFAULT_ERROR_RETRY_SECONDS_INT = 60
DEFAULT_RELEASES_ROOT_PATH_STR = str(Path(__file__).resolve().parent / "releases")
DEFAULT_DB_PATH_STR = str(Path(__file__).resolve().parent / "live_state.sqlite3")


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


def _load_release_list_and_sync(
    releases_root_path_str: str,
    state_store_obj: LiveStateStore,
) -> list[LiveRelease]:
    release_list = load_release_list(releases_root_path_str)
    state_store_obj.upsert_release_list(release_list)
    return release_list


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
    idle_probe_pod_id_list: list[str] = []

    for release_obj in enabled_release_list:
        latest_decision_plan_obj = state_store_obj.get_latest_decision_plan_for_pod(release_obj.pod_id_str)
        current_vplan_obj = _get_current_cycle_vplan_obj(state_store_obj, latest_decision_plan_obj)
        build_gate_dict = scheduler_utils.evaluate_build_gate_dict(release_obj, as_of_ts)

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
            and latest_decision_plan_obj.target_execution_timestamp_ts <= as_of_ts
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

        if current_vplan_obj is not None and current_vplan_obj.status_str == "submitted":
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
            f"created_decision_plan_count_int={int(tick_detail_dict.get('created_decision_plan_count_int', 0))}, "
            f"created_vplan_count_int={int(tick_detail_dict.get('created_vplan_count_int', 0))}, "
            f"submitted_vplan_count_int={int(tick_detail_dict.get('submitted_vplan_count_int', 0))}, "
            f"completed_vplan_count_int={int(tick_detail_dict.get('completed_vplan_count_int', 0))}"
        )
    return "\n".join(line_list)


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
    broker_adapter_obj: IBKRGatewayBrokerAdapter,
    as_of_ts: datetime,
    releases_root_path_str: str,
    env_mode_str: str,
    reconcile_grace_seconds_int: int = DEFAULT_RECONCILE_GRACE_SECONDS_INT,
    idle_max_sleep_seconds_int: int = DEFAULT_IDLE_MAX_SLEEP_SECONDS_INT,
    log_path_str: str = DEFAULT_LOG_PATH_STR,
) -> dict[str, object]:
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
        log_event(
            "scheduler_tick_invoked",
            {
                "as_of_timestamp_str": as_of_ts.isoformat(),
                "env_mode_str": env_mode_str,
                "next_phase_str": scheduler_decision_obj.next_phase_str,
                "reason_code_str": scheduler_decision_obj.reason_code_str,
                "related_pod_id_list": scheduler_decision_obj.related_pod_id_list,
            },
            log_path_str=log_path_str,
        )
        tick_detail_dict = runner.tick(
            state_store_obj=state_store_obj,
            broker_adapter_obj=broker_adapter_obj,
            as_of_ts=as_of_ts,
            releases_root_path_str=releases_root_path_str,
            env_mode_str=env_mode_str,
            log_path_str=log_path_str,
        )
        detail_dict["tick_invoked_bool"] = True
        detail_dict["tick_detail_dict"] = tick_detail_dict
    return detail_dict


def serve(
    state_store_obj: LiveStateStore,
    broker_adapter_obj: IBKRGatewayBrokerAdapter,
    releases_root_path_str: str,
    env_mode_str: str,
    active_poll_seconds_int: int = DEFAULT_ACTIVE_POLL_SECONDS_INT,
    idle_max_sleep_seconds_int: int = DEFAULT_IDLE_MAX_SLEEP_SECONDS_INT,
    reconcile_grace_seconds_int: int = DEFAULT_RECONCILE_GRACE_SECONDS_INT,
    error_retry_seconds_int: int = DEFAULT_ERROR_RETRY_SECONDS_INT,
    log_path_str: str = DEFAULT_LOG_PATH_STR,
) -> None:
    log_event(
        "scheduler_started",
        {
            "env_mode_str": env_mode_str,
            "active_poll_seconds_int": int(active_poll_seconds_int),
            "idle_max_sleep_seconds_int": int(idle_max_sleep_seconds_int),
            "reconcile_grace_seconds_int": int(reconcile_grace_seconds_int),
        },
        log_path_str=log_path_str,
    )

    while True:
        as_of_ts = datetime.now(tz=UTC)
        try:
            scheduler_decision_obj = get_scheduler_decision(
                state_store_obj=state_store_obj,
                as_of_ts=as_of_ts,
                releases_root_path_str=releases_root_path_str,
                env_mode_str=env_mode_str,
                reconcile_grace_seconds_int=reconcile_grace_seconds_int,
                idle_max_sleep_seconds_int=idle_max_sleep_seconds_int,
            )
            if scheduler_decision_obj.due_now_bool:
                log_event(
                    "scheduler_due_now",
                    {
                        "as_of_timestamp_str": as_of_ts.isoformat(),
                        "env_mode_str": env_mode_str,
                        "next_phase_str": scheduler_decision_obj.next_phase_str,
                        "reason_code_str": scheduler_decision_obj.reason_code_str,
                        "related_pod_id_list": scheduler_decision_obj.related_pod_id_list,
                    },
                    log_path_str=log_path_str,
                )
                tick_detail_dict = runner.tick(
                    state_store_obj=state_store_obj,
                    broker_adapter_obj=broker_adapter_obj,
                    as_of_ts=as_of_ts,
                    releases_root_path_str=releases_root_path_str,
                    env_mode_str=env_mode_str,
                    log_path_str=log_path_str,
                )
                log_event(
                    "scheduler_phase_idle",
                    {
                        "as_of_timestamp_str": as_of_ts.isoformat(),
                        "env_mode_str": env_mode_str,
                        "next_phase_str": scheduler_decision_obj.next_phase_str,
                        "tick_detail_dict": tick_detail_dict,
                    },
                    log_path_str=log_path_str,
                )
                continue

            sleep_seconds_float = _build_sleep_seconds_float(
                scheduler_decision_obj=scheduler_decision_obj,
                as_of_ts=as_of_ts,
                active_poll_seconds_int=active_poll_seconds_int,
                idle_max_sleep_seconds_int=idle_max_sleep_seconds_int,
            )
            log_event(
                "scheduler_sleeping",
                {
                    "as_of_timestamp_str": as_of_ts.isoformat(),
                    "env_mode_str": env_mode_str,
                    "next_phase_str": scheduler_decision_obj.next_phase_str,
                    "reason_code_str": scheduler_decision_obj.reason_code_str,
                    "next_due_timestamp_str": scheduler_decision_obj.next_due_timestamp_ts.isoformat(),
                    "sleep_seconds_float": sleep_seconds_float,
                    "related_pod_id_list": scheduler_decision_obj.related_pod_id_list,
                },
                log_path_str=log_path_str,
            )
            time.sleep(sleep_seconds_float)
            log_event(
                "scheduler_woke",
                {
                    "env_mode_str": env_mode_str,
                    "scheduled_wake_timestamp_str": datetime.now(tz=UTC).isoformat(),
                },
                log_path_str=log_path_str,
            )
        except Exception as exc:  # pragma: no cover - daemon safety
            log_event(
                "scheduler_error_retry",
                {
                    "as_of_timestamp_str": as_of_ts.isoformat(),
                    "env_mode_str": env_mode_str,
                    "error_str": str(exc),
                    "error_retry_seconds_int": int(error_retry_seconds_int),
                },
                log_path_str=log_path_str,
            )
            time.sleep(float(error_retry_seconds_int))


def main(argv_list: list[str] | None = None) -> int:
    parser_obj = argparse.ArgumentParser(description="alpha.live scheduler service")
    parser_obj.add_argument(
        "command_name_str",
        choices=("serve", "next_due", "run_once"),
    )
    parser_obj.add_argument("--db-path", dest="db_path_str", default=DEFAULT_DB_PATH_STR)
    parser_obj.add_argument("--releases-root", dest="releases_root_path_str", default=DEFAULT_RELEASES_ROOT_PATH_STR)
    parser_obj.add_argument("--as-of-ts", dest="as_of_timestamp_str", default=None)
    parser_obj.add_argument("--mode", dest="env_mode_str", default="paper")
    parser_obj.add_argument("--log-path", dest="log_path_str", default=DEFAULT_LOG_PATH_STR)
    parser_obj.add_argument("--json", dest="json_output_bool", action="store_true")
    parser_obj.add_argument("--broker-host", dest="broker_host_str", default="127.0.0.1")
    parser_obj.add_argument("--broker-port", dest="broker_port_int", type=int, default=7497)
    parser_obj.add_argument("--broker-client-id", dest="broker_client_id_int", type=int, default=31)
    parser_obj.add_argument("--broker-timeout-seconds", dest="broker_timeout_seconds_float", type=float, default=4.0)
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

    state_store_obj = LiveStateStore(parsed_args_obj.db_path_str)
    broker_adapter_obj = IBKRGatewayBrokerAdapter(
        host_str=parsed_args_obj.broker_host_str,
        port_int=parsed_args_obj.broker_port_int,
        client_id_int=parsed_args_obj.broker_client_id_int,
        timeout_seconds_float=parsed_args_obj.broker_timeout_seconds_float,
    )
    as_of_ts = _parse_as_of_timestamp_ts(parsed_args_obj.as_of_timestamp_str)

    if parsed_args_obj.command_name_str == "serve":
        serve(
            state_store_obj=state_store_obj,
            broker_adapter_obj=broker_adapter_obj,
            releases_root_path_str=parsed_args_obj.releases_root_path_str,
            env_mode_str=parsed_args_obj.env_mode_str,
            active_poll_seconds_int=parsed_args_obj.active_poll_seconds_int,
            idle_max_sleep_seconds_int=parsed_args_obj.idle_max_sleep_seconds_int,
            reconcile_grace_seconds_int=parsed_args_obj.reconcile_grace_seconds_int,
            error_retry_seconds_int=parsed_args_obj.error_retry_seconds_int,
            log_path_str=parsed_args_obj.log_path_str,
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
    else:
        print(_render_scheduler_output_str(detail_dict))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
