from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
import uuid

from alpha.live import scheduler_utils, strategy_host
from alpha.live.execution_quality import build_execution_quality_snapshot
from alpha.live.logging_utils import DEFAULT_LOG_PATH_STR, log_event
from alpha.live.models import LiveRelease, PodState
from alpha.live.order_clerk import (
    BrokerAdapter,
    IBKRGatewayBrokerAdapter,
    build_broker_order_request_list,
    validate_order_intent_list,
)
from alpha.live.reconcile import reconcile_account_state
from alpha.live.release_manifest import load_release_list
from alpha.live.state_store import LiveStateStore


DEFAULT_RELEASES_ROOT_PATH_STR = str(Path(__file__).resolve().parent / "releases")
DEFAULT_DB_PATH_STR = str(Path(__file__).resolve().parent / "live_state.sqlite3")


def _pluralize_label_str(count_int: int, singular_label_str: str, plural_label_str: str | None = None) -> str:
    if count_int == 1:
        return singular_label_str
    if plural_label_str is not None:
        return plural_label_str
    return f"{singular_label_str}s"


def _render_tick_detail_str(detail_dict: dict[str, object]) -> str:
    if not bool(detail_dict.get("lease_acquired_bool", False)):
        return "\n".join(
            [
                "Tick Result",
                "- Another tick is already running.",
                "- Nothing was done in this run.",
                "",
                "Raw Fields",
                "- lease_acquired_bool: false",
            ]
        )

    created_plan_count_int = int(detail_dict.get("created_plan_count_int", 0))
    skipped_plan_count_int = int(detail_dict.get("skipped_plan_count_int", 0))
    submitted_plan_count_int = int(detail_dict.get("submitted_plan_count_int", 0))
    blocked_plan_count_int = int(detail_dict.get("blocked_plan_count_int", 0))
    completed_plan_count_int = int(detail_dict.get("completed_plan_count_int", 0))
    reason_count_map_dict = dict(detail_dict.get("reason_count_map_dict", {}))

    summary_line_list: list[str] = ["Tick Result"]
    action_line_list: list[str] = []
    next_step_line_str = "- Nothing is due right now."

    if created_plan_count_int > 0:
        action_line_list.append(
            f"- Built {created_plan_count_int} new {_pluralize_label_str(created_plan_count_int, 'plan')}."
        )
    if skipped_plan_count_int > 0:
        action_line_list.append(
            f"- Skipped {skipped_plan_count_int} existing active {_pluralize_label_str(skipped_plan_count_int, 'plan')}."
        )
    if submitted_plan_count_int > 0:
        action_line_list.append(
            f"- Submitted {submitted_plan_count_int} {_pluralize_label_str(submitted_plan_count_int, 'plan')} to the broker."
        )
    if completed_plan_count_int > 0:
        action_line_list.append(
            f"- Completed post-trade reconcile for {completed_plan_count_int} {_pluralize_label_str(completed_plan_count_int, 'plan')}."
        )
    if blocked_plan_count_int > 0:
        action_line_list.append(
            f"- Blocked {blocked_plan_count_int} {_pluralize_label_str(blocked_plan_count_int, 'plan')} for safety."
        )
        if "broker_not_ready" in reason_count_map_dict:
            next_step_line_str = "- Next step: start or connect the broker session, then run tick again."
        elif "reconciliation_failed" in reason_count_map_dict:
            next_step_line_str = "- Next step: inspect reconciliation and fix the broker/model mismatch before retrying."
        elif "account_not_visible" in reason_count_map_dict:
            next_step_line_str = "- Next step: verify the routed account is visible in the current broker session."
        elif "env_mode_mismatch" in reason_count_map_dict:
            next_step_line_str = "- Next step: use the correct --mode or fix the manifest mode."
        else:
            next_step_line_str = "- Next step: inspect the reason list below and fix the blocker."

    if len(action_line_list) == 0:
        action_line_list.append("- No new action was needed in this run.")

    summary_line_list.extend(action_line_list)

    if len(reason_count_map_dict) > 0:
        summary_line_list.append("- Reasons:")
        for reason_code_str, count_obj in sorted(reason_count_map_dict.items()):
            reason_count_int = int(count_obj)
            summary_line_list.append(
                f"  - {reason_code_str}: {reason_count_int}"
            )

    summary_line_list.append(next_step_line_str)
    summary_line_list.extend(
        [
            "",
            "Raw Fields",
            f"- lease_acquired_bool: {str(bool(detail_dict.get('lease_acquired_bool', False))).lower()}",
            f"- created_plan_count_int: {created_plan_count_int}",
            f"- skipped_plan_count_int: {skipped_plan_count_int}",
            f"- submitted_plan_count_int: {submitted_plan_count_int}",
            f"- blocked_plan_count_int: {blocked_plan_count_int}",
            f"- completed_plan_count_int: {completed_plan_count_int}",
            f"- reason_count_map_dict: {json.dumps(reason_count_map_dict, sort_keys=True)}",
        ]
    )
    return "\n".join(summary_line_list)


def _render_status_detail_str(detail_dict: dict[str, object]) -> str:
    pod_status_dict_list = list(detail_dict.get("pod_status_dict_list", []))
    line_list = [
        "Status",
        f"- Enabled pods: {int(detail_dict.get('enabled_pod_count_int', 0))}",
    ]

    if len(pod_status_dict_list) == 0:
        line_list.append("- No enabled pods were found.")
        return "\n".join(line_list)

    for pod_status_dict in pod_status_dict_list:
        line_list.extend(
            [
                "",
                f"Pod: {pod_status_dict['pod_id_str']}",
                f"- Release: {pod_status_dict['release_id_str']}",
                f"- Account: {pod_status_dict['account_route_str']}",
                f"- Plan status: {pod_status_dict['latest_order_plan_status_str'] or 'none'}",
                f"- Next action: {pod_status_dict['next_action_str']}",
                f"- Why: {pod_status_dict['reason_code_str']}",
                f"- Latest signal time: {pod_status_dict['latest_signal_timestamp_str'] or 'none'}",
                f"- Planned submit time: {pod_status_dict['latest_submission_timestamp_str'] or 'none'}",
                f"- Latest fill time: {pod_status_dict['latest_fill_timestamp_str'] or 'none'}",
            ]
        )

    return "\n".join(line_list)


def _render_execution_report_detail_str(detail_dict: dict[str, object]) -> str:
    execution_report_dict_list = list(detail_dict.get("execution_report_dict_list", []))
    line_list = ["Execution Report"]

    if len(execution_report_dict_list) == 0:
        line_list.append("- No fills were found yet.")
        return "\n".join(line_list)

    for execution_report_dict in execution_report_dict_list:
        line_list.extend(
            [
                "",
                f"Pod: {execution_report_dict['pod_id_str']}",
                f"- Release: {execution_report_dict['release_id_str']}",
                f"- Latest order plan id: {execution_report_dict['latest_order_plan_id_int']}",
                f"- Fill count: {execution_report_dict['fill_count_int']}",
            ]
        )
        for fill_row_dict in execution_report_dict["fill_row_dict_list"]:
            line_list.append(
                "- Fill: "
                f"{fill_row_dict['asset_str']} | "
                f"amount={fill_row_dict['fill_amount_float']} | "
                f"price={fill_row_dict['fill_price_float']} | "
                f"time={fill_row_dict['fill_timestamp_str']}"
            )

    return "\n".join(line_list)


def _render_command_output_str(command_name_str: str, detail_dict: dict[str, object]) -> str:
    if command_name_str == "tick":
        return _render_tick_detail_str(detail_dict)
    if command_name_str == "status":
        return _render_status_detail_str(detail_dict)
    if command_name_str == "execution_report":
        return _render_execution_report_detail_str(detail_dict)
    return json.dumps(detail_dict, indent=2, sort_keys=True)


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


def _get_model_state_or_default(
    release_obj: LiveRelease,
    state_store_obj: LiveStateStore,
    as_of_ts: datetime,
) -> PodState:
    pod_state_obj = state_store_obj.get_pod_state(release_obj.pod_id_str)
    if pod_state_obj is not None:
        return pod_state_obj

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


def _build_release_log_payload_dict(
    release_obj: LiveRelease,
    as_of_ts: datetime,
    extra_payload_dict: dict | None = None,
) -> dict:
    payload_dict = {
        "as_of_timestamp_str": as_of_ts.isoformat(),
        "pod_id_str": release_obj.pod_id_str,
        "release_id_str": release_obj.release_id_str,
        "account_route_str": release_obj.account_route_str,
        "session_calendar_id_str": release_obj.session_calendar_id_str,
        "signal_clock_str": release_obj.signal_clock_str,
        "execution_policy_str": release_obj.execution_policy_str,
        "data_profile_str": release_obj.data_profile_str,
    }
    if extra_payload_dict is not None:
        payload_dict.update(extra_payload_dict)
    return payload_dict


def _build_order_plan_log_payload_dict(
    release_obj: LiveRelease,
    order_plan_obj,
    as_of_ts: datetime,
    extra_payload_dict: dict | None = None,
) -> dict:
    payload_dict = _build_release_log_payload_dict(
        release_obj,
        as_of_ts,
        {
            "order_plan_id_int": int(order_plan_obj.order_plan_id_int or 0),
            "plan_status_str": order_plan_obj.status_str,
            "signal_timestamp_str": order_plan_obj.signal_timestamp_ts.isoformat(),
            "submission_timestamp_str": order_plan_obj.submission_timestamp_ts.isoformat(),
            "target_execution_timestamp_str": order_plan_obj.target_execution_timestamp_ts.isoformat(),
        },
    )
    if extra_payload_dict is not None:
        payload_dict.update(extra_payload_dict)
    return payload_dict


def build_order_plans(
    state_store_obj: LiveStateStore,
    as_of_ts: datetime,
    releases_root_path_str: str,
    log_path_str: str = DEFAULT_LOG_PATH_STR,
) -> dict[str, object]:
    """Build and freeze any due strategy decisions into order plans."""
    release_list = _load_release_list_and_sync(releases_root_path_str, state_store_obj)
    due_release_list = scheduler_utils.select_due_release_list(release_list, as_of_ts)
    created_plan_count_int = 0
    skipped_plan_count_int = 0
    reason_counter_obj: Counter[str] = Counter()

    for release_obj in due_release_list:
        pod_state_obj = _get_model_state_or_default(release_obj, state_store_obj, as_of_ts)
        order_plan_obj = strategy_host.build_order_plan_for_release(
            release_obj=release_obj,
            as_of_ts=as_of_ts,
            pod_state_obj=pod_state_obj,
        )
        if state_store_obj.has_active_order_plan(
            pod_id_str=release_obj.pod_id_str,
            signal_timestamp_ts=order_plan_obj.signal_timestamp_ts,
            execution_policy_str=order_plan_obj.execution_policy_str,
        ):
            skipped_plan_count_int += 1
            reason_counter_obj["active_plan_exists"] += 1
            log_event(
                "build_plan_skipped",
                _build_release_log_payload_dict(
                    release_obj,
                    as_of_ts,
                    {"reason_code_str": "active_plan_exists"},
                ),
                log_path_str=log_path_str,
            )
            continue

        validate_order_intent_list(order_plan_obj.order_intent_list)
        inserted_order_plan_obj = state_store_obj.insert_order_plan(order_plan_obj)
        created_plan_count_int += 1
        log_event(
            "build_plan_created",
            _build_order_plan_log_payload_dict(
                release_obj,
                inserted_order_plan_obj,
                as_of_ts,
                {
                    "reason_code_str": "snapshot_ready",
                    "order_intent_count_int": len(inserted_order_plan_obj.order_intent_list),
                },
            ),
            log_path_str=log_path_str,
        )

    return {
        "created_plan_count_int": created_plan_count_int,
        "skipped_plan_count_int": skipped_plan_count_int,
        "reason_count_map_dict": dict(reason_counter_obj),
    }


def execute_order_plans(
    state_store_obj: LiveStateStore,
    broker_adapter_obj: BrokerAdapter,
    as_of_ts: datetime,
    env_mode_str: str,
    log_path_str: str = DEFAULT_LOG_PATH_STR,
) -> dict[str, object]:
    """Submit due frozen plans after broker/session/reconciliation checks pass."""
    submittable_plan_list = state_store_obj.get_submittable_order_plan_list(as_of_ts)
    submitted_plan_count_int = 0
    blocked_plan_count_int = 0
    reason_counter_obj: Counter[str] = Counter()
    visible_account_route_set = broker_adapter_obj.get_visible_account_route_set()

    for order_plan_obj in submittable_plan_list:
        order_plan_id_int = int(order_plan_obj.order_plan_id_int or 0)
        release_obj = state_store_obj.get_release_by_id(order_plan_obj.release_id_str)

        reason_code_str: str | None = None
        next_status_str: str | None = None

        if not release_obj.enabled_bool:
            reason_code_str = "release_disabled"
            next_status_str = "blocked"
        elif release_obj.mode_str != env_mode_str:
            reason_code_str = "env_mode_mismatch"
            next_status_str = "blocked"
        elif (
            state_store_obj.count_active_order_plans_for_window(
                pod_id_str=order_plan_obj.pod_id_str,
                submission_timestamp_ts=order_plan_obj.submission_timestamp_ts,
            )
            != 1
        ):
            reason_code_str = "submission_window_collision"
            next_status_str = "failed"
        elif state_store_obj.count_broker_orders_for_plan(order_plan_id_int) > 0:
            reason_code_str = "duplicate_submission_guard"
            next_status_str = "blocked"
        elif not state_store_obj.claim_order_plan_for_submission(order_plan_id_int):
            reason_code_str = "submission_claim_failed"
            next_status_str = None
        elif not broker_adapter_obj.is_session_ready(order_plan_obj.account_route_str):
            reason_code_str = "broker_not_ready"
            next_status_str = "frozen"
        elif (
            visible_account_route_set is not None
            and order_plan_obj.account_route_str not in visible_account_route_set
        ):
            reason_code_str = "account_not_visible"
            next_status_str = "blocked"
        else:
            session_mode_str = broker_adapter_obj.get_session_mode_str(order_plan_obj.account_route_str)
            if session_mode_str is not None and session_mode_str != release_obj.mode_str:
                reason_code_str = "session_mode_mismatch"
                next_status_str = "blocked"

        if reason_code_str is not None:
            if next_status_str is not None:
                state_store_obj.mark_order_plan_status(order_plan_id_int, next_status_str)
            blocked_plan_count_int += 1
            reason_counter_obj[reason_code_str] += 1
            log_event(
                "submit_plan_blocked",
                _build_order_plan_log_payload_dict(
                    release_obj,
                    order_plan_obj,
                    as_of_ts,
                    {"reason_code_str": reason_code_str},
                ),
                log_path_str=log_path_str,
            )
            continue

        pod_state_obj = _get_model_state_or_default(release_obj, state_store_obj, as_of_ts)
        broker_snapshot_obj = broker_adapter_obj.get_account_snapshot(order_plan_obj.account_route_str)
        reconciliation_result_obj = reconcile_account_state(
            model_position_map=pod_state_obj.position_amount_map,
            model_cash_float=pod_state_obj.cash_float,
            broker_snapshot_obj=broker_snapshot_obj,
        )
        state_store_obj.insert_reconciliation_snapshot(
            pod_id_str=order_plan_obj.pod_id_str,
            order_plan_id_int=order_plan_obj.order_plan_id_int,
            stage_str="pre_submit",
            reconciliation_result_obj=reconciliation_result_obj,
        )
        if not reconciliation_result_obj.passed_bool:
            state_store_obj.mark_order_plan_status(order_plan_id_int, "blocked")
            blocked_plan_count_int += 1
            reason_counter_obj["reconciliation_failed"] += 1
            log_event(
                "submit_plan_blocked",
                _build_order_plan_log_payload_dict(
                    release_obj,
                    order_plan_obj,
                    as_of_ts,
                    {
                        "reason_code_str": "reconciliation_failed",
                        "mismatch_dict": reconciliation_result_obj.mismatch_dict,
                    },
                ),
                log_path_str=log_path_str,
            )
            continue

        order_intent_row_list = state_store_obj.get_order_intent_row_list(order_plan_id_int)
        order_intent_id_list = [int(row_obj["order_intent_id_int"]) for row_obj in order_intent_row_list]
        validate_order_intent_list(order_plan_obj.order_intent_list)
        broker_order_request_list = build_broker_order_request_list(
            order_plan_id_int=order_plan_id_int,
            release_id_str=order_plan_obj.release_id_str,
            pod_id_str=order_plan_obj.pod_id_str,
            account_route_str=order_plan_obj.account_route_str,
            submission_key_str=str(order_plan_obj.submission_key_str or f"order_plan:{order_plan_id_int}"),
            order_intent_list=order_plan_obj.order_intent_list,
            order_intent_id_list=order_intent_id_list,
        )
        try:
            broker_order_record_list, broker_order_fill_list = broker_adapter_obj.submit_order_request_list(
                account_route_str=order_plan_obj.account_route_str,
                broker_order_request_list=broker_order_request_list,
                submitted_timestamp_ts=as_of_ts,
            )
            state_store_obj.insert_broker_order_record_list(broker_order_record_list)
            state_store_obj.insert_fill_list(broker_order_fill_list)
            state_store_obj.mark_order_plan_status(order_plan_id_int, "submitted")
            submitted_plan_count_int += 1
            log_event(
                "submit_plan_completed",
                _build_order_plan_log_payload_dict(
                    release_obj,
                    order_plan_obj,
                    as_of_ts,
                    {
                        "reason_code_str": "submitted",
                        "broker_order_count_int": len(broker_order_record_list),
                        "fill_count_int": len(broker_order_fill_list),
                    },
                ),
                log_path_str=log_path_str,
            )
        except Exception as exc:
            state_store_obj.mark_order_plan_status(order_plan_id_int, "blocked")
            log_event(
                "submit_plan_failed",
                _build_order_plan_log_payload_dict(
                    release_obj,
                    order_plan_obj,
                    as_of_ts,
                    {
                        "reason_code_str": "broker_submission_exception",
                        "error_str": str(exc),
                    },
                ),
                log_path_str=log_path_str,
            )
            raise

    return {
        "submitted_plan_count_int": submitted_plan_count_int,
        "blocked_plan_count_int": blocked_plan_count_int,
        "reason_count_map_dict": dict(reason_counter_obj),
    }


def post_execution_reconcile(
    state_store_obj: LiveStateStore,
    broker_adapter_obj: BrokerAdapter,
    as_of_ts: datetime,
    log_path_str: str = DEFAULT_LOG_PATH_STR,
) -> dict[str, object]:
    """Pull broker fills, update pod state, and complete submitted plans."""
    submitted_plan_list = state_store_obj.get_submitted_order_plan_list()
    completed_plan_count_int = 0

    for order_plan_obj in submitted_plan_list:
        release_obj = state_store_obj.get_release_by_id(order_plan_obj.release_id_str)
        broker_snapshot_obj = broker_adapter_obj.get_account_snapshot(order_plan_obj.account_route_str)
        broker_fill_list = broker_adapter_obj.get_recent_fill_list(
            account_route_str=order_plan_obj.account_route_str,
            since_timestamp_ts=order_plan_obj.submission_timestamp_ts,
        )
        state_store_obj.insert_fill_list(broker_fill_list)
        execution_fill_input_row_list = state_store_obj.get_execution_fill_input_row_list(
            int(order_plan_obj.order_plan_id_int or 0)
        )
        execution_quality_snapshot_obj = build_execution_quality_snapshot(
            order_plan_id_int=int(order_plan_obj.order_plan_id_int or 0),
            pod_id_str=order_plan_obj.pod_id_str,
            execution_fill_input_row_list=execution_fill_input_row_list,
        )
        state_store_obj.upsert_execution_quality_snapshot(execution_quality_snapshot_obj)
        post_reconciliation_result_obj = reconcile_account_state(
            model_position_map=broker_snapshot_obj.position_amount_map,
            model_cash_float=broker_snapshot_obj.cash_float,
            broker_snapshot_obj=broker_snapshot_obj,
        )
        state_store_obj.insert_reconciliation_snapshot(
            pod_id_str=order_plan_obj.pod_id_str,
            order_plan_id_int=order_plan_obj.order_plan_id_int,
            stage_str="post_execution",
            reconciliation_result_obj=post_reconciliation_result_obj,
        )
        state_store_obj.upsert_pod_state(
            PodState(
                pod_id_str=order_plan_obj.pod_id_str,
                user_id_str=release_obj.user_id_str,
                account_route_str=order_plan_obj.account_route_str,
                position_amount_map=broker_snapshot_obj.position_amount_map,
                cash_float=broker_snapshot_obj.cash_float,
                total_value_float=broker_snapshot_obj.total_value_float,
                strategy_state_dict=order_plan_obj.strategy_state_dict,
                updated_timestamp_ts=as_of_ts,
            )
        )
        state_store_obj.mark_order_plan_status(int(order_plan_obj.order_plan_id_int or 0), "completed")
        completed_plan_count_int += 1
        log_event(
            "post_execution_reconcile_completed",
            _build_order_plan_log_payload_dict(
                release_obj,
                order_plan_obj,
                as_of_ts,
                {
                    "reason_code_str": "completed",
                    "fill_count_int": len(broker_fill_list),
                },
            ),
            log_path_str=log_path_str,
        )

    return {"completed_plan_count_int": completed_plan_count_int}


def get_status_summary(
    state_store_obj: LiveStateStore,
    as_of_ts: datetime,
    releases_root_path_str: str,
) -> dict[str, object]:
    """Return a human-oriented snapshot of each enabled pod."""
    release_list = _load_release_list_and_sync(releases_root_path_str, state_store_obj)
    enabled_release_list = [release_obj for release_obj in release_list if release_obj.enabled_bool]
    pod_status_dict_list: list[dict[str, object]] = []

    for release_obj in enabled_release_list:
        latest_order_plan_obj = state_store_obj.get_latest_order_plan_for_pod(release_obj.pod_id_str)
        pod_state_obj = state_store_obj.get_pod_state(release_obj.pod_id_str)
        latest_fill_timestamp_str = None
        if latest_order_plan_obj is not None and latest_order_plan_obj.order_plan_id_int is not None:
            fill_row_dict_list = state_store_obj.get_fill_row_dict_list(latest_order_plan_obj.order_plan_id_int)
            if len(fill_row_dict_list) > 0:
                latest_fill_timestamp_str = str(fill_row_dict_list[-1]["fill_timestamp_str"])
        build_gate_dict = scheduler_utils.evaluate_build_gate_dict(release_obj, as_of_ts)

        next_action_str = "wait"
        reason_code_str = str(build_gate_dict["reason_code_str"])

        if latest_order_plan_obj is None:
            if bool(build_gate_dict["due_bool"]):
                next_action_str = "build_order_plan"
                reason_code_str = "ready_to_build"
        elif latest_order_plan_obj.status_str == "frozen":
            execution_window_dict = scheduler_utils.evaluate_execution_window_dict(
                latest_order_plan_obj.submission_timestamp_ts,
                as_of_ts,
            )
            if bool(execution_window_dict["due_bool"]):
                next_action_str = "execute_order_plan"
                reason_code_str = "ready_to_submit"
            else:
                reason_code_str = str(execution_window_dict["reason_code_str"])
        elif latest_order_plan_obj.status_str == "submitting":
            reason_code_str = "submission_in_progress"
        elif latest_order_plan_obj.status_str == "submitted":
            next_action_str = "post_execution_reconcile"
            reason_code_str = "waiting_for_post_execution_reconcile"
        elif latest_order_plan_obj.status_str in ("completed", "failed", "blocked"):
            if bool(build_gate_dict["due_bool"]):
                next_action_str = "build_order_plan"
                reason_code_str = "ready_to_build"
            else:
                next_action_str = "wait"

        pod_status_dict_list.append(
            {
                "release_id_str": release_obj.release_id_str,
                "pod_id_str": release_obj.pod_id_str,
                "account_route_str": release_obj.account_route_str,
                "session_calendar_id_str": release_obj.session_calendar_id_str,
                "signal_clock_str": release_obj.signal_clock_str,
                "execution_policy_str": release_obj.execution_policy_str,
                "latest_heartbeat_session_date_str": build_gate_dict["latest_heartbeat_session_date_str"],
                "latest_order_plan_status_str": None if latest_order_plan_obj is None else latest_order_plan_obj.status_str,
                "latest_signal_timestamp_str": (
                    None if latest_order_plan_obj is None else latest_order_plan_obj.signal_timestamp_ts.isoformat()
                ),
                "latest_submission_timestamp_str": (
                    None if latest_order_plan_obj is None else latest_order_plan_obj.submission_timestamp_ts.isoformat()
                ),
                "next_action_str": next_action_str,
                "reason_code_str": reason_code_str,
                "pod_state_updated_timestamp_str": (
                    None if pod_state_obj is None else pod_state_obj.updated_timestamp_ts.isoformat()
                ),
                "latest_fill_timestamp_str": latest_fill_timestamp_str,
            }
        )

    return {
        "as_of_timestamp_str": as_of_ts.isoformat(),
        "enabled_pod_count_int": len(enabled_release_list),
        "pod_status_dict_list": pod_status_dict_list,
    }


def get_execution_report_summary(
    state_store_obj: LiveStateStore,
    as_of_ts: datetime,
    releases_root_path_str: str,
) -> dict[str, object]:
    """Return raw fill data for the latest completed plan of each enabled pod."""
    release_list = _load_release_list_and_sync(releases_root_path_str, state_store_obj)
    execution_report_dict_list: list[dict[str, object]] = []

    for release_obj in release_list:
        if not release_obj.enabled_bool:
            continue
        latest_order_plan_obj = state_store_obj.get_latest_order_plan_for_pod(release_obj.pod_id_str)
        if latest_order_plan_obj is None or latest_order_plan_obj.order_plan_id_int is None:
            continue
        fill_row_dict_list = state_store_obj.get_fill_row_dict_list(latest_order_plan_obj.order_plan_id_int)
        if len(fill_row_dict_list) == 0:
            continue

        execution_report_dict_list.append(
            {
                "release_id_str": release_obj.release_id_str,
                "pod_id_str": release_obj.pod_id_str,
                "latest_order_plan_id_int": latest_order_plan_obj.order_plan_id_int,
                "latest_signal_timestamp_str": latest_order_plan_obj.signal_timestamp_ts.isoformat(),
                "fill_count_int": len(fill_row_dict_list),
                "fill_row_dict_list": fill_row_dict_list,
            }
        )

    return {
        "as_of_timestamp_str": as_of_ts.isoformat(),
        "execution_report_dict_list": execution_report_dict_list,
    }


def tick(
    state_store_obj: LiveStateStore,
    broker_adapter_obj: BrokerAdapter,
    as_of_ts: datetime,
    releases_root_path_str: str,
    env_mode_str: str,
    log_path_str: str = DEFAULT_LOG_PATH_STR,
) -> dict[str, object]:
    """Run one scheduler cycle: build due plans, submit due plans, then reconcile."""
    lease_owner_token_str = uuid.uuid4().hex
    lease_acquired_bool = state_store_obj.acquire_scheduler_lease(
        lease_name_str="tick",
        owner_token_str=lease_owner_token_str,
        expires_timestamp_ts=scheduler_utils.utc_now_ts() + timedelta(minutes=5),
    )
    if not lease_acquired_bool:
        detail_dict = {
            "lease_acquired_bool": False,
            "reason_code_str": "scheduler_lease_busy",
            "created_plan_count_int": 0,
            "skipped_plan_count_int": 0,
            "submitted_plan_count_int": 0,
            "blocked_plan_count_int": 0,
            "completed_plan_count_int": 0,
        }
        log_event(
            "tick_skipped",
            {
                "as_of_timestamp_str": as_of_ts.isoformat(),
                "reason_code_str": "scheduler_lease_busy",
            },
            log_path_str=log_path_str,
        )
        return detail_dict

    try:
        build_detail_dict = build_order_plans(
            state_store_obj=state_store_obj,
            as_of_ts=as_of_ts,
            releases_root_path_str=releases_root_path_str,
            log_path_str=log_path_str,
        )
        execute_detail_dict = execute_order_plans(
            state_store_obj=state_store_obj,
            broker_adapter_obj=broker_adapter_obj,
            as_of_ts=as_of_ts,
            env_mode_str=env_mode_str,
            log_path_str=log_path_str,
        )
        reconcile_detail_dict = post_execution_reconcile(
            state_store_obj=state_store_obj,
            broker_adapter_obj=broker_adapter_obj,
            as_of_ts=as_of_ts,
            log_path_str=log_path_str,
        )
        detail_dict = {
            "lease_acquired_bool": True,
            **build_detail_dict,
            **execute_detail_dict,
            **reconcile_detail_dict,
        }
        log_event(
            "tick_completed",
            {
                "as_of_timestamp_str": as_of_ts.isoformat(),
                **detail_dict,
            },
            log_path_str=log_path_str,
        )
        return detail_dict
    finally:
        state_store_obj.release_scheduler_lease(
            lease_name_str="tick",
            owner_token_str=lease_owner_token_str,
        )


def main(argv_list: list[str] | None = None) -> int:
    parser_obj = argparse.ArgumentParser(description="alpha.live runner")
    parser_obj.add_argument(
        "command_name_str",
        choices=(
            "build_order_plans",
            "execute_order_plans",
            "post_execution_reconcile",
            "tick",
            "status",
            "execution_report",
        ),
    )
    parser_obj.add_argument("--db-path", dest="db_path_str", default=DEFAULT_DB_PATH_STR)
    parser_obj.add_argument("--releases-root", dest="releases_root_path_str", default=DEFAULT_RELEASES_ROOT_PATH_STR)
    parser_obj.add_argument("--as-of-ts", dest="as_of_timestamp_str", default=None)
    parser_obj.add_argument("--mode", dest="env_mode_str", default="paper")
    parser_obj.add_argument("--log-path", dest="log_path_str", default=DEFAULT_LOG_PATH_STR)
    parser_obj.add_argument("--json", dest="json_output_bool", action="store_true")
    parsed_args_obj = parser_obj.parse_args(argv_list)

    state_store_obj = LiveStateStore(parsed_args_obj.db_path_str)
    job_run_id_int = state_store_obj.record_job_start(parsed_args_obj.command_name_str)
    as_of_ts = _parse_as_of_timestamp_ts(parsed_args_obj.as_of_timestamp_str)

    try:
        if parsed_args_obj.command_name_str == "build_order_plans":
            detail_dict = build_order_plans(
                state_store_obj=state_store_obj,
                as_of_ts=as_of_ts,
                releases_root_path_str=parsed_args_obj.releases_root_path_str,
                log_path_str=parsed_args_obj.log_path_str,
            )
        elif parsed_args_obj.command_name_str == "execute_order_plans":
            detail_dict = execute_order_plans(
                state_store_obj=state_store_obj,
                broker_adapter_obj=IBKRGatewayBrokerAdapter(),
                as_of_ts=as_of_ts,
                env_mode_str=parsed_args_obj.env_mode_str,
                log_path_str=parsed_args_obj.log_path_str,
            )
        elif parsed_args_obj.command_name_str == "post_execution_reconcile":
            detail_dict = post_execution_reconcile(
                state_store_obj=state_store_obj,
                broker_adapter_obj=IBKRGatewayBrokerAdapter(),
                as_of_ts=as_of_ts,
                log_path_str=parsed_args_obj.log_path_str,
            )
        elif parsed_args_obj.command_name_str == "tick":
            detail_dict = tick(
                state_store_obj=state_store_obj,
                broker_adapter_obj=IBKRGatewayBrokerAdapter(),
                as_of_ts=as_of_ts,
                releases_root_path_str=parsed_args_obj.releases_root_path_str,
                env_mode_str=parsed_args_obj.env_mode_str,
                log_path_str=parsed_args_obj.log_path_str,
            )
        elif parsed_args_obj.command_name_str == "execution_report":
            detail_dict = get_execution_report_summary(
                state_store_obj=state_store_obj,
                as_of_ts=as_of_ts,
                releases_root_path_str=parsed_args_obj.releases_root_path_str,
            )
        else:
            detail_dict = get_status_summary(
                state_store_obj=state_store_obj,
                as_of_ts=as_of_ts,
                releases_root_path_str=parsed_args_obj.releases_root_path_str,
            )

        state_store_obj.record_job_finish(job_run_id_int, "completed", detail_dict)
        if parsed_args_obj.json_output_bool:
            print(json.dumps(detail_dict, indent=2, sort_keys=True))
        else:
            print(_render_command_output_str(parsed_args_obj.command_name_str, detail_dict))
    except Exception as exc:  # pragma: no cover - CLI safety
        error_detail_dict = {"error_str": str(exc)}
        state_store_obj.record_job_finish(
            job_run_id_int,
            "failed",
            error_detail_dict,
        )
        log_event(
            "runner_failed",
            {
                "command_name_str": parsed_args_obj.command_name_str,
                "as_of_timestamp_str": as_of_ts.isoformat(),
                "error_str": str(exc),
            },
            log_path_str=parsed_args_obj.log_path_str,
        )
        raise

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
