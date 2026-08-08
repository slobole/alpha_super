"""Cross-pod schedule builders for Dashboard V3.

Surfaces the next ~4 scheduled events across the whole multi-pod book
sorted by target execution time. Operators live by the clock — knowing
"submit_vplan for dv2_caspersky in 1h 38m" is more useful than scanning
individual pod cards.

The pending-action list uses the persisted pod rows.  The Overview trading
window also surfaces the next month-end cycle while the scheduler is in its
normal ``not_month_end_session`` wait state.  It uses the same exchange
calendar helpers and execution-policy constants as the scheduler; none of the
values built here can advance state or submit an order.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from alpha.live import scheduler_utils


DEFAULT_SCHEDULE_LIMIT_INT = 6
WINDOW_SEVERITY_RANK_DICT = {"red": 0, "yellow": 1, "gray": 2, "green": 3}


@dataclass
class ScheduleEntry:
    pod_id_str: str
    mode_str: str
    action_str: str
    target_timestamp_str: str | None
    relative_str: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "pod_id_str": self.pod_id_str,
            "mode_str": self.mode_str,
            "action_str": self.action_str,
            "target_timestamp_str": self.target_timestamp_str,
            "relative_str": self.relative_str,
        }


@dataclass
class TradingWindow:
    has_data_bool: bool = False
    severity_str: str = "gray"
    status_label_str: str = "Cannot verify"
    detail_str: str = "No live trading window is available."
    signal_timestamp_str: str | None = None
    submission_timestamp_str: str | None = None
    target_timestamp_str: str | None = None
    relative_str: str = "—"
    trading_session_count_int: int | None = None
    action_required_bool: bool = False
    action_str: str | None = None
    reason_code_str: str | None = None
    norgate_label_str: str = "—"
    pod_id_str_list: list[str] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "has_data_bool": self.has_data_bool,
            "severity_str": self.severity_str,
            "status_label_str": self.status_label_str,
            "detail_str": self.detail_str,
            "signal_timestamp_str": self.signal_timestamp_str,
            "submission_timestamp_str": self.submission_timestamp_str,
            "target_timestamp_str": self.target_timestamp_str,
            "relative_str": self.relative_str,
            "trading_session_count_int": self.trading_session_count_int,
            "action_required_bool": self.action_required_bool,
            "action_str": self.action_str,
            "reason_code_str": self.reason_code_str,
            "norgate_label_str": self.norgate_label_str,
            "pod_id_str_list": list(self.pod_id_str_list or []),
        }


def build_schedule_entry_list(
    summary_dict: dict[str, Any],
    now_dt: datetime | None = None,
    limit_int: int = DEFAULT_SCHEDULE_LIMIT_INT,
    mode_str: str | None = None,
) -> list[ScheduleEntry]:
    if now_dt is None:
        now_dt = datetime.now(timezone.utc)
    pod_row_dict_list = [
        row_dict
        for row_dict in summary_dict.get("pod_row_dict_list") or []
        if mode_str is None or str(row_dict.get("mode_str") or "") == mode_str
    ]
    candidate_list: list[ScheduleEntry] = []
    for row_dict in pod_row_dict_list:
        action_str = row_dict.get("next_action_str")
        if not action_str or action_str == "wait":
            continue
        target_timestamp_str = (
            row_dict.get("latest_vplan_target_execution_timestamp_str")
            or row_dict.get("latest_decision_plan_target_execution_timestamp_str")
        )
        candidate_list.append(
            ScheduleEntry(
                pod_id_str=str(row_dict.get("pod_id_str") or "?"),
                mode_str=str(row_dict.get("mode_str") or "?"),
                action_str=str(action_str),
                target_timestamp_str=str(target_timestamp_str) if target_timestamp_str else None,
                relative_str=_format_relative_time_str(target_timestamp_str, now_dt),
            )
        )
    candidate_list.sort(key=lambda entry_obj: (
        entry_obj.target_timestamp_str or "9999",
        entry_obj.pod_id_str,
    ))
    return candidate_list[:limit_int]


def build_next_trading_window(
    summary_dict: dict[str, Any],
    *,
    mode_str: str,
    now_dt: datetime | None = None,
) -> TradingWindow:
    """Return the nearest actionable or scheduled monthly trading window."""
    now_dt = _aware_utc_dt(now_dt or datetime.now(timezone.utc))
    pod_row_dict_list = [
        row_dict
        for row_dict in summary_dict.get("pod_row_dict_list") or []
        if str(row_dict.get("mode_str") or "") == mode_str
    ]
    if not pod_row_dict_list:
        return TradingWindow(
            detail_str=f"No {mode_str} pods are enabled.",
            pod_id_str_list=[],
        )

    active_row_dict_list = [
        row_dict
        for row_dict in pod_row_dict_list
        if _row_has_active_window_bool(row_dict, now_dt)
    ]
    if active_row_dict_list:
        selected_row_dict = min(
            active_row_dict_list,
            key=lambda row_dict: (
                _row_action_priority_int(row_dict),
                _row_severity_rank_int(row_dict),
                str(_target_timestamp_obj(row_dict) or "9999"),
            ),
        )
        return _build_active_trading_window(selected_row_dict, active_row_dict_list, now_dt)

    monthly_row_dict_list = [
        row_dict
        for row_dict in pod_row_dict_list
        if scheduler_utils.normalize_signal_clock_str(
            str(row_dict.get("signal_clock_str") or "")
        )
        == "month_end_snapshot_ready"
        and str(row_dict.get("execution_policy_str") or "")
        == "next_month_first_open"
    ]
    daily_row_dict_list = [
        row_dict
        for row_dict in pod_row_dict_list
        if scheduler_utils.normalize_signal_clock_str(
            str(row_dict.get("signal_clock_str") or "")
        )
        == "eod_snapshot_ready"
        and str(row_dict.get("execution_policy_str") or "")
        == "next_open_moo"
    ]
    if not monthly_row_dict_list and not daily_row_dict_list:
        return TradingWindow(
            detail_str="No future trading window can be derived from the enabled pods.",
            pod_id_str_list=[
                str(row_dict.get("pod_id_str") or "?")
                for row_dict in pod_row_dict_list
            ],
        )

    # *** CRITICAL *** display-only calendar parity: signal, submission and
    # execution timestamps must use the same exchange-calendar helpers and MOO
    # lead time as the live scheduler.  This code never advances scheduler state.
    candidate_window_obj_list = [
        _build_monthly_wait_window(row_dict, now_dt)
        for row_dict in monthly_row_dict_list
    ] + [
        _build_daily_wait_window(row_dict, now_dt)
        for row_dict in daily_row_dict_list
    ]
    valid_window_obj_list = [
        window_obj
        for window_obj in candidate_window_obj_list
        if window_obj.has_data_bool and window_obj.signal_timestamp_str
    ]
    if not valid_window_obj_list:
        return TradingWindow(
            detail_str="The next signal session could not be resolved.",
            pod_id_str_list=[
                str(row_dict.get("pod_id_str") or "?")
                for row_dict in monthly_row_dict_list + daily_row_dict_list
            ],
        )

    selected_window_obj = min(
        valid_window_obj_list,
        key=lambda window_obj: (
            str(window_obj.signal_timestamp_str),
            WINDOW_SEVERITY_RANK_DICT.get(window_obj.severity_str, 9),
        ),
    )
    matching_pod_id_str_list = sorted(
        {
            pod_id_str
            for window_obj in valid_window_obj_list
            if window_obj.signal_timestamp_str == selected_window_obj.signal_timestamp_str
            and window_obj.target_timestamp_str == selected_window_obj.target_timestamp_str
            for pod_id_str in window_obj.pod_id_str_list or []
        }
    )
    selected_window_obj.pod_id_str_list = matching_pod_id_str_list
    pod_label_str = "pod" if len(matching_pod_id_str_list) == 1 else "pods"
    selected_window_obj.detail_str = (
        f"{selected_window_obj.detail_str} "
        f"{len(matching_pod_id_str_list)} {mode_str} {pod_label_str} share this window."
    )
    return selected_window_obj


def _row_has_active_window_bool(row_dict: dict[str, Any], now_dt: datetime) -> bool:
    action_str = str(row_dict.get("next_action_str") or "")
    if action_str and action_str not in ("wait", "no_db"):
        return True
    reason_code_str = str(row_dict.get("reason_code_str") or "")
    if reason_code_str == "not_month_end_session":
        return False
    target_timestamp_obj = _target_timestamp_obj(row_dict)
    if target_timestamp_obj is None:
        return False
    try:
        target_dt = _parse_iso_datetime(str(target_timestamp_obj))
    except ValueError:
        return False
    return _aware_utc_dt(target_dt) >= now_dt


def _build_active_trading_window(
    selected_row_dict: dict[str, Any],
    active_row_dict_list: list[dict[str, Any]],
    now_dt: datetime,
) -> TradingWindow:
    action_str = str(selected_row_dict.get("next_action_str") or "wait")
    if (
        action_str == "build_decision_plan"
        and scheduler_utils.normalize_signal_clock_str(
            str(selected_row_dict.get("signal_clock_str") or "")
        )
        == "month_end_snapshot_ready"
        and str(selected_row_dict.get("execution_policy_str") or "")
        == "next_month_first_open"
    ):
        trading_window_obj = _build_monthly_wait_window(selected_row_dict, now_dt)
        trading_window_obj.action_str = action_str
        trading_window_obj.action_required_bool = True
        if trading_window_obj.status_label_str == "No operator action required":
            trading_window_obj.status_label_str = "Operator action required"
        return trading_window_obj

    target_timestamp_obj = _target_timestamp_obj(selected_row_dict)
    target_timestamp_str = str(target_timestamp_obj) if target_timestamp_obj else None
    target_match_str = target_timestamp_str or ""
    matching_pod_id_str_list = sorted(
        {
            str(row_dict.get("pod_id_str") or "?")
            for row_dict in active_row_dict_list
            if str(_target_timestamp_obj(row_dict) or "") == target_match_str
        }
    )
    required_action_dict = selected_row_dict.get("required_action_dict") or {}
    required_severity_str = str(required_action_dict.get("severity_str") or "")
    if not required_severity_str and action_str not in ("wait", ""):
        required_severity_str = "yellow"
    action_required_bool = _operator_action_required_bool(required_action_dict) or (
        not required_action_dict and action_str not in ("wait", "")
    )
    previous_cycle_action_bool = _required_action_targets_previous_cycle_bool(
        selected_row_dict
    )
    signal_timestamp_str = (
        None
        if previous_cycle_action_bool
        else _optional_text_str(
            selected_row_dict.get("latest_decision_signal_timestamp_str")
        )
    )
    submission_timestamp_str = _optional_text_str(
        (
            selected_row_dict.get("latest_vplan_submission_timestamp_str")
            if _use_latest_vplan_timestamps_bool(selected_row_dict)
            else selected_row_dict.get("latest_decision_plan_submission_timestamp_str")
        )
    )
    missing_stage_evidence_bool = (
        action_str == "wait"
        and bool(target_timestamp_str)
        and required_severity_str not in ("red", "yellow")
        and (not signal_timestamp_str or not submission_timestamp_str)
    )
    if missing_stage_evidence_bool:
        required_severity_str = "gray"
        action_required_bool = False
    return TradingWindow(
        has_data_bool=True,
        severity_str=(
            required_severity_str
            if required_severity_str in ("red", "yellow")
            else "gray"
        ),
        status_label_str=str(
            ("Cannot verify" if missing_stage_evidence_bool else None)
            or required_action_dict.get("label_str")
            or ("Operator action required" if action_required_bool else "On schedule")
        ),
        detail_str=(
            "Persisted target is missing signal or submission evidence."
            if missing_stage_evidence_bool
            else str(
                required_action_dict.get("detail_str")
                or required_action_dict.get("reason_str")
                or "The current trading cycle is active."
            )
        ),
        signal_timestamp_str=signal_timestamp_str,
        submission_timestamp_str=submission_timestamp_str,
        target_timestamp_str=target_timestamp_str,
        relative_str=(
            _format_relative_time_str(target_timestamp_str, now_dt)
            if target_timestamp_str
            else "—"
        ),
        action_required_bool=action_required_bool,
        action_str=action_str,
        reason_code_str=_optional_text_str(selected_row_dict.get("reason_code_str")),
        norgate_label_str=(
            f"Snapshot {selected_row_dict.get('latest_decision_norgate_snapshot_date_str')}"
            if selected_row_dict.get("latest_decision_norgate_snapshot_date_str")
            else "Current-cycle gate shown in pod detail"
        ),
        pod_id_str_list=matching_pod_id_str_list,
    )


def _build_monthly_wait_window(
    row_dict: dict[str, Any],
    now_dt: datetime,
) -> TradingWindow:
    calendar_id_str = str(row_dict.get("session_calendar_id_str") or "")
    if not calendar_id_str:
        return TradingWindow(pod_id_str_list=[str(row_dict.get("pod_id_str") or "?")])
    reason_code_str = str(row_dict.get("reason_code_str") or "")
    signal_session_label_ts = _open_month_end_cycle_session_label_ts(
        now_dt,
        calendar_id_str,
    )
    current_cycle_wait_bool = signal_session_label_ts is not None
    if signal_session_label_ts is None:
        signal_session_label_ts = _next_month_end_session_label_ts(
            now_dt,
            calendar_id_str,
        )
    if signal_session_label_ts is None:
        return TradingWindow(pod_id_str_list=[str(row_dict.get("pod_id_str") or "?")])
    signal_timestamp_dt = scheduler_utils.get_session_close_timestamp_ts(
        signal_session_label_ts,
        calendar_id_str,
    )
    execution_session_label_ts = scheduler_utils.get_first_next_month_session_label_ts(
        signal_session_label_ts,
        calendar_id_str,
    )
    target_timestamp_dt = scheduler_utils.get_session_open_timestamp_ts(
        execution_session_label_ts,
        calendar_id_str,
    )
    submission_timestamp_dt = target_timestamp_dt - timedelta(
        seconds=scheduler_utils.DEFAULT_OPEN_SUBMISSION_LEAD_SECONDS_INT
    )
    calendar_obj = scheduler_utils.get_exchange_calendar_obj(calendar_id_str)
    market_now_dt = scheduler_utils.to_market_timestamp_ts(now_dt, calendar_id_str)
    trading_session_count_int = 0
    if signal_timestamp_dt > market_now_dt:
        count_start_date_obj = market_now_dt.date()
        current_session_label_ts = scheduler_utils.session_label_from_timestamp_ts(
            now_dt,
            calendar_id_str,
        )
        if current_session_label_ts is not None:
            current_session_close_dt = scheduler_utils.get_session_close_timestamp_ts(
                current_session_label_ts,
                calendar_id_str,
            )
            if market_now_dt >= current_session_close_dt:
                count_start_date_obj += timedelta(days=1)
        trading_session_count_int = len(
            calendar_obj.sessions_in_range(
                pd.Timestamp(count_start_date_obj),
                signal_session_label_ts,
            )
        )
    required_action_dict = row_dict.get("required_action_dict") or {}
    required_severity_str = str(required_action_dict.get("severity_str") or "")
    status_label_str = str(
        required_action_dict.get("label_str") or "No operator action required"
    )
    detail_str = str(
        required_action_dict.get("detail_str")
        or required_action_dict.get("reason_str")
        or (
            "Waiting for the current month-end cycle to advance."
            if current_cycle_wait_bool
            else "Waiting for the scheduled month-end signal session."
        )
    )
    stale_preclose_state_bool = (
        current_cycle_wait_bool
        and reason_code_str == "not_month_end_session"
        and required_severity_str not in ("red", "yellow")
    )
    if stale_preclose_state_bool:
        required_severity_str = "gray"
        status_label_str = "Awaiting scheduler refresh"
        detail_str = (
            "Month-end just closed; waiting for persisted scheduler and "
            "Norgate state to refresh."
        )
    return TradingWindow(
        has_data_bool=True,
        severity_str=(
            required_severity_str
            if required_severity_str in ("red", "yellow")
            else "gray"
        ),
        status_label_str=status_label_str,
        detail_str=detail_str,
        signal_timestamp_str=signal_timestamp_dt.isoformat(),
        submission_timestamp_str=submission_timestamp_dt.isoformat(),
        target_timestamp_str=target_timestamp_dt.isoformat(),
        relative_str=(
            _format_relative_time_str(target_timestamp_dt.isoformat(), now_dt)
            if current_cycle_wait_bool
            else (
                f"in {trading_session_count_int} trading "
                f"{'session' if trading_session_count_int == 1 else 'sessions'}"
            )
        ),
        trading_session_count_int=trading_session_count_int,
        action_required_bool=(
            False
            if stale_preclose_state_bool
            else _operator_action_required_bool(required_action_dict)
        ),
        action_str="wait",
        reason_code_str=reason_code_str or "not_month_end_session",
        norgate_label_str=f"Required for {signal_session_label_ts.date().isoformat()}",
        pod_id_str_list=[str(row_dict.get("pod_id_str") or "?")],
    )


def _build_daily_wait_window(
    row_dict: dict[str, Any],
    now_dt: datetime,
) -> TradingWindow:
    """Future window for the daily cycle: eod_snapshot_ready → next_open_moo.

    Signal is a session close (the EOD snapshot follows it), execution is the
    NEXT session's open (MOO), submission is that open minus the same lead
    the live scheduler uses. Two cases: after a close whose next open has not
    passed, the current cycle is still pending; otherwise the coming close is
    the next signal. Same *** CRITICAL *** display-only calendar parity as
    the monthly builder — this never advances scheduler state.
    """
    calendar_id_str = str(row_dict.get("session_calendar_id_str") or "")
    if not calendar_id_str:
        return TradingWindow(pod_id_str_list=[str(row_dict.get("pod_id_str") or "?")])
    reason_code_str = str(row_dict.get("reason_code_str") or "")
    market_now_dt = scheduler_utils.to_market_timestamp_ts(now_dt, calendar_id_str)

    # Open cycle: the latest completed close whose MOO execution is still ahead.
    signal_session_label_ts = scheduler_utils.get_latest_completed_session_label_ts(
        now_dt,
        calendar_id_str,
        snapshot_ready_buffer_minutes_int=0,
    )
    current_cycle_wait_bool = False
    if signal_session_label_ts is not None:
        execution_session_label_ts = scheduler_utils.get_next_session_label_ts(
            signal_session_label_ts,
            calendar_id_str,
        )
        target_timestamp_dt = scheduler_utils.get_session_open_timestamp_ts(
            execution_session_label_ts,
            calendar_id_str,
        )
        if not scheduler_utils.is_execution_window_expired_bool(
            "next_open_moo",
            target_timestamp_dt,
            market_now_dt,
        ):
            current_cycle_wait_bool = True
        else:
            # That cycle is done; the next signal is the coming session close.
            signal_session_label_ts = scheduler_utils.get_next_session_label_ts(
                signal_session_label_ts,
                calendar_id_str,
            )
    if signal_session_label_ts is None:
        return TradingWindow(pod_id_str_list=[str(row_dict.get("pod_id_str") or "?")])
    if not current_cycle_wait_bool:
        execution_session_label_ts = scheduler_utils.get_next_session_label_ts(
            signal_session_label_ts,
            calendar_id_str,
        )
        target_timestamp_dt = scheduler_utils.get_session_open_timestamp_ts(
            execution_session_label_ts,
            calendar_id_str,
        )
    signal_timestamp_dt = scheduler_utils.get_session_close_timestamp_ts(
        signal_session_label_ts,
        calendar_id_str,
    )
    submission_timestamp_dt = target_timestamp_dt - timedelta(
        seconds=scheduler_utils.DEFAULT_OPEN_SUBMISSION_LEAD_SECONDS_INT
    )
    required_action_dict = row_dict.get("required_action_dict") or {}
    required_severity_str = str(required_action_dict.get("severity_str") or "")
    return TradingWindow(
        has_data_bool=True,
        severity_str=(
            required_severity_str
            if required_severity_str in ("red", "yellow")
            else "gray"
        ),
        status_label_str=str(
            required_action_dict.get("label_str") or "No operator action required"
        ),
        detail_str=str(
            required_action_dict.get("detail_str")
            or required_action_dict.get("reason_str")
            or (
                "EOD signal captured; waiting for the next open (MOO)."
                if current_cycle_wait_bool
                else "Waiting for the next session close and EOD snapshot."
            )
        ),
        signal_timestamp_str=signal_timestamp_dt.isoformat(),
        submission_timestamp_str=submission_timestamp_dt.isoformat(),
        target_timestamp_str=target_timestamp_dt.isoformat(),
        relative_str=_format_relative_time_str(
            target_timestamp_dt.isoformat(), now_dt
        ),
        trading_session_count_int=0 if current_cycle_wait_bool else 1,
        action_required_bool=_operator_action_required_bool(required_action_dict),
        action_str="wait",
        reason_code_str=reason_code_str or "awaiting_eod_cycle",
        norgate_label_str=f"Required for {signal_session_label_ts.date().isoformat()}",
        pod_id_str_list=[str(row_dict.get("pod_id_str") or "?")],
    )


def _operator_action_required_bool(required_action_dict: dict[str, Any]) -> bool:
    severity_str = str(required_action_dict.get("severity_str") or "")
    label_str = str(required_action_dict.get("label_str") or "")
    if severity_str == "red":
        return True
    return severity_str == "yellow" and not label_str.startswith(("Wait ", "Waiting "))


def _row_severity_rank_int(row_dict: dict[str, Any]) -> int:
    required_action_dict = row_dict.get("required_action_dict") or {}
    severity_str = str(
        required_action_dict.get("severity_str")
        or row_dict.get("health_str")
        or "gray"
    )
    return WINDOW_SEVERITY_RANK_DICT.get(severity_str, 9)


def _row_action_priority_int(row_dict: dict[str, Any]) -> int:
    required_action_dict = row_dict.get("required_action_dict") or {}
    action_str = str(row_dict.get("next_action_str") or "")
    action_required_bool = _operator_action_required_bool(required_action_dict) or (
        not required_action_dict and action_str not in ("wait", "")
    )
    return 0 if action_required_bool else 1


def _next_month_end_session_label_ts(
    now_dt: datetime,
    calendar_id_str: str,
) -> pd.Timestamp | None:
    calendar_obj = scheduler_utils.get_exchange_calendar_obj(calendar_id_str)
    market_now_dt = scheduler_utils.to_market_timestamp_ts(now_dt, calendar_id_str)
    session_label_list = calendar_obj.sessions_in_range(
        pd.Timestamp(market_now_dt.date()),
        pd.Timestamp(market_now_dt.date() + timedelta(days=70)),
    )
    for session_label_ts in session_label_list:
        signal_close_dt = scheduler_utils.get_session_close_timestamp_ts(
            session_label_ts,
            calendar_id_str,
        )
        if signal_close_dt <= market_now_dt:
            continue
        if scheduler_utils.is_last_session_of_month_bool(
            session_label_ts,
            calendar_id_str,
        ):
            return session_label_ts
    return None


def _open_month_end_cycle_session_label_ts(
    now_dt: datetime,
    calendar_id_str: str,
) -> pd.Timestamp | None:
    signal_session_label_ts = (
        scheduler_utils.get_latest_completed_month_end_session_label_ts(
            now_dt,
            calendar_id_str,
            snapshot_ready_buffer_minutes_int=0,
        )
    )
    if signal_session_label_ts is None:
        return None
    execution_session_label_ts = scheduler_utils.get_first_next_month_session_label_ts(
        signal_session_label_ts,
        calendar_id_str,
    )
    target_timestamp_dt = scheduler_utils.get_session_open_timestamp_ts(
        execution_session_label_ts,
        calendar_id_str,
    )
    if scheduler_utils.is_execution_window_expired_bool(
        "next_month_first_open",
        target_timestamp_dt,
        now_dt,
    ):
        return None
    return signal_session_label_ts


def _target_timestamp_obj(row_dict: dict[str, Any]) -> Any:
    if _use_latest_vplan_timestamps_bool(row_dict):
        return (
            row_dict.get("latest_vplan_target_execution_timestamp_str")
            or row_dict.get("latest_decision_plan_target_execution_timestamp_str")
            or row_dict.get("missed_target_execution_timestamp_str")
        )
    return (
        row_dict.get("latest_decision_plan_target_execution_timestamp_str")
        or row_dict.get("missed_target_execution_timestamp_str")
    )


def _use_latest_vplan_timestamps_bool(row_dict: dict[str, Any]) -> bool:
    if _required_action_targets_previous_cycle_bool(row_dict):
        return True
    vplan_match_obj = row_dict.get("latest_vplan_is_for_latest_decision_bool")
    vplan_cycle_role_str = str(row_dict.get("latest_vplan_cycle_role_str") or "")
    if vplan_match_obj is True or vplan_cycle_role_str == "current":
        return True
    if vplan_match_obj is False or vplan_cycle_role_str in ("previous", "none"):
        return False
    return bool(
        row_dict.get("latest_vplan_submission_timestamp_str")
        or row_dict.get("latest_vplan_target_execution_timestamp_str")
    )


def _required_action_targets_previous_cycle_bool(row_dict: dict[str, Any]) -> bool:
    required_action_dict = row_dict.get("required_action_dict") or {}
    return str(required_action_dict.get("label_str") or "").startswith("Review previous")


def _optional_text_str(value_obj: Any) -> str | None:
    return str(value_obj) if value_obj not in (None, "") else None


def _aware_utc_dt(value_dt: datetime) -> datetime:
    if value_dt.tzinfo is None:
        return value_dt.replace(tzinfo=timezone.utc)
    return value_dt.astimezone(timezone.utc)


def _format_relative_time_str(target_timestamp_str: Any, now_dt: datetime) -> str:
    if not target_timestamp_str:
        return "—"
    try:
        target_dt = _parse_iso_datetime(str(target_timestamp_str))
    except ValueError:
        return str(target_timestamp_str)
    if target_dt.tzinfo is None:
        target_dt = target_dt.replace(tzinfo=timezone.utc)
    delta_seconds_int = int((target_dt - now_dt).total_seconds())
    if delta_seconds_int <= 0:
        absolute_seconds_int = -delta_seconds_int
        return f"{_format_duration_str(absolute_seconds_int)} ago"
    return f"in {_format_duration_str(delta_seconds_int)}"


def _parse_iso_datetime(text_str: str) -> datetime:
    return datetime.fromisoformat(text_str.replace("Z", "+00:00"))


def _format_duration_str(seconds_int: int) -> str:
    if seconds_int < 60:
        return f"{seconds_int}s"
    if seconds_int < 3600:
        minutes_int = seconds_int // 60
        seconds_remaining_int = seconds_int % 60
        if seconds_remaining_int == 0:
            return f"{minutes_int}m"
        return f"{minutes_int}m {seconds_remaining_int}s"
    hours_int = seconds_int // 3600
    minutes_remaining_int = (seconds_int % 3600) // 60
    if minutes_remaining_int == 0:
        return f"{hours_int}h"
    return f"{hours_int}h {minutes_remaining_int}m"
