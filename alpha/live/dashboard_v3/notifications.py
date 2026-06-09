"""Red-transition Discord notifications for Dashboard V3.

A pod's severity moving from {gray, green, yellow} → red fires exactly one
webhook. Subsequent polls re-read the persisted severity state; no second
notification is sent until the pod recovers to non-red and then turns red
again. Operators can close the laptop and trust the system to surface
genuine new problems.

Stdlib-only HTTP — no `requests` dep added. Missing webhook URL → silent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import threading
from typing import Any, Callable
import urllib.error
import urllib.request

from alpha.live.dashboard_v3.data import _effective_severity_str
from alpha.live.ops_report import apply_consumer_staleness_dict


DEFAULT_NOTIFICATION_STATE_PATH_STR = "alpha/live/logs/notification_state.json"
DEFAULT_WEBHOOK_TIMEOUT_FLOAT = 3.0
DISCORD_WEBHOOK_ENV_VAR_NAME_STR = "ALPHA_DISCORD_WEBHOOK_URL"
INSPECTOR_NOTIFICATION_KEY_STR = "__inspector__"


@dataclass
class NotificationState:
    pod_severity_map_dict: dict[str, str] = field(default_factory=dict)
    last_updated_str: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "pod_severity_map_dict": dict(self.pod_severity_map_dict),
            "last_updated_str": self.last_updated_str,
        }


@dataclass
class NotificationStateStore:
    state_path_str: str = DEFAULT_NOTIFICATION_STATE_PATH_STR
    _lock_obj: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def load_state(self) -> NotificationState:
        state_path_obj = Path(self.state_path_str)
        if not state_path_obj.exists():
            return NotificationState()
        try:
            payload_obj = json.loads(state_path_obj.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return NotificationState()
        if not isinstance(payload_obj, dict):
            return NotificationState()
        raw_map_obj = payload_obj.get("pod_severity_map_dict") or {}
        if not isinstance(raw_map_obj, dict):
            raw_map_obj = {}
        return NotificationState(
            pod_severity_map_dict={str(k): str(v) for k, v in raw_map_obj.items()},
            last_updated_str=str(payload_obj.get("last_updated_str") or ""),
        )

    def save_state(self, state_obj: NotificationState) -> None:
        state_path_obj = Path(self.state_path_str)
        state_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_obj:
            state_path_obj.write_text(
                json.dumps(state_obj.as_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )


WebhookPosterFn = Callable[[str, dict[str, Any]], bool]


def post_discord_webhook_bool(webhook_url_str: str, payload_dict: dict[str, Any]) -> bool:
    if not webhook_url_str:
        return False
    request_body_bytes = json.dumps(payload_dict).encode("utf-8")
    request_obj = urllib.request.Request(
        webhook_url_str,
        data=request_body_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request_obj, timeout=DEFAULT_WEBHOOK_TIMEOUT_FLOAT) as response_obj:
            return 200 <= response_obj.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def discord_webhook_url_from_env_str() -> str:
    return os.environ.get(DISCORD_WEBHOOK_ENV_VAR_NAME_STR, "")


@dataclass
class NotificationFiredRecord:
    pod_id_str: str
    mode_str: str
    previous_severity_str: str
    reason_str: str
    delivered_bool: bool


def check_and_notify_for_red_transitions(
    summary_dict: dict[str, Any],
    *,
    state_store_obj: NotificationStateStore,
    webhook_url_str: str,
    webhook_poster_fn: WebhookPosterFn = post_discord_webhook_bool,
) -> list[NotificationFiredRecord]:
    """Compare each pod's current severity against the persisted last-known
    severity. Fire a webhook for every pod that just transitioned to red.
    Always update the state file even when the webhook URL is empty so the
    first time the operator configures the env var we don't fire a flood of
    backfilled alerts.
    """
    fired_list: list[NotificationFiredRecord] = []
    state_obj = state_store_obj.load_state()
    new_severity_map_dict: dict[str, str] = {}
    for row_dict in summary_dict.get("pod_row_dict_list") or []:
        pod_id_str = str(row_dict.get("pod_id_str") or "")
        if not pod_id_str:
            continue
        current_severity_str = _effective_severity_str(row_dict)
        new_severity_map_dict[pod_id_str] = current_severity_str
        previous_severity_str = state_obj.pod_severity_map_dict.get(pod_id_str, "")
        if current_severity_str == "red" and previous_severity_str != "red":
            payload_dict = _build_discord_payload_dict(row_dict, previous_severity_str)
            delivered_bool = bool(webhook_url_str) and webhook_poster_fn(webhook_url_str, payload_dict)
            fired_list.append(
                NotificationFiredRecord(
                    pod_id_str=pod_id_str,
                    mode_str=str(row_dict.get("mode_str") or ""),
                    previous_severity_str=previous_severity_str or "unknown",
                    reason_str=str(
                        (row_dict.get("debug_summary_dict") or {}).get("primary_reason_str")
                        or (row_dict.get("required_action_dict") or {}).get("reason_str")
                        or row_dict.get("reason_code_str")
                        or ""
                    ),
                    delivered_bool=delivered_bool,
                )
            )
    raw_inspector_report_dict = summary_dict.get("inspector_report_dict") or {}
    inspector_report_dict = (
        apply_consumer_staleness_dict(raw_inspector_report_dict)
        if raw_inspector_report_dict
        else {}
    )
    inspector_severity_str = str(inspector_report_dict.get("overall_severity_str") or "")
    if inspector_severity_str:
        new_severity_map_dict[INSPECTOR_NOTIFICATION_KEY_STR] = inspector_severity_str
        previous_severity_str = state_obj.pod_severity_map_dict.get(
            INSPECTOR_NOTIFICATION_KEY_STR,
            "",
        )
        if inspector_severity_str == "red" and previous_severity_str != "red":
            payload_dict = _build_inspector_payload_dict(
                inspector_report_dict,
                previous_severity_str,
            )
            delivered_bool = bool(webhook_url_str) and webhook_poster_fn(webhook_url_str, payload_dict)
            fired_list.append(
                NotificationFiredRecord(
                    pod_id_str=INSPECTOR_NOTIFICATION_KEY_STR,
                    mode_str=str(inspector_report_dict.get("mode_str") or "all"),
                    previous_severity_str=previous_severity_str or "unknown",
                    reason_str=str(inspector_report_dict.get("overall_reason_str") or ""),
                    delivered_bool=delivered_bool,
                )
            )
    state_obj.pod_severity_map_dict = new_severity_map_dict
    state_obj.last_updated_str = str(summary_dict.get("as_of_timestamp_str") or "")
    state_store_obj.save_state(state_obj)
    return fired_list


def _build_discord_payload_dict(
    row_dict: dict[str, Any], previous_severity_str: str
) -> dict[str, Any]:
    pod_id_str = str(row_dict.get("pod_id_str") or "?")
    mode_str = str(row_dict.get("mode_str") or "?")
    action_label_str = str(
        (row_dict.get("required_action_dict") or {}).get("label_str")
        or row_dict.get("next_action_str")
        or "needs operator review"
    )
    reason_str = str(
        (row_dict.get("debug_summary_dict") or {}).get("primary_reason_str")
        or (row_dict.get("required_action_dict") or {}).get("reason_str")
        or row_dict.get("reason_code_str")
        or "no reason recorded"
    )
    previous_label_str = previous_severity_str or "unknown"
    message_str = (
        f":rotating_light: **{mode_str.upper()} / {pod_id_str}** turned **RED** "
        f"(was `{previous_label_str}`).\n"
        f"Required: **{action_label_str}**\n"
        f"Reason: {reason_str}"
    )
    return {"content": message_str}


def _build_inspector_payload_dict(
    inspector_report_dict: dict[str, Any],
    previous_severity_str: str,
) -> dict[str, Any]:
    mode_str = str(inspector_report_dict.get("mode_str") or "all")
    previous_label_str = previous_severity_str or "unknown"
    reason_str = str(
        inspector_report_dict.get("overall_reason_str")
        or "Inspector report needs operator review."
    )
    vps_id_str = str(inspector_report_dict.get("vps_id_str") or "?")
    message_str = (
        f":rotating_light: **INSPECTOR / {mode_str.upper()} / {vps_id_str}** "
        f"turned **RED** (was `{previous_label_str}`).\n"
        f"Required: **Open Dashboard and inspect attention queue**\n"
        f"Reason: {reason_str}"
    )
    return {"content": message_str}
