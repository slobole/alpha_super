"""Scheduled Live OPS watchdog: Inspector report + alerts + dead-man heartbeat.

One process, strict order: build the Inspector report, persist it atomically,
fire red-transition webhooks, and only then ping the external dead-man switch.
If any earlier step crashes or hangs, no ping is sent and the external watcher
(healthchecks.io) alerts — silence externally always means "the inspector did
not complete a run". Run every ~5 minutes via Windows Task Scheduler (see
scripts/setup_live_ops_watchdog_task.ps1).

Design notes:
- No enabled PODs yields overall gray -> exit 0 -> plain success ping. The
  dead-man switch monitors watchdog liveness, not pod existence; gray/yellow
  surface on the dashboard, red is the only fail signal.
- The "__inspector__" webhook transition inside
  check_and_notify_for_red_transitions reads the summary's embedded all-modes
  inspector_report_dict (built by build_dashboard_summary_dict with default
  args), while the persisted report applies --vps-id/--stale-after-seconds.
  With default flags at the same as_of they agree; this is intentional.
- The watchdog keeps its own notification state file, separate from the
  dashboard's, to avoid cross-process races on one JSON file. If the dashboard
  also has ALPHA_DISCORD_WEBHOOK_URL set, the same red transition can alert
  twice; harmless and accepted.
"""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile


REPO_ROOT_PATH = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

import alpha.live.dashboard as dashboard_module
import alpha.live.dashboard_v3.notifications as notifications_module
import alpha.live.ops_report as ops_report_module
from scripts.norgate_config_env import load_config_env_file


WATCHDOG_REPORT_PATH_STR = "alpha/live/logs/ops_report_latest.json"
WATCHDOG_NOTIFICATION_STATE_PATH_STR = "alpha/live/logs/watchdog_notification_state.json"
HEARTBEAT_URL_ENV_VAR_NAME_STR = "ALPHA_INSPECTOR_HEARTBEAT_URL"
FATAL_EXIT_CODE_INT = 2
RUN_RECEIPT_SCHEMA_STR = "live_ops_watchdog_run.v1"
RECEIPT_IDENTITY_FIELD_TUPLE = ("mode_str", "user_id_str", "pod_id_str", "account_route_str", "release_id_str")
RECEIPT_IDENTITY_PATTERN_OBJ = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}\Z")


def write_report_atomic(report_dict: dict[str, object], output_path_str: str) -> None:
    output_path_obj = Path(output_path_str)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    # *** CRITICAL*** tmp file must live in the same directory as the target so
    # os.replace is an atomic same-volume rename; readers never see a torn file.
    tmp_path_obj = output_path_obj.with_name(output_path_obj.name + ".tmp")
    tmp_path_obj.write_text(
        json.dumps(report_dict, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp_path_obj, output_path_obj)


def heartbeat_target_url_str(heartbeat_url_str: str, overall_severity_str: str) -> str:
    if overall_severity_str == "red":
        return heartbeat_url_str.rstrip("/") + "/fail"
    return heartbeat_url_str


def _completed_run_receipt_dict(report_dict, fired_list, receipt_context_dict, *,
        heartbeat_status_str, heartbeat_fail_signal_bool):
    """Project this completed pass only; no URLs, errors or notification bodies."""
    completed_ts = ops_report_module.utc_now_ts()
    if completed_ts.tzinfo is None or completed_ts.utcoffset() is None:
        raise ValueError("Invalid receipt clock")
    completed_ts = completed_ts.astimezone(timezone.utc)
    report_timestamp_str = report_dict.get("generated_at_utc_str")
    report_ts = datetime.fromisoformat(report_timestamp_str.replace("Z", "+00:00"))
    if report_ts.tzinfo is None or report_ts.utcoffset() is None or report_ts > completed_ts:
        raise ValueError("Invalid report clock")
    mode_str = report_dict.get("mode_str")
    if mode_str not in {"all", "live", "paper", "incubation"}:
        raise ValueError("Invalid receipt mode")
    row_list = receipt_context_dict["summary_dict"].get("pod_row_dict_list")
    if not isinstance(row_list, list) or len(row_list) > 128 or any(not isinstance(row_dict, dict) for row_dict in row_list):
        raise ValueError("Invalid receipt scope")
    scope_list, pod_set, account_set = [], set(), set()
    for row_dict in row_list:
        if mode_str != "all" and row_dict.get("mode_str") != mode_str:
            continue
        identity_dict = {field_str: row_dict.get(field_str) for field_str in RECEIPT_IDENTITY_FIELD_TUPLE}
        if any(not isinstance(value_str, str) or not RECEIPT_IDENTITY_PATTERN_OBJ.fullmatch(value_str) for value_str in identity_dict.values()):
            raise ValueError("Invalid receipt identity")
        if identity_dict["mode_str"] not in {"live", "paper", "incubation"}:
            raise ValueError("Invalid receipt identity")
        pod_tuple = (identity_dict["mode_str"], identity_dict["pod_id_str"])
        account_tuple = (identity_dict["mode_str"], identity_dict["account_route_str"])
        if pod_tuple in pod_set or account_tuple in account_set:
            raise ValueError("Ambiguous receipt identity")
        pod_set.add(pod_tuple)
        account_set.add(account_tuple)
        scope_list.append(identity_dict)
    configured_bool = receipt_context_dict["notification_configured_bool"]
    if type(configured_bool) is not bool or heartbeat_status_str not in {"sent", "failed", "disabled"} or type(heartbeat_fail_signal_bool) is not bool:
        raise ValueError("Invalid receipt result")
    # With a configured webhook every remaining red retry is attempted in this
    # pass. Its failed LIVE attempts are exactly the saved LIVE retry backlog.
    # Without a webhook, new red alerts and old retries cannot be distinguished
    # from fired_list alone, so retain unknown instead of inventing zero.
    pending_live_int = sum(record_obj.delivered_bool is False for record_obj in fired_list
        if record_obj.mode_str == "live" and record_obj.pod_id_str != notifications_module.INSPECTOR_NOTIFICATION_KEY_STR) if configured_bool else None
    return {"schema_version_str": RUN_RECEIPT_SCHEMA_STR, "completed_at_utc_str": completed_ts.isoformat(),
        "report_generated_at_utc_str": report_timestamp_str,
        "report_sha256_str": hashlib.sha256(json.dumps(report_dict, sort_keys=True,
            separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest(),
        "mode_str": mode_str, "scope_list": sorted(scope_list, key=lambda identity_dict: (
            identity_dict["mode_str"], identity_dict["pod_id_str"], identity_dict["release_id_str"])),
        "heartbeat_status_str": heartbeat_status_str, "heartbeat_fail_signal_bool": heartbeat_fail_signal_bool,
        "notification_configured_bool": configured_bool, "notification_pending_live_count_int": pending_live_int}


def _write_run_receipt_atomic(receipt_dict, output_path_str):
    receipt_path_obj = Path(output_path_str).with_suffix(".run.json")
    tmp_path_obj = None
    try:
        # A unique sibling also avoids collisions with manually overlapping runs.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=receipt_path_obj.parent,
                prefix="." + receipt_path_obj.name + ".", suffix=".tmp", delete=False) as file_obj:
            tmp_path_obj = Path(file_obj.name)
            json.dump(receipt_dict, file_obj, sort_keys=True, indent=2)
        os.replace(tmp_path_obj, receipt_path_obj)
    finally:
        if tmp_path_obj is not None:
            with contextlib.suppress(OSError):
                tmp_path_obj.unlink(missing_ok=True)


def _resolve_as_of_ts(as_of_timestamp_str: str | None) -> datetime:
    if not as_of_timestamp_str:
        return ops_report_module.utc_now_ts()
    as_of_ts = ops_report_module.parse_timestamp_ts(as_of_timestamp_str)
    if as_of_ts is None:
        raise ValueError(f"Unparseable --as-of-ts value '{as_of_timestamp_str}'.")
    return as_of_ts


def _build_summary_and_report_tuple(
    parsed_args_obj: argparse.Namespace,
    as_of_ts: datetime,
) -> tuple[dict[str, object], dict[str, object]]:
    dashboard_app_kwargs_dict: dict[str, object] = {}
    if parsed_args_obj.releases_root_path_str:
        dashboard_app_kwargs_dict["releases_root_path_str"] = parsed_args_obj.releases_root_path_str
    if parsed_args_obj.dashboard_config_path_str:
        dashboard_app_kwargs_dict["config_path_str"] = parsed_args_obj.dashboard_config_path_str
    dashboard_app_obj = dashboard_module.DashboardApp(**dashboard_app_kwargs_dict)
    summary_dict = dashboard_module.build_dashboard_summary_dict(
        dashboard_app_obj,
        as_of_ts=as_of_ts,
    )
    report_dict = ops_report_module.build_ops_report_dict(
        summary_dict,
        mode_str=parsed_args_obj.mode_str,
        generated_at_ts=as_of_ts,
        stale_after_seconds_int=parsed_args_obj.stale_after_seconds_int,
        vps_id_str=parsed_args_obj.vps_id_str,
    )
    return summary_dict, report_dict


def _run_report_pipeline_tuple(
    parsed_args_obj: argparse.Namespace,
    as_of_ts: datetime,
) -> tuple[dict[str, object], list[object], dict[str, object]]:
    summary_dict, report_dict = _build_summary_and_report_tuple(parsed_args_obj, as_of_ts)
    write_report_atomic(report_dict, parsed_args_obj.output_path_str)
    webhook_url_str = notifications_module.discord_webhook_url_from_env_str()
    fired_list = notifications_module.check_and_notify_for_red_transitions(
        summary_dict,
        state_store_obj=notifications_module.NotificationStateStore(
            state_path_str=parsed_args_obj.notification_state_path_str
        ),
        webhook_url_str=webhook_url_str,
        webhook_poster_fn=notifications_module.post_discord_webhook_bool,
    )
    return report_dict, fired_list, {"summary_dict": summary_dict,
        "notification_configured_bool": bool(webhook_url_str)}


def main(argv_list: list[str] | None = None) -> int:
    # Task Scheduler starts processes in System32; every default path above is
    # CWD-relative, so anchor to the repo root before anything else.
    os.chdir(REPO_ROOT_PATH)

    parser_obj = argparse.ArgumentParser(
        description="Live OPS watchdog: Inspector report, red alerts, dead-man heartbeat."
    )
    parser_obj.add_argument("--vps-id", dest="vps_id_str", default=None)
    parser_obj.add_argument("--releases-root", dest="releases_root_path_str", default=None)
    parser_obj.add_argument(
        "--mode",
        dest="mode_str",
        choices=("live", "paper", "incubation"),
        default=None,
    )
    parser_obj.add_argument(
        "--stale-after-seconds",
        dest="stale_after_seconds_int",
        type=int,
        default=ops_report_module.DEFAULT_STALE_AFTER_SECONDS_INT,
    )
    parser_obj.add_argument(
        "--output-path",
        dest="output_path_str",
        default=WATCHDOG_REPORT_PATH_STR,
    )
    parser_obj.add_argument("--heartbeat-url", dest="heartbeat_url_str", default=None)
    parser_obj.add_argument(
        "--heartbeat-timeout-seconds",
        dest="heartbeat_timeout_seconds_float",
        type=float,
        default=ops_report_module.DEFAULT_HEARTBEAT_TIMEOUT_SECONDS_FLOAT,
    )
    parser_obj.add_argument(
        "--notification-state-path",
        dest="notification_state_path_str",
        default=WATCHDOG_NOTIFICATION_STATE_PATH_STR,
    )
    parser_obj.add_argument(
        "--dashboard-config",
        dest="dashboard_config_path_str",
        default=None,
    )
    parser_obj.add_argument("--as-of-ts", dest="as_of_timestamp_str", default=None)
    parser_obj.add_argument("--json", dest="json_output_bool", action="store_true")
    parsed_args_obj = parser_obj.parse_args(argv_list)

    as_of_ts = _resolve_as_of_ts(parsed_args_obj.as_of_timestamp_str)
    load_config_env_file(override_existing_bool=True)

    try:
        if parsed_args_obj.json_output_bool:
            # Imports inside the build (e.g. norgatedata init) print to stdout;
            # keep --json output a single parseable JSON document.
            with contextlib.redirect_stdout(io.StringIO()):
                report_dict, fired_list, receipt_context_dict = _run_report_pipeline_tuple(parsed_args_obj, as_of_ts)
        else:
            report_dict, fired_list, receipt_context_dict = _run_report_pipeline_tuple(parsed_args_obj, as_of_ts)
    except Exception as exc:
        # Fatal-by-design: an unwritable report or state path also lands here.
        # No heartbeat ping — the external dead-man switch must fire.
        _print_result(
            {
                "status_str": "error",
                "reason_code_str": "watchdog_fatal_error",
                "error_str": str(exc),
            },
            json_output_bool=parsed_args_obj.json_output_bool,
        )
        return FATAL_EXIT_CODE_INT

    overall_severity_str = str(report_dict.get("overall_severity_str") or "gray")
    heartbeat_url_str = parsed_args_obj.heartbeat_url_str or os.getenv(
        HEARTBEAT_URL_ENV_VAR_NAME_STR,
        "",
    )
    heartbeat_fail_signal_bool = False
    if not heartbeat_url_str:
        heartbeat_status_str = "disabled"
    else:
        target_url_str = heartbeat_target_url_str(heartbeat_url_str, overall_severity_str)
        heartbeat_fail_signal_bool = target_url_str != heartbeat_url_str
        delivered_bool = ops_report_module.post_heartbeat_bool(
            target_url_str,
            ops_report_module.build_heartbeat_payload_dict(
                generated_at_ts=as_of_ts,
                vps_id_str=parsed_args_obj.vps_id_str,
            ),
            timeout_seconds_float=parsed_args_obj.heartbeat_timeout_seconds_float,
        )
        heartbeat_status_str = "sent" if delivered_bool else "failed"

    run_receipt_status_str = "saved"
    try:
        receipt_dict = _completed_run_receipt_dict(report_dict, fired_list, receipt_context_dict,
            heartbeat_status_str=heartbeat_status_str, heartbeat_fail_signal_bool=heartbeat_fail_signal_bool)
        _write_run_receipt_atomic(receipt_dict, parsed_args_obj.output_path_str)
    except Exception:
        # This extra observation must never suppress an alert/ping, alter their
        # results, or change the watchdog's established 0/1/2 exit contract.
        run_receipt_status_str = "unavailable"

    _print_result(
        {
            "status_str": "red" if overall_severity_str == "red" else "ok",
            "overall_severity_str": overall_severity_str,
            "overall_reason_str": str(report_dict.get("overall_reason_str") or ""),
            "report_output_path_str": parsed_args_obj.output_path_str,
            "notification_fired_count_int": len(fired_list),
            "heartbeat_status_str": heartbeat_status_str,
            "heartbeat_fail_signal_bool": heartbeat_fail_signal_bool,
            "vps_id_str": str(report_dict.get("vps_id_str") or ""),
            "generated_at_utc_str": str(report_dict.get("generated_at_utc_str") or ""),
            "run_receipt_status_str": run_receipt_status_str,
            "run_receipt_reason_code_str": "watchdog_run_receipt_unavailable" if run_receipt_status_str == "unavailable" else "",
        },
        json_output_bool=parsed_args_obj.json_output_bool,
    )
    # Heartbeat delivery failure does not change the exit code: a missed ping is
    # exactly the condition the external watcher alerts on.
    return 1 if overall_severity_str == "red" else 0


def _print_result(result_dict: dict[str, object], *, json_output_bool: bool) -> None:
    if json_output_bool:
        print(json.dumps(result_dict, indent=2, sort_keys=True))
        return
    if result_dict.get("status_str") == "error":
        print(f"Watchdog error: {result_dict.get('error_str')}")
        return
    print(
        f"Watchdog {result_dict['status_str']}: "
        f"overall={result_dict['overall_severity_str']} "
        f"notifications={result_dict['notification_fired_count_int']} "
        f"heartbeat={result_dict['heartbeat_status_str']}"
        + (" receipt=watchdog_run_receipt_unavailable" if result_dict.get("run_receipt_status_str") == "unavailable" else "")
    )


if __name__ == "__main__":
    raise SystemExit(main())
