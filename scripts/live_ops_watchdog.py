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
from datetime import datetime
import io
import json
import os
from pathlib import Path
import sys


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
) -> tuple[dict[str, object], list[object]]:
    summary_dict, report_dict = _build_summary_and_report_tuple(parsed_args_obj, as_of_ts)
    write_report_atomic(report_dict, parsed_args_obj.output_path_str)
    fired_list = notifications_module.check_and_notify_for_red_transitions(
        summary_dict,
        state_store_obj=notifications_module.NotificationStateStore(
            state_path_str=parsed_args_obj.notification_state_path_str
        ),
        webhook_url_str=notifications_module.discord_webhook_url_from_env_str(),
        webhook_poster_fn=notifications_module.post_discord_webhook_bool,
    )
    return report_dict, fired_list


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
                report_dict, fired_list = _run_report_pipeline_tuple(parsed_args_obj, as_of_ts)
        else:
            report_dict, fired_list = _run_report_pipeline_tuple(parsed_args_obj, as_of_ts)
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
    )


if __name__ == "__main__":
    raise SystemExit(main())
