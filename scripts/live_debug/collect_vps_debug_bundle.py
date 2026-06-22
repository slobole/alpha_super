"""Collect a read-first VPS debug bundle for live/paper/incubation triage.

The collector is intentionally an evidence gatherer. It does not run tick,
submit, reconcile, EOD snapshot, service restart, or any DB repair command.
Every command result is captured even when the command exits non-zero.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import zipfile


REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from scripts.norgate_config_env import default_config_env_path_obj, load_config_env_file


DEFAULT_OUTPUT_ROOT_PATH_STR = "results/vps_debug_bundles"
DEFAULT_TIMEOUT_SECONDS_FLOAT = 120.0
DEFAULT_TAIL_LINE_COUNT_INT = 300
WATCHDOG_TASK_NAME_STR = "AlphaLiveOpsWatchdog"

SECRET_KEY_TOKEN_RE = re.compile(
    r"(TOKEN|SECRET|PASSWORD|PASSWD|WEBHOOK|HEARTBEAT|API_KEY|ACCESS_KEY|PRIVATE_KEY)",
    re.IGNORECASE,
)
JSON_SECRET_RE = re.compile(
    r'("?[A-Za-z0-9_.-]*(?:token|secret|password|passwd|webhook|heartbeat|api[_-]?key|access[_-]?key|private[_-]?key)[A-Za-z0-9_.-]*"?\s*:\s*)'
    r'("[^"\r\n]*"|\'[^\'\r\n]*\'|[^\s,}\]\r\n]+)',
    re.IGNORECASE,
)
SENSITIVE_URL_RE = re.compile(
    r"https://(?:discord(?:app)?\.com/api/webhooks|hc-ping\.com)/[^\s\"'<>]+",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CommandResult:
    name_str: str
    argv_list: list[str]
    return_code_int: int
    duration_seconds_float: float
    stdout_path_str: str
    stderr_path_str: str
    meta_path_str: str
    timed_out_bool: bool


def utc_timestamp_label_str(now_ts: datetime | None = None) -> str:
    timestamp_ts = now_ts or datetime.now(UTC)
    return timestamp_ts.strftime("%Y%m%dT%H%M%SZ")


def redact_text_str(raw_text_str: str) -> str:
    line_list: list[str] = []
    for raw_line_str in raw_text_str.splitlines():
        line_str = _redact_key_value_line_str(raw_line_str)
        line_str = JSON_SECRET_RE.sub(lambda match_obj: match_obj.group(1) + '"<REDACTED>"', line_str)
        line_str = SENSITIVE_URL_RE.sub("<REDACTED_URL>", line_str)
        line_list.append(line_str)
    if raw_text_str.endswith(("\n", "\r")):
        return "\n".join(line_list) + "\n"
    return "\n".join(line_list)


def _redact_key_value_line_str(raw_line_str: str) -> str:
    line_str = raw_line_str.strip()
    if not line_str or line_str.startswith("#") or "=" not in line_str:
        return raw_line_str
    prefix_str = raw_line_str[: len(raw_line_str) - len(raw_line_str.lstrip())]
    export_prefix_str = ""
    working_line_str = raw_line_str.strip()
    if working_line_str.startswith("export "):
        export_prefix_str = "export "
        working_line_str = working_line_str[len("export ") :].strip()
    key_str, _value_str = working_line_str.split("=", 1)
    key_str = key_str.strip()
    if SECRET_KEY_TOKEN_RE.search(key_str):
        return f"{prefix_str}{export_prefix_str}{key_str}=<REDACTED>"
    return raw_line_str


def safe_read_text_str(path_obj: Path) -> str:
    try:
        return path_obj.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path_obj.read_text(encoding="utf-8", errors="replace")


def write_text_file(path_obj: Path, text_str: str) -> None:
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(text_str, encoding="utf-8")


def write_json_file(path_obj: Path, payload_dict: dict[str, object]) -> None:
    write_text_file(path_obj, json.dumps(payload_dict, indent=2, sort_keys=True) + "\n")


def copy_redacted_file(source_path_obj: Path, dest_path_obj: Path) -> bool:
    if not source_path_obj.exists():
        return False
    write_text_file(dest_path_obj, redact_text_str(safe_read_text_str(source_path_obj)))
    return True


def tail_text_str(path_obj: Path, line_count_int: int) -> str:
    if not path_obj.exists():
        return ""
    text_str = safe_read_text_str(path_obj)
    line_list = text_str.splitlines()
    tail_list = line_list[-max(0, line_count_int) :]
    return "\n".join(tail_list) + ("\n" if tail_list else "")


def run_command_result(
    *,
    name_str: str,
    argv_list: list[str],
    output_dir_path: Path,
    timeout_seconds_float: float,
) -> CommandResult:
    start_time_float = time.monotonic()
    timed_out_bool = False
    try:
        completed_obj = subprocess.run(
            argv_list,
            cwd=REPO_ROOT_PATH,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds_float,
            check=False,
        )
        return_code_int = int(completed_obj.returncode)
        stdout_str = completed_obj.stdout
        stderr_str = completed_obj.stderr
    except subprocess.TimeoutExpired as exception_obj:
        timed_out_bool = True
        return_code_int = 124
        stdout_obj = exception_obj.stdout or ""
        stderr_obj = exception_obj.stderr or ""
        stdout_str = stdout_obj if isinstance(stdout_obj, str) else stdout_obj.decode("utf-8", errors="replace")
        stderr_str = stderr_obj if isinstance(stderr_obj, str) else stderr_obj.decode("utf-8", errors="replace")
        stderr_str += f"\nCOMMAND TIMED OUT AFTER {timeout_seconds_float:.1f} SECONDS\n"
    duration_seconds_float = time.monotonic() - start_time_float

    stdout_path_obj = output_dir_path / f"{name_str}.stdout.txt"
    stderr_path_obj = output_dir_path / f"{name_str}.stderr.txt"
    meta_path_obj = output_dir_path / f"{name_str}.meta.json"
    write_text_file(stdout_path_obj, redact_text_str(stdout_str))
    write_text_file(stderr_path_obj, redact_text_str(stderr_str))
    meta_dict: dict[str, object] = {
        "name_str": name_str,
        "argv_list": argv_list,
        "return_code_int": return_code_int,
        "duration_seconds_float": round(duration_seconds_float, 3),
        "timed_out_bool": timed_out_bool,
    }
    write_json_file(meta_path_obj, meta_dict)
    return CommandResult(
        name_str=name_str,
        argv_list=argv_list,
        return_code_int=return_code_int,
        duration_seconds_float=duration_seconds_float,
        stdout_path_str=str(stdout_path_obj.relative_to(output_dir_path.parent)),
        stderr_path_str=str(stderr_path_obj.relative_to(output_dir_path.parent)),
        meta_path_str=str(meta_path_obj.relative_to(output_dir_path.parent)),
        timed_out_bool=timed_out_bool,
    )


def python_argv_list(*arg_str_list: str) -> list[str]:
    return [sys.executable, *arg_str_list]


def base_runner_argv_list(command_name_str: str, parsed_args_obj: argparse.Namespace) -> list[str]:
    argv_list = python_argv_list("-m", "alpha.live.runner", command_name_str)
    argv_list.extend(["--mode", parsed_args_obj.mode_str])
    if parsed_args_obj.pod_id_str:
        argv_list.extend(["--pod-id", parsed_args_obj.pod_id_str])
    if parsed_args_obj.releases_root_path_str:
        argv_list.extend(["--releases-root", parsed_args_obj.releases_root_path_str])
    if parsed_args_obj.db_path_str:
        argv_list.extend(["--db-path", parsed_args_obj.db_path_str])
    if parsed_args_obj.as_of_timestamp_str:
        argv_list.extend(["--as-of-ts", parsed_args_obj.as_of_timestamp_str])
    if command_name_str == "doctor" and parsed_args_obj.doctor_broker_client_id_int is not None:
        argv_list.extend(["--broker-client-id", str(parsed_args_obj.doctor_broker_client_id_int)])
    argv_list.append("--json")
    return argv_list


def scheduler_next_due_argv_list(parsed_args_obj: argparse.Namespace) -> list[str]:
    argv_list = python_argv_list("-m", "alpha.live.scheduler_service", "next_due")
    argv_list.extend(["--mode", parsed_args_obj.mode_str])
    if parsed_args_obj.pod_id_str:
        argv_list.extend(["--pod-id", parsed_args_obj.pod_id_str])
    if parsed_args_obj.releases_root_path_str:
        argv_list.extend(["--releases-root", parsed_args_obj.releases_root_path_str])
    if parsed_args_obj.db_path_str:
        argv_list.extend(["--db-path", parsed_args_obj.db_path_str])
    if parsed_args_obj.as_of_timestamp_str:
        argv_list.extend(["--as-of-ts", parsed_args_obj.as_of_timestamp_str])
    argv_list.append("--json")
    return argv_list


def powershell_scheduled_task_argv_list() -> list[str]:
    command_str = (
        f"Get-ScheduledTaskInfo -TaskName {WATCHDOG_TASK_NAME_STR} | "
        "Format-List *"
    )
    return ["powershell", "-NoProfile", "-Command", command_str]


def build_command_spec_list(parsed_args_obj: argparse.Namespace) -> list[tuple[str, list[str], bool]]:
    command_spec_list: list[tuple[str, list[str], bool]] = [
        ("git_status", ["git", "status", "--short", "--branch"], True),
        ("git_head", ["git", "rev-parse", "HEAD"], True),
        ("watchdog_scheduled_task", powershell_scheduled_task_argv_list(), True),
        ("ops_report", base_runner_argv_list("ops_report", parsed_args_obj), True),
    ]

    if parsed_args_obj.pod_id_str:
        if (
            parsed_args_obj.include_doctor_bool
            and parsed_args_obj.doctor_broker_client_id_int is not None
            and parsed_args_obj.mode_str in {"live", "paper"}
        ):
            command_spec_list.append(
                ("doctor", base_runner_argv_list("doctor", parsed_args_obj), True)
            )
        elif parsed_args_obj.include_doctor_bool and parsed_args_obj.mode_str in {"live", "paper"}:
            command_spec_list.append(
                (
                    "doctor",
                    [
                        "SKIP",
                        "requires --doctor-broker-client-id so doctor does not reuse the scheduler's manifest client ID",
                    ],
                    False,
                )
            )
        elif parsed_args_obj.include_doctor_bool:
            command_spec_list.append(("doctor", ["SKIP", "doctor is paper/live only"], False))
        else:
            command_spec_list.append(
                (
                    "doctor",
                    [
                        "SKIP",
                        "requires --include-doctor because doctor can query broker state and update snapshot readiness metadata",
                    ],
                    False,
                )
            )
        if parsed_args_obj.include_runner_details_bool:
            command_spec_list.extend(
                [
                    ("status", base_runner_argv_list("status", parsed_args_obj), True),
                    ("scheduler_next_due", scheduler_next_due_argv_list(parsed_args_obj), True),
                    (
                        "show_decision_plan",
                        base_runner_argv_list("show_decision_plan", parsed_args_obj),
                        True,
                    ),
                    ("show_vplan", base_runner_argv_list("show_vplan", parsed_args_obj), True),
                    ("execution_report", base_runner_argv_list("execution_report", parsed_args_obj), True),
                ]
            )
        else:
            command_spec_list.append(
                (
                    "runner_detail_commands",
                    [
                        "SKIP",
                        "requires --include-runner-details because these commands can record diagnostic job metadata in the pod DB",
                    ],
                    False,
                )
            )
    else:
        command_spec_list.append(("pod_scoped_commands", ["SKIP", "--pod-id not provided"], False))

    if parsed_args_obj.include_norgate_doctor_bool:
        norgate_argv_list = python_argv_list("scripts/doctor_norgate_client.py")
        norgate_argv_list.extend(
            ["--report-json", str(parsed_args_obj.norgate_report_path_obj)]
        )
        command_spec_list.append(("norgate_client_doctor", norgate_argv_list, True))
    else:
        command_spec_list.append(
            (
                "norgate_client_doctor",
                ["SKIP", "requires --include-norgate-doctor because it may sync snapshot files"],
                False,
            )
        )

    if parsed_args_obj.release_manifest_path_str and parsed_args_obj.ibkr_probe_client_id_int is not None:
        probe_argv_list = python_argv_list("scripts/live_debug/ibkr_connectivity_probe.py")
        probe_argv_list.extend(
            [
                "--release-manifest-path",
                parsed_args_obj.release_manifest_path_str,
                "--client-id",
                str(parsed_args_obj.ibkr_probe_client_id_int),
                "--json",
            ]
        )
        command_spec_list.append(("ibkr_connectivity_probe", probe_argv_list, True))
    else:
        reason_str = "requires --release-manifest-path and --ibkr-probe-client-id"
        command_spec_list.append(("ibkr_connectivity_probe", ["SKIP", reason_str], False))

    return command_spec_list


def collect_static_files(
    *,
    bundle_dir_path: Path,
    parsed_args_obj: argparse.Namespace,
    loaded_env_dict: dict[str, str],
) -> list[str]:
    collected_path_list: list[str] = []
    config_env_path_obj = default_config_env_path_obj()
    if copy_redacted_file(config_env_path_obj, bundle_dir_path / "config_env_redacted.txt"):
        collected_path_list.append("config_env_redacted.txt")

    if parsed_args_obj.release_manifest_path_str:
        manifest_path_obj = Path(parsed_args_obj.release_manifest_path_str)
        if not manifest_path_obj.is_absolute():
            manifest_path_obj = REPO_ROOT_PATH / manifest_path_obj
        if copy_redacted_file(manifest_path_obj, bundle_dir_path / "release_manifest_redacted.yaml"):
            collected_path_list.append("release_manifest_redacted.yaml")

    norgate_root_str = (
        os.getenv("NORGATE_SNAPSHOT_ROOT", "").strip()
        or loaded_env_dict.get("NORGATE_SNAPSHOT_ROOT", "").strip()
    )
    if norgate_root_str:
        sync_status_path_obj = Path(norgate_root_str).expanduser() / ".client_sync_status.json"
        if copy_redacted_file(
            sync_status_path_obj,
            bundle_dir_path / "norgate_client_sync_status_redacted.json",
        ):
            collected_path_list.append("norgate_client_sync_status_redacted.json")

    return collected_path_list


def collect_log_tails(
    *,
    bundle_dir_path: Path,
    pod_id_str: str | None,
    tail_line_count_int: int,
) -> list[str]:
    collected_path_list: list[str] = []
    log_root_path = REPO_ROOT_PATH / "alpha" / "live" / "logs"
    log_file_path_list = [
        log_root_path / "ops_report_latest.json",
        log_root_path / "live_critical_events.jsonl",
        log_root_path / "live_events.jsonl",
        log_root_path / "live_operator.log",
        log_root_path / "operator_journal.jsonl",
        log_root_path / "watchdog_notification_state.json",
    ]
    for source_path_obj in log_file_path_list:
        if not source_path_obj.exists():
            continue
        dest_path_obj = bundle_dir_path / "logs" / f"{source_path_obj.name}.tail.txt"
        write_text_file(dest_path_obj, redact_text_str(tail_text_str(source_path_obj, tail_line_count_int)))
        collected_path_list.append(str(dest_path_obj.relative_to(bundle_dir_path)))

    if pod_id_str:
        pod_log_root_path = log_root_path / "pods" / pod_id_str
        if pod_log_root_path.exists():
            pod_log_path_list = sorted(
                [
                    path_obj
                    for path_obj in pod_log_root_path.rglob("*")
                    if path_obj.is_file() and path_obj.suffix.lower() in {".jsonl", ".log", ".txt"}
                ],
                key=lambda path_obj: path_obj.stat().st_mtime,
                reverse=True,
            )[:20]
            for source_path_obj in pod_log_path_list:
                relative_path_obj = source_path_obj.relative_to(pod_log_root_path)
                dest_path_obj = bundle_dir_path / "logs" / "pod_logs_tail" / relative_path_obj
                write_text_file(
                    dest_path_obj.with_suffix(dest_path_obj.suffix + ".tail.txt"),
                    redact_text_str(tail_text_str(source_path_obj, tail_line_count_int)),
                )
                collected_path_list.append(
                    str(dest_path_obj.with_suffix(dest_path_obj.suffix + ".tail.txt").relative_to(bundle_dir_path))
                )
    return collected_path_list


def collect_system_info(bundle_dir_path: Path, parsed_args_obj: argparse.Namespace) -> None:
    env_key_list = sorted(
        key_str
        for key_str in os.environ
        if key_str.startswith("ALPHA_")
        or key_str.startswith("NORGATE_")
        or key_str.startswith("IBKR_")
    )
    info_dict: dict[str, object] = {
        "schema_version_str": "vps_debug_bundle.v1",
        "generated_at_utc_str": datetime.now(UTC).isoformat(),
        "repo_root_path_str": str(REPO_ROOT_PATH),
        "mode_str": parsed_args_obj.mode_str,
        "pod_id_str": parsed_args_obj.pod_id_str or "",
        "release_manifest_path_str": parsed_args_obj.release_manifest_path_str or "",
        "releases_root_path_str": parsed_args_obj.releases_root_path_str or "",
        "db_path_str": parsed_args_obj.db_path_str or "",
        "python_version_str": sys.version,
        "platform_str": platform.platform(),
        "env_key_list": env_key_list,
        "safety_contract_list": [
            "Does not run tick.",
            "Does not submit, modify, or cancel orders.",
            "Does not run post_execution_reconcile.",
            "Does not run eod_snapshot.",
            "Default bundle does not run runner detail commands that record diagnostic job metadata.",
            "Use --include-runner-details only when pod DB job metadata writes are acceptable.",
            "Use --include-doctor only when broker reads and snapshot readiness side effects are acceptable.",
            "Skips standalone IBKR probe unless an explicit unused client ID is supplied.",
            "Skips Norgate client doctor unless explicitly requested because that doctor may sync snapshot files.",
        ],
        "sharing_notice_str": (
            "For trusted operator/Codex review only. Token and webhook values are redacted, "
            "but opt-in broker/doctor outputs can include account routes, positions, cash, NetLiq, "
            "open orders, and other portfolio state."
        ),
    }
    write_json_file(bundle_dir_path / "system_info.json", info_dict)


def write_skip_result(
    *,
    name_str: str,
    argv_list: list[str],
    command_dir_path: Path,
) -> CommandResult:
    stdout_path_obj = command_dir_path / f"{name_str}.stdout.txt"
    stderr_path_obj = command_dir_path / f"{name_str}.stderr.txt"
    meta_path_obj = command_dir_path / f"{name_str}.meta.json"
    reason_str = " ".join(argv_list[1:]) if len(argv_list) > 1 else "skipped"
    write_text_file(stdout_path_obj, reason_str + "\n")
    write_text_file(stderr_path_obj, "")
    write_json_file(
        meta_path_obj,
        {
            "name_str": name_str,
            "argv_list": argv_list,
            "return_code_int": 0,
            "duration_seconds_float": 0.0,
            "timed_out_bool": False,
            "skipped_bool": True,
        },
    )
    return CommandResult(
        name_str=name_str,
        argv_list=argv_list,
        return_code_int=0,
        duration_seconds_float=0.0,
        stdout_path_str=str(stdout_path_obj.relative_to(command_dir_path.parent)),
        stderr_path_str=str(stderr_path_obj.relative_to(command_dir_path.parent)),
        meta_path_str=str(meta_path_obj.relative_to(command_dir_path.parent)),
        timed_out_bool=False,
    )


def create_zip_path(bundle_dir_path: Path) -> Path:
    zip_path_obj = bundle_dir_path.with_suffix(".zip")
    if zip_path_obj.exists():
        for index_int in range(1, 1000):
            candidate_path_obj = bundle_dir_path.with_name(
                f"{bundle_dir_path.name}_{index_int}"
            ).with_suffix(".zip")
            if not candidate_path_obj.exists():
                zip_path_obj = candidate_path_obj
                break
        else:
            raise FileExistsError(f"Could not choose a non-existing zip path near {zip_path_obj}.")
    with zipfile.ZipFile(zip_path_obj, "x", compression=zipfile.ZIP_DEFLATED) as zip_file_obj:
        for source_path_obj in sorted(bundle_dir_path.rglob("*")):
            if source_path_obj.is_file():
                zip_file_obj.write(source_path_obj, source_path_obj.relative_to(bundle_dir_path))
    return zip_path_obj


def collect_bundle_dict(parsed_args_obj: argparse.Namespace) -> dict[str, object]:
    os.chdir(REPO_ROOT_PATH)
    loaded_env_dict = load_config_env_file(override_existing_bool=False)
    timestamp_label_str = utc_timestamp_label_str()
    pod_label_str = parsed_args_obj.pod_id_str or "all_pods"
    bundle_dir_path = (
        Path(parsed_args_obj.output_root_path_str)
        / parsed_args_obj.mode_str
        / pod_label_str
        / timestamp_label_str
    )
    if not bundle_dir_path.is_absolute():
        bundle_dir_path = REPO_ROOT_PATH / bundle_dir_path
    bundle_dir_path.mkdir(parents=True, exist_ok=False)

    parsed_args_obj.norgate_report_path_obj = bundle_dir_path / "commands" / "norgate_client_doctor_report.json"

    collect_system_info(bundle_dir_path, parsed_args_obj)
    static_file_list = collect_static_files(
        bundle_dir_path=bundle_dir_path,
        parsed_args_obj=parsed_args_obj,
        loaded_env_dict=loaded_env_dict,
    )
    log_tail_file_list = collect_log_tails(
        bundle_dir_path=bundle_dir_path,
        pod_id_str=parsed_args_obj.pod_id_str,
        tail_line_count_int=parsed_args_obj.tail_line_count_int,
    )

    command_dir_path = bundle_dir_path / "commands"
    command_dir_path.mkdir(parents=True, exist_ok=True)
    command_result_list: list[CommandResult] = []
    for name_str, argv_list, should_run_bool in build_command_spec_list(parsed_args_obj):
        if should_run_bool:
            result_obj = run_command_result(
                name_str=name_str,
                argv_list=argv_list,
                output_dir_path=command_dir_path,
                timeout_seconds_float=parsed_args_obj.timeout_seconds_float,
            )
        else:
            result_obj = write_skip_result(
                name_str=name_str,
                argv_list=argv_list,
                command_dir_path=command_dir_path,
            )
        command_result_list.append(result_obj)

    manifest_dict: dict[str, object] = {
        "schema_version_str": "vps_debug_bundle.v1",
        "bundle_dir_path_str": str(bundle_dir_path),
        "mode_str": parsed_args_obj.mode_str,
        "pod_id_str": parsed_args_obj.pod_id_str or "",
        "created_at_utc_str": datetime.now(UTC).isoformat(),
        "static_file_list": static_file_list,
        "log_tail_file_list": log_tail_file_list,
        "command_result_list": [asdict(result_obj) for result_obj in command_result_list],
    }
    write_json_file(bundle_dir_path / "bundle_manifest.json", manifest_dict)
    zip_path_obj: Path | None = None
    if not parsed_args_obj.no_zip_bool:
        zip_path_obj = create_zip_path(bundle_dir_path)
    summary_dict: dict[str, object] = {
        "status_str": "ok",
        "bundle_dir_path_str": str(bundle_dir_path),
        "zip_path_str": "" if zip_path_obj is None else str(zip_path_obj),
        "failed_command_name_list": [
            result_obj.name_str
            for result_obj in command_result_list
            if result_obj.return_code_int not in (0,)
        ],
        "skipped_command_name_list": [
            result_obj.name_str
            for result_obj in command_result_list
            if Path(command_dir_path / f"{result_obj.name_str}.meta.json").exists()
            and '"skipped_bool": true' in (command_dir_path / f"{result_obj.name_str}.meta.json").read_text(encoding="utf-8")
        ],
    }
    write_json_file(bundle_dir_path / "bundle_summary.json", summary_dict)
    return summary_dict


def print_result(result_dict: dict[str, object], *, json_output_bool: bool) -> None:
    if json_output_bool:
        print(json.dumps(result_dict, indent=2, sort_keys=True))
        return
    print(f"Bundle directory: {result_dict['bundle_dir_path_str']}")
    zip_path_str = str(result_dict.get("zip_path_str") or "")
    if zip_path_str:
        print(f"Bundle zip:       {zip_path_str}")
    failed_list = list(result_dict.get("failed_command_name_list") or [])
    if failed_list:
        print("Non-zero commands: " + ", ".join(str(item_obj) for item_obj in failed_list))


def main(argv_list: list[str] | None = None) -> int:
    parser_obj = argparse.ArgumentParser(
        description="Collect a read-first VPS debug bundle for alpha_super live ops."
    )
    parser_obj.add_argument("--mode", dest="mode_str", choices=("live", "paper", "incubation"), default="live")
    parser_obj.add_argument("--pod-id", dest="pod_id_str", default=None)
    parser_obj.add_argument("--release-manifest-path", dest="release_manifest_path_str", default=None)
    parser_obj.add_argument("--releases-root", dest="releases_root_path_str", default=None)
    parser_obj.add_argument("--db-path", dest="db_path_str", default=None)
    parser_obj.add_argument("--as-of-ts", dest="as_of_timestamp_str", default=None)
    parser_obj.add_argument("--output-root", dest="output_root_path_str", default=DEFAULT_OUTPUT_ROOT_PATH_STR)
    parser_obj.add_argument("--tail-lines", dest="tail_line_count_int", type=int, default=DEFAULT_TAIL_LINE_COUNT_INT)
    parser_obj.add_argument(
        "--timeout-seconds",
        dest="timeout_seconds_float",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS_FLOAT,
    )
    parser_obj.add_argument(
        "--include-runner-details",
        dest="include_runner_details_bool",
        action="store_true",
        help="Run status/next_due/show plan commands. These can record diagnostic job metadata in the pod DB.",
    )
    parser_obj.add_argument(
        "--include-doctor",
        dest="include_doctor_bool",
        action="store_true",
        help="Run runner doctor. Doctor can query broker state and update snapshot readiness metadata.",
    )
    parser_obj.add_argument(
        "--doctor-broker-client-id",
        dest="doctor_broker_client_id_int",
        type=int,
        default=None,
        help="Required with --include-doctor so doctor does not reuse the scheduler's manifest client ID.",
    )
    parser_obj.add_argument(
        "--include-norgate-doctor",
        dest="include_norgate_doctor_bool",
        action="store_true",
        help="Run doctor_norgate_client.py. This may sync snapshot files under NORGATE_SNAPSHOT_ROOT.",
    )
    parser_obj.add_argument(
        "--ibkr-probe-client-id",
        dest="ibkr_probe_client_id_int",
        type=int,
        default=None,
        help="Run the standalone IBKR probe with this explicit unused client ID.",
    )
    parser_obj.add_argument("--no-zip", dest="no_zip_bool", action="store_true")
    parser_obj.add_argument("--json", dest="json_output_bool", action="store_true")
    parsed_args_obj = parser_obj.parse_args(argv_list)

    result_dict = collect_bundle_dict(parsed_args_obj)
    print_result(result_dict, json_output_bool=parsed_args_obj.json_output_bool)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
