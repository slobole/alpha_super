from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.error import HTTPError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

repo_root_path = Path(__file__).resolve().parents[1]
repo_root_str = str(repo_root_path)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

from alpha.live.scheduler_utils import load_latest_norgate_heartbeat_session_label_ts
from data.norgate_snapshot_store import (
    ALPHA_USE_NORGATE_SNAPSHOT_ENV_STR,
    NORGATE_SNAPSHOT_ROOT_ENV_STR,
    is_snapshot_mode_enabled_bool,
    load_index_constituent_matrix_df,
    load_price_timeseries_df,
    load_valid_snapshot_manifest,
)
from scripts.export_norgate_snapshot import SUPPORTED_EOD_PROFILE_TUPLE
from scripts.norgate_config_env import (
    NORGATE_CLIENT_ID_ENV_STR,
    NORGATE_RELEASES_ROOT_ENV_STR,
    env_str,
    load_config_env_file,
    norgate_api_url_from_env_str,
)
from scripts.serve_norgate_snapshot_api import (
    NORGATE_API_TOKEN_ENV_STR,
    NORGATE_API_TOKEN_HEADER_STR,
)
from scripts.sync_norgate_snapshots_api import (
    derive_required_profile_list,
    sync_required_snapshots,
)


PrinterFn = Callable[[str], None]


@dataclass(frozen=True)
class DoctorCheckResult:
    name_str: str
    status_str: str
    detail_str: str
    timestamp_utc_str: str
    metadata_dict: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DoctorReport:
    result_str: str
    generated_timestamp_utc_str: str
    check_list: list[DoctorCheckResult]


def _utc_now_str() -> str:
    return datetime.now(tz=UTC).isoformat()


def _format_result_line(result_obj: DoctorCheckResult) -> str:
    if result_obj.detail_str:
        return f"[{result_obj.status_str}] {result_obj.name_str}: {result_obj.detail_str}"
    return f"[{result_obj.status_str}] {result_obj.name_str}"


def _record_result(
    check_list: list[DoctorCheckResult],
    *,
    name_str: str,
    status_str: str,
    detail_str: str,
    printer_fn: PrinterFn,
    metadata_dict: dict[str, Any] | None = None,
) -> None:
    result_obj = DoctorCheckResult(
        name_str=name_str,
        status_str=status_str,
        detail_str=detail_str,
        timestamp_utc_str=_utc_now_str(),
        metadata_dict=dict(metadata_dict or {}),
    )
    check_list.append(result_obj)
    printer_fn(_format_result_line(result_obj))


def _record_pass(
    check_list: list[DoctorCheckResult],
    name_str: str,
    detail_str: str,
    printer_fn: PrinterFn,
    metadata_dict: dict[str, Any] | None = None,
) -> None:
    _record_result(
        check_list,
        name_str=name_str,
        status_str="PASS",
        detail_str=detail_str,
        printer_fn=printer_fn,
        metadata_dict=metadata_dict,
    )


def _record_fail(
    check_list: list[DoctorCheckResult],
    name_str: str,
    detail_str: str,
    printer_fn: PrinterFn,
    metadata_dict: dict[str, Any] | None = None,
) -> None:
    _record_result(
        check_list,
        name_str=name_str,
        status_str="FAIL",
        detail_str=detail_str,
        printer_fn=printer_fn,
        metadata_dict=metadata_dict,
    )


def _record_skip(
    check_list: list[DoctorCheckResult],
    name_str: str,
    detail_str: str,
    printer_fn: PrinterFn,
) -> None:
    _record_result(
        check_list,
        name_str=name_str,
        status_str="SKIP",
        detail_str=detail_str,
        printer_fn=printer_fn,
    )


@contextmanager
def _temporary_snapshot_root_env(snapshot_root_path_obj: Path) -> Iterator[None]:
    old_snapshot_root_str = os.environ.get(NORGATE_SNAPSHOT_ROOT_ENV_STR)
    os.environ[NORGATE_SNAPSHOT_ROOT_ENV_STR] = str(snapshot_root_path_obj)
    try:
        yield
    finally:
        if old_snapshot_root_str is None:
            os.environ.pop(NORGATE_SNAPSHOT_ROOT_ENV_STR, None)
        else:
            os.environ[NORGATE_SNAPSHOT_ROOT_ENV_STR] = old_snapshot_root_str


def _build_url_str(api_url_str: str, url_path_str: str) -> str:
    return urljoin(api_url_str.rstrip("/") + "/", url_path_str.lstrip("/"))


def _read_error_body_str(error_obj: HTTPError) -> str:
    try:
        return error_obj.read().decode("utf-8")
    except Exception:
        return str(error_obj)


def _get_json_dict(
    *,
    api_url_str: str,
    url_path_str: str,
    token_str: str | None = None,
    timeout_seconds_float: float,
) -> dict[str, Any]:
    header_dict: dict[str, str] = {}
    if token_str is not None:
        header_dict[NORGATE_API_TOKEN_HEADER_STR] = token_str
    request_obj = Request(
        _build_url_str(api_url_str, url_path_str),
        method="GET",
        headers=header_dict,
    )
    with urlopen(request_obj, timeout=timeout_seconds_float) as response_obj:
        payload_obj = json.loads(response_obj.read().decode("utf-8"))
    if not isinstance(payload_obj, dict):
        raise RuntimeError("Norgate API response was not a JSON object.")
    return payload_obj


def _check_required_value_bool(
    check_list: list[DoctorCheckResult],
    *,
    name_str: str,
    value_str: str | None,
    printer_fn: PrinterFn,
) -> bool:
    if value_str is None or not str(value_str).strip():
        _record_fail(check_list, name_str, "missing", printer_fn)
        return False
    _record_pass(check_list, name_str, "set", printer_fn)
    return True


def _check_release_profiles(
    check_list: list[DoctorCheckResult],
    *,
    releases_root_path_str: str,
    printer_fn: PrinterFn,
) -> list[str]:
    try:
        profile_list = derive_required_profile_list(releases_root_path_str)
    except Exception as exc:
        _record_fail(
            check_list,
            "enabled release profiles",
            str(exc),
            printer_fn,
            {"exception_type_str": type(exc).__name__},
        )
        return []
    _record_pass(
        check_list,
        "enabled release profiles",
        ",".join(profile_list),
        printer_fn,
        {"profile_list": profile_list},
    )
    return profile_list


def _check_server_supported_profiles_bool(
    check_list: list[DoctorCheckResult],
    *,
    profile_list: list[str],
    printer_fn: PrinterFn,
) -> bool:
    unsupported_profile_list = [
        profile_str
        for profile_str in profile_list
        if profile_str not in SUPPORTED_EOD_PROFILE_TUPLE
    ]
    if unsupported_profile_list:
        _record_fail(
            check_list,
            "server-supported profiles",
            f"unsupported={unsupported_profile_list}",
            printer_fn,
            {"unsupported_profile_list": unsupported_profile_list},
        )
        return False
    _record_pass(check_list, "server-supported profiles", "ok", printer_fn)
    return True


def _check_api_health_bool(
    check_list: list[DoctorCheckResult],
    *,
    api_url_str: str,
    timeout_seconds_float: float,
    printer_fn: PrinterFn,
) -> bool:
    try:
        response_dict = _get_json_dict(
            api_url_str=api_url_str,
            url_path_str="/healthz",
            timeout_seconds_float=timeout_seconds_float,
        )
        if response_dict.get("status_str") != "ok":
            raise RuntimeError(f"unexpected healthz response: {response_dict}")
    except Exception as exc:
        _record_fail(
            check_list,
            "api healthz",
            str(exc),
            printer_fn,
            {"exception_type_str": type(exc).__name__},
        )
        return False
    _record_pass(check_list, "api healthz", api_url_str, printer_fn)
    return True


def _check_api_token_bool(
    check_list: list[DoctorCheckResult],
    *,
    api_url_str: str,
    token_str: str,
    client_id_str: str,
    timeout_seconds_float: float,
    printer_fn: PrinterFn,
) -> bool:
    try:
        _get_json_dict(
            api_url_str=api_url_str,
            url_path_str=f"/v1/clients/{client_id_str}/status",
            token_str=token_str,
            timeout_seconds_float=timeout_seconds_float,
        )
    except HTTPError as exc:
        if int(exc.code) == 404:
            _record_pass(check_list, "api token auth", "valid token accepted; no prior status", printer_fn)
            return True
        _record_fail(check_list, "api token auth", f"HTTP {exc.code}: {_read_error_body_str(exc)}", printer_fn)
        return False
    except Exception as exc:
        _record_fail(
            check_list,
            "api token auth",
            str(exc),
            printer_fn,
            {"exception_type_str": type(exc).__name__},
        )
        return False
    _record_pass(check_list, "api token auth", "valid token accepted", printer_fn)
    return True


def _check_sync_snapshots(
    check_list: list[DoctorCheckResult],
    *,
    api_url_str: str,
    token_str: str,
    client_id_str: str,
    releases_root_path_str: str,
    local_root_path_str: str,
    overwrite_bool: bool,
    printer_fn: PrinterFn,
) -> list[Path]:
    try:
        promoted_path_list = sync_required_snapshots(
            api_url_str=api_url_str,
            token_str=token_str,
            client_id_str=client_id_str,
            releases_root_path_str=releases_root_path_str,
            local_root_path_str=local_root_path_str,
            overwrite_bool=overwrite_bool,
        )
    except Exception as exc:
        _record_fail(
            check_list,
            "sync snapshots",
            str(exc),
            printer_fn,
            {"exception_type_str": type(exc).__name__},
        )
        return []
    _record_pass(
        check_list,
        "sync snapshots",
        f"promoted={len(promoted_path_list)}",
        printer_fn,
        {"promoted_path_list": [str(path_obj) for path_obj in promoted_path_list]},
    )
    return promoted_path_list


def _check_manifest_validation(
    check_list: list[DoctorCheckResult],
    *,
    local_root_path_obj: Path,
    promoted_path_list: list[Path],
    printer_fn: PrinterFn,
) -> None:
    for promoted_path_obj in promoted_path_list:
        profile_str = promoted_path_obj.parent.name
        snapshot_date_str = promoted_path_obj.name
        try:
            with _temporary_snapshot_root_env(local_root_path_obj):
                snapshot_manifest_obj = load_valid_snapshot_manifest(
                    profile_str,
                    snapshot_date_str=snapshot_date_str,
                )
        except Exception as exc:
            _record_fail(
                check_list,
                "manifest hash validation",
                f"profile={profile_str} error={exc}",
                printer_fn,
                {"exception_type_str": type(exc).__name__},
            )
            continue
        _record_pass(
            check_list,
            "manifest hash validation",
            f"profile={profile_str} snapshot_date={snapshot_manifest_obj.snapshot_date_ts.date().isoformat()}",
            printer_fn,
            {"manifest_hash_str": snapshot_manifest_obj.manifest_hash_str},
        )


def _check_profile_smoke(
    check_list: list[DoctorCheckResult],
    *,
    local_root_path_obj: Path,
    profile_list: list[str],
    printer_fn: PrinterFn,
) -> None:
    smoke_map_dict: dict[str, list[tuple[str, str]]] = {
        "norgate_eod_etf_plus_vix_helper": [("symbol", "SPY"), ("symbol", "$VIX")],
        "norgate_eod_sp500_pit": [("index", "S&P 500")],
        "norgate_eod_ndx_pit": [("index", "Nasdaq 100")],
        "norgate_eod_ndx_pit_plus_vxn_helper": [("index", "Nasdaq 100"), ("symbol", "$VXN")],
    }
    with _temporary_snapshot_root_env(local_root_path_obj):
        for profile_str in profile_list:
            for smoke_type_str, value_str in smoke_map_dict.get(profile_str, []):
                try:
                    if smoke_type_str == "symbol":
                        price_df = load_price_timeseries_df(value_str, data_profile_str=profile_str)
                        detail_str = f"profile={profile_str} symbol={value_str} rows={len(price_df.index)}"
                    else:
                        symbol_list, universe_df = load_index_constituent_matrix_df(
                            value_str,
                            data_profile_str=profile_str,
                        )
                        detail_str = (
                            f"profile={profile_str} index={value_str} "
                            f"symbols={len(symbol_list)} rows={len(universe_df.index)}"
                        )
                except Exception as exc:
                    _record_fail(
                        check_list,
                        "profile smoke read",
                        f"profile={profile_str} {value_str}: {exc}",
                        printer_fn,
                        {"exception_type_str": type(exc).__name__},
                    )
                    continue
                _record_pass(check_list, "profile smoke read", detail_str, printer_fn)


def _check_scheduler_heartbeat(
    check_list: list[DoctorCheckResult],
    *,
    local_root_path_obj: Path,
    profile_list: list[str],
    printer_fn: PrinterFn,
) -> None:
    with _temporary_snapshot_root_env(local_root_path_obj):
        for profile_str in profile_list:
            try:
                heartbeat_ts = load_latest_norgate_heartbeat_session_label_ts(profile_str)
            except Exception as exc:
                _record_fail(
                    check_list,
                    "scheduler snapshot heartbeat",
                    f"profile={profile_str} error={exc}",
                    printer_fn,
                    {"exception_type_str": type(exc).__name__},
                )
                continue
            if heartbeat_ts is None:
                _record_fail(
                    check_list,
                    "scheduler snapshot heartbeat",
                    f"profile={profile_str} no snapshot date",
                    printer_fn,
                )
                continue
            _record_pass(
                check_list,
                "scheduler snapshot heartbeat",
                f"profile={profile_str} latest={heartbeat_ts.date().isoformat()}",
                printer_fn,
            )


def run_norgate_client_doctor(
    *,
    api_url_str: str | None,
    token_str: str | None,
    client_id_str: str | None,
    releases_root_path_str: str | None,
    local_root_path_str: str | None,
    overwrite_bool: bool = False,
    timeout_seconds_float: float = 120.0,
    report_json_path_str: str | None = None,
    printer_fn: PrinterFn = print,
) -> DoctorReport:
    check_list: list[DoctorCheckResult] = []

    required_ok_bool = True
    required_ok_bool &= _check_required_value_bool(
        check_list,
        name_str="api url",
        value_str=api_url_str,
        printer_fn=printer_fn,
    )
    required_ok_bool &= _check_required_value_bool(
        check_list,
        name_str="api token env",
        value_str=token_str,
        printer_fn=printer_fn,
    )
    required_ok_bool &= _check_required_value_bool(
        check_list,
        name_str="client id",
        value_str=client_id_str,
        printer_fn=printer_fn,
    )
    required_ok_bool &= _check_required_value_bool(
        check_list,
        name_str="releases root",
        value_str=releases_root_path_str,
        printer_fn=printer_fn,
    )
    required_ok_bool &= _check_required_value_bool(
        check_list,
        name_str="local snapshot root",
        value_str=local_root_path_str,
        printer_fn=printer_fn,
    )

    if is_snapshot_mode_enabled_bool():
        _record_pass(
            check_list,
            "snapshot mode env",
            f"{ALPHA_USE_NORGATE_SNAPSHOT_ENV_STR}=true",
            printer_fn,
        )
    else:
        required_ok_bool = False
        _record_fail(
            check_list,
            "snapshot mode env",
            f"{ALPHA_USE_NORGATE_SNAPSHOT_ENV_STR} must be true on client VPSs",
            printer_fn,
        )

    if not required_ok_bool:
        result_str = "FAIL"
        report_obj = DoctorReport(result_str, _utc_now_str(), check_list)
        _write_report_if_requested(report_obj, report_json_path_str)
        printer_fn(f"RESULT: {result_str}")
        return report_obj

    assert api_url_str is not None
    assert token_str is not None
    assert client_id_str is not None
    assert releases_root_path_str is not None
    assert local_root_path_str is not None

    local_root_path_obj = Path(local_root_path_str).expanduser()
    os.environ[NORGATE_SNAPSHOT_ROOT_ENV_STR] = str(local_root_path_obj)
    local_root_path_obj.mkdir(parents=True, exist_ok=True)

    profile_list = _check_release_profiles(
        check_list,
        releases_root_path_str=releases_root_path_str,
        printer_fn=printer_fn,
    )
    profiles_supported_bool = bool(profile_list) and _check_server_supported_profiles_bool(
        check_list,
        profile_list=profile_list,
        printer_fn=printer_fn,
    )

    health_ok_bool = _check_api_health_bool(
        check_list,
        api_url_str=api_url_str,
        timeout_seconds_float=timeout_seconds_float,
        printer_fn=printer_fn,
    )
    token_ok_bool = False
    if health_ok_bool:
        token_ok_bool = _check_api_token_bool(
            check_list,
            api_url_str=api_url_str,
            token_str=token_str,
            client_id_str=client_id_str,
            timeout_seconds_float=timeout_seconds_float,
            printer_fn=printer_fn,
        )
    else:
        _record_skip(check_list, "api token auth", "api healthz failed", printer_fn)

    promoted_path_list: list[Path] = []
    if profiles_supported_bool and health_ok_bool and token_ok_bool:
        promoted_path_list = _check_sync_snapshots(
            check_list,
            api_url_str=api_url_str,
            token_str=token_str,
            client_id_str=client_id_str,
            releases_root_path_str=releases_root_path_str,
            local_root_path_str=str(local_root_path_obj),
            overwrite_bool=overwrite_bool,
            printer_fn=printer_fn,
        )
    else:
        _record_skip(check_list, "sync snapshots", "blocked by earlier failure", printer_fn)

    if promoted_path_list:
        _check_manifest_validation(
            check_list,
            local_root_path_obj=local_root_path_obj,
            promoted_path_list=promoted_path_list,
            printer_fn=printer_fn,
        )
        _check_profile_smoke(
            check_list,
            local_root_path_obj=local_root_path_obj,
            profile_list=profile_list,
            printer_fn=printer_fn,
        )
        _check_scheduler_heartbeat(
            check_list,
            local_root_path_obj=local_root_path_obj,
            profile_list=profile_list,
            printer_fn=printer_fn,
        )
    else:
        _record_skip(check_list, "manifest hash validation", "sync snapshots did not promote files", printer_fn)
        _record_skip(check_list, "profile smoke read", "sync snapshots did not promote files", printer_fn)
        _record_skip(check_list, "scheduler snapshot heartbeat", "sync snapshots did not promote files", printer_fn)

    result_str = "PASS" if all(result_obj.status_str != "FAIL" for result_obj in check_list) else "FAIL"
    report_obj = DoctorReport(
        result_str=result_str,
        generated_timestamp_utc_str=_utc_now_str(),
        check_list=check_list,
    )
    _write_report_if_requested(report_obj, report_json_path_str)
    printer_fn(f"RESULT: {result_str}")
    return report_obj


def _write_report_if_requested(report_obj: DoctorReport, report_json_path_str: str | None) -> None:
    if report_json_path_str is None:
        return
    report_path_obj = Path(report_json_path_str).expanduser()
    report_path_obj.parent.mkdir(parents=True, exist_ok=True)
    report_path_obj.write_text(
        json.dumps(asdict(report_obj), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    load_config_env_file(override_existing_bool=True)

    parser_obj = argparse.ArgumentParser(description="Check whether a client VPS can use Norgate snapshots.")
    parser_obj.add_argument("--api-url", default=norgate_api_url_from_env_str())
    parser_obj.add_argument("--client-id", default=env_str(NORGATE_CLIENT_ID_ENV_STR))
    parser_obj.add_argument("--releases-root", default=env_str(NORGATE_RELEASES_ROOT_ENV_STR))
    parser_obj.add_argument("--local-root", default=env_str(NORGATE_SNAPSHOT_ROOT_ENV_STR))
    parser_obj.add_argument("--overwrite", action="store_true")
    parser_obj.add_argument("--timeout-seconds", type=float, default=120.0)
    parser_obj.add_argument("--report-json", default=None)
    args_obj = parser_obj.parse_args()

    report_obj = run_norgate_client_doctor(
        api_url_str=None if args_obj.api_url is None else str(args_obj.api_url),
        token_str=os.getenv(NORGATE_API_TOKEN_ENV_STR, "").strip(),
        client_id_str=None if args_obj.client_id is None else str(args_obj.client_id),
        releases_root_path_str=None if args_obj.releases_root is None else str(args_obj.releases_root),
        local_root_path_str=None if args_obj.local_root is None else str(args_obj.local_root),
        overwrite_bool=bool(args_obj.overwrite),
        timeout_seconds_float=float(args_obj.timeout_seconds),
        report_json_path_str=None if args_obj.report_json is None else str(args_obj.report_json),
    )
    return 0 if report_obj.result_str == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
