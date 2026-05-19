from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.error import HTTPError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import pandas as pd

repo_root_path = Path(__file__).resolve().parents[1]
repo_root_str = str(repo_root_path)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

from data.norgate_snapshot_store import (
    NORGATE_SNAPSHOT_ROOT_ENV_STR,
    load_valid_snapshot_manifest,
)
from scripts.export_norgate_snapshot import export_profile_snapshot
from scripts.serve_norgate_snapshot_api import (
    NORGATE_API_TOKEN_ENV_STR,
    NORGATE_API_TOKEN_HEADER_STR,
)
from scripts.norgate_config_env import (
    NORGATE_CLIENT_ID_ENV_STR,
    NORGATE_SERVICE_ROOT_ENV_STR,
    env_str,
    load_config_env_file,
    norgate_api_url_from_env_str,
)


DEFAULT_CLIENT_ID_STR = "doctor_server"
DEFAULT_PROFILE_STR = "norgate_eod_etf_plus_vix_helper"
DEFAULT_MIN_FREE_GB_FLOAT = 5.0

NorgateLoaderFn = Callable[[], Any]
ExporterFn = Callable[..., Path]
ManifestLoaderFn = Callable[..., Any]
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


class DoctorFailure(RuntimeError):
    pass


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
) -> DoctorCheckResult:
    result_obj = DoctorCheckResult(
        name_str=name_str,
        status_str=status_str,
        detail_str=detail_str,
        timestamp_utc_str=_utc_now_str(),
        metadata_dict=dict(metadata_dict or {}),
    )
    check_list.append(result_obj)
    printer_fn(_format_result_line(result_obj))
    return result_obj


def _record_pass(
    check_list: list[DoctorCheckResult],
    name_str: str,
    detail_str: str,
    printer_fn: PrinterFn,
    metadata_dict: dict[str, Any] | None = None,
) -> DoctorCheckResult:
    return _record_result(
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
) -> DoctorCheckResult:
    return _record_result(
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
    metadata_dict: dict[str, Any] | None = None,
) -> DoctorCheckResult:
    return _record_result(
        check_list,
        name_str=name_str,
        status_str="SKIP",
        detail_str=detail_str,
        printer_fn=printer_fn,
        metadata_dict=metadata_dict,
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


def _http_json_request_dict(
    *,
    api_url_str: str,
    url_path_str: str,
    method_str: str = "GET",
    token_str: str | None = None,
    payload_dict: dict[str, Any] | None = None,
    timeout_seconds_float: float = 120.0,
) -> dict[str, Any]:
    request_data_bytes = None
    header_dict: dict[str, str] = {}
    if token_str is not None:
        header_dict[NORGATE_API_TOKEN_HEADER_STR] = token_str
    if payload_dict is not None:
        request_data_bytes = json.dumps(payload_dict).encode("utf-8")
        header_dict["Content-Type"] = "application/json"

    request_obj = Request(
        _build_url_str(api_url_str, url_path_str),
        data=request_data_bytes,
        method=method_str,
        headers=header_dict,
    )
    with urlopen(request_obj, timeout=timeout_seconds_float) as response_obj:
        payload_obj = json.loads(response_obj.read().decode("utf-8"))
    if not isinstance(payload_obj, dict):
        raise DoctorFailure("HTTP response was not a JSON object.")
    return payload_obj


def _load_norgatedata_module_obj() -> Any:
    return importlib.import_module("norgatedata")


def _query_norgate_symbol(
    norgatedata_module_obj: Any,
    symbol_str: str,
) -> tuple[int, str]:
    price_df = norgatedata_module_obj.price_timeseries(
        symbol_str,
        timeseriesformat="pandas-dataframe",
    )
    if price_df is None or len(price_df.index) == 0:
        raise DoctorFailure(f"Norgate returned no rows for {symbol_str}.")
    latest_session_str = pd.Timestamp(price_df.index[-1]).date().isoformat()
    return int(len(price_df.index)), latest_session_str


def _check_required_python_modules(check_list: list[DoctorCheckResult], printer_fn: PrinterFn) -> None:
    module_name_list = ["pandas"]
    for module_name_str in module_name_list:
        importlib.import_module(module_name_str)
    _record_pass(
        check_list,
        "python runtime modules",
        ",".join(module_name_list),
        printer_fn,
    )


def _check_token_env(
    check_list: list[DoctorCheckResult],
    printer_fn: PrinterFn,
) -> str | None:
    token_str = os.getenv(NORGATE_API_TOKEN_ENV_STR, "").strip()
    if not token_str:
        _record_fail(
            check_list,
            "api token env",
            f"{NORGATE_API_TOKEN_ENV_STR} is not set",
            printer_fn,
        )
        return None
    _record_pass(check_list, "api token env", f"{NORGATE_API_TOKEN_ENV_STR}=set", printer_fn)
    return token_str


def _check_service_root_writable(
    check_list: list[DoctorCheckResult],
    service_root_path_obj: Path,
    printer_fn: PrinterFn,
) -> bool:
    try:
        service_root_path_obj.mkdir(parents=True, exist_ok=True)
        test_path_obj = service_root_path_obj / ".doctor_write_test.tmp"
        test_path_obj.write_text("ok", encoding="utf-8")
        test_path_obj.unlink()
    except Exception as exc:
        _record_fail(
            check_list,
            "service root writable",
            str(exc),
            printer_fn,
            {"exception_type_str": type(exc).__name__},
        )
        return False
    _record_pass(check_list, "service root writable", str(service_root_path_obj), printer_fn)
    return True


def _check_disk_space(
    check_list: list[DoctorCheckResult],
    service_root_path_obj: Path,
    min_free_gb_float: float,
    printer_fn: PrinterFn,
) -> None:
    try:
        service_root_path_obj.mkdir(parents=True, exist_ok=True)
        usage_obj = shutil.disk_usage(service_root_path_obj)
        free_gb_float = float(usage_obj.free) / float(1024**3)
        detail_str = f"free_gb={free_gb_float:.2f} min_free_gb={min_free_gb_float:.2f}"
        metadata_dict = {
            "free_gb_float": free_gb_float,
            "min_free_gb_float": min_free_gb_float,
        }
        if free_gb_float < min_free_gb_float:
            _record_fail(check_list, "disk free space", detail_str, printer_fn, metadata_dict)
            return
        _record_pass(check_list, "disk free space", detail_str, printer_fn, metadata_dict)
    except Exception as exc:
        _record_fail(
            check_list,
            "disk free space",
            str(exc),
            printer_fn,
            {"exception_type_str": type(exc).__name__},
        )


def _check_norgatedata_import(
    check_list: list[DoctorCheckResult],
    norgate_loader_fn: NorgateLoaderFn,
    printer_fn: PrinterFn,
) -> Any | None:
    try:
        norgatedata_module_obj = norgate_loader_fn()
    except Exception as exc:
        _record_fail(
            check_list,
            "norgatedata import",
            str(exc),
            printer_fn,
            {"exception_type_str": type(exc).__name__},
        )
        return None
    _record_pass(check_list, "norgatedata import", "ok", printer_fn)
    return norgatedata_module_obj


def _check_norgate_symbol(
    check_list: list[DoctorCheckResult],
    *,
    norgatedata_module_obj: Any,
    symbol_str: str,
    printer_fn: PrinterFn,
) -> str | None:
    try:
        row_count_int, latest_session_str = _query_norgate_symbol(norgatedata_module_obj, symbol_str)
    except Exception as exc:
        _record_fail(
            check_list,
            f"{symbol_str} query",
            str(exc),
            printer_fn,
            {"exception_type_str": type(exc).__name__},
        )
        return None
    _record_pass(
        check_list,
        f"{symbol_str} query",
        f"latest={latest_session_str} rows={row_count_int}",
        printer_fn,
        {"latest_session_str": latest_session_str, "row_count_int": row_count_int},
    )
    return latest_session_str


def _check_api_healthz(
    check_list: list[DoctorCheckResult],
    *,
    api_url_str: str,
    timeout_seconds_float: float,
    printer_fn: PrinterFn,
) -> bool:
    try:
        health_dict = _http_json_request_dict(
            api_url_str=api_url_str,
            url_path_str="/healthz",
            timeout_seconds_float=timeout_seconds_float,
        )
        if health_dict.get("status_str") != "ok":
            raise DoctorFailure(f"unexpected health payload: {health_dict}")
    except Exception as exc:
        _record_fail(
            check_list,
            "api healthz",
            str(exc),
            printer_fn,
            {"exception_type_str": type(exc).__name__},
        )
        return False
    _record_pass(check_list, "api healthz", f"{api_url_str}/healthz", printer_fn)
    return True


def _check_api_rejects_missing_token(
    check_list: list[DoctorCheckResult],
    *,
    api_url_str: str,
    client_id_str: str,
    timeout_seconds_float: float,
    printer_fn: PrinterFn,
) -> None:
    try:
        _http_json_request_dict(
            api_url_str=api_url_str,
            url_path_str=f"/v1/clients/{client_id_str}/status",
            timeout_seconds_float=timeout_seconds_float,
        )
    except HTTPError as exc:
        if int(exc.code) == 401:
            _record_pass(check_list, "api rejects missing token", "HTTP 401", printer_fn)
            return
        _record_fail(check_list, "api rejects missing token", f"HTTP {exc.code}", printer_fn)
        return
    except Exception as exc:
        _record_fail(
            check_list,
            "api rejects missing token",
            str(exc),
            printer_fn,
            {"exception_type_str": type(exc).__name__},
        )
        return
    _record_fail(
        check_list,
        "api rejects missing token",
        "tokenless request unexpectedly succeeded",
        printer_fn,
    )


def _post_requirements_dict(
    *,
    api_url_str: str,
    token_str: str,
    client_id_str: str,
    profile_str: str,
    timeout_seconds_float: float,
) -> dict[str, Any]:
    return _http_json_request_dict(
        api_url_str=api_url_str,
        url_path_str=f"/v1/clients/{client_id_str}/requirements",
        method_str="POST",
        token_str=token_str,
        payload_dict={"profile_list": [profile_str]},
        timeout_seconds_float=timeout_seconds_float,
    )


def _extract_snapshot_date_str(response_dict: dict[str, Any], profile_str: str) -> str:
    if response_dict.get("status_str") != "ready":
        raise DoctorFailure(f"export status is not ready: {response_dict}")
    snapshot_file_list = response_dict.get("snapshot_file_list")
    if not isinstance(snapshot_file_list, list) or not snapshot_file_list:
        raise DoctorFailure(f"requirements response has no snapshot files: {response_dict}")

    for snapshot_file_obj in snapshot_file_list:
        if not isinstance(snapshot_file_obj, dict):
            continue
        if str(snapshot_file_obj.get("profile_str", "")) != profile_str:
            continue
        snapshot_date_str = str(snapshot_file_obj.get("snapshot_date_str", "")).strip()
        if snapshot_date_str:
            return snapshot_date_str
    raise DoctorFailure(f"snapshot date missing for profile {profile_str}: {response_dict}")


def _check_audit_files(
    check_list: list[DoctorCheckResult],
    *,
    service_root_path_obj: Path,
    client_id_str: str,
    printer_fn: PrinterFn,
) -> None:
    client_dir_path_obj = service_root_path_obj / client_id_str
    required_file_list = [
        "required_profiles.txt",
        "accepted_profiles.txt",
        "export_status.json",
        "export.log",
    ]
    missing_file_list = [
        file_name_str
        for file_name_str in required_file_list
        if not (client_dir_path_obj / file_name_str).exists()
    ]
    if missing_file_list:
        _record_fail(
            check_list,
            "server audit files",
            f"missing files: {missing_file_list}",
            printer_fn,
            {"client_dir_str": str(client_dir_path_obj), "missing_file_list": missing_file_list},
        )
        return
    _record_pass(check_list, "server audit files", str(client_dir_path_obj), printer_fn)


def _validate_snapshot(
    check_list: list[DoctorCheckResult],
    *,
    snapshot_root_path_obj: Path,
    profile_str: str,
    snapshot_date_str: str,
    manifest_loader_fn: ManifestLoaderFn,
    printer_fn: PrinterFn,
) -> None:
    try:
        with _temporary_snapshot_root_env(snapshot_root_path_obj):
            snapshot_manifest_obj = manifest_loader_fn(
                profile_str,
                snapshot_date_str=snapshot_date_str,
            )
        validated_date_str = snapshot_manifest_obj.snapshot_date_ts.date().isoformat()
        metadata_dict = {
            "profile_str": profile_str,
            "snapshot_date_str": validated_date_str,
            "manifest_hash_str": snapshot_manifest_obj.manifest_hash_str,
        }
        _record_pass(
            check_list,
            "manifest hash validation",
            f"profile={profile_str} snapshot_date={validated_date_str}",
            printer_fn,
            metadata_dict,
        )
    except Exception as exc:
        _record_fail(
            check_list,
            "manifest hash validation",
            str(exc),
            printer_fn,
            {
                "exception_type_str": type(exc).__name__,
                "profile_str": profile_str,
                "snapshot_date_str": snapshot_date_str,
            },
        )


def _run_local_export_check(
    check_list: list[DoctorCheckResult],
    *,
    service_root_path_obj: Path,
    client_id_str: str,
    profile_str: str,
    start_date_str: str,
    exporter_fn: ExporterFn,
    manifest_loader_fn: ManifestLoaderFn,
    printer_fn: PrinterFn,
) -> None:
    snapshot_root_path_obj = service_root_path_obj / client_id_str / "snapshots"
    try:
        snapshot_dir_path_obj = exporter_fn(
            snapshot_root_str=str(snapshot_root_path_obj),
            profile_str=profile_str,
            snapshot_date_str=None,
            start_date_str=start_date_str,
            end_date_str=None,
            overwrite_bool=False,
        )
    except Exception as exc:
        _record_fail(
            check_list,
            f"export {profile_str}",
            str(exc),
            printer_fn,
            {"exception_type_str": type(exc).__name__},
        )
        return

    snapshot_date_str = Path(snapshot_dir_path_obj).name
    _record_pass(
        check_list,
        f"export {profile_str}",
        f"snapshot_date={snapshot_date_str}",
        printer_fn,
        {"snapshot_dir_str": str(snapshot_dir_path_obj)},
    )
    _validate_snapshot(
        check_list,
        snapshot_root_path_obj=snapshot_root_path_obj,
        profile_str=profile_str,
        snapshot_date_str=snapshot_date_str,
        manifest_loader_fn=manifest_loader_fn,
        printer_fn=printer_fn,
    )


def _run_api_export_check(
    check_list: list[DoctorCheckResult],
    *,
    api_url_str: str,
    service_root_path_obj: Path,
    token_str: str,
    client_id_str: str,
    profile_str: str,
    timeout_seconds_float: float,
    manifest_loader_fn: ManifestLoaderFn,
    printer_fn: PrinterFn,
    record_token_auth_bool: bool,
) -> None:
    try:
        response_dict = _post_requirements_dict(
            api_url_str=api_url_str,
            token_str=token_str,
            client_id_str=client_id_str,
            profile_str=profile_str,
            timeout_seconds_float=timeout_seconds_float,
        )
    except HTTPError as exc:
        check_name_str = "api token auth" if int(exc.code) == 401 else f"export {profile_str}"
        _record_fail(check_list, check_name_str, f"HTTP {exc.code}", printer_fn)
        return
    except Exception as exc:
        _record_fail(
            check_list,
            f"export {profile_str}",
            str(exc),
            printer_fn,
            {"exception_type_str": type(exc).__name__},
        )
        return

    if record_token_auth_bool:
        _record_pass(check_list, "api token auth", "valid token accepted", printer_fn)

    try:
        snapshot_date_str = _extract_snapshot_date_str(response_dict, profile_str)
    except Exception as exc:
        _record_fail(
            check_list,
            f"export {profile_str}",
            str(exc),
            printer_fn,
            {"response_dict": response_dict, "exception_type_str": type(exc).__name__},
        )
        return

    _record_pass(
        check_list,
        f"export {profile_str}",
        f"snapshot_date={snapshot_date_str}",
        printer_fn,
        {"response_dict": response_dict},
    )
    _check_audit_files(
        check_list,
        service_root_path_obj=service_root_path_obj,
        client_id_str=client_id_str,
        printer_fn=printer_fn,
    )
    _validate_snapshot(
        check_list,
        snapshot_root_path_obj=service_root_path_obj / client_id_str / "snapshots",
        profile_str=profile_str,
        snapshot_date_str=snapshot_date_str,
        manifest_loader_fn=manifest_loader_fn,
        printer_fn=printer_fn,
    )


def run_norgate_server_doctor(
    *,
    service_root_path_str: str,
    api_url_str: str | None = None,
    profile_str: str = DEFAULT_PROFILE_STR,
    heavy_profile_str: str | None = None,
    client_id_str: str = DEFAULT_CLIENT_ID_STR,
    start_date_str: str = "1990-01-01",
    min_free_gb_float: float = DEFAULT_MIN_FREE_GB_FLOAT,
    timeout_seconds_float: float = 120.0,
    report_json_path_str: str | None = None,
    norgate_loader_fn: NorgateLoaderFn = _load_norgatedata_module_obj,
    exporter_fn: ExporterFn = export_profile_snapshot,
    manifest_loader_fn: ManifestLoaderFn = load_valid_snapshot_manifest,
    printer_fn: PrinterFn = print,
) -> DoctorReport:
    check_list: list[DoctorCheckResult] = []
    service_root_path_obj = Path(service_root_path_str).expanduser()
    profile_list = [profile_str]
    if heavy_profile_str:
        profile_list.append(heavy_profile_str)
    profile_list = list(dict.fromkeys(profile_list))

    _check_required_python_modules(check_list, printer_fn)
    token_str = _check_token_env(check_list, printer_fn)
    _check_service_root_writable(check_list, service_root_path_obj, printer_fn)
    _check_disk_space(check_list, service_root_path_obj, min_free_gb_float, printer_fn)

    norgatedata_module_obj = _check_norgatedata_import(
        check_list,
        norgate_loader_fn,
        printer_fn,
    )
    latest_session_str: str | None = None
    if norgatedata_module_obj is not None:
        spy_latest_session_str = _check_norgate_symbol(
            check_list,
            norgatedata_module_obj=norgatedata_module_obj,
            symbol_str="SPY",
            printer_fn=printer_fn,
        )
        spx_latest_session_str = _check_norgate_symbol(
            check_list,
            norgatedata_module_obj=norgatedata_module_obj,
            symbol_str="$SPX",
            printer_fn=printer_fn,
        )
        latest_session_str = spx_latest_session_str or spy_latest_session_str
    else:
        _record_skip(check_list, "SPY query", "norgatedata import failed", printer_fn)
        _record_skip(check_list, "$SPX query", "norgatedata import failed", printer_fn)

    if latest_session_str:
        _record_pass(check_list, "latest Norgate session", latest_session_str, printer_fn)
    else:
        _record_fail(check_list, "latest Norgate session", "not found", printer_fn)

    if api_url_str:
        api_health_ok_bool = _check_api_healthz(
            check_list,
            api_url_str=api_url_str,
            timeout_seconds_float=timeout_seconds_float,
            printer_fn=printer_fn,
        )
        if api_health_ok_bool:
            _check_api_rejects_missing_token(
                check_list,
                api_url_str=api_url_str,
                client_id_str=client_id_str,
                timeout_seconds_float=timeout_seconds_float,
                printer_fn=printer_fn,
            )
            if token_str is None:
                _record_skip(
                    check_list,
                    "api token auth",
                    f"{NORGATE_API_TOKEN_ENV_STR} is not set",
                    printer_fn,
                )
            else:
                for profile_index_int, current_profile_str in enumerate(profile_list):
                    _run_api_export_check(
                        check_list,
                        api_url_str=api_url_str,
                        service_root_path_obj=service_root_path_obj,
                        token_str=token_str,
                        client_id_str=client_id_str,
                        profile_str=current_profile_str,
                        timeout_seconds_float=timeout_seconds_float,
                        manifest_loader_fn=manifest_loader_fn,
                        printer_fn=printer_fn,
                        record_token_auth_bool=profile_index_int == 0,
                    )
        else:
            _record_skip(check_list, "api rejects missing token", "api healthz failed", printer_fn)
            _record_skip(check_list, "api token auth", "api healthz failed", printer_fn)
    else:
        _record_skip(check_list, "api checks", "api-url was not provided", printer_fn)
        for current_profile_str in profile_list:
            _run_local_export_check(
                check_list,
                service_root_path_obj=service_root_path_obj,
                client_id_str=client_id_str,
                profile_str=current_profile_str,
                start_date_str=start_date_str,
                exporter_fn=exporter_fn,
                manifest_loader_fn=manifest_loader_fn,
                printer_fn=printer_fn,
            )

    result_str = "PASS" if all(result_obj.status_str != "FAIL" for result_obj in check_list) else "FAIL"
    report_obj = DoctorReport(
        result_str=result_str,
        generated_timestamp_utc_str=_utc_now_str(),
        check_list=check_list,
    )
    if report_json_path_str is not None:
        report_path_obj = Path(report_json_path_str).expanduser()
        report_path_obj.parent.mkdir(parents=True, exist_ok=True)
        report_path_obj.write_text(
            json.dumps(asdict(report_obj), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    printer_fn(f"RESULT: {result_str}")
    return report_obj


def main() -> int:
    load_config_env_file()

    parser_obj = argparse.ArgumentParser(description="Check whether the Norgate artifact server is ready.")
    parser_obj.add_argument(
        "--service-root",
        default=env_str(NORGATE_SERVICE_ROOT_ENV_STR),
        help="Norgate service root directory.",
    )
    parser_obj.add_argument(
        "--api-url",
        default=norgate_api_url_from_env_str(),
        help="Base API URL, for example http://100.123.13.69:8787.",
    )
    parser_obj.add_argument(
        "--profile",
        default=DEFAULT_PROFILE_STR,
        help="Small EOD profile to request/export.",
    )
    parser_obj.add_argument(
        "--heavy-profile",
        default=None,
        help="Optional heavy profile to request/export.",
    )
    parser_obj.add_argument(
        "--client-id",
        default=env_str(NORGATE_CLIENT_ID_ENV_STR, DEFAULT_CLIENT_ID_STR),
        help="Doctor client id.",
    )
    parser_obj.add_argument(
        "--start-date",
        default="1990-01-01",
        help="First historical date for local exports.",
    )
    parser_obj.add_argument(
        "--min-free-gb",
        type=float,
        default=DEFAULT_MIN_FREE_GB_FLOAT,
    )
    parser_obj.add_argument("--timeout-seconds", type=float, default=120.0)
    parser_obj.add_argument(
        "--report-json",
        default=None,
        help="Optional JSON report output path.",
    )
    args_obj = parser_obj.parse_args()

    if not args_obj.service_root:
        raise RuntimeError(f"--service-root or {NORGATE_SERVICE_ROOT_ENV_STR} must be set before running the doctor.")

    report_obj = run_norgate_server_doctor(
        service_root_path_str=str(args_obj.service_root),
        api_url_str=None if args_obj.api_url is None else str(args_obj.api_url),
        profile_str=str(args_obj.profile),
        heavy_profile_str=None if args_obj.heavy_profile is None else str(args_obj.heavy_profile),
        client_id_str=str(args_obj.client_id),
        start_date_str=str(args_obj.start_date),
        min_free_gb_float=float(args_obj.min_free_gb),
        timeout_seconds_float=float(args_obj.timeout_seconds),
        report_json_path_str=None if args_obj.report_json is None else str(args_obj.report_json),
    )
    return 0 if report_obj.result_str == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
