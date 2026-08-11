"""CLI for the once-daily IBKR Flex performance shadow sync.

The command reads token/query settings from ignored ``config.env``. Secrets
are never written to the database, UI, or logs.
"""

from __future__ import annotations

import argparse
from datetime import UTC, date, datetime, timedelta
import hashlib
import os
from pathlib import Path
import sys
import time
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo
import xml.etree.ElementTree as ElementTree

import pandas as pd

from alpha.live import runner, scheduler_utils
from alpha.live.dashboard import (
    DEFAULT_CONFIG_PATH_STR,
    DEFAULT_RELEASES_ROOT_PATH_STR,
    load_dashboard_config,
    resolve_db_path_for_release_str,
)
from alpha.live.ibkr_performance import (
    PerformanceContractError,
    PerformanceStore,
    PodPerformanceBinding,
    resolve_performance_db_path_str,
    status_json_str,
)
from alpha.live.release_manifest import load_release_list
from alpha.live.state_store import LiveStateStore
from scripts.norgate_config_env import load_config_env_file


FLEX_TOKEN_ENV_STR = "IBKR_FLEX_TOKEN_STR"
FLEX_QUERY_ID_ENV_STR = "IBKR_FLEX_QUERY_ID_STR"
FLEX_QUERY_NAME_ENV_STR = "IBKR_FLEX_QUERY_NAME_STR"
DEFAULT_QUERY_NAME_STR = "ALPHA_DAILY_TWR"
FLEX_SEND_URL_STR = (
    "https://ndcdyn.interactivebrokers.com/AccountManagement/"
    "FlexWebService/SendRequest"
)
FLEX_GET_URL_STR = (
    "https://ndcdyn.interactivebrokers.com/AccountManagement/"
    "FlexWebService/GetStatement"
)
FLEX_API_VERSION_STR = "3"
FLEX_USER_AGENT_STR = "alpha-super/ibkr-performance-shadow-v1"
FLEX_HTTP_TIMEOUT_SECONDS_FLOAT = 30.0
FLEX_POLL_SECONDS_FLOAT = 10.0
FLEX_POLL_ATTEMPT_COUNT_INT = 12
SYNC_LOOKBACK_CALENDAR_DAY_COUNT_INT = 35
TEMPORARY_ERROR_CODE_SET = {
    "1001",
    "1003",
    "1004",
    "1005",
    "1006",
    "1007",
    "1008",
    "1009",
    "1019",
    "1021",
}


class FlexSyncError(RuntimeError):
    """A sanitized Flex transport/service error safe for operator output."""


def build_live_binding_obj_list(
    *,
    releases_root_path_str: str = DEFAULT_RELEASES_ROOT_PATH_STR,
    config_path_str: str = DEFAULT_CONFIG_PATH_STR,
) -> list[PodPerformanceBinding]:
    release_obj_list = [
        release_obj
        for release_obj in load_release_list(releases_root_path_str)
        if release_obj.mode_str == "live"
    ]
    if not release_obj_list:
        raise PerformanceContractError("No LIVE release manifests were found.")
    config_obj = load_dashboard_config(config_path_str)
    binding_obj_list: list[PodPerformanceBinding] = []
    seen_account_route_set: set[str] = set()
    for release_obj in release_obj_list:
        if release_obj.account_route_str in seen_account_route_set:
            raise PerformanceContractError(
                f"LIVE account {release_obj.account_route_str} is assigned more than once."
            )
        seen_account_route_set.add(release_obj.account_route_str)
        if release_obj.session_calendar_id_str != "XNYS":
            raise PerformanceContractError(
                f"Pod {release_obj.pod_id_str} uses {release_obj.session_calendar_id_str}; "
                "the performance shadow supports XNYS only."
            )
        db_path_str = resolve_db_path_for_release_str(release_obj, config_obj)
        first_eod_date_str, last_eod_date_str = _eod_boundary_date_tuple(
            release_obj, db_path_str
        )
        return_start_date_str = (
            None
            if first_eod_date_str is None
            else _next_session_date_str(first_eod_date_str)
        )
        return_end_date_str = (
            None if release_obj.enabled_bool else last_eod_date_str
        )
        binding_obj_list.append(
            PodPerformanceBinding(
                account_route_str=release_obj.account_route_str,
                pod_id_str=release_obj.pod_id_str,
                return_start_date_str=return_start_date_str,
                return_end_date_str=return_end_date_str,
                enabled_bool=release_obj.enabled_bool,
                session_calendar_id_str=release_obj.session_calendar_id_str,
            )
        )
    return sorted(
        binding_obj_list, key=lambda binding_obj: binding_obj.account_route_str
    )


def fetch_flex_statement_xml_str(
    *,
    token_str: str,
    query_id_str: str,
    from_date_str: str,
    to_date_str: str,
    urlopen_fn: Callable = urlopen,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> str:
    send_query_dict = {
        "t": token_str,
        "q": query_id_str,
        "fd": from_date_str.replace("-", ""),
        "td": to_date_str.replace("-", ""),
        "v": FLEX_API_VERSION_STR,
    }
    send_xml_text_str = _http_get_text_str(
        FLEX_SEND_URL_STR,
        send_query_dict,
        "SendRequest",
        urlopen_fn,
    )
    reference_code_str = _success_reference_code_str(send_xml_text_str)
    get_query_dict = {
        "t": token_str,
        "q": reference_code_str,
        "v": FLEX_API_VERSION_STR,
    }
    for attempt_index_int in range(FLEX_POLL_ATTEMPT_COUNT_INT):
        statement_xml_text_str = _http_get_text_str(
            FLEX_GET_URL_STR,
            get_query_dict,
            "GetStatement",
            urlopen_fn,
        )
        root_obj = _xml_root_obj(statement_xml_text_str)
        if root_obj.tag == "FlexQueryResponse":
            return statement_xml_text_str
        error_code_str, _ = _flex_error_tuple(root_obj)
        if (
            error_code_str in TEMPORARY_ERROR_CODE_SET
            and attempt_index_int + 1 < FLEX_POLL_ATTEMPT_COUNT_INT
        ):
            sleep_fn(FLEX_POLL_SECONDS_FLOAT)
            continue
        raise FlexSyncError(
            f"IBKR Flex GetStatement failed ({error_code_str or 'unknown'})."
        )
    raise FlexSyncError("IBKR Flex statement was not ready within two minutes.")


def sync_range_bool(
    *,
    db_path_str: str,
    binding_obj_list: list[PodPerformanceBinding],
    from_date_str: str,
    to_date_str: str,
    query_name_str: str,
    token_str: str,
    query_id_str: str,
    urlopen_fn: Callable = urlopen,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> bool:
    store_obj = PerformanceStore(db_path_str)
    try:
        xml_text_str = fetch_flex_statement_xml_str(
            token_str=token_str,
            query_id_str=query_id_str,
            from_date_str=from_date_str,
            to_date_str=to_date_str,
            urlopen_fn=urlopen_fn,
            sleep_fn=sleep_fn,
        )
        changed_bool = store_obj.replace_range(
            xml_text_str=xml_text_str,
            query_name_str=query_name_str,
            request_from_date_str=from_date_str,
            request_to_date_str=to_date_str,
            binding_obj_list=binding_obj_list,
        )
    except Exception as exception_obj:
        store_obj.record_sync_attempt(
            status_str="failed",
            request_from_date_str=from_date_str,
            request_to_date_str=to_date_str,
            detail_str=_safe_operator_error_str(exception_obj),
        )
        raise
    store_obj.record_sync_attempt(
        status_str="success",
        request_from_date_str=from_date_str,
        request_to_date_str=to_date_str,
        detail_str=None,
    )
    return changed_bool


def import_xml_bool(
    *,
    db_path_str: str,
    xml_path_str: str,
    binding_obj_list: list[PodPerformanceBinding],
    query_name_str: str,
    replace_bool: bool = False,
) -> bool:
    xml_path_obj = Path(xml_path_str).expanduser().resolve()
    xml_text_str = xml_path_obj.read_text(encoding="utf-8")
    from_date_str, to_date_str = _xml_date_range_tuple(xml_text_str)
    store_obj = PerformanceStore(db_path_str)
    checksum_str = hashlib.sha256(xml_text_str.encode("utf-8")).hexdigest()
    if (
        store_obj.has_rows_in_range_bool(from_date_str, to_date_str)
        and not store_obj.contains_checksum_bool(checksum_str)
        and not replace_bool
    ):
        raise FlexSyncError(
            "The XML overlaps existing performance rows; use --replace only "
            "after explicitly reviewing the correction."
        )
    return store_obj.replace_range(
        xml_text_str=xml_text_str,
        query_name_str=query_name_str,
        request_from_date_str=from_date_str,
        request_to_date_str=to_date_str,
        binding_obj_list=binding_obj_list,
    )


def bootstrap_remote(
    *,
    db_path_str: str,
    binding_obj_list: list[PodPerformanceBinding],
    from_date_str: str,
    to_date_str: str,
    query_name_str: str,
    token_str: str,
    query_id_str: str,
    replace_bool: bool,
) -> None:
    db_path_obj = Path(db_path_str).expanduser()
    temporary_db_path_obj = db_path_obj.with_name(db_path_obj.name + ".bootstrap.tmp")
    bootstrap_lock_path_obj = _acquire_bootstrap_lock_path_obj(db_path_obj)
    try:
        if db_path_obj.exists() and not replace_bool:
            raise FlexSyncError(
                f"Performance database already exists at {db_path_obj}; "
                "use --replace only for an intentional full rebuild."
            )
        _cleanup_bootstrap_temp_files(
            temporary_db_path_obj, suppress_cleanup_error_bool=False
        )
        for chunk_from_date_str, chunk_to_date_str in _date_chunk_tuple_list(
            from_date_str, to_date_str
        ):
            sync_range_bool(
                db_path_str=str(temporary_db_path_obj),
                binding_obj_list=binding_obj_list,
                from_date_str=chunk_from_date_str,
                to_date_str=chunk_to_date_str,
                query_name_str=query_name_str,
                token_str=token_str,
                query_id_str=query_id_str,
            )
        db_path_obj.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary_db_path_obj, db_path_obj)
    except Exception:
        _cleanup_bootstrap_temp_files(
            temporary_db_path_obj, suppress_cleanup_error_bool=True
        )
        raise
    finally:
        try:
            bootstrap_lock_path_obj.unlink(missing_ok=True)
        except OSError:
            # The work result or original failure is more important than a
            # stale lock file; a verified stale lock can be removed manually.
            pass


def _acquire_bootstrap_lock_path_obj(db_path_obj: Path) -> Path:
    lock_path_obj = db_path_obj.with_name(db_path_obj.name + ".bootstrap.lock")
    db_path_obj.parent.mkdir(parents=True, exist_ok=True)
    try:
        file_descriptor_int = os.open(
            str(lock_path_obj), os.O_CREAT | os.O_EXCL | os.O_WRONLY
        )
    except FileExistsError as exception_obj:
        raise FlexSyncError(
            "Another performance bootstrap is already running, or a prior run "
            f"left a stale lock at {lock_path_obj}. Verify that no performance "
            "bootstrap process is active before removing that lock."
        ) from exception_obj
    with os.fdopen(file_descriptor_int, "w", encoding="utf-8") as lock_file_obj:
        lock_file_obj.write(str(os.getpid()))
    return lock_path_obj


def _cleanup_bootstrap_temp_files(
    temporary_db_path_obj: Path, *, suppress_cleanup_error_bool: bool
) -> None:
    temporary_path_obj_list = [
        temporary_db_path_obj,
        Path(str(temporary_db_path_obj) + "-journal"),
        Path(str(temporary_db_path_obj) + "-wal"),
        Path(str(temporary_db_path_obj) + "-shm"),
    ]
    for temporary_path_obj in temporary_path_obj_list:
        try:
            temporary_path_obj.unlink(missing_ok=True)
        except OSError as exception_obj:
            if suppress_cleanup_error_bool:
                # Preserve the original Flex/validation failure. Windows cleanup
                # errors must never replace the reason bootstrap failed.
                continue
            raise FlexSyncError(
                "Cannot clean the previous performance bootstrap temporary file "
                f"{temporary_path_obj}. Verify no performance sync is running, "
                "then remove the temporary files before retrying."
            ) from exception_obj


def main() -> int:
    load_config_env_file(override_existing_bool=True)
    parser_obj = argparse.ArgumentParser(
        prog="python -m alpha.live.ibkr_performance_sync",
        description="Import IBKR Flex TWR into the read-only performance shadow.",
    )
    parser_obj.add_argument(
        "--db-path", default=resolve_performance_db_path_str()
    )
    parser_obj.add_argument(
        "--releases-root", default=DEFAULT_RELEASES_ROOT_PATH_STR
    )
    parser_obj.add_argument("--config-path", default=DEFAULT_CONFIG_PATH_STR)
    subparser_obj = parser_obj.add_subparsers(dest="command_name_str", required=True)

    bootstrap_parser_obj = subparser_obj.add_parser("bootstrap")
    bootstrap_parser_obj.add_argument("--xml")
    bootstrap_parser_obj.add_argument("--from-date")
    bootstrap_parser_obj.add_argument("--to-date")
    bootstrap_parser_obj.add_argument("--replace", action="store_true")

    subparser_obj.add_parser("sync")
    status_parser_obj = subparser_obj.add_parser("status")
    status_parser_obj.add_argument("--json", action="store_true")

    argument_obj = parser_obj.parse_args()
    if argument_obj.command_name_str == "status":
        print(status_json_str(argument_obj.db_path))
        return 0

    try:
        binding_obj_list = build_live_binding_obj_list(
            releases_root_path_str=argument_obj.releases_root,
            config_path_str=argument_obj.config_path,
        )
    except Exception as exception_obj:
        _record_preflight_failure_if_daily_sync(argument_obj, exception_obj)
        raise
    query_name_str = os.getenv(
        FLEX_QUERY_NAME_ENV_STR, DEFAULT_QUERY_NAME_STR
    ).strip() or DEFAULT_QUERY_NAME_STR
    if argument_obj.command_name_str == "bootstrap" and argument_obj.xml:
        changed_bool = import_xml_bool(
            db_path_str=argument_obj.db_path,
            xml_path_str=argument_obj.xml,
            binding_obj_list=binding_obj_list,
            query_name_str=query_name_str,
            replace_bool=argument_obj.replace,
        )
        print("Imported Flex XML." if changed_bool else "Flex XML was already imported.")
        return 0

    latest_completed_date_str = _latest_completed_session_date_str()
    try:
        token_str = _required_secret_env_str(FLEX_TOKEN_ENV_STR)
        query_id_str = _required_secret_env_str(FLEX_QUERY_ID_ENV_STR)
    except Exception as exception_obj:
        _record_preflight_failure_if_daily_sync(argument_obj, exception_obj)
        raise
    if argument_obj.command_name_str == "bootstrap":
        default_start_date_str = _earliest_return_start_date_str(binding_obj_list)
        from_date_str = argument_obj.from_date or default_start_date_str
        to_date_str = argument_obj.to_date or latest_completed_date_str
        bootstrap_remote(
            db_path_str=argument_obj.db_path,
            binding_obj_list=binding_obj_list,
            from_date_str=from_date_str,
            to_date_str=to_date_str,
            query_name_str=query_name_str,
            token_str=token_str,
            query_id_str=query_id_str,
            replace_bool=argument_obj.replace,
        )
        print(f"Bootstrapped performance through {to_date_str}.")
        return 0

    if not Path(argument_obj.db_path).expanduser().exists():
        raise FlexSyncError(
            "Performance database is missing; run bootstrap before the daily sync."
        )
    lookback_date_obj = date.fromisoformat(latest_completed_date_str) - timedelta(
        days=SYNC_LOOKBACK_CALENDAR_DAY_COUNT_INT
    )
    from_date_str = _next_or_same_session_date_str(lookback_date_obj.isoformat())
    changed_bool = sync_range_bool(
        db_path_str=argument_obj.db_path,
        binding_obj_list=binding_obj_list,
        from_date_str=from_date_str,
        to_date_str=latest_completed_date_str,
        query_name_str=query_name_str,
        token_str=token_str,
        query_id_str=query_id_str,
    )
    print(
        f"Synced performance through {latest_completed_date_str}."
        if changed_bool
        else f"Performance through {latest_completed_date_str} was already current."
    )
    return 0


def _eod_boundary_date_tuple(
    release_obj,
    db_path_str: str,
) -> tuple[str | None, str | None]:
    db_path_obj = Path(db_path_str)
    if not db_path_obj.exists():
        return None, None
    history_row_dict_list = LiveStateStore(db_path_str).get_pod_state_history_row_dict_list(
        release_obj.pod_id_str
    )
    eod_market_date_str_list = sorted(
        {
            runner._market_date_str_from_timestamp_str(
                timestamp_str=str(history_row_dict["updated_timestamp_str"]),
                release_obj=release_obj,
            )
            for history_row_dict in history_row_dict_list
            if str(history_row_dict.get("snapshot_stage_str") or "") == "eod"
            and str(history_row_dict.get("snapshot_source_str") or "") == "broker"
        }
    )
    if not eod_market_date_str_list:
        return None, None
    return eod_market_date_str_list[0], eod_market_date_str_list[-1]


def _http_get_text_str(
    base_url_str: str,
    query_dict: dict[str, str],
    operation_label_str: str,
    urlopen_fn: Callable,
) -> str:
    request_url_str = f"{base_url_str}?{urlencode(query_dict)}"
    request_obj = Request(
        request_url_str,
        headers={"User-Agent": FLEX_USER_AGENT_STR},
        method="GET",
    )
    try:
        with urlopen_fn(
            request_obj, timeout=FLEX_HTTP_TIMEOUT_SECONDS_FLOAT
        ) as response_obj:
            return response_obj.read().decode("utf-8")
    except HTTPError as exception_obj:
        raise FlexSyncError(
            f"IBKR Flex {operation_label_str} returned HTTP {exception_obj.code}."
        ) from exception_obj
    except URLError as exception_obj:
        raise FlexSyncError(
            f"IBKR Flex {operation_label_str} network request failed."
        ) from exception_obj
    except TimeoutError as exception_obj:
        raise FlexSyncError(
            f"IBKR Flex {operation_label_str} timed out."
        ) from exception_obj


def _success_reference_code_str(xml_text_str: str) -> str:
    root_obj = _xml_root_obj(xml_text_str)
    if root_obj.tag != "FlexStatementResponse":
        raise FlexSyncError("IBKR Flex SendRequest returned an unexpected XML root.")
    status_str = str(root_obj.findtext("Status") or "").strip()
    if status_str != "Success":
        error_code_str, _ = _flex_error_tuple(root_obj)
        raise FlexSyncError(
            f"IBKR Flex SendRequest failed ({error_code_str or 'unknown'})."
        )
    reference_code_str = str(root_obj.findtext("ReferenceCode") or "").strip()
    if not reference_code_str:
        raise FlexSyncError("IBKR Flex SendRequest omitted ReferenceCode.")
    return reference_code_str


def _xml_root_obj(xml_text_str: str) -> ElementTree.Element:
    try:
        return ElementTree.fromstring(xml_text_str)
    except ElementTree.ParseError as exception_obj:
        raise FlexSyncError("IBKR Flex returned malformed XML.") from exception_obj


def _flex_error_tuple(root_obj: ElementTree.Element) -> tuple[str, str]:
    return (
        str(root_obj.findtext("ErrorCode") or "").strip(),
        str(root_obj.findtext("ErrorMessage") or "").strip(),
    )


def _xml_date_range_tuple(xml_text_str: str) -> tuple[str, str]:
    root_obj = _xml_root_obj(xml_text_str)
    raw_date_str_list = [
        str(change_in_nav_obj.attrib.get("toDate") or "").strip()
        for change_in_nav_obj in root_obj.findall(".//ChangeInNAV")
    ]
    try:
        date_obj_list = [
            datetime.strptime(raw_date_str, "%Y%m%d").date()
            for raw_date_str in raw_date_str_list
        ]
    except ValueError as exception_obj:
        raise PerformanceContractError("Flex contains an invalid Change in NAV date.") from exception_obj
    if not date_obj_list:
        raise PerformanceContractError("Flex contains no Change in NAV rows.")
    return min(date_obj_list).isoformat(), max(date_obj_list).isoformat()


def _latest_completed_session_date_str(
    now_ts: datetime | None = None,
) -> str:
    now_ts = now_ts or datetime.now(UTC)
    if now_ts.tzinfo is None:
        raise ValueError("now_ts must be timezone-aware.")
    calendar_obj = scheduler_utils.get_exchange_calendar_obj("XNYS")
    new_york_now_ts = now_ts.astimezone(ZoneInfo("America/New_York"))
    local_date_ts = pd.Timestamp(new_york_now_ts.date())
    if calendar_obj.is_session(local_date_ts):
        # Activity Flex is a D+1 reporting source. Even after today's close,
        # do not claim today's statement is finalized until the next date.
        return str(calendar_obj.previous_session(local_date_ts).date())
    return str(calendar_obj.date_to_session(local_date_ts, direction="previous").date())


def _next_session_date_str(market_date_str: str) -> str:
    calendar_obj = scheduler_utils.get_exchange_calendar_obj("XNYS")
    return str(calendar_obj.next_session(pd.Timestamp(market_date_str)).date())


def _next_or_same_session_date_str(market_date_str: str) -> str:
    calendar_obj = scheduler_utils.get_exchange_calendar_obj("XNYS")
    market_date_ts = pd.Timestamp(market_date_str)
    if calendar_obj.is_session(market_date_ts):
        return str(market_date_ts.date())
    return str(calendar_obj.date_to_session(market_date_ts, direction="next").date())


def _earliest_return_start_date_str(
    binding_obj_list: list[PodPerformanceBinding],
) -> str:
    start_date_str_list = sorted(
        binding_obj.return_start_date_str
        for binding_obj in binding_obj_list
        if binding_obj.return_start_date_str is not None
    )
    if not start_date_str_list:
        raise PerformanceContractError(
            "No LIVE Pod has a trusted EOD baseline yet; bootstrap cannot start."
        )
    return start_date_str_list[0]


def _required_secret_env_str(env_name_str: str) -> str:
    value_str = os.getenv(env_name_str, "").strip()
    if not value_str:
        raise FlexSyncError(f"{env_name_str} is required in config.env.")
    return value_str


def _record_preflight_failure_if_daily_sync(
    argument_obj: argparse.Namespace, exception_obj: Exception
) -> None:
    if argument_obj.command_name_str != "sync":
        return
    db_path_obj = Path(argument_obj.db_path).expanduser()
    if not db_path_obj.exists():
        return
    to_date_str = _latest_completed_session_date_str()
    lookback_date_obj = date.fromisoformat(to_date_str) - timedelta(
        days=SYNC_LOOKBACK_CALENDAR_DAY_COUNT_INT
    )
    from_date_str = _next_or_same_session_date_str(lookback_date_obj.isoformat())
    PerformanceStore(str(db_path_obj)).record_sync_attempt(
        status_str="failed",
        request_from_date_str=from_date_str,
        request_to_date_str=to_date_str,
        detail_str=_safe_operator_error_str(exception_obj),
    )


def _safe_operator_error_str(exception_obj: Exception) -> str:
    message_str = str(exception_obj) or exception_obj.__class__.__name__
    for env_name_str in (FLEX_TOKEN_ENV_STR, FLEX_QUERY_ID_ENV_STR):
        secret_str = os.getenv(env_name_str, "").strip()
        if secret_str:
            message_str = message_str.replace(secret_str, "[redacted]")
    return message_str


def _date_chunk_tuple_list(
    from_date_str: str,
    to_date_str: str,
) -> list[tuple[str, str]]:
    from_date_obj = date.fromisoformat(from_date_str)
    to_date_obj = date.fromisoformat(to_date_str)
    if from_date_obj > to_date_obj:
        raise PerformanceContractError("Bootstrap date range is reversed.")
    chunk_tuple_list: list[tuple[str, str]] = []
    chunk_start_date_obj = from_date_obj
    while chunk_start_date_obj <= to_date_obj:
        chunk_end_date_obj = min(
            chunk_start_date_obj + timedelta(days=364), to_date_obj
        )
        chunk_tuple_list.append(
            (chunk_start_date_obj.isoformat(), chunk_end_date_obj.isoformat())
        )
        chunk_start_date_obj = chunk_end_date_obj + timedelta(days=1)
    return chunk_tuple_list


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exception_obj:
        # CLI failures must stay token-safe. In particular, urllib exceptions
        # can contain the full request URL, whose query string carries token t.
        print(f"ERROR: {_safe_operator_error_str(exception_obj)}", file=sys.stderr)
        raise SystemExit(2) from None
