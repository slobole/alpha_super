"""Exact-date raw closing prices for display, from local Norgate sources only.

Prefer an exact-date snapshot when configured. If its dated directory is absent,
display valuation can read the installed NDU's loopback service independently of
the trading snapshot schedule. Never download artifacts or change trading mode.
Nothing here changes data or contacts a broker.
"""

from collections import OrderedDict
from copy import deepcopy
from datetime import date, datetime
import math
import json
from pathlib import Path
import re
from threading import Lock
import time
from urllib.parse import quote

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from alpha.live import scheduler_utils
from data import norgate_snapshot_store


SYMBOL_LIMIT_INT = 128
CACHE_LIMIT_INT = 32
CACHE_SECONDS_FLOAT = 300.0
REQUEST_BUDGET_SECONDS_FLOAT = 10.0
RESPONSE_BYTE_LIMIT_INT = 16_384
PRICE_BYTE_LIMIT_INT = 1_000_000_000
PROFILE_SET = {"norgate_eod_core5", "norgate_eod_sp500_pit", "norgate_eod_sp500_hpi_pit",
    "norgate_eod_etf_plus_vix_helper", "norgate_eod_ndx_pit", "norgate_eod_ndx_pit_plus_vxn_helper"}
_cache_dict = OrderedDict()
_cache_lock = Lock()


def _unavailable_dict(reason_str="Closing prices unavailable"):
    return {"available_bool": False, "reason_str": reason_str,
        "price_map_dict": {}, "source_str": "", "close_date_str": ""}


def _positive_float(value_obj):
    if isinstance(value_obj, (bool, str, bytes)):
        raise ValueError("Invalid price")
    value_float = float(value_obj)
    if not math.isfinite(value_float) or value_float <= 0:
        raise ValueError("Invalid price")
    return value_float


def _file_identity_tuple(path_obj, limit_int):
    stat_obj = path_obj.stat()
    if not path_obj.is_file() or not 0 < stat_obj.st_size <= limit_int:
        raise ValueError("Invalid snapshot file")
    return (str(path_obj), stat_obj.st_dev, stat_obj.st_ino, stat_obj.st_size, stat_obj.st_mtime_ns)


def _snapshot_context_tuple(profile_str, close_date_str):
    root_obj = norgate_snapshot_store.get_snapshot_root_path_obj().resolve()
    directory_obj = (root_obj / profile_str / close_date_str).resolve()
    if not directory_obj.is_relative_to(root_obj):
        raise ValueError("Snapshot escaped configured root")
    if not directory_obj.exists():
        return None
    manifest_obj, prices_obj = (directory_obj / "manifest.json").resolve(), (directory_obj / "prices.parquet").resolve()
    if not manifest_obj.is_relative_to(root_obj) or not prices_obj.is_relative_to(root_obj):
        raise ValueError("Snapshot file escaped configured root")
    identity_list = [_file_identity_tuple(manifest_obj, 2_000_000), _file_identity_tuple(prices_obj, PRICE_BYTE_LIMIT_INT)]
    manifest_dict = json.loads(manifest_obj.read_text(encoding="utf-8"))
    if any("universe.parquet" in (manifest_dict.get(field_str) or {}) for field_str in ("files", "file_hashes")):
        universe_obj = (directory_obj / "universe.parquet").resolve()
        if not universe_obj.is_relative_to(root_obj):
            raise ValueError("Snapshot universe escaped configured root")
        identity_list.append(_file_identity_tuple(universe_obj, PRICE_BYTE_LIMIT_INT))
    return root_obj, prices_obj, tuple(identity_list)


def _snapshot_prices_dict(symbol_tuple, profile_str, close_date_str, as_of_ts, root_obj, prices_obj, identity_tuple):
    # A replaced non-CORE5 artifact must not inherit the shared manifest cache's
    # earlier hash validation. Our own cache is keyed by current file identities.
    norgate_snapshot_store.clear_snapshot_manifest_cache()
    manifest_obj = norgate_snapshot_store.load_valid_snapshot_manifest(profile_str,
        snapshot_date_str=close_date_str, snapshot_root_str=str(root_obj))
    generated_ts = datetime.fromisoformat(manifest_obj.manifest_dict["generated_timestamp_utc_str"])
    if generated_ts.tzinfo is None or generated_ts > as_of_ts:
        raise ValueError("Snapshot generated after assessment")
    schema_obj = pq.read_schema(prices_obj)
    required_set = {"date", "symbol_str", "adjustment_str", "Unadjusted Close"}
    if not required_set.issubset(schema_obj.names):
        raise ValueError("Raw close missing")
    date_type_obj = schema_obj.field("date").type
    if pa.types.is_string(date_type_obj) or pa.types.is_large_string(date_type_obj):
        date_filter_obj = close_date_str
    elif pa.types.is_timestamp(date_type_obj):
        if date_type_obj.tz is not None:
            raise ValueError("Ambiguous session date")
        date_filter_obj = datetime.fromisoformat(close_date_str)
    elif pa.types.is_date(date_type_obj):
        date_filter_obj = date.fromisoformat(close_date_str)
    else:
        raise ValueError("Invalid session date")
    column_list = ["date", "symbol_str", "adjustment_str", "Unadjusted Close"]
    if "Volume" in schema_obj.names:
        column_list.append("Volume")
    row_list = pq.read_table(prices_obj, columns=column_list, filters=[("date", "==", date_filter_obj),
        ("symbol_str", "in", list(symbol_tuple)), ("adjustment_str", "==", "CAPITALSPECIAL")]).to_pylist()
    if len(row_list) != len(symbol_tuple):
        raise ValueError("Missing or duplicate closing rows")
    contract_dict = manifest_obj.manifest_dict.get("data_contract") or {}
    endpoint_dict = contract_dict.get("observed_endpoint_date_by_symbol_dict") or {}
    price_map_dict = {}
    for row_dict in row_list:
        symbol_str = row_dict["symbol_str"]
        if symbol_str in price_map_dict:
            raise ValueError("Duplicate closing symbol")
        observed_bool = contract_dict.get("price_padding_setting_str") == "NONE" or endpoint_dict.get(symbol_str) == close_date_str
        if not observed_bool:
            _positive_float(row_dict.get("Volume"))
        # *** CRITICAL *** actual broker shares require the raw historical Close_D.
        # Never substitute CAPITALSPECIAL/TOTALRETURN Close or carry an older row.
        price_map_dict[symbol_str] = _positive_float(row_dict["Unadjusted Close"])
    if set(price_map_dict) != set(symbol_tuple) or _snapshot_context_tuple(profile_str, close_date_str)[2] != identity_tuple:
        raise ValueError("Snapshot changed during acquisition")
    return price_map_dict


def _local_response_tuple(session_obj, endpoint_str, deadline_float, *, parameter_dict=None):
    remaining_float = deadline_float - time.monotonic()
    if remaining_float <= 0:
        raise ValueError("Local price read exceeded its budget")
    # Fixed loopback address; no redirects, proxies, credentials or remote API.
    with session_obj.get("http://127.0.0.1:38889/api/v1/" + endpoint_str, params=parameter_dict,
            timeout=(min(1.0, remaining_float / 2), min(2.0, remaining_float / 2)),
            allow_redirects=False, stream=True) as response_obj:
        if response_obj.status_code != 200:
            raise ValueError("Local Norgate source unavailable")
        body_bytes = b""
        for chunk_bytes in response_obj.iter_content(4096):
            body_bytes += chunk_bytes
            if len(body_bytes) > RESPONSE_BYTE_LIMIT_INT or time.monotonic() > deadline_float:
                raise ValueError("Local response exceeded its bound")
        return response_obj.headers, body_bytes


def _direct_prices_dict(symbol_tuple, close_date_str):
    price_map_dict = {}
    deadline_float = time.monotonic() + REQUEST_BUDGET_SECONDS_FLOAT
    with requests.Session() as session_obj:
        session_obj.trust_env = False
        for symbol_str in symbol_tuple:
            encoded_str = quote(symbol_str, safe="")

            def metadata_str(field_str):
                return _local_response_tuple(session_obj, "security/" + encoded_str + "/" + field_str, deadline_float)[1].decode("utf-8").strip()

            if metadata_str("currency") != "USD" or metadata_str("basetype") != "Stock Market":
                raise ValueError("Unsupported instrument")
            subtype_str = metadata_str("subtype1")
            if subtype_str != "Equity" and not (subtype_str == "Exchange Traded Product"
                    and metadata_str("subtype2") == "Exchange Traded Fund (ETF)"):
                raise ValueError("Unsupported instrument")
            header_dict, body_bytes = _local_response_tuple(session_obj, "prices/" + encoded_str, deadline_float,
                parameter_dict={"start_date": close_date_str, "end_date": close_date_str,
                    "stock_price_adjustment_setting": "NONE", "padding_setting": "NONE",
                    "format": "numpy-ndarray", "interval": "D", "fields": "Close"})
            format_list = header_dict.get("X-Norgate-Data-Field-Formats", "").split(",")
            if (header_dict.get("X-Norgate-Data-Record-Count") != "1"
                    or header_dict.get("X-Norgate-Data-Field-Names") != "Date,Close"
                    or header_dict.get("X-Norgate-Data-Field-Count") != "2" or len(format_list) != 2
                    or format_list[0] != "<M8[D]" or format_list[1] not in {"f4", "f8", "<f4", "<f8"}):
                raise ValueError("Unexpected local price response")
            # Same structured-array protocol used by installed norgatedata's
            # create_numpy_ndarray, narrowed to one Date/Close record only.
            dtype_obj = np.dtype([("Date", format_list[0]), ("Close", format_list[1])])
            if len(body_bytes) != dtype_obj.itemsize:
                raise ValueError("Invalid local price record")
            row_obj = np.frombuffer(body_bytes, dtype=dtype_obj, count=1)[0]
            if str(row_obj["Date"]) != close_date_str:
                raise ValueError("Wrong close date")
            price_map_dict[symbol_str] = _positive_float(row_obj["Close"])
    return price_map_dict


def load_close_prices_dict(symbol_list, *, profile_str, close_date_str, as_of_ts):
    """Read complete USD equity/ETF raw closes for one already-closed session.

    The six existing snapshot profiles are US equity/ETF profiles; helper symbols
    are excluded. Direct NDU additionally verifies currency and instrument type.
    Cache contents are bounded and copied; source errors never return partial
    prices. A missing dated snapshot may use same-host NDU for display only;
    existing invalid snapshots fail closed. Trading mode remains unchanged.
    Account identity and EOD quantities belong to caller.
    """
    result_dict = _unavailable_dict()
    reason_str = "Closing prices unavailable"
    try:
        if (profile_str not in PROFILE_SET or not isinstance(symbol_list, (list, tuple))
                or len(symbol_list) > SYMBOL_LIMIT_INT or len(set(symbol_list)) != len(symbol_list)
                or any(not isinstance(symbol_str, str) or not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,39}", symbol_str) for symbol_str in symbol_list)):
            return result_dict
        close_obj = date.fromisoformat(close_date_str)
        if close_obj.isoformat() != close_date_str or as_of_ts.tzinfo is None:
            return result_dict
        if as_of_ts < scheduler_utils.get_session_close_timestamp_ts(pd.Timestamp(close_obj), "XNYS"):
            return result_dict
        symbol_tuple = tuple(sorted(symbol_list))
        if not symbol_tuple:
            return {"available_bool": True, "reason_str": "", "price_map_dict": {}, "source_str": "No holdings", "close_date_str": close_date_str}
        with _cache_lock:
            snapshot_bool = norgate_snapshot_store.is_snapshot_mode_enabled_bool()
            reason_str = "Closing price snapshot could not be read" if snapshot_bool else "Local closing prices unavailable"
            context_tuple = _snapshot_context_tuple(profile_str, close_date_str) if snapshot_bool else None
            if context_tuple is None:
                reason_str = "No saved closing prices; local prices unavailable" if snapshot_bool else reason_str
            cache_key_tuple = (snapshot_bool, profile_str, close_date_str, symbol_tuple, context_tuple[2] if context_tuple else None)
            cached_tuple = _cache_dict.get(cache_key_tuple)
            if cached_tuple and time.monotonic() - cached_tuple[0] < CACHE_SECONDS_FLOAT and as_of_ts >= cached_tuple[2]:
                _cache_dict.move_to_end(cache_key_tuple)
                return deepcopy(cached_tuple[1])
            price_map_dict = (_snapshot_prices_dict(symbol_tuple, profile_str, close_date_str, as_of_ts, *context_tuple)
                if context_tuple is not None else _direct_prices_dict(symbol_tuple, close_date_str))
            result_dict = {"available_bool": True, "reason_str": "", "price_map_dict": price_map_dict,
                "source_str": "Norgate snapshot" if context_tuple is not None else "Norgate local", "close_date_str": close_date_str}
            _cache_dict[cache_key_tuple] = (time.monotonic(), deepcopy(result_dict), as_of_ts)
            _cache_dict.move_to_end(cache_key_tuple)
            while len(_cache_dict) > CACHE_LIMIT_INT:
                _cache_dict.popitem(last=False)
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError, OSError, requests.RequestException,
            norgate_snapshot_store.NorgateSnapshotError, pa.ArrowException):
        return _unavailable_dict(reason_str)
    return result_dict
