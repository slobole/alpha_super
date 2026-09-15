from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ALPHA_USE_NORGATE_SNAPSHOT_ENV_STR = "ALPHA_USE_NORGATE_SNAPSHOT_BOOL"
NORGATE_SNAPSHOT_ROOT_ENV_STR = "NORGATE_SNAPSHOT_ROOT"
LEGACY_SNAPSHOT_SCHEMA_VERSION_INT = 1
SNAPSHOT_SCHEMA_VERSION_INT = 2
SUPPORTED_SNAPSHOT_SCHEMA_VERSION_SET = {
    LEGACY_SNAPSHOT_SCHEMA_VERSION_INT,
    SNAPSHOT_SCHEMA_VERSION_INT,
}
MANIFEST_FILE_NAME_STR = "manifest.json"
PRICE_FILE_NAME_STR = "prices.parquet"
UNIVERSE_FILE_NAME_STR = "universe.parquet"
CAPITALSPECIAL_ADJUSTMENT_STR = "CAPITALSPECIAL"
TOTALRETURN_ADJUSTMENT_STR = "TOTALRETURN"
CORE5_PROFILE_STR = "norgate_eod_core5"
CORE5_HISTORY_START_DATE_STR = "1990-01-01"
CORE5_CAPITAL_SYMBOL_TUPLE = ("SPY", "IEF", "GLD", "DBC", "UUP", "BIL")
CORE5_TOTAL_RETURN_SYMBOL_TUPLE = ("SPY", "IEF", "GLD", "DBC", "UUP", "$SPX", "$SPXTR")
CORE5_DATA_CONTRACT_DICT = {
    "price_padding_setting_str": "ALLMARKETDAYS",
    "past_member_tail_policy_str": "exact",
    "history_start_date_str": CORE5_HISTORY_START_DATE_STR,
}
HPI_SP500_PROFILE_STR = "norgate_eod_sp500_hpi_pit"
HPI_SP500_DATA_CONTRACT_DICT: dict[str, str] = {
    "price_padding_setting_str": "NONE",
    "past_member_tail_policy_str": "exact",
}

PIT_PROFILE_BY_INDEX_NAME_DICT: dict[str, str] = {
    "S&P 500": "norgate_eod_sp500_pit",
    "Nasdaq 100": "norgate_eod_ndx_pit",
}

HELPER_PROFILE_BY_SYMBOL_DICT: dict[str, str] = {
    "$VIX": "norgate_eod_etf_plus_vix_helper",
    "$VXN": "norgate_eod_ndx_pit_plus_vxn_helper",
}
NDX_PROFILE_SET = {
    "norgate_eod_ndx_pit",
    "norgate_eod_ndx_pit_plus_vxn_helper",
}

_ACTIVE_DATA_PROFILE_CONTEXT_VAR: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "active_norgate_data_profile_str",
    default=None,
)


class NorgateSnapshotError(RuntimeError):
    """Base error for snapshot-mode data failures."""


class NorgateSnapshotNotReadyError(NorgateSnapshotError):
    """Raised when the requested snapshot does not exist yet."""


class NorgateSnapshotValidationError(NorgateSnapshotError):
    """Raised when a snapshot exists but fails validation."""


@dataclass(frozen=True)
class NorgateSnapshotManifest:
    profile_str: str
    snapshot_date_ts: pd.Timestamp
    snapshot_dir_path_obj: Path
    manifest_dict: dict[str, Any]
    manifest_hash_str: str


def is_snapshot_mode_enabled_bool() -> bool:
    raw_value_str = os.getenv(ALPHA_USE_NORGATE_SNAPSHOT_ENV_STR, "false").strip().lower()
    return raw_value_str in {"1", "true", "yes", "y", "on"}


def get_snapshot_root_path_obj() -> Path:
    snapshot_root_str = os.getenv(NORGATE_SNAPSHOT_ROOT_ENV_STR, "").strip()
    if not snapshot_root_str:
        raise NorgateSnapshotNotReadyError(
            f"{NORGATE_SNAPSHOT_ROOT_ENV_STR} must be set when "
            f"{ALPHA_USE_NORGATE_SNAPSHOT_ENV_STR}=true."
        )
    return Path(snapshot_root_str).expanduser()


def normalize_adjustment_str(adjustment_obj: object | None) -> str:
    if adjustment_obj is None:
        return CAPITALSPECIAL_ADJUSTMENT_STR
    adjustment_name_str = str(getattr(adjustment_obj, "name", adjustment_obj)).upper()
    if "TOTALRETURN" in adjustment_name_str:
        return TOTALRETURN_ADJUSTMENT_STR
    if "CAPITALSPECIAL" in adjustment_name_str:
        return CAPITALSPECIAL_ADJUSTMENT_STR
    raise ValueError(f"Unsupported Norgate adjustment setting: {adjustment_obj!r}")


def default_profile_for_indexname_str(indexname_str: str) -> str:
    if indexname_str in PIT_PROFILE_BY_INDEX_NAME_DICT:
        return PIT_PROFILE_BY_INDEX_NAME_DICT[indexname_str]
    raise NorgateSnapshotValidationError(
        f"No default Norgate snapshot profile is configured for index '{indexname_str}'."
    )


def default_profile_for_symbol_str(symbol_str: str) -> str:
    if symbol_str in HELPER_PROFILE_BY_SYMBOL_DICT:
        return HELPER_PROFILE_BY_SYMBOL_DICT[symbol_str]
    return "norgate_eod_etf_plus_vix_helper"


def get_active_data_profile_str(default_profile_str: str | None = None) -> str | None:
    active_profile_str = _ACTIVE_DATA_PROFILE_CONTEXT_VAR.get()
    if active_profile_str:
        return active_profile_str
    return default_profile_str


@contextlib.contextmanager
def use_norgate_data_profile(data_profile_str: str | None) -> Iterator[None]:
    token_obj = _ACTIVE_DATA_PROFILE_CONTEXT_VAR.set(data_profile_str)
    try:
        yield
    finally:
        _ACTIVE_DATA_PROFILE_CONTEXT_VAR.reset(token_obj)


def _hash_file_path(file_path_obj: Path) -> str:
    hash_obj = hashlib.sha256()
    with file_path_obj.open("rb") as file_obj:
        for chunk_bytes in iter(lambda: file_obj.read(1024 * 1024), b""):
            hash_obj.update(chunk_bytes)
    return hash_obj.hexdigest()


def _coerce_snapshot_date_ts(raw_date_obj: object) -> pd.Timestamp:
    snapshot_date_ts = pd.Timestamp(raw_date_obj).normalize()
    if pd.isna(snapshot_date_ts):
        raise NorgateSnapshotValidationError(f"Invalid snapshot date: {raw_date_obj!r}")
    return snapshot_date_ts


def _get_manifest_snapshot_date_ts(manifest_dict: dict[str, Any]) -> pd.Timestamp:
    raw_date_obj = (
        manifest_dict.get("snapshot_market_session_date_str")
        or manifest_dict.get("market_session_date_str")
        or manifest_dict.get("snapshot_date_str")
    )
    if raw_date_obj is None:
        raise NorgateSnapshotValidationError("manifest.json is missing snapshot market-session date.")
    return _coerce_snapshot_date_ts(raw_date_obj)


def _get_file_entry_dict(manifest_dict: dict[str, Any], file_name_str: str) -> dict[str, Any]:
    files_dict = manifest_dict.get("files", {})
    if isinstance(files_dict, dict) and file_name_str in files_dict:
        entry_obj = files_dict[file_name_str]
        if not isinstance(entry_obj, dict):
            raise NorgateSnapshotValidationError(f"manifest files.{file_name_str} must be an object.")
        return entry_obj

    file_hashes_dict = manifest_dict.get("file_hashes", {})
    if isinstance(file_hashes_dict, dict) and file_name_str in file_hashes_dict:
        row_counts_dict = manifest_dict.get("row_counts", {})
        return {
            "sha256": file_hashes_dict[file_name_str],
            "row_count_int": (
                row_counts_dict.get(file_name_str)
                if isinstance(row_counts_dict, dict)
                else None
            ),
        }

    raise NorgateSnapshotValidationError(f"manifest.json is missing file entry for {file_name_str}.")


def _validate_file_hash(snapshot_dir_path_obj: Path, manifest_dict: dict[str, Any], file_name_str: str) -> None:
    file_entry_dict = _get_file_entry_dict(manifest_dict, file_name_str)
    expected_hash_str = str(file_entry_dict.get("sha256", "")).strip().lower()
    if not expected_hash_str:
        raise NorgateSnapshotValidationError(f"manifest entry for {file_name_str} is missing sha256.")

    file_path_obj = snapshot_dir_path_obj / file_name_str
    if not file_path_obj.exists():
        raise NorgateSnapshotNotReadyError(f"Snapshot file is missing: {file_path_obj}")

    actual_hash_str = _hash_file_path(file_path_obj)
    if actual_hash_str != expected_hash_str:
        raise NorgateSnapshotValidationError(
            f"SHA256 mismatch for {file_name_str}: expected {expected_hash_str}, got {actual_hash_str}."
        )


def _validate_price_contract(
    snapshot_dir_path_obj: Path,
    schema_version_int: int,
    profile_str: str,
) -> None:
    if schema_version_int < SNAPSHOT_SCHEMA_VERSION_INT:
        return

    price_path_obj = snapshot_dir_path_obj / PRICE_FILE_NAME_STR
    price_schema_obj = pq.read_schema(price_path_obj)
    price_column_set = {
        str(column_name_obj)
        for column_name_obj in price_schema_obj.names
    }
    required_price_field_set = {"Close", "Dividend"}
    missing_price_field_set = required_price_field_set.difference(price_column_set)
    if len(missing_price_field_set) > 0:
        raise NorgateSnapshotValidationError(
            "prices.parquet is missing current-contract fields: "
            f"{sorted(missing_price_field_set)}"
        )
    dividend_field_obj = price_schema_obj.field("Dividend")
    if not (
        pa.types.is_floating(dividend_field_obj.type)
        or pa.types.is_integer(dividend_field_obj.type)
    ):
        raise NorgateSnapshotValidationError(
            "prices.parquet Dividend must be numeric."
        )
    price_contract_table_obj = pq.read_table(
        price_path_obj,
        columns=["Close", "Dividend"],
    )
    close_ser = price_contract_table_obj.column("Close").to_pandas()
    dividend_ser = price_contract_table_obj.column("Dividend").to_pandas()
    applicable_price_mask_ser = close_ser.notna()
    if dividend_ser.loc[applicable_price_mask_ser].isna().any():
        raise NorgateSnapshotValidationError(
            "prices.parquet Dividend must not contain null values on price rows."
        )
    if profile_str in NDX_PROFILE_SET:
        spy_adjustment_df = pd.read_parquet(
            price_path_obj,
            columns=["symbol_str", "adjustment_str"],
            filters=[("symbol_str", "==", "SPY")],
        )
        actual_spy_adjustment_set = {
            str(adjustment_obj).upper()
            for adjustment_obj in spy_adjustment_df["adjustment_str"].dropna().unique()
        }
        required_spy_adjustment_set = {
            CAPITALSPECIAL_ADJUSTMENT_STR,
            TOTALRETURN_ADJUSTMENT_STR,
        }
        missing_spy_adjustment_set = required_spy_adjustment_set.difference(
            actual_spy_adjustment_set
        )
        if len(missing_spy_adjustment_set) > 0:
            raise NorgateSnapshotValidationError(
                "NDX schema-v2 snapshot is missing SPY adjustment rows: "
                f"{sorted(missing_spy_adjustment_set)}"
            )


def _load_manifest_from_dir(snapshot_dir_path_obj: Path) -> tuple[dict[str, Any], str]:
    manifest_path_obj = snapshot_dir_path_obj / MANIFEST_FILE_NAME_STR
    if not manifest_path_obj.exists():
        raise NorgateSnapshotNotReadyError(f"Snapshot manifest is missing: {manifest_path_obj}")
    manifest_hash_str = _hash_file_path(manifest_path_obj)
    try:
        manifest_dict = json.loads(manifest_path_obj.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise NorgateSnapshotValidationError(f"Invalid manifest JSON: {manifest_path_obj}") from exc
    if not isinstance(manifest_dict, dict):
        raise NorgateSnapshotValidationError("manifest.json must contain a JSON object.")
    return manifest_dict, manifest_hash_str


def core5_price_coverage_dict(
    price_df: pd.DataFrame,
    snapshot_date_ts: pd.Timestamp,
) -> dict[str, dict[str, object]]:
    """Validate CORE5 rows and describe the complete exported source history."""
    required_field_set = {
        "date", "symbol_str", "adjustment_str", "Open", "High", "Low", "Close", "Dividend",
    }
    if not required_field_set.issubset(price_df.columns):
        raise NorgateSnapshotValidationError("CORE5 prices are missing required fields.")
    price_df = price_df.copy()
    price_df["date"] = pd.to_datetime(price_df["date"])
    if (
        price_df["date"].isna().any()
        or not price_df["date"].equals(price_df["date"].dt.normalize())
        or price_df["date"].gt(snapshot_date_ts).any()
        or price_df["date"].lt(pd.Timestamp(CORE5_HISTORY_START_DATE_STR)).any()
        or price_df.duplicated(["date", "symbol_str", "adjustment_str"]).any()
    ):
        raise NorgateSnapshotValidationError("CORE5 prices contain invalid, future or duplicate dates.")
    required_pair_set = {(symbol_str, CAPITALSPECIAL_ADJUSTMENT_STR) for symbol_str in CORE5_CAPITAL_SYMBOL_TUPLE}
    required_pair_set.update((symbol_str, TOTALRETURN_ADJUSTMENT_STR) for symbol_str in CORE5_TOTAL_RETURN_SYMBOL_TUPLE)
    price_group_dict = dict(tuple(price_df.groupby(["symbol_str", "adjustment_str"], sort=False)))
    if set(price_group_dict) != required_pair_set:
        raise NorgateSnapshotValidationError("CORE5 symbol/adjustment pairs do not match the frozen profile.")
    benchmark_date_idx = pd.DatetimeIndex(
        price_group_dict[("$SPXTR", TOTALRETURN_ADJUSTMENT_STR)]["date"]
    ).sort_values()
    coverage_dict: dict[str, dict[str, object]] = {}
    for (symbol_str, adjustment_str), series_price_df in price_group_dict.items():
        series_price_df = series_price_df.sort_values("date")
        observed_price_df = series_price_df.loc[series_price_df["Close"].notna()]
        if observed_price_df.empty:
            raise NorgateSnapshotValidationError(f"CORE5 has no observed prices for {symbol_str}/{adjustment_str}.")
        numeric_price_mat = observed_price_df[["Open", "High", "Low", "Close", "Dividend"]].to_numpy(dtype=float)
        if not np.isfinite(numeric_price_mat).all() or (numeric_price_mat[:, :4] <= 0.0).any():
            raise NorgateSnapshotValidationError(f"CORE5 has unusable prices/dividends for {symbol_str}/{adjustment_str}.")
        first_observed_ts = pd.Timestamp(observed_price_df["date"].iloc[0])
        last_observed_ts = pd.Timestamp(observed_price_df["date"].iloc[-1])
        # *** CRITICAL*** Preserve pre-inception unavailability. After inception,
        # every benchmark session through Close_T must have a usable source row;
        # never repair a gap with a carried price at this validation boundary.
        expected_date_idx = benchmark_date_idx[benchmark_date_idx >= first_observed_ts]
        if (
            last_observed_ts != snapshot_date_ts
            or not pd.DatetimeIndex(observed_price_df["date"]).equals(expected_date_idx)
        ):
            raise NorgateSnapshotValidationError(f"CORE5 has stale or incomplete history for {symbol_str}/{adjustment_str}.")
        coverage_dict[f"{symbol_str}|{adjustment_str}"] = {
            "first_observed_date_str": first_observed_ts.date().isoformat(),
            "last_observed_date_str": last_observed_ts.date().isoformat(),
            "observed_row_count_int": len(observed_price_df),
            "row_count_int": len(series_price_df),
        }
    return coverage_dict


def _validate_core5_snapshot_contract(
    snapshot_dir_path_obj: Path,
    manifest_dict: dict[str, Any],
    snapshot_date_ts: pd.Timestamp,
) -> None:
    if int(manifest_dict["schema_version"]) != SNAPSHOT_SCHEMA_VERSION_INT:
        raise NorgateSnapshotValidationError("CORE5 requires snapshot schema v2 with native dividends.")
    data_contract_dict = manifest_dict.get("data_contract", {})
    if any(data_contract_dict.get(field_str) != value_obj for field_str, value_obj in CORE5_DATA_CONTRACT_DICT.items()):
        raise NorgateSnapshotValidationError("CORE5 padding/history data contract mismatch.")
    required_symbol_set = set(CORE5_CAPITAL_SYMBOL_TUPLE + CORE5_TOTAL_RETURN_SYMBOL_TUPLE)
    declared_adjustment_dict = manifest_dict.get("adjustment_modes", {})
    if set(manifest_dict.get("required_symbols", [])) != required_symbol_set:
        raise NorgateSnapshotValidationError("CORE5 manifest symbols are incomplete.")
    for symbol_str in required_symbol_set:
        expected_adjustment_set = set()
        if symbol_str in CORE5_CAPITAL_SYMBOL_TUPLE:
            expected_adjustment_set.add(CAPITALSPECIAL_ADJUSTMENT_STR)
        if symbol_str in CORE5_TOTAL_RETURN_SYMBOL_TUPLE:
            expected_adjustment_set.add(TOTALRETURN_ADJUSTMENT_STR)
        declared_adjustment_obj = declared_adjustment_dict.get(symbol_str)
        declared_adjustment_list = declared_adjustment_obj if isinstance(declared_adjustment_obj, list) else [declared_adjustment_obj]
        if set(declared_adjustment_list) != expected_adjustment_set:
            raise NorgateSnapshotValidationError(f"CORE5 manifest adjustments are incomplete for {symbol_str}.")
    expected_endpoint_dict = {symbol_str: snapshot_date_ts.date().isoformat() for symbol_str in required_symbol_set}
    if data_contract_dict.get("observed_endpoint_date_by_symbol_dict") != expected_endpoint_dict:
        raise NorgateSnapshotValidationError("CORE5 requires unpadded source observations through the snapshot session.")
    price_df = pd.read_parquet(snapshot_dir_path_obj / PRICE_FILE_NAME_STR)
    actual_coverage_dict = core5_price_coverage_dict(price_df, snapshot_date_ts)
    if data_contract_dict.get("series_coverage_dict") != actual_coverage_dict:
        raise NorgateSnapshotValidationError("CORE5 exported history differs from declared source coverage.")
    source_field_dict = data_contract_dict.get("source_field_by_pair_dict", {})
    source_dtype_dict = data_contract_dict.get("source_dtype_by_pair_dict", {})
    if set(source_field_dict) != set(actual_coverage_dict) or set(source_dtype_dict) != set(actual_coverage_dict):
        raise NorgateSnapshotValidationError("CORE5 native source field provenance is incomplete.")
    for pair_str, source_field_list in source_field_dict.items():
        required_field_set = {"Open", "High", "Low", "Close"}
        if pair_str.split("|")[0] in CORE5_CAPITAL_SYMBOL_TUPLE:
            required_field_set.add("Dividend")
        if (
            not isinstance(source_field_list, list)
            or not required_field_set.issubset(source_field_list)
            or not set(source_field_list).issubset(set(price_df.columns) - {"date", "symbol_str", "adjustment_str"})
            or len(set(source_field_list)) != len(source_field_list)
        ):
            raise NorgateSnapshotValidationError(f"CORE5 native source fields are invalid for {pair_str}.")
        if set(source_dtype_dict[pair_str]) != set(source_field_list):
            raise NorgateSnapshotValidationError(f"CORE5 native source dtypes are incomplete for {pair_str}.")
        symbol_str, adjustment_str = pair_str.split("|")
        source_price_df = price_df.loc[
            price_df["symbol_str"].eq(symbol_str) & price_df["adjustment_str"].eq(adjustment_str),
            source_field_list,
        ]
        for field_str, dtype_str in source_dtype_dict[pair_str].items():
            try:
                dtype_obj = np.dtype(dtype_str)
                if dtype_obj.kind not in "fiu":
                    raise ValueError("Native price fields must be numeric.")
                restored_value_vec = source_price_df[field_str].to_numpy().astype(dtype_obj)
                if not np.array_equal(source_price_df[field_str].to_numpy(), restored_value_vec, equal_nan=True):
                    raise ValueError("Restoring the native dtype would change values.")
            except (TypeError, ValueError, OverflowError) as error_obj:
                raise NorgateSnapshotValidationError(f"CORE5 native dtype is invalid or lossy for {pair_str}/{field_str}.") from error_obj


@lru_cache(maxsize=128)
def _load_valid_snapshot_manifest_cached(
    snapshot_root_str: str,
    profile_str: str,
    snapshot_date_str: str | None,
    minimum_snapshot_date_str: str | None,
) -> NorgateSnapshotManifest:
    if snapshot_date_str is None:
        profile_dir_path_obj = Path(snapshot_root_str) / profile_str
        if not profile_dir_path_obj.exists():
            raise NorgateSnapshotNotReadyError(f"No snapshot profile directory exists: {profile_dir_path_obj}")

        snapshot_dir_path_obj_list: list[Path] = []
        for child_path_obj in profile_dir_path_obj.iterdir():
            if not child_path_obj.is_dir():
                continue
            try:
                _coerce_snapshot_date_ts(child_path_obj.name)
            except NorgateSnapshotValidationError:
                continue
            snapshot_dir_path_obj_list.append(child_path_obj)
        snapshot_dir_path_obj_list = sorted(
            snapshot_dir_path_obj_list,
            key=lambda path_obj: path_obj.name,
            reverse=True,
        )
        if len(snapshot_dir_path_obj_list) == 0:
            raise NorgateSnapshotNotReadyError(f"No snapshots exist for profile {profile_str}.")
        snapshot_dir_path_obj = snapshot_dir_path_obj_list[0]
    else:
        snapshot_dir_path_obj = Path(snapshot_root_str) / profile_str / snapshot_date_str
        if not snapshot_dir_path_obj.exists():
            raise NorgateSnapshotNotReadyError(f"Requested snapshot does not exist: {snapshot_dir_path_obj}")

    manifest_dict, manifest_hash_str = _load_manifest_from_dir(snapshot_dir_path_obj)
    manifest_profile_str = str(manifest_dict.get("profile", ""))
    if manifest_profile_str != profile_str:
        raise NorgateSnapshotValidationError(
            f"Manifest profile mismatch: expected {profile_str}, got {manifest_profile_str}."
        )
    if profile_str == HPI_SP500_PROFILE_STR:
        data_contract_dict = manifest_dict.get("data_contract", {})
        if data_contract_dict != HPI_SP500_DATA_CONTRACT_DICT:
            raise NorgateSnapshotValidationError(
                "Strict HPI snapshot data_contract mismatch: expected "
                f"{HPI_SP500_DATA_CONTRACT_DICT}, got {data_contract_dict!r}."
            )

    schema_version_int = int(manifest_dict.get("schema_version", -1))
    if schema_version_int not in SUPPORTED_SNAPSHOT_SCHEMA_VERSION_SET:
        raise NorgateSnapshotValidationError(
            f"Unsupported Norgate snapshot schema_version {schema_version_int}; "
            f"expected one of {sorted(SUPPORTED_SNAPSHOT_SCHEMA_VERSION_SET)}."
        )

    snapshot_date_ts = _get_manifest_snapshot_date_ts(manifest_dict)
    if snapshot_dir_path_obj.name != snapshot_date_ts.date().isoformat():
        raise NorgateSnapshotValidationError(
            "Snapshot folder date and manifest date differ: "
            f"folder={snapshot_dir_path_obj.name} manifest={snapshot_date_ts.date().isoformat()}."
        )

    if minimum_snapshot_date_str is not None:
        minimum_snapshot_date_ts = _coerce_snapshot_date_ts(minimum_snapshot_date_str)
        if snapshot_date_ts < minimum_snapshot_date_ts:
            raise NorgateSnapshotValidationError(
                "Latest Norgate snapshot is stale: "
                f"profile={profile_str} latest={snapshot_date_ts.date().isoformat()} "
                f"minimum={minimum_snapshot_date_ts.date().isoformat()}."
            )

    _validate_file_hash(snapshot_dir_path_obj, manifest_dict, PRICE_FILE_NAME_STR)
    _validate_price_contract(
        snapshot_dir_path_obj,
        schema_version_int,
        profile_str,
    )
    if profile_str == CORE5_PROFILE_STR:
        _validate_core5_snapshot_contract(snapshot_dir_path_obj, manifest_dict, snapshot_date_ts)
    files_dict = manifest_dict.get("files", {})
    file_hashes_dict = manifest_dict.get("file_hashes", {})
    has_universe_entry_bool = (
        isinstance(files_dict, dict)
        and UNIVERSE_FILE_NAME_STR in files_dict
    ) or (
        isinstance(file_hashes_dict, dict)
        and UNIVERSE_FILE_NAME_STR in file_hashes_dict
    )
    if has_universe_entry_bool:
        _validate_file_hash(snapshot_dir_path_obj, manifest_dict, UNIVERSE_FILE_NAME_STR)

    return NorgateSnapshotManifest(
        profile_str=profile_str,
        snapshot_date_ts=snapshot_date_ts,
        snapshot_dir_path_obj=snapshot_dir_path_obj,
        manifest_dict=manifest_dict,
        manifest_hash_str=manifest_hash_str,
    )


def load_valid_snapshot_manifest(
    profile_str: str,
    *,
    snapshot_date_str: str | None = None,
    minimum_snapshot_date_str: str | None = None,
    snapshot_root_str: str | None = None,
) -> NorgateSnapshotManifest:
    snapshot_root_str = snapshot_root_str or str(get_snapshot_root_path_obj())
    # CORE5 is small. Recheck immutable file bytes on every public validation so
    # replacing/corrupting a previously read artifact cannot reuse a green cache.
    manifest_loader_fn = (
        _load_valid_snapshot_manifest_cached.__wrapped__
        if profile_str == CORE5_PROFILE_STR
        else _load_valid_snapshot_manifest_cached
    )
    return manifest_loader_fn(
        snapshot_root_str,
        profile_str,
        snapshot_date_str,
        minimum_snapshot_date_str,
    )


def clear_snapshot_manifest_cache() -> None:
    _load_valid_snapshot_manifest_cached.cache_clear()


def load_latest_snapshot_session_label_ts(profile_str: str) -> pd.Timestamp | None:
    try:
        snapshot_manifest_obj = load_valid_snapshot_manifest(profile_str)
    except (NorgateSnapshotNotReadyError, NorgateSnapshotValidationError):
        return None
    return snapshot_manifest_obj.snapshot_date_ts


@lru_cache(maxsize=32)
def _read_prices_cached_df(snapshot_dir_str: str, manifest_hash_str: str) -> pd.DataFrame:
    del manifest_hash_str
    price_df = pd.read_parquet(Path(snapshot_dir_str) / PRICE_FILE_NAME_STR)
    required_column_set = {"date", "symbol_str", "adjustment_str"}
    missing_column_set = required_column_set.difference(price_df.columns)
    if len(missing_column_set) > 0:
        raise NorgateSnapshotValidationError(
            f"prices.parquet is missing required columns: {sorted(missing_column_set)}"
        )

    price_df = price_df.copy()
    price_df["date"] = pd.to_datetime(price_df["date"]).dt.normalize()
    price_df["symbol_str"] = price_df["symbol_str"].astype(str)
    price_df["adjustment_str"] = price_df["adjustment_str"].astype(str).str.upper()
    return price_df


def _read_prices_df(snapshot_manifest_obj: NorgateSnapshotManifest) -> pd.DataFrame:
    return _read_prices_cached_df(
        str(snapshot_manifest_obj.snapshot_dir_path_obj),
        snapshot_manifest_obj.manifest_hash_str,
    ).copy()


def _price_field_name_list(price_df: pd.DataFrame) -> list[str]:
    metadata_column_set = {"date", "symbol_str", "adjustment_str"}
    field_name_list = [
        str(column_name_obj)
        for column_name_obj in price_df.columns
        if str(column_name_obj) not in metadata_column_set
    ]
    if len(field_name_list) == 0:
        raise NorgateSnapshotValidationError("prices.parquet contains no price fields.")
    return field_name_list


def load_price_timeseries_df(
    symbol_str: str,
    adjustment_str: str = CAPITALSPECIAL_ADJUSTMENT_STR,
    *,
    start_date_str: str | None = None,
    end_date_str: str | None = None,
    data_profile_str: str | None = None,
) -> pd.DataFrame:
    profile_str = get_active_data_profile_str(data_profile_str or default_profile_for_symbol_str(symbol_str))
    if profile_str is None:
        raise NorgateSnapshotValidationError(f"No data profile was provided for symbol {symbol_str}.")

    normalized_adjustment_str = normalize_adjustment_str(adjustment_str)
    snapshot_manifest_obj = load_valid_snapshot_manifest(profile_str)
    price_df = _read_prices_df(snapshot_manifest_obj)
    field_name_list = _price_field_name_list(price_df)

    symbol_price_df = price_df.loc[
        (price_df["symbol_str"] == str(symbol_str))
        & (price_df["adjustment_str"] == normalized_adjustment_str),
        ["date", *field_name_list],
    ].copy()
    if len(symbol_price_df) == 0:
        raise NorgateSnapshotValidationError(
            "Snapshot is missing required symbol/adjustment data: "
            f"profile={profile_str} symbol={symbol_str} adjustment={normalized_adjustment_str}."
        )

    if start_date_str is not None:
        symbol_price_df = symbol_price_df.loc[
            symbol_price_df["date"] >= pd.Timestamp(start_date_str).normalize()
        ]
    if end_date_str is not None:
        symbol_price_df = symbol_price_df.loc[
            symbol_price_df["date"] <= pd.Timestamp(end_date_str).normalize()
        ]
    if len(symbol_price_df) == 0:
        raise NorgateSnapshotValidationError(
            "Snapshot has no rows for requested date range: "
            f"profile={profile_str} symbol={symbol_str} start={start_date_str} end={end_date_str}."
        )

    symbol_price_df = symbol_price_df.set_index("date").sort_index()
    symbol_price_df.index.name = None
    if profile_str == CORE5_PROFILE_STR:
        symbol_price_df.index.name = snapshot_manifest_obj.manifest_dict["data_contract"].get("source_index_name_str")
        field_name_list = snapshot_manifest_obj.manifest_dict["data_contract"]["source_field_by_pair_dict"][
            f"{symbol_str}|{normalized_adjustment_str}"
        ]
        return symbol_price_df[field_name_list].astype(
            snapshot_manifest_obj.manifest_dict["data_contract"]["source_dtype_by_pair_dict"][f"{symbol_str}|{normalized_adjustment_str}"]
        )
    return symbol_price_df[field_name_list]


def load_raw_prices_df(
    symbols: Sequence[str],
    benchmarks: Sequence[str],
    *,
    start_date_str: str = "1998-01-01",
    end_date_str: str | None = None,
    data_profile_str: str | None = None,
) -> pd.DataFrame:
    benchmark_set = {str(symbol_str) for symbol_str in benchmarks}
    symbol_list = [str(symbol_str) for symbol_str in list(symbols) + list(benchmarks)]
    default_profile_str = (
        default_profile_for_symbol_str(symbol_list[0])
        if symbol_list
        else None
    )
    profile_str = get_active_data_profile_str(
        data_profile_str or default_profile_str
    )
    if profile_str is None:
        raise NorgateSnapshotValidationError(
            "No data profile was provided for raw snapshot prices."
        )
    snapshot_manifest_obj = load_valid_snapshot_manifest(profile_str)
    price_df = _read_prices_df(snapshot_manifest_obj)
    field_name_list = _price_field_name_list(price_df)

    adjustment_by_symbol_dict = {
        symbol_str: (
            TOTALRETURN_ADJUSTMENT_STR
            if symbol_str in benchmark_set
            else CAPITALSPECIAL_ADJUSTMENT_STR
        )
        for symbol_str in symbol_list
    }
    # CORE5 alone uses the same true total-return benchmark as the direct
    # loader. Existing live profiles retain their historical loading contract.
    data_symbol_dict = {
        symbol_str: (
            "$SPXTR" if profile_str == CORE5_PROFILE_STR and symbol_str == "$SPX"
            and symbol_str in benchmark_set else symbol_str
        )
        for symbol_str in symbol_list
    }
    requested_pair_set = {
        (data_symbol_dict[symbol_str], adjustment_by_symbol_dict[symbol_str])
        for symbol_str in symbol_list
    }
    available_pair_set = set(
        price_df[["symbol_str", "adjustment_str"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    start_date_ts = pd.Timestamp(start_date_str).normalize()
    date_mask_ser = price_df["date"] >= start_date_ts
    if end_date_str is not None:
        date_mask_ser &= (
            price_df["date"] <= pd.Timestamp(end_date_str).normalize()
        )
    requested_pair_mask_ser = pd.MultiIndex.from_arrays(
        [
            price_df["symbol_str"],
            price_df["adjustment_str"],
        ]
    ).isin(requested_pair_set)
    selected_price_df = price_df.loc[
        date_mask_ser & requested_pair_mask_ser,
        ["date", "symbol_str", "adjustment_str", *field_name_list],
    ]
    price_group_dict = {
        (str(symbol_str), str(adjustment_str)): symbol_price_df
        for (
            symbol_str,
            adjustment_str,
        ), symbol_price_df in selected_price_df.groupby(
            ["symbol_str", "adjustment_str"],
            sort=False,
        )
    }

    price_frame_list: list[pd.DataFrame] = []
    for symbol_str in symbol_list:
        adjustment_str = adjustment_by_symbol_dict[symbol_str]
        pair_tuple = (data_symbol_dict[symbol_str], adjustment_str)
        symbol_price_df = price_group_dict.get(pair_tuple)
        if symbol_price_df is None:
            if (
                symbol_str not in benchmark_set
                and pair_tuple in available_pair_set
            ):
                continue
            if pair_tuple not in available_pair_set:
                raise NorgateSnapshotValidationError(
                    "Snapshot is missing required symbol/adjustment data: "
                    f"profile={profile_str} symbol={symbol_str} "
                    f"adjustment={adjustment_str}."
                )
            raise NorgateSnapshotValidationError(
                "Snapshot has no rows for requested date range: "
                f"profile={profile_str} symbol={symbol_str} "
                f"start={start_date_str} end={end_date_str}."
            )

        symbol_price_df = (
            symbol_price_df[["date", *field_name_list]]
            .set_index("date")
            .sort_index()
        )
        symbol_price_df.index.name = None
        symbol_field_list = field_name_list
        if profile_str == CORE5_PROFILE_STR:
            symbol_price_df.index.name = snapshot_manifest_obj.manifest_dict["data_contract"].get("source_index_name_str")
            symbol_field_list = snapshot_manifest_obj.manifest_dict["data_contract"]["source_field_by_pair_dict"][
                f"{data_symbol_dict[symbol_str]}|{adjustment_str}"
            ]
        symbol_price_df = symbol_price_df[symbol_field_list]
        if profile_str == CORE5_PROFILE_STR:
            symbol_price_df = symbol_price_df.astype(
                snapshot_manifest_obj.manifest_dict["data_contract"]["source_dtype_by_pair_dict"][f"{data_symbol_dict[symbol_str]}|{adjustment_str}"]
            )
        symbol_price_df.columns = pd.MultiIndex.from_tuples(
            [(symbol_str, field_str) for field_str in symbol_price_df.columns]
        )
        price_frame_list.append(symbol_price_df)

    if len(price_frame_list) == 0:
        raise RuntimeError("No snapshot price data was loaded.")
    return pd.concat(price_frame_list, axis=1).sort_index()


def load_index_constituent_matrix_df(
    indexname_str: str,
    *,
    data_profile_str: str | None = None,
) -> tuple[list[str], pd.DataFrame]:
    profile_str = get_active_data_profile_str(data_profile_str or default_profile_for_indexname_str(indexname_str))
    if profile_str is None:
        raise NorgateSnapshotValidationError(f"No data profile was provided for index {indexname_str}.")

    snapshot_manifest_obj = load_valid_snapshot_manifest(profile_str)
    universe_path_obj = snapshot_manifest_obj.snapshot_dir_path_obj / UNIVERSE_FILE_NAME_STR
    if not universe_path_obj.exists():
        raise NorgateSnapshotValidationError(f"Snapshot profile {profile_str} is missing universe.parquet.")

    universe_df = pd.read_parquet(universe_path_obj)
    if "date" in universe_df.columns:
        universe_df = universe_df.set_index("date")
    universe_df.index = pd.to_datetime(universe_df.index).normalize()
    universe_df = universe_df.sort_index().fillna(0).astype(int)
    universe_df.index.name = None
    symbol_list = [str(symbol_str) for symbol_str in universe_df.columns.tolist()]
    return symbol_list, universe_df


def build_data_source_metadata_dict(data_profile_str: str | None = None) -> dict[str, Any]:
    profile_str = get_active_data_profile_str(data_profile_str)
    if not is_snapshot_mode_enabled_bool():
        return {
            "norgate_data_source_mode_str": "direct",
            "norgate_data_profile_str": profile_str,
        }
    if profile_str is None:
        raise NorgateSnapshotValidationError("Snapshot mode is enabled but no Norgate data profile is active.")
    snapshot_manifest_obj = load_valid_snapshot_manifest(profile_str)
    return {
        "norgate_data_source_mode_str": "snapshot",
        "norgate_data_profile_str": snapshot_manifest_obj.profile_str,
        "norgate_snapshot_date_str": snapshot_manifest_obj.snapshot_date_ts.date().isoformat(),
        "norgate_manifest_hash_str": snapshot_manifest_obj.manifest_hash_str,
        "norgate_snapshot_schema_version_int": int(
            snapshot_manifest_obj.manifest_dict["schema_version"]
        ),
        "norgate_dividend_field_required_bool": (
            int(snapshot_manifest_obj.manifest_dict["schema_version"])
            >= SNAPSHOT_SCHEMA_VERSION_INT
        ),
    }


def _row_count_for_file_path(file_path_obj: Path) -> int:
    data_df = pd.read_parquet(file_path_obj)
    return int(len(data_df.index))


def _build_file_manifest_entry_dict(file_path_obj: Path) -> dict[str, object]:
    return {
        "sha256": _hash_file_path(file_path_obj),
        "row_count_int": _row_count_for_file_path(file_path_obj),
    }


def write_snapshot_files(
    *,
    snapshot_root_str: str,
    profile_str: str,
    snapshot_date_str: str,
    price_df: pd.DataFrame,
    universe_df: pd.DataFrame | None = None,
    required_symbol_list: Iterable[str] | None = None,
    required_helper_symbol_list: Iterable[str] | None = None,
    adjustment_mode_map_dict: dict[str, object] | None = None,
    data_contract_dict: dict[str, object] | None = None,
    generated_timestamp_ts: datetime | None = None,
    overwrite_bool: bool = False,
    schema_version_int: int = LEGACY_SNAPSHOT_SCHEMA_VERSION_INT,
) -> Path:
    if schema_version_int not in SUPPORTED_SNAPSHOT_SCHEMA_VERSION_SET:
        raise NorgateSnapshotValidationError(
            f"Unsupported Norgate snapshot schema_version {schema_version_int}; "
            f"expected one of {sorted(SUPPORTED_SNAPSHOT_SCHEMA_VERSION_SET)}."
        )
    if schema_version_int >= SNAPSHOT_SCHEMA_VERSION_INT:
        if "Dividend" not in price_df.columns:
            raise NorgateSnapshotValidationError(
                "price_df is missing current-contract fields: ['Dividend']"
            )
        if not pd.api.types.is_numeric_dtype(price_df["Dividend"]):
            raise NorgateSnapshotValidationError(
                "price_df Dividend must be numeric."
            )
        applicable_price_mask_ser = (
            price_df["Close"].notna()
            if "Close" in price_df.columns
            else pd.Series(True, index=price_df.index)
        )
        if price_df.loc[applicable_price_mask_ser, "Dividend"].isna().any():
            raise NorgateSnapshotValidationError(
                "price_df Dividend must not contain null values on price rows."
            )

    snapshot_date_ts = _coerce_snapshot_date_ts(snapshot_date_str)
    snapshot_dir_path_obj = (
        Path(snapshot_root_str).expanduser()
        / profile_str
        / snapshot_date_ts.date().isoformat()
    )
    if snapshot_dir_path_obj.exists() and not overwrite_bool:
        raise FileExistsError(f"Snapshot directory already exists: {snapshot_dir_path_obj}")
    snapshot_dir_path_obj.mkdir(parents=True, exist_ok=True)

    required_price_column_set = {"date", "symbol_str", "adjustment_str"}
    missing_price_column_set = required_price_column_set.difference(price_df.columns)
    if len(missing_price_column_set) > 0:
        raise NorgateSnapshotValidationError(
            f"price_df is missing required columns: {sorted(missing_price_column_set)}"
        )

    price_path_obj = snapshot_dir_path_obj / PRICE_FILE_NAME_STR
    price_df.to_parquet(price_path_obj, index=False)

    file_manifest_dict: dict[str, dict[str, object]] = {
        PRICE_FILE_NAME_STR: _build_file_manifest_entry_dict(price_path_obj)
    }
    if universe_df is not None:
        universe_path_obj = snapshot_dir_path_obj / UNIVERSE_FILE_NAME_STR
        universe_df.to_parquet(universe_path_obj)
        file_manifest_dict[UNIVERSE_FILE_NAME_STR] = _build_file_manifest_entry_dict(universe_path_obj)

    generated_ts = generated_timestamp_ts or datetime.now(tz=UTC)
    manifest_dict: dict[str, object] = {
        "profile": profile_str,
        "snapshot_market_session_date_str": snapshot_date_ts.date().isoformat(),
        "generated_timestamp_utc_str": generated_ts.astimezone(UTC).isoformat(),
        "schema_version": int(schema_version_int),
        "files": file_manifest_dict,
        "required_symbols": sorted({str(symbol_str) for symbol_str in (required_symbol_list or [])}),
        "required_helpers": sorted({str(symbol_str) for symbol_str in (required_helper_symbol_list or [])}),
        "adjustment_modes": dict(adjustment_mode_map_dict or {}),
        "data_contract": dict(data_contract_dict or {}),
    }
    manifest_path_obj = snapshot_dir_path_obj / MANIFEST_FILE_NAME_STR
    manifest_path_obj.write_text(
        json.dumps(manifest_dict, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return snapshot_dir_path_obj
