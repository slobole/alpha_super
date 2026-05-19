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

import pandas as pd


ALPHA_USE_NORGATE_SNAPSHOT_ENV_STR = "ALPHA_USE_NORGATE_SNAPSHOT_BOOL"
NORGATE_SNAPSHOT_ROOT_ENV_STR = "NORGATE_SNAPSHOT_ROOT"
SNAPSHOT_SCHEMA_VERSION_INT = 1
MANIFEST_FILE_NAME_STR = "manifest.json"
PRICE_FILE_NAME_STR = "prices.parquet"
UNIVERSE_FILE_NAME_STR = "universe.parquet"
CAPITALSPECIAL_ADJUSTMENT_STR = "CAPITALSPECIAL"
TOTALRETURN_ADJUSTMENT_STR = "TOTALRETURN"

PIT_PROFILE_BY_INDEX_NAME_DICT: dict[str, str] = {
    "S&P 500": "norgate_eod_sp500_pit",
    "Nasdaq 100": "norgate_eod_ndx_pit",
}

HELPER_PROFILE_BY_SYMBOL_DICT: dict[str, str] = {
    "$VIX": "norgate_eod_etf_plus_vix_helper",
    "$VXN": "norgate_eod_ndx_pit_plus_vxn_helper",
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

    schema_version_int = int(manifest_dict.get("schema_version", -1))
    if schema_version_int != SNAPSHOT_SCHEMA_VERSION_INT:
        raise NorgateSnapshotValidationError(
            f"Unsupported Norgate snapshot schema_version {schema_version_int}; "
            f"expected {SNAPSHOT_SCHEMA_VERSION_INT}."
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
) -> NorgateSnapshotManifest:
    snapshot_root_str = str(get_snapshot_root_path_obj())
    return _load_valid_snapshot_manifest_cached(
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
    return symbol_price_df[field_name_list]


def load_raw_prices_df(
    symbols: Sequence[str],
    benchmarks: Sequence[str],
    *,
    start_date_str: str = "1998-01-01",
    end_date_str: str | None = None,
    data_profile_str: str | None = None,
) -> pd.DataFrame:
    price_frame_list: list[pd.DataFrame] = []
    benchmark_set = {str(symbol_str) for symbol_str in benchmarks}
    symbol_list = [str(symbol_str) for symbol_str in list(symbols) + list(benchmarks)]

    for symbol_str in symbol_list:
        adjustment_str = (
            TOTALRETURN_ADJUSTMENT_STR
            if symbol_str in benchmark_set
            else CAPITALSPECIAL_ADJUSTMENT_STR
        )
        symbol_price_df = load_price_timeseries_df(
            symbol_str,
            adjustment_str,
            start_date_str=start_date_str,
            end_date_str=end_date_str,
            data_profile_str=data_profile_str,
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
    generated_timestamp_ts: datetime | None = None,
    overwrite_bool: bool = False,
) -> Path:
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
        "schema_version": SNAPSHOT_SCHEMA_VERSION_INT,
        "files": file_manifest_dict,
        "required_symbols": sorted({str(symbol_str) for symbol_str in (required_symbol_list or [])}),
        "required_helpers": sorted({str(symbol_str) for symbol_str in (required_helper_symbol_list or [])}),
        "adjustment_modes": dict(adjustment_mode_map_dict or {}),
    }
    manifest_path_obj = snapshot_dir_path_obj / MANIFEST_FILE_NAME_STR
    manifest_path_obj.write_text(
        json.dumps(manifest_dict, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return snapshot_dir_path_obj
