from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

import pandas as pd


KENNETH_FRENCH_SOURCE_NAME_STR = "Kenneth French Data Library"
KENNETH_FRENCH_DAILY_MOM_DATASET_NAME_STR = "F-F_Momentum_Factor_daily"
KENNETH_FRENCH_DAILY_MOM_VALUE_COLUMN_STR = "Mom"
DEFAULT_KENNETH_FRENCH_TIMEOUT_INT = 30
SUPPORTED_KENNETH_FRENCH_MODE_TUPLE: tuple[str, ...] = ("backtest",)
KENNETH_FRENCH_MISSING_VALUE_SET = frozenset({-99.99, -999.0})


@dataclass(frozen=True)
class KennethFrenchSeriesSnapshot:
    value_ser: pd.Series
    source_name_str: str
    dataset_name_str: str
    value_column_str: str
    download_attempt_timestamp_ts: datetime
    download_status_str: str
    latest_observation_date_ts: pd.Timestamp
    used_cache_bool: bool


class KennethFrenchSeriesLoadError(RuntimeError):
    def __init__(
        self,
        message_str: str,
        dataset_name_str: str,
        reason_code_str: str,
        series_snapshot_obj: KennethFrenchSeriesSnapshot | None = None,
    ) -> None:
        super().__init__(message_str)
        self.dataset_name_str = str(dataset_name_str)
        self.reason_code_str = str(reason_code_str)
        self.series_snapshot_obj = series_snapshot_obj


class KennethFrenchSeriesUnavailableError(KennethFrenchSeriesLoadError):
    pass


def build_kenneth_french_zip_url(dataset_name_str: str) -> str:
    dataset_name_str = str(dataset_name_str)
    return (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        f"{quote(dataset_name_str)}_CSV.zip"
    )


def _normalize_as_of_timestamp_ts(as_of_ts: datetime | None) -> datetime:
    if as_of_ts is None:
        return datetime.now(tz=UTC)
    if as_of_ts.tzinfo is None:
        return as_of_ts.replace(tzinfo=UTC)
    return as_of_ts.astimezone(UTC)


def _extract_daily_value_ser(
    csv_text_str: str,
    dataset_name_str: str,
    value_column_str: str,
) -> pd.Series:
    data_row_list: list[tuple[str, str]] = []
    header_found_bool = False

    for raw_line_str in csv_text_str.splitlines():
        line_str = str(raw_line_str).replace("\x00", "").strip()
        if not line_str:
            continue

        cell_list = [cell_str.strip() for cell_str in line_str.split(",")]
        if not header_found_bool:
            if len(cell_list) >= 2 and cell_list[0] == "" and cell_list[1] == value_column_str:
                header_found_bool = True
            continue

        if len(cell_list) < 2:
            if len(data_row_list) > 0:
                break
            continue

        date_str = str(cell_list[0])
        value_str = str(cell_list[1])
        if len(date_str) == 8 and date_str.isdigit():
            data_row_list.append((date_str, value_str))
            continue

        if len(data_row_list) > 0:
            break

    if not header_found_bool:
        raise ValueError(
            f"Kenneth French CSV for {dataset_name_str} is missing the '{value_column_str}' header."
        )
    if len(data_row_list) == 0:
        raise ValueError(f"Kenneth French CSV for {dataset_name_str} contains no daily observations.")

    factor_df = pd.DataFrame(data_row_list, columns=["date_str", "value_str"])
    factor_df["date_ts"] = pd.to_datetime(factor_df["date_str"], format="%Y%m%d")
    factor_df["value_float"] = pd.to_numeric(factor_df["value_str"], errors="coerce")
    factor_df.loc[factor_df["value_float"].isin(KENNETH_FRENCH_MISSING_VALUE_SET), "value_float"] = pd.NA
    factor_df = factor_df.dropna(subset=["value_float"])
    if len(factor_df.index) == 0:
        raise ValueError(
            f"Kenneth French CSV for {dataset_name_str} contains no numeric observations after cleaning."
        )

    value_ser = pd.Series(
        factor_df["value_float"].to_numpy(dtype=float) / 100.0,
        index=pd.DatetimeIndex(factor_df["date_ts"]),
        name=str(value_column_str),
        dtype=float,
    ).sort_index()
    value_ser.index.name = "date_ts"
    return value_ser


def _read_csv_text_from_zip_bytes(zip_bytes: bytes, dataset_name_str: str) -> str:
    with ZipFile(BytesIO(zip_bytes)) as zip_file_obj:
        zip_member_name_list = zip_file_obj.namelist()
        if len(zip_member_name_list) == 0:
            raise ValueError(f"Kenneth French zip for {dataset_name_str} is empty.")
        zip_member_name_str = str(zip_member_name_list[0])
        return zip_file_obj.read(zip_member_name_str).decode("utf-8", errors="replace")


def load_daily_kenneth_french_momentum_snapshot(
    cache_zip_path_str: str,
    as_of_ts: datetime | None,
    mode_str: str,
    dataset_name_str: str = KENNETH_FRENCH_DAILY_MOM_DATASET_NAME_STR,
    value_column_str: str = KENNETH_FRENCH_DAILY_MOM_VALUE_COLUMN_STR,
) -> KennethFrenchSeriesSnapshot:
    mode_str = str(mode_str)
    if mode_str not in SUPPORTED_KENNETH_FRENCH_MODE_TUPLE:
        raise ValueError(
            f"Unsupported mode_str '{mode_str}'. Expected one of {SUPPORTED_KENNETH_FRENCH_MODE_TUPLE}."
        )

    normalized_as_of_ts = _normalize_as_of_timestamp_ts(as_of_ts)
    as_of_date_ts = pd.Timestamp(normalized_as_of_ts.date())
    download_attempt_timestamp_ts = datetime.now(tz=UTC)
    cache_zip_path = Path(cache_zip_path_str)

    try:
        zip_url_str = build_kenneth_french_zip_url(dataset_name_str)
        with urlopen(zip_url_str, timeout=DEFAULT_KENNETH_FRENCH_TIMEOUT_INT) as response_obj:
            downloaded_zip_bytes = response_obj.read()
        csv_text_str = _read_csv_text_from_zip_bytes(downloaded_zip_bytes, dataset_name_str)
        cache_zip_path.parent.mkdir(parents=True, exist_ok=True)
        cache_zip_path.write_bytes(downloaded_zip_bytes)
        download_status_str = "download_success"
        used_cache_bool = False
    except (
        HTTPError,
        URLError,
        TimeoutError,
        OSError,
        UnicodeDecodeError,
        ValueError,
        BadZipFile,
    ) as download_exception_obj:
        if not cache_zip_path.exists():
            raise KennethFrenchSeriesUnavailableError(
                message_str=(
                    f"Failed to load Kenneth French dataset '{dataset_name_str}'. "
                    "The refresh attempt failed and no local cache zip exists."
                ),
                dataset_name_str=dataset_name_str,
                reason_code_str=f"{str(dataset_name_str).lower()}_unavailable",
            ) from download_exception_obj

        try:
            csv_text_str = _read_csv_text_from_zip_bytes(cache_zip_path.read_bytes(), dataset_name_str)
        except (OSError, UnicodeDecodeError, ValueError, BadZipFile) as cache_exception_obj:
            raise KennethFrenchSeriesUnavailableError(
                message_str=(
                    f"Failed to load Kenneth French dataset '{dataset_name_str}'. "
                    "The refresh attempt failed and the local cache zip is unreadable."
                ),
                dataset_name_str=dataset_name_str,
                reason_code_str=f"{str(dataset_name_str).lower()}_unavailable",
            ) from cache_exception_obj

        download_status_str = "cache_fallback_after_download_error"
        used_cache_bool = True

    raw_value_ser = _extract_daily_value_ser(
        csv_text_str=csv_text_str,
        dataset_name_str=dataset_name_str,
        value_column_str=value_column_str,
    )
    # *** CRITICAL*** Only observations on or before `as_of_date_ts` may enter
    # the returned series. Allowing later observations would leak future factor
    # history into the backtest path.
    available_value_ser = raw_value_ser[raw_value_ser.index.normalize() <= as_of_date_ts]
    if len(available_value_ser.index) == 0:
        raise KennethFrenchSeriesUnavailableError(
            message_str=(
                f"Kenneth French dataset '{dataset_name_str}' has no observation on or before "
                f"{as_of_date_ts.date().isoformat()}."
            ),
            dataset_name_str=dataset_name_str,
            reason_code_str=f"{str(dataset_name_str).lower()}_unavailable",
        )

    latest_observation_date_ts = pd.Timestamp(available_value_ser.index[-1]).normalize()
    return KennethFrenchSeriesSnapshot(
        value_ser=available_value_ser,
        source_name_str=KENNETH_FRENCH_SOURCE_NAME_STR,
        dataset_name_str=str(dataset_name_str),
        value_column_str=str(value_column_str),
        download_attempt_timestamp_ts=download_attempt_timestamp_ts,
        download_status_str=download_status_str,
        latest_observation_date_ts=latest_observation_date_ts,
        used_cache_bool=used_cache_bool,
    )


__all__ = [
    "KENNETH_FRENCH_DAILY_MOM_DATASET_NAME_STR",
    "KENNETH_FRENCH_DAILY_MOM_VALUE_COLUMN_STR",
    "KENNETH_FRENCH_SOURCE_NAME_STR",
    "KennethFrenchSeriesLoadError",
    "KennethFrenchSeriesSnapshot",
    "KennethFrenchSeriesUnavailableError",
    "build_kenneth_french_zip_url",
    "load_daily_kenneth_french_momentum_snapshot",
]
