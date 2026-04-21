from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from urllib.request import urlopen

import pandas as pd


FRED_SOURCE_NAME_STR = "FRED"
DEFAULT_FRED_TIMEOUT_INT = 30
SUPPORTED_FRED_MODE_TUPLE: tuple[str, ...] = ("backtest", "live")


@dataclass(frozen=True)
class FredSeriesSnapshot:
    value_ser: pd.Series
    source_name_str: str
    series_id_str: str
    download_attempt_timestamp_ts: datetime
    download_status_str: str
    latest_observation_date_ts: pd.Timestamp
    used_cache_bool: bool
    freshness_business_days_int: int


class FredSeriesLoadError(RuntimeError):
    def __init__(
        self,
        message_str: str,
        series_id_str: str,
        reason_code_str: str,
        series_snapshot_obj: FredSeriesSnapshot | None = None,
    ) -> None:
        super().__init__(message_str)
        self.series_id_str = str(series_id_str)
        self.reason_code_str = str(reason_code_str)
        self.series_snapshot_obj = series_snapshot_obj


class FredSeriesUnavailableError(FredSeriesLoadError):
    pass


class FredSeriesStaleError(FredSeriesLoadError):
    pass


def build_fred_csv_url(series_id_str: str) -> str:
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={quote_plus(series_id_str)}"


def _normalize_as_of_timestamp_ts(as_of_ts: datetime | None) -> datetime:
    if as_of_ts is None:
        return datetime.now(tz=UTC)
    if as_of_ts.tzinfo is None:
        return as_of_ts.replace(tzinfo=UTC)
    return as_of_ts.astimezone(UTC)


def _load_csv_df_from_text(csv_text_str: str) -> pd.DataFrame:
    return pd.read_csv(StringIO(csv_text_str))


def _load_csv_df_from_path(cache_csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(cache_csv_path)


def _extract_value_ser(series_df: pd.DataFrame, series_id_str: str) -> pd.Series:
    if len(series_df.columns) < 2:
        raise ValueError(f"FRED CSV for {series_id_str} must contain at least two columns.")

    date_col_str = next(
        (col_str for col_str in series_df.columns if "date" in str(col_str).lower() or str(col_str) == "DATE"),
        None,
    )
    if date_col_str is None:
        raise ValueError(f"FRED CSV for {series_id_str} is missing a date column.")

    value_col_str = next((col_str for col_str in series_df.columns if col_str != date_col_str), None)
    if value_col_str is None:
        raise ValueError(f"FRED CSV for {series_id_str} is missing a value column.")

    value_ser = series_df.set_index(date_col_str)[value_col_str]
    value_ser.index = pd.to_datetime(value_ser.index)
    value_ser = pd.to_numeric(value_ser, errors="coerce").dropna()
    value_ser.index.name = str(date_col_str)
    value_ser.name = str(series_id_str)
    if len(value_ser.index) == 0:
        raise ValueError(f"FRED CSV for {series_id_str} contains no numeric observations.")
    return value_ser.sort_index()


def _compute_freshness_business_days_int(
    latest_observation_date_ts: pd.Timestamp,
    as_of_date_ts: pd.Timestamp,
) -> int:
    latest_observation_date_ts = pd.Timestamp(latest_observation_date_ts).normalize()
    as_of_date_ts = pd.Timestamp(as_of_date_ts).normalize()
    if as_of_date_ts <= latest_observation_date_ts:
        return 0

    business_day_idx = pd.bdate_range(
        start=latest_observation_date_ts + pd.Timedelta(days=1),
        end=as_of_date_ts,
    )
    return int(len(business_day_idx))


def load_daily_fred_series_snapshot(
    series_id_str: str,
    cache_csv_path_str: str,
    as_of_ts: datetime | None,
    mode_str: str,
) -> FredSeriesSnapshot:
    mode_str = str(mode_str)
    if mode_str not in SUPPORTED_FRED_MODE_TUPLE:
        raise ValueError(
            f"Unsupported mode_str '{mode_str}'. Expected one of {SUPPORTED_FRED_MODE_TUPLE}."
        )

    normalized_as_of_ts = _normalize_as_of_timestamp_ts(as_of_ts)
    as_of_date_ts = pd.Timestamp(normalized_as_of_ts.date())
    download_attempt_timestamp_ts = datetime.now(tz=UTC)
    cache_csv_path = Path(cache_csv_path_str)

    try:
        csv_url_str = build_fred_csv_url(series_id_str)
        with urlopen(csv_url_str, timeout=DEFAULT_FRED_TIMEOUT_INT) as response_obj:
            downloaded_csv_text_str = response_obj.read().decode("utf-8")
        series_df = _load_csv_df_from_text(downloaded_csv_text_str)
        cache_csv_path.parent.mkdir(parents=True, exist_ok=True)
        cache_csv_path.write_text(downloaded_csv_text_str, encoding="utf-8")
        download_status_str = "download_success"
        used_cache_bool = False
    except (HTTPError, URLError, TimeoutError, OSError, UnicodeDecodeError, ValueError, pd.errors.ParserError) as download_exception_obj:
        if not cache_csv_path.exists():
            raise FredSeriesUnavailableError(
                message_str=(
                    f"Failed to load FRED series '{series_id_str}'. "
                    "The refresh attempt failed and no local cache file exists."
                ),
                series_id_str=series_id_str,
                reason_code_str=f"{str(series_id_str).lower()}_unavailable",
            ) from download_exception_obj

        try:
            series_df = _load_csv_df_from_path(cache_csv_path)
        except (OSError, ValueError, pd.errors.ParserError) as cache_exception_obj:
            raise FredSeriesUnavailableError(
                message_str=(
                    f"Failed to load FRED series '{series_id_str}'. "
                    "The refresh attempt failed and the local cache file is unreadable."
                ),
                series_id_str=series_id_str,
                reason_code_str=f"{str(series_id_str).lower()}_unavailable",
            ) from cache_exception_obj

        download_status_str = "cache_fallback_after_download_error"
        used_cache_bool = True

    raw_value_ser = _extract_value_ser(series_df, series_id_str)
    # *** CRITICAL*** Only observations published on or before `as_of_date_ts`
    # may enter the returned series. Allowing later observations here would
    # leak future macro information into backtests or live decision builds.
    available_value_ser = raw_value_ser[raw_value_ser.index.normalize() <= as_of_date_ts]
    if len(available_value_ser.index) == 0:
        raise FredSeriesUnavailableError(
            message_str=(
                f"FRED series '{series_id_str}' has no observation on or before "
                f"{as_of_date_ts.date().isoformat()}."
            ),
            series_id_str=series_id_str,
            reason_code_str=f"{str(series_id_str).lower()}_unavailable",
        )

    latest_observation_date_ts = pd.Timestamp(available_value_ser.index[-1]).normalize()
    freshness_business_days_int = _compute_freshness_business_days_int(
        latest_observation_date_ts=latest_observation_date_ts,
        as_of_date_ts=as_of_date_ts,
    )
    series_snapshot_obj = FredSeriesSnapshot(
        value_ser=available_value_ser,
        source_name_str=FRED_SOURCE_NAME_STR,
        series_id_str=str(series_id_str),
        download_attempt_timestamp_ts=download_attempt_timestamp_ts,
        download_status_str=download_status_str,
        latest_observation_date_ts=latest_observation_date_ts,
        used_cache_bool=used_cache_bool,
        freshness_business_days_int=freshness_business_days_int,
    )

    if mode_str == "live" and freshness_business_days_int > 1:
        raise FredSeriesStaleError(
            message_str=(
                f"FRED series '{series_id_str}' is too stale for live use. "
                f"latest_observation_date={latest_observation_date_ts.date().isoformat()} "
                f"freshness_business_days_int={freshness_business_days_int}."
            ),
            series_id_str=series_id_str,
            reason_code_str=f"{str(series_id_str).lower()}_stale",
            series_snapshot_obj=series_snapshot_obj,
        )

    return series_snapshot_obj


__all__ = [
    "FredSeriesLoadError",
    "FredSeriesSnapshot",
    "FredSeriesStaleError",
    "FredSeriesUnavailableError",
    "FRED_SOURCE_NAME_STR",
    "build_fred_csv_url",
    "load_daily_fred_series_snapshot",
]
