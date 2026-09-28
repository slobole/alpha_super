"""Point-in-time FRED data from ALFRED vintages, stored as a compact run table.

ALFRED (the archival FRED) returns a series *as it was published on a given
vintage date*. For a vintage date ``v`` the vintage contains only observations
that FRED had published by the end of day ``v``, with the values known on that
day. Later backfills and revisions are absent.

A snapshot samples a fixed list of vintage dates (for example every month-end
decision date of a strategy). Storing every full vintage would repeat the same
history hundreds of times, so each series is stored as *runs*:

    observation_date, value, first_vintage_date, last_vintage_date

A run means: this observation had this value in every sampled vintage from
``first_vintage_date`` through ``last_vintage_date`` (consecutive in the sampled
list). Reconstruction for a sampled vintage ``v`` is exact:

    value_ser_as_of(v) = {rows with first_vintage_date <= v <= last_vintage_date}

Only sampled vintage dates can be reconstructed. Any other date raises, because
the run table carries no information about unsampled days.

The keyless public endpoint used here is the same one the 2026-09-27 leakage
hunt used. It silently substitutes a different vintage when a requested date
precedes the first archived vintage, so every returned column label is checked
against the requested date and a mismatch fails loud.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
from io import StringIO
import json
from pathlib import Path
import time
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

import pandas as pd


ALFRED_GRAPH_CSV_URL_STR = "https://alfred.stlouisfed.org/graph/alfredgraph.csv"
ALFRED_SNAPSHOT_SCHEMA_STR = "alfred_vintage_runs_v1"
ALFRED_RUN_COLUMN_TUPLE = (
    "observation_date",
    "value",
    "first_vintage_date",
    "last_vintage_date",
)
DEFAULT_ALFRED_TIMEOUT_INT = 120
DEFAULT_ALFRED_BATCH_SIZE_INT = 10
ALFRED_USER_AGENT_STR = "alpha-super-research/1.0 (point-in-time snapshot)"


class AlfredVintageError(RuntimeError):
    """ALFRED did not return the exact vintage that was requested."""


class AlfredRequestError(RuntimeError):
    """ALFRED could not be reached after retries (network or HTTP failure)."""


class AlfredSnapshotIntegrityError(RuntimeError):
    """A stored snapshot does not match its manifest or cannot answer a query."""


def _date_str(date_obj: object) -> str:
    return pd.Timestamp(date_obj).strftime("%Y-%m-%d")


def build_alfred_vintage_csv_url(
    series_id_str: str,
    vintage_date_list: Sequence[pd.Timestamp],
) -> str:
    """One request returns one full-history column per requested vintage date."""
    id_list_str = ",".join([quote_plus(series_id_str)] * len(vintage_date_list))
    vintage_list_str = ",".join(_date_str(date_obj) for date_obj in vintage_date_list)
    return f"{ALFRED_GRAPH_CSV_URL_STR}?id={id_list_str}&vintage_date={vintage_list_str}"


def alfred_vintage_column_str(series_id_str: str, vintage_date_ts: pd.Timestamp) -> str:
    return f"{series_id_str}_{pd.Timestamp(vintage_date_ts):%Y%m%d}"


def parse_alfred_vintage_csv(
    csv_text_str: str,
    series_id_str: str,
    vintage_date_list: Sequence[pd.Timestamp],
) -> dict[pd.Timestamp, pd.Series]:
    """Return {vintage_date: published observations} or raise on any substitution."""
    if not csv_text_str.strip():
        raise AlfredVintageError(f"Empty ALFRED response for {series_id_str}.")
    response_df = pd.read_csv(StringIO(csv_text_str), dtype=str)
    if response_df.columns[0] != "observation_date":
        raise AlfredVintageError(
            f"Unexpected ALFRED header for {series_id_str}: {list(response_df.columns)[:3]}."
        )
    observation_index = pd.DatetimeIndex(pd.to_datetime(response_df["observation_date"]))
    if not observation_index.is_unique:
        raise AlfredVintageError(f"ALFRED response for {series_id_str} repeats observation dates.")
    vintage_value_dict: dict[pd.Timestamp, pd.Series] = {}
    for vintage_date_ts in vintage_date_list:
        vintage_date_ts = pd.Timestamp(vintage_date_ts).normalize()
        column_str = alfred_vintage_column_str(series_id_str, vintage_date_ts)
        # ALFRED answers a date before its first archived vintage with another
        # vintage under a different label. Accept only the exact label.
        if column_str not in response_df.columns:
            raise AlfredVintageError(
                f"ALFRED did not return vintage {column_str}; "
                f"columns were {list(response_df.columns)[1:]}."
            )
        value_ser = pd.to_numeric(
            pd.Series(response_df[column_str].to_numpy(), index=observation_index),
            errors="coerce",
        ).dropna()
        value_ser = value_ser.astype(float).sort_index()
        value_ser.index.name = "observation_date"
        value_ser.name = series_id_str
        if value_ser.empty:
            raise AlfredVintageError(f"ALFRED vintage {column_str} has no numeric values.")
        # *** CRITICAL*** publication boundary: a vintage dated v can only hold
        # observations dated on or before v. Anything later means the column is
        # not the vintage we asked for.
        if pd.Timestamp(value_ser.index[-1]) > vintage_date_ts:
            raise AlfredVintageError(
                f"ALFRED vintage {column_str} contains observation "
                f"{value_ser.index[-1].date()} after its vintage date."
            )
        vintage_value_dict[vintage_date_ts] = value_ser
    return vintage_value_dict


def fetch_alfred_vintage_dict(
    series_id_str: str,
    vintage_date_list: Sequence[pd.Timestamp],
    batch_size_int: int = DEFAULT_ALFRED_BATCH_SIZE_INT,
    pause_seconds_float: float = 1.0,
    timeout_int: int = DEFAULT_ALFRED_TIMEOUT_INT,
) -> tuple[dict[pd.Timestamp, pd.Series], list[dict[str, object]]]:
    """Download full-history vintages in polite batches.

    Returns the vintages and one provenance record per HTTP request (retrieval
    time and SHA-256 of the raw response bytes). A request that still fails
    after retries raises AlfredRequestError; a substituted or invalid vintage
    raises AlfredVintageError. Nothing is silently skipped.
    """
    sorted_vintage_list = sorted({pd.Timestamp(date_obj).normalize() for date_obj in vintage_date_list})
    vintage_value_dict: dict[pd.Timestamp, pd.Series] = {}
    request_record_list: list[dict[str, object]] = []
    for start_int in range(0, len(sorted_vintage_list), batch_size_int):
        batch_vintage_list = sorted_vintage_list[start_int : start_int + batch_size_int]
        url_str = build_alfred_vintage_csv_url(series_id_str, batch_vintage_list)
        request_obj = Request(url_str, headers={"User-Agent": ALFRED_USER_AGENT_STR})
        response_bytes = b""
        last_exception_obj: Exception | None = None
        for attempt_int in range(4):
            try:
                with urlopen(request_obj, timeout=timeout_int) as response_obj:
                    response_bytes = response_obj.read()
                last_exception_obj = None
                break
            except (HTTPError, URLError, TimeoutError, OSError) as exception_obj:
                last_exception_obj = exception_obj
                time.sleep(5.0 * (attempt_int + 1))
        if last_exception_obj is not None:
            raise AlfredRequestError(
                f"ALFRED request failed for {series_id_str} "
                f"{_date_str(batch_vintage_list[0])}..{_date_str(batch_vintage_list[-1])}."
            ) from last_exception_obj
        retrieved_at_utc_str = datetime.now(tz=UTC).isoformat()
        batch_value_dict = parse_alfred_vintage_csv(
            response_bytes.decode("utf-8"),
            series_id_str,
            batch_vintage_list,
        )
        vintage_value_dict.update(batch_value_dict)
        request_record_list.append(
            {
                "first_vintage_date_str": _date_str(batch_vintage_list[0]),
                "last_vintage_date_str": _date_str(batch_vintage_list[-1]),
                "vintage_count_int": len(batch_vintage_list),
                "retrieved_at_utc_str": retrieved_at_utc_str,
                "response_sha256_str": hashlib.sha256(response_bytes).hexdigest(),
                "response_byte_count_int": len(response_bytes),
            }
        )
        time.sleep(pause_seconds_float)
    return vintage_value_dict, request_record_list


def encode_vintage_run_df(
    vintage_value_dict: dict[pd.Timestamp, pd.Series],
) -> pd.DataFrame:
    """Compress sampled vintages into runs of identical published values."""
    vintage_date_list = sorted(vintage_value_dict)
    if not vintage_date_list:
        raise ValueError("At least one vintage is required.")
    open_run_dict: dict[pd.Timestamp, list[object]] = {}
    closed_run_list: list[list[object]] = []
    for vintage_date_ts in vintage_date_list:
        value_by_date_dict = {
            pd.Timestamp(observation_date_obj): float(value_float)
            for observation_date_obj, value_float in vintage_value_dict[vintage_date_ts].items()
        }
        # A run ends when the observation disappears or its value changes.
        for observation_date_ts in list(open_run_dict):
            run_list = open_run_dict[observation_date_ts]
            if value_by_date_dict.get(observation_date_ts) != run_list[1]:
                closed_run_list.append(run_list)
                del open_run_dict[observation_date_ts]
        for observation_date_ts, value_float in value_by_date_dict.items():
            run_list = open_run_dict.get(observation_date_ts)
            if run_list is None:
                open_run_dict[observation_date_ts] = [
                    observation_date_ts,
                    value_float,
                    vintage_date_ts,
                    vintage_date_ts,
                ]
            else:
                run_list[3] = vintage_date_ts
    closed_run_list.extend(open_run_dict.values())
    run_df = pd.DataFrame(closed_run_list, columns=list(ALFRED_RUN_COLUMN_TUPLE))
    run_df = run_df.sort_values(
        ["observation_date", "first_vintage_date"],
        kind="mergesort",
    ).reset_index(drop=True)
    return run_df


def decode_vintage_value_ser(
    run_df: pd.DataFrame,
    vintage_date_ts: pd.Timestamp,
    series_id_str: str,
) -> pd.Series:
    """Rebuild one sampled vintage from its runs (caller checks it was sampled)."""
    vintage_date_ts = pd.Timestamp(vintage_date_ts).normalize()
    # *** CRITICAL*** point-in-time boundary: only values whose run covers the
    # vintage date were published by that date. A run that starts later is a
    # later publication (backfill or revision) and must stay invisible here.
    active_bool_ser = (run_df["first_vintage_date"] <= vintage_date_ts) & (
        run_df["last_vintage_date"] >= vintage_date_ts
    )
    active_df = run_df.loc[active_bool_ser]
    if active_df["observation_date"].duplicated().any():
        raise AlfredSnapshotIntegrityError(
            f"{series_id_str} has overlapping runs at vintage {_date_str(vintage_date_ts)}."
        )
    value_ser = pd.Series(
        active_df["value"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(active_df["observation_date"], name="observation_date"),
        name=series_id_str,
    ).sort_index()
    return value_ser


def verify_vintage_run_round_trip(
    run_df: pd.DataFrame,
    vintage_value_dict: dict[pd.Timestamp, pd.Series],
    series_id_str: str,
) -> None:
    for vintage_date_ts, expected_value_ser in vintage_value_dict.items():
        decoded_value_ser = decode_vintage_value_ser(run_df, vintage_date_ts, series_id_str)
        if not decoded_value_ser.index.equals(expected_value_ser.index) or not (
            decoded_value_ser.to_numpy() == expected_value_ser.to_numpy()
        ).all():
            raise AlfredSnapshotIntegrityError(
                f"Run table for {series_id_str} does not reproduce vintage "
                f"{_date_str(vintage_date_ts)}."
            )


def run_df_to_csv_bytes(run_df: pd.DataFrame) -> bytes:
    csv_df = run_df.loc[:, list(ALFRED_RUN_COLUMN_TUPLE)].copy()
    for column_str in ("observation_date", "first_vintage_date", "last_vintage_date"):
        csv_df[column_str] = pd.to_datetime(csv_df[column_str]).dt.strftime("%Y-%m-%d")
    # Default float formatting is the shortest exact round trip (e.g. 4.06).
    return csv_df.to_csv(index=False, lineterminator="\n").encode("utf-8")


def sha256_bytes_str(content_bytes: bytes) -> str:
    return hashlib.sha256(content_bytes).hexdigest()


def sha256_file_str(file_path: Path) -> str:
    return sha256_bytes_str(Path(file_path).read_bytes())


@dataclass(frozen=True)
class AlfredVintageSnapshot:
    series_id_str: str
    run_df: pd.DataFrame
    vintage_date_index: pd.DatetimeIndex
    source_path_str: str
    sha256_str: str

    def value_ser_as_of(self, vintage_date_ts: pd.Timestamp) -> pd.Series:
        """The series exactly as published on a sampled vintage date."""
        vintage_date_ts = pd.Timestamp(vintage_date_ts).normalize()
        if vintage_date_ts not in self.vintage_date_index:
            raise AlfredSnapshotIntegrityError(
                f"{self.series_id_str} vintage {_date_str(vintage_date_ts)} was not sampled; "
                "the snapshot cannot say what was published on that date."
            )
        return decode_vintage_value_ser(self.run_df, vintage_date_ts, self.series_id_str)


def load_alfred_snapshot_manifest(
    snapshot_dir_path: Path,
    expected_manifest_sha256_str: str | None,
) -> dict[str, object]:
    manifest_path = Path(snapshot_dir_path) / "manifest.json"
    actual_manifest_sha256_str = sha256_file_str(manifest_path)
    if (
        expected_manifest_sha256_str is not None
        and actual_manifest_sha256_str != expected_manifest_sha256_str
    ):
        raise AlfredSnapshotIntegrityError(
            f"ALFRED manifest hash mismatch: expected {expected_manifest_sha256_str}, "
            f"found {actual_manifest_sha256_str}."
        )
    manifest_dict = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest_dict.get("schema_str") != ALFRED_SNAPSHOT_SCHEMA_STR:
        raise AlfredSnapshotIntegrityError(
            f"Unsupported ALFRED snapshot schema: {manifest_dict.get('schema_str')}."
        )
    return manifest_dict


def load_alfred_vintage_snapshot(
    snapshot_dir_path: Path,
    series_id_str: str,
    manifest_dict: dict[str, object],
) -> AlfredVintageSnapshot:
    """Load one series and verify its bytes against the manifest."""
    series_by_id_dict = manifest_dict["series_by_id_dict"]
    if series_id_str not in series_by_id_dict:
        raise AlfredSnapshotIntegrityError(f"{series_id_str} is not in the ALFRED snapshot.")
    series_manifest_dict = series_by_id_dict[series_id_str]
    source_path = Path(snapshot_dir_path) / str(series_manifest_dict["file_str"])
    actual_sha256_str = sha256_file_str(source_path)
    if actual_sha256_str != series_manifest_dict["sha256_str"]:
        raise AlfredSnapshotIntegrityError(
            f"ALFRED run file hash mismatch for {series_id_str}: "
            f"expected {series_manifest_dict['sha256_str']}, found {actual_sha256_str}."
        )
    run_df = pd.read_csv(source_path)
    if tuple(run_df.columns) != ALFRED_RUN_COLUMN_TUPLE:
        raise AlfredSnapshotIntegrityError(
            f"ALFRED run file for {series_id_str} has columns {list(run_df.columns)}."
        )
    for column_str in ("observation_date", "first_vintage_date", "last_vintage_date"):
        run_df[column_str] = pd.to_datetime(run_df[column_str])
    run_df["value"] = run_df["value"].astype(float)
    vintage_date_index = pd.DatetimeIndex(
        pd.to_datetime(series_manifest_dict["vintage_date_list"])
    )
    if len(run_df) != int(series_manifest_dict["run_row_count_int"]):
        raise AlfredSnapshotIntegrityError(f"ALFRED run row count mismatch for {series_id_str}.")
    return AlfredVintageSnapshot(
        series_id_str=series_id_str,
        run_df=run_df,
        vintage_date_index=vintage_date_index,
        source_path_str=str(source_path),
        sha256_str=actual_sha256_str,
    )


__all__ = [
    "ALFRED_GRAPH_CSV_URL_STR",
    "ALFRED_RUN_COLUMN_TUPLE",
    "ALFRED_SNAPSHOT_SCHEMA_STR",
    "AlfredRequestError",
    "AlfredSnapshotIntegrityError",
    "AlfredVintageError",
    "AlfredVintageSnapshot",
    "alfred_vintage_column_str",
    "build_alfred_vintage_csv_url",
    "decode_vintage_value_ser",
    "encode_vintage_run_df",
    "fetch_alfred_vintage_dict",
    "load_alfred_snapshot_manifest",
    "load_alfred_vintage_snapshot",
    "parse_alfred_vintage_csv",
    "run_df_to_csv_bytes",
    "sha256_bytes_str",
    "sha256_file_str",
    "verify_vintage_run_round_trip",
]
