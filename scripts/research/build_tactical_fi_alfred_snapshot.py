"""Build the ALFRED point-in-time snapshot for Tactical FI (L14).

For every month-end decision date T from 2014-04-30 through 2026-07-31, and for
the Norgate session before each T, download DGS10, DGS3MO, DAAA and DBAA
exactly as FRED had published them on that vintage date. The vintages are
stored as run tables (see ``alpha/data/alfred_snapshot.py``) with a manifest
holding per-request retrieval times, raw-response hashes and file hashes.

Decision dates come from the strategy's own frozen Norgate IEF/LQD session
calendar, so the snapshot and the backtest can never disagree on which dates
were decisions. The previous-session vintages support the conservative timing
sensitivity.

No API key is used: the keyless ALFRED graph endpoint serves the vintages.
The snapshot is research data. It does not approve PAPER or LIVE use.

Run from the repository root:

    uv run python scripts/research/build_tactical_fi_alfred_snapshot.py
    uv run python scripts/research/build_tactical_fi_alfred_snapshot.py --output-dir <dir>
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from alpha.data.alfred_snapshot import (  # noqa: E402
    ALFRED_GRAPH_CSV_URL_STR,
    ALFRED_SNAPSHOT_SCHEMA_STR,
    AlfredVintageError,
    encode_vintage_run_df,
    fetch_alfred_vintage_dict,
    run_df_to_csv_bytes,
    sha256_bytes_str,
    verify_vintage_run_round_trip,
)
from strategies.taa_beyond_6040 import (  # noqa: E402
    strategy_taa_tactical_fixed_income_ief_lqd as tactical_module,
)


PRE_ARCHIVE_PROBE_DATE_STR = "2014-04-01"


def decision_vintage_date_frame() -> pd.DataFrame:
    """Decision dates T >= the first ALFRED decision, with their prior session."""
    (
        execution_price_df,
        _yield_df,
        signal_df,
        _weight_df,
        _cash_return_ser,
        _snapshot_tuple,
    ) = tactical_module.get_tactical_yield_data(tactical_module.DEFAULT_CONFIG)
    session_index = pd.DatetimeIndex(execution_price_df.index)
    first_decision_ts = pd.Timestamp(tactical_module.FIRST_ALFRED_DECISION_DATE_STR)
    decision_index = pd.DatetimeIndex(signal_df.index[signal_df.index >= first_decision_ts])
    return pd.DataFrame(
        {
            "decision_date": decision_index,
            "previous_session_date": [
                tactical_module.previous_session(decision_date_ts, session_index)
                for decision_date_ts in decision_index
            ],
        }
    )


def probe_pre_archive_vintage_str(series_id_str: str) -> str:
    """Record that ALFRED has no vintage of this series before the archive start."""
    probe_date_ts = pd.Timestamp(PRE_ARCHIVE_PROBE_DATE_STR)
    try:
        fetch_alfred_vintage_dict(series_id_str, [probe_date_ts], pause_seconds_float=0.5)
    except AlfredVintageError as exception_obj:
        return f"no_exact_vintage: {exception_obj}"
    return "exact_vintage_returned"


def git_head_str() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT_PATH,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_snapshot(output_dir_path: Path) -> Path:
    if output_dir_path.exists() and any(output_dir_path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a non-empty snapshot: {output_dir_path}")
    output_dir_path.mkdir(parents=True, exist_ok=True)

    vintage_frame_df = decision_vintage_date_frame()
    vintage_date_list = sorted(
        set(vintage_frame_df["decision_date"]) | set(vintage_frame_df["previous_session_date"])
    )
    started_at_utc_str = datetime.now(tz=UTC).isoformat()
    series_manifest_dict: dict[str, object] = {}
    for series_id_str in tactical_module.FRED_SERIES_ID_TUPLE:
        vintage_value_dict, request_record_list = fetch_alfred_vintage_dict(
            series_id_str,
            vintage_date_list,
        )
        run_df = encode_vintage_run_df(vintage_value_dict)
        verify_vintage_run_round_trip(run_df, vintage_value_dict, series_id_str)
        csv_bytes = run_df_to_csv_bytes(run_df)
        file_name_str = f"alfred_{series_id_str.lower()}_runs.csv"
        (output_dir_path / file_name_str).write_bytes(csv_bytes)
        last_observation_by_vintage_dict = {
            vintage_date_ts.strftime("%Y-%m-%d"): value_ser.index[-1].strftime("%Y-%m-%d")
            for vintage_date_ts, value_ser in sorted(vintage_value_dict.items())
        }
        series_manifest_dict[series_id_str] = {
            "file_str": file_name_str,
            "sha256_str": sha256_bytes_str(csv_bytes),
            "run_row_count_int": int(len(run_df)),
            "vintage_date_list": [date_ts.strftime("%Y-%m-%d") for date_ts in vintage_date_list],
            "last_observation_date_by_vintage_dict": last_observation_by_vintage_dict,
            "request_list": request_record_list,
        }
        print(
            f"{series_id_str}: {len(vintage_value_dict)} vintages, {len(run_df)} runs",
            flush=True,
        )

    pre_archive_probe_dict = {
        series_id_str: probe_pre_archive_vintage_str(series_id_str)
        for series_id_str in sorted(tactical_module.CORPORATE_SERIES_ID_SET)
    }
    manifest_dict = {
        "schema_str": ALFRED_SNAPSHOT_SCHEMA_STR,
        "strategy_str": tactical_module.STRATEGY_NAME_STR,
        "source_str": "ALFRED public graph CSV endpoint (keyless)",
        "source_url_str": ALFRED_GRAPH_CSV_URL_STR,
        "source_query_str": "id=<SERIES>,...&vintage_date=<YYYY-MM-DD>,... (full history per vintage)",
        "vintage_semantics_str": (
            "Each vintage is the series as FRED had published it by the end of the "
            "vintage date (day granularity; the publication time of day is unknown)."
        ),
        "vintage_selection_str": (
            "Every month-end decision date T from the frozen Norgate IEF/LQD session "
            "calendar with T >= FIRST_ALFRED_DECISION_DATE_STR, plus the session before each T."
        ),
        "first_alfred_decision_date_str": tactical_module.FIRST_ALFRED_DECISION_DATE_STR,
        "decision_vintage_list": [
            {
                "decision_date_str": row_obj.decision_date.strftime("%Y-%m-%d"),
                "previous_session_date_str": row_obj.previous_session_date.strftime("%Y-%m-%d"),
            }
            for row_obj in vintage_frame_df.itertuples()
        ],
        "pre_archive_probe_date_str": PRE_ARCHIVE_PROBE_DATE_STR,
        "pre_archive_probe_result_by_series_dict": pre_archive_probe_dict,
        "started_at_utc_str": started_at_utc_str,
        "finished_at_utc_str": datetime.now(tz=UTC).isoformat(),
        "builder_script_str": "scripts/research/build_tactical_fi_alfred_snapshot.py",
        "builder_git_head_str": git_head_str(),
        "series_by_id_dict": series_manifest_dict,
    }
    manifest_bytes = (json.dumps(manifest_dict, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (output_dir_path / "manifest.json").write_bytes(manifest_bytes)
    print(f"manifest sha256 {sha256_bytes_str(manifest_bytes)}", flush=True)
    return output_dir_path


def main() -> int:
    parser_obj = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser_obj.add_argument(
        "--output-dir",
        default=str(tactical_module.DEFAULT_ALFRED_SNAPSHOT_DIR_PATH),
        help="New, empty snapshot directory.",
    )
    args_obj = parser_obj.parse_args()
    build_snapshot(Path(args_obj.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
