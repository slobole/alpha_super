"""Sequential direct/snapshot input qualification; never touches an active deployment.

Example:
    python scripts/research/mr_capsule_build_20261004/qualify_wiring.py \
        --output-dir results/research/mr_capsule_wiring_20261005 --end-date 2026-10-02

The output directory must be empty and strictly below this checkout's
results/research. Every Norgate load/export runs sequentially. Saved compressed
pickles preserve the direct frames and attrs for subsequent host qualification.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_PATH_OBJ = Path(__file__).resolve().parents[3]
if str(REPO_PATH_OBJ) not in sys.path:
    sys.path.insert(0, str(REPO_PATH_OBJ))

DECISION_FIELD_SET = {"Open", "High", "Low", "Close", "Volume", "Turnover", "Dividend"}
SOURCE_PATH_TUPLE = (
    "scripts/research/mr_capsule_build_20261004/qualify_wiring.py",
    "scripts/export_norgate_snapshot.py", "data/norgate_loader.py", "data/norgate_snapshot_store.py",
    "strategies/dv2/strategy_mr_dv2.py", "strategies/hpi/stateful_long.py",
    "strategies/mr_capsule/dv2_vix_gated.py", "strategies/mr_capsule/hpi_vote_vix_gated.py",
    "strategies/mr_capsule/capsule_pod.py", "strategies/mr_capsule/parking.py",
    "strategies/mr_capsule/vix_stress_gate.py", "alpha/live/mr_capsule_adapter.py",
    "alpha/live/strategy_host.py", "alpha/live/execution_engine.py", "alpha/live/models.py",
)


def _sha256_str(path_obj: Path) -> str:
    hash_obj = hashlib.sha256()
    with path_obj.open("rb") as file_obj:
        for chunk_bytes in iter(lambda: file_obj.read(1024 * 1024), b""):
            hash_obj.update(chunk_bytes)
    return hash_obj.hexdigest()


def _source_hash_dict() -> dict[str, str]:
    return {path_str: _sha256_str(REPO_PATH_OBJ / path_str) for path_str in SOURCE_PATH_TUPLE}


def validate_output_path(output_path_obj: Path, research_root_path_obj: Path) -> Path:
    output_path_obj = output_path_obj.expanduser().resolve()
    research_root_path_obj = research_root_path_obj.resolve()
    if output_path_obj == research_root_path_obj or not output_path_obj.is_relative_to(research_root_path_obj):
        raise ValueError("--output-dir must be strictly below this checkout's results/research directory.")
    if output_path_obj.exists() and (not output_path_obj.is_dir() or any(output_path_obj.iterdir())):
        raise ValueError("--output-dir must be new or empty; existing evidence will not be overwritten.")
    return output_path_obj


@contextmanager
def _norgate_mode(snapshot_root_path_obj: Path | None = None):
    # Process-local only: restore both environment values even after a failure.
    value_dict = {
        "ALPHA_USE_NORGATE_SNAPSHOT_BOOL": "false" if snapshot_root_path_obj is None else "true",
        "NORGATE_SNAPSHOT_ROOT": None if snapshot_root_path_obj is None else str(snapshot_root_path_obj),
    }
    prior_value_dict = {name_str: os.environ.get(name_str) for name_str in value_dict}
    try:
        for name_str, value_str in value_dict.items():
            if value_str is None:
                os.environ.pop(name_str, None)
            else:
                os.environ[name_str] = value_str
        yield
    finally:
        for name_str, value_str in prior_value_dict.items():
            if value_str is None:
                os.environ.pop(name_str, None)
            else:
                os.environ[name_str] = value_str


def _label_str(label_obj) -> str:
    return "|".join(map(str, label_obj)) if isinstance(label_obj, tuple) else str(label_obj)


def compare_frames(direct_df: pd.DataFrame, snapshot_df: pd.DataFrame, *, allow_empty_schema_bool: bool = False, allow_nonmember_columns_bool: bool = False, require_observed_dtype_parity_bool: bool = False) -> dict:
    """Compare numeric values exactly; dtype and column order alone are not differences.

    A one-sided all-NaN field may be a shared-parquet schema expansion. Record
    every such field explicitly. Entirely zero one-sided PIT columns may also
    be recorded as nonmembers for the declared comparison window. Different dates, null masks, infinities or
    nonempty one-sided columns are failures; no tolerance conceals value drift.
    Decision inputs additionally require native dtype parity: promoting values
    before later arithmetic can change a threshold despite exact stored values.
    """
    result_dict = {
        "passed_bool": True, "direct_shape_list": list(direct_df.shape), "snapshot_shape_list": list(snapshot_df.shape),
        "representation_rule_str": "Numeric equality after float64 conversion; absolute and relative tolerance both zero. Null masks must match.",
        "column_order_equal_bool": direct_df.columns.equals(snapshot_df.columns),
        "ignored_all_nan_schema_column_list": [], "dtype_difference_list": [], "column_error_list": [],
        "ignored_zero_membership_column_list": [],
        "compared_column_count_int": 0, "compared_cell_count_int": 0, "mismatched_cell_count_int": 0,
        "max_absolute_difference_float": 0.0,
    }
    for side_str, frame_df in (("direct", direct_df), ("snapshot", snapshot_df)):
        if frame_df.empty:
            result_dict["passed_bool"] = False
            result_dict["column_error_list"].append({"error_str": f"{side_str} frame is empty"})
        if frame_df.index.has_duplicates or frame_df.columns.has_duplicates or not frame_df.index.is_monotonic_increasing:
            result_dict["passed_bool"] = False
            result_dict["column_error_list"].append({"error_str": f"{side_str} has duplicate labels or unordered dates"})
    if not direct_df.index.equals(snapshot_df.index):
        result_dict["passed_bool"] = False
        result_dict["date_difference_dict"] = {
            "direct_only_list": [str(value_obj) for value_obj in direct_df.index.difference(snapshot_df.index)],
            "snapshot_only_list": [str(value_obj) for value_obj in snapshot_df.index.difference(direct_df.index)],
        }
    if not result_dict["passed_bool"]:
        return result_dict
    direct_column_set, snapshot_column_set = set(direct_df.columns), set(snapshot_df.columns)
    for side_str, column_set, frame_df in (
        ("direct", direct_column_set - snapshot_column_set, direct_df),
        ("snapshot", snapshot_column_set - direct_column_set, snapshot_df),
    ):
        for column_obj in sorted(column_set, key=_label_str):
            column_dict = {"side_str": side_str, "column_str": _label_str(column_obj)}
            if allow_empty_schema_bool and frame_df[column_obj].isna().all():
                result_dict["ignored_all_nan_schema_column_list"].append(column_dict)
            elif allow_nonmember_columns_bool and frame_df[column_obj].eq(0).all():
                result_dict["ignored_zero_membership_column_list"].append(column_dict)
            else:
                result_dict["passed_bool"] = False
                result_dict["column_error_list"].append({**column_dict, "error_str": "nonmatching column"})
    for column_obj in sorted(direct_column_set & snapshot_column_set, key=_label_str):
        direct_ser, snapshot_ser = direct_df[column_obj], snapshot_df[column_obj]
        if direct_ser.dtype != snapshot_ser.dtype:
            result_dict["dtype_difference_list"].append({"column_str": _label_str(column_obj), "direct_dtype_str": str(direct_ser.dtype), "snapshot_dtype_str": str(snapshot_ser.dtype)})
            if require_observed_dtype_parity_bool and (direct_ser.notna().any() or snapshot_ser.notna().any()):
                result_dict["passed_bool"] = False
                result_dict["column_error_list"].append({"column_str": _label_str(column_obj), "error_str": "observed decision-input dtype differs"})
        try:
            direct_vec = pd.to_numeric(direct_ser, errors="raise").to_numpy(dtype=np.float64, na_value=np.nan)
            snapshot_vec = pd.to_numeric(snapshot_ser, errors="raise").to_numpy(dtype=np.float64, na_value=np.nan)
        except (ValueError, TypeError) as exception_obj:
            result_dict["passed_bool"] = False
            result_dict["column_error_list"].append({"column_str": _label_str(column_obj), "error_str": str(exception_obj)})
            continue
        # Prices/turnover/volume are numeric; reject integer precision loss rather
        # than treating two distinct integers above float64's exact range as equal.
        if any(pd.api.types.is_integer_dtype(value_ser.dtype) and value_ser.abs().gt(2**53).any() for value_ser in (direct_ser, snapshot_ser)):
            result_dict["passed_bool"] = False
            result_dict["column_error_list"].append({"column_str": _label_str(column_obj), "error_str": "integer exceeds exact float64 comparison range"})
            continue
        invalid_vec = np.isinf(direct_vec) | np.isinf(snapshot_vec)
        mismatch_vec = ~((direct_vec == snapshot_vec) | (np.isnan(direct_vec) & np.isnan(snapshot_vec))) | invalid_vec
        finite_vec = np.isfinite(direct_vec) & np.isfinite(snapshot_vec)
        max_diff_float = float(np.max(np.abs(direct_vec[finite_vec] - snapshot_vec[finite_vec]))) if finite_vec.any() else 0.0
        mismatch_count_int = int(mismatch_vec.sum())
        result_dict["compared_column_count_int"] += 1
        result_dict["compared_cell_count_int"] += len(direct_vec)
        result_dict["mismatched_cell_count_int"] += mismatch_count_int
        result_dict["max_absolute_difference_float"] = max(result_dict["max_absolute_difference_float"], max_diff_float)
        if mismatch_count_int:
            first_row_int = int(np.flatnonzero(mismatch_vec)[0])
            result_dict["passed_bool"] = False
            result_dict["column_error_list"].append({
                "column_str": _label_str(column_obj), "mismatched_cell_count_int": mismatch_count_int,
                "max_absolute_difference_float": max_diff_float, "first_date_str": str(direct_df.index[first_row_int]),
                "first_direct_value_str": str(direct_vec[first_row_int]), "first_snapshot_value_str": str(snapshot_vec[first_row_int]),
            })
    return result_dict


def _load_pod_inputs(pod_str: str, end_date_str: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pod_str == "dv2":
        from strategies.mr_capsule.dv2_vix_gated import load_pricing_data
        pricing_df, universe_df = load_pricing_data(end_date_str)
    else:
        from strategies.hpi.stateful_long import load_exact_hpi_inputs
        from strategies.mr_capsule.hpi_vote_vix_gated import append_parking_prices
        _, universe_df, pricing_df = load_exact_hpi_inputs("S&P 500", "$SPXTR", "1998-01-01", end_date_str)
        pricing_df = append_parking_prices(pricing_df, "1998-01-01", end_date_str)
        pricing_df.attrs["norgate_adjustment_by_symbol_dict"] = {
            str(symbol_str): "TOTALRETURN" if symbol_str == "$SPXTR" else "CAPITALSPECIAL"
            for symbol_str in pricing_df.columns.get_level_values(0).unique()
        }
    # Membership outside the declared loading window cannot affect this study.
    # Within it, preserve every row/column/value; mismatches must remain visible.
    return pricing_df, universe_df.loc["1998-01-01":end_date_str].copy()


def _save_frame(output_path_obj: Path, name_str: str, frame_obj, report_dict: dict) -> None:
    frame_path_obj = output_path_obj / f"{name_str}.pkl.gz"
    frame_obj.to_pickle(frame_path_obj, compression="gzip")
    report_dict["artifact_dict"][name_str] = {"path_str": str(frame_path_obj), "sha256_str": _sha256_str(frame_path_obj), "shape_list": list(frame_obj.shape)}


def _pricing_fields(pricing_df: pd.DataFrame) -> pd.DataFrame:
    return pricing_df.loc[:, pricing_df.columns.get_level_values(1).isin(DECISION_FIELD_SET)]


def run_qualification(output_path_obj: Path, end_date_str: str) -> dict:
    from data.norgate_loader import load_price_timeseries, use_norgate_data_profile
    from data.norgate_snapshot_store import MR_CAPSULE_DV2_PROFILE_STR, MR_CAPSULE_HPI_PROFILE_STR, load_valid_snapshot_manifest
    from scripts.export_norgate_snapshot import export_profile_snapshot
    from strategies.mr_capsule.vix_stress_gate import load_vix_close_ser

    output_path_obj = validate_output_path(output_path_obj, REPO_PATH_OBJ / "results" / "research")
    if date.fromisoformat(end_date_str).isoformat() != end_date_str:
        raise ValueError("--end-date must be YYYY-MM-DD.")
    output_path_obj.mkdir(parents=True, exist_ok=True)
    snapshot_root_path_obj = output_path_obj / "snapshots"
    report_dict = {
        "status_str": "running", "end_date_str": end_date_str, "start_date_str": "1998-01-01",
        "export_history_start_date_str": "1998-01-01",
        "empty_pre_history_fallback_start_date_str": "1990-01-01",
        "started_at_str": datetime.now(timezone.utc).isoformat(), "source_sha256_by_path_dict": _source_hash_dict(),
        "artifact_dict": {}, "snapshot_manifest_dict": {}, "comparison_dict": {},
        "sequential_loads_bool": True, "active_config_changed_bool": False,
    }
    report_path_obj = output_path_obj / "qualification.json"
    report_path_obj.write_text(json.dumps(report_dict, indent=2, allow_nan=False), encoding="utf-8")
    try:
        with _norgate_mode(), use_norgate_data_profile(None):
            for pod_str in ("dv2", "hpi"):
                print(f"Loading direct {pod_str} inputs", flush=True)
                pricing_df, universe_df = _load_pod_inputs(pod_str, end_date_str)
                _save_frame(output_path_obj, f"direct_{pod_str}_pricing", pricing_df, report_dict)
                _save_frame(output_path_obj, f"direct_{pod_str}_universe", universe_df, report_dict)
                report_path_obj.write_text(json.dumps(report_dict, indent=2, allow_nan=False), encoding="utf-8")
                del pricing_df, universe_df
            print("Loading direct VIX and true SPX total-return benchmark", flush=True)
            _save_frame(output_path_obj, "direct_vix", load_vix_close_ser(end_date_str), report_dict)
            _save_frame(output_path_obj, "direct_spxtr", load_price_timeseries("$SPXTR", adjustment_str="TOTALRETURN", start_date_str="1998-01-01", end_date_str=end_date_str), report_dict)
            for profile_str in (MR_CAPSULE_DV2_PROFILE_STR, MR_CAPSULE_HPI_PROFILE_STR):
                print(f"Exporting isolated {profile_str}", flush=True)
                export_profile_snapshot(snapshot_root_str=str(snapshot_root_path_obj), profile_str=profile_str,
                    snapshot_date_str=end_date_str, start_date_str="1998-01-01", end_date_str=end_date_str)
                manifest_obj = load_valid_snapshot_manifest(profile_str, snapshot_date_str=end_date_str, snapshot_root_str=str(snapshot_root_path_obj))
                manifest_path_obj = manifest_obj.snapshot_dir_path_obj / "manifest.json"
                report_dict["snapshot_manifest_dict"][profile_str] = {
                    "manifest_hash_str": manifest_obj.manifest_hash_str, "file_sha256_str": _sha256_str(manifest_path_obj),
                    "path_str": str(manifest_path_obj),
                }
                report_path_obj.write_text(json.dumps(report_dict, indent=2, allow_nan=False), encoding="utf-8")
        with _norgate_mode(snapshot_root_path_obj):
            for pod_str, profile_str in (("dv2", MR_CAPSULE_DV2_PROFILE_STR), ("hpi", MR_CAPSULE_HPI_PROFILE_STR)):
                with use_norgate_data_profile(profile_str):
                    print(f"Comparing snapshot {pod_str} inputs", flush=True)
                    pricing_df, universe_df = _load_pod_inputs(pod_str, end_date_str)
                    direct_pricing_df = pd.read_pickle(output_path_obj / f"direct_{pod_str}_pricing.pkl.gz")
                    direct_universe_df = pd.read_pickle(output_path_obj / f"direct_{pod_str}_universe.pkl.gz")
                    report_dict["comparison_dict"][f"{pod_str}_pricing"] = compare_frames(_pricing_fields(direct_pricing_df), _pricing_fields(pricing_df), allow_empty_schema_bool=True, require_observed_dtype_parity_bool=True)
                    report_dict["comparison_dict"][f"{pod_str}_universe"] = compare_frames(direct_universe_df, universe_df, allow_nonmember_columns_bool=True)
                    benchmark_label_str = "$SPX" if pod_str == "dv2" else "$SPXTR"
                    benchmark_df = pd.read_pickle(output_path_obj / "direct_spxtr.pkl.gz")[["Close"]]
                    for mode_str, frame_df in (("direct", direct_pricing_df), ("snapshot", pricing_df)):
                        report_dict["comparison_dict"][f"{pod_str}_{mode_str}_true_benchmark"] = compare_frames(
                            benchmark_df, frame_df[[(benchmark_label_str, "Close")]].set_axis(["Close"], axis=1))
                    direct_vix_ser = pd.read_pickle(output_path_obj / "direct_vix.pkl.gz")
                    snapshot_vix_ser = load_vix_close_ser(end_date_str)
                    report_dict["comparison_dict"][f"{pod_str}_vix"] = compare_frames(direct_vix_ser.to_frame("Close"), snapshot_vix_ser.to_frame("Close"))
                    report_dict["comparison_dict"][f"{pod_str}_field_scope"] = {
                        "passed_bool": True, "included_field_list": sorted(DECISION_FIELD_SET),
                        "direct_other_field_list": sorted(set(direct_pricing_df.columns.get_level_values(1)) - DECISION_FIELD_SET),
                        "snapshot_other_field_list": sorted(set(pricing_df.columns.get_level_values(1)) - DECISION_FIELD_SET),
                    }
                    del pricing_df, universe_df, direct_pricing_df, direct_universe_df
        report_dict["source_sha256_after_dict"] = _source_hash_dict()
        report_dict["changed_source_path_list"] = [path_str for path_str, hash_str in report_dict["source_sha256_by_path_dict"].items() if report_dict["source_sha256_after_dict"][path_str] != hash_str]
        report_dict["status_str"] = "pass" if not report_dict["changed_source_path_list"] and all(comparison_dict["passed_bool"] for comparison_dict in report_dict["comparison_dict"].values()) else "fail"
    except Exception as exception_obj:
        report_dict["status_str"] = "fail"
        report_dict["error_dict"] = {"type_str": type(exception_obj).__name__, "message_str": str(exception_obj)}
    finally:
        report_dict["completed_at_str"] = datetime.now(timezone.utc).isoformat()
        report_path_obj.write_text(json.dumps(report_dict, indent=2, allow_nan=False), encoding="utf-8")
    print(f"{report_dict['status_str'].upper()}: {report_path_obj}", flush=True)
    return report_dict


def main() -> int:
    parser_obj = argparse.ArgumentParser(description=__doc__)
    parser_obj.add_argument("--output-dir", required=True)
    parser_obj.add_argument("--end-date", required=True)
    args_obj = parser_obj.parse_args()
    report_dict = run_qualification(Path(args_obj.output_dir), str(args_obj.end_date))
    return 0 if report_dict["status_str"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
