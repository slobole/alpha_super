"""Compare saved HPI capsule engine inputs; no provider or engine imports."""
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import pandas as pd

REPO_PATH = Path(__file__).resolve().parents[3]
AUDIT_PATH = REPO_PATH / "results/review/wired_signoff_20261015"


def file_manifest(path_obj):
    digest_obj = hashlib.sha256()
    with path_obj.open("rb") as handle_obj:
        for chunk_bytes in iter(lambda: handle_obj.read(1024 * 1024), b""):
            digest_obj.update(chunk_bytes)
    return {"path": str(path_obj.resolve()), "bytes": path_obj.stat().st_size, "sha256": digest_obj.hexdigest()}


def main():
    row_list = []
    source_dict = file_manifest(Path(__file__))
    for parking_str in ("cash", "bil", "spmo"):
        key_str = f"strategy_mr_hpi_vote_vix_gated_{parking_str}"
        prepared_path = AUDIT_PATH / "input_preparation" / key_str / "pricing.parquet"
        baseline_path = AUDIT_PATH / "baselines" / key_str / "pricing.parquet"
        prepared_auxiliary_path = prepared_path.parent / "auxiliary.pkl.gz"
        baseline_auxiliary_path = baseline_path.parent / "auxiliary.pkl.gz"
        input_path_list = [prepared_path, baseline_path, prepared_auxiliary_path, baseline_auxiliary_path]
        input_list = [file_manifest(path_obj) for path_obj in input_path_list]
        prepared_df = pd.read_parquet(prepared_path)
        baseline_df = pd.read_parquet(baseline_path)
        row_dict = {"key": key_str, "input_manifest": input_list}
        try:
            pd.testing.assert_frame_equal(prepared_df, baseline_df, check_exact=True)
            row_dict["exact_values_index_columns_dtypes"] = True
        except AssertionError as exception_obj:
            row_dict.update(exact_values_index_columns_dtypes=False, content_difference=str(exception_obj)[:1500])
        prepared_attrs_str = json.dumps(prepared_df.attrs, sort_keys=True, default=str)
        baseline_attrs_str = json.dumps(baseline_df.attrs, sort_keys=True, default=str)
        row_dict.update(exact_attrs=prepared_attrs_str == baseline_attrs_str,
            prepared_attrs_sha256=hashlib.sha256(prepared_attrs_str.encode()).hexdigest(),
            baseline_attrs_sha256=hashlib.sha256(baseline_attrs_str.encode()).hexdigest(),
            rows=len(prepared_df), columns=len(prepared_df.columns))
        if prepared_attrs_str != baseline_attrs_str:
            row_dict["prepared_attrs"] = prepared_df.attrs
            row_dict["baseline_attrs"] = baseline_df.attrs
        prepared_auxiliary_dict = pd.read_pickle(prepared_auxiliary_path)
        baseline_auxiliary_dict = pd.read_pickle(baseline_auxiliary_path)
        row_dict["exact_auxiliary_inputs"] = prepared_auxiliary_dict.keys() == baseline_auxiliary_dict.keys()
        row_dict["auxiliary_keys"] = sorted(prepared_auxiliary_dict)
        for name_str, prepared_obj in prepared_auxiliary_dict.items():
            try:
                if isinstance(prepared_obj, pd.DataFrame):
                    pd.testing.assert_frame_equal(prepared_obj, baseline_auxiliary_dict[name_str], check_exact=True)
                else:
                    pd.testing.assert_series_equal(prepared_obj, baseline_auxiliary_dict[name_str], check_exact=True)
            except (AssertionError, KeyError) as exception_obj:
                row_dict["exact_auxiliary_inputs"] = False
                row_dict["auxiliary_difference"] = str(exception_obj)[:1500]
        row_dict["input_hashes_unchanged"] = input_list == [file_manifest(path_obj) for path_obj in input_path_list]
        row_dict["status"] = "pass" if all(row_dict[name_str] for name_str in
            ("exact_values_index_columns_dtypes", "exact_attrs", "exact_auxiliary_inputs", "input_hashes_unchanged")) else "difference"
        row_list.append(row_dict)
        print(json.dumps({"key": key_str, "status": row_dict["status"], "exact_attrs": row_dict["exact_attrs"]}), flush=True)
        del prepared_df, baseline_df
    report_dict = {"generated_at_utc": datetime.now(UTC).isoformat(), "source": source_dict,
        "source_unchanged": source_dict == file_manifest(Path(__file__)), "rows": row_list,
        "scope": "Saved original preparation versus saved completed baseline engine-boundary input, exact content/dtype/calendar/order and metadata. No new strategy, provider IO, engine run or broker acceptance claim."}
    (AUDIT_PATH / "ops_hpi_input_equivalence.json").write_text(json.dumps(report_dict, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
