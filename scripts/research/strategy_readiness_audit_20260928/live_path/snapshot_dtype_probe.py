"""
Audit probe (read-only, local Norgate): do non-CORE5 snapshot profiles deliver the same
dtype as the direct Norgate loader used by backtests?

Direct Norgate price_timeseries returns float32 for every field. The snapshot exporter
concatenates per-symbol frames (scripts/export_norgate_snapshot.py:206-239) and inserts a
float64 Dividend=0.0 for index symbols ($VIX/$VXN); only CORE5 restores source dtypes on
read (data/norgate_snapshot_store.py:627-634, 748-757). This probe builds the price frame
for a few symbols with the exporter's own function, round-trips it through parquet in a
temp dir and reports the dtypes a live host would receive.

Output: results/research/strategy_readiness_audit_20260928/live_path/snapshot_dtype_probe.json
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scripts.export_norgate_snapshot import _build_price_snapshot_df  # noqa: E402

OUTPUT_PATH_OBJ = Path("results/research/strategy_readiness_audit_20260928/live_path/snapshot_dtype_probe.json")


def main() -> None:
    price_df = _build_price_snapshot_df(
        capital_symbol_list=["AAPL", "SPY", "$VXN"],
        total_return_symbol_list=["SPY"],
        start_date_str="2024-01-01",
        end_date_str="2026-09-25",
    )
    price_df.attrs.clear()
    concat_dtype_dict = {str(k): str(v) for k, v in price_df.dtypes.items()}
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        parquet_path_obj = Path(tmp_dir_str) / "prices.parquet"
        price_df.to_parquet(parquet_path_obj, index=False)
        read_df = pd.read_parquet(parquet_path_obj)
    read_dtype_dict = {str(k): str(v) for k, v in read_df.dtypes.items()}
    result_dict = {"concat_dtype_dict": concat_dtype_dict, "parquet_read_dtype_dict": read_dtype_dict}
    OUTPUT_PATH_OBJ.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH_OBJ.write_text(json.dumps(result_dict, indent=2), encoding="utf-8")
    print(json.dumps(result_dict, indent=2))


if __name__ == "__main__":
    main()
