"""The real Timing hook on real data: its default cell (next open, next open) against the saved Bench run of the same pod, trade for trade.

Usage: timing_check.py <strategy stem>, for example strategy_mr_dv2_vix_gated_liq_bil. Needs a saved vanilla run in results/research/strategy/<stem>/vanilla_backtest. Norgate loads. Writes results/research/mr_capsule_liq_build_20261101/timing_check_<stem>.json.
"""

from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUT = REPO / "results" / "research" / "mr_capsule_liq_build_20261101"


def main(stem_str: str) -> None:
    from alpha.engine.execution_timing import ExecutionTimingAnalysis

    start_float = time.time()
    run_folder = sorted((REPO / "results" / "research" / "strategy" / stem_str / "vanilla_backtest").iterdir())[-1]
    bench_summary_dict = json.loads((run_folder / "summary.json").read_text(encoding="utf-8"))
    bench_df = pd.read_csv(run_folder / "transactions.csv")
    module_obj = importlib.import_module(f"strategies.mr_capsule.{stem_str}")
    input_dict = module_obj.build_execution_timing_analysis_inputs()
    input_dict.update(entry_timing_str_tuple=("next_open",), exit_timing_str_tuple=("next_open",))
    result_obj = ExecutionTimingAnalysis(**input_dict, save_output_bool=False).run()
    timing_obj = result_obj.strategy_map[("next_open", "next_open")]
    timing_df = timing_obj.get_transactions()[["bar", "asset", "amount", "price"]].copy()
    timing_df["bar"] = pd.to_datetime(timing_df["bar"]).dt.strftime("%Y-%m-%d")
    bench_df = bench_df[["bar", "asset", "amount", "price"]].copy()
    bench_df["bar"] = pd.to_datetime(bench_df["bar"]).dt.strftime("%Y-%m-%d")
    key_list = ["bar", "asset", "amount"]
    timing_df, bench_df = timing_df.sort_values(key_list).reset_index(drop=True), bench_df.sort_values(key_list).reset_index(drop=True)
    merged_df = timing_df.merge(bench_df, on=["bar", "asset"], how="outer", suffixes=("_timing", "_bench"), indicator=True)
    both_df = merged_df[merged_df["_merge"] == "both"]
    report_dict = {
        "stem_str": stem_str, "bench_run_str": run_folder.name, "seconds_int": round(time.time() - start_float),
        "final_equity_timing_float": float(timing_obj.results["total_value"].iloc[-1]), "final_equity_bench_float": float(bench_summary_dict["final_equity"]),
        "rows_timing_int": len(timing_df), "rows_bench_int": len(bench_df),
        "rows_only_timing_int": int((merged_df["_merge"] == "left_only").sum()), "rows_only_bench_int": int((merged_df["_merge"] == "right_only").sum()),
        "max_share_gap_float": float((both_df["amount_timing"] - both_df["amount_bench"]).abs().max()), "max_price_gap_float": float((both_df["price_timing"] - both_df["price_bench"]).abs().max()),
        "first_only_rows_list": merged_df[merged_df["_merge"] != "both"].head(12).astype(str).to_dict("records"),
        "default_row_dict": {key_str: result_obj.metric_df.iloc[0][key_str] for key_str in ("Ann. Return [%]", "Sharpe", "Max Drawdown [%]")},
        "bench_summary_dict": {key_str: bench_summary_dict[key_str] for key_str in ("ann_return_pct", "sharpe", "max_drawdown_pct")},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"timing_check_{stem_str}.json").write_text(json.dumps(report_dict, indent=1, default=str), encoding="utf-8")
    print(json.dumps(report_dict, default=str), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
