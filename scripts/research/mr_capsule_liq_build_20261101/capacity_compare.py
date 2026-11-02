"""The Bench capacity engine on the 1/N and sized BIL pods, side by side at the same sizes, from the saved runs.

Usage: capacity_compare.py. Reads results/research/strategy/<pod>/capacity_analysis. Writes results/research/mr_capsule_liq_build_20261101/capacity_compare.json.
A book weight of 30% in the capsule, half in each pod, puts 15% of the book in each pod. The capsule row averages the two pods' annual returns: it ignores the daily rebalance between them, a small effect next to the costs here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "results" / "research" / "strategy"
OUT = REPO / "results" / "research" / "mr_capsule_liq_build_20261101"
POD_DICT = {"dv2": ("strategy_mr_dv2_vix_gated_bil", "strategy_mr_dv2_vix_gated_liq_bil"), "hpi": ("strategy_mr_hpi_vote_vix_gated_bil", "strategy_mr_hpi_vote_vix_gated_liq_bil")}
SIZE_LIST = [100_000.0, 250_000.0, 500_000.0, 1_000_000.0, 2_000_000.0, 5_000_000.0]
FIELD_DICT = {
    "baseline_annual_return_float": "base", "central_annual_return_float": "central", "stress_annual_return_float": "stress",
    "baseline_sharpe_float": "base_sharpe", "central_sharpe_float": "central_sharpe", "stress_sharpe_float": "stress_sharpe",
    "order_adv_p50_float": "p50", "order_adv_p95_float": "p95", "order_adv_p99_float": "p99", "order_adv_max_float": "max",
}


def latest_run(stem_str: str) -> Path:
    return sorted((ROOT / stem_str / "capacity_analysis").iterdir())[-1]


def curve(stem_str: str, window_str: str) -> pd.DataFrame:
    frame_df = pd.read_csv(latest_run(stem_str) / "capacity_curve.csv")
    frame_df = frame_df[frame_df["window_str"] == window_str].set_index("capital_base_float")
    return frame_df.reindex(SIZE_LIST)[list(FIELD_DICT)].rename(columns=FIELD_DICT)


def main() -> None:
    pd.set_option("display.width", 200)
    report_dict: dict[str, object] = {}
    for window_str in ("recent_5y", "full_history"):
        pod_frame_dict = {}
        for pod_str, (plain_str, sized_str) in POD_DICT.items():
            plain_df, sized_df = curve(plain_str, window_str), curve(sized_str, window_str)
            frame_df = pd.DataFrame({
                "1n_base": plain_df["base"], "1n_central": plain_df["central"], "1n_stress": plain_df["stress"], "1n_sharpe": plain_df["central_sharpe"], "1n_p95": plain_df["p95"], "1n_p99": plain_df["p99"],
                "sized_base": sized_df["base"], "sized_central": sized_df["central"], "sized_stress": sized_df["stress"], "sized_sharpe": sized_df["central_sharpe"], "sized_p95": sized_df["p95"], "sized_p99": sized_df["p99"],
            })
            frame_df["gap_central"] = frame_df["sized_central"] - frame_df["1n_central"]
            pod_frame_dict[pod_str] = frame_df
            print(f"== {window_str} {pod_str}\n", frame_df.round(4).to_string())
        capsule_df = pd.DataFrame({
            "1n_central": (pod_frame_dict["dv2"]["1n_central"] + pod_frame_dict["hpi"]["1n_central"]) / 2.0,
            "sized_central": (pod_frame_dict["dv2"]["sized_central"] + pod_frame_dict["hpi"]["sized_central"]) / 2.0,
            "1n_stress": (pod_frame_dict["dv2"]["1n_stress"] + pod_frame_dict["hpi"]["1n_stress"]) / 2.0,
            "sized_stress": (pod_frame_dict["dv2"]["sized_stress"] + pod_frame_dict["hpi"]["sized_stress"]) / 2.0,
            "1n_base": (pod_frame_dict["dv2"]["1n_base"] + pod_frame_dict["hpi"]["1n_base"]) / 2.0,
            "sized_base": (pod_frame_dict["dv2"]["sized_base"] + pod_frame_dict["hpi"]["sized_base"]) / 2.0,
        })
        capsule_df["gap_central"] = capsule_df["sized_central"] - capsule_df["1n_central"]
        capsule_df["book_gap_points"] = capsule_df["gap_central"] * 30.0
        capsule_df["book_size"] = capsule_df.index / 0.15
        capsule_df["sized_cost_points"] = (capsule_df["sized_base"] - capsule_df["sized_central"]) * 100.0
        capsule_df["1n_cost_points"] = (capsule_df["1n_base"] - capsule_df["1n_central"]) * 100.0
        print(f"== {window_str} capsule\n", capsule_df.round(4).to_string())
        report_dict[window_str] = {"capsule": {str(int(k)): v for k, v in capsule_df.round(5).to_dict("index").items()}, **{pod_str: {str(int(k)): v for k, v in frame_df.round(5).to_dict("index").items()} for pod_str, frame_df in pod_frame_dict.items()}}
    summary_dict = {}
    for pod_str, stem_tuple in POD_DICT.items():
        for stem_str in stem_tuple:
            summary = json.loads((latest_run(stem_str) / "summary.json").read_text(encoding="utf-8"))
            summary_dict[stem_str] = {key_str: summary[key_str] for key_str in ("recommended_capacity_float", "outer_capacity_float", "break_even_capacity_bracket_str", "headline_window_str", "aum_grid_list")}
            print(stem_str, summary_dict[stem_str])
    report_dict["summary"] = summary_dict
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "capacity_compare.json").write_text(json.dumps(report_dict, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
