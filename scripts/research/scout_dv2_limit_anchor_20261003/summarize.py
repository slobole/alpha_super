"""Tables of the DV2 limit-anchor study from results/scout/dv2_limit_anchor/ (printed as Markdown, written as CSV under
results/scout/dv2_limit_anchor/tables/).

    PYTHONUTF8=1 uv run python scripts/research/scout_dv2_limit_anchor_20261003/summarize.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH
from scripts.research.scout_dv2_limit_anchor_20261003.register import UNIVERSE_TUPLE

OUT_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "dv2_limit_anchor"
ERA_LIST = ["2004-2007", "2008-2015", "2016-2022"]


def slug(name_str: str) -> str:
    return name_str.replace(" ", "_").replace("&", "and")


def load(folder_str: str, name_str: str) -> dict | None:
    path = OUT_PATH / folder_str / f"{slug(name_str)}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def cell_row(universe_str: str, row: dict) -> dict:
    events = row.get("events", {})
    out = {"universe": universe_str, "anchor": row["anchor_str"], "measure": row["measure_str"], "target": row["fill_target_float"],
           "exit": row["exit_str"], "param": row["param_float"], "fill_cal": row["fill_rate_2004_2012_float"],
           "fill_full": row["orders"]["fill_rate_float"], "fill_2013_22": row["fill_rate_2013_2022_float"],
           "trades_yr": row["orders"]["trades_per_year_float"], "sharpe_gross": row["gross"]["sharpe_float"],
           "sharpe_ar": row["ar"]["sharpe_float"], "sharpe_pooled": row["pooled"]["sharpe_float"],
           "cagr_pooled": row["pooled"]["cagr_float"], "maxdd_pooled": row["pooled"]["max_drawdown_float"],
           "cost_rt_ar": row["ar"].get("cost_per_round_trip_bp_float"), "cost_rt_pooled": row["pooled"].get("cost_per_round_trip_bp_float"),
           "post_filled_bp": events.get("post_fill_day_filled", {}).get("mean_bp_float"),
           "post_unfilled_bp": events.get("post_fill_day_unfilled", {}).get("mean_bp_float"),
           "post_gap_bp": events.get("post_fill_day_gap_bp_float"), "jaccard_base": events.get("jaccard_with_baseline_float"),
           "stress_f05_ar": row["fill_stress"]["f0.5"]["sharpe_ar"], "stress_f05_pooled": row["fill_stress"]["f0.5"]["sharpe_pooled"],
           "stress_f1_ar": row["fill_stress"]["f1"]["sharpe_ar"], "stress_f1_pooled": row["fill_stress"]["f1"]["sharpe_pooled"],
           "stress_f1_fill": row["fill_stress"]["f1"]["fill_rate_float"]}
    for era_str in ERA_LIST:
        out[f"pooled_{era_str}"] = row["pooled"]["era_sharpe_dict"][era_str]
    comparison = row.get("vs_baseline")
    if comparison:
        for case_str in ("ar", "pooled"):
            out[f"d_{case_str}"] = comparison[case_str]["difference_float"]
            out[f"d_{case_str}_lo"] = comparison[case_str]["ci_low_float"]
            out[f"d_{case_str}_hi"] = comparison[case_str]["ci_high_float"]
            out[f"p_not_better_{case_str}"] = comparison[case_str]["p_not_better_float"]
    return out


def main() -> None:
    (OUT_PATH / "tables").mkdir(parents=True, exist_ok=True)
    rows, correlation_rows, reference_rows = [], [], []
    for name_str in UNIVERSE_TUPLE:
        result = load("universes", name_str)
        if result is None:
            continue
        rows += [cell_row(name_str, row) for row in result["cells"].values()]
        moo_result = load("moo_exit", name_str)
        if moo_result is not None:
            rows += [cell_row(name_str, row) for key_str, row in moo_result.items() if isinstance(row, dict) and "anchor_str" in row]
        for measure_str, stats in result["correlation"].items():
            correlation_rows.append({"universe": name_str, "measure vs NATR14": measure_str, "daily_pearson": stats["mean_daily_pearson_float"],
                                     "daily_spearman": stats["mean_daily_spearman_float"], "pooled_pearson": stats["pooled_pearson_float"]})
        reference = result["reference_moo_moo"]
        reference_rows.append({"universe": name_str, "sharpe_gross": reference["gross"]["sharpe_float"], "sharpe_ar": reference["ar"]["sharpe_float"],
                               "sharpe_pooled": reference["pooled"]["sharpe_float"], "cagr_pooled": reference["pooled"]["cagr_float"],
                               "maxdd_pooled": reference["pooled"]["max_drawdown_float"],
                               "uncalibrated_quantile_fill": result["uncalibrated_quantile_event_fill_share"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_PATH / "tables" / "cells.csv", index=False)
    pd.DataFrame(correlation_rows).to_csv(OUT_PATH / "tables" / "correlation.csv", index=False)
    pd.set_option("display.width", 250)
    print("## Reference: market-on-open entry and exit\n")
    print(pd.DataFrame(reference_rows).drop(columns="uncalibrated_quantile_fill").round(2).to_string(index=False))
    for row in reference_rows:
        print(row["universe"], "uncalibrated quantile event fill share:", {k: round(v, 2) for k, v in row["uncalibrated_quantile_fill"].items()})
    print("\n## Correlation with NATR14 on signal days\n")
    print(pd.DataFrame(correlation_rows).round(2).to_string(index=False))
    main_columns = ["anchor", "measure", "target", "exit", "param", "fill_cal", "fill_full", "trades_yr", "sharpe_gross", "sharpe_ar", "sharpe_pooled",
                    "cagr_pooled", "maxdd_pooled", "cost_rt_ar", "cost_rt_pooled", "post_filled_bp", "post_unfilled_bp", "stress_f1_ar",
                    "stress_f1_pooled"]
    compare_columns = ["anchor", "measure", "target", "exit", "d_ar", "d_ar_lo", "d_ar_hi", "d_pooled", "d_pooled_lo", "d_pooled_hi", "jaccard_base",
                       "fill_2013_22", "stress_f05_ar", "stress_f05_pooled"] + [f"pooled_{e}" for e in ERA_LIST]
    for name_str in UNIVERSE_TUPLE:
        sub = frame[frame["universe"] == name_str].sort_values(["exit", "target", "anchor", "measure"])
        if sub.empty:
            continue
        print(f"\n## {name_str}: cells\n")
        print(sub[main_columns].round(3).to_string(index=False))
        print(f"\n## {name_str}: paired difference vs close x natr14 at the same target (95% stationary-bootstrap CI), eras, overlap\n")
        print(sub[compare_columns].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
