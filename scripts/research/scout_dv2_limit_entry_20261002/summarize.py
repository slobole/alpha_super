"""Tables of the DV2 limit-entry study from results/scout/dv2_limit_entry/universes/*.json (printed as Markdown and
written as CSV under results/scout/dv2_limit_entry/tables/).

    PYTHONUTF8=1 uv run python scripts/research/scout_dv2_limit_entry_20261002/summarize.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from register import UNIVERSE_TUPLE
from run_limit import ERA_TUPLE, OUT_PATH, slug


def load(name_str: str) -> dict | None:
    path = OUT_PATH / "universes" / f"{slug(name_str)}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def pod_rows(result: dict) -> list[dict]:
    rows = []
    for row in result["pods"].values():
        orders = row["orders"]
        out = {"universe": result["universe_str"], "entry": row["entry"], "exit": row["exit_str"],
               "fill_rate": orders["fill_rate_float"], "trades_yr": orders["trades_per_year_float"], "mean_positions": orders["mean_positions_float"],
               "entry_passive": orders["entry_passive_share_float"], "exit_passive": orders["exit_passive_share_float"],
               "exit_forced": orders["exit_forced_share_float"], "sharpe_gross": row["gross"]["sharpe_float"]}
        for case_str in ("ar", "pooled"):
            case = row[case_str]
            out.update({f"sharpe_{case_str}": case["sharpe_float"], f"cagr_{case_str}": case["cagr_float"],
                        f"maxdd_{case_str}": case["max_drawdown_float"], f"cost_rt_bp_{case_str}": case.get("cost_per_round_trip_bp_float"),
                        f"active_sharpe_{case_str}": case["active_sharpe_float"], f"ruin_{case_str}": case["ruin_date_str"]})
            for era_str, _, _ in ERA_TUPLE:
                out[f"{era_str}_{case_str}"] = case["era_sharpe_dict"][era_str]
        for era_str, _, _ in ERA_TUPLE:
            out[f"{era_str}_gross"] = row["gross"]["era_sharpe_dict"][era_str]
        out["capacity_recent_musd"] = row["capacity_pooled"]["recent_3y_aum_float"] / 1e6
        out["capacity_full_musd"] = row["capacity_pooled"]["full_history_aum_float"] / 1e6
        out["capacity_binding"] = row["capacity_pooled"]["binding_asset_str"]
        rows.append(out)
    return rows


def event_rows(result: dict) -> list[dict]:
    events = result["events"]
    base = events["all_events"]
    rows = [{"universe": result["universe_str"], "order": "moo (all events)", "fill_share": 1.0,
             "all_at_open_bp": base["moo_entry"]["mean_bp_float"], "all_at_open_t": base["moo_entry"]["nw_t_float"],
             "all_net_pooled_bp": base["moo_entry_net_pooled"]["mean_bp_float"], "all_close_anchor_bp": base["close_anchor"]["mean_bp_float"],
             **{f"all_{e}_bp": base["moo_entry"]["era_bp_dict"][e] for e, _, _ in ERA_TUPLE}}]
    for key_str, row in events.items():
        if key_str == "all_events":
            continue
        rows.append({"universe": result["universe_str"], "order": key_str, "fill_share": row["fill_share_float"],
                     "open_fill_share_of_fills": row["open_fill_share_of_fills_float"],
                     "filled_at_open_bp": row["filled_moo_entry"]["mean_bp_float"], "filled_at_open_t": row["filled_moo_entry"]["nw_t_float"],
                     "unfilled_at_open_bp": row["unfilled_moo_entry"]["mean_bp_float"], "unfilled_at_open_t": row["unfilled_moo_entry"]["nw_t_float"],
                     "filled_from_fill_bp": row["filled_from_fill_price"]["mean_bp_float"], "filled_from_fill_t": row["filled_from_fill_price"]["nw_t_float"],
                     "filled_from_fill_net_pooled_bp": row["filled_from_fill_price_net_pooled"]["mean_bp_float"],
                     "filled_close_anchor_bp": row["filled_close_anchor"]["mean_bp_float"], "unfilled_close_anchor_bp": row["unfilled_close_anchor"]["mean_bp_float"],
                     "saving_vs_open_bp": row["filled_saving_vs_open"]["mean_bp_float"],
                     **{f"filled_from_fill_{e}_bp": row["filled_from_fill_price"].get("era_bp_dict", {}).get(e) for e, _, _ in ERA_TUPLE},
                     **{f"filled_at_open_{e}_bp": row["filled_moo_entry"].get("era_bp_dict", {}).get(e) for e, _, _ in ERA_TUPLE},
                     **{f"unfilled_at_open_{e}_bp": row["unfilled_moo_entry"].get("era_bp_dict", {}).get(e) for e, _, _ in ERA_TUPLE}})
    return rows


def main() -> None:
    results = [r for r in (load(n) for n in UNIVERSE_TUPLE) if r is not None]
    pod_df = pd.DataFrame([row for r in results for row in pod_rows(r)])
    event_df = pd.DataFrame([row for r in results for row in event_rows(r)])
    table_path = OUT_PATH / "tables"
    table_path.mkdir(parents=True, exist_ok=True)
    pod_df.to_csv(table_path / "pods.csv", index=False)
    event_df.to_csv(table_path / "events.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    def fmt(df: pd.DataFrame) -> str:  # a Markdown table without the optional tabulate dependency
        cell = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)
        lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
        lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
        return "\n".join(lines)
    print("## Pods\n")
    print(fmt(pod_df[["universe", "entry", "exit", "fill_rate", "trades_yr", "mean_positions", "sharpe_gross", "sharpe_ar", "sharpe_pooled",
                      "cagr_ar", "cagr_pooled", "maxdd_ar", "maxdd_pooled", "cost_rt_bp_ar", "cost_rt_bp_pooled"]]))
    print("\n## Order mix\n")
    print(fmt(pod_df[["universe", "entry", "exit", "entry_passive", "exit_passive", "exit_forced", "active_sharpe_ar", "active_sharpe_pooled",
                      "ruin_ar", "ruin_pooled"]]))
    print("\n## Eras (net Sharpe)\n")
    era_column_list = [f"{e}_{c}" for c in ("gross", "ar", "pooled") for e, _, _ in ERA_TUPLE]
    print(fmt(pod_df[["universe", "entry", "exit", *era_column_list]]))
    print("\n## Adverse selection (h3 excess over same-date regime members, bp)\n")
    print(fmt(event_df[["universe", "order", "fill_share", "all_at_open_bp", "all_at_open_t", "filled_at_open_bp", "unfilled_at_open_bp",
                        "filled_from_fill_bp", "filled_from_fill_t", "filled_from_fill_net_pooled_bp", "all_net_pooled_bp",
                        "filled_close_anchor_bp", "unfilled_close_anchor_bp", "saving_vs_open_bp"]]))
    print("\n## Adverse selection by era (bp)\n")
    era_event_list = [c for c in event_df.columns if any(c.endswith(f"{e}_bp") for e, _, _ in ERA_TUPLE)]
    print(fmt(event_df[["universe", "order", *era_event_list]]))
    print("\n## Capacity (pooled-case fills, 1% of ADV63) for the best variant (highest min of AR and pooled net Sharpe) and the baseline\n")
    pod_df["min_net"] = pod_df[["sharpe_ar", "sharpe_pooled"]].min(axis=1)
    best_df = pod_df.loc[pod_df.groupby("universe")["min_net"].idxmax()]
    base_df = pod_df[(pod_df["entry"] == "moo") & (pod_df["exit"] == "moo")]
    print(fmt(pd.concat([base_df.assign(row="baseline"), best_df.assign(row="best")])[
        ["universe", "row", "entry", "exit", "sharpe_ar", "sharpe_pooled", "capacity_recent_musd", "capacity_full_musd", "capacity_binding", "entry_passive"]]))


if __name__ == "__main__":
    main()
