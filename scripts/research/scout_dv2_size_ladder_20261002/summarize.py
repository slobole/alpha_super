"""Collect the size-ladder JSON results into CSV tables (results/scout/dv2_size_ladder/tables/) and print them.

    PYTHONUTF8=1 uv run python scripts/research/scout_dv2_size_ladder_20261002/summarize.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from register import UNIVERSE_TUPLE
from run_ladder import OUT_PATH, slug


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> None:
    table_path = OUT_PATH / "tables"
    table_path.mkdir(parents=True, exist_ok=True)
    s3_rows, pod_rows, cap_rows = [], [], []
    for name_str, bucket_str in UNIVERSE_TUPLE:
        result = _load(OUT_PATH / "universes" / f"{slug(name_str)}.json")
        if result is None:
            continue
        head = result["s3"]["headline"]
        liquidity = result["s3"]["liquidity_cost"]
        mcpt = _load(OUT_PATH / "mcpt" / f"{slug(name_str)}.json")
        s3_rows.append({
            "universe": name_str, "size": bucket_str, "start": result["start_str"], "members_median": result["members_median_int"],
            "events": head["events_int"], "dates": head["event_dates_int"], "bp": round(head["date_mean_excess_float"] * 1e4, 2),
            "nw_t": round(head["nw_t_float"], 2), "placebo_p": round(head["placebo_p_float"], 3),
            "pos_years": round(head["positive_year_share_float"], 2), "cov_s3": round(head["cost_coverage_float"], 2),
            "cov_liq": round(liquidity["coverage_float"], 2), "event_hs_bp": round(liquidity["event_median_half_spread_float"] * 1e4, 1),
            "event_adv63_musd": round(liquidity["event_median_adv63_float"] / 1e6, 1), "verdict": result["s3"]["verdict_str"],
        })
        pod = result["pod"]
        row = {"universe": name_str, "size": bucket_str, "hold_med": pod["median_hold_sessions_int"],
               "trades_yr": round(pod["engine"]["trades_per_year_float"], 0)}
        for case_str, short_str in (("gross", "gross"), ("engine", "eng"), ("engine_adjusted_units", "eng_adj"), ("stress_2x_plus_10bp", "stress"),
                                    ("liquidity_aware", "liq")):
            metric = pod[case_str]
            row[f"{short_str}_cagr"] = round(metric["cagr_float"] * 100, 1)
            row[f"{short_str}_sharpe"] = round(metric["sharpe_float"], 2)
            row[f"{short_str}_maxdd"] = round(metric["max_drawdown_float"] * 100, 0)
            row[f"{short_str}_active_sharpe"] = round(metric["active_sharpe_float"], 2)
        row["liq_slip_entry_med_bp"] = round(pod["liquidity_slippage"]["entry_median_bp_float"], 1)
        row["mcpt_p"] = mcpt["p_value_float"] if mcpt else None
        pod_rows.append(row)
        cap_rows.append({"universe": name_str, "cap_recent3y_musd": round(pod["capacity"]["recent_3y_aum_float"] / 1e6, 2),
                         "cap_full_musd": round(pod["capacity"]["full_history_aum_float"] / 1e6, 2),
                         "binding": pod["capacity"]["binding_asset_str"], "fills": pod["capacity"]["trade_count_int"]})
    for file_str, rows in (("s3_by_universe.csv", s3_rows), ("pod_by_universe.csv", pod_rows), ("capacity_by_universe.csv", cap_rows)):
        frame = pd.DataFrame(rows)
        frame.to_csv(table_path / file_str, index=False)
        print(f"\n== {file_str}\n{frame.to_string(index=False)}")
    for file_str in ("Russell_3000_+_Micro_Cap", "SandP_Composite_1500"):
        result = _load(OUT_PATH / "buckets" / f"{file_str}.json")
        if result is None:
            continue
        rows = []
        whole = result["whole_universe"]
        for kind_str, row_list in (("membership", result["membership_buckets"]), ("adv63", result["adv_terciles"]["rows"])):
            for row in row_list:
                w, v = row["within_bucket"], row["vs_whole_universe"]
                rows.append({"kind": kind_str, "bucket": row["bucket_str"], "members_med": row.get("members_median_int"), "events": w["events_int"],
                             "dates": w["event_dates_int"], "bp": round(w["date_mean_excess_float"] * 1e4, 2), "nw_t": round(w["nw_t_float"], 2),
                             "placebo_p": round(w.get("placebo_p_float", float("nan")), 3), "pos_years": round(w.get("positive_year_share_float", float("nan")), 2),
                             "bp_vs_universe": round(v["date_mean_excess_float"] * 1e4, 2), "t_vs_universe": round(v["nw_t_float"], 2),
                             "hs_bp": round(w.get("event_median_half_spread_bp_float", float("nan")), 1),
                             "rt_cost_bp": round(w.get("liquidity_round_trip_float", float("nan")) * 1e4, 1),
                             "cov_liq": round(w.get("liquidity_coverage_float", float("nan")), 2),
                             "adv63_musd": round(w.get("event_median_adv63_musd_float", float("nan")), 1),
                             "era_04_07": round(w.get("eras", {}).get("1998-2007", float("nan")) * 1e4, 1),
                             "era_08_15": round(w.get("eras", {}).get("2008-2015", float("nan")) * 1e4, 1),
                             "era_16_22": round(w.get("eras", {}).get("2016-2022", float("nan")) * 1e4, 1)})
        frame = pd.DataFrame(rows)
        frame.to_csv(table_path / f"buckets_{file_str}.csv", index=False)
        print(f"\n== buckets {result['universe_str']} (whole: {whole['date_mean_excess_float'] * 1e4:+.2f} bp, t {whole['nw_t_float']:.2f}, "
              f"{whole['events_int']} events; ADV unranked share {result['adv_terciles']['event_unranked_share_float']:.3f})\n{frame.to_string(index=False)}")


if __name__ == "__main__":
    main()
