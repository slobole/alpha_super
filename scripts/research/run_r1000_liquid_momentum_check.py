"""Russell 1000 scale-free momentum leg with a liquidity filter (research only).

Frozen plan: docs/research/R1000_LIQUID_MOMENTUM_PREREG_20260926.md. Uses the NDX parameter-robustness replica and
the cached Russell 1000 universe (run_ndx_param_robustness_study.py --prepare R1000 first).

    uv run python scripts/research/run_r1000_liquid_momentum_check.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_ndx_param_robustness_study as an  # noqa: E402
import ndx_param_robustness_core as core  # noqa: E402
from run_ndx_param_robustness_study import capacity_stats  # noqa: E402

W_CELL = core.Cell(
    numerator_str="ROC12-1", denominator_str="NATR20", n_int=30, weight_str="IV", stock_filter_int=200,
    regime_str="SPY", buffer_int=2, offset_int=0, vxn_target_float=18.0, vxn_floor_float=0.125,
)
A0R_CELL = core.ANCHOR_CELL
FILTER_TUPLE = ("none", "REL25", "REL50", "ABS5M")
COST_DICT = {"engine": core.ENGINE_SLIPPAGE_FLOAT, "stress5": core.STRESS_SLIPPAGE_FLOAT}


def main() -> None:
    universe_dict = core.load_universe("R1000")
    feature_obj = core.FeatureBook(universe_dict)
    adv_arr = feature_obj.adv20_dollar()
    date_index = universe_dict["date_index"]
    taa_long_ser = an.load_taa_long_ser()
    ndx_ret_dict = {c: pd.read_parquet(an.OUT_PATH / f"returns_NDX_{c}.parquet")[an.L_KEY] for c in COST_DICT}

    ret_dict = {c: {} for c in COST_DICT}
    run_meta = {}
    selection_dict = {}
    for base_str, base_cell in (("W", W_CELL), ("A0R", A0R_CELL)):
        for filter_str in FILTER_TUPLE:
            cell = dataclasses.replace(base_cell, liquidity_str=filter_str)
            label_str = f"{base_str}_{filter_str}"
            target_list = core.build_target_list(feature_obj, cell)
            selection_dict[label_str] = {t["decision_pos"]: set(t["symbol_idx_vec"].tolist()) for t in target_list}
            for cost_str, slip_float in COST_DICT.items():
                sim = core.simulate(universe_dict, target_list, slippage_float=slip_float,
                                    adv_arr=adv_arr if cost_str == "engine" else None)
                ret_dict[cost_str][label_str] = sim["return_ser"]
                if cost_str == "engine":
                    window = sim["total_ser"].loc["2000-01-04":an.END_TS]
                    run_meta[label_str] = {
                        "cell_key": cell.key_str,
                        "turnover_x_per_year": float(sim["traded_notional_ser"].loc["2000-01-04":an.END_TS].sum() / window.mean() / (len(window) / 252.0)),
                        "capacity_full": capacity_stats(sim["order_frac_arr"], date_index, None),
                        "capacity_2021_2026": capacity_stats(sim["order_frac_arr"], date_index, pd.Timestamp("2021-01-01")),
                        "mean_names_held": float(np.mean([len(t["symbol_idx_vec"]) for t in target_list if len(t["symbol_idx_vec"])])),
                    }

    # pool removed by the filter (PIT members with finite ADV20, decision dates of the month-end schedule)
    schedule_dict = core.build_schedule(universe_dict, 0)
    decision_pos_vec = schedule_dict["decision_pos_vec"][schedule_dict["trade_vec"]]
    member_arr = universe_dict["member_arr"]
    pool_dict = {}
    for filter_str in FILTER_TUPLE[1:]:
        removed_list = []
        for pos in decision_pos_vec:
            members = member_arr[pos] == 1
            keep = core.liquidity_pass_vec(feature_obj, int(pos), member_arr[pos], filter_str)
            removed_list.append(1.0 - (members & keep).sum() / max(members.sum(), 1))
        pool_dict[filter_str] = {"mean_share_of_members_removed": float(np.mean(removed_list)),
                                 "first_2000s": float(np.mean(removed_list[:24])), "last_24m": float(np.mean(removed_list[-24:]))}
    for base_str in ("W", "A0R"):
        for filter_str in FILTER_TUPLE[1:]:
            a, b = selection_dict[f"{base_str}_none"], selection_dict[f"{base_str}_{filter_str}"]
            ov = [len(a[p] & b[p]) / max(len(a[p]), 1) for p in a if a[p]]
            run_meta[f"{base_str}_{filter_str}"]["overlap_with_unfiltered_share"] = float(np.mean(ov))

    # metrics
    res: dict = {"meta": run_meta, "pool_removed": pool_dict}
    for cost_str in COST_DICT:
        frame = pd.DataFrame(ret_dict[cost_str])
        frame["NDX_L"] = ndx_ret_dict[cost_str]
        s, g = an.all_metrics(frame, taa_long_ser)
        res[cost_str] = {"standalone": s, "g3": g}

    def rule(cost_str: str, label_str: str) -> dict:
        g = res[cost_str]["g3"]
        margin = {b: g[label_str][b]["sharpe"] - g["NDX_L"][b]["sharpe"] for b in an.RULE_BLOCK_TUPLE}
        dd_gap = {b: (g[label_str][b]["max_dd"] - g["NDX_L"][b]["max_dd"]) * 100 for b in ("G-FULL", "G-LONG")}
        return {"margin": margin, "dd_gap_pp": dd_gap, "R1": all(v > 0 for v in margin.values()),
                "R2": all(v >= -2.0 for v in dd_gap.values())}

    decision = {}
    for label_str in ret_dict["engine"]:
        eng, st = rule("engine", label_str), rule("stress5", label_str)
        cap = run_meta[label_str]["capacity_2021_2026"]
        r5 = bool(cap.get("aum_max_5pct_usd", 0) >= 5_000_000)
        decision[label_str] = {"engine": eng, "stress5": st, "R3": st["R1"] and st["R2"], "R5": r5,
                               "passes": bool(eng["R1"] and eng["R2"] and st["R1"] and st["R2"] and r5)}
    res["decision"] = decision

    # three-leg book and correlation (descriptive)
    eng = pd.DataFrame(ret_dict["engine"])
    both = pd.concat([taa_long_ser.rename("taa"), ndx_ret_dict["engine"].rename("ndx"), eng["W_REL25"].rename("r1k")], axis=1).loc["2008-03-04":an.END_TS].dropna()
    # *** CRITICAL *** daily-rebalanced weights of same-day returns (same convention as G3).
    three_ser = 0.5 * both["taa"] + 0.25 * both["ndx"] + 0.25 * both["r1k"]
    res["three_leg_book"] = {b: an.metric_dict(three_ser.loc[s:e]) for b, (s, e) in an.G3_BLOCK_DICT.items()}
    res["corr_W_REL25_vs_NDX_L_2000_2026"] = float(eng["W_REL25"].loc["2000-01-04":an.END_TS].corr(ndx_ret_dict["engine"].loc["2000-01-04":an.END_TS]))

    # paired bootstrap, G-FULL
    g3_df = an.g3_frame(pd.DataFrame({"W": eng["W_REL25"], "L": ndx_ret_dict["engine"]}), taa_long_ser).loc["2012-10-02":an.END_TS].dropna()
    idx = an.stationary_boot_index(len(g3_df), np.random.default_rng(an.BOOT_SEED_INT), an.BOOT_N_INT)
    x = g3_df.to_numpy()
    diff = np.array([an.sharpe_cols(x[i])[0] - an.sharpe_cols(x[i])[1] for i in idx])
    res["bootstrap_W_REL25_minus_L_g3_full"] = {"obs": float(an.sharpe_cols(x)[0] - an.sharpe_cols(x)[1]),
                                               "p_one_sided": float(np.mean(diff <= 0)),
                                               "ci90": [float(np.quantile(diff, 0.05)), float(np.quantile(diff, 0.95))]}
    (an.OUT_PATH / "r1000_liquid_check.json").write_text(json.dumps(res, indent=1, default=str))

    # print summary
    for label_str in list(ret_dict["engine"]) + ["NDX_L"]:
        s = res["engine"]["standalone"][label_str]
        g = res["engine"]["g3"][label_str]
        m = run_meta.get(label_str, {})
        print(f"{label_str:10s} SA FULL {s['FULL']['cagr']:.3f}/{s['FULL']['sharpe']:.2f}/{s['FULL']['max_dd']:.3f} "
              f"P1-3 {s['P1']['sharpe']:.2f}/{s['P2']['sharpe']:.2f}/{s['P3']['sharpe']:.2f} | G3 {g['G-P1']['sharpe']:.2f}/{g['G-P2']['sharpe']:.2f}/{g['G-P3']['sharpe']:.2f} "
              f"FULL {g['G-FULL']['cagr']:.3f}/{g['G-FULL']['sharpe']:.2f}/{g['G-FULL']['max_dd']:.3f} LONG DD {g['G-LONG']['max_dd']:.3f}"
              + (f" | turn {m['turnover_x_per_year']:.1f} names {m['mean_names_held']:.1f} cap21 max5% {m['capacity_2021_2026']['aum_max_5pct_usd']/1e6:.1f}M p95 1% {m['capacity_2021_2026']['aum_p95_1pct_usd']/1e6:.1f}M"
                 f" max1% full {m['capacity_full']['aum_max_1pct_usd']/1e6:.2f}M" if m else ""))
    print(json.dumps({k: {"passes": v["passes"], "margin": {b: round(x, 3) for b, x in v["engine"]["margin"].items()}, "dd": {b: round(x, 2) for b, x in v["engine"]["dd_gap_pp"].items()}, "R3": v["R3"], "R5": v["R5"]} for k, v in decision.items()}, indent=0))
    print("pool removed", pool_dict)
    print("overlap", {k: round(v.get("overlap_with_unfiltered_share", 1), 2) for k, v in run_meta.items()})
    print("3-leg", {b: (round(v["cagr"], 3), round(v["sharpe"], 2), round(v["max_dd"], 3)) for b, v in res["three_leg_book"].items()})
    print("corr", res["corr_W_REL25_vs_NDX_L_2000_2026"], "boot", res["bootstrap_W_REL25_minus_L_g3_full"])


if __name__ == "__main__":
    main()
