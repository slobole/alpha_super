"""Descriptive data for the combined shelf page (pods, defensive cores, aggressive candidates, client accounts).

Fixed, pre-specified books only; nothing selects. Usage: python shelf_data.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import bb_lib as bb
from bb_lib import EXACT_START, LONG_START, TBILL, ga, lib

OUT = bb.STUDY / "shelf_data.json"
PODS = {"taa3x_1n": "רוטציה טקטית 3x (TAA3x-1N)", "taa3x": "רוטציה טקטית 3x, דירוג (TAA3x)", "ndx_vxn": "מומנטום נאסד״ק 100 (NDX-VXN)",
        "core5": "מאקרו רב־נכסי (CORE5)", "btal_qqq": "גידור ומגמה (BTAL_QQQ)", "etf_dv2": "היפוך לממוצע בקרנות ענפיות (DV2-IND)",
        "eom_flow": "זרימות סוף חודש (EOM)"}
DEF = {"D0": "ליבה הגנתית: CORE5 60 / BTAL_QQQ 40", "D2": "ליבה הגנתית: CORE5 + BTAL_QQQ לפי תנודתיות",
       "D3": "ליבת יעד: CORE5 + BTAL_QQQ + DV2-IND", "D5": "ליבת יעד: ארבעה פודים"}
AGG = {bb.AGGR_V: "אגרסיבי עם כרית", "TAA3x-1N + NDX-VXN 70:30": "אגרסיבי טהור", bb.GROWTH_V: "מתון (צמיחה)", bb.G3: "G3 הקודם"}


def metrics(r: pd.Series, data: dict) -> dict:
    fm = lib.full_metrics(r, data, "long")
    st = ga.window_stats(r.to_frame(), data["index"])
    ex = r.loc[EXACT_START:]
    rec = r.loc["2023-08-21":]
    down = r[r < 0]
    yrs = (1 + r).groupby(r.index.year).prod() - 1
    out = {k.replace("long_", ""): (v if isinstance(v, (str, type(None))) else float(v)) for k, v in fm.items()}
    out.update({"sortino": float(r.mean() / down.std() * np.sqrt(252)), "net_cagr": float(st["net_cagr"][0]),
                "net_dd": float(st["net_dd"][0]),
                "exact_cagr": float(ga.window_stats(ex.to_frame(), data["index"])["gross_cagr"][0]),
                "recent_cagr": float(ga.window_stats(rec.to_frame(), data["index"])["gross_cagr"][0]),
                "pos_years": int((yrs > 0).sum()), "n_years": int(len(yrs)),
                "years": {int(y): float(v) for y, v in yrs.items()},
                "crises": {k: lib.common.window_return_float(r, lo, hi) for k, (lo, hi) in lib.CRISIS_DICT.items()},
                "nav": [[d.strftime("%Y-%m"), round(float(v), 4)] for d, v in (1 + r).cumprod().resample("ME").last().items()]})
    for lo, hi, ret in lib.cofall_windows(data["bench"], LONG_START, bb.END):
        out["crises"][f"cofall|{lo.date()}|{hi.date()}|{ret:.3f}"] = lib.common.window_return_float(
            r, lo.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d"))
    return out


def main() -> int:
    data = lib.load_inputs()
    frame, start = ga.frames(data)["main"]
    sl = {s.name: s for s in bb.risky_sleeves()}
    idx = lib.book_returns(frame, sl[bb.AGGR_V].book(), start).index
    series = {}
    for p in PODS:
        series[f"pod|{p}"] = frame[p].loc[idx]
    for d in DEF:
        series[f"def|{d}"] = lib.book_returns(frame, bb.def_book(d), start)
    for a in AGG:
        series[f"agg|{a}"] = lib.book_returns(frame, sl[a].book(), start)
    per = bb.period_labels(idx, "annual")
    clients = {"acct|A+D0": (bb.AGGR_V, "D0"), "acct|B+D0": ("TAA3x-1N + NDX-VXN 70:30", "D0"),
               "acct|C+D0": (bb.GROWTH_V, "D0"), "acct|A+D3": (bb.AGGR_V, "D3"), "acct|A+D2": (bb.AGGR_V, "D2")}
    for k, (a, d) in clients.items():
        series[k] = pd.Series(bb.account_returns(series[f"agg|{a}"].to_numpy(), series[f"def|{d}"].to_numpy(),
                                                 bb.S_CLIENT, per)[:, 0], index=idx)
    for lab, col in (("bench|S&P 500", "SPXTR"), ("bench|60/40", "SIXTY_FORTY"), ("bench|QQQ", "QQQ")):
        series[lab] = data["bench"][col].loc[idx]
    out = {"books": {k: metrics(v.dropna(), data) for k, v in series.items()}}
    # Bootstrap tail risk (10 seeds) for the accounts, cores and aggressive candidates.
    keys = [k for k in series if not k.startswith("bench")]
    R = np.column_stack([series[k].fillna(0.0).to_numpy() for k in keys])
    acc: dict = {}
    for i, seed in enumerate(bb.SEEDS):
        b = ga.bootstrap_paths(R, bb.boot_index(len(idx), seed))
        acc.setdefault("ddar10", []).append(np.percentile(b["gross_dd"], 10, axis=0))
        for B in (-0.10, -0.15, -0.20, -0.25):
            acc.setdefault(f"p{int(-B * 100)}", []).append((b["gross_dd"] < B).mean(axis=0))
    for j, k in enumerate(keys):
        out["books"][k]["boot"] = {m: float(np.mean([x[j] for x in v])) for m, v in acc.items()}
    # Correlations: all days, and on the S&P 500's worst 5% of days (falls).
    cols = [k for k in series if k.startswith(("pod|", "bench|S&P", "bench|QQQ"))] + ["def|D0", f"agg|{bb.AGGR_V}"]
    M = pd.DataFrame({k: series[k] for k in cols}).dropna()
    spx = data["bench"]["SPXTR"].reindex(M.index)
    worst = spx <= spx.quantile(0.05)
    out["corr"] = {"names": cols, "all": M.corr().round(3).values.tolist(), "falls": M[worst].corr().round(3).values.tolist(),
                   "falls_days": int(worst.sum())}
    # Sleeve-vs-core co-movement for the plan: correlation overall, on S&P worst days, and on the core's worst days.
    a, d = series[f"agg|{bb.AGGR_V}"], series["def|D0"]
    out["pair"] = {"corr_all": float(a.corr(d)), "corr_spx_falls": float(a[worst.reindex(a.index, fill_value=False)].corr(d[worst.reindex(a.index, fill_value=False)])),
                   "agg_on_spx_falls_mean": float(a[worst.reindex(a.index, fill_value=False)].mean()),
                   "def_on_spx_falls_mean": float(d[worst.reindex(a.index, fill_value=False)].mean())}
    out["labels"] = {"pod": PODS, "def": DEF, "agg": AGG}
    OUT.write_text(json.dumps(out, default=str), encoding="utf-8")
    bb.ledger("shelf_data_written")
    for k, v in out["books"].items():
        print(k, {m: round(v[m], 3) for m in ("cagr", "vol", "sharpe", "sortino", "maxdd", "calmar", "cvar5_daily", "cvar5_21d", "crisis_corr", "beta")},
              v.get("boot", {}).get("ddar10"))
    print(out["pair"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
