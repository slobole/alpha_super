"""Mean-reversion shorts study (SPEC_SHORTS.md): short pods SH1-SH4 and the capsule tests."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "fund_menu_20260923"))
import common  # noqa: E402
import replica as rp  # noqa: E402

OUT = HERE.parents[2] / "results/research/dv2_deep_20260925/shorts"
BASE = rp.Rule(floor=True, side="short")
CANDS = {
    "SH1_mirror": BASE,
    "SH2_mirror_adv": BASE.with_(rank="adv"),
    "SH3_no_trend": BASE.with_(short_trend="none"),
    "SH4_uptrend": BASE.with_(short_trend="up"),
}
PERIODS = {"P1": ("2004-01-01", "2014-12-31"), "P2": ("2015-01-01", "2020-12-31"), "P3": ("2021-01-01", "2026-08-19")}


def st(x: pd.Series) -> dict:
    x = x.dropna()
    nav = (1 + x).cumprod()
    yrs = (x.index[-1] - x.index[0]).days / 365.25
    dd = (nav / nav.cummax() - 1).min()
    c = nav.iloc[-1] ** (1 / yrs) - 1
    return {"cagr": c, "sharpe": x.mean() / x.std() * np.sqrt(252), "maxdd": dd, "calmar": c / abs(dd)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p = rp.Panel("sp500")
    sleeve = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "sleeve_returns.csv.gz", index_col=0, parse_dates=True).loc[:"2026-08-19"]
    bench = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "benchmark_returns.csv.gz", index_col=0, parse_dates=True)
    spx = bench["SPXTR"].reindex(sleeve.index).fillna(0.0)
    tb = bench["TBILL"].reindex(sleeve.index).fillna(0.0)
    pods, rep = {}, {"pods": {}, "capsule": {}}
    for name, rule in CANDS.items():
        for bps in (50, 300):
            res = rp.run(p, rule.with_(borrow_bps_yr=bps), start="2000-01-03")
            r = pd.Series(res.nav, index=res.dates).pct_change()
            s = rp.summarize(res)
            key = f"{name}_borrow{bps}"
            rep["pods"][key] = {k: s.get(k) for k in ("cagr", "sharpe", "maxdd", "turnover_x", "entries_per_year", "exposure_days",
                                                       "P1_sharpe", "P1_cagr", "P2_sharpe", "P2_cagr", "P3_sharpe", "P3_cagr")}
            rep["pods"][key]["corr_spx"] = float(r.corr(spx.reindex(r.index)))
            rep["pods"][key]["viable"] = bool(s["cagr"] > 0 and s["sharpe"] > 0.3 and sum(s[f"P{i}_cagr"] > 0 for i in (1, 2, 3)) >= 2)
            if bps == 50:
                pods[name] = r.reindex(sleeve.index)
            print(key, round(s["cagr"], 3), round(s["sharpe"], 3), round(s["maxdd"], 3), flush=True)
    df = pd.DataFrame({"dv2": sleeve["dv2"], "hpi_vote": sleeve["hpi_vote"], "tbill": tb, **pods}).loc["2004-01-05":]
    long_cap = 0.5 * df["dv2"].fillna(0) + 0.5 * df["hpi_vote"].fillna(0)
    # *** CRITICAL*** beta for the hedge uses the trailing 252 sessions, lagged one day (causal)
    beta = long_cap.rolling(252).cov(spx.reindex(df.index)).shift(1) / spx.reindex(df.index).rolling(252).var().shift(1)
    hedged = long_cap - beta.fillna(0.7) * (spx.reindex(df.index) - df["tbill"])
    books = {"L (DV2 50 / HPI 50)": long_cap, "L 80 + cash 20": 0.8 * long_cap + 0.2 * df["tbill"],
             "L beta-hedged (S&P futures)": hedged}
    for name in CANDS:
        books[f"L 80 + {name} 20"] = 0.8 * long_cap + 0.2 * df[name].fillna(0)
    for b, x in books.items():
        row = {"full": st(x)}
        for k, (a0, a1) in PERIODS.items():
            row[k] = st(x.loc[a0:a1])
        row["corr_spx"] = float(x.corr(spx.reindex(x.index)))
        rep["capsule"][b] = row
    base = rep["capsule"]["L 80 + cash 20"]
    for name in CANDS:
        r = rep["capsule"][f"L 80 + {name} 20"]
        rep["capsule"][f"L 80 + {name} 20"]["useful"] = bool(r["full"]["sharpe"] > base["full"]["sharpe"] and r["full"]["maxdd"] > base["full"]["maxdd"]
                                                             and r["P3"]["sharpe"] > base["P3"]["sharpe"] and r["P3"]["maxdd"] > base["P3"]["maxdd"])
    (OUT / "shorts_study.json").write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rep["pods"]).T.round(3).to_string())
    rows = []
    for b, r in rep["capsule"].items():
        rows.append({"book": b, **{f"{k}_{m}": r[k][m] for k in ("full", "P1", "P2", "P3") for m in ("cagr", "sharpe", "maxdd")}, "corr_spx": r["corr_spx"], "useful": r.get("useful")})
    print(pd.DataFrame(rows).set_index("book").round(3).to_string())


if __name__ == "__main__":
    main()
