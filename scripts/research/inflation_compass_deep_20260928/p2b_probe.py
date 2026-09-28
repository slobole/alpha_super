"""Amendment A1 (EXPLORATORY, post-hoc): is T5YIE an oil/commodity momentum proxy? + timing-luck average NAV."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common as cm


def run(w):
    return cm.run_replica(cm.map_to_execution(w, cm.sessions(), lag=1))


def main():
    sig = cm.signal_close_df()
    t5 = cm.fred_series("T5YIE")
    t5 = t5[t5.index <= pd.Timestamp(cm.MAIN_END_STR)]
    p = cm.Params()
    f = cm.daily_features(sig, t5, p)
    base_w, reg = cm.compass_weights(sig, t5, p)
    out = {}

    # (i) correlations of 60-session changes, sampled at month-ends (non-overlapping-ish: monthly obs)
    me = reg.index
    d_t5 = (f.level - f.prior).reindex(me)
    r60 = sig.pct_change(60, fill_method=None).reindex(me)
    rel = (r60["XLE"] - r60["XLK"])
    corr = pd.DataFrame({"dT5YIE_60": d_t5, "DBC_60": r60["DBC"], "XLE_60": r60["XLE"], "XLE_minus_XLK_60": rel,
                         "sector_slope": reg["slope"]}).dropna()
    out["monthly_corr_with_dT5YIE"] = corr.corr()["dT5YIE_60"].round(3).to_dict()
    out["corr_sample"] = f"{corr.index[0].date()}..{corr.index[-1].date()} n={len(corr)}"

    # (ii) swap the T5YIE 60-session change for commodity / relative momentum, keep the 2% level gate
    lvl_on = reg["level"] > p.threshold
    variants = {}
    v = reg.copy(); v["infl"] = lvl_on & ((r60["DBC"].reindex(me) > 0) | (reg["slope"] > 0))
    variants["level & (DBC_60>0 or slope>0)"] = v[v.index >= "2006-06-30"]
    v = reg.copy(); v["infl"] = lvl_on & ((rel.reindex(me) > 0) | (reg["slope"] > 0))
    variants["level & (XLE-XLK_60>0 or slope>0)"] = v
    v = reg.copy()
    variants["BASE (level & (dT5YIE>0 or slope>0))"] = v
    rows = []
    navs = {}
    for name, r in variants.items():
        nav = run(cm.weights_from_regimes(r))
        navs[name] = nav
        rows.append({"variant": name, **{k: v for k, v in cm.metrics(nav).items() if k in ("start", "cagr", "sharpe", "maxdd")}})
    # same window as the DBC variant for the base
    b06 = run(cm.weights_from_regimes(reg[reg.index >= "2006-06-30"]))
    rows.append({"variant": "BASE from 2006-07", **{k: v for k, v in cm.metrics(b06).items() if k in ("start", "cagr", "sharpe", "maxdd")}})
    out["swap_variants"] = rows
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # timing-luck: average NAV over offsets -10..+10 (each offset a separate $100k book, summed)
    offs = []
    for off in range(-10, 11):
        w, _ = cm.compass_weights(sig, t5, p, offset=off)
        offs.append(run(w).rename(off))
    df = pd.concat(offs, axis=1).dropna()
    avg = df.sum(axis=1)
    m_avg = cm.period_table(avg)
    out["offset_average_book"] = {k: {kk: vv for kk, vv in v.items() if kk in ("start", "cagr", "sharpe", "maxdd")}
                                  for k, v in m_avg.items()}
    base_same = cm.period_table(df[0])
    out["offset0_same_window"] = {k: {kk: vv for kk, vv in v.items() if kk in ("start", "cagr", "sharpe", "maxdd")}
                                  for k, v in base_same.items()}
    print(json.dumps(out, indent=1, default=str))
    (cm.OUT / "p2b_probe.json").write_text(json.dumps(out, indent=1, default=str))
    avg.to_frame("offset_avg_nav").to_pickle(cm.OUT / "p2b_offset_avg_nav.pkl")


if __name__ == "__main__":
    main()
