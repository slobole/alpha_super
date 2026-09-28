"""Locked pre-2003 holdout (SPEC_FROZEN.md Phase 2 + amendments A1/A2). Used once: baseline, benchmarks, finalists.

Primary (A2): 1983-01..2002-12 on proxies throughout (Kenneth French 12 industries for the sectors, $SPXTR/$SPX for
SPY, synthetic 7-10y Treasury for IEF), inflation = Cleveland Fed EXPINF5YR usable from the last session of the
month after its label month. Secondary: 1999-04..2002-12 with the real sector SPDRs (TOTALRETURN) and SPY, same
inflation input and the synthetic bond before IEF exists.
Execution: next close after the decision, 5 bps per side, no withholding (total-return indices).
Rule: growth = SPY > SMA200; inflation = EXP > 2.0 AND (EXP > EXP 60 sessions earlier OR basket slope > 0).
"""

from __future__ import annotations

import itertools
import json
import sys

import numpy as np
import pandas as pd

import common as cm
import holdout_data as hd

TRADE = ["XLE", "XLK", "XLU", "XLP", "IEF"]


def regimes(tr: pd.DataFrame, infl: pd.Series, be=60, slope_n=60, sma=200, threshold=2.0, offset=0):
    idx = pd.DatetimeIndex(tr.index)
    r = tr.pct_change(fill_method=None)
    pos = 0.5 * r["XLE"] + (r["XLI"] + r["XLF"] + r["XLB"]) / 6.0
    neg = (r["XLU"] + r["XLV"] + r["XLP"]) / 3.0
    ratio = (1 + pos).cumprod() / (1 + neg).cumprod()
    slope = cm.rolling_slope(ratio, slope_n)
    ma = tr["SPY"].rolling(sma, min_periods=sma).mean()
    growth = tr["SPY"] > ma
    level = infl.reindex(idx)
    # *** CRITICAL*** the inflation series is already availability-dated; the anchor is its value 60 sessions ago
    prior = level.shift(be)
    me = cm.month_end_index(idx, offset)
    lvl, pri, slo, gro, mav = (s.reindex(me) for s in (level, prior, slope, growth, ma))
    ok = lvl.notna() & pri.notna() & slo.notna() & mav.notna()
    infl_on = (lvl > threshold) & ((lvl > pri) | (slo > 0))
    return pd.DataFrame({"growth": gro.astype(bool), "infl": infl_on.astype(bool)}, index=me)[ok]


def nav_from_weights(w: pd.DataFrame, tr: pd.DataFrame, start, end, capital=100_000.0, scale=None):
    syms = list(w.columns)
    idx = pd.DatetimeIndex(tr.index)
    key = tuple(sorted(syms))
    # inject total-return "bars" (open = close = TR level, no separate dividends) into the replica cache
    cm._BARS[key] = cm.Bars(idx=idx, open=tr[list(key)], close=tr[list(key)],
                            div=pd.DataFrame(0.0, index=idx, columns=list(key)))
    ew = cm.map_to_execution(w, idx, lag=1)
    sc = None if scale is None else cm.map_to_execution(scale.to_frame("s"), idx, lag=1)["s"]
    nav = cm.run_replica(ew, start=start, end=end, fill="close", capital=capital, withholding=0.0, scale_ser=sc)
    del cm._BARS[key]
    return nav


def static_w(idx, wd, start):
    me = cm.month_end_index(pd.DatetimeIndex(idx))
    me = me[me >= pd.Timestamp(start) - pd.Timedelta(days=40)]
    return pd.DataFrame([wd] * len(me), index=me).fillna(0.0)


def trend_w(tr, start):
    ma = tr["SPY"].rolling(200, min_periods=200).mean()
    me = cm.month_end_index(pd.DatetimeIndex(tr.index))
    on = (tr["SPY"] > ma).reindex(me)
    ok = ma.reindex(me).notna() & (me >= pd.Timestamp(start) - pd.Timedelta(days=40))
    return pd.DataFrame([{"SPY": 1.0} if o else {"IEF": 1.0} for o in on[ok]], index=me[ok]).fillna(0.0)


def evaluate(tr, infl, start, end, label, finalists):
    out = {}
    reg = regimes(tr, infl)
    reg = reg[reg.index >= pd.Timestamp(start) - pd.Timedelta(days=40)]
    out["time_in_regime"] = {f"g={g},i={i}": float(((reg.growth == g) & (reg.infl == i)).mean())
                             for g, i in itertools.product([True, False], [True, False])}
    navs = {"BASE literal": nav_from_weights(cm.weights_from_regimes(reg).reindex(columns=TRADE).fillna(0.0), tr, start, end),
            "SPY buy&hold": nav_from_weights(static_w(tr.index, {"SPY": 1.0}, start), tr, start, end),
            "SPY>SMA200 else IEF": nav_from_weights(trend_w(tr, start), tr, start, end),
            "Growth axis only (XLK / XLP+IEF)": None,
            "EW 8 sector proxies": nav_from_weights(static_w(tr.index, {s: 1 / 8 for s in
                                                    ["XLE", "XLK", "XLU", "XLP", "XLI", "XLF", "XLB", "XLV"]}, start),
                                                    tr, start, end)}
    g_only = reg.copy(); g_only["infl"] = False
    navs["Growth axis only (XLK / XLP+IEF)"] = nav_from_weights(
        cm.weights_from_regimes(g_only).reindex(columns=TRADE).fillna(0.0), tr, start, end)
    for name, fn in finalists.items():
        nav = fn(tr, infl, start, end)
        if nav is not None:
            navs[name] = nav
    rows = []
    for k, nav in navs.items():
        m = cm.metrics(nav)
        rows.append({"segment": label, "strategy": k, **{kk: m[kk] for kk in ("start", "end", "cagr", "sharpe", "maxdd")}})
    return rows, out, navs


# ---- finalists (filled from Phase 4; identical definitions, adapted to the proxy data) ----
def f_ensemble(tr, infl, start, end, offset=0):
    frames = []
    for th, be, sl in itertools.product((1.8, 2.0, 2.2), (40, 60, 80), (40, 60, 80)):
        reg = regimes(tr, infl, be=be, slope_n=sl, threshold=th, offset=offset)
        frames.append(cm.weights_from_regimes(reg).reindex(columns=TRADE).fillna(0.0))
    ci = frames[0].index
    for f in frames[1:]:
        ci = ci.intersection(f.index)
    return sum(f.loc[ci] for f in frames) / len(frames)


def f_tranche(weight_fn):
    def run(tr, infl, start, end):
        navs = [nav_from_weights(weight_fn(tr, infl, off), tr, start, end, capital=25_000.0)
                for off in (0, -5, -10, -15)]
        return pd.concat(navs, axis=1).dropna().sum(axis=1)
    return run


def f_qqq(tr, infl, start, end):
    """C2: QQQ replaces XLK. Only expressible where real QQQ exists (secondary segment)."""
    if "QQQ" not in tr.columns:
        return None
    reg = regimes(tr, infl)
    cmap = {**cm.REGIME_WEIGHTS, (True, False): {"QQQ": 1.0}}
    w = cm.weights_from_regimes(reg, cmap).reindex(columns=["XLE", "QQQ", "XLU", "XLP", "IEF"]).fillna(0.0)
    return nav_from_weights(w, tr, start, end)


FINALISTS = {
    "C2 QQQ for XLK": f_qqq,
    "C1 ensemble 27": lambda tr, infl, s, e: nav_from_weights(f_ensemble(tr, infl, s, e), tr, s, e),
    "C4 tranching x4": f_tranche(lambda tr, infl, off: cm.weights_from_regimes(regimes(tr, infl, offset=off))
                                 .reindex(columns=TRADE).fillna(0.0)),
    "C7 ensemble + tranching": f_tranche(lambda tr, infl, off: f_ensemble(tr, infl, None, None, offset=off)),
}


def main():
    which = sys.argv[1:] or list(FINALISTS)
    fin = {k: FINALISTS[k] for k in which if k in FINALISTS}
    inp = hd.holdout_daily_inputs("1982-01-01", "2002-12-31")
    tr_p, infl = inp["tr_index"], inp["infl_available"]
    rows, meta, navs = evaluate(tr_p, infl, "1983-01-03", "2002-12-31", "primary 1983-2002 proxies", fin)
    for k, (a, b) in {"1983-1990": ("1983-01-03", "1990-12-31"), "1991-1998": ("1991-01-02", "1998-12-31"),
                      "1999-2002": ("1999-01-04", "2002-12-31")}.items():
        for name, nav in navs.items():
            m = cm.metrics(nav, a, b)
            rows.append({"segment": f"primary sub {k}", "strategy": name,
                         **{kk: m[kk] for kk in ("start", "end", "cagr", "sharpe", "maxdd")}})
    # secondary: real ETFs 1999-2002
    etf = cm.signal_close_df(["SPY", "XLE", "XLK", "XLU", "XLP", "XLI", "XLF", "XLB", "XLV", "IEF", "QQQ"])
    etf = etf.loc["1998-01-01":"2002-12-31"]
    bond = tr_p["IEF"].reindex(etf.index).ffill()
    ief = etf["IEF"].copy()
    first = ief.first_valid_index()
    spliced = bond.copy()
    spliced.loc[first:] = ief.loc[first:] / ief.loc[first] * bond.loc[first]
    etf["IEF"] = spliced
    etf = etf.dropna(subset=["SPY"])
    r2, _, _ = evaluate(etf.dropna(), infl, "1999-06-01", "2002-12-31", "secondary 1999-06..2002 real ETFs", fin)
    rows += r2
    t = pd.DataFrame(rows)
    t.to_csv(cm.OUT / "p6_holdout.csv", index=False)
    print(t.round(3).to_string(index=False))
    meta["gate_never_off_note"] = "EXPINF5YR > 2.0 in all 240 months 1983-2002 (min 2.014): the level gate never binds"
    (cm.OUT / "p6_holdout_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
