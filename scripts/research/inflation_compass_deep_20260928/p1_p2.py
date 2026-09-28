"""Phase 1 (reconcile with the article) and Phase 2 (is there an edge) - see SPEC_FROZEN.md."""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

import common as cm

SEC9 = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
CELLS = [(True, True), (True, False), (False, True), (False, False)]
HOLDINGS = [{"XLE": 1.0}, {"XLK": 1.0}, {"XLU": 1.0}, {"XLP": 0.5, "IEF": 0.5}]


def run_w(w, fill="open", lag=1, **kw):
    return cm.run_replica(cm.map_to_execution(w, cm.sessions(), lag=lag), fill=fill, **kw)


def monthly_stats(nav):
    m = nav.resample("ME").last()
    r = m.pct_change().dropna()
    yrs = (m.index[-1] - m.index[0]).days / 365.25
    return {"cagr_monthly": ((m.iloc[-1] / m.iloc[0]) ** (1 / yrs) - 1) * 100,
            "sharpe_monthly": r.mean() / r.std(ddof=1) * np.sqrt(12),
            "maxdd_monthly": (m / m.cummax() - 1).min() * 100}


def trend_weights(sig, sym, alt, sma=200, idx=None):
    px = sig[sym]
    ma = px.rolling(sma, min_periods=sma).mean()
    me = cm.month_end_index(pd.DatetimeIndex(sig.index))
    on = (px > ma).reindex(me)
    ok = ma.reindex(me).notna()
    rows = [{sym: 1.0} if o else alt for o in on[ok]]
    return pd.DataFrame(rows, index=me[ok]).fillna(0.0)


def static_weights(sig, wdict):
    me = cm.month_end_index(pd.DatetimeIndex(sig.index))
    me = me[me >= pd.Timestamp("2003-01-01")]
    return pd.DataFrame([wdict] * len(me), index=me).fillna(0.0)


def main():
    sig = cm.signal_close_df()
    t5 = cm.fred_series("T5YIE")
    t5 = t5[t5.index <= pd.Timestamp(cm.MAIN_END_STR)]
    res = {}

    # ---------------- Phase 1: reconcile ----------------
    w_pub, reg = cm.compass_weights(sig, t5)
    w_leak, reg_leak = cm.compass_weights(sig, t5, t5_same_date=True)
    steps = [
        ("S0 ours: next open, published T5YIE, 5bps, 25% WHT", w_pub, dict()),
        ("S1 no dividend withholding", w_pub, dict(withholding=0.0)),
        ("S2 + no costs", w_pub, dict(withholding=0.0, slippage=0.0)),
        ("S3 + same-date T5YIE (old leak)", w_leak, dict(withholding=0.0, slippage=0.0)),
        ("S4 + fill at the same month-end close (article)", w_leak, dict(withholding=0.0, slippage=0.0,
                                                                          fill="close", lag=0)),
        ("S4b same-close fill with published T5YIE", w_pub, dict(withholding=0.0, slippage=0.0, fill="close",
                                                                  lag=0)),
    ]
    rec = []
    for name, w, kw in steps:
        fill = kw.pop("fill", "open")
        lag = kw.pop("lag", 1)
        for end in (cm.MAIN_END_STR, "2026-06-30"):
            nav = run_w(w, fill=fill, lag=lag, end=end, **kw)
            m = cm.metrics(nav)
            m.update(monthly_stats(nav))
            m.update({"step": name, "window_end": end})
            rec.append(m)
    p1 = pd.DataFrame(rec)
    p1.to_csv(cm.OUT / "p1_reconcile.csv", index=False)
    print(p1[["step", "window_end", "cagr", "sharpe", "maxdd", "cagr_monthly", "sharpe_monthly", "maxdd_monthly"]].round(3)
          .to_string(index=False))

    # ---------------- Phase 2: benchmarks & decomposition ----------------
    base_nav = run_w(w_pub)
    base_r = base_nav.pct_change().dropna()
    tim = reg.loc[w_pub.index]
    tim = tim[(tim.index >= pd.Timestamp("2003-04-01"))]
    share = {c: float(((tim["growth"] == c[0]) & (tim["infl"] == c[1])).mean()) for c in CELLS}
    static = {}
    for c, h in zip(CELLS, HOLDINGS):
        for k, v in h.items():
            static[k] = static.get(k, 0.0) + share[c] * v
    res["time_in_regime"] = {f"growth={c[0]},infl={c[1]}": share[c] for c in CELLS}
    res["static_mix"] = static

    bench = {
        "Compass (fixed)": w_pub,
        "SPY buy&hold": static_weights(sig, {"SPY": 1.0}),
        "QQQ buy&hold": static_weights(sig, {"QQQ": 1.0}),
        "XLK buy&hold": static_weights(sig, {"XLK": 1.0}),
        "EW 9 sectors": static_weights(sig, {s: 1 / 9 for s in SEC9}),
        "60/40 SPY/IEF": static_weights(sig, {"SPY": 0.6, "IEF": 0.4}),
        "SPY>SMA200 else IEF": trend_weights(sig, "SPY", {"IEF": 1.0}),
        "XLK>SMA200 else IEF": trend_weights(sig, "XLK", {"IEF": 1.0}),
        "QQQ>SMA200 else IEF": trend_weights(sig, "QQQ", {"IEF": 1.0}),
        "SPY>SMA200: XLK else XLP/IEF (growth axis only)": None,
        "Static mix (time-in-regime weights)": static_weights(sig, static),
    }
    # growth-only: inflation forced off; inflation-only: growth forced on
    g_only = reg.copy(); g_only["infl"] = False
    i_only = reg.copy(); i_only["growth"] = True
    bench["SPY>SMA200: XLK else XLP/IEF (growth axis only)"] = cm.weights_from_regimes(g_only)
    bench["Inflation axis only (growth forced on: XLE/XLK)"] = cm.weights_from_regimes(i_only)
    # inflation definition ablations (Varadi's 'remove T5YIE' test and its mirror)
    lvl_on = reg["level"] > 2.0
    abl1 = reg.copy(); abl1["infl"] = reg["slope"] > 0
    abl2 = reg.copy(); abl2["infl"] = lvl_on & (reg["level"] > reg["prior"])
    abl3 = reg.copy(); abl3["infl"] = lvl_on
    bench["Inflation = sector slope only (no T5YIE)"] = cm.weights_from_regimes(abl1)
    bench["Inflation = T5YIE only (level & 60d change)"] = cm.weights_from_regimes(abl2)
    bench["Inflation = T5YIE > 2% only"] = cm.weights_from_regimes(abl3)
    rows, navs = [], {}
    for name, w in bench.items():
        nav = run_w(w)
        navs[name] = nav
        pt = cm.period_table(nav)
        row = {"strategy": name}
        for per, m in pt.items():
            row[f"{per}_cagr"] = m["cagr"]; row[f"{per}_sharpe"] = m["sharpe"]
        row["maxdd"] = pt["ALL"]["maxdd"]; row["vol"] = pt["ALL"]["vol"]
        rows.append(row)
    p2b = pd.DataFrame(rows)
    p2b.to_csv(cm.OUT / "p2_benchmarks.csv", index=False)
    print(p2b.round(2).to_string(index=False))

    # map permutations (24)
    perm_rows = []
    for perm in itertools.permutations(range(4)):
        cmap = {CELLS[i]: HOLDINGS[perm[i]] for i in range(4)}
        nav = run_w(cm.weights_from_regimes(reg, cmap))
        m = cm.metrics(nav)
        perm_rows.append({"map": " | ".join(
            f"{'G+' if c[0] else 'G-'}{'I+' if c[1] else 'I-'}:{'+'.join(cmap[c])}" for c in CELLS),
            "literal": perm == (0, 1, 2, 3), "cagr": m["cagr"], "sharpe": m["sharpe"], "maxdd": m["maxdd"]})
    pp = pd.DataFrame(perm_rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    pp.to_csv(cm.OUT / "p2_map_permutations.csv", index=False)
    lit_rank = int(pp.index[pp["literal"]][0]) + 1
    res["map_permutation_rank_of_literal_by_sharpe"] = f"{lit_rank} of 24"
    print(pp.round(3).head(8).to_string())

    # placebo inflation axis
    rng = np.random.default_rng(20260928)
    base_sh = cm.metrics(base_nav)["sharpe"]
    infl = reg["infl"].to_numpy()
    n = len(infl)
    circ = []
    for k in range(12, n - 12):
        r2 = reg.copy(); r2["infl"] = np.roll(infl, k)
        circ.append(cm.metrics(run_w(cm.weights_from_regimes(r2)))["sharpe"])
    # spell shuffle
    spells, cur, ln = [], infl[0], 0
    for v in infl:
        if v == cur:
            ln += 1
        else:
            spells.append((cur, ln)); cur, ln = v, 1
    spells.append((cur, ln))
    on_sp = [l for v, l in spells if v]
    off_sp = [l for v, l in spells if not v]
    shuf = []
    for _ in range(1000):
        a, b = list(rng.permutation(on_sp)), list(rng.permutation(off_sp))
        state = bool(rng.random() < len(on_sp) / (len(on_sp) + len(off_sp)))
        seq = []
        while a or b:
            src = a if state else b
            if src:
                seq += [state] * int(src.pop())
            state = not state
        seq = np.array(seq[:n] + [False] * max(0, n - len(seq)))
        r2 = reg.copy(); r2["infl"] = seq
        shuf.append(cm.metrics(run_w(cm.weights_from_regimes(r2)))["sharpe"])
    res["placebo"] = {
        "actual_sharpe": base_sh,
        "circular_shift_n": len(circ), "circular_p": float(np.mean(np.array(circ) >= base_sh)),
        "circular_median": float(np.median(circ)), "circular_p95": float(np.percentile(circ, 95)),
        "spell_shuffle_n": len(shuf), "spell_shuffle_p": float(np.mean(np.array(shuf) >= base_sh)),
        "spell_shuffle_median": float(np.median(shuf)), "spell_shuffle_p95": float(np.percentile(shuf, 95)),
        "growth_only_sharpe": cm.metrics(navs["SPY>SMA200: XLK else XLP/IEF (growth axis only)"])["sharpe"],
    }
    pd.DataFrame({"circular": pd.Series(circ), "spell_shuffle": pd.Series(shuf)}).to_csv(
        cm.OUT / "p2_placebo_sharpes.csv", index=False)
    print(json.dumps(res["placebo"], indent=1))

    # forward-return test: next-month XLE-XLK TR spread on inflation_on, growth-up months
    tr = sig[["XLE", "XLK", "XLU", "XLP", "IEF"]]
    me_px = tr.reindex(reg.index)
    fwd = me_px.shift(-1) / me_px - 1
    df = pd.DataFrame({"spread": fwd["XLE"] - fwd["XLK"], "infl": reg["infl"].astype(float),
                       "growth": reg["growth"]}).dropna()
    fr = {}
    for per, (a, b) in {"ALL": ("2003-01-01", cm.MAIN_END_STR), **cm.PERIODS}.items():
        d = df[(df.index >= pd.Timestamp(a)) & (df.index <= pd.Timestamp(b)) & df["growth"]]
        mdl = sm.OLS(d["spread"], sm.add_constant(d["infl"])).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        fr[per] = {"n": int(len(d)), "coef_pct_per_month": float(mdl.params["infl"] * 100),
                   "t_hac": float(mdl.tvalues["infl"]), "share_on": float(d["infl"].mean())}
    # growth-down cell: XLU vs 50/50 XLP+IEF
    dd = pd.DataFrame({"spread": fwd["XLU"] - 0.5 * (fwd["XLP"] + fwd["IEF"]), "infl": reg["infl"].astype(float),
                       "growth": reg["growth"]}).dropna()
    d = dd[~dd["growth"]]
    mdl = sm.OLS(d["spread"], sm.add_constant(d["infl"])).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    fr["growth_down_XLU_minus_XLP_IEF"] = {"n": int(len(d)), "coef_pct_per_month": float(mdl.params["infl"] * 100),
                                           "t_hac": float(mdl.tvalues["infl"]), "share_on": float(d["infl"].mean())}
    res["forward_return_test"] = fr
    print(json.dumps(fr, indent=1))

    # timing vs static exposure: monthly regressions
    def mret(nav):
        return nav.resample("ME").last().pct_change()

    y = mret(base_nav)
    X = pd.concat({"static": mret(navs["Static mix (time-in-regime weights)"]),
                   "growth_only": mret(navs["SPY>SMA200: XLK else XLP/IEF (growth axis only)"]),
                   "spy": mret(navs["SPY buy&hold"]), "xlk": mret(navs["XLK buy&hold"])}, axis=1)
    regs = {}
    for cols in (["static"], ["spy"], ["xlk"], ["growth_only"], ["spy", "xlk"]):
        d = pd.concat([y.rename("y"), X[cols]], axis=1).dropna()
        mdl = sm.OLS(d["y"], sm.add_constant(d[cols])).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        regs["+".join(cols)] = {"alpha_ann_pct": float(mdl.params["const"] * 12 * 100),
                                "t_alpha": float(mdl.tvalues["const"]),
                                "betas": {c: float(mdl.params[c]) for c in cols}, "r2": float(mdl.rsquared)}
    res["regressions_monthly"] = regs
    print(json.dumps(regs, indent=1))
    (cm.OUT / "p1_p2_results.json").write_text(json.dumps(res, indent=1, default=str))
    pd.DataFrame({k: v for k, v in navs.items()}).to_pickle(cm.OUT / "p2_navs.pkl")


if __name__ == "__main__":
    main()
