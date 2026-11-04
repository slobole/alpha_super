"""Independent research replica of the locked rule (weights held between trades, 5 bp all-in) and its review checks:
threshold grid, RSI2 placebo, stress, VX futures 2006-2011, capacity. Saves series to results/research/vrp_rsi2_build_20261103/.

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe scripts/research/vrp_rsi2_build_20261103/research_checks.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.dont_write_bytecode = True
ROOT = Path(r"C:\Users\User\Documents\workspace\alpha_super")
OUT = ROOT / "results" / "research" / "vrp_rsi2_build_20261103"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/research/residual_momentum_20261031"))
import norgatedata as nd
import rm_lib as rm
a, hunt = rm.a, rm.hunt
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)

def px(sym, tr=True):
    adj = nd.StockPriceAdjustmentType.TOTALRETURN if tr else nd.StockPriceAdjustmentType.CAPITALSPECIAL
    return nd.price_timeseries(sym, stock_price_adjustment_setting=adj, padding_setting=nd.PaddingType.ALLMARKETDAYS, start_date="2000-01-01", timeseriesformat="pandas-dataframe")

lab, _, meta = hunt._frames(); fs, fl = pd.Timestamp(meta["start_str"]), pd.Timestamp(meta["end_str"])
L = lab.loc[fs:fl]; rfL = L[meta["tbill_column_str"]].fillna(0.0)
E = pd.Timestamp("2026-10-30"); S = pd.Timestamp("2011-10-10")
d = {s: px(s) for s in ("SPY", "SVXY", "VIXM", "VIXY")}
vixf, v3f, spxf = (px(s, False)["Close"] for s in ("$VIX", "$VIX3M", "$SPX"))
idx = d["SVXY"]["Close"].dropna().index
O = pd.DataFrame({k: d[k]["Open"].reindex(idx) for k in ("SVXY", "VIXM", "VIXY")}); C = pd.DataFrame({k: d[k]["Close"].reindex(idx) for k in ("SVXY", "VIXM", "VIXY")})
rf = a.long_tbill_series(idx)

def rsi2(c, n=2):
    x = c.diff(); u = x.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(); v = (-x).clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + u / v.replace(0, np.nan))).fillna(50)

def weights(ix, c=.98, boost=True, flags=None, long_col="VIXM", use_long=True, lev_cut=True, svxy=2.0):
    v = vixf.reindex(ix).ffill(); t = v3f.reindex(ix).ffill(); rs = rsi2(spxf.reindex(ix).ffill())
    erv = d["SPY"]["Close"].reindex(ix).pct_change().rolling(10).std() * 252 ** .5 * 100; ev = (v - erv).to_numpy(); vv = v.to_numpy(); ts = (v / t).to_numpy()
    lo25 = (rs < 25).to_numpy() if flags is None else flags[0]; lo20 = (rs < 20).to_numpy() if flags is None else flags[1]
    la = np.where((ix < pd.Timestamp("2018-02-28")) & lev_cut, .5, 1.0)
    cont = ts < c; back = ts > 1.0; good = ~np.isnan(ev)
    b1 = np.minimum(svxy * vv / 100, 1); b1b = np.where(boost & lo25, np.minimum(b1 * 1.25, 1), b1)
    b2 = np.minimum(.5 * svxy * vv / 100, 1); b2b = np.where(boost & lo20, np.minimum(.75 * svxy * vv / 100, 1), b2)
    s = np.where(cont & (ev > 0) & good, b1b, np.where(cont & (ev <= 0) & good, b2b, 0.0)) * la
    l = np.where(back & (ev <= 0) & good & use_long, np.minimum(1.25 * vv / 100, 1), 0.0)
    W = pd.DataFrame({"SVXY": s, long_col: l}, index=ix); W.iloc[:25] = 0
    return W

def sim(W, cost=None, lag=1, rf_=None, O_=None, C_=None, band=.05):
    O_ = O if O_ is None else O_; C_ = C if C_ is None else C_; rf_ = rf if rf_ is None else rf_
    cols = list(W.columns); T = W.shift(lag - 1).fillna(0).to_numpy(float); ix = W.index
    co = (O_[cols].reindex(ix) / C_[cols].reindex(ix).shift(1) - 1).fillna(0).to_numpy(float); oc = (C_[cols].reindex(ix) / O_[cols].reindex(ix) - 1).fillna(0).to_numpy(float); f = rf_.reindex(ix).fillna(0).to_numpy(float)
    cv = np.array([(cost or {}).get(c, 5.0) * 1e-4 for c in cols]); out = np.zeros(len(ix)); to = np.zeros(len(ix)); w = np.zeros(len(cols))
    for s in range(2, len(ix)):
        tg = T[s - 1]
        new = tg.copy() if (np.any((tg == 0) != (w == 0)) or np.max(np.abs(tg - w)) >= band) else w.copy()
        dw = np.abs(new - w); to[s] = dw.sum() / 2
        out[s] = (1 + w @ co[s] - dw @ cv) * (1 + new @ oc[s]) - 1 + max(0, 1 - new.sum()) * f[s]; w = new
    return pd.Series(out, index=ix), pd.Series(to, index=ix)

def met(r, f_=None):
    f_ = rf if f_ is None else f_; x = r.dropna(); e = x - f_.reindex(x.index); q = (1 + x).cumprod(); dd = q / q.cummax() - 1
    return dict(cagr=q.iloc[-1] ** (252 / len(x)) - 1, xs=e.mean() / e.std() * 252 ** .5, vol=x.std() * 252 ** .5, dd=dd.min())
def xsw(r, a_, b_, f_=None):
    f_ = rf if f_ is None else f_; e = (r - f_.reindex(r.index)).loc[a_:b_]; return e.mean() / e.std() * 252 ** .5
def tot(r, a_, b_): return (1 + r.loc[a_:b_]).prod() - 1
def blank(r, a_, b_):
    z = r.copy(); z.loc[a_:b_] = rf.reindex(z.loc[a_:b_].index); return z
EL = fl  # report window end for the doc's table

W0 = weights(idx); r0, t0 = sim(W0)
print("== A locked rule, 2011-10-10..2026-08-19"); m = met(r0.loc[S:EL]); print({k: round(v, 3) for k, v in m.items()}, "turn", round(t0.loc[S:EL].mean() * 252, 2))
print("to 2026-10-30", {k: round(v, 3) for k, v in met(r0.loc[S:E]).items()}, "2026 YTD", round(tot(r0, "2026-01-01", E), 3), "T-bills", round((1 + rf.loc["2026-01-01":E]).prod() - 1, 3))
wpost = W0.loc["2018-03-01":E]; print("avg SVXY+VIXM weight, full", round(W0.loc[S:E].sum(axis=1).mean(), 3), "post-2018", round(wpost.sum(axis=1).mean(), 3), "| avg cash full", round(1 - W0.loc[S:E].sum(axis=1).mean(), 3), "post-2018", round(1 - wpost.sum(axis=1).mean(), 3), "| max SVXY", round(wpost.SVXY.max(), 3), "p95", round(wpost.SVXY.quantile(.95), 3))
yr = r0.loc[S:E].groupby(r0.loc[S:E].index.year).apply(lambda x: (1 + x).prod() - 1); yf = rf.loc[S:E].groupby(rf.loc[S:E].index.year).apply(lambda x: (1 + x).prod() - 1); print("excess by year", (yr - yf).round(3).to_dict())

print("\n== B threshold grid with the boost")
rows = {}
for c in (.90, .94, .95, .96, .97, .98, .99, 1.0, 1.01, 1.0164, 1.02, 1.03, 1.05):
    r, t = sim(weights(idx, c=c)); m = met(r.loc[S:EL]); m.update(x18=xsw(r, S, "2018-12-31"), x19=xsw(r, "2019-01-01", EL), x23=xsw(r, "2023-01-01", EL), feb18=tot(r, "2018-01-26", "2018-02-12"), aug24=tot(r, "2024-07-31", "2024-08-08"), apr25=tot(r, "2025-02-19", "2025-04-08"), covid=tot(r, "2020-02-19", "2020-03-23"), turn=t.loc[S:EL].mean() * 252); rows[c] = m
print(pd.DataFrame(rows).T.round(3).to_string())
v = vixf.reindex(idx).ffill(); t_ = v3f.reindex(idx).ffill()
for dt in ("2018-02-01", "2018-02-02", "2024-08-01", "2024-08-02", "2020-02-21", "2020-02-24"):
    print(dt, "VIX", round(v[dt], 2), "VIX3M", round(t_[dt], 2), "ratio", round(v[dt] / t_[dt], 4), "points above .98 line", round(v[dt] - .98 * t_[dt], 2))
print("near-miss days: session after ratio crossed .98 from below where SVXY fell >10% in next two sessions")
ts = v / t_; nxt = C.SVXY.pct_change().shift(-1)

print("\n== C RSI boost vs placebo (circular shift of the low-RSI flags, 400 draws, c=.98)")
rs = rsi2(spxf.reindex(idx).ffill()); l25 = (rs < 25).to_numpy(); l20 = (rs < 20).to_numpy(); rng = np.random.default_rng(20261103)
rb, tb = sim(weights(idx, boost=False)); real = met(r0.loc[S:EL])["xs"]; base = met(rb.loc[S:EL])["xs"]
pl = []
for k in range(400):
    sh = int(rng.integers(30, len(idx) - 30)); r, _ = sim(weights(idx, flags=(np.roll(l25, sh), np.roll(l20, sh)))); pl.append(met(r.loc[S:EL])["xs"])
pl = np.array(pl); print("base", round(base, 3), "boost", round(real, 3), "placebo median", round(np.median(pl), 3), "p5/p95", np.round(np.percentile(pl, [5, 95]), 3), "share of placebos >= boost", round((pl >= real).mean(), 3))
gain = {k: round(xsw(r0, a_, b_) - xsw(rb, a_, b_), 3) for k, (a_, b_) in {"11-18": (S, "2018-12-31"), "19-22": ("2019-01-01", "2022-12-31"), "23+": ("2023-01-01", EL)}.items()}; print("boost Sharpe gain by era", gain)
print("no-boost cagr", round(met(rb.loc[S:EL])["cagr"], 3), "vol", round(met(rb.loc[S:EL])["vol"], 3), "boost vol", round(met(r0.loc[S:EL])["vol"], 3))
for mult in (2.2, 2.4):
    r, t = sim(weights(idx, boost=False, svxy=mult)); print("no boost, SVXY multiple", mult, {k: round(x, 3) for k, x in met(r.loc[S:EL]).items()})

print("\n== D stress on the locked rule")
rows = {}
hi = {"SVXY": 35, "VIXM": 35, "VIXY": 35}
series = {"locked": r0}
for n_, W, kw in [("5 bp", W0, {}), ("10 bp", W0, {"cost": {"SVXY": 10, "VIXM": 10}}), ("15/35 bp", W0, {"cost": hi}), ("one session late", W0, {"lag": 2}), ("no boost", weights(idx, boost=False), {}), ("no long vol", weights(idx, use_long=False), {}), ("VIXY for VIXM", weights(idx, long_col="VIXY"), {})]:
    r, t = sim(W, **kw); series[n_] = r; m = met(r.loc[S:EL]); m.update(x23=xsw(r, "2023-01-01", EL), feb18=tot(r, "2018-01-26", "2018-02-12"), covid=tot(r, "2020-02-19", "2020-03-23"), bear22=tot(r, "2022-01-03", "2022-12-30"), apr25=tot(r, "2025-02-19", "2025-04-08"), turn=t.loc[S:EL].mean() * 252); rows[n_] = m
print(pd.DataFrame(rows).T.round(3).to_string())
for nm, a_, b_ in (("COVID Feb19-Mar23", "2020-02-19", "2020-03-23"), ("COVID Feb19-Apr30 (their window)", "2020-02-19", "2020-04-30"), ("all 2020", "2020-01-01", "2020-12-31"), ("2022", "2022-01-01", "2022-12-31")):
    print(nm, {k: round(x, 3) for k, x in met(blank(r0, a_, b_).loc[S:EL]).items()})
print("worst pod days", {str(k.date()): round(x, 3) for k, x in r0.loc[S:E].nsmallest(5).items()}); e_ = (r0 - rf).loc[S:EL]; print("top 10 days share of excess", round(e_.nlargest(10).sum() / e_.sum(), 3))

print("\n== E VX futures 2006-2011 on the locked rule")
import re
src = open(ROOT / "scripts/research/vrp_multipod_scout_20261101/run_clean_vrp_rebuild_20261102.py", encoding="utf8").read()
parts = re.findall(r'"([^"]+)"', src.split('PAKAL_VX_PATH = (')[1].split(')')[0]); vxp = ROOT.parent.joinpath(*parts); print("vx file", vxp, vxp.exists())
vx = pd.read_csv(vxp, index_col=0, parse_dates=True); print(vx.columns.tolist()[:12], vx.index[0].date(), vx.index[-1].date())
sv = d["SVXY"]["Close"].pct_change(); vi = d["VIXY"]["Close"].pct_change()
chk = pd.concat([vx["vx_st_ret"], vi, vi.shift(-1), vi.shift(1)], axis=1).dropna().loc["2012":"2025"]; chk.columns = ["vx", "vixy_same", "vixy_next", "vixy_prev"]
print("vx_st_ret dated t, correlation with VIXY same day / next / prev:", chk.corr().iloc[0, 1:].round(3).to_dict()); print("around Feb 5 2018:", vx["vx_st_ret"].loc["2018-02-01":"2018-02-07"].round(3).to_dict())
ix = vx.index.intersection(vixf.dropna().index).intersection(v3f.dropna().index); ix = ix[ix >= pd.Timestamp("2006-07-18")]
Wv = weights(ix, lev_cut=False); rfv = a.long_tbill_series(ix).fillna(0)
st = -.5 * vx["vx_st_ret"].reindex(ix).fillna(0).to_numpy() - .0095 / 252; mt = vx["vx_mt_ret"].reindex(ix).fillna(0).to_numpy() - .0085 / 252
def vxsim(lag, cost=10.0):
    T = Wv[["SVXY", "VIXM"]].shift(lag - 1).fillna(0).to_numpy(); w = np.zeros(2); o_ = np.zeros(len(ix)); f = rfv.to_numpy()
    for t in range(2, len(ix)):
        tg = T[t - 1]; new = tg.copy() if (np.any((tg == 0) != (w == 0)) or np.max(np.abs(tg - w)) >= .05) else w.copy()
        o_[t] = new[0] * st[t] + new[1] * mt[t] + max(0, 1 - new.sum()) * f[t] - np.abs(new - w).sum() * cost * 1e-4; w = new
    return pd.Series(o_, index=ix)
rv1, rv2 = vxsim(1), vxsim(2)
rows = {}
for nm, a_, b_ in (("2006-07..2011-10", "2006-07-18", "2011-10-07"), ("2008", "2008-01-01", "2008-12-31"), ("Sep-Nov 2008", "2008-09-01", "2008-11-30"), ("2007-10..2009-03", "2007-10-09", "2009-03-09"), ("May-Jul 2010", "2010-04-23", "2010-07-02"), ("Aug 2011", "2011-07-22", "2011-10-07"), ("2011-10..2026-08", S, EL)):
    rows[nm] = dict(same_close_tot=tot(rv1, a_, b_), one_late_tot=tot(rv2, a_, b_), same_close_xs=xsw(rv1, a_, b_, rfv), one_late_xs=xsw(rv2, a_, b_, rfv), same_close_dd=met(rv1.loc[a_:b_], rfv)["dd"], one_late_dd=met(rv2.loc[a_:b_], rfv)["dd"])
print(pd.DataFrame(rows).T.round(3).to_string())
print("ETF post-2011 for comparison", round(met(r0.loc[S:EL])["xs"], 3), "corr VX proxy vs ETF pod", round(rv1.loc[S:EL].corr(r0.loc[S:EL]), 3))
m08 = (1 + rv1.loc["2008"]).groupby(rv1.loc["2008"].index.month).prod() - 1; print("2008 by month, same close", m08.round(3).to_dict()); print("2008 regime weights by month", Wv.loc["2008"].groupby(Wv.loc["2008"].index.month).mean().round(2).to_dict())

print("\n== F capacity")
adv = {}
for s_ in ("SVXY", "VIXM", "VIXY"):
    q = nd.price_timeseries(s_, stock_price_adjustment_setting=nd.StockPriceAdjustmentType.NONE, start_date="2019-01-01", timeseriesformat="pandas-dataframe"); dv = q["Close"] * q["Volume"]; adv[s_] = dv.iloc[-60:].median(); print(s_, "ADV60 $M", round(adv[s_] / 1e6, 1), "p20", round(dv.iloc[-60:].quantile(.2) / 1e6, 1), "Mar 2020", round(dv.loc["2020-03"].median() / 1e6, 1))
for aum in (5e5, 5e6, 1e7, 2.5e7):
    pod = aum * .1; print(f"fund ${aum/1e6:.1f}M, pod ${pod/1e6:.2f}M: VIXM order at VIX 30 ${pod*.375/1e3:.0f}K = {pod*.375/adv['VIXM']*100:.1f}% ADV; VIXY same {pod*.375/adv['VIXY']*100:.1f}%; SVXY full entry at VIX 20 ${pod*.4/1e3:.0f}K = {pod*.4/adv['SVXY']*100:.1f}%")

ix_lab = L.index
def t11(r):
    z = r.reindex(ix_lab).fillna(rfL); z.loc[:S] = rfL.loc[:S]; return z
frame = pd.DataFrame({"locked": t11(r0), "locked_p5": t11(sim(W0, {"SVXY": 10, "VIXM": 10})[0]), "stress": t11(sim(W0, hi)[0]), "stress_p5": t11(sim(W0, {"SVXY": 40, "VIXM": 40})[0]), "base": t11(rb), "base_p5": t11(sim(weights(idx, boost=False), {"SVXY": 10, "VIXM": 10})[0]),
                      "no_long": t11(series["no long vol"]), "no_long_p5": t11(sim(weights(idx, use_long=False), {"SVXY": 10, "VIXM": 10})[0]), "vixy": t11(series["VIXY for VIXM"]), "vixy_p5": t11(sim(weights(idx, long_col="VIXY"), {"SVXY": 10, "VIXY": 10})[0]),
                      "late": t11(series["one session late"]), "late_p5": t11(sim(W0, {"SVXY": 10, "VIXM": 10}, lag=2)[0]), "nocovid": t11(blank(r0, "2020-02-19", "2020-03-23")), "nocovid_p5": t11(blank(sim(W0, {"SVXY": 10, "VIXM": 10})[0], "2020-02-19", "2020-03-23"))})
for tag, rv in (("vx1", rv1), ("vx2", rv2)):
    z = t11(r0); pre = rv.reindex(ix_lab[ix_lab < S]).fillna(0); z.loc[pre.index] = pre.values; frame[tag] = z; frame[tag + "_p5"] = z
frame.to_parquet(OUT / "locked_series.parquet"); print("saved")
