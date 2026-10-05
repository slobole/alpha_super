"""Timing lens 6: battery items by independent arithmetic - decay, common shock size, gate alignment, drifted exposure, DSR units, after-window."""
import json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import g_lib as g
lib = g.lib
lab = g.Lab(); frame, rf = lab.frame, lab.rf
LONG, END, EX = g.LONG_START, g.END, g.EXACT_START
bat = json.loads((g.OUT / "battery.json").read_text())
# ---- decay all at 0.75 by hand
def shifted(fr, shift):
    out = fr.copy()
    for a, s in shift.items():
        out[a] = fr[a] - s
    return out
pods = ["taa3x", "taa3x_1n", "ndx_atr_cap", "ndx_natr_cap", "dv2_g", "hpi_g", "ndx_vxn", "core5", "btal_qqq"]
win = frame.loc[LONG:END]
mu = {a: float((win[a] - win["tbill"]).mean()) for a in pods}
print("pod mean excess (annualised x252):", {a: round(v * 252, 4) for a, v in mu.items()})
for k in (0.75, 0.5):
    fr = shifted(frame, {a: (1 - k) * mu[a] for a in pods})
    for n in ("GR1", "GR2", "GR3"):
        s = g.stats(g.book_returns(fr, g.PRODUCTS[n], LONG), rf)
        ref = bat["edge_decay"]["scenarios"][f"all at {k}"][n]
        print(f"  k={k} {n}: cagr {s['cagr']:.4f} xs {s['xs']:.3f} | battery {ref['cagr']:.4f} {ref['xs']:.3f}")
# ---- common shock: regression beta vs time-average exposure
qqq = frame["qqq_tr"]; qmu = float((qqq - rf).loc[LONG:END].mean())
print("\nQQQ mean daily excess x252:", round(qmu * 252, 4), " half premium:", round(0.5 * qmu * 252, 4))
beta = bat["edge_decay"]["beta_qqq"]
tq = pickle.load(open(g.STUDY / "audit/review/timing_lens/tqqq.pkl", "rb"))["cs_close"]
def tqw(a):
    tx = lab.data["tx"][a]; nav = lab.data["nav"][a]
    sh = tx[tx.asset_str == "TQQQ"].groupby("date")["amount_float"].sum().reindex(nav.index).fillna(0).cumsum()
    return (sh * tq.reindex(nav.index) / nav).loc[EX:END]
inv = lambda p: (p["portfolio_value_float"] / p["total_value_float"])
expo = {"taa3x": 3 * tqw("taa3x"), "taa3x_1n": 3 * tqw("taa3x_1n"),
        "ndx_atr_cap": inv(lab.data["path"]["ndx_atr_cap"]).loc[LONG:END], "ndx_natr_cap": inv(lab.data["path"]["ndx_natr_cap"]).loc[LONG:END],
        "dv2_g": inv(lab.data["path"]["dv2_g_cash"]).loc[LONG:END], "hpi_g": inv(lab.data["path"]["hpi_g_cash"]).loc[LONG:END]}
# daily-beta too
wk = lambda s: (1 + s).resample("W-FRI").prod(min_count=1) - 1
print("pod: weekly beta (battery) | daily beta | mean look-through equity exposure (prior close) | shock used x252 | exposure-based shock x252")
alt = {}
for a in expo:
    x = (frame[a] - rf).loc[LONG:END]; q = (qqq - rf).loc[LONG:END]
    bd = float(np.cov(x, q)[0, 1] / q.var())
    e = float(expo[a].shift(1).mean())
    alt[a] = e * 0.5 * qmu
    print(f"  {a:13s} {beta[a]:.3f} | {bd:.3f} | {e:.3f} | {beta[a] * 0.5 * qmu * 252:.4f} | {alt[a] * 252:.4f}")
for n in ("GR1", "GR2", "GR3"):
    used = g.stats(g.book_returns(shifted(frame, {a: beta[a] * 0.5 * qmu for a in beta}), g.PRODUCTS[n], LONG), rf)
    al = g.stats(g.book_returns(shifted(frame, alt), g.PRODUCTS[n], LONG), rf)
    base = g.stats(g.book_returns(frame, g.PRODUCTS[n], LONG), rf)
    R = np.column_stack([g.book_returns(shifted(frame, alt), g.PRODUCTS[n], LONG).to_numpy()])
    hard = g.RUNGS[g.TARGET_RUNG[n]][1]
    tm = lab.tail_matrix(R, limits=(hard,))
    ref = bat["edge_decay"]["scenarios"]["common shock (half the QQQ premium)"][n]
    print(f"  {n}: base cagr {base['cagr']:.4f} xs {base['xs']:.3f} | regression-beta shock cagr {used['cagr']:.4f} xs {used['xs']:.3f} (battery {ref['cagr']:.4f} {ref['xs']:.3f}) | exposure-based shock cagr {al['cagr']:.4f} xs {al['xs']:.3f} breach at {hard}: {float(tm[hard].mean()):.4f}")
# ---- gate alignment
import norgatedata
vix = norgatedata.price_timeseries("$VIX", start_date="1990-01-01", end_date=END.strftime("%Y-%m-%d"), timeseriesformat="pandas-dataframe")["Close"]
vix.index = pd.to_datetime(vix.index).normalize()
idx = frame.loc[LONG:END].index
print("\nVIX rows", len(vix), "sessions in LONG missing from VIX:", int((~idx.isin(vix.index)).sum()), "VIX dates in LONG window not sessions:", int((~vix.loc[LONG:END].index.isin(idx)).sum()), "NaN", int(vix.isna().sum()))
gate = g.vix_gate.stress_gate_open_ser(vix)
stock = 0.5 * inv(lab.data["path"]["dv2_g_cash"]) + 0.5 * inv(lab.data["path"]["hpi_g_cash"])
tx = lab.data["tx"]["dv2_g"]; buys = tx[(tx.asset_str != "BIL") & (tx.amount_float > 0) & (tx.date >= LONG) & (tx.date <= END)]
for sh in (0, 1, 2):
    gp = gate.shift(sh).reindex(idx)
    bshare = float(gp.reindex(buys.date).to_numpy().astype(float).mean())
    print(f"  shift({sh}): gate open share {float((gp == True).mean()):.4f}; MR stock weight open {float(stock.reindex(idx)[gp == True].mean()):.3f} closed {float(stock.reindex(idx)[gp == False].mean()):.3f}; share of dv2_g stock BUY fills on gate-open sessions {bshare:.4f}")
# ---- drifted exposure (EXACT era): prior-close pod weights of the LONG book instead of target weights
exp = json.loads((g.OUT / "exposure.json").read_text())
ixe = lab.data["nav"]["taa3x"].loc[EX:END].index
for n in ("GR1", "GR2", "GR3"):
    wp = []; g.book_returns(frame, g.PRODUCTS[n], LONG, weight_path=wp); ix, cols, pw = wp[0]
    pw = pd.DataFrame(pw, index=ix, columns=cols).shift(-1).reindex(ixe)       # weights at the close of t (= prior-close weights of t+1)
    w = g.PRODUCTS[n]
    taa = "taa3x" if "taa3x" in w else "taa3x_1n"
    mom = 0.5 * expo["ndx_atr_cap"].reindex(ixe) + 0.5 * expo["ndx_natr_cap"].reindex(ixe)
    tgt = w[taa] * expo[taa].reindex(ixe) + (w["ndx_atr_cap"] + w["ndx_natr_cap"]) * mom + (w["dv2_g"] * expo["dv2_g"].reindex(ixe) + w["hpi_g"] * expo["hpi_g"].reindex(ixe))
    dr = pw[taa] * expo[taa].reindex(ixe) + pw["ndx_atr_cap"] * expo["ndx_atr_cap"].reindex(ixe) + pw["ndx_natr_cap"] * expo["ndx_natr_cap"].reindex(ixe) + pw["dv2_g"] * expo["dv2_g"].reindex(ixe) + pw["hpi_g"] * expo["hpi_g"].reindex(ixe)
    e = exp["books"][n]["equity_exposure"]
    print(f"{n}: equity exposure target-weights mean {tgt.mean():.3f} p90 {tgt.quantile(.9):.3f} max {tgt.max():.3f} (exposure.json {e['mean']:.3f} {e['p90']:.3f} {e['max']:.3f}) | drifted weights mean {dr.mean():.3f} p90 {dr.quantile(.9):.3f} max {dr.max():.3f} on {dr.idxmax().date()} (TAA share then {pw.loc[dr.idxmax(), taa]:.3f})")
# worst realised one-day loss vs gap table
for n in ("GR1", "GR2", "GR3"):
    r = lab.ret(g.PRODUCTS[n]); q = qqq.reindex(r.index)
    d = q.idxmin(); 
    print(n, "worst QQQ day", d.date(), f"QQQ {q.loc[d]:.4f} book {r.loc[d]:.4f}; worst book day {r.idxmin().date()} {r.min():.4f} (QQQ {q.loc[r.idxmin()]:.4f})")
# ---- int(-L*100) keys
print("\nint(-L*100) for limits:", {L: int(-L * 100) for L in (-0.10, -0.15, -0.17, -0.20, -0.22, -0.25, -0.27, -0.30, -0.35)})
