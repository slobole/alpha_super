"""Timing lens 9: (a) old frame columns unchanged vs a fresh lib.load_inputs; (b) holdings-based QQQ exposure of the TAA pods vs regression beta;
(c) common shock with holdings-based exposure; (d) proxy-era transactions."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import g_lib as g
import norgatedata
lib, ga = g.lib, g.ga
lab = g.Lab(); frame, rf = lab.frame, lab.rf; data = lab.data
LONG, END, EX = g.LONG_START, g.END, g.EXACT_START
bat = json.loads((g.OUT / "battery.json").read_text())
# (a)
fresh = lib.load_inputs(); ff = ga.frames(fresh)
for k in ("main", "s1_house_cash", "s2_proxy_unscaled", "s3_plus_5bps", "s6_exact"):
    a, b = lab.frames[k][0], ff[k][0]
    cols = list(b.columns)
    same = bool(((a[cols] - b[cols]).abs().fillna(0).to_numpy().max() == 0) and (a[cols].isna() == b[cols].isna()).all().all())
    print(f"old columns of frame {k}: {len(cols)} columns identical to a fresh lib.load_inputs -> {same}; extra columns {sorted(set(a.columns) - set(cols))}")
# (b) holdings of the TAA pods, EXACT era
def cs_close(sym):
    p = norgatedata.price_timeseries(sym, stock_price_adjustment_setting=norgatedata.StockPriceAdjustmentType.CAPITALSPECIAL, padding_setting=norgatedata.PaddingType.NONE,
                                     start_date="2010-01-01", end_date="2026-10-02", timeseriesformat="pandas-dataframe")["Close"]
    p.index = pd.to_datetime(p.index).normalize(); return p
def tr_ret(sym):
    return lib.common.load_total_return_close_ser(sym, "2010-01-01", END.strftime("%Y-%m-%d")).pct_change(fill_method=None)
qqq = frame["qqq_tr"]
hb = {}
for a in ("taa3x", "taa3x_1n"):
    tx = data["tx"][a]; nav = data["nav"][a]
    sh = tx.pivot_table(index="date", columns="asset_str", values="amount_float", aggfunc="sum").reindex(nav.index).fillna(0).cumsum()
    w = pd.DataFrame({s: sh[s] * cs_close(s).reindex(nav.index) / nav for s in sh.columns}).loc[EX:END]
    betas = {}
    for s in w.columns:
        r = tr_ret(s).loc[EX:END]; q = qqq.reindex(r.index)
        ok = r.notna() & q.notna()
        betas[s] = float(np.cov(r[ok], q[ok])[0, 1] / q[ok].var())
    hbeta = sum(w[s].shift(1) * betas[s] for s in w.columns)
    hb[a] = hbeta
    x = (frame[a] - rf).loc[EX:END]; q = (qqq - rf).loc[EX:END]
    wk = lambda s: (1 + s).resample("W-FRI").prod(min_count=1) - 1
    xw, qw = wk(x).iloc[1:-1], wk(q).iloc[1:-1]
    print(f"\n{a} (2012-10-02..END): mean weights {w.mean().round(3).to_dict()}")
    print(f"   asset betas to QQQ (daily, total return) {dict((k, round(v, 2)) for k, v in betas.items())}")
    print(f"   time-average holdings beta {hbeta.mean():.3f} (TQQQ part {(w['TQQQ'].shift(1) * betas['TQQQ']).mean():.3f}) | regression beta of the pod on the same window: daily {np.cov(x, q)[0, 1] / q.var():.3f}, weekly {np.cov(xw, qw.reindex(xw.index))[0, 1] / qw.var():.3f} | battery LONG weekly beta {bat['edge_decay']['beta_qqq'][a]:.3f}")
    # premium earned through QQQ exposure: mean of (holdings beta x QQQ excess) vs regression-beta x mean QQQ excess
    print(f"   mean daily (holdings beta_t-1 x QQQ excess_t) x252 = {float((hbeta * q).mean() * 252):.4f}; regression beta x mean QQQ excess x252 = {float(np.cov(x, q)[0, 1] / q.var() * q.mean() * 252):.4f}; time-average beta x mean QQQ excess x252 = {float(hbeta.mean() * q.mean() * 252):.4f}")
# (c) common shock with holdings-based exposure, LONG window (EXACT-era mean applied to the whole window for the TAA pods; MOM and MR at stock weight, beta 1)
inv = lambda p: (p["portfolio_value_float"] / p["total_value_float"])
qmu = float((qqq - rf).loc[LONG:END].mean())
e = {"taa3x": float(hb["taa3x"].mean()), "taa3x_1n": float(hb["taa3x_1n"].mean()),
     "ndx_atr_cap": float(inv(data["path"]["ndx_atr_cap"]).loc[LONG:END].mean()), "ndx_natr_cap": float(inv(data["path"]["ndx_natr_cap"]).loc[LONG:END].mean()),
     "dv2_g": float(inv(data["path"]["dv2_g_cash"]).loc[LONG:END].mean()), "hpi_g": float(inv(data["path"]["hpi_g_cash"]).loc[LONG:END].mean())}
beta = bat["edge_decay"]["beta_qqq"]
print("\npod: regression beta used | time-average exposure:", {a: (round(beta[a], 3), round(e[a], 3)) for a in e})
def shifted(shift):
    out = frame.copy()
    for a, s in shift.items():
        out[a] = frame[a] - s
    return out
for n in ("GR1", "GR2", "GR3"):
    w = g.PRODUCTS[n]
    hard = g.RUNGS[g.TARGET_RUNG[n]][1]
    res = {}
    for lab_, sh in (("regression beta (battery)", {a: beta[a] * 0.5 * qmu for a in e}), ("time-average exposure", {a: e[a] * 0.5 * qmu for a in e})):
        r = g.book_returns(shifted(sh), w, LONG); s = g.stats(r, rf)
        tm = lab.tail_matrix(np.column_stack([r.to_numpy()]), limits=(hard,))
        res[lab_] = (round(s["cagr"], 4), round(s["xs"], 3), round(float(tm[hard].mean()), 4))
    wb = sum(w[a] * beta[a] for a in w); we = sum(w[a] * e[a] for a in w)
    print(f"{n}: book-weighted beta {wb:.3f} vs exposure {we:.3f} (ratio {we / wb:.2f}); (CAGR, excess Sharpe, breach at {hard:.0%}) {res}")
# (d) proxy-era transactions
ptx = lib.read_tx(lib.PROXY / "splice_scaled", "taa3x")
print("\nproxy tx assets:", ptx.asset_str.value_counts().head(12).to_dict(), ptx.date.min().date(), ptx.date.max().date())
