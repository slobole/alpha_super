"""Timing lens 4: S7 walk-forward inverse vol: weights strictly before each reset; degenerate years; effect."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import g_lib as g, study as st
lib = g.lib
lab = g.Lab()
log = st.register_s7(lab)
r7 = lab.r(g.S7)
caps = {"TAA": {"taa3x": 1.0}, "MOM": g.MOM, "MR": g.MR}
cf = pd.DataFrame({k: g.book_returns(lab.frame, g.blend((1.0, w)), lab.start) for k, w in caps.items()})
yrs = sorted(set(cf.index.year))
print("year  n_hist  trailing-252 ann vol TAA / MOM / MR   -> my IV weights | logged weights")
for (d, n, ww) in log:
    hist = cf.loc[:d].iloc[:-1].iloc[-252:]
    if len(hist) < 60:
        print(d.date(), len(hist), "equal-weight fallback", {k: round(v, 4) for k, v in ww.items()}); continue
    sd = hist.std(); iv = (1 / sd) / (1 / sd).sum()
    print(d.date(), len(hist), (sd * 252 ** .5).round(4).to_dict(), iv.round(4).to_dict(), "| max abs diff vs log", float(max(abs(iv[k] - ww[k]) for k in ww)), "| last hist date", hist.index[-1].date())
# what does the capsule hold when vol is ~0?
for alias in ("ndx_atr_cap", "ndx_natr_cap"):
    p = lab.data["path"][alias]
    inv = (p["portfolio_value_float"] / p["total_value_float"]).loc["2008-03-04":"2008-12-31"]
    print(alias, "2008 invested weight: mean", round(float(inv.mean()), 3), "share of days < 1%", round(float((inv < 0.01).mean()), 3))
tx = lab.data["tx"]["dv2_g"]
for alias in ("dv2_g", "hpi_g"):
    cp = lab.data["path"][alias + "_cash"]
    inv = (cp["portfolio_value_float"] / cp["total_value_float"]).loc["2017-01-01":"2017-12-31"]
    print(alias, "2017 stock weight (parking-off run): mean", round(float(inv.mean()), 3), "share of days < 1%", round(float((inv < 0.01).mean()), 3))
q7 = lab.cands[g.S7]["q"]
print("S7 MAIN: cagr", round(q7["cagr"], 4), "xs", round(q7["xs"], 3), "dd", round(q7["dd"], 4), "p20", lab.tails(g.S7)["p20"])
# S7 with weights floored: each capsule between 10% and 60% (illustration of what the degenerate years do)
def clipped(frame_key="main"):
    fr, s = lab.frames[frame_key]
    out, value = [], 1.0
    idx = cf.index
    per = idx.year.to_numpy(); b = np.r_[np.flatnonzero(np.r_[True, per[1:] != per[:-1]]), len(idx)]
    for a, e in zip(b[:-1], b[1:]):
        hist = cf.iloc[:a].iloc[-252:]
        if len(hist) < 60: w = np.full(3, 1 / 3)
        else:
            sd = hist.std().to_numpy(); sd = np.maximum(sd, 0.05 / 252 ** .5)   # vol floor 5% a year
            w = (1 / sd) / (1 / sd).sum()
        gr = np.cumprod(1 + cf.iloc[a:e].to_numpy(), axis=0) @ w
        lev = value * gr; out += list(lev / np.r_[value, lev[:-1]] - 1); value = lev[-1]
    return pd.Series(out, index=idx)
rc = clipped()
print("S7 with a 5% vol floor: ", {k: round(v, 4) for k, v in g.stats(rc, lab.rf).items()})
print("GR1:", {k: round(v, 4) for k, v in g.stats(lab.ret(g.PRODUCTS['GR1']), lab.rf).items()})
for y in (2009, 2018):
    print(y, "returns  S7", round(float((1 + r7[r7.index.year == y]).prod() - 1), 4), "GR1", round(float((1 + lab.ret(g.PRODUCTS['GR1'])[r7.index.year == y]).prod() - 1), 4),
          "capsules", {k: round(float((1 + cf.loc[cf.index.year == y, k]).prod() - 1), 4) for k in cf})
study = json.loads((g.OUT / "study.json").read_text())
print("construction", json.dumps(study["construction"], indent=0)[:1500])
for ch in study["challenges"]:
    if ch["challenger"].startswith("S7"):
        print("S7 challenge", {k: ch[k] for k in ("share_xs", "share_cagr", "xs", "cagr", "breach", "exact_xs", "plus5_xs", "halves_xs", "checks", "passed")})
