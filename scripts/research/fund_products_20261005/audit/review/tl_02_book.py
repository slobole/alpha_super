"""Timing lens 2: g_lib.book_returns by hand (annual reset, prior-close weights, reset cost, debt pod, period ids)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import g_lib as g
lib = g.lib
# tiny synthetic: 2 pods + debt, 6 sessions across a year boundary
idx = pd.to_datetime(["2020-12-29", "2020-12-30", "2020-12-31", "2021-01-04", "2021-01-05", "2021-01-06"])
fr = pd.DataFrame({"a": [0.10, -0.05, 0.02, 0.03, -0.10, 0.04], "b": [0.00, 0.01, 0.00, -0.02, 0.01, 0.00], "debt": [0.001] * 6}, index=idx)
w = {"a": 0.6, "b": 0.4}
r = g.book_returns(fr, w, idx[0], end=idx[-1])
# by hand
A = 0.6 * np.cumprod(1 + fr["a"].iloc[:3]); B = 0.4 * np.cumprod(1 + fr["b"].iloc[:3]); V1 = (A + B).to_numpy()
A2 = 0.6 * V1[-1] * np.cumprod(1 + fr["a"].iloc[3:]); B2 = 0.4 * V1[-1] * np.cumprod(1 + fr["b"].iloc[3:]); V2 = (A2 + B2).to_numpy()
V = np.r_[1.0, V1, V2]; hand = V[1:] / V[:-1] - 1
print("fixed-weight annual: max diff vs hand", float(np.abs(r.to_numpy() - hand).max()))
print("  first session of 2021 uses target weights:", np.isclose(r.iloc[3], 0.6 * 0.03 + 0.4 * -0.02))
wp = []
g.book_returns(fr, w, idx[0], end=idx[-1], weight_path=wp)
print("  prior-close weights\n", pd.DataFrame(wp[0][2], index=idx, columns=wp[0][1]).round(5))
print("  contribution identity (sum prior_w x r == book):", float(np.abs((wp[0][2] * fr[["a", "b"]].to_numpy()).sum(axis=1) - r.to_numpy()).max()))
# reset cost
rc = g.book_returns(fr, w, idx[0], end=idx[-1], reset_cost=0.00025)
drift = np.array([A.iloc[-1], B.iloc[-1]]) / V1[-1]
moved = np.abs(drift - np.array([0.6, 0.4])).sum()
print("reset cost: drift", drift.round(5), "sum|drift-target|", round(moved, 6), "one-way capital moved", round(moved / 2, 6))
print("  reset-day return diff (free - charged)", float(r.iloc[3] - rc.iloc[3]), "expected 2.5bp x sum|.| x (1+r)", 0.00025 * moved * (1 + r.iloc[3]))
# levered
L = 1.5
wl = g.levered(w, L)
rl = g.book_returns(fr, wl, idx[0], end=idx[-1])
E = 1.0; out = []
for yr in (slice(0, 3), slice(3, 6)):
    pods = L * E * np.cumprod(1 + (0.6 * np.cumprod(1 + fr["a"].iloc[yr]) + 0.4 * np.cumprod(1 + fr["b"].iloc[yr])).pct_change().fillna(0.6 * fr["a"].iloc[yr].iloc[0] + 0.4 * fr["b"].iloc[yr].iloc[0]))
    debt = (L - 1) * E * np.cumprod(1 + fr["debt"].iloc[yr])
    eq = (pods - debt).to_numpy()
    out += list(eq); E = eq[-1]
Vl = np.r_[1.0, out]
print("levered x1.5 annual reset: max diff vs hand (equity = L x pods - (L-1) x debt)", float(np.abs(rl.to_numpy() - (Vl[1:] / Vl[:-1] - 1)).max()))
wp = []
g.book_returns(fr, wl, idx[0], end=idx[-1], weight_path=wp)
pw = pd.DataFrame(wp[0][2], index=idx, columns=wp[0][1])
print("  gross leverage path (1 - debt weight):", (1 - pw["debt"]).round(4).tolist())
# lib parity on the real main frame with random weights and with lib period ids
lab = g.Lab()
rng = np.random.default_rng(0)
for _ in range(3):
    cols = ["taa3x", "ndx_atr_cap", "dv2_g", "hpi_g", "tbill"]
    x = rng.random(len(cols)); x /= x.sum(); ww = dict(zip(cols, x))
    a = g.book_returns(lab.frame, ww, lab.start); b = lib.book_returns(lab.frame, g.Book("x", tuple(ww), "EQ", ww), lab.start)
    c = lib.common.book_return_ser(lab.frame.loc[lab.start:g.END, cols], ww, "annual")[0]
    print("real frame parity g vs lib", float((a - b).abs().max()), "vs common.book_return_ser", float((a - c).abs().max()))
# period ids
ix = lab.frame.loc[lab.start:g.END].index
for pol in ("annual", "annual-2", "annual-3", "annual-7", "annual-12", "quarterly", "monthly"):
    p = g.period_ids(ix, pol)
    b = ix[np.flatnonzero(np.r_[True, p[1:] != p[:-1]])]
    print(pol, "resets:", len(b) - 1, "first three boundaries", [str(x.date()) for x in b[:4]])
