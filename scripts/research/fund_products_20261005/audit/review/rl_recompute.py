"""Reviewer scratch (report lens): independent recomputation of numbers quoted in the report prose.
Read-only on the study: uses g_lib.load_inputs / frames / book_returns / stats only (no Lab.tails, no ledger)."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve()
STUDY_DIR = HERE.parents[2]
sys.path.insert(0, str(STUDY_DIR))
import g_lib as g
from g_lib import END, LONG_START, EXACT_START, TBILL, lib

OUT = g.STUDY / "audit" / "review" / "report_lens"
data = g.load_inputs()
F = g.frames(data)
main, start = F["main"]
rf = main[TBILL]
res = {}

def st(w, key="main", frame=None):
    fr, s = F[key]
    r = g.book_returns(fr if frame is None else frame, w, s)
    return r, g.stats(r, rf)

books = {"GR1": g.PRODUCTS["GR1"], "GR2": g.PRODUCTS["GR2"], "GR3": g.PRODUCTS["GR3"], "S9": g.INCUMBENT}
print("frames:", list(F))
for n, w in books.items():
    for key in ("main", "s3_plus_5bps", "s1_house_cash", "s6_exact"):
        r, s = st(w, key)
        res[f"{n}|{key}"] = s
        print(n, key, {k: round(v, 4) for k, v in s.items()})

# planning / floor: k on the +5 bps frame, every capsule (incl. DEF)
f5 = F["s3_plus_5bps"][0]
win = f5.loc[LONG_START:END]
for label, k in (("planning", 0.75), ("floor", 0.5)):
    fr = f5.copy()
    for alias in g.CAPSULE_OF:
        if alias in fr.columns:
            fr[alias] = f5[alias] - (1 - k) * float((win[alias] - win[TBILL]).mean())
    for n, w in books.items():
        r, s = st(w, "s3_plus_5bps", frame=fr)
        res[f"{n}|{label}"] = s
        print(label, n, {k_: round(v, 4) for k_, v in s.items()})

# GR1 vs S9 by sub-block and breakeven extra cost on excess Sharpe
r1, _ = st(books["GR1"]); r9, _ = st(books["S9"])
for lab_, lo, hi in (("A 2008-03..2012-10-01", LONG_START, pd.Timestamp("2012-10-01")), ("B 2012-10-02..2021", EXACT_START, pd.Timestamp("2021-12-31")),
                     ("C 2022..END", pd.Timestamp("2022-01-01"), END), ("RECENT 2023-08..END", pd.Timestamp("2023-08-01"), END)):
    a, b = g.stats(r1.loc[lo:hi], rf), g.stats(r9.loc[lo:hi], rf)
    print("block", lab_, "GR1 cagr/xs/dd", round(a["cagr"], 4), round(a["xs"], 3), round(a["dd"], 4), "| S9", round(b["cagr"], 4), round(b["xs"], 3), round(b["dd"], 4))
    res[f"block|{lab_}"] = {"GR1": a, "S9": b}

# +10 bps frame (linear extrapolation as in A6): main - 2 x (main - plus5)
def plus_n(w, n_bps):
    fr0, s0 = F["main"]; fr5 = F["s3_plus_5bps"][0]
    fr = fr0.copy()
    for c in w:
        fr[c] = fr0[c] - (n_bps / 5.0) * (fr0[c] - fr5[c])
    return g.stats(g.book_returns(fr, w, s0), rf)
for nb in (0, 4, 5, 8, 10):
    a, b = plus_n(books["GR1"], nb), plus_n(books["S9"], nb)
    print(f"+{nb} bps: GR1 cagr {a['cagr']:.4f} xs {a['xs']:.3f} | S9 cagr {b['cagr']:.4f} xs {b['xs']:.3f} | xs gap {a['xs']-b['xs']:+.3f} cagr gap {a['cagr']-b['cagr']:+.4f}")
    res[f"plus{nb}"] = {"GR1": a, "S9": b}

# capsule series (house frame, whole run incl. after END) for the hard-coded caveats
full = data["full"]
mom_full = 0.5 * full["ndx_atr_cap"] + 0.5 * full["ndx_natr_cap"]     # daily-rebalanced 50/50 approximation
nav = (1 + mom_full.dropna()).cumprod()
for cut in (END, nav.index[-1]):
    n_ = nav.loc[:cut]
    pk = n_.idxmax()
    print("MOM capsule (house, 50/50 daily) at", cut.date(), "drawdown from peak", round(float(n_.iloc[-1] / n_.max() - 1), 4), "peak date", pk.date())
    res[f"mom_dd_at_{cut.date()}"] = {"dd": float(n_.iloc[-1] / n_.max() - 1), "peak": str(pk.date())}
# book-model version on LONG (annual reset), MAIN frame
rm, sm = st(g.blend((1.0, g.MOM)))
nv = (1 + rm).cumprod()
print("MOM capsule (MAIN, book model) at END: dd from peak", round(float(nv.iloc[-1] / nv.max() - 1), 4), "peak", nv.idxmax().date(), "stats", {k: round(v, 4) for k, v in sm.items()})
res["mom_dd_main_END"] = {"dd": float(nv.iloc[-1] / nv.max() - 1), "peak": str(nv.idxmax().date())}

# MR capsule: share of profit from 2020 + 2021
rr, sr = st(g.blend((1.0, g.MR)))
yr = (1 + rr).groupby(rr.index.year).prod() - 1
lg = np.log1p(rr).groupby(rr.index.year).sum()
print("MR capsule MAIN stats", {k: round(v, 4) for k, v in sr.items()})
print("MR yearly returns", {int(y): round(float(v), 3) for y, v in yr.items()})
print("MR share of total log growth from 2020+2021:", round(float((lg.loc[2020] + lg.loc[2021]) / lg.sum()), 3))
xsl = np.log1p(rr) - np.log1p(rf.reindex(rr.index)); xl = xsl.groupby(xsl.index.year).sum()
print("MR share of excess log growth from 2020+2021:", round(float((xl.loc[2020] + xl.loc[2021]) / xl.sum()), 3))
res["mr_share_2020_2021"] = {"log_total": float((lg.loc[2020] + lg.loc[2021]) / lg.sum()), "log_excess": float((xl.loc[2020] + xl.loc[2021]) / xl.sum())}

# cash convention sizes (fair cash minus house cash), LONG, per sleeve
house = F["s1_house_cash"][0]
for a in ("taa3x", "taa3x_1n", "ndx_atr_cap", "ndx_natr_cap", "dv2_g", "hpi_g", "ndx_vxn", "core5", "btal_qqq"):
    cm = g.stats(main[a].loc[LONG_START:END], rf)["cagr"]; ch = g.stats(house[a].loc[LONG_START:END], rf)["cagr"]
    l3 = slice(pd.Timestamp("2023-08-20"), END)
    cm3 = g.stats(main[a].loc[l3], rf)["cagr"]; ch3 = g.stats(house[a].loc[l3], rf)["cagr"]
    print(f"cash gap {a}: LONG {100*(cm-ch):+.3f} pp, last 3y {100*(cm3-ch3):+.3f} pp")
    res[f"cashgap|{a}"] = {"long_pp": 100 * (cm - ch), "last3y_pp": 100 * (cm3 - ch3)}

# MR capsule: BIL-held (MAIN) vs parking-off fair cash (s9) -> the "0.6 pp" conservatism
for key in ("main", "s9_mr_fair_cash", "s10_mr_cash_0", "s1_house_cash", "s3_plus_5bps"):
    r_, s_ = st(g.blend((1.0, g.MR)), key)
    print("MR capsule", key, round(s_["cagr"], 4), round(s_["xs"], 3))
    res[f"mrcap|{key}"] = s_

# capsule-level check of the $1M MR runs against the stored $100K capsule (audit/rerun_mr/mr_capsule_daily.csv)
try:
    old = pd.read_csv(g.STUDY / "audit" / "rerun_mr" / "mr_capsule_daily.csv", index_col=0, parse_dates=True)
    print("stored capsule csv columns:", list(old.columns)[:12])
except Exception as e:
    print("stored capsule csv:", e)
(OUT / "recompute.json").write_text(json.dumps(g.r6(res), indent=1), encoding="utf-8")
