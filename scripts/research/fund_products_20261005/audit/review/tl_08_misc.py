"""Timing lens 8: MR-only breakeven, after-window recompute, gap-table beta assumptions, planning column, DSR, reverse challenge."""
import json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import g_lib as g
lib = g.lib
lab = g.Lab(); frame, rf = lab.frame, lab.rf; data = lab.data
LONG, END, EX = g.LONG_START, g.END, g.EXACT_START
study = json.loads((g.OUT / "study.json").read_text()); bat = json.loads((g.OUT / "battery.json").read_text())
GR1, GR1L = g.PRODUCTS["GR1"], g.SLOT_TESTS[g.GR1_L]
# (a) breakeven with cost on the MR pods only, all fills and stock fills only
base_l = g.stats(g.book_returns(frame, GR1L, LONG), rf)
def gr1_with_mr_cost(bps, ex_bil):
    fr = frame.copy()
    for a in ("dv2_g", "hpi_g"):
        d = data["drag_ex_bil"][a] if ex_bil else data["drag"][a]
        fr[a] = frame[a] - d * (bps / 5.0)
    return g.stats(g.book_returns(fr, GR1, LONG), rf)
for ex_bil in (False, True):
    grid = np.arange(0, 60.5, 0.5)
    xs = np.array([gr1_with_mr_cost(b, ex_bil)["xs"] for b in grid]); cg = np.array([gr1_with_mr_cost(b, ex_bil)["cagr"] for b in grid])
    be_xs = grid[np.argmax(xs < base_l["xs"])]; be_cg = grid[np.argmax(cg < base_l["cagr"])] if (cg < base_l["cagr"]).any() else None
    print(f"MR-only extra cost, {'stock fills only' if ex_bil else 'all MR fills incl. BIL'}: GR1 = GR1-L on excess Sharpe at ~{be_xs} bps/side, on CAGR at ~{be_cg} bps/side (study: 12.1 / 24.8 with cost on every pod)")
    print("    xs at 0/4/5/10:", [round(float(xs[grid == b][0]), 3) for b in (0, 4, 5, 10)], "GR1-L xs", round(base_l["xs"], 3))
# (b) after window
rer = g.STUDY / "audit" / "rerun_taa_def"
late = {}
for a in ("taa3x", "taa3x_1n", "core5", "btal_qqq", "ndx_vxn"):
    p = lib.read_path(rer, a); r = lib.nav_to_returns(p); late[a] = r
    st = data["sleeve"][a].dropna()
    common = st.index.intersection(r.index)
    print(f"rerun {a}: ends {p.index[-1].date()}, max |rerun - stored house return| on stored dates {float((r.reindex(common) - st.reindex(common)).abs().max()):.2e} over {len(common)} sessions")
for a in g.NEW_ALIAS_LIST:
    late[a] = data["full"][a]
bil = lib.common.load_total_return_close_ser("BIL", "2025-06-01", "2026-10-02"); late["tbill"] = bil.pct_change(fill_method=None)
lf = pd.DataFrame(late).loc["2026-01-02":"2026-10-02"]
print("after-window frame", lf.index[0].date(), lf.index[-1].date(), "NaN", int(lf.isna().sum().sum()), "sessions after END", int((lf.index > END).sum()))
for n, w in (("GR1", GR1), ("GR2", g.PRODUCTS["GR2"]), ("GR3", g.PRODUCTS["GR3"]), ("S9 incumbent launch", g.INCUMBENT)):
    x = lf[list(w)].to_numpy(); pods = np.array([w[k] for k in w]) * np.cumprod(1 + x, axis=0); lev = pods.sum(axis=1)
    i0 = int(np.flatnonzero(lf.index >= "2026-08-20")[0])
    print(f"  {n}: after-window {lev[-1] / lev[i0 - 1] - 1:.6f} (battery {bat['after_window']['books'][n]['after_window']:.6f}); ytd {lev[-1] - 1:.6f} (battery {bat['after_window']['books'][n]['ytd_2026']:.6f})")
# (c) gap-table beta assumptions: realised return per unit of look-through exposure on big QQQ down days (EXACT era)
tq = pickle.load(open(g.STUDY / "audit/review/timing_lens/tqqq.pkl", "rb"))["cs_close"]
qqq = frame["qqq_tr"]
inv = lambda p: (p["portfolio_value_float"] / p["total_value_float"])
house = lab.frames["s1_house_cash"][0]
def tqw(a):
    tx = data["tx"][a]; nav = data["nav"][a]
    sh = tx[tx.asset_str == "TQQQ"].groupby("date")["amount_float"].sum().reindex(nav.index).fillna(0).cumsum()
    return (sh * tq.reindex(nav.index) / nav)
rows = {"taa3x (3 x TQQQ weight)": (house["taa3x"], 3 * tqw("taa3x")), "taa3x_1n (3 x TQQQ weight)": (house["taa3x_1n"], 3 * tqw("taa3x_1n")),
        "MOM capsule (invested weight, beta 1)": (0.5 * house["ndx_atr_cap"] + 0.5 * house["ndx_natr_cap"], 0.5 * inv(data["path"]["ndx_atr_cap"]) + 0.5 * inv(data["path"]["ndx_natr_cap"])),
        "MR capsule (stock weight, beta 1)": (0.5 * data["mr_cash_run"]["dv2_g_cash"] + 0.5 * data["mr_cash_run"]["hpi_g_cash"], 0.5 * inv(data["path"]["dv2_g_cash"]) + 0.5 * inv(data["path"]["hpi_g_cash"]))}
for thr in (-0.02, -0.03):
    print(f"days with QQQ <= {thr:.0%} (2012-10-02..END):")
    for name, (r, e) in rows.items():
        df = pd.DataFrame({"r": r, "e": e.shift(1), "q": qqq}).loc[EX:END].dropna()
        df = df[(df.q <= thr) & (df.e > 0.05)]
        model = df.e * df.q
        b = float((df.r * model).sum() / (model ** 2).sum())
        print(f"   {name:40s} n={len(df):3d}  realised / assumed loss ratio (OLS through origin on exposure x QQQ move): {b:.2f};  sum realised {df.r.sum():+.3f} vs sum assumed {model.sum():+.3f}")
# (d) planning column check and DSR
f5 = lab.frames["s3_plus_5bps"][0]
win = f5.loc[LONG:END]
for k, lab_ in ((0.75, "planning"), (0.5, "floor")):
    fr = f5.copy()
    for a in g.CAPSULE_OF:
        fr[a] = f5[a] - (1 - k) * float((win[a] - win["tbill"]).mean())
    s = g.stats(g.book_returns(fr, GR1, LONG), rf); ref = bat["edge_decay"]["headline"]["GR1"][lab_]
    print(f"{lab_} GR1: cagr {s['cagr']:.4f} xs {s['xs']:.3f} dd {s['dd']:.4f} | battery {ref['cagr']:.4f} {ref['xs']:.3f} {ref['dd']:.4f}")
from scipy import stats as ss, integrate
x = (lab.ret(GR1) - rf.reindex(lab.ret(GR1).index)).to_numpy(); T = len(x); sr = x.mean() / x.std(ddof=1)
for N in (100, 1000):
    emax = integrate.quad(lambda z: z * N * ss.norm.pdf(z) * ss.norm.cdf(z) ** (N - 1), -12, 12, limit=200)[0]
    bench = np.sqrt(1 / (T - 1)) * emax
    sig = np.sqrt(1 - ss.skew(x) * sr + (ss.kurtosis(x, fisher=False) - 1) / 4 * sr ** 2)
    print(f"DSR N={N}: benchmark annual Sharpe {bench * 252 ** .5:.3f}; DSR {ss.norm.cdf((sr - bench) * np.sqrt(T - 1) / sig):.4f} | battery {bat['believe']['GR1'][f'dsr_N{N}']}")
rev = next(c for c in study["challenges_reverse"] if c["default"] == g.S9)
print("reverse GR1 vs S9:", {k: rev[k] for k in ("share_xs", "share_cagr", "gap_xs_p5_50_95", "breach", "exact_xs", "plus5_xs", "halves_xs", "checks", "passed")})
o2 = study["gr2_vs_old_plus"]; print("GR2 vs old plus:", {k: o2[k] for k in ("share_xs", "share_cagr", "breach", "breach_key", "rung_pass", "exact_xs", "plus5_xs", "halves_xs", "checks", "passed")})
for n in ("GR1", "GR2", "GR3"):
    print(n, "rungs", {r: (v["pass"], round(v["main"]["dd"], 4), v["main"]["breach_mean"], v["main"]["breach_worst_seed"], round(v["s3_plus_5bps"]["dd"], 4), v["s3_plus_5bps"]["breach_mean"], v["s3_plus_5bps"]["breach_worst_seed"]) for r, v in study["books"][n]["rungs"].items()})
