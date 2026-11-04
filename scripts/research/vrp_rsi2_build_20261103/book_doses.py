import sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.dont_write_bytecode = True
HERE = Path(r"C:\Users\User\Documents\workspace\alpha_super\scripts\research\fund_products_opt_20261102"); sys.path.insert(0, str(HERE))
SP = Path(r"C:\Users\User\Documents\workspace\alpha_super\results\research\vrp_rsi2_build_20261103")
import lens_lib as LL, opt_lib as O
g, hunt = O.g, O.hunt
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 44)
lab = O.load_lab("tail_cache_extras.json"); lab._disk_path = SP / "tail_cache_locked.json"
s = pd.concat([pd.read_parquet(SP / "locked_series.parquet"), pd.read_parquet(SP / "locked_series2.parquet")], axis=1)
legs = {"VRP": ("locked", "locked_p5"), "VRP 15/35": ("stress", "stress_p5"), "VRP no boost": ("base", "base_p5"), "VRP no long vol": ("no_long", "no_long_p5"), "VRP half VIXY": ("vixy_half", "vixy_half_p5"), "VRP one late": ("late", "late_p5"), "VRP no COVID": ("nocovid", "nocovid_p5"), "VRP VX pre-2011": ("vx1", "vx1_p5"), "VRP VX late pre-2011": ("vx2", "vx2_p5")}
for k, (m_, p_) in legs.items(): O.fl.add_leg(lab, k, s[m_], s[p_])
O.fl.add_leg(lab, "TB", lab.rf.reindex(s.index).fillna(0.0), lab.rf.reindex(s.index).fillna(0.0))
picks = json.loads((O.OUT / "picks.json").read_text(encoding="utf-8"))["picks"]
rf = lab.rf
def row(name, base, full=False):
    r = lab.r(name); out = {}
    for tag, a_ in (("08", None), ("11", "2011-10-10"), ("23", "2023-01-01")):
        st = g.stats(r.loc[a_:] if a_ else r, rf); out.update({f"cagr{tag}": st["cagr"], f"xs{tag}": st["xs"], f"dd{tag}": st["dd"]})
    if base:
        rb = lab.r(base)
        out.update(share=lab.paired(r, rb)["share_xs"], s11=lab.paired(r.loc["2011-10-10":], rb.loc["2011-10-10":])["share_xs"], s23=lab.paired(r.loc["2023-01-01":], rb.loc["2023-01-01":])["share_xs"])
    return out
for prod in ("Growth", "Growth Plus"):
    p = picks[f"A {prod}"]; w = O.weights_of(p["t"], p["m"], p["o"], p["e"], p["c"], p["f"]); b = lab.add(f"{prod} pick", w)
    res = {f"{prod} pick": row(b, None)}
    for leg in ("TB", *legs):
        n = lab.add(f"{prod} +10 {leg}", g.blend((.9, w), (.1, {leg: 1.0}))); res[n] = row(n, b)
    n = lab.add(f"{prod} +5 VRP", g.blend((.95, w), (.05, {"VRP": 1.0}))); res[n] = row(n, b)
    w2 = dict(w); w2[O.CORE] = w2[O.CORE] - .10; w2["VRP"] = .10; n = lab.add(f"{prod} CORE5 -10 to VRP", w2); res[n] = row(n, b)
    print(f"\n== {prod}: {p['label']}"); print(pd.DataFrame(res).T.round(3).to_string())
    n = f"{prod} +10 VRP"; ch = lab.challenge(n, b, breach_key="p25", rung="GROWTH PLUS"); print("checks", {k[:2]: bool(v) for k, v in ch["checks"].items()}, "passed", ch["passed"]); print("check names", list(ch["checks"].keys()))
# the doc's table
ws = {O.TAA1N: .5, O.DV2_L: .15, O.HPI_L: .15, O.CORE: .2}; sb = lab.add("Sized REC", ws); n = lab.add("Sized REC +10 VRP", g.blend((.9, ws), (.1, {"VRP": 1.0})))
print("\n== doc's Sized REC"); print(pd.DataFrame({"Sized REC": row(sb, None), n: row(n, sb)}).T.round(3).to_string())
r = lab.r("Growth Plus +10 VRP"); rb = lab.r("Growth Plus pick")
for nm, a_, b_ in (("Oct 2008", "2008-09-01", "2008-11-30"), ("Feb 2018", "2018-01-26", "2018-02-12"), ("COVID", "2020-02-19", "2020-03-23"), ("2022", "2022-01-03", "2022-12-30"), ("Aug 2024", "2024-07-31", "2024-08-08"), ("Apr 2025", "2025-02-19", "2025-04-08")):
    print(nm, "pick", round((1 + rb.loc[a_:b_]).prod() - 1, 3), "+10 VRP", round((1 + r.loc[a_:b_]).prod() - 1, 3), "+10 VX", round((1 + lab.r("Growth Plus +10 VRP VX pre-2011").loc[a_:b_]).prod() - 1, 3))
e = s["locked"].loc["2011-10-10":]; print("corr VRP vs pick, vs Vote, vs MR, ex 2020", {k: round(e.corr(lab.r(k).reindex(e.index)), 2) for k in ("Growth Plus pick",)}, round(e[e.index.year != 2020].corr(lab.r("Growth Plus pick").reindex(e.index)[e.index.year != 2020]), 2))
