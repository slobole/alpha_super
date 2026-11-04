import sys, json
from pathlib import Path
import pandas as pd
sys.dont_write_bytecode = True
HERE = Path(r"C:\Users\User\Documents\workspace\alpha_super\scripts\research\fund_products_opt_20261102"); sys.path.insert(0, str(HERE))
SP = Path(r"C:\Users\User\Documents\workspace\alpha_super\results\research\vrp_rsi2_build_20261103")
import opt_lib as O
g = O.g
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)
lab = O.load_lab("tail_cache_extras.json"); lab._disk_path = SP / "tail_cache_locked3.json"
c = pd.read_parquet(SP / "locked_costs.parquet"); s = pd.read_parquet(SP / "locked_series.parquet"); s2 = pd.read_parquet(SP / "locked_series2.parquet")
for k in (5, 10, 15, 20, 25, 30, 35, 40): O.fl.add_leg(lab, f"V{k}", c[f"c{k}"], c[f"c{k+5}"])
O.fl.add_leg(lab, "VX", s["vx1"], s["vx1_p5"]); O.fl.add_leg(lab, "NOCOV", s["nocovid"], s["nocovid_p5"]); O.fl.add_leg(lab, "HALFY", s2["vixy_half"], s2["vixy_half_p5"]); O.fl.add_leg(lab, "LATE", s["late"], s["late_p5"]); O.fl.add_leg(lab, "NOLONG", s["no_long"], s["no_long_p5"])
picks = json.loads((O.OUT / "picks.json").read_text(encoding="utf-8"))["picks"]; rf = lab.rf
def row(n, b):
    r, rb = lab.r(n), lab.r(b); out = {}
    for tag, a_ in (("08", None), ("11", "2011-10-10")):
        st = g.stats(r.loc[a_:] if a_ else r, rf); out.update({f"cagr{tag}": st["cagr"], f"xs{tag}": st["xs"], f"dd{tag}": st["dd"]})
    out.update(share=lab.paired(r, rb)["share_xs"], s11=lab.paired(r.loc["2011-10-10":], rb.loc["2011-10-10":])["share_xs"], s23=lab.paired(r.loc["2023-01-01":], rb.loc["2023-01-01":])["share_xs"])
    return out
for prod, rung in (("Growth", "GROWTH"), ("Growth Plus", "GROWTH PLUS")):
    p = picks[f"A {prod}"]; w = O.weights_of(p["t"], p["m"], p["o"], p["e"], p["c"], p["f"]); b = lab.add(f"{prod} pick", w); res = {}
    for leg in ["V5", "V10", "V15", "V20", "V25", "V30", "V35", "NOCOV", "HALFY", "LATE", "NOLONG", "VX"]:
        w2 = dict(w); w2[O.CORE] -= .10; w2[leg] = .10; n = lab.add(f"{prod} CORE5 to {leg}", w2); res[leg] = row(n, b)
        if leg in ("V5", "V10", "V15", "V20", "HALFY", "NOCOV"):
            ch = lab.challenge(n, b, breach_key=O.BREACH[rung], rung=rung); res[leg]["fails"] = ",".join(k[:2] for k, v in ch["checks"].items() if not v) or "pass"
    print(f"\n== {prod}, 10 points from CORE5"); print(pd.DataFrame(res).T.to_string(float_format=lambda x: f"{x:.3f}"))
    st = g.stats(lab.r(b), rf); print("pick", round(st["cagr"], 3), round(st["xs"], 3), round(st["dd"], 3))
