"""Evaluate check C (HPI vote at +5 bps per side, gated vs ungated)."""
import json
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_b_evaluate as cb  # noqa: E402
fp, rs, npc, tbc, OUT = cb.fp, cb.rs, cb.npc, cb.tbc, cb.OUT
BLK = ("G-P1", "G-P2", "G-P3")


def main():
    idx = pd.read_csv(OUT / "hpi_ungated_stress_nav.csv", index_col=0, parse_dates=True).index
    rate = rs.cash_rate(pd.DatetimeIndex(idx))
    taa, Ls, bil = tbc.load_taa_ser(), npc.load_l_ret_ser("stress"), npc.load_bil_ret_ser()
    rep = {"C_BIL_stress": npc.candidate_book_blocks(taa, Ls, bil)}
    for arm in ("ungated", "gated"):
        r, expo = cb.swept(f"{arm}_stress", rate)
        rep[arm] = {"standalone": fp.stats(r.loc["2004-01-05":]), "book": npc.candidate_book_blocks(taa, Ls, r),
                    "halves": {"2004_14": fp.stats(r.loc[:"2014-12-31"])["sharpe"], "2015_26": fp.stats(r.loc["2015-01-01":])["sharpe"]}}
    u, g = rep["ungated"], rep["gated"]
    rep["gated_preferred"] = bool(all(g["book"][k]["sharpe"] >= u["book"][k]["sharpe"] for k in ("G-FULL", "G-LONG"))
                                  and all(g["book"][k]["sharpe"] >= u["book"][k]["sharpe"] - 0.03 for k in BLK))
    (OUT / "check_c.json").write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    eng = json.loads((OUT / "check_b.json").read_text())
    for arm in ("ungated", "gated"):
        s = rep[arm]["standalone"]
        print(arm, f"STRESS CAGR {s['cagr']:.3f} Sharpe {s['sharpe']:.3f} DD {s['max_dd']:.3f} halves", {k: round(v, 3) for k, v in rep[arm]["halves"].items()},
              f"| engine-cost Sharpe {eng[arm]['standalone']['sharpe']:.3f} CAGR {eng[arm]['standalone']['cagr']:.3f}")
        print("   book stress", {k: round(rep[arm]["book"][k]["sharpe"], 3) for k in BLK + ("G-FULL", "G-LONG")}, "| engine", {k: round(eng[arm]["book"][k]["sharpe"], 3) for k in BLK + ("G-FULL", "G-LONG")})
    print("C_BIL stress", {k: round(rep["C_BIL_stress"][k]["sharpe"], 3) for k in BLK + ("G-FULL", "G-LONG")}, "gated_preferred", rep["gated_preferred"])


if __name__ == "__main__":
    main()
