"""Evaluate check B (HPI vote gated vs ungated, real engine)."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mr_gate_selfcal_20261003"))
import final_page_data as fp  # noqa: E402
rs, npc, tbc = fp.rs, fp.npc, fp.tbc
OUT = Path(__file__).resolve().parents[3] / "results/research/mr_gate_final_checks_20261003"
BLK = ("G-P1", "G-P2", "G-P3")
rate_all = None


def swept(arm, rate):
    d = pd.read_csv(OUT / f"hpi_{arm}_nav.csv", index_col=0, parse_dates=True).astype(float)
    r = d["total_value"].pct_change().fillna(0.0)
    cw = (d["cash"].clip(lower=0) / d["total_value"]).shift(1).fillna(0.0)
    return r + cw * rate.reindex(r.index).fillna(0.0), float(1 - (d["cash"] / d["total_value"]).mean())


def main():
    idx = pd.read_csv(OUT / "hpi_ungated_nav.csv", index_col=0, parse_dates=True).index
    rate = rs.cash_rate(pd.DatetimeIndex(idx))
    taa, L = tbc.load_taa_ser(), npc.load_l_ret_ser("engine")
    rep = {}
    for arm in ("ungated", "gated"):
        r, expo = swept(arm, rate)
        tx = pd.read_csv(OUT / f"hpi_{arm}_transactions.csv")
        yrs = len(r) / 252
        rep[arm] = {"standalone": fp.stats(r.loc["2004-01-05":]), "exposure": expo, "entries_per_year": float((tx["amount"] > 0).sum() / yrs),
                    "book": npc.candidate_book_blocks(taa, L, r),
                    "halves": {"2004_14": fp.stats(r.loc[:"2014-12-31"])["sharpe"], "2015_26": fp.stats(r.loc["2015-01-01":])["sharpe"]},
                    "crises": {n: float((1 + r.loc[a:b]).prod() - 1) for n, a, b in fp.CRISES if a >= "2004"}}
    u, g = rep["ungated"], rep["gated"]
    rep["pass"] = bool(all(g["book"][k]["sharpe"] >= u["book"][k]["sharpe"] for k in ("G-FULL", "G-LONG"))
                       and all(g["book"][k]["sharpe"] >= u["book"][k]["sharpe"] - 0.03 for k in BLK)
                       and g["standalone"]["sharpe"] >= u["standalone"]["sharpe"] - 0.05)
    bil = npc.load_bil_ret_ser()
    rep["C_BIL"] = npc.candidate_book_blocks(taa, L, bil)
    (OUT / "check_b.json").write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    for arm in ("ungated", "gated"):
        d = rep[arm]; s = d["standalone"]
        print(arm, f"CAGR {s['cagr']:.3f} Sharpe {s['sharpe']:.3f} DD {s['max_dd']:.3f} worst yr {s['worst_year']:.3f} exp {d['exposure']:.2f} entries/yr {d['entries_per_year']:.0f} halves {d['halves']}")
        print("   book", {k: round(d["book"][k]["sharpe"], 3) for k in BLK + ("G-FULL", "G-LONG")}, "DD", round(d["book"]["G-LONG"]["max_dd"], 3))
        print("   crises", {k: round(v, 3) for k, v in d["crises"].items()})
    print("C_BIL", {k: round(rep["C_BIL"][k]["sharpe"], 3) for k in BLK + ("G-FULL", "G-LONG")}, "pass", rep["pass"])


if __name__ == "__main__":
    main()
