"""Chart data for DV2 ANY_OFF vs ungated DV2 (owner request 2026-10-03). Writes results/.../charts_data.json."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_stress as rs  # noqa: E402

q, ll, rp, npc, tbc = rs.q, rs.ll, rs.rp, rs.npc, rs.tbc


def trade_stats(res, yrs):
    t = res.trades
    hold = (pd.to_datetime(t["exit_date"]) - pd.to_datetime(t["entry_date"])).dt.days
    return {"trades_per_year": len(t) / yrs, "win_rate": float((t["ret"] > 0).mean()), "avg_trade": float(t["ret"].mean()),
            "avg_win": float(t.loc[t["ret"] > 0, "ret"].mean()), "avg_loss": float(t.loc[t["ret"] <= 0, "ret"].mean()),
            "avg_hold_days": float(hold.mean())}


def main():
    p = rp.Panel("sp500")
    G = rs.gates(p)
    e, s, x = ll.dv2_masks(p, rp.Rule())
    rate = rs.cash_rate(p.dates)
    specs = {"ungated": ll.Spec("u", e, s, x), "any_off": ll.Spec("a", e & G["ANY"][:, None], s, x)}
    out = {"series": {}, "stats": {}, "annual": {}}
    weekly = {}
    for k, sp in specs.items():
        res = ll.run(p, sp, rs.MAIN0, rs.END)
        r = rs.swept(res, rate)
        res_s = ll.run(p, ll.Spec(k, sp.open_entry, s, x, slip_extra_bps=5.0), rs.MAIN0, rs.END)
        r_s = rs.swept(res_s, rate)
        nav = (1 + r).cumprod()
        yrs = len(r) / 252
        out["stats"][k] = {**rs.stats(r), "vol": float(r.std() * np.sqrt(252)), "stress": rs.stats(r_s),
                           "exposure": float(np.mean(res.diag["gross_ser"])), **trade_stats(res, yrs),
                           "blocks": rs.blocks(r, rs.STANDALONE_BLOCKS)}
        out["annual"][k] = {str(y): float((1 + g).prod() - 1) for y, g in r.groupby(r.index.year)}
        weekly[f"{k}_nav"] = nav
        weekly[f"{k}_dd"] = nav / nav.cummax() - 1
        weekly[f"{k}_exp"] = pd.Series(res.diag["gross_ser"], index=res.dates).rolling(21).mean()
    gate = pd.Series(G["ANY"], index=p.dates).loc[rs.MAIN0:rs.END].astype(float)
    weekly["gate"] = gate.rolling(5).mean()
    w = pd.DataFrame(weekly).resample("W-FRI").last().dropna()
    out["series"]["dates"] = [d.strftime("%Y-%m-%d") for d in w.index]
    for c in w.columns:
        out["series"][c] = [round(float(v), 4) for v in w[c]]
    out["gate_open_by_year"] = {str(y): float(g.mean()) for y, g in gate.groupby(gate.index.year)}
    # book, G-LONG and blocks
    taa = tbc.load_taa_ser()
    L = npc.load_l_ret_ser("engine")
    bil = npc.load_bil_ret_ser()
    ret = pd.read_parquet(rs.OUT / "returns.parquet")
    xs = {"tbills": bil, "ungated": ret["DV2|base|main"], "any_off": ret["DV2|ANY_OFF|main"]}
    a, b = npc.BOOK_BLOCK_DICT["G-LONG"]
    bk = {}
    for k, xser in xs.items():
        br = tbc.book_window_return_ser({"taa": taa, "L": L, "X": xser}, npc.CANDIDATE_WEIGHT_DICT, a, b)
        bk[k] = br
    bdf = pd.DataFrame({k: (1 + v).cumprod() for k, v in bk.items()}).resample("W-FRI").last().dropna()
    out["book_series"] = {"dates": [d.strftime("%Y-%m-%d") for d in bdf.index], **{k: [round(float(v), 4) for v in bdf[k]] for k in bdf}}
    res_json = json.loads((rs.OUT / "results.json").read_text())
    out["book_blocks"] = {"tbills": res_json["controls"]["C_BIL"], "ungated": res_json["controls"]["C_DV2"],
                          "any_off": res_json["DV2_book"]["ANY_OFF"]["book"],
                          "stress": {"tbills": res_json["controls"]["C_BIL_stress_L"], "ungated": res_json["controls"]["C_DV2_stress"],
                                     "any_off": res_json["DV2_book"]["ANY_OFF"]["book_stress"]}}
    out["edge_by_regime"] = res_json["DV2_trades_by_regime"]["ANY"]
    out["holdout"] = {k: res_json["DV2_standalone"][v]["holdout"] for k, v in (("ungated", "base"), ("any_off", "ANY_OFF"))}
    (rs.OUT / "charts_data.json").write_text(json.dumps(out, default=float), encoding="utf-8")
    print(json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else None) for kk, vv in v.items()} for k, v in out["stats"].items()}, indent=1))


if __name__ == "__main__":
    main()
