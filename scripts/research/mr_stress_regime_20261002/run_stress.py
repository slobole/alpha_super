"""Stress-gated MR study (SPEC_FROZEN.md). Usage: python run_stress.py"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "qpi_moc_20261002"))
sys.path.insert(0, str(HERE.parent))  # scripts/research
import qpi_lib as q  # noqa: E402
import loc_lib as ll  # noqa: E402
from new_pod_search_20260927 import common as npc  # noqa: E402
from trend_breakout_20260927 import common as tbc  # noqa: E402

rp = q.rp
OUT = q.REPO / "results/research/mr_stress_regime_20261002"
ETFX = q.REPO / "results/research/mr_beyond_dv2_20260926/cache/etfx"
MAIN0, END = "2000-01-03", "2026-09-24"
HOLD_DV2, HOLD_QPI = ("1991-01-02", "1999-12-31"), ("1995-01-03", "2003-12-31")
QPI0 = "2004-01-02"
STANDALONE_BLOCKS = {"2000-09": ("2000-01-03", "2009-12-31"), "2010-19": ("2010-01-01", "2019-12-31"), "2020-26": ("2020-01-01", END)}
DTB3 = q.REPO.parent / "1_data" / "DTB3.csv"


def gates(p):
    d = pd.DatetimeIndex(np.load(ETFX / "dates.npy"))
    vix = pd.Series(np.load(ETFX / "vix_close.npy"), index=d).reindex(p.dates).ffill(limit=3)
    spx = pd.Series(np.load(ETFX / "spx_close.npy"), index=d).reindex(p.dates).ffill(limit=3)
    v20 = vix > 20
    vrel = vix > vix.rolling(252, min_periods=252).median()
    mkt = spx < spx.rolling(200, min_periods=200).mean()
    g = {"V20": v20, "VREL": vrel, "MKT": mkt, "ANY": v20 | mkt}
    return {k: v.fillna(False).to_numpy() for k, v in g.items()}


def cash_rate(idx):
    """Daily cash return: BIL total return from its first return, lagged DTB3 / 252 before."""
    bil = npc.load_bil_ret_ser()
    d = pd.read_csv(DTB3, parse_dates=["observation_date"], na_values=["."]).set_index("observation_date")["DTB3"].dropna()
    dtb = (d.reindex(d.index.union(idx)).ffill().shift(1).reindex(idx) / 100.0 / 252.0).fillna(0.0)
    out = dtb.copy()
    common = idx.intersection(bil.index)
    out.loc[common] = bil.loc[common].to_numpy()
    return out


def swept(res, rate):
    r = pd.Series(res.nav, index=res.dates).pct_change().fillna(0.0)
    cash_prev = pd.Series(1.0 - res.diag["gross_ser"], index=res.dates).clip(lower=0).shift(1).fillna(0.0)
    return r + cash_prev * rate.reindex(res.dates).fillna(0.0)


def stats(r):
    m = tbc.metric_dict(r)
    return {"cagr": m["cagr"], "sharpe": m["sharpe"], "max_dd": m["max_dd"]}


def blocks(r, bd):
    return {k: stats(r.loc[a:b]) for k, (a, b) in bd.items()}


def trade_table(p, res, g):
    """Per-trade return minus the S&P 500 price return over the same holding, labelled by the gate at decision."""
    d = pd.DatetimeIndex(np.load(ETFX / "dates.npy"))
    spx = pd.Series(np.load(ETFX / "spx_close.npy"), index=d).reindex(p.dates).ffill()
    pos = {x: k for k, x in enumerate(p.dates)}
    rows = []
    for a, e, x, r in res.trades[["asset", "entry_date", "exit_date", "ret"]].itertuples(index=False):
        te, tx = pos[e], pos[x]
        mkt = spx.iloc[tx - 1] / spx.iloc[te - 1] - 1.0 if tx - 1 > te - 1 else 0.0
        rows.append({"entry": e, "adj": r - mkt, "ret": r, **{k: bool(v[te - 1]) for k, v in g.items()}})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p = rp.Panel("sp500")
    G = gates(p)
    fin = q.Features(p)
    turn = p.feat("turn", lambda: np.asarray(p.RAW) * np.asarray(p.V))
    d_entry, d_score, d_exit = ll.dv2_masks(p, rp.Rule())
    base = {"DV2": (d_entry, d_score, d_exit), "QPI": (fin.entry, turn, fin.exit)}
    rate = cash_rate(p.dates)
    rep = {"gate_open_share_2000_26": {k: float(v[(p.dates >= MAIN0)].mean()) for k, v in G.items()}}

    def spec(strat, gate=None, mode=None, stress=0.0):
        e, s, x = base[strat]
        kw = {"slip_extra_bps": stress}
        if gate is None:
            return ll.Spec(strat, e, s, x, **kw)
        g = G[gate]
        if mode == "OFF":
            return ll.Spec(strat, e & g[:, None], s, x, **kw)
        return ll.Spec(strat, e, s, x, size_mult=np.where(g, 1.0, 0.5), **kw)

    variants = [(g, m) for g in G for m in ("OFF", "HALF")]
    # ---- DV2 main, stress, holdout; QPI main, holdout
    R = {}
    for strat, (a0, h) in (("DV2", (MAIN0, HOLD_DV2)), ("QPI", (QPI0, HOLD_QPI))):
        for v in [None] + variants:
            key = "base" if v is None else f"{v[0]}_{v[1]}"
            sp = spec(strat, *(v or (None, None)))
            R[(strat, key, "main")] = swept(ll.run(p, sp, a0, END), rate)
            R[(strat, key, "hold")] = swept(ll.run(p, sp, *h), rate)
            if strat == "DV2":
                R[(strat, key, "stress")] = swept(ll.run(p, spec(strat, *(v or (None, None)), stress=5.0), "2007-01-03", END), rate)
            if v is None:
                rep[f"{strat}_trades_by_regime"] = None  # filled below
            print(strat, key, round(stats(R[(strat, key, "main")])["sharpe"], 3), flush=True)
    # ---- descriptive per-trade edge by regime (ungated)
    for strat, (a0, h) in (("DV2", (MAIN0, HOLD_DV2)), ("QPI", (QPI0, HOLD_QPI))):
        res = ll.run(p, spec(strat), h[0], END)
        tt = trade_table(p, res, G)
        tt["block"] = pd.cut(pd.to_datetime(tt["entry"]).dt.year, [0, 1999, 2009, 2019, 2100], labels=["1990s", "2000-09", "2010-19", "2020-26"])
        out = {}
        for gname in G:
            grp = tt.groupby(["block", gname], observed=True)["adj"].agg(["count", "mean"])
            out[gname] = {f"{b}|{'open' if o else 'closed'}": {"n": int(r["count"]), "mkt_adj_bps": float(r["mean"] * 1e4)} for (b, o), r in grp.iterrows()}
        rep[f"{strat}_trades_by_regime"] = out
    # ---- standalone tables
    for strat in ("DV2", "QPI"):
        rep[f"{strat}_standalone"] = {}
        for v in ["base"] + [f"{g}_{m}" for g, m in variants]:
            r = R[(strat, v, "main")]
            rep[f"{strat}_standalone"][v] = {"main": stats(r), "holdout": stats(R[(strat, v, "hold")]),
                                             "blocks": blocks(r, STANDALONE_BLOCKS) if strat == "DV2" else {}}
    # ---- book
    taa = tbc.load_taa_ser()
    L, Ls = npc.load_l_ret_ser("engine"), npc.load_l_ret_ser("stress")
    bil = npc.load_bil_ret_ser()
    c_bil = npc.candidate_book_blocks(taa, L, bil)
    c_bil_s = npc.candidate_book_blocks(taa, Ls, bil)
    c_dv2 = npc.candidate_book_blocks(taa, L, R[("DV2", "base", "main")])
    c_dv2_s = npc.candidate_book_blocks(taa, Ls, R[("DV2", "base", "stress")])
    rep["controls"] = {"C_BIL": c_bil, "C_DV2": c_dv2, "C_BIL_stress_L": c_bil_s, "C_DV2_stress": c_dv2_s}
    blk = ("G-P1", "G-P2", "G-P3")
    rep["DV2_book"] = {}
    q_base = stats(R[("QPI", "base", "main")])["sharpe"]
    for g, m in variants:
        v = f"{g}_{m}"
        b = npc.candidate_book_blocks(taa, L, R[("DV2", v, "main")])
        bs = npc.candidate_book_blocks(taa, Ls, R[("DV2", v, "stress")])

        def r12(bk, cb, cd):
            r1 = all(bk[k]["sharpe"] > max(cb[k]["sharpe"], cd[k]["sharpe"]) for k in blk)
            r2 = all(bk[k]["max_dd"] >= cb[k]["max_dd"] - 0.02 for k in ("G-FULL", "G-LONG"))
            return r1, r2
        r1, r2 = r12(b, c_bil, c_dv2)
        r1s, r2s = r12(bs, c_bil_s, c_dv2_s)
        r4 = stats(R[("QPI", v, "main")])["sharpe"] > q_base
        rep["DV2_book"][v] = {"book": b, "book_stress": bs, "R1": r1, "R2": r2, "R3": bool(r1s and r2s), "R4": bool(r4),
                              "pass": bool(r1 and r2 and r1s and r2s and r4)}
        print("BOOK", v, {k: round(b[k]["sharpe"], 3) for k in blk + ("G-FULL",)}, "R1-4", r1, r2, r1s and r2s, r4, flush=True)
    rep["controls_sharpe"] = {n: {k: round(c[k]["sharpe"], 3) for k in blk + ("G-FULL", "G-LONG")} for n, c in rep["controls"].items()}
    print("controls", rep["controls_sharpe"])
    pd.DataFrame({f"{s}|{v}|{w}": r for (s, v, w), r in R.items()}).to_parquet(OUT / "returns.parquet")
    (OUT / "results.json").write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
