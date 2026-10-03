"""Data for the HPI-G page (owner request 2026-10-03): same page as DV2-G, for HPI 2/3/5 vote.

Source: real-engine runs (check B / check C): HPI vote ungated and gated (DV2-G gate: VIX > expanding mean of VIX,
open >= 15 sessions), engine costs and +5 bps. Idle cash is swept from the engine's cash column.
Variants: ungated, gated + T-bills, gated + momentum (SPMO, 8% vol target; from 2015-11), gated + T-bills levered to
the ungated pod's volatility (financing T-bill + 1.5%).
Writes results/research/mr_gate_final_checks_20261003/hpi_page_data.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mr_gate_selfcal_20261003"))
import final_page_data as fp  # noqa: E402

sc, rg, rs, rp, npc, tbc = fp.sc, fp.rg, fp.rs, fp.rp, fp.npc, fp.tbc
OUT = HERE.parents[2] / "results/research/mr_gate_final_checks_20261003"
FULL0, MOM0, BOOK0, BOOK1 = "2004-01-05", "2015-11-02", "2004-01-05", "2026-08-19"


def engine(arm):
    d = pd.read_csv(OUT / f"hpi_{arm}_nav.csv", index_col=0, parse_dates=True).astype(float)
    r = d["total_value"].pct_change().fillna(0.0)
    cash_w = (d["cash"].clip(lower=0) / d["total_value"]).shift(1).fillna(0.0)
    gross = 1.0 - d["cash"] / d["total_value"]
    return r, cash_w, gross


def trades(arm):
    tx = pd.read_csv(OUT / f"hpi_{arm}_transactions.csv", parse_dates=["bar"])
    ent = tx[tx["amount"] > 0].groupby("trade_id").agg(entry=("price", "first"), d0=("bar", "first"))
    ext = tx[tx["amount"] < 0].groupby("trade_id").agg(exit=("price", "last"), d1=("bar", "last"))
    t = ent.join(ext, how="inner")
    return pd.DataFrame({"ret": t["exit"] / t["entry"] - 1.0, "d0": t["d0"], "d1": t["d1"]})


def main():
    r_u, cw_u, g_u = engine("ungated")
    r_g, cw_g, g_g = engine("gated")
    r_us, cw_us, _ = engine("ungated_stress")
    r_gs, cw_gs, _ = engine("gated_stress")
    idx = r_u.index
    rate = rs.cash_rate(pd.DatetimeIndex(idx))
    spmo = npc.load_total_return_ret_ser("SPMO", "SPMO").reindex(idx).fillna(0.0)
    rv = spmo.rolling(20).std().shift(1) * np.sqrt(252)
    w = (0.08 / rv).clip(upper=1).fillna(0.0)
    mom_park = w * spmo + (1 - w) * rate

    def park(r, cw, pk):
        return r + cw * pk.reindex(r.index).fillna(0.0)

    u = park(r_u, cw_u, rate)
    g = park(r_g, cw_g, rate)
    mom = park(r_g, cw_g, mom_park)
    fin = rate + 0.015 / 252
    k = float(u.loc[FULL0:].std() / g.loc[FULL0:].std())
    lev = k * g - (k - 1) * fin.reindex(g.index).fillna(0.0)
    V = {"ungated": u, "gated_tbills": g, "gated_mom": mom, "gated_levered": lev}
    spy = npc.load_spy_tr_ret_ser().reindex(idx).fillna(0.0)
    last = str(idx[-1].date())

    # gate (same definition as the engine run) on the panel calendar
    p = rp.Panel("sp500")
    vix = rg.inputs(p)[0]
    thr = sc.selfcal_params(vix)[0]
    gate = pd.Series(sc.gate_mem(vix, thr.to_numpy(), 15).astype(float), index=p.dates)

    tu, tg = trades("ungated"), trades("gated")
    yrs = len(u.loc[FULL0:]) / 252
    out = {"meta": {"memory": 15, "leverage": k, "threshold_today": float(thr.dropna().iloc[-1]), "vix_today": float(vix.dropna().iloc[-1]),
                    "gate_open_today": bool(gate.iloc[-1] > 0.5), "last_date": str(p.dates[-1].date()), "engine_last": last}}
    su = fp.stats(u.loc[FULL0:])
    su.update(exposure=float(g_u.loc[FULL0:].mean()), trades_per_year=len(tu) / yrs, win_rate=float((tu["ret"] > 0).mean()), avg_trade=float(tu["ret"].mean()),
              sharpe_5bps=fp.stats(park(r_us, cw_us, rate).loc[FULL0:])["sharpe"], cagr_5bps=fp.stats(park(r_us, cw_us, rate).loc[FULL0:])["cagr"])
    sg = fp.stats(g.loc[FULL0:])
    sg.update(exposure=float(g_g.loc[FULL0:].mean()), trades_per_year=len(tg) / yrs, win_rate=float((tg["ret"] > 0).mean()), avg_trade=float(tg["ret"].mean()),
              sharpe_5bps=fp.stats(park(r_gs, cw_gs, rate).loc[FULL0:])["sharpe"], cagr_5bps=fp.stats(park(r_gs, cw_gs, rate).loc[FULL0:])["cagr"])
    out["stats_full"] = {"ungated": su, "gated_tbills": sg, "gated_levered": fp.stats(lev.loc[FULL0:]), "spy": fp.stats(spy.loc[FULL0:])}
    out["stats_mom"] = {k2: fp.stats(v.loc[MOM0:]) for k2, v in {**V, "spy": spy}.items()}
    out["stats_mom"]["gated_mom"]["sharpe_5bps"] = fp.stats(park(r_gs, cw_gs, mom_park).loc[MOM0:])["sharpe"]
    out["stats_mom"]["gated_tbills"]["sharpe_5bps"] = fp.stats(park(r_gs, cw_gs, rate).loc[MOM0:])["sharpe"]
    out["stats_mom"]["ungated"]["sharpe_5bps"] = fp.stats(park(r_us, cw_us, rate).loc[MOM0:])["sharpe"]
    # crises (those inside the engine window)
    out["crises"] = []
    for name, a, b in fp.CRISES:
        if a < FULL0:
            continue
        row = {"name": name, "start": a, "end": b}
        for k2, v in {**V, "spy": spy}.items():
            if k2 == "gated_mom" and a < MOM0:
                row[k2] = None
                continue
            nav = (1 + v.loc[a:b]).cumprod()
            row[k2] = {"ret": float(nav.iloc[-1] - 1), "dd": float((nav / nav.cummax() - 1).min())}
        out["crises"].append(row)
    out["annual"] = {k2: {str(yy): float((1 + x).prod() - 1) for yy, x in v.loc[FULL0:].groupby(v.loc[FULL0:].index.year)} for k2, v in {**V, "spy": spy}.items()}

    def wk(series_dict, start):
        df = pd.DataFrame({k2: (1 + v.loc[start:]).cumprod() for k2, v in series_dict.items()})
        dd = df / df.cummax() - 1
        wnav, wdd = df.resample("W-FRI").last().dropna(how="all"), dd.resample("W-FRI").min().dropna(how="all")
        return {"dates": [d.strftime("%Y-%m-%d") for d in wnav.index],
                **{f"{k2}_nav": [round(float(x), 4) for x in wnav[k2]] for k2 in wnav},
                **{f"{k2}_dd": [round(float(x), 4) for x in wdd[k2]] for k2 in wdd}}

    out["series_full"] = wk({"ungated": u, "gated_tbills": g, "gated_levered": lev}, FULL0)
    wd = pd.DatetimeIndex(out["series_full"]["dates"])
    out["series_full"]["gate"] = [round(float(x), 3) for x in gate.loc[FULL0:].resample("W-FRI").mean().reindex(wd).fillna(0)]
    out["series_full"]["vix"] = [round(float(x), 2) for x in vix.loc[FULL0:].resample("W-FRI").last().reindex(wd).ffill()]
    out["series_full"]["thr"] = [round(float(x), 2) for x in thr.loc[FULL0:].resample("W-FRI").last().reindex(wd).ffill()]
    out["series_mom"] = wk(V, MOM0)
    # book
    taa, L, Ls, bil = tbc.load_taa_ser(), npc.load_l_ret_ser("engine"), npc.load_l_ret_ser("stress"), npc.load_bil_ret_ser()

    def book(x, a="2008-03-04", b=BOOK1, Lx=L):
        return tbc.book_window_return_ser({"taa": taa, "L": Lx, "X": x}, npc.CANDIDATE_WEIGHT_DICT, a, b)

    xs = {"tbills_slot": bil, **V}
    out["book_blocks"] = {k2: {blk: tbc.metric_dict(book(v, a, b)) for blk, (a, b) in npc.BOOK_BLOCK_DICT.items()} for k2, v in xs.items() if k2 != "gated_mom"}
    xs_s = {"tbills_slot": bil, "ungated": park(r_us, cw_us, rate), "gated_tbills": park(r_gs, cw_gs, rate)}
    out["book_blocks_stress"] = {k2: {blk: tbc.metric_dict(book(v, a, b, Ls)) for blk, (a, b) in npc.BOOK_BLOCK_DICT.items()} for k2, v in xs_s.items()}
    out["book_mom"] = {k2: tbc.metric_dict(book(v, MOM0, BOOK1)) for k2, v in xs.items()}
    bser = pd.DataFrame({k2: (1 + book(v)).cumprod() for k2, v in xs.items() if k2 != "gated_mom"}).resample("W-FRI").last().dropna()
    out["book_series"] = {"dates": [d.strftime("%Y-%m-%d") for d in bser.index], **{k2: [round(float(x), 4) for x in bser[k2]] for k2 in bser}}
    # routing
    sl = pd.read_csv(fp.SLEEVES, index_col=0, parse_dates=True)
    parks = {"T-bills": rate, "SPMO (8% vol target)": mom_park, "NDX VXN pod": sl["ndx_vxn"], "TAA 3x pod": sl["taa_btal_tqqq"], "CORE5 pod": sl["core5"]}
    out["routing"] = {}
    for name, pk in parks.items():
        x = park(r_g, cw_g, pk)
        start = MOM0 if name.startswith("SPMO") else "2008-03-04"
        out["routing"][name] = {"start": start, "pod": fp.stats(x.loc[start:BOOK1]), "book": tbc.metric_dict(book(x, start, BOOK1)),
                                "book_mom_window": tbc.metric_dict(book(x, MOM0, BOOK1)),
                                "corr_ndx": float(pd.DataFrame({"a": x, "b": L}).dropna().loc[start:BOOK1].corr().iloc[0, 1])}
    out["gate_open_by_year"] = {str(yy): float(v.mean()) for yy, v in gate.loc[FULL0:].groupby(gate.loc[FULL0:].index.year)}
    (OUT / "hpi_page_data.json").write_text(json.dumps(out, default=float), encoding="utf-8")
    print(json.dumps({"meta": out["meta"], "full": {k2: {kk: round(vv, 3) for kk, vv in v.items()} for k2, v in out["stats_full"].items()},
                      "mom": {k2: {kk: round(out["stats_mom"][k2][kk], 3) for kk in ("cagr", "sharpe", "max_dd", "wealth_100k")} for k2 in out["stats_mom"]}}, indent=1, default=float))
    for c in out["crises"]:
        print(f"{c['name']:<26}", "  ".join(f"{k2[:7]}:{c[k2]['ret'] * 100:6.1f}%" if c[k2] else f"{k2[:7]}:   -  " for k2 in ("ungated", "gated_tbills", "gated_mom", "gated_levered", "spy")))
    print("book", {k2: {b: round(v[b]["sharpe"], 3) for b in ("G-P1", "G-P2", "G-P3", "G-FULL", "G-LONG")} for k2, v in out["book_blocks"].items()})
    print("book stress", {k2: {b: round(v[b]["sharpe"], 3) for b in ("G-P1", "G-P2", "G-P3", "G-FULL", "G-LONG")} for k2, v in out["book_blocks_stress"].items()})
    print("book since 2015", {k2: (round(v["sharpe"], 3), round(v["cagr"], 3), round(v["max_dd"], 3)) for k2, v in out["book_mom"].items()})
    print("routing", {n: (round(v["book"]["sharpe"], 3), round(v["book_mom_window"]["sharpe"], 3), round(v["pod"]["sharpe"], 3), round(v["corr_ndx"], 2)) for n, v in out["routing"].items()})
    print("gate open by year", {y: round(v, 2) for y, v in out["gate_open_by_year"].items()})


if __name__ == "__main__":
    main()
