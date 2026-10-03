"""Data for the MR capsule page (owner request 2026-10-03): DV2-G + HPI-G 50/50, with a 3x3 parking grid.

Each pod parks its idle cash independently in one of:
    tbills   idle cash at T-bills (BIL / DTB3)
    lev      the gated pod (T-bills parking) scaled to its own ungated pod's volatility, financed at T-bill + 1.5%
             (scale set on 2004-2026, the same data: reported, not tested)
    spmo     idle cash in SPMO with an 8% volatility target (rest in T-bills); SPMO starts 2015-10, so every combination
             that uses it is compared from 2015-11-02
Capsule = pod model, 50/50 capital, annual reset. Book = TAA 0.5 + NDX 0.25 + capsule 0.25 (official pod model).
Writes results/research/mr_capsule_20261003/capsule_page_data.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate as ev  # noqa: E402

cp, fp, rs, npc, tbc = ev.cp, ev.fp, ev.rs, ev.npc, ev.tbc
OUT = cp.OUT
C0, M0, S_END, B0, B1 = "2004-01-05", "2015-11-02", "2026-09-24", "2008-03-04", "2026-08-19"
PARKS = ("tbills", "lev", "spmo")
PNAME = {"tbills": "T-bills", "lev": "T-bills levered", "spmo": "SPMO"}


def main():
    comp = pd.read_parquet(OUT / "components.parquet")
    idx = pd.DatetimeIndex(comp.index)
    rate = rs.cash_rate(idx)
    spmo = npc.load_total_return_ret_ser("SPMO", "SPMO").reindex(idx).fillna(0.0)
    rv = spmo.rolling(20).std().shift(1) * np.sqrt(252)
    w = (0.08 / rv).clip(upper=1).fillna(0.0)
    mom = w * spmo + (1 - w) * rate
    fin = rate + 0.015 / 252

    def parked(name, cost, park):
        return comp[f"{name}|{cost}|base"] + comp[f"{name}|{cost}|cw"] * park

    pods, lev_k = {}, {}
    for pod in ("DV2", "HPI"):
        for cost in ("engine", "stress"):
            g_t = parked(f"{pod}-G", cost, rate)
            u_t = parked(f"{pod}-U", cost, rate)
            k = float(parked(f"{pod}-U", "engine", rate).loc[C0:S_END].std() / parked(f"{pod}-G", "engine", rate).loc[C0:S_END].std())
            lev_k[pod] = k
            pods[(pod, "tbills", cost)] = g_t
            pods[(pod, "lev", cost)] = k * g_t - (k - 1) * fin
            pods[(pod, "spmo", cost)] = parked(f"{pod}-G", cost, mom)
            pods[(pod, "ungated", cost)] = u_t

    def capsule(dp, hp, cost, start):
        return ev.capsule({"DV2": pods[("DV2", dp, cost)], "HPI": pods[("HPI", hp, cost)]}, {"DV2": 0.5, "HPI": 0.5}, start=start, end=S_END)

    taa, L, Ls, bil = tbc.load_taa_ser(), npc.load_l_ret_ser("engine"), npc.load_l_ret_ser("stress"), npc.load_bil_ret_ser()
    LX = {"engine": L, "stress": Ls}

    def book(x, cost, a, b):
        return tbc.book_window_return_ser({"taa": taa, "L": LX[cost], "X": x}, npc.CANDIDATE_WEIGHT_DICT, a, b)

    def money(r):
        return float(100_000 * (1 + r).prod())

    out = {"meta": {"lev_k": lev_k, "window_grid": [M0, S_END], "book_grid": [M0, B1], "window_full": [C0, S_END], "book_full": [B0, B1]}}
    # ---- 3x3 grid since 2015-11
    grid = {}
    for dp in PARKS:
        for hp in PARKS:
            cap_e, cap_s = capsule(dp, hp, "engine", M0), capsule(dp, hp, "stress", M0)
            be, bs = book(cap_e, "engine", M0, B1), book(cap_s, "stress", M0, B1)
            st = fp.stats(cap_e)
            grid[f"{dp}|{hp}"] = {"capsule": st, "capsule_money": money(cap_e), "capsule_sharpe_5bps": fp.stats(cap_s)["sharpe"],
                                  "capsule_money_5bps": money(cap_s),
                                  "book": tbc.metric_dict(be), "book_money": money(be), "book_sharpe_5bps": tbc.metric_dict(bs)["sharpe"],
                                  "book_money_5bps": money(bs)}
    out["grid"] = grid
    # references since 2015-11
    refs = {"ungated capsule": ev.capsule({"DV2": pods[("DV2", "ungated", "engine")], "HPI": pods[("HPI", "ungated", "engine")]}, {"DV2": 0.5, "HPI": 0.5}, start=M0, end=S_END),
            "DV2-G alone": pods[("DV2", "tbills", "engine")].loc[M0:S_END], "HPI-G alone": pods[("HPI", "tbills", "engine")].loc[M0:S_END]}
    out["grid_refs"] = {k: {"capsule": fp.stats(v), "capsule_money": money(v), "book": tbc.metric_dict(book(v, "engine", M0, B1)),
                            "book_money": money(book(v, "engine", M0, B1))} for k, v in refs.items()}
    out["grid_refs"]["T-bills slot"] = {"book": tbc.metric_dict(book(bil, "engine", M0, B1)), "book_money": money(book(bil, "engine", M0, B1))}
    spy = npc.load_spy_tr_ret_ser().reindex(idx).fillna(0.0)
    out["grid_refs"]["SPY"] = {"capsule": fp.stats(spy.loc[M0:S_END]), "capsule_money": money(spy.loc[M0:S_END])}
    # ---- full window (no SPMO): 2x2 + references
    full = {}
    for dp in ("tbills", "lev"):
        for hp in ("tbills", "lev"):
            ce, cs = capsule(dp, hp, "engine", C0), capsule(dp, hp, "stress", C0)
            be = book(ce, "engine", B0, B1)
            full[f"{dp}|{hp}"] = {"capsule": fp.stats(ce), "capsule_money": money(ce), "capsule_sharpe_5bps": fp.stats(cs)["sharpe"],
                                  "book": tbc.metric_dict(be), "book_money": money(be), "book_sharpe_5bps": tbc.metric_dict(book(cs, "stress", B0, B1))["sharpe"]}
    ung = ev.capsule({"DV2": pods[("DV2", "ungated", "engine")], "HPI": pods[("HPI", "ungated", "engine")]}, {"DV2": 0.5, "HPI": 0.5}, start=C0, end=S_END)
    ung_s = ev.capsule({"DV2": pods[("DV2", "ungated", "stress")], "HPI": pods[("HPI", "ungated", "stress")]}, {"DV2": 0.5, "HPI": 0.5}, start=C0, end=S_END)
    full["ungated"] = {"capsule": fp.stats(ung), "capsule_money": money(ung), "capsule_sharpe_5bps": fp.stats(ung_s)["sharpe"],
                       "book": tbc.metric_dict(book(ung, "engine", B0, B1)), "book_money": money(book(ung, "engine", B0, B1)),
                       "book_sharpe_5bps": tbc.metric_dict(book(ung_s, "stress", B0, B1))["sharpe"]}
    for pod in ("DV2", "HPI"):
        x = pods[(pod, "tbills", "engine")].loc[C0:S_END]
        full[f"{pod}-G alone"] = {"capsule": fp.stats(x), "capsule_money": money(x), "book": tbc.metric_dict(book(x, "engine", B0, B1)),
                                  "book_money": money(book(x, "engine", B0, B1))}
    full["T-bills slot"] = {"book": tbc.metric_dict(book(bil, "engine", B0, B1)), "book_money": money(book(bil, "engine", B0, B1))}
    full["SPY"] = {"capsule": fp.stats(spy.loc[C0:S_END]), "capsule_money": money(spy.loc[C0:S_END])}
    out["full"] = full
    # ---- series for charts
    def wk(d: dict, start):
        df = pd.DataFrame({k: (1 + v.loc[start:S_END]).cumprod() * 100_000 for k, v in d.items()}).dropna()
        dd = df / df.cummax() - 1
        wn, wd = df.resample("W-FRI").last(), dd.resample("W-FRI").min()
        return {"dates": [x.strftime("%Y-%m-%d") for x in wn.index], **{f"{k}_nav": [round(float(v), 1) for v in wn[k]] for k in wn},
                **{f"{k}_dd": [round(float(v), 4) for v in wd[k]] for k in wd}}
    out["series_full"] = wk({"ungated": ung, "tbills|tbills": capsule("tbills", "tbills", "engine", C0), "lev|lev": capsule("lev", "lev", "engine", C0)}, C0)
    gate = pd.Series(cp.gate_on(idx).astype(float), index=idx)
    out["series_full"]["gate"] = [round(float(x), 3) for x in gate.loc[C0:S_END].resample("W-FRI").mean().reindex(pd.DatetimeIndex(out["series_full"]["dates"])).fillna(0)]
    out["series_grid"] = wk({"ungated": refs["ungated capsule"], "tbills|tbills": capsule("tbills", "tbills", "engine", M0),
                             "tbills|spmo": capsule("tbills", "spmo", "engine", M0), "spmo|spmo": capsule("spmo", "spmo", "engine", M0),
                             "lev|lev": capsule("lev", "lev", "engine", M0)}, M0)
    # book wealth curves since 2015-11 (engine)
    bdf = pd.DataFrame({k: (1 + book(v, "engine", M0, B1)).cumprod() * 100_000 for k, v in
                        {"tbills_slot": bil, "ungated": refs["ungated capsule"], "tbills|tbills": capsule("tbills", "tbills", "engine", M0),
                         "tbills|spmo": capsule("tbills", "spmo", "engine", M0), "spmo|spmo": capsule("spmo", "spmo", "engine", M0),
                         "lev|lev": capsule("lev", "lev", "engine", M0)}.items()}).resample("W-FRI").last().dropna()
    out["book_series_grid"] = {"dates": [x.strftime("%Y-%m-%d") for x in bdf.index], **{k: [round(float(v), 1) for v in bdf[k]] for k in bdf}}
    # ---- crises (capsule variants vs SPY)
    out["crises"] = []
    cap_full = {"ungated": ung, "tbills|tbills": capsule("tbills", "tbills", "engine", C0), "lev|lev": capsule("lev", "lev", "engine", C0)}
    cap_grid = {"spmo|spmo": capsule("spmo", "spmo", "engine", M0), "tbills|spmo": capsule("tbills", "spmo", "engine", M0)}
    for name, a, b in fp.CRISES:
        if a < "2007-01-01":
            continue
        row = {"name": name, "start": a, "end": b}
        for k, v in {**cap_full, **cap_grid, "spy": spy}.items():
            seg = v.loc[a:b]
            if len(seg) == 0 or (k in cap_grid and a < M0):
                row[k] = None
                continue
            nav = (1 + seg).cumprod()
            row[k] = {"ret": float(nav.iloc[-1] - 1), "dd": float((nav / nav.cummax() - 1).min())}
        out["crises"].append(row)
    # ---- annual returns
    out["annual"] = {k: {str(y): float((1 + g).prod() - 1) for y, g in v.loc[C0:S_END].groupby(v.loc[C0:S_END].index.year)} for k, v in {**cap_full, "spy": spy}.items()}
    out["annual"]["spmo|spmo"] = {str(y): float((1 + g).prod() - 1) for y, g in cap_grid["spmo|spmo"].groupby(cap_grid["spmo|spmo"].index.year) if y >= 2016}
    # ---- carry over the frozen-rule results and construction facts
    evj = json.loads((OUT / "evaluation.json").read_text())
    out["book_blocks"] = {k: evj["book"][k] for k in ("T-bills slot|engine", "DV2-G alone|engine", "HPI-G alone|engine", "ungated|engine", "main|engine",
                                                     "park CORE5|engine", "levered T-bills|engine", "ETF capsule3|engine",
                                                     "T-bills slot|stress", "DV2-G alone|stress", "HPI-G alone|stress", "ungated|stress", "main|stress")}
    out["decision"] = evj["decision"]
    out["diversification"] = evj["diversification"]
    out["overlap_gated"] = evj["overlap_gated"]
    out["bootstrap"] = evj["bootstrap"]
    out["weights"] = {k: evj["book"][f"{k}|engine"]["G-FULL"]["sharpe"] for k in ("DV2 share 0.25", "main", "DV2 share 0.75", "inverse vol")}
    out["etf"] = {"book": evj["book"]["ETF capsule3|engine"], "standalone": evj["standalone"]["ETF capsule3|engine"]}
    out["proxy"] = evj["proxy_1995_2003"]
    lim = pd.read_parquet(OUT / "limit_axis_daily.parquet")
    def lb(x, a, b2):
        return tbc.metric_dict(tbc.book_window_return_ser({"taa": taa, "L": L, "X": x}, npc.CANDIDATE_WEIGHT_DICT, a, b2))["sharpe"]
    out["limit_axis"] = {c: {"2008-11": lb(lim[c], "2008-03-04", "2011-12-31"), "2012-22": lb(lim[c], "2012-10-02", "2022-12-30"),
                             "2008-22": lb(lim[c], "2008-03-04", "2022-12-30")} for c in lim.columns if not c.endswith("gross")}
    out["limit_axis"]["T-bills slot"] = {"2008-11": lb(bil, "2008-03-04", "2011-12-31"), "2012-22": lb(bil, "2012-10-02", "2022-12-30"), "2008-22": lb(bil, "2008-03-04", "2022-12-30")}
    vix_thr = cp.sc.selfcal_params(cp.rg.inputs(cp.rp.Panel("sp500"))[0])[0]
    out["meta"].update({"threshold_today": float(vix_thr.dropna().iloc[-1]), "gate_open_today": bool(gate.loc[:S_END].iloc[-1] > 0.5), "last_date": S_END})
    (OUT / "capsule_page_data.json").write_text(json.dumps(out, default=float), encoding="utf-8")
    # ---- print the grid
    print("lev k", {k: round(v, 3) for k, v in lev_k.items()})
    print("GRID since 2015-11 (DV2 park | HPI park): capsule $ / Sharpe / DD || book $ / Sharpe / +5bps Sharpe / DD")
    for k, v in grid.items():
        print(f"  {k:<14} ${v['capsule_money']:>9,.0f}  {v['capsule']['sharpe']:.2f}  {v['capsule']['max_dd']:.3f} || ${v['book_money']:>9,.0f}  {v['book']['sharpe']:.3f}  {v['book_sharpe_5bps']:.3f}  {v['book']['max_dd']:.3f}")
    for k, v in out["grid_refs"].items():
        print(f"  REF {k:<16}", f"capsule ${v.get('capsule_money', float('nan')):>9,.0f}" if "capsule_money" in v else "", f"book ${v['book_money']:,.0f} Sharpe {v['book']['sharpe']:.3f}" if "book" in v else "")
    print("FULL 2004-26 (capsule) / 2008-26 (book)")
    for k, v in full.items():
        print(f"  {k:<14}", f"capsule ${v['capsule_money']:,.0f} Sh {v['capsule']['sharpe']:.2f} DD {v['capsule']['max_dd']:.3f}" if "capsule" in v else "",
              f"|| book ${v['book_money']:,.0f} Sh {v['book']['sharpe']:.3f} DD {v['book']['max_dd']:.3f}" if "book" in v else "")


if __name__ == "__main__":
    main()
