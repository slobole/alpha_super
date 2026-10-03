"""Data for the final DV2-G page (owner request 2026-10-03).

DV2-G = DV2 wired; new entries only while the gate is open. The gate opens when VIX closes above its expanding
mean (all history since 1990, known at the close) and stays open >= 15 sessions from the opening.
Variants: ungated DV2, DV2-G + T-bills, DV2-G + momentum (SPMO, 8% vol target; from 2015-11),
DV2-G + T-bills levered to the ungated pod's volatility (financing T-bill + 1.5%).
Writes results/research/mr_gate_selfcal_20261003/final_page_data.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_selfcal as sc  # noqa: E402

rg, rs, ll, rp, npc, tbc = sc.rg, sc.rs, sc.ll, sc.rp, sc.npc, sc.tbc
OUT = sc.OUT
SLEEVES = sc.rg.q.REPO / "results/research/portfolio/portfolio_refresh_20260927/sleeve_series_incl_2008.csv.gz"
FULL0, MOM0, END, BOOK0, BOOK1 = "2000-01-03", "2015-11-02", "2026-09-24", "2008-03-04", "2026-08-19"
MEMORY = 15
CRISES = [
    ("Dot-com bear", "2000-03-24", "2002-10-09"), ("Global financial crisis", "2007-10-09", "2009-03-09"),
    ("Euro / US downgrade 2011", "2011-07-22", "2011-10-03"), ("China / oil 2015–16", "2015-08-17", "2016-02-11"),
    ("Volmageddon 2018", "2018-01-26", "2018-02-08"), ("Q4 2018", "2018-09-20", "2018-12-24"),
    ("COVID crash", "2020-02-19", "2020-03-23"), ("COVID rebound", "2020-03-23", "2020-06-08"),
    ("2022 bear", "2022-01-03", "2022-10-12"), ("2025 tariff shock", "2025-02-19", "2025-04-08"),
]


def stats(r: pd.Series, gross=None, trades=None):
    r = r.dropna()
    nav = (1 + r).cumprod()
    yrs = len(r) / 252
    dn = r[r < 0]
    m = (1 + r).resample("ME").prod() - 1
    y = (1 + r).resample("YE").prod() - 1
    dd = nav / nav.cummax() - 1
    out = {"cagr": float(nav.iloc[-1] ** (1 / yrs) - 1), "vol": float(r.std() * np.sqrt(252)),
           "sharpe": float(r.mean() / r.std() * np.sqrt(252)), "sortino": float(r.mean() / np.sqrt((dn ** 2).mean()) * np.sqrt(252)),
           "max_dd": float(dd.min()), "ulcer": float(np.sqrt((dd ** 2).mean())), "worst_month": float(m.min()), "best_month": float(m.max()),
           "worst_year": float(y.min()), "pos_months": float((m > 0).mean()), "wealth_100k": float(100_000 * nav.iloc[-1])}
    out["calmar"] = out["cagr"] / abs(out["max_dd"])
    if gross is not None:
        out["exposure"] = float(np.mean(gross))
    if trades is not None and len(trades):
        out.update(trades_per_year=len(trades) / yrs, win_rate=float((trades["ret"] > 0).mean()), avg_trade=float(trades["ret"].mean()))
    return out


def main():
    p = rp.Panel("sp500")
    vix = rg.inputs(p)[0]
    rate = rs.cash_rate(p.dates)
    DV = ll.dv2_masks(p, rp.Rule())
    thr = sc.selfcal_params(vix)[0]
    gate = sc.gate_mem(vix, thr.to_numpy(), MEMORY)
    res_u = ll.run(p, ll.Spec("u", *DV), FULL0, END)
    res_g = ll.run(p, rg.spec(DV, "off", gate), FULL0, END)
    res_g5 = ll.run(p, rg.spec(DV, "off", gate, 5.0), FULL0, END)
    res_u5 = ll.run(p, ll.Spec("u", *DV, slip_extra_bps=5.0), FULL0, END)
    base_g = pd.Series(res_g.nav, index=res_g.dates).pct_change().fillna(0.0)
    cash_g = pd.Series(1 - res_g.diag["gross_ser"], index=res_g.dates).clip(lower=0).shift(1).fillna(0.0)

    def routed(park):
        return base_g + cash_g * park.reindex(res_g.dates).fillna(0.0)

    spmo = npc.load_total_return_ret_ser("SPMO", "SPMO").reindex(p.dates).fillna(0.0)
    rv = spmo.rolling(20).std().shift(1) * np.sqrt(252)
    w = (0.08 / rv).clip(upper=1).fillna(0.0)
    mom_park = w * spmo + (1 - w) * rate
    u = rs.swept(res_u, rate)
    g = routed(rate)
    mom = routed(mom_park)
    fin = rate + 0.015 / 252
    k = float(u.loc[FULL0:].std() / g.loc[FULL0:].std())
    lev = k * g - (k - 1) * fin.reindex(g.index).fillna(0.0)
    V = {"ungated": u, "gated_tbills": g, "gated_mom": mom, "gated_levered": lev}
    spy = npc.load_spy_tr_ret_ser().reindex(p.dates).fillna(0.0)

    out = {"meta": {"memory": MEMORY, "leverage": k, "threshold_today": float(thr.iloc[-1]), "vix_today": float(vix.iloc[-1]),
                    "gate_open_today": bool(gate[-1]), "last_date": str(p.dates[-1].date()),
                    "threshold_by_year": {str(yr): float(thr[thr.index.year == yr].iloc[-1]) for yr in range(1995, 2027)}}}
    # stats
    out["stats_full"] = {"ungated": stats(u.loc[FULL0:], res_u.diag["gross_ser"], res_u.trades),
                         "gated_tbills": stats(g.loc[FULL0:], res_g.diag["gross_ser"], res_g.trades),
                         "gated_levered": stats(lev.loc[FULL0:]), "spy": stats(spy.loc[FULL0:END])}
    out["stats_full"]["ungated"]["sharpe_5bps"] = stats(rs.swept(res_u5, rate))["sharpe"]
    out["stats_full"]["gated_tbills"]["sharpe_5bps"] = stats(routed(rate) * 0 + rs.swept(res_g5, rate))["sharpe"]
    out["stats_mom"] = {k2: stats(v.loc[MOM0:]) for k2, v in {**V, "spy": spy.loc[:END]}.items()}
    # crises
    out["crises"] = []
    for name, a, b in CRISES:
        row = {"name": name, "start": a, "end": b}
        for k2, v in {**V, "spy": spy}.items():
            if k2 == "gated_mom" and a < MOM0:
                row[k2] = None
                continue
            seg = v.loc[a:b]
            nav = (1 + seg).cumprod()
            row[k2] = {"ret": float(nav.iloc[-1] - 1), "dd": float((nav / nav.cummax() - 1).min())}
        out["crises"].append(row)
    # annual
    out["annual"] = {k2: {str(yy): float((1 + x).prod() - 1) for yy, x in v.loc[FULL0:].groupby(v.loc[FULL0:].index.year)} for k2, v in {**V, "spy": spy.loc[:END]}.items()}
    # weekly series
    def wk(series_dict, start):
        df = pd.DataFrame({k2: (1 + v.loc[start:]).cumprod() for k2, v in series_dict.items()})
        dd = df / df.cummax() - 1
        wnav, wdd = df.resample("W-FRI").last().dropna(how="all"), dd.resample("W-FRI").min().dropna(how="all")
        return {"dates": [d.strftime("%Y-%m-%d") for d in wnav.index],
                **{f"{k2}_nav": [None if not np.isfinite(x) else round(float(x), 4) for x in wnav[k2]] for k2 in wnav},
                **{f"{k2}_dd": [None if not np.isfinite(x) else round(float(x), 4) for x in wdd[k2]] for k2 in wdd}}
    gate_s = pd.Series(gate.astype(float), index=p.dates)
    out["series_full"] = wk({"ungated": u, "gated_tbills": g, "gated_levered": lev}, FULL0)
    out["series_full"]["gate"] = [round(float(x), 3) for x in gate_s.loc[FULL0:].resample("W-FRI").mean().reindex(pd.DatetimeIndex(out["series_full"]["dates"])).fillna(0)]
    out["series_full"]["vix"] = [round(float(x), 2) for x in vix.loc[FULL0:].resample("W-FRI").last().reindex(pd.DatetimeIndex(out["series_full"]["dates"])).ffill()]
    out["series_full"]["thr"] = [round(float(x), 2) for x in thr.loc[FULL0:].resample("W-FRI").last().reindex(pd.DatetimeIndex(out["series_full"]["dates"])).ffill()]
    out["series_mom"] = wk(V, MOM0)
    # book
    taa = tbc.load_taa_ser()
    L = npc.load_l_ret_ser("engine")
    bil = npc.load_bil_ret_ser()
    def book(x, a=BOOK0, b=BOOK1):
        return tbc.book_window_return_ser({"taa": taa, "L": L, "X": x}, npc.CANDIDATE_WEIGHT_DICT, a, b)
    xs = {"tbills_slot": bil, **V}
    out["book_blocks"] = {k2: {blk: tbc.metric_dict(book(v, a, b)) for blk, (a, b) in npc.BOOK_BLOCK_DICT.items()} for k2, v in xs.items() if k2 != "gated_mom"}
    out["book_mom"] = {k2: tbc.metric_dict(book(v, MOM0, BOOK1)) for k2, v in xs.items()}
    bser = pd.DataFrame({k2: (1 + book(v)).cumprod() for k2, v in xs.items() if k2 != "gated_mom"}).resample("W-FRI").last().dropna()
    out["book_series"] = {"dates": [d.strftime("%Y-%m-%d") for d in bser.index], **{k2: [round(float(x), 4) for x in bser[k2]] for k2 in bser}}
    # routing idle cash to another pod
    sl = pd.read_csv(SLEEVES, index_col=0, parse_dates=True)
    parks = {"T-bills": rate, "SPMO (8% vol target)": mom_park, "NDX VXN pod": sl["ndx_vxn"], "TAA 3x pod": sl["taa_btal_tqqq"], "CORE5 pod": sl["core5"]}
    out["routing"] = {}
    for name, pk in parks.items():
        x = routed(pk)
        start = MOM0 if name.startswith("SPMO") else BOOK0
        pod = x.loc[start:BOOK1]
        out["routing"][name] = {"start": start, "pod": stats(pod), "book": tbc.metric_dict(book(x, start, BOOK1)),
                                "book_mom_window": tbc.metric_dict(book(x, MOM0, BOOK1)),
                                "corr_ndx": float(pd.DataFrame({"a": x, "b": L}).dropna().loc[start:BOOK1].corr().iloc[0, 1])}
    out["gate_open_by_year"] = {str(yy): float(v.mean()) for yy, v in gate_s.loc[FULL0:].groupby(gate_s.loc[FULL0:].index.year)}
    (OUT / "final_page_data.json").write_text(json.dumps(out, default=float), encoding="utf-8")
    print(json.dumps({"meta": out["meta"] | {"threshold_by_year": None}, "full": {k2: {kk: round(vv, 3) for kk, vv in v.items()} for k2, v in out["stats_full"].items()}}, indent=1, default=float))
    print({n: (round(v["book"]["sharpe"], 3), round(v["book_mom_window"]["sharpe"], 3), round(v["pod"]["sharpe"], 3)) for n, v in out["routing"].items()})


if __name__ == "__main__":
    main()
