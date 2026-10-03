"""MR capsule components (SPEC_FROZEN.md): base daily return (idle cash at 0), idle-cash weight and holdings intervals.

DV2 (replica) and industry-ETF DV2 (replica) are run here; HPI comes from the real-engine runs of the gate checks.
Gate = DV2-G gate (VIX > expanding mean of VIX, open >= 15 sessions from the opening).
Writes results/research/mr_capsule_20261003/components.parquet, holdings_*.parquet, etf_validation.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mr_gate_selfcal_20261003"))
sys.path.insert(0, str(HERE.parent / "dv2_deep_20260925"))
import run_selfcal as sc  # noqa: E402
import phase4_universes as p4  # noqa: E402

rg, rs, ll, rp, npc, tbc = sc.rg, sc.rs, sc.ll, sc.rp, sc.npc, sc.tbc
q = rg.q
REPO = HERE.parents[2]
OUT = REPO / "results/research/mr_capsule_20261003"
CHECKS = REPO / "results/research/mr_gate_final_checks_20261003"
SLEEVES = REPO / "results/research/portfolio/portfolio_refresh_20260927/sleeve_series_incl_2008.csv.gz"
START, END, PROXY = "2000-01-03", "2026-09-24", ("1995-01-03", "2003-12-31")
ETF_LIST = p4.GROUPS["industries"]


def gate_on(dates: pd.DatetimeIndex) -> np.ndarray:
    d = pd.DatetimeIndex(np.load(rs.ETFX / "dates.npy"))
    vix = pd.Series(np.load(rs.ETFX / "vix_close.npy"), index=d).reindex(dates).ffill(limit=3)
    thr = sc.selfcal_params(vix)[0]
    return sc.gate_mem(vix, thr.to_numpy(), 15)


def base_and_cash(res):
    r = pd.Series(res.nav, index=res.dates).pct_change().fillna(0.0)
    cw = pd.Series(1.0 - res.diag["gross_ser"], index=res.dates).clip(lower=0).shift(1).fillna(0.0)
    return r, cw


def holdings(res, name):
    t = res.trades[["asset", "entry_date", "exit_date"]].copy()
    t["pod"] = name
    return t


def replica_component(p, entry, score, exit_, gate, label, cols, hold, start=START, end=END):
    for gated in (False, True):
        e = entry & gate[:, None] if gated else entry
        for cost, bps in (("engine", 0.0), ("stress", 5.0)):
            res = ll.run(p, ll.Spec(label, e, score, exit_, slip_extra_bps=bps), start, end)
            r, cw = base_and_cash(res)
            key = f"{label}-{'G' if gated else 'U'}|{cost}"
            cols[f"{key}|base"], cols[f"{key}|cw"] = r, cw
            if cost == "engine":
                hold[f"{label}-{'G' if gated else 'U'}"] = holdings(res, label)
            print(key, "Sharpe(T-bills)", round(rs.stats(r + cw * rs.cash_rate(r.index))["sharpe"], 3), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cols, hold = {}, {}
    # ---- DV2 (S&P 500 replica)
    p = rp.Panel("sp500")
    g = gate_on(p.dates)
    e, s, x = ll.dv2_masks(p, rp.Rule())
    replica_component(p, e, s, x, g, "DV2", cols, hold)
    # ---- pre-2004 proxy: DV2 and QPI (HPI stand-in), engine costs
    fin = q.Features(p)
    turn = p.feat("turn", lambda: np.asarray(p.RAW) * np.asarray(p.V))
    proxy = {}
    for label, (pe, ps, px) in {"DV2": (e, s, x), "QPI": (fin.entry, turn, fin.exit)}.items():
        for gated in (False, True):
            res = ll.run(p, ll.Spec(label, pe & g[:, None] if gated else pe, ps, px), *PROXY)
            r, cw = base_and_cash(res)
            proxy[f"{label}-{'G' if gated else 'U'}|engine|base"], proxy[f"{label}-{'G' if gated else 'U'}|engine|cw"] = r, cw
    pd.DataFrame(proxy).to_parquet(OUT / "proxy_1995_2003.parquet")
    del p, fin
    # ---- HPI vote (real engine runs)
    for gated, arm in ((False, "ungated"), (True, "gated")):
        for cost, suffix in (("engine", ""), ("stress", "_stress")):
            d = pd.read_csv(CHECKS / f"hpi_{arm}{suffix}_nav.csv", index_col=0, parse_dates=True).astype(float)
            key = f"HPI-{'G' if gated else 'U'}|{cost}"
            cols[f"{key}|base"] = d["total_value"].pct_change().fillna(0.0)
            cols[f"{key}|cw"] = (d["cash"].clip(lower=0) / d["total_value"]).shift(1).fillna(0.0)
        tx = pd.read_csv(CHECKS / f"hpi_{arm}_transactions.csv", parse_dates=["bar"])
        ent = tx[tx["amount"] > 0].groupby("trade_id").agg(asset=("asset", "first"), entry_date=("bar", "first"))
        ext = tx[tx["amount"] < 0].groupby("trade_id").agg(exit_date=("bar", "last"))
        h = ent.join(ext, how="left").reset_index(drop=True)
        h["pod"] = "HPI"
        hold[f"HPI-{'G' if gated else 'U'}"] = h
    # ---- industry-ETF DV2 (optional A): 19 ETFs, 252 sessions of history, ADV63 > $50M
    pe_ = p4.etf_panel(ETF_LIST)
    ge = gate_on(pe_.dates)
    ee, se, xe = ll.dv2_masks(pe_, rp.Rule())
    with np.errstate(invalid="ignore"):
        ee = ee & (np.asarray(rp.adv63(pe_)) > 50e6)
    replica_component(pe_, ee, se, xe, ge, "ETF", cols, hold)
    sleeve = pd.read_csv(SLEEVES, index_col=0, parse_dates=True)["etf_ind_fix"]
    rep_u = cols["ETF-U|engine|base"]
    both = pd.concat([rep_u, sleeve], axis=1, keys=["replica", "engine"]).dropna().loc["2012-01-03":]
    val = {"corr_daily_2012_plus": float(both.corr().iloc[0, 1]),
           "sharpe_replica_cash0": rs.stats(both["replica"])["sharpe"], "sharpe_engine_sleeve": rs.stats(both["engine"])["sharpe"],
           "cagr_replica_cash0": rs.stats(both["replica"])["cagr"], "cagr_engine_sleeve": rs.stats(both["engine"])["cagr"]}
    (OUT / "etf_validation.json").write_text(json.dumps(val, indent=2), encoding="utf-8")
    print("ETF validation", {k: round(v, 3) for k, v in val.items()}, flush=True)
    df = pd.DataFrame(cols).sort_index()
    df.to_parquet(OUT / "components.parquet")
    pd.concat([h.assign(variant=k) for k, h in hold.items()], ignore_index=True).to_parquet(OUT / "holdings.parquet")
    print("saved", df.shape, flush=True)


if __name__ == "__main__":
    main()
