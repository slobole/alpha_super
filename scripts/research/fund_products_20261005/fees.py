"""Fund products, final pass: fee income per fee schedule (owner request 2026-10-05, descriptive, after results).

Fee model (stated on the page): management fee m a year, accrued daily on NAV (m / 252 a session); performance fee p on
the gain above the high-water mark, crystallised at the last session of each calendar year, no hurdle. Income is shown
per unit of AUM: (fees of the year) / (NAV at the start of the year), averaged over the calendar years of LONG. At a
constant AUM of $1M the dollar income is that percentage times $1M.

Two return paths per book: the backtest (MAIN frame) and the conservative case (every engine keeps three quarters of
its excess return, model costs: battery.decay_frame at k = 0.75).

Usage: PYTHONDONTWRITEBYTECODE=1 python fees.py   (after study.py). Writes <study>/report/fees.json.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import battery
import g_lib as g
from g_lib import TBILL, Lab

SCHEDULES = [(0.02, 0.20), (0.015, 0.20), (0.01, 0.15), (0.01, 0.10), (0.01, 0.05)]


def fee_path(r: pd.Series, m: float, p: float) -> dict:
    """Investor NAV net of fees and the manager's income per calendar year (as a share of start-of-year NAV)."""
    nav, hwm = 1.0, 1.0
    net = []
    years: dict = {}
    idx = r.index
    for i, (d, x) in enumerate(r.items()):
        y = d.year
        st = years.setdefault(y, {"start": nav, "mgmt": 0.0, "perf": 0.0})
        gross = nav * (1.0 + x)
        fee_m = gross * m / 252.0
        nav_new = gross - fee_m
        st["mgmt"] += fee_m
        last = i == len(idx) - 1 or idx[i + 1].year != y
        if last and nav_new > hwm:                       # crystallise on the last session of the year
            fee_p = p * (nav_new - hwm)
            nav_new -= fee_p
            st["perf"] += fee_p
            hwm = nav_new
        net.append(nav_new / nav - 1.0)
        nav = nav_new
    out_years = {int(y): {"mgmt": v["mgmt"] / v["start"], "perf": v["perf"] / v["start"]} for y, v in years.items()}
    return {"net": pd.Series(net, index=idx), "years": out_years}


def summarise(r: pd.Series, rf: pd.Series, m: float, p: float) -> dict:
    fp = fee_path(r, m, p)
    gross, net = g.stats(r, rf), g.stats(fp["net"], rf)
    bil = g.stats(rf.reindex(r.index), rf)["cagr"]
    tot = {y: v["mgmt"] + v["perf"] for y, v in fp["years"].items()}
    full = [y for y in tot if y not in (min(tot), max(tot))]           # full calendar years only for the spread
    arr = np.array([tot[y] for y in full])
    perf = np.array([fp["years"][y]["perf"] for y in full])
    mg = np.array([fp["years"][y]["mgmt"] for y in full])
    drag = gross["cagr"] - net["cagr"]
    return {"m": m, "p": p, "gross_cagr": gross["cagr"], "net_cagr": net["cagr"], "net_xs": net["xs"], "gross_xs": gross["xs"], "net_dd": net["dd"],
            "fee_drag_cagr": drag, "share_of_gross_cagr": drag / gross["cagr"], "share_of_excess_over_bil": drag / max(gross["cagr"] - bil, 1e-9),
            "income_mean": float(arr.mean()), "income_median": float(np.median(arr)), "income_min": float(arr.min()), "income_max": float(arr.max()),
            "mgmt_mean": float(mg.mean()), "perf_mean": float(perf.mean()), "years_no_perf_fee": int((perf <= 1e-12).sum()), "years": len(full),
            "worst_year": int(full[int(arr.argmin())]), "bil_cagr": bil}


def main() -> int:
    lab = Lab()
    rf = lab.rf
    d_launch = g.with_cash(g.DEF, 0.10)
    books = {"Defensive launch": d_launch, "Growth": g.PRODUCTS["GR1"], "Growth Plus": g.PRODUCTS["GR2"], "Aggressive": g.PRODUCTS["GR3"], "Monthly": g.MONTHLY, "Growth x1.40 (leverage)": g.levered(g.PRODUCTS["GR1"], 1.40),
             "Blend Growth 40 / Defensive 60": g.blend((0.4, g.PRODUCTS["GR1"]), (0.6, d_launch)),
             "Blend Growth 60 / Defensive 40": g.blend((0.6, g.PRODUCTS["GR1"]), (0.4, d_launch))}
    fr75 = battery.decay_frame(lab.frame, dict.fromkeys(battery.ALL_CAPS, 0.75))
    out: dict = {"schedules": [[m, p] for m, p in SCHEDULES], "books": {},
                 "model": "mgmt accrued daily on NAV; performance fee on gains above the high-water mark, crystallised at calendar year end, no hurdle"}
    for name, w in books.items():
        paths = {"backtest": lab.ret(w), "conservative": lab.ret(w, frame=fr75)}
        out["books"][name] = {"weights": {k: float(v) for k, v in w.items()},
                              **{case: {f"{m * 100:g}/{p * 100:g}": summarise(r, rf, m, p) for m, p in SCHEDULES} for case, r in paths.items()}}
        b = out["books"][name]["backtest"]
        print(name, {k: (round(v["income_mean"] * 100, 2), round(v["net_cagr"] * 100, 1), round(v["net_xs"], 2)) for k, v in b.items()}, flush=True)
    (g.OUT / "fees.json").write_text(json.dumps(g.r6(out), indent=1, default=str), encoding="utf-8")
    g.ledger("fees_finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
