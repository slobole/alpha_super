"""What does the momentum leg (E2) add to the book? (owner question 2026-10-05). Exploratory, descriptive.

Legs: TAA 3x (taa_btal_tqqq), momentum E2 (PortfolioManager run of ndx_e2_sector_cap_5050), MR capsule with BIL
(real engine), QQQ total return (the passive alternative for the momentum slot), T-bills.
Reports: Sharpe by block for each leg; E2 alpha and beta vs QQQ (daily OLS, Newey-West t); crisis returns; books
TAA 0.5 + X 0.25 + MR 0.25 with X = E2 / QQQ / T-bills / (none: MR 0.5); a paired block bootstrap vs the E2 book;
and the rolling 3-year Sharpe gap between the E2 book and the no-momentum book.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare as cmp  # noqa: E402
import mr_final_slot_stats as mss  # noqa: E402
import page_v2_data as pvd  # noqa: E402

ev, tbc, st, npc = cmp.ev, cmp.sr.tbc, mss.st, cmp.npc
START, END = "2008-03-04", "2026-08-19"
BLOCKS = {"2008–11": ("2008-03-04", "2011-12-30"), "2012–21": ("2012-01-03", "2021-12-31"), "2022–26": ("2022-01-03", END), "since Nov 2017": ("2017-11-01", END)}
CRISES = {"GFC": ("2008-03-04", "2009-03-09"), "Aug 2011": ("2011-07-22", "2011-10-03"), "Q4 2018": ("2018-10-01", "2018-12-24"),
          "COVID": ("2020-02-19", "2020-03-23"), "2022": ("2022-01-03", "2022-12-30"), "Feb–Apr 2025": ("2025-02-19", "2025-04-08")}


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(252))


def newey_west_alpha(y: pd.Series, x: pd.Series, lags: int = 10) -> dict:
    df = pd.concat([y, x], axis=1).dropna()
    yv, xv = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    X = np.column_stack([np.ones(len(xv)), xv])
    beta = np.linalg.lstsq(X, yv, rcond=None)[0]
    e = yv - X @ beta
    xtx_inv = np.linalg.inv(X.T @ X)
    S = (X * e[:, None]).T @ (X * e[:, None])
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        g = (X[lag:] * e[lag:, None]).T @ (X[:-lag] * e[:-lag, None])
        S += w * (g + g.T)
    cov = xtx_inv @ S @ xtx_inv
    return {"alpha_ann": float(beta[0] * 252), "beta": float(beta[1]), "t_alpha": float(beta[0] / np.sqrt(cov[0, 0]))}


def main() -> None:
    sl = pd.read_csv(tbc.SLEEVE_SERIES_PATH, index_col=0, parse_dates=True)
    taa = sl["taa_btal_tqqq"].astype(float)
    e2 = pvd.load_e2_ret_ser()
    runs = {p: cmp.load_engine(p, "bil") for p in ("dv2", "hpi")}
    idx = runs["dv2"][0].index
    mr = ev.capsule({p.upper(): runs[p][0]["total_value"].pct_change().reindex(idx) for p in ("dv2", "hpi")}, {"DV2": .5, "HPI": .5}, start=pvd.FULL_START, end=pvd.END)
    qqq = npc.load_total_return_ret_ser("QQQ", "QQQ")
    rate = st.cash_rate(pd.DatetimeIndex(taa.index))
    legs = {"TAA 3x": taa, "Momentum E2": e2, "MR capsule": mr, "QQQ": qqq}
    out: dict = {"legs": {}, "alpha": {}, "books": {}}
    for name, r in legs.items():
        r = r.loc[START:END].dropna()
        out["legs"][name] = {"sharpe_blocks": {b: sharpe(r.loc[a:z]) for b, (a, z) in BLOCKS.items()},
                             "full": mss.stats(r, rate), "crises": {c: float((1 + r.loc[a:z]).prod() - 1) for c, (a, z) in CRISES.items()}}
    for b, (a, z) in {"2008–26": (START, END), **BLOCKS}.items():
        out["alpha"][b] = newey_west_alpha(e2.loc[a:z], qqq.loc[a:z])
    w = {"taa": .5, "x": .25, "mr": .25}
    variants = {"TAA + E2 + MR (base)": {"taa": taa, "x": e2, "mr": mr}, "TAA + QQQ + MR": {"taa": taa, "x": qqq, "mr": mr},
                "TAA + T-bills + MR": {"taa": taa, "x": rate, "mr": mr}}
    series = {k: tbc.book_window_return_ser(v, w, START, END) for k, v in variants.items()}
    series["TAA 0.5 + MR 0.5 (no momentum)"] = tbc.book_window_return_ser({"taa": taa, "mr": mr}, {"taa": .5, "mr": .5}, START, END)
    for k, r in series.items():
        out["books"][k] = {"full": mss.stats(r, rate), "sharpe_blocks": {b: sharpe(r.loc[a:z]) for b, (a, z) in BLOCKS.items()},
                           "crises": {c: float((1 + r.loc[a:z]).prod() - 1) for c, (a, z) in CRISES.items()},
                           "p_beats_base": None if k.endswith("(base)") else ev.bootstrap_p(r, series["TAA + E2 + MR (base)"])}
    base, nomom = series["TAA + E2 + MR (base)"], series["TAA 0.5 + MR 0.5 (no momentum)"]
    roll = (base.rolling(756).mean() / base.rolling(756).std() - nomom.rolling(756).mean() / nomom.rolling(756).std()) * np.sqrt(252)
    roll = roll.dropna()
    out["rolling_3y_sharpe_gap_base_minus_nomom"] = {"share_base_better": float((roll > 0).mean()), "median": float(roll.median()),
                                                     "min": float(roll.min()), "max": float(roll.max())}
    (cmp.OUT / "momentum_role.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print("LEGS: Sharpe by block | full CAGR / Sharpe / DD")
    for n, d in out["legs"].items():
        f = d["full"]
        print(f"  {n:<12}", {b: round(v, 2) for b, v in d["sharpe_blocks"].items()}, f"| {f['cagr'] * 100:.1f}% / {f['sharpe']:.2f} / {f['max_dd'] * 100:.1f}%",
              "| crises", {c: round(v * 100, 1) for c, v in d["crises"].items()})
    print("E2 vs QQQ:", {b: {k: round(v, 3) for k, v in d.items()} for b, d in out["alpha"].items()})
    print("BOOKS:")
    for n, d in out["books"].items():
        f = d["full"]
        print(f"  {n:<32} {f['cagr'] * 100:.1f}% / {f['sharpe']:.3f} / {f['max_dd'] * 100:.1f}% worstY {f['worst_year'] * 100:.1f}%",
              {b: round(v, 3) for b, v in d["sharpe_blocks"].items()}, "P>base", d["p_beats_base"], "| crises", {c: round(v * 100, 1) for c, v in d["crises"].items()})
    print("rolling 3y Sharpe gap (base - no momentum):", {k: round(v, 3) for k, v in out["rolling_3y_sharpe_gap_base_minus_nomom"].items()})


if __name__ == "__main__":
    main()
