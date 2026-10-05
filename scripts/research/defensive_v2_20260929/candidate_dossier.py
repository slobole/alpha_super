"""A3 follow-up (owner request 2026-09-30, descriptive): CORE5 + BTAL_QQQ + DV2-IND as a candidate beside the EOM
book, because EOM looks weaker in recent years. Nothing here selects.

Per book (fair and house cash): window metrics, the R4 slot test per window, forward-split scores and ranks among
all 949 books, cost stress on the third pod's own fills, crises, calendar years. Per sleeve: rolling 36-month excess
return over BIL (is EOM's edge fading? is DV2-IND's?).

Usage: python candidate_dossier.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "shelf_rebuild_20260929"))
sys.path.insert(0, str(HERE))

import lib  # noqa: E402
from lib import BLOCK_DICT, END, EXACT_START, LONG_START, TBILL  # noqa: E402
import defensive_v2 as dv  # noqa: E402
import sharpe_checks as sc  # noqa: E402

OUT = dv.OUT / "sharpe_checks"
DV2_IV, DV2_EQ = "CORE5 + BTAL_QQQ + DV2-IND [IV]", "CORE5 + BTAL_QQQ + DV2-IND [EQ]"
EOM_IV = "CORE5 + BTAL_QQQ + EOM [IV]"
BOTH_EQ = "CORE5 + BTAL_QQQ + EOM + DV2-IND [EQ]"  # both edges, a quarter each
BOOKS = [DV2_IV, DV2_EQ, EOM_IV, BOTH_EQ, sc.CHAMPION, sc.A1_PICK, sc.BENCH]
THIRD = {DV2_IV: "etf_dv2", DV2_EQ: "etf_dv2", EOM_IV: "eom_flow"}
MID = pd.Timestamp("2017-06-30")
WINDOWS = {"LONG": (LONG_START, END), "A 2008-12": BLOCK_DICT["A"], "B 2012-21": BLOCK_DICT["B"],
           "C 2022-26": BLOCK_DICT["C"], "RECENT 3y": BLOCK_DICT["RECENT"], "EXACT 2012-26": (EXACT_START, END),
           "H1 2008-17": (LONG_START, MID), "H2 2017-26": (MID + pd.Timedelta(days=1), END)}


def main() -> int:
    data = lib.load_inputs()
    idx = data["index"]
    fair, house = data["cash_long"], data["long"]
    tb = fair[TBILL]
    by = {b.name: b for b in dv.family("MAIN")}
    out: dict = {"windows": {}, "slot": {}, "forward_rank": {}, "cost": {}, "rolling": {}}

    series = {cash: {b: dv.book_series(f, by[b]) for b in BOOKS} for cash, f in (("fair", fair), ("house", house))}
    for cash in series:
        for b, r in series[cash].items():
            for w, (lo, hi) in WINDOWS.items():
                x = lib.window(r, lo, hi)
                out["windows"][f"{cash}|{b}|{w}"] = {
                    "cagr": lib.cagr(x, lib.base_date(idx, x)), "xs_cagr": lib.excess_cagr(x, tb, idx),
                    "sharpe": sc.sharpe(x), "xs_sharpe": sc.xs_sharpe(x, tb), "maxdd": lib.maxdd(x)}

    # R4 per window: does the third pod earn its slot (excess Calmar with it > with T-bills in its place)?
    for b in [DV2_IV, DV2_EQ, EOM_IV, BOTH_EQ]:
        for cash, f in (("fair", fair), ("house", house)):
            r = series[cash][b]
            alt = {p: dv.book_series(f, by[b], replace=p, weight_source=f) for p in by[b].pods}
            for w, (lo, hi) in WINDOWS.items():
                e = lib.excess_calmar(lib.window(r, lo, hi), tb, idx)
                out["slot"][f"{cash}|{b}|{w}"] = {p: {"with": e, "tbills_instead": lib.excess_calmar(lib.window(a, lo, hi), tb, idx)}
                                                  for p, a in alt.items()}

    # forward-split ranks among all 949 books (fair)
    R = pd.read_csv(dv.OUT / "main_long_returns.csv.gz", index_col=0, parse_dates=True)
    for w in ("H1 2008-17", "H2 2017-26", "RECENT 3y", "EXACT 2012-26"):
        lo, hi = WINDOWS[w]
        Rw = R.loc[lo:hi]
        sh = Rw.mean() / Rw.std() * np.sqrt(252)
        rank = sh.rank(pct=True)
        out["forward_rank"][w] = {b: {"sharpe": float(sh[b]), "pct_rank": float(rank[b])} for b in BOOKS}

    # cost stress on the third pod's own fills (fair, LONG)
    for b, pods in (THIRD | {BOTH_EQ: "eom_flow+etf_dv2"}).items():
        pod = pods.split("+")[0]
        tx, nav = data["tx"][pod], data["nav"][pod]
        t = tx[(tx["date"] >= EXACT_START) & (tx["date"] <= END)]
        notional = t["signed_notional_float"].abs().groupby(t["date"]).sum()
        turnover = float((notional / nav.shift(1).reindex(notional.index)).sum() / ((END - EXACT_START).days / 365.25))
        rows = {}
        for bps in (0, 5, 10, 20):
            f = fair.copy()
            for pod_i in (pods.split("+") if bps else []):
                drag = lib.evaluation.extra_slippage_cost_ser(data["tx"][pod_i], data["nav"][pod_i], bps / 1e4)
                drag = drag.reindex(f.index).fillna(0.0)
                live = f[pod_i].notna()
                f.loc[live, pod_i] = f.loc[live, pod_i] - drag[live]
            r = dv.book_series(f, by[b])
            rows[bps] = {"sharpe": sc.sharpe(r), "xs_sharpe": sc.xs_sharpe(r, tb),
                         "exact_sharpe": sc.sharpe(lib.window(r, EXACT_START, END))}
        out["cost"][b] = {"third_pod": pods, "turnover_per_year_exact": turnover, "by_bps": rows}

    # rolling 36-month excess return over BIL, month-end, per sleeve (fair and house)
    for cash, f in (("fair", fair), ("house", house)):
        for s in ("eom_flow", "etf_dv2", "core5", "btal_qqq", "disp"):
            r = f[s].loc[LONG_START:END]
            m = (1 + r).resample("ME").prod() - 1
            mb = (1 + tb.loc[LONG_START:END]).resample("ME").prod() - 1
            roll = ((1 + m).rolling(36).apply(np.prod, raw=True) ** (1 / 3) - (1 + mb).rolling(36).apply(np.prod, raw=True) ** (1 / 3))
            vol = m.rolling(36).std() * np.sqrt(12)
            out["rolling"][f"{cash}|{s}"] = {"dates": [d.strftime("%Y-%m") for d in roll.dropna().index],
                                             "xs": [round(float(v), 5) for v in roll.dropna()],
                                             "xs_sharpe": [round(float(v), 4) for v in (roll / vol).dropna()]}

    # crises and calendar years (fair)
    cof = lib.cofall_windows(data["bench"], LONG_START, END)
    out["crises"] = {b: sc.crisis_rows(series["fair"][b], data, cof) for b in BOOKS}
    out["years"] = {b: sc.calendar_excess(series["fair"][b], tb) for b in BOOKS}
    # capacity, and a weight plateau for the new candidates (fair and house)
    import part_m  # noqa: PLC0415
    base_t = pd.read_csv(dv.OUT / "main_books_sharpe.csv", index_col=0)
    products = {}
    for b in [BOTH_EQ, DV2_IV, EOM_IV, sc.A1_PICK, sc.CHAMPION]:
        w = json.loads(base_t.at[b, "avg_weights"])
        products[b] = {p: v / sum(w.values()) for p, v in w.items()}
    excess = {b: lib.excess_cagr(lib.book_returns(data["cash_exact"], lib.Book(b, tuple(w), "EQ", w), EXACT_START),
                                 data["cash_exact"][TBILL], idx) for b, w in products.items()}
    part_m.capacity(products, data, excess).to_csv(OUT / "capacity_candidates.csv", float_format="%.6g")
    prow = []
    for b in [BOTH_EQ, DV2_IV]:
        variants = sc.plateau_books(by[b], products[b])
        for cash, f in (("fair", fair), ("house", house)):
            P = pd.DataFrame({v.name: dv.book_series(f, v) for v in variants})
            P[sc.CHAMPION] = dv.book_series(f, dv.family("MAIN")[0])
            shb, _, ddb = sc.boot_sharpe_dd(P.to_numpy(), tb.reindex(P.index).to_numpy())
            jc = list(P.columns).index(sc.CHAMPION)
            for j, name in enumerate(P.columns[:-1]):
                prow.append({"book": name, "cash": cash, "sharpe": sc.sharpe(P[name]), "maxdd": lib.maxdd(P[name]),
                             "p_breach10": float((ddb[:, j] < dv.DD_LIMIT).mean()),
                             "sharpe_beats_champion": float(np.mean(shb[:, j] > shb[:, jc]))})
    pd.DataFrame(prow).to_csv(OUT / "plateau_candidates.csv", index=False, float_format="%.6g")
    (OUT / "candidate_dossier.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")

    # console summary
    for w in WINDOWS:
        print(w.ljust(14), " | ".join(f"{b.split(' [')[0][-16:]}{b[-5:]} {out['windows'][f'fair|{b}|{w}']['sharpe']:.2f}/"
                                      f"{out['windows'][f'fair|{b}|{w}']['cagr']*100:.1f}%" for b in BOOKS))
    for b in [DV2_IV, EOM_IV]:
        for w in WINDOWS:
            s = out["slot"][f"fair|{b}|{w}"][THIRD[b]]
            print("slot", b[-25:], w.ljust(14), f"with {s['with']:.2f}  T-bills instead {s['tbills_instead']:.2f}")
    print(json.dumps(out["forward_rank"], indent=0))
    print(json.dumps({b: {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in c["by_bps"].items()} | {"turnover": round(c["turnover_per_year_exact"], 1)}
                      for b, c in out["cost"].items()}, indent=0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
