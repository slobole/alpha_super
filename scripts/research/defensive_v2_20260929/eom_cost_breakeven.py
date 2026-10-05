"""A3 follow-up (descriptive): how much extra cost per side on EOM's own fills the EOM books can take before they stop
beating the 60/40 champion; EOM's annual turnover. Fair cash, LONG window.

Usage: python eom_cost_breakeven.py
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
from lib import END, LONG_START, TBILL  # noqa: E402
import defensive_v2 as dv  # noqa: E402
import sharpe_checks as sc  # noqa: E402

BOOKS = ["CORE5 + BTAL_QQQ + EOM [IV]", "CORE5 + EOM + DISP [IV]", sc.CHAMPION]


def main() -> int:
    data = lib.load_inputs()
    fair = data["cash_long"]
    tb = fair[TBILL]
    by = {b.name: b for b in dv.family("MAIN")}
    tx, nav = data["tx"]["eom_flow"], data["nav"]["eom_flow"]
    t = tx[(tx["date"] >= LONG_START) & (tx["date"] <= END)]
    notional = t["signed_notional_float"].abs().groupby(t["date"]).sum()
    turnover = float((notional / nav.shift(1).reindex(notional.index)).sum() / ((END - LONG_START).days / 365.25))
    rows = []
    for bps in (0, 2, 5, 10, 15, 20, 30):
        f = fair.copy()
        if bps:
            drag = lib.evaluation.extra_slippage_cost_ser(tx, nav, bps / 1e4).reindex(f.index).fillna(0.0)
            live = f["eom_flow"].notna()
            f.loc[live, "eom_flow"] = f.loc[live, "eom_flow"] - drag[live]
        for b in BOOKS:
            r = dv.book_series(f, by[b]) if b != sc.CHAMPION else dv.book_series(f, dv.family("MAIN")[0])
            rows.append({"bps_per_side": bps, "book": b, "sharpe": sc.sharpe(r), "xs_sharpe": sc.xs_sharpe(r, tb),
                         "cagr": lib.cagr(r, lib.base_date(data["index"], r)), "maxdd": lib.maxdd(r)})
    out = pd.DataFrame(rows)
    OUT = dv.OUT / "sharpe_checks"
    out.to_csv(OUT / "eom_cost_breakeven.csv", index=False, float_format="%.6g")
    (OUT / "eom_turnover.json").write_text(json.dumps({"eom_turnover_per_year": turnover}), encoding="utf-8")
    print(f"EOM one-way turnover per year (traded notional / NAV): {turnover:.1f}x")
    print(out.pivot(index="bps_per_side", columns="book", values=["sharpe", "xs_sharpe"]).round(3).to_string())

    # weight plateau of the recommended book (same method as sharpe_checks), fair and house cash
    base = by[BOOKS[0]]
    w = {"core5": 0.4703, "btal_qqq": 0.2797, "eom_flow": 0.2500}
    variants = sc.plateau_books(base, w)
    prow = []
    for cash, frame in (("fair", fair), ("house", data["long"])):
        P = pd.DataFrame({v.name: dv.book_series(frame, v) for v in variants})
        P[sc.CHAMPION] = dv.book_series(frame, dv.family("MAIN")[0])
        sh, xs, dd = sc.boot_sharpe_dd(P.to_numpy(), tb.reindex(P.index).to_numpy())
        jc = list(P.columns).index(sc.CHAMPION)
        for j, name in enumerate(P.columns):
            prow.append({"book": name, "cash": cash, "sharpe": sc.sharpe(P[name]), "xs_sharpe": sc.xs_sharpe(P[name], tb),
                         "maxdd": lib.maxdd(P[name]), "p_breach10": float((dd[:, j] < dv.DD_LIMIT).mean()),
                         "sharpe_beats_champion": float(np.mean(sh[:, j] > sh[:, jc]))})
    pd.DataFrame(prow).to_csv(OUT / "plateau_runner_up.csv", index=False, float_format="%.6g")
    print(pd.DataFrame(prow).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
