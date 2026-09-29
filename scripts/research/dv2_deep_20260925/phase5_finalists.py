"""Phase 5a: finalists (chosen from the in-sample grid by the frozen rules) + the locked 1991-1999 holdout.

Finalists (see SPEC amendment A5 for the selection reasoning, written before the holdout was run):
  F0 floor            baseline
  F1 floor_adv        floor, rank by ADV63 (non-inferior; capacity objective)
  F2 floor_adv_s15    F1 with 15 slots (post-hoc capacity combination, labelled)
  F3 E_vote           owner's ensemble idea (5 of 9 DV percentiles < 10); borderline non-inferior
  F4 w252             DV2 rank window 252 (Varadi's original length); non-inferior, all periods positive
Outputs growth-shelf style path/transaction files for the book-level evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replica as rp  # noqa: E402
import batch  # noqa: E402

OUT = batch.OUT
SRC = OUT / "sources"
FINALISTS = {
    "F0_floor": rp.Rule(floor=True),
    "F1_floor_adv": rp.Rule(floor=True, rank="adv"),
    "F2_floor_adv_s15": rp.Rule(floor=True, rank="adv", slots=15),
    "F3_E_vote": rp.Rule(floor=True, ensemble="vote"),
    "F4_w252": rp.Rule(floor=True, dv_window=252),
    "wired": rp.Rule(),
}


def write_source(alias, res: rp.Result):
    SRC.mkdir(parents=True, exist_ok=True)
    nav = pd.Series(res.nav, index=res.dates)
    tr = res.trades
    path = pd.DataFrame({"total_value_float": nav})
    path.index.name = "date"
    path.to_csv(SRC / f"{alias}__path.csv.gz")
    tx = pd.DataFrame({"source_id_str": alias, "date": tr["date"], "asset_str": tr["asset"], "amount_float": tr["amount"],
                       "fill_price_float": tr["price"], "signed_notional_float": tr["amount"] * tr["price"], "commission_float": tr["commission"]})
    tx.to_csv(SRC / f"{alias}__transactions.csv.gz", index=False)


def main():
    p = rp.Panel("sp500")
    report = {"main": {}, "holdout_1991_1999": {}, "stress_main": {}}
    for alias, rule in FINALISTS.items():
        res = rp.run(p, rule)
        report["main"][alias] = rp.summarize(res)
        write_source(alias, res)
        report["stress_main"][alias] = rp.summarize(rp.run(p, rule.with_(slippage=rule.slippage + 0.0005)))
    # *** LOCKED HOLDOUT *** run once, after the finalists above were fixed.
    base = None
    for alias, rule in FINALISTS.items():
        res = rp.run(p, rule, start="1991-01-02", end="1999-12-31")
        s = rp.summarize(res)
        report["holdout_1991_1999"][alias] = s
    b = report["holdout_1991_1999"]["F0_floor"]["sharpe"]
    for alias in FINALISTS:
        h = report["holdout_1991_1999"][alias]
        h["passes_holdout_rule"] = bool(h["sharpe"] > 0 and h["sharpe"] >= b - 0.15)
    (OUT / "phase5_finalists.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    rows = []
    for part in ("main", "stress_main", "holdout_1991_1999"):
        for a, s in report[part].items():
            rows.append({"part": part, "alias": a, **{k: s.get(k) for k in ("cagr", "sharpe", "maxdd", "calmar", "turnover_x", "P1_sharpe", "P2_sharpe", "P3_sharpe", "passes_holdout_rule")}})
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(3).to_string())


if __name__ == "__main__":
    main()
