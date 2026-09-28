"""Full-history runs for the DV2 / QPI readiness audit (research-only).

Arms (all Vanilla engine, calendar 2004-01-01 -> 2026-09-25, full pre-start history from 1998 for warm-up):
  <fam>_base         production accounting, $100k, trimmed universe (= production run_variant), recording
  <fam>_base_rep     identical re-run (A10 determinism; plain class, so also proves the recorder is inert)
  <fam>_untrimmed    universe without the 5-session trim = the membership the live host sees at T (A6 / B1)
  <fam>_10m          production accounting at $10M (A9 capital scaling)
  <fam>_hsu_100k     historical_share_units_bool=True at $100k (raw whole shares, raw-equivalent fees; A8)
  <fam>_hsu_30k      historical_share_units_bool=True at $30k (owner size, raw whole shares; C2)

Usage: uv run python aud_full_runs.py <arm>
"""

from __future__ import annotations

import json
import pickle
import sys
import time

import numpy as np
import pandas as pd

import aud_common as ac
import aud_data

ARMS = {}
for fam in ("dv2", "qpi"):
    ARMS[f"{fam}_base"] = dict(fam=fam, universe="trimmed", capital=100_000.0, hsu=False, record=True)
    ARMS[f"{fam}_base_rep"] = dict(fam=fam, universe="trimmed", capital=100_000.0, hsu=False, record=False)
    ARMS[f"{fam}_untrimmed"] = dict(fam=fam, universe="untrimmed", capital=100_000.0, hsu=False, record=True)
    ARMS[f"{fam}_10m"] = dict(fam=fam, universe="trimmed", capital=10_000_000.0, hsu=False, record=False)
    ARMS[f"{fam}_hsu_100k"] = dict(fam=fam, universe="trimmed", capital=100_000.0, hsu=True, record=False)
    ARMS[f"{fam}_hsu_30k"] = dict(fam=fam, universe="trimmed", capital=30_000.0, hsu=True, record=True)
    # Owner size without 22 years of compounding: last 3 years only, raw whole shares, $30k vs $10M.
    ARMS[f"{fam}_hsu_30k_l3y"] = dict(fam=fam, universe="trimmed", capital=30_000.0, hsu=True, record=True,
                                      start="2023-09-25")
    ARMS[f"{fam}_hsu_10m_l3y"] = dict(fam=fam, universe="trimmed", capital=10_000_000.0, hsu=True, record=False,
                                      start="2023-09-25")
    # Decomposition: same two arms with zero commission (isolates whole-share rounding from the $1 minimum fee).
    ARMS[f"{fam}_hsu_30k_l3y_nofee"] = dict(fam=fam, universe="trimmed", capital=30_000.0, hsu=True, record=False,
                                            start="2023-09-25", nofee=True)
    ARMS[f"{fam}_hsu_10m_l3y_nofee"] = dict(fam=fam, universe="trimmed", capital=10_000_000.0, hsu=True,
                                            record=False, start="2023-09-25", nofee=True)


def main(arm: str) -> None:
    cfg = ARMS[arm]
    data = aud_data.load()
    pricing = data["pricing_df"]
    universe = data["universe_trimmed"] if cfg["universe"] == "trimmed" else data["universe_untrimmed"]
    strategy = ac.make(cfg["fam"], universe, capital=cfg["capital"], record=cfg["record"])
    strategy.historical_share_units_bool = cfg["hsu"]
    if cfg.get("nofee"):
        strategy._commission_per_share = 0.0
        strategy._commission_minimum = 0.0
    t0 = time.time()
    ac.run(strategy, pricing, cfg.get("start", ac.FULL_START), ac.STUDY_END)
    runtime = time.time() - t0
    out = ac.OUT / "full_runs" / arm
    out.mkdir(parents=True, exist_ok=True)
    res = strategy.results.copy()
    res[["total_value", "cash", "portfolio_value"]].astype(float).to_csv(out / "daily.csv.gz")
    tx = strategy.get_transactions().copy()
    tx.to_csv(out / "transactions.csv.gz", index=False)
    np.save(out / "total_value.npy", res["total_value"].astype(float).to_numpy())
    if cfg["record"]:
        with (out / "decision_log.pkl").open("wb") as handle:
            pickle.dump(strategy.decision_log, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tv = res["total_value"].astype(float)
    notional = (tx["amount"].astype(float) * tx["price"].astype(float)).abs().sum()
    summary = {
        "arm": arm, **{k: v for k, v in cfg.items()},
        "full": ac.metrics(tv),
        "post_2012_10_02": ac.metrics(tv, start="2012-10-02"),
        "last_3y": ac.metrics(tv, start="2023-09-25") if len(tv.loc["2023-09-25":]) > 2 else None,
        "total_commission": float(tx["commission"].astype(float).sum()),
        "gross_notional": float(notional),
        "n_transactions": int(len(tx)),
        "n_synthetic_liquidations": int((tx["order_id"].astype(int) == -1).sum()),
        "min_cash_over_nav": float((res["cash"].astype(float) / tv).min()),
        "n_days_negative_cash": int((res["cash"].astype(float) < -1e-6).sum()),
        "final_value": float(tv.iloc[-1]),
        "missing_open_cancels": strategy._captured_stdout.count("no tradable open"),
        "missing_price_liquidations": strategy._captured_stdout.count("liquidating at last available close"),
        "accounting_policy": {k: str(v) for k, v in strategy._accounting_policy_dict.items()},
        "runtime_s": round(runtime, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "stdout.txt").write_text(strategy._captured_stdout, encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main(sys.argv[1])
