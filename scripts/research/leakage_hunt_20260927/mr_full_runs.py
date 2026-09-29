"""MR capsule leakage hunt: full-history VANILLA re-runs, one arm per process (research-only).

Arms (all $100k, 2004-01-01 .. 2026-08-19 unless stated; pricing truncated at 2026-08-19, full pre-start warm-up):
  dv2_base        production DV2 (trimmed PIT universe, adjusted-unit shares and fees)
  dv2_untrimmed   DV2 with the untrimmed PIT universe (no ``idx.iloc[:-5]``)
  dv2_hsu         DV2 with historical_share_units_bool=True (raw whole shares, raw-equivalent per-share fees)
  hpi_base        production HPI 2/3/5 vote
  hpi_hsu         HPI with historical_share_units_bool=True
  hpi_liveslot    HPI with live slot semantics (exit frees its slot only after it filled; E-03)
  dv2_trimmed_1m / dv2_untrimmed_1m  same as above at $1M from 2000-01-03 (book sleeve configuration)
  etf_base, etf_hsu  industry-ETF DV2 from 2012-01-03 (its default)

Usage: uv run python scripts/research/leakage_hunt_20260927/mr_full_runs.py <arm>
"""

from __future__ import annotations

import json
import sys
import time

import pandas as pd

import mr_common as mc
import mr_data

ARM_DICT = {
    "dv2_base": ("dv2", "trimmed", False, 100_000.0, "2004-01-01"),
    "dv2_untrimmed": ("dv2", "untrimmed", False, 100_000.0, "2004-01-01"),
    "dv2_hsu": ("dv2", "trimmed", True, 100_000.0, "2004-01-01"),
    "dv2_trimmed_1m": ("dv2", "trimmed", False, 1_000_000.0, "2000-01-03"),
    "dv2_untrimmed_1m": ("dv2", "untrimmed", False, 1_000_000.0, "2000-01-03"),
    "hpi_base": ("hpi", "exact", False, 100_000.0, "2004-01-01"),
    "hpi_hsu": ("hpi", "exact", True, 100_000.0, "2004-01-01"),
    "hpi_liveslot": ("hpi_live", "exact", False, 100_000.0, "2004-01-01"),
    "etf_base": ("etf", "history", False, 100_000.0, "2012-01-03"),
    "etf_hsu": ("etf", "history", True, 100_000.0, "2012-01-03"),
}


def main(arm: str) -> None:
    family, universe_kind, hsu, capital, start = ARM_DICT[arm]
    t0 = time.time()
    if family == "dv2":
        data = mr_data.load("dv2")
        universe = data["universe_trimmed"] if universe_kind == "trimmed" else data["universe_untrimmed"]
        strategy = mc.make_dv2(universe, capital)
    elif family in ("hpi", "hpi_live"):
        data = mr_data.load("hpi")
        cls = mc.HPILiveSlotStrategy if family == "hpi_live" else mc.hpi_mod.HPIStatefulLongStrategy
        strategy = mc.make_hpi(data["universe"], capital, start, cls=cls)
    else:
        data = mr_data.load("etf")
        strategy = mc.make_etf(data["universe"], capital)
    strategy.historical_share_units_bool = hsu
    error = None
    try:
        mc.run(strategy, data["pricing_df"], start, mc.STUDY_END, quiet=True)
    except Exception as exc:  # report exactly why an opted-in run fails
        error = f"{type(exc).__name__}: {exc}"
    out_dir = mc.OUT / "full_runs" / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stdout.txt").write_text(getattr(strategy, "_captured_stdout", ""), encoding="utf-8")
    if error is not None:
        (out_dir / "summary.json").write_text(json.dumps({"arm": arm, "error": error}, indent=2), encoding="utf-8")
        print("ERROR", error)
        return
    results = strategy.results[["total_value", "portfolio_value", "cash"]].copy()
    results.to_csv(out_dir / "daily.csv.gz", float_format="%.6f")
    tx = strategy.get_transactions().copy()
    tx.to_csv(out_dir / "transactions.csv.gz", index=False)
    trades = getattr(strategy, "_trades", None)
    if trades is not None:
        trades.to_csv(out_dir / "trades.csv.gz", index=False)
    summary = {
        "arm": arm, "capital": capital, "start": start, "end": str(mc.STUDY_END.date()), "historical_share_units": hsu,
        "full": mc.metrics(results["total_value"]),
        "post_2012_10_02": mc.metrics(results["total_value"], "2012-10-02"),
        "total_commission": float(tx["commission"].sum()),
        "gross_notional": float((tx["amount"] * tx["price"]).abs().sum()),
        "n_transactions": int(len(tx)),
        "n_synthetic_liquidations": int((tx["order_id"] == -1).sum()),
        "runtime_s": round(time.time() - t0, 1),
        "accounting_policy": {k: str(v) for k, v in getattr(strategy, "_accounting_policy_dict", {}).items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
