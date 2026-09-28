"""Full-history HPI runs on real Norgate data (production path of run_hpi_variant), for A8/A9/A10 and as the
reference state for the live replay (B1) and tradability (C).

Usage: uv run python hpi_full_runs.py <variant> <arm>
  variant in {vote, baseline}
  arm in {base, base_repeat, cap30k, cap1m, cap10m, hsu}
Outputs under results/research/strategy_readiness_audit_20260928/hpi/full_runs/<variant>_<arm>/
"""

from __future__ import annotations

import pickle
import sys
import time

import pandas as pd

import hpi_common as hc

ARMS = {
    "base": dict(capital=100_000.0, hsu=False),
    "base_repeat": dict(capital=100_000.0, hsu=False),
    "cap30k": dict(capital=30_000.0, hsu=False),
    "cap1m": dict(capital=1_000_000.0, hsu=False),
    "cap10m": dict(capital=10_000_000.0, hsu=False),
    "hsu": dict(capital=100_000.0, hsu=True),
    "cap30k_hsu": dict(capital=30_000.0, hsu=True),
}


def main(variant: str, arm: str) -> None:
    cfg = ARMS[arm]
    data = hc.load_full_inputs()
    pricing_df = hc.set_adjustment_attrs(data["pricing_df"])
    strategy = hc.make_strategy(variant, data["universe"], capital=cfg["capital"], hsu=cfg["hsu"])
    t0 = time.time()
    hc.run(strategy, pricing_df, hc.START)
    elapsed = time.time() - t0
    strategy.universe_df = None
    out = hc.OUT / "full_runs" / f"{variant}_{arm}"
    out.mkdir(parents=True, exist_ok=True)
    res = strategy.results.copy()
    res.index = pd.to_datetime(res.index)
    res[["portfolio_value", "cash", "total_value"]].to_csv(out / "daily.csv.gz")
    strategy.get_transactions().to_csv(out / "transactions.csv.gz", index=False)
    with (out / "records.pkl").open("wb") as handle:
        pickle.dump(strategy.records, handle, protocol=pickle.HIGHEST_PROTOCOL)
    (out / "stdout.txt").write_text(strategy.captured_stdout, encoding="utf-8")
    try:
        strategy.get_dividend_ledger().to_csv(out / "dividends.csv.gz", index=False)
    except Exception as exc:  # pragma: no cover - diagnostic only
        print("dividend ledger unavailable", exc)
    m = hc.metrics(res["total_value"])
    tx = strategy.get_transactions()
    summary = {
        "variant": variant, "arm": arm, **cfg, "elapsed_s": round(elapsed, 1), **m,
        "n_fills": int(len(tx)), "commission_total": float(tx["commission"].astype(float).sum()),
        "dividend_net_total": float(getattr(strategy, "dividend_cash_net_total_float", 0.0)),
        "dividend_gross_total": float(getattr(strategy, "dividend_cash_gross_total_float", 0.0)),
        "accounting_policy": dict(strategy._accounting_policy_dict),
        "data_adjustment_policy": {k: str(v) for k, v in strategy._data_adjustment_policy_dict.items()},
        "final_total_value": float(res["total_value"].iloc[-1]),
        "min_cash_pct_nav": float((res["cash"] / res["total_value"]).min()),
    }
    hc.dump_json(summary, f"full_runs/{variant}_{arm}/summary.json")
    print(summary)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
