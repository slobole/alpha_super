"""Baseline CORE5 run at cb29d4f on the local Norgate vintage; cached for later checks."""
import pickle, time, hashlib
import numpy as np, pandas as pd
import core5_common as c

df = c.load_pricing()
t0 = time.time()
s = c.run_backtest(df, 100_000.0)
el = time.time() - t0
ret = s.results["daily_returns"].astype(float)
tv = s.results["total_value"].astype(float) if "total_value" in s.results else None
tx = s.get_transactions().copy()
out = {
    "elapsed_s": el,
    "metrics_full": c.metrics(ret),
    "metrics_2012_10_02_on": c.metrics(ret, "2012-10-02"),
    "n_transactions": int(len(tx)),
    "n_rebalance_rows": int(len(s.rebalance_target_weight_df)),
    "n_daily_target_rows": int(len(s.daily_target_weights)),
    "accounting_policy": {k: v for k, v in s._accounting_policy_dict.items()},
    "dividend_gross_total": float(s.dividend_cash_gross_total_float),
    "dividend_withholding_total": float(s.dividend_withholding_total_float),
    "borrow_fee_total": float(s.borrow_fee_total_float),
    "commission_total": float(tx["commission"].sum()),
    "equity_sha256": hashlib.sha256(np.ascontiguousarray(s.results["total_value"].to_numpy(dtype=float)).tobytes()).hexdigest() if "total_value" in s.results else None,
    "results_columns": list(map(str, s.results.columns)),
}
print(out)
c.dump(out, "a_baseline.json")
with (c.CACHE / "baseline_strategy.pkl").open("wb") as fh:
    pickle.dump({"results": s.results, "tx": tx, "rebal": s.rebalance_target_weight_df, "daily": s.daily_target_weights,
                 "borrow": s.borrow_fee_df, "div": s.get_dividend_ledger(), "signal_diag": s.signal_diagnostic_df}, fh)
