"""A3 follow-up: where does prefix-run equity differ from the full run, and is it exactly the last-day borrow accrual
(the prefix calendar has no next session, so `_next_borrow_session_ts` returns None and no fee is booked at T)?"""
import pickle, json
import numpy as np, pandas as pd
import core5_common as c
df = c.load_pricing()
with (c.CACHE / "baseline_strategy.pkl").open("rb") as fh:
    base = pickle.load(fh)
bf = base["borrow"].set_index("accrual_start_date_ts")
rows = []
for d in ("2015-08-24", "2020-03-16", "2019-08-30", "2020-05-29"):
    T = pd.Timestamp(d)
    s = c.run_backtest(df.loc[:T].copy(), 100_000.0)
    full = base["results"]["total_value"].loc[:T].astype(float)
    pre = s.results["total_value"].astype(float).reindex(full.index)
    diff = (pre - full)
    nz = diff[diff != 0]
    fee_T = float(bf.loc[T, "borrow_fee_float"]) if T in bf.index else 0.0
    rows.append({"cutoff": d, "n_days_equity_differs": int(len(nz)), "days": [str(x.date()) for x in nz.index[:5]],
                 "diff_at_T": float(diff.loc[T]), "full_run_borrow_fee_booked_at_T": fee_T,
                 "diff_equals_fee": bool(abs(float(diff.loc[T]) - fee_T) < 1e-6)})
    print(rows[-1], flush=True)
c.dump(rows, "a3_equity_diff_explain.json")
