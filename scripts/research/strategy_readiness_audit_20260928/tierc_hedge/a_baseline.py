"""Baseline runs (full history, default config) for the three Tier C strategies; caches strategy objects."""
import pickle, time, traceback, sys
import pandas as pd
import tc_common as c

which = sys.argv[1:] or ["vixm", "trin", "ctc"]
out = {}
for name in which:
    t = time.time()
    try:
        df = {"vixm": c.load_vixm, "trin": c.load_trin, "ctc": c.load_ctc_workaround}[name]()
        s = {"vixm": c.run_vixm, "trin": c.run_trin, "ctc": c.run_ctc}[name](df)
        r = c.strat_returns(s)
        m = c.metrics(r)
        m3 = c.metrics(r, start="2023-09-25")
        tx = s.get_transactions().copy()
        res = {"full": m, "last3y": m3, "n_tx": int(len(tx)), "secs": round(time.time() - t, 1),
               "final_value": float(s.total_value_series.iloc[-1]),
               "min_cash": float(s.results["cash"].astype(float).min()) if "cash" in s.results else None,
               "commission_total": float(tx["commission"].sum()) if "commission" in tx else None}
        out[name] = res
        with (c.CACHE / f"{name}_baseline.pkl").open("wb") as fh:
            pickle.dump({"results": s.results, "tx": tx,
                         "daily_target": getattr(s, "daily_target_weights", None),
                         "rebal": getattr(s, "rebalance_target_weight_df", None),
                         "accounting": dict(getattr(s, "_accounting_policy_dict", {})),
                         "div_ledger": s.get_dividend_ledger() if hasattr(s, "get_dividend_ledger") else None,
                         "borrow": getattr(s, "borrow_fee_df", None)}, fh)
        print(name, res)
    except Exception as e:
        out[name] = {"error": repr(e)}
        traceback.print_exc()
c.dump(out, f"baseline_{'_'.join(which)}.json")
