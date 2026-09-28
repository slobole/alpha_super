"""C2: the current target book (decision 2026-09-25) held in raw whole shares at USD 30K; worst case per name."""
import json, pickle
import numpy as np, pandas as pd
import core5_common as c
df = c.load_pricing()
with (c.CACHE / "baseline_strategy.pkl").open("rb") as fh:
    base = pickle.load(fh)
w = base["daily"].iloc[-1]
T = base["daily"].index[-1]
out = {"decision_date": str(T.date()), "rows": {}}
for nav in (30_000.0, 100_000.0):
    rows = {}
    for a in c.TRADEABLES:
        px = float(df.loc[T, (a, "Unadjusted Close")])
        q = int(nav * float(w[a]) / px)
        rows[a] = {"target_w": float(w[a]), "price": px, "shares": q, "held_w": q * px / nav,
                   "err_pct_nav": 100 * (float(w[a]) - q * px / nav), "worst_case_err_pct_nav": 100 * px / nav}
    out["rows"][str(int(nav))] = rows
    out[f"cash_drag_pct_{int(nav)}"] = sum(r["err_pct_nav"] for r in rows.values())
# DBC short at 30K when active (10% cap): shares
out["dbc_short_10pct_at_30k_shares"] = int(-0.10 * 30_000 / float(df.loc[T, ("DBC", "Unadjusted Close")]))
print(json.dumps(out, indent=1)); c.dump(out, "c2_current_book_30k.json")
