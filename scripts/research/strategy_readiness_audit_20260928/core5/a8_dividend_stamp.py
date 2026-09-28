"""A8: which session does Norgate stamp Dividend on (ex-date vs the session before)? Compare with TR/CAPITALSPECIAL ratio jumps."""
import json
import numpy as np, pandas as pd
import core5_common as c
df = c.load_pricing()
out = {}
for sym in ("SPY", "IEF", "DBC", "UUP", "GLD"):
    div = df[(sym, "Dividend")].astype(float)
    ratio = df[(f"ADAPTIVE_TR_{sym}", "Close")].astype(float) / df[(sym, "Close")].astype(float)
    jump = ratio.pct_change().abs() > 1e-5
    stamp = div[div > 0].index
    jump_days = ratio.index[jump.fillna(False).to_numpy()]
    same = sum(1 for d in stamp if d in set(jump_days))
    nextday = sum(1 for d in stamp if df.index[df.index.get_loc(d) + 1] in set(jump_days)) if len(stamp) else 0
    out[sym] = {"n_div_rows": int(len(stamp)), "ratio_jump_on_stamp_row": same, "ratio_jump_on_next_row": nextday,
                "last5_stamps": [str(d.date()) for d in stamp[-5:]]}
bil = df[("BIL", "Dividend")].astype(float)
bs = bil[bil > 0].index
out["BIL"] = {"n_div_rows": int(len(bs)), "last8_stamps": [str(d.date()) for d in bs[-8:]],
              "share_of_stamps_on_first_session_of_month": float(np.mean([d.to_period("M") != df.index[df.index.get_loc(d) - 1].to_period("M") for d in bs])),
              "share_of_stamps_on_last_session_of_month": float(np.mean([d.to_period("M") != df.index[df.index.get_loc(d) + 1].to_period("M") for d in bs if df.index.get_loc(d) + 1 < len(df.index)]))}
print(json.dumps(out, indent=1))
c.dump(out, "a8_dividend_stamp.json")
