"""Dividend stamping check (A8): is Norgate's CAPITALSPECIAL 'Dividend' stamped on the last cum-dividend session T
(the engine's entitlement convention: credited/debited before the open of T+1 = ex-date)?

For a dividend D stamped on session d, TOTALRETURN back-adjusts every price BEFORE the ex-date, so the ratio
R_t = TR_t / CS_t jumps between the last cum session and the ex-date by about D / Close_cum.
  jump_after  = R_(d+1) / R_d - 1      (expected ~ D / CS_d if d is the last cum session)
  jump_before = R_d / R_(d-1) - 1      (expected ~ D / CS_(d-1) if d were the ex-date)

Usage: uv run python tb_divstamp.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import tb_common as tc
from data.norgate_loader import norgatedata

SYMBOLS = ("TLT", "SPY", "XLK", "XLE", "VOX", "IYR", "KIE", "IHI", "SOXX", "IGV", "IBB", "XLC")


def _load(sym, adj):
    return norgatedata.price_timeseries(sym, stock_price_adjustment_setting=getattr(norgatedata.StockPriceAdjustmentType, adj),
                                        padding_setting=norgatedata.PaddingType.NONE, start_date="2002-01-01",
                                        end_date=tc.END_STR, timeseriesformat="pandas-dataframe")


def main() -> None:
    out = {}
    for sym in SYMBOLS:
        cs, tr = _load(sym, "CAPITALSPECIAL"), _load(sym, "TOTALRETURN")
        ratio = (tr["Close"].astype(float) / cs["Close"].astype(float)).reindex(cs.index)
        div = cs["Dividend"].astype(float)
        rows = []
        for d in div.index[div > 0]:
            i = cs.index.get_loc(d)
            if i == 0 or i + 1 >= len(cs.index):
                continue
            expected = float(div.loc[d] / cs["Close"].iloc[i])
            after = float(ratio.iloc[i + 1] / ratio.iloc[i] - 1)
            before = float(ratio.iloc[i] / ratio.iloc[i - 1] - 1)
            rows.append({"date": d, "expected": expected, "after": after, "before": before})
        frame = pd.DataFrame(rows)
        if frame.empty:
            out[sym] = {"events": 0}
            continue
        ok_after = (frame["after"] - frame["expected"]).abs() <= 0.25 * frame["expected"].abs() + 1e-6
        ok_before = (frame["before"] - frame["expected"]).abs() <= 0.25 * frame["expected"].abs() + 1e-6
        out[sym] = {"events": int(len(frame)), "jump_on_next_session_matches": int(ok_after.sum()),
                    "jump_on_stamp_session_matches": int(ok_before.sum()),
                    "median_ratio_after_over_expected": float((frame["after"] / frame["expected"]).median()),
                    "first": str(frame["date"].iloc[0].date()), "last": str(frame["date"].iloc[-1].date())}
        if sym == "TLT":
            # TLT ex-dates vs the EOM short window: day-of-month distribution of stamp sessions
            out["TLT_stamp_day_of_month_counts"] = frame["date"].dt.day.value_counts().sort_index().to_dict()
        print(sym, out[sym], flush=True)
    tc.write_json("dividend_stamping.json", out)


if __name__ == "__main__":
    main()
