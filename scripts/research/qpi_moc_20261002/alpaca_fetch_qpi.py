"""Fetch the 15:45 state for (date, symbol) pairs the QPI test needs and the DV2 study did not download.

Same definitions as scripts/research/dv2_deep_20260925/alpaca_fetch.py (bars ending at or before
scheduled close - 15 min, SIP feed, raw prices, official close from the daily bar).
Input: results/research/qpi_moc_20261002/needed_missing.npz (rows, cols) from needed_pairs.py.
Output: results/research/qpi_moc_20261002/alpaca/sessions/<date>.csv.gz (resumable; one file per session).
Usage: python alpaca_fetch_qpi.py [worker_index worker_count reverse(0|1)]
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qpi_lib as q  # noqa: E402
import alpaca_fetch as af  # noqa: E402  (dv2_deep_20260925, on sys.path via qpi_lib)

OUT = q.OUT / "alpaca" / "sessions"
NY = af.NY


def fetch_session(d, row, syms_by_alp):
    open_ts = pd.Timestamp(f"{d.date()} {row['open']}", tz=NY)
    close_ts = pd.Timestamp(f"{d.date()} {row['close']}", tz=NY)
    cutoff = close_ts - pd.Timedelta(minutes=15)
    syms = sorted(syms_by_alp)
    rows = []
    for chunk in [syms[i:i + 500] for i in range(0, len(syms), 500)]:
        base = {"symbols": ",".join(chunk), "feed": "sip", "adjustment": "raw", "asof": str(d.date()), "limit": 10000}
        # *** CRITICAL*** only bars that END at or before the 15:45 cutoff
        b30 = af.bars({**base, "timeframe": "30Min", "start": open_ts.tz_convert("UTC").isoformat(),
                       "end": (cutoff - pd.Timedelta(minutes=31)).tz_convert("UTC").isoformat()})
        b15 = af.bars({**base, "timeframe": "15Min", "start": (cutoff - pd.Timedelta(minutes=15)).tz_convert("UTC").isoformat(),
                       "end": (cutoff - pd.Timedelta(minutes=14)).tz_convert("UTC").isoformat()})
        day = af.bars({**base, "timeframe": "1Day", "start": str(d.date()), "end": str(d.date())})
        recs = [(s_, x["t"], 30, x["h"], x["l"], x["c"], x["v"]) for s_, lst in b30.items() for x in lst]
        recs += [(s_, x["t"], 15, x["h"], x["l"], x["c"], x["v"]) for s_, lst in b15.items() for x in lst]
        bdf = pd.DataFrame(recs, columns=["s", "t", "m", "h", "l", "c", "v"])
        agg = last15 = None
        if len(bdf):
            bdf["t"] = pd.to_datetime(bdf["t"], utc=True).dt.tz_convert(NY)
            bdf = bdf[(bdf["t"] >= open_ts) & (bdf["t"] + pd.to_timedelta(bdf["m"], unit="min") <= cutoff)]
            last15 = bdf[bdf["m"] == 15].set_index("s")
            agg = bdf.groupby("s").agg(h1545=("h", "max"), l1545=("l", "min"), v1545=("v", "sum"), n_bars=("t", "size"))
        for s_ in chunk:
            rec = {"alpaca_symbol": s_}
            if agg is not None and s_ in agg.index and s_ in last15.index:
                lb = last15.loc[[s_]].iloc[-1]
                rec.update(p1545=float(lb["c"]), h1545=float(agg.at[s_, "h1545"]), l1545=float(agg.at[s_, "l1545"]),
                           v1545=float(agg.at[s_, "v1545"]), n_bars=int(agg.at[s_, "n_bars"]))
            db = (day.get(s_) or [None])[-1]
            if db:
                rec.update(close_official=float(db["c"]), day_high=float(db["h"]), day_low=float(db["l"]), day_volume=float(db["v"]))
            for ns in syms_by_alp[s_]:
                rows.append({**rec, "norgate_symbol": ns})
    return pd.DataFrame(rows).assign(date=str(d.date()), cutoff=str(cutoff))


def main(wi=0, wn=1, reverse=0):
    p = q.rp.Panel("sp500")
    z = np.load(q.OUT / "needed_missing.npz")
    rows, cols = z["rows"], z["cols"]
    OUT.mkdir(parents=True, exist_ok=True)
    cal = pd.read_csv(q.REPO / "results/research/dv2_deep_20260925/alpaca/calendar.csv", dtype=str)
    cal["date"] = pd.to_datetime(cal["date"])
    cal = cal.set_index("date")
    by_row = pd.Series(cols).groupby(rows).apply(list)
    items = list(by_row.items())
    for k, (t, cl) in enumerate(items[::-1] if reverse else items):
        if k % wn != wi:
            continue
        d = p.dates[t]
        f = OUT / f"{d.date()}{os.environ.get('QPI_FETCH_SUFFIX', '')}.csv.gz"
        if f.exists() or d not in cal.index:
            continue
        m = {}
        for i in cl:
            m.setdefault(af.alpaca_symbol(p.symbols[i]), []).append(p.symbols[i])
        fetch_session(d, cal.loc[d], m).to_csv(f, index=False, compression="gzip")
        print(d.date(), len(cl), flush=True)


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:4]))
