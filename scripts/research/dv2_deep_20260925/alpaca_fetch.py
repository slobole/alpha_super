"""Download the 15:45 decision state for every S&P 500 member from Alpaca SIP (free historical data).

Per session and symbol (regular session only, New York time):
    p1545  = close of the last 15-minute bar that ends at or before cutoff = scheduled close - 15 min
    h1545  = max high of regular-session 15-minute bars ending at or before the cutoff
    l1545  = min low of the same bars
    v1545  = volume of the same bars
    close_official = Alpaca daily bar close (SIP official close), day_high / day_low / day_volume from the daily bar
All raw (unadjusted) prices; the study uses ratios to the official close, so adjustments cancel.
Output: results/research/dv2_deep_20260925/alpaca/sessions/<date>.csv.gz (resumable).
Usage: python alpaca_fetch.py START END
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = REPO / "results/research/dv2_deep_20260925/alpaca"
BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
CAL_URL = "https://paper-api.alpaca.markets/v2/calendar"
NY = "America/New_York"


def headers():
    return {"APCA-API-KEY-ID": os.environ["ALPACA_API_KEY"].strip(), "APCA-API-SECRET-KEY": os.environ["ALPACA_API_SECRET"].strip()}


def get_json(url, params, retries=8):
    last = ""
    for a in range(retries):
        try:
            r = requests.get(url, headers=headers(), params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            last = f"{r.status_code} {r.text[:200]}"
            if r.status_code not in (429, 500, 502, 503, 504):
                break
        except requests.RequestException as e:
            last = str(e)
        time.sleep(min(30, 1.5 * 2 ** a))
    raise RuntimeError(last)


def bars(params):
    out, token = {}, None
    while True:
        q = dict(params)
        if token:
            q["page_token"] = token
        js = get_json(BARS_URL, q)
        for s, lst in (js.get("bars") or {}).items():
            out.setdefault(s, []).extend(lst)
        token = js.get("next_page_token")
        if not token:
            return out


def alpaca_symbol(norgate_symbol):
    return re.sub(r"-\d{6}$", "", str(norgate_symbol).strip().upper()).replace(".", ".")


def main(start, end):
    import replica as rp
    p = rp.Panel("sp500")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sessions").mkdir(exist_ok=True)
    cal_path = OUT / "calendar.csv"
    if not cal_path.exists():
        pd.DataFrame(get_json(CAL_URL, {"start": "2015-01-01", "end": "2026-12-31"})).to_csv(cal_path, index=False)
    cal = pd.read_csv(cal_path, dtype=str)
    cal["date"] = pd.to_datetime(cal["date"])
    cal = cal[(cal["date"] >= start) & (cal["date"] <= end)]
    date_pos = {d: i for i, d in enumerate(p.dates)}
    needed = np.load(rp.CACHE_DIR_PATH / "moc_needed.npy")
    for _, row in cal.iterrows():
        d = row["date"]
        f = OUT / "sessions" / f"{d.date()}.csv.gz"
        if f.exists() or d not in date_pos:
            continue
        t = date_pos[d]
        # members on the decision date plus the previous day (held names can leave the index)
        # Names DV2 could act on at 15:45 (moc_needed.py; SPEC amendment A3 revised)
        idx = np.nonzero(needed[t])[0]
        sym_map = {}
        for i in idx:
            if np.isfinite(p.C[t, i]):
                sym_map.setdefault(alpaca_symbol(p.symbols[i]), []).append(p.symbols[i])
        syms = sorted(sym_map)
        open_ts = pd.Timestamp(f"{d.date()} {row['open']}", tz=NY)
        close_ts = pd.Timestamp(f"{d.date()} {row['close']}", tz=NY)
        cutoff = close_ts - pd.Timedelta(minutes=15)
        rows = []
        last_start = cutoff - pd.Timedelta(minutes=30)  # 15-min bar [cutoff-30, cutoff-15]? no: [cutoff-15, cutoff]
        for chunk in [syms[i:i + 500] for i in range(0, len(syms), 500)]:
            q = {"symbols": ",".join(chunk), "feed": "sip", "adjustment": "raw", "asof": str(d.date()), "limit": 10000}
            # *** CRITICAL*** 30-min bars that END at or before cutoff-15min, then the 15-min bar ending at the cutoff:
            # together they cover [open, cutoff] and nothing after the 15:45 decision time.
            b30 = bars({**q, "timeframe": "30Min", "start": open_ts.tz_convert("UTC").isoformat(),
                        "end": (cutoff - pd.Timedelta(minutes=31)).tz_convert("UTC").isoformat()})
            b15 = bars({**q, "timeframe": "15Min", "start": (cutoff - pd.Timedelta(minutes=15)).tz_convert("UTC").isoformat(),
                        "end": (cutoff - pd.Timedelta(minutes=14)).tz_convert("UTC").isoformat()})
            day = bars({**q, "timeframe": "1Day", "start": str(d.date()), "end": str(d.date())})
            recs = []
            for s_, lst in b30.items():
                for x in lst:
                    recs.append((s_, x["t"], 30, x["o"], x["h"], x["l"], x["c"], x["v"]))
            for s_, lst in b15.items():
                for x in lst:
                    recs.append((s_, x["t"], 15, x["o"], x["h"], x["l"], x["c"], x["v"]))
            bdf = pd.DataFrame(recs, columns=["s", "t", "m", "o", "h", "l", "c", "v"])
            if len(bdf):
                bdf["t"] = pd.to_datetime(bdf["t"], utc=True).dt.tz_convert(NY)
                bdf = bdf[(bdf["t"] >= open_ts) & (bdf["t"] + pd.to_timedelta(bdf["m"], unit="min") <= cutoff)]
                last15 = bdf[(bdf["m"] == 15)].set_index("s")
                agg = bdf.groupby("s").agg(h1545=("h", "max"), l1545=("l", "min"), v1545=("v", "sum"), n_bars=("t", "size"))
            for s_ in chunk:
                rec = {"alpaca_symbol": s_}
                if len(bdf) and s_ in agg.index and s_ in last15.index:
                    lb = last15.loc[[s_]].iloc[-1]
                    rec.update(p1545=float(lb["c"]), h1545=float(agg.at[s_, "h1545"]), l1545=float(agg.at[s_, "l1545"]),
                               v1545=float(agg.at[s_, "v1545"]), n_bars=int(agg.at[s_, "n_bars"]))
                db = (day.get(s_) or [None])[-1]
                if db:
                    rec.update(close_official=float(db["c"]), day_high=float(db["h"]), day_low=float(db["l"]), day_volume=float(db["v"]))
                for ns in sym_map[s_]:
                    rows.append({**rec, "norgate_symbol": ns})
        pd.DataFrame(rows).assign(date=str(d.date()), cutoff=str(cutoff)).to_csv(f, index=False, compression="gzip")
        print(d.date(), len(rows), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
