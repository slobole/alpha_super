"""Stage 1, family B: forced-flow rebound after index exits (event study; SPEC_FROZEN.md).

Event: raw Norgate membership flag 1 on day d and 0 on day d+1, and the stock trades on d+1 (amendment A2).
Primary timing (conservative, membership-file only): known after Close_{d+1}, entry Open_{d+2}.
Diagnostic timing: entry Open_{d+1} (index changes are announced before the effective date).

    CAR_h = O_{e+h} / O_e - 1 - b * (SPY_{e+h} / SPY_e - 1),   e = entry row,   b = 252-day shrunk beta to SPY
    (a missing exit open uses the last finite close on or before e+h)

Control: additions (flag 0 -> 1). Statistics: event mean, t with standard errors clustered by event date.

Usage: python stage1_b.py [events|stats]      (events: build events.csv; stats: screen table event_stats.csv)
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

HERE_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE_PATH))
import features as ft  # noqa: E402
from data.norgate_loader import _load_direct_norgate_module, load_price_timeseries  # noqa: E402

OUT_PATH = ft.STUDY_OUT_PATH / "stage1_b"
INDEX_DICT = {"sp500": "S&P 500", "sp400": "S&P MidCap 400", "sp600": "S&P SmallCap 600", "ndx": "Nasdaq 100",
              "r1000": "Russell 1000", "r2000": "Russell 2000"}
FAMILY_DICT = {"sp500": "sp", "sp400": "sp", "sp600": "sp", "r1000": "russell", "r2000": "russell", "ndx": "ndx"}
SIZE_ETF_DICT = {"sp500": "SPY", "ndx": "SPY", "sp400": "MDY", "r1000": "MDY", "sp600": "IWM", "r2000": "IWM"}
HORIZON_LIST = [5, 10, 20, 40]


def membership(nd, index_str: str) -> dict[str, pd.Series]:
    out = {}
    for sym in nd.watchlist_symbols(f"{index_str} Current & Past"):
        ts = nd.index_constituent_timeseries(sym, index_str, timeseriesformat="pandas-dataframe")
        s = ts.iloc[:, 0].astype(np.int8)
        if s.sum() > 0:
            out[sym] = s
    return out


def transitions(s: pd.Series) -> tuple[list, list]:
    v = s.to_numpy()
    d = np.diff(v)
    idx = s.index
    exits = [idx[i + 1] for i in np.nonzero(d == -1)[0]]      # first day out
    adds = [idx[i + 1] for i in np.nonzero(d == 1)[0]]        # first day in
    return exits, adds


def main():
    started = time.time()
    nd = _load_direct_norgate_module()
    etf = ft.Panel("etfx")
    cal = etf.dates
    spy_o = np.asarray(etf.O[:, etf.col("SPY")])
    spy_c = np.asarray(etf.C[:, etf.col("SPY")])
    size_open = {s: np.asarray(etf.O[:, etf.col(s)]) for s in ["SPY", "MDY", "IWM"]}
    members = {}
    for key, name in INDEX_DICT.items():
        members[key] = membership(nd, name)
        print(key, len(members[key]), "members-ever", round(time.time() - started, 1), flush=True)
    # on-date membership lookup for exit typing (joined a sibling index on the same date?)
    def is_member(key, sym, date):
        s = members[key].get(sym)
        if s is None or date not in s.index:
            return False
        return bool(s.loc[date] == 1)
    sibling = {"sp500": ["sp400", "sp600"], "sp400": ["sp500", "sp600"], "sp600": ["sp500", "sp400"],
               "r1000": ["r2000"], "r2000": ["r1000"], "ndx": []}
    events = []
    for key in INDEX_DICT:
        for sym, s in members[key].items():
            ex, ad = transitions(s)
            for d1 in ex:
                to = [k for k in sibling[key] if is_member(k, sym, d1)]
                events.append((key, sym, "exit", d1, to[0] if to else "none"))
            for d1 in ad:
                fr = [k for k in sibling[key] if is_member(k, sym, s.index[s.index.get_loc(d1) - 1])]
                events.append((key, sym, "add", d1, fr[0] if fr else "none"))
    ev = pd.DataFrame(events, columns=["index", "symbol", "kind", "first_day", "other"])
    print("raw events", len(ev), round(time.time() - started, 1), flush=True)
    price_cache = {}
    rows = []
    for sym, grp in ev.groupby("symbol"):
        try:
            px = load_price_timeseries(sym, start_date_str="1989-01-01")
        except Exception:
            continue
        if px is None or len(px) == 0:
            continue
        px = px.reindex(cal)
        O, C = px["Open"].to_numpy(dtype=float), px["Close"].to_numpy(dtype=float)
        raw = px["Unadjusted Close"].to_numpy(dtype=float) if "Unadjusted Close" in px else np.full(len(cal), np.nan)
        vol = px["Volume"].to_numpy(dtype=float)
        last_close = pd.Series(C).ffill().to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            r = C / np.r_[np.nan, C[:-1]] - 1.0
        spy_r = spy_c / np.r_[np.nan, spy_c[:-1]] - 1.0
        for _, e in grp.iterrows():
            d1 = cal.get_indexer([e.first_day])[0]
            if d1 < 260 or d1 + 2 + max(HORIZON_LIST) >= len(cal):
                continue
            # *** CRITICAL*** only what is known at the decision: the stock traded on d+1 (A2). Later delistings stay
            # in the sample and exit at their last finite close (engine-like), so no survivorship filter is applied.
            if not np.isfinite(C[d1]):
                continue
            # 252-day beta to SPY measured through the last day in (d = d1 - 1)
            win = slice(d1 - 252, d1)
            y, x = r[win], spy_r[win]
            ok = np.isfinite(y) & np.isfinite(x)
            if ok.sum() >= 200:
                b = np.cov(y[ok], x[ok])[0, 1] / np.var(x[ok], ddof=1)
                b = float(np.clip(0.67 * b + 0.33, 0.3, 2.0))
            else:
                b = 1.0
            adv = np.nanmean((raw * vol)[d1 - 63: d1])
            pre20 = C[d1 - 1] / C[d1 - 21] - 1.0 if np.isfinite(C[d1 - 21]) else np.nan
            row = {"index": e["index"], "symbol": sym, "kind": e.kind, "other": e.other, "first_day": e.first_day,
                   "beta": b, "adv63": adv, "raw_close": raw[d1 - 1], "ret_pre20": pre20,
                   "spy_pre20": spy_c[d1 - 1] / spy_c[d1 - 21] - 1.0}
            for tag, e0 in (("d2", d1 + 1), ("d1", d1)):  # entry row e0: Open_{d+2} (primary) or Open_{d+1}
                if not np.isfinite(O[e0]):
                    continue
                for hz in HORIZON_LIST:
                    xr = O[e0 + hz] if np.isfinite(O[e0 + hz]) else last_close[e0 + hz]
                    raw_ret = xr / O[e0] - 1.0
                    spy_ret = spy_o[e0 + hz] / spy_o[e0] - 1.0
                    so = size_open[SIZE_ETF_DICT[e["index"]]]
                    size_ret = so[e0 + hz] / so[e0] - 1.0 if np.isfinite(so[e0]) and np.isfinite(so[e0 + hz]) else np.nan
                    row[f"raw_{tag}_{hz}"] = raw_ret
                    row[f"car_{tag}_{hz}"] = raw_ret - b * spy_ret
                    row[f"size_{tag}_{hz}"] = raw_ret - size_ret
            rows.append(row)
    out = pd.DataFrame(rows)
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH / "events.csv", index=False)
    print("events kept", len(out), round(time.time() - started, 1), flush=True)


def cluster_t(x: np.ndarray, groups: np.ndarray) -> tuple[float, float, int]:
    """Event mean and t with standard errors clustered by event date (CR0)."""
    ok = np.isfinite(x)
    x, groups = x[ok], groups[ok]
    if len(x) < 10:
        return np.nan, np.nan, len(x)
    e = x - x.mean()
    s = pd.Series(e).groupby(groups).sum()
    se = np.sqrt((s ** 2).sum()) / len(x)
    return x.mean(), x.mean() / se, len(x)


def stats():
    """Screen table (SPEC Stage 1 B): mean SPY-hedged CAR, clustered t, t of (CAR - 30 bps), both halves."""
    ev = pd.read_csv(OUT_PATH / "events.csv", parse_dates=["first_day"])
    periods = {"MAIN": ("2000-01-01", "2026-08-19"), "H1_00_12": ("2000-01-01", "2012-12-31"),
               "H2_13_26": ("2013-01-01", "2026-08-19"), "HO_91_99": ("1991-01-01", "1999-12-31")}
    rows = []
    for (idx, kind, other), grp in ev.groupby(["index", "kind", "other"]):
        for hz in HORIZON_LIST:
            for tag in ("d2", "d1"):
                col = f"car_{tag}_{hz}"
                for per, (a, b) in periods.items():
                    g = grp[(grp.first_day >= a) & (grp.first_day <= b)]
                    m, t, n = cluster_t(g[col].to_numpy(), g.first_day.to_numpy())
                    _, tn, _ = cluster_t(g[col].to_numpy() - 0.003, g.first_day.to_numpy())
                    rows.append((idx, kind, other, hz, tag, per, m * 1e4 if m == m else np.nan, t, tn, n))
    out = pd.DataFrame(rows, columns=["index", "kind", "other", "h", "tag", "period", "mean_bps", "t", "t_net", "n"])
    out.to_csv(OUT_PATH / "event_stats.csv", index=False)
    print("saved event_stats.csv", len(out))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        stats()
    else:
        main()
