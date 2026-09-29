"""Stage 1, family F: Treasury auction supply cycle (amendment A6 of SPEC_FROZEN.md).

Per auction day A of a bucket (LONG -> TLT, TEN -> IEF), total-return window returns with dividends:

    R(s, e) = (P_e + sum of dividends entitled on rows s .. e-1) / P_s - 1
    excess  = R - mean daily total return of the ETF in the same half * window length (sessions)

Windows: POST_OPEN_k = Open_A -> Open_{A+k}; POST_CLOSE_k = Close_A -> Close_{A+k}; PRE = Close_{A-5} -> Close_{A-1}.

Usage: python stage1_f.py download | screen | placebo | holdout | dom
  holdout: A7 as declared (raw yield change Close_{A-5} -> Close_{A-1}, 1983-01-01 -> 2002-06-30), plus a labelled
           drift-adjusted sensitivity (the first report draft quoted the sensitivity by mistake).
  dom:     post-review control: each auction's PRE return minus the mean PRE return of non-auction days on the same
           trading day of the month (the placebo only matched the month, not the position in it).
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

HERE_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE_PATH))
import features as ft  # noqa: E402

OUT_PATH = ft.STUDY_OUT_PATH / "stage1_f"
API_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query"
BUCKET_ETF = {"LONG": "TLT", "TEN": "IEF"}
HALVES = {"H1_2003_14": ("2003-01-01", "2014-12-31"), "H2_2015_26": ("2015-01-01", "2026-08-19")}
POOL = ("2003-01-01", "2026-08-19")
HURDLE = 0.0006
WINDOWS = [("POST_OPEN", 1), ("POST_OPEN", 3), ("POST_OPEN", 5), ("POST_CLOSE", 1), ("POST_CLOSE", 3), ("POST_CLOSE", 5), ("PRE", 5)]


def download() -> pd.DataFrame:
    import requests
    rows, page = [], 1
    fields = "auction_date,security_type,security_term,cusip,reopening,inflation_index_security,floating_rate,offering_amt"
    while True:
        params = {"filter": "security_type:in:(Note,Bond)", "fields": fields, "sort": "auction_date",
                  "page[size]": 1000, "page[number]": page}
        r = requests.get(API_URL, params=params, timeout=60)
        r.raise_for_status()
        d = r.json()
        rows += d["data"]
        if page >= d["meta"]["total-pages"]:
            break
        page += 1
    df = pd.DataFrame(rows)
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH / "auctions_raw.csv", index=False)
    print("downloaded", len(df), df.auction_date.min(), df.auction_date.max())
    return df


def term_years(term: str) -> float:
    m = re.match(r"\s*(\d+)-Year(?:\s+(\d+)-Month)?", str(term))
    if not m:
        return np.nan
    return int(m.group(1)) + (int(m.group(2)) / 12 if m.group(2) else 0.0)


def auction_days() -> dict[str, pd.DatetimeIndex]:
    df = pd.read_csv(OUT_PATH / "auctions_raw.csv")
    df = df[(df.inflation_index_security.astype(str) != "Yes") & (df.floating_rate.astype(str) != "Yes")]
    df["years"] = df.security_term.map(term_years)
    df["auction_date"] = pd.to_datetime(df.auction_date)
    out = {"LONG": pd.DatetimeIndex(sorted(df[df.security_type == "Bond"].auction_date.unique())),
           "TEN": pd.DatetimeIndex(sorted(df[(df.security_type == "Note") & (df.years >= 8.9)].auction_date.unique()))}
    return out


def etf_series(etf: ft.Panel, sym: str):
    c = etf.col(sym)
    O, C = np.asarray(etf.O[:, c], dtype=float), np.asarray(etf.C[:, c], dtype=float)
    D = np.nan_to_num(np.asarray(etf.DIV[:, c], dtype=float))
    return O, C, D


def window_return(O, C, D, s_row, s_px, e_row, e_px) -> float:
    """Price from s to e plus the dividends the position is entitled to.

    *** CRITICAL*** Norgate (as loaded here, matching the engine ledger) dates a dividend on the last cum-dividend
    session: Dividend_p is paid to shares held at Close_p and the price drops at Open_{p+1}. A position bought at
    Open_s or Close_s is held at Close_s; one sold at Open_e or Close_e is not held at Close_e. So the entitled rows
    are s .. e-1 for every window type (verified on SPY 2024-03-14, ex-date 2024-03-15)."""
    lo, hi = s_row, e_row - 1
    div = D[lo: hi + 1].sum() if hi >= lo else 0.0
    ps = O[s_row] if s_px == "open" else C[s_row]
    pe = O[e_row] if e_px == "open" else C[e_row]
    return (pe + div) / ps - 1.0


def event_table(days: dict, etf: ft.Panel, pseudo: dict | None = None) -> pd.DataFrame:
    cal = etf.dates
    rows = []
    for bucket, sym in BUCKET_ETF.items():
        O, C, D = etf_series(etf, sym)
        tr = np.r_[np.nan, (C[1:] + D[:-1]) / C[:-1] - 1.0]  # close-to-close total return, dividend of row t-1
        mean_daily = {}
        for h, (a, b) in HALVES.items():
            m = (cal >= a) & (cal <= b)
            mean_daily[h] = np.nanmean(tr[m])
        ev_days = (pseudo or days)[bucket]
        rows_idx = cal.get_indexer(ev_days)
        for d, A in zip(ev_days, rows_idx):
            if A < 6 or A + 6 >= len(cal) or not np.isfinite(C[A - 6]):
                continue
            half = "H1_2003_14" if d <= pd.Timestamp("2014-12-31") else "H2_2015_26"
            md = mean_daily[half]
            for kind, k in WINDOWS:
                if kind == "POST_OPEN":
                    r = window_return(O, C, D, A, "open", A + k, "open")
                elif kind == "POST_CLOSE":
                    r = window_return(O, C, D, A, "close", A + k, "close")
                else:
                    r = window_return(O, C, D, A - 5, "close", A - 1, "close")
                    k = 4
                rows.append((bucket, d, half, f"{kind}_{k}" if kind != "PRE" else "PRE", r, r - md * k))
    return pd.DataFrame(rows, columns=["bucket", "date", "half", "window", "ret", "excess"])


def screen_table(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev[(ev.date >= POOL[0]) & (ev.date <= POOL[1])]
    rows = []
    for (b, w), g in ev.groupby(["bucket", "window"]):
        x = g.excess.dropna().to_numpy()
        t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
        tn = (x.mean() - HURDLE) / (x.std(ddof=1) / np.sqrt(len(x)))
        h1 = g[g.half == "H1_2003_14"].excess.mean()
        h2 = g[g.half == "H2_2015_26"].excess.mean()
        rows.append({"bucket": b, "window": w, "n": len(x), "mean_bps": x.mean() * 1e4, "t": t, "t_net": tn,
                     "H1_bps": h1 * 1e4, "H2_bps": h2 * 1e4, "hit": (x > 0).mean(),
                     "LIVE": bool(tn >= 2 and h1 > 0 and h2 > 0 and not w.startswith("PRE"))})
    return pd.DataFrame(rows)


def main(step: str):
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    if step == "download":
        download()
        return
    etf = ft.Panel("etfx")
    days = auction_days()
    if step == "screen":
        ev = event_table(days, etf)
        ev.to_csv(OUT_PATH / "events.csv", index=False)
        sc = screen_table(ev)
        sc.to_csv(OUT_PATH / "screen.csv", index=False)
        print({k: len(v[(v >= POOL[0]) & (v <= POOL[1])]) for k, v in days.items()})
        print(sc.round(3).to_string())
    elif step == "placebo":
        rng = np.random.default_rng(20260926)
        cal = etf.dates
        sc_real = pd.read_csv(OUT_PATH / "screen.csv")
        res = []
        for it in range(200):
            pseudo = {}
            for b, dd in days.items():
                dd = dd[(dd >= "2002-09-01") & (dd <= POOL[1])]
                picks = []
                for d in dd:
                    month_days = cal[(cal.year == d.year) & (cal.month == d.month)]
                    cand = month_days.difference(days["LONG"].union(days["TEN"]))
                    if len(cand):
                        picks.append(cand[rng.integers(len(cand))])
                pseudo[b] = pd.DatetimeIndex(sorted(set(picks)))
            sc = screen_table(event_table(days, etf, pseudo))
            sc["iter"] = it
            res.append(sc)
        pl = pd.concat(res)
        pl.to_csv(OUT_PATH / "placebo.csv", index=False)
        q = pl.groupby(["bucket", "window"]).mean_bps.quantile([0.5, 0.95]).unstack()
        print(q.join(sc_real.set_index(["bucket", "window"])["mean_bps"].rename("actual")).round(2).to_string())
    elif step == "holdout":
        import io
        import requests
        rows = []
        for bucket, sid in (("LONG", "DGS30"), ("TEN", "DGS10")):
            txt = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", timeout=60).text
            y = pd.to_numeric(pd.read_csv(io.StringIO(txt), index_col=0, parse_dates=True).iloc[:, 0], errors="coerce").dropna()
            ycal = y.index
            ev_rows = []
            for d in days[bucket]:
                if not (pd.Timestamp("1983-01-01") <= d <= pd.Timestamp("2002-06-30")):
                    continue
                i = ycal.searchsorted(d)
                if i < 6 or i >= len(ycal) or ycal[i] != d:
                    continue
                post3 = (y.iloc[i + 3] - y.iloc[i]) * 100 if i + 3 < len(ycal) else np.nan
                ev_rows.append((d, (y.iloc[i - 1] - y.iloc[i - 5]) * 100, post3))
            df = pd.DataFrame(ev_rows, columns=["date", "pre_bps", "post3_bps"])
            df.to_csv(OUT_PATH / f"holdout_{bucket}.csv", index=False)
            drift4 = (y.loc["1983-01-01":"2002-06-30"].diff(4) * 100).mean()
            for label, x in (("A7_raw", df.pre_bps), ("drift_adjusted_sensitivity", df.pre_bps - drift4)):
                rows.append({"bucket": bucket, "series": sid, "version": label, "n": len(x), "mean_bps": x.mean(),
                             "t": x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))})
        out = pd.DataFrame(rows)
        out.to_csv(OUT_PATH / "holdout_summary.csv", index=False)
        print(out.round(3).to_string())
    elif step == "dom":
        cal = etf.dates
        all_auct = days["LONG"].union(days["TEN"])
        rows = []
        for bucket, sym in BUCKET_ETF.items():
            O, C, D = etf_series(etf, sym)
            vals = {}
            for A in range(10, len(cal) - 1):
                if np.isfinite(C[A - 5]) and np.isfinite(C[A - 1]):
                    vals[cal[A]] = window_return(O, C, D, A - 5, "close", A - 1, "close")
            pre = pd.Series(vals).loc[POOL[0]:POOL[1]]
            # *** CRITICAL*** trading day of the month of the auction day A, from the market calendar
            day_of_month = pd.Series(1, index=cal).groupby([cal.year, cal.month]).cumsum().reindex(pre.index)
            is_auction = pre.index.isin(days[bucket])
            non = ~pre.index.isin(all_auct)
            dom_mean = pre[non].groupby(day_of_month[non]).mean()
            adj = pre[is_auction] - day_of_month[is_auction].map(dom_mean)
            for per, (a, b) in {"POOL": POOL, **HALVES}.items():
                x = adj.loc[a:b]
                rows.append({"bucket": bucket, "period": per, "n": len(x), "raw_bps": pre[is_auction].loc[a:b].mean() * 1e4,
                             "dom_adjusted_bps": x.mean() * 1e4, "t": x.mean() / (x.std(ddof=1) / np.sqrt(len(x))),
                             "median_auction_day_of_month": float(day_of_month[is_auction].median())})
        out = pd.DataFrame(rows)
        out.to_csv(OUT_PATH / "dom_control.csv", index=False)
        print(out.round(2).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
