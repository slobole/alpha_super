"""TAA leakage hunt - TAA-03 scan on REAL data: padded / NaN / stale month-end endpoints and padded execution rows.

Run:  uv run python scripts/research/leakage_hunt_20260927/taa_staleness_scan.py

Norgate ALLMARKETDAYS padding (data/norgate_loader.py:73-79) inserts rows for market days on which a symbol did
not trade, carrying the prior close (Volume 0).  ``resample('ME').last()`` (strategy_taa_df.py:295,
..._vix_cash_variant_utils.py:166, linearity :134/:183) takes the last NON-NULL value per column, so:
  * a padded month-end row gives a stale (repeated) close silently;
  * a NaN month-end row makes resample pick an earlier date silently.
We scan every signal (TOTALRETURN), execution (CAPITALSPECIAL) and helper ($VIX) frame against the official XNYS
calendar (exchange_calendars) and report for each strategy's decision window:
  - month-end endpoints that are padded / NaN / not the last XNYS session;
  - rows in the loaded index that are NOT XNYS sessions (e.g. 2007-01-02 Ford mourning day, 2012-10-29/30 Sandy,
    2018-12-05, 2025-01-09) and whether any of them is a first-of-month rebalance "open" (execution at a padded bar).
Output: results/research/leakage_hunt_20260927/taa/taa03_staleness_scan.json (+ csv)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exchange_calendars as xc
import numpy as np
import pandas as pd

from taa_common import OUT_DIR, compute_decisions, load_norgate_full, patched

WINDOWS = {  # decision window per strategy (first decision label .. last completed month)
    "taa3x_rank": ("2012-09-01", "2026-08-31", ("GLD", "UUP", "TLT", "DBC", "BTAL"), ("GLD", "UUP", "TLT", "DBC", "BTAL", "TQQQ"), ("SPY", "$VIX")),
    "btal_qqq_linearity": ("2011-09-13", "2026-08-31", ("GLD", "UUP", "TLT", "DBC", "BTAL"), ("GLD", "UUP", "TLT", "DBC", "BTAL", "QQQ"), ("SPY", "$VIX")),
    "compass": ("2002-01-01", "2026-08-31", ("SPY", "XLE", "XLI", "XLF", "XLB", "XLU", "XLV", "XLP"), ("XLE", "XLK", "XLU", "XLP", "IEF"), ()),
}


def is_padded(df: pd.DataFrame) -> pd.Series:
    vol = df["Volume"] if "Volume" in df.columns else pd.Series(np.nan, index=df.index)
    flat = (df["Open"] == df["High"]) & (df["High"] == df["Low"]) & (df["Low"] == df["Close"]) & (df["Close"] == df["Close"].shift(1))
    if vol.notna().any() and (vol.fillna(0) > 0).mean() > 0.5:  # tradable instrument with volume
        return (vol.fillna(0) <= 0) & flat.fillna(False) | (vol.fillna(0) <= 0) & df["Close"].notna()
    return flat.fillna(False)  # index ($VIX): no volume field meaningfully populated


def main():
    cal = xc.get_calendar("XNYS", start="1990-01-02")
    xnys = pd.DatetimeIndex(cal.sessions).tz_localize(None)
    last_sess = pd.Series(xnys, index=xnys).groupby(xnys.to_period("M")).max()
    first_sess = pd.Series(xnys, index=xnys).groupby(xnys.to_period("M")).min()
    report, rows = {}, []
    for key, (lo, hi, sig_syms, exe_syms, helper_syms) in WINDOWS.items():
        lo_ts, hi_ts = pd.Timestamp(lo), pd.Timestamp(hi)
        months = pd.period_range(lo_ts, hi_ts, freq="M")
        rep = {}
        for role, syms, adj in (("signal_TR", sig_syms, "TOTALRETURN"), ("execution_CS", exe_syms, "CAPITALSPECIAL"),
                                ("helper_CS", helper_syms, "CAPITALSPECIAL")):
            for s in syms:
                df = load_norgate_full(s, adj).loc[lo_ts: hi_ts]
                pad = is_padded(df)
                nan = df["Close"].isna()
                non_session = df.index.difference(xnys)
                missing_sessions = xnys[(xnys >= max(lo_ts, df.index[0])) & (xnys <= hi_ts)].difference(df.index)
                bad_me = []
                for m in months:
                    if m not in last_sess.index:
                        continue
                    msub = df[df.index.to_period("M") == m]
                    if len(msub) == 0:
                        continue
                    last_valid = msub["Close"].last_valid_index()
                    ls = last_sess.loc[m]
                    flags = []
                    if last_valid is None:
                        flags.append("no_valid_close")
                    else:
                        if last_valid != ls:
                            flags.append(f"endpoint_{last_valid.date()}_ne_last_session_{ls.date()}")
                        if bool(pad.get(last_valid, False)):
                            flags.append("endpoint_row_padded")
                    if flags:
                        bad_me.append({"month": str(m), "flags": flags})
                rep[f"{role}:{s}"] = {
                    "n_rows": int(len(df)), "n_padded_rows": int(pad.sum()), "n_nan_close": int(nan.sum()),
                    "padded_dates_sample": [str(d.date()) for d in df.index[pad][:8]],
                    "n_rows_not_xnys_session": int(len(non_session)),
                    "rows_not_xnys_session": [str(d.date()) for d in non_session[:20]],
                    "n_xnys_sessions_missing": int(len(missing_sessions)),
                    "n_month_end_flags": len(bad_me), "month_end_flags": bad_me[:20],
                }
                rows.append({"strategy": key, "role": role, "symbol": s, **{k: v for k, v in rep[f"{role}:{s}"].items()
                                                                            if not isinstance(v, list)}})
        # execution calendar: do rebalance dates land on true first sessions?
        with patched():
            dec = compute_decisions(key if key != "taa3x_rank" else "taa3x_rank")
        rb = dec["rebalance_weight_df"].index
        rb = rb[(rb >= lo_ts) & (rb <= hi_ts + pd.Timedelta(days=40))]
        wrong = [str(d.date()) for d in rb if d not in xnys or d != first_sess.get(d.to_period("M"), pd.NaT)]
        rep["rebalance_dates_not_first_xnys_session"] = wrong
        # decision labels whose month's last XNYS session is absent from the signal frame index
        report[key] = rep
    pd.DataFrame(rows).to_csv(OUT_DIR / "taa03_staleness_scan.csv", index=False)
    (OUT_DIR / "taa03_staleness_scan.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    for key, rep in report.items():
        print(key, "rebalance dates not first XNYS session:", rep["rebalance_dates_not_first_xnys_session"])
        for k, v in rep.items():
            if isinstance(v, dict) and (v["n_month_end_flags"] or v["n_rows_not_xnys_session"] or v["n_padded_rows"]):
                print("  ", k, "pad", v["n_padded_rows"], v["padded_dates_sample"], "| non-session", v["rows_not_xnys_session"],
                      "| ME flags", v["month_end_flags"][:5])


if __name__ == "__main__":
    main()
