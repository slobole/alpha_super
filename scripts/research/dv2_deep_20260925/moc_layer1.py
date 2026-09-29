"""MOC layer 1: how far is each DV2 decision from flipping if it were taken at 15:45 instead of the close?

Let b = P_1545 / Close - 1 (the 15:45 price relative to the close; high/low changes after 15:45 ignored here).
DV2 < 10 on day t  <=>  DV_t < s12, the 12th smallest of the 125 prior DV values (window 126, no ties), where
    DV_t = (DV1_{t-1} + DV1_t) / 2,  DV1_t = Close_t / mid_t - 1
so the entry still fires at 15:45 iff  b < b_dv = mid_t * (1 + 2*s12 - DV1_{t-1}) / Close_t - 1.
Trend filters with today's price replaced: SMA200 needs b > S199/(199*Close) - 1; R126 > 5% needs
b > 1.05 * Close_{t-126} / Close - 1.  Exit X0 (Close > High_{t-1}) still fires iff b > High_{t-1} / Close - 1.
The probability of each event uses the empirical 15:45 basis of the pakal Alpaca panel (2021-2026, top-250
S&P 500 by ADV), scaled by each stock-day's 20-day volatility: b = z * sigma_20d_daily.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replica as rp  # noqa: E402
import batch  # noqa: E402

PAKAL = Path(r"C:\Users\User\Documents\workspace\pakal\pakal-research\reports\preclose_official_close_moc_basis_study\tables\analysis_panel.csv.gz")


def basis_z() -> np.ndarray:
    df = pd.read_csv(PAKAL, usecols=["universe_label_str", "cutoff_price_float", "official_close_price_float", "prior_realized_volatility_20d_float"])
    df = df[df.universe_label_str == "stocks"].dropna()
    b = df.cutoff_price_float / df.official_close_price_float - 1
    sd = df.prior_realized_volatility_20d_float / np.sqrt(252)
    z = (b / sd).to_numpy()
    return z[np.isfinite(z)]


def s12_threshold(dv: np.ndarray, w=126, k=12) -> np.ndarray:
    """k-th smallest of the w-1 values before t (NaN if any missing)."""
    T, N = dv.shape
    out = np.full((T, N), np.nan)
    from numpy.lib.stride_tricks import sliding_window_view
    for c0 in range(0, N, 40):
        win = sliding_window_view(dv[:, c0:c0 + 40], w - 1, axis=0)  # window j = rows j..j+w-2
        part = np.partition(win, k - 1, axis=2)[:, :, k - 1]
        part[np.isnan(win).any(axis=2)] = np.nan
        # *** CRITICAL*** decision day t uses the w-1 values strictly before t: window j = t-(w-1)
        out[w - 1:, c0:c0 + 40] = part[:-1]
    return out


def main():
    p = rp.Panel("sp500")
    z = np.sort(basis_z())
    cdf = lambda x: np.searchsorted(z, x) / len(z)
    dv1 = rp.dv1(p)
    dv = rp.dvk(p, 2)
    s12 = s12_threshold(dv)
    mid = (p.H + p.L) / 2
    prev_dv1 = np.vstack([np.full((1, dv1.shape[1]), np.nan), dv1[:-1]])
    with np.errstate(invalid="ignore", divide="ignore"):
        b_dv = mid * (1 + 2 * s12 - prev_dv1) / p.C - 1
        S199 = pd.DataFrame(p.C).shift(1).rolling(199).sum().to_numpy()
        b_sma = S199 / (199 * p.C) - 1
        b_mom = 1.05 * np.vstack([np.full((126, p.C.shape[1]), np.nan), p.C[:-126]]) / p.C - 1
        prevH = np.vstack([np.full((1, p.H.shape[1]), np.nan), p.H[:-1]])
        b_exit = prevH / p.C - 1
        sig = pd.DataFrame(p.C).pct_change().rolling(20).std().to_numpy()
    report = {"basis_z_quantiles": {q: float(np.quantile(z, q)) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)},
              "basis_z_n": int(len(z))}
    for name, rule in [("floor", rp.Rule(floor=True)), ("wired", rp.Rule())]:
        res = rp.run(p, rule)
        tr = res.trades
        rows = p.dates.get_indexer(pd.DatetimeIndex(tr["date"])) - 1
        cols = tr["asset"].map({s: i for i, s in enumerate(p.symbols)}).to_numpy()
        ent = (tr["kind"] == "entry").to_numpy()
        exi = (tr["kind"] == "exit").to_numpy()
        # entries: probability the 15:45 decision still fires
        r_e, c_e = rows[ent], cols[ent]
        sd = sig[r_e, c_e]
        up = b_dv[r_e, c_e] / sd
        lo = np.fmax(b_sma[r_e, c_e], b_mom[r_e, c_e]) / sd
        p_keep = np.clip(cdf(up) - cdf(lo), 0, 1)
        # exits: probability the exit still fires at 15:45
        r_x, c_x = rows[exi], cols[exi]
        ex_d = b_exit[r_x, c_x] / sig[r_x, c_x]
        p_exit_keep = 1 - cdf(ex_d)
        # false entries: stock-days that did not signal at the close but would at 15:45 (members, complete rows)
        ok = p.member & p.valid & np.isfinite(b_dv) & np.isfinite(b_sma) & np.isfinite(b_mom) & np.isfinite(sig)
        t0, t1 = p.dates.searchsorted(pd.Timestamp("2000-01-03")), p.dates.searchsorted(pd.Timestamp("2026-08-19"), side="right")
        okw = ok[t0 - 1:t1 - 1]
        up_all = (b_dv / sig)[t0 - 1:t1 - 1][okw]
        lo_all = (np.fmax(b_sma, b_mom) / sig)[t0 - 1:t1 - 1][okw]
        fires_close = (up_all > 0) & (lo_all < 0)
        p_fire = np.clip(cdf(up_all) - cdf(lo_all), 0, 1)
        exp_new = float(p_fire[~fires_close].sum())
        exp_lost = float((1 - p_fire[fires_close]).sum())
        report[name] = {
            "entries": int(ent.sum()), "exits": int(exi.sum()),
            "entry_dv_distance_bps_median": float(np.nanmedian(b_dv[r_e, c_e]) * 1e4),
            "entry_dv_distance_sigma_quantiles": {q: float(np.nanquantile(up, q)) for q in (0.05, 0.1, 0.25, 0.5)},
            "share_entries_within_1_basis_mad": float(np.nanmean(up < np.median(np.abs(z)))),
            "expected_share_entries_kept": float(np.nanmean(p_keep)),
            "exit_distance_sigma_median": float(np.nanmedian(ex_d)),
            "expected_share_exits_kept": float(np.nanmean(p_exit_keep)),
            "universe_signals_at_close": int(fires_close.sum()),
            "expected_signals_lost_at_1545": exp_lost, "expected_new_signals_at_1545": exp_new,
        }
    out = batch.OUT / "moc_layer1.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
