"""Stage 1, family A: signal map of stock reversal signals (descriptive; SPEC_FROZEN.md).

For each universe, liquidity tier, signal, horizon h and hedge, per decision date t:

    pod_t    = mean over the bucket of the hedged forward return F^H_h (what a pod would earn per trade)
    excess_t = pod_t - mean over all eligible names of F^H_h                 (signal quality)

bucket = bottom decile of the signal among eligible names (>= 20 eligible), or the 10 most extreme names.
Statistics per block: mean, Newey-West t with h-1 lags, and t of (excess - hurdle).

Usage: python stage1_a.py sp500 r1000x sp400 sp600 ndx
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

OUT_PATH = ft.STUDY_OUT_PATH / "stage1_a"
BLOCK_DICT = {
    "H_1991_99": ("1991-01-02", "1999-12-31"),
    "E_2000_09": ("2000-01-01", "2009-12-31"),
    "C1_2010_14": ("2010-01-01", "2014-12-31"),
    "C2_2015_19": ("2015-01-01", "2019-12-31"),
    "S_2020_22": ("2020-01-01", "2022-12-31"),
    "C3_2023_26": ("2023-01-01", "2026-08-19"),
    "MAIN_2000_26": ("2000-01-01", "2026-08-19"),
}
CALM_BLOCK_LIST = ["C1_2010_14", "C2_2015_19", "C3_2023_26"]
HURDLE_BY_HEDGE = {"none": 0.0008, "spy": 0.0010, "sec": 0.0010}
HEDGE_LIST = ["none", "spy", "sec"]
TIER_LIST = ["L", "ALL"]


def nw_t(x: np.ndarray, lags: int) -> tuple[float, float, int]:
    x = x[np.isfinite(x)]
    n = x.size
    if n < 30:
        return np.nan, np.nan, n
    m = x.mean()
    d = x - m
    var = d @ d / n
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        var += 2.0 * w * (d[l:] @ d[:-l]) / n
    se = np.sqrt(max(var, 1e-18) / n)
    return m, m / se, n


def signals(p: ft.Panel, h: ft.Hedges) -> dict:
    rz = ft.raw_z(p)
    es = ft.residual_z(p, h, "spy")
    ec = ft.residual_z(p, h, "sec")
    intra, ovn = ft.intra_overnight_z(p)
    return {
        "R1z": rz[1], "R5z": rz[5], "R21z": rz[21], "R5raw": ft.raw_ret(p, 5),
        "E1_SPY": es[1], "E5_SPY": es[5], "E21_SPY": es[21],
        "E1_SEC": ec[1], "E5_SEC": ec[5], "E21_SEC": ec[21],
        "INTRA5": intra, "OVN5": ovn, "DV2": ft.dv2(p).astype(np.float32), "IBS": ft.ibs(p),
    }


def bucket_masks(values: np.ndarray, ok: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bottom-decile and bottom-10 masks per row among ok names; rows with < 20 ok names are dropped."""
    v = np.where(ok, values, np.inf).astype(np.float64)
    order = np.argsort(v, axis=1, kind="stable")
    ranks = np.empty_like(order)
    T, N = v.shape
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(N), (T, N)), axis=1)
    n_ok = ok.sum(axis=1)
    row_ok = n_ok >= 20
    k_dec = np.maximum(n_ok // 10, 1)
    dec = ok & (ranks < k_dec[:, None]) & row_ok[:, None]
    b10 = ok & (ranks < 10) & row_ok[:, None]
    return dec, b10, row_ok


def row_mean(F: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask & np.isfinite(F)
    cnt = m.sum(axis=1)
    s = np.where(m, F, 0.0).sum(axis=1, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, s / cnt, np.nan)


def r1000x_panel() -> ft.Panel:
    p = ft.Panel("r1000")
    sp = ft.Panel("sp500")
    pos = sp.dates.get_indexer(p.dates)
    sp_member = np.zeros(p.C.shape, dtype=bool)
    sp_col = {s: j for j, s in enumerate(sp.symbols)}
    spm = np.asarray(sp.member)
    rows_ok = pos >= 0
    for i, s in enumerate(p.symbols):
        j = sp_col.get(s)
        if j is not None:
            sp_member[rows_ok, i] = spm[pos[rows_ok], j]
    # *** CRITICAL*** disjoint validation universe: Russell 1000 members that are NOT S&P 500 members that day.
    p.member = np.asarray(p.member) & ~sp_member
    p.label_str = "r1000x"
    return p


def run_universe(label_str: str) -> pd.DataFrame:
    started = time.time()
    p = r1000x_panel() if label_str == "r1000x" else ft.Panel(label_str)
    h = ft.Hedges(p)
    sig_dict = signals(p, h)
    print(label_str, "signals", round(time.time() - started, 1), flush=True)
    vix = h.vix
    main_mask = (p.dates >= "2000-01-01") & (p.dates <= "2026-08-19")
    vix_q = np.nanquantile(vix[(p.dates >= "1990-01-01")], [1 / 3, 2 / 3])
    vix_bin = np.where(vix <= vix_q[0], "VIX_low", np.where(vix <= vix_q[1], "VIX_mid", "VIX_high"))
    fwd = {(hz, hg): ft.hedged_forward(p, h, hz, hg) for hz in ft.HORIZON_LIST for hg in HEDGE_LIST}
    print(label_str, "forwards", round(time.time() - started, 1), flush=True)
    abn = ft.abnormal_turnover(p)
    assign, _ = ft.sector_assignment(p, h)
    sec_trend = np.full(p.C.shape, np.nan, dtype=np.float32)
    for s_i, sym in enumerate(ft.SECTOR_SPDR_LIST):
        c = h.close[sym]
        with np.errstate(invalid="ignore", divide="ignore"):
            tr = c / ft._lag(c, 63) - 1.0
        sel = assign == s_i
        sec_trend[sel] = np.broadcast_to(tr[:, None], p.C.shape)[sel]
    rows, daily = [], {}
    for tier in TIER_LIST:
        elig = ft.eligible(p, tier)
        for sig_name, sig in sig_dict.items():
            ok = elig & np.isfinite(sig)
            dec, b10, row_ok = bucket_masks(sig, ok)
            splits = {}
            if sig_name.endswith("_SEC"):
                # A2: abnormal-turnover terciles (cross-sectional, among eligible names that day)
                a = np.where(ok, abn, np.nan)
                q1, q2 = np.nanquantile(a, 1 / 3, axis=1), np.nanquantile(a, 2 / 3, axis=1)
                splits["turn_low"] = dec & (a <= q1[:, None])
                splits["turn_high"] = dec & (a > q2[:, None])
                # A3: assigned sector ETF 63-day trend
                splits["sec_up"] = dec & (sec_trend > 0)
                splits["sec_down"] = dec & (sec_trend <= 0)
            for (hz, hg), F in fwd.items():
                uni = row_mean(F, ok & row_ok[:, None])
                for bname, bmask in [("dec", dec), ("b10", b10)] + [(f"dec_{k}", v) for k, v in splits.items()]:
                    if bname.startswith("dec_") and hz not in (5, 21):
                        continue
                    pod = row_mean(F, bmask)
                    exc = pod - uni
                    key = (label_str, tier, sig_name, hz, hg, bname)
                    daily[key] = exc.astype(np.float32)
                    hurdle = HURDLE_BY_HEDGE[hg]
                    blocks = dict(BLOCK_DICT)
                    for blk, (a0, b0) in blocks.items():
                        m = (p.dates >= a0) & (p.dates <= b0)
                        mp, _, _ = nw_t(pod[m], hz - 1)
                        me, te, n = nw_t(exc[m], hz - 1)
                        _, tn, _ = nw_t(exc[m] - hurdle, hz - 1)
                        rows.append(key + (blk, mp, me, te, tn, n))
                    calm = np.zeros(len(p.dates), dtype=bool)
                    for blk in CALM_BLOCK_LIST:
                        a0, b0 = BLOCK_DICT[blk]
                        calm |= (p.dates >= a0) & (p.dates <= b0)
                    mp, _, _ = nw_t(pod[calm], hz - 1)
                    me, te, n = nw_t(exc[calm], hz - 1)
                    _, tn, _ = nw_t(exc[calm] - hurdle, hz - 1)
                    rows.append(key + ("CALM", mp, me, te, tn, n))
                    for vb in ["VIX_low", "VIX_mid", "VIX_high"]:
                        m = main_mask & (vix_bin == vb)
                        mp, _, _ = nw_t(pod[m], hz - 1)
                        me, te, n = nw_t(exc[m], hz - 1)
                        _, tn, _ = nw_t(exc[m] - hurdle, hz - 1)
                        rows.append(key + (vb, mp, me, te, tn, n))
        print(label_str, tier, "done", round(time.time() - started, 1), flush=True)
    df = pd.DataFrame(rows, columns=["universe", "tier", "signal", "h", "hedge", "bucket", "block", "mean_pod", "mean_excess",
                                     "t_excess", "t_net", "n"])
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH / f"map_{label_str}.csv", index=False)
    daily_df = pd.DataFrame({"|".join(map(str, k)): v for k, v in daily.items()}, index=p.dates)
    daily_df.loc["1990-01-01":].to_parquet(OUT_PATH / f"daily_excess_{label_str}.parquet")
    print(label_str, "saved", round(time.time() - started, 1), flush=True)
    return df


if __name__ == "__main__":
    for label in sys.argv[1:]:
        run_universe(label)
