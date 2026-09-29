"""Stage 1, families C and D: ETF signal maps (descriptive; SPEC_FROZEN.md with amendments A1-A3).

C (cross-asset liquidity provision), per ETF i and decision date t (after Close_t), next-open forward F_h:

    excess_{i,t} = F_h[i, t] - mean(F_h[i, s] : s in the same block, i eligible)      on signal days
    daily_t      = mean over the class's ETFs with a signal on t

D (close-substitute relative value), per pair (narrow n, broad B), with the family-A residual machinery:

    e_t  = r_{n,t} - b_{t-1} r_{B,t};   E5 = sum_{j<5} e_{t-j} / (sigma_63 sqrt(5));   signal E5 < -2
    rel_{t} = F_h[n, t] - b_t F_h[B, t]        (hedged relative forward return, h = 5, 10)

Usage: python stage1_cd.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE_PATH))
import features as ft  # noqa: E402
import replica as rp  # noqa: E402
from cache_build import CLASS_ETF_DICT, PAIR_GROUP_DICT  # noqa: E402
from stage1_a import nw_t  # noqa: E402

OUT_PATH = ft.STUDY_OUT_PATH / "stage1_cd"
C_HALVES = {"H1_2003_14": ("2003-01-01", "2014-12-31"), "H2_2015_26": ("2015-01-01", "2026-08-19"),
            "POOL_2010_26": ("2010-01-01", "2026-08-19"), "ALL_2003_26": ("2003-01-01", "2026-08-19")}
D_HALVES = {"H1_2005_14": ("2005-01-01", "2014-12-31"), "H2_2015_26": ("2015-01-01", "2026-08-19"),
            "POOL_2005_26": ("2005-01-01", "2026-08-19")}
BLOCK_DICT = {"E_2000_09": ("2000-01-01", "2009-12-31"), "C1_2010_14": ("2010-01-01", "2014-12-31"),
              "C2_2015_19": ("2015-01-01", "2019-12-31"), "S_2020_22": ("2020-01-01", "2022-12-31"),
              "C3_2023_26": ("2023-01-01", "2026-08-19")}


def etf_arrays(p: ft.Panel):
    r = ft.returns(p)
    hist = ft.history_ok(p, 252)
    adv = ft.adv63(p)
    dv2 = rp.dvpct(p, 2, 126)
    ib = ft.ibs(p)
    z3 = ft.raw_z(p, (3,))[3]
    return r, hist, adv, {"DV2_lt10": dv2 < 10, "IBS_lt015": ib < 0.15, "Z3_lt_m15": z3 < -1.5}


def block_of(dates: pd.DatetimeIndex) -> np.ndarray:
    lab = np.full(len(dates), "", dtype=object)
    for k, (a, b) in BLOCK_DICT.items():
        lab[(dates >= a) & (dates <= b)] = k
    return lab


def family_c(p: ft.Panel, vix_bin: np.ndarray) -> pd.DataFrame:
    r, hist, adv, sig_dict = etf_arrays(p)
    blocks = block_of(p.dates)
    rows = []
    for cls, sym_list in CLASS_ETF_DICT.items():
        cols = [p.col(s) for s in sym_list if s in p.symbols]
        for hz in (1, 5):
            F = ft.forward_open_returns(p, hz)
            elig = hist & (adv > 10e6) & np.isfinite(F)
            # unconditional mean per ETF per block (drift removed so only the conditional effect is measured)
            unc = np.full(F.shape, np.nan)
            for blk in BLOCK_DICT:
                m = blocks == blk
                for c in cols:
                    v = F[m, c][elig[m, c]]
                    if v.size:
                        unc[m, c] = v.mean()
            for sname, sig in sig_dict.items():
                ev = elig & sig
                ex = np.where(ev, F - unc, np.nan)[:, cols]
                raw = np.where(ev, F, np.nan)[:, cols]
                with np.errstate(invalid="ignore"):
                    daily = np.nanmean(ex, axis=1)
                    daily_raw = np.nanmean(raw, axis=1)
                n_ev = np.isfinite(ex).sum()
                for per, (a, b) in {**C_HALVES, **BLOCK_DICT}.items():
                    m = (p.dates >= a) & (p.dates <= b)
                    me, te, n = nw_t(daily[m], hz - 1)
                    _, tn, _ = nw_t(daily[m] - 0.0006, hz - 1)
                    mr, _, _ = nw_t(daily_raw[m], hz - 1)
                    rows.append(("C", cls, sname, hz, per, me, te, tn, mr, n, int(n_ev)))
                for vb in ("VIX_low", "VIX_mid", "VIX_high"):
                    m = (vix_bin == vb) & (p.dates >= "2003-01-01") & (p.dates <= "2026-08-19")
                    me, te, n = nw_t(daily[m], hz - 1)
                    rows.append(("C", cls, sname, hz, vb, me, te, np.nan, np.nan, n, int(n_ev)))
    return pd.DataFrame(rows, columns=["family", "group", "signal", "h", "period", "mean_excess", "t", "t_net", "mean_raw",
                                       "n_days", "n_events"])


def family_d(p: ft.Panel, vix_bin: np.ndarray) -> pd.DataFrame:
    r = ft.returns(p)
    hist = ft.history_ok(p, 252)
    rows = []
    per_group_daily = {}
    for grp, (narrow_list, broad) in PAIR_GROUP_DICT.items():
        b_col = p.col(broad)
        rb = r[:, b_col]
        for hz in (5, 10):
            F = ft.forward_open_returns(p, hz)
            series = []
            for n_sym in narrow_list:
                if n_sym not in p.symbols:
                    continue
                n_col = p.col(n_sym)
                beta, _ = ft.roll_beta_corr(r[:, [n_col]], rb)
                b = np.clip(0.67 * beta[:, 0] + 0.33, 0.3, 2.0)
                # *** CRITICAL*** beta applied to day t's broad return is measured through t-1.
                e = r[:, n_col] - ft._lag(b) * rb
                sig = ft.roll_std(e[:, None], 63, 50)[:, 0]
                with np.errstate(invalid="ignore", divide="ignore"):
                    z5 = ft.roll_sum_strict(e[:, None], 5)[:, 0] / (sig * np.sqrt(5))
                rel = F[:, n_col] - b * F[:, b_col]
                ok = hist[:, n_col] & hist[:, b_col] & np.isfinite(rel) & (z5 < -2.0)
                series.append(np.where(ok, rel, np.nan))
            arr = np.vstack(series).T
            with np.errstate(invalid="ignore"):
                daily = np.nanmean(arr, axis=1)
            per_group_daily[(grp, hz)] = arr
            for per, (a, b0) in {**D_HALVES, **BLOCK_DICT}.items():
                m = (p.dates >= a) & (p.dates <= b0)
                me, te, n = nw_t(daily[m], hz - 1)
                _, tn, _ = nw_t(daily[m] - 0.0012, hz - 1)
                rows.append(("D", grp, "E5_lt_m2", hz, per, me, te, tn, np.nan, n, int(np.isfinite(arr[m]).sum())))
    for hz in (5, 10):
        arr = np.hstack([v for (g, h), v in per_group_daily.items() if h == hz])
        with np.errstate(invalid="ignore"):
            daily = np.nanmean(arr, axis=1)
        for per, (a, b0) in {**D_HALVES, **BLOCK_DICT}.items():
            m = (p.dates >= a) & (p.dates <= b0)
            me, te, n = nw_t(daily[m], hz - 1)
            _, tn, _ = nw_t(daily[m] - 0.0012, hz - 1)
            rows.append(("D", "ALL_PAIRS", "E5_lt_m2", hz, per, me, te, tn, np.nan, n, int(np.isfinite(arr[m]).sum())))
        for vb in ("VIX_low", "VIX_mid", "VIX_high"):
            m = (vix_bin == vb) & (p.dates >= "2005-01-01") & (p.dates <= "2026-08-19")
            me, te, n = nw_t(daily[m], hz - 1)
            rows.append(("D", "ALL_PAIRS", "E5_lt_m2", hz, vb, me, te, np.nan, np.nan, n, int(np.isfinite(arr[m]).sum())))
    return pd.DataFrame(rows, columns=["family", "group", "signal", "h", "period", "mean_excess", "t", "t_net", "mean_raw",
                                       "n_days", "n_events"])


def main():
    p = ft.Panel("etfx")
    vix = np.asarray(p.extra["vix_close"])
    q = np.nanquantile(vix[p.dates >= "1990-01-01"], [1 / 3, 2 / 3])
    vix_bin = np.where(vix <= q[0], "VIX_low", np.where(vix <= q[1], "VIX_mid", "VIX_high"))
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    c = family_c(p, vix_bin)
    c.to_csv(OUT_PATH / "map_c.csv", index=False)
    d = family_d(p, vix_bin)
    d.to_csv(OUT_PATH / "map_d.csv", index=False)
    print("saved", len(c), len(d))


if __name__ == "__main__":
    main()
