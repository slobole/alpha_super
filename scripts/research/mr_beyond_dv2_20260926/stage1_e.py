"""Stage 1, family E: calendar-flow reversal in stocks (amendment A4 of SPEC_FROZEN.md).

E-M: losers at each month-end (signals E21_SEC, E21_SPY, R21z), entry at the first open of the next month.
E-Y: year-to-date losers at the last December close, entry at the first January open.

    excess_m = mean(F^H_h over the bottom decile) - mean(F^H_h over all eligible)   on period-end decision rows
    control  = the same statistic on every other decision row

Usage: python stage1_e.py sp500 r1000x sp400 sp600 ndx
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE_PATH))
import features as ft  # noqa: E402
from stage1_a import HURDLE_BY_HEDGE, bucket_masks, r1000x_panel, row_mean  # noqa: E402

OUT_PATH = ft.STUDY_OUT_PATH / "stage1_e"
BLOCKS = {"H_1991_99": ("1991-01-02", "1999-12-31"), "E_2000_09": ("2000-01-01", "2009-12-31"),
          "C1_2010_14": ("2010-01-01", "2014-12-31"), "C2_2015_19": ("2015-01-01", "2019-12-31"),
          "S_2020_22": ("2020-01-01", "2022-12-31"), "C3_2023_26": ("2023-01-01", "2026-08-19"),
          "MAIN_2000_26": ("2000-01-01", "2026-08-19"), "HALF1_2000_12": ("2000-01-01", "2012-12-31"),
          "HALF2_2013_26": ("2013-01-01", "2026-08-19")}


def t_stat(x: np.ndarray) -> tuple[float, float, int]:
    x = x[np.isfinite(x)]
    if x.size < 5:
        return np.nan, np.nan, x.size
    return x.mean(), x.mean() / (x.std(ddof=1) / np.sqrt(x.size)), x.size


def ytd_return(p: ft.Panel) -> np.ndarray:
    C = np.asarray(p.C, dtype=np.float64)
    me = ft.month_end_rows(p.dates)
    dec_rows = me[p.dates[me].month == 12]
    out = np.full(C.shape, np.nan, dtype=np.float32)
    for k in range(1, len(dec_rows)):
        base, row = dec_rows[k - 1], dec_rows[k]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[row] = C[row] / C[base] - 1.0
    return out


def run_universe(label_str: str) -> pd.DataFrame:
    p = r1000x_panel() if label_str == "r1000x" else ft.Panel(label_str)
    h = ft.Hedges(p)
    me_rows = ft.month_end_rows(p.dates)
    me_rows = me_rows[me_rows < len(p.dates) - 12]
    is_me = np.zeros(len(p.dates), dtype=bool)
    is_me[me_rows] = True
    month = p.dates.month.to_numpy()
    es, ec, rz = ft.residual_z(p, h, "spy"), ft.residual_z(p, h, "sec"), ft.raw_z(p)
    sig_dict = {"E21_SEC": ec[21], "E21_SPY": es[21], "R21z": rz[21], "YTD": ytd_return(p)}
    rows = []
    for tier in ("L", "ALL"):
        elig = ft.eligible(p, tier)
        for sname, sig in sig_dict.items():
            ok = elig & np.isfinite(sig)
            dec, _, row_ok = bucket_masks(sig, ok)
            h_list = (5, 10) if sname == "YTD" else (3, 5)
            for hz in h_list:
                for hg in ("none", "spy", "sec"):
                    F = ft.hedged_forward(p, h, hz, hg)
                    exc = row_mean(F, dec) - row_mean(F, ok & row_ok[:, None])
                    subsets = {"month_end": is_me, "quarter_end": is_me & np.isin(month, [3, 6, 9, 12]),
                               "december": is_me & (month == 12), "other_days": ~is_me}
                    if sname == "YTD":
                        subsets = {"december": is_me & (month == 12)}
                    for sub, sm in subsets.items():
                        for blk, (a, b) in BLOCKS.items():
                            m = sm & (p.dates >= a) & (p.dates <= b)
                            x = exc[m]
                            me_, t_, n = t_stat(x)
                            _, tn, _ = t_stat(x - HURDLE_BY_HEDGE[hg])
                            rows.append((label_str, tier, sname, hz, hg, sub, blk, me_, t_, tn, n))
    df = pd.DataFrame(rows, columns=["universe", "tier", "signal", "h", "hedge", "subset", "block", "mean_excess", "t", "t_net", "n"])
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH / f"map_e_{label_str}.csv", index=False)
    print(label_str, "saved", len(df), flush=True)
    return df


if __name__ == "__main__":
    for label in sys.argv[1:]:
        run_universe(label)
