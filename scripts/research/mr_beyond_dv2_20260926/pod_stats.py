"""Shared pod metrics for Stage 2/3 (research-only). Sharpe uses a zero risk-free rate (QUANT_PHILOSOPHY 9).

    r_t      = NAV_t / NAV_{t-1} - 1
    CAGR     = (NAV_end / NAV_start) ^ (365.25 / days) - 1
    Sharpe   = mean(r) / std(r) * sqrt(252)
    drawdown = NAV_t / max(NAV_1..NAV_t) - 1
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BLOCKS = {"E_2000_09": ("2000-01-01", "2009-12-31"), "C1_2010_14": ("2010-01-01", "2014-12-31"),
          "C2_2015_19": ("2015-01-01", "2019-12-31"), "S_2020_22": ("2020-01-01", "2022-12-31"),
          "C3_2023_26": ("2023-01-01", "2026-08-19")}
CALM = ["C1_2010_14", "C2_2015_19", "C3_2023_26"]
INVENTORY_PATH = Path(__file__).resolve().parents[3] / "results" / "research" / "portfolio" / "fund_product_menu_20260923" / "inventory"


def basic(nav: pd.Series) -> dict:
    s = nav.dropna()
    if len(s) < 30 or s.iloc[0] <= 0:
        return {"cagr": np.nan, "sharpe": np.nan, "maxdd": np.nan, "vol": np.nan}
    r = s.pct_change().dropna()
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1 if s.iloc[-1] > 0 else -1.0
    sd = r.std()
    return {"cagr": cagr, "sharpe": r.mean() / sd * np.sqrt(252) if sd > 0 else np.nan,
            "maxdd": (s / s.cummax() - 1).min(), "vol": sd * np.sqrt(252)}


def full(nav: pd.Series, gross: pd.Series | None = None, trades: pd.DataFrame | None = None) -> dict:
    out = basic(nav)
    out["calmar"] = out["cagr"] / abs(out["maxdd"]) if out["maxdd"] and out["maxdd"] < 0 else np.nan
    r = nav.pct_change()
    calm_r = []
    for blk, (a, b) in BLOCKS.items():
        seg = nav.loc[a:b]
        if len(seg) > 30:
            # include the prior close so the block's first return is counted
            prev = nav.loc[:a].iloc[:-1]
            if len(prev):
                seg = pd.concat([prev.iloc[[-1]], seg])
            st = basic(seg)
            out[f"{blk}_cagr"], out[f"{blk}_sharpe"] = st["cagr"], st["sharpe"]
        if blk in CALM:
            calm_r.append(r.loc[a:b])
    cr = pd.concat(calm_r).dropna() if calm_r else pd.Series(dtype=float)
    out["calm_sharpe"] = cr.mean() / cr.std() * np.sqrt(252) if len(cr) > 30 and cr.std() > 0 else np.nan
    out["calm_all_pos"] = all(out.get(f"{b}_cagr", -1) > 0 for b in CALM)
    if gross is not None:
        out["avg_gross"] = float(gross.mean())
        out["days_invested"] = float((gross > 0).mean())
    if trades is not None and len(trades):
        yrs = (nav.index[-1] - nav.index[0]).days / 365.25
        out["entries_per_year"] = float((trades["kind"] == "entry").sum() / yrs)
    return out


def sleeve_returns() -> pd.DataFrame:
    return pd.read_csv(INVENTORY_PATH / "sleeve_returns.csv.gz", index_col=0, parse_dates=True)


def benchmark_returns() -> pd.DataFrame:
    return pd.read_csv(INVENTORY_PATH / "benchmark_returns.csv.gz", index_col=0, parse_dates=True)


def dv2_adv_returns() -> pd.Series:
    """F1 (liquidity floor + ADV63 rank) from the DV2 deep study (replica equals engine trade for trade)."""
    p = Path(__file__).resolve().parents[3] / "results" / "research" / "dv2_deep_20260925" / "sources" / "F1_floor_adv__path.csv.gz"
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    col = [c for c in df.columns if "total_value" in c][0]
    return df[col].pct_change()


def difference_stats(r: pd.Series) -> dict:
    """G1: correlation with DV2-ADV and HPI vote, beta to $SPXTR over the common window."""
    sl = sleeve_returns()
    bm = benchmark_returns()
    dv2 = dv2_adv_returns()
    out = {}
    for name, ref in (("dv2_adv", dv2), ("hpi_vote", sl["hpi_vote"]), ("dv2_wired", sl["dv2"])):
        j = pd.concat([r, ref], axis=1).dropna()
        out[f"corr_{name}"] = j.iloc[:, 0].corr(j.iloc[:, 1]) if len(j) > 60 else np.nan
    spx_col = [c for c in bm.columns if "SPX" in c.upper()][0]
    j = pd.concat([r, bm[spx_col]], axis=1).dropna()
    out["beta_spx"] = np.cov(j.iloc[:, 0], j.iloc[:, 1])[0, 1] / j.iloc[:, 1].var()
    out["corr_spx"] = j.iloc[:, 0].corr(j.iloc[:, 1])
    return out
