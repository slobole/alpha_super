"""Defensive-book leakage hunt: shared helpers (study 2026-09-27; research-only).

No strategy/engine source is modified.  Data are cached (pickle) in the session scratchpad so that every test in the
defensive audit reads exactly the same REAL Norgate frames.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import harness  # noqa: E402

CACHE = Path(r"C:\Users\User\AppData\Local\Temp\claude\C--Users-User-Documents-workspace-alpha-super"
             r"\2b26c675-1db0-4ceb-8ed6-fb7880170526\scratchpad\def_cache")
CACHE.mkdir(parents=True, exist_ok=True)
OUT = REPO / "results" / "research" / "leakage_hunt_20260927" / "def"
OUT.mkdir(parents=True, exist_ok=True)

BOOK_START = pd.Timestamp("2012-10-02")
BOOK_END = pd.Timestamp("2026-08-19")
FACTORS = harness.DEFAULT_FACTOR_TUPLE


def cached(name: str, fn):
    path = CACHE / f"{name}.pkl"
    if path.exists():
        with path.open("rb") as fh:
            return pickle.load(fh)
    obj = fn()
    with path.open("wb") as fh:
        pickle.dump(obj, fh)
    return obj


def metrics_from_returns(ret: pd.Series, start=None, end=None) -> dict:
    r = ret.astype(float).dropna()
    if start is not None:
        r = r[r.index >= pd.Timestamp(start)]
    if end is not None:
        r = r[r.index <= pd.Timestamp(end)]
    if len(r) < 2:
        return {"cagr": np.nan, "sharpe": np.nan, "vol": np.nan, "maxdd": np.nan, "n": len(r)}
    eq = (1.0 + r).cumprod()
    years = (r.index[-1] - r.index[0]).days / 365.25
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
    sd = float(r.std(ddof=1))
    sharpe = float(r.mean() / sd * np.sqrt(252.0)) if sd > 0 else np.nan
    dd = float((eq / eq.cummax() - 1.0).min())
    return {"cagr": cagr, "sharpe": sharpe, "vol": sd * np.sqrt(252.0), "maxdd": dd, "n": int(len(r)),
            "start": r.index[0].date().isoformat(), "end": r.index[-1].date().isoformat()}


def strategy_returns(strategy_obj) -> pd.Series:
    return strategy_obj.results["daily_returns"].astype(float)


def metric_rows(label: str, ret: pd.Series, windows: dict) -> list[dict]:
    rows = []
    for wname, (s, e) in windows.items():
        m = metrics_from_returns(ret, s, e)
        rows.append({"variant": label, "window": wname, **m})
    return rows


def rescale_frame(pricing_df: pd.DataFrame, namespace_list, factor: float) -> pd.DataFrame:
    out = pricing_df
    for ns in namespace_list:
        out = harness.rescale_symbol_history(out, ns, factor)
    return out


def price_scale_k(pricing_df: pd.DataFrame, symbol: str) -> pd.Series:
    """k_T = UnadjustedClose_T / AdjustedClose_T (1 raw share = k adjusted ledger units)."""
    return (pricing_df[(symbol, "Unadjusted Close")].astype(float)
            / pricing_df[(symbol, "Close")].astype(float))


def per_share_fee_effect(tx: pd.DataFrame, pricing_df: pd.DataFrame, per_share: float, minimum: float) -> dict:
    """Commission charged on adjusted units vs raw-equivalent units, from an existing fill ledger.

    raw-equivalent shares at fill date = adjusted shares / k_fill, k = Unadj/Adj Close of the fill bar.
    """
    rows = []
    for rec in tx.itertuples(index=False):
        bar = pd.Timestamp(rec.bar)
        k = float(pricing_df.loc[bar, (rec.asset, "Unadjusted Close")] / pricing_df.loc[bar, (rec.asset, "Close")])
        adj_sh = abs(float(rec.amount))
        raw_sh = adj_sh / k
        fee_adj = max(minimum, per_share * adj_sh) if per_share else 0.0
        fee_raw = max(minimum, per_share * raw_sh) if per_share else 0.0
        rows.append({"bar": bar, "asset": rec.asset, "adj_shares": adj_sh, "raw_shares": raw_sh, "k": k,
                     "fee_model": float(rec.commission), "fee_adj_recalc": fee_adj, "fee_raw": fee_raw})
    df = pd.DataFrame(rows)
    if df.empty:
        return {"n_fills": 0}
    return {
        "n_fills": int(len(df)),
        "fee_model_total": float(df["fee_model"].sum()),
        "fee_adj_recalc_total": float(df["fee_adj_recalc"].sum()),
        "fee_raw_total": float(df["fee_raw"].sum()),
        "fee_excess_model_minus_raw": float(df["fee_adj_recalc"].sum() - df["fee_raw"].sum()),
        "k_min": float(df["k"].min()), "k_max": float(df["k"].max()),
        "by_asset": df.groupby("asset")[["fee_adj_recalc", "fee_raw"]].sum().round(2).to_dict(orient="index"),
        "_frame": df,
    }


def fee_drag_cagr_bound(fee_excess_total: float, capital: float, years: float, avg_nav: float) -> float:
    """Approximate CAGR effect (pp) of a total $ fee difference spread evenly over the run."""
    return 100.0 * fee_excess_total / avg_nav / years


def dump_json(obj, name: str) -> Path:
    path = OUT / name
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return path


def xnys_sessions(start="1990-01-01", end="2026-12-31") -> pd.DatetimeIndex:
    import exchange_calendars as xcals
    return xcals.get_calendar("XNYS", start=start, end=end).sessions.tz_localize(None)


def month_end_sessions(sessions: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(sessions, index=sessions)
    return pd.DatetimeIndex(s.groupby(sessions.to_period("M")).max().to_numpy())


def truncation_dates(sessions: pd.DatetimeIndex, lo="2008-01-01", hi="2026-08-19") -> dict:
    """>= 6 cut-offs: mid-month, pre-month-end, month-end whose last calendar day is weekend, holiday, plain."""
    sessions = sessions[(sessions >= pd.Timestamp(lo)) & (sessions <= pd.Timestamp(hi))]
    me = month_end_sessions(sessions)
    out = {}
    out["mid_month_2013-10-22"] = pd.Timestamp("2013-10-22")
    out["mid_month_2020-03-16"] = pd.Timestamp("2020-03-16")
    # month-end where the last calendar day is a weekend (May 2020: 31st Sunday -> 29th Fri)
    out["month_end_weekend_2020-05-29"] = pd.Timestamp("2020-05-29")
    # month-end where the last calendar day is a holiday (May 2021: 31st Memorial Day -> 28th Fri)
    out["month_end_holiday_2021-05-28"] = pd.Timestamp("2021-05-28")
    out["day_before_month_end_2021-05-27"] = pd.Timestamp("2021-05-27")
    out["month_end_plain_2008-09-30"] = pd.Timestamp("2008-09-30")
    out["month_end_weekend_2022-12-30"] = pd.Timestamp("2022-12-30")
    out["mid_month_2025-04-08"] = pd.Timestamp("2025-04-08")
    for k, v in out.items():
        assert v in sessions, (k, v)
    return out
