"""CORE5 readiness audit (2026-09-28): shared helpers. Research-only; no production code is modified."""
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
REPO = HERE.parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = REPO / "results" / "research" / "strategy_readiness_audit_20260928" / "core5"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "_cache"
CACHE.mkdir(parents=True, exist_ok=True)

from strategies.taa_beyond_6040 import strategy_taa_adaptive_macro_core5 as core5  # noqa: E402

TRADEABLES = ("SPY", "IEF", "GLD", "DBC", "UUP", "BIL")
PRICE_FIELDS = ("Open", "High", "Low", "Close")


def load_pricing(refresh: bool = False) -> pd.DataFrame:
    path = CACHE / "core5_pricing_full.pkl"
    if path.exists() and not refresh:
        with path.open("rb") as fh:
            return pickle.load(fh)
    df = core5.get_adaptive_macro_core5_data(core5.DEFAULT_CONFIG)
    with path.open("wb") as fh:
        pickle.dump(df, fh)
    return df


def rescale_symbol(pricing_df: pd.DataFrame, namespace: str, k: float) -> pd.DataFrame:
    """Future k:1 split after the last row: OHLC and Dividend / k, Volume * k; Unadjusted Close/Turnover nominal."""
    out = pricing_df.copy()
    for f in PRICE_FIELDS + ("Dividend",):
        key = (namespace, f)
        if key in out.columns:
            out[key] = out[key] / k
    key = (namespace, "Volume")
    if key in out.columns:
        out[key] = out[key] * k
    return out


def metrics(ret: pd.Series, start=None, end=None) -> dict:
    r = ret.astype(float).dropna()
    if start is not None:
        r = r[r.index >= pd.Timestamp(start)]
    if end is not None:
        r = r[r.index <= pd.Timestamp(end)]
    eq = (1.0 + r).cumprod()
    years = (r.index[-1] - r.index[0]).days / 365.25
    sd = float(r.std(ddof=1))
    return {
        "cagr_pct": 100 * float(eq.iloc[-1] ** (1 / years) - 1),
        "sharpe": float(r.mean() / sd * np.sqrt(252)),
        "maxdd_pct": 100 * float((eq / eq.cummax() - 1).min()),
        "n": int(len(r)),
        "start": r.index[0].date().isoformat(),
        "end": r.index[-1].date().isoformat(),
    }


def run_backtest(pricing_df: pd.DataFrame, capital: float = 100_000.0, end_date_str=None, config=None):
    from dataclasses import replace
    cfg = config or replace(core5.DEFAULT_CONFIG, capital_base_float=float(capital), end_date_str=end_date_str)
    return core5._run_strategy(cfg, pricing_df, cfg.backtest_start_date_str, False)


def dump(obj, name: str) -> Path:
    path = OUT / name
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return path


def xnys_sessions(start="1990-01-01", end="2026-12-31") -> pd.DatetimeIndex:
    import exchange_calendars as xcals
    return xcals.get_calendar("XNYS", start=start, end=end).sessions.tz_localize(None)
