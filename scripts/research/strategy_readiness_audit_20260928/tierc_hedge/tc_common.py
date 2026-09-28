"""Tier C hedge audit (2026-09-28): shared helpers. Research-only; no production code is modified.

Strategies:
  CTC   strategies.tail_hedge.strategy_crisis_trend_core
  VIXM  strategies.tail_hedge.strategy_vixm_backwardation
  TRIN  strategies.taa_beyond_6040.strategy_taa_trinity_vol_control_8_bil
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = REPO / "results" / "research" / "strategy_readiness_audit_20260928" / "tierc_hedge"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "_cache"
CACHE.mkdir(parents=True, exist_ok=True)

from alpha.engine.backtest import run_daily  # noqa: E402
from strategies.tail_hedge import strategy_crisis_trend_core as ctc  # noqa: E402
from strategies.tail_hedge import strategy_vixm_backwardation as vixm  # noqa: E402
from strategies.taa_beyond_6040 import strategy_taa_trinity_vol_control_8_bil as trin  # noqa: E402
from strategies.taa_beyond_6040 import strategy_taa_beyond_6040 as b6040  # noqa: E402

PRICE_FIELDS = ("Open", "High", "Low", "Close")


def dump(obj, name: str) -> Path:
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return path


def _cached(name: str, fn, refresh: bool = False):
    path = CACHE / name
    if path.exists() and not refresh:
        with path.open("rb") as fh:
            return pickle.load(fh)
    obj = fn()
    with path.open("wb") as fh:
        pickle.dump(obj, fh)
    return obj


def load_ctc(refresh: bool = False) -> pd.DataFrame:
    return _cached("ctc_pricing.pkl", lambda: ctc.get_crisis_trend_core_data(ctc.DEFAULT_CONFIG), refresh)


# HEAD's run_variant raises on real data (finding C-CTC-01): SHY TR starts 2002-07-26, the loader starts 2002-01-01
# and the first eligible row (2003-01-13) needs SHY at T-252. The audit workaround trims the loaded frame to start at
# SHY's first bar; nothing in the strategy code changes.
CTC_WORKAROUND_START_STR = "2002-07-26"


def load_ctc_workaround(refresh: bool = False) -> pd.DataFrame:
    df = load_ctc(refresh)
    out = df.loc[df.index >= pd.Timestamp(CTC_WORKAROUND_START_STR)].copy()
    out.attrs.update(df.attrs)
    return out


def load_vixm(refresh: bool = False) -> pd.DataFrame:
    return _cached("vixm_pricing.pkl", lambda: vixm.get_vixm_backwardation_data(vixm.DEFAULT_CONFIG), refresh)


def load_trin(refresh: bool = False) -> pd.DataFrame:
    return _cached("trin_pricing.pkl", lambda: b6040.get_beyond_6040_data(config=trin.DEFAULT_CONFIG), refresh)


def rescale_namespace(pricing_df: pd.DataFrame, namespace: str, k: float, fields=PRICE_FIELDS + ("Dividend",)) -> pd.DataFrame:
    """Future k:1 split after the last row: OHLC and Dividend / k, Volume * k; Unadjusted Close/Turnover nominal."""
    out = pricing_df.copy()
    for f in fields:
        key = (namespace, f)
        if key in out.columns:
            out[key] = out[key] / k
    key = (namespace, "Volume")
    if key in out.columns:
        out[key] = out[key] * k
    return out


def run_ctc(pricing_df: pd.DataFrame, capital: float = 100_000.0, start: str | None = None, strategy_cls=None):
    cfg = replace(ctc.DEFAULT_CONFIG, capital_base_float=float(capital))
    cal = ctc.build_execution_calendar_idx(pricing_df, start or cfg.backtest_start_date_str)
    s = (strategy_cls or ctc.CrisisTrendCoreStrategy)(cfg)
    run_daily(s, pricing_df, calendar=cal, show_progress=False, show_signal_progress_bool=False)
    return s


def run_vixm(pricing_df: pd.DataFrame, capital: float = 100_000.0, start: str | None = None, strategy_cls=None):
    cfg = replace(vixm.DEFAULT_CONFIG, capital_base_float=float(capital))
    cal = vixm.build_execution_calendar_idx(pricing_df, start or cfg.backtest_start_date_str)
    s = (strategy_cls or vixm.VixmBackwardationStrategy)(cfg)
    run_daily(s, pricing_df, calendar=cal, show_progress=False, show_signal_progress_bool=False)
    return s


def run_trin(pricing_df: pd.DataFrame, capital: float = 100_000.0, start: str | None = None, strategy_cls=None,
             **kwargs):
    cfg = replace(trin.DEFAULT_CONFIG, capital_base_float=float(capital))
    cal = trin._execution_calendar_index(pricing_df, cfg, start)
    s = trin._build_trinity_strategy(cfg, float(capital), strategy_cls or trin.TrinityVolControlStrategy)
    for k, v in kwargs.items():
        setattr(s, k, v)
    run_daily(s, pricing_df, calendar=cal, show_progress=False, show_signal_progress_bool=False, audit_override_bool=False)
    return s


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
        "sharpe": float(r.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan"),
        "vol_pct": 100 * sd * np.sqrt(252),
        "maxdd_pct": 100 * float((eq / eq.cummax() - 1).min()),
        "n": int(len(r)),
        "start": r.index[0].date().isoformat(),
        "end": r.index[-1].date().isoformat(),
    }


def strat_returns(s) -> pd.Series:
    tv = s.total_value_series.astype(float)
    tv.index = pd.to_datetime(tv.index)
    return tv.pct_change().dropna()


def xnys_sessions(start="1990-01-01", end="2026-12-31") -> pd.DatetimeIndex:
    import exchange_calendars as xcals
    return xcals.get_calendar("XNYS", start=start, end=end).sessions.tz_localize(None)
