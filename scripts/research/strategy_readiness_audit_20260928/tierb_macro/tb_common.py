"""Shared helpers for the Tier B macro audit (Inflation Compass, Compass QQQ, Tactical FI).

Audit-only. Nothing here edits production code. FRED caches are redirected to the audit results folder so the
owner's shared ../1_data caches are never written:

- Compass: ``t5yie_csv_path_str`` is replaced at runtime, and after one fresh download the T5YIE series is frozen in
  ``OUT/T5YIE_audit_frozen.csv`` and injected through ``load_t5yie_snapshot`` so every audit run uses the same bytes.
- Tactical FI reads only its hash-locked repo files and ALFRED snapshot (it never calls the shared FRED loader).

Run from the worktree root with ``uv run python scripts/research/strategy_readiness_audit_20260928/tierb_macro/<x>.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = REPO / "results" / "research" / "strategy_readiness_audit_20260928" / "tierb_macro"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "_cache"
CACHE.mkdir(parents=True, exist_ok=True)

T5YIE_DOWNLOAD_CACHE = OUT / "T5YIE_audit_download_cache.csv"
T5YIE_FROZEN = OUT / "T5YIE_audit_frozen.csv"

NORGATE_LAST_BAR_STR = "2026-09-25"
MAIN_START_STR = "2003-05-01"
MAIN_END_STR = "2026-08-19"

import strategies.taa_df.strategy_taa_inflation_compass as cmp_mod  # noqa: E402
import strategies.taa_df.strategy_taa_inflation_compass_qqq as cmpq_mod  # noqa: E402
from alpha.data import FredSeriesSnapshot, load_daily_fred_series_snapshot  # noqa: E402


def write_json(name_str: str, obj) -> Path:
    path = OUT / name_str
    path.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    return path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------------------------
def metrics(nav: pd.Series, start=None, end=None) -> dict:
    """Same definitions as the Compass deep research (daily Sharpe rf=0, CAGR on 365.25-day years)."""
    s = nav.astype(float)
    if start is not None:
        s = s[s.index >= pd.Timestamp(start)]
    if end is not None:
        s = s[s.index <= pd.Timestamp(end)]
    r = s.pct_change().dropna()
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    dd = s / s.cummax() - 1
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1
    return {
        "start": str(s.index[0].date()),
        "end": str(s.index[-1].date()),
        "cagr_pct": float(cagr * 100),
        "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(252)),
        "maxdd_pct": float(dd.min() * 100),
    }


# ---------------------------------------------------------------------------------------------------------------
# T5YIE (Compass)
# ---------------------------------------------------------------------------------------------------------------
def ensure_frozen_t5yie() -> pd.Series:
    """Download T5YIE once into the audit folder (never ../1_data) and freeze it."""
    if not T5YIE_FROZEN.exists():
        snap = load_daily_fred_series_snapshot(
            series_id_str="T5YIE",
            cache_csv_path_str=str(T5YIE_DOWNLOAD_CACHE),
            as_of_ts=datetime.now(tz=UTC),
            mode_str="backtest",
        )
        frozen_df = snap.value_ser.rename("T5YIE").to_frame()
        frozen_df.index.name = "observation_date"
        frozen_df.to_csv(T5YIE_FROZEN)
        write_json(
            "t5yie_frozen_provenance.json",
            {
                "download_status": snap.download_status_str,
                "download_utc": snap.download_attempt_timestamp_ts.isoformat(),
                "latest_observation": str(snap.latest_observation_date_ts.date()),
                "n_obs": int(len(snap.value_ser)),
                "sha256_frozen_csv": sha256_file(T5YIE_FROZEN),
            },
        )
    frozen_df = pd.read_csv(T5YIE_FROZEN, index_col=0, parse_dates=True)
    ser = frozen_df["T5YIE"].astype(float)
    ser.index = pd.DatetimeIndex(ser.index).normalize()
    ser.name = "T5YIE"
    return ser


def frozen_t5yie_snapshot(end_date_str: str | None = None) -> FredSeriesSnapshot:
    ser = ensure_frozen_t5yie()
    if end_date_str is not None:
        ser = ser[ser.index <= pd.Timestamp(end_date_str)]
    return FredSeriesSnapshot(
        value_ser=ser,
        source_name_str="FRED",
        series_id_str="T5YIE",
        download_attempt_timestamp_ts=datetime(2026, 9, 28, tzinfo=UTC),
        download_status_str="audit_frozen_copy",
        latest_observation_date_ts=pd.Timestamp(ser.index[-1]),
        used_cache_bool=True,
        freshness_business_days_int=0,
    )


_ORIGINAL_LOAD_T5YIE = cmp_mod.load_t5yie_snapshot


def install_frozen_t5yie() -> None:
    """Route every Compass T5YIE load to the frozen audit copy (and never to ../1_data)."""

    def _patched(config_obj):
        return frozen_t5yie_snapshot(None)

    cmp_mod.load_t5yie_snapshot = _patched
    safe_path_str = str(T5YIE_DOWNLOAD_CACHE)
    cmp_mod.DEFAULT_CONFIG = replace(cmp_mod.DEFAULT_CONFIG, t5yie_csv_path_str=safe_path_str)
    cmpq_mod.DEFAULT_CONFIG = replace(cmpq_mod.DEFAULT_CONFIG, t5yie_csv_path_str=safe_path_str)


def compass_config(variant_str: str):
    install_frozen_t5yie()
    if variant_str == "xlk":
        return cmp_mod.DEFAULT_CONFIG
    if variant_str == "qqq":
        return cmpq_mod.DEFAULT_CONFIG
    raise ValueError(variant_str)


# ---------------------------------------------------------------------------------------------------------------
# Norgate data (cached pickles; one local vintage, last bar 2026-09-25)
# ---------------------------------------------------------------------------------------------------------------
def _cache_path(name_str: str) -> Path:
    return CACHE / f"{name_str}.pkl"


def compass_signal_close_df(end_date_str: str | None = None) -> pd.DataFrame:
    path = _cache_path("compass_signal_close")
    if path.exists():
        df = pd.read_pickle(path)
    else:
        df = cmp_mod.load_signal_close_df(
            symbol_list=cmp_mod.SIGNAL_ASSET_TUPLE, start_date_str="2002-01-01", end_date_str=None
        )
        df.to_pickle(path)
    if end_date_str is not None:
        df = df.loc[: pd.Timestamp(end_date_str)]
    return df.copy()


def compass_execution_price_df(variant_str: str, end_date_str: str | None = None) -> pd.DataFrame:
    config_obj = compass_config(variant_str)
    path = _cache_path(f"compass_exec_{variant_str}")
    if path.exists():
        df = pd.read_pickle(path)
    else:
        df = cmp_mod.load_execution_price_df(
            tradeable_asset_list=config_obj.tradeable_asset_tuple,
            benchmark_list=config_obj.benchmark_tuple,
            start_date_str="2002-01-01",
            end_date_str=None,
        )
        df.to_pickle(path)
    if end_date_str is not None:
        df = df.loc[: pd.Timestamp(end_date_str)]
    return df.copy()


def run_compass_engine(
    variant_str: str,
    month_end_weight_df: pd.DataFrame | None = None,
    execution_price_df: pd.DataFrame | None = None,
    capital_base_float: float = 100_000.0,
    backtest_start_date_str: str | None = MAIN_START_STR,
    end_date_str: str | None = MAIN_END_STR,
    t5yie_ser: pd.Series | None = None,
    signal_close_df: pd.DataFrame | None = None,
    historical_share_units_bool: bool = False,
    withholding_rate_float: float | None = None,
):
    """Run the module's own engine path on audit-controlled inputs (weights may be injected)."""
    config_obj = replace(compass_config(variant_str), capital_base_float=capital_base_float, end_date_str=end_date_str)
    if execution_price_df is None:
        execution_price_df = compass_execution_price_df(variant_str, end_date_str)
    if month_end_weight_df is None:
        if signal_close_df is None:
            signal_close_df = compass_signal_close_df(end_date_str)
        if t5yie_ser is None:
            t5yie_ser = ensure_frozen_t5yie()
        month_end_feature_df, month_end_weight_df = cmp_mod.compute_month_end_signal_and_weight_df(
            signal_close_df, t5yie_ser, config_obj
        )
    else:
        month_end_feature_df = pd.DataFrame(index=month_end_weight_df.index)
    rebalance_weight_df = cmp_mod.map_month_end_weights_to_rebalance_open_df(
        month_end_weight_df, execution_price_df.index
    )
    strategy_obj = cmp_mod._build_strategy_obj(config_obj, rebalance_weight_df)
    if historical_share_units_bool:
        strategy_obj.historical_share_units_bool = True
    if withholding_rate_float is not None:
        strategy_obj.configure_dividend_cash_ledger(enabled_bool=None, withholding_rate_float=withholding_rate_float)
    strategy_obj.month_end_feature_df = month_end_feature_df
    strategy_obj.month_end_weight_df = month_end_weight_df
    strategy_obj.daily_target_weights = rebalance_weight_df.reindex(execution_price_df.index).ffill().dropna()
    calendar_index = cmp_mod._execution_calendar_index(execution_price_df, rebalance_weight_df, backtest_start_date_str)
    from alpha.engine.backtest import run_daily

    run_daily(strategy_obj, execution_price_df, calendar=calendar_index, show_progress=False,
              show_signal_progress_bool=False, audit_override_bool=None)
    return strategy_obj


def nav_ser(strategy_obj) -> pd.Series:
    ser = strategy_obj.results["total_value"].astype(float)
    ser.index = pd.to_datetime(ser.index)
    return ser


def month_end_weights(variant_str: str, signal_close_df=None, t5yie_ser=None, config_obj=None):
    if config_obj is None:
        config_obj = compass_config(variant_str)
    if signal_close_df is None:
        signal_close_df = compass_signal_close_df()
    if t5yie_ser is None:
        t5yie_ser = ensure_frozen_t5yie()
    return cmp_mod.compute_month_end_signal_and_weight_df(signal_close_df, t5yie_ser, config_obj)
