"""TAA leakage-hunt common helpers (study 2026-09-27; research-only, read-only on strategy/engine code).

What this module does
---------------------
1. Caches REAL Norgate frames (TOTALRETURN and CAPITALSPECIAL, ALLMARKETDAYS padding, full history) for every
   symbol the four audited TAA strategies touch, in the session scratchpad (pickle).
2. Installs a runtime patch of ``load_price_timeseries`` inside the strategy modules (not a source edit) so the
   UNMODIFIED strategy data/signal functions read from that cache.  The patch can
      - rescale one symbol's whole history by k (a future k:1 split / TR factor), both adjustments consistently:
        O/H/L/C/Dividend / k, Volume * k, 'Unadjusted Close' and 'Turnover' unchanged (see harness.py);
      - slice by start/end exactly like the provider call.
3. Freezes FRED DTB3 / T5YIE to the current local cache (copied to scratch) so that no network download happens
   and the shared ../1_data cache is never overwritten by the audit.  Optional replacement series can be injected
   (lagged / ALFRED vintage) for the macro-vintage study.
4. Thin adapters returning month-end target weights (after the VRP gate) for each audited strategy, using the
   strategy modules' own functions.

Usage: ``from taa_common import *`` from scripts in this folder (they add the folder to sys.path).
"""

from __future__ import annotations

import os
import pickle
import shutil
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from harness import DIVIDED_FIELD_TUPLE, MULTIPLIED_FIELD_TUPLE  # noqa: E402

SCRATCH = Path(os.environ.get(
    "TAA_SCRATCH",
    r"C:\Users\User\AppData\Local\Temp\claude\C--Users-User-Documents-workspace-alpha-super"
    r"\2b26c675-1db0-4ceb-8ed6-fb7880170526\scratchpad\taa_cache",
))
SCRATCH.mkdir(parents=True, exist_ok=True)
OUT_DIR = REPO / "results" / "research" / "leakage_hunt_20260927" / "taa"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MACRO_SRC_DIR = REPO.parent / "1_data"
FROZEN_DTB3 = SCRATCH / "DTB3_frozen.csv"
FROZEN_T5YIE = SCRATCH / "T5YIE_frozen.csv"

TAA_SYMBOLS = ("GLD", "UUP", "TLT", "DBC", "BTAL", "TQQQ", "QQQ", "SPY", "$VIX", "$SPXTR")
COMPASS_SYMBOLS = ("SPY", "XLE", "XLI", "XLF", "XLB", "XLU", "XLV", "XLP", "XLK", "IEF")
ALL_SYMBOLS = tuple(dict.fromkeys(TAA_SYMBOLS + COMPASS_SYMBOLS))

BOOK_START_STR = "2012-10-02"
BOOK_END_STR = "2026-08-19"


# ----------------------------------------------------------------------------------------------------------------
# Norgate cache
# ----------------------------------------------------------------------------------------------------------------
def _cache_path(symbol_str: str, adjustment_str: str) -> Path:
    safe = symbol_str.replace("$", "IDX_")
    return SCRATCH / f"norgate_{safe}_{adjustment_str}.pkl"


def load_norgate_full(symbol_str: str, adjustment_str: str) -> pd.DataFrame:
    """Full-history Norgate frame (direct mode, ALLMARKETDAYS padding, same call as data/norgate_loader)."""
    path = _cache_path(symbol_str, adjustment_str)
    if path.exists():
        return pd.read_pickle(path)
    import norgatedata as nd

    adj = (nd.StockPriceAdjustmentType.TOTALRETURN if adjustment_str == "TOTALRETURN"
           else nd.StockPriceAdjustmentType.CAPITALSPECIAL)
    df = nd.price_timeseries(symbol_str, stock_price_adjustment_setting=adj,
                             padding_setting=nd.PaddingType.ALLMARKETDAYS, timeseriesformat="pandas-dataframe")
    df.to_pickle(path)
    return df


def warm_cache(symbols=ALL_SYMBOLS) -> None:
    for s in symbols:
        for a in ("TOTALRETURN", "CAPITALSPECIAL"):
            load_norgate_full(s, a)


def rescale_single_frame(df: pd.DataFrame, factor_float: float) -> pd.DataFrame:
    """Single-symbol frame version of harness.rescale_symbol_history (same field rules)."""
    out = df.copy()
    for f in DIVIDED_FIELD_TUPLE:
        if f in out.columns:
            out[f] = out[f] / factor_float
    for f in MULTIPLIED_FIELD_TUPLE:
        if f in out.columns:
            out[f] = out[f] * factor_float
    return out


class PatchState:
    """Mutable patch configuration read by the fake loader."""
    rescale: dict = {}          # symbol -> factor
    hard_end_str: str | None = None   # simulate provider data ending at this date (truncation test)


def fake_load_price_timeseries(symbol_str, *, adjustment_str="CAPITALSPECIAL", start_date_str=None,
                               end_date_str=None, data_profile_str=None):
    adjustment_str = "TOTALRETURN" if str(adjustment_str).upper().startswith("TOTAL") else "CAPITALSPECIAL"
    df = load_norgate_full(symbol_str, adjustment_str)
    if symbol_str in PatchState.rescale:
        df = rescale_single_frame(df, float(PatchState.rescale[symbol_str]))
    if start_date_str is not None:
        df = df.loc[pd.Timestamp(start_date_str):]
    end_candidates = [pd.Timestamp(x) for x in (end_date_str, PatchState.hard_end_str) if x is not None]
    if end_candidates:
        df = df.loc[: min(end_candidates)]
    return df.copy()


# ----------------------------------------------------------------------------------------------------------------
# FRED freeze / injection
# ----------------------------------------------------------------------------------------------------------------
def freeze_macro_files() -> None:
    for src, dst in ((MACRO_SRC_DIR / "DTB3.csv", FROZEN_DTB3), (MACRO_SRC_DIR / "T5YIE.csv", FROZEN_T5YIE)):
        if not dst.exists():
            shutil.copy2(src, dst)


def read_frozen_series(series_id_str: str) -> pd.Series:
    path = FROZEN_DTB3 if series_id_str == "DTB3" else FROZEN_T5YIE
    df = pd.read_csv(path)
    ser = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    ser.index = pd.to_datetime(df.iloc[:, 0])
    ser = ser.dropna().sort_index()
    ser.name = series_id_str
    return ser


class MacroState:
    """Optional override: series_id -> pd.Series of values indexed by (pseudo) observation date."""
    override: dict = {}


def fake_load_daily_fred_series_snapshot(series_id_str, cache_csv_path_str, as_of_ts, mode_str):
    """Offline, frozen replacement for alpha.data.load_daily_fred_series_snapshot (same as-of filter rule)."""
    from alpha.data import FredSeriesSnapshot

    ser = MacroState.override.get(series_id_str)
    if ser is None:
        ser = read_frozen_series(series_id_str)
    as_of_date = pd.Timestamp(pd.Timestamp(as_of_ts).date()) if as_of_ts is not None else ser.index[-1]
    avail = ser[ser.index.normalize() <= as_of_date]
    return FredSeriesSnapshot(
        value_ser=avail, source_name_str="FRED_FROZEN_AUDIT", series_id_str=series_id_str,
        download_attempt_timestamp_ts=pd.Timestamp("2026-09-27").to_pydatetime(),
        download_status_str="audit_frozen_cache", latest_observation_date_ts=pd.Timestamp(avail.index[-1]),
        used_cache_bool=True, freshness_business_days_int=0,
    )


_PATCHED = False


def install_patches() -> None:
    """Patch module-level names used by the audited strategies (runtime only; no file edits)."""
    global _PATCHED
    if _PATCHED:
        return
    freeze_macro_files()
    import strategies.taa_df.strategy_taa_df as base
    import strategies.taa_df.strategy_taa_df_fallback_vix_cash_variant_utils as vix
    import strategies.taa_df.strategy_taa_inflation_compass as compass

    base.load_price_timeseries = fake_load_price_timeseries
    vix.load_price_timeseries = fake_load_price_timeseries
    base.load_daily_fred_series_snapshot = fake_load_daily_fred_series_snapshot
    compass.load_daily_fred_series_snapshot = fake_load_daily_fred_series_snapshot
    # quiet the per-row tqdm in compute_month_end_weight_df
    base.tqdm = lambda it, *a, **k: it
    _PATCHED = True


@contextmanager
def patched(rescale: dict | None = None, hard_end_str: str | None = None, macro_override: dict | None = None):
    install_patches()
    old = (dict(PatchState.rescale), PatchState.hard_end_str, dict(MacroState.override))
    PatchState.rescale = dict(rescale or {})
    PatchState.hard_end_str = hard_end_str
    MacroState.override = dict(macro_override or {})
    try:
        yield
    finally:
        PatchState.rescale, PatchState.hard_end_str, MacroState.override = old


# ----------------------------------------------------------------------------------------------------------------
# Strategy adapters -> month-end target weights (after the VRP gate), rebalance weights, diagnostics
# ----------------------------------------------------------------------------------------------------------------
STRATEGY_KEYS = ("taa3x_rank", "taa3x_1n", "btal_qqq_linearity", "compass")
MODULE_BY_KEY = {
    "taa3x_rank": "strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash",
    "taa3x_1n": "strategies.taa_df.strategy_taa_df_btal_1n_fallback_tqqq_vix_cash",
    "btal_qqq_linearity": "strategies.taa_df.strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash",
    "compass": "strategies.taa_df.strategy_taa_inflation_compass",
}


def get_config(key: str, end_date_str: str | None = None):
    import importlib
    install_patches()
    mod = importlib.import_module(MODULE_BY_KEY[key])
    cfg = mod.DEFAULT_CONFIG
    if end_date_str is not None:
        cfg = replace(cfg, end_date_str=end_date_str)
    return cfg


def compute_decisions(key: str, end_date_str: str | None = None) -> dict:
    """Return dict with month_end_weight_df (decision labels = calendar month-end), rebalance_weight_df,
    execution_price_df and a per-month diagnostic frame (scores / hurdle / VRP)."""
    install_patches()
    cfg = get_config(key, end_date_str)
    if key in ("taa3x_rank", "taa3x_1n"):
        import strategies.taa_df.strategy_taa_df as base
        import strategies.taa_df.strategy_taa_df_fallback_vix_cash_variant_utils as vix
        (execution_price_df, momentum_score_df, daily_vrp_signal_df, month_end_weight_df, rebalance_weight_df,
         vrp_diag_df) = vix.get_standard_fallback_vix_cash_data(config=cfg, base_data_loader_fn=base.get_defense_first_data)
        return dict(month_end_weight_df=month_end_weight_df, rebalance_weight_df=rebalance_weight_df,
                    execution_price_df=execution_price_df, score_df=momentum_score_df, vrp_diag_df=vrp_diag_df,
                    daily_vrp_signal_df=daily_vrp_signal_df, config=cfg)
    if key == "btal_qqq_linearity":
        import strategies.taa_df.strategy_taa_df_btal_linearity_1n as lin
        import strategies.taa_df.strategy_taa_df_fallback_vix_cash_variant_utils as vix
        (execution_price_df, daily_score_df, month_end_score_df, daily_vrp_signal_df, month_end_weight_df,
         rebalance_weight_df, vrp_diag_df) = vix.get_linearity_1n_fallback_vix_cash_data(
            config=cfg, base_data_loader_fn=lin.get_defense_first_linearity_1n_data)
        return dict(month_end_weight_df=month_end_weight_df, rebalance_weight_df=rebalance_weight_df,
                    execution_price_df=execution_price_df, score_df=month_end_score_df, vrp_diag_df=vrp_diag_df,
                    daily_vrp_signal_df=daily_vrp_signal_df, config=cfg)
    if key == "compass":
        import strategies.taa_df.strategy_taa_inflation_compass as compass
        (execution_price_df, month_end_feature_df, month_end_weight_df, rebalance_weight_df,
         _snap) = compass.get_inflation_compass_data(cfg)
        return dict(month_end_weight_df=month_end_weight_df, rebalance_weight_df=rebalance_weight_df,
                    execution_price_df=execution_price_df, score_df=month_end_feature_df, vrp_diag_df=None,
                    config=cfg)
    raise KeyError(key)


def to_decision_session_index(month_end_weight_df: pd.DataFrame, session_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Relabel calendar month-end labels to the last session <= label (for readable reports)."""
    out = month_end_weight_df.copy()
    sess = pd.DatetimeIndex(session_index).sort_values()
    new_idx = []
    for ts in out.index:
        pos = sess.searchsorted(pd.Timestamp(ts), side="right") - 1
        new_idx.append(sess[pos] if pos >= 0 else pd.Timestamp(ts))
    out.index = pd.DatetimeIndex(new_idx)
    return out


def run_backtest_from_weights(key: str, decisions: dict, start_str: str = BOOK_START_STR,
                              historical_share_units_bool: bool = False, capital_base_float: float = 100_000.0):
    """Replicates run_standard_fallback_vix_cash_variant / run_linearity_1n_fallback_vix_cash_variant's run path
    (same strategy factory, same calendar clip) but lets the audit set engine flags before run_daily."""
    import strategies.taa_df.strategy_taa_df_fallback_variant_utils as fu
    cfg = decisions["config"]
    strat = fu._build_defense_first_strategy(strategy_name_str=key, config=cfg,
                                             rebalance_weight_df=decisions["rebalance_weight_df"],
                                             capital_base_float=capital_base_float)
    if historical_share_units_bool:
        strat.historical_share_units_bool = True
    fu._run_strategy_from_weight_df(strategy=strat, execution_price_df=decisions["execution_price_df"],
                                    rebalance_weight_df=decisions["rebalance_weight_df"],
                                    backtest_start_date_str=start_str)
    return strat


def metrics_from_strategy(strat) -> dict:
    s = strat.summary["Strategy"]
    tv = strat.results["total_value"].astype(float)
    r = tv.pct_change().dropna()
    years = (tv.index[-1] - tv.index[0]).days / 365.25
    tx = strat.get_transactions() if hasattr(strat, "get_transactions") else strat._transactions
    return {
        "start": str(tv.index[0].date()), "end": str(tv.index[-1].date()),
        "final_value": float(tv.iloc[-1]),
        "cagr_pct_calc": float(((tv.iloc[-1] / tv.iloc[0]) ** (1 / years) - 1) * 100),
        "sharpe_calc": float(r.mean() / r.std(ddof=1) * np.sqrt(252)),
        "summary_return_ann_pct": float(s.get("Return (Ann.) [%]", np.nan)),
        "summary_sharpe": float(s.get("Sharpe Ratio", np.nan)),
        "summary_max_dd_pct": float(s.get("Max. Drawdown [%]", np.nan)),
        "total_commissions": float(pd.to_numeric(tx["commission"], errors="coerce").sum()),
        "n_transactions": int(len(tx)),
    }
