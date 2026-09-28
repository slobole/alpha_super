"""Tier C audit helpers for the four PM_READY TAA 2x / linearity variants. Research-only, no repository file changed.

Runtime patches (study only):
- every variant's DEFAULT_CONFIG.dtb3_csv_path_str points to a COPY of the audit's DTB3 cache in this audit's results
  folder, and ``alpha.data.fred_loader.urlopen`` raises URLError, so the loader falls back to that copy and the
  owner's shared ``1_data/DTB3.csv`` is never read or written;
- optional cached Norgate loader: the full history (to 2026-09-25) is loaded once per (symbol, adjustment, start) and
  later calls return ``full.loc[:end]``. ``verify_loader_slices`` proves on real data that this equals a fresh
  truncated Norgate load for every symbol at >= 8 cut-offs, so row-T tests on the cache are loader-faithful.
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")

import sys
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from urllib.error import URLError

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
for _p in (REPO, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import alpha.data.fred_loader as fred_loader_module  # noqa: E402

OUT = REPO / "results" / "research" / "strategy_readiness_audit_20260928" / "tierbc_dv2etf_taa2x" / "taa2x"
OUT.mkdir(parents=True, exist_ok=True)
DTB3_CACHE_STR = str(OUT.parent / "DTB3_tierbc_cache.csv")
END_DATE_STR = "2026-09-25"


def _offline(*args, **kwargs):
    raise URLError("audit: network disabled, use the copied DTB3 cache")


fred_loader_module.urlopen = _offline

base_module = import_module("strategies.taa_df.strategy_taa_df")
utils_module = import_module("strategies.taa_df.strategy_taa_df_fallback_vix_cash_variant_utils")
fallback_utils_module = import_module("strategies.taa_df.strategy_taa_df_fallback_variant_utils")
linearity_1n_module = import_module("strategies.taa_df.strategy_taa_df_btal_linearity_1n")

VARIANT_DICT = {
    "qld_1n": {"module": "strategies.taa_df.strategy_taa_df_1n_fallback_qld_vix_cash", "kind": "standard"},
    "sso_1n": {"module": "strategies.taa_df.strategy_taa_df_1n_fallback_sso_vix_cash", "kind": "standard"},
    "btal_qld_1n": {"module": "strategies.taa_df.strategy_taa_df_btal_1n_fallback_qld_vix_cash", "kind": "standard"},
    "lin_qqq": {"module": "strategies.taa_df.strategy_taa_df_linearity_1n_fallback_qqq_vix_cash", "kind": "linearity"},
}

for _v in VARIANT_DICT.values():
    _m = import_module(_v["module"])
    _m.DEFAULT_CONFIG = replace(_m.DEFAULT_CONFIG, dtb3_csv_path_str=DTB3_CACHE_STR)

_REAL_LOADER = base_module.load_price_timeseries
_CACHE: dict[tuple, pd.DataFrame] = {}


def cached_loader(symbol_str, adjustment_str="CAPITALSPECIAL", start_date_str=None, end_date_str=None, **kwargs):
    key = (symbol_str, str(adjustment_str), str(start_date_str), tuple(sorted(kwargs.items())))
    if key not in _CACHE:
        _CACHE[key] = _REAL_LOADER(symbol_str, adjustment_str=adjustment_str, start_date_str=start_date_str,
                                   end_date_str=END_DATE_STR, **kwargs)
    full_df = _CACHE[key]
    if end_date_str is None:
        return full_df.copy()
    return full_df.loc[: pd.Timestamp(end_date_str)].copy()


def use_cached_loader(on: bool = True) -> None:
    fn = cached_loader if on else _REAL_LOADER
    base_module.load_price_timeseries = fn
    utils_module.load_price_timeseries = fn


def config(key: str, end_date_str: str | None = END_DATE_STR):
    module_obj = import_module(VARIANT_DICT[key]["module"])
    return replace(module_obj.DEFAULT_CONFIG, end_date_str=end_date_str, dtb3_csv_path_str=DTB3_CACHE_STR)


def weight_frames(key: str, config_obj) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(month_end_weight_df after the VIX gate, rebalance_weight_df, daily_vrp_signal_df, vrp diagnostic df)."""
    if VARIANT_DICT[key]["kind"] == "standard":
        t = utils_module.get_standard_fallback_vix_cash_data(config=config_obj, base_data_loader_fn=base_module.get_defense_first_data)
        return t[3], t[4], t[2], t[5]
    t = utils_module.get_linearity_1n_fallback_vix_cash_data(
        config=config_obj, base_data_loader_fn=linearity_1n_module.get_defense_first_linearity_1n_data)
    return t[4], t[5], t[3], t[6]


def run(key: str, capital: float = 100_000.0, hsu: bool = False, end_date_str: str = END_DATE_STR):
    module_obj = import_module(VARIANT_DICT[key]["module"])
    real_build = utils_module._build_defense_first_strategy

    def build(*args, **kwargs):
        strategy = real_build(*args, **kwargs)
        strategy.historical_share_units_bool = hsu
        return strategy

    utils_module._build_defense_first_strategy = build
    try:
        return module_obj.run_variant(show_display_bool=False, save_results_bool=False, end_date_str=end_date_str,
                                      capital_base_float=capital)
    finally:
        utils_module._build_defense_first_strategy = real_build


def metrics(total_value: pd.Series, start=None, end=None) -> dict:
    nav = total_value.astype(float)
    nav.index = pd.to_datetime(nav.index)
    if start is not None:
        nav = nav.loc[pd.Timestamp(start):]
    if end is not None:
        nav = nav.loc[: pd.Timestamp(end)]
    ret = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    return {"start": nav.index[0].date().isoformat(), "end": nav.index[-1].date().isoformat(),
            "cagr": float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1),
            "sharpe": float(ret.mean() / ret.std() * np.sqrt(252)),
            "max_dd": float((nav / nav.cummax() - 1).min())}


def xnys_sessions() -> pd.DatetimeIndex:
    import exchange_calendars

    cal = exchange_calendars.get_calendar("XNYS", start="1999-01-04", end="2027-12-31")
    return pd.DatetimeIndex(cal.sessions).tz_localize(None)


def last_session_of_month(sessions: pd.DatetimeIndex, label_ts: pd.Timestamp) -> pd.Timestamp:
    month_sessions = sessions[sessions.to_period("M") == pd.Timestamp(label_ts).to_period("M")]
    return pd.Timestamp(month_sessions[-1])
