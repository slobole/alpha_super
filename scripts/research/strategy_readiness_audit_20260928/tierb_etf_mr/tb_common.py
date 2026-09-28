"""Tier B readiness audit (2026-09-28): EOM flow + sector-ETF IBS pods. Shared factories and helpers (research-only).

Strategies (all PM_READY, NOT WIRED):
  eom      strategies.taa_beyond_6040.strategy_taa_month_end_rebalancing_flow
  vox      strategies.mean_reversion.strategy_mr_us_sector_etf_ibs_downshock_vox_iyr
  xlc      strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_xlc
  xlc200   strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_xlc_asset_sma200
  kie200   strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_asset_sma200

Every factory reproduces the module's own run_variant path (same loader, config, calendar resolver, strategy class).
Nothing here edits production code; variants are subclasses or data copies.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
for path in (REPO, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from alpha.engine.backtest import run_daily  # noqa: E402
import strategies.taa_beyond_6040.strategy_taa_month_end_rebalancing_flow as eom_mod  # noqa: E402
import strategies.mean_reversion.strategy_mr_us_sector_etf_ibs_downshock_vox_iyr as vox_mod  # noqa: E402
import strategies.mean_reversion.strategy_mr_us_sector_etf_ibs_downshock as vox_base_mod  # noqa: E402
import strategies.mean_reversion.strategy_mr_sector_dispersion_ibs as disp_base_mod  # noqa: E402
import strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_xlc as xlc_mod  # noqa: E402
import strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_xlc_asset_sma200 as xlc200_mod  # noqa: E402
import strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_asset_sma200 as kie200_mod  # noqa: E402

OUT = REPO / "results" / "research" / "strategy_readiness_audit_20260928" / "tierb_etf_mr"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
END_STR = "2026-09-25"
LAST3Y_START_STR = "2023-09-25"
MR_KEYS = ("vox", "xlc", "xlc200", "kie200")
ALL_KEYS = ("eom",) + MR_KEYS

MR_SPEC = {
    "vox": {"mod": vox_mod, "cls": vox_mod.UsSectorEtfIbsDownshockVoxIyrStrategy, "family": "downshock"},
    "xlc": {"mod": xlc_mod, "cls": xlc_mod.SectorDispersionIbsKieIhiXlcStrategy, "family": "dispersion"},
    "xlc200": {"mod": xlc200_mod, "cls": xlc200_mod.SectorDispersionIbsKieIhiXlcAssetSma200Strategy,
               "family": "dispersion", "close_hist": 200},
    "kie200": {"mod": kie200_mod, "cls": kie200_mod.SectorDispersionIbsKieIhiAssetSma200Strategy,
               "family": "dispersion", "close_hist": 200},
}


def config_for(key: str, **overrides):
    if key == "eom":
        return replace(eom_mod.DEFAULT_CONFIG, end_date_str=END_STR, **overrides)
    return replace(MR_SPEC[key]["mod"].DEFAULT_CONFIG, end_date_str=END_STR, **overrides)


def symbols_for(key: str) -> tuple[str, ...]:
    if key == "eom":
        return eom_mod.TRADED_ASSET_TUPLE
    return tuple(config_for(key).symbol_tuple)


def load_pricing(key: str, refresh: bool = False) -> pd.DataFrame:
    """Production loader output (cached as parquet; attrs kept in a side JSON)."""
    path = CACHE / f"pricing_{key}.parquet"
    meta = CACHE / f"pricing_{key}.attrs.json"
    if path.exists() and meta.exists() and not refresh:
        frame = pd.read_parquet(path)
        frame.attrs.update(json.loads(meta.read_text(encoding="utf-8")))
        return frame
    cfg = config_for(key)
    with contextlib.redirect_stderr(io.StringIO()):
        if key == "eom":
            frame = eom_mod.get_month_end_flow_data(cfg)
        elif MR_SPEC[key]["family"] == "downshock":
            frame = vox_base_mod.get_us_sector_etf_ibs_downshock_data(cfg)
        else:
            frame = disp_base_mod.get_sector_dispersion_ibs_data(config_obj=cfg)
    frame.to_parquet(path)
    meta.write_text(json.dumps(frame.attrs, default=str), encoding="utf-8")
    return frame


def calendar_for(key: str, pricing: pd.DataFrame, cfg=None) -> pd.DatetimeIndex:
    cfg = cfg or config_for(key)
    if key == "eom":
        return pricing.index[pricing.index >= pd.Timestamp(cfg.backtest_start_date_str)]
    if MR_SPEC[key]["family"] == "downshock":
        return vox_base_mod.resolve_us_sector_etf_execution_calendar_idx(pricing_data_df=pricing, config_obj=cfg)
    return disp_base_mod.resolve_full_basket_calendar_idx(
        pricing_data_df=pricing, config_obj=cfg,
        required_close_history_observation_count_int=MR_SPEC[key].get("close_hist"))


def make_strategy(key: str, cfg=None, cls=None):
    cfg = cfg or config_for(key)
    if key == "eom":
        return (cls or eom_mod.MonthEndRebalancingFlowStrategy)(cfg)
    klass = cls or MR_SPEC[key]["cls"]
    name = getattr(MR_SPEC[key]["mod"], "STRATEGY_NAME_STR")
    return klass(name=name, benchmarks=[cfg.benchmark_symbol_str], config_obj=cfg)


def run(key: str, pricing: pd.DataFrame | None = None, cfg=None, cls=None, hsu: bool = False,
        calendar: pd.DatetimeIndex | None = None, strategy=None):
    pricing = load_pricing(key) if pricing is None else pricing
    cfg = cfg or config_for(key)
    strategy = strategy or make_strategy(key, cfg, cls)
    strategy.historical_share_units_bool = bool(hsu)
    calendar = calendar_for(key, pricing, cfg) if calendar is None else calendar
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(io.StringIO()):
        run_daily(strategy, pricing, calendar, show_progress=False, show_signal_progress_bool=False,
                  audit_override_bool=False)
    strategy._captured_stdout = sink.getvalue()
    return strategy


def metrics(total_value: pd.Series, start=None, end=None) -> dict:
    nav = total_value.astype(float)
    nav.index = pd.to_datetime(nav.index)
    if start is not None:
        nav = nav.loc[pd.Timestamp(start):]
    if end is not None:
        nav = nav.loc[: pd.Timestamp(end)]
    ret = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else float("nan")
    maxdd = (nav / nav.cummax() - 1).min()
    return {"start": nav.index[0].date().isoformat(), "end": nav.index[-1].date().isoformat(),
            "cagr_pct": round(float(cagr) * 100, 4), "sharpe": round(float(sharpe), 4),
            "max_dd_pct": round(float(maxdd) * 100, 3)}


def metric_pair(strategy) -> dict:
    tv = strategy.total_value_series
    return {"full": metrics(tv), "last3y": metrics(tv, start=LAST3Y_START_STR)}


def fills(strategy) -> pd.DataFrame:
    tx = strategy.get_transactions()
    if len(tx) == 0:
        return pd.DataFrame(columns=["date", "asset", "side", "notional", "amount"])
    tx = tx.copy()
    tx["notional"] = tx["amount"].astype(float) * tx["price"].astype(float)
    tx["side"] = np.sign(tx["amount"].astype(float)).astype(int)
    grouped = tx.groupby(["bar", "asset", "side"]).agg(notional=("notional", "sum"), amount=("amount", "sum"))
    return grouped.reset_index().rename(columns={"bar": "date"})


def compare_fill_sets(ref: pd.DataFrame, cand: pd.DataFrame, rel_tol: float = 1e-9) -> dict:
    key = ["date", "asset", "side"]
    merged = ref.merge(cand, on=key, how="outer", suffixes=("_ref", "_cand"), indicator=True)
    only_ref = merged[merged["_merge"] == "left_only"]
    only_cand = merged[merged["_merge"] == "right_only"]
    both = merged[merged["_merge"] == "both"].copy()
    rel = ((both["notional_cand"] - both["notional_ref"]).abs() / both["notional_ref"].abs().clip(lower=1e-12)
           if len(both) else pd.Series(dtype=float))
    return {"n_ref": int(len(ref)), "n_cand": int(len(cand)), "n_only_ref": int(len(only_ref)),
            "n_only_cand": int(len(only_cand)), "max_rel_notional_diff": float(rel.max()) if len(rel) else 0.0,
            "n_beyond_tol": int((rel > rel_tol).sum()) if len(rel) else 0,
            "examples_only_ref": only_ref[key].head(4).astype(str).to_dict("records"),
            "examples_only_cand": only_cand[key].head(4).astype(str).to_dict("records")}


def rescale_symbol(pricing: pd.DataFrame, namespace_list, k: float) -> pd.DataFrame:
    """Protocol A2: history re-based as if a k:1 split happened after the last date.
    OHLC and Dividend / k, Volume * k; 'Unadjusted Close' and 'Turnover' nominal (unchanged)."""
    out = pricing.copy()
    for ns in namespace_list:
        for field in ("Open", "High", "Low", "Close", "Dividend"):
            if (ns, field) in out.columns:
                out[(ns, field)] = out[(ns, field)] / k
        if (ns, "Volume") in out.columns:
            out[(ns, "Volume")] = out[(ns, "Volume")] * k
    out.attrs.update(pricing.attrs)
    return out


def upcast64(pricing: pd.DataFrame) -> pd.DataFrame:
    out = pricing.astype({c: "float64" for c in pricing.columns if str(pricing[c].dtype) == "float32"})
    out.attrs.update(pricing.attrs)
    return out


def write_json(name: str, payload) -> Path:
    path = OUT / name
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
