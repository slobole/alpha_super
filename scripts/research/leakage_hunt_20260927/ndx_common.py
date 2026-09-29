"""NDX leakage hunt (2026-09-27): shared helpers.  Research-only; no strategy/engine source edits.

Scope: strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled (live NDX rule, 'atr_vxn') and
strategies.momentum.strategy_mo_natr20_ndx_vxn_scaled ('natr_vxn').

Data are loaded ONCE through each module's own loader and pickled in the scratch cache.  Two universe variants:
  trimmed   : data/norgate_loader.build_index_constituent_matrix as committed (idx.iloc[:-5] on past members)
  untrimmed : identical code without the iloc[:-5] trim (monkeypatched into the module namespace at runtime)
"""

from __future__ import annotations

import importlib
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

OUT = REPO / "results" / "research" / "leakage_hunt_20260927" / "ndx"
CACHE = Path(os.environ.get(
    "NDX_CACHE_DIR",
    r"C:\Users\User\AppData\Local\Temp\claude\C--Users-User-Documents-workspace-alpha-super"
    r"\2b26c675-1db0-4ceb-8ed6-fb7880170526\scratchpad\ndx_cache",
))
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

STUDY_END = pd.Timestamp("2026-08-19")
EXACT_START = pd.Timestamp("2012-10-02")

MODULES = {
    "atr_vxn": dict(
        module="strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled",
        cls="VxnScaledAtrNormalizedNdxStrategy",
        loader="get_vxn_scaled_atr_normalized_ndx_data",
        name="strategy_mo_atr_normalized_ndx_vxn_scaled",
        sleeve_alias="ndx_atrfix",
        audit_arm=("vxn", "asof_atr_corrected"),
    ),
    "natr_vxn": dict(
        module="strategies.momentum.strategy_mo_natr20_ndx_vxn_scaled",
        cls="Natr20VxnScaledNdxStrategy",
        loader="get_natr20_vxn_scaled_ndx_data",
        name="strategy_mo_natr20_ndx_vxn_scaled",
        sleeve_alias="ndx_natr20",
        audit_arm=("vxn", "natr20_corrected"),
    ),
}
AUDIT_RUNS = Path(r"C:\Users\User\Documents\Codex\2026-09-26\new-chat\outputs\leakage-audit\runs")


def mod(key: str):
    return importlib.import_module(MODULES[key]["module"])


# ─── untrimmed universe ───────────────────────────────────────────────────────


def build_untrimmed_index_constituent_matrix(indexname: str = "Nasdaq 100"):
    """data/norgate_loader.build_index_constituent_matrix minus the `idx.iloc[:-5]` trim (direct Norgate)."""
    import norgatedata
    symbols = norgatedata.watchlist_symbols(f"{indexname} Current & Past")
    frames = []
    for symbol in symbols:
        idx = norgatedata.index_constituent_timeseries(symbol, indexname, timeseriesformat="pandas-dataframe")
        if idx["Index Constituent"].sum() > 0:
            idx = idx.rename(columns={"Index Constituent": symbol})
            idx = idx.loc[idx[symbol] == 1]
            frames.append(idx)
    universe_df = pd.concat(frames, axis=1).fillna(0).astype(int).sort_index()
    return symbols, universe_df


# ─── cached data ──────────────────────────────────────────────────────────────


def cache_path(variant: str) -> Path:
    return CACHE / f"ndx_data_{variant}.pkl"


def build_data(variant: str, key: str = "natr_vxn") -> dict:
    """Call the module's own loader (full history to latest Norgate date, TR benchmark included)."""
    module = mod(key)
    loader = getattr(module, MODULES[key]["loader"])
    original = module.build_index_constituent_matrix if hasattr(module, "build_index_constituent_matrix") else None
    base_module = importlib.import_module("strategies.momentum.strategy_mo_atr_normalized_ndx")
    base_original = base_module.build_index_constituent_matrix
    try:
        if variant == "untrimmed":
            if original is not None:
                module.build_index_constituent_matrix = build_untrimmed_index_constituent_matrix
            base_module.build_index_constituent_matrix = build_untrimmed_index_constituent_matrix
        t0 = time.time()
        pricing_df, universe_df, schedule_df, vxn_df = loader(module.DEFAULT_CONFIG, include_total_return_benchmark_bool=True)
        print(f"loaded {key}/{variant} in {time.time() - t0:.0f}s: pricing {pricing_df.shape}, universe {universe_df.shape}")
    finally:
        if original is not None:
            module.build_index_constituent_matrix = original
        base_module.build_index_constituent_matrix = base_original
    return {"pricing": pricing_df, "universe": universe_df, "schedule": schedule_df, "vxn": vxn_df,
            "attrs": dict(pricing_df.attrs)}


def load_data(variant: str = "trimmed") -> dict:
    path = cache_path(variant)
    if not path.exists():
        data = build_data(variant)
        with path.open("wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with path.open("rb") as handle:
        data = pickle.load(handle)
    data["pricing"].attrs.update(data.get("attrs", {}))
    return data


# ─── decisions without the engine ─────────────────────────────────────────────


def make_strategy(key: str, schedule_df: pd.DataFrame, vxn_df: pd.DataFrame, universe_df: pd.DataFrame,
                  capital_base: float | None = None):
    module = mod(key)
    cfg = module.DEFAULT_CONFIG
    cls = getattr(module, MODULES[key]["cls"])
    strategy = cls(
        name=MODULES[key]["name"],
        benchmarks=[cfg.performance_benchmark_symbol_str],
        rebalance_schedule_df=schedule_df,
        vxn_scale_signal_df=vxn_df,
        regime_symbol_str=cfg.regime_symbol_str,
        capital_base=cfg.capital_base_float if capital_base is None else capital_base,
        slippage=cfg.slippage_float,
        commission_per_share=cfg.commission_per_share_float,
        commission_minimum=cfg.commission_minimum_float,
        lookback_month_int=cfg.lookback_month_int,
        index_trend_window_int=cfg.index_trend_window_int,
        stock_trend_window_int=cfg.stock_trend_window_int,
        max_positions_int=cfg.max_positions_int,
    )
    strategy.universe_df = universe_df
    module.configure_total_return_benchmark_provenance(strategy_obj=strategy, config_obj=cfg)
    strategy.trade_id_int = 0
    strategy.current_trade_map = defaultdict(module.default_trade_id_int)
    return strategy


def _dummy_schedule(decision_ts: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame({"decision_date_ts": [pd.Timestamp(decision_ts)]},
                        index=pd.DatetimeIndex([pd.Timestamp(decision_ts) + pd.Timedelta(days=1)], name="execution_date_ts"))


def signals(key: str, pricing_df: pd.DataFrame, universe_df: pd.DataFrame, vxn_df: pd.DataFrame,
            decision_ts=None):
    """compute_signals once on the given frame; returns (strategy, signal_df)."""
    decision_ts = pd.Timestamp(decision_ts) if decision_ts is not None else pricing_df.index[-1]
    strategy = make_strategy(key, _dummy_schedule(decision_ts), vxn_df, universe_df)
    signal_df = strategy.compute_signals(pricing_df.copy())
    return strategy, signal_df


def decision_at(strategy, signal_df: pd.DataFrame, decision_ts, top_n: int = 20) -> dict:
    """Mirror the live host: previous_bar = decision date, then rank / target weights from the close row."""
    decision_ts = pd.Timestamp(decision_ts)
    row = signal_df.loc[decision_ts]
    strategy.previous_bar = decision_ts
    ranked = strategy.get_ranked_candidate_feature_df(close_row_ser=row)
    weights = strategy.get_target_weight_ser(close_row_ser=row)
    return {
        "date": decision_ts.date().isoformat(),
        "selected": sorted(weights.index.astype(str).tolist()),
        "weights": {str(k): float(v) for k, v in weights.sort_index().items()},
        "exposure": float(weights.sum()),
        "top_rank": [str(s) for s in ranked.index[:top_n]],
        "top_scores": [float(x) for x in ranked["risk_adj_score_float"].iloc[:top_n]] if len(ranked) else [],
        "n_eligible": int(len(ranked)),
    }


def decide(key: str, pricing_df, universe_df, vxn_df, decision_ts, top_n: int = 20) -> dict:
    strategy, signal_df = signals(key, pricing_df, universe_df, vxn_df, decision_ts)
    return decision_at(strategy, signal_df, decision_ts, top_n)


def same_decision(a: dict, b: dict, weight_tol: float = 1e-12) -> tuple[bool, dict]:
    detail = {}
    ok = True
    if a["selected"] != b["selected"]:
        ok = False
        detail["selected_only_ref"] = sorted(set(a["selected"]) - set(b["selected"]))
        detail["selected_only_cand"] = sorted(set(b["selected"]) - set(a["selected"]))
    if a["top_rank"] != b["top_rank"]:
        ok = False
        detail["rank_ref"] = a["top_rank"][:12]
        detail["rank_cand"] = b["top_rank"][:12]
    wdiff = max([abs(a["weights"].get(s, 0.0) - b["weights"].get(s, 0.0)) for s in set(a["weights"]) | set(b["weights"])] or [0.0])
    if wdiff > weight_tol:
        ok = False
    detail["max_weight_diff"] = wdiff
    detail["exposure_ref"] = a["exposure"]
    detail["exposure_cand"] = b["exposure"]
    return ok, detail


def month_end_decision_dates(pricing_df: pd.DataFrame, key: str = "natr_vxn") -> pd.DatetimeIndex:
    close = pricing_df[("SPY", "Close")].to_frame("SPY")
    return pd.DatetimeIndex(mod(key).get_monthly_decision_close_df(close).index)


# ─── engine runs ──────────────────────────────────────────────────────────────


def run_backtest(key: str, data: dict, start: str = "2000-01-01"):
    from alpha.engine.backtest import run_daily
    strategy = make_strategy(key, data["schedule"], data["vxn"], data["universe"])
    pricing_df = data["pricing"]
    calendar_idx = pricing_df.index[pricing_df.index >= pd.Timestamp(start)]
    t0 = time.time()
    run_daily(strategy, pricing_df, calendar=calendar_idx, show_progress=False, show_signal_progress_bool=False,
              audit_override_bool=None)
    print(f"backtest {key} done in {time.time() - t0:.0f}s")
    return strategy


def nav_returns(nav: pd.Series, invested: pd.Series) -> pd.Series:
    """Same rule as refresh_books.nav_to_returns."""
    first = invested[invested].index[0]
    position = max(nav.index.get_loc(first) - 1, 0)
    return nav.iloc[position:].pct_change().iloc[1:]


def window_metrics(ret: pd.Series, start=None, end=STUDY_END) -> dict:
    """CAGR (calendar, base = prior session) / Sharpe rf=0 / vol / MaxDD, as fund_menu common.metric_dict."""
    full_index = ret.index
    ser = ret.loc[start:end].dropna() if start is not None else ret.loc[:end].dropna()
    pos = full_index.get_loc(ser.index[0])
    base = full_index[pos - 1] if pos > 0 else ser.index[0] - pd.Timedelta(days=1)
    nav = pd.concat([pd.Series([1.0], index=[base]), (1.0 + ser).cumprod()])
    days = (nav.index[-1] - nav.index[0]).days
    return {
        "start": ser.index[0].date().isoformat(), "end": ser.index[-1].date().isoformat(),
        "cagr": float(nav.iloc[-1] ** (365.25 / days) - 1.0),
        "vol": float(ser.std() * np.sqrt(252)),
        "sharpe": float(ser.mean() / ser.std() * np.sqrt(252)),
        "maxdd": float((nav / nav.cummax() - 1.0).min()),
    }


def strategy_return_ser(strategy) -> pd.Series:
    res = strategy.results
    return nav_returns(res["total_value"].astype(float), res["portfolio_value"].astype(float).abs() > 1e-9)


def save_run(strategy, tag: str) -> Path:
    path = CACHE / f"run_{tag}.pkl"
    payload = {"results": strategy.results.copy(), "transactions": strategy.get_transactions().copy()}
    try:
        payload["dividends"] = strategy.get_dividend_ledger().copy()
    except Exception:  # pragma: no cover - diagnostic only
        pass
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_run(tag: str) -> dict:
    with (CACHE / f"run_{tag}.pkl").open("rb") as handle:
        return pickle.load(handle)
