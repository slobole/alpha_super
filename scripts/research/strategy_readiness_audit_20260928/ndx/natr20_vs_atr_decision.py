"""NATR20 vs live ATR-VXN rule: the owner's decision study (2026-09-28).

The two rules differ in one line only:
    live ATR rule : score = ROC12 / ATR20$              (ATR in nominal dollars at T)
    NATR20        : score = ROC12 / (ATR20$ / Price_T)  (ATR in percent of price)
Everything else (PIT NDX members, Close > SMA100, SPY > SMA200 regime, top 10, VXN scale
clip(22/VXN, 0.25, 1), next-open MOO) is identical.

Phases (select with argv[1]):
  bc       NATR20 split invariance (with the pre-fb81e86 control) + row-T truncation at 8 cut-offs
           (truncated loader vs full history) + a planted one-bar leak control.
  compare  both rules, same data, production (trimmed) and live-semantics (untrimmed) universes;
           metrics over full / 2016+ / 2019+ / last 3y; turnover; price of picks; takeover picks.
  small    both rules at a USD 12K pod starting 2016-01-04 and 2021-01-04, IBKR Fixed
           (USD 0.005/share, min USD 1) vs an IBKR Tiered approximation (USD 0.004/share all-in,
           min USD 0.35). Raw whole shares (both classes already use historical share units).

Study code only; no repository file is changed. Runtime patches: the universe builder is cached
(trimmed, as production) or replaced by an untrimmed builder (live semantics).
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT_PATH))

import data.norgate_loader as norgate_loader_module  # noqa: E402
import strategies.momentum.strategy_mo_atr_normalized_ndx as atr_module  # noqa: E402
import strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled as vxn_module  # noqa: E402
import strategies.momentum.strategy_mo_natr20_ndx_vxn_scaled as natr_module  # noqa: E402
from alpha.engine.backtest import run_daily  # noqa: E402

OUTPUT_DIR_PATH = REPO_ROOT_PATH / "results/research/strategy_readiness_audit_20260928/ndx/natr20_decision"
OUTPUT_DIR_PATH.mkdir(parents=True, exist_ok=True)
END_DATE_STR = "2026-09-25"
LAST3Y_START_STR = "2023-09-25"

_REAL_BUILDER = norgate_loader_module.build_index_constituent_matrix
_CACHE: dict[tuple[str, bool], tuple] = {}


def _untrimmed_builder(indexname: str = "S&P 500"):
    """build_index_constituent_matrix without the 5-session tail trim (live membership semantics)."""
    import norgatedata

    symbol_list = norgatedata.watchlist_symbols(f"{indexname} Current & Past")
    frame_list = []
    for symbol_str in symbol_list:
        idx_df = norgatedata.index_constituent_timeseries(symbol_str, indexname, timeseriesformat="pandas-dataframe")
        if idx_df["Index Constituent"].sum() > 0:
            idx_df = idx_df.rename(columns={"Index Constituent": symbol_str})
            frame_list.append(idx_df.loc[idx_df[symbol_str] == 1])
    return symbol_list, pd.concat(frame_list, axis=1).fillna(0).astype(int).sort_index()


def _install_universe(untrimmed_bool: bool) -> None:
    def cached(indexname: str = "S&P 500"):
        key = (indexname, untrimmed_bool)
        if key not in _CACHE:
            _CACHE[key] = (_untrimmed_builder if untrimmed_bool else _REAL_BUILDER)(indexname=indexname)
        symbol_list, universe_df = _CACHE[key]
        return list(symbol_list), universe_df.copy()

    atr_module.build_index_constituent_matrix = cached
    natr_module.build_index_constituent_matrix = cached


def _load(untrimmed_bool: bool, end_date_str: str = END_DATE_STR):
    _install_universe(untrimmed_bool)
    config_obj = replace(vxn_module.DEFAULT_CONFIG, end_date_str=end_date_str)
    return vxn_module.get_vxn_scaled_atr_normalized_ndx_data(config_obj, include_total_return_benchmark_bool=True)


def _build(rule_str: str, data_tuple, capital_float=100_000.0, per_share_float=0.005, minimum_float=1.0):
    pricing_df, universe_df, schedule_df, vxn_df = data_tuple
    common = dict(
        name=f"{rule_str}", benchmarks=["$SPX"], rebalance_schedule_df=schedule_df, vxn_scale_signal_df=vxn_df,
        capital_base=capital_float, commission_per_share=per_share_float, commission_minimum=minimum_float,
    )
    if rule_str == "atr_live":
        strategy_obj = vxn_module.VxnScaledAtrNormalizedNdxStrategy(**common)
        vxn_module.configure_total_return_benchmark_provenance(strategy_obj, vxn_module.DEFAULT_CONFIG)
    else:
        strategy_obj = natr_module.Natr20VxnScaledNdxStrategy(**common)
        natr_module.configure_total_return_benchmark_provenance(strategy_obj, natr_module.DEFAULT_CONFIG)
    strategy_obj.universe_df = universe_df
    return strategy_obj


def _run(strategy_obj, pricing_df, start_str: str):
    calendar_idx = pricing_df.index[pricing_df.index >= pd.Timestamp(start_str)]
    run_daily(strategy_obj, pricing_df, calendar=calendar_idx, show_progress=False,
              show_signal_progress_bool=False, audit_override_bool=False)
    nav_ser = strategy_obj.results["total_value"].astype(float)
    nav_ser.index = pd.to_datetime(nav_ser.index)
    return nav_ser


def _metrics(nav_ser: pd.Series) -> dict:
    ret_ser = nav_ser.pct_change().dropna()
    years_float = len(ret_ser) / 252.0
    return {
        "cagr_pct": round(((nav_ser.iloc[-1] / nav_ser.iloc[0]) ** (1 / years_float) - 1) * 100, 2),
        "sharpe": round(float(ret_ser.mean() / ret_ser.std() * np.sqrt(252)), 3),
        "max_dd_pct": round(float((nav_ser / nav_ser.cummax() - 1).min()) * 100, 1),
        "vol_pct": round(float(ret_ser.std() * np.sqrt(252)) * 100, 1),
    }


def _windows(nav_ser: pd.Series) -> dict:
    return {
        "full_2000": _metrics(nav_ser),
        "from_2016": _metrics(nav_ser.loc["2016-01-04":]),
        "from_2019": _metrics(nav_ser.loc["2019-01-02":]),
        "last_3y": _metrics(nav_ser.loc[LAST3Y_START_STR:]),
    }


def _selection_stats(strategy_obj, pricing_df) -> dict:
    """Price of picks and takeover-target share from the fills of each rebalance."""
    tx_df = strategy_obj.get_transactions().copy()
    tx_df = tx_df[(tx_df["amount"] > 0) & (tx_df["order_id"] != -1)]
    tx_df["bar"] = pd.to_datetime(tx_df["bar"])
    price_list, delist_soon_list = [], []
    for _, row in tx_df.iterrows():
        asset_str, bar_ts = str(row["asset"]), pd.Timestamp(row["bar"])
        unadj_ser = pricing_df[(asset_str, "Unadjusted Close")].loc[:bar_ts].dropna()
        if len(unadj_ser):
            price_list.append(float(unadj_ser.iloc[-2] if len(unadj_ser) > 1 else unadj_ser.iloc[-1]))
        match_obj = re.search(r"-(\d{6})$", asset_str)
        if match_obj:
            delist_ts = pd.Timestamp(f"{match_obj.group(1)[:4]}-{match_obj.group(1)[4:]}-01")
            delist_soon_list.append(delist_ts <= bar_ts + pd.DateOffset(months=3))
        else:
            delist_soon_list.append(False)
    recent_mask = tx_df["bar"] >= pd.Timestamp("2016-01-01")
    years_float = max((tx_df["bar"].max() - tx_df["bar"].min()).days / 365.25, 1e-9)
    all_tx_df = strategy_obj.get_transactions()
    return {
        "buy_fills": int(len(tx_df)),
        "fills_per_year_all_sides": round(len(all_tx_df) / years_float, 1),
        "median_nominal_price_of_buys": round(float(np.median(price_list)), 1),
        "median_nominal_price_of_buys_2016plus": round(float(np.median(np.array(price_list)[recent_mask.to_numpy()])), 1),
        "share_buys_delisted_within_3m_pct": round(100 * float(np.mean(delist_soon_list)), 2),
        "share_buys_delisted_within_3m_2016plus_pct": round(100 * float(np.mean(np.array(delist_soon_list)[recent_mask.to_numpy()])), 2),
        "commission_total_usd": round(float(all_tx_df["commission"].sum()), 0),
    }


# ----------------------------------------------------------------------------- phases
def phase_compare() -> dict:
    out = {}
    for untrimmed_bool in (False, True):
        universe_label = "live_untrimmed" if untrimmed_bool else "production_trimmed"
        data_tuple = _load(untrimmed_bool)
        for rule_str in ("atr_live", "natr20"):
            start_float = time.time()
            strategy_obj = _build(rule_str, data_tuple)
            nav_ser = _run(strategy_obj, data_tuple[0], "2000-01-01")
            out[f"{rule_str}|{universe_label}"] = {"windows": _windows(nav_ser), "selection": _selection_stats(strategy_obj, data_tuple[0])}
            nav_ser.to_csv(OUTPUT_DIR_PATH / f"nav_{rule_str}_{universe_label}.csv")
            print(rule_str, universe_label, json.dumps(out[f"{rule_str}|{universe_label}"]), f"{time.time()-start_float:.0f}s", flush=True)
    (OUTPUT_DIR_PATH / "compare.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def phase_small() -> dict:
    out = {}
    data_tuple = _load(untrimmed_bool=True)  # live semantics
    fee_plan_dict = {"ibkr_fixed": (0.005, 1.0), "ibkr_tiered_approx": (0.004, 0.35), "no_min_reference": (0.005, 0.0)}
    for start_str in ("2016-01-04", "2021-01-04"):
        for rule_str in ("atr_live", "natr20"):
            for fee_str, (per_share_float, minimum_float) in fee_plan_dict.items():
                for capital_float in (12_000.0, 1_000_000.0):
                    if capital_float > 20_000 and fee_str != "ibkr_fixed":
                        continue
                    strategy_obj = _build(rule_str, data_tuple, capital_float, per_share_float, minimum_float)
                    nav_ser = _run(strategy_obj, data_tuple[0], start_str)
                    key_str = f"{rule_str}|start {start_str}|{int(capital_float)}|{fee_str}"
                    tx_df = strategy_obj.get_transactions()
                    out[key_str] = {**_metrics(nav_ser), "commission_usd": round(float(tx_df["commission"].sum()), 0),
                                    "final_nav": round(float(nav_ser.iloc[-1]), 0)}
                    print(key_str, json.dumps(out[key_str]), flush=True)
    (OUTPUT_DIR_PATH / "small_account.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def phase_bc() -> dict:
    out = {"split_invariance": [], "truncation": [], "planted_leak_truncation": []}
    data_tuple = _load(untrimmed_bool=False)
    pricing_df, universe_df, schedule_df, vxn_df = data_tuple
    decision_list = [pd.Timestamp(d) for d in schedule_df["decision_date_ts"].iloc[-60:]]

    def weights(pdf, leaky_bool=False):
        strategy_obj = _build("natr20", (pdf, universe_df, schedule_df, vxn_df))
        signal_df = strategy_obj.compute_signals(pdf.copy())
        if leaky_bool:
            # Planted one-bar leak: the decision row reads the NEXT session's features.
            signal_df = signal_df.shift(-1)
        res = {}
        for d in decision_list:
            if d not in signal_df.index:
                continue
            strategy_obj.previous_bar = d
            res[d] = {k: round(float(v), 12) for k, v in strategy_obj.get_target_weight_ser(signal_df.loc[d]).items()}
        return res

    base = weights(pricing_df)
    for symbol_str in ("MU", "NVDA", "AAPL", "BKNG", "WBD", "CSCO"):
        if (symbol_str, "Close") not in pricing_df.columns:
            continue
        for k_float in (40.0, 0.1, 1.5):
            scaled_df = pricing_df.copy()
            for f in ("Open", "High", "Low", "Close", "Dividend"):
                if (symbol_str, f) in scaled_df.columns:
                    scaled_df[(symbol_str, f)] = scaled_df[(symbol_str, f)] / k_float
            scaled_df[(symbol_str, "Volume")] = scaled_df[(symbol_str, "Volume")] * k_float
            scaled = weights(scaled_df)
            out["split_invariance"].append({"symbol": symbol_str, "k": k_float, "changed_decisions": sum(scaled[d] != base[d] for d in base)})
    print("split", json.dumps(out["split_invariance"]), flush=True)

    leaky_full = weights(pricing_df, leaky_bool=True)
    full_strategy_obj = _build("natr20", data_tuple)
    full_signal_df = full_strategy_obj.compute_signals(pricing_df.copy())
    leaky_signal_df = full_signal_df.shift(-1)
    for cutoff_str in ("2008-10-31", "2013-03-28", "2018-12-31", "2020-03-31", "2022-06-30", "2024-05-31", "2026-07-31", "2026-08-31"):
        truncated_tuple = _load(untrimmed_bool=False, end_date_str=cutoff_str)
        t_pricing, t_universe, t_schedule, t_vxn = truncated_tuple
        cutoff_ts = pd.Timestamp(cutoff_str)
        strategy_obj = _build("natr20", truncated_tuple)
        signal_df = strategy_obj.compute_signals(t_pricing.copy())
        strategy_obj.previous_bar = cutoff_ts
        trunc_w = {k: round(float(v), 12) for k, v in strategy_obj.get_target_weight_ser(signal_df.loc[cutoff_ts]).items()}
        full_strategy_obj.previous_bar = cutoff_ts
        full_w = {k: round(float(v), 12) for k, v in full_strategy_obj.get_target_weight_ser(full_signal_df.loc[cutoff_ts]).items()}
        leaky_w = {k: round(float(v), 12) for k, v in full_strategy_obj.get_target_weight_ser(leaky_signal_df.loc[cutoff_ts]).items()}
        out["truncation"].append({"cutoff": cutoff_str, "match": trunc_w == full_w, "n": len(trunc_w)})
        # Planted leak: the leaky full-history decision at the cut-off differs from the honest truncated one?
        out["planted_leak_truncation"].append({"cutoff": cutoff_str, "caught": leaky_w != trunc_w})
        print("trunc", cutoff_str, out["truncation"][-1], flush=True)
    (OUTPUT_DIR_PATH / "natr20_bc.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


if __name__ == "__main__":
    {"compare": phase_compare, "small": phase_small, "bc": phase_bc}[sys.argv[1]]()
