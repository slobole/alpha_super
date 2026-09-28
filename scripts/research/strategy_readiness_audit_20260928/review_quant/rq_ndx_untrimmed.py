"""NDX ATR / ATR-VXN: re-verify the 5-session membership-trim look-ahead (cited <= 0.12 pp in the leakage hunt).

The primary live replay serves ONE cached universe (built today, trimmed with today's knowledge of removals) to both
the backtest and the live host, so it cannot see this difference. Here the backtest is re-run with an exact PIT
(untrimmed) universe, which is what the live host sees at each month-end (a name that will be removed next week is
still a member today).

Usage: uv run python .../review_quant/rq_ndx_untrimmed.py [vxn|atr ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT_PATH))

import data.norgate_loader as norgate_loader_module  # noqa: E402
import strategies.momentum.strategy_mo_atr_normalized_ndx as atr_module  # noqa: E402
import strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled as vxn_module  # noqa: E402

OUT_DIR_PATH = REPO_ROOT_PATH / "results/research/strategy_readiness_audit_20260928/review_quant"
END_DATE_STR = "2026-09-25"
_CACHE: dict = {}


def build_universe(indexname: str, trim_bool: bool):
    key = (indexname, trim_bool)
    if key in _CACHE:
        symbols, universe_df = _CACHE[key]
        return list(symbols), universe_df.copy()
    nd = norgate_loader_module._load_direct_norgate_module()
    symbols = nd.watchlist_symbols(f"{indexname} Current & Past")
    last_trading_day = nd.price_timeseries("$SPX", timeseriesformat="pandas-dataframe").index[-1]
    frame_list = []
    trimmed_count_int = 0
    for symbol in symbols:
        idx = nd.index_constituent_timeseries(symbol, indexname, timeseriesformat="pandas-dataframe")
        if idx["Index Constituent"].sum() > 0:
            idx = idx.rename(columns={"Index Constituent": symbol})
            idx = idx.loc[idx[symbol] == 1]
            if trim_bool and last_trading_day != idx.index[-1]:
                idx = idx.iloc[:-5]
                trimmed_count_int += 1
            frame_list.append(idx)
    universe_df = pd.concat(frame_list, axis=1).fillna(0).astype(int).sort_index()
    _CACHE[key] = (symbols, universe_df)
    print(f"universe {indexname} trim={trim_bool}: {universe_df.shape}, trimmed symbols={trimmed_count_int}", flush=True)
    return list(symbols), universe_df.copy()


def _metrics(total_value_ser: pd.Series) -> dict:
    total_value_ser = total_value_ser.astype(float)
    r = total_value_ser.pct_change().dropna()
    years = len(r) / 252.0
    return {"cagr": float((total_value_ser.iloc[-1] / total_value_ser.iloc[0]) ** (1 / years) - 1),
            "sharpe": float(r.mean() / r.std() * np.sqrt(252)),
            "max_dd": float((total_value_ser / total_value_ser.cummax() - 1).min())}


def main() -> None:
    variant_list = sys.argv[1:] or ["vxn", "atr"]
    result = {}
    for variant_str in variant_list:
        module_obj = vxn_module if variant_str == "vxn" else atr_module
        arm_dict = {}
        tx_dict = {}
        for trim_bool in (True, False):
            atr_module.build_index_constituent_matrix = (
                lambda indexname="S&P 500", _t=trim_bool: build_universe(indexname, _t)
            )
            strategy_obj = module_obj.run_variant(show_display_bool=False, save_results_bool=False, end_date_str=END_DATE_STR)
            arm_str = "trimmed_production" if trim_bool else "untrimmed_live_semantics"
            tv = strategy_obj.results["total_value"].astype(float)
            arm_dict[arm_str] = _metrics(tv)
            arm_dict[arm_str]["last3y"] = _metrics(tv[tv.index >= tv.index[-1] - pd.DateOffset(years=3)])
            tx = strategy_obj.get_transactions().copy()
            tx["bar"] = pd.to_datetime(tx["bar"]).dt.date.astype(str)
            tx_dict[arm_str] = tx
            print(variant_str, arm_str, arm_dict[arm_str], flush=True)
        a = tx_dict["trimmed_production"]
        b = tx_dict["untrimmed_live_semantics"]
        buys_a = set(map(tuple, a[a["amount"] > 0][["bar", "asset"]].astype(str).to_numpy()))
        buys_b = set(map(tuple, b[b["amount"] > 0][["bar", "asset"]].astype(str).to_numpy()))
        only_b = sorted(buys_b - buys_a)
        only_a = sorted(buys_a - buys_b)
        u, t = arm_dict["untrimmed_live_semantics"], arm_dict["trimmed_production"]
        result[variant_str] = {
            **arm_dict,
            "delta_untrimmed_minus_trimmed_cagr_pp": 100 * (u["cagr"] - t["cagr"]),
            "delta_sharpe": u["sharpe"] - t["sharpe"],
            "delta_maxdd_pp": 100 * (u["max_dd"] - t["max_dd"]),
            "delta_last3y_cagr_pp": 100 * (u["last3y"]["cagr"] - t["last3y"]["cagr"]),
            "buy_fills_only_untrimmed": len(only_b),
            "buy_fills_only_trimmed": len(only_a),
            "examples_only_untrimmed": only_b[:25],
            "examples_only_trimmed": only_a[:25],
        }
        print(json.dumps({k: v for k, v in result[variant_str].items() if not k.startswith("examples")}), flush=True)
        (OUT_DIR_PATH / "rq_ndx_untrimmed.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
