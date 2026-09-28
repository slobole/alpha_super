"""Tradability study (C1/C2) for the monthly wired pods: TAA 3x, TAA 1/N, BTAL_QQQ, NDX ATR VXN, NDX ATR.

Runs each module's own run_variant (end 2026-09-25, legacy default settings, as published),
then measures order participation against native-Turnover ADV20 and whole-share weight error.

Usage: uv run python .../common/run_monthly_tradability.py <key> [<key> ...]
keys: taa3x taa1n btal_qqq ndx_vxn ndx_atr
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT_PATH))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tradability import (  # noqa: E402
    load_turnover_ser,
    participation_table_df,
    summarize_participation_df,
    whole_share_table_df,
)
from data.norgate_loader import load_price_timeseries  # noqa: E402

OUTPUT_ROOT_PATH = REPO_ROOT_PATH / "results/research/strategy_readiness_audit_20260928/tradability"
OUTPUT_ROOT_PATH.mkdir(parents=True, exist_ok=True)
END_DATE_STR = "2026-09-25"

MODULE_DICT = {
    "taa3x": "strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash",
    "taa1n": "strategies.taa_df.strategy_taa_df_btal_1n_fallback_tqqq_vix_cash",
    "btal_qqq": "strategies.taa_df.strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash",
    "ndx_vxn": "strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled",
    "ndx_atr": "strategies.momentum.strategy_mo_atr_normalized_ndx",
}

_UNADJ_CACHE: dict[str, pd.Series] = {}


def _unadjusted_close_on_or_before(asset_str: str, date_str: str) -> float:
    if asset_str not in _UNADJ_CACHE:
        price_df = load_price_timeseries(asset_str, start_date_str="1998-01-01", end_date_str=END_DATE_STR)
        _UNADJ_CACHE[asset_str] = price_df["Unadjusted Close"].astype(float).dropna()
    value_ser = _UNADJ_CACHE[asset_str].loc[: pd.Timestamp(date_str)]
    return float(value_ser.iloc[-1]) if len(value_ser) else np.nan


def _target_weights_by_decision(key_str: str, strategy_obj) -> dict[str, dict[str, float]]:
    """Decision-date target weights, taken from the rebalance table the backtest actually traded."""
    weight_by_decision_dict: dict[str, dict[str, float]] = {}
    if key_str.startswith("ndx"):
        # Held weights after each rebalance are not needed; use the schedule plus full-history targets.
        full_signal_df = strategy_obj._audit_full_signal_df
        for _, row_ser in strategy_obj.rebalance_schedule_df.iterrows():
            decision_ts = pd.Timestamp(row_ser["decision_date_ts"])
            strategy_obj.previous_bar = decision_ts
            target_ser = strategy_obj.get_target_weight_ser(close_row_ser=full_signal_df.loc[decision_ts])
            weight_by_decision_dict[decision_ts.date().isoformat()] = {str(k): float(v) for k, v in target_ser.items()}
        return weight_by_decision_dict
    rebalance_weight_df = strategy_obj.rebalance_weight_df
    for rebalance_ts, weight_ser in rebalance_weight_df.iterrows():
        decision_date_str = (pd.Timestamp(rebalance_ts) - pd.offsets.BDay(1)).date().isoformat()
        weight_by_decision_dict[decision_date_str] = {str(k): float(v) for k, v in weight_ser.items() if float(v) > 1e-12}
    return weight_by_decision_dict


def run_key(key_str: str) -> dict:
    module_obj = import_module(MODULE_DICT[key_str])
    if hasattr(module_obj.DEFAULT_CONFIG, "dtb3_csv_path_str"):
        # Study-only: keep the owner's shared DTB3 cache untouched.
        from dataclasses import replace as dataclass_replace

        module_obj.DEFAULT_CONFIG = dataclass_replace(
            module_obj.DEFAULT_CONFIG, dtb3_csv_path_str=str(OUTPUT_ROOT_PATH / "DTB3_audit_cache.csv")
        )
    strategy_obj = module_obj.run_variant(show_display_bool=False, save_results_bool=False, end_date_str=END_DATE_STR)
    if key_str.startswith("ndx"):
        # Rebuild the signal frame once for whole-share target reconstruction.
        config_obj = module_obj.DEFAULT_CONFIG.__class__(**{**module_obj.DEFAULT_CONFIG.__dict__, "end_date_str": END_DATE_STR})
        loader_fn = getattr(module_obj, "get_vxn_scaled_atr_normalized_ndx_data", None) or module_obj.get_atr_normalized_ndx_data
        pricing_data_df = loader_fn(config_obj)[0]
        strategy_obj._audit_full_signal_df = strategy_obj.compute_signals(pricing_data_df.copy())

    traded_asset_list = sorted(set(strategy_obj.get_transactions()["asset"].astype(str)))
    turnover_by_symbol_dict = {}
    for asset_str in traded_asset_list:
        try:
            turnover_by_symbol_dict[asset_str] = load_turnover_ser(asset_str, end_date_str=END_DATE_STR)
        except Exception as exception_obj:  # noqa: BLE001
            print(f"no turnover for {asset_str}: {exception_obj}", flush=True)

    participation_df = participation_table_df(strategy_obj, turnover_by_symbol_dict)
    output_dir_path = OUTPUT_ROOT_PATH / key_str
    output_dir_path.mkdir(parents=True, exist_ok=True)
    participation_df.to_csv(output_dir_path / "participation_by_fill.csv", index=False)
    summary_df = summarize_participation_df(participation_df)
    summary_df.to_csv(output_dir_path / "participation_summary.csv", index=False)

    per_asset_df = (
        participation_df.assign(part_1m_pct=participation_df["part_1000000"] * 100.0)
        .groupby("asset")["part_1m_pct"].agg(["count", "median", "max"])
        .sort_values("max", ascending=False)
    )
    per_asset_df.to_csv(output_dir_path / "participation_by_asset_1m.csv")

    weight_by_decision_dict = _target_weights_by_decision(key_str, strategy_obj)
    recent_weight_dict = {d: w for d, w in weight_by_decision_dict.items() if d >= "2016-01-01"}
    whole_share_df = whole_share_table_df(recent_weight_dict, _unadjusted_close_on_or_before)
    whole_share_df.to_csv(output_dir_path / "whole_share_since_2016.csv", index=False)
    whole_share_summary_df = (
        whole_share_df.groupby("capital_usd")
        .agg(
            max_name_weight_error=("weight_error", "max"),
            p95_name_weight_error=("weight_error", lambda s: float(s.quantile(0.95))),
            zero_share_names=("zero_share_bool", "sum"),
            name_decisions=("asset", "count"),
        )
        .reset_index()
    )
    per_decision_cash_df = whole_share_df.groupby(["capital_usd", "decision_date"])["weight_error"].sum().reset_index()
    whole_share_summary_df["mean_cash_drag_from_rounding"] = (
        per_decision_cash_df.groupby("capital_usd")["weight_error"].mean().values
    )
    whole_share_summary_df["max_cash_drag_from_rounding"] = (
        per_decision_cash_df.groupby("capital_usd")["weight_error"].max().values
    )
    whole_share_summary_df.to_csv(output_dir_path / "whole_share_summary.csv", index=False)

    summary_dict = {
        "key": key_str,
        "final_equity": float(strategy_obj.total_value),
        "participation": summary_df.to_dict(orient="records"),
        "top_assets_by_max_part_1m": per_asset_df.head(8).reset_index().to_dict(orient="records"),
        "whole_share": whole_share_summary_df.to_dict(orient="records"),
    }
    (output_dir_path / "summary.json").write_text(json.dumps(summary_dict, indent=2, default=str), encoding="utf-8")
    return summary_dict


if __name__ == "__main__":
    for key_str in sys.argv[1:]:
        result_dict = run_key(key_str)
        print(json.dumps(result_dict, default=str)[:4000], flush=True)
