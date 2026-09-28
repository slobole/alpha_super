"""Missing positive controls for the TAA harness families (protocol A4: at least one per harness family).

PC-R  Live-parity replay (primary: taa_live_parity_replay.py) with a planted ONE-session look-ahead
      (signal close read at T+1). The replay truncates prices at T on the live side only, so it must see diffs.
      Run for taa3x (standard family) and btal_qqq (linearity family) on decisions since 2023-01.
PC-S  Loader-boundary split harness (primary: taa_bc_checks.check_invariance) with a planted LEVEL-sensitive
      score (standard: price difference instead of return; linearity: slope of raw price instead of log price).
      Must change weights for a k:1 rescale.

Outputs go to the review folder; the primary replay's OUTPUT_DIR_PATH is redirected so its files are not touched.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT_PATH))
sys.path.insert(0, str(REPO_ROOT_PATH / "scripts/research/strategy_readiness_audit_20260928/taa"))

OUT_DIR_PATH = REPO_ROOT_PATH / "results/research/strategy_readiness_audit_20260928/review_quant"
REVIEW_DTB3 = OUT_DIR_PATH / "DTB3_review_cache.csv"

import taa_live_parity_replay as replay_module  # noqa: E402

replay_module.OUTPUT_DIR_PATH = OUT_DIR_PATH / "pc_replay"
replay_module.OUTPUT_DIR_PATH.mkdir(parents=True, exist_ok=True)
replay_module.DTB3_CACHE_PATH = REVIEW_DTB3
replay_module.FIRST_REBALANCE_STR = "2023-01-01"

base_module = import_module("strategies.taa_df.strategy_taa_df")
linearity_module = import_module("strategies.taa_df.strategy_taa_df_btal_linearity_1n")
utils_module = import_module("strategies.taa_df.strategy_taa_df_fallback_vix_cash_variant_utils")
linearity_core_module = import_module("strategies.taa_df.strategy_taa_df_btal_linearity")


def _offline(*args, **kwargs):
    from urllib.error import URLError

    raise URLError("review: offline, use the copied DTB3 cache")


import alpha.data.fred_loader as _fred_loader_module  # noqa: E402

_fred_loader_module.urlopen = _offline


def _patch_replay_env():
    replay_module._patch_environment()
    replay_module.real_urlopen = _offline  # the replay's cached urlopen then fails -> cache fallback, no network
    for variant_dict in replay_module.VARIANT_DICT.values():
        module_obj = import_module(variant_dict["module_str"])
        module_obj.DEFAULT_CONFIG = replace(module_obj.DEFAULT_CONFIG, dtb3_csv_path_str=str(REVIEW_DTB3))


def pc_replay() -> dict:
    _patch_replay_env()
    out = {}
    real_std = base_module.compute_month_end_weight_df
    real_lin = linearity_module.compute_daily_linearity_score_df

    def leaky_std(signal_close_df, cash_return_ser, config):
        return real_std(signal_close_df.shift(-1), cash_return_ser, config)  # *** CRITICAL*** planted leak

    def leaky_lin(signal_close_df, lookback_day_vec):
        return real_lin(signal_close_df=signal_close_df.shift(-1), lookback_day_vec=lookback_day_vec)  # planted leak

    for variant_key_str in ("taa3x", "btal_qqq"):
        clean_summary = replay_module.replay_variant(variant_key_str)
        base_module.compute_month_end_weight_df = leaky_std
        linearity_module.compute_daily_linearity_score_df = leaky_lin
        try:
            leaky_summary = replay_module.replay_variant(variant_key_str)
        finally:
            base_module.compute_month_end_weight_df = real_std
            linearity_module.compute_daily_linearity_score_df = real_lin
        out[variant_key_str] = {"clean": clean_summary, "planted_one_session_leak": leaky_summary,
                                "caught": leaky_summary["exact_match_1e-9"] < leaky_summary["decisions_ok"]}
        print(variant_key_str, json.dumps(out[variant_key_str]), flush=True)
    return out


def _split_case(variant_key_str: str, symbol_str: str, k_float: float) -> float:
    bc = import_module("taa_bc_checks")
    bc.DTB3_CACHE_STR = str(REVIEW_DTB3)
    config_obj = bc._config(variant_key_str)
    ref_df = bc._month_end_weights(variant_key_str, config_obj)
    real_loader = base_module.load_price_timeseries

    def scaled_loader(sym, *args, **kwargs):
        price_df = real_loader(sym, *args, **kwargs).copy()
        if sym == symbol_str:
            for f in ("Open", "High", "Low", "Close", "Dividend"):
                if f in price_df.columns:
                    price_df[f] = price_df[f] / k_float
        return price_df

    base_module.load_price_timeseries = scaled_loader
    try:
        new_df = bc._month_end_weights(variant_key_str, config_obj)
    finally:
        base_module.load_price_timeseries = real_loader
    common = ref_df.index.intersection(new_df.index)
    return float((new_df.loc[common] - ref_df.loc[common]).abs().max().max())


def pc_split() -> dict:
    out = {}
    real_std = base_module.compute_month_end_weight_df

    def level_std(signal_close_df, cash_return_ser, config):
        # planted level sensitivity: price DIFFERENCE / 100 instead of a return
        return real_std(signal_close_df.diff().cumsum().fillna(0.0) / 100.0 + 1.0, cash_return_ser, config)

    real_lin = linearity_module.compute_daily_linearity_score_df

    def level_lin(signal_close_df, lookback_day_vec):
        # planted level sensitivity: regress exp(price) instead of price -> log gives raw price level
        return real_lin(signal_close_df=np.exp(signal_close_df / 10.0), lookback_day_vec=lookback_day_vec)

    for variant_key_str, patch_pair in (("taa3x", ("std", level_std)), ("btal_qqq", ("lin", level_lin))):
        row_list = []
        for symbol_str in ("GLD", "TLT"):
            clean_diff = _split_case(variant_key_str, symbol_str, 40.0)
            if patch_pair[0] == "std":
                base_module.compute_month_end_weight_df = patch_pair[1]
            else:
                linearity_module.compute_daily_linearity_score_df = patch_pair[1]
            try:
                planted_diff = _split_case(variant_key_str, symbol_str, 40.0)
            finally:
                base_module.compute_month_end_weight_df = real_std
                linearity_module.compute_daily_linearity_score_df = real_lin
            row_list.append({"symbol": symbol_str, "k": 40.0, "clean_max_weight_diff": clean_diff,
                             "planted_max_weight_diff": planted_diff, "caught": planted_diff > 1e-12})
        out[variant_key_str] = row_list
        print(variant_key_str, json.dumps(row_list), flush=True)
    return out


def main() -> None:
    result = {"PC_S_split_harness": pc_split(), "PC_R_replay_harness": pc_replay()}
    (OUT_DIR_PATH / "rq_taa_positive_controls.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
