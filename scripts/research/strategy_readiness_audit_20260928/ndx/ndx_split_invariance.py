"""NDX ATR / ATR-VXN future-split invariance on real data (protocol A2) with a positive control (A4).

One symbol's full loaded history is rescaled as if a k:1 split happened after the last date:
Open/High/Low/Close divided by k, Volume multiplied by k, Unadjusted Close kept nominal.
Selections and weights at the last 60 month-end decisions must not change.

Positive control: the pre-fb81e86 score (ATR in adjusted units, no nominal rebase) is computed
the same way and must change for at least one case.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT_PATH))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import strategies.momentum.strategy_mo_atr_normalized_ndx as atr_module  # noqa: E402
import strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled as vxn_module  # noqa: E402
from ndx_live_parity_replay import _cached_build_universe  # noqa: E402

OUTPUT_DIR_PATH = REPO_ROOT_PATH / "results/research/strategy_readiness_audit_20260928/ndx"
END_DATE_STR = "2026-09-25"
N_DECISIONS_INT = 60


def _selections(pricing_data_df, universe_df, vxn_df, decision_index, leaky_bool=False) -> dict:
    config_obj = replace(vxn_module.DEFAULT_CONFIG, end_date_str=END_DATE_STR)
    strategy_obj = vxn_module.VxnScaledAtrNormalizedNdxStrategy(
        name="audit",
        benchmarks=["$SPX"],
        rebalance_schedule_df=pd.DataFrame({"decision_date_ts": decision_index}, index=decision_index),
        vxn_scale_signal_df=vxn_df,
    )
    strategy_obj.universe_df = universe_df
    if leaky_bool:
        real_fn = atr_module.get_unadjusted_close_df

        def adjusted_as_unadjusted(pricing_df, symbol_list):
            # Pre-fix behaviour: k_T == 1, ATR stays in (future-)adjusted units.
            return pd.DataFrame({s: pricing_df[(s, "Close")] for s in symbol_list}, index=pricing_df.index).astype(float)

        atr_module.get_unadjusted_close_df = adjusted_as_unadjusted
        vxn_module.get_unadjusted_close_df = adjusted_as_unadjusted
    try:
        signal_df = strategy_obj.compute_signals(pricing_data_df.copy())
    finally:
        if leaky_bool:
            atr_module.get_unadjusted_close_df = real_fn
            vxn_module.get_unadjusted_close_df = real_fn
    out = {}
    for decision_ts in decision_index:
        strategy_obj.previous_bar = decision_ts
        weight_ser = strategy_obj.get_target_weight_ser(close_row_ser=signal_df.loc[decision_ts])
        out[decision_ts] = {str(k): round(float(v), 12) for k, v in weight_ser.items()}
    return out


def _rescale(pricing_data_df: pd.DataFrame, symbol_str: str, k_float: float) -> pd.DataFrame:
    scaled_df = pricing_data_df.copy()
    for field_str in ("Open", "High", "Low", "Close", "Dividend"):
        if (symbol_str, field_str) in scaled_df.columns:
            scaled_df[(symbol_str, field_str)] = scaled_df[(symbol_str, field_str)] / k_float
    if (symbol_str, "Volume") in scaled_df.columns:
        scaled_df[(symbol_str, "Volume")] = scaled_df[(symbol_str, "Volume")] * k_float
    return scaled_df


def main() -> None:
    atr_module.build_index_constituent_matrix = _cached_build_universe
    config_obj = replace(vxn_module.DEFAULT_CONFIG, end_date_str=END_DATE_STR)
    pricing_data_df, universe_df, rebalance_schedule_df, vxn_df = vxn_module.get_vxn_scaled_atr_normalized_ndx_data(config_obj)
    decision_index = pd.DatetimeIndex(rebalance_schedule_df["decision_date_ts"].iloc[-N_DECISIONS_INT:])
    base_dict = _selections(pricing_data_df, universe_df, vxn_df, decision_index)
    base_leaky_dict = _selections(pricing_data_df, universe_df, vxn_df, decision_index, leaky_bool=True)

    held_symbol_list = sorted({s for w in base_dict.values() for s in w})
    symbol_list = ["MU", "NVDA", "AAPL", "CSCO", "BKNG", "WBD", "TSLA", "AVGO"]
    symbol_list = [s for s in dict.fromkeys(symbol_list + held_symbol_list[:4]) if (s, "Close") in pricing_data_df.columns]

    case_list = []
    for symbol_str in symbol_list:
        for k_float in (40.0, 0.1, 1.5):
            scaled_df = _rescale(pricing_data_df, symbol_str, k_float)
            scaled_dict = _selections(scaled_df, universe_df, vxn_df, decision_index)
            diff_count_int = sum(scaled_dict[d] != base_dict[d] for d in decision_index)
            leaky_scaled_dict = _selections(scaled_df, universe_df, vxn_df, decision_index, leaky_bool=True)
            leaky_diff_int = sum(leaky_scaled_dict[d] != base_leaky_dict[d] for d in decision_index)
            case_list.append({"symbol": symbol_str, "k": k_float, "decisions": len(decision_index),
                              "changed_decisions": int(diff_count_int), "control_changed_decisions": int(leaky_diff_int)})
            print(json.dumps(case_list[-1]), flush=True)
    summary_dict = {
        "cases": len(case_list),
        "passed": sum(c["changed_decisions"] == 0 for c in case_list),
        "positive_control_cases_detected": sum(c["control_changed_decisions"] > 0 for c in case_list),
        "detail": case_list,
    }
    (OUTPUT_DIR_PATH / "split_invariance_vxn.json").write_text(json.dumps(summary_dict, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary_dict.items() if k != "detail"}), flush=True)


if __name__ == "__main__":
    main()
