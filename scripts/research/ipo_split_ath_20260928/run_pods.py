"""Stage B - run the grid and the labelled variants of the rebuilt rule (PREREG section 5).

Writes (results dir): pod_returns.parquet (engine returns, no sweep), pod_cash_weight.parquet, pod_held.parquet,
pod_trades.parquet, pod_costs.json. Keys: "<POP>|N<n>|P<p>L<l>|<label>".

Usage: python run_pods.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE_PATH = Path(__file__).resolve().parent
if str(HERE_PATH) not in sys.path:
    sys.path.insert(0, str(HERE_PATH))

import common  # noqa: E402
import events  # noqa: E402
import simulate  # noqa: E402

START_TS = pd.Timestamp("1993-01-04")
END_TS = pd.Timestamp("2026-08-19")
SLOT_LIST = [10, 20, 40]
PAIR_LIST = [(0.07, 0.04), (0.15, 0.10), (0.15, 0.15), (0.20, 0.10), (0.20, 0.15)]
ANCHOR_TUPLE = (20, 0.20, 0.10)
ENGINE_SLIPPAGE_FLOAT = 0.00025
STRESS_SLIPPAGE_FLOAT = 0.00075
WIDE_SLIPPAGE_FLOAT = 0.00225
GRID_POPULATION_LIST = ["IPO_ATH", "SPLIT_ATH"]


def key_str(pop_str: str, n_int: int, p_float: float, l_float: float, label_str: str) -> str:
    return f"{pop_str}|N{n_int}|P{round(p_float * 100)}L{round(l_float * 100)}|{label_str}"


def run_spec_list() -> list[dict]:
    spec_list = []
    for pop_str in GRID_POPULATION_LIST:
        for n_int in SLOT_LIST:
            for p_float, l_float in PAIR_LIST:
                spec_list.append({"pop": pop_str, "top_k": 1000, "label": "base",
                                  "config": simulate.SimConfig(n_int, p_float, l_float, slippage_float=ENGINE_SLIPPAGE_FLOAT)})
        n_int, p_float, l_float = ANCHOR_TUPLE
        base_config = simulate.SimConfig(n_int, p_float, l_float, slippage_float=ENGINE_SLIPPAGE_FLOAT)
        label_dict = {
            "stress": ({}, {"slippage_float": STRESS_SLIPPAGE_FLOAT}),
            "u500": ({"top_k": 500}, {}),
            "e2": ({}, {"exit_mode_str": "E2"}),
            "e2_stress": ({}, {"exit_mode_str": "E2", "slippage_float": STRESS_SLIPPAGE_FLOAT}),
            "close_entry": ({}, {"entry_mode_str": "close"}),
            "wide20": ({}, {"slippage_float": WIDE_SLIPPAGE_FLOAT}),
            "haircut": ({}, {"terminal_haircut_float": 0.75}),
            "cap30k": ({}, {"capital_float": 30_000.0}),
            "cap10k": ({}, {"capital_float": 10_000.0}),
        }
        for label_str, (spec_override_dict, config_override_dict) in label_dict.items():
            config_dict = {**base_config.__dict__, **config_override_dict}
            spec_list.append({"pop": pop_str, "top_k": spec_override_dict.get("top_k", 1000), "label": label_str,
                              "config": simulate.SimConfig(**config_dict)})
    n_int, p_float, l_float = ANCHOR_TUPLE
    spec_list.append({"pop": "IPO_ATH_SP", "top_k": 1000, "label": "spac",
                      "config": simulate.SimConfig(n_int, p_float, l_float, slippage_float=ENGINE_SLIPPAGE_FLOAT)})
    return spec_list


def main() -> None:
    started_float = time.time()
    rows_df = events.load_rows()
    calendar_idx = events.load_calendar_idx()
    start_pos_int = int(calendar_idx.searchsorted(START_TS))
    end_pos_int = int(calendar_idx.searchsorted(END_TS, side="right")) - 1
    spec_list = run_spec_list()
    candidate_cache_dict = {}
    symbol_set = set()
    for spec_dict in spec_list:
        cache_key = (spec_dict["pop"], spec_dict["top_k"])
        if cache_key not in candidate_cache_dict:
            mask_arr = events.population_mask_dict(rows_df, spec_dict["top_k"])[spec_dict["pop"]]
            candidate_cache_dict[cache_key] = events.candidates_by_pos(rows_df, mask_arr)
            symbol_set |= {s for lst in candidate_cache_dict[cache_key].values() for s, _ in lst}
    store = simulate.BarStore(events.load_bar_store_dict(symbol_set))
    missing_list = sorted(symbol_set - set(store.array_dict))
    if missing_list:
        raise RuntimeError(f"bars missing for {len(missing_list)} candidate symbols, e.g. {missing_list[:5]}")
    common.log_progress(f"run_pods: {len(spec_list)} runs, {len(symbol_set)} candidate symbols")

    return_dict, cash_dict, held_dict, trade_list, cost_dict = {}, {}, {}, [], {}
    for spec_dict in spec_list:
        config = spec_dict["config"]
        run_key = key_str(spec_dict["pop"], config.slot_count_int, config.profit_target_float, config.trailing_stop_float, spec_dict["label"])
        out_dict = simulate.run_pod(candidate_cache_dict[(spec_dict["pop"], spec_dict["top_k"])], store, calendar_idx,
                                    start_pos_int, end_pos_int, config)
        return_dict[run_key] = out_dict["return_ser"]
        cash_dict[run_key] = out_dict["cash_weight_ser"]
        held_dict[run_key] = out_dict["held_count_ser"]
        trade_df = out_dict["trade_df"].copy()
        trade_df["key"] = run_key
        trade_list.append(trade_df)
        cost_dict[run_key] = {"commission_total": out_dict["commission_total_float"], "slippage_total": out_dict["slippage_total_float"],
                              "capital": config.capital_float}
    pd.DataFrame(return_dict).to_parquet(common.RESULTS_DIR_PATH / "pod_returns.parquet")
    pd.DataFrame(cash_dict).to_parquet(common.RESULTS_DIR_PATH / "pod_cash_weight.parquet")
    pd.DataFrame(held_dict).to_parquet(common.RESULTS_DIR_PATH / "pod_held.parquet")
    trades_df = pd.concat(trade_list, ignore_index=True)
    trades_df["entry_date"] = calendar_idx[trades_df["entry_pos"].to_numpy()]
    trades_df["exit_date"] = calendar_idx[np.minimum(trades_df["exit_pos"].to_numpy(), len(calendar_idx) - 1)]
    trades_df.to_parquet(common.RESULTS_DIR_PATH / "pod_trades.parquet")
    (common.RESULTS_DIR_PATH / "pod_costs.json").write_text(json.dumps(cost_dict, indent=1))
    common.log_progress(f"run_pods: done in {time.time() - started_float:.0f}s")


if __name__ == "__main__":
    main()
