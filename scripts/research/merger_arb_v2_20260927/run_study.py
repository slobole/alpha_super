"""Runner for merger arbitrage v2 (research only). Frozen plan: docs/research/MERGER_ARB_V2_PREREG_20260927.md.

    python scripts/research/merger_arb_v2_20260927/run_study.py --pass1 --pass2
    python ... --parity
    python ... --grid --costs engine,stress      (14 grid cells + 4 sensitivities, ALL events)
    python ... --halves                          (14 cells x {R1000, R2000} events, engine costs)
    python ... --small-account
    python ... --invariance-v1 | --invariance-v2
    python ... --analyze
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from merger_arb_v2_20260927 import cells as cells_module  # noqa: E402
from merger_arb_v2_20260927 import common  # noqa: E402
from merger_arb_v2_20260927 import data as data_module  # noqa: E402
from merger_arb_v2_20260927.features import V2FeatureBook  # noqa: E402
from merger_arb_v2_20260927.policies import PolicyV  # noqa: E402
from merger_arb_v2_20260927.simulate import simulate  # noqa: E402
from trend_breakout_20260927.simulate import run_summary  # noqa: E402

COST_DICT = {"engine": common.ENGINE_SLIPPAGE_FLOAT, "stress": common.STRESS_SLIPPAGE_FLOAT}


def safe_name(key_str: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in key_str)


def load_feature_book() -> V2FeatureBook:
    return V2FeatureBook(data_module.load_panel())


def v_diagnostics(sim_dict: dict, policy_obj: PolicyV, feature_obj: V2FeatureBook, cell: cells_module.VCell) -> dict:
    date_index = feature_obj.date_index
    end_ts = common.END_TS
    trade_df = sim_dict["trade_df"]
    if len(trade_df):
        trade_df = trade_df[date_index[trade_df["exit_pos"].to_numpy()] <= end_ts]
    years_float = float((end_ts - common.TRADING_START_TS).days / 365.25)
    start_pos_int, end_pos_int = sim_dict["start_pos_int"], int(date_index.searchsorted(end_ts, side="right"))
    event_panel = feature_obj.event(cell.jump_float, cell.half_str)
    conf_panel = policy_obj.conf_arr
    concurrency_vec = np.zeros(len(date_index) + 400, dtype=int)
    for row in trade_df.itertuples(index=False):
        concurrency_vec[row.entry_pos : row.exit_pos] += 1
    concurrency_vec = concurrency_vec[start_pos_int:end_pos_int]
    by_reason = {}
    for reason_str, group_df in (trade_df.groupby("reason") if len(trade_df) else []):
        by_reason[str(reason_str)] = {"count_int": int(len(group_df)), "pnl_usd": float(group_df["pnl"].sum()), "mean_return": float((group_df["pnl"] / group_df["cost"]).mean()),
                                      "median_return": float((group_df["pnl"] / group_df["cost"]).median()), "median_holding_sessions": float(group_df["holding_sessions"].median())}
    terminal_within_df = trade_df[(trade_df["reason"] == "terminal") & (trade_df["holding_sessions"] <= 252)] if len(trade_df) else trade_df
    break_df = trade_df[trade_df["reason"] == "break"] if len(trade_df) else trade_df
    return {
        "events_per_year": float(event_panel[start_pos_int:end_pos_int].sum() / years_float),
        "confirmations_per_year": float(conf_panel[start_pos_int:end_pos_int].sum() / years_float),
        "entries_per_year": float(len(trade_df) / years_float), "entries_total_int": int(len(trade_df)),
        "entries_by_year": {str(k): int(v) for k, v in pd.Series(1, index=date_index[trade_df["entry_pos"].to_numpy()]).groupby(lambda ts: ts.year).sum().items()} if len(trade_df) else {},
        "positions_mean": float(concurrency_vec.mean()), "positions_max": int(concurrency_vec.max()), "share_sessions_no_position": float((concurrency_vec == 0).mean()),
        "exits_by_reason": by_reason,
        "precision_terminal_within_252": float(len(terminal_within_df) / len(trade_df)) if len(trade_df) else float("nan"),
        "break_fill_vs_level_mean": float(break_df["fill_vs_stop"].mean()) if len(break_df) else float("nan"),
        "break_fill_vs_level_p5": float(break_df["fill_vs_stop"].quantile(0.05)) if len(break_df) else float("nan"),
        "worst_episode_return": float((trade_df["pnl"] / trade_df["cost"]).min()) if len(trade_df) else float("nan"),
        "queue_drops": dict(policy_obj.dropped_dict), "queue_left_at_end_int": len(policy_obj.queue_list),
        "commission_total_usd": float(sim_dict["commission_ser"].loc[:end_ts].sum()), "total_pnl_usd": float(sim_dict["total_ser"].loc[:end_ts].iloc[-1] - sim_dict["capital_float"]),
    }


def run_one_cell(feature_obj: V2FeatureBook, cell: cells_module.VCell, cost_list: list[str], record_bool: bool, capital_float: float = common.CAPITAL_BASE_FLOAT) -> dict:
    out_dict = {"key": cell.key_str, "returns": {}, "cashw": {}, "meta": {}}
    for cost_str in cost_list:
        policy_obj = PolicyV(feature_obj, cell)
        sim_dict = simulate(feature_obj, policy_obj, slippage_float=COST_DICT[cost_str], terminal_factor_float=cell.terminal_factor_float, capital_float=capital_float,
                            record_positions_bool=record_bool and cost_str == "engine")
        out_dict["returns"][cost_str] = sim_dict["return_ser"]
        out_dict["cashw"][cost_str] = sim_dict["cash_weight_ser"]
        meta_dict = run_summary(sim_dict, feature_obj.date_index)
        meta_dict["v"] = v_diagnostics(sim_dict, policy_obj, feature_obj, cell)
        out_dict["meta"][cost_str] = meta_dict
        if cost_str == "engine":
            out_dict["trade_df"] = sim_dict["trade_df"]
            out_dict["exposure"] = sim_dict["exposure_ser"]
            if record_bool:
                out_dict["intent_df"] = sim_dict["intent_df"]
                out_dict["position_log"] = sim_dict["position_log"]
    return out_dict


def run_cells(tag_str: str, cell_list: list, cost_list: list[str], record_keys: tuple[str, ...] = (), capital_float: float = common.CAPITAL_BASE_FLOAT) -> None:
    feature_obj = load_feature_book()
    common.log_progress(f"{tag_str}: {len(cell_list)} cells x {cost_list}, capital {capital_float:.0f}")
    start_float = time.perf_counter()
    result_list = []
    for cell in cell_list:
        cell_start_float = time.perf_counter()
        out_dict = run_one_cell(feature_obj, cell, cost_list, cell.key_str in record_keys, capital_float)
        result_list.append(out_dict)
        print(f"  {cell.key_str}: {time.perf_counter() - cell_start_float:.1f}s, entries {out_dict['meta'][cost_list[0]]['v']['entries_total_int']}", flush=True)
    for cost_str in cost_list:
        pd.DataFrame({o["key"]: o["returns"][cost_str] for o in result_list}).to_parquet(common.RESULTS_DIR_PATH / f"returns_{tag_str}_{cost_str}.parquet")
        pd.DataFrame({o["key"]: o["cashw"][cost_str] for o in result_list}).to_parquet(common.RESULTS_DIR_PATH / f"cashw_{tag_str}_{cost_str}.parquet")
    common.write_json(f"meta_{tag_str}.json", {o["key"]: o["meta"] for o in result_list})
    if "engine" in cost_list:
        with open(common.RESULTS_DIR_PATH / f"trades_{tag_str}.pkl", "wb") as file_obj:
            pickle.dump({o["key"]: o["trade_df"] for o in result_list}, file_obj)
        pd.DataFrame({o["key"]: o["exposure"] for o in result_list}).to_parquet(common.RESULTS_DIR_PATH / f"exposure_{tag_str}.parquet")
    for out_dict in result_list:
        if "intent_df" in out_dict:
            with open(common.RESULTS_DIR_PATH / f"intents_{safe_name(out_dict['key'])}.pkl", "wb") as file_obj:
                pickle.dump({"intent_df": out_dict["intent_df"], "position_log": out_dict["position_log"], "trade_df": out_dict["trade_df"]}, file_obj)
    common.log_progress(f"{tag_str} done: {len(result_list)} cells in {time.perf_counter() - start_float:.0f}s")


def run_parity() -> dict:
    from new_pod_search_20260927 import engine_parity

    feature_obj = load_feature_book()
    symbol_list = feature_obj.symbol_list
    cell = cells_module.V0_CELL
    replica_sim = simulate(feature_obj, PolicyV(feature_obj, cell), record_positions_bool=True)
    intents_dict = engine_parity.intents_from_intent_df(replica_sim["intent_df"], symbol_list)
    traded_symbol_list = sorted({symbol_str for intent_list in intents_dict.values() for symbol_str, _, _ in intent_list})
    engine_obj = engine_parity.run_engine_replay(intents_dict, traded_symbol_list + ["SPY"], f"parity_{safe_name(cell.key_str)}")
    report_dict = engine_parity.parity_report(replica_sim, engine_obj, feature_obj.date_index, symbol_list)
    report_dict["traded_symbols_int"] = len(traded_symbol_list)
    report_dict["gate_passed_bool"] = bool(report_dict["passed_bool"])
    common.write_json("parity_gate.json", {cell.key_str: report_dict, "gate_passed_bool": report_dict["gate_passed_bool"]})
    common.log_progress(f"PARITY {cell.key_str}: corr {report_dict['daily_return_corr_float']:.7f}, cagr gap {report_dict['cagr_gap_pp_float']:.4f} pp, max daily diff "
                        f"{report_dict['max_abs_daily_diff_float']:.2e}, position mismatches {report_dict['position_mismatch_sessions_int']}; GATE {'PASSED' if report_dict['gate_passed_bool'] else 'FAILED'}")
    return report_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass1", action="store_true")
    parser.add_argument("--pass2", action="store_true")
    parser.add_argument("--parity", action="store_true")
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--halves", action="store_true")
    parser.add_argument("--small-account", action="store_true")
    parser.add_argument("--costs", default="engine,stress")
    parser.add_argument("--invariance-v1", action="store_true")
    parser.add_argument("--invariance-v2", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    common.RESULTS_DIR_PATH.mkdir(parents=True, exist_ok=True)
    if args.pass1:
        data_module.run_pass1()
    if args.pass2:
        data_module.run_pass2()
    if args.parity:
        run_parity()
    if args.grid:
        run_cells("V_ALL", cells_module.all_primary_cells(), [c for c in args.costs.split(",") if c], record_keys=(cells_module.V0_CELL.key_str,))
    if args.halves:
        for half_str in cells_module.HALF_TUPLE:
            run_cells(f"V_{half_str}", cells_module.half_cells(half_str), ["engine"])
    if args.small_account:
        run_cells("V_ALL_25k", [cells_module.V0_CELL], ["engine"], capital_float=common.SMALL_CAPITAL_FLOAT)
    if args.invariance_v1 or args.invariance_v2:
        from merger_arb_v2_20260927 import invariance

        if args.invariance_v1:
            invariance.run_v1()
        if args.invariance_v2:
            invariance.run_v2()
    if args.analyze:
        from merger_arb_v2_20260927 import analyze

        analyze.main()


if __name__ == "__main__":
    main()
