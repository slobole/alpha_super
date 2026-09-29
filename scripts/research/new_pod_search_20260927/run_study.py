"""Runner for the new-pod search (research only). Frozen plan: docs/research/NEW_POD_SEARCH_PREREG_20260927.md.

    python scripts/research/new_pod_search_20260927/run_study.py --prepare-r1000 | --prepare-monthly SP500
    python ... --parity
    python ... --family M --universe R1000 --costs engine,stress [--workers 1]
    python ... --family S --universe SP500 --costs engine,stress --workers 3 [--offsets]
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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from new_pod_search_20260927 import cells as cells_module  # noqa: E402
from new_pod_search_20260927 import common  # noqa: E402
from new_pod_search_20260927 import data as data_module  # noqa: E402
from new_pod_search_20260927.features import PodFeatureBook, seasonality_score_tables  # noqa: E402
from new_pod_search_20260927.policies import PolicyM, PolicyS  # noqa: E402
from new_pod_search_20260927.simulate import simulate  # noqa: E402
from trend_breakout_20260927.simulate import run_summary  # noqa: E402

COST_DICT = {"engine": common.ENGINE_SLIPPAGE_FLOAT, "stress": common.STRESS_SLIPPAGE_FLOAT}
M_PRIMARY_STR = "R1000"
S_PRIMARY_STR = "SP500"
PARITY_CELL_TUPLE = (("M", M_PRIMARY_STR, cells_module.M0_CELL), ("S", S_PRIMARY_STR, cells_module.S_ANCHOR_DICT["GATED"]), ("S", S_PRIMARY_STR, cells_module.S_ANCHOR_DICT["HEDGED"]))

_WORKER: dict = {}


def safe_name(key_str: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in key_str)


def cell_catalogue(family_str: str, universe_str: str, offsets_bool: bool = False) -> list:
    if family_str == "M":
        return cells_module.family_m_cells(include_sensitivities_bool=universe_str == M_PRIMARY_STR)
    if family_str == "S":
        if offsets_bool:
            return [dataclasses.replace(cell, offset_int=offset_int) for cell in cells_module.S_ANCHOR_DICT.values() for offset_int in cells_module.OFFSET_TUPLE if offset_int != 0]
        return cells_module.family_s_cells()
    raise ValueError(family_str)


def load_context(family_str: str, universe_str: str) -> dict:
    universe_dict = data_module.get_universe(universe_str, sh_bool=family_str == "S")
    feature_obj = PodFeatureBook(universe_dict)
    context_dict = {"universe": universe_dict, "features": feature_obj, "sh_idx": universe_dict.get("sh_idx")}
    if family_str == "S":
        monthly_close_df = data_module.load_monthly_closes_1990(universe_str)
        context_dict["scores"] = seasonality_score_tables(universe_dict, monthly_close_df, list(cells_module.HORIZON_TUPLE))
    return context_dict


def build_policy(context_dict: dict, cell):
    if isinstance(cell, cells_module.MCell):
        return PolicyM(context_dict["features"], cell)
    return PolicyS(context_dict["features"], cell, context_dict["scores"][cell.horizon_str], context_dict["sh_idx"])


def m_diagnostics(sim_dict: dict, policy_obj: PolicyM, feature_obj: PodFeatureBook) -> dict:
    trade_df = sim_dict["trade_df"]
    date_index = feature_obj.date_index
    end_ts = common.END_TS
    if len(trade_df):
        trade_df = trade_df[date_index[trade_df["exit_pos"].to_numpy()] <= end_ts]
    years_float = float((end_ts - common.TRADING_START_TS).days / 365.25)
    event_arr = policy_obj.event_pos_arr  # unused; events counted from the feature panel
    event_panel = feature_obj.event(policy_obj.cell.jump_float)
    start_pos_int = sim_dict["start_pos_int"]
    end_pos_int = int(date_index.searchsorted(end_ts, side="right"))
    event_count_by_year = pd.Series(event_panel[start_pos_int:end_pos_int].sum(axis=1), index=date_index[start_pos_int:end_pos_int]).groupby(lambda ts: ts.year).sum()
    conf_panel = policy_obj.conf_arr
    conf_count_by_year = pd.Series(conf_panel[start_pos_int:end_pos_int].sum(axis=1), index=date_index[start_pos_int:end_pos_int]).groupby(lambda ts: ts.year).sum()
    entries_by_year = pd.Series(1, index=date_index[trade_df["entry_pos"].to_numpy()]).groupby(lambda ts: ts.year).sum() if len(trade_df) else pd.Series(dtype=int)
    by_reason = {}
    for reason_str, group_df in (trade_df.groupby("reason") if len(trade_df) else []):
        by_reason[str(reason_str)] = {"count_int": int(len(group_df)), "pnl_usd": float(group_df["pnl"].sum()), "mean_return": float((group_df["pnl"] / group_df["cost"]).mean()),
                                      "median_holding_sessions": float(group_df["holding_sessions"].median())}
    terminal_within_df = trade_df[(trade_df["reason"] == "terminal") & (trade_df["holding_sessions"] <= 252)] if len(trade_df) else trade_df
    break_df = trade_df[trade_df["reason"] == "break"] if len(trade_df) else trade_df
    return {
        "events_total_int": int(event_panel[start_pos_int:end_pos_int].sum()),
        "events_per_year": float(event_panel[start_pos_int:end_pos_int].sum() / years_float),
        "confirmations_per_year": float(conf_panel[start_pos_int:end_pos_int].sum() / years_float),
        "entries_total_int": int(len(trade_df)),
        "entries_per_year": float(len(trade_df) / years_float),
        "events_by_year": {str(k): int(v) for k, v in event_count_by_year.items()},
        "confirmations_by_year": {str(k): int(v) for k, v in conf_count_by_year.items()},
        "entries_by_year": {str(k): int(v) for k, v in entries_by_year.items()},
        "exits_by_reason": by_reason,
        "precision_terminal_within_252": float(len(terminal_within_df) / len(trade_df)) if len(trade_df) else float("nan"),
        "break_fill_vs_level_mean": float(break_df["fill_vs_stop"].mean()) if len(break_df) else float("nan"),
        "break_fill_vs_level_p5": float(break_df["fill_vs_stop"].quantile(0.05)) if len(break_df) else float("nan"),
        "queue_dropped_zero_share_int": int(policy_obj.dropped_zero_share_int),
        "queue_dropped_delisted_int": int(policy_obj.dropped_delisted_int),
        "queue_left_at_end_int": int(len(policy_obj.queue_list)),
    }


def s_diagnostics(policy_obj: PolicyS, feature_obj: PodFeatureBook, score_arr: np.ndarray) -> dict:
    member_arr = feature_obj.u["member_arr"]
    coverage_list = []
    for pos_int, row_int in policy_obj.row_by_pos_dict.items():
        member_vec = member_arr[pos_int] == 1
        if member_vec.sum() == 0:
            continue
        coverage_list.append(float(np.isfinite(score_arr[row_int][member_vec]).mean()))
    return {"score_coverage_mean": float(np.mean(coverage_list)) if coverage_list else float("nan"), "decisions_int": len(policy_obj.row_by_pos_dict),
            "invested_decisions_int": int(sum(1 for v in policy_obj.selection_log_dict.values() if v))}


def run_one_cell(context_dict: dict, family_str: str, key_str: str, cell, cost_list: list[str], record_bool: bool) -> dict:
    feature_obj = context_dict["features"]
    out_dict = {"key": key_str, "returns": {}, "cashw": {}, "meta": {}}
    for cost_str in cost_list:
        policy_obj = build_policy(context_dict, cell)
        terminal_factor_float = cell.terminal_factor_float if isinstance(cell, cells_module.MCell) else 1.0
        sim_dict = simulate(feature_obj, policy_obj, slippage_float=COST_DICT[cost_str], terminal_factor_float=terminal_factor_float, record_positions_bool=record_bool and cost_str == "engine")
        out_dict["returns"][cost_str] = sim_dict["return_ser"]
        out_dict["cashw"][cost_str] = sim_dict["cash_weight_ser"]
        meta_dict = run_summary(sim_dict, feature_obj.date_index)
        if family_str == "M":
            meta_dict["m"] = m_diagnostics(sim_dict, policy_obj, feature_obj)
        else:
            meta_dict["s"] = s_diagnostics(policy_obj, feature_obj, context_dict["scores"][cell.horizon_str])
        out_dict["meta"][cost_str] = meta_dict
        if cost_str == "engine":
            out_dict["trade_df"] = sim_dict["trade_df"]
            out_dict["exposure"] = sim_dict["exposure_ser"]
            if record_bool:
                out_dict["intent_df"] = sim_dict["intent_df"]
                out_dict["position_log"] = sim_dict["position_log"]
    return out_dict


def _worker_init(family_str: str, universe_str: str) -> None:
    _WORKER["context"] = load_context(family_str, universe_str)
    _WORKER["family"] = family_str


def _worker_run(task_tuple) -> dict:
    key_str, cell, cost_list, record_bool = task_tuple
    start_float = time.perf_counter()
    out_dict = run_one_cell(_WORKER["context"], _WORKER["family"], key_str, cell, cost_list, record_bool)
    out_dict["seconds"] = time.perf_counter() - start_float
    return out_dict


def run_family(family_str: str, universe_str: str, cost_list: list[str], workers_int: int, offsets_bool: bool, record_keys: tuple[str, ...]) -> None:
    cell_list = cell_catalogue(family_str, universe_str, offsets_bool)
    tag_str = f"{family_str}_{universe_str}" + ("_offsets" if offsets_bool else "")
    common.log_progress(f"family {tag_str}: {len(cell_list)} cells x {cost_list}, workers {workers_int}")
    task_list = [(cell.key_str, cell, cost_list, cell.key_str in record_keys) for cell in cell_list]
    start_float = time.perf_counter()
    result_list: list[dict] = []
    if workers_int <= 1:
        context_dict = load_context(family_str, universe_str)
        for key_str, cell, cost_list_, record_bool in task_list:
            cell_start_float = time.perf_counter()
            out_dict = run_one_cell(context_dict, family_str, key_str, cell, cost_list_, record_bool)
            out_dict["seconds"] = time.perf_counter() - cell_start_float
            result_list.append(out_dict)
            print(f"  {key_str}: {out_dict['seconds']:.1f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers_int, initializer=_worker_init, initargs=(family_str, universe_str)) as pool:
            for out_dict in pool.map(_worker_run, task_list, chunksize=1):
                result_list.append(out_dict)
                print(f"  {out_dict['key']}: {out_dict['seconds']:.1f}s", flush=True)
    for cost_str in cost_list:
        pd.DataFrame({o["key"]: o["returns"][cost_str] for o in result_list}).to_parquet(common.RESULTS_DIR_PATH / f"returns_{tag_str}_{cost_str}.parquet")
        pd.DataFrame({o["key"]: o["cashw"][cost_str] for o in result_list}).to_parquet(common.RESULTS_DIR_PATH / f"cashw_{tag_str}_{cost_str}.parquet")
    common.write_json(f"meta_{tag_str}.json", {o["key"]: o["meta"] for o in result_list})
    if "engine" in cost_list and not offsets_bool:
        with open(common.RESULTS_DIR_PATH / f"trades_{tag_str}.pkl", "wb") as file_obj:
            pickle.dump({o["key"]: o["trade_df"] for o in result_list}, file_obj)
        pd.DataFrame({o["key"]: o["exposure"] for o in result_list}).to_parquet(common.RESULTS_DIR_PATH / f"exposure_{tag_str}.parquet")
    for out_dict in result_list:
        if "intent_df" in out_dict:
            with open(common.RESULTS_DIR_PATH / f"intents_{safe_name(out_dict['key'])}_{universe_str}.pkl", "wb") as file_obj:
                pickle.dump({"intent_df": out_dict["intent_df"], "position_log": out_dict["position_log"], "trade_df": out_dict["trade_df"]}, file_obj)
    common.log_progress(f"family {tag_str} done: {len(result_list)} cells in {time.perf_counter() - start_float:.0f}s")


# ----------------------------------------------------------------------------------------------------------------------
# parity gate
# ----------------------------------------------------------------------------------------------------------------------
def run_parity() -> dict:
    from new_pod_search_20260927 import engine_parity

    report_dict = {}
    for family_str, universe_str, cell in PARITY_CELL_TUPLE:
        context_dict = load_context(family_str, universe_str)
        feature_obj = context_dict["features"]
        symbol_list = feature_obj.symbol_list
        replica_sim = simulate(feature_obj, build_policy(context_dict, cell), record_positions_bool=True)
        intents_dict = engine_parity.intents_from_intent_df(replica_sim["intent_df"], symbol_list)
        traded_symbol_list = sorted({symbol_str for intent_list in intents_dict.values() for symbol_str, _, _ in intent_list})
        frame_symbol_list = traded_symbol_list + ([data_module.SH_SYMBOL_STR] if family_str == "S" else []) + ["SPY"]
        engine_obj = engine_parity.run_engine_replay(intents_dict, frame_symbol_list, f"parity_{safe_name(cell.key_str)}")
        cell_report = engine_parity.parity_report(replica_sim, engine_obj, feature_obj.date_index, symbol_list)
        cell_report["universe"] = universe_str
        cell_report["traded_symbols_int"] = len(traded_symbol_list)
        report_dict[cell.key_str] = cell_report
        common.log_progress(f"PARITY {cell.key_str} on {universe_str}: corr {cell_report['daily_return_corr_float']:.7f}, cagr gap {cell_report['cagr_gap_pp_float']:.4f} pp, "
                            f"max daily diff {cell_report['max_abs_daily_diff_float']:.2e}, position mismatches {cell_report['position_mismatch_sessions_int']}, passed {cell_report['passed_bool']}")
        common.write_json("parity_gate.json", report_dict)
        del context_dict, feature_obj, replica_sim, engine_obj
    report_dict["gate_passed_bool"] = bool(all(v["passed_bool"] for k, v in report_dict.items() if k != "gate_passed_bool"))
    common.write_json("parity_gate.json", report_dict)
    common.log_progress(f"PARITY GATE {'PASSED' if report_dict['gate_passed_bool'] else 'FAILED'}")
    return report_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-r1000", action="store_true")
    parser.add_argument("--prepare-monthly", choices=("NDX", "SP500", "R1000"))
    parser.add_argument("--parity", action="store_true")
    parser.add_argument("--family", choices=("M", "S"))
    parser.add_argument("--universe", default=None)
    parser.add_argument("--costs", default="engine")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--offsets", action="store_true")
    parser.add_argument("--record", default="")
    parser.add_argument("--invariance-v1", action="store_true")
    parser.add_argument("--invariance-v2", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    common.RESULTS_DIR_PATH.mkdir(parents=True, exist_ok=True)
    if args.prepare_r1000:
        data_module.prepare_r1000_f64()
    if args.prepare_monthly:
        data_module.prepare_monthly_closes_1990(args.prepare_monthly)
    if args.parity:
        run_parity()
    if args.family:
        universe_str = args.universe or (M_PRIMARY_STR if args.family == "M" else S_PRIMARY_STR)
        run_family(args.family, universe_str, [c for c in args.costs.split(",") if c], args.workers, args.offsets, tuple(k for k in args.record.split(",") if k))
    if args.invariance_v1 or args.invariance_v2:
        from new_pod_search_20260927 import invariance

        if args.invariance_v1:
            invariance.run_v1()
        if args.invariance_v2:
            invariance.run_v2()
    if args.analyze:
        from new_pod_search_20260927 import analyze

        analyze.main()


if __name__ == "__main__":
    main()
