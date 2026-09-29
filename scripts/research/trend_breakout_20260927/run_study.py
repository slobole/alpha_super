"""Runner for the trend / breakout study (research only). Frozen plan: docs/research/TREND_BREAKOUT_PREREG_20260927.md.

    python scripts/research/trend_breakout_20260927/run_study.py --prepare NDX [--prepare-monthly NDX]
    python ... --parity-g1
    python ... --parity-g2
    python ... --family A --universe NDX --costs engine,stress,haircut [--offsets] [--workers 6]
    python ... --family B --universe SP500 --costs engine,stress,haircut
    python ... --family C --universe NDX --costs engine,stress,haircut
    python ... --invariance-v1 | --invariance-v2
    python ... --analyze

Outputs (results/research/trend_breakout_20260927/): returns_{family}_{universe}_{cost}.parquet, meta_{family}_{universe}.json,
intents_*.pkl / positions_*.pkl for the parity and invariance cells, parity_*.json, invariance_*.json, results.json.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/research (package parent; also for spawned workers)
from trend_breakout_20260927 import common  # noqa: E402
from trend_breakout_20260927 import cells as cells_module  # noqa: E402
from trend_breakout_20260927 import data as data_module  # noqa: E402
from trend_breakout_20260927 import family_c  # noqa: E402
from trend_breakout_20260927.features import DailyFeatureBook  # noqa: E402
from trend_breakout_20260927.policies import PolicyA, PolicyB, PolicyMonthlyTargets  # noqa: E402
from trend_breakout_20260927.simulate import run_summary, simulate  # noqa: E402

import ndx_param_robustness_core as core  # noqa: E402

COST_DICT = {"engine": (common.ENGINE_SLIPPAGE_FLOAT, 1.0), "stress": (common.STRESS_SLIPPAGE_FLOAT, 1.0), "haircut": (common.ENGINE_SLIPPAGE_FLOAT, common.TERMINAL_HAIRCUT_FLOAT)}
G2_CELL_KEY_TUPLE = ("A|CH-3|CASH|k+0", "A|PT-15|REFILL|k+0", "B|N100|k5|K20|R1|noVXN|noRX", "B|N250|k8|K10|R2|noVXN|noRX")
V2_CELL_KEY_TUPLE = ("A|CH-3|CASH|k+0", "A|PT-15|REFILL|k+0", "B|N100|k5|K20|R1|noVXN|noRX")

_WORKER_FEATURE_OBJ: DailyFeatureBook | None = None
_WORKER_UNIVERSE_STR: str | None = None


def safe_name(key_str: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in key_str)


# ----------------------------------------------------------------------------------------------------------------------
# cell catalogue per (family, universe)
# ----------------------------------------------------------------------------------------------------------------------
def cell_catalogue(family_str: str, universe_str: str, offsets_bool: bool = False) -> list[tuple[str, object]]:
    """(key, spec) pairs. spec is an ACell / BCell / CCell, or a core.Cell reference (family C anchors)."""
    if family_str == "A":
        cell_list = [cells_module.L_REFERENCE_CELL] + cells_module.family_a_cells()
        if offsets_bool:
            base_list = list(cell_list)
            cell_list = [dataclasses.replace(cell, offset_int=offset_int) for cell in base_list for offset_int in cells_module.OFFSET_TUPLE if offset_int != 0]
        return [(cell.key_str, cell) for cell in cell_list]
    if family_str == "B":
        return [(cell.key_str, cell) for cell in cells_module.family_b_cells()]
    if family_str == "C":
        liquidity_str = "none" if universe_str == "NDX" else "REL25"
        pair_list = [(cell.key_str, cell) for cell in cells_module.family_c_cells()]
        pair_list.append((f"C|A0_ref|{liquidity_str}", family_c.a0_reference_cell(liquidity_str)))
        return pair_list
    raise ValueError(family_str)


def build_policy(feature_obj: DailyFeatureBook, universe_str: str, spec_obj):
    if isinstance(spec_obj, cells_module.ACell):
        return PolicyA(feature_obj, spec_obj)
    if isinstance(spec_obj, cells_module.BCell):
        return PolicyB(feature_obj, spec_obj)
    if isinstance(spec_obj, cells_module.CCell):
        liquidity_str = "none" if universe_str == "NDX" else "REL25"
        return PolicyMonthlyTargets(core.build_target_list(feature_obj, family_c.c_core_cell(spec_obj, liquidity_str)))
    if isinstance(spec_obj, core.Cell):
        return PolicyMonthlyTargets(core.build_target_list(feature_obj, spec_obj))
    raise TypeError(type(spec_obj))


def load_feature_book(universe_str: str, family_str: str | None = None) -> DailyFeatureBook:
    universe_dict = data_module.load_universe(universe_str)
    feature_obj = DailyFeatureBook(universe_dict)
    if family_str == "C":
        monthly_close_df = data_module.load_monthly_closes(universe_str)
        feature_obj.numerator_override_dict = family_c.build_c_score_tables(universe_dict, monthly_close_df, cells_module.family_c_cells())
    return feature_obj


def _worker_init(universe_str: str, family_str: str) -> None:
    global _WORKER_FEATURE_OBJ, _WORKER_UNIVERSE_STR
    _WORKER_FEATURE_OBJ = load_feature_book(universe_str, family_str)
    _WORKER_UNIVERSE_STR = universe_str


def run_one_cell(feature_obj: DailyFeatureBook, universe_str: str, key_str: str, spec_obj, cost_list: list[str], record_bool: bool) -> dict:
    out_dict = {"key": key_str, "returns": {}, "meta": {}}
    for cost_str in cost_list:
        slippage_float, haircut_float = COST_DICT[cost_str]
        policy_obj = build_policy(feature_obj, universe_str, spec_obj)
        sim_dict = simulate(feature_obj, policy_obj, slippage_float=slippage_float, historical_share_bool=True,
                            terminal_haircut_float=haircut_float, record_positions_bool=record_bool and cost_str == "engine")
        out_dict["returns"][cost_str] = sim_dict["return_ser"]
        out_dict["meta"][cost_str] = run_summary(sim_dict, feature_obj.date_index)
        if cost_str == "engine":
            out_dict["meta"]["engine"]["mean_names_held"] = float(np.mean([len(h) for _, h, _ in sim_dict["position_log"]])) if record_bool else float("nan")
            out_dict["trade_df"] = sim_dict["trade_df"]
            out_dict["exposure"] = sim_dict["exposure_ser"]
            if record_bool:
                out_dict["intent_df"] = sim_dict["intent_df"]
                out_dict["position_log"] = sim_dict["position_log"]
    return out_dict


def _worker_run(task_tuple) -> dict:
    key_str, spec_obj, cost_list, record_bool = task_tuple
    start_float = time.perf_counter()
    out_dict = run_one_cell(_WORKER_FEATURE_OBJ, _WORKER_UNIVERSE_STR, key_str, spec_obj, cost_list, record_bool)
    out_dict["seconds"] = time.perf_counter() - start_float
    return out_dict


def run_family(family_str: str, universe_str: str, cost_list: list[str], workers_int: int, offsets_bool: bool, record_keys: tuple[str, ...]) -> None:
    pair_list = cell_catalogue(family_str, universe_str, offsets_bool)
    tag_str = f"{family_str}_{universe_str}" + ("_offsets" if offsets_bool else "")
    common.log_progress(f"family {tag_str}: {len(pair_list)} cells x {cost_list}, workers {workers_int}")
    task_list = [(key_str, spec_obj, cost_list, key_str in record_keys) for key_str, spec_obj in pair_list]
    start_float = time.perf_counter()
    result_list: list[dict] = []
    if workers_int <= 1:
        feature_obj = load_feature_book(universe_str, family_str)
        for task_tuple in task_list:
            key_str, spec_obj, cost_list_, record_bool = task_tuple
            cell_start_float = time.perf_counter()
            out_dict = run_one_cell(feature_obj, universe_str, key_str, spec_obj, cost_list_, record_bool)
            out_dict["seconds"] = time.perf_counter() - cell_start_float
            result_list.append(out_dict)
            print(f"  {key_str}: {out_dict['seconds']:.1f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers_int, initializer=_worker_init, initargs=(universe_str, family_str)) as pool:
            for out_dict in pool.map(_worker_run, task_list, chunksize=1):
                result_list.append(out_dict)
                print(f"  {out_dict['key']}: {out_dict['seconds']:.1f}s", flush=True)
    for cost_str in cost_list:
        frame_df = pd.DataFrame({out_dict["key"]: out_dict["returns"][cost_str] for out_dict in result_list})
        frame_df.to_parquet(common.RESULTS_DIR_PATH / f"returns_{tag_str}_{cost_str}.parquet")
    meta_dict = {out_dict["key"]: out_dict["meta"] for out_dict in result_list}
    common.write_json(f"meta_{tag_str}.json", meta_dict)
    if "engine" in cost_list and not offsets_bool:
        with open(common.RESULTS_DIR_PATH / f"trades_{tag_str}.pkl", "wb") as file_obj:
            pickle.dump({out_dict["key"]: out_dict["trade_df"] for out_dict in result_list}, file_obj)
        pd.DataFrame({out_dict["key"]: out_dict["exposure"] for out_dict in result_list}).to_parquet(common.RESULTS_DIR_PATH / f"exposure_{tag_str}.parquet")
    for out_dict in result_list:
        if "intent_df" in out_dict:
            with open(common.RESULTS_DIR_PATH / f"intents_{safe_name(out_dict['key'])}_{universe_str}.pkl", "wb") as file_obj:
                pickle.dump({"intent_df": out_dict["intent_df"], "position_log": out_dict["position_log"], "trade_df": out_dict["trade_df"]}, file_obj)
    common.log_progress(f"family {tag_str} done: {len(result_list)} cells in {time.perf_counter() - start_float:.0f}s")


# ----------------------------------------------------------------------------------------------------------------------
# parity gate
# ----------------------------------------------------------------------------------------------------------------------
def run_parity_g1() -> dict:
    from trend_breakout_20260927 import engine_parity

    universe_str = "NDX"
    feature_obj = load_feature_book(universe_str)
    symbol_list = feature_obj.symbol_list
    replica_sim = simulate(feature_obj, PolicyA(feature_obj, cells_module.L_REFERENCE_CELL), record_positions_bool=True)
    replica_nav_ser = replica_sim["total_ser"]
    replica_pos_dict = engine_parity.replica_daily_positions(replica_sim["position_log"], feature_obj.date_index, symbol_list)
    stored_l_ser = common.load_stored_l_ser()
    stored_nav_ser = common.CAPITAL_BASE_FLOAT * (1.0 + stored_l_ser).cumprod()
    live_obj = engine_parity.run_live_pod()
    live_nav_ser = engine_parity.engine_nav_ser(live_obj)
    live_pos_dict = engine_parity.engine_daily_positions(live_obj, feature_obj.date_index)
    report_dict = {
        "engine_vs_replica": engine_parity.compare_nav(live_nav_ser, replica_nav_ser) | engine_parity.compare_positions(live_pos_dict, replica_pos_dict),
        "engine_vs_stored_ndx_atrfix": engine_parity.compare_nav(live_nav_ser, stored_nav_ser),
        "replica_vs_stored_ndx_atrfix": engine_parity.compare_nav(replica_nav_ser, stored_nav_ser),
        "engine_transactions_int": int(len(live_obj.get_transactions())),
        "replica_round_trips_int": int(len(replica_sim["trade_df"])),
    }
    report_dict["g1_passed_bool"] = bool(report_dict["engine_vs_replica"]["passed_bool"] and report_dict["engine_vs_replica"]["identical_bool"])
    common.write_json("parity_g1.json", report_dict)
    with open(common.RESULTS_DIR_PATH / "parity_g1_engine_nav.pkl", "wb") as file_obj:
        pickle.dump({"engine_nav": live_nav_ser, "replica_nav": replica_nav_ser, "engine_transactions": live_obj.get_transactions()}, file_obj)
    common.log_progress(f"G1: {json.dumps({k: v for k, v in report_dict.items() if k != 'engine_transactions_int'}, default=str)[:1500]}")
    return report_dict


def parse_b_key(key_str: str) -> cells_module.BCell:
    """B|N{n}|k{k}|K{K}|{R1|R2}|{VXN|noVXN}|{RX|noRX} -> BCell (the G2 check cell B/N250/k8/K10/R2 is off-grid by design)."""
    part_list = key_str.split("|")
    if len(part_list) != 7 or part_list[0] != "B":
        raise ValueError(key_str)
    return cells_module.BCell(
        n_int=int(part_list[1][1:]), k_float=float(part_list[2][1:]), slots_int=int(part_list[3][1:]), rank_str=part_list[4],
        vxn_bool=part_list[5] == "VXN", regime_exit_bool=part_list[6] == "RX",
    )


def run_parity_g2(key_list: list[str], universe_str: str = "NDX") -> dict:
    from trend_breakout_20260927 import engine_parity

    feature_obj = load_feature_book(universe_str)
    symbol_list = feature_obj.symbol_list
    catalogue_dict = {key_str: spec_obj for family_str in ("A", "B") for key_str, spec_obj in cell_catalogue(family_str, universe_str)}
    report_dict = {}
    for key_str in key_list:
        spec_obj = catalogue_dict[key_str] if key_str in catalogue_dict else parse_b_key(key_str)
        assert spec_obj.key_str == key_str
        replica_sim = simulate(feature_obj, build_policy(feature_obj, universe_str, spec_obj), record_positions_bool=True)
        intents_dict = engine_parity.intents_from_intent_df(replica_sim["intent_df"], symbol_list)
        engine_obj = engine_parity.run_engine_replay(universe_str, intents_dict, f"parity_{safe_name(key_str)}")
        engine_nav_ser = engine_parity.engine_nav_ser(engine_obj)
        engine_pos_dict = engine_parity.engine_daily_positions(engine_obj, feature_obj.date_index)
        replica_pos_dict = engine_parity.replica_daily_positions(replica_sim["position_log"], feature_obj.date_index, symbol_list)
        cell_report = engine_parity.compare_nav(engine_nav_ser, replica_sim["total_ser"]) | engine_parity.compare_positions(engine_pos_dict, replica_pos_dict)
        cell_report["intents_int"] = int(len(replica_sim["intent_df"]))
        cell_report["engine_transactions_int"] = int(len(engine_obj.get_transactions()))
        cell_report["passed_bool"] = bool(cell_report["passed_bool"] and cell_report["identical_bool"])
        report_dict[key_str] = cell_report
        common.log_progress(f"G2 {key_str}: corr {cell_report['daily_return_corr_float']:.7f}, cagr gap {cell_report['cagr_gap_pp_float']:.4f} pp, "
                            f"max daily diff {cell_report['max_abs_daily_diff_float']:.2e}, position mismatches {cell_report['position_mismatch_sessions_int']}")
        existing_path = common.RESULTS_DIR_PATH / f"parity_g2_{universe_str}.json"
        merged_dict = json.loads(existing_path.read_text()) if existing_path.exists() else {}
        merged_dict.update({key_str: cell_report})
        common.write_json(f"parity_g2_{universe_str}.json", merged_dict)
    return report_dict


# ----------------------------------------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", choices=tuple(common.UNIVERSE_INDEXNAME_DICT))
    parser.add_argument("--prepare-monthly", choices=tuple(common.UNIVERSE_INDEXNAME_DICT))
    parser.add_argument("--family", choices=("A", "B", "C"))
    parser.add_argument("--universe", choices=tuple(common.UNIVERSE_INDEXNAME_DICT), default="NDX")
    parser.add_argument("--costs", default="engine")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--offsets", action="store_true")
    parser.add_argument("--record", default="")
    parser.add_argument("--parity-g1", action="store_true")
    parser.add_argument("--parity-g2", action="store_true")
    parser.add_argument("--g2-cells", default=",".join(G2_CELL_KEY_TUPLE))
    parser.add_argument("--invariance-v1", action="store_true")
    parser.add_argument("--invariance-v1-phantom-free", action="store_true")
    parser.add_argument("--invariance-v2", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    common.RESULTS_DIR_PATH.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        data_module.prepare_universe(args.prepare)
    if args.prepare_monthly:
        data_module.prepare_monthly_closes(args.prepare_monthly)
    if args.parity_g1:
        run_parity_g1()
    if args.parity_g2:
        run_parity_g2([k for k in args.g2_cells.split(",") if k], args.universe)
    if args.family:
        record_keys = tuple(k for k in args.record.split(",") if k)
        run_family(args.family, args.universe, [c for c in args.costs.split(",") if c], args.workers, args.offsets, record_keys)
    if args.invariance_v1 or args.invariance_v2 or args.invariance_v1_phantom_free:
        from trend_breakout_20260927 import invariance

        if args.invariance_v1:
            invariance.run_v1()
        if args.invariance_v1_phantom_free:
            invariance.run_v1_phantom_free()
        if args.invariance_v2:
            invariance.run_v2()
    if args.analyze:
        from trend_breakout_20260927 import analyze

        analyze.main()


if __name__ == "__main__":
    main()
