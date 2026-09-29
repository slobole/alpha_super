"""Stage B pod runs (PREREG section 7; research only): the 24-cell grid, candidate selection by plateau, the HEDGED,
REV1 / REV5 and IndNeutralize-free label pods, the small-account runs and the U2 cross-check cells.

Outputs (results dir): returns_<U>_<cost>.parquet, cashw_<U>_<cost>.parquet (cash share for the sweep),
meta_<U>.json (run summaries), trades_<U>.pkl, exposure_<U>.parquet; small-account and U2 files carry a suffix.
"""

from __future__ import annotations

import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from alpha101_20260928 import common
from alpha101_20260928 import composites
from alpha101_20260928 import data as data_module
from alpha101_20260928.policies import PolicyAlpha
from alpha101_20260928.simulate import run_summary, simulate
from trend_breakout_20260927 import analyze as trend_analyze
from trend_breakout_20260927.features import DailyFeatureBook

_WORKER: dict = {}


def safe_name(key_str: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in key_str)


def load_context(universe_str: str, hedged_bool: bool) -> dict:
    universe_dict = data_module.load_universe(universe_str, sh_bool=hedged_bool)
    feature_obj = DailyFeatureBook(universe_dict)
    score_dict = {}
    control_dict = composites.load_controls(universe_str)
    for name_str in ("C_EQ", "C_WF", "C_EQ_noInd"):
        score_dict[name_str] = np.asarray(composites.load_composite(universe_str, name_str), dtype=np.float64)
    score_dict["REV1"] = control_dict["REV1_raw"]
    score_dict["REV5"] = control_dict["REV5_raw"]
    if hedged_bool:  # SH is the appended last column: no score
        score_dict = {k: np.column_stack([v, np.full(v.shape[0], np.nan)]) for k, v in score_dict.items()}
    return {"universe": universe_dict, "features": feature_obj, "scores": score_dict, "sh_idx": universe_dict.get("sh_idx"), "universe_str": universe_str}


def build_policy(context_dict: dict, composite_str: str, n_int: int, b_int: int, form_str: str) -> PolicyAlpha:
    return PolicyAlpha(context_dict["features"], context_dict["scores"][composite_str], n_int, b_int, hedged_bool=form_str == "HEDGED", sh_idx=context_dict["sh_idx"])


def run_one_cell(context_dict: dict, key_str: str, cost_list: list[str], capital_float: float = common.CAPITAL_BASE_FLOAT, record_bool: bool = False) -> dict:
    spec = common.parse_cell_key(key_str)
    feature_obj = context_dict["features"]
    out_dict = {"key": key_str, "returns": {}, "cashw": {}, "meta": {}, "capital": capital_float}
    for cost_str in cost_list:
        policy_obj = build_policy(context_dict, spec["composite"], spec["n_int"], spec["b_int"], spec["form"])
        sim_dict = simulate(feature_obj, policy_obj, slippage_float=common.COST_DICT[cost_str], capital_float=capital_float, record_positions_bool=record_bool and cost_str == "engine")
        out_dict["returns"][cost_str] = sim_dict["return_ser"]
        out_dict["cashw"][cost_str] = sim_dict["cash_weight_ser"]
        meta_dict = run_summary(sim_dict, feature_obj.date_index)
        meta_dict["policy"] = {"decisions_int": policy_obj.decisions_int, "no_candidate_days_int": policy_obj.no_slot_int, "zero_share_skips_int": policy_obj.zero_share_skips_int}
        out_dict["meta"][cost_str] = meta_dict
        if cost_str == "engine":
            out_dict["trade_df"] = sim_dict["trade_df"]
            out_dict["exposure"] = sim_dict["exposure_ser"]
            out_dict["positions"] = sim_dict["position_count_ser"]
            if record_bool:
                out_dict["intent_df"] = sim_dict["intent_df"]
                out_dict["position_log"] = sim_dict["position_log"]
    return out_dict


def _worker_init(universe_str: str, hedged_bool: bool) -> None:
    _WORKER["context"] = load_context(universe_str, hedged_bool)


def _worker_run(task_tuple) -> dict:
    key_str, cost_list, capital_float = task_tuple
    start_float = time.perf_counter()
    out_dict = run_one_cell(_WORKER["context"], key_str, cost_list, capital_float)
    out_dict["seconds"] = time.perf_counter() - start_float
    return out_dict


def run_cells(universe_str: str, key_list: list[str], cost_list: list[str], workers_int: int, tag_str: str, hedged_bool: bool, capital_float: float = common.CAPITAL_BASE_FLOAT) -> list[dict]:
    common.log_progress(f"pods {tag_str}: {len(key_list)} cells x {cost_list}, workers {workers_int}, capital {capital_float:.0f}")
    task_list = [(key_str, cost_list, capital_float) for key_str in key_list]
    start_float = time.perf_counter()
    result_list: list[dict] = []
    if workers_int <= 1 or len(task_list) == 1:
        context_dict = load_context(universe_str, hedged_bool)
        for task_tuple in task_list:
            cell_start_float = time.perf_counter()
            out_dict = run_one_cell(context_dict, task_tuple[0], task_tuple[1], task_tuple[2])
            out_dict["seconds"] = time.perf_counter() - cell_start_float
            result_list.append(out_dict)
            common.log_progress(f"  {out_dict['key']}: {out_dict['seconds']:.0f}s, final NAV {out_dict['meta'][cost_list[0]]['final_nav_usd']:.0f}", f"log_pods_{tag_str}.txt")
    else:
        with ProcessPoolExecutor(max_workers=min(workers_int, len(task_list)), initializer=_worker_init, initargs=(universe_str, hedged_bool)) as pool:
            for out_dict in pool.map(_worker_run, task_list, chunksize=1):
                result_list.append(out_dict)
                common.log_progress(f"  {out_dict['key']}: {out_dict['seconds']:.0f}s, final NAV {out_dict['meta'][cost_list[0]]['final_nav_usd']:.0f}", f"log_pods_{tag_str}.txt")
    write_results(result_list, cost_list, tag_str)
    common.log_progress(f"pods {tag_str} done: {len(result_list)} cells in {time.perf_counter() - start_float:.0f}s")
    return result_list


def write_results(result_list: list[dict], cost_list: list[str], tag_str: str) -> None:
    """Merge into the tag's parquet / json files (existing keys are replaced)."""
    for cost_str in cost_list:
        for kind_str in ("returns", "cashw"):
            path_obj = common.RESULTS_DIR_PATH / f"{kind_str}_{tag_str}_{cost_str}.parquet"
            new_df = pd.DataFrame({o["key"]: o[kind_str][cost_str] for o in result_list})
            if path_obj.exists():
                old_df = pd.read_parquet(path_obj)
                old_df.index = pd.to_datetime(old_df.index)
                old_df = old_df.drop(columns=[c for c in old_df.columns if c in new_df.columns])
                new_df = pd.concat([old_df, new_df], axis=1)
            new_df.to_parquet(path_obj)
    meta_path = common.RESULTS_DIR_PATH / f"meta_{tag_str}.json"
    meta_dict = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta_dict.update({o["key"]: o["meta"] for o in result_list})
    common.write_json(f"meta_{tag_str}.json", meta_dict)
    if "engine" in cost_list:
        trade_path = common.RESULTS_DIR_PATH / f"trades_{tag_str}.pkl"
        trade_dict = pickle.load(open(trade_path, "rb")) if trade_path.exists() else {}
        trade_dict.update({o["key"]: o["trade_df"] for o in result_list})
        with open(trade_path, "wb") as file_obj:
            pickle.dump(trade_dict, file_obj)
        for kind_str in ("exposure", "positions"):
            path_obj = common.RESULTS_DIR_PATH / f"{kind_str}_{tag_str}.parquet"
            new_df = pd.DataFrame({o["key"]: o[kind_str] for o in result_list})
            if path_obj.exists():
                old_df = pd.read_parquet(path_obj)
                old_df.index = pd.to_datetime(old_df.index)
                old_df = old_df.drop(columns=[c for c in old_df.columns if c in new_df.columns])
                new_df = pd.concat([old_df, new_df], axis=1)
            new_df.to_parquet(path_obj)


def grid_keys() -> list[str]:
    return [common.cell_key(c, n, b) for c in common.COMPOSITE_TUPLE for n in common.N_TUPLE for b in common.B_TUPLE]


def run_grid(universe_str: str, cost_list: list[str], workers_int: int) -> None:
    run_cells(universe_str, grid_keys(), cost_list, workers_int, universe_str, hedged_bool=False)


# ----------------------------------------------------------------------------------------------------------------------
# candidates by plateau (PREREG section 10)
# ----------------------------------------------------------------------------------------------------------------------
def load_returns(tag_str: str, cost_str: str, sweep_bool: bool, bil_ser: pd.Series | None = None) -> pd.DataFrame | None:
    path_obj = common.RESULTS_DIR_PATH / f"returns_{tag_str}_{cost_str}.parquet"
    if not path_obj.exists():
        return None
    ret_df = pd.read_parquet(path_obj)
    ret_df.index = pd.to_datetime(ret_df.index)
    ret_df = ret_df.loc[:common.END_TS]
    if not sweep_bool:
        return ret_df
    cashw_df = pd.read_parquet(common.RESULTS_DIR_PATH / f"cashw_{tag_str}_{cost_str}.parquet")
    cashw_df.index = pd.to_datetime(cashw_df.index)
    bil_ser = common.load_bil_ret_ser() if bil_ser is None else bil_ser
    return pd.DataFrame({k: common.sweep_return_ser(ret_df[k], cashw_df[k], bil_ser) for k in ret_df.columns})


def plateau_selection(sharpe_by_key_dict: dict[str, float], composite_str: str) -> dict:
    """The cell with the highest median standalone FULL Sharpe over its clipped 3x3 (N, B) box; ties nearest A0."""
    key_arr = np.empty((len(common.N_TUPLE), len(common.B_TUPLE)), dtype=object)
    for i, n_int in enumerate(common.N_TUPLE):
        for j, b_int in enumerate(common.B_TUPLE):
            key_arr[i, j] = common.cell_key(composite_str, n_int, b_int)
    value_arr, plateau_arr = trend_analyze.plateau_matrix(key_arr, lambda k: sharpe_by_key_dict.get(k, float("nan")), True, True)
    a0_pos = (common.N_TUPLE.index(common.A0_N_INT), common.B_TUPLE.index(common.A0_B_INT))
    best_pos = max(((i, j) for i in range(key_arr.shape[0]) for j in range(key_arr.shape[1])),
                   key=lambda p: (round(float(np.nan_to_num(plateau_arr[p], nan=-9.0)), 12), -(abs(p[0] - a0_pos[0]) + abs(p[1] - a0_pos[1]))))
    return {"composite": composite_str, "centre": key_arr[best_pos], "grid_pos": [int(best_pos[0]), int(best_pos[1])], "plateau_value": float(plateau_arr[best_pos]),
            "neighbourhood": [key_arr[p] for p in trend_analyze.neighbour_list(*best_pos, key_arr.shape, True, True)],
            "rows_N": list(common.N_TUPLE), "cols_B": list(common.B_TUPLE), "keys": key_arr.tolist(), "values": value_arr.tolist(), "plateau": plateau_arr.tolist(), "a0_pos": [a0_pos[0], a0_pos[1]]}


def select_candidates(universe_str: str = "U1") -> dict:
    """Plateau selection on standalone FULL Sharpe with the sweep at engine costs; writes candidates.json."""
    sweep_df = load_returns(universe_str, "engine", True)
    full_start_str, full_end_str = common.BLOCK_DICT["FULL"]
    p1_start_str, p1_end_str = common.BLOCK_DICT["P1"]
    sharpe_dict = {k: common.metric_dict(sweep_df[k].loc[full_start_str:full_end_str])["sharpe"] for k in grid_keys() if k in sweep_df}
    sharpe_p1_dict = {k: common.metric_dict(sweep_df[k].loc[p1_start_str:p1_end_str])["sharpe"] for k in grid_keys() if k in sweep_df}
    candidate_dict: dict = {"selection_metric": "standalone FULL Sharpe with the sweep, engine costs; plateau = median over the clipped 3x3 (N, B) box", "long_only": {}, "walk_forward_P1": {}, "hedged": {}}
    for composite_str in common.COMPOSITE_TUPLE:
        candidate_dict["long_only"][composite_str] = plateau_selection(sharpe_dict, composite_str)
        candidate_dict["walk_forward_P1"][composite_str] = plateau_selection(sharpe_p1_dict, composite_str)
        centre_spec = common.parse_cell_key(candidate_dict["long_only"][composite_str]["centre"])
        candidate_dict["hedged"][composite_str] = {"composite": composite_str, "centre": common.cell_key(composite_str, centre_spec["n_int"], centre_spec["b_int"], "HEDGED"), "built_on": candidate_dict["long_only"][composite_str]["centre"]}
    candidate_dict["candidate_keys"] = [v["centre"] for v in candidate_dict["long_only"].values()] + [v["centre"] for v in candidate_dict["hedged"].values()]
    common.write_json("candidates.json", candidate_dict)
    common.log_progress("candidates: " + json.dumps({c: (candidate_dict["long_only"][c]["centre"], round(candidate_dict["long_only"][c]["plateau_value"], 3)) for c in common.COMPOSITE_TUPLE}))
    return candidate_dict


def load_candidates() -> dict:
    return common.read_json("candidates.json")


# ----------------------------------------------------------------------------------------------------------------------
# label pods and cross-checks
# ----------------------------------------------------------------------------------------------------------------------
def run_extras(universe_str: str, cost_list: list[str], workers_int: int) -> None:
    candidate_dict = load_candidates() or select_candidates(universe_str)
    hedged_keys = [v["centre"] for v in candidate_dict["hedged"].values()]
    run_cells(universe_str, hedged_keys, cost_list, workers_int, universe_str, hedged_bool=True)
    label_keys = [common.cell_key("REV1", common.LABEL_N_INT, common.LABEL_B_INT), common.cell_key("REV5", common.LABEL_N_INT, common.LABEL_B_INT)]
    c_eq_spec = common.parse_cell_key(candidate_dict["long_only"]["C_EQ"]["centre"])
    label_keys.append(common.cell_key("C_EQ_noInd", c_eq_spec["n_int"], c_eq_spec["b_int"]))
    run_cells(universe_str, label_keys, cost_list, workers_int, universe_str, hedged_bool=False)


def run_small_accounts(universe_str: str = "U1") -> None:
    candidate_dict = load_candidates()
    for capital_float in common.SMALL_ACCOUNT_CAPITAL_TUPLE:
        tag_str = f"{universe_str}_cap{int(capital_float)}"
        long_keys = [v["centre"] for v in candidate_dict["long_only"].values()]
        hedged_keys = [v["centre"] for v in candidate_dict["hedged"].values()]
        run_cells(universe_str, long_keys, ["engine"], 2, tag_str, hedged_bool=False, capital_float=capital_float)
        run_cells(universe_str, hedged_keys, ["engine"], 2, tag_str, hedged_bool=True, capital_float=capital_float)


def run_u2_candidates(cost_list: list[str], workers_int: int) -> None:
    """R4: the same cells and forms on U2 with U2's own composite and weights."""
    candidate_dict = load_candidates()
    long_keys = [v["centre"] for v in candidate_dict["long_only"].values()]
    hedged_keys = [v["centre"] for v in candidate_dict["hedged"].values()]
    a0_key = common.cell_key(common.A0_COMPOSITE_STR, common.A0_N_INT, common.A0_B_INT)
    if a0_key not in long_keys:
        long_keys.append(a0_key)
    run_cells("U2", long_keys, cost_list, workers_int, "U2", hedged_bool=False)
    run_cells("U2", hedged_keys, cost_list, workers_int, "U2", hedged_bool=True)
