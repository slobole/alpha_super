"""Runner for the NDX momentum pod parameter-robustness study (research only).

Frozen plan: docs/research/NDX_PARAM_ROBUSTNESS_PREREG_20260926.md.

Steps (run in this order; each writes to results/research/ndx_param_robustness_20260926/):
    uv run python scripts/research/run_ndx_param_robustness_study.py --prepare NDX
    uv run python scripts/research/run_ndx_param_robustness_study.py --parity
    uv run python scripts/research/run_ndx_param_robustness_study.py --grids NDX
    uv run python scripts/research/run_ndx_param_robustness_study.py --invariance
    uv run python scripts/research/run_ndx_param_robustness_study.py --prepare SP500 --grids SP500
    uv run python scripts/research/run_ndx_param_robustness_study.py --prepare R1000 --grids R1000
    uv run python scripts/research/run_ndx_param_robustness_study.py --engine-parity <cell_key> ...
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ndx_param_robustness_core as core  # noqa: E402

STAGE2_OUT_PATH = Path(r"C:/Users/User/Documents/workspace/0_papers/index/review/stage2/ndx_split_bias/outputs")
STAGE3_OUT_PATH = Path(r"C:/Users/User/Documents/workspace/0_papers/index/review/stage3/ndx_score_redesign/outputs")
PARITY_END_TS = pd.Timestamp("2026-07-24")


def out_path(name_str: str) -> Path:
    core.RESULTS_DIR_PATH.mkdir(parents=True, exist_ok=True)
    return core.RESULTS_DIR_PATH / name_str


def selection_record(universe_dict: dict, target_list: list[dict]) -> dict[str, list[str]]:
    date_index = universe_dict["date_index"]
    symbol_list = universe_dict["symbol_list"]
    return {
        str(date_index[t["decision_pos"]].date()): sorted(symbol_list[i] for i in t["symbol_idx_vec"])
        for t in target_list
    }


# ----------------------------------------------------------------------------------------------------------------------
# parity gate (PREREG section 6): replica vs real engine runs of already-known variants
# ----------------------------------------------------------------------------------------------------------------------
def run_parity() -> None:
    universe_dict = core.load_universe("NDX")
    feature_obj = core.FeatureBook(universe_dict)

    # schedule identity with the repo's own rebalance schedule
    schedule_dict = core.build_schedule(universe_dict, 0)
    traded_decision_index = universe_dict["date_index"][schedule_dict["decision_pos_vec"][schedule_dict["trade_vec"]]]
    repo_decision_index = pd.DatetimeIndex(universe_dict["repo_schedule_df"]["decision_date_ts"])
    schedule_match_bool = bool(traded_decision_index.equals(repo_decision_index))

    reference_dict = {
        "L": (core.L_CELL, STAGE2_OUT_PATH / "variant_L"),
        "B": (core.B_CELL, STAGE2_OUT_PATH / "variant_B"),
        "S20": (core.ANCHOR_CELL, STAGE3_OUT_PATH / "variant_S20"),
        "R": (dataclasses.replace(core.ANCHOR_CELL, denominator_str="none"), STAGE3_OUT_PATH / "variant_R"),
    }
    report_dict: dict = {"schedule_matches_repo_bool": schedule_match_bool, "variants": {}}
    for label_str, (cell, prefix_path) in reference_dict.items():
        start_float = time.perf_counter()
        target_list = core.build_target_list(feature_obj, cell)
        sim_dict = core.simulate(universe_dict, target_list)
        engine_nav_ser = pd.read_csv(f"{prefix_path}_daily.csv.gz", index_col="date", parse_dates=True)["total_value"]
        engine_ret_ser = engine_nav_ser.pct_change().iloc[1:]
        replica_ret_ser = sim_dict["return_ser"]
        common_index = engine_ret_ser.index.intersection(replica_ret_ser.index)
        common_index = common_index[common_index <= PARITY_END_TS]
        diff_ser = (replica_ret_ser.reindex(common_index) - engine_ret_ser.reindex(common_index)).abs()

        def cagr(return_ser: pd.Series) -> float:
            return float((1 + return_ser).prod() ** (252.0 / len(return_ser)) - 1)

        ranked_df = pd.read_csv(f"{prefix_path}_ranked.csv.gz")
        engine_top_dict = {
            d: sorted(g.sort_values("rank")["symbol"].head(10).tolist()) for d, g in ranked_df.groupby("decision_date")
        }
        replica_top_dict = selection_record(universe_dict, target_list)
        compared_list = [d for d in engine_top_dict if d in replica_top_dict]
        mismatch_list = [d for d in compared_list if engine_top_dict[d] != replica_top_dict[d]]
        # engine logs no ranking when the regime gate is closed; the replica then holds an empty list
        replica_extra_list = [d for d in replica_top_dict if d not in engine_top_dict and replica_top_dict[d]]
        report_dict["variants"][label_str] = {
            "cell_key_str": cell.key_str,
            "daily_return_corr_float": float(np.corrcoef(replica_ret_ser.reindex(common_index), engine_ret_ser.reindex(common_index))[0, 1]),
            "max_abs_daily_diff_float": float(diff_ser.max()),
            "cagr_replica_float": cagr(replica_ret_ser.reindex(common_index)),
            "cagr_engine_float": cagr(engine_ret_ser.reindex(common_index)),
            "final_nav_replica_float": float(sim_dict["total_ser"].iloc[-1]),
            "final_nav_engine_float": float(engine_nav_ser.iloc[-1]),
            "top10_dates_compared_int": len(compared_list),
            "top10_mismatch_dates": mismatch_list[:10],
            "top10_mismatch_count_int": len(mismatch_list),
            "replica_nonempty_dates_without_engine_ranking": replica_extra_list[:10],
            "seconds_float": round(time.perf_counter() - start_float, 2),
        }
        print(label_str, json.dumps(report_dict["variants"][label_str], indent=1))
    gate_bool = schedule_match_bool and all(
        v["daily_return_corr_float"] >= 0.9999
        and abs(v["cagr_replica_float"] - v["cagr_engine_float"]) <= 0.0005
        and v["top10_mismatch_count_int"] == 0
        for v in report_dict["variants"].values()
    )
    report_dict["parity_gate_passed_bool"] = bool(gate_bool)
    out_path("parity_replica_vs_engine.json").write_text(json.dumps(report_dict, indent=2))
    print("PARITY GATE PASSED" if gate_bool else "PARITY GATE FAILED")


# ----------------------------------------------------------------------------------------------------------------------
# grids
# ----------------------------------------------------------------------------------------------------------------------
def capacity_stats(order_frac_arr: np.ndarray, date_index: pd.DatetimeIndex, start_ts: pd.Timestamp | None) -> dict:
    """Participation per $1 of pod AUM: p = (|order| / V) / ADV20. AUM at which p reaches x = x / p."""
    if len(order_frac_arr) == 0:
        return {}
    pos_vec = order_frac_arr[:, 0].astype(int)
    frac_vec = order_frac_arr[:, 1]
    adv_vec = order_frac_arr[:, 2]
    keep_vec = np.isfinite(adv_vec) & (adv_vec > 0) & (date_index[pos_vec] <= PARITY_END_TS)
    if start_ts is not None:
        keep_vec &= date_index[pos_vec] >= start_ts
    per_dollar_vec = frac_vec[keep_vec] / adv_vec[keep_vec]
    q95_float = float(np.quantile(per_dollar_vec, 0.95))
    max_float = float(per_dollar_vec.max())
    return {
        "orders_int": int(keep_vec.sum()),
        "orders_without_adv_int": int((~(np.isfinite(adv_vec) & (adv_vec > 0))).sum()),
        "aum_p95_1pct_usd": 0.01 / q95_float,
        "aum_p95_5pct_usd": 0.05 / q95_float,
        "aum_max_1pct_usd": 0.01 / max_float,
        "aum_max_5pct_usd": 0.05 / max_float,
    }


def run_grids(universe_str: str) -> None:
    universe_dict = core.load_universe(universe_str)
    feature_obj = core.FeatureBook(universe_dict)
    date_index = universe_dict["date_index"]
    adv_arr = feature_obj.adv20_dollar()
    split_factor_arr = feature_obj.split_factor() if universe_str == "NDX" else None
    cost_tuple = (("engine", core.ENGINE_SLIPPAGE_FLOAT, False),)
    if universe_str == "NDX":
        cost_tuple += (("stress5", core.STRESS_SLIPPAGE_FLOAT, False), ("commfix", core.ENGINE_SLIPPAGE_FLOAT, True))
    cell_list = core.all_cell_list()
    schedule_cache_dict: dict = {}
    return_dict: dict[str, dict[str, pd.Series]] = {cost_str: {} for cost_str, _, _ in cost_tuple}
    meta_dict: dict[str, dict] = {}
    selection_dict: dict[str, dict] = {}
    start_float = time.perf_counter()
    for cell_int, cell in enumerate(cell_list):
        target_list = core.build_target_list(feature_obj, cell, schedule_cache_dict)
        selection_dict[cell.key_str] = selection_record(universe_dict, target_list)
        for cost_str, slippage_float, commission_fixed_bool in cost_tuple:
            sim_dict = core.simulate(
                universe_dict,
                target_list,
                slippage_float=slippage_float,
                commission_fixed_bool=commission_fixed_bool,
                split_factor_arr=split_factor_arr,
                adv_arr=adv_arr if cost_str == "engine" else None,
            )
            return_dict[cost_str][cell.key_str] = sim_dict["return_ser"]
            if cost_str == "engine":
                window_mask = (sim_dict["total_ser"].index >= pd.Timestamp("2000-01-04")) & (sim_dict["total_ser"].index <= PARITY_END_TS)
                nav_ser = sim_dict["total_ser"][window_mask]
                years_float = len(nav_ser) / 252.0
                exposure_vec = np.array([t["weight_vec"].sum() for t in target_list])
                meta_dict[cell.key_str] = {
                    "cell": dataclasses.asdict(cell),
                    "scale_free_bool": cell.scale_free_bool,
                    "turnover_x_per_year": float(sim_dict["traded_notional_ser"][window_mask].sum() / nav_ser.mean() / years_float),
                    "commission_pct_nav_per_year": float(sim_dict["commission_ser"][window_mask].sum() / nav_ser.mean() / years_float),
                    "mean_target_exposure": float(exposure_vec.mean()),
                    "share_invested_decisions": float((exposure_vec > 0).mean()),
                    "missing_dividend_count_int": sim_dict["missing_dividend_count_int"],
                    "capacity_full": capacity_stats(sim_dict["order_frac_arr"], date_index, None),
                    "capacity_2021_2026": capacity_stats(sim_dict["order_frac_arr"], date_index, pd.Timestamp("2021-01-01")),
                }
        if cell_int % 20 == 0:
            print(f"{universe_str}: {cell_int + 1}/{len(cell_list)} cells, {time.perf_counter() - start_float:.0f}s", flush=True)
    for cost_str, series_dict in return_dict.items():
        pd.DataFrame(series_dict).to_parquet(out_path(f"returns_{universe_str}_{cost_str}.parquet"))
    out_path(f"cell_meta_{universe_str}.json").write_text(json.dumps(meta_dict, indent=1))
    out_path(f"selections_{universe_str}.json").write_text(json.dumps(selection_dict))
    stage_map_dict = {
        stage_str: [{"row": row_col[0], "col": row_col[1], "key": cell.key_str} for row_col, cell in cell_list_]
        for stage_str, cell_list_ in core.stage_grid_dict().items()
    }
    out_path("stage_map.json").write_text(json.dumps(stage_map_dict, indent=1))
    print(f"{universe_str}: {len(cell_list)} cells done in {time.perf_counter() - start_float:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", choices=tuple(core.UNIVERSE_INDEXNAME_DICT))
    parser.add_argument("--parity", action="store_true")
    parser.add_argument("--grids", choices=tuple(core.UNIVERSE_INDEXNAME_DICT))
    args = parser.parse_args()
    if args.prepare:
        start_float = time.perf_counter()
        universe_dict = core.prepare_universe(args.prepare)
        print(
            f"{args.prepare}: {len(universe_dict['symbol_list'])} symbols, {len(universe_dict['date_index'])} sessions "
            f"{universe_dict['date_index'][0].date()}..{universe_dict['date_index'][-1].date()}, "
            f"{time.perf_counter() - start_float:.0f}s"
        )
    if args.parity:
        run_parity()
    if args.grids:
        run_grids(args.grids)


if __name__ == "__main__":
    main()
