"""Amendment 1 runner and frozen decision rules (PROTOCOL_AMENDMENT_1.md).

    uv run python scripts/research/scout_p2_calibration_20260930/run_a1.py run
    uv run python scripts/research/scout_p2_calibration_20260930/run_a1.py analyze
"""

from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR_PATH))
sys.path.insert(0, str(SCRIPT_DIR_PATH.parents[2]))

from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH  # noqa: E402

from a1_core import FAMILY_A, FAMILY_B, SESSION_COUNT_INT, calibrate_momentum_a, evaluate_history, garch_returns  # noqa: E402

OUTPUT_DIR_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "p2_calibration"
CASE_SEED_COUNT_DICT = {"noise_garch": 2000, "edge_030": 500, "edge_050": 500, "edge_080": 500}
EDGE_TARGET_DICT = {"edge_030": 0.3, "edge_050": 0.5, "edge_080": 0.8}
FAMILY_DICT = {FAMILY_A.name_str: FAMILY_A, FAMILY_B.name_str: FAMILY_B}


def _task(task_tuple) -> dict:
    family_str, case_str, seed_int, momentum_a_float = task_tuple
    rng_obj = np.random.default_rng(3_000_000 + 10_000 * list(CASE_SEED_COUNT_DICT).index(case_str) + seed_int)
    return_vec = garch_returns(SESSION_COUNT_INT, rng_obj, momentum_a_float)
    result_dict = evaluate_history(return_vec, FAMILY_DICT[family_str], seed_int)
    result_dict.update({"family_str": family_str, "case_str": case_str, "seed_int": seed_int})
    return result_dict


def run() -> None:
    OUTPUT_DIR_PATH.mkdir(parents=True, exist_ok=True)
    started_float = time.time()
    momentum_a_dict = {case_str: calibrate_momentum_a(target_float) for case_str, target_float in EDGE_TARGET_DICT.items()}
    (OUTPUT_DIR_PATH / "a1_momentum_a.json").write_text(json.dumps(momentum_a_dict, indent=2), encoding="utf-8")
    print("calibrated a:", momentum_a_dict, f"({time.time() - started_float:.0f}s)", flush=True)
    # The same simulated histories are used for both families (paired comparison).
    task_list = [
        (family_str, case_str, seed_int, momentum_a_dict.get(case_str, 0.0))
        for family_str in FAMILY_DICT
        for case_str, seed_count_int in CASE_SEED_COUNT_DICT.items()
        for seed_int in range(seed_count_int)
    ]
    row_list = []
    with Pool(14) as pool_obj:
        for row_idx_int, row_dict in enumerate(pool_obj.imap_unordered(_task, task_list, chunksize=4), start=1):
            row_list.append(row_dict)
            if row_idx_int % 500 == 0:
                print(f"{row_idx_int}/{len(task_list)} ({time.time() - started_float:.0f}s)", flush=True)
    pd.DataFrame(row_list).to_parquet(OUTPUT_DIR_PATH / "a1_synthetic.parquet")
    print("done", f"{time.time() - started_float:.0f}s", flush=True)


def gate_set_dict(frame: pd.DataFrame, mcpt_column_str: str) -> dict[str, pd.Series]:
    dsr_ser = frame["dsr_corr_float"] >= 0.95
    mcpt_ser = frame[mcpt_column_str] <= 0.05
    return {
        "DSR-corr": dsr_ser,
        "MCPT plain p<=0.05": frame["mcpt_plain_p_float"] <= 0.05,
        "MCPT stratified p<=0.05": frame["mcpt_stratified_p_float"] <= 0.05,
        "DSR-corr + MCPT": dsr_ser & mcpt_ser,
        "any 2 of DSR-corr, MCPT, WF": (dsr_ser.astype(int) + mcpt_ser.astype(int) + frame["wf_pass_bool"].astype(int)) >= 2,
    }


TEST_COUNT_DICT = {"DSR-corr": 1, "MCPT plain p<=0.05": 1, "MCPT stratified p<=0.05": 1, "DSR-corr + MCPT": 2, "any 2 of DSR-corr, MCPT, WF": 3}
N_FREE_SET = {"MCPT plain p<=0.05", "MCPT stratified p<=0.05"}


def _count_str(pass_ser: pd.Series) -> str:
    return f"{int(pass_ser.sum())}/{len(pass_ser)}"


def analyze() -> None:
    frame = pd.read_parquet(OUTPUT_DIR_PATH / "a1_synthetic.parquet")
    case_list = list(CASE_SEED_COUNT_DICT)
    print("Mean Sharpe of the true configuration by case (family A):",
          frame[frame.family_str == FAMILY_A.name_str].groupby("case_str")["true_config_sharpe_float"].mean().reindex(case_list).round(3).to_dict())

    # Rule 1: MCPT null.
    plain_false_pass_ser = (frame[frame.case_str == "noise_garch"]["mcpt_plain_p_float"] <= 0.05).groupby(frame["family_str"]).mean()
    mcpt_column_str = "mcpt_plain_p_float" if (plain_false_pass_ser <= 0.06).all() else "mcpt_stratified_p_float"
    print(f"Rule 1: plain MCPT false pass by family {plain_false_pass_ser.round(4).to_dict()} -> use {mcpt_column_str}")

    set_dict = gate_set_dict(frame, mcpt_column_str)
    extra_dict = {
        "DSR-cluster (old)": frame["dsr_cluster_float"] >= 0.95,
        "MCPT plain p<=0.01": frame["mcpt_plain_p_float"] <= 0.01,
        "walk-forward": frame["wf_pass_bool"],
        "PBO <= 0.2": frame["pbo_float"] <= 0.2,
        "naive t >= 2": frame["naive_t_float"] >= 2.0,
    }
    table_row_list = []
    for label_str, pass_ser in {**set_dict, **extra_dict}.items():
        for family_str in FAMILY_DICT:
            row_dict = {"set_str": label_str, "family_str": family_str}
            for case_str in case_list:
                mask = (frame.family_str == family_str) & (frame.case_str == case_str)
                row_dict[case_str] = float(pass_ser[mask].mean())
                row_dict[case_str + "_count"] = _count_str(pass_ser[mask])
            table_row_list.append(row_dict)
    table = pd.DataFrame(table_row_list)
    pd.set_option("display.width", 250)
    print(table[["set_str", "family_str"] + case_list].round(4).to_string(index=False))

    # Rules 2-4.
    candidate_list = list(set_dict)
    summary_list = []
    for label_str in candidate_list:
        sub = table[table.set_str == label_str].set_index("family_str")
        admissible_bool = bool((sub["noise_garch"] <= 0.06).all() and ((1 - sub["edge_080"]) <= 0.30 + 1e-12).all())
        summary_list.append(
            {
                "set_str": label_str,
                "admissible_bool": admissible_bool,
                "max_false_pass_float": float(sub["noise_garch"].max()),
                "min_power_080_float": float(sub["edge_080"].min()),
                "mean_power_050_float": float(sub["edge_050"].mean()),
                "mean_power_030_float": float(sub["edge_030"].mean()),
                "test_count_int": TEST_COUNT_DICT[label_str],
                "n_free_bool": label_str in N_FREE_SET,
            }
        )
    summary = pd.DataFrame(summary_list)
    print(summary.round(4).to_string(index=False))
    admissible = summary[summary.admissible_bool]
    if admissible.empty:
        fallback = summary[summary.max_false_pass_float <= 0.06]
        chosen_str = fallback.sort_values("min_power_080_float", ascending=False).iloc[0]["set_str"] if not fallback.empty else "none"
        print(f"No admissible set; fallback choice: {chosen_str}")
    else:
        best_float = admissible["mean_power_050_float"].max()
        tied = admissible[admissible["mean_power_050_float"] >= best_float - 0.03]
        chosen_str = tied.sort_values(["test_count_int", "n_free_bool", "mean_power_050_float"], ascending=[True, False, False]).iloc[0]["set_str"]
        print(f"Chosen gate set: {chosen_str}")

    print("\nMedian benchmark (annual Sharpe) and N_eff by family:")
    print(frame.groupby("family_str")[["benchmark_annual_float", "n_eff_float"]].median().round(3).to_string())
    print("\nOld clustered DSR false pass by N_eff (noise_garch):")
    noise = frame[frame.case_str == "noise_garch"]
    print((noise["dsr_cluster_float"] >= 0.95).groupby([noise.family_str, noise.n_eff_float]).agg(["mean", "count"]).round(3).to_string())

    (OUTPUT_DIR_PATH / "a1_summary.json").write_text(
        json.dumps({"mcpt_column_str": mcpt_column_str, "chosen_str": chosen_str, "table": table.to_dict("records"),
                    "summary": summary.to_dict("records")}, indent=2, default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    {"run": run, "analyze": analyze}[sys.argv[1]]()
