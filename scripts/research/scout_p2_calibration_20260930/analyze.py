"""Summaries and the frozen decision rules of PROTOCOL.md applied to the results.

    uv run python scripts/research/scout_p2_calibration_20260930/analyze.py
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR_PATH.parents[2]))

from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH  # noqa: E402

RESULT_DIR_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "p2_calibration"
CASE_ORDER_LIST = ["noise_iid", "noise_garch", "edge_030", "edge_050", "edge_080"]
SINGLE_GATE_DICT = {
    "naive t>=2": "pass_naive_bool",
    "MCPT plain": "pass_mcpt_plain_bool",
    "MCPT stratified": "pass_mcpt_stratified_bool",
    "DSR": "pass_dsr_bool",
    "WF": "pass_wf_bool",
    "PBO": "pass_pbo_bool",
}


def _rate_table(frame: pd.DataFrame, column_dict: dict[str, pd.Series]) -> pd.DataFrame:
    table_dict = {}
    for label_str, pass_ser in column_dict.items():
        table_dict[label_str] = pass_ser.groupby(frame["case_str"]).mean().reindex(CASE_ORDER_LIST)
    return pd.DataFrame(table_dict).T


def gate_set_columns(frame: pd.DataFrame, mcpt_column_str: str) -> dict[str, pd.Series]:
    core_dict = {"MCPT": frame[mcpt_column_str], "DSR": frame["pass_dsr_bool"], "WF": frame["pass_wf_bool"]}
    set_dict: dict[str, pd.Series] = {}
    for size_int in (1, 2, 3):
        for name_tuple in combinations(core_dict, size_int):
            set_dict[" + ".join(name_tuple)] = np.logical_and.reduce([core_dict[name] for name in name_tuple])
    set_dict["MCPT + DSR + WF + PBO"] = set_dict["MCPT + DSR + WF"] & frame["pass_pbo_bool"]
    set_dict["any 2 of MCPT, DSR, WF"] = (sum(core_dict[name].astype(int) for name in core_dict) >= 2)
    return {label_str: pd.Series(values, index=frame.index) for label_str, values in set_dict.items()}


def main() -> None:
    frame = pd.read_parquet(RESULT_DIR_PATH / "synthetic.parquet")
    single_table = _rate_table(frame, {label: frame[column] for label, column in SINGLE_GATE_DICT.items()})
    print("Single gates: pass rate by case (noise = false pass, edge = power)")
    print(single_table.round(3).to_string())

    # Decision 1: MCPT null.
    stratified_false_pass_float = single_table.loc["MCPT stratified", "noise_garch"]
    power_loss_float = single_table.loc["MCPT plain", "edge_050"] - single_table.loc["MCPT stratified", "edge_050"]
    keep_stratified_bool = stratified_false_pass_float <= 0.05 and power_loss_float <= 0.10
    mcpt_column_str = "pass_mcpt_stratified_bool" if keep_stratified_bool else "pass_mcpt_plain_bool"
    print(f"\nDecision 1: stratified false pass {stratified_false_pass_float:.3f}, power loss vs plain on edge_050 "
          f"{power_loss_float:.3f} -> keep {'stratified' if keep_stratified_bool else 'plain'} null")

    # Decision 2: gate set.
    set_table = _rate_table(frame, gate_set_columns(frame, mcpt_column_str))
    set_table["admissible"] = (set_table["noise_garch"] <= 0.05) & (set_table["edge_080"] >= 0.70 - 1e-9)  # float-safe (fixed after the review; the first run used 1 - x <= 0.30)
    set_table["test_count"] = [label.count("+") + 1 if not label.startswith("any") else 3 for label in set_table.index]
    print("\nGate sets: pass rate by case")
    print(set_table.round(3).to_string())
    admissible_table = set_table[set_table["admissible"]]
    if admissible_table.empty:
        fallback_table = set_table[set_table["noise_garch"] <= 0.05]
        chosen_str = fallback_table["edge_080"].idxmax()
        print(f"\nDecision 2: no admissible set; lowest false reject on edge_080 with false pass <= 5%: {chosen_str}")
    else:
        best_power_float = admissible_table["edge_050"].max()
        near_best_table = admissible_table[admissible_table["edge_050"] >= best_power_float - 0.03]
        chosen_str = near_best_table.sort_values(["test_count", "edge_050"], ascending=[True, False]).index[0]
        print(f"\nDecision 2: chosen gate set = {chosen_str}")

    # Decision 3: PBO as an extra gate.
    chosen_ser = gate_set_columns(frame, mcpt_column_str).get(chosen_str)
    with_pbo_ser = chosen_ser & frame["pass_pbo_bool"]
    false_pass_drop_float = chosen_ser[frame.case_str == "noise_garch"].mean() - with_pbo_ser[frame.case_str == "noise_garch"].mean()
    power_cost_float = chosen_ser[frame.case_str == "edge_050"].mean() - with_pbo_ser[frame.case_str == "edge_050"].mean()
    pbo_gate_bool = false_pass_drop_float >= 0.02 and power_cost_float <= 0.03
    print(f"Decision 3: PBO lowers false pass by {false_pass_drop_float:.3f} at power cost {power_cost_float:.3f} -> "
          f"{'gate' if pbo_gate_bool else 'diagnostic only'}")

    print("\nMedians by case")
    print(frame.groupby("case_str")[["chosen_sharpe_float", "dsr_float", "mcpt_plain_p_float", "pbo_float", "n_eff_float",
                                     "wf_positive_designs_int"]].median().reindex(CASE_ORDER_LIST).round(3).to_string())
    print("\nTrue configuration chosen (L=20, theta=0):",
          frame.assign(true_bool=frame.chosen_index_int == 6).groupby("case_str")["true_bool"].mean().reindex(CASE_ORDER_LIST).round(2).to_dict())

    summary_dict = {
        "single_gate_table": single_table.round(4).to_dict(),
        "gate_set_table": set_table.round(4).to_dict(),
        "mcpt_null_str": "stratified" if keep_stratified_bool else "plain",
        "chosen_gate_set_str": chosen_str,
        "pbo_gate_bool": pbo_gate_bool,
    }
    (RESULT_DIR_PATH / "synthetic_summary.json").write_text(json.dumps(summary_dict, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
