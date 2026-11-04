"""Which input of a leg's path changes between two runs in separate processes. Prints a hash per input and saves the path's names.

Usage: determinism_probe.py <leg 1|2|3> <output json>
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_PATH = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_PATH))
from strategies.momentum.momentum_capsule_legs import LEG1_CONFIG, LEG2_CONFIG, LEG3_CONFIG, build_leg_panel, build_target_path, load_leg_inputs  # noqa: E402

CONFIG_DICT = {"1": LEG1_CONFIG, "2": LEG2_CONFIG, "3": LEG3_CONFIG}


def array_hash_str(value_vec: np.ndarray) -> str:
    return hashlib.sha256(np.nan_to_num(np.asarray(value_vec, dtype=float), nan=-1.2345e300).tobytes()).hexdigest()[:16]


def main(leg_str: str, output_str: str) -> None:
    config_obj = CONFIG_DICT[leg_str]
    input_dict = load_leg_inputs(config_obj)
    pricing_data_df = input_dict.pop("pricing_data_df")
    panel = build_leg_panel(config_obj, pricing_data_df, **input_dict)
    path_obj = build_target_path(config_obj, panel)
    hash_dict = {
        "columns": hashlib.sha256("|".join(map(str, pricing_data_df.columns.tolist())).encode()).hexdigest()[:16],
        "universe": array_hash_str(input_dict["universe_df"].to_numpy(dtype=float)),
        "symbols": hashlib.sha256("|".join(map(str, panel.symbol_vec)).encode()).hexdigest()[:16],
        "sector_labels": hashlib.sha256(json.dumps(input_dict["sector_label_dict"], sort_keys=True).encode()).hexdigest()[:16],
        "group_labels": hashlib.sha256(json.dumps(input_dict["group_label_dict"], sort_keys=True).encode()).hexdigest()[:16],
        "close": array_hash_str(panel.close_mat),
        "high": array_hash_str(panel.high_mat),
        "low": array_hash_str(panel.low_mat),
        "raw_close": array_hash_str(panel.raw_close_mat),
        "turnover": array_hash_str(panel.dollar_volume_mat),
        "member": array_hash_str(panel.member_mat),
        "gate": array_hash_str(panel.gate_close_vec),
        "market": array_hash_str(panel.market_ret_vec) if panel.market_ret_vec is not None else None,
        "sector_returns": array_hash_str(panel.sector_ret_df.to_numpy(dtype=float)) if panel.sector_ret_df is not None else None,
        "vxn": array_hash_str(panel.vxn_vec) if panel.vxn_vec is not None else None,
    }
    field_mat_dict = {"close": panel.close_mat, "high": panel.high_mat, "low": panel.low_mat, "raw_close": panel.raw_close_mat, "turnover": panel.dollar_volume_mat}
    output_dict = {
        "hash_dict": hash_dict,
        "column_hash_dict": {field_str: {str(symbol_str): array_hash_str(value_mat[:, column_int]) for column_int, symbol_str in enumerate(panel.symbol_vec)} for field_str, value_mat in field_mat_dict.items()},
        "unknown_sector_list": sorted(symbol_str for symbol_str, label_str in input_dict["sector_label_dict"].items() if label_str == "UNKNOWN"),
        "unknown_group_list": sorted(symbol_str for symbol_str, label_str in (input_dict["group_label_dict"] or {}).items() if label_str == "UNKNOWN"),
        "sector_label_dict": input_dict["sector_label_dict"],
        "group_label_dict": input_dict["group_label_dict"],
        "book_dict": {str(decision_ts.date()): sorted(book_ser.index) for decision_ts, book_ser in path_obj.book_by_decision_dict.items()},
    }
    Path(output_str).write_text(json.dumps(output_dict), encoding="utf-8")
    print(json.dumps(hash_dict), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
