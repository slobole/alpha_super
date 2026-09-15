"""Research-only metadata-copy acceleration; economic runner stays unchanged."""
from __future__ import annotations
from collections.abc import Mapping
from types import MappingProxyType
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import pandas as pd

REPO_PATH = Path(__file__).resolve().parents[2]
if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))
from scripts.research import run_mr_deeper_dip_study as study


class FrozenMetadata(Mapping):
    """Recursively read-only metadata; copying an immutable value is unnecessary."""
    __slots__ = ("_value_map",)

    def __init__(self, value_map):
        object.__setattr__(self, "_value_map", MappingProxyType({
            key_str: FrozenMetadata(value_obj) if isinstance(value_obj, Mapping)
            else value_obj for key_str, value_obj in value_map.items()}))
        if any(not isinstance(value_obj, (str, int, float, bool, type(None), FrozenMetadata))
               for value_obj in self._value_map.values()):
            raise TypeError("Unexpected mutable metadata value.")

    def __setattr__(self, name_str, value_obj):
        raise TypeError("Frozen research metadata cannot be changed.")

    def __getitem__(self, key_str):
        return self._value_map[key_str]

    def __iter__(self):
        return iter(self._value_map)

    def __len__(self):
        return len(self._value_map)

    def __deepcopy__(self, memo_dict):
        return self


def freeze_frame_metadata(pricing_df):
    before_dict = pricing_df.attrs.copy()
    pricing_df.attrs = {key_str: FrozenMetadata(value_obj) if isinstance(value_obj, Mapping)
                        else value_obj for key_str, value_obj in before_dict.items()}
    if pricing_df.attrs != before_dict:
        raise AssertionError("Metadata values changed.")
    return pricing_df


def main():
    parser_obj = argparse.ArgumentParser()
    parser_obj.add_argument("strategy", choices=study.STRATEGY_TUPLE)
    args_obj = parser_obj.parse_args()
    strategy_str = args_obj.strategy
    result_path = study.STUDY_PATH/"runs"/strategy_str
    reference_path = result_path/"original_native_reference"
    reference_path.mkdir(exist_ok=True)
    reference_names = ("native_transactions.csv", "native_equity.csv", "native_parity.json")
    for name_str in reference_names:
        if not (reference_path/name_str).exists():
            if not (result_path/name_str).exists():
                raise AssertionError("Unaccelerated full native reference must finish first.")
            shutil.copy2(result_path/name_str, reference_path/name_str)
    reference_dict = json.loads((reference_path/"native_parity.json").read_text())
    receipt_dict = {
        "wrapper_sha256": study.sha256_file(Path(__file__)),
        "economic_runner_sha256": study.sha256_file(Path(study.__file__)),
        "change": "In-memory immutable metadata mappings; all keys/values and raw frozen files unchanged.",
        "original_reference_sha256": {name_str: study.sha256_file(reference_path/name_str)
                                     for name_str in reference_names},
        "full_native_parity": False,
    }
    original_load = study.load_inputs
    original_run = study.run_daily

    def accelerated_load(strategy_name_str):
        pricing_df, universe_df, calendar_idx = original_load(strategy_name_str)
        return freeze_frame_metadata(pricing_df), universe_df, calendar_idx

    def checked_run(strategy_obj, pricing_df, calendar_idx, **kwargs_dict):
        started_float = time.perf_counter()
        result_obj = original_run(strategy_obj, pricing_df, calendar_idx, **kwargs_dict)
        if not isinstance(strategy_obj, study.DeeperDipResearchMixin):
            expected_df = pd.read_csv(reference_path/"native_transactions.csv", float_precision="round_trip")
            actual_df = strategy_obj.get_transactions().copy()
            actual_df["bar"] = actual_df["bar"].astype(str)
            pd.testing.assert_frame_equal(actual_df.drop(columns="order_id").reset_index(drop=True),
                                          expected_df.drop(columns="order_id").reset_index(drop=True),
                                          check_dtype=False, rtol=0, atol=1e-10)
            expected_equity_df = pd.read_csv(reference_path/"native_equity.csv", index_col=0, parse_dates=True, float_precision="round_trip")
            pd.testing.assert_frame_equal(strategy_obj.results, expected_equity_df,
                                          check_dtype=False, check_freq=False, rtol=0, atol=1e-9)
            if abs(strategy_obj.cash-reference_dict["cash"])>1e-8:
                raise AssertionError("Fast native cash differs from original full native reference.")
            receipt_dict.update(full_native_parity=True,
                                accelerated_native_seconds=time.perf_counter()-started_float)
            study.write_json(result_path/"runtime_acceleration.json", receipt_dict)
            print("PASS full unaccelerated/accelerated native parity", strategy_str, flush=True)
        return result_obj

    receipt_dict["status"] = "running"
    existing_dict = {str(path_obj.relative_to(result_path)): study.sha256_file(path_obj)
                     for path_obj in result_path.glob("*/complete.json")}
    study.write_json(result_path/"runtime_acceleration.json", receipt_dict)
    study.load_inputs, study.run_daily = accelerated_load, checked_run
    try:
        study.run_cells(strategy_str)
    except Exception as error_obj:
        receipt_dict.update(status="failed", error=str(error_obj))
        study.write_json(result_path/"runtime_acceleration.json", receipt_dict)
        raise
    complete_dict = {str(path_obj.relative_to(result_path)): study.sha256_file(path_obj)
                     for path_obj in result_path.glob("*/complete.json")}
    if any(complete_dict.get(key_str)!=value_str for key_str,value_str in existing_dict.items()):
        raise AssertionError("Previously completed cell was overwritten.")
    receipt_dict.update(
        status="completed", skipped_existing_cells=existing_dict,
        newly_executed_cells={key_str:value_str for key_str,value_str in complete_dict.items()
                              if key_str not in existing_dict})
    study.write_json(result_path/"runtime_acceleration.json", receipt_dict)


if __name__ == "__main__":
    main()
