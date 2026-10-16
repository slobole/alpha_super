"""Run the unchanged TAA host replay while retaining its exact native inputs."""
from __future__ import annotations

import hashlib
import copy
import gzip
import inspect
import json
import pickle
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT_PATH = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT_PATH))

import data.norgate_loader as loader_module
from scripts.research.wired_signoff_20261015 import taa_ndx_native_replay as replay_module


def file_hash_str(path_obj: Path) -> str:
    return hashlib.sha256(path_obj.read_bytes()).hexdigest()


def main() -> None:
    output_path = REPO_ROOT_PATH / "results/review/wired_signoff_20261015/taa_ndx_native"
    input_path = output_path / "taa_inputs"
    input_path.mkdir(parents=True, exist_ok=True)
    frozen_path = output_path.parent / "frozen_manifest.json"
    frozen_dict = json.loads(frozen_path.read_text(encoding="utf-8"))
    source_dict = {row_dict["path"]: row_dict["sha256"] for row_dict in frozen_dict["source_files"]}
    for path_str, expected_str in source_dict.items():
        if file_hash_str(REPO_ROOT_PATH / path_str) != expected_str:
            raise RuntimeError(f"Frozen production source changed before replay: {path_str}")
    fred_path = output_path.parent / "inputs/DTB3.csv"
    fred_hash_str = file_hash_str(fred_path)
    replay_path = Path(replay_module.__file__)
    replay_hash_str = file_hash_str(replay_path)
    wrapper_hash_str = file_hash_str(Path(__file__))
    input_row_list = []
    cache_path = output_path.parent / "inputs/native_loader_cache.pkl.gz"
    cache_hash_str = file_hash_str(cache_path)
    with gzip.open(cache_path, "rb") as cache_file:
        cache_dict = pickle.load(cache_file)

    def captured_loader_fn(*argument_list, **keyword_dict):
        request_str = repr(("price", argument_list, sorted(keyword_dict.items())))
        if request_str not in cache_dict:
            raise RuntimeError(f"Frozen TAA input is missing: {request_str}")
        price_df = copy.deepcopy(cache_dict[request_str])
        file_path = input_path / f"price_{len(input_row_list):02d}.parquet"
        price_df.to_parquet(file_path, compression="zstd")
        input_row_list.append({"request_str": repr((argument_list, sorted(keyword_dict.items()))),
            "path_str": str(file_path.relative_to(REPO_ROOT_PATH)).replace("\\", "/"),
            "bytes_int": file_path.stat().st_size, "sha256_str": file_hash_str(file_path),
            "rows_int": len(price_df), "start_str": str(price_df.index[0]) if len(price_df) else None,
            "end_str": str(price_df.index[-1]) if len(price_df) else None,
            "attrs_dict": dict(price_df.attrs)})
        return price_df
    captured_loader_fn.__signature__ = inspect.signature(loader_module.load_price_timeseries)

    arguments_obj = SimpleNamespace(output_dir=output_path, fred_cache=fred_path, capital=1_000_000,
        minimum_dates=24)
    started_float = time.monotonic()
    # Only frozen prepared frames are read. There is no provider fallback.
    with patch.object(loader_module, "load_price_timeseries", captured_loader_fn):
        result_dict = replay_module.replay_taa(arguments_obj)
    for path_str, expected_str in source_dict.items():
        if file_hash_str(REPO_ROOT_PATH / path_str) != expected_str:
            raise RuntimeError(f"Frozen production source changed during replay: {path_str}")
    if file_hash_str(fred_path) != fred_hash_str or file_hash_str(replay_path) != replay_hash_str or file_hash_str(cache_path) != cache_hash_str:
        raise RuntimeError("Replay source or frozen FRED input changed during capture.")
    if file_hash_str(Path(__file__)) != wrapper_hash_str:
        raise RuntimeError("Capture wrapper changed during replay.")
    for row_dict in input_row_list:
        if file_hash_str(REPO_ROOT_PATH / row_dict["path_str"]) != row_dict["sha256_str"]:
            raise RuntimeError("Captured native input changed during replay.")
    result_dict.update(family_str="taa", elapsed_seconds_float=time.monotonic() - started_float,
        script_sha256_str=replay_hash_str, capture_script_sha256_str=wrapper_hash_str,
        frozen_source_manifest_sha256_str=file_hash_str(frozen_path),
        source_unchanged_bool=True, input_files_unchanged_bool=True, fred_input_sha256_str=fred_hash_str,
        source_str="frozen_prepared_native_loader_cache_plus_frozen_FRED",
        frozen_native_cache_sha256_str=cache_hash_str, native_input_manifest_list=input_row_list)
    original_path = output_path / "taa_actual_host_replay.json"
    preserved_path = output_path / "taa_actual_host_replay_direct_read.json"
    if original_path.exists() and not preserved_path.exists():
        shutil.copyfile(original_path, preserved_path)
    (output_path / "taa_actual_host_replay.json").write_text(json.dumps(result_dict, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"native_input_count_int": len(input_row_list), "source_unchanged_bool": True,
        "elapsed_seconds_float": result_dict["elapsed_seconds_float"], "variant_summary_dict": {
            variant_str: {key_str: value_obj for key_str, value_obj in variant_dict.items() if key_str != "row_list"}
            for variant_str, variant_dict in result_dict["variant_result_dict"].items()}}), flush=True)


if __name__ == "__main__":
    main()
