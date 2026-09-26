"""Freeze local Norgate inputs for the 2026-09-26 leakage investigation."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys

import pandas as pd

from strategies.momentum import strategy_mo_atr_normalized_ndx as base_module
from strategies.momentum import strategy_mo_atr_normalized_ndx_vxn_scaled as vxn_module
from strategies.momentum import strategy_mo_mosaic_russell1000 as mosaic_module


def main():
    output_path = Path(sys.argv[1])
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_list = []
    for family_str in ["ndx", "mosaic"]:
        cache_path = output_path / f"{family_str}_inputs.pkl"
        if not cache_path.exists():
            config_obj = replace(
                vxn_module.DEFAULT_CONFIG if family_str == "ndx" else mosaic_module.DEFAULT_CONFIG,
                end_date_str="2026-09-25",
            )
            if family_str == "ndx":
                input_tuple = vxn_module.get_vxn_scaled_atr_normalized_ndx_data(
                    config_obj, include_total_return_benchmark_bool=True,
                )
            else:
                input_tuple = base_module.get_atr_normalized_ndx_data(
                    config_obj, include_total_return_benchmark_bool=True,
                )
            pd.to_pickle(input_tuple, cache_path)
        else:
            input_tuple = pd.read_pickle(cache_path)
        prices_df, universe_df, schedule_df = input_tuple[:3]
        digest_obj = sha256()
        with cache_path.open("rb") as input_file:
            for block_bytes in iter(lambda: input_file.read(1024 * 1024), b""):
                digest_obj.update(block_bytes)
        record_dict = {
            "family": family_str, "sha256": digest_obj.hexdigest(),
            "cache_file": str(cache_path), "rows": len(prices_df),
            "columns": len(prices_df.columns), "symbols": len(universe_df.columns),
            "first_date": str(prices_df.index.min()), "last_date": str(prices_df.index.max()),
            "monthly_execution_count": len(schedule_df),
            "padding": "ALLMARKETDAYS preserved for controlled comparison; padded observations remain a disclosed data limitation",
            "adjustments": prices_df.attrs.get("norgate_adjustment_by_symbol_dict", {}),
        }
        manifest_list.append(record_dict)
        print({key_str: value_obj for key_str, value_obj in record_dict.items() if key_str != "adjustments"}, flush=True)
        (output_path / "input_manifest.json").write_text(json.dumps(manifest_list, indent=2), encoding="utf-8")
        del input_tuple, prices_df, universe_df, schedule_df


if __name__ == "__main__":
    main()
