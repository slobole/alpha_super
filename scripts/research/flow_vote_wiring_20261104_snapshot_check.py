"""Proof: a real Flow Vote snapshot exported by the actual exporter, read by the actual reader, equals the research loader's direct frame.

    PYTHONUTF8=1 .venv/Scripts/python.exe scripts/research/flow_vote_wiring_20261104_snapshot_check.py <temp snapshot root>
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH))

import pandas as pd


def main() -> None:
    snapshot_root_str = sys.argv[1]
    from scripts.export_norgate_snapshot import export_profile_snapshot, _latest_norgate_session_date_str
    from data.flow_vote_snapshot_contract import FLOW_VOTE_PROFILE_STR
    from strategies.taa_beyond_6040 import strategy_taa_month_end_flow_vote as vote_module

    snapshot_date_str = _latest_norgate_session_date_str()
    start_float = time.time()
    snapshot_dir_path = export_profile_snapshot(snapshot_root_str=snapshot_root_str, profile_str=FLOW_VOTE_PROFILE_STR, snapshot_date_str=snapshot_date_str, start_date_str="2002-07-26", overwrite_bool=True)
    export_seconds_float = time.time() - start_float
    config_obj = replace(vote_module.DEFAULT_CONFIG, end_date_str=snapshot_date_str)
    direct_df = vote_module.get_month_end_flow_data(config_obj)
    os.environ["ALPHA_USE_NORGATE_SNAPSHOT_BOOL"] = "true"
    from data import norgate_snapshot_store
    os.environ[norgate_snapshot_store.NORGATE_SNAPSHOT_ROOT_ENV_STR] = snapshot_root_str
    norgate_snapshot_store.clear_snapshot_manifest_cache()
    snapshot_df = vote_module.get_month_end_flow_data(config_obj)
    result_dict = {"snapshot_date_str": snapshot_date_str, "snapshot_dir_str": str(snapshot_dir_path), "export_seconds_float": round(export_seconds_float, 1),
                   "rows_int": len(snapshot_df), "columns_equal_bool": list(direct_df.columns) == list(snapshot_df.columns),
                   "index_equal_bool": bool(direct_df.index.equals(snapshot_df.index)), "dtypes_equal_bool": bool((direct_df.dtypes == snapshot_df.dtypes).all()),
                   "attrs_equal_bool": direct_df.attrs == snapshot_df.attrs, "index_name_direct_str": str(direct_df.index.name), "index_name_snapshot_str": str(snapshot_df.index.name)}
    try:
        pd.testing.assert_frame_equal(direct_df, snapshot_df, check_exact=True)
        result_dict["frame_equal_bool"] = True
    except AssertionError as error_obj:
        result_dict["frame_equal_bool"] = False
        result_dict["frame_error_str"] = str(error_obj)[:2000]
    signal_direct_df = vote_module.MonthEndVoteFlowStrategy(config_obj).compute_signals(direct_df)
    signal_snapshot_df = vote_module.MonthEndVoteFlowStrategy(config_obj).compute_signals(snapshot_df)
    result_dict["signals_equal_bool"] = bool(signal_direct_df.equals(signal_snapshot_df))
    out_path = ROOT_PATH / "results" / "research" / "flow_vote_wiring_20261104" / "snapshot_check.json"
    out_path.write_text(json.dumps(result_dict, indent=1), encoding="utf-8")
    print(json.dumps(result_dict, indent=1))


if __name__ == "__main__":
    main()
