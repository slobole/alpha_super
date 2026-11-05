"""Proof: a real T1 snapshot exported by the actual exporter, read by the actual reader, equals the research loader's direct frame.

    PYTHONUTF8=1 ../alpha_super/.venv/Scripts/python.exe scripts/research/beyond6040_wiring_20261105_snapshot_check.py <temp snapshot root>
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
    from data.beyond6040_snapshot_contract import BEYOND6040_HISTORY_START_STR, BEYOND6040_PROFILE_STR
    from strategies.taa_beyond_6040 import strategy_taa_beyond_6040_trend_tilt as tilt_module

    snapshot_date_str = _latest_norgate_session_date_str()
    start_float = time.time()
    snapshot_dir_path = export_profile_snapshot(snapshot_root_str=snapshot_root_str, profile_str=BEYOND6040_PROFILE_STR, snapshot_date_str=snapshot_date_str,
                                                start_date_str=BEYOND6040_HISTORY_START_STR, overwrite_bool=True)
    export_seconds_float = time.time() - start_float
    config_obj = replace(tilt_module.DEFAULT_CONFIG, end_date_str=snapshot_date_str)
    direct_df = tilt_module.get_trend_tilt_data(config_obj)
    os.environ["ALPHA_USE_NORGATE_SNAPSHOT_BOOL"] = "true"
    from data import norgate_snapshot_store
    os.environ[norgate_snapshot_store.NORGATE_SNAPSHOT_ROOT_ENV_STR] = snapshot_root_str
    norgate_snapshot_store.clear_snapshot_manifest_cache()
    snapshot_df = tilt_module.get_trend_tilt_data(config_obj)
    result_dict = {"snapshot_date_str": snapshot_date_str, "snapshot_dir_str": str(snapshot_dir_path), "export_seconds_float": round(export_seconds_float, 1),
                   "rows_int": len(snapshot_df), "columns_equal_bool": list(direct_df.columns) == list(snapshot_df.columns),
                   "index_equal_bool": bool(direct_df.index.equals(snapshot_df.index)), "dtypes_equal_bool": bool((direct_df.dtypes == snapshot_df.dtypes).all()),
                   "attrs_equal_bool": direct_df.attrs == snapshot_df.attrs}
    try:
        pd.testing.assert_frame_equal(direct_df, snapshot_df, check_exact=True)
        result_dict["frame_equal_bool"] = True
    except AssertionError as error_obj:
        result_dict["frame_equal_bool"] = False
        result_dict["frame_error_str"] = str(error_obj)[:2000]
    signal_direct_df = tilt_module.Beyond6040TrendTiltStrategy(**tilt_module.strategy_kwargs_dict(config_obj)).compute_signals(direct_df) if hasattr(tilt_module, "strategy_kwargs_dict") else None
    if signal_direct_df is not None:
        signal_snapshot_df = tilt_module.Beyond6040TrendTiltStrategy(**tilt_module.strategy_kwargs_dict(config_obj)).compute_signals(snapshot_df)
        result_dict["signals_equal_bool"] = bool(signal_direct_df.equals(signal_snapshot_df))
    out_path = ROOT_PATH / "results" / "research" / "beyond6040_wiring_20261105" / "snapshot_check.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result_dict, indent=1), encoding="utf-8")
    print(json.dumps(result_dict, indent=1))


if __name__ == "__main__":
    main()
