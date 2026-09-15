"""Qualify CORE5 data locally using a fresh, isolated Norgate snapshot.

Reads the local Norgate database; writes only a new output directory. This does
not publish snapshots, contact a VPS/broker, or run a live scheduler.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import replace
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT_PATH))

from data import norgate_snapshot_store as snapshot_module
from scripts.export_norgate_snapshot import export_profile_snapshot
from strategies.taa_beyond_6040 import strategy_taa_adaptive_macro_core5 as core5_module


def compare_core5_frames(direct_price_df: pd.DataFrame, snapshot_price_df: pd.DataFrame) -> dict[str, object]:
    # *** CRITICAL*** Compare the entire histories, including unavailable
    # prefixes. Intersecting dates or filling gaps would hide changed AMA seeds.
    pd.testing.assert_frame_equal(direct_price_df, snapshot_price_df, check_exact=True, check_freq=False)
    if direct_price_df.attrs != snapshot_price_df.attrs:
        raise AssertionError("Direct and snapshot price provenance differs.")
    direct_strategy_obj = core5_module.AdaptiveMacroCore5Strategy()
    snapshot_strategy_obj = core5_module.AdaptiveMacroCore5Strategy()
    direct_signal_df = direct_strategy_obj.compute_signals(direct_price_df)
    snapshot_signal_df = snapshot_strategy_obj.compute_signals(snapshot_price_df)
    pd.testing.assert_frame_equal(direct_signal_df, snapshot_signal_df, check_exact=True, check_freq=False)
    return {
        "price_rows_int": len(direct_price_df),
        "price_columns_int": len(direct_price_df.columns),
        "signal_columns_int": len(direct_signal_df.columns),
        "first_price_date_str": str(direct_price_df.index[0].date()),
        "last_price_date_str": str(direct_price_df.index[-1].date()),
        "exact_full_price_and_feature_parity_bool": True,
    }


def compare_core5_transactions(direct_transaction_df: pd.DataFrame, snapshot_transaction_df: pd.DataFrame) -> None:
    direct_transaction_df = direct_transaction_df.copy()
    snapshot_transaction_df = snapshot_transaction_df.copy()
    # Order.counter is process-global, so consecutive identical runs receive
    # different absolute IDs. Preserve their gaps/grouping and all trade fields.
    for transaction_df in (direct_transaction_df, snapshot_transaction_df):
        if not transaction_df.empty:
            transaction_df["order_id"] = transaction_df["order_id"] - transaction_df["order_id"].iloc[0]
    pd.testing.assert_frame_equal(direct_transaction_df, snapshot_transaction_df, check_exact=True)


def run_qualification(output_path: Path, snapshot_date_str: str) -> dict[str, object]:
    output_path = output_path.resolve()
    output_path.mkdir(parents=True, exist_ok=False)
    prior_env_dict = {
        field_str: os.environ.get(field_str)
        for field_str in ("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "NORGATE_SNAPSHOT_ROOT")
    }
    report_dict: dict[str, object] = {
        "status_str": "running",
        "scope_str": "local_data_transport_and_engine_parity_only",
        "generated_at_utc_str": datetime.now(UTC).isoformat(),
        "snapshot_date_str": snapshot_date_str,
        "history_start_date_str": snapshot_module.CORE5_HISTORY_START_DATE_STR,
        "source_str": "current_local_Norgate_vintage_not_decision_time_replay",
        "new_strategy_variants_int": 0,
        "vps_verified_bool": False,
        "broker_verified_bool": False,
    }
    try:
        os.environ["ALPHA_USE_NORGATE_SNAPSHOT_BOOL"] = "false"
        config_obj = replace(core5_module.DEFAULT_CONFIG, end_date_str=snapshot_date_str)
        direct_price_df = core5_module.get_adaptive_macro_core5_data(config_obj)
        snapshot_root_path = output_path / "snapshots"
        snapshot_path = export_profile_snapshot(
            snapshot_root_str=str(snapshot_root_path), profile_str=snapshot_module.CORE5_PROFILE_STR,
            snapshot_date_str=snapshot_date_str,
        )
        os.environ["ALPHA_USE_NORGATE_SNAPSHOT_BOOL"] = "true"
        os.environ["NORGATE_SNAPSHOT_ROOT"] = str(snapshot_root_path)
        snapshot_price_df = core5_module.get_adaptive_macro_core5_data(config_obj)
        report_dict.update(compare_core5_frames(direct_price_df, snapshot_price_df))
        direct_price_df.to_parquet(output_path / "direct_prices.parquet")
        snapshot_price_df.to_parquet(output_path / "snapshot_prices.parquet")
        direct_strategy_obj = core5_module.run_variant(
            end_date_str=snapshot_date_str, pricing_data_df=direct_price_df,
            show_display_bool=False, save_results_bool=False,
        )
        snapshot_strategy_obj = core5_module.run_variant(
            end_date_str=snapshot_date_str, pricing_data_df=snapshot_price_df,
            show_display_bool=False, save_results_bool=False,
        )
        for field_str in ("daily_target_weights", "rebalance_target_weight_df", "borrow_fee_df", "results"):
            pd.testing.assert_frame_equal(
                getattr(direct_strategy_obj, field_str), getattr(snapshot_strategy_obj, field_str),
                check_exact=True, check_freq=False,
            )
        compare_core5_transactions(direct_strategy_obj.get_transactions(), snapshot_strategy_obj.get_transactions())
        direct_strategy_obj.rebalance_target_weight_df.to_csv(output_path / "verified_rebalance_targets.csv")
        report_dict.update({
            "status_str": "passed",
            "exact_targets_transactions_borrow_and_nav_parity_bool": True,
            "transaction_id_comparison_str": "Process-global order IDs rebased to first ID; every other field and relative ID gap compared exactly.",
            "decision_count_int": len(direct_strategy_obj.daily_target_weights),
            "rebalance_count_int": len(direct_strategy_obj.rebalance_target_weight_df),
            "transaction_count_int": len(direct_strategy_obj.get_transactions()),
            "diagnostic_capital_float": core5_module.DEFAULT_CONFIG.capital_base_float,
            "manifest_sha256_str": hashlib.sha256((snapshot_path / "manifest.json").read_bytes()).hexdigest(),
            "data_contract_dict": json.loads((snapshot_path / "manifest.json").read_text())["data_contract"],
            "limitations_list": [
                "Current-vintage transport parity is not historical decision-time replay or alpha validation.",
                "ALLMARKETDAYS historical padding preserved; only the export endpoint is additionally checked unpadded.",
                "Both engine paths retain the existing terminal-row month-end behavior; live adapter remains outstanding.",
                "Live state persistence, close-based sizing, DBC borrowing and account reconciliation remain outstanding.",
            ],
        })
        print(json.dumps({field_str: report_dict[field_str] for field_str in (
            "status_str", "price_rows_int", "decision_count_int", "rebalance_count_int", "transaction_count_int",
        )}, indent=2))
    except Exception as error_obj:
        report_dict.update({"status_str": "failed", "error_str": f"{type(error_obj).__name__}: {error_obj}"})
        raise
    finally:
        for field_str, prior_value_str in prior_env_dict.items():
            if prior_value_str is None:
                os.environ.pop(field_str, None)
            else:
                os.environ[field_str] = prior_value_str
        package_version_dict = {}
        for package_str in ("pandas", "numpy", "pyarrow", "norgatedata"):
            try:
                package_version_dict[package_str] = version(package_str)
            except PackageNotFoundError:
                package_version_dict[package_str] = "unavailable"
        report_dict["package_version_dict"] = package_version_dict
        source_path_list = [
            "strategies/taa_beyond_6040/strategy_taa_adaptive_macro_core5.py", "data/norgate_loader.py",
            "data/norgate_snapshot_store.py", "scripts/export_norgate_snapshot.py",
            "alpha/engine/strategy.py", "alpha/engine/backtest.py", "alpha/engine/execution_timing.py",
            "alpha/engine/backtester.py", "alpha/engine/order.py",
            "scripts/review/verify_core5_snapshot_parity.py",
        ]
        report_dict["source_sha256_dict"] = {
            source_str: hashlib.sha256((REPO_ROOT_PATH / source_str).read_bytes()).hexdigest()
            for source_str in source_path_list
        }
        report_dict["artifact_sha256_dict"] = {
            str(artifact_path.relative_to(output_path)): hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            for artifact_path in output_path.rglob("*") if artifact_path.is_file()
        }
        (output_path / "qualification.json").write_text(json.dumps(report_dict, indent=2, sort_keys=True), encoding="utf-8")
    return report_dict


def main() -> int:
    parser_obj = argparse.ArgumentParser(description=__doc__)
    parser_obj.add_argument("--output-dir", required=True, type=Path, help="New isolated directory; existing paths are rejected.")
    parser_obj.add_argument("--snapshot-date", required=True, help="Frozen completed market session, YYYY-MM-DD.")
    args_obj = parser_obj.parse_args()
    run_qualification(args_obj.output_dir, args_obj.snapshot_date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
