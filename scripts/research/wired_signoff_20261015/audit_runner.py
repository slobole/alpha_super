"""Isolated, broker-free evidence capture for the owner's wired sign-off audit.

No strategy, release, database or production source is changed. Native inputs
are read once per exact loader request; callers get private copies. FRED uses a
frozen local CSV and cannot refresh the owner's shared cache or use the network.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import gzip
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import pickle
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT_PATH))
OUTPUT_ROOT_PATH = REPO_ROOT_PATH / "results/review/wired_signoff_20261015"


class AuditInputsPrepared(Exception):
    """Stop a wrapper at its engine boundary after capturing real inputs."""


class AuditCacheMiss(BaseException):
    """Must bypass production helpers that intentionally catch Exception."""


def write_json(output_path: Path, payload_obj: object) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload_obj, indent=2, default=str, allow_nan=False), encoding="utf-8")


def file_hash_str(source_path: Path) -> str:
    digest_obj = hashlib.sha256()
    with source_path.open("rb") as source_file:
        for chunk_bytes in iter(lambda: source_file.read(1024 * 1024), b""):
            digest_obj.update(chunk_bytes)
    return digest_obj.hexdigest()


def frame_hash_str(value_obj: pd.DataFrame | pd.Series) -> str:
    digest_obj = hashlib.sha256(pd.util.hash_pandas_object(value_obj, index=True).to_numpy().tobytes())
    column_list = list(value_obj.columns) if isinstance(value_obj, pd.DataFrame) else [value_obj.name]
    dtype_list = list(value_obj.dtypes.astype(str)) if isinstance(value_obj, pd.DataFrame) else [str(value_obj.dtype)]
    digest_obj.update(json.dumps(column_list, default=str).encode())
    digest_obj.update(json.dumps(dtype_list).encode())
    return digest_obj.hexdigest()


def save_pickle(output_path: Path, payload_obj: object) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wb", compresslevel=2) as output_file:
        pickle.dump(payload_obj, output_file, protocol=pickle.HIGHEST_PROTOCOL)


def build_saved_hpi_loader(output_root_path: Path, end_date_str: str, evidence_list: list[dict]):
    """Replay the exact post-loader frame captured from the canonical HPI parent.

    HPI's direct no-padding native loader bypasses the shared loader cache. The
    preparation pass already ran it unchanged. All four HPI identities request
    this same stock/membership input; capsules append parking prices afterwards.
    """
    prepared_path = output_root_path / "input_preparation/strategy_mr_hpi_sp500_2_3_5_vote"
    prepared_dict = json.loads((prepared_path / "summary.json").read_text(encoding="utf-8"))
    if prepared_dict.get("status") != "inputs_prepared_not_run" or prepared_dict.get("end_requested") != end_date_str:
        raise AuditCacheMiss("Exact HPI input preparation is absent or has another endpoint.")
    source_path_list = [prepared_path / "pricing.parquet", prepared_path / "auxiliary.pkl.gz"]
    source_hash_dict = {str(source_path): file_hash_str(source_path) for source_path in source_path_list}
    pricing_df = pd.read_parquet(source_path_list[0])
    with gzip.open(source_path_list[1], "rb") as handle_obj:
        universe_df = pickle.load(handle_obj)["universe_df"]
    if not isinstance(universe_df, pd.DataFrame) or not pricing_df.index.is_monotonic_increasing:
        raise AuditCacheMiss("Exact HPI prepared data has invalid structure.")
    evidence_list.extend({"path": source_str, "sha256": hash_str, "boundary": "canonical load_exact_hpi_inputs output before engine"} for source_str, hash_str in source_hash_dict.items())

    def saved_hpi_loader(*, indexname_str, benchmark_symbol_str, start_date_str, end_date_str):
        request_tuple = (indexname_str, benchmark_symbol_str, start_date_str, end_date_str)
        if request_tuple != ("S&P 500", "$SPXTR", "1998-01-01", prepared_dict["end_requested"]):
            raise AuditCacheMiss(f"Unprepared exact HPI request: {request_tuple}")
        if any(file_hash_str(Path(source_str)) != hash_str for source_str, hash_str in source_hash_dict.items()):
            raise AuditCacheMiss("Exact HPI saved input changed after loading.")
        # *** CRITICAL *** No current-membership replacement or fresh forward
        # fill is introduced here. This is the entire frozen no-padding loader
        # output, including PIT universe and valuation-only Close padding.
        return universe_df.columns.astype(str).tolist(), universe_df.copy(deep=True), pricing_df.copy(deep=True)

    return saved_hpi_loader


def calculate_metrics(value_ser: pd.Series, initial_capital_float: float | None = None) -> dict:
    value_ser = value_ser.astype(float)
    if not value_ser.index.is_unique or not value_ser.index.is_monotonic_increasing:
        raise ValueError("Equity dates must be unique and chronological.")
    if len(value_ser) < 2 or not np.isfinite(value_ser.to_numpy()).all() or (value_ser <= 0).any():
        raise ValueError("Equity must contain at least two positive finite observations.")
    starting_value_float = float(value_ser.iloc[0]) if initial_capital_float is None else float(initial_capital_float)
    # *** CRITICAL *** Reporting only: returns use adjacent recorded closes;
    # the first return includes the first day's result relative to initial cash.
    return_ser = value_ser.pct_change(fill_method=None)
    return_ser.iloc[0] = float(value_ser.iloc[0]) / starting_value_float - 1.0
    year_float = (value_ser.index[-1] - value_ser.index[0]).days / 365.25
    volatility_float = float(return_ser.std(ddof=1) * np.sqrt(252))
    running_peak_vec = np.maximum.accumulate(np.r_[starting_value_float, value_ser.to_numpy()])[1:]
    drawdown_ser = value_ser / running_peak_vec - 1.0
    return {
        "start": str(value_ser.index[0].date()), "end": str(value_ser.index[-1].date()),
        "sessions": len(value_ser), "initial_capital": starting_value_float,
        "terminal_equity": float(value_ser.iloc[-1]),
        "cagr": float((value_ser.iloc[-1] / starting_value_float) ** (1 / year_float) - 1),
        "volatility": volatility_float,
        "sharpe_zero_rf": float(return_ser.mean() * 252 / volatility_float) if volatility_float else None,
        "max_drawdown": float(drawdown_ser.min()),
        "return_convention": "close-to-close, first observation versus supplied initial cash; 252 sessions; sample SD; RF=0; CAGR elapsed days/365.25",
    }


def freeze_checkout(output_root_path: Path) -> dict:
    from alpha.strategy_registry import wired_import_tuple, tier_label_for
    source_path_list = []
    for folder_str in ("alpha", "data", "strategies"):
        source_path_list.extend((REPO_ROOT_PATH / folder_str).rglob("*.py"))
    source_path_list.extend(REPO_ROOT_PATH / name_str for name_str in ("uv.lock", "pyproject.toml", "QUANT_PHILOSOPHY.md", "ASSUMPTIONS_AND_GAPS.md"))
    source_row_list = [{"path": str(source_path.relative_to(REPO_ROOT_PATH)).replace("\\", "/"), "bytes": source_path.stat().st_size, "sha256": file_hash_str(source_path)} for source_path in sorted(set(source_path_list))]
    package_version_dict = {}
    for package_str in ("pandas", "numpy", "norgatedata", "ib-async", "exchange-calendars", "pytest"):
        package_version_dict[package_str] = importlib.metadata.version(package_str)
    manifest_dict = {
        "captured_utc": datetime.now(UTC).isoformat(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT_PATH, text=True).strip(),
        "initial_git_status": subprocess.check_output(["git", "-c", "core.fsmonitor=false", "status", "--short"], cwd=REPO_ROOT_PATH, text=True),
        "capital_per_strategy": 1_000_000, "python": sys.version,
        "package_versions": package_version_dict,
        "strategies": [{"import": reference_str, "tier": tier_label_for(reference_str)} for reference_str in wired_import_tuple()],
        "source_files": source_row_list,
        "authority": "Offline research and tests only; no broker, scheduler, release changes, allocation or deployment.",
    }
    write_json(output_root_path / "frozen_manifest.json", manifest_dict)
    return manifest_dict


def run_baselines(output_root_path: Path, end_date_str: str, only_key_list: list[str] | None, check_repeat_bool: bool,
                  prepared_only_bool: bool = False, cached_input_path: Path | None = None) -> None:
    import alpha.data as alpha_data_module
    import alpha.data.fred_loader as fred_module
    import alpha.engine.backtest as backtest_module
    import data.norgate_loader as loader_module
    from alpha.strategy_registry import wired_import_tuple, module_import_str
    if loader_module.is_snapshot_mode_enabled_bool():
        raise RuntimeError("This audit runner requires explicit direct-native preparation; deployed snapshot contexts are separate evidence.")

    frozen_fred_path = output_root_path / "inputs/DTB3.csv"
    frozen_fred_path.parent.mkdir(parents=True, exist_ok=True)
    if not frozen_fred_path.exists():
        shutil.copy2(REPO_ROOT_PATH.parent / "1_data/DTB3.csv", frozen_fred_path)
    original_fred_fn = fred_module.load_daily_fred_series_snapshot
    def block_fred_network(*argument_tuple, **keyword_dict):
        raise OSError("Audit: network disabled; frozen CSV only.")
    def frozen_fred_fn(*argument_tuple, **keyword_dict):
        if argument_tuple:
            raise ValueError("Expected named FRED arguments in this audit.")
        keyword_dict["cache_csv_path_str"] = str(frozen_fred_path)
        return original_fred_fn(**keyword_dict)
    fred_module.urlopen = block_fred_network
    fred_module.load_daily_fred_series_snapshot = frozen_fred_fn
    alpha_data_module.load_daily_fred_series_snapshot = frozen_fred_fn

    if cached_input_path is None:
        input_cache_dict = {}
        initial_cache_hash_str = None
    else:
        initial_cache_hash_str = file_hash_str(cached_input_path)
        with gzip.open(cached_input_path, "rb") as input_file:
            input_cache_dict = pickle.load(input_file)
    initial_fred_hash_str = file_hash_str(frozen_fred_path)
    input_row_list = []
    supplemental_input_list = []
    def cache_loader_fn(original_fn, loader_name_str):
        def cached_fn(*argument_tuple, **keyword_dict):
            request_key_str = repr((loader_name_str, argument_tuple, sorted(keyword_dict.items())))
            if request_key_str not in input_cache_dict:
                if cached_input_path is not None:
                    raise AuditCacheMiss(f"Offline audit input not prepared: {request_key_str}")
                value_obj = original_fn(*argument_tuple, **keyword_dict)
                input_cache_dict[request_key_str] = value_obj
                frame_obj = value_obj[1] if isinstance(value_obj, tuple) else value_obj
                input_row_list.append({"request": request_key_str, "sha256": frame_hash_str(frame_obj), "rows": len(frame_obj), "first": str(frame_obj.index[0]) if len(frame_obj) else None, "last": str(frame_obj.index[-1]) if len(frame_obj) else None})
            return copy.deepcopy(input_cache_dict[request_key_str])
        return cached_fn
    loader_module.load_price_timeseries = cache_loader_fn(loader_module.load_price_timeseries, "price")
    loader_module.build_index_constituent_matrix = cache_loader_fn(loader_module.build_index_constituent_matrix, "membership")
    native_module = loader_module._load_direct_norgate_module()
    original_classification_fn = native_module.classification_at_level
    def cached_classification_fn(*argument_tuple, **keyword_dict):
        request_key_str = repr(("classification", argument_tuple, sorted(keyword_dict.items())))
        if request_key_str not in input_cache_dict:
            if cached_input_path is not None:
                raise AuditCacheMiss(f"Offline audit classification not prepared: {request_key_str}")
            input_cache_dict[request_key_str] = original_classification_fn(*argument_tuple, **keyword_dict)
        return input_cache_dict[request_key_str]
    native_module.classification_at_level = cached_classification_fn
    if cached_input_path is not None:
        if not only_key_list or any(key_str.startswith("strategy_mr_hpi") for key_str in only_key_list):
            import strategies.hpi.stateful_long as hpi_module
            hpi_module.load_exact_hpi_inputs = build_saved_hpi_loader(output_root_path, end_date_str, supplemental_input_list)
        def reject_native_io(*argument_tuple, **keyword_dict):
            raise AuditCacheMiss("Offline audit worker attempted unprepared native provider IO.")
        for attribute_str in ("price_timeseries", "index_constituent_timeseries", "watchlist_symbols", "database_symbols"):
            if hasattr(native_module, attribute_str):
                setattr(native_module, attribute_str, reject_native_io)

    capture_context_dict = {}
    original_run_daily_fn = backtest_module.run_daily
    def captured_run_daily_fn(*argument_tuple, **keyword_dict):
        strategy_obj = argument_tuple[0] if argument_tuple else keyword_dict["strategy"]
        pricing_df = argument_tuple[1] if len(argument_tuple) > 1 else keyword_dict["pricing_data"]
        if not capture_context_dict.get("capture", False):
            return original_run_daily_fn(*argument_tuple, **keyword_dict)
        from alpha.live.strategy_host import _extract_strategy_state_dict, _build_order_shape_dict
        artifact_path = capture_context_dict["path"]
        pricing_df.to_parquet(artifact_path / "pricing.parquet", compression="zstd")
        auxiliary_dict = {name_str: copy.deepcopy(getattr(strategy_obj, name_str)) for name_str in ("universe_df", "vix_close_ser") if getattr(strategy_obj, name_str, None) is not None}
        save_pickle(artifact_path / "auxiliary.pkl.gz", auxiliary_dict)
        if prepared_only_bool:
            raise AuditInputsPrepared("Canonical wrapper inputs captured before engine; no backtest result claimed.")
        trace_row_list = []
        original_iterate_fn = strategy_obj.iterate
        def captured_iterate_fn(*iterate_argument_tuple, **iterate_keyword_dict):
            # *** CRITICAL *** Account snapshot precedes T+1 fills/dividends.
            # previous_bar is the completed decision session T, not current_bar.
            trace_dict = {
                "decision_date": strategy_obj.previous_bar,
                "execution_date": strategy_obj.current_bar,
                "positions": {str(asset_str): float(amount_float) for asset_str, amount_float in strategy_obj.get_positions().items() if amount_float != 0},
                "cash": float(strategy_obj.cash), "nav": float(strategy_obj.previous_total_value),
                "state": copy.deepcopy(_extract_strategy_state_dict(strategy_obj)),
            }
            result_obj = original_iterate_fn(*iterate_argument_tuple, **iterate_keyword_dict)
            trace_dict["orders"] = [_build_order_shape_dict(order_obj) for order_obj in strategy_obj.get_orders()]
            trace_dict["after_state"] = copy.deepcopy(_extract_strategy_state_dict(strategy_obj))
            trace_row_list.append(trace_dict)
            return result_obj
        strategy_obj.iterate = captured_iterate_fn
        try:
            result_obj = original_run_daily_fn(*argument_tuple, **keyword_dict)
        finally:
            strategy_obj.iterate = original_iterate_fn
            save_pickle(artifact_path / "trace.pkl.gz", trace_row_list)
        capture_context_dict["pricing_hash"] = frame_hash_str(pricing_df)
        return result_obj
    backtest_module.run_daily = captured_run_daily_fn
    # The saved HPI loader is installed before strategy imports. Its module
    # binds run_daily at import time, so redirect that existing binding too.
    preloaded_hpi_module = sys.modules.get("strategies.hpi.stateful_long")
    if preloaded_hpi_module is not None and preloaded_hpi_module.run_daily is original_run_daily_fn:
        preloaded_hpi_module.run_daily = captured_run_daily_fn

    run_summary_list = []
    for reference_str in wired_import_tuple():
        module_name_str = module_import_str(reference_str)
        key_str = module_name_str.rsplit(".", 1)[-1]
        if only_key_list and key_str not in only_key_list:
            continue
        artifact_path = output_root_path / ("input_preparation" if prepared_only_bool else "baselines") / key_str
        artifact_path.mkdir(parents=True, exist_ok=True)
        print(f"BASELINE_START {key_str}", flush=True)
        started_float = time.monotonic()
        summary_dict = {"key": key_str, "import": reference_str, "end_requested": end_date_str, "capital": 1_000_000}
        try:
            strategy_module = importlib.import_module(module_name_str)
            run_fn = strategy_module.run_variant
            parameter_map_dict = inspect.signature(run_fn).parameters
            run_keyword_dict = {"show_display_bool": False, "save_results_bool": False, "output_dir_str": str(artifact_path), "capital_base_float": 1_000_000.0, "end_date_str": end_date_str}
            run_keyword_dict = {name_str: value_obj for name_str, value_obj in run_keyword_dict.items() if name_str in parameter_map_dict}
            capture_context_dict.update(path=artifact_path, capture=True)
            with (artifact_path / "run.log").open("w", encoding="utf-8") as log_file, contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                strategy_obj = run_fn(**run_keyword_dict)
            capture_context_dict["capture"] = False
            results_df = strategy_obj.results.copy()
            results_df.to_parquet(artifact_path / "results.parquet", compression="zstd")
            transaction_df = strategy_obj.get_transactions().copy()
            transaction_df.to_csv(artifact_path / "transactions.csv.gz", index=False)
            equity_ser = results_df["total_value"].astype(float)
            summary_dict.update(status="pass", metrics=calculate_metrics(equity_ser, 1_000_000), pricing_hash=capture_context_dict.get("pricing_hash"), transaction_count=len(transaction_df), benchmark_labels=list(getattr(strategy_obj, "_benchmarks", [])), accounting_policy=getattr(strategy_obj, "_accounting_policy_dict", {}))
            if hasattr(strategy_obj, "summary") and strategy_obj.summary is not None:
                strategy_obj.summary.to_csv(artifact_path / "native_summary.csv")
            config_obj = getattr(strategy_module, "DEFAULT_CONFIG", None)
            summary_dict["module_default_config"] = asdict(config_obj) if is_dataclass(config_obj) else str(config_obj)
            summary_dict["actual_run_kwargs"] = run_keyword_dict
            summary_dict["actual_capital_base"] = float(strategy_obj._capital_base)
            summary_dict["subperiods"] = {}
            for label_str, start_str in (("since2016", "2016-01-01"), ("last3y", "2023-10-15"), ("covid", "2020-01-01"), ("rates2022", "2022-01-01")):
                end_str = "2020-12-31" if label_str == "covid" else "2022-12-31" if label_str == "rates2022" else end_date_str
                period_ser = equity_ser.loc[start_str:end_str]
                if len(period_ser) > 1:
                    summary_dict["subperiods"][label_str] = calculate_metrics(period_ser)
            if "cash" in results_df:
                summary_dict["min_cash_fraction"] = float((results_df["cash"].astype(float) / equity_ser).min())
                summary_dict["negative_cash_days"] = int((results_df["cash"].astype(float) < -1e-6).sum())
            summary_dict["commission_total"] = float(pd.to_numeric(transaction_df.get("commission", pd.Series(dtype=float))).sum())
            write_json(artifact_path / "summary.json", summary_dict)
            if check_repeat_bool:
                with (artifact_path / "repeat.log").open("w", encoding="utf-8") as log_file, contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                    repeat_obj = run_fn(**run_keyword_dict)
                    repeat_ser = repeat_obj.results["total_value"].astype(float)
                    summary_dict["deterministic_equity"] = bool(equity_ser.equals(repeat_ser))
                    economic_column_list = [name_str for name_str in ("bar", "asset", "amount", "price", "commission") if name_str in transaction_df.columns]
                    summary_dict["deterministic_economic_transactions"] = bool(transaction_df[economic_column_list].reset_index(drop=True).equals(repeat_obj.get_transactions()[economic_column_list].reset_index(drop=True)))
                    double_obj = run_fn(**{**run_keyword_dict, "capital_base_float": 2_000_000.0})
                    double_ser = double_obj.results["total_value"].astype(float)
                    summary_dict["double_capital_ratio"] = float(double_ser.iloc[-1] / equity_ser.iloc[-1])
                    double_obj.results.to_parquet(artifact_path / "double_results.parquet", compression="zstd")
                    summary_dict["capital_honored"] = bool(1.8 < summary_dict["double_capital_ratio"] < 2.2)
                    summary_dict["actual_double_capital_base"] = float(double_obj._capital_base)
                    del repeat_obj, double_obj
        except AuditCacheMiss as exception_obj:
            summary_dict.update(status="error", error=str(exception_obj), traceback=traceback.format_exc())
            print(summary_dict["traceback"], flush=True)
        except AuditInputsPrepared:
            summary_dict["status"] = "inputs_prepared_not_run"
        except Exception as exception_obj:
            summary_dict.update(status="error", error=str(exception_obj), traceback=traceback.format_exc())
            print(summary_dict["traceback"], flush=True)
        finally:
            capture_context_dict["capture"] = False
            if any(file_hash_str(Path(row_dict["path"])) != row_dict["sha256"] for row_dict in supplemental_input_list):
                summary_dict.update(status="error", error="Frozen supplemental HPI input changed during execution.")
            if file_hash_str(frozen_fred_path) != initial_fred_hash_str or (cached_input_path is not None and file_hash_str(cached_input_path) != initial_cache_hash_str):
                summary_dict.update(status="error", error="Frozen audit cache/FRED changed during execution.")
            summary_dict["seconds"] = time.monotonic() - started_float
            write_json(artifact_path / "summary.json", summary_dict)
            run_summary_list.append(summary_dict)
            if cached_input_path is None and not prepared_only_bool:
                write_json(output_root_path / "baseline_summary.json", run_summary_list)
            write_json(artifact_path / "input_manifest.json", {"fred": {"path": str(frozen_fred_path), "sha256": file_hash_str(frozen_fred_path)}, "native_reads": input_row_list,
                "supplemental_saved_inputs": supplemental_input_list,
                "offline_cache": None if cached_input_path is None else {"path": str(cached_input_path), "sha256": initial_cache_hash_str}, "unchanged_at_completion": summary_dict.get("error") != "Frozen audit cache/FRED changed during execution."})
            print(f"BASELINE_END {key_str} {summary_dict['status']} {summary_dict['seconds']:.1f}s", flush=True)
    if prepared_only_bool:
        save_pickle(output_root_path / "inputs/native_loader_cache.pkl.gz", input_cache_dict)
        write_json(output_root_path / "input_manifest.json", {"fred": {"path": str(frozen_fred_path), "sha256": file_hash_str(frozen_fred_path)}, "native_reads": input_row_list,
            "cache_sha256": file_hash_str(output_root_path / "inputs/native_loader_cache.pkl.gz")})


def main() -> None:
    parser_obj = argparse.ArgumentParser(description=__doc__)
    parser_obj.add_argument("action", choices=("freeze", "baselines", "prepare"))
    parser_obj.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_PATH)
    parser_obj.add_argument("--end-date", default="2026-10-14")
    parser_obj.add_argument("--only", nargs="*")
    parser_obj.add_argument("--check-repeat", action="store_true")
    parser_obj.add_argument("--cached-inputs", type=Path)
    argument_obj = parser_obj.parse_args()
    if argument_obj.action == "freeze":
        manifest_dict = freeze_checkout(argument_obj.output_root)
        print(f"Frozen {len(manifest_dict['strategies'])} strategies at {manifest_dict['head']}")
    else:
        run_baselines(argument_obj.output_root, argument_obj.end_date, argument_obj.only, argument_obj.check_repeat,
            argument_obj.action == "prepare", argument_obj.cached_inputs)


if __name__ == "__main__":
    main()
