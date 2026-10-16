"""Repair missing audit trace only: actual HPI wrapper on canonical saved inputs.

No provider, broker or network calls. Require exact equity equality to the
already completed USD1M baseline before treating the new trace as qualified.
"""
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pandas as pd

REPO_PATH_OBJ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_PATH_OBJ))


def main():
    from strategies.hpi import stateful_long as hpi_module
    from strategies.hpi import strategy_mr_hpi_sp500_2_3_5_vote as wrapper_module
    from scripts.research.wired_signoff_20261015.mr_native_replay import install_decision_recorder, source_hash_dict
    from data import norgate_loader

    root_path_obj = REPO_PATH_OBJ / "results/review/wired_signoff_20261015"
    input_path_obj = root_path_obj / "input_preparation/strategy_mr_hpi_sp500_2_3_5_vote"
    baseline_path_obj = root_path_obj / "baselines/strategy_mr_hpi_sp500_2_3_5_vote"
    output_path_obj = root_path_obj / "mr_hpi235_trace_only"
    output_path_obj.mkdir(parents=True, exist_ok=False)
    pricing_df = pd.read_parquet(input_path_obj / "pricing.parquet")
    auxiliary_dict = pd.read_pickle(input_path_obj / "auxiliary.pkl.gz")
    universe_df = auxiliary_dict["universe_df"]
    initial_source_dict = source_hash_dict()
    report_dict = {"scope_str": __doc__, "status_str": "running", "new_strategy_variants_int": 0,
        "source_hash_dict": initial_source_dict,
        "checker_sha256_str": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "canonical_input_sha256_str": hashlib.sha256((input_path_obj / "pricing.parquet").read_bytes()).hexdigest()}
    report_path_obj = output_path_obj / "qualification.json"
    report_path_obj.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
    original_run_daily_fn = hpi_module.run_daily

    def cached_exact_loader(**request_dict):
        expected_dict = {"indexname_str": "S&P 500", "benchmark_symbol_str": "$SPXTR", "start_date_str": "1998-01-01", "end_date_str": "2026-10-14"}
        if request_dict != expected_dict:
            raise AssertionError(f"Unexpected canonical loader signature: {request_dict!r}")
        return list(universe_df.columns), universe_df.copy(), pricing_df.copy()

    def recorded_run_daily(strategy_obj, price_df, *argument_tuple, **keyword_dict):
        trace_dict = install_decision_recorder(strategy_obj)
        recorded_iterate_fn = strategy_obj.iterate

        def with_execution_date(*iterate_argument_tuple, **iterate_keyword_dict):
            result_obj = recorded_iterate_fn(*iterate_argument_tuple, **iterate_keyword_dict)
            trace_dict[pd.Timestamp(strategy_obj.previous_bar)]["execution_date"] = strategy_obj.current_bar
            return result_obj

        strategy_obj.iterate = with_execution_date
        price_df.to_parquet(output_path_obj / "pricing.parquet", compression="zstd")
        pd.to_pickle(auxiliary_dict, output_path_obj / "auxiliary.pkl.gz", compression="gzip")
        try:
            return original_run_daily_fn(strategy_obj, price_df, *argument_tuple, **keyword_dict)
        finally:
            pd.to_pickle(trace_dict, output_path_obj / "trace.pkl.gz", compression="gzip")
            strategy_obj.iterate = recorded_iterate_fn

    with ExitStack() as stack_obj:
        stack_obj.enter_context(patch.object(hpi_module, "load_exact_hpi_inputs", side_effect=cached_exact_loader))
        stack_obj.enter_context(patch.object(hpi_module, "run_daily", side_effect=recorded_run_daily))
        stack_obj.enter_context(patch.object(norgate_loader, "_load_direct_norgate_module", side_effect=AssertionError("Provider IO forbidden in cached HPI trace repair.")))
        with (output_path_obj / "engine.log").open("w", encoding="utf-8") as log_obj, redirect_stdout(log_obj), redirect_stderr(log_obj):
            strategy_obj = wrapper_module.run_variant(show_display_bool=False, save_results_bool=False,
                capital_base_float=1_000_000.0, end_date_str="2026-10-14")
    reference_df = pd.read_parquet(baseline_path_obj / "results.parquet")
    pd.testing.assert_series_equal(strategy_obj.results["total_value"], reference_df["total_value"], check_exact=True, check_names=False)
    strategy_obj.results.to_parquet(output_path_obj / "results.parquet", compression="zstd")
    report_dict.update({"status_str": "pass", "exact_original_baseline_equity_bool": True,
        "source_unchanged_bool": initial_source_dict == source_hash_dict(), "trace_session_count_int": len(pd.read_pickle(output_path_obj / "trace.pkl.gz"))})
    report_path_obj.write_text(json.dumps(report_dict, indent=2, default=str), encoding="utf-8")
    print(json.dumps({key_str: value_obj for key_str, value_obj in report_dict.items() if key_str != "source_hash_dict"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
