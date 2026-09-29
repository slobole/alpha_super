"""Run the TAA variants needed to isolate BTAL and leverage (owner question, 2026-09-24).

  nobtal_rank_tqqq     : strategies.taa_df.strategy_taa_df_fallback_tqqq_vix_cash       (rank weights, no BTAL, 3x)
  nobtal_1n_tqqq       : strategies.taa_df.strategy_taa_df_1n_fallback_tqqq_vix_cash    (equal slots, no BTAL, 3x)
  btal_rank_qqq        : strategies.taa_df.strategy_taa_df_btal_fallback_qqq_vix_cash   (rank weights, BTAL, 1x)
  nobtal_rank_qqq      : strategies.taa_df.strategy_taa_df_fallback_qqq_vix_cash        (rank weights, no BTAL, 1x)
  btal_rank_tqqq_check : strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash  (the WIRED sleeve, rerun to
                         prove this runner reproduces the fund-menu source taa_btal_tqqq)
Same settings as the fund-menu sources: $1M, requested start 2000-01-03 (falls back to the variant's own start),
end 2026-08-19. Two modules predate the start/capital/end arguments, so they are called through the shared
run_standard_fallback_vix_cash_variant with their own DEFAULT_CONFIG. One process per sleeve.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import importlib
import json
from pathlib import Path
import sys
import time

REPO_ROOT_PATH = Path(__file__).resolve().parents[3]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

import pandas as pd  # noqa: E402

STUDY_DIR_PATH = REPO_ROOT_PATH / "results" / "research" / "portfolio" / "btal_leverage_20260924"
SOURCE_DIR_PATH = STUDY_DIR_PATH / "sources"
REFERENCE_CAPITAL_FLOAT = 1_000_000.0
REQUESTED_START_DATE_STR = "2000-01-03"
COMMON_END_DATE_STR = "2026-08-19"
JOB_DICT = {
    "nobtal_rank_tqqq": ("strategies.taa_df.strategy_taa_df_fallback_tqqq_vix_cash", "module"),
    "nobtal_1n_tqqq": ("strategies.taa_df.strategy_taa_df_1n_fallback_tqqq_vix_cash", "module"),
    "btal_rank_qqq": ("strategies.taa_df.strategy_taa_df_btal_fallback_qqq_vix_cash", "standard"),
    "nobtal_rank_qqq": ("strategies.taa_df.strategy_taa_df_fallback_qqq_vix_cash", "standard"),
    "btal_rank_tqqq_check": ("strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash", "module"),
}


def run_strategy(module_str: str, mode_str: str, start_obj):
    module_obj = importlib.import_module(module_str)
    common_kwargs_dict = dict(show_display_bool=False, save_results_bool=False, output_dir_str=str(STUDY_DIR_PATH / "scratch_unused"),
                              backtest_start_date_str=start_obj, capital_base_float=REFERENCE_CAPITAL_FLOAT)
    if mode_str == "module":
        return module_obj.run_variant(end_date_str=COMMON_END_DATE_STR, **common_kwargs_dict)
    from strategies.taa_df.strategy_taa_df import get_defense_first_data
    from strategies.taa_df.strategy_taa_df_fallback_vix_cash_variant_utils import run_standard_fallback_vix_cash_variant
    return run_standard_fallback_vix_cash_variant(strategy_name_str=module_str.rsplit(".", 1)[1],
                                                  config=replace(module_obj.DEFAULT_CONFIG, end_date_str=COMMON_END_DATE_STR),
                                                  base_data_loader_fn=get_defense_first_data, **common_kwargs_dict)


def run_one(alias_str: str) -> dict:
    from scripts.research import run_ladder4_candidate_value_add_study as ladder_runner

    started_float = time.time()
    module_str, mode_str = JOB_DICT[alias_str]
    start_note_str = ""
    try:
        strategy_obj = run_strategy(module_str, mode_str, REQUESTED_START_DATE_STR)
    except Exception as first_exc:  # data may not reach the requested start
        strategy_obj = run_strategy(module_str, mode_str, None)
        start_note_str = f"requested {REQUESTED_START_DATE_STR} failed: {first_exc!r}"
    result_df = ladder_runner.extract_source_result_df(strategy_obj)
    transaction_df = ladder_runner.extract_source_transaction_df(strategy_obj, alias_str)
    if result_df.index[-1] != pd.Timestamp(COMMON_END_DATE_STR):
        raise RuntimeError(f"{alias_str} ended {result_df.index[-1].date()}, not {COMMON_END_DATE_STR}.")
    invested_mask_ser = result_df["portfolio_value_float"].abs() > 1e-9
    SOURCE_DIR_PATH.mkdir(parents=True, exist_ok=True)
    ladder_runner.write_csv_gzip(result_df, SOURCE_DIR_PATH / f"{alias_str}__path.csv.gz", index_bool=True, index_label_str="date")
    ladder_runner.write_csv_gzip(transaction_df, SOURCE_DIR_PATH / f"{alias_str}__transactions.csv.gz", index_bool=False)
    metadata_dict = {"alias_str": alias_str, "strategy_module_str": module_str, "call_mode_str": mode_str, "start_note_str": start_note_str,
                     "first_invested_date_str": invested_mask_ser[invested_mask_ser].index[0].date().isoformat(),
                     "end_date_str": result_df.index[-1].date().isoformat(), "transaction_count_int": int(len(transaction_df)),
                     "runtime_seconds_float": round(time.time() - started_float, 1)}
    (SOURCE_DIR_PATH / f"{alias_str}__metadata.json").write_text(json.dumps(metadata_dict, indent=2), encoding="utf-8")
    return metadata_dict


def main() -> int:
    with ProcessPoolExecutor(max_workers=5, max_tasks_per_child=1) as executor_obj:
        future_dict = {executor_obj.submit(run_one, a): a for a in JOB_DICT}
        for future_obj in as_completed(future_dict):
            print(json.dumps(future_obj.result()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
