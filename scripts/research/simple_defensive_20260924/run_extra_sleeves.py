"""Run the sleeves the simple-defensive study needs that the fund-menu sources do not have (2026-09-24).

  btal_lin_spy   : strategies.taa_df.strategy_taa_df_btal_linearity_1n_fallback_spy_vix_cash (RESEARCH tier),
                   the SPY-fallback twin of the WIRED BTAL linearity QQQ sleeve.
  nobtal_lin_spy : its no-BTAL counterpart, built exactly like strategy_taa_df_linearity_1n_fallback_qqq_vix_cash
                   with SPY as the fallback. Research-only 2008 proxy, no strategy module is added.
  nobtal_lin_qqq_check : the no-BTAL QQQ config rebuilt the same way, to prove this runner reproduces the
                   fund-menu taa_lin_qqq path before the SPY proxy is trusted.
Same settings as the fund-menu sources: $1M reference capital, requested start 2000-01-03 (falls back to the
variant's own start when data do not reach it), common end 2026-08-19. One process per sleeve.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

REPO_ROOT_PATH = Path(__file__).resolve().parents[3]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

import pandas as pd  # noqa: E402

STUDY_DIR_PATH = REPO_ROOT_PATH / "results" / "research" / "portfolio" / "simple_defensive_20260924"
SOURCE_DIR_PATH = STUDY_DIR_PATH / "sources"
REFERENCE_CAPITAL_FLOAT = 1_000_000.0
REQUESTED_START_DATE_STR = "2000-01-03"
COMMON_END_DATE_STR = "2026-08-19"
NO_BTAL_DEFENSIVE_ASSET_TUPLE = ("GLD", "UUP", "TLT", "DBC")
NO_BTAL_1N_RANK_WEIGHT_TUPLE = (0.25, 0.25, 0.25, 0.25)


def build_config(alias_str: str):
    from strategies.taa_df.strategy_taa_df import DEFAULT_CONFIG as TAA_BASE_CONFIG
    from strategies.taa_df.strategy_taa_df_fallback_variant_utils import build_fallback_variant_config
    from strategies.taa_df.strategy_taa_df_fallback_vix_cash_variant_utils import build_vix_cash_variant_config

    if alias_str == "btal_lin_spy":
        from strategies.taa_df.strategy_taa_df_btal_linearity_1n_fallback_spy_vix_cash import DEFAULT_CONFIG as config_obj
    else:
        fallback_str = "SPY" if alias_str == "nobtal_lin_spy" else "QQQ"
        no_btal_config_obj = replace(TAA_BASE_CONFIG, defensive_asset_list=NO_BTAL_DEFENSIVE_ASSET_TUPLE,
                                     rank_weight_vec=NO_BTAL_1N_RANK_WEIGHT_TUPLE)
        config_obj = build_vix_cash_variant_config(build_fallback_variant_config(no_btal_config_obj, fallback_str))
    return replace(config_obj, end_date_str=COMMON_END_DATE_STR)


def run_one(alias_str: str) -> dict:
    from scripts.research import run_ladder4_candidate_value_add_study as ladder_runner
    from strategies.taa_df.strategy_taa_df_btal_linearity_1n import get_defense_first_linearity_1n_data
    from strategies.taa_df.strategy_taa_df_fallback_vix_cash_variant_utils import run_linearity_1n_fallback_vix_cash_variant

    started_float = time.time()
    config_obj = build_config(alias_str)
    call_kwargs_dict = dict(strategy_name_str=alias_str, config=config_obj, base_data_loader_fn=get_defense_first_linearity_1n_data,
                            show_display_bool=False, save_results_bool=False, output_dir_str=str(STUDY_DIR_PATH / "scratch_unused"),
                            capital_base_float=REFERENCE_CAPITAL_FLOAT)
    start_note_str = ""
    try:
        strategy_obj = run_linearity_1n_fallback_vix_cash_variant(backtest_start_date_str=REQUESTED_START_DATE_STR, **call_kwargs_dict)
        start_used_str = REQUESTED_START_DATE_STR
    except Exception as first_exc:  # data may not reach the requested start
        strategy_obj = run_linearity_1n_fallback_vix_cash_variant(backtest_start_date_str=None, **call_kwargs_dict)
        start_used_str, start_note_str = "variant default", f"requested {REQUESTED_START_DATE_STR} failed: {first_exc!r}"
    result_df = ladder_runner.extract_source_result_df(strategy_obj)
    transaction_df = ladder_runner.extract_source_transaction_df(strategy_obj, alias_str)
    if result_df.index[-1] != pd.Timestamp(COMMON_END_DATE_STR):
        raise RuntimeError(f"{alias_str} ended {result_df.index[-1].date()}, not {COMMON_END_DATE_STR}.")
    invested_mask_ser = result_df["portfolio_value_float"].abs() > 1e-9
    SOURCE_DIR_PATH.mkdir(parents=True, exist_ok=True)
    ladder_runner.write_csv_gzip(result_df, SOURCE_DIR_PATH / f"{alias_str}__path.csv.gz", index_bool=True, index_label_str="date")
    ladder_runner.write_csv_gzip(transaction_df, SOURCE_DIR_PATH / f"{alias_str}__transactions.csv.gz", index_bool=False)
    metadata_dict = {"alias_str": alias_str, "defensive_asset_list": list(config_obj.defensive_asset_list),
                     "requested_start_date_str": start_used_str, "start_note_str": start_note_str,
                     "first_invested_date_str": invested_mask_ser[invested_mask_ser].index[0].date().isoformat(),
                     "end_date_str": result_df.index[-1].date().isoformat(), "transaction_count_int": int(len(transaction_df)),
                     "runtime_seconds_float": round(time.time() - started_float, 1)}
    (SOURCE_DIR_PATH / f"{alias_str}__metadata.json").write_text(json.dumps(metadata_dict, indent=2), encoding="utf-8")
    return metadata_dict


def main() -> int:
    alias_list = ["btal_lin_spy", "nobtal_lin_spy", "nobtal_lin_qqq_check"]
    with ProcessPoolExecutor(max_workers=3, max_tasks_per_child=1) as executor_obj:
        future_dict = {executor_obj.submit(run_one, a): a for a in alias_list}
        for future_obj in as_completed(future_dict):
            print(json.dumps(future_obj.result()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
