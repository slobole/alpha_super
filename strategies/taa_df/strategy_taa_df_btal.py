"""
Defense First tactical allocation with BTAL added to the defensive sleeve and
SPY as the fallback sleeve.

This variant preserves the exact signal and execution logic from
`strategy_taa_df.py` and changes only the configuration:

    defensive_asset_list = ("GLD", "UUP", "TLT", "DBC", "BTAL")
    fallback_asset = "SPY"

The defensive rank weights switch from fixed four-slot weights to five-slot
rank weights:

    rank_score_vec = [5, 4, 3, 2, 1]
    rank_weight_vec = rank_score_vec / sum(rank_score_vec)
                    = [5, 4, 3, 2, 1] / 15

The momentum formula is unchanged:

    momentum_score_{i,t}
        = (r_{1m,i,t} + r_{3m,i,t} + r_{6m,i,t} + r_{12m,i,t}) / 4

    r_{k,i,t} = close_{i,t} / close_{i,t-k} - 1

Quantitative consequence:

    start_date_str = max(first_BTAL_date, first_SPY_date) = first_BTAL_date

Because BTAL starts later than DBC and SPY, the evaluation window is
intentionally clipped to the BTAL inception date to avoid requesting
pre-inception signal history or fills.
"""

from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path

repo_root_path = Path(__file__).resolve().parents[2]
repo_root_str = str(repo_root_path)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

try:
    from strategies.taa_df.strategy_taa_df import (
        DefenseFirstConfig,
        get_defense_first_data,
        run_defense_first_variant,
    )
except ModuleNotFoundError:
    from strategy_taa_df import (
        DefenseFirstConfig,
        get_defense_first_data,
        run_defense_first_variant,
    )


STRATEGY_NAME_STR = "strategy_taa_df_btal"
btal_inception_date_str = "2011-09-13"
effective_start_date_str = btal_inception_date_str
btal_defensive_asset_list = ("GLD", "UUP", "TLT", "DBC", "BTAL")
btal_rank_weight_vec = (
    5.0 / 15.0,
    4.0 / 15.0,
    3.0 / 15.0,
    2.0 / 15.0,
    1.0 / 15.0,
)

DEFAULT_CONFIG = DefenseFirstConfig(
    defensive_asset_list=btal_defensive_asset_list,
    fallback_asset="SPY",
    rank_weight_vec=btal_rank_weight_vec,
    start_date_str=effective_start_date_str,
)


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
):
    config = DEFAULT_CONFIG if end_date_str is None else replace(DEFAULT_CONFIG, end_date_str=end_date_str)
    return run_defense_first_variant(
        strategy_name_str=STRATEGY_NAME_STR,
        config=config,
        data_loader_fn=get_defense_first_data,
        show_display_bool=show_display_bool,
        save_results_bool=save_results_bool,
        output_dir_str=output_dir_str,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
    )


if __name__ == "__main__":
    run_variant()
