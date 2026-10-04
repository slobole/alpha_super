"""
NATR20 Nasdaq-100 momentum with monthly VXN scaling and a 40% GICS-sector cap.

The NATR20 book of the NDX design of record "E2 + 40% sector cap" (Scout amendment A15, 2026-10-04); the
dollar-ATR book is strategy_mo_atr_normalized_ndx_vxn_scaled_sector_cap.py and the pair is held 50/50 in
portfolios/ndx_e2_sector_cap_5050.yaml. Research / PM only; nothing here is wired to a live account.

Everything is strategy_mo_natr20_ndx_vxn_scaled except the top-N walk:

    ranked candidates by score_{i,t} = ROC12_{i,t} / (ATR20_{i,t} / UnadjustedClose_{i,t})   (unchanged)

    accept candidate i iff count(sector(i) in selected) < sector_cap      (sector_cap = 4 of 10 names = 40%)
    stop when max_positions are accepted or the candidates run out

    target_weight_{i,t} = (1 / max_positions) * clip(22 / VXN_t, 0.25, 1.0)

The cap mechanics, the current-label GICS realism gap and the cost model are those of the dollar-ATR sibling
(see its docstring and ASSUMPTIONS_AND_GAPS.md). History through 2026 has been examined; not an untouched holdout.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from IPython.display import display

from alpha.engine.report import save_results
from strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled_sector_cap import (
    SectorCapSelectionMixin,
    build_strategy,
    configured_config,
    run_built_strategy,
    sector_map_for_universe,
    timing_inputs,
)
from strategies.momentum.strategy_mo_natr20_ndx_vxn_scaled import (
    Natr20VxnScaledNdxConfig,
    Natr20VxnScaledNdxStrategy,
    get_natr20_vxn_scaled_ndx_data,
)


STRATEGY_NAME_STR = "strategy_mo_natr20_ndx_vxn_scaled_sector_cap"


@dataclass(frozen=True)
class SectorCapNatr20VxnScaledNdxConfig(Natr20VxnScaledNdxConfig):
    sector_cap_int: int = 4

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.sector_cap_int <= 0:
            raise ValueError("sector_cap_int must be positive.")


DEFAULT_CONFIG = SectorCapNatr20VxnScaledNdxConfig()

__all__ = [
    "DEFAULT_CONFIG",
    "SectorCapNatr20VxnScaledNdxConfig",
    "SectorCapNatr20VxnScaledNdxStrategy",
    "build_capacity_analysis_inputs",
    "build_execution_timing_analysis_inputs",
    "run_variant",
]


class SectorCapNatr20VxnScaledNdxStrategy(SectorCapSelectionMixin, Natr20VxnScaledNdxStrategy):
    """NATR20 NDX VXN momentum with a hard GICS-sector cap on selection."""


def _prepare(config_obj):
    pricing_data_df, universe_df, rebalance_schedule_df, vxn_scale_signal_df = get_natr20_vxn_scaled_ndx_data(
        config_obj, include_total_return_benchmark_bool=True
    )
    strategy_obj = build_strategy(
        SectorCapNatr20VxnScaledNdxStrategy, STRATEGY_NAME_STR, config_obj, rebalance_schedule_df,
        vxn_scale_signal_df, universe_df, sector_map_for_universe(universe_df),
    )
    return strategy_obj, pricing_data_df


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
) -> SectorCapNatr20VxnScaledNdxStrategy:
    config_obj = configured_config(DEFAULT_CONFIG, backtest_start_date_str, capital_base_float, end_date_str)
    strategy_obj, pricing_data_df = _prepare(config_obj)
    run_built_strategy(strategy_obj, pricing_data_df, config_obj, show_display_bool)
    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        display(strategy_obj.summary)
        display(strategy_obj.summary_trades)
    if save_results_bool:
        save_results(strategy_obj, output_dir=output_dir_str)
    return strategy_obj


def build_capacity_analysis_inputs(
    show_display_bool: bool = False,
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
) -> dict[str, object]:
    config_obj = configured_config(DEFAULT_CONFIG, backtest_start_date_str, capital_base_float, end_date_str)
    strategy_obj, pricing_data_df = _prepare(config_obj)
    # *** CRITICAL *** CapacityAnalysis must assess the same completed order
    # ledger as the reference backtest.
    run_built_strategy(strategy_obj, pricing_data_df, config_obj, show_display_bool)
    strategy_obj.universe_df = None
    return {
        "strategy_obj": strategy_obj,
        "pricing_data_df": pricing_data_df,
        "execution_policy_str": "MOO",
        "impact_profile_str": "MOO_NASDAQ_LARGE",
    }


def build_execution_timing_analysis_inputs() -> dict[str, object]:
    return timing_inputs(
        SectorCapNatr20VxnScaledNdxStrategy, STRATEGY_NAME_STR, DEFAULT_CONFIG, get_natr20_vxn_scaled_ndx_data
    )


if __name__ == "__main__":
    run_variant()
