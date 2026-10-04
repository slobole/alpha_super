"""
The LIVE NDX VXN pod rule (dollar ATR20 ranking) with a 40% GICS-sector cap.

One of the two books of the NDX design of record "E2 + 40% sector cap" (Scout amendment A15, 2026-10-04,
docs/research/SCOUT_ROBUSTNESS_20261002.md): this file is the dollar-ATR book, and its sibling
strategy_mo_natr20_ndx_vxn_scaled_sector_cap.py is the NATR20 book. The pair is held 50/50 in
portfolios/ndx_e2_sector_cap_5050.yaml. Research / PM only; nothing here is wired to a live account.

Everything is the live strategy (strategy_mo_atr_normalized_ndx_vxn_scaled) except the top-N walk:

    ranked candidates by score_{i,t} = ROC12_{i,t} / ATR20$_{i,t}          (unchanged, ties: symbol ascending)

    accept candidate i iff count(sector(i) in selected) < sector_cap      (sector_cap = 4 of 10 names = 40%)
    stop when max_positions are accepted or the candidates run out

    target_weight_{i,t} = (1 / max_positions) * clip(22 / VXN_t, 0.25, 1.0)

A skipped name's slot goes to the next-ranked eligible name, so the book stays fully invested whenever ten
eligible names exist.

*** REALISM GAP *** Sector labels are Norgate's CURRENT GICS level-1 classification (Norgate keeps no history),
so a reclassified name (e.g. the 2018 Communication Services restructure) carries today's sector back in time.
A mild look-ahead, documented in ASSUMPTIONS_AND_GAPS.md (as strategy_mo_atr_normalized_sector_cap.py).
Unclassified names share the bucket "UNKNOWN", capped like any sector.

Selection evidence (Scout weights engine, 2000-09 to 2026-09, cash credited at T-bills): the 50/50 pair Sharpe
0.87, CAGR 13.6%, Max DD -21.5%. History through 2026 has been examined; this is not an untouched holdout.

Execution mapping is unchanged: decision at the actual last tradable close of the month, execution at the next
tradable open. Costs: 2.5 bps/side, $0.005/share, $1 minimum.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from strategies.momentum.strategy_mo_atr_normalized_ndx import (
    _map_rebalance_schedule_to_decision_close_schedule_df,
    configure_total_return_benchmark_provenance,
    default_trade_id_int,
)
from strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled import (
    VxnScaledAtrNormalizedNdxConfig,
    VxnScaledAtrNormalizedNdxStrategy,
    get_asof_vxn_scale_float,
    get_vxn_scaled_atr_normalized_ndx_data,
)
from strategies.momentum.strategy_mo_atr_normalized_sector_cap import (
    UNKNOWN_SECTOR_STR,
    build_current_gics_sector_map,
    select_sector_capped_symbol_list,
)


STRATEGY_NAME_STR = "strategy_mo_atr_normalized_ndx_vxn_scaled_sector_cap"


@dataclass(frozen=True)
class SectorCapVxnScaledAtrNormalizedNdxConfig(VxnScaledAtrNormalizedNdxConfig):
    sector_cap_int: int = 4

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.sector_cap_int <= 0:
            raise ValueError("sector_cap_int must be positive.")


DEFAULT_CONFIG = SectorCapVxnScaledAtrNormalizedNdxConfig()

__all__ = [
    "DEFAULT_CONFIG",
    "SectorCapSelectionMixin",
    "SectorCapVxnScaledAtrNormalizedNdxConfig",
    "SectorCapVxnScaledAtrNormalizedNdxStrategy",
    "build_capacity_analysis_inputs",
    "build_execution_timing_analysis_inputs",
    "run_variant",
]


class SectorCapSelectionMixin:
    """
    Replaces the top-N slice of an NDX VXN strategy with the sector-capped walk; the rest of the parent is unchanged.

    Requires the parent to provide get_ranked_candidate_feature_df, max_positions_int, vxn_scale_signal_df and
    previous_bar, as both VxnScaledAtrNormalizedNdxStrategy and Natr20VxnScaledNdxStrategy do.
    """

    def configure_sector_cap(self, sector_by_symbol_map: Mapping[str, str], sector_cap_int: int) -> None:
        if sector_cap_int <= 0:
            raise ValueError("sector_cap_int must be positive.")
        if len(sector_by_symbol_map) == 0:
            raise ValueError("sector_by_symbol_map must not be empty.")
        self.sector_by_symbol_map = dict(sector_by_symbol_map)
        self.sector_cap_int = int(sector_cap_int)
        self.selection_audit_row_list: list[dict[str, object]] = []

    def get_target_weight_ser(self, close_row_ser: pd.Series) -> pd.Series:
        ranked_candidate_feature_df = self.get_ranked_candidate_feature_df(close_row_ser=close_row_ser)
        if len(ranked_candidate_feature_df) == 0:
            return pd.Series(dtype=float)

        ranked_symbol_list = ranked_candidate_feature_df.index.astype(str).tolist()
        selected_symbol_list = select_sector_capped_symbol_list(
            ranked_symbol_list=ranked_symbol_list,
            sector_by_symbol_map=self.sector_by_symbol_map,
            max_positions_int=self.max_positions_int,
            sector_cap_int=self.sector_cap_int,
        )
        sector_count_ser = pd.Series(
            [self.sector_by_symbol_map.get(s, UNKNOWN_SECTOR_STR) for s in selected_symbol_list], dtype=object
        ).value_counts()
        self.selection_audit_row_list.append(
            {
                "decision_date_ts": pd.Timestamp(self.previous_bar),
                "candidate_count_int": int(len(ranked_symbol_list)),
                "selected_symbol_list": list(selected_symbol_list),
                "max_sector_count_int": int(sector_count_ser.max()) if len(sector_count_ser) > 0 else 0,
            }
        )

        target_weight_ser = pd.Series(1.0 / float(self.max_positions_int), index=selected_symbol_list, dtype=float)
        # *** CRITICAL *** At Close_T select the latest VXN observation <= T;
        # do not read the execution-day or terminal dataset VXN value.
        exposure_scale_float = get_asof_vxn_scale_float(
            vxn_scale_signal_df=self.vxn_scale_signal_df,
            decision_date_ts=pd.Timestamp(self.previous_bar),
        )
        return target_weight_ser * exposure_scale_float

    def get_selection_audit_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.selection_audit_row_list).set_index("decision_date_ts") if self.selection_audit_row_list else pd.DataFrame()


class SectorCapVxnScaledAtrNormalizedNdxStrategy(SectorCapSelectionMixin, VxnScaledAtrNormalizedNdxStrategy):
    """The live dollar-ATR NDX VXN pod with a hard GICS-sector cap on selection."""


def sector_map_for_universe(universe_df: pd.DataFrame) -> dict[str, str]:
    """Current GICS level-1 sector for every symbol that was ever a member (Norgate, fetched once)."""
    return build_current_gics_sector_map(sorted(str(s) for s in universe_df.columns))


def configured_config(
    default_config,
    backtest_start_date_str: str | None,
    capital_base_float: float | None,
    end_date_str: str | None,
):
    if backtest_start_date_str is None and capital_base_float is None and end_date_str is None:
        return default_config
    return replace(
        default_config,
        backtest_start_date_str=default_config.backtest_start_date_str if backtest_start_date_str is None else backtest_start_date_str,
        capital_base_float=default_config.capital_base_float if capital_base_float is None else float(capital_base_float),
        end_date_str=end_date_str,
    )


def build_strategy(
    strategy_cls: type,
    name_str: str,
    config_obj,
    rebalance_schedule_df: pd.DataFrame,
    vxn_scale_signal_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    sector_by_symbol_map: Mapping[str, str],
):
    strategy_obj = strategy_cls(
        name=name_str,
        benchmarks=[config_obj.performance_benchmark_symbol_str],
        rebalance_schedule_df=rebalance_schedule_df,
        vxn_scale_signal_df=vxn_scale_signal_df,
        regime_symbol_str=config_obj.regime_symbol_str,
        capital_base=config_obj.capital_base_float,
        slippage=config_obj.slippage_float,
        commission_per_share=config_obj.commission_per_share_float,
        commission_minimum=config_obj.commission_minimum_float,
        lookback_month_int=config_obj.lookback_month_int,
        index_trend_window_int=config_obj.index_trend_window_int,
        stock_trend_window_int=config_obj.stock_trend_window_int,
        max_positions_int=config_obj.max_positions_int,
    )
    strategy_obj.configure_sector_cap(sector_by_symbol_map, config_obj.sector_cap_int)
    strategy_obj.universe_df = universe_df
    configure_total_return_benchmark_provenance(strategy_obj=strategy_obj, config_obj=config_obj)
    return strategy_obj


def run_built_strategy(strategy_obj, pricing_data_df: pd.DataFrame, config_obj, show_display_bool: bool) -> None:
    # *** CRITICAL*** Deployment-reference backtests keep full pre-start
    # history for monthly ATR and trend features, but the executable calendar
    # starts at the first deployment fill session.
    calendar_idx = pricing_data_df.index[pricing_data_df.index >= pd.Timestamp(config_obj.backtest_start_date_str)]
    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar=calendar_idx,
        show_progress=show_display_bool,
        show_signal_progress_bool=show_display_bool,
        audit_override_bool=None,
    )


def timing_inputs(strategy_cls: type, name_str: str, config_obj, data_fn) -> dict[str, object]:
    pricing_data_df, universe_df, rebalance_schedule_df, vxn_scale_signal_df = data_fn(
        config_obj, include_total_return_benchmark_bool=True
    )
    decision_close_schedule_df = _map_rebalance_schedule_to_decision_close_schedule_df(
        rebalance_schedule_df=rebalance_schedule_df,
    )
    sector_by_symbol_map = sector_map_for_universe(universe_df)

    def strategy_factory_fn():
        strategy_obj = build_strategy(
            strategy_cls, name_str, config_obj, decision_close_schedule_df, vxn_scale_signal_df, universe_df, sector_by_symbol_map
        )
        strategy_obj.trade_id_int = 0
        strategy_obj.current_trade_map = defaultdict(default_trade_id_int)
        return strategy_obj

    calendar_idx = pricing_data_df.index[pricing_data_df.index >= pd.Timestamp(config_obj.backtest_start_date_str)]
    return {
        "strategy_factory_fn": strategy_factory_fn,
        "pricing_data_df": pricing_data_df,
        "calendar_idx": pd.DatetimeIndex(calendar_idx),
        "order_generation_mode_str": "signal_bar",
        "risk_model_str": "taa_rebalance",
        "entry_timing_str_tuple": ("same_close_moc", "next_open", "next_close"),
        "exit_timing_str_tuple": ("same_close_moc", "next_open", "next_close"),
        "default_entry_timing_str": "next_open",
        "default_exit_timing_str": "next_open",
    }


def _prepare(config_obj):
    pricing_data_df, universe_df, rebalance_schedule_df, vxn_scale_signal_df = get_vxn_scaled_atr_normalized_ndx_data(
        config_obj, include_total_return_benchmark_bool=True
    )
    strategy_obj = build_strategy(
        SectorCapVxnScaledAtrNormalizedNdxStrategy, STRATEGY_NAME_STR, config_obj, rebalance_schedule_df,
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
) -> SectorCapVxnScaledAtrNormalizedNdxStrategy:
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
        SectorCapVxnScaledAtrNormalizedNdxStrategy, STRATEGY_NAME_STR, DEFAULT_CONFIG, get_vxn_scaled_atr_normalized_ndx_data
    )


if __name__ == "__main__":
    run_variant()
