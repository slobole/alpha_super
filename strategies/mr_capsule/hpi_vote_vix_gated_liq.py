"""HPI-G pod of the MR capsule with liquidity sizing: the rules of strategies/mr_capsule/hpi_vote_vix_gated.py, another rule for entry dollars.

Rule and caveats of the sizing: strategies/mr_capsule/liquidity_sizing.py. Rules and caveats of the pod: hpi_vote_vix_gated.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from data.norgate_loader import TOTALRETURN_ADJUSTMENT_STR
from strategies.hpi.stateful_long import (
    ENTRY_HORIZON_VOTE_STR,
    HPI_2D_FIELD_STR,
    HPI_5D_FIELD_STR,
    HPI_THRESHOLD_FLOAT,
    MAX_ENTRY_IBS_FLOAT,
    RETURN_2D_FIELD_STR,
    RETURN_5D_FIELD_STR,
    TURNOVER_FIELD_STR,
)
from strategies.mr_capsule.hpi_vote_vix_gated import BENCHMARK_SYMBOL_STR, HPIVoteVixGatedStrategy, load_hpi_capsule_pricing_data
from strategies.mr_capsule.liquidity_sizing import LiquiditySizedPodMixin, member_at_each_session_df
from strategies.mr_capsule.vix_stress_gate import load_vix_close_ser

HORIZON_FIELD_TUPLE = ((RETURN_2D_FIELD_STR, HPI_2D_FIELD_STR), ("return_3d_ser", "hpi_value_ser"), (RETURN_5D_FIELD_STR, HPI_5D_FIELD_STR))


class HPIVoteVixGatedLiquidityStrategy(LiquiditySizedPodMixin, HPIVoteVixGatedStrategy):
    """HPIVoteVixGatedStrategy with each entry sized by liquidity."""

    def _liquidity_candidate_df(self, signal_df: pd.DataFrame, stock_list: list[str]) -> pd.DataFrame:
        # The conditions of HPIStatefulLongStrategy.get_opportunity_list in vote mode, on every session.
        def field_mat(field_str: str) -> np.ndarray:
            return signal_df.xs(field_str, axis=1, level=1).reindex(columns=stock_list).astype(float).to_numpy()

        close_mat, sma_mat, ibs_mat = field_mat("Close"), field_mat("sma_200_price_ser"), field_mat("ibs_value_ser")
        present_mat = np.isfinite(close_mat) & np.isfinite(field_mat(TURNOVER_FIELD_STR)) & np.isfinite(sma_mat) & np.isfinite(ibs_mat)
        vote_mat = np.zeros(close_mat.shape, dtype=int)
        with np.errstate(invalid="ignore"):
            for return_field_str, hpi_field_str in HORIZON_FIELD_TUPLE:
                return_mat, hpi_mat = field_mat(return_field_str), field_mat(hpi_field_str)
                present_mat &= np.isfinite(return_mat) & np.isfinite(hpi_mat)
                vote_mat += ((return_mat < 0.0) & (hpi_mat < HPI_THRESHOLD_FLOAT)).astype(int)
            candidate_mat = present_mat & (vote_mat >= 2) & (ibs_mat < MAX_ENTRY_IBS_FLOAT) & (close_mat > sma_mat)
        return pd.DataFrame(candidate_mat & member_at_each_session_df(self.universe_df, signal_df.index, stock_list).to_numpy(), index=signal_df.index, columns=stock_list)


def build_hpi_capsule_liquidity_strategy(
    *,
    strategy_name_str: str,
    parking_enabled_bool: bool,
    spmo_parking_enabled_bool: bool,
    universe_df: pd.DataFrame,
    vix_close_ser: pd.Series,
    capital_base_float: float = 100_000.0,
    slippage_float: float = 0.00025,
    backtest_start_date_str: str = "2004-01-01",
) -> HPIVoteVixGatedLiquidityStrategy:
    """One configured object: the backtest, the analysis hooks and the live adapter all build it here."""
    strategy_obj = HPIVoteVixGatedLiquidityStrategy(
        name=strategy_name_str,
        benchmarks=[BENCHMARK_SYMBOL_STR],
        ranking_field_str=TURNOVER_FIELD_STR,
        capital_base=capital_base_float,
        slippage=slippage_float,
        entry_mode_str=ENTRY_HORIZON_VOTE_STR,
        backtest_start_date_str=backtest_start_date_str,
    )
    strategy_obj.universe_df = universe_df
    strategy_obj.vix_close_ser = vix_close_ser
    strategy_obj.parking_enabled_bool = parking_enabled_bool
    strategy_obj.spmo_parking_enabled_bool = spmo_parking_enabled_bool
    return strategy_obj


def run_hpi_capsule_liquidity_pod(
    *,
    strategy_name_str: str,
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str = "2004-01-01",
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
    slippage_float: float = 0.00025,
) -> HPIVoteVixGatedLiquidityStrategy:
    """Run the sized pod with idle cash in BIL."""
    pricing_data_df, universe_df = load_hpi_capsule_pricing_data(end_date_str)
    strategy_obj = build_hpi_capsule_liquidity_strategy(
        strategy_name_str=strategy_name_str,
        parking_enabled_bool=True,
        spmo_parking_enabled_bool=False,
        universe_df=universe_df,
        vix_close_ser=load_vix_close_ser(end_date_str),
        capital_base_float=capital_base_float,
        slippage_float=slippage_float,
        backtest_start_date_str=backtest_start_date_str,
    )
    # *** CRITICAL*** pre-start history is kept for the HPI warm-up and for m's windows; trading starts on the
    # requested execution calendar (same convention as run_hpi_variant).
    calendar_idx = pricing_data_df.index[pricing_data_df.index >= pd.Timestamp(backtest_start_date_str)]
    run_daily(strategy_obj, pricing_data_df, calendar_idx, show_progress=show_display_bool, show_signal_progress_bool=show_display_bool)
    strategy_obj.universe_df = None
    if show_display_bool:
        display(strategy_obj.summary)
        display(strategy_obj.summary_trades)
    if save_results_bool:
        save_results(strategy_obj, output_dir=output_dir_str)
    return strategy_obj


def build_hpi_capsule_liquidity_capacity_analysis_inputs(
    *,
    strategy_name_str: str,
    show_display_bool: bool = False,
    backtest_start_date_str: str = "2004-01-01",
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
) -> dict[str, object]:
    """One completed run for CapacityAnalysis (the order ledger includes the BIL parking orders)."""
    pricing_data_df, universe_df = load_hpi_capsule_pricing_data(end_date_str)
    strategy_obj = build_hpi_capsule_liquidity_strategy(
        strategy_name_str=strategy_name_str,
        parking_enabled_bool=True,
        spmo_parking_enabled_bool=False,
        universe_df=universe_df,
        vix_close_ser=load_vix_close_ser(end_date_str),
        capital_base_float=capital_base_float,
        backtest_start_date_str=backtest_start_date_str,
    )
    # *** CRITICAL*** CapacityAnalysis must assess the same completed order ledger as the Bench run.
    calendar_idx = pricing_data_df.index[pricing_data_df.index >= pd.Timestamp(backtest_start_date_str)]
    run_daily(strategy_obj, pricing_data_df, calendar_idx, show_progress=show_display_bool, show_signal_progress_bool=show_display_bool)
    strategy_obj.universe_df = None
    strategy_obj._performance_benchmark_symbol_str = BENCHMARK_SYMBOL_STR
    strategy_obj._performance_benchmark_adjustment_str = TOTALRETURN_ADJUSTMENT_STR
    return {
        "strategy_obj": strategy_obj,
        "pricing_data_df": pricing_data_df,
        "execution_policy_str": "MOO",
        "impact_profile_str": "MOO_LARGE_MIXED",
    }


def build_hpi_capsule_liquidity_execution_timing_analysis_inputs(*, strategy_name_str: str) -> dict[str, object]:
    """Inputs for ExecutionTimingAnalysis; a timing variant moves the stock and the parking fills alike."""
    from strategies.mr_capsule.timing_hook import prepare_pod_for_timing_analysis

    pricing_data_df, universe_df = load_hpi_capsule_pricing_data(None)
    vix_close_ser = load_vix_close_ser(None)
    # *** CRITICAL*** the same post-warm-up execution calendar as the Bench run (2004 onward).
    calendar_idx = pricing_data_df.index[pricing_data_df.index >= pd.Timestamp("2004-01-01")]

    def strategy_factory_fn() -> HPIVoteVixGatedLiquidityStrategy:
        return prepare_pod_for_timing_analysis(build_hpi_capsule_liquidity_strategy(
            strategy_name_str=strategy_name_str,
            parking_enabled_bool=True,
            spmo_parking_enabled_bool=False,
            universe_df=universe_df,
            vix_close_ser=vix_close_ser,
        ), pricing_data_df.index)

    return {
        "strategy_factory_fn": strategy_factory_fn,
        "pricing_data_df": pricing_data_df,
        "calendar_idx": pd.DatetimeIndex(calendar_idx),
        "order_generation_mode_str": "signal_bar",
        "risk_model_str": "daily_ohlc_signal",
        "entry_timing_str_tuple": ("same_close_moc", "next_open", "next_close"),
        "exit_timing_str_tuple": ("same_close_moc", "next_open", "next_close"),
        "default_entry_timing_str": "next_open",
        "default_exit_timing_str": "next_open",
    }
