"""DV2-G pod of the MR capsule with liquidity sizing: the rules of strategies/mr_capsule/dv2_vix_gated.py, another rule for entry dollars.

Rule and caveats of the sizing: strategies/mr_capsule/liquidity_sizing.py. Rules and caveats of the pod: dv2_vix_gated.py.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from data.norgate_loader import TOTALRETURN_ADJUSTMENT_STR
from strategies.dv2.strategy_mr_dv2 import default_trade_id_int
from strategies.mr_capsule.dv2_vix_gated import BENCHMARK_LIST, DV2VixGatedStrategy, load_pricing_data
from strategies.mr_capsule.liquidity_sizing import LiquiditySizedPodMixin, member_at_each_session_df
from strategies.mr_capsule.vix_stress_gate import load_vix_close_ser


class DV2VixGatedLiquidityStrategy(LiquiditySizedPodMixin, DV2VixGatedStrategy):
    """DV2VixGatedStrategy with each entry sized by liquidity."""

    def _liquidity_candidate_df(self, signal_df: pd.DataFrame, stock_list: list[str]) -> pd.DataFrame:
        # The conditions of DVO2Strategy.get_opportunities, on every session: `close.unstack().dropna()` keeps a stock
        # only with every field present, and a field a stock lacks altogether counts as missing.
        field_list = sorted({str(field_obj) for _, field_obj in signal_df.columns})
        complete_mat = np.ones((len(signal_df), len(stock_list)), dtype=bool)
        frame_dict = {}
        for field_str in field_list:
            frame_df = signal_df.xs(field_str, axis=1, level=1).reindex(columns=stock_list)
            complete_mat &= frame_df.notna().to_numpy()
            if field_str in ("Close", "dv2", "sma_200", "p126d_return"):
                frame_dict[field_str] = frame_df.astype(float).to_numpy()
        with np.errstate(invalid="ignore"):
            candidate_mat = (
                complete_mat
                & (frame_dict["dv2"] < 10.0)
                & (frame_dict["Close"] > frame_dict["sma_200"])
                & (frame_dict["p126d_return"] > 0.05)
            )
        return pd.DataFrame(candidate_mat & member_at_each_session_df(self.universe_df, signal_df.index, stock_list).to_numpy(), index=signal_df.index, columns=stock_list)


def build_dv2_capsule_liquidity_strategy(
    *,
    strategy_name_str: str,
    parking_enabled_bool: bool,
    spmo_parking_enabled_bool: bool,
    universe_df: pd.DataFrame,
    vix_close_ser: pd.Series,
    capital_base_float: float = 100_000.0,
    slippage_float: float = 0.00025,
) -> DV2VixGatedLiquidityStrategy:
    """One configured object: the backtest, the analysis hooks and the live adapter all build it here."""
    strategy = DV2VixGatedLiquidityStrategy(
        name=strategy_name_str,
        benchmarks=BENCHMARK_LIST,
        capital_base=capital_base_float,
        slippage=slippage_float,
        commission_per_share=0.005,
        commission_minimum=1.0,
        performance_benchmark_adjustment_str=TOTALRETURN_ADJUSTMENT_STR,
    )
    strategy.universe_df = universe_df
    strategy.trade_id = 0
    strategy.current_trade = defaultdict(default_trade_id_int)
    strategy.vix_close_ser = vix_close_ser
    strategy.parking_enabled_bool = parking_enabled_bool
    strategy.spmo_parking_enabled_bool = spmo_parking_enabled_bool
    return strategy


def run_dv2_capsule_liquidity_pod(
    *,
    strategy_name_str: str,
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str = "2004-01-01",
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
    slippage_float: float = 0.00025,
) -> DV2VixGatedLiquidityStrategy:
    """Run the sized pod with idle cash in BIL."""
    pricing_data_df, universe_df = load_pricing_data(end_date_str)
    strategy = build_dv2_capsule_liquidity_strategy(
        strategy_name_str=strategy_name_str,
        parking_enabled_bool=True,
        spmo_parking_enabled_bool=False,
        universe_df=universe_df,
        vix_close_ser=load_vix_close_ser(end_date_str),
        capital_base_float=capital_base_float,
        slippage_float=slippage_float,
    )
    # *** CRITICAL*** full pre-start history stays in the frame for indicators and for m's windows; trading starts at
    # the first deployment fill session (same convention as DVO2Strategy.run_variant).
    calendar_idx = pricing_data_df.index[pricing_data_df.index >= pd.Timestamp(backtest_start_date_str)]
    run_daily(strategy, pricing_data_df, calendar_idx, show_progress=show_display_bool, show_signal_progress_bool=show_display_bool)
    strategy.universe_df = None
    if show_display_bool:
        display(strategy.summary)
        display(strategy.summary_trades)
    if save_results_bool:
        save_results(strategy, output_dir=output_dir_str)
    return strategy


def build_dv2_capsule_liquidity_capacity_analysis_inputs(
    *,
    strategy_name_str: str,
    show_display_bool: bool = False,
    backtest_start_date_str: str = "2004-01-01",
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
) -> dict[str, object]:
    """One completed run for CapacityAnalysis (the order ledger includes the BIL parking orders)."""
    pricing_data_df, universe_df = load_pricing_data(end_date_str)
    strategy = build_dv2_capsule_liquidity_strategy(
        strategy_name_str=strategy_name_str,
        parking_enabled_bool=True,
        spmo_parking_enabled_bool=False,
        universe_df=universe_df,
        vix_close_ser=load_vix_close_ser(end_date_str),
        capital_base_float=capital_base_float,
    )
    # *** CRITICAL*** CapacityAnalysis must assess the same completed order ledger as the Bench run.
    calendar_idx = pricing_data_df.index[pricing_data_df.index >= pd.Timestamp(backtest_start_date_str)]
    run_daily(strategy, pricing_data_df, calendar_idx, show_progress=show_display_bool, show_signal_progress_bool=show_display_bool)
    strategy.universe_df = None
    strategy._performance_benchmark_symbol_str = BENCHMARK_LIST[0]
    strategy._performance_benchmark_adjustment_str = TOTALRETURN_ADJUSTMENT_STR
    return {
        "strategy_obj": strategy,
        "pricing_data_df": pricing_data_df,
        "execution_policy_str": "MOO",
        "impact_profile_str": "MOO_LARGE_MIXED",
    }


def build_dv2_capsule_liquidity_execution_timing_analysis_inputs(*, strategy_name_str: str) -> dict[str, object]:
    """Inputs for ExecutionTimingAnalysis; a timing variant moves the stock and the parking fills alike."""
    from strategies.mr_capsule.timing_hook import prepare_pod_for_timing_analysis

    pricing_data_df, universe_df = load_pricing_data(None)
    vix_close_ser = load_vix_close_ser(None)
    # *** CRITICAL*** the same post-warm-up execution calendar as the Bench run (2004 onward).
    calendar_idx = pricing_data_df.index[pricing_data_df.index >= pd.Timestamp("2004-01-01")]

    def strategy_factory_fn() -> DV2VixGatedLiquidityStrategy:
        return prepare_pod_for_timing_analysis(build_dv2_capsule_liquidity_strategy(
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
