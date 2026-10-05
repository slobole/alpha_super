"""HPI-G pod of the MR capsule: HPI 2/3/5 vote behind the shared VIX stress gate, idle cash parked.

Shared code of the Bench entry points strategy_mr_hpi_vote_vix_gated_spmo.py (the capsule spec: SPMO while the gate
is closed, BIL otherwise) and strategy_mr_hpi_vote_vix_gated_bil.py (idle cash all in BIL).

Stock rules are those of strategies/hpi/strategy_mr_hpi_sp500_2_3_5_vote.py (HPIStatefulLongStrategy, vote mode,
Turnover rank), unchanged:
    entry: HPI < 30 on >= 2 of the 2/3/5-day horizons, IBS < 0.10, Close > SMA200, PIT S&P 500 member
    exit:  IBS > 0.90 or RSI2 > 90 or leaving the index; same-open slot reuse (G-033)
Capsule additions (docs/research/MR_CAPSULE_20261003.md):
    new entries only when the VIX stress gate is open at Close_T; exits never gated;
    SPMO / BIL are excluded from the slot count, the exit rules and the membership exit;
    idle cash: SPMO (8% vol target, weekly) while the gate is closed, BIL otherwise.
Decisions after Close_T, fills at Open_(T+1). CAPITALSPECIAL fills and marks; dividends net of 25% withholding.

Known caveats: as strategies/mr_capsule/dv2_vix_gated.py, plus
- issue: HPI's calm-market trades carry a small real edge (unlike DV2), so the gate gives up some return at engine
  costs (research: standalone Sharpe 1.05 vs 1.08 ungated; 0.95 vs 0.90 at +5 bps); bias: n/a; impact: low-medium.
- issue: SPMO / BIL use Norgate's market-day padding (as the DV2 pod's frame), unlike HPI's unpadded stocks, so a
  parking order on a session without a trade would fill at the padded price (prior close); bias: optimistic,
  negligible (BIL has no such session; SPMO's guard B1 keeps it out of 2015-2017, none since 2018); impact: low.
- issue: negative cash, as DV2-G: 128 sessions with parking, 127 without (same-open slot reuse, G-033); minimum
  -7.6% of NAV; not financed (G-023); bias: optimistic, small.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from data.norgate_loader import CAPITALSPECIAL_ADJUSTMENT_STR, TOTALRETURN_ADJUSTMENT_STR, load_price_timeseries
from strategies.hpi.stateful_long import (
    ENTRY_HORIZON_VOTE_STR,
    EXIT_IBS_THRESHOLD_FLOAT,
    EXIT_RSI2_THRESHOLD_FLOAT,
    TURNOVER_FIELD_STR,
    HPIStatefulLongStrategy,
    get_asof_universe_symbol_set,
    load_exact_hpi_inputs,
)
from strategies.mr_capsule.capsule_pod import CapsulePodMixin
from strategies.mr_capsule.parking import PARKING_SYMBOL_TUPLE, require_parking_dividends
from strategies.mr_capsule.vix_stress_gate import load_vix_close_ser

BENCHMARK_SYMBOL_STR = "$SPXTR"


class HPIVoteVixGatedStrategy(CapsulePodMixin, HPIStatefulLongStrategy):
    """HPIStatefulLongStrategy (vote mode) with the capsule gate on entries and SPMO / BIL parking."""

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        signal_data_df = super().compute_signals(pricing_data_df)
        self._prepare_capsule_state()
        return signal_data_df

    def iterate(self, data_df: pd.DataFrame, close_row_ser: pd.Series, open_price_ser: pd.Series) -> None:
        # Same decision logic as HPIStatefulLongStrategy.iterate, with SPMO / BIL outside the slot count and the exit
        # rules, and entries only while the gate is open (tests/test_strategy_mr_capsule_pods.py pins the gate-open,
        # no-parking version to HPIStatefulLongStrategy trade for trade).
        if data_df is None or close_row_ser is None:
            return
        if self.universe_df is None:
            raise RuntimeError("HPI strategy requires a point-in-time universe.")

        decision_date_ts = pd.Timestamp(self.previous_bar)
        member_symbol_set = get_asof_universe_symbol_set(self.universe_df, decision_date_ts)
        long_position_ser = self._stock_position_ser()
        long_symbol_set = set(long_position_ser.index.astype(str))
        self.pending_exit_symbol_set.intersection_update(long_symbol_set)
        long_slots_int = self.max_positions_int - len(long_position_ser)
        exiting_symbol_set: set[str] = set()

        for symbol_str in long_position_ser.index.astype(str):
            ibs_value_float = close_row_ser.get((symbol_str, "ibs_value_ser"), np.nan)
            rsi2_value_float = close_row_ser.get((symbol_str, "rsi2_value_ser"), np.nan)
            exit_for_ibs_bool = pd.notna(ibs_value_float) and float(ibs_value_float) > EXIT_IBS_THRESHOLD_FLOAT
            exit_for_rsi2_bool = pd.notna(rsi2_value_float) and float(rsi2_value_float) > EXIT_RSI2_THRESHOLD_FLOAT
            exit_for_membership_bool = symbol_str not in member_symbol_set
            if exit_for_ibs_bool or exit_for_rsi2_bool or exit_for_membership_bool:
                self.pending_exit_symbol_set.add(symbol_str)
            current_open_float = open_price_ser.get(symbol_str, np.nan)
            has_tradable_open_bool = pd.notna(current_open_float) and np.isfinite(float(current_open_float))
            if symbol_str in self.pending_exit_symbol_set and has_tradable_open_bool:
                self.order_target_value(symbol_str, 0.0, trade_id=self.current_trade_map[symbol_str])
                # *** CRITICAL*** same-open slot reuse as the parent (G-033): the backtest knows Open_(T+1) printed.
                exiting_symbol_set.add(symbol_str)
                long_slots_int += 1

        capital_per_trade_float = self.previous_total_value / float(self.max_positions_int)
        gate_open_bool = self._gate_open_at_decision()
        entry_value_float = 0.0
        if gate_open_bool:
            opportunity_symbol_list = self.get_opportunity_list(close_row_ser, member_symbol_set)
            while long_slots_int > 0 and opportunity_symbol_list:
                symbol_str = opportunity_symbol_list.pop(0)
                if self.get_position(symbol_str) != 0:
                    continue
                self.trade_id_int += 1
                self.current_trade_map[symbol_str] = self.trade_id_int
                self.order_value(symbol_str, capital_per_trade_float, trade_id=self.trade_id_int)
                entry_value_float += capital_per_trade_float
                long_slots_int -= 1
        elif long_slots_int > 0:
            self._gate_closed_free_slot_session_int += 1

        # *** CRITICAL*** stock value after this decision's orders, at Close_T (the engine's sizing price).
        kept_value_float = 0.0
        for symbol_str, share_float in long_position_ser.items():
            if str(symbol_str) in exiting_symbol_set:
                continue
            kept_value_float += float(share_float) * float(close_row_ser.get((str(symbol_str), "Close"), np.nan))
        self._place_parking_orders(data_df, close_row_ser, kept_value_float + entry_value_float, gate_open_bool)

    def finalize(self, current_data):
        self._record_capsule_diagnostics()
        parent_finalize = getattr(super(), "finalize", None)
        if callable(parent_finalize):
            parent_finalize(current_data)


def append_parking_prices(pricing_data_df: pd.DataFrame, start_date_str: str, end_date_str: str | None) -> pd.DataFrame:
    """Add CAPITALSPECIAL SPMO and BIL on the HPI calendar, padded like the DV2 pod's frame (load_price_timeseries).

    The ETFs keep Norgate's market-day padding (Volume 0, prices = prior close) so a parking position is held through
    a session without a trade instead of being liquidated by HPI's removed-name rule; SPMO's tradability guard (B1)
    keeps it out of the 2015-2017 sessions without trades. HPI's own stock frame stays unpadded.
    """
    frame_list = [pricing_data_df]
    price_field_tuple = ("Open", "High", "Low", "Close")
    for symbol_str in PARKING_SYMBOL_TUPLE:
        price_df = load_price_timeseries(symbol_str, adjustment_str=CAPITALSPECIAL_ADJUSTMENT_STR, start_date_str=start_date_str, end_date_str=end_date_str)
        price_df.index = pd.to_datetime(price_df.index)
        price_df = price_df.reindex(pricing_data_df.index)
        price_df.columns = pd.MultiIndex.from_tuples([(symbol_str, field_str) for field_str in price_df.columns])
        observed_bool_ser = price_df[[(symbol_str, f) for f in price_field_tuple if (symbol_str, f) in price_df.columns]].notna().any(axis=1)
        dividend_key = (symbol_str, "Dividend")
        if dividend_key in price_df.columns:
            # *** CRITICAL*** as load_exact_hpi_inputs: a calendar row with no OHLC observation cannot carry a dividend.
            price_df.loc[~observed_bool_ser & price_df[dividend_key].isna(), dividend_key] = 0.0
        # *** CRITICAL*** forward-filled Close is valuation-only; Open/High/Low stay NaN so no synthetic trade fills.
        price_df[(symbol_str, "Close")] = price_df[(symbol_str, "Close")].ffill()
        frame_list.append(price_df)
    combined_df = pd.concat(frame_list, axis=1)
    combined_df.attrs = dict(pricing_data_df.attrs)  # concat drops attrs; run_variant re-declares the adjustments
    require_parking_dividends(combined_df)
    return combined_df


def load_hpi_capsule_pricing_data(end_date_str: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """HPI's exact S&P 500 inputs from 1998 plus SPMO / BIL, with every symbol's adjustment declared for the engine."""
    _, universe_df, pricing_data_df = load_exact_hpi_inputs(
        indexname_str="S&P 500", benchmark_symbol_str=BENCHMARK_SYMBOL_STR, start_date_str="1998-01-01", end_date_str=end_date_str
    )
    pricing_data_df = append_parking_prices(pricing_data_df, "1998-01-01", end_date_str)
    pricing_symbol_list = pricing_data_df.columns.get_level_values(0).unique().astype(str)
    # *** CRITICAL*** the engine's dividend guard needs every traded symbol declared CAPITALSPECIAL (concat drops attrs).
    pricing_data_df.attrs["norgate_adjustment_by_symbol_dict"] = {
        symbol_str: (TOTALRETURN_ADJUSTMENT_STR if symbol_str == BENCHMARK_SYMBOL_STR else CAPITALSPECIAL_ADJUSTMENT_STR)
        for symbol_str in pricing_symbol_list
    }
    return pricing_data_df, universe_df


def build_hpi_capsule_strategy(
    *,
    strategy_name_str: str,
    parking_enabled_bool: bool,
    spmo_parking_enabled_bool: bool,
    universe_df: pd.DataFrame,
    vix_close_ser: pd.Series,
    capital_base_float: float = 100_000.0,
    slippage_float: float = 0.00025,
    backtest_start_date_str: str = "2004-01-01",
) -> HPIVoteVixGatedStrategy:
    """One configured HPI-G object: the backtest, the analysis hooks and the live adapter all build it here."""
    strategy_obj = HPIVoteVixGatedStrategy(
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


def run_hpi_capsule_pod(
    *,
    strategy_name_str: str,
    parking_enabled_bool: bool,
    spmo_parking_enabled_bool: bool,
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str = "2004-01-01",
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
    slippage_float: float = 0.00025,
) -> HPIVoteVixGatedStrategy:
    """Run HPI-G with the given parking (the Bench entry points fix it; parking off = idle cash at 0%)."""
    pricing_data_df, universe_df = load_hpi_capsule_pricing_data(end_date_str)
    strategy_obj = build_hpi_capsule_strategy(
        strategy_name_str=strategy_name_str,
        parking_enabled_bool=parking_enabled_bool,
        spmo_parking_enabled_bool=spmo_parking_enabled_bool,
        universe_df=universe_df,
        vix_close_ser=load_vix_close_ser(end_date_str),
        capital_base_float=capital_base_float,
        slippage_float=slippage_float,
        backtest_start_date_str=backtest_start_date_str,
    )
    # *** CRITICAL*** pre-start history is kept for the 1,260-observation HPI warm-up; trading starts on the
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


def build_hpi_capsule_capacity_analysis_inputs(
    *,
    strategy_name_str: str,
    parking_enabled_bool: bool,
    spmo_parking_enabled_bool: bool,
    show_display_bool: bool = False,
    backtest_start_date_str: str = "2004-01-01",
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
) -> dict[str, object]:
    """One completed HPI-G run for CapacityAnalysis (the order ledger includes the BIL / SPMO parking orders)."""
    pricing_data_df, universe_df = load_hpi_capsule_pricing_data(end_date_str)
    strategy_obj = build_hpi_capsule_strategy(
        strategy_name_str=strategy_name_str,
        parking_enabled_bool=parking_enabled_bool,
        spmo_parking_enabled_bool=spmo_parking_enabled_bool,
        universe_df=universe_df,
        vix_close_ser=load_vix_close_ser(end_date_str),
        capital_base_float=capital_base_float,
        backtest_start_date_str=backtest_start_date_str,
    )
    # *** CRITICAL*** CapacityAnalysis must assess the same completed order ledger as the Bench run:
    # the 1,260-observation warm-up stays in the frame, execution runs on the requested calendar only.
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


def build_hpi_capsule_execution_timing_analysis_inputs(
    *,
    strategy_name_str: str,
    parking_enabled_bool: bool,
    spmo_parking_enabled_bool: bool,
) -> dict[str, object]:
    """Inputs for ExecutionTimingAnalysis; a timing variant moves the stock and the parking fills alike."""
    pricing_data_df, universe_df = load_hpi_capsule_pricing_data(None)
    vix_close_ser = load_vix_close_ser(None)
    # *** CRITICAL*** the same post-warm-up execution calendar as the Bench run (2004 onward).
    calendar_idx = pricing_data_df.index[pricing_data_df.index >= pd.Timestamp("2004-01-01")]

    def strategy_factory_fn() -> HPIVoteVixGatedStrategy:
        return build_hpi_capsule_strategy(
            strategy_name_str=strategy_name_str,
            parking_enabled_bool=parking_enabled_bool,
            spmo_parking_enabled_bool=spmo_parking_enabled_bool,
            universe_df=universe_df,
            vix_close_ser=vix_close_ser,
        )

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

