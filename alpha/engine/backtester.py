"""
Named backtester runners for the engine.

Current baseline
----------------
The house default is the Vanilla backtest:

    decision_t = f(I_{t-1})

    q^{target}_{i,t} = f(V^{close}_{t-1}, Close_{i,t-1})

    fill_{i,t} = Open_{i,t}
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter

import pandas as pd
from tqdm import tqdm

from alpha.engine.strategy import Strategy


class Backtester(ABC):
    """Base interface for engine backtest runners."""

    @abstractmethod
    def run(
        self,
        strategy: Strategy,
        pricing_data_df: pd.DataFrame,
        calendar_idx: pd.DatetimeIndex | None = None,
        show_progress_bool: bool = True,
        show_signal_progress_bool: bool = True,
        audit_override_bool: bool | None = False,
        audit_sample_size_int: int | None = None,
    ) -> Strategy:
        """Execute the backtest and return the mutated strategy."""


class VanillaBacktester(Backtester):
    """
    Current one-path daily runner.

    This is the baseline engine contract used by the existing strategies.
    """

    def run(
        self,
        strategy: Strategy,
        pricing_data_df: pd.DataFrame,
        calendar_idx: pd.DatetimeIndex | None = None,
        show_progress_bool: bool = True,
        show_signal_progress_bool: bool = True,
        audit_override_bool: bool | None = False,
        audit_sample_size_int: int | None = None,
    ) -> Strategy:
        if calendar_idx is None:
            calendar_idx = pricing_data_df.index

        strategy.show_signal_progress_bool = show_signal_progress_bool
        strategy.show_audit_progress_bool = show_signal_progress_bool

        print("signal precompute...")
        signal_start_float = perf_counter()
        full_data_df = strategy.compute_signals(pricing_data_df)
        signal_elapsed_float = perf_counter() - signal_start_float
        print(f"signal precompute completed in {signal_elapsed_float:.2f}s")

        audit_enabled_bool = (
            strategy.enable_signal_audit if audit_override_bool is None else audit_override_bool
        )
        if audit_enabled_bool:
            print("signal audit...")
            audit_start_float = perf_counter()
            strategy.audit_signals(
                pricing_data_df,
                full_data_df,
                sample_size=audit_sample_size_int,
            )
            audit_elapsed_float = perf_counter() - audit_start_float
            print(f"signal audit completed in {audit_elapsed_float:.2f}s")
        else:
            print("signal audit skipped.")

        if pricing_data_df.index.get_loc(calendar_idx[0]) != 0:
            strategy.previous_bar = pricing_data_df.index[pricing_data_df.index.get_loc(calendar_idx[0]) - 1]

        print("backtest loop...")
        progress_iter_obj = tqdm(calendar_idx, desc="backtest") if show_progress_bool else calendar_idx

        for bar_ts in progress_iter_obj:
            if strategy.previous_bar is None:
                strategy.current_bar = bar_ts
                strategy.update_metrics(pricing_data_df, calendar_idx[0])
                strategy.previous_bar = bar_ts
                continue

            strategy.current_bar = bar_ts

            current_data_df, close_row_ser, open_price_ser = strategy.restrict_data(full_data_df)

            # *** CRITICAL*** The Vanilla runner passes only data through
            # previous_bar into iterate(). Orders created here are then filled
            # by process_orders() on the current_bar open.
            strategy.iterate(current_data_df, close_row_ser, open_price_ser)
            strategy.process_orders(pricing_data_df)
            strategy.update_metrics(pricing_data_df, calendar_idx[0])
            strategy.previous_bar = bar_ts

            total_return_float = strategy.total_value / strategy._capital_base - 1.0
            num_days_int = strategy.num_days
            annualized_return_float = (1.0 + total_return_float) ** (252.0 / (num_days_int + 1)) - 1.0

            if show_progress_bool:
                progress_iter_obj.set_postfix(
                    bar=bar_ts.strftime("%Y-%m-%d"),
                    total_return=f"{total_return_float:.1%}",
                    annualized_return=f"{annualized_return_float:.1%}",
                )

        strategy.finalize(full_data_df)
        strategy.summarize()
        return strategy
