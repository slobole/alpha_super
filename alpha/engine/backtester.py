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
from datetime import datetime, timezone
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
        run_id_str: str | None = None,
        audit_log_path_str: str | None = None,
        trace_enabled_bool: bool = False,
        trace_log_path_str: str | None = None,
        trace_log_root_path_str: str | None = None,
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
        run_id_str: str | None = None,
        audit_log_path_str: str | None = None,
        trace_enabled_bool: bool = False,
        trace_log_path_str: str | None = None,
        trace_log_root_path_str: str | None = None,
    ) -> Strategy:
        if calendar_idx is None:
            calendar_idx = pricing_data_df.index

        resolved_run_id_str = run_id_str
        if resolved_run_id_str is None and (
            audit_log_path_str is not None or trace_enabled_bool or trace_log_path_str is not None
        ):
            resolved_run_id_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        strategy.configure_structured_logging(
            run_id_str=resolved_run_id_str,
            audit_log_path_str=audit_log_path_str,
            trace_enabled_bool=trace_enabled_bool,
            trace_log_path_str=trace_log_path_str,
            trace_log_root_path_str=trace_log_root_path_str,
        )
        strategy.show_signal_progress_bool = show_signal_progress_bool
        strategy.show_audit_progress_bool = show_signal_progress_bool

        strategy.log_audit_event(
            "engine.signal_precompute.started",
            {
                "pricing_row_count_int": int(len(pricing_data_df.index)),
                "pricing_column_count_int": int(len(pricing_data_df.columns)),
            },
        )
        print("signal precompute...")
        signal_start_float = perf_counter()
        full_data_df = strategy.compute_signals(pricing_data_df)
        signal_elapsed_float = perf_counter() - signal_start_float
        print(f"signal precompute completed in {signal_elapsed_float:.2f}s")
        strategy.log_audit_event(
            "engine.signal_precompute.completed",
            {
                "elapsed_seconds_float": float(signal_elapsed_float),
                "signal_row_count_int": int(len(full_data_df.index)),
                "signal_column_count_int": int(len(full_data_df.columns)),
            },
        )

        audit_enabled_bool = (
            strategy.enable_signal_audit if audit_override_bool is None else audit_override_bool
        )
        if audit_enabled_bool:
            strategy.log_audit_event(
                "engine.signal_audit.started",
                {"audit_sample_size_int": audit_sample_size_int},
            )
            print("signal audit...")
            audit_start_float = perf_counter()
            strategy.audit_signals(
                pricing_data_df,
                full_data_df,
                sample_size=audit_sample_size_int,
            )
            audit_elapsed_float = perf_counter() - audit_start_float
            print(f"signal audit completed in {audit_elapsed_float:.2f}s")
            strategy.log_audit_event(
                "engine.signal_audit.completed",
                {"elapsed_seconds_float": float(audit_elapsed_float)},
            )
        else:
            print("signal audit skipped.")
            strategy.log_audit_event(
                "engine.signal_audit.skipped",
                {"reason_code_str": "audit_disabled"},
            )

        if pricing_data_df.index.get_loc(calendar_idx[0]) != 0:
            strategy.previous_bar = pricing_data_df.index[pricing_data_df.index.get_loc(calendar_idx[0]) - 1]

        print("backtest loop...")
        strategy.log_audit_event(
            "engine.backtest_loop.started",
            {
                "calendar_row_count_int": int(len(calendar_idx)),
                "start_timestamp_str": pd.Timestamp(calendar_idx[0]).isoformat(),
                "end_timestamp_str": pd.Timestamp(calendar_idx[-1]).isoformat(),
            },
        )
        progress_iter_obj = tqdm(calendar_idx, desc="backtest") if show_progress_bool else calendar_idx

        for bar_ts in progress_iter_obj:
            if strategy.previous_bar is None:
                strategy.current_bar = bar_ts
                strategy.update_metrics(pricing_data_df, calendar_idx[0])
                strategy.log_trace_event(
                    "engine.bar.completed",
                    {
                        "open_order_count_int": int(len(strategy.get_orders())),
                        "transaction_count_int": int(len(strategy.get_transactions())),
                        "cash_float": float(strategy.cash),
                        "portfolio_value_float": float(strategy.portfolio_value),
                        "total_value_float": float(strategy.total_value),
                    },
                )
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
            strategy.log_trace_event(
                "engine.bar.completed",
                {
                    "open_order_count_int": int(len(strategy.get_orders())),
                    "transaction_count_int": int(len(strategy.get_transactions())),
                    "cash_float": float(strategy.cash),
                    "portfolio_value_float": float(strategy.portfolio_value),
                    "total_value_float": float(strategy.total_value),
                },
            )
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
        strategy.log_audit_event(
            "engine.backtest_loop.completed",
            {
                "transaction_count_int": int(len(strategy.get_transactions())),
                "final_cash_float": float(strategy.cash),
                "final_portfolio_value_float": float(strategy.portfolio_value),
                "final_total_value_float": float(strategy.total_value),
            },
        )
        return strategy
