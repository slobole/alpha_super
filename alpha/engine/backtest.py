"""Compatibility wrapper for the current Vanilla backtest runner."""

import pandas as pd

from alpha.engine.backtester import VanillaBacktester
from alpha.engine.strategy import Strategy


def run_daily(
    strategy: Strategy,
    pricing_data: pd.DataFrame,
    calendar: pd.DatetimeIndex = None,
    show_progress: bool = True,
    show_signal_progress_bool: bool = True,
    audit_override_bool: bool | None = False,
    audit_sample_size_int: int | None = None,
):
    vanilla_backtester = VanillaBacktester()
    return vanilla_backtester.run(
        strategy=strategy,
        pricing_data_df=pricing_data,
        calendar_idx=calendar,
        show_progress_bool=show_progress,
        show_signal_progress_bool=show_signal_progress_bool,
        audit_override_bool=audit_override_bool,
        audit_sample_size_int=audit_sample_size_int,
    )
