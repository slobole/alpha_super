"""ExecutionTimingAnalysis on the MR capsule pods. Research only: the live path never imports this module."""

from __future__ import annotations

import pandas as pd


def prepare_pod_for_timing_analysis(strategy_obj, session_index: pd.DatetimeIndex):
    """Make one timing cell's pod behave as in the Bench run.

    The analysis computes signals once and builds a fresh pod per cell, so the gate and parking state that
    compute_signals prepares must be set here. Its signal-bar mode also sets current_bar to the decision session, while
    the parking rule reads the execution session to find the last session of the week: without the next session the
    weekly BIL sweep never fires, and every cell runs with idle cash outside BIL and no error.
    """
    strategy_obj._prepare_capsule_state()
    pod_iterate_fn = strategy_obj.iterate

    def iterate_with_the_next_session(data, close, open_prices):
        decision_ts = pd.Timestamp(strategy_obj.current_bar)
        # *** CRITICAL*** a calendar label only, as in the live adapter: no price after Close_T is read.
        strategy_obj.current_bar = session_index[session_index.searchsorted(decision_ts, side="right")]
        try:
            return pod_iterate_fn(data, close, open_prices)
        finally:
            strategy_obj.current_bar = decision_ts

    strategy_obj.iterate = iterate_with_the_next_session
    return strategy_obj
