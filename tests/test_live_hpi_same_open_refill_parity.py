"""Real-data parity: wired HPI vote and shared baseline helper versus backtest.

For every session T in each window, the backtest's own state at Close_T
(positions, pending exits, trade ids, previous total value) is seeded into the
live host, which then builds a DecisionPlan for Open_(T+1). Its exits and
ranked entries must equal the orders the backtest placed in iterate().

The one expected difference is a pending exit whose Open_(T+1) did not print
(halt or removal): the backtest knows that ex post and keeps the slot; live
cannot know it and refills. Such days are counted and must be the only
mismatches, and even then exits, ranking and weights must otherwise agree.

Limits: each day is re-seeded from the backtest state (open loop), so path
effects are not tested here. Both sides share one signal frame built on the
window's member symbols; that is decision-neutral for both HPI pods (per-stock
time-series features, LIQUIDITY_NONE), but the live loader and full-universe
readiness gate are covered only by the unit tests.

Opt-in because it needs real Norgate data (several minutes, several GB RAM):

    ALPHA_RUN_HPI_LIVE_PARITY_BOOL=true uv run pytest -s tests/test_live_hpi_same_open_refill_parity.py

Set ALPHA_HPI_PARITY_INPUTS_PICKLE to a pickle with keys "universe" and
"pricing_df" (the load_exact_hpi_inputs output, e.g. the 2026-09-27 leakage
hunt cache) to skip the Norgate load.
"""

from __future__ import annotations

import functools
import os
import pickle
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest


RUN_PARITY_BOOL = os.getenv("ALPHA_RUN_HPI_LIVE_PARITY_BOOL", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

pytestmark = pytest.mark.skipif(
    not RUN_PARITY_BOOL,
    reason=(
        "Live HPI parity needs real Norgate data. "
        "Set ALPHA_RUN_HPI_LIVE_PARITY_BOOL=true to run."
    ),
)

MARKET_TIMEZONE_OBJ = ZoneInfo("America/New_York")
VOTE_IMPORT_STR = "strategies.hpi.strategy_mr_hpi_sp500_2_3_5_vote"
BASELINE_IMPORT_STR = "strategies.hpi.strategy_mr_hpi_sp500_ibs_rsi_exit"
# (live release import string, backtest entry mode) for each live HPI pod.
POD_LIST = [
    pytest.param(VOTE_IMPORT_STR, "hpi_2_3_5_vote", id="vote"),
    pytest.param(BASELINE_IMPORT_STR, "baseline", id="baseline"),
]
BENCHMARK_SYMBOL_STR = "$SPXTR"
# The live host maps T to the next XNYS session with exchange_calendars, whose
# default calendar starts 20 years back (2006-09-28 today), so windows must
# start after that date.
PARITY_WINDOW_LIST = [
    ("2008-09-02", "2008-12-31"),
    ("2020-02-18", "2020-05-29"),
    ("2025-03-03", "2025-05-30"),
]


@functools.cache
def _load_hpi_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    inputs_pickle_path_str = os.getenv("ALPHA_HPI_PARITY_INPUTS_PICKLE")
    if inputs_pickle_path_str:
        with open(inputs_pickle_path_str, "rb") as handle_obj:
            inputs_dict = pickle.load(handle_obj)
        return inputs_dict["universe"], inputs_dict["pricing_df"]

    from strategies.hpi.stateful_long import load_exact_hpi_inputs

    _, universe_df, pricing_df = load_exact_hpi_inputs(
        indexname_str="S&P 500",
        benchmark_symbol_str=BENCHMARK_SYMBOL_STR,
        start_date_str="1998-01-01",
        end_date_str=None,
    )
    return universe_df, pricing_df


def _window_pricing_df(
    pricing_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    start_date_str: str,
    end_date_str: str,
) -> pd.DataFrame:
    """Keep symbols that are PIT members at any time in the window.

    Decision-neutral for HPI: entries require membership, ranking uses only
    members, and no statistic is taken across non-members.
    """

    window_universe_df = universe_df.loc[
        pd.Timestamp(start_date_str) - pd.Timedelta(days=30) : pd.Timestamp(end_date_str)
    ]
    member_symbol_set = set(
        window_universe_df.columns[window_universe_df.sum(axis=0) > 0].astype(str)
    )
    member_symbol_set.add(BENCHMARK_SYMBOL_STR)
    keep_column_list = [
        column_tuple
        for column_tuple in pricing_df.columns
        if str(column_tuple[0]) in member_symbol_set
    ]
    window_pricing_df = pricing_df.loc[: pd.Timestamp(end_date_str), keep_column_list].copy()
    window_pricing_df.attrs["norgate_adjustment_by_symbol_dict"] = {
        str(symbol_str): (
            "TOTALRETURN" if symbol_str == BENCHMARK_SYMBOL_STR else "CAPITALSPECIAL"
        )
        for symbol_str in window_pricing_df.columns.get_level_values(0).unique()
    }
    return window_pricing_df


def _run_recording_backtest(
    window_pricing_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    start_date_str: str,
    entry_mode_str: str,
) -> tuple[list[dict], pd.DataFrame]:
    import strategies.hpi.stateful_long as hpi_module
    from alpha.engine.backtest import run_daily

    class RecordingHPIStrategy(hpi_module.HPIStatefulLongStrategy):
        def iterate(self, data_df, close_row_ser, open_price_ser):
            if data_df is None or close_row_ser is None:
                return super().iterate(data_df, close_row_ser, open_price_ser)
            position_ser = self.get_positions()
            held_position_ser = position_ser[position_ser > 0]
            prior_order_id_set = {id(order_obj) for order_obj in self.get_orders()}
            record_dict = {
                "signal_date_ts": pd.Timestamp(self.previous_bar),
                "position_amount_map": {
                    str(symbol_obj): float(amount_float)
                    for symbol_obj, amount_float in held_position_ser.items()
                },
                "pending_exit_symbol_list": sorted(self.pending_exit_symbol_set),
                "trade_id_int": int(self.trade_id_int),
                "current_trade_map": {
                    str(symbol_str): int(trade_id_int)
                    for symbol_str, trade_id_int in self.current_trade_map.items()
                },
                "previous_total_value_float": float(self.previous_total_value),
                "missing_open_symbol_list": sorted(
                    str(symbol_obj)
                    for symbol_obj in held_position_ser.index
                    if not np.isfinite(float(open_price_ser.get(symbol_obj, np.nan)))
                ),
            }
            super().iterate(data_df, close_row_ser, open_price_ser)
            new_order_list = [
                order_obj
                for order_obj in self.get_orders()
                if id(order_obj) not in prior_order_id_set
            ]
            record_dict["exit_symbol_set"] = {
                str(order_obj.asset)
                for order_obj in new_order_list
                if order_obj.target and abs(float(order_obj.amount)) <= 1e-9
            }
            record_dict["entry_symbol_list"] = [
                str(order_obj.asset)
                for order_obj in new_order_list
                if not order_obj.target and order_obj.unit == "value"
            ]
            record_dict["entry_weight_list"] = [
                float(order_obj.amount) / record_dict["previous_total_value_float"]
                for order_obj in new_order_list
                if not order_obj.target and order_obj.unit == "value"
            ]
            self.parity_record_list.append(record_dict)

    strategy_obj = RecordingHPIStrategy(
        name="hpi_live_parity",
        benchmarks=[BENCHMARK_SYMBOL_STR],
        ranking_field_str=hpi_module.TURNOVER_FIELD_STR,
        entry_mode_str=entry_mode_str,
        backtest_start_date_str=start_date_str,
    )
    strategy_obj.parity_record_list = []
    strategy_obj.universe_df = universe_df
    # The backtest and the live replay use the same causal signal frame.
    full_signal_df = strategy_obj.compute_signals(window_pricing_df.copy())
    calendar_idx = window_pricing_df.index[window_pricing_df.index >= pd.Timestamp(start_date_str)]
    run_daily(
        strategy_obj,
        window_pricing_df,
        calendar_idx,
        show_progress=False,
        show_signal_progress_bool=False,
    )
    return strategy_obj.parity_record_list, full_signal_df


def _build_live_plan(
    monkeypatch,
    record_dict: dict,
    universe_df: pd.DataFrame,
    window_pricing_df: pd.DataFrame,
    full_signal_df: pd.DataFrame,
    pod_import_str: str,
    old_host_rule_bool: bool = False,
):
    import alpha.live.strategy_host as strategy_host_module
    import strategies.hpi.stateful_long as hpi_module
    from alpha.live.models import LiveRelease, PodState

    signal_date_ts = record_dict["signal_date_ts"]
    # The host reads only the last index label of the loaded prices (the
    # signal date); features come from the precomputed causal signal frame.
    clock_df = window_pricing_df.loc[:signal_date_ts, [(BENCHMARK_SYMBOL_STR, "Close")]]
    monkeypatch.setattr(
        hpi_module,
        "load_exact_hpi_inputs",
        lambda **_kwargs: ([], universe_df, clock_df),
    )
    monkeypatch.setattr(
        hpi_module.HPIStatefulLongStrategy,
        "compute_signals",
        lambda self, _pricing_df: full_signal_df.loc[:signal_date_ts],
    )
    release_obj = LiveRelease(
        release_id_str="release::hpi_live_parity",
        user_id_str="user_001",
        pod_id_str="pod_hpi_parity",
        account_route_str="DU1",
        strategy_import_str=pod_import_str,
        mode_str="paper",
        session_calendar_id_str="XNYS",
        signal_clock_str="eod_snapshot_ready",
        execution_policy_str="next_open_moo",
        data_profile_str="norgate_eod_sp500_hpi_pit",
        params_dict={"capital_base_float": 100_000.0, "max_positions_int": 10},
        risk_profile_str="standard",
        enabled_bool=True,
        source_path_str="manifest.yaml",
    )
    pod_state_obj = PodState(
        pod_id_str=release_obj.pod_id_str,
        user_id_str=release_obj.user_id_str,
        account_route_str=release_obj.account_route_str,
        position_amount_map=dict(record_dict["position_amount_map"]),
        cash_float=0.0,
        total_value_float=record_dict["previous_total_value_float"],
        strategy_state_dict={
            "trade_id_int": record_dict["trade_id_int"],
            "current_trade_map": dict(record_dict["current_trade_map"]),
            "pending_exit_symbol_list": list(record_dict["pending_exit_symbol_list"]),
        },
        updated_timestamp_ts=datetime(2000, 1, 1, tzinfo=MARKET_TIMEZONE_OBJ),
    )
    as_of_ts = datetime(2030, 1, 1, tzinfo=MARKET_TIMEZONE_OBJ)

    def build_plan():
        if pod_import_str == BASELINE_IMPORT_STR:
            return strategy_host_module._build_hpi_decision_plan(
                release_obj, as_of_ts, pod_state_obj,
                entry_mode_str="baseline",
                strategy_family_str="hpi_sp500_ibs_rsi_exit",
            )
        return strategy_host_module.build_decision_plan_for_release(
            release_obj, as_of_ts, pod_state_obj
        )

    if not old_host_rule_bool:
        return build_plan()

    # Replica of the pre-2026-09-28 host: iterate() with no opens (no slot
    # freed), then an exit for every pending name still held.
    original_iterate_func = hpi_module.HPIStatefulLongStrategy.iterate

    def old_host_iterate(self, data_df, close_row_ser, _open_price_ser):
        original_iterate_func(self, data_df, close_row_ser, pd.Series(dtype=float))
        position_ser = self.get_positions()
        held_symbol_set = set(position_ser[position_ser > 0.0].index.astype(str))
        for symbol_str in sorted(self.pending_exit_symbol_set & held_symbol_set):
            self.order_target_value(symbol_str, 0.0, trade_id=self.current_trade_map[symbol_str])

    with monkeypatch.context() as old_rule_monkeypatch:
        old_rule_monkeypatch.setattr(
            hpi_module.HPIStatefulLongStrategy,
            "iterate",
            old_host_iterate,
        )
        return build_plan()


def _is_expected_missing_open_mismatch_bool(decision_plan_obj, record_dict: dict) -> bool:
    """Live may differ only by exiting held names whose Open_(T+1) did not print.

    Live exits = backtest exits + those names; backtest entries are a prefix of
    live entries; live adds at most one entry per such name, each at the same
    slot weight.
    """

    stuck_symbol_set = set(record_dict["missing_open_symbol_list"]) & set(
        decision_plan_obj.exit_asset_set
    )
    if not stuck_symbol_set:
        return False
    backtest_entry_list = record_dict["entry_symbol_list"]
    live_entry_list = decision_plan_obj.entry_priority_list
    extra_entry_count_int = len(live_entry_list) - len(backtest_entry_list)
    live_weight_list = [
        decision_plan_obj.entry_target_weight_map_dict[symbol_str]
        for symbol_str in live_entry_list
    ]
    return (
        decision_plan_obj.exit_asset_set == record_dict["exit_symbol_set"] | stuck_symbol_set
        and live_entry_list[: len(backtest_entry_list)] == backtest_entry_list
        and 0 <= extra_entry_count_int <= len(stuck_symbol_set)
        and np.allclose(live_weight_list, 0.1, rtol=0.0, atol=1e-12)
    )


def _plan_matches_record_bool(decision_plan_obj, record_dict: dict) -> bool:
    entry_weight_list = [
        decision_plan_obj.entry_target_weight_map_dict[symbol_str]
        for symbol_str in decision_plan_obj.entry_priority_list
    ]
    return (
        decision_plan_obj.exit_asset_set == record_dict["exit_symbol_set"]
        and decision_plan_obj.entry_priority_list == record_dict["entry_symbol_list"]
        and np.allclose(entry_weight_list, record_dict["entry_weight_list"], rtol=0.0, atol=1e-12)
    )


def _assert_live_plans_reproduce_backtest(
    monkeypatch,
    pod_import_str: str,
    entry_mode_str: str,
    backtest_start_date_str: str,
    compare_start_date_str: str,
    end_date_str: str,
) -> None:
    universe_df, pricing_df = _load_hpi_inputs()
    window_pricing_df = _window_pricing_df(
        pricing_df, universe_df, backtest_start_date_str, end_date_str
    )
    record_list, full_signal_df = _run_recording_backtest(
        window_pricing_df,
        universe_df,
        backtest_start_date_str,
        entry_mode_str,
    )
    record_list = [
        record_dict
        for record_dict in record_list
        if record_dict["signal_date_ts"] >= pd.Timestamp(compare_start_date_str)
    ]

    match_count_int = 0
    refill_day_count_int = 0
    refill_entry_count_int = 0
    old_refill_mismatch_count_int = 0
    expected_mismatch_date_list: list[str] = []
    unexpected_mismatch_list: list[dict] = []
    example_line_list: list[str] = []
    for record_dict in record_list:
        slot_count_int = 10 - len(record_dict["position_amount_map"])
        refill_entry_int = max(0, len(record_dict["entry_symbol_list"]) - max(0, slot_count_int))
        new_plan_obj = _build_live_plan(
            monkeypatch, record_dict, universe_df, window_pricing_df, full_signal_df, pod_import_str
        )
        if _plan_matches_record_bool(new_plan_obj, record_dict):
            match_count_int += 1
        elif _is_expected_missing_open_mismatch_bool(new_plan_obj, record_dict):
            expected_mismatch_date_list.append(record_dict["signal_date_ts"].date().isoformat())
        else:
            unexpected_mismatch_list.append(
                {
                    "signal_date": record_dict["signal_date_ts"].date().isoformat(),
                    "backtest_exits": sorted(record_dict["exit_symbol_set"]),
                    "live_exits": sorted(new_plan_obj.exit_asset_set),
                    "backtest_entries": record_dict["entry_symbol_list"],
                    "live_entries": new_plan_obj.entry_priority_list,
                }
            )
        if refill_entry_int == 0:
            continue
        refill_day_count_int += 1
        refill_entry_count_int += refill_entry_int
        old_plan_obj = _build_live_plan(
            monkeypatch,
            record_dict,
            universe_df,
            window_pricing_df,
            full_signal_df,
            pod_import_str,
            old_host_rule_bool=True,
        )
        if not _plan_matches_record_bool(old_plan_obj, record_dict):
            old_refill_mismatch_count_int += 1
        if len(example_line_list) < 3:
            example_line_list.append(
                f"  {record_dict['signal_date_ts'].date()} held={len(record_dict['position_amount_map'])} "
                f"backtest exits={sorted(record_dict['exit_symbol_set'])} "
                f"entries={record_dict['entry_symbol_list']} | "
                f"old live entries={old_plan_obj.entry_priority_list} | "
                f"new live entries={new_plan_obj.entry_priority_list}"
            )

    print(
        f"\n[{entry_mode_str} {compare_start_date_str}..{end_date_str}] days={len(record_list)} "
        f"match={match_count_int} expected_missing_open_mismatch={len(expected_mismatch_date_list)} "
        f"{expected_mismatch_date_list} unexpected={len(unexpected_mismatch_list)} "
        f"refill_days={refill_day_count_int} refill_entries={refill_entry_count_int} "
        f"old_host_wrong_on_refill_days={old_refill_mismatch_count_int}"
    )
    print("\n".join(example_line_list))

    assert unexpected_mismatch_list == []
    assert refill_day_count_int > 0
    assert old_refill_mismatch_count_int == refill_day_count_int
    assert match_count_int + len(expected_mismatch_date_list) == len(record_list)


@pytest.mark.parametrize(("pod_import_str", "entry_mode_str"), POD_LIST)
@pytest.mark.parametrize(("start_date_str", "end_date_str"), PARITY_WINDOW_LIST)
def test_live_plan_reproduces_backtest_orders(
    monkeypatch, pod_import_str, entry_mode_str, start_date_str, end_date_str
):
    _assert_live_plans_reproduce_backtest(
        monkeypatch,
        pod_import_str,
        entry_mode_str,
        start_date_str,
        start_date_str,
        end_date_str,
    )


@pytest.mark.skipif(
    os.getenv("ALPHA_RUN_HPI_LIVE_PARITY_FULL_HISTORY_BOOL", "false").lower()
    not in {"1", "true", "yes", "on"},
    reason="Full-history replay takes about 12 minutes per pod; set ALPHA_RUN_HPI_LIVE_PARITY_FULL_HISTORY_BOOL=true.",
)
@pytest.mark.parametrize(("pod_import_str", "entry_mode_str"), POD_LIST)
def test_live_plan_reproduces_full_history(monkeypatch, pod_import_str, entry_mode_str):
    # Backtest path from the production start, so state is the real one; the
    # comparison starts where the live host's XNYS calendar starts.
    _, pricing_df = _load_hpi_inputs()
    _assert_live_plans_reproduce_backtest(
        monkeypatch,
        pod_import_str,
        entry_mode_str,
        "2004-01-02",
        "2006-10-02",
        str(pricing_df.index[-1].date()),
    )
