"""Live HPI same-open slot refill for both HPI pods (owner decisions 2026-09-28).

The backtest frees an exit's slot at Open_(T+1) and refills it in the same
auction. These tests pin the live decision plan to that rule and cover the
edge cases around it: several exits and candidates, a candidate that is being
exited, an exit that did not fill, partial exit fills, a held name without a
bar on T, idempotent re-runs, the MOO basket order and the release kill
switch. Every plan test runs for the 2/3/5-vote pod and the baseline pod.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

import alpha.live.strategy_host as strategy_host_module
from alpha.live.execution_engine import build_broker_order_request_list_from_vplan, build_vplan
from alpha.live.models import BrokerSnapshot, LivePriceSnapshot, LiveRelease, PodState
from alpha.live.release_manifest import select_enabled_release_list_for_mode
from alpha.live.strategy_host import build_decision_plan_for_release


MARKET_TIMEZONE_OBJ = ZoneInfo("America/New_York")
VOTE_IMPORT_STR = "strategies.hpi.strategy_mr_hpi_sp500_2_3_5_vote"
BASELINE_IMPORT_STR = "strategies.hpi.strategy_mr_hpi_sp500_ibs_rsi_exit"
DATE_IDX = pd.bdate_range("2023-01-02", periods=260)
SIGNAL_DATE_TS = DATE_IDX[-1]

# Feature rows used on the signal date. Entry: all three horizons vote, IBS
# below 0.10, Close above SMA200. Neutral: no entry and no exit.
ENTRY_FEATURE_DICT = {
    "ibs_value_ser": 0.05,
    "rsi2_value_ser": 10.0,
    "return_2d_ser": -0.01,
    "return_3d_ser": -0.01,
    "return_5d_ser": -0.01,
    "hpi_2d_ser": 20.0,
    "hpi_value_ser": 20.0,
    "hpi_5d_ser": 20.0,
    "sma_200_price_ser": 90.0,
}
NEUTRAL_FEATURE_DICT = {
    **ENTRY_FEATURE_DICT,
    "ibs_value_ser": 0.50,
    "rsi2_value_ser": 50.0,
    "return_2d_ser": 0.01,
    "return_3d_ser": 0.01,
    "return_5d_ser": 0.01,
    "hpi_2d_ser": 80.0,
    "hpi_value_ser": 80.0,
    "hpi_5d_ser": 80.0,
}
IBS_EXIT_FEATURE_DICT = {**NEUTRAL_FEATURE_DICT, "ibs_value_ser": 0.95}
# RSI2 exit while every entry filter also passes: the name is both an exit
# and a ranked candidate on the same day.
RSI_EXIT_AND_ENTRY_FEATURE_DICT = {**ENTRY_FEATURE_DICT, "rsi2_value_ser": 95.0}


def _make_release(strategy_import_str: str, max_positions_int: int) -> LiveRelease:
    return LiveRelease(
        release_id_str="release::hpi_same_open_refill",
        user_id_str="user_001",
        pod_id_str="pod_hpi_001",
        account_route_str="DU1",
        strategy_import_str=strategy_import_str,
        mode_str="paper",
        session_calendar_id_str="XNYS",
        signal_clock_str="eod_snapshot_ready",
        execution_policy_str="next_open_moo",
        data_profile_str="norgate_eod_sp500_hpi_pit",
        params_dict={
            "capital_base_float": 100_000.0,
            "max_positions_int": max_positions_int,
        },
        risk_profile_str="standard",
        enabled_bool=True,
        source_path_str="manifest.yaml",
    )


def _make_pod_state(
    release_obj: LiveRelease,
    position_amount_map: dict[str, float],
    pending_exit_symbol_list: list[str],
) -> PodState:
    return PodState(
        pod_id_str=release_obj.pod_id_str,
        user_id_str=release_obj.user_id_str,
        account_route_str=release_obj.account_route_str,
        position_amount_map=dict(position_amount_map),
        cash_float=1_000.0,
        total_value_float=90_000.0,
        strategy_state_dict={
            "trade_id_int": 40,
            "current_trade_map": {
                symbol_str: 30 + symbol_idx_int
                for symbol_idx_int, symbol_str in enumerate(sorted(position_amount_map))
            },
            "pending_exit_symbol_list": list(pending_exit_symbol_list),
        },
        updated_timestamp_ts=datetime(2023, 12, 28, 21, 0, tzinfo=UTC),
    )


def _install_hpi_inputs(
    monkeypatch,
    feature_by_symbol_dict: dict[str, dict[str, float]],
    turnover_by_symbol_dict: dict[str, float],
    member_symbol_list: list[str] | None = None,
) -> None:
    """Stub the HPI loader and signal step with fixed signal-date features.

    The real get_opportunity_list(), iterate() and plan mapping still run.
    """

    import strategies.hpi.stateful_long as hpi_module

    monkeypatch.setattr(strategy_host_module, "HPI_MINIMUM_READY_MEMBER_COUNT_INT", 1)
    monkeypatch.setattr(strategy_host_module, "HPI_MINIMUM_READY_MEMBER_RATIO_FLOAT", 0.5)

    symbol_list = sorted(feature_by_symbol_dict)
    price_column_dict: dict[tuple[str, str], pd.Series] = {}
    for symbol_idx_int, symbol_str in enumerate(symbol_list + ["$SPXTR"]):
        close_ser = pd.Series(
            100.0 + symbol_idx_int + np.arange(len(DATE_IDX), dtype=float) * 0.2,
            index=DATE_IDX,
        )
        price_column_dict[(symbol_str, "Open")] = close_ser - 0.1
        price_column_dict[(symbol_str, "High")] = close_ser + 0.5
        price_column_dict[(symbol_str, "Low")] = close_ser - 0.5
        price_column_dict[(symbol_str, "Close")] = close_ser
    pricing_data_df = pd.DataFrame(price_column_dict, index=DATE_IDX)

    member_symbol_list = symbol_list if member_symbol_list is None else member_symbol_list
    universe_df = pd.DataFrame(0, index=DATE_IDX, columns=symbol_list)
    universe_df[member_symbol_list] = 1

    def compute_signals_stub(self, pricing_data_df):
        feature_column_dict: dict[tuple[str, str], pd.Series] = {}
        for symbol_str in symbol_list:
            for field_str, value_float in feature_by_symbol_dict[symbol_str].items():
                feature_column_dict[(symbol_str, field_str)] = pd.Series(
                    value_float,
                    index=DATE_IDX,
                    dtype=float,
                )
            feature_column_dict[(symbol_str, "Turnover")] = pd.Series(
                turnover_by_symbol_dict[symbol_str],
                index=DATE_IDX,
                dtype=float,
            )
        feature_df = pd.DataFrame(feature_column_dict, index=DATE_IDX)
        return pd.concat([pricing_data_df.copy(), feature_df], axis=1)

    monkeypatch.setattr(
        hpi_module,
        "load_exact_hpi_inputs",
        lambda **_kwargs: (symbol_list, universe_df, pricing_data_df),
    )
    monkeypatch.setattr(
        hpi_module.HPIStatefulLongStrategy,
        "compute_signals",
        compute_signals_stub,
    )


def _build_plan(
    strategy_import_str: str,
    max_positions_int: int,
    position_amount_map: dict[str, float],
    pending_exit_symbol_list: list[str] | None = None,
    pod_state_obj: PodState | None = None,
):
    release_obj = _make_release(strategy_import_str, max_positions_int)
    if pod_state_obj is None:
        pod_state_obj = _make_pod_state(
            release_obj,
            position_amount_map,
            pending_exit_symbol_list or [],
        )
    return build_decision_plan_for_release(
        release_obj=release_obj,
        as_of_ts=datetime(2024, 1, 2, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ),
        pod_state_obj=pod_state_obj,
    )


def _full_book_features() -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    feature_by_symbol_dict = {
        "AAA": IBS_EXIT_FEATURE_DICT,
        "BBB": NEUTRAL_FEATURE_DICT,
        "CCC": NEUTRAL_FEATURE_DICT,
        "DDD": ENTRY_FEATURE_DICT,
        "EEE": ENTRY_FEATURE_DICT,
    }
    turnover_by_symbol_dict = {
        "AAA": 9.0e9,
        "BBB": 8.0e9,
        "CCC": 7.0e9,
        "DDD": 5.0e9,
        "EEE": 3.0e9,
    }
    return feature_by_symbol_dict, turnover_by_symbol_dict


FULL_BOOK_POSITION_MAP = {"AAA": 10.0, "BBB": 20.0, "CCC": 30.0}


@pytest.fixture(params=[VOTE_IMPORT_STR, BASELINE_IMPORT_STR], ids=["vote", "baseline"])
def pod_import_str(request) -> str:
    return request.param


def test_full_book_exit_refills_its_slot_in_the_same_batch(monkeypatch, pod_import_str):
    feature_by_symbol_dict, turnover_by_symbol_dict = _full_book_features()
    _install_hpi_inputs(monkeypatch, feature_by_symbol_dict, turnover_by_symbol_dict)

    decision_plan_obj = _build_plan(pod_import_str, 3, FULL_BOOK_POSITION_MAP)

    # Backtest rule: AAA exits at Open_(T+1) and its slot is reused there.
    assert decision_plan_obj.exit_asset_set == {"AAA"}
    assert decision_plan_obj.entry_priority_list == ["DDD"]
    # Sizing unchanged: previous_total_value / max_positions_int.
    assert decision_plan_obj.entry_target_weight_map_dict == pytest.approx({"DDD": 1.0 / 3.0})
    assert decision_plan_obj.decision_base_position_map == FULL_BOOK_POSITION_MAP
    assert decision_plan_obj.strategy_state_dict["pending_exit_symbol_list"] == ["AAA"]
    assert decision_plan_obj.strategy_state_dict["trade_id_int"] == 41
    assert decision_plan_obj.strategy_state_dict["current_trade_map"]["DDD"] == 41
    assert decision_plan_obj.snapshot_metadata_dict["hpi_exit_slot_reuse_str"] == "same_open_moo_batch"


def test_several_exits_refill_with_ranked_candidates(monkeypatch, pod_import_str):
    feature_by_symbol_dict = {
        "AAA": IBS_EXIT_FEATURE_DICT,
        "BBB": {**NEUTRAL_FEATURE_DICT, "rsi2_value_ser": 95.0},
        "CCC": NEUTRAL_FEATURE_DICT,
        "DDD": ENTRY_FEATURE_DICT,
        "EEE": ENTRY_FEATURE_DICT,
        "FFF": ENTRY_FEATURE_DICT,
    }
    turnover_by_symbol_dict = {
        "AAA": 1.0e9,
        "BBB": 1.0e9,
        "CCC": 1.0e9,
        "DDD": 5.0e9,
        "EEE": 7.0e9,
        "FFF": 7.0e9,
    }
    _install_hpi_inputs(monkeypatch, feature_by_symbol_dict, turnover_by_symbol_dict)

    decision_plan_obj = _build_plan(pod_import_str, 3, FULL_BOOK_POSITION_MAP)

    assert decision_plan_obj.exit_asset_set == {"AAA", "BBB"}
    # Turnover descending, symbol ascending on ties; two freed slots.
    assert decision_plan_obj.entry_priority_list == ["EEE", "FFF"]
    assert decision_plan_obj.entry_target_weight_map_dict == pytest.approx(
        {"EEE": 1.0 / 3.0, "FFF": 1.0 / 3.0}
    )


def test_candidate_that_is_being_exited_is_not_reentered(monkeypatch, pod_import_str):
    feature_by_symbol_dict = {
        "AAA": RSI_EXIT_AND_ENTRY_FEATURE_DICT,
        "BBB": NEUTRAL_FEATURE_DICT,
        "CCC": NEUTRAL_FEATURE_DICT,
        "DDD": ENTRY_FEATURE_DICT,
    }
    turnover_by_symbol_dict = {"AAA": 9.0e9, "BBB": 1.0e9, "CCC": 1.0e9, "DDD": 5.0e9}
    _install_hpi_inputs(monkeypatch, feature_by_symbol_dict, turnover_by_symbol_dict)

    decision_plan_obj = _build_plan(pod_import_str, 3, FULL_BOOK_POSITION_MAP)

    # AAA ranks first but is still held at Close_T, so its freed slot goes to
    # the next candidate, exactly as in the backtest.
    assert decision_plan_obj.exit_asset_set == {"AAA"}
    assert decision_plan_obj.entry_priority_list == ["DDD"]


@pytest.mark.parametrize(
    ("position_amount_map", "pending_exit_symbol_list", "extra_exit_bool", "expected_exit_set", "expected_entry_list"),
    [
        # Yesterday's AAA exit did not print; its replacement DDD filled.
        # Four names against three slots: retry AAA, add nothing.
        ({"AAA": 10.0, "BBB": 20.0, "CCC": 30.0, "DDD": 40.0}, ["AAA"], False, {"AAA"}, []),
        # Same, plus a new BBB exit: exactly one slot is refilled.
        ({"AAA": 10.0, "BBB": 20.0, "CCC": 30.0, "DDD": 40.0}, ["AAA"], True, {"AAA", "BBB"}, ["EEE"]),
        # Partial exit fill: 3 residual AAA shares are retried; no new entry.
        ({"AAA": 3.0, "BBB": 20.0, "CCC": 30.0, "DDD": 40.0}, ["AAA"], False, {"AAA"}, []),
        # Two stuck exits: five names against three slots; retry both, add nothing.
        (
            {"AAA": 10.0, "BBB": 20.0, "CCC": 30.0, "DDD": 40.0, "FFF": 50.0},
            ["AAA", "BBB"],
            False,
            {"AAA", "BBB"},
            [],
        ),
        # A stale pending name that is no longer held frees no slot.
        ({"BBB": 20.0, "CCC": 30.0, "DDD": 40.0}, ["ZZZ"], False, set(), []),
    ],
    ids=[
        "stuck_exit_only",
        "stuck_exit_plus_new_exit",
        "partial_exit_residual",
        "two_stuck_exits",
        "stale_pending_not_held",
    ],
)
def test_unfilled_exit_is_retried_without_extra_entries(
    monkeypatch,
    pod_import_str,
    position_amount_map,
    pending_exit_symbol_list,
    extra_exit_bool,
    expected_exit_set,
    expected_entry_list,
):
    feature_by_symbol_dict = {
        "AAA": NEUTRAL_FEATURE_DICT,
        "BBB": IBS_EXIT_FEATURE_DICT if extra_exit_bool else NEUTRAL_FEATURE_DICT,
        "CCC": NEUTRAL_FEATURE_DICT,
        "DDD": NEUTRAL_FEATURE_DICT,
        "EEE": ENTRY_FEATURE_DICT,
        "FFF": NEUTRAL_FEATURE_DICT,
    }
    turnover_by_symbol_dict = {
        "AAA": 1.0e9,
        "BBB": 1.0e9,
        "CCC": 1.0e9,
        "DDD": 1.0e9,
        "EEE": 7.0e9,
        "FFF": 5.0e9,
    }
    _install_hpi_inputs(monkeypatch, feature_by_symbol_dict, turnover_by_symbol_dict)

    decision_plan_obj = _build_plan(
        pod_import_str,
        3,
        position_amount_map,
        pending_exit_symbol_list,
    )

    assert decision_plan_obj.exit_asset_set == expected_exit_set
    assert decision_plan_obj.entry_priority_list == expected_entry_list


def _next_pod_state(
    pod_state_obj: PodState,
    decision_plan_obj,
    failed_exit_symbol_set: set[str],
) -> PodState:
    """Apply one open: exits fill unless listed as failed; every entry fills."""

    position_amount_map = {
        symbol_str: amount_float
        for symbol_str, amount_float in pod_state_obj.position_amount_map.items()
        if symbol_str not in decision_plan_obj.exit_asset_set
        or symbol_str in failed_exit_symbol_set
    }
    for symbol_str in decision_plan_obj.entry_priority_list:
        position_amount_map[symbol_str] = 5.0
    return dataclasses.replace(
        pod_state_obj,
        position_amount_map=position_amount_map,
        strategy_state_dict=dict(decision_plan_obj.strategy_state_dict),
    )


def test_stuck_exits_bound_the_book_across_cycles(monkeypatch, pod_import_str):
    """Closed loop: each plan's outcome is the next cycle's pod state."""

    turnover_by_symbol_dict = {
        "AAA": 1.0e9,
        "BBB": 1.0e9,
        "CCC": 1.0e9,
        "DDD": 5.0e9,
        "EEE": 3.0e9,
    }
    cycle_list = [
        # (features, exits that fail at the open, expected exits, expected entries)
        (
            {"AAA": IBS_EXIT_FEATURE_DICT, "DDD": ENTRY_FEATURE_DICT, "EEE": ENTRY_FEATURE_DICT},
            {"AAA"},
            {"AAA"},
            ["DDD"],
        ),
        ({"EEE": ENTRY_FEATURE_DICT}, {"AAA"}, {"AAA"}, []),
        (
            {"BBB": IBS_EXIT_FEATURE_DICT, "EEE": ENTRY_FEATURE_DICT},
            {"AAA", "BBB"},
            {"AAA", "BBB"},
            ["EEE"],
        ),
        ({}, set(), {"AAA", "BBB"}, []),
    ]
    release_obj = _make_release(pod_import_str, 3)
    pod_state_obj = _make_pod_state(release_obj, FULL_BOOK_POSITION_MAP, [])
    held_count_list: list[int] = []
    for cycle_feature_dict, failed_exit_symbol_set, expected_exit_set, expected_entry_list in cycle_list:
        feature_by_symbol_dict = {
            symbol_str: cycle_feature_dict.get(symbol_str, NEUTRAL_FEATURE_DICT)
            for symbol_str in turnover_by_symbol_dict
        }
        _install_hpi_inputs(monkeypatch, feature_by_symbol_dict, turnover_by_symbol_dict)
        decision_plan_obj = _build_plan(pod_import_str, 3, {}, pod_state_obj=pod_state_obj)

        assert decision_plan_obj.exit_asset_set == expected_exit_set
        assert decision_plan_obj.entry_priority_list == expected_entry_list
        pod_state_obj = _next_pod_state(pod_state_obj, decision_plan_obj, failed_exit_symbol_set)
        held_count_int = len(pod_state_obj.position_amount_map)
        # Never more than the slot count plus the exits stuck right now.
        assert held_count_int <= 3 + len(failed_exit_symbol_set)
        held_count_list.append(held_count_int)

    assert held_count_list == [4, 4, 5, 3]
    assert sorted(pod_state_obj.position_amount_map) == ["CCC", "DDD", "EEE"]


def test_held_name_without_features_is_not_exited_by_the_marker(monkeypatch, pod_import_str):
    # The tradability marker alone never creates an exit: a held, non-pending
    # name with no bar on T keeps its slot, so a full book adds nothing.
    feature_by_symbol_dict = {
        "AAA": {field_str: np.nan for field_str in NEUTRAL_FEATURE_DICT},
        "BBB": NEUTRAL_FEATURE_DICT,
        "CCC": NEUTRAL_FEATURE_DICT,
        "DDD": ENTRY_FEATURE_DICT,
    }
    turnover_by_symbol_dict = {"AAA": np.nan, "BBB": 1.0e9, "CCC": 1.0e9, "DDD": 5.0e9}
    _install_hpi_inputs(monkeypatch, feature_by_symbol_dict, turnover_by_symbol_dict)

    decision_plan_obj = _build_plan(pod_import_str, 3, FULL_BOOK_POSITION_MAP)

    assert decision_plan_obj.exit_asset_set == set()
    assert decision_plan_obj.entry_priority_list == []


def test_partly_empty_book_adds_free_and_freed_slots(monkeypatch, pod_import_str):
    # Five slots, two held, one exiting: 3 free + 1 freed = 4 slots, but only
    # three candidates exist.
    feature_by_symbol_dict = {
        "AAA": IBS_EXIT_FEATURE_DICT,
        "BBB": NEUTRAL_FEATURE_DICT,
        "DDD": ENTRY_FEATURE_DICT,
        "EEE": ENTRY_FEATURE_DICT,
        "FFF": ENTRY_FEATURE_DICT,
    }
    turnover_by_symbol_dict = {"AAA": 1.0e9, "BBB": 1.0e9, "DDD": 5.0e9, "EEE": 7.0e9, "FFF": 6.0e9}
    _install_hpi_inputs(monkeypatch, feature_by_symbol_dict, turnover_by_symbol_dict)

    decision_plan_obj = _build_plan(pod_import_str, 5, {"AAA": 10.0, "BBB": 20.0})

    assert decision_plan_obj.exit_asset_set == {"AAA"}
    assert decision_plan_obj.entry_priority_list == ["EEE", "FFF", "DDD"]
    assert decision_plan_obj.entry_target_weight_map_dict == pytest.approx(
        {"EEE": 0.2, "FFF": 0.2, "DDD": 0.2}
    )


def test_pending_exit_without_a_bar_on_t_is_still_exited_and_refilled(monkeypatch, pod_import_str):
    # AAA had no bar on T (halted); its features are NaN at Close_T. It was
    # already a pending exit, so live submits it and assumes Open_(T+1) prints.
    feature_by_symbol_dict = {
        "AAA": {field_str: np.nan for field_str in NEUTRAL_FEATURE_DICT},
        "BBB": NEUTRAL_FEATURE_DICT,
        "CCC": NEUTRAL_FEATURE_DICT,
        "DDD": ENTRY_FEATURE_DICT,
    }
    turnover_by_symbol_dict = {"AAA": np.nan, "BBB": 1.0e9, "CCC": 1.0e9, "DDD": 5.0e9}
    _install_hpi_inputs(monkeypatch, feature_by_symbol_dict, turnover_by_symbol_dict)

    decision_plan_obj = _build_plan(pod_import_str, 3, FULL_BOOK_POSITION_MAP, ["AAA"])

    assert decision_plan_obj.exit_asset_set == {"AAA"}
    assert decision_plan_obj.entry_priority_list == ["DDD"]


def test_membership_loss_exit_refills_its_slot(monkeypatch, pod_import_str):
    feature_by_symbol_dict, turnover_by_symbol_dict = _full_book_features()
    feature_by_symbol_dict["AAA"] = NEUTRAL_FEATURE_DICT
    _install_hpi_inputs(
        monkeypatch,
        feature_by_symbol_dict,
        turnover_by_symbol_dict,
        member_symbol_list=["BBB", "CCC", "DDD", "EEE"],
    )

    decision_plan_obj = _build_plan(pod_import_str, 3, FULL_BOOK_POSITION_MAP)

    assert decision_plan_obj.exit_asset_set == {"AAA"}
    assert decision_plan_obj.entry_priority_list == ["DDD"]


def test_rerun_of_the_same_cycle_is_identical_and_does_not_mutate_state(monkeypatch, pod_import_str):
    feature_by_symbol_dict, turnover_by_symbol_dict = _full_book_features()
    _install_hpi_inputs(monkeypatch, feature_by_symbol_dict, turnover_by_symbol_dict)
    release_obj = _make_release(pod_import_str, 3)
    pod_state_obj = _make_pod_state(release_obj, FULL_BOOK_POSITION_MAP, [])
    pod_state_before_obj = copy.deepcopy(pod_state_obj)

    first_plan_obj = _build_plan(pod_import_str, 3, {}, pod_state_obj=pod_state_obj)
    second_plan_obj = _build_plan(pod_import_str, 3, {}, pod_state_obj=pod_state_obj)

    assert first_plan_obj == second_plan_obj
    assert first_plan_obj.strategy_state_dict["trade_id_int"] == 41
    assert pod_state_obj == pod_state_before_obj


def test_plan_does_not_depend_on_the_open_marker_value(monkeypatch, pod_import_str):
    feature_by_symbol_dict, turnover_by_symbol_dict = _full_book_features()
    _install_hpi_inputs(monkeypatch, feature_by_symbol_dict, turnover_by_symbol_dict)
    reference_plan_obj = _build_plan(pod_import_str, 3, FULL_BOOK_POSITION_MAP)

    # The marker is a tradability flag. If a future change used it as a price,
    # the plan would move with it and this test would fail.
    monkeypatch.setattr(strategy_host_module, "HPI_LIVE_TRADABLE_OPEN_MARKER_FLOAT", 987.65)
    marker_plan_obj = _build_plan(pod_import_str, 3, FULL_BOOK_POSITION_MAP)

    assert marker_plan_obj == reference_plan_obj


def test_vplan_sends_exit_and_refill_in_one_moo_basket_exits_first(monkeypatch, pod_import_str):
    feature_by_symbol_dict, turnover_by_symbol_dict = _full_book_features()
    _install_hpi_inputs(monkeypatch, feature_by_symbol_dict, turnover_by_symbol_dict)
    release_obj = _make_release(pod_import_str, 3)
    decision_plan_obj = _build_plan(pod_import_str, 3, FULL_BOOK_POSITION_MAP)
    snapshot_ts = datetime(2024, 1, 2, 14, 20, tzinfo=UTC)
    broker_snapshot_obj = BrokerSnapshot(
        account_route_str="DU1",
        snapshot_timestamp_ts=snapshot_ts,
        cash_float=0.0,
        total_value_float=90_000.0,
        net_liq_float=90_000.0,
        position_amount_map=dict(FULL_BOOK_POSITION_MAP),
    )
    live_price_snapshot_obj = LivePriceSnapshot(
        account_route_str="DU1",
        snapshot_timestamp_ts=snapshot_ts,
        price_source_str="stub",
        asset_reference_price_map={"AAA": 3_000.0, "DDD": 150.0},
    )

    vplan_obj = build_vplan(
        release_obj=release_obj,
        decision_plan_obj=decision_plan_obj,
        broker_snapshot_obj=broker_snapshot_obj,
        live_price_snapshot_obj=live_price_snapshot_obj,
    )
    order_request_list = build_broker_order_request_list_from_vplan(vplan_obj)

    # One MOO basket (one submission key), the sale listed before the buy.
    expected_entry_share_float = float(
        math.floor(
            decision_plan_obj.entry_target_weight_map_dict["DDD"]
            * vplan_obj.pod_budget_float
            / 150.0
        )
    )
    assert expected_entry_share_float > 0.0
    assert [request_obj.asset_str for request_obj in order_request_list] == ["AAA", "DDD"]
    assert [request_obj.amount_float for request_obj in order_request_list] == [
        -10.0,
        expected_entry_share_float,
    ]
    assert {request_obj.broker_order_type_str for request_obj in order_request_list} == {"MOO"}
    assert len({request_obj.submission_key_str for request_obj in order_request_list}) == 1


def test_disabled_hpi_release_is_never_selected():
    enabled_release_obj = _make_release(VOTE_IMPORT_STR, 10)
    disabled_release_obj = dataclasses.replace(enabled_release_obj, enabled_bool=False)

    assert select_enabled_release_list_for_mode([disabled_release_obj], "paper") == []
    assert select_enabled_release_list_for_mode([enabled_release_obj], "paper") == [enabled_release_obj]
