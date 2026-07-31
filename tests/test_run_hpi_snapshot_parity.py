from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.research import run_hpi_snapshot_parity as parity


def _run_replay(
    monkeypatch,
    *,
    open_by_symbol_dict: dict[str, list[float]],
    universe_by_symbol_dict: dict[str, list[int]],
    candidate_by_date_dict: dict[str, list[str]],
    ibs_by_symbol_dict: dict[str, list[float]] | None = None,
) -> dict:
    date_idx = pd.bdate_range("2024-01-02", periods=4)
    symbol_list = sorted(open_by_symbol_dict)
    pricing_column_dict: dict[tuple[str, str], list[float]] = {}
    signal_column_dict: dict[tuple[str, str], list[float]] = {}
    for symbol_str in symbol_list:
        open_list = open_by_symbol_dict[symbol_str]
        pricing_column_dict[(symbol_str, "Open")] = open_list
        pricing_column_dict[(symbol_str, "Close")] = [
            100.0 + date_position_int
            for date_position_int in range(len(date_idx))
        ]
        signal_column_dict[(symbol_str, "ibs_value_ser")] = (
            ibs_by_symbol_dict or {}
        ).get(symbol_str, [0.1] * len(date_idx))
        signal_column_dict[(symbol_str, "rsi2_value_ser")] = [
            10.0
        ] * len(date_idx)

    pricing_data_df = pd.DataFrame(
        pricing_column_dict,
        index=date_idx,
    )
    signal_df = pd.DataFrame(signal_column_dict, index=date_idx)
    universe_df = pd.DataFrame(
        universe_by_symbol_dict,
        index=date_idx,
    )
    monkeypatch.setattr(
        parity,
        "_build_candidate_map_dict",
        lambda **_: candidate_by_date_dict,
    )
    runtime_dict = {
        "exit_ibs_threshold_float": 0.5,
        "exit_rsi2_threshold_float": 70.0,
        "max_positions_int": 1,
    }
    return parity._run_arm_variant(
        arm_str="synthetic",
        variant_str="baseline",
        entry_mode_str="baseline",
        pricing_data_df=pricing_data_df,
        signal_df=signal_df,
        universe_df=universe_df,
        runtime_dict=runtime_dict,
        backtest_start_date_str=date_idx[1].date().isoformat(),
        snapshot_date_str=date_idx[-1].date().isoformat(),
    )


def _event_tuple_list(result_dict: dict) -> list[tuple[str, str, str, float]]:
    return [
        (
            row_dict["bar_date_str"],
            row_dict["asset_str"],
            row_dict["direction_str"],
            row_dict["price_float"],
        )
        for row_dict in result_dict["transaction_row_list"]
    ]


def test_replay_persists_exit_until_next_tradable_open(monkeypatch):
    date_idx = pd.bdate_range("2024-01-02", periods=4)
    result_dict = _run_replay(
        monkeypatch,
        open_by_symbol_dict={"OLD": [100.0, 101.0, np.nan, 103.0]},
        universe_by_symbol_dict={"OLD": [1, 1, 1, 1]},
        candidate_by_date_dict={
            date_idx[0].date().isoformat(): ["OLD"],
        },
        ibs_by_symbol_dict={"OLD": [0.1, 0.9, 0.1, 0.1]},
    )

    assert _event_tuple_list(result_dict) == [
        (date_idx[1].date().isoformat(), "OLD", "buy", 101.0),
        (date_idx[3].date().isoformat(), "OLD", "sell", 103.0),
    ]


def test_replay_forces_removed_missing_open_at_prior_close(monkeypatch):
    date_idx = pd.bdate_range("2024-01-02", periods=4)
    result_dict = _run_replay(
        monkeypatch,
        open_by_symbol_dict={"OLD": [100.0, 101.0, np.nan, np.nan]},
        universe_by_symbol_dict={"OLD": [1, 1, 0, 0]},
        candidate_by_date_dict={
            date_idx[0].date().isoformat(): ["OLD"],
        },
    )

    assert _event_tuple_list(result_dict) == [
        (date_idx[1].date().isoformat(), "OLD", "buy", 101.0),
        (date_idx[2].date().isoformat(), "OLD", "sell", 101.0),
    ]


def test_replay_does_not_substitute_for_missing_top_ranked_open(monkeypatch):
    date_idx = pd.bdate_range("2024-01-02", periods=4)
    result_dict = _run_replay(
        monkeypatch,
        open_by_symbol_dict={
            "AAA": [100.0, np.nan, 102.0, 103.0],
            "BBB": [100.0, 101.0, 102.0, 103.0],
        },
        universe_by_symbol_dict={
            "AAA": [1, 1, 1, 1],
            "BBB": [1, 1, 1, 1],
        },
        candidate_by_date_dict={
            date_idx[0].date().isoformat(): ["AAA", "BBB"],
        },
    )

    assert _event_tuple_list(result_dict) == []
