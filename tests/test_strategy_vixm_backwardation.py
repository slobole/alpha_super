"""Causal rule and engine tests for VIXM Backwardation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha.engine.backtest import run_daily
from strategies.tail_hedge import strategy_vixm_backwardation as strategy_module


def _synthetic_pricing_data_df() -> pd.DataFrame:
    session_idx = pd.bdate_range("2020-01-02", periods=7)
    field_map: dict[tuple[str, str], pd.Series] = {}
    price_by_asset_dict = {
        "VIXM": pd.Series([30.0, 30.0, 31.0, 32.0, 32.0, 31.0, 31.0], index=session_idx),
        "SHY": pd.Series([85.0, 85.0, 85.1, 85.1, 85.2, 85.2, 85.3], index=session_idx),
    }
    for asset_str, close_ser in price_by_asset_dict.items():
        for field_str in ("Open", "High", "Low", "Close"):
            field_map[(asset_str, field_str)] = close_ser
        field_map[(asset_str, "Dividend")] = pd.Series(0.0, index=session_idx)
    benchmark_ser = pd.Series(
        [100.0, 101.0, 99.0, 98.0, 100.0, 101.0, 102.0],
        index=session_idx,
    )
    for field_str in ("Open", "High", "Low", "Close"):
        field_map[("$SPX", field_str)] = benchmark_ser
    field_map[(strategy_module.VIX_SIGNAL_NAMESPACE_STR, "vix_close_float")] = pd.Series(
        [18.0, 25.0, 26.0, 19.0, 19.0, 18.0, 18.0],
        index=session_idx,
    )
    field_map[(strategy_module.VIX_SIGNAL_NAMESPACE_STR, "vix3m_close_float")] = pd.Series(
        [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
        index=session_idx,
    )
    pricing_data_df = pd.DataFrame(field_map)
    pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
    pricing_data_df.attrs["norgate_adjustment_by_symbol_dict"] = {
        "VIXM": "CAPITALSPECIAL",
        "SHY": "CAPITALSPECIAL",
        "$SPX": "TOTALRETURN",
    }
    pricing_data_df.attrs["benchmark_data_symbol_dict"] = {"$SPX": "$SPXTR"}
    return pricing_data_df


def test_backwardation_rule_is_strict_and_missing_inputs_stay_missing() -> None:
    session_idx = pd.bdate_range("2024-01-02", periods=4)
    vix_close_ser = pd.Series([21.0, 20.0, 19.0, np.nan], index=session_idx)
    vix3m_close_ser = pd.Series([20.0, 20.0, 20.0, 20.0], index=session_idx)

    state_ser = strategy_module.compute_backwardation_state_ser(
        vix_close_ser,
        vix3m_close_ser,
    )

    assert state_ser.iloc[0] == 1.0
    assert state_ser.iloc[1] == 0.0
    assert state_ser.iloc[2] == 0.0
    assert np.isnan(state_ser.iloc[3])


def test_target_is_always_all_vixm_or_all_shy() -> None:
    session_idx = pd.bdate_range("2024-01-02", periods=3)
    state_ser = pd.Series([0.0, 1.0, 0.0], index=session_idx)

    target_weight_df = strategy_module.build_vixm_target_weight_df(state_ser)

    assert target_weight_df.sum(axis=1).eq(1.0).all()
    assert target_weight_df.loc[session_idx[0], "SHY"] == 1.0
    assert target_weight_df.loc[session_idx[1], "VIXM"] == 1.0


def test_close_state_change_trades_at_next_open_only() -> None:
    pricing_data_df = _synthetic_pricing_data_df()
    session_idx = pricing_data_df.index
    strategy_obj = strategy_module.VixmBackwardationStrategy()

    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar=session_idx[1:],
        show_progress=False,
        show_signal_progress_bool=False,
    )

    transaction_df = strategy_obj.get_transactions()
    transaction_bar_ser = pd.to_datetime(transaction_df["bar"])
    assert session_idx[1] in set(transaction_bar_ser)
    assert session_idx[2] in set(transaction_bar_ser)
    assert session_idx[4] in set(transaction_bar_ser)
    assert session_idx[0] not in set(transaction_bar_ser)
    assert strategy_obj.daily_target_weights.loc[session_idx[0], "VIXM"] == 0.0
    assert strategy_obj.daily_target_weights.loc[session_idx[1], "VIXM"] == 1.0
    assert strategy_obj.daily_target_weights.loc[session_idx[3], "VIXM"] == 0.0


def test_engine_fails_loud_on_missing_decision_close() -> None:
    pricing_data_df = _synthetic_pricing_data_df()
    session_idx = pricing_data_df.index
    pricing_data_df.loc[
        session_idx[2],
        (strategy_module.VIX_SIGNAL_NAMESPACE_STR, "vix_close_float"),
    ] = np.nan
    strategy_obj = strategy_module.VixmBackwardationStrategy()

    with pytest.raises(RuntimeError, match="Missing VIX/VIX3M state"):
        run_daily(
            strategy_obj,
            pricing_data_df,
            calendar=session_idx[1:5],
            show_progress=False,
            show_signal_progress_bool=False,
        )


def test_canceled_transition_is_retried_while_state_remains_active() -> None:
    pricing_data_df = _synthetic_pricing_data_df()
    session_idx = pricing_data_df.index
    pricing_data_df.loc[session_idx[2], ("VIXM", "Open")] = np.nan
    strategy_obj = strategy_module.VixmBackwardationStrategy()

    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar=session_idx[1:4],
        show_progress=False,
        show_signal_progress_bool=False,
    )

    transaction_df = strategy_obj.get_transactions()
    vixm_buy_df = transaction_df.loc[
        (transaction_df["asset"] == "VIXM") & (transaction_df["amount"] > 0.0)
    ]
    assert pd.Timestamp(vixm_buy_df.iloc[0]["bar"]) == session_idx[3]
    assert strategy_obj.get_position("VIXM") > 0.0
    assert strategy_obj.get_position("SHY") == 0.0


def test_open_gap_does_not_change_close_fixed_target_shares() -> None:
    pricing_data_df = _synthetic_pricing_data_df()
    session_idx = pricing_data_df.index
    pricing_data_df.loc[session_idx[2], ("VIXM", "Open")] = 60.0
    strategy_obj = strategy_module.VixmBackwardationStrategy()
    strategy_obj.previous_bar = session_idx[1]
    target_weight_ser = pd.Series(
        {"VIXM": 1.0, "SHY": 0.0},
        dtype=float,
    )

    strategy_obj._submit_target_orders(
        target_weight_ser,
        pricing_data_df.loc[session_idx[1]],
    )

    vixm_order_obj = next(
        order_obj for order_obj in strategy_obj.get_orders() if order_obj.asset == "VIXM"
    )
    expected_share_int = int(strategy_obj.config_obj.capital_base_float / 30.0)
    assert int(vixm_order_obj.amount) == expected_share_int
