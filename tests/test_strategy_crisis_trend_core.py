"""Causal and mathematical tests for the frozen Crisis Trend Core."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha.engine.backtest import run_daily
from strategies.tail_hedge import strategy_crisis_trend_core as strategy_module


def _synthetic_total_return_close_df(
    sessions_int: int = 900,
) -> pd.DataFrame:
    session_idx = pd.bdate_range("2015-01-02", periods=sessions_int)
    price_by_asset_dict: dict[str, np.ndarray] = {}
    for asset_position_int, asset_str in enumerate(
        strategy_module.UNIVERSE_ASSET_TUPLE
    ):
        if asset_str in strategy_module.LONG_ONLY_ASSET_TUPLE:
            daily_return_float = 0.0006 + asset_position_int * 0.000001
        else:
            daily_return_float = -0.0006 - asset_position_int * 0.000001
        price_by_asset_dict[asset_str] = 100.0 * np.cumprod(
            np.full(sessions_int, 1.0 + daily_return_float)
        )
    price_by_asset_dict[strategy_module.RESERVE_ASSET_STR] = 100.0 * np.cumprod(
        np.full(sessions_int, 1.00005)
    )
    return pd.DataFrame(price_by_asset_dict, index=session_idx)


def _synthetic_engine_pricing_data_df(
    total_return_close_df: pd.DataFrame,
) -> pd.DataFrame:
    execution_field_map: dict[tuple[str, str], pd.Series] = {}
    for asset_str in strategy_module.TRADEABLE_ASSET_TUPLE:
        close_ser = total_return_close_df[asset_str]
        execution_field_map[(asset_str, "Open")] = close_ser
        execution_field_map[(asset_str, "High")] = close_ser
        execution_field_map[(asset_str, "Low")] = close_ser
        execution_field_map[(asset_str, "Close")] = close_ser
        execution_field_map[(asset_str, "Dividend")] = pd.Series(
            0.0,
            index=total_return_close_df.index,
        )
    benchmark_return_vec = np.where(
        np.arange(len(total_return_close_df)) % 2 == 0,
        0.0008,
        -0.0002,
    )
    benchmark_close_ser = pd.Series(
        100.0 * np.cumprod(1.0 + benchmark_return_vec),
        index=total_return_close_df.index,
    )
    for field_str in ("Open", "High", "Low", "Close"):
        execution_field_map[("$SPX", field_str)] = benchmark_close_ser
    for asset_str in strategy_module.TRADEABLE_ASSET_TUPLE:
        execution_field_map[
            (strategy_module.signal_namespace_str(asset_str), "Close")
        ] = total_return_close_df[asset_str]
    pricing_data_df = pd.DataFrame(execution_field_map)
    pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
    pricing_data_df.attrs["norgate_adjustment_by_symbol_dict"] = {
        **{
            asset_str: "CAPITALSPECIAL"
            for asset_str in strategy_module.TRADEABLE_ASSET_TUPLE
        },
        "$SPX": "TOTALRETURN",
    }
    pricing_data_df.attrs["benchmark_data_symbol_dict"] = {"$SPX": "$SPXTR"}
    return pricing_data_df


def test_targets_do_not_depend_on_future_closes() -> None:
    total_return_close_df = _synthetic_total_return_close_df()
    base_bundle_obj = strategy_module.compute_crisis_trend_signal_bundle(
        total_return_close_df
    )
    cutoff_position_int = 700
    cutoff_ts = total_return_close_df.index[cutoff_position_int]
    perturbed_close_df = total_return_close_df.copy()
    perturbed_close_df.iloc[cutoff_position_int + 1 :] *= 1.37

    perturbed_bundle_obj = strategy_module.compute_crisis_trend_signal_bundle(
        perturbed_close_df
    )

    pd.testing.assert_frame_equal(
        base_bundle_obj.month_end_target_weight_df.loc[:cutoff_ts],
        perturbed_bundle_obj.month_end_target_weight_df.loc[:cutoff_ts],
    )


def test_direction_gross_and_month_end_contracts_hold() -> None:
    total_return_close_df = _synthetic_total_return_close_df()
    signal_bundle_obj = strategy_module.compute_crisis_trend_signal_bundle(
        total_return_close_df
    )

    for asset_str in strategy_module.LONG_ONLY_ASSET_TUPLE:
        assert (signal_bundle_obj.desired_weight_df[asset_str] >= -1e-12).all()
    for asset_str in strategy_module.SHORT_ONLY_ASSET_TUPLE:
        assert (signal_bundle_obj.desired_weight_df[asset_str] <= 1e-12).all()
    gross_exposure_ser = signal_bundle_obj.desired_weight_df.abs().sum(axis=1)
    assert (
        gross_exposure_ser
        <= strategy_module.GROSS_EXPOSURE_CAP_FLOAT + 1e-12
    ).all()

    month_end_bool_ser = strategy_module.month_end_decision_bool_ser(
        total_return_close_df.index
    )
    inside_month_change_df = signal_bundle_obj.month_end_target_weight_df.diff().abs().loc[
        ~month_end_bool_ser
    ]
    assert float(inside_month_change_df.iloc[1:].sum().sum()) == 0.0


def test_missing_close_breaks_260_session_eligibility_without_forward_fill() -> None:
    total_return_close_df = _synthetic_total_return_close_df()
    missing_ts = total_return_close_df.index[700]
    total_return_close_df.loc[missing_ts, "SPY"] = np.nan

    signal_bundle_obj = strategy_module.compute_crisis_trend_signal_bundle(
        total_return_close_df
    )

    assert not bool(signal_bundle_obj.eligible_df.loc[missing_ts, "SPY"])
    assert not bool(
        signal_bundle_obj.eligible_df.loc[
            total_return_close_df.index[701:900],
            "SPY",
        ].any()
    )
    assert signal_bundle_obj.desired_weight_df.loc[missing_ts, "SPY"] == 0.0


@pytest.mark.parametrize("lookback_int", [0, 63, 126, 252])
@pytest.mark.parametrize("invalid_price_float", [np.nan, np.inf, 0.0, -1.0])
def test_invalid_reserve_signal_endpoint_fails_loud(
    lookback_int: int, invalid_price_float: float,
) -> None:
    total_return_close_df = _synthetic_total_return_close_df()
    decision_ts = pd.Timestamp("2017-07-31")
    decision_position_int = int(total_return_close_df.index.get_loc(decision_ts))
    endpoint_ts = total_return_close_df.index[decision_position_int - lookback_int]
    total_return_close_df.loc[endpoint_ts, "SHY"] = invalid_price_float

    with pytest.raises(ValueError, match="SHY TOTALRETURN signal endpoints"):
        strategy_module.compute_crisis_trend_signal_bundle(total_return_close_df)


@pytest.mark.parametrize("lookback_int", [63, 126, 252])
def test_missing_warmup_reserve_endpoint_blocks_first_eligible_row(
    lookback_int: int,
) -> None:
    total_return_close_df = _synthetic_total_return_close_df()
    eligible_position_int = strategy_module.MINIMUM_HISTORY_SESSIONS_INT - 1
    endpoint_ts = total_return_close_df.index[eligible_position_int - lookback_int]
    total_return_close_df.loc[endpoint_ts, "SHY"] = np.nan

    with pytest.raises(ValueError, match="SHY TOTALRETURN signal endpoints") as error_obj:
        strategy_module.compute_crisis_trend_signal_bundle(total_return_close_df)
    assert str(error_obj.value).endswith(
        f"at Close_{total_return_close_df.index[eligible_position_int]}."
    )


def test_missing_reserve_during_unused_warmup_preserves_targets() -> None:
    total_return_close_df = _synthetic_total_return_close_df()
    expected_bundle_obj = strategy_module.compute_crisis_trend_signal_bundle(
        total_return_close_df
    )
    # The first eligible row is 259; its oldest SHY endpoint is row 7.
    total_return_close_df.loc[total_return_close_df.index[:7], "SHY"] = np.nan
    actual_bundle_obj = strategy_module.compute_crisis_trend_signal_bundle(
        total_return_close_df
    )
    pd.testing.assert_frame_equal(
        actual_bundle_obj.desired_weight_df, expected_bundle_obj.desired_weight_df
    )
    pd.testing.assert_frame_equal(
        actual_bundle_obj.month_end_target_weight_df,
        expected_bundle_obj.month_end_target_weight_df,
    )


def test_rebalance_band_is_strict_and_force_exit_is_unconditional() -> None:
    target_weight_ser = pd.Series(
        0.0,
        index=strategy_module.UNIVERSE_ASSET_TUPLE,
    )
    target_weight_ser.loc["SPY"] = -0.10
    held_weight_ser = target_weight_ser.copy()
    held_weight_ser.loc["SPY"] += strategy_module.REBALANCE_BAND_FLOAT

    assert not strategy_module.should_rebalance_bool(
        target_weight_ser,
        held_weight_ser,
        initialized_bool=True,
    )
    held_weight_ser.loc["SPY"] += 1e-9
    assert strategy_module.should_rebalance_bool(
        target_weight_ser,
        held_weight_ser,
        initialized_bool=True,
    )
    zero_target_ser = target_weight_ser * 0.0
    assert strategy_module.should_rebalance_bool(
        zero_target_ser,
        pd.Series({"SPY": -0.001}),
        initialized_bool=True,
    )


def test_month_end_close_decision_fills_at_next_open() -> None:
    total_return_close_df = _synthetic_total_return_close_df()
    pricing_data_df = _synthetic_engine_pricing_data_df(total_return_close_df)
    month_end_bool_ser = strategy_module.month_end_decision_bool_ser(
        total_return_close_df.index
    )
    eligible_month_end_idx = month_end_bool_ser[
        month_end_bool_ser
        & (np.arange(len(month_end_bool_ser)) > 650)
        & (np.arange(len(month_end_bool_ser)) < 800)
    ].index
    decision_ts = pd.Timestamp(eligible_month_end_idx[0])
    decision_position_int = int(total_return_close_df.index.get_loc(decision_ts))
    execution_ts = pd.Timestamp(total_return_close_df.index[decision_position_int + 1])
    calendar_idx = total_return_close_df.index[
        decision_position_int + 1 : decision_position_int + 6
    ]
    strategy_obj = strategy_module.CrisisTrendCoreStrategy()

    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar=calendar_idx,
        show_progress=False,
        show_signal_progress_bool=False,
    )

    transaction_df = strategy_obj.get_transactions()
    assert len(transaction_df) > 0
    assert set(pd.to_datetime(transaction_df["bar"])) == {execution_ts}
    assert decision_ts in strategy_obj.rebalance_target_weight_df.index
    for asset_str in strategy_module.LONG_ONLY_ASSET_TUPLE:
        assert strategy_obj.daily_target_weights.loc[decision_ts, asset_str] >= 0.0
    for asset_str in strategy_module.SHORT_ONLY_ASSET_TUPLE:
        assert strategy_obj.daily_target_weights.loc[decision_ts, asset_str] <= 0.0
