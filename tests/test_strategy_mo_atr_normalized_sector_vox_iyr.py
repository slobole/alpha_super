from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from strategies.momentum.run_atr_normalized_sector_vox_iyr_sweep import (
    NATR_AUDIT_VARIANT_SPEC_TUPLE,
    VARIANT_SPEC_TUPLE,
)
from strategies.momentum.run_atr_normalized_sector_vox_iyr_vix_leverage_sweep import (
    EXPOSURE_VARIANT_SPEC_TUPLE,
    _financing_adjusted_equity_ser,
)
from strategies.momentum.strategy_mo_atr_normalized_ndx import (
    map_month_end_decision_dates_to_rebalance_schedule_df,
)
from strategies.momentum.strategy_mo_atr_normalized_sector_vox_iyr import (
    DEFAULT_CONFIG,
    DIMENSIONLESS_NATR_SCORE_STR,
    SECTOR_SYMBOL_TUPLE,
    SOURCE_ATR_SCORE_STR,
    AtrNormalizedSectorStrategy,
    compute_vix_scale_signal_df,
    get_asof_vix_scale_float,
)


def _strategy_obj(
    *,
    apply_market_trend_bool: bool = True,
    apply_asset_trend_bool: bool = True,
    use_vix_scale_bool: bool = False,
    score_mode_str: str = SOURCE_ATR_SCORE_STR,
    max_positions_int: int = 3,
    static_exposure_scale_float: float = 1.0,
) -> AtrNormalizedSectorStrategy:
    config_obj = replace(
        DEFAULT_CONFIG,
        apply_market_trend_bool=apply_market_trend_bool,
        apply_asset_trend_bool=apply_asset_trend_bool,
        use_vix_scale_bool=use_vix_scale_bool,
        score_mode_str=score_mode_str,
        max_positions_int=max_positions_int,
        static_exposure_scale_float=static_exposure_scale_float,
    )
    rebalance_schedule_df = pd.DataFrame(
        {"decision_date_ts": [pd.Timestamp("2024-01-31")]},
        index=pd.DatetimeIndex(["2024-02-01"]),
    )
    vix_scale_signal_df = pd.DataFrame(
        {"vix_exposure_scale_float": [0.5]},
        index=pd.DatetimeIndex(["2024-01-31"]),
    )
    strategy_obj = AtrNormalizedSectorStrategy(
        name="test_sector_trend",
        benchmarks=["$SPX"],
        rebalance_schedule_df=rebalance_schedule_df,
        vix_scale_signal_df=vix_scale_signal_df,
        config_obj=config_obj,
    )
    strategy_obj.universe_df = pd.DataFrame(
        1,
        index=pd.DatetimeIndex(["2024-01-31"]),
        columns=list(SECTOR_SYMBOL_TUPLE),
    )
    strategy_obj.previous_bar = pd.Timestamp("2024-01-31")
    return strategy_obj


def _close_row_ser(
    *,
    regime_pass_bool: bool,
    first_asset_trend_bool: bool,
) -> pd.Series:
    feature_map = {
        ("SPY", "regime_pass_bool"): regime_pass_bool,
    }
    for rank_int, symbol_str in enumerate(SECTOR_SYMBOL_TUPLE):
        feature_map[(symbol_str, "risk_adj_score_ser")] = float(
            len(SECTOR_SYMBOL_TUPLE) - rank_int
        )
        feature_map[(symbol_str, "stock_trend_pass_bool")] = (
            first_asset_trend_bool if rank_int == 0 else True
        )
    close_row_ser = pd.Series(feature_map, dtype=object)
    close_row_ser.index = pd.MultiIndex.from_tuples(close_row_ser.index)
    return close_row_ser


def test_fixed_basket_and_cost_defaults_match_engine_house_model() -> None:
    assert SECTOR_SYMBOL_TUPLE == (
        "XLB",
        "XLE",
        "XLF",
        "XLI",
        "XLK",
        "XLP",
        "XLU",
        "XLV",
        "XLY",
        "VOX",
        "IYR",
    )
    assert DEFAULT_CONFIG.slippage_float == 0.00025
    assert DEFAULT_CONFIG.commission_per_share_float == 0.005
    assert DEFAULT_CONFIG.commission_minimum_float == 1.0


def test_sweep_is_frozen_to_twelve_unique_variants() -> None:
    assert len(VARIANT_SPEC_TUPLE) == 12
    assert len({variant_spec_tuple[0] for variant_spec_tuple in VARIANT_SPEC_TUPLE}) == 12


def test_post_review_natr_audit_covers_full_filter_and_position_grid() -> None:
    assert len(NATR_AUDIT_VARIANT_SPEC_TUPLE) == 8
    observed_config_set = {
        (
            variant_spec_tuple[1],
            variant_spec_tuple[2],
            variant_spec_tuple[3],
            variant_spec_tuple[4],
            variant_spec_tuple[5],
        )
        for variant_spec_tuple in NATR_AUDIT_VARIANT_SPEC_TUPLE
    }
    expected_config_set = {
        (
            max_positions_int,
            apply_market_trend_bool,
            apply_asset_trend_bool,
            DIMENSIONLESS_NATR_SCORE_STR,
            False,
        )
        for max_positions_int in (3, 5)
        for apply_market_trend_bool in (False, True)
        for apply_asset_trend_bool in (False, True)
    }
    assert observed_config_set == expected_config_set


def test_vix_leverage_sweep_is_frozen_to_seven_unique_rows() -> None:
    assert len(EXPOSURE_VARIANT_SPEC_TUPLE) == 7
    assert len(
        {
            variant_spec_obj.variant_name_str
            for variant_spec_obj in EXPOSURE_VARIANT_SPEC_TUPLE
        }
    ) == 7
    assert {
        variant_spec_obj.max_exposure_scale_float
        for variant_spec_obj in EXPOSURE_VARIANT_SPEC_TUPLE
    } == {1.0, 1.25, 1.5}


def test_month_end_decision_maps_strictly_to_next_open() -> None:
    execution_index = pd.DatetimeIndex(
        ["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02"]
    )
    schedule_df = map_month_end_decision_dates_to_rebalance_schedule_df(
        decision_date_index=pd.DatetimeIndex(["2024-01-31"]),
        execution_index=execution_index,
    )
    assert schedule_df.index.tolist() == [pd.Timestamp("2024-02-01")]
    assert schedule_df.iloc[0]["decision_date_ts"] == pd.Timestamp("2024-01-31")


def test_market_and_asset_filters_can_be_disabled_independently() -> None:
    close_row_ser = _close_row_ser(
        regime_pass_bool=False,
        first_asset_trend_bool=False,
    )
    both_filter_strategy_obj = _strategy_obj()
    assert both_filter_strategy_obj.get_target_weight_ser(close_row_ser).empty

    asset_only_strategy_obj = _strategy_obj(apply_market_trend_bool=False)
    asset_only_target_ser = asset_only_strategy_obj.get_target_weight_ser(close_row_ser)
    assert "XLB" not in asset_only_target_ser.index
    assert len(asset_only_target_ser) == 3

    no_filter_strategy_obj = _strategy_obj(
        apply_market_trend_bool=False,
        apply_asset_trend_bool=False,
    )
    no_filter_target_ser = no_filter_strategy_obj.get_target_weight_ser(close_row_ser)
    assert no_filter_target_ser.index[0] == "XLB"
    np.testing.assert_allclose(no_filter_target_ser.to_numpy(), np.repeat(1.0 / 3.0, 3))


def test_vix_scaler_is_causal_clipped_and_scales_total_exposure() -> None:
    vix_close_ser = pd.Series(
        [10.0, 40.0],
        index=pd.DatetimeIndex(["2024-01-30", "2024-02-01"]),
    )
    vix_scale_signal_df = compute_vix_scale_signal_df(
        vix_close_ser=vix_close_ser,
        target_vix_pct_float=20.0,
        min_exposure_scale_float=0.25,
        max_exposure_scale_float=1.0,
    )
    assert get_asof_vix_scale_float(
        vix_scale_signal_df,
        pd.Timestamp("2024-01-31"),
    ) == 1.0
    assert get_asof_vix_scale_float(
        vix_scale_signal_df,
        pd.Timestamp("2024-02-01"),
    ) == 0.5

    strategy_obj = _strategy_obj(use_vix_scale_bool=True)
    target_weight_ser = strategy_obj.get_target_weight_ser(
        _close_row_ser(regime_pass_bool=True, first_asset_trend_bool=True)
    )
    assert np.isclose(float(target_weight_ser.sum()), 0.5)


def test_static_exposure_supports_capped_one_point_five_leverage() -> None:
    strategy_obj = _strategy_obj(
        apply_market_trend_bool=False,
        apply_asset_trend_bool=False,
        static_exposure_scale_float=1.5,
    )
    target_weight_ser = strategy_obj.get_target_weight_ser(
        _close_row_ser(regime_pass_bool=False, first_asset_trend_bool=False)
    )
    assert np.isclose(float(target_weight_ser.sum()), 1.5)
    assert replace(
        DEFAULT_CONFIG,
        max_exposure_scale_float=1.5,
    ).max_exposure_scale_float == 1.5


def test_one_point_five_is_close_sized_target_not_realized_open_cap() -> None:
    strategy_obj = _strategy_obj(
        apply_market_trend_bool=False,
        apply_asset_trend_bool=False,
        static_exposure_scale_float=1.5,
        max_positions_int=5,
    )
    close_row_ser = _close_row_ser(
        regime_pass_bool=False,
        first_asset_trend_bool=False,
    )
    close_price_ser = pd.Series(
        {
            (symbol_str, "Close"): 100.0
            for symbol_str in SECTOR_SYMBOL_TUPLE
        },
        dtype=float,
    )
    close_price_ser.index = pd.MultiIndex.from_tuples(close_price_ser.index)
    close_row_ser = pd.concat([close_row_ser, close_price_ser])
    target_weight_ser = strategy_obj.get_target_weight_ser(close_row_ser)
    target_share_int_map = strategy_obj.get_target_share_int_map(
        target_weight_ser=target_weight_ser,
        close_row_ser=close_row_ser,
    )
    prior_total_value_float = float(strategy_obj.previous_total_value)
    open_gap_multiplier_float = 1.10
    realized_open_gross_exposure_float = sum(
        target_share_int
        * float(close_row_ser[(symbol_str, "Close")])
        * open_gap_multiplier_float
        for symbol_str, target_share_int in target_share_int_map.items()
    ) / prior_total_value_float

    assert np.isclose(float(target_weight_ser.sum()), 1.5)
    assert realized_open_gross_exposure_float > 1.5
    assert np.isclose(realized_open_gross_exposure_float, 1.65, atol=0.001)


def test_financing_sensitivity_charges_prior_close_negative_cash() -> None:
    date_idx = pd.DatetimeIndex(
        ["2024-01-02", "2024-01-03", "2024-01-04"]
    )
    total_value_ser = pd.Series([100.0, 110.0, 121.0], index=date_idx)
    cash_ser = pd.Series([-50.0, -50.0, -50.0], index=date_idx)
    dtb3_annual_rate_ser = pd.Series([0.05], index=date_idx[:1])

    (
        financing_adjusted_equity_ser,
        financing_cost_return_ser,
        borrowed_weight_ser,
    ) = _financing_adjusted_equity_ser(
        total_value_ser=total_value_ser,
        cash_ser=cash_ser,
        dtb3_annual_rate_ser=dtb3_annual_rate_ser,
    )
    daily_rate_float = (1.05 ** (1.0 / 252.0)) - 1.0

    assert financing_cost_return_ser.iloc[0] == 0.0
    assert np.isclose(
        financing_cost_return_ser.iloc[1],
        0.5 * daily_rate_float,
    )
    assert np.isclose(
        financing_cost_return_ser.iloc[2],
        (50.0 / 110.0) * daily_rate_float,
    )
    assert np.isclose(borrowed_weight_ser.iloc[0], 0.5)
    assert financing_adjusted_equity_ser.iloc[-1] < total_value_ser.iloc[-1]


def test_score_modes_are_explicit_and_distinct() -> None:
    assert SOURCE_ATR_SCORE_STR != DIMENSIONLESS_NATR_SCORE_STR
    assert replace(
        DEFAULT_CONFIG,
        score_mode_str=DIMENSIONLESS_NATR_SCORE_STR,
    ).score_mode_str == DIMENSIONLESS_NATR_SCORE_STR
