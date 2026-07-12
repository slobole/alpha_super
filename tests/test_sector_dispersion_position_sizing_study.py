from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from scripts.research.run_sector_dispersion_position_sizing_study import (
    POSITION_SCALE_FIELD_STR,
    SIZING_EQUAL_SLOT_STR,
    PositionSizingConfig,
    SectorDispersionPositionSizingStrategy,
    _combine_pod_equity_ser,
    _rename_oos_metrics_as_recent_diagnostics,
    _save_equity_chart,
    compute_asset_vol_position_scale_df,
    compute_vix_position_scale_ser,
)
from strategies.mean_reversion.strategy_mr_sector_dispersion_ibs import DEFAULT_CONFIG


def test_vix_position_scale_only_reduces_exposure():
    vix_close_ser = pd.Series([15.0, 30.0, 40.0, 60.0, np.nan])

    position_scale_ser = compute_vix_position_scale_ser(vix_close_ser)

    assert position_scale_ser.iloc[:4].tolist() == pytest.approx([1.0, 1.0, 0.75, 0.50])
    assert np.isnan(position_scale_ser.iloc[4])


def test_asset_vol_scale_is_causal_and_capped():
    date_index = pd.bdate_range("2024-01-02", periods=25)
    calm_return_ser = pd.Series([0.0] + [0.005, -0.005] * 12, index=date_index)
    volatile_return_ser = pd.Series([0.0] + [0.03, -0.03] * 12, index=date_index)
    close_price_df = pd.DataFrame(
        {
            "CALM": 100.0 * (1.0 + calm_return_ser).cumprod(),
            "VOL": 100.0 * (1.0 + volatile_return_ser).cumprod(),
        },
        index=date_index,
    )

    _asset_volatility_ann_df, position_scale_df = compute_asset_vol_position_scale_df(
        close_price_df
    )
    original_scale_float = float(position_scale_df.loc[date_index[21], "VOL"])
    changed_close_price_df = close_price_df.copy()
    changed_close_price_df.loc[date_index[22]:, "VOL"] *= 2.0
    _changed_volatility_df, changed_scale_df = compute_asset_vol_position_scale_df(
        changed_close_price_df
    )

    assert float(position_scale_df.loc[date_index[21], "CALM"]) == pytest.approx(1.0)
    assert 0.50 <= original_scale_float < 1.0
    assert float(changed_scale_df.loc[date_index[21], "VOL"]) == pytest.approx(
        original_scale_float
    )


def test_dynamic_entry_sizing_applies_scale_without_up_leverage():
    base_config_obj = replace(
        DEFAULT_CONFIG,
        symbol_tuple=("AAA", "BBB"),
        portfolio_leverage_float=1.0,
        capital_base_float=100_000.0,
    )
    positioning_config_obj = PositionSizingConfig(
        base_config_obj=base_config_obj,
        sizing_mode_str=SIZING_EQUAL_SLOT_STR,
    )
    strategy_obj = SectorDispersionPositionSizingStrategy(
        name="position_sizing_test",
        benchmarks=[],
        positioning_config_obj=positioning_config_obj,
    )
    strategy_obj.previous_bar = pd.Timestamp("2024-01-05")
    close_row_ser = pd.Series(
        {
            ("AAA", "Close"): 100.0,
            ("AAA", POSITION_SCALE_FIELD_STR): 0.75,
        }
    )

    target_share_float = strategy_obj._entry_target_share_float("AAA", close_row_ser)

    assert target_share_float == pytest.approx(375.0)
    assert strategy_obj.entry_scale_record_list[0]["position_scale_float"] == pytest.approx(0.75)


def test_positioning_config_rejects_unknown_mode():
    with pytest.raises(ValueError, match="sizing_mode_str"):
        PositionSizingConfig(base_config_obj=DEFAULT_CONFIG, sizing_mode_str="unknown")


def test_equity_chart_handles_baskets_with_different_start_dates(tmp_path):
    date_index = pd.bdate_range("2024-01-02", periods=8)
    equity_df = pd.DataFrame(
        {
            "no_xlc__equal_slot": np.linspace(1.0, 1.1, len(date_index)),
            "xlc__equal_slot": pd.Series(
                np.linspace(1.0, 1.05, 4),
                index=date_index[-4:],
            ),
        },
        index=date_index,
    )

    _save_equity_chart(output_path=tmp_path, equity_df=equity_df)

    assert (tmp_path / "position_sizing_equity_curves.png").is_file()


def test_buy_and_hold_pod_combination_matches_independent_compounding():
    date_index = pd.bdate_range("2024-01-02", periods=3)
    sector_equity_ser = pd.Series([100.0, 110.0, 121.0], index=date_index)
    taa_equity_ser = pd.Series([200.0, 180.0, 198.0], index=date_index)

    portfolio_equity_ser = _combine_pod_equity_ser(sector_equity_ser, taa_equity_ser)

    assert portfolio_equity_ser.tolist() == pytest.approx([1.0, 1.0, 1.1])


def test_inherited_oos_fields_are_relabelled_as_recent_diagnostics():
    row_dict = {"oos_sharpe_float": 1.2, "sharpe_float": 1.0}

    _rename_oos_metrics_as_recent_diagnostics(row_dict)

    assert row_dict == {"recent_2022_sharpe_float": 1.2, "sharpe_float": 1.0}
