from __future__ import annotations

import pandas as pd
import pytest

from scripts.research.run_momentum_sleeve_split_study import (
    calculate_strategy_exposure_metrics,
    simulate_split_book,
)


def test_split_book_endpoints_match_component_returns() -> None:
    date_index = pd.bdate_range("2024-01-02", periods=5)
    strategy_return_df = pd.DataFrame(
        {
            "ndx": [0.0, 0.01, -0.02, 0.03, 0.01],
            "mosaic": [0.0, -0.01, 0.01, 0.02, -0.01],
        },
        index=date_index,
    )

    ndx_return_ser, _, _ = simulate_split_book(
        strategy_return_df,
        mosaic_weight_float=0.0,
        rebalance_frequency_str="annually",
    )
    mosaic_return_ser, _, _ = simulate_split_book(
        strategy_return_df,
        mosaic_weight_float=1.0,
        rebalance_frequency_str="annually",
    )

    pd.testing.assert_series_equal(
        ndx_return_ser,
        strategy_return_df["ndx"].rename("book_return"),
    )
    pd.testing.assert_series_equal(
        mosaic_return_ser,
        strategy_return_df["mosaic"].rename("book_return"),
    )


def test_annual_reset_uses_previous_close_book_value() -> None:
    date_index = pd.DatetimeIndex(
        ["2024-12-30", "2024-12-31", "2025-01-02", "2025-01-03"]
    )
    strategy_return_df = pd.DataFrame(
        {
            "ndx": [0.0, 1.0, 0.10, 0.0],
            "mosaic": [0.0, 0.0, 0.0, 0.0],
        },
        index=date_index,
    )

    book_return_ser, sleeve_equity_df, sleeve_weight_df = simulate_split_book(
        strategy_return_df,
        mosaic_weight_float=0.5,
        rebalance_frequency_str="annually",
    )

    assert sleeve_equity_df.loc[pd.Timestamp("2024-12-31")].to_dict() == {
        "ndx": 1.0,
        "mosaic": 0.5,
    }
    assert sleeve_equity_df.loc[pd.Timestamp("2025-01-02"), "ndx"] == pytest.approx(0.825)
    assert sleeve_equity_df.loc[pd.Timestamp("2025-01-02"), "mosaic"] == pytest.approx(0.75)
    assert abs(book_return_ser.loc[pd.Timestamp("2025-01-02")] - 0.05) < 1e-12
    assert abs(sleeve_weight_df.loc[pd.Timestamp("2025-01-02"), "ndx"] - 0.5238095238) < 1e-10


def test_effective_position_count_ignores_all_cash_days() -> None:
    class StrategyStub:
        realized_weight_df = pd.DataFrame(
            {
                "AAA": [0.0, 0.5],
                "BBB": [0.0, 0.5],
                "Cash": [1.0, 0.0],
            },
            index=pd.bdate_range("2025-01-02", periods=2),
        )

    metric_dict = calculate_strategy_exposure_metrics(StrategyStub())

    assert metric_dict["average_effective_position_count_float"] == pytest.approx(2.0)
