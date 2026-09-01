from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.research import run_core5_vs_tactical_fi_equal_earliest as study


def test_source_contract_freezes_earliest_common_comparison() -> None:
    source_contract_dict = study.build_source_contract_dict()

    assert source_contract_dict["portfolio_contract"] == {
        "capital_base_float": 750_000.0,
        "requested_start_date_str": "2002-07-26",
        "capital_anchor_date_str": "2007-08-31",
        "effective_execution_start_date_str": "2007-09-04",
        "end_date_str": "2026-08-19",
        "outer_rebalance": None,
        "allocation_semantics_str": "fixed_initial_capital_then_independent_drift",
        "decision_execution_timing_str": "Close_T_to_Open_T_plus_1",
    }
    assert set(source_contract_dict["source_runs"]) == {
        "core5_375000",
        "core5_750000",
        "tactical_fi_375000",
    }


def test_product_builder_sums_independently_compounded_subaccounts() -> None:
    date_idx = pd.DatetimeIndex(
        ["2007-08-31", "2007-09-04", "2026-08-19"], name="date"
    )
    source_path_by_id_dict = {
        "core5_375000": pd.DataFrame(
            {"total_value_float": [375_000.0, 380_000.0, 700_000.0]},
            index=date_idx,
        ),
        "core5_750000": pd.DataFrame(
            {"total_value_float": [750_000.0, 760_000.0, 1_500_000.0]},
            index=date_idx,
        ),
        "tactical_fi_375000": pd.DataFrame(
            {"total_value_float": [375_000.0, 376_000.0, 500_000.0]},
            index=date_idx,
        ),
    }

    product_value_df = study.build_product_value_df(source_path_by_id_dict)

    assert np.allclose(
        product_value_df["CORE5"].to_numpy(),
        [750_000.0, 760_000.0, 1_500_000.0],
    )
    assert np.allclose(
        product_value_df["CORE5_50_TACTICAL_FI_50"].to_numpy(),
        [750_000.0, 756_000.0, 1_200_000.0],
    )


def test_calendar_returns_marks_first_year_partial() -> None:
    date_idx = pd.DatetimeIndex(
        ["2007-08-31", "2007-12-31", "2008-12-31"], name="date"
    )
    product_value_df = pd.DataFrame(
        {
            "CORE5": [100.0, 110.0, 99.0],
            "CORE5_50_TACTICAL_FI_50": [100.0, 105.0, 105.0],
        },
        index=date_idx,
    )

    calendar_df = study.calendar_return_df(product_value_df).set_index("year_int")

    assert calendar_df.loc[2007, "window_status_str"] == "partial_from_2007_08_31"
    assert np.isclose(calendar_df.loc[2007, "CORE5"], 0.10)
    assert np.isclose(calendar_df.loc[2008, "CORE5"], -0.10)
    assert np.isclose(
        calendar_df.loc[2008, "equal_minus_core5_float"], 0.10
    )


def test_calendar_returns_marks_final_year_partial() -> None:
    date_idx = pd.DatetimeIndex(
        ["2025-12-31", "2026-08-19"], name="date"
    )
    product_value_df = pd.DataFrame(
        {
            "CORE5": [100.0, 110.0],
            "CORE5_50_TACTICAL_FI_50": [100.0, 105.0],
        },
        index=date_idx,
    )

    calendar_df = study.calendar_return_df(product_value_df).set_index("year_int")

    assert calendar_df.loc[2026, "window_status_str"] == (
        "partial_through_2026_08_19"
    )


def test_execution_reality_surfaces_negative_cash_financing_gap() -> None:
    date_idx = pd.DatetimeIndex(["2007-08-31", "2007-09-04"], name="date")
    source_path_by_id_dict = {
        "core5_750000": pd.DataFrame(
            {
                "total_value_float": [750_000.0, 749_000.0],
                "cash_float": [750_000.0, -7_490.0],
            },
            index=date_idx,
        )
    }
    source_summary_df = pd.DataFrame(
        [
            {
                "source_id_str": "core5_750000",
                "negative_cash_day_count_int": 1,
                "minimum_cash_float": -7_490.0,
                "negative_cash_financing_policy_str": "not_modeled",
                "slippage_per_side_float": 0.00025,
                "commission_per_share_float": 0.005,
                "commission_minimum_float": 1.0,
            }
        ]
    )

    execution_df = study.execution_reality_df(
        source_path_by_id_dict, source_summary_df
    )

    assert execution_df.loc[0, "negative_cash_day_count_int"] == 1
    assert np.isclose(execution_df.loc[0, "minimum_cash_weight_float"], -0.01)
    assert execution_df.loc[0, "negative_cash_financing_policy_str"] == (
        "not_modeled"
    )
