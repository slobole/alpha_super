"""Focused tests for the defensive-sleeve V3 Phase-0 runner."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.research.run_defensive_sleeve_v3_phase0 import (
    candidate_gate_df,
    headline_metric_df,
    inflation_window_metric_df,
    markdown_table_str,
    product_weight_by_alias_dict,
)


def test_frozen_product_weights_are_exact():
    spec_dict = {
        "v3_candidate_addition": {
            "DR3_core5_inflation_compass_05": {
                "defensive_sleeve_weight_by_alias": {
                    "core5": 0.95,
                    "inflation_compass": 0.05,
                }
            },
            "DR4_core5_tactical_fi_05": {
                "defensive_sleeve_weight_by_alias": {
                    "core5": 0.95,
                    "tactical_fi": 0.05,
                }
            },
            "DR5_core5_tactical_fi_equal_inflation_compass_05": {
                "defensive_sleeve_weight_by_alias": {
                    "core5": 0.475,
                    "tactical_fi": 0.475,
                    "inflation_compass": 0.05,
                }
            },
        },
        "matched_controls": {
            "MC1_core5_bil_05": {
                "defensive_sleeve_weight_by_alias": {
                    "core5": 0.95,
                    "bil": 0.05,
                }
            },
            "MC2_core5_tactical_fi_equal_bil_05": {
                "defensive_sleeve_weight_by_alias": {
                    "core5": 0.475,
                    "tactical_fi": 0.475,
                    "bil": 0.05,
                }
            },
        },
    }

    product_weight_dict = product_weight_by_alias_dict(spec_dict)

    assert product_weight_dict["B1_core5"] == {"core5": 1.0}
    assert product_weight_dict["DR2_core5_tfi_equal"] == {
        "core5": 0.5,
        "tactical_fi": 0.5,
    }
    assert all(
        np.isclose(sum(weight_dict.values()), 1.0)
        for weight_dict in product_weight_dict.values()
    )


def test_metrics_and_gate_use_frozen_directions():
    date_index = pd.bdate_range("2021-01-01", "2023-02-01")
    base_return_arr = np.full(len(date_index), 0.00015)
    product_return_by_id_dict = {
        "B1_core5": base_return_arr,
        "DR2_core5_tfi_equal": base_return_arr + 0.00001,
        "MC1_core5_bil_05": base_return_arr - 0.00002,
        "MC2_core5_tactical_fi_equal_bil_05": base_return_arr - 0.00001,
        "DR3_core5_inflation_compass_05": base_return_arr + 0.00004,
        "DR4_core5_tactical_fi_05": base_return_arr + 0.00003,
        "DR5_core5_tactical_fi_equal_inflation_compass_05": (
            base_return_arr + 0.00005
        ),
    }
    # Add deterministic variation so zero-rate Sharpe is finite.
    variation_arr = np.sin(np.arange(len(date_index), dtype=float)) * 0.001
    total_value_df = pd.DataFrame(index=date_index)
    for product_id_str, return_arr in product_return_by_id_dict.items():
        total_value_df[product_id_str] = 750_000.0 * np.cumprod(
            1.0 + return_arr + variation_arr
        )
    total_value_df.index.name = "date"

    headline_df = headline_metric_df(total_value_df)
    inflation_df = inflation_window_metric_df(total_value_df)
    gate_df = candidate_gate_df(headline_df, inflation_df)

    assert set(gate_df["candidate_id_str"]) == {
        "DR3_core5_inflation_compass_05",
        "DR4_core5_tactical_fi_05",
        "DR5_core5_tactical_fi_equal_inflation_compass_05",
    }
    assert gate_df["formal_accounting_gate_bool"].eq(False).all()
    assert gate_df.loc[
        gate_df["candidate_id_str"] == "DR4_core5_tactical_fi_05",
        "inflation_gate_required_bool",
    ].eq(False).all()


def test_markdown_table_does_not_require_optional_tabulate_package():
    table_str = markdown_table_str(
        pd.DataFrame(
            {
                "label_str": ["A|B", "C"],
                "gate_bool": [True, False],
                "metric_float": [0.125, np.nan],
            }
        )
    )

    assert "A\\|B" in table_str
    assert "true" in table_str
    assert "false" in table_str


def test_candidate_gate_enforces_frozen_threshold_directions():
    headline_df = pd.DataFrame(
        [
            ["B1_core5", 0.06, 1.00, -0.050, -0.040],
            ["DR2_core5_tfi_equal", 0.05, 1.50, -0.040, -0.030],
            ["MC1_core5_bil_05", 0.05, 1.00, -0.050, -0.040],
            [
                "MC2_core5_tactical_fi_equal_bil_05",
                0.05,
                1.50,
                -0.040,
                -0.030,
            ],
            ["DR3_core5_inflation_compass_05", 0.06, 1.021, -0.054, -0.044],
            ["DR4_core5_tactical_fi_05", 0.06, 1.019, -0.049, -0.039],
            [
                "DR5_core5_tactical_fi_equal_inflation_compass_05",
                0.06,
                1.521,
                -0.044,
                -0.034,
            ],
        ],
        columns=[
            "product_id_str",
            "cagr_float",
            "sharpe_float",
            "max_drawdown_float",
            "worst_252d_return_float",
        ],
    )
    inflation_df = pd.DataFrame(
        [
            ["B1_core5", 0.020, -0.040],
            ["DR2_core5_tfi_equal", 0.020, -0.030],
            ["DR3_core5_inflation_compass_05", 0.026, -0.044],
            ["DR4_core5_tactical_fi_05", 0.020, -0.040],
            [
                "DR5_core5_tactical_fi_equal_inflation_compass_05",
                0.026,
                -0.034,
            ],
            ["MC1_core5_bil_05", 0.020, -0.040],
            ["MC2_core5_tactical_fi_equal_bil_05", 0.020, -0.030],
        ],
        columns=[
            "product_id_str",
            "total_return_float",
            "max_drawdown_float",
        ],
    )

    gate_df = candidate_gate_df(headline_df, inflation_df).set_index(
        "candidate_id_str"
    )

    assert bool(
        gate_df.loc[
            "DR3_core5_inflation_compass_05", "phase0_numeric_gate_bool"
        ]
    )
    assert bool(
        gate_df.loc[
            "DR5_core5_tactical_fi_equal_inflation_compass_05",
            "phase0_numeric_gate_bool",
        ]
    )
    assert not bool(
        gate_df.loc[
            "DR4_core5_tactical_fi_05", "sharpe_gate_bool"
        ]
    )
    assert gate_df.loc[
        "DR4_core5_tactical_fi_05", "decision_str"
    ] == "phase0_rejected"
    assert gate_df.loc[
        "DR3_core5_inflation_compass_05", "decision_str"
    ] == "phase0_numeric_pass_accounting_blocked"
