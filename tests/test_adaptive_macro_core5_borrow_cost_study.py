from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.research.run_adaptive_macro_core5_borrow_cost_study import (
    BorrowCostAdaptiveMacroCore5Strategy,
    DBC_EXTERNAL_SNAPSHOT_RATE_PCT_TUPLE,
    build_gate_df,
)
from strategies.taa_beyond_6040.strategy_taa_adaptive_macro_core5 import (
    AdaptiveMacroCore5Strategy,
)


def _borrow_strategy_obj(annual_borrow_rate_float: float) -> BorrowCostAdaptiveMacroCore5Strategy:
    strategy_obj = BorrowCostAdaptiveMacroCore5Strategy(
        annual_borrow_rate_float=annual_borrow_rate_float,
    )
    strategy_obj.current_bar = pd.Timestamp("2024-01-05")
    strategy_obj.borrow_calendar_idx = pd.DatetimeIndex(
        ["2024-01-05", "2024-01-08"]
    )
    strategy_obj.cash = 50_000.0
    strategy_obj.total_value = 100_000.0
    strategy_obj._position_amount_map = {"DBC": -100.0}
    return strategy_obj


def test_borrow_fee_uses_102_percent_rounded_collateral_and_calendar_days(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        AdaptiveMacroCore5Strategy,
        "process_orders",
        lambda self, prices: None,
    )
    strategy_obj = _borrow_strategy_obj(annual_borrow_rate_float=0.01)
    pricing_data_df = pd.DataFrame(
        {("DBC", "Close"): [25.10]},
        index=[strategy_obj.current_bar],
    )

    strategy_obj.process_orders(pricing_data_df)

    expected_fee_float = 100.0 * np.ceil(1.02 * 25.10) * 0.01 * 3.0 / 360.0
    assert strategy_obj.cash == pytest.approx(50_000.0 - expected_fee_float)
    assert strategy_obj.total_value == pytest.approx(100_000.0 - expected_fee_float)
    assert strategy_obj.borrow_fee_row_dict_list[0]["calendar_day_count_int"] == 3
    assert strategy_obj.borrow_fee_row_dict_list[0]["collateral_price_float"] == 26.0


def test_zero_rate_records_exposure_without_changing_nav(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        AdaptiveMacroCore5Strategy,
        "process_orders",
        lambda self, prices: None,
    )
    strategy_obj = _borrow_strategy_obj(annual_borrow_rate_float=0.0)
    pricing_data_df = pd.DataFrame(
        {("DBC", "Close"): [25.10]},
        index=[strategy_obj.current_bar],
    )

    strategy_obj.process_orders(pricing_data_df)

    assert strategy_obj.cash == 50_000.0
    assert strategy_obj.total_value == 100_000.0
    assert strategy_obj.borrow_fee_row_dict_list[0]["borrow_fee_float"] == 0.0


def test_terminal_bar_does_not_accrue_fee_beyond_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        AdaptiveMacroCore5Strategy,
        "process_orders",
        lambda self, prices: None,
    )
    strategy_obj = _borrow_strategy_obj(annual_borrow_rate_float=0.01)
    strategy_obj.current_bar = pd.Timestamp("2024-01-08")
    pricing_data_df = pd.DataFrame(
        {("DBC", "Close"): [25.10]},
        index=[strategy_obj.current_bar],
    )

    strategy_obj.process_orders(pricing_data_df)

    assert strategy_obj.cash == 50_000.0
    assert strategy_obj.total_value == 100_000.0
    assert strategy_obj.borrow_fee_row_dict_list == []


def test_no_short_target_keeps_long_book_and_zero_dbc_short() -> None:
    strategy_obj = BorrowCostAdaptiveMacroCore5Strategy(
        annual_borrow_rate_float=0.01,
        disable_dbc_short_bool=True,
    )
    long_state_ser = pd.Series(
        {"SPY": 1.0, "IEF": 0.0, "GLD": 1.0, "DBC": 0.0, "UUP": 0.0}
    )

    target_weight_ser = strategy_obj._target_weight_ser(
        close_row_ser=pd.Series(dtype=float),
        long_state_ser=long_state_ser,
    )

    assert target_weight_ser.loc["DBC"] == 0.0
    assert target_weight_ser.loc["BIL"] == pytest.approx(0.60)
    assert target_weight_ser.loc["Cash"] == 0.0
    assert target_weight_ser.sum() == pytest.approx(1.0)


def test_frozen_gate_table_checks_all_eight_rules() -> None:
    comparison_df = pd.DataFrame(
        [
            {"variant_key_str": "borrow_0pct", "cagr_float": 0.0700, "sharpe_float": 1.20, "max_drawdown_float": -0.070},
            {"variant_key_str": "borrow_1pct", "cagr_float": 0.0695, "sharpe_float": 1.19, "max_drawdown_float": -0.069},
            {"variant_key_str": "borrow_5pct", "cagr_float": 0.0675, "sharpe_float": 1.10, "max_drawdown_float": -0.072},
            {"variant_key_str": "no_short", "cagr_float": 0.0690, "sharpe_float": 1.18, "max_drawdown_float": -0.068},
        ]
    )

    gate_df = build_gate_df(comparison_df)

    assert len(gate_df) == 8
    assert bool(gate_df["pass_bool"].all())
    external_gate_ser = gate_df.set_index("gate_key_str").loc[
        "external_plausibility"
    ]
    assert external_gate_ser["threshold_float"] == max(
        DBC_EXTERNAL_SNAPSHOT_RATE_PCT_TUPLE
    )
    assert bool(external_gate_ser["pass_bool"])
