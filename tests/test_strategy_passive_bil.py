"""Focused contract tests for the research-only passive BIL control."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.taa_beyond_6040.strategy_taa_adaptive_macro_core5 import (
    AdaptiveMacroCore5Strategy,
    DEFAULT_CONFIG as CORE5_DEFAULT_CONFIG,
)
from strategies.portfolio_controls.strategy_passive_bil import (
    ASSET_STR,
    BENCHMARK_TUPLE,
    CORE5_MATCHED_PROFILE_STR,
    DEFAULT_COMMISSION_MINIMUM_FLOAT,
    DEFAULT_COMMISSION_PER_SHARE_FLOAT,
    DEFAULT_DIVIDEND_WITHHOLDING_RATE_FLOAT,
    DEFAULT_SLIPPAGE_FLOAT,
    TACTICAL_FI_MATCHED_PROFILE_STR,
    TACTICAL_FI_SLIPPAGE_FLOAT,
    affordable_share_count_int,
    run_variant,
)


def make_pricing_data_df(
    *,
    capital_price_float: float = 10.0,
    first_fill_open_float: float = 10.0,
    dividend_per_share_float: float = 0.10,
    pre_inception_row_count_int: int = 0,
) -> pd.DataFrame:
    date_index = pd.date_range("2024-01-02", periods=6, freq="B")
    column_index = pd.MultiIndex.from_product(
        [
            [ASSET_STR, BENCHMARK_TUPLE[0]],
            ["Open", "High", "Low", "Close", "Volume", "Dividend"],
        ]
    )
    pricing_data_df = pd.DataFrame(index=date_index, columns=column_index, dtype=float)
    for row_position_int, date_ts in enumerate(date_index):
        bil_open_float = (
            first_fill_open_float if row_position_int == 1 else capital_price_float
        )
        pricing_data_df.loc[date_ts, (ASSET_STR, "Open")] = bil_open_float
        pricing_data_df.loc[date_ts, (ASSET_STR, "High")] = max(
            bil_open_float,
            capital_price_float,
        )
        pricing_data_df.loc[date_ts, (ASSET_STR, "Low")] = min(
            bil_open_float,
            capital_price_float,
        )
        pricing_data_df.loc[date_ts, (ASSET_STR, "Close")] = capital_price_float
        pricing_data_df.loc[date_ts, (ASSET_STR, "Volume")] = 1_000_000.0
        pricing_data_df.loc[date_ts, (ASSET_STR, "Dividend")] = (
            dividend_per_share_float if row_position_int == 1 else 0.0
        )

        pricing_data_df.loc[date_ts, (BENCHMARK_TUPLE[0], "Open")] = 100.0
        pricing_data_df.loc[date_ts, (BENCHMARK_TUPLE[0], "High")] = 101.0
        pricing_data_df.loc[date_ts, (BENCHMARK_TUPLE[0], "Low")] = 99.0
        pricing_data_df.loc[date_ts, (BENCHMARK_TUPLE[0], "Close")] = (
            100.0 + row_position_int
        )
        pricing_data_df.loc[date_ts, (BENCHMARK_TUPLE[0], "Volume")] = 0.0
        pricing_data_df.loc[date_ts, (BENCHMARK_TUPLE[0], "Dividend")] = 0.0

    if pre_inception_row_count_int > 0:
        pre_inception_idx = date_index[:pre_inception_row_count_int]
        pricing_data_df.loc[pre_inception_idx, ASSET_STR] = np.nan

    pricing_data_df.attrs["norgate_adjustment_by_symbol_dict"] = {
        ASSET_STR: "CAPITALSPECIAL",
        BENCHMARK_TUPLE[0]: "TOTALRETURN",
    }
    pricing_data_df.attrs["benchmark_data_symbol_dict"] = {
        BENCHMARK_TUPLE[0]: "$SPXTR"
    }
    return pricing_data_df


def test_entry_uses_close_t_whole_shares_and_fills_open_t_plus_1():
    capital_base_float = 1_000.0
    pricing_data_df = make_pricing_data_df(
        capital_price_float=10.0,
        first_fill_open_float=11.0,
        dividend_per_share_float=0.0,
    )

    strategy_obj = run_variant(
        show_display_bool=False,
        save_results_bool=False,
        backtest_start_date_str="2024-01-01",
        capital_base_float=capital_base_float,
        end_date_str="2024-01-31",
        pricing_data_df=pricing_data_df,
    )

    expected_share_count_int = affordable_share_count_int(
        cash_float=capital_base_float,
        sizing_close_float=10.0,
        slippage_float=DEFAULT_SLIPPAGE_FLOAT,
        commission_per_share_float=DEFAULT_COMMISSION_PER_SHARE_FLOAT,
        commission_minimum_float=DEFAULT_COMMISSION_MINIMUM_FLOAT,
    )
    transaction_df = strategy_obj.get_transactions()
    assert len(transaction_df) == 1
    assert int(transaction_df.iloc[0]["amount"]) == expected_share_count_int
    assert float(transaction_df.iloc[0]["price"]) == pytest.approx(
        11.0 * (1.0 + DEFAULT_SLIPPAGE_FLOAT)
    )
    assert pd.Timestamp(transaction_df.iloc[0]["bar"]) == pricing_data_df.index[1]
    assert strategy_obj.entry_decision_date_ts == pricing_data_df.index[0]
    assert strategy_obj.results.index[0] == pricing_data_df.index[1]
    assert strategy_obj._accounting_policy_dict[
        "negative_cash_day_count_int"
    ] > 0
    assert strategy_obj._accounting_policy_dict["minimum_cash_float"] < 0.0
    assert strategy_obj._accounting_policy_dict[
        "negative_cash_financing_policy_str"
    ] == "not_modeled"


def test_dividend_is_withheld_and_left_in_cash_without_reinvestment():
    capital_base_float = 1_000.0
    dividend_per_share_float = 0.10
    strategy_obj = run_variant(
        show_display_bool=False,
        save_results_bool=False,
        backtest_start_date_str="2024-01-01",
        capital_base_float=capital_base_float,
        pricing_data_df=make_pricing_data_df(
            capital_price_float=10.0,
            first_fill_open_float=10.0,
            dividend_per_share_float=dividend_per_share_float,
        ),
    )

    transaction_df = strategy_obj.get_transactions()
    share_count_int = int(transaction_df.iloc[0]["amount"])
    dividend_ledger_df = strategy_obj.get_dividend_ledger()
    assert len(transaction_df) == 1
    assert len(dividend_ledger_df) == 1
    expected_gross_float = share_count_int * dividend_per_share_float
    expected_withholding_float = (
        expected_gross_float * DEFAULT_DIVIDEND_WITHHOLDING_RATE_FLOAT
    )
    assert float(dividend_ledger_df.iloc[0]["gross_dividend_cash_float"]) == (
        pytest.approx(expected_gross_float)
    )
    assert float(dividend_ledger_df.iloc[0]["withholding_cash_float"]) == (
        pytest.approx(expected_withholding_float)
    )
    assert float(dividend_ledger_df.iloc[0]["net_dividend_cash_float"]) == (
        pytest.approx(expected_gross_float - expected_withholding_float)
    )
    assert strategy_obj._accounting_policy_dict[
        "dividend_reinvestment_policy_str"
    ] == "none_cash_accumulates"


def test_close_budget_reserves_costs_when_there_is_no_positive_gap():
    capital_base_float = 997.37
    strategy_obj = run_variant(
        show_display_bool=False,
        save_results_bool=False,
        backtest_start_date_str="2024-01-01",
        capital_base_float=capital_base_float,
        pricing_data_df=make_pricing_data_df(
            capital_price_float=10.0,
            first_fill_open_float=10.0,
            dividend_per_share_float=0.0,
        ),
    )

    transaction_df = strategy_obj.get_transactions()
    assert len(transaction_df) == 1
    assert float(strategy_obj.results["cash"].min()) >= -1e-9
    assert float(transaction_df.iloc[0]["commission"]) >= 1.0
    assert float(transaction_df.iloc[0]["amount"]).is_integer()


def test_control_creates_no_synthetic_pre_inception_rows():
    pricing_data_df = make_pricing_data_df(
        dividend_per_share_float=0.0,
        pre_inception_row_count_int=2,
    )
    strategy_obj = run_variant(
        show_display_bool=False,
        save_results_bool=False,
        backtest_start_date_str="2004-01-01",
        capital_base_float=1_000.0,
        pricing_data_df=pricing_data_df,
    )

    first_real_bil_date_ts = pricing_data_df.index[2]
    first_fill_date_ts = pricing_data_df.index[3]
    assert strategy_obj.entry_decision_date_ts == first_real_bil_date_ts
    assert strategy_obj.results.index[0] == first_fill_date_ts
    assert not strategy_obj.results.index.isin(pricing_data_df.index[:2]).any()


@pytest.mark.parametrize(
    "invalid_column_tuple",
    [
        (ASSET_STR, "Close"),
        (ASSET_STR, "Dividend"),
        (BENCHMARK_TUPLE[0], "Close"),
    ],
)
def test_internal_missing_session_is_rejected_without_calendar_compression(
    invalid_column_tuple: tuple[str, str],
):
    pricing_data_df = make_pricing_data_df(dividend_per_share_float=0.0)
    pricing_data_df.loc[pricing_data_df.index[3], invalid_column_tuple] = np.nan

    with pytest.raises(RuntimeError, match="incomplete internal session"):
        run_variant(
            show_display_bool=False,
            save_results_bool=False,
            backtest_start_date_str="2024-01-01",
            capital_base_float=1_000.0,
            pricing_data_df=pricing_data_df,
        )


@pytest.mark.parametrize(
    "adjustment_by_symbol_dict",
    [
        None,
        {ASSET_STR: "TOTALRETURN", BENCHMARK_TUPLE[0]: "TOTALRETURN"},
        {ASSET_STR: "CAPITALSPECIAL", BENCHMARK_TUPLE[0]: "CAPITALSPECIAL"},
    ],
)
def test_adjustment_provenance_is_required_and_exact(
    adjustment_by_symbol_dict: dict[str, str] | None,
):
    pricing_data_df = make_pricing_data_df(dividend_per_share_float=0.0)
    if adjustment_by_symbol_dict is None:
        pricing_data_df.attrs.clear()
    else:
        pricing_data_df.attrs["norgate_adjustment_by_symbol_dict"] = (
            adjustment_by_symbol_dict
        )

    with pytest.raises(ValueError, match="explicit adjustment provenance"):
        run_variant(
            show_display_bool=False,
            save_results_bool=False,
            backtest_start_date_str="2024-01-01",
            capital_base_float=1_000.0,
            pricing_data_df=pricing_data_df,
        )


@pytest.mark.parametrize(
    "benchmark_data_symbol_dict",
    [None, {BENCHMARK_TUPLE[0]: "$SPX"}],
)
def test_total_return_benchmark_symbol_provenance_is_required(
    benchmark_data_symbol_dict: dict[str, str] | None,
):
    pricing_data_df = make_pricing_data_df(dividend_per_share_float=0.0)
    if benchmark_data_symbol_dict is None:
        pricing_data_df.attrs.pop("benchmark_data_symbol_dict")
    else:
        pricing_data_df.attrs["benchmark_data_symbol_dict"] = (
            benchmark_data_symbol_dict
        )

    with pytest.raises(ValueError, match="genuine total-return benchmark mapping"):
        run_variant(
            show_display_bool=False,
            save_results_bool=False,
            backtest_start_date_str="2024-01-01",
            capital_base_float=1_000.0,
            pricing_data_df=pricing_data_df,
        )


def test_injected_pricing_is_cut_at_frozen_end_date():
    pricing_data_df = make_pricing_data_df(dividend_per_share_float=0.0)
    frozen_end_date_str = pricing_data_df.index[3].date().isoformat()
    strategy_obj = run_variant(
        show_display_bool=False,
        save_results_bool=False,
        backtest_start_date_str="2024-01-01",
        end_date_str=frozen_end_date_str,
        capital_base_float=1_000.0,
        pricing_data_df=pricing_data_df,
    )

    assert strategy_obj.results.index[-1] == pd.Timestamp(frozen_end_date_str)


def test_tactical_fi_profile_matches_gross_dividend_cash_rate_and_cost_contract():
    pricing_data_df = make_pricing_data_df(
        capital_price_float=10.0,
        first_fill_open_float=10.0,
        dividend_per_share_float=0.10,
    )
    cash_return_ser = pd.Series(0.001, index=pricing_data_df.index)
    strategy_obj = run_variant(
        show_display_bool=False,
        save_results_bool=False,
        backtest_start_date_str="2024-01-01",
        capital_base_float=1_000.0,
        pricing_data_df=pricing_data_df,
        accounting_profile_str=TACTICAL_FI_MATCHED_PROFILE_STR,
        cash_return_ser=cash_return_ser,
    )

    dividend_ledger_df = strategy_obj.get_dividend_ledger()
    assert strategy_obj.config_obj.slippage_float == TACTICAL_FI_SLIPPAGE_FLOAT
    assert strategy_obj.config_obj.commission_per_share_float == 0.0
    assert strategy_obj.config_obj.commission_minimum_float == 0.0
    assert dividend_ledger_df["withholding_cash_float"].eq(0.0).all()
    assert strategy_obj.cash_interest_total_float > 0.0
    assert strategy_obj._accounting_policy_dict[
        "positive_cash_rate_policy_str"
    ] == "causal_DGS3MO_ACT_365"
    cash_interest_ledger_df = strategy_obj.get_cash_interest_ledger().set_index(
        "date"
    )
    dividend_ledger_df = strategy_obj.get_dividend_ledger()
    dividend_credit_date_ts = pd.Timestamp(dividend_ledger_df.iloc[0]["ex_date"])
    entitlement_date_ts = pd.Timestamp(
        dividend_ledger_df.iloc[0]["entitlement_date"]
    )
    # Vanilla calls iterate() before process_orders(), so interest for the credit
    # date uses prior-close cash and excludes that date's dividend posting.
    assert float(
        cash_interest_ledger_df.loc[
            dividend_credit_date_ts,
            "positive_cash_base_float",
        ]
    ) == pytest.approx(float(strategy_obj.results.loc[entitlement_date_ts, "cash"]))


def test_core5_control_matches_engine_auto_dividend_and_cost_accounting():
    core5_strategy_obj = AdaptiveMacroCore5Strategy()
    dividend_field_df = pd.DataFrame(
        [[0.0]],
        columns=pd.MultiIndex.from_tuples([("BIL", "Dividend")]),
    )

    assert core5_strategy_obj._dividend_cash_ledger_active_bool(dividend_field_df)
    assert core5_strategy_obj._accounting_policy_dict[
        "dividend_withholding_rate_float"
    ] == pytest.approx(DEFAULT_DIVIDEND_WITHHOLDING_RATE_FLOAT)
    assert core5_strategy_obj._accounting_policy_dict[
        "positive_cash_rate_policy_str"
    ] == "zero_percent_intentional"
    assert CORE5_DEFAULT_CONFIG.slippage_float == pytest.approx(
        DEFAULT_SLIPPAGE_FLOAT
    )
    assert CORE5_DEFAULT_CONFIG.commission_per_share_float == pytest.approx(
        DEFAULT_COMMISSION_PER_SHARE_FLOAT
    )
    assert CORE5_DEFAULT_CONFIG.commission_minimum_float == pytest.approx(
        DEFAULT_COMMISSION_MINIMUM_FLOAT
    )


def test_capital_below_one_share_stays_in_cash_without_commission():
    strategy_obj = run_variant(
        show_display_bool=False,
        save_results_bool=False,
        backtest_start_date_str="2024-01-01",
        capital_base_float=5.0,
        pricing_data_df=make_pricing_data_df(
            capital_price_float=10.0,
            first_fill_open_float=10.0,
            dividend_per_share_float=0.0,
        ),
    )

    assert len(strategy_obj.get_transactions()) == 0
    assert strategy_obj.entry_target_share_int == 0
    assert strategy_obj.results["cash"].astype(float).eq(5.0).all()
    assert strategy_obj.config_obj.accounting_profile_str == CORE5_MATCHED_PROFILE_STR
