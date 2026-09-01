from __future__ import annotations

from dataclasses import replace
from inspect import signature
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from alpha.bench import catalog
from alpha.engine.crisis import SUPPORTED_CRISIS_STRATEGY_SPEC_MAP
from alpha.engine.execution_timing import ExecutionTimingAnalyzer
from alpha.engine.strategy import Strategy
from alpha.strategy_registry import MaturityTier, tier_for
from scripts.research.run_adaptive_macro_core5_borrow_cost_study import (
    run_research_strategy,
)
from scripts.research import run_strategy_analysis as analysis_runner
from strategies.taa_beyond_6040 import strategy_taa_adaptive_macro_core5 as core5_module


def _make_pricing_data_df(period_count_int: int = 340) -> pd.DataFrame:
    date_idx = pd.date_range("2023-01-02", periods=period_count_int, freq="B")
    bar_number_vec = np.arange(period_count_int, dtype=float)
    column_data_dict: dict[tuple[str, str], np.ndarray] = {}

    risk_pattern_dict = {
        "SPY": 0.10 * bar_number_vec + 10.0 * np.sin(bar_number_vec / 18.0),
        "IEF": 0.02 * bar_number_vec + 6.0 * np.sin(bar_number_vec / 15.0 + 0.5),
        "GLD": 0.06 * bar_number_vec + 8.0 * np.sin(bar_number_vec / 21.0 + 1.0),
        "DBC": 0.01 * bar_number_vec + 18.0 * np.sin(bar_number_vec / 13.0),
        "UUP": 0.03 * bar_number_vec + 5.0 * np.sin(bar_number_vec / 25.0 + 2.0),
    }
    base_price_dict = {
        "SPY": 180.0,
        "IEF": 100.0,
        "GLD": 140.0,
        "DBC": 70.0,
        "UUP": 30.0,
        "BIL": 90.0,
    }

    for asset_str in core5_module.TRADEABLE_ASSET_TUPLE:
        if asset_str == "BIL":
            close_vec = base_price_dict[asset_str] + 0.005 * bar_number_vec
        else:
            close_vec = base_price_dict[asset_str] + risk_pattern_dict[asset_str]
        gap_return_vec = np.where(
            (bar_number_vec.astype(int) % 9) == 0,
            0.012,
            -0.003,
        )
        open_vec = close_vec * (1.0 + gap_return_vec)
        column_data_dict[(asset_str, "Open")] = open_vec
        column_data_dict[(asset_str, "High")] = np.maximum(open_vec, close_vec) + 0.5
        column_data_dict[(asset_str, "Low")] = np.minimum(open_vec, close_vec) - 0.5
        column_data_dict[(asset_str, "Close")] = close_vec
        column_data_dict[(asset_str, "Unadjusted Close")] = close_vec
        column_data_dict[(asset_str, "Dividend")] = np.zeros(period_count_int)

    for asset_str in core5_module.RISK_ASSET_TUPLE:
        signal_close_vec = (
            base_price_dict[asset_str] * 2.0
            + 2.0 * risk_pattern_dict[asset_str]
        )
        column_data_dict[
            (core5_module.signal_namespace_str(asset_str), "Close")
        ] = signal_close_vec

    benchmark_close_vec = 4_000.0 + 1.5 * bar_number_vec
    benchmark_open_vec = benchmark_close_vec * 0.999
    column_data_dict[("$SPX", "Open")] = benchmark_open_vec
    column_data_dict[("$SPX", "High")] = benchmark_close_vec + 2.0
    column_data_dict[("$SPX", "Low")] = benchmark_open_vec - 2.0
    column_data_dict[("$SPX", "Close")] = benchmark_close_vec
    column_data_dict[("$SPX", "Unadjusted Close")] = benchmark_close_vec
    column_data_dict[("$SPX", "Dividend")] = np.zeros(period_count_int)

    pricing_data_df = pd.DataFrame(column_data_dict, index=date_idx)
    pricing_data_df.attrs["norgate_adjustment_by_symbol_dict"] = {
        **{
            asset_str: "CAPITALSPECIAL"
            for asset_str in core5_module.TRADEABLE_ASSET_TUPLE
        },
        "$SPX": "TOTALRETURN",
    }
    return pricing_data_df


def test_midrank_percentile_uses_literal_tie_formula():
    severity_ser = pd.Series([0.0, 0.0, 1.0, 1.0], dtype=float)

    percentile_ser = core5_module.compute_midrank_trailing_percentile_ser(
        severity_ser,
        lookback_int=4,
    )

    # N_less=2, N_equal=2 -> (2 + (2+1)/2) / 4 = 0.875.
    assert percentile_ser.iloc[:3].isna().all()
    assert percentile_ser.iloc[3] == 0.875


def test_adaptive_average_starts_only_after_rank_warmup_and_matches_recursion():
    date_idx = pd.date_range("2024-01-02", periods=8, freq="B")
    price_ser = pd.Series(
        [100.0, 102.0, 101.0, 98.0, 99.0, 103.0, 104.0, 106.0],
        index=date_idx,
    )
    config_obj = core5_module.AdaptiveMacroCore5Config(
        percentile_lookback_int=4,
        fast_lookback_int=2,
        slow_lookback_int=6,
        price_filter_lookback_int=2,
        commodity_vol_lookback_int=3,
    )

    signal_df = core5_module.compute_adaptive_asset_signal_df(
        price_ser,
        config_obj=config_obj,
    )

    first_signal_idx_int = 3
    assert signal_df["adaptive_moving_average_ser"].iloc[:first_signal_idx_int].isna().all()
    assert (
        signal_df["adaptive_moving_average_ser"].iloc[first_signal_idx_int]
        == price_ser.iloc[first_signal_idx_int]
    )
    next_alpha_float = signal_df["adaptive_alpha_ser"].iloc[first_signal_idx_int + 1]
    expected_next_ama_float = (
        next_alpha_float * price_ser.iloc[first_signal_idx_int + 1]
        + (1.0 - next_alpha_float) * price_ser.iloc[first_signal_idx_int]
    )
    assert np.isclose(
        signal_df["adaptive_moving_average_ser"].iloc[first_signal_idx_int + 1],
        expected_next_ama_float,
    )


def test_literal_signal_formula_components_match_manual_values():
    date_idx = pd.date_range("2024-01-02", periods=8, freq="B")
    price_ser = pd.Series(
        [100.0, 110.0, 99.0, 104.5, 94.05, 96.0, 100.0, 98.0],
        index=date_idx,
    )
    config_obj = core5_module.AdaptiveMacroCore5Config(
        percentile_lookback_int=4,
        fast_lookback_int=2,
        slow_lookback_int=6,
        price_filter_lookback_int=3,
        commodity_vol_lookback_int=3,
    )

    signal_df = core5_module.compute_adaptive_asset_signal_df(
        price_ser,
        config_obj=config_obj,
    )

    expected_high_ser = price_ser.cummax()
    expected_drawdown_ser = price_ser.divide(expected_high_ser).sub(1.0)
    pd.testing.assert_series_equal(
        signal_df["reference_high_ser"],
        expected_high_ser,
        check_names=False,
    )
    pd.testing.assert_series_equal(
        signal_df["drawdown_ser"],
        expected_drawdown_ser,
        check_names=False,
    )
    assert np.isclose(signal_df["drawdown_severity_ser"].iloc[4], 0.145)

    rank_float = float(signal_df["drawdown_percentile_ser"].iloc[4])
    fast_alpha_float = 2.0 / 3.0
    slow_alpha_float = 2.0 / 7.0
    expected_alpha_float = rank_float**2 * fast_alpha_float + (
        1.0 - rank_float**2
    ) * slow_alpha_float
    assert np.isclose(signal_df["adaptive_alpha_ser"].iloc[4], expected_alpha_float)
    assert np.isclose(
        signal_df["filtered_price_ser"].iloc[4],
        price_ser.iloc[2:5].mean(),
    )
    expected_volatility_float = (
        price_ser.pct_change(fill_method=None).iloc[2:5].std(ddof=1)
        * np.sqrt(252.0)
    )
    assert np.isclose(
        signal_df["annualized_volatility_ser"].iloc[4],
        expected_volatility_float,
    )


def test_exact_sma_ama_equality_is_neither_long_nor_short():
    date_idx = pd.date_range("2024-01-02", periods=140, freq="B")
    constant_price_ser = pd.Series(100.0, index=date_idx)

    signal_df = core5_module.compute_adaptive_asset_signal_df(constant_price_ser)
    valid_signal_df = signal_df.dropna(
        subset=["long_state_ser", "short_state_ser"]
    )

    assert len(valid_signal_df) > 0
    assert valid_signal_df["long_state_ser"].eq(0.0).all()
    assert valid_signal_df["short_state_ser"].eq(0.0).all()


def test_target_weights_keep_fixed_sleeves_and_restrict_short_proceeds():
    long_state_ser = pd.Series(
        {"SPY": 1.0, "IEF": 0.0, "GLD": 1.0, "DBC": 0.0, "UUP": 1.0}
    )

    target_weight_ser = core5_module.build_target_weight_ser(
        long_state_ser=long_state_ser,
        commodity_short_state_bool=True,
        commodity_annualized_volatility_float=0.50,
    )

    assert target_weight_ser["SPY"] == 0.20
    assert target_weight_ser["IEF"] == 0.0
    assert target_weight_ser["GLD"] == 0.20
    assert target_weight_ser["UUP"] == 0.20
    assert target_weight_ser["DBC"] == -0.05
    assert np.isclose(target_weight_ser["BIL"], 0.40)
    assert target_weight_ser["Cash"] == 0.05
    assert np.isclose(target_weight_ser.sum(), 1.0)
    assert np.isclose(target_weight_ser.clip(lower=0.0).drop("Cash").sum(), 1.0)


@pytest.mark.parametrize(
    ("annualized_volatility_float", "expected_short_weight_float"),
    [(0.10, -0.10), (0.25, -0.10), (1.00, -0.025)],
)
def test_dbc_short_sizing_obeys_volatility_rule_and_ten_percent_cap(
    annualized_volatility_float: float,
    expected_short_weight_float: float,
):
    long_state_ser = pd.Series(0.0, index=core5_module.RISK_ASSET_TUPLE)

    target_weight_ser = core5_module.build_target_weight_ser(
        long_state_ser=long_state_ser,
        commodity_short_state_bool=True,
        commodity_annualized_volatility_float=annualized_volatility_float,
    )

    assert np.isclose(target_weight_ser["DBC"], expected_short_weight_float)
    assert np.isclose(target_weight_ser["Cash"], abs(expected_short_weight_float))
    assert np.isclose(target_weight_ser["BIL"], 1.0)


def test_rebalance_trigger_matrix_excludes_dbc_volatility_only_drift():
    strategy_obj = core5_module.AdaptiveMacroCore5Strategy()
    date_idx = pd.date_range("2024-01-29", periods=4, freq="B")

    def close_row_ser(
        decision_ts: pd.Timestamp,
        *,
        long_state_changed_bool: bool,
        month_end_rebalance_bool: bool,
        dbc_volatility_float: float,
    ) -> pd.Series:
        value_dict: dict[tuple[str, str], float | bool] = {
            (asset_str, "Close"): 100.0
            for asset_str in core5_module.TRADEABLE_ASSET_TUPLE
        }
        for asset_str in core5_module.RISK_ASSET_TUPLE:
            value_dict[(core5_module.signal_namespace_str(asset_str), "long_state_ser")] = 0.0
        value_dict[(core5_module.signal_namespace_str("DBC"), "short_state_ser")] = 1.0
        value_dict[
            (core5_module.signal_namespace_str("DBC"), "annualized_volatility_ser")
        ] = dbc_volatility_float
        value_dict[(core5_module.PORTFOLIO_NAMESPACE_STR, core5_module.LONG_STATE_CHANGED_FIELD_STR)] = long_state_changed_bool
        value_dict[(core5_module.PORTFOLIO_NAMESPACE_STR, core5_module.MONTH_END_REBALANCE_FIELD_STR)] = month_end_rebalance_bool
        strategy_obj.previous_bar = decision_ts
        return pd.Series(value_dict)

    with patch.object(strategy_obj, "_submit_target_orders") as submit_mock_obj:
        strategy_obj.iterate(
            pd.DataFrame(),
            close_row_ser(
                date_idx[0],
                long_state_changed_bool=False,
                month_end_rebalance_bool=False,
                dbc_volatility_float=0.50,
            ),
            pd.Series(dtype=float),
        )
        assert submit_mock_obj.call_count == 1

        strategy_obj.iterate(
            pd.DataFrame(),
            close_row_ser(
                date_idx[1],
                long_state_changed_bool=False,
                month_end_rebalance_bool=False,
                dbc_volatility_float=1.00,
            ),
            pd.Series(dtype=float),
        )
        assert submit_mock_obj.call_count == 1

        strategy_obj.iterate(
            pd.DataFrame(),
            close_row_ser(
                date_idx[2],
                long_state_changed_bool=True,
                month_end_rebalance_bool=False,
                dbc_volatility_float=1.00,
            ),
            pd.Series(dtype=float),
        )
        assert submit_mock_obj.call_count == 2

        strategy_obj.iterate(
            pd.DataFrame(),
            close_row_ser(
                date_idx[3],
                long_state_changed_bool=False,
                month_end_rebalance_bool=True,
                dbc_volatility_float=1.00,
            ),
            pd.Series(dtype=float),
        )
        assert submit_mock_obj.call_count == 3

    assert len(strategy_obj.daily_target_weight_row_dict_list) == 4
    assert len(strategy_obj.rebalance_target_weight_row_dict_list) == 3


def test_vanilla_runs_with_default_costs_and_records_rebalance_targets():
    pricing_data_df = _make_pricing_data_df()

    strategy_obj = core5_module.run_variant(
        show_display_bool=False,
        save_results_bool=False,
        pricing_data_df=pricing_data_df,
    )

    engine_signature_obj = signature(Strategy.__init__)
    assert strategy_obj._slippage == engine_signature_obj.parameters["slippage"].default
    assert (
        strategy_obj._commission_per_share
        == engine_signature_obj.parameters["commission_per_share"].default
    )
    assert (
        strategy_obj._commission_minimum
        == engine_signature_obj.parameters["commission_minimum"].default
    )
    assert len(strategy_obj.get_transactions()) > 0
    assert len(strategy_obj.rebalance_target_weight_df) > 0
    assert len(strategy_obj.daily_target_weights) >= len(
        strategy_obj.rebalance_target_weight_df
    )
    assert strategy_obj._performance_benchmark_adjustment_str == "TOTALRETURN"
    assert (
        strategy_obj._accounting_policy_dict["short_proceeds_policy_str"]
        == "restricted_cash_not_reinvested"
    )
    assert (
        strategy_obj._accounting_policy_dict["short_borrow_cost_policy_str"]
        == "fixed_annual_dbc_research_baseline"
    )
    assert (
        strategy_obj._accounting_policy_dict["annual_dbc_borrow_rate_float"]
        == core5_module.DEFAULT_ANNUAL_DBC_BORROW_RATE_FLOAT
    )
    assert len(strategy_obj.borrow_fee_df) > 0
    assert strategy_obj.borrow_fee_df["annual_borrow_rate_float"].eq(0.01).all()
    expected_borrow_fee_ser = (
        strategy_obj.borrow_fee_df["collateral_value_float"]
        * strategy_obj.borrow_fee_df["annual_borrow_rate_float"]
        * strategy_obj.borrow_fee_df["calendar_day_count_int"]
        / core5_module.BORROW_DAY_COUNT_DENOMINATOR_INT
    )
    assert np.allclose(
        strategy_obj.borrow_fee_df["borrow_fee_float"],
        expected_borrow_fee_ser,
    )
    assert strategy_obj._accounting_policy_dict["borrow_accrual_row_count_int"] == len(
        strategy_obj.borrow_fee_df
    )
    assert strategy_obj._accounting_policy_dict["borrow_fee_total_float"] == pytest.approx(
        strategy_obj.borrow_fee_df["borrow_fee_float"].sum()
    )
    daily_target_weight_df = strategy_obj.daily_target_weights
    long_book_weight_ser = (
        daily_target_weight_df.loc[
            :,
            list(core5_module.TRADEABLE_ASSET_TUPLE),
        ]
        .clip(lower=0.0)
        .sum(axis=1)
    )
    assert np.allclose(long_book_weight_ser, 1.0)
    assert daily_target_weight_df.loc[:, list(core5_module.RISK_ASSET_TUPLE)].max().max() <= 0.20
    short_dbc_bool_ser = daily_target_weight_df["DBC"] < 0.0
    assert short_dbc_bool_ser.any()
    assert np.allclose(
        daily_target_weight_df.loc[short_dbc_bool_ser, "Cash"],
        daily_target_weight_df.loc[short_dbc_bool_ser, "DBC"].abs(),
    )


def test_default_dbc_borrow_fee_uses_weekend_days_and_rounded_collateral() -> None:
    strategy_obj = core5_module.AdaptiveMacroCore5Strategy()
    strategy_obj.current_bar = pd.Timestamp("2024-01-05")
    strategy_obj.borrow_calendar_idx = pd.DatetimeIndex(
        ["2024-01-05", "2024-01-08"]
    )
    strategy_obj.cash = 50_000.0
    strategy_obj.total_value = 100_000.0
    strategy_obj._position_amount_map = {"DBC": -100.0}
    pricing_data_df = pd.DataFrame(
        {("DBC", "Close"): [25.10]},
        index=[strategy_obj.current_bar],
    )

    strategy_obj.apply_post_mark_accounting(pricing_data_df)

    expected_fee_float = 100.0 * np.ceil(1.02 * 25.10) * 0.01 * 3.0 / 360.0
    assert strategy_obj.cash == pytest.approx(50_000.0 - expected_fee_float)
    assert strategy_obj.total_value == pytest.approx(100_000.0 - expected_fee_float)
    assert strategy_obj.borrow_fee_row_dict_list[0]["calendar_day_count_int"] == 3
    assert strategy_obj.borrow_fee_row_dict_list[0]["collateral_price_float"] == 26.0


def test_zero_rate_override_and_non_short_position_do_not_charge_fee() -> None:
    zero_rate_strategy_obj = core5_module.AdaptiveMacroCore5Strategy(
        config_obj=replace(
            core5_module.DEFAULT_CONFIG,
            annual_dbc_borrow_rate_float=0.0,
        )
    )
    zero_rate_strategy_obj.current_bar = pd.Timestamp("2024-01-05")
    zero_rate_strategy_obj._position_amount_map = {"DBC": -100.0}
    pricing_data_df = pd.DataFrame(
        {("DBC", "Close"): [25.10]},
        index=[zero_rate_strategy_obj.current_bar],
    )

    zero_rate_strategy_obj.apply_post_mark_accounting(pricing_data_df)

    assert zero_rate_strategy_obj.borrow_fee_row_dict_list == []
    assert zero_rate_strategy_obj.borrow_fee_total_float == 0.0

    default_strategy_obj = core5_module.AdaptiveMacroCore5Strategy()
    default_strategy_obj.current_bar = pd.Timestamp("2024-01-05")
    default_strategy_obj._position_amount_map = {"DBC": 100.0}
    default_strategy_obj.apply_post_mark_accounting(pricing_data_df)

    assert default_strategy_obj.borrow_fee_row_dict_list == []
    assert default_strategy_obj.borrow_fee_total_float == 0.0


def test_formal_one_percent_baseline_matches_research_oracle() -> None:
    pricing_data_df = _make_pricing_data_df()
    calendar_idx = core5_module.build_execution_calendar_idx(pricing_data_df)

    formal_strategy_obj = core5_module.run_variant(
        show_display_bool=False,
        save_results_bool=False,
        pricing_data_df=pricing_data_df,
    )
    research_strategy_obj = run_research_strategy(
        pricing_data_df=pricing_data_df,
        calendar_idx=calendar_idx,
        config_obj=core5_module.DEFAULT_CONFIG,
        annual_borrow_rate_float=0.01,
        disable_dbc_short_bool=False,
        show_progress_bool=False,
    )

    parity_column_list = [
        "portfolio_value",
        "cash",
        "total_value",
        "daily_returns",
        "total_returns",
        "$SPX",
    ]
    pd.testing.assert_frame_equal(
        formal_strategy_obj.results.loc[:, parity_column_list],
        research_strategy_obj.results.loc[:, parity_column_list],
        check_freq=False,
    )
    fee_column_list = [
        "accrual_start_date_ts",
        "next_session_date_ts",
        "calendar_day_count_int",
        "dbc_share_float",
        "collateral_price_float",
        "collateral_value_float",
        "annual_borrow_rate_float",
        "borrow_fee_float",
    ]
    pd.testing.assert_frame_equal(
        formal_strategy_obj.borrow_fee_df.loc[:, fee_column_list].reset_index(drop=True),
        research_strategy_obj.borrow_fee_df.loc[:, fee_column_list].reset_index(drop=True),
    )


def test_truncated_run_calendar_does_not_accrue_borrow_beyond_terminal_bar() -> None:
    pricing_data_df = _make_pricing_data_df()
    full_strategy_obj = core5_module.run_variant(
        show_display_bool=False,
        save_results_bool=False,
        pricing_data_df=pricing_data_df,
    )
    terminal_bar_ts = pd.Timestamp(
        full_strategy_obj.borrow_fee_df.iloc[0]["accrual_start_date_ts"]
    )
    full_calendar_idx = core5_module.build_execution_calendar_idx(pricing_data_df)
    truncated_calendar_idx = full_calendar_idx[full_calendar_idx <= terminal_bar_ts]
    truncated_strategy_obj = core5_module._build_strategy_obj(
        core5_module.DEFAULT_CONFIG,
        full_calendar_idx,
    )

    core5_module.run_daily(
        truncated_strategy_obj,
        pricing_data_df,
        calendar=truncated_calendar_idx,
        show_progress=False,
        show_signal_progress_bool=False,
        audit_override_bool=False,
    )

    assert truncated_strategy_obj.borrow_calendar_idx.equals(truncated_calendar_idx)
    if len(truncated_strategy_obj.borrow_fee_df) > 0:
        assert pd.Timestamp(
            truncated_strategy_obj.borrow_fee_df["accrual_start_date_ts"].max()
        ) < terminal_bar_ts
        assert pd.Timestamp(
            truncated_strategy_obj.borrow_fee_df["next_session_date_ts"].max()
        ) <= terminal_bar_ts


def test_close_order_recovers_filled_trade_id_after_delayed_intent_state():
    strategy_obj = core5_module.run_variant(
        show_display_bool=False,
        save_results_bool=False,
        pricing_data_df=_make_pricing_data_df(),
    )
    held_position_ser = strategy_obj.get_positions().loc[
        lambda position_ser: position_ser.ne(0.0)
    ]
    asset_str = str(held_position_ser.index[0])
    current_share_int = int(held_position_ser.iloc[0])
    expected_trade_id_int = int(
        strategy_obj.get_latest_transaction(asset_str)["trade_id"]
    )
    strategy_obj.current_trade_id_map[asset_str] = core5_module.default_trade_id_int()
    strategy_obj.clear_orders()

    strategy_obj._queue_close_order(asset_str, current_share_int)

    order_list = list(strategy_obj.get_orders())
    assert len(order_list) == 1
    assert order_list[0].trade_id == expected_trade_id_int


def test_missing_required_close_after_activation_fails_loud():
    pricing_data_df = _make_pricing_data_df()
    strategy_obj = core5_module.AdaptiveMacroCore5Strategy()
    signal_data_df = strategy_obj.compute_signals(pricing_data_df)
    calendar_idx = core5_module.build_execution_calendar_idx(pricing_data_df)
    execution_bar_ts = pd.Timestamp(calendar_idx[0])
    previous_bar_ts = pd.Timestamp(
        pricing_data_df.index[pricing_data_df.index.get_loc(execution_bar_ts) - 1]
    )
    strategy_obj.previous_bar = previous_bar_ts
    strategy_obj.current_bar = execution_bar_ts
    strategy_obj.initialized_bool = True
    strategy_obj.last_target_weight_ser = pd.Series(
        0.0,
        index=[*core5_module.TRADEABLE_ASSET_TUPLE, "Cash"],
        dtype=float,
    )
    close_row_ser = signal_data_df.loc[previous_bar_ts].copy()
    close_row_ser.loc[("BIL", "Close")] = np.nan

    with pytest.raises(RuntimeError, match="execution-price snapshot"):
        strategy_obj.iterate(
            signal_data_df.loc[:previous_bar_ts],
            close_row_ser,
            pd.Series(dtype=float),
        )


def test_default_timing_cell_matches_vanilla_on_gapped_opens():
    pricing_data_df = _make_pricing_data_df()
    vanilla_strategy_obj = core5_module.run_variant(
        show_display_bool=False,
        save_results_bool=False,
        pricing_data_df=pricing_data_df,
    )
    with patch.object(
        core5_module,
        "get_adaptive_macro_core5_data",
        return_value=pricing_data_df,
    ):
        timing_input_dict = core5_module.build_execution_timing_analysis_inputs()

    timing_factory_strategy_obj = timing_input_dict["strategy_factory_fn"]()
    assert timing_factory_strategy_obj.config_obj.annual_dbc_borrow_rate_float == 0.01
    assert timing_factory_strategy_obj.borrow_calendar_idx.equals(
        timing_input_dict["calendar_idx"]
    )

    timing_result_obj = ExecutionTimingAnalyzer(
        strategy_factory_fn=timing_input_dict["strategy_factory_fn"],
        pricing_data_df=timing_input_dict["pricing_data_df"],
        calendar_idx=timing_input_dict["calendar_idx"],
        entry_timing_str_tuple=("same_open",),
        exit_timing_str_tuple=("same_open",),
        save_output_bool=False,
        audit_override_bool=None,
        order_generation_mode_str=timing_input_dict["order_generation_mode_str"],
        risk_model_str=timing_input_dict["risk_model_str"],
        default_entry_timing_str="same_open",
        default_exit_timing_str="same_open",
    ).run()
    timing_strategy_obj = timing_result_obj.strategy_map[("same_open", "same_open")]

    assert timing_result_obj.metric_df.loc[0, "Risk Label"] == "Clean"
    parity_column_list = [
        "portfolio_value",
        "cash",
        "total_value",
        "daily_returns",
        "total_returns",
        "$SPX",
    ]
    pd.testing.assert_frame_equal(
        vanilla_strategy_obj.results.loc[:, parity_column_list],
        timing_strategy_obj.results.loc[:, parity_column_list],
        check_freq=False,
    )
    transaction_column_list = [
        "trade_id",
        "bar",
        "asset",
        "amount",
        "price",
        "total_value",
        "commission",
    ]
    pd.testing.assert_frame_equal(
        vanilla_strategy_obj.get_transactions()
        .loc[:, transaction_column_list]
        .reset_index(drop=True),
        timing_strategy_obj.get_transactions()
        .loc[:, transaction_column_list]
        .reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        vanilla_strategy_obj.borrow_fee_df.reset_index(drop=True),
        timing_strategy_obj.borrow_fee_df.reset_index(drop=True),
    )
    assert (
        timing_strategy_obj._accounting_policy_dict["borrow_accrual_row_count_int"]
        == len(timing_strategy_obj.borrow_fee_df)
    )
    assert timing_strategy_obj._accounting_policy_dict[
        "borrow_fee_total_float"
    ] == pytest.approx(timing_strategy_obj.borrow_fee_df["borrow_fee_float"].sum())


def test_first_actionable_close_executes_on_the_next_session_open():
    pricing_data_df = _make_pricing_data_df()
    strategy_obj = core5_module.AdaptiveMacroCore5Strategy()
    signal_data_df = strategy_obj.compute_signals(pricing_data_df)
    valid_decision_bool_ser = pd.Series(True, index=pricing_data_df.index)
    for asset_str in core5_module.RISK_ASSET_TUPLE:
        valid_decision_bool_ser &= signal_data_df[
            (core5_module.signal_namespace_str(asset_str), "long_state_ser")
        ].notna()
    valid_decision_bool_ser &= signal_data_df[
        (core5_module.signal_namespace_str("DBC"), "annualized_volatility_ser")
    ].notna()
    first_decision_ts = pd.Timestamp(
        valid_decision_bool_ser[valid_decision_bool_ser].index[0]
    )
    expected_execution_ts = pd.Timestamp(
        pricing_data_df.index[pricing_data_df.index.get_loc(first_decision_ts) + 1]
    )

    calendar_idx = core5_module.build_execution_calendar_idx(pricing_data_df)

    assert calendar_idx[0] == expected_execution_ts


def test_data_loader_keeps_total_return_signals_separate_from_execution_prices():
    pricing_data_df = _make_pricing_data_df(20)
    execution_column_list = [
        column_tuple
        for column_tuple in pricing_data_df.columns
        if column_tuple[0] in (*core5_module.TRADEABLE_ASSET_TUPLE, "$SPX")
    ]
    execution_price_df = pricing_data_df.loc[:, execution_column_list].copy()
    total_return_signal_df = pd.DataFrame(
        {
            (asset_str, "Close"): pricing_data_df[
                (core5_module.signal_namespace_str(asset_str), "Close")
            ]
            for asset_str in core5_module.RISK_ASSET_TUPLE
        },
        index=pricing_data_df.index,
    )

    with patch.object(
        core5_module,
        "load_raw_prices",
        side_effect=[execution_price_df, total_return_signal_df],
    ) as load_mock_obj:
        loaded_price_df = core5_module.get_adaptive_macro_core5_data()

    assert load_mock_obj.call_count == 2
    assert load_mock_obj.call_args_list[0].kwargs["symbols"] == list(
        core5_module.TRADEABLE_ASSET_TUPLE
    )
    assert load_mock_obj.call_args_list[1].kwargs["symbols"] == []
    assert load_mock_obj.call_args_list[1].kwargs["benchmarks"] == list(
        core5_module.RISK_ASSET_TUPLE
    )
    for asset_str in core5_module.RISK_ASSET_TUPLE:
        pd.testing.assert_series_equal(
            loaded_price_df[(core5_module.signal_namespace_str(asset_str), "Close")],
            total_return_signal_df[(asset_str, "Close")],
            check_names=False,
        )
    assert loaded_price_df.attrs["signal_adjustment_by_symbol_dict"] == {
        core5_module.signal_namespace_str(asset_str): "TOTALRETURN"
        for asset_str in core5_module.RISK_ASSET_TUPLE
    }


def test_pm_ready_and_all_bench_hooks_are_registered():
    strategy_entry_obj = catalog.get_strategy_by_module(core5_module.__name__)

    assert tier_for(core5_module.__name__) is MaturityTier.PM_READY
    assert strategy_entry_obj is not None
    assert strategy_entry_obj.is_pm_ready_bool is True
    assert strategy_entry_obj.is_wired_bool is False
    assert strategy_entry_obj.has_run_variant_bool is True
    assert strategy_entry_obj.has_capacity_analysis_bool is True
    assert strategy_entry_obj.has_timing_analysis_bool is True
    for analysis_str in ("vanilla", "capacity", "timing", "risk", "stress"):
        assert analysis_runner._missing_hook_detail_str(
            core5_module,
            analysis_str,
        ) is None

    strategy_spec_obj = SUPPORTED_CRISIS_STRATEGY_SPEC_MAP[
        core5_module.STRATEGY_NAME_STR
    ]
    assert strategy_spec_obj.full_history_replay_bool is True


def test_capacity_hook_honors_requested_capital_and_dates():
    pricing_data_df = _make_pricing_data_df()
    with patch.object(
        core5_module,
        "get_adaptive_macro_core5_data",
        return_value=pricing_data_df,
    ):
        capacity_input_dict = core5_module.build_capacity_analysis_inputs(
            show_display_bool=False,
            backtest_start_date_str="2023-08-01",
            capital_base_float=250_000.0,
            end_date_str="2024-04-19",
        )

    strategy_obj = capacity_input_dict["strategy_obj"]
    assert strategy_obj._capital_base == 250_000.0
    assert strategy_obj.config_obj.annual_dbc_borrow_rate_float == 0.01
    assert strategy_obj._accounting_policy_dict["annual_dbc_borrow_rate_float"] == 0.01
    assert strategy_obj.config_obj.end_date_str == "2024-04-19"
    assert strategy_obj.results.index[0] >= pd.Timestamp("2023-08-01")
    assert capacity_input_dict["pricing_data_df"] is pricing_data_df
    assert capacity_input_dict["execution_policy_str"] == "MOO"
    assert capacity_input_dict["impact_profile_str"] == "MOO_ETF_PROXY"


def test_stress_factory_preserves_vanilla_strategy_contract():
    pricing_data_df = _make_pricing_data_df()
    with patch.object(
        core5_module,
        "get_adaptive_macro_core5_data",
        return_value=pricing_data_df,
    ):
        context_dict = core5_module.build_stress_test_context_dict()
        strategy_obj = core5_module.build_stress_test_strategy_obj(context_dict)

    assert isinstance(strategy_obj, core5_module.AdaptiveMacroCore5Strategy)
    assert strategy_obj.config_obj.annual_dbc_borrow_rate_float == 0.01
    assert strategy_obj._accounting_policy_dict["annual_dbc_borrow_rate_float"] == 0.01
    assert context_dict["pricing_data_df"] is pricing_data_df
    assert context_dict["calendar_idx"].equals(
        core5_module.build_execution_calendar_idx(pricing_data_df)
    )
