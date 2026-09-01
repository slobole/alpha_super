from __future__ import annotations

from inspect import signature
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from alpha.bench import catalog
from alpha.engine.crisis import SUPPORTED_CRISIS_STRATEGY_SPEC_MAP
from alpha.engine.execution_timing import ExecutionTimingAnalyzer
from alpha.engine.strategy import Strategy
from scripts.research import run_strategy_analysis as analysis_runner
from strategies.momentum import strategy_mo_ibit_adaptive_momentum_regime as ibit_module
from strategies.momentum import strategy_mo_qqq_adaptive_momentum_regime as qqq_module
from strategies.momentum import strategy_mo_spy_adaptive_momentum_regime as spy_module


CASE_TUPLE = (
    (
        "SPY",
        "SPY_TR_SIGNAL",
        "strategy_mo_spy_adaptive_momentum_regime",
        spy_module,
        "get_spy_adaptive_momentum_regime_data",
        spy_module.SpyAdaptiveMomentumRegimeStrategy,
    ),
    (
        "QQQ",
        "QQQ_TR_SIGNAL",
        "strategy_mo_qqq_adaptive_momentum_regime",
        qqq_module,
        "get_qqq_adaptive_momentum_regime_data",
        qqq_module.QqqAdaptiveMomentumRegimeStrategy,
    ),
    (
        "IBIT",
        "IBIT_TR_SIGNAL",
        "strategy_mo_ibit_adaptive_momentum_regime",
        ibit_module,
        "get_ibit_adaptive_momentum_regime_data",
        ibit_module.IbitAdaptiveMomentumRegimeStrategy,
    ),
)


def _make_pricing_data_df(
    asset_symbol_str: str,
    signal_symbol_str: str,
) -> pd.DataFrame:
    date_idx = pd.date_range("2024-01-02", periods=8, freq="B")
    asset_close_vec = np.array(
        [100.0, 101.0, 99.0, 98.0, 103.0, 104.0, 105.0, 106.0]
    )
    signal_close_vec = np.array(
        [200.0, 202.0, 198.0, 196.0, 206.0, 208.0, 210.0, 212.0]
    )
    benchmark_close_vec = np.arange(5_000.0, 5_008.0)
    pricing_data_df = pd.DataFrame(
        {
            (asset_symbol_str, "Open"): asset_close_vec - 0.5,
            (asset_symbol_str, "High"): asset_close_vec + 1.0,
            (asset_symbol_str, "Low"): asset_close_vec - 1.0,
            (asset_symbol_str, "Close"): asset_close_vec,
            (asset_symbol_str, "Unadjusted Close"): asset_close_vec,
            (asset_symbol_str, "Dividend"): np.zeros(len(date_idx)),
            (signal_symbol_str, "Open"): signal_close_vec - 0.5,
            (signal_symbol_str, "Close"): signal_close_vec,
            ("$SPX", "Open"): benchmark_close_vec - 1.0,
            ("$SPX", "High"): benchmark_close_vec + 1.0,
            ("$SPX", "Low"): benchmark_close_vec - 1.0,
            ("$SPX", "Close"): benchmark_close_vec,
            ("$SPX", "Unadjusted Close"): benchmark_close_vec,
            ("$SPX", "Dividend"): np.zeros(len(date_idx)),
        },
        index=date_idx,
    )
    return pricing_data_df


def _make_parity_pricing_data_df(
    asset_symbol_str: str,
    signal_symbol_str: str,
) -> pd.DataFrame:
    date_idx = pd.date_range("2023-01-02", periods=280, freq="B")
    bar_number_vec = np.arange(len(date_idx), dtype=float)
    asset_close_vec = (
        100.0
        + 0.04 * bar_number_vec
        + 14.0 * np.sin(bar_number_vec / 12.0)
    )
    gap_return_vec = np.where(
        (bar_number_vec.astype(int) % 7) == 0,
        0.018,
        -0.006,
    )
    asset_open_vec = asset_close_vec * (1.0 + gap_return_vec)
    signal_close_vec = (
        200.0
        + 0.08 * bar_number_vec
        + 28.0 * np.sin(bar_number_vec / 12.0)
    )
    benchmark_close_vec = 5_000.0 + 2.0 * bar_number_vec
    return pd.DataFrame(
        {
            (asset_symbol_str, "Open"): asset_open_vec,
            (asset_symbol_str, "High"): np.maximum(asset_open_vec, asset_close_vec) + 1.0,
            (asset_symbol_str, "Low"): np.minimum(asset_open_vec, asset_close_vec) - 1.0,
            (asset_symbol_str, "Close"): asset_close_vec,
            (asset_symbol_str, "Unadjusted Close"): asset_close_vec,
            (asset_symbol_str, "Dividend"): np.zeros(len(date_idx)),
            (signal_symbol_str, "Open"): signal_close_vec * 0.999,
            (signal_symbol_str, "Close"): signal_close_vec,
            ("$SPX", "Open"): benchmark_close_vec - 1.0,
            ("$SPX", "High"): benchmark_close_vec + 1.0,
            ("$SPX", "Low"): benchmark_close_vec - 1.0,
            ("$SPX", "Close"): benchmark_close_vec,
            ("$SPX", "Unadjusted Close"): benchmark_close_vec,
            ("$SPX", "Dividend"): np.zeros(len(date_idx)),
        },
        index=date_idx,
    )


@pytest.mark.parametrize(
    (
        "asset_symbol_str",
        "signal_symbol_str",
        "strategy_key_str",
        "strategy_module_obj",
        "loader_name_str",
        "strategy_class_obj",
    ),
    CASE_TUPLE,
)
def test_bench_hooks_preserve_self_signal_factory_calendar_and_costs(
    asset_symbol_str,
    signal_symbol_str,
    strategy_key_str,
    strategy_module_obj,
    loader_name_str,
    strategy_class_obj,
):
    pricing_data_df = _make_pricing_data_df(asset_symbol_str, signal_symbol_str)
    with patch.object(
        strategy_module_obj,
        loader_name_str,
        return_value=pricing_data_df,
    ):
        timing_input_dict = strategy_module_obj.build_execution_timing_analysis_inputs()
        timing_strategy_obj = timing_input_dict["strategy_factory_fn"]()
        stress_context_dict = strategy_module_obj.build_stress_test_context_dict()
        stress_strategy_obj = strategy_module_obj.build_stress_test_strategy_obj(
            stress_context_dict
        )

        strategy_spec_obj = SUPPORTED_CRISIS_STRATEGY_SPEC_MAP[strategy_key_str]
        registered_context_dict = strategy_spec_obj.load_context_fn()
        registered_strategy_obj = strategy_spec_obj.build_strategy_fn(
            registered_context_dict
        )

    engine_signature_obj = signature(Strategy.__init__)
    for strategy_obj in (
        timing_strategy_obj,
        stress_strategy_obj,
        registered_strategy_obj,
    ):
        assert isinstance(strategy_obj, strategy_class_obj)
        assert strategy_obj.config.trade_symbol_str == asset_symbol_str
        assert strategy_obj.config.signal_symbol_str == signal_symbol_str
        assert (
            strategy_obj.config.slippage_float
            == engine_signature_obj.parameters["slippage"].default
        )
        assert (
            strategy_obj.config.commission_per_share_float
            == engine_signature_obj.parameters["commission_per_share"].default
        )
        assert (
            strategy_obj.config.commission_minimum_float
            == engine_signature_obj.parameters["commission_minimum"].default
        )

    assert timing_input_dict["pricing_data_df"] is pricing_data_df
    assert timing_input_dict["calendar_idx"].equals(pricing_data_df.index)
    assert timing_input_dict["order_generation_mode_str"] == "vanilla_current_bar"
    assert timing_input_dict["default_entry_timing_str"] == "same_open"
    assert timing_input_dict["default_exit_timing_str"] == "same_open"
    assert strategy_spec_obj.full_history_replay_bool is True


@pytest.mark.parametrize(
    (
        "asset_symbol_str",
        "signal_symbol_str",
        "_strategy_key_str",
        "strategy_module_obj",
        "loader_name_str",
        "strategy_class_obj",
    ),
    CASE_TUPLE,
)
def test_capacity_hook_reruns_requested_aum_and_date_window(
    asset_symbol_str,
    signal_symbol_str,
    _strategy_key_str,
    strategy_module_obj,
    loader_name_str,
    strategy_class_obj,
):
    pricing_data_df = _make_pricing_data_df(asset_symbol_str, signal_symbol_str)
    with patch.object(
        strategy_module_obj,
        loader_name_str,
        return_value=pricing_data_df,
    ):
        capacity_input_dict = strategy_module_obj.build_capacity_analysis_inputs(
            capital_base_float=250_000.0,
            backtest_start_date_str="2024-01-02",
            end_date_str="2024-01-11",
        )

    strategy_obj = capacity_input_dict["strategy_obj"]
    assert isinstance(strategy_obj, strategy_class_obj)
    assert strategy_obj.config.trade_symbol_str == asset_symbol_str
    assert strategy_obj.config.signal_symbol_str == signal_symbol_str
    assert strategy_obj.config.capital_base_float == 250_000.0
    assert strategy_obj.config.backtest_start_date_str == "2024-01-02"
    assert strategy_obj.config.end_date_str == "2024-01-11"
    engine_signature_obj = signature(Strategy.__init__)
    assert (
        strategy_obj.config.slippage_float
        == engine_signature_obj.parameters["slippage"].default
    )
    assert (
        strategy_obj.config.commission_per_share_float
        == engine_signature_obj.parameters["commission_per_share"].default
    )
    assert (
        strategy_obj.config.commission_minimum_float
        == engine_signature_obj.parameters["commission_minimum"].default
    )
    assert capacity_input_dict["pricing_data_df"] is pricing_data_df
    assert capacity_input_dict["execution_policy_str"] == "MOO"
    assert capacity_input_dict["impact_profile_str"] == "MOO_ETF_PROXY"


@pytest.mark.parametrize(
    (
        "asset_symbol_str",
        "signal_symbol_str",
        "_strategy_key_str",
        "strategy_module_obj",
        "loader_name_str",
        "_strategy_class_obj",
    ),
    CASE_TUPLE,
)
def test_default_timing_cell_matches_vanilla_with_gapped_opens(
    asset_symbol_str,
    signal_symbol_str,
    _strategy_key_str,
    strategy_module_obj,
    loader_name_str,
    _strategy_class_obj,
):
    pricing_data_df = _make_parity_pricing_data_df(
        asset_symbol_str,
        signal_symbol_str,
    )
    vanilla_strategy_obj = strategy_module_obj.run_variant(
        show_display_bool=False,
        save_results_bool=False,
        pricing_data_df=pricing_data_df,
    )
    with patch.object(
        strategy_module_obj,
        loader_name_str,
        return_value=pricing_data_df,
    ):
        timing_input_dict = strategy_module_obj.build_execution_timing_analysis_inputs()

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
    transaction_parity_column_list = [
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
        .loc[:, transaction_parity_column_list]
        .reset_index(drop=True),
        timing_strategy_obj.get_transactions()
        .loc[:, transaction_parity_column_list]
        .reset_index(drop=True),
    )
    assert len(vanilla_strategy_obj.get_transactions()) > 0


@pytest.mark.parametrize(
    ("_asset_symbol_str", "_signal_symbol_str", "_strategy_key_str", "strategy_module_obj", "_loader_name_str", "_strategy_class_obj"),
    CASE_TUPLE,
)
def test_all_five_bench_analyzers_resolve_without_skip(
    _asset_symbol_str,
    _signal_symbol_str,
    _strategy_key_str,
    strategy_module_obj,
    _loader_name_str,
    _strategy_class_obj,
):
    strategy_entry_obj = catalog.get_strategy_by_module(strategy_module_obj.__name__)

    assert strategy_entry_obj is not None
    assert strategy_entry_obj.has_run_variant_bool is True
    assert strategy_entry_obj.has_capacity_analysis_bool is True
    assert strategy_entry_obj.has_timing_analysis_bool is True
    for analysis_str in ("vanilla", "capacity", "timing", "risk", "stress"):
        assert analysis_runner._missing_hook_detail_str(
            strategy_module_obj,
            analysis_str,
        ) is None
