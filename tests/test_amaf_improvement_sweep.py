import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

TEST_NORGATEDATA_ROOT = Path(__file__).resolve().parents[1] / ".tmp_norgatedata"
TEST_NORGATEDATA_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("NORGATEDATA_ROOT", str(TEST_NORGATEDATA_ROOT))

from scripts.research.run_amaf_improvement_sweep import (
    BUFFERED_VARIANT_KEY_STR,
    DECLARED_CANDIDATE_COUNT_INT,
    PrecomputedAmafResearchStrategy,
    RESEARCH_CAPITAL_BASE_FLOAT,
    RESEARCH_SPEC_DICT,
    STATIC_VARIANT_KEY_STR,
    _bundle_from_score_df,
    _comparison_row_dict,
    build_buffered_signal_bundle,
    build_classic_momentum_score_df,
    build_cost_stress_daily_return,
    build_promotion_gate_df,
    build_static_composite_score_df,
)
from strategies.momentum.adaptive_moving_average_factor import (
    AdaptiveMovingAverageFactorConfig,
    AdaptiveMovingAverageFactorSignalBundle,
    AdaptiveMovingAverageFactorStrategy,
    build_pit_intersection_universe_df,
    build_source_panel_membership_df,
    source_panel_membership_is_structurally_implied_bool,
)


def _baseline_bundle(
    forecast_by_date_dict: dict[str, dict[str, float]],
    selected_count_int: int,
) -> AdaptiveMovingAverageFactorSignalBundle:
    forecast_record_list: list[dict[str, object]] = []
    target_weight_row_dict: dict[pd.Timestamp, pd.Series] = {}
    for date_str, score_by_symbol_dict in forecast_by_date_dict.items():
        decision_date_ts = pd.Timestamp(date_str)
        ranked_symbol_list = [
            symbol_str
            for symbol_str, _score_float in sorted(
                score_by_symbol_dict.items(),
                key=lambda item_tuple: (item_tuple[1], item_tuple[0]),
            )
        ]
        selected_symbol_list = ranked_symbol_list[-selected_count_int:]
        selected_symbol_set = set(selected_symbol_list)
        target_weight_row_dict[decision_date_ts] = pd.Series(
            1.0 / selected_count_int,
            index=selected_symbol_list,
            dtype=float,
        )
        for symbol_str, score_float in score_by_symbol_dict.items():
            forecast_record_list.append(
                {
                    "decision_date_ts": decision_date_ts,
                    "symbol_str": symbol_str,
                    "forecast_float": float(score_float),
                    "quintile_int": 5 if symbol_str in selected_symbol_set else 1,
                    "selected_bool": symbol_str in selected_symbol_set,
                    "target_weight_float": (
                        1.0 / selected_count_int
                        if symbol_str in selected_symbol_set
                        else 0.0
                    ),
                }
            )
    target_weight_df = pd.DataFrame.from_dict(
        target_weight_row_dict,
        orient="index",
        dtype=float,
    ).sort_index()
    target_weight_df.index.name = "decision_date_ts"
    return AdaptiveMovingAverageFactorSignalBundle(
        target_weight_df=target_weight_df,
        forecast_df=pd.DataFrame(forecast_record_list),
        coefficient_df=pd.DataFrame(),
        coverage_df=pd.DataFrame(),
    )


def test_frozen_search_space_contains_exactly_two_candidates():
    assert DECLARED_CANDIDATE_COUNT_INT == 2
    assert RESEARCH_SPEC_DICT["candidate_variant_key_list"] == [
        BUFFERED_VARIANT_KEY_STR,
        STATIC_VARIANT_KEY_STR,
    ]
    assert RESEARCH_CAPITAL_BASE_FLOAT == 100_000_000.0
    assert (
        RESEARCH_SPEC_DICT["research_capital_base_float"]
        == RESEARCH_CAPITAL_BASE_FLOAT
    )


def test_pit_source_intersection_does_not_admit_future_source_member():
    target_date_index = pd.DatetimeIndex(
        pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    )
    target_universe_df = pd.DataFrame(
        {"AAA": [1, 1, 1]},
        index=target_date_index,
    )
    source_universe_df = pd.DataFrame(
        {"AAA": [0, 1]},
        index=pd.DatetimeIndex(
            pd.to_datetime(["2024-01-03", "2024-01-04"])
        ),
    )

    intersection_universe_df = build_pit_intersection_universe_df(
        target_universe_df=target_universe_df,
        source_universe_df=source_universe_df,
    )

    assert intersection_universe_df["AAA"].tolist() == [0, 0, 1]


def test_snapshot_source_panel_loader_keeps_only_target_symbols():
    source_universe_df = pd.DataFrame(
        {
            "AAA": [0, 1],
            "UNRELATED": [1, 1],
        },
        index=pd.DatetimeIndex(
            pd.to_datetime(["2024-01-02", "2024-01-03"])
        ),
    )
    module_path_str = (
        "strategies.momentum.adaptive_moving_average_factor"
    )
    with (
        patch(
            f"{module_path_str}.is_snapshot_mode_enabled_bool",
            return_value=True,
        ),
        patch(
            f"{module_path_str}.build_index_constituent_matrix",
            return_value=(["AAA", "UNRELATED"], source_universe_df),
        ) as build_matrix_mock_obj,
    ):
        filtered_source_df = build_source_panel_membership_df(
            source_panel_indexname_str="Russell 3000",
            target_symbol_list=["AAA"],
        )

    assert filtered_source_df.columns.tolist() == ["AAA"]
    build_matrix_mock_obj.assert_called_once_with(indexname="Russell 3000")


def test_only_russell_1000_has_structurally_implied_source_membership():
    assert source_panel_membership_is_structurally_implied_bool(
        target_indexname_str="Russell 1000",
        source_panel_indexname_str="Russell 3000",
    )
    assert not source_panel_membership_is_structurally_implied_bool(
        target_indexname_str="Nasdaq 100",
        source_panel_indexname_str="Russell 3000",
    )


def test_buffer_retains_incumbent_inside_top_thirty_percent():
    symbol_list = [f"S{symbol_int:02d}" for symbol_int in range(10)]
    first_score_dict = {
        symbol_str: float(symbol_int)
        for symbol_int, symbol_str in enumerate(symbol_list)
    }
    second_score_dict = dict(first_score_dict)
    second_score_dict["S08"] = 7.5
    second_score_dict["S07"] = 8.0
    baseline_bundle_obj = _baseline_bundle(
        {
            "2024-01-31": first_score_dict,
            "2024-02-29": second_score_dict,
        },
        selected_count_int=2,
    )

    buffered_bundle_obj = build_buffered_signal_bundle(
        baseline_bundle_obj=baseline_bundle_obj,
    )

    assert set(buffered_bundle_obj.target_weight_df.loc["2024-02-29"].dropna().index) == {
        "S08",
        "S09",
    }
    february_coverage_ser = buffered_bundle_obj.coverage_df.loc[
        buffered_bundle_obj.coverage_df["decision_date_ts"].eq(
            pd.Timestamp("2024-02-29")
        )
    ].iloc[0]
    assert int(february_coverage_ser["retained_count_int"]) == 2
    assert int(february_coverage_ser["entry_count_int"]) == 0
    assert int(february_coverage_ser["exit_count_int"]) == 0


def test_score_bundle_uses_stable_symbol_tie_break_and_exact_baseline_breadth():
    baseline_bundle_obj = _baseline_bundle(
        {
            "2024-01-31": {
                symbol_str: float(symbol_int)
                for symbol_int, symbol_str in enumerate(
                    ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
                )
            }
        },
        selected_count_int=2,
    )
    score_df = pd.DataFrame(
        1.0,
        index=[pd.Timestamp("2024-01-31")],
        columns=["J", "I", "H", "G", "F", "E", "D", "C", "B", "A"],
    )
    score_bundle_obj = _bundle_from_score_df(
        baseline_bundle_obj=baseline_bundle_obj,
        score_df=score_df,
        variant_key_str=STATIC_VARIANT_KEY_STR,
    )
    assert set(score_bundle_obj.target_weight_df.iloc[0].dropna().index) == {
        "I",
        "J",
    }
    np.testing.assert_allclose(
        score_bundle_obj.target_weight_df.sum(axis=1).to_numpy(),
        1.0,
    )


def test_static_composite_is_negative_mean_cross_sectional_zscore():
    decision_date_ts = pd.Timestamp("2024-01-05")
    date_index = pd.bdate_range("2024-01-02", periods=4)
    price_close_df = pd.DataFrame(
        {
            "A": [10.0, 11.0, 12.0, 13.0],
            "B": [10.0, 10.5, 11.0, 11.5],
            "C": [10.0, 10.0, 10.0, 10.0],
        },
        index=date_index,
    )
    baseline_bundle_obj = _baseline_bundle(
        {
            decision_date_ts.date().isoformat(): {
                "A": 1.0,
                "B": 2.0,
                "C": 3.0,
            }
        },
        selected_count_int=1,
    )
    score_df = build_static_composite_score_df(
        price_close_df=price_close_df,
        baseline_bundle_obj=baseline_bundle_obj,
        sma_lookback_tuple=(2, 3),
    )

    feature_df = pd.DataFrame(
        {
            "sma_2_ratio_float": [
                (12.0 + 13.0) / 2.0 / 13.0,
                (11.0 + 11.5) / 2.0 / 11.5,
                1.0,
            ],
            "sma_3_ratio_float": [
                (11.0 + 12.0 + 13.0) / 3.0 / 13.0,
                (10.5 + 11.0 + 11.5) / 3.0 / 11.5,
                1.0,
            ],
        },
        index=["A", "B", "C"],
    )
    expected_score_ser = -(
        (feature_df - feature_df.mean())
        / feature_df.std(ddof=0)
    ).mean(axis=1)
    pd.testing.assert_series_equal(
        score_df.loc[decision_date_ts],
        expected_score_ser,
        check_names=False,
    )


def test_classic_momentum_skips_latest_twenty_one_observed_sessions():
    date_index = pd.bdate_range("2023-01-02", periods=270)
    price_close_df = pd.DataFrame(
        {"A": np.arange(1.0, 271.0)},
        index=date_index,
    )
    decision_date_ts = date_index[-1]
    score_df = build_classic_momentum_score_df(
        price_close_df=price_close_df,
        decision_date_index=pd.DatetimeIndex([decision_date_ts]),
    )
    expected_momentum_float = (
        price_close_df["A"].iloc[-22]
        / price_close_df["A"].iloc[-253]
        - 1.0
    )
    assert np.isclose(
        float(score_df.loc[decision_date_ts, "A"]),
        expected_momentum_float,
    )


def test_cost_stress_subtracts_incremental_reference_notional_slippage():
    date_index = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    )
    result_df = pd.DataFrame(
        {
            "daily_returns": [0.01, -0.005],
            "total_value": [1_010.0, 1_004.95],
        },
        index=date_index,
    )
    transaction_df = pd.DataFrame(
        {
            "bar": [date_index[0], date_index[1]],
            "amount": [10.0, -5.0],
            "total_value": [100.025, -49.9875],
        }
    )
    strategy_obj = SimpleNamespace(
        _slippage=0.00025,
        results=result_df,
        get_transactions=lambda: transaction_df,
    )

    stressed_return_ser, incremental_cost_ser = (
        build_cost_stress_daily_return(
            strategy_obj=strategy_obj,
            target_slippage_per_side_float=0.001,
        )
    )

    expected_reference_notional_arr = np.array([100.0, 50.0])
    expected_starting_equity_arr = np.array([1_000.0, 1_010.0])
    expected_incremental_cost_arr = (
        expected_reference_notional_arr
        * (0.001 - 0.00025)
        / expected_starting_equity_arr
    )
    np.testing.assert_allclose(
        incremental_cost_ser.to_numpy(),
        expected_incremental_cost_arr,
    )
    np.testing.assert_allclose(
        stressed_return_ser.to_numpy(),
        result_df["daily_returns"].to_numpy()
        - expected_incremental_cost_arr,
    )


def test_platform_cost_tier_leaves_realized_returns_unchanged():
    date_index = pd.DatetimeIndex([pd.Timestamp("2024-01-02")])
    result_df = pd.DataFrame(
        {"daily_returns": [0.01], "total_value": [1_010.0]},
        index=date_index,
    )
    strategy_obj = SimpleNamespace(
        _slippage=0.00025,
        results=result_df,
        get_transactions=lambda: pd.DataFrame(),
    )

    platform_return_ser, incremental_cost_ser = (
        build_cost_stress_daily_return(
            strategy_obj=strategy_obj,
            target_slippage_per_side_float=0.00025,
        )
    )

    pd.testing.assert_series_equal(
        platform_return_ser,
        result_df["daily_returns"],
    )
    assert float(incremental_cost_ser.sum()) == 0.0


def test_research_transaction_buffer_preserves_engine_transaction_contract():
    strategy_obj = object.__new__(PrecomputedAmafResearchStrategy)
    strategy_obj.research_transaction_record_list = []
    strategy_obj.research_transaction_cache_df = None
    strategy_obj._position_amount_map = {}
    strategy_obj.log_audit_event = lambda _event_type_str, _payload_dict: None

    strategy_obj.add_transaction(
        7,
        pd.Timestamp("2024-01-03"),
        "AAA",
        10.0,
        100.0,
        1_000.0,
        order_id=3,
        commission=1.0,
    )
    strategy_obj.add_transaction(
        7,
        pd.Timestamp("2024-02-01"),
        "AAA",
        -4.0,
        110.0,
        -440.0,
        order_id=4,
        commission=1.0,
    )

    transaction_df = strategy_obj.get_transactions()
    assert len(transaction_df) == 2
    assert float(strategy_obj._position_amount_map["AAA"]) == 6.0
    assert float(strategy_obj._get_open_trade_amount_ser("AAA").loc[7]) == 6.0
    assert len(
        strategy_obj.get_transactions(bar=pd.Timestamp("2024-02-01"))
    ) == 1


def test_research_market_executor_matches_vanilla_market_target_fills():
    previous_bar_ts = pd.Timestamp("2024-01-31")
    current_bar_ts = pd.Timestamp("2024-02-01")
    symbol_list = ["AAA", "BBB", "SPY"]
    field_list = ["Open", "High", "Low", "Close"]
    price_column_index = pd.MultiIndex.from_product(
        [symbol_list, field_list]
    )
    pricing_data_df = pd.DataFrame(
        [
            [
                100.0, 101.0, 99.0, 100.0,
                50.0, 51.0, 49.0, 50.0,
                400.0, 401.0, 399.0, 400.0,
            ],
            [
                102.0, 104.0, 101.0, 103.0,
                49.0, 50.0, 48.0, 49.5,
                402.0, 403.0, 400.0, 401.0,
            ],
        ],
        index=[previous_bar_ts, current_bar_ts],
        columns=price_column_index,
    )
    config_obj = AdaptiveMovingAverageFactorConfig(
        strategy_name_str="amaf_executor_parity",
        variant_key_str="test",
        indexname_str="test",
        source_panel_indexname_str="test",
        benchmark_list=("SPY",),
        min_eligible_count_int=1,
    )
    universe_df = pd.DataFrame(
        {"AAA": [1], "BBB": [1]},
        index=[previous_bar_ts],
    )
    rebalance_schedule_df = pd.DataFrame(
        {"decision_date_ts": [previous_bar_ts]},
        index=[current_bar_ts],
    )
    empty_bundle_obj = AdaptiveMovingAverageFactorSignalBundle(
        target_weight_df=pd.DataFrame(),
        forecast_df=pd.DataFrame(),
        coefficient_df=pd.DataFrame(),
        coverage_df=pd.DataFrame(),
    )
    vanilla_strategy_obj = AdaptiveMovingAverageFactorStrategy(
        name="vanilla_executor",
        benchmarks=["SPY"],
        universe_df=universe_df,
        rebalance_schedule_df=rebalance_schedule_df,
        config_obj=config_obj,
    )
    research_strategy_obj = PrecomputedAmafResearchStrategy(
        name="research_executor",
        benchmarks=["SPY"],
        universe_df=universe_df,
        rebalance_schedule_df=rebalance_schedule_df,
        config_obj=config_obj,
        signal_bundle_obj=empty_bundle_obj,
    )
    for strategy_obj in [vanilla_strategy_obj, research_strategy_obj]:
        strategy_obj.previous_bar = previous_bar_ts
        strategy_obj.current_bar = current_bar_ts
        strategy_obj.order_target_percent("AAA", 0.50, trade_id=1)
        strategy_obj.order_target_percent("BBB", 0.25, trade_id=2)
        strategy_obj.process_orders(pricing_data_df)

    vanilla_transaction_df = vanilla_strategy_obj.get_transactions().reset_index(
        drop=True
    )
    research_transaction_df = research_strategy_obj.get_transactions().reset_index(
        drop=True
    )
    pd.testing.assert_frame_equal(
        research_transaction_df.drop(columns=["order_id"]),
        vanilla_transaction_df.drop(columns=["order_id"]),
        check_dtype=False,
    )
    assert np.isclose(research_strategy_obj.cash, vanilla_strategy_obj.cash)
    assert np.isclose(
        research_strategy_obj.portfolio_value,
        vanilla_strategy_obj.portfolio_value,
    )
    assert np.isclose(
        research_strategy_obj.total_value,
        vanilla_strategy_obj.total_value,
    )


def test_research_dividend_cash_path_matches_vanilla():
    previous_bar_ts = pd.Timestamp("2024-01-31")
    current_bar_ts = pd.Timestamp("2024-02-01")
    price_column_index = pd.MultiIndex.from_product(
        [["AAA", "SPY"], ["Open", "High", "Low", "Close", "Dividend"]]
    )
    pricing_data_df = pd.DataFrame(
        [
            [
                100.0, 101.0, 99.0, 100.0, 1.0,
                400.0, 401.0, 399.0, 400.0, 0.0,
            ],
            [
                99.0, 100.0, 98.0, 99.5, 0.0,
                402.0, 403.0, 400.0, 401.0, 0.0,
            ],
        ],
        index=[previous_bar_ts, current_bar_ts],
        columns=price_column_index,
    )
    config_obj = AdaptiveMovingAverageFactorConfig(
        strategy_name_str="amaf_dividend_parity",
        variant_key_str="test",
        indexname_str="test",
        source_panel_indexname_str="test",
        benchmark_list=("SPY",),
        min_eligible_count_int=1,
    )
    universe_df = pd.DataFrame({"AAA": [1]}, index=[previous_bar_ts])
    rebalance_schedule_df = pd.DataFrame(
        {"decision_date_ts": [previous_bar_ts]},
        index=[current_bar_ts],
    )
    empty_bundle_obj = AdaptiveMovingAverageFactorSignalBundle(
        target_weight_df=pd.DataFrame(),
        forecast_df=pd.DataFrame(),
        coefficient_df=pd.DataFrame(),
        coverage_df=pd.DataFrame(),
    )
    vanilla_strategy_obj = AdaptiveMovingAverageFactorStrategy(
        name="vanilla_dividend",
        benchmarks=["SPY"],
        universe_df=universe_df,
        rebalance_schedule_df=rebalance_schedule_df,
        config_obj=config_obj,
    )
    research_strategy_obj = PrecomputedAmafResearchStrategy(
        name="research_dividend",
        benchmarks=["SPY"],
        universe_df=universe_df,
        rebalance_schedule_df=rebalance_schedule_df,
        config_obj=config_obj,
        signal_bundle_obj=empty_bundle_obj,
    )
    for strategy_obj in [vanilla_strategy_obj, research_strategy_obj]:
        strategy_obj.previous_bar = previous_bar_ts
        strategy_obj.current_bar = current_bar_ts
        strategy_obj.add_transaction(
            1,
            previous_bar_ts,
            "AAA",
            10.0,
            100.0,
            1_000.0,
            1,
            0.0,
        )
        strategy_obj.cash = 99_000.0
        strategy_obj.portfolio_value = 1_000.0
        strategy_obj.total_value = 100_000.0
        strategy_obj.process_orders(pricing_data_df)

    assert np.isclose(research_strategy_obj.cash, vanilla_strategy_obj.cash)
    assert np.isclose(
        research_strategy_obj.total_value,
        vanilla_strategy_obj.total_value,
    )
    pd.testing.assert_frame_equal(
        research_strategy_obj.get_dividend_ledger(),
        vanilla_strategy_obj.get_dividend_ledger(),
        check_dtype=False,
    )


def test_comparison_reports_target_realized_positions_and_cash_separately():
    date_index = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    )
    metric_name_list = [
        "Turnover (Ann.) [%]",
        "Cost Drag (Ann.) [%]",
        "Exposure Time [%]",
    ]
    strategy_obj = SimpleNamespace(
        results=pd.DataFrame(
            {
                "daily_returns": [0.0, 0.01],
                "total_value": [100.0, 101.0],
            },
            index=date_index,
        ),
        realized_weight_df=pd.DataFrame(
            {
                "AAA": [0.75, 0.40],
                "BBB": [np.nan, 0.35],
                "Cash": [0.25, 0.25],
            },
            index=date_index,
        ),
        summary=pd.DataFrame(
            {"Strategy": [100.0, 1.0, 90.0]},
            index=metric_name_list,
        ),
        missing_price_liquidation_count_int=0,
        _slippage=0.00025,
        config_obj=SimpleNamespace(
            commission_per_share_float=0.005,
            commission_minimum_float=1.0,
        ),
    )
    signal_bundle_obj = AdaptiveMovingAverageFactorSignalBundle(
        target_weight_df=pd.DataFrame(
            {"AAA": [0.5], "BBB": [0.5]},
            index=[pd.Timestamp("2024-01-01")],
        ),
        forecast_df=pd.DataFrame(),
        coefficient_df=pd.DataFrame(),
        coverage_df=pd.DataFrame(),
    )
    daily_return_ser = strategy_obj.results["daily_returns"]
    comparison_row_dict = _comparison_row_dict(
        strategy_obj=strategy_obj,
        universe_key_str="russell1000",
        variant_key_str="amaf_baseline",
        cost_tier_key_str="platform",
        signal_bundle_obj=signal_bundle_obj,
        daily_return_ser=daily_return_ser,
        incremental_cost_fraction_ser=pd.Series(
            0.0,
            index=date_index,
        ),
    )

    assert comparison_row_dict["average_target_position_count_float"] == 2.0
    assert (
        comparison_row_dict["average_realized_position_count_float"] == 1.5
    )
    assert comparison_row_dict["average_cash_weight_pct_float"] == 25.0


def test_approximate_cost_screens_can_never_set_mechanical_pass():
    comparison_record_list: list[dict[str, object]] = []
    for universe_key_str in ["russell1000", "nasdaq100"]:
        for cost_tier_key_str in [
            "platform",
            "round_trip_20bps",
            "round_trip_50bps",
        ]:
            for (
                variant_key_str,
                annual_return_float,
                sharpe_float,
                turnover_float,
            ) in [
                ("amaf_baseline", 10.0, 1.0, 100.0),
                ("amaf_buffered_20_30", 10.0, 1.0, 50.0),
                ("static_amaf_composite", 12.0, 1.2, 100.0),
                ("eligible_equal_weight_control", 5.0, 0.5, 20.0),
            ]:
                comparison_record_list.append(
                    {
                        "universe_key_str": universe_key_str,
                        "variant_key_str": variant_key_str,
                        "cost_tier_key_str": cost_tier_key_str,
                        "annual_return_pct_float": annual_return_float,
                        "sharpe_float": sharpe_float,
                        "max_drawdown_pct_float": -10.0,
                        "turnover_ann_pct_float": turnover_float,
                        "missing_price_liquidation_count_int": 0,
                    }
                )
    inference_record_list = [
        {
            "universe_key_str": universe_key_str,
            "candidate_variant_key_str": candidate_variant_key_str,
            "cost_tier_key_str": cost_tier_key_str,
            "mean_return_delta_annual_pct_float": 1.0,
        }
        for universe_key_str in ["russell1000", "nasdaq100"]
        for candidate_variant_key_str in [
            "amaf_buffered_20_30",
            "static_amaf_composite",
        ]
        for cost_tier_key_str in [
            "platform",
            "round_trip_20bps",
            "round_trip_50bps",
        ]
    ]

    gate_df = build_promotion_gate_df(
        comparison_df=pd.DataFrame(comparison_record_list),
        inference_df=pd.DataFrame(inference_record_list),
    )

    assert gate_df["historical_screen_pass_bool"].all()
    assert not gate_df["mechanical_gate_pass_bool"].any()
    assert gate_df["mechanical_gate_block_reason_str"].str.contains(
        "exact engine reruns"
    ).all()
