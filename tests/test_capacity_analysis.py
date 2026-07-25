import json

import numpy as np
import pandas as pd
import pytest

from alpha.engine.capacity_analysis import (
    BASELINE_SLIPPAGE_BPS_FLOAT,
    CAPACITY_CURVE_CSV_FILENAME_STR,
    CAPACITY_ORDER_CSV_FILENAME_STR,
    CAPACITY_MODEL_VERSION_STR,
    FULL_HISTORY_WINDOW_STR,
    METADATA_FILENAME_STR,
    MOC_CENTRAL_LAMBDA_1PCT_ADV_BPS_FLOAT,
    MOC_HARD_ORDER_ADV_LIMIT_FLOAT,
    MOC_SOFT_ORDER_ADV_LIMIT_FLOAT,
    MOO_HARD_ORDER_ADV_LIMIT_FLOAT,
    MOO_IMPACT_PROFILE_DICT,
    MOO_LARGE_MIXED_PROFILE_STR,
    RECENT_FIVE_YEAR_WINDOW_STR,
    MOO_SOFT_ORDER_ADV_LIMIT_FLOAT,
    REPORT_FILENAME_STR,
    SUMMARY_FILENAME_STR,
    CapacityAnalysis,
    CapacityRunResult,
    _adjusted_equity_ser,
    _benchmark_annual_return_tuple,
    _break_even_bracket_str,
    _eligible_rolling_sharpe_erosion_tuple,
    build_capacity_study_result,
    capacity_implicit_cost_bps_float,
    normalize_execution_policy_str,
    normalize_impact_profile_str,
    policy_limit_tuple,
    square_root_impact_bps_float,
)
from alpha.engine.strategy import Strategy


class ToyStrategy(Strategy):
    def iterate(self, data: pd.DataFrame, close: pd.DataFrame, open_prices: pd.Series):
        return None


def _pricing_data_df(bar_count_int: int = 900, turnover_float: float = 1_000_000.0) -> pd.DataFrame:
    date_idx = pd.date_range("2020-01-02", periods=bar_count_int, freq="B")
    benchmark_close_ser = pd.Series(
        100.0 * np.cumprod(np.full(bar_count_int, 1.0002)),
        index=date_idx,
    )
    pricing_data_df = pd.DataFrame(
        {
            ("AAA", "Close"): np.full(bar_count_int, 10.0),
            ("AAA", "Volume"): np.full(bar_count_int, turnover_float / 10.0),
            ("AAA", "Turnover"): np.full(bar_count_int, turnover_float),
            ("$SPX", "Close"): benchmark_close_ser,
        },
        index=date_idx,
    )
    pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
    return pricing_data_df


def _strategy_obj(
    pricing_data_df: pd.DataFrame,
    capital_base_float: float = 100_000.0,
    order_notional_float: float = 1_000.0,
) -> ToyStrategy:
    strategy_obj = ToyStrategy(
        name="toy_capacity",
        benchmarks=["$SPX"],
        capital_base=capital_base_float,
        slippage=0.00025,
        commission_per_share=0.005,
        commission_minimum=1.0,
        performance_benchmark_symbol_str="$SPX",
        performance_benchmark_adjustment_str="TOTALRETURN",
    )
    order_bar_ts = pricing_data_df.index[-1]
    strategy_obj._transactions = pd.DataFrame(
        [
            {
                "trade_id": 1,
                "bar": order_bar_ts,
                "asset": "AAA",
                "amount": order_notional_float / 10.0,
                "price": 10.0,
                "total_value": order_notional_float,
                "order_id": 1,
                "commission": 1.0,
            }
        ]
    )
    total_value_ser = pd.Series(
        capital_base_float * np.cumprod(np.full(len(pricing_data_df), 1.0006)),
        index=pricing_data_df.index,
    )
    strategy_obj.results = pd.DataFrame(
        {
            "total_value": total_value_ser,
            "portfolio_value": total_value_ser,
        }
    )
    return strategy_obj


def test_policy_and_square_root_model_are_explicit():
    assert normalize_execution_policy_str("moo") == "MOO"
    assert normalize_execution_policy_str("MOC") == "MOC"
    with pytest.raises(ValueError, match="MOO.*MOC"):
        normalize_execution_policy_str("next_open")

    assert policy_limit_tuple("MOO") == (
        MOO_SOFT_ORDER_ADV_LIMIT_FLOAT,
        MOO_HARD_ORDER_ADV_LIMIT_FLOAT,
    )
    assert policy_limit_tuple("MOC") == (
        MOC_SOFT_ORDER_ADV_LIMIT_FLOAT,
        MOC_HARD_ORDER_ADV_LIMIT_FLOAT,
    )
    assert square_root_impact_bps_float(
        0.01,
        MOC_CENTRAL_LAMBDA_1PCT_ADV_BPS_FLOAT,
    ) == pytest.approx(MOC_CENTRAL_LAMBDA_1PCT_ADV_BPS_FLOAT)
    assert square_root_impact_bps_float(0.0025, 8.2) == pytest.approx(4.1)


def test_moc_and_moo_profiles_use_floor_and_square_root_impact():
    assert capacity_implicit_cost_bps_float(0.0001, "MOC") == pytest.approx(
        BASELINE_SLIPPAGE_BPS_FLOAT
    )
    assert capacity_implicit_cost_bps_float(0.01, "MOC") == pytest.approx(8.2)
    assert capacity_implicit_cost_bps_float(0.01, "MOC", stress_bool=True) == pytest.approx(17.8)
    assert capacity_implicit_cost_bps_float(
        0.01,
        "MOO",
        impact_profile_str=MOO_LARGE_MIXED_PROFILE_STR,
    ) == pytest.approx(40.0)
    assert capacity_implicit_cost_bps_float(
        0.01,
        "MOO",
        stress_bool=True,
        impact_profile_str=MOO_LARGE_MIXED_PROFILE_STR,
    ) == pytest.approx(66.4)
    with pytest.raises(ValueError, match="requires impact_profile_str"):
        normalize_impact_profile_str("MOO", None)
    with pytest.raises(ValueError, match="Supported profiles"):
        normalize_impact_profile_str("MOO", "not_a_profile")
    assert normalize_impact_profile_str("MOC", None) is None


@pytest.mark.parametrize("profile_str", sorted(MOO_IMPACT_PROFILE_DICT))
def test_each_moo_profile_matches_lambda_at_one_percent_adv(profile_str):
    profile_dict = MOO_IMPACT_PROFILE_DICT[profile_str]
    assert capacity_implicit_cost_bps_float(
        0.01,
        "MOO",
        impact_profile_str=profile_str,
    ) == pytest.approx(profile_dict["central_lambda_1pct_adv_bps_float"])
    assert capacity_implicit_cost_bps_float(
        0.01,
        "MOO",
        stress_bool=True,
        impact_profile_str=profile_str,
    ) == pytest.approx(profile_dict["stress_lambda_1pct_adv_bps_float"])


def test_moo_impact_is_monotonic_and_incremental_cost_has_floor_crossover():
    below_floor_cost_float = capacity_implicit_cost_bps_float(
        0.00001,
        "MOO",
        impact_profile_str=MOO_LARGE_MIXED_PROFILE_STR,
    )
    larger_order_cost_float = capacity_implicit_cost_bps_float(
        0.001,
        "MOO",
        impact_profile_str=MOO_LARGE_MIXED_PROFILE_STR,
    )
    assert below_floor_cost_float == pytest.approx(BASELINE_SLIPPAGE_BPS_FLOAT)
    assert larger_order_cost_float > below_floor_cost_float


def test_lagged_adv_requires_both_windows_and_uses_lower_value():
    pricing_data_df = _pricing_data_df(bar_count_int=22)
    pricing_data_df.loc[pricing_data_df.index[-11:-1], ("AAA", "Turnover")] = 2_000_000.0
    strategy_obj = _strategy_obj(
        pricing_data_df,
        order_notional_float=100_000.0,
    )
    result_obj = CapacityAnalysis(strategy_obj, pricing_data_df, "MOC").run()
    order_ser = result_obj.order_diagnostics_df.iloc[0]

    assert order_ser["adv10_mean_dollar_lagged_float"] == pytest.approx(2_000_000.0)
    assert order_ser["adv20_median_dollar_lagged_float"] == pytest.approx(1_500_000.0)
    assert order_ser["robust_adv_dollar_lagged_float"] == pytest.approx(1_500_000.0)
    assert order_ser["order_adv_ratio_float"] == pytest.approx(1 / 15)
    assert order_ser["dollar_volume_source_str"] == "Norgate Turnover"


def test_same_asset_date_and_side_orders_are_aggregated():
    pricing_data_df = _pricing_data_df(bar_count_int=30)
    strategy_obj = _strategy_obj(pricing_data_df, order_notional_float=1_000.0)
    duplicate_df = strategy_obj._transactions.copy()
    duplicate_df["total_value"] = 2_000.0
    duplicate_df["amount"] = 200.0
    duplicate_df["commission"] = 2.0
    strategy_obj._transactions = pd.concat([strategy_obj._transactions, duplicate_df], ignore_index=True)

    result_obj = CapacityAnalysis(
        strategy_obj,
        pricing_data_df,
        "MOO",
        MOO_LARGE_MIXED_PROFILE_STR,
    ).run()
    assert len(result_obj.order_diagnostics_df) == 1
    order_ser = result_obj.order_diagnostics_df.iloc[0]
    assert order_ser["order_notional_float"] == pytest.approx(3_000.0)
    assert order_ser["commission_float"] == pytest.approx(3.0)
    assert order_ser["source_transaction_count_int"] == 2


def test_capacity_overlay_never_improves_baseline():
    pricing_data_df = _pricing_data_df()
    strategy_obj = _strategy_obj(
        pricing_data_df,
        order_notional_float=250_000.0,
    )
    result_obj = CapacityAnalysis(strategy_obj, pricing_data_df, "MOC").run()

    assert result_obj.summary_dict["central_incremental_cost_float"] >= 0.0
    assert (
        result_obj.summary_dict["central_annual_return_float"]
        <= result_obj.summary_dict["baseline_annual_return_float"]
    )
    assert (
        result_obj.summary_dict["central_sharpe_float"]
        <= result_obj.summary_dict["baseline_sharpe_float"]
    )


def test_moo_stress_equity_never_exceeds_central_or_baseline():
    pricing_data_df = _pricing_data_df(turnover_float=100_000.0)
    strategy_obj = _strategy_obj(pricing_data_df, order_notional_float=50_000.0)
    result_obj = CapacityAnalysis(
        strategy_obj,
        pricing_data_df,
        "MOO",
        MOO_LARGE_MIXED_PROFILE_STR,
    ).run()
    equity_curve_df = result_obj.equity_curve_df
    assert (
        equity_curve_df["central_equity_float"]
        <= equity_curve_df["baseline_equity_float"] + 1e-12
    ).all()
    assert (
        equity_curve_df["stress_equity_float"]
        <= equity_curve_df["central_equity_float"] + 1e-12
    ).all()
    assert equity_curve_df["central_equity_float"].iloc[-1] < equity_curve_df[
        "baseline_equity_float"
    ].iloc[-1]
    assert equity_curve_df["stress_equity_float"].iloc[-1] < equity_curve_df[
        "central_equity_float"
    ].iloc[-1]


def test_early_capacity_cost_reduces_later_compounding():
    date_idx = pd.date_range("2020-01-02", periods=3, freq="B")
    baseline_equity_ser = pd.Series([100.0, 200.0, 400.0], index=date_idx)
    cost_df = pd.DataFrame(
        {"bar": [date_idx[0]], "central_incremental_cost_float": [10.0]}
    )

    adjusted_equity_ser = _adjusted_equity_ser(
        baseline_equity_ser,
        cost_df,
        "central_incremental_cost_float",
    )

    assert adjusted_equity_ser.tolist() == pytest.approx([90.0, 180.0, 360.0])


def _manual_run_result(
    capital_base_float: float,
    execution_policy_str: str,
    central_benchmark_excess_return_float: float,
    recommended_pass_inputs_bool: bool,
) -> CapacityRunResult:
    pricing_data_df = _pricing_data_df()
    strategy_obj = _strategy_obj(pricing_data_df, capital_base_float=capital_base_float)
    soft_limit_float, hard_limit_float = policy_limit_tuple(execution_policy_str)
    central_lambda_float = 40.0 if execution_policy_str == "MOO" else 8.2
    stress_lambda_float = 66.4 if execution_policy_str == "MOO" else 17.8
    good_p95_float = soft_limit_float * 0.5
    good_p99_float = hard_limit_float * 0.5
    summary_dict = {
        "strategy_name_str": "toy_capacity",
        "analysis_type_str": "capacity_analysis",
        "capital_base_float": capital_base_float,
        "execution_policy_str": execution_policy_str,
        "impact_profile_str": (
            MOO_LARGE_MIXED_PROFILE_STR if execution_policy_str == "MOO" else None
        ),
        "central_lambda_1pct_adv_bps_float": central_lambda_float,
        "stress_lambda_1pct_adv_bps_float": stress_lambda_float,
        "model_confidence_str": "medium",
        "proxy_bool": False,
        "benchmark_annual_return_float": 0.05,
        "order_adv_p95_float": good_p95_float if recommended_pass_inputs_bool else soft_limit_float * 2,
        "order_adv_p99_float": good_p99_float if recommended_pass_inputs_bool else hard_limit_float * 2,
        "soft_limit_float": soft_limit_float,
        "hard_limit_float": hard_limit_float,
        "soft_breach_share_float": 0.0,
        "hard_breach_share_float": 0.0 if recommended_pass_inputs_bool else 0.10,
        "baseline_sharpe_float": 1.0,
        "central_sharpe_float": 0.9,
        "stress_sharpe_float": 0.8,
        "sharpe_erosion_float": 0.10,
        "central_cost_consumption_of_benchmark_excess_float": 0.10,
        "stress_cost_consumption_of_benchmark_excess_float": 0.20,
        "baseline_benchmark_excess_return_float": 0.10,
        "central_benchmark_excess_return_float": central_benchmark_excess_return_float,
        "stress_benchmark_excess_return_float": central_benchmark_excess_return_float - 0.01,
        "worst_eligible_rolling_3y_sharpe_erosion_float": 0.10,
        "rolling_3y_eligible_window_count_int": 10,
        "rolling_3y_available_bool": True,
        "actual_start_date_str": str(pricing_data_df.index[0].date()),
        "actual_end_date_str": str(pricing_data_df.index[-1].date()),
        "total_order_count_int": 1,
        "assessed_order_count_int": 1,
        "unavailable_order_count_int": 0,
        "unavailable_order_share_float": 0.0,
        "liquidity_complete_bool": True,
        "order_adv_p50_float": good_p95_float,
        "order_adv_max_float": good_p99_float,
        "baseline_annual_return_float": 0.15,
        "central_annual_return_float": 0.14,
        "stress_annual_return_float": 0.13,
        "stress_sharpe_erosion_float": 0.2,
        "central_incremental_cost_float": 10.0,
        "stress_incremental_cost_float": 20.0,
        "academic_extrapolation_share_float": 0.0,
        "proxy_extrapolation_share_float": 0.0,
        "model_extrapolation_share_float": 0.0,
    }
    order_diagnostics_df = pd.DataFrame(
        [
            {
                "bar": pricing_data_df.index[-1],
                "asset_str": "AAA",
                "side_str": "Buy",
                "assessed_bool": True,
                "order_adv_ratio_float": good_p95_float,
                "order_notional_float": 1_000.0,
                "robust_adv_dollar_lagged_float": 1_000_000.0,
                "central_implicit_cost_bps_float": 2.5,
                "model_extrapolation_bool": False,
                "academic_extrapolation_bool": False,
                "proxy_extrapolation_bool": False,
            }
        ]
    )
    return CapacityRunResult(
        strategy_name_str="toy_capacity",
        capital_base_float=capital_base_float,
        execution_policy_str=execution_policy_str,
        impact_profile_str=(
            MOO_LARGE_MIXED_PROFILE_STR if execution_policy_str == "MOO" else None
        ),
        order_diagnostics_df=order_diagnostics_df,
        summary_dict=summary_dict,
        strategy_obj=strategy_obj,
        pricing_data_df=pricing_data_df,
        equity_curve_df=pd.DataFrame(
            {
                "baseline_equity_float": [1.0, 1.1],
                "central_equity_float": [1.0, 1.08],
                "stress_equity_float": [1.0, 1.06],
            },
            index=pricing_data_df.index[:2],
        ),
    )


def test_moc_study_classifies_capacity_and_break_even(tmp_path):
    run_result_list = [
        _manual_run_result(100_000.0, "MOC", 0.08, True),
        _manual_run_result(1_000_000.0, "MOC", 0.03, True),
        _manual_run_result(10_000_000.0, "MOC", -0.01, False),
    ]
    study_result_obj = build_capacity_study_result(
        {RECENT_FIVE_YEAR_WINDOW_STR: run_result_list},
        output_dir_str=str(tmp_path),
        save_output_bool=True,
    )

    assert study_result_obj.summary_dict["recommended_capacity_float"] == 1_000_000.0
    assert study_result_obj.summary_dict["outer_capacity_float"] == 1_000_000.0
    assert study_result_obj.summary_dict["break_even_capacity_bracket_str"] == "$1,000,000 to $10,000,000"
    output_dir_path = study_result_obj.output_dir_path
    assert output_dir_path is not None
    for filename_str in [
        CAPACITY_CURVE_CSV_FILENAME_STR,
        CAPACITY_ORDER_CSV_FILENAME_STR,
        SUMMARY_FILENAME_STR,
        METADATA_FILENAME_STR,
        REPORT_FILENAME_STR,
    ]:
        assert (output_dir_path / filename_str).is_file()
    assert sorted(output_path.name for output_path in output_dir_path.iterdir()) == sorted(
        [
            CAPACITY_CURVE_CSV_FILENAME_STR,
            CAPACITY_ORDER_CSV_FILENAME_STR,
            SUMMARY_FILENAME_STR,
            METADATA_FILENAME_STR,
            REPORT_FILENAME_STR,
        ]
    )
    report_html_str = (output_dir_path / REPORT_FILENAME_STR).read_text(encoding="utf-8")
    assert "Read this first" in report_html_str
    assert report_html_str.count("<svg") == 4
    assert "Worked MOC example" in report_html_str
    assert "chart.js" not in report_html_str.lower()
    assert json.loads((output_dir_path / SUMMARY_FILENAME_STR).read_text())["execution_policy_str"] == "MOC"


def test_dual_window_outputs_use_recent_headline_and_preserve_full_history(tmp_path):
    full_run_result_obj = _manual_run_result(100_000.0, "MOO", 0.08, False)
    recent_run_result_obj = _manual_run_result(100_000.0, "MOO", 0.08, True)
    full_run_result_obj.summary_dict["actual_start_date_str"] = "2010-01-04"
    full_run_result_obj.summary_dict["actual_end_date_str"] = "2025-06-30"
    recent_run_result_obj.summary_dict["actual_start_date_str"] = "2020-06-30"
    recent_run_result_obj.summary_dict["actual_end_date_str"] = "2025-06-30"

    study_result_obj = build_capacity_study_result(
        {
            FULL_HISTORY_WINDOW_STR: [full_run_result_obj],
            RECENT_FIVE_YEAR_WINDOW_STR: [recent_run_result_obj],
        },
        output_dir_str=str(tmp_path),
        save_output_bool=True,
    )

    assert study_result_obj.summary_dict["headline_window_str"] == RECENT_FIVE_YEAR_WINDOW_STR
    assert study_result_obj.summary_dict["recommended_capacity_float"] == 100_000.0
    assert (
        study_result_obj.summary_dict["window_summary_dict"][FULL_HISTORY_WINDOW_STR][
            "recommended_capacity_float"
        ]
        is None
    )
    assert study_result_obj.summary_dict["historical_feasibility_warning_bool"] is True
    assert set(study_result_obj.capacity_curve_df["window_str"]) == {
        FULL_HISTORY_WINDOW_STR,
        RECENT_FIVE_YEAR_WINDOW_STR,
    }
    assert set(study_result_obj.order_diagnostics_df["window_str"]) == {
        FULL_HISTORY_WINDOW_STR,
        RECENT_FIVE_YEAR_WINDOW_STR,
    }
    metadata_dict = json.loads(
        (study_result_obj.output_dir_path / METADATA_FILENAME_STR).read_text()
    )
    assert metadata_dict["model_version_str"] == CAPACITY_MODEL_VERSION_STR
    assert metadata_dict["window_date_dict"][RECENT_FIVE_YEAR_WINDOW_STR] == {
        "actual_start_date_str": "2020-06-30",
        "actual_end_date_str": "2025-06-30",
    }
    assert metadata_dict["full_history_actual_start_date_str"] == "2010-01-04"
    assert metadata_dict["recent_5y_actual_start_date_str"] == "2020-06-30"
    assert metadata_dict["common_actual_end_date_str"] == "2025-06-30"
    report_html_str = (study_result_obj.output_dir_path / REPORT_FILENAME_STR).read_text(
        encoding="utf-8"
    )
    assert "Historical feasibility warning" in report_html_str
    assert "Full-history feasibility" in report_html_str
    assert "active alpha" not in report_html_str.lower()
    assert "active return" not in report_html_str.lower()


def test_classification_stops_at_first_failure_and_warns_on_later_pass(tmp_path):
    low_run_result_obj = _manual_run_result(100_000.0, "MOO", 0.08, True)
    middle_run_result_obj = _manual_run_result(1_000_000.0, "MOO", 0.07, False)
    high_run_result_obj = _manual_run_result(10_000_000.0, "MOO", 0.06, True)

    study_result_obj = build_capacity_study_result(
        {
            RECENT_FIVE_YEAR_WINDOW_STR: [
                low_run_result_obj,
                middle_run_result_obj,
                high_run_result_obj,
            ]
        },
        output_dir_str=str(tmp_path),
        save_output_bool=True,
    )

    assert study_result_obj.summary_dict["recommended_capacity_float"] == 100_000.0
    assert study_result_obj.summary_dict["recommended_non_contiguous_pass_bool"] is True
    assert study_result_obj.capacity_curve_df["recommended_pass_bool"].tolist() == [
        True,
        False,
        False,
    ]
    assert study_result_obj.capacity_curve_df["recommended_raw_pass_bool"].tolist() == [
        True,
        False,
        True,
    ]
    assert study_result_obj.summary_dict["outer_capacity_float"] == 100_000.0
    assert study_result_obj.summary_dict["outer_non_contiguous_pass_bool"] is True
    assert study_result_obj.capacity_curve_df["outer_pass_bool"].tolist() == [
        True,
        False,
        False,
    ]
    report_html_str = (study_result_obj.output_dir_path / REPORT_FILENAME_STR).read_text(
        encoding="utf-8"
    )
    assert "Recommended Max and Outer Capacity" in report_html_str


def test_top_grid_capacity_is_right_censored(tmp_path):
    study_result_obj = build_capacity_study_result(
        {
            RECENT_FIVE_YEAR_WINDOW_STR: [
                _manual_run_result(100_000.0, "MOO", 0.08, True),
                _manual_run_result(1_000_000.0, "MOO", 0.07, True),
            ]
        },
        output_dir_str=str(tmp_path),
        save_output_bool=True,
    )

    assert study_result_obj.summary_dict["recommended_capacity_float"] == 1_000_000.0
    assert study_result_obj.summary_dict["recommended_capacity_censored_bool"] is True
    assert study_result_obj.summary_dict["outer_capacity_censored_bool"] is True
    report_html_str = (study_result_obj.output_dir_path / REPORT_FILENAME_STR).read_text(
        encoding="utf-8"
    )
    assert "≥ $1,000,000" in report_html_str


def test_equal_recent_and_full_capacity_has_no_historical_warning(tmp_path):
    full_run_result_obj = _manual_run_result(100_000.0, "MOO", 0.08, True)
    recent_run_result_obj = _manual_run_result(100_000.0, "MOO", 0.08, True)
    full_run_result_obj.summary_dict["actual_start_date_str"] = "2010-01-04"
    full_run_result_obj.summary_dict["actual_end_date_str"] = "2025-06-30"
    recent_run_result_obj.summary_dict["actual_start_date_str"] = "2020-06-30"
    recent_run_result_obj.summary_dict["actual_end_date_str"] = "2025-06-30"

    study_result_obj = build_capacity_study_result(
        {
            FULL_HISTORY_WINDOW_STR: [full_run_result_obj],
            RECENT_FIVE_YEAR_WINDOW_STR: [recent_run_result_obj],
        },
        output_dir_str=str(tmp_path),
        save_output_bool=True,
    )

    assert study_result_obj.summary_dict["historical_feasibility_warning_bool"] is False
    report_html_str = (study_result_obj.output_dir_path / REPORT_FILENAME_STR).read_text(
        encoding="utf-8"
    )
    assert "Historical feasibility warning" not in report_html_str


def test_recent_numeric_capacity_remains_headline_when_full_history_is_lower():
    full_run_result_obj = _manual_run_result(100_000.0, "MOO", 0.08, True)
    recent_low_run_result_obj = _manual_run_result(100_000.0, "MOO", 0.08, True)
    recent_high_run_result_obj = _manual_run_result(1_000_000.0, "MOO", 0.07, True)
    full_run_result_obj.summary_dict["actual_start_date_str"] = "2010-01-04"
    full_run_result_obj.summary_dict["actual_end_date_str"] = "2025-06-30"
    for recent_run_result_obj in [recent_low_run_result_obj, recent_high_run_result_obj]:
        recent_run_result_obj.summary_dict["actual_start_date_str"] = "2020-06-30"
        recent_run_result_obj.summary_dict["actual_end_date_str"] = "2025-06-30"

    study_result_obj = build_capacity_study_result(
        {
            FULL_HISTORY_WINDOW_STR: [full_run_result_obj],
            RECENT_FIVE_YEAR_WINDOW_STR: [
                recent_low_run_result_obj,
                recent_high_run_result_obj,
            ],
        },
        save_output_bool=False,
    )

    assert study_result_obj.summary_dict["recommended_capacity_float"] == 1_000_000.0
    assert (
        study_result_obj.summary_dict["window_summary_dict"][FULL_HISTORY_WINDOW_STR][
            "recommended_capacity_float"
        ]
        == 100_000.0
    )
    assert study_result_obj.summary_dict["historical_feasibility_warning_bool"] is True


def test_rolling_sharpe_floor_includes_exact_threshold_and_withholds_when_empty():
    baseline_sharpe_ser = pd.Series([0.29, 0.30, 0.31])
    central_sharpe_ser = pd.Series([0.10, 0.24, 0.248])

    erosion_float, eligible_count_int = _eligible_rolling_sharpe_erosion_tuple(
        baseline_sharpe_ser,
        central_sharpe_ser,
    )
    assert eligible_count_int == 2
    assert erosion_float == pytest.approx(0.20)

    empty_erosion_float, empty_count_int = _eligible_rolling_sharpe_erosion_tuple(
        pd.Series([0.10, 0.29]),
        pd.Series([0.08, 0.25]),
    )
    assert np.isnan(empty_erosion_float)
    assert empty_count_int == 0

    run_result_obj = _manual_run_result(100_000.0, "MOO", 0.08, True)
    run_result_obj.summary_dict["rolling_3y_available_bool"] = False
    run_result_obj.summary_dict["rolling_3y_eligible_window_count_int"] = 0
    run_result_obj.summary_dict["worst_eligible_rolling_3y_sharpe_erosion_float"] = np.nan
    study_result_obj = build_capacity_study_result(
        {RECENT_FIVE_YEAR_WINDOW_STR: [run_result_obj]},
        save_output_bool=False,
    )
    assert study_result_obj.summary_dict["recommended_capacity_float"] is None


def test_optimal_capacity_uses_supported_grid_not_all_recommended_gates():
    low_run_result_obj = _manual_run_result(100_000.0, "MOO", 0.08, True)
    high_run_result_obj = _manual_run_result(1_000_000.0, "MOO", 0.07, True)
    high_run_result_obj.summary_dict["central_sharpe_float"] = 1.1
    high_run_result_obj.summary_dict[
        "central_cost_consumption_of_benchmark_excess_float"
    ] = 0.30
    study_result_obj = build_capacity_study_result(
        {RECENT_FIVE_YEAR_WINDOW_STR: [low_run_result_obj, high_run_result_obj]},
        save_output_bool=False,
    )
    assert study_result_obj.summary_dict["optimal_capacity_float"] == 1_000_000.0
    assert study_result_obj.summary_dict["recommended_capacity_float"] == 100_000.0


def test_moo_study_uses_impact_profile_and_reports_equity(tmp_path):
    run_result_list = [
        _manual_run_result(100_000.0, "MOO", 0.08, True),
        _manual_run_result(1_000_000.0, "MOO", 0.07, False),
    ]
    study_result_obj = build_capacity_study_result(
        {RECENT_FIVE_YEAR_WINDOW_STR: run_result_list},
        output_dir_str=str(tmp_path),
        save_output_bool=True,
    )

    assert study_result_obj.summary_dict["recommended_capacity_float"] == 100_000.0
    assert study_result_obj.summary_dict["optimal_capacity_float"] == 100_000.0
    assert study_result_obj.summary_dict["outer_capacity_float"] == 100_000.0
    assert study_result_obj.summary_dict["break_even_capacity_bracket_str"] == "Above $1,000,000"
    report_html_str = (study_result_obj.output_dir_path / REPORT_FILENAME_STR).read_text(
        encoding="utf-8"
    )
    assert "Worked MOO example" in report_html_str
    assert "MOO_LARGE_MIXED" in report_html_str
    assert "Central lambda" in report_html_str
    assert report_html_str.count("<svg") == 4


def test_etf_profile_report_discloses_low_confidence_proxy(tmp_path):
    run_result_obj = _manual_run_result(100_000.0, "MOO", 0.08, True)
    run_result_obj.impact_profile_str = "MOO_ETF_PROXY"
    run_result_obj.summary_dict["impact_profile_str"] = "MOO_ETF_PROXY"
    run_result_obj.summary_dict["model_confidence_str"] = "low"
    run_result_obj.summary_dict["proxy_bool"] = True
    study_result_obj = build_capacity_study_result(
        {RECENT_FIVE_YEAR_WINDOW_STR: [run_result_obj]},
        output_dir_str=str(tmp_path),
        save_output_bool=True,
    )
    report_html_str = (study_result_obj.output_dir_path / REPORT_FILENAME_STR).read_text(
        encoding="utf-8"
    )
    assert "MOO_ETF_PROXY" in report_html_str
    assert "common-stock auction estimates as a sensitivity proxy" in report_html_str
    assert "not ETF-specific empirical calibration" in report_html_str


def test_common_stock_moo_extrapolation_is_flagged_and_warned(tmp_path):
    pricing_data_df = _pricing_data_df(turnover_float=1_000_000.0)
    strategy_obj = _strategy_obj(pricing_data_df, order_notional_float=20_000.0)
    run_result_obj = CapacityAnalysis(
        strategy_obj,
        pricing_data_df,
        "MOO",
        MOO_LARGE_MIXED_PROFILE_STR,
    ).run()
    assert run_result_obj.order_diagnostics_df["academic_extrapolation_bool"].all()
    study_result_obj = build_capacity_study_result(
        {RECENT_FIVE_YEAR_WINDOW_STR: [run_result_obj]},
        output_dir_str=str(tmp_path),
        save_output_bool=True,
    )
    report_html_str = (study_result_obj.output_dir_path / REPORT_FILENAME_STR).read_text(
        encoding="utf-8"
    )
    assert "Academic extrapolation warning" in report_html_str
    assert "diagnostic only" in report_html_str
    assert "Diagnostic only - no AUM point met every Recommended Max rule." in report_html_str


def test_etf_moo_extrapolation_uses_low_confidence_proxy_flag(tmp_path):
    pricing_data_df = _pricing_data_df(turnover_float=1_000_000.0)
    strategy_obj = _strategy_obj(pricing_data_df, order_notional_float=20_000.0)
    run_result_obj = CapacityAnalysis(
        strategy_obj,
        pricing_data_df,
        "MOO",
        "MOO_ETF_PROXY",
    ).run()

    assert run_result_obj.order_diagnostics_df["model_extrapolation_bool"].all()
    assert run_result_obj.order_diagnostics_df["proxy_extrapolation_bool"].all()
    assert not run_result_obj.order_diagnostics_df["academic_extrapolation_bool"].any()
    assert run_result_obj.summary_dict["proxy_extrapolation_share_float"] == 1.0


def test_break_even_requires_adjacent_finite_sign_crossing():
    negative_to_positive_df = pd.DataFrame(
        {
            "capital_base_float": [100_000.0, 1_000_000.0],
            "central_benchmark_excess_return_float": [-0.01, 0.02],
        }
    )
    assert _break_even_bracket_str(negative_to_positive_df) == "$100,000 to $1,000,000"

    missing_middle_df = pd.DataFrame(
        {
            "capital_base_float": [100_000.0, 1_000_000.0, 10_000_000.0],
            "central_benchmark_excess_return_float": [0.02, np.nan, -0.01],
        }
    )
    assert _break_even_bracket_str(missing_middle_df) == (
        "Not estimable from adjacent finite grid points"
    )
    exact_zero_df = pd.DataFrame(
        {
            "capital_base_float": [100_000.0, 1_000_000.0],
            "central_benchmark_excess_return_float": [0.02, 0.0],
        }
    )
    assert _break_even_bracket_str(exact_zero_df) == "$1,000,000"


def test_missing_liquidity_fails_capacity_classifications():
    run_result_obj = _manual_run_result(100_000.0, "MOO", 0.08, True)
    run_result_obj.summary_dict["liquidity_complete_bool"] = False
    run_result_obj.summary_dict["unavailable_order_share_float"] = 0.5

    study_result_obj = build_capacity_study_result(
        {RECENT_FIVE_YEAR_WINDOW_STR: [run_result_obj]},
        save_output_bool=False,
    )

    assert study_result_obj.summary_dict["recommended_capacity_float"] is None
    assert study_result_obj.summary_dict["outer_capacity_float"] is None


def test_undeclared_performance_benchmark_makes_recommended_unavailable():
    pricing_data_df = _pricing_data_df()
    strategy_obj = _strategy_obj(pricing_data_df)
    strategy_obj._performance_benchmark_adjustment_str = "not_declared"

    run_result_obj = CapacityAnalysis(
        strategy_obj,
        pricing_data_df,
        "MOO",
        MOO_LARGE_MIXED_PROFILE_STR,
    ).run()
    study_result_obj = build_capacity_study_result(
        {RECENT_FIVE_YEAR_WINDOW_STR: [run_result_obj]},
        save_output_bool=False,
    )

    assert np.isnan(run_result_obj.summary_dict["benchmark_annual_return_float"])
    assert study_result_obj.summary_dict["recommended_capacity_float"] is None


def test_capacity_benchmark_uses_declared_data_symbol_mapping():
    pricing_data_df = _pricing_data_df(bar_count_int=252)
    pricing_data_df[("$SPX_TR", "Close")] = np.linspace(
        100.0,
        130.0,
        len(pricing_data_df),
    )
    strategy_obj = _strategy_obj(pricing_data_df)
    strategy_obj._benchmark_data_symbol_map_dict = {"$SPX": "$SPX_TR"}

    annual_return_float, benchmark_label_str = _benchmark_annual_return_tuple(
        strategy_obj,
        pricing_data_df,
        pricing_data_df.index,
    )

    assert benchmark_label_str == "$SPX"
    expected_annual_return_float = (1.30 ** (252.0 / 251.0)) - 1.0
    assert annual_return_float == pytest.approx(expected_annual_return_float)


def test_moo_outer_is_unavailable_when_benchmark_is_unavailable():
    run_result_obj = _manual_run_result(100_000.0, "MOO", 0.08, True)
    run_result_obj.summary_dict["benchmark_annual_return_float"] = np.nan
    run_result_obj.summary_dict["central_benchmark_excess_return_float"] = np.nan
    run_result_obj.summary_dict["stress_benchmark_excess_return_float"] = np.nan

    study_result_obj = build_capacity_study_result(
        {RECENT_FIVE_YEAR_WINDOW_STR: [run_result_obj]},
        save_output_bool=False,
    )

    assert study_result_obj.summary_dict["recommended_capacity_float"] is None
    assert study_result_obj.summary_dict["outer_capacity_float"] is None
