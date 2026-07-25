import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from alpha.engine.risk_analysis import (
    RISK_ANALYSIS_TYPE_STR,
    VERDICT_STATUS_AMBER_STR,
    VERDICT_STATUS_GREEN_STR,
    VERDICT_STATUS_NA_STR,
    VERDICT_STATUS_RED_STR,
    RiskAnalysis,
    RiskAnalysisResult,
    _observed_percentile_float,
    _format_metric_value,
    _path_horizon_metric_dict,
    _worst_rolling_return_float,
    _band_status_str,
    _build_verdict_row_list,
    build_bootstrap_equity_path_df,
    build_bootstrap_interval_df,
    build_bootstrap_path_metric_df,
    build_horizon_probability_df,
    build_investor_scenario_df,
    build_observed_calendar_month_df,
    compute_path_metric_dict,
    extract_realized_return_ser,
    build_return_histogram_df,
    stationary_bootstrap_index_mat,
)


def _toy_strategy_obj():
    calendar_idx = pd.date_range("2020-01-01", periods=7, freq="B")
    daily_return_ser = pd.Series(
        [0.0, 0.01, -0.02, 0.03, -0.01, 0.0, 0.02],
        index=calendar_idx,
        name="daily_returns",
    )
    total_value_ser = 100.0 * (1.0 + daily_return_ser).cumprod()
    result_df = pd.DataFrame(
        {
            "daily_returns": daily_return_ser,
            "total_value": total_value_ser,
        },
        index=calendar_idx,
    )
    return SimpleNamespace(name="toy_strategy", results=result_df)


def test_stationary_bootstrap_index_matrix_is_deterministic_and_in_range():
    first_index_mat = stationary_bootstrap_index_mat(
        sample_size_int=6,
        simulation_count_int=4,
        mean_block_length_int=3,
        random_seed_int=123,
    )
    second_index_mat = stationary_bootstrap_index_mat(
        sample_size_int=6,
        simulation_count_int=4,
        mean_block_length_int=3,
        random_seed_int=123,
    )

    assert first_index_mat.shape == (4, 6)
    assert np.array_equal(first_index_mat, second_index_mat)
    assert int(first_index_mat.min()) >= 0
    assert int(first_index_mat.max()) < 6


def test_stationary_bootstrap_index_matrix_accepts_shorter_path_length():
    index_mat = stationary_bootstrap_index_mat(
        sample_size_int=6,
        simulation_count_int=4,
        mean_block_length_int=3,
        random_seed_int=123,
        path_length_int=2,
    )

    assert index_mat.shape == (4, 2)
    assert int(index_mat.min()) >= 0
    assert int(index_mat.max()) < 6


def test_compute_path_metrics_match_synthetic_formula():
    return_vec = np.array([0.10, -0.20, 0.05], dtype=float)

    metric_dict = compute_path_metric_dict(return_vec, rolling_loss_window_tuple=(1, 2))

    terminal_return_float = (1.10 * 0.80 * 1.05) - 1.0
    assert np.isclose(metric_dict["terminal_return_float"], terminal_return_float)
    assert np.isclose(metric_dict["max_drawdown_float"], -0.20)
    assert np.isclose(metric_dict["var_95_daily_return_float"], -0.175)
    assert np.isclose(metric_dict["cvar_95_daily_return_float"], -0.20)
    assert np.isclose(metric_dict["worst_1d_return_float"], -0.20)
    assert np.isclose(metric_dict["worst_2d_return_float"], -0.16)


def test_compute_path_metrics_include_monthly_and_time_underwater():
    rng_obj = np.random.default_rng(0)
    daily_return_vec = rng_obj.normal(loc=0.0005, scale=0.012, size=300)

    metric_dict = compute_path_metric_dict(
        daily_return_vec,
        rolling_loss_window_tuple=(1, 21, 63, 126, 252),
    )

    monthly_key_list = [
        "monthly_expected_return_float",
        "monthly_volatility_float",
        "monthly_sharpe_float",
        "monthly_var_95_return_float",
        "monthly_cvar_95_return_float",
        "monthly_var_99_return_float",
        "monthly_cvar_99_return_float",
    ]
    for key_str in monthly_key_list:
        assert key_str in metric_dict, f"missing monthly metric {key_str}"

    assert "longest_underwater_days_float" in metric_dict
    assert metric_dict["longest_underwater_days_float"] >= 0.0
    assert metric_dict["longest_underwater_days_float"] <= float(daily_return_vec.size)

    var_95_float = float(metric_dict["monthly_var_95_return_float"])
    cvar_95_float = float(metric_dict["monthly_cvar_95_return_float"])
    assert cvar_95_float <= var_95_float, "CVaR should be <= VaR (further into left tail)"

    var_99_float = float(metric_dict["monthly_var_99_return_float"])
    cvar_99_float = float(metric_dict["monthly_cvar_99_return_float"])
    assert cvar_99_float <= var_99_float
    assert var_99_float <= var_95_float, "99% VaR should be at least as bad as 95% VaR"

    assert "worst_126d_return_float" in metric_dict
    assert "worst_252d_return_float" in metric_dict
    # 300 days < 252 + buffer? 300 >= 252 so worst_252d should be finite.
    assert np.isfinite(float(metric_dict["worst_126d_return_float"]))
    assert np.isfinite(float(metric_dict["worst_252d_return_float"]))


def test_observed_calendar_month_returns_compound_real_months():
    calendar_idx = pd.to_datetime(
        ["2020-01-02", "2020-01-03", "2020-02-03", "2020-02-04"]
    )
    realized_return_ser = pd.Series(
        [0.10, -0.05, -0.10, 0.05],
        index=calendar_idx,
        name="realized_return_float",
    )

    calendar_month_df = build_observed_calendar_month_df(realized_return_ser)

    assert calendar_month_df["calendar_month_str"].tolist() == ["2020-01", "2020-02"]
    assert calendar_month_df["trading_day_count_int"].tolist() == [2, 2]
    assert calendar_month_df["calendar_month_end_str"].tolist() == [
        "2020-01-03",
        "2020-02-04",
    ]
    assert calendar_month_df["scheduled_calendar_month_end_str"].tolist() == [
        "2020-01-31",
        "2020-02-29",
    ]
    assert calendar_month_df["sample_boundary_month_bool"].tolist() == [True, True]
    assert np.isclose(float(calendar_month_df.iloc[0]["calendar_month_return_float"]), 1.10 * 0.95 - 1.0)
    assert np.isclose(float(calendar_month_df.iloc[1]["calendar_month_return_float"]), 0.90 * 1.05 - 1.0)


def test_path_horizon_metrics_measure_recovery_and_censoring():
    recovered_metric_dict = _path_horizon_metric_dict(
        np.array([0.10, -0.20, 0.25], dtype=float),
        horizon_day_int=3,
    )
    assert np.isclose(float(recovered_metric_dict["max_drawdown_float"]), -0.20)
    assert recovered_metric_dict["max_drawdown_unrecovered_bool"] is False
    assert np.isclose(float(recovered_metric_dict["max_drawdown_recovery_days_float"]), 2.0)
    assert np.isclose(float(recovered_metric_dict["longest_underwater_days_float"]), 1.0)

    unrecovered_metric_dict = _path_horizon_metric_dict(
        np.array([0.10, -0.20, 0.00], dtype=float),
        horizon_day_int=3,
    )
    assert unrecovered_metric_dict["max_drawdown_unrecovered_bool"] is True
    assert unrecovered_metric_dict["max_drawdown_recovery_days_float"] is None
    assert np.isclose(float(unrecovered_metric_dict["longest_underwater_days_float"]), 2.0)

    repeated_deepest_metric_dict = _path_horizon_metric_dict(
        np.array([0.10, -0.20, 0.25, -0.20], dtype=float),
        horizon_day_int=4,
    )
    assert repeated_deepest_metric_dict["max_drawdown_unrecovered_bool"] is True
    assert repeated_deepest_metric_dict["max_drawdown_recovery_days_float"] is None

    recovered_then_underwater_metric_dict = _path_horizon_metric_dict(
        np.array([0.10, -0.20, 0.25, -0.05], dtype=float),
        horizon_day_int=4,
    )
    assert recovered_then_underwater_metric_dict["max_drawdown_unrecovered_bool"] is False
    assert recovered_then_underwater_metric_dict["terminal_underwater_bool"] is True


def test_worst_rolling_return_matches_direct_window_products():
    return_vec = np.array(
        [0.02, -0.01, 0.03, -0.04, 0.01, 0.005, -0.02],
        dtype=float,
    )
    for window_int in (1, 2, 3, 5, 7):
        expected_float = min(
            float(np.prod(1.0 + return_vec[start_int : start_int + window_int]) - 1.0)
            for start_int in range(return_vec.size - window_int + 1)
        )
        actual_float = _worst_rolling_return_float(return_vec, window_int)
        assert np.isclose(actual_float, expected_float, rtol=1e-12, atol=1e-12)


def test_observed_percentile_treats_floating_noise_as_a_tie():
    bootstrap_metric_ser = pd.Series([0.10 - 1e-15, 0.10 + 1e-15, 0.20])

    percentile_float = _observed_percentile_float(bootstrap_metric_ser, 0.10)

    assert np.isclose(float(percentile_float), 2.0 / 3.0)


def test_monthly_sharpe_formats_as_ratio_not_percentage():
    assert _format_metric_value(1.7906, "monthly_sharpe_float") == "1.79"


def _verdict_status_by_label(verdict_row_list):
    return {row_dict["label_str"]: row_dict["status_str"] for row_dict in verdict_row_list}


def test_band_status_helper_handles_none_and_nan():
    assert _band_status_str(None, 0.1, 0.3) == VERDICT_STATUS_NA_STR
    assert _band_status_str(float("nan"), 0.1, 0.3) == VERDICT_STATUS_NA_STR
    assert _band_status_str(0.05, 0.1, 0.3) == VERDICT_STATUS_GREEN_STR
    assert _band_status_str(0.20, 0.1, 0.3) == VERDICT_STATUS_AMBER_STR
    assert _band_status_str(0.50, 0.1, 0.3) == VERDICT_STATUS_RED_STR


def test_verdict_bands_classify_expected():
    green_summary_dict = {
        "primary_terminal_loss_probability_float": 0.04,
        "primary_time_underwater_breach_probabilities": {"underwater_ge_12m": 0.10},
        "primary_intervals": {
            "sharpe_float": {"ci_lower_float": 0.5},
            "max_drawdown_float": {"p05_float": -0.15},
            "worst_252d_return_float": {"p05_float": -0.10},
        },
    }
    green_status_dict = _verdict_status_by_label(_build_verdict_row_list(green_summary_dict))
    assert green_status_dict["Historical resampling"] == VERDICT_STATUS_GREEN_STR
    assert green_status_dict["Drawdown depth"] == VERDICT_STATUS_GREEN_STR
    assert green_status_dict["Time underwater"] == VERDICT_STATUS_GREEN_STR
    assert green_status_dict["Worst year"] == VERDICT_STATUS_GREEN_STR

    # Low terminal-loss but Sharpe CI includes zero -> edge capped at amber.
    capped_summary_dict = dict(green_summary_dict)
    capped_summary_dict["primary_intervals"] = dict(green_summary_dict["primary_intervals"])
    capped_summary_dict["primary_intervals"]["sharpe_float"] = {"ci_lower_float": -0.1}
    capped_status_dict = _verdict_status_by_label(_build_verdict_row_list(capped_summary_dict))
    assert capped_status_dict["Historical resampling"] == VERDICT_STATUS_AMBER_STR

    red_summary_dict = {
        "primary_terminal_loss_probability_float": 0.40,
        "primary_time_underwater_breach_probabilities": {"underwater_ge_12m": 0.70},
        "primary_intervals": {
            "sharpe_float": {"ci_lower_float": -0.3},
            "max_drawdown_float": {"p05_float": -0.45},
            "worst_252d_return_float": {"p05_float": -0.40},
        },
    }
    red_status_dict = _verdict_status_by_label(_build_verdict_row_list(red_summary_dict))
    assert red_status_dict["Historical resampling"] == VERDICT_STATUS_RED_STR
    assert red_status_dict["Drawdown depth"] == VERDICT_STATUS_RED_STR
    assert red_status_dict["Time underwater"] == VERDICT_STATUS_RED_STR
    assert red_status_dict["Worst year"] == VERDICT_STATUS_RED_STR

    # Missing worst-12m input -> Worst year renders as N/A band.
    na_summary_dict = {
        "primary_terminal_loss_probability_float": 0.04,
        "primary_time_underwater_breach_probabilities": {"underwater_ge_12m": 0.10},
        "primary_intervals": {
            "sharpe_float": {"ci_lower_float": 0.5},
            "max_drawdown_float": {"p05_float": -0.15},
        },
    }
    na_status_dict = _verdict_status_by_label(_build_verdict_row_list(na_summary_dict))
    assert na_status_dict["Worst year"] == VERDICT_STATUS_NA_STR


def test_return_histogram_counts_realized_returns():
    strategy_obj = _toy_strategy_obj()
    realized_return_ser = extract_realized_return_ser(strategy_obj)

    histogram_df = build_return_histogram_df(realized_return_ser, bin_count_int=4)

    assert int(histogram_df["count_int"].sum()) == len(realized_return_ser)
    assert np.isclose(histogram_df["probability_float"].sum(), 1.0)


def test_bootstrap_equity_paths_include_observed_and_percentiles():
    strategy_obj = _toy_strategy_obj()
    realized_return_ser = extract_realized_return_ser(strategy_obj)

    equity_path_df = build_bootstrap_equity_path_df(
        realized_return_ser=realized_return_ser,
        mean_block_length_int=2,
        simulation_count_int=5,
        random_seed_int=17,
        path_sample_count_int=3,
    )

    assert {"bootstrap", "observed", "p05", "p50", "p95"} == set(equity_path_df["path_kind_str"])
    observed_df = equity_path_df[equity_path_df["path_kind_str"] == "observed"]
    assert len(observed_df) == len(realized_return_ser) + 1
    assert np.isclose(observed_df.iloc[0]["equity_float"], 1.0)


def test_bootstrap_intervals_include_confidence_bounds():
    strategy_obj = _toy_strategy_obj()
    realized_return_ser = extract_realized_return_ser(strategy_obj)
    bootstrap_metric_df = build_bootstrap_path_metric_df(
        realized_return_ser=realized_return_ser,
        mean_block_length_tuple=(2,),
        simulation_count_int=10,
        random_seed_int=7,
        rolling_loss_window_tuple=(1, 5),
    )
    observed_metric_dict = compute_path_metric_dict(
        realized_return_ser.to_numpy(dtype=float),
        rolling_loss_window_tuple=(1, 5),
    )

    interval_df = build_bootstrap_interval_df(
        bootstrap_path_metric_df=bootstrap_metric_df,
        observed_metric_dict=observed_metric_dict,
        confidence_level_float=0.95,
    )

    terminal_row = interval_df[interval_df["metric_name_str"] == "terminal_return_float"].iloc[0]
    assert int(terminal_row["mean_block_length_int"]) == 2
    assert terminal_row["ci_lower_float"] <= terminal_row["ci_upper_float"]
    assert pd.notna(terminal_row["observed_value_float"])
    assert pd.notna(terminal_row["bootstrap_mean_float"])


def test_horizon_probability_table_counts_positive_constant_sample():
    calendar_idx = pd.date_range("2020-01-01", periods=252, freq="B")
    realized_return_ser = pd.Series(
        [0.01] * 252,
        index=calendar_idx,
        name="realized_return_float",
    )

    horizon_df = build_horizon_probability_df(
        realized_return_ser=realized_return_ser,
        mean_block_length_int=21,
        simulation_count_int=3,
        random_seed_int=7,
        horizon_year_tuple=(1, 2),
        drawdown_threshold_tuple=(-0.10,),
        upside_threshold_tuple=(0.10, 0.50),
    )

    one_year_ser = horizon_df[horizon_df["horizon_year_int"] == 1].iloc[0]
    assert int(one_year_ser["simulation_path_count_int"]) == 3
    assert np.isclose(float(one_year_ser["drawdown_lte_10pct_probability_float"]), 0.0)
    assert np.isclose(float(one_year_ser["gain_gte_10pct_probability_float"]), 1.0)
    assert np.isclose(float(one_year_ser["gain_gte_50pct_probability_float"]), 1.0)
    assert float(one_year_ser["max_gain_p50_float"]) > 0.50
    assert float(one_year_ser["terminal_return_p05_float"]) > 0.0
    assert np.isclose(float(one_year_ser["terminal_loss_probability_float"]), 0.0)
    assert np.isclose(float(one_year_ser["underwater_ge_12m_probability_float"]), 0.0)
    assert np.isclose(float(one_year_ser["max_drawdown_unrecovered_probability_float"]), 0.0)
    assert np.isclose(float(one_year_ser["terminal_underwater_probability_float"]), 0.0)

    two_year_ser = horizon_df[horizon_df["horizon_year_int"] == 2].iloc[0]
    assert int(two_year_ser["simulation_path_count_int"]) == 0
    assert pd.isna(two_year_ser["gain_gte_10pct_probability_float"])


def test_horizon_probability_table_counts_negative_constant_sample():
    calendar_idx = pd.date_range("2020-01-01", periods=252, freq="B")
    realized_return_ser = pd.Series(
        [-0.01] * 252,
        index=calendar_idx,
        name="realized_return_float",
    )

    horizon_df = build_horizon_probability_df(
        realized_return_ser=realized_return_ser,
        mean_block_length_int=21,
        simulation_count_int=3,
        random_seed_int=7,
        horizon_year_tuple=(1,),
        drawdown_threshold_tuple=(-0.10, -0.50),
        upside_threshold_tuple=(0.10,),
    )

    one_year_ser = horizon_df.iloc[0]
    assert int(one_year_ser["simulation_path_count_int"]) == 3
    assert np.isclose(float(one_year_ser["drawdown_lte_10pct_probability_float"]), 1.0)
    assert np.isclose(float(one_year_ser["drawdown_lte_50pct_probability_float"]), 1.0)
    assert np.isclose(float(one_year_ser["gain_gte_10pct_probability_float"]), 0.0)
    assert float(one_year_ser["max_drawdown_p50_float"]) < -0.50
    assert float(one_year_ser["terminal_return_p95_float"]) < 0.0
    assert np.isclose(float(one_year_ser["terminal_loss_probability_float"]), 1.0)
    assert np.isclose(float(one_year_ser["underwater_ge_12m_probability_float"]), 1.0)
    assert np.isclose(float(one_year_ser["max_drawdown_unrecovered_probability_float"]), 1.0)
    assert np.isclose(float(one_year_ser["terminal_underwater_probability_float"]), 1.0)


def test_investor_scenarios_separate_observed_calendar_and_modeled_month():
    calendar_idx = pd.date_range("2020-01-01", periods=252 * 5, freq="B")
    realized_return_ser = pd.Series(
        [0.001] * (252 * 5),
        index=calendar_idx,
        name="realized_return_float",
    )
    horizon_df = build_horizon_probability_df(
        realized_return_ser=realized_return_ser,
        mean_block_length_int=21,
        simulation_count_int=4,
        random_seed_int=7,
        horizon_year_tuple=(1, 3, 5),
    )

    investor_scenario_df = build_investor_scenario_df(
        realized_return_ser=realized_return_ser,
        horizon_probability_df=horizon_df,
        mean_block_length_int=21,
        simulation_count_int=4,
        random_seed_int=7,
        investor_horizon_year_tuple=(1, 3, 5),
    )

    assert investor_scenario_df["scenario_key_str"].tolist() == [
        "observed_daily",
        "observed_calendar_month",
        "modeled_21d",
        "modeled_1y",
        "modeled_3y",
        "modeled_5y",
    ]
    observed_month_ser = investor_scenario_df[
        investor_scenario_df["scenario_key_str"] == "observed_calendar_month"
    ].iloc[0]
    modeled_month_ser = investor_scenario_df[
        investor_scenario_df["scenario_key_str"] == "modeled_21d"
    ].iloc[0]
    assert observed_month_ser["evidence_kind_str"] == "observed"
    assert modeled_month_ser["evidence_kind_str"] == "bootstrap_implied"
    assert pd.isna(observed_month_ser["horizon_day_int"])
    assert int(modeled_month_ser["horizon_day_int"]) == 21
    assert str(investor_scenario_df["horizon_day_int"].dtype) == "Int64"
    assert str(investor_scenario_df["simulation_path_count_int"].dtype) == "Int64"
    assert np.isclose(float(modeled_month_ser["terminal_loss_probability_float"]), 0.0)
    assert float(modeled_month_ser["terminal_return_p05_float"]) > 0.0


def test_horizon_probability_table_exercises_default_one_to_five_years():
    calendar_idx = pd.date_range("2020-01-01", periods=252 * 5, freq="B")
    realized_return_ser = pd.Series(
        [0.001] * (252 * 5),
        index=calendar_idx,
        name="realized_return_float",
    )

    horizon_df = build_horizon_probability_df(
        realized_return_ser=realized_return_ser,
        mean_block_length_int=21,
        simulation_count_int=2,
        random_seed_int=7,
    )

    assert horizon_df["horizon_year_int"].tolist() == [1, 2, 3, 4, 5]
    assert horizon_df["horizon_day_int"].tolist() == [252, 504, 756, 1008, 1260]
    assert horizon_df["simulation_path_count_int"].tolist() == [2, 2, 2, 2, 2]
    assert np.isclose(float(horizon_df.iloc[0]["drawdown_lte_10pct_probability_float"]), 0.0)
    assert np.isclose(float(horizon_df.iloc[0]["gain_gte_10pct_probability_float"]), 1.0)
    assert np.isclose(float(horizon_df.iloc[-1]["gain_gte_50pct_probability_float"]), 1.0)


def test_bootstrap_intervals_reject_invalid_confidence_level():
    bootstrap_metric_df = pd.DataFrame(
        {
            "mean_block_length_int": [2, 2],
            "terminal_return_float": [0.01, -0.02],
        }
    )

    with pytest.raises(ValueError, match="confidence_level_float"):
        build_bootstrap_interval_df(
            bootstrap_path_metric_df=bootstrap_metric_df,
            observed_metric_dict={"terminal_return_float": 0.0},
            confidence_level_float=1.5,
        )


def test_risk_analysis_saves_expected_artifacts(tmp_path):
    strategy_obj = _toy_strategy_obj()
    risk_result_obj = RiskAnalysis(
        strategy_obj,
        source_strategy_ref_str="strategies.toy_strategy",
        output_dir_str=str(tmp_path),
        save_output_bool=True,
        primary_mean_block_length_int=2,
        mean_block_length_tuple=(2, 3),
        simulation_count_int=8,
        random_seed_int=11,
        confidence_level_float=0.95,
        rolling_loss_window_tuple=(1, 5),
    ).run()

    assert risk_result_obj.output_dir_path is not None
    output_path = risk_result_obj.output_dir_path
    assert output_path.relative_to(tmp_path).parts[:4] == (
        "research",
        "strategy",
        "toy_strategy",
        RISK_ANALYSIS_TYPE_STR,
    )
    for filename_str in [
        "return_histogram.csv",
        "bootstrap_equity_paths.csv",
        "bootstrap_path_metrics.csv",
        "bootstrap_metric_intervals.csv",
        "horizon_probabilities.csv",
        "observed_calendar_months.csv",
        "investor_scenarios.csv",
        "investor_summary.json",
        "summary.json",
        "run_info.json",
        "metadata.json",
        "report.html",
    ]:
        assert (output_path / filename_str).exists()

    summary_dict = json.loads((output_path / "summary.json").read_text(encoding="utf-8"))
    assert summary_dict["primary_mean_block_length_int"] == 2
    assert summary_dict["simulation_count_int"] == 8
    assert summary_dict["confidence_level_float"] == 0.95
    assert summary_dict["drawdown_threshold_list"] == [-0.10, -0.20, -0.30, -0.40, -0.50]
    assert summary_dict["upside_threshold_list"] == [0.10, 0.20, 0.30, 0.40, 0.50]
    assert "stress_status_str" not in summary_dict
    assert "var_95_daily_return_float" in summary_dict["observed_metrics"]
    assert summary_dict["horizon_year_list"] == [1, 2, 3, 4, 5]
    assert summary_dict["schema_version_int"] == 2
    assert summary_dict["investor_horizon_year_list"] == [1, 3, 5]
    assert summary_dict["time_underwater_breach_month_list"] == [3, 6, 12, 24]
    assert len(summary_dict["primary_horizon_probabilities"]) == 5
    assert summary_dict["investor_summary"]["status_str"] == "historically_conditioned_not_forecast"
    first_horizon_dict = summary_dict["primary_horizon_probabilities"][0]
    assert first_horizon_dict["horizon_day_int"] == 252
    assert first_horizon_dict["simulation_path_count_int"] == 0
    assert isinstance(first_horizon_dict["simulation_path_count_int"], int)
    assert first_horizon_dict["drawdown_lte_10pct_probability_float"] is None
    assert first_horizon_dict["gain_gte_10pct_probability_float"] is None

    horizon_df = pd.read_csv(output_path / "horizon_probabilities.csv")
    assert horizon_df["horizon_year_int"].tolist() == [1, 2, 3, 4, 5]
    assert "drawdown_lte_10pct_probability_float" in horizon_df.columns
    assert "gain_gte_10pct_probability_float" in horizon_df.columns
    assert "max_drawdown_p05_float" in horizon_df.columns
    assert "max_gain_p95_float" in horizon_df.columns
    assert "terminal_return_p05_float" in horizon_df.columns
    assert "terminal_loss_probability_float" in horizon_df.columns
    assert "max_drawdown_unrecovered_probability_float" in horizon_df.columns

    investor_summary_dict = json.loads(
        (output_path / "investor_summary.json").read_text(encoding="utf-8")
    )
    assert investor_summary_dict["month_definition_dict"] == {
        "modeled_month_str": "21_trading_days",
        "observed_month_str": "calendar_month",
    }
    assert investor_summary_dict["analysis_context_dict"] == {}
    modeled_month_dict = next(
        scenario_dict
        for scenario_dict in investor_summary_dict["scenario_list"]
        if scenario_dict["scenario_key_str"] == "modeled_21d"
    )
    assert isinstance(modeled_month_dict["horizon_day_int"], int)
    assert isinstance(modeled_month_dict["simulation_path_count_int"], int)

    report_html_str = (output_path / "report.html").read_text(encoding="utf-8")
    assert "Horizon Probability Tables" in report_html_str
    assert "Bootstrap-implied horizon probabilities from realized returns." in report_html_str
    assert "Downside drawdown path shares" in report_html_str
    assert "Upside reach path shares" in report_html_str
    assert "DD &lt;= -10%" in report_html_str
    assert "Gain &gt;= +10%" in report_html_str
    assert "Investor Scenario Summary" in report_html_str
    assert "edge looks real" not in report_html_str.lower()
    assert "1-in-20 bad case" not in report_html_str.lower()

    # The whole-sample drawdown probabilities belong to the downside horizon
    # table as one more row, not to a second table over the same thresholds.
    assert "Full sample" in report_html_str
    assert "Drawdown and Time-Underwater Breach Probabilities" not in report_html_str
    assert "Time Underwater" in report_html_str
    # Storage identifiers must not reach the page.
    assert "max_drawdown_lte_" not in report_html_str
    assert "underwater_ge_" not in report_html_str
    assert "terminal_return_lt_0" not in report_html_str
    assert "Underwater for 3 months or more" in report_html_str


def test_risk_analysis_saves_portfolio_entity_path_and_context(tmp_path):
    portfolio_obj = _toy_strategy_obj()
    portfolio_obj.name = "toy_portfolio"
    analysis_context_dict = {
        "analysis_status_str": "provisional_research_only",
        "source_config_path_str": "portfolios/toy.yaml",
    }

    risk_result_obj = RiskAnalysis(
        portfolio_obj,
        source_strategy_ref_str="results/toy_portfolio.pkl",
        source_entity_type_str="portfolio",
        analysis_context_dict=analysis_context_dict,
        output_dir_str=str(tmp_path),
        save_output_bool=True,
        primary_mean_block_length_int=2,
        mean_block_length_tuple=(2,),
        simulation_count_int=4,
        random_seed_int=11,
        horizon_year_tuple=(1,),
        rolling_loss_window_tuple=(1, 5),
    ).run()

    output_path = risk_result_obj.output_dir_path
    assert output_path is not None
    assert output_path.relative_to(tmp_path).parts[:4] == (
        "research",
        "portfolio",
        "toy_portfolio",
        RISK_ANALYSIS_TYPE_STR,
    )
    run_info_dict = json.loads((output_path / "run_info.json").read_text(encoding="utf-8"))
    assert run_info_dict["entity_type"] == "portfolio"
    investor_summary_dict = json.loads(
        (output_path / "investor_summary.json").read_text(encoding="utf-8")
    )
    assert investor_summary_dict["analysis_context_dict"] == analysis_context_dict
    report_html_str = (output_path / "report.html").read_text(encoding="utf-8")
    assert "ANALYSIS STATUS: PROVISIONAL RESEARCH ONLY" in report_html_str
    assert "NOT APPROVED FOR INVESTOR USE" in report_html_str


def test_risk_analysis_keeps_legacy_non_datetime_index_usable():
    strategy_obj = _toy_strategy_obj()
    strategy_obj.results.index = pd.RangeIndex(len(strategy_obj.results))

    risk_result_obj = RiskAnalysis(
        strategy_obj,
        save_output_bool=False,
        primary_mean_block_length_int=2,
        mean_block_length_tuple=(2,),
        simulation_count_int=4,
        horizon_year_tuple=(1,),
        rolling_loss_window_tuple=(1, 5),
    ).run()

    assert risk_result_obj.observed_calendar_month_df.empty
    observed_month_ser = risk_result_obj.investor_scenario_df[
        risk_result_obj.investor_scenario_df["scenario_key_str"]
        == "observed_calendar_month"
    ].iloc[0]
    assert int(observed_month_ser["sample_count_int"]) == 0


def test_risk_analysis_result_keeps_legacy_output_path_positional_slot(tmp_path):
    empty_df = pd.DataFrame()
    output_dir_path = tmp_path / "legacy-output"
    result_obj = RiskAnalysisResult(
        "toy",
        "strategies.toy",
        pd.Series(dtype=float),
        empty_df,
        empty_df,
        empty_df,
        empty_df,
        {},
        empty_df,
        output_dir_path,
    )

    assert result_obj.output_dir_path == output_dir_path


def test_time_underwater_rows_are_ordered_by_month_not_by_key():
    """Regression: the saved summary sorts keys lexicographically.

    Rendering a reloaded result in dict order printed 12m, 24m, 3m, 6m. A
    monotonically decreasing probability shown out of order reads as a broken
    model, so the table must order by the month the key encodes.
    """
    from alpha.engine.risk_analysis import _breach_table_html

    lexicographically_sorted_dict = {
        "underwater_ge_12m": 0.5203,
        "underwater_ge_24m": 0.0544,
        "underwater_ge_3m": 1.0,
        "underwater_ge_6m": 0.9854,
    }
    table_html_str = _breach_table_html(0.0, lexicographically_sorted_dict)
    position_list = [
        table_html_str.index(f"Underwater for {month_int} months or more")
        for month_int in (3, 6, 12, 24)
    ]
    assert position_list == sorted(position_list)


def test_full_sample_row_reports_every_drawdown_threshold():
    """The appended row must carry the same thresholds as the columns above it."""
    from alpha.engine.risk_analysis import _horizon_probability_tables_html

    horizon_df = pd.DataFrame(
        [{"horizon_year_int": 1, "simulation_path_count_int": 10,
          "drawdown_lte_10pct_probability_float": 0.27,
          "gain_gte_10pct_probability_float": 0.8}]
    )
    table_html_str = _horizon_probability_tables_html(
        horizon_df,
        drawdown_threshold_tuple=(0.10,),
        upside_threshold_tuple=(0.10,),
        full_sample_drawdown_row_dict={
            "max_drawdown_lte_10%": 0.9937,
            "simulation_path_count_int": 10000,
            "max_drawdown_p50_float": -0.1512,
            "max_drawdown_p05_float": -0.2162,
        },
    )
    assert "Full sample" in table_html_str
    assert "99.37%" in table_html_str
    assert 'class="full-sample-row"' in table_html_str
    # Omitting the row must leave the table unchanged apart from that row.
    assert "Full sample" not in _horizon_probability_tables_html(
        horizon_df, drawdown_threshold_tuple=(0.10,), upside_threshold_tuple=(0.10,)
    )

