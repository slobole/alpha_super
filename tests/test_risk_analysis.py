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
    _band_status_str,
    _build_verdict_row_list,
    build_bootstrap_equity_path_df,
    build_bootstrap_interval_df,
    build_bootstrap_path_metric_df,
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
    assert green_status_dict["Edge"] == VERDICT_STATUS_GREEN_STR
    assert green_status_dict["Drawdown depth"] == VERDICT_STATUS_GREEN_STR
    assert green_status_dict["Time underwater"] == VERDICT_STATUS_GREEN_STR
    assert green_status_dict["Worst year"] == VERDICT_STATUS_GREEN_STR

    # Low terminal-loss but Sharpe CI includes zero -> edge capped at amber.
    capped_summary_dict = dict(green_summary_dict)
    capped_summary_dict["primary_intervals"] = dict(green_summary_dict["primary_intervals"])
    capped_summary_dict["primary_intervals"]["sharpe_float"] = {"ci_lower_float": -0.1}
    capped_status_dict = _verdict_status_by_label(_build_verdict_row_list(capped_summary_dict))
    assert capped_status_dict["Edge"] == VERDICT_STATUS_AMBER_STR

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
    assert red_status_dict["Edge"] == VERDICT_STATUS_RED_STR
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
    assert "stress_status_str" not in summary_dict
    assert "var_95_daily_return_float" in summary_dict["observed_metrics"]
