import hashlib
import json

import numpy as np
import pandas as pd

from alpha.engine.portfolio import Portfolio
from scripts.research.run_investor_risk_sensitivity import (
    _run_sensitivity_variant_df,
    _build_sensitivity_variant_list,
    amplify_daily_volatility_ser,
    apply_additional_annual_drag_ser,
    build_modeled_21d_terminal_summary_dict,
    build_observed_crisis_window_df,
    build_observed_calendar_year_df,
    build_rolling_12m_return_df,
    run_monthly_numerical_stability_followup,
)


def test_additional_annual_drag_matches_declared_compounding_formula():
    return_idx = pd.date_range("2020-01-01", periods=252, freq="B")
    realized_return_ser = pd.Series(0.0, index=return_idx)

    stressed_return_ser = apply_additional_annual_drag_ser(
        realized_return_ser,
        annual_drag_float=0.04,
    )

    assert np.isclose(
        float(np.prod(1.0 + stressed_return_ser.to_numpy(dtype=float)) - 1.0),
        -0.04,
    )


def test_volatility_overlay_preserves_arithmetic_mean_and_scales_deviation():
    return_idx = pd.date_range("2020-01-01", periods=3, freq="B")
    realized_return_ser = pd.Series([-0.01, 0.00, 0.02], index=return_idx)

    stressed_return_ser = amplify_daily_volatility_ser(
        realized_return_ser,
        volatility_multiplier_float=1.5,
    )

    assert np.isclose(float(stressed_return_ser.mean()), float(realized_return_ser.mean()))
    original_deviation_vec = (
        realized_return_ser - float(realized_return_ser.mean())
    ).to_numpy(dtype=float)
    stressed_deviation_vec = (
        stressed_return_ser - float(stressed_return_ser.mean())
    ).to_numpy(dtype=float)
    assert np.allclose(stressed_deviation_vec, 1.5 * original_deviation_vec)


def test_sensitivity_variants_are_predeclared_without_weight_search():
    return_idx = pd.date_range("2012-10-01", "2026-07-09", freq="B")
    realized_return_ser = pd.Series(0.0005, index=return_idx)

    variant_list = _build_sensitivity_variant_list(realized_return_ser, random_seed_int=42)
    variant_key_list = [str(variant_dict["variant_key_str"]) for variant_dict in variant_list]

    assert len(variant_list) == 10
    assert variant_key_list == [
        "block_5",
        "block_10",
        "block_21",
        "block_63",
        "regime_pre_2020",
        "regime_2020_onward",
        "additional_drag_200bps",
        "additional_drag_400bps",
        "volatility_1p25x",
        "volatility_1p5x",
    ]
    assert all("weight" not in variant_key_str for variant_key_str in variant_key_list)


def test_observed_year_and_rolling_rows_label_partial_and_overlapping_windows():
    return_idx = pd.date_range("2020-07-01", periods=300, freq="B")
    realized_return_ser = pd.Series(0.001, index=return_idx)

    calendar_year_df = build_observed_calendar_year_df(realized_return_ser)
    rolling_12m_df = build_rolling_12m_return_df(realized_return_ser)

    assert bool(calendar_year_df.iloc[0]["partial_year_bool"]) is True
    assert len(rolling_12m_df) == 49
    assert rolling_12m_df["overlapping_observation_bool"].all()


def test_monthly_terminal_stability_is_deterministic_and_seed_sensitive():
    return_idx = pd.date_range("2020-01-01", periods=252, freq="B")
    realized_return_ser = pd.Series(
        np.sin(np.arange(252, dtype=float)) * 0.01 + 0.0005,
        index=return_idx,
    )

    first_summary_dict = build_modeled_21d_terminal_summary_dict(
        realized_return_ser,
        simulation_count_int=50,
        random_seed_int=7,
    )
    repeated_summary_dict = build_modeled_21d_terminal_summary_dict(
        realized_return_ser,
        simulation_count_int=50,
        random_seed_int=7,
    )
    alternate_summary_dict = build_modeled_21d_terminal_summary_dict(
        realized_return_ser,
        simulation_count_int=50,
        random_seed_int=99,
    )

    assert first_summary_dict == repeated_summary_dict
    assert any(
        not np.isclose(
            float(first_summary_dict[metric_name_str]),
            float(alternate_summary_dict[metric_name_str]),
        )
        for metric_name_str in (
            "terminal_return_p05_float",
            "terminal_return_p50_float",
            "terminal_loss_probability_float",
        )
    )


def test_empty_regime_and_crisis_outputs_keep_explicit_schema(tmp_path):
    return_idx = pd.date_range("2021-01-01", periods=60, freq="B")
    realized_return_ser = pd.Series(0.001, index=return_idx)
    variant_list = _build_sensitivity_variant_list(realized_return_ser, random_seed_int=42)
    pre_2020_variant_dict = next(
        variant_dict
        for variant_dict in variant_list
        if variant_dict["variant_key_str"] == "regime_pre_2020"
    )

    skipped_variant_df = _run_sensitivity_variant_df(
        pre_2020_variant_dict,
        simulation_count_int=4,
    )
    short_available_variant_dict = next(
        variant_dict
        for variant_dict in variant_list
        if variant_dict["variant_key_str"] == "block_21"
    )
    short_available_variant_df = _run_sensitivity_variant_df(
        short_available_variant_dict,
        simulation_count_int=4,
    )
    crisis_df = build_observed_crisis_window_df(realized_return_ser)
    crisis_csv_path = tmp_path / "empty_crisis.csv"
    crisis_df.to_csv(crisis_csv_path, index=False)
    roundtrip_crisis_df = pd.read_csv(crisis_csv_path)

    assert skipped_variant_df["variant_status_str"].eq(
        "skipped_empty_return_history"
    ).all()
    assert skipped_variant_df["scenario_key_str"].tolist() == [
        "modeled_21d",
        "modeled_1y",
        "modeled_3y",
        "modeled_5y",
    ]
    short_status_dict = dict(
        zip(
            short_available_variant_df["scenario_key_str"],
            short_available_variant_df["scenario_status_str"],
            strict=True,
        )
    )
    assert short_status_dict["modeled_21d"] == "available"
    assert short_status_dict["modeled_1y"] == "unavailable_insufficient_history"
    assert short_status_dict["modeled_3y"] == "unavailable_insufficient_history"
    assert short_status_dict["modeled_5y"] == "unavailable_insufficient_history"
    assert list(roundtrip_crisis_df.columns) == list(crisis_df.columns)
    assert crisis_df.empty


def test_monthly_followup_links_unresolved_parent_without_false_resolution(
    tmp_path,
    monkeypatch,
):
    result_idx = pd.date_range("2020-01-01", periods=252, freq="B")
    daily_return_ser = pd.Series(
        np.sin(np.arange(252, dtype=float)) * 0.005 + 0.0005,
        index=result_idx,
    )
    daily_return_ser.iloc[0] = 0.0
    portfolio_obj = Portfolio.__new__(Portfolio)
    portfolio_obj.name = "toy_portfolio"
    portfolio_obj.results = pd.DataFrame(
        {
            "daily_returns": daily_return_ser,
            "total_value": 100.0 * (1.0 + daily_return_ser).cumprod(),
        },
        index=result_idx,
    )
    portfolio_obj.weights = [1.0]
    portfolio_obj.pod_info_list = [
        {
            "pod_id_str": "pod_a",
            "strategy_name": "strategy_a",
            "strategy_import_str": "strategies.strategy_a",
            "weight": 1.0,
        }
    ]
    portfolio_obj._capital_base = 100.0
    portfolio_obj._rebalance = None
    portfolio_obj._rebalance_policy = "fixed"
    portfolio_obj._pod_equities = pd.DataFrame(
        {"strategy_a": portfolio_obj.results["total_value"]},
        index=result_idx,
    )
    source_dir_path = tmp_path / "source"
    source_dir_path.mkdir()
    portfolio_pickle_path = source_dir_path / "toy_portfolio.pkl"
    portfolio_obj.to_pickle(portfolio_pickle_path)
    source_hash_str = hashlib.sha256(portfolio_pickle_path.read_bytes()).hexdigest()
    (source_dir_path / "metadata.json").write_text(
        json.dumps(
            {
                "artifact_type": "portfolio",
                "portfolio_name": "toy_portfolio",
                "pickle_path": str(portfolio_pickle_path),
                "common_start": result_idx.min().isoformat(),
                "common_end": result_idx.max().isoformat(),
                "capital_base": 100.0,
                "rebalance": None,
                "rebalance_policy": "fixed",
                "pods": portfolio_obj.pod_info_list,
            }
        ),
        encoding="utf-8",
    )
    parent_dir_path = tmp_path / "parent"
    parent_dir_path.mkdir()
    (parent_dir_path / "summary.json").write_text(
        json.dumps(
            {
                "source_context_dict": {
                    "source_artifact_sha256_str": source_hash_str,
                },
                "seed_stability_dict": {
                    "status_str": "requires_more_simulations_or_review",
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "scripts.research.run_investor_risk_sensitivity.MONTHLY_STABILITY_ABSOLUTE_TOLERANCE_FLOAT",
        0.0,
    )
    followup_dir_path = run_monthly_numerical_stability_followup(
        portfolio_pickle_path=portfolio_pickle_path,
        output_dir_str=str(tmp_path / "output"),
        simulation_count_int=100,
        random_seed_int=42,
        parent_sensitivity_dir_path=parent_dir_path,
    )

    resolution_dict = json.loads(
        (parent_dir_path / "numerical_stability_followup.json").read_text(
            encoding="utf-8"
        )
    )
    followup_summary_dict = json.loads(
        (followup_dir_path / "summary.json").read_text(encoding="utf-8")
    )
    assert resolution_dict["followup_dir_path_str"] == str(followup_dir_path.resolve())
    assert resolution_dict["numerical_convergence_resolved_bool"] is False
    assert "resolved_metric_list" not in resolution_dict
    assert "did not resolve" in resolution_dict["interpretation_str"]
    followup_markdown_str = (
        parent_dir_path / "NUMERICAL_STABILITY_FOLLOWUP.md"
    ).read_text(encoding="utf-8")
    assert "does not resolve" in followup_markdown_str
    assert followup_summary_dict["parent_sensitivity_link_dict"][
        "parent_original_seed_stability_status_str"
    ] == "requires_more_simulations_or_review"
