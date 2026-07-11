from pathlib import Path

import pandas as pd
import pytest

from alpha.engine.portfolio import Portfolio
from strategies.run_portfolio_risk_analysis import (
    _build_portfolio_analysis_context_dict,
    _validate_portfolio_artifact,
)


def _portfolio_without_constructor() -> Portfolio:
    portfolio_obj = Portfolio.__new__(Portfolio)
    portfolio_obj.name = "toy_portfolio"
    result_idx = pd.to_datetime(["2020-01-02", "2020-01-03"])
    portfolio_obj.results = pd.DataFrame(
        {
            "total_value": [100.0, 101.0],
            "daily_returns": [0.0, 0.01],
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
        {"strategy_a": [100.0, 101.0]},
        index=result_idx,
    )
    return portfolio_obj


def test_validate_portfolio_artifact_matches_explicit_metadata(tmp_path):
    portfolio_pickle_path = tmp_path / "toy_portfolio.pkl"
    portfolio_pickle_path.write_bytes(b"trusted-test-placeholder")
    portfolio_obj = _portfolio_without_constructor()
    source_metadata_dict = {
        "artifact_type": "portfolio",
        "portfolio_name": "toy_portfolio",
        "pickle_path": str(portfolio_pickle_path),
        "common_start": "2020-01-02T00:00:00",
        "common_end": "2020-01-03T00:00:00",
        "capital_base": 100.0,
        "rebalance": None,
        "rebalance_policy": "fixed",
        "pods": [
            {
                "pod_id_str": "pod_a",
                "strategy_name": "strategy_a",
                "strategy_import_str": "strategies.strategy_a",
                "weight": 1.0,
            }
        ],
    }

    _validate_portfolio_artifact(
        portfolio_obj,
        portfolio_pickle_path,
        source_metadata_dict,
    )

    source_metadata_dict["common_end"] = "2020-01-04T00:00:00"
    with pytest.raises(ValueError, match="realized end"):
        _validate_portfolio_artifact(
            portfolio_obj,
            portfolio_pickle_path,
            source_metadata_dict,
        )


def test_portfolio_context_preserves_configured_and_realized_windows(tmp_path):
    portfolio_pickle_path = tmp_path / "toy_portfolio.pkl"
    portfolio_pickle_path.write_bytes(b"portfolio-bytes")
    source_config_path = tmp_path / "toy_portfolio.yaml"
    source_config_path.write_text("name: toy_portfolio\n", encoding="utf-8")
    source_metadata_dict = {
        "saved_at": "2026-07-10T00:44:55",
        "source_config_path": str(source_config_path),
        "common_start": "2012-10-01T00:00:00",
        "common_end": "2026-07-09T00:00:00",
        "capital_base": 100000.0,
        "rebalance": None,
        "rebalance_policy": "fixed",
        "pods": [
            {
                "pod_id_str": "pod_a",
                "strategy_name": "strategy_a",
                "strategy_import_str": "strategies.strategy_a",
                "weight": 0.6,
                "allocated_capital": 60000.0,
                "source_type_str": "fresh_run",
            }
        ],
    }
    manager_metadata_dict = {
        "backtest_start_date_str": "2004-01-01",
        "end_date_str": None,
        "allocation_policy_str": "fixed",
        "validation_status_str": "passed",
    }

    analysis_context_dict = _build_portfolio_analysis_context_dict(
        portfolio_pickle_path=portfolio_pickle_path,
        source_metadata_dict=source_metadata_dict,
        manager_metadata_dict=manager_metadata_dict,
    )

    assert analysis_context_dict["analysis_status_str"] == "provisional_research_only"
    assert analysis_context_dict["investor_use_approved_bool"] is False
    assert analysis_context_dict["configured_backtest_start_date_str"] == "2004-01-01"
    assert analysis_context_dict["realized_common_start_date_str"] == "2012-10-01T00:00:00"
    assert analysis_context_dict["rebalance_frequency_str"] == "none"
    assert analysis_context_dict["source_config_exists_bool"] is True
    assert len(str(analysis_context_dict["source_artifact_sha256_str"])) == 64
    assert len(str(analysis_context_dict["analysis_time_source_config_sha256_str"])) == 64
    assert analysis_context_dict["pod_list"][0]["weight_float"] == 0.6
    assert (
        analysis_context_dict["strategy_source_revision_validation_status_str"]
        == "not_captured_by_source_artifact"
    )


def test_validate_portfolio_rejects_bad_order_returns_and_metadata(tmp_path):
    portfolio_pickle_path = tmp_path / "toy_portfolio.pkl"
    portfolio_pickle_path.write_bytes(b"trusted-test-placeholder")
    source_metadata_dict = {
        "artifact_type": "portfolio",
        "portfolio_name": "toy_portfolio",
        "pickle_path": str(portfolio_pickle_path),
        "common_start": "2020-01-02T00:00:00",
        "common_end": "2020-01-03T00:00:00",
        "capital_base": 100.0,
        "rebalance": None,
        "rebalance_policy": "fixed",
        "pods": [
            {
                "pod_id_str": "pod_a",
                "strategy_name": "strategy_a",
                "strategy_import_str": "strategies.strategy_a",
                "weight": 1.0,
            }
        ],
    }

    unsorted_portfolio_obj = _portfolio_without_constructor()
    unsorted_portfolio_obj.results = unsorted_portfolio_obj.results.iloc[::-1]
    with pytest.raises(ValueError, match="monotonic"):
        _validate_portfolio_artifact(
            unsorted_portfolio_obj,
            portfolio_pickle_path,
            source_metadata_dict,
        )

    duplicate_portfolio_obj = _portfolio_without_constructor()
    duplicate_portfolio_obj.results.index = pd.to_datetime(
        ["2020-01-02", "2020-01-02"]
    )
    with pytest.raises(ValueError, match="unique"):
        _validate_portfolio_artifact(
            duplicate_portfolio_obj,
            portfolio_pickle_path,
            source_metadata_dict,
        )

    inconsistent_return_portfolio_obj = _portfolio_without_constructor()
    inconsistent_return_portfolio_obj.results.loc[
        pd.Timestamp("2020-01-03"), "daily_returns"
    ] = 0.02
    with pytest.raises(ValueError, match="disagrees"):
        _validate_portfolio_artifact(
            inconsistent_return_portfolio_obj,
            portfolio_pickle_path,
            source_metadata_dict,
        )

    nonfinite_return_portfolio_obj = _portfolio_without_constructor()
    nonfinite_return_portfolio_obj.results.loc[
        pd.Timestamp("2020-01-03"), "daily_returns"
    ] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        _validate_portfolio_artifact(
            nonfinite_return_portfolio_obj,
            portfolio_pickle_path,
            source_metadata_dict,
        )

    nonzero_initial_return_portfolio_obj = _portfolio_without_constructor()
    nonzero_initial_return_portfolio_obj.results.iloc[0, nonzero_initial_return_portfolio_obj.results.columns.get_loc("daily_returns")] = 0.01
    with pytest.raises(ValueError, match="zero initial-state placeholder"):
        _validate_portfolio_artifact(
            nonzero_initial_return_portfolio_obj,
            portfolio_pickle_path,
            source_metadata_dict,
        )

    mismatched_weight_metadata_dict = dict(source_metadata_dict)
    mismatched_weight_metadata_dict["pods"] = [
        {**source_metadata_dict["pods"][0], "weight": 0.9}
    ]
    with pytest.raises(ValueError, match="weights"):
        _validate_portfolio_artifact(
            _portfolio_without_constructor(),
            portfolio_pickle_path,
            mismatched_weight_metadata_dict,
        )

    mismatched_rebalance_metadata_dict = dict(source_metadata_dict)
    mismatched_rebalance_metadata_dict["rebalance"] = "monthly"
    with pytest.raises(ValueError, match="rebalance frequency"):
        _validate_portfolio_artifact(
            _portfolio_without_constructor(),
            portfolio_pickle_path,
            mismatched_rebalance_metadata_dict,
        )


def test_portfolio_context_reports_realized_end_weight_drift(tmp_path):
    portfolio_pickle_path = tmp_path / "toy_portfolio.pkl"
    portfolio_pickle_path.write_bytes(b"portfolio-bytes")
    portfolio_obj = _portfolio_without_constructor()
    portfolio_obj.weights = [0.8, 0.2]
    portfolio_obj._pod_equities = pd.DataFrame(
        {"strategy_a": [80.0, 90.0], "strategy_b": [20.0, 10.0]},
        index=portfolio_obj.results.index,
    )

    analysis_context_dict = _build_portfolio_analysis_context_dict(
        portfolio_pickle_path=portfolio_pickle_path,
        source_metadata_dict={"pods": [], "rebalance": None},
        manager_metadata_dict={},
        portfolio_obj=portfolio_obj,
    )

    assert analysis_context_dict["realized_end_weight_list"] == [
        {"strategy_name_str": "strategy_a", "realized_end_weight_float": 0.9},
        {"strategy_name_str": "strategy_b", "realized_end_weight_float": 0.1},
    ]
    assert analysis_context_dict["max_absolute_weight_drift_float"] == pytest.approx(0.1)
