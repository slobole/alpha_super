from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from alpha import strategy_registry
from scripts.research import run_clean_sheet_portfolio_foundry_study as study


def test_frozen_spec_exactly_covers_promoted_registry() -> None:
    spec_dict = study.load_spec_dict()
    assert spec_dict["portfolio_contract"]["end_date_str"] == "2026-08-19"
    selected_strategy_dict = spec_dict["registry_contract"][
        "selected_strategy_by_alias"
    ]
    selected_import_set = {
        strategy_spec_dict["strategy_import_str"]
        for strategy_spec_dict in selected_strategy_dict.values()
    }
    excluded_import_set = {
        strategy_spec_dict["strategy_import_str"]
        for strategy_spec_dict in spec_dict["registry_contract"][
            "excluded_promoted_strategy_list"
        ]
    }

    assert selected_import_set.isdisjoint(excluded_import_set)
    assert selected_import_set | excluded_import_set == set(
        strategy_registry.STRATEGY_TIER_DICT
    )
    assert len(selected_import_set) == 10
    assert len(excluded_import_set) == 12


def test_hebrew_report_uses_frozen_portfolio_dates(tmp_path: Path) -> None:
    spec_dict = study.load_spec_dict()
    headline_row_list = []
    for product_id_str, product_spec_dict in study.all_product_spec_dict(
        spec_dict
    ).items():
        headline_row_list.append(
            {
                "product_id_str": product_id_str,
                "objective_str": product_spec_dict["objective_str"],
                "cagr_float": 0.10,
                "sharpe_float": 1.0,
                "max_drawdown_float": -0.10,
                "es5_loss_float": 0.02,
                "market_beta_float": 0.50,
            }
        )
    gate_df = pd.DataFrame(
        [
            {
                "candidate_id_str": "C1_foundry_defensive",
                "reference_id_str": "R1_ladder_defensive",
                "economic_gate_bool": False,
                "subperiod_gate_bool": False,
                "statistical_gate_bool": False,
                "unmodeled_financing_gate_bool": False,
                "non_capacity_gate_bool": False,
                "decision_str": "reject_without_weight_tuning",
            }
        ]
    )

    report_path = study.write_hebrew_report(
        headline_metric_df=pd.DataFrame(headline_row_list),
        gate_df=gate_df,
        source_run_summary_df=pd.DataFrame({"source_id_str": ["one_source"]}),
        spec_dict=spec_dict,
        output_dir_path=tmp_path,
    )
    report_str = report_path.read_text(encoding="utf-8")

    assert "2012-09-28 כעוגן מזומן" in report_str
    assert "ביצוע ראשון ב־2012-10-01" in report_str
    assert "ועד 2026-08-19" in report_str
    assert "V_{portfolio,t}" in report_str
    assert "r_{portfolio,t}" in report_str
    assert "\\sum_{k=1}^{K}" in report_str
    assert "\\frac{V_{portfolio,t}}{V_{portfolio,t-1}}" in report_str
    assert "[נתונים זמינים עד T]" in report_str
    assert "משברי 2008 ו־2011 קודמים לעוגן המשותף" in report_str


def test_source_expansion_is_deterministic_and_reuses_exact_capital_paths() -> None:
    spec_dict = study.load_spec_dict()
    first_expanded_spec_dict, first_source_map_dict = (
        study.expanded_source_spec_dict(spec_dict)
    )
    second_expanded_spec_dict, second_source_map_dict = (
        study.expanded_source_spec_dict(spec_dict)
    )

    assert first_expanded_spec_dict["source_runs"] == second_expanded_spec_dict[
        "source_runs"
    ]
    assert first_source_map_dict == second_source_map_dict
    assert len(first_expanded_spec_dict["source_runs"]) == 23
    assert (
        first_source_map_dict["C2_foundry_balanced"]["core5"]
        == first_source_map_dict["C3_foundry_growth"]["core5"]
        == first_source_map_dict["C4_foundry_low_touch"]["core5"]
    )
    sector_source_spec_list = [
        source_spec_dict
        for source_spec_dict in first_expanded_spec_dict["source_runs"].values()
        if source_spec_dict["strategy_import_str"]
        == "strategies.mean_reversion.strategy_mr_us_sector_etf_ibs_downshock_vox_iyr"
    ]
    assert sector_source_spec_list
    assert {
        source_spec_dict["engine_request_start_date_str"]
        for source_spec_dict in sector_source_spec_list
    } == {"2012-09-28"}


def test_low_touch_candidate_contains_no_daily_signal_strategy() -> None:
    spec_dict = study.load_spec_dict()
    selected_strategy_dict = spec_dict["registry_contract"][
        "selected_strategy_by_alias"
    ]
    low_touch_alias_set = set(
        spec_dict["candidate_portfolios"]["C4_foundry_low_touch"][
            "weight_by_strategy_alias"
        ]
    )

    assert all(
        selected_strategy_dict[strategy_alias_str]["cadence_str"] != "daily_signal"
        for strategy_alias_str in low_touch_alias_set
    )


def test_global_product_frames_sum_independent_pod_equities_without_fill(
    monkeypatch,
    tmp_path: Path,
) -> None:
    spec_dict = study.load_spec_dict()
    expanded_spec_dict, source_map_dict = study.expanded_source_spec_dict(spec_dict)
    date_idx = pd.DatetimeIndex(
        ["2012-09-28", "2012-10-01", "2026-08-19"],
        name="date",
    )
    source_path_by_id_dict = {}
    for source_id_str, source_spec_dict in expanded_spec_dict["source_runs"].items():
        capital_float = float(source_spec_dict["allocated_capital_float"])
        source_path_by_id_dict[source_id_str] = pd.DataFrame(
            {
                "total_value_float": [capital_float, capital_float * 1.01, capital_float * 1.02],
                "portfolio_value_float": [0.0, capital_float * 1.005, capital_float * 1.01],
                "cash_float": [capital_float, capital_float * 0.005, capital_float * 0.01],
            },
            index=date_idx,
        )
    benchmark_price_df = pd.DataFrame(
        {"Close": [100.0, 101.0, 102.0]},
        index=date_idx,
    )
    monkeypatch.setattr(
        study.ladder_runner,
        "load_all_source_path_dict",
        lambda expanded_spec_dict, output_dir_path: source_path_by_id_dict,
    )
    monkeypatch.setattr(
        study,
        "load_price_timeseries",
        lambda *args, **kwargs: benchmark_price_df,
    )

    (
        total_value_df,
        portfolio_value_df,
        cash_df,
        benchmark_total_value_ser,
        _,
        _,
    ) = study.build_global_product_frames(
        expanded_spec_dict,
        source_map_dict,
        tmp_path,
    )

    assert np.allclose(total_value_df.iloc[0].to_numpy(dtype=float), 1_000_000.0)
    assert np.allclose(total_value_df.iloc[-1].to_numpy(dtype=float), 1_020_000.0)
    assert np.allclose(portfolio_value_df.iloc[0].to_numpy(dtype=float), 0.0)
    assert np.allclose(cash_df.iloc[0].to_numpy(dtype=float), 1_000_000.0)
    assert benchmark_total_value_ser.iloc[-1] == 1_020_000.0


def test_subperiods_cover_every_realized_return_once() -> None:
    spec_dict = study.load_spec_dict()
    date_idx = pd.bdate_range("2020-01-01", periods=11)
    total_value_df = pd.DataFrame(
        {
            product_id_str: 1_000_000.0
            * np.cumprod(np.r_[1.0, np.repeat(1.001, 10)])
            for product_id_str in study.all_product_spec_dict(spec_dict)
        },
        index=date_idx,
    )
    benchmark_return_ser = pd.Series(0.001, index=date_idx, dtype=float)
    benchmark_return_ser.iloc[0] = 0.0

    subperiod_metric_df = study.calculate_subperiod_metric_df(
        total_value_df,
        benchmark_return_ser,
        spec_dict,
    )

    one_product_df = subperiod_metric_df.loc[
        subperiod_metric_df["product_id_str"] == "C1_foundry_defensive"
    ]
    assert one_product_df["observation_count_int"].tolist() == [4, 3, 3]
    assert int(one_product_df["observation_count_int"].sum()) == 10


def test_bootstrap_is_deterministic_for_same_frozen_seed() -> None:
    spec_dict = deepcopy(study.load_spec_dict())
    spec_dict["statistical_contract"]["bootstrap_iteration_count_int"] = 20
    spec_dict["statistical_contract"]["bootstrap_chunk_size_int"] = 5
    date_idx = pd.bdate_range("2020-01-01", periods=65)
    random_generator_obj = np.random.default_rng(7)
    return_df = pd.DataFrame(
        {
            product_id_str: np.r_[
                0.0,
                random_generator_obj.normal(0.0004, 0.007, len(date_idx) - 1),
            ]
            for product_id_str in study.all_product_spec_dict(spec_dict)
        },
        index=date_idx,
    )

    first_metric_dict = study.bootstrap_metric_array_by_product_dict(
        return_df,
        spec_dict,
        mean_block_length_int=5,
    )
    second_metric_dict = study.bootstrap_metric_array_by_product_dict(
        return_df,
        spec_dict,
        mean_block_length_int=5,
    )

    for product_id_str in return_df.columns:
        for metric_name_str in first_metric_dict[product_id_str]:
            assert np.array_equal(
                first_metric_dict[product_id_str][metric_name_str],
                second_metric_dict[product_id_str][metric_name_str],
            )


def test_bootstrap_drawdown_includes_unit_nav_anchor(monkeypatch) -> None:
    spec_dict = deepcopy(study.load_spec_dict())
    spec_dict["statistical_contract"]["bootstrap_iteration_count_int"] = 1
    spec_dict["statistical_contract"]["bootstrap_chunk_size_int"] = 1
    return_df = pd.DataFrame(
        {"one_product": [0.0, -0.10, 0.0]},
        index=pd.bdate_range("2020-01-01", periods=3),
    )
    monkeypatch.setattr(
        study.ladder_runner,
        "stationary_bootstrap_index_chunk_mat",
        lambda **kwargs: np.array([[0, 1]], dtype=int),
    )

    metric_dict = study.bootstrap_metric_array_by_product_dict(
        return_df,
        spec_dict,
        mean_block_length_int=2,
    )

    assert np.isclose(
        metric_dict["one_product"]["max_drawdown_float"][0],
        -0.10,
    )


def test_frozen_study_forbids_resume_before_creating_output(tmp_path: Path) -> None:
    output_dir_path = tmp_path / "must_remain_absent"

    with np.testing.assert_raises_regex(ValueError, "forbid --resume"):
        study.run_study(output_dir_path=output_dir_path, resume_bool=True)

    assert not output_dir_path.exists()


def test_source_lineage_rejects_strategy_module_drift(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_id_str = "test_100"
    strategy_import_str = "strategies.test_strategy"
    source_path = tmp_path / "source_paths" / f"{source_id_str}.csv.gz"
    transaction_path = (
        tmp_path / "source_transactions" / f"{source_id_str}.csv.gz"
    )
    metadata_path = tmp_path / "source_metadata" / f"{source_id_str}.json"
    for parent_path in (source_path.parent, transaction_path.parent, metadata_path.parent):
        parent_path.mkdir(parents=True, exist_ok=True)
    source_path_df = pd.DataFrame(
        {
            "total_value_float": [100.0, 100.0, 101.0],
            "portfolio_value_float": [0.0, 90.0, 91.0],
            "cash_float": [100.0, 10.0, 10.0],
        },
        index=pd.DatetimeIndex(["2012-09-28", "2012-10-01", "2026-08-19"]),
    )
    source_path_df.to_csv(source_path, compression="gzip", index_label="date")
    pd.DataFrame({"date": pd.Series(dtype=str)}).to_csv(
        transaction_path,
        compression="gzip",
        index=False,
    )
    module_path = tmp_path / "test_strategy.py"
    module_path.write_text("VALUE_INT = 1\n", encoding="utf-8")
    shared_hash_dict = {"shared.py": "current"}
    monkeypatch.setattr(
        study.ladder_runner,
        "shared_execution_dependency_hash_dict",
        lambda: shared_hash_dict,
    )
    metadata_dict = {
        "source_id_str": source_id_str,
        "strategy_import_str": strategy_import_str,
        "requested_history_start_date_str": "2004-01-01",
        "native_history_request_start_date_str": "1998-01-01",
        "allocated_capital_float": 100.0,
        "run_variant_kwargs_dict": {},
        "engine_request_start_date_str": "2012-10-01",
        "actual_start_date_str": "2012-09-28",
        "strategy_result_start_date_str": "2012-10-01",
        "actual_end_date_str": "2026-08-19",
        "source_path_sha256_str": study.ladder_runner.sha256_file_str(source_path),
        "transaction_path_sha256_str": study.ladder_runner.sha256_file_str(
            transaction_path
        ),
        "module_path_str": str(module_path),
        "module_sha256_str": "stale-module-hash",
        "shared_execution_dependency_hash_dict": shared_hash_dict,
    }
    metadata_path.write_text(
        json.dumps(metadata_dict, sort_keys=True),
        encoding="utf-8",
    )
    source_run_summary_df = pd.DataFrame(
        [
            {
                "source_id_str": source_id_str,
                "actual_start_date_str": "2012-09-28",
                "actual_end_date_str": "2026-08-19",
                "source_path_sha256_str": study.ladder_runner.sha256_file_str(
                    source_path
                ),
                "transaction_path_sha256_str": study.ladder_runner.sha256_file_str(
                    transaction_path
                ),
                "metadata_sha256_str": study.ladder_runner.sha256_file_str(
                    metadata_path
                ),
            }
        ]
    )
    expanded_spec_dict = {
        "source_runs": {
            source_id_str: {
                "strategy_import_str": strategy_import_str,
                "allocated_capital_float": 100.0,
                "run_variant_kwargs_dict": {},
            }
        },
        "portfolio_contract": {
            "requested_start_date_str": "2004-01-01",
            "capital_anchor_date_str": "2012-09-28",
            "effective_execution_start_date_str": "2012-10-01",
            "end_date_str": "2026-08-19",
        },
        "lineage_contract": {
            "native_history_request_start_by_strategy_import": {
                strategy_import_str: "1998-01-01"
            }
        },
    }

    with np.testing.assert_raises_regex(RuntimeError, "strategy module changed"):
        study.validate_source_lineage_bool(
            expanded_spec_dict,
            source_run_summary_df,
            tmp_path,
        )


def test_phase1_capacity_can_never_promote_a_candidate() -> None:
    spec_dict = study.load_spec_dict()
    assert spec_dict["capacity_contract"]["phase1_capacity_gate_bool"] is False
    assert all(
        candidate_spec_dict["reference_id_str"]
        in spec_dict["reference_portfolios"]
        for candidate_spec_dict in spec_dict["candidate_portfolios"].values()
    )


def _passing_gate_fixture_tuple():
    spec_dict = study.load_spec_dict()
    expanded_spec_dict, source_map_dict = study.expanded_source_spec_dict(spec_dict)
    headline_row_list = []
    for product_id_str in study.all_product_spec_dict(spec_dict):
        candidate_bool = product_id_str in spec_dict["candidate_portfolios"]
        headline_row_list.append(
            {
                "product_id_str": product_id_str,
                "cagr_float": 0.12 if candidate_bool else 0.10,
                "sharpe_float": 1.20 if candidate_bool else 1.00,
                "max_drawdown_float": -0.10 if candidate_bool else -0.15,
                "es5_loss_float": 0.015 if candidate_bool else 0.020,
                "market_beta_float": 0.50 if candidate_bool else 0.60,
            }
        )
    headline_metric_df = pd.DataFrame(headline_row_list)
    subperiod_row_list = []
    for third_position_int in range(1, 4):
        for product_id_str in study.all_product_spec_dict(spec_dict):
            candidate_bool = product_id_str in spec_dict["candidate_portfolios"]
            subperiod_row_list.append(
                {
                    "subperiod_id_str": f"third_{third_position_int}",
                    "product_id_str": product_id_str,
                    "cagr_float": 0.12 if candidate_bool else 0.10,
                    "max_drawdown_float": -0.10 if candidate_bool else -0.15,
                }
            )
    subperiod_metric_df = pd.DataFrame(subperiod_row_list)
    bootstrap_row_list = []
    for candidate_id_str in spec_dict["candidate_portfolios"]:
        for block_length_int in (21, 63, 126):
            bootstrap_row_list.append(
                {
                    "candidate_id_str": candidate_id_str,
                    "mean_block_length_int": block_length_int,
                    "composite_success_probability_float": 0.90,
                }
            )
    bootstrap_summary_df = pd.DataFrame(bootstrap_row_list)
    holm_df = pd.DataFrame(
        {
            "candidate_id_str": list(spec_dict["candidate_portfolios"]),
            "holm_adjusted_p_value_float": 0.01,
        }
    )
    source_run_summary_df = pd.DataFrame(
        {
            "source_id_str": list(expanded_spec_dict["source_runs"]),
            "negative_cash_day_count_int": 0,
        }
    )
    return (
        spec_dict,
        source_map_dict,
        headline_metric_df,
        subperiod_metric_df,
        bootstrap_summary_df,
        holm_df,
        source_run_summary_df,
    )


def test_perfect_phase1_evidence_can_advance_but_never_promote() -> None:
    fixture_tuple = _passing_gate_fixture_tuple()
    gate_df = study.evaluate_candidate_gate_df(
        fixture_tuple[2],
        fixture_tuple[3],
        fixture_tuple[4],
        fixture_tuple[5],
        fixture_tuple[6],
        fixture_tuple[1],
        fixture_tuple[0],
        source_lineage_gate_bool=True,
    )

    assert gate_df["non_capacity_gate_bool"].all()
    assert not gate_df["capacity_gate_bool"].any()
    assert not gate_df["promotion_gate_bool"].any()
    assert set(gate_df["decision_str"]) == {
        "advance_to_separately_preregistered_capacity_phase"
    }


def test_one_negative_cash_source_blocks_its_candidate() -> None:
    fixture_tuple = _passing_gate_fixture_tuple()
    source_run_summary_df = fixture_tuple[6].copy()
    blocked_source_id_str = next(iter(fixture_tuple[1]["C1_foundry_defensive"].values()))
    source_run_summary_df.loc[
        source_run_summary_df["source_id_str"] == blocked_source_id_str,
        "negative_cash_day_count_int",
    ] = 1
    gate_df = study.evaluate_candidate_gate_df(
        fixture_tuple[2],
        fixture_tuple[3],
        fixture_tuple[4],
        fixture_tuple[5],
        source_run_summary_df,
        fixture_tuple[1],
        fixture_tuple[0],
        source_lineage_gate_bool=True,
    ).set_index("candidate_id_str")

    assert not bool(
        gate_df.loc["C1_foundry_defensive", "unmodeled_financing_gate_bool"]
    )
    assert not bool(gate_df.loc["C1_foundry_defensive", "non_capacity_gate_bool"])
    assert gate_df.loc["C1_foundry_defensive", "decision_str"] == (
        "reject_without_weight_tuning"
    )


def test_one_negative_cash_reference_source_blocks_the_comparison() -> None:
    fixture_tuple = _passing_gate_fixture_tuple()
    source_run_summary_df = fixture_tuple[6].copy()
    candidate_source_id_set = set(fixture_tuple[1]["C1_foundry_defensive"].values())
    reference_source_id_set = set(fixture_tuple[1]["R1_ladder_defensive"].values())
    blocked_source_id_str = next(iter(reference_source_id_set - candidate_source_id_set))
    source_run_summary_df.loc[
        source_run_summary_df["source_id_str"] == blocked_source_id_str,
        "negative_cash_day_count_int",
    ] = 1
    gate_df = study.evaluate_candidate_gate_df(
        fixture_tuple[2],
        fixture_tuple[3],
        fixture_tuple[4],
        fixture_tuple[5],
        source_run_summary_df,
        fixture_tuple[1],
        fixture_tuple[0],
        source_lineage_gate_bool=True,
    ).set_index("candidate_id_str")

    assert gate_df.loc[
        "C1_foundry_defensive", "candidate_negative_cash_day_count_int"
    ] == 0
    assert gate_df.loc[
        "C1_foundry_defensive", "reference_negative_cash_day_count_int"
    ] == 1
    assert not bool(
        gate_df.loc["C1_foundry_defensive", "unmodeled_financing_gate_bool"]
    )
    assert not bool(gate_df.loc["C1_foundry_defensive", "non_capacity_gate_bool"])


def test_bootstrap_p_values_use_64_trial_family_before_holm(monkeypatch) -> None:
    spec_dict = study.load_spec_dict()
    product_id_list = list(study.all_product_spec_dict(spec_dict))
    return_df = pd.DataFrame(
        {product_id_str: [0.0, 0.001] for product_id_str in product_id_list},
        index=pd.bdate_range("2020-01-01", periods=2),
    )
    headline_metric_df = pd.DataFrame(
        {
            "product_id_str": product_id_list,
            "cagr_float": 0.10,
            "sharpe_float": 1.0,
            "max_drawdown_float": -0.10,
            "es5_loss_float": 0.02,
        }
    )
    metric_array_by_product_dict = {
        product_id_str: {
            "cagr_float": np.full(4, 0.10),
            "sharpe_float": np.full(4, 1.0),
            "max_drawdown_float": np.full(4, -0.10),
            "es5_loss_float": np.full(4, 0.02),
        }
        for product_id_str in product_id_list
    }
    monkeypatch.setattr(
        study,
        "bootstrap_metric_array_by_product_dict",
        lambda *args, **kwargs: metric_array_by_product_dict,
    )
    monkeypatch.setattr(
        study.ladder_runner,
        "centered_one_sided_p_value_float",
        lambda *args, **kwargs: 0.01,
    )
    captured_p_value_arr_list = []

    def fake_multipletests(p_value_arr, **kwargs):
        captured_p_value_arr_list.append(np.asarray(p_value_arr, dtype=float))
        return (
            np.zeros(len(p_value_arr), dtype=bool),
            np.asarray(p_value_arr, dtype=float),
            0.0,
            0.0,
        )

    monkeypatch.setattr(study, "multipletests", fake_multipletests)

    _, holm_df = study.calculate_bootstrap_evidence(
        return_df,
        headline_metric_df,
        spec_dict,
    )

    assert len(captured_p_value_arr_list) == 1
    assert np.allclose(captured_p_value_arr_list[0], 0.64)
    assert (holm_df["historical_trial_count_int"] == 60).all()
    assert (holm_df["cumulative_trial_family_count_int"] == 64).all()
