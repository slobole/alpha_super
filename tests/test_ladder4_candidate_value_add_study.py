"""Unit tests for the frozen Ladder4 H0-H8 research runner."""

from __future__ import annotations

from pathlib import Path
import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

import scripts.research.run_ladder4_candidate_value_add_study as study_module
from alpha.engine.portfolio import Portfolio
from alpha.engine.strategy import Strategy
from scripts.research.run_ladder4_candidate_value_add_study import (
    ALL_HYPOTHESIS_ID_TUPLE,
    DEFAULT_SPEC_PATH,
    bootstrap_path_metric_array_dict,
    build_anchored_source_result_df,
    build_candidate_accounting_cost_gate_df,
    build_no_rebalance_hypothesis,
    calculate_crisis_metric_df,
    calculate_subperiod_metric_df,
    centered_one_sided_p_value_float,
    equal_observation_third_slice_tuple,
    expected_shortfall_loss_float,
    load_candidate_capacity_evidence_df,
    load_spec_dict,
    read_source_path_df,
    sha256_file_str,
    write_csv_gzip,
)


def source_path_df(total_value_list: list[float], date_list: list[str]) -> pd.DataFrame:
    date_index = pd.to_datetime(date_list)
    return pd.DataFrame(
        {
            "total_value_float": total_value_list,
            "portfolio_value_float": total_value_list,
            "cash_float": np.zeros(len(total_value_list)),
        },
        index=date_index,
    )


def test_frozen_spec_has_exact_paths_weights_and_matched_controls():
    spec_dict = load_spec_dict(DEFAULT_SPEC_PATH)

    assert len(spec_dict["source_runs"]) == 19
    assert tuple(spec_dict["hypotheses"]) == ALL_HYPOTHESIS_ID_TUPLE
    assert spec_dict["hypotheses"]["H1"]["matched_control_id_str"] == "H5"
    assert spec_dict["hypotheses"]["H2"]["matched_control_id_str"] == "H6"
    assert spec_dict["hypotheses"]["H3"]["matched_control_id_str"] == "H7"
    assert spec_dict["hypotheses"]["H4"]["matched_control_id_str"] == "H8"
    assert spec_dict["source_runs"]["bil_tfi_50000"][
        "run_variant_kwargs_dict"
    ]["accounting_profile_str"] == "tactical_fi_gross_dgs3mo_cash"
    assert spec_dict["source_runs"]["bil_core5_50000"][
        "run_variant_kwargs_dict"
    ]["accounting_profile_str"] == "core5_net25_zero_cash"
    assert spec_dict["portfolio_contract"][
        "capital_anchor_date_str"
    ] == "2012-09-28"
    assert spec_dict["portfolio_contract"][
        "effective_execution_start_date_str"
    ] == "2012-10-01"
    assert spec_dict["portfolio_contract"]["resume_policy_str"].startswith(
        "forbidden_fresh_output_only"
    )
    for hypothesis_dict in spec_dict["hypotheses"].values():
        assert sum(
            float(weight_float)
            for _, weight_float in hypothesis_dict["source_weight_list"]
        ) == pytest.approx(1.0, abs=1e-12)


def test_mutated_spec_is_rejected_without_explicit_test_override(tmp_path: Path):
    spec_dict = load_spec_dict(DEFAULT_SPEC_PATH)
    spec_dict["statistical_contract"]["simulation_count_int"] = 20
    mutated_spec_path = tmp_path / "mutated_spec.yaml"
    mutated_spec_path.write_text(
        yaml.safe_dump(spec_dict, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="approved frozen contract"):
        load_spec_dict(mutated_spec_path)


def test_negative_cash_fails_candidate_financing_gate():
    spec_dict = load_spec_dict(DEFAULT_SPEC_PATH)
    source_metadata_by_id_dict: dict[str, dict] = {}
    for candidate_id_str in ("H1", "H2", "H3", "H4"):
        candidate_source_id_str = spec_dict["hypotheses"][candidate_id_str][
            "source_weight_list"
        ][-1][0]
        control_id_str = spec_dict["hypotheses"][candidate_id_str][
            "matched_control_id_str"
        ]
        control_source_id_str = spec_dict["hypotheses"][control_id_str][
            "source_weight_list"
        ][-1][0]
        tactical_profile_bool = candidate_id_str in ("H1", "H2")
        accounting_profile_str = (
            "tactical_fi_gross_dgs3mo_cash"
            if tactical_profile_bool
            else "core5_net25_zero_cash"
        )
        metadata_dict = {
            "dividend_withholding_rate_float": 0.0 if tactical_profile_bool else 0.25,
            "positive_cash_rate_policy_str": (
                "causal_DGS3MO_ACT_365"
                if tactical_profile_bool
                else "zero_percent_intentional"
            ),
            "negative_cash_financing_policy_str": "not_modeled",
            "execution_adjustment_str": "CAPITALSPECIAL",
            "slippage_per_side_float": 0.0005 if tactical_profile_bool else 0.00025,
            "commission_per_share_float": 0.0 if tactical_profile_bool else 0.005,
            "commission_minimum_float": 0.0 if tactical_profile_bool else 1.0,
            "negative_cash_day_count_int": 0,
            "run_variant_kwargs_dict": {
                "accounting_profile_str": accounting_profile_str
            },
        }
        source_metadata_by_id_dict[candidate_source_id_str] = {
            **metadata_dict,
            "run_variant_kwargs_dict": {},
        }
        source_metadata_by_id_dict[control_source_id_str] = metadata_dict.copy()
    source_metadata_by_id_dict["tactical_fi_50000"][
        "negative_cash_day_count_int"
    ] = 1

    gate_df = build_candidate_accounting_cost_gate_df(
        spec_dict,
        source_metadata_by_id_dict,
    ).set_index("candidate_id_str")

    assert not bool(gate_df.loc["H1", "unmodeled_financing_gate_bool"])
    assert not bool(gate_df.loc["H1", "accounting_cost_financing_gate_bool"])
    assert bool(gate_df.loc["H2", "accounting_cost_financing_gate_bool"])


def test_no_rebalance_builder_matches_independent_sleeve_compounding():
    source_path_by_id_dict = {
        "A": source_path_df(
            [400.0, 440.0, 396.0],
            ["2024-01-02", "2024-01-03", "2024-01-04"],
        ),
        "B": source_path_df(
            [600.0, 600.0, 660.0],
            ["2024-01-02", "2024-01-03", "2024-01-04"],
        ),
    }

    hypothesis_path_df, sleeve_equity_df = build_no_rebalance_hypothesis(
        source_path_by_id_dict,
        [["A", 0.4], ["B", 0.6]],
        1_000.0,
    )

    assert sleeve_equity_df["A"].tolist() == pytest.approx([400.0, 440.0, 396.0])
    assert sleeve_equity_df["B"].tolist() == pytest.approx([600.0, 600.0, 660.0])
    assert hypothesis_path_df["total_value_float"].tolist() == pytest.approx(
        [1_000.0, 1_040.0, 1_056.0]
    )
    assert hypothesis_path_df["return_float"].iloc[0] == 0.0


def test_no_rebalance_builder_uses_exact_intersection_without_filling():
    source_path_by_id_dict = {
        "A": source_path_df(
            [500.0, 550.0, 495.0],
            ["2024-01-02", "2024-01-03", "2024-01-04"],
        ),
        "B": source_path_df(
            [500.0, 550.0],
            ["2024-01-02", "2024-01-04"],
        ),
    }

    hypothesis_path_df, _sleeve_equity_df = build_no_rebalance_hypothesis(
        source_path_by_id_dict,
        [["A", 0.5], ["B", 0.5]],
        1_000.0,
    )

    assert hypothesis_path_df.index.tolist() == pd.to_datetime(
        ["2024-01-02", "2024-01-04"]
    ).tolist()
    assert pd.Timestamp("2024-01-03") not in hypothesis_path_df.index


def test_no_rebalance_builder_rejects_a_predrifted_overlap_start():
    source_path_by_id_dict = {
        "early": source_path_df(
            [80.0, 100.0, 105.0, 102.9],
            ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        ),
        "late": source_path_df(
            [200.0, 220.0, 231.0],
            ["2024-01-02", "2024-01-03", "2024-01-04"],
        ),
        "latest": source_path_df(
            [50.0, 47.5],
            ["2024-01-03", "2024-01-04"],
        ),
    }
    frozen_global_idx = pd.to_datetime(["2024-01-03", "2024-01-04"])

    with pytest.raises(RuntimeError, match="frozen cash allocation"):
        build_no_rebalance_hypothesis(
            source_path_by_id_dict,
            [["early", 0.2], ["late", 0.3], ["latest", 0.5]],
            1_000.0,
            common_idx=frozen_global_idx,
        )


def test_cash_anchor_preserves_first_fill_cost_and_return():
    strategy_source_result_df = source_path_df(
        [970.0, 980.0],
        ["2012-10-01", "2012-10-02"],
    )
    anchored_source_result_df = build_anchored_source_result_df(
        strategy_source_result_df,
        strategy_name_str="candidate",
        allocated_capital_float=1_000.0,
        capital_anchor_date_str="2012-09-28",
        effective_execution_start_date_str="2012-10-01",
    )
    hypothesis_path_df, _sleeve_equity_df = build_no_rebalance_hypothesis(
        {"candidate": anchored_source_result_df},
        [["candidate", 1.0]],
        1_000.0,
    )

    assert anchored_source_result_df.index[0] == pd.Timestamp("2012-09-28")
    assert anchored_source_result_df.iloc[0].to_dict() == pytest.approx(
        {
            "total_value_float": 1_000.0,
            "portfolio_value_float": 0.0,
            "cash_float": 1_000.0,
        }
    )
    assert hypothesis_path_df.loc[
        pd.Timestamp("2012-10-01"), "return_float"
    ] == pytest.approx(-0.03)


def test_cash_anchor_rejects_a_late_first_engine_result():
    with pytest.raises(RuntimeError, match="first engine result"):
        build_anchored_source_result_df(
            source_path_df([1_000.0], ["2012-10-02"]),
            strategy_name_str="late_candidate",
            allocated_capital_float=1_000.0,
            capital_anchor_date_str="2012-09-28",
            effective_execution_start_date_str="2012-10-01",
        )


def test_execute_source_runs_passes_source_engine_start_and_records_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeSourceStrategy(Strategy):
        def iterate(
            self,
            data: pd.DataFrame,
            close: pd.DataFrame,
            open_prices: pd.Series,
        ):
            return None

    captured_engine_start_date_list: list[str] = []

    def fake_run_variant(
        *,
        show_display_bool: bool,
        save_results_bool: bool,
        output_dir_str: str,
        backtest_start_date_str: str,
        capital_base_float: float,
        end_date_str: str,
    ) -> Strategy:
        del show_display_bool, save_results_bool, output_dir_str
        captured_engine_start_date_list.append(backtest_start_date_str)
        strategy_obj = FakeSourceStrategy(
            name="fake_source",
            benchmarks=[],
            capital_base=capital_base_float,
        )
        result_date_idx = pd.to_datetime(["2012-10-01", end_date_str])
        strategy_obj.results = pd.DataFrame(
            {
                "total_value": [capital_base_float * 0.99, capital_base_float * 1.01],
                "portfolio_value": [capital_base_float * 0.50] * 2,
                "cash": [capital_base_float * 0.49, capital_base_float * 0.51],
            },
            index=result_date_idx,
        )
        return strategy_obj

    fake_module_path = tmp_path / "fake_sector_module.py"
    fake_module_path.write_text("# fake source module for runner regression\n", encoding="utf-8")
    fake_module_obj = SimpleNamespace(
        __file__=str(fake_module_path),
        run_variant=fake_run_variant,
    )
    original_import_module_fn = study_module.importlib.import_module
    monkeypatch.setattr(
        study_module.importlib,
        "import_module",
        lambda module_import_str: (
            fake_module_obj
            if module_import_str == "fake_sector_module"
            else original_import_module_fn(module_import_str)
        ),
    )
    spec_dict = {
        "portfolio_contract": {
            "requested_start_date_str": "2012-09-28",
            "capital_anchor_date_str": "2012-09-28",
            "effective_execution_start_date_str": "2012-10-01",
            "end_date_str": "2012-10-02",
        },
        "lineage_contract": {
            "native_history_request_start_by_strategy_import": {
                "fake_sector_module": "2000-01-01",
            }
        },
        "source_runs": {
            "fake_sector_source": {
                "strategy_import_str": "fake_sector_module",
                "allocated_capital_float": 1_000.0,
                "engine_request_start_date_str": "2012-09-28",
            }
        },
    }

    source_summary_df = study_module.execute_source_runs(
        spec_dict,
        tmp_path / "study_output",
    )

    assert captured_engine_start_date_list == ["2012-09-28"]
    assert source_summary_df.loc[
        0, "engine_request_start_date_str"
    ] == "2012-09-28"
    source_metadata_dict = json.loads(
        (tmp_path / "study_output" / "source_metadata" / "fake_sector_source.json")
        .read_text(encoding="utf-8")
    )
    assert source_metadata_dict["engine_request_start_date_str"] == "2012-09-28"
    persisted_source_df = study_module.read_source_path_df(
        tmp_path / "study_output" / "source_paths" / "fake_sector_source.csv.gz"
    )
    assert persisted_source_df.index.tolist() == pd.to_datetime(
        ["2012-09-28", "2012-10-01", "2012-10-02"]
    ).tolist()


def test_compact_builder_matches_real_portfolio_without_rebalance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source_path_by_id_dict = {
        "A": source_path_df(
            [400.0, 420.0, 399.0, 438.9],
            ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        ),
        "B": source_path_df(
            [600.0, 630.0, 693.0],
            ["2024-01-02", "2024-01-04", "2024-01-05"],
        ),
    }
    roundtrip_source_path_by_id_dict: dict[str, pd.DataFrame] = {}
    for source_id_str, source_df in source_path_by_id_dict.items():
        source_file_path = tmp_path / f"{source_id_str}.csv.gz"
        write_csv_gzip(
            source_df,
            source_file_path,
            index_bool=True,
            index_label_str="date",
        )
        roundtrip_source_path_by_id_dict[source_id_str] = read_source_path_df(
            source_file_path
        )

    compact_path_df, _sleeve_equity_df = build_no_rebalance_hypothesis(
        roundtrip_source_path_by_id_dict,
        [["A", 0.4], ["B", 0.6]],
        1_000.0,
    )
    monkeypatch.setattr(Portfolio, "_summarize", lambda _portfolio_obj: None)
    strategy_obj_list = []
    for source_id_str in ("A", "B"):
        engine_result_df = roundtrip_source_path_by_id_dict[source_id_str].rename(
            columns={
                "total_value_float": "total_value",
                "portfolio_value_float": "portfolio_value",
                "cash_float": "cash",
            }
        )
        strategy_obj_list.append(
            SimpleNamespace(
                name=source_id_str,
                results=engine_result_df,
                _capital_base=float(engine_result_df["total_value"].iloc[0]),
            )
        )
    portfolio_obj = Portfolio(
        strategy_obj_list,
        weights=[0.4, 0.6],
        capital_base=1_000.0,
        rebalance=None,
    )

    assert compact_path_df.index.equals(portfolio_obj.results.index)
    assert compact_path_df["total_value_float"].to_numpy() == pytest.approx(
        portfolio_obj.results["total_value"].to_numpy(),
        abs=1e-9,
    )


def test_no_rebalance_builder_rejects_nan_on_common_index():
    source_path_by_id_dict = {
        "A": source_path_df(
            [500.0, np.nan, 505.0],
            ["2024-01-02", "2024-01-03", "2024-01-04"],
        ),
        "B": source_path_df(
            [500.0, 502.5, 505.0],
            ["2024-01-02", "2024-01-03", "2024-01-04"],
        ),
    }

    with pytest.raises(RuntimeError, match="missing common-index return"):
        build_no_rebalance_hypothesis(
            source_path_by_id_dict,
            [["A", 0.5], ["B", 0.5]],
            1_000.0,
        )


def test_expected_shortfall_is_positive_loss_magnitude():
    return_arr = np.asarray([-0.10, -0.05, -0.01, 0.01, 0.02], dtype=float)
    assert expected_shortfall_loss_float(return_arr, quantile_float=0.20) == (
        pytest.approx(0.10)
    )


def test_centered_p_value_uses_finite_sample_correction():
    bootstrap_estimate_arr = np.asarray([0.8, 0.9, 1.0, 1.1], dtype=float)
    assert centered_one_sided_p_value_float(
        bootstrap_estimate_arr,
        observed_estimate_float=1.0,
    ) == pytest.approx(1.0 / 5.0)
    assert centered_one_sided_p_value_float(
        bootstrap_estimate_arr,
        observed_estimate_float=-0.1,
    ) == 1.0


def test_equal_observation_thirds_cover_every_position_once():
    third_slice_tuple = equal_observation_third_slice_tuple(10)
    covered_position_list: list[int] = []
    for third_slice_obj in third_slice_tuple:
        covered_position_list.extend(range(*third_slice_obj.indices(10)))
    assert covered_position_list == list(range(10))
    assert [len(range(*slice_obj.indices(10))) for slice_obj in third_slice_tuple] == [
        4,
        3,
        3,
    ]


def test_subperiod_metrics_cover_every_non_anchor_return_exactly_once():
    spec_dict = load_spec_dict(DEFAULT_SPEC_PATH)
    date_index = pd.bdate_range("2020-01-02", periods=11)
    total_value_df = pd.DataFrame(
        {
            hypothesis_id_str: 100.0 * np.cumprod(1.0 + np.arange(11) / 10_000.0)
            for hypothesis_id_str in ALL_HYPOTHESIS_ID_TUPLE
        },
        index=date_index,
    )
    benchmark_return_ser = pd.Series(0.0, index=date_index)

    subperiod_metric_df = calculate_subperiod_metric_df(
        total_value_df,
        benchmark_return_ser,
        spec_dict,
    )
    h0_metric_df = subperiod_metric_df.loc[
        subperiod_metric_df["hypothesis_id_str"] == "H0"
    ]

    assert h0_metric_df["observation_count_int"].astype(int).tolist() == [4, 3, 3]
    assert int(h0_metric_df["observation_count_int"].sum()) == len(date_index) - 1
    assert h0_metric_df["anchor_date_str"].tolist() == [
        date_index[0].date().isoformat(),
        date_index[4].date().isoformat(),
        date_index[7].date().isoformat(),
    ]


def test_one_session_crisis_keeps_return_into_first_crisis_date():
    spec_dict = copy.deepcopy(load_spec_dict(DEFAULT_SPEC_PATH))
    date_index = pd.bdate_range("2020-01-02", periods=4)
    total_value_df = pd.DataFrame(
        {
            hypothesis_id_str: [100.0, 90.0, 99.0, 100.0]
            for hypothesis_id_str in ALL_HYPOTHESIS_ID_TUPLE
        },
        index=date_index,
    )
    crisis_date_str = date_index[1].date().isoformat()
    spec_dict["crisis_diagnostic_list"] = [
        ["one_session", crisis_date_str, crisis_date_str]
    ]

    crisis_metric_df = calculate_crisis_metric_df(total_value_df, spec_dict)
    h0_metric_ser = crisis_metric_df.loc[
        crisis_metric_df["hypothesis_id_str"] == "H0"
    ].iloc[0]

    assert h0_metric_ser["observation_count_int"] == 1
    assert h0_metric_ser["cumulative_return_float"] == pytest.approx(-0.10)
    assert h0_metric_ser["anchor_date_str"] == date_index[0].date().isoformat()


def test_synchronized_bootstrap_gives_identical_metrics_for_identical_paths():
    date_index = pd.date_range("2020-01-01", periods=81, freq="B")
    base_return_ser = pd.Series(
        np.sin(np.arange(len(date_index)) / 7.0) * 0.002,
        index=date_index,
    )
    base_return_ser.iloc[0] = 0.0
    return_df = pd.DataFrame(
        {
            hypothesis_id_str: base_return_ser
            for hypothesis_id_str in ALL_HYPOTHESIS_ID_TUPLE
        }
    )

    metric_array_by_id_dict = bootstrap_path_metric_array_dict(
        return_df,
        simulation_count_int=40,
        mean_block_length_int=10,
        random_seed_int=123,
        annualization_day_int=252,
        es_quantile_float=0.05,
        chunk_size_int=13,
    )

    for hypothesis_id_str in ALL_HYPOTHESIS_ID_TUPLE[1:]:
        for metric_name_str in (
            "cagr_float",
            "sharpe_float",
            "max_drawdown_float",
            "es5_loss_float",
        ):
            assert metric_array_by_id_dict[hypothesis_id_str][
                metric_name_str
            ] == pytest.approx(metric_array_by_id_dict["H0"][metric_name_str])


def test_bootstrap_metrics_are_exactly_invariant_to_chunk_size():
    date_index = pd.date_range("2020-01-01", periods=61, freq="B")
    return_df = pd.DataFrame(
        {
            hypothesis_id_str: (
                np.cos(np.arange(len(date_index)) / 5.0) * 0.003
                + hypothesis_position_int * 0.00001
            )
            for hypothesis_position_int, hypothesis_id_str in enumerate(
                ALL_HYPOTHESIS_ID_TUPLE
            )
        },
        index=date_index,
    )
    return_df.iloc[0] = 0.0
    metric_by_id_chunk_7_dict = bootstrap_path_metric_array_dict(
        return_df,
        simulation_count_int=31,
        mean_block_length_int=10,
        random_seed_int=123,
        annualization_day_int=252,
        es_quantile_float=0.05,
        chunk_size_int=7,
    )
    metric_by_id_chunk_13_dict = bootstrap_path_metric_array_dict(
        return_df,
        simulation_count_int=31,
        mean_block_length_int=10,
        random_seed_int=123,
        annualization_day_int=252,
        es_quantile_float=0.05,
        chunk_size_int=13,
    )

    for hypothesis_id_str in ALL_HYPOTHESIS_ID_TUPLE:
        for metric_name_str in (
            "cagr_float",
            "sharpe_float",
            "max_drawdown_float",
            "es5_loss_float",
        ):
            np.testing.assert_array_equal(
                metric_by_id_chunk_7_dict[hypothesis_id_str][metric_name_str],
                metric_by_id_chunk_13_dict[hypothesis_id_str][metric_name_str],
            )


def test_gzip_writer_is_byte_deterministic(tmp_path: Path):
    data_df = pd.DataFrame({"value_float": [1.0, 2.0]})
    first_path = tmp_path / "first.csv.gz"
    second_path = tmp_path / "second.csv.gz"
    write_csv_gzip(data_df, first_path, index_bool=False)
    write_csv_gzip(data_df, second_path, index_bool=False)
    assert sha256_file_str(first_path) == sha256_file_str(second_path)


def test_external_capacity_artifact_cannot_clear_phase_1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    spec_dict = copy.deepcopy(load_spec_dict(DEFAULT_SPEC_PATH))
    monkeypatch.setattr(study_module, "REPO_ROOT_PATH", tmp_path)
    global_idx = pd.to_datetime(["2012-09-28", "2012-10-01"])
    source_metadata_by_id_dict: dict[str, dict] = {}

    for candidate_id_str in ("H1", "H2", "H3", "H4"):
        candidate_source_id_str = str(
            spec_dict["hypotheses"][candidate_id_str]["source_weight_list"][-1][0]
        )
        source_metadata_by_id_dict[candidate_source_id_str] = {
            "allocated_capital_float": float(
                spec_dict["source_runs"][candidate_source_id_str][
                    "allocated_capital_float"
                ]
            )
        }
        exact_relative_path_str = f"capacity/{candidate_id_str}/summary.json"
        spec_dict["capacity_contract"]["exact_artifact_path_by_candidate"][
            candidate_id_str
        ] = exact_relative_path_str
        study_module.write_json(
            tmp_path / exact_relative_path_str,
            {
                "recommended_capacity_float": 1_000_000_000.0,
                "self_attested_pass_bool": True,
            },
        )

    capacity_evidence_df = load_candidate_capacity_evidence_df(
        spec_dict,
        global_idx,
        global_index_sha256_str="global-index-sha",
        spec_sha256_str="spec-sha",
        source_metadata_by_id_dict=source_metadata_by_id_dict,
        norgate_database_vintage_dict={"database": "frozen"},
    )

    assert not capacity_evidence_df["capacity_gate_bool"].astype(bool).any()
    assert not capacity_evidence_df["exact_lineage_matched_bool"].astype(bool).any()
    assert capacity_evidence_df["reason_str"].eq(
        "separate_preregistered_capacity_phase_required"
    ).all()


def test_frozen_study_rejects_resume_to_prevent_mixed_vintages(tmp_path: Path):
    with pytest.raises(ValueError, match="forbid --resume"):
        study_module.run_study(
            DEFAULT_SPEC_PATH,
            tmp_path / "output",
            resume_bool=True,
        )


def test_full_artifact_pipeline_completes_on_synthetic_source_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    frozen_spec_dict = load_spec_dict(DEFAULT_SPEC_PATH)
    synthetic_spec_dict = copy.deepcopy(frozen_spec_dict)
    synthetic_spec_dict["statistical_contract"]["simulation_count_int"] = 20
    date_index = pd.bdate_range(end="2026-08-14", periods=300)
    synthetic_spec_dict["portfolio_contract"]["capital_anchor_date_str"] = (
        date_index[0].date().isoformat()
    )
    synthetic_spec_dict["portfolio_contract"][
        "effective_execution_start_date_str"
    ] = date_index[1].date().isoformat()
    synthetic_spec_path = tmp_path / "synthetic_spec.yaml"
    synthetic_spec_path.write_text(
        yaml.safe_dump(synthetic_spec_dict, sort_keys=False),
        encoding="utf-8",
    )
    output_dir_path = tmp_path / "artifacts"

    def fake_execute_source_runs(
        spec_dict: dict,
        output_dir_path: Path,
        *,
        resume_bool: bool = False,
    ) -> pd.DataFrame:
        assert resume_bool is False
        source_summary_row_list = []
        empty_transaction_df = pd.DataFrame(
            columns=[
                "source_id_str",
                "date",
                "asset_str",
                "amount_float",
                "fill_price_float",
                "signed_notional_float",
                "commission_float",
            ]
        )
        for source_position_int, (source_id_str, source_spec_dict) in enumerate(
            spec_dict["source_runs"].items(),
            start=1,
        ):
            allocated_capital_float = float(
                source_spec_dict["allocated_capital_float"]
            )
            daily_return_arr = (
                0.0003
                + 0.004 * np.sin(np.arange(len(date_index)) / 9.0)
                + source_position_int * 0.000002
            )
            daily_return_arr[0] = 0.0
            total_value_arr = allocated_capital_float * np.cumprod(
                1.0 + daily_return_arr
            )
            source_df = pd.DataFrame(
                {
                    "total_value_float": total_value_arr,
                    "portfolio_value_float": total_value_arr * 0.95,
                    "cash_float": total_value_arr * 0.05,
                },
                index=date_index,
            )
            source_df.loc[date_index[0], "total_value_float"] = (
                allocated_capital_float
            )
            source_df.loc[date_index[0], "portfolio_value_float"] = 0.0
            source_df.loc[date_index[0], "cash_float"] = allocated_capital_float
            source_df.index.name = "date"
            write_csv_gzip(
                source_df,
                output_dir_path / "source_paths" / f"{source_id_str}.csv.gz",
                index_bool=True,
                index_label_str="date",
            )
            write_csv_gzip(
                empty_transaction_df,
                output_dir_path
                / "source_transactions"
                / f"{source_id_str}.csv.gz",
                index_bool=False,
            )
            source_path = (
                output_dir_path / "source_paths" / f"{source_id_str}.csv.gz"
            )
            transaction_path = (
                output_dir_path
                / "source_transactions"
                / f"{source_id_str}.csv.gz"
            )
            tactical_profile_bool = source_id_str.startswith(
                ("tactical_fi_", "bil_tfi_")
            )
            module_path = Path(study_module.__file__).resolve()
            metadata_dict = {
                "source_id_str": source_id_str,
                "strategy_import_str": source_spec_dict["strategy_import_str"],
                "native_history_request_start_date_str": spec_dict[
                    "lineage_contract"
                ]["native_history_request_start_by_strategy_import"][
                    source_spec_dict["strategy_import_str"]
                ],
                "allocated_capital_float": allocated_capital_float,
                "run_variant_kwargs_dict": dict(
                    source_spec_dict.get("run_variant_kwargs_dict", {})
                ),
                "actual_start_date_str": date_index[0].date().isoformat(),
                "strategy_result_start_date_str": date_index[1].date().isoformat(),
                "capital_anchor_date_str": date_index[0].date().isoformat(),
                    "effective_execution_start_date_str": (
                        date_index[1].date().isoformat()
                    ),
                    "engine_request_start_date_str": str(
                        source_spec_dict.get(
                            "engine_request_start_date_str",
                            date_index[1].date().isoformat(),
                        )
                    ),
                "actual_end_date_str": date_index[-1].date().isoformat(),
                "observation_count_int": len(date_index),
                "transaction_count_int": 0,
                "negative_cash_day_count_int": 0,
                "minimum_cash_float": float(source_df["cash_float"].min()),
                "slippage_per_side_float": 0.0005 if tactical_profile_bool else 0.00025,
                "commission_per_share_float": 0.0 if tactical_profile_bool else 0.005,
                "commission_minimum_float": 0.0 if tactical_profile_bool else 1.0,
                "dividend_withholding_rate_float": (
                    0.0 if tactical_profile_bool else 0.25
                ),
                "positive_cash_rate_policy_str": (
                    "causal_DGS3MO_ACT_365"
                    if tactical_profile_bool
                    else "zero_percent_intentional"
                ),
                "negative_cash_financing_policy_str": "not_modeled",
                "execution_adjustment_str": "CAPITALSPECIAL",
                "module_path_str": str(module_path),
                "module_sha256_str": sha256_file_str(module_path),
                "shared_execution_dependency_hash_dict": (
                    study_module.shared_execution_dependency_hash_dict()
                ),
                "source_path_sha256_str": sha256_file_str(source_path),
                "transaction_path_sha256_str": sha256_file_str(transaction_path),
                "accounting_policy_dict": {},
                "data_adjustment_policy_dict": {},
            }
            metadata_path = (
                output_dir_path / "source_metadata" / f"{source_id_str}.json"
            )
            study_module.write_json(metadata_path, metadata_dict)
            source_summary_row_list.append(
                {
                    "source_id_str": source_id_str,
                    "actual_start_date_str": date_index[0].date().isoformat(),
                    "actual_end_date_str": date_index[-1].date().isoformat(),
                    "module_sha256_str": sha256_file_str(module_path),
                    "allocated_capital_float": allocated_capital_float,
                    "metadata_sha256_str": sha256_file_str(metadata_path),
                    "source_path_sha256_str": sha256_file_str(source_path),
                    "transaction_path_sha256_str": sha256_file_str(
                        transaction_path
                    ),
                }
            )
        source_summary_df = pd.DataFrame(source_summary_row_list)
        source_summary_df.to_csv(
            output_dir_path / "source_run_summary.csv",
            index=False,
        )
        return source_summary_df

    def fake_load_price_timeseries(*_args, **_kwargs) -> pd.DataFrame:
        call_order_list.append("benchmark_read")
        return pd.DataFrame(
            {"Close": 100.0 * np.cumprod(1.0 + np.full(len(date_index), 0.0002))},
            index=date_index,
        )

    call_order_list: list[str] = []

    def fake_norgate_database_vintage_dict() -> dict:
        call_order_list.append("norgate_fingerprint")
        return {
            "norgatedata_package_version_str": "synthetic",
            "database_last_update_by_name_dict": {"synthetic": "frozen"},
        }

    monkeypatch.setattr(study_module, "execute_source_runs", fake_execute_source_runs)
    monkeypatch.setattr(
        study_module,
        "norgate_database_vintage_dict",
        fake_norgate_database_vintage_dict,
    )
    monkeypatch.setattr(
        study_module,
        "load_price_timeseries",
        fake_load_price_timeseries,
    )

    report_path = study_module.run_study(
        synthetic_spec_path,
        output_dir_path,
        enforce_frozen_contract_bool=False,
    )

    assert report_path.is_file()
    assert call_order_list == [
        "norgate_fingerprint",
        "norgate_fingerprint",
        "benchmark_read",
        "norgate_fingerprint",
    ]
    assert (output_dir_path / "run_manifest.json").is_file()
    assert (output_dir_path / "promotion_gates.csv").is_file()
    assert (output_dir_path / "charts" / "equity_and_drawdown.png").is_file()
    promotion_gate_df = pd.read_csv(output_dir_path / "promotion_gates.csv")
    assert promotion_gate_df["candidate_id_str"].tolist() == ["H1", "H2", "H3", "H4"]
    manifest_dict = study_module.json.loads(
        (output_dir_path / "run_manifest.json").read_text(encoding="utf-8")
    )
    for artifact_dict in manifest_dict["artifact_row_list"]:
        artifact_path = output_dir_path / artifact_dict["relative_path_str"]
        assert artifact_path.stat().st_size == artifact_dict["size_byte_int"]
        assert sha256_file_str(artifact_path) == artifact_dict["sha256_str"]
