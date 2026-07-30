"""Tests for the Bench portfolio builder.

The builder writes a config the runners consume, so the contract that matters
is: the YAML it renders parses back into the pods and weights that were asked
for, its checks fire on the input mistakes they exist to catch, and it cannot
be talked into writing outside ``portfolios/``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from alpha.bench import portfolio_builder
from alpha.strategy_registry import TIER_LABEL_DICT, MaturityTier


def _notice_title_set(diagnostics_obj) -> set[str]:
    return {notice_obj.title_str for notice_obj in diagnostics_obj.notice_list}


def _stub_candidate(
    stem_str: str,
    is_wired_bool: bool = True,
    benchmark_symbol_str: str | None = "$SPX",
    tier_obj: MaturityTier | None = None,
):
    run_obj = portfolio_builder.runs.RunEntry(
        run_name_str=stem_str,
        analysis_dir_str="vanilla_backtest",
        analysis_label_str="Vanilla",
        timestamp_str="2026-07-30_120000",
        rel_dir_from_results_str=f"research/strategy/{stem_str}/vanilla_backtest/2026-07-30_120000",
        has_report_bool=True,
        activity_timestamp_float=1.0,
        summary_dict={"ann_return_pct": 12.0, "sharpe": 1.1, "max_drawdown_pct": -9.0},
        metadata_dict={"benchmarks": [benchmark_symbol_str] if benchmark_symbol_str else []},
        run_info_dict={"parameters": {"start_date": "2012-10-01", "end_date": "2026-07-24"}},
    )
    resolved_tier_obj = tier_obj or (
        MaturityTier.WIRED if is_wired_bool else MaturityTier.RESEARCH
    )
    return portfolio_builder.PodCandidate(
        stem_str=stem_str,
        display_name_str=stem_str.replace("strategy_", "").title(),
        module_import_str=f"strategies.x.{stem_str}",
        category_label_str="Test",
        tier_int=int(resolved_tier_obj),
        tier_label_str=TIER_LABEL_DICT[resolved_tier_obj],
        run_obj=run_obj,
        benchmark_symbol_str=benchmark_symbol_str,
    )


@pytest.fixture
def stub_pods(monkeypatch):
    """Two uncorrelated pods with six years of daily returns each."""
    date_index = pd.bdate_range("2019-01-01", "2025-12-31")
    return_ser_by_stem_dict = {
        "strategy_alpha": pd.Series(
            [0.001 if index_int % 2 == 0 else -0.0005 for index_int in range(len(date_index))],
            index=date_index,
        ),
        "strategy_beta": pd.Series(
            [0.0008 if index_int % 3 == 0 else -0.0003 for index_int in range(len(date_index))],
            index=date_index,
        ),
    }
    candidate_list = [_stub_candidate("strategy_alpha"), _stub_candidate("strategy_beta")]

    def fake_candidates():
        return candidate_list

    def fake_return_ser(run_obj):
        return return_ser_by_stem_dict.get(run_obj.run_name_str)

    monkeypatch.setattr(portfolio_builder, "list_pod_candidates", fake_candidates)
    monkeypatch.setattr(portfolio_builder, "_pod_return_ser", fake_return_ser)
    monkeypatch.setattr(portfolio_builder, "_trading_days_per_month_float", lambda run_obj: 1.0)
    return return_ser_by_stem_dict, candidate_list


def test_rendered_yaml_round_trips_into_the_requested_pods_and_weights(stub_pods):
    diagnostics_obj = portfolio_builder.analyze_selection(
        selection_pair_list=[("strategy_alpha", 0.6), ("strategy_beta", 0.4)],
        name_str="RoundTrip",
        capital_float=100_000.0,
    )

    config_dict = yaml.safe_load(diagnostics_obj.yaml_text_str)
    assert config_dict["name"] == "RoundTrip"
    assert config_dict["capital"] == 100_000
    assert [pod_dict["strategy"] for pod_dict in config_dict["pods"]] == [
        "strategy_alpha",
        "strategy_beta",
    ]
    assert [pod_dict["weight"] for pod_dict in config_dict["pods"]] == [0.6, 0.4]


def test_weights_are_normalized_and_the_rescaling_is_reported(stub_pods):
    diagnostics_obj = portfolio_builder.analyze_selection(
        selection_pair_list=[("strategy_alpha", 2.0), ("strategy_beta", 2.0)],
        name_str="Unnormalized",
        capital_float=100_000.0,
    )

    config_dict = yaml.safe_load(diagnostics_obj.yaml_text_str)
    assert sum(pod_dict["weight"] for pod_dict in config_dict["pods"]) == pytest.approx(1.0)
    assert "Weights normalized" in _notice_title_set(diagnostics_obj)


@pytest.mark.parametrize("pod_count_int", [2, 3, 6, 7])
def test_equal_weights_written_sum_to_exactly_one(monkeypatch, pod_count_int):
    """A third rendered three times must not sum to 0.999999.

    run_portfolio rejects a config whose weights miss 1.0 by more than 1e-6, so
    a book the builder wrote must always load.
    """
    stem_list = [f"strategy_pod{index_int}" for index_int in range(pod_count_int)]
    candidate_list = [_stub_candidate(stem_str) for stem_str in stem_list]
    monkeypatch.setattr(portfolio_builder, "list_pod_candidates", lambda: candidate_list)
    monkeypatch.setattr(portfolio_builder, "_pod_return_ser", lambda run_obj: None)

    equal_float = round(1.0 / pod_count_int, 4)
    diagnostics_obj = portfolio_builder.analyze_selection(
        selection_pair_list=[(stem_str, equal_float) for stem_str in stem_list],
        name_str="EqualWeights",
        capital_float=100_000.0,
    )

    config_dict = yaml.safe_load(diagnostics_obj.yaml_text_str)
    written_weight_list = [pod_dict["weight"] for pod_dict in config_dict["pods"]]
    assert abs(sum(written_weight_list) - 1.0) <= 1e-6
    assert len(written_weight_list) == pod_count_int


def test_redundant_pods_are_flagged_as_the_same_trade(stub_pods, monkeypatch):
    return_ser_by_stem_dict, _candidate_list = stub_pods
    # Beta becomes a near-copy of alpha: the pair the check exists to catch.
    return_ser_by_stem_dict["strategy_beta"] = (
        return_ser_by_stem_dict["strategy_alpha"] * 1.01
    )

    diagnostics_obj = portfolio_builder.analyze_selection(
        selection_pair_list=[("strategy_alpha", 0.5), ("strategy_beta", 0.5)],
        name_str="Redundant",
        capital_float=100_000.0,
    )
    assert "Pods are the same trade" in _notice_title_set(diagnostics_obj)


def test_distinct_pods_are_not_flagged_as_redundant(stub_pods):
    diagnostics_obj = portfolio_builder.analyze_selection(
        selection_pair_list=[("strategy_alpha", 0.5), ("strategy_beta", 0.5)],
        name_str="Distinct",
        capital_float=100_000.0,
    )
    assert "Pods are the same trade" not in _notice_title_set(diagnostics_obj)
    assert diagnostics_obj.correlation_row_list


def test_mixed_benchmarks_are_resolved_explicitly_and_reported(monkeypatch, stub_pods):
    candidate_list = [
        _stub_candidate("strategy_alpha", benchmark_symbol_str="$SPX"),
        _stub_candidate("strategy_beta", benchmark_symbol_str="SPY"),
    ]
    monkeypatch.setattr(portfolio_builder, "list_pod_candidates", lambda: candidate_list)

    diagnostics_obj = portfolio_builder.analyze_selection(
        selection_pair_list=[("strategy_alpha", 0.5), ("strategy_beta", 0.5)],
        name_str="MixedBench",
        capital_float=100_000.0,
    )

    assert "Pods disagree on benchmark" in _notice_title_set(diagnostics_obj)
    # Written explicitly: without it the runner refuses to pick and the report
    # loses every benchmark-relative section.
    assert diagnostics_obj.resolved_benchmark_str in {"$SPX", "SPY"}
    config_dict = yaml.safe_load(diagnostics_obj.yaml_text_str)
    assert config_dict["benchmark"] == diagnostics_obj.resolved_benchmark_str


def test_matching_benchmarks_need_no_explicit_key(stub_pods):
    diagnostics_obj = portfolio_builder.analyze_selection(
        selection_pair_list=[("strategy_alpha", 0.5), ("strategy_beta", 0.5)],
        name_str="SameBench",
        capital_float=100_000.0,
    )
    assert "Pods disagree on benchmark" not in _notice_title_set(diagnostics_obj)


def test_underfunded_sleeve_is_flagged(stub_pods, monkeypatch):
    monkeypatch.setattr(
        portfolio_builder,
        "_pod_minimum_capital_dict",
        lambda: {"strategy_alpha": 25_000.0},
    )
    diagnostics_obj = portfolio_builder.analyze_selection(
        selection_pair_list=[("strategy_alpha", 0.5), ("strategy_beta", 0.5)],
        name_str="Underfunded",
        capital_float=20_000.0,
    )
    assert "Sleeve underfunded" in _notice_title_set(diagnostics_obj)


def test_research_pods_are_named_not_blocked(monkeypatch, stub_pods):
    candidate_list = [
        _stub_candidate("strategy_alpha", is_wired_bool=True),
        _stub_candidate("strategy_beta", is_wired_bool=False),
    ]
    monkeypatch.setattr(portfolio_builder, "list_pod_candidates", lambda: candidate_list)

    diagnostics_obj = portfolio_builder.analyze_selection(
        selection_pair_list=[("strategy_alpha", 0.5), ("strategy_beta", 0.5)],
        name_str="WithResearch",
        capital_float=100_000.0,
    )
    assert "Contains research pods" in _notice_title_set(diagnostics_obj)
    assert not diagnostics_obj.has_block_bool


def test_single_pod_selection_is_blocked(stub_pods):
    diagnostics_obj = portfolio_builder.analyze_selection(
        selection_pair_list=[("strategy_alpha", 1.0)],
        name_str="Lonely",
        capital_float=100_000.0,
    )
    assert diagnostics_obj.has_block_bool


def test_unknown_strategy_is_blocked(stub_pods):
    diagnostics_obj = portfolio_builder.analyze_selection(
        selection_pair_list=[("strategy_alpha", 0.5), ("strategy_missing", 0.5)],
        name_str="Unknown",
        capital_float=100_000.0,
    )
    assert diagnostics_obj.has_block_bool
    assert "Unknown strategy" in _notice_title_set(diagnostics_obj)


def test_non_overlapping_pods_are_blocked(stub_pods, monkeypatch):
    return_ser_by_stem_dict, _candidate_list = stub_pods
    return_ser_by_stem_dict["strategy_beta"] = pd.Series(
        [0.001] * 300, index=pd.bdate_range("2000-01-03", periods=300)
    )
    diagnostics_obj = portfolio_builder.analyze_selection(
        selection_pair_list=[("strategy_alpha", 0.5), ("strategy_beta", 0.5)],
        name_str="NoOverlap",
        capital_float=100_000.0,
    )
    assert diagnostics_obj.has_block_bool
    assert "No overlapping history" in _notice_title_set(diagnostics_obj)


@pytest.mark.parametrize(
    "filename_str",
    ["../escape.yaml", "sub/dir.yaml", "bad name.yaml", "", "..\\escape.yaml"],
)
def test_write_path_refuses_anything_outside_portfolios(filename_str):
    with pytest.raises(ValueError):
        portfolio_builder.resolve_write_path(filename_str)


def test_write_path_accepts_a_plain_name_and_adds_the_suffix():
    write_path = portfolio_builder.resolve_write_path("my_book")
    assert write_path.name == "my_book.yaml"
    assert write_path.parent == portfolio_builder.PORTFOLIOS_ROOT_PATH.resolve()


def test_write_refuses_to_clobber_without_overwrite(monkeypatch, tmp_path):
    monkeypatch.setattr(portfolio_builder, "PORTFOLIOS_ROOT_PATH", tmp_path)
    portfolio_builder.write_portfolio_yaml("book.yaml", "name: A\n")
    with pytest.raises(FileExistsError):
        portfolio_builder.write_portfolio_yaml("book.yaml", "name: B\n")

    portfolio_builder.write_portfolio_yaml("book.yaml", "name: B\n", overwrite_bool=True)
    assert (tmp_path / "book.yaml").read_text(encoding="utf-8") == "name: B\n"


@pytest.mark.parametrize(
    ("name_str", "expected_str"),
    [
        ("MonthlyDefensive", "monthly_defensive"),
        ("Monthly Defensive", "monthly_defensive"),
        ("my-book 2", "my_book_2"),
        ("!!!", "portfolio"),
    ],
)
def test_slugify_filename(name_str, expected_str):
    assert portfolio_builder.slugify_filename_str(name_str) == expected_str


def test_rendered_yaml_matches_the_house_style(stub_pods):
    """The config must read like portfolios/multipod.yaml, blank lines and all."""
    diagnostics_obj = portfolio_builder.analyze_selection(
        selection_pair_list=[("strategy_alpha", 0.5), ("strategy_beta", 0.5)],
        name_str="StyleCheck",
        capital_float=100_000.0,
    )
    assert diagnostics_obj.yaml_text_str.startswith("name: StyleCheck\ncapital: 100000\n")
    assert "\npods:\n  - strategy: strategy_alpha\n    weight: 0.5\n" in (
        diagnostics_obj.yaml_text_str
    )
    assert diagnostics_obj.yaml_text_str.endswith("\n")


def test_candidate_exposes_sort_and_search_keys():
    """The catalog's sort and search run client-side off these fields."""
    candidate_obj = _stub_candidate("strategy_alpha")
    # 2012-10-01 -> 2026-07-24 is ~13.8 years.
    assert candidate_obj.window_year_float == pytest.approx(13.8, abs=0.2)
    assert "strategy_alpha" in candidate_obj.search_text_str
    assert candidate_obj.search_text_str == candidate_obj.search_text_str.lower()


def test_trades_per_year_separates_a_monthly_rotation_from_a_daily_book():
    """The free activity proxy: exact cadence costs a CSV read per strategy."""
    monthly_candidate_obj = _stub_candidate("strategy_monthly")
    monthly_candidate_obj.run_obj.summary_dict["trade_count"] = 867
    daily_candidate_obj = _stub_candidate("strategy_daily")
    daily_candidate_obj.run_obj.summary_dict["trade_count"] = 10613

    assert monthly_candidate_obj.trades_per_year_float < 100
    assert daily_candidate_obj.trades_per_year_float > 500


def test_trades_per_year_is_none_without_a_window_or_trades():
    """Absence of a figure must stay distinct from zero so sorting can sink it."""
    candidate_obj = _stub_candidate("strategy_alpha")
    candidate_obj.run_obj.summary_dict.pop("trade_count", None)
    assert candidate_obj.trades_per_year_float is None

    windowless_candidate_obj = _stub_candidate("strategy_beta")
    windowless_candidate_obj.run_obj.run_info_dict["parameters"] = {}
    assert windowless_candidate_obj.window_year_float is None
    assert windowless_candidate_obj.trades_per_year_float is None


def test_candidate_list_only_offers_strategies_with_a_saved_vanilla_run():
    """Against the real results tree: every candidate must have a pickle to combine."""
    for candidate_obj in portfolio_builder.list_pod_candidates()[:12]:
        assert candidate_obj.run_obj.analysis_dir_str == "vanilla_backtest"
        assert Path(
            portfolio_builder.RESULTS_ROOT_PATH / candidate_obj.run_obj.rel_dir_from_results_str
        ).exists()
