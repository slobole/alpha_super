"""Tests for the Bench portfolio overview.

The contract that matters: metrics are lifted from the run the runner recorded
(never recomputed), MAR is derived only when both of its inputs exist, and the
staleness flag fires when — and only when — a pod has genuinely been re-run
since the book consumed it.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from alpha.bench import catalog, portfolio_overview, runs


def _portfolio_run(
    pod_payload_list: list[dict] | None = None,
    summary_dict: dict | None = None,
    timestamp_str: str = "2026-07-30_120000",
) -> runs.RunEntry:
    return runs.RunEntry(
        run_name_str="TestBook",
        analysis_dir_str="vanilla_backtest",
        analysis_label_str="Vanilla",
        timestamp_str=timestamp_str,
        rel_dir_from_results_str=f"research/portfolio/TestBook/vanilla_backtest/{timestamp_str}",
        has_report_bool=True,
        activity_timestamp_float=1.0,
        summary_dict=summary_dict
        if summary_dict is not None
        else {"ann_return_pct": 12.0, "sharpe": 1.4, "max_drawdown_pct": -9.0},
        metadata_dict={
            "pods": pod_payload_list if pod_payload_list is not None else [],
            "common_start": "2012-10-01T00:00:00",
            "common_end": "2026-07-29T00:00:00",
        },
        run_info_dict={},
    )


def _pod_payload(strategy_name_str: str, saved_at_str: str) -> dict:
    return {
        "strategy_name": strategy_name_str,
        "weight": 0.5,
        "result_metadata": {"saved_at": saved_at_str},
    }


def _overview(
    latest_metric_run_obj: runs.RunEntry | None,
    stale_pod_list: list | None = None,
) -> portfolio_overview.PortfolioOverview:
    portfolio_entry_obj = catalog.PortfolioEntry(
        name_str="test_book",
        config_name_str="TestBook",
        rel_path_str="portfolios/test_book.yaml",
        schema_str=catalog.SCHEMA_SIMPLE_STR,
        capital_float=100_000.0,
        rebalance_str=None,
        pod_tuple=(catalog.PortfolioPod(strategy_str="strategy_a", weight_float=1.0),),
        error_str=None,
    )
    return portfolio_overview.PortfolioOverview(
        portfolio=portfolio_entry_obj,
        latest_report_run=latest_metric_run_obj,
        latest_metric_run=latest_metric_run_obj,
        run_entry_list=[latest_metric_run_obj] if latest_metric_run_obj else [],
        stale_pod_list=stale_pod_list or [],
    )


def test_metrics_are_lifted_from_the_recorded_run():
    overview_obj = _overview(_portfolio_run())
    assert overview_obj.ann_return_float == 12.0
    assert overview_obj.sharpe_float == 1.4
    assert overview_obj.max_drawdown_float == -9.0
    assert overview_obj.has_run_bool


def test_mar_is_return_over_the_worst_loss():
    overview_obj = _overview(_portfolio_run())
    assert overview_obj.mar_float == pytest.approx(12.0 / 9.0)


@pytest.mark.parametrize(
    "summary_dict",
    [
        {"ann_return_pct": 12.0},  # no drawdown recorded
        {"max_drawdown_pct": -9.0},  # no return recorded
        {"ann_return_pct": 12.0, "max_drawdown_pct": 0.0},  # a book that never lost
    ],
)
def test_mar_is_none_when_an_input_is_missing_or_zero(summary_dict):
    """Absence must stay distinct from zero so sorting can sink it."""
    overview_obj = _overview(_portfolio_run(summary_dict=summary_dict))
    assert overview_obj.mar_float is None


def test_a_book_with_no_run_reports_nothing_rather_than_zero():
    overview_obj = _overview(None)
    assert overview_obj.ann_return_float is None
    assert overview_obj.mar_float is None
    assert overview_obj.window_str is None
    assert not overview_obj.has_run_bool
    assert not overview_obj.is_stale_bool


def test_window_comes_from_the_common_overlap_the_runner_recorded():
    overview_obj = _overview(_portfolio_run())
    assert overview_obj.window_str == "2012-10-01 → 2026-07-29"
    assert overview_obj.window_start_str == "2012-10-01"


def test_stale_when_a_pod_was_re_run_after_the_book_consumed_it():
    """The failure this flag exists for: the book keeps quoting a superseded pod."""
    run_obj = _portfolio_run([_pod_payload("strategy_a", "2026-07-30T10:51:49")])
    stale_pod_list = portfolio_overview._stale_pod_list(
        run_obj, {"strategy_a": datetime(2026, 7, 30, 12, 33, 18)}
    )
    assert len(stale_pod_list) == 1
    assert stale_pod_list[0].strategy_name_str == "strategy_a"


def test_not_stale_when_the_book_used_the_newest_pod_run():
    run_obj = _portfolio_run([_pod_payload("strategy_a", "2026-07-30T12:33:18")])
    assert portfolio_overview._stale_pod_list(
        run_obj, {"strategy_a": datetime(2026, 7, 30, 12, 33, 18)}
    ) == []


def test_sub_second_disagreement_is_not_staleness():
    """saved_at and the run folder name are written by different code paths."""
    run_obj = _portfolio_run([_pod_payload("strategy_a", "2026-07-30T12:33:18")])
    assert portfolio_overview._stale_pod_list(
        run_obj, {"strategy_a": datetime(2026, 7, 30, 12, 33, 19)}
    ) == []


def test_pod_with_no_recorded_provenance_is_not_guessed_stale():
    run_obj = _portfolio_run([{"strategy_name": "strategy_a", "weight": 1.0}])
    assert portfolio_overview._stale_pod_list(
        run_obj, {"strategy_a": datetime(2026, 7, 30, 12, 33, 18)}
    ) == []


def test_pod_the_results_tree_no_longer_knows_is_not_stale():
    run_obj = _portfolio_run([_pod_payload("strategy_gone", "2026-07-30T10:00:00")])
    assert portfolio_overview._stale_pod_list(run_obj, {}) == []


def test_a_run_without_pod_metadata_reports_no_staleness():
    assert portfolio_overview._stale_pod_list(_portfolio_run(), {}) == []
    assert portfolio_overview._stale_pod_list(None, {}) == []


def test_staleness_reads_only_the_pods_a_book_references():
    """Targeted directory reads, not a walk of the whole strategy tree."""
    run_obj = _portfolio_run(
        [
            _pod_payload("strategy_a", "2026-07-30T10:00:00"),
            _pod_payload("strategy_b", "2026-07-30T10:00:00"),
            {"weight": 0.1},  # malformed entry must not become a lookup
        ]
    )
    assert portfolio_overview._referenced_strategy_name_set(run_obj) == {
        "strategy_a",
        "strategy_b",
    }
    assert portfolio_overview._referenced_strategy_name_set(None) == set()


def test_search_text_covers_the_book_and_its_pods():
    overview_obj = _overview(_portfolio_run())
    assert "strategy_a" in overview_obj.search_text_str
    assert "testbook" in overview_obj.search_text_str
    assert overview_obj.search_text_str == overview_obj.search_text_str.lower()


def test_overview_list_reads_the_real_tree_without_recomputing_anything():
    """Against real artifacts: every reported metric must exist in its summary."""
    overview_list = portfolio_overview.list_portfolio_overviews()
    assert overview_list
    for overview_obj in overview_list:
        if overview_obj.latest_metric_run is None:
            assert overview_obj.ann_return_float is None
            continue
        summary_dict = overview_obj.latest_metric_run.summary_dict
        assert overview_obj.ann_return_float == summary_dict["ann_return_pct"]
