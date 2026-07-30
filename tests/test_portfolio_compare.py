"""Tests for the Bench book comparison.

The contract that matters: columns are recomputed on the window the selected
books share (never read off their own summaries), the benchmark is judged by
its series rather than its label, and a comparison that cannot be made fairly
says so instead of showing numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha.bench import catalog, portfolio_compare, portfolio_overview, runs


def _notice_title_set(result_obj) -> set[str]:
    return {notice_obj.title_str for notice_obj in result_obj.notice_list}


class _FakePortfolio:
    """Minimal stand-in carrying only what the comparison reads."""

    def __init__(self, total_value_ser, benchmark_value_ser=None, benchmark_label_str=None):
        self.results = pd.DataFrame({"total_value": total_value_ser})
        self.regression_benchmark_value_ser = benchmark_value_ser
        self.regression_benchmark_label_str = benchmark_label_str


def _equity_ser(start_str: str, day_count_int: int, daily_return_float: float) -> pd.Series:
    date_index = pd.bdate_range(start_str, periods=day_count_int)
    return pd.Series(
        100_000.0 * (1.0 + daily_return_float) ** np.arange(day_count_int),
        index=date_index,
    )


def _overview(rel_path_str: str, name_str: str, is_stale_bool: bool = False):
    run_obj = runs.RunEntry(
        run_name_str=name_str,
        analysis_dir_str="vanilla_backtest",
        analysis_label_str="Vanilla",
        timestamp_str="2026-07-30_120000",
        rel_dir_from_results_str=f"research/portfolio/{name_str}/vanilla_backtest/2026-07-30_120000",
        has_report_bool=True,
        activity_timestamp_float=1.0,
        summary_dict={"ann_return_pct": 10.0},
        metadata_dict={"common_start": "2019-01-01T00:00:00", "common_end": "2026-01-01T00:00:00"},
        run_info_dict={},
    )
    portfolio_entry_obj = catalog.PortfolioEntry(
        name_str=name_str,
        config_name_str=name_str,
        rel_path_str=rel_path_str,
        schema_str=catalog.SCHEMA_SIMPLE_STR,
        capital_float=100_000.0,
        rebalance_str=None,
        pod_tuple=(catalog.PortfolioPod(strategy_str="strategy_a", weight_float=1.0),),
        error_str=None,
    )
    return portfolio_overview.PortfolioOverview(
        portfolio=portfolio_entry_obj,
        latest_report_run=run_obj,
        latest_metric_run=run_obj,
        run_entry_list=[run_obj],
        stale_pod_list=[
            portfolio_overview.StalePod("strategy_a", "2026-07-30 10:00", "2026-07-30 12:00")
        ]
        if is_stale_bool
        else [],
    )


@pytest.fixture
def two_books(monkeypatch):
    """One long book and one that starts three years later."""
    long_ser = _equity_ser("2016-01-04", 2600, 0.0004)
    short_ser = _equity_ser("2019-01-02", 1800, 0.0006)
    portfolio_by_path_dict = {
        "portfolios/long.yaml": _FakePortfolio(long_ser),
        "portfolios/short.yaml": _FakePortfolio(short_ser),
    }
    overview_list = [_overview("portfolios/long.yaml", "Long"), _overview("portfolios/short.yaml", "Short")]

    monkeypatch.setattr(
        portfolio_overview, "list_portfolio_overviews", lambda: overview_list
    )
    monkeypatch.setattr(
        portfolio_compare,
        "_load_portfolio",
        lambda run_obj: portfolio_by_path_dict.get(f"portfolios/{run_obj.run_name_str.lower()}.yaml"),
    )
    return portfolio_by_path_dict, overview_list


def test_columns_are_measured_on_the_shared_window(two_books):
    """The whole point: a later-starting book is judged on the same days."""
    result_obj = portfolio_compare.compare_books(
        ["portfolios/long.yaml", "portfolios/short.yaml"]
    )
    assert result_obj.common_start_str == "2019-01-02"
    assert "Measured on the shared window" in _notice_title_set(result_obj)
    assert len(result_obj.column_list) == 2


def test_shared_window_figures_differ_from_each_book_own_history(two_books):
    """A book's own summary is not reused, so the numbers must be recomputed.

    The long book compounds at a fixed daily rate, so its CAGR is the same on
    any window; what proves recomputation is that the reported figure comes
    from the sliced curve rather than the summary_dict value of 10.0.
    """
    result_obj = portfolio_compare.compare_books(
        ["portfolios/long.yaml", "portfolios/short.yaml"]
    )
    long_column_obj = next(c for c in result_obj.column_list if c.label_str == "Long")
    cagr_float = long_column_obj.metric_by_name_dict["Return (Ann.) [%]"]
    assert cagr_float is not None
    assert cagr_float != pytest.approx(10.0)


def test_a_single_book_cannot_be_compared(two_books):
    result_obj = portfolio_compare.compare_books(["portfolios/long.yaml"])
    assert "Not enough books" in _notice_title_set(result_obj)
    assert not result_obj.has_columns_bool


def test_unreadable_books_are_named_and_skipped(two_books):
    result_obj = portfolio_compare.compare_books(
        ["portfolios/long.yaml", "portfolios/short.yaml", "portfolios/missing.yaml"]
    )
    assert "Books left out" in _notice_title_set(result_obj)
    assert len(result_obj.column_list) == 2


def test_books_with_no_overlap_are_refused(monkeypatch):
    early_ser = _equity_ser("2005-01-03", 500, 0.0004)
    late_ser = _equity_ser("2020-01-02", 500, 0.0004)
    overview_list = [_overview("portfolios/long.yaml", "Long"), _overview("portfolios/short.yaml", "Short")]
    portfolio_by_name_dict = {"Long": _FakePortfolio(early_ser), "Short": _FakePortfolio(late_ser)}
    monkeypatch.setattr(portfolio_overview, "list_portfolio_overviews", lambda: overview_list)
    monkeypatch.setattr(
        portfolio_compare, "_load_portfolio", lambda run_obj: portfolio_by_name_dict[run_obj.run_name_str]
    )

    result_obj = portfolio_compare.compare_books(
        ["portfolios/long.yaml", "portfolios/short.yaml"]
    )
    assert "No shared history" in _notice_title_set(result_obj)
    assert not result_obj.has_columns_bool


def test_same_benchmark_under_different_labels_is_one_column(monkeypatch):
    """The two runners spell the same yardstick differently.

    run_portfolio writes "$SPX"; PortfolioManager writes "$SPX · TOTALRETURN".
    Refusing to compare over that would drop the benchmark from most real
    comparisons, so agreement is judged on the series.
    """
    date_index = pd.bdate_range("2019-01-02", periods=900)
    benchmark_ser = pd.Series(3000.0 * (1.0003 ** np.arange(900)), index=date_index)
    overview_list = [_overview("portfolios/long.yaml", "Long"), _overview("portfolios/short.yaml", "Short")]
    portfolio_by_name_dict = {
        "Long": _FakePortfolio(_equity_ser("2019-01-02", 900, 0.0004), benchmark_ser, "$SPX"),
        "Short": _FakePortfolio(
            _equity_ser("2019-01-02", 900, 0.0005), benchmark_ser.copy(), "$SPX · TOTALRETURN"
        ),
    }
    monkeypatch.setattr(portfolio_overview, "list_portfolio_overviews", lambda: overview_list)
    monkeypatch.setattr(
        portfolio_compare, "_load_portfolio", lambda run_obj: portfolio_by_name_dict[run_obj.run_name_str]
    )

    result_obj = portfolio_compare.compare_books(
        ["portfolios/long.yaml", "portfolios/short.yaml"]
    )
    assert "Books disagree on benchmark" not in _notice_title_set(result_obj)
    assert result_obj.benchmark_label_str == "$SPX"
    assert any(column_obj.is_benchmark_bool for column_obj in result_obj.column_list)


def test_genuinely_different_benchmarks_drop_the_column(monkeypatch):
    date_index = pd.bdate_range("2019-01-02", periods=900)
    overview_list = [_overview("portfolios/long.yaml", "Long"), _overview("portfolios/short.yaml", "Short")]
    portfolio_by_name_dict = {
        "Long": _FakePortfolio(
            _equity_ser("2019-01-02", 900, 0.0004),
            pd.Series(3000.0 * (1.0003 ** np.arange(900)), index=date_index),
            "$SPX",
        ),
        "Short": _FakePortfolio(
            _equity_ser("2019-01-02", 900, 0.0005),
            pd.Series(200.0 * (1.0007 ** np.arange(900)), index=date_index),
            "$NDX",
        ),
    }
    monkeypatch.setattr(portfolio_overview, "list_portfolio_overviews", lambda: overview_list)
    monkeypatch.setattr(
        portfolio_compare, "_load_portfolio", lambda run_obj: portfolio_by_name_dict[run_obj.run_name_str]
    )

    result_obj = portfolio_compare.compare_books(
        ["portfolios/long.yaml", "portfolios/short.yaml"]
    )
    assert "Books disagree on benchmark" in _notice_title_set(result_obj)
    assert not any(column_obj.is_benchmark_bool for column_obj in result_obj.column_list)


def test_stale_books_are_flagged_in_the_comparison(monkeypatch):
    overview_list = [
        _overview("portfolios/long.yaml", "Long", is_stale_bool=True),
        _overview("portfolios/short.yaml", "Short"),
    ]
    portfolio_by_name_dict = {
        "Long": _FakePortfolio(_equity_ser("2019-01-02", 900, 0.0004)),
        "Short": _FakePortfolio(_equity_ser("2019-01-02", 900, 0.0005)),
    }
    monkeypatch.setattr(portfolio_overview, "list_portfolio_overviews", lambda: overview_list)
    monkeypatch.setattr(
        portfolio_compare, "_load_portfolio", lambda run_obj: portfolio_by_name_dict[run_obj.run_name_str]
    )

    result_obj = portfolio_compare.compare_books(
        ["portfolios/long.yaml", "portfolios/short.yaml"]
    )
    assert "Stale books in this comparison" in _notice_title_set(result_obj)


def test_near_identical_books_are_called_one_product(monkeypatch):
    date_index = pd.bdate_range("2019-01-02", periods=900)
    base_return_vec = np.random.default_rng(7).normal(0.0004, 0.008, 900)
    twin_return_vec = base_return_vec + np.random.default_rng(8).normal(0, 0.00002, 900)
    overview_list = [_overview("portfolios/long.yaml", "Long"), _overview("portfolios/short.yaml", "Short")]
    portfolio_by_name_dict = {
        "Long": _FakePortfolio(pd.Series(100_000 * np.cumprod(1 + base_return_vec), index=date_index)),
        "Short": _FakePortfolio(pd.Series(100_000 * np.cumprod(1 + twin_return_vec), index=date_index)),
    }
    monkeypatch.setattr(portfolio_overview, "list_portfolio_overviews", lambda: overview_list)
    monkeypatch.setattr(
        portfolio_compare, "_load_portfolio", lambda run_obj: portfolio_by_name_dict[run_obj.run_name_str]
    )

    result_obj = portfolio_compare.compare_books(
        ["portfolios/long.yaml", "portfolios/short.yaml"]
    )
    assert "Two books are one product" in _notice_title_set(result_obj)
    assert result_obj.correlation_row_list


def test_correlation_to_a_missing_benchmark_is_absent_not_one(two_books):
    """The metrics default that reported perfect correlation as a measured fact."""
    result_obj = portfolio_compare.compare_books(
        ["portfolios/long.yaml", "portfolios/short.yaml"]
    )
    for column_obj in result_obj.column_list:
        assert column_obj.metric_by_name_dict["Correlation"] is None
