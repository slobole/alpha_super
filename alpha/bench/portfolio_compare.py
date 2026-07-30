"""Compare books over the one window on which they are comparable.

Books are measured over whatever range their pods happened to share, and those
ranges differ: a book starting 2019-04 is judged over a stretch in which the
passive index itself compounded at 15.4%, while one starting 2012-10 is not.
Standing their headline returns side by side is not a comparison, it is a
comparison of market regimes.

So this module does not read the numbers off each book's summary. It slices
every selected book's stored equity curve to the window they all share and
recomputes there — through ``alpha.engine.metrics.generate_overall_metrics``,
the same function that wrote the originals. Bench still adds no quant logic:
the engine's metric definitions are reused verbatim, only the date range
changes, and the page says plainly that it did.

The benchmark is measured over that same window and shown as one more column,
because "how did this do against the index" is the question a book exists to
answer and the index also looks different depending on when you start.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from alpha.bench import portfolio_overview, runs
from alpha.engine.metrics import cross_correlation_matrix, generate_overall_metrics


COMPARE_MIN_INT = 2
COMPARE_MAX_INT = 5

# Read off the frame generate_overall_metrics returns; the labels are its own.
METRIC_ROW_TUPLE: tuple[tuple[str, str, str], ...] = (
    ("Return (Ann.) [%]", "CAGR", "pct"),
    ("Volatility (Ann.) [%]", "Volatility", "pct"),
    ("Sharpe Ratio", "Sharpe", "num"),
    ("Max. Drawdown [%]", "Max drawdown", "pct"),
    ("MAR Ratio", "MAR", "num"),
    ("Correlation", "Correlation to benchmark", "num"),
)

# Two products this alike are one product sold twice.
DUPLICATE_CORRELATION_FLOAT = 0.95


@dataclass(frozen=True)
class CompareNotice:
    severity_str: str
    title_str: str
    detail_str: str


@dataclass
class BookColumn:
    label_str: str
    rel_path_str: str
    metric_by_name_dict: dict[str, float | None] = field(default_factory=dict)
    pod_label_list: list[str] = field(default_factory=list)
    full_window_str: str | None = None
    is_stale_bool: bool = False
    is_benchmark_bool: bool = False


@dataclass
class ComparisonResult:
    column_list: list[BookColumn] = field(default_factory=list)
    notice_list: list[CompareNotice] = field(default_factory=list)
    correlation_label_list: list[str] = field(default_factory=list)
    correlation_row_list: list[dict] = field(default_factory=list)
    common_start_str: str | None = None
    common_end_str: str | None = None
    common_year_float: float | None = None
    benchmark_label_str: str | None = None

    @property
    def has_columns_bool(self) -> bool:
        return len(self.column_list) > 0


def _load_portfolio(run_obj: runs.RunEntry):
    """Load one saved book. Returns None when its pickle cannot be read."""
    pickle_path = (
        runs.RESULTS_ROOT_PATH
        / run_obj.rel_dir_from_results_str
        / f"{run_obj.run_name_str}.pkl"
    )
    if not Path(pickle_path).exists():
        return None
    try:
        from alpha.engine.portfolio import Portfolio

        return Portfolio.read_pickle(str(pickle_path))
    except Exception:
        return None


def _metric_dict(
    total_value_ser: pd.Series,
    benchmark_return_ser: pd.Series | None,
) -> dict[str, float | None]:
    """Engine metrics for one equity curve over the window it is sliced to."""
    metric_ser = generate_overall_metrics(
        total_value_ser.astype(float),
        series_to_correlate=benchmark_return_ser,
        capital_base=float(total_value_ser.iloc[0]),
    )
    metric_dict: dict[str, float | None] = {}
    for metric_name_str, _label_str, _format_str in METRIC_ROW_TUPLE:
        value_obj = metric_ser.get(metric_name_str)
        try:
            metric_dict[metric_name_str] = (
                None if value_obj is None or pd.isna(value_obj) else float(value_obj)
            )
        except (TypeError, ValueError):
            metric_dict[metric_name_str] = None
    return metric_dict


def compare_books(rel_path_list: list[str]) -> ComparisonResult:
    """Compare 2-5 configured books over their shared window."""
    result_obj = ComparisonResult()

    overview_by_path_dict = {
        overview_obj.portfolio.rel_path_str: overview_obj
        for overview_obj in portfolio_overview.list_portfolio_overviews()
    }

    selected_list: list[tuple[str, object, object]] = []
    unavailable_list: list[str] = []
    for rel_path_str in rel_path_list:
        overview_obj = overview_by_path_dict.get(rel_path_str)
        if overview_obj is None or overview_obj.latest_metric_run is None:
            unavailable_list.append(rel_path_str)
            continue
        portfolio_obj = _load_portfolio(overview_obj.latest_metric_run)
        if portfolio_obj is None:
            unavailable_list.append(rel_path_str)
            continue
        selected_list.append((rel_path_str, overview_obj, portfolio_obj))

    if unavailable_list:
        result_obj.notice_list.append(
            CompareNotice(
                severity_str="warn",
                title_str="Books left out",
                detail_str=(
                    "No readable run for: "
                    + ", ".join(sorted(unavailable_list))
                    + ". Build them before comparing."
                ),
            )
        )
    if len(selected_list) < COMPARE_MIN_INT:
        result_obj.notice_list.append(
            CompareNotice(
                severity_str="block",
                title_str="Not enough books",
                detail_str=f"Pick at least {COMPARE_MIN_INT} books with a completed run.",
            )
        )
        return result_obj

    # *** CRITICAL*** The shared window is the whole point: every column is
    # recomputed here, so a book that merely started later cannot look better
    # for having missed a drawdown the others lived through.
    common_index = None
    for _rel_path_str, _overview_obj, portfolio_obj in selected_list:
        book_index = portfolio_obj.results.index
        common_index = book_index if common_index is None else common_index.intersection(book_index)

    if common_index is None or len(common_index) < 2:
        result_obj.notice_list.append(
            CompareNotice(
                severity_str="block",
                title_str="No shared history",
                detail_str="These books share no overlapping trading days.",
            )
        )
        return result_obj

    result_obj.common_start_str = str(common_index[0].date())
    result_obj.common_end_str = str(common_index[-1].date())
    result_obj.common_year_float = (common_index[-1] - common_index[0]).days / 365.25

    benchmark_value_ser, benchmark_label_str = _shared_benchmark(
        result_obj, selected_list, common_index
    )
    benchmark_return_ser = (
        None
        if benchmark_value_ser is None
        else benchmark_value_ser.pct_change(fill_method=None)
    )
    result_obj.benchmark_label_str = benchmark_label_str

    return_ser_by_label_dict: dict[str, pd.Series] = {}
    limiting_label_list: list[str] = []
    for rel_path_str, overview_obj, portfolio_obj in selected_list:
        label_str = overview_obj.portfolio.config_name_str
        book_index = portfolio_obj.results.index
        if book_index[0] < common_index[0] or book_index[-1] > common_index[-1]:
            pass  # this book is wider than the shared window; it is not the binder
        else:
            limiting_label_list.append(label_str)

        total_value_ser = portfolio_obj.results.loc[common_index, "total_value"].astype(float)
        return_ser_by_label_dict[label_str] = total_value_ser.pct_change(fill_method=None)
        result_obj.column_list.append(
            BookColumn(
                label_str=label_str,
                rel_path_str=rel_path_str,
                metric_by_name_dict=_metric_dict(total_value_ser, benchmark_return_ser),
                pod_label_list=[
                    f"{pod_obj.strategy_str} {pod_obj.weight_float:.0%}"
                    for pod_obj in overview_obj.portfolio.pod_tuple
                ],
                full_window_str=overview_obj.window_str,
                is_stale_bool=overview_obj.is_stale_bool,
            )
        )

    if benchmark_value_ser is not None:
        result_obj.column_list.append(
            BookColumn(
                label_str=benchmark_label_str or "Benchmark",
                rel_path_str="",
                metric_by_name_dict=_metric_dict(benchmark_value_ser, benchmark_return_ser),
                pod_label_list=[],
                full_window_str=None,
                is_benchmark_bool=True,
            )
        )

    _append_window_notice(result_obj, limiting_label_list)
    _append_stale_notice(result_obj)
    _append_correlation(result_obj, return_ser_by_label_dict)
    return result_obj


def _shared_benchmark(
    result_obj: ComparisonResult,
    selected_list: list[tuple[str, object, object]],
    common_index: pd.DatetimeIndex,
) -> tuple[pd.Series | None, str | None]:
    """One benchmark for every column, or none if the books disagree."""
    label_by_book_dict: dict[str, str] = {}
    series_by_label_dict: dict[str, pd.Series] = {}
    for _rel_path_str, overview_obj, portfolio_obj in selected_list:
        value_ser = getattr(portfolio_obj, "regression_benchmark_value_ser", None)
        label_str = getattr(portfolio_obj, "regression_benchmark_label_str", None)
        if value_ser is None or not label_str:
            continue
        label_by_book_dict[overview_obj.portfolio.config_name_str] = str(label_str)
        series_by_label_dict[str(label_str)] = value_ser.astype(float)

    if not series_by_label_dict:
        result_obj.notice_list.append(
            CompareNotice(
                severity_str="info",
                title_str="No benchmark column",
                detail_str="None of these books stored a benchmark, so there is nothing to compare against.",
            )
        )
        return (None, None)

    # *** CRITICAL*** Agreement is judged on the series, not the label. The two
    # runners spell the same yardstick differently ("$SPX" and
    # "$SPX · TOTALRETURN"), and refusing to compare over a naming difference
    # would drop the benchmark column from most real comparisons. Two series
    # that track each other are one benchmark whatever they are called.
    aligned_by_label_dict = {
        label_str: value_ser.reindex(common_index)
        for label_str, value_ser in series_by_label_dict.items()
    }
    reference_label_str = next(iter(aligned_by_label_dict))
    reference_return_ser = aligned_by_label_dict[reference_label_str].pct_change(fill_method=None)
    for label_str, aligned_candidate_ser in aligned_by_label_dict.items():
        if label_str == reference_label_str:
            continue
        candidate_return_ser = aligned_candidate_ser.pct_change(fill_method=None)
        difference_ser = (candidate_return_ser - reference_return_ser).abs()
        if float(difference_ser.max(skipna=True) or 0.0) > 1e-6:
            result_obj.notice_list.append(
                CompareNotice(
                    severity_str="warn",
                    title_str="Books disagree on benchmark",
                    detail_str=(
                        ", ".join(
                            f"{book_str}={book_label_str}"
                            for book_str, book_label_str in sorted(label_by_book_dict.items())
                        )
                        + ". These are genuinely different series, so no benchmark column is "
                        "shown — one column cannot stand for two yardsticks."
                    ),
                )
            )
            return (None, None)

    label_str = reference_label_str
    aligned_ser = aligned_by_label_dict[label_str]
    if aligned_ser.isna().any():
        result_obj.notice_list.append(
            CompareNotice(
                severity_str="warn",
                title_str="Benchmark incomplete",
                detail_str=f"{label_str} is missing days inside the shared window and is not shown.",
            )
        )
        return (None, None)
    return (aligned_ser, label_str)


def _append_window_notice(result_obj: ComparisonResult, limiting_label_list: list[str]) -> None:
    detail_str = (
        f"Every column is recomputed over {result_obj.common_start_str} to "
        f"{result_obj.common_end_str} ({result_obj.common_year_float:.1f}y), the range these "
        "books share. Figures here will differ from each book's own report, which covers its "
        "full history."
    )
    if limiting_label_list:
        detail_str += " Window bound by: " + ", ".join(sorted(set(limiting_label_list))) + "."
    result_obj.notice_list.append(
        CompareNotice(
            severity_str="info",
            title_str="Measured on the shared window",
            detail_str=detail_str,
        )
    )


def _append_stale_notice(result_obj: ComparisonResult) -> None:
    stale_label_list = [
        column_obj.label_str for column_obj in result_obj.column_list if column_obj.is_stale_bool
    ]
    if not stale_label_list:
        return
    result_obj.notice_list.append(
        CompareNotice(
            severity_str="warn",
            title_str="Stale books in this comparison",
            detail_str=(
                ", ".join(stale_label_list)
                + " were built from pod runs that have since been superseded. Rebuild before "
                "quoting these figures."
            ),
        )
    )


def _append_correlation(
    result_obj: ComparisonResult,
    return_ser_by_label_dict: dict[str, pd.Series],
) -> None:
    """Are two of these the same product under different names?"""
    if len(return_ser_by_label_dict) < 2:
        return
    return_df = pd.DataFrame(return_ser_by_label_dict).dropna()
    if len(return_df) < 252:
        return

    correlation_df = cross_correlation_matrix(return_df)
    result_obj.correlation_label_list = [str(column) for column in correlation_df.columns]
    result_obj.correlation_row_list = [
        {
            "label_str": str(row_label),
            "value_list": [
                {
                    "value_float": float(correlation_df.loc[row_label, column_label]),
                    "self_bool": row_label == column_label,
                    "duplicate_bool": row_label != column_label
                    and float(correlation_df.loc[row_label, column_label])
                    >= DUPLICATE_CORRELATION_FLOAT,
                }
                for column_label in correlation_df.columns
            ],
        }
        for row_label in correlation_df.index
    ]

    duplicate_pair_list = [
        (str(row_label), str(column_label), float(correlation_df.loc[row_label, column_label]))
        for row_index_int, row_label in enumerate(correlation_df.index)
        for column_index_int, column_label in enumerate(correlation_df.columns)
        if column_index_int > row_index_int
        and float(correlation_df.loc[row_label, column_label]) >= DUPLICATE_CORRELATION_FLOAT
    ]
    if duplicate_pair_list:
        result_obj.notice_list.append(
            CompareNotice(
                severity_str="warn",
                title_str="Two books are one product",
                detail_str=(
                    "; ".join(
                        f"{left_str} vs {right_str} = {value_float:.2f}"
                        for left_str, right_str, value_float in duplicate_pair_list
                    )
                    + ". Offering both gives a client a choice that is not a choice."
                ),
            )
        )
