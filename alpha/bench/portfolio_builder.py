"""Compose a portfolio YAML from analyzed strategies, with input diagnostics.

Bench adds no quant logic, and this module keeps that contract: it discovers
combinable runs, reads their saved artifacts, and renders a YAML the existing
runners consume. Every portfolio *result* still comes from
``alpha.engine.Portfolio`` when the book is actually run.

What it does add is a check on the *inputs* — the questions that are cheap to
answer before a run and expensive to notice after one:

  * two pods that are really the same trade (high pairwise correlation),
  * a common window silently truncated by one stale pod,
  * pods that disagree on which benchmark they measured against,
  * a sleeve funded below the capital its strategy needs,
  * pods that are research, not wired for live.

Deliberately absent: any estimate of the portfolio's return. Two numbers that
can disagree with the engine is exactly the failure the house rules forbid, so
the builder diagnoses inputs and lets the engine produce outputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from alpha.bench import catalog, runs
from alpha.engine.metrics import cross_correlation_matrix


PORTFOLIOS_ROOT_PATH = catalog.PORTFOLIOS_ROOT_PATH
RESULTS_ROOT_PATH = runs.RESULTS_ROOT_PATH

# Two pods this correlated are one bet held twice: the book carries the
# concentration of a single sleeve while the weights suggest diversification.
# Calibrated against the pairs already in the catalog — QPI/HPI at 0.94 and the
# two NDX momentum variants at 0.95 are the same trade; 0.75 (DV2 vs QPI) is a
# deliberate, defensible overlap.
REDUNDANT_CORRELATION_FLOAT = 0.85
# Below this, a "since inception" number is a claim about one market regime.
SHORT_WINDOW_YEAR_FLOAT = 5.0
# Correlation over a shorter overlap than this is noise, so it is not reported.
MINIMUM_CORRELATION_OVERLAP_DAY_INT = 252

SEVERITY_BLOCK_STR = "block"
SEVERITY_WARN_STR = "warn"
SEVERITY_INFO_STR = "info"

_pod_return_ser_cache_dict: dict[str, pd.Series] = {}
_cadence_float_cache_dict: dict[str, float | None] = {}


@dataclass(frozen=True)
class PodCandidate:
    """One strategy that has a saved vanilla run, so it can join a book."""

    stem_str: str
    display_name_str: str
    module_import_str: str
    category_label_str: str
    is_wired_bool: bool
    run_obj: runs.RunEntry
    benchmark_symbol_str: str | None

    @property
    def window_str(self) -> str | None:
        return self.run_obj.backtest_window_str

    @property
    def ann_return_float(self) -> float | None:
        return _summary_float(self.run_obj, "ann_return_pct")

    @property
    def sharpe_float(self) -> float | None:
        return _summary_float(self.run_obj, "sharpe")

    @property
    def max_drawdown_float(self) -> float | None:
        return _summary_float(self.run_obj, "max_drawdown_pct")


@dataclass(frozen=True)
class BuilderNotice:
    """One thing the operator should know before running this book."""

    severity_str: str
    title_str: str
    detail_str: str


@dataclass
class SelectionDiagnostics:
    pod_view_list: list[dict] = field(default_factory=list)
    notice_list: list[BuilderNotice] = field(default_factory=list)
    correlation_row_list: list[dict] = field(default_factory=list)
    correlation_label_list: list[str] = field(default_factory=list)
    common_start_str: str | None = None
    common_end_str: str | None = None
    common_year_float: float | None = None
    resolved_benchmark_str: str | None = None
    yaml_text_str: str = ""
    suggested_filename_str: str = ""

    @property
    def has_block_bool(self) -> bool:
        return any(
            notice_obj.severity_str == SEVERITY_BLOCK_STR for notice_obj in self.notice_list
        )


def _summary_float(run_obj: runs.RunEntry, key_str: str) -> float | None:
    value_obj = run_obj.summary_dict.get(key_str)
    if isinstance(value_obj, bool) or not isinstance(value_obj, (int, float)):
        return None
    return float(value_obj)


def _benchmark_symbol_str(run_obj: runs.RunEntry) -> str | None:
    benchmark_obj = run_obj.metadata_dict.get("benchmarks")
    if isinstance(benchmark_obj, list) and len(benchmark_obj) > 0:
        return str(benchmark_obj[0])
    return None


def list_pod_candidates() -> list[PodCandidate]:
    """Every strategy with a saved vanilla run, wired ones first.

    A pod can only join a combine-pkls book if it has a completed vanilla run
    to combine, so the candidate list *is* the set of strategies with one —
    offering the rest would only produce a config that dies on launch.
    """
    run_index_obj = runs.build_strategy_run_index()
    candidate_list: list[PodCandidate] = []
    for strategy_entry_obj in catalog.list_strategies():
        run_obj = run_index_obj.latest_vanilla_for(
            strategy_entry_obj.module_import_str, strategy_entry_obj.stem_str
        )
        if run_obj is None:
            continue
        candidate_list.append(
            PodCandidate(
                stem_str=run_obj.run_name_str,
                display_name_str=strategy_entry_obj.display_name_str,
                module_import_str=strategy_entry_obj.module_import_str,
                category_label_str=strategy_entry_obj.category_label_str,
                is_wired_bool=strategy_entry_obj.is_wired_bool,
                run_obj=run_obj,
                benchmark_symbol_str=_benchmark_symbol_str(run_obj),
            )
        )
    candidate_list.sort(
        key=lambda candidate_obj: (
            not candidate_obj.is_wired_bool,
            candidate_obj.category_label_str.lower(),
            candidate_obj.stem_str.lower(),
        )
    )
    return candidate_list


def candidate_by_stem_dict() -> dict[str, PodCandidate]:
    return {candidate_obj.stem_str: candidate_obj for candidate_obj in list_pod_candidates()}


def _pod_pickle_path(run_obj: runs.RunEntry) -> Path:
    return (
        RESULTS_ROOT_PATH / run_obj.rel_dir_from_results_str / f"{run_obj.run_name_str}.pkl"
    )


def _pod_return_ser(run_obj: runs.RunEntry) -> pd.Series | None:
    """Daily returns of one saved run, read from its pickle.

    Cached on the run directory: a timestamped artifact never changes, so the
    expensive unpickle happens once per Bench process.
    """
    cache_key_str = run_obj.rel_dir_from_results_str
    if cache_key_str in _pod_return_ser_cache_dict:
        return _pod_return_ser_cache_dict[cache_key_str]

    pickle_path = _pod_pickle_path(run_obj)
    if not pickle_path.exists():
        return None
    try:
        # Imported lazily: this pulls the strategy-class registration machinery
        # (and IPython) that the rest of Bench never needs.
        from strategies.run_portfolio import load_strategy_pickle

        strategy_obj, _metadata_dict = load_strategy_pickle(pickle_path)
        return_ser = (
            strategy_obj.results["total_value"].astype(float).pct_change(fill_method=None)
        )
    except Exception:
        # A pod whose pickle cannot be read is reported as an unavailable
        # diagnostic, never as an absence of correlation.
        return None
    _pod_return_ser_cache_dict[cache_key_str] = return_ser
    return return_ser


def _trading_days_per_month_float(run_obj: runs.RunEntry) -> float | None:
    """Distinct trading days per month on which this pod transacts.

    The honest cadence measure. Counting raw transactions conflates "how often
    it trades" with "how many names it holds": a 10-name book rebalanced once a
    month books ~10 transactions on a single day, which is monthly, not daily.
    """
    cache_key_str = run_obj.rel_dir_from_results_str
    if cache_key_str in _cadence_float_cache_dict:
        return _cadence_float_cache_dict[cache_key_str]

    transaction_path = (
        RESULTS_ROOT_PATH / run_obj.rel_dir_from_results_str / "transactions.csv"
    )
    cadence_float: float | None = None
    if transaction_path.exists():
        try:
            transaction_df = pd.read_csv(transaction_path, usecols=["bar"])
            bar_ser = pd.to_datetime(transaction_df["bar"])
            month_count_int = len(bar_ser.dt.to_period("M").unique())
            if month_count_int > 0:
                cadence_float = bar_ser.dt.date.nunique() / month_count_int
        except (OSError, ValueError, KeyError):
            cadence_float = None
    _cadence_float_cache_dict[cache_key_str] = cadence_float
    return cadence_float


def _pod_minimum_capital_dict() -> dict[str, float]:
    """Per-strategy minimum funding, keyed by stem.

    Reuses the PortfolioManager's table rather than restating the numbers, so
    the two paths cannot drift apart.
    """
    from alpha.engine.portfolio_manager import POD_MINIMUM_ALLOCATED_CAPITAL_FLOAT_DICT

    minimum_by_stem_dict: dict[str, float] = {}
    for import_str, minimum_float in POD_MINIMUM_ALLOCATED_CAPITAL_FLOAT_DICT.items():
        stem_str = import_str.split(":", maxsplit=1)[0].rsplit(".", maxsplit=1)[-1]
        minimum_by_stem_dict[stem_str] = float(minimum_float)
    return minimum_by_stem_dict


def slugify_filename_str(name_str: str) -> str:
    """``Monthly Defensive`` -> ``monthly_defensive`` for the YAML filename."""
    lowered_str = re.sub(r"(?<!^)(?=[A-Z])", "_", str(name_str).strip()).lower()
    slug_str = re.sub(r"[^a-z0-9]+", "_", lowered_str).strip("_")
    return slug_str or "portfolio"


def render_yaml_text(
    name_str: str,
    capital_float: float,
    benchmark_str: str | None,
    pod_pair_list: list[tuple[str, float]],
) -> str:
    """Render the config in the house style of ``portfolios/multipod.yaml``."""
    line_list = [f"name: {name_str}", f"capital: {capital_float:g}"]
    if benchmark_str:
        line_list.append(f"benchmark: {benchmark_str}")
    line_list.append("")
    line_list.append("pods:")
    for index_int, (stem_str, weight_float) in enumerate(pod_pair_list):
        if index_int > 0:
            line_list.append("")
        line_list.append(f"  - strategy: {stem_str}")
        line_list.append(f"    weight: {weight_float:g}")
    return "\n".join(line_list) + "\n"


WEIGHT_DECIMAL_INT = 4


def _exact_sum_weight_list(weight_list: list[float]) -> list[float]:
    """Round weights for display while keeping their sum exactly 1.0.

    *** CRITICAL*** The runner rejects a config whose weights miss 1.0 by more
    than 1e-6, and a third rendered as 0.333333 three times sums to 0.999999 —
    right on that boundary. Round to a readable precision, then carry the whole
    residue on the largest weight so the written file always sums exactly.
    """
    rounded_list = [round(weight_float, WEIGHT_DECIMAL_INT) for weight_float in weight_list]
    residue_float = round(1.0 - sum(rounded_list), WEIGHT_DECIMAL_INT)
    if residue_float != 0.0:
        largest_index_int = max(
            range(len(rounded_list)), key=lambda index_int: rounded_list[index_int]
        )
        rounded_list[largest_index_int] = round(
            rounded_list[largest_index_int] + residue_float, WEIGHT_DECIMAL_INT
        )
    return rounded_list


def _normalized_weight_list(weight_list: list[float]) -> tuple[list[float], bool]:
    total_float = sum(weight_list)
    if total_float <= 0:
        equal_float = 1.0 / len(weight_list)
        return (_exact_sum_weight_list([equal_float] * len(weight_list)), True)
    if abs(total_float - 1.0) <= 1e-9:
        return (list(weight_list), False)
    return (
        _exact_sum_weight_list(
            [weight_float / total_float for weight_float in weight_list]
        ),
        True,
    )


def analyze_selection(
    selection_pair_list: list[tuple[str, float]],
    name_str: str,
    capital_float: float,
    benchmark_override_str: str | None = None,
) -> SelectionDiagnostics:
    """Diagnose one candidate book and render its YAML.

    ``selection_pair_list`` is ``[(strategy_stem, weight), ...]`` exactly as the
    form submitted it; weights are normalized here and the normalization is
    reported rather than applied silently.
    """
    diagnostics_obj = SelectionDiagnostics()
    candidate_dict = candidate_by_stem_dict()

    known_pair_list = [
        (stem_str, weight_float)
        for stem_str, weight_float in selection_pair_list
        if stem_str in candidate_dict
    ]
    unknown_stem_list = [
        stem_str for stem_str, _ in selection_pair_list if stem_str not in candidate_dict
    ]
    if unknown_stem_list:
        diagnostics_obj.notice_list.append(
            BuilderNotice(
                severity_str=SEVERITY_BLOCK_STR,
                title_str="Unknown strategy",
                detail_str=(
                    "No saved vanilla run for: " + ", ".join(sorted(unknown_stem_list))
                ),
            )
        )
    if len(known_pair_list) < 2:
        diagnostics_obj.notice_list.append(
            BuilderNotice(
                severity_str=SEVERITY_BLOCK_STR,
                title_str="Too few pods",
                detail_str="A book needs at least two pods.",
            )
        )
        return diagnostics_obj

    stem_list = [stem_str for stem_str, _ in known_pair_list]
    weight_list, was_normalized_bool = _normalized_weight_list(
        [weight_float for _, weight_float in known_pair_list]
    )
    if was_normalized_bool:
        diagnostics_obj.notice_list.append(
            BuilderNotice(
                severity_str=SEVERITY_INFO_STR,
                title_str="Weights normalized",
                detail_str=(
                    "Submitted weights did not sum to 1.0 and were rescaled "
                    "proportionally; the runner requires an exact sum."
                ),
            )
        )

    minimum_by_stem_dict = _pod_minimum_capital_dict()
    return_ser_by_stem_dict: dict[str, pd.Series] = {}

    for stem_str, weight_float in zip(stem_list, weight_list):
        candidate_obj = candidate_dict[stem_str]
        allocated_float = capital_float * weight_float
        minimum_float = minimum_by_stem_dict.get(stem_str)
        return_ser = _pod_return_ser(candidate_obj.run_obj)
        if return_ser is not None:
            return_ser_by_stem_dict[stem_str] = return_ser
        diagnostics_obj.pod_view_list.append(
            {
                "candidate": candidate_obj,
                "weight_float": weight_float,
                "allocated_float": allocated_float,
                "minimum_float": minimum_float,
                "underfunded_bool": minimum_float is not None
                and allocated_float < minimum_float,
                "cadence_float": _trading_days_per_month_float(candidate_obj.run_obj),
            }
        )

    _append_funding_notices(diagnostics_obj)
    _append_wired_notice(diagnostics_obj)
    _append_window_notices(diagnostics_obj, return_ser_by_stem_dict)
    resolved_benchmark_str = _append_benchmark_notices(
        diagnostics_obj, candidate_dict, stem_list, benchmark_override_str
    )
    _append_correlation_notices(diagnostics_obj, return_ser_by_stem_dict)

    diagnostics_obj.resolved_benchmark_str = resolved_benchmark_str
    diagnostics_obj.yaml_text_str = render_yaml_text(
        name_str=name_str,
        capital_float=capital_float,
        benchmark_str=resolved_benchmark_str,
        pod_pair_list=list(zip(stem_list, weight_list)),
    )
    diagnostics_obj.suggested_filename_str = f"{slugify_filename_str(name_str)}.yaml"
    return diagnostics_obj


def _append_funding_notices(diagnostics_obj: SelectionDiagnostics) -> None:
    underfunded_view_list = [
        pod_view_dict
        for pod_view_dict in diagnostics_obj.pod_view_list
        if pod_view_dict["underfunded_bool"]
    ]
    if not underfunded_view_list:
        return
    detail_str = "; ".join(
        f"{pod_view_dict['candidate'].stem_str} gets "
        f"${pod_view_dict['allocated_float']:,.0f} but needs "
        f"${pod_view_dict['minimum_float']:,.0f}"
        for pod_view_dict in underfunded_view_list
    )
    diagnostics_obj.notice_list.append(
        BuilderNotice(
            severity_str=SEVERITY_WARN_STR,
            title_str="Sleeve underfunded",
            detail_str=(
                f"{detail_str}. Below its minimum, fixed commissions and whole-share "
                "sizing eat the edge this strategy is supposed to earn."
            ),
        )
    )


def _append_wired_notice(diagnostics_obj: SelectionDiagnostics) -> None:
    research_stem_list = [
        pod_view_dict["candidate"].stem_str
        for pod_view_dict in diagnostics_obj.pod_view_list
        if not pod_view_dict["candidate"].is_wired_bool
    ]
    if not research_stem_list:
        return
    diagnostics_obj.notice_list.append(
        BuilderNotice(
            severity_str=SEVERITY_INFO_STR,
            title_str="Contains research pods",
            detail_str=(
                "Not wired for live: " + ", ".join(research_stem_list) + ". "
                "Fine for research; label it before showing the book to anyone else."
            ),
        )
    )


def _append_window_notices(
    diagnostics_obj: SelectionDiagnostics,
    return_ser_by_stem_dict: dict[str, pd.Series],
) -> None:
    if len(return_ser_by_stem_dict) < 2:
        diagnostics_obj.notice_list.append(
            BuilderNotice(
                severity_str=SEVERITY_WARN_STR,
                title_str="Diagnostics unavailable",
                detail_str=(
                    "Could not read saved returns for enough pods, so the common "
                    "window and correlations are not shown. The book can still run."
                ),
            )
        )
        return

    common_index = None
    for return_ser in return_ser_by_stem_dict.values():
        common_index = (
            return_ser.index if common_index is None else common_index.intersection(return_ser.index)
        )
    if common_index is None or len(common_index) == 0:
        diagnostics_obj.notice_list.append(
            BuilderNotice(
                severity_str=SEVERITY_BLOCK_STR,
                title_str="No overlapping history",
                detail_str="These pods share no common trading days, so no book can be built.",
            )
        )
        return

    year_float = (common_index[-1] - common_index[0]).days / 365.25
    diagnostics_obj.common_start_str = str(common_index[0].date())
    diagnostics_obj.common_end_str = str(common_index[-1].date())
    diagnostics_obj.common_year_float = year_float

    # The book is measured only where every pod has history, so one short or
    # stale pod silently decides what the whole book can claim.
    limiting_stem_list = [
        stem_str
        for stem_str, return_ser in return_ser_by_stem_dict.items()
        if return_ser.index[0] > common_index[0] or return_ser.index[-1] < common_index[-1]
    ]
    if limiting_stem_list:
        diagnostics_obj.notice_list.append(
            BuilderNotice(
                severity_str=SEVERITY_INFO_STR,
                title_str="Window set by one pod",
                detail_str=(
                    f"The book is measured over {diagnostics_obj.common_start_str} to "
                    f"{diagnostics_obj.common_end_str} ({year_float:.1f}y), bounded by: "
                    + ", ".join(sorted(limiting_stem_list))
                    + ". Re-run a stale pod to extend it."
                ),
            )
        )
    if year_float < SHORT_WINDOW_YEAR_FLOAT:
        diagnostics_obj.notice_list.append(
            BuilderNotice(
                severity_str=SEVERITY_WARN_STR,
                title_str="Short common window",
                detail_str=(
                    f"Only {year_float:.1f} years overlap. Metrics from a window this "
                    "short describe one market regime, not an edge."
                ),
            )
        )


def _append_benchmark_notices(
    diagnostics_obj: SelectionDiagnostics,
    candidate_dict: dict[str, PodCandidate],
    stem_list: list[str],
    benchmark_override_str: str | None,
) -> str | None:
    symbol_by_stem_dict = {
        stem_str: candidate_dict[stem_str].benchmark_symbol_str
        for stem_str in stem_list
        if candidate_dict[stem_str].benchmark_symbol_str
    }
    distinct_symbol_set = set(symbol_by_stem_dict.values())

    if benchmark_override_str:
        chosen_str = benchmark_override_str.strip()
        if chosen_str not in distinct_symbol_set:
            diagnostics_obj.notice_list.append(
                BuilderNotice(
                    severity_str=SEVERITY_BLOCK_STR,
                    title_str="Benchmark not stored",
                    detail_str=(
                        f"No selected pod stores '{chosen_str}'. Stored: "
                        + (", ".join(sorted(distinct_symbol_set)) or "none")
                    ),
                )
            )
        return chosen_str

    if len(distinct_symbol_set) <= 1:
        return next(iter(distinct_symbol_set), None)

    # Mixed benchmarks: auto-derivation refuses to choose, which would drop
    # every benchmark-relative section from the report. Pick the most common
    # and write it explicitly, saying so.
    symbol_count_dict: dict[str, int] = {}
    for symbol_str in symbol_by_stem_dict.values():
        symbol_count_dict[symbol_str] = symbol_count_dict.get(symbol_str, 0) + 1
    chosen_str = max(sorted(symbol_count_dict), key=lambda key_str: symbol_count_dict[key_str])
    diagnostics_obj.notice_list.append(
        BuilderNotice(
            severity_str=SEVERITY_WARN_STR,
            title_str="Pods disagree on benchmark",
            detail_str=(
                "Stored benchmarks: "
                + ", ".join(f"{stem}={sym}" for stem, sym in sorted(symbol_by_stem_dict.items()))
                + f". Writing 'benchmark: {chosen_str}' explicitly — without it the runner "
                "refuses to choose and the report loses every benchmark-relative section."
            ),
        )
    )
    return chosen_str


def _append_correlation_notices(
    diagnostics_obj: SelectionDiagnostics,
    return_ser_by_stem_dict: dict[str, pd.Series],
) -> None:
    if len(return_ser_by_stem_dict) < 2:
        return

    return_df = pd.DataFrame(return_ser_by_stem_dict).dropna()
    if len(return_df) < MINIMUM_CORRELATION_OVERLAP_DAY_INT:
        diagnostics_obj.notice_list.append(
            BuilderNotice(
                severity_str=SEVERITY_WARN_STR,
                title_str="Correlations not shown",
                detail_str=(
                    f"Only {len(return_df)} overlapping days — fewer than a year, so a "
                    "correlation here would be noise."
                ),
            )
        )
        return

    correlation_df = cross_correlation_matrix(return_df)
    diagnostics_obj.correlation_label_list = [str(column) for column in correlation_df.columns]
    diagnostics_obj.correlation_row_list = [
        {
            "label_str": str(row_label),
            "value_list": [
                {
                    "value_float": float(correlation_df.loc[row_label, column_label]),
                    "self_bool": row_label == column_label,
                    "redundant_bool": row_label != column_label
                    and float(correlation_df.loc[row_label, column_label])
                    >= REDUNDANT_CORRELATION_FLOAT,
                }
                for column_label in correlation_df.columns
            ],
        }
        for row_label in correlation_df.index
    ]

    redundant_pair_list = [
        (str(row_label), str(column_label), float(correlation_df.loc[row_label, column_label]))
        for row_index_int, row_label in enumerate(correlation_df.index)
        for column_index_int, column_label in enumerate(correlation_df.columns)
        if column_index_int > row_index_int
        and float(correlation_df.loc[row_label, column_label]) >= REDUNDANT_CORRELATION_FLOAT
    ]
    if redundant_pair_list:
        diagnostics_obj.notice_list.append(
            BuilderNotice(
                severity_str=SEVERITY_WARN_STR,
                title_str="Pods are the same trade",
                detail_str=(
                    "; ".join(
                        f"{left_str} vs {right_str} = {value_float:.2f}"
                        for left_str, right_str, value_float in redundant_pair_list
                    )
                    + ". The book carries one sleeve's concentration while the weights "
                    "suggest diversification."
                ),
            )
        )


def resolve_write_path(filename_str: str) -> Path:
    """Resolve a YAML filename inside ``portfolios/``, refusing anything else."""
    cleaned_str = str(filename_str).strip()
    if not cleaned_str:
        raise ValueError("Filename must not be empty.")
    if not cleaned_str.endswith(".yaml"):
        cleaned_str = f"{cleaned_str}.yaml"
    if Path(cleaned_str).name != cleaned_str:
        raise ValueError("Filename must not contain a path.")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", cleaned_str):
        raise ValueError("Filename may use letters, digits, dot, dash and underscore only.")

    portfolios_root_resolved_path = PORTFOLIOS_ROOT_PATH.resolve()
    candidate_path = (portfolios_root_resolved_path / cleaned_str).resolve()
    if candidate_path.parent != portfolios_root_resolved_path:
        raise ValueError("Refusing to write outside portfolios/.")
    return candidate_path


def write_portfolio_yaml(
    filename_str: str,
    yaml_text_str: str,
    overwrite_bool: bool = False,
) -> Path:
    """Write the config, refusing to clobber an existing file by default."""
    write_path = resolve_write_path(filename_str)
    if write_path.exists() and not overwrite_bool:
        raise FileExistsError(f"{write_path.name} already exists.")
    write_path.parent.mkdir(parents=True, exist_ok=True)
    write_path.write_text(yaml_text_str, encoding="utf-8")
    return write_path
