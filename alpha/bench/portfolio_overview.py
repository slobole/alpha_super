"""Portfolio list view: measured books, ranked, with a staleness check.

The Portfolios page used to be a wall of cards carrying composition but no
result, which answers "what did I configure" and not "which book is better".
This module pairs each config with the numbers its latest run recorded, so the
page can be sorted the way the strategy catalog is.

It also answers a question nothing in Bench asked before: **is this book's
number still true?** A combine-pkls book is a snapshot of the pod artifacts it
was built from. Re-run a pod and the book silently keeps quoting the old one —
which happened during the total-return benchmark fix, when every monthly book
kept reporting figures computed against a superseded pod until it was rebuilt.
Both timestamps are recorded on disk, so the staleness is free to detect and
only needed a place to be shown.

Read-only, like everything else in Bench: no metric is recomputed here beyond
MAR, which is a ratio of two figures the runner already wrote.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from alpha.bench import catalog, runs


VANILLA_ANALYSIS_DIR_STR = "vanilla_backtest"


@dataclass(frozen=True)
class StalePod:
    """One pod whose newest run is newer than the one the book was built from."""

    strategy_name_str: str
    used_timestamp_str: str
    latest_timestamp_str: str


@dataclass(frozen=True)
class PortfolioOverview:
    portfolio: catalog.PortfolioEntry
    latest_report_run: runs.RunEntry | None
    latest_metric_run: runs.RunEntry | None
    run_entry_list: list[runs.RunEntry]
    stale_pod_list: list[StalePod]

    @property
    def ann_return_float(self) -> float | None:
        return _summary_float(self.latest_metric_run, "ann_return_pct")

    @property
    def sharpe_float(self) -> float | None:
        return _summary_float(self.latest_metric_run, "sharpe")

    @property
    def max_drawdown_float(self) -> float | None:
        return _summary_float(self.latest_metric_run, "max_drawdown_pct")

    @property
    def trade_count_float(self) -> float | None:
        return _summary_float(self.latest_metric_run, "trade_count")

    @property
    def mar_float(self) -> float | None:
        """Return over the worst loss it took to earn — how books rank for a client.

        Free arithmetic on two figures the runner already recorded; nothing is
        re-derived from the equity curve.
        """
        ann_return_float = self.ann_return_float
        max_drawdown_float = self.max_drawdown_float
        if ann_return_float is None or not max_drawdown_float:
            return None
        return ann_return_float / abs(max_drawdown_float)

    @property
    def window_str(self) -> str | None:
        if self.latest_metric_run is None:
            return None
        metadata_dict = self.latest_metric_run.metadata_dict
        start_obj = metadata_dict.get("common_start")
        end_obj = metadata_dict.get("common_end")
        if start_obj and end_obj:
            return f"{str(start_obj)[:10]} → {str(end_obj)[:10]}"
        return self.latest_metric_run.backtest_window_str

    @property
    def window_start_str(self) -> str:
        """Sort key for the window column; empty sinks to the bottom."""
        if self.latest_metric_run is None:
            return ""
        return str(self.latest_metric_run.metadata_dict.get("common_start") or "")[:10]

    @property
    def has_run_bool(self) -> bool:
        return self.latest_metric_run is not None

    @property
    def is_stale_bool(self) -> bool:
        return len(self.stale_pod_list) > 0

    @property
    def search_text_str(self) -> str:
        pod_name_str = " ".join(pod_obj.strategy_str for pod_obj in self.portfolio.pod_tuple)
        return " ".join(
            [self.portfolio.name_str, self.portfolio.config_name_str, pod_name_str]
        ).lower()


def _summary_float(run_obj: runs.RunEntry | None, key_str: str) -> float | None:
    if run_obj is None:
        return None
    value_obj = run_obj.summary_dict.get(key_str)
    if isinstance(value_obj, bool) or not isinstance(value_obj, (int, float)):
        return None
    return float(value_obj)


def _parse_saved_at(value_obj) -> datetime | None:
    if not value_obj:
        return None
    try:
        return datetime.fromisoformat(str(value_obj))
    except ValueError:
        return None


def _latest_vanilla_datetime_by_run_name_dict(
    strategy_name_set: set[str],
) -> dict[str, datetime]:
    """Newest vanilla run for each named strategy, as a datetime.

    Reads only the directories for pods a book actually references, rather than
    indexing the whole strategy tree: staleness is a question about a handful of
    named pods, and the full index costs a walk of every result folder to answer
    it.
    """
    latest_by_name_dict: dict[str, datetime] = {}
    for strategy_name_str in strategy_name_set:
        vanilla_dir_path = (
            runs.RESEARCH_STRATEGY_ROOT_PATH / strategy_name_str / VANILLA_ANALYSIS_DIR_STR
        )
        if not vanilla_dir_path.is_dir():
            continue
        for run_dir_path in vanilla_dir_path.iterdir():
            if not run_dir_path.is_dir():
                continue
            try:
                timestamp_datetime_obj = datetime.strptime(
                    run_dir_path.name, runs.RUN_TIMESTAMP_FORMAT_STR
                )
            except ValueError:
                continue
            current_obj = latest_by_name_dict.get(strategy_name_str)
            if current_obj is None or timestamp_datetime_obj > current_obj:
                latest_by_name_dict[strategy_name_str] = timestamp_datetime_obj
    return latest_by_name_dict


def _referenced_strategy_name_set(portfolio_run_obj: runs.RunEntry | None) -> set[str]:
    if portfolio_run_obj is None:
        return set()
    pod_payload_list = portfolio_run_obj.metadata_dict.get("pods")
    if not isinstance(pod_payload_list, list):
        return set()
    return {
        str(pod_payload_dict.get("strategy_name"))
        for pod_payload_dict in pod_payload_list
        if isinstance(pod_payload_dict, dict) and pod_payload_dict.get("strategy_name")
    }


def _stale_pod_list(
    portfolio_run_obj: runs.RunEntry | None,
    latest_by_name_dict: dict[str, datetime],
) -> list[StalePod]:
    """Pods that have been re-run since this book was built.

    Only meaningful for a run whose metadata recorded the pod artifacts it
    consumed; a fresh-run book re-runs its pods every time, so it records the
    runs it just produced and nothing is ever stale.
    """
    if portfolio_run_obj is None:
        return []
    pod_payload_list = portfolio_run_obj.metadata_dict.get("pods")
    if not isinstance(pod_payload_list, list):
        return []

    stale_pod_list: list[StalePod] = []
    for pod_payload_dict in pod_payload_list:
        if not isinstance(pod_payload_dict, dict):
            continue
        strategy_name_str = str(pod_payload_dict.get("strategy_name") or "")
        if not strategy_name_str:
            continue
        result_metadata_dict = pod_payload_dict.get("result_metadata") or {}
        used_datetime_obj = _parse_saved_at(result_metadata_dict.get("saved_at"))
        latest_datetime_obj = latest_by_name_dict.get(strategy_name_str)
        if used_datetime_obj is None or latest_datetime_obj is None:
            continue
        # A whole second of slack: the pod artifact's saved_at and the run
        # folder's timestamp are written by different code paths in the same
        # run and can disagree by sub-second rounding.
        if (latest_datetime_obj - used_datetime_obj).total_seconds() <= 1.0:
            continue
        stale_pod_list.append(
            StalePod(
                strategy_name_str=strategy_name_str,
                used_timestamp_str=used_datetime_obj.strftime("%Y-%m-%d %H:%M"),
                latest_timestamp_str=latest_datetime_obj.strftime("%Y-%m-%d %H:%M"),
            )
        )
    return stale_pod_list


def list_portfolio_overviews() -> list[PortfolioOverview]:
    """Every configured book with its latest measured run and staleness."""
    overview_list: list[PortfolioOverview] = []
    for portfolio_entry_obj in catalog.list_portfolios():
        run_entry_list = runs.scan_portfolio_runs(
            portfolio_entry_obj.name_str,
            portfolio_entry_obj.config_name_str,
        )
        latest_report_run_obj = next(
            (run_obj for run_obj in run_entry_list if run_obj.has_report_bool), None
        )
        # Metrics come from the newest run that actually recorded a return; a
        # later diagnostic-only run must not blank the book's headline.
        latest_metric_run_obj = next(
            (
                run_obj
                for run_obj in run_entry_list
                if isinstance(run_obj.summary_dict.get("ann_return_pct"), (int, float))
            ),
            None,
        )
        latest_by_name_dict = _latest_vanilla_datetime_by_run_name_dict(
            _referenced_strategy_name_set(latest_metric_run_obj)
        )
        overview_list.append(
            PortfolioOverview(
                portfolio=portfolio_entry_obj,
                latest_report_run=latest_report_run_obj,
                latest_metric_run=latest_metric_run_obj,
                run_entry_list=run_entry_list,
                stale_pod_list=_stale_pod_list(latest_metric_run_obj, latest_by_name_dict),
            )
        )
    return overview_list
