"""Reader for the ``results/`` artifact tree.

Every analyzer run already writes a tidy, timestamped folder:

    results/research/strategy/{run_name}/{analysis}/{YYYY-MM-DD_HHMMSS}/
        summary.json   metadata.json   run_info.json   report.html   ...

    results/research/portfolio/{name}/{analysis}/{YYYY-MM-DD_HHMMSS}/
        summary.json   run_info.json   report.html     ...

Bench just lists those folders and reads the small JSON sidecars. It writes
nothing here. Linking a results folder back to a catalog strategy uses the
``class_module`` field in ``metadata.json`` (robust to ``*_research`` run-name
suffixes), and falls back to matching the run-name to the file stem.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
RESULTS_ROOT_PATH = REPO_ROOT_PATH / "results"
STRATEGIES_ROOT_PATH = REPO_ROOT_PATH / "strategies"
RESEARCH_STRATEGY_ROOT_PATH = RESULTS_ROOT_PATH / "research" / "strategy"
RESEARCH_PORTFOLIO_ROOT_PATH = RESULTS_ROOT_PATH / "research" / "portfolio"
RUN_TIMESTAMP_FORMAT_STR = "%Y-%m-%d_%H%M%S"

ANALYSIS_LABEL_DICT: dict[str, str] = {
    "vanilla_backtest": "Vanilla",
    "friction_analysis": "Friction",
    "capacity_analysis": "Capacity",
    "execution_timing_analyzer": "Timing",
    "risk_analysis": "Risk",
    "stress_test": "Stress",
    "crisis_replay": "Crisis",
}

# Headline numbers we know how to colour. Anything else is shown as a plain
# key/value pair in the run's detail table.
_PERCENT_KEY_SET = {"ann_return_pct", "max_drawdown_pct", "cagr_pct", "total_return_pct"}
_HEADLINE_KEY_ORDER = ("ann_return_pct", "sharpe", "max_drawdown_pct")
_HEADLINE_LABEL_DICT = {
    "ann_return_pct": "CAGR",
    "sharpe": "Sharpe",
    "max_drawdown_pct": "Max DD",
}


@dataclass(frozen=True)
class MetricChip:
    label_str: str
    value_str: str
    tone_str: str  # "pos" | "neg" | "neutral"


@dataclass
class RunEntry:
    run_name_str: str  # the results folder name (the strategy's run name)
    analysis_dir_str: str  # raw folder, e.g. "vanilla_backtest"
    analysis_label_str: str  # friendly label, e.g. "Vanilla"
    timestamp_str: str  # raw folder, e.g. "2026-05-20_160858"
    rel_dir_from_results_str: str  # posix path relative to results/, for the artifact route
    has_report_bool: bool
    activity_timestamp_float: float = 0.0
    summary_dict: dict = field(default_factory=dict)
    metadata_dict: dict = field(default_factory=dict)
    run_info_dict: dict = field(default_factory=dict)

    @property
    def timestamp_datetime_obj(self) -> datetime | None:
        try:
            return datetime.strptime(self.timestamp_str, RUN_TIMESTAMP_FORMAT_STR)
        except ValueError:
            return None

    @property
    def effective_activity_timestamp_float(self) -> float:
        if self.activity_timestamp_float > 0:
            return self.activity_timestamp_float
        timestamp_datetime_obj = self.timestamp_datetime_obj
        return timestamp_datetime_obj.timestamp() if timestamp_datetime_obj is not None else 0.0

    @property
    def is_indexable_run_bool(self) -> bool:
        return self.effective_activity_timestamp_float > 0

    @property
    def display_timestamp_str(self) -> str:
        if self.activity_timestamp_float > 0:
            return datetime.fromtimestamp(self.activity_timestamp_float).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        timestamp_datetime_obj = self.timestamp_datetime_obj
        if timestamp_datetime_obj is not None:
            return timestamp_datetime_obj.strftime("%Y-%m-%d %H:%M:%S")
        return self.timestamp_str

    @property
    def report_artifact_str(self) -> str:
        return f"{self.rel_dir_from_results_str}/report.html"

    @property
    def parameter_dict(self) -> dict:
        parameter_obj = self.run_info_dict.get("parameters")
        return parameter_obj if isinstance(parameter_obj, dict) else {}

    @property
    def backtest_window_str(self) -> str | None:
        """The tested date range, as the runner recorded it.

        Two runs of the same strategy over different windows are otherwise
        indistinguishable in the run history — same analysis, same metrics
        columns — which makes the table unusable for comparison. Returns None
        when the runner wrote no window, so the UI can show an explicit dash
        rather than implying full history.
        """
        start_obj = self.parameter_dict.get("start_date")
        end_obj = self.parameter_dict.get("end_date")
        if not (isinstance(start_obj, str) and isinstance(end_obj, str)):
            return None
        return f"{start_obj[:10]} → {end_obj[:10]}"

    @property
    def capital_display_str(self) -> str | None:
        capital_obj = self.parameter_dict.get("capital")
        if not isinstance(capital_obj, (int, float)) or isinstance(capital_obj, bool):
            return None
        return f"{float(capital_obj):,.0f}"

    @property
    def capacity_model_version_str(self) -> str | None:
        if self.analysis_dir_str != "capacity_analysis":
            return None
        model_version_obj = self.metadata_dict.get("model_version_str")
        return str(model_version_obj) if model_version_obj else "capacity_v1"

    @property
    def is_legacy_capacity_bool(self) -> bool:
        return self.capacity_model_version_str == "capacity_v1"

    @property
    def display_analysis_label_str(self) -> str:
        if self.is_legacy_capacity_bool:
            return "Capacity · Legacy v1"
        if self.capacity_model_version_str:
            version_str = self.capacity_model_version_str.replace("capacity_", "").replace(
                "_",
                ".",
            )
            return f"Capacity · {version_str}"
        return self.analysis_label_str

    @property
    def capacity_window_date_summary_str(self) -> str | None:
        if self.analysis_dir_str != "capacity_analysis":
            return None
        window_date_obj = self.metadata_dict.get("window_date_dict")
        if not isinstance(window_date_obj, dict):
            return "Window dates unavailable"

        def window_date_range_str(window_str: str) -> str:
            date_obj = window_date_obj.get(window_str)
            if not isinstance(date_obj, dict):
                return "N/A"
            start_obj = date_obj.get("actual_start_date_str")
            end_obj = date_obj.get("actual_end_date_str")
            return f"{start_obj} to {end_obj}" if start_obj and end_obj else "N/A"

        return (
            f"Recent: {window_date_range_str('recent_5y')} · "
            f"Full: {window_date_range_str('full_history')}"
        )

    def headline_chip_list(self) -> list[MetricChip]:
        return [
            chip_obj
            for key_str in _HEADLINE_KEY_ORDER
            if (chip_obj := _metric_chip(key_str, self.summary_dict.get(key_str))) is not None
        ]

    def summary_item_list(self) -> list[tuple[str, str]]:
        """All summary fields as ``(label, formatted_value)`` for a detail table."""
        item_list: list[tuple[str, str]] = []
        for key_str, value_obj in self.summary_dict.items():
            item_list.append((key_str.replace("_", " "), _format_metric_value(key_str, value_obj)))
        return item_list


def _format_metric_value(key_str: str, value_obj: object) -> str:
    if isinstance(value_obj, bool):
        return "yes" if value_obj else "no"
    if isinstance(value_obj, (int, float)):
        if key_str in _PERCENT_KEY_SET:
            return f"{value_obj:,.2f}%"
        if abs(float(value_obj)) >= 1000:
            return f"{value_obj:,.0f}"
        return f"{value_obj:,.2f}"
    return str(value_obj)


def _metric_chip(key_str: str, value_obj: object) -> MetricChip | None:
    if not isinstance(value_obj, (int, float)) or isinstance(value_obj, bool):
        return None
    value_float = float(value_obj)
    tone_str = "neutral"
    if key_str == "ann_return_pct":
        tone_str = "pos" if value_float >= 0 else "neg"
    elif key_str == "max_drawdown_pct":
        tone_str = "neg"
    elif key_str == "sharpe":
        tone_str = "pos" if value_float >= 1.0 else ("neg" if value_float < 0 else "neutral")
    return MetricChip(
        label_str=_HEADLINE_LABEL_DICT.get(key_str, key_str),
        value_str=_format_metric_value(key_str, value_obj),
        tone_str=tone_str,
    )


def _read_json_dict(json_path: Path) -> dict:
    try:
        loaded_obj = json.loads(json_path.read_text(encoding="utf-8"))
        return loaded_obj if isinstance(loaded_obj, dict) else {}
    except (OSError, ValueError):
        return {}


def _activity_timestamp_float(
    timestamp_dir_path: Path,
    artifact_path_list: list[Path],
    metadata_dict: dict,
) -> float:
    saved_at_obj = metadata_dict.get("saved_at")
    if isinstance(saved_at_obj, str):
        try:
            return datetime.fromisoformat(saved_at_obj).timestamp()
        except ValueError:
            pass

    try:
        return datetime.strptime(timestamp_dir_path.name, RUN_TIMESTAMP_FORMAT_STR).timestamp()
    except ValueError:
        return max(artifact_path.stat().st_mtime for artifact_path in artifact_path_list)


def _run_sort_key(run_obj: RunEntry) -> tuple[float, str]:
    return (run_obj.effective_activity_timestamp_float, run_obj.timestamp_str)


def _scan_run_entries(name_dir_path: Path, run_name_str: str) -> list[RunEntry]:
    """Scan one ``{run_name}/`` folder into RunEntry rows, newest first."""
    run_entry_list: list[RunEntry] = []
    if not name_dir_path.is_dir():
        return run_entry_list

    for analysis_dir_path in sorted(name_dir_path.iterdir()):
        if not analysis_dir_path.is_dir():
            continue
        analysis_dir_str = analysis_dir_path.name
        for timestamp_dir_path in sorted(analysis_dir_path.iterdir(), reverse=True):
            if not timestamp_dir_path.is_dir():
                continue
            artifact_path_list = [
                artifact_path
                for artifact_path in timestamp_dir_path.iterdir()
                if artifact_path.is_file()
            ]
            # Nested study trees can occupy this position. An actual run leaf
            # has at least one immediate artifact; empty containers do not.
            if not artifact_path_list:
                continue
            report_path = timestamp_dir_path / "report.html"
            metadata_dict = _read_json_dict(timestamp_dir_path / "metadata.json")
            rel_dir_from_results_str = timestamp_dir_path.resolve().relative_to(
                RESULTS_ROOT_PATH
            ).as_posix()
            run_entry_list.append(
                RunEntry(
                    run_name_str=run_name_str,
                    analysis_dir_str=analysis_dir_str,
                    analysis_label_str=ANALYSIS_LABEL_DICT.get(
                        analysis_dir_str, analysis_dir_str.replace("_", " ").title()
                    ),
                    timestamp_str=timestamp_dir_path.name,
                    rel_dir_from_results_str=rel_dir_from_results_str,
                    has_report_bool=report_path.is_file(),
                    activity_timestamp_float=_activity_timestamp_float(
                        timestamp_dir_path,
                        artifact_path_list,
                        metadata_dict,
                    ),
                    summary_dict=_read_json_dict(timestamp_dir_path / "summary.json"),
                    metadata_dict=metadata_dict,
                    run_info_dict=_read_json_dict(timestamp_dir_path / "run_info.json"),
                )
            )

    run_entry_list.sort(key=_run_sort_key, reverse=True)
    return run_entry_list


def _module_import_for_runs(run_entry_list: list[RunEntry]) -> str | None:
    for run_obj in run_entry_list:
        class_module_str = run_obj.metadata_dict.get("class_module")
        if isinstance(class_module_str, str) and class_module_str and class_module_str != "__main__":
            return class_module_str
    return None


@dataclass
class StrategyRunIndex:
    runs_by_module_dict: dict[str, list[RunEntry]]
    runs_by_run_name_dict: dict[str, list[RunEntry]]
    strategy_stem_set: set[str] = field(default_factory=set)

    def runs_for(self, module_import_str: str, stem_str: str) -> list[RunEntry]:
        module_run_entry_list = self.runs_by_module_dict.get(module_import_str, [])
        direct_run_name_entry_list = self.runs_by_run_name_dict.get(stem_str, [])
        owned_module_run_entry_list = [
            run_obj
            for run_obj in module_run_entry_list
            if (
                run_obj.run_name_str == stem_str
                or run_obj.run_name_str not in self.strategy_stem_set
            )
        ]
        combined_run_entry_list: list[RunEntry] = []
        seen_rel_dir_set: set[str] = set()
        for run_obj in [*direct_run_name_entry_list, *owned_module_run_entry_list]:
            if run_obj.rel_dir_from_results_str in seen_rel_dir_set:
                continue
            seen_rel_dir_set.add(run_obj.rel_dir_from_results_str)
            combined_run_entry_list.append(run_obj)
        combined_run_entry_list.sort(key=_run_sort_key, reverse=True)
        return combined_run_entry_list

    def latest_vanilla_for(self, module_import_str: str, stem_str: str) -> RunEntry | None:
        for run_obj in self.runs_for(module_import_str, stem_str):
            if run_obj.analysis_dir_str == "vanilla_backtest" and run_obj.is_indexable_run_bool:
                return run_obj
        return None

    def latest_run_for(self, module_import_str: str, stem_str: str) -> RunEntry | None:
        for run_obj in self.runs_for(module_import_str, stem_str):
            if run_obj.is_indexable_run_bool:
                return run_obj
        return None

    def run_count_for(self, module_import_str: str, stem_str: str) -> int:
        return len(self.runs_for(module_import_str, stem_str))

    def recent_runs(self, limit_int: int = 12) -> list[RunEntry]:
        """Newest result rows that have metrics or an actionable report."""
        recent_run_list = [
            run_obj
            for run_entry_list in self.runs_by_run_name_dict.values()
            for run_obj in run_entry_list
            if run_obj.is_indexable_run_bool and (run_obj.has_report_bool or run_obj.summary_dict)
        ]
        recent_run_list.sort(key=_run_sort_key, reverse=True)
        return recent_run_list[:limit_int]


def build_strategy_run_index(strategy_stem_set: set[str] | None = None) -> StrategyRunIndex:
    """One pass over ``results/research/strategy`` returning runs keyed two ways."""
    runs_by_module_dict: dict[str, list[RunEntry]] = {}
    runs_by_run_name_dict: dict[str, list[RunEntry]] = {}

    if RESEARCH_STRATEGY_ROOT_PATH.is_dir():
        for name_dir_path in sorted(RESEARCH_STRATEGY_ROOT_PATH.iterdir()):
            if not name_dir_path.is_dir():
                continue
            run_name_str = name_dir_path.name
            run_entry_list = _scan_run_entries(name_dir_path, run_name_str)
            if not run_entry_list:
                continue
            runs_by_run_name_dict[run_name_str] = run_entry_list
            module_import_str = _module_import_for_runs(run_entry_list)
            if module_import_str is not None:
                runs_by_module_dict.setdefault(module_import_str, []).extend(run_entry_list)

    for module_import_str, run_entry_list in runs_by_module_dict.items():
        run_entry_list.sort(key=_run_sort_key, reverse=True)

    if strategy_stem_set is None:
        strategy_stem_set = {
            strategy_path.stem for strategy_path in STRATEGIES_ROOT_PATH.rglob("strategy_*.py")
        }

    return StrategyRunIndex(
        runs_by_module_dict=runs_by_module_dict,
        runs_by_run_name_dict=runs_by_run_name_dict,
        strategy_stem_set=set(strategy_stem_set),
    )


def scan_portfolio_runs(*portfolio_name_str_tuple: str) -> list[RunEntry]:
    """Read portfolio runs written under any configured portfolio name.

    PortfolioManager writes results below the YAML ``name_str`` while Bench
    identifies a card by its YAML filename. Most configs use the same value,
    but both are valid result owners when they differ.
    """
    combined_run_entry_list: list[RunEntry] = []
    seen_rel_dir_set: set[str] = set()
    seen_name_set: set[str] = set()
    for portfolio_name_str in portfolio_name_str_tuple:
        if not portfolio_name_str or portfolio_name_str in seen_name_set:
            continue
        seen_name_set.add(portfolio_name_str)
        for run_obj in _scan_run_entries(
            RESEARCH_PORTFOLIO_ROOT_PATH / portfolio_name_str,
            portfolio_name_str,
        ):
            if run_obj.rel_dir_from_results_str in seen_rel_dir_set:
                continue
            seen_rel_dir_set.add(run_obj.rel_dir_from_results_str)
            combined_run_entry_list.append(run_obj)
    combined_run_entry_list.sort(key=_run_sort_key, reverse=True)
    return combined_run_entry_list


class ProducedRunFinder:
    """Resolves "what did this job produce?" against one scan of ``results/``.

    *** CRITICAL*** The scan must be shared across a whole page render. Building
    a fresh index per job turned the Jobs page into a ~49 s render (156 finished
    jobs x a 0.3 s walk of the results tree) on a view that polls every two
    seconds. The index is read once here and reused for every row.

    *** Matched by time, not by identity.*** The runners return no run id to the
    caller, so this is a heuristic: if two jobs for the same target overlap
    (concurrency is 2), the newest run may belong to the other one. It is used
    only to offer a convenience link, never to attribute metrics to a job.
    """

    def __init__(self) -> None:
        self._strategy_run_index_obj: StrategyRunIndex | None = None
        self._portfolio_run_list_by_name_dict: dict[str, list[RunEntry]] = {}

    def _candidate_run_entry_list(self, target_name_str: str, kind_str: str) -> list[RunEntry]:
        if kind_str == "portfolio":
            if target_name_str not in self._portfolio_run_list_by_name_dict:
                self._portfolio_run_list_by_name_dict[target_name_str] = scan_portfolio_runs(
                    target_name_str
                )
            return self._portfolio_run_list_by_name_dict[target_name_str]

        if self._strategy_run_index_obj is None:
            self._strategy_run_index_obj = build_strategy_run_index()
        return self._strategy_run_index_obj.runs_by_run_name_dict.get(target_name_str, [])

    def find_run_produced_after(
        self,
        target_name_str: str,
        started_at_timestamp_float: float,
        kind_str: str,
    ) -> RunEntry | None:
        """Newest report this target wrote after ``started_at``, if any."""
        for run_obj in self._candidate_run_entry_list(target_name_str, kind_str):
            # already newest-first
            if not run_obj.has_report_bool:
                continue
            if run_obj.effective_activity_timestamp_float >= started_at_timestamp_float:
                return run_obj
        return None


def recent_runs(limit_int: int = 12) -> list[RunEntry]:
    """Newest analyzer runs across all strategies, for the home feed."""
    return build_strategy_run_index().recent_runs(limit_int=limit_int)


def resolve_artifact_path(rel_path_str: str) -> Path | None:
    """Resolve a results-relative path, refusing anything outside ``results/``."""
    results_root_resolved_path = RESULTS_ROOT_PATH.resolve()
    candidate_path = (results_root_resolved_path / rel_path_str).resolve()
    try:
        candidate_path.relative_to(results_root_resolved_path)
    except ValueError:
        return None
    if not candidate_path.is_file():
        return None
    return candidate_path
