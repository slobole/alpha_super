"""Reader for the ``results/`` artifact tree.

Every analyzer run already writes a tidy, timestamped folder:

    results/research/strategy/{run_name}/{analysis}/{YYYY-MM-DD_HHMMSS}/
        summary.json   metadata.json   report.html   ...

    results/research/portfolio/{name}/{analysis}/{YYYY-MM-DD_HHMMSS}/
        summary.json   report.html     ...

Bench just lists those folders and reads the small JSON sidecars. It writes
nothing here. Linking a results folder back to a catalog strategy uses the
``class_module`` field in ``metadata.json`` (robust to ``*_research`` run-name
suffixes), and falls back to matching the run-name to the file stem.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
RESULTS_ROOT_PATH = REPO_ROOT_PATH / "results"
RESEARCH_STRATEGY_ROOT_PATH = RESULTS_ROOT_PATH / "research" / "strategy"
RESEARCH_PORTFOLIO_ROOT_PATH = RESULTS_ROOT_PATH / "research" / "portfolio"

ANALYSIS_LABEL_DICT: dict[str, str] = {
    "vanilla_backtest": "Vanilla",
    "friction_analysis": "Friction",
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
    summary_dict: dict = field(default_factory=dict)
    metadata_dict: dict = field(default_factory=dict)

    @property
    def display_timestamp_str(self) -> str:
        # "2026-05-20_160858" -> "2026-05-20 16:08:58"
        if "_" in self.timestamp_str and len(self.timestamp_str) >= 15:
            date_part_str, _, time_part_str = self.timestamp_str.partition("_")
            if len(time_part_str) == 6 and time_part_str.isdigit():
                return f"{date_part_str} {time_part_str[0:2]}:{time_part_str[2:4]}:{time_part_str[4:6]}"
        return self.timestamp_str

    @property
    def report_artifact_str(self) -> str:
        return f"{self.rel_dir_from_results_str}/report.html"

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
            report_path = timestamp_dir_path / "report.html"
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
                    summary_dict=_read_json_dict(timestamp_dir_path / "summary.json"),
                    metadata_dict=_read_json_dict(timestamp_dir_path / "metadata.json"),
                )
            )

    run_entry_list.sort(key=lambda run_obj: run_obj.timestamp_str, reverse=True)
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

    def runs_for(self, module_import_str: str, stem_str: str) -> list[RunEntry]:
        run_entry_list = self.runs_by_module_dict.get(module_import_str)
        if run_entry_list:
            return run_entry_list
        # Fall back to a direct run-name match (covers runs with no metadata).
        return self.runs_by_run_name_dict.get(stem_str, [])

    def latest_vanilla_for(self, module_import_str: str, stem_str: str) -> RunEntry | None:
        for run_obj in self.runs_for(module_import_str, stem_str):
            if run_obj.analysis_dir_str == "vanilla_backtest":
                return run_obj
        return None

    def run_count_for(self, module_import_str: str, stem_str: str) -> int:
        return len(self.runs_for(module_import_str, stem_str))


def build_strategy_run_index() -> StrategyRunIndex:
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
        run_entry_list.sort(key=lambda run_obj: run_obj.timestamp_str, reverse=True)

    return StrategyRunIndex(
        runs_by_module_dict=runs_by_module_dict,
        runs_by_run_name_dict=runs_by_run_name_dict,
    )


def scan_portfolio_runs(portfolio_name_str: str) -> list[RunEntry]:
    return _scan_run_entries(RESEARCH_PORTFOLIO_ROOT_PATH / portfolio_name_str, portfolio_name_str)


def recent_runs(limit_int: int = 12) -> list[RunEntry]:
    """Newest analyzer runs across all strategies, for the home feed."""
    all_run_list: list[RunEntry] = []
    index_obj = build_strategy_run_index()
    for run_entry_list in index_obj.runs_by_run_name_dict.values():
        all_run_list.extend(run_entry_list)
    all_run_list.sort(key=lambda run_obj: run_obj.timestamp_str, reverse=True)
    return all_run_list[:limit_int]


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
