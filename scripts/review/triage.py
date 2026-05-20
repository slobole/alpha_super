from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Sequence


@dataclass(frozen=True)
class PathClassification:
    path_str: str
    tier_int: int
    reason_str: str


@dataclass(frozen=True)
class TriageResult:
    tier_int: int
    tier_label_str: str
    live_impact_checklist_required_bool: bool
    required_agent_tuple: tuple[str, ...]
    suggested_test_target_tuple: tuple[str, ...]
    path_classification_tuple: tuple[PathClassification, ...]


TIER_LABEL_BY_INT = {
    0: "Tier 0 - docs/tooling/comments",
    1: "Tier 1 - research/backtest",
    2: "Tier 2 - engine/shared utilities",
    3: "Tier 3 - live/orders/reconcile/released configs",
}

AGENT_BY_TIER = {
    0: (),
    1: ("quant-pitfalls",),
    2: ("quant-pitfalls", "parity", "coverage"),
    3: ("parity", "failure-modes", "coverage"),
}

LIVE_TEST_TARGET_TUPLE = (
    "tests/test_live_runner.py",
    "tests/test_live_scheduler_service.py",
    "tests/test_live_reconcile.py",
    "tests/test_live_release_manifest.py",
)

ENGINE_TEST_TARGET_TUPLE = (
    "tests/test_run_daily_runtime_controls.py",
    "tests/test_execution_timing.py",
    "tests/test_indicators_public_api.py",
)

RESEARCH_TEST_TARGET_TUPLE = (
    "tests/test_strategy_*.py",
    "tests/test_run_strategy_analysis.py",
)

TOOLING_TEST_TARGET_TUPLE = (
    "git diff --check",
    "targeted docs/tooling tests",
)


def normalize_path_str(path_str: str) -> str:
    normalized_path_str = path_str.strip().replace("\\", "/")
    while normalized_path_str.startswith("./"):
        normalized_path_str = normalized_path_str[2:]
    return normalized_path_str


def _path_name_str(path_str: str) -> str:
    return PurePosixPath(path_str).name


def _is_live_path_bool(path_str: str) -> bool:
    path_name_str = _path_name_str(path_str).upper()
    return (
        path_str.startswith("alpha/live/")
        or path_str.startswith("docs/live/")
        or path_str.startswith("docs/reviews/live/")
        or path_str.startswith("tests/test_live_")
        or path_name_str.startswith("LIVE_")
        or path_name_str in {"INCUBATION_FLOW.MD", "LIVE_START_HERE.MD"}
    )


def _is_engine_or_shared_path_bool(path_str: str) -> bool:
    path_name_str = _path_name_str(path_str)
    return (
        path_str.startswith("alpha/engine/")
        or path_str.startswith("alpha/data/")
        or path_str.startswith("data/")
        or path_name_str
        in {
            "indicators.py",
            "metrics.py",
            "portfolio.py",
            "portfolio_manager.py",
        }
        or path_str.startswith("tests/test_execution_timing")
        or path_str.startswith("tests/test_run_daily")
        or path_str.startswith("tests/test_fast_indicators")
        or path_str.startswith("tests/test_indicators")
        or path_str.startswith("tests/test_friction_analysis")
    )


def _is_research_or_backtest_path_bool(path_str: str) -> bool:
    path_name_str = _path_name_str(path_str)
    return (
        path_str.startswith("strategies/")
        or path_str.startswith("notebooks/")
        or path_str.startswith("research/")
        or path_str.startswith("scripts/research/")
        or path_str.startswith("results/research/")
        or path_str.startswith("tests/test_strategy_")
        or path_name_str.endswith(".ipynb")
        or path_name_str == "archive_research_results.py"
        or path_name_str == "test_archive_research_results.py"
    )


def classify_path(path_str: str) -> PathClassification:
    normalized_path_str = normalize_path_str(path_str)
    if _is_live_path_bool(normalized_path_str):
        return PathClassification(
            path_str=normalized_path_str,
            tier_int=3,
            reason_str="live execution, operator, release, or live-test surface",
        )
    if _is_engine_or_shared_path_bool(normalized_path_str):
        return PathClassification(
            path_str=normalized_path_str,
            tier_int=2,
            reason_str="engine, shared utility, indicator, metrics, or execution-sensitive surface",
        )
    if _is_research_or_backtest_path_bool(normalized_path_str):
        return PathClassification(
            path_str=normalized_path_str,
            tier_int=1,
            reason_str="strategy, notebook, research, or backtest-only surface",
        )
    return PathClassification(
        path_str=normalized_path_str,
        tier_int=0,
        reason_str="docs, comments, isolated tooling, or unclassified low-blast-radius path",
    )


def classify_path_list(path_list: Sequence[str]) -> TriageResult:
    classification_tuple = tuple(
        classify_path(path_str)
        for path_str in path_list
        if normalize_path_str(path_str)
    )
    tier_int = max(
        (classification.tier_int for classification in classification_tuple),
        default=0,
    )
    return TriageResult(
        tier_int=tier_int,
        tier_label_str=TIER_LABEL_BY_INT[tier_int],
        live_impact_checklist_required_bool=tier_int == 3,
        required_agent_tuple=_required_agent_tuple(classification_tuple, tier_int),
        suggested_test_target_tuple=_suggested_test_target_tuple(tier_int),
        path_classification_tuple=classification_tuple,
    )


def _required_agent_tuple(
    classification_tuple: Sequence[PathClassification],
    tier_int: int,
) -> tuple[str, ...]:
    agent_list = list(AGENT_BY_TIER[tier_int])
    if tier_int == 3 and any(
        classification.tier_int in {1, 2} for classification in classification_tuple
    ):
        agent_list.insert(0, "quant-pitfalls")
    return tuple(dict.fromkeys(agent_list))


def _suggested_test_target_tuple(tier_int: int) -> tuple[str, ...]:
    if tier_int == 3:
        return LIVE_TEST_TARGET_TUPLE
    if tier_int == 2:
        return ENGINE_TEST_TARGET_TUPLE
    if tier_int == 1:
        return RESEARCH_TEST_TARGET_TUPLE
    return TOOLING_TEST_TARGET_TUPLE


def _read_git_path_list(command_list: Sequence[str]) -> list[str]:
    completed_process = subprocess.run(
        command_list,
        capture_output=True,
        check=False,
        text=True,
    )
    if completed_process.returncode != 0:
        stderr_str = completed_process.stderr.strip()
        raise RuntimeError(stderr_str or f"{' '.join(command_list)} failed")
    return [
        path_str
        for path_str in completed_process.stdout.splitlines()
        if path_str.strip()
    ]


def _dedupe_path_list(path_list: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(path_list))


def read_git_diff_name_only(base_str: str | None = None) -> list[str]:
    diff_command_list = ["git", "diff", "--name-only"]
    if base_str:
        diff_command_list.append(base_str)
    tracked_path_list = _read_git_path_list(diff_command_list)
    untracked_path_list = _read_git_path_list(
        ["git", "ls-files", "--others", "--exclude-standard"]
    )
    return _dedupe_path_list([*tracked_path_list, *untracked_path_list])


def format_triage_result(result: TriageResult) -> str:
    line_list = [
        "Post-Change Verification Triage",
        f"tier_int={result.tier_int}",
        f"tier_label={result.tier_label_str}",
        "live_impact_checklist_required_bool="
        f"{result.live_impact_checklist_required_bool}",
        "required_agent_list:",
    ]
    if result.required_agent_tuple:
        line_list.extend(f"- {agent_str}" for agent_str in result.required_agent_tuple)
    else:
        line_list.append("- none")

    line_list.append("reason_list:")
    if result.path_classification_tuple:
        line_list.extend(
            "- "
            f"{classification.path_str} -> "
            f"Tier {classification.tier_int}: {classification.reason_str}"
            for classification in result.path_classification_tuple
        )
    else:
        line_list.append("- no changed paths -> Tier 0: nothing to classify")

    line_list.append("suggested_test_target_list:")
    line_list.extend(
        f"- {target_str}" for target_str in result.suggested_test_target_tuple
    )

    line_list.append("final_response_required_fields:")
    line_list.extend(
        [
            "- tier",
            "- agents used",
            "- findings fixed",
            "- tests run",
            "- residual risk",
        ]
    )
    return "\n".join(line_list)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify changed files by post-change verification tier.",
    )
    parser.add_argument(
        "--base",
        default=None,
        help="Optional git diff base, for example HEAD or origin/main.",
    )
    parser.add_argument(
        "--name-only",
        nargs="*",
        default=None,
        metavar="PATH",
        help="Classify these paths instead of reading git diff --name-only.",
    )
    return parser


def main(argv_list: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv_list)
    try:
        path_list = (
            list(args.name_only)
            if args.name_only is not None
            else read_git_diff_name_only(args.base)
        )
    except RuntimeError as exc:
        print(f"triage_error={exc}", file=sys.stderr)
        return 2

    result = classify_path_list(path_list)
    print(format_triage_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
