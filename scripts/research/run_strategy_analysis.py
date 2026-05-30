"""
Run a strategy's research analyses from one command.

This is an orchestration CLI only. It delegates to the existing Vanilla,
FrictionAnalysis, ExecutionTimingAnalyzer, RiskAnalysis, and StressTestAnalyzer
CLIs without changing strategy or execution semantics.
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from strategies.run_strategy import _resolve_strategy_module_import_str
from alpha.engine.stress_test import supported_stress_test_strategy_key_list

ANALYSIS_VANILLA_STR = "vanilla"
ANALYSIS_FRICTION_STR = "friction"
ANALYSIS_TIMING_STR = "timing"
ANALYSIS_RISK_STR = "risk"
ANALYSIS_STRESS_STR = "stress"
DEFAULT_ANALYSIS_TUPLE = (
    ANALYSIS_VANILLA_STR,
    ANALYSIS_FRICTION_STR,
    ANALYSIS_TIMING_STR,
)
SUPPORTED_ANALYSIS_TUPLE = (
    ANALYSIS_VANILLA_STR,
    ANALYSIS_FRICTION_STR,
    ANALYSIS_TIMING_STR,
    ANALYSIS_RISK_STR,
    ANALYSIS_STRESS_STR,
)


@dataclass(frozen=True)
class AnalysisRunResult:
    analysis_str: str
    status_str: str
    elapsed_seconds_float: float
    command_tuple: tuple[str, ...] = ()
    detail_str: str = ""


def _unique_analysis_tuple(raw_analysis_list: list[str] | None) -> tuple[str, ...]:
    if raw_analysis_list is None or len(raw_analysis_list) == 0:
        return DEFAULT_ANALYSIS_TUPLE

    seen_analysis_set: set[str] = set()
    analysis_list: list[str] = []
    for analysis_str in raw_analysis_list:
        if analysis_str in seen_analysis_set:
            continue
        seen_analysis_set.add(analysis_str)
        analysis_list.append(analysis_str)
    return tuple(analysis_list)


def _module_ref_str(strategy_ref_str: str) -> str:
    return str(strategy_ref_str).split(":", maxsplit=1)[0]


def _load_strategy_module(strategy_ref_str: str):
    module_import_str = _resolve_strategy_module_import_str(_module_ref_str(strategy_ref_str))
    strategy_module_obj = importlib.import_module(module_import_str)
    return module_import_str, strategy_module_obj


def _missing_hook_detail_str(strategy_module_obj, analysis_str: str) -> str | None:
    if analysis_str == ANALYSIS_STRESS_STR:
        stress_key_str = _stress_strategy_key_str(strategy_module_obj.__name__)
        if stress_key_str in supported_stress_test_strategy_key_list():
            return None
        return f"unsupported stress strategy key: {stress_key_str}"

    hook_by_analysis_dict = {
        ANALYSIS_VANILLA_STR: "run_variant",
        ANALYSIS_FRICTION_STR: "run_friction_analysis",
        ANALYSIS_TIMING_STR: "build_execution_timing_analysis_inputs",
        ANALYSIS_RISK_STR: "run_variant",
    }
    hook_name_str = hook_by_analysis_dict[analysis_str]
    hook_obj = getattr(strategy_module_obj, hook_name_str, None)
    if callable(hook_obj):
        return None
    return f"missing strategy hook: {hook_name_str}(...)"


def _stress_strategy_key_str(module_import_str: str) -> str:
    return str(module_import_str).rsplit(".", maxsplit=1)[-1]


def _analysis_command_tuple(
    analysis_str: str,
    module_import_str: str,
    output_dir_str: str,
    save_results_bool: bool,
    show_display_bool: bool,
    show_signal_progress_bool: bool,
    performance_warnings_as_errors_bool: bool,
    strategy_kwarg_tuple: tuple[str, ...] = (),
) -> tuple[str, ...]:
    if analysis_str == ANALYSIS_VANILLA_STR:
        command_list = [
            sys.executable,
            str(REPO_ROOT_PATH / "strategies" / "run_strategy.py"),
            module_import_str,
            "--output-dir",
            output_dir_str,
        ]
        if not save_results_bool:
            command_list.append("--no-save")
        if show_display_bool:
            command_list.append("--show-display")
        if performance_warnings_as_errors_bool:
            command_list.append("--performance-warnings-as-errors")
        for strategy_kwarg_str in strategy_kwarg_tuple:
            command_list.extend(["--strategy-kwarg", strategy_kwarg_str])
        return tuple(command_list)

    if analysis_str == ANALYSIS_FRICTION_STR:
        command_list = [
            sys.executable,
            str(REPO_ROOT_PATH / "strategies" / "run_friction_analysis.py"),
            module_import_str,
            "--output-dir",
            output_dir_str,
        ]
        if not save_results_bool:
            command_list.append("--no-save")
        if show_display_bool:
            command_list.append("--show-display")
        return tuple(command_list)

    if analysis_str == ANALYSIS_TIMING_STR:
        command_list = [
            sys.executable,
            str(REPO_ROOT_PATH / "scripts" / "research" / "execution_timing_analyzer.py"),
            module_import_str,
            "--output-dir",
            output_dir_str,
        ]
        if not save_results_bool:
            command_list.append("--no-save")
        if show_signal_progress_bool:
            command_list.append("--show-signal-progress")
        return tuple(command_list)

    if analysis_str == ANALYSIS_RISK_STR:
        command_list = [
            sys.executable,
            str(REPO_ROOT_PATH / "strategies" / "run_risk_analysis.py"),
            module_import_str,
            "--output-dir",
            output_dir_str,
        ]
        if not save_results_bool:
            command_list.append("--no-save")
        if show_display_bool:
            command_list.append("--show-display")
        for strategy_kwarg_str in strategy_kwarg_tuple:
            command_list.extend(["--strategy-kwarg", strategy_kwarg_str])
        return tuple(command_list)

    if analysis_str == ANALYSIS_STRESS_STR:
        command_list = [
            sys.executable,
            str(REPO_ROOT_PATH / "strategies" / "run_stress_test.py"),
            _stress_strategy_key_str(module_import_str),
            "--output-dir",
            output_dir_str,
        ]
        if not save_results_bool:
            command_list.append("--no-save")
        if show_signal_progress_bool:
            command_list.append("--show-signal-progress")
        return tuple(command_list)

    raise ValueError(f"Unsupported analysis_str: {analysis_str}")


def _format_command_str(command_tuple: Sequence[str]) -> str:
    return subprocess.list2cmdline([str(part_str) for part_str in command_tuple])


def _run_command_result(command_tuple: tuple[str, ...], analysis_str: str) -> AnalysisRunResult:
    started_at_float = time.monotonic()
    completed_process_obj = subprocess.run(
        list(command_tuple),
        cwd=REPO_ROOT_PATH,
        check=False,
    )
    elapsed_seconds_float = time.monotonic() - started_at_float
    if completed_process_obj.returncode == 0:
        return AnalysisRunResult(
            analysis_str=analysis_str,
            status_str="PASS",
            elapsed_seconds_float=elapsed_seconds_float,
            command_tuple=command_tuple,
        )
    return AnalysisRunResult(
        analysis_str=analysis_str,
        status_str="FAIL",
        elapsed_seconds_float=elapsed_seconds_float,
        command_tuple=command_tuple,
        detail_str=f"return code {completed_process_obj.returncode}",
    )


def run_strategy_analysis(
    strategy_ref_str: str,
    analysis_tuple: tuple[str, ...] = DEFAULT_ANALYSIS_TUPLE,
    output_dir_str: str = "results",
    save_results_bool: bool = True,
    show_display_bool: bool = False,
    show_signal_progress_bool: bool = False,
    performance_warnings_as_errors_bool: bool = False,
    keep_going_bool: bool = False,
    strategy_kwarg_tuple: tuple[str, ...] = (),
) -> tuple[int, list[AnalysisRunResult]]:
    module_import_str, strategy_module_obj = _load_strategy_module(strategy_ref_str)
    result_list: list[AnalysisRunResult] = []

    print(f"Strategy module: {module_import_str}")
    print(f"Analyses: {', '.join(analysis_tuple)}")
    print("")

    for step_int, analysis_str in enumerate(analysis_tuple, start=1):
        print(f"[{step_int}/{len(analysis_tuple)}] {analysis_str}")
        missing_hook_detail_str = _missing_hook_detail_str(strategy_module_obj, analysis_str)
        if missing_hook_detail_str is not None:
            result_obj = AnalysisRunResult(
                analysis_str=analysis_str,
                status_str="SKIP",
                elapsed_seconds_float=0.0,
                detail_str=missing_hook_detail_str,
            )
            result_list.append(result_obj)
            print(f"SKIP: {missing_hook_detail_str}\n")
            continue

        command_tuple = _analysis_command_tuple(
            analysis_str=analysis_str,
            module_import_str=module_import_str,
            output_dir_str=output_dir_str,
            save_results_bool=save_results_bool,
            show_display_bool=show_display_bool,
            show_signal_progress_bool=show_signal_progress_bool,
            performance_warnings_as_errors_bool=performance_warnings_as_errors_bool,
            strategy_kwarg_tuple=strategy_kwarg_tuple,
        )
        print(f"Command: {_format_command_str(command_tuple)}")
        result_obj = _run_command_result(command_tuple, analysis_str)
        result_list.append(result_obj)
        if result_obj.status_str == "FAIL":
            print(f"FAIL: {result_obj.detail_str}\n")
            if not keep_going_bool:
                break
        else:
            print("PASS\n")

    return_code_int = _return_code_int(result_list)
    _print_summary(result_list)
    return return_code_int, result_list


def _return_code_int(result_list: list[AnalysisRunResult]) -> int:
    if any(result_obj.status_str == "FAIL" for result_obj in result_list):
        return 1
    return 0


def _print_summary(result_list: list[AnalysisRunResult]) -> None:
    print("Summary")
    print("Analysis  Status  Seconds  Detail")
    for result_obj in result_list:
        detail_str = result_obj.detail_str
        elapsed_str = f"{result_obj.elapsed_seconds_float:.1f}"
        print(
            f"{result_obj.analysis_str:<8}  "
            f"{result_obj.status_str:<6}  "
            f"{elapsed_str:>7}  "
            f"{detail_str}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the available research analyses for one strategy module.",
    )
    parser.add_argument(
        "strategy_ref_str",
        help="Strategy module name, full import path, or .py path.",
    )
    parser.add_argument(
        "--analysis",
        action="append",
        choices=SUPPORTED_ANALYSIS_TUPLE,
        help="Analysis to run. Repeat for a subset. Defaults to vanilla, friction, and timing.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Root directory for saved research artifacts.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run without writing report artifacts.",
    )
    parser.add_argument(
        "--show-display",
        action="store_true",
        help="Let strategy and friction analyses print their verbose displays.",
    )
    parser.add_argument(
        "--show-signal-progress",
        action="store_true",
        help="Show signal precompute progress for ExecutionTimingAnalyzer and StressTestAnalyzer.",
    )
    parser.add_argument(
        "--performance-warnings-as-errors",
        action="store_true",
        help="Fail the Vanilla run on pandas PerformanceWarning.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue later analyses after a failed command.",
    )
    parser.add_argument(
        "--strategy-kwarg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Extra run_variant kwarg forwarded to the Vanilla run. Repeat as needed. "
            "Useful for research-only data inputs such as market_cap_csv_path_str=..."
        ),
    )
    arg_namespace = parser.parse_args()

    return_code_int, _result_list = run_strategy_analysis(
        strategy_ref_str=arg_namespace.strategy_ref_str,
        analysis_tuple=_unique_analysis_tuple(arg_namespace.analysis),
        output_dir_str=arg_namespace.output_dir,
        save_results_bool=not arg_namespace.no_save,
        show_display_bool=arg_namespace.show_display,
        show_signal_progress_bool=arg_namespace.show_signal_progress,
        performance_warnings_as_errors_bool=arg_namespace.performance_warnings_as_errors,
        keep_going_bool=arg_namespace.keep_going,
        strategy_kwarg_tuple=tuple(arg_namespace.strategy_kwarg),
    )
    raise SystemExit(return_code_int)


if __name__ == "__main__":
    main()
