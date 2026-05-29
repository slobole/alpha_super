"""
Run a strategy module's RiskAnalysis by name.

Usage:
    uv run python strategies/run_risk_analysis.py strategy_mr_dv2.py
    uv run python strategies/run_risk_analysis.py dv2/strategy_mr_dv2.py
    uv run python strategies/run_risk_analysis.py strategies.dv2.strategy_mr_dv2

The selected module must expose run_variant(...). RiskAnalysis intentionally
runs vanilla first with save_results_bool=False and analyzes the realized
post-run return path.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT_PATH = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from alpha.engine.risk_analysis import (
    DEFAULT_CONFIDENCE_LEVEL_FLOAT,
    DEFAULT_PRIMARY_MEAN_BLOCK_LENGTH_INT,
    DEFAULT_RANDOM_SEED_INT,
    DEFAULT_SENSITIVITY_BLOCK_LENGTH_TUPLE,
    DEFAULT_SIMULATION_COUNT_INT,
    RiskAnalysis,
)
from strategies.run_strategy import (
    _parse_run_kwarg_tuple,
    _resolve_strategy_module_import_str,
    _run_strategy_module,
)


def _run_risk_analysis_module(
    strategy_name_str: str,
    save_results_bool: bool,
    output_dir_str: str,
    show_display_bool: bool,
    dry_run_bool: bool,
    simulation_count_int: int,
    primary_mean_block_length_int: int,
    mean_block_length_tuple: tuple[int, ...],
    confidence_level_float: float,
    random_seed_int: int,
    strategy_kwarg_dict: dict[str, object] | None = None,
):
    module_import_str = _resolve_strategy_module_import_str(strategy_name_str)
    if dry_run_bool:
        print(f"Resolved strategy module: {module_import_str}")
        return None

    strategy_obj = _run_strategy_module(
        strategy_name_str=module_import_str,
        show_display_bool=show_display_bool,
        save_results_bool=False,
        output_dir_str=output_dir_str,
        dry_run_bool=False,
        strategy_kwarg_dict=strategy_kwarg_dict,
    )
    risk_analysis_obj = RiskAnalysis(
        strategy_obj,
        source_strategy_ref_str=module_import_str,
        output_dir_str=output_dir_str,
        save_output_bool=save_results_bool,
        primary_mean_block_length_int=primary_mean_block_length_int,
        mean_block_length_tuple=mean_block_length_tuple,
        simulation_count_int=simulation_count_int,
        random_seed_int=random_seed_int,
        confidence_level_float=confidence_level_float,
    )
    risk_result_obj = risk_analysis_obj.run()
    _print_risk_analysis_summary(risk_result_obj)
    return risk_result_obj


def _print_risk_analysis_summary(risk_result_obj) -> None:
    if risk_result_obj is None:
        return
    summary_dict = risk_result_obj.summary_dict
    print(f"Ran RiskAnalysis: {risk_result_obj.strategy_name_str}")
    output_dir_path = getattr(risk_result_obj, "output_dir_path", None)
    if output_dir_path is not None:
        print(f"Report folder: {Path(output_dir_path).resolve()}")
    print(f"  Returns: {summary_dict.get('return_count_int', 'N/A')}")
    print(f"  Primary block length: {summary_dict.get('primary_mean_block_length_int', 'N/A')}")
    print(f"  Simulations: {summary_dict.get('simulation_count_int', 'N/A')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "strategy_name_str",
        help="Strategy module name, full import path, or .py path.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Root directory for saved reports.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run without writing report artifacts.",
    )
    parser.add_argument(
        "--show-display",
        action="store_true",
        help="Let the strategy print its own verbose diagnostic displays.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only resolve the strategy module.",
    )
    parser.add_argument(
        "--simulation-count",
        type=int,
        default=DEFAULT_SIMULATION_COUNT_INT,
        help="Bootstrap path count per block length.",
    )
    parser.add_argument(
        "--primary-block-length",
        type=int,
        default=DEFAULT_PRIMARY_MEAN_BLOCK_LENGTH_INT,
        help="Primary stationary-bootstrap expected block length.",
    )
    parser.add_argument(
        "--block-length",
        action="append",
        type=int,
        default=[],
        help="Stationary-bootstrap expected block length. Repeat for sensitivity.",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=DEFAULT_CONFIDENCE_LEVEL_FLOAT,
        help="Empirical confidence interval level.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED_INT,
        help="Bootstrap random seed.",
    )
    parser.add_argument(
        "--strategy-kwarg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra run_variant kwarg. Repeat as needed.",
    )
    arg_namespace = parser.parse_args()
    block_length_tuple = tuple(arg_namespace.block_length or DEFAULT_SENSITIVITY_BLOCK_LENGTH_TUPLE)

    _run_risk_analysis_module(
        strategy_name_str=arg_namespace.strategy_name_str,
        save_results_bool=not arg_namespace.no_save,
        output_dir_str=arg_namespace.output_dir,
        show_display_bool=arg_namespace.show_display,
        dry_run_bool=arg_namespace.dry_run,
        simulation_count_int=arg_namespace.simulation_count,
        primary_mean_block_length_int=arg_namespace.primary_block_length,
        mean_block_length_tuple=block_length_tuple,
        confidence_level_float=arg_namespace.confidence_level,
        random_seed_int=arg_namespace.random_seed,
        strategy_kwarg_dict=_parse_run_kwarg_tuple(arg_namespace.strategy_kwarg),
    )


if __name__ == "__main__":
    main()
