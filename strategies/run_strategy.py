"""
Run a strategy module by name.

Usage:
    uv run python strategies/run_strategy.py strategy_taa_df_btal_fallback_tqqq_vix_cash
    uv run python strategies/run_strategy.py strategy_taa_df_btal_fallback_tqqq_vix_cash.py
    uv run python strategies/run_strategy.py strategy_taa_df_btal_fallback_tqqq_vix_cash --no-save
    uv run python strategies/run_strategy.py strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash

The selected module must expose:

    run_variant(show_display_bool, save_results_bool, output_dir_str)
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys
import warnings
from pathlib import Path

import pandas as pd


REPO_ROOT_PATH = Path(__file__).resolve().parents[1]
STRATEGIES_ROOT_PATH = REPO_ROOT_PATH / "strategies"

if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))


def _module_path_to_import_str(module_path: Path) -> str:
    relative_module_path = module_path.resolve().relative_to(REPO_ROOT_PATH)
    return ".".join(relative_module_path.with_suffix("").parts)


def _resolve_strategy_module_import_str(strategy_name_str: str) -> str:
    """
    Resolve a strategy argument into a Python import path.

    Accepted forms:
        strategy_taa_df_btal_fallback_tqqq_vix_cash
        strategy_taa_df_btal_fallback_tqqq_vix_cash.py
        taa_df/strategy_taa_df_btal_fallback_tqqq_vix_cash.py
        strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash
        strategies/taa_df/strategy_taa_df_btal_fallback_tqqq_vix_cash.py
    """
    strategy_path = Path(strategy_name_str)
    if strategy_path.suffix == ".py":
        candidate_path_list = []
        if strategy_path.is_absolute():
            candidate_path_list.append(strategy_path.resolve())
        else:
            candidate_path_list.append((REPO_ROOT_PATH / strategy_path).resolve())
            candidate_path_list.append((STRATEGIES_ROOT_PATH / strategy_path).resolve())

        for candidate_path in candidate_path_list:
            if candidate_path.exists():
                return _module_path_to_import_str(candidate_path)

        matching_module_path_list = sorted(STRATEGIES_ROOT_PATH.rglob(strategy_path.name))
        if len(matching_module_path_list) == 0:
            searched_path_str = "\n".join(str(candidate_path) for candidate_path in candidate_path_list)
            raise FileNotFoundError(
                f"Could not find strategy file '{strategy_name_str}'. Searched:\n{searched_path_str}"
            )
        if len(matching_module_path_list) > 1:
            match_str = "\n".join(str(module_path) for module_path in matching_module_path_list)
            raise RuntimeError(
                f"Strategy file name '{strategy_name_str}' is ambiguous. Matches:\n{match_str}"
            )

        return _module_path_to_import_str(matching_module_path_list[0])

    if "." in strategy_name_str:
        return strategy_name_str

    matching_module_path_list = sorted(STRATEGIES_ROOT_PATH.rglob(f"{strategy_name_str}.py"))
    if len(matching_module_path_list) == 0:
        raise FileNotFoundError(
            f"Could not find strategy module '{strategy_name_str}' under {STRATEGIES_ROOT_PATH}."
        )
    if len(matching_module_path_list) > 1:
        match_str = "\n".join(str(module_path) for module_path in matching_module_path_list)
        raise RuntimeError(
            f"Strategy module name '{strategy_name_str}' is ambiguous. Matches:\n{match_str}"
        )

    return _module_path_to_import_str(matching_module_path_list[0])


def _run_strategy_module(
    strategy_name_str: str,
    show_display_bool: bool,
    save_results_bool: bool,
    output_dir_str: str,
    dry_run_bool: bool,
):
    module_import_str = _resolve_strategy_module_import_str(strategy_name_str)
    strategy_module = importlib.import_module(module_import_str)

    if dry_run_bool:
        print(f"Resolved strategy module: {module_import_str}")
        return None

    run_variant_fn = getattr(strategy_module, "run_variant", None)
    if run_variant_fn is None:
        raise AttributeError(
            f"Module '{module_import_str}' does not expose run_variant(...). "
            "This generic runner only supports strategy modules with run_variant."
        )

    signature_obj = inspect.signature(run_variant_fn)
    run_kwarg_dict = {}
    if "show_display_bool" in signature_obj.parameters:
        run_kwarg_dict["show_display_bool"] = show_display_bool
    if "save_results_bool" in signature_obj.parameters:
        run_kwarg_dict["save_results_bool"] = save_results_bool
    if "output_dir_str" in signature_obj.parameters:
        run_kwarg_dict["output_dir_str"] = output_dir_str

    strategy_obj = run_variant_fn(**run_kwarg_dict)
    print(f"Ran strategy module: {module_import_str}")
    _print_strategy_summary(strategy_obj)
    return strategy_obj


def _print_strategy_summary(strategy_obj) -> None:
    summary_df = getattr(strategy_obj, "summary", None)
    if summary_df is not None and len(summary_df) > 0:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        print("\nSummary:")
        print(summary_df.to_string())

    realized_weight_df = getattr(strategy_obj, "realized_weight_df", None)
    if realized_weight_df is not None and len(realized_weight_df) > 0:
        print("\nLast realized weights:")
        print(realized_weight_df.tail(3).to_string())


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
        help="Only resolve and import the strategy module.",
    )
    parser.add_argument(
        "--performance-warnings-as-errors",
        action="store_true",
        help="Fail the run on pandas PerformanceWarning.",
    )
    arg_namespace = parser.parse_args()

    if arg_namespace.performance_warnings_as_errors:
        warnings.simplefilter("error", pd.errors.PerformanceWarning)

    _run_strategy_module(
        strategy_name_str=arg_namespace.strategy_name_str,
        show_display_bool=arg_namespace.show_display,
        save_results_bool=not arg_namespace.no_save,
        output_dir_str=arg_namespace.output_dir,
        dry_run_bool=arg_namespace.dry_run,
    )


if __name__ == "__main__":
    main()
