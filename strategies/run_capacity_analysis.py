"""Run a strategy CapacityAnalysis study across an AUM grid."""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from pathlib import Path


REPO_ROOT_PATH = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from alpha.engine.capacity_analysis import (
    DEFAULT_AUM_GRID_TUPLE,
    CapacityAnalysis,
    CapacityStudyResult,
    build_capacity_study_result,
)
from strategies.run_strategy import _resolve_strategy_module_import_str


def _run_capacity_analysis_module(
    strategy_name_str: str,
    aum_float_tuple: tuple[float, ...] = DEFAULT_AUM_GRID_TUPLE,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    show_display_bool: bool = False,
    dry_run_bool: bool = False,
    backtest_start_date_str: str | None = None,
    end_date_str: str | None = None,
) -> CapacityStudyResult | None:
    module_import_str = _resolve_strategy_module_import_str(strategy_name_str)
    if dry_run_bool:
        print(f"Resolved strategy module: {module_import_str}")
        return None

    strategy_module_obj = importlib.import_module(module_import_str)
    build_inputs_fn = getattr(strategy_module_obj, "build_capacity_analysis_inputs", None)
    if not callable(build_inputs_fn):
        raise AttributeError(
            f"Module '{module_import_str}' does not expose "
            "build_capacity_analysis_inputs(...)."
        )
    if "capital_base_float" not in inspect.signature(build_inputs_fn).parameters:
        raise TypeError(
            "build_capacity_analysis_inputs(...) must accept capital_base_float so "
            "each AUM point is a full strategy rerun."
        )

    normalized_aum_float_tuple = _normalize_aum_float_tuple(aum_float_tuple)
    run_result_list = []
    for index_int, capital_base_float in enumerate(normalized_aum_float_tuple, start=1):
        print(
            f"[{index_int}/{len(normalized_aum_float_tuple)}] "
            f"Running {_format_dollar_str(capital_base_float)}"
        )
        input_kwarg_dict = _supported_input_kwarg_dict(
            build_inputs_fn=build_inputs_fn,
            show_display_bool=show_display_bool,
            backtest_start_date_str=backtest_start_date_str,
            capital_base_float=capital_base_float,
            end_date_str=end_date_str,
        )
        capacity_input_dict = build_inputs_fn(**input_kwarg_dict)
        required_key_set = {
            "strategy_obj",
            "pricing_data_df",
            "execution_policy_str",
        }
        missing_key_list = sorted(required_key_set.difference(capacity_input_dict))
        if missing_key_list:
            raise ValueError(
                "build_capacity_analysis_inputs(...) is missing keys: "
                f"{missing_key_list}."
            )
        actual_capital_float = float(capacity_input_dict["strategy_obj"]._capital_base)
        if not abs(actual_capital_float - capital_base_float) <= max(
            1e-9,
            abs(capital_base_float) * 1e-12,
        ):
            raise ValueError(
                "build_capacity_analysis_inputs(...) returned capital_base="
                f"{actual_capital_float:g}, expected {capital_base_float:g}."
            )
        run_result_list.append(
            CapacityAnalysis(
                strategy_obj=capacity_input_dict["strategy_obj"],
                pricing_data_df=capacity_input_dict["pricing_data_df"],
                execution_policy_str=capacity_input_dict["execution_policy_str"],
                impact_profile_str=capacity_input_dict.get("impact_profile_str"),
            ).run()
        )

    study_result_obj = build_capacity_study_result(
        run_result_list,
        output_dir_str=output_dir_str,
        save_output_bool=save_results_bool,
    )
    _print_capacity_summary(study_result_obj)
    return study_result_obj


def _supported_input_kwarg_dict(
    build_inputs_fn,
    show_display_bool: bool,
    backtest_start_date_str: str | None,
    capital_base_float: float,
    end_date_str: str | None,
) -> dict[str, object]:
    signature_obj = inspect.signature(build_inputs_fn)
    candidate_kwarg_dict = {
        "show_display_bool": show_display_bool,
        "backtest_start_date_str": backtest_start_date_str,
        "capital_base_float": capital_base_float,
        "end_date_str": end_date_str,
    }
    return {
        key_str: value_obj
        for key_str, value_obj in candidate_kwarg_dict.items()
        if key_str in signature_obj.parameters
        and (value_obj is not None or key_str == "show_display_bool")
    }


def _normalize_aum_float_tuple(aum_float_tuple: tuple[float, ...]) -> tuple[float, ...]:
    normalized_tuple = tuple(sorted({float(value_float) for value_float in aum_float_tuple}))
    if not normalized_tuple or any(value_float <= 0.0 for value_float in normalized_tuple):
        raise ValueError("AUM values must be positive and non-empty.")
    return normalized_tuple


def _print_capacity_summary(study_result_obj: CapacityStudyResult) -> None:
    summary_dict = study_result_obj.summary_dict
    print("\nCapacityAnalysis summary:")
    print(f"  Strategy: {study_result_obj.strategy_name_str}")
    print(f"  Execution policy: {study_result_obj.execution_policy_str}")
    print(f"  Impact profile: {study_result_obj.impact_profile_str or 'MOC default'}")
    print(
        "  Optimal capacity: "
        f"{_format_optional_dollar_str(summary_dict.get('optimal_capacity_float'))}"
    )
    print(
        "  Recommended capacity: "
        f"{_format_optional_dollar_str(summary_dict.get('recommended_capacity_float'))}"
    )
    print(
        "  Outer capacity: "
        f"{_format_optional_dollar_str(summary_dict.get('outer_capacity_float'))}"
    )
    print(
        "  Break-even capacity: "
        f"{summary_dict.get('break_even_capacity_bracket_str', 'N/A')}"
    )
    if study_result_obj.output_dir_path is not None:
        print(f"  Report folder: {study_result_obj.output_dir_path.resolve()}")


def _format_dollar_str(value_float: float) -> str:
    return f"${float(value_float):,.0f}"


def _format_optional_dollar_str(value_obj) -> str:
    if value_obj is None:
        return "N/A"
    try:
        return _format_dollar_str(float(value_obj))
    except (TypeError, ValueError):
        return "N/A"


def main() -> None:
    parser_obj = argparse.ArgumentParser()
    parser_obj.add_argument(
        "strategy_name_str",
        help="Strategy module name, full import path, or .py path.",
    )
    parser_obj.add_argument(
        "--aum",
        dest="aum_float_list",
        action="append",
        type=float,
        default=None,
        help="AUM grid value. Repeat to override the default grid.",
    )
    parser_obj.add_argument("--output-dir", default="results")
    parser_obj.add_argument("--no-save", action="store_true")
    parser_obj.add_argument("--show-display", action="store_true")
    parser_obj.add_argument("--dry-run", action="store_true")
    parser_obj.add_argument("--backtest-start-date", default=None)
    parser_obj.add_argument("--end-date", default=None)
    arg_namespace = parser_obj.parse_args()
    aum_float_tuple = (
        DEFAULT_AUM_GRID_TUPLE
        if arg_namespace.aum_float_list is None
        else tuple(arg_namespace.aum_float_list)
    )
    _run_capacity_analysis_module(
        strategy_name_str=arg_namespace.strategy_name_str,
        aum_float_tuple=aum_float_tuple,
        save_results_bool=not arg_namespace.no_save,
        output_dir_str=arg_namespace.output_dir,
        show_display_bool=arg_namespace.show_display,
        dry_run_bool=arg_namespace.dry_run,
        backtest_start_date_str=arg_namespace.backtest_start_date,
        end_date_str=arg_namespace.end_date,
    )


if __name__ == "__main__":
    main()
