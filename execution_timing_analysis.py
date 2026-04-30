"""
CLI runner for ExecutionTimingAnalysis.

Formula
-------
For each order-intent bar t:

    entry_fill = t + entry_lag at entry_price_field
    exit_fill  = t + exit_lag  at exit_price_field

The strategy module must expose:

    build_execution_timing_analysis_inputs() -> dict

with:

    strategy_factory_fn
    pricing_data_df
    calendar_idx
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

import pandas as pd

from alpha.engine.execution_timing import ExecutionTimingAnalysis, SUPPORTED_TIMING_MODE_TUPLE


def _timing_mode_tuple(
    raw_timing_list: list[str] | None,
    default_timing_tuple: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...] | None:
    if raw_timing_list is None or len(raw_timing_list) == 0:
        if default_timing_tuple is None:
            return None
        return tuple(default_timing_tuple)
    return tuple(raw_timing_list)


def _strategy_module_name_str(raw_strategy_ref_str: str) -> str:
    """
    Accept either a dotted module path or a repo-relative Python file path.

    Examples:

        strategies.taa_df.example
        strategies/taa_df/example.py
        .\\strategies\\taa_df\\example.py
    """
    normalized_strategy_ref_str = str(raw_strategy_ref_str).strip()
    path_like_bool = (
        normalized_strategy_ref_str.endswith(".py")
        or "\\" in normalized_strategy_ref_str
        or "/" in normalized_strategy_ref_str
    )
    if not path_like_bool:
        return normalized_strategy_ref_str

    strategy_path = Path(normalized_strategy_ref_str)
    if not strategy_path.is_absolute():
        strategy_path = Path.cwd() / strategy_path
    strategy_path = strategy_path.resolve()

    if strategy_path.suffix != ".py":
        raise ValueError(f"Strategy file path must end with .py: {raw_strategy_ref_str}")

    repo_root_path = Path(__file__).resolve().parent
    try:
        relative_strategy_path = strategy_path.relative_to(repo_root_path)
    except ValueError as exc:
        raise ValueError(
            f"Strategy path must be inside repo root {repo_root_path}: {strategy_path}"
        ) from exc

    return ".".join(relative_strategy_path.with_suffix("").parts)


def _load_strategy_input_dict(strategy_module_str: str) -> dict[str, Any]:
    resolved_strategy_module_str = _strategy_module_name_str(strategy_module_str)
    strategy_module_obj = importlib.import_module(resolved_strategy_module_str)
    if not hasattr(strategy_module_obj, "build_execution_timing_analysis_inputs"):
        raise AttributeError(
            f"{resolved_strategy_module_str} must expose build_execution_timing_analysis_inputs()."
        )

    strategy_input_dict = strategy_module_obj.build_execution_timing_analysis_inputs()
    required_key_tuple = ("strategy_factory_fn", "pricing_data_df", "calendar_idx")
    missing_key_list = [
        key_str for key_str in required_key_tuple if key_str not in strategy_input_dict
    ]
    if len(missing_key_list) > 0:
        raise KeyError(
            "build_execution_timing_analysis_inputs() is missing keys: "
            f"{missing_key_list}"
        )
    return strategy_input_dict


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an entry/exit execution timing matrix for one strategy module.",
    )
    parser.add_argument(
        "strategy_module_str",
        help="Dotted module path, e.g. strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash.",
    )
    parser.add_argument(
        "--entry-timing",
        action="append",
        choices=SUPPORTED_TIMING_MODE_TUPLE,
        help="Entry timing mode. Repeat to run a subset. Defaults to the strategy's analysis set.",
    )
    parser.add_argument(
        "--exit-timing",
        action="append",
        choices=SUPPORTED_TIMING_MODE_TUPLE,
        help="Exit timing mode. Repeat to run a subset. Defaults to the strategy's analysis set.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Root output directory.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run without writing HTML/CSV artifacts.",
    )
    parser.add_argument(
        "--show-signal-progress",
        action="store_true",
        help="Show signal precompute progress when the strategy supports it.",
    )
    args = parser.parse_args()

    strategy_input_dict = _load_strategy_input_dict(args.strategy_module_str)

    timing_result_obj = ExecutionTimingAnalysis(
        strategy_factory_fn=strategy_input_dict["strategy_factory_fn"],
        pricing_data_df=strategy_input_dict["pricing_data_df"],
        calendar_idx=strategy_input_dict["calendar_idx"],
        entry_timing_str_tuple=_timing_mode_tuple(
            args.entry_timing,
            strategy_input_dict.get("entry_timing_str_tuple"),
        ),
        exit_timing_str_tuple=_timing_mode_tuple(
            args.exit_timing,
            strategy_input_dict.get("exit_timing_str_tuple"),
        ),
        output_dir_str=args.output_dir,
        save_output_bool=not args.no_save,
        show_signal_progress_bool=args.show_signal_progress,
        order_generation_mode_str=strategy_input_dict.get(
            "order_generation_mode_str",
            "signal_bar",
        ),
        risk_model_str=strategy_input_dict.get("risk_model_str", "daily_ohlc_signal"),
        default_entry_timing_str=strategy_input_dict.get("default_entry_timing_str"),
        default_exit_timing_str=strategy_input_dict.get("default_exit_timing_str"),
    ).run()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(timing_result_obj.metric_df.to_string(index=False))

    if timing_result_obj.output_dir_path is not None:
        print(
            "\nSaved execution timing artifacts to: "
            f"{timing_result_obj.output_dir_path.resolve()}"
        )


if __name__ == "__main__":
    main()
