"""
Run the historical pre-crisis StressTestAnalyzer for a supported strategy key.

Core timing
-----------
For each crisis window:

    L = S - launch_offset trading bars
    P = previous tradable bar before S

The strategy starts flat at L, trades normally through E, and entering-event
exposure is measured at P close.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

repo_root_path = Path(__file__).resolve().parent.parent
repo_root_str = str(repo_root_path)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

from alpha.engine.stress_test import (
    DEFAULT_LAUNCH_OFFSET_TUPLE,
    StressTestAnalyzer,
)


def main() -> None:
    strategy_key_tuple = StressTestAnalyzer.supported_strategy_key_tuple()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "strategy_key",
        choices=strategy_key_tuple,
        help="Supported strategy key to stress across configured crisis windows.",
    )
    parser.add_argument(
        "--launch-offset",
        action="append",
        type=int,
        default=[],
        help=(
            "Trading bars before each crisis start to launch the strategy. "
            "Repeat for multiple offsets. Defaults to 5, 21, 42, 63."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Root output directory.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run the suite without writing HTML/CSV artifacts.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show the inner backtest progress bar for each stress run.",
    )
    parser.add_argument(
        "--show-signal-progress",
        action="store_true",
        help="Show signal precompute progress inside strategy loaders when available.",
    )
    args = parser.parse_args()

    launch_offset_tuple = (
        tuple(args.launch_offset)
        if len(args.launch_offset) > 0
        else DEFAULT_LAUNCH_OFFSET_TUPLE
    )
    stress_test_analyzer = StressTestAnalyzer.from_strategy_key(
        args.strategy_key,
        launch_offset_tuple=launch_offset_tuple,
        output_dir_str=args.output_dir,
        save_output_bool=not args.no_save,
        show_progress_bool=args.show_progress,
        show_signal_progress_bool=args.show_signal_progress,
    )
    stress_result_obj = stress_test_analyzer.run()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(stress_result_obj.stress_metric_df.to_string(index=False))
    if stress_result_obj.output_dir_path is not None:
        print(f"\nSaved stress test artifacts to: {stress_result_obj.output_dir_path.resolve()}")


if __name__ == "__main__":
    main()
