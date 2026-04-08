"""
Run the crisis replay suite for `strategy_mo_atr_normalized_ndx`.

Core formulas
-------------
For each crisis window c:

    R_c = V_end / V_0 - 1

    DD_t = V_t / max(V_0, ..., V_t) - 1

    normalized_equity_t = V_t / V_0

with:

    V_0 = fresh capital at the effective crisis start
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

repo_root_path = Path(__file__).resolve().parents[2]
repo_root_str = str(repo_root_path)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

from alpha.engine.crisis import CrisisAnalyzer


STRATEGY_KEY_STR = "strategy_mo_atr_normalized_ndx"


def main() -> None:
    parser = argparse.ArgumentParser()
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
        help="Show the inner backtest progress bar for each crisis run.",
    )
    parser.add_argument(
        "--show-signal-progress",
        action="store_true",
        help="Show signal precompute progress inside strategy loaders when available.",
    )
    args = parser.parse_args()

    crisis_analyzer = CrisisAnalyzer.from_strategy_key(
        STRATEGY_KEY_STR,
        output_dir_str=args.output_dir,
        save_output_bool=not args.no_save,
        show_progress_bool=args.show_progress,
        show_signal_progress_bool=args.show_signal_progress,
    )
    crisis_replay_result = crisis_analyzer.run()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(crisis_replay_result.crisis_metric_df.to_string(index=False))
    if crisis_replay_result.output_dir_path is not None:
        print(f"\nSaved crisis replay artifacts to: {crisis_replay_result.output_dir_path.resolve()}")


if __name__ == "__main__":
    main()
