"""Run a fresh multi-pod portfolio backtest.

Usage:
    uv run --python 3.12 python strategies/run_portfolio_manager.py portfolios/current_book_fresh.yaml
    uv run --python 3.12 python strategies/run_portfolio_manager.py portfolios/current_book_fresh.yaml --no-save
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT_PATH = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from alpha.engine.portfolio_manager import PortfolioManager  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fresh PortfolioManager V1 backtests.")
    parser.add_argument("config_path_str", help="Path to fresh-run portfolio YAML config.")
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Root output directory for manager, pod, and portfolio artifacts.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run without writing pod or portfolio artifacts.",
    )
    parser.add_argument(
        "--show-display",
        action="store_true",
        help="Show per-pod strategy progress and display output.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Override config max_workers_int. Use 1 for serial/debug mode.",
    )
    arg_namespace = parser.parse_args()

    try:
        manager_obj = PortfolioManager.from_yaml(Path(arg_namespace.config_path_str))
        result_obj = manager_obj.run(
            output_dir_str=arg_namespace.output_dir,
            save_results_bool=not arg_namespace.no_save,
            show_display_bool=arg_namespace.show_display,
            max_workers_int=arg_namespace.max_workers,
        )
    except ValueError as exc:
        print(f"PortfolioManager config error: {exc}", file=sys.stderr)
        return 2

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print("\n--- PortfolioManager Summary ---")
    print(result_obj.portfolio.summary.to_string())

    print("\n--- Pod Artifacts ---")
    for pod_run_result in result_obj.pod_run_result_list:
        print(
            f"{pod_run_result.pod_config.pod_id_str}: "
            f"{pod_run_result.pod_artifact_dir_path or 'not saved'}"
        )

    if result_obj.portfolio_output_dir_path is not None:
        print(f"\nPortfolio results saved to: {result_obj.portfolio_output_dir_path.resolve()}")
    if result_obj.manager_metadata_path is not None:
        print(f"PortfolioManager metadata saved to: {result_obj.manager_metadata_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
