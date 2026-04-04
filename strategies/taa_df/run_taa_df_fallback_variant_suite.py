"""
Run the requested Defense First fallback-asset variant grid and save a summary
table.

Metric formulas
---------------
Let:

    r_t = V_t / V_{t-1} - 1

Then:

    Sharpe = mean(r_t) / std(r_t) * sqrt(252)
    ann_ret = (V_T / V_0)^(252 / N) - 1
    volatility = std(r_t) * sqrt(252)
    drawdown_t = V_t / max(V_0, ..., V_t) - 1
    corr = Corr(r_strategy,t, r_$SPX,t)
"""

from __future__ import annotations

import argparse
import importlib
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

repo_root_path = Path(__file__).resolve().parents[2]
repo_root_str = str(repo_root_path)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

try:
    from strategies.taa_df.strategy_taa_df_fallback_variant_utils import (
        REQUESTED_FALLBACK_ASSET_TUPLE,
        build_metric_row_from_strategy,
        save_variant_metric_table,
    )
except ModuleNotFoundError:
    from strategy_taa_df_fallback_variant_utils import (
        REQUESTED_FALLBACK_ASSET_TUPLE,
        build_metric_row_from_strategy,
        save_variant_metric_table,
    )


BASE_STRATEGY_NAME_TUPLE: tuple[str, ...] = (
    "strategy_taa_df",
    "strategy_taa_df_btal",
    "strategy_taa_df_btal_1n",
    "strategy_taa_df_btal_linearity_1n",
)


def _import_variant_module(module_name_str: str):
    try:
        return importlib.import_module(f"strategies.taa_df.{module_name_str}")
    except ModuleNotFoundError:
        return importlib.import_module(module_name_str)


def _variant_module_name_tuple() -> tuple[str, ...]:
    variant_module_name_list: list[str] = []
    for base_strategy_name_str in BASE_STRATEGY_NAME_TUPLE:
        for fallback_asset_str in REQUESTED_FALLBACK_ASSET_TUPLE:
            asset_key_str = fallback_asset_str.lower()
            variant_module_name_list.append(f"{base_strategy_name_str}_fallback_{asset_key_str}")
    return tuple(variant_module_name_list)


def run_variant_suite(
    save_results_bool: bool = False,
    output_dir_str: str = "results",
) -> tuple[pd.DataFrame, Path]:
    variant_metric_row_list: list[dict[str, float | str]] = []

    for module_name_str in _variant_module_name_tuple():
        variant_module = _import_variant_module(module_name_str)
        strategy = variant_module.run_variant(
            show_display_bool=False,
            save_results_bool=save_results_bool,
            output_dir_str=output_dir_str,
        )
        variant_metric_row_list.append(build_metric_row_from_strategy(strategy))

    variant_metric_df = pd.DataFrame(variant_metric_row_list)
    variant_metric_df = variant_metric_df.sort_values("strategy_name").reset_index(drop=True)

    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir_path = Path(output_dir_str) / "taa_df_fallback_variant_suite" / timestamp_str
    csv_path = save_variant_metric_table(
        variant_metric_df=variant_metric_df,
        output_dir_path=output_dir_path,
    )
    return variant_metric_df, csv_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Also save full HTML/pickle strategy artifacts for each variant.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Root output directory.",
    )
    args = parser.parse_args()

    variant_metric_df, csv_path = run_variant_suite(
        save_results_bool=args.save_results,
        output_dir_str=args.output_dir,
    )
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(variant_metric_df.to_string(index=False))
    print(f"\nSaved metric table to: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
