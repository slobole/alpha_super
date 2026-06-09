"""
Run the SPY/TQQQ Defense First VIX-cash Risk Parity comparison.

Rows:
    - BTAL fallback SPY VIX-cash baseline
    - BTAL fallback SPY VIX-cash Risk Parity
    - BTAL fallback TQQQ VIX-cash baseline
    - BTAL fallback TQQQ VIX-cash Risk Parity

Metric formulas
---------------
Let:

    r_t = V_t / V_{t-1} - 1

Then:

    Sharpe = mean(r_t) / std(r_t) * sqrt(252)
    ann_ret = (V_T / V_0)^(252 / N) - 1
    volatility = std(r_t) * sqrt(252)
    drawdown_t = V_t / max(V_0, ..., V_t) - 1
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

from strategies.taa_df.strategy_taa_df_fallback_variant_utils import (
    build_metric_row_from_strategy,
    save_variant_metric_table,
)


VARIANT_MODULE_NAME_TUPLE: tuple[str, ...] = (
    "strategy_taa_df_btal_fallback_spy_vix_cash",
    "strategy_taa_df_btal_fallback_spy_vix_cash_risk_parity",
    "strategy_taa_df_btal_fallback_tqqq_vix_cash",
    "strategy_taa_df_btal_fallback_tqqq_vix_cash_risk_parity",
)


def _import_variant_module(module_name_str: str):
    return importlib.import_module(f"strategies.taa_df.{module_name_str}")


def _fallback_asset_from_module_name_str(module_name_str: str) -> str:
    if "_fallback_spy_" in module_name_str:
        return "SPY"
    if "_fallback_tqqq_" in module_name_str:
        return "TQQQ"
    raise ValueError(f"Could not infer fallback asset from {module_name_str}.")


def _variant_type_from_module_name_str(module_name_str: str) -> str:
    return "risk_parity" if module_name_str.endswith("_risk_parity") else "baseline"


def run_taa_risk_parity_sweep(
    risk_parity_lookback_day_int: int = 63,
    save_strategy_results_bool: bool = False,
    output_dir_str: str = "results",
) -> tuple[pd.DataFrame, Path]:
    metric_row_list: list[dict[str, float | int | str]] = []

    for module_name_str in VARIANT_MODULE_NAME_TUPLE:
        variant_module = _import_variant_module(module_name_str)
        variant_type_str = _variant_type_from_module_name_str(module_name_str)
        fallback_asset_str = _fallback_asset_from_module_name_str(module_name_str)

        run_kwarg_dict: dict[str, object] = {
            "show_display_bool": False,
            "save_results_bool": save_strategy_results_bool,
            "output_dir_str": output_dir_str,
        }
        if variant_type_str == "risk_parity":
            run_kwarg_dict["risk_parity_lookback_day_int"] = int(risk_parity_lookback_day_int)

        strategy_obj = variant_module.run_variant(**run_kwarg_dict)
        metric_row_dict = build_metric_row_from_strategy(strategy_obj)
        metric_row_dict.update(
            {
                "fallback_asset": fallback_asset_str,
                "variant_type": variant_type_str,
                "risk_parity_lookback_day_int": (
                    int(risk_parity_lookback_day_int)
                    if variant_type_str == "risk_parity"
                    else 0
                ),
            }
        )
        metric_row_list.append(metric_row_dict)

    metric_df = pd.DataFrame(metric_row_list)
    metric_df = metric_df.sort_values(["fallback_asset", "variant_type"]).reset_index(drop=True)

    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir_path = Path(output_dir_str) / "taa_df_spy_tqqq_risk_parity_sweep" / timestamp_str
    csv_path = save_variant_metric_table(
        variant_metric_df=metric_df,
        output_dir_path=output_dir_path,
    )
    return metric_df, csv_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--risk-parity-lookback-days",
        type=int,
        default=63,
        help="Trailing trading-day volatility lookback for Risk Parity rows.",
    )
    parser.add_argument(
        "--save-strategy-results",
        action="store_true",
        help="Also save full HTML/pickle strategy artifacts for each row.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Root output directory.",
    )
    args = parser.parse_args()

    metric_df, csv_path = run_taa_risk_parity_sweep(
        risk_parity_lookback_day_int=int(args.risk_parity_lookback_days),
        save_strategy_results_bool=bool(args.save_strategy_results),
        output_dir_str=str(args.output_dir),
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(metric_df.to_string(index=False))
    print(f"\nSaved metric table to: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
