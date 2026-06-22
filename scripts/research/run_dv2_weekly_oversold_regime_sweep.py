"""
Run the weekly DV2 oversold regime strategy across S&P 500, Nasdaq 100, and Russell 1000.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from strategies.dv2.strategy_mr_dv2_weekly_oversold_regime import (
    CONFIG_BY_VARIANT_KEY_DICT,
    run_variant,
)


DEFAULT_OUTPUT_DIR_STR = "results/research/strategy/dv2_weekly_oversold_regime"


def get_summary_value_float(summary_df: pd.DataFrame, label_list: list[str]) -> float:
    if summary_df is None or len(summary_df) == 0:
        return float("nan")

    for label_str in label_list:
        if label_str in summary_df.index:
            value_obj = summary_df.loc[label_str]
            if isinstance(value_obj, pd.Series):
                value_obj = value_obj.iloc[0]
            return float(value_obj)
        if label_str in summary_df.columns:
            value_obj = summary_df[label_str].iloc[0]
            return float(value_obj)
    return float("nan")


def build_result_row_dict(strategy_obj) -> dict[str, object]:
    config_obj = strategy_obj.config
    transaction_df = strategy_obj.get_transactions()
    result_df = strategy_obj.results
    realized_weight_df = getattr(strategy_obj, "realized_weight_df", pd.DataFrame())

    start_ts = pd.Timestamp(result_df.index[0]) if result_df is not None and len(result_df) > 0 else pd.NaT
    end_ts = pd.Timestamp(result_df.index[-1]) if result_df is not None and len(result_df) > 0 else pd.NaT
    trade_count_int = int(len(transaction_df)) if transaction_df is not None else 0
    average_position_count_float = float("nan")
    average_exposure_float = float("nan")
    if isinstance(realized_weight_df, pd.DataFrame) and len(realized_weight_df) > 0:
        non_cash_column_list = [column_str for column_str in realized_weight_df.columns if column_str != "Cash"]
        if len(non_cash_column_list) > 0:
            average_position_count_float = float((realized_weight_df[non_cash_column_list].fillna(0.0) > 0.0).sum(axis=1).mean())
            average_exposure_float = float(realized_weight_df[non_cash_column_list].fillna(0.0).sum(axis=1).mean())

    return {
        "variant": config_obj.variant_key_str,
        "universe": config_obj.indexname_str,
        "regime_symbol": config_obj.regime_symbol_str,
        "start": "" if pd.isna(start_ts) else start_ts.strftime("%Y-%m-%d"),
        "end": "" if pd.isna(end_ts) else end_ts.strftime("%Y-%m-%d"),
        "annual_return_pct": get_summary_value_float(strategy_obj.summary, ["Return (Ann.) [%]", "Annual Return"]),
        "annual_volatility_pct": get_summary_value_float(
            strategy_obj.summary,
            ["Volatility (Ann.) [%]", "Annual Volatility"],
        ),
        "sharpe": get_summary_value_float(strategy_obj.summary, ["Sharpe Ratio", "sharpe"]),
        "max_drawdown_pct": get_summary_value_float(strategy_obj.summary, ["Max. Drawdown [%]", "Max Drawdown"]),
        "mar": get_summary_value_float(strategy_obj.summary, ["MAR Ratio", "MAR"]),
        "turnover_ann_pct": get_summary_value_float(strategy_obj.summary, ["Turnover (Ann.) [%]"]),
        "cost_drag_ann_pct": get_summary_value_float(strategy_obj.summary, ["Cost Drag (Ann.) [%]"]),
        "trade_count": trade_count_int,
        "avg_positions": average_position_count_float,
        "avg_exposure": average_exposure_float,
        "selection_fraction": config_obj.selection_fraction_float,
        "dv2_window": config_obj.dv2_window_int,
        "regime_sma_window": config_obj.regime_sma_window_int,
        "slippage": config_obj.slippage_float,
        "commission_per_share": config_obj.commission_per_share_float,
        "commission_minimum": config_obj.commission_minimum_float,
    }


def dataframe_to_markdown_str(result_df: pd.DataFrame) -> str:
    column_list = result_df.columns.astype(str).tolist()
    row_list = [
        "| " + " | ".join(column_list) + " |",
        "| " + " | ".join(["---"] * len(column_list)) + " |",
    ]
    for _row_index, row_ser in result_df.iterrows():
        value_list = ["" if pd.isna(value_obj) else str(value_obj) for value_obj in row_ser.tolist()]
        row_list.append("| " + " | ".join(value_list) + " |")
    return "\n".join(row_list) + "\n"


def run_sweep(
    output_dir_str: str = DEFAULT_OUTPUT_DIR_STR,
    backtest_start_date_str: str | None = None,
    end_date_str: str | None = None,
    save_results_bool: bool = True,
    audit_override_bool: bool | None = False,
    variant_key_list: list[str] | None = None,
) -> pd.DataFrame:
    output_dir_path = Path(output_dir_str)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir_path / "dv2_weekly_oversold_regime_comparison.csv"
    markdown_path = output_dir_path / "dv2_weekly_oversold_regime_comparison.md"

    result_row_dict_list: list[dict[str, object]] = []
    selected_variant_key_list = list(CONFIG_BY_VARIANT_KEY_DICT) if variant_key_list is None else variant_key_list
    for variant_key_str in selected_variant_key_list:
        if variant_key_str not in CONFIG_BY_VARIANT_KEY_DICT:
            raise ValueError(f"Unknown variant_key_str: {variant_key_str}")

        config_obj = CONFIG_BY_VARIANT_KEY_DICT[variant_key_str]
        try:
            strategy_obj = run_variant(
                variant_key_str=variant_key_str,
                show_display_bool=False,
                save_results_bool=save_results_bool,
                output_dir_str=output_dir_str,
                backtest_start_date_str=backtest_start_date_str,
                end_date_str=end_date_str,
                audit_override_bool=audit_override_bool,
            )
            result_row_dict = build_result_row_dict(strategy_obj)
            result_row_dict["status"] = "ok"
            result_row_dict["error"] = ""
        except Exception as exception_obj:
            result_row_dict = {
                "variant": config_obj.variant_key_str,
                "universe": config_obj.indexname_str,
                "regime_symbol": config_obj.regime_symbol_str,
                "start": backtest_start_date_str or config_obj.backtest_start_date_str,
                "end": end_date_str or "",
                "annual_return_pct": float("nan"),
                "annual_volatility_pct": float("nan"),
                "sharpe": float("nan"),
                "max_drawdown_pct": float("nan"),
                "mar": float("nan"),
                "turnover_ann_pct": float("nan"),
                "cost_drag_ann_pct": float("nan"),
                "trade_count": float("nan"),
                "avg_positions": float("nan"),
                "avg_exposure": float("nan"),
                "selection_fraction": config_obj.selection_fraction_float,
                "dv2_window": config_obj.dv2_window_int,
                "regime_sma_window": config_obj.regime_sma_window_int,
                "slippage": config_obj.slippage_float,
                "commission_per_share": config_obj.commission_per_share_float,
                "commission_minimum": config_obj.commission_minimum_float,
                "status": "failed",
                "error": str(exception_obj).replace("\n", " "),
            }

        result_row_dict_list.append(result_row_dict)
        partial_result_df = pd.DataFrame(result_row_dict_list)
        partial_result_df.to_csv(csv_path, index=False)
        markdown_path.write_text(dataframe_to_markdown_str(partial_result_df), encoding="utf-8")

    result_df = pd.DataFrame(result_row_dict_list)
    result_df.to_csv(csv_path, index=False)
    markdown_path.write_text(dataframe_to_markdown_str(result_df), encoding="utf-8")
    return result_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR_STR)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--variants", default=None, help="Comma-separated variant keys. Default: all.")
    parser.add_argument("--no-save-results", action="store_true")
    parser.add_argument("--with-signal-audit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_df = run_sweep(
        output_dir_str=args.output_dir,
        backtest_start_date_str=args.start,
        end_date_str=args.end,
        save_results_bool=not args.no_save_results,
        audit_override_bool=None if args.with_signal_audit else False,
        variant_key_list=None if args.variants is None else [item.strip() for item in args.variants.split(",") if item.strip()],
    )
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
