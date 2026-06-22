"""
Run PTA-conditioned winner-continuation Vanilla research variants.

The initial suite contains the Russell 3000 long-short 10x10 config. More
configs can be added to CONFIG_BY_VARIANT_KEY_DICT in the strategy module
without changing this runner.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from strategies.momentum.strategy_mo_pta_winner_continuation import (
    CONFIG_BY_VARIANT_KEY_DICT,
    run_variant,
)


DEFAULT_OUTPUT_DIR_STR = "results/research/strategy/pta_winner_continuation"


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

    start_ts = pd.Timestamp(result_df.index[0]) if result_df is not None and len(result_df) > 0 else pd.NaT
    end_ts = pd.Timestamp(result_df.index[-1]) if result_df is not None and len(result_df) > 0 else pd.NaT
    trade_count_int = int(len(transaction_df)) if transaction_df is not None else 0
    average_trade_day_name_count_float = float("nan")
    if transaction_df is not None and len(transaction_df) > 0 and "bar" in transaction_df.columns:
        trade_day_name_count_ser = transaction_df.groupby("bar")["asset"].nunique()
        average_trade_day_name_count_float = float(trade_day_name_count_ser.mean())

    return {
        "variant": config_obj.variant_key_str,
        "universe": config_obj.indexname_str,
        "benchmark_symbol": config_obj.benchmark_symbol_str,
        "regime_symbol": config_obj.regime_symbol_str,
        "bucket_count": config_obj.bucket_count_int,
        "winner_return_bucket": config_obj.winner_return_bucket_int,
        "long_pta_bucket": config_obj.long_pta_bucket_int,
        "short_pta_bucket": config_obj.short_pta_bucket_int,
        "long_gross": config_obj.long_gross_exposure_float,
        "short_gross": config_obj.short_gross_exposure_float,
        "index_sma_filter": config_obj.index_sma_filter_enabled_bool,
        "index_sma_window": config_obj.index_sma_window_int,
        "stock_sma_filter": config_obj.stock_sma_filter_enabled_bool,
        "stock_sma_window": config_obj.stock_sma_window_int,
        "inverse_vol_weighting": config_obj.inverse_vol_weighting_enabled_bool,
        "inverse_vol_window": config_obj.inverse_vol_window_int,
        "max_long_positions": config_obj.max_long_positions_int,
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
        "avg_names_per_trade_day": average_trade_day_name_count_float,
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
    variant_key_list: list[str] | None = None,
    backtest_start_date_str: str | None = None,
    end_date_str: str | None = None,
    save_results_bool: bool = True,
    audit_override_bool: bool | None = False,
) -> pd.DataFrame:
    output_dir_path = Path(output_dir_str)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    selected_variant_key_list = list(CONFIG_BY_VARIANT_KEY_DICT) if variant_key_list is None else variant_key_list
    unknown_variant_key_list = [
        variant_key_str
        for variant_key_str in selected_variant_key_list
        if variant_key_str not in CONFIG_BY_VARIANT_KEY_DICT
    ]
    if len(unknown_variant_key_list) > 0:
        raise ValueError(f"Unknown variant keys: {unknown_variant_key_list}")

    result_row_dict_list: list[dict[str, object]] = []
    for variant_key_str in selected_variant_key_list:
        strategy_obj = run_variant(
            variant_key_str=variant_key_str,
            show_display_bool=False,
            save_results_bool=save_results_bool,
            output_dir_str=output_dir_str,
            backtest_start_date_str=backtest_start_date_str,
            end_date_str=end_date_str,
            audit_override_bool=audit_override_bool,
        )
        result_row_dict_list.append(build_result_row_dict(strategy_obj))

    result_df = pd.DataFrame(result_row_dict_list)
    csv_path = output_dir_path / "pta_winner_continuation_comparison.csv"
    markdown_path = output_dir_path / "pta_winner_continuation_comparison.md"
    result_df.to_csv(csv_path, index=False)
    markdown_path.write_text(dataframe_to_markdown_str(result_df), encoding="utf-8")
    return result_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR_STR)
    parser.add_argument("--variant", action="append", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--no-save-results", action="store_true")
    parser.add_argument("--with-signal-audit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_df = run_sweep(
        output_dir_str=args.output_dir,
        variant_key_list=args.variant,
        backtest_start_date_str=args.start,
        end_date_str=args.end,
        save_results_bool=not args.no_save_results,
        audit_override_bool=None if args.with_signal_audit else False,
    )
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
