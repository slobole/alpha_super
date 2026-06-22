"""
Run monthly 12-1 Top 20 momentum across SP500, Nasdaq 100, and Russell 1000.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from strategies.momentum.strategy_mo_jt_12_1_top20 import (
    CONFIG_BY_VARIANT_KEY_DICT,
    SMA100_INDEX_SMA200_CONFIG_BY_VARIANT_KEY_DICT,
    SMA100_INDEX_SMA200_PN_RANKING_CONFIG_BY_VARIANT_KEY_DICT,
    SMA100_INDEX_SMA200_RANKING_SWEEP_CONFIG_BY_VARIANT_KEY_DICT,
    SMA100_INDEX_SMA200_VOL_TARGET_CONFIG_BY_VARIANT_KEY_DICT,
    SMA100_CONFIG_BY_VARIANT_KEY_DICT,
    SMA100_VOL_TARGET_CONFIG_BY_VARIANT_KEY_DICT,
    VOL_TARGET_CONFIG_BY_VARIANT_KEY_DICT,
    run_variant,
)


DEFAULT_OUTPUT_DIR_STR = "results/research/strategy/jt_12_1_top20_monthly"
DEFAULT_VOL_TARGET_OUTPUT_DIR_STR = "results/research/strategy/jt_12_1_top20_vol_target_monthly"
DEFAULT_SMA100_OUTPUT_DIR_STR = "results/research/strategy/jt_12_1_top20_sma100_monthly"
DEFAULT_SMA100_VOL_TARGET_OUTPUT_DIR_STR = "results/research/strategy/jt_12_1_top20_sma100_vol_target_monthly"
DEFAULT_SMA100_INDEX_SMA200_OUTPUT_DIR_STR = "results/research/strategy/jt_12_1_top20_sma100_index_sma200_monthly"
DEFAULT_SMA100_INDEX_SMA200_VOL_TARGET_OUTPUT_DIR_STR = (
    "results/research/strategy/jt_12_1_top20_sma100_index_sma200_vol_target_monthly"
)
DEFAULT_SMA100_INDEX_SMA200_RANKING_SWEEP_OUTPUT_DIR_STR = (
    "results/research/strategy/jt_12_1_top20_sma100_index_sma200_ranking_sweep_monthly"
)
DEFAULT_SMA100_INDEX_SMA200_PN_RANKING_OUTPUT_DIR_STR = (
    "results/research/strategy/jt_12_1_top20_sma100_index_sma200_pn_ranking_monthly"
)


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


def get_summary_column_value_float(
    summary_df: pd.DataFrame,
    column_name_str: str,
    label_str: str,
) -> float:
    if summary_df is None or len(summary_df) == 0:
        return float("nan")
    if column_name_str not in summary_df.columns or label_str not in summary_df.index:
        return float("nan")
    value_obj = summary_df.loc[label_str, column_name_str]
    return float(value_obj)


def get_benchmark_column_str(summary_df: pd.DataFrame, benchmark_symbol_str: str) -> str | None:
    if summary_df is None or len(summary_df) == 0:
        return None
    if benchmark_symbol_str in summary_df.columns:
        return benchmark_symbol_str
    benchmark_column_list = [str(column_obj) for column_obj in summary_df.columns if str(column_obj) != "Strategy"]
    if len(benchmark_column_list) == 0:
        return None
    return benchmark_column_list[0]


def get_average_position_count_float(strategy_obj) -> float:
    realized_weight_df = getattr(strategy_obj, "realized_weight_df", None)
    if realized_weight_df is None or len(realized_weight_df) == 0:
        return float("nan")

    position_weight_df = realized_weight_df.drop(columns=["Cash"], errors="ignore").fillna(0.0)
    if len(position_weight_df.columns) == 0:
        return 0.0
    return float((position_weight_df.abs() > 1e-12).sum(axis=1).mean())


def get_average_gross_exposure_float(strategy_obj) -> float:
    realized_weight_df = getattr(strategy_obj, "realized_weight_df", None)
    if realized_weight_df is None or len(realized_weight_df) == 0:
        return float("nan")

    position_weight_df = realized_weight_df.drop(columns=["Cash"], errors="ignore").fillna(0.0)
    if len(position_weight_df.columns) == 0:
        return 0.0
    return float(position_weight_df.abs().sum(axis=1).mean())


def build_result_row_dict(strategy_obj) -> dict[str, object]:
    config_obj = strategy_obj.config_obj
    transaction_df = strategy_obj.get_transactions()
    result_df = strategy_obj.results
    benchmark_column_str = get_benchmark_column_str(strategy_obj.summary, config_obj.benchmark_symbol_str)

    start_ts = pd.Timestamp(result_df.index[0]) if result_df is not None and len(result_df) > 0 else pd.NaT
    end_ts = pd.Timestamp(result_df.index[-1]) if result_df is not None and len(result_df) > 0 else pd.NaT
    trade_count_int = int(len(transaction_df)) if transaction_df is not None else 0
    missing_liquidation_count_int = 0
    if transaction_df is not None and len(transaction_df) > 0 and "order_id" in transaction_df.columns:
        missing_liquidation_count_int = int((transaction_df["order_id"] == -1).sum())

    return {
        "variant": config_obj.variant_key_str,
        "universe": config_obj.indexname_str,
        "benchmark_symbol": config_obj.benchmark_symbol_str,
        "start": "" if pd.isna(start_ts) else start_ts.strftime("%Y-%m-%d"),
        "end": "" if pd.isna(end_ts) else end_ts.strftime("%Y-%m-%d"),
        "annual_return_pct": get_summary_value_float(strategy_obj.summary, ["Return (Ann.) [%]", "Annual Return"]),
        "benchmark_annual_return_pct": (
            float("nan")
            if benchmark_column_str is None
            else get_summary_column_value_float(strategy_obj.summary, benchmark_column_str, "Return (Ann.) [%]")
        ),
        "annual_volatility_pct": get_summary_value_float(
            strategy_obj.summary,
            ["Volatility (Ann.) [%]", "Annual Volatility"],
        ),
        "benchmark_annual_volatility_pct": (
            float("nan")
            if benchmark_column_str is None
            else get_summary_column_value_float(strategy_obj.summary, benchmark_column_str, "Volatility (Ann.) [%]")
        ),
        "sharpe": get_summary_value_float(strategy_obj.summary, ["Sharpe Ratio", "sharpe"]),
        "benchmark_sharpe": (
            float("nan")
            if benchmark_column_str is None
            else get_summary_column_value_float(strategy_obj.summary, benchmark_column_str, "Sharpe Ratio")
        ),
        "max_drawdown_pct": get_summary_value_float(strategy_obj.summary, ["Max. Drawdown [%]", "Max Drawdown"]),
        "benchmark_max_drawdown_pct": (
            float("nan")
            if benchmark_column_str is None
            else get_summary_column_value_float(strategy_obj.summary, benchmark_column_str, "Max. Drawdown [%]")
        ),
        "mar": get_summary_value_float(strategy_obj.summary, ["MAR Ratio", "MAR"]),
        "benchmark_mar": (
            float("nan")
            if benchmark_column_str is None
            else get_summary_column_value_float(strategy_obj.summary, benchmark_column_str, "MAR Ratio")
        ),
        "turnover_ann_pct": get_summary_value_float(strategy_obj.summary, ["Turnover (Ann.) [%]"]),
        "cost_drag_ann_pct": get_summary_value_float(strategy_obj.summary, ["Cost Drag (Ann.) [%]"]),
        "trade_count": trade_count_int,
        "missing_liquidations": missing_liquidation_count_int,
        "avg_position_count": get_average_position_count_float(strategy_obj),
        "avg_gross_exposure": get_average_gross_exposure_float(strategy_obj),
        "lookback_trading_days": config_obj.lookback_trading_day_int,
        "skip_trading_days": config_obj.skip_trading_day_int,
        "max_positions": config_obj.max_positions_int,
        "ranking_method": config_obj.ranking_method_str,
        "volatility_target_enabled": config_obj.volatility_target_enabled_bool,
        "target_annual_volatility": config_obj.target_annual_volatility_float,
        "realized_vol_window": config_obj.realized_vol_window_int,
        "max_gross_exposure_config": config_obj.max_gross_exposure_float,
        "stock_sma_filter_enabled": config_obj.stock_sma_filter_enabled_bool,
        "stock_sma_window": config_obj.stock_sma_window_int,
        "index_sma_filter_enabled": config_obj.index_sma_filter_enabled_bool,
        "index_sma_window": config_obj.index_sma_window_int,
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
    config_by_variant_key_dict: dict[str, object] | None = None,
) -> pd.DataFrame:
    output_dir_path = Path(output_dir_str)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    selected_config_by_variant_key_dict = (
        CONFIG_BY_VARIANT_KEY_DICT if config_by_variant_key_dict is None else config_by_variant_key_dict
    )

    result_row_dict_list: list[dict[str, object]] = []
    for variant_key_str in selected_config_by_variant_key_dict:
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
    csv_path = output_dir_path / "jt_12_1_top20_comparison.csv"
    markdown_path = output_dir_path / "jt_12_1_top20_comparison.md"
    result_df.to_csv(csv_path, index=False)
    markdown_path.write_text(dataframe_to_markdown_str(result_df), encoding="utf-8")
    return result_df


def parse_args() -> argparse.Namespace:
    parser_obj = argparse.ArgumentParser()
    parser_obj.add_argument("--output-dir", default=None)
    parser_obj.add_argument("--start", default=None)
    parser_obj.add_argument("--end", default=None)
    parser_obj.add_argument("--no-save-results", action="store_true")
    parser_obj.add_argument("--with-signal-audit", action="store_true")
    parser_obj.add_argument(
        "--vol-target",
        action="store_true",
        help="Run the 12% annual-volatility-targeted, max-100%-gross variants.",
    )
    parser_obj.add_argument(
        "--sma100-filter",
        action="store_true",
        help="Require selected stocks to close above their own trailing 100-day SMA.",
    )
    parser_obj.add_argument(
        "--index-sma200-filter",
        action="store_true",
        help="Require the matching benchmark index to close above its trailing 200-day SMA.",
    )
    parser_obj.add_argument(
        "--ranking-sweep",
        action="store_true",
        help="Run raw, vol-normalized, trend-quality, multi-horizon, and residual ranking variants.",
    )
    parser_obj.add_argument(
        "--pn-ranking-sweep",
        action="store_true",
        help="Run raw, multi-window EV, and multi-window LRB P/N ranking variants.",
    )
    return parser_obj.parse_args()


def select_config_by_variant_key_dict(
    vol_target_bool: bool,
    sma100_filter_bool: bool,
    index_sma200_filter_bool: bool,
    ranking_sweep_bool: bool,
    pn_ranking_sweep_bool: bool,
) -> dict[str, object]:
    if ranking_sweep_bool and pn_ranking_sweep_bool:
        raise ValueError("Choose either --ranking-sweep or --pn-ranking-sweep, not both.")
    if pn_ranking_sweep_bool:
        if vol_target_bool:
            raise ValueError("--pn-ranking-sweep is wired for the unscaled baseline, not --vol-target.")
        if not sma100_filter_bool or not index_sma200_filter_bool:
            raise ValueError("--pn-ranking-sweep requires --sma100-filter --index-sma200-filter.")
        return SMA100_INDEX_SMA200_PN_RANKING_CONFIG_BY_VARIANT_KEY_DICT
    if ranking_sweep_bool:
        if vol_target_bool:
            raise ValueError("--ranking-sweep is wired for the unscaled baseline, not --vol-target.")
        if not sma100_filter_bool or not index_sma200_filter_bool:
            raise ValueError("--ranking-sweep requires --sma100-filter --index-sma200-filter.")
        return SMA100_INDEX_SMA200_RANKING_SWEEP_CONFIG_BY_VARIANT_KEY_DICT
    if index_sma200_filter_bool and not sma100_filter_bool:
        raise ValueError("--index-sma200-filter is only wired for the current --sma100-filter follow-up.")
    if index_sma200_filter_bool and vol_target_bool:
        return SMA100_INDEX_SMA200_VOL_TARGET_CONFIG_BY_VARIANT_KEY_DICT
    if index_sma200_filter_bool:
        return SMA100_INDEX_SMA200_CONFIG_BY_VARIANT_KEY_DICT
    if vol_target_bool and sma100_filter_bool:
        return SMA100_VOL_TARGET_CONFIG_BY_VARIANT_KEY_DICT
    if sma100_filter_bool:
        return SMA100_CONFIG_BY_VARIANT_KEY_DICT
    if vol_target_bool:
        return VOL_TARGET_CONFIG_BY_VARIANT_KEY_DICT
    return CONFIG_BY_VARIANT_KEY_DICT


def get_default_output_dir_str(
    vol_target_bool: bool,
    sma100_filter_bool: bool,
    index_sma200_filter_bool: bool,
    ranking_sweep_bool: bool,
    pn_ranking_sweep_bool: bool,
) -> str:
    if pn_ranking_sweep_bool:
        return DEFAULT_SMA100_INDEX_SMA200_PN_RANKING_OUTPUT_DIR_STR
    if ranking_sweep_bool:
        return DEFAULT_SMA100_INDEX_SMA200_RANKING_SWEEP_OUTPUT_DIR_STR
    if index_sma200_filter_bool and vol_target_bool:
        return DEFAULT_SMA100_INDEX_SMA200_VOL_TARGET_OUTPUT_DIR_STR
    if index_sma200_filter_bool:
        return DEFAULT_SMA100_INDEX_SMA200_OUTPUT_DIR_STR
    if vol_target_bool and sma100_filter_bool:
        return DEFAULT_SMA100_VOL_TARGET_OUTPUT_DIR_STR
    if sma100_filter_bool:
        return DEFAULT_SMA100_OUTPUT_DIR_STR
    if vol_target_bool:
        return DEFAULT_VOL_TARGET_OUTPUT_DIR_STR
    return DEFAULT_OUTPUT_DIR_STR


def main() -> None:
    args_obj = parse_args()
    config_by_variant_key_dict = select_config_by_variant_key_dict(
        vol_target_bool=bool(args_obj.vol_target),
        sma100_filter_bool=bool(args_obj.sma100_filter),
        index_sma200_filter_bool=bool(args_obj.index_sma200_filter),
        ranking_sweep_bool=bool(args_obj.ranking_sweep),
        pn_ranking_sweep_bool=bool(args_obj.pn_ranking_sweep),
    )
    output_dir_str = args_obj.output_dir
    if output_dir_str is None:
        output_dir_str = get_default_output_dir_str(
            vol_target_bool=bool(args_obj.vol_target),
            sma100_filter_bool=bool(args_obj.sma100_filter),
            index_sma200_filter_bool=bool(args_obj.index_sma200_filter),
            ranking_sweep_bool=bool(args_obj.ranking_sweep),
            pn_ranking_sweep_bool=bool(args_obj.pn_ranking_sweep),
        )
    result_df = run_sweep(
        output_dir_str=output_dir_str,
        backtest_start_date_str=args_obj.start,
        end_date_str=args_obj.end,
        save_results_bool=not args_obj.no_save_results,
        audit_override_bool=None if args_obj.with_signal_audit else False,
        config_by_variant_key_dict=config_by_variant_key_dict,
    )
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
