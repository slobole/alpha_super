"""
Run the VXN-scaled ATR-normalized NDX ROC-window comparison suite.

The suite compares:

1. The existing 12-month ROC VXN-scaled ATR model.
2. Last 1-month ROC.
3. Prior 1-month ROC, skipping the newest month.
4. Last 3-month ROC.
5. Classic skip-month momentum windows: 12-1, 6-1, and 3-1.
6. Research blends that change only the momentum numerator.

All runs share the same VXN scaling, NDX point-in-time universe, trend filters,
execution timing, capital base, and cost settings.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from alpha.engine.backtest import run_daily
from alpha.engine.report import build_research_output_path
from strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled import (
    DEFAULT_CONFIG as BASE_VXN_CONFIG,
    VxnScaledAtrNormalizedNdxStrategy,
    get_vxn_scaled_atr_normalized_ndx_data,
)
from strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled_roc_variants import (
    DEFAULT_ATR_WINDOW_INT,
    ROC_MODE_ANTI_REVERSAL_SKIP_BLEND_STR,
    ROC_MODE_CONSISTENCY_SKIP_BLEND_STR,
    ROC_MODE_EQUAL_SKIP_BLEND_STR,
    ROC_MODE_LAST_12M_STR,
    ROC_MODE_LAST_1M_STR,
    ROC_MODE_LAST_3M_STR,
    ROC_MODE_PRIOR_1M_STR,
    ROC_MODE_SKIP_12_1_STR,
    ROC_MODE_SKIP_3_1_STR,
    ROC_MODE_SKIP_6_1_STR,
    ROC_MODE_WEIGHTED_SKIP_BLEND_STR,
    VxnScaledAtrNormalizedNdxRocVariantStrategy,
    build_roc_variant_config,
    make_roc_variant_rebalance_schedule_df,
)


DEFAULT_BACKTEST_START_DATE_STR = "2012-01-01"
DEFAULT_CAPITAL_BASE_FLOAT = 100_000.0
SUITE_ENTITY_ID_STR = "ndx_vxn_scaled_atr_roc_variants"
SUITE_ANALYSIS_TYPE_STR = "roc_window_comparison"
FOCUSED_ATR_ENTITY_ID_STR = "ndx_vxn_scaled_atr_focused_atr_windows"
FOCUSED_ATR_ANALYSIS_TYPE_STR = "atr_window_comparison"
DEFAULT_ATR_WINDOW_LIST = [20, 63, 126]


def _run_strategy_obj(
    strategy_obj,
    pricing_data_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    backtest_start_date_str: str,
):
    strategy_obj.universe_df = universe_df

    # *** CRITICAL*** The suite keeps full pre-start data for trailing signal
    # features, but simulation and reported performance begin only at the
    # requested comparison start date.
    calendar_idx = pricing_data_df.index[
        pricing_data_df.index >= pd.Timestamp(backtest_start_date_str)
    ]
    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar=calendar_idx,
        show_progress=False,
        show_signal_progress_bool=False,
        audit_override_bool=None,
    )
    return strategy_obj


def _make_base_strategy_obj(
    rebalance_schedule_df: pd.DataFrame,
    vxn_scale_signal_df: pd.DataFrame,
    capital_base_float: float,
) -> VxnScaledAtrNormalizedNdxStrategy:
    return VxnScaledAtrNormalizedNdxStrategy(
        name="vxn_scaled_atr_roc_12m_base",
        benchmarks=[BASE_VXN_CONFIG.regime_symbol_str],
        rebalance_schedule_df=rebalance_schedule_df,
        vxn_scale_signal_df=vxn_scale_signal_df,
        regime_symbol_str=BASE_VXN_CONFIG.regime_symbol_str,
        capital_base=capital_base_float,
        slippage=BASE_VXN_CONFIG.slippage_float,
        commission_per_share=BASE_VXN_CONFIG.commission_per_share_float,
        commission_minimum=BASE_VXN_CONFIG.commission_minimum_float,
        lookback_month_int=BASE_VXN_CONFIG.lookback_month_int,
        index_trend_window_int=BASE_VXN_CONFIG.index_trend_window_int,
        stock_trend_window_int=BASE_VXN_CONFIG.stock_trend_window_int,
        max_positions_int=BASE_VXN_CONFIG.max_positions_int,
    )


def _make_roc_variant_strategy_obj(
    pricing_data_df: pd.DataFrame,
    vxn_scale_signal_df: pd.DataFrame,
    roc_mode_str: str,
    atr_window_int: int,
    backtest_start_date_str: str,
    capital_base_float: float,
    end_date_str: str | None,
) -> VxnScaledAtrNormalizedNdxRocVariantStrategy:
    config_obj = build_roc_variant_config(
        roc_mode_str=roc_mode_str,
        atr_window_int=atr_window_int,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
    )
    rebalance_schedule_df = make_roc_variant_rebalance_schedule_df(
        pricing_data_df=pricing_data_df,
        config=config_obj,
    )
    return VxnScaledAtrNormalizedNdxRocVariantStrategy(
        name=f"vxn_scaled_atr_roc_{roc_mode_str}",
        benchmarks=[config_obj.regime_symbol_str],
        rebalance_schedule_df=rebalance_schedule_df,
        vxn_scale_signal_df=vxn_scale_signal_df,
        roc_mode_str=config_obj.roc_mode_str,
        regime_symbol_str=config_obj.regime_symbol_str,
        capital_base=config_obj.capital_base_float,
        slippage=config_obj.slippage_float,
        commission_per_share=config_obj.commission_per_share_float,
        commission_minimum=config_obj.commission_minimum_float,
        lookback_month_int=config_obj.lookback_month_int,
        index_trend_window_int=config_obj.index_trend_window_int,
        stock_trend_window_int=config_obj.stock_trend_window_int,
        max_positions_int=config_obj.max_positions_int,
        atr_window_int=config_obj.atr_window_int,
    )


def _summary_value_obj(strategy_obj, metric_name_str: str):
    summary_df = getattr(strategy_obj, "summary", None)
    if summary_df is None or metric_name_str not in summary_df.index:
        return None
    return summary_df.loc[metric_name_str, "Strategy"]


def _summary_value_float(strategy_obj, metric_name_str: str) -> float | None:
    value_obj = _summary_value_obj(strategy_obj=strategy_obj, metric_name_str=metric_name_str)
    if value_obj is None or pd.isna(value_obj):
        return None
    return float(value_obj)


def _summary_value_date_str(strategy_obj, metric_name_str: str) -> str | None:
    value_obj = _summary_value_obj(strategy_obj=strategy_obj, metric_name_str=metric_name_str)
    if value_obj is None or pd.isna(value_obj):
        return None
    return pd.Timestamp(value_obj).date().isoformat()


def _trade_count_int(strategy_obj) -> int | None:
    summary_trades_df = getattr(strategy_obj, "summary_trades", None)
    if summary_trades_df is not None and "# Trades" in summary_trades_df.index:
        trade_count_obj = summary_trades_df.loc["# Trades"]
        if isinstance(trade_count_obj, pd.Series):
            trade_count_obj = trade_count_obj.iloc[0]
        if pd.notna(trade_count_obj):
            return int(float(trade_count_obj))
    trade_df = getattr(strategy_obj, "_trades", None)
    if trade_df is not None:
        return int(len(trade_df))
    return None


def _comparison_row_dict(
    strategy_obj,
    label_str: str,
    roc_definition_str: str,
    atr_window_int: int | None = None,
) -> dict[str, object]:
    return {
        "strategy": label_str,
        "roc_definition": roc_definition_str,
        "atr_window": atr_window_int,
        "start": _summary_value_date_str(strategy_obj, "Start"),
        "end": _summary_value_date_str(strategy_obj, "End"),
        "final_equity": _summary_value_float(strategy_obj, "Final [$]"),
        "total_return_pct": _summary_value_float(strategy_obj, "Return [%]"),
        "ann_return_pct": _summary_value_float(strategy_obj, "Return (Ann.) [%]"),
        "ann_vol_pct": _summary_value_float(strategy_obj, "Volatility (Ann.) [%]"),
        "sharpe": _summary_value_float(strategy_obj, "Sharpe Ratio"),
        "max_drawdown_pct": _summary_value_float(strategy_obj, "Max. Drawdown [%]"),
        "mar": _summary_value_float(strategy_obj, "MAR Ratio"),
        "exposure_pct": _summary_value_float(strategy_obj, "Exposure Time [%]"),
        "turnover_ann_pct": _summary_value_float(strategy_obj, "Turnover (Ann.) [%]"),
        "cost_drag_ann_pct": _summary_value_float(strategy_obj, "Cost Drag (Ann.) [%]"),
        "trade_count": _trade_count_int(strategy_obj),
    }


def _format_value_str(value_obj) -> str:
    if value_obj is None or pd.isna(value_obj):
        return ""
    if isinstance(value_obj, float):
        return f"{value_obj:,.2f}"
    return str(value_obj)


def _markdown_table_str(display_df: pd.DataFrame) -> str:
    column_list = list(display_df.columns)
    header_str = "| " + " | ".join(column_list) + " |"
    separator_str = "| " + " | ".join(["---"] * len(column_list)) + " |"
    row_str_list = []
    for _row_index, row_ser in display_df.iterrows():
        value_str_list = [_format_value_str(row_ser[column_str]) for column_str in column_list]
        row_str_list.append("| " + " | ".join(value_str_list) + " |")
    return "\n".join([header_str, separator_str] + row_str_list)


def _write_equity_curve_png(equity_curve_df: pd.DataFrame, output_path: Path) -> None:
    figure_obj, axis_obj = plt.subplots(figsize=(12, 7))
    normalized_equity_df = equity_curve_df / equity_curve_df.iloc[0]
    normalized_equity_df.plot(ax=axis_obj, linewidth=1.6)
    axis_obj.set_title("VXN-Scaled ATR NDX ROC Window Comparison")
    axis_obj.set_ylabel("Growth of $1")
    axis_obj.set_xlabel("Date")
    axis_obj.grid(True, alpha=0.25)
    axis_obj.legend(loc="best", fontsize=9)
    figure_obj.tight_layout()
    figure_obj.savefig(output_path, dpi=160)
    plt.close(figure_obj)


def _parse_atr_window_list(raw_atr_windows_str: str) -> list[int]:
    atr_window_list = [
        int(part_str.strip())
        for part_str in raw_atr_windows_str.split(",")
        if part_str.strip()
    ]
    if len(atr_window_list) == 0:
        raise ValueError("At least one ATR window must be provided.")
    if any(atr_window_int <= 0 for atr_window_int in atr_window_list):
        raise ValueError(f"ATR windows must be positive, got {atr_window_list}.")
    return atr_window_list


def run_comparison_suite(
    backtest_start_date_str: str = DEFAULT_BACKTEST_START_DATE_STR,
    capital_base_float: float = DEFAULT_CAPITAL_BASE_FLOAT,
    end_date_str: str | None = None,
    output_dir_str: str = "results",
) -> tuple[pd.DataFrame, Path]:
    base_config_obj = replace(
        BASE_VXN_CONFIG,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=float(capital_base_float),
        end_date_str=end_date_str,
    )
    pricing_data_df, universe_df, base_rebalance_schedule_df, vxn_scale_signal_df = (
        get_vxn_scaled_atr_normalized_ndx_data(config=base_config_obj)
    )

    strategy_result_list = []
    base_strategy_obj = _make_base_strategy_obj(
        rebalance_schedule_df=base_rebalance_schedule_df,
        vxn_scale_signal_df=vxn_scale_signal_df,
        capital_base_float=capital_base_float,
    )
    strategy_result_list.append(
        (
            "12m_base",
            "Close_ME_t / Close_ME_t-12 - 1",
            _run_strategy_obj(
                strategy_obj=base_strategy_obj,
                pricing_data_df=pricing_data_df,
                universe_df=universe_df,
                backtest_start_date_str=backtest_start_date_str,
            ),
        )
    )

    roc_variant_spec_list = [
        (ROC_MODE_LAST_1M_STR, "Close_ME_t / Close_ME_t-1 - 1"),
        (ROC_MODE_PRIOR_1M_STR, "Close_ME_t-1 / Close_ME_t-2 - 1"),
        (ROC_MODE_LAST_3M_STR, "Close_ME_t / Close_ME_t-3 - 1"),
        (ROC_MODE_SKIP_12_1_STR, "Close_ME_t-1 / Close_ME_t-12 - 1"),
        (ROC_MODE_SKIP_6_1_STR, "Close_ME_t-1 / Close_ME_t-6 - 1"),
        (ROC_MODE_SKIP_3_1_STR, "Close_ME_t-1 / Close_ME_t-3 - 1"),
        (
            ROC_MODE_EQUAL_SKIP_BLEND_STR,
            "mean(skip_3_1, skip_6_1, skip_12_1)",
        ),
        (
            ROC_MODE_WEIGHTED_SKIP_BLEND_STR,
            "0.20*skip_3_1 + 0.30*skip_6_1 + 0.50*skip_12_1",
        ),
        (
            ROC_MODE_CONSISTENCY_SKIP_BLEND_STR,
            "mean(skip_3_1, skip_6_1, skip_12_1) * positive_horizon_share",
        ),
        (
            ROC_MODE_ANTI_REVERSAL_SKIP_BLEND_STR,
            "weighted_skip_blend - 0.25*last_1m",
        ),
    ]
    for roc_mode_str, roc_definition_str in roc_variant_spec_list:
        variant_strategy_obj = _make_roc_variant_strategy_obj(
            pricing_data_df=pricing_data_df,
            vxn_scale_signal_df=vxn_scale_signal_df,
            roc_mode_str=roc_mode_str,
            atr_window_int=DEFAULT_ATR_WINDOW_INT,
            backtest_start_date_str=backtest_start_date_str,
            capital_base_float=capital_base_float,
            end_date_str=end_date_str,
        )
        strategy_result_list.append(
            (
                roc_mode_str,
                roc_definition_str,
                _run_strategy_obj(
                    strategy_obj=variant_strategy_obj,
                    pricing_data_df=pricing_data_df,
                    universe_df=universe_df,
                    backtest_start_date_str=backtest_start_date_str,
                ),
            )
        )

    comparison_row_list = [
        _comparison_row_dict(
            strategy_obj=strategy_obj,
            label_str=label_str,
            roc_definition_str=roc_definition_str,
            atr_window_int=DEFAULT_ATR_WINDOW_INT,
        )
        for label_str, roc_definition_str, strategy_obj in strategy_result_list
    ]
    comparison_df = pd.DataFrame(comparison_row_list)

    output_path = build_research_output_path(
        output_dir=output_dir_str,
        entity_type_str="strategy",
        entity_id_str=SUITE_ENTITY_ID_STR,
        analysis_type_str=SUITE_ANALYSIS_TYPE_STR,
    )
    output_path.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_path / "comparison_table.csv", index=False)

    display_column_list = [
        "strategy",
        "atr_window",
        "roc_definition",
        "start",
        "end",
        "final_equity",
        "ann_return_pct",
        "ann_vol_pct",
        "sharpe",
        "max_drawdown_pct",
        "mar",
        "trade_count",
    ]
    markdown_table_str = _markdown_table_str(comparison_df.loc[:, display_column_list])
    (output_path / "comparison_table.md").write_text(markdown_table_str + "\n", encoding="utf-8")

    equity_curve_df = pd.DataFrame(
        {
            label_str: strategy_obj.results["total_value"].astype(float)
            for label_str, _roc_definition_str, strategy_obj in strategy_result_list
        }
    )
    equity_curve_df.to_csv(output_path / "equity_curve.csv", index_label="date")
    _write_equity_curve_png(equity_curve_df=equity_curve_df, output_path=output_path / "equity_curve.png")

    metadata_dict = {
        "backtest_start_date": backtest_start_date_str,
        "end_date": end_date_str,
        "capital_base": float(capital_base_float),
        "shared_assumptions": {
            "universe": "Nasdaq 100 point-in-time membership",
            "regime_filter": "SPY close > trailing 200-day SMA",
            "stock_filter": "stock close > trailing 100-day SMA",
            "score": "selected monthly momentum formula / trailing ATR20",
            "vxn_scale": "clip(22 / VXN_close, 0.25, 1.0)",
            "execution": "month-end decision close, next tradable open",
        },
        "multiple_comparison_note": (
            "This is a research comparison of ROC windows, not a live-deployment approval."
        ),
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata_dict, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return comparison_df, output_path


def run_focused_atr_window_suite(
    backtest_start_date_str: str = DEFAULT_BACKTEST_START_DATE_STR,
    capital_base_float: float = DEFAULT_CAPITAL_BASE_FLOAT,
    end_date_str: str | None = None,
    output_dir_str: str = "results",
    atr_window_int_list: list[int] | None = None,
) -> tuple[pd.DataFrame, Path]:
    if atr_window_int_list is None:
        atr_window_int_list = list(DEFAULT_ATR_WINDOW_LIST)
    if len(atr_window_int_list) == 0:
        raise ValueError("atr_window_int_list must not be empty.")

    base_config_obj = replace(
        BASE_VXN_CONFIG,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=float(capital_base_float),
        end_date_str=end_date_str,
    )
    pricing_data_df, universe_df, _base_rebalance_schedule_df, vxn_scale_signal_df = (
        get_vxn_scaled_atr_normalized_ndx_data(config=base_config_obj)
    )

    focused_spec_list = [
        (
            "12m_base",
            ROC_MODE_LAST_12M_STR,
            "Close_ME_t / Close_ME_t-12 - 1",
        ),
        (
            "skip_12_1",
            ROC_MODE_SKIP_12_1_STR,
            "Close_ME_t-1 / Close_ME_t-12 - 1",
        ),
        (
            "weighted_skip_3_6_12",
            ROC_MODE_WEIGHTED_SKIP_BLEND_STR,
            "0.20*skip_3_1 + 0.30*skip_6_1 + 0.50*skip_12_1",
        ),
    ]

    strategy_result_list = []
    for strategy_label_str, roc_mode_str, roc_definition_str in focused_spec_list:
        for atr_window_int in atr_window_int_list:
            variant_strategy_obj = _make_roc_variant_strategy_obj(
                pricing_data_df=pricing_data_df,
                vxn_scale_signal_df=vxn_scale_signal_df,
                roc_mode_str=roc_mode_str,
                atr_window_int=atr_window_int,
                backtest_start_date_str=backtest_start_date_str,
                capital_base_float=capital_base_float,
                end_date_str=end_date_str,
            )
            variant_strategy_obj.name = f"{strategy_label_str}_atr{atr_window_int}"
            strategy_result_list.append(
                (
                    strategy_label_str,
                    roc_definition_str,
                    atr_window_int,
                    _run_strategy_obj(
                        strategy_obj=variant_strategy_obj,
                        pricing_data_df=pricing_data_df,
                        universe_df=universe_df,
                        backtest_start_date_str=backtest_start_date_str,
                    ),
                )
            )

    comparison_row_list = [
        _comparison_row_dict(
            strategy_obj=strategy_obj,
            label_str=label_str,
            roc_definition_str=roc_definition_str,
            atr_window_int=atr_window_int,
        )
        for label_str, roc_definition_str, atr_window_int, strategy_obj in strategy_result_list
    ]
    comparison_df = pd.DataFrame(comparison_row_list)

    output_path = build_research_output_path(
        output_dir=output_dir_str,
        entity_type_str="strategy",
        entity_id_str=FOCUSED_ATR_ENTITY_ID_STR,
        analysis_type_str=FOCUSED_ATR_ANALYSIS_TYPE_STR,
    )
    output_path.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_path / "comparison_table.csv", index=False)

    display_column_list = [
        "strategy",
        "atr_window",
        "roc_definition",
        "start",
        "end",
        "final_equity",
        "ann_return_pct",
        "ann_vol_pct",
        "sharpe",
        "max_drawdown_pct",
        "mar",
        "trade_count",
    ]
    markdown_table_str = _markdown_table_str(comparison_df.loc[:, display_column_list])
    (output_path / "comparison_table.md").write_text(markdown_table_str + "\n", encoding="utf-8")

    equity_curve_df = pd.DataFrame(
        {
            f"{label_str}_atr{atr_window_int}": strategy_obj.results["total_value"].astype(float)
            for label_str, _roc_definition_str, atr_window_int, strategy_obj in strategy_result_list
        }
    )
    equity_curve_df.to_csv(output_path / "equity_curve.csv", index_label="date")
    _write_equity_curve_png(equity_curve_df=equity_curve_df, output_path=output_path / "equity_curve.png")

    metadata_dict = {
        "backtest_start_date": backtest_start_date_str,
        "end_date": end_date_str,
        "capital_base": float(capital_base_float),
        "atr_windows": [int(atr_window_int) for atr_window_int in atr_window_int_list],
        "strategy_set": [spec_tuple[0] for spec_tuple in focused_spec_list],
        "shared_assumptions": {
            "universe": "Nasdaq 100 point-in-time membership",
            "regime_filter": "SPY close > trailing 200-day SMA",
            "stock_filter": "stock close > trailing 100-day SMA",
            "score": "selected monthly momentum formula / trailing ATR_N",
            "vxn_scale": "clip(22 / VXN_close, 0.25, 1.0)",
            "execution": "month-end decision close, next tradable open",
        },
        "multiple_comparison_note": (
            "This is a focused research comparison of ATR denominators and selected momentum formulas, "
            "not a live-deployment approval."
        ),
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata_dict, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return comparison_df, output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest-start-date", default=DEFAULT_BACKTEST_START_DATE_STR)
    parser.add_argument("--capital-base", type=float, default=DEFAULT_CAPITAL_BASE_FLOAT)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--focused-atr-grid",
        action="store_true",
        help="Run only 12m_base, skip_12_1, and weighted_skip_3_6_12 across ATR windows.",
    )
    parser.add_argument(
        "--atr-windows",
        default=",".join(str(atr_window_int) for atr_window_int in DEFAULT_ATR_WINDOW_LIST),
        help="Comma-separated ATR windows for --focused-atr-grid.",
    )
    arg_namespace = parser.parse_args()

    if arg_namespace.focused_atr_grid:
        comparison_df, output_path = run_focused_atr_window_suite(
            backtest_start_date_str=arg_namespace.backtest_start_date,
            capital_base_float=arg_namespace.capital_base,
            end_date_str=arg_namespace.end_date,
            output_dir_str=arg_namespace.output_dir,
            atr_window_int_list=_parse_atr_window_list(arg_namespace.atr_windows),
        )
    else:
        comparison_df, output_path = run_comparison_suite(
            backtest_start_date_str=arg_namespace.backtest_start_date,
            capital_base_float=arg_namespace.capital_base,
            end_date_str=arg_namespace.end_date,
            output_dir_str=arg_namespace.output_dir,
        )
    display_column_list = [
        "strategy",
        "atr_window",
        "roc_definition",
        "start",
        "end",
        "final_equity",
        "ann_return_pct",
        "ann_vol_pct",
        "sharpe",
        "max_drawdown_pct",
        "mar",
        "trade_count",
    ]
    print(_markdown_table_str(comparison_df.loc[:, display_column_list]))
    print(f"\nSaved comparison output to: {output_path}")


if __name__ == "__main__":
    main()
