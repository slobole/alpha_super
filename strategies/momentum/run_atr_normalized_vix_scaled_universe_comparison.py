"""
Run the ATR-normalized monthly momentum universe comparison.

Rows:
1. Original Nasdaq 100 model with $VXN exposure scaling.
2. S&P 500 PIT-universe model with $VIX exposure scaling.
3. Russell 1000 PIT-universe model with $VIX exposure scaling.
4. Russell 3000 PIT-universe model with $VIX exposure scaling.
5. NYSE Composite PIT-universe model with $VIX exposure scaling.
6. NASDAQ Biotechnology PIT-universe model with $VIX exposure scaling.

All rows preserve the original signal, sizing, next-open execution, slippage,
commission, SPY regime filter, and max-position settings.
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
from alpha.engine.strategy import Strategy
from strategies.momentum.strategy_mo_atr_normalized_index_vix_scaled import (
    NASDAQ100_CONFIG,
    NASDAQ_BIOTECHNOLOGY_CONFIG,
    NYSE_COMPOSITE_CONFIG,
    RUSSELL1000_CONFIG,
    RUSSELL3000_CONFIG,
    SP500_CONFIG,
    VixScaledAtrNormalizedIndexConfig,
    build_inverse_vol_63_strategy,
    build_risk_parity_63_strategy,
    build_strategy,
    get_atr_normalized_index_data,
    get_vix_scaled_atr_normalized_index_data,
)
from strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled import (
    DEFAULT_CONFIG as NDX_VXN_CONFIG,
    VxnScaledAtrNormalizedNdxConfig,
    VxnScaledAtrNormalizedNdxStrategy,
    get_vxn_scaled_atr_normalized_ndx_data,
)


DEFAULT_BACKTEST_START_DATE_STR = NDX_VXN_CONFIG.backtest_start_date_str
DEFAULT_CAPITAL_BASE_FLOAT = NDX_VXN_CONFIG.capital_base_float
SUITE_ENTITY_ID_STR = "mo_atr_normalized_vix_scaled_universe_comparison"
SUITE_ANALYSIS_TYPE_STR = "index_universe_comparison"
WEIGHTING_EQUAL_STR = "equal"
WEIGHTING_INVERSE_VOL_63_STR = "inverse_vol_63"
WEIGHTING_RISK_PARITY_63_STR = "risk_parity_63"

UNIVERSE_KEY_LIST = [
    "original",
    "nasdaq100",
    "sp500",
    "russell1000",
    "russell3000",
    "nyse_composite",
    "nasdaq_biotechnology",
]


def _run_strategy_obj(
    strategy_obj: Strategy,
    pricing_data_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    backtest_start_date_str: str,
) -> Strategy:
    strategy_obj.universe_df = universe_df

    # *** CRITICAL*** The comparison keeps full pre-start price history for
    # trailing ROC, ATR, and SMA features, but only executes/reports from the
    # requested start date.
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


def _make_original_ndx_strategy_obj(
    config: VxnScaledAtrNormalizedNdxConfig,
    rebalance_schedule_df: pd.DataFrame,
    vxn_scale_signal_df: pd.DataFrame,
) -> VxnScaledAtrNormalizedNdxStrategy:
    return VxnScaledAtrNormalizedNdxStrategy(
        name="strategy_mo_atr_normalized_ndx_vxn_scaled",
        benchmarks=[config.regime_symbol_str],
        rebalance_schedule_df=rebalance_schedule_df,
        vxn_scale_signal_df=vxn_scale_signal_df,
        regime_symbol_str=config.regime_symbol_str,
        capital_base=config.capital_base_float,
        slippage=config.slippage_float,
        commission_per_share=config.commission_per_share_float,
        commission_minimum=config.commission_minimum_float,
        lookback_month_int=config.lookback_month_int,
        index_trend_window_int=config.index_trend_window_int,
        stock_trend_window_int=config.stock_trend_window_int,
        max_positions_int=config.max_positions_int,
    )


def _summary_value_obj(strategy_obj: Strategy, metric_name_str: str):
    summary_df = getattr(strategy_obj, "summary", None)
    if summary_df is None or metric_name_str not in summary_df.index:
        return None
    return summary_df.loc[metric_name_str, "Strategy"]


def _summary_value_float(strategy_obj: Strategy, metric_name_str: str) -> float | None:
    value_obj = _summary_value_obj(strategy_obj=strategy_obj, metric_name_str=metric_name_str)
    if value_obj is None or pd.isna(value_obj):
        return None
    return float(value_obj)


def _summary_value_date_str(strategy_obj: Strategy, metric_name_str: str) -> str | None:
    value_obj = _summary_value_obj(strategy_obj=strategy_obj, metric_name_str=metric_name_str)
    if value_obj is None or pd.isna(value_obj):
        return None
    return pd.Timestamp(value_obj).date().isoformat()


def _position_diagnostics_dict(strategy_obj: Strategy) -> dict[str, float | int | None]:
    realized_weight_df = getattr(strategy_obj, "realized_weight_df", pd.DataFrame()).copy()
    if len(realized_weight_df) == 0:
        return {
            "avg_positions": None,
            "median_positions": None,
            "max_positions": None,
            "avg_gross_exposure_pct": None,
            "avg_cash_weight_pct": None,
        }

    asset_weight_df = realized_weight_df.drop(columns=["Cash"], errors="ignore")
    position_count_ser = asset_weight_df.notna().sum(axis=1)
    gross_exposure_ser = asset_weight_df.fillna(0.0).abs().sum(axis=1)
    cash_ser = realized_weight_df.get(
        "Cash",
        pd.Series(index=realized_weight_df.index, dtype=float),
    ).astype(float)
    return {
        "avg_positions": float(position_count_ser.mean()),
        "median_positions": float(position_count_ser.median()),
        "max_positions": int(position_count_ser.max()),
        "avg_gross_exposure_pct": float(gross_exposure_ser.mean() * 100.0),
        "avg_cash_weight_pct": float(cash_ser.mean() * 100.0),
    }


def _comparison_row_dict(
    strategy_obj: Strategy,
    label_str: str,
    universe_str: str,
    volatility_helper_str: str,
    max_positions_int: int,
    weighting_scheme_str: str,
    inverse_vol_window_int: int | None,
) -> dict[str, object]:
    transaction_df = strategy_obj.get_transactions()
    missing_liquidation_count_int = 0
    if transaction_df is not None and len(transaction_df) > 0 and "order_id" in transaction_df.columns:
        missing_liquidation_count_int = int((transaction_df["order_id"] == -1).sum())

    return {
        "variant": label_str,
        "universe": universe_str,
        "regime_symbol": getattr(strategy_obj, "regime_symbol_str", None),
        "volatility_helper": volatility_helper_str,
        "weighting_scheme": weighting_scheme_str,
        "inverse_vol_window": inverse_vol_window_int,
        "max_positions_config": int(max_positions_int),
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
        "transactions": int(len(transaction_df)) if transaction_df is not None else None,
        "missing_liquidations": missing_liquidation_count_int,
        **_position_diagnostics_dict(strategy_obj),
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
    normalized_equity_df = equity_curve_df.apply(
        lambda equity_ser: equity_ser / equity_ser.dropna().iloc[0]
        if len(equity_ser.dropna()) > 0
        else equity_ser
    )
    normalized_equity_df.plot(ax=axis_obj, linewidth=1.6)
    axis_obj.set_title("ATR-Normalized Momentum Universe Comparison")
    axis_obj.set_ylabel("Growth of $1")
    axis_obj.set_xlabel("Date")
    axis_obj.grid(True, alpha=0.25)
    axis_obj.legend(loc="best", fontsize=9)
    figure_obj.tight_layout()
    figure_obj.savefig(output_path, dpi=160)
    plt.close(figure_obj)


def _selected_universe_key_list(universes_str: str | None) -> list[str]:
    if universes_str is None:
        return list(UNIVERSE_KEY_LIST)
    selected_key_list = [
        universe_key_str.strip()
        for universe_key_str in universes_str.split(",")
        if universe_key_str.strip()
    ]
    unknown_key_list = [
        universe_key_str
        for universe_key_str in selected_key_list
        if universe_key_str not in UNIVERSE_KEY_LIST
    ]
    if len(unknown_key_list) > 0:
        raise ValueError(f"Unknown universe keys: {unknown_key_list}. Valid keys: {UNIVERSE_KEY_LIST}.")
    return selected_key_list


def _validate_weighting_scheme_str(weighting_scheme_str: str) -> str:
    valid_weighting_set = {
        WEIGHTING_EQUAL_STR,
        WEIGHTING_INVERSE_VOL_63_STR,
        WEIGHTING_RISK_PARITY_63_STR,
    }
    if weighting_scheme_str not in valid_weighting_set:
        raise ValueError(
            "weighting_scheme_str must be one of "
            f"{sorted(valid_weighting_set)}, got {weighting_scheme_str!r}."
        )
    return weighting_scheme_str


def _common_override_dict(
    backtest_start_date_str: str,
    capital_base_float: float,
    end_date_str: str | None,
) -> dict[str, object]:
    return {
        "backtest_start_date_str": backtest_start_date_str,
        "capital_base_float": float(capital_base_float),
        "end_date_str": end_date_str,
    }


def _resolved_row_backtest_start_date_str(
    requested_backtest_start_date_str: str,
    config_backtest_start_date_str: str,
) -> str:
    requested_backtest_start_ts = pd.Timestamp(requested_backtest_start_date_str)
    config_backtest_start_ts = pd.Timestamp(config_backtest_start_date_str)
    return max(requested_backtest_start_ts, config_backtest_start_ts).date().isoformat()


def _run_original_row(
    backtest_start_date_str: str,
    capital_base_float: float,
    end_date_str: str | None,
) -> tuple[str, str, str, int, Strategy]:
    config_obj = replace(
        NDX_VXN_CONFIG,
        **_common_override_dict(
            backtest_start_date_str=backtest_start_date_str,
            capital_base_float=capital_base_float,
            end_date_str=end_date_str,
        ),
    )
    pricing_data_df, universe_df, rebalance_schedule_df, vxn_scale_signal_df = (
        get_vxn_scaled_atr_normalized_ndx_data(config=config_obj)
    )
    strategy_obj = _make_original_ndx_strategy_obj(
        config=config_obj,
        rebalance_schedule_df=rebalance_schedule_df,
        vxn_scale_signal_df=vxn_scale_signal_df,
    )
    return (
        "original_ndx_vxn_scaled",
        config_obj.indexname_str,
        config_obj.vxn_symbol_str,
        config_obj.max_positions_int,
        _run_strategy_obj(
            strategy_obj=strategy_obj,
            pricing_data_df=pricing_data_df,
            universe_df=universe_df,
            backtest_start_date_str=backtest_start_date_str,
        ),
    )


def _run_vix_scaled_row(
    config: VixScaledAtrNormalizedIndexConfig,
    backtest_start_date_str: str,
    capital_base_float: float,
    end_date_str: str | None,
    weighting_scheme_str: str,
) -> tuple[str, str, str, int, Strategy]:
    row_backtest_start_date_str = _resolved_row_backtest_start_date_str(
        requested_backtest_start_date_str=backtest_start_date_str,
        config_backtest_start_date_str=config.backtest_start_date_str,
    )
    config_obj = replace(
        config,
        **_common_override_dict(
            backtest_start_date_str=row_backtest_start_date_str,
            capital_base_float=capital_base_float,
            end_date_str=end_date_str,
        ),
    )
    if weighting_scheme_str == WEIGHTING_RISK_PARITY_63_STR:
        pricing_data_df, universe_df, rebalance_schedule_df = get_atr_normalized_index_data(
            config=config_obj,
        )
        strategy_obj = build_risk_parity_63_strategy(
            config=config_obj,
            rebalance_schedule_df=rebalance_schedule_df,
        )
        return (
            strategy_obj.name,
            config_obj.indexname_str,
            "none",
            config_obj.max_positions_int,
            _run_strategy_obj(
                strategy_obj=strategy_obj,
                pricing_data_df=pricing_data_df,
                universe_df=universe_df,
                backtest_start_date_str=row_backtest_start_date_str,
            ),
        )

    pricing_data_df, universe_df, rebalance_schedule_df, vix_scale_signal_df = (
        get_vix_scaled_atr_normalized_index_data(config=config_obj)
    )
    if weighting_scheme_str == WEIGHTING_INVERSE_VOL_63_STR:
        strategy_obj = build_inverse_vol_63_strategy(
            config=config_obj,
            rebalance_schedule_df=rebalance_schedule_df,
            vix_scale_signal_df=vix_scale_signal_df,
        )
    else:
        strategy_obj = build_strategy(
            config=config_obj,
            rebalance_schedule_df=rebalance_schedule_df,
            vix_scale_signal_df=vix_scale_signal_df,
        )
    return (
        strategy_obj.name,
        config_obj.indexname_str,
        config_obj.vxn_symbol_str,
        config_obj.max_positions_int,
        _run_strategy_obj(
            strategy_obj=strategy_obj,
            pricing_data_df=pricing_data_df,
            universe_df=universe_df,
            backtest_start_date_str=row_backtest_start_date_str,
        ),
    )


def run_comparison_suite(
    backtest_start_date_str: str = DEFAULT_BACKTEST_START_DATE_STR,
    capital_base_float: float = DEFAULT_CAPITAL_BASE_FLOAT,
    end_date_str: str | None = None,
    output_dir_str: str = "results",
    timestamp_str: str | None = None,
    universes_str: str | None = None,
    weighting_scheme_str: str = WEIGHTING_EQUAL_STR,
) -> tuple[pd.DataFrame, Path]:
    selected_key_list = _selected_universe_key_list(universes_str)
    weighting_scheme_str = _validate_weighting_scheme_str(weighting_scheme_str)
    if weighting_scheme_str == WEIGHTING_RISK_PARITY_63_STR and "original" in selected_key_list:
        raise ValueError(
            "Use universe key 'nasdaq100' for the pure risk_parity_63 Nasdaq 100 row. "
            "'original' is reserved for the deployment-reference NDX/VXN-scaled row."
        )
    strategy_result_list: list[tuple[str, str, str, int, Strategy]] = []

    for universe_key_str in selected_key_list:
        if universe_key_str == "original":
            strategy_result_list.append(
                _run_original_row(
                    backtest_start_date_str=backtest_start_date_str,
                    capital_base_float=capital_base_float,
                    end_date_str=end_date_str,
                )
            )
            continue

        config_by_key_dict = {
            "nasdaq100": NASDAQ100_CONFIG,
            "sp500": SP500_CONFIG,
            "russell1000": RUSSELL1000_CONFIG,
            "russell3000": RUSSELL3000_CONFIG,
            "nyse_composite": NYSE_COMPOSITE_CONFIG,
            "nasdaq_biotechnology": NASDAQ_BIOTECHNOLOGY_CONFIG,
        }
        strategy_result_list.append(
            _run_vix_scaled_row(
                config=config_by_key_dict[universe_key_str],
                backtest_start_date_str=backtest_start_date_str,
                capital_base_float=capital_base_float,
                end_date_str=end_date_str,
                weighting_scheme_str=weighting_scheme_str,
            )
        )

    comparison_row_list = [
        _comparison_row_dict(
            strategy_obj=strategy_obj,
            label_str=label_str,
            universe_str=universe_str,
            volatility_helper_str=volatility_helper_str,
            max_positions_int=max_positions_int,
            weighting_scheme_str=(
                WEIGHTING_EQUAL_STR
                if label_str == "original_ndx_vxn_scaled"
                else weighting_scheme_str
            ),
            inverse_vol_window_int=(
                None
                if label_str == "original_ndx_vxn_scaled"
                or weighting_scheme_str == WEIGHTING_EQUAL_STR
                else 63
            ),
        )
        for label_str, universe_str, volatility_helper_str, max_positions_int, strategy_obj
        in strategy_result_list
    ]
    comparison_df = pd.DataFrame(comparison_row_list)

    output_path = build_research_output_path(
        output_dir=output_dir_str,
        entity_type_str="strategy",
        entity_id_str=SUITE_ENTITY_ID_STR,
        analysis_type_str=SUITE_ANALYSIS_TYPE_STR,
        timestamp_str=timestamp_str,
    )
    output_path.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_path / "comparison_table.csv", index=False)

    display_column_list = [
        "variant",
        "universe",
        "regime_symbol",
        "volatility_helper",
        "weighting_scheme",
        "inverse_vol_window",
        "start",
        "end",
        "ann_return_pct",
        "ann_vol_pct",
        "sharpe",
        "max_drawdown_pct",
        "mar",
        "turnover_ann_pct",
        "cost_drag_ann_pct",
        "avg_positions",
        "avg_gross_exposure_pct",
        "missing_liquidations",
    ]
    markdown_table_str = _markdown_table_str(comparison_df.loc[:, display_column_list])
    (output_path / "comparison_table.md").write_text(markdown_table_str + "\n", encoding="utf-8")

    equity_curve_df = pd.DataFrame(
        {
            label_str: strategy_obj.results["total_value"].astype(float)
            for label_str, _universe_str, _volatility_helper_str, _max_positions_int, strategy_obj
            in strategy_result_list
        }
    )
    equity_curve_df.to_csv(output_path / "equity_curve.csv", index_label="date")
    _write_equity_curve_png(equity_curve_df=equity_curve_df, output_path=output_path / "equity_curve.png")

    metadata_dict = {
        "backtest_start_date": backtest_start_date_str,
        "end_date": end_date_str,
        "capital_base": float(capital_base_float),
        "variant_count": int(len(comparison_df)),
        "weighting_scheme": weighting_scheme_str,
        "search_space_note": (
            "Universe comparison only: original Nasdaq 100 plus requested broad-index variants."
        ),
        "shared_assumptions": {
            "signal": "12-month month-end ROC divided by trailing ATR20",
            "regime_filter": "configured regime benchmark close > trailing 200-day SMA",
            "stock_filter": "stock close > trailing 100-day SMA",
            "base_sizing": (
                "1 / max_positions per selected stock before volatility scaling"
                if weighting_scheme_str == WEIGHTING_EQUAL_STR
                else (
                    "pure risk parity: (1 / trailing 63-day close-to-close volatility) "
                    "/ sum_j(1 / trailing 63-day close-to-close volatility_j), no VIX/VXN scale"
                    if weighting_scheme_str == WEIGHTING_RISK_PARITY_63_STR
                    else "selected stocks weighted by normalized inverse trailing 63-day close-to-close volatility"
                )
            ),
            "volatility_scale": (
                "none"
                if weighting_scheme_str == WEIGHTING_RISK_PARITY_63_STR
                else "clip(22 / helper_close, 0.25, 1.0)"
            ),
            "execution": "month-end decision close, next tradable open",
            "slippage": NDX_VXN_CONFIG.slippage_float,
            "commission_per_share": NDX_VXN_CONFIG.commission_per_share_float,
            "commission_minimum": NDX_VXN_CONFIG.commission_minimum_float,
        },
        "volatility_helpers": {
            "original_ndx_vxn_scaled": "$VXN",
            "broad_index_variants": (
                "none"
                if weighting_scheme_str == WEIGHTING_RISK_PARITY_63_STR
                else "$VIX"
            ),
        },
        "regime_symbols_by_variant": {
            label_str: getattr(strategy_obj, "regime_symbol_str", None)
            for label_str, _universe_str, _volatility_helper_str, _max_positions_int, strategy_obj
            in strategy_result_list
        },
        "multiple_comparison_note": (
            "This is a requested-universe research comparison, not a live-deployment approval."
        ),
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata_dict, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"wrote results: {output_path}")
    print(comparison_df.loc[:, display_column_list].to_string(index=False))
    return comparison_df, output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backtest-start-date", default=DEFAULT_BACKTEST_START_DATE_STR)
    parser.add_argument("--capital-base", type=float, default=DEFAULT_CAPITAL_BASE_FLOAT)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument(
        "--universes",
        default=None,
        help=(
            "Comma-separated keys: original,nasdaq100,sp500,russell1000,"
            "russell3000,nyse_composite,nasdaq_biotechnology."
        ),
    )
    parser.add_argument(
        "--weighting",
        default=WEIGHTING_EQUAL_STR,
        choices=[
            WEIGHTING_EQUAL_STR,
            WEIGHTING_INVERSE_VOL_63_STR,
            WEIGHTING_RISK_PARITY_63_STR,
        ],
        help="Selected-name weighting scheme for broad-index variants.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_comparison_suite(
        backtest_start_date_str=args.backtest_start_date,
        capital_base_float=float(args.capital_base),
        end_date_str=args.end_date,
        output_dir_str=args.output_dir,
        timestamp_str=args.timestamp,
        universes_str=args.universes,
        weighting_scheme_str=args.weighting,
    )
