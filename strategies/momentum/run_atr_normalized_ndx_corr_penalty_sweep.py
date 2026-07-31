"""
Run a correlation-penalty x position-count sweep for the ATR-normalized NDX
momentum rotation.

All rows keep the same contract:

    universe: Nasdaq 100 point-in-time members
    regime: SPY close > SPY SMA200
    stock filter: close > SMA100
    score: ROC12 / ATR20
    execution: month-end decision close, next tradable open

Only two things change per row:

    max_positions N in {10, 15, 20}
    correlation penalty lambda in {0.0, 0.25, 0.5, 1.0}

lambda = 0.0 rows reproduce the base top-N selection exactly and serve as the
per-N benchmark. Episode columns report behavior inside the concentrated-theme
drawdown windows that motivated the experiment.
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
import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from alpha.engine.report import build_research_output_path
from strategies.momentum.run_atr_normalized_vix_scaled_universe_comparison import (
    WEIGHTING_EQUAL_STR,
    _comparison_row_dict,
    _format_value_str,
    _run_strategy_obj,
)
from strategies.momentum.strategy_mo_atr_normalized_ndx import (
    configure_total_return_benchmark_provenance,
    get_atr_normalized_ndx_data,
)
from strategies.momentum.strategy_mo_atr_normalized_ndx_corr_penalty import (
    DEFAULT_CONFIG,
    build_corr_penalty_strategy,
)


SUITE_ENTITY_ID_STR = "mo_atr_normalized_ndx_corr_penalty_sweep"
SUITE_ANALYSIS_TYPE_STR = "corr_penalty_sweep"

MAX_POSITIONS_LIST = [10, 15, 20]
CORR_PENALTY_LAMBDA_LIST = [0.0, 0.25, 0.5, 1.0]

# Concentrated-theme stress windows. Each is [start, end] inclusive; the last
# window runs to the end of the backtest.
EPISODE_WINDOW_MAP = {
    "dot_com_2000_2002": ("2000-03-01", "2002-10-31"),
    "gfc_2007_2009": ("2007-10-01", "2009-03-31"),
    "tech_2021_2022": ("2021-11-01", "2022-12-31"),
    "chip_theme_2026": ("2026-05-01", None),
}


def _variant_label_str(
    max_positions_int: int,
    corr_penalty_lambda_float: float,
    corr_window_int: int | None = None,
) -> str:
    lambda_str = f"{corr_penalty_lambda_float:g}".replace(".", "p")
    label_str = f"n{max_positions_int}_lam{lambda_str}"
    if corr_window_int is not None:
        label_str = f"{label_str}_w{corr_window_int}"
    return label_str


def _markdown_table_str(display_df: pd.DataFrame) -> str:
    column_list = list(display_df.columns)
    header_str = "| " + " | ".join(column_list) + " |"
    separator_str = "| " + " | ".join(["---"] * len(column_list)) + " |"
    row_str_list = []
    for _row_index, row_ser in display_df.iterrows():
        value_str_list = [_format_value_str(row_ser[column_str]) for column_str in column_list]
        row_str_list.append("| " + " | ".join(value_str_list) + " |")
    return "\n".join([header_str, separator_str] + row_str_list)


def _episode_metrics_dict(total_value_ser: pd.Series) -> dict[str, float | None]:
    episode_metrics_map: dict[str, float | None] = {}
    clean_total_value_ser = total_value_ser.dropna().astype(float).sort_index()
    for episode_key_str, (start_date_str, end_date_str) in EPISODE_WINDOW_MAP.items():
        window_ser = clean_total_value_ser.loc[pd.Timestamp(start_date_str):]
        if end_date_str is not None:
            window_ser = window_ser.loc[: pd.Timestamp(end_date_str)]
        if len(window_ser) < 2:
            episode_metrics_map[f"{episode_key_str}_return_pct"] = None
            episode_metrics_map[f"{episode_key_str}_max_dd_pct"] = None
            continue
        window_return_float = float(window_ser.iloc[-1] / window_ser.iloc[0] - 1.0) * 100.0
        running_peak_ser = window_ser.cummax()
        drawdown_ser = window_ser / running_peak_ser - 1.0
        episode_metrics_map[f"{episode_key_str}_return_pct"] = window_return_float
        episode_metrics_map[f"{episode_key_str}_max_dd_pct"] = float(drawdown_ser.min()) * 100.0
    return episode_metrics_map


def _write_equity_curve_png(equity_curve_df: pd.DataFrame, output_path: Path) -> None:
    figure_obj, axis_obj = plt.subplots(figsize=(12, 7))
    normalized_equity_df = equity_curve_df.apply(
        lambda equity_ser: equity_ser / equity_ser.dropna().iloc[0]
        if len(equity_ser.dropna()) > 0
        else equity_ser
    )
    normalized_equity_df.plot(ax=axis_obj, linewidth=1.4, logy=True)
    axis_obj.set_title("NDX ATR-Normalized Momentum: Correlation-Penalty Sweep")
    axis_obj.set_ylabel("Growth of $1 (log scale)")
    axis_obj.set_xlabel("Date")
    axis_obj.grid(True, alpha=0.25)
    axis_obj.legend(loc="best", fontsize=8)
    figure_obj.tight_layout()
    figure_obj.savefig(output_path, dpi=160)
    plt.close(figure_obj)


def _write_selected_corr_png(selected_corr_df: pd.DataFrame, output_path: Path) -> None:
    figure_obj, axis_obj = plt.subplots(figsize=(12, 6))
    # 12-month rolling mean keeps the monthly series readable.
    selected_corr_df.rolling(window=12, min_periods=6).mean().plot(
        ax=axis_obj, linewidth=1.4
    )
    axis_obj.set_title("Realized Avg Pairwise Correlation of Selected Basket (12m rolling mean)")
    axis_obj.set_ylabel("Avg pairwise correlation")
    axis_obj.set_xlabel("Decision date")
    axis_obj.grid(True, alpha=0.25)
    axis_obj.legend(loc="best", fontsize=8)
    figure_obj.tight_layout()
    figure_obj.savefig(output_path, dpi=160)
    plt.close(figure_obj)


def run_corr_penalty_sweep(
    backtest_start_date_str: str = DEFAULT_CONFIG.backtest_start_date_str,
    capital_base_float: float = DEFAULT_CONFIG.capital_base_float,
    end_date_str: str | None = None,
    output_dir_str: str = "results",
    timestamp_str: str | None = None,
    max_positions_list: list[int] | None = None,
    corr_penalty_lambda_list: list[float] | None = None,
    indexname_str: str = DEFAULT_CONFIG.indexname_str,
    regime_symbol_str: str = DEFAULT_CONFIG.regime_symbol_str,
    corr_window_list: list[int] | None = None,
    min_dollar_adv_float: float = DEFAULT_CONFIG.min_dollar_adv_float,
) -> tuple[pd.DataFrame, Path]:
    max_positions_list = list(max_positions_list or MAX_POSITIONS_LIST)
    corr_penalty_lambda_list = list(corr_penalty_lambda_list or CORR_PENALTY_LAMBDA_LIST)
    corr_window_list = list(corr_window_list or [DEFAULT_CONFIG.corr_window_int])
    label_window_bool = len(corr_window_list) > 1

    base_config_obj = replace(
        DEFAULT_CONFIG,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=float(capital_base_float),
        end_date_str=end_date_str,
        indexname_str=indexname_str,
        regime_symbol_str=regime_symbol_str,
        min_dollar_adv_float=float(min_dollar_adv_float),
    )
    # Data prep does not depend on max_positions or lambda: load once, reuse.
    pricing_data_df, universe_df, rebalance_schedule_df = get_atr_normalized_ndx_data(
        base_config_obj,
        include_total_return_benchmark_bool=True,
    )

    # Distinct output folder per universe so NDX and Russell runs never mix.
    universe_slug_str = "".join(
        char_str for char_str in indexname_str.lower().replace(" ", "_") if char_str.isalnum() or char_str == "_"
    )
    suite_entity_id_str = (
        SUITE_ENTITY_ID_STR
        if indexname_str == DEFAULT_CONFIG.indexname_str
        else f"mo_atr_normalized_{universe_slug_str}_corr_penalty_sweep"
    )
    output_path = build_research_output_path(
        output_dir=output_dir_str,
        entity_type_str="strategy",
        entity_id_str=suite_entity_id_str,
        analysis_type_str=SUITE_ANALYSIS_TYPE_STR,
        timestamp_str=timestamp_str,
    )
    output_path.mkdir(parents=True, exist_ok=True)

    strategy_result_list = []
    comparison_df = pd.DataFrame()
    variant_tuple_list = []
    for max_positions_int in max_positions_list:
        for corr_window_int in corr_window_list:
            for corr_penalty_lambda_float in corr_penalty_lambda_list:
                # lambda = 0 ignores the correlation entirely: run it once for
                # the first window only instead of duplicating the baseline.
                if corr_penalty_lambda_float == 0.0 and corr_window_int != corr_window_list[0]:
                    continue
                variant_tuple_list.append(
                    (max_positions_int, corr_window_int, corr_penalty_lambda_float)
                )

    for max_positions_int, corr_window_int, corr_penalty_lambda_float in variant_tuple_list:
            variant_label_str = _variant_label_str(
                max_positions_int=max_positions_int,
                corr_penalty_lambda_float=corr_penalty_lambda_float,
                corr_window_int=corr_window_int if label_window_bool else None,
            )
            config_obj = replace(
                base_config_obj,
                max_positions_int=max_positions_int,
                corr_penalty_lambda_float=corr_penalty_lambda_float,
                corr_window_int=corr_window_int,
            )
            strategy_obj = build_corr_penalty_strategy(
                config=config_obj,
                rebalance_schedule_df=rebalance_schedule_df,
                name_str=f"strategy_mo_atr_normalized_ndx_corr_penalty_{variant_label_str}",
            )
            configure_total_return_benchmark_provenance(
                strategy_obj=strategy_obj,
                config_obj=config_obj,
            )
            strategy_obj = _run_strategy_obj(
                strategy_obj=strategy_obj,
                pricing_data_df=pricing_data_df,
                universe_df=universe_df,
                backtest_start_date_str=config_obj.backtest_start_date_str,
            )
            strategy_result_list.append(
                (variant_label_str, max_positions_int, corr_penalty_lambda_float, strategy_obj)
            )
            # Persist everything after every variant so an interrupted sweep
            # still leaves usable partial results on disk.
            comparison_df = _write_sweep_outputs(
                strategy_result_list=strategy_result_list,
                base_config_obj=base_config_obj,
                output_path=output_path,
            )
            print(f"finished variant: {variant_label_str} ({len(strategy_result_list)} rows written)")
    display_column_list = [
        column_str for column_str in DISPLAY_COLUMN_LIST if column_str in comparison_df.columns
    ]
    metadata_dict = {
        "backtest_start_date": backtest_start_date_str,
        "end_date": end_date_str,
        "capital_base": float(capital_base_float),
        "variant_count": int(len(comparison_df)),
        "max_positions_list": max_positions_list,
        "corr_penalty_lambda_list": corr_penalty_lambda_list,
        "corr_window": base_config_obj.corr_window_int,
        "corr_min_overlap": base_config_obj.corr_min_overlap_int,
        "min_dollar_adv": base_config_obj.min_dollar_adv_float,
        "adv_window": base_config_obj.adv_window_int,
        "episode_window_map": {
            episode_key_str: list(window_tuple)
            for episode_key_str, window_tuple in EPISODE_WINDOW_MAP.items()
        },
        "shared_assumptions": {
            "universe": base_config_obj.indexname_str,
            "regime_filter": f"{base_config_obj.regime_symbol_str} close > trailing 200-day SMA",
            "stock_filter": "stock close > trailing 100-day SMA",
            "selection_score": "ROC12 / ATR20",
            "selection_rule": (
                "greedy: slot 1 = top raw score; slot k>1 maximizes "
                "score - lambda * avg_corr_to_selected * |score|"
            ),
            "corr_estimator": (
                "Pearson correlation of daily close-to-close returns over the "
                "trailing corr_window days ending at the decision close; NaN "
                "pairs fall back to the median valid candidate correlation"
            ),
            "weighting": "equal weight 1/N per selected name",
            "execution": "month-end decision close, next tradable open",
            "slippage": base_config_obj.slippage_float,
            "commission_per_share": base_config_obj.commission_per_share_float,
            "commission_minimum": base_config_obj.commission_minimum_float,
        },
        "multiple_comparison_note": (
            "This is a 12-row research sweep motivated by the 2026 chip-theme "
            "drawdown. Any lambda/N chosen from this table is in-sample on that "
            "episode; treat improvements as a hypothesis, not a live-deployment "
            "approval."
        ),
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata_dict, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"wrote results: {output_path}")
    print(comparison_df.loc[:, display_column_list].to_string(index=False))
    return comparison_df, output_path


DISPLAY_COLUMN_LIST = [
    "variant",
    "max_positions_config",
    "corr_penalty_lambda",
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
    "avg_selected_pairwise_corr",
    "dot_com_2000_2002_max_dd_pct",
    "gfc_2007_2009_max_dd_pct",
    "tech_2021_2022_max_dd_pct",
    "chip_theme_2026_return_pct",
    "chip_theme_2026_max_dd_pct",
    "missing_liquidations",
]


def _write_sweep_outputs(
    strategy_result_list: list[tuple[str, int, float, object]],
    base_config_obj,
    output_path: Path,
) -> pd.DataFrame:
    comparison_row_list = []
    for variant_label_str, max_positions_int, corr_penalty_lambda_float, strategy_obj in strategy_result_list:
        row_dict = _comparison_row_dict(
            strategy_obj=strategy_obj,
            label_str=variant_label_str,
            universe_str=base_config_obj.indexname_str,
            volatility_helper_str="none",
            max_positions_int=max_positions_int,
            weighting_scheme_str=WEIGHTING_EQUAL_STR,
            inverse_vol_window_int=None,
        )
        row_dict["corr_penalty_lambda"] = corr_penalty_lambda_float
        row_dict["corr_window"] = int(strategy_obj.corr_window_int)

        selection_audit_df = strategy_obj.get_selection_audit_df()
        avg_corr_ser = selection_audit_df["avg_selected_pairwise_corr_float"].dropna()
        row_dict["avg_selected_pairwise_corr"] = (
            float(avg_corr_ser.mean()) if len(avg_corr_ser) > 0 else None
        )
        row_dict.update(_episode_metrics_dict(strategy_obj.results["total_value"]))
        comparison_row_list.append(row_dict)

        # Persist the per-month selection audit so downstream liquidity and
        # concentration analyses can run without re-running the backtest.
        audit_out_df = selection_audit_df.copy()
        audit_out_df["selected_symbol_list"] = audit_out_df["selected_symbol_list"].map("|".join)
        audit_out_df.to_csv(output_path / f"selection_audit_{variant_label_str}.csv")

    comparison_df = pd.DataFrame(comparison_row_list)
    comparison_df.to_csv(output_path / "comparison_table.csv", index=False)

    display_column_list = [
        column_str for column_str in DISPLAY_COLUMN_LIST if column_str in comparison_df.columns
    ]
    markdown_table_str = _markdown_table_str(comparison_df.loc[:, display_column_list])
    (output_path / "comparison_table.md").write_text(markdown_table_str + "\n", encoding="utf-8")

    equity_curve_df = pd.DataFrame(
        {
            variant_label_str: strategy_obj.results["total_value"].astype(float)
            for variant_label_str, _max_positions_int, _lambda_float, strategy_obj in strategy_result_list
        }
    )
    equity_curve_df.to_csv(output_path / "equity_curve.csv", index_label="date")
    _write_equity_curve_png(
        equity_curve_df=equity_curve_df,
        output_path=output_path / "equity_curve.png",
    )

    selected_corr_df = pd.DataFrame(
        {
            variant_label_str: strategy_obj.get_selection_audit_df()[
                "avg_selected_pairwise_corr_float"
            ]
            for variant_label_str, _max_positions_int, _lambda_float, strategy_obj in strategy_result_list
        }
    )
    selected_corr_df.to_csv(output_path / "selected_basket_avg_pairwise_corr.csv", index_label="decision_date")
    _write_selected_corr_png(
        selected_corr_df=selected_corr_df,
        output_path=output_path / "selected_basket_avg_pairwise_corr.png",
    )
    return comparison_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backtest-start-date", default=DEFAULT_CONFIG.backtest_start_date_str)
    parser.add_argument("--capital-base", type=float, default=DEFAULT_CONFIG.capital_base_float)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--indexname", default=DEFAULT_CONFIG.indexname_str)
    parser.add_argument("--regime-symbol", default=DEFAULT_CONFIG.regime_symbol_str)
    parser.add_argument(
        "--max-positions",
        default=None,
        help="Comma-separated N list, e.g. 10,20. Defaults to the module list.",
    )
    parser.add_argument(
        "--lambdas",
        default=None,
        help="Comma-separated lambda list, e.g. 0,0.5,1. Defaults to the module list.",
    )
    parser.add_argument(
        "--corr-windows",
        default=None,
        help="Comma-separated correlation window list, e.g. 63,126,252. Defaults to the config window.",
    )
    parser.add_argument(
        "--min-adv",
        type=float,
        default=DEFAULT_CONFIG.min_dollar_adv_float,
        help="Minimum trailing median dollar ADV for candidate eligibility. 0 disables the gate.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_corr_penalty_sweep(
        backtest_start_date_str=args.backtest_start_date,
        capital_base_float=float(args.capital_base),
        end_date_str=args.end_date,
        output_dir_str=args.output_dir,
        timestamp_str=args.timestamp,
        indexname_str=args.indexname,
        regime_symbol_str=args.regime_symbol,
        max_positions_list=(
            [int(value_str) for value_str in args.max_positions.split(",")]
            if args.max_positions
            else None
        ),
        corr_penalty_lambda_list=(
            [float(value_str) for value_str in args.lambdas.split(",")]
            if args.lambdas
            else None
        ),
        corr_window_list=(
            [int(value_str) for value_str in args.corr_windows.split(",")]
            if args.corr_windows
            else None
        ),
        min_dollar_adv_float=float(args.min_adv),
    )
