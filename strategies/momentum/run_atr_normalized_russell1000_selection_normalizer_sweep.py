"""
Run a Russell 1000 selection-normalizer sweep for the ATR momentum RP-63 model.

All rows keep the same contract:

    universe: Russell 1000 point-in-time members
    regime: $RUI close > $RUI SMA200
    sizing: pure RP-63 selected-name inverse volatility
    execution: month-end decision close, next tradable open

Only the cross-sectional selection score normalizer changes.
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

from alpha.engine.report import build_research_output_path
from strategies.momentum.run_atr_normalized_vix_scaled_universe_comparison import (
    WEIGHTING_RISK_PARITY_63_STR,
    _comparison_row_dict,
    _format_value_str,
    _run_strategy_obj,
)
from strategies.momentum.strategy_mo_atr_normalized_index_vix_scaled import (
    RUSSELL1000_CONFIG,
    SELECTION_SCORE_MODE_ATR20_STR,
    SELECTION_SCORE_MODE_NATR20_STR,
    SELECTION_SCORE_MODE_NATR63_STR,
    SELECTION_SCORE_MODE_VOL63_STR,
    build_risk_parity_63_strategy,
    get_atr_normalized_index_data,
)


SUITE_ENTITY_ID_STR = "mo_atr_normalized_russell1000_selection_normalizer_sweep"
SUITE_ANALYSIS_TYPE_STR = "selection_normalizer_sweep"

SELECTION_SCORE_MODE_LIST = [
    SELECTION_SCORE_MODE_ATR20_STR,
    SELECTION_SCORE_MODE_NATR20_STR,
    SELECTION_SCORE_MODE_NATR63_STR,
    SELECTION_SCORE_MODE_VOL63_STR,
]
SELECTION_SCORE_FORMULA_BY_MODE_DICT = {
    SELECTION_SCORE_MODE_ATR20_STR: "ROC12 / ATR20",
    SELECTION_SCORE_MODE_NATR20_STR: "ROC12 / (ATR20 / Close)",
    SELECTION_SCORE_MODE_NATR63_STR: "ROC12 / (ATR63 / Close)",
    SELECTION_SCORE_MODE_VOL63_STR: "ROC12 / trailing 63-day close-to-close volatility",
}


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
    axis_obj.set_title("Russell 1000 RP-63 Selection Normalizer Sweep")
    axis_obj.set_ylabel("Growth of $1")
    axis_obj.set_xlabel("Date")
    axis_obj.grid(True, alpha=0.25)
    axis_obj.legend(loc="best", fontsize=9)
    figure_obj.tight_layout()
    figure_obj.savefig(output_path, dpi=160)
    plt.close(figure_obj)


def run_selection_normalizer_sweep(
    backtest_start_date_str: str = RUSSELL1000_CONFIG.backtest_start_date_str,
    capital_base_float: float = RUSSELL1000_CONFIG.capital_base_float,
    end_date_str: str | None = None,
    output_dir_str: str = "results",
    timestamp_str: str | None = None,
) -> tuple[pd.DataFrame, Path]:
    base_config_obj = replace(
        RUSSELL1000_CONFIG,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=float(capital_base_float),
        end_date_str=end_date_str,
    )
    pricing_data_df, universe_df, rebalance_schedule_df = get_atr_normalized_index_data(
        config=base_config_obj,
    )

    strategy_result_list = []
    for selection_score_mode_str in SELECTION_SCORE_MODE_LIST:
        config_obj = replace(
            base_config_obj,
            selection_score_mode_str=selection_score_mode_str,
        )
        strategy_obj = build_risk_parity_63_strategy(
            config=config_obj,
            rebalance_schedule_df=rebalance_schedule_df,
        )
        strategy_obj = _run_strategy_obj(
            strategy_obj=strategy_obj,
            pricing_data_df=pricing_data_df,
            universe_df=universe_df,
            backtest_start_date_str=config_obj.backtest_start_date_str,
        )
        strategy_result_list.append((selection_score_mode_str, strategy_obj))

    comparison_row_list = []
    for selection_score_mode_str, strategy_obj in strategy_result_list:
        row_dict = _comparison_row_dict(
            strategy_obj=strategy_obj,
            label_str=strategy_obj.name,
            universe_str=base_config_obj.indexname_str,
            volatility_helper_str="none",
            max_positions_int=base_config_obj.max_positions_int,
            weighting_scheme_str=WEIGHTING_RISK_PARITY_63_STR,
            inverse_vol_window_int=base_config_obj.inverse_vol_window_int,
        )
        row_dict["selection_score_mode"] = selection_score_mode_str
        row_dict["selection_score_formula"] = SELECTION_SCORE_FORMULA_BY_MODE_DICT[
            selection_score_mode_str
        ]
        comparison_row_list.append(row_dict)

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
        "selection_score_mode",
        "selection_score_formula",
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
            selection_score_mode_str: strategy_obj.results["total_value"].astype(float)
            for selection_score_mode_str, strategy_obj in strategy_result_list
        }
    )
    equity_curve_df.to_csv(output_path / "equity_curve.csv", index_label="date")
    _write_equity_curve_png(
        equity_curve_df=equity_curve_df,
        output_path=output_path / "equity_curve.png",
    )

    metadata_dict = {
        "backtest_start_date": backtest_start_date_str,
        "end_date": end_date_str,
        "capital_base": float(capital_base_float),
        "variant_count": int(len(comparison_df)),
        "search_space_note": (
            "Four controlled Russell 1000 selection-score normalizers: "
            "raw ATR20, NATR20, NATR63, and close-to-close vol63."
        ),
        "shared_assumptions": {
            "universe": base_config_obj.indexname_str,
            "regime_filter": "$RUI close > trailing 200-day SMA",
            "stock_filter": "stock close > trailing 100-day SMA",
            "sizing": (
                "pure risk parity: (1 / trailing 63-day close-to-close volatility) "
                "/ sum_j(1 / trailing 63-day close-to-close volatility_j), no VIX/VXN scale"
            ),
            "execution": "month-end decision close, next tradable open",
            "slippage": base_config_obj.slippage_float,
            "commission_per_share": base_config_obj.commission_per_share_float,
            "commission_minimum": base_config_obj.commission_minimum_float,
        },
        "selection_score_formula_by_mode": SELECTION_SCORE_FORMULA_BY_MODE_DICT,
        "multiple_comparison_note": (
            "This is a four-row research sweep, not a live-deployment approval."
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
    parser.add_argument("--backtest-start-date", default=RUSSELL1000_CONFIG.backtest_start_date_str)
    parser.add_argument("--capital-base", type=float, default=RUSSELL1000_CONFIG.capital_base_float)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--timestamp", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_selection_normalizer_sweep(
        backtest_start_date_str=args.backtest_start_date,
        capital_base_float=float(args.capital_base),
        end_date_str=args.end_date,
        output_dir_str=args.output_dir,
        timestamp_str=args.timestamp,
    )
