"""
Run a GICS sector-cap sweep for the ATR-normalized momentum rotation.

Comparison arm to run_atr_normalized_ndx_corr_penalty_sweep: identical gates,
score, sizing, and execution; only the selection constraint differs (hard cap
of sector_cap names per current GICS level-1 sector instead of a correlation
penalty).

*** REALISM GAP *** Sector labels are Norgate's CURRENT GICS classification,
not point-in-time. See ASSUMPTIONS_AND_GAPS.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from alpha.engine.report import build_research_output_path
from strategies.momentum.run_atr_normalized_vix_scaled_universe_comparison import (
    WEIGHTING_EQUAL_STR,
    _comparison_row_dict,
    _run_strategy_obj,
)
from strategies.momentum.run_atr_normalized_ndx_corr_penalty_sweep import (
    _episode_metrics_dict,
    _markdown_table_str,
    _write_equity_curve_png,
    EPISODE_WINDOW_MAP,
)
from strategies.momentum.strategy_mo_atr_normalized_ndx import (
    AtrNormalizedNdxConfig,
    configure_total_return_benchmark_provenance,
    get_atr_normalized_ndx_data,
)
from strategies.momentum.strategy_mo_atr_normalized_sector_cap import (
    SectorCapAtrNormalizedStrategy,
    UNKNOWN_SECTOR_STR,
    build_current_gics_sector_map,
)


SUITE_ANALYSIS_TYPE_STR = "sector_cap_sweep"
SECTOR_CAP_LIST = [3, 4]
MAX_POSITIONS_LIST = [20]


def run_sector_cap_sweep(
    indexname_str: str = "Russell 1000",
    regime_symbol_str: str = "$RUI",
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
    output_dir_str: str = "results",
    timestamp_str: str | None = None,
    max_positions_list: list[int] | None = None,
    sector_cap_list: list[int] | None = None,
) -> tuple[pd.DataFrame, Path]:
    max_positions_list = list(max_positions_list or MAX_POSITIONS_LIST)
    sector_cap_list = list(sector_cap_list or SECTOR_CAP_LIST)

    base_config_obj = AtrNormalizedNdxConfig(
        indexname_str=indexname_str,
        regime_symbol_str=regime_symbol_str,
    )
    if backtest_start_date_str is not None or capital_base_float is not None or end_date_str is not None:
        base_config_obj = replace(
            base_config_obj,
            backtest_start_date_str=(
                base_config_obj.backtest_start_date_str
                if backtest_start_date_str is None
                else backtest_start_date_str
            ),
            capital_base_float=(
                base_config_obj.capital_base_float
                if capital_base_float is None
                else float(capital_base_float)
            ),
            end_date_str=end_date_str,
        )

    pricing_data_df, universe_df, rebalance_schedule_df = get_atr_normalized_ndx_data(
        base_config_obj,
        include_total_return_benchmark_bool=True,
    )
    tradeable_symbol_list = universe_df.columns.astype(str).tolist()
    sector_by_symbol_map = build_current_gics_sector_map(tradeable_symbol_list)

    universe_slug_str = "".join(
        char_str for char_str in indexname_str.lower().replace(" ", "_") if char_str.isalnum() or char_str == "_"
    )
    output_path = build_research_output_path(
        output_dir=output_dir_str,
        entity_type_str="strategy",
        entity_id_str=f"mo_atr_normalized_{universe_slug_str}_sector_cap_sweep",
        analysis_type_str=SUITE_ANALYSIS_TYPE_STR,
        timestamp_str=timestamp_str,
    )
    output_path.mkdir(parents=True, exist_ok=True)

    strategy_result_list = []
    comparison_row_list = []
    for max_positions_int in max_positions_list:
        for sector_cap_int in sector_cap_list:
            variant_label_str = f"n{max_positions_int}_cap{sector_cap_int}"
            config_obj = replace(base_config_obj, max_positions_int=max_positions_int)
            strategy_obj = SectorCapAtrNormalizedStrategy(
                name=f"strategy_mo_atr_normalized_sector_cap_{variant_label_str}",
                benchmarks=[config_obj.performance_benchmark_symbol_str],
                rebalance_schedule_df=rebalance_schedule_df,
                sector_by_symbol_map=sector_by_symbol_map,
                sector_cap_int=sector_cap_int,
                regime_symbol_str=config_obj.regime_symbol_str,
                capital_base=config_obj.capital_base_float,
                slippage=config_obj.slippage_float,
                commission_per_share=config_obj.commission_per_share_float,
                commission_minimum=config_obj.commission_minimum_float,
                lookback_month_int=config_obj.lookback_month_int,
                index_trend_window_int=config_obj.index_trend_window_int,
                stock_trend_window_int=config_obj.stock_trend_window_int,
                max_positions_int=config_obj.max_positions_int,
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
            strategy_result_list.append((variant_label_str, strategy_obj))

            row_dict = _comparison_row_dict(
                strategy_obj=strategy_obj,
                label_str=variant_label_str,
                universe_str=indexname_str,
                volatility_helper_str="none",
                max_positions_int=max_positions_int,
                weighting_scheme_str=WEIGHTING_EQUAL_STR,
                inverse_vol_window_int=None,
            )
            row_dict["sector_cap"] = sector_cap_int
            selection_audit_df = strategy_obj.get_selection_audit_df()
            row_dict["avg_max_sector_count"] = (
                float(selection_audit_df["max_sector_count_int"].mean())
                if len(selection_audit_df) > 0
                else None
            )
            row_dict.update(_episode_metrics_dict(strategy_obj.results["total_value"]))
            comparison_row_list.append(row_dict)

            comparison_df = pd.DataFrame(comparison_row_list)
            comparison_df.to_csv(output_path / "comparison_table.csv", index=False)
            audit_out_df = selection_audit_df.copy()
            audit_out_df["selected_symbol_list"] = audit_out_df["selected_symbol_list"].map("|".join)
            audit_out_df["sector_count_map"] = audit_out_df["sector_count_map"].map(json.dumps)
            audit_out_df.to_csv(output_path / f"selection_audit_{variant_label_str}.csv")
            print(f"finished variant: {variant_label_str}")

    comparison_df = pd.DataFrame(comparison_row_list)
    display_column_list = [
        column_str
        for column_str in [
            "variant",
            "max_positions_config",
            "sector_cap",
            "start",
            "end",
            "ann_return_pct",
            "ann_vol_pct",
            "sharpe",
            "max_drawdown_pct",
            "mar",
            "turnover_ann_pct",
            "avg_positions",
            "avg_max_sector_count",
            "missing_liquidations",
        ]
        if column_str in comparison_df.columns
    ]
    (output_path / "comparison_table.md").write_text(
        _markdown_table_str(comparison_df.loc[:, display_column_list]) + "\n",
        encoding="utf-8",
    )
    equity_curve_df = pd.DataFrame(
        {
            variant_label_str: strategy_obj.results["total_value"].astype(float)
            for variant_label_str, strategy_obj in strategy_result_list
        }
    )
    equity_curve_df.to_csv(output_path / "equity_curve.csv", index_label="date")
    _write_equity_curve_png(equity_curve_df, output_path / "equity_curve.png")

    unknown_count_int = sum(
        1 for sector_str in sector_by_symbol_map.values() if sector_str == UNKNOWN_SECTOR_STR
    )
    metadata_dict = {
        "universe": indexname_str,
        "regime_symbol": regime_symbol_str,
        "sector_cap_list": sector_cap_list,
        "max_positions_list": max_positions_list,
        "sector_source": "Norgate GICS level-1 ClassificationId, CURRENT labels (not point-in-time)",
        "unknown_sector_symbol_count": unknown_count_int,
        "total_symbol_count": len(sector_by_symbol_map),
        "episode_window_map": {k: list(v) for k, v in EPISODE_WINDOW_MAP.items()},
        "realism_gap_note": (
            "Current-label GICS applied to full history; see ASSUMPTIONS_AND_GAPS.md. "
            "The correlation-penalty sweep is the label-free control arm."
        ),
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata_dict, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote results: {output_path}")
    print(comparison_df.loc[:, display_column_list].to_string(index=False))
    return comparison_df, output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indexname", default="Russell 1000")
    parser.add_argument("--regime-symbol", default="$RUI")
    parser.add_argument("--backtest-start-date", default=None)
    parser.add_argument("--capital-base", type=float, default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--max-positions", default=None, help="Comma-separated N list, e.g. 20")
    parser.add_argument("--sector-caps", default=None, help="Comma-separated cap list, e.g. 3,4")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sector_cap_sweep(
        indexname_str=args.indexname,
        regime_symbol_str=args.regime_symbol,
        backtest_start_date_str=args.backtest_start_date,
        capital_base_float=args.capital_base,
        end_date_str=args.end_date,
        output_dir_str=args.output_dir,
        timestamp_str=args.timestamp,
        max_positions_list=(
            [int(v) for v in args.max_positions.split(",")] if args.max_positions else None
        ),
        sector_cap_list=(
            [int(v) for v in args.sector_caps.split(",")] if args.sector_caps else None
        ),
    )
