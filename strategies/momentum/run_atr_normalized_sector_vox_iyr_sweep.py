"""Run the frozen 12-row trend matrix for the VOX/IYR sector ETF basket."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
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

from alpha.engine.backtest import run_daily
from alpha.engine.report import build_research_output_path
from strategies.momentum.run_atr_normalized_vix_scaled_universe_comparison import (
    WEIGHTING_EQUAL_STR,
    _comparison_row_dict,
)
from strategies.momentum.strategy_mo_atr_normalized_sector_vox_iyr import (
    DEFAULT_CONFIG,
    DIMENSIONLESS_NATR_SCORE_STR,
    SOURCE_ATR_SCORE_STR,
    AtrNormalizedSectorConfig,
    build_strategy,
    get_atr_normalized_sector_data,
)


SUITE_ENTITY_ID_STR = "strategy_mo_atr_normalized_sector_vox_iyr"
SUITE_ANALYSIS_TYPE_STR = "trend_matrix"
NATR_AUDIT_ANALYSIS_TYPE_STR = "natr_score_audit"

# Frozen before the first run. These 12 rows are the complete search space.
VARIANT_SPEC_TUPLE = (
    ("n3_both_source", 3, True, True, SOURCE_ATR_SCORE_STR, False),
    ("n5_both_source", 5, True, True, SOURCE_ATR_SCORE_STR, False),
    ("n3_asset_only_source", 3, False, True, SOURCE_ATR_SCORE_STR, False),
    ("n5_asset_only_source", 5, False, True, SOURCE_ATR_SCORE_STR, False),
    ("n3_market_only_source", 3, True, False, SOURCE_ATR_SCORE_STR, False),
    ("n5_market_only_source", 5, True, False, SOURCE_ATR_SCORE_STR, False),
    ("n3_no_filters_source", 3, False, False, SOURCE_ATR_SCORE_STR, False),
    ("n5_no_filters_source", 5, False, False, SOURCE_ATR_SCORE_STR, False),
    ("n3_both_source_vix", 3, True, True, SOURCE_ATR_SCORE_STR, True),
    ("n5_both_source_vix", 5, True, True, SOURCE_ATR_SCORE_STR, True),
    ("n3_both_natr", 3, True, True, DIMENSIONLESS_NATR_SCORE_STR, False),
    ("n5_both_natr", 5, True, True, DIMENSIONLESS_NATR_SCORE_STR, False),
)

# Reviewer-mandated score-invariance audit. This is a separate, explicitly
# post-hoc matrix and is not represented as part of the original frozen 12 rows.
NATR_AUDIT_VARIANT_SPEC_TUPLE = (
    ("n3_both_natr", 3, True, True, DIMENSIONLESS_NATR_SCORE_STR, False),
    ("n5_both_natr", 5, True, True, DIMENSIONLESS_NATR_SCORE_STR, False),
    ("n3_asset_only_natr", 3, False, True, DIMENSIONLESS_NATR_SCORE_STR, False),
    ("n5_asset_only_natr", 5, False, True, DIMENSIONLESS_NATR_SCORE_STR, False),
    ("n3_market_only_natr", 3, True, False, DIMENSIONLESS_NATR_SCORE_STR, False),
    ("n5_market_only_natr", 5, True, False, DIMENSIONLESS_NATR_SCORE_STR, False),
    ("n3_no_filters_natr", 3, False, False, DIMENSIONLESS_NATR_SCORE_STR, False),
    ("n5_no_filters_natr", 5, False, False, DIMENSIONLESS_NATR_SCORE_STR, False),
)

DISPLAY_COLUMN_LIST = [
    "variant",
    "max_positions_config",
    "market_sma200",
    "asset_sma100",
    "score_mode",
    "vix_scaled",
    "start",
    "end",
    "ann_return_pct",
    "ann_vol_pct",
    "sharpe",
    "max_drawdown_pct",
    "mar",
    "turnover_ann_pct",
    "cost_drag_ann_pct",
    "avg_gross_exposure_pct",
    "transactions",
    "missing_liquidations",
]


def _markdown_table_str(display_df: pd.DataFrame) -> str:
    column_list = list(display_df.columns)
    header_str = "| " + " | ".join(column_list) + " |"
    separator_str = "| " + " | ".join(["---"] * len(column_list)) + " |"
    row_str_list = []
    for _row_index, row_ser in display_df.iterrows():
        value_str_list = []
        for column_str in column_list:
            value_obj = row_ser[column_str]
            if value_obj is None or pd.isna(value_obj):
                value_str_list.append("")
            elif isinstance(value_obj, (float, np.floating)):
                value_str_list.append(f"{float(value_obj):,.2f}")
            else:
                value_str_list.append(str(value_obj))
        row_str_list.append("| " + " | ".join(value_str_list) + " |")
    return "\n".join([header_str, separator_str] + row_str_list)


def _subperiod_metric_row_dict(
    total_value_ser: pd.Series,
    start_date_str: str,
    end_date_str: str,
) -> dict[str, float | str]:
    window_value_ser = total_value_ser.loc[
        pd.Timestamp(start_date_str) : pd.Timestamp(end_date_str)
    ].dropna()
    if len(window_value_ser) < 2:
        return {
            "period": f"{start_date_str}_{end_date_str}",
            "cagr_pct": np.nan,
            "sharpe": np.nan,
            "max_drawdown_pct": np.nan,
        }
    daily_return_ser = window_value_ser.pct_change().dropna()
    elapsed_year_float = (
        pd.Timestamp(window_value_ser.index[-1])
        - pd.Timestamp(window_value_ser.index[0])
    ).days / 365.25
    cagr_float = (
        float(window_value_ser.iloc[-1] / window_value_ser.iloc[0])
        ** (1.0 / elapsed_year_float)
        - 1.0
    )
    annual_vol_float = float(daily_return_ser.std(ddof=1) * np.sqrt(252.0))
    sharpe_float = (
        float(daily_return_ser.mean() * 252.0 / annual_vol_float)
        if annual_vol_float > 0.0
        else np.nan
    )
    drawdown_ser = window_value_ser / window_value_ser.cummax() - 1.0
    return {
        "period": f"{start_date_str}_{end_date_str}",
        "cagr_pct": cagr_float * 100.0,
        "sharpe": sharpe_float,
        "max_drawdown_pct": float(drawdown_ser.min()) * 100.0,
    }


def _write_equity_curve_png(
    equity_curve_df: pd.DataFrame,
    output_path: Path,
) -> None:
    normalized_equity_df = equity_curve_df.apply(
        lambda value_ser: value_ser / value_ser.dropna().iloc[0]
    )
    figure_obj, axis_obj = plt.subplots(figsize=(13, 8))
    normalized_equity_df.plot(ax=axis_obj, logy=True, linewidth=1.25)
    axis_obj.set_title("US Sector ETF ATR-Normalized Trend Matrix")
    axis_obj.set_ylabel("Growth of $1 (log scale)")
    axis_obj.set_xlabel("Date")
    axis_obj.grid(True, alpha=0.25)
    axis_obj.legend(loc="best", fontsize=7, ncol=2)
    figure_obj.tight_layout()
    figure_obj.savefig(output_path, dpi=160)
    plt.close(figure_obj)


def _file_sha256_str(file_path: Path) -> str:
    sha256_obj = hashlib.sha256()
    with file_path.open("rb") as file_obj:
        for content_bytes in iter(lambda: file_obj.read(1024 * 1024), b""):
            sha256_obj.update(content_bytes)
    return sha256_obj.hexdigest()


def _git_provenance_dict() -> dict[str, object]:
    strategy_path = Path(
        "strategies/momentum/strategy_mo_atr_normalized_sector_vox_iyr.py"
    )
    sweep_path = Path(
        "strategies/momentum/run_atr_normalized_sector_vox_iyr_sweep.py"
    )
    git_head_str = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT_PATH,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    relevant_status_str = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            str(strategy_path),
            str(sweep_path),
        ],
        cwd=REPO_ROOT_PATH,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "git_head": git_head_str,
        "relevant_worktree_dirty": bool(relevant_status_str),
        "relevant_git_status": relevant_status_str.splitlines(),
        "strategy_sha256": _file_sha256_str(REPO_ROOT_PATH / strategy_path),
        "sweep_sha256": _file_sha256_str(REPO_ROOT_PATH / sweep_path),
    }


def run_sweep(
    backtest_start_date_str: str = DEFAULT_CONFIG.backtest_start_date_str,
    capital_base_float: float = DEFAULT_CONFIG.capital_base_float,
    end_date_str: str | None = None,
    output_dir_str: str = "results",
    timestamp_str: str | None = None,
    matrix_name_str: str = "initial",
) -> tuple[pd.DataFrame, Path]:
    if matrix_name_str == "initial":
        variant_spec_tuple = VARIANT_SPEC_TUPLE
        analysis_type_str = SUITE_ANALYSIS_TYPE_STR
        matrix_design_note_str = (
            "Original 12-row matrix frozen before its first result was observed."
        )
    elif matrix_name_str == "natr-audit":
        variant_spec_tuple = NATR_AUDIT_VARIANT_SPEC_TUPLE
        analysis_type_str = NATR_AUDIT_ANALYSIS_TYPE_STR
        matrix_design_note_str = (
            "Post-hoc score-invariance audit requested after quant review found "
            "that source ROC12/ATR20 rankings have inverse-dollar units."
        )
    else:
        raise ValueError(f"Unsupported matrix_name_str: {matrix_name_str}.")

    base_config_obj = replace(
        DEFAULT_CONFIG,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=float(capital_base_float),
        end_date_str=end_date_str,
    )
    (
        pricing_data_df,
        universe_df,
        rebalance_schedule_df,
        vix_scale_signal_df,
    ) = get_atr_normalized_sector_data(
        config_obj=base_config_obj,
        include_total_return_benchmark_bool=True,
    )
    output_path = build_research_output_path(
        output_dir=output_dir_str,
        entity_type_str="strategy",
        entity_id_str=SUITE_ENTITY_ID_STR,
        analysis_type_str=analysis_type_str,
        timestamp_str=timestamp_str,
    )
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing research artifact directory: {output_path}"
        )
    output_path.mkdir(parents=True, exist_ok=True)

    comparison_row_list = []
    equity_curve_map: dict[str, pd.Series] = {}
    subperiod_row_list = []
    calendar_idx = pricing_data_df.index[
        pricing_data_df.index >= pd.Timestamp(backtest_start_date_str)
    ]
    actual_backtest_end_date_str = pd.Timestamp(calendar_idx[-1]).date().isoformat()
    for (
        variant_name_str,
        max_positions_int,
        apply_market_trend_bool,
        apply_asset_trend_bool,
        score_mode_str,
        use_vix_scale_bool,
    ) in variant_spec_tuple:
        config_obj = replace(
            base_config_obj,
            max_positions_int=max_positions_int,
            apply_market_trend_bool=apply_market_trend_bool,
            apply_asset_trend_bool=apply_asset_trend_bool,
            score_mode_str=score_mode_str,
            use_vix_scale_bool=use_vix_scale_bool,
        )
        strategy_obj = build_strategy(
            config_obj=config_obj,
            rebalance_schedule_df=rebalance_schedule_df,
            vix_scale_signal_df=vix_scale_signal_df,
            name_str=f"{SUITE_ENTITY_ID_STR}_{variant_name_str}",
        )
        strategy_obj.universe_df = universe_df
        # *** CRITICAL*** Signal features end at month-end Close_T. The engine
        # receives orders only on the next session, so fills occur at Open_T+1.
        run_daily(
            strategy_obj,
            pricing_data_df,
            calendar=calendar_idx,
            show_progress=False,
            show_signal_progress_bool=False,
            audit_override_bool=None,
        )
        comparison_row_dict = _comparison_row_dict(
            strategy_obj=strategy_obj,
            label_str=variant_name_str,
            universe_str="XLB,XLE,XLF,XLI,XLK,XLP,XLU,XLV,XLY,VOX,IYR",
            volatility_helper_str=(
                config_obj.vix_symbol_str if use_vix_scale_bool else "none"
            ),
            max_positions_int=max_positions_int,
            weighting_scheme_str=WEIGHTING_EQUAL_STR,
            inverse_vol_window_int=None,
        )
        comparison_row_dict.update(
            {
                "market_sma200": apply_market_trend_bool,
                "asset_sma100": apply_asset_trend_bool,
                "score_mode": score_mode_str,
                "vix_scaled": use_vix_scale_bool,
            }
        )
        comparison_row_list.append(comparison_row_dict)
        equity_curve_map[variant_name_str] = strategy_obj.results[
            "total_value"
        ].astype(float)
        for period_start_str, period_end_str in (
            ("2006-01-01", "2012-12-31"),
            ("2013-01-01", "2019-12-31"),
            ("2020-01-01", actual_backtest_end_date_str),
        ):
            subperiod_row_dict = _subperiod_metric_row_dict(
                total_value_ser=strategy_obj.results["total_value"].astype(float),
                start_date_str=period_start_str,
                end_date_str=period_end_str,
            )
            subperiod_row_dict["variant"] = variant_name_str
            subperiod_row_list.append(subperiod_row_dict)

        comparison_df = pd.DataFrame(comparison_row_list)
        comparison_df.to_csv(output_path / "comparison_table.csv", index=False)
        print(
            f"finished {variant_name_str}: "
            f"{len(comparison_row_list)}/{len(variant_spec_tuple)}"
        )

    display_df = comparison_df.loc[:, DISPLAY_COLUMN_LIST]
    (output_path / "comparison_table.md").write_text(
        _markdown_table_str(display_df) + "\n",
        encoding="utf-8",
    )
    subperiod_df = pd.DataFrame(subperiod_row_list)
    subperiod_df.to_csv(output_path / "subperiod_table.csv", index=False)
    equity_curve_df = pd.DataFrame(equity_curve_map)
    equity_curve_df.to_csv(output_path / "equity_curve.csv", index_label="date")
    _write_equity_curve_png(
        equity_curve_df=equity_curve_df,
        output_path=output_path / "equity_curve.png",
    )
    metadata_dict = {
        "matrix_name": matrix_name_str,
        "matrix_design_note": matrix_design_note_str,
        "variant_count": len(variant_spec_tuple),
        "variant_spec": [
            list(variant_spec_obj) for variant_spec_obj in variant_spec_tuple
        ],
        "universe": list(base_config_obj.sector_symbol_tuple),
        "resolved_data": {
            "price_start": pd.Timestamp(pricing_data_df.index[0]).date().isoformat(),
            "backtest_start": pd.Timestamp(calendar_idx[0]).date().isoformat(),
            "backtest_end": actual_backtest_end_date_str,
            "norgatedata_package_version": importlib.metadata.version("norgatedata"),
            "execution_and_marks_adjustment": "CAPITALSPECIAL",
            "performance_benchmark_adjustment": "TOTALRETURN",
        },
        "code_provenance": _git_provenance_dict(),
        "shared_rules": {
            "decision": "actual month-end Close_T",
            "execution": "next tradable Open_T+1",
            "lookback_months": base_config_obj.lookback_month_int,
            "atr_days": 20,
            "source_score": "ROC12 / ATR20",
            "dimensionless_score": "ROC12 / (ATR20 / Close_T)",
            "market_filter": "optional SPY Close_T > SMA200_T",
            "asset_filter": "optional ETF Close_T > SMA100_T",
            "weighting": "equal fixed slots, 1 / max_positions",
            "vix_scale": "optional clip(20 / VIX_T, 0.25, 1.00)",
            "slippage": base_config_obj.slippage_float,
            "commission_per_share": base_config_obj.commission_per_share_float,
            "commission_minimum": base_config_obj.commission_minimum_float,
        },
        "multiple_comparison_note": (
            f"This {len(variant_spec_tuple)}-row matrix is in-sample. Any winner "
            "remains a research hypothesis until validated on an untouched "
            "period or forward shadow ledger."
        ),
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata_dict, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote results: {output_path}")
    print(display_df.to_string(index=False))
    return comparison_df, output_path


def parse_args() -> argparse.Namespace:
    parser_obj = argparse.ArgumentParser(description=__doc__)
    parser_obj.add_argument(
        "--backtest-start-date",
        default=DEFAULT_CONFIG.backtest_start_date_str,
    )
    parser_obj.add_argument(
        "--capital-base",
        type=float,
        default=DEFAULT_CONFIG.capital_base_float,
    )
    parser_obj.add_argument("--end-date", default=None)
    parser_obj.add_argument("--output-dir", default="results")
    parser_obj.add_argument("--timestamp", default=None)
    parser_obj.add_argument(
        "--matrix",
        choices=("initial", "natr-audit"),
        default="initial",
    )
    return parser_obj.parse_args()


if __name__ == "__main__":
    args_obj = parse_args()
    run_sweep(
        backtest_start_date_str=args_obj.backtest_start_date,
        capital_base_float=float(args_obj.capital_base),
        end_date_str=args_obj.end_date,
        output_dir_str=args_obj.output_dir,
        timestamp_str=args_obj.timestamp,
        matrix_name_str=args_obj.matrix,
    )
