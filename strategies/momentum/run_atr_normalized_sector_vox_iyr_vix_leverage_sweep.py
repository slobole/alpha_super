"""
Run the frozen VIX and modest-leverage exposure study for the sector ETF trend model.

Every row keeps the corrected candidate fixed:

    universe: XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY, VOX, IYR
    score: ROC12 / (ATR20 / Close_T)
    selection: top 5
    market gate: SPY Close_T > SMA200_T
    asset gate: none
    execution: month-end Close_T decision -> next tradable Open_T+1

Only total target exposure changes:

    static exposure = 1.00, 1.25, or 1.50

or:

    VIX target exposure_T = clip(20 / VIX_T, min_exposure, max_exposure)

Important: the engine reports negative cash but does not charge borrowing
interest. The primary results therefore keep the engine defaults unchanged,
while a separate report-only sensitivity deducts prior-day DTB3 + 1.5% from
the prior close's negative-cash weight.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from alpha.engine.backtest import run_daily
from alpha.engine.report import build_research_output_path
from strategies.momentum.run_atr_normalized_sector_vox_iyr_sweep import (
    _file_sha256_str,
    _markdown_table_str,
    _subperiod_metric_row_dict,
    _write_equity_curve_png,
)
from strategies.momentum.run_atr_normalized_vix_scaled_universe_comparison import (
    WEIGHTING_EQUAL_STR,
    _comparison_row_dict,
)
from strategies.momentum.strategy_mo_atr_normalized_sector_vox_iyr import (
    DEFAULT_CONFIG,
    DIMENSIONLESS_NATR_SCORE_STR,
    AtrNormalizedSectorConfig,
    build_strategy,
    compute_vix_scale_signal_df,
    get_atr_normalized_sector_data,
)


SUITE_ENTITY_ID_STR = "strategy_mo_atr_normalized_sector_vox_iyr"
ANALYSIS_TYPE_STR = "vix_leverage_sweep"
DTB3_CSV_PATH = REPO_ROOT_PATH.parent / "1_data" / "DTB3.csv"
BORROW_SPREAD_PCT_FLOAT = 1.5
TRADING_DAYS_PER_YEAR_FLOAT = 252.0


@dataclass(frozen=True)
class ExposureVariantSpec:
    variant_name_str: str
    use_vix_scale_bool: bool
    static_exposure_scale_float: float
    min_exposure_scale_float: float
    max_exposure_scale_float: float


# Frozen before the first result. This is the complete seven-row search space.
EXPOSURE_VARIANT_SPEC_TUPLE = (
    ExposureVariantSpec("static_1p00", False, 1.00, 1.00, 1.00),
    ExposureVariantSpec("static_1p25", False, 1.25, 1.25, 1.25),
    ExposureVariantSpec("static_1p50", False, 1.50, 1.50, 1.50),
    ExposureVariantSpec("vix_0p25_1p00", True, 1.00, 0.25, 1.00),
    ExposureVariantSpec("vix_0p50_1p00", True, 1.00, 0.50, 1.00),
    ExposureVariantSpec("vix_0p25_1p25", True, 1.00, 0.25, 1.25),
    ExposureVariantSpec("vix_0p25_1p50", True, 1.00, 0.25, 1.50),
)

DISPLAY_COLUMN_LIST = [
    "variant",
    "exposure_rule",
    "target_exposure_min",
    "target_exposure_max",
    "ann_return_pct",
    "ann_vol_pct",
    "sharpe",
    "max_drawdown_pct",
    "mar",
    "turnover_ann_pct",
    "cost_drag_ann_pct",
    "avg_gross_exposure_pct",
    "max_gross_exposure_pct",
    "avg_cash_weight_pct",
    "minimum_cash_weight_pct",
    "negative_cash_day_count",
    "transactions",
    "missing_liquidations",
]

FINANCING_DISPLAY_COLUMN_LIST = [
    "variant",
    "recomputed_engine_ann_return_pct",
    "financing_adjusted_ann_return_pct",
    "financing_drag_ann_pct",
    "recomputed_engine_sharpe",
    "financing_adjusted_sharpe",
    "financing_sharpe_delta",
    "financing_adjusted_max_drawdown_pct",
    "average_borrowed_weight_pct",
    "maximum_borrowed_weight_pct",
]


def _git_head_str() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT_PATH,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _leverage_diagnostic_dict(strategy_obj) -> dict[str, float | int]:
    realized_weight_df = strategy_obj.realized_weight_df.copy()
    asset_weight_df = realized_weight_df.drop(columns=["Cash"], errors="ignore")
    gross_exposure_ser = asset_weight_df.fillna(0.0).abs().sum(axis=1)
    accounting_policy_dict = dict(strategy_obj._accounting_policy_dict)
    return {
        "max_gross_exposure_pct": float(gross_exposure_ser.max() * 100.0),
        "minimum_cash_weight_pct": float(
            accounting_policy_dict["minimum_cash_weight_float"]
        )
        * 100.0,
        "average_negative_cash_weight_pct": float(
            accounting_policy_dict["average_negative_cash_weight_float"]
        )
        * 100.0,
        "negative_cash_day_count": int(
            accounting_policy_dict["negative_cash_day_count_int"]
        ),
        "negative_cash_episode_count": int(
            accounting_policy_dict["negative_cash_episode_count_int"]
        ),
    }


def _load_dtb3_annual_rate_ser() -> pd.Series:
    dtb3_df = pd.read_csv(DTB3_CSV_PATH, parse_dates=["observation_date"])
    dtb3_value_ser = pd.to_numeric(
        dtb3_df.set_index("observation_date")["DTB3"],
        errors="coerce",
    ).dropna()
    dtb3_annual_rate_ser = (
        dtb3_value_ser.astype(float) + BORROW_SPREAD_PCT_FLOAT
    ) / 100.0
    dtb3_annual_rate_ser.name = "dtb3_plus_spread_annual_rate"
    return dtb3_annual_rate_ser.sort_index()


def _financing_adjusted_equity_ser(
    total_value_ser: pd.Series,
    cash_ser: pd.Series,
    dtb3_annual_rate_ser: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    total_value_ser = total_value_ser.astype(float).sort_index()
    cash_ser = cash_ser.astype(float).reindex(total_value_ser.index)
    available_rate_ser = dtb3_annual_rate_ser.reindex(
        total_value_ser.index,
        method="ffill",
    )
    if available_rate_ser.isna().any():
        raise RuntimeError("DTB3 + spread is unavailable for part of the backtest.")

    borrowed_weight_ser = (
        -cash_ser / total_value_ser.replace(0.0, np.nan)
    ).clip(lower=0.0)
    daily_borrow_rate_ser = (
        (1.0 + available_rate_ser) ** (1.0 / TRADING_DAYS_PER_YEAR_FLOAT)
        - 1.0
    )
    # *** CRITICAL*** lookahead-sensitive financing boundary: cash and DTB3
    # observed at Close_T are charged against the return ending at Close_T+1.
    # Current-day cash or rates must not be charged retroactively to Close_T.
    financing_cost_return_ser = (
        borrowed_weight_ser.shift(1).fillna(0.0)
        * daily_borrow_rate_ser.shift(1).fillna(0.0)
    )
    engine_return_ser = total_value_ser.pct_change().fillna(0.0)
    financing_adjusted_return_ser = (
        engine_return_ser - financing_cost_return_ser
    )
    if financing_adjusted_return_ser.le(-1.0).any():
        raise RuntimeError("Financing-adjusted daily return is at or below -100%.")

    financing_adjusted_equity_ser = (
        (1.0 + financing_adjusted_return_ser).cumprod()
        * float(total_value_ser.iloc[0])
    )
    financing_adjusted_equity_ser.name = "financing_adjusted_total_value"
    return (
        financing_adjusted_equity_ser,
        financing_cost_return_ser,
        borrowed_weight_ser,
    )


def _config_for_variant(
    base_config_obj: AtrNormalizedSectorConfig,
    variant_spec_obj: ExposureVariantSpec,
) -> AtrNormalizedSectorConfig:
    return replace(
        base_config_obj,
        max_positions_int=5,
        apply_market_trend_bool=True,
        apply_asset_trend_bool=False,
        score_mode_str=DIMENSIONLESS_NATR_SCORE_STR,
        use_vix_scale_bool=variant_spec_obj.use_vix_scale_bool,
        static_exposure_scale_float=(
            variant_spec_obj.static_exposure_scale_float
        ),
        min_exposure_scale_float=variant_spec_obj.min_exposure_scale_float,
        max_exposure_scale_float=variant_spec_obj.max_exposure_scale_float,
    )


def run_sweep(
    backtest_start_date_str: str = DEFAULT_CONFIG.backtest_start_date_str,
    capital_base_float: float = DEFAULT_CONFIG.capital_base_float,
    end_date_str: str | None = None,
    output_dir_str: str = "results",
    timestamp_str: str | None = None,
) -> tuple[pd.DataFrame, Path]:
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
        base_vix_scale_signal_df,
    ) = get_atr_normalized_sector_data(
        config_obj=base_config_obj,
        include_total_return_benchmark_bool=True,
    )
    vix_close_ser = base_vix_scale_signal_df["vix_close_float"].astype(float)
    calendar_idx = pricing_data_df.index[
        pricing_data_df.index >= pd.Timestamp(backtest_start_date_str)
    ]
    actual_end_date_str = pd.Timestamp(calendar_idx[-1]).date().isoformat()

    output_path = build_research_output_path(
        output_dir=output_dir_str,
        entity_type_str="strategy",
        entity_id_str=SUITE_ENTITY_ID_STR,
        analysis_type_str=ANALYSIS_TYPE_STR,
        timestamp_str=timestamp_str,
    )
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing research artifact directory: {output_path}"
        )
    output_path.mkdir(parents=True, exist_ok=True)

    comparison_row_list: list[dict[str, object]] = []
    financing_row_list: list[dict[str, object]] = []
    subperiod_row_list: list[dict[str, object]] = []
    equity_curve_map: dict[str, pd.Series] = {}
    financing_equity_curve_map: dict[str, pd.Series] = {}
    dtb3_annual_rate_ser = _load_dtb3_annual_rate_ser()
    for variant_spec_obj in EXPOSURE_VARIANT_SPEC_TUPLE:
        config_obj = _config_for_variant(
            base_config_obj=base_config_obj,
            variant_spec_obj=variant_spec_obj,
        )
        vix_scale_signal_df = compute_vix_scale_signal_df(
            vix_close_ser=vix_close_ser,
            target_vix_pct_float=config_obj.target_vix_pct_float,
            min_exposure_scale_float=config_obj.min_exposure_scale_float,
            max_exposure_scale_float=config_obj.max_exposure_scale_float,
        )
        strategy_obj = build_strategy(
            config_obj=config_obj,
            rebalance_schedule_df=rebalance_schedule_df,
            vix_scale_signal_df=vix_scale_signal_df,
            name_str=(
                f"{SUITE_ENTITY_ID_STR}_{variant_spec_obj.variant_name_str}"
            ),
        )
        strategy_obj.universe_df = universe_df
        # *** CRITICAL*** Exposure uses VIX known at month-end Close_T and
        # target shares use Close_T. Orders fill only at Open_T+1.
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
            label_str=variant_spec_obj.variant_name_str,
            universe_str="XLB,XLE,XLF,XLI,XLK,XLP,XLU,XLV,XLY,VOX,IYR",
            volatility_helper_str=(
                config_obj.vix_symbol_str
                if variant_spec_obj.use_vix_scale_bool
                else "none"
            ),
            max_positions_int=config_obj.max_positions_int,
            weighting_scheme_str=WEIGHTING_EQUAL_STR,
            inverse_vol_window_int=None,
        )
        comparison_row_dict.update(
            {
                "exposure_rule": (
                    "clip(20/VIX,min,max)"
                    if variant_spec_obj.use_vix_scale_bool
                    else "static"
                ),
                "target_exposure_min": variant_spec_obj.min_exposure_scale_float,
                "target_exposure_max": variant_spec_obj.max_exposure_scale_float,
                **_leverage_diagnostic_dict(strategy_obj),
            }
        )
        comparison_row_list.append(comparison_row_dict)
        total_value_ser = strategy_obj.results["total_value"].astype(float)
        equity_curve_map[variant_spec_obj.variant_name_str] = total_value_ser
        (
            financing_adjusted_equity_ser,
            financing_cost_return_ser,
            borrowed_weight_ser,
        ) = _financing_adjusted_equity_ser(
            total_value_ser=total_value_ser,
            cash_ser=strategy_obj.results["cash"].astype(float),
            dtb3_annual_rate_ser=dtb3_annual_rate_ser,
        )
        financing_equity_curve_map[variant_spec_obj.variant_name_str] = (
            financing_adjusted_equity_ser
        )
        recomputed_engine_metric_dict = _subperiod_metric_row_dict(
            total_value_ser=total_value_ser,
            start_date_str=backtest_start_date_str,
            end_date_str=actual_end_date_str,
        )
        financing_metric_dict = _subperiod_metric_row_dict(
            total_value_ser=financing_adjusted_equity_ser,
            start_date_str=backtest_start_date_str,
            end_date_str=actual_end_date_str,
        )
        financing_row_list.append(
            {
                "variant": variant_spec_obj.variant_name_str,
                "recomputed_engine_ann_return_pct": (
                    recomputed_engine_metric_dict["cagr_pct"]
                ),
                "financing_adjusted_ann_return_pct": financing_metric_dict[
                    "cagr_pct"
                ],
                "financing_drag_ann_pct": (
                    float(recomputed_engine_metric_dict["cagr_pct"])
                    - float(financing_metric_dict["cagr_pct"])
                ),
                "recomputed_engine_sharpe": (
                    recomputed_engine_metric_dict["sharpe"]
                ),
                "financing_adjusted_sharpe": financing_metric_dict["sharpe"],
                "financing_sharpe_delta": (
                    float(financing_metric_dict["sharpe"])
                    - float(recomputed_engine_metric_dict["sharpe"])
                ),
                "financing_adjusted_max_drawdown_pct": financing_metric_dict[
                    "max_drawdown_pct"
                ],
                "average_borrowed_weight_pct": float(
                    borrowed_weight_ser.mean() * 100.0
                ),
                "maximum_borrowed_weight_pct": float(
                    borrowed_weight_ser.max() * 100.0
                ),
                "cumulative_financing_cost_return_pct": float(
                    financing_cost_return_ser.sum() * 100.0
                ),
            }
        )
        for period_start_date_str, period_end_date_str in (
            ("2006-01-01", "2012-12-31"),
            ("2013-01-01", "2019-12-31"),
            ("2020-01-01", actual_end_date_str),
        ):
            subperiod_row_dict = _subperiod_metric_row_dict(
                total_value_ser=strategy_obj.results["total_value"].astype(float),
                start_date_str=period_start_date_str,
                end_date_str=period_end_date_str,
            )
            subperiod_row_dict["variant"] = variant_spec_obj.variant_name_str
            subperiod_row_list.append(subperiod_row_dict)

        comparison_df = pd.DataFrame(comparison_row_list)
        comparison_df.to_csv(output_path / "comparison_table.csv", index=False)
        print(
            f"finished {variant_spec_obj.variant_name_str}: "
            f"{len(comparison_row_list)}/{len(EXPOSURE_VARIANT_SPEC_TUPLE)}"
        )

    display_df = comparison_df.loc[:, DISPLAY_COLUMN_LIST]
    (output_path / "comparison_table.md").write_text(
        _markdown_table_str(display_df) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(subperiod_row_list).to_csv(
        output_path / "subperiod_table.csv",
        index=False,
    )
    financing_df = pd.DataFrame(financing_row_list)
    financing_df.to_csv(
        output_path / "financing_sensitivity_table.csv",
        index=False,
    )
    (output_path / "financing_sensitivity_table.md").write_text(
        _markdown_table_str(
            financing_df.loc[:, FINANCING_DISPLAY_COLUMN_LIST]
        )
        + "\n",
        encoding="utf-8",
    )
    equity_curve_df = pd.DataFrame(equity_curve_map)
    equity_curve_df.to_csv(output_path / "equity_curve.csv", index_label="date")
    pd.DataFrame(financing_equity_curve_map).to_csv(
        output_path / "financing_adjusted_equity_curve.csv",
        index_label="date",
    )
    _write_equity_curve_png(
        equity_curve_df=equity_curve_df,
        output_path=output_path / "equity_curve.png",
    )

    sweep_path = Path(__file__).resolve()
    strategy_path = (
        REPO_ROOT_PATH
        / "strategies/momentum/strategy_mo_atr_normalized_sector_vox_iyr.py"
    )
    metadata_dict = {
        "variant_count": len(EXPOSURE_VARIANT_SPEC_TUPLE),
        "variant_spec": [
            {
                "variant": variant_spec_obj.variant_name_str,
                "use_vix_scale": variant_spec_obj.use_vix_scale_bool,
                "static_target_exposure": (
                    variant_spec_obj.static_exposure_scale_float
                ),
                "minimum_target_exposure": (
                    variant_spec_obj.min_exposure_scale_float
                ),
                "maximum_target_exposure": (
                    variant_spec_obj.max_exposure_scale_float
                ),
            }
            for variant_spec_obj in EXPOSURE_VARIANT_SPEC_TUPLE
        ],
        "fixed_candidate": {
            "score": "ROC12 / (ATR20 / Close_T)",
            "positions": 5,
            "market_filter": "SPY Close_T > SMA200_T",
            "asset_filter": "none",
            "execution": "month-end Close_T -> next tradable Open_T+1",
        },
        "costs": {
            "slippage": base_config_obj.slippage_float,
            "commission_per_share": base_config_obj.commission_per_share_float,
            "commission_minimum": base_config_obj.commission_minimum_float,
            "positive_cash_rate": "0%",
            "negative_cash_financing": "not modeled",
        },
        "financing_sensitivity": {
            "status": "report-only; primary engine results unchanged",
            "rate": "prior-day DTB3 + 1.5 percentage points",
            "rate_source": str(DTB3_CSV_PATH),
            "timing": (
                "Close_T negative-cash weight and DTB3 rate are charged "
                "against the return ending at Close_T+1"
            ),
        },
        "realism_warning": (
            "Target exposure is not a hard realized cap. Close_T sizing, "
            "Open_T+1 gaps, and subsequent drift can make realized gross "
            "exposure exceed the named target. Primary engine returns above "
            "1.0 remain optimistic because financing is only in the separate "
            "sensitivity table."
        ),
        "resolved_data": {
            "price_start": pd.Timestamp(pricing_data_df.index[0]).date().isoformat(),
            "backtest_start": pd.Timestamp(calendar_idx[0]).date().isoformat(),
            "backtest_end": actual_end_date_str,
            "norgatedata_package_version": importlib.metadata.version("norgatedata"),
        },
        "code_provenance": {
            "git_head": _git_head_str(),
            "strategy_sha256": _file_sha256_str(strategy_path),
            "sweep_sha256": _file_sha256_str(sweep_path),
        },
        "multiple_comparison_note": (
            "This seven-row exposure study is in-sample and follows prior "
            "selection of the fixed NATR candidate."
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
    return parser_obj.parse_args()


if __name__ == "__main__":
    args_obj = parse_args()
    run_sweep(
        backtest_start_date_str=args_obj.backtest_start_date,
        capital_base_float=float(args_obj.capital_base),
        end_date_str=args_obj.end_date,
        output_dir_str=args_obj.output_dir,
        timestamp_str=args_obj.timestamp,
    )
