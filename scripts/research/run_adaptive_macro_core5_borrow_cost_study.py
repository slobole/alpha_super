"""Run the frozen Adaptive Macro CORE5 DBC borrow-cost study.

Research only. This module does not alter the PM_READY strategy, BENCH hooks,
portfolio-manager files, releases, scheduler, broker, or LIVE wiring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient


REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from alpha.engine.backtest import run_daily
from strategies.taa_beyond_6040.strategy_taa_adaptive_macro_core5 import (
    DEFAULT_CONFIG,
    AdaptiveMacroCore5Config,
    AdaptiveMacroCore5Strategy,
    build_execution_calendar_idx,
    build_target_weight_ser,
    get_adaptive_macro_core5_data,
)


STUDY_ROOT_PATH = (
    REPO_ROOT_PATH
    / "results"
    / "research"
    / "strategy"
    / "strategy_taa_adaptive_macro_core5"
    / "borrow_cost_study"
    / "2026-08-25_core5_dbc_borrow_cost"
)
ANNUAL_BORROW_RATE_PCT_TUPLE = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)
DBC_EXTERNAL_SNAPSHOT_RATE_PCT_TUPLE = (0.43, 0.50, 0.66, 0.93)
SUBPERIOD_TUPLE = (
    ("early_2007_2012", "2007-09-04", "2012-12-31"),
    ("middle_2013_2019", "2013-01-01", "2019-12-31"),
    ("recent_2020_latest", "2020-01-01", None),
)


class BorrowCostAdaptiveMacroCore5Strategy(AdaptiveMacroCore5Strategy):
    """Research-only stateful DBC borrow-fee accounting variant."""

    def __init__(
        self,
        annual_borrow_rate_float: float,
        disable_dbc_short_bool: bool = False,
        name: str = "strategy_taa_adaptive_macro_core5_borrow_research",
        config_obj: AdaptiveMacroCore5Config = DEFAULT_CONFIG,
    ) -> None:
        if not np.isfinite(annual_borrow_rate_float) or annual_borrow_rate_float < 0.0:
            raise ValueError("annual_borrow_rate_float must be finite and non-negative.")
        zero_base_borrow_config_obj = replace(
            config_obj,
            annual_dbc_borrow_rate_float=0.0,
        )
        super().__init__(
            name=name,
            benchmarks=zero_base_borrow_config_obj.benchmark_list,
            config_obj=zero_base_borrow_config_obj,
        )
        self.annual_borrow_rate_float = float(annual_borrow_rate_float)
        self.disable_dbc_short_bool = bool(disable_dbc_short_bool)
        self.borrow_calendar_idx = pd.DatetimeIndex([])
        self.borrow_fee_row_dict_list: list[dict[str, object]] = []
        self._accounting_policy_dict.update(
            {
                "short_borrow_cost_policy_str": "research_stateful_constant_rate",
                "annual_borrow_rate_float": self.annual_borrow_rate_float,
                "borrow_day_count_denominator_int": 360,
                "borrow_collateral_multiplier_float": 1.02,
                "borrow_collateral_rounding_str": "ceil_per_share",
                "short_proceeds_interest_float": 0.0,
                "borrow_accrual_start_str": "trade_date_proxy_settlement_mismatch",
                "dbc_short_disabled_bool": self.disable_dbc_short_bool,
            }
        )

    def _target_weight_ser(
        self,
        close_row_ser: pd.Series,
        long_state_ser: pd.Series,
    ) -> pd.Series:
        if not self.disable_dbc_short_bool:
            return super()._target_weight_ser(close_row_ser, long_state_ser)

        return build_target_weight_ser(
            long_state_ser=long_state_ser,
            commodity_short_state_bool=False,
            commodity_annualized_volatility_float=1.0,
            config_obj=self.config_obj,
        )

    def _next_session_ts(self) -> pd.Timestamp | None:
        if len(self.borrow_calendar_idx) == 0:
            raise RuntimeError("borrow_calendar_idx must be assigned before the run.")
        current_position_int = int(self.borrow_calendar_idx.get_loc(self.current_bar))
        if current_position_int + 1 >= len(self.borrow_calendar_idx):
            return None
        return pd.Timestamp(self.borrow_calendar_idx[current_position_int + 1])

    def process_orders(self, prices: pd.DataFrame) -> None:
        super().process_orders(prices)

        commodity_asset_str = self.config_obj.commodity_asset_str
        held_share_float = float(self.get_position(commodity_asset_str))
        next_session_ts = self._next_session_ts()
        if held_share_float >= 0.0 or next_session_ts is None:
            return

        close_price_float = float(
            prices.loc[self.current_bar, (commodity_asset_str, "Close")]
        )
        if not np.isfinite(close_price_float) or close_price_float <= 0.0:
            raise RuntimeError("DBC borrow accounting requires a valid current close.")
        calendar_day_count_int = int(
            (next_session_ts.normalize() - pd.Timestamp(self.current_bar).normalize()).days
        )
        if calendar_day_count_int <= 0:
            raise RuntimeError("Borrow accrual requires a positive calendar-day interval.")

        # *** CRITICAL *** post-fill accounting boundary: the current open order
        # has already filled and the current close has already marked the held
        # position. This fee changes cash/NAV only; it cannot alter the prior
        # Close_T signal or the already executed Open_(T+1) order.
        collateral_price_float = float(np.ceil(1.02 * close_price_float))
        collateral_value_float = abs(held_share_float) * collateral_price_float
        borrow_fee_float = float(
            collateral_value_float
            * self.annual_borrow_rate_float
            * calendar_day_count_int
            / 360.0
        )
        self.cash -= borrow_fee_float
        self.total_value -= borrow_fee_float
        self.borrow_fee_row_dict_list.append(
            {
                "accrual_start_date_ts": pd.Timestamp(self.current_bar).normalize(),
                "next_session_date_ts": next_session_ts.normalize(),
                "calendar_day_count_int": calendar_day_count_int,
                "dbc_share_float": held_share_float,
                "dbc_close_float": close_price_float,
                "collateral_price_float": collateral_price_float,
                "collateral_value_float": collateral_value_float,
                "annual_borrow_rate_float": self.annual_borrow_rate_float,
                "borrow_fee_float": borrow_fee_float,
                "cash_after_fee_float": float(self.cash),
                "total_value_after_fee_float": float(self.total_value),
            }
        )
        self.borrow_fee_total_float += borrow_fee_float
        self._accounting_policy_dict.update(
            {
                "borrow_accrual_row_count_int": len(
                    self.borrow_fee_row_dict_list
                ),
                "borrow_fee_total_float": self.borrow_fee_total_float,
            }
        )

    def finalize(self, current_data_df: pd.DataFrame) -> None:
        super().finalize(current_data_df)
        self.borrow_fee_df = pd.DataFrame(self.borrow_fee_row_dict_list)


def sha256_file_str(file_path: Path) -> str:
    hash_obj = hashlib.sha256()
    with file_path.open("rb") as file_obj:
        for chunk_bytes in iter(lambda: file_obj.read(1024 * 1024), b""):
            hash_obj.update(chunk_bytes)
    return hash_obj.hexdigest()


def write_json(file_path: Path, value_obj: object) -> None:
    file_path.write_text(
        json.dumps(value_obj, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def markdown_table_str(table_df: pd.DataFrame) -> str:
    """Render a compact Markdown table without an optional dependency."""

    def cell_str(value_obj: object) -> str:
        if pd.isna(value_obj):
            return ""
        return str(value_obj).replace("|", "\\|").replace("\n", " ")

    header_str = "| " + " | ".join(map(str, table_df.columns)) + " |"
    separator_str = "| " + " | ".join("---" for _ in table_df.columns) + " |"
    row_str_list = [
        "| " + " | ".join(cell_str(value_obj) for value_obj in row_tuple) + " |"
        for row_tuple in table_df.itertuples(index=False, name=None)
    ]
    return "\n".join([header_str, separator_str, *row_str_list])


def run_research_strategy(
    pricing_data_df: pd.DataFrame,
    calendar_idx: pd.DatetimeIndex,
    config_obj: AdaptiveMacroCore5Config,
    annual_borrow_rate_float: float,
    disable_dbc_short_bool: bool,
    show_progress_bool: bool,
) -> BorrowCostAdaptiveMacroCore5Strategy:
    variant_label_str = (
        "no_short"
        if disable_dbc_short_bool
        else f"borrow_{annual_borrow_rate_float * 100.0:g}pct"
    )
    strategy_obj = BorrowCostAdaptiveMacroCore5Strategy(
        annual_borrow_rate_float=annual_borrow_rate_float,
        disable_dbc_short_bool=disable_dbc_short_bool,
        name=f"strategy_taa_adaptive_macro_core5_{variant_label_str}",
        config_obj=config_obj,
    )
    strategy_obj.borrow_calendar_idx = pd.DatetimeIndex(calendar_idx)
    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar=calendar_idx,
        show_progress=show_progress_bool,
        show_signal_progress_bool=False,
        audit_override_bool=False,
    )
    return strategy_obj


def run_literal_baseline(
    pricing_data_df: pd.DataFrame,
    calendar_idx: pd.DatetimeIndex,
    config_obj: AdaptiveMacroCore5Config,
    show_progress_bool: bool,
) -> AdaptiveMacroCore5Strategy:
    zero_borrow_config_obj = replace(
        config_obj,
        annual_dbc_borrow_rate_float=0.0,
    )
    strategy_obj = AdaptiveMacroCore5Strategy(
        name=zero_borrow_config_obj.strategy_name_str,
        benchmarks=zero_borrow_config_obj.benchmark_list,
        config_obj=zero_borrow_config_obj,
    )
    strategy_obj.borrow_calendar_idx = pd.DatetimeIndex(calendar_idx)
    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar=calendar_idx,
        show_progress=show_progress_bool,
        show_signal_progress_bool=False,
        audit_override_bool=False,
    )
    return strategy_obj


def performance_metric_dict(
    total_value_ser: pd.Series,
    benchmark_value_ser: pd.Series | None = None,
) -> dict[str, float | int | str | None]:
    clean_value_ser = pd.to_numeric(total_value_ser, errors="coerce").dropna().astype(float)
    if len(clean_value_ser) < 2:
        raise ValueError("At least two portfolio values are required.")
    # *** CRITICAL *** report-only backward return: pct_change uses only the
    # immediately prior realized value and never feeds strategy decisions.
    daily_return_ser = clean_value_ser.pct_change(fill_method=None).dropna()
    elapsed_year_float = max(
        (clean_value_ser.index[-1] - clean_value_ser.index[0]).days / 365.25,
        1.0 / 365.25,
    )
    cagr_float = float(
        (float(clean_value_ser.iloc[-1]) / float(clean_value_ser.iloc[0]))
        ** (1.0 / elapsed_year_float)
        - 1.0
    )
    annualized_volatility_float = float(daily_return_ser.std(ddof=1) * np.sqrt(252.0))
    sharpe_float = (
        float(daily_return_ser.mean() / daily_return_ser.std(ddof=1) * np.sqrt(252.0))
        if daily_return_ser.std(ddof=1) > 0.0
        else np.nan
    )
    drawdown_ser = clean_value_ser / clean_value_ser.cummax() - 1.0
    max_drawdown_float = float(drawdown_ser.min())
    result_dict: dict[str, float | int | str | None] = {
        "start_date_str": str(clean_value_ser.index[0].date()),
        "end_date_str": str(clean_value_ser.index[-1].date()),
        "observation_count_int": int(len(clean_value_ser)),
        "cagr_float": cagr_float,
        "annualized_volatility_float": annualized_volatility_float,
        "sharpe_float": sharpe_float,
        "max_drawdown_float": max_drawdown_float,
        "mar_float": cagr_float / abs(max_drawdown_float) if max_drawdown_float < 0.0 else np.nan,
        "final_equity_float": float(clean_value_ser.iloc[-1]),
    }
    if benchmark_value_ser is None:
        result_dict.update(
            {
                "daily_market_correlation_float": None,
                "monthly_market_correlation_float": None,
                "market_beta_float": None,
            }
        )
        return result_dict

    benchmark_return_ser = (
        pd.to_numeric(benchmark_value_ser, errors="coerce")
        .pct_change(fill_method=None)
        .rename("benchmark_return_float")
    )
    paired_return_df = pd.concat(
        [daily_return_ser.rename("strategy_return_float"), benchmark_return_ser],
        axis=1,
        join="inner",
    ).dropna()
    benchmark_variance_float = float(paired_return_df["benchmark_return_float"].var(ddof=1))
    result_dict["daily_market_correlation_float"] = float(
        paired_return_df.corr().iloc[0, 1]
    )
    result_dict["market_beta_float"] = float(
        paired_return_df.cov().iloc[0, 1] / benchmark_variance_float
    )
    monthly_return_df = paired_return_df.add(1.0).resample("ME").prod().sub(1.0)
    result_dict["monthly_market_correlation_float"] = float(
        monthly_return_df.corr().iloc[0, 1]
    )
    return result_dict


def variant_row_dict(
    variant_key_str: str,
    strategy_obj: BorrowCostAdaptiveMacroCore5Strategy,
    annual_borrow_rate_pct_float: float | None,
    disable_dbc_short_bool: bool,
) -> dict[str, object]:
    result_df = strategy_obj.results.copy()
    metric_dict = performance_metric_dict(
        result_df["total_value"],
        result_df["$SPX"],
    )
    borrow_fee_df = strategy_obj.borrow_fee_df
    realized_weight_df = strategy_obj.realized_weight_df
    dbc_weight_ser = pd.to_numeric(
        realized_weight_df.get("DBC", pd.Series(0.0, index=realized_weight_df.index)),
        errors="coerce",
    ).fillna(0.0)
    short_weight_ser = dbc_weight_ser.clip(upper=0.0).abs()
    return {
        "variant_key_str": variant_key_str,
        "annual_borrow_rate_pct_float": annual_borrow_rate_pct_float,
        "disable_dbc_short_bool": disable_dbc_short_bool,
        **metric_dict,
        "total_borrow_fee_float": (
            float(borrow_fee_df["borrow_fee_float"].sum())
            if len(borrow_fee_df) > 0
            else 0.0
        ),
        "borrow_accrual_row_count_int": int(len(borrow_fee_df)),
        "dbc_short_day_count_int": int(short_weight_ser.gt(0.0).sum()),
        "dbc_short_day_fraction_float": float(short_weight_ser.gt(0.0).mean()),
        "average_dbc_short_weight_float": float(short_weight_ser.mean()),
        "average_dbc_short_weight_when_active_float": (
            float(short_weight_ser.loc[short_weight_ser.gt(0.0)].mean())
            if short_weight_ser.gt(0.0).any()
            else 0.0
        ),
        "maximum_dbc_short_weight_float": float(short_weight_ser.max()),
        "transaction_count_int": int(len(strategy_obj.get_transactions())),
    }


def subperiod_row_list(
    variant_key_str: str,
    strategy_obj: BorrowCostAdaptiveMacroCore5Strategy,
) -> list[dict[str, object]]:
    row_list: list[dict[str, object]] = []
    for period_key_str, start_date_str, end_date_str in SUBPERIOD_TUPLE:
        end_date_obj = pd.Timestamp(end_date_str) if end_date_str is not None else None
        period_result_df = strategy_obj.results.loc[pd.Timestamp(start_date_str) : end_date_obj]
        if len(period_result_df) < 2:
            continue
        row_list.append(
            {
                "variant_key_str": variant_key_str,
                "period_key_str": period_key_str,
                **performance_metric_dict(
                    period_result_df["total_value"],
                    period_result_df["$SPX"],
                ),
            }
        )
    return row_list


def build_gate_df(comparison_df: pd.DataFrame) -> pd.DataFrame:
    row_by_variant_dict = comparison_df.set_index("variant_key_str").to_dict("index")
    zero_row_dict = row_by_variant_dict["borrow_0pct"]
    one_row_dict = row_by_variant_dict["borrow_1pct"]
    five_row_dict = row_by_variant_dict["borrow_5pct"]
    no_short_row_dict = row_by_variant_dict["no_short"]
    candidate_borrow_rate_pct_float = 1.0
    maximum_external_snapshot_pct_float = max(
        DBC_EXTERNAL_SNAPSHOT_RATE_PCT_TUPLE
    )

    gate_row_list = [
        {
            "gate_key_str": "external_plausibility",
            "rule_str": "1% >= maximum observed DBC snapshot used (0.93%)",
            "observed_value_float": candidate_borrow_rate_pct_float,
            "threshold_float": maximum_external_snapshot_pct_float,
            "pass_bool": (
                candidate_borrow_rate_pct_float
                >= maximum_external_snapshot_pct_float
            ),
        },
        {
            "gate_key_str": "one_percent_cagr_drag",
            "rule_str": "CAGR loss versus zero rate <= 0.10 percentage point",
            "observed_value_float": float(zero_row_dict["cagr_float"] - one_row_dict["cagr_float"]),
            "threshold_float": 0.001,
            "pass_bool": float(zero_row_dict["cagr_float"] - one_row_dict["cagr_float"]) <= 0.001,
        },
        {
            "gate_key_str": "one_percent_sharpe_erosion",
            "rule_str": "Sharpe erosion versus zero rate <= 2%",
            "observed_value_float": float(1.0 - one_row_dict["sharpe_float"] / zero_row_dict["sharpe_float"]),
            "threshold_float": 0.02,
            "pass_bool": float(one_row_dict["sharpe_float"] / zero_row_dict["sharpe_float"]) >= 0.98,
        },
        {
            "gate_key_str": "overlay_cagr",
            "rule_str": "1% CAGR >= no-short CAGR minus 0.10 percentage point",
            "observed_value_float": float(one_row_dict["cagr_float"] - no_short_row_dict["cagr_float"]),
            "threshold_float": -0.001,
            "pass_bool": float(one_row_dict["cagr_float"] - no_short_row_dict["cagr_float"]) >= -0.001,
        },
        {
            "gate_key_str": "overlay_sharpe",
            "rule_str": "1% Sharpe >= no-short Sharpe",
            "observed_value_float": float(one_row_dict["sharpe_float"] - no_short_row_dict["sharpe_float"]),
            "threshold_float": 0.0,
            "pass_bool": float(one_row_dict["sharpe_float"] - no_short_row_dict["sharpe_float"]) >= 0.0,
        },
        {
            "gate_key_str": "overlay_max_drawdown",
            "rule_str": "1% maximum drawdown no more than 0.50 percentage point worse than no-short",
            "observed_value_float": float(one_row_dict["max_drawdown_float"] - no_short_row_dict["max_drawdown_float"]),
            "threshold_float": -0.005,
            "pass_bool": float(one_row_dict["max_drawdown_float"] - no_short_row_dict["max_drawdown_float"]) >= -0.005,
        },
        {
            "gate_key_str": "five_percent_positive_cagr",
            "rule_str": "5% CAGR > 0",
            "observed_value_float": float(five_row_dict["cagr_float"]),
            "threshold_float": 0.0,
            "pass_bool": float(five_row_dict["cagr_float"]) > 0.0,
        },
        {
            "gate_key_str": "five_percent_sharpe_retention",
            "rule_str": "5% Sharpe >= 90% of zero-rate Sharpe",
            "observed_value_float": float(five_row_dict["sharpe_float"] / zero_row_dict["sharpe_float"]),
            "threshold_float": 0.90,
            "pass_bool": float(five_row_dict["sharpe_float"] / zero_row_dict["sharpe_float"]) >= 0.90,
        },
    ]
    return pd.DataFrame(gate_row_list)


def create_charts(
    strategy_by_variant_dict: dict[str, BorrowCostAdaptiveMacroCore5Strategy],
    comparison_df: pd.DataFrame,
    chart_output_path: Path,
) -> None:
    chart_output_path.mkdir(parents=True, exist_ok=True)
    color_by_variant_dict = {
        "borrow_0pct": "#1f77b4",
        "borrow_1pct": "#2ca02c",
        "borrow_5pct": "#ff7f0e",
        "borrow_10pct": "#d62728",
        "no_short": "#7f7f7f",
    }
    selected_variant_key_list = list(color_by_variant_dict)

    figure_obj, axis_obj = plt.subplots(figsize=(11, 6))
    for variant_key_str in selected_variant_key_list:
        total_value_ser = strategy_by_variant_dict[variant_key_str].results["total_value"].astype(float)
        normalized_value_ser = total_value_ser / float(total_value_ser.iloc[0])
        axis_obj.plot(
            normalized_value_ser.index,
            normalized_value_ser,
            label=variant_key_str,
            color=color_by_variant_dict[variant_key_str],
            linewidth=1.5,
        )
    benchmark_value_ser = strategy_by_variant_dict["borrow_1pct"].results["$SPX"].astype(float)
    axis_obj.plot(
        benchmark_value_ser.index,
        benchmark_value_ser / float(benchmark_value_ser.iloc[0]),
        label="$SPX total return",
        color="#9467bd",
        linewidth=1.2,
        alpha=0.8,
    )
    axis_obj.set_title("Adaptive Macro CORE5 equity by DBC borrow assumption | 2007-2026 | net of trading costs")
    axis_obj.set_ylabel("Growth of $1")
    axis_obj.grid(alpha=0.25)
    axis_obj.legend(ncol=2)
    figure_obj.tight_layout()
    figure_obj.savefig(chart_output_path / "01_equity_by_borrow_rate.png", dpi=160)
    plt.close(figure_obj)

    figure_obj, axis_obj = plt.subplots(figsize=(11, 5))
    for variant_key_str in ("borrow_0pct", "borrow_1pct", "no_short"):
        total_value_ser = strategy_by_variant_dict[variant_key_str].results["total_value"].astype(float)
        drawdown_ser = total_value_ser / total_value_ser.cummax() - 1.0
        axis_obj.plot(
            drawdown_ser.index,
            drawdown_ser * 100.0,
            label=variant_key_str,
            color=color_by_variant_dict[variant_key_str],
        )
    axis_obj.set_title("Adaptive Macro CORE5 drawdown | 0%, 1%, and no DBC short | 2007-2026")
    axis_obj.set_ylabel("Drawdown (%)")
    axis_obj.grid(alpha=0.25)
    axis_obj.legend()
    figure_obj.tight_layout()
    figure_obj.savefig(chart_output_path / "02_drawdown_comparison.png", dpi=160)
    plt.close(figure_obj)

    rate_df = comparison_df.loc[~comparison_df["disable_dbc_short_bool"]].sort_values(
        "annual_borrow_rate_pct_float"
    )
    figure_obj, left_axis_obj = plt.subplots(figsize=(9, 5))
    right_axis_obj = left_axis_obj.twinx()
    left_axis_obj.plot(
        rate_df["annual_borrow_rate_pct_float"],
        rate_df["cagr_float"] * 100.0,
        marker="o",
        color="#1f77b4",
        label="CAGR",
    )
    right_axis_obj.plot(
        rate_df["annual_borrow_rate_pct_float"],
        rate_df["sharpe_float"],
        marker="s",
        color="#d62728",
        label="Sharpe",
    )
    left_axis_obj.axvline(1.0, color="#2ca02c", linestyle="--", alpha=0.8)
    left_axis_obj.set_title("DBC constant borrow-rate sensitivity | full sample | stateful reruns")
    left_axis_obj.set_xlabel("Annual borrow rate (%)")
    left_axis_obj.set_ylabel("CAGR (%)", color="#1f77b4")
    right_axis_obj.set_ylabel("Sharpe", color="#d62728")
    left_axis_obj.grid(alpha=0.25)
    figure_obj.tight_layout()
    figure_obj.savefig(chart_output_path / "03_borrow_rate_sensitivity.png", dpi=160)
    plt.close(figure_obj)

    one_result_df = strategy_by_variant_dict["borrow_1pct"].results
    paired_return_df = pd.concat(
        [
            one_result_df["total_value"].astype(float).pct_change(fill_method=None).rename("strategy"),
            one_result_df["$SPX"].astype(float).pct_change(fill_method=None).rename("market"),
        ],
        axis=1,
    ).dropna()
    # *** CRITICAL *** report-only backward rolling window: every correlation
    # point uses the 126 realized return pairs ending on that date.
    rolling_correlation_ser = paired_return_df["strategy"].rolling(126).corr(
        paired_return_df["market"]
    )
    figure_obj, axis_obj = plt.subplots(figsize=(11, 4.5))
    axis_obj.plot(rolling_correlation_ser.index, rolling_correlation_ser, color="#2ca02c")
    axis_obj.axhline(0.0, color="black", linewidth=0.7)
    axis_obj.set_title("Adaptive Macro CORE5 at 1% borrow | rolling 126-session correlation with $SPX")
    axis_obj.set_ylabel("Correlation")
    axis_obj.grid(alpha=0.25)
    figure_obj.tight_layout()
    figure_obj.savefig(chart_output_path / "04_rolling_market_correlation.png", dpi=160)
    plt.close(figure_obj)


def create_notebook(study_root_path: Path) -> Path:
    notebook_path = study_root_path / "executed_notebook.ipynb"
    notebook_obj = nbformat.v4.new_notebook()
    notebook_obj.cells = [
        nbformat.v4.new_markdown_cell(
            "# Adaptive Macro CORE5 DBC Borrow Cost Validation\n\n"
            "Decision notebook. It loads immutable saved results; it does not rerun the backtest."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\n"
            "study_path = Path.cwd()\n"
            "comparison_df = pd.read_csv(study_path / 'tables' / 'borrow_rate_sweep.csv')\n"
            "gate_df = pd.read_csv(study_path / 'tables' / 'promotion_gates.csv')\n"
            "display(comparison_df[['variant_key_str','cagr_float','sharpe_float','max_drawdown_float','total_borrow_fee_float']])\n"
            "display(gate_df)"
        ),
        nbformat.v4.new_code_cell(
            "display(Image(filename=str(study_path / 'charts' / '01_equity_by_borrow_rate.png')))"
        ),
        nbformat.v4.new_code_cell(
            "display(Image(filename=str(study_path / 'charts' / '03_borrow_rate_sensitivity.png')))"
        ),
    ]
    nbformat.write(notebook_obj, notebook_path)
    NotebookClient(
        notebook_obj,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(study_root_path)}},
    ).execute()
    nbformat.write(notebook_obj, notebook_path)
    return notebook_path


def write_reports(
    comparison_df: pd.DataFrame,
    subperiod_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    parity_dict: dict[str, object],
    study_root_path: Path,
) -> None:
    row_by_variant_dict = comparison_df.set_index("variant_key_str").to_dict("index")
    zero_row_dict = row_by_variant_dict["borrow_0pct"]
    one_row_dict = row_by_variant_dict["borrow_1pct"]
    five_row_dict = row_by_variant_dict["borrow_5pct"]
    no_short_row_dict = row_by_variant_dict["no_short"]
    all_gate_pass_bool = bool(gate_df["pass_bool"].all())
    verdict_str = (
        "1% מתאים כ־baseline שמרני למחקר, אך עדיין לא כתחליף לנתוני borrow יומיים של החשבון."
        if all_gate_pass_bool
        else "1% אינו עובר את שערי המחקר הקפואים ולכן אין להוסיף אותו כברירת מחדל."
    )

    compact_table_df = comparison_df.loc[
        comparison_df["variant_key_str"].isin(
            ["borrow_0pct", "borrow_0.5pct", "borrow_1pct", "borrow_3pct", "borrow_5pct", "borrow_10pct", "no_short"]
        ),
        [
            "variant_key_str",
            "cagr_float",
            "annualized_volatility_float",
            "sharpe_float",
            "max_drawdown_float",
            "final_equity_float",
            "total_borrow_fee_float",
        ],
    ].copy()
    for column_str in ("cagr_float", "annualized_volatility_float", "max_drawdown_float"):
        compact_table_df[column_str] = compact_table_df[column_str].map(lambda value_float: f"{value_float * 100.0:.3f}%")
    compact_table_df["sharpe_float"] = compact_table_df["sharpe_float"].map(lambda value_float: f"{value_float:.3f}")
    compact_table_df["final_equity_float"] = compact_table_df["final_equity_float"].map(lambda value_float: f"${value_float:,.0f}")
    compact_table_df["total_borrow_fee_float"] = compact_table_df["total_borrow_fee_float"].map(lambda value_float: f"${value_float:,.0f}")
    compact_table_md_str = markdown_table_str(compact_table_df)

    report_str = rf"""# בדיקת עלות Short Borrow ל־Adaptive Macro CORE5

> **מחקר בלבד — לא PAPER, לא LIVE ולא אישור הקצאה.**

## TL;DR

האסטרטגיה נבדקה מחדש במסלול stateful מלא עם עלות borrow שנתית קבועה ל־DBC, נוסף על slippage ועמלות שכבר קיימים במנוע. התקופה היא {one_row_dict['start_date_str']} עד {one_row_dict['end_date_str']} על נתוני Norgate. {verdict_str}

ב־1% שנתי ה־CAGR הוא {one_row_dict['cagr_float'] * 100.0:.3f}%, ה־Sharpe הוא {one_row_dict['sharpe_float']:.3f}, וה־Max DD הוא {one_row_dict['max_drawdown_float'] * 100.0:.3f}%. עלות ה־borrow המצטברת בפועל היא ${one_row_dict['total_borrow_fee_float']:,.0f} על הון התחלתי של $100,000. כל שמונת שערי הקבלה הקפואים {'עברו' if all_gate_pass_bool else 'לא עברו'}.

## מה נבדק

- אותה לוגיקת Adaptive Macro CORE5 ללא שינוי באותות, נכסים, משקולות או תזמון.
- שמונה שיעורים שנתיים קבועים: 0%, 0.25%, 0.5%, 1%, 2%, 3%, 5%, 10%.
- counterfactual אחד ללא שכבת short על DBC.
- 1% נקבע מראש כתרחיש המרכזי; 5% נקבע מראש כמבחן הישרדות.

## הנוסחה ותזמון החיוב

עלות כל תקופת החזקה בין שני ימי מסחר היא:

$$
fee_t = |q_t| \times \lceil 1.02 P_t \rceil \times b \times \frac{{d_t}}{{360}}
$$

גודל שכבת השורט עצמה נשאר בדיוק לפי האסטרטגיה המקורית:

$$
w_{{short,T}} = -\min\left(0.10, \frac{{0.025}}{{\sigma_{{63,T}}}}\right)
$$

כאשר `q_t` הוא מספר מניות DBC השלילי לאחר ביצוע פקודות ב־Open, `P_t` הוא מחיר ה־Close באותו יום, `b` הוא שיעור ה־borrow השנתי ו־`d_t` הוא מספר הימים הקלנדריים עד יום המסחר הבא. כך סופי שבוע וחגים מחויבים. אין ריבית לזכות על תקבולי השורט.

```text
Close_T signal -> Open_(T+1) fill -> Close_(T+1) mark -> borrow accrual
                                             |              to next session
                                             +-- fee affects future NAV only
```

## תוצאות מרכזיות

{compact_table_md_str}

ביחס ל־0%, עלות 1% הפחיתה CAGR ב־{(zero_row_dict['cagr_float'] - one_row_dict['cagr_float']) * 100.0:.3f} נקודת אחוז ושינתה Sharpe ב־{one_row_dict['sharpe_float'] - zero_row_dict['sharpe_float']:+.3f}. ביחס לביטול השורט, שכבת DBC לאחר עלות 1% שינתה CAGR ב־{(one_row_dict['cagr_float'] - no_short_row_dict['cagr_float']) * 100.0:+.3f} נקודת אחוז, Sharpe ב־{one_row_dict['sharpe_float'] - no_short_row_dict['sharpe_float']:+.3f}, ו־Max DD ב־{(one_row_dict['max_drawdown_float'] - no_short_row_dict['max_drawdown_float']) * 100.0:+.3f} נקודת אחוז.

השורט היה פעיל ב־{one_row_dict['dbc_short_day_fraction_float'] * 100.0:.1f}% מימי המסחר, במשקל ממוצע של {one_row_dict['average_dbc_short_weight_when_active_float'] * 100.0:.2f}% כאשר היה פעיל. לכן 1% עלות על הנייר אינו 1% על כל התיק: ה־drag השנתי שנמדד על CAGR היה כ־{(zero_row_dict['cagr_float'] - one_row_dict['cagr_float']) * 10000.0:.2f} bps בלבד.

![Equity](charts/01_equity_by_borrow_rate.png)

![Borrow sensitivity](charts/03_borrow_rate_sensitivity.png)

## תלות בשוק

בתרחיש 1% הקורלציה היומית ל־$SPX היא {one_row_dict['daily_market_correlation_float']:.3f}, הקורלציה החודשית היא {one_row_dict['monthly_market_correlation_float']:.3f}, וה־beta היומי הוא {one_row_dict['market_beta_float']:.3f}. אלה מאפייני כל תיק CORE5, לא של שכבת DBC לבדה.

![Rolling correlation](charts/04_rolling_market_correlation.png)

## אימות חיצוני של 1%

IBKR מתאר עלות שורט כשילוב של borrow fee וריבית על תקבולי השורט, עם שיעורים שמשתנים לפי היצע וביקוש. נוסחת הדוח שלו משתמשת ב־`Value × Fee Rate / 360`, ובקונבנציית collateral של 102% ממחיר הסגירה הקודם המעוגל כלפי מעלה למניה. נקודות DBC ציבוריות שנמצאו היו 0.43%–0.66% בתחילת 2026, 0.50% במרץ 2026, ו־0.93% בתצלום מחקרי מ־2015. לכן 1% הוא baseline שמרני ביחס לנקודות אלה — אך אינו ממוצע היסטורי מאומת.

## מגבלות ופסיקה

- לא הושגה סדרת borrow יומית מלאה, היסטורית וספציפית לחשבון עבור DBC.
- שיעור קבוע אינו מדמה hard-to-borrow spikes, locate failure, recall או forced buy-in.
- החיוב ממופה מיום העסקה, בעוד IBKR מתאר חיוב על settled positions. זהו timing proxy שאינו מובטח כשמרני לכל מסלול מחיר.
- manufactured dividends על שורט DBC כבר מחויבים במלוא ה־gross בלג'ר הדיבידנדים של המנוע; הפער שנותר הוא ex-date כלכלי מול pay-date ו־statement בפועל של הברוקר.
- parity ב־0% {'עבר בדיוק' if parity_dict['exact_parity_bool'] else 'נכשל'} מול האסטרטגיה המקורית.

**המלצה:** {verdict_str} לפני PAPER או LIVE יש לחבר סדרת rate/availability יומית מהברוקר, להוסיף locate/recall policy ולבצע reconciliation מול statement אמיתי.
"""
    (study_root_path / "REPORT.md").write_text(report_str, encoding="utf-8")

    full_report_str = report_str + f"""

## נספח: כללי הקבלה הקפואים

{markdown_table_str(gate_df)}

## נספח: תתי־תקופות

{markdown_table_str(subperiod_df)}

## נספח: lineage ובקרות כמותיות

- מספר וריאנטים כלכליים שנבדקו: 9. ריצת האסטרטגיה המקורית שימשה oracle ל־parity בלבד.
- אין בחירת פרמטר מתוך התוצאות; כל שיעורי העלות וכל השערים הוקפאו מראש.
- signals משתמשים ב־TOTALRETURN Close_T; פקודות מתבצעות ב־CAPITALSPECIAL Open_(T+1).
- אין universe של מניות ולכן survivorship selection אינו רלוונטי; היסטוריית ETF לפני inception נשארת חסרה עד שכל האותות זמינים.
- עלויות המסחר הקיימות נשמרו: 2.5 bps לכל צד, $0.005 למניה ומינימום $1.
- cash על תקבולי השורט מוגבל ואינו נושא ריבית.

## מפת ארטיפקטים

- `tables/borrow_rate_sweep.csv`: טבלת הווריאנטים המלאה.
- `tables/subperiod_metrics.csv`: תתי־תקופות שנקבעו מראש.
- `tables/promotion_gates.csv`: תוצאת כל שער קבלה.
- `tables/borrow_fee_ledger_1pct.csv`: כל חיוב יומי בתרחיש 1%.
- `tables/equity_curves.csv.gz`: מסלולי equity ו־benchmark.
- `executed_notebook.ipynb`: notebook החלטה שבוצע ללא הרצת backtest מחדש.
- `SOURCE_RULE_MAP.md` ו־`research_spec_frozen.json`: חוזה המקור והמחקר לפני תוצאות.
- `source_evidence.json`: מקורות חיצוניים והיקף הראיה.
"""
    (study_root_path / "REPORT_FULL.md").write_text(full_report_str, encoding="utf-8")


def write_knowledge_record(
    comparison_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    study_root_path: Path,
) -> None:
    one_row_dict = comparison_df.set_index("variant_key_str").loc["borrow_1pct"].to_dict()
    all_gate_pass_bool = bool(gate_df["pass_bool"].all())
    knowledge_record_dict = {
        "schema_version": "quant-research-knowledge-v1",
        "study_id": "adaptive_macro_core5_dbc_borrow_cost",
        "title": "Adaptive Macro CORE5 DBC Borrow Cost Validation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_status": "diagnostic",
        "disposition": "promising_component" if all_gate_pass_bool else "rejected",
        "replication_outcome": "replicated",
        "signal_family": "adaptive_macro_multi_asset",
        "objective": "Validate a 1% annual DBC borrow-cost baseline and the net value of the DBC short overlay.",
        "verdict": (
            "1% is a defensible conservative research planning baseline, not a historical or live broker truth."
            if all_gate_pass_bool
            else "The frozen 1% planning baseline failed at least one economic gate."
        ),
        "verdicts": {
            "source_replication": "Zero-rate research accounting exactly matches the unchanged strategy.",
            "predictive_value": "Not assessed; this is an implementation-cost study.",
            "economic_value": "Assessed through the 1% overlay and 5% survival gates.",
            "promotion": "Research accounting recommendation only; no PAPER or LIVE authority.",
        },
        "universes": ["SPY, IEF, GLD, DBC, UUP, BIL"],
        "decision_timing": "Close_T",
        "fill_timing": "Open_T+1",
        "timing_attribution": {
            "status": "not_applicable",
            "diagnostic_path": "not_applicable",
            "executable_path": "Close_T decision to Open_T+1 fill",
            "method": "stateful post-fill fee deduction",
            "headline_result": "Borrow fees affect current close NAV and future sizing only.",
            "metrics": {},
            "artifact": "SOURCE_RULE_MAP.md",
        },
        "primary_cost_layer": "central_research",
        "primary_metrics": {
            "period": f"{one_row_dict['start_date_str']} to {one_row_dict['end_date_str']}",
            "universe": "fixed six-ETF execution universe",
            "cost_layer": "1% DBC borrow",
            "CAGR": float(one_row_dict["cagr_float"]),
            "annualized_volatility": float(one_row_dict["annualized_volatility_float"]),
            "Sharpe": float(one_row_dict["sharpe_float"]),
            "maximum_drawdown": float(one_row_dict["max_drawdown_float"]),
            "turnover": None,
        },
        "feature_findings": [],
        "cost_capacity": {
            "paper_like_round_trip_bps": 5.0,
            "central_research_round_trip_bps": 5.0,
            "conservative_survival_round_trip_bps": 5.0,
            "capacity_impact_separate": True,
            "comfortable_capacity": None,
            "soft_capacity": None,
            "strained_capacity": None,
            "hard_capacity": None,
            "unresolved_reason": "Borrow availability and rate history are not capacity calibration; cost tiers differ by annual borrow rate, not round-trip bps.",
        },
        "limitations": [
            "No complete historical account-specific DBC rate series.",
            "No locate, recall, or forced buy-in model; manufactured dividends are booked at economic ex-date rather than exact broker pay-date.",
            "Trade-date accrual is a timing proxy, not exact settled-position billing and not uniformly conservative.",
        ],
        "next_tests": [
            "Ingest daily account-specific DBC rate and availability.",
            "Reconcile modeled fees to a real broker statement before any deployment validation.",
        ],
        "sources": ["source_evidence.json"],
        "artifacts": {
            "concise_report": "REPORT.md",
            "full_report": "REPORT_FULL.md",
            "notebook": "executed_notebook.ipynb",
            "frozen_specification": "research_spec_frozen.json",
            "manifest": "run_manifest.json",
            "primary_source_code": ["scripts/research/run_adaptive_macro_core5_borrow_cost_study.py"],
            "primary_tables": ["tables/borrow_rate_sweep.csv", "tables/promotion_gates.csv"],
            "primary_charts": ["charts/01_equity_by_borrow_rate.png", "charts/03_borrow_rate_sensitivity.png"],
            "research_state": "research_state.json",
            "hypothesis_registry": "hypothesis_registry.json",
            "experiment_ledger": "experiment_ledger.jsonl",
            "decision_log": "decision_log.jsonl",
            "source_rule_map": "SOURCE_RULE_MAP.md",
        },
        "adaptive_lineage": {
            "profile": "standard",
            "rounds_completed": 1,
            "declared_total_variants": 9,
            "actual_total_variants": 9,
            "active_minutes_used": 35,
            "stop_reason": "predeclared deterministic cost grid completed",
        },
        "tags": ["CORE5", "DBC", "short_borrow", "implementation_cost", "research_only"],
    }
    write_json(study_root_path / "knowledge_record.json", knowledge_record_dict)


def write_manifest(study_root_path: Path) -> None:
    material_file_path_list = sorted(
        file_path
        for file_path in study_root_path.rglob("*")
        if file_path.is_file() and file_path.name != "run_manifest.json"
    )
    external_file_path_list = [
        REPO_ROOT_PATH / "scripts" / "research" / "run_adaptive_macro_core5_borrow_cost_study.py",
        REPO_ROOT_PATH / "tests" / "test_adaptive_macro_core5_borrow_cost_study.py",
        REPO_ROOT_PATH / "strategies" / "taa_beyond_6040" / "strategy_taa_adaptive_macro_core5.py",
    ]
    manifest_dict = {
        "schema_version": "quant-research-run-manifest-v1",
        "study_id": "adaptive_macro_core5_dbc_borrow_cost",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "bundle_files": [
            {
                "path": str(file_path.relative_to(study_root_path)).replace("\\", "/"),
                "sha256": sha256_file_str(file_path),
                "bytes": file_path.stat().st_size,
            }
            for file_path in material_file_path_list
        ],
        "external_deliverables": [
            {
                "path": str(file_path.relative_to(REPO_ROOT_PATH)).replace("\\", "/"),
                "sha256": sha256_file_str(file_path),
                "bytes": file_path.stat().st_size,
            }
            for file_path in external_file_path_list
            if file_path.exists()
        ],
        "data_lineage": {
            "vendor": "Norgate Data local database",
            "loaded_at": datetime.now(timezone.utc).isoformat(),
            "licensed_mutable_input": True,
            "location": "local Norgate database through data.norgate_loader.load_raw_prices",
        },
    }
    write_json(study_root_path / "run_manifest.json", manifest_dict)


def run_study(
    study_root_path: Path = STUDY_ROOT_PATH,
    end_date_str: str | None = None,
    show_progress_bool: bool = False,
) -> Path:
    study_root_path.mkdir(parents=True, exist_ok=True)
    table_output_path = study_root_path / "tables"
    chart_output_path = study_root_path / "charts"
    table_output_path.mkdir(parents=True, exist_ok=True)
    chart_output_path.mkdir(parents=True, exist_ok=True)

    config_obj = replace(DEFAULT_CONFIG, end_date_str=end_date_str)
    print("LOAD Norgate data", flush=True)
    pricing_data_df = get_adaptive_macro_core5_data(config_obj=config_obj)
    calendar_idx = build_execution_calendar_idx(
        pricing_data_df=pricing_data_df,
        config_obj=config_obj,
        backtest_start_date_str=config_obj.backtest_start_date_str,
    )
    print(
        f"CALENDAR {calendar_idx[0].date()} to {calendar_idx[-1].date()} "
        f"({len(calendar_idx)} sessions)",
        flush=True,
    )

    print("RUN literal baseline oracle", flush=True)
    baseline_strategy_obj = run_literal_baseline(
        pricing_data_df=pricing_data_df,
        calendar_idx=calendar_idx,
        config_obj=config_obj,
        show_progress_bool=show_progress_bool,
    )

    strategy_by_variant_dict: dict[str, BorrowCostAdaptiveMacroCore5Strategy] = {}
    comparison_row_list: list[dict[str, object]] = []
    subperiod_record_list: list[dict[str, object]] = []
    equity_curve_df = pd.DataFrame(index=baseline_strategy_obj.results.index)
    for annual_borrow_rate_pct_float in ANNUAL_BORROW_RATE_PCT_TUPLE:
        variant_key_str = f"borrow_{annual_borrow_rate_pct_float:g}pct"
        print(f"RUN {variant_key_str}", flush=True)
        strategy_obj = run_research_strategy(
            pricing_data_df=pricing_data_df,
            calendar_idx=calendar_idx,
            config_obj=config_obj,
            annual_borrow_rate_float=annual_borrow_rate_pct_float / 100.0,
            disable_dbc_short_bool=False,
            show_progress_bool=show_progress_bool,
        )
        strategy_by_variant_dict[variant_key_str] = strategy_obj
        comparison_row_list.append(
            variant_row_dict(
                variant_key_str=variant_key_str,
                strategy_obj=strategy_obj,
                annual_borrow_rate_pct_float=annual_borrow_rate_pct_float,
                disable_dbc_short_bool=False,
            )
        )
        subperiod_record_list.extend(subperiod_row_list(variant_key_str, strategy_obj))
        equity_curve_df[variant_key_str] = strategy_obj.results["total_value"].astype(float)

    print("RUN no_short", flush=True)
    no_short_strategy_obj = run_research_strategy(
        pricing_data_df=pricing_data_df,
        calendar_idx=calendar_idx,
        config_obj=config_obj,
        annual_borrow_rate_float=0.0,
        disable_dbc_short_bool=True,
        show_progress_bool=show_progress_bool,
    )
    strategy_by_variant_dict["no_short"] = no_short_strategy_obj
    comparison_row_list.append(
        variant_row_dict(
            variant_key_str="no_short",
            strategy_obj=no_short_strategy_obj,
            annual_borrow_rate_pct_float=None,
            disable_dbc_short_bool=True,
        )
    )
    subperiod_record_list.extend(subperiod_row_list("no_short", no_short_strategy_obj))
    equity_curve_df["no_short"] = no_short_strategy_obj.results["total_value"].astype(float)
    equity_curve_df["benchmark_spx_total_return"] = baseline_strategy_obj.results["$SPX"].astype(float)
    equity_curve_df.index.name = "date_ts"

    zero_strategy_obj = strategy_by_variant_dict["borrow_0pct"]
    baseline_value_ser = baseline_strategy_obj.results["total_value"].astype(float)
    zero_value_ser = zero_strategy_obj.results["total_value"].astype(float)
    maximum_equity_gap_float = float((baseline_value_ser - zero_value_ser).abs().max())
    baseline_transaction_df = baseline_strategy_obj.get_transactions().reset_index(drop=True)
    zero_transaction_df = zero_strategy_obj.get_transactions().reset_index(drop=True)
    transaction_economic_column_list = [
        "trade_id",
        "bar",
        "asset",
        "amount",
        "price",
        "total_value",
        "commission",
    ]
    exact_transaction_parity_bool = baseline_transaction_df[
        transaction_economic_column_list
    ].equals(zero_transaction_df[transaction_economic_column_list])
    parity_dict = {
        "exact_parity_bool": bool(maximum_equity_gap_float == 0.0 and exact_transaction_parity_bool),
        "maximum_daily_equity_gap_float": maximum_equity_gap_float,
        "baseline_transaction_count_int": int(len(baseline_transaction_df)),
        "zero_rate_transaction_count_int": int(len(zero_transaction_df)),
        "exact_transaction_parity_bool": bool(exact_transaction_parity_bool),
        "excluded_identity_column_list": ["order_id"],
        "exclusion_reason_str": "order_id is a process-global sequence and has no economic effect",
    }
    if not parity_dict["exact_parity_bool"]:
        raise RuntimeError(f"Zero-rate parity failed: {parity_dict}")

    comparison_df = pd.DataFrame(comparison_row_list)
    subperiod_df = pd.DataFrame(subperiod_record_list)
    gate_df = build_gate_df(comparison_df)
    comparison_df.to_csv(table_output_path / "borrow_rate_sweep.csv", index=False)
    subperiod_df.to_csv(table_output_path / "subperiod_metrics.csv", index=False)
    gate_df.to_csv(table_output_path / "promotion_gates.csv", index=False)
    equity_curve_df.to_csv(table_output_path / "equity_curves.csv.gz", compression="gzip")
    strategy_by_variant_dict["borrow_1pct"].borrow_fee_df.to_csv(
        table_output_path / "borrow_fee_ledger_1pct.csv",
        index=False,
    )
    baseline_strategy_obj.realized_weight_df.to_csv(
        table_output_path / "realized_weights_zero_rate.csv"
    )
    write_json(table_output_path / "zero_rate_parity.json", parity_dict)

    source_evidence_dict = {
        "as_of_date_str": "2026-08-25",
        "scope_str": "public methodology and DBC snapshots; not account-specific live quotes",
        "official_methodology": [
            {
                "publisher_str": "Interactive Brokers",
                "url_str": "https://www.interactivebrokers.com/en/pricing/short-sale-cost.php?menu=B",
                "finding_str": "Short-sale cost combines borrow fee and short-sale-proceeds interest; rates vary with supply and demand.",
            },
            {
                "publisher_str": "Interactive Brokers",
                "url_str": "https://ibkrguides.com/reportingreference/reportguide/borrowfeedetails_default.htm",
                "finding_str": "Daily fee is Value times Fee Rate divided by 360; collateral convention is 102% of prior settlement price rounded up per share.",
            },
            {
                "publisher_str": "SEC",
                "url_str": "https://www.sec.gov/investor/pubs/regsho.htm",
                "finding_str": "Short sellers face borrow or locate requirements, interest, and responsibility for distributions.",
            },
        ],
        "dbc_specific_snapshots": [
            {
                "source_str": "Fintel public DBC short-interest page",
                "url_str": "https://fintel.io/ko/ss/us/dbc",
                "observed_period_str": "2026-01 to 2026-02",
                "observed_rate_range_pct_str": "0.43% to 0.66%",
                "limitation_str": "third-party snapshots, not a complete historical series",
            },
            {
                "source_str": "ChartExchange DBC borrow-fee page",
                "url_str": "https://chartexchange.com/symbol/nyse-dbc/borrow-fee/",
                "observed_period_str": "2026-03-12",
                "observed_rate_pct_str": "0.50%",
                "limitation_str": "third-party snapshot attributed to a public IBKR feed",
            },
            {
                "source_str": "2015 ETF shorting research snapshot",
                "url_str": "https://www.fullertreacymoney.com/system/data/files/PDFs/2015/June/0900b8c0899e9106.pdf",
                "observed_period_str": "2015 snapshot",
                "observed_rate_pct_str": "0.93%",
                "limitation_str": "single historical cross-sectional snapshot",
            },
        ],
        "current_account_specific_quote_retrieved_bool": False,
        "conclusion_str": "1% is conservative relative to available DBC snapshots, but is not a validated historical mean or current account quote.",
    }
    write_json(study_root_path / "source_evidence.json", source_evidence_dict)
    create_charts(strategy_by_variant_dict, comparison_df, chart_output_path)
    write_reports(comparison_df, subperiod_df, gate_df, parity_dict, study_root_path)
    write_knowledge_record(comparison_df, gate_df, study_root_path)
    create_notebook(study_root_path)
    write_manifest(study_root_path)
    print(f"STUDY SAVED {study_root_path}", flush=True)
    return study_root_path


def parse_args() -> argparse.Namespace:
    parser_obj = argparse.ArgumentParser(description=__doc__)
    parser_obj.add_argument("--study-root", type=Path, default=STUDY_ROOT_PATH)
    parser_obj.add_argument("--end", default=None)
    parser_obj.add_argument("--show-progress", action="store_true")
    return parser_obj.parse_args()


def main() -> None:
    arg_namespace = parse_args()
    run_study(
        study_root_path=arg_namespace.study_root.resolve(),
        end_date_str=arg_namespace.end,
        show_progress_bool=arg_namespace.show_progress,
    )


if __name__ == "__main__":
    main()
