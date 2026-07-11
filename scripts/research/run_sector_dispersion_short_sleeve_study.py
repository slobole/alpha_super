from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
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
from alpha.engine.strategy import Strategy
from scripts.research.run_sector_dispersion_marginal_universe_study import (
    OUT_OF_SAMPLE_START_TS,
    _json_default_obj,
    _run_strategy_variant,
    _strategy_summary_row_dict,
)
from strategies.mean_reversion.strategy_mr_sector_dispersion_ibs import (
    DEFAULT_CONFIG,
    ORIGINAL_SYMBOL_TUPLE,
    UNIVERSE_A_SYMBOL_TUPLE,
    UNIVERSE_B_SYMBOL_TUPLE,
    UNIVERSE_C_SYMBOL_TUPLE,
    SectorDispersionIbsConfig,
    compute_sector_dispersion_ibs_signal_df,
    get_sector_dispersion_ibs_data,
)


SHORT_MODE_MIRROR_STR = "mirror"
SHORT_MODE_SPX_SMA200_STR = "spx_sma200"
DEFAULT_SHORT_GROSS_EXPOSURE_FLOAT = 1.0
DEFAULT_SPX_SMA_LOOKBACK_DAY_INT = 200
DEFAULT_MARKET_TAIL_QUANTILE_FLOAT = 0.10
DEFAULT_SHORT_ALLOCATION_TUPLE = (0.10, 0.20, 0.30)


def default_trade_id_int() -> int:
    return -1


@dataclass(frozen=True)
class SectorDispersionShortSleeveConfig:
    base_config_obj: SectorDispersionIbsConfig
    short_mode_str: str = SHORT_MODE_MIRROR_STR
    short_gross_exposure_float: float = DEFAULT_SHORT_GROSS_EXPOSURE_FLOAT
    spx_sma_lookback_day_int: int = DEFAULT_SPX_SMA_LOOKBACK_DAY_INT

    def __post_init__(self) -> None:
        if self.short_mode_str not in {SHORT_MODE_MIRROR_STR, SHORT_MODE_SPX_SMA200_STR}:
            raise ValueError(
                f"short_mode_str must be {SHORT_MODE_MIRROR_STR!r} or {SHORT_MODE_SPX_SMA200_STR!r}."
            )
        if self.short_gross_exposure_float <= 0.0:
            raise ValueError("short_gross_exposure_float must be positive.")
        if self.spx_sma_lookback_day_int <= 1:
            raise ValueError("spx_sma_lookback_day_int must be greater than 1.")


class SectorDispersionIbsShortSleeveStrategy(Strategy):
    """
    Research-only mirrored short sleeve for sector-dispersion IBS.

    For ETF i on decision date t:

        short_entry_{i,t}
            = 1[IBS_{i,t} > exit_ibs_min]
              * 1[RelativeRange_{i,t} > min_relative_range]
              * optional_regime_gate_t

        cover_{i,t}
            = 1[IBS_{i,t} < entry_ibs_max]
              * 1[RelativeRange_{i,t} > min_relative_range]

        target_weight_i = -short_gross_exposure / N

    Orders are submitted from bar T and filled by the standard engine at
    Open_{T+1}. This path is research-only and does not model borrow,
    locates, recalls, or borrow fees.
    """

    enable_signal_audit = True
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: list[str] | tuple[str, ...],
        short_config_obj: SectorDispersionShortSleeveConfig,
    ):
        base_config_obj = short_config_obj.base_config_obj
        super().__init__(
            name=name,
            benchmarks=list(benchmarks),
            capital_base=base_config_obj.capital_base_float,
            slippage=base_config_obj.slippage_float,
            commission_per_share=base_config_obj.commission_per_share_float,
            commission_minimum=base_config_obj.commission_minimum_float,
        )
        self.short_config_obj = short_config_obj
        self.config_obj = base_config_obj
        self.symbol_tuple = tuple(base_config_obj.symbol_tuple)
        self.trade_id_int = 0
        self.current_trade_map: defaultdict[str, int] = defaultdict(default_trade_id_int)
        self.target_short_weight_float = float(short_config_obj.short_gross_exposure_float) / float(
            len(self.symbol_tuple)
        )

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        signal_data_df = compute_sector_dispersion_ibs_signal_df(
            pricing_data_df=pricing_data_df,
            config_obj=self.config_obj,
        )
        if self.short_config_obj.short_mode_str != SHORT_MODE_SPX_SMA200_STR:
            return signal_data_df

        benchmark_symbol_str = self.config_obj.benchmark_symbol_str
        benchmark_close_key = (benchmark_symbol_str, "Close")
        if benchmark_close_key not in signal_data_df.columns:
            raise RuntimeError(f"Missing benchmark close column: {benchmark_close_key}")
        benchmark_close_ser = pd.to_numeric(signal_data_df[benchmark_close_key], errors="coerce")

        # *** CRITICAL*** SPX regime is evaluated after the decision bar close
        # and executed at Open T+1. Including Close_T in SMA_T is intentional
        # because the base sector signal also uses the completed daily bar T.
        benchmark_sma_ser = benchmark_close_ser.rolling(
            window=int(self.short_config_obj.spx_sma_lookback_day_int),
            min_periods=int(self.short_config_obj.spx_sma_lookback_day_int),
        ).mean()
        benchmark_below_sma_bool_ser = pd.Series(False, index=benchmark_close_ser.index, dtype=bool)
        valid_sma_bool_ser = benchmark_sma_ser.notna()
        benchmark_below_sma_bool_ser.loc[valid_sma_bool_ser] = benchmark_close_ser.loc[
            valid_sma_bool_ser
        ].lt(benchmark_sma_ser.loc[valid_sma_bool_ser])

        regime_feature_df = pd.DataFrame(
            {
                (
                    benchmark_symbol_str,
                    f"sma_{self.short_config_obj.spx_sma_lookback_day_int}_ser",
                ): benchmark_sma_ser,
                (benchmark_symbol_str, "below_sma_regime_bool"): benchmark_below_sma_bool_ser,
            },
            index=signal_data_df.index,
        )
        regime_feature_df.columns = pd.MultiIndex.from_tuples(regime_feature_df.columns)
        return pd.concat([signal_data_df, regime_feature_df], axis=1)

    def iterate(self, data_df: pd.DataFrame, close_row_ser: pd.Series, open_price_ser: pd.Series):
        if close_row_ser is None or data_df is None:
            return

        short_entry_allowed_bool = self._short_entry_allowed_bool(close_row_ser=close_row_ser)
        position_ser = self.get_positions()
        held_symbol_set = {
            str(symbol_str)
            for symbol_str, position_float in position_ser.items()
            if str(symbol_str) in self.symbol_tuple and float(position_float) < 0.0
        }

        for symbol_str in self.symbol_tuple:
            cover_signal_bool = bool(close_row_ser.get((symbol_str, "entry_signal_bool"), False))
            if symbol_str in held_symbol_set and (cover_signal_bool or not short_entry_allowed_bool):
                self.order_target(
                    symbol_str,
                    0.0,
                    trade_id=self.current_trade_map[symbol_str],
                )
                held_symbol_set.remove(symbol_str)

        if not short_entry_allowed_bool:
            return

        for symbol_str in self.symbol_tuple:
            if symbol_str in held_symbol_set:
                continue
            if self.get_position(symbol_str) != 0:
                continue

            short_entry_signal_bool = bool(close_row_ser.get((symbol_str, "exit_signal_bool"), False))
            if not short_entry_signal_bool:
                continue

            target_share_float = self._short_target_share_float(
                symbol_str=symbol_str,
                close_row_ser=close_row_ser,
            )
            self.trade_id_int += 1
            self.current_trade_map[symbol_str] = self.trade_id_int
            self.order_target(
                symbol_str,
                target_share_float,
                trade_id=self.trade_id_int,
            )

    def _short_entry_allowed_bool(self, close_row_ser: pd.Series) -> bool:
        if self.short_config_obj.short_mode_str == SHORT_MODE_MIRROR_STR:
            return True

        regime_key = (self.config_obj.benchmark_symbol_str, "below_sma_regime_bool")
        regime_value_obj = close_row_ser.get(regime_key, False)
        if pd.isna(regime_value_obj):
            return False
        return bool(regime_value_obj)

    def _short_target_share_float(
        self,
        symbol_str: str,
        close_row_ser: pd.Series,
    ) -> float:
        close_price_float = float(close_row_ser.get((symbol_str, "Close"), np.nan))
        if not np.isfinite(close_price_float) or close_price_float <= 0.0:
            raise RuntimeError(f"Cannot size {symbol_str} short entry without a valid decision-bar close.")

        return -float(self.previous_total_value) * self.target_short_weight_float / close_price_float


@dataclass(frozen=True)
class AssetBasketSpec:
    basket_key_str: str
    label_str: str
    symbol_tuple: tuple[str, ...]


ASSET_BASKET_SPEC_TUPLE = (
    AssetBasketSpec("base", "Base", ORIGINAL_SYMBOL_TUPLE),
    AssetBasketSpec("kie_ihi", "Base+KIE+IHI", ORIGINAL_SYMBOL_TUPLE + ("KIE", "IHI")),
    AssetBasketSpec("kie_ihi_xlc", "Base+KIE+IHI+XLC", ORIGINAL_SYMBOL_TUPLE + ("KIE", "IHI", "XLC")),
    AssetBasketSpec("kie_xlre_ihi", "Base+KIE+XLRE+IHI", ORIGINAL_SYMBOL_TUPLE + ("KIE", "XLRE", "IHI")),
    AssetBasketSpec("universe_a", "Full Universe A", UNIVERSE_A_SYMBOL_TUPLE),
    AssetBasketSpec("universe_b", "Full Universe B", UNIVERSE_B_SYMBOL_TUPLE),
    AssetBasketSpec("universe_c", "Full Universe C", UNIVERSE_C_SYMBOL_TUPLE),
)

LONG_BASELINE_KEY_TUPLE = ("kie_ihi", "kie_ihi_xlc")


def _slug_str(raw_value_str: str) -> str:
    keep_char_list: list[str] = []
    for char_str in str(raw_value_str).lower():
        keep_char_list.append(char_str if char_str.isalnum() else "_")
    return "_".join(filter(None, "".join(keep_char_list).split("_")))


def _daily_return_ser(total_value_ser: pd.Series) -> pd.Series:
    clean_total_value_ser = pd.to_numeric(total_value_ser, errors="coerce").dropna()
    clean_total_value_ser.index = pd.to_datetime(clean_total_value_ser.index).normalize()
    # *** CRITICAL*** Post-run diagnostic only:
    # r_t = equity_t / equity_{t-1} - 1. These realized returns are not used
    # inside the backtest loop, signal construction, or order sizing.
    return clean_total_value_ser.pct_change(fill_method=None).dropna()


def _normalize_equity_ser(total_value_ser: pd.Series) -> pd.Series:
    clean_total_value_ser = pd.to_numeric(total_value_ser, errors="coerce").dropna()
    if len(clean_total_value_ser) == 0:
        raise ValueError("Cannot normalize an empty equity series.")
    first_value_float = float(clean_total_value_ser.iloc[0])
    if not np.isfinite(first_value_float) or first_value_float <= 0.0:
        raise ValueError("Cannot normalize an equity series with non-positive start value.")
    return clean_total_value_ser / first_value_float


def _correlation_float(left_return_ser: pd.Series, right_return_ser: pd.Series) -> float:
    aligned_return_df = pd.concat(
        [left_return_ser.rename("left"), right_return_ser.rename("right")],
        axis=1,
    ).dropna()
    if len(aligned_return_df) < 3:
        return float("nan")
    return float(aligned_return_df["left"].corr(aligned_return_df["right"]))


def _beta_float(strategy_return_ser: pd.Series, benchmark_return_ser: pd.Series) -> float:
    aligned_return_df = pd.concat(
        [strategy_return_ser.rename("strategy"), benchmark_return_ser.rename("benchmark")],
        axis=1,
    ).dropna()
    if len(aligned_return_df) < 3:
        return float("nan")
    benchmark_variance_float = float(aligned_return_df["benchmark"].var())
    if not np.isfinite(benchmark_variance_float) or benchmark_variance_float == 0.0:
        return float("nan")
    return float(aligned_return_df["strategy"].cov(aligned_return_df["benchmark"]) / benchmark_variance_float)


def _benchmark_return_ser(pricing_data_df: pd.DataFrame, benchmark_symbol_str: str) -> pd.Series:
    benchmark_close_ser = pd.to_numeric(pricing_data_df[(benchmark_symbol_str, "Close")], errors="coerce")
    benchmark_close_ser.index = pd.to_datetime(benchmark_close_ser.index).normalize()
    # *** CRITICAL*** Benchmark returns are diagnostics only and are computed
    # after strategy runs complete. They must not feed signal generation.
    return benchmark_close_ser.pct_change(fill_method=None).dropna()


def _performance_metric_dict(total_value_ser: pd.Series, prefix_str: str = "") -> dict[str, object]:
    clean_total_value_ser = pd.to_numeric(total_value_ser, errors="coerce").dropna()
    clean_total_value_ser.index = pd.to_datetime(clean_total_value_ser.index).normalize()
    metric_prefix_str = "" if prefix_str == "" else f"{prefix_str}_"
    metric_dict: dict[str, object] = {
        f"{metric_prefix_str}start_date_str": None,
        f"{metric_prefix_str}end_date_str": None,
        f"{metric_prefix_str}day_count_int": int(len(clean_total_value_ser)),
        f"{metric_prefix_str}ann_return_pct_float": np.nan,
        f"{metric_prefix_str}volatility_ann_pct_float": np.nan,
        f"{metric_prefix_str}sharpe_float": np.nan,
        f"{metric_prefix_str}max_drawdown_pct_float": np.nan,
        f"{metric_prefix_str}terminal_multiple_float": np.nan,
    }
    if len(clean_total_value_ser) < 2:
        return metric_dict

    daily_return_ser = clean_total_value_ser.pct_change(fill_method=None).dropna()
    running_peak_ser = clean_total_value_ser.cummax()
    drawdown_ser = clean_total_value_ser / running_peak_ser - 1.0
    day_count_float = float(len(clean_total_value_ser))
    std_return_float = float(daily_return_ser.std())
    sharpe_float = np.nan if std_return_float == 0.0 else float(daily_return_ser.mean() / std_return_float * np.sqrt(252.0))
    terminal_multiple_float = float(clean_total_value_ser.iloc[-1] / clean_total_value_ser.iloc[0])
    ann_return_pct_float = (terminal_multiple_float ** (252.0 / day_count_float) - 1.0) * 100.0

    metric_dict.update(
        {
            f"{metric_prefix_str}start_date_str": clean_total_value_ser.index[0].date().isoformat(),
            f"{metric_prefix_str}end_date_str": clean_total_value_ser.index[-1].date().isoformat(),
            f"{metric_prefix_str}ann_return_pct_float": ann_return_pct_float,
            f"{metric_prefix_str}volatility_ann_pct_float": float(std_return_float * np.sqrt(252.0) * 100.0),
            f"{metric_prefix_str}sharpe_float": sharpe_float,
            f"{metric_prefix_str}max_drawdown_pct_float": float(drawdown_ser.min() * 100.0),
            f"{metric_prefix_str}terminal_multiple_float": terminal_multiple_float,
        }
    )
    return metric_dict


def _period_metric_dict(
    total_value_ser: pd.Series,
    start_ts: pd.Timestamp,
    prefix_str: str,
) -> dict[str, object]:
    clean_total_value_ser = pd.to_numeric(total_value_ser, errors="coerce").dropna()
    clean_total_value_ser.index = pd.to_datetime(clean_total_value_ser.index).normalize()
    return _performance_metric_dict(
        total_value_ser=clean_total_value_ser.loc[clean_total_value_ser.index >= start_ts],
        prefix_str=prefix_str,
    )


def _market_metric_dict(
    total_value_ser: pd.Series,
    benchmark_return_ser: pd.Series,
    market_tail_quantile_float: float,
) -> dict[str, object]:
    strategy_return_ser = _daily_return_ser(total_value_ser)
    aligned_return_df = pd.concat(
        [strategy_return_ser.rename("strategy"), benchmark_return_ser.rename("benchmark")],
        axis=1,
    ).dropna()
    if len(aligned_return_df) < 3:
        return {
            "corr_to_spx_float": np.nan,
            "market_down_corr_to_spx_float": np.nan,
            "market_tail_corr_to_spx_float": np.nan,
            "market_tail_beta_to_spx_float": np.nan,
            "market_tail_mean_return_pct_float": np.nan,
            "market_tail_spx_mean_return_pct_float": np.nan,
            "market_tail_day_count_int": 0,
        }

    market_down_return_df = aligned_return_df.loc[aligned_return_df["benchmark"].lt(0.0)]
    market_tail_threshold_float = float(aligned_return_df["benchmark"].quantile(market_tail_quantile_float))
    market_tail_return_df = aligned_return_df.loc[aligned_return_df["benchmark"].le(market_tail_threshold_float)]
    return {
        "corr_to_spx_float": _correlation_float(aligned_return_df["strategy"], aligned_return_df["benchmark"]),
        "market_down_corr_to_spx_float": _correlation_float(
            market_down_return_df["strategy"],
            market_down_return_df["benchmark"],
        ),
        "market_tail_corr_to_spx_float": _correlation_float(
            market_tail_return_df["strategy"],
            market_tail_return_df["benchmark"],
        ),
        "market_tail_beta_to_spx_float": _beta_float(
            market_tail_return_df["strategy"],
            market_tail_return_df["benchmark"],
        ),
        "market_tail_mean_return_pct_float": float(market_tail_return_df["strategy"].mean() * 100.0),
        "market_tail_spx_mean_return_pct_float": float(market_tail_return_df["benchmark"].mean() * 100.0),
        "market_tail_day_count_int": int(len(market_tail_return_df)),
    }


def _score_portfolio_row_dict(row_dict: dict[str, object]) -> float:
    dd_improvement_float = _safe_float(row_dict.get("delta_max_drawdown_pct_float"))
    tail_improvement_float = _safe_float(row_dict.get("delta_market_tail_mean_return_pct_float"))
    beta_reduction_float = -_safe_float(row_dict.get("delta_market_tail_beta_to_spx_float"))
    ann_penalty_float = -max(0.0, -_safe_float(row_dict.get("delta_ann_return_pct_float")))
    return (
        0.35 * dd_improvement_float
        + 0.30 * tail_improvement_float
        + 0.25 * beta_reduction_float
        + 0.10 * ann_penalty_float
    )


def _safe_float(value_obj: object) -> float:
    try:
        value_float = float(value_obj)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(value_float):
        return float("nan")
    return value_float


def _run_short_strategy_variant(
    strategy_name_str: str,
    symbol_tuple: tuple[str, ...],
    base_config_obj: SectorDispersionIbsConfig,
    short_mode_str: str,
    pricing_data_df: pd.DataFrame,
    show_progress_bool: bool,
) -> SectorDispersionIbsShortSleeveStrategy:
    config_obj = replace(
        base_config_obj,
        symbol_tuple=tuple(symbol_tuple),
        universe_name_str="original",
    )
    short_config_obj = SectorDispersionShortSleeveConfig(
        base_config_obj=config_obj,
        short_mode_str=short_mode_str,
    )
    strategy_obj = SectorDispersionIbsShortSleeveStrategy(
        name=strategy_name_str,
        benchmarks=[config_obj.benchmark_symbol_str],
        short_config_obj=short_config_obj,
    )

    # *** CRITICAL*** Keep pre-start history for both the sector range scale
    # and the optional SPX SMA gate, but only execute on/after backtest start.
    calendar_idx = pricing_data_df.index[
        pricing_data_df.index >= pd.Timestamp(config_obj.backtest_start_date_str)
    ]
    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar=calendar_idx,
        show_progress=show_progress_bool,
        show_signal_progress_bool=show_progress_bool,
        audit_override_bool=False,
    )
    return strategy_obj


def _combined_total_value_ser(
    long_total_value_ser: pd.Series,
    short_total_value_ser: pd.Series,
    short_allocation_float: float,
    capital_base_float: float,
) -> pd.Series:
    if not 0.0 < short_allocation_float < 1.0:
        raise ValueError("short_allocation_float must lie between 0 and 1.")
    long_weight_float = 1.0 - float(short_allocation_float)
    long_norm_ser = _normalize_equity_ser(long_total_value_ser)
    short_norm_ser = _normalize_equity_ser(short_total_value_ser)
    aligned_norm_df = pd.concat(
        [long_norm_ser.rename("long"), short_norm_ser.rename("short")],
        axis=1,
    ).dropna()
    combined_norm_ser = long_weight_float * aligned_norm_df["long"] + float(short_allocation_float) * aligned_norm_df["short"]
    return combined_norm_ser * float(capital_base_float)


def _markdown_table_str(table_df: pd.DataFrame) -> str:
    if table_df.empty:
        return "_No rows._"

    display_df = table_df.copy()
    for column_str in display_df.columns:
        display_df[column_str] = display_df[column_str].map(
            lambda value_obj: f"{float(value_obj):.4f}"
            if isinstance(value_obj, (float, np.floating)) and np.isfinite(float(value_obj))
            else str(value_obj)
        )

    header_list = [str(column_str) for column_str in display_df.columns]
    row_list = display_df.astype(str).values.tolist()
    line_list = [
        "| " + " | ".join(header_list) + " |",
        "| " + " | ".join(["---"] * len(header_list)) + " |",
    ]
    for row_value_list in row_list:
        line_list.append("| " + " | ".join(row_value_list) + " |")
    return "\n".join(line_list)


def _write_markdown_summary(
    output_path: Path,
    short_summary_df: pd.DataFrame,
    portfolio_leaderboard_df: pd.DataFrame,
    search_count_int: int,
) -> None:
    top_short_df = short_summary_df.sort_values(
        by=["market_tail_mean_return_pct_float", "max_drawdown_pct_float"],
        ascending=[False, False],
    ).head(10)
    top_portfolio_df = portfolio_leaderboard_df.head(10)
    top_short_table_str = _markdown_table_str(
        top_short_df[
            [
                "short_mode_str",
                "basket_label_str",
                "ann_return_pct_float",
                "sharpe_float",
                "max_drawdown_pct_float",
                "market_tail_mean_return_pct_float",
                "market_tail_beta_to_spx_float",
                "trade_count_int",
            ]
        ]
    )
    top_portfolio_table_str = _markdown_table_str(
        top_portfolio_df[
            [
                "long_label_str",
                "short_mode_str",
                "short_basket_label_str",
                "short_allocation_float",
                "ann_return_pct_float",
                "sharpe_float",
                "max_drawdown_pct_float",
                "market_tail_mean_return_pct_float",
                "market_tail_beta_to_spx_float",
                "delta_max_drawdown_pct_float",
                "delta_market_tail_mean_return_pct_float",
                "delta_market_tail_beta_to_spx_float",
            ]
        ]
    )
    summary_md_str = f"""# Sector Dispersion Short Sleeve Study

## Scope

- Research-only; no live/release wiring.
- Short side does not model borrow availability, borrow fees, recalls, hard-to-borrow constraints, or locate failures.
- Search count: `{search_count_int}` short sleeve runs plus portfolio allocation diagnostics.
- Order timing: signal from daily bar `T` fills at `Open_(T+1)` through the standard engine.

## Short Hypotheses

1. `mirror`: short when the long strategy would issue its overbought exit signal.
2. `spx_sma200`: same mirror short, but entries are allowed only when `$SPX < SMA200`; existing shorts are covered when the gate turns off.

## Top Standalone Short Sleeves By Market-Tail Return

{top_short_table_str}

## Top Long+Short Portfolio Diagnostics

{top_portfolio_table_str}

## Interpretation Reminder

A short sleeve is accepted only if it improves portfolio stress behavior after realistic caveats, not because it has one attractive standalone metric.
"""
    (output_path / "recommendations.md").write_text(summary_md_str, encoding="utf-8")


def _save_top_portfolio_chart(
    output_path: Path,
    portfolio_equity_df: pd.DataFrame,
    selected_column_list: list[str],
) -> None:
    if len(selected_column_list) == 0:
        return
    fig_obj, axis_obj = plt.subplots(figsize=(14, 8))
    for column_str in selected_column_list:
        norm_ser = _normalize_equity_ser(portfolio_equity_df[column_str])
        axis_obj.plot(norm_ser.index, norm_ser.values, label=column_str, linewidth=1.8)
    axis_obj.set_title("Sector Dispersion Short Sleeve: Top Portfolio Equity Curves")
    axis_obj.set_xlabel("Date")
    axis_obj.set_ylabel("Growth of 1.0")
    axis_obj.grid(True, alpha=0.25)
    axis_obj.legend(loc="best", fontsize=8)
    fig_obj.autofmt_xdate()
    fig_obj.tight_layout()
    fig_obj.savefig(output_path / "top_portfolio_equity_curves.png", dpi=160)
    plt.close(fig_obj)


def run_short_sleeve_study(
    output_dir_str: str = "results",
    end_date_str: str | None = None,
    show_progress_bool: bool = False,
) -> Path:
    timestamp_str = pd.Timestamp.now().strftime("%Y-%m-%d_%H%M%S")
    output_path = build_research_output_path(
        output_dir=output_dir_str,
        entity_type_str="strategy",
        entity_id_str="strategy_mr_sector_dispersion_ibs",
        analysis_type_str="short_sleeve_study",
        timestamp_str=timestamp_str,
    )
    output_path.mkdir(parents=True, exist_ok=False)

    all_symbol_tuple = tuple(
        dict.fromkeys(
            symbol_str
            for basket_spec_obj in ASSET_BASKET_SPEC_TUPLE
            for symbol_str in basket_spec_obj.symbol_tuple
        )
    )
    base_config_obj = replace(
        DEFAULT_CONFIG,
        symbol_tuple=all_symbol_tuple,
        universe_name_str="original",
        end_date_str=end_date_str,
    )
    pricing_data_df = get_sector_dispersion_ibs_data(config_obj=base_config_obj)
    benchmark_return_ser = _benchmark_return_ser(
        pricing_data_df=pricing_data_df,
        benchmark_symbol_str=base_config_obj.benchmark_symbol_str,
    )
    basket_by_key_dict = {basket_spec_obj.basket_key_str: basket_spec_obj for basket_spec_obj in ASSET_BASKET_SPEC_TUPLE}

    manifest_row_dict_list: list[dict[str, object]] = []
    for basket_spec_obj in ASSET_BASKET_SPEC_TUPLE:
        manifest_row_dict_list.append(
            {
                "basket_key_str": basket_spec_obj.basket_key_str,
                "basket_label_str": basket_spec_obj.label_str,
                "symbol_count_int": len(basket_spec_obj.symbol_tuple),
                "symbol_tuple_str": ",".join(basket_spec_obj.symbol_tuple),
            }
        )
    pd.DataFrame(manifest_row_dict_list).to_csv(output_path / "asset_basket_manifest.csv", index=False)

    long_strategy_dict: dict[str, object] = {}
    long_summary_row_list: list[dict[str, object]] = []
    for long_key_str in LONG_BASELINE_KEY_TUPLE:
        basket_spec_obj = basket_by_key_dict[long_key_str]
        strategy_obj = _run_strategy_variant(
            strategy_name_str=f"strategy_mr_sector_dispersion_ibs_long_{basket_spec_obj.basket_key_str}",
            symbol_tuple=basket_spec_obj.symbol_tuple,
            base_config_obj=base_config_obj,
            pricing_data_df=pricing_data_df,
            show_progress_bool=show_progress_bool,
        )
        long_strategy_dict[long_key_str] = strategy_obj
        row_dict = _strategy_summary_row_dict(
            strategy_obj=strategy_obj,
            variant_kind_str="long_baseline",
            candidate_symbol_str=None,
            bucket_str=None,
        )
        row_dict["long_key_str"] = long_key_str
        row_dict["long_label_str"] = basket_spec_obj.label_str
        row_dict.update(_market_metric_dict(strategy_obj.results["total_value"], benchmark_return_ser, DEFAULT_MARKET_TAIL_QUANTILE_FLOAT))
        long_summary_row_list.append(row_dict)

    short_strategy_dict: dict[tuple[str, str], SectorDispersionIbsShortSleeveStrategy] = {}
    short_summary_row_list: list[dict[str, object]] = []
    short_mode_tuple = (SHORT_MODE_MIRROR_STR, SHORT_MODE_SPX_SMA200_STR)
    for short_mode_str in short_mode_tuple:
        for basket_spec_obj in ASSET_BASKET_SPEC_TUPLE:
            strategy_name_str = (
                "strategy_mr_sector_dispersion_ibs_short_"
                f"{short_mode_str}_{_slug_str(basket_spec_obj.basket_key_str)}"
            )
            print(f"Running short sleeve {short_mode_str} / {basket_spec_obj.label_str}...", flush=True)
            strategy_obj = _run_short_strategy_variant(
                strategy_name_str=strategy_name_str,
                symbol_tuple=basket_spec_obj.symbol_tuple,
                base_config_obj=base_config_obj,
                short_mode_str=short_mode_str,
                pricing_data_df=pricing_data_df,
                show_progress_bool=show_progress_bool,
            )
            short_strategy_dict[(short_mode_str, basket_spec_obj.basket_key_str)] = strategy_obj
            row_dict = _strategy_summary_row_dict(
                strategy_obj=strategy_obj,
                variant_kind_str="short_sleeve",
                candidate_symbol_str=None,
                bucket_str=None,
            )
            row_dict["short_mode_str"] = short_mode_str
            row_dict["basket_key_str"] = basket_spec_obj.basket_key_str
            row_dict["basket_label_str"] = basket_spec_obj.label_str
            row_dict["short_gross_exposure_float"] = DEFAULT_SHORT_GROSS_EXPOSURE_FLOAT
            row_dict.update(_market_metric_dict(strategy_obj.results["total_value"], benchmark_return_ser, DEFAULT_MARKET_TAIL_QUANTILE_FLOAT))
            short_summary_row_list.append(row_dict)

    long_summary_df = pd.DataFrame(long_summary_row_list)
    short_summary_df = pd.DataFrame(short_summary_row_list)
    long_summary_df.to_csv(output_path / "long_baseline_summary.csv", index=False)
    short_summary_df.to_csv(output_path / "short_sleeve_summary.csv", index=False)

    portfolio_row_list: list[dict[str, object]] = []
    portfolio_equity_dict: dict[str, pd.Series] = {}
    for long_key_str, long_strategy_obj in long_strategy_dict.items():
        long_basket_spec_obj = basket_by_key_dict[long_key_str]
        long_total_value_ser = long_strategy_obj.results["total_value"]
        long_metric_dict = _performance_metric_dict(long_total_value_ser)
        long_market_dict = _market_metric_dict(long_total_value_ser, benchmark_return_ser, DEFAULT_MARKET_TAIL_QUANTILE_FLOAT)
        for (short_mode_str, short_basket_key_str), short_strategy_obj in short_strategy_dict.items():
            short_basket_spec_obj = basket_by_key_dict[short_basket_key_str]
            for short_allocation_float in DEFAULT_SHORT_ALLOCATION_TUPLE:
                combined_total_value_ser = _combined_total_value_ser(
                    long_total_value_ser=long_total_value_ser,
                    short_total_value_ser=short_strategy_obj.results["total_value"],
                    short_allocation_float=short_allocation_float,
                    capital_base_float=base_config_obj.capital_base_float,
                )
                portfolio_label_str = (
                    f"{long_basket_spec_obj.label_str} + {short_allocation_float:.0%} "
                    f"{short_mode_str} short {short_basket_spec_obj.label_str}"
                )
                portfolio_equity_dict[portfolio_label_str] = combined_total_value_ser
                row_dict = {
                    "portfolio_label_str": portfolio_label_str,
                    "long_key_str": long_key_str,
                    "long_label_str": long_basket_spec_obj.label_str,
                    "short_mode_str": short_mode_str,
                    "short_basket_key_str": short_basket_key_str,
                    "short_basket_label_str": short_basket_spec_obj.label_str,
                    "short_allocation_float": float(short_allocation_float),
                }
                row_dict.update(_performance_metric_dict(combined_total_value_ser))
                row_dict.update(_period_metric_dict(combined_total_value_ser, OUT_OF_SAMPLE_START_TS, "oos"))
                row_dict.update(_market_metric_dict(combined_total_value_ser, benchmark_return_ser, DEFAULT_MARKET_TAIL_QUANTILE_FLOAT))
                row_dict["delta_ann_return_pct_float"] = (
                    _safe_float(row_dict.get("ann_return_pct_float"))
                    - _safe_float(long_metric_dict.get("ann_return_pct_float"))
                )
                row_dict["delta_sharpe_float"] = (
                    _safe_float(row_dict.get("sharpe_float"))
                    - _safe_float(long_metric_dict.get("sharpe_float"))
                )
                row_dict["delta_max_drawdown_pct_float"] = (
                    _safe_float(row_dict.get("max_drawdown_pct_float"))
                    - _safe_float(long_metric_dict.get("max_drawdown_pct_float"))
                )
                row_dict["delta_market_tail_mean_return_pct_float"] = (
                    _safe_float(row_dict.get("market_tail_mean_return_pct_float"))
                    - _safe_float(long_market_dict.get("market_tail_mean_return_pct_float"))
                )
                row_dict["delta_market_tail_beta_to_spx_float"] = (
                    _safe_float(row_dict.get("market_tail_beta_to_spx_float"))
                    - _safe_float(long_market_dict.get("market_tail_beta_to_spx_float"))
                )
                row_dict["composite_score_float"] = _score_portfolio_row_dict(row_dict)
                portfolio_row_list.append(row_dict)

    portfolio_summary_df = pd.DataFrame(portfolio_row_list)
    portfolio_leaderboard_df = portfolio_summary_df.sort_values(
        by=["composite_score_float", "delta_market_tail_mean_return_pct_float", "delta_max_drawdown_pct_float"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    portfolio_summary_df.to_csv(output_path / "portfolio_combination_summary.csv", index=False)
    portfolio_leaderboard_df.to_csv(output_path / "portfolio_leaderboard.csv", index=False)

    portfolio_equity_df = pd.DataFrame(portfolio_equity_dict).sort_index()
    portfolio_equity_df.to_csv(output_path / "portfolio_equity_curves.csv", index_label="date")
    top_portfolio_label_list = portfolio_leaderboard_df["portfolio_label_str"].head(6).astype(str).tolist()
    _save_top_portfolio_chart(
        output_path=output_path,
        portfolio_equity_df=portfolio_equity_df,
        selected_column_list=top_portfolio_label_list,
    )
    _write_markdown_summary(
        output_path=output_path,
        short_summary_df=short_summary_df,
        portfolio_leaderboard_df=portfolio_leaderboard_df,
        search_count_int=len(short_mode_tuple) * len(ASSET_BASKET_SPEC_TUPLE),
    )

    metadata_dict = {
        "analysis_type_str": "short_sleeve_study",
        "generated_at_str": pd.Timestamp.now().isoformat(),
        "output_path_str": str(output_path),
        "end_date_str": end_date_str,
        "short_mode_tuple": short_mode_tuple,
        "asset_basket_count_int": len(ASSET_BASKET_SPEC_TUPLE),
        "short_run_count_int": len(short_mode_tuple) * len(ASSET_BASKET_SPEC_TUPLE),
        "short_allocation_tuple": DEFAULT_SHORT_ALLOCATION_TUPLE,
        "market_tail_quantile_float": DEFAULT_MARKET_TAIL_QUANTILE_FLOAT,
        "spx_sma_lookback_day_int": DEFAULT_SPX_SMA_LOOKBACK_DAY_INT,
        "short_gross_exposure_float": DEFAULT_SHORT_GROSS_EXPOSURE_FLOAT,
        "borrow_model_note_str": "No borrow availability, borrow fee, locate, recall, or hard-to-borrow model is included.",
        "execution_timing_note_str": "Signal from daily bar T fills at Open T+1 through the standard runner.",
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata_dict, indent=2, default=_json_default_obj),
        encoding="utf-8",
    )
    return output_path


def _parse_args() -> argparse.Namespace:
    parser_obj = argparse.ArgumentParser(
        description="Run research-only sector-dispersion mirrored short-sleeve diagnostics."
    )
    parser_obj.add_argument("--output-dir", default="results")
    parser_obj.add_argument("--end-date", default=None)
    parser_obj.add_argument("--show-progress", action="store_true")
    return parser_obj.parse_args()


def main() -> int:
    args_obj = _parse_args()
    output_path = run_short_sleeve_study(
        output_dir_str=str(args_obj.output_dir),
        end_date_str=args_obj.end_date,
        show_progress_bool=bool(args_obj.show_progress),
    )
    print(f"Saved short-sleeve study to {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
