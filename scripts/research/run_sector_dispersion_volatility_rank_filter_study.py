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
    _strategy_summary_row_dict,
)
from scripts.research.run_sector_dispersion_short_sleeve_study import (
    _benchmark_return_ser,
    _market_metric_dict,
    _markdown_table_str,
    _normalize_equity_ser,
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


BULLISH_FILTER_NONE_STR = "none"
BULLISH_FILTER_SPX_SMA_STR = "spx_sma"
BULLISH_FILTER_ASSET_SMA_STR = "asset_sma"
DEFAULT_BULLISH_SMA_LOOKBACK_DAY_INT = 200
DEFAULT_MARKET_TAIL_QUANTILE_FLOAT = 0.10


def default_trade_id_int() -> int:
    return -1


@dataclass(frozen=True)
class SectorDispersionVolatilityRankFilterConfig:
    base_config_obj: SectorDispersionIbsConfig
    vol_rank_top_n_int: int | None = None
    bullish_filter_str: str = BULLISH_FILTER_NONE_STR
    bullish_sma_lookback_day_int: int = DEFAULT_BULLISH_SMA_LOOKBACK_DAY_INT

    def __post_init__(self) -> None:
        symbol_count_int = len(self.base_config_obj.symbol_tuple)
        if symbol_count_int <= 0:
            raise ValueError("base_config_obj.symbol_tuple must not be empty.")
        if self.vol_rank_top_n_int is not None:
            if self.vol_rank_top_n_int <= 0:
                raise ValueError("vol_rank_top_n_int must be positive when provided.")
            if self.vol_rank_top_n_int > symbol_count_int:
                raise ValueError("vol_rank_top_n_int must not exceed the symbol count.")
        if self.bullish_filter_str not in {
            BULLISH_FILTER_NONE_STR,
            BULLISH_FILTER_SPX_SMA_STR,
            BULLISH_FILTER_ASSET_SMA_STR,
        }:
            raise ValueError("bullish_filter_str must be none, spx_sma, or asset_sma.")
        if self.bullish_sma_lookback_day_int <= 1:
            raise ValueError("bullish_sma_lookback_day_int must be greater than 1.")


class SectorDispersionVolatilityRankFilterStrategy(Strategy):
    """
    Research-only sector-dispersion IBS variant with volatility-ranked entries.

    For ETF i on decision date t:

        entry_base_{i,t}
            = 1[IBS_{i,t} < entry_ibs_max]
              * 1[RelativeRange_{i,t} > min_relative_range]

        vol_rank_{i,t}
            = descending cross-sectional rank of RangeVol_{i,t}

        vol_allowed_{i,t}
            = 1[vol_rank_{i,t} <= top_n]

        bull_allowed_{i,t}
            = 1, or 1[$SPX_t > SMA($SPX)_t],
              or 1[Close_{i,t} > SMA(Close_i)_t]

        entry_{i,t}
            = entry_base_{i,t} * vol_allowed_{i,t} * bull_allowed_{i,t}

    Orders are still submitted from bar T and filled by the standard engine at
    Open_{T+1}. Existing positions keep the original strategy's exit rule.
    """

    enable_signal_audit = True
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: list[str] | tuple[str, ...],
        filter_config_obj: SectorDispersionVolatilityRankFilterConfig,
    ):
        base_config_obj = filter_config_obj.base_config_obj
        super().__init__(
            name=name,
            benchmarks=list(benchmarks),
            capital_base=base_config_obj.capital_base_float,
            slippage=base_config_obj.slippage_float,
            commission_per_share=base_config_obj.commission_per_share_float,
            commission_minimum=base_config_obj.commission_minimum_float,
        )
        self.filter_config_obj = filter_config_obj
        self.config_obj = base_config_obj
        self.symbol_tuple = tuple(base_config_obj.symbol_tuple)
        self.trade_id_int = 0
        self.current_trade_map: defaultdict[str, int] = defaultdict(default_trade_id_int)
        denominator_int = (
            len(self.symbol_tuple)
            if filter_config_obj.vol_rank_top_n_int is None
            else int(filter_config_obj.vol_rank_top_n_int)
        )
        self.target_weight_float = float(base_config_obj.portfolio_leverage_float) / float(denominator_int)

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        signal_data_df = compute_sector_dispersion_ibs_signal_df(
            pricing_data_df=pricing_data_df,
            config_obj=self.config_obj,
        )
        feature_frame_list: list[pd.DataFrame] = []

        if self.filter_config_obj.vol_rank_top_n_int is not None:
            feature_frame_list.append(self._vol_rank_feature_df(signal_data_df=signal_data_df))

        if self.filter_config_obj.bullish_filter_str == BULLISH_FILTER_SPX_SMA_STR:
            feature_frame_list.append(self._spx_bullish_feature_df(signal_data_df=signal_data_df))
        elif self.filter_config_obj.bullish_filter_str == BULLISH_FILTER_ASSET_SMA_STR:
            feature_frame_list.append(self._asset_bullish_feature_df(signal_data_df=signal_data_df))

        if len(feature_frame_list) == 0:
            return signal_data_df
        return pd.concat([signal_data_df] + feature_frame_list, axis=1)

    def _vol_rank_feature_df(self, signal_data_df: pd.DataFrame) -> pd.DataFrame:
        range_vol_field_str = f"range_vol_{self.config_obj.range_vol_lookback_day_int}_ser"
        range_vol_df = pd.DataFrame(
            {
                symbol_str: pd.to_numeric(signal_data_df[(symbol_str, range_vol_field_str)], errors="coerce")
                for symbol_str in self.symbol_tuple
            },
            index=signal_data_df.index,
            dtype=float,
        )
        # *** CRITICAL*** range_vol_df is already lagged by one full trading
        # day inside compute_sector_dispersion_ibs_signal_df. The rank below
        # may order candidates at T, but it must not use Range_T or later data.
        vol_rank_df = range_vol_df.rank(axis=1, ascending=False, method="first")
        allowed_bool_df = vol_rank_df.le(float(self.filter_config_obj.vol_rank_top_n_int))
        allowed_bool_df = allowed_bool_df & vol_rank_df.notna()

        rank_feature_df = pd.concat(
            [
                _multiindex_feature_df(vol_rank_df, "vol_rank_int"),
                _multiindex_feature_df(allowed_bool_df.astype(bool), "vol_rank_allowed_bool"),
            ],
            axis=1,
        )
        return rank_feature_df

    def _spx_bullish_feature_df(self, signal_data_df: pd.DataFrame) -> pd.DataFrame:
        benchmark_symbol_str = self.config_obj.benchmark_symbol_str
        benchmark_close_key = (benchmark_symbol_str, "Close")
        if benchmark_close_key not in signal_data_df.columns:
            raise RuntimeError(f"Missing benchmark close column: {benchmark_close_key}")
        benchmark_close_ser = pd.to_numeric(signal_data_df[benchmark_close_key], errors="coerce")

        # *** CRITICAL*** SPX SMA_T uses the completed decision bar close and
        # can only affect orders filled at Open T+1. Do not use it for same-bar
        # open or intraday decisions.
        benchmark_sma_ser = benchmark_close_ser.rolling(
            window=int(self.filter_config_obj.bullish_sma_lookback_day_int),
            min_periods=int(self.filter_config_obj.bullish_sma_lookback_day_int),
        ).mean()
        benchmark_bull_bool_ser = pd.Series(False, index=benchmark_close_ser.index, dtype=bool)
        valid_sma_bool_ser = benchmark_sma_ser.notna()
        benchmark_bull_bool_ser.loc[valid_sma_bool_ser] = benchmark_close_ser.loc[
            valid_sma_bool_ser
        ].gt(benchmark_sma_ser.loc[valid_sma_bool_ser])

        feature_df = pd.DataFrame(
            {
                (
                    benchmark_symbol_str,
                    f"sma_{self.filter_config_obj.bullish_sma_lookback_day_int}_ser",
                ): benchmark_sma_ser,
                (benchmark_symbol_str, "spx_bullish_bool"): benchmark_bull_bool_ser,
            },
            index=signal_data_df.index,
        )
        feature_df.columns = pd.MultiIndex.from_tuples(feature_df.columns)
        return feature_df

    def _asset_bullish_feature_df(self, signal_data_df: pd.DataFrame) -> pd.DataFrame:
        close_price_df = pd.DataFrame(
            {
                symbol_str: pd.to_numeric(signal_data_df[(symbol_str, "Close")], errors="coerce")
                for symbol_str in self.symbol_tuple
            },
            index=signal_data_df.index,
            dtype=float,
        )
        # *** CRITICAL*** Asset SMA_T uses the completed daily close for T and
        # only gates entries that fill at Open T+1.
        asset_sma_df = close_price_df.rolling(
            window=int(self.filter_config_obj.bullish_sma_lookback_day_int),
            min_periods=int(self.filter_config_obj.bullish_sma_lookback_day_int),
        ).mean()
        asset_bull_bool_df = close_price_df.gt(asset_sma_df) & asset_sma_df.notna()
        return pd.concat(
            [
                _multiindex_feature_df(
                    asset_sma_df,
                    f"asset_sma_{self.filter_config_obj.bullish_sma_lookback_day_int}_ser",
                ),
                _multiindex_feature_df(asset_bull_bool_df.astype(bool), "asset_bullish_bool"),
            ],
            axis=1,
        )

    def iterate(self, data_df: pd.DataFrame, close_row_ser: pd.Series, open_price_ser: pd.Series):
        if close_row_ser is None or data_df is None:
            return

        position_ser = self.get_positions()
        held_symbol_set = {
            str(symbol_str)
            for symbol_str, position_float in position_ser.items()
            if str(symbol_str) in self.symbol_tuple and float(position_float) > 0.0
        }

        for symbol_str in self.symbol_tuple:
            exit_signal_bool = bool(close_row_ser.get((symbol_str, "exit_signal_bool"), False))
            if symbol_str in held_symbol_set and exit_signal_bool:
                self.order_target(
                    symbol_str,
                    0.0,
                    trade_id=self.current_trade_map[symbol_str],
                )
                held_symbol_set.remove(symbol_str)

        for symbol_str in self.symbol_tuple:
            if symbol_str in held_symbol_set:
                continue
            if self.get_position(symbol_str) != 0:
                continue
            if not self._entry_allowed_bool(symbol_str=symbol_str, close_row_ser=close_row_ser):
                continue

            target_share_float = self._entry_target_share_float(
                symbol_str=symbol_str,
                close_row_ser=close_row_ser,
            )
            self.trade_id_int += 1
            self.current_trade_map[symbol_str] = self.trade_id_int
            self.order_target(symbol_str, target_share_float, trade_id=self.trade_id_int)

    def _entry_allowed_bool(self, symbol_str: str, close_row_ser: pd.Series) -> bool:
        entry_signal_bool = bool(close_row_ser.get((symbol_str, "entry_signal_bool"), False))
        if not entry_signal_bool:
            return False

        if self.filter_config_obj.vol_rank_top_n_int is not None:
            vol_rank_allowed_bool = bool(close_row_ser.get((symbol_str, "vol_rank_allowed_bool"), False))
            if not vol_rank_allowed_bool:
                return False

        if self.filter_config_obj.bullish_filter_str == BULLISH_FILTER_SPX_SMA_STR:
            spx_bullish_obj = close_row_ser.get((self.config_obj.benchmark_symbol_str, "spx_bullish_bool"), False)
            if pd.isna(spx_bullish_obj) or not bool(spx_bullish_obj):
                return False

        if self.filter_config_obj.bullish_filter_str == BULLISH_FILTER_ASSET_SMA_STR:
            asset_bullish_obj = close_row_ser.get((symbol_str, "asset_bullish_bool"), False)
            if pd.isna(asset_bullish_obj) or not bool(asset_bullish_obj):
                return False

        return True

    def _entry_target_share_float(self, symbol_str: str, close_row_ser: pd.Series) -> float:
        close_price_float = float(close_row_ser.get((symbol_str, "Close"), np.nan))
        if not np.isfinite(close_price_float) or close_price_float <= 0.0:
            raise RuntimeError(f"Cannot size {symbol_str} entry without a valid decision-bar close.")
        return float(self.previous_total_value) * self.target_weight_float / close_price_float


def _multiindex_feature_df(feature_df: pd.DataFrame, field_str: str) -> pd.DataFrame:
    output_feature_df = feature_df.copy()
    output_feature_df.columns = pd.MultiIndex.from_tuples(
        [(str(symbol_str), field_str) for symbol_str in output_feature_df.columns]
    )
    return output_feature_df


@dataclass(frozen=True)
class VariantSpec:
    variant_key_str: str
    basket_label_str: str
    symbol_tuple: tuple[str, ...]
    vol_rank_top_n_int: int | None
    bullish_filter_str: str


def _variant_spec_tuple() -> tuple[VariantSpec, ...]:
    base_plus_kie_ihi_tuple = ORIGINAL_SYMBOL_TUPLE + ("KIE", "IHI")
    base_plus_kie_ihi_xlc_tuple = ORIGINAL_SYMBOL_TUPLE + ("KIE", "IHI", "XLC")
    return (
        VariantSpec("base_kie_ihi", "Base+KIE+IHI", base_plus_kie_ihi_tuple, None, BULLISH_FILTER_NONE_STR),
        VariantSpec(
            "base_kie_ihi_spx_bull",
            "Base+KIE+IHI",
            base_plus_kie_ihi_tuple,
            None,
            BULLISH_FILTER_SPX_SMA_STR,
        ),
        VariantSpec(
            "base_kie_ihi_asset_bull",
            "Base+KIE+IHI",
            base_plus_kie_ihi_tuple,
            None,
            BULLISH_FILTER_ASSET_SMA_STR,
        ),
        VariantSpec("base_kie_ihi_xlc", "Base+KIE+IHI+XLC", base_plus_kie_ihi_xlc_tuple, None, BULLISH_FILTER_NONE_STR),
        VariantSpec(
            "base_kie_ihi_xlc_spx_bull",
            "Base+KIE+IHI+XLC",
            base_plus_kie_ihi_xlc_tuple,
            None,
            BULLISH_FILTER_SPX_SMA_STR,
        ),
        VariantSpec(
            "base_kie_ihi_xlc_asset_bull",
            "Base+KIE+IHI+XLC",
            base_plus_kie_ihi_xlc_tuple,
            None,
            BULLISH_FILTER_ASSET_SMA_STR,
        ),
        VariantSpec("universe_a_top5", "Universe A", UNIVERSE_A_SYMBOL_TUPLE, 5, BULLISH_FILTER_NONE_STR),
        VariantSpec("universe_a_top5_spx_bull", "Universe A", UNIVERSE_A_SYMBOL_TUPLE, 5, BULLISH_FILTER_SPX_SMA_STR),
        VariantSpec(
            "universe_a_top5_asset_bull",
            "Universe A",
            UNIVERSE_A_SYMBOL_TUPLE,
            5,
            BULLISH_FILTER_ASSET_SMA_STR,
        ),
        VariantSpec("universe_b_top5", "Universe B", UNIVERSE_B_SYMBOL_TUPLE, 5, BULLISH_FILTER_NONE_STR),
        VariantSpec("universe_b_top5_spx_bull", "Universe B", UNIVERSE_B_SYMBOL_TUPLE, 5, BULLISH_FILTER_SPX_SMA_STR),
        VariantSpec(
            "universe_b_top5_asset_bull",
            "Universe B",
            UNIVERSE_B_SYMBOL_TUPLE,
            5,
            BULLISH_FILTER_ASSET_SMA_STR,
        ),
        VariantSpec("universe_b_top8", "Universe B", UNIVERSE_B_SYMBOL_TUPLE, 8, BULLISH_FILTER_NONE_STR),
        VariantSpec("universe_b_top8_spx_bull", "Universe B", UNIVERSE_B_SYMBOL_TUPLE, 8, BULLISH_FILTER_SPX_SMA_STR),
        VariantSpec("universe_c_top3", "Universe C", UNIVERSE_C_SYMBOL_TUPLE, 3, BULLISH_FILTER_NONE_STR),
        VariantSpec("universe_c_top3_spx_bull", "Universe C", UNIVERSE_C_SYMBOL_TUPLE, 3, BULLISH_FILTER_SPX_SMA_STR),
        VariantSpec(
            "universe_c_top3_asset_bull",
            "Universe C",
            UNIVERSE_C_SYMBOL_TUPLE,
            3,
            BULLISH_FILTER_ASSET_SMA_STR,
        ),
        VariantSpec("universe_c_top5", "Universe C", UNIVERSE_C_SYMBOL_TUPLE, 5, BULLISH_FILTER_NONE_STR),
        VariantSpec("universe_c_top5_spx_bull", "Universe C", UNIVERSE_C_SYMBOL_TUPLE, 5, BULLISH_FILTER_SPX_SMA_STR),
        VariantSpec(
            "universe_c_top5_asset_bull",
            "Universe C",
            UNIVERSE_C_SYMBOL_TUPLE,
            5,
            BULLISH_FILTER_ASSET_SMA_STR,
        ),
        VariantSpec("universe_c_top8", "Universe C", UNIVERSE_C_SYMBOL_TUPLE, 8, BULLISH_FILTER_NONE_STR),
        VariantSpec("universe_c_top8_spx_bull", "Universe C", UNIVERSE_C_SYMBOL_TUPLE, 8, BULLISH_FILTER_SPX_SMA_STR),
        VariantSpec(
            "universe_c_top8_asset_bull",
            "Universe C",
            UNIVERSE_C_SYMBOL_TUPLE,
            8,
            BULLISH_FILTER_ASSET_SMA_STR,
        ),
    )


def _run_variant_spec(
    variant_spec_obj: VariantSpec,
    base_config_obj: SectorDispersionIbsConfig,
    pricing_data_df: pd.DataFrame,
    show_progress_bool: bool,
) -> SectorDispersionVolatilityRankFilterStrategy:
    config_obj = replace(
        base_config_obj,
        symbol_tuple=variant_spec_obj.symbol_tuple,
        universe_name_str="original",
    )
    filter_config_obj = SectorDispersionVolatilityRankFilterConfig(
        base_config_obj=config_obj,
        vol_rank_top_n_int=variant_spec_obj.vol_rank_top_n_int,
        bullish_filter_str=variant_spec_obj.bullish_filter_str,
    )
    strategy_obj = SectorDispersionVolatilityRankFilterStrategy(
        name=f"strategy_mr_sector_dispersion_ibs_{variant_spec_obj.variant_key_str}",
        benchmarks=[config_obj.benchmark_symbol_str],
        filter_config_obj=filter_config_obj,
    )

    # *** CRITICAL*** Keep pre-start history for range-vol ranks and SMA gates,
    # but only submit orders on and after backtest_start_date_str. Every order
    # is still signal T -> Open T+1 through run_daily.
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


def _exposure_metric_dict(strategy_obj: SectorDispersionVolatilityRankFilterStrategy) -> dict[str, object]:
    realized_weight_df = getattr(strategy_obj, "realized_weight_df", pd.DataFrame())
    if realized_weight_df is None or len(realized_weight_df) == 0:
        return {
            "avg_position_count_float": np.nan,
            "avg_gross_exposure_pct_float": np.nan,
            "active_day_pct_float": np.nan,
        }
    weight_df = realized_weight_df.copy()
    weight_df.index = pd.to_datetime(weight_df.index).normalize()
    weight_df.columns = [str(column_obj) for column_obj in weight_df.columns]
    symbol_column_list = [symbol_str for symbol_str in strategy_obj.symbol_tuple if symbol_str in weight_df.columns]
    if len(symbol_column_list) == 0:
        return {
            "avg_position_count_float": np.nan,
            "avg_gross_exposure_pct_float": np.nan,
            "active_day_pct_float": np.nan,
        }
    symbol_weight_df = weight_df[symbol_column_list].apply(pd.to_numeric, errors="coerce")
    gross_exposure_ser = symbol_weight_df.abs().sum(axis=1)
    position_count_ser = symbol_weight_df.abs().gt(1e-9).sum(axis=1)
    return {
        "avg_position_count_float": float(position_count_ser.mean()),
        "avg_gross_exposure_pct_float": float(gross_exposure_ser.mean() * 100.0),
        "active_day_pct_float": float(gross_exposure_ser.gt(1e-9).mean() * 100.0),
    }


def _score_row_dict(row_dict: dict[str, object]) -> float:
    oos_sharpe_float = _safe_float(row_dict.get("oos_sharpe_float"))
    sharpe_float = _safe_float(row_dict.get("sharpe_float"))
    max_drawdown_pct_float = _safe_float(row_dict.get("max_drawdown_pct_float"))
    market_tail_mean_return_pct_float = _safe_float(row_dict.get("market_tail_mean_return_pct_float"))
    cost_drag_ann_pct_float = _safe_float(row_dict.get("cost_drag_ann_pct_float"))
    return (
        0.45 * oos_sharpe_float
        + 0.25 * sharpe_float
        + 0.02 * max_drawdown_pct_float
        + 0.20 * market_tail_mean_return_pct_float
        - 0.10 * cost_drag_ann_pct_float
    )


def _safe_float(value_obj: object) -> float:
    try:
        value_float = float(value_obj)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(value_float):
        return float("nan")
    return value_float


def _save_top_chart(
    output_path: Path,
    equity_df: pd.DataFrame,
    selected_column_list: list[str],
) -> None:
    fig_obj, axis_obj = plt.subplots(figsize=(14, 8))
    for column_str in selected_column_list:
        norm_ser = _normalize_equity_ser(equity_df[column_str])
        axis_obj.plot(norm_ser.index, norm_ser.values, label=column_str, linewidth=1.8)
    axis_obj.set_title("Sector Dispersion Volatility Rank / Bullish Filter Variants")
    axis_obj.set_xlabel("Date")
    axis_obj.set_ylabel("Growth of 1.0")
    axis_obj.grid(True, alpha=0.25)
    axis_obj.legend(loc="best", fontsize=8)
    fig_obj.autofmt_xdate()
    fig_obj.tight_layout()
    fig_obj.savefig(output_path / "top_variant_equity_curves.png", dpi=160)
    plt.close(fig_obj)


def _write_recommendations_md(
    output_path: Path,
    leaderboard_df: pd.DataFrame,
    search_count_int: int,
) -> None:
    column_list = [
        "variant_key_str",
        "basket_label_str",
        "vol_rank_top_n_int",
        "bullish_filter_str",
        "ann_return_pct_float",
        "oos_sharpe_float",
        "sharpe_float",
        "max_drawdown_pct_float",
        "market_tail_mean_return_pct_float",
        "market_tail_beta_to_spx_float",
        "trade_count_int",
        "avg_gross_exposure_pct_float",
        "composite_score_float",
    ]
    recommendation_md_str = f"""# Sector Dispersion Volatility Rank / Bullish Filter Study

## Scope

- Research-only; no live/release wiring.
- Search count: `{search_count_int}` variants.
- Execution convention remains `signal from daily bar T -> Open T+1`.
- Volatility rank uses the already-lagged range-volatility denominator from the base strategy.
- Bullish filters are entry filters only; existing positions exit on the original overbought IBS rule.

## Top Ranked Variants

{_markdown_table_str(leaderboard_df[column_list].head(15))}

## Interpretation Reminder

This sweep tests whether high-volatility asset selection and bullish entry gates improve the current sector-dispersion IBS sleeve. Treat this as a controlled research screen, not a deployment decision.
"""
    (output_path / "recommendations.md").write_text(recommendation_md_str, encoding="utf-8")


def run_volatility_rank_filter_study(
    output_dir_str: str = "results",
    end_date_str: str | None = None,
    show_progress_bool: bool = False,
) -> Path:
    variant_spec_tuple = _variant_spec_tuple()
    timestamp_str = pd.Timestamp.now().strftime("%Y-%m-%d_%H%M%S")
    output_path = build_research_output_path(
        output_dir=output_dir_str,
        entity_type_str="strategy",
        entity_id_str="strategy_mr_sector_dispersion_ibs",
        analysis_type_str="volatility_rank_filter_study",
        timestamp_str=timestamp_str,
    )
    output_path.mkdir(parents=True, exist_ok=False)

    all_symbol_tuple = tuple(
        dict.fromkeys(
            symbol_str
            for variant_spec_obj in variant_spec_tuple
            for symbol_str in variant_spec_obj.symbol_tuple
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

    manifest_df = pd.DataFrame(
        [
            {
                "variant_key_str": variant_spec_obj.variant_key_str,
                "basket_label_str": variant_spec_obj.basket_label_str,
                "symbol_count_int": len(variant_spec_obj.symbol_tuple),
                "symbol_tuple_str": ",".join(variant_spec_obj.symbol_tuple),
                "vol_rank_top_n_int": variant_spec_obj.vol_rank_top_n_int,
                "bullish_filter_str": variant_spec_obj.bullish_filter_str,
            }
            for variant_spec_obj in variant_spec_tuple
        ]
    )
    manifest_df.to_csv(output_path / "variant_manifest.csv", index=False)

    row_dict_list: list[dict[str, object]] = []
    equity_dict: dict[str, pd.Series] = {}
    for variant_spec_obj in variant_spec_tuple:
        print(f"Running {variant_spec_obj.variant_key_str}...", flush=True)
        strategy_obj = _run_variant_spec(
            variant_spec_obj=variant_spec_obj,
            base_config_obj=base_config_obj,
            pricing_data_df=pricing_data_df,
            show_progress_bool=show_progress_bool,
        )
        row_dict = _strategy_summary_row_dict(
            strategy_obj=strategy_obj,
            variant_kind_str="vol_rank_filter",
            candidate_symbol_str=None,
            bucket_str=None,
        )
        row_dict["variant_key_str"] = variant_spec_obj.variant_key_str
        row_dict["basket_label_str"] = variant_spec_obj.basket_label_str
        row_dict["vol_rank_top_n_int"] = variant_spec_obj.vol_rank_top_n_int
        row_dict["bullish_filter_str"] = variant_spec_obj.bullish_filter_str
        row_dict["target_weight_float"] = strategy_obj.target_weight_float
        row_dict.update(_market_metric_dict(strategy_obj.results["total_value"], benchmark_return_ser, DEFAULT_MARKET_TAIL_QUANTILE_FLOAT))
        row_dict.update(_exposure_metric_dict(strategy_obj))
        row_dict["composite_score_float"] = _score_row_dict(row_dict)
        row_dict_list.append(row_dict)
        equity_dict[variant_spec_obj.variant_key_str] = strategy_obj.results["total_value"]

    summary_df = pd.DataFrame(row_dict_list)
    leaderboard_df = summary_df.sort_values(
        by=[
            "composite_score_float",
            "oos_sharpe_float",
            "sharpe_float",
            "max_drawdown_pct_float",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    equity_df = pd.DataFrame(equity_dict).sort_index()

    summary_df.to_csv(output_path / "variant_summary.csv", index=False)
    leaderboard_df.to_csv(output_path / "variant_leaderboard.csv", index=False)
    equity_df.to_csv(output_path / "variant_equity_curves.csv", index_label="date")
    _save_top_chart(
        output_path=output_path,
        equity_df=equity_df,
        selected_column_list=leaderboard_df["variant_key_str"].head(8).astype(str).tolist(),
    )
    _write_recommendations_md(
        output_path=output_path,
        leaderboard_df=leaderboard_df,
        search_count_int=len(variant_spec_tuple),
    )

    metadata_dict = {
        "analysis_type_str": "volatility_rank_filter_study",
        "generated_at_str": pd.Timestamp.now().isoformat(),
        "output_path_str": str(output_path.resolve()),
        "end_date_str": end_date_str,
        "variant_count_int": len(variant_spec_tuple),
        "market_tail_quantile_float": DEFAULT_MARKET_TAIL_QUANTILE_FLOAT,
        "bullish_sma_lookback_day_int": DEFAULT_BULLISH_SMA_LOOKBACK_DAY_INT,
        "execution_timing_note_str": "Signal from daily bar T fills at Open T+1 through the standard runner.",
        "vol_rank_note_str": (
            "Volatility rank is based on lagged range volatility already computed as "
            "std(Range_{T-1} ... Range_{T-L})."
        ),
        "bullish_filter_note_str": "Bullish filters gate new entries only; original exits remain unchanged.",
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata_dict, indent=2, default=_json_default_obj),
        encoding="utf-8",
    )
    print(f"Saved volatility rank/filter study to {output_path}", flush=True)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser_obj = argparse.ArgumentParser(
        description="Run research-only sector-dispersion volatility-rank and bullish-filter diagnostics."
    )
    parser_obj.add_argument("--output-dir", default="results")
    parser_obj.add_argument("--end-date", default=None)
    parser_obj.add_argument("--show-progress", action="store_true")
    return parser_obj.parse_args()


def main() -> int:
    args_obj = _parse_args()
    run_volatility_rank_filter_study(
        output_dir_str=str(args_obj.output_dir),
        end_date_str=args_obj.end_date,
        show_progress_bool=bool(args_obj.show_progress),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
