"""
Monthly ATR-adjusted momentum rotation for the liquid US sector ETF basket.

Fixed universe:

    XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY, VOX, IYR

For ETF i on the actual month-end decision close t:

    ROC12_{i,t} = Close_{i,t} / Close_{i,t-12 month ends} - 1

    ATR20_{i,t} = mean(TrueRange_{i,t-19:t})

    source_score_{i,t} = ROC12_{i,t} / ATR20_{i,t}

    natr_score_{i,t} = ROC12_{i,t} / (ATR20_{i,t} / Close_{i,t})

Optional gates:

    market_trend_t = 1[SPY_t > SMA200(SPY)_t]
    asset_trend_{i,t} = 1[Close_{i,t} > SMA100(Close_i)_t]

Optional VIX exposure scaling:

    exposure_t = clip(20 / VIX_t, min_exposure, max_exposure)

Static exposure alternative:

    exposure_t = static_exposure

Selected ETFs receive equal fixed-slot weights, 1 / max_positions, multiplied
by exposure_t. Exposure above 1.0 creates negative cash. The engine reports
that borrowing but does not charge financing interest, so leveraged backtests
are optimistic sensitivity tests rather than deployment evidence.

Execution is deliberately realistic:

    month-end Close_T decision -> next tradable Open_T+1 execution

The strategy inherits the engine's usual cost parameters. It is research-only
and is not wired to LIVE, released pod configuration, or broker routing.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from data.norgate_loader import load_raw_prices
from strategies.momentum.strategy_mo_atr_normalized_ndx import (
    ATR_WINDOW_INT,
    AtrNormalizedNdxConfig,
    AtrNormalizedNdxStrategy,
    append_total_return_benchmark_data_df,
    compute_atr_normalized_signal_tables,
    configure_total_return_benchmark_provenance,
    get_asof_universe_membership_ser,
    map_month_end_decision_dates_to_rebalance_schedule_df,
)
from strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled import (
    load_vxn_close_ser,
)


SECTOR_SYMBOL_TUPLE = (
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLU",
    "XLV",
    "XLY",
    "VOX",
    "IYR",
)
SOURCE_ATR_SCORE_STR = "source_atr"
DIMENSIONLESS_NATR_SCORE_STR = "dimensionless_natr"
VIX_EXPOSURE_FIELD_STR = "vix_exposure_scale_float"


@dataclass(frozen=True)
class AtrNormalizedSectorConfig(AtrNormalizedNdxConfig):
    sector_symbol_tuple: tuple[str, ...] = SECTOR_SYMBOL_TUPLE
    history_start_date_str: str = "2004-09-01"
    backtest_start_date_str: str = "2005-10-01"
    max_positions_int: int = 3
    apply_market_trend_bool: bool = True
    apply_asset_trend_bool: bool = True
    score_mode_str: str = SOURCE_ATR_SCORE_STR
    use_vix_scale_bool: bool = True
    vix_symbol_str: str = "$VIX"
    target_vix_pct_float: float = 20.0
    min_exposure_scale_float: float = 0.25
    max_exposure_scale_float: float = 1.0
    static_exposure_scale_float: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.sector_symbol_tuple) == 0:
            raise ValueError("sector_symbol_tuple must not be empty.")
        if len(set(self.sector_symbol_tuple)) != len(self.sector_symbol_tuple):
            raise ValueError("sector_symbol_tuple must not contain duplicates.")
        if self.score_mode_str not in {
            SOURCE_ATR_SCORE_STR,
            DIMENSIONLESS_NATR_SCORE_STR,
        }:
            raise ValueError(f"Unsupported score_mode_str: {self.score_mode_str}.")
        if not self.vix_symbol_str:
            raise ValueError("vix_symbol_str must not be empty.")
        if self.target_vix_pct_float <= 0.0:
            raise ValueError("target_vix_pct_float must be positive.")
        if self.min_exposure_scale_float < 0.0:
            raise ValueError("min_exposure_scale_float must be non-negative.")
        if self.min_exposure_scale_float > self.max_exposure_scale_float:
            raise ValueError(
                "min_exposure_scale_float must be <= max_exposure_scale_float."
            )
        if self.max_exposure_scale_float > 1.5:
            raise ValueError(
                "max_exposure_scale_float must be <= 1.5 for this research strategy."
            )
        if not 0.0 < self.static_exposure_scale_float <= 1.5:
            raise ValueError(
                "static_exposure_scale_float must be in the interval (0.0, 1.5]."
            )


DEFAULT_CONFIG = AtrNormalizedSectorConfig()


def compute_vix_scale_signal_df(
    vix_close_ser: pd.Series,
    target_vix_pct_float: float,
    min_exposure_scale_float: float,
    max_exposure_scale_float: float,
) -> pd.DataFrame:
    """Compute the causal daily VIX exposure scale."""
    if target_vix_pct_float <= 0.0:
        raise ValueError("target_vix_pct_float must be positive.")
    if min_exposure_scale_float < 0.0:
        raise ValueError("min_exposure_scale_float must be non-negative.")
    if min_exposure_scale_float > max_exposure_scale_float:
        raise ValueError(
            "min_exposure_scale_float must be <= max_exposure_scale_float."
        )
    if max_exposure_scale_float > 1.5:
        raise ValueError(
            "max_exposure_scale_float must be <= 1.5 for this research strategy."
        )
    clean_vix_close_ser = vix_close_ser.astype(float).sort_index().dropna()
    if len(clean_vix_close_ser) == 0:
        raise ValueError("vix_close_ser must contain at least one non-null close.")

    vix_scale_signal_df = pd.DataFrame({"vix_close_float": clean_vix_close_ser})
    # *** CRITICAL*** The scaler uses only the VIX close available at decision
    # Close_T. The later as-of lookup may move backward, never forward.
    raw_exposure_scale_ser = (
        float(target_vix_pct_float) / vix_scale_signal_df["vix_close_float"]
    )
    vix_scale_signal_df[VIX_EXPOSURE_FIELD_STR] = raw_exposure_scale_ser.replace(
        [np.inf, -np.inf],
        np.nan,
    ).clip(
        lower=float(min_exposure_scale_float),
        upper=float(max_exposure_scale_float),
    )
    return vix_scale_signal_df.dropna(subset=[VIX_EXPOSURE_FIELD_STR])


def get_asof_vix_scale_float(
    vix_scale_signal_df: pd.DataFrame,
    decision_date_ts: pd.Timestamp,
) -> float:
    """Return the latest VIX exposure scale known on or before decision T."""
    if VIX_EXPOSURE_FIELD_STR not in vix_scale_signal_df.columns:
        raise RuntimeError(
            f"vix_scale_signal_df must contain {VIX_EXPOSURE_FIELD_STR}."
        )
    sorted_vix_scale_signal_df = vix_scale_signal_df.sort_index()
    # *** CRITICAL*** side="right" followed by -1 is a causal as-of lookup.
    # A VIX close after decision Close_T must never affect the order for T+1.
    vix_row_int = int(
        sorted_vix_scale_signal_df.index.searchsorted(
            pd.Timestamp(decision_date_ts),
            side="right",
        )
    ) - 1
    if vix_row_int < 0:
        raise RuntimeError(
            f"No VIX scale exists on or before decision date {decision_date_ts}."
        )
    exposure_scale_float = float(
        sorted_vix_scale_signal_df.iloc[vix_row_int][VIX_EXPOSURE_FIELD_STR]
    )
    if not np.isfinite(exposure_scale_float):
        raise RuntimeError(f"Invalid VIX scale on {decision_date_ts}.")
    return exposure_scale_float


def _trim_to_complete_fixed_basket_df(
    pricing_data_df: pd.DataFrame,
    config_obj: AtrNormalizedSectorConfig,
) -> pd.DataFrame:
    required_column_list = [
        (symbol_str, field_str)
        for symbol_str in config_obj.sector_symbol_tuple
        for field_str in ("Open", "High", "Low", "Close")
    ] + [(config_obj.regime_symbol_str, "Close")]
    missing_column_list = [
        column_tuple
        for column_tuple in required_column_list
        if column_tuple not in pricing_data_df.columns
    ]
    if missing_column_list:
        raise RuntimeError(f"Missing required price columns: {missing_column_list}.")

    complete_row_bool_ser = pricing_data_df.loc[:, required_column_list].notna().all(axis=1)
    if not complete_row_bool_ser.any():
        raise RuntimeError("The fixed ETF basket has no common complete price date.")
    first_complete_date_ts = pd.Timestamp(complete_row_bool_ser[complete_row_bool_ser].index[0])
    complete_pricing_data_df = pricing_data_df.loc[first_complete_date_ts:].copy()
    remaining_missing_count_int = int(
        complete_pricing_data_df.loc[:, required_column_list].isna().sum().sum()
    )
    if remaining_missing_count_int > 0:
        raise RuntimeError(
            "The fixed ETF basket has missing OHLC values after its common start: "
            f"missing_count={remaining_missing_count_int}."
        )
    return complete_pricing_data_df


def get_atr_normalized_sector_data(
    config_obj: AtrNormalizedSectorConfig = DEFAULT_CONFIG,
    *,
    include_total_return_benchmark_bool: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the fixed basket and build its month-end Close_T schedule."""
    execution_symbol_list = list(
        dict.fromkeys(
            list(config_obj.sector_symbol_tuple) + [config_obj.regime_symbol_str]
        )
    )
    pricing_data_df = load_raw_prices(
        symbols=execution_symbol_list,
        benchmarks=[],
        start_date=config_obj.history_start_date_str,
        end_date=config_obj.end_date_str,
    )
    pricing_data_df = _trim_to_complete_fixed_basket_df(
        pricing_data_df=pricing_data_df,
        config_obj=config_obj,
    )

    sector_symbol_list = list(config_obj.sector_symbol_tuple)
    price_close_df = pd.DataFrame(
        {
            symbol_str: pricing_data_df[(symbol_str, "Close")]
            for symbol_str in sector_symbol_list
        },
        index=pricing_data_df.index,
        dtype=float,
    )
    price_high_df = pd.DataFrame(
        {
            symbol_str: pricing_data_df[(symbol_str, "High")]
            for symbol_str in sector_symbol_list
        },
        index=pricing_data_df.index,
        dtype=float,
    )
    price_low_df = pd.DataFrame(
        {
            symbol_str: pricing_data_df[(symbol_str, "Low")]
            for symbol_str in sector_symbol_list
        },
        index=pricing_data_df.index,
        dtype=float,
    )
    regime_close_ser = pricing_data_df[
        (config_obj.regime_symbol_str, "Close")
    ].astype(float)
    monthly_decision_close_df, *_unused_result_tuple = (
        compute_atr_normalized_signal_tables(
            price_close_df=price_close_df,
            price_high_df=price_high_df,
            price_low_df=price_low_df,
            regime_close_ser=regime_close_ser,
            config=config_obj,
        )
    )
    rebalance_schedule_df = map_month_end_decision_dates_to_rebalance_schedule_df(
        decision_date_index=pd.DatetimeIndex(monthly_decision_close_df.index),
        execution_index=pd.DatetimeIndex(pricing_data_df.index),
    )

    fixed_universe_df = pd.DataFrame(
        1,
        index=pricing_data_df.index,
        columns=sector_symbol_list,
        dtype=int,
    )
    vix_close_ser = load_vxn_close_ser(
        symbol_str=config_obj.vix_symbol_str,
        start_date_str=config_obj.history_start_date_str,
        end_date_str=config_obj.end_date_str,
    )
    vix_scale_signal_df = compute_vix_scale_signal_df(
        vix_close_ser=vix_close_ser,
        target_vix_pct_float=config_obj.target_vix_pct_float,
        min_exposure_scale_float=config_obj.min_exposure_scale_float,
        max_exposure_scale_float=config_obj.max_exposure_scale_float,
    )
    if include_total_return_benchmark_bool:
        pricing_data_df = append_total_return_benchmark_data_df(
            pricing_data_df=pricing_data_df,
            config_obj=config_obj,
        )
    return (
        pricing_data_df.sort_index(),
        fixed_universe_df.sort_index(),
        rebalance_schedule_df.sort_index(),
        vix_scale_signal_df.sort_index(),
    )


class AtrNormalizedSectorStrategy(AtrNormalizedNdxStrategy):
    """Configurable research-only monthly trend rotation for the ETF basket."""

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        rebalance_schedule_df: pd.DataFrame,
        vix_scale_signal_df: pd.DataFrame,
        config_obj: AtrNormalizedSectorConfig,
    ) -> None:
        super().__init__(
            name=name,
            benchmarks=benchmarks,
            rebalance_schedule_df=rebalance_schedule_df,
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
        self.config_obj = config_obj
        self.sector_symbol_tuple = tuple(config_obj.sector_symbol_tuple)
        self.vix_scale_signal_df = vix_scale_signal_df.copy().sort_index()

    def get_tradeable_symbol_list(self, pricing_data: pd.DataFrame) -> list[str]:
        missing_symbol_list = [
            symbol_str
            for symbol_str in self.sector_symbol_tuple
            if (symbol_str, "Close") not in pricing_data.columns
        ]
        if missing_symbol_list:
            raise RuntimeError(f"Missing fixed-basket symbols: {missing_symbol_list}.")
        return list(self.sector_symbol_tuple)

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        signal_data_df = pricing_data.copy()
        sector_symbol_list = list(self.sector_symbol_tuple)
        price_close_df = pd.DataFrame(
            {
                symbol_str: signal_data_df[(symbol_str, "Close")]
                for symbol_str in sector_symbol_list
            },
            index=signal_data_df.index,
            dtype=float,
        )
        price_high_df = pd.DataFrame(
            {
                symbol_str: signal_data_df[(symbol_str, "High")]
                for symbol_str in sector_symbol_list
            },
            index=signal_data_df.index,
            dtype=float,
        )
        price_low_df = pd.DataFrame(
            {
                symbol_str: signal_data_df[(symbol_str, "Low")]
                for symbol_str in sector_symbol_list
            },
            index=signal_data_df.index,
            dtype=float,
        )
        regime_close_ser = signal_data_df[
            (self.regime_symbol_str, "Close")
        ].astype(float)
        (
            monthly_decision_close_df,
            monthly_roc_df,
            atr_decision_df,
            stock_trend_pass_df,
            regime_sma_ser,
            regime_pass_ser,
            source_score_df,
        ) = compute_atr_normalized_signal_tables(
            price_close_df=price_close_df,
            price_high_df=price_high_df,
            price_low_df=price_low_df,
            regime_close_ser=regime_close_ser,
            config=self.config_obj,
        )

        # *** CRITICAL*** Both score variants use only Close_T and the trailing
        # ATR20 ending at T. No T+1 open enters selection or sizing.
        natr_decision_df = atr_decision_df / monthly_decision_close_df
        dimensionless_score_df = monthly_roc_df / natr_decision_df
        dimensionless_score_df = dimensionless_score_df.replace(
            [np.inf, -np.inf],
            np.nan,
        )
        selected_score_df = (
            source_score_df
            if self.config_obj.score_mode_str == SOURCE_ATR_SCORE_STR
            else dimensionless_score_df
        )

        feature_frame_list: list[pd.DataFrame] = []
        feature_map_dict = {
            f"monthly_roc_{self.lookback_month_int}_ser": monthly_roc_df,
            f"atr_{ATR_WINDOW_INT}_ser": atr_decision_df,
            "stock_trend_pass_bool": stock_trend_pass_df,
            "risk_adj_score_ser": selected_score_df,
        }
        for field_str, feature_df in feature_map_dict.items():
            aligned_feature_df = feature_df.reindex(signal_data_df.index).copy()
            aligned_feature_df.columns = pd.MultiIndex.from_tuples(
                [
                    (symbol_str, field_str)
                    for symbol_str in aligned_feature_df.columns.astype(str)
                ]
            )
            feature_frame_list.append(aligned_feature_df)

        regime_feature_df = pd.DataFrame(
            {
                (
                    self.regime_symbol_str,
                    f"regime_sma_{self.index_trend_window_int}_ser",
                ): regime_sma_ser.reindex(signal_data_df.index),
                (
                    self.regime_symbol_str,
                    "regime_pass_bool",
                ): regime_pass_ser.reindex(signal_data_df.index),
            },
            index=signal_data_df.index,
        )
        regime_feature_df.columns = pd.MultiIndex.from_tuples(
            regime_feature_df.columns
        )
        return pd.concat(
            [signal_data_df] + feature_frame_list + [regime_feature_df],
            axis=1,
        )

    def get_ranked_candidate_feature_df(
        self,
        close_row_ser: pd.Series,
    ) -> pd.DataFrame:
        if self.universe_df is None:
            raise RuntimeError("universe_df must be set before monthly rebalances.")
        candidate_feature_df = close_row_ser.unstack()
        empty_candidate_df = pd.DataFrame(
            columns=["risk_adj_score_float", "symbol_str"]
        )
        if self.config_obj.apply_market_trend_bool:
            regime_pass_value = candidate_feature_df.loc[
                self.regime_symbol_str
            ].get("regime_pass_bool", np.nan)
            if pd.isna(regime_pass_value) or not bool(regime_pass_value):
                return empty_candidate_df

        universe_member_ser = get_asof_universe_membership_ser(
            universe_df=self.universe_df,
            decision_date_ts=pd.Timestamp(self.previous_bar),
        )
        active_symbol_list = universe_member_ser[
            universe_member_ser == 1
        ].index.astype(str)
        candidate_feature_df = candidate_feature_df.loc[
            candidate_feature_df.index.intersection(active_symbol_list)
        ].copy()
        if "risk_adj_score_ser" not in candidate_feature_df.columns:
            return empty_candidate_df

        candidate_feature_df = candidate_feature_df.assign(
            risk_adj_score_float=pd.to_numeric(
                candidate_feature_df["risk_adj_score_ser"],
                errors="coerce",
            ),
            symbol_str=candidate_feature_df.index.astype(str),
        )
        finite_score_bool_vec = np.isfinite(
            candidate_feature_df["risk_adj_score_float"].to_numpy(dtype=float)
        )
        eligible_bool_vec = finite_score_bool_vec
        if self.config_obj.apply_asset_trend_bool:
            if "stock_trend_pass_bool" not in candidate_feature_df.columns:
                return empty_candidate_df
            asset_trend_bool_vec = (
                candidate_feature_df["stock_trend_pass_bool"]
                .where(
                    candidate_feature_df["stock_trend_pass_bool"].notna(),
                    False,
                )
                .astype(bool)
                .to_numpy()
            )
            eligible_bool_vec = eligible_bool_vec & asset_trend_bool_vec
        candidate_feature_df = candidate_feature_df.loc[eligible_bool_vec]
        return candidate_feature_df.sort_values(
            by=["risk_adj_score_float", "symbol_str"],
            ascending=[False, True],
            kind="mergesort",
        )

    def get_target_weight_ser(self, close_row_ser: pd.Series) -> pd.Series:
        ranked_candidate_df = self.get_ranked_candidate_feature_df(
            close_row_ser=close_row_ser
        )
        if len(ranked_candidate_df) == 0:
            return pd.Series(dtype=float)
        selected_symbol_list = ranked_candidate_df.index[
            : self.max_positions_int
        ].astype(str)
        target_weight_ser = pd.Series(
            1.0 / float(self.max_positions_int),
            index=selected_symbol_list,
            dtype=float,
        )
        if self.config_obj.use_vix_scale_bool:
            exposure_scale_float = get_asof_vix_scale_float(
                vix_scale_signal_df=self.vix_scale_signal_df,
                decision_date_ts=pd.Timestamp(self.previous_bar),
            )
        else:
            exposure_scale_float = float(
                self.config_obj.static_exposure_scale_float
            )
        target_weight_ser = target_weight_ser * exposure_scale_float
        return target_weight_ser


def build_strategy(
    config_obj: AtrNormalizedSectorConfig,
    rebalance_schedule_df: pd.DataFrame,
    vix_scale_signal_df: pd.DataFrame,
    name_str: str,
) -> AtrNormalizedSectorStrategy:
    strategy_obj = AtrNormalizedSectorStrategy(
        name=name_str,
        benchmarks=[config_obj.performance_benchmark_symbol_str],
        rebalance_schedule_df=rebalance_schedule_df,
        vix_scale_signal_df=vix_scale_signal_df,
        config_obj=config_obj,
    )
    configure_total_return_benchmark_provenance(
        strategy_obj=strategy_obj,
        config_obj=config_obj,
    )
    return strategy_obj


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
) -> AtrNormalizedSectorStrategy:
    config_obj = replace(
        DEFAULT_CONFIG,
        backtest_start_date_str=(
            DEFAULT_CONFIG.backtest_start_date_str
            if backtest_start_date_str is None
            else backtest_start_date_str
        ),
        capital_base_float=(
            DEFAULT_CONFIG.capital_base_float
            if capital_base_float is None
            else float(capital_base_float)
        ),
        end_date_str=end_date_str,
    )
    (
        pricing_data_df,
        universe_df,
        rebalance_schedule_df,
        vix_scale_signal_df,
    ) = get_atr_normalized_sector_data(
        config_obj=config_obj,
        include_total_return_benchmark_bool=True,
    )
    strategy_obj = build_strategy(
        config_obj=config_obj,
        rebalance_schedule_df=rebalance_schedule_df,
        vix_scale_signal_df=vix_scale_signal_df,
        name_str="strategy_mo_atr_normalized_sector_vox_iyr",
    )
    strategy_obj.universe_df = universe_df
    # *** CRITICAL*** Keep pre-start history for ROC12, ATR20, SMA100, and
    # SMA200, but execute only from the requested calendar at Open_T+1.
    calendar_idx = pricing_data_df.index[
        pricing_data_df.index >= pd.Timestamp(config_obj.backtest_start_date_str)
    ]
    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar=calendar_idx,
        show_progress=show_display_bool,
        show_signal_progress_bool=show_display_bool,
        audit_override_bool=None,
    )
    if show_display_bool:
        display(strategy_obj.summary)
        display(strategy_obj.summary_trades)
    if save_results_bool:
        save_results(strategy_obj, output_dir=output_dir_str)
    return strategy_obj


if __name__ == "__main__":
    run_variant()
