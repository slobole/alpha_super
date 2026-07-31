"""
Monthly ATR-adjusted Nasdaq-100 momentum rotation with a correlation-penalized
greedy selection.

Motivation
----------
The base model picks the top-N names by risk-adjusted momentum. When one theme
(e.g. semiconductors) dominates the momentum ranks, all N slots fill with the
same trade and the basket's effective diversification collapses. This variant
keeps every gate and the execution mapping of the base model, and only changes
*which* of the eligible candidates fill the N slots: candidates highly
correlated to the names already selected are penalized, so the greedy pass
prefers diversifiers over near-duplicates.

Core formulas
-------------
For eligible stock i on month-end decision date t (after the base regime,
universe, and stock-trend gates):

    score_{i,t}
        = monthly_roc_{i,t}^{(L)} / ATR20_{i,t}          (unchanged base score)

    corr_{i,j,t}
        = Pearson correlation of daily close-to-close returns of i and j
          over the trailing corr_window days ending at decision close t

Greedy selection (slots k = 1..max_positions):

    slot 1:
        pick argmax_i score_{i,t}

    slot k > 1, with S = already-selected set:
        avg_corr_{i,t} = mean_{j in S} corr_{i,j,t}
        adjusted_score_{i,t} = score_{i,t} - lambda * avg_corr_{i,t} * |score_{i,t}|
        pick argmax_i adjusted_score_{i,t}   (ties broken by symbol ascending)

    target_weight_{i,t}
        = 1 / max_positions   if i selected
        = 0                   otherwise

The sign-safe penalty (subtracting lambda * avg_corr * |score| instead of
multiplying by (1 - lambda * avg_corr)) guarantees that higher correlation
always ranks a candidate lower, for negative scores too. lambda = 0 reproduces
the base top-N selection exactly.

Missing correlations (e.g. a recent IPO without corr_min_overlap overlapping
return days) fall back to the median of the valid pairwise correlations among
the day's candidates — a deliberately conservative choice so short-history
names get no free diversification credit.

Execution mapping is unchanged:

    decision_date_t = actual last tradable close of month t
    execution_date_t = next tradable open after decision_date_t
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from strategies.momentum.strategy_mo_atr_normalized_ndx import (
    AtrNormalizedNdxConfig,
    AtrNormalizedNdxStrategy,
    configure_total_return_benchmark_provenance,
    get_atr_normalized_ndx_data,
)


@dataclass(frozen=True)
class CorrPenaltyAtrNormalizedNdxConfig(AtrNormalizedNdxConfig):
    corr_window_int: int = 126
    corr_min_overlap_int: int = 63
    corr_penalty_lambda_float: float = 0.5
    # 0.0 disables the liquidity gate (legacy behavior). When positive, a
    # candidate is eligible only if its trailing median dollar ADV on the
    # decision close is at least this many dollars.
    min_dollar_adv_float: float = 0.0
    adv_window_int: int = 20

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.corr_window_int <= 1:
            raise ValueError("corr_window_int must be greater than 1.")
        if self.corr_min_overlap_int <= 1:
            raise ValueError("corr_min_overlap_int must be greater than 1.")
        if self.corr_min_overlap_int > self.corr_window_int:
            raise ValueError("corr_min_overlap_int must be <= corr_window_int.")
        if self.corr_penalty_lambda_float < 0.0:
            raise ValueError("corr_penalty_lambda_float must be non-negative.")
        if self.min_dollar_adv_float < 0.0:
            raise ValueError("min_dollar_adv_float must be non-negative.")
        if self.adv_window_int <= 1:
            raise ValueError("adv_window_int must be greater than 1.")


DEFAULT_CONFIG = CorrPenaltyAtrNormalizedNdxConfig()

__all__ = [
    "CorrPenaltyAtrNormalizedNdxConfig",
    "CorrPenaltyAtrNormalizedNdxStrategy",
    "DEFAULT_CONFIG",
    "run_variant",
    "select_corr_penalized_symbol_list",
]


def select_corr_penalized_symbol_list(
    candidate_score_ser: pd.Series,
    candidate_corr_df: pd.DataFrame,
    max_positions_int: int,
    corr_penalty_lambda_float: float,
) -> list[str]:
    """
    Greedy correlation-penalized selection over ranked candidates.

    candidate_score_ser: risk-adjusted score per candidate symbol.
    candidate_corr_df: pairwise correlation among the same symbols; NaN entries
    fall back to the median valid off-diagonal correlation (0.0 if none exist).
    """
    if max_positions_int <= 0:
        raise ValueError("max_positions_int must be positive.")
    if corr_penalty_lambda_float < 0.0:
        raise ValueError("corr_penalty_lambda_float must be non-negative.")
    if len(candidate_score_ser) == 0:
        return []
    if candidate_score_ser.isna().any():
        raise ValueError("candidate_score_ser must not contain NaN scores.")
    if candidate_score_ser.index.has_duplicates:
        raise ValueError("candidate_score_ser index must not contain duplicates.")

    symbol_list = candidate_score_ser.index.astype(str).tolist()
    missing_symbol_list = [
        symbol_str
        for symbol_str in symbol_list
        if symbol_str not in candidate_corr_df.index or symbol_str not in candidate_corr_df.columns
    ]
    if len(missing_symbol_list) > 0:
        raise ValueError(f"candidate_corr_df is missing candidate symbols: {missing_symbol_list[:5]}")

    aligned_corr_df = candidate_corr_df.loc[symbol_list, symbol_list].astype(float)

    # *** CRITICAL*** NaN pairwise correlations (short-history names) fall back
    # to the median valid off-diagonal correlation of the day's candidates so a
    # recent IPO cannot earn diversification credit just by lacking history.
    off_diagonal_mask_arr = ~np.eye(len(symbol_list), dtype=bool)
    valid_corr_value_arr = aligned_corr_df.to_numpy()[off_diagonal_mask_arr]
    valid_corr_value_arr = valid_corr_value_arr[np.isfinite(valid_corr_value_arr)]
    fallback_corr_float = float(np.median(valid_corr_value_arr)) if len(valid_corr_value_arr) > 0 else 0.0
    filled_corr_df = aligned_corr_df.fillna(fallback_corr_float)

    remaining_score_ser = candidate_score_ser.astype(float).copy()
    remaining_score_ser.index = remaining_score_ser.index.astype(str)
    selected_symbol_list: list[str] = []

    while len(selected_symbol_list) < max_positions_int and len(remaining_score_ser) > 0:
        if len(selected_symbol_list) == 0:
            adjusted_score_ser = remaining_score_ser
        else:
            avg_corr_ser = filled_corr_df.loc[remaining_score_ser.index, selected_symbol_list].mean(axis=1)
            # Sign-safe penalty: higher correlation always ranks lower, even
            # for negative scores.
            adjusted_score_ser = (
                remaining_score_ser
                - float(corr_penalty_lambda_float) * avg_corr_ser * remaining_score_ser.abs()
            )

        ranked_df = pd.DataFrame(
            {
                "adjusted_score_float": adjusted_score_ser,
                "symbol_str": adjusted_score_ser.index,
            }
        ).sort_values(
            by=["adjusted_score_float", "symbol_str"],
            ascending=[False, True],
            kind="mergesort",
        )
        picked_symbol_str = str(ranked_df.index[0])
        selected_symbol_list.append(picked_symbol_str)
        remaining_score_ser = remaining_score_ser.drop(index=picked_symbol_str)

    return selected_symbol_list


class CorrPenaltyAtrNormalizedNdxStrategy(AtrNormalizedNdxStrategy):
    """
    ATR-normalized NDX momentum with correlation-penalized greedy selection.

    For selected stock i at rebalance open t (unchanged from base):

        q^{intent}_{i,t}
            = floor(V_{t-1} * (1 / max_positions) / Close_{i,t-1})
    """

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        rebalance_schedule_df: pd.DataFrame,
        regime_symbol_str: str = "SPY",
        capital_base: float = 100_000.0,
        slippage: float = 0.00025,
        commission_per_share: float = 0.005,
        commission_minimum: float = 1.0,
        lookback_month_int: int = 12,
        index_trend_window_int: int = 200,
        stock_trend_window_int: int = 100,
        max_positions_int: int = 10,
        corr_window_int: int = 126,
        corr_min_overlap_int: int = 63,
        corr_penalty_lambda_float: float = 0.5,
        min_dollar_adv_float: float = 0.0,
        adv_window_int: int = 20,
    ):
        super().__init__(
            name=name,
            benchmarks=benchmarks,
            rebalance_schedule_df=rebalance_schedule_df,
            regime_symbol_str=regime_symbol_str,
            capital_base=capital_base,
            slippage=slippage,
            commission_per_share=commission_per_share,
            commission_minimum=commission_minimum,
            lookback_month_int=lookback_month_int,
            index_trend_window_int=index_trend_window_int,
            stock_trend_window_int=stock_trend_window_int,
            max_positions_int=max_positions_int,
        )
        if corr_window_int <= 1:
            raise ValueError("corr_window_int must be greater than 1.")
        if corr_min_overlap_int <= 1:
            raise ValueError("corr_min_overlap_int must be greater than 1.")
        if corr_min_overlap_int > corr_window_int:
            raise ValueError("corr_min_overlap_int must be <= corr_window_int.")
        if corr_penalty_lambda_float < 0.0:
            raise ValueError("corr_penalty_lambda_float must be non-negative.")
        if min_dollar_adv_float < 0.0:
            raise ValueError("min_dollar_adv_float must be non-negative.")
        if adv_window_int <= 1:
            raise ValueError("adv_window_int must be greater than 1.")

        self.corr_window_int = int(corr_window_int)
        self.corr_min_overlap_int = int(corr_min_overlap_int)
        self.corr_penalty_lambda_float = float(corr_penalty_lambda_float)
        self.min_dollar_adv_float = float(min_dollar_adv_float)
        self.adv_window_int = int(adv_window_int)
        self.price_return_df: pd.DataFrame | None = None
        self.dollar_adv_df: pd.DataFrame | None = None
        self.selection_audit_row_list: list[dict[str, object]] = []

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        signal_data_df = super().compute_signals(pricing_data)

        tradeable_symbol_list = self.get_tradeable_symbol_list(pricing_data)
        price_close_df = pd.DataFrame(
            {symbol_str: pricing_data[(symbol_str, "Close")] for symbol_str in tradeable_symbol_list},
            index=pricing_data.index,
        ).astype(float)
        # *** CRITICAL*** return_t = Close_t / Close_{t-1} - 1 uses only the
        # current and prior close; the trailing correlation window is later
        # restricted to rows on or before the decision close. fill_method=None
        # keeps missing closes as NaN instead of padding them into synthetic
        # zero returns that would deflate correlations for gappy symbols.
        self.price_return_df = price_close_df.pct_change(fill_method=None)

        if self.min_dollar_adv_float > 0.0:
            dollar_volume_frame_map: dict[str, pd.Series] = {}
            for symbol_str in tradeable_symbol_list:
                volume_key = (symbol_str, "Volume")
                unadjusted_close_key = (symbol_str, "Unadjusted Close")
                if volume_key not in pricing_data.columns or unadjusted_close_key not in pricing_data.columns:
                    # Missing liquidity fields -> ADV stays NaN -> candidate is
                    # never eligible. Conservative by construction.
                    dollar_volume_frame_map[symbol_str] = pd.Series(
                        np.nan, index=pricing_data.index
                    )
                    continue
                dollar_volume_frame_map[symbol_str] = (
                    pricing_data[volume_key].astype(float)
                    * pricing_data[unadjusted_close_key].astype(float)
                )
            dollar_volume_df = pd.DataFrame(dollar_volume_frame_map, index=pricing_data.index)
            # *** CRITICAL*** The ADV gate is a trailing rolling median of past
            # dollar volume only. min_periods equals the full window, so names
            # with under adv_window trading days (fresh IPOs) are ineligible.
            self.dollar_adv_df = dollar_volume_df.rolling(
                window=self.adv_window_int,
                min_periods=self.adv_window_int,
            ).median()
        return signal_data_df

    def get_asof_candidate_corr_df(self, candidate_symbol_list: list[str]) -> pd.DataFrame:
        if self.price_return_df is None:
            raise RuntimeError("price_return_df must be computed before monthly rebalances.")

        # *** CRITICAL*** Correlations use only returns observed on or before
        # the month-end decision close (previous_bar). No same-execution-bar or
        # future return may enter this window.
        asof_return_df = self.price_return_df.loc[: pd.Timestamp(self.previous_bar)]
        window_return_df = asof_return_df.tail(self.corr_window_int)
        return window_return_df.loc[:, candidate_symbol_list].corr(
            min_periods=self.corr_min_overlap_int
        )

    def get_asof_liquid_symbol_set(self, candidate_symbol_list: list[str]) -> set[str]:
        """
        Return the candidates whose trailing dollar ADV passes the gate.
        """
        if self.min_dollar_adv_float <= 0.0:
            return set(candidate_symbol_list)
        if self.dollar_adv_df is None:
            raise RuntimeError("dollar_adv_df must be computed before monthly rebalances.")

        # *** CRITICAL*** As-of lookup: use only the latest ADV row on or
        # before the decision close. NaN ADV (missing data or short history)
        # fails the gate.
        asof_adv_df = self.dollar_adv_df.loc[: pd.Timestamp(self.previous_bar)]
        if len(asof_adv_df) == 0:
            return set()
        candidate_adv_ser = asof_adv_df.iloc[-1].reindex(candidate_symbol_list)
        liquid_mask_ser = candidate_adv_ser >= self.min_dollar_adv_float
        return set(liquid_mask_ser.index[liquid_mask_ser.fillna(False)].astype(str))

    def get_target_weight_ser(self, close_row_ser: pd.Series) -> pd.Series:
        ranked_candidate_feature_df = self.get_ranked_candidate_feature_df(close_row_ser=close_row_ser)
        if len(ranked_candidate_feature_df) == 0:
            return pd.Series(dtype=float)

        all_candidate_symbol_list = ranked_candidate_feature_df.index.astype(str).tolist()
        liquid_symbol_set = self.get_asof_liquid_symbol_set(all_candidate_symbol_list)
        adv_excluded_count_int = len(all_candidate_symbol_list) - len(liquid_symbol_set)
        ranked_candidate_feature_df = ranked_candidate_feature_df[
            ranked_candidate_feature_df.index.astype(str).isin(liquid_symbol_set)
        ]
        if len(ranked_candidate_feature_df) == 0:
            return pd.Series(dtype=float)

        candidate_symbol_list = ranked_candidate_feature_df.index.astype(str).tolist()
        candidate_corr_df = self.get_asof_candidate_corr_df(candidate_symbol_list)
        selected_symbol_list = select_corr_penalized_symbol_list(
            candidate_score_ser=ranked_candidate_feature_df["risk_adj_score_float"],
            candidate_corr_df=candidate_corr_df,
            max_positions_int=self.max_positions_int,
            corr_penalty_lambda_float=self.corr_penalty_lambda_float,
        )

        selected_pair_corr_df = candidate_corr_df.loc[selected_symbol_list, selected_symbol_list]
        off_diagonal_mask_arr = ~np.eye(len(selected_symbol_list), dtype=bool)
        selected_pair_value_arr = selected_pair_corr_df.to_numpy()[off_diagonal_mask_arr]
        selected_pair_value_arr = selected_pair_value_arr[np.isfinite(selected_pair_value_arr)]
        self.selection_audit_row_list.append(
            {
                "decision_date_ts": pd.Timestamp(self.previous_bar),
                "candidate_count_int": int(len(candidate_symbol_list)),
                "adv_excluded_count_int": int(adv_excluded_count_int),
                "selected_symbol_list": list(selected_symbol_list),
                "avg_selected_pairwise_corr_float": (
                    float(np.mean(selected_pair_value_arr)) if len(selected_pair_value_arr) > 0 else np.nan
                ),
            }
        )

        target_weight_float = 1.0 / float(self.max_positions_int)
        return pd.Series(target_weight_float, index=selected_symbol_list, dtype=float)

    def get_selection_audit_df(self) -> pd.DataFrame:
        if len(self.selection_audit_row_list) == 0:
            return pd.DataFrame(
                columns=[
                    "decision_date_ts",
                    "candidate_count_int",
                    "adv_excluded_count_int",
                    "selected_symbol_list",
                    "avg_selected_pairwise_corr_float",
                ]
            )
        return pd.DataFrame(self.selection_audit_row_list).set_index("decision_date_ts")


def build_corr_penalty_strategy(
    config: CorrPenaltyAtrNormalizedNdxConfig,
    rebalance_schedule_df: pd.DataFrame,
    name_str: str = "strategy_mo_atr_normalized_ndx_corr_penalty",
) -> CorrPenaltyAtrNormalizedNdxStrategy:
    return CorrPenaltyAtrNormalizedNdxStrategy(
        name=name_str,
        benchmarks=[config.performance_benchmark_symbol_str],
        rebalance_schedule_df=rebalance_schedule_df,
        regime_symbol_str=config.regime_symbol_str,
        capital_base=config.capital_base_float,
        slippage=config.slippage_float,
        commission_per_share=config.commission_per_share_float,
        commission_minimum=config.commission_minimum_float,
        lookback_month_int=config.lookback_month_int,
        index_trend_window_int=config.index_trend_window_int,
        stock_trend_window_int=config.stock_trend_window_int,
        max_positions_int=config.max_positions_int,
        corr_window_int=config.corr_window_int,
        corr_min_overlap_int=config.corr_min_overlap_int,
        corr_penalty_lambda_float=config.corr_penalty_lambda_float,
        min_dollar_adv_float=config.min_dollar_adv_float,
        adv_window_int=config.adv_window_int,
    )


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
) -> CorrPenaltyAtrNormalizedNdxStrategy:
    config_obj = DEFAULT_CONFIG
    if (
        backtest_start_date_str is not None
        or capital_base_float is not None
        or end_date_str is not None
    ):
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
    pricing_data_df, universe_df, rebalance_schedule_df = get_atr_normalized_ndx_data(
        config_obj,
        include_total_return_benchmark_bool=True,
    )

    strategy_obj = build_corr_penalty_strategy(
        config=config_obj,
        rebalance_schedule_df=rebalance_schedule_df,
    )
    strategy_obj.universe_df = universe_df
    configure_total_return_benchmark_provenance(
        strategy_obj=strategy_obj,
        config_obj=config_obj,
    )

    # *** CRITICAL*** Deployment-reference backtests keep full pre-start
    # history for monthly ATR, trend, and correlation features, but the
    # executable calendar starts at the first deployment fill session.
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
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        display(strategy_obj.summary)
        display(strategy_obj.summary_trades)

    if save_results_bool:
        save_results(strategy_obj, output_dir=output_dir_str)

    return strategy_obj


if __name__ == "__main__":
    run_variant()
