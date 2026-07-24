"""Short-only Paper-B variant of VXN-scaled ATR-normalized NDX momentum.

For stock i on month-end decision date t:

    classic_momentum_{i,t} = Close_ME_{i,t-1} / Close_ME_{i,t-12} - 1
    last_month_return_{i,t} = Close_ME_{i,t} / Close_ME_{i,t-1} - 1
    paper_b_{i,t} = (1 + last_month_return_{i,t}) * classic_momentum_{i,t}
    risk_adj_score_{i,t} = paper_b_{i,t} / ATR20_{i,t}

The strategy is active only when SPY is not above SMA200. It shorts the bottom
10 PIT Nasdaq-100 members whose own close is not above SMA100, at -10% base weight
each. The existing causal VXN multiplier scales the total short exposure.
Decision signals use the completed month-end close and execute at the next open.

Borrow availability, borrow fees, recalls, and squeeze-specific execution are
not modeled. This is a research-only result and its short returns are optimistic.
"""

from __future__ import annotations

import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from strategies.momentum.strategy_mo_atr_normalized_ndx_short import (
    AtrNormalizedNdxShortStrategy,
)
from strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled import (
    get_asof_vxn_scale_float,
)
from strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled_roc_variants import (
    DEFAULT_ATR_WINDOW_INT,
    ROC_MODE_PAPER_B_STR,
    VxnScaledAtrNormalizedNdxRocVariantStrategy,
    build_roc_variant_config,
    get_vxn_scaled_atr_normalized_ndx_roc_variant_data,
)


class VxnScaledAtrNormalizedNdxPaperBShortStrategy(
    VxnScaledAtrNormalizedNdxRocVariantStrategy
):
    """Paper-B signal construction with the canonical NDX short book."""

    def get_target_weight_ser(self, close_row_ser: pd.Series) -> pd.Series:
        base_target_weight_ser = AtrNormalizedNdxShortStrategy.get_target_weight_ser(
            self,
            close_row_ser=close_row_ser,
        )
        if len(base_target_weight_ser) == 0:
            return base_target_weight_ser

        exposure_scale_float = get_asof_vxn_scale_float(
            vxn_scale_signal_df=self.vxn_scale_signal_df,
            decision_date_ts=pd.Timestamp(self.previous_bar),
        )
        return base_target_weight_ser * exposure_scale_float

    def get_target_share_int_map(
        self,
        target_weight_ser: pd.Series,
        close_row_ser: pd.Series,
    ) -> dict[str, int]:
        return AtrNormalizedNdxShortStrategy.get_target_share_int_map(
            self,
            target_weight_ser=target_weight_ser,
            close_row_ser=close_row_ser,
        )

    def iterate(
        self,
        data_df: pd.DataFrame,
        close_row_ser: pd.Series,
        open_price_ser: pd.Series,
    ):
        # *** CRITICAL*** Reuse the canonical short execution path so the
        # completed month-end signal at T still fills only at the T+1 open.
        return AtrNormalizedNdxShortStrategy.iterate(
            self,
            data_df=data_df,
            close_row_ser=close_row_ser,
            open_price_ser=open_price_ser,
        )


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
) -> VxnScaledAtrNormalizedNdxPaperBShortStrategy:
    """Run the research-only Paper-B short strategy through Vanilla."""
    config_obj = build_roc_variant_config(
        roc_mode_str=ROC_MODE_PAPER_B_STR,
        atr_window_int=DEFAULT_ATR_WINDOW_INT,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
    )
    pricing_data_df, universe_df, rebalance_schedule_df, vxn_scale_signal_df = (
        get_vxn_scaled_atr_normalized_ndx_roc_variant_data(config_obj)
    )

    strategy_obj = VxnScaledAtrNormalizedNdxPaperBShortStrategy(
        name="strategy_mo_atr_normalized_ndx_vxn_scaled_paper_b_short",
        benchmarks=[config_obj.regime_symbol_str],
        rebalance_schedule_df=rebalance_schedule_df,
        vxn_scale_signal_df=vxn_scale_signal_df,
        roc_mode_str=ROC_MODE_PAPER_B_STR,
        regime_symbol_str=config_obj.regime_symbol_str,
        capital_base=config_obj.capital_base_float,
        slippage=config_obj.slippage_float,
        commission_per_share=config_obj.commission_per_share_float,
        commission_minimum=config_obj.commission_minimum_float,
        lookback_month_int=config_obj.lookback_month_int,
        index_trend_window_int=config_obj.index_trend_window_int,
        stock_trend_window_int=config_obj.stock_trend_window_int,
        max_positions_int=config_obj.max_positions_int,
        atr_window_int=config_obj.atr_window_int,
    )
    strategy_obj.universe_df = universe_df

    # *** CRITICAL*** Full pre-start history is retained for causal trailing
    # signals; only execution begins at the configured comparison start date.
    calendar_idx = pricing_data_df.index[
        pricing_data_df.index >= pd.Timestamp(config_obj.backtest_start_date_str)
    ]
    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar=calendar_idx,
        show_progress=show_display_bool,
        show_signal_progress_bool=show_display_bool,
        audit_override_bool=False,
    )

    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        display(strategy_obj.summary)
        display(strategy_obj.summary_trades)

    if save_results_bool:
        save_results(strategy_obj, output_dir=output_dir_str)

    return strategy_obj


__all__ = [
    "ROC_MODE_PAPER_B_STR",
    "VxnScaledAtrNormalizedNdxPaperBShortStrategy",
    "run_variant",
]


if __name__ == "__main__":
    run_variant()
