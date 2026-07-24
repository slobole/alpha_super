"""Paper-B numerator variant of VXN-scaled ATR-normalized NDX momentum.

This research-only wrapper changes one baseline rule:

    classic_momentum_t = Close_ME_(t-1) / Close_ME_(t-12) - 1
    last_month_return_t = Close_ME_t / Close_ME_(t-1) - 1
    paper_b_t = (1 + last_month_return_t) * classic_momentum_t
    risk_adj_score_t = paper_b_t / ATR20_t

PIT Nasdaq-100 membership, SPY/SMA200 regime, stock/SMA100 filter, top-10
equal-weight selection, VXN scaling, costs, and next-open execution are inherited
unchanged from the baseline research path.
"""

from __future__ import annotations

from strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled_roc_variants import (
    DEFAULT_ATR_WINDOW_INT,
    ROC_MODE_PAPER_B_STR,
    VxnScaledAtrNormalizedNdxRocVariantStrategy,
    run_variant as run_roc_variant,
)


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
) -> VxnScaledAtrNormalizedNdxRocVariantStrategy:
    """Run the controlled Paper-B numerator variant through Vanilla."""
    return run_roc_variant(
        roc_mode_str=ROC_MODE_PAPER_B_STR,
        atr_window_int=DEFAULT_ATR_WINDOW_INT,
        show_display_bool=show_display_bool,
        save_results_bool=save_results_bool,
        output_dir_str=output_dir_str,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
    )


__all__ = [
    "ROC_MODE_PAPER_B_STR",
    "run_variant",
]


if __name__ == "__main__":
    run_variant()
