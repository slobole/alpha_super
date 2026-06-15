"""
Russell 1000 ROC12/ATR20 momentum with $RUI regime and RP-63 sizing.

This is a research-only Bench wrapper around
strategy_mo_atr_normalized_index_vix_scaled. It pins the best tested row:

    score_{i,t} = ROC12_{i,t} / ATR20_{i,t}

    weight_{i,t}
        = (1 / vol63_{i,t}) / sum_j(1 / vol63_{j,t})

where the inverse-volatility normalization is applied only across the selected
positions. There is no VIX or VXN total-exposure scaler in this wrapper.

Timing is inherited from the base monthly ATR-normalized momentum model:
decision at the actual last tradable close of the month, execution at the next
tradable open under the Vanilla engine contract.
"""

from __future__ import annotations

from dataclasses import replace
from typing import cast

from strategies.momentum.strategy_mo_atr_normalized_index_vix_scaled import (
    RUSSELL1000_CONFIG,
    RiskParity63AtrNormalizedIndexStrategy,
    SELECTION_SCORE_MODE_ATR20_STR,
    run_variant as run_base_variant,
)


STRATEGY_NAME_STR = "strategy_mo_atr_normalized_russell1000_rui_rp63"

# The base builder derives the final risk-parity strategy name by replacing the
# "_vix_scaled" suffix with "_risk_parity_63".
DEFAULT_CONFIG = replace(
    RUSSELL1000_CONFIG,
    regime_symbol_str="$RUI",
    strategy_name_str="strategy_mo_atr_normalized_russell1000_rui_vix_scaled",
    selection_score_mode_str=SELECTION_SCORE_MODE_ATR20_STR,
    inverse_vol_window_int=63,
)


__all__ = [
    "DEFAULT_CONFIG",
    "STRATEGY_NAME_STR",
    "run_variant",
]


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
) -> RiskParity63AtrNormalizedIndexStrategy:
    strategy_obj = run_base_variant(
        config=DEFAULT_CONFIG,
        risk_parity_63_bool=True,
        show_display_bool=show_display_bool,
        save_results_bool=save_results_bool,
        output_dir_str=output_dir_str,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
    )
    return cast(RiskParity63AtrNormalizedIndexStrategy, strategy_obj)


if __name__ == "__main__":
    run_variant()
