"""
SPHB/SPLV 2-day canary with TLT as the risk-off sleeve.

This preserves the base canary signal:

    risk_on_t = 1[momentum_t > 0 and momentum_{t-1} > 0]

and changes only the allocation target:

    if risk_on_t:
        hold SPY
    else:
        hold TLT
"""

from __future__ import annotations

from strategies.momentum.strategy_mo_spy_sphb_splv_canary import (
    SphbSplvCanaryConfig,
    SphbSplvCanaryStrategy,
    run_variant as run_base_variant,
)


DEFAULT_CONFIG = SphbSplvCanaryConfig(risk_off_symbol_str="TLT")


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    capital_base_float: float = DEFAULT_CONFIG.capital_base_float,
    end_date_str: str | None = None,
) -> SphbSplvCanaryStrategy:
    return run_base_variant(
        show_display_bool=show_display_bool,
        save_results_bool=save_results_bool,
        output_dir_str=output_dir_str,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
        risk_off_symbol_str=DEFAULT_CONFIG.risk_off_symbol_str,
        strategy_name_str="strategy_mo_spy_sphb_splv_canary_tlt",
    )


if __name__ == "__main__":
    run_variant()
