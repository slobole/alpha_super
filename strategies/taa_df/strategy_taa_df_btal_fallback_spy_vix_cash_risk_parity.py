"""
Defense First BTAL fallback-SPY VIX-cash variant with inverse-volatility
Risk Parity sizing.

Base behavior preserved:
    - defensive sleeve: GLD, UUP, TLT, DBC, BTAL
    - fallback asset: SPY
    - month-end momentum and cash-hurdle signal
    - VIX/RV cash gate on the fallback sleeve
    - first tradable open of next month execution

Sizing overlay:
    active tradeable weights are resized by 63-day inverse volatility.
"""

from __future__ import annotations

from dataclasses import replace

try:
    from strategies.taa_df.strategy_taa_df import get_defense_first_data
    from strategies.taa_df.strategy_taa_df_btal_fallback_spy_vix_cash import DEFAULT_CONFIG as BASE_CONFIG
    from strategies.taa_df.strategy_taa_df_fallback_risk_parity_variant_utils import (
        risk_parity_lookback_day_int,
        run_standard_fallback_vix_cash_risk_parity_variant,
    )
except ModuleNotFoundError:
    from strategy_taa_df import get_defense_first_data
    from strategy_taa_df_btal_fallback_spy_vix_cash import DEFAULT_CONFIG as BASE_CONFIG
    from strategy_taa_df_fallback_risk_parity_variant_utils import (
        risk_parity_lookback_day_int,
        run_standard_fallback_vix_cash_risk_parity_variant,
    )


STRATEGY_NAME_STR = "strategy_taa_df_btal_fallback_spy_vix_cash_risk_parity"
DEFAULT_CONFIG = BASE_CONFIG


def run_variant(
    risk_parity_lookback_day_int: int = risk_parity_lookback_day_int,
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
):
    config = DEFAULT_CONFIG if end_date_str is None else replace(DEFAULT_CONFIG, end_date_str=end_date_str)
    return run_standard_fallback_vix_cash_risk_parity_variant(
        strategy_name_str=STRATEGY_NAME_STR,
        config=config,
        base_data_loader_fn=get_defense_first_data,
        lookback_day_int=risk_parity_lookback_day_int,
        show_display_bool=show_display_bool,
        save_results_bool=save_results_bool,
        output_dir_str=output_dir_str,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
    )


if __name__ == "__main__":
    run_variant()
