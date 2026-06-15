"""
Beyond 60/40 inverse-volatility strategy with UUP, DBC, and an 8% volatility target.

Tradeable assets:

    VTI, GLD, TLT, UUP, DBC

Base weights:

    w_{i,t}^{base}
        = (1 / sigma_{i,t}^{(63)}) / sum_j(1 / sigma_{j,t}^{(63)})

Daily exposure overlay:

    exposure_t
        = 1                                      if sigma_{portfolio,t}^{(63)} <= 0.085
        = min(1, 0.08 / sigma_{portfolio,t}^{(63)}) otherwise

Final weights:

    w_{i,t}^{final}
        = exposure_t * w_{i,t}^{base}

    w_t^{cash}
        = 1 - exposure_t

There is no leverage. If trailing realized portfolio volatility is below the
trigger, the strategy stays fully invested. Exposure is allowed to update daily.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from strategies.taa_beyond_6040.strategy_taa_beyond_6040 import (
    Beyond6040Strategy,
    DEFAULT_CONFIG as BASE_DEFAULT_CONFIG,
    get_beyond_6040_data,
    get_first_actionable_rebalance_ts,
)


STRATEGY_NAME_STR = "strategy_taa_beyond_6040_inverse_vol_cap_8_uup_dbc"
ASSET_TUPLE = ("VTI", "GLD", "TLT", "UUP", "DBC")
TARGET_PORTFOLIO_VOL_FLOAT = 0.08
TRIGGER_PORTFOLIO_VOL_FLOAT = 0.085
DEFAULT_CONFIG = replace(
    BASE_DEFAULT_CONFIG,
    asset_list=ASSET_TUPLE,
    target_portfolio_vol_float=TARGET_PORTFOLIO_VOL_FLOAT,
    trigger_portfolio_vol_float=TRIGGER_PORTFOLIO_VOL_FLOAT,
)


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float = DEFAULT_CONFIG.capital_base_float,
    end_date_str: str | None = None,
) -> Beyond6040Strategy:
    config = replace(
        DEFAULT_CONFIG,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
    )
    pricing_data_df = get_beyond_6040_data(config=config)
    relevant_start_ts = get_first_actionable_rebalance_ts(
        pricing_data_df=pricing_data_df,
        asset_list=config.asset_list,
        asset_vol_lookback_int=config.asset_vol_lookback_int,
    )
    calendar_start_ts = relevant_start_ts
    if backtest_start_date_str is not None:
        calendar_start_ts = max(calendar_start_ts, pd.Timestamp(backtest_start_date_str))

    # *** CRITICAL*** Keep full pre-start price history for inverse-volatility
    # and volatility-overlay signal formation; only the executable fill calendar is clipped.
    calendar_index = pricing_data_df.index[pricing_data_df.index >= calendar_start_ts]

    strategy_obj = Beyond6040Strategy(
        name=STRATEGY_NAME_STR,
        benchmarks=config.benchmark_list,
        asset_list=config.asset_list,
        asset_vol_lookback_int=config.asset_vol_lookback_int,
        portfolio_vol_lookback_int=config.portfolio_vol_lookback_int,
        target_portfolio_vol_float=config.target_portfolio_vol_float,
        trigger_portfolio_vol_float=config.trigger_portfolio_vol_float,
        capital_base=config.capital_base_float,
        slippage=config.slippage_float,
        commission_per_share=config.commission_per_share_float,
        commission_minimum=config.commission_minimum_float,
    )

    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar=calendar_index,
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
