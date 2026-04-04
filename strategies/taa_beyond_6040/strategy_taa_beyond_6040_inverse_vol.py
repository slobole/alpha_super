"""
Beyond 60/40 inverse-volatility-only strategy.

Core formulas
-------------
Daily asset returns:

    r_{i,t}
        = P_{i,t} / P_{i,t-1} - 1

Trailing annualized asset volatility:

    sigma_{i,t}^{(63)}
        = sqrt(252) * std(r_{i,t-62:t})

Inverse-volatility monthly base weights at month-end t:

    inv_vol_{i,t}
        = 1 / sigma_{i,t}^{(63)}

    w_{i,t}^{base}
        = inv_vol_{i,t} / sum_j(inv_vol_{j,t})

This variant removes the portfolio-level volatility target, so:

    w_{i,t}^{final}
        = w_{i,t}^{base}

    w_t^{cash}
        = 0

Execution rule
--------------
The strategy rebalances only on the first trading day of each month using the
previous month-end close information:

    q_{i,t}^{target}
        = floor(V_{t-1} * w_{i,t}^{base} / O_{i,t})
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
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


@dataclass(frozen=True)
class Beyond6040InverseVolConfig:
    asset_list: tuple[str, ...] = BASE_DEFAULT_CONFIG.asset_list
    benchmark_list: tuple[str, ...] = BASE_DEFAULT_CONFIG.benchmark_list
    asset_vol_lookback_int: int = BASE_DEFAULT_CONFIG.asset_vol_lookback_int
    start_date_str: str = BASE_DEFAULT_CONFIG.start_date_str
    end_date_str: str | None = BASE_DEFAULT_CONFIG.end_date_str
    capital_base_float: float = BASE_DEFAULT_CONFIG.capital_base_float
    slippage_float: float = BASE_DEFAULT_CONFIG.slippage_float
    commission_per_share_float: float = BASE_DEFAULT_CONFIG.commission_per_share_float
    commission_minimum_float: float = BASE_DEFAULT_CONFIG.commission_minimum_float

    def __post_init__(self):
        if len(self.asset_list) < 2:
            raise ValueError("asset_list must contain at least two assets.")
        if len(set(self.asset_list)) != len(self.asset_list):
            raise ValueError("asset_list contains duplicate symbols.")
        if len(set(self.benchmark_list)) != len(self.benchmark_list):
            raise ValueError("benchmark_list contains duplicate symbols.")
        if set(self.asset_list).intersection(self.benchmark_list):
            raise ValueError("benchmark_list must not overlap asset_list.")
        if self.asset_vol_lookback_int < 2:
            raise ValueError("asset_vol_lookback_int must be at least 2.")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if self.slippage_float < 0.0:
            raise ValueError("slippage_float must be non-negative.")
        if self.commission_per_share_float < 0.0:
            raise ValueError("commission_per_share_float must be non-negative.")
        if self.commission_minimum_float < 0.0:
            raise ValueError("commission_minimum_float must be non-negative.")


DEFAULT_CONFIG = Beyond6040InverseVolConfig()


def get_beyond_6040_inverse_vol_data(
    config: Beyond6040InverseVolConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    return get_beyond_6040_data(
        BASE_DEFAULT_CONFIG.__class__(
            asset_list=config.asset_list,
            benchmark_list=config.benchmark_list,
            asset_vol_lookback_int=config.asset_vol_lookback_int,
            portfolio_vol_lookback_int=BASE_DEFAULT_CONFIG.portfolio_vol_lookback_int,
            target_portfolio_vol_float=BASE_DEFAULT_CONFIG.target_portfolio_vol_float,
            trigger_portfolio_vol_float=BASE_DEFAULT_CONFIG.trigger_portfolio_vol_float,
            start_date_str=config.start_date_str,
            end_date_str=config.end_date_str,
            capital_base_float=config.capital_base_float,
            slippage_float=config.slippage_float,
            commission_per_share_float=config.commission_per_share_float,
            commission_minimum_float=config.commission_minimum_float,
        )
    )


class Beyond6040InverseVolStrategy(Beyond6040Strategy):
    """
    Monthly inverse-volatility allocator without a daily volatility overlay.
    """

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str] | None = None,
        asset_list: Sequence[str] = DEFAULT_CONFIG.asset_list,
        asset_vol_lookback_int: int = DEFAULT_CONFIG.asset_vol_lookback_int,
        capital_base: float = DEFAULT_CONFIG.capital_base_float,
        slippage: float = DEFAULT_CONFIG.slippage_float,
        commission_per_share: float = DEFAULT_CONFIG.commission_per_share_float,
        commission_minimum: float = DEFAULT_CONFIG.commission_minimum_float,
    ):
        super().__init__(
            name=name,
            benchmarks=benchmarks,
            asset_list=asset_list,
            asset_vol_lookback_int=asset_vol_lookback_int,
            portfolio_vol_lookback_int=BASE_DEFAULT_CONFIG.portfolio_vol_lookback_int,
            target_portfolio_vol_float=BASE_DEFAULT_CONFIG.target_portfolio_vol_float,
            trigger_portfolio_vol_float=BASE_DEFAULT_CONFIG.trigger_portfolio_vol_float,
            capital_base=capital_base,
            slippage=slippage,
            commission_per_share=commission_per_share,
            commission_minimum=commission_minimum,
        )

    def iterate(self, data_df: pd.DataFrame, close_row_ser: pd.Series, open_price_ser: pd.Series):
        if close_row_ser is None or data_df is None:
            return

        base_weight_ser = self._current_base_weight_ser(close_row_ser).astype(float)
        if base_weight_ser.isna().any():
            cash_only_weight_ser = pd.Series(
                [0.0] * len(self.asset_list) + [1.0],
                index=self.asset_list + ["Cash"],
                dtype=float,
            )
            self._record_daily_target_weight_ser(cash_only_weight_ser)
            return

        target_weight_ser = pd.concat(
            [base_weight_ser, pd.Series({"Cash": 0.0}, dtype=float)]
        )
        if not np.isclose(float(target_weight_ser.sum()), 1.0, atol=1e-12):
            raise ValueError(
                f"Final target weights must sum to 1.0 on {self.current_bar}, "
                f"found {target_weight_ser.sum():.12f}."
            )

        self._record_daily_target_weight_ser(target_weight_ser)

        is_month_turn_bool = (
            self.previous_bar is not None
            and pd.Timestamp(self.current_bar).to_period("M") != pd.Timestamp(self.previous_bar).to_period("M")
        )
        if not is_month_turn_bool:
            return

        current_position_ser = self.get_positions().reindex(self.asset_list, fill_value=0.0).astype(int)
        budget_value_float = float(self.previous_total_value)

        for asset_str in self.asset_list:
            target_weight_float = float(target_weight_ser.loc[asset_str])
            current_share_int = int(current_position_ser.loc[asset_str])
            open_price_float = float(open_price_ser.get(asset_str, np.nan))
            if not np.isfinite(open_price_float) or open_price_float <= 0.0:
                raise RuntimeError(f"Invalid open price for target asset {asset_str} on {self.current_bar}.")

            # *** CRITICAL*** Monthly target shares are computed from the
            # previous-bar total value and executed at the current month-open.
            target_share_int = int(np.floor(budget_value_float * target_weight_float / open_price_float))

            if target_share_int == current_share_int:
                continue

            if target_share_int <= 0:
                if current_share_int <= 0:
                    continue
                self.order_target_value(
                    asset_str,
                    0.0,
                    trade_id=self.current_trade_id_map[asset_str],
                )
                self.current_trade_id_map[asset_str] = -1
                continue

            if current_share_int <= 0 or self.current_trade_id_map[asset_str] == -1:
                self.trade_id_int += 1
                self.current_trade_id_map[asset_str] = self.trade_id_int

            self.order_target_percent(
                asset_str,
                target_weight_float,
                trade_id=self.current_trade_id_map[asset_str],
            )


if __name__ == "__main__":
    config = DEFAULT_CONFIG
    pricing_data_df = get_beyond_6040_inverse_vol_data(config=config)
    relevant_start_ts = get_first_actionable_rebalance_ts(
        pricing_data_df=pricing_data_df,
        asset_list=config.asset_list,
        asset_vol_lookback_int=config.asset_vol_lookback_int,
    )
    calendar_index = pricing_data_df.index[pricing_data_df.index >= relevant_start_ts]

    strategy = Beyond6040InverseVolStrategy(
        name="strategy_taa_beyond_6040_inverse_vol",
        benchmarks=config.benchmark_list,
        asset_list=config.asset_list,
        asset_vol_lookback_int=config.asset_vol_lookback_int,
        capital_base=config.capital_base_float,
        slippage=config.slippage_float,
        commission_per_share=config.commission_per_share_float,
        commission_minimum=config.commission_minimum_float,
    )

    run_daily(
        strategy,
        pricing_data_df,
        calendar=calendar_index,
        audit_override_bool=None,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    display(strategy.summary)
    display(strategy.summary_trades)
    save_results(strategy)
