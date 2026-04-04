"""
SPY-only volatility-adjusted EMA trend-following baseline.

Core formulas
-------------
Signal inputs at close t:

    r_t
        = P_t / P_{t-1} - 1

    sigma_t^2
        = (1 - eta) * sigma_{t-1}^2 + eta * r_t^2

    z_t
        = r_t / sigma_{t-1}

    phi_t
        = (1 - eta) * phi_{t-1} + sqrt(eta) * z_t

    w_t
        = clip(phi_t, 0, w_max)

Target shares at next-open execution bar t + 1:

    q_{t+1}^{target}
        = floor(V_t * w_t / O_{t+1})

This file deliberately keeps the recursion explicit instead of using compact
`ewm()` chains so the quantitative timing can be audited line by line.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from data.norgate_loader import load_raw_prices


def default_trade_id_int() -> int:
    return -1


@dataclass(frozen=True)
class SpyVolAdjustedEmaConfig:
    symbol_str: str = "SPY"
    benchmark_symbol_str: str = "$SPX"
    eta_float: float = 1.0 / 112.0
    max_weight_float: float = 1.5
    start_date_str: str = "1998-01-01"
    end_date_str: str | None = None
    capital_base_float: float = 100_000.0
    slippage_float: float = 0.0001
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0

    def __post_init__(self):
        if not self.symbol_str:
            raise ValueError("symbol_str must not be empty.")
        if not self.benchmark_symbol_str:
            raise ValueError("benchmark_symbol_str must not be empty.")
        if self.symbol_str == self.benchmark_symbol_str:
            raise ValueError("benchmark_symbol_str must differ from symbol_str.")
        if not np.isfinite(self.eta_float) or self.eta_float <= 0.0 or self.eta_float > 1.0:
            raise ValueError("eta_float must be in the interval (0, 1].")
        if not np.isfinite(self.max_weight_float) or self.max_weight_float <= 0.0:
            raise ValueError("max_weight_float must be positive.")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if self.slippage_float < 0.0:
            raise ValueError("slippage_float must be non-negative.")
        if self.commission_per_share_float < 0.0:
            raise ValueError("commission_per_share_float must be non-negative.")
        if self.commission_minimum_float < 0.0:
            raise ValueError("commission_minimum_float must be non-negative.")


DEFAULT_CONFIG = SpyVolAdjustedEmaConfig()


def get_spy_vol_adj_ema_data(
    config: SpyVolAdjustedEmaConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    return load_raw_prices(
        symbols=[config.symbol_str],
        benchmarks=[config.benchmark_symbol_str],
        start_date=config.start_date_str,
        end_date=config.end_date_str,
    )


def compute_spy_vol_adjusted_signal_df(
    price_close_ser: pd.Series,
    eta_float: float,
    max_weight_float: float = 1.5,
) -> pd.DataFrame:
    """
    Compute the single-asset normalized EMA trend signal with explicit recursions.

    The implementation follows:

        r_t = P_t / P_{t-1} - 1
        sigma_t^2 = (1 - eta) * sigma_{t-1}^2 + eta * r_t^2
        z_t = r_t / sigma_{t-1}
        phi_t = (1 - eta) * phi_{t-1} + sqrt(eta) * z_t
        w_t = clip(phi_t, 0, w_max)
        turnover_t = |w_t - w_{t-1}|

    The first bar with a finite return seeds sigma_t^2 with r_t^2. The first
    bar with a finite normalized return seeds phi_t with:

        phi_t = sqrt(eta) * z_t
    """
    if not np.isfinite(eta_float) or eta_float <= 0.0 or eta_float > 1.0:
        raise ValueError("eta_float must be in the interval (0, 1].")
    if not np.isfinite(max_weight_float) or max_weight_float <= 0.0:
        raise ValueError("max_weight_float must be positive.")

    price_close_ser = pd.Series(price_close_ser, copy=True).astype(float)

    # *** CRITICAL*** Daily returns must be computed from trailing closes only.
    return_ser = price_close_ser.pct_change(fill_method=None)
    return_vec = return_ser.to_numpy(dtype=float)

    volatility_ser = pd.Series(np.nan, index=price_close_ser.index, dtype=float)
    normalized_return_ser = pd.Series(np.nan, index=price_close_ser.index, dtype=float)
    trend_signal_ser = pd.Series(np.nan, index=price_close_ser.index, dtype=float)
    target_weight_ser = pd.Series(np.nan, index=price_close_ser.index, dtype=float)
    turnover_ser = pd.Series(np.nan, index=price_close_ser.index, dtype=float)

    sqrt_eta_float = float(np.sqrt(eta_float))
    prior_volatility_sq_float = np.nan
    prior_volatility_float = np.nan
    prior_trend_signal_float = np.nan
    prior_target_weight_float = np.nan

    for bar_idx_int, bar_ts in enumerate(price_close_ser.index):
        return_float = float(return_vec[bar_idx_int])
        if not np.isfinite(return_float):
            continue

        return_sq_float = return_float * return_float
        if np.isfinite(prior_volatility_sq_float):
            volatility_sq_float = (
                (1.0 - eta_float) * prior_volatility_sq_float
                + eta_float * return_sq_float
            )
        else:
            volatility_sq_float = return_sq_float

        volatility_float = float(np.sqrt(volatility_sq_float))
        volatility_ser.loc[bar_ts] = volatility_float

        # *** CRITICAL*** Normalization must use lagged volatility only.
        if np.isfinite(prior_volatility_float) and prior_volatility_float > 0.0:
            normalized_return_float = return_float / prior_volatility_float
            normalized_return_ser.loc[bar_ts] = normalized_return_float

            if np.isfinite(prior_trend_signal_float):
                trend_signal_float = (
                    (1.0 - eta_float) * prior_trend_signal_float
                    + sqrt_eta_float * normalized_return_float
                )
            else:
                trend_signal_float = sqrt_eta_float * normalized_return_float

            target_weight_float = float(np.clip(trend_signal_float, 0.0, max_weight_float))
            trend_signal_ser.loc[bar_ts] = trend_signal_float
            target_weight_ser.loc[bar_ts] = target_weight_float

            if np.isfinite(prior_target_weight_float):
                turnover_ser.loc[bar_ts] = abs(target_weight_float - prior_target_weight_float)

            prior_trend_signal_float = trend_signal_float
            prior_target_weight_float = target_weight_float

        prior_volatility_sq_float = volatility_sq_float
        prior_volatility_float = volatility_float

    signal_df = pd.DataFrame(
        {
            "return_ser": return_ser,
            "volatility_ser": volatility_ser,
            "normalized_return_ser": normalized_return_ser,
            "trend_signal_ser": trend_signal_ser,
            "target_weight_ser": target_weight_ser,
            "turnover_ser": turnover_ser,
        },
        index=price_close_ser.index,
    )
    return signal_df


class SpyVolAdjustedEmaStrategy(Strategy):
    """
    Long-only SPY baseline with a volatility-adjusted EMA trend signal.

    At close t:

        w_t = clip(phi_t, 0, w_max)

    At the next open t + 1:

        q_{t+1}^{target}
            = floor(V_t * w_t / O_{t+1})

    The strategy reuses the same trade ID across intermediate resizes while the
    position stays open, and opens a new trade ID only when going from flat to
    long.
    """

    enable_signal_audit = True
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: list[str] | tuple[str, ...],
        symbol_str: str = "SPY",
        eta_float: float = 1.0 / 112.0,
        max_weight_float: float = 1.5,
        capital_base: float = 100_000.0,
        slippage: float = 0.0001,
        commission_per_share: float = 0.005,
        commission_minimum: float = 1.0,
    ):
        super().__init__(
            name=name,
            benchmarks=list(benchmarks),
            capital_base=capital_base,
            slippage=slippage,
            commission_per_share=commission_per_share,
            commission_minimum=commission_minimum,
        )
        if not symbol_str:
            raise ValueError("symbol_str must not be empty.")
        if not np.isfinite(eta_float) or eta_float <= 0.0 or eta_float > 1.0:
            raise ValueError("eta_float must be in the interval (0, 1].")
        if not np.isfinite(max_weight_float) or max_weight_float <= 0.0:
            raise ValueError("max_weight_float must be positive.")

        self.symbol_str = str(symbol_str)
        self.eta_float = float(eta_float)
        self.max_weight_float = float(max_weight_float)
        self.trade_id_int = 0
        self.current_trade_id_int = default_trade_id_int()

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        if (self.symbol_str, "Close") not in pricing_data_df.columns:
            raise RuntimeError(f"Missing close data for {self.symbol_str}.")

        signal_data_df = pricing_data_df.copy()
        price_close_ser = signal_data_df[(self.symbol_str, "Close")].astype(float)
        signal_feature_df = compute_spy_vol_adjusted_signal_df(
            price_close_ser=price_close_ser,
            eta_float=self.eta_float,
            max_weight_float=self.max_weight_float,
        )
        signal_feature_df.columns = pd.MultiIndex.from_tuples(
            [(self.symbol_str, field_str) for field_str in signal_feature_df.columns]
        )
        return pd.concat([signal_data_df, signal_feature_df], axis=1)

    def iterate(self, data_df: pd.DataFrame, close_row_ser: pd.Series, open_price_ser: pd.Series):
        if close_row_ser is None or data_df is None:
            return

        target_weight_key = (self.symbol_str, "target_weight_ser")
        if target_weight_key not in close_row_ser.index:
            return

        target_weight_float = float(close_row_ser.loc[target_weight_key])
        if not np.isfinite(target_weight_float):
            return

        open_price_float = float(open_price_ser.get(self.symbol_str, np.nan))
        if not np.isfinite(open_price_float) or open_price_float <= 0.0:
            raise RuntimeError(f"Invalid open price for {self.symbol_str} on {self.current_bar}.")

        budget_value_float = float(self.previous_total_value)
        target_share_int = int(np.floor(budget_value_float * target_weight_float / open_price_float))
        current_share_int = int(self.get_position(self.symbol_str))

        # *** CRITICAL*** The target share count must be computed from the
        # previous close portfolio value and executed at the current open.
        if target_share_int == current_share_int:
            return

        if target_share_int <= 0:
            if current_share_int <= 0:
                return
            self.order_target(
                self.symbol_str,
                0,
                trade_id=self.current_trade_id_int,
            )
            self.current_trade_id_int = default_trade_id_int()
            return

        if current_share_int <= 0 or self.current_trade_id_int == default_trade_id_int():
            self.trade_id_int += 1
            self.current_trade_id_int = self.trade_id_int

        self.order_target(
            self.symbol_str,
            target_share_int,
            trade_id=self.current_trade_id_int,
        )


if __name__ == "__main__":
    config = DEFAULT_CONFIG
    pricing_data_df = get_spy_vol_adj_ema_data(config=config)

    strategy = SpyVolAdjustedEmaStrategy(
        name="strategy_mo_spy_vol_adj_ema",
        benchmarks=[config.benchmark_symbol_str],
        symbol_str=config.symbol_str,
        eta_float=config.eta_float,
        max_weight_float=config.max_weight_float,
        capital_base=config.capital_base_float,
        slippage=config.slippage_float,
        commission_per_share=config.commission_per_share_float,
        commission_minimum=config.commission_minimum_float,
    )

    run_daily(
        strategy,
        pricing_data_df,
        calendar=pricing_data_df.index,
        audit_override_bool=None,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    display(strategy.summary)
    display(strategy.summary_trades)
    save_results(strategy)
