"""
SPX RSI(2)-driven short-volatility sleeve implemented through long `SVIX`.

TL;DR: Buy `SVIX` at the next open when the prior close shows:
1. `SPX` RSI(2) below 30, and
2. the position size implied by the target weight is positive.

Core formulas
-------------
At close t:

    Delta C_t
        = C_t - C_{t-1}

    U_t
        = max(Delta C_t, 0)

    D_t
        = max(-Delta C_t, 0)

    RS_t^{(2)}
        = WilderEMA_2(U_t) / WilderEMA_2(D_t)

    RSI2_t
        = 100 - 100 / (1 + RS_t^{(2)})

    entry_signal_t
        = 1[RSI2_t < theta_rsi]

    target_weight_t
        = w_alloc * entry_signal_t

At the next open t + 1:

    q_{t+1}^{target}
        = floor(V_t * target_weight_t / O_{t+1}^{SVIX})

where:

    theta_rsi = 30 by default
    w_alloc   = 1.0 by default
    V_t       = prior-close total portfolio value
    O_{t+1}^{SVIX} = current-bar open for `SVIX`

Holding-horizon note
--------------------
Default behavior is a rolling target-position implementation:

    if entry_signal_t = 1:
        hold short-vol exposure at open t + 1

    if entry_signal_t = 0:
        target flat at open t + 1

So a true signal at close t creates a position for the session from
open t + 1 to open t + 2, and repeated true signals can chain into a
multi-session holding period.

An optional strict mode is also supported:

    if strict_one_session_hold_bool = True and entry_signal_t = 1:
        enter at open t + 1
        exit at open t + 2

That strict mode increases turnover and trading costs relative to the
rolling-hold implementation.

Instrument note
---------------
Long `SVIX` is a tradable proxy for short-vol exposure, but it is not an exact
short of spot VIX or a single VX futures contract. Its return is linked to an
inverse short-term VIX futures index, so:

    return_t^{SVIX} != return_t^{short_VX}

in general because of daily reset, index construction, roll mechanics, fees,
and convexity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from IPython.display import display
import talib

WORKSPACE_ROOT_PATH = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT_PATH))

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from data.norgate_loader import load_raw_prices


def default_trade_id_int() -> int:
    return -1


@dataclass(frozen=True)
class SpxRsi2Svix1DayConfig:
    trade_symbol_str: str = "SVXY"
    signal_symbol_str: str = "$SPX"
    benchmark_symbol_str: str = "SPY"
    rsi_window_int: int = 2
    rsi_threshold_float: float = 30.0
    target_weight_float: float = 1.00
    strict_one_session_hold_bool: bool = False
    start_date_str: str = "2019-01-01"
    end_date_str: str | None = None
    capital_base_float: float = 10_000.0
    slippage_float: float = 0.001
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0

    def __post_init__(self):
        symbol_set = {
            self.trade_symbol_str,
            self.signal_symbol_str,
            self.benchmark_symbol_str,
        }
        if len(symbol_set) != 3:
            raise ValueError("All symbols must be distinct.")
        if self.rsi_window_int <= 0:
            raise ValueError("rsi_window_int must be positive.")
        if not np.isfinite(self.rsi_threshold_float):
            raise ValueError("rsi_threshold_float must be finite.")
        if not 0.0 <= self.rsi_threshold_float <= 100.0:
            raise ValueError("rsi_threshold_float must lie in [0, 100].")
        if not np.isfinite(self.target_weight_float) or self.target_weight_float <= 0.0:
            raise ValueError("target_weight_float must be positive and finite.")
        if self.target_weight_float > 1.0:
            raise ValueError("target_weight_float must be <= 1.0 for this long-only sleeve.")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if self.slippage_float < 0.0:
            raise ValueError("slippage_float must be non-negative.")
        if self.commission_per_share_float < 0.0:
            raise ValueError("commission_per_share_float must be non-negative.")
        if self.commission_minimum_float < 0.0:
            raise ValueError("commission_minimum_float must be non-negative.")


DEFAULT_CONFIG = SpxRsi2Svix1DayConfig()


def get_spx_rsi2_svix_1day_data(
    config: SpxRsi2Svix1DayConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    return load_raw_prices(
        symbols=[config.trade_symbol_str, config.signal_symbol_str],
        benchmarks=[config.benchmark_symbol_str],
        start_date=config.start_date_str,
        end_date=config.end_date_str,
    )


def get_first_tradeable_bar_ts(
    pricing_data_df: pd.DataFrame,
    trade_symbol_str: str,
) -> pd.Timestamp:
    open_price_ser = pricing_data_df[(trade_symbol_str, "Open")].astype(float)
    valid_open_ser = open_price_ser.dropna()
    if len(valid_open_ser) == 0:
        raise RuntimeError(f"No valid open prices found for {trade_symbol_str}.")
    return pd.Timestamp(valid_open_ser.index[0])


def compute_spx_rsi2_svix_signal_df(
    spx_close_ser: pd.Series,
    rsi_window_int: int,
    rsi_threshold_float: float,
    target_weight_float: float,
) -> pd.DataFrame:
    """
    Compute the SPX RSI(2) signal state for the SVIX sleeve.

        entry_signal_t = 1[RSI2_t < theta_rsi]
        target_weight_t = w_alloc * entry_signal_t

    The returned target weight is observed at close t and executed by the
    engine at the next open t + 1.
    """
    if rsi_window_int <= 0:
        raise ValueError("rsi_window_int must be positive.")
    if not np.isfinite(rsi_threshold_float):
        raise ValueError("rsi_threshold_float must be finite.")
    if not 0.0 <= rsi_threshold_float <= 100.0:
        raise ValueError("rsi_threshold_float must lie in [0, 100].")
    if not np.isfinite(target_weight_float) or target_weight_float <= 0.0:
        raise ValueError("target_weight_float must be positive and finite.")

    spx_close_ser = pd.Series(spx_close_ser, copy=True).astype(float)

    # *** CRITICAL*** RSI2 must be computed from trailing SPX close history
    # only. The library call is causal as long as the input close series is.
    spx_rsi2_ser = pd.Series(
        talib.RSI(spx_close_ser.to_numpy(dtype=float), timeperiod=rsi_window_int),
        index=spx_close_ser.index,
        dtype=float,
        name="spx_rsi2_ser",
    )
    entry_signal_bool_ser = spx_rsi2_ser < float(rsi_threshold_float)
    target_weight_ser = pd.Series(
        np.where(entry_signal_bool_ser, float(target_weight_float), 0.0),
        index=spx_close_ser.index,
        dtype=float,
        name="target_weight_ser",
    )

    signal_feature_df = pd.DataFrame(
        {
            "spx_close_ser": spx_close_ser,
            "spx_rsi2_ser": spx_rsi2_ser,
            "entry_signal_bool_ser": entry_signal_bool_ser.astype(bool),
            "target_weight_ser": target_weight_ser,
        },
        index=spx_close_ser.index,
    )
    return signal_feature_df


class SpxRsi2Svix1DayStrategy(Strategy):
    """
    Single-asset inverse-vol sleeve driven by prior-close SPX RSI(2).

        target_weight_t = w_alloc * 1[RSI2_t < theta_rsi]

    and the target shares submitted at the next open are:

        q_{t+1}^{target} = floor(V_t * target_weight_t / O_{t+1}^{SVIX})

    Optional strict mode:

        q_{t+2}^{target} = 0
    """

    enable_signal_audit = True
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: list[str] | tuple[str, ...],
        trade_symbol_str: str,
        signal_symbol_str: str,
        rsi_window_int: int,
        rsi_threshold_float: float,
        target_weight_float: float,
        strict_one_session_hold_bool: bool = False,
        capital_base: float = 10_000.0,
        slippage: float = 0.001,
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
        if not trade_symbol_str or not signal_symbol_str:
            raise ValueError("Symbols must not be empty.")
        if rsi_window_int <= 0:
            raise ValueError("rsi_window_int must be positive.")
        if not np.isfinite(rsi_threshold_float):
            raise ValueError("rsi_threshold_float must be finite.")
        if not 0.0 <= rsi_threshold_float <= 100.0:
            raise ValueError("rsi_threshold_float must lie in [0, 100].")
        if not np.isfinite(target_weight_float) or target_weight_float <= 0.0:
            raise ValueError("target_weight_float must be positive and finite.")

        self.trade_symbol_str = str(trade_symbol_str)
        self.signal_symbol_str = str(signal_symbol_str)
        self.rsi_window_int = int(rsi_window_int)
        self.rsi_threshold_float = float(rsi_threshold_float)
        self.target_weight_float = float(target_weight_float)
        self.strict_one_session_hold_bool = bool(strict_one_session_hold_bool)
        self.trade_id_int = 0
        self.current_trade_id_int = default_trade_id_int()

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        signal_key_tup = (self.signal_symbol_str, "Close")
        if signal_key_tup not in pricing_data_df.columns:
            raise RuntimeError(f"Missing required signal column: {signal_key_tup}")

        signal_data_df = pricing_data_df.copy()
        signal_feature_df = compute_spx_rsi2_svix_signal_df(
            spx_close_ser=signal_data_df[(self.signal_symbol_str, "Close")],
            rsi_window_int=self.rsi_window_int,
            rsi_threshold_float=self.rsi_threshold_float,
            target_weight_float=self.target_weight_float,
        )

        # *** CRITICAL*** No shift is applied inside compute_signals(). The
        # engine passes previous_bar close data into iterate(), so the signal
        # computed at date t is executed at the open of date t + 1.
        signal_feature_df.columns = pd.MultiIndex.from_tuples(
            [(self.trade_symbol_str, field_str) for field_str in signal_feature_df.columns]
        )
        return pd.concat([signal_data_df, signal_feature_df], axis=1)

    def iterate(self, data_df: pd.DataFrame, close_row_ser: pd.Series, open_price_ser: pd.Series):
        if close_row_ser is None or data_df is None:
            return

        target_weight_key_tup = (self.trade_symbol_str, "target_weight_ser")
        if target_weight_key_tup not in close_row_ser.index:
            return

        target_weight_float = float(close_row_ser.loc[target_weight_key_tup])
        if not np.isfinite(target_weight_float):
            return

        current_share_int = int(self.get_position(self.trade_symbol_str))

        if self.strict_one_session_hold_bool:
            carried_position_bool = current_share_int > 0

            # *** CRITICAL*** A strict one-session hold means any carried
            # position from yesterday must be liquidated at today's open before
            # any new entry is considered. This avoids accidental multi-day
            # holding.
            if carried_position_bool:
                self.order_target(
                    self.trade_symbol_str,
                    0,
                    trade_id=self.current_trade_id_int,
                )
                self.current_trade_id_int = default_trade_id_int()

            if target_weight_float <= 0.0:
                return

            open_price_float = float(open_price_ser.get(self.trade_symbol_str, np.nan))
            if not np.isfinite(open_price_float) or open_price_float <= 0.0:
                return

            budget_value_float = float(self.previous_total_value)

            # *** CRITICAL*** Target shares are computed from prior-close
            # equity and current-bar open:
            #
            #     q_{t+1}^{target} = floor(V_t * w_t / O_{t+1})
            #
            # This preserves next-open execution and avoids same-bar close
            # fills or other time-series leakage.
            target_share_int = int(np.floor(budget_value_float * target_weight_float / open_price_float))

            if target_share_int <= 0:
                return

            self.trade_id_int += 1
            self.current_trade_id_int = self.trade_id_int
            self.order_target(
                self.trade_symbol_str,
                target_share_int,
                trade_id=self.current_trade_id_int,
            )
            return

        open_price_float = float(open_price_ser.get(self.trade_symbol_str, np.nan))
        if not np.isfinite(open_price_float) or open_price_float <= 0.0:
            if current_share_int == 0:
                return
            raise RuntimeError(f"Invalid open price for {self.trade_symbol_str} on {self.current_bar}.")

        budget_value_float = float(self.previous_total_value)

        # *** CRITICAL*** Target shares are computed from prior-close equity and
        # current-bar open:
        #
        #     q_{t+1}^{target} = floor(V_t * w_t / O_{t+1})
        #
        # This preserves next-open execution and avoids same-bar close fills or
        # other time-series leakage.
        target_share_int = int(np.floor(budget_value_float * target_weight_float / open_price_float))

        if target_share_int == current_share_int:
            return

        if target_share_int <= 0:
            if current_share_int <= 0:
                return
            self.order_target(
                self.trade_symbol_str,
                0,
                trade_id=self.current_trade_id_int,
            )
            self.current_trade_id_int = default_trade_id_int()
            return

        if current_share_int <= 0 or self.current_trade_id_int == default_trade_id_int():
            self.trade_id_int += 1
            self.current_trade_id_int = self.trade_id_int

        self.order_target(
            self.trade_symbol_str,
            target_share_int,
            trade_id=self.current_trade_id_int,
        )


if __name__ == "__main__":
    config = DEFAULT_CONFIG
    pricing_data_df = get_spx_rsi2_svix_1day_data(config=config)
    first_tradeable_bar_ts = get_first_tradeable_bar_ts(
        pricing_data_df=pricing_data_df,
        trade_symbol_str=config.trade_symbol_str,
    )
    calendar_idx = pricing_data_df.index[pricing_data_df.index >= first_tradeable_bar_ts]

    strategy = SpxRsi2Svix1DayStrategy(
        name="strategy_mr_spx_rsi2_svix_1day",
        benchmarks=[config.benchmark_symbol_str],
        trade_symbol_str=config.trade_symbol_str,
        signal_symbol_str=config.signal_symbol_str,
        rsi_window_int=config.rsi_window_int,
        rsi_threshold_float=config.rsi_threshold_float,
        target_weight_float=config.target_weight_float,
        strict_one_session_hold_bool=config.strict_one_session_hold_bool,
        capital_base=config.capital_base_float,
        slippage=config.slippage_float,
        commission_per_share=config.commission_per_share_float,
        commission_minimum=config.commission_minimum_float,
    )

    run_daily(
        strategy,
        pricing_data_df,
        calendar=calendar_idx,
        audit_override_bool=None,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    display(strategy.summary)
    display(strategy.summary_trades)
    save_results(strategy)
