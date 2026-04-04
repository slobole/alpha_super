"""
Merged VIX1D regime strategy using volatility ETPs.

TL;DR: At each prior close:
1. Buy `VIXY` next open if `VIX1D <= 10` and `VIX <= VIX3M`.
2. Buy `SVIX` next open if `VIX1D >= 15` and `VIX <= VIX3M`.
3. Stay flat otherwise.
4. If `VIX > VIX3M`, force flat and close any open position.

Core formulas
-------------
At close t:

    term_spread_t
        = VIX_t - VIX3M_t

    trade_filter_ok_t
        = 1[term_spread_t <= 0]

    long_vol_signal_t
        = 1[VIX1D_t <= theta_long] * trade_filter_ok_t

    short_vol_signal_t
        = 1[VIX1D_t >= theta_short] * trade_filter_ok_t

with:

    theta_long  = 10
    theta_short = 15

The mutually-exclusive target weights are:

    w_t^{VIXY}
        = w_alloc * long_vol_signal_t

    w_t^{SVIX}
        = w_alloc * short_vol_signal_t

subject to:

    w_t^{VIXY} * w_t^{SVIX} = 0

At the next open t + 1:

    q_{t+1,i}^{target}
        = floor(V_t * w_{t,i} / O_{t+1,i})

where:

    i in {VIXY, short_vol_ETP}
    V_t = prior-close total portfolio value
    O_{t+1,i} = current-bar open for asset i

Instrument note
---------------
These are ETPs linked to VIX futures, not spot-volatility instruments. So:

    return_t^{VIXY} != delta VIX_t
    return_t^{SVIX} != -delta VIX_t

in general.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from IPython.display import display

WORKSPACE_ROOT_PATH = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT_PATH))

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from data.norgate_loader import load_raw_prices


SIGNAL_NAMESPACE_STR = "signal_state"


def default_trade_id_int() -> int:
    return -1


@dataclass(frozen=True)
class Vix1dRegimeConfig:
    long_vol_trade_symbol_str: str = "VIXY"
    short_vol_trade_symbol_str: str = "SVIX"
    vix1d_symbol_str: str = "$VIX1D"
    vix_symbol_str: str = "$VIX"
    vix3m_symbol_str: str = "$VIX3M"
    benchmark_symbol_str: str = "SPY"
    long_threshold_float: float = 10.0
    short_threshold_float: float = 15.0
    target_weight_float: float = 1.00
    start_date_str: str = "2022-01-06"
    end_date_str: str | None = None
    capital_base_float: float = 10_000.0
    slippage_float: float = 0.001
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0

    def __post_init__(self):
        symbol_set = {
            self.long_vol_trade_symbol_str,
            self.short_vol_trade_symbol_str,
            self.vix1d_symbol_str,
            self.vix_symbol_str,
            self.vix3m_symbol_str,
            self.benchmark_symbol_str,
        }
        if len(symbol_set) != 6:
            raise ValueError("All symbols must be distinct.")
        if not np.isfinite(self.long_threshold_float):
            raise ValueError("long_threshold_float must be finite.")
        if not np.isfinite(self.short_threshold_float):
            raise ValueError("short_threshold_float must be finite.")
        if not self.long_threshold_float < self.short_threshold_float:
            raise ValueError("long_threshold_float must be strictly less than short_threshold_float.")
        if not np.isfinite(self.target_weight_float) or self.target_weight_float <= 0.0:
            raise ValueError("target_weight_float must be positive.")
        if self.target_weight_float > 1.0:
            raise ValueError("target_weight_float must be <= 1.0.")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if self.slippage_float < 0.0:
            raise ValueError("slippage_float must be non-negative.")
        if self.commission_per_share_float < 0.0:
            raise ValueError("commission_per_share_float must be non-negative.")
        if self.commission_minimum_float < 0.0:
            raise ValueError("commission_minimum_float must be non-negative.")

    @property
    def trade_symbol_list(self) -> tuple[str, str]:
        return (
            self.long_vol_trade_symbol_str,
            self.short_vol_trade_symbol_str,
        )

    @property
    def signal_symbol_list(self) -> tuple[str, str, str]:
        return (
            self.vix1d_symbol_str,
            self.vix_symbol_str,
            self.vix3m_symbol_str,
        )


DEFAULT_CONFIG = Vix1dRegimeConfig()


def get_vix1d_regime_data(
    config: Vix1dRegimeConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    return load_raw_prices(
        symbols=[*config.trade_symbol_list, *config.signal_symbol_list],
        benchmarks=[config.benchmark_symbol_str],
        start_date=config.start_date_str,
        end_date=config.end_date_str,
    )


def compute_vix1d_regime_signal_df(
    vix1d_close_ser: pd.Series,
    vix_close_ser: pd.Series,
    vix3m_close_ser: pd.Series,
    long_threshold_float: float,
    short_threshold_float: float,
    target_weight_float: float,
    long_vol_trade_symbol_str: str,
    short_vol_trade_symbol_str: str,
) -> pd.DataFrame:
    """
    Compute the merged regime signal state from the three volatility series.
    """
    if not np.isfinite(long_threshold_float):
        raise ValueError("long_threshold_float must be finite.")
    if not np.isfinite(short_threshold_float):
        raise ValueError("short_threshold_float must be finite.")
    if not long_threshold_float < short_threshold_float:
        raise ValueError("long_threshold_float must be strictly less than short_threshold_float.")
    if not np.isfinite(target_weight_float) or target_weight_float <= 0.0:
        raise ValueError("target_weight_float must be positive.")

    vix1d_close_ser = pd.Series(vix1d_close_ser, copy=True).astype(float)
    vix_close_ser = pd.Series(vix_close_ser, copy=True).astype(float)
    vix3m_close_ser = pd.Series(vix3m_close_ser, copy=True).astype(float)

    signal_input_df = pd.concat(
        [
            vix1d_close_ser.rename("vix1d_close_ser"),
            vix_close_ser.rename("vix_close_ser"),
            vix3m_close_ser.rename("vix3m_close_ser"),
        ],
        axis=1,
    ).sort_index()

    term_spread_ser = signal_input_df["vix_close_ser"] - signal_input_df["vix3m_close_ser"]
    trade_filter_ok_bool_ser = term_spread_ser <= 0.0

    long_vol_signal_bool_ser = (
        (signal_input_df["vix1d_close_ser"] <= float(long_threshold_float))
        & trade_filter_ok_bool_ser
    )
    short_vol_signal_bool_ser = (
        (signal_input_df["vix1d_close_ser"] >= float(short_threshold_float))
        & trade_filter_ok_bool_ser
    )

    if bool((long_vol_signal_bool_ser & short_vol_signal_bool_ser).any()):
        raise ValueError("Merged regime signals must be mutually exclusive.")

    long_vol_target_weight_ser = pd.Series(
        np.where(long_vol_signal_bool_ser, float(target_weight_float), 0.0),
        index=signal_input_df.index,
        dtype=float,
        name="long_vol_target_weight_ser",
    )
    short_vol_target_weight_ser = pd.Series(
        np.where(short_vol_signal_bool_ser, float(target_weight_float), 0.0),
        index=signal_input_df.index,
        dtype=float,
        name="short_vol_target_weight_ser",
    )

    target_symbol_obj_ser = pd.Series(None, index=signal_input_df.index, dtype=object, name="target_symbol_obj_ser")
    target_symbol_obj_ser.loc[long_vol_signal_bool_ser] = str(long_vol_trade_symbol_str)
    target_symbol_obj_ser.loc[short_vol_signal_bool_ser] = str(short_vol_trade_symbol_str)

    signal_feature_df = pd.DataFrame(
        {
            "vix1d_close_ser": signal_input_df["vix1d_close_ser"],
            "vix_close_ser": signal_input_df["vix_close_ser"],
            "vix3m_close_ser": signal_input_df["vix3m_close_ser"],
            "term_spread_ser": term_spread_ser,
            "trade_filter_ok_bool": trade_filter_ok_bool_ser.astype(bool),
            "long_vol_signal_bool": long_vol_signal_bool_ser.astype(bool),
            "short_vol_signal_bool": short_vol_signal_bool_ser.astype(bool),
            "long_vol_target_weight_ser": long_vol_target_weight_ser,
            "short_vol_target_weight_ser": short_vol_target_weight_ser,
            "target_symbol_obj_ser": target_symbol_obj_ser,
        },
        index=signal_input_df.index,
    )
    return signal_feature_df


class Vix1dRegimeStrategy(Strategy):
    """
    Merged regime-switching sleeve across one long-vol ETP and one inverse-vol ETP.
    """

    enable_signal_audit = True
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: list[str] | tuple[str, ...],
        long_vol_trade_symbol_str: str,
        short_vol_trade_symbol_str: str,
        vix1d_symbol_str: str,
        vix_symbol_str: str,
        vix3m_symbol_str: str,
        long_threshold_float: float,
        short_threshold_float: float,
        target_weight_float: float,
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
        if not long_vol_trade_symbol_str or not short_vol_trade_symbol_str:
            raise ValueError("Trade symbols must not be empty.")
        if long_vol_trade_symbol_str == short_vol_trade_symbol_str:
            raise ValueError("Trade symbols must differ.")
        if not vix1d_symbol_str or not vix_symbol_str or not vix3m_symbol_str:
            raise ValueError("Signal symbols must not be empty.")
        if not np.isfinite(long_threshold_float) or not np.isfinite(short_threshold_float):
            raise ValueError("Thresholds must be finite.")
        if not long_threshold_float < short_threshold_float:
            raise ValueError("long_threshold_float must be strictly less than short_threshold_float.")
        if not np.isfinite(target_weight_float) or target_weight_float <= 0.0:
            raise ValueError("target_weight_float must be positive.")

        self.long_vol_trade_symbol_str = str(long_vol_trade_symbol_str)
        self.short_vol_trade_symbol_str = str(short_vol_trade_symbol_str)
        self.vix1d_symbol_str = str(vix1d_symbol_str)
        self.vix_symbol_str = str(vix_symbol_str)
        self.vix3m_symbol_str = str(vix3m_symbol_str)
        self.long_threshold_float = float(long_threshold_float)
        self.short_threshold_float = float(short_threshold_float)
        self.target_weight_float = float(target_weight_float)
        self.trade_symbol_list = [
            self.long_vol_trade_symbol_str,
            self.short_vol_trade_symbol_str,
        ]
        self.trade_id_int = 0
        self.current_trade_id_map = {
            symbol_str: default_trade_id_int() for symbol_str in self.trade_symbol_list
        }

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        required_key_list = [
            (self.vix1d_symbol_str, "Close"),
            (self.vix_symbol_str, "Close"),
            (self.vix3m_symbol_str, "Close"),
        ]
        missing_key_list = [key_tup for key_tup in required_key_list if key_tup not in pricing_data_df.columns]
        if len(missing_key_list) > 0:
            raise RuntimeError(f"Missing required signal columns: {missing_key_list}")

        signal_data_df = pricing_data_df.copy()
        signal_feature_df = compute_vix1d_regime_signal_df(
            vix1d_close_ser=signal_data_df[(self.vix1d_symbol_str, "Close")],
            vix_close_ser=signal_data_df[(self.vix_symbol_str, "Close")],
            vix3m_close_ser=signal_data_df[(self.vix3m_symbol_str, "Close")],
            long_threshold_float=self.long_threshold_float,
            short_threshold_float=self.short_threshold_float,
            target_weight_float=self.target_weight_float,
            long_vol_trade_symbol_str=self.long_vol_trade_symbol_str,
            short_vol_trade_symbol_str=self.short_vol_trade_symbol_str,
        )

        # *** CRITICAL*** No shift is applied inside compute_signals(). The
        # engine passes previous_bar close data into iterate(), so the signal
        # computed at date t is executed at the open of date t + 1.
        signal_feature_df.columns = pd.MultiIndex.from_tuples(
            [(SIGNAL_NAMESPACE_STR, field_str) for field_str in signal_feature_df.columns]
        )
        return pd.concat([signal_data_df, signal_feature_df], axis=1)

    def iterate(self, data_df: pd.DataFrame, close_row_ser: pd.Series, open_price_ser: pd.Series):
        if close_row_ser is None or data_df is None:
            return

        long_weight_key = (SIGNAL_NAMESPACE_STR, "long_vol_target_weight_ser")
        short_weight_key = (SIGNAL_NAMESPACE_STR, "short_vol_target_weight_ser")
        if long_weight_key not in close_row_ser.index or short_weight_key not in close_row_ser.index:
            return

        long_vol_target_weight_float = float(close_row_ser.loc[long_weight_key])
        short_vol_target_weight_float = float(close_row_ser.loc[short_weight_key])
        if not np.isfinite(long_vol_target_weight_float) or not np.isfinite(short_vol_target_weight_float):
            return

        if long_vol_target_weight_float > 0.0 and short_vol_target_weight_float > 0.0:
            raise RuntimeError("Merged regime strategy cannot target both trade assets at once.")

        if long_vol_target_weight_float > 0.0:
            target_symbol_str = self.long_vol_trade_symbol_str
            target_weight_float = long_vol_target_weight_float
        elif short_vol_target_weight_float > 0.0:
            target_symbol_str = self.short_vol_trade_symbol_str
            target_weight_float = short_vol_target_weight_float
        else:
            target_symbol_str = None
            target_weight_float = 0.0

        current_position_ser = self.get_positions().reindex(self.trade_symbol_list, fill_value=0.0).astype(int)

        for asset_str in self.trade_symbol_list:
            current_share_int = int(current_position_ser.loc[asset_str])
            if current_share_int == 0:
                continue
            if asset_str == target_symbol_str:
                continue

            self.order_target(
                asset_str,
                0,
                trade_id=self.current_trade_id_map[asset_str],
            )
            self.current_trade_id_map[asset_str] = default_trade_id_int()

        if target_symbol_str is None or target_weight_float <= 0.0:
            return

        open_price_float = float(open_price_ser.get(target_symbol_str, np.nan))
        current_share_int = int(current_position_ser.loc[target_symbol_str])
        if not np.isfinite(open_price_float) or open_price_float <= 0.0:
            # Missing open on the target asset means no new entry can be placed.
            # Any non-target exits have already been queued above.
            return

        budget_value_float = float(self.previous_total_value)

        # *** CRITICAL*** Target shares are computed from prior-close equity and
        # current-bar open. This preserves next-open execution and avoids any
        # same-bar close fill fantasy.
        target_share_int = int(np.floor(budget_value_float * target_weight_float / open_price_float))

        if target_share_int == current_share_int:
            return

        if target_share_int <= 0:
            if current_share_int <= 0:
                return
            self.order_target(
                target_symbol_str,
                0,
                trade_id=self.current_trade_id_map[target_symbol_str],
            )
            self.current_trade_id_map[target_symbol_str] = default_trade_id_int()
            return

        if current_share_int <= 0 or self.current_trade_id_map[target_symbol_str] == default_trade_id_int():
            self.trade_id_int += 1
            self.current_trade_id_map[target_symbol_str] = self.trade_id_int

        self.order_target(
            target_symbol_str,
            target_share_int,
            trade_id=self.current_trade_id_map[target_symbol_str],
        )


if __name__ == "__main__":
    config = DEFAULT_CONFIG
    pricing_data_df = get_vix1d_regime_data(config=config)

    strategy = Vix1dRegimeStrategy(
        name="strategy_mr_vix1d_regime",
        benchmarks=[config.benchmark_symbol_str],
        long_vol_trade_symbol_str=config.long_vol_trade_symbol_str,
        short_vol_trade_symbol_str=config.short_vol_trade_symbol_str,
        vix1d_symbol_str=config.vix1d_symbol_str,
        vix_symbol_str=config.vix_symbol_str,
        vix3m_symbol_str=config.vix3m_symbol_str,
        long_threshold_float=config.long_threshold_float,
        short_threshold_float=config.short_threshold_float,
        target_weight_float=config.target_weight_float,
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
