"""
Configurable VIX contango VRP sleeve using a long-only inverse-volatility ETP.

TL;DR: Buy the inverse-volatility ETP at the next open when the prior close
shows both deep contango and elevated spot volatility.

Core formulas
-------------
At close t:

    ratio_t
        = VIX_t / VIX3M_t

    deep_contango_t
        = 1[ratio_t < theta_ratio]

    elevated_vix_t
        = 1[VIX_t >= theta_vix]

    short_vol_signal_t
        = deep_contango_t * elevated_vix_t

    target_weight_t
        = w_alloc * short_vol_signal_t

At the next open t + 1:

    q_{t+1}^{target}
        = floor(V_t * target_weight_t / O_{t+1}^{trade})

where:

    theta_ratio = 0.90 by default
    theta_vix   = 20.0 by default
    w_alloc     = 1.0 by default
    V_t         = prior-close total portfolio value
    O_{t+1}^{trade} = current-bar open of the inverse-volatility ETP

Instrument note
---------------
This sleeve trades a long position in an inverse-volatility ETP such as `SVIX`
or `SVXY`. It is not a direct short of spot VIX and it is not a borrow-based
short in a long-vol ETP.
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


def default_trade_id_int() -> int:
    return -1


@dataclass(frozen=True)
class VixContangoVrpConfig:
    trade_symbol_str: str = "SVXY"
    vix_symbol_str: str = "$VIX"
    vix3m_symbol_str: str = "$VIX3M"
    benchmark_symbol_str: str = "SPY"
    ratio_threshold_float: float = 0.90
    min_vix_level_float: float = 20.0
    target_weight_float: float = 1.00
    start_date_str: str = "2011-10-04"
    end_date_str: str | None = None
    capital_base_float: float = 10_000.0
    slippage_float: float = 0.001
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0

    def __post_init__(self):
        symbol_set = {
            self.trade_symbol_str,
            self.vix_symbol_str,
            self.vix3m_symbol_str,
            self.benchmark_symbol_str,
        }
        if len(symbol_set) != 4:
            raise ValueError("All symbols must be distinct.")
        if not np.isfinite(self.ratio_threshold_float) or self.ratio_threshold_float <= 0.0:
            raise ValueError("ratio_threshold_float must be positive and finite.")
        if not np.isfinite(self.min_vix_level_float) or self.min_vix_level_float <= 0.0:
            raise ValueError("min_vix_level_float must be positive and finite.")
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


DEFAULT_CONFIG = VixContangoVrpConfig()


def get_vix_contango_vrp_data(
    config: VixContangoVrpConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    return load_raw_prices(
        symbols=[config.trade_symbol_str, config.vix_symbol_str, config.vix3m_symbol_str],
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


def compute_vix_contango_vrp_signal_df(
    vix_close_ser: pd.Series,
    vix3m_close_ser: pd.Series,
    ratio_threshold_float: float,
    min_vix_level_float: float,
    target_weight_float: float,
) -> pd.DataFrame:
    """
    Compute the VIX contango VRP signal state from the VIX and VIX3M close series.

        ratio_t = VIX_t / VIX3M_t
        short_vol_signal_t = 1[ratio_t < theta_ratio] * 1[VIX_t >= theta_vix]

    The returned target weight is observed at close t and executed by the
    engine at the next open t + 1.
    """
    if not np.isfinite(ratio_threshold_float) or ratio_threshold_float <= 0.0:
        raise ValueError("ratio_threshold_float must be positive and finite.")
    if not np.isfinite(min_vix_level_float) or min_vix_level_float <= 0.0:
        raise ValueError("min_vix_level_float must be positive and finite.")
    if not np.isfinite(target_weight_float) or target_weight_float <= 0.0:
        raise ValueError("target_weight_float must be positive and finite.")

    vix_close_ser = pd.Series(vix_close_ser, copy=True).astype(float)
    vix3m_close_ser = pd.Series(vix3m_close_ser, copy=True).astype(float)

    signal_input_df = pd.concat(
        [
            vix_close_ser.rename("vix_close_ser"),
            vix3m_close_ser.rename("vix3m_close_ser"),
        ],
        axis=1,
    ).sort_index()

    vix_close_float_arr = signal_input_df["vix_close_ser"].to_numpy(dtype=float)
    vix3m_close_float_arr = signal_input_df["vix3m_close_ser"].to_numpy(dtype=float)
    vix_vix3m_ratio_float_arr = np.divide(
        vix_close_float_arr,
        vix3m_close_float_arr,
        out=np.full(len(signal_input_df.index), np.nan, dtype=float),
        where=vix3m_close_float_arr > 0.0,
    )
    vix_vix3m_ratio_ser = pd.Series(
        vix_vix3m_ratio_float_arr,
        index=signal_input_df.index,
        dtype=float,
        name="vix_vix3m_ratio_ser",
    )

    deep_contango_bool_ser = vix_vix3m_ratio_ser < float(ratio_threshold_float)
    elevated_vix_bool_ser = signal_input_df["vix_close_ser"] >= float(min_vix_level_float)
    short_vol_signal_bool_ser = deep_contango_bool_ser & elevated_vix_bool_ser
    target_weight_ser = pd.Series(
        np.where(short_vol_signal_bool_ser, float(target_weight_float), 0.0),
        index=signal_input_df.index,
        dtype=float,
        name="target_weight_ser",
    )

    signal_feature_df = pd.DataFrame(
        {
            "vix_close_ser": signal_input_df["vix_close_ser"],
            "vix3m_close_ser": signal_input_df["vix3m_close_ser"],
            "vix_vix3m_ratio_ser": vix_vix3m_ratio_ser,
            "deep_contango_bool": deep_contango_bool_ser.astype(bool),
            "elevated_vix_bool": elevated_vix_bool_ser.astype(bool),
            "short_vol_signal_bool": short_vol_signal_bool_ser.astype(bool),
            "target_weight_ser": target_weight_ser,
        },
        index=signal_input_df.index,
    )
    return signal_feature_df


class VixContangoVrpStrategy(Strategy):
    """
    Single-asset inverse-volatility sleeve driven by VIX term structure.

        ratio_t = VIX_t / VIX3M_t
        target_weight_t = w_alloc * 1[ratio_t < theta_ratio] * 1[VIX_t >= theta_vix]

    The target shares submitted at the next open are:

        q_{t+1}^{target} = floor(V_t * target_weight_t / O_{t+1}^{trade})
    """

    enable_signal_audit = True
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: list[str] | tuple[str, ...],
        trade_symbol_str: str,
        vix_symbol_str: str,
        vix3m_symbol_str: str,
        ratio_threshold_float: float,
        min_vix_level_float: float,
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
        if not trade_symbol_str or not vix_symbol_str or not vix3m_symbol_str:
            raise ValueError("Symbols must not be empty.")
        if not np.isfinite(ratio_threshold_float) or ratio_threshold_float <= 0.0:
            raise ValueError("ratio_threshold_float must be positive and finite.")
        if not np.isfinite(min_vix_level_float) or min_vix_level_float <= 0.0:
            raise ValueError("min_vix_level_float must be positive and finite.")
        if not np.isfinite(target_weight_float) or target_weight_float <= 0.0:
            raise ValueError("target_weight_float must be positive and finite.")

        self.trade_symbol_str = str(trade_symbol_str)
        self.vix_symbol_str = str(vix_symbol_str)
        self.vix3m_symbol_str = str(vix3m_symbol_str)
        self.ratio_threshold_float = float(ratio_threshold_float)
        self.min_vix_level_float = float(min_vix_level_float)
        self.target_weight_float = float(target_weight_float)
        self.trade_id_int = 0
        self.current_trade_id_int = default_trade_id_int()

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        required_key_list = [
            (self.trade_symbol_str, "Open"),
            (self.trade_symbol_str, "Close"),
            (self.vix_symbol_str, "Close"),
            (self.vix3m_symbol_str, "Close"),
        ]
        missing_key_list = [key_tup for key_tup in required_key_list if key_tup not in pricing_data_df.columns]
        if len(missing_key_list) > 0:
            raise RuntimeError(f"Missing required signal columns: {missing_key_list}")

        signal_data_df = pricing_data_df.copy()
        signal_feature_df = compute_vix_contango_vrp_signal_df(
            vix_close_ser=signal_data_df[(self.vix_symbol_str, "Close")],
            vix3m_close_ser=signal_data_df[(self.vix3m_symbol_str, "Close")],
            ratio_threshold_float=self.ratio_threshold_float,
            min_vix_level_float=self.min_vix_level_float,
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

        target_weight_key = (self.trade_symbol_str, "target_weight_ser")
        if target_weight_key not in close_row_ser.index:
            return

        target_weight_float = float(close_row_ser.loc[target_weight_key])
        if not np.isfinite(target_weight_float):
            return

        open_price_float = float(open_price_ser.get(self.trade_symbol_str, np.nan))
        current_share_int = int(self.get_position(self.trade_symbol_str))

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
        # This preserves next-open execution and avoids any same-bar close fill
        # fantasy or other time-series leakage.
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
    pricing_data_df = get_vix_contango_vrp_data(config=config)
    first_tradeable_bar_ts = get_first_tradeable_bar_ts(
        pricing_data_df=pricing_data_df,
        trade_symbol_str=config.trade_symbol_str,
    )
    calendar_idx = pricing_data_df.index[pricing_data_df.index >= first_tradeable_bar_ts]

    strategy = VixContangoVrpStrategy(
        name="strategy_mr_vix_contango_vrp",
        benchmarks=[config.benchmark_symbol_str],
        trade_symbol_str=config.trade_symbol_str,
        vix_symbol_str=config.vix_symbol_str,
        vix3m_symbol_str=config.vix3m_symbol_str,
        ratio_threshold_float=config.ratio_threshold_float,
        min_vix_level_float=config.min_vix_level_float,
        target_weight_float=config.target_weight_float,
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
