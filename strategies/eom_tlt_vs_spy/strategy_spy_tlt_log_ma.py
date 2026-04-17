"""
SPY / TLT short-term log-ratio moving-average switch strategy.

TL;DR: At each close t, compute:

1. x_t = log(Close_SPY_t / Close_TLT_t)
2. m_t = trailing 5-day moving average of x_t
3. If x_t < m_t, target SPY at the next open.
4. If x_t > m_t, target TLT at the next open.
5. If x_t = m_t, keep the prior target asset.

This is the article-faithful interpretation of the second idea, implemented
under the repository's normal next-open execution contract.

Core formulas
-------------
Let:

    x_t
        = log(Close_SPY_t / Close_TLT_t)

For lookback L = 5:

    m_t
        = (1 / L) * sum_{j=0}^{L-1} x_{t-j}

Close-indexed desired asset:

    a_t
        = SPY, if x_t < m_t
        = TLT, if x_t > m_t
        = a_{t-1}, if x_t = m_t

Close-indexed target weights:

    w^{intent,SPY}_t
        = 1[a_t = SPY] * w

    w^{intent,TLT}_t
        = 1[a_t = TLT] * w

Execution mapping under the engine contract:

    order intent observed at close t
        -> fills at open t+1

So the live/backtest semantics are:

    position_{t+1} = a_t

Data note
---------
Tradeable ETFs use Norgate `CAPITALSPECIAL` prices for both signal formation
and execution. Benchmarks use Norgate benchmark-series settings through the
shared loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from IPython.display import display

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from data.norgate_loader import load_raw_prices


SIGNAL_NAMESPACE_STR = "signal_state"


def default_trade_id_int() -> int:
    return -1


@dataclass(frozen=True)
class SpyTltLogMaConfig:
    trade_symbol_list: tuple[str, ...] = ("SPY", "TLT")
    benchmark_list: tuple[str, ...] = ("$SPX",)
    log_ma_lookback_day_int: int = 5
    target_weight_float: float = 1.0
    start_date_str: str = "2002-07-01"
    end_date_str: str | None = None
    capital_base_float: float = 100_000.0
    slippage_float: float = 0.0001
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0

    def __post_init__(self):
        if set(self.trade_symbol_list) != {"SPY", "TLT"}:
            raise ValueError("trade_symbol_list must contain exactly SPY and TLT.")
        if len(set(self.trade_symbol_list).intersection(set(self.benchmark_list))) > 0:
            raise ValueError(
                "benchmark_list must not overlap trade_symbol_list because "
                "tradeable assets use CAPITALSPECIAL while benchmarks use benchmark settings."
            )
        if self.log_ma_lookback_day_int <= 1:
            raise ValueError("log_ma_lookback_day_int must be greater than 1.")
        if not np.isfinite(self.target_weight_float) or self.target_weight_float <= 0.0:
            raise ValueError("target_weight_float must be positive and finite.")
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


DEFAULT_CONFIG = SpyTltLogMaConfig()


def load_pricing_data_df(
    config: SpyTltLogMaConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    pricing_data_df = load_raw_prices(
        symbols=list(config.trade_symbol_list),
        benchmarks=list(config.benchmark_list),
        start_date=config.start_date_str,
        end_date=config.end_date_str,
    )
    if len(pricing_data_df) == 0:
        raise RuntimeError("Norgate returned no rows for the requested symbols.")

    pricing_data_df = pricing_data_df.sort_index()
    required_field_list = ["Open", "High", "Low", "Close"]
    trade_required_column_list = [
        (symbol_str, field_str)
        for symbol_str in config.trade_symbol_list
        for field_str in required_field_list
    ]

    # *** CRITICAL*** This strategy requires synchronized SPY/TLT OHLC bars.
    # Any row where either tradable ETF lacks required fields would corrupt
    # both the log-ratio signal and the next-open execution path.
    valid_trade_bar_mask_ser = pricing_data_df.loc[:, trade_required_column_list].notna().all(axis=1)
    pricing_data_df = pricing_data_df.loc[valid_trade_bar_mask_ser].copy()
    if len(pricing_data_df) == 0:
        raise RuntimeError("No synchronized SPY/TLT bars remain after dropping incomplete trade rows.")

    return pricing_data_df


def compute_spy_tlt_log_ma_signal_df(
    spy_close_ser: pd.Series,
    tlt_close_ser: pd.Series,
    log_ma_lookback_day_int: int,
    target_weight_float: float,
) -> pd.DataFrame:
    """
    Compute the close-indexed SPY/TLT log-ratio switch signal.

    The target weights in the returned DataFrame are decision-time weights
    observed at close t. Under the engine contract they become holdings at the
    next open t + 1.
    """
    if log_ma_lookback_day_int <= 1:
        raise ValueError("log_ma_lookback_day_int must be greater than 1.")
    if not np.isfinite(target_weight_float) or target_weight_float <= 0.0:
        raise ValueError("target_weight_float must be positive and finite.")

    signal_input_df = pd.concat(
        [
            pd.Series(spy_close_ser, copy=True).astype(float).rename("spy_close_ser"),
            pd.Series(tlt_close_ser, copy=True).astype(float).rename("tlt_close_ser"),
        ],
        axis=1,
    ).sort_index()

    invalid_price_mask_ser = (
        ~np.isfinite(signal_input_df["spy_close_ser"])
        | ~np.isfinite(signal_input_df["tlt_close_ser"])
        | (signal_input_df["spy_close_ser"] <= 0.0)
        | (signal_input_df["tlt_close_ser"] <= 0.0)
    )
    if bool(invalid_price_mask_ser.any()):
        first_invalid_bar_ts = pd.Timestamp(signal_input_df.index[invalid_price_mask_ser.argmax()])
        raise RuntimeError(f"Invalid SPY/TLT close inputs found at {first_invalid_bar_ts}.")

    log_ratio_ser = pd.Series(
        np.log(signal_input_df["spy_close_ser"] / signal_input_df["tlt_close_ser"]),
        index=signal_input_df.index,
        dtype=float,
        name="log_ratio_ser",
    )

    # *** CRITICAL*** This moving average must remain trailing-only:
    #
    #   m_t = (1 / L) * sum_{j=0}^{L-1} x_{t-j}
    #
    # Any centered or forward-looking window would leak future ratio values.
    log_ratio_ma_ser = log_ratio_ser.rolling(log_ma_lookback_day_int).mean()

    raw_desired_asset_obj_ser = pd.Series(pd.NA, index=signal_input_df.index, dtype=object, name="raw_desired_asset_obj_ser")
    raw_desired_asset_obj_ser.loc[log_ratio_ser < log_ratio_ma_ser] = "SPY"
    raw_desired_asset_obj_ser.loc[log_ratio_ser > log_ratio_ma_ser] = "TLT"

    # *** CRITICAL*** Exact equality keeps the prior state instead of forcing
    # a fresh arbitrary switch. This forward fill does not create look-ahead
    # because it only carries the last already-known desired asset forward.
    raw_desired_asset_obj_ser = raw_desired_asset_obj_ser.ffill()

    # *** CRITICAL*** This shift computes a_t-1 so regime_flip_t is evaluated
    # using only information available by the close of bar t. The engine then
    # converts that decision into an open t+1 fill.
    prior_desired_asset_obj_ser = raw_desired_asset_obj_ser.shift(1)

    prior_comparison_asset_obj_ser = prior_desired_asset_obj_ser.fillna("__FLAT__")
    current_comparison_asset_obj_ser = raw_desired_asset_obj_ser.fillna("__FLAT__")
    regime_flip_bool_ser = pd.Series(
        raw_desired_asset_obj_ser.notna() & current_comparison_asset_obj_ser.ne(prior_comparison_asset_obj_ser),
        index=signal_input_df.index,
        dtype=bool,
        name="regime_flip_bool_ser",
    )

    target_weight_spy_ser = pd.Series(
        np.where(raw_desired_asset_obj_ser == "SPY", float(target_weight_float), 0.0),
        index=signal_input_df.index,
        dtype=float,
        name="target_weight_spy_ser",
    )
    target_weight_tlt_ser = pd.Series(
        np.where(raw_desired_asset_obj_ser == "TLT", float(target_weight_float), 0.0),
        index=signal_input_df.index,
        dtype=float,
        name="target_weight_tlt_ser",
    )

    signal_feature_df = pd.DataFrame(
        {
            "spy_close_ser": signal_input_df["spy_close_ser"],
            "tlt_close_ser": signal_input_df["tlt_close_ser"],
            "log_ratio_ser": log_ratio_ser,
            "log_ratio_ma_ser": log_ratio_ma_ser,
            "raw_desired_asset_obj_ser": raw_desired_asset_obj_ser,
            "prior_desired_asset_obj_ser": prior_desired_asset_obj_ser,
            "regime_flip_bool_ser": regime_flip_bool_ser,
            "target_weight_spy_ser": target_weight_spy_ser,
            "target_weight_tlt_ser": target_weight_tlt_ser,
        },
        index=signal_input_df.index,
    )
    return signal_feature_df


def build_execution_target_weight_df(
    signal_feature_df: pd.DataFrame,
) -> pd.DataFrame:
    execution_target_weight_df = signal_feature_df[["target_weight_spy_ser", "target_weight_tlt_ser"]].copy()

    # *** CRITICAL*** Close-indexed target intent from bar t becomes the
    # realized target holding schedule for bar t+1 under the next-open engine
    # contract, so the report-facing weight path must be shifted by one bar.
    execution_target_weight_df = execution_target_weight_df.shift(1).fillna(0.0)
    execution_target_weight_df = execution_target_weight_df.rename(
        columns={
            "target_weight_spy_ser": "SPY",
            "target_weight_tlt_ser": "TLT",
        }
    )
    return execution_target_weight_df


class SpyTltLogMaStrategy(Strategy):
    """
    Daily SPY/TLT switch pod driven by the log-ratio versus its trailing mean.

    At close t:

        if log(SPY_t / TLT_t) < MA_t:
            target SPY
        else if log(SPY_t / TLT_t) > MA_t:
            target TLT

    At open t+1:

        close the prior ETF if needed
        allocate target_weight_float to the newly selected ETF
    """

    enable_signal_audit = True
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: list[str] | tuple[str, ...],
        trade_symbol_list: tuple[str, ...],
        log_ma_lookback_day_int: int,
        target_weight_float: float,
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
        if set(trade_symbol_list) != {"SPY", "TLT"}:
            raise ValueError("trade_symbol_list must contain exactly SPY and TLT.")
        if log_ma_lookback_day_int <= 1:
            raise ValueError("log_ma_lookback_day_int must be greater than 1.")
        if not np.isfinite(target_weight_float) or target_weight_float <= 0.0:
            raise ValueError("target_weight_float must be positive and finite.")

        self.trade_symbol_list = tuple(trade_symbol_list)
        self.log_ma_lookback_day_int = int(log_ma_lookback_day_int)
        self.target_weight_float = float(target_weight_float)
        self.trade_id_int = 0
        self.active_trade_id_map = {
            symbol_str: default_trade_id_int()
            for symbol_str in self.trade_symbol_list
        }

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        required_key_list = [
            ("SPY", "Close"),
            ("TLT", "Close"),
        ]
        missing_key_list = [key_tup for key_tup in required_key_list if key_tup not in pricing_data_df.columns]
        if len(missing_key_list) > 0:
            raise RuntimeError(f"Missing required signal columns: {missing_key_list}")

        signal_feature_df = compute_spy_tlt_log_ma_signal_df(
            spy_close_ser=pricing_data_df[("SPY", "Close")],
            tlt_close_ser=pricing_data_df[("TLT", "Close")],
            log_ma_lookback_day_int=self.log_ma_lookback_day_int,
            target_weight_float=self.target_weight_float,
        )
        signal_feature_df.columns = pd.MultiIndex.from_tuples(
            [(SIGNAL_NAMESPACE_STR, column_name_str) for column_name_str in signal_feature_df.columns]
        )
        return pd.concat([pricing_data_df, signal_feature_df], axis=1)

    def iterate(self, data_df: pd.DataFrame, close_row_ser: pd.Series, open_price_ser: pd.Series):
        if close_row_ser is None or data_df is None:
            return

        regime_flip_key = (SIGNAL_NAMESPACE_STR, "regime_flip_bool_ser")
        target_weight_spy_key = (SIGNAL_NAMESPACE_STR, "target_weight_spy_ser")
        target_weight_tlt_key = (SIGNAL_NAMESPACE_STR, "target_weight_tlt_ser")
        if (
            regime_flip_key not in close_row_ser.index
            or target_weight_spy_key not in close_row_ser.index
            or target_weight_tlt_key not in close_row_ser.index
        ):
            return

        current_target_weight_map = {
            "SPY": float(close_row_ser.loc[target_weight_spy_key]),
            "TLT": float(close_row_ser.loc[target_weight_tlt_key]),
        }
        current_position_map = {
            symbol_str: float(self.get_position(symbol_str))
            for symbol_str in self.trade_symbol_list
        }

        regime_flip_bool = bool(close_row_ser.loc[regime_flip_key])
        target_mismatch_bool = any(
            (
                np.isclose(current_target_weight_map[symbol_str], 0.0, atol=1e-12)
                and not np.isclose(current_position_map[symbol_str], 0.0, atol=1e-12)
            )
            or (
                current_target_weight_map[symbol_str] > 0.0
                and np.isclose(current_position_map[symbol_str], 0.0, atol=1e-12)
            )
            for symbol_str in self.trade_symbol_list
        )
        if not regime_flip_bool and not target_mismatch_bool:
            return

        desired_asset_str = None
        for symbol_str in self.trade_symbol_list:
            if current_target_weight_map[symbol_str] > 0.0:
                desired_asset_str = symbol_str
                break

        if desired_asset_str is not None:
            desired_open_price_float = float(open_price_ser.get(desired_asset_str, np.nan))
            if not np.isfinite(desired_open_price_float) or desired_open_price_float <= 0.0:
                raise RuntimeError(
                    f"Invalid open price for {desired_asset_str} on {self.current_bar}: {desired_open_price_float}"
                )

        for symbol_str in self.trade_symbol_list:
            current_target_weight_float = current_target_weight_map[symbol_str]
            current_position_float = current_position_map[symbol_str]
            if current_target_weight_float > 0.0:
                continue
            if np.isclose(current_position_float, 0.0, atol=1e-12):
                continue

            exit_trade_id_int = (
                None
                if self.active_trade_id_map[symbol_str] == default_trade_id_int()
                else self.active_trade_id_map[symbol_str]
            )
            self.order_target_value(
                symbol_str,
                0.0,
                trade_id=exit_trade_id_int,
            )
            self.active_trade_id_map[symbol_str] = default_trade_id_int()

        if desired_asset_str is None:
            return

        if self.active_trade_id_map[desired_asset_str] == default_trade_id_int():
            self.trade_id_int += 1
            self.active_trade_id_map[desired_asset_str] = self.trade_id_int

        # *** CRITICAL*** The selected ETF target is generated from the prior
        # close's signal and submitted now for next-open execution. There is no
        # same-bar use of current open information in the signal itself.
        self.order_target_percent(
            desired_asset_str,
            current_target_weight_map[desired_asset_str],
            trade_id=self.active_trade_id_map[desired_asset_str],
        )


if __name__ == "__main__":
    config = DEFAULT_CONFIG
    pricing_data_df = load_pricing_data_df(config=config)

    signal_feature_df = compute_spy_tlt_log_ma_signal_df(
        spy_close_ser=pricing_data_df[("SPY", "Close")],
        tlt_close_ser=pricing_data_df[("TLT", "Close")],
        log_ma_lookback_day_int=config.log_ma_lookback_day_int,
        target_weight_float=config.target_weight_float,
    )
    execution_target_weight_df = build_execution_target_weight_df(signal_feature_df=signal_feature_df)

    strategy = SpyTltLogMaStrategy(
        name="strategy_spy_tlt_log_ma",
        benchmarks=list(config.benchmark_list),
        trade_symbol_list=config.trade_symbol_list,
        log_ma_lookback_day_int=config.log_ma_lookback_day_int,
        target_weight_float=config.target_weight_float,
        capital_base=config.capital_base_float,
        slippage=config.slippage_float,
        commission_per_share=config.commission_per_share_float,
        commission_minimum=config.commission_minimum_float,
    )
    strategy.daily_target_weights = execution_target_weight_df.copy()

    run_daily(
        strategy,
        pricing_data_df,
        calendar=pricing_data_df.index,
        audit_override_bool=None,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print("Signal preview:")
    display(signal_feature_df.head(12))
    display(strategy.summary)
    display(strategy.summary_trades)

    save_results(strategy)
