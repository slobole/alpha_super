"""
SPY canary strategy using SPHB/SPLV 2-day confirmation.

Core formulas
-------------
At close t:

    ratio_t
        = Close(SPHB)_t / Close(SPLV)_t

    sma_t
        = (1 / L) * sum_{k=0}^{L-1} ratio_{t-k}

    momentum_t
        = ratio_t / sma_t - 1

    target_weight_t
        = 1[momentum_t > 0 and momentum_{t-1} > 0]

Execution uses the Vanilla engine contract:

    signal at close t -> order intent on t + 1 -> fill at open t + 1

Risk-on holds `trade_symbol_str`, which defaults to SPY. By default, risk-off
holds cash. A variant can set a `risk_off_symbol_str` such as IAU to rotate
fully into that asset instead. The default L is 252 trading days, which is the
daily-data approximation of the article's 12-month SMA momentum rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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
class SphbSplvCanaryConfig:
    trade_symbol_str: str = "SPY"
    risk_off_symbol_str: str | None = None
    high_beta_symbol_str: str = "SPHB"
    low_vol_symbol_str: str = "SPLV"
    benchmark_list: tuple[str, ...] = ("$SPXTR",)
    start_date_str: str = "2011-05-05"
    end_date_str: str | None = None
    sma_lookback_day_int: int = 252
    confirmation_day_int: int = 2
    capital_base_float: float = 100_000.0
    slippage_float: float = 0.00025
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0

    def __post_init__(self) -> None:
        if not self.trade_symbol_str:
            raise ValueError("trade_symbol_str must not be empty.")
        if self.risk_off_symbol_str is not None and not self.risk_off_symbol_str:
            raise ValueError("risk_off_symbol_str must not be empty when provided.")
        if not self.high_beta_symbol_str:
            raise ValueError("high_beta_symbol_str must not be empty.")
        if not self.low_vol_symbol_str:
            raise ValueError("low_vol_symbol_str must not be empty.")
        if self.high_beta_symbol_str == self.low_vol_symbol_str:
            raise ValueError("high_beta_symbol_str must differ from low_vol_symbol_str.")
        if self.trade_symbol_str in {self.high_beta_symbol_str, self.low_vol_symbol_str}:
            raise ValueError("trade_symbol_str must differ from canary signal symbols.")
        if self.risk_off_symbol_str is not None:
            if self.risk_off_symbol_str == self.trade_symbol_str:
                raise ValueError("risk_off_symbol_str must differ from trade_symbol_str.")
            if self.risk_off_symbol_str in {self.high_beta_symbol_str, self.low_vol_symbol_str}:
                raise ValueError("risk_off_symbol_str must differ from canary signal symbols.")
        if len(set(self.benchmark_list)) != len(self.benchmark_list):
            raise ValueError("benchmark_list contains duplicate symbols.")
        if self.sma_lookback_day_int <= 0:
            raise ValueError("sma_lookback_day_int must be positive.")
        if self.confirmation_day_int <= 0:
            raise ValueError("confirmation_day_int must be positive.")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if self.slippage_float < 0.0:
            raise ValueError("slippage_float must be non-negative.")
        if self.commission_per_share_float < 0.0:
            raise ValueError("commission_per_share_float must be non-negative.")
        if self.commission_minimum_float < 0.0:
            raise ValueError("commission_minimum_float must be non-negative.")

    @property
    def symbol_list(self) -> list[str]:
        symbol_list = [
            self.trade_symbol_str,
            self.high_beta_symbol_str,
            self.low_vol_symbol_str,
        ]
        if self.risk_off_symbol_str is not None:
            symbol_list.append(self.risk_off_symbol_str)
        return symbol_list


DEFAULT_CONFIG = SphbSplvCanaryConfig()

__all__ = [
    "DEFAULT_CONFIG",
    "SphbSplvCanaryConfig",
    "SphbSplvCanaryStrategy",
    "build_backtest_calendar_idx",
    "compute_sphb_splv_canary_signal_df",
    "get_sphb_splv_canary_data",
    "run_variant",
]


def get_sphb_splv_canary_data(
    config: SphbSplvCanaryConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    return load_raw_prices(
        symbols=config.symbol_list,
        benchmarks=list(config.benchmark_list),
        start_date=config.start_date_str,
        end_date=config.end_date_str,
    )


def compute_sphb_splv_canary_signal_df(
    high_beta_close_ser: pd.Series,
    low_vol_close_ser: pd.Series,
    sma_lookback_day_int: int = DEFAULT_CONFIG.sma_lookback_day_int,
    confirmation_day_int: int = DEFAULT_CONFIG.confirmation_day_int,
) -> pd.DataFrame:
    """
    Compute the SPHB/SPLV canary signal.

    The binary target is:

        target_weight_t = 1[momentum_t > 0 for the last C closes]

    where C is `confirmation_day_int`.
    """
    if sma_lookback_day_int <= 0:
        raise ValueError("sma_lookback_day_int must be positive.")
    if confirmation_day_int <= 0:
        raise ValueError("confirmation_day_int must be positive.")

    high_beta_close_ser = pd.Series(high_beta_close_ser, copy=True).astype(float)
    low_vol_close_ser = pd.Series(low_vol_close_ser, copy=True).astype(float)

    # *** CRITICAL*** The canary ratio uses same-row completed closes only.
    # In the engine this row is previous_bar, so orders created from it fill at
    # the next open. Any future close in this ratio is a lookahead bug.
    canary_ratio_ser = high_beta_close_ser / low_vol_close_ser

    # *** CRITICAL*** The SMA must be a trailing window over already-observed
    # canary ratios. A centered or forward window would leak future risk state.
    canary_sma_ser = canary_ratio_ser.rolling(
        window=sma_lookback_day_int,
        min_periods=sma_lookback_day_int,
    ).mean()
    canary_momentum_ser = canary_ratio_ser / canary_sma_ser - 1.0

    positive_momentum_ser = pd.Series(np.nan, index=canary_momentum_ser.index, dtype=float)
    finite_momentum_mask_ser = canary_momentum_ser.notna()
    positive_momentum_ser.loc[finite_momentum_mask_ser] = (
        canary_momentum_ser.loc[finite_momentum_mask_ser] > 0.0
    ).astype(float)

    # *** CRITICAL*** Confirmation is a trailing count of positive completed
    # momentum observations. Requiring the last C rows prevents same-day peeking.
    confirmation_count_ser = positive_momentum_ser.rolling(
        window=confirmation_day_int,
        min_periods=confirmation_day_int,
    ).sum()
    target_weight_ser = (confirmation_count_ser == float(confirmation_day_int)).astype(float)
    target_weight_ser = target_weight_ser.where(confirmation_count_ser.notna())

    return pd.DataFrame(
        {
            "canary_ratio_ser": canary_ratio_ser,
            "canary_sma_ser": canary_sma_ser,
            "canary_momentum_ser": canary_momentum_ser,
            "positive_momentum_ser": positive_momentum_ser,
            "confirmation_count_ser": confirmation_count_ser,
            "target_weight_ser": target_weight_ser,
        },
        index=canary_ratio_ser.index,
    )


def build_backtest_calendar_idx(
    pricing_data_df: pd.DataFrame,
    signal_feature_df: pd.DataFrame,
) -> pd.DatetimeIndex:
    """
    Start the executable calendar at the first bar after a valid signal.
    """
    valid_signal_index = signal_feature_df.index[signal_feature_df["target_weight_ser"].notna()]
    if len(valid_signal_index) == 0:
        raise RuntimeError("No valid SPHB/SPLV canary target weights were generated.")

    first_signal_ts = pd.Timestamp(valid_signal_index[0])
    first_signal_pos_int = int(pricing_data_df.index.get_loc(first_signal_ts))
    first_calendar_pos_int = first_signal_pos_int + 1
    if first_calendar_pos_int >= len(pricing_data_df.index):
        raise RuntimeError("No executable bar exists after the first valid canary signal.")

    return pd.DatetimeIndex(pricing_data_df.index[first_calendar_pos_int:])


class SphbSplvCanaryStrategy(Strategy):
    """
    Single risk-on asset strategy gated by the SPHB/SPLV canary.

    At close t:

        w_t = 1[momentum_t > 0 and momentum_{t-1} > 0]

    At the next open:

        if w_t == 1, hold trade_symbol_str
        if w_t == 0 and risk_off_symbol_str is None, hold cash
        if w_t == 0 and risk_off_symbol_str is set, hold that asset

    The strategy does not rebalance every day while already long.
    """

    enable_signal_audit = True
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        trade_symbol_str: str = DEFAULT_CONFIG.trade_symbol_str,
        risk_off_symbol_str: str | None = DEFAULT_CONFIG.risk_off_symbol_str,
        high_beta_symbol_str: str = DEFAULT_CONFIG.high_beta_symbol_str,
        low_vol_symbol_str: str = DEFAULT_CONFIG.low_vol_symbol_str,
        sma_lookback_day_int: int = DEFAULT_CONFIG.sma_lookback_day_int,
        confirmation_day_int: int = DEFAULT_CONFIG.confirmation_day_int,
        capital_base: float = DEFAULT_CONFIG.capital_base_float,
        slippage: float = DEFAULT_CONFIG.slippage_float,
        commission_per_share: float = DEFAULT_CONFIG.commission_per_share_float,
        commission_minimum: float = DEFAULT_CONFIG.commission_minimum_float,
    ) -> None:
        super().__init__(
            name=name,
            benchmarks=list(benchmarks),
            capital_base=capital_base,
            slippage=slippage,
            commission_per_share=commission_per_share,
            commission_minimum=commission_minimum,
        )
        if not trade_symbol_str:
            raise ValueError("trade_symbol_str must not be empty.")
        if risk_off_symbol_str is not None and not risk_off_symbol_str:
            raise ValueError("risk_off_symbol_str must not be empty when provided.")
        if risk_off_symbol_str is not None and risk_off_symbol_str == trade_symbol_str:
            raise ValueError("risk_off_symbol_str must differ from trade_symbol_str.")
        if not high_beta_symbol_str:
            raise ValueError("high_beta_symbol_str must not be empty.")
        if not low_vol_symbol_str:
            raise ValueError("low_vol_symbol_str must not be empty.")
        if risk_off_symbol_str is not None and risk_off_symbol_str in {
            high_beta_symbol_str,
            low_vol_symbol_str,
        }:
            raise ValueError("risk_off_symbol_str must differ from canary signal symbols.")
        if sma_lookback_day_int <= 0:
            raise ValueError("sma_lookback_day_int must be positive.")
        if confirmation_day_int <= 0:
            raise ValueError("confirmation_day_int must be positive.")

        self.trade_symbol_str = str(trade_symbol_str)
        self.risk_off_symbol_str = str(risk_off_symbol_str) if risk_off_symbol_str is not None else None
        self.high_beta_symbol_str = str(high_beta_symbol_str)
        self.low_vol_symbol_str = str(low_vol_symbol_str)
        self.sma_lookback_day_int = int(sma_lookback_day_int)
        self.confirmation_day_int = int(confirmation_day_int)
        self.trade_id_int = 0
        self.current_trade_id_int = default_trade_id_int()
        self.current_trade_id_map = {self.trade_symbol_str: default_trade_id_int()}
        if self.risk_off_symbol_str is not None:
            self.current_trade_id_map[self.risk_off_symbol_str] = default_trade_id_int()

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        high_beta_close_key = (self.high_beta_symbol_str, "Close")
        low_vol_close_key = (self.low_vol_symbol_str, "Close")
        trade_close_key = (self.trade_symbol_str, "Close")

        for required_key in [high_beta_close_key, low_vol_close_key, trade_close_key]:
            if required_key not in pricing_data_df.columns:
                raise RuntimeError(f"Missing required close column: {required_key}")
        if self.risk_off_symbol_str is not None:
            risk_off_close_key = (self.risk_off_symbol_str, "Close")
            if risk_off_close_key not in pricing_data_df.columns:
                raise RuntimeError(f"Missing required close column: {risk_off_close_key}")

        signal_data_df = pricing_data_df.copy()
        signal_feature_df = compute_sphb_splv_canary_signal_df(
            high_beta_close_ser=signal_data_df.loc[:, high_beta_close_key],
            low_vol_close_ser=signal_data_df.loc[:, low_vol_close_key],
            sma_lookback_day_int=self.sma_lookback_day_int,
            confirmation_day_int=self.confirmation_day_int,
        )

        canary_feature_df = signal_feature_df.drop(columns=["target_weight_ser"])
        canary_feature_df.columns = pd.MultiIndex.from_tuples(
            [(self.high_beta_symbol_str, field_str) for field_str in canary_feature_df.columns]
        )
        target_weight_df = signal_feature_df.loc[:, ["target_weight_ser"]].copy()
        target_weight_df.columns = pd.MultiIndex.from_tuples(
            [(self.trade_symbol_str, "target_weight_ser")]
        )

        return pd.concat([signal_data_df, canary_feature_df, target_weight_df], axis=1).sort_index(axis=1)

    def iterate(
        self,
        data_df: pd.DataFrame,
        close_row_ser: pd.Series,
        open_price_ser: pd.Series,
    ) -> None:
        if data_df is None or close_row_ser is None:
            return

        target_weight_key = (self.trade_symbol_str, "target_weight_ser")
        if target_weight_key not in close_row_ser.index:
            return

        target_weight_float = float(close_row_ser.loc[target_weight_key])
        if not np.isfinite(target_weight_float):
            return

        target_asset_str = self.trade_symbol_str if target_weight_float >= 1.0 else self.risk_off_symbol_str
        managed_asset_list = [self.trade_symbol_str]
        if self.risk_off_symbol_str is not None:
            managed_asset_list.append(self.risk_off_symbol_str)

        for managed_asset_str in managed_asset_list:
            if managed_asset_str == target_asset_str:
                continue
            current_share_int = int(self.get_position(managed_asset_str))
            if current_share_int <= 0:
                continue
            exit_trade_id_int = self.current_trade_id_map.get(managed_asset_str, default_trade_id_int())
            if (
                managed_asset_str == self.trade_symbol_str
                and exit_trade_id_int == default_trade_id_int()
                and self.current_trade_id_int != default_trade_id_int()
            ):
                exit_trade_id_int = self.current_trade_id_int
            self.order_target_percent(
                managed_asset_str,
                0.0,
                trade_id=exit_trade_id_int,
            )
            self.current_trade_id_map[managed_asset_str] = default_trade_id_int()
            if managed_asset_str == self.trade_symbol_str:
                self.current_trade_id_int = default_trade_id_int()

        if target_asset_str is None:
            return

        if int(self.get_position(target_asset_str)) > 0:
            return

        target_open_price_float = float(open_price_ser.get(target_asset_str, np.nan))
        if not np.isfinite(target_open_price_float) or target_open_price_float <= 0.0:
            raise RuntimeError(f"Invalid open price for {target_asset_str} on {self.current_bar}.")

        self.trade_id_int += 1
        self.current_trade_id_map[target_asset_str] = self.trade_id_int
        if target_asset_str == self.trade_symbol_str:
            self.current_trade_id_int = self.trade_id_int
        self.order_target_percent(
            target_asset_str,
            1.0,
            trade_id=self.trade_id_int,
        )


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    capital_base_float: float = DEFAULT_CONFIG.capital_base_float,
    end_date_str: str | None = None,
    trade_symbol_str: str = DEFAULT_CONFIG.trade_symbol_str,
    risk_off_symbol_str: str | None = DEFAULT_CONFIG.risk_off_symbol_str,
    strategy_name_str: str = "strategy_mo_spy_sphb_splv_canary",
) -> SphbSplvCanaryStrategy:
    config = SphbSplvCanaryConfig(
        trade_symbol_str=trade_symbol_str,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
        risk_off_symbol_str=risk_off_symbol_str,
    )
    pricing_data_df = get_sphb_splv_canary_data(config=config)
    signal_feature_df = compute_sphb_splv_canary_signal_df(
        high_beta_close_ser=pricing_data_df.loc[:, (config.high_beta_symbol_str, "Close")],
        low_vol_close_ser=pricing_data_df.loc[:, (config.low_vol_symbol_str, "Close")],
        sma_lookback_day_int=config.sma_lookback_day_int,
        confirmation_day_int=config.confirmation_day_int,
    )
    calendar_idx = build_backtest_calendar_idx(
        pricing_data_df=pricing_data_df,
        signal_feature_df=signal_feature_df,
    )

    strategy_obj = SphbSplvCanaryStrategy(
        name=strategy_name_str,
        benchmarks=config.benchmark_list,
        trade_symbol_str=config.trade_symbol_str,
        risk_off_symbol_str=config.risk_off_symbol_str,
        high_beta_symbol_str=config.high_beta_symbol_str,
        low_vol_symbol_str=config.low_vol_symbol_str,
        sma_lookback_day_int=config.sma_lookback_day_int,
        confirmation_day_int=config.confirmation_day_int,
        capital_base=config.capital_base_float,
        slippage=config.slippage_float,
        commission_per_share=config.commission_per_share_float,
        commission_minimum=config.commission_minimum_float,
    )

    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar=calendar_idx,
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
