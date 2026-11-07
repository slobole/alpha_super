"""Adaptive Macro CORE6: CORE5 plus a half-size, volatility-sized Bitcoin overlay held through IBIT and funded from BIL.

PM_READY 2026-11-07 (capital x2.013, total-return benchmark gap 0.00%, deterministic;
results/research/pm_readiness/2026-11-07_122829). No live route. The rule of record is the Scout spec
alpha/scout/specs/core6.py (LIVE_CONFIG) and docs/research/CORE6_20261107.md; this module matches it (daily return
correlation 0.99986, the same BTC target on all 486 rebalance dates).

Rule (CORE5 untouched, imported from strategy_taa_adaptive_macro_core5):
    five sleeves   0.20 x long_i for SPY IEF GLD DBC UUP, the DBC short exactly as CORE5;
    IBIT           (1/6) x min(1, 0.10 / sigma60) x long_IBIT, taken from BIL;
                   long_IBIT is CORE5's adaptive switch on IBIT's own close, sigma60 the std of its last 60 session
                   returns x sqrt(252); 0 while either is not yet defined;
    no leverage    if all five are long the five are scaled down pro rata to make room;
    rebalance      any long-state change (IBIT included) or the last XNYS session of the month; Close_T shares, Open_(T+1) fills.

Data: CORE5's Norgate frame plus an "IBIT" column. From 2024-01-11 it is Norgate IBIT (CAPITALSPECIAL). Before that it
is Bitcoin from Pakal's cache (research_cache/core6_qualification_study/btc_unit.parquet: Binance 09:30/16:00 ET bars
from 2017-08-17, the CoinMetrics end-of-day stamp before that), divided by the IBIT/BTC ratio on the listing day, so
returns are Bitcoin's and prices are on IBIT's scale. Before 2024 IBIT therefore stands for spot Bitcoin, which did not
trade as an ETF; the price level below a cent there is why IBIT pays a notional fee instead of a per-share one.

Costs: the house slippage on every asset; IBIT pays max($1, 12.5 bp of filled notional) instead of $0.005 a share,
about the 15 bp a side of the study.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from data.norgate_loader import load_raw_prices
from strategies.taa_beyond_6040 import strategy_taa_adaptive_macro_core5 as core5_module


STRATEGY_NAME_STR = "strategy_taa_adaptive_macro_core6"
STRATEGY_DISPLAY_NAME_STR = "Adaptive Macro CORE6"
CRYPTO_ASSET_STR = "IBIT"
IBIT_LISTING_TS = pd.Timestamp("2024-01-11")
BTC_UNIT_PATH = Path(
    r"C:\Users\User\Documents\workspace\pakal\pakal-research\research_cache\core6_qualification_study\btc_unit.parquet"
)
CRYPTO_VOL_FIELD_STR = "crypto_volatility_ser"


@dataclass(frozen=True)
class AdaptiveMacroCore6Config(core5_module.AdaptiveMacroCore5Config):
    strategy_name_str: str = STRATEGY_NAME_STR
    crypto_asset_str: str = CRYPTO_ASSET_STR
    crypto_sleeve_fraction_float: float = 1.0 / 6.0
    crypto_vol_target_float: float = 0.10
    crypto_vol_lookback_int: int = 60
    crypto_fee_rate_float: float = 0.00125

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.crypto_asset_str in self.risk_asset_tuple or self.crypto_asset_str == self.reserve_asset_str:
            raise ValueError("crypto_asset_str must be a separate asset.")
        if not 0.0 < self.crypto_sleeve_fraction_float < 1.0 or self.crypto_vol_target_float <= 0.0:
            raise ValueError("crypto sleeve fraction must be in (0, 1) and its volatility target positive.")
        if self.crypto_vol_lookback_int <= 1 or self.crypto_fee_rate_float < 0.0:
            raise ValueError("crypto volatility lookback must exceed one and its fee must be non-negative.")


DEFAULT_CONFIG = AdaptiveMacroCore6Config()


def load_crypto_price_df(config_obj: AdaptiveMacroCore6Config = DEFAULT_CONFIG, unit_path: Path = BTC_UNIT_PATH) -> pd.DataFrame:
    """IBIT execution fields: Norgate IBIT from its listing, Bitcoin on IBIT's price scale before it."""
    unit_df = pd.read_parquet(unit_path)
    unit_df.index = pd.DatetimeIndex(unit_df.index).tz_localize(None).normalize()
    ibit_df = load_raw_prices(symbols=[config_obj.crypto_asset_str], benchmarks=[], start_date="2024-01-01",
                              end_date=config_obj.end_date_str)[config_obj.crypto_asset_str]
    ibit_df.index = pd.DatetimeIndex(ibit_df.index).tz_localize(None).normalize()
    ibit_df = ibit_df.loc[IBIT_LISTING_TS:].dropna(subset=["Open", "Close"])
    level_float = float(unit_df["BTC_close"].loc[IBIT_LISTING_TS] / ibit_df["Close"].loc[IBIT_LISTING_TS])
    before_df = unit_df.loc[unit_df.index < IBIT_LISTING_TS, ["BTC_open", "BTC_close"]].div(level_float)
    before_df.columns = ["Open", "Close"]
    before_df["High"] = before_df[["Open", "Close"]].max(axis=1)
    before_df["Low"] = before_df[["Open", "Close"]].min(axis=1)
    before_df["Unadjusted Close"] = before_df["Close"]
    before_df["Dividend"] = 0.0
    # No ETF volume before the listing: Volume and Turnover stay empty there.
    return pd.concat([before_df, ibit_df[list(before_df.columns) + ["Volume", "Turnover"]]]).sort_index()


def get_adaptive_macro_core6_data(config_obj: AdaptiveMacroCore6Config = DEFAULT_CONFIG,
                                  unit_path: Path = BTC_UNIT_PATH) -> pd.DataFrame:
    """CORE5's frame plus IBIT execution fields and IBIT's own close as its signal close."""
    pricing_data_df = core5_module.get_adaptive_macro_core5_data(config_obj=config_obj)
    crypto_df = load_crypto_price_df(config_obj, unit_path).reindex(pricing_data_df.index)
    crypto_str = config_obj.crypto_asset_str
    for field_str in crypto_df.columns:
        pricing_data_df[(crypto_str, field_str)] = crypto_df[field_str]
    # *** CRITICAL*** the IBIT switch reads the same 16:00 ET close the order is sized from; nothing later in the day.
    pricing_data_df[(core5_module.signal_namespace_str(crypto_str), "Close")] = crypto_df["Close"]
    pricing_data_df = pricing_data_df.sort_index(axis=1)
    pricing_data_df.attrs.setdefault("norgate_adjustment_by_symbol_dict", {})[crypto_str] = "CAPITALSPECIAL"
    pricing_data_df.attrs.setdefault("signal_adjustment_by_symbol_dict", {})[
        core5_module.signal_namespace_str(crypto_str)] = "CAPITALSPECIAL"
    return pricing_data_df


class AdaptiveMacroCore6Strategy(core5_module.AdaptiveMacroCore5Strategy):
    """CORE5 with a vol-sized IBIT overlay taken from BIL."""

    def __init__(self, name: str = STRATEGY_NAME_STR, benchmarks: Sequence[str] | None = None,
                 config_obj: AdaptiveMacroCore6Config = DEFAULT_CONFIG) -> None:
        super().__init__(name=name, benchmarks=benchmarks, config_obj=config_obj)
        self.crypto_asset_str = config_obj.crypto_asset_str
        self.crypto_namespace_str = core5_module.signal_namespace_str(self.crypto_asset_str)
        self.current_trade_id_map[self.crypto_asset_str] = core5_module.default_trade_id_int()

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        signal_df = super().compute_signals(pricing_data_df)
        close_ser = pricing_data_df[(self.crypto_namespace_str, "Close")].astype(float)
        crypto_signal_df = core5_module.compute_adaptive_asset_signal_df(close_ser, config_obj=self.config_obj)
        observed_ser = close_ser.dropna()
        # *** CRITICAL*** rolling boundary: the last 60 session returns ending at Close_T.
        volatility_ser = (observed_ser.pct_change(fill_method=None)
                          .rolling(self.config_obj.crypto_vol_lookback_int, min_periods=self.config_obj.crypto_vol_lookback_int)
                          .std(ddof=1).mul(np.sqrt(252.0)).reindex(close_ser.index))
        long_ser = crypto_signal_df["long_state_ser"]
        signal_df[(self.crypto_namespace_str, "long_state_ser")] = long_ser
        signal_df[(self.crypto_namespace_str, CRYPTO_VOL_FIELD_STR)] = volatility_ser
        changed_key_tuple = (core5_module.PORTFOLIO_NAMESPACE_STR, core5_module.LONG_STATE_CHANGED_FIELD_STR)
        # *** CRITICAL*** Close_T against Close_(T-1) only; an undefined switch counts as out.
        crypto_changed_ser = long_ser.fillna(0.0).diff().abs().fillna(0.0).gt(0.0)
        signal_df[changed_key_tuple] = signal_df[changed_key_tuple].astype(bool) | crypto_changed_ser
        return signal_df

    def _crypto_close_float(self, close_row_ser: pd.Series) -> float:
        return float(close_row_ser.get((self.crypto_asset_str, "Close"), np.nan))

    def _validate_required_close_prices(self, close_row_ser: pd.Series) -> None:
        super()._validate_required_close_prices(close_row_ser)
        crypto_close_float = self._crypto_close_float(close_row_ser)
        if self.get_position(self.crypto_asset_str) != 0 and not (np.isfinite(crypto_close_float) and crypto_close_float > 0.0):
            raise RuntimeError(f"Held {self.crypto_asset_str} has no valid close at Close_{self.previous_bar}.")

    def _target_weight_ser(self, close_row_ser: pd.Series, long_state_ser: pd.Series) -> pd.Series:
        target_weight_ser = super()._target_weight_ser(close_row_ser=close_row_ser, long_state_ser=long_state_ser)
        config_obj = self.config_obj
        crypto_long_float = float(close_row_ser.get((self.crypto_namespace_str, "long_state_ser"), np.nan))
        crypto_vol_float = float(close_row_ser.get((self.crypto_namespace_str, CRYPTO_VOL_FIELD_STR), np.nan))
        crypto_close_float = self._crypto_close_float(close_row_ser)
        crypto_weight_float = 0.0
        if (crypto_long_float == 1.0 and np.isfinite(crypto_vol_float) and crypto_vol_float > 0.0
                and np.isfinite(crypto_close_float) and crypto_close_float > 0.0):
            crypto_weight_float = config_obj.crypto_sleeve_fraction_float * min(1.0, config_obj.crypto_vol_target_float / crypto_vol_float)
        risk_list = list(config_obj.risk_asset_tuple)
        long_weight_ser = target_weight_ser.loc[risk_list].clip(lower=0.0)
        if long_weight_ser.sum() + crypto_weight_float > 1.0:  # no leverage: make room inside the five sleeves
            scaled_ser = long_weight_ser * (1.0 - crypto_weight_float) / long_weight_ser.sum()
            target_weight_ser.loc[scaled_ser.index[scaled_ser > 0.0]] = scaled_ser[scaled_ser > 0.0]
            long_weight_ser = scaled_ser
        target_weight_ser.loc[config_obj.reserve_asset_str] = float(1.0 - long_weight_ser.sum() - crypto_weight_float)
        target_weight_ser.loc[self.crypto_asset_str] = crypto_weight_float
        long_book_float = float(target_weight_ser.drop(labels=["Cash"]).clip(lower=0.0).sum())
        if not np.isclose(long_book_float, 1.0, atol=1e-12) or target_weight_ser.loc[config_obj.reserve_asset_str] < -1e-12:
            raise ValueError("CORE6 long targets must sum to 100% of NAV with a non-negative BIL reserve.")
        return target_weight_ser

    def _submit_target_orders(self, target_weight_ser: pd.Series, close_row_ser: pd.Series) -> None:
        crypto_close_float = self._crypto_close_float(close_row_ser)
        tradable_bool = bool(np.isfinite(crypto_close_float) and crypto_close_float > 0.0)
        core_asset_list = self.asset_list
        if tradable_bool:
            self.asset_list = core_asset_list + [self.crypto_asset_str]
        try:
            super()._submit_target_orders(target_weight_ser=target_weight_ser, close_row_ser=close_row_ser)
        finally:
            self.asset_list = core_asset_list

    def _compute_execution_commission_float(self, prices_df, asset_str, amount_float, anchor_bar_ts) -> float:
        if asset_str != self.crypto_asset_str:
            return super()._compute_execution_commission_float(prices_df, asset_str, amount_float, anchor_bar_ts)
        fill_price_float = float(prices_df.loc[anchor_bar_ts, (asset_str, "Open")])
        return max(self._commission_minimum, abs(float(amount_float)) * fill_price_float * self.config_obj.crypto_fee_rate_float)


def _run_strategy(config_obj: AdaptiveMacroCore6Config, pricing_data_df: pd.DataFrame,
                  backtest_start_date_str: str | None, show_progress_bool: bool) -> AdaptiveMacroCore6Strategy:
    calendar_idx = core5_module.build_execution_calendar_idx(
        pricing_data_df=pricing_data_df, config_obj=config_obj, backtest_start_date_str=backtest_start_date_str,
    )
    strategy_obj = AdaptiveMacroCore6Strategy(name=config_obj.strategy_name_str, benchmarks=config_obj.benchmark_list,
                                              config_obj=config_obj)
    strategy_obj.configure_run_calendar(calendar_idx)
    run_daily(strategy_obj, pricing_data_df, calendar=calendar_idx, show_progress=show_progress_bool,
              show_signal_progress_bool=show_progress_bool, audit_override_bool=None)
    return strategy_obj


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = DEFAULT_CONFIG.backtest_start_date_str,
    capital_base_float: float = DEFAULT_CONFIG.capital_base_float,
    end_date_str: str | None = None,
    pricing_data_df: pd.DataFrame | None = None,
) -> AdaptiveMacroCore6Strategy:
    """Run the CORE6 research strategy in the house engine."""
    config_obj = replace(DEFAULT_CONFIG, capital_base_float=float(capital_base_float), end_date_str=end_date_str)
    if pricing_data_df is None:
        pricing_data_df = get_adaptive_macro_core6_data(config_obj=config_obj)
    strategy_obj = _run_strategy(config_obj, pricing_data_df, backtest_start_date_str, show_display_bool)
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
