"""
Growth and inflation sector timing strategy.

TL;DR: classify each daily close into one of four macro regimes using a
trend-based growth proxy and a market-implied inflation ratio, then rotate into
one sector at the next open.

Core formulas
-------------
Let L_g be the growth SMA lookback and L_pi be the inflation-median lookback.

Growth proxy:

    growth_sma_t
        = (1 / L_g) * sum_{k=0}^{L_g - 1} SPY^{sig}_{t-k}

    growth_rising_t
        = 1[SPY^{sig}_t > growth_sma_t]

Positive expected-inflation beta basket:

    pos_t
        = 0.50 * XLE^{sig}_t
        + (1 / 6) * XLI^{sig}_t
        + (1 / 6) * XLF^{sig}_t
        + (1 / 6) * XLB^{sig}_t

Negative expected-inflation beta basket:

    neg_t
        = (1 / 3) * XLU^{sig}_t
        + (1 / 3) * XLP^{sig}_t
        + (1 / 3) * XLV^{sig}_t

Inflation proxy:

    inflation_ratio_t
        = pos_t / neg_t

    inflation_median_t
        = median(inflation_ratio_{t-L_pi+1}, ..., inflation_ratio_t)

    inflation_rising_t
        = 1[inflation_ratio_t > inflation_median_t]

Regime map:

    if growth_rising_t and not inflation_rising_t: target = XLK
    if growth_rising_t and inflation_rising_t:     target = XLE
    if not growth_rising_t and inflation_rising_t: target = XLV
    if not growth_rising_t and not inflation_rising_t: target = XLP

Engine timing:

    decision_t = f(close_t)

    fill_{t+1}
        = open_{t+1}

This preserves the repository's causal contract. The strategy does not assume
same-bar close fills.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd
import norgatedata
from IPython.display import display
from tqdm.auto import tqdm

WORKSPACE_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT_PATH))

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from data.norgate_loader import load_raw_prices


MODEL_SYMBOL_STR = "_MODEL"


def default_trade_id_int() -> int:
    return -1


@dataclass(frozen=True)
class BasketWeightSpec:
    asset_str: str
    weight_float: float


@dataclass(frozen=True)
class GrowthInflationSectorConfig:
    growth_asset_str: str = "SPY"
    positive_beta_basket_tuple: tuple[BasketWeightSpec, ...] = (
        BasketWeightSpec("XLE", 0.50),
        BasketWeightSpec("XLI", 1.0 / 6.0),
        BasketWeightSpec("XLF", 1.0 / 6.0),
        BasketWeightSpec("XLB", 1.0 / 6.0),
    )
    negative_beta_basket_tuple: tuple[BasketWeightSpec, ...] = (
        BasketWeightSpec("XLU", 1.0 / 3.0),
        BasketWeightSpec("XLP", 1.0 / 3.0),
        BasketWeightSpec("XLV", 1.0 / 3.0),
    )
    goldilocks_asset_str: str = "XLK"
    reflation_asset_str: str = "XLE"
    stagflation_asset_str: str = "XLV"
    deflation_asset_str: str = "XLP"
    fallback_asset_str: str | None = None
    benchmark_list: tuple[str, ...] = ("$SPXTR",)
    growth_sma_window_int: int = 200
    inflation_median_window_int: int = 200
    start_date_str: str = "1999-01-01"
    end_date_str: str | None = None
    capital_base_float: float = 100_000.0
    slippage_float: float = 0.0001
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0

    def __post_init__(self):
        self._validate_basket_tuple(
            basket_weight_spec_tuple=self.positive_beta_basket_tuple,
            basket_name_str="positive_beta_basket_tuple",
        )
        self._validate_basket_tuple(
            basket_weight_spec_tuple=self.negative_beta_basket_tuple,
            basket_name_str="negative_beta_basket_tuple",
        )
        if self.growth_sma_window_int <= 0:
            raise ValueError("growth_sma_window_int must be positive.")
        if self.inflation_median_window_int <= 0:
            raise ValueError("inflation_median_window_int must be positive.")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if self.slippage_float < 0.0:
            raise ValueError("slippage_float must be non-negative.")
        if self.commission_per_share_float < 0.0:
            raise ValueError("commission_per_share_float must be non-negative.")
        if self.commission_minimum_float < 0.0:
            raise ValueError("commission_minimum_float must be non-negative.")

        regime_asset_list = [
            self.goldilocks_asset_str,
            self.reflation_asset_str,
            self.stagflation_asset_str,
            self.deflation_asset_str,
        ]
        if len(set(regime_asset_list)) != len(regime_asset_list):
            raise ValueError("Regime target assets must be unique.")
        if self.fallback_asset_str is not None and self.fallback_asset_str in regime_asset_list:
            raise ValueError("fallback_asset_str must not also appear in the regime asset set.")

    @staticmethod
    def _validate_basket_tuple(
        basket_weight_spec_tuple: tuple[BasketWeightSpec, ...],
        basket_name_str: str,
    ) -> None:
        if len(basket_weight_spec_tuple) == 0:
            raise ValueError(f"{basket_name_str} must not be empty.")

        asset_str_list = [basket_weight_spec.asset_str for basket_weight_spec in basket_weight_spec_tuple]
        if len(set(asset_str_list)) != len(asset_str_list):
            raise ValueError(f"{basket_name_str} contains duplicate assets.")

        weight_vec = np.array(
            [float(basket_weight_spec.weight_float) for basket_weight_spec in basket_weight_spec_tuple],
            dtype=float,
        )
        if np.any(weight_vec <= 0.0):
            raise ValueError(f"{basket_name_str} must contain strictly positive weights.")
        if not np.isclose(weight_vec.sum(), 1.0, atol=1e-12):
            raise ValueError(f"{basket_name_str} must sum to 1.0.")

    @property
    def signal_asset_list(self) -> list[str]:
        asset_str_list = [self.growth_asset_str]
        asset_str_list.extend(
            basket_weight_spec.asset_str for basket_weight_spec in self.positive_beta_basket_tuple
        )
        asset_str_list.extend(
            basket_weight_spec.asset_str for basket_weight_spec in self.negative_beta_basket_tuple
        )
        return list(dict.fromkeys(asset_str_list))

    @property
    def trade_asset_list(self) -> list[str]:
        asset_str_list = [
            self.goldilocks_asset_str,
            self.reflation_asset_str,
            self.stagflation_asset_str,
            self.deflation_asset_str,
        ]
        if self.fallback_asset_str is not None:
            asset_str_list.append(self.fallback_asset_str)
        return list(dict.fromkeys(asset_str_list))

    @property
    def data_asset_list(self) -> list[str]:
        return list(dict.fromkeys(self.signal_asset_list + self.trade_asset_list))

    def regime_name_str(
        self,
        growth_rising_bool: bool,
        inflation_rising_bool: bool,
    ) -> str:
        if growth_rising_bool and not inflation_rising_bool:
            return "Goldilocks"
        if growth_rising_bool and inflation_rising_bool:
            return "Reflation"
        if not growth_rising_bool and inflation_rising_bool:
            return "Stagflation"
        return "Deflation"

    def target_asset_str(
        self,
        growth_rising_bool: bool,
        inflation_rising_bool: bool,
    ) -> str:
        if growth_rising_bool and not inflation_rising_bool:
            return self.goldilocks_asset_str
        if growth_rising_bool and inflation_rising_bool:
            return self.reflation_asset_str
        if not growth_rising_bool and inflation_rising_bool:
            return self.stagflation_asset_str
        return self.deflation_asset_str


DEFAULT_CONFIG = GrowthInflationSectorConfig()


def load_signal_close_df(
    config: GrowthInflationSectorConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Load total-return signal closes for the macro regime classifier.
    """
    signal_close_map: dict[str, pd.Series] = {}

    for symbol_str in tqdm(config.signal_asset_list, desc="loading signal closes"):
        signal_price_df = norgatedata.price_timeseries(
            symbol_str,
            stock_price_adjustment_setting=norgatedata.StockPriceAdjustmentType.TOTALRETURN,
            padding_setting=norgatedata.PaddingType.ALLMARKETDAYS,
            start_date=config.start_date_str,
            end_date=config.end_date_str,
            timeseriesformat="pandas-dataframe",
        )
        if len(signal_price_df) == 0:
            continue
        signal_close_map[symbol_str] = signal_price_df["Close"].astype(float)

    if len(signal_close_map) == 0:
        raise RuntimeError("No signal close data was loaded.")

    signal_close_df = pd.DataFrame(signal_close_map).sort_index()
    missing_symbol_list = [
        symbol_str for symbol_str in config.signal_asset_list if symbol_str not in signal_close_df.columns
    ]
    if len(missing_symbol_list) > 0:
        raise RuntimeError(f"Missing signal data for symbols: {missing_symbol_list}")

    return signal_close_df


def load_execution_price_df(
    config: GrowthInflationSectorConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Load tradable OHLC bars for fills, valuation, and signal alignment.
    """
    trade_symbol_list = [
        asset_str for asset_str in config.data_asset_list if asset_str not in config.benchmark_list
    ]
    return load_raw_prices(
        symbols=trade_symbol_list,
        benchmarks=list(config.benchmark_list),
        start_date=config.start_date_str,
        end_date=config.end_date_str,
    )


def merge_signal_close_into_pricing_data_df(
    execution_price_df: pd.DataFrame,
    signal_close_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach `SignalClose` fields alongside tradable OHLC bars.
    """
    pricing_data_df = execution_price_df.copy()
    for symbol_str in signal_close_df.columns:
        pricing_data_df[(symbol_str, "SignalClose")] = signal_close_df[symbol_str].reindex(pricing_data_df.index)

    pricing_data_df = pricing_data_df.sort_index(axis=1)
    return pricing_data_df


def extract_signal_close_df(
    pricing_data_df: pd.DataFrame,
    symbol_list: Sequence[str],
) -> pd.DataFrame:
    """
    Extract signal closes from `SignalClose` when present, otherwise from `Close`.
    """
    signal_close_map: dict[str, pd.Series] = {}

    for symbol_str in symbol_list:
        signal_close_key = (symbol_str, "SignalClose")
        close_key = (symbol_str, "Close")

        if signal_close_key in pricing_data_df.columns:
            signal_close_map[symbol_str] = pricing_data_df.loc[:, signal_close_key].astype(float)
            continue
        if close_key in pricing_data_df.columns:
            signal_close_map[symbol_str] = pricing_data_df.loc[:, close_key].astype(float)
            continue

        raise RuntimeError(f"Missing SignalClose/Close history for {symbol_str}.")

    signal_close_df = pd.DataFrame(signal_close_map, index=pricing_data_df.index, dtype=float)
    return signal_close_df


def compute_growth_inflation_signal_tables(
    signal_close_df: pd.DataFrame,
    config: GrowthInflationSectorConfig = DEFAULT_CONFIG,
) -> tuple[
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.DataFrame,
]:
    """
    Compute the daily growth/inflation regime state and one-hot target weights.
    """
    missing_symbol_list = [
        symbol_str for symbol_str in config.signal_asset_list if symbol_str not in signal_close_df.columns
    ]
    if len(missing_symbol_list) > 0:
        raise RuntimeError(f"Missing signal-close columns for: {missing_symbol_list}")

    signal_close_df = signal_close_df.loc[:, config.signal_asset_list].astype(float).copy()

    growth_price_ser = signal_close_df[config.growth_asset_str].astype(float)

    # *** CRITICAL*** The growth filter is a backward-only rolling SMA over
    # closes available by decision date t. No forward observations may enter.
    growth_sma_ser = growth_price_ser.rolling(
        window=config.growth_sma_window_int,
        min_periods=config.growth_sma_window_int,
    ).mean()
    growth_rising_bool_ser = (growth_price_ser > growth_sma_ser).where(growth_sma_ser.notna())

    positive_beta_portfolio_ser = pd.Series(0.0, index=signal_close_df.index, dtype=float)
    for basket_weight_spec in config.positive_beta_basket_tuple:
        positive_beta_portfolio_ser = (
            positive_beta_portfolio_ser
            + signal_close_df[basket_weight_spec.asset_str].astype(float) * float(basket_weight_spec.weight_float)
        )
    positive_beta_portfolio_ser.name = "positive_beta_portfolio_ser"

    negative_beta_portfolio_ser = pd.Series(0.0, index=signal_close_df.index, dtype=float)
    for basket_weight_spec in config.negative_beta_basket_tuple:
        negative_beta_portfolio_ser = (
            negative_beta_portfolio_ser
            + signal_close_df[basket_weight_spec.asset_str].astype(float) * float(basket_weight_spec.weight_float)
        )
    negative_beta_portfolio_ser.name = "negative_beta_portfolio_ser"

    inflation_ratio_ser = (
        positive_beta_portfolio_ser / negative_beta_portfolio_ser
    ).where(negative_beta_portfolio_ser > 0.0)
    inflation_ratio_ser.name = "inflation_ratio_ser"

    # *** CRITICAL*** The inflation-state threshold is a backward-only rolling
    # median. Centered or forward windows would create look-ahead leakage.
    inflation_median_ser = inflation_ratio_ser.rolling(
        window=config.inflation_median_window_int,
        min_periods=config.inflation_median_window_int,
    ).median()
    inflation_median_ser.name = "inflation_median_ser"
    inflation_rising_bool_ser = (inflation_ratio_ser > inflation_median_ser).where(inflation_median_ser.notna())
    inflation_rising_bool_ser.name = "inflation_rising_bool"

    regime_name_ser = pd.Series(index=signal_close_df.index, dtype=object, name="regime_name_str")
    target_asset_ser = pd.Series(index=signal_close_df.index, dtype=object, name="target_asset_str")

    valid_signal_mask_ser = growth_rising_bool_ser.notna() & inflation_rising_bool_ser.notna()
    for bar_ts in signal_close_df.index[valid_signal_mask_ser]:
        growth_rising_bool = bool(growth_rising_bool_ser.loc[bar_ts])
        inflation_rising_bool = bool(inflation_rising_bool_ser.loc[bar_ts])
        regime_name_ser.loc[bar_ts] = config.regime_name_str(
            growth_rising_bool=growth_rising_bool,
            inflation_rising_bool=inflation_rising_bool,
        )
        target_asset_ser.loc[bar_ts] = config.target_asset_str(
            growth_rising_bool=growth_rising_bool,
            inflation_rising_bool=inflation_rising_bool,
        )

    target_weight_df = pd.DataFrame(
        0.0,
        index=signal_close_df.index,
        columns=config.trade_asset_list,
        dtype=float,
    )
    for bar_ts, target_asset_obj in target_asset_ser.items():
        if pd.isna(target_asset_obj):
            continue
        target_weight_df.loc[bar_ts, str(target_asset_obj)] = 1.0

    weight_sum_ser = target_weight_df.sum(axis=1)
    valid_weight_mask_vec = np.isclose(weight_sum_ser.to_numpy(dtype=float), 0.0, atol=1e-12) | np.isclose(
        weight_sum_ser.to_numpy(dtype=float),
        1.0,
        atol=1e-12,
    )
    if not np.all(valid_weight_mask_vec):
        raise ValueError("Target weights must sum to either 0.0 or 1.0 on every date.")

    return (
        growth_price_ser,
        growth_sma_ser,
        growth_rising_bool_ser,
        inflation_ratio_ser,
        inflation_median_ser,
        inflation_rising_bool_ser,
        regime_name_ser,
        target_asset_ser,
        target_weight_df,
    )


def build_daily_target_weight_df(
    target_weight_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a report-ready daily weight schedule including residual cash.
    """
    if len(target_weight_df) == 0:
        return pd.DataFrame(columns=list(target_weight_df.columns) + ["Cash"], dtype=float)

    actionable_weight_mask_ser = target_weight_df.sum(axis=1) > 0.0
    if not actionable_weight_mask_ser.any():
        return pd.DataFrame(columns=list(target_weight_df.columns) + ["Cash"], dtype=float)

    first_actionable_ts = pd.Timestamp(actionable_weight_mask_ser[actionable_weight_mask_ser].index[0])
    daily_target_weight_df = target_weight_df.loc[target_weight_df.index >= first_actionable_ts].copy()
    daily_target_weight_df["Cash"] = 1.0 - daily_target_weight_df.sum(axis=1)

    weight_sum_ser = daily_target_weight_df.sum(axis=1)
    if not np.allclose(weight_sum_ser.to_numpy(dtype=float), 1.0, atol=1e-12):
        raise ValueError("Daily target weights, including cash, must sum to 1.0.")

    return daily_target_weight_df


def get_growth_inflation_pricing_data_df(
    config: GrowthInflationSectorConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Load execution OHLC data and attach total-return signal closes.
    """
    signal_close_df = load_signal_close_df(config=config)
    execution_price_df = load_execution_price_df(config=config)
    pricing_data_df = merge_signal_close_into_pricing_data_df(
        execution_price_df=execution_price_df,
        signal_close_df=signal_close_df,
    )
    return pricing_data_df


class GrowthInflationSectorTimingStrategy(Strategy):
    """
    Daily one-of-four sector rotation strategy.

    At each decision date t the model selects one sector, then any trade occurs
    only at the next open under the engine contract.
    """

    enable_signal_audit = True
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        config: GrowthInflationSectorConfig = DEFAULT_CONFIG,
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
        self.config = config
        self.trade_asset_list = list(config.trade_asset_list)
        self.trade_id_int = 0
        self.current_trade_id_map: defaultdict[str, int] = defaultdict(default_trade_id_int)
        self.show_taa_weights_report = True
        self.daily_target_weights = pd.DataFrame(columns=self.trade_asset_list + ["Cash"], dtype=float)

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        signal_data_df = pricing_data_df.copy()
        signal_close_df = extract_signal_close_df(
            pricing_data_df=pricing_data_df,
            symbol_list=self.config.signal_asset_list,
        )
        (
            _growth_price_ser,
            growth_sma_ser,
            growth_rising_bool_ser,
            inflation_ratio_ser,
            inflation_median_ser,
            inflation_rising_bool_ser,
            regime_name_ser,
            target_asset_ser,
            target_weight_df,
        ) = compute_growth_inflation_signal_tables(
            signal_close_df=signal_close_df,
            config=self.config,
        )

        growth_sma_field_str = f"growth_sma_{self.config.growth_sma_window_int}d_ser"
        inflation_median_field_str = f"inflation_median_{self.config.inflation_median_window_int}d_ser"

        signal_data_df[(self.config.growth_asset_str, growth_sma_field_str)] = growth_sma_ser.astype(float)
        signal_data_df[(self.config.growth_asset_str, "growth_rising_bool")] = growth_rising_bool_ser
        signal_data_df[(MODEL_SYMBOL_STR, "inflation_ratio_ser")] = inflation_ratio_ser.astype(float)
        signal_data_df[(MODEL_SYMBOL_STR, inflation_median_field_str)] = inflation_median_ser.astype(float)
        signal_data_df[(MODEL_SYMBOL_STR, "inflation_rising_bool")] = inflation_rising_bool_ser
        signal_data_df[(MODEL_SYMBOL_STR, "regime_name_str")] = regime_name_ser
        signal_data_df[(MODEL_SYMBOL_STR, "target_asset_str")] = target_asset_ser

        for asset_str in self.trade_asset_list:
            signal_data_df[(asset_str, "target_weight_ser")] = target_weight_df[asset_str].astype(float)

        return signal_data_df

    def _ensure_trade_id_int(self, asset_str: str) -> int:
        if self.current_trade_id_map[asset_str] == default_trade_id_int():
            self.trade_id_int += 1
            self.current_trade_id_map[asset_str] = self.trade_id_int
        return int(self.current_trade_id_map[asset_str])

    def iterate(self, data_df: pd.DataFrame, close_row_ser: pd.Series, open_price_ser: pd.Series):
        if close_row_ser is None or data_df is None:
            return

        target_weight_ser = pd.Series(
            {
                asset_str: float(close_row_ser.get((asset_str, "target_weight_ser"), 0.0))
                for asset_str in self.trade_asset_list
            },
            dtype=float,
        )
        if float(target_weight_ser.sum()) > 1.0 + 1e-12:
            raise RuntimeError(f"Target weights exceed 100% on {self.previous_bar}.")

        desired_asset_list = target_weight_ser[target_weight_ser > 0.0].index.tolist()
        current_position_ser = self.get_positions().reindex(self.trade_asset_list, fill_value=0.0).astype(int)

        # Submit liquidations first so the rotation path is explicit and easy to audit.
        for asset_str in self.trade_asset_list:
            current_share_int = int(current_position_ser.loc[asset_str])
            if current_share_int == 0 or asset_str in desired_asset_list:
                continue

            self.order_target_value(
                asset_str,
                0.0,
                trade_id=self._ensure_trade_id_int(asset_str),
            )
            self.current_trade_id_map[asset_str] = default_trade_id_int()

        for asset_str in desired_asset_list:
            current_share_int = int(current_position_ser.loc[asset_str])
            if current_share_int != 0:
                continue

            open_price_float = float(open_price_ser.get(asset_str, np.nan))
            if not np.isfinite(open_price_float) or open_price_float <= 0.0:
                raise RuntimeError(f"Invalid open price for target asset {asset_str} on {self.current_bar}.")

            target_weight_float = float(target_weight_ser.loc[asset_str])
            self._ensure_trade_id_int(asset_str)
            self.order_target_percent(
                asset_str,
                target_weight_float,
                trade_id=self.current_trade_id_map[asset_str],
            )

    def finalize(self, current_data: pd.DataFrame):
        target_weight_df = pd.DataFrame(
            {
                asset_str: current_data[(asset_str, "target_weight_ser")].astype(float)
                for asset_str in self.trade_asset_list
            },
            index=current_data.index,
            dtype=float,
        )
        self.daily_target_weights = build_daily_target_weight_df(target_weight_df=target_weight_df)


if __name__ == "__main__":
    config = DEFAULT_CONFIG
    pricing_data_df = get_growth_inflation_pricing_data_df(config=config)
    signal_close_df = extract_signal_close_df(
        pricing_data_df=pricing_data_df,
        symbol_list=config.signal_asset_list,
    )
    (
        growth_price_ser,
        growth_sma_ser,
        growth_rising_bool_ser,
        inflation_ratio_ser,
        inflation_median_ser,
        inflation_rising_bool_ser,
        regime_name_ser,
        target_asset_ser,
        target_weight_df,
    ) = compute_growth_inflation_signal_tables(
        signal_close_df=signal_close_df,
        config=config,
    )

    strategy = GrowthInflationSectorTimingStrategy(
        name="strategy_taa_growth_inflation_sector_timing",
        benchmarks=config.benchmark_list,
        config=config,
        capital_base=config.capital_base_float,
        slippage=config.slippage_float,
        commission_per_share=config.commission_per_share_float,
        commission_minimum=config.commission_minimum_float,
    )
    strategy.daily_target_weights = build_daily_target_weight_df(target_weight_df=target_weight_df)

    actionable_weight_mask_ser = target_weight_df.sum(axis=1) > 0.0
    if not actionable_weight_mask_ser.any():
        raise RuntimeError("No actionable dates were generated for strategy_taa_growth_inflation_sector_timing.")

    first_actionable_ts = pd.Timestamp(actionable_weight_mask_ser[actionable_weight_mask_ser].index[0])
    calendar_index = pricing_data_df.index[pricing_data_df.index >= first_actionable_ts]

    run_daily(
        strategy,
        pricing_data_df,
        calendar=calendar_index,
        audit_override_bool=None,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    diagnostic_df = pd.DataFrame(
        {
            "growth_price_ser": growth_price_ser,
            f"growth_sma_{config.growth_sma_window_int}d_ser": growth_sma_ser,
            "growth_rising_bool": growth_rising_bool_ser,
            "inflation_ratio_ser": inflation_ratio_ser,
            f"inflation_median_{config.inflation_median_window_int}d_ser": inflation_median_ser,
            "inflation_rising_bool": inflation_rising_bool_ser,
            "regime_name_str": regime_name_ser,
            "target_asset_str": target_asset_ser,
        }
    )

    print("Growth/inflation diagnostics preview:")
    display(diagnostic_df.dropna().head())

    print("Target weights preview:")
    display(target_weight_df[target_weight_df.sum(axis=1) > 0.0].head())

    display(strategy.summary)
    display(strategy.summary_trades)
    save_results(strategy)
