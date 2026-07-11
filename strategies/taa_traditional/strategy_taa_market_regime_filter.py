"""
Financial Hacker market-regime filter replication.

TL;DR: Trade `SPY` with a daily regime filter built from trend, volatility,
and credit-risk detectors. Full exposure is used when all detectors pass,
half exposure when exactly two pass, and cash otherwise.

Article rule
------------
For completed daily close t:

    trend_pass_t
        = 1[Close_SPY,t > SMA_200(Close_SPY)_t]

    volatility_pass_t
        = 1[Close_VIX,t < Close_VIX3M,t]

    credit_ratio_t
        = Close_HYG,t / Close_IEF,t

    credit_z_t
        = (credit_ratio_t - mean_100(credit_ratio)_t)
          / std_100(credit_ratio)_t

    credit_pass_t
        = 1[credit_z_t > -2]

    score_t
        = trend_pass_t + volatility_pass_t + credit_pass_t

    target_weight_t
        = 1.0  if score_t = 3
        = 0.5  if score_t = 2
        = 0.0  otherwise

Execution-model note
--------------------
The article's Zorro code evaluates daily bars directly. This repo's Vanilla
engine contract is causal next-open execution:

    completed close t signal -> next tradable open t+1 fill

This module preserves the article's detector math, but uses the repo-native
next-open timing rather than same-close execution.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from IPython.display import display

WORKSPACE_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT_PATH))

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from data.norgate_loader import load_raw_prices


SIGNAL_NAMESPACE_STR = "_MARKET_REGIME"
DEFAULT_TRADE_ID_INT = -1


@dataclass(frozen=True)
class MarketRegimeFilterConfig:
    strategy_name_str: str = "strategy_taa_market_regime_filter"
    trade_symbol_str: str = "SPY"
    green_trade_symbol_str: str | None = None
    caution_trade_symbol_str: str | None = None
    caution_defensive_trade_symbol_str: str | None = None
    risk_trade_symbol_str: str | None = None
    trend_symbol_str: str = "SPY"
    vix_symbol_str: str = "$VIX"
    vix3m_symbol_str: str = "$VIX3M"
    credit_risk_symbol_str: str = "HYG"
    credit_defensive_symbol_str: str = "IEF"
    benchmark_list: tuple[str, ...] = ("$SPXTR",)
    trend_window_day_int: int = 200
    credit_z_window_day_int: int = 100
    credit_z_min_float: float = -2.0
    green_score_int: int = 3
    caution_score_int: int = 2
    risk_score_int: int = 1
    green_target_weight_float: float = 1.0
    caution_target_weight_float: float = 0.5
    caution_defensive_target_weight_float: float = 0.0
    risk_target_weight_float: float = 0.0
    history_start_date_str: str = "2007-01-01"
    backtest_start_date_str: str = "2008-01-01"
    end_date_str: str | None = None
    capital_base_float: float = 100_000.0
    slippage_float: float = 0.0001
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0

    def __post_init__(self) -> None:
        if not self.strategy_name_str:
            raise ValueError("strategy_name_str must not be empty.")
        if not self.trade_symbol_str:
            raise ValueError("trade_symbol_str must not be empty.")
        if self.green_trade_symbol_str is not None and not self.green_trade_symbol_str:
            raise ValueError("green_trade_symbol_str must not be empty when provided.")
        if self.caution_trade_symbol_str is not None and not self.caution_trade_symbol_str:
            raise ValueError("caution_trade_symbol_str must not be empty when provided.")
        if (
            self.caution_defensive_trade_symbol_str is not None
            and not self.caution_defensive_trade_symbol_str
        ):
            raise ValueError("caution_defensive_trade_symbol_str must not be empty when provided.")
        if self.risk_trade_symbol_str is not None and not self.risk_trade_symbol_str:
            raise ValueError("risk_trade_symbol_str must not be empty when provided.")
        if self.trend_window_day_int <= 1:
            raise ValueError("trend_window_day_int must be greater than 1.")
        if self.credit_z_window_day_int <= 1:
            raise ValueError("credit_z_window_day_int must be greater than 1.")
        if self.green_score_int <= self.caution_score_int:
            raise ValueError("green_score_int must be greater than caution_score_int.")
        if self.caution_score_int <= self.risk_score_int:
            raise ValueError("caution_score_int must be greater than risk_score_int.")
        if self.risk_score_int < 0:
            raise ValueError("risk_score_int must be non-negative.")
        if not 0.0 <= self.caution_target_weight_float <= 1.0:
            raise ValueError("caution_target_weight_float must be in [0, 1].")
        if not 0.0 <= self.caution_defensive_target_weight_float <= 1.0:
            raise ValueError("caution_defensive_target_weight_float must be in [0, 1].")
        if self.caution_target_weight_float + self.caution_defensive_target_weight_float > 1.0:
            raise ValueError("score=2 target weights must not exceed 1.0.")
        if not 0.0 <= self.risk_target_weight_float <= 1.0:
            raise ValueError("risk_target_weight_float must be in [0, 1].")
        if not 0.0 <= self.green_target_weight_float <= 1.0:
            raise ValueError("green_target_weight_float must be in [0, 1].")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if self.slippage_float < 0.0:
            raise ValueError("slippage_float must be non-negative.")
        if self.commission_per_share_float < 0.0:
            raise ValueError("commission_per_share_float must be non-negative.")
        if self.commission_minimum_float < 0.0:
            raise ValueError("commission_minimum_float must be non-negative.")

    @property
    def green_trade_symbol(self) -> str:
        return self.trade_symbol_str if self.green_trade_symbol_str is None else self.green_trade_symbol_str

    @property
    def caution_trade_symbol(self) -> str:
        return self.trade_symbol_str if self.caution_trade_symbol_str is None else self.caution_trade_symbol_str

    @property
    def caution_defensive_trade_symbol(self) -> str:
        if self.caution_defensive_trade_symbol_str is None:
            return self.credit_defensive_symbol_str
        return self.caution_defensive_trade_symbol_str

    @property
    def risk_trade_symbol(self) -> str:
        if self.risk_trade_symbol_str is None:
            return self.credit_defensive_symbol_str
        return self.risk_trade_symbol_str

    @property
    def trade_symbol_tuple(self) -> tuple[str, ...]:
        trade_symbol_list = [self.green_trade_symbol, self.caution_trade_symbol]
        if self.caution_defensive_target_weight_float > 0.0:
            trade_symbol_list.append(self.caution_defensive_trade_symbol)
        if self.risk_target_weight_float > 0.0:
            trade_symbol_list.append(self.risk_trade_symbol)
        return tuple(dict.fromkeys(trade_symbol_list))

    @property
    def signal_symbol_tuple(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                [
                    self.trade_symbol_str,
                    self.green_trade_symbol,
                    self.caution_trade_symbol,
                    self.caution_defensive_trade_symbol,
                    self.risk_trade_symbol,
                    self.trend_symbol_str,
                    self.vix_symbol_str,
                    self.vix3m_symbol_str,
                    self.credit_risk_symbol_str,
                    self.credit_defensive_symbol_str,
                ]
            )
        )


DEFAULT_CONFIG = MarketRegimeFilterConfig()


def _as_float_ser(input_ser: pd.Series, name_str: str) -> pd.Series:
    output_ser = pd.Series(input_ser, copy=True).astype(float).sort_index()
    output_ser.name = name_str
    return output_ser


def _target_weight_ser_for_score(
    detector_score_int: int,
    config: MarketRegimeFilterConfig,
) -> pd.Series:
    target_weight_ser = pd.Series(0.0, index=list(config.trade_symbol_tuple), dtype=float)
    if detector_score_int == config.green_score_int:
        target_weight_ser.loc[config.green_trade_symbol] = float(config.green_target_weight_float)
    elif detector_score_int == config.caution_score_int:
        target_weight_ser.loc[config.caution_trade_symbol] = float(config.caution_target_weight_float)
        if config.caution_defensive_target_weight_float > 0.0:
            target_weight_ser.loc[config.caution_defensive_trade_symbol] = float(
                config.caution_defensive_target_weight_float
            )
    elif detector_score_int == config.risk_score_int and config.risk_target_weight_float > 0.0:
        target_weight_ser.loc[config.risk_trade_symbol] = float(config.risk_target_weight_float)
    return target_weight_ser


def _target_map_str(target_weight_ser: pd.Series) -> str:
    positive_target_weight_ser = target_weight_ser[target_weight_ser > 0.0]
    if len(positive_target_weight_ser) == 0:
        return "cash"
    return "+".join(
        f"{str(symbol_str)}:{float(weight_float):.6g}"
        for symbol_str, weight_float in positive_target_weight_ser.items()
    )


def _symbol_column_token_str(symbol_str: str) -> str:
    token_str = "".join(
        character_str.lower() if character_str.isalnum() else "_"
        for character_str in symbol_str
    ).strip("_")
    return token_str if token_str else "symbol"


def compute_market_regime_signal_df(
    spy_close_ser: pd.Series,
    vix_close_ser: pd.Series,
    vix3m_close_ser: pd.Series,
    hyg_close_ser: pd.Series,
    ief_close_ser: pd.Series,
    config: MarketRegimeFilterConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Compute the article's three-detector daily regime table.
    """
    spy_close_ser = _as_float_ser(spy_close_ser, "spy_close_ser")
    vix_close_ser = _as_float_ser(vix_close_ser, "vix_close_ser")
    vix3m_close_ser = _as_float_ser(vix3m_close_ser, "vix3m_close_ser")
    hyg_close_ser = _as_float_ser(hyg_close_ser, "hyg_close_ser")
    ief_close_ser = _as_float_ser(ief_close_ser, "ief_close_ser")

    common_index = spy_close_ser.index
    common_index = common_index.union(vix_close_ser.index)
    common_index = common_index.union(vix3m_close_ser.index)
    common_index = common_index.union(hyg_close_ser.index)
    common_index = common_index.union(ief_close_ser.index).sort_values()

    spy_close_ser = spy_close_ser.reindex(common_index)
    vix_close_ser = vix_close_ser.reindex(common_index)
    vix3m_close_ser = vix3m_close_ser.reindex(common_index)
    hyg_close_ser = hyg_close_ser.reindex(common_index)
    ief_close_ser = ief_close_ser.reindex(common_index)

    # *** CRITICAL*** The trend detector uses only the trailing SMA ending at
    # completed close t. A centered or forward window would leak future trend.
    spy_sma_ser = spy_close_ser.rolling(
        window=config.trend_window_day_int,
        min_periods=config.trend_window_day_int,
    ).mean()
    trend_ready_bool_ser = spy_close_ser.notna() & spy_sma_ser.notna()
    trend_pass_bool_ser = pd.Series(pd.NA, index=common_index, dtype="boolean")
    trend_pass_bool_ser.loc[trend_ready_bool_ser] = (
        spy_close_ser.loc[trend_ready_bool_ser] > spy_sma_ser.loc[trend_ready_bool_ser]
    )

    volatility_ready_bool_ser = vix_close_ser.notna() & vix3m_close_ser.notna()
    volatility_pass_bool_ser = pd.Series(pd.NA, index=common_index, dtype="boolean")
    volatility_pass_bool_ser.loc[volatility_ready_bool_ser] = (
        vix_close_ser.loc[volatility_ready_bool_ser]
        < vix3m_close_ser.loc[volatility_ready_bool_ser]
    )

    credit_ratio_ser = hyg_close_ser / ief_close_ser.replace(0.0, np.nan)
    # *** CRITICAL*** The credit z-score uses a trailing window ending at
    # completed close t. It is not shifted because the close t signal is only
    # allowed to trade at the next open t+1 in this engine.
    credit_ratio_mean_ser = credit_ratio_ser.rolling(
        window=config.credit_z_window_day_int,
        min_periods=config.credit_z_window_day_int,
    ).mean()
    credit_ratio_std_ser = credit_ratio_ser.rolling(
        window=config.credit_z_window_day_int,
        min_periods=config.credit_z_window_day_int,
    ).std(ddof=0)
    credit_z_ser = (credit_ratio_ser - credit_ratio_mean_ser) / credit_ratio_std_ser.replace(0.0, np.nan)
    credit_ready_bool_ser = credit_z_ser.notna()
    credit_pass_bool_ser = pd.Series(pd.NA, index=common_index, dtype="boolean")
    credit_pass_bool_ser.loc[credit_ready_bool_ser] = (
        credit_z_ser.loc[credit_ready_bool_ser] > float(config.credit_z_min_float)
    )

    all_detector_ready_bool_ser = (
        trend_pass_bool_ser.notna()
        & volatility_pass_bool_ser.notna()
        & credit_pass_bool_ser.notna()
    )
    raw_score_ser = (
        trend_pass_bool_ser.astype("Int64")
        + volatility_pass_bool_ser.astype("Int64")
        + credit_pass_bool_ser.astype("Int64")
    )
    detector_score_ser = pd.Series(pd.NA, index=common_index, dtype="Int64")
    detector_score_ser.loc[all_detector_ready_bool_ser] = raw_score_ser.loc[
        all_detector_ready_bool_ser
    ].astype(int)

    target_weight_by_symbol_df = pd.DataFrame(
        0.0,
        index=common_index,
        columns=list(config.trade_symbol_tuple),
        dtype=float,
    )
    target_map_ser = pd.Series(pd.NA, index=common_index, dtype="object", name="target_map_str")
    for signal_date_ts, detector_score_value in detector_score_ser.dropna().items():
        score_target_weight_ser = _target_weight_ser_for_score(
            detector_score_int=int(detector_score_value),
            config=config,
        )
        target_weight_by_symbol_df.loc[signal_date_ts, score_target_weight_ser.index] = (
            score_target_weight_ser
        )
        target_map_ser.loc[signal_date_ts] = _target_map_str(score_target_weight_ser)

    target_weight_ser = pd.Series(np.nan, index=common_index, dtype=float, name="target_weight_ser")
    target_weight_ser.loc[detector_score_ser.notna()] = target_weight_by_symbol_df.sum(axis=1)
    target_weight_column_dict = {
        f"target_weight_{_symbol_column_token_str(symbol_str)}_ser": target_weight_by_symbol_df[
            symbol_str
        ]
        for symbol_str in config.trade_symbol_tuple
    }

    regime_signal_df = pd.DataFrame(
        {
            "spy_close_ser": spy_close_ser,
            "spy_sma_ser": spy_sma_ser,
            "trend_pass_bool": trend_pass_bool_ser,
            "vix_close_ser": vix_close_ser,
            "vix3m_close_ser": vix3m_close_ser,
            "volatility_pass_bool": volatility_pass_bool_ser,
            "credit_ratio_ser": credit_ratio_ser,
            "credit_ratio_mean_ser": credit_ratio_mean_ser,
            "credit_ratio_std_ser": credit_ratio_std_ser,
            "credit_z_ser": credit_z_ser,
            "credit_pass_bool": credit_pass_bool_ser,
            "detector_score_int": detector_score_ser,
            "target_weight_ser": target_weight_ser,
            **target_weight_column_dict,
            "target_map_str": target_map_ser,
        },
        index=common_index,
    )
    return regime_signal_df


def get_market_regime_filter_prices(
    config: MarketRegimeFilterConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    pricing_data_df = load_raw_prices(
        symbols=list(config.signal_symbol_tuple),
        benchmarks=list(config.benchmark_list),
        start_date=config.history_start_date_str,
        end_date=config.end_date_str,
    )
    required_symbol_set = set(config.signal_symbol_tuple).union(set(config.benchmark_list))
    loaded_symbol_set = set(pricing_data_df.columns.get_level_values(0))
    missing_symbol_list = sorted(required_symbol_set - loaded_symbol_set)
    if len(missing_symbol_list) > 0:
        raise RuntimeError(f"Missing required market-regime symbols: {missing_symbol_list}")
    return pricing_data_df


def _multiindex_feature_df(regime_signal_df: pd.DataFrame) -> pd.DataFrame:
    feature_data_df = regime_signal_df.copy()
    feature_data_df.columns = pd.MultiIndex.from_tuples(
        [(SIGNAL_NAMESPACE_STR, str(column_str)) for column_str in feature_data_df.columns]
    )
    return feature_data_df


def build_execution_target_weight_df(
    signal_data_df: pd.DataFrame,
    trade_symbol_str: str | None = None,
    config: MarketRegimeFilterConfig | None = None,
    result_index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """
    Map close-t signals onto the next execution bar for reporting.
    """
    config_obj = DEFAULT_CONFIG if config is None else config
    symbol_list = list(config_obj.trade_symbol_tuple)
    if trade_symbol_str is not None and config is None:
        symbol_list = [trade_symbol_str]

    detector_score_key_tuple = (SIGNAL_NAMESPACE_STR, "detector_score_int")
    target_key_tuple = (SIGNAL_NAMESPACE_STR, "target_weight_ser")
    if detector_score_key_tuple not in signal_data_df.columns and target_key_tuple not in signal_data_df.columns:
        return pd.DataFrame(columns=symbol_list + ["Cash"], dtype=float)

    if detector_score_key_tuple in signal_data_df.columns:
        # *** CRITICAL*** signal row t is tradable only on the next engine bar.
        # shift(1) prevents the reporting schedule from implying same-close fills.
        execution_score_ser = signal_data_df[detector_score_key_tuple].astype("Float64").shift(1)
        if result_index is not None:
            execution_score_ser = execution_score_ser.reindex(result_index)

        daily_target_weight_df = pd.DataFrame(0.0, index=execution_score_ser.index, columns=symbol_list)
        green_mask_ser = execution_score_ser == float(config_obj.green_score_int)
        caution_mask_ser = execution_score_ser == float(config_obj.caution_score_int)
        daily_target_weight_df.loc[green_mask_ser, config_obj.green_trade_symbol] = (
            float(config_obj.green_target_weight_float)
        )
        daily_target_weight_df.loc[caution_mask_ser, config_obj.caution_trade_symbol] = (
            float(config_obj.caution_target_weight_float)
        )
        if config_obj.caution_defensive_target_weight_float > 0.0:
            daily_target_weight_df.loc[caution_mask_ser, config_obj.caution_defensive_trade_symbol] = (
                float(config_obj.caution_defensive_target_weight_float)
            )
        if config_obj.risk_target_weight_float > 0.0:
            risk_mask_ser = execution_score_ser == float(config_obj.risk_score_int)
            daily_target_weight_df.loc[risk_mask_ser, config_obj.risk_trade_symbol] = (
                float(config_obj.risk_target_weight_float)
            )
    else:
        if config is not None and len(symbol_list) != 1:
            raise RuntimeError(
                "Multi-asset market-regime reporting requires detector_score_int. "
                "Falling back to target_weight_ser would collapse the target map onto one symbol."
            )
        # *** CRITICAL*** Legacy single-asset reporting also shifts close-t
        # exposure to the next execution bar.
        execution_target_weight_ser = signal_data_df[target_key_tuple].astype(float).shift(1)
        if result_index is not None:
            execution_target_weight_ser = execution_target_weight_ser.reindex(result_index)
        execution_target_weight_ser = execution_target_weight_ser.fillna(0.0).clip(lower=0.0, upper=1.0)
        daily_target_weight_df = pd.DataFrame(
            {symbol_list[0]: execution_target_weight_ser},
            index=execution_target_weight_ser.index,
            dtype=float,
        )

    target_sum_ser = daily_target_weight_df.sum(axis=1).clip(lower=0.0, upper=1.0)
    daily_target_weight_df["Cash"] = 1.0 - target_sum_ser
    return daily_target_weight_df


class MarketRegimeFilterStrategy(Strategy):
    """
    Daily SPY exposure strategy controlled by the article's regime score.
    """

    enable_signal_audit = True
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        config: MarketRegimeFilterConfig = DEFAULT_CONFIG,
    ):
        super().__init__(
            name=name,
            benchmarks=list(config.benchmark_list),
            capital_base=config.capital_base_float,
            slippage=config.slippage_float,
            commission_per_share=config.commission_per_share_float,
            commission_minimum=config.commission_minimum_float,
        )
        self.config = config
        self.trade_id_int = 0
        self.current_trade_id_map: dict[str, int] = {
            trade_symbol_str: DEFAULT_TRADE_ID_INT
            for trade_symbol_str in config.trade_symbol_tuple
        }
        self.regime_signal_df = pd.DataFrame()
        self.daily_target_weights = pd.DataFrame(columns=list(config.trade_symbol_tuple) + ["Cash"], dtype=float)
        self.show_taa_weights_report = True

    @property
    def current_trade_id_int(self) -> int:
        return int(self.current_trade_id_map.get(self.config.trade_symbol_str, DEFAULT_TRADE_ID_INT))

    @current_trade_id_int.setter
    def current_trade_id_int(self, trade_id_int: int) -> None:
        self.current_trade_id_map[self.config.trade_symbol_str] = int(trade_id_int)

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        config = self.config
        required_key_list = [
            (config.trend_symbol_str, "Close"),
            (config.vix_symbol_str, "Close"),
            (config.vix3m_symbol_str, "Close"),
            (config.credit_risk_symbol_str, "Close"),
            (config.credit_defensive_symbol_str, "Close"),
        ]
        missing_key_list = [key_tuple for key_tuple in required_key_list if key_tuple not in pricing_data.columns]
        if len(missing_key_list) > 0:
            raise RuntimeError(f"Missing required market-regime close columns: {missing_key_list}")

        regime_signal_df = compute_market_regime_signal_df(
            spy_close_ser=pricing_data[(config.trend_symbol_str, "Close")],
            vix_close_ser=pricing_data[(config.vix_symbol_str, "Close")],
            vix3m_close_ser=pricing_data[(config.vix3m_symbol_str, "Close")],
            hyg_close_ser=pricing_data[(config.credit_risk_symbol_str, "Close")],
            ief_close_ser=pricing_data[(config.credit_defensive_symbol_str, "Close")],
            config=config,
        ).reindex(pricing_data.index)
        self.regime_signal_df = regime_signal_df.copy()

        feature_data_df = _multiindex_feature_df(regime_signal_df)
        return pd.concat([pricing_data, feature_data_df], axis=1)

    def _ensure_trade_id_int(self, trade_symbol_str: str) -> int:
        if self.current_trade_id_map.get(trade_symbol_str, DEFAULT_TRADE_ID_INT) == DEFAULT_TRADE_ID_INT:
            self.trade_id_int += 1
            self.current_trade_id_map[trade_symbol_str] = self.trade_id_int
        return int(self.current_trade_id_map[trade_symbol_str])

    def _target_weight_ser_from_close(self, close: pd.Series) -> pd.Series:
        config = self.config
        detector_score_key_tuple = (SIGNAL_NAMESPACE_STR, "detector_score_int")
        if detector_score_key_tuple not in close.index:
            raise RuntimeError(
                f"Missing market-regime detector score on {self.previous_bar}. "
                "The strategy cannot safely preserve exposure without a known regime."
            )
        if pd.isna(close.loc[detector_score_key_tuple]):
            raise RuntimeError(
                f"Market-regime detector score is NaN on {self.previous_bar}. "
                "The strategy cannot safely preserve exposure without a known regime."
            )

        detector_score_int = int(close.loc[detector_score_key_tuple])
        return _target_weight_ser_for_score(detector_score_int=detector_score_int, config=config)

    def iterate(
        self,
        data: pd.DataFrame,
        close: pd.Series,
        open_prices: pd.Series,
    ) -> None:
        if data is None or close is None:
            return

        target_weight_ser = self._target_weight_ser_from_close(close)
        if len(target_weight_ser) == 0:
            return
        position_ser = self.get_positions().reindex(target_weight_ser.index, fill_value=0.0).astype(float)

        for trade_symbol_str, target_weight_float in target_weight_ser.items():
            current_share_float = float(position_ser.loc[trade_symbol_str])
            if target_weight_float <= 0.0:
                if np.isclose(current_share_float, 0.0, atol=1e-12):
                    continue
                self.order_target_value(
                    trade_symbol_str,
                    0.0,
                    trade_id=self._ensure_trade_id_int(trade_symbol_str),
                )
                self.current_trade_id_map[trade_symbol_str] = DEFAULT_TRADE_ID_INT
                continue

            close_key_tuple = (trade_symbol_str, "Close")
            sizing_price_float = np.nan
            if close_key_tuple in close.index:
                sizing_price_float = float(close.loc[close_key_tuple])
            if not np.isfinite(sizing_price_float) or sizing_price_float <= 0.0:
                sizing_price_float = float(open_prices.get(trade_symbol_str, np.nan))
            if not np.isfinite(sizing_price_float) or sizing_price_float <= 0.0:
                raise RuntimeError(
                    f"Invalid sizing price for {trade_symbol_str} on {self.current_bar}."
                )

            # *** CRITICAL*** Percent target orders are sized from previous-bar
            # portfolio value and previous close, matching Strategy.process_orders().
            target_share_int = int(
                float(self.previous_total_value) * target_weight_float / sizing_price_float
            )
            if np.isclose(float(target_share_int) - current_share_float, 0.0, atol=1e-12):
                continue

            self.order_target_percent(
                trade_symbol_str,
                target_weight_float,
                trade_id=self._ensure_trade_id_int(trade_symbol_str),
            )

    def finalize(self, current_data: pd.DataFrame) -> None:
        result_index = pd.DatetimeIndex(self.results.index) if len(self.results.index) > 0 else None
        self.daily_target_weights = build_execution_target_weight_df(
            signal_data_df=current_data,
            config=self.config,
            result_index=result_index,
        )


def _with_run_overrides(
    config: MarketRegimeFilterConfig,
    backtest_start_date_str: str | None,
    capital_base_float: float | None,
    end_date_str: str | None,
) -> MarketRegimeFilterConfig:
    override_dict: dict[str, object] = {}
    if backtest_start_date_str is not None:
        override_dict["backtest_start_date_str"] = backtest_start_date_str
    if capital_base_float is not None:
        override_dict["capital_base_float"] = float(capital_base_float)
    if end_date_str is not None:
        override_dict["end_date_str"] = end_date_str
    if len(override_dict) == 0:
        return config
    return replace(config, **override_dict)


def run_variant(
    config: MarketRegimeFilterConfig = DEFAULT_CONFIG,
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
) -> MarketRegimeFilterStrategy:
    config_obj = _with_run_overrides(
        config=config,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
    )
    pricing_data_df = get_market_regime_filter_prices(config=config_obj)
    strategy_obj = MarketRegimeFilterStrategy(
        name=config_obj.strategy_name_str,
        config=config_obj,
    )

    # *** CRITICAL*** Keep pre-start history for SMA200 and credit z-score,
    # but execute/report only from the configured article start date.
    calendar_idx = pricing_data_df.index[
        pricing_data_df.index >= pd.Timestamp(config_obj.backtest_start_date_str)
    ]
    if len(calendar_idx) == 0:
        raise RuntimeError(
            f"No backtest calendar rows on or after {config_obj.backtest_start_date_str}."
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
        display(strategy_obj.regime_signal_df.tail())
        display(strategy_obj.summary)
        display(strategy_obj.summary_trades)

    if save_results_bool:
        output_path = save_results(strategy_obj, output_dir=output_dir_str)
        strategy_obj.regime_signal_df.to_csv(
            output_path / "market_regime_signal.csv",
            date_format="%Y-%m-%d",
        )
        strategy_obj.daily_target_weights.to_csv(
            output_path / "daily_target_weights.csv",
            date_format="%Y-%m-%d",
        )

    return strategy_obj


if __name__ == "__main__":
    run_variant()
