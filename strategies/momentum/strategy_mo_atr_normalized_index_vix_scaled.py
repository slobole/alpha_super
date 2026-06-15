"""
Research-only broad-index variants of the ATR-normalized momentum model.

These wrappers preserve the production-reference NDX/VXN strategy mechanics:

    score_{i,t} = ROC12_{i,t} / ATR20_{i,t}

    base_weight_{i,t}
        = 1 / max_positions    if i is selected
        = 0                    otherwise

    exposure_scale_t = clip(22 / VIX_t, 0.25, 1.0)

    target_weight_{i,t} = base_weight_{i,t} * exposure_scale_t

Timing is unchanged: use the actual last tradable close of the month for the
decision, then execute at the next tradable open.

The VIX-scaled inverse-volatility research class changes selected-name weighting
and then applies the VIX total-exposure scaler:

    daily_return_{i,d} = Close_{i,d} / Close_{i,d-1} - 1

    vol63_{i,t} = std(daily_return_{i,t-62:t})

    base_weight_{i,t}
        = (1 / vol63_{i,t}) / sum_j(1 / vol63_{j,t})

for selected names with finite positive vol63.

The pure risk-parity class uses the same selected-name inverse-volatility
weights, but does not apply any VIX/VXN exposure scaler.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from strategies.momentum.strategy_mo_atr_normalized_ndx import (
    ATR_WINDOW_INT,
    AtrNormalizedNdxStrategy,
    get_atr_normalized_ndx_data,
    get_monthly_decision_close_df,
)
from strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled import (
    VxnScaledAtrNormalizedNdxConfig,
    VxnScaledAtrNormalizedNdxStrategy,
    get_asof_vxn_scale_float,
    get_vxn_scaled_atr_normalized_ndx_data,
)


TRAILING_VOL_FIELD_TEMPLATE_STR = "trailing_vol_{window_int}_float"
SELECTION_SCORE_MODE_ATR20_STR = "atr20"
SELECTION_SCORE_MODE_NATR20_STR = "natr20"
SELECTION_SCORE_MODE_NATR63_STR = "natr63"
SELECTION_SCORE_MODE_VOL63_STR = "vol63"
SELECTION_SCORE_MODE_SET = {
    SELECTION_SCORE_MODE_ATR20_STR,
    SELECTION_SCORE_MODE_NATR20_STR,
    SELECTION_SCORE_MODE_NATR63_STR,
    SELECTION_SCORE_MODE_VOL63_STR,
}
SELECTION_SCORE_FIELD_STR = "risk_adj_score_ser"


@dataclass(frozen=True)
class VixScaledAtrNormalizedIndexConfig(VxnScaledAtrNormalizedNdxConfig):
    """Generic broad-index preset using the original scaler field names."""

    index_label_str: str = "S&P 500"
    strategy_name_str: str = "strategy_mo_atr_normalized_sp500_vix_scaled"
    vxn_symbol_str: str = "$VIX"
    inverse_vol_window_int: int = 63
    selection_score_mode_str: str = "atr20"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.index_label_str:
            raise ValueError("index_label_str must not be empty.")
        if not self.strategy_name_str:
            raise ValueError("strategy_name_str must not be empty.")
        if self.inverse_vol_window_int <= 1:
            raise ValueError("inverse_vol_window_int must be greater than 1.")
        if self.selection_score_mode_str not in SELECTION_SCORE_MODE_SET:
            raise ValueError(
                "selection_score_mode_str must be one of "
                f"{sorted(SELECTION_SCORE_MODE_SET)}, got {self.selection_score_mode_str!r}."
            )


SP500_CONFIG = VixScaledAtrNormalizedIndexConfig(
    indexname_str="S&P 500",
    index_label_str="S&P 500",
    strategy_name_str="strategy_mo_atr_normalized_sp500_vix_scaled",
)
NASDAQ100_CONFIG = VixScaledAtrNormalizedIndexConfig(
    indexname_str="Nasdaq 100",
    index_label_str="Nasdaq 100",
    strategy_name_str="strategy_mo_atr_normalized_nasdaq100_vix_scaled",
    vxn_symbol_str="$VXN",
)
RUSSELL1000_CONFIG = VixScaledAtrNormalizedIndexConfig(
    indexname_str="Russell 1000",
    index_label_str="Russell 1000",
    regime_symbol_str="$RUI",
    strategy_name_str="strategy_mo_atr_normalized_russell1000_vix_scaled",
)
RUSSELL3000_CONFIG = VixScaledAtrNormalizedIndexConfig(
    indexname_str="Russell 3000",
    index_label_str="Russell 3000",
    strategy_name_str="strategy_mo_atr_normalized_russell3000_vix_scaled",
)
NYSE_COMPOSITE_CONFIG = VixScaledAtrNormalizedIndexConfig(
    indexname_str="NYSE Composite",
    index_label_str="NYSE Composite",
    strategy_name_str="strategy_mo_atr_normalized_nyse_composite_vix_scaled",
)
NASDAQ_BIOTECHNOLOGY_CONFIG = VixScaledAtrNormalizedIndexConfig(
    indexname_str="NASDAQ Biotechnology",
    index_label_str="NASDAQ Biotechnology",
    history_start_date_str="2001-05-21",
    backtest_start_date_str="2002-06-01",
    strategy_name_str="strategy_mo_atr_normalized_nasdaq_biotechnology_vix_scaled",
)
DEFAULT_CONFIG = SP500_CONFIG


__all__ = [
    "DEFAULT_CONFIG",
    "NASDAQ100_CONFIG",
    "NASDAQ_BIOTECHNOLOGY_CONFIG",
    "SP500_CONFIG",
    "RUSSELL1000_CONFIG",
    "RUSSELL3000_CONFIG",
    "NYSE_COMPOSITE_CONFIG",
    "SELECTION_SCORE_MODE_ATR20_STR",
    "SELECTION_SCORE_MODE_NATR20_STR",
    "SELECTION_SCORE_MODE_NATR63_STR",
    "SELECTION_SCORE_MODE_VOL63_STR",
    "TRAILING_VOL_FIELD_TEMPLATE_STR",
    "VixScaledAtrNormalizedIndexConfig",
    "VixScaledAtrNormalizedIndexStrategy",
    "InverseVol63VixScaledAtrNormalizedIndexStrategy",
    "RiskParity63AtrNormalizedIndexStrategy",
    "VixScaledAtrNormalizedSp500Strategy",
    "VixScaledAtrNormalizedRussell1000Strategy",
    "VixScaledAtrNormalizedRussell3000Strategy",
    "build_strategy",
    "build_inverse_vol_63_strategy",
    "build_risk_parity_63_strategy",
    "get_atr_normalized_index_data",
    "get_vix_scaled_atr_normalized_index_data",
    "run_variant",
]


class VixScaledAtrNormalizedIndexStrategy(VxnScaledAtrNormalizedNdxStrategy):
    """ATR-normalized broad-index momentum with a VIX total-exposure scaler."""


def _trailing_vol_feature_df(
    pricing_data: pd.DataFrame,
    benchmark_list: list[str],
    inverse_vol_window_int: int,
    inverse_vol_field_str: str,
) -> pd.DataFrame:
    tradeable_symbol_list = [
        str(symbol_str)
        for symbol_str in pricing_data.columns.get_level_values(0).unique()
        if str(symbol_str) not in benchmark_list
    ]
    price_close_df = pd.DataFrame(
        {symbol_str: pricing_data[(symbol_str, "Close")] for symbol_str in tradeable_symbol_list},
        index=pricing_data.index,
    ).astype(float)

    # *** CRITICAL*** inverse-vol sizing must use only close-to-close
    # returns known at the month-end decision close. shift(1) forms
    # return_d from Close_d and Close_{d-1}; the rolling window ending at
    # decision_t must never include any post-decision open or close.
    daily_return_df = (price_close_df / price_close_df.shift(1)) - 1.0
    trailing_vol_df = daily_return_df.rolling(
        window=inverse_vol_window_int,
        min_periods=inverse_vol_window_int,
    ).std()

    trailing_vol_feature_df = trailing_vol_df.copy()
    trailing_vol_feature_df.columns = pd.MultiIndex.from_tuples(
        [
            (symbol_str, inverse_vol_field_str)
            for symbol_str in trailing_vol_feature_df.columns
        ]
    )
    return trailing_vol_feature_df


def _selection_score_feature_df(
    pricing_data: pd.DataFrame,
    benchmark_list: list[str],
    lookback_month_int: int,
    selection_score_mode_str: str,
) -> pd.DataFrame:
    tradeable_symbol_list = [
        str(symbol_str)
        for symbol_str in pricing_data.columns.get_level_values(0).unique()
        if str(symbol_str) not in benchmark_list
    ]
    price_close_df = pd.DataFrame(
        {symbol_str: pricing_data[(symbol_str, "Close")] for symbol_str in tradeable_symbol_list},
        index=pricing_data.index,
    ).astype(float)
    price_high_df = pd.DataFrame(
        {symbol_str: pricing_data[(symbol_str, "High")] for symbol_str in tradeable_symbol_list},
        index=pricing_data.index,
    ).astype(float)
    price_low_df = pd.DataFrame(
        {symbol_str: pricing_data[(symbol_str, "Low")] for symbol_str in tradeable_symbol_list},
        index=pricing_data.index,
    ).astype(float)

    monthly_decision_close_df = get_monthly_decision_close_df(price_close_df=price_close_df)

    # *** CRITICAL*** selection momentum must use only completed month-end
    # decision closes. shift(lookback_month_int) references older month-end
    # closes only, so the score at decision_t cannot see future months.
    monthly_roc_df = (
        monthly_decision_close_df / monthly_decision_close_df.shift(lookback_month_int)
    ) - 1.0

    if selection_score_mode_str in {
        SELECTION_SCORE_MODE_NATR20_STR,
        SELECTION_SCORE_MODE_NATR63_STR,
    }:
        atr_window_int = (
            ATR_WINDOW_INT
            if selection_score_mode_str == SELECTION_SCORE_MODE_NATR20_STR
            else 63
        )

        # *** CRITICAL*** true range uses prior close via shift(1), then a
        # trailing rolling mean. NATR_t = ATR_t / Close_t, all known at the
        # month-end decision close.
        prior_close_df = price_close_df.shift(1)
        true_range_df = (price_high_df - price_low_df).combine(
            (price_high_df - prior_close_df).abs(),
            np.maximum,
        )
        true_range_df = true_range_df.combine(
            (price_low_df - prior_close_df).abs(),
            np.maximum,
        )
        atr_value_df = true_range_df.rolling(
            window=atr_window_int,
            min_periods=atr_window_int,
        ).mean()
        normalizer_df = (
            atr_value_df.reindex(monthly_decision_close_df.index)
            / monthly_decision_close_df.replace(0.0, np.nan)
        )
    elif selection_score_mode_str == SELECTION_SCORE_MODE_VOL63_STR:
        # *** CRITICAL*** vol63 selection normalization uses only trailing
        # close-to-close returns known at the month-end decision close.
        daily_return_df = (price_close_df / price_close_df.shift(1)) - 1.0
        normalizer_df = daily_return_df.rolling(
            window=63,
            min_periods=63,
        ).std().reindex(monthly_decision_close_df.index)
    else:
        raise ValueError(f"Unsupported selection_score_mode_str: {selection_score_mode_str!r}.")

    selection_score_df = (monthly_roc_df / normalizer_df).replace([np.inf, -np.inf], np.nan)
    selection_score_feature_df = selection_score_df.reindex(pricing_data.index)
    selection_score_feature_df.columns = pd.MultiIndex.from_tuples(
        [
            (symbol_str, SELECTION_SCORE_FIELD_STR)
            for symbol_str in selection_score_feature_df.columns
        ]
    )
    return selection_score_feature_df


def _replace_selection_score_feature_df(
    signal_data_df: pd.DataFrame,
    selection_score_feature_df: pd.DataFrame,
) -> pd.DataFrame:
    existing_score_column_list = [
        column_tuple
        for column_tuple in signal_data_df.columns
        if isinstance(column_tuple, tuple)
        and len(column_tuple) >= 2
        and column_tuple[1] == SELECTION_SCORE_FIELD_STR
    ]
    return pd.concat(
        [
            signal_data_df.drop(columns=existing_score_column_list),
            selection_score_feature_df,
        ],
        axis=1,
    )


def _selected_risk_parity_weight_ser(
    close_row_ser: pd.Series,
    selected_base_target_weight_ser: pd.Series,
    inverse_vol_field_str: str,
) -> pd.Series:
    if len(selected_base_target_weight_ser) == 0:
        return selected_base_target_weight_ser

    candidate_feature_df = close_row_ser.unstack()
    if inverse_vol_field_str not in candidate_feature_df.columns:
        return pd.Series(dtype=float)

    selected_vol_ser = pd.to_numeric(
        candidate_feature_df.reindex(selected_base_target_weight_ser.index)[
            inverse_vol_field_str
        ],
        errors="coerce",
    )
    finite_positive_mask_ser = pd.Series(
        np.isfinite(selected_vol_ser.to_numpy(dtype=float)),
        index=selected_vol_ser.index,
    ) & (selected_vol_ser > 0.0)
    finite_positive_vol_ser = selected_vol_ser[finite_positive_mask_ser].astype(float)
    if len(finite_positive_vol_ser) == 0:
        return pd.Series(dtype=float)

    inverse_vol_ser = 1.0 / finite_positive_vol_ser
    return inverse_vol_ser / float(inverse_vol_ser.sum())


class InverseVol63VixScaledAtrNormalizedIndexStrategy(VixScaledAtrNormalizedIndexStrategy):
    """
    ATR-normalized broad-index momentum with inverse trailing-vol selected weights.
    """

    def __init__(
        self,
        *args,
        inverse_vol_window_int: int = 63,
        **kwargs,
    ):
        if inverse_vol_window_int <= 1:
            raise ValueError("inverse_vol_window_int must be greater than 1.")
        self.inverse_vol_window_int = int(inverse_vol_window_int)
        super().__init__(*args, **kwargs)

    @property
    def inverse_vol_field_str(self) -> str:
        return TRAILING_VOL_FIELD_TEMPLATE_STR.format(
            window_int=self.inverse_vol_window_int,
        )

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        signal_data_df = super().compute_signals(pricing_data=pricing_data)
        trailing_vol_feature_df = _trailing_vol_feature_df(
            pricing_data=pricing_data,
            benchmark_list=list(self._benchmarks),
            inverse_vol_window_int=self.inverse_vol_window_int,
            inverse_vol_field_str=self.inverse_vol_field_str,
        )
        return pd.concat([signal_data_df, trailing_vol_feature_df], axis=1)

    def get_target_weight_ser(self, close_row_ser: pd.Series) -> pd.Series:
        selected_base_target_weight_ser = AtrNormalizedNdxStrategy.get_target_weight_ser(
            self,
            close_row_ser=close_row_ser,
        )
        if len(selected_base_target_weight_ser) == 0:
            return selected_base_target_weight_ser

        normalized_base_weight_ser = _selected_risk_parity_weight_ser(
            close_row_ser=close_row_ser,
            selected_base_target_weight_ser=selected_base_target_weight_ser,
            inverse_vol_field_str=self.inverse_vol_field_str,
        )
        if len(normalized_base_weight_ser) == 0:
            return normalized_base_weight_ser
        exposure_scale_float = get_asof_vxn_scale_float(
            vxn_scale_signal_df=self.vxn_scale_signal_df,
            decision_date_ts=pd.Timestamp(self.previous_bar),
        )
        return normalized_base_weight_ser * exposure_scale_float


class RiskParity63AtrNormalizedIndexStrategy(AtrNormalizedNdxStrategy):
    """
    ATR-normalized broad-index momentum with pure inverse-volatility sizing.
    """

    def __init__(
        self,
        *args,
        inverse_vol_window_int: int = 63,
        selection_score_mode_str: str = SELECTION_SCORE_MODE_ATR20_STR,
        **kwargs,
    ):
        if inverse_vol_window_int <= 1:
            raise ValueError("inverse_vol_window_int must be greater than 1.")
        if selection_score_mode_str not in SELECTION_SCORE_MODE_SET:
            raise ValueError(
                "selection_score_mode_str must be one of "
                f"{sorted(SELECTION_SCORE_MODE_SET)}, got {selection_score_mode_str!r}."
            )
        self.inverse_vol_window_int = int(inverse_vol_window_int)
        self.selection_score_mode_str = str(selection_score_mode_str)
        super().__init__(*args, **kwargs)

    @property
    def inverse_vol_field_str(self) -> str:
        return TRAILING_VOL_FIELD_TEMPLATE_STR.format(
            window_int=self.inverse_vol_window_int,
        )

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        signal_data_df = super().compute_signals(pricing_data=pricing_data)
        if self.selection_score_mode_str != SELECTION_SCORE_MODE_ATR20_STR:
            selection_score_feature_df = _selection_score_feature_df(
                pricing_data=pricing_data,
                benchmark_list=list(self._benchmarks),
                lookback_month_int=self.lookback_month_int,
                selection_score_mode_str=self.selection_score_mode_str,
            )
            signal_data_df = _replace_selection_score_feature_df(
                signal_data_df=signal_data_df,
                selection_score_feature_df=selection_score_feature_df,
            )
        trailing_vol_feature_df = _trailing_vol_feature_df(
            pricing_data=pricing_data,
            benchmark_list=list(self._benchmarks),
            inverse_vol_window_int=self.inverse_vol_window_int,
            inverse_vol_field_str=self.inverse_vol_field_str,
        )
        return pd.concat([signal_data_df, trailing_vol_feature_df], axis=1)

    def get_target_weight_ser(self, close_row_ser: pd.Series) -> pd.Series:
        selected_base_target_weight_ser = super().get_target_weight_ser(
            close_row_ser=close_row_ser,
        )
        return _selected_risk_parity_weight_ser(
            close_row_ser=close_row_ser,
            selected_base_target_weight_ser=selected_base_target_weight_ser,
            inverse_vol_field_str=self.inverse_vol_field_str,
        )


class VixScaledAtrNormalizedSp500Strategy(VixScaledAtrNormalizedIndexStrategy):
    """S&P 500 PIT-universe preset."""


class VixScaledAtrNormalizedRussell1000Strategy(VixScaledAtrNormalizedIndexStrategy):
    """Russell 1000 PIT-universe preset."""


class VixScaledAtrNormalizedRussell3000Strategy(VixScaledAtrNormalizedIndexStrategy):
    """Russell 3000 PIT-universe preset."""


def get_vix_scaled_atr_normalized_index_data(
    config: VixScaledAtrNormalizedIndexConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return get_vxn_scaled_atr_normalized_ndx_data(config=config)


def get_atr_normalized_index_data(
    config: VixScaledAtrNormalizedIndexConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return get_atr_normalized_ndx_data(config=config)


def build_strategy(
    config: VixScaledAtrNormalizedIndexConfig,
    rebalance_schedule_df: pd.DataFrame,
    vix_scale_signal_df: pd.DataFrame,
) -> VixScaledAtrNormalizedIndexStrategy:
    return VixScaledAtrNormalizedIndexStrategy(
        name=config.strategy_name_str,
        benchmarks=[config.regime_symbol_str],
        rebalance_schedule_df=rebalance_schedule_df,
        vxn_scale_signal_df=vix_scale_signal_df,
        regime_symbol_str=config.regime_symbol_str,
        capital_base=config.capital_base_float,
        slippage=config.slippage_float,
        commission_per_share=config.commission_per_share_float,
        commission_minimum=config.commission_minimum_float,
        lookback_month_int=config.lookback_month_int,
        index_trend_window_int=config.index_trend_window_int,
        stock_trend_window_int=config.stock_trend_window_int,
        max_positions_int=config.max_positions_int,
    )


def build_inverse_vol_63_strategy(
    config: VixScaledAtrNormalizedIndexConfig,
    rebalance_schedule_df: pd.DataFrame,
    vix_scale_signal_df: pd.DataFrame,
) -> InverseVol63VixScaledAtrNormalizedIndexStrategy:
    if config.strategy_name_str.endswith("_vix_scaled"):
        strategy_name_str = config.strategy_name_str.replace(
            "_vix_scaled",
            "_vix_scaled_invvol63",
        )
    else:
        strategy_name_str = f"{config.strategy_name_str}_invvol63"

    return InverseVol63VixScaledAtrNormalizedIndexStrategy(
        name=strategy_name_str,
        benchmarks=[config.regime_symbol_str],
        rebalance_schedule_df=rebalance_schedule_df,
        vxn_scale_signal_df=vix_scale_signal_df,
        regime_symbol_str=config.regime_symbol_str,
        capital_base=config.capital_base_float,
        slippage=config.slippage_float,
        commission_per_share=config.commission_per_share_float,
        commission_minimum=config.commission_minimum_float,
        lookback_month_int=config.lookback_month_int,
        index_trend_window_int=config.index_trend_window_int,
        stock_trend_window_int=config.stock_trend_window_int,
        max_positions_int=config.max_positions_int,
        inverse_vol_window_int=config.inverse_vol_window_int,
    )


def build_risk_parity_63_strategy(
    config: VixScaledAtrNormalizedIndexConfig,
    rebalance_schedule_df: pd.DataFrame,
) -> RiskParity63AtrNormalizedIndexStrategy:
    if config.strategy_name_str.endswith("_vix_scaled"):
        strategy_name_str = config.strategy_name_str.replace(
            "_vix_scaled",
            "_risk_parity_63",
        )
    else:
        strategy_name_str = f"{config.strategy_name_str}_risk_parity_63"
    if config.selection_score_mode_str != SELECTION_SCORE_MODE_ATR20_STR:
        strategy_name_str = f"{strategy_name_str}_{config.selection_score_mode_str}"

    return RiskParity63AtrNormalizedIndexStrategy(
        name=strategy_name_str,
        benchmarks=[config.regime_symbol_str],
        rebalance_schedule_df=rebalance_schedule_df,
        regime_symbol_str=config.regime_symbol_str,
        capital_base=config.capital_base_float,
        slippage=config.slippage_float,
        commission_per_share=config.commission_per_share_float,
        commission_minimum=config.commission_minimum_float,
        lookback_month_int=config.lookback_month_int,
        index_trend_window_int=config.index_trend_window_int,
        stock_trend_window_int=config.stock_trend_window_int,
        max_positions_int=config.max_positions_int,
        inverse_vol_window_int=config.inverse_vol_window_int,
        selection_score_mode_str=config.selection_score_mode_str,
    )


def _with_run_overrides(
    config: VixScaledAtrNormalizedIndexConfig,
    backtest_start_date_str: str | None,
    capital_base_float: float | None,
    end_date_str: str | None,
) -> VixScaledAtrNormalizedIndexConfig:
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
    config: VixScaledAtrNormalizedIndexConfig = DEFAULT_CONFIG,
    inverse_vol_63_bool: bool = False,
    risk_parity_63_bool: bool = False,
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
) -> VixScaledAtrNormalizedIndexStrategy:
    config_obj = _with_run_overrides(
        config=config,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
    )
    if inverse_vol_63_bool and risk_parity_63_bool:
        raise ValueError("inverse_vol_63_bool and risk_parity_63_bool are mutually exclusive.")

    if risk_parity_63_bool:
        pricing_data_df, universe_df, rebalance_schedule_df = get_atr_normalized_index_data(
            config=config_obj,
        )
        strategy_obj = build_risk_parity_63_strategy(
            config=config_obj,
            rebalance_schedule_df=rebalance_schedule_df,
        )
    else:
        pricing_data_df, universe_df, rebalance_schedule_df, vix_scale_signal_df = (
            get_vix_scaled_atr_normalized_index_data(config=config_obj)
        )
    if inverse_vol_63_bool and not risk_parity_63_bool:
        strategy_obj = build_inverse_vol_63_strategy(
            config=config_obj,
            rebalance_schedule_df=rebalance_schedule_df,
            vix_scale_signal_df=vix_scale_signal_df,
        )
    elif not risk_parity_63_bool:
        strategy_obj = build_strategy(
            config=config_obj,
            rebalance_schedule_df=rebalance_schedule_df,
            vix_scale_signal_df=vix_scale_signal_df,
        )
    strategy_obj.universe_df = universe_df

    # *** CRITICAL*** Keep full pre-start history for trailing ROC, ATR, and
    # SMA features, but execute/report only from the configured start date.
    calendar_idx = pricing_data_df.index[
        pricing_data_df.index >= pd.Timestamp(config_obj.backtest_start_date_str)
    ]
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
