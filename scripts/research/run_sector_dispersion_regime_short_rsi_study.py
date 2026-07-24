"""Research-only regime, mirrored-short, and RSI-exit sector IBS study."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import talib

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from alpha.engine.backtest import run_daily
from alpha.engine.report import build_research_output_path
from data.norgate_loader import load_raw_prices
from scripts.research.run_sector_dispersion_family_universe_study import (
    COMMON_BENCHMARK_SYMBOL_STR,
    HISTORY_START_DATE_STR,
    STUDY_END_DATE_STR,
    SUBPERIOD_TUPLE,
    SectorDispersionDividendStrategy,
    _save_universe_artifacts,
    _summary_metric_float,
    _symbol_tuple_from_manifest_row,
    build_data_quality_df,
    build_execution_calendar_idx,
    build_exposure_diagnostic_dict,
    build_universe_config_obj,
    build_universe_manifest_df,
    compute_equal_weight_benchmark_return_ser,
    compute_equity_metric_dict,
    prepare_isolated_no_print_sessions,
)
from strategies.mean_reversion.strategy_mr_sector_dispersion_ibs import (
    DEFAULT_CONFIG,
    SectorDispersionIbsConfig,
)


SPY_REGIME_SYMBOL_STR = "SPY"
STUDY_UNIVERSE_ID_TUPLE = ("spdr_9", "vanguard_11", "spdr_11")
SIDE_LONG_STR = "long"
SIDE_SHORT_STR = "short"
REGIME_ABOVE_STR = "above"
REGIME_BELOW_STR = "below"
RSI_EXIT_OVERBOUGHT_STR = "overbought"
RSI_EXIT_OVERSOLD_STR = "oversold"
DEFAULT_RSI_WINDOW_DAY_INT = 2
DEFAULT_RSI_OVERBOUGHT_FLOAT = 90.0
DEFAULT_RSI_OVERSOLD_FLOAT = 10.0
DEFAULT_SHORT_BORROW_RATE_ANNUAL_FLOAT = 0.01
FAMILY_TEST_COUNT_INT = 18
HAC_MAX_LAG_INT = 5


@dataclass(frozen=True)
class VariantSpec:
    priority_int: int
    variant_id_str: str
    side_str: str
    market_sma_lookback_day_int: int | None
    required_regime_str: str | None
    rsi_exit_mode_str: str | None
    control_variant_id_str: str
    description_str: str

    def __post_init__(self) -> None:
        if self.side_str not in {SIDE_LONG_STR, SIDE_SHORT_STR}:
            raise ValueError("side_str must be 'long' or 'short'.")
        if self.required_regime_str not in {None, REGIME_ABOVE_STR, REGIME_BELOW_STR}:
            raise ValueError("required_regime_str must be None, 'above', or 'below'.")
        if (self.market_sma_lookback_day_int is None) != (self.required_regime_str is None):
            raise ValueError("Market SMA length and required regime must be specified together.")
        if self.rsi_exit_mode_str not in {
            None,
            RSI_EXIT_OVERBOUGHT_STR,
            RSI_EXIT_OVERSOLD_STR,
        }:
            raise ValueError("Unsupported RSI exit mode.")


VARIANT_SPEC_TUPLE = (
    VariantSpec(1, "B0", SIDE_LONG_STR, None, None, None, "NONE", "Baseline long"),
    VariantSpec(2, "L200", SIDE_LONG_STR, 200, REGIME_ABOVE_STR, None, "B0", "Long entries only above SPY SMA200"),
    VariantSpec(3, "S0", SIDE_SHORT_STR, None, None, None, "CASH", "Unconditional mirrored short"),
    VariantSpec(4, "S200", SIDE_SHORT_STR, 200, REGIME_BELOW_STR, None, "S0", "Short entries only below SPY SMA200"),
    VariantSpec(5, "S100", SIDE_SHORT_STR, 100, REGIME_BELOW_STR, None, "S0", "Short entries only below SPY SMA100"),
    VariantSpec(6, "L200_RSI", SIDE_LONG_STR, 200, REGIME_ABOVE_STR, RSI_EXIT_OVERBOUGHT_STR, "L200", "L200 with IBS/range AND RSI2>90 exit"),
    VariantSpec(7, "S200_RSI", SIDE_SHORT_STR, 200, REGIME_BELOW_STR, RSI_EXIT_OVERSOLD_STR, "S200", "S200 with IBS/range AND RSI2<10 cover"),
)


@dataclass(frozen=True)
class SectorDispersionRegimeShortRsiConfig:
    base_config_obj: SectorDispersionIbsConfig
    variant_spec_obj: VariantSpec
    spy_symbol_str: str = SPY_REGIME_SYMBOL_STR
    rsi_window_day_int: int = DEFAULT_RSI_WINDOW_DAY_INT
    rsi_overbought_float: float = DEFAULT_RSI_OVERBOUGHT_FLOAT
    rsi_oversold_float: float = DEFAULT_RSI_OVERSOLD_FLOAT
    short_borrow_rate_annual_float: float = DEFAULT_SHORT_BORROW_RATE_ANNUAL_FLOAT
    spy_sma_override_day_int: int | None = None

    def __post_init__(self) -> None:
        if self.rsi_window_day_int <= 1:
            raise ValueError("rsi_window_day_int must be greater than one.")
        if not 0.0 < self.rsi_oversold_float < self.rsi_overbought_float < 100.0:
            raise ValueError("RSI thresholds must satisfy 0 < oversold < overbought < 100.")
        if self.short_borrow_rate_annual_float < 0.0:
            raise ValueError("short_borrow_rate_annual_float cannot be negative.")
        if self.spy_sma_override_day_int is not None and self.spy_sma_override_day_int <= 1:
            raise ValueError("spy_sma_override_day_int must be greater than one.")

    @property
    def effective_spy_sma_lookback_day_int(self) -> int | None:
        if self.variant_spec_obj.market_sma_lookback_day_int is None:
            return None
        if self.spy_sma_override_day_int is not None:
            return int(self.spy_sma_override_day_int)
        return int(self.variant_spec_obj.market_sma_lookback_day_int)


class SectorDispersionRegimeShortRsiStrategy(SectorDispersionDividendStrategy):
    """One frozen long-only or short-only study row."""

    def __init__(
        self,
        name: str,
        benchmarks: list[str] | tuple[str, ...],
        study_config_obj: SectorDispersionRegimeShortRsiConfig,
    ):
        super().__init__(
            name=name,
            benchmarks=benchmarks,
            config_obj=study_config_obj.base_config_obj,
        )
        self.study_config_obj = study_config_obj
        self.variant_spec_obj = study_config_obj.variant_spec_obj
        self.side_sign_float = 1.0 if self.variant_spec_obj.side_str == SIDE_LONG_STR else -1.0
        self.borrow_fee_total_float = 0.0
        self._borrow_fee_row_dict_list: list[dict[str, object]] = []

    @property
    def borrow_fee_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._borrow_fee_row_dict_list)

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        signal_data_df = super().compute_signals(pricing_data_df)
        rsi_value_map: dict[str, pd.Series] = {}
        for symbol_str in self.symbol_tuple:
            close_key_tuple = (symbol_str, "Close")
            if close_key_tuple not in signal_data_df.columns:
                raise RuntimeError(f"Missing close column for RSI2: {close_key_tuple}")
            close_ser = pd.to_numeric(signal_data_df[close_key_tuple], errors="coerce")
            # *** CRITICAL*** RSI2 uses only trailing ETF closes through the
            # completed decision bar T. The resulting order fills at Open T+1.
            rsi_value_map[symbol_str] = pd.Series(
                talib.RSI(
                    close_ser.to_numpy(dtype=float),
                    timeperiod=int(self.study_config_obj.rsi_window_day_int),
                ),
                index=close_ser.index,
                dtype=float,
            )
        rsi_value_df = pd.DataFrame(rsi_value_map, index=signal_data_df.index)
        rsi_value_df.columns = pd.MultiIndex.from_tuples(
            [(symbol_str, "rsi2_value_ser") for symbol_str in rsi_value_df.columns]
        )
        signal_data_df = pd.concat([signal_data_df, rsi_value_df], axis=1)

        sma_lookback_day_int = self.study_config_obj.effective_spy_sma_lookback_day_int
        if sma_lookback_day_int is None:
            return signal_data_df

        spy_close_key_tuple = (self.study_config_obj.spy_symbol_str, "Close")
        if spy_close_key_tuple not in signal_data_df.columns:
            raise RuntimeError(f"Missing SPY regime close column: {spy_close_key_tuple}")
        spy_close_ser = pd.to_numeric(signal_data_df[spy_close_key_tuple], errors="coerce")
        # *** CRITICAL*** SMA_T includes the completed SPY Close_T. The regime
        # is known after T close and cannot affect an order before Open T+1.
        spy_sma_ser = spy_close_ser.rolling(
            window=sma_lookback_day_int,
            min_periods=sma_lookback_day_int,
        ).mean()
        valid_sma_bool_ser = spy_sma_ser.notna() & spy_close_ser.notna()
        above_sma_bool_ser = pd.Series(False, index=spy_close_ser.index, dtype=bool)
        below_sma_bool_ser = pd.Series(False, index=spy_close_ser.index, dtype=bool)
        above_sma_bool_ser.loc[valid_sma_bool_ser] = spy_close_ser.loc[
            valid_sma_bool_ser
        ].gt(spy_sma_ser.loc[valid_sma_bool_ser])
        below_sma_bool_ser.loc[valid_sma_bool_ser] = spy_close_ser.loc[
            valid_sma_bool_ser
        ].lt(spy_sma_ser.loc[valid_sma_bool_ser])
        regime_feature_df = pd.DataFrame(
            {
                (self.study_config_obj.spy_symbol_str, f"sma_{sma_lookback_day_int}_ser"): spy_sma_ser,
                (self.study_config_obj.spy_symbol_str, "above_sma_regime_bool"): above_sma_bool_ser,
                (self.study_config_obj.spy_symbol_str, "below_sma_regime_bool"): below_sma_bool_ser,
            },
            index=signal_data_df.index,
        )
        regime_feature_df.columns = pd.MultiIndex.from_tuples(regime_feature_df.columns)
        return pd.concat([signal_data_df, regime_feature_df], axis=1)

    def iterate(
        self,
        data_df: pd.DataFrame,
        close_row_ser: pd.Series,
        open_price_ser: pd.Series,
    ) -> None:
        if close_row_ser is None or data_df is None:
            return

        position_ser = self.get_positions()
        held_symbol_set = {
            str(symbol_str)
            for symbol_str, position_float in position_ser.items()
            if str(symbol_str) in self.symbol_tuple
            and float(position_float) * self.side_sign_float > 0.0
        }

        for symbol_str in self.symbol_tuple:
            if symbol_str not in held_symbol_set:
                continue
            if not self._exit_allowed_bool(symbol_str=symbol_str, close_row_ser=close_row_ser):
                continue
            self.order_target(
                symbol_str,
                0.0,
                trade_id=self.current_trade_map[symbol_str],
            )
            held_symbol_set.remove(symbol_str)

        if not self._market_entry_allowed_bool(close_row_ser=close_row_ser):
            return

        for symbol_str in self.symbol_tuple:
            if symbol_str in held_symbol_set or self.get_position(symbol_str) != 0:
                continue
            if not self._entry_signal_bool(symbol_str=symbol_str, close_row_ser=close_row_ser):
                continue
            close_price_float = float(close_row_ser.get((symbol_str, "Close"), np.nan))
            if not np.isfinite(close_price_float) or close_price_float <= 0.0:
                raise RuntimeError(
                    f"Cannot size {symbol_str} entry without a valid decision-bar close."
                )
            target_share_float = (
                self.side_sign_float
                * float(self.previous_total_value)
                * float(self.target_weight_float)
                / close_price_float
            )
            self.trade_id_int += 1
            self.current_trade_map[symbol_str] = self.trade_id_int
            self.order_target(
                symbol_str,
                target_share_float,
                trade_id=self.trade_id_int,
            )

    def _market_entry_allowed_bool(self, close_row_ser: pd.Series) -> bool:
        regime_str = self.variant_spec_obj.required_regime_str
        if regime_str is None:
            return True
        regime_field_str = (
            "above_sma_regime_bool"
            if regime_str == REGIME_ABOVE_STR
            else "below_sma_regime_bool"
        )
        regime_value_obj = close_row_ser.get(
            (self.study_config_obj.spy_symbol_str, regime_field_str),
            False,
        )
        return False if pd.isna(regime_value_obj) else bool(regime_value_obj)

    def _entry_signal_bool(self, symbol_str: str, close_row_ser: pd.Series) -> bool:
        signal_field_str = (
            "entry_signal_bool"
            if self.variant_spec_obj.side_str == SIDE_LONG_STR
            else "exit_signal_bool"
        )
        return bool(close_row_ser.get((symbol_str, signal_field_str), False))

    def _exit_allowed_bool(self, symbol_str: str, close_row_ser: pd.Series) -> bool:
        signal_field_str = (
            "exit_signal_bool"
            if self.variant_spec_obj.side_str == SIDE_LONG_STR
            else "entry_signal_bool"
        )
        if not bool(close_row_ser.get((symbol_str, signal_field_str), False)):
            return False
        if self.variant_spec_obj.rsi_exit_mode_str is None:
            return True
        rsi_value_float = float(close_row_ser.get((symbol_str, "rsi2_value_ser"), np.nan))
        if not np.isfinite(rsi_value_float):
            return False
        if self.variant_spec_obj.rsi_exit_mode_str == RSI_EXIT_OVERBOUGHT_STR:
            return rsi_value_float > float(self.study_config_obj.rsi_overbought_float)
        return rsi_value_float < float(self.study_config_obj.rsi_oversold_float)

    def _accrue_current_bar_borrow_fee(self, pricing_data_df: pd.DataFrame) -> None:
        if self.variant_spec_obj.side_str != SIDE_SHORT_STR:
            return
        current_bar_ts = pd.Timestamp(self.current_bar)
        prior_bar_index = pricing_data_df.index[pricing_data_df.index < current_bar_ts]
        if len(prior_bar_index) == 0:
            return
        prior_bar_ts = pd.Timestamp(prior_bar_index[-1])
        calendar_day_count_int = int((current_bar_ts - prior_bar_ts).days)
        if calendar_day_count_int <= 0:
            raise RuntimeError("Borrow accrual requires increasing calendar dates.")

        preopen_position_ser = self.get_positions()
        for symbol_str in self.symbol_tuple:
            position_share_float = float(preopen_position_ser.get(symbol_str, 0.0))
            if position_share_float >= 0.0:
                continue
            prior_close_float = float(
                pd.to_numeric(
                    pd.Series([pricing_data_df.loc[prior_bar_ts, (symbol_str, "Close")]]),
                    errors="coerce",
                ).iloc[0]
            )
            if not np.isfinite(prior_close_float) or prior_close_float <= 0.0:
                raise RuntimeError(
                    f"Cannot accrue {symbol_str} borrow without prior close on {prior_bar_ts.date()}."
                )
            borrow_fee_float = (
                abs(position_share_float)
                * prior_close_float
                * float(self.study_config_obj.short_borrow_rate_annual_float)
                * float(calendar_day_count_int)
                / 365.0
            )
            self.cash -= borrow_fee_float
            self.borrow_fee_total_float += borrow_fee_float
            self._borrow_fee_row_dict_list.append(
                {
                    "bar": current_bar_ts,
                    "asset": symbol_str,
                    "position_share_float": position_share_float,
                    "prior_close_float": prior_close_float,
                    "calendar_day_count_int": calendar_day_count_int,
                    "borrow_rate_annual_float": float(
                        self.study_config_obj.short_borrow_rate_annual_float
                    ),
                    "borrow_fee_float": borrow_fee_float,
                }
            )

    def process_orders(self, prices: pd.DataFrame):
        # *** CRITICAL*** Borrow is charged to positions held before the current
        # open, before any Open_T order changes the position.
        self._accrue_current_bar_borrow_fee(pricing_data_df=prices)
        return super().process_orders(prices)


def compute_hac_mean_test_dict(
    difference_return_ser: pd.Series,
    family_test_count_int: int = FAMILY_TEST_COUNT_INT,
) -> dict[str, float | int]:
    clean_difference_return_ser = pd.to_numeric(
        difference_return_ser,
        errors="coerce",
    ).dropna()
    if family_test_count_int <= 0:
        raise ValueError("family_test_count_int must be positive.")
    if len(clean_difference_return_ser) < 3:
        return {
            "hac_observation_count_int": int(len(clean_difference_return_ser)),
            "mean_daily_difference_pct_float": np.nan,
            "hac_mean_t_stat_float": np.nan,
            "raw_p_value_float": np.nan,
            "bonferroni_p_value_float": np.nan,
        }

    # *** CRITICAL*** This is post-run inference on realized daily return
    # differences. It never feeds signal selection or the backtest state.
    design_mat = np.ones((len(clean_difference_return_ser), 1), dtype=float)
    regression_obj = sm.OLS(
        clean_difference_return_ser.to_numpy(dtype=float),
        design_mat,
    ).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_MAX_LAG_INT})
    raw_p_value_float = float(regression_obj.pvalues[0])
    return {
        "hac_observation_count_int": int(len(clean_difference_return_ser)),
        "mean_daily_difference_pct_float": float(clean_difference_return_ser.mean() * 100.0),
        "hac_mean_t_stat_float": float(regression_obj.tvalues[0]),
        "raw_p_value_float": raw_p_value_float,
        "bonferroni_p_value_float": min(
            1.0,
            raw_p_value_float * float(family_test_count_int),
        ),
    }


def build_study_universe_manifest_df() -> pd.DataFrame:
    family_manifest_df = build_universe_manifest_df()
    study_manifest_df = family_manifest_df.loc[
        family_manifest_df["universe_id_str"].isin(STUDY_UNIVERSE_ID_TUPLE)
    ].copy()
    priority_map_dict = {
        universe_id_str: priority_int
        for priority_int, universe_id_str in enumerate(STUDY_UNIVERSE_ID_TUPLE, start=1)
    }
    study_manifest_df["study_priority_int"] = study_manifest_df["universe_id_str"].map(
        priority_map_dict
    )
    return study_manifest_df.sort_values("study_priority_int").reset_index(drop=True)


def build_variant_manifest_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "priority_int": spec_obj.priority_int,
                "variant_id_str": spec_obj.variant_id_str,
                "side_str": spec_obj.side_str,
                "market_sma_lookback_day_int": spec_obj.market_sma_lookback_day_int,
                "required_regime_str": spec_obj.required_regime_str,
                "rsi_exit_mode_str": spec_obj.rsi_exit_mode_str,
                "control_variant_id_str": spec_obj.control_variant_id_str,
                "description_str": spec_obj.description_str,
            }
            for spec_obj in VARIANT_SPEC_TUPLE
        ]
    )


def _market_metric_dict(
    strategy_return_ser: pd.Series,
    market_return_ser: pd.Series,
) -> dict[str, float]:
    paired_return_df = pd.concat(
        [
            pd.to_numeric(strategy_return_ser, errors="coerce").rename("strategy"),
            pd.to_numeric(market_return_ser, errors="coerce").rename("market"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if len(paired_return_df) < 3:
        return {"corr_to_spx_float": np.nan, "beta_to_spx_float": np.nan}
    market_variance_float = float(paired_return_df["market"].var())
    beta_float = (
        np.nan
        if np.isclose(market_variance_float, 0.0)
        else float(
            paired_return_df["strategy"].cov(paired_return_df["market"])
            / market_variance_float
        )
    )
    return {
        "corr_to_spx_float": float(
            paired_return_df["strategy"].corr(paired_return_df["market"])
        ),
        "beta_to_spx_float": beta_float,
    }


def _net_exposure_metric_dict(
    strategy_obj: SectorDispersionRegimeShortRsiStrategy,
) -> dict[str, float]:
    realized_weight_df = strategy_obj.realized_weight_df.copy()
    realized_weight_df.columns = [str(column_obj) for column_obj in realized_weight_df.columns]
    symbol_weight_df = realized_weight_df.reindex(
        columns=list(strategy_obj.symbol_tuple),
        fill_value=0.0,
    ).fillna(0.0)
    net_exposure_ser = symbol_weight_df.sum(axis=1)
    return {
        "average_net_exposure_pct_float": float(net_exposure_ser.mean() * 100.0),
        "minimum_net_exposure_pct_float": float(net_exposure_ser.min() * 100.0),
        "maximum_net_exposure_pct_float": float(net_exposure_ser.max() * 100.0),
    }


def _trade_metric_dict(
    strategy_obj: SectorDispersionRegimeShortRsiStrategy,
) -> dict[str, float | int]:
    trade_df = getattr(strategy_obj, "_trades", pd.DataFrame())
    if trade_df is None or len(trade_df) == 0:
        return {
            "trade_count_int": 0,
            "win_rate_pct_float": np.nan,
            "median_holding_days_float": np.nan,
            "average_holding_days_float": np.nan,
            "worst_trade_return_pct_float": np.nan,
            "best_trade_return_pct_float": np.nan,
        }
    holding_day_ser = pd.to_timedelta(trade_df["duration"]).dt.total_seconds() / 86400.0
    profit_ser = pd.to_numeric(trade_df["profit"], errors="coerce")
    trade_return_ser = pd.to_numeric(trade_df["return"], errors="coerce")
    return {
        "trade_count_int": int(len(trade_df)),
        "win_rate_pct_float": float(profit_ser.gt(0.0).mean() * 100.0),
        "median_holding_days_float": float(holding_day_ser.median()),
        "average_holding_days_float": float(holding_day_ser.mean()),
        "worst_trade_return_pct_float": float(trade_return_ser.min() * 100.0),
        "best_trade_return_pct_float": float(trade_return_ser.max() * 100.0),
    }


def _annual_return_row_dict_list(
    universe_id_str: str,
    variant_id_str: str,
    strategy_return_ser: pd.Series,
) -> list[dict[str, object]]:
    row_dict_list: list[dict[str, object]] = []
    clean_return_ser = pd.to_numeric(strategy_return_ser, errors="coerce").dropna()
    for year_int, year_return_ser in clean_return_ser.groupby(clean_return_ser.index.year):
        row_dict_list.append(
            {
                "universe_id_str": universe_id_str,
                "variant_id_str": variant_id_str,
                "year_int": int(year_int),
                "return_pct_float": float(((1.0 + year_return_ser).prod() - 1.0) * 100.0),
                "observation_count_int": int(len(year_return_ser)),
            }
        )
    return row_dict_list


def _subperiod_row_dict_list(
    universe_id_str: str,
    variant_id_str: str,
    strategy_total_value_ser: pd.Series,
) -> list[dict[str, object]]:
    row_dict_list: list[dict[str, object]] = []
    for period_id_str, period_start_ts, period_end_ts in SUBPERIOD_TUPLE:
        period_equity_ser = strategy_total_value_ser.loc[
            (strategy_total_value_ser.index >= period_start_ts)
            & (strategy_total_value_ser.index <= period_end_ts)
        ]
        row_dict_list.append(
            {
                "universe_id_str": universe_id_str,
                "variant_id_str": variant_id_str,
                "period_id_str": period_id_str,
                **compute_equity_metric_dict(period_equity_ser, prefix_str="period"),
            }
        )
    return row_dict_list


def _markdown_table_str(source_df: pd.DataFrame, column_list: list[str]) -> str:
    table_df = source_df.loc[:, column_list].copy()
    header_str = "| " + " | ".join(column_list) + " |"
    separator_str = "| " + " | ".join("---" for _ in column_list) + " |"
    row_str_list: list[str] = []
    for _, row_ser in table_df.iterrows():
        value_str_list: list[str] = []
        for value_obj in row_ser.tolist():
            if isinstance(value_obj, (float, np.floating)):
                value_str_list.append(
                    "" if not np.isfinite(float(value_obj)) else f"{float(value_obj):.3f}"
                )
            else:
                value_str_list.append(str(value_obj))
        row_str_list.append("| " + " | ".join(value_str_list) + " |")
    return "\n".join([header_str, separator_str, *row_str_list])


def _save_common_equity_chart(
    output_path: Path,
    common_equity_df: pd.DataFrame,
) -> None:
    figure_obj, axis_list = plt.subplots(3, 1, figsize=(15, 15), constrained_layout=True)
    for axis_obj, universe_id_str in zip(axis_list, STUDY_UNIVERSE_ID_TUPLE):
        for variant_spec_obj in VARIANT_SPEC_TUPLE:
            column_str = f"{universe_id_str}_{variant_spec_obj.variant_id_str}"
            axis_obj.plot(
                common_equity_df.index,
                common_equity_df[column_str],
                label=variant_spec_obj.variant_id_str,
                linewidth=1.4,
            )
        axis_obj.set_title(universe_id_str)
        axis_obj.set_yscale("log")
        axis_obj.grid(alpha=0.25)
        axis_obj.legend(ncol=4, frameon=False)
    figure_obj.suptitle("Sector Dispersion Regime / Short / RSI: Common-Overlap Equity")
    figure_obj.savefig(output_path / "common_overlap_equity_curves.png", dpi=180)
    plt.close(figure_obj)


def run_regime_short_rsi_study(
    output_dir_str: str = "results",
    end_date_str: str = STUDY_END_DATE_STR,
    show_progress_bool: bool = False,
) -> Path:
    universe_manifest_df = build_study_universe_manifest_df()
    variant_manifest_df = build_variant_manifest_df()
    all_trade_symbol_tuple = tuple(
        dict.fromkeys(
            symbol_str
            for _, manifest_row_ser in universe_manifest_df.iterrows()
            for symbol_str in _symbol_tuple_from_manifest_row(manifest_row_ser)
        )
    )
    raw_pricing_data_df = load_raw_prices(
        symbols=[*all_trade_symbol_tuple, SPY_REGIME_SYMBOL_STR],
        benchmarks=[COMMON_BENCHMARK_SYMBOL_STR],
        start_date=HISTORY_START_DATE_STR,
        end_date=end_date_str,
    )
    pricing_data_df, stale_session_df = prepare_isolated_no_print_sessions(
        pricing_data_df=raw_pricing_data_df,
        universe_manifest_df=universe_manifest_df,
    )
    data_quality_df = build_data_quality_df(
        pricing_data_df=pricing_data_df,
        universe_manifest_df=universe_manifest_df,
    )
    bad_data_quality_df = data_quality_df.loc[
        ~data_quality_df["status_str"].astype(str).str.startswith("ok")
    ]
    if len(bad_data_quality_df) > 0:
        raise RuntimeError(
            "Study data audit failed: "
            f"{bad_data_quality_df[['symbol_str', 'status_str']].to_dict('records')}"
        )
    spy_close_ser = pd.to_numeric(
        pricing_data_df[(SPY_REGIME_SYMBOL_STR, "Close")],
        errors="coerce",
    )
    if spy_close_ser.dropna().empty or spy_close_ser.loc[spy_close_ser.first_valid_index() :].isna().any():
        raise RuntimeError("SPY regime history contains a post-inception missing close.")

    timestamp_str = pd.Timestamp.now().strftime("%Y-%m-%d_%H%M%S")
    output_path = build_research_output_path(
        output_dir=output_dir_str,
        entity_type_str="strategy",
        entity_id_str="strategy_mr_sector_dispersion_ibs",
        analysis_type_str="regime_short_rsi_study",
        timestamp_str=timestamp_str,
    )
    output_path.mkdir(parents=True, exist_ok=False)
    universe_manifest_df.to_csv(output_path / "universe_manifest.csv", index=False)
    variant_manifest_df.to_csv(output_path / "variant_manifest.csv", index=False)
    data_quality_df.to_csv(output_path / "data_quality.csv", index=False)
    stale_session_df.to_csv(output_path / "stale_sessions.csv", index=False)

    config_by_universe_dict: dict[str, SectorDispersionIbsConfig] = {}
    calendar_by_universe_dict: dict[str, pd.DatetimeIndex] = {}
    equal_weight_return_by_universe_dict: dict[str, pd.Series] = {}
    equal_weight_equity_by_universe_dict: dict[str, pd.Series] = {}
    for _, manifest_row_ser in universe_manifest_df.iterrows():
        universe_id_str = str(manifest_row_ser["universe_id_str"])
        config_obj = build_universe_config_obj(manifest_row_ser, end_date_str=end_date_str)
        calendar_idx = build_execution_calendar_idx(pricing_data_df, config_obj=config_obj)
        config_by_universe_dict[universe_id_str] = config_obj
        calendar_by_universe_dict[universe_id_str] = calendar_idx
        equal_weight_return_ser = compute_equal_weight_benchmark_return_ser(
            pricing_data_df=pricing_data_df,
            symbol_tuple=config_obj.symbol_tuple,
            calendar_idx=calendar_idx,
        )
        equal_weight_return_by_universe_dict[universe_id_str] = equal_weight_return_ser
        equal_weight_equity_by_universe_dict[universe_id_str] = (
            (1.0 + equal_weight_return_ser).cumprod() * config_obj.capital_base_float
        )

    common_overlap_start_ts = max(
        calendar_idx[0] for calendar_idx in calendar_by_universe_dict.values()
    )
    market_close_ser = pd.to_numeric(
        pricing_data_df[(COMMON_BENCHMARK_SYMBOL_STR, "Close")],
        errors="coerce",
    )
    market_return_ser = market_close_ser.pct_change(fill_method=None).dropna()

    comparison_row_dict_list: list[dict[str, object]] = []
    annual_return_row_dict_list: list[dict[str, object]] = []
    subperiod_row_dict_list: list[dict[str, object]] = []
    common_equity_df = pd.DataFrame()

    for universe_index_int, universe_id_str in enumerate(STUDY_UNIVERSE_ID_TUPLE, start=1):
        config_obj = config_by_universe_dict[universe_id_str]
        calendar_idx = calendar_by_universe_dict[universe_id_str]
        equal_weight_equity_ser = equal_weight_equity_by_universe_dict[universe_id_str]
        variant_return_dict: dict[str, pd.Series] = {}
        variant_metric_dict: dict[str, dict[str, object]] = {}

        for variant_index_int, variant_spec_obj in enumerate(VARIANT_SPEC_TUPLE, start=1):
            print(
                f"Running universe {universe_index_int}/3 {universe_id_str}, "
                f"variant {variant_index_int}/7 {variant_spec_obj.variant_id_str}...",
                flush=True,
            )
            study_config_obj = SectorDispersionRegimeShortRsiConfig(
                base_config_obj=config_obj,
                variant_spec_obj=variant_spec_obj,
            )
            strategy_obj = SectorDispersionRegimeShortRsiStrategy(
                name=(
                    "strategy_mr_sector_dispersion_ibs_"
                    f"{universe_id_str}_{variant_spec_obj.variant_id_str.lower()}"
                ),
                benchmarks=[COMMON_BENCHMARK_SYMBOL_STR],
                study_config_obj=study_config_obj,
            )
            strategy_obj._performance_benchmark_symbol_str = COMMON_BENCHMARK_SYMBOL_STR
            strategy_obj._performance_benchmark_adjustment_str = "TOTALRETURN"
            run_daily(
                strategy_obj,
                pricing_data_df,
                calendar=calendar_idx,
                show_progress=show_progress_bool,
                show_signal_progress_bool=show_progress_bool,
                audit_override_bool=True,
            )

            artifact_id_str = f"{universe_id_str}/{variant_spec_obj.variant_id_str}"
            _save_universe_artifacts(
                output_path=output_path,
                universe_id_str=artifact_id_str,
                strategy_obj=strategy_obj,
                equal_weight_return_ser=equal_weight_return_by_universe_dict[universe_id_str],
                equal_weight_total_value_ser=equal_weight_equity_ser,
            )
            variant_output_path = output_path / "universes" / universe_id_str / variant_spec_obj.variant_id_str
            strategy_obj.borrow_fee_df.to_csv(variant_output_path / "borrow_fees.csv", index=False)

            strategy_total_value_ser = pd.to_numeric(
                strategy_obj.results["total_value"],
                errors="coerce",
            ).astype(float)
            strategy_return_ser = pd.to_numeric(
                strategy_obj.results["daily_returns"],
                errors="coerce",
            ).astype(float)
            full_metric_dict = compute_equity_metric_dict(
                strategy_total_value_ser,
                prefix_str="full_strategy",
            )
            common_metric_dict = compute_equity_metric_dict(
                strategy_total_value_ser.loc[
                    strategy_total_value_ser.index >= common_overlap_start_ts
                ],
                prefix_str="common_strategy",
            )
            variant_return_dict[variant_spec_obj.variant_id_str] = strategy_return_ser
            variant_metric_dict[variant_spec_obj.variant_id_str] = full_metric_dict

            if variant_spec_obj.control_variant_id_str == "CASH":
                control_return_ser = pd.Series(0.0, index=strategy_return_ser.index)
                control_ann_return_float = 0.0
                control_sharpe_float = np.nan
                control_max_drawdown_float = 0.0
            elif variant_spec_obj.control_variant_id_str in variant_return_dict:
                control_return_ser = variant_return_dict[
                    variant_spec_obj.control_variant_id_str
                ]
                control_metric_dict = variant_metric_dict[
                    variant_spec_obj.control_variant_id_str
                ]
                control_ann_return_float = float(
                    control_metric_dict["full_strategy_ann_return_pct_float"]
                )
                control_sharpe_float = float(
                    control_metric_dict["full_strategy_sharpe_float"]
                )
                control_max_drawdown_float = float(
                    control_metric_dict["full_strategy_max_drawdown_pct_float"]
                )
            else:
                control_return_ser = pd.Series(dtype=float)
                control_ann_return_float = np.nan
                control_sharpe_float = np.nan
                control_max_drawdown_float = np.nan

            if variant_spec_obj.control_variant_id_str == "NONE":
                hac_metric_dict = compute_hac_mean_test_dict(pd.Series(dtype=float))
            else:
                paired_control_df = pd.concat(
                    [
                        strategy_return_ser.rename("strategy"),
                        control_return_ser.rename("control"),
                    ],
                    axis=1,
                    join="inner",
                ).dropna()
                hac_metric_dict = compute_hac_mean_test_dict(
                    paired_control_df["strategy"] - paired_control_df["control"]
                )

            exposure_metric_dict = build_exposure_diagnostic_dict(
                realized_weight_df=strategy_obj.realized_weight_df,
                result_df=strategy_obj.results,
                symbol_tuple=strategy_obj.symbol_tuple,
            )
            trade_metric_dict = _trade_metric_dict(strategy_obj)
            trading_cost_drag_float = _summary_metric_float(
                strategy_obj.summary,
                "Cost Drag (Ann.) [%]",
            )
            average_equity_float = float(strategy_total_value_ser.mean())
            observation_count_int = int(len(strategy_total_value_ser))
            borrow_drag_ann_pct_float = (
                0.0
                if observation_count_int == 0 or average_equity_float <= 0.0
                else float(
                    strategy_obj.borrow_fee_total_float
                    / average_equity_float
                    * 252.0
                    / float(observation_count_int)
                    * 100.0
                )
            )
            full_equal_weight_metric_dict = compute_equity_metric_dict(
                equal_weight_equity_ser,
                prefix_str="full_equal_weight",
            )
            comparison_row_dict_list.append(
                {
                    "universe_id_str": universe_id_str,
                    "variant_priority_int": variant_spec_obj.priority_int,
                    "variant_id_str": variant_spec_obj.variant_id_str,
                    "side_str": variant_spec_obj.side_str,
                    "market_sma_lookback_day_int": variant_spec_obj.market_sma_lookback_day_int,
                    "required_regime_str": variant_spec_obj.required_regime_str,
                    "rsi_exit_mode_str": variant_spec_obj.rsi_exit_mode_str,
                    "control_variant_id_str": variant_spec_obj.control_variant_id_str,
                    "symbol_count_int": len(strategy_obj.symbol_tuple),
                    "target_abs_weight_pct_float": float(strategy_obj.target_weight_float * 100.0),
                    "turnover_ann_pct_float": _summary_metric_float(
                        strategy_obj.summary,
                        "Turnover (Ann.) [%]",
                    ),
                    "trading_cost_drag_ann_pct_float": trading_cost_drag_float,
                    "borrow_drag_ann_pct_float": borrow_drag_ann_pct_float,
                    "total_modeled_cost_drag_ann_pct_float": (
                        trading_cost_drag_float + borrow_drag_ann_pct_float
                    ),
                    "dividend_cash_total_float": strategy_obj.dividend_cash_total_float,
                    "borrow_fee_total_float": strategy_obj.borrow_fee_total_float,
                    "delta_ann_return_vs_control_pct_float": (
                        float(full_metric_dict["full_strategy_ann_return_pct_float"])
                        - control_ann_return_float
                    ),
                    "delta_sharpe_vs_control_float": (
                        float(full_metric_dict["full_strategy_sharpe_float"])
                        - control_sharpe_float
                    ) if np.isfinite(control_sharpe_float) else np.nan,
                    "delta_max_drawdown_vs_control_pct_float": (
                        float(full_metric_dict["full_strategy_max_drawdown_pct_float"])
                        - control_max_drawdown_float
                    ),
                    **hac_metric_dict,
                    **_market_metric_dict(strategy_return_ser, market_return_ser),
                    **exposure_metric_dict,
                    **_net_exposure_metric_dict(strategy_obj),
                    **trade_metric_dict,
                    **full_metric_dict,
                    **common_metric_dict,
                    **full_equal_weight_metric_dict,
                }
            )
            annual_return_row_dict_list.extend(
                _annual_return_row_dict_list(
                    universe_id_str=universe_id_str,
                    variant_id_str=variant_spec_obj.variant_id_str,
                    strategy_return_ser=strategy_return_ser,
                )
            )
            subperiod_row_dict_list.extend(
                _subperiod_row_dict_list(
                    universe_id_str=universe_id_str,
                    variant_id_str=variant_spec_obj.variant_id_str,
                    strategy_total_value_ser=strategy_total_value_ser,
                )
            )
            common_equity_ser = strategy_total_value_ser.loc[
                strategy_total_value_ser.index >= common_overlap_start_ts
            ]
            common_equity_df[
                f"{universe_id_str}_{variant_spec_obj.variant_id_str}"
            ] = common_equity_ser / common_equity_ser.iloc[0]

    comparison_df = pd.DataFrame(comparison_row_dict_list).sort_values(
        ["universe_id_str", "variant_priority_int"]
    )
    annual_return_df = pd.DataFrame(annual_return_row_dict_list)
    subperiod_df = pd.DataFrame(subperiod_row_dict_list)
    comparison_df.to_csv(output_path / "comparison.csv", index=False)
    annual_return_df.to_csv(output_path / "annual_returns.csv", index=False)
    subperiod_df.to_csv(output_path / "subperiod_metrics.csv", index=False)
    common_equity_df.to_csv(output_path / "common_overlap_equity.csv")
    _save_common_equity_chart(output_path, common_equity_df)

    summary_column_list = [
        "universe_id_str",
        "variant_id_str",
        "full_strategy_ann_return_pct_float",
        "full_strategy_sharpe_float",
        "full_strategy_max_drawdown_pct_float",
        "corr_to_spx_float",
        "beta_to_spx_float",
        "average_gross_exposure_pct_float",
        "total_modeled_cost_drag_ann_pct_float",
        "delta_ann_return_vs_control_pct_float",
        "bonferroni_p_value_float",
    ]
    summary_md_str = f"""# Sector Dispersion Regime / Short / RSI Study

- Research only; no live/release wiring.
- Search: 21 rows, comprising 18 new controlled comparisons and 3 baselines.
- Signal: completed daily bar T.
- Execution: Open T+1.
- Market regime: SPY capital close versus SMA100/SMA200.
- RSI2: TA-Lib 0-100 scale, 90/10 thresholds.
- Short borrow: fixed 1.00% annual plus signed ETF distributions.
- Common overlap: {common_overlap_start_ts.date().isoformat()} to {end_date_str}.

## Mechanical Comparison

{_markdown_table_str(comparison_df, summary_column_list)}

This table is not the verdict. The final decision must apply the preregistered
cross-universe, subperiod, cost, tail, and multiple-comparison rules.
"""
    (output_path / "study_summary.md").write_text(summary_md_str, encoding="utf-8")
    metadata_dict = {
        "analysis_type_str": "regime_short_rsi_study",
        "research_only_bool": True,
        "output_path_str": str(output_path.resolve()),
        "study_end_date_str": end_date_str,
        "common_overlap_start_date_str": common_overlap_start_ts.date().isoformat(),
        "universe_id_tuple": STUDY_UNIVERSE_ID_TUPLE,
        "variant_id_tuple": tuple(spec_obj.variant_id_str for spec_obj in VARIANT_SPEC_TUPLE),
        "strategy_row_count_int": len(STUDY_UNIVERSE_ID_TUPLE) * len(VARIANT_SPEC_TUPLE),
        "new_comparison_count_int": FAMILY_TEST_COUNT_INT,
        "execution_mapping_str": "completed daily signal T -> Open T+1",
        "spy_adjustment_str": "CAPITALSPECIAL",
        "market_benchmark_adjustment_str": "TOTALRETURN",
        "short_borrow_rate_annual_float": DEFAULT_SHORT_BORROW_RATE_ANNUAL_FLOAT,
        "cash_yield_float": 0.0,
        "slippage_float": DEFAULT_CONFIG.slippage_float,
        "commission_per_share_float": DEFAULT_CONFIG.commission_per_share_float,
        "bonferroni_family_test_count_int": FAMILY_TEST_COUNT_INT,
        "hac_max_lag_int": HAC_MAX_LAG_INT,
        "preregistration_path_str": str(
            (
                REPO_ROOT_PATH
                / "docs"
                / "research"
                / "SECTOR_DISPERSION_REGIME_SHORT_RSI_PREREGISTRATION.md"
            ).resolve()
        ),
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata_dict, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Saved regime/short/RSI study to {output_path.resolve()}", flush=True)
    return output_path


def parse_args(argv_list: list[str] | None = None) -> argparse.Namespace:
    parser_obj = argparse.ArgumentParser(
        description="Run the preregistered sector regime, short, and RSI exit study."
    )
    parser_obj.add_argument("--output-dir", default="results")
    parser_obj.add_argument("--end-date", default=STUDY_END_DATE_STR)
    parser_obj.add_argument("--show-progress", action="store_true")
    return parser_obj.parse_args(argv_list)


def main(argv_list: list[str] | None = None) -> None:
    args_obj = parse_args(argv_list)
    run_regime_short_rsi_study(
        output_dir_str=str(args_obj.output_dir),
        end_date_str=str(args_obj.end_date),
        show_progress_bool=bool(args_obj.show_progress),
    )


if __name__ == "__main__":
    main()
