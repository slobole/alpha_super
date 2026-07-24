"""Research-only four-positioning study for the daily sector IBS strategy."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    SectorDispersionDividendStrategy,
    _save_universe_artifacts,
    _summary_metric_float,
    _symbol_tuple_from_manifest_row,
    build_data_quality_df,
    build_execution_calendar_idx,
    build_exposure_diagnostic_dict,
    build_universe_config_obj,
    compute_equal_weight_benchmark_return_ser,
    compute_equity_metric_dict,
    prepare_isolated_no_print_sessions,
)
from scripts.research.run_sector_dispersion_regime_short_rsi_study import (
    SPY_REGIME_SYMBOL_STR,
    _annual_return_row_dict_list,
    _market_metric_dict,
    _markdown_table_str,
    _net_exposure_metric_dict,
    _subperiod_row_dict_list,
    _trade_metric_dict,
    build_study_universe_manifest_df,
    compute_hac_mean_test_dict,
)
from strategies.mean_reversion.strategy_mr_sector_dispersion_ibs import (
    DEFAULT_CONFIG,
    SectorDispersionIbsConfig,
    SectorDispersionIbsStrategy,
)


STUDY_UNIVERSE_ID_TUPLE = ("spdr_9", "vanguard_11", "spdr_11")
SIZING_EQUAL_STR = "equal"
SIZING_INVOL20_STR = "invol20"
DEFAULT_INVOL_LOOKBACK_DAY_INT = 20
DEFAULT_SPY_SMA_LOOKBACK_DAY_INT = 200
DEFAULT_SOFT_BEAR_SCALE_FLOAT = 0.50
FAMILY_TEST_COUNT_INT = 18
STRICT_CASH_TOLERANCE_FLOAT = 0.01
INVERSE_VOLATILITY_FIELD_STR = "volatility_ann_20_ser"
INVERSE_VOL_TARGET_WEIGHT_FIELD_STR = "inverse_vol_target_weight_ser"
MARKET_SCALE_FIELD_STR = "market_scale_ser"


@dataclass(frozen=True)
class PositioningVariantSpec:
    priority_int: int
    variant_id_str: str
    sizing_mode_str: str
    uses_soft_sma_bool: bool
    strict_cash_cap_bool: bool
    primary_control_variant_id_str: str
    description_str: str

    def __post_init__(self) -> None:
        if self.sizing_mode_str not in {SIZING_EQUAL_STR, SIZING_INVOL20_STR}:
            raise ValueError("Unsupported sizing_mode_str.")
        if self.variant_id_str != "B0_REF" and not self.strict_cash_cap_bool:
            raise ValueError("Every new positioning proposal must enforce the strict cash cap.")


VARIANT_SPEC_TUPLE = (
    PositioningVariantSpec(
        0,
        "B0_REF",
        SIZING_EQUAL_STR,
        False,
        False,
        "NONE",
        "Inherited equal-slot B0 reference",
    ),
    PositioningVariantSpec(
        1,
        "P0_STRICT",
        SIZING_EQUAL_STR,
        False,
        True,
        "B0_REF",
        "Equal 1/N target with strict next-open cash clipping",
    ),
    PositioningVariantSpec(
        2,
        "P1_INVOL20",
        SIZING_INVOL20_STR,
        False,
        True,
        "P0_STRICT",
        "Full-universe inverse 20-day volatility target weights",
    ),
    PositioningVariantSpec(
        3,
        "P2_SOFT200",
        SIZING_EQUAL_STR,
        True,
        True,
        "P0_STRICT",
        "Equal 1/N above SPY SMA200 and half-size otherwise",
    ),
    PositioningVariantSpec(
        4,
        "P3_INVOL20_SOFT200",
        SIZING_INVOL20_STR,
        True,
        True,
        "P0_STRICT",
        "Inverse-volatility weights multiplied by the soft SPY regime scale",
    ),
)


PAIRED_COMPARISON_TUPLE = (
    ("P0_STRICT", "B0_REF"),
    ("P1_INVOL20", "P0_STRICT"),
    ("P2_SOFT200", "P0_STRICT"),
    ("P3_INVOL20_SOFT200", "P0_STRICT"),
    ("P3_INVOL20_SOFT200", "P1_INVOL20"),
    ("P3_INVOL20_SOFT200", "P2_SOFT200"),
)


@dataclass(frozen=True)
class FourPositioningConfig:
    base_config_obj: SectorDispersionIbsConfig
    variant_spec_obj: PositioningVariantSpec
    inverse_vol_lookback_day_int: int = DEFAULT_INVOL_LOOKBACK_DAY_INT
    spy_sma_lookback_day_int: int = DEFAULT_SPY_SMA_LOOKBACK_DAY_INT
    soft_bear_scale_float: float = DEFAULT_SOFT_BEAR_SCALE_FLOAT
    spy_symbol_str: str = SPY_REGIME_SYMBOL_STR

    def __post_init__(self) -> None:
        if self.inverse_vol_lookback_day_int <= 1:
            raise ValueError("inverse_vol_lookback_day_int must be greater than one.")
        if self.spy_sma_lookback_day_int <= 1:
            raise ValueError("spy_sma_lookback_day_int must be greater than one.")
        if not 0.0 < self.soft_bear_scale_float <= 1.0:
            raise ValueError("soft_bear_scale_float must lie in (0, 1].")
        if not np.isclose(float(self.base_config_obj.portfolio_leverage_float), 1.0):
            raise ValueError("The positioning study requires portfolio_leverage_float == 1.0.")


def compute_inverse_vol_target_weight_df(
    close_price_df: pd.DataFrame,
    lookback_day_int: int = DEFAULT_INVOL_LOOKBACK_DAY_INT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return causal annualized volatility and full-universe inverse-vol weights."""
    if lookback_day_int <= 1:
        raise ValueError("lookback_day_int must be greater than one.")
    numeric_close_price_df = close_price_df.apply(pd.to_numeric, errors="coerce")
    # *** CRITICAL*** The rolling window ends at completed Close_T and sizes an
    # order filled at Open_T+1. No future return or full-sample normalization is
    # permitted here.
    close_return_df = numeric_close_price_df.pct_change(fill_method=None)
    volatility_ann_df = close_return_df.rolling(
        window=int(lookback_day_int),
        min_periods=int(lookback_day_int),
    ).std(ddof=1) * np.sqrt(252.0)
    valid_volatility_df = volatility_ann_df.where(
        np.isfinite(volatility_ann_df) & volatility_ann_df.gt(0.0)
    )
    inverse_volatility_df = 1.0 / valid_volatility_df
    full_universe_count_int = int(len(numeric_close_price_df.columns))
    inverse_volatility_sum_ser = inverse_volatility_df.sum(
        axis=1,
        min_count=full_universe_count_int,
    )
    target_weight_df = inverse_volatility_df.div(inverse_volatility_sum_ser, axis=0)
    return volatility_ann_df, target_weight_df


def _multiindex_feature_df(feature_df: pd.DataFrame, field_str: str) -> pd.DataFrame:
    output_feature_df = feature_df.copy()
    output_feature_df.columns = pd.MultiIndex.from_tuples(
        [(str(symbol_str), field_str) for symbol_str in output_feature_df.columns]
    )
    return output_feature_df


class SectorDispersionFourPositioningStrategy(SectorDispersionDividendStrategy):
    """B0 signal with one frozen entry-positioning rule."""

    def __init__(
        self,
        name: str,
        benchmarks: list[str] | tuple[str, ...],
        positioning_config_obj: FourPositioningConfig,
    ) -> None:
        super().__init__(
            name=name,
            benchmarks=benchmarks,
            config_obj=positioning_config_obj.base_config_obj,
        )
        self.positioning_config_obj = positioning_config_obj
        self.variant_spec_obj = positioning_config_obj.variant_spec_obj
        self._entry_sizing_decision_row_dict_list: list[dict[str, object]] = []
        self._cash_cap_event_row_dict_list: list[dict[str, object]] = []

    @property
    def entry_sizing_decision_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._entry_sizing_decision_row_dict_list)

    @property
    def cash_cap_event_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._cash_cap_event_row_dict_list)

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        signal_data_df = super().compute_signals(pricing_data_df)
        close_price_df = pd.DataFrame(
            {
                symbol_str: pd.to_numeric(
                    signal_data_df[(symbol_str, "Close")],
                    errors="coerce",
                )
                for symbol_str in self.symbol_tuple
            },
            index=signal_data_df.index,
            dtype=float,
        )
        volatility_ann_df, inverse_vol_target_weight_df = (
            compute_inverse_vol_target_weight_df(
                close_price_df=close_price_df,
                lookback_day_int=self.positioning_config_obj.inverse_vol_lookback_day_int,
            )
        )

        spy_close_key_tuple = (self.positioning_config_obj.spy_symbol_str, "Close")
        if spy_close_key_tuple not in signal_data_df.columns:
            raise RuntimeError(f"Missing SPY close column: {spy_close_key_tuple}")
        spy_close_ser = pd.to_numeric(signal_data_df[spy_close_key_tuple], errors="coerce")
        # *** CRITICAL*** SMA_T includes completed SPY Close_T. The multiplier
        # is frozen for the order that fills at Open_T+1 and never backfilled.
        spy_sma_ser = spy_close_ser.rolling(
            window=int(self.positioning_config_obj.spy_sma_lookback_day_int),
            min_periods=int(self.positioning_config_obj.spy_sma_lookback_day_int),
        ).mean()
        market_scale_ser = pd.Series(np.nan, index=spy_close_ser.index, dtype=float)
        valid_sma_bool_ser = spy_close_ser.notna() & spy_sma_ser.notna()
        market_scale_ser.loc[valid_sma_bool_ser] = np.where(
            spy_close_ser.loc[valid_sma_bool_ser].gt(spy_sma_ser.loc[valid_sma_bool_ser]),
            1.0,
            float(self.positioning_config_obj.soft_bear_scale_float),
        )
        regime_feature_df = pd.DataFrame(
            {
                (
                    self.positioning_config_obj.spy_symbol_str,
                    f"sma_{self.positioning_config_obj.spy_sma_lookback_day_int}_ser",
                ): spy_sma_ser,
                (
                    self.positioning_config_obj.spy_symbol_str,
                    MARKET_SCALE_FIELD_STR,
                ): market_scale_ser,
            },
            index=signal_data_df.index,
        )
        regime_feature_df.columns = pd.MultiIndex.from_tuples(regime_feature_df.columns)
        return pd.concat(
            [
                signal_data_df,
                _multiindex_feature_df(
                    volatility_ann_df,
                    INVERSE_VOLATILITY_FIELD_STR,
                ),
                _multiindex_feature_df(
                    inverse_vol_target_weight_df,
                    INVERSE_VOL_TARGET_WEIGHT_FIELD_STR,
                ),
                regime_feature_df,
            ],
            axis=1,
        )

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
            if str(symbol_str) in self.symbol_tuple and float(position_float) > 0.0
        }

        for symbol_str in self.symbol_tuple:
            exit_signal_bool = bool(
                close_row_ser.get((symbol_str, "exit_signal_bool"), False)
            )
            if symbol_str not in held_symbol_set or not exit_signal_bool:
                continue
            self.order_target(
                symbol_str,
                0.0,
                trade_id=self.current_trade_map[symbol_str],
            )
            held_symbol_set.remove(symbol_str)

        for symbol_str in self.symbol_tuple:
            if symbol_str in held_symbol_set or self.get_position(symbol_str) != 0:
                continue
            if not bool(close_row_ser.get((symbol_str, "entry_signal_bool"), False)):
                continue

            target_weight_float = self._entry_target_weight_float(
                symbol_str=symbol_str,
                close_row_ser=close_row_ser,
            )
            if not np.isfinite(target_weight_float):
                continue
            if not 0.0 < target_weight_float <= 1.0:
                raise RuntimeError(
                    f"Invalid target weight {target_weight_float} for {symbol_str}."
                )
            close_price_float = float(close_row_ser.get((symbol_str, "Close"), np.nan))
            if not np.isfinite(close_price_float) or close_price_float <= 0.0:
                raise RuntimeError(
                    f"Cannot size {symbol_str} entry without a valid decision-bar close."
                )
            target_share_float = (
                float(self.previous_total_value)
                * target_weight_float
                / close_price_float
            )
            self.trade_id_int += 1
            self.current_trade_map[symbol_str] = self.trade_id_int
            self.order_target(
                symbol_str,
                target_share_float,
                trade_id=self.trade_id_int,
            )
            self._entry_sizing_decision_row_dict_list.append(
                {
                    "decision_bar": self.previous_bar,
                    "execution_bar": self.current_bar,
                    "asset": symbol_str,
                    "variant_id_str": self.variant_spec_obj.variant_id_str,
                    "decision_close_float": close_price_float,
                    "requested_target_weight_float": target_weight_float,
                    "requested_target_share_float": target_share_float,
                    "inverse_vol_target_weight_float": float(
                        close_row_ser.get(
                            (symbol_str, INVERSE_VOL_TARGET_WEIGHT_FIELD_STR),
                            np.nan,
                        )
                    ),
                    "market_scale_float": float(
                        close_row_ser.get(
                            (
                                self.positioning_config_obj.spy_symbol_str,
                                MARKET_SCALE_FIELD_STR,
                            ),
                            np.nan,
                        )
                    ),
                }
            )

    def _entry_target_weight_float(
        self,
        symbol_str: str,
        close_row_ser: pd.Series,
    ) -> float:
        if self.variant_spec_obj.sizing_mode_str == SIZING_INVOL20_STR:
            base_target_weight_float = float(
                close_row_ser.get(
                    (symbol_str, INVERSE_VOL_TARGET_WEIGHT_FIELD_STR),
                    np.nan,
                )
            )
        else:
            base_target_weight_float = 1.0 / float(len(self.symbol_tuple))

        if not self.variant_spec_obj.uses_soft_sma_bool:
            return base_target_weight_float
        market_scale_float = float(
            close_row_ser.get(
                (
                    self.positioning_config_obj.spy_symbol_str,
                    MARKET_SCALE_FIELD_STR,
                ),
                np.nan,
            )
        )
        return base_target_weight_float * market_scale_float

    def _cash_after_open_orders_float(
        self,
        pricing_data_df: pd.DataFrame,
        buy_scale_float: float,
    ) -> float:
        projected_cash_float = float(self.cash)
        for order_obj in self.get_orders():
            if str(order_obj.unit) != "shares" or not bool(order_obj.target):
                raise RuntimeError("Strict cap supports target-share orders only.")
            open_value_obj = pricing_data_df.loc[
                self.current_bar,
                (str(order_obj.asset), "Open"),
            ]
            if pd.isna(open_value_obj):
                continue
            current_position_float = float(self.get_position(order_obj.asset))
            desired_delta_share_float = float(order_obj.amount) - current_position_float
            execution_delta_share_float = (
                desired_delta_share_float * float(buy_scale_float)
                if desired_delta_share_float > 0.0
                else desired_delta_share_float
            )
            if np.isclose(execution_delta_share_float, 0.0):
                continue
            open_price_float = float(open_value_obj)
            execution_price_float = open_price_float * (
                1.0 + np.sign(execution_delta_share_float) * float(self._slippage)
            )
            commission_float = float(
                self._compute_commission(execution_delta_share_float)
            )
            projected_cash_float -= execution_delta_share_float * execution_price_float
            projected_cash_float -= commission_float
        return projected_cash_float

    def _apply_strict_cash_cap(self, pricing_data_df: pd.DataFrame) -> None:
        active_buy_order_count_int = 0
        for order_obj in self.get_orders():
            current_position_float = float(self.get_position(order_obj.asset))
            desired_delta_share_float = float(order_obj.amount) - current_position_float
            if desired_delta_share_float <= 0.0:
                continue
            open_value_obj = pricing_data_df.loc[
                self.current_bar,
                (str(order_obj.asset), "Open"),
            ]
            if pd.isna(open_value_obj):
                continue
            active_buy_order_count_int += 1
        if active_buy_order_count_int == 0:
            return

        desired_cash_float = self._cash_after_open_orders_float(
            pricing_data_df=pricing_data_df,
            buy_scale_float=1.0,
        )
        if desired_cash_float >= 0.0:
            return
        zero_buy_cash_float = self._cash_after_open_orders_float(
            pricing_data_df=pricing_data_df,
            buy_scale_float=0.0,
        )
        if zero_buy_cash_float < -STRICT_CASH_TOLERANCE_FLOAT:
            raise RuntimeError(
                "Strict positioning row entered the open with infeasible cash before buys: "
                f"{zero_buy_cash_float:.6f}."
            )

        feasible_scale_float = 0.0
        infeasible_scale_float = 1.0
        for _ in range(80):
            candidate_scale_float = (
                feasible_scale_float + infeasible_scale_float
            ) / 2.0
            candidate_cash_float = self._cash_after_open_orders_float(
                pricing_data_df=pricing_data_df,
                buy_scale_float=candidate_scale_float,
            )
            if candidate_cash_float >= 0.0:
                feasible_scale_float = candidate_scale_float
            else:
                infeasible_scale_float = candidate_scale_float

        clipped_order_count_int = 0
        for order_obj in self.get_orders():
            current_position_float = float(self.get_position(order_obj.asset))
            desired_delta_share_float = float(order_obj.amount) - current_position_float
            if desired_delta_share_float <= 0.0:
                continue
            order_obj.amount = current_position_float + (
                desired_delta_share_float * feasible_scale_float
            )
            clipped_order_count_int += 1

        projected_final_cash_float = self._cash_after_open_orders_float(
            pricing_data_df=pricing_data_df,
            buy_scale_float=1.0,
        )
        self._cash_cap_event_row_dict_list.append(
            {
                "bar": pd.Timestamp(self.current_bar),
                "variant_id_str": self.variant_spec_obj.variant_id_str,
                "cash_before_orders_float": float(self.cash),
                "cash_without_cap_float": desired_cash_float,
                "cash_after_exits_before_buys_float": zero_buy_cash_float,
                "cash_cap_scale_float": feasible_scale_float,
                "projected_final_cash_float": projected_final_cash_float,
                "clipped_order_count_int": clipped_order_count_int,
            }
        )

    def process_orders(self, prices: pd.DataFrame):
        if not self.variant_spec_obj.strict_cash_cap_bool:
            return super().process_orders(prices)

        self._cancel_current_bar_stale_orders(pricing_data_df=prices)
        self._credit_current_bar_dividend_cash(pricing_data_df=prices)
        # *** CRITICAL*** Target shares were chosen from completed Close_T. The
        # cap is applied only now, using tradable Open_T+1 and modeled costs, so
        # it cannot improve the signal with future information.
        self._apply_strict_cash_cap(pricing_data_df=prices)
        result_obj = SectorDispersionIbsStrategy.process_orders(self, prices)
        if float(self.cash) < -STRICT_CASH_TOLERANCE_FLOAT:
            raise RuntimeError(
                "Strict cash cap failed after execution: "
                f"bar={self.current_bar}, cash={float(self.cash):.6f}."
            )
        return result_obj


def build_variant_manifest_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "priority_int": spec_obj.priority_int,
                "variant_id_str": spec_obj.variant_id_str,
                "sizing_mode_str": spec_obj.sizing_mode_str,
                "uses_soft_sma_bool": spec_obj.uses_soft_sma_bool,
                "strict_cash_cap_bool": spec_obj.strict_cash_cap_bool,
                "primary_control_variant_id_str": spec_obj.primary_control_variant_id_str,
                "description_str": spec_obj.description_str,
            }
            for spec_obj in VARIANT_SPEC_TUPLE
        ]
    )


def _entry_metric_dict(
    strategy_obj: SectorDispersionFourPositioningStrategy,
) -> dict[str, float | int]:
    entry_df = strategy_obj.entry_sizing_decision_df
    if entry_df.empty:
        return {
            "entry_decision_count_int": 0,
            "average_entry_target_weight_pct_float": np.nan,
            "maximum_entry_target_weight_pct_float": np.nan,
            "cash_cap_event_count_int": int(len(strategy_obj.cash_cap_event_df)),
            "minimum_cash_cap_scale_float": np.nan,
        }
    target_weight_ser = pd.to_numeric(
        entry_df["requested_target_weight_float"],
        errors="coerce",
    )
    cash_cap_scale_ser = pd.to_numeric(
        strategy_obj.cash_cap_event_df.get("cash_cap_scale_float", pd.Series(dtype=float)),
        errors="coerce",
    )
    return {
        "entry_decision_count_int": int(len(entry_df)),
        "average_entry_target_weight_pct_float": float(target_weight_ser.mean() * 100.0),
        "maximum_entry_target_weight_pct_float": float(target_weight_ser.max() * 100.0),
        "cash_cap_event_count_int": int(len(strategy_obj.cash_cap_event_df)),
        "minimum_cash_cap_scale_float": (
            np.nan if cash_cap_scale_ser.empty else float(cash_cap_scale_ser.min())
        ),
    }


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
        axis_obj.legend(ncol=3, frameon=False)
    figure_obj.suptitle("Sector Dispersion Four Positioning Rules: Common-Overlap Equity")
    figure_obj.savefig(output_path / "common_overlap_equity_curves.png", dpi=180)
    plt.close(figure_obj)


def _exposure_matched_p0_metric_dict(
    strategy_return_ser: pd.Series,
    variant_average_gross_pct_float: float,
    p0_return_ser: pd.Series,
    p0_average_gross_pct_float: float,
    capital_base_float: float,
) -> dict[str, object]:
    if p0_average_gross_pct_float <= 0.0:
        return compute_equity_metric_dict(pd.Series(dtype=float), "exposure_matched_p0")
    exposure_ratio_float = variant_average_gross_pct_float / p0_average_gross_pct_float
    aligned_p0_return_ser = pd.to_numeric(p0_return_ser, errors="coerce").reindex(
        strategy_return_ser.index
    ).fillna(0.0)
    # *** CRITICAL*** This uses an ex-post full-sample average-exposure ratio
    # for diagnosis only. It never enters orders or claims tradability.
    exposure_matched_equity_ser = (
        1.0 + aligned_p0_return_ser * exposure_ratio_float
    ).cumprod() * float(capital_base_float)
    metric_dict = compute_equity_metric_dict(
        exposure_matched_equity_ser,
        prefix_str="exposure_matched_p0",
    )
    metric_dict["exposure_matched_p0_scale_float"] = exposure_ratio_float
    return metric_dict


def run_four_positioning_study(
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

    timestamp_str = pd.Timestamp.now().strftime("%Y-%m-%d_%H%M%S")
    output_path = build_research_output_path(
        output_dir=output_dir_str,
        entity_type_str="strategy",
        entity_id_str="strategy_mr_sector_dispersion_ibs",
        analysis_type_str="four_positioning_study",
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

    variant_summary_row_dict_list: list[dict[str, object]] = []
    paired_comparison_row_dict_list: list[dict[str, object]] = []
    annual_return_row_dict_list: list[dict[str, object]] = []
    subperiod_row_dict_list: list[dict[str, object]] = []
    common_equity_df = pd.DataFrame()

    for universe_index_int, universe_id_str in enumerate(STUDY_UNIVERSE_ID_TUPLE, start=1):
        config_obj = config_by_universe_dict[universe_id_str]
        calendar_idx = calendar_by_universe_dict[universe_id_str]
        equal_weight_equity_ser = equal_weight_equity_by_universe_dict[universe_id_str]
        return_by_variant_dict: dict[str, pd.Series] = {}
        equity_by_variant_dict: dict[str, pd.Series] = {}
        metric_by_variant_dict: dict[str, dict[str, object]] = {}
        average_gross_by_variant_dict: dict[str, float] = {}
        raw_summary_by_variant_dict: dict[str, dict[str, object]] = {}

        for variant_index_int, variant_spec_obj in enumerate(VARIANT_SPEC_TUPLE, start=1):
            print(
                f"Running universe {universe_index_int}/3 {universe_id_str}, "
                f"variant {variant_index_int}/5 {variant_spec_obj.variant_id_str}...",
                flush=True,
            )
            positioning_config_obj = FourPositioningConfig(
                base_config_obj=config_obj,
                variant_spec_obj=variant_spec_obj,
            )
            strategy_obj = SectorDispersionFourPositioningStrategy(
                name=(
                    "strategy_mr_sector_dispersion_ibs_"
                    f"{universe_id_str}_{variant_spec_obj.variant_id_str.lower()}"
                ),
                benchmarks=[COMMON_BENCHMARK_SYMBOL_STR],
                positioning_config_obj=positioning_config_obj,
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
            variant_output_path = (
                output_path / "universes" / universe_id_str / variant_spec_obj.variant_id_str
            )
            strategy_obj.entry_sizing_decision_df.to_csv(
                variant_output_path / "entry_sizing_decisions.csv",
                index=False,
            )
            strategy_obj.cash_cap_event_df.to_csv(
                variant_output_path / "cash_cap_events.csv",
                index=False,
            )

            strategy_equity_ser = pd.to_numeric(
                strategy_obj.results["total_value"],
                errors="coerce",
            ).astype(float)
            strategy_return_ser = pd.to_numeric(
                strategy_obj.results["daily_returns"],
                errors="coerce",
            ).astype(float)
            full_metric_dict = compute_equity_metric_dict(
                strategy_equity_ser,
                prefix_str="full_strategy",
            )
            common_metric_dict = compute_equity_metric_dict(
                strategy_equity_ser.loc[
                    strategy_equity_ser.index >= common_overlap_start_ts
                ],
                prefix_str="common_strategy",
            )
            exposure_metric_dict = build_exposure_diagnostic_dict(
                realized_weight_df=strategy_obj.realized_weight_df,
                result_df=strategy_obj.results,
                symbol_tuple=strategy_obj.symbol_tuple,
            )
            if variant_spec_obj.strict_cash_cap_bool:
                if int(exposure_metric_dict["negative_cash_day_count_int"]) != 0:
                    raise RuntimeError(
                        f"{universe_id_str}/{variant_spec_obj.variant_id_str} has negative cash."
                    )
                if float(exposure_metric_dict["max_gross_exposure_pct_float"]) > 100.0001:
                    raise RuntimeError(
                        f"{universe_id_str}/{variant_spec_obj.variant_id_str} exceeds 100% gross."
                    )

            raw_summary_dict = {
                "universe_id_str": universe_id_str,
                "variant_priority_int": variant_spec_obj.priority_int,
                "variant_id_str": variant_spec_obj.variant_id_str,
                "sizing_mode_str": variant_spec_obj.sizing_mode_str,
                "uses_soft_sma_bool": variant_spec_obj.uses_soft_sma_bool,
                "strict_cash_cap_bool": variant_spec_obj.strict_cash_cap_bool,
                "primary_control_variant_id_str": (
                    variant_spec_obj.primary_control_variant_id_str
                ),
                "symbol_count_int": len(strategy_obj.symbol_tuple),
                "turnover_ann_pct_float": _summary_metric_float(
                    strategy_obj.summary,
                    "Turnover (Ann.) [%]",
                ),
                "total_modeled_cost_drag_ann_pct_float": _summary_metric_float(
                    strategy_obj.summary,
                    "Cost Drag (Ann.) [%]",
                ),
                "dividend_cash_total_float": strategy_obj.dividend_cash_total_float,
                **_market_metric_dict(strategy_return_ser, market_return_ser),
                **exposure_metric_dict,
                **_net_exposure_metric_dict(strategy_obj),
                **_entry_metric_dict(strategy_obj),
                **_trade_metric_dict(strategy_obj),
                **full_metric_dict,
                **common_metric_dict,
                **compute_equity_metric_dict(
                    equal_weight_equity_ser,
                    prefix_str="full_equal_weight",
                ),
            }
            raw_summary_by_variant_dict[variant_spec_obj.variant_id_str] = raw_summary_dict
            return_by_variant_dict[variant_spec_obj.variant_id_str] = strategy_return_ser
            equity_by_variant_dict[variant_spec_obj.variant_id_str] = strategy_equity_ser
            metric_by_variant_dict[variant_spec_obj.variant_id_str] = full_metric_dict
            average_gross_by_variant_dict[variant_spec_obj.variant_id_str] = float(
                exposure_metric_dict["average_gross_exposure_pct_float"]
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
                    strategy_total_value_ser=strategy_equity_ser,
                )
            )
            common_equity_ser = strategy_equity_ser.loc[
                strategy_equity_ser.index >= common_overlap_start_ts
            ]
            common_equity_df[
                f"{universe_id_str}_{variant_spec_obj.variant_id_str}"
            ] = common_equity_ser / common_equity_ser.iloc[0]

        p0_return_ser = return_by_variant_dict["P0_STRICT"]
        p0_average_gross_pct_float = average_gross_by_variant_dict["P0_STRICT"]
        for variant_spec_obj in VARIANT_SPEC_TUPLE:
            variant_id_str = variant_spec_obj.variant_id_str
            summary_row_dict = raw_summary_by_variant_dict[variant_id_str]
            control_variant_id_str = variant_spec_obj.primary_control_variant_id_str
            if control_variant_id_str == "NONE":
                summary_row_dict.update(
                    {
                        "delta_ann_return_vs_control_pct_float": np.nan,
                        "delta_sharpe_vs_control_float": np.nan,
                        "delta_max_drawdown_vs_control_pct_float": np.nan,
                        "primary_bonferroni_p_value_float": np.nan,
                    }
                )
            else:
                variant_metric_dict = metric_by_variant_dict[variant_id_str]
                control_metric_dict = metric_by_variant_dict[control_variant_id_str]
                paired_return_df = pd.concat(
                    [
                        return_by_variant_dict[variant_id_str].rename("variant"),
                        return_by_variant_dict[control_variant_id_str].rename("control"),
                    ],
                    axis=1,
                    join="inner",
                ).dropna()
                hac_metric_dict = compute_hac_mean_test_dict(
                    paired_return_df["variant"] - paired_return_df["control"],
                    family_test_count_int=FAMILY_TEST_COUNT_INT,
                )
                summary_row_dict.update(
                    {
                        "delta_ann_return_vs_control_pct_float": float(
                            variant_metric_dict["full_strategy_ann_return_pct_float"]
                        )
                        - float(control_metric_dict["full_strategy_ann_return_pct_float"]),
                        "delta_sharpe_vs_control_float": float(
                            variant_metric_dict["full_strategy_sharpe_float"]
                        )
                        - float(control_metric_dict["full_strategy_sharpe_float"]),
                        "delta_max_drawdown_vs_control_pct_float": float(
                            variant_metric_dict["full_strategy_max_drawdown_pct_float"]
                        )
                        - float(control_metric_dict["full_strategy_max_drawdown_pct_float"]),
                        "primary_bonferroni_p_value_float": hac_metric_dict[
                            "bonferroni_p_value_float"
                        ],
                    }
                )
            summary_row_dict.update(
                _exposure_matched_p0_metric_dict(
                    strategy_return_ser=return_by_variant_dict[variant_id_str],
                    variant_average_gross_pct_float=average_gross_by_variant_dict[
                        variant_id_str
                    ],
                    p0_return_ser=p0_return_ser,
                    p0_average_gross_pct_float=p0_average_gross_pct_float,
                    capital_base_float=config_obj.capital_base_float,
                )
            )
            variant_summary_row_dict_list.append(summary_row_dict)

        for variant_id_str, control_variant_id_str in PAIRED_COMPARISON_TUPLE:
            variant_metric_dict = metric_by_variant_dict[variant_id_str]
            control_metric_dict = metric_by_variant_dict[control_variant_id_str]
            paired_return_df = pd.concat(
                [
                    return_by_variant_dict[variant_id_str].rename("variant"),
                    return_by_variant_dict[control_variant_id_str].rename("control"),
                ],
                axis=1,
                join="inner",
            ).dropna()
            paired_comparison_row_dict_list.append(
                {
                    "universe_id_str": universe_id_str,
                    "variant_id_str": variant_id_str,
                    "control_variant_id_str": control_variant_id_str,
                    "delta_ann_return_pct_float": float(
                        variant_metric_dict["full_strategy_ann_return_pct_float"]
                    )
                    - float(control_metric_dict["full_strategy_ann_return_pct_float"]),
                    "delta_sharpe_float": float(
                        variant_metric_dict["full_strategy_sharpe_float"]
                    )
                    - float(control_metric_dict["full_strategy_sharpe_float"]),
                    "delta_max_drawdown_pct_float": float(
                        variant_metric_dict["full_strategy_max_drawdown_pct_float"]
                    )
                    - float(control_metric_dict["full_strategy_max_drawdown_pct_float"]),
                    **compute_hac_mean_test_dict(
                        paired_return_df["variant"] - paired_return_df["control"],
                        family_test_count_int=FAMILY_TEST_COUNT_INT,
                    ),
                }
            )

    variant_summary_df = pd.DataFrame(variant_summary_row_dict_list).sort_values(
        ["universe_id_str", "variant_priority_int"]
    )
    paired_comparison_df = pd.DataFrame(paired_comparison_row_dict_list)
    annual_return_df = pd.DataFrame(annual_return_row_dict_list)
    subperiod_df = pd.DataFrame(subperiod_row_dict_list)
    variant_summary_df.to_csv(output_path / "variant_summary.csv", index=False)
    paired_comparison_df.to_csv(output_path / "paired_comparisons.csv", index=False)
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
        "max_gross_exposure_pct_float",
        "minimum_cash_float",
        "total_modeled_cost_drag_ann_pct_float",
        "delta_ann_return_vs_control_pct_float",
    ]
    summary_md_str = f"""# Sector Dispersion Four-Positioning Study

- Research only; no live/release wiring.
- Four frozen positioning proposals plus one inherited B0 reference.
- Search: 15 strategy rows and 18 frozen paired comparisons.
- Signal: completed daily bar T; execution: Open T+1.
- Inverse volatility: full-universe 20-day close-to-close volatility.
- Soft regime: 1.0 above SPY SMA200, otherwise 0.5, fixed at entry.
- P0-P3 enforce non-negative post-fill cash and gross exposure at or below 100%.
- Common overlap: {common_overlap_start_ts.date().isoformat()} to {end_date_str}.

## Mechanical Summary

{_markdown_table_str(variant_summary_df, summary_column_list)}

This table is not the verdict. The final decision must apply the preregistered
cross-universe, common-period, subperiod, exposure, concentration, and
multiple-comparison rules.
"""
    (output_path / "study_summary.md").write_text(summary_md_str, encoding="utf-8")
    metadata_dict = {
        "analysis_type_str": "four_positioning_study",
        "research_only_bool": True,
        "output_path_str": str(output_path.resolve()),
        "study_end_date_str": end_date_str,
        "common_overlap_start_date_str": common_overlap_start_ts.date().isoformat(),
        "universe_id_tuple": STUDY_UNIVERSE_ID_TUPLE,
        "variant_id_tuple": tuple(
            spec_obj.variant_id_str for spec_obj in VARIANT_SPEC_TUPLE
        ),
        "strategy_row_count_int": len(STUDY_UNIVERSE_ID_TUPLE)
        * len(VARIANT_SPEC_TUPLE),
        "new_proposal_row_count_int": 12,
        "paired_comparison_count_int": FAMILY_TEST_COUNT_INT,
        "execution_mapping_str": "completed daily signal T -> Open T+1",
        "strict_cash_cap_bool": True,
        "inverse_vol_lookback_day_int": DEFAULT_INVOL_LOOKBACK_DAY_INT,
        "spy_sma_lookback_day_int": DEFAULT_SPY_SMA_LOOKBACK_DAY_INT,
        "soft_bear_scale_float": DEFAULT_SOFT_BEAR_SCALE_FLOAT,
        "spy_adjustment_str": "CAPITALSPECIAL",
        "market_benchmark_adjustment_str": "TOTALRETURN",
        "cash_yield_float": 0.0,
        "slippage_float": DEFAULT_CONFIG.slippage_float,
        "commission_per_share_float": DEFAULT_CONFIG.commission_per_share_float,
        "commission_minimum_float": DEFAULT_CONFIG.commission_minimum_float,
        "preregistration_path_str": str(
            (
                REPO_ROOT_PATH
                / "docs"
                / "research"
                / "SECTOR_DISPERSION_FOUR_POSITIONING_PREREGISTRATION.md"
            ).resolve()
        ),
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata_dict, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Saved four-positioning study to {output_path.resolve()}", flush=True)
    return output_path


def parse_args(argv_list: list[str] | None = None) -> argparse.Namespace:
    parser_obj = argparse.ArgumentParser(
        description="Run the preregistered four-positioning sector IBS study."
    )
    parser_obj.add_argument("--output-dir", default="results")
    parser_obj.add_argument("--end-date", default=STUDY_END_DATE_STR)
    parser_obj.add_argument("--show-progress", action="store_true")
    return parser_obj.parse_args(argv_list)


def main(argv_list: list[str] | None = None) -> None:
    args_obj = parse_args(argv_list)
    run_four_positioning_study(
        output_dir_str=str(args_obj.output_dir),
        end_date_str=str(args_obj.end_date),
        show_progress_bool=bool(args_obj.show_progress),
    )


if __name__ == "__main__":
    main()
