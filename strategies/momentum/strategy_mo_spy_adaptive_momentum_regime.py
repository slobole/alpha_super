"""Research-only SPY Adaptive Momentum market-regime strategy.

The source-inspired signal is formed after the final close of session T:

    drawdown_T = P_T / max(P_0, ..., P_T) - 1
    severity_T = -drawdown_T
    q_T = trailing_126_session_strict_percentile(severity_T)
    Q_T = q_T ** 2
    alpha_fast = 2 / (50 + 1)
    alpha_slow = 2 / (200 + 1)
    alpha_T = Q_T * alpha_fast + (1 - Q_T) * alpha_slow
    AMA_T = alpha_T * P_T + (1 - alpha_T) * AMA_(T-1)
    filtered_price_T = SMA_10(P)_T
    target_weight_T = 1[filtered_price_T > AMA_T]

The signal price is SPY total return adjusted, matching the source's use of
``SPY.Adjusted``. Execution and accounting use the repository's normal
CAPITALSPECIAL SPY series with its separate dividend ledger. The decision at
Close_T can first change the position at Open_(T+1).

This module contains no LIVE, release, scheduler, broker, allocation, or
strategy-registry wiring.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from data.norgate_loader import load_raw_prices


SIGNAL_NAMESPACE_STR = "ADAPTIVE_MOMENTUM_REGIME"


def default_trade_id_int() -> int:
    return -1


@dataclass(frozen=True)
class SpyAdaptiveMomentumRegimeConfig:
    trade_symbol_str: str = "SPY"
    signal_symbol_str: str = "SPY_TR_SIGNAL"
    benchmark_symbol_str: str = "$SPX"
    percentile_lookback_int: int = 126
    high_lookback_int: int | None = None
    percentile_method_str: str = "strict"
    fast_lookback_int: int = 50
    slow_lookback_int: int = 200
    percentile_power_float: float = 2.0
    price_filter_lookback_int: int = 10
    history_start_date_str: str = "1993-01-01"
    backtest_start_date_str: str = "1995-01-03"
    end_date_str: str | None = None
    capital_base_float: float = 100_000.0
    slippage_float: float = 0.00025
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0

    def __post_init__(self) -> None:
        if not self.trade_symbol_str:
            raise ValueError("trade_symbol_str must not be empty.")
        if not self.signal_symbol_str:
            raise ValueError("signal_symbol_str must not be empty.")
        if self.trade_symbol_str == self.signal_symbol_str:
            raise ValueError("signal_symbol_str must differ from trade_symbol_str.")
        if not self.benchmark_symbol_str:
            raise ValueError("benchmark_symbol_str must not be empty.")
        if self.percentile_lookback_int <= 1:
            raise ValueError("percentile_lookback_int must be greater than one.")
        if self.high_lookback_int is not None and self.high_lookback_int <= 1:
            raise ValueError("high_lookback_int must be greater than one when set.")
        if self.percentile_method_str not in {"strict", "weak"}:
            raise ValueError("percentile_method_str must be 'strict' or 'weak'.")
        if self.fast_lookback_int <= 1:
            raise ValueError("fast_lookback_int must be greater than one.")
        if self.slow_lookback_int <= self.fast_lookback_int:
            raise ValueError("slow_lookback_int must exceed fast_lookback_int.")
        if not np.isfinite(self.percentile_power_float) or self.percentile_power_float <= 0.0:
            raise ValueError("percentile_power_float must be positive.")
        if self.price_filter_lookback_int <= 1:
            raise ValueError("price_filter_lookback_int must be greater than one.")
        if pd.Timestamp(self.history_start_date_str) >= pd.Timestamp(
            self.backtest_start_date_str
        ):
            raise ValueError(
                "history_start_date_str must be earlier than backtest_start_date_str."
            )
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if self.slippage_float < 0.0:
            raise ValueError("slippage_float must be non-negative.")
        if self.commission_per_share_float < 0.0:
            raise ValueError("commission_per_share_float must be non-negative.")
        if self.commission_minimum_float < 0.0:
            raise ValueError("commission_minimum_float must be non-negative.")


DEFAULT_CONFIG = SpyAdaptiveMomentumRegimeConfig()


def build_adaptive_momentum_calendar_idx(
    pricing_data_df: pd.DataFrame,
    config_obj: SpyAdaptiveMomentumRegimeConfig,
) -> pd.DatetimeIndex:
    """Return executable sessions with both signal and trade prices present."""

    # *** CRITICAL*** availability boundary: Close_T may create an order only
    # when the traded asset has finite Open/Close data and its own total-return
    # Close_T signal is finite. This prevents pre-inception execution rows.
    common_data_mask_ser = (
        np.isfinite(
            pricing_data_df[(config_obj.trade_symbol_str, "Open")].astype(float)
        )
        & np.isfinite(
            pricing_data_df[(config_obj.trade_symbol_str, "Close")].astype(float)
        )
        & np.isfinite(
            pricing_data_df[(config_obj.signal_symbol_str, "Close")].astype(float)
        )
    )
    calendar_idx = pricing_data_df.index[
        common_data_mask_ser
        & (
            pricing_data_df.index
            >= pd.Timestamp(config_obj.backtest_start_date_str)
        )
    ]
    return pd.DatetimeIndex(calendar_idx)


def compute_strict_trailing_percentile_ser(
    severity_ser: pd.Series,
    lookback_int: int,
) -> pd.Series:
    """Return the causal strict ECDF percentile of each current observation.

    The current value is compared with the complete trailing window ending at
    the current close. The denominator excludes the current observation, so a
    new maximum severity maps to 1.0. Strict comparison gives repeated zero
    drawdowns at an all-time high a percentile of 0.0.
    """

    if lookback_int <= 1:
        raise ValueError("lookback_int must be greater than one.")

    severity_ser = pd.Series(severity_ser, copy=True).astype(float)
    percentile_ser = pd.Series(np.nan, index=severity_ser.index, dtype=float)
    severity_vec = severity_ser.to_numpy(dtype=float)

    # *** CRITICAL*** lookahead-sensitive rolling boundary: each percentile
    # uses exactly the trailing lookback_int observations ending at Close_T.
    # No later severity observation may enter this window.
    for bar_idx_int in range(lookback_int - 1, len(severity_vec)):
        window_start_idx_int = bar_idx_int - lookback_int + 1
        trailing_severity_vec = severity_vec[
            window_start_idx_int : bar_idx_int + 1
        ]
        if not np.isfinite(trailing_severity_vec).all():
            continue
        current_severity_float = float(trailing_severity_vec[-1])
        strictly_lower_count_int = int(
            np.count_nonzero(
                trailing_severity_vec[:-1] < current_severity_float
            )
        )
        percentile_ser.iloc[bar_idx_int] = (
            strictly_lower_count_int / float(lookback_int - 1)
        )

    return percentile_ser


def compute_weak_trailing_percentile_ser(
    severity_ser: pd.Series,
    lookback_int: int,
) -> pd.Series:
    """Return the inclusive ECDF percentile used for a source ambiguity test."""

    if lookback_int <= 1:
        raise ValueError("lookback_int must be greater than one.")
    severity_ser = pd.Series(severity_ser, copy=True).astype(float)
    percentile_ser = pd.Series(np.nan, index=severity_ser.index, dtype=float)
    severity_vec = severity_ser.to_numpy(dtype=float)

    # *** CRITICAL*** lookahead-sensitive rolling boundary: each weak ECDF
    # uses only the trailing window ending at the current Close_T.
    for bar_idx_int in range(lookback_int - 1, len(severity_vec)):
        window_start_idx_int = bar_idx_int - lookback_int + 1
        trailing_severity_vec = severity_vec[
            window_start_idx_int : bar_idx_int + 1
        ]
        if not np.isfinite(trailing_severity_vec).all():
            continue
        current_severity_float = float(trailing_severity_vec[-1])
        percentile_ser.iloc[bar_idx_int] = float(
            np.count_nonzero(
                trailing_severity_vec <= current_severity_float
            )
        ) / float(lookback_int)
    return percentile_ser


def compute_spy_adaptive_momentum_signal_df(
    signal_price_close_ser: pd.Series,
    percentile_lookback_int: int = 126,
    high_lookback_int: int | None = None,
    percentile_method_str: str = "strict",
    fast_lookback_int: int = 50,
    slow_lookback_int: int = 200,
    percentile_power_float: float = 2.0,
    price_filter_lookback_int: int = 10,
) -> pd.DataFrame:
    """Compute the source-inspired binary SPY market-regime signal."""

    if percentile_lookback_int <= 1:
        raise ValueError("percentile_lookback_int must be greater than one.")
    if high_lookback_int is not None and high_lookback_int <= 1:
        raise ValueError("high_lookback_int must be greater than one when set.")
    if percentile_method_str not in {"strict", "weak"}:
        raise ValueError("percentile_method_str must be 'strict' or 'weak'.")
    if fast_lookback_int <= 1:
        raise ValueError("fast_lookback_int must be greater than one.")
    if slow_lookback_int <= fast_lookback_int:
        raise ValueError("slow_lookback_int must exceed fast_lookback_int.")
    if not np.isfinite(percentile_power_float) or percentile_power_float <= 0.0:
        raise ValueError("percentile_power_float must be positive.")
    if price_filter_lookback_int <= 1:
        raise ValueError("price_filter_lookback_int must be greater than one.")

    signal_price_close_ser = pd.Series(
        signal_price_close_ser,
        copy=True,
    ).astype(float)
    if signal_price_close_ser.index.has_duplicates:
        raise ValueError("signal_price_close_ser index must be unique.")
    if not signal_price_close_ser.index.is_monotonic_increasing:
        raise ValueError("signal_price_close_ser index must be increasing.")
    observed_price_ser = signal_price_close_ser.dropna()
    if (
        not np.isfinite(observed_price_ser.to_numpy(dtype=float)).all()
        or observed_price_ser.le(0.0).any()
    ):
        raise ValueError("observed signal prices must be finite and positive.")

    if high_lookback_int is None:
        # *** CRITICAL*** expanding boundary: the all-time high at Close_T uses
        # only signal closes observed on or before T.
        reference_high_ser = signal_price_close_ser.cummax()
    else:
        # *** CRITICAL*** rolling boundary: the alternative reference high at
        # Close_T contains only the trailing high_lookback_int closes through T.
        reference_high_ser = signal_price_close_ser.rolling(
            window=high_lookback_int,
            min_periods=high_lookback_int,
        ).max()
    drawdown_ser = signal_price_close_ser.divide(reference_high_ser).sub(1.0)
    drawdown_severity_ser = drawdown_ser.mul(-1.0)
    percentile_function_obj = (
        compute_strict_trailing_percentile_ser
        if percentile_method_str == "strict"
        else compute_weak_trailing_percentile_ser
    )
    drawdown_percentile_ser = percentile_function_obj(
        severity_ser=drawdown_severity_ser,
        lookback_int=percentile_lookback_int,
    )
    adaptive_weight_ser = drawdown_percentile_ser.pow(
        percentile_power_float
    )

    fast_alpha_float = 2.0 / float(fast_lookback_int + 1)
    slow_alpha_float = 2.0 / float(slow_lookback_int + 1)
    adaptive_alpha_ser = (
        adaptive_weight_ser.mul(fast_alpha_float)
        .add((1.0 - adaptive_weight_ser).mul(slow_alpha_float))
    )
    # Before the percentile window is complete, use the slow source endpoint.
    adaptive_alpha_ser = adaptive_alpha_ser.fillna(slow_alpha_float)

    adaptive_moving_average_ser = pd.Series(
        np.nan,
        index=signal_price_close_ser.index,
        dtype=float,
    )
    prior_adaptive_average_float = np.nan
    for bar_idx_int, bar_ts in enumerate(signal_price_close_ser.index):
        price_float = float(signal_price_close_ser.iloc[bar_idx_int])
        if not np.isfinite(price_float):
            continue
        alpha_float = float(adaptive_alpha_ser.iloc[bar_idx_int])
        # *** CRITICAL*** recursive boundary: AMA_T combines only Close_T,
        # alpha_T, and AMA_(T-1). It never uses a future close.
        if np.isfinite(prior_adaptive_average_float):
            adaptive_average_float = (
                alpha_float * price_float
                + (1.0 - alpha_float) * prior_adaptive_average_float
            )
        else:
            adaptive_average_float = price_float
        adaptive_moving_average_ser.loc[bar_ts] = adaptive_average_float
        prior_adaptive_average_float = adaptive_average_float

    # *** CRITICAL*** rolling boundary: filtered_price_T is the inclusive mean
    # of the last price_filter_lookback_int closes ending at Close_T.
    filtered_price_ser = signal_price_close_ser.rolling(
        window=price_filter_lookback_int,
        min_periods=price_filter_lookback_int,
    ).mean()
    valid_signal_ser = filtered_price_ser.notna() & adaptive_moving_average_ser.notna()
    target_weight_ser = pd.Series(
        np.nan,
        index=signal_price_close_ser.index,
        dtype=float,
    )
    target_weight_ser.loc[valid_signal_ser] = (
        filtered_price_ser.loc[valid_signal_ser]
        .gt(adaptive_moving_average_ser.loc[valid_signal_ser])
        .astype(float)
    )
    turnover_ser = target_weight_ser.diff().abs()

    return pd.DataFrame(
        {
            "signal_price_close_ser": signal_price_close_ser,
            "reference_high_ser": reference_high_ser,
            "drawdown_ser": drawdown_ser,
            "drawdown_severity_ser": drawdown_severity_ser,
            "drawdown_percentile_ser": drawdown_percentile_ser,
            "adaptive_weight_ser": adaptive_weight_ser,
            "adaptive_alpha_ser": adaptive_alpha_ser,
            "adaptive_moving_average_ser": adaptive_moving_average_ser,
            "filtered_price_ser": filtered_price_ser,
            "target_weight_ser": target_weight_ser,
            "turnover_ser": turnover_ser,
        },
        index=signal_price_close_ser.index,
    )


def get_spy_adaptive_momentum_regime_data(
    config: SpyAdaptiveMomentumRegimeConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Load execution, benchmark, and total-return signal series."""

    execution_price_df = load_raw_prices(
        symbols=[config.trade_symbol_str],
        benchmarks=[config.benchmark_symbol_str],
        start_date=config.history_start_date_str,
        end_date=config.end_date_str,
    )
    total_return_signal_df = load_raw_prices(
        symbols=[],
        benchmarks=[config.trade_symbol_str],
        start_date=config.history_start_date_str,
        end_date=config.end_date_str,
    )
    signal_price_df = total_return_signal_df.loc[
        :,
        [
            (config.trade_symbol_str, "Open"),
            (config.trade_symbol_str, "Close"),
        ],
    ].copy()
    signal_price_df.columns = pd.MultiIndex.from_tuples(
        [
            (config.signal_symbol_str, field_str)
            for _, field_str in signal_price_df.columns
        ]
    )
    pricing_data_df = pd.concat(
        [execution_price_df, signal_price_df],
        axis=1,
    ).sort_index()
    pricing_data_df.attrs.update(execution_price_df.attrs)
    pricing_data_df.attrs["signal_adjustment_by_symbol_dict"] = {
        config.signal_symbol_str: "TOTALRETURN"
    }
    return pricing_data_df


class SpyAdaptiveMomentumRegimeStrategy(Strategy):
    """Long/flat SPY strategy controlled by the Adaptive Momentum regime."""

    enable_signal_audit = True
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: list[str] | tuple[str, ...],
        config: SpyAdaptiveMomentumRegimeConfig = DEFAULT_CONFIG,
    ) -> None:
        super().__init__(
            name=name,
            benchmarks=list(benchmarks),
            capital_base=config.capital_base_float,
            slippage=config.slippage_float,
            commission_per_share=config.commission_per_share_float,
            commission_minimum=config.commission_minimum_float,
        )
        self.config = config
        self.trade_id_int = 0
        self.current_trade_id_int = default_trade_id_int()
        self.regime_signal_df = pd.DataFrame()

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        signal_close_key_tuple = (self.config.signal_symbol_str, "Close")
        if signal_close_key_tuple not in pricing_data_df.columns:
            raise RuntimeError(
                f"Missing total-return signal close for {self.config.signal_symbol_str}."
            )
        if (self.config.trade_symbol_str, "Close") not in pricing_data_df.columns:
            raise RuntimeError(
                f"Missing execution close for {self.config.trade_symbol_str}."
            )

        self.regime_signal_df = compute_spy_adaptive_momentum_signal_df(
            signal_price_close_ser=pricing_data_df[signal_close_key_tuple],
            percentile_lookback_int=self.config.percentile_lookback_int,
            high_lookback_int=self.config.high_lookback_int,
            percentile_method_str=self.config.percentile_method_str,
            fast_lookback_int=self.config.fast_lookback_int,
            slow_lookback_int=self.config.slow_lookback_int,
            percentile_power_float=self.config.percentile_power_float,
            price_filter_lookback_int=self.config.price_filter_lookback_int,
        )
        multiindex_signal_df = self.regime_signal_df.copy()
        multiindex_signal_df.columns = pd.MultiIndex.from_tuples(
            [
                (SIGNAL_NAMESPACE_STR, field_str)
                for field_str in multiindex_signal_df.columns
            ]
        )
        return pd.concat([pricing_data_df, multiindex_signal_df], axis=1)

    def iterate(
        self,
        data_df: pd.DataFrame,
        close_row_ser: pd.Series,
        open_price_ser: pd.Series,
    ) -> None:
        if close_row_ser is None or data_df is None:
            return

        target_weight_key_tuple = (SIGNAL_NAMESPACE_STR, "target_weight_ser")
        if target_weight_key_tuple not in close_row_ser.index:
            return
        target_weight_float = float(close_row_ser.loc[target_weight_key_tuple])
        if not np.isfinite(target_weight_float):
            return

        current_share_int = int(self.get_position(self.config.trade_symbol_str))

        # *** CRITICAL*** target_weight_T is read from previous_bar=Close_T;
        # the order is created on current_bar and fills at Open_(T+1).
        if target_weight_float <= 0.0:
            if current_share_int <= 0:
                return
            self.order_target(
                self.config.trade_symbol_str,
                0,
                trade_id=self.current_trade_id_int,
            )
            self.current_trade_id_int = default_trade_id_int()
            return

        # The frozen research path buys on a 0 -> 1 state transition and then
        # holds the shares. Re-sizing an existing long every day would create
        # turnover that is absent from the vectorized research contract.
        if current_share_int > 0:
            return

        sizing_price_key_tuple = (self.config.trade_symbol_str, "Close")
        sizing_price_float = float(close_row_ser.get(sizing_price_key_tuple, np.nan))
        if not np.isfinite(sizing_price_float) or sizing_price_float <= 0.0:
            raise RuntimeError(
                f"Invalid prior close price for {self.config.trade_symbol_str} "
                f"on {self.previous_bar}."
            )

        # *** CRITICAL*** The MOO share quantity must be fixed from Close_T,
        # before Open_(T+1) exists. The fill still occurs at Open_(T+1), so an
        # overnight gap can leave a small cash balance or temporary overdraw.
        budget_value_float = float(self.previous_total_value)
        target_share_int = int(np.floor(budget_value_float / sizing_price_float))
        if target_share_int <= 0:
            return
        self.trade_id_int += 1
        self.current_trade_id_int = self.trade_id_int
        self.order_target(
            self.config.trade_symbol_str,
            target_share_int,
            trade_id=self.current_trade_id_int,
        )


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    pricing_data_df: pd.DataFrame | None = None,
    config_obj: SpyAdaptiveMomentumRegimeConfig = DEFAULT_CONFIG,
) -> SpyAdaptiveMomentumRegimeStrategy:
    """Run the research-only SPY variant with causal next-open execution."""

    if pricing_data_df is None:
        pricing_data_df = get_spy_adaptive_momentum_regime_data(config=config_obj)
    strategy_obj = SpyAdaptiveMomentumRegimeStrategy(
        name="strategy_mo_spy_adaptive_momentum_regime_research",
        benchmarks=[config_obj.benchmark_symbol_str],
        config=config_obj,
    )
    calendar_idx = build_adaptive_momentum_calendar_idx(
        pricing_data_df=pricing_data_df,
        config_obj=config_obj,
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


def _build_analysis_context_dict() -> dict[str, object]:
    config_obj = DEFAULT_CONFIG
    pricing_data_df = get_spy_adaptive_momentum_regime_data(config=config_obj)
    calendar_idx = build_adaptive_momentum_calendar_idx(
        pricing_data_df=pricing_data_df,
        config_obj=config_obj,
    )
    return {
        "strategy_name_str": "strategy_mo_spy_adaptive_momentum_regime_research",
        "capital_base_float": float(config_obj.capital_base_float),
        "config_obj": config_obj,
        "pricing_data_df": pricing_data_df,
        "calendar_idx": calendar_idx,
    }


def _build_strategy_obj(context_dict: dict[str, object]) -> SpyAdaptiveMomentumRegimeStrategy:
    config_obj = context_dict["config_obj"]
    if not isinstance(config_obj, SpyAdaptiveMomentumRegimeConfig):
        raise TypeError("context config_obj must be SpyAdaptiveMomentumRegimeConfig.")
    return SpyAdaptiveMomentumRegimeStrategy(
        name=str(context_dict["strategy_name_str"]),
        benchmarks=[config_obj.benchmark_symbol_str],
        config=config_obj,
    )


def build_execution_timing_analysis_inputs() -> dict[str, object]:
    """Build BENCH Timing inputs with the same factory/calendar as Vanilla."""

    context_dict = _build_analysis_context_dict()

    def strategy_factory_fn() -> SpyAdaptiveMomentumRegimeStrategy:
        return _build_strategy_obj(context_dict)

    return {
        "strategy_factory_fn": strategy_factory_fn,
        "pricing_data_df": context_dict["pricing_data_df"],
        "calendar_idx": context_dict["calendar_idx"],
        "order_generation_mode_str": "vanilla_current_bar",
        "risk_model_str": "daily_ohlc_signal",
        "entry_timing_str_tuple": (
            "same_open",
            "same_close_moc",
            "next_open",
            "next_close",
        ),
        "exit_timing_str_tuple": (
            "same_open",
            "same_close_moc",
            "next_open",
            "next_close",
        ),
        "default_entry_timing_str": "same_open",
        "default_exit_timing_str": "same_open",
    }


def build_stress_test_context_dict() -> dict[str, object]:
    """Build BENCH Stress inputs from the unchanged Vanilla data contract."""

    return _build_analysis_context_dict()


def build_stress_test_strategy_obj(
    context_dict: dict[str, object],
) -> SpyAdaptiveMomentumRegimeStrategy:
    return _build_strategy_obj(context_dict)


def build_capacity_analysis_inputs(
    capital_base_float: float,
    show_display_bool: bool = False,
    backtest_start_date_str: str | None = None,
    end_date_str: str | None = None,
) -> dict[str, object]:
    """Run one full-engine BENCH Capacity point with default trading costs."""

    config_obj = replace(
        DEFAULT_CONFIG,
        capital_base_float=float(capital_base_float),
        backtest_start_date_str=(
            DEFAULT_CONFIG.backtest_start_date_str
            if backtest_start_date_str is None
            else str(backtest_start_date_str)
        ),
        end_date_str=end_date_str,
    )
    pricing_data_df = get_spy_adaptive_momentum_regime_data(config=config_obj)
    strategy_obj = run_variant(
        show_display_bool=show_display_bool,
        save_results_bool=False,
        pricing_data_df=pricing_data_df,
        config_obj=config_obj,
    )
    return {
        "strategy_obj": strategy_obj,
        "pricing_data_df": pricing_data_df,
        "execution_policy_str": "MOO",
        "impact_profile_str": "MOO_ETF_PROXY",
    }


if __name__ == "__main__":
    run_variant()
