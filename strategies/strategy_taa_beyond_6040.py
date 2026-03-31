"""
Beyond 60/40 portfolio strategy.

Core formulas
-------------
Daily asset returns:

    r_{i,t}
        = P_{i,t} / P_{i,t-1} - 1

Trailing annualized asset volatility:

    sigma_{i,t}^{(63)}
        = sqrt(252) * std(r_{i,t-62:t})

Inverse-volatility monthly base weights at month-end t:

    inv_vol_{i,t}
        = 1 / sigma_{i,t}^{(63)}

    w_{i,t}^{base}
        = inv_vol_{i,t} / sum_j(inv_vol_{j,t})

Portfolio-level daily gross exposure overlay:

    sigma_{p,t}^{(63)}
        = sqrt(252) * std(r_{p,t-62:t})

    g_t
        = 1                                  if sigma_{p,t}^{(63)} <= trigger_vol
        = min(1, target_vol / sigma_{p,t})   otherwise

Final target weights:

    w_{i,t}^{final}
        = g_t * w_{i,t}^{base}

    w_t^{cash}
        = 1 - g_t

Implementation notes
--------------------
1. Signals and execution both use CAPITALSPECIAL prices for the tradeable ETFs.
2. Month-end decisions are formed from the close and become actionable at the
   next open through the engine's previous-bar contract.
3. No leverage is used. Residual allocation remains in engine cash.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.metrics import (
    generate_drawdowns,
    generate_monthly_returns,
    generate_overall_metrics,
    generate_trades,
    generate_trades_metrics,
)
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from data.norgate_loader import load_raw_prices


def default_trade_id_int() -> int:
    return -1


@dataclass(frozen=True)
class Beyond6040Config:
    asset_list: tuple[str, ...] = ("VTI", "GLD", "TLT")
    benchmark_list: tuple[str, ...] = ("$SPX",)
    asset_vol_lookback_int: int = 63
    portfolio_vol_lookback_int: int = 63
    target_portfolio_vol_float: float = 0.08
    trigger_portfolio_vol_float: float = 0.085
    start_date_str: str = "1995-01-01"
    end_date_str: str | None = None
    capital_base_float: float = 100_000.0
    slippage_float: float = 0.0001
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0

    def __post_init__(self):
        if len(self.asset_list) < 2:
            raise ValueError("asset_list must contain at least two assets.")
        if len(set(self.asset_list)) != len(self.asset_list):
            raise ValueError("asset_list contains duplicate symbols.")
        if len(set(self.benchmark_list)) != len(self.benchmark_list):
            raise ValueError("benchmark_list contains duplicate symbols.")
        if set(self.asset_list).intersection(self.benchmark_list):
            raise ValueError("benchmark_list must not overlap asset_list.")
        if self.asset_vol_lookback_int < 2:
            raise ValueError("asset_vol_lookback_int must be at least 2.")
        if self.portfolio_vol_lookback_int < 2:
            raise ValueError("portfolio_vol_lookback_int must be at least 2.")
        if not np.isfinite(self.target_portfolio_vol_float) or self.target_portfolio_vol_float <= 0.0:
            raise ValueError("target_portfolio_vol_float must be positive.")
        if (
            not np.isfinite(self.trigger_portfolio_vol_float)
            or self.trigger_portfolio_vol_float <= 0.0
        ):
            raise ValueError("trigger_portfolio_vol_float must be positive.")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if self.slippage_float < 0.0:
            raise ValueError("slippage_float must be non-negative.")
        if self.commission_per_share_float < 0.0:
            raise ValueError("commission_per_share_float must be non-negative.")
        if self.commission_minimum_float < 0.0:
            raise ValueError("commission_minimum_float must be non-negative.")


DEFAULT_CONFIG = Beyond6040Config()


def get_beyond_6040_data(
    config: Beyond6040Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Load CAPITALSPECIAL prices for tradeable assets and TOTALRETURN prices for
    optional benchmarks.
    """
    return load_raw_prices(
        symbols=list(config.asset_list),
        benchmarks=list(config.benchmark_list),
        start_date=config.start_date_str,
        end_date=config.end_date_str,
    )


def _flatten_close_df(price_close_df: pd.DataFrame) -> pd.DataFrame:
    flat_close_df = price_close_df.copy()
    if isinstance(flat_close_df.columns, pd.MultiIndex):
        flat_close_df.columns = flat_close_df.columns.get_level_values(0)
    return flat_close_df.astype(float)


def compute_month_end_inverse_vol_weight_df(
    price_close_df: pd.DataFrame,
    asset_vol_lookback_int: int = 63,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute daily asset returns, daily trailing volatilities, and month-end
    inverse-volatility base weights.
    """
    price_close_df = _flatten_close_df(price_close_df)

    # *** CRITICAL*** Returns must use only trailing closes to preserve causal
    # signal formation at each month-end decision point.
    asset_return_df = price_close_df.pct_change(fill_method=None)

    # *** CRITICAL*** Trailing volatility must be computed with a backward-only
    # rolling window. Any centered or forward-looking window would leak future risk.
    asset_vol_df = asset_return_df.rolling(asset_vol_lookback_int).std(ddof=1) * np.sqrt(252.0)

    # *** CRITICAL*** Sample the trailing volatility state at calendar
    # month-end close labels. This prevents truncated mid-month histories from
    # being misclassified as completed month-end decisions during the signal audit.
    month_end_asset_vol_df = asset_vol_df.resample("ME").last()
    valid_month_end_asset_vol_df = month_end_asset_vol_df.where(month_end_asset_vol_df > 0.0)
    inverse_vol_df = 1.0 / valid_month_end_asset_vol_df
    base_weight_df = inverse_vol_df.div(inverse_vol_df.sum(axis=1), axis=0)
    base_weight_df = base_weight_df.replace([np.inf, -np.inf], np.nan).dropna(how="any")

    if len(base_weight_df) > 0:
        weight_sum_ser = base_weight_df.sum(axis=1)
        if not np.allclose(weight_sum_ser.to_numpy(dtype=float), 1.0, atol=1e-12):
            raise ValueError("Month-end inverse-volatility weights must sum to 1.0.")

    return asset_return_df, asset_vol_df, base_weight_df


def build_signal_base_weight_df(
    month_end_weight_df: pd.DataFrame,
    execution_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Expand month-end decisions into a daily close-indexed schedule that can be
    read from `close_row_ser` inside `iterate()`.
    """
    rebalance_weight_df = map_month_end_weights_to_rebalance_open_df(
        month_end_weight_df=month_end_weight_df,
        execution_index=execution_index,
    )
    signal_base_weight_df = pd.DataFrame(
        index=pd.DatetimeIndex(execution_index),
        columns=month_end_weight_df.columns,
        dtype=float,
    )
    if len(rebalance_weight_df) == 0:
        signal_base_weight_df.index.name = "bar"
        return signal_base_weight_df

    rebalance_position_vec = execution_index.get_indexer(pd.DatetimeIndex(rebalance_weight_df.index))
    valid_mask_vec = rebalance_position_vec > 0
    if not np.any(valid_mask_vec):
        signal_base_weight_df.index.name = "bar"
        return signal_base_weight_df

    signal_effective_index = pd.DatetimeIndex(execution_index[rebalance_position_vec[valid_mask_vec] - 1])
    shifted_rebalance_weight_df = rebalance_weight_df.iloc[np.where(valid_mask_vec)[0]].copy()
    shifted_rebalance_weight_df.index = signal_effective_index

    # *** CRITICAL*** Attach each month-end decision to the previous trading bar
    # so iterate() reads it from close_row_ser before the next-month rebalance open.
    signal_base_weight_df.loc[shifted_rebalance_weight_df.index, shifted_rebalance_weight_df.columns] = (
        shifted_rebalance_weight_df
    )
    signal_base_weight_df = signal_base_weight_df.ffill()
    signal_base_weight_df.index.name = "bar"
    return signal_base_weight_df


def map_month_end_weights_to_rebalance_open_df(
    month_end_weight_df: pd.DataFrame,
    execution_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Map each month-end decision to the first trading day of the next month.
    """
    execution_month_ser = pd.Series(execution_index, index=execution_index.to_period("M"))
    first_trading_day_ser = execution_month_ser.groupby(level=0).min()

    rebalance_weight_map: dict[pd.Timestamp, pd.Series] = {}
    for decision_date, base_weight_ser in month_end_weight_df.iterrows():
        # *** CRITICAL*** Decision date t must map to the first tradable open in
        # month t + 1. Executing on the same month-end bar would create look-ahead bias.
        next_month_period = (decision_date + pd.offsets.MonthBegin(1)).to_period("M")
        if next_month_period not in first_trading_day_ser.index:
            continue

        rebalance_date = pd.Timestamp(first_trading_day_ser.loc[next_month_period])
        rebalance_weight_map[rebalance_date] = base_weight_ser.copy()

    if len(rebalance_weight_map) == 0:
        return pd.DataFrame(columns=month_end_weight_df.columns, dtype=float)

    rebalance_weight_df = pd.DataFrame.from_dict(rebalance_weight_map, orient="index").sort_index()
    rebalance_weight_df.index.name = "rebalance_date"
    return rebalance_weight_df


def get_first_actionable_rebalance_ts(
    pricing_data_df: pd.DataFrame,
    asset_list: Sequence[str],
    asset_vol_lookback_int: int = 63,
) -> pd.Timestamp:
    """
    Return the first rebalance open with valid month-end inverse-vol weights.

    This preserves the full earlier price history for signal warmup, but lets
    the actual backtest/report start at the first economically relevant bar.
    """
    asset_close_key_list = [(asset_str, "Close") for asset_str in asset_list]
    missing_key_list = [key for key in asset_close_key_list if key not in pricing_data_df.columns]
    if missing_key_list:
        raise RuntimeError(f"Missing close data for {missing_key_list}.")

    price_close_df = pricing_data_df.loc[:, asset_close_key_list].astype(float)
    _, _, month_end_weight_df = compute_month_end_inverse_vol_weight_df(
        price_close_df=price_close_df,
        asset_vol_lookback_int=asset_vol_lookback_int,
    )
    rebalance_weight_df = map_month_end_weights_to_rebalance_open_df(
        month_end_weight_df=month_end_weight_df,
        execution_index=pricing_data_df.index,
    )

    if len(rebalance_weight_df) == 0:
        raise RuntimeError("No actionable rebalance date was generated for Beyond 60/40.")

    return pd.Timestamp(rebalance_weight_df.index[0])


def compute_gross_exposure_float(
    realized_return_ser: pd.Series,
    portfolio_vol_lookback_int: int = 63,
    target_portfolio_vol_float: float = 0.08,
    trigger_portfolio_vol_float: float = 0.085,
) -> float:
    """
    Compute the no-leverage portfolio gross exposure scalar g_t.

    If fewer than `portfolio_vol_lookback_int` realized returns are available,
    return full exposure:

        g_t = 1
    """
    realized_return_ser = pd.Series(realized_return_ser, copy=True).astype(float).dropna()
    if len(realized_return_ser) < portfolio_vol_lookback_int:
        return 1.0

    trailing_return_ser = realized_return_ser.iloc[-portfolio_vol_lookback_int:]
    portfolio_vol_float = float(trailing_return_ser.std(ddof=1) * np.sqrt(252.0))
    if not np.isfinite(portfolio_vol_float) or portfolio_vol_float <= 0.0:
        return 1.0
    if portfolio_vol_float <= trigger_portfolio_vol_float:
        return 1.0
    return float(min(1.0, target_portfolio_vol_float / portfolio_vol_float))


class Beyond6040Strategy(Strategy):
    """
    Multi-asset inverse-volatility allocator with a daily portfolio volatility overlay.

    Monthly base weights are formed from trailing 63-day asset volatility at
    month-end close. Daily gross exposure is scaled by trailing realized
    strategy volatility, with residual capital left in cash.
    """

    enable_signal_audit = True
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str] | None = None,
        asset_list: Sequence[str] = DEFAULT_CONFIG.asset_list,
        asset_vol_lookback_int: int = DEFAULT_CONFIG.asset_vol_lookback_int,
        portfolio_vol_lookback_int: int = DEFAULT_CONFIG.portfolio_vol_lookback_int,
        target_portfolio_vol_float: float = DEFAULT_CONFIG.target_portfolio_vol_float,
        trigger_portfolio_vol_float: float = DEFAULT_CONFIG.trigger_portfolio_vol_float,
        capital_base: float = DEFAULT_CONFIG.capital_base_float,
        slippage: float = DEFAULT_CONFIG.slippage_float,
        commission_per_share: float = DEFAULT_CONFIG.commission_per_share_float,
        commission_minimum: float = DEFAULT_CONFIG.commission_minimum_float,
    ):
        benchmark_list = [] if benchmarks is None else list(benchmarks)
        super().__init__(
            name=name,
            benchmarks=benchmark_list,
            capital_base=capital_base,
            slippage=slippage,
            commission_per_share=commission_per_share,
            commission_minimum=commission_minimum,
        )
        self.asset_list = list(asset_list)
        if len(self.asset_list) != len(set(self.asset_list)):
            raise ValueError("asset_list contains duplicate symbols.")
        self.asset_vol_lookback_int = int(asset_vol_lookback_int)
        self.portfolio_vol_lookback_int = int(portfolio_vol_lookback_int)
        self.target_portfolio_vol_float = float(target_portfolio_vol_float)
        self.trigger_portfolio_vol_float = float(trigger_portfolio_vol_float)

        self.trade_id_int = 0
        self.current_trade_id_map: defaultdict[str, int] = defaultdict(default_trade_id_int)
        self.show_taa_weights_report = True
        self.daily_target_weight_map: dict[pd.Timestamp, pd.Series] = {}
        self.daily_target_weights = pd.DataFrame(columns=self.asset_list + ["Cash"], dtype=float)
        self.month_end_weight_df = pd.DataFrame(columns=self.asset_list, dtype=float)
        self.rebalance_weight_df = pd.DataFrame(columns=self.asset_list, dtype=float)

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        signal_data_df = pricing_data_df.copy()
        asset_close_key_list = [(asset_str, "Close") for asset_str in self.asset_list]
        missing_key_list = [key for key in asset_close_key_list if key not in signal_data_df.columns]
        if missing_key_list:
            raise RuntimeError(f"Missing close data for {missing_key_list}.")

        price_close_df = signal_data_df.loc[:, asset_close_key_list].astype(float)
        asset_return_df, asset_vol_df, month_end_weight_df = compute_month_end_inverse_vol_weight_df(
            price_close_df=price_close_df,
            asset_vol_lookback_int=self.asset_vol_lookback_int,
        )
        signal_base_weight_df = build_signal_base_weight_df(
            month_end_weight_df=month_end_weight_df,
            execution_index=signal_data_df.index,
        )

        feature_df = pd.DataFrame(index=signal_data_df.index)
        for asset_str in self.asset_list:
            feature_df[(asset_str, "return_ser")] = asset_return_df[asset_str]
            feature_df[(asset_str, "volatility_ser")] = asset_vol_df[asset_str]
            feature_df[(asset_str, "base_weight_ser")] = signal_base_weight_df[asset_str]

        feature_df.columns = pd.MultiIndex.from_tuples(feature_df.columns)
        return pd.concat([signal_data_df, feature_df], axis=1)

    def signal_audit_fields(self, pricing_data: pd.DataFrame, signal_data: pd.DataFrame):
        audit_col_list = super().signal_audit_fields(pricing_data, signal_data)
        return [
            col
            for col in audit_col_list
            if not (isinstance(col, tuple) and len(col) >= 2 and col[1] == "base_weight_ser")
        ]

    def _current_base_weight_ser(self, close_row_ser: pd.Series) -> pd.Series:
        base_weight_dict: dict[str, float] = {}
        for asset_str in self.asset_list:
            base_weight_key = (asset_str, "base_weight_ser")
            base_weight_dict[asset_str] = float(close_row_ser.get(base_weight_key, np.nan))
        return pd.Series(base_weight_dict, dtype=float)

    def _realized_strategy_return_ser(self) -> pd.Series:
        if len(self._daily_return_history_list) <= 1:
            return pd.Series(dtype=float)
        return pd.Series(self._daily_return_history_list[1:], dtype=float)

    def _record_daily_target_weight_ser(self, target_weight_ser: pd.Series):
        ordered_weight_ser = target_weight_ser.reindex(self.asset_list + ["Cash"]).astype(float)
        self.daily_target_weight_map[pd.Timestamp(self.current_bar)] = ordered_weight_ser
        self.daily_target_weights.loc[pd.Timestamp(self.current_bar), ordered_weight_ser.index] = ordered_weight_ser

    def iterate(self, data_df: pd.DataFrame, close_row_ser: pd.Series, open_price_ser: pd.Series):
        if close_row_ser is None or data_df is None:
            return

        base_weight_ser = self._current_base_weight_ser(close_row_ser).astype(float)
        if base_weight_ser.isna().any():
            cash_only_weight_ser = pd.Series(
                [0.0] * len(self.asset_list) + [1.0],
                index=self.asset_list + ["Cash"],
                dtype=float,
            )
            self._record_daily_target_weight_ser(cash_only_weight_ser)
            return

        realized_return_ser = self._realized_strategy_return_ser()
        gross_exposure_float = compute_gross_exposure_float(
            realized_return_ser=realized_return_ser,
            portfolio_vol_lookback_int=self.portfolio_vol_lookback_int,
            target_portfolio_vol_float=self.target_portfolio_vol_float,
            trigger_portfolio_vol_float=self.trigger_portfolio_vol_float,
        )

        target_weight_ser = (base_weight_ser * gross_exposure_float).astype(float)
        cash_weight_float = float(1.0 - target_weight_ser.sum())
        target_weight_ser = pd.concat(
            [target_weight_ser, pd.Series({"Cash": cash_weight_float}, dtype=float)]
        )

        if not np.isclose(float(target_weight_ser.sum()), 1.0, atol=1e-12):
            raise ValueError(
                f"Final target weights must sum to 1.0 on {self.current_bar}, "
                f"found {target_weight_ser.sum():.12f}."
            )

        self._record_daily_target_weight_ser(target_weight_ser)
        current_position_ser = self.get_positions().reindex(self.asset_list, fill_value=0.0).astype(int)
        budget_value_float = float(self.previous_total_value)

        for asset_str in self.asset_list:
            target_weight_float = float(target_weight_ser.loc[asset_str])
            current_share_int = int(current_position_ser.loc[asset_str])
            open_price_float = float(open_price_ser.get(asset_str, np.nan))
            if not np.isfinite(open_price_float) or open_price_float <= 0.0:
                raise RuntimeError(f"Invalid open price for target asset {asset_str} on {self.current_bar}.")

            # *** CRITICAL*** Target shares are computed from previous-bar
            # total value and executed at the current open. Same-bar close data is forbidden.
            target_share_int = int(np.floor(budget_value_float * target_weight_float / open_price_float))

            if target_share_int == current_share_int:
                continue

            if target_share_int <= 0:
                if current_share_int <= 0:
                    continue
                self.order_target_value(
                    asset_str,
                    0.0,
                    trade_id=self.current_trade_id_map[asset_str],
                )
                self.current_trade_id_map[asset_str] = default_trade_id_int()
                continue

            if current_share_int <= 0 or self.current_trade_id_map[asset_str] == default_trade_id_int():
                self.trade_id_int += 1
                self.current_trade_id_map[asset_str] = self.trade_id_int

            self.order_target_percent(
                asset_str,
                target_weight_float,
                trade_id=self.current_trade_id_map[asset_str],
            )

    def finalize(self, current_data: pd.DataFrame):
        asset_close_key_list = [(asset_str, "Close") for asset_str in self.asset_list]
        if not all(key in current_data.columns for key in asset_close_key_list):
            return

        price_close_df = current_data.loc[:, asset_close_key_list].astype(float)
        _, _, month_end_weight_df = compute_month_end_inverse_vol_weight_df(
            price_close_df=price_close_df,
            asset_vol_lookback_int=self.asset_vol_lookback_int,
        )
        self.month_end_weight_df = month_end_weight_df
        self.rebalance_weight_df = map_month_end_weights_to_rebalance_open_df(
            month_end_weight_df=month_end_weight_df,
            execution_index=current_data.index,
        )

        if len(self.daily_target_weight_map) > 0:
            self.daily_target_weights = pd.DataFrame.from_dict(
                self.daily_target_weight_map,
                orient="index",
            ).sort_index()
            self.daily_target_weights.index = pd.to_datetime(self.daily_target_weights.index)
            self.daily_target_weights = self.daily_target_weights.reindex(
                columns=self.asset_list + ["Cash"]
            )

    def summarize(self, include_benchmarks: bool = True):
        self._trades = generate_trades(self.get_transactions())
        self._drawdowns = generate_drawdowns(self.results["drawdown"])
        self.summary = pd.DataFrame()
        total_commissions_float = float(self.get_transactions()["commission"].sum())
        self.summary["Strategy"] = generate_overall_metrics(
            self.results["total_value"].astype(float),
            self._trades,
            self.results["portfolio_value"],
            self.results["daily_returns"],
            capital_base=self._capital_base,
            total_commissions=total_commissions_float,
        )

        if include_benchmarks:
            for benchmark_str in self._benchmarks:
                self.summary[benchmark_str] = generate_overall_metrics(
                    self.results[benchmark_str].astype(float),
                    None,
                    None,
                    self.results["daily_returns"],
                )

        if self._trades is not None and len(self._trades) > 0:
            self.summary_trades = generate_trades_metrics(self._trades, self.results.index)
        else:
            self.summary_trades = pd.DataFrame()

        self.results.index = pd.to_datetime(self.results.index)
        self.monthly_returns = generate_monthly_returns(
            self.results["total_value"],
            add_sharpe_ratios=True,
            add_max_drawdowns=True,
        )


if __name__ == "__main__":
    config = DEFAULT_CONFIG
    pricing_data_df = get_beyond_6040_data(config=config)
    relevant_start_ts = get_first_actionable_rebalance_ts(
        pricing_data_df=pricing_data_df,
        asset_list=config.asset_list,
        asset_vol_lookback_int=config.asset_vol_lookback_int,
    )
    calendar_index = pricing_data_df.index[pricing_data_df.index >= relevant_start_ts]

    strategy = Beyond6040Strategy(
        name="strategy_taa_beyond_6040",
        benchmarks=config.benchmark_list,
        asset_list=config.asset_list,
        asset_vol_lookback_int=config.asset_vol_lookback_int,
        portfolio_vol_lookback_int=config.portfolio_vol_lookback_int,
        target_portfolio_vol_float=config.target_portfolio_vol_float,
        trigger_portfolio_vol_float=config.trigger_portfolio_vol_float,
        capital_base=config.capital_base_float,
        slippage=config.slippage_float,
        commission_per_share=config.commission_per_share_float,
        commission_minimum=config.commission_minimum_float,
    )

    run_daily(
        strategy,
        pricing_data_df,
        calendar=calendar_index,
        audit_override_bool=None,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    display(strategy.summary)
    display(strategy.summary_trades)
    save_results(strategy)
