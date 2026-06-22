"""
Daily-data EVRP VIX ETN research approximation.

TL;DR: This is the first research-only implementation of the Concretum EVRP
idea using tradable daily ETF/ETN data available in this repo:

    short-vol leg: SVXY
    long-vol leg:  VXX

It is not an exact paper replication. The paper computes the signal around
15:44/15:45 ET from intraday SPY, VIX, and VIX3M data, then submits a
Market-on-Close order. This module uses daily closes as a first-pass MOC
approximation and labels that timing assumption explicitly.

Core formulas
-------------
At close t:

    r_t^{SPY} = SPY_t / SPY_{t-1} - 1

    eRV30_t = std(r_{t-9}^{SPY}, ..., r_t^{SPY}) * sqrt(252) * 100

    eVRP_t = VIX_t - eRV30_t

Target weights:

    if eVRP_t > 0 and VIX_t < VIX3M_t:
        w_t^{SVXY} = min(2 * VIX_t / 100, cap)

    elif eVRP_t <= 0 and VIX_t < VIX3M_t:
        w_t^{SVXY} = min(VIX_t / 100, cap)

    elif eVRP_t <= 0 and VIX_t > VIX3M_t:
        w_t^{VXX} = min(VIX_t / 100, cap)

    else:
        cash

Execution approximation:

    target_t is formed at close t
    rebalance_t happens at close t if the 2pp band is breached
    realized P&L starts from close t to close t+1

Costs:

    cost_t = turnover_t * 5bps

Research-only gaps
------------------
This path uses actual SVXY/VXX daily closes, not the paper's Bloomberg
VIXSHORT/VIXLONG proxy indexes. It starts after SVXY's post-Volmageddon
0.5x leverage convention and caps dollar weight at 100% by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd
from IPython.display import display

WORKSPACE_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT_PATH))

from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from data.norgate_loader import load_raw_prices


MOC_APPROX_TIMING_MODE_STR = "daily_close_moc_approx"
ONE_DAY_DELAY_TIMING_MODE_STR = "one_day_delayed_close"
SUPPORTED_TIMING_MODE_TUPLE = (
    MOC_APPROX_TIMING_MODE_STR,
    ONE_DAY_DELAY_TIMING_MODE_STR,
)


@dataclass(frozen=True)
class EvrpVixEtnResearchConfig:
    short_vol_symbol_str: str = "SVXY"
    long_vol_symbol_str: str = "VXX"
    spy_symbol_str: str = "SPY"
    vix_symbol_str: str = "$VIX"
    vix3m_symbol_str: str = "$VIX3M"
    benchmark_symbol_str: str = "SPY"
    realized_vol_lookback_int: int = 10
    svxy_short_vol_multiplier_float: float = 2.0
    transaction_cost_bps_float: float = 5.0
    rebalance_threshold_float: float = 0.02
    max_asset_weight_float: float = 1.0
    timing_mode_str: str = MOC_APPROX_TIMING_MODE_STR
    start_date_str: str = "2018-03-01"
    end_date_str: str | None = None
    capital_base_float: float = 100_000.0

    def __post_init__(self):
        if self.short_vol_symbol_str == self.long_vol_symbol_str:
            raise ValueError("short_vol_symbol_str and long_vol_symbol_str must differ.")
        if self.realized_vol_lookback_int < 2:
            raise ValueError("realized_vol_lookback_int must be >= 2.")
        if self.svxy_short_vol_multiplier_float <= 0.0 or not np.isfinite(self.svxy_short_vol_multiplier_float):
            raise ValueError("svxy_short_vol_multiplier_float must be positive and finite.")
        if self.transaction_cost_bps_float < 0.0 or not np.isfinite(self.transaction_cost_bps_float):
            raise ValueError("transaction_cost_bps_float must be non-negative and finite.")
        if self.rebalance_threshold_float < 0.0 or not np.isfinite(self.rebalance_threshold_float):
            raise ValueError("rebalance_threshold_float must be non-negative and finite.")
        if self.max_asset_weight_float <= 0.0 or not np.isfinite(self.max_asset_weight_float):
            raise ValueError("max_asset_weight_float must be positive and finite.")
        if self.timing_mode_str not in SUPPORTED_TIMING_MODE_TUPLE:
            raise ValueError(f"timing_mode_str must be one of {SUPPORTED_TIMING_MODE_TUPLE}.")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")


DEFAULT_CONFIG = EvrpVixEtnResearchConfig()


def get_evrp_vix_etn_research_data(
    config: EvrpVixEtnResearchConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    return load_raw_prices(
        symbols=[
            config.short_vol_symbol_str,
            config.long_vol_symbol_str,
            config.spy_symbol_str,
            config.vix_symbol_str,
            config.vix3m_symbol_str,
        ],
        benchmarks=[],
        start_date=config.start_date_str,
        end_date=config.end_date_str,
    )


def _timing_lag_int(timing_mode_str: str) -> int:
    if timing_mode_str == MOC_APPROX_TIMING_MODE_STR:
        return 1
    if timing_mode_str == ONE_DAY_DELAY_TIMING_MODE_STR:
        return 2
    raise ValueError(f"Unsupported timing_mode_str: {timing_mode_str}")


def _cagr_float(total_value_ser: pd.Series) -> float:
    elapsed_year_float = max((total_value_ser.index[-1] - total_value_ser.index[0]).days / 365.25, 1.0 / 365.25)
    return float((float(total_value_ser.iloc[-1]) / float(total_value_ser.iloc[0])) ** (1.0 / elapsed_year_float) - 1.0)


def compute_evrp_signal_df(
    spy_close_ser: pd.Series,
    vix_close_ser: pd.Series,
    vix3m_close_ser: pd.Series,
    config: EvrpVixEtnResearchConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Compute EVRP regime state and raw target weights.

        eRV30_t = std(last N SPY close-to-close returns through t) * sqrt(252) * 100
        eVRP_t = VIX_t - eRV30_t

    This function intentionally uses same-date daily close as the unavailable
    15:44/15:45 snapshot proxy for the first research pass.
    """
    signal_input_df = pd.concat(
        [
            pd.Series(spy_close_ser, copy=True).astype(float).rename("spy_close_ser"),
            pd.Series(vix_close_ser, copy=True).astype(float).rename("vix_close_ser"),
            pd.Series(vix3m_close_ser, copy=True).astype(float).rename("vix3m_close_ser"),
        ],
        axis=1,
    ).sort_index()

    # *** CRITICAL*** MOC approximation: pct_change() and rolling() include the
    # signal day's final daily close because the repo has no historical 15:44
    # intraday feed. This is research-only and must not be described as exact
    # paper timing or live-clean same-day MOC execution.
    spy_return_ser = signal_input_df["spy_close_ser"].pct_change(fill_method=None)
    expected_realized_vol_ser = (
        spy_return_ser.rolling(window=int(config.realized_vol_lookback_int), min_periods=int(config.realized_vol_lookback_int))
        .std(ddof=1)
        * np.sqrt(252.0)
        * 100.0
    )
    evrp_ser = signal_input_df["vix_close_ser"] - expected_realized_vol_ser

    contango_bool_ser = signal_input_df["vix_close_ser"] < signal_input_df["vix3m_close_ser"]
    backwardation_bool_ser = signal_input_df["vix_close_ser"] > signal_input_df["vix3m_close_ser"]
    positive_evrp_bool_ser = evrp_ser > 0.0
    nonpositive_evrp_bool_ser = evrp_ser <= 0.0

    full_short_weight_ser = (
        signal_input_df["vix_close_ser"] / 100.0 * float(config.svxy_short_vol_multiplier_float)
    ).clip(lower=0.0, upper=float(config.max_asset_weight_float))
    half_short_weight_ser = (signal_input_df["vix_close_ser"] / 100.0).clip(
        lower=0.0,
        upper=float(config.max_asset_weight_float),
    )
    long_vol_weight_ser = (signal_input_df["vix_close_ser"] / 100.0).clip(
        lower=0.0,
        upper=float(config.max_asset_weight_float),
    )

    raw_short_weight_ser = pd.Series(0.0, index=signal_input_df.index, dtype=float)
    raw_long_weight_ser = pd.Series(0.0, index=signal_input_df.index, dtype=float)

    high_conviction_short_bool_ser = positive_evrp_bool_ser & contango_bool_ser
    medium_conviction_short_bool_ser = nonpositive_evrp_bool_ser & contango_bool_ser
    long_vol_bool_ser = nonpositive_evrp_bool_ser & backwardation_bool_ser
    cash_bool_ser = positive_evrp_bool_ser & backwardation_bool_ser

    raw_short_weight_ser.loc[high_conviction_short_bool_ser] = full_short_weight_ser.loc[
        high_conviction_short_bool_ser
    ]
    raw_short_weight_ser.loc[medium_conviction_short_bool_ser] = half_short_weight_ser.loc[
        medium_conviction_short_bool_ser
    ]
    raw_long_weight_ser.loc[long_vol_bool_ser] = long_vol_weight_ser.loc[long_vol_bool_ser]

    regime_label_ser = pd.Series("cash", index=signal_input_df.index, dtype=object)
    regime_label_ser.loc[high_conviction_short_bool_ser] = "short_vol_high_conviction"
    regime_label_ser.loc[medium_conviction_short_bool_ser] = "short_vol_medium_conviction"
    regime_label_ser.loc[long_vol_bool_ser] = "long_vol"
    regime_label_ser.loc[cash_bool_ser] = "cash_mixed_signal"

    signal_df = pd.DataFrame(
        {
            "spy_close_ser": signal_input_df["spy_close_ser"],
            "spy_return_ser": spy_return_ser,
            "expected_realized_vol_ser": expected_realized_vol_ser,
            "vix_close_ser": signal_input_df["vix_close_ser"],
            "vix3m_close_ser": signal_input_df["vix3m_close_ser"],
            "evrp_ser": evrp_ser,
            "contango_bool_ser": contango_bool_ser.astype(bool),
            "backwardation_bool_ser": backwardation_bool_ser.astype(bool),
            "positive_evrp_bool_ser": positive_evrp_bool_ser.astype(bool),
            "nonpositive_evrp_bool_ser": nonpositive_evrp_bool_ser.astype(bool),
            "regime_label_ser": regime_label_ser,
            config.short_vol_symbol_str: raw_short_weight_ser,
            config.long_vol_symbol_str: raw_long_weight_ser,
        },
        index=signal_input_df.index,
    )
    return signal_df


def build_rebalanced_weight_df(
    raw_target_weight_df: pd.DataFrame,
    rebalance_threshold_float: float,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply the paper's no-trade band to target weights.

        trade_t = 1[max_i |target_{i,t} - held_{i,t-1}| > threshold]
        turnover_t = sum_i |held_{i,t} - held_{i,t-1}|
    """
    if rebalance_threshold_float < 0.0 or not np.isfinite(rebalance_threshold_float):
        raise ValueError("rebalance_threshold_float must be non-negative and finite.")
    if raw_target_weight_df.empty:
        raise ValueError("raw_target_weight_df must not be empty.")

    raw_target_weight_df = raw_target_weight_df.astype(float).fillna(0.0)
    previous_weight_ser = pd.Series(0.0, index=raw_target_weight_df.columns, dtype=float)
    executed_weight_row_list: list[pd.Series] = []
    turnover_float_list: list[float] = []

    for _bar_ts, target_weight_ser in raw_target_weight_df.iterrows():
        weight_diff_ser = target_weight_ser.astype(float) - previous_weight_ser
        if float(weight_diff_ser.abs().max()) > float(rebalance_threshold_float):
            next_weight_ser = target_weight_ser.astype(float)
        else:
            next_weight_ser = previous_weight_ser.copy()

        turnover_float = float((next_weight_ser - previous_weight_ser).abs().sum())
        executed_weight_row_list.append(next_weight_ser)
        turnover_float_list.append(turnover_float)
        previous_weight_ser = next_weight_ser

    executed_weight_df = pd.DataFrame(executed_weight_row_list, index=raw_target_weight_df.index)
    turnover_ser = pd.Series(turnover_float_list, index=raw_target_weight_df.index, dtype=float, name="turnover_ser")
    return executed_weight_df.astype(float), turnover_ser


def build_evrp_results_df(
    total_value_ser: pd.Series,
    portfolio_value_ser: pd.Series,
    cash_ser: pd.Series,
    benchmark_equity_map: dict[str, pd.Series],
) -> pd.DataFrame:
    results_df = pd.DataFrame(index=total_value_ser.index)
    results_df["portfolio_value"] = portfolio_value_ser.astype(float)
    results_df["cash"] = cash_ser.astype(float)
    results_df["total_value"] = total_value_ser.astype(float)

    daily_return_ser = total_value_ser.pct_change(fill_method=None).fillna(0.0).astype(float)
    results_df["daily_returns"] = daily_return_ser
    results_df["total_returns"] = total_value_ser.astype(float) / float(total_value_ser.iloc[0]) - 1.0

    elapsed_day_count_ser = pd.Series(
        np.arange(1, len(total_value_ser.index) + 1, dtype=float),
        index=total_value_ser.index,
        dtype=float,
    )
    results_df["annualized_returns"] = (
        (total_value_ser.astype(float) / float(total_value_ser.iloc[0])) ** (252.0 / elapsed_day_count_ser) - 1.0
    )
    results_df["annualized_volatility"] = daily_return_ser.expanding(min_periods=2).std(ddof=1) * np.sqrt(252.0)
    results_df["sharpe_ratio"] = (
        daily_return_ser.expanding(min_periods=2).mean()
        / daily_return_ser.expanding(min_periods=2).std(ddof=1)
        * np.sqrt(252.0)
    )
    drawdown_ser = total_value_ser.astype(float) / total_value_ser.astype(float).cummax() - 1.0
    results_df["drawdown"] = drawdown_ser.astype(float)
    results_df["max_drawdown"] = drawdown_ser.cummin().astype(float)

    for benchmark_str, benchmark_equity_ser in benchmark_equity_map.items():
        benchmark_equity_ser = benchmark_equity_ser.astype(float)
        benchmark_drawdown_ser = benchmark_equity_ser / benchmark_equity_ser.cummax() - 1.0
        results_df[benchmark_str] = benchmark_equity_ser
        results_df[f"{benchmark_str}_drawdown"] = benchmark_drawdown_ser.astype(float)
        results_df[f"{benchmark_str}_max_drawdown"] = benchmark_drawdown_ser.cummin().astype(float)

    return results_df


def build_metric_summary_df(
    results_df: pd.DataFrame,
    realized_weight_df: pd.DataFrame,
    turnover_ser: pd.Series,
    transaction_cost_ser: pd.Series,
    config: EvrpVixEtnResearchConfig,
) -> pd.DataFrame:
    daily_return_ser = results_df["daily_returns"].astype(float)
    total_value_ser = results_df["total_value"].astype(float)
    elapsed_year_float = max((total_value_ser.index[-1] - total_value_ser.index[0]).days / 365.25, 1.0 / 365.25)
    annual_volatility_float = float(daily_return_ser.std(ddof=1) * np.sqrt(252.0))
    sharpe_float = np.nan
    if annual_volatility_float > 0.0 and np.isfinite(annual_volatility_float):
        sharpe_float = float(daily_return_ser.mean() / daily_return_ser.std(ddof=1) * np.sqrt(252.0))

    max_drawdown_float = float(results_df["drawdown"].min())
    cagr_float = _cagr_float(total_value_ser)
    mar_float = np.nan if max_drawdown_float == 0.0 else float(cagr_float / abs(max_drawdown_float))
    exposure_bool_ser = realized_weight_df[[config.short_vol_symbol_str, config.long_vol_symbol_str]].abs().sum(axis=1) > 0.0

    metric_dict = {
        "timing_mode_str": config.timing_mode_str,
        "start_date_str": str(total_value_ser.index[0].date()),
        "end_date_str": str(total_value_ser.index[-1].date()),
        "cagr_float": cagr_float,
        "annual_volatility_float": annual_volatility_float,
        "sharpe_float": sharpe_float,
        "max_drawdown_float": max_drawdown_float,
        "mar_float": mar_float,
        "final_equity_float": float(total_value_ser.iloc[-1]),
        "total_turnover_float": float(turnover_ser.sum()),
        "annual_turnover_float": float(turnover_ser.sum() / elapsed_year_float),
        "total_cost_drag_float": float(transaction_cost_ser.sum()),
        "annual_cost_drag_float": float(transaction_cost_ser.sum() / elapsed_year_float),
        "trade_count_int": int((turnover_ser > 0.0).sum()),
        "trades_per_year_float": float((turnover_ser > 0.0).sum() / elapsed_year_float),
        "avg_abs_exposure_float": float(
            realized_weight_df[[config.short_vol_symbol_str, config.long_vol_symbol_str]].abs().sum(axis=1).mean()
        ),
        "pct_short_vol_float": float((realized_weight_df[config.short_vol_symbol_str] > 0.0).mean()),
        "pct_long_vol_float": float((realized_weight_df[config.long_vol_symbol_str] > 0.0).mean()),
        "pct_cash_float": float((~exposure_bool_ser).mean()),
    }
    return pd.DataFrame([metric_dict])


def build_annual_return_df(results_df: pd.DataFrame) -> pd.DataFrame:
    annual_return_ser = (1.0 + results_df["daily_returns"].astype(float)).groupby(results_df.index.year).prod() - 1.0
    annual_return_df = annual_return_ser.rename("strategy_return_float").to_frame()
    annual_return_df.index.name = "year_int"
    return annual_return_df


class EvrpVixEtnResearchStrategy(Strategy):
    """Research-only Strategy wrapper around the vectorized EVRP close model."""

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        config: EvrpVixEtnResearchConfig,
        capital_base: float = 100_000.0,
    ):
        super().__init__(
            name=name,
            benchmarks=list(benchmarks),
            capital_base=capital_base,
            slippage=0.0,
            commission_per_share=0.0,
            commission_minimum=0.0,
        )
        self.config = config
        self.signal_state_df = pd.DataFrame()
        self.daily_target_weight_df = pd.DataFrame()
        self.turnover_ser = pd.Series(dtype=float)
        self.transaction_cost_ser = pd.Series(dtype=float)
        self.metric_summary_df = pd.DataFrame()
        self.annual_return_df = pd.DataFrame()

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        return pricing_data_df

    def iterate(self, data_df: pd.DataFrame, close_row_ser: pd.Series, open_price_ser: pd.Series):
        return


def run_evrp_research_backtest(
    strategy: EvrpVixEtnResearchStrategy,
    pricing_data_df: pd.DataFrame,
) -> EvrpVixEtnResearchStrategy:
    config = strategy.config
    required_key_list = [
        (config.short_vol_symbol_str, "Close"),
        (config.long_vol_symbol_str, "Close"),
        (config.spy_symbol_str, "Close"),
        (config.vix_symbol_str, "Close"),
        (config.vix3m_symbol_str, "Close"),
    ]
    missing_key_list = [key_tup for key_tup in required_key_list if key_tup not in pricing_data_df.columns]
    if len(missing_key_list) > 0:
        raise RuntimeError(f"Missing required close columns: {missing_key_list}")

    close_price_df = pricing_data_df.loc[:, required_key_list].copy()
    close_price_df.columns = [key_tup[0] for key_tup in required_key_list]
    close_price_df = close_price_df.astype(float).dropna(how="any")
    if len(close_price_df) <= config.realized_vol_lookback_int + _timing_lag_int(config.timing_mode_str):
        raise RuntimeError("Not enough shared close history for EVRP research backtest.")

    signal_state_df = compute_evrp_signal_df(
        spy_close_ser=close_price_df[config.spy_symbol_str],
        vix_close_ser=close_price_df[config.vix_symbol_str],
        vix3m_close_ser=close_price_df[config.vix3m_symbol_str],
        config=config,
    )
    raw_target_weight_df = signal_state_df[[config.short_vol_symbol_str, config.long_vol_symbol_str]].fillna(0.0)
    executed_weight_df, turnover_ser = build_rebalanced_weight_df(
        raw_target_weight_df=raw_target_weight_df,
        rebalance_threshold_float=config.rebalance_threshold_float,
    )

    timing_lag_int = _timing_lag_int(config.timing_mode_str)
    # *** CRITICAL*** Timing alignment: target weights are formed on signal day
    # t, but returns are earned only after the modeled close fill. For the
    # daily-close MOC approximation, return_t uses weight_{t-1}. For the
    # strict one-day-delayed diagnostic, return_t uses weight_{t-2}.
    realized_asset_weight_df = executed_weight_df.shift(timing_lag_int).fillna(0.0)
    realized_turnover_ser = turnover_ser.shift(timing_lag_int).fillna(0.0)
    transaction_cost_ser = realized_turnover_ser * (float(config.transaction_cost_bps_float) / 10_000.0)

    trade_return_df = close_price_df[[config.short_vol_symbol_str, config.long_vol_symbol_str]].pct_change(
        fill_method=None
    ).fillna(0.0)
    gross_daily_return_ser = (realized_asset_weight_df * trade_return_df).sum(axis=1)
    daily_return_ser = gross_daily_return_ser - transaction_cost_ser
    total_value_ser = float(config.capital_base_float) * (1.0 + daily_return_ser).cumprod()

    asset_exposure_ser = realized_asset_weight_df.abs().sum(axis=1)
    portfolio_value_ser = total_value_ser * asset_exposure_ser
    cash_ser = total_value_ser - portfolio_value_ser
    benchmark_equity_ser = close_price_df[config.benchmark_symbol_str] / float(close_price_df[config.benchmark_symbol_str].iloc[0])
    benchmark_equity_ser = benchmark_equity_ser * float(config.capital_base_float)

    realized_weight_df = realized_asset_weight_df.copy()
    realized_weight_df["Cash"] = 1.0 - realized_asset_weight_df.sum(axis=1)
    daily_target_weight_df = executed_weight_df.copy()
    daily_target_weight_df["Cash"] = 1.0 - executed_weight_df.sum(axis=1)

    results_df = build_evrp_results_df(
        total_value_ser=total_value_ser,
        portfolio_value_ser=portfolio_value_ser,
        cash_ser=cash_ser,
        benchmark_equity_map={config.benchmark_symbol_str: benchmark_equity_ser},
    )

    strategy.results = results_df
    strategy.signal_state_df = signal_state_df.assign(
        executed_short_weight_ser=executed_weight_df[config.short_vol_symbol_str],
        executed_long_weight_ser=executed_weight_df[config.long_vol_symbol_str],
        realized_short_weight_ser=realized_weight_df[config.short_vol_symbol_str],
        realized_long_weight_ser=realized_weight_df[config.long_vol_symbol_str],
        turnover_ser=turnover_ser,
        realized_turnover_ser=realized_turnover_ser,
        transaction_cost_ser=transaction_cost_ser,
        gross_daily_return_ser=gross_daily_return_ser,
        daily_return_ser=daily_return_ser,
    )
    strategy.daily_target_weight_df = daily_target_weight_df
    strategy.realized_weight_df = realized_weight_df
    strategy.turnover_ser = realized_turnover_ser
    strategy.transaction_cost_ser = transaction_cost_ser
    strategy.metric_summary_df = build_metric_summary_df(
        results_df=results_df,
        realized_weight_df=realized_weight_df,
        turnover_ser=realized_turnover_ser,
        transaction_cost_ser=transaction_cost_ser,
        config=config,
    )
    strategy.annual_return_df = build_annual_return_df(results_df)
    strategy._transactions = strategy.initialize_transactions()
    strategy.cash = float(cash_ser.iloc[-1])
    strategy.portfolio_value = float(portfolio_value_ser.iloc[-1])
    strategy.total_value = float(total_value_ser.iloc[-1])
    strategy.current_bar = pd.Timestamp(total_value_ser.index[-1])
    strategy._latest_close_price_ser = close_price_df.iloc[-1].astype(float)
    strategy.summarize()

    if strategy.summary is not None and "Strategy" in strategy.summary.columns:
        exposure_time_float = float(
            (
                realized_weight_df[[config.short_vol_symbol_str, config.long_vol_symbol_str]]
                .abs()
                .sum(axis=1)
                > 0.0
            ).mean()
            * 100.0
        )
        strategy.summary.loc["Exposure Time [%]", "Strategy"] = exposure_time_float
        if exposure_time_float > 0.0:
            strategy.summary.loc["Exposure-Adjusted Return (Ann.) [%]", "Strategy"] = (
                strategy.summary.loc["Return (Ann.) [%]", "Strategy"] / exposure_time_float * 100.0
            )

    return strategy


def _write_assumptions_md(output_path: Path, config: EvrpVixEtnResearchConfig) -> None:
    assumption_md_str = f"""# EVRP VIX ETN Research Assumptions

- This is a research-only daily-data approximation, not live wiring.
- Timing mode: `{config.timing_mode_str}`.
- Daily close is used as a proxy for the paper's 15:44/15:45 ET signal snapshot.
- Position returns start only after the modeled close rebalance lag.
- Tradable legs are `{config.short_vol_symbol_str}` and `{config.long_vol_symbol_str}`.
- `{config.short_vol_symbol_str}` target weight uses multiplier `{config.svxy_short_vol_multiplier_float}` and is capped at `{config.max_asset_weight_float}`.
- Rebalance threshold is `{config.rebalance_threshold_float:.4f}`.
- Cost is `{config.transaction_cost_bps_float:.2f}` bps of traded notional.
- The paper's VIXSHORT/VIXLONG proxy indexes and IQFeed minute data are not used here.
"""
    (output_path / "evrp_assumptions.md").write_text(assumption_md_str, encoding="utf-8")


def run_variant(
    show_display_bool: bool = False,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    end_date_str: str | None = None,
    capital_base_float: float | None = None,
    timing_mode_str: str | None = None,
) -> EvrpVixEtnResearchStrategy:
    config = DEFAULT_CONFIG
    if (
        backtest_start_date_str is not None
        or end_date_str is not None
        or capital_base_float is not None
        or timing_mode_str is not None
    ):
        config = EvrpVixEtnResearchConfig(
            start_date_str=backtest_start_date_str or DEFAULT_CONFIG.start_date_str,
            end_date_str=end_date_str if end_date_str is not None else DEFAULT_CONFIG.end_date_str,
            capital_base_float=capital_base_float if capital_base_float is not None else DEFAULT_CONFIG.capital_base_float,
            timing_mode_str=timing_mode_str or DEFAULT_CONFIG.timing_mode_str,
        )

    pricing_data_df = get_evrp_vix_etn_research_data(config=config)
    strategy = EvrpVixEtnResearchStrategy(
        name="strategy_evrp_vix_etn_research",
        benchmarks=[config.benchmark_symbol_str],
        config=config,
        capital_base=config.capital_base_float,
    )
    run_evrp_research_backtest(strategy=strategy, pricing_data_df=pricing_data_df)

    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        display(strategy.metric_summary_df)
        display(strategy.annual_return_df.tail(10))
        display(strategy.summary)

    if save_results_bool:
        output_path = save_results(strategy, output_dir=output_dir_str)
        strategy.metric_summary_df.to_csv(output_path / "metric_summary.csv", index=False, float_format="%.10f")
        strategy.annual_return_df.to_csv(output_path / "annual_returns.csv", float_format="%.10f")
        strategy.signal_state_df.to_csv(output_path / "signal_state.csv", float_format="%.10f")
        strategy.daily_target_weight_df.to_csv(output_path / "daily_target_weights.csv", float_format="%.10f")
        strategy.realized_weight_df.to_csv(output_path / "realized_weights.csv", float_format="%.10f")
        _write_assumptions_md(output_path=output_path, config=config)

    return strategy


if __name__ == "__main__":
    run_variant(show_display_bool=True, save_results_bool=True)
