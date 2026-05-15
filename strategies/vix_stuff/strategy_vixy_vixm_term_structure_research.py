"""
VIXY / VIXM fixed-beta term-structure spread research replication.

TL;DR: This is an article-faithful research path for the ValueIytica
"Volatility Term Structure Arbitrage" idea. It keeps a fixed-beta spread:

    short VIXY
    long  VIXM

and sweeps gross allocation from 20% to 50%.

This module is intentionally not live-clean. The article assumes a constant
hedge beta of 2, derived from the full sample. That is a lookahead-biased
research assumption, so this file is kept separate from live/deployment
surfaces.

Core formulas
-------------
For allocation A and fixed beta b:

    w_vixy = -A
    w_vixm =  bA
    w_cash =  1 - A

At close-to-close day t:

    r_t^{spread}
        = w_vixy * r_t^{VIXY}
        + w_vixm * r_t^{VIXM}

    E_t = E_{t-1} * (1 + r_t^{spread})

The implementation is before transaction costs, borrow fees, margin costs,
slippage, and locate/recall constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd
from IPython.display import display

WORKSPACE_ROOT_PATH = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT_PATH))

from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from data.norgate_loader import load_raw_prices
from strategies.eom_tlt_vs_spy.strategy_eom_trend_ibit import build_results_df


ALLOCATION_LIST: tuple[float, ...] = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)


def validate_allocation_float(allocation_float: float) -> None:
    if not np.isfinite(allocation_float) or allocation_float <= 0.0:
        raise ValueError("allocation_float must be positive and finite.")
    if allocation_float > 1.0:
        raise ValueError("allocation_float must be <= 1.0 because the article leaves the rest in cash.")


@dataclass(frozen=True)
class VixyVixmTermStructureResearchConfig:
    short_symbol_str: str = "VIXY"
    hedge_symbol_str: str = "VIXM"
    benchmark_list: tuple[str, ...] = ("SPY",)
    allocation_tuple: tuple[float, ...] = ALLOCATION_LIST
    primary_allocation_float: float = 0.50
    fixed_beta_float: float = 2.0
    start_date_str: str = "2019-01-01"
    end_date_str: str | None = "2025-12-31"
    capital_base_float: float = 100_000.0

    def __post_init__(self):
        if not self.short_symbol_str or not self.hedge_symbol_str:
            raise ValueError("short_symbol_str and hedge_symbol_str must be non-empty.")
        if self.short_symbol_str == self.hedge_symbol_str:
            raise ValueError("short_symbol_str and hedge_symbol_str must differ.")
        if self.fixed_beta_float <= 0.0 or not np.isfinite(self.fixed_beta_float):
            raise ValueError("fixed_beta_float must be positive and finite.")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if len(self.allocation_tuple) == 0:
            raise ValueError("allocation_tuple must not be empty.")
        for allocation_float in self.allocation_tuple:
            validate_allocation_float(float(allocation_float))
            if self.fixed_beta_float * float(allocation_float) > 1.0:
                raise ValueError("fixed_beta_float * allocation_float must be <= 1.0 for this article replication.")
        validate_allocation_float(float(self.primary_allocation_float))
        if self.fixed_beta_float * float(self.primary_allocation_float) > 1.0:
            raise ValueError("fixed_beta_float * primary_allocation_float must be <= 1.0 for this article replication.")
        if self.primary_allocation_float not in self.allocation_tuple:
            raise ValueError("primary_allocation_float must be included in allocation_tuple.")


DEFAULT_CONFIG = VixyVixmTermStructureResearchConfig()


def get_vixy_vixm_term_structure_research_data(
    config: VixyVixmTermStructureResearchConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    return load_raw_prices(
        symbols=[config.short_symbol_str, config.hedge_symbol_str],
        benchmarks=list(config.benchmark_list),
        start_date=config.start_date_str,
        end_date=config.end_date_str,
    )


def compute_fixed_beta_weight_ser(
    allocation_float: float,
    fixed_beta_float: float,
    short_symbol_str: str,
    hedge_symbol_str: str,
) -> pd.Series:
    """
    Convert short-leg allocation and fixed beta into article-style signed weights.

        |w_vixy| = A
        |w_vixm| = beta * A

    In the article table, the reported volatility lines up with this short-leg
    allocation convention. With beta=2 and A=50%, gross exposure is 150%.
    """
    validate_allocation_float(float(allocation_float))
    if not np.isfinite(fixed_beta_float) or fixed_beta_float <= 0.0:
        raise ValueError("fixed_beta_float must be positive and finite.")
    if fixed_beta_float * allocation_float > 1.0:
        raise ValueError("fixed_beta_float * allocation_float must be <= 1.0 for this article replication.")

    short_abs_weight_float = float(allocation_float)
    hedge_abs_weight_float = float(fixed_beta_float) * short_abs_weight_float
    return pd.Series(
        {
            short_symbol_str: -short_abs_weight_float,
            hedge_symbol_str: hedge_abs_weight_float,
            "CashReserve": 1.0 - float(allocation_float),
        },
        dtype=float,
        name="target_weight_ser",
    )


def compute_close_to_close_return_df(
    pricing_data_df: pd.DataFrame,
    short_symbol_str: str,
    hedge_symbol_str: str,
) -> pd.DataFrame:
    required_key_list = [
        (short_symbol_str, "Close"),
        (hedge_symbol_str, "Close"),
    ]
    missing_key_list = [key_tup for key_tup in required_key_list if key_tup not in pricing_data_df.columns]
    if len(missing_key_list) > 0:
        raise RuntimeError(f"Missing required close columns: {missing_key_list}")

    close_price_df = pricing_data_df.xs("Close", axis=1, level=1)[[short_symbol_str, hedge_symbol_str]].astype(float)
    close_price_df = close_price_df.dropna(how="any")
    if len(close_price_df) < 2:
        raise RuntimeError("At least two shared close bars are required.")
    if (close_price_df <= 0.0).any().any():
        raise RuntimeError("Close prices must be strictly positive.")

    # *** CRITICAL*** article-faithful close-to-close research path:
    # daily returns use Close_t / Close_{t-1} - 1 and are not mapped through
    # the repo's next-open execution engine. This is a research replication,
    # not a live-tradable execution model.
    simple_return_df = close_price_df.pct_change(fill_method=None).fillna(0.0)

    # *** CRITICAL*** lookahead-sensitive diagnostic only: the article's beta
    # discussion is based on the full-sample VIXY/VIXM log-return relationship.
    # These log returns must not be used to form live hedge ratios.
    log_return_df = np.log(close_price_df / close_price_df.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return pd.DataFrame(
        {
            "vixy_simple_return_ser": simple_return_df[short_symbol_str],
            "vixm_simple_return_ser": simple_return_df[hedge_symbol_str],
            "vixy_log_return_ser": log_return_df[short_symbol_str],
            "vixm_log_return_ser": log_return_df[hedge_symbol_str],
        },
        index=close_price_df.index,
    )


def compute_full_sample_log_beta_float(close_to_close_return_df: pd.DataFrame) -> float:
    """
    Estimate the full-sample log-return beta used only as a diagnostic.

        beta = Cov(log_ret_vixy, log_ret_vixm) / Var(log_ret_vixm)
    """
    log_return_df = close_to_close_return_df[["vixy_log_return_ser", "vixm_log_return_ser"]].iloc[1:].dropna()
    if len(log_return_df) < 2:
        return np.nan
    hedge_var_float = float(log_return_df["vixm_log_return_ser"].var(ddof=1))
    if hedge_var_float == 0.0 or not np.isfinite(hedge_var_float):
        return np.nan
    return float(log_return_df["vixy_log_return_ser"].cov(log_return_df["vixm_log_return_ser"]) / hedge_var_float)


def compute_spread_daily_return_ser(
    close_to_close_return_df: pd.DataFrame,
    target_weight_ser: pd.Series,
    short_symbol_str: str,
    hedge_symbol_str: str,
) -> pd.Series:
    short_weight_float = float(target_weight_ser.loc[short_symbol_str])
    hedge_weight_float = float(target_weight_ser.loc[hedge_symbol_str])
    spread_return_ser = (
        short_weight_float * close_to_close_return_df["vixy_simple_return_ser"].astype(float)
        + hedge_weight_float * close_to_close_return_df["vixm_simple_return_ser"].astype(float)
    )
    return spread_return_ser.astype(float).rename("spread_daily_return_ser")


def compute_summary_metric_dict(
    daily_return_ser: pd.Series,
    capital_base_float: float,
) -> dict[str, float]:
    total_value_ser = float(capital_base_float) * (1.0 + daily_return_ser.astype(float)).cumprod()
    elapsed_year_float = max((total_value_ser.index[-1] - total_value_ser.index[0]).days / 365.25, 1.0 / 365.25)
    cagr_float = float((float(total_value_ser.iloc[-1]) / float(total_value_ser.iloc[0])) ** (1.0 / elapsed_year_float) - 1.0)
    annual_volatility_float = float(daily_return_ser.std(ddof=1) * np.sqrt(252.0))
    if annual_volatility_float == 0.0 or not np.isfinite(annual_volatility_float):
        sharpe_float = np.nan
    else:
        sharpe_float = float(daily_return_ser.mean() / daily_return_ser.std(ddof=1) * np.sqrt(252.0))
    drawdown_ser = total_value_ser / total_value_ser.cummax() - 1.0
    return {
        "cagr_float": cagr_float,
        "annual_volatility_float": annual_volatility_float,
        "sharpe_float": sharpe_float,
        "max_drawdown_float": float(drawdown_ser.min()),
        "final_equity_float": float(total_value_ser.iloc[-1]),
    }


def build_allocation_summary_df(
    close_to_close_return_df: pd.DataFrame,
    config: VixyVixmTermStructureResearchConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    row_dict_list: list[dict[str, float]] = []
    diagnostic_beta_float = compute_full_sample_log_beta_float(close_to_close_return_df)

    for allocation_float in config.allocation_tuple:
        target_weight_ser = compute_fixed_beta_weight_ser(
            allocation_float=float(allocation_float),
            fixed_beta_float=config.fixed_beta_float,
            short_symbol_str=config.short_symbol_str,
            hedge_symbol_str=config.hedge_symbol_str,
        )
        daily_return_ser = compute_spread_daily_return_ser(
            close_to_close_return_df=close_to_close_return_df,
            target_weight_ser=target_weight_ser,
            short_symbol_str=config.short_symbol_str,
            hedge_symbol_str=config.hedge_symbol_str,
        )
        metric_dict = compute_summary_metric_dict(
            daily_return_ser=daily_return_ser,
            capital_base_float=config.capital_base_float,
        )
        row_dict_list.append(
            {
                "allocation_float": float(allocation_float),
                "fixed_beta_float": float(config.fixed_beta_float),
                "full_sample_log_beta_diagnostic_float": float(diagnostic_beta_float),
                "vixy_weight_float": float(target_weight_ser.loc[config.short_symbol_str]),
                "vixm_weight_float": float(target_weight_ser.loc[config.hedge_symbol_str]),
                "cash_reserve_weight_float": float(target_weight_ser.loc["CashReserve"]),
                "gross_exposure_float": float(
                    abs(target_weight_ser.loc[config.short_symbol_str])
                    + abs(target_weight_ser.loc[config.hedge_symbol_str])
                ),
                "net_market_weight_float": float(
                    target_weight_ser.loc[config.short_symbol_str]
                    + target_weight_ser.loc[config.hedge_symbol_str]
                ),
                **metric_dict,
            }
        )

    return pd.DataFrame(row_dict_list).set_index("allocation_float")


class VixyVixmTermStructureResearchStrategy(Strategy):
    """
    Research-only Strategy wrapper around the vectorized close-to-close spread.
    """

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        config: VixyVixmTermStructureResearchConfig,
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
        self.close_to_close_return_df = pd.DataFrame()
        self.allocation_summary_df = pd.DataFrame()
        self.daily_target_weight_df = pd.DataFrame()

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        return pricing_data_df

    def iterate(self, data_df: pd.DataFrame, close_row_ser: pd.Series, open_price_ser: pd.Series):
        return


def run_article_research_backtest(
    strategy: VixyVixmTermStructureResearchStrategy,
    pricing_data_df: pd.DataFrame,
) -> VixyVixmTermStructureResearchStrategy:
    close_to_close_return_df = compute_close_to_close_return_df(
        pricing_data_df=pricing_data_df,
        short_symbol_str=strategy.config.short_symbol_str,
        hedge_symbol_str=strategy.config.hedge_symbol_str,
    )
    allocation_summary_df = build_allocation_summary_df(
        close_to_close_return_df=close_to_close_return_df,
        config=strategy.config,
    )
    primary_weight_ser = compute_fixed_beta_weight_ser(
        allocation_float=strategy.config.primary_allocation_float,
        fixed_beta_float=strategy.config.fixed_beta_float,
        short_symbol_str=strategy.config.short_symbol_str,
        hedge_symbol_str=strategy.config.hedge_symbol_str,
    )
    daily_return_ser = compute_spread_daily_return_ser(
        close_to_close_return_df=close_to_close_return_df,
        target_weight_ser=primary_weight_ser,
        short_symbol_str=strategy.config.short_symbol_str,
        hedge_symbol_str=strategy.config.hedge_symbol_str,
    )

    total_value_ser = float(strategy.config.capital_base_float) * (1.0 + daily_return_ser).cumprod()
    net_market_weight_float = float(primary_weight_ser.loc[strategy.config.short_symbol_str]) + float(
        primary_weight_ser.loc[strategy.config.hedge_symbol_str]
    )

    # The report's portfolio/cash fields are used here as article-style gross
    # spread state, not as a broker-level short-sale cash ledger.
    portfolio_value_ser = total_value_ser * net_market_weight_float
    cash_ser = total_value_ser - portfolio_value_ser

    benchmark_equity_map: dict[str, pd.Series] = {}
    for benchmark_str in strategy.config.benchmark_list:
        benchmark_key_tup = (benchmark_str, "Close")
        if benchmark_key_tup not in pricing_data_df.columns:
            raise RuntimeError(f"Missing benchmark close column: {benchmark_key_tup}")
        benchmark_close_ser = pricing_data_df.loc[total_value_ser.index, benchmark_key_tup].astype(float)
        benchmark_equity_map[benchmark_str] = benchmark_close_ser / float(benchmark_close_ser.iloc[0]) * float(
            strategy.config.capital_base_float
        )

    strategy.results = build_results_df(
        total_value_ser=total_value_ser,
        portfolio_value_ser=portfolio_value_ser,
        cash_ser=cash_ser,
        benchmark_equity_map=benchmark_equity_map,
    )
    strategy.close_to_close_return_df = close_to_close_return_df
    strategy.allocation_summary_df = allocation_summary_df
    strategy.daily_target_weight_df = pd.DataFrame(
        {
            strategy.config.short_symbol_str: float(primary_weight_ser.loc[strategy.config.short_symbol_str]),
            strategy.config.hedge_symbol_str: float(primary_weight_ser.loc[strategy.config.hedge_symbol_str]),
            "CashReserve": float(primary_weight_ser.loc["CashReserve"]),
        },
        index=total_value_ser.index,
    )
    strategy.realized_weight_df = strategy.daily_target_weight_df.copy()
    strategy._transactions = strategy.initialize_transactions()
    strategy.cash = float(cash_ser.iloc[-1])
    strategy.portfolio_value = float(portfolio_value_ser.iloc[-1])
    strategy.total_value = float(total_value_ser.iloc[-1])
    strategy.current_bar = pd.Timestamp(total_value_ser.index[-1])
    strategy._latest_close_price_ser = pricing_data_df.xs("Close", axis=1, level=1).iloc[-1].astype(float)
    strategy.summarize()
    # This vectorized research path has no order ledger, but it is allocated
    # every day in the article model. The engine's trade-derived exposure
    # metric would otherwise report 0% because there are no transaction rows.
    if strategy.summary is not None and "Strategy" in strategy.summary.columns:
        strategy.summary.loc["Exposure Time [%]", "Strategy"] = 100.0
        strategy.summary.loc["Exposure-Adjusted Return (Ann.) [%]", "Strategy"] = strategy.summary.loc[
            "Return (Ann.) [%]",
            "Strategy",
        ]
    return strategy


def run_variant(
    show_display_bool: bool = False,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    end_date_str: str | None = None,
    capital_base_float: float | None = None,
) -> VixyVixmTermStructureResearchStrategy:
    config = DEFAULT_CONFIG
    if (
        backtest_start_date_str is not None
        or end_date_str is not None
        or capital_base_float is not None
    ):
        config = VixyVixmTermStructureResearchConfig(
            start_date_str=backtest_start_date_str or DEFAULT_CONFIG.start_date_str,
            end_date_str=end_date_str if end_date_str is not None else DEFAULT_CONFIG.end_date_str,
            capital_base_float=capital_base_float if capital_base_float is not None else DEFAULT_CONFIG.capital_base_float,
        )

    pricing_data_df = get_vixy_vixm_term_structure_research_data(config=config)
    strategy = VixyVixmTermStructureResearchStrategy(
        name="strategy_vixy_vixm_term_structure_research",
        benchmarks=config.benchmark_list,
        config=config,
        capital_base=config.capital_base_float,
    )
    run_article_research_backtest(
        strategy=strategy,
        pricing_data_df=pricing_data_df,
    )

    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        print("Allocation sweep, before costs:")
        display(strategy.allocation_summary_df)
        display(strategy.summary)

    if save_results_bool:
        output_path = save_results(strategy, output_dir=output_dir_str)
        strategy.allocation_summary_df.to_csv(output_path / "allocation_summary.csv", float_format="%.10f")
        strategy.daily_target_weight_df.to_csv(output_path / "daily_target_weights.csv", float_format="%.10f")

    return strategy


if __name__ == "__main__":
    run_variant(show_display_bool=True, save_results_bool=True)
