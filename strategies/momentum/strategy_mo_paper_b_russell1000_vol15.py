"""
Research-only Russell 1000 Paper-B momentum with 15% volatility sizing.

This is a controlled universe/risk-target variant of the Russell 3000 Paper-B
strategy. It preserves the same signal, filters, 50-long/50-short equal-weight
books, monthly next-open execution, unscaled-base volatility estimator, repo
default costs, and reduction-only exposure cap.

Signal audit is intentionally disabled for both Vanilla passes. The signal and
selection tables are computed once from completed month-end information before
the engine runs.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from strategies.momentum import strategy_mo_paper_b_russell3000_vol10 as paper_b_base


STRATEGY_NAME_STR = "strategy_mo_paper_b_russell1000_vol15"
AUDIT_ENABLED_BOOL = False

DEFAULT_CONFIG = replace(
    paper_b_base.DEFAULT_CONFIG,
    variant_key_str="paper_b_russell1000_top50_bottom50_vol15",
    indexname_str="Russell 1000",
    benchmark_list=("$RUI",),
    max_long_positions_int=50,
    max_short_positions_int=50,
    target_annualized_volatility_float=0.15,
)


class PaperBRussell1000Vol15Strategy(paper_b_base.PaperBRussell3000Strategy):
    """Russell 1000 / 15% Paper-B variant executed by Vanilla."""

    enable_signal_audit = AUDIT_ENABLED_BOOL


def get_paper_b_russell1000_data(
    config: paper_b_base.PaperBRussell3000Config = DEFAULT_CONFIG,
    pricing_data_df: pd.DataFrame | None = None,
    universe_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the PIT Russell 1000 data through the shared Paper-B pipeline."""
    return paper_b_base.get_paper_b_russell3000_data(
        config=config,
        pricing_data_df=pricing_data_df,
        universe_df=universe_df,
    )


def _write_assumptions_md(
    output_path_obj: Path,
    strategy_obj: PaperBRussell1000Vol15Strategy,
) -> None:
    config_obj = strategy_obj.config
    assumption_md_str = f"""# Paper-B Russell 1000 Volatility-15 Assumptions

- Research-only strategy; no live or release wiring.
- Universe: `{config_obj.indexname_str}` historical point-in-time membership.
- Signal price basis: repo-required Norgate `CAPITALSPECIAL` stock close. The repository forbids `TOTALRETURN` for individual stocks because of forward-looking dividend bias.
- Eligibility price: Norgate `Unadjusted Close >= {config_obj.minimum_unadjusted_close_float:.2f}` at the completed decision close.
- Liquidity: trailing `{config_obj.adv_lookback_day_int}`-session mean of Norgate raw `Turnover >= {config_obj.minimum_adv_dollar_float:.2f}`.
- Signal: `M_t = P_(t-2) / P_(t-13) - 1`, `r_t = P_(t-1) / P_(t-2) - 1`, `B_t = (1 + r_t) * M_t`.
- Selection: top `{config_obj.max_long_positions_int}` B scores long and bottom `{config_obj.max_short_positions_int}` short; equal weight inside each side.
- Base target: `+100%` long and `-100%` short, `200%` gross, approximately dollar-neutral after integer-share and next-open effects.
- Volatility input: the separate unscaled base Vanilla path, including repository-default trading costs and excluding all prior volatility multipliers.
- Volatility estimate: sample standard deviation of exactly `{config_obj.volatility_lookback_month_int}` completed calendar-month base returns times `sqrt(12)`.
- Exposure: `min({config_obj.maximum_exposure_multiplier_float:.2f}, {config_obj.target_annualized_volatility_float:.2%} / annualized_base_volatility)`.
- Warm-up: exposure is zero until exactly `{config_obj.volatility_lookback_month_int}` base returns exist; all warm-up months are excluded from the saved performance report.
- Decision: actual final tradable close of each completed month. Execution: next tradable open under the Vanilla engine.
- Signal audit: hard-disabled for both the unscaled and scaled Vanilla passes.
- Cash earns zero.
- Repository-default costs: slippage `{float(strategy_obj._slippage):.6f}` per order and commission `max({float(strategy_obj._commission_minimum):.2f}, {float(strategy_obj._commission_per_share):.6f} * abs(shares))`.
- Stale held symbols follow the engine's documented G-014 fallback: force-liquidation at the last available prior close. This is conservative bookkeeping, not exact corporate-action replay.
- Borrow cost, locate availability, recalls, financing, market impact, and partial fills are not modeled.
"""
    (output_path_obj / "paper_b_assumptions.md").write_text(
        assumption_md_str,
        encoding="utf-8",
    )


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
    pricing_data_df: pd.DataFrame | None = None,
    universe_df: pd.DataFrame | None = None,
) -> PaperBRussell1000Vol15Strategy:
    config_obj = DEFAULT_CONFIG
    if (
        backtest_start_date_str is not None
        or capital_base_float is not None
        or end_date_str is not None
    ):
        config_obj = replace(
            DEFAULT_CONFIG,
            backtest_start_date_str=(
                DEFAULT_CONFIG.backtest_start_date_str
                if backtest_start_date_str is None
                else backtest_start_date_str
            ),
            capital_base_float=(
                DEFAULT_CONFIG.capital_base_float
                if capital_base_float is None
                else float(capital_base_float)
            ),
            end_date_str=end_date_str,
        )

    (
        loaded_pricing_data_df,
        _audited_universe_df,
        rebalance_schedule_df,
        selection_df,
    ) = get_paper_b_russell1000_data(
        config=config_obj,
        pricing_data_df=pricing_data_df,
        universe_df=universe_df,
    )

    base_exposure_schedule_df = pd.DataFrame(
        {
            "decision_date_ts": rebalance_schedule_df["decision_date_ts"],
            "exposure_multiplier_float": 1.0,
        },
        index=rebalance_schedule_df.index,
    )
    base_strategy_obj = PaperBRussell1000Vol15Strategy(
        name=f"{STRATEGY_NAME_STR}_unscaled_base",
        benchmarks=list(config_obj.benchmark_list),
        rebalance_schedule_df=rebalance_schedule_df,
        selection_df=selection_df,
        exposure_schedule_df=base_exposure_schedule_df,
        config=config_obj,
    )
    base_calendar_idx = paper_b_base._get_base_calendar_idx(
        pricing_date_idx=pd.DatetimeIndex(loaded_pricing_data_df.index),
        rebalance_schedule_df=rebalance_schedule_df,
    )

    # *** CRITICAL*** This hidden pass is always the unscaled +100%/-100%
    # portfolio. Its returns never inherit a prior exposure multiplier.
    run_daily(
        base_strategy_obj,
        loaded_pricing_data_df,
        calendar=base_calendar_idx,
        show_progress=show_display_bool,
        show_signal_progress_bool=False,
        audit_override_bool=AUDIT_ENABLED_BOOL,
    )
    base_monthly_return_ser = paper_b_base.compound_daily_returns_to_calendar_month_ser(
        base_strategy_obj.results["daily_returns"]
    )
    exposure_schedule_df = paper_b_base.build_exposure_schedule_df(
        base_monthly_return_ser=base_monthly_return_ser,
        rebalance_schedule_df=rebalance_schedule_df,
        config=config_obj,
    )
    reportable_exposure_df = exposure_schedule_df.loc[
        exposure_schedule_df["warmup_complete_bool"]
    ]
    if len(reportable_exposure_df) == 0:
        raise RuntimeError(
            "The requested window does not contain 12 completed unscaled base returns."
        )
    reported_start_date_ts = pd.Timestamp(reportable_exposure_df.index[0])

    strategy_obj = PaperBRussell1000Vol15Strategy(
        name=STRATEGY_NAME_STR,
        benchmarks=list(config_obj.benchmark_list),
        rebalance_schedule_df=rebalance_schedule_df,
        selection_df=selection_df,
        exposure_schedule_df=exposure_schedule_df,
        config=config_obj,
    )
    strategy_obj.base_monthly_return_ser = base_monthly_return_ser.copy()
    strategy_obj.reported_start_date_ts = reported_start_date_ts

    # *** CRITICAL*** Starting the reported engine calendar here keeps all 12
    # warm-up months out of every saved metric and benchmark comparison.
    reported_calendar_idx = loaded_pricing_data_df.index[
        loaded_pricing_data_df.index >= reported_start_date_ts
    ]
    run_daily(
        strategy_obj,
        loaded_pricing_data_df,
        calendar=reported_calendar_idx,
        show_progress=show_display_bool,
        show_signal_progress_bool=False,
        audit_override_bool=AUDIT_ENABLED_BOOL,
    )

    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        display(strategy_obj.summary)
        display(strategy_obj.summary_trades)
        display(exposure_schedule_df.tail(24))
        display(strategy_obj.rebalance_execution_df.tail(20))

    if save_results_bool:
        output_path_obj = save_results(strategy_obj, output_dir=output_dir_str)
        strategy_obj.output_path_obj = output_path_obj
        strategy_obj.rebalance_execution_df.to_csv(
            output_path_obj / "rebalance_selection.csv",
            index=False,
        )
        exposure_schedule_df.to_csv(output_path_obj / "exposure_schedule.csv")
        base_monthly_return_ser.to_csv(output_path_obj / "base_monthly_returns.csv")
        _write_assumptions_md(output_path_obj=output_path_obj, strategy_obj=strategy_obj)

    return strategy_obj


__all__ = [
    "AUDIT_ENABLED_BOOL",
    "DEFAULT_CONFIG",
    "PaperBRussell1000Vol15Strategy",
    "get_paper_b_russell1000_data",
    "run_variant",
]


if __name__ == "__main__":
    run_variant()
