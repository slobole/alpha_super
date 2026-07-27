"""
Research-only US sector ETF IBS Downshock variant without XLC.

This is a controlled universe variant of
`strategy_mr_us_sector_etf_ibs_downshock`. It removes XLC so the backtest can
begin after XLRE's 2015 inception instead of XLC's 2018 inception.

Everything else stays unchanged:

- completed Close_T signal -> Open_(T+1) execution;
- entry, exit, ATR/NATR, and ranking formulas;
- five-position limit;
- 1.5 / 11 sizing per new position;
- existing positions are not rebalanced;
- engine-default costs and cash accounting.

No historical proxy is used because pre-2018 telecom ETFs are not equivalent
to the post-2018 Communication Services sector represented by XLC.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from strategies.mean_reversion.strategy_mr_sector_dispersion_ibs import (
    resolve_effective_backtest_start_date_str,
    resolve_history_start_date_str,
)
from strategies.mean_reversion.strategy_mr_us_sector_etf_ibs_downshock import (
    DEFAULT_CONFIG as BASE_DEFAULT_CONFIG,
    SECTOR_ETF_SYMBOL_TUPLE,
    UsSectorEtfIbsDownshockConfig,
    UsSectorEtfIbsDownshockStrategy,
    _write_assumptions_md,
    get_us_sector_etf_ibs_downshock_data,
    resolve_us_sector_etf_execution_calendar_idx,
)


STRATEGY_NAME_STR = "strategy_mr_us_sector_etf_ibs_downshock_no_xlc"
NO_XLC_SYMBOL_TUPLE = tuple(
    symbol_str
    for symbol_str in SECTOR_ETF_SYMBOL_TUPLE
    if symbol_str != "XLC"
)
DEFAULT_CONFIG = replace(
    BASE_DEFAULT_CONFIG,
    symbol_tuple=NO_XLC_SYMBOL_TUPLE,
    history_start_date_str="1998-01-01",
    backtest_start_date_str="1999-01-01",
)


class UsSectorEtfIbsDownshockNoXlcStrategy(UsSectorEtfIbsDownshockStrategy):
    """The base strategy with XLC removed and all other semantics unchanged."""

    def __init__(
        self,
        name: str,
        benchmarks: list[str] | tuple[str, ...],
        config_obj: UsSectorEtfIbsDownshockConfig = DEFAULT_CONFIG,
    ):
        super().__init__(
            name=name,
            benchmarks=benchmarks,
            config_obj=config_obj,
        )


def _run_strategy(
    config_obj: UsSectorEtfIbsDownshockConfig,
    pricing_data_df: pd.DataFrame,
    show_display_bool: bool,
    audit_override_bool: bool | None,
) -> UsSectorEtfIbsDownshockNoXlcStrategy:
    strategy_obj = UsSectorEtfIbsDownshockNoXlcStrategy(
        name=STRATEGY_NAME_STR,
        benchmarks=[config_obj.benchmark_symbol_str],
        config_obj=config_obj,
    )
    # *** CRITICAL*** The first executable open follows the first completed
    # full-basket-ready decision close. Removing XLC changes only when that
    # readiness occurs; it does not change Close_T -> Open_(T+1) timing.
    calendar_idx = resolve_us_sector_etf_execution_calendar_idx(
        pricing_data_df=pricing_data_df,
        config_obj=config_obj,
    )
    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar_idx,
        show_progress=show_display_bool,
        show_signal_progress_bool=show_display_bool,
        audit_override_bool=audit_override_bool,
    )
    return strategy_obj


def build_capacity_analysis_inputs(
    capital_base_float: float,
    show_display_bool: bool = False,
    backtest_start_date_str: str | None = None,
    end_date_str: str | None = None,
) -> dict[str, object]:
    effective_backtest_start_date_str = resolve_effective_backtest_start_date_str(
        config_obj=DEFAULT_CONFIG,
        requested_backtest_start_date_str=backtest_start_date_str,
    )
    config_obj = replace(
        DEFAULT_CONFIG,
        history_start_date_str=resolve_history_start_date_str(
            config_obj=DEFAULT_CONFIG,
            backtest_start_date_str=effective_backtest_start_date_str,
        ),
        backtest_start_date_str=effective_backtest_start_date_str,
        capital_base_float=float(capital_base_float),
        end_date_str=end_date_str,
    )
    pricing_data_df = get_us_sector_etf_ibs_downshock_data(config_obj)
    strategy_obj = _run_strategy(
        config_obj=config_obj,
        pricing_data_df=pricing_data_df,
        show_display_bool=show_display_bool,
        audit_override_bool=None,
    )
    return {
        "strategy_obj": strategy_obj,
        "pricing_data_df": pricing_data_df,
        "execution_policy_str": "MOO",
        "impact_profile_str": "MOO_ETF_PROXY",
    }


def _write_no_xlc_notes_md(output_path: Path) -> None:
    notes_md_str = f"""# US Sector ETF IBS Downshock No-XLC Variant

- Controlled universe variant of `strategy_mr_us_sector_etf_ibs_downshock`.
- Fixed basket: `{", ".join(NO_XLC_SYMBOL_TUPLE)}`.
- XLC is omitted; no telecom or communication-services proxy is spliced in.
- XLRE is now the latest-inception ETF and therefore sets the real-data start.
- Position sizing remains `1.5 / 11`, not `1.5 / 10`, so removing XLC does not silently increase risk per trade.
- Entry, exit, ranking, slots, execution timing, costs, and cash accounting are unchanged.
- Research-only; no LIVE or release wiring.
"""
    (output_path / "us_sector_etf_ibs_downshock_no_xlc_notes.md").write_text(
        notes_md_str,
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
    audit_override_bool: bool | None = None,
) -> UsSectorEtfIbsDownshockNoXlcStrategy:
    effective_backtest_start_date_str = resolve_effective_backtest_start_date_str(
        config_obj=DEFAULT_CONFIG,
        requested_backtest_start_date_str=backtest_start_date_str,
    )
    config_obj = replace(
        DEFAULT_CONFIG,
        history_start_date_str=resolve_history_start_date_str(
            config_obj=DEFAULT_CONFIG,
            backtest_start_date_str=effective_backtest_start_date_str,
        ),
        backtest_start_date_str=effective_backtest_start_date_str,
        capital_base_float=(
            DEFAULT_CONFIG.capital_base_float
            if capital_base_float is None
            else float(capital_base_float)
        ),
        end_date_str=end_date_str,
    )
    if pricing_data_df is None:
        pricing_data_df = get_us_sector_etf_ibs_downshock_data(config_obj)

    strategy_obj = _run_strategy(
        config_obj=config_obj,
        pricing_data_df=pricing_data_df,
        show_display_bool=show_display_bool,
        audit_override_bool=audit_override_bool,
    )

    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        display(strategy_obj.summary)
        display(strategy_obj.summary_trades)

    if save_results_bool:
        output_path = save_results(strategy_obj, output_dir=output_dir_str)
        _write_assumptions_md(
            output_path=output_path,
            strategy_obj=strategy_obj,
        )
        _write_no_xlc_notes_md(output_path)
    return strategy_obj


__all__ = [
    "DEFAULT_CONFIG",
    "NO_XLC_SYMBOL_TUPLE",
    "STRATEGY_NAME_STR",
    "UsSectorEtfIbsDownshockNoXlcStrategy",
    "build_capacity_analysis_inputs",
    "run_variant",
]


if __name__ == "__main__":
    run_variant()
