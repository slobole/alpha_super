"""
Research-only US sector ETF IBS Downshock variant using VOX and IYR.

Fixed basket:

    XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY, VOX, IYR

This is a controlled historical-proxy variant of
`strategy_mr_us_sector_etf_ibs_downshock`:

- VOX replaces XLC;
- IYR replaces XLRE;
- every signal, timing, sizing, slot, cost, and cash rule stays unchanged.

VOX and IYR are not treated as synthetic backfilled XLC/XLRE histories. They
remain distinct tradable ETFs whose older sector definitions and breadth can
differ from the modern Sector SPDR exposures.
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
    UsSectorEtfIbsDownshockConfig,
    UsSectorEtfIbsDownshockStrategy,
    _write_assumptions_md,
    get_us_sector_etf_ibs_downshock_data,
    resolve_us_sector_etf_execution_calendar_idx,
)


STRATEGY_NAME_STR = "strategy_mr_us_sector_etf_ibs_downshock_vox_iyr"
VOX_IYR_SYMBOL_TUPLE = (
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLU",
    "XLV",
    "XLY",
    "VOX",
    "IYR",
)
DEFAULT_CONFIG = replace(
    BASE_DEFAULT_CONFIG,
    symbol_tuple=VOX_IYR_SYMBOL_TUPLE,
    history_start_date_str="1998-01-01",
    backtest_start_date_str="1999-01-01",
)


class UsSectorEtfIbsDownshockVoxIyrStrategy(UsSectorEtfIbsDownshockStrategy):
    """The base strategy using the requested VOX/IYR proxy basket."""

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
) -> UsSectorEtfIbsDownshockVoxIyrStrategy:
    strategy_obj = UsSectorEtfIbsDownshockVoxIyrStrategy(
        name=STRATEGY_NAME_STR,
        benchmarks=[config_obj.benchmark_symbol_str],
        config_obj=config_obj,
    )
    # *** CRITICAL*** Proxy selection changes only the fixed tradable basket.
    # The first fill still occurs one session after every ETF has a completed
    # causal warm-up, and every later signal remains Close_T -> Open_(T+1).
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


def _write_vox_iyr_notes_md(output_path: Path) -> None:
    notes_md_str = f"""# US Sector ETF IBS Downshock VOX/IYR Variant

- Controlled proxy-universe variant of `strategy_mr_us_sector_etf_ibs_downshock`.
- Fixed basket: `{", ".join(VOX_IYR_SYMBOL_TUPLE)}`.
- VOX replaces XLC and IYR replaces XLRE as independently traded ETFs.
- VOX is not a definitionally exact pre-2018 XLC history.
- IYR is broader than the Sector SPDR XLRE exposure.
- Signals, timing, five slots, `1.5 / 11` sizing, costs, and cash accounting are unchanged.
- No price series are spliced, extended, or synthetically backfilled.
- Research-only; no LIVE or release wiring.
"""
    (output_path / "us_sector_etf_ibs_downshock_vox_iyr_notes.md").write_text(
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
) -> UsSectorEtfIbsDownshockVoxIyrStrategy:
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
        _write_vox_iyr_notes_md(output_path)
    return strategy_obj


__all__ = [
    "DEFAULT_CONFIG",
    "STRATEGY_NAME_STR",
    "UsSectorEtfIbsDownshockVoxIyrStrategy",
    "VOX_IYR_SYMBOL_TUPLE",
    "build_capacity_analysis_inputs",
    "run_variant",
]


if __name__ == "__main__":
    run_variant()
