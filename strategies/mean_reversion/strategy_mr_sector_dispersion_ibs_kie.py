"""
Research-only sector-dispersion IBS variant with KIE added.

This is the runnable strategy surface for the balanced recommendation from the
marginal universe and stress studies:

    SOXX, IGV, IBB, KIE

It intentionally reuses the base sector-dispersion IBS implementation so the
signal, sizing, costs, and next-open execution semantics stay identical.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from strategies.mean_reversion.strategy_mr_sector_dispersion_ibs import (
    DEFAULT_CONFIG as BASE_DEFAULT_CONFIG,
    ORIGINAL_SYMBOL_TUPLE,
    SectorDispersionIbsConfig,
    SectorDispersionIbsStrategy,
    _write_assumptions_md,
    get_sector_dispersion_ibs_data,
    resolve_history_start_date_str,
)


STRATEGY_NAME_STR = "strategy_mr_sector_dispersion_ibs_kie"
STRATEGY_SYMBOL_TUPLE = ORIGINAL_SYMBOL_TUPLE + ("KIE",)


class SectorDispersionIbsKieStrategy(SectorDispersionIbsStrategy):
    """Metadata wrapper for the fixed KIE basket variant."""


DEFAULT_CONFIG = replace(
    BASE_DEFAULT_CONFIG,
    symbol_tuple=STRATEGY_SYMBOL_TUPLE,
    universe_name_str="original",
)


__all__ = [
    "DEFAULT_CONFIG",
    "SectorDispersionIbsKieStrategy",
    "STRATEGY_NAME_STR",
    "STRATEGY_SYMBOL_TUPLE",
    "run_variant",
]


def _write_kie_variant_notes_md(output_path: Path) -> None:
    notes_md_str = f"""# Sector Dispersion IBS KIE Variant

- Research-only strategy; no live/release wiring.
- Runnable strategy module: `{STRATEGY_NAME_STR}`.
- Fixed basket: `{", ".join(STRATEGY_SYMBOL_TUPLE)}`.
- This is the balanced single-addition candidate from the marginal universe and stress studies.
- Signal, sizing, costs, and execution timing are inherited unchanged from `strategy_mr_sector_dispersion_ibs`.
- Execution convention remains `signal T -> Open T+1`; this does not add MOC support.
"""
    (output_path / "sector_dispersion_ibs_kie_variant_notes.md").write_text(
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
) -> SectorDispersionIbsStrategy:
    config_obj: SectorDispersionIbsConfig = DEFAULT_CONFIG
    if backtest_start_date_str is not None or capital_base_float is not None or end_date_str is not None:
        config_obj = replace(
            config_obj,
            history_start_date_str=resolve_history_start_date_str(
                config_obj=config_obj,
                backtest_start_date_str=backtest_start_date_str,
            ),
            backtest_start_date_str=(
                config_obj.backtest_start_date_str
                if backtest_start_date_str is None
                else backtest_start_date_str
            ),
            capital_base_float=(
                config_obj.capital_base_float
                if capital_base_float is None
                else float(capital_base_float)
            ),
            end_date_str=end_date_str,
        )

    if pricing_data_df is None:
        pricing_data_df = get_sector_dispersion_ibs_data(config_obj=config_obj)

    strategy_obj = SectorDispersionIbsKieStrategy(
        name=STRATEGY_NAME_STR,
        benchmarks=[config_obj.benchmark_symbol_str],
        config_obj=config_obj,
    )

    # *** CRITICAL*** Keep pre-start history for the lagged range scale, but
    # only execute the backtest on and after backtest_start_date_str. Orders
    # still fill at the next bar open under the Vanilla engine contract.
    calendar_idx = pricing_data_df.index[
        pricing_data_df.index >= pd.Timestamp(config_obj.backtest_start_date_str)
    ]
    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar_idx,
        show_progress=show_display_bool,
        show_signal_progress_bool=show_display_bool,
        audit_override_bool=audit_override_bool,
    )

    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        display(strategy_obj.summary)
        display(strategy_obj.summary_trades)

    if save_results_bool:
        output_path = save_results(strategy_obj, output_dir=output_dir_str)
        _write_assumptions_md(output_path=output_path, strategy_obj=strategy_obj)
        _write_kie_variant_notes_md(output_path=output_path)

    return strategy_obj


if __name__ == "__main__":
    run_variant()
