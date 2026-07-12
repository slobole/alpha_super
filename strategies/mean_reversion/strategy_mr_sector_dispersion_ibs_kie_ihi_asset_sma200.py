"""
Research-only KIE+IHI sector-dispersion IBS variant with an asset SMA200 gate.

This Bench-runnable variant trades the fixed basket:

    SOXX, IGV, IBB, KIE, IHI

It preserves the KIE+IHI baseline's exit, sizing, costs, and next-open
execution semantics. A new entry is allowed only when that ETF's completed
decision close is above its own 200-day simple moving average.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from strategies.mean_reversion.strategy_mr_sector_dispersion_ibs import (
    SectorDispersionIbsConfig,
    _write_assumptions_md,
    build_sector_dispersion_capacity_analysis_inputs,
    get_sector_dispersion_ibs_data,
    resolve_effective_backtest_start_date_str,
    resolve_full_basket_calendar_idx,
    resolve_history_start_date_str,
)
from strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi import (
    DEFAULT_CONFIG as BASE_DEFAULT_CONFIG,
    STRATEGY_SYMBOL_TUPLE,
    SectorDispersionIbsKieIhiStrategy,
)
from strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_xlc_asset_sma200 import (
    ASSET_BULLISH_FIELD_STR,
    ASSET_SMA_FIELD_STR,
    ASSET_SMA_LOOKBACK_DAY_INT,
    RAW_ENTRY_SIGNAL_FIELD_STR,
    compute_asset_sma200_filtered_signal_df as _compute_asset_sma200_filtered_signal_df,
)


STRATEGY_NAME_STR = "strategy_mr_sector_dispersion_ibs_kie_ihi_asset_sma200"


class SectorDispersionIbsKieIhiAssetSma200Strategy(SectorDispersionIbsKieIhiStrategy):
    """Fixed KIE+IHI basket with per-asset SMA200 entry gating."""

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        return compute_asset_sma200_filtered_signal_df(
            pricing_data_df=pricing_data_df,
            config_obj=self.config_obj,
        )


DEFAULT_CONFIG = replace(
    BASE_DEFAULT_CONFIG,
    symbol_tuple=STRATEGY_SYMBOL_TUPLE,
)


def compute_asset_sma200_filtered_signal_df(
    pricing_data_df: pd.DataFrame,
    config_obj: SectorDispersionIbsConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Apply the shared per-asset SMA200 gate with the KIE+IHI default basket."""
    return _compute_asset_sma200_filtered_signal_df(
        pricing_data_df=pricing_data_df,
        config_obj=config_obj,
    )


__all__ = [
    "ASSET_BULLISH_FIELD_STR",
    "ASSET_SMA_FIELD_STR",
    "ASSET_SMA_LOOKBACK_DAY_INT",
    "DEFAULT_CONFIG",
    "RAW_ENTRY_SIGNAL_FIELD_STR",
    "STRATEGY_NAME_STR",
    "STRATEGY_SYMBOL_TUPLE",
    "SectorDispersionIbsKieIhiAssetSma200Strategy",
    "build_capacity_analysis_inputs",
    "compute_asset_sma200_filtered_signal_df",
    "run_variant",
]


def build_capacity_analysis_inputs(
    capital_base_float: float,
    show_display_bool: bool = False,
    backtest_start_date_str: str | None = None,
    end_date_str: str | None = None,
) -> dict[str, object]:
    return build_sector_dispersion_capacity_analysis_inputs(
        strategy_class=SectorDispersionIbsKieIhiAssetSma200Strategy,
        strategy_name_str=STRATEGY_NAME_STR,
        config_obj=DEFAULT_CONFIG,
        capital_base_float=capital_base_float,
        show_display_bool=show_display_bool,
        backtest_start_date_str=backtest_start_date_str,
        end_date_str=end_date_str,
        required_close_history_observation_count_int=ASSET_SMA_LOOKBACK_DAY_INT,
    )


def _write_asset_sma200_variant_notes_md(output_path: Path) -> None:
    notes_md_str = f"""# Sector Dispersion IBS KIE+IHI Asset SMA200 Variant

- Research-only strategy; no live/release wiring.
- Runnable strategy module: `{STRATEGY_NAME_STR}`.
- Fixed basket: `{", ".join(STRATEGY_SYMBOL_TUPLE)}`.
- Entry signal: base KIE+IHI sector-dispersion IBS entry plus `Close_i,T > SMA200_i,T`.
- Exit signal, sizing, and costs: inherited unchanged from `strategy_mr_sector_dispersion_ibs`.
- Execution remains `signal T -> Open T+1`; this does not add MOC support.
- The SMA gate affects new entries only; it does not force an existing position to exit.
"""
    (output_path / "sector_dispersion_ibs_kie_ihi_asset_sma200_variant_notes.md").write_text(
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
) -> SectorDispersionIbsKieIhiAssetSma200Strategy:
    config_obj: SectorDispersionIbsConfig = DEFAULT_CONFIG
    effective_backtest_start_date_str = resolve_effective_backtest_start_date_str(
        config_obj=config_obj,
        requested_backtest_start_date_str=backtest_start_date_str,
    )
    if backtest_start_date_str is not None or capital_base_float is not None or end_date_str is not None:
        config_obj = replace(
            config_obj,
            history_start_date_str=resolve_history_start_date_str(
                config_obj=config_obj,
                backtest_start_date_str=effective_backtest_start_date_str,
            ),
            backtest_start_date_str=effective_backtest_start_date_str,
            capital_base_float=(
                config_obj.capital_base_float
                if capital_base_float is None
                else float(capital_base_float)
            ),
            end_date_str=end_date_str,
        )

    if pricing_data_df is None:
        pricing_data_df = get_sector_dispersion_ibs_data(config_obj=config_obj)

    strategy_obj = SectorDispersionIbsKieIhiAssetSma200Strategy(
        name=STRATEGY_NAME_STR,
        benchmarks=[config_obj.benchmark_symbol_str],
        config_obj=config_obj,
    )

    # *** CRITICAL*** The calendar retains causal history for both the lagged
    # range scale and SMA200_T. Signals use completed Close_T and orders still
    # execute only at Open T+1.
    calendar_idx = resolve_full_basket_calendar_idx(
        pricing_data_df=pricing_data_df,
        config_obj=config_obj,
        required_close_history_observation_count_int=ASSET_SMA_LOOKBACK_DAY_INT,
    )
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
        _write_asset_sma200_variant_notes_md(output_path=output_path)

    return strategy_obj


if __name__ == "__main__":
    run_variant()
