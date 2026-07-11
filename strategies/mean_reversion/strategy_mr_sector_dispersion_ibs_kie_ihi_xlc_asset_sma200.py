"""
Research-only sector-dispersion IBS variant with an asset SMA200 entry gate.

This Bench-runnable defensive variant trades the fixed basket:

    SOXX, IGV, IBB, KIE, IHI, XLC

It keeps the base sector-dispersion IBS exit, sizing, costs, and next-open
execution semantics, but only allows new entries when the ETF is above its own
200-day simple moving average at the completed decision close.
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
    compute_sector_dispersion_ibs_signal_df,
    get_sector_dispersion_ibs_data,
    resolve_history_start_date_str,
)


STRATEGY_NAME_STR = "strategy_mr_sector_dispersion_ibs_kie_ihi_xlc_asset_sma200"
STRATEGY_SYMBOL_TUPLE = ORIGINAL_SYMBOL_TUPLE + ("KIE", "IHI", "XLC")
ASSET_SMA_LOOKBACK_DAY_INT = 200
ASSET_SMA_FIELD_STR = "asset_sma_200_ser"
ASSET_BULLISH_FIELD_STR = "asset_sma_200_bullish_bool"
RAW_ENTRY_SIGNAL_FIELD_STR = "raw_entry_signal_bool"


class SectorDispersionIbsKieIhiXlcAssetSma200Strategy(SectorDispersionIbsStrategy):
    """Fixed KIE+IHI+XLC basket with asset-SMA200 entry gating."""

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        return compute_asset_sma200_filtered_signal_df(
            pricing_data_df=pricing_data_df,
            config_obj=self.config_obj,
        )


DEFAULT_CONFIG = replace(
    BASE_DEFAULT_CONFIG,
    symbol_tuple=STRATEGY_SYMBOL_TUPLE,
    universe_name_str="original",
)


__all__ = [
    "ASSET_BULLISH_FIELD_STR",
    "ASSET_SMA_FIELD_STR",
    "ASSET_SMA_LOOKBACK_DAY_INT",
    "DEFAULT_CONFIG",
    "RAW_ENTRY_SIGNAL_FIELD_STR",
    "STRATEGY_NAME_STR",
    "STRATEGY_SYMBOL_TUPLE",
    "SectorDispersionIbsKieIhiXlcAssetSma200Strategy",
    "compute_asset_sma200_filtered_signal_df",
    "run_variant",
]


def _symbol_close_df(
    pricing_data_df: pd.DataFrame,
    symbol_tuple: tuple[str, ...],
) -> pd.DataFrame:
    missing_column_list = [
        (symbol_str, "Close")
        for symbol_str in symbol_tuple
        if (symbol_str, "Close") not in pricing_data_df.columns
    ]
    if len(missing_column_list) > 0:
        raise RuntimeError(f"Missing required asset-SMA columns: {missing_column_list}")
    return pd.DataFrame(
        {
            symbol_str: pd.to_numeric(pricing_data_df[(symbol_str, "Close")], errors="coerce")
            for symbol_str in symbol_tuple
        },
        index=pricing_data_df.index,
        dtype=float,
    )


def _multiindex_feature_df(feature_df: pd.DataFrame, field_str: str) -> pd.DataFrame:
    output_feature_df = feature_df.copy()
    output_feature_df.columns = pd.MultiIndex.from_tuples(
        [(str(symbol_str), field_str) for symbol_str in output_feature_df.columns]
    )
    return output_feature_df


def compute_asset_sma200_filtered_signal_df(
    pricing_data_df: pd.DataFrame,
    config_obj: SectorDispersionIbsConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Compute sector-dispersion IBS signals with asset-trend entry gating.

    Entry formula for ETF i on decision date t:

        raw_entry_{i,t}
            = 1[IBS_{i,t} < entry_ibs_max]
              * 1[RelativeRange_{i,t} > min_relative_range]

        asset_bullish_{i,t}
            = 1[Close_{i,t} > SMA200(Close_i)_t]

        entry_{i,t}
            = raw_entry_{i,t} * asset_bullish_{i,t}
    """
    signal_data_df = compute_sector_dispersion_ibs_signal_df(
        pricing_data_df=pricing_data_df,
        config_obj=config_obj,
    )
    symbol_tuple = tuple(config_obj.symbol_tuple)
    close_price_df = _symbol_close_df(pricing_data_df=signal_data_df, symbol_tuple=symbol_tuple)

    # *** CRITICAL*** SMA200_T uses the completed daily close at decision bar T
    # and can only gate orders that the Vanilla engine fills at Open T+1.
    # It must not be used for same-bar open or intraday execution claims.
    asset_sma_df = close_price_df.rolling(
        window=ASSET_SMA_LOOKBACK_DAY_INT,
        min_periods=ASSET_SMA_LOOKBACK_DAY_INT,
    ).mean()
    asset_bullish_bool_df = close_price_df.gt(asset_sma_df) & asset_sma_df.notna()

    raw_entry_signal_df = pd.DataFrame(
        {
            symbol_str: signal_data_df[(symbol_str, "entry_signal_bool")].astype(bool)
            for symbol_str in symbol_tuple
        },
        index=signal_data_df.index,
    )
    gated_entry_signal_df = raw_entry_signal_df & asset_bullish_bool_df

    feature_frame_list = [
        _multiindex_feature_df(raw_entry_signal_df, RAW_ENTRY_SIGNAL_FIELD_STR),
        _multiindex_feature_df(asset_sma_df, ASSET_SMA_FIELD_STR),
        _multiindex_feature_df(asset_bullish_bool_df.astype(bool), ASSET_BULLISH_FIELD_STR),
        _multiindex_feature_df(gated_entry_signal_df.astype(bool), "entry_signal_bool"),
    ]
    signal_without_base_entry_df = signal_data_df.drop(
        columns=pd.MultiIndex.from_tuples([(symbol_str, "entry_signal_bool") for symbol_str in symbol_tuple])
    )
    return pd.concat([signal_without_base_entry_df] + feature_frame_list, axis=1)


def _write_asset_sma200_variant_notes_md(output_path: Path) -> None:
    notes_md_str = f"""# Sector Dispersion IBS KIE+IHI+XLC Asset SMA200 Variant

- Research-only strategy; no live/release wiring.
- Runnable strategy module: `{STRATEGY_NAME_STR}`.
- Fixed basket: `{", ".join(STRATEGY_SYMBOL_TUPLE)}`.
- Entry signal: base sector-dispersion IBS entry plus `Close_i,T > SMA200_i,T`.
- Exit signal: inherited unchanged from `strategy_mr_sector_dispersion_ibs`.
- Sizing and costs: inherited unchanged from `strategy_mr_sector_dispersion_ibs`.
- Execution convention remains `signal T -> Open T+1`; this does not add MOC support.
- The SMA gate is an entry filter only; existing positions are not force-liquidated when the asset falls below SMA200.
"""
    (output_path / "sector_dispersion_ibs_kie_ihi_xlc_asset_sma200_variant_notes.md").write_text(
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
) -> SectorDispersionIbsKieIhiXlcAssetSma200Strategy:
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

    strategy_obj = SectorDispersionIbsKieIhiXlcAssetSma200Strategy(
        name=STRATEGY_NAME_STR,
        benchmarks=[config_obj.benchmark_symbol_str],
        config_obj=config_obj,
    )

    # *** CRITICAL*** Keep pre-start history for both the lagged range scale
    # and SMA200 gate, but execute only on and after backtest_start_date_str.
    # Orders still fill at the next bar open under the Vanilla engine contract.
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
        _write_asset_sma200_variant_notes_md(output_path=output_path)

    return strategy_obj


if __name__ == "__main__":
    run_variant()
