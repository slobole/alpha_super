"""Research-only positioning study for the asset-SMA200 sector IBS sleeves."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from alpha.engine.backtest import run_daily
from alpha.engine.report import build_research_output_path
from data.norgate_loader import load_raw_prices
from scripts.research.run_sector_dispersion_marginal_universe_study import (
    _json_default_obj,
    _strategy_summary_row_dict,
    compute_period_metric_dict,
)
from scripts.research.run_sector_dispersion_short_sleeve_study import (
    _benchmark_return_ser,
    _markdown_table_str,
    _market_metric_dict,
    _normalize_equity_ser,
    _performance_metric_dict,
)
from scripts.research.run_sector_dispersion_volatility_rank_filter_study import (
    _exposure_metric_dict,
)
from strategies.mean_reversion import (
    strategy_mr_sector_dispersion_ibs_kie_ihi_asset_sma200 as no_xlc_module,
)
from strategies.mean_reversion import (
    strategy_mr_sector_dispersion_ibs_kie_ihi_xlc_asset_sma200 as xlc_module,
)
from strategies.mean_reversion.strategy_mr_sector_dispersion_ibs import (
    SectorDispersionIbsConfig,
    SectorDispersionIbsStrategy,
    resolve_full_basket_calendar_idx,
    resolve_history_start_date_str,
)
from strategies.taa_df.strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash import (
    run_variant as run_taa_variant,
)


SIZING_EQUAL_SLOT_STR = "equal_slot"
SIZING_VIX_SCALE_STR = "vix_scale"
SIZING_ASSET_VOL_CAP_STR = "asset_vol_cap"
SIZING_MODE_TUPLE = (
    SIZING_EQUAL_SLOT_STR,
    SIZING_VIX_SCALE_STR,
    SIZING_ASSET_VOL_CAP_STR,
)
VIX_SYMBOL_STR = "$VIX"
POSITION_SCALE_FIELD_STR = "position_scale_ser"
ASSET_VOLATILITY_FIELD_STR = "asset_volatility_ann_ser"
DEFAULT_VIX_ANCHOR_FLOAT = 30.0
DEFAULT_SCALE_FLOOR_FLOAT = 0.50
DEFAULT_ASSET_VOL_LOOKBACK_DAY_INT = 20
DEFAULT_ASSET_VOL_TARGET_FLOAT = 0.30
DEFAULT_MARKET_TAIL_QUANTILE_FLOAT = 0.10
RECENT_DIAGNOSTIC_START_TS = pd.Timestamp("2022-01-01")


@dataclass(frozen=True)
class BasketSpec:
    basket_key_str: str
    basket_label_str: str
    base_config_obj: SectorDispersionIbsConfig
    requested_start_date_str: str


@dataclass(frozen=True)
class PositionSizingConfig:
    base_config_obj: SectorDispersionIbsConfig
    sizing_mode_str: str
    vix_anchor_float: float = DEFAULT_VIX_ANCHOR_FLOAT
    scale_floor_float: float = DEFAULT_SCALE_FLOOR_FLOAT
    asset_vol_lookback_day_int: int = DEFAULT_ASSET_VOL_LOOKBACK_DAY_INT
    asset_vol_target_float: float = DEFAULT_ASSET_VOL_TARGET_FLOAT

    def __post_init__(self) -> None:
        if self.sizing_mode_str not in SIZING_MODE_TUPLE:
            raise ValueError(f"sizing_mode_str must be one of {SIZING_MODE_TUPLE}.")
        if self.vix_anchor_float <= 0.0:
            raise ValueError("vix_anchor_float must be positive.")
        if not 0.0 < self.scale_floor_float <= 1.0:
            raise ValueError("scale_floor_float must lie in (0, 1].")
        if self.asset_vol_lookback_day_int <= 1:
            raise ValueError("asset_vol_lookback_day_int must be greater than 1.")
        if self.asset_vol_target_float <= 0.0:
            raise ValueError("asset_vol_target_float must be positive.")


def compute_vix_position_scale_ser(
    vix_close_ser: pd.Series,
    vix_anchor_float: float = DEFAULT_VIX_ANCHOR_FLOAT,
    scale_floor_float: float = DEFAULT_SCALE_FLOOR_FLOAT,
) -> pd.Series:
    """Return a one-sided VIX scaler that never increases exposure."""
    clean_vix_close_ser = pd.to_numeric(vix_close_ser, errors="coerce")
    valid_vix_bool_ser = np.isfinite(clean_vix_close_ser) & clean_vix_close_ser.gt(0.0)
    position_scale_ser = (float(vix_anchor_float) / clean_vix_close_ser).clip(
        lower=float(scale_floor_float),
        upper=1.0,
    )
    return position_scale_ser.where(valid_vix_bool_ser)


def compute_asset_vol_position_scale_df(
    close_price_df: pd.DataFrame,
    lookback_day_int: int = DEFAULT_ASSET_VOL_LOOKBACK_DAY_INT,
    asset_vol_target_float: float = DEFAULT_ASSET_VOL_TARGET_FLOAT,
    scale_floor_float: float = DEFAULT_SCALE_FLOOR_FLOAT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return causal annualized volatility and a one-sided entry-size cap."""
    numeric_close_price_df = close_price_df.apply(pd.to_numeric, errors="coerce")
    # *** CRITICAL*** decision-date volatility uses close-to-close returns only
    # through completed bar T. It sizes orders filled at Open T+1 and never uses
    # future returns or a full-sample normalization.
    close_return_df = numeric_close_price_df.pct_change(fill_method=None)
    asset_volatility_ann_df = close_return_df.rolling(
        window=int(lookback_day_int),
        min_periods=int(lookback_day_int),
    ).std(ddof=1) * np.sqrt(252.0)
    position_scale_df = (float(asset_vol_target_float) / asset_volatility_ann_df).clip(
        lower=float(scale_floor_float),
        upper=1.0,
    )
    return asset_volatility_ann_df, position_scale_df


def _multiindex_feature_df(feature_df: pd.DataFrame, field_str: str) -> pd.DataFrame:
    output_feature_df = feature_df.copy()
    output_feature_df.columns = pd.MultiIndex.from_tuples(
        [(str(symbol_str), field_str) for symbol_str in output_feature_df.columns]
    )
    return output_feature_df


class SectorDispersionPositionSizingStrategy(SectorDispersionIbsStrategy):
    """Asset-SMA200 sector IBS with entry-only defensive position scaling."""

    def __init__(
        self,
        name: str,
        benchmarks: list[str] | tuple[str, ...],
        positioning_config_obj: PositionSizingConfig,
    ) -> None:
        super().__init__(
            name=name,
            benchmarks=benchmarks,
            config_obj=positioning_config_obj.base_config_obj,
        )
        self.positioning_config_obj = positioning_config_obj
        self.entry_scale_record_list: list[dict[str, object]] = []

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        signal_data_df = xlc_module.compute_asset_sma200_filtered_signal_df(
            pricing_data_df=pricing_data_df,
            config_obj=self.config_obj,
        )
        close_price_df = pd.DataFrame(
            {
                symbol_str: pd.to_numeric(
                    signal_data_df[(symbol_str, "Close")],
                    errors="coerce",
                )
                for symbol_str in self.symbol_tuple
            },
            index=signal_data_df.index,
            dtype=float,
        )
        asset_volatility_ann_df, asset_vol_scale_df = compute_asset_vol_position_scale_df(
            close_price_df=close_price_df,
            lookback_day_int=self.positioning_config_obj.asset_vol_lookback_day_int,
            asset_vol_target_float=self.positioning_config_obj.asset_vol_target_float,
            scale_floor_float=self.positioning_config_obj.scale_floor_float,
        )

        if self.positioning_config_obj.sizing_mode_str == SIZING_VIX_SCALE_STR:
            if (VIX_SYMBOL_STR, "Close") not in signal_data_df.columns:
                raise RuntimeError(f"Missing required VIX close column: {(VIX_SYMBOL_STR, 'Close')}")
            # *** CRITICAL*** VIX Close_T is a completed decision-bar value and
            # scales only orders submitted for Open T+1. It is never backfilled.
            vix_position_scale_ser = compute_vix_position_scale_ser(
                signal_data_df[(VIX_SYMBOL_STR, "Close")],
                vix_anchor_float=self.positioning_config_obj.vix_anchor_float,
                scale_floor_float=self.positioning_config_obj.scale_floor_float,
            )
            position_scale_df = pd.DataFrame(
                {
                    symbol_str: vix_position_scale_ser
                    for symbol_str in self.symbol_tuple
                },
                index=signal_data_df.index,
            )
        elif self.positioning_config_obj.sizing_mode_str == SIZING_ASSET_VOL_CAP_STR:
            position_scale_df = asset_vol_scale_df
        else:
            position_scale_df = pd.DataFrame(
                1.0,
                index=signal_data_df.index,
                columns=self.symbol_tuple,
            )

        return pd.concat(
            [
                signal_data_df,
                _multiindex_feature_df(asset_volatility_ann_df, ASSET_VOLATILITY_FIELD_STR),
                _multiindex_feature_df(position_scale_df, POSITION_SCALE_FIELD_STR),
            ],
            axis=1,
        )

    def _entry_target_share_float(
        self,
        symbol_str: str,
        close_row_ser: pd.Series,
    ) -> float:
        close_price_float = float(close_row_ser.get((symbol_str, "Close"), np.nan))
        position_scale_float = float(
            close_row_ser.get((symbol_str, POSITION_SCALE_FIELD_STR), np.nan)
        )
        if not np.isfinite(close_price_float) or close_price_float <= 0.0:
            raise RuntimeError(f"Cannot size {symbol_str} without a valid decision Close_T.")
        if not np.isfinite(position_scale_float):
            raise RuntimeError(f"Cannot size {symbol_str} without a causal position scale.")
        if not self.positioning_config_obj.scale_floor_float <= position_scale_float <= 1.0:
            raise RuntimeError(f"Position scale for {symbol_str} lies outside the configured bounds.")

        self.entry_scale_record_list.append(
            {
                "decision_date_str": pd.Timestamp(self.previous_bar).date().isoformat(),
                "symbol_str": symbol_str,
                "position_scale_float": position_scale_float,
            }
        )
        return (
            float(self.previous_total_value)
            * self.target_weight_float
            * position_scale_float
            / close_price_float
        )


def _basket_spec_tuple() -> tuple[BasketSpec, ...]:
    return (
        BasketSpec(
            basket_key_str="no_xlc",
            basket_label_str="KIE+IHI asset SMA200",
            base_config_obj=no_xlc_module.DEFAULT_CONFIG,
            requested_start_date_str="2012-10-01",
        ),
        BasketSpec(
            basket_key_str="xlc",
            basket_label_str="KIE+IHI+XLC asset SMA200",
            base_config_obj=xlc_module.DEFAULT_CONFIG,
            requested_start_date_str="2016-01-01",
        ),
    )


def _run_positioning_variant(
    basket_spec_obj: BasketSpec,
    sizing_mode_str: str,
    pricing_data_df: pd.DataFrame,
    end_date_str: str | None,
    show_progress_bool: bool,
) -> SectorDispersionPositionSizingStrategy:
    base_config_obj = replace(
        basket_spec_obj.base_config_obj,
        history_start_date_str=resolve_history_start_date_str(
            config_obj=basket_spec_obj.base_config_obj,
            backtest_start_date_str=basket_spec_obj.requested_start_date_str,
        ),
        backtest_start_date_str=basket_spec_obj.requested_start_date_str,
        end_date_str=end_date_str,
    )
    positioning_config_obj = PositionSizingConfig(
        base_config_obj=base_config_obj,
        sizing_mode_str=sizing_mode_str,
    )
    strategy_obj = SectorDispersionPositionSizingStrategy(
        name=f"sector_dispersion_{basket_spec_obj.basket_key_str}_{sizing_mode_str}",
        benchmarks=[base_config_obj.benchmark_symbol_str],
        positioning_config_obj=positioning_config_obj,
    )
    calendar_idx = resolve_full_basket_calendar_idx(
        pricing_data_df=pricing_data_df,
        config_obj=base_config_obj,
        required_close_history_observation_count_int=xlc_module.ASSET_SMA_LOOKBACK_DAY_INT,
    )
    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar=calendar_idx,
        show_progress=show_progress_bool,
        show_signal_progress_bool=show_progress_bool,
        audit_override_bool=True,
    )
    return strategy_obj


def _entry_scale_metric_dict(
    strategy_obj: SectorDispersionPositionSizingStrategy,
) -> dict[str, object]:
    entry_scale_df = pd.DataFrame(strategy_obj.entry_scale_record_list)
    if len(entry_scale_df) == 0:
        return {
            "average_entry_scale_float": np.nan,
            "scaled_entry_pct_float": np.nan,
        }
    entry_scale_ser = pd.to_numeric(entry_scale_df["position_scale_float"], errors="coerce")
    return {
        "average_entry_scale_float": float(entry_scale_ser.mean()),
        "scaled_entry_pct_float": float(entry_scale_ser.lt(1.0 - 1e-12).mean() * 100.0),
    }


def _combine_pod_equity_ser(
    strategy_equity_ser: pd.Series,
    taa_equity_ser: pd.Series,
) -> pd.Series:
    normalized_equity_df = pd.concat(
        [
            _normalize_equity_ser(strategy_equity_ser).rename("sector"),
            _normalize_equity_ser(taa_equity_ser).rename("taa"),
        ],
        axis=1,
    ).dropna()
    if len(normalized_equity_df) == 0:
        raise RuntimeError("Sector and TAA equity curves do not overlap.")
    return 0.5 * normalized_equity_df["sector"] + 0.5 * normalized_equity_df["taa"]


def _portfolio_summary_row_dict(
    basket_spec_obj: BasketSpec,
    sizing_mode_str: str,
    portfolio_equity_ser: pd.Series,
    benchmark_return_ser: pd.Series,
) -> dict[str, object]:
    row_dict: dict[str, object] = {
        "basket_key_str": basket_spec_obj.basket_key_str,
        "basket_label_str": basket_spec_obj.basket_label_str,
        "sizing_mode_str": sizing_mode_str,
    }
    row_dict.update(_performance_metric_dict(portfolio_equity_ser))
    row_dict.update(
        compute_period_metric_dict(
            total_value_ser=portfolio_equity_ser,
            start_ts=RECENT_DIAGNOSTIC_START_TS,
            end_ts=None,
            prefix_str="recent_2022",
        )
    )
    row_dict.update(
        _market_metric_dict(
            portfolio_equity_ser,
            benchmark_return_ser,
            DEFAULT_MARKET_TAIL_QUANTILE_FLOAT,
        )
    )
    return row_dict


def _rename_oos_metrics_as_recent_diagnostics(row_dict: dict[str, object]) -> None:
    """Relabel the inherited split honestly; 2022+ was visible when designed."""
    for key_str in list(row_dict):
        if not key_str.startswith("oos_"):
            continue
        row_dict[f"recent_2022_{key_str.removeprefix('oos_')}"] = row_dict.pop(key_str)


def _save_equity_chart(
    output_path: Path,
    equity_df: pd.DataFrame,
) -> None:
    fig_obj, axis_arr = plt.subplots(2, 1, figsize=(14, 11), sharex=False)
    for axis_obj, basket_key_str in zip(axis_arr, ("no_xlc", "xlc"), strict=True):
        basket_column_list = [
            column_str
            for column_str in equity_df.columns
            if str(column_str).startswith(f"{basket_key_str}__")
        ]
        for column_str in basket_column_list:
            normalized_equity_ser = _normalize_equity_ser(equity_df[column_str])
            axis_obj.plot(
                normalized_equity_ser.index,
                normalized_equity_ser.values,
                label=column_str.split("__", maxsplit=1)[1],
                linewidth=1.6,
            )
        axis_obj.set_title(f"Sector Dispersion Position Sizing - {basket_key_str}")
        axis_obj.set_ylabel("Growth of 1.0")
        axis_obj.grid(True, alpha=0.25)
        axis_obj.legend(loc="best")
    axis_arr[-1].set_xlabel("Date")
    fig_obj.tight_layout()
    fig_obj.savefig(output_path / "position_sizing_equity_curves.png", dpi=170)
    plt.close(fig_obj)


def _write_recommendations_md(
    output_path: Path,
    standalone_summary_df: pd.DataFrame,
    multipod_summary_df: pd.DataFrame,
) -> None:
    standalone_column_list = [
        "basket_label_str",
        "sizing_mode_str",
        "start_date_str",
        "end_date_str",
        "ann_return_pct_float",
        "volatility_ann_pct_float",
        "sharpe_float",
        "max_drawdown_pct_float",
        "mar_float",
        "turnover_ann_pct_float",
        "cost_drag_ann_pct_float",
        "avg_gross_exposure_pct_float",
        "average_entry_scale_float",
        "market_tail_mean_return_pct_float",
        "market_tail_beta_to_spx_float",
    ]
    multipod_column_list = [
        "basket_label_str",
        "sizing_mode_str",
        "ann_return_pct_float",
        "volatility_ann_pct_float",
        "sharpe_float",
        "max_drawdown_pct_float",
        "recent_2022_sharpe_float",
        "market_tail_mean_return_pct_float",
        "market_tail_beta_to_spx_float",
    ]
    recommendation_md_str = f"""# Sector Dispersion Position Sizing Study

## Scope

- Research-only; no strategy default, PortfolioManager config, or live/release wiring changed.
- Local search count: `6` runs = 2 fixed baskets x 3 predeclared sizing rules.
- The broader strategy-family search is larger and includes prior universe, SMA200, short-sleeve, leverage, and filter studies.
- Baseline gross target: `1.0`, divided equally across every basket slot.
- VIX rule: `clip(30 / VIX_T, 0.50, 1.00)` applied at entry only.
- Asset-vol rule: `clip(30% / Vol20_i,T, 0.50, 1.00)` applied at entry only.
- Signals use completed bar `T`; all orders fill at `Open T+1` with unchanged costs.
- Existing positions are not resized; every variant keeps the original IBS exit.
- `recent_2022_*` fields are descriptive recent-period diagnostics, not untouched out-of-sample evidence.

## Standalone Sleeves

{_markdown_table_str(standalone_summary_df[standalone_column_list])}

## 50/50 TAA Multipod Diagnostics

{_markdown_table_str(multipod_summary_df[multipod_column_list])}

## Interpretation Reminder

VIX and asset-vol sizing are defensive overlays, not evidence of improved raw alpha. Market-tail days use each aligned sample's ex-post worst 10% SPX threshold. The XLC sample begins only after the complete basket and SMA200 history are available. The fixed ETF baskets retain ex-post universe-selection bias.
"""
    (output_path / "recommendations.md").write_text(recommendation_md_str, encoding="utf-8")


def run_position_sizing_study(
    output_dir_str: str = "results",
    end_date_str: str | None = None,
    show_progress_bool: bool = False,
) -> Path:
    basket_spec_tuple = _basket_spec_tuple()
    timestamp_str = pd.Timestamp.now().strftime("%Y-%m-%d_%H%M%S")
    output_path = build_research_output_path(
        output_dir=output_dir_str,
        entity_type_str="strategy",
        entity_id_str="strategy_mr_sector_dispersion_ibs",
        analysis_type_str="position_sizing_study",
        timestamp_str=timestamp_str,
    )
    output_path.mkdir(parents=True, exist_ok=False)

    all_symbol_tuple = tuple(
        dict.fromkeys(
            symbol_str
            for basket_spec_obj in basket_spec_tuple
            for symbol_str in basket_spec_obj.base_config_obj.symbol_tuple
        )
    )
    history_start_date_str = min(
        basket_spec_obj.base_config_obj.history_start_date_str
        for basket_spec_obj in basket_spec_tuple
    )
    pricing_data_df = load_raw_prices(
        symbols=list(all_symbol_tuple),
        benchmarks=[no_xlc_module.DEFAULT_CONFIG.benchmark_symbol_str, VIX_SYMBOL_STR],
        start_date=history_start_date_str,
        end_date=end_date_str,
    )
    benchmark_return_ser = _benchmark_return_ser(
        pricing_data_df=pricing_data_df,
        benchmark_symbol_str=no_xlc_module.DEFAULT_CONFIG.benchmark_symbol_str,
    )

    manifest_df = pd.DataFrame(
        [
            {
                "basket_key_str": basket_spec_obj.basket_key_str,
                "basket_label_str": basket_spec_obj.basket_label_str,
                "symbol_tuple_str": ",".join(basket_spec_obj.base_config_obj.symbol_tuple),
                "requested_start_date_str": basket_spec_obj.requested_start_date_str,
                "sizing_mode_str": sizing_mode_str,
            }
            for basket_spec_obj in basket_spec_tuple
            for sizing_mode_str in SIZING_MODE_TUPLE
        ]
    )
    manifest_df.to_csv(output_path / "variant_manifest.csv", index=False)

    standalone_row_list: list[dict[str, object]] = []
    multipod_row_list: list[dict[str, object]] = []
    standalone_equity_dict: dict[str, pd.Series] = {}
    multipod_equity_dict: dict[str, pd.Series] = {}
    taa_equity_dict: dict[str, pd.Series] = {}
    taa_provenance_dict: dict[str, dict[str, object]] = {}
    for basket_spec_obj in basket_spec_tuple:
        taa_strategy_obj = None
        for sizing_mode_str in SIZING_MODE_TUPLE:
            variant_key_str = f"{basket_spec_obj.basket_key_str}__{sizing_mode_str}"
            print(f"Running {variant_key_str}...", flush=True)
            strategy_obj = _run_positioning_variant(
                basket_spec_obj=basket_spec_obj,
                sizing_mode_str=sizing_mode_str,
                pricing_data_df=pricing_data_df,
                end_date_str=end_date_str,
                show_progress_bool=show_progress_bool,
            )
            if taa_strategy_obj is None:
                effective_start_date_str = pd.Timestamp(strategy_obj.results.index[0]).date().isoformat()
                taa_strategy_obj = run_taa_variant(
                    show_display_bool=False,
                    save_results_bool=False,
                    backtest_start_date_str=effective_start_date_str,
                    capital_base_float=100_000.0,
                    end_date_str=end_date_str,
                )
                taa_equity_dict[basket_spec_obj.basket_key_str] = taa_strategy_obj.results[
                    "total_value"
                ]
                taa_provenance_dict[basket_spec_obj.basket_key_str] = {
                    "strategy_name_str": taa_strategy_obj.name,
                    "class_module_str": taa_strategy_obj.__class__.__module__,
                    "effective_start_date_str": pd.Timestamp(
                        taa_strategy_obj.results.index[0]
                    ).date().isoformat(),
                    "end_date_str": pd.Timestamp(
                        taa_strategy_obj.results.index[-1]
                    ).date().isoformat(),
                    "slippage_float": float(taa_strategy_obj._slippage),
                    "commission_per_share_float": float(
                        taa_strategy_obj._commission_per_share
                    ),
                    "commission_minimum_float": float(
                        taa_strategy_obj._commission_minimum
                    ),
                }

            standalone_row_dict = _strategy_summary_row_dict(
                strategy_obj=strategy_obj,
                variant_kind_str="position_sizing",
                candidate_symbol_str=None,
                bucket_str=None,
            )
            standalone_row_dict["variant_key_str"] = variant_key_str
            standalone_row_dict["basket_key_str"] = basket_spec_obj.basket_key_str
            standalone_row_dict["basket_label_str"] = basket_spec_obj.basket_label_str
            standalone_row_dict["sizing_mode_str"] = sizing_mode_str
            standalone_row_dict["portfolio_leverage_float"] = strategy_obj.config_obj.portfolio_leverage_float
            _rename_oos_metrics_as_recent_diagnostics(standalone_row_dict)
            standalone_row_dict.update(_entry_scale_metric_dict(strategy_obj))
            standalone_row_dict.update(_exposure_metric_dict(strategy_obj))
            standalone_row_dict.update(
                _market_metric_dict(
                    strategy_obj.results["total_value"],
                    benchmark_return_ser,
                    DEFAULT_MARKET_TAIL_QUANTILE_FLOAT,
                )
            )
            standalone_row_list.append(standalone_row_dict)
            standalone_equity_dict[variant_key_str] = strategy_obj.results["total_value"]

            portfolio_equity_ser = _combine_pod_equity_ser(
                strategy_equity_ser=strategy_obj.results["total_value"],
                taa_equity_ser=taa_strategy_obj.results["total_value"],
            )
            multipod_row_list.append(
                _portfolio_summary_row_dict(
                    basket_spec_obj=basket_spec_obj,
                    sizing_mode_str=sizing_mode_str,
                    portfolio_equity_ser=portfolio_equity_ser,
                    benchmark_return_ser=benchmark_return_ser,
                )
            )
            multipod_equity_dict[variant_key_str] = portfolio_equity_ser

    standalone_summary_df = pd.DataFrame(standalone_row_list)
    multipod_summary_df = pd.DataFrame(multipod_row_list)
    standalone_equity_df = pd.DataFrame(standalone_equity_dict).sort_index()
    multipod_equity_df = pd.DataFrame(multipod_equity_dict).sort_index()
    taa_equity_df = pd.DataFrame(taa_equity_dict).sort_index()

    standalone_summary_df.to_csv(output_path / "standalone_summary.csv", index=False)
    multipod_summary_df.to_csv(output_path / "multipod_summary.csv", index=False)
    standalone_equity_df.to_csv(output_path / "standalone_equity_curves.csv", index_label="date")
    multipod_equity_df.to_csv(output_path / "multipod_equity_curves.csv", index_label="date")
    taa_equity_df.to_csv(output_path / "taa_input_equity_curves.csv", index_label="date")
    _save_equity_chart(output_path=output_path, equity_df=multipod_equity_df)
    _write_recommendations_md(
        output_path=output_path,
        standalone_summary_df=standalone_summary_df,
        multipod_summary_df=multipod_summary_df,
    )

    metadata_dict = {
        "analysis_type_str": "position_sizing_study",
        "generated_at_str": pd.Timestamp.now().isoformat(),
        "output_path_str": str(output_path.resolve()),
        "end_date_str": end_date_str,
        "local_variant_count_int": len(manifest_df),
        "broader_strategy_family_search_note_str": (
            "This local count excludes prior universe, combination, SMA200, short-sleeve, "
            "leverage, and volatility-filter studies."
        ),
        "sizing_mode_tuple": SIZING_MODE_TUPLE,
        "portfolio_leverage_float": no_xlc_module.DEFAULT_CONFIG.portfolio_leverage_float,
        "vix_symbol_str": VIX_SYMBOL_STR,
        "vix_anchor_float": DEFAULT_VIX_ANCHOR_FLOAT,
        "scale_floor_float": DEFAULT_SCALE_FLOOR_FLOAT,
        "asset_vol_lookback_day_int": DEFAULT_ASSET_VOL_LOOKBACK_DAY_INT,
        "asset_vol_target_float": DEFAULT_ASSET_VOL_TARGET_FLOAT,
        "sector_sleeve_cost_model_dict": {
            "slippage_float": no_xlc_module.DEFAULT_CONFIG.slippage_float,
            "commission_per_share_float": no_xlc_module.DEFAULT_CONFIG.commission_per_share_float,
            "commission_minimum_float": no_xlc_module.DEFAULT_CONFIG.commission_minimum_float,
        },
        "taa_sleeve_provenance_by_basket_dict": taa_provenance_dict,
        "execution_timing_note_str": "Completed bar T signal and sizing; market fill at Open T+1.",
        "sizing_timing_note_str": "Defensive scale is set only when a new position opens; held positions are not resized.",
        "recent_diagnostic_note_str": (
            "2022+ metrics are descriptive recent-period diagnostics, not untouched OOS evidence."
        ),
        "tail_diagnostic_note_str": (
            "Market-tail metrics use each aligned sample's ex-post worst 10% SPX daily-return threshold."
        ),
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata_dict, indent=2, default=_json_default_obj),
        encoding="utf-8",
    )
    print(f"Saved position sizing study to {output_path}", flush=True)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser_obj = argparse.ArgumentParser(
        description="Run the research-only sector-dispersion positioning study."
    )
    parser_obj.add_argument("--output-dir", default="results")
    parser_obj.add_argument("--end-date", default=None)
    parser_obj.add_argument("--show-progress", action="store_true")
    return parser_obj.parse_args()


def main() -> int:
    args_obj = _parse_args()
    run_position_sizing_study(
        output_dir_str=str(args_obj.output_dir),
        end_date_str=args_obj.end_date,
        show_progress_bool=bool(args_obj.show_progress),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
