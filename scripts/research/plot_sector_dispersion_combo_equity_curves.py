from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from alpha.engine.report import build_research_output_path
from scripts.research.run_sector_dispersion_marginal_universe_study import (
    _json_default_obj,
    _run_strategy_variant,
    _strategy_summary_row_dict,
)
from strategies.mean_reversion.strategy_mr_sector_dispersion_ibs import (
    DEFAULT_CONFIG,
    ORIGINAL_SYMBOL_TUPLE,
    UNIVERSE_A_SYMBOL_TUPLE,
    UNIVERSE_B_SYMBOL_TUPLE,
    UNIVERSE_C_SYMBOL_TUPLE,
    get_sector_dispersion_ibs_data,
)


@dataclass(frozen=True)
class EquityCurveVariantSpec:
    label_str: str
    addition_symbol_tuple: tuple[str, ...]
    note_str: str


DEFAULT_COMPARISON_VARIANT_SPEC_TUPLE = (
    EquityCurveVariantSpec(
        label_str="Base",
        addition_symbol_tuple=(),
        note_str="Original paper basket.",
    ),
    EquityCurveVariantSpec(
        label_str="Base+KIE",
        addition_symbol_tuple=("KIE",),
        note_str="Best single-add balanced candidate.",
    ),
    EquityCurveVariantSpec(
        label_str="Base+KIE+IHI",
        addition_symbol_tuple=("KIE", "IHI"),
        note_str="Recommended default finalist from the combination study.",
    ),
    EquityCurveVariantSpec(
        label_str="Base+KIE+IHI+XLC",
        addition_symbol_tuple=("KIE", "IHI", "XLC"),
        note_str="Aggressive balanced finalist.",
    ),
    EquityCurveVariantSpec(
        label_str="Base+KIE+XLRE+IHI",
        addition_symbol_tuple=("KIE", "XLRE", "IHI"),
        note_str="Defensive tail finalist.",
    ),
    EquityCurveVariantSpec(
        label_str="Base+KIE+XLRE",
        addition_symbol_tuple=("KIE", "XLRE"),
        note_str="Direct KIE plus XLRE comparison requested during review.",
    ),
    EquityCurveVariantSpec(
        label_str="Base+XLRE",
        addition_symbol_tuple=("XLRE",),
        note_str="Pure XLRE marginal check.",
    ),
)


FULL_UNIVERSE_COMPARISON_VARIANT_SPEC_TUPLE = (
    EquityCurveVariantSpec(
        label_str="Full Universe A",
        addition_symbol_tuple=tuple(
            symbol_str for symbol_str in UNIVERSE_A_SYMBOL_TUPLE if symbol_str not in ORIGINAL_SYMBOL_TUPLE
        ),
        note_str="Full clean/conservative Universe A diagnostic.",
    ),
    EquityCurveVariantSpec(
        label_str="Full Universe B",
        addition_symbol_tuple=tuple(
            symbol_str for symbol_str in UNIVERSE_B_SYMBOL_TUPLE if symbol_str not in ORIGINAL_SYMBOL_TUPLE
        ),
        note_str="Full alpha/subsector Universe B diagnostic.",
    ),
    EquityCurveVariantSpec(
        label_str="Full Universe C",
        addition_symbol_tuple=tuple(
            symbol_str for symbol_str in UNIVERSE_C_SYMBOL_TUPLE if symbol_str not in ORIGINAL_SYMBOL_TUPLE
        ),
        note_str="Full research-only Universe C diagnostic.",
    ),
)


def build_variant_spec_tuple(include_full_universes_bool: bool = False) -> tuple[EquityCurveVariantSpec, ...]:
    if not include_full_universes_bool:
        return DEFAULT_COMPARISON_VARIANT_SPEC_TUPLE
    return DEFAULT_COMPARISON_VARIANT_SPEC_TUPLE + FULL_UNIVERSE_COMPARISON_VARIANT_SPEC_TUPLE


def _slug_str(raw_value_str: str) -> str:
    keep_char_list: list[str] = []
    for char_str in raw_value_str.lower():
        keep_char_list.append(char_str if char_str.isalnum() else "_")
    return "_".join(filter(None, "".join(keep_char_list).split("_")))


def build_comparison_manifest_df(
    variant_spec_tuple: tuple[EquityCurveVariantSpec, ...] = DEFAULT_COMPARISON_VARIANT_SPEC_TUPLE,
) -> pd.DataFrame:
    row_dict_list: list[dict[str, object]] = []
    for variant_rank_int, variant_spec_obj in enumerate(variant_spec_tuple, start=1):
        symbol_tuple = tuple(dict.fromkeys(ORIGINAL_SYMBOL_TUPLE + variant_spec_obj.addition_symbol_tuple))
        row_dict_list.append(
            {
                "variant_rank_int": variant_rank_int,
                "variant_label_str": variant_spec_obj.label_str,
                "addition_count_int": len(variant_spec_obj.addition_symbol_tuple),
                "addition_tuple_str": ",".join(variant_spec_obj.addition_symbol_tuple),
                "symbol_tuple_str": ",".join(symbol_tuple),
                "note_str": variant_spec_obj.note_str,
            }
        )

    manifest_df = pd.DataFrame(row_dict_list)
    if manifest_df["variant_label_str"].duplicated().any():
        raise RuntimeError("Equity-curve comparison manifest contains duplicate labels.")
    if manifest_df["addition_tuple_str"].duplicated().any():
        raise RuntimeError("Equity-curve comparison manifest contains duplicate addition tuples.")
    return manifest_df


def normalize_equity_curve_df(equity_curve_df: pd.DataFrame) -> pd.DataFrame:
    first_value_dict: dict[str, float] = {}
    for column_str in equity_curve_df.columns:
        value_ser = pd.to_numeric(equity_curve_df[column_str], errors="coerce").dropna()
        if len(value_ser) == 0:
            raise ValueError(f"Cannot normalize empty equity curve column {column_str!r}.")
        first_value_float = float(value_ser.iloc[0])
        if not np.isfinite(first_value_float) or first_value_float <= 0.0:
            raise ValueError(f"Cannot normalize non-positive equity curve column {column_str!r}.")
        first_value_dict[column_str] = first_value_float

    first_value_ser = pd.Series(first_value_dict, dtype=float)
    return equity_curve_df.astype(float).divide(first_value_ser, axis="columns")


def compute_drawdown_curve_df(normalized_equity_curve_df: pd.DataFrame) -> pd.DataFrame:
    # *** CRITICAL*** Post-run diagnostic only:
    # drawdown_{j,t} = equity_{j,t} / max(equity_{j,0}, ..., equity_{j,t}) - 1.
    # This must never feed signal generation, sizing, or variant selection inside a backtest.
    running_peak_df = normalized_equity_curve_df.cummax()
    return normalized_equity_curve_df / running_peak_df - 1.0


def _daily_return_ser(equity_ser: pd.Series) -> pd.Series:
    clean_equity_ser = pd.to_numeric(equity_ser, errors="coerce").dropna()
    clean_equity_ser.index = pd.to_datetime(clean_equity_ser.index).normalize()
    # *** CRITICAL*** Post-run diagnostic only:
    # r_t = equity_t / equity_{t-1} - 1. These realized returns are computed
    # after the backtests complete and must not affect signals, sizing, or fills.
    return clean_equity_ser.pct_change(fill_method=None).dropna()


def _correlation_float(left_return_ser: pd.Series, right_return_ser: pd.Series) -> float:
    aligned_return_df = pd.concat(
        [left_return_ser.rename("left"), right_return_ser.rename("right")],
        axis=1,
    ).dropna()
    if len(aligned_return_df) < 3:
        return float("nan")
    return float(aligned_return_df["left"].corr(aligned_return_df["right"]))


def _beta_float(strategy_return_ser: pd.Series, benchmark_return_ser: pd.Series) -> float:
    aligned_return_df = pd.concat(
        [strategy_return_ser.rename("strategy"), benchmark_return_ser.rename("benchmark")],
        axis=1,
    ).dropna()
    if len(aligned_return_df) < 3:
        return float("nan")
    benchmark_variance_float = float(aligned_return_df["benchmark"].var())
    if not np.isfinite(benchmark_variance_float) or benchmark_variance_float == 0.0:
        return float("nan")
    return float(aligned_return_df["strategy"].cov(aligned_return_df["benchmark"]) / benchmark_variance_float)


def compute_market_correlation_summary_df(
    equity_curve_df: pd.DataFrame,
    benchmark_label_str: str,
    market_tail_quantile_float: float = 0.10,
) -> pd.DataFrame:
    if benchmark_label_str not in equity_curve_df.columns:
        raise ValueError(f"benchmark_label_str={benchmark_label_str!r} is not in equity_curve_df.")
    if not 0.0 < market_tail_quantile_float < 1.0:
        raise ValueError("market_tail_quantile_float must lie between 0 and 1.")

    benchmark_return_ser = _daily_return_ser(equity_curve_df[benchmark_label_str])
    market_down_bool_ser = benchmark_return_ser.lt(0.0)
    market_tail_threshold_float = float(benchmark_return_ser.quantile(market_tail_quantile_float))
    market_tail_bool_ser = benchmark_return_ser.le(market_tail_threshold_float)

    row_dict_list: list[dict[str, object]] = []
    for variant_label_str in equity_curve_df.columns:
        if variant_label_str == benchmark_label_str:
            continue
        strategy_return_ser = _daily_return_ser(equity_curve_df[variant_label_str])
        aligned_return_df = pd.concat(
            [strategy_return_ser.rename("strategy"), benchmark_return_ser.rename("benchmark")],
            axis=1,
        ).dropna()
        common_index = benchmark_return_ser.index.intersection(aligned_return_df.index)
        aligned_market_down_bool_ser = pd.Series(False, index=aligned_return_df.index)
        aligned_market_tail_bool_ser = pd.Series(False, index=aligned_return_df.index)
        aligned_market_down_bool_ser.loc[common_index] = market_down_bool_ser.loc[common_index].astype(bool)
        aligned_market_tail_bool_ser.loc[common_index] = market_tail_bool_ser.loc[common_index].astype(bool)
        market_down_return_df = aligned_return_df.loc[aligned_market_down_bool_ser]
        market_tail_return_df = aligned_return_df.loc[aligned_market_tail_bool_ser]

        row_dict_list.append(
            {
                "variant_label_str": variant_label_str,
                "corr_to_spx_float": _correlation_float(
                    aligned_return_df["strategy"],
                    aligned_return_df["benchmark"],
                ),
                "market_down_day_count_int": int(len(market_down_return_df)),
                "market_down_corr_to_spx_float": _correlation_float(
                    market_down_return_df["strategy"],
                    market_down_return_df["benchmark"],
                ),
                "market_down_beta_to_spx_float": _beta_float(
                    market_down_return_df["strategy"],
                    market_down_return_df["benchmark"],
                ),
                "market_down_mean_return_pct_float": float(market_down_return_df["strategy"].mean() * 100.0),
                "market_tail_quantile_float": float(market_tail_quantile_float),
                "market_tail_threshold_return_pct_float": market_tail_threshold_float * 100.0,
                "market_tail_day_count_int": int(len(market_tail_return_df)),
                "market_tail_corr_to_spx_float": _correlation_float(
                    market_tail_return_df["strategy"],
                    market_tail_return_df["benchmark"],
                ),
                "market_tail_beta_to_spx_float": _beta_float(
                    market_tail_return_df["strategy"],
                    market_tail_return_df["benchmark"],
                ),
                "market_tail_mean_return_pct_float": float(market_tail_return_df["strategy"].mean() * 100.0),
                "market_tail_spx_mean_return_pct_float": float(market_tail_return_df["benchmark"].mean() * 100.0),
            }
        )

    return pd.DataFrame(row_dict_list)


def _variant_strategy_name_str(variant_spec_obj: EquityCurveVariantSpec) -> str:
    if len(variant_spec_obj.addition_symbol_tuple) == 0:
        return "strategy_mr_sector_dispersion_ibs_equity_base"
    slug_str = _slug_str("_".join(variant_spec_obj.addition_symbol_tuple))
    return f"strategy_mr_sector_dispersion_ibs_equity_{slug_str}"


def _benchmark_equity_ser(
    pricing_data_df: pd.DataFrame,
    benchmark_symbol_str: str,
    reference_index: pd.Index,
    capital_base_float: float,
) -> pd.Series:
    benchmark_close_ser = pd.to_numeric(
        pricing_data_df[(benchmark_symbol_str, "Close")],
        errors="coerce",
    )
    benchmark_close_ser.index = pd.to_datetime(benchmark_close_ser.index).normalize()
    benchmark_close_ser = benchmark_close_ser.reindex(pd.to_datetime(reference_index).normalize()).dropna()
    if len(benchmark_close_ser) == 0:
        raise RuntimeError(f"No benchmark close data available for {benchmark_symbol_str}.")

    # *** CRITICAL*** Benchmark equity is a post-run visual comparator only.
    # It is normalized from benchmark close prices after backtests are complete
    # and does not alter signals, sizing, fills, or variant inclusion.
    return benchmark_close_ser / float(benchmark_close_ser.iloc[0]) * float(capital_base_float)


def _curve_summary_row_dict(label_str: str, equity_ser: pd.Series, drawdown_ser: pd.Series) -> dict[str, object]:
    clean_equity_ser = pd.to_numeric(equity_ser, errors="coerce").dropna()
    clean_drawdown_ser = pd.to_numeric(drawdown_ser, errors="coerce").dropna()
    if len(clean_equity_ser) == 0:
        raise ValueError(f"Cannot summarize empty equity curve {label_str!r}.")
    return {
        "variant_label_str": label_str,
        "start_date_str": clean_equity_ser.index[0].date().isoformat(),
        "end_date_str": clean_equity_ser.index[-1].date().isoformat(),
        "start_total_value_float": float(clean_equity_ser.iloc[0]),
        "final_total_value_float": float(clean_equity_ser.iloc[-1]),
        "terminal_multiple_float": float(clean_equity_ser.iloc[-1] / clean_equity_ser.iloc[0]),
        "curve_max_drawdown_pct_float": float(clean_drawdown_ser.min() * 100.0),
    }


def _save_line_chart(
    curve_df: pd.DataFrame,
    output_file_path: Path,
    title_str: str,
    ylabel_str: str,
    percent_axis_bool: bool,
    log_scale_bool: bool = False,
) -> None:
    fig_obj, axis_obj = plt.subplots(figsize=(14, 8))
    for column_str in curve_df.columns:
        series_obj = pd.to_numeric(curve_df[column_str], errors="coerce").dropna()
        line_width_float = 2.4 if column_str in {"Base", "Base+KIE+IHI", "Base+KIE+IHI+XLC"} else 1.7
        alpha_float = 0.95 if column_str != "Benchmark $SPX" else 0.75
        axis_obj.plot(
            series_obj.index,
            series_obj.values,
            label=column_str,
            linewidth=line_width_float,
            alpha=alpha_float,
        )

    axis_obj.set_title(title_str)
    axis_obj.set_xlabel("Date")
    axis_obj.set_ylabel(ylabel_str)
    if log_scale_bool:
        axis_obj.set_yscale("log")
    if percent_axis_bool:
        axis_obj.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis_obj.grid(True, alpha=0.25)
    axis_obj.legend(loc="best", fontsize=9)
    fig_obj.autofmt_xdate()
    fig_obj.tight_layout()
    fig_obj.savefig(output_file_path, dpi=160)
    plt.close(fig_obj)


def _write_readme_md(
    output_path: Path,
    manifest_df: pd.DataFrame,
    include_benchmark_bool: bool,
    end_date_str: str | None,
    market_tail_quantile_float: float,
) -> None:
    variant_line_list = [
        f"- `{row_ser['variant_label_str']}`: `{row_ser['symbol_tuple_str']}` - {row_ser['note_str']}"
        for _, row_ser in manifest_df.iterrows()
    ]
    benchmark_note_str = "- Benchmark curve: `$SPX` close-only normalized comparator." if include_benchmark_bool else ""
    end_date_note_str = end_date_str if end_date_str is not None else "latest available data"
    readme_md_str = f"""# Sector Dispersion Equity Curve Comparison

This is a research-only visual comparison of selected sector-dispersion IBS variants.

## Timing And Formula

- Strategy signal timing is unchanged: daily bar `T` -> fill at `Open_(T+1)`.
- Normalized equity: `norm_equity_(j,t) = equity_(j,t) / equity_(j,start)`.
- Drawdown: `drawdown_(j,t) = norm_equity_(j,t) / running_max(norm_equity_j) - 1`.
- Full-period market correlation: `corr(strategy_daily_return, SPX_daily_return)`.
- Market-down correlation: same correlation restricted to `$SPX` daily return `< 0`.
- Market-tail correlation and beta: same diagnostics restricted to the worst `{market_tail_quantile_float:.0%}` `$SPX` daily returns.
- Slippage and commission settings are inherited from `DEFAULT_CONFIG`.
- End date: `{end_date_note_str}`.
{benchmark_note_str}

## Variants

{chr(10).join(variant_line_list)}

## Files

- `equity_curves.csv`: raw total-value curves.
- `normalized_equity_curves.csv`: normalized growth of 1.
- `drawdown_curves.csv`: drawdowns from each curve's own running peak.
- `market_correlation_summary.csv`: full, market-down, and market-tail correlation to `$SPX`.
- `variant_summary.csv`: backtest summary fields plus terminal multiple.
- `normalized_equity_curves.png`: linear normalized equity chart.
- `normalized_equity_curves_log.png`: log-scale normalized equity chart.
- `drawdown_curves.png`: drawdown chart.
"""
    (output_path / "README.md").write_text(readme_md_str, encoding="utf-8")


def run_equity_curve_comparison(
    output_dir_str: str = "results",
    end_date_str: str | None = None,
    show_progress_bool: bool = False,
    include_benchmark_bool: bool = True,
    include_full_universes_bool: bool = False,
    market_tail_quantile_float: float = 0.10,
    variant_spec_tuple: tuple[EquityCurveVariantSpec, ...] = DEFAULT_COMPARISON_VARIANT_SPEC_TUPLE,
) -> Path:
    if variant_spec_tuple == DEFAULT_COMPARISON_VARIANT_SPEC_TUPLE:
        variant_spec_tuple = build_variant_spec_tuple(include_full_universes_bool=include_full_universes_bool)
    manifest_df = build_comparison_manifest_df(variant_spec_tuple=variant_spec_tuple)
    timestamp_str = pd.Timestamp.now().strftime("%Y-%m-%d_%H%M%S")
    output_path = build_research_output_path(
        output_dir=output_dir_str,
        entity_type_str="strategy",
        entity_id_str="strategy_mr_sector_dispersion_ibs",
        analysis_type_str="equity_curve_comparison",
        timestamp_str=timestamp_str,
    )
    output_path.mkdir(parents=True, exist_ok=False)

    added_symbol_tuple = tuple(
        dict.fromkeys(
            symbol_str
            for variant_spec_obj in variant_spec_tuple
            for symbol_str in variant_spec_obj.addition_symbol_tuple
        )
    )
    all_symbol_tuple = tuple(dict.fromkeys(ORIGINAL_SYMBOL_TUPLE + added_symbol_tuple))
    base_config_obj = replace(
        DEFAULT_CONFIG,
        symbol_tuple=all_symbol_tuple,
        universe_name_str="original",
        end_date_str=end_date_str,
    )
    pricing_data_df = get_sector_dispersion_ibs_data(config_obj=base_config_obj)

    equity_curve_dict: dict[str, pd.Series] = {}
    strategy_summary_row_dict_list: list[dict[str, object]] = []

    for variant_spec_obj in variant_spec_tuple:
        symbol_tuple = tuple(dict.fromkeys(ORIGINAL_SYMBOL_TUPLE + variant_spec_obj.addition_symbol_tuple))
        print(f"Running equity curve: {variant_spec_obj.label_str}...", flush=True)
        strategy_obj = _run_strategy_variant(
            strategy_name_str=_variant_strategy_name_str(variant_spec_obj),
            symbol_tuple=symbol_tuple,
            base_config_obj=base_config_obj,
            pricing_data_df=pricing_data_df,
            show_progress_bool=show_progress_bool,
        )
        total_value_ser = pd.to_numeric(strategy_obj.results["total_value"], errors="coerce")
        total_value_ser.index = pd.to_datetime(total_value_ser.index).normalize()
        equity_curve_dict[variant_spec_obj.label_str] = total_value_ser

        summary_row_dict = _strategy_summary_row_dict(
            strategy_obj=strategy_obj,
            variant_kind_str="equity_curve_comparison",
            candidate_symbol_str=",".join(variant_spec_obj.addition_symbol_tuple) or None,
            bucket_str=None,
        )
        summary_row_dict["variant_label_str"] = variant_spec_obj.label_str
        summary_row_dict["addition_tuple_str"] = ",".join(variant_spec_obj.addition_symbol_tuple)
        summary_row_dict["note_str"] = variant_spec_obj.note_str
        strategy_summary_row_dict_list.append(summary_row_dict)

    equity_curve_df = pd.DataFrame(equity_curve_dict).sort_index()

    if include_benchmark_bool:
        benchmark_equity_ser = _benchmark_equity_ser(
            pricing_data_df=pricing_data_df,
            benchmark_symbol_str=base_config_obj.benchmark_symbol_str,
            reference_index=equity_curve_df.index,
            capital_base_float=base_config_obj.capital_base_float,
        )
        equity_curve_df[f"Benchmark {base_config_obj.benchmark_symbol_str}"] = benchmark_equity_ser

    normalized_equity_curve_df = normalize_equity_curve_df(equity_curve_df)
    drawdown_curve_df = compute_drawdown_curve_df(normalized_equity_curve_df)
    benchmark_label_str = f"Benchmark {base_config_obj.benchmark_symbol_str}"
    if include_benchmark_bool:
        market_correlation_summary_df = compute_market_correlation_summary_df(
            equity_curve_df=equity_curve_df,
            benchmark_label_str=benchmark_label_str,
            market_tail_quantile_float=market_tail_quantile_float,
        )
    else:
        market_correlation_summary_df = pd.DataFrame(columns=["variant_label_str"])

    curve_summary_row_dict_list = [
        _curve_summary_row_dict(
            label_str=column_str,
            equity_ser=equity_curve_df[column_str],
            drawdown_ser=drawdown_curve_df[column_str],
        )
        for column_str in equity_curve_df.columns
    ]
    curve_summary_df = pd.DataFrame(curve_summary_row_dict_list)
    strategy_summary_df = pd.DataFrame(strategy_summary_row_dict_list)
    variant_summary_df = strategy_summary_df.merge(
        curve_summary_df,
        on="variant_label_str",
        how="left",
        suffixes=("", "_curve"),
    )
    variant_summary_df = variant_summary_df.merge(
        market_correlation_summary_df,
        on="variant_label_str",
        how="left",
    )

    manifest_df.to_csv(output_path / "comparison_manifest.csv", index=False)
    equity_curve_df.to_csv(output_path / "equity_curves.csv", index_label="date")
    normalized_equity_curve_df.to_csv(output_path / "normalized_equity_curves.csv", index_label="date")
    drawdown_curve_df.to_csv(output_path / "drawdown_curves.csv", index_label="date")
    curve_summary_df.to_csv(output_path / "curve_summary.csv", index=False)
    market_correlation_summary_df.to_csv(output_path / "market_correlation_summary.csv", index=False)
    variant_summary_df.to_csv(output_path / "variant_summary.csv", index=False)

    _save_line_chart(
        curve_df=normalized_equity_curve_df,
        output_file_path=output_path / "normalized_equity_curves.png",
        title_str="Sector Dispersion IBS: Normalized Equity Curves",
        ylabel_str="Growth of 1.0",
        percent_axis_bool=False,
    )
    _save_line_chart(
        curve_df=normalized_equity_curve_df,
        output_file_path=output_path / "normalized_equity_curves_log.png",
        title_str="Sector Dispersion IBS: Normalized Equity Curves (Log Scale)",
        ylabel_str="Growth of 1.0",
        percent_axis_bool=False,
        log_scale_bool=True,
    )
    _save_line_chart(
        curve_df=drawdown_curve_df,
        output_file_path=output_path / "drawdown_curves.png",
        title_str="Sector Dispersion IBS: Drawdown Curves",
        ylabel_str="Drawdown",
        percent_axis_bool=True,
    )

    metadata_dict = {
        "analysis_type_str": "equity_curve_comparison",
        "generated_at_str": pd.Timestamp.now().isoformat(),
        "output_path_str": str(output_path),
        "end_date_str": end_date_str,
        "include_benchmark_bool": bool(include_benchmark_bool),
        "include_full_universes_bool": bool(include_full_universes_bool),
        "variant_count_int": int(len(variant_spec_tuple)),
        "benchmark_symbol_str": base_config_obj.benchmark_symbol_str,
        "market_tail_quantile_float": market_tail_quantile_float,
        "base_symbol_tuple": ORIGINAL_SYMBOL_TUPLE,
        "all_symbol_tuple": all_symbol_tuple,
        "slippage_float": base_config_obj.slippage_float,
        "commission_per_share_float": base_config_obj.commission_per_share_float,
        "commission_minimum_float": base_config_obj.commission_minimum_float,
        "execution_timing_note_str": "Signal from daily bar T fills at Open T+1 through the standard runner.",
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata_dict, indent=2, default=_json_default_obj),
        encoding="utf-8",
    )
    _write_readme_md(
        output_path=output_path,
        manifest_df=manifest_df,
        include_benchmark_bool=include_benchmark_bool,
        end_date_str=end_date_str,
        market_tail_quantile_float=market_tail_quantile_float,
    )

    return output_path


def _parse_args() -> argparse.Namespace:
    parser_obj = argparse.ArgumentParser(
        description="Plot selected equity curves for the sector-dispersion IBS combination finalists."
    )
    parser_obj.add_argument("--output-dir", default="results")
    parser_obj.add_argument("--end-date", default=None)
    parser_obj.add_argument("--show-progress", action="store_true")
    parser_obj.add_argument("--no-benchmark", action="store_true")
    parser_obj.add_argument("--include-full-universes", action="store_true")
    parser_obj.add_argument("--market-tail-quantile", type=float, default=0.10)
    return parser_obj.parse_args()


def main() -> int:
    args_obj = _parse_args()
    output_path = run_equity_curve_comparison(
        output_dir_str=str(args_obj.output_dir),
        end_date_str=args_obj.end_date,
        show_progress_bool=bool(args_obj.show_progress),
        include_benchmark_bool=not bool(args_obj.no_benchmark),
        include_full_universes_bool=bool(args_obj.include_full_universes),
        market_tail_quantile_float=float(args_obj.market_tail_quantile),
    )
    print(f"Saved equity-curve comparison to {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
