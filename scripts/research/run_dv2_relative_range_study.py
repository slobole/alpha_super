"""Research-only DV2 study of lagged Relative Range filters and ranking."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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
from data.norgate_loader import build_index_constituent_matrix
from scripts.research.run_sector_dispersion_marginal_universe_study import (
    _json_default_obj,
    _summary_value_float,
    compute_period_metric_dict,
)
from scripts.research.run_sector_dispersion_short_sleeve_study import (
    _benchmark_return_ser,
    _markdown_table_str,
    _market_metric_dict,
    _normalize_equity_ser,
    _performance_metric_dict,
)
from strategies.dv2.strategy_mr_dv2 import (
    DVO2Strategy,
    default_trade_id_int,
    get_asof_universe_symbol_list,
    get_prices,
)


VARIANT_BASELINE_STR = "baseline_natr_rank"
VARIANT_RANGE_FILTER_STR = "relative_range_filter"
VARIANT_RANGE_RANK_STR = "relative_range_rank"
VARIANT_MODE_TUPLE = (
    VARIANT_BASELINE_STR,
    VARIANT_RANGE_FILTER_STR,
    VARIANT_RANGE_RANK_STR,
)
RANGE_LOOKBACK_DAY_INT = 21
MIN_RELATIVE_RANGE_FLOAT = 1.0
BACKTEST_START_DATE_STR = "2004-01-01"
MARKET_TAIL_QUANTILE_FLOAT = 0.10


def compute_relative_range_feature_df(
    high_price_df: pd.DataFrame,
    low_price_df: pd.DataFrame,
    lookback_day_int: int = RANGE_LOOKBACK_DAY_INT,
) -> pd.DataFrame:
    """Compute Range_T / StdDev(Range_T-1 ... Range_T-L)."""
    if lookback_day_int <= 1:
        raise ValueError("lookback_day_int must be greater than 1.")
    numeric_high_price_df = high_price_df.apply(pd.to_numeric, errors="coerce")
    numeric_low_price_df = low_price_df.apply(pd.to_numeric, errors="coerce")
    valid_range_bool_df = (
        numeric_high_price_df.gt(0.0)
        & numeric_low_price_df.gt(0.0)
        & numeric_high_price_df.gt(numeric_low_price_df)
    )
    log_range_df = np.log(numeric_high_price_df / numeric_low_price_df).where(
        valid_range_bool_df
    )
    # *** CRITICAL*** RelativeRange_T uses today's completed High_T/Low_T only
    # in the numerator. The denominator is shifted and contains exactly the
    # prior L ranges, so orders generated at T still fill only at Open T+1.
    lagged_range_volatility_df = log_range_df.rolling(
        window=int(lookback_day_int),
        min_periods=int(lookback_day_int),
    ).std().shift(1)
    return log_range_df / lagged_range_volatility_df.replace(0.0, np.nan)


class DV2RelativeRangeResearchStrategy(DVO2Strategy):
    """Current DV2 strategy with one controlled Relative Range interpretation."""

    signal_audit_sample_size = 3

    def __init__(
        self,
        name: str,
        benchmarks: list[str],
        variant_mode_str: str,
        capital_base_float: float = 100_000.0,
    ) -> None:
        if variant_mode_str not in VARIANT_MODE_TUPLE:
            raise ValueError(f"variant_mode_str must be one of {VARIANT_MODE_TUPLE}.")
        super().__init__(
            name=name,
            benchmarks=benchmarks,
            capital_base=capital_base_float,
            slippage=0.00025,
            commission_per_share=0.005,
            commission_minimum=1.0,
        )
        self.variant_mode_str = variant_mode_str
        self.candidate_day_count_int = 0
        self.baseline_candidate_count_int = 0
        self.range_pass_candidate_count_int = 0
        self.baseline_candidate_count_list: list[int] = []
        self.candidate_relative_range_list: list[float] = []

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        signal_data_df = super().compute_signals(pricing_data_df)
        tradable_symbol_list = [
            str(symbol_obj)
            for symbol_obj in signal_data_df.columns.get_level_values(0).unique()
            if not str(symbol_obj).startswith("$")
            and (symbol_obj, "High") in signal_data_df.columns
            and (symbol_obj, "Low") in signal_data_df.columns
        ]
        high_price_df = pd.DataFrame(
            {
                symbol_str: signal_data_df[(symbol_str, "High")]
                for symbol_str in tradable_symbol_list
            },
            index=signal_data_df.index,
        )
        low_price_df = pd.DataFrame(
            {
                symbol_str: signal_data_df[(symbol_str, "Low")]
                for symbol_str in tradable_symbol_list
            },
            index=signal_data_df.index,
        )
        relative_range_df = compute_relative_range_feature_df(
            high_price_df=high_price_df,
            low_price_df=low_price_df,
        )
        relative_range_feature_df = relative_range_df.copy()
        relative_range_feature_df.columns = pd.MultiIndex.from_tuples(
            [
                (str(symbol_str), "relative_range_ser")
                for symbol_str in relative_range_feature_df.columns
            ]
        )
        return pd.concat([signal_data_df, relative_range_feature_df], axis=1)

    def _baseline_candidate_df(self, close_row_ser: pd.Series) -> pd.DataFrame:
        candidate_df = close_row_ser.unstack().dropna(
            subset=["dv2", "Close", "sma_200", "p126d_return", "natr"]
        )
        candidate_df = candidate_df[~candidate_df.index.astype(str).str.startswith("$")]
        candidate_df = candidate_df[
            (candidate_df["dv2"] < 10)
            & (candidate_df["Close"] > candidate_df["sma_200"])
            & (candidate_df["p126d_return"] > 0.05)
        ]
        pit_symbol_list = get_asof_universe_symbol_list(
            self.universe_df,
            pd.Timestamp(self.previous_bar),
        )
        return candidate_df[candidate_df.index.isin(pit_symbol_list)]

    def get_opportunities(self, close_row_ser: pd.Series) -> list[str]:
        candidate_df = self._baseline_candidate_df(close_row_ser)
        finite_relative_range_ser = pd.to_numeric(
            candidate_df.get("relative_range_ser"),
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
        range_pass_bool_ser = finite_relative_range_ser.gt(MIN_RELATIVE_RANGE_FLOAT)

        self.candidate_day_count_int += 1
        self.baseline_candidate_count_int += int(len(candidate_df))
        self.range_pass_candidate_count_int += int(range_pass_bool_ser.sum())
        self.baseline_candidate_count_list.append(int(len(candidate_df)))
        self.candidate_relative_range_list.extend(
            finite_relative_range_ser.dropna().astype(float).tolist()
        )

        if self.variant_mode_str == VARIANT_RANGE_FILTER_STR:
            candidate_df = candidate_df.loc[range_pass_bool_ser]
            candidate_df = candidate_df.sort_values("natr", ascending=False)
        elif self.variant_mode_str == VARIANT_RANGE_RANK_STR:
            candidate_df = candidate_df.assign(
                relative_range_rank_value_ser=finite_relative_range_ser
            ).dropna(subset=["relative_range_rank_value_ser"])
            candidate_df = candidate_df.sort_values(
                "relative_range_rank_value_ser",
                ascending=False,
            )
        else:
            candidate_df = candidate_df.sort_values("natr", ascending=False)
        return candidate_df.index.astype(str).tolist()


def _candidate_diagnostic_dict(
    strategy_obj: DV2RelativeRangeResearchStrategy,
) -> dict[str, object]:
    candidate_count_int = int(strategy_obj.baseline_candidate_count_int)
    relative_range_arr = np.asarray(
        strategy_obj.candidate_relative_range_list,
        dtype=float,
    )
    relative_range_arr = relative_range_arr[np.isfinite(relative_range_arr)]
    return {
        "candidate_day_count_int": int(strategy_obj.candidate_day_count_int),
        "baseline_candidate_count_int": candidate_count_int,
        "range_pass_candidate_count_int": int(
            strategy_obj.range_pass_candidate_count_int
        ),
        "range_pass_candidate_pct_float": (
            np.nan
            if candidate_count_int == 0
            else float(strategy_obj.range_pass_candidate_count_int / candidate_count_int * 100.0)
        ),
        "average_candidates_per_day_float": (
            np.nan
            if strategy_obj.candidate_day_count_int == 0
            else float(candidate_count_int / strategy_obj.candidate_day_count_int)
        ),
        "candidate_days_above_10_pct_float": (
            np.nan
            if len(strategy_obj.baseline_candidate_count_list) == 0
            else float(
                np.mean(np.asarray(strategy_obj.baseline_candidate_count_list) > 10)
                * 100.0
            )
        ),
        "candidate_relative_range_median_float": (
            np.nan if len(relative_range_arr) == 0 else float(np.median(relative_range_arr))
        ),
        "candidate_relative_range_p90_float": (
            np.nan
            if len(relative_range_arr) == 0
            else float(np.quantile(relative_range_arr, 0.90))
        ),
    }


def _exposure_metric_dict(
    strategy_obj: DV2RelativeRangeResearchStrategy,
) -> dict[str, object]:
    realized_weight_df = getattr(strategy_obj, "realized_weight_df", pd.DataFrame())
    if realized_weight_df is None or len(realized_weight_df) == 0:
        return {
            "avg_position_count_float": np.nan,
            "avg_gross_exposure_pct_float": np.nan,
            "active_day_pct_float": np.nan,
        }
    weight_df = realized_weight_df.copy()
    weight_df.columns = [str(column_obj) for column_obj in weight_df.columns]
    tradable_column_list = [
        column_str
        for column_str in weight_df.columns
        if not column_str.startswith("$") and column_str.lower() != "cash"
    ]
    tradable_weight_df = weight_df[tradable_column_list].apply(
        pd.to_numeric,
        errors="coerce",
    ).fillna(0.0)
    gross_exposure_ser = tradable_weight_df.abs().sum(axis=1)
    position_count_ser = tradable_weight_df.abs().gt(1e-9).sum(axis=1)
    return {
        "avg_position_count_float": float(position_count_ser.mean()),
        "avg_gross_exposure_pct_float": float(gross_exposure_ser.mean() * 100.0),
        "active_day_pct_float": float(gross_exposure_ser.gt(1e-9).mean() * 100.0),
    }


def _summary_row_dict(
    strategy_obj: DV2RelativeRangeResearchStrategy,
    benchmark_return_ser: pd.Series,
) -> dict[str, object]:
    total_value_ser = strategy_obj.results["total_value"]
    row_dict: dict[str, object] = {
        "variant_mode_str": strategy_obj.variant_mode_str,
        "strategy_name_str": strategy_obj.name,
        "universe_str": "PIT S&P 500",
        "turnover_ann_pct_float": _summary_value_float(
            strategy_obj.summary,
            "Turnover (Ann.) [%]",
        ),
        "cost_drag_ann_pct_float": _summary_value_float(
            strategy_obj.summary,
            "Cost Drag (Ann.) [%]",
        ),
        "exposure_time_pct_float": _summary_value_float(
            strategy_obj.summary,
            "Exposure Time [%]",
        ),
        "trade_count_int": int(len(strategy_obj.get_transactions())),
    }
    row_dict.update(_performance_metric_dict(total_value_ser))
    ann_return_float = float(row_dict["ann_return_pct_float"])
    max_drawdown_float = float(row_dict["max_drawdown_pct_float"])
    row_dict["mar_float"] = (
        np.nan
        if max_drawdown_float == 0.0
        else ann_return_float / abs(max_drawdown_float)
    )
    row_dict.update(
        compute_period_metric_dict(
            total_value_ser=total_value_ser,
            start_ts=pd.Timestamp("2004-01-01"),
            end_ts=pd.Timestamp("2011-12-31"),
            prefix_str="period_2004_2011",
        )
    )
    row_dict.update(
        compute_period_metric_dict(
            total_value_ser=total_value_ser,
            start_ts=pd.Timestamp("2012-01-01"),
            end_ts=pd.Timestamp("2019-12-31"),
            prefix_str="period_2012_2019",
        )
    )
    row_dict.update(
        compute_period_metric_dict(
            total_value_ser=total_value_ser,
            start_ts=pd.Timestamp("2020-01-01"),
            end_ts=None,
            prefix_str="period_2020_plus",
        )
    )
    row_dict.update(
        _market_metric_dict(
            total_value_ser,
            benchmark_return_ser,
            MARKET_TAIL_QUANTILE_FLOAT,
        )
    )
    row_dict.update(_exposure_metric_dict(strategy_obj))
    row_dict.update(_candidate_diagnostic_dict(strategy_obj))
    return row_dict


def _save_equity_chart(output_path: Path, equity_df: pd.DataFrame) -> None:
    fig_obj, axis_obj = plt.subplots(figsize=(14, 8))
    for column_str in equity_df.columns:
        normalized_equity_ser = _normalize_equity_ser(equity_df[column_str])
        axis_obj.plot(
            normalized_equity_ser.index,
            normalized_equity_ser.values,
            label=column_str,
            linewidth=1.6,
        )
    axis_obj.set_title("DV2 Relative Range Variants")
    axis_obj.set_xlabel("Date")
    axis_obj.set_ylabel("Growth of 1.0")
    axis_obj.grid(True, alpha=0.25)
    axis_obj.legend(loc="best")
    fig_obj.tight_layout()
    fig_obj.savefig(output_path / "equity_curves.png", dpi=170)
    plt.close(fig_obj)


def _write_recommendations_md(output_path: Path, summary_df: pd.DataFrame) -> None:
    display_column_list = [
        "variant_mode_str",
        "start_date_str",
        "end_date_str",
        "ann_return_pct_float",
        "volatility_ann_pct_float",
        "sharpe_float",
        "max_drawdown_pct_float",
        "mar_float",
        "turnover_ann_pct_float",
        "cost_drag_ann_pct_float",
        "avg_position_count_float",
        "avg_gross_exposure_pct_float",
        "range_pass_candidate_pct_float",
        "market_tail_mean_return_pct_float",
        "market_tail_beta_to_spx_float",
    ]
    subperiod_column_list = [
        "variant_mode_str",
        "period_2004_2011_ann_return_pct_float",
        "period_2004_2011_sharpe_float",
        "period_2004_2011_max_drawdown_pct_float",
        "period_2012_2019_ann_return_pct_float",
        "period_2012_2019_sharpe_float",
        "period_2012_2019_max_drawdown_pct_float",
        "period_2020_plus_ann_return_pct_float",
        "period_2020_plus_sharpe_float",
        "period_2020_plus_max_drawdown_pct_float",
    ]
    recommendations_md_str = f"""# DV2 Relative Range Study

## Scope

- Research-only; the released DV2 strategy and live wiring are unchanged.
- Local search count: `3` predeclared variants.
- Broader DV2 strategy-family search is larger and includes prior timing, VIX, sizing, and weekly studies.
- Universe: point-in-time S&P 500 membership.
- Execution: completed bar `T` signal, market fill at `Open T+1`.
- Costs: 2.5 bps slippage per side, `$0.005/share`, `$1` minimum.
- Relative Range: `ln(High_T / Low_T) / StdDev(Range_T-1 ... Range_T-21)`.

## Full-Sample Results

{_markdown_table_str(summary_df[display_column_list])}

## Descriptive Subperiods

{_markdown_table_str(summary_df[subperiod_column_list])}

## Interpretation Reminder

Subperiods and market-tail rows are descriptive diagnostics, not untouched out-of-sample evidence. The Relative Range threshold came from the separate ETF study and is not independently validated for stocks. A lower drawdown caused only by lower exposure is not evidence of better alpha.
"""
    (output_path / "recommendations.md").write_text(
        recommendations_md_str,
        encoding="utf-8",
    )


def run_dv2_relative_range_study(
    output_dir_str: str = "results",
    end_date_str: str | None = None,
    show_progress_bool: bool = False,
) -> Path:
    timestamp_str = pd.Timestamp.now().strftime("%Y-%m-%d_%H%M%S")
    output_path = build_research_output_path(
        output_dir=output_dir_str,
        entity_type_str="strategy",
        entity_id_str="strategy_mr_dv2",
        analysis_type_str="relative_range_study",
        timestamp_str=timestamp_str,
    )
    output_path.mkdir(parents=True, exist_ok=False)

    benchmark_list = ["$SPX"]
    symbol_list, universe_df = build_index_constituent_matrix(indexname="S&P 500")
    pricing_data_df = get_prices(
        symbol_list,
        benchmark_list,
        start_date="1998-01-01",
        end_date=end_date_str,
    )
    calendar_idx = pricing_data_df.index[
        pricing_data_df.index >= pd.Timestamp(BACKTEST_START_DATE_STR)
    ]
    benchmark_return_ser = _benchmark_return_ser(pricing_data_df, "$SPX")

    summary_row_list: list[dict[str, object]] = []
    equity_dict: dict[str, pd.Series] = {}
    for variant_mode_str in VARIANT_MODE_TUPLE:
        print(f"Running {variant_mode_str}...", flush=True)
        strategy_obj = DV2RelativeRangeResearchStrategy(
            name=f"strategy_mr_dv2_{variant_mode_str}",
            benchmarks=benchmark_list,
            variant_mode_str=variant_mode_str,
        )
        strategy_obj.universe_df = universe_df
        strategy_obj.trade_id = 0
        strategy_obj.current_trade = defaultdict(default_trade_id_int)
        run_daily(
            strategy_obj,
            pricing_data_df,
            calendar=calendar_idx,
            show_progress=show_progress_bool,
            show_signal_progress_bool=show_progress_bool,
            audit_override_bool=True,
        )
        summary_row_list.append(
            _summary_row_dict(
                strategy_obj=strategy_obj,
                benchmark_return_ser=benchmark_return_ser,
            )
        )
        equity_dict[variant_mode_str] = strategy_obj.results["total_value"]

    summary_df = pd.DataFrame(summary_row_list)
    equity_df = pd.DataFrame(equity_dict).sort_index()
    summary_df.to_csv(output_path / "summary.csv", index=False)
    equity_df.to_csv(output_path / "equity_curves.csv", index_label="date")
    _save_equity_chart(output_path=output_path, equity_df=equity_df)
    _write_recommendations_md(output_path=output_path, summary_df=summary_df)

    baseline_row_ser = summary_df.loc[
        summary_df["variant_mode_str"].eq(VARIANT_BASELINE_STR)
    ].iloc[0]
    candidate_distribution_dict = {
        "baseline_candidate_count_int": int(
            baseline_row_ser["baseline_candidate_count_int"]
        ),
        "range_pass_candidate_pct_float": float(
            baseline_row_ser["range_pass_candidate_pct_float"]
        ),
        "candidate_relative_range_median_float": float(
            baseline_row_ser["candidate_relative_range_median_float"]
        ),
        "candidate_relative_range_p90_float": float(
            baseline_row_ser["candidate_relative_range_p90_float"]
        ),
    }
    metadata_dict = {
        "analysis_type_str": "relative_range_study",
        "generated_at_str": pd.Timestamp.now().isoformat(),
        "output_path_str": str(output_path.resolve()),
        "local_variant_count_int": len(VARIANT_MODE_TUPLE),
        "broader_strategy_family_search_note_str": (
            "The local count excludes prior DV2 timing, VIX, sizing, and weekly studies."
        ),
        "variant_mode_tuple": VARIANT_MODE_TUPLE,
        "backtest_start_date_str": BACKTEST_START_DATE_STR,
        "end_date_str": end_date_str,
        "universe_str": "Norgate point-in-time S&P 500 Current & Past",
        "stock_adjustment_str": "CAPITALSPECIAL",
        "benchmark_symbol_str": "$SPX",
        "benchmark_adjustment_str": "TOTALRETURN",
        "range_lookback_day_int": RANGE_LOOKBACK_DAY_INT,
        "min_relative_range_float": MIN_RELATIVE_RANGE_FLOAT,
        "candidate_distribution_dict": candidate_distribution_dict,
        "slippage_float": 0.00025,
        "commission_per_share_float": 0.005,
        "commission_minimum_float": 1.0,
        "execution_timing_note_str": "Completed bar T signal; market fill at Open T+1.",
        "tail_diagnostic_note_str": (
            "Market-tail metrics use the aligned sample's ex-post worst 10% SPX return threshold."
        ),
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata_dict, indent=2, default=_json_default_obj),
        encoding="utf-8",
    )
    print(f"Saved DV2 Relative Range study to {output_path}", flush=True)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser_obj = argparse.ArgumentParser(
        description="Run the research-only DV2 Relative Range study."
    )
    parser_obj.add_argument("--output-dir", default="results")
    parser_obj.add_argument("--end-date", default=None)
    parser_obj.add_argument("--show-progress", action="store_true")
    return parser_obj.parse_args()


def main() -> int:
    args_obj = _parse_args()
    run_dv2_relative_range_study(
        output_dir_str=str(args_obj.output_dir),
        end_date_str=args_obj.end_date,
        show_progress_bool=bool(args_obj.show_progress),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
