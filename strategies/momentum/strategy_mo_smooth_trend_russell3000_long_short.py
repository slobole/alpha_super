"""
Research-only Russell 3000 smooth-trend long/short strategy.

Core formulas
-------------
For stock i on month-end decision date t:

    formation prices
        = Close_{i,t-252:t-21}

    r_{i,j}
        = Close_{i,j} / Close_{i,j-1} - 1

    S_{i,k}
        = sum_{j=1:k} r_{i,j}

    S_{i,k}
        = alpha_i + beta_i * k + epsilon_{i,k}

Option A selection:

    eligible_t
        = active PIT members with Close_t > 1.00

    RQ5_t
        = eligible_t names in the highest trend_r2 quintile

    SQ5_t
        = highest slope quintile inside RQ5_t

    RQ1_t
        = eligible_t names in the lowest trend_r2 quintile

    SQ1_t
        = lowest slope quintile inside RQ1_t

    long_t
        = top n names from RQ5/SQ5 by slope

    short_t
        = bottom n names from RQ1/SQ1 by slope

No explicit slope sign gate is applied. That matches the conditional corner
sort: in an unusually one-sided market, the lowest-slope corner can contain
weak uptrends rather than outright downtrends.

Execution mapping:

    decision_date_t
        = actual last tradable close of month t

    execution_date_t
        = next tradable open after decision_date_t under the Vanilla engine
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from data.norgate_loader import build_index_constituent_matrix, load_raw_prices
from strategies.momentum.strategy_mo_atr_normalized_ndx import (
    audit_pit_universe_df,
    get_asof_universe_membership_ser,
    map_month_end_decision_dates_to_rebalance_schedule_df,
)
from strategies.momentum.strategy_mo_smooth_trend_long_sp500 import (
    compute_smooth_trend_signal_tables,
    get_trend_r2_field_str,
    get_trend_slope_field_str,
)


MAX_LONG_POSITIONS_INT = 20
MAX_SHORT_POSITIONS_INT = 20
LONG_GROSS_EXPOSURE_FLOAT = 1.0
SHORT_GROSS_EXPOSURE_FLOAT = 1.0


def default_trade_id_int() -> int:
    return -1


@dataclass(frozen=True)
class SmoothTrendRussell3000LongShortConfig:
    variant_key_str: str = "russell3000_option_a_n20_long_short_close_gt_1"
    indexname_str: str = "Russell 3000"
    benchmark_list: tuple[str, ...] = ("$RUA",)
    history_start_date_str: str = "1998-01-01"
    backtest_start_date_str: str = "2000-01-01"
    end_date_str: str | None = None
    lookback_trading_day_int: int = 252
    skip_trading_day_int: int = 21
    quintile_count_int: int = 5
    max_long_positions_int: int = MAX_LONG_POSITIONS_INT
    max_short_positions_int: int = MAX_SHORT_POSITIONS_INT
    long_gross_exposure_float: float = LONG_GROSS_EXPOSURE_FLOAT
    short_gross_exposure_float: float = SHORT_GROSS_EXPOSURE_FLOAT
    capital_base_float: float = 100_000.0
    slippage_float: float = 0.00025
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0
    minimum_close_price_float: float = 1.0

    def __post_init__(self) -> None:
        if not self.variant_key_str:
            raise ValueError("variant_key_str must not be empty.")
        if not self.indexname_str:
            raise ValueError("indexname_str must not be empty.")
        if len(self.benchmark_list) == 0:
            raise ValueError("benchmark_list must not be empty.")
        if pd.Timestamp(self.history_start_date_str) >= pd.Timestamp(self.backtest_start_date_str):
            raise ValueError("history_start_date_str must be earlier than backtest_start_date_str.")
        if self.lookback_trading_day_int <= self.skip_trading_day_int + 2:
            raise ValueError("lookback_trading_day_int must exceed skip_trading_day_int by more than 2.")
        if self.quintile_count_int <= 1:
            raise ValueError("quintile_count_int must be greater than 1.")
        if self.max_long_positions_int <= 0:
            raise ValueError("max_long_positions_int must be positive.")
        if self.max_short_positions_int < 0:
            raise ValueError("max_short_positions_int must be non-negative.")
        if self.long_gross_exposure_float < 0.0:
            raise ValueError("long_gross_exposure_float must be non-negative.")
        if self.short_gross_exposure_float < 0.0:
            raise ValueError("short_gross_exposure_float must be non-negative.")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if self.slippage_float < 0.0:
            raise ValueError("slippage_float must be non-negative.")
        if self.commission_per_share_float < 0.0:
            raise ValueError("commission_per_share_float must be non-negative.")
        if self.commission_minimum_float < 0.0:
            raise ValueError("commission_minimum_float must be non-negative.")
        if self.minimum_close_price_float < 0.0:
            raise ValueError("minimum_close_price_float must be non-negative.")


DEFAULT_CONFIG = SmoothTrendRussell3000LongShortConfig()


__all__ = [
    "DEFAULT_CONFIG",
    "LONG_GROSS_EXPOSURE_FLOAT",
    "MAX_LONG_POSITIONS_INT",
    "MAX_SHORT_POSITIONS_INT",
    "SHORT_GROSS_EXPOSURE_FLOAT",
    "SmoothTrendRussell3000LongShortConfig",
    "SmoothTrendRussell3000LongShortStrategy",
    "get_smooth_trend_russell3000_long_short_data",
    "run_variant",
]


def _ranked_fraction_df(
    candidate_feature_df: pd.DataFrame,
    score_column_str: str,
    quintile_count_int: int,
    highest_bool: bool,
) -> pd.DataFrame:
    if len(candidate_feature_df) == 0:
        return candidate_feature_df.copy()

    selected_count_int = max(1, int(np.ceil(len(candidate_feature_df) / float(quintile_count_int))))
    ranked_feature_df = candidate_feature_df.sort_values(
        by=[score_column_str, "symbol_str"],
        ascending=[not highest_bool, True],
        kind="mergesort",
    )
    return ranked_feature_df.iloc[:selected_count_int].copy()


def _build_side_selection_df(
    corner_feature_df: pd.DataFrame,
    side_str: str,
    max_positions_int: int,
    highest_slope_bool: bool,
) -> pd.DataFrame:
    if len(corner_feature_df) == 0 or max_positions_int <= 0:
        return pd.DataFrame(
            columns=[
                "side_str",
                "rank_int",
                "symbol_str",
                "decision_close_price_float",
                "trend_slope_float",
                "trend_r2_float",
            ]
        )

    sorted_feature_df = corner_feature_df.sort_values(
        by=["trend_slope_float", "trend_r2_float", "symbol_str"],
        ascending=[not highest_slope_bool, not highest_slope_bool, True],
        kind="mergesort",
    ).head(int(max_positions_int))
    selected_feature_df = sorted_feature_df.loc[
        :,
        ["symbol_str", "decision_close_price_float", "trend_slope_float", "trend_r2_float"],
    ].copy()
    selected_feature_df.insert(0, "rank_int", np.arange(1, len(selected_feature_df) + 1))
    selected_feature_df.insert(0, "side_str", side_str)
    return selected_feature_df.reset_index(drop=True)


def get_smooth_trend_russell3000_long_short_data(
    config: SmoothTrendRussell3000LongShortConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _, raw_universe_df = build_index_constituent_matrix(indexname=config.indexname_str)

    history_start_ts = pd.Timestamp(config.history_start_date_str)
    backtest_start_ts = pd.Timestamp(config.backtest_start_date_str)
    filtered_universe_df = raw_universe_df.loc[raw_universe_df.index >= history_start_ts].copy()
    active_universe_df = filtered_universe_df.loc[filtered_universe_df.index >= backtest_start_ts].copy()
    if config.end_date_str is not None:
        active_universe_df = active_universe_df.loc[active_universe_df.index <= pd.Timestamp(config.end_date_str)]

    active_symbol_list = active_universe_df.columns[
        active_universe_df.sum(axis=0) > 0
    ].astype(str).tolist()
    if len(active_symbol_list) == 0:
        raise RuntimeError(f"No active {config.indexname_str} symbols were found for the requested window.")

    pricing_data_df = load_raw_prices(
        symbols=active_symbol_list,
        benchmarks=list(config.benchmark_list),
        start_date=config.history_start_date_str,
        end_date=config.end_date_str,
    )
    loaded_symbol_list = [
        symbol_str
        for symbol_str in active_symbol_list
        if symbol_str in pricing_data_df.columns.get_level_values(0)
    ]
    audited_universe_df = audit_pit_universe_df(
        universe_df=filtered_universe_df,
        execution_index=pricing_data_df.index,
        tradeable_symbol_list=loaded_symbol_list,
    )

    keep_symbol_set = set(audited_universe_df.columns.astype(str).tolist() + list(config.benchmark_list))
    pricing_data_df = pricing_data_df.loc[
        :,
        pricing_data_df.columns.get_level_values(0).isin(keep_symbol_set),
    ].sort_index()

    price_close_df = pd.DataFrame(
        {
            symbol_str: pricing_data_df[(symbol_str, "Close")]
            for symbol_str in audited_universe_df.columns.astype(str).tolist()
        },
        index=pricing_data_df.index,
    ).astype(float)
    monthly_decision_close_df, trend_slope_df, trend_r2_df = compute_smooth_trend_signal_tables(
        price_close_df=price_close_df,
        lookback_trading_day_int=config.lookback_trading_day_int,
        skip_trading_day_int=config.skip_trading_day_int,
    )
    rebalance_schedule_df = map_month_end_decision_dates_to_rebalance_schedule_df(
        decision_date_index=pd.DatetimeIndex(monthly_decision_close_df.index),
        execution_index=pricing_data_df.index,
    )
    return pricing_data_df, audited_universe_df, rebalance_schedule_df, trend_slope_df, trend_r2_df


class SmoothTrendRussell3000LongShortStrategy(Strategy):
    """
    Research-only monthly Russell 3000 Option A long/short selector.

    The strategy sorts PIT members by R2 first, then by slope inside the
    smoothness bucket. It holds the RQ5/SQ5 corner long and the RQ1/SQ1 corner
    short with fixed n=20 per leg by default.
    """

    enable_signal_audit = True
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        rebalance_schedule_df: pd.DataFrame,
        config: SmoothTrendRussell3000LongShortConfig = DEFAULT_CONFIG,
        precomputed_trend_slope_df: pd.DataFrame | None = None,
        precomputed_trend_r2_df: pd.DataFrame | None = None,
    ):
        super().__init__(
            name=name,
            benchmarks=list(benchmarks),
            capital_base=config.capital_base_float,
            slippage=config.slippage_float,
            commission_per_share=config.commission_per_share_float,
            commission_minimum=config.commission_minimum_float,
        )

        if len(rebalance_schedule_df) == 0:
            raise ValueError("rebalance_schedule_df must not be empty.")
        if "decision_date_ts" not in rebalance_schedule_df.columns:
            raise ValueError("rebalance_schedule_df must contain decision_date_ts.")

        self.rebalance_schedule_df = rebalance_schedule_df.copy().sort_index()
        self.config = config
        self.trade_id_int = 0
        self.current_trade_map: defaultdict[str, int] = defaultdict(default_trade_id_int)
        self.universe_df: pd.DataFrame | None = None
        self.rebalance_selection_row_list: list[dict[str, object]] = []
        self.rebalance_selection_df = pd.DataFrame()
        if (precomputed_trend_slope_df is None) != (precomputed_trend_r2_df is None):
            raise ValueError("precomputed_trend_slope_df and precomputed_trend_r2_df must be provided together.")
        self.precomputed_trend_slope_df = (
            None if precomputed_trend_slope_df is None else precomputed_trend_slope_df.copy()
        )
        self.precomputed_trend_r2_df = (
            None if precomputed_trend_r2_df is None else precomputed_trend_r2_df.copy()
        )

    @property
    def trend_slope_field_str(self) -> str:
        return get_trend_slope_field_str(
            lookback_trading_day_int=self.config.lookback_trading_day_int,
            skip_trading_day_int=self.config.skip_trading_day_int,
        )

    @property
    def trend_r2_field_str(self) -> str:
        return get_trend_r2_field_str(
            lookback_trading_day_int=self.config.lookback_trading_day_int,
            skip_trading_day_int=self.config.skip_trading_day_int,
        )

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        signal_data_df = pricing_data.copy()
        benchmark_symbol_set = {str(symbol_str) for symbol_str in self._benchmarks}
        tradeable_symbol_list = [
            str(symbol_str)
            for symbol_str in signal_data_df.columns.get_level_values(0).unique()
            if str(symbol_str) not in benchmark_symbol_set
        ]
        if len(tradeable_symbol_list) == 0:
            raise RuntimeError("No tradeable stock symbols were found in pricing_data.")

        price_close_df = pd.DataFrame(
            {symbol_str: signal_data_df[(symbol_str, "Close")] for symbol_str in tradeable_symbol_list},
            index=signal_data_df.index,
        ).astype(float)
        if self.precomputed_trend_slope_df is None or self.precomputed_trend_r2_df is None:
            _monthly_decision_close_df, trend_slope_df, trend_r2_df = compute_smooth_trend_signal_tables(
                price_close_df=price_close_df,
                lookback_trading_day_int=self.config.lookback_trading_day_int,
                skip_trading_day_int=self.config.skip_trading_day_int,
            )
        else:
            trend_slope_df = self.precomputed_trend_slope_df.reindex(columns=tradeable_symbol_list)
            trend_r2_df = self.precomputed_trend_r2_df.reindex(columns=tradeable_symbol_list)

        feature_frame_list: list[pd.DataFrame] = []
        feature_map: dict[str, pd.DataFrame] = {
            self.trend_slope_field_str: trend_slope_df.reindex(signal_data_df.index),
            self.trend_r2_field_str: trend_r2_df.reindex(signal_data_df.index),
        }
        for field_str, field_df in feature_map.items():
            feature_df = field_df.copy()
            feature_df.columns = pd.MultiIndex.from_tuples(
                [(symbol_str, field_str) for symbol_str in feature_df.columns]
            )
            feature_frame_list.append(feature_df)

        return pd.concat([signal_data_df] + feature_frame_list, axis=1)

    def get_selection_df(self, close_row_ser: pd.Series) -> pd.DataFrame:
        if self.universe_df is None:
            raise RuntimeError("universe_df must be set before monthly rebalances.")

        candidate_feature_df = close_row_ser.unstack()
        required_field_list = ["Close", self.trend_slope_field_str, self.trend_r2_field_str]
        if any(field_str not in candidate_feature_df.columns for field_str in required_field_list):
            return pd.DataFrame()

        universe_member_ser = get_asof_universe_membership_ser(
            self.universe_df,
            pd.Timestamp(self.previous_bar),
        )
        active_symbol_list = universe_member_ser[universe_member_ser == 1].index.astype(str).tolist()
        candidate_feature_df = candidate_feature_df[candidate_feature_df.index.isin(active_symbol_list)].copy()
        if len(candidate_feature_df) == 0:
            return pd.DataFrame()

        candidate_feature_df = candidate_feature_df.assign(
            decision_close_price_float=pd.to_numeric(
                candidate_feature_df["Close"],
                errors="coerce",
            ),
            trend_slope_float=pd.to_numeric(
                candidate_feature_df[self.trend_slope_field_str],
                errors="coerce",
            ),
            trend_r2_float=pd.to_numeric(
                candidate_feature_df[self.trend_r2_field_str],
                errors="coerce",
            ),
            symbol_str=candidate_feature_df.index.astype(str),
        )
        finite_mask_vec = np.isfinite(
            candidate_feature_df[
                ["decision_close_price_float", "trend_slope_float", "trend_r2_float"]
            ].to_numpy(dtype=float)
        ).all(axis=1)
        # *** CRITICAL*** The price eligibility filter uses the same
        # previous_bar decision close as the monthly signal. It must not use
        # current_bar open or any post-decision price.
        price_mask_vec = (
            candidate_feature_df["decision_close_price_float"].to_numpy(dtype=float)
            > float(self.config.minimum_close_price_float)
        )
        candidate_feature_df = candidate_feature_df.loc[finite_mask_vec & price_mask_vec].copy()
        if len(candidate_feature_df) == 0:
            return pd.DataFrame()

        high_r2_bucket_df = _ranked_fraction_df(
            candidate_feature_df=candidate_feature_df,
            score_column_str="trend_r2_float",
            quintile_count_int=self.config.quintile_count_int,
            highest_bool=True,
        )
        low_r2_bucket_df = _ranked_fraction_df(
            candidate_feature_df=candidate_feature_df,
            score_column_str="trend_r2_float",
            quintile_count_int=self.config.quintile_count_int,
            highest_bool=False,
        )
        long_corner_df = _ranked_fraction_df(
            candidate_feature_df=high_r2_bucket_df,
            score_column_str="trend_slope_float",
            quintile_count_int=self.config.quintile_count_int,
            highest_bool=True,
        )
        long_selection_df = _build_side_selection_df(
            corner_feature_df=long_corner_df,
            side_str="long",
            max_positions_int=self.config.max_long_positions_int,
            highest_slope_bool=True,
        )
        short_corner_df = pd.DataFrame()
        if self.config.max_short_positions_int > 0 and self.config.short_gross_exposure_float > 0.0:
            short_corner_df = _ranked_fraction_df(
                candidate_feature_df=low_r2_bucket_df,
                score_column_str="trend_slope_float",
                quintile_count_int=self.config.quintile_count_int,
                highest_bool=False,
            )
        short_selection_df = _build_side_selection_df(
            corner_feature_df=short_corner_df,
            side_str="short",
            max_positions_int=self.config.max_short_positions_int,
            highest_slope_bool=False,
        )
        selection_frame_list = [
            selection_df
            for selection_df in (long_selection_df, short_selection_df)
            if len(selection_df) > 0
        ]
        if len(selection_frame_list) == 0:
            return pd.DataFrame()
        selection_df = pd.concat(selection_frame_list, ignore_index=True)
        if len(selection_df) == 0:
            return selection_df

        long_count_int = int((selection_df["side_str"] == "long").sum())
        short_count_int = int((selection_df["side_str"] == "short").sum())
        selection_df["target_weight_float"] = 0.0
        if long_count_int > 0 and self.config.long_gross_exposure_float > 0.0:
            selection_df.loc[selection_df["side_str"] == "long", "target_weight_float"] = (
                self.config.long_gross_exposure_float / float(long_count_int)
            )
        if short_count_int > 0 and self.config.short_gross_exposure_float > 0.0:
            selection_df.loc[selection_df["side_str"] == "short", "target_weight_float"] = (
                -self.config.short_gross_exposure_float / float(short_count_int)
            )
        return selection_df

    def get_target_weight_ser(self, close_row_ser: pd.Series) -> pd.Series:
        selection_df = self.get_selection_df(close_row_ser=close_row_ser)
        if len(selection_df) == 0:
            return pd.Series(dtype=float)
        target_weight_ser = pd.Series(
            selection_df["target_weight_float"].to_numpy(dtype=float),
            index=selection_df["symbol_str"].astype(str),
            dtype=float,
        )
        return target_weight_ser.sort_index()

    def iterate(self, data: pd.DataFrame, close: pd.Series, open_prices: pd.Series):
        if close is None or data is None:
            return
        if self.current_bar not in self.rebalance_schedule_df.index:
            return

        decision_date_ts = pd.Timestamp(self.rebalance_schedule_df.loc[self.current_bar, "decision_date_ts"])
        # *** CRITICAL*** The scheduled month-end decision close must equal
        # previous_bar exactly. Vanilla then fills all target orders at the
        # current_bar open; same-bar decision or fill changes the strategy.
        if pd.Timestamp(self.previous_bar) != decision_date_ts:
            raise RuntimeError(
                f"Schedule misalignment on {self.current_bar}: "
                f"decision_date_ts={decision_date_ts}, previous_bar={self.previous_bar}."
            )

        selection_df = self.get_selection_df(close_row_ser=close)
        target_weight_ser = pd.Series(dtype=float)
        if len(selection_df) > 0:
            selection_record_df = selection_df.copy()
            selection_record_df.insert(0, "execution_date_ts", pd.Timestamp(self.current_bar))
            selection_record_df.insert(0, "decision_date_ts", decision_date_ts)
            self.rebalance_selection_row_list.extend(selection_record_df.to_dict("records"))
            target_weight_ser = pd.Series(
                selection_df["target_weight_float"].to_numpy(dtype=float),
                index=selection_df["symbol_str"].astype(str),
                dtype=float,
            ).sort_index()

        target_symbol_set = set(target_weight_ser.index.astype(str).tolist())
        current_position_ser = self.get_positions()
        active_position_ser = current_position_ser[current_position_ser != 0]
        for symbol_str in active_position_ser.index.astype(str):
            if symbol_str in target_symbol_set:
                continue
            self.order_target_value(
                symbol_str,
                0.0,
                trade_id=self.current_trade_map[symbol_str],
            )

        for symbol_str, target_weight_float in target_weight_ser.items():
            current_share_float = float(current_position_ser.get(symbol_str, 0.0))
            if current_share_float == 0.0:
                self.trade_id_int += 1
                self.current_trade_map[symbol_str] = self.trade_id_int

            self.order_target_percent(
                symbol_str,
                float(target_weight_float),
                trade_id=self.current_trade_map[symbol_str],
            )

    def finalize(self, current_data: pd.DataFrame):
        self.rebalance_selection_df = pd.DataFrame(self.rebalance_selection_row_list)


def _write_assumptions_md(
    output_path: Path,
    strategy: SmoothTrendRussell3000LongShortStrategy,
) -> None:
    config = strategy.config
    assumption_md_str = f"""# Smooth Trend Russell 3000 Long/Short Assumptions

- Research-only strategy; no live/release wiring.
- Universe: `{config.indexname_str}` point-in-time membership through Norgate.
- Benchmark list: `{list(config.benchmark_list)}`.
- Stock price basis: Norgate `CAPITALSPECIAL` OHLC loaded through repo `load_raw_prices`.
- Decision cadence: actual last tradable close of each month.
- Execution cadence: next tradable open after the decision close under the Vanilla engine.
- Formation window: `Close_(t-{config.lookback_trading_day_int})` through `Close_(t-{config.skip_trading_day_int})`.
- Signal math: regress cumulative simple returns `S_k = sum(r_1..r_k)` on time `k`; use slope as direction and unsigned `R2` as smoothness.
- Price eligibility: stock `Close_T > {config.minimum_close_price_float:.2f}` at the decision close, using the same `CAPITALSPECIAL` price basis.
- Sort: `R2` quintile first, then slope quintile within the selected `R2` bucket.
- Long leg: top `{config.max_long_positions_int}` names from RQ5/SQ5 by slope.
- Short leg: bottom `{config.max_short_positions_int}` names from RQ1/SQ1 by slope.
- No explicit positive/negative slope gate is applied; this is the paper-style corner sort, not a sign-filtered rule.
- Long gross exposure: `{config.long_gross_exposure_float:.4f}`.
- Short gross exposure: `{config.short_gross_exposure_float:.4f}`.
- Slippage: `{config.slippage_float:.6f}` per side.
- Commission: `{config.commission_per_share_float:.6f}` per share, minimum `{config.commission_minimum_float:.2f}`.
- Short borrow, locate availability, recall risk, and financing are not modeled.
"""
    (output_path / "smooth_trend_russell3000_long_short_assumptions.md").write_text(
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
    audit_override_bool: bool | None = False,
) -> SmoothTrendRussell3000LongShortStrategy:
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
        pricing_data_df,
        universe_df,
        rebalance_schedule_df,
        trend_slope_df,
        trend_r2_df,
    ) = get_smooth_trend_russell3000_long_short_data(config=config_obj)
    strategy_obj = SmoothTrendRussell3000LongShortStrategy(
        name="strategy_mo_smooth_trend_russell3000_long_short",
        benchmarks=list(config_obj.benchmark_list),
        rebalance_schedule_df=rebalance_schedule_df,
        config=config_obj,
        precomputed_trend_slope_df=trend_slope_df,
        precomputed_trend_r2_df=trend_r2_df,
    )
    strategy_obj.universe_df = universe_df

    # *** CRITICAL*** Keep full pre-start history for the skipped OLS
    # formation window, but execute/report only from backtest_start_date_str.
    calendar_idx = pricing_data_df.index[
        pricing_data_df.index >= pd.Timestamp(config_obj.backtest_start_date_str)
    ]
    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar=calendar_idx,
        show_progress=show_display_bool,
        show_signal_progress_bool=show_display_bool,
        audit_override_bool=audit_override_bool,
    )

    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        display(strategy_obj.summary)
        display(strategy_obj.summary_trades)
        display(strategy_obj.rebalance_selection_df.tail(40))

    if save_results_bool:
        output_path = save_results(strategy_obj, output_dir=output_dir_str)
        strategy_obj.rebalance_selection_df.to_csv(
            output_path / "rebalance_selection.csv",
            index=False,
        )
        _write_assumptions_md(output_path=output_path, strategy=strategy_obj)

    return strategy_obj


if __name__ == "__main__":
    run_variant()
