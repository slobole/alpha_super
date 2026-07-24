"""
Research-only Russell 3000 Paper-B momentum with reduction-only volatility sizing.

For a portfolio held during month t, the decision is made at the final close
of month t-1. Using monthly decision closes P:

    M_t = P_(t-2) / P_(t-13) - 1
    r_t = P_(t-1) / P_(t-2) - 1
    B_t = (1 + r_t) * M_t

The highest 50 B scores are held long and the lowest 50 are held short. The
unscaled base portfolio is +100% long / -100% short. A hidden Vanilla pass
produces its completed calendar-month returns. After exactly 12 completed base
returns, the next holding month uses:

    annualized_base_vol_t = Std(R_(t-12), ..., R_(t-1)) * sqrt(12)
    exposure_t = min(1, 10% / annualized_base_vol_t)

Warm-up exposure is zero and warm-up months are absent from the saved strategy
performance. Orders are decided from the completed month-end close and filled
at the next tradable open by the standard Vanilla engine.
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
    get_monthly_decision_close_df,
    map_month_end_decision_dates_to_rebalance_schedule_df,
)


STRATEGY_NAME_STR = "strategy_mo_paper_b_russell3000_vol10"


def default_trade_id_int() -> int:
    return -1


@dataclass(frozen=True)
class PaperBRussell3000Config:
    variant_key_str: str = "paper_b_russell3000_top50_bottom50_vol10"
    indexname_str: str = "Russell 3000"
    benchmark_list: tuple[str, ...] = ("$RUA",)
    history_start_date_str: str = "1998-01-01"
    backtest_start_date_str: str = "2000-01-01"
    end_date_str: str | None = None
    max_long_positions_int: int = 50
    max_short_positions_int: int = 50
    minimum_unadjusted_close_float: float = 1.0
    minimum_adv_dollar_float: float = 1_000_000.0
    adv_lookback_day_int: int = 63
    volatility_lookback_month_int: int = 12
    target_annualized_volatility_float: float = 0.10
    maximum_exposure_multiplier_float: float = 1.0
    capital_base_float: float = 100_000.0

    def __post_init__(self) -> None:
        if not self.variant_key_str:
            raise ValueError("variant_key_str must not be empty.")
        if not self.indexname_str:
            raise ValueError("indexname_str must not be empty.")
        if len(self.benchmark_list) == 0:
            raise ValueError("benchmark_list must not be empty.")
        if pd.Timestamp(self.history_start_date_str) >= pd.Timestamp(self.backtest_start_date_str):
            raise ValueError("history_start_date_str must be earlier than backtest_start_date_str.")
        if self.max_long_positions_int <= 0:
            raise ValueError("max_long_positions_int must be positive.")
        if self.max_short_positions_int <= 0:
            raise ValueError("max_short_positions_int must be positive.")
        if self.minimum_unadjusted_close_float < 0.0:
            raise ValueError("minimum_unadjusted_close_float must be non-negative.")
        if self.minimum_adv_dollar_float < 0.0:
            raise ValueError("minimum_adv_dollar_float must be non-negative.")
        if self.adv_lookback_day_int <= 0:
            raise ValueError("adv_lookback_day_int must be positive.")
        if self.volatility_lookback_month_int != 12:
            raise ValueError("Paper-B volatility_lookback_month_int must remain exactly 12.")
        if self.target_annualized_volatility_float <= 0.0:
            raise ValueError("target_annualized_volatility_float must be positive.")
        if not 0.0 < self.maximum_exposure_multiplier_float <= 1.0:
            raise ValueError("maximum_exposure_multiplier_float must be in (0, 1].")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")


DEFAULT_CONFIG = PaperBRussell3000Config()


__all__ = [
    "DEFAULT_CONFIG",
    "PaperBRussell3000Config",
    "PaperBRussell3000Strategy",
    "build_exposure_schedule_df",
    "build_paper_b_selection_df",
    "compound_daily_returns_to_calendar_month_ser",
    "compute_paper_b_signal_tables",
    "get_paper_b_russell3000_data",
    "run_variant",
]


def compute_paper_b_signal_tables(
    price_close_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return decision closes, classic momentum, last-month return, and B."""
    monthly_decision_close_df = get_monthly_decision_close_df(price_close_df=price_close_df)

    # *** CRITICAL*** At the decision close for holding month t, shift(1)
    # is P_(t-2) and shift(12) is P_(t-13). The most recently completed
    # month is excluded from classic momentum and enters only through r_t.
    classic_momentum_df = (
        monthly_decision_close_df.shift(1) / monthly_decision_close_df.shift(12)
    ) - 1.0

    # *** CRITICAL*** r_t uses only the just-completed month: P_(t-1) / P_(t-2) - 1.
    last_month_return_df = (
        monthly_decision_close_df / monthly_decision_close_df.shift(1)
    ) - 1.0
    paper_b_score_df = (1.0 + last_month_return_df) * classic_momentum_df
    paper_b_score_df = paper_b_score_df.replace([np.inf, -np.inf], np.nan)
    return (
        monthly_decision_close_df,
        classic_momentum_df,
        last_month_return_df,
        paper_b_score_df,
    )


def build_paper_b_selection_df(
    rebalance_schedule_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    paper_b_score_df: pd.DataFrame,
    classic_momentum_df: pd.DataFrame,
    last_month_return_df: pd.DataFrame,
    unadjusted_close_decision_df: pd.DataFrame,
    adv_dollar_decision_df: pd.DataFrame,
    config: PaperBRussell3000Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Build the audited top-50 and bottom-50 selection at every rebalance."""
    required_candidate_count_int = (
        config.max_long_positions_int + config.max_short_positions_int
    )
    selection_frame_list: list[pd.DataFrame] = []

    for execution_date_ts, schedule_row_ser in rebalance_schedule_df.iterrows():
        decision_date_ts = pd.Timestamp(schedule_row_ser["decision_date_ts"])
        universe_member_ser = get_asof_universe_membership_ser(
            universe_df=universe_df,
            decision_date_ts=decision_date_ts,
        )
        active_symbol_list = universe_member_ser[universe_member_ser == 1].index.astype(str).tolist()

        candidate_feature_df = pd.DataFrame(
            {
                "paper_b_score_float": paper_b_score_df.loc[decision_date_ts].reindex(active_symbol_list),
                "classic_momentum_float": classic_momentum_df.loc[decision_date_ts].reindex(
                    active_symbol_list
                ),
                "last_month_return_float": last_month_return_df.loc[decision_date_ts].reindex(
                    active_symbol_list
                ),
                "unadjusted_close_float": unadjusted_close_decision_df.loc[
                    decision_date_ts
                ].reindex(active_symbol_list),
                "adv63_dollar_float": adv_dollar_decision_df.loc[decision_date_ts].reindex(
                    active_symbol_list
                ),
            }
        )
        candidate_feature_df.index = candidate_feature_df.index.astype(str)
        candidate_feature_df.index.name = "symbol_str"

        finite_mask_vec = np.isfinite(candidate_feature_df.to_numpy(dtype=float)).all(axis=1)
        # *** CRITICAL*** Both eligibility fields are sampled at the completed
        # decision close. No next-month open, volume, or price enters the rank.
        eligibility_mask_vec = (
            finite_mask_vec
            & (
                candidate_feature_df["unadjusted_close_float"].to_numpy(dtype=float)
                >= float(config.minimum_unadjusted_close_float)
            )
            & (
                candidate_feature_df["adv63_dollar_float"].to_numpy(dtype=float)
                >= float(config.minimum_adv_dollar_float)
            )
        )
        eligible_feature_df = candidate_feature_df.loc[eligibility_mask_vec].reset_index()
        eligible_count_int = len(eligible_feature_df)
        if eligible_count_int < required_candidate_count_int:
            raise RuntimeError(
                f"Only {eligible_count_int} eligible stocks on {decision_date_ts.date()}; "
                f"Paper-B requires at least {required_candidate_count_int}."
            )

        long_selection_df = (
            eligible_feature_df.sort_values(
                by=["paper_b_score_float", "symbol_str"],
                ascending=[False, True],
                kind="mergesort",
            )
            .head(config.max_long_positions_int)
            .copy()
        )
        short_selection_df = (
            eligible_feature_df.sort_values(
                by=["paper_b_score_float", "symbol_str"],
                ascending=[True, True],
                kind="mergesort",
            )
            .head(config.max_short_positions_int)
            .copy()
        )
        overlap_symbol_set = set(long_selection_df["symbol_str"]) & set(
            short_selection_df["symbol_str"]
        )
        if overlap_symbol_set:
            raise RuntimeError(
                f"Long/short selection overlap on {decision_date_ts.date()}: "
                f"{sorted(overlap_symbol_set)}"
            )

        for side_str, side_selection_df, side_weight_float in (
            (
                "long",
                long_selection_df,
                1.0 / float(config.max_long_positions_int),
            ),
            (
                "short",
                short_selection_df,
                -1.0 / float(config.max_short_positions_int),
            ),
        ):
            side_selection_df = side_selection_df.copy()
            side_selection_df.insert(0, "rank_int", np.arange(1, len(side_selection_df) + 1))
            side_selection_df.insert(0, "side_str", side_str)
            side_selection_df.insert(0, "execution_date_ts", pd.Timestamp(execution_date_ts))
            side_selection_df.insert(0, "decision_date_ts", decision_date_ts)
            side_selection_df["eligible_count_int"] = eligible_count_int
            side_selection_df["base_target_weight_float"] = side_weight_float
            selection_frame_list.append(side_selection_df)

    if len(selection_frame_list) == 0:
        raise RuntimeError("No Paper-B monthly selections were generated.")
    return pd.concat(selection_frame_list, ignore_index=True)


def compound_daily_returns_to_calendar_month_ser(
    daily_return_ser: pd.Series,
) -> pd.Series:
    """Compound one daily base-return path into completed calendar months."""
    if len(daily_return_ser) == 0:
        raise ValueError("daily_return_ser must not be empty.")
    clean_daily_return_ser = pd.to_numeric(daily_return_ser, errors="coerce")
    if clean_daily_return_ser.isna().any():
        raise ValueError("daily_return_ser contains missing or non-numeric values.")
    clean_daily_return_ser.index = pd.DatetimeIndex(clean_daily_return_ser.index)

    # *** CRITICAL*** These are non-overlapping calendar holding months. The
    # current incomplete month is never used by an exposure decision because
    # only completed decision dates are present in rebalance_schedule_df.
    month_period_idx = clean_daily_return_ser.index.to_period("M")
    monthly_return_ser = (1.0 + clean_daily_return_ser).groupby(month_period_idx).prod() - 1.0
    monthly_return_ser.index.name = "holding_month"
    monthly_return_ser.name = "base_monthly_return_float"
    return monthly_return_ser.astype(float)


def build_exposure_schedule_df(
    base_monthly_return_ser: pd.Series,
    rebalance_schedule_df: pd.DataFrame,
    config: PaperBRussell3000Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Map exactly 12 completed unscaled base returns to next-month exposure."""
    if len(base_monthly_return_ser) == 0:
        raise ValueError("base_monthly_return_ser must not be empty.")
    clean_base_monthly_return_ser = pd.to_numeric(base_monthly_return_ser, errors="coerce")
    if clean_base_monthly_return_ser.isna().any():
        raise ValueError("base_monthly_return_ser contains missing or non-numeric values.")
    if not isinstance(clean_base_monthly_return_ser.index, pd.PeriodIndex):
        clean_base_monthly_return_ser.index = pd.PeriodIndex(
            clean_base_monthly_return_ser.index,
            freq="M",
        )
    clean_base_monthly_return_ser = clean_base_monthly_return_ser.sort_index()

    # *** CRITICAL*** rolling(12) ends at the completed decision month. The
    # multiplier is mapped only to the following execution date, so the
    # upcoming holding-month return cannot enter its own exposure estimate.
    annualized_base_volatility_ser = clean_base_monthly_return_ser.rolling(
        window=config.volatility_lookback_month_int,
        min_periods=config.volatility_lookback_month_int,
    ).std(ddof=1) * np.sqrt(12.0)

    exposure_row_list: list[dict[str, object]] = []
    for execution_date_ts, schedule_row_ser in rebalance_schedule_df.iterrows():
        decision_date_ts = pd.Timestamp(schedule_row_ser["decision_date_ts"])
        decision_month_obj = decision_date_ts.to_period("M")
        completed_base_return_count_int = int(
            (clean_base_monthly_return_ser.index <= decision_month_obj).sum()
        )
        base_monthly_return_float = clean_base_monthly_return_ser.get(decision_month_obj, np.nan)
        annualized_base_volatility_float = annualized_base_volatility_ser.get(
            decision_month_obj,
            np.nan,
        )
        warmup_complete_bool = bool(
            completed_base_return_count_int >= config.volatility_lookback_month_int
            and np.isfinite(annualized_base_volatility_float)
        )

        exposure_multiplier_float = 0.0
        if warmup_complete_bool:
            if float(annualized_base_volatility_float) == 0.0:
                exposure_multiplier_float = float(config.maximum_exposure_multiplier_float)
            else:
                exposure_multiplier_float = min(
                    float(config.maximum_exposure_multiplier_float),
                    float(config.target_annualized_volatility_float)
                    / float(annualized_base_volatility_float),
                )

        exposure_row_list.append(
            {
                "execution_date_ts": pd.Timestamp(execution_date_ts),
                "decision_date_ts": decision_date_ts,
                "decision_month_str": str(decision_month_obj),
                "completed_base_return_count_int": completed_base_return_count_int,
                "base_monthly_return_float": float(base_monthly_return_float)
                if np.isfinite(base_monthly_return_float)
                else np.nan,
                "annualized_base_volatility_float": float(annualized_base_volatility_float)
                if np.isfinite(annualized_base_volatility_float)
                else np.nan,
                "warmup_complete_bool": warmup_complete_bool,
                "exposure_multiplier_float": exposure_multiplier_float,
                "long_gross_target_float": exposure_multiplier_float,
                "short_gross_target_float": -exposure_multiplier_float,
                "gross_target_float": 2.0 * exposure_multiplier_float,
            }
        )

    exposure_schedule_df = pd.DataFrame(exposure_row_list).set_index("execution_date_ts")
    exposure_schedule_df.index = pd.DatetimeIndex(
        exposure_schedule_df.index,
        name="execution_date_ts",
    )
    return exposure_schedule_df.sort_index()


def _get_base_calendar_idx(
    pricing_date_idx: pd.DatetimeIndex,
    rebalance_schedule_df: pd.DataFrame,
) -> pd.DatetimeIndex:
    """Start the hidden base path on its first actual monthly rebalance."""
    if len(rebalance_schedule_df) == 0:
        raise ValueError("rebalance_schedule_df must not be empty.")
    first_rebalance_date_ts = pd.Timestamp(rebalance_schedule_df.index[0])
    # *** CRITICAL*** A custom mid-month requested start must not create a
    # partial all-cash calendar month that counts toward the 12-return warm-up.
    base_calendar_idx = pd.DatetimeIndex(pricing_date_idx)[
        pd.DatetimeIndex(pricing_date_idx) >= first_rebalance_date_ts
    ]
    if len(base_calendar_idx) == 0:
        raise RuntimeError("No base-backtest dates fall on or after the first rebalance.")
    return base_calendar_idx


class PaperBRussell3000Strategy(Strategy):
    """Monthly Paper-B long/short strategy executed by the Vanilla engine."""

    enable_signal_audit = False

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        rebalance_schedule_df: pd.DataFrame,
        selection_df: pd.DataFrame,
        exposure_schedule_df: pd.DataFrame,
        config: PaperBRussell3000Config = DEFAULT_CONFIG,
    ):
        # Cost arguments are intentionally omitted. Strategy owns the repo
        # defaults: 2.5 bps slippage plus the default IBKR-like commission.
        super().__init__(
            name=name,
            benchmarks=list(benchmarks),
            capital_base=config.capital_base_float,
        )
        self.config = config
        self.rebalance_schedule_df = rebalance_schedule_df.copy().sort_index()
        self.selection_df = selection_df.copy()
        self.exposure_schedule_df = exposure_schedule_df.copy().sort_index()
        self.selection_by_execution_date_dict = {
            pd.Timestamp(execution_date_ts): group_df.copy()
            for execution_date_ts, group_df in self.selection_df.groupby("execution_date_ts")
        }
        self.trade_id_int = 0
        self.current_trade_map: defaultdict[str, int] = defaultdict(default_trade_id_int)
        self.rebalance_execution_row_list: list[dict[str, object]] = []
        self.rebalance_execution_df = pd.DataFrame()
        self.base_monthly_return_ser = pd.Series(dtype=float)
        self.output_path_obj: Path | None = None

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        # Cross-sectional signals and eligibility were already computed from
        # completed month-end data before the engine pass.
        return pricing_data_df

    def iterate(
        self,
        data_df: pd.DataFrame,
        close_row_ser: pd.Series,
        open_price_ser: pd.Series,
    ) -> None:
        if self.current_bar not in self.rebalance_schedule_df.index:
            return

        decision_date_ts = pd.Timestamp(
            self.rebalance_schedule_df.loc[self.current_bar, "decision_date_ts"]
        )
        # *** CRITICAL*** The selection and exposure decision are both fixed
        # at previous_bar. Vanilla fills target orders at current_bar open.
        if pd.Timestamp(self.previous_bar) != decision_date_ts:
            raise RuntimeError(
                f"Schedule misalignment on {self.current_bar}: "
                f"decision_date_ts={decision_date_ts}, previous_bar={self.previous_bar}."
            )
        if self.current_bar not in self.exposure_schedule_df.index:
            raise RuntimeError(f"Missing exposure multiplier for {self.current_bar}.")

        exposure_multiplier_float = float(
            self.exposure_schedule_df.loc[self.current_bar, "exposure_multiplier_float"]
        )
        selection_df = self.selection_by_execution_date_dict.get(
            pd.Timestamp(self.current_bar),
            pd.DataFrame(),
        )
        if len(selection_df) == 0:
            raise RuntimeError(f"Missing Paper-B selection for {self.current_bar}.")

        target_weight_ser = pd.Series(
            selection_df["base_target_weight_float"].to_numpy(dtype=float)
            * exposure_multiplier_float,
            index=selection_df["symbol_str"].astype(str),
            dtype=float,
        ).sort_index()
        if not np.isclose(float(target_weight_ser.sum()), 0.0, atol=1e-12):
            raise RuntimeError(f"Target weights are not dollar-neutral on {self.current_bar}.")
        if float(target_weight_ser.abs().sum()) > 2.0 + 1e-12:
            raise RuntimeError(f"Gross target exceeds the unscaled base on {self.current_bar}.")

        execution_record_df = selection_df.copy()
        execution_record_df["exposure_multiplier_float"] = exposure_multiplier_float
        execution_record_df["target_weight_float"] = (
            execution_record_df["base_target_weight_float"] * exposure_multiplier_float
        )
        self.rebalance_execution_row_list.extend(execution_record_df.to_dict("records"))

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
            sizing_close_price_float = float(close_row_ser.loc[(symbol_str, "Close")])
            if not np.isfinite(sizing_close_price_float) or sizing_close_price_float <= 0.0:
                raise RuntimeError(
                    f"Invalid prior-close sizing price for {symbol_str} on {decision_date_ts}."
                )
            target_share_int = int(
                float(self.previous_total_value)
                * float(target_weight_float)
                / sizing_close_price_float
            )
            # Vanilla converts target-percent orders to integer shares from
            # previous equity and previous close. Suppress exact no-ops here;
            # otherwise the shared engine records a zero-share transaction and
            # charges its minimum commission.
            if float(target_share_int) == current_share_float:
                continue
            if current_share_float == 0.0:
                self.trade_id_int += 1
                self.current_trade_map[symbol_str] = self.trade_id_int
            self.order_target_percent(
                symbol_str,
                float(target_weight_float),
                trade_id=self.current_trade_map[symbol_str],
            )

    def finalize(self, current_data_df: pd.DataFrame) -> None:
        self.rebalance_execution_df = pd.DataFrame(self.rebalance_execution_row_list)


def get_paper_b_russell3000_data(
    config: PaperBRussell3000Config = DEFAULT_CONFIG,
    pricing_data_df: pd.DataFrame | None = None,
    universe_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load data once and return lean prices, PIT universe, schedule, selections."""
    if (pricing_data_df is None) != (universe_df is None):
        raise ValueError("pricing_data_df and universe_df must be provided together.")

    if pricing_data_df is None or universe_df is None:
        _, raw_universe_df = build_index_constituent_matrix(indexname=config.indexname_str)
        history_start_ts = pd.Timestamp(config.history_start_date_str)
        filtered_universe_df = raw_universe_df.loc[raw_universe_df.index >= history_start_ts].copy()
        if config.end_date_str is not None:
            filtered_universe_df = filtered_universe_df.loc[
                filtered_universe_df.index <= pd.Timestamp(config.end_date_str)
            ]

        first_required_membership_ts = pd.Timestamp(config.backtest_start_date_str) - pd.Timedelta(
            days=45
        )
        active_universe_df = filtered_universe_df.loc[
            filtered_universe_df.index >= first_required_membership_ts
        ]
        active_symbol_list = active_universe_df.columns[
            active_universe_df.sum(axis=0) > 0
        ].astype(str).tolist()
        if len(active_symbol_list) == 0:
            raise RuntimeError("No active Russell 3000 symbols were found for the requested window.")

        pricing_data_df = load_raw_prices(
            symbols=active_symbol_list,
            benchmarks=list(config.benchmark_list),
            start_date=config.history_start_date_str,
            end_date=config.end_date_str,
        )
        universe_df = filtered_universe_df
    else:
        pricing_data_df = pricing_data_df.copy()
        universe_df = universe_df.copy()

    benchmark_symbol_set = set(config.benchmark_list)
    available_column_set = set(pricing_data_df.columns.to_list())
    candidate_symbol_list = [
        str(symbol_str)
        for symbol_str in pricing_data_df.columns.get_level_values(0).unique()
        if str(symbol_str) not in benchmark_symbol_set
    ]
    required_stock_field_tuple = (
        "Open",
        "High",
        "Low",
        "Close",
        "Unadjusted Close",
        "Turnover",
    )
    loaded_symbol_list = [
        symbol_str
        for symbol_str in candidate_symbol_list
        if all((symbol_str, field_str) in available_column_set for field_str in required_stock_field_tuple)
    ]
    if len(loaded_symbol_list) == 0:
        raise RuntimeError(
            "No stocks contain OHLC, Unadjusted Close, and Turnover fields."
        )

    audited_universe_df = audit_pit_universe_df(
        universe_df=universe_df,
        execution_index=pricing_data_df.index,
        tradeable_symbol_list=loaded_symbol_list,
    )
    loaded_symbol_list = audited_universe_df.columns.astype(str).tolist()

    price_close_df = pd.DataFrame(
        {
            symbol_str: pricing_data_df[(symbol_str, "Close")]
            for symbol_str in loaded_symbol_list
        },
        index=pricing_data_df.index,
        dtype=float,
    )
    (
        monthly_decision_close_df,
        classic_momentum_df,
        last_month_return_df,
        paper_b_score_df,
    ) = compute_paper_b_signal_tables(price_close_df=price_close_df)
    del price_close_df

    unadjusted_close_df = pd.DataFrame(
        {
            symbol_str: pricing_data_df[(symbol_str, "Unadjusted Close")]
            for symbol_str in loaded_symbol_list
        },
        index=pricing_data_df.index,
        dtype=float,
    )
    unadjusted_close_decision_df = unadjusted_close_df.reindex(
        monthly_decision_close_df.index
    )
    del unadjusted_close_df

    turnover_dollar_df = pd.DataFrame(
        {
            symbol_str: pricing_data_df[(symbol_str, "Turnover")]
            for symbol_str in loaded_symbol_list
        },
        index=pricing_data_df.index,
        dtype=float,
    )

    # *** CRITICAL*** ADV63 includes only sessions through the completed
    # decision close. Norgate Turnover is the raw dollar-volume field, so no
    # adjusted-price multiplication is introduced here.
    adv_dollar_decision_df = turnover_dollar_df.rolling(
        window=config.adv_lookback_day_int,
        min_periods=config.adv_lookback_day_int,
    ).mean().reindex(monthly_decision_close_df.index)
    del turnover_dollar_df

    rebalance_schedule_df = map_month_end_decision_dates_to_rebalance_schedule_df(
        decision_date_index=pd.DatetimeIndex(monthly_decision_close_df.index),
        execution_index=pd.DatetimeIndex(pricing_data_df.index),
    )
    rebalance_schedule_df = rebalance_schedule_df.loc[
        rebalance_schedule_df.index >= pd.Timestamp(config.backtest_start_date_str)
    ].copy()
    if config.end_date_str is not None:
        rebalance_schedule_df = rebalance_schedule_df.loc[
            rebalance_schedule_df.index <= pd.Timestamp(config.end_date_str)
        ]
    if len(rebalance_schedule_df) == 0:
        raise RuntimeError("No Paper-B rebalances fall inside the requested backtest window.")

    selection_df = build_paper_b_selection_df(
        rebalance_schedule_df=rebalance_schedule_df,
        universe_df=audited_universe_df,
        paper_b_score_df=paper_b_score_df,
        classic_momentum_df=classic_momentum_df,
        last_month_return_df=last_month_return_df,
        unadjusted_close_decision_df=unadjusted_close_decision_df,
        adv_dollar_decision_df=adv_dollar_decision_df,
        config=config,
    )

    keep_column_list: list[tuple[str, str]] = []
    for symbol_str in loaded_symbol_list:
        keep_column_list.extend(
            (symbol_str, field_str) for field_str in ("Open", "High", "Low", "Close")
        )
    for benchmark_str in config.benchmark_list:
        for field_str in ("Open", "Close"):
            if (benchmark_str, field_str) in available_column_set:
                keep_column_list.append((benchmark_str, field_str))
    lean_pricing_data_df = pricing_data_df.loc[:, keep_column_list].copy().sort_index()
    return (
        lean_pricing_data_df,
        audited_universe_df,
        rebalance_schedule_df,
        selection_df,
    )


def _write_assumptions_md(
    output_path_obj: Path,
    strategy_obj: PaperBRussell3000Strategy,
) -> None:
    config_obj = strategy_obj.config
    assumption_md_str = f"""# Paper-B Russell 3000 Volatility-10 Assumptions

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
    audit_override_bool: bool | None = False,
    pricing_data_df: pd.DataFrame | None = None,
    universe_df: pd.DataFrame | None = None,
) -> PaperBRussell3000Strategy:
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
    ) = get_paper_b_russell3000_data(
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
    base_strategy_obj = PaperBRussell3000Strategy(
        name=f"{STRATEGY_NAME_STR}_unscaled_base",
        benchmarks=list(config_obj.benchmark_list),
        rebalance_schedule_df=rebalance_schedule_df,
        selection_df=selection_df,
        exposure_schedule_df=base_exposure_schedule_df,
        config=config_obj,
    )
    base_calendar_idx = _get_base_calendar_idx(
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
        audit_override_bool=audit_override_bool,
    )
    base_monthly_return_ser = compound_daily_returns_to_calendar_month_ser(
        base_strategy_obj.results["daily_returns"]
    )
    exposure_schedule_df = build_exposure_schedule_df(
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

    strategy_obj = PaperBRussell3000Strategy(
        name=STRATEGY_NAME_STR,
        benchmarks=list(config_obj.benchmark_list),
        rebalance_schedule_df=rebalance_schedule_df,
        selection_df=selection_df,
        exposure_schedule_df=exposure_schedule_df,
        config=config_obj,
    )
    strategy_obj.base_monthly_return_ser = base_monthly_return_ser.copy()
    strategy_obj.reported_start_date_ts = reported_start_date_ts

    # *** CRITICAL*** Starting the reported engine calendar here, rather than
    # running zero exposure beforehand, keeps all 12 warm-up months out of
    # every saved performance metric and benchmark comparison.
    reported_calendar_idx = loaded_pricing_data_df.index[
        loaded_pricing_data_df.index >= reported_start_date_ts
    ]
    run_daily(
        strategy_obj,
        loaded_pricing_data_df,
        calendar=reported_calendar_idx,
        show_progress=show_display_bool,
        show_signal_progress_bool=False,
        audit_override_bool=audit_override_bool,
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


if __name__ == "__main__":
    run_variant()
