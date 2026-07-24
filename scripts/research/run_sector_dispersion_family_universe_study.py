from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from alpha.engine.backtest import run_daily
from alpha.engine.report import build_research_output_path
from strategies.mean_reversion.strategy_mr_sector_dispersion_ibs import (
    DEFAULT_CONFIG,
    SectorDispersionIbsConfig,
    SectorDispersionIbsStrategy,
    get_sector_dispersion_ibs_data,
    resolve_full_basket_calendar_idx,
)


STUDY_END_DATE_STR = "2026-07-17"
HISTORY_START_DATE_STR = "1997-01-01"
COMMON_BENCHMARK_SYMBOL_STR = "$SPX"
CAPITAL_BASE_FLOAT = 100_000.0
SEARCH_SPACE_COUNT_INT = 6
NEGATIVE_CASH_TOLERANCE_FLOAT = -1e-8
STALE_NO_PRINT_FIELD_STR = "stale_no_print_bool"
MAX_STALE_SESSION_COUNT_INT = 5


@dataclass(frozen=True)
class UniverseSpec:
    priority_int: int
    universe_id_str: str
    symbol_tuple: tuple[str, ...]
    raw_common_start_date_str: str
    research_role_str: str


UNIVERSE_SPEC_TUPLE = (
    UniverseSpec(
        priority_int=1,
        universe_id_str="spdr_11",
        symbol_tuple=("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLRE", "XLC"),
        raw_common_start_date_str="2018-06-19",
        research_role_str="Article-universe next-open translation",
    ),
    UniverseSpec(
        priority_int=2,
        universe_id_str="spdr_9",
        symbol_tuple=("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"),
        raw_common_start_date_str="1998-12-22",
        research_role_str="Clean long-history SPDR test",
    ),
    UniverseSpec(
        priority_int=3,
        universe_id_str="vanguard_11",
        symbol_tuple=("VAW", "VDE", "VFH", "VIS", "VGT", "VDC", "VPU", "VHT", "VCR", "VOX", "VNQ"),
        raw_common_start_date_str="2004-09-29",
        research_role_str="Broad-US family robustness",
    ),
    UniverseSpec(
        priority_int=4,
        universe_id_str="spdr_proxy_11",
        symbol_tuple=("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "VOX", "IYR"),
        raw_common_start_date_str="2004-09-29",
        research_role_str="Proxy test, not historical SPDR reconstruction",
    ),
    UniverseSpec(
        priority_int=5,
        universe_id_str="ishares_us_11",
        symbol_tuple=("IYM", "IYE", "IYF", "IYJ", "IYW", "IYK", "IDU", "IYH", "IYC", "IYZ", "IYR"),
        raw_common_start_date_str="2000-07-14",
        research_role_str="Long alternative US classification",
    ),
    UniverseSpec(
        priority_int=6,
        universe_id_str="ishares_global_11",
        symbol_tuple=("MXI", "IXC", "IXG", "EXI", "IXN", "KXI", "JXI", "IXJ", "RXI", "IXP", "RWO"),
        raw_common_start_date_str="2008-05-13",
        research_role_str="Global robustness",
    ),
)

SUBPERIOD_TUPLE = (
    ("1999_2007", pd.Timestamp("1999-01-01"), pd.Timestamp("2007-12-31")),
    ("2008_2017", pd.Timestamp("2008-01-01"), pd.Timestamp("2017-12-31")),
    ("2018_2021", pd.Timestamp("2018-01-01"), pd.Timestamp("2021-12-31")),
    ("2022_2026", pd.Timestamp("2022-01-01"), pd.Timestamp(STUDY_END_DATE_STR)),
)


def build_universe_manifest_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "priority_int": universe_spec_obj.priority_int,
                "universe_id_str": universe_spec_obj.universe_id_str,
                "symbol_count_int": len(universe_spec_obj.symbol_tuple),
                "symbol_tuple_str": ",".join(universe_spec_obj.symbol_tuple),
                "raw_common_start_date_str": universe_spec_obj.raw_common_start_date_str,
                "research_role_str": universe_spec_obj.research_role_str,
            }
            for universe_spec_obj in UNIVERSE_SPEC_TUPLE
        ]
    )


def prepare_isolated_no_print_sessions(
    pricing_data_df: pd.DataFrame,
    universe_manifest_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared_data_df = pricing_data_df.copy()
    all_symbol_tuple = tuple(
        dict.fromkeys(
            symbol_str
            for _, manifest_row_ser in universe_manifest_df.iterrows()
            for symbol_str in _symbol_tuple_from_manifest_row(manifest_row_ser)
        )
    )
    stale_row_dict_list: list[dict[str, object]] = []
    for symbol_str in all_symbol_tuple:
        field_ser_dict = {
            field_str: pd.to_numeric(
                prepared_data_df[(symbol_str, field_str)],
                errors="coerce",
            )
            for field_str in ("Open", "High", "Low", "Close")
        }
        field_df = pd.DataFrame(field_ser_dict)
        any_ohlc_bool_ser = field_df.notna().any(axis=1)
        complete_ohlc_bool_ser = field_df.notna().all(axis=1)
        if not complete_ohlc_bool_ser.any():
            raise RuntimeError(f"No complete OHLC history exists for {symbol_str}.")

        first_complete_ts = complete_ohlc_bool_ser.loc[complete_ohlc_bool_ser].index[0]
        in_life_bool_ser = pd.Series(
            prepared_data_df.index >= first_complete_ts,
            index=prepared_data_df.index,
            dtype=bool,
        )
        partial_ohlc_bool_ser = in_life_bool_ser & any_ohlc_bool_ser & ~complete_ohlc_bool_ser
        if partial_ohlc_bool_ser.any():
            first_partial_ts = partial_ohlc_bool_ser.loc[partial_ohlc_bool_ser].index[0]
            raise RuntimeError(
                f"Partial OHLC row for {symbol_str} on {pd.Timestamp(first_partial_ts).date()}."
            )

        valid_complete_bool_ser = (
            complete_ohlc_bool_ser
            & field_df["Open"].gt(0.0)
            & field_df["High"].gt(0.0)
            & field_df["Low"].gt(0.0)
            & field_df["Close"].gt(0.0)
            & field_df["High"].ge(field_df["Open"])
            & field_df["High"].ge(field_df["Close"])
            & field_df["High"].ge(field_df["Low"])
            & field_df["Low"].le(field_df["Open"])
            & field_df["Low"].le(field_df["Close"])
            & field_df["Low"].le(field_df["High"])
        )
        malformed_bool_ser = in_life_bool_ser & complete_ohlc_bool_ser & ~valid_complete_bool_ser
        if malformed_bool_ser.any():
            first_malformed_ts = malformed_bool_ser.loc[malformed_bool_ser].index[0]
            raise RuntimeError(
                f"Malformed OHLC row for {symbol_str} on {pd.Timestamp(first_malformed_ts).date()}."
            )

        stale_bool_ser = in_life_bool_ser & ~any_ohlc_bool_ser
        stale_position_arr = np.flatnonzero(stale_bool_ser.to_numpy())
        if len(stale_position_arr) > MAX_STALE_SESSION_COUNT_INT:
            raise RuntimeError(
                f"{symbol_str} has {len(stale_position_arr)} stale sessions; "
                f"maximum allowed is {MAX_STALE_SESSION_COUNT_INT}."
            )
        if len(stale_position_arr) > 1 and np.any(np.diff(stale_position_arr) == 1):
            raise RuntimeError(f"{symbol_str} has consecutive stale sessions.")

        prepared_data_df[(symbol_str, STALE_NO_PRINT_FIELD_STR)] = False
        for stale_ts in stale_bool_ser.loc[stale_bool_ser].index:
            # *** CRITICAL*** A complete no-print day receives only the prior
            # completed close for valuation. High equals Low, so this synthetic
            # row cannot create an IBS or RelativeRange signal.
            prior_close_ser = field_df.loc[field_df.index < stale_ts, "Close"].dropna()
            if len(prior_close_ser) == 0:
                raise RuntimeError(
                    f"Cannot stale-value {symbol_str} on {pd.Timestamp(stale_ts).date()} without a prior close."
                )
            prior_close_float = float(prior_close_ser.iloc[-1])
            for field_str in ("Open", "High", "Low", "Close"):
                prepared_data_df.loc[stale_ts, (symbol_str, field_str)] = prior_close_float
            dividend_value_obj = prepared_data_df.loc[stale_ts, (symbol_str, "Dividend")]
            if pd.isna(dividend_value_obj):
                prepared_data_df.loc[stale_ts, (symbol_str, "Dividend")] = 0.0
            prepared_data_df.loc[stale_ts, (symbol_str, STALE_NO_PRINT_FIELD_STR)] = True
            stale_row_dict_list.append(
                {
                    "symbol_str": symbol_str,
                    "bar": pd.Timestamp(stale_ts),
                    "prior_close_float": prior_close_float,
                    "policy_str": "prior_close_valuation_and_cancel_orders",
                }
            )

    return prepared_data_df.sort_index(axis=1), pd.DataFrame(stale_row_dict_list)


def _symbol_tuple_from_manifest_row(manifest_row_ser: pd.Series) -> tuple[str, ...]:
    symbol_tuple = tuple(
        symbol_str.strip()
        for symbol_str in str(manifest_row_ser["symbol_tuple_str"]).split(",")
        if symbol_str.strip()
    )
    if len(symbol_tuple) == 0:
        raise ValueError("Universe symbol tuple must not be empty.")
    if len(symbol_tuple) != len(set(symbol_tuple)):
        raise ValueError(f"Universe contains duplicate symbols: {symbol_tuple}")
    return symbol_tuple


def build_universe_config_obj(
    manifest_row_ser: pd.Series,
    end_date_str: str = STUDY_END_DATE_STR,
) -> SectorDispersionIbsConfig:
    return replace(
        DEFAULT_CONFIG,
        symbol_tuple=_symbol_tuple_from_manifest_row(manifest_row_ser),
        universe_name_str="original",
        benchmark_symbol_str=COMMON_BENCHMARK_SYMBOL_STR,
        history_start_date_str=HISTORY_START_DATE_STR,
        backtest_start_date_str=str(manifest_row_ser["raw_common_start_date_str"]),
        end_date_str=end_date_str,
        portfolio_leverage_float=1.0,
        capital_base_float=CAPITAL_BASE_FLOAT,
    )


def build_execution_calendar_idx(
    pricing_data_df: pd.DataFrame,
    config_obj: SectorDispersionIbsConfig,
) -> pd.DatetimeIndex:
    # *** CRITICAL*** The execution calendar begins only after every fixed ETF
    # has the completed bar T plus the 21 preceding positive-range bars needed
    # for the lagged denominator. No pre-inception partial basket is allowed.
    return resolve_full_basket_calendar_idx(
        pricing_data_df=pricing_data_df,
        config_obj=config_obj,
    )


class SectorDispersionDividendStrategy(SectorDispersionIbsStrategy):
    """Sector IBS strategy with explicit ETF cash-distribution accounting."""

    def __init__(
        self,
        name: str,
        benchmarks: list[str] | tuple[str, ...],
        config_obj: SectorDispersionIbsConfig,
    ):
        super().__init__(name=name, benchmarks=benchmarks, config_obj=config_obj)
        self._dividend_credit_row_dict_list: list[dict[str, object]] = []
        self._stale_order_cancellation_row_dict_list: list[dict[str, object]] = []
        self.dividend_cash_total_float = 0.0
        self.stale_order_cancellation_count_int = 0

    @property
    def dividend_credit_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._dividend_credit_row_dict_list)

    @property
    def stale_order_cancellation_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._stale_order_cancellation_row_dict_list)

    def _cancel_current_bar_stale_orders(self, pricing_data_df: pd.DataFrame) -> None:
        if self.current_bar not in pricing_data_df.index:
            return
        for symbol_str in self.symbol_tuple:
            stale_column_tuple = (symbol_str, STALE_NO_PRINT_FIELD_STR)
            if stale_column_tuple not in pricing_data_df.columns:
                continue
            stale_bar_bool = bool(pricing_data_df.loc[self.current_bar, stale_column_tuple])
            if not stale_bar_bool:
                continue

            stale_order_list = [
                order_obj
                for order_obj in self.get_orders()
                if str(order_obj.asset) == symbol_str
            ]
            if len(stale_order_list) == 0:
                continue
            # *** CRITICAL*** A next-open order cannot fill on a session with no
            # ETF print. Cancel it before the shared engine sees the synthetic
            # valuation-only OHLC row.
            for order_obj in stale_order_list:
                self._stale_order_cancellation_row_dict_list.append(
                    {
                        "bar": pd.Timestamp(self.current_bar),
                        "asset": symbol_str,
                        "order_id_int": int(order_obj.id),
                        "trade_id": order_obj.trade_id,
                    }
                )
            self.stale_order_cancellation_count_int += len(stale_order_list)
            self.clear_orders(asset=symbol_str)

    def _credit_current_bar_dividend_cash(self, pricing_data_df: pd.DataFrame) -> None:
        if self.current_bar not in pricing_data_df.index:
            return

        preopen_position_ser = self.get_positions()
        for symbol_str in self.symbol_tuple:
            position_share_float = float(preopen_position_ser.get(symbol_str, 0.0))
            if np.isclose(position_share_float, 0.0):
                continue

            dividend_column_tuple = (symbol_str, "Dividend")
            if dividend_column_tuple not in pricing_data_df.columns:
                raise RuntimeError(f"Missing Norgate dividend field for {symbol_str}.")
            dividend_per_share_float = float(
                pd.to_numeric(
                    pd.Series([pricing_data_df.loc[self.current_bar, dividend_column_tuple]]),
                    errors="coerce",
                ).iloc[0]
            )
            if not np.isfinite(dividend_per_share_float):
                raise RuntimeError(
                    f"Invalid dividend value for {symbol_str} on {pd.Timestamp(self.current_bar).date()}."
                )
            if np.isclose(dividend_per_share_float, 0.0):
                continue

            # *** CRITICAL*** Dividend entitlement is based on shares held
            # before current_bar's open. This credit occurs before open orders:
            # a same-open buy receives nothing, while a same-open sale retains
            # the distribution earned by the prior close position.
            dividend_cash_float = position_share_float * dividend_per_share_float
            self.cash += dividend_cash_float
            self.dividend_cash_total_float += dividend_cash_float
            self._dividend_credit_row_dict_list.append(
                {
                    "bar": pd.Timestamp(self.current_bar),
                    "asset": symbol_str,
                    "position_share_float": position_share_float,
                    "dividend_per_share_float": dividend_per_share_float,
                    "dividend_cash_float": dividend_cash_float,
                }
            )

    def process_orders(self, prices: pd.DataFrame):
        self._cancel_current_bar_stale_orders(pricing_data_df=prices)
        self._credit_current_bar_dividend_cash(pricing_data_df=prices)
        return super().process_orders(prices)


def compute_equal_weight_benchmark_return_ser(
    pricing_data_df: pd.DataFrame,
    symbol_tuple: tuple[str, ...],
    calendar_idx: pd.DatetimeIndex,
) -> pd.Series:
    close_price_df = pd.DataFrame(
        {
            symbol_str: pd.to_numeric(
                pricing_data_df[(symbol_str, "Close")],
                errors="coerce",
            )
            for symbol_str in symbol_tuple
        }
    )
    dividend_df = pd.DataFrame(
        {
            symbol_str: pd.to_numeric(
                pricing_data_df[(symbol_str, "Dividend")],
                errors="coerce",
            )
            for symbol_str in symbol_tuple
        }
    ).fillna(0.0)

    # *** CRITICAL*** Daily ETF total return at T uses the completed capital
    # close at T, the cash dividend attached to T, and Close_(T-1). This series
    # is a post-run benchmark and never feeds strategy signals or orders.
    asset_total_return_df = (
        (close_price_df + dividend_df) / close_price_df.shift(1) - 1.0
    )
    benchmark_return_ser = asset_total_return_df.mean(axis=1, skipna=False).reindex(calendar_idx)
    if len(benchmark_return_ser) == 0:
        raise RuntimeError("Equal-weight benchmark has no observations.")
    benchmark_return_ser.iloc[0] = 0.0
    invalid_return_ser = benchmark_return_ser.iloc[1:].loc[
        ~np.isfinite(benchmark_return_ser.iloc[1:])
    ]
    if len(invalid_return_ser) > 0:
        raise RuntimeError(
            "Equal-weight benchmark became invalid after its start on "
            f"{pd.Timestamp(invalid_return_ser.index[0]).date()}."
        )
    benchmark_return_ser.name = "equal_weight_total_return"
    return benchmark_return_ser.astype(float)


def compute_equity_metric_dict(
    total_value_ser: pd.Series,
    prefix_str: str,
) -> dict[str, object]:
    normalized_total_value_ser = pd.to_numeric(total_value_ser, errors="coerce").dropna().astype(float)
    metric_dict: dict[str, object] = {
        f"{prefix_str}_start_date_str": None,
        f"{prefix_str}_end_date_str": None,
        f"{prefix_str}_day_count_int": int(len(normalized_total_value_ser)),
        f"{prefix_str}_ann_return_pct_float": np.nan,
        f"{prefix_str}_volatility_ann_pct_float": np.nan,
        f"{prefix_str}_sharpe_float": np.nan,
        f"{prefix_str}_max_drawdown_pct_float": np.nan,
        f"{prefix_str}_mar_float": np.nan,
    }
    if len(normalized_total_value_ser) < 2:
        return metric_dict

    # *** CRITICAL*** These are post-run diagnostics only. They may compare
    # completed equity paths but must never feed the strategy state machine.
    daily_return_ser = normalized_total_value_ser.pct_change(fill_method=None).dropna()
    running_peak_ser = normalized_total_value_ser.cummax()
    drawdown_ser = normalized_total_value_ser / running_peak_ser - 1.0
    observation_count_int = int(len(daily_return_ser))
    ann_return_float = float(
        (normalized_total_value_ser.iloc[-1] / normalized_total_value_ser.iloc[0])
        ** (252.0 / float(observation_count_int))
        - 1.0
    )
    volatility_float = float(daily_return_ser.std() * np.sqrt(252.0))
    daily_std_float = float(daily_return_ser.std())
    sharpe_float = (
        np.nan
        if np.isclose(daily_std_float, 0.0)
        else float(daily_return_ser.mean() / daily_std_float * np.sqrt(252.0))
    )
    max_drawdown_float = float(drawdown_ser.min())
    mar_float = (
        np.nan
        if np.isclose(max_drawdown_float, 0.0)
        else float(ann_return_float / abs(max_drawdown_float))
    )
    metric_dict.update(
        {
            f"{prefix_str}_start_date_str": normalized_total_value_ser.index[0].date().isoformat(),
            f"{prefix_str}_end_date_str": normalized_total_value_ser.index[-1].date().isoformat(),
            f"{prefix_str}_ann_return_pct_float": ann_return_float * 100.0,
            f"{prefix_str}_volatility_ann_pct_float": volatility_float * 100.0,
            f"{prefix_str}_sharpe_float": sharpe_float,
            f"{prefix_str}_max_drawdown_pct_float": max_drawdown_float * 100.0,
            f"{prefix_str}_mar_float": mar_float,
        }
    )
    return metric_dict


def build_exposure_diagnostic_dict(
    realized_weight_df: pd.DataFrame,
    result_df: pd.DataFrame,
    symbol_tuple: tuple[str, ...],
) -> dict[str, object]:
    normalized_weight_df = realized_weight_df.copy()
    normalized_weight_df.columns = [str(column_obj) for column_obj in normalized_weight_df.columns]
    symbol_weight_df = normalized_weight_df.reindex(columns=list(symbol_tuple), fill_value=0.0).fillna(0.0)
    gross_exposure_ser = symbol_weight_df.abs().sum(axis=1)
    active_position_count_ser = symbol_weight_df.abs().gt(1e-12).sum(axis=1)
    cash_ser = pd.to_numeric(result_df["cash"], errors="coerce")
    return {
        "average_gross_exposure_pct_float": float(gross_exposure_ser.mean() * 100.0),
        "max_gross_exposure_pct_float": float(gross_exposure_ser.max() * 100.0),
        "average_active_position_count_float": float(active_position_count_ser.mean()),
        "max_active_position_count_int": int(active_position_count_ser.max()),
        "minimum_cash_float": float(cash_ser.min()),
        "negative_cash_day_count_int": int(cash_ser.lt(NEGATIVE_CASH_TOLERANCE_FLOAT).sum()),
    }


def build_data_quality_df(
    pricing_data_df: pd.DataFrame,
    universe_manifest_df: pd.DataFrame,
) -> pd.DataFrame:
    universe_symbol_map = {
        str(manifest_row_ser["universe_id_str"]): _symbol_tuple_from_manifest_row(manifest_row_ser)
        for _, manifest_row_ser in universe_manifest_df.iterrows()
    }
    all_symbol_tuple = tuple(
        dict.fromkeys(
            symbol_str
            for symbol_tuple in universe_symbol_map.values()
            for symbol_str in symbol_tuple
        )
    )
    row_dict_list: list[dict[str, object]] = []
    for symbol_str in all_symbol_tuple:
        required_column_tuple = tuple((symbol_str, field_str) for field_str in ("Open", "High", "Low", "Close", "Dividend"))
        missing_field_list = [field_str for _, field_str in required_column_tuple if (symbol_str, field_str) not in pricing_data_df.columns]
        if missing_field_list:
            row_dict_list.append(
                {
                    "symbol_str": symbol_str,
                    "status_str": "missing_fields",
                    "missing_field_tuple_str": ",".join(missing_field_list),
                }
            )
            continue

        ohlc_df = pricing_data_df.loc[:, pd.IndexSlice[symbol_str, ("Open", "High", "Low", "Close")]].copy()
        ohlc_df.columns = ohlc_df.columns.get_level_values(1)
        numeric_ohlc_df = ohlc_df.apply(pd.to_numeric, errors="coerce")
        valid_ohlc_bool_ser = (
            numeric_ohlc_df.notna().all(axis=1)
            & numeric_ohlc_df["Open"].gt(0.0)
            & numeric_ohlc_df["High"].gt(0.0)
            & numeric_ohlc_df["Low"].gt(0.0)
            & numeric_ohlc_df["Close"].gt(0.0)
            & numeric_ohlc_df["High"].ge(numeric_ohlc_df["Open"])
            & numeric_ohlc_df["High"].ge(numeric_ohlc_df["Close"])
            & numeric_ohlc_df["High"].ge(numeric_ohlc_df["Low"])
            & numeric_ohlc_df["Low"].le(numeric_ohlc_df["Open"])
            & numeric_ohlc_df["Low"].le(numeric_ohlc_df["Close"])
            & numeric_ohlc_df["Low"].le(numeric_ohlc_df["High"])
        )
        valid_date_index = pricing_data_df.index[valid_ohlc_bool_ser]
        dividend_ser = pd.to_numeric(pricing_data_df[(symbol_str, "Dividend")], errors="coerce").fillna(0.0)
        stale_column_tuple = (symbol_str, STALE_NO_PRINT_FIELD_STR)
        stale_session_count_int = (
            0
            if stale_column_tuple not in pricing_data_df.columns
            else int(pricing_data_df[stale_column_tuple].fillna(False).astype(bool).sum())
        )
        if len(valid_date_index) == 0:
            status_str = "no_valid_ohlc"
            first_valid_date_str = None
            last_valid_date_str = None
            invalid_after_first_count_int = 0
        else:
            status_str = (
                "ok"
                if stale_session_count_int == 0
                else "ok_with_stale_no_print"
            )
            first_valid_date_str = valid_date_index[0].date().isoformat()
            last_valid_date_str = valid_date_index[-1].date().isoformat()
            invalid_after_first_count_int = int((~valid_ohlc_bool_ser.loc[valid_date_index[0] :]).sum())

        row_dict_list.append(
            {
                "symbol_str": symbol_str,
                "status_str": status_str,
                "missing_field_tuple_str": "",
                "first_valid_date_str": first_valid_date_str,
                "last_valid_date_str": last_valid_date_str,
                "valid_ohlc_observation_count_int": int(len(valid_date_index)),
                "invalid_after_first_count_int": invalid_after_first_count_int,
                "stale_session_count_int": stale_session_count_int,
                "dividend_event_count_int": int(dividend_ser.ne(0.0).sum()),
                "universe_membership_tuple_str": ",".join(
                    universe_id_str
                    for universe_id_str, symbol_tuple in universe_symbol_map.items()
                    if symbol_str in symbol_tuple
                ),
            }
        )
    return pd.DataFrame(row_dict_list)


def _summary_metric_float(summary_df: pd.DataFrame | None, metric_name_str: str) -> float:
    if summary_df is None or metric_name_str not in summary_df.index:
        return float("nan")
    value_obj = summary_df.loc[metric_name_str]
    if isinstance(value_obj, pd.Series):
        value_obj = value_obj.iloc[0]
    try:
        return float(value_obj)
    except (TypeError, ValueError):
        return float("nan")


def _annual_return_row_dict_list(
    universe_id_str: str,
    strategy_return_ser: pd.Series,
    equal_weight_return_ser: pd.Series,
) -> list[dict[str, object]]:
    return_row_dict_list: list[dict[str, object]] = []
    for series_name_str, daily_return_ser in (
        ("strategy", strategy_return_ser),
        ("equal_weight", equal_weight_return_ser),
    ):
        normalized_return_ser = pd.to_numeric(daily_return_ser, errors="coerce").dropna()
        for year_int, year_return_ser in normalized_return_ser.groupby(normalized_return_ser.index.year):
            return_row_dict_list.append(
                {
                    "universe_id_str": universe_id_str,
                    "series_name_str": series_name_str,
                    "year_int": int(year_int),
                    "return_pct_float": float(((1.0 + year_return_ser).prod() - 1.0) * 100.0),
                    "observation_count_int": int(len(year_return_ser)),
                }
            )
    return return_row_dict_list


def _subperiod_row_dict_list(
    universe_id_str: str,
    strategy_total_value_ser: pd.Series,
    equal_weight_total_value_ser: pd.Series,
) -> list[dict[str, object]]:
    row_dict_list: list[dict[str, object]] = []
    for period_id_str, start_ts, end_ts in SUBPERIOD_TUPLE:
        for series_name_str, total_value_ser in (
            ("strategy", strategy_total_value_ser),
            ("equal_weight", equal_weight_total_value_ser),
        ):
            period_total_value_ser = total_value_ser.loc[
                (total_value_ser.index >= start_ts) & (total_value_ser.index <= end_ts)
            ]
            metric_dict = compute_equity_metric_dict(
                total_value_ser=period_total_value_ser,
                prefix_str="period",
            )
            row_dict_list.append(
                {
                    "universe_id_str": universe_id_str,
                    "period_id_str": period_id_str,
                    "series_name_str": series_name_str,
                    **metric_dict,
                }
            )
    return row_dict_list


def _run_universe_strategy(
    manifest_row_ser: pd.Series,
    pricing_data_df: pd.DataFrame,
    end_date_str: str,
    show_progress_bool: bool,
) -> tuple[SectorDispersionDividendStrategy, pd.DatetimeIndex, pd.Series, pd.Series]:
    universe_id_str = str(manifest_row_ser["universe_id_str"])
    config_obj = build_universe_config_obj(
        manifest_row_ser=manifest_row_ser,
        end_date_str=end_date_str,
    )
    calendar_idx = build_execution_calendar_idx(
        pricing_data_df=pricing_data_df,
        config_obj=config_obj,
    )
    strategy_obj = SectorDispersionDividendStrategy(
        name=f"strategy_mr_sector_dispersion_ibs_family_{universe_id_str}",
        benchmarks=[config_obj.benchmark_symbol_str],
        config_obj=config_obj,
    )
    strategy_obj._performance_benchmark_symbol_str = config_obj.benchmark_symbol_str
    strategy_obj._performance_benchmark_adjustment_str = "TOTALRETURN"
    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar=calendar_idx,
        show_progress=show_progress_bool,
        show_signal_progress_bool=show_progress_bool,
        audit_override_bool=True,
    )

    equal_weight_return_ser = compute_equal_weight_benchmark_return_ser(
        pricing_data_df=pricing_data_df,
        symbol_tuple=config_obj.symbol_tuple,
        calendar_idx=pd.DatetimeIndex(strategy_obj.results.index),
    )
    equal_weight_total_value_ser = (
        (1.0 + equal_weight_return_ser).cumprod() * config_obj.capital_base_float
    ).rename("equal_weight_total_value")
    return strategy_obj, calendar_idx, equal_weight_return_ser, equal_weight_total_value_ser


def _save_universe_artifacts(
    output_path: Path,
    universe_id_str: str,
    strategy_obj: SectorDispersionDividendStrategy,
    equal_weight_return_ser: pd.Series,
    equal_weight_total_value_ser: pd.Series,
) -> None:
    universe_output_path = output_path / "universes" / universe_id_str
    universe_output_path.mkdir(parents=True, exist_ok=False)
    strategy_obj.results.to_csv(universe_output_path / "daily_results.csv")
    strategy_obj.realized_weight_df.to_csv(universe_output_path / "realized_weights.csv")
    strategy_obj.get_transactions().to_csv(universe_output_path / "transactions.csv", index=False)
    completed_trade_df = getattr(strategy_obj, "_trades", pd.DataFrame())
    if completed_trade_df is None:
        completed_trade_df = pd.DataFrame()
    completed_trade_df.to_csv(universe_output_path / "completed_trades.csv", index=False)
    strategy_obj.dividend_credit_df.to_csv(universe_output_path / "dividend_credits.csv", index=False)
    strategy_obj.stale_order_cancellation_df.to_csv(
        universe_output_path / "stale_order_cancellations.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "equal_weight_return": equal_weight_return_ser,
            "equal_weight_total_value": equal_weight_total_value_ser,
        }
    ).to_csv(universe_output_path / "equal_weight_benchmark.csv")
    if strategy_obj.summary is not None:
        strategy_obj.summary.to_csv(universe_output_path / "summary.csv")
    if strategy_obj.summary_trades is not None:
        strategy_obj.summary_trades.to_csv(universe_output_path / "summary_trades.csv")


def _markdown_table_str(source_df: pd.DataFrame, column_list: list[str]) -> str:
    table_df = source_df.loc[:, column_list].copy()
    header_str = "| " + " | ".join(column_list) + " |"
    separator_str = "| " + " | ".join("---" for _ in column_list) + " |"
    row_str_list = []
    for _, row_ser in table_df.iterrows():
        value_str_list = []
        for value_obj in row_ser.tolist():
            if isinstance(value_obj, float):
                value_str_list.append("" if not np.isfinite(value_obj) else f"{value_obj:.3f}")
            else:
                value_str_list.append(str(value_obj))
        row_str_list.append("| " + " | ".join(value_str_list) + " |")
    return "\n".join([header_str, separator_str, *row_str_list])


def run_family_universe_study(
    output_dir_str: str = "results",
    end_date_str: str = STUDY_END_DATE_STR,
    show_progress_bool: bool = False,
) -> Path:
    universe_manifest_df = build_universe_manifest_df()
    all_symbol_tuple = tuple(
        dict.fromkeys(
            symbol_str
            for _, manifest_row_ser in universe_manifest_df.iterrows()
            for symbol_str in _symbol_tuple_from_manifest_row(manifest_row_ser)
        )
    )
    load_config_obj = replace(
        DEFAULT_CONFIG,
        symbol_tuple=all_symbol_tuple,
        universe_name_str="original",
        benchmark_symbol_str=COMMON_BENCHMARK_SYMBOL_STR,
        history_start_date_str=HISTORY_START_DATE_STR,
        backtest_start_date_str="1998-01-01",
        end_date_str=end_date_str,
    )
    raw_pricing_data_df = get_sector_dispersion_ibs_data(config_obj=load_config_obj)
    pricing_data_df, stale_session_df = prepare_isolated_no_print_sessions(
        pricing_data_df=raw_pricing_data_df,
        universe_manifest_df=universe_manifest_df,
    )
    data_quality_df = build_data_quality_df(
        pricing_data_df=pricing_data_df,
        universe_manifest_df=universe_manifest_df,
    )
    bad_data_quality_df = data_quality_df.loc[
        ~data_quality_df["status_str"].astype(str).str.startswith("ok")
    ]
    if len(bad_data_quality_df) > 0:
        raise RuntimeError(
            "Family-universe data audit failed: "
            f"{bad_data_quality_df[['symbol_str', 'status_str']].to_dict('records')}"
        )

    timestamp_str = pd.Timestamp.now().strftime("%Y-%m-%d_%H%M%S")
    output_path = build_research_output_path(
        output_dir=output_dir_str,
        entity_type_str="strategy",
        entity_id_str="strategy_mr_sector_dispersion_ibs",
        analysis_type_str="family_universe_study",
        timestamp_str=timestamp_str,
    )
    output_path.mkdir(parents=True, exist_ok=False)
    universe_manifest_df.to_csv(output_path / "universe_manifest.csv", index=False)
    data_quality_df.to_csv(output_path / "data_quality.csv", index=False)
    stale_session_df.to_csv(output_path / "stale_sessions.csv", index=False)

    strategy_by_universe_dict: dict[str, SectorDispersionDividendStrategy] = {}
    equal_weight_return_by_universe_dict: dict[str, pd.Series] = {}
    equal_weight_equity_by_universe_dict: dict[str, pd.Series] = {}
    effective_start_by_universe_dict: dict[str, pd.Timestamp] = {}

    for row_index_int, manifest_row_ser in universe_manifest_df.iterrows():
        universe_id_str = str(manifest_row_ser["universe_id_str"])
        print(
            f"Running {row_index_int + 1}/{len(universe_manifest_df)} {universe_id_str}...",
            flush=True,
        )
        strategy_obj, calendar_idx, equal_weight_return_ser, equal_weight_total_value_ser = (
            _run_universe_strategy(
                manifest_row_ser=manifest_row_ser,
                pricing_data_df=pricing_data_df,
                end_date_str=end_date_str,
                show_progress_bool=show_progress_bool,
            )
        )
        strategy_by_universe_dict[universe_id_str] = strategy_obj
        equal_weight_return_by_universe_dict[universe_id_str] = equal_weight_return_ser
        equal_weight_equity_by_universe_dict[universe_id_str] = equal_weight_total_value_ser
        effective_start_by_universe_dict[universe_id_str] = pd.Timestamp(calendar_idx[0])
        _save_universe_artifacts(
            output_path=output_path,
            universe_id_str=universe_id_str,
            strategy_obj=strategy_obj,
            equal_weight_return_ser=equal_weight_return_ser,
            equal_weight_total_value_ser=equal_weight_total_value_ser,
        )

    common_overlap_start_ts = max(effective_start_by_universe_dict.values())
    comparison_row_dict_list: list[dict[str, object]] = []
    subperiod_row_dict_list: list[dict[str, object]] = []
    annual_return_row_dict_list: list[dict[str, object]] = []
    common_equity_df = pd.DataFrame()

    for _, manifest_row_ser in universe_manifest_df.iterrows():
        universe_id_str = str(manifest_row_ser["universe_id_str"])
        strategy_obj = strategy_by_universe_dict[universe_id_str]
        strategy_total_value_ser = pd.to_numeric(
            strategy_obj.results["total_value"],
            errors="coerce",
        ).astype(float)
        strategy_return_ser = pd.to_numeric(
            strategy_obj.results["daily_returns"],
            errors="coerce",
        ).astype(float)
        equal_weight_return_ser = equal_weight_return_by_universe_dict[universe_id_str]
        equal_weight_total_value_ser = equal_weight_equity_by_universe_dict[universe_id_str]
        common_strategy_equity_ser = strategy_total_value_ser.loc[
            strategy_total_value_ser.index >= common_overlap_start_ts
        ]
        common_equal_weight_equity_ser = equal_weight_total_value_ser.loc[
            equal_weight_total_value_ser.index >= common_overlap_start_ts
        ]

        full_strategy_metric_dict = compute_equity_metric_dict(
            total_value_ser=strategy_total_value_ser,
            prefix_str="full_strategy",
        )
        full_benchmark_metric_dict = compute_equity_metric_dict(
            total_value_ser=equal_weight_total_value_ser,
            prefix_str="full_equal_weight",
        )
        common_strategy_metric_dict = compute_equity_metric_dict(
            total_value_ser=common_strategy_equity_ser,
            prefix_str="common_strategy",
        )
        common_benchmark_metric_dict = compute_equity_metric_dict(
            total_value_ser=common_equal_weight_equity_ser,
            prefix_str="common_equal_weight",
        )
        exposure_diagnostic_dict = build_exposure_diagnostic_dict(
            realized_weight_df=strategy_obj.realized_weight_df,
            result_df=strategy_obj.results,
            symbol_tuple=strategy_obj.symbol_tuple,
        )
        universe_stale_session_count_int = int(
            stale_session_df["symbol_str"].isin(strategy_obj.symbol_tuple).sum()
        ) if len(stale_session_df) > 0 else 0
        trade_count_float = _summary_metric_float(strategy_obj.summary_trades, "# Trades")
        comparison_row_dict_list.append(
            {
                "priority_int": int(manifest_row_ser["priority_int"]),
                "universe_id_str": universe_id_str,
                "symbol_count_int": len(strategy_obj.symbol_tuple),
                "symbol_tuple_str": ",".join(strategy_obj.symbol_tuple),
                "research_role_str": str(manifest_row_ser["research_role_str"]),
                "target_weight_pct_float": strategy_obj.target_weight_float * 100.0,
                "turnover_ann_pct_float": _summary_metric_float(strategy_obj.summary, "Turnover (Ann.) [%]"),
                "cost_drag_ann_pct_float": _summary_metric_float(strategy_obj.summary, "Cost Drag (Ann.) [%]"),
                "trade_count_int": None if not np.isfinite(trade_count_float) else int(trade_count_float),
                "dividend_cash_total_float": strategy_obj.dividend_cash_total_float,
                "stale_session_count_int": universe_stale_session_count_int,
                "stale_order_cancellation_count_int": (
                    strategy_obj.stale_order_cancellation_count_int
                ),
                "strategy_minus_equal_weight_ann_return_pct_float": (
                    float(full_strategy_metric_dict["full_strategy_ann_return_pct_float"])
                    - float(full_benchmark_metric_dict["full_equal_weight_ann_return_pct_float"])
                ),
                "common_strategy_minus_equal_weight_ann_return_pct_float": (
                    float(common_strategy_metric_dict["common_strategy_ann_return_pct_float"])
                    - float(common_benchmark_metric_dict["common_equal_weight_ann_return_pct_float"])
                ),
                "forced_liquidation_count_int": 0,
                # Resolved stale sessions are counted separately above. This
                # field is reserved for unresolved/fatal issues in a completed
                # universe run.
                "unresolved_data_quality_issue_count_int": 0,
                **exposure_diagnostic_dict,
                **full_strategy_metric_dict,
                **full_benchmark_metric_dict,
                **common_strategy_metric_dict,
                **common_benchmark_metric_dict,
            }
        )
        subperiod_row_dict_list.extend(
            _subperiod_row_dict_list(
                universe_id_str=universe_id_str,
                strategy_total_value_ser=strategy_total_value_ser,
                equal_weight_total_value_ser=equal_weight_total_value_ser,
            )
        )
        annual_return_row_dict_list.extend(
            _annual_return_row_dict_list(
                universe_id_str=universe_id_str,
                strategy_return_ser=strategy_return_ser,
                equal_weight_return_ser=equal_weight_return_ser,
            )
        )

        common_equity_df[f"{universe_id_str}_strategy"] = (
            common_strategy_equity_ser / common_strategy_equity_ser.iloc[0]
        )
        common_equity_df[f"{universe_id_str}_equal_weight"] = (
            common_equal_weight_equity_ser / common_equal_weight_equity_ser.iloc[0]
        )

    comparison_df = pd.DataFrame(comparison_row_dict_list).sort_values("priority_int")
    subperiod_df = pd.DataFrame(subperiod_row_dict_list)
    annual_return_df = pd.DataFrame(annual_return_row_dict_list)
    comparison_df.to_csv(output_path / "comparison.csv", index=False)
    subperiod_df.to_csv(output_path / "subperiod_metrics.csv", index=False)
    annual_return_df.to_csv(output_path / "annual_returns.csv", index=False)
    common_equity_df.to_csv(output_path / "common_overlap_equity.csv")

    summary_column_list = [
        "universe_id_str",
        "full_strategy_ann_return_pct_float",
        "full_strategy_sharpe_float",
        "full_strategy_max_drawdown_pct_float",
        "full_equal_weight_ann_return_pct_float",
        "average_gross_exposure_pct_float",
        "turnover_ann_pct_float",
        "cost_drag_ann_pct_float",
        "negative_cash_day_count_int",
    ]
    summary_md_str = f"""# Sector Dispersion IBS Family-Universe Study

This output is governed by `docs/research/SECTOR_DISPERSION_FAMILY_UNIVERSE_PREREGISTRATION.md`.

- Search space: `{SEARCH_SPACE_COUNT_INT}` frozen universe rows.
- Signal timing: completed daily bar `T`.
- Execution: `Open_T+1`.
- Sizing: unlevered `1/N` target per ETF.
- End date: `{end_date_str}`.
- Common overlap start: `{common_overlap_start_ts.date().isoformat()}`.
- Idle cash yield: `0%`.

## Mechanical Result Table

{_markdown_table_str(comparison_df, summary_column_list)}

This table is evidence, not an automatic winner selection. The final verdict
must inspect subperiods, annual returns, costs, exposure drift, dividends,
trade concentration, and the known classification differences.
"""
    (output_path / "study_summary.md").write_text(summary_md_str, encoding="utf-8")

    metadata_dict = {
        "strategy_id_str": "strategy_mr_sector_dispersion_ibs",
        "analysis_type_str": "family_universe_study",
        "research_only_bool": True,
        "search_space_count_int": SEARCH_SPACE_COUNT_INT,
        "output_path_str": str(output_path.resolve()),
        "study_end_date_str": end_date_str,
        "common_overlap_start_date_str": common_overlap_start_ts.date().isoformat(),
        "history_start_date_str": HISTORY_START_DATE_STR,
        "benchmark_symbol_str": COMMON_BENCHMARK_SYMBOL_STR,
        "execution_mapping_str": "completed daily signal T -> Open T+1",
        "target_weight_formula_str": "1 / fixed universe N",
        "portfolio_leverage_float": 1.0,
        "slippage_float": DEFAULT_CONFIG.slippage_float,
        "commission_per_share_float": DEFAULT_CONFIG.commission_per_share_float,
        "commission_minimum_float": DEFAULT_CONFIG.commission_minimum_float,
        "cash_yield_float": 0.0,
        "tradable_adjustment_str": "CAPITALSPECIAL",
        "dividend_policy_str": "credit cash distribution to pre-open held shares",
        "stale_no_print_policy_str": (
            "prior-close valuation, zero range, and cancel same-symbol open orders"
        ),
        "stale_session_count_int": int(len(stale_session_df)),
        "market_benchmark_adjustment_str": "TOTALRETURN",
        "effective_start_by_universe_dict": {
            universe_id_str: start_ts.date().isoformat()
            for universe_id_str, start_ts in effective_start_by_universe_dict.items()
        },
        "preregistration_path_str": str(
            (REPO_ROOT_PATH / "docs" / "research" / "SECTOR_DISPERSION_FAMILY_UNIVERSE_PREREGISTRATION.md").resolve()
        ),
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata_dict, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Saved family-universe study to {output_path.resolve()}", flush=True)
    return output_path


def parse_args(argv_list: list[str] | None = None) -> argparse.Namespace:
    parser_obj = argparse.ArgumentParser(
        description="Run the preregistered six-universe Sector Dispersion IBS study."
    )
    parser_obj.add_argument("--output-dir", default="results", help="Root output directory.")
    parser_obj.add_argument(
        "--end-date",
        default=STUDY_END_DATE_STR,
        help="Inclusive Norgate end date. The preregistered run uses 2026-07-17.",
    )
    parser_obj.add_argument("--show-progress", action="store_true", help="Show backtest progress bars.")
    return parser_obj.parse_args(argv_list)


def main(argv_list: list[str] | None = None) -> None:
    args_obj = parse_args(argv_list)
    run_family_universe_study(
        output_dir_str=str(args_obj.output_dir),
        end_date_str=str(args_obj.end_date),
        show_progress_bool=bool(args_obj.show_progress),
    )


if __name__ == "__main__":
    main()
