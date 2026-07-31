"""Shared research-only implementation for the stateful HPI long strategy.

For signal date T:

    Return3D_T = Close_T / Close_(T-3) - 1

Let P_T contain the previous 1,260 Return3D observations, excluding T.

For Return3D_T <= 0:

    HPI_T = 100 * count(x_i <= Return3D_T, i in P_T)
                    / count(x_i <= 0, i in P_T)

For Return3D_T > 0:

    HPI_T = 100 * count(x_i > Return3D_T, i in P_T)
                    / count(x_i > 0, i in P_T)

Entry requires HPI < 30, Return3D < 0, IBS < 0.10, Close > SMA200,
and point-in-time index membership. Exit requires IBS > 0.90, RSI2 > 90,
or loss of point-in-time membership. Decisions use Close_T information and
market orders execute at Open_(T+1) under the Vanilla engine contract.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import talib
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from alpha.indicators import ibs_indicator
from data.norgate_loader import (
    CAPITALSPECIAL_ADJUSTMENT_STR,
    TOTALRETURN_ADJUSTMENT_STR,
    build_index_constituent_matrix,
    load_raw_prices,
)
from data.norgate_snapshot_store import (
    HPI_SP500_PROFILE_STR,
    get_active_data_profile_str,
    is_snapshot_mode_enabled_bool,
)


HPI_LOOKBACK_INT = 1_260
HPI_THRESHOLD_FLOAT = 30.0
RETURN_LOOKBACK_INT = 3
SMA_WINDOW_INT = 200
MAX_ENTRY_IBS_FLOAT = 0.10
EXIT_IBS_THRESHOLD_FLOAT = 0.90
RSI_WINDOW_INT = 2
EXIT_RSI2_THRESHOLD_FLOAT = 90.0
NATR_WINDOW_INT = 14
MAX_POSITIONS_INT = 10
TURNOVER_FIELD_STR = "Turnover"
NATR_FIELD_STR = "natr_14_ser"
RANKING_FIELD_SET = {TURNOVER_FIELD_STR, NATR_FIELD_STR}
ENTRY_BASELINE_STR = "baseline"
ENTRY_HORIZON_VOTE_STR = "hpi_2_3_5_vote"
ENTRY_MODE_SET = {ENTRY_BASELINE_STR, ENTRY_HORIZON_VOTE_STR}
LIQUIDITY_NONE_STR = "none"
LIQUIDITY_RELATIVE_STR = "raw_price_5_adv63_above_median"
LIQUIDITY_MODE_SET = {LIQUIDITY_NONE_STR, LIQUIDITY_RELATIVE_STR}
RETURN_2D_FIELD_STR = "return_2d_ser"
RETURN_5D_FIELD_STR = "return_5d_ser"
HPI_2D_FIELD_STR = "hpi_2d_ser"
HPI_5D_FIELD_STR = "hpi_5d_ser"
RAW_PRICE_FIELD_STR = "raw_price_ser"
ADV_63_FIELD_STR = "adv_63_ser"
RAW_PRICE_MIN_FLOAT = 5.0


def default_trade_id_int() -> int:
    return -1


def compute_strict_hpi(
    return_3d_ser: pd.Series,
    lookback_int: int = HPI_LOOKBACK_INT,
) -> pd.Series:
    """Compute HPI from prior observations only, using the supplied tie rules."""

    if lookback_int < 2:
        raise ValueError("lookback_int must be at least 2.")

    full_window_int = lookback_int + 1

    # *** CRITICAL*** lookahead-sensitive rolling boundary: this temporary
    # window is [T-lookback, ..., T]. T is removed algebraically below, leaving
    # exactly the prior `lookback_int` observations as the HPI reference set.
    rolling_rank_ser = return_3d_ser.rolling(
        window=full_window_int,
        min_periods=full_window_int,
    ).rank(method="max")
    nonpositive_count_ser = return_3d_ser.le(0.0).rolling(
        window=full_window_int,
        min_periods=full_window_int,
    ).sum()
    positive_count_ser = return_3d_ser.gt(0.0).rolling(
        window=full_window_int,
        min_periods=full_window_int,
    ).sum()

    prior_leq_count_ser = rolling_rank_ser - 1.0
    prior_nonpositive_count_ser = (
        nonpositive_count_ser - return_3d_ser.le(0.0).astype(float)
    )
    prior_greater_count_ser = float(full_window_int) - rolling_rank_ser
    prior_positive_count_ser = positive_count_ser - return_3d_ser.gt(0.0).astype(float)

    hpi_value_ser = pd.Series(np.nan, index=return_3d_ser.index, dtype=float)
    nonpositive_mask_ser = (
        return_3d_ser.le(0.0) & prior_nonpositive_count_ser.gt(0.0)
    )
    positive_mask_ser = return_3d_ser.gt(0.0) & prior_positive_count_ser.gt(0.0)

    hpi_value_ser.loc[nonpositive_mask_ser] = (
        100.0
        * prior_leq_count_ser.loc[nonpositive_mask_ser]
        / prior_nonpositive_count_ser.loc[nonpositive_mask_ser]
    )
    hpi_value_ser.loc[positive_mask_ser] = (
        100.0
        * prior_greater_count_ser.loc[positive_mask_ser]
        / prior_positive_count_ser.loc[positive_mask_ser]
    )
    return hpi_value_ser.clip(lower=0.0, upper=100.0)


def get_asof_universe_symbol_set(
    universe_df: pd.DataFrame,
    decision_date_ts: pd.Timestamp,
) -> set[str]:
    """Return point-in-time members from the latest row known by decision time."""

    sorted_universe_df = (
        universe_df
        if universe_df.index.is_monotonic_increasing
        else universe_df.sort_index()
    )
    # *** CRITICAL*** PIT membership uses only the latest universe row on or
    # before T. A later row would leak future index membership.
    universe_row_int = int(
        sorted_universe_df.index.searchsorted(decision_date_ts, side="right")
    ) - 1
    if universe_row_int < 0:
        return set()

    universe_membership_ser = sorted_universe_df.iloc[universe_row_int]
    return set(
        universe_membership_ser[universe_membership_ser.eq(1)].index.astype(str)
    )


def load_exact_hpi_inputs(
    indexname_str: str,
    benchmark_symbol_str: str,
    start_date_str: str,
    end_date_str: str | None,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """Load exact PIT membership and no-padding prices for HPI."""

    if is_snapshot_mode_enabled_bool():
        active_profile_str = get_active_data_profile_str()
        if active_profile_str != HPI_SP500_PROFILE_STR:
            raise RuntimeError(
                "Strict HPI snapshot mode requires data profile "
                f"'{HPI_SP500_PROFILE_STR}', got {active_profile_str!r}."
            )
        symbol_list, universe_df = build_index_constituent_matrix(
            indexname=indexname_str
        )
        pricing_data_df = load_raw_prices(
            symbol_list,
            [benchmark_symbol_str],
            start_date=start_date_str,
            end_date=end_date_str,
        )
    else:
        import norgatedata

        watchlist_symbol_list = list(
            norgatedata.watchlist_symbols(f"{indexname_str} Current & Past")
        )
        membership_ser_list: list[pd.Series] = []
        symbol_list = []
        for symbol_str in watchlist_symbol_list:
            membership_df = norgatedata.index_constituent_timeseries(
                symbol_str,
                indexname_str,
                start_date=start_date_str,
                end_date=end_date_str,
                timeseriesformat="pandas-dataframe",
            )
            if membership_df is None or membership_df.empty:
                continue
            membership_ser = membership_df["Index Constituent"].rename(
                symbol_str
            )
            if not membership_ser.eq(1).any():
                continue
            membership_ser_list.append(membership_ser)
            symbol_list.append(symbol_str)

        if not membership_ser_list:
            raise RuntimeError(f"No PIT members found for {indexname_str}.")
        universe_df = (
            pd.concat(membership_ser_list, axis=1)
            .fillna(0)
            .astype(int)
            .sort_index()
        )

        pricing_frame_list: list[pd.DataFrame] = []
        for symbol_str in symbol_list + [benchmark_symbol_str]:
            adjustment_obj = (
                norgatedata.StockPriceAdjustmentType.TOTALRETURN
                if symbol_str == benchmark_symbol_str
                else norgatedata.StockPriceAdjustmentType.CAPITALSPECIAL
            )
            price_df = norgatedata.price_timeseries(
                symbol_str,
                stock_price_adjustment_setting=adjustment_obj,
                padding_setting=norgatedata.PaddingType.NONE,
                start_date=start_date_str,
                end_date=end_date_str,
                timeseriesformat="pandas-dataframe",
            )
            if price_df is None or price_df.empty:
                continue
            price_df.columns = pd.MultiIndex.from_tuples(
                [(symbol_str, field_str) for field_str in price_df.columns]
            )
            pricing_frame_list.append(price_df)

        if not pricing_frame_list:
            raise RuntimeError(f"No prices loaded for {indexname_str}.")
        pricing_data_df = pd.concat(pricing_frame_list, axis=1).sort_index()
    for symbol_str in symbol_list + [benchmark_symbol_str]:
        observed_price_key_list = [
            (symbol_str, field_str)
            for field_str in ("Open", "High", "Low", "Close")
            if (symbol_str, field_str) in pricing_data_df.columns
        ]
        dividend_key = (symbol_str, "Dividend")
        if observed_price_key_list and dividend_key in pricing_data_df.columns:
            # *** CRITICAL*** A union-calendar row with no OHLC observation is
            # synthetic, so it cannot carry a Norgate dividend event. Preserve
            # NaN on observed sessions so malformed source data still fails loud.
            observed_session_bool_ser = pricing_data_df[
                observed_price_key_list
            ].notna().any(axis=1)
            synthetic_session_bool_ser = ~observed_session_bool_ser
            pricing_data_df.loc[
                synthetic_session_bool_ser
                & pricing_data_df[dividend_key].isna(),
                dividend_key,
            ] = 0.0

        close_key = (symbol_str, "Close")
        if close_key in pricing_data_df.columns:
            # *** CRITICAL*** No-padding preserves the feature observation
            # clock. Forward-filled Close is valuation-only on non-trading
            # sessions; Open/High/Low stay NaN, so no synthetic trade can fill.
            pricing_data_df[close_key] = pricing_data_df[close_key].ffill()
    return symbol_list, universe_df, pricing_data_df


class HPIStatefulLongStrategy(Strategy):
    """Stateful HPI long strategy with fixed equal-slot sizing."""

    def __init__(
        self,
        name: str,
        benchmarks: list[str] | tuple[str, ...],
        ranking_field_str: str,
        capital_base: float = 100_000.0,
        slippage: float = 0.00025,
        commission_per_share: float = 0.005,
        commission_minimum: float = 1.0,
        max_positions_int: int = MAX_POSITIONS_INT,
        entry_mode_str: str = ENTRY_BASELINE_STR,
        liquidity_mode_str: str = LIQUIDITY_NONE_STR,
        backtest_start_date_str: str | None = None,
    ) -> None:
        super().__init__(
            name=name,
            benchmarks=list(benchmarks),
            capital_base=capital_base,
            slippage=slippage,
            commission_per_share=commission_per_share,
            commission_minimum=commission_minimum,
            performance_benchmark_adjustment_str=TOTALRETURN_ADJUSTMENT_STR,
        )
        self._data_adjustment_policy_dict.update(
            {
                "stock_signal_adjustment_str": CAPITALSPECIAL_ADJUSTMENT_STR,
                "execution_and_marks_adjustment_str": (
                    CAPITALSPECIAL_ADJUSTMENT_STR
                ),
                "performance_benchmark_adjustment_str": (
                    TOTALRETURN_ADJUSTMENT_STR
                ),
            }
        )
        if ranking_field_str not in RANKING_FIELD_SET:
            raise ValueError(f"Unsupported ranking field: {ranking_field_str}")
        if max_positions_int <= 0:
            raise ValueError("max_positions_int must be positive.")
        if entry_mode_str not in ENTRY_MODE_SET:
            raise ValueError(f"Unsupported entry mode: {entry_mode_str}")
        if liquidity_mode_str not in LIQUIDITY_MODE_SET:
            raise ValueError(f"Unsupported liquidity mode: {liquidity_mode_str}")

        self.ranking_field_str = ranking_field_str
        self.max_positions_int = max_positions_int
        self.entry_mode_str = entry_mode_str
        self.liquidity_mode_str = liquidity_mode_str
        self.backtest_start_date_ts = (
            pd.Timestamp(backtest_start_date_str)
            if backtest_start_date_str is not None
            else None
        )
        self.trade_id_int = 0
        self.current_trade_map: defaultdict[str, int] = defaultdict(
            default_trade_id_int
        )
        self.pending_exit_symbol_set: set[str] = set()
        self.universe_df: pd.DataFrame | None = None

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        signal_data_df = pricing_data_df.copy()
        symbol_list = signal_data_df.columns.get_level_values(0).unique()
        tradeable_symbol_list = [
            str(symbol_str)
            for symbol_str in symbol_list
            if not str(symbol_str).startswith("$")
            and (symbol_str, "Close") in signal_data_df.columns
            and (symbol_str, "High") in signal_data_df.columns
            and (symbol_str, "Low") in signal_data_df.columns
        ]
        if self.liquidity_mode_str == LIQUIDITY_RELATIVE_STR:
            missing_input_list: list[str] = []
            for symbol_str in tradeable_symbol_list:
                raw_close_column_tuple = (symbol_str, "Unadjusted Close")
                volume_column_tuple = (symbol_str, "Volume")
                for column_tuple in (raw_close_column_tuple, volume_column_tuple):
                    if column_tuple not in pricing_data_df.columns:
                        missing_input_list.append(".".join(column_tuple))
                if (
                    raw_close_column_tuple not in pricing_data_df.columns
                    or volume_column_tuple not in pricing_data_df.columns
                ):
                    continue

                raw_close_ser = pd.to_numeric(
                    pricing_data_df[raw_close_column_tuple],
                    errors="coerce",
                )
                volume_ser = pd.to_numeric(
                    pricing_data_df[volume_column_tuple],
                    errors="coerce",
                )
                observed_liquidity_ser = (
                    raw_close_ser.notna()
                    & volume_ser.notna()
                    & np.isfinite(raw_close_ser)
                    & np.isfinite(volume_ser)
                )
                if not observed_liquidity_ser.any():
                    missing_input_list.append(
                        f"{symbol_str}.Unadjusted Close/Volume has no "
                        "overlapping finite observations"
                    )
            if missing_input_list:
                missing_input_str = ", ".join(missing_input_list)
                raise RuntimeError(
                    "Relative-liquidity HPI requires raw price and volume for "
                    f"every tradeable symbol. Missing: {missing_input_str}"
                )

        feature_ser_dict: dict[tuple[str, str], pd.Series] = {}
        for symbol_str in tradeable_symbol_list:
            symbol_price_df = pd.DataFrame(
                {
                    "Close": signal_data_df[(symbol_str, "Close")],
                    "High": signal_data_df[(symbol_str, "High")],
                    "Low": signal_data_df[(symbol_str, "Low")],
                }
            ).dropna(subset=["Close", "High", "Low"])
            if symbol_price_df.empty:
                continue

            close_price_ser = symbol_price_df["Close"].astype(float)
            high_price_ser = symbol_price_df["High"].astype(float)
            low_price_ser = symbol_price_df["Low"].astype(float)

            # *** CRITICAL*** Return3D_T uses Close_T and the third prior valid
            # close observation only. It is evaluated after Close_T.
            return_3d_ser = close_price_ser / close_price_ser.shift(
                RETURN_LOOKBACK_INT
            ) - 1.0
            hpi_value_ser = compute_strict_hpi(return_3d_ser)

            # *** CRITICAL*** SMA200 ends at Close_T and is used only for an
            # order that executes at Open_(T+1).
            sma_200_price_ser = close_price_ser.rolling(
                window=SMA_WINDOW_INT,
                min_periods=SMA_WINDOW_INT,
            ).mean()
            ibs_value_ser = ibs_indicator(
                close_price_ser,
                high_price_ser,
                low_price_ser,
            )
            rsi2_value_ser = pd.Series(
                talib.RSI(
                    close_price_ser.to_numpy(dtype=float),
                    timeperiod=RSI_WINDOW_INT,
                ),
                index=close_price_ser.index,
                dtype=float,
            )

            symbol_feature_ser_dict = {
                "return_3d_ser": return_3d_ser,
                "hpi_value_ser": hpi_value_ser,
                "sma_200_price_ser": sma_200_price_ser,
                "ibs_value_ser": ibs_value_ser,
                "rsi2_value_ser": rsi2_value_ser,
            }
            if self.entry_mode_str == ENTRY_HORIZON_VOTE_STR:
                for horizon_int, return_field_str, hpi_field_str in (
                    (2, RETURN_2D_FIELD_STR, HPI_2D_FIELD_STR),
                    (5, RETURN_5D_FIELD_STR, HPI_5D_FIELD_STR),
                ):
                    # *** CRITICAL*** Return_w,T uses Close_T and the w-th
                    # prior valid close. HPI excludes T from its reference.
                    return_ser = (
                        close_price_ser / close_price_ser.shift(horizon_int) - 1.0
                    )
                    symbol_feature_ser_dict[return_field_str] = return_ser
                    symbol_feature_ser_dict[hpi_field_str] = compute_strict_hpi(
                        return_ser
                    )
            if self.liquidity_mode_str == LIQUIDITY_RELATIVE_STR:
                raw_liquidity_df = pd.DataFrame(
                    {
                        "raw_close": pricing_data_df[
                            (symbol_str, "Unadjusted Close")
                        ],
                        "volume": pricing_data_df[(symbol_str, "Volume")],
                    }
                ).dropna()
                raw_price_ser = raw_liquidity_df["raw_close"].astype(float)
                dollar_volume_ser = (
                    raw_price_ser * raw_liquidity_df["volume"].astype(float)
                )
                # *** CRITICAL*** ADV63_T uses raw observations from
                # [T-62, T], known after Close_T for Open_(T+1).
                symbol_feature_ser_dict[RAW_PRICE_FIELD_STR] = raw_price_ser
                symbol_feature_ser_dict[ADV_63_FIELD_STR] = (
                    dollar_volume_ser.rolling(
                        window=63,
                        min_periods=63,
                    ).mean()
                )
            if self.ranking_field_str == NATR_FIELD_STR:
                # *** CRITICAL*** NATR14 uses OHLC through T only and ranks
                # candidates for execution at Open_(T+1).
                symbol_feature_ser_dict[NATR_FIELD_STR] = pd.Series(
                    talib.NATR(
                        high_price_ser.to_numpy(dtype=float),
                        low_price_ser.to_numpy(dtype=float),
                        close_price_ser.to_numpy(dtype=float),
                        timeperiod=NATR_WINDOW_INT,
                    ),
                    index=close_price_ser.index,
                    dtype=float,
                )

            for field_str, feature_ser in symbol_feature_ser_dict.items():
                feature_ser_dict[(symbol_str, field_str)] = feature_ser.reindex(
                    signal_data_df.index
                )

        if not feature_ser_dict:
            return signal_data_df

        feature_df = pd.DataFrame(feature_ser_dict, index=signal_data_df.index)
        if self.liquidity_mode_str == LIQUIDITY_RELATIVE_STR:
            execution_feature_df = feature_df
            if self.backtest_start_date_ts is not None:
                execution_feature_df = feature_df.loc[
                    feature_df.index >= self.backtest_start_date_ts
                ]
            adv_column_list = [
                column_tuple
                for column_tuple in execution_feature_df.columns
                if column_tuple[1] == ADV_63_FIELD_STR
            ]
            adv_value_arr = execution_feature_df[adv_column_list].to_numpy(
                dtype=float
            )
            if not adv_column_list or not np.isfinite(adv_value_arr).any():
                start_label_str = (
                    str(self.backtest_start_date_ts.date())
                    if self.backtest_start_date_ts is not None
                    else "the available history"
                )
                raise RuntimeError(
                    "Relative-liquidity HPI has no usable ADV63 data from "
                    f"{start_label_str}."
                )
        return pd.concat([signal_data_df, feature_df], axis=1)

    def iterate(
        self,
        data_df: pd.DataFrame,
        close_row_ser: pd.Series,
        open_price_ser: pd.Series,
    ) -> None:
        if data_df is None or close_row_ser is None:
            return
        if self.universe_df is None:
            raise RuntimeError("HPI strategy requires a point-in-time universe.")

        decision_date_ts = pd.Timestamp(self.previous_bar)
        member_symbol_set = get_asof_universe_symbol_set(
            self.universe_df,
            decision_date_ts,
        )
        position_ser = self.get_positions()
        long_position_ser = position_ser[position_ser > 0]
        long_symbol_set = set(long_position_ser.index.astype(str))
        self.pending_exit_symbol_set.intersection_update(long_symbol_set)
        long_slots_int = self.max_positions_int - len(long_position_ser)

        for symbol_str in long_position_ser.index.astype(str):
            ibs_value_float = close_row_ser.get((symbol_str, "ibs_value_ser"), np.nan)
            rsi2_value_float = close_row_ser.get(
                (symbol_str, "rsi2_value_ser"),
                np.nan,
            )
            exit_for_ibs_bool = (
                pd.notna(ibs_value_float)
                and float(ibs_value_float) > EXIT_IBS_THRESHOLD_FLOAT
            )
            exit_for_rsi2_bool = (
                pd.notna(rsi2_value_float)
                and float(rsi2_value_float) > EXIT_RSI2_THRESHOLD_FLOAT
            )
            exit_for_membership_bool = symbol_str not in member_symbol_set

            if exit_for_ibs_bool or exit_for_rsi2_bool or exit_for_membership_bool:
                self.pending_exit_symbol_set.add(symbol_str)

            current_open_float = open_price_ser.get(symbol_str, np.nan)
            has_tradable_open_bool = (
                pd.notna(current_open_float)
                and np.isfinite(float(current_open_float))
            )
            if symbol_str in self.pending_exit_symbol_set and has_tradable_open_bool:
                self.order_target_value(
                    symbol_str,
                    0.0,
                    trade_id=self.current_trade_map[symbol_str],
                )
                # *** CRITICAL*** Backtests know this Open-(T+1) is finite and
                # therefore model the exit fill before reusing the slot. The
                # live host passes no open here because the future auction fill
                # is unknown; live conservatively waits for broker confirmation.
                long_slots_int += 1

        capital_per_trade_float = self.previous_total_value / float(
            self.max_positions_int
        )
        opportunity_symbol_list = self.get_opportunity_list(
            close_row_ser,
            member_symbol_set,
        )

        while long_slots_int > 0 and opportunity_symbol_list:
            symbol_str = opportunity_symbol_list.pop(0)
            if self.get_position(symbol_str) != 0:
                continue

            self.trade_id_int += 1
            self.current_trade_map[symbol_str] = self.trade_id_int
            self.order_value(
                symbol_str,
                capital_per_trade_float,
                trade_id=self.trade_id_int,
            )
            long_slots_int -= 1

    def _liquidate_missing_price_positions(
        self,
        prices: pd.DataFrame,
    ) -> tuple[float, float]:
        """Liquidate removed symbols; defer missing opens for current members."""

        if self.universe_df is None or self.current_bar is None:
            return 0.0, 0.0

        # *** CRITICAL*** Current-session PIT membership identifies a symbol
        # discontinuity only. The fallback price remains capped at previous_bar,
        # so the liquidation cannot use a future close.
        current_member_symbol_set = get_asof_universe_symbol_set(
            self.universe_df,
            pd.Timestamp(self.current_bar),
        )
        active_position_ser = self.get_positions()
        active_position_ser = active_position_ser[active_position_ser != 0]
        transaction_value_sum_float = 0.0
        commission_sum_float = 0.0

        for asset_obj in active_position_ser.index:
            asset_str = str(asset_obj)
            current_open_key = (asset_str, "Open")
            current_open_float = np.nan
            if current_open_key in prices.columns:
                current_open_value_obj = prices.loc[
                    self.current_bar,
                    current_open_key,
                ]
                if pd.notna(current_open_value_obj):
                    current_open_float = float(current_open_value_obj)

            if asset_str in current_member_symbol_set or np.isfinite(
                current_open_float
            ):
                continue

            liquidation_bar_ts, liquidation_price_float = (
                self._get_last_available_close_before_current_bar(
                    prices=prices,
                    asset_str=asset_str,
                )
            )
            open_trade_amount_ser = self._get_open_trade_amount_ser(
                asset_str=asset_str
            )
            if len(open_trade_amount_ser) == 0:
                raise RuntimeError(
                    f"Found a live position in {asset_str} without open trades."
                )

            print(
                f"Removed asset {asset_str} has no open on {self.current_bar}; "
                f"liquidating at last close from {liquidation_bar_ts.date()}."
            )
            self.log_audit_event(
                "hpi.pit_removal_position_liquidated",
                {
                    "asset_str": asset_str,
                    "liquidation_bar_timestamp_str": liquidation_bar_ts.isoformat(),
                    "liquidation_price_float": liquidation_price_float,
                    "open_trade_count_int": int(len(open_trade_amount_ser)),
                },
            )
            self.clear_orders(asset=asset_str)

            for trade_id_obj, open_amount_float in open_trade_amount_ser.items():
                liquidation_amount_float = -float(open_amount_float)
                commission_float = float(
                    self._compute_commission(liquidation_amount_float)
                )
                liquidation_value_float = (
                    liquidation_amount_float * liquidation_price_float
                )
                self.add_transaction(
                    trade_id_obj,
                    self.current_bar,
                    asset_str,
                    liquidation_amount_float,
                    liquidation_price_float,
                    liquidation_value_float,
                    order_id=-1,
                    commission=commission_float,
                )
                transaction_value_sum_float += liquidation_value_float
                commission_sum_float += commission_float

            self.pending_exit_symbol_set.discard(asset_str)

        return transaction_value_sum_float, commission_sum_float

    def get_opportunity_list(
        self,
        close_row_ser: pd.Series,
        member_symbol_set: set[str] | None = None,
    ) -> list[str]:
        if self.universe_df is None:
            raise RuntimeError("HPI strategy requires a point-in-time universe.")
        if member_symbol_set is None:
            member_symbol_set = get_asof_universe_symbol_set(
                self.universe_df,
                pd.Timestamp(self.previous_bar),
            )

        candidate_df = close_row_ser.unstack()
        candidate_df = candidate_df[
            ~candidate_df.index.astype(str).str.startswith("$")
        ]
        candidate_df = candidate_df[
            candidate_df.index.astype(str).isin(member_symbol_set)
        ]
        if self.liquidity_mode_str == LIQUIDITY_RELATIVE_STR:
            liquidity_field_list = [RAW_PRICE_FIELD_STR, ADV_63_FIELD_STR]
            if any(
                field_str not in candidate_df.columns
                for field_str in liquidity_field_list
            ):
                return []
            median_adv_float = float(
                candidate_df[ADV_63_FIELD_STR].dropna().astype(float).median()
            )
            if not np.isfinite(median_adv_float):
                return []
            candidate_df = candidate_df.dropna(subset=liquidity_field_list)
            candidate_df = candidate_df[
                candidate_df[RAW_PRICE_FIELD_STR].astype(float)
                > RAW_PRICE_MIN_FLOAT
            ]
            candidate_df = candidate_df[
                candidate_df[ADV_63_FIELD_STR].astype(float) > median_adv_float
            ]
        required_field_list = [
            "Close",
            self.ranking_field_str,
            "sma_200_price_ser",
            "ibs_value_ser",
        ]
        if self.entry_mode_str == ENTRY_HORIZON_VOTE_STR:
            required_field_list.extend(
                [
                    RETURN_2D_FIELD_STR,
                    "return_3d_ser",
                    RETURN_5D_FIELD_STR,
                    HPI_2D_FIELD_STR,
                    "hpi_value_ser",
                    HPI_5D_FIELD_STR,
                ]
            )
        else:
            required_field_list.extend(["return_3d_ser", "hpi_value_ser"])
        if any(
            field_str not in candidate_df.columns
            for field_str in required_field_list
        ):
            return []

        candidate_df = candidate_df.dropna(subset=required_field_list)
        if self.entry_mode_str == ENTRY_HORIZON_VOTE_STR:
            hpi_vote_ser = (
                (
                    candidate_df[RETURN_2D_FIELD_STR].astype(float).lt(0.0)
                    & candidate_df[HPI_2D_FIELD_STR]
                    .astype(float)
                    .lt(HPI_THRESHOLD_FLOAT)
                ).astype(int)
                + (
                    candidate_df["return_3d_ser"].astype(float).lt(0.0)
                    & candidate_df["hpi_value_ser"]
                    .astype(float)
                    .lt(HPI_THRESHOLD_FLOAT)
                ).astype(int)
                + (
                    candidate_df[RETURN_5D_FIELD_STR].astype(float).lt(0.0)
                    & candidate_df[HPI_5D_FIELD_STR]
                    .astype(float)
                    .lt(HPI_THRESHOLD_FLOAT)
                ).astype(int)
            )
            candidate_df = candidate_df[hpi_vote_ser.ge(2)]
        else:
            candidate_df = candidate_df[
                candidate_df["hpi_value_ser"].astype(float)
                < HPI_THRESHOLD_FLOAT
            ]
            candidate_df = candidate_df[
                candidate_df["return_3d_ser"].astype(float) < 0.0
            ]
        candidate_df = candidate_df[
            candidate_df["ibs_value_ser"].astype(float) < MAX_ENTRY_IBS_FLOAT
        ]
        candidate_df = candidate_df[
            candidate_df["Close"].astype(float)
            > candidate_df["sma_200_price_ser"].astype(float)
        ]
        candidate_df = candidate_df.assign(
            symbol_str=candidate_df.index.astype(str)
        ).sort_values(
            by=[self.ranking_field_str, "symbol_str"],
            ascending=[False, True],
            kind="mergesort",
        )
        return candidate_df.index.astype(str).tolist()


def run_hpi_variant(
    *,
    strategy_name_str: str,
    indexname_str: str,
    benchmark_symbol_str: str,
    ranking_field_str: str,
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str = "2004-01-01",
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
    entry_mode_str: str = ENTRY_BASELINE_STR,
    liquidity_mode_str: str = LIQUIDITY_NONE_STR,
) -> HPIStatefulLongStrategy:
    _, universe_df, pricing_data_df = load_exact_hpi_inputs(
        indexname_str=indexname_str,
        benchmark_symbol_str=benchmark_symbol_str,
        start_date_str="1998-01-01",
        end_date_str=end_date_str,
    )
    pricing_symbol_list = (
        pricing_data_df.columns.get_level_values(0).unique().astype(str)
    )
    pricing_data_df.attrs["norgate_adjustment_by_symbol_dict"] = {
        symbol_str: (
            "TOTALRETURN"
            if symbol_str == benchmark_symbol_str
            else "CAPITALSPECIAL"
        )
        for symbol_str in pricing_symbol_list
    }
    strategy_obj = HPIStatefulLongStrategy(
        name=strategy_name_str,
        benchmarks=[benchmark_symbol_str],
        ranking_field_str=ranking_field_str,
        capital_base=capital_base_float,
        entry_mode_str=entry_mode_str,
        liquidity_mode_str=liquidity_mode_str,
        backtest_start_date_str=backtest_start_date_str,
    )
    strategy_obj.universe_df = universe_df

    # *** CRITICAL*** Pre-start history is retained for the 1,260-observation
    # HPI warm-up. Trading begins only on the requested execution calendar.
    calendar_idx = pricing_data_df.index[
        pricing_data_df.index >= pd.Timestamp(backtest_start_date_str)
    ]
    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar_idx,
        show_progress=show_display_bool,
        show_signal_progress_bool=show_display_bool,
    )
    strategy_obj.universe_df = None

    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1_000)
        display(strategy_obj.summary)
        display(strategy_obj.summary_trades)
    if save_results_bool:
        save_results(strategy_obj, output_dir=output_dir_str)
    return strategy_obj


def build_hpi_execution_timing_analysis_inputs(
    *,
    strategy_name_str: str,
    entry_mode_str: str,
) -> dict[str, object]:
    benchmark_symbol_str = "$SPXTR"
    _, universe_df, pricing_data_df = load_exact_hpi_inputs(
        indexname_str="S&P 500",
        benchmark_symbol_str=benchmark_symbol_str,
        start_date_str="1998-01-01",
        end_date_str=None,
    )
    calendar_idx = pricing_data_df.index[
        pricing_data_df.index >= pd.Timestamp("2004-01-01")
    ]

    def strategy_factory_fn() -> HPIStatefulLongStrategy:
        strategy_obj = HPIStatefulLongStrategy(
            name=strategy_name_str,
            benchmarks=[benchmark_symbol_str],
            ranking_field_str=TURNOVER_FIELD_STR,
            entry_mode_str=entry_mode_str,
            backtest_start_date_str="2004-01-01",
        )
        strategy_obj.universe_df = universe_df
        return strategy_obj

    return {
        "strategy_factory_fn": strategy_factory_fn,
        "pricing_data_df": pricing_data_df,
        "calendar_idx": pd.DatetimeIndex(calendar_idx),
        "order_generation_mode_str": "signal_bar",
        "risk_model_str": "daily_ohlc_signal",
        "entry_timing_str_tuple": (
            "same_close_moc",
            "next_open",
            "next_close",
        ),
        "exit_timing_str_tuple": (
            "same_close_moc",
            "next_open",
            "next_close",
        ),
        "default_entry_timing_str": "next_open",
        "default_exit_timing_str": "next_open",
    }


def build_hpi_capacity_analysis_inputs(
    *,
    strategy_name_str: str,
    entry_mode_str: str,
    show_display_bool: bool = False,
    backtest_start_date_str: str = "2004-01-01",
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
) -> dict[str, object]:
    benchmark_symbol_str = "$SPXTR"
    _, universe_df, pricing_data_df = load_exact_hpi_inputs(
        indexname_str="S&P 500",
        benchmark_symbol_str=benchmark_symbol_str,
        start_date_str="1998-01-01",
        end_date_str=end_date_str,
    )
    strategy_obj = HPIStatefulLongStrategy(
        name=strategy_name_str,
        benchmarks=[benchmark_symbol_str],
        ranking_field_str=TURNOVER_FIELD_STR,
        capital_base=capital_base_float,
        entry_mode_str=entry_mode_str,
        backtest_start_date_str=backtest_start_date_str,
    )
    strategy_obj.universe_df = universe_df

    # *** CRITICAL*** Retain pre-start history for the prior-only 1,260-row
    # HPI reference set, but execute only on the requested calendar.
    calendar_idx = pricing_data_df.index[
        pricing_data_df.index >= pd.Timestamp(backtest_start_date_str)
    ]
    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar_idx,
        show_progress=show_display_bool,
        show_signal_progress_bool=show_display_bool,
    )
    strategy_obj.universe_df = None
    strategy_obj._performance_benchmark_symbol_str = benchmark_symbol_str
    strategy_obj._performance_benchmark_adjustment_str = (
        TOTALRETURN_ADJUSTMENT_STR
    )
    return {
        "strategy_obj": strategy_obj,
        "pricing_data_df": pricing_data_df,
        "execution_policy_str": "MOO",
        "impact_profile_str": "MOO_LARGE_MIXED",
    }
