"""
Beyond 60/40 inverse-volatility strategy with a monthly 10-month SMA trend tilt
("T1" in the 2026-11-05 research).

Status
------
WIRED candidate, route being built; see docs/live/BEYOND6040_T1_WIRING_20261105.md
Source: docs/research/BEYOND_6040_20261105.md and
scripts/research/beyond_6040_20261105/ (SPEC.md, out/a4, out/gates, out/vault).

Rule T1
-------
Base weights (unchanged from the inverse-volatility strategy), set on the last
session of each month from trailing 63-day realised volatility:

    w_{i,t}^{base}
        = (1 / sigma_{i,t}^{(63)}) / sum_j(1 / sigma_{j,t}^{(63)})

Trend tilt, set on the same month-end close from the Total-Return Close:

    s_{i,t}
        = +1    if TRClose_{i,t} >  SMA_{210}(TRClose_i)_t      (10-month SMA)
        = -1    otherwise

    m_{i,t}
        = (1 - k) + k * s_{i,t},   k = 0.5   ->   m in {0, 1}

    w_{i,t}^{tilt}
        = w_{i,t}^{base} * m_{i,t}           (sum <= 1, remainder is cash)

Daily volatility overlay, measured on the risky sleeve held at the BASE weights
(not the tilted weights), as in the author's rule:

    r_{sleeve,t}
        = sum_i w_{i}^{base,effective} * r_{i,t}

    sigma_{sleeve,t}^{(63)}
        = sqrt(252) * std(r_{sleeve,t-62:t})

    exposure_t
        = 1                                       if sigma_{sleeve,t} <= 0.085
        = min(1, 0.08 / sigma_{sleeve,t})         otherwise

Final weights:

    w_{i,t}^{final}
        = exposure_t * w_{i,t}^{base} * m_{i,t}

    w_t^{residual}
        = 1 - sum_i w_{i,t}^{final}

Reserve (added for live, owner decision 2026-11-05): the residual is parked in
BIL (``reserve_asset_str = "BIL"``), as CORE5 does; the research series
credited Fed funds on idle cash and BIL is its tradable form. Before BIL's
first observed session (2007-05-30) the residual stays in engine cash. BIL is
a tradable only: it enters neither the inverse-volatility weights, nor the
sleeve volatility overlay, nor the SMA tilt.

    w_t^{BIL}  = w_t^{residual}   if Close_T(BIL) is observed, else 0
    w_t^{cash} = w_t^{residual} - w_t^{BIL}

No leverage, no shorts. Signals are decided after Close_T and fill at
Open_(T+1) through the engine's previous-bar contract, exactly as the other
Beyond 60/40 variants. Dividends are credited as cash by the engine ledger.

Data: VTI, TLT, GLD and BIL CAPITALSPECIAL (traded prices), VTI, TLT and GLD
TOTALRETURN (trend signal) and the $SPX benchmark (read from $SPXTR), from
VTI inception 2001-05-31. In snapshot mode the same frame is read from the
``norgate_eod_beyond6040`` profile (data/beyond6040_snapshot_contract.py).

Pre-registered caveats (keep with the strategy)
-----------------------------------------------
1. The tilt is insurance: it costs about 0.5%/yr of CAGR in bull markets and
   pays in regimes of positive stock-bond correlation with falling stocks.
2. Sharpe over cash is 0.63 (1970-2022). The author's published Sharpe ignores
   cash; the raw figure is not comparable with our excess-over-cash figures.
3. Same engine as CORE5 (defensive multi-asset allocator). It has no book slot
   while CORE5 holds the defensive seat; one pod per engine.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from data.beyond6040_snapshot_contract import BEYOND6040_HISTORY_START_STR, BEYOND6040_PROFILE_STR
from data.norgate_loader import TOTALRETURN_ADJUSTMENT_STR, load_price_timeseries, load_raw_prices
from data.norgate_snapshot_store import (
    NorgateSnapshotValidationError,
    is_snapshot_mode_enabled_bool,
    load_valid_snapshot_manifest,
    use_norgate_data_profile,
)
from strategies.taa_beyond_6040.strategy_taa_beyond_6040 import (
    Beyond6040Config,
    Beyond6040Strategy,
    _flatten_close_df,
    build_signal_base_weight_df,
    compute_gross_exposure_float,
    compute_month_end_inverse_vol_weight_df,
    map_month_end_weights_to_rebalance_open_df,
)


STRATEGY_NAME_STR = "strategy_taa_beyond_6040_trend_tilt"
TARGET_PORTFOLIO_VOL_FLOAT = 0.08
TRIGGER_PORTFOLIO_VOL_FLOAT = 0.085
TREND_SMA_LOOKBACK_INT = 210
TILT_STRENGTH_FLOAT = 0.5
TREND_PRICE_FIELD_STR = "TotalReturnClose"
RESERVE_ASSET_STR = "BIL"
SLEEVE_LABEL_STR = "PortfolioSleeve"
SLEEVE_RETURN_FIELD_STR = "base_weight_return_ser"


@dataclass(frozen=True)
class Beyond6040TrendTiltConfig(Beyond6040Config):
    trend_sma_lookback_int: int = TREND_SMA_LOOKBACK_INT
    tilt_strength_float: float = TILT_STRENGTH_FLOAT
    trend_price_field_str: str = TREND_PRICE_FIELD_STR
    # Live addition (owner decision 2026-11-05): the residual is parked in BIL;
    # None keeps the research convention of idle engine cash.
    reserve_asset_str: str | None = RESERVE_ASSET_STR
    # The data profile's window: VTI inception. Nothing before it enters any
    # signal (VTI has no earlier bar and every rule needs all three assets).
    start_date_str: str = BEYOND6040_HISTORY_START_STR

    def __post_init__(self):
        super().__post_init__()
        if self.trend_sma_lookback_int < 2:
            raise ValueError("trend_sma_lookback_int must be at least 2.")
        if not (0.0 <= self.tilt_strength_float <= 1.0):
            raise ValueError("tilt_strength_float must lie in [0, 1].")
        if self.reserve_asset_str is not None and (
            self.reserve_asset_str in self.asset_list or self.reserve_asset_str in self.benchmark_list
        ):
            raise ValueError("reserve_asset_str must not be a risk asset or a benchmark.")

    @property
    def tradable_list(self) -> list[str]:
        return list(self.asset_list) + ([] if self.reserve_asset_str is None else [self.reserve_asset_str])


DEFAULT_CONFIG = Beyond6040TrendTiltConfig(
    target_portfolio_vol_float=TARGET_PORTFOLIO_VOL_FLOAT,
    trigger_portfolio_vol_float=TRIGGER_PORTFOLIO_VOL_FLOAT,
)


def _load_trend_tilt_frame_df(config: Beyond6040TrendTiltConfig) -> pd.DataFrame:
    """The research frame: CAPITALSPECIAL tradables + benchmark, then one (asset, "TotalReturnClose") column per risk asset."""
    pricing_data_df = load_raw_prices(
        symbols=config.tradable_list,
        benchmarks=list(config.benchmark_list),
        start_date=config.start_date_str,
        end_date=config.end_date_str,
    )
    for asset_str in config.asset_list:
        total_return_df = load_price_timeseries(
            asset_str,
            adjustment_str=TOTALRETURN_ADJUSTMENT_STR,
            start_date_str=config.start_date_str,
            end_date_str=config.end_date_str,
        )
        # *** CRITICAL*** The total-return series is aligned on the base index
        # without any fill: a missing session stays NaN and the signal stays NaN.
        pricing_data_df[(asset_str, config.trend_price_field_str)] = (
            total_return_df["Close"].astype(float).reindex(pricing_data_df.index)
        )
    return pricing_data_df


def get_trend_tilt_data(config: Beyond6040TrendTiltConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """
    Load the CAPITALSPECIAL tradables (risk assets and the BIL reserve), the
    benchmark, and one Total-Return Close column per risk asset,
    (asset, "TotalReturnClose"), for the trend signal.

    Direct mode reads Norgate. Snapshot mode reads the same series from the
    qualified ``norgate_eod_beyond6040`` profile (observed rows, native fields
    and dtypes), so the live frame equals the research frame.
    """
    if not is_snapshot_mode_enabled_bool():
        return _load_trend_tilt_frame_df(config)
    with use_norgate_data_profile(BEYOND6040_PROFILE_STR):
        initial_manifest_obj = load_valid_snapshot_manifest(BEYOND6040_PROFILE_STR)
        pricing_data_df = _load_trend_tilt_frame_df(config)
        final_manifest_obj = load_valid_snapshot_manifest(BEYOND6040_PROFILE_STR)
    # *** CRITICAL*** A publication between the reads must not mix two data vintages in one decision.
    if (
        initial_manifest_obj.snapshot_dir_path_obj != final_manifest_obj.snapshot_dir_path_obj
        or initial_manifest_obj.manifest_hash_str != final_manifest_obj.manifest_hash_str
    ):
        raise NorgateSnapshotValidationError("Beyond 60/40 snapshot changed during the data load; retry with one complete snapshot.")
    return pricing_data_df


def compute_month_end_tilt_multiplier_df(
    trend_price_df: pd.DataFrame,
    trend_sma_lookback_int: int = TREND_SMA_LOOKBACK_INT,
    tilt_strength_float: float = TILT_STRENGTH_FLOAT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Daily trend SMA, daily trend state s in {+1, -1}, and month-end tilt
    multipliers m = (1 - k) + k * s sampled at the calendar month-end label.
    """
    trend_price_df = _flatten_close_df(trend_price_df)

    # *** CRITICAL*** Backward-only rolling mean; the state at close t uses
    # closes through t only.
    trend_sma_df = trend_price_df.rolling(trend_sma_lookback_int).mean()
    trend_state_df = pd.DataFrame(
        np.where(trend_price_df > trend_sma_df, 1.0, -1.0),
        index=trend_price_df.index,
        columns=trend_price_df.columns,
    ).where(trend_sma_df.notna() & trend_price_df.notna())

    month_end_state_df = trend_state_df.resample("ME").last()
    month_end_multiplier_df = (1.0 - tilt_strength_float) + tilt_strength_float * month_end_state_df
    month_end_multiplier_df = month_end_multiplier_df.dropna(how="any")
    return trend_sma_df, trend_state_df, month_end_multiplier_df


def get_first_actionable_tilted_rebalance_ts(
    pricing_data_df: pd.DataFrame,
    config: Beyond6040TrendTiltConfig = DEFAULT_CONFIG,
) -> pd.Timestamp:
    """First rebalance open with valid base weights AND a valid tilt for every asset."""
    asset_close_key_list = [(asset_str, "Close") for asset_str in config.asset_list]
    trend_key_list = [(asset_str, config.trend_price_field_str) for asset_str in config.asset_list]
    missing_key_list = [
        key for key in asset_close_key_list + trend_key_list if key not in pricing_data_df.columns
    ]
    if missing_key_list:
        raise RuntimeError(f"Missing data for {missing_key_list}.")

    _, _, month_end_weight_df = compute_month_end_inverse_vol_weight_df(
        price_close_df=pricing_data_df.loc[:, asset_close_key_list].astype(float),
        asset_vol_lookback_int=config.asset_vol_lookback_int,
    )
    _, _, month_end_multiplier_df = compute_month_end_tilt_multiplier_df(
        trend_price_df=pricing_data_df.loc[:, trend_key_list].astype(float),
        trend_sma_lookback_int=config.trend_sma_lookback_int,
        tilt_strength_float=config.tilt_strength_float,
    )
    common_index = month_end_weight_df.index.intersection(month_end_multiplier_df.index)
    rebalance_weight_df = map_month_end_weights_to_rebalance_open_df(
        month_end_weight_df=month_end_weight_df.loc[common_index],
        execution_index=pricing_data_df.index,
    )
    if len(rebalance_weight_df) == 0:
        raise RuntimeError("No actionable rebalance date was generated for the trend-tilt variant.")
    return pd.Timestamp(rebalance_weight_df.index[0])


class Beyond6040TrendTiltStrategy(Beyond6040Strategy):
    """
    Beyond 60/40 with a monthly 10-month SMA trend tilt per asset and a daily
    volatility overlay measured on the base-weight sleeve. See module docstring.
    """

    def __init__(
        self,
        name: str,
        trend_sma_lookback_int: int = DEFAULT_CONFIG.trend_sma_lookback_int,
        tilt_strength_float: float = DEFAULT_CONFIG.tilt_strength_float,
        trend_price_field_str: str = DEFAULT_CONFIG.trend_price_field_str,
        reserve_asset_str: str | None = DEFAULT_CONFIG.reserve_asset_str,
        **base_kwargs,
    ):
        super().__init__(name=name, **base_kwargs)
        self.trend_sma_lookback_int = int(trend_sma_lookback_int)
        self.tilt_strength_float = float(tilt_strength_float)
        self.trend_price_field_str = str(trend_price_field_str)
        self.reserve_asset_str = None if reserve_asset_str is None else str(reserve_asset_str)
        if self.reserve_asset_str is not None and self.reserve_asset_str in self.asset_list:
            raise ValueError("reserve_asset_str must not be a risk asset.")
        # BIL is a tradable only; asset_list (the risk assets) drives every signal.
        self.tradable_list = list(self.asset_list) + ([] if self.reserve_asset_str is None else [self.reserve_asset_str])
        self.weight_column_list = self.tradable_list + ["Cash"]
        self.daily_target_weights = pd.DataFrame(columns=self.weight_column_list, dtype=float)
        self.daily_exposure_map: dict[pd.Timestamp, float] = {}
        self.daily_exposure_ser = pd.Series(dtype=float)
        self.month_end_tilt_multiplier_df = pd.DataFrame(columns=self.asset_list, dtype=float)
        self.rebalance_tilted_weight_df = pd.DataFrame(columns=self.asset_list, dtype=float)

    # ------------------------------------------------------------------ signals
    def _trend_key_list(self) -> list[tuple[str, str]]:
        return [(asset_str, self.trend_price_field_str) for asset_str in self.asset_list]

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        signal_data_df = super().compute_signals(pricing_data_df)

        trend_key_list = self._trend_key_list()
        missing_key_list = [key for key in trend_key_list if key not in signal_data_df.columns]
        if missing_key_list:
            raise RuntimeError(f"Missing total-return close data for {missing_key_list}.")

        trend_sma_df, trend_state_df, month_end_multiplier_df = compute_month_end_tilt_multiplier_df(
            trend_price_df=signal_data_df.loc[:, trend_key_list].astype(float),
            trend_sma_lookback_int=self.trend_sma_lookback_int,
            tilt_strength_float=self.tilt_strength_float,
        )
        # *** CRITICAL*** Same month-end -> previous-bar schedule as the base
        # weights, so base weight and tilt are read together from close_row_ser.
        signal_multiplier_df = build_signal_base_weight_df(
            month_end_weight_df=month_end_multiplier_df,
            execution_index=signal_data_df.index,
        )

        # Risky sleeve at the BASE weights in force during each bar: the weights
        # read at close t-1 are the ones held through bar t.
        base_weight_df = pd.DataFrame(
            {asset_str: signal_data_df[(asset_str, "base_weight_ser")] for asset_str in self.asset_list}
        ).astype(float)
        asset_return_df = pd.DataFrame(
            {asset_str: signal_data_df[(asset_str, "return_ser")] for asset_str in self.asset_list}
        ).astype(float)
        sleeve_return_ser = (base_weight_df.shift(1) * asset_return_df).sum(axis=1, min_count=len(self.asset_list))

        feature_df = pd.DataFrame(index=signal_data_df.index)
        for asset_str in self.asset_list:
            feature_df[(asset_str, "trend_sma_ser")] = trend_sma_df[asset_str]
            feature_df[(asset_str, "trend_state_ser")] = trend_state_df[asset_str]
            feature_df[(asset_str, "tilt_multiplier_ser")] = signal_multiplier_df[asset_str]
        feature_df[(SLEEVE_LABEL_STR, SLEEVE_RETURN_FIELD_STR)] = sleeve_return_ser
        feature_df.columns = pd.MultiIndex.from_tuples(feature_df.columns)
        return pd.concat([signal_data_df, feature_df], axis=1)

    def signal_audit_fields(self, pricing_data: pd.DataFrame, signal_data: pd.DataFrame):
        # The month-end schedule columns need the next month's first session to
        # exist, so they cannot be reproduced on a truncated history; the base
        # class excludes base_weight_ser for the same reason.
        audit_col_list = super().signal_audit_fields(pricing_data, signal_data)
        return [
            col
            for col in audit_col_list
            if not (isinstance(col, tuple) and len(col) >= 2 and col[1] == "tilt_multiplier_ser")
        ]

    # ------------------------------------------------------------------ trading
    def _current_tilt_multiplier_ser(self, close_row_ser: pd.Series) -> pd.Series:
        return pd.Series(
            {
                asset_str: float(close_row_ser.get((asset_str, "tilt_multiplier_ser"), np.nan))
                for asset_str in self.asset_list
            },
            dtype=float,
        )

    def _sleeve_gross_exposure_float(self, data_df: pd.DataFrame) -> float:
        sleeve_key = (SLEEVE_LABEL_STR, SLEEVE_RETURN_FIELD_STR)
        if sleeve_key not in data_df.columns:
            return 1.0
        sleeve_return_ser = data_df[sleeve_key].astype(float).dropna()
        return compute_gross_exposure_float(
            realized_return_ser=sleeve_return_ser.iloc[-self.portfolio_vol_lookback_int :],
            portfolio_vol_lookback_int=self.portfolio_vol_lookback_int,
            target_portfolio_vol_float=self.target_portfolio_vol_float,
            trigger_portfolio_vol_float=self.trigger_portfolio_vol_float,
        )

    def _reserve_tradable_bool(self, close_row_ser: pd.Series) -> bool:
        """BIL takes the residual only once it has an observed Close_T; before its inception the residual is cash."""
        if self.reserve_asset_str is None:
            return False
        reserve_close_float = float(close_row_ser.get((self.reserve_asset_str, "Close"), np.nan))
        return bool(np.isfinite(reserve_close_float) and reserve_close_float > 0.0)

    def _record_daily_target_weight_ser(self, target_weight_ser: pd.Series):
        ordered_weight_ser = target_weight_ser.reindex(self.weight_column_list).astype(float)
        self.daily_target_weight_map[pd.Timestamp(self.current_bar)] = ordered_weight_ser
        self.daily_target_weights.loc[pd.Timestamp(self.current_bar), ordered_weight_ser.index] = ordered_weight_ser

    def iterate(self, data_df: pd.DataFrame, close_row_ser: pd.Series, open_price_ser: pd.Series):
        if close_row_ser is None or data_df is None:
            return

        base_weight_ser = self._current_base_weight_ser(close_row_ser).astype(float)
        tilt_multiplier_ser = self._current_tilt_multiplier_ser(close_row_ser).astype(float)
        if base_weight_ser.isna().any() or tilt_multiplier_ser.isna().any():
            cash_only_weight_ser = pd.Series(
                [0.0] * len(self.tradable_list) + [1.0],
                index=self.weight_column_list,
                dtype=float,
            )
            self.daily_exposure_map[pd.Timestamp(self.current_bar)] = np.nan
            self._record_daily_target_weight_ser(cash_only_weight_ser)
            return

        gross_exposure_float = self._sleeve_gross_exposure_float(data_df)
        self.daily_exposure_map[pd.Timestamp(self.current_bar)] = gross_exposure_float

        target_weight_ser = (base_weight_ser * tilt_multiplier_ser * gross_exposure_float).astype(float)
        if (target_weight_ser < -1e-12).any() or float(target_weight_ser.sum()) > 1.0 + 1e-12:
            raise ValueError(f"Tilted weights must lie in [0, 1] and sum to at most 1 on {self.current_bar}.")
        residual_weight_float = float(1.0 - target_weight_ser.sum())
        reserve_tradable_bool = self._reserve_tradable_bool(close_row_ser)
        reserve_weight_ser = pd.Series(dtype=float)
        if self.reserve_asset_str is not None:
            # *** CRITICAL*** Residual -> BIL once it has a Close_T; it never enters the vol weights or the tilt.
            reserve_weight_ser = pd.Series(
                {self.reserve_asset_str: residual_weight_float if reserve_tradable_bool else 0.0}, dtype=float,
            )
        cash_weight_float = float(residual_weight_float - reserve_weight_ser.sum())
        target_weight_ser = pd.concat(
            [target_weight_ser, reserve_weight_ser, pd.Series({"Cash": cash_weight_float}, dtype=float)]
        )
        self._record_daily_target_weight_ser(target_weight_ser)

        current_position_ser = self.get_positions().reindex(self.tradable_list, fill_value=0.0).astype(int)
        budget_value_float = float(self.previous_total_value)

        for asset_str in self.tradable_list:
            target_weight_float = float(target_weight_ser.loc[asset_str])
            current_share_int = int(current_position_ser.loc[asset_str])
            if asset_str == self.reserve_asset_str and not reserve_tradable_bool:
                if current_share_int != 0:
                    raise RuntimeError(f"Reserve {asset_str} is held without an observed close on {self.current_bar}.")
                continue
            open_price_float = float(open_price_ser.get(asset_str, np.nan))
            if not np.isfinite(open_price_float) or open_price_float <= 0.0:
                raise RuntimeError(f"Invalid open price for target asset {asset_str} on {self.current_bar}.")

            # *** CRITICAL*** The trade/no-trade test uses Close_T, the same price
            # that sizes the order (order_target_percent sizes from the previous
            # close). Open_(T+1) is unknown at decision time, so a skip test on it
            # cannot be reproduced live (wiring review 2026-11-05). Fills stay at
            # the current open; same-bar close data is forbidden.
            sizing_close_float = float(close_row_ser.get((asset_str, "Close"), np.nan))
            if not np.isfinite(sizing_close_float) or sizing_close_float <= 0.0:
                raise RuntimeError(f"Invalid Close_T for target asset {asset_str} on {self.current_bar}.")
            target_share_int = int(np.floor(budget_value_float * target_weight_float / sizing_close_float))
            if target_share_int == current_share_int:
                continue

            if target_share_int <= 0:
                if current_share_int <= 0:
                    continue
                self.order_target_value(asset_str, 0.0, trade_id=self.current_trade_id_map[asset_str])
                self.current_trade_id_map[asset_str] = -1
                continue

            if current_share_int <= 0 or self.current_trade_id_map[asset_str] == -1:
                self.trade_id_int += 1
                self.current_trade_id_map[asset_str] = self.trade_id_int

            self.order_target_percent(asset_str, target_weight_float, trade_id=self.current_trade_id_map[asset_str])

    def finalize(self, current_data: pd.DataFrame):
        super().finalize(current_data)
        if len(self.daily_target_weight_map) > 0:
            # The base class keeps only the risk assets and Cash; keep the reserve column too.
            self.daily_target_weights = pd.DataFrame.from_dict(self.daily_target_weight_map, orient="index").sort_index()
            self.daily_target_weights.index = pd.to_datetime(self.daily_target_weights.index)
            self.daily_target_weights = self.daily_target_weights.reindex(columns=self.weight_column_list)
        trend_key_list = self._trend_key_list()
        if all(key in current_data.columns for key in trend_key_list):
            _, _, month_end_multiplier_df = compute_month_end_tilt_multiplier_df(
                trend_price_df=current_data.loc[:, trend_key_list].astype(float),
                trend_sma_lookback_int=self.trend_sma_lookback_int,
                tilt_strength_float=self.tilt_strength_float,
            )
            self.month_end_tilt_multiplier_df = month_end_multiplier_df
            common_index = self.month_end_weight_df.index.intersection(month_end_multiplier_df.index)
            self.rebalance_tilted_weight_df = map_month_end_weights_to_rebalance_open_df(
                month_end_weight_df=(
                    self.month_end_weight_df.loc[common_index] * month_end_multiplier_df.loc[common_index]
                ),
                execution_index=current_data.index,
            )
        if len(self.daily_exposure_map) > 0:
            self.daily_exposure_ser = pd.Series(self.daily_exposure_map, dtype=float).sort_index()
            self.daily_exposure_ser.index = pd.to_datetime(self.daily_exposure_ser.index)


def _build_strategy_obj(config: Beyond6040TrendTiltConfig) -> Beyond6040TrendTiltStrategy:
    return Beyond6040TrendTiltStrategy(
        name=STRATEGY_NAME_STR,
        trend_sma_lookback_int=config.trend_sma_lookback_int,
        tilt_strength_float=config.tilt_strength_float,
        trend_price_field_str=config.trend_price_field_str,
        reserve_asset_str=config.reserve_asset_str,
        benchmarks=config.benchmark_list,
        asset_list=config.asset_list,
        asset_vol_lookback_int=config.asset_vol_lookback_int,
        portfolio_vol_lookback_int=config.portfolio_vol_lookback_int,
        target_portfolio_vol_float=config.target_portfolio_vol_float,
        trigger_portfolio_vol_float=config.trigger_portfolio_vol_float,
        capital_base=config.capital_base_float,
        slippage=config.slippage_float,
        commission_per_share=config.commission_per_share_float,
        commission_minimum=config.commission_minimum_float,
    )


def _build_analysis_context_dict(
    config: Beyond6040TrendTiltConfig = DEFAULT_CONFIG,
    pricing_data_df: pd.DataFrame | None = None,
    backtest_start_date_str: str | None = None,
) -> dict[str, object]:
    """One causal frame and fill calendar shared by Vanilla and the BENCH hooks."""
    if pricing_data_df is None:
        pricing_data_df = get_trend_tilt_data(config=config)
    calendar_start_ts = get_first_actionable_tilted_rebalance_ts(pricing_data_df=pricing_data_df, config=config)
    if backtest_start_date_str is not None:
        calendar_start_ts = max(calendar_start_ts, pd.Timestamp(backtest_start_date_str))
    # *** CRITICAL*** Keep the full pre-start price history for the volatility,
    # trend and overlay signals; only the executable fill calendar is clipped.
    calendar_idx = pricing_data_df.index[pricing_data_df.index >= calendar_start_ts]
    return {"config_obj": config, "pricing_data_df": pricing_data_df, "calendar_idx": calendar_idx}


def build_execution_timing_analysis_inputs() -> dict[str, object]:
    """BENCH Timing inputs with the Vanilla frame, calendar and factory."""
    context_dict = _build_analysis_context_dict()

    def strategy_factory_fn() -> Beyond6040TrendTiltStrategy:
        return _build_strategy_obj(context_dict["config_obj"])

    return {
        "strategy_factory_fn": strategy_factory_fn,
        "pricing_data_df": context_dict["pricing_data_df"],
        "calendar_idx": context_dict["calendar_idx"],
        "order_generation_mode_str": "vanilla_current_bar",
        "risk_model_str": "taa_rebalance",
        "entry_timing_str_tuple": ("same_open", "same_close_moc", "next_open", "next_close"),
        "exit_timing_str_tuple": ("same_open", "same_close_moc", "next_open", "next_close"),
        "default_entry_timing_str": "same_open",
        "default_exit_timing_str": "same_open",
    }


def build_stress_test_context_dict() -> dict[str, object]:
    return _build_analysis_context_dict()


def build_stress_test_strategy_obj(context_dict: dict[str, object]) -> Beyond6040TrendTiltStrategy:
    return _build_strategy_obj(context_dict["config_obj"])


def build_capacity_analysis_inputs(
    show_display_bool: bool = False,
    backtest_start_date_str: str | None = None,
    capital_base_float: float = DEFAULT_CONFIG.capital_base_float,
    end_date_str: str | None = None,
) -> dict[str, object]:
    """One full-engine BENCH Capacity point with default trading costs."""
    strategy_obj = run_variant(
        show_display_bool=show_display_bool,
        save_results_bool=False,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=float(capital_base_float),
        end_date_str=end_date_str,
    )
    pricing_data_df = get_trend_tilt_data(replace(DEFAULT_CONFIG, end_date_str=end_date_str))
    return {
        "strategy_obj": strategy_obj,
        "pricing_data_df": pricing_data_df,
        "execution_policy_str": "MOO",
        "impact_profile_str": "MOO_ETF_PROXY",
    }


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float = DEFAULT_CONFIG.capital_base_float,
    end_date_str: str | None = None,
) -> Beyond6040TrendTiltStrategy:
    config = replace(
        DEFAULT_CONFIG,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
    )
    context_dict = _build_analysis_context_dict(config, backtest_start_date_str=backtest_start_date_str)
    strategy_obj = _build_strategy_obj(config)

    run_daily(
        strategy_obj,
        context_dict["pricing_data_df"],
        calendar=context_dict["calendar_idx"],
        show_progress=show_display_bool,
        show_signal_progress_bool=show_display_bool,
        audit_override_bool=None,
    )

    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        display(strategy_obj.summary)
        display(strategy_obj.summary_trades)

    if save_results_bool:
        save_results(strategy_obj, output_dir=output_dir_str)

    return strategy_obj


if __name__ == "__main__":
    run_variant()
