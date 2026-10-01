"""Scout spec of the sector-dispersion IBS pods (PM_READY; `strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_*`):
the KIE+IHI+XLC basket, the same basket with an asset SMA200 entry gate, and the KIE+IHI basket with that gate.

An independent re-implementation of the signal; execution is the shared weights engine with the event hook of
`sector_ibs.py` (fractional entries, held ETFs untouched). Mapped on 2026-10-02 against main b3855fc:
strategy_mr_sector_dispersion_ibs.py (lines "S:"), strategy_mr_sector_dispersion_ibs_kie_ihi_xlc_asset_sma200.py
(lines "A:"; the KIE+IHI SMA200 module reuses its signal function) and alpha/engine.

Data (S: get_sector_dispersion_ibs_data :574-596 -> load_raw_prices): CAPITALSPECIAL bars of the basket plus the $SPX
benchmark (its TR data symbol, which only widens the date index), from 2003-01-01; configured start 2004-01-01.

Signal at the close of T, per ETF (S: compute_sector_dispersion_ibs_signal_df :414-471)
    IBS_T            (Close_T - Low_T) / (High_T - Low_T); a zero range is NaN
    Range_T          ln(High_T / Low_T) where High > 0, Low > 0, High > Low, else NaN
    RelativeRange_T  Range_T / std(Range_(T-21) .. Range_(T-1)) (pandas rolling 21, min_periods 21, ddof 1, shifted one
                     session; a zero std is NaN)
    entry_T          IBS_T < 0.10 and RelativeRange_T > 1.0
    exit_T           IBS_T > 0.90 and RelativeRange_T > 1.0
    SMA200 variants  entry_T also needs Close_T > SMA200_T (pandas rolling 200 mean, min_periods 200) (A: :125-178);
                     the gate never forces an exit

Decision after the close of T (S: SectorDispersionIbsStrategy.iterate :518-560)
    exits            held ETFs (shares > 0) with exit_T -> 0 shares
    entries          every ETF with zero shares at the close of T and entry_T, in basket order (no slot cap: each ETF has
                     its own 1/N sleeve; an ETF exiting today cannot re-enter today)
    sizing           shares = V_T x (1.0 / N) / Close_T, fractional; held ETFs are never resized
    costs            the pod's own: 2.5 bp slippage, $0.00525 a share, no minimum (S: SectorDispersionIbsConfig :149-151);
                     dividends net of 25% withholding before the next open; no cash check
Calendar (S: resolve_full_basket_calendar_idx :252-388): the first session T on or after 2004-01-01 where every ETF has
    22 consecutive valid-OHLC rows with High > Low ending at T (SMA200 variants: also 200 consecutive valid closes) is
    the engine's FIRST session: its iterate reads T-1, whose signals are still NaN, so nothing trades on it.

*** CRITICAL*** Every feature at T uses bars up to T only (the range std is shifted); fills at Open(T+1).

Family parameters (`DispersionIbsConfig`; the default of each variant is its engine configuration):
    entry_ibs_max_float, exit_ibs_min_float, min_relative_range_float, range_vol_lookback_int, asset_sma_int (0 = no
    gate), portfolio_leverage_float; decision_offset_int must be 0 (no rebalance schedule, no luck band).

MCPT: the matrix and the replica's ledger are `sector_ibs`'s (TR returns, then four log-bar blocks per ETF); the
replica recomputes IBS, RelativeRange and the SMA from the rebuilt bars.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha.scout.engines.weights import CostModel, WeightsResult
from alpha.scout.specs.sector_ibs import (
    S3_HORIZON_TUPLE,
    SEAL_END_STR,
    SectorEtfInputs,
    _event_book_daily,
    bar_matrix,
    execution_start_ts,
    ibs_df,
    load_etf_inputs,
    log_range_df,
    median_holding_sessions_int,
    rebuilt_ohlc_dict,
    s3_panel,
    simulate_event_rule,
)

HISTORY_START_STR = "2003-01-01"
BACKTEST_START_STR = "2004-01-01"
ENGINE_COST_MODEL = CostModel(slippage_float=0.00025, fee_per_share_float=0.00525, min_fee_float=0.0, dividend_withholding_float=0.25)
KIE_IHI_TUPLE = ("SOXX", "IGV", "IBB", "KIE", "IHI")
KIE_IHI_XLC_TUPLE = KIE_IHI_TUPLE + ("XLC",)


@dataclass(frozen=True)
class DispersionIbsConfig:
    entry_ibs_max_float: float = 0.10
    exit_ibs_min_float: float = 0.90
    min_relative_range_float: float = 1.0
    range_vol_lookback_int: int = 21
    asset_sma_int: int = 0  # 0 = no asset trend gate; 200 in the SMA200 variants
    portfolio_leverage_float: float = 1.0
    decision_offset_int: int = 0

    def __post_init__(self):
        if not 0.0 <= self.entry_ibs_max_float < self.exit_ibs_min_float <= 1.0 or self.min_relative_range_float <= 0.0:
            raise ValueError("DispersionIbsConfig: 0 <= entry IBS < exit IBS <= 1 and a positive relative-range threshold.")
        if self.range_vol_lookback_int < 2 or self.asset_sma_int < 0 or self.portfolio_leverage_float <= 0.0:
            raise ValueError("DispersionIbsConfig: range lookback >= 2, asset_sma_int >= 0, positive leverage.")
        if self.decision_offset_int != 0:
            raise ValueError("DispersionIbsConfig: a daily event rule has no rebalance offset (decision_offset_int = 0).")

    @property
    def required_history_int(self) -> int:
        return self.range_vol_lookback_int + 1  # S: :259-262


@dataclass(frozen=True)
class DispersionVariant:
    strategy_import_str: str
    symbol_tuple: tuple
    config: DispersionIbsConfig


VARIANT_DICT = {
    "dispersion_ibs_kie_ihi_xlc": DispersionVariant(
        "strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_xlc", KIE_IHI_XLC_TUPLE, DispersionIbsConfig()),
    "dispersion_ibs_kie_ihi_xlc_sma200": DispersionVariant(
        "strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_xlc_asset_sma200", KIE_IHI_XLC_TUPLE,
        DispersionIbsConfig(asset_sma_int=200)),
    "dispersion_ibs_kie_ihi_sma200": DispersionVariant(
        "strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_asset_sma200", KIE_IHI_TUPLE,
        DispersionIbsConfig(asset_sma_int=200)),
}


def load_inputs(variant_name_str: str, end_date_str: str | None = None) -> SectorEtfInputs:
    return load_etf_inputs(VARIANT_DICT[variant_name_str].symbol_tuple, HISTORY_START_STR, BACKTEST_START_STR, end_date_str)


def dispersion_feature_dict(high_df: pd.DataFrame, low_df: pd.DataFrame, close_df: pd.DataFrame, range_vol_lookback_int: int,
                            asset_sma_int: int) -> dict:
    """IBS, RelativeRange and (optionally) the SMA gate. *** CRITICAL*** the range std is shifted one session."""
    range_df = log_range_df(high_df, low_df)
    range_vol_df = range_df.rolling(range_vol_lookback_int, min_periods=range_vol_lookback_int).std().shift(1)
    feature_dict = {"ibs_df": ibs_df(close_df, high_df, low_df), "relative_range_df": range_df / range_vol_df.replace(0.0, np.nan)}
    if asset_sma_int > 0:
        sma_df = close_df.rolling(asset_sma_int, min_periods=asset_sma_int).mean()
        feature_dict["bullish_df"] = close_df.gt(sma_df) & sma_df.notna()
    return feature_dict


def signal_mats(feature_dict: dict, config: DispersionIbsConfig) -> tuple[np.ndarray, np.ndarray]:
    ibs_mat = feature_dict["ibs_df"].to_numpy(float)
    relative_range_mat = feature_dict["relative_range_df"].to_numpy(float)
    with np.errstate(invalid="ignore"):
        wide_bool_mat = relative_range_mat > config.min_relative_range_float
        entry_mat = (ibs_mat < config.entry_ibs_max_float) & wide_bool_mat
        exit_mat = (ibs_mat > config.exit_ibs_min_float) & wide_bool_mat
    if config.asset_sma_int > 0:
        entry_mat &= feature_dict["bullish_df"].to_numpy(bool)
    return entry_mat, exit_mat


def feature_dict(inputs: SectorEtfInputs, config: DispersionIbsConfig) -> dict:
    return dispersion_feature_dict(inputs.high_df, inputs.low_df, inputs.close_df, config.range_vol_lookback_int, config.asset_sma_int)


def entry_weight_float(config: DispersionIbsConfig, asset_count_int: int) -> float:
    return float(config.portfolio_leverage_float) / float(asset_count_int)  # S: __init__ :510


def simulate_config(inputs: SectorEtfInputs, config: DispersionIbsConfig, cost_model: CostModel = ENGINE_COST_MODEL,
                    capital_float: float = 100_000.0) -> WeightsResult:
    entry_mat, exit_mat = signal_mats(feature_dict(inputs, config), config)
    asset_count_int = inputs.close_df.shape[1]
    start_ts = execution_start_ts(inputs, config.required_history_int, config.asset_sma_int or None, skip_ready_bar_bool=False)
    return simulate_event_rule(inputs, entry_mat, exit_mat, None, asset_count_int, entry_weight_float(config, asset_count_int),
                               start_ts, cost_model, capital_float)


# ---------------------------------------------------------------- MCPT replica (S5)
def mcpt_matrix(inputs: SectorEtfInputs, end_date_str: str = SEAL_END_STR) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """`sector_ibs.bar_matrix`: TR daily returns, then gap / high / low / close log-bar blocks per ETF."""
    return bar_matrix(inputs, end_date_str)


def fast_daily_list(matrix: np.ndarray, date_index: pd.DatetimeIndex, config_list: list[dict], base_config: DispersionIbsConfig) -> list[np.ndarray]:
    """Gross daily returns per configuration (`config_list` holds overrides of `base_config`): OHLC rebuilt from the log
    bars, IBS / RelativeRange / SMA recomputed (the SMA warms up inside the matrix), every ETF its own 1/N sleeve."""
    from dataclasses import replace

    asset_count_int = (matrix.shape[1]) // 5
    column_list = [f"a{i}" for i in range(asset_count_int)]
    ohlc_dict = rebuilt_ohlc_dict(matrix, asset_count_int, date_index, column_list)
    open_mat, close_mat = ohlc_dict["Open"].to_numpy(), ohlc_dict["Close"].to_numpy()
    no_rank_mat = np.zeros_like(close_mat)
    feature_cache_dict, daily_list = {}, []
    for config_dict in config_list:
        config = replace(base_config, **config_dict)
        key_tuple = (config.range_vol_lookback_int, config.asset_sma_int)
        if key_tuple not in feature_cache_dict:
            feature_cache_dict[key_tuple] = dispersion_feature_dict(ohlc_dict["High"], ohlc_dict["Low"], ohlc_dict["Close"], *key_tuple)
        entry_mat, exit_mat = signal_mats(feature_cache_dict[key_tuple], config)
        daily_list.append(_event_book_daily(open_mat, close_mat, entry_mat, exit_mat, no_rank_mat, False, asset_count_int,
                                            entry_weight_float(config, asset_count_int)))
    return daily_list


# ---------------------------------------------------------------- S3 (class E)
def s3_inputs(variant_name_str: str, inputs: SectorEtfInputs | None = None, end_date_str: str = SEAL_END_STR) -> dict:
    """Inputs for alpha.scout.stations.s3_edge.run_s3, as `sector_ibs.s3_inputs`: event = the entry signal at T (with the
    SMA gate where the variant has it), regime = every basket ETF with a valid bar, horizon = the S3 horizon nearest the
    live run's median holding period. No indicator deciles: per-date deciles of five or six ETFs are degenerate. The
    excess is over the same-date basket mean, so S3 tests the relative (dispersion) edge; the absolute bounce shared by
    the whole basket is S5's question."""
    variant = VARIANT_DICT[variant_name_str]
    inputs = inputs or load_inputs(variant_name_str)
    panel = s3_panel(inputs, end_date_str)
    features = feature_dict(inputs, variant.config)
    entry_mat, _ = signal_mats(features, variant.config)
    event_df = pd.DataFrame(entry_mat, index=inputs.close_df.index, columns=inputs.close_df.columns).loc[:end_date_str]
    hold_int = median_holding_sessions_int(simulate_config(inputs, variant.config))
    horizon_int = min(S3_HORIZON_TUPLE, key=lambda h: (abs(h - hold_int), h))
    return {"name_str": f"sector dispersion IBS ({variant_name_str})", "panel": panel, "regime_mask_df": panel.member_df == 1,
            "event_mask_df": event_df, "horizon_int": horizon_int, "indicator_df": None, "expected_sign_int": 1}
