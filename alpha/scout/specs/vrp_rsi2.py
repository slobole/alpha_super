"""Scout spec of the VRP RSI2 sleeve (scripts/research/vrp_rsi2_build_20261103/SPEC_FROZEN.md): the engine module's rule
through the Scout weights engine, for the identity gate and the S8 shadow.

The rule itself (targets and the band) is `strategies.vix_stuff.strategy_vrp_svxy_vixm_rsi2.compute_feature_df`; this
spec owns the execution replica: a close T whose band fires becomes a full-target rebalance at the next session's open,
shares trunc(V x w / Close_T), 5 bp slippage and the IBKR fixed fee, both assets traded to target, drift in between.
Its parity with the engine is the gate (`python -m alpha.scout gate vrp_rsi2 --fresh`).

*** CRITICAL*** A target decided at Close_T trades at Open of the next session in the price index, never at T.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from alpha.scout.engines.weights import CostModel, WeightsResult, simulate
from strategies.vix_stuff import strategy_vrp_svxy_vixm_rsi2 as module

TRADED_TUPLE = module.TRADED_ASSET_TUPLE
LIVE_CONFIG = module.DEFAULT_CONFIG
ENGINE_COST_MODEL = CostModel(slippage_float=module.SLIPPAGE_FLOAT, fee_per_share_float=module.COMMISSION_PER_SHARE_FLOAT,
                              min_fee_float=module.COMMISSION_MINIMUM_FLOAT, dividend_withholding_float=0.25)


@dataclass(frozen=True)
class VrpInputs:
    pricing_data_df: pd.DataFrame
    calendar_idx: pd.DatetimeIndex
    feature_df: pd.DataFrame

    @property
    def open_df(self) -> pd.DataFrame:
        return pd.DataFrame({a: self.pricing_data_df[(a, "Open")] for a in TRADED_TUPLE}).astype(float)

    @property
    def close_df(self) -> pd.DataFrame:
        return pd.DataFrame({a: self.pricing_data_df[(a, "Close")] for a in TRADED_TUPLE}).astype(float)

    @property
    def dividend_df(self) -> pd.DataFrame:
        return pd.DataFrame({a: self.pricing_data_df[(a, "Dividend")] for a in TRADED_TUPLE}).astype(float).fillna(0.0)


def load_inputs(end_date_str: str | None = None, config_obj: module.VrpRsi2Config = LIVE_CONFIG) -> VrpInputs:
    config_obj = replace(config_obj, end_date_str=end_date_str)
    pricing_data_df = module.get_vrp_data(config_obj)
    calendar_idx = module.build_execution_calendar_idx(pricing_data_df, config_obj)
    feature_df = module.compute_feature_df(module.signal_frame_df(pricing_data_df), config_obj,
                                           module.first_decision_date_ts(pricing_data_df.index, config_obj))
    return VrpInputs(pricing_data_df, calendar_idx, feature_df)


def rebalance_weight_df(inputs: VrpInputs) -> pd.DataFrame:
    """Fill-dated targets: the decision of Close_T on the next session of the price index."""
    session_idx = inputs.pricing_data_df.index
    decided_df = inputs.feature_df.loc[inputs.feature_df["rebalance_bool"], ["target_SVXY", "target_VIXM"]]
    position_vec = session_idx.get_indexer(decided_df.index) + 1
    keep_vec = position_vec < len(session_idx)
    weight_df = pd.DataFrame(decided_df.to_numpy()[keep_vec], index=session_idx[position_vec[keep_vec]], columns=list(TRADED_TUPLE))
    return weight_df.loc[weight_df.index >= inputs.calendar_idx[0]]


def simulate_config(inputs: VrpInputs, cost_model: CostModel = ENGINE_COST_MODEL, capital_float: float = LIVE_CONFIG.capital_base_float) -> WeightsResult:
    """The rule is in `inputs.feature_df` (load_inputs takes the config); this replays its trades."""
    calendar_idx = inputs.calendar_idx
    # The weights engine reads Close(T) of the session before start_date, so keep one session ahead of the calendar.
    first_pos_int = inputs.pricing_data_df.index.get_loc(calendar_idx[0]) - 1
    frame_idx = inputs.pricing_data_df.index[first_pos_int:]
    frame_idx = frame_idx[frame_idx <= calendar_idx[-1]]
    return simulate(inputs.open_df.loc[frame_idx], inputs.close_df.loc[frame_idx], inputs.dividend_df.loc[frame_idx],
                    rebalance_weight_df(inputs), calendar_idx[0], capital_float=capital_float,
                    cost_model=cost_model)


def frozen_config_dict() -> dict:
    return {**asdict(replace(LIVE_CONFIG, end_date_str=None)), "cost_model": asdict(ENGINE_COST_MODEL),
            "strategy_import_str": "strategies.vix_stuff.strategy_vrp_svxy_vixm_rsi2"}


def shadow_daily_return_ser(shadow_row_dict: dict) -> pd.Series:
    """S8 provider: the frozen rule on the newest data. Refuses when the rule in code no longer equals the frozen row."""
    if shadow_row_dict["frozen_config_dict"] != frozen_config_dict():
        raise RuntimeError("The VRP RSI2 rule in code differs from the frozen shadow row; a changed rule is a new registration.")
    inputs = load_inputs()
    # *** CRITICAL*** A missing input is a silent no-trade in the band; the engine raises, so the provider must too.
    decision_idx = inputs.pricing_data_df.index[inputs.pricing_data_df.index.get_loc(inputs.calendar_idx[0]) - 1:]
    target_df = inputs.feature_df.reindex(decision_idx)[["target_SVXY", "target_VIXM"]]
    if target_df.iloc[:-1].isna().any().any():
        raise RuntimeError("A VRP signal input is missing on a decision date; the shadow refuses to fill it silently.")
    daily_ser = simulate_config(inputs).daily_return_ser.astype(float)
    # Norgate can hold a partial bar for today during market hours: keep only sessions before today (New York).
    today_ts = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    return daily_ser.loc[daily_ser.index < today_ts]


def daily_target_cash_ser(inputs: VrpInputs) -> pd.Series:
    """Decided cash share (1 - targets) carried between trades, for books that add T-bills on idle cash."""
    weight_df = rebalance_weight_df(inputs).reindex(inputs.calendar_idx).ffill().fillna(0.0)
    return (1.0 - weight_df.sum(axis=1)).clip(lower=0.0).astype(float)
