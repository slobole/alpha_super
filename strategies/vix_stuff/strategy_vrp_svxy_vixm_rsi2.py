"""Volatility risk premium sleeve: short SVXY in VIX contango with an RSI(2) dip boost, long VIXM in a crash.

Frozen rule of scripts/research/vrp_rsi2_build_20261103/SPEC_FROZEN.md (owner lock of 2026-11-03, Scout registration
vrp_rsi2_locked_retro_20261103). After the official Close_T:

    ERV10 = sample std (ddof 1) of the last 10 SPY total-return close-to-close returns x sqrt(252) x 100
    eVRP  = VIX - ERV10,  TS = VIX / VIX3M,  RSI2 = Wilder RSI(2) of the $SPX price close

    TS < 0.98 and eVRP > 0    SVXY = min(2.0 VIX/100, 1), x1.25 (cap 1) when RSI2 < 25
    TS < 0.98 and eVRP <= 0   SVXY = min(1.0 VIX/100, 1), or min(1.5 VIX/100, 1) when RSI2 < 20
    0.98 <= TS <= 1.00        cash
    TS > 1.00 and eVRP <= 0   VIXM = min(1.25 VIX/100, 1)
    TS > 1.00 and eVRP > 0    cash

SVXY was -1x before 2018-02-28 and -0.5x since, so its target is halved for decision dates before that day. A decision
trades when a target turns on or off or either target moves 5 points from the last traded target; positions drift
in between. Shares are fixed from the prior-close NAV and Close_T; the fill is Open_(T+1).

Research only: no LIVE, broker, scheduler or allocation wiring.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
import pandas as pd

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from data.norgate_loader import load_raw_prices

STRATEGY_NAME_STR = "strategy_vrp_svxy_vixm_rsi2"
SHORT_VOL_ASSET_STR = "SVXY"
LONG_VOL_ASSET_STR = "VIXM"
TRADED_ASSET_TUPLE = (SHORT_VOL_ASSET_STR, LONG_VOL_ASSET_STR)
BENCHMARK_TUPLE = ("$SPX",)
SIGNAL_NAMESPACE_STR = "VRP_SIGNAL"
SIGNAL_SOURCE_DICT = {"vix_close": "$VIX", "vix3m_close": "$VIX3M", "spx_close": "$SPX"}
SPY_TOTAL_RETURN_FIELD_STR = "spy_tr_close"
FEATURE_FIELD_TUPLE = ("erv10", "evrp", "term_ratio", "rsi2", "regime_int", "target_SVXY", "target_VIXM", "rebalance_bool")
SLIPPAGE_FLOAT = 0.0005
COMMISSION_PER_SHARE_FLOAT = 0.005
COMMISSION_MINIMUM_FLOAT = 1.0


@dataclass(frozen=True)
class VrpRsi2Config:
    contango_threshold_float: float = 0.98
    backwardation_threshold_float: float = 1.00
    svxy_vix_multiple_float: float = 2.0
    vixm_vix_multiple_float: float = 1.25
    erv_window_int: int = 10
    rsi_period_int: int = 2
    rsi_boost_threshold_float: float = 25.0
    rsi_boost_multiple_float: float = 1.25
    rsi_half_threshold_float: float = 20.0
    rsi_half_vix_multiple_float: float = 1.5
    band_float: float = 0.05
    svxy_halving_end_date_str: str = "2018-02-28"  # first decision date with SVXY at -0.5x
    history_start_date_str: str = "2011-01-03"
    backtest_start_date_str: str = "2011-10-05"  # the first open after SVXY's first close (2011-10-04)
    end_date_str: str | None = None
    capital_base_float: float = 100_000.0

    def __post_init__(self) -> None:
        if not 0.0 < self.contango_threshold_float <= self.backwardation_threshold_float:
            raise ValueError("contango_threshold_float must be positive and at most backwardation_threshold_float.")
        if self.band_float < 0.0 or self.erv_window_int < 2 or self.rsi_period_int < 1:
            raise ValueError("band_float must be >= 0, erv_window_int >= 2 and rsi_period_int >= 1.")
        if not np.isfinite(self.capital_base_float) or self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive and finite.")


DEFAULT_CONFIG = VrpRsi2Config()


def wilder_rsi_ser(close_ser: pd.Series, period_int: int) -> pd.Series:
    """Wilder RSI with exponential smoothing alpha = 1/period. A zero average loss reads 50, as the research did."""
    # *** CRITICAL*** Validate the entire supplied Close_T history before recursive RSI smoothing.
    # Missing/non-positive SPX prices must never become RSI=50 or contaminate later T+1 orders.
    invalid_close_ser = ~np.isfinite(close_ser) | (close_ser <= 0.0)
    if invalid_close_ser.any():
        invalid_date_ts = pd.Timestamp(close_ser.index[invalid_close_ser][0])
        raise ValueError(f"Invalid SPX close on {invalid_date_ts.date()}: RSI requires finite positive history.")
    change_ser = close_ser.diff()
    gain_ser = change_ser.clip(lower=0.0).ewm(alpha=1.0 / period_int, adjust=False).mean()
    loss_ser = (-change_ser).clip(lower=0.0).ewm(alpha=1.0 / period_int, adjust=False).mean()
    rsi_ser = 100.0 - 100.0 / (1.0 + gain_ser / loss_ser.replace(0.0, np.nan))
    return rsi_ser.fillna(50.0)


def compute_feature_df(signal_df: pd.DataFrame, config_obj: VrpRsi2Config = DEFAULT_CONFIG, first_decision_date=None) -> pd.DataFrame:
    """Targets at each close from that close and earlier closes only. `signal_df` holds vix_close, vix3m_close,
    spx_close and spy_tr_close on one session index; the targets are NaN until every input exists and before
    `first_decision_date` (the close before the first scored open), so the band starts flat there."""
    vix_ser, vix3m_ser = signal_df["vix_close"].astype(float), signal_df["vix3m_close"].astype(float)
    spy_return_ser = signal_df[SPY_TOTAL_RETURN_FIELD_STR].astype(float).pct_change(fill_method=None)
    # *** CRITICAL*** rolling and ewm read only rows at or before T; nothing is shifted backward.
    erv_ser = spy_return_ser.rolling(config_obj.erv_window_int, min_periods=config_obj.erv_window_int).std(ddof=1) * np.sqrt(252.0) * 100.0
    evrp_ser = vix_ser - erv_ser
    term_ratio_ser = vix_ser / vix3m_ser
    rsi_ser = wilder_rsi_ser(signal_df["spx_close"].astype(float), config_obj.rsi_period_int)

    valid_vec = (np.isfinite(vix_ser) & np.isfinite(vix3m_ser) & (vix_ser > 0) & (vix3m_ser > 0) & np.isfinite(evrp_ser)).to_numpy()
    if first_decision_date is not None:
        valid_vec &= signal_df.index >= pd.Timestamp(first_decision_date)
    vix_vec, evrp_vec, ratio_vec, rsi_vec = (s.to_numpy(dtype=float) for s in (vix_ser, evrp_ser, term_ratio_ser, rsi_ser))
    contango_vec = ratio_vec < config_obj.contango_threshold_float
    backwardation_vec = ratio_vec > config_obj.backwardation_threshold_float
    positive_vec = evrp_vec > 0.0
    full_vec = np.minimum(config_obj.svxy_vix_multiple_float * vix_vec / 100.0, 1.0)
    full_vec = np.where(rsi_vec < config_obj.rsi_boost_threshold_float, np.minimum(full_vec * config_obj.rsi_boost_multiple_float, 1.0), full_vec)
    half_vec = np.where(rsi_vec < config_obj.rsi_half_threshold_float,
                        np.minimum(config_obj.rsi_half_vix_multiple_float * vix_vec / 100.0, 1.0),
                        np.minimum(0.5 * config_obj.svxy_vix_multiple_float * vix_vec / 100.0, 1.0))
    regime_vec = np.select([contango_vec & positive_vec, contango_vec & ~positive_vec, backwardation_vec & ~positive_vec, backwardation_vec & positive_vec],
                           [1, 2, 4, 5], default=3)
    halving_vec = np.where(signal_df.index < pd.Timestamp(config_obj.svxy_halving_end_date_str), 0.5, 1.0)
    svxy_vec = np.select([regime_vec == 1, regime_vec == 2], [full_vec, half_vec], default=0.0) * halving_vec
    vixm_vec = np.where(regime_vec == 4, np.minimum(config_obj.vixm_vix_multiple_float * vix_vec / 100.0, 1.0), 0.0)

    feature_df = pd.DataFrame({
        "erv10": erv_ser, "evrp": evrp_ser, "term_ratio": term_ratio_ser, "rsi2": rsi_ser,
        "regime_int": np.where(valid_vec, regime_vec, np.nan),
        "target_SVXY": np.where(valid_vec, svxy_vec, np.nan), "target_VIXM": np.where(valid_vec, vixm_vec, np.nan),
    }, index=signal_df.index)
    feature_df["rebalance_bool"] = rebalance_flag_ser(feature_df[["target_SVXY", "target_VIXM"]], config_obj.band_float)
    return feature_df


def rebalance_flag_ser(target_df: pd.DataFrame, band_float: float) -> pd.Series:
    """True on a close whose targets must be traded: a target turns on or off, or one moved `band_float` or more since
    the last traded target. Starts flat. NaN targets (warm-up) never trade."""
    flag_list, last_vec = [], np.zeros(target_df.shape[1])
    for target_vec in target_df.to_numpy(dtype=float):
        if not np.isfinite(target_vec).all():
            flag_list.append(False)
            continue
        trade_bool = band_trade_bool(target_vec, last_vec, band_float)
        if trade_bool:
            last_vec = target_vec.copy()
        flag_list.append(trade_bool)
    return pd.Series(flag_list, index=target_df.index, dtype=bool)


def band_trade_bool(target_vec: np.ndarray, last_vec: np.ndarray, band_float: float) -> bool:
    """A target turns on or off, or one moved `band_float` or more from the last traded target."""
    switch_bool = bool(np.any((target_vec == 0.0) != (last_vec == 0.0)))
    return switch_bool or float(np.max(np.abs(target_vec - last_vec))) >= band_float


def get_vrp_data(config_obj: VrpRsi2Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """CAPITALSPECIAL bars for SVXY and VIXM, $SPX total return for the benchmark, and the four signal closes."""
    execution_df = load_raw_prices(symbols=list(TRADED_ASSET_TUPLE), benchmarks=list(BENCHMARK_TUPLE),
                                   start_date=config_obj.history_start_date_str, end_date=config_obj.end_date_str)
    index_df = load_raw_prices(symbols=list(SIGNAL_SOURCE_DICT.values()), benchmarks=[],
                               start_date=config_obj.history_start_date_str, end_date=config_obj.end_date_str)
    spy_df = load_raw_prices(symbols=[], benchmarks=["SPY"], start_date=config_obj.history_start_date_str, end_date=config_obj.end_date_str)
    signal_df = pd.DataFrame({field_str: index_df[(symbol_str, "Close")] for field_str, symbol_str in SIGNAL_SOURCE_DICT.items()})
    signal_df[SPY_TOTAL_RETURN_FIELD_STR] = spy_df[("SPY", "Close")]
    signal_df.columns = pd.MultiIndex.from_tuples([(SIGNAL_NAMESPACE_STR, field_str) for field_str in signal_df.columns])
    pricing_data_df = pd.concat([execution_df, signal_df], axis=1).sort_index()
    # Keep only sessions where the signal history exists; SVXY rows before its listing stay NaN and are never scored.
    pricing_data_df = pricing_data_df.loc[pricing_data_df[(SIGNAL_NAMESPACE_STR, SPY_TOTAL_RETURN_FIELD_STR)].notna()].copy()
    pricing_data_df.attrs.update(execution_df.attrs)
    adjustment_dict = dict(execution_df.attrs.get("norgate_adjustment_by_symbol_dict", {}))
    adjustment_dict[SIGNAL_NAMESPACE_STR] = "RAW_INDEX_CLOSE + SPY TOTALRETURN"
    pricing_data_df.attrs["norgate_adjustment_by_symbol_dict"] = adjustment_dict
    return pricing_data_df


def signal_frame_df(pricing_data_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({field_str: pricing_data_df[(SIGNAL_NAMESPACE_STR, field_str)] for field_str in (*SIGNAL_SOURCE_DICT, SPY_TOTAL_RETURN_FIELD_STR)})


def first_decision_date_ts(session_idx: pd.DatetimeIndex, config_obj: VrpRsi2Config = DEFAULT_CONFIG) -> pd.Timestamp:
    """The close before the first scored open: the first decision."""
    earlier_idx = session_idx[session_idx < pd.Timestamp(config_obj.backtest_start_date_str)]
    if len(earlier_idx) == 0:
        raise ValueError("The first scored open needs a prior close.")
    return pd.Timestamp(earlier_idx[-1])


def build_execution_calendar_idx(pricing_data_df: pd.DataFrame, config_obj: VrpRsi2Config = DEFAULT_CONFIG) -> pd.DatetimeIndex:
    calendar_idx = pricing_data_df.index[pricing_data_df.index >= pd.Timestamp(config_obj.backtest_start_date_str)]
    if len(calendar_idx) < 2:
        raise ValueError("At least two scoring sessions are required.")
    bar_df = pricing_data_df.loc[calendar_idx, [(a, f) for a in TRADED_ASSET_TUPLE for f in ("Open", "Close")]].astype(float)
    if not (np.isfinite(bar_df.to_numpy()).all() and (bar_df.to_numpy() > 0).all()):
        raise ValueError("SVXY or VIXM has a missing or non-positive Open/Close inside the scoring calendar.")
    return pd.DatetimeIndex(calendar_idx)


class VrpRsi2Strategy(Strategy):
    """Daily decision, trades only on band crossings; SVXY and VIXM are never held together."""

    enable_signal_audit = True
    signal_audit_sample_size = 10
    holding_check_bool = True

    def __init__(self, config_obj: VrpRsi2Config = DEFAULT_CONFIG) -> None:
        super().__init__(name=STRATEGY_NAME_STR, benchmarks=list(BENCHMARK_TUPLE), capital_base=config_obj.capital_base_float,
                         slippage=SLIPPAGE_FLOAT, commission_per_share=COMMISSION_PER_SHARE_FLOAT, commission_minimum=COMMISSION_MINIMUM_FLOAT,
                         performance_benchmark_symbol_str=BENCHMARK_TUPLE[0], performance_benchmark_adjustment_str="TOTALRETURN")
        self.config_obj = config_obj
        self.asset_list = list(TRADED_ASSET_TUPLE)
        self.trade_id_int = 0
        self.trade_id_by_asset_dict = {asset_str: -1 for asset_str in self.asset_list}
        self.last_traded_share_dict = {asset_str: 0 for asset_str in self.asset_list}
        self.last_traded_target_vec = np.zeros(len(self.asset_list))
        self.feature_df = pd.DataFrame()
        self.decision_row_list: list[dict] = []
        self.configure_dividend_cash_ledger(enabled_bool=True)
        self._data_adjustment_policy_dict.update({"signal_adjustment_str": "RAW_INDEX_CLOSE + SPY TOTALRETURN",
                                                  "execution_and_marks_adjustment_str": "CAPITALSPECIAL",
                                                  "performance_benchmark_adjustment_str": "TOTALRETURN"})
        self._accounting_policy_dict.update({"state_transition_execution_str": "Close_T_to_Open_(T+1)", "missing_signal_policy_str": "fail_loud",
                                             "svxy_history_halving_str": f"x0.5 for decision dates before {config_obj.svxy_halving_end_date_str}"})

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        first_decision_ts = first_decision_date_ts(pricing_data_df.index, self.config_obj)
        feature_df = compute_feature_df(signal_frame_df(pricing_data_df), self.config_obj, first_decision_ts)
        if len(feature_df) >= len(self.feature_df):
            self.feature_df = feature_df
        feature_df = feature_df.copy()
        feature_df.columns = pd.MultiIndex.from_tuples([(SIGNAL_NAMESPACE_STR, field_str) for field_str in feature_df.columns])
        return pd.concat([pricing_data_df, feature_df], axis=1)

    def _trade_id_for(self, asset_str: str, target_share_int: int) -> int:
        if self.last_traded_share_dict[asset_str] == 0 and target_share_int != 0:
            self.trade_id_int += 1
            self.trade_id_by_asset_dict[asset_str] = self.trade_id_int
        return self.trade_id_by_asset_dict[asset_str]

    def iterate(self, data_df: pd.DataFrame, close_row_ser: pd.Series, _open_price_ser: pd.Series) -> None:
        if data_df is None or close_row_ser is None:
            return
        for asset_str in self.asset_list:
            # *** CRITICAL*** A cancelled fill must not be forgotten silently: holdings equal the last traded shares.
            if self.holding_check_bool and int(self.get_position(asset_str)) != self.last_traded_share_dict[asset_str]:
                raise RuntimeError(f"{asset_str} holding differs from the last traded target on {self.previous_bar}.")
        target_svxy_float = float(close_row_ser[(SIGNAL_NAMESPACE_STR, "target_SVXY")])
        target_vixm_float = float(close_row_ser[(SIGNAL_NAMESPACE_STR, "target_VIXM")])
        if not (np.isfinite(target_svxy_float) and np.isfinite(target_vixm_float)):
            raise RuntimeError(f"Missing VRP signal at Close_{self.previous_bar}.")
        target_vec = np.array([target_svxy_float, target_vixm_float])
        # The band runs on this run's own traded targets, so a run that starts later (stress launches) starts flat
        # there. From the first decision this equals the precomputed rebalance_bool.
        if not band_trade_bool(target_vec, self.last_traded_target_vec, self.config_obj.band_float):
            return
        self.last_traded_target_vec = target_vec
        nav_float = float(self.previous_total_value)
        if not np.isfinite(nav_float) or nav_float <= 0.0:
            raise RuntimeError("Prior-close NAV must be positive.")
        target_share_dict = {}
        for asset_str, weight_float in ((SHORT_VOL_ASSET_STR, target_svxy_float), (LONG_VOL_ASSET_STR, target_vixm_float)):
            close_float = float(close_row_ser[(asset_str, "Close")])
            if not np.isfinite(close_float) or close_float <= 0.0:
                raise RuntimeError(f"Invalid sizing close for {asset_str} on {self.previous_bar}.")
            # *** CRITICAL*** Shares from Close_T and the prior-close NAV; Open_(T+1) only prices the fill.
            target_share_dict[asset_str] = int(np.trunc(nav_float * weight_float / close_float))
        for asset_str, share_int in target_share_dict.items():
            self._submit_share_order(asset_str, share_int)
            self.last_traded_share_dict[asset_str] = share_int
        self.decision_row_list.append({"decision_date": self.previous_bar, "fill_date": self.current_bar, "sizing_nav_float": nav_float,
                                       "target_SVXY": target_svxy_float, "target_VIXM": target_vixm_float, **target_share_dict})

    def _submit_share_order(self, asset_str: str, share_int: int) -> None:
        if share_int != int(self.get_position(asset_str)):
            self.order_target(asset_str, share_int, trade_id=self._trade_id_for(asset_str, share_int))

    def finalize(self, current_data_df: pd.DataFrame) -> None:
        self.decision_df = pd.DataFrame(self.decision_row_list)


class VrpRsi2TimingStrategy(VrpRsi2Strategy):
    """For the Timing analyzer, where a delayed fill can still be pending at the next decision."""

    # The analyzer queues fills outside the strategy, so holdings may lag the last traded target by design.
    holding_check_bool = False

    def _submit_share_order(self, asset_str: str, share_int: int) -> None:
        # *** CRITICAL*** Order the change from the last traded shares, not the held shares: a pending fill must not be
        # ordered twice.
        delta_int = share_int - self.last_traded_share_dict[asset_str]
        if delta_int != 0:
            self.order(asset_str, delta_int, trade_id=self._trade_id_for(asset_str, share_int))


def _build_analysis_context_dict(config_obj: VrpRsi2Config = DEFAULT_CONFIG, pricing_data_df: pd.DataFrame | None = None) -> dict[str, object]:
    if pricing_data_df is None:
        pricing_data_df = get_vrp_data(config_obj)
    calendar_idx = build_execution_calendar_idx(pricing_data_df, config_obj)
    return {"strategy_name_str": STRATEGY_NAME_STR, "capital_base_float": config_obj.capital_base_float, "config_obj": config_obj,
            "pricing_data_df": pricing_data_df, "calendar_idx": calendar_idx}


@lru_cache(maxsize=2)
def _capacity_pricing_df(end_date_str: str | None) -> pd.DataFrame:
    """Freeze one input frame per endpoint across the AUM grid in this process."""
    return get_vrp_data(replace(DEFAULT_CONFIG, end_date_str=end_date_str))


def build_capacity_analysis_inputs(capital_base_float: float, show_display_bool: bool = False, backtest_start_date_str: str | None = None, end_date_str: str | None = None) -> dict[str, object]:
    config_obj = replace(DEFAULT_CONFIG, capital_base_float=capital_base_float, end_date_str=end_date_str,
                         backtest_start_date_str=backtest_start_date_str or DEFAULT_CONFIG.backtest_start_date_str)
    context_dict = _build_analysis_context_dict(config_obj, _capacity_pricing_df(end_date_str).copy(deep=True))
    strategy_obj = VrpRsi2Strategy(config_obj)
    run_daily(strategy_obj, context_dict["pricing_data_df"], calendar=context_dict["calendar_idx"], show_progress=show_display_bool, show_signal_progress_bool=show_display_bool)
    return {"strategy_obj": strategy_obj, "pricing_data_df": context_dict["pricing_data_df"], "execution_policy_str": "MOO",
            "impact_profile_str": "MOO_ETF_PROXY"}


def build_execution_timing_analysis_inputs() -> dict[str, object]:
    context_dict = _build_analysis_context_dict()

    def strategy_factory_fn() -> VrpRsi2TimingStrategy:
        strategy_obj = VrpRsi2TimingStrategy(context_dict["config_obj"])
        strategy_obj.compute_signals(context_dict["pricing_data_df"])
        return strategy_obj

    return {"strategy_factory_fn": strategy_factory_fn, "pricing_data_df": context_dict["pricing_data_df"], "calendar_idx": context_dict["calendar_idx"],
            "order_generation_mode_str": "vanilla_current_bar", "risk_model_str": "taa_rebalance",
            "entry_timing_str_tuple": ("same_open", "same_close_moc", "next_open", "next_close"), "exit_timing_str_tuple": ("same_open", "same_close_moc", "next_open", "next_close"),
            "default_entry_timing_str": "same_open", "default_exit_timing_str": "same_open"}


def build_stress_test_context_dict() -> dict[str, object]:
    return _build_analysis_context_dict()


def build_stress_test_strategy_obj(context_dict: dict[str, object]) -> VrpRsi2Strategy:
    return VrpRsi2Strategy(context_dict["config_obj"])


def run_variant(show_display_bool: bool = False, save_results_bool: bool = True, output_dir_str: str = "results",
                backtest_start_date_str: str | None = None, end_date_str: str | None = None,
                capital_base_float: float = DEFAULT_CONFIG.capital_base_float, config_obj: VrpRsi2Config | None = None) -> VrpRsi2Strategy:
    config_obj = replace(config_obj or DEFAULT_CONFIG, capital_base_float=capital_base_float, end_date_str=end_date_str,
                         backtest_start_date_str=backtest_start_date_str or DEFAULT_CONFIG.backtest_start_date_str)
    pricing_data_df = get_vrp_data(config_obj)
    calendar_idx = build_execution_calendar_idx(pricing_data_df, config_obj)
    strategy_obj = VrpRsi2Strategy(config_obj)
    run_daily(strategy_obj, pricing_data_df, calendar=calendar_idx, show_progress=show_display_bool, show_signal_progress_bool=show_display_bool)
    if show_display_bool:
        print(strategy_obj.summary.to_string())
    if save_results_bool:
        output_path = save_results(strategy_obj, output_dir=output_dir_str)
        strategy_obj.output_path_str = str(output_path.resolve())
        strategy_obj.feature_df.to_csv(output_path / "feature_table.csv")
        strategy_obj.decision_df.to_csv(output_path / "decision_table.csv", index=False)
    return strategy_obj


if __name__ == "__main__":
    run_variant(show_display_bool=True)
