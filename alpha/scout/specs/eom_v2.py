"""Scout spec of the month-end flow split into its rules (study scripts/research/eom_tradable_20261101/SPEC_FROZEN.md).

The schedule, the drift P, the rank and the costs are those of `alpha.scout.specs.eom`, whose gate against the engine
stays untouched. This spec adds what that one fixes: which final-week rule, which next-month rule, and where the fills
happen. `final_rule_str="rank"`, `early_rule_str="main"`, `fill_str="close"` reproduce `eom` exactly (tested).

For month m with XNYS sessions S_m and k = entry_dtme_int, x = exit_session_int:
    measure  S_m[-(k+1)], the close that sets P and the rank (one full session before a close fill)
    half     SPY minus TLT, total return since the last close of m-1, at session min(15, N-k) of m (at or before measure)
    first15  SPY minus TLT, same base, at session min(15, N) of m; used only by the next month's leg
Votes for SPY in the final week: rank (P in the bottom 20% of strictly prior months), sign (P < 0), laggard (half < 0).

Fill sessions, decision always at the close of the session before the fill:
    close      final S_m[-k] close, switch S_m[-1] close, exit S_(m+1)[x-1] close
    open       final S_m[-k] open,  switch S_m[-1] open,  exit S_(m+1)[x-1] open
    open_late  final S_m[-k] open,  switch S_(m+1)[0] open, exit S_(m+1)[x-1] open

*** CRITICAL*** Every signal is a close at or before the decision close. An open fill on session t reads nothing of t.
The first15 spread of m is measured at session 15, at least two sessions before its use at the switch of m.

MCPT (`mcpt_matrix`, `fast_daily_list`): columns SPY, TLT (TR close-to-close, traded), IEF (signal only), then SPY and TLT
overnight TR returns, so a plain date shuffle keeps each day's overnight and intraday parts together. The replica is
gross and holds constant weights between fills.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alpha.scout.engines.weights import BorrowModel, CostModel, WeightsResult, simulate
from alpha.scout.specs import eom

TRADED_TUPLE = eom.TRADED_TUPLE
FINAL_RULE_TUPLE = ("tlt", "rank", "vote", "sign", "laggard", "half")
EARLY_RULE_TUPLE = ("flat", "pair_top", "pair_sign", "main", "spy_led")
FILL_TUPLE = ("close", "open", "open_late")
ENGINE_COST_MODEL = eom.ENGINE_COST_MODEL
FLAT_TUPLE, PAIR_TUPLE = (0.0, 0.0), (0.5, -0.5)


@dataclass(frozen=True)
class EomV2Config:
    final_rule_str: str = "vote"
    early_rule_str: str = "pair_top"
    fill_str: str = "close"
    entry_dtme_int: int = 6
    exit_session_int: int = 5
    decision_offset_int: int = 0
    min_prior_month_int: int = 24
    annual_borrow_rate_float: float = 0.01
    backtest_start_date_str: str = "2003-01-02"

    def __post_init__(self):
        if self.final_rule_str not in FINAL_RULE_TUPLE or self.early_rule_str not in EARLY_RULE_TUPLE or self.fill_str not in FILL_TUPLE:
            raise ValueError("EomV2Config: unknown rule or fill.")
        if not 2 <= self.entry_dtme_int <= eom.MIN_MONTH_SESSION_INT - 2 or not 2 <= self.exit_session_int <= eom.MIN_MONTH_SESSION_INT:
            raise ValueError("EomV2Config: entry_dtme_int in 2..7 and exit_session_int in 2..9.")
        if self.decision_offset_int != 0:
            raise ValueError("EomV2Config: a calendar-flow rule has no luck band (decision_offset_int must be 0).")

    def base_config(self) -> eom.EomConfig:
        return eom.EomConfig(entry_dtme_int=self.entry_dtme_int, exit_session_int=self.exit_session_int,
                             min_prior_month_int=self.min_prior_month_int, annual_borrow_rate_float=self.annual_borrow_rate_float,
                             backtest_start_date_str=self.backtest_start_date_str)


LIVE_CONFIG = EomV2Config()


@dataclass(frozen=True)
class EomV2Inputs:
    base: eom.EomInputs
    total_return_open_df: pd.DataFrame  # TOTALRETURN Open of SPY and TLT (MCPT only)
    cache_dict: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def close_df(self) -> pd.DataFrame:
        return self.base.close_df  # the gate compares marks on this panel


def load_inputs(end_date_str: str | None = None) -> EomV2Inputs:
    from data.norgate_loader import norgatedata

    base = eom.load_inputs(end_date_str)
    open_dict = {}
    for symbol_str in TRADED_TUPLE:
        frame = norgatedata.price_timeseries(symbol_str, stock_price_adjustment_setting=norgatedata.StockPriceAdjustmentType.TOTALRETURN,
                                             padding_setting=norgatedata.PaddingType.NONE, start_date=eom.HISTORY_START_STR,
                                             end_date=end_date_str, timeseriesformat="pandas-dataframe")
        open_dict[symbol_str] = frame["Open"]
    return EomV2Inputs(base, pd.DataFrame(open_dict).reindex(base.close_df.index).astype(float))


# ---------------------------------------------------------------- signal
def month_table_df(inputs: EomV2Inputs, config: EomV2Config = LIVE_CONFIG) -> pd.DataFrame:
    """The `eom` month table plus the two SPY-minus-TLT spreads and the first session of the next month."""
    key_tuple = (config.entry_dtme_int, config.exit_session_int, config.min_prior_month_int)
    if key_tuple in inputs.cache_dict:
        return inputs.cache_dict[key_tuple]
    base = inputs.base
    table_df = eom.month_table_df(base, config.base_config()).copy()
    total_return_df, session_index = base.total_return_close_df, base.session_index
    period_index = session_index.to_period("M")
    half_list, first15_list, next_first_list = [], [], []
    for row in table_df.itertuples(index=False):
        month_index = session_index[period_index == row.month_period]
        previous_close_ts = session_index[period_index == row.month_period - 1][-1]
        count_int = len(month_index)
        # *** CRITICAL*** session min(15, N-k) (1-based) is at or before the measure session N-k: known at the decision.
        half_ts = month_index[min(15, count_int - config.entry_dtme_int) - 1]
        first15_ts = month_index[min(15, count_int) - 1]

        def spread_float(end_ts: pd.Timestamp) -> float:
            if end_ts > total_return_df.index[-1]:
                return float("nan")
            spy_float = total_return_df.at[end_ts, "SPY"] / total_return_df.at[previous_close_ts, "SPY"]
            tlt_float = total_return_df.at[end_ts, "TLT"] / total_return_df.at[previous_close_ts, "TLT"]
            return float(spy_float - tlt_float)

        half_list.append(spread_float(half_ts))
        first15_list.append(spread_float(first15_ts))
        next_first_list.append(session_index[period_index == row.month_period + 1][0])
    table_df["half_spread_float"] = half_list
    table_df["first15_spread_float"] = first15_list
    table_df["next_first_date"] = next_first_list
    inputs.cache_dict[key_tuple] = table_df
    return table_df


def spy_vote_count_int(state_str: str, pressure_float: float, half_spread_float: float) -> int:
    """Votes for SPY: the rank is in its low state, the drift is negative, SPY lagged TLT. A missing rank does not vote."""
    return int(state_str == "low") + int(pressure_float < 0.0) + int(half_spread_float < 0.0)


def final_weight_tuple(rule_str: str, state_str: str, pressure_float: float, half_spread_float: float) -> tuple[float, float]:
    """[SPY, TLT] of the final-week leg; gross 100%."""
    if rule_str == "tlt":
        spy_float = 0.0
    elif rule_str == "half":
        spy_float = 0.5
    elif rule_str == "rank":
        spy_float = float(state_str == "low")
    elif rule_str == "sign":
        spy_float = float(pressure_float < 0.0)
    elif rule_str == "laggard":
        spy_float = float(half_spread_float < 0.0)
    elif rule_str == "vote":
        spy_float = spy_vote_count_int(state_str, pressure_float, half_spread_float) / 3.0
    else:
        raise ValueError(f"Unknown final rule: {rule_str}")
    return (spy_float, 1.0 - spy_float)


def early_weight_tuple(rule_str: str, state_str: str, pressure_float: float, first15_spread_float: float) -> tuple[float, float]:
    """[SPY, TLT] of the next-month leg, decided from month m's state."""
    if rule_str == "flat":
        return FLAT_TUPLE
    if rule_str == "main":
        return eom.target_weight_tuple(state_str, "early")
    if rule_str == "pair_top":
        return PAIR_TUPLE if state_str == "high" else FLAT_TUPLE
    if rule_str == "pair_sign":
        return PAIR_TUPLE if pressure_float > 0.0 else FLAT_TUPLE
    if rule_str == "spy_led":
        return (1.0, 0.0) if first15_spread_float > 0.0 else FLAT_TUPLE
    raise ValueError(f"Unknown early rule: {rule_str}")


def rebalance_weight_df(inputs: EomV2Inputs, config: EomV2Config = LIVE_CONFIG) -> pd.DataFrame:
    """[SPY, TLT] target weights indexed by fill session."""
    base = inputs.base
    data_index, session_index = base.close_df.index, base.session_index
    row_dict = {}
    for row in month_table_df(inputs, config).itertuples(index=False):
        switch_ts = row.next_first_date if config.fill_str == "open_late" else row.early_fill_date
        target_dict = {
            row.final_fill_date: final_weight_tuple(config.final_rule_str, row.state_str, row.pressure_bps_float, row.half_spread_float),
            switch_ts: early_weight_tuple(config.early_rule_str, row.state_str, row.pressure_bps_float, row.first15_spread_float),
            row.exit_fill_date: FLAT_TUPLE,
        }
        for fill_ts, weight_tuple in target_dict.items():
            # *** CRITICAL*** the decision is the close of the XNYS session before the fill, whatever the fill price.
            decision_ts = session_index[session_index.get_loc(fill_ts) - 1]
            if decision_ts not in data_index:
                continue
            if row.measure_date > decision_ts:
                raise RuntimeError("The signal is not known at the close before its fill.")
            decision_int = int(data_index.get_loc(decision_ts))
            if decision_int + 1 >= len(data_index):
                continue
            if data_index[decision_int + 1] != fill_ts:
                raise RuntimeError(f"Missing XNYS session before {fill_ts.date()}; refusing to move a fill.")
            if fill_ts in row_dict:
                raise RuntimeError(f"Two legs share the fill session {fill_ts.date()}.")
            row_dict[fill_ts] = weight_tuple
    return pd.DataFrame.from_dict(row_dict, orient="index", columns=list(TRADED_TUPLE)).sort_index()


def simulate_config(inputs: EomV2Inputs, config: EomV2Config = LIVE_CONFIG, cost_model: CostModel = ENGINE_COST_MODEL,
                    capital_float: float = 100_000.0) -> WeightsResult:
    base, traded_list = inputs.base, list(TRADED_TUPLE)
    return simulate(
        base.open_df[traded_list], base.close_df[traded_list], base.dividend_df[traded_list], rebalance_weight_df(inputs, config),
        start_date=config.backtest_start_date_str, capital_float=capital_float, share_unit_mode_str="adjusted", cost_model=cost_model,
        allow_short_bool=True, borrow_model=BorrowModel(annual_rate_float=config.annual_borrow_rate_float),
        fill_at_close_bool=config.fill_str == "close", close_and_reopen_bool=True,
    )


# ---------------------------------------------------------------- MCPT (fast replica)
def mcpt_matrix(inputs: EomV2Inputs, end_date_str: str | None = "2022-12-30") -> tuple[pd.DatetimeIndex, np.ndarray]:
    """(date_index, matrix): SPY, TLT, IEF TR close-to-close, then SPY and TLT TR overnight, from the first session after
    the July 2002 month end to `end_date_str` (default the vault seal; None = all data)."""
    base = inputs.base
    close_df = base.total_return_close_df[["SPY", "TLT", "IEF"]]
    base_ts = base.session_index[base.session_index.to_period("M") == pd.Period(eom.PRESSURE_START_PERIOD_STR) - 1][-1]
    date_index = close_df.index[close_df.index > base_ts]
    if end_date_str is not None:
        date_index = date_index[date_index <= pd.Timestamp(end_date_str)]
    # *** CRITICAL*** return of day t = close_t / close_(t-1) - 1; overnight of t = open_t / close_(t-1) - 1.
    return_df = close_df.loc[base_ts:].pct_change(fill_method=None).reindex(date_index)
    previous_close_df = close_df[["SPY", "TLT"]].shift(1).reindex(date_index)
    overnight_df = inputs.total_return_open_df[["SPY", "TLT"]].reindex(date_index) / previous_close_df - 1.0
    matrix = np.column_stack([return_df.to_numpy(dtype=float), overnight_df.to_numpy(dtype=float)])
    if not np.isfinite(matrix).all():
        raise RuntimeError("The month-end MCPT matrix has a missing return.")
    return date_index, matrix


def _month_row_list(date_index: pd.DatetimeIndex) -> list[np.ndarray]:
    position_vec = np.arange(len(date_index))
    period_vec = date_index.to_period("M")
    return [position_vec[period_vec == p] for p in period_vec.unique()]


def _signal_table(matrix: np.ndarray, month_row_list: list[np.ndarray], entry_dtme_int: int, min_prior_int: int) -> dict:
    """One row per usable month: P, causal CDF, half and first15 spreads, from the (possibly shuffled) matrix alone."""
    growth_mat = np.vstack([np.ones((1, 3)), np.cumprod(1.0 + matrix[:, :3], axis=0)])  # row r + 1 = close of row r
    month_list, pressure_list, half_list, first15_list = [], [], [], []
    for month_int, row_vec in enumerate(month_row_list):
        if len(row_vec) < eom.MIN_MONTH_SESSION_INT:
            continue
        previous_int = month_row_list[month_int - 1][-1] if month_int > 0 else -1
        base_vec = growth_mat[previous_int + 1]
        measure_int = row_vec[-(entry_dtme_int + 1)]
        spy_float, ief_float = growth_mat[measure_int + 1, 0] / base_vec[0], growth_mat[measure_int + 1, 2] / base_vec[2]
        bond_weight_float = 0.4 * ief_float / (0.6 * spy_float + 0.4 * ief_float)
        count_int = len(row_vec)
        half_int = row_vec[min(15, count_int - entry_dtme_int) - 1]
        first15_int = row_vec[min(15, count_int) - 1]
        month_list.append(month_int)
        pressure_list.append(10_000.0 * (0.4 - bond_weight_float))
        half_list.append(growth_mat[half_int + 1, 0] / base_vec[0] - growth_mat[half_int + 1, 1] / base_vec[1])
        first15_list.append(growth_mat[first15_int + 1, 0] / base_vec[0] - growth_mat[first15_int + 1, 1] / base_vec[1])
    pressure_vec = np.asarray(pressure_list)
    count_vec = np.tril((pressure_vec[None, :] <= pressure_vec[:, None]).astype(float), k=-1).sum(axis=1)  # strictly prior
    prior_vec = np.arange(len(pressure_vec), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        cdf_vec = np.where(prior_vec >= min_prior_int, count_vec / prior_vec, np.nan)
    state_list = [eom.month_state_str(cdf_float, eom.LIVE_CONFIG) for cdf_float in cdf_vec]
    return {"month_list": month_list, "pressure_vec": pressure_vec, "half_vec": np.asarray(half_list),
            "first15_vec": np.asarray(first15_list), "state_list": state_list}


def fast_daily_list(matrix: np.ndarray, date_index: pd.DatetimeIndex, config_list: list) -> list[np.ndarray]:
    """Gross daily return of each configuration from the matrix alone. Month boundaries are the matrix's own."""
    month_row_list = _month_row_list(date_index)
    row_count_int = len(date_index)
    close_mat, overnight_mat = matrix[:, :2], matrix[:, 3:5]
    intraday_mat = (1.0 + close_mat) / (1.0 + overnight_mat) - 1.0
    table_cache_dict, daily_list = {}, []
    for config_obj in config_list:
        config = config_obj if isinstance(config_obj, EomV2Config) else EomV2Config(**config_obj)
        key_tuple = (config.entry_dtme_int, config.min_prior_month_int)
        if key_tuple not in table_cache_dict:
            table_cache_dict[key_tuple] = _signal_table(matrix, month_row_list, config.entry_dtme_int, config.min_prior_month_int)
        signal_dict = table_cache_dict[key_tuple]
        start_int = int(date_index.searchsorted(pd.Timestamp(config.backtest_start_date_str)))
        fill_dict = {}
        for table_int, month_int in enumerate(signal_dict["month_list"]):
            row_vec = month_row_list[month_int]
            state_str, pressure_float = signal_dict["state_list"][table_int], signal_dict["pressure_vec"][table_int]
            next_row_vec = month_row_list[month_int + 1] if month_int + 1 < len(month_row_list) else np.array([], dtype=int)
            final_int, switch_int = row_vec[-config.entry_dtme_int], row_vec[-1]
            if config.fill_str == "open_late":
                switch_int = next_row_vec[0] if len(next_row_vec) else None
            leg_list = [(final_int, final_weight_tuple(config.final_rule_str, state_str, pressure_float, signal_dict["half_vec"][table_int])),
                        (switch_int, early_weight_tuple(config.early_rule_str, state_str, pressure_float, signal_dict["first15_vec"][table_int]))]
            if len(next_row_vec) >= config.exit_session_int:
                leg_list.append((next_row_vec[config.exit_session_int - 1], FLAT_TUPLE))
            for fill_int, weight_tuple in leg_list:
                if fill_int is not None and fill_int >= start_int:
                    fill_dict[int(fill_int)] = weight_tuple
        if not fill_dict:
            daily_list.append(np.zeros(row_count_int))
            continue
        fill_row_vec = np.array(sorted(fill_dict))
        weight_mat = np.array([fill_dict[r] for r in fill_row_vec])
        # The weights set at fill f are held for rows after f; on row f itself the old weights earn up to the fill.
        slot_vec = np.searchsorted(fill_row_vec, np.arange(row_count_int), side="left") - 1
        held_mat = np.where(slot_vec[:, None] >= 0, weight_mat[np.maximum(slot_vec, 0)], 0.0)
        daily_vec = (held_mat * close_mat).sum(axis=1)
        if config.fill_str != "close":
            # *** CRITICAL*** open fill on row f: old weights earn the overnight of f, new weights its intraday.
            old_mat = held_mat[fill_row_vec]
            daily_vec[fill_row_vec] = (old_mat * overnight_mat[fill_row_vec]).sum(axis=1) + (weight_mat * intraday_mat[fill_row_vec]).sum(axis=1)
        daily_list.append(daily_vec)
    return daily_list


# ---------------------------------------------------------------- S3 (the final-week SPY share against the window it trades)
def s3_inputs(inputs: EomV2Inputs | None = None, config: EomV2Config = LIVE_CONFIG, end_date_str: str = "2022-12-30") -> dict:
    """Inputs for `stations.s3_allocation.predictive_tests`, one row per month, indexed by the measure date.

    Score: minus the vote count (known at the measure close), so that a higher score means more bonds. Labels (TR, from
    the close of the final fill to the last close of the month, and from there to the exit close): TLT minus SPY in the
    final week (the premise of the vote), and TLT minus SPY after it (reported, not traded by a flat next-month leg)."""
    inputs = inputs or load_inputs()
    table_df = month_table_df(inputs, config)
    table_df = table_df[table_df["exit_fill_date"] <= pd.Timestamp(end_date_str)]
    close_df = inputs.base.total_return_close_df

    def window_return_df(start_ser: pd.Series, end_ser: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(close_df.reindex(end_ser).to_numpy() / close_df.reindex(start_ser).to_numpy() - 1.0, columns=close_df.columns)

    final_df = window_return_df(table_df["final_fill_date"], table_df["early_fill_date"])
    early_df = window_return_df(table_df["early_fill_date"], table_df["exit_fill_date"])
    index = pd.DatetimeIndex(table_df["measure_date"])
    next_return_df = pd.DataFrame({
        "final_TLT_minus_SPY": (final_df["TLT"] - final_df["SPY"]).to_numpy(),
        "early_TLT_minus_SPY": (early_df["TLT"] - early_df["SPY"]).to_numpy(),
    }, index=index)
    vote_vec = np.array([spy_vote_count_int(r.state_str, r.pressure_bps_float, r.half_spread_float) for r in table_df.itertuples(index=False)], dtype=float)
    score_df = pd.DataFrame({"final_TLT_minus_SPY": -vote_vec, "early_TLT_minus_SPY": -vote_vec}, index=index)
    return {"predictive_tests": {"score_df": score_df, "next_return_df": next_return_df}}
