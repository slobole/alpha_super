"""Scout spec of the month-end rebalancing flow (PM_READY;
`strategies.taa_beyond_6040.strategy_taa_month_end_rebalancing_flow`, "EOM").

An independent re-implementation of the signal; execution is the shared weights engine with its opt-in MOC
(`fill_at_close_bool`), close-and-reopen orders, shorts and borrow. Mapped on 2026-10-01 against main b3855fc, with the
engine lines (strategy_taa_month_end_rebalancing_flow.py) they mirror:

Data (get_month_end_flow_data :179-209): direct Norgate, PaddingType.NONE (observed rows only, never the padded
loader), from 2002-07-26 to today; snapshot mode refused.
    execution   CAPITALSPECIAL Open / Close / Dividend of SPY and TLT (adjusted share units)
    signal      TOTALRETURN Close of SPY and IEF (IEF is signal-only)
    calendar    the union of those series' dates and the $SPX benchmark (read as $SPXTR); scoring from 2003-01-02
    TLT's TOTALRETURN close is loaded for the MCPT matrix and S3 only (the engine does not read it).

Schedule (exchange_session_idx :56-62, build_month_table_df :86-138): XNYS sessions from exchange_calendars through
the end of the month after the last price date (half-days count, historical closures included). For month m with
sessions S_m (months with fewer than 9 sessions are skipped), starting 2002-08:
    measure     S_m[-7] (dtme 7); a month whose measure date is after the last price date is not built
    pressure    R_X = TR_X(measure) / TR_X(last session of m-1) - 1 for X = SPY, IEF;
                bond weight wB = 0.4 (1 + R_IEF) / (0.6 (1 + R_SPY) + 0.4 (1 + R_IEF)); P_m = 10000 (0.4 - wB)
    bucket      fewer than 24 prior months: missing; else F = share of PRIOR months' P <= P_m (strictly prior: P_m is
                appended after) and bucket = min(5, floor(5 F) + 1) (causal_bucket_float :65-71)
    fills       final S_m[-6] (dtme 6), early S_m[-1] (month end), exit S_(m+1)[4] (session 5 of the next month)
Targets (target_weight_tuple :74-83), [SPY, TLT]:
    bucket 1               final (+1, 0)   early (0, 0)
    bucket 2, 3 or missing final (0, +1)   early (0, -1)
    bucket 4, 5            final (0, +1)   early (+0.5, -0.5)
    exit                   (0, 0)
Decision and execution (compute_signals :277-285; iterate :295-327; process_orders :329-351):
    the decision for a fill on XNYS session t is taken at the close of the XNYS session before t (one full session
    after the measure close); shares q = trunc(w x V_T / Close_T) with V_T the total value at the close of T; every
    held position is closed and every non-zero target opened in t's closing auction (separate orders, each with
    2.5 bp slippage and max($1, $0.005 x shares)): weights engine `fill_at_close_bool`, `close_and_reopen_bool`.
    Dividends of T are credited before t's auction on the pre-fill shares (25% withholding on longs, shorts pay in
    full: the engine's dividend cash ledger). Positions are held between auctions (fixed shares, weights drift).
Borrow (apply_post_mark_accounting :353-384): after the close mark, a held TLT short pays
    |shares| x ceil(1.02 x Close_t) x 1% x calendar days to the next XNYS session / 360; none on the run's last session.

*** CRITICAL*** Every signal input is a close at or before the measure date, one full session before the first
fill. TOTALRETURN back-adjustment scales both endpoints of a growth ratio by the same factor, so a later dividend
cannot change a pressure.

Family parameters (`EomConfig`; the default is the engine's configuration):
    entry_dtme_int       the final leg enters at this dtme; the measure is always one session earlier (dtme + 1)
    exit_session_int     the early leg exits at this session of the next month
    cut_tuple            (low, high) cuts on the causal CDF F: low (bucket 1) when F < low, high (buckets 4-5) when
                         F >= high. For the engine's (0.2, 0.6) this equals the quintile formula exactly: F = c / n is
                         a correctly rounded ratio, floor(5 F) = 0 iff F < 0.2 and floor(5 F) >= 3 iff F >= 0.6, also
                         at F = 1/5 and 3/5 (5 x fl(0.2) rounds to 1.0, 5 x fl(0.6) to 3.0).
    decision_offset_int  must be 0: the decision day IS the hypothesis of a calendar-flow rule, so there is no luck
                         band (the family runs one offset); the timing axes above probe the day instead.
    notice_time_calendar_bool  truth mode for deviation `xnys_closure_hindsight` (the schedule of October 2012 is
                         built as if Hurricane Sandy's closures were unknown, which they were at its measure date).

MCPT (`mcpt_matrix`, `fast_daily_list`): columns are the TR daily returns of SPY and TLT (traded), then IEF (signal
only; SPY's TR is already column 0). The replica uses the matrix's own month boundaries, measures, fills at the same
closes as the rule and holds constant weights from the close of the fill row (gross, no costs, no borrow).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alpha.scout.engines.weights import BorrowModel, CostModel, WeightsResult, simulate

STRATEGY_IMPORT_STR = "strategies.taa_beyond_6040.strategy_taa_month_end_rebalancing_flow"
TRADED_TUPLE = ("SPY", "TLT")
SIGNAL_TUPLE = ("SPY", "IEF")
HISTORY_START_STR = "2002-07-26"
PRESSURE_START_PERIOD_STR = "2002-08"  # the first pressure uses the July 2002 month-end close
CALENDAR_SYMBOL_STR = "$SPXTR"  # the engine's $SPX benchmark; it only widens the date index
CALENDAR_START_STR = "2002-07-01"
MIN_MONTH_SESSION_INT = 9
ENGINE_COST_MODEL = CostModel()  # Strategy defaults: 2.5 bp, $0.005 a share, $1 minimum, 25% withholding
# Unscheduled XNYS closures announced after their month's measure date (Hurricane Sandy, announced 2012-10-28 and
# 2012-10-29; the October 2012 measure was 2012-10-19). The other closures since 2002 (2004-06-11, 2007-01-02,
# 2018-12-05, 2025-01-09) were announced before any schedule date they move, or move none.
HINDSIGHT_CLOSURE_TUPLE = ("2012-10-29", "2012-10-30")


@dataclass(frozen=True)
class EomConfig:
    entry_dtme_int: int = 6
    exit_session_int: int = 5
    cut_tuple: tuple = (0.2, 0.6)
    decision_offset_int: int = 0
    # Structural (never grid axes).
    min_prior_month_int: int = 24
    annual_borrow_rate_float: float = 0.01
    backtest_start_date_str: str = "2003-01-02"
    notice_time_calendar_bool: bool = False

    def __post_init__(self):
        if not 2 <= self.entry_dtme_int <= MIN_MONTH_SESSION_INT - 2 or not 1 <= self.exit_session_int <= MIN_MONTH_SESSION_INT:
            raise ValueError("EomConfig: entry_dtme_int in 2..7 and exit_session_int in 1..9.")
        if len(self.cut_tuple) != 2 or not 0.0 < self.cut_tuple[0] <= self.cut_tuple[1] < 1.0:
            raise ValueError("EomConfig: cut_tuple = (low, high) with 0 < low <= high < 1.")
        if self.decision_offset_int != 0:
            raise ValueError("EomConfig: a calendar-flow rule has no luck band (decision_offset_int must be 0).")


LIVE_CONFIG = EomConfig()  # the PM_READY engine configuration


@dataclass(frozen=True)
class EomInputs:
    open_df: pd.DataFrame  # CAPITALSPECIAL SPY, TLT on the engine's index (from 2002-07-26)
    close_df: pd.DataFrame
    dividend_df: pd.DataFrame  # NaN where a symbol has no row (the engine raises on a held one, as `simulate` does)
    total_return_close_df: pd.DataFrame  # TOTALRETURN SPY, IEF (signal), TLT (MCPT / S3 only)
    session_index: pd.DatetimeIndex  # XNYS sessions from 2002-07-01 to the end of the month after the last date
    cache_dict: dict = field(default_factory=dict, repr=False, compare=False)


def xnys_session_index(last_ts: pd.Timestamp) -> pd.DatetimeIndex:
    """XNYS sessions through the end of the month after `last_ts` (a truncated month never defines a month end)."""
    import exchange_calendars

    end_ts = (pd.Timestamp(last_ts).to_period("M") + 1).end_time.normalize()
    return exchange_calendars.get_calendar("XNYS", start=CALENDAR_START_STR, end=end_ts.strftime("%Y-%m-%d")).sessions.tz_localize(None)


def load_inputs(end_date_str: str | None = None) -> EomInputs:
    from data.norgate_loader import is_snapshot_mode_enabled_bool, norgatedata

    if is_snapshot_mode_enabled_bool():
        raise RuntimeError("The EOM spec reads direct Norgate with PaddingType.NONE; snapshot mode is not supported.")

    def observed_df(symbol_str: str, adjustment_str: str) -> pd.DataFrame:
        frame = norgatedata.price_timeseries(
            symbol_str, stock_price_adjustment_setting=getattr(norgatedata.StockPriceAdjustmentType, adjustment_str),
            padding_setting=norgatedata.PaddingType.NONE, start_date=HISTORY_START_STR, end_date=end_date_str,
            timeseriesformat="pandas-dataframe",
        )
        if frame is None or frame.empty:
            raise RuntimeError(f"No observed Norgate prices for {symbol_str}.")
        return frame

    capital_dict = {s: observed_df(s, "CAPITALSPECIAL") for s in TRADED_TUPLE}
    total_return_dict = {s: observed_df(s, "TOTALRETURN")["Close"] for s in ("SPY", "IEF", "TLT")}
    calendar_index = observed_df(CALENDAR_SYMBOL_STR, "TOTALRETURN").index
    # The engine's pd.concat(axis=1) index: SPY, TLT, FLOW_TR_SPY, FLOW_TR_IEF and $SPXTR (not TLT's TR series).
    date_index = pd.DatetimeIndex(sorted(set(calendar_index).union(
        *[frame.index for frame in capital_dict.values()], total_return_dict["SPY"].index, total_return_dict["IEF"].index,
    )))
    session_index = xnys_session_index(date_index[-1])
    scored_index = date_index[date_index >= pd.Timestamp(LIVE_CONFIG.backtest_start_date_str)]
    expected_index = session_index[(session_index >= scored_index[0]) & (session_index <= scored_index[-1])]
    if not scored_index.equals(expected_index):
        raise RuntimeError("The EOM price index is not the XNYS session list (the engine would refuse to move a MOC fill).")

    def field_df(field_str: str) -> pd.DataFrame:
        return pd.DataFrame({s: capital_dict[s][field_str] for s in TRADED_TUPLE}).reindex(date_index).astype(float)

    return EomInputs(
        open_df=field_df("Open"), close_df=field_df("Close"), dividend_df=field_df("Dividend"),
        total_return_close_df=pd.DataFrame(total_return_dict).reindex(date_index).astype(float),
        session_index=session_index,
    )


# ---------------------------------------------------------------- signal
def month_state_str(cdf_float: float, config: EomConfig) -> str:
    if not np.isfinite(cdf_float):
        return "missing"
    low_float, high_float = config.cut_tuple
    if cdf_float < low_float:
        return "low"
    return "high" if cdf_float >= high_float else "mid"


def target_weight_tuple(state_str: str, leg_str: str) -> tuple[float, float]:
    """[SPY, TLT] target of one leg; "missing" (fewer than 24 prior months) trades as the middle buckets."""
    if leg_str == "final":
        return (1.0, 0.0) if state_str == "low" else (0.0, 1.0)
    if leg_str == "early":
        if state_str == "low":
            return (0.0, 0.0)
        return (0.5, -0.5) if state_str == "high" else (0.0, -1.0)
    if leg_str == "exit":
        return (0.0, 0.0)
    raise ValueError(f"Unknown leg: {leg_str}")


def schedule_session_index(inputs: EomInputs, config: EomConfig) -> pd.DatetimeIndex:
    if not config.notice_time_calendar_bool:
        return inputs.session_index
    return inputs.session_index.union(pd.DatetimeIndex(HINDSIGHT_CLOSURE_TUPLE))


def month_table_df(inputs: EomInputs, config: EomConfig = LIVE_CONFIG) -> pd.DataFrame:
    """One row per month: measure and fill dates, pressure (bps), causal CDF F and state."""
    key_tuple = ("month_table", config.entry_dtme_int, config.exit_session_int, config.cut_tuple, config.min_prior_month_int,
                 config.notice_time_calendar_bool)
    if key_tuple in inputs.cache_dict:
        return inputs.cache_dict[key_tuple]
    total_return_df = inputs.total_return_close_df
    last_ts = total_return_df.index[-1]
    session_index = schedule_session_index(inputs, config)
    month_period_index = session_index.to_period("M")
    prior_pressure_list: list[float] = []
    row_list = []
    for month_period in pd.period_range(PRESSURE_START_PERIOD_STR, last_ts.to_period("M"), freq="M"):
        month_session_index = session_index[month_period_index == month_period]
        if len(month_session_index) < MIN_MONTH_SESSION_INT:
            continue
        # *** CRITICAL*** dtme counts XNYS sessions from the month's end, never a price-truncated month.
        measure_ts = month_session_index[-(config.entry_dtme_int + 1)]
        if measure_ts > last_ts:
            continue
        previous_close_ts = session_index[month_period_index == month_period - 1][-1]
        endpoint_mat = total_return_df.reindex(index=[previous_close_ts, measure_ts], columns=list(SIGNAL_TUPLE)).to_numpy(dtype=float)
        if not np.isfinite(endpoint_mat).all() or (endpoint_mat <= 0.0).any():
            raise ValueError(f"Missing or invalid TOTALRETURN signal endpoint in {month_period}.")
        spy_growth_float = float(endpoint_mat[1, 0] / endpoint_mat[0, 0])
        ief_growth_float = float(endpoint_mat[1, 1] / endpoint_mat[0, 1])
        bond_weight_float = 0.4 * ief_growth_float / (0.6 * spy_growth_float + 0.4 * ief_growth_float)
        pressure_float = 10_000.0 * (0.4 - bond_weight_float)
        # *** CRITICAL*** strictly prior months: the current pressure joins the history only after its own CDF.
        if len(prior_pressure_list) < config.min_prior_month_int:
            cdf_float = float("nan")
        else:
            cdf_float = float(np.mean(np.asarray(prior_pressure_list) <= pressure_float))
        next_session_index = session_index[month_period_index == month_period + 1]
        row_list.append({
            "month_period": month_period, "measure_date": measure_ts, "pressure_bps_float": pressure_float,
            "cdf_float": cdf_float, "state_str": month_state_str(cdf_float, config),
            "final_fill_date": month_session_index[-config.entry_dtme_int], "early_fill_date": month_session_index[-1],
            "exit_fill_date": next_session_index[config.exit_session_int - 1],
        })
        prior_pressure_list.append(pressure_float)
    table_df = pd.DataFrame(row_list)
    inputs.cache_dict[key_tuple] = table_df
    return table_df


def rebalance_weight_df(inputs: EomInputs, config: EomConfig = LIVE_CONFIG) -> pd.DataFrame:
    """[SPY, TLT] target weights indexed by fill (closing-auction) session."""
    data_index = inputs.close_df.index
    actual_session_index = inputs.session_index
    row_dict = {}
    for month_row in month_table_df(inputs, config).itertuples(index=False):
        for leg_str in ("final", "early", "exit"):
            fill_ts = getattr(month_row, f"{leg_str}_fill_date")
            # *** CRITICAL*** the decision is the close of the XNYS session before the fill (compute_signals :280).
            decision_ts = actual_session_index[actual_session_index.get_loc(fill_ts) - 1]
            if decision_ts not in data_index:
                continue
            if month_row.measure_date > decision_ts:
                raise RuntimeError("The MOC signal is not known a full session before its fill.")
            decision_int = int(data_index.get_loc(decision_ts))
            if decision_int + 1 >= len(data_index):
                continue  # the fill session is beyond the data: the engine never reaches it
            if data_index[decision_int + 1] != fill_ts:
                raise RuntimeError(f"Missing XNYS session before {fill_ts.date()}; the engine refuses to move a MOC fill.")
            if fill_ts in row_dict:
                raise RuntimeError(f"Two legs share the fill session {fill_ts.date()}.")
            row_dict[fill_ts] = target_weight_tuple(month_row.state_str, leg_str)
    return pd.DataFrame.from_dict(row_dict, orient="index", columns=list(TRADED_TUPLE)).sort_index()


def simulate_config(inputs: EomInputs, config: EomConfig = LIVE_CONFIG, cost_model: CostModel = ENGINE_COST_MODEL,
                    capital_float: float = 100_000.0) -> WeightsResult:
    """The parity weights-engine run of one configuration (what the gate and the family execute)."""
    traded_list = list(TRADED_TUPLE)
    return simulate(
        inputs.open_df[traded_list], inputs.close_df[traded_list], inputs.dividend_df[traded_list], rebalance_weight_df(inputs, config),
        start_date=config.backtest_start_date_str, capital_float=capital_float, share_unit_mode_str="adjusted",
        cost_model=cost_model, allow_short_bool=True, borrow_model=BorrowModel(annual_rate_float=config.annual_borrow_rate_float),
        fill_at_close_bool=True, close_and_reopen_bool=True,
    )


# ---------------------------------------------------------------- MCPT (fast replica)
def mcpt_matrix(inputs: EomInputs, end_date_str: str | None = "2022-12-30") -> tuple[pd.DatetimeIndex, np.ndarray]:
    """(date_index, matrix): TR daily returns of SPY, TLT (traded) and IEF (signal), from the first session after the
    July 2002 month end (the first pressure's base) to `end_date_str` (default the vault seal; None = all data)."""
    close_df = inputs.total_return_close_df[["SPY", "TLT", "IEF"]]
    base_ts = inputs.session_index[inputs.session_index.to_period("M") == pd.Period(PRESSURE_START_PERIOD_STR) - 1][-1]
    date_index = close_df.index[close_df.index > base_ts]
    if end_date_str is not None:
        date_index = date_index[date_index <= pd.Timestamp(end_date_str)]
    # *** CRITICAL*** return of day t = close_t / close_(t-1) - 1 (observed rows only: PaddingType.NONE).
    return_df = close_df.loc[base_ts:].pct_change(fill_method=None).reindex(date_index)
    if not np.isfinite(return_df.to_numpy()).all():
        raise RuntimeError("The EOM MCPT matrix has a missing TR return.")
    return date_index, return_df.to_numpy(dtype=float)


def _month_row_list(date_index: pd.DatetimeIndex) -> list[np.ndarray]:
    position_vec = np.arange(len(date_index))
    period_vec = date_index.to_period("M")
    return [position_vec[period_vec == p] for p in period_vec.unique()]


def fast_daily_list(matrix: np.ndarray, date_index: pd.DatetimeIndex, config_list: list) -> list[np.ndarray]:
    """Gross daily return of each configuration (dicts of EomConfig fields, or EomConfig) from the matrix alone.

    Month boundaries are the matrix's own (a row shuffle keeps the calendar and breaks the return-calendar link, as
    intended); the pressure compounds the SPY and IEF returns since the previous month's last row (the session before
    the first row for the first month), the CDF uses strictly prior months, and each leg's weights are entered at the
    close of its fill row and held from the next row's return. Fills before the backtest start are not traded (the
    account starts flat, as the engine's)."""
    price_mat = np.vstack([np.ones((1, 2)), np.cumprod(1.0 + matrix[:, [0, 2]], axis=0)])  # row r + 1 = close of row r
    month_row_list = _month_row_list(date_index)
    row_count_int = len(date_index)
    pressure_cache_dict, daily_list = {}, []
    for config_obj in config_list:
        config = config_obj if isinstance(config_obj, EomConfig) else EomConfig(**config_obj)
        key_tuple = (config.entry_dtme_int, config.min_prior_month_int)
        if key_tuple not in pressure_cache_dict:
            month_list, pressure_list = [], []
            for month_int, row_vec in enumerate(month_row_list):
                if len(row_vec) < MIN_MONTH_SESSION_INT:
                    continue
                previous_int = month_row_list[month_int - 1][-1] if month_int > 0 else -1
                growth_vec = price_mat[row_vec[-(config.entry_dtme_int + 1)] + 1] / price_mat[previous_int + 1]
                bond_weight_float = 0.4 * growth_vec[1] / (0.6 * growth_vec[0] + 0.4 * growth_vec[1])
                month_list.append(month_int)
                pressure_list.append(10_000.0 * (0.4 - bond_weight_float))
            pressure_vec = np.asarray(pressure_list)
            # F_i = share of months j < i with P_j <= P_i
            count_vec = np.tril((pressure_vec[None, :] <= pressure_vec[:, None]).astype(float), k=-1).sum(axis=1)
            prior_vec = np.arange(len(pressure_vec), dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                cdf_vec = np.where(prior_vec >= config.min_prior_month_int, count_vec / prior_vec, np.nan)
            pressure_cache_dict[key_tuple] = (month_list, cdf_vec)
        month_list, cdf_vec = pressure_cache_dict[key_tuple]
        start_int = int(date_index.searchsorted(pd.Timestamp(config.backtest_start_date_str)))
        fill_dict = {}
        for month_int, cdf_float in zip(month_list, cdf_vec, strict=True):
            row_vec, state_str = month_row_list[month_int], month_state_str(cdf_float, config)
            leg_list = [("final", row_vec[-config.entry_dtme_int]), ("early", row_vec[-1])]
            if month_int + 1 < len(month_row_list) and len(month_row_list[month_int + 1]) >= config.exit_session_int:
                leg_list.append(("exit", month_row_list[month_int + 1][config.exit_session_int - 1]))
            for leg_str, fill_int in leg_list:
                if fill_int >= start_int:
                    fill_dict[int(fill_int)] = target_weight_tuple(state_str, leg_str)
        if not fill_dict:
            daily_list.append(np.zeros(row_count_int))
            continue
        fill_row_vec = np.array(sorted(fill_dict))
        weight_mat = np.array([fill_dict[r] for r in fill_row_vec])
        # *** CRITICAL*** MOC: weights filled at the close of row f earn from row f + 1.
        slot_vec = np.searchsorted(fill_row_vec, np.arange(row_count_int), side="left") - 1
        held_mat = np.where(slot_vec[:, None] >= 0, weight_mat[np.maximum(slot_vec, 0)], 0.0)
        daily_list.append((held_mat * matrix[:, :2]).sum(axis=1))
    return daily_list


# ---------------------------------------------------------------- S3 (calendar flow: the pressure CDF vs the next windows)
def s3_inputs(inputs: EomInputs | None = None, config: EomConfig = LIVE_CONFIG, end_date_str: str = "2022-12-30") -> dict:
    """Inputs for `stations.s3_allocation.predictive_tests`, one row per month (indexed by the measure date).

    Score: the causal pressure CDF F (known at the measure close), the same number in both columns. Labels (TR,
    from the fill closes; labels only): "final_TLT_minus_SPY" = TLT minus SPY from the final-leg fill close to the
    month-end close (the rule's premise: a high F, stocks having outrun bonds, means rebalancers buy bonds into the
    month end); "early_SPY_minus_TLT" = SPY minus TLT from the month-end close to the exit close (the reversal). The
    per-column (time-series) slopes are this rule's hypothesis, both expected positive; with two columns the
    cross-sectional slope needs three assets and the on/off spread compares columns, so neither applies."""
    inputs = inputs or load_inputs()
    table_df = month_table_df(inputs, config)
    table_df = table_df[table_df["exit_fill_date"] <= pd.Timestamp(end_date_str)]
    close_df = inputs.total_return_close_df

    def window_return_df(start_ser: pd.Series, end_ser: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(close_df.reindex(end_ser).to_numpy() / close_df.reindex(start_ser).to_numpy() - 1.0, columns=close_df.columns)

    final_df = window_return_df(table_df["final_fill_date"], table_df["early_fill_date"])
    early_df = window_return_df(table_df["early_fill_date"], table_df["exit_fill_date"])
    index = pd.DatetimeIndex(table_df["measure_date"])
    next_return_df = pd.DataFrame({
        "final_TLT_minus_SPY": (final_df["TLT"] - final_df["SPY"]).to_numpy(),
        "early_SPY_minus_TLT": (early_df["SPY"] - early_df["TLT"]).to_numpy(),
    }, index=index)
    cdf_vec = table_df["cdf_float"].to_numpy()
    score_df = pd.DataFrame({"final_TLT_minus_SPY": cdf_vec, "early_SPY_minus_TLT": cdf_vec}, index=index)
    return {"predictive_tests": {"score_df": score_df, "next_return_df": next_return_df}}
