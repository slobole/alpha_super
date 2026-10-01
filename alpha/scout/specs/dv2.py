"""Scout spec of the LIVE DV2 mean-reversion pod (WIRED; `strategies.dv2.strategy_mr_dv2:DVO2Strategy`) on point-in-time
S&P 500 members.

An independent re-implementation of the signal; execution is the shared weights engine with the path-dependent
`decision_fn` hook and the opt-in `hold_nan_bool` (a held stock is never resized until its exit). Whole shares in
adjusted units (`share_unit_mode_str="adjusted"`, the engine's default `historical_share_units_bool = False`). No engine
change was needed. Mapped on 2026-10-01 against main 8fd3c37: strategies/dv2/strategy_mr_dv2.py (lines "S:"),
alpha/engine/strategy.py (process_orders, order_value, _get_order_sizing_price_float, _liquidate_missing_price_positions,
_credit_dividend_cash_before_open; lines "E:"), alpha/engine/order.py (amount_in_shares; "O:"), backtester.py ("B:").

Data (S: run_variant :277-279)
    universe_df     build_index_constituent_matrix("S&P 500"): Norgate's exact constituent flag per session (no tail
                    trim), columns = every "S&P 500 Current & Past" symbol that was ever a member
    prices          load_raw_prices(every watchlist symbol, ["$SPX"], 1998-01-01 .. today): CAPITALSPECIAL Open, High,
                    Low, Close, Volume, Turnover, Unadjusted Close, Dividend; $SPX (read as $SPXTR) only widens the index

Features at the close of T, per stock, on the full session index (S: compute_signals :165-197)
    p126d_return    Close_T / Close_(T-126) - 1 (positional shift on the session index), computed in the PRICE DTYPE:
                    Norgate returns float32 and pandas keeps it, so the ratio rounds in float32 (MCIP on 2005-02-11:
                    exactly 0.25 in float32, 0.2500000368 in float64; the Nasdaq-100 rule's "> 0.25" turns on it). The
                    other features are float64 from exact upcasts (pandas rolling, talib and the DV2 kernel cast).
    natr            TA-Lib NATR(High, Low, Close, 14): from the first row with all three finite, first ATR = mean of the
                    first 14 true ranges, then Wilder (ATR x 13 + TR) / 14; NATR = (ATR / Close) x 100. A NaN inside the
                    history propagates (re-implemented in numba; tests check it against talib bit for bit)
    dv2             alpha.indicators.dv2_indicator(Close, High, Low, 126) = alpha.engine.dv2_indicator_fast:
                    DV1 = Close / ((High + Low) / 2) - 1, DV = (DV1_(T-1) + DV1_T) / 2, DV2 = 100 x (average rank of
                    DV_T in the trailing 126 DV values) / 126, NaN unless the whole window is finite (numba replica)
    sma_200         Close.rolling(200).mean() (pandas, the engine's own call)

Decision after the close of T (S: iterate :199-239, get_opportunities :241-266; B: iterate runs before process_orders)
    held            stocks with shares > 0 at the close of T
    exits           held stocks with Close_T > High_(T-1) (the previous row of the session index, `.iloc[-2]`; NaN -> no
                    exit) -> target 0 shares
    slots           10 - |held| + |exits|
    candidates      `close.unstack().dropna()`: every raw field and every feature of the stock finite at T; then DV2 < 10,
                    Close > SMA200, p126d_return > 0.05; sorted by NATR descending with pandas `sort_values` (quicksort,
                    reproduced by the same pandas call on the same symbol-sorted rows); then kept only if a member in the
                    last universe row dated <= T (`get_asof_universe_symbol_list`, S :79-97)
    entries         walk the ranked list; a stock with a position (held, including one exiting today) is skipped without
                    using a slot; the first `slots` flat stocks enter
    sizing          order_value(V_T / 10): shares = int(V_T / 10 / Close_T) (O: amount_in_shares; E: sizing price =
                    Close of previous_bar); Scout computes int(V_T x 0.1 / Close_T), which differs only if the quotient
                    sits within one ulp of an integer (not seen; the gate would show it). Zero shares -> no order.
    fills           Open_(T+1) x (1 +- 2.5 bp), fee max(1, 0.005 x shares) (S: slippage 0.00025, commission 0.005 per
                    share, minimum 1.0), no cash check; an order without an Open is cancelled
    dividends       shares at the close of T x Dividend_T, net of 25% withholding, credited before the T+1 open
    delisting       a held stock without Open or Close at t is sold at its last close <= T, with the fee (E :1164-1259)
Calendar            the first decision is the close of the session before 2004-01-01 (S: calendar :297; B: previous_bar)

*** CRITICAL*** every feature at T reads bars up to T only; membership is the row dated <= T; fills at Open(T+1).

Family parameters (`Dv2Config`; the default is the engine's configuration and the identity gate runs on it):
    entry_dv2_max_float, exit_rule_str ("prev_close": Close_T > Close_(T-1); "prev_high": the live Close_T > High_(T-1);
    "high_2d": Close_T > both High_(T-1) and High_(T-2)), max_positions_int (sizing V_T / max_positions_int), and the
    windows dv2_length_int, trend_sma_int, momentum_lookback_int, momentum_min_float, natr_length_int.
    decision_offset_int must be 0: a daily event rule has no rebalance schedule (the family runs offset_count_int = 1).

Variants (`VARIANT_DICT`; `load_inputs`, `s3_inputs` and the families take the variant name):
    "dv2"       the WIRED pod above (S&P 500).
    "dv2_ndx"   Nasdaq-100 members, the last committed rule of strategies.dv2.strategy_mr_dv2_nasdaq100 (300ca70):
                DV2 < 20, 126-session return > 25%, slippage 1 bp; everything else as above. That module is an empty
                self-importing stub since bf1a334, so the gate runs the restored code (alpha/scout/gate/legacy_dv2_ndx.py)
                on the real engine. It reads the universe row dated exactly T (`universe_df.loc[previous_bar]`), which
                equals the as-of row whenever the row exists (it raises otherwise; the gate would show it).

MCPT (`fast_daily_list_panel`): the same rule on a `alpha.scout.panel.Panel` alone (its Open/High/Low/Close and the
exact membership flag of T), every feature recomputed, on a gross fractional-share ledger (no costs, no dividends),
plus the equal-weight member baseline (members at T, held from the close of T+1, as alpha.scout.searches). A draw of
`alpha.scout.null.permuted_panel` is therefore a valid null. The momentum ratio runs in the panel's dtype (float32 on
the cached panels, as the engine). Checked 2026-10-01 against the engine family (gross, all 27 configurations, to
2022-12-30): the sealed panels hold exactly the loader's members, membership cells and closes from 2004; daily
correlation min 0.998 (S&P 500) / 0.999 (Nasdaq-100), Spearman of configuration Sharpes 0.997 / 0.998; the replica
Sharpe sits about 0.01-0.04 lower (no dividends). About 2 s per draw for the 27-configuration grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numba import njit

from alpha.scout.engines.weights import CostModel, WeightsResult, simulate

STRATEGY_IMPORT_STR = "strategies.dv2.strategy_mr_dv2"
INDEX_NAME_STR = "S&P 500"
BENCHMARK_STR = "$SPX"
HISTORY_START_STR = "1998-01-01"
BACKTEST_START_STR = "2004-01-01"
ENGINE_COST_MODEL = CostModel(slippage_float=0.00025, fee_per_share_float=0.005, min_fee_float=1.0)  # S: run_variant :281-289
SEAL_END_STR = "2022-12-30"
EXIT_RULE_TUPLE = ("prev_close", "prev_high", "high_2d")  # ordered by the rebound the exit asks for


@dataclass(frozen=True)
class Dv2Config:
    entry_dv2_max_float: float = 10.0
    exit_rule_str: str = "prev_high"
    max_positions_int: int = 10
    dv2_length_int: int = 126
    trend_sma_int: int = 200
    momentum_lookback_int: int = 126
    momentum_min_float: float = 0.05
    natr_length_int: int = 14
    decision_offset_int: int = 0

    def __post_init__(self):
        if self.exit_rule_str not in EXIT_RULE_TUPLE:
            raise ValueError(f"Dv2Config: exit_rule_str must be one of {EXIT_RULE_TUPLE}.")
        if not 0.0 < self.entry_dv2_max_float <= 100.0 or self.max_positions_int < 1:
            raise ValueError("Dv2Config: 0 < entry DV2 threshold <= 100 and max_positions_int >= 1.")
        if min(self.dv2_length_int, self.trend_sma_int, self.momentum_lookback_int, self.natr_length_int) < 2:
            raise ValueError("Dv2Config: every window >= 2.")
        if self.decision_offset_int != 0:
            raise ValueError("Dv2Config: a daily event rule has no rebalance offset (decision_offset_int = 0).")

    @property
    def feature_key_tuple(self) -> tuple:
        return (self.dv2_length_int, self.trend_sma_int, self.momentum_lookback_int, self.natr_length_int)


LIVE_CONFIG = Dv2Config()  # the WIRED engine configuration (S: max_positions = 10, get_opportunities filters)


@dataclass(frozen=True)
class Dv2Variant:
    index_name_str: str  # the Norgate index of the universe and of the Scout panel
    config: Dv2Config
    cost_model: CostModel
    strategy_import_str: str  # the engine module the identity gate runs
    note_str: str = ""


VARIANT_DICT = {
    "dv2": Dv2Variant("S&P 500", LIVE_CONFIG, ENGINE_COST_MODEL, STRATEGY_IMPORT_STR, "WIRED live pod"),
    # strategies/dv2/strategy_mr_dv2_nasdaq100.py is an empty self-importing stub since bf1a334; its last committed rule
    # (300ca70) is restored verbatim as the gate's engine side in alpha/scout/gate/legacy_dv2_ndx.py: DV2 < 20,
    # 126-session return > 25%, 1 bp slippage, membership = the universe row dated exactly T (same as as-of when present).
    "dv2_ndx": Dv2Variant(
        "Nasdaq 100", Dv2Config(entry_dv2_max_float=20.0, momentum_min_float=0.25),
        CostModel(slippage_float=0.0001, fee_per_share_float=0.005, min_fee_float=1.0), "alpha.scout.gate.legacy_dv2_ndx",
        "research variant; engine module is a stub, gated against its restored 300ca70 code",
    ),
}


@dataclass(frozen=True)
class Dv2Inputs:
    """The engine's data on its session index (union of every loaded series), symbols sorted, $SPX dropped."""

    open_df: pd.DataFrame
    high_df: pd.DataFrame
    low_df: pd.DataFrame
    close_df: pd.DataFrame
    dividend_df: pd.DataFrame  # NaN kept: a NaN dividend on a held stock raises, as in the engine
    complete_df: pd.DataFrame  # every raw field finite at T (the raw half of `close.unstack().dropna()`)
    member_df: pd.DataFrame  # bool: member in the last universe row dated <= T
    backtest_start_str: str = BACKTEST_START_STR
    cache_dict: dict = field(default_factory=dict, compare=False)  # features per window tuple


def load_inputs(end_date_str: str | None = None, variant_name_str: str = "dv2") -> Dv2Inputs:
    from data.norgate_loader import build_index_constituent_matrix, load_raw_prices

    index_name_str = VARIANT_DICT[variant_name_str].index_name_str
    symbol_list, universe_df = build_index_constituent_matrix(indexname=index_name_str)  # S: run_variant :278
    pricing_df = load_raw_prices(symbol_list, [BENCHMARK_STR], start_date=HISTORY_START_STR, end_date=end_date_str)
    return inputs_from_frames(pricing_df, universe_df)


def inputs_from_frames(pricing_df: pd.DataFrame, universe_df: pd.DataFrame, backtest_start_str: str = BACKTEST_START_STR) -> Dv2Inputs:
    """Inputs from a (symbol, field) price frame (load_raw_prices layout) and a 1/0 universe frame (date x symbol)."""
    pricing_df = pricing_df.sort_index()
    session_index = pd.DatetimeIndex(pricing_df.index)
    stock_list = sorted({str(s) for s, _ in pricing_df.columns if not str(s).startswith("$")})
    field_list = sorted({f for _, f in pricing_df.columns})

    def field_df(field_str: str) -> pd.DataFrame:
        # *** CRITICAL*** keep the loader's dtype (Norgate float32): the engine's p126d_return is float32 arithmetic.
        return pd.DataFrame({s: pd.to_numeric(pricing_df[(s, field_str)], errors="coerce") if (s, field_str) in pricing_df.columns
                             else pd.Series(np.nan, index=session_index) for s in stock_list}, index=session_index)

    # `close.unstack().dropna()` (S :246) drops a stock whose row has ANY NaN field: the raw fields here, the features in
    # `qualify_mat`. A field a stock lacks altogether is NaN for it.
    complete_mat = np.ones((len(session_index), len(stock_list)), dtype=bool)
    frame_dict = {}
    for field_str in field_list:
        frame = field_df(field_str)
        complete_mat &= frame.notna().to_numpy()
        if field_str in ("Open", "High", "Low", "Close", "Dividend"):
            frame_dict[field_str] = frame
    # *** CRITICAL*** membership as of T: the last universe row dated <= T (S :79-97), never a later row.
    universe_sorted_df = universe_df.sort_index()
    member_df = (universe_sorted_df.reindex(columns=stock_list).fillna(0).reindex(session_index, method="ffill").fillna(0) == 1)
    return Dv2Inputs(open_df=frame_dict["Open"], high_df=frame_dict["High"], low_df=frame_dict["Low"], close_df=frame_dict["Close"],
                     dividend_df=frame_dict["Dividend"], complete_df=pd.DataFrame(complete_mat, index=session_index, columns=stock_list),
                     member_df=member_df, backtest_start_str=backtest_start_str)


# ---------------------------------------------------------------- indicators (numba re-implementations)
@njit(cache=False)
def dv2_mat(close_mat, high_mat, low_mat, length_int, rank_mask_mat=None):
    """alpha.engine.dv2_indicator_fast per column, the same float operations in the same order. `rank_mask_mat`
    (replica speed only): the percent rank is computed only where it is True (NaN elsewhere)."""
    row_count_int, column_count_int = close_mat.shape
    out_mat = np.full((row_count_int, column_count_int), np.nan)
    dv1_vec = np.empty(row_count_int)
    dv_vec = np.empty(row_count_int)
    for c_int in range(column_count_int):
        for i_int in range(row_count_int):
            dv1_vec[i_int] = np.nan
            dv_vec[i_int] = np.nan
            close_float, high_float, low_float = close_mat[i_int, c_int], high_mat[i_int, c_int], low_mat[i_int, c_int]
            if np.isnan(close_float) or np.isnan(high_float) or np.isnan(low_float):
                continue
            hl2_float = (high_float + low_float) / 2.0
            if np.isnan(hl2_float) or hl2_float == 0.0:
                continue
            dv1_vec[i_int] = (close_float / hl2_float) - 1.0
        for i_int in range(1, row_count_int):
            if np.isnan(dv1_vec[i_int - 1]) or np.isnan(dv1_vec[i_int]):
                continue
            dv_vec[i_int] = (dv1_vec[i_int - 1] + dv1_vec[i_int]) / 2.0
        for end_int in range(length_int - 1, row_count_int):
            last_float = dv_vec[end_int]
            if np.isnan(last_float) or (rank_mask_mat is not None and not rank_mask_mat[end_int, c_int]):
                continue
            less_int, equal_int, nan_bool = 0, 0, False
            for k_int in range(end_int - length_int + 1, end_int + 1):
                value_float = dv_vec[k_int]
                if np.isnan(value_float):
                    nan_bool = True
                    break
                if value_float < last_float:
                    less_int += 1
                elif value_float == last_float:
                    equal_int += 1
            if not nan_bool:
                out_mat[end_int, c_int] = ((less_int + ((equal_int + 1.0) / 2.0)) / length_int) * 100.0
    return out_mat


@njit(cache=False)
def _true_range_float(high_float, low_float, previous_close_float):
    """TA-Lib TRUE_RANGE: comparisons with NaN are false, so a NaN previous close leaves High - Low."""
    out_float = high_float - low_float
    candidate_float = abs(high_float - previous_close_float)
    if candidate_float > out_float:  # noqa: PLR1730 - the C macro's NaN semantics, kept explicit for numba
        out_float = candidate_float
    candidate_float = abs(low_float - previous_close_float)
    if candidate_float > out_float:  # noqa: PLR1730 - the C macro's NaN semantics, kept explicit for numba
        out_float = candidate_float
    return out_float


@njit(cache=False)
def natr_mat(high_mat, low_mat, close_mat, period_int):
    """talib.NATR per column: start at the first row with High, Low and Close all non-NaN (the wrapper's begidx); first
    ATR = sequential sum of the first `period` true ranges / period; Wilder after; NATR = (ATR / Close) x 100 (0 when
    |Close| < 1e-14). Later NaNs propagate (no skipping)."""
    row_count_int, column_count_int = close_mat.shape
    out_mat = np.full((row_count_int, column_count_int), np.nan)
    for c_int in range(column_count_int):
        begin_int = -1
        for i_int in range(row_count_int):
            if not (np.isnan(high_mat[i_int, c_int]) or np.isnan(low_mat[i_int, c_int]) or np.isnan(close_mat[i_int, c_int])):
                begin_int = i_int
                break
        if begin_int < 0 or row_count_int - begin_int <= period_int:
            continue
        sum_float = 0.0
        for i_int in range(begin_int + 1, begin_int + period_int + 1):
            sum_float += _true_range_float(high_mat[i_int, c_int], low_mat[i_int, c_int], close_mat[i_int - 1, c_int])
        atr_float = sum_float / period_int
        first_int = begin_int + period_int
        for i_int in range(first_int, row_count_int):
            if i_int > first_int:
                tr_float = _true_range_float(high_mat[i_int, c_int], low_mat[i_int, c_int], close_mat[i_int - 1, c_int])
                atr_float = atr_float * (period_int - 1)
                atr_float = atr_float + tr_float
                atr_float = atr_float / period_int
            close_float = close_mat[i_int, c_int]
            out_mat[i_int, c_int] = (atr_float / close_float) * 100.0 if not (-1e-14 < close_float < 1e-14) else 0.0
    return out_mat


def _momentum_df(close_df: pd.DataFrame, lookback_int: int) -> pd.DataFrame:
    """p126d_return (S :188) in the frame's own dtype: pandas keeps float32 / float32 - 1.0 in float32."""
    return close_df / close_df.shift(lookback_int) - 1.0


def feature_dict(close_df: pd.DataFrame, high_df: pd.DataFrame, low_df: pd.DataFrame, config: Dv2Config = LIVE_CONFIG) -> dict:
    """The four engine features as (date x symbol) float64 matrices on the frames' index (S :188-191). Pass the frames
    in the loader's dtype: the momentum ratio is computed in it (float32 for Norgate, as the engine)."""
    close_mat, high_mat, low_mat = (frame.to_numpy(dtype=float) for frame in (close_df, high_df, low_df))
    with np.errstate(divide="ignore", invalid="ignore"):
        momentum_mat = _momentum_df(close_df, config.momentum_lookback_int).to_numpy(dtype=float)
    return {
        "momentum_mat": momentum_mat,
        "natr_mat": natr_mat(high_mat, low_mat, close_mat, config.natr_length_int),
        "dv2_mat": dv2_mat(close_mat, high_mat, low_mat, config.dv2_length_int),
        "sma_mat": close_df.rolling(config.trend_sma_int).mean().to_numpy(dtype=float),
    }


def _features(inputs: Dv2Inputs, config: Dv2Config) -> dict:
    key_tuple = config.feature_key_tuple
    if key_tuple not in inputs.cache_dict:
        inputs.cache_dict[key_tuple] = feature_dict(inputs.close_df, inputs.high_df, inputs.low_df, config)
    return inputs.cache_dict[key_tuple]


def exit_mat(close_mat: np.ndarray, high_mat: np.ndarray, exit_rule_str: str) -> np.ndarray:
    """Exit signal at T (S :213-218 for "prev_high"); a NaN comparison is no exit."""
    previous_close_mat = np.vstack([np.full(close_mat.shape[1], np.nan), close_mat[:-1]])
    previous_high_mat = np.vstack([np.full(high_mat.shape[1], np.nan), high_mat[:-1]])
    with np.errstate(invalid="ignore"):
        if exit_rule_str == "prev_close":
            return close_mat > previous_close_mat
        if exit_rule_str == "prev_high":
            return close_mat > previous_high_mat
        second_high_mat = np.vstack([np.full((2, high_mat.shape[1]), np.nan), high_mat[:-2]])
        return (close_mat > previous_high_mat) & (close_mat > second_high_mat)


def qualify_mat(features: dict, close_mat: np.ndarray, complete_mat: np.ndarray, config: Dv2Config) -> np.ndarray:
    """Entry filter at T before membership and slots (S :246-260): a complete row and the three conditions."""
    with np.errstate(invalid="ignore"):
        ok_mat = complete_mat & np.isfinite(close_mat)
        for name_str in ("momentum_mat", "natr_mat", "dv2_mat", "sma_mat"):
            ok_mat &= ~np.isnan(features[name_str])  # dropna drops NaN only (an inf survives it)
        return (ok_mat & (features["dv2_mat"] < config.entry_dv2_max_float) & (close_mat > features["sma_mat"])
                & (features["momentum_mat"] > config.momentum_min_float))


# ---------------------------------------------------------------- the event rule as a weights-engine hook
def dv2_decision_fn(qualify: np.ndarray, member: np.ndarray, exit_signal: np.ndarray, natr: np.ndarray, max_positions_int: int):
    """Exits before entries; candidates ranked by NATR descending over every qualifying stock (members or not, the
    engine's sort), then filtered to members; held stocks skipped without a slot. NaN = untouched."""
    asset_count_int = qualify.shape[1]
    entry_weight_float = 1.0 / max_positions_int
    qualify_row_list = [np.flatnonzero(row) for row in qualify]

    def decide(t_idx_int: int, position_vec: np.ndarray, total_float: float):
        p_int = t_idx_int - 1  # T
        held_vec = position_vec > 0.0
        exit_vec = held_vec & exit_signal[p_int]
        slot_int = max_positions_int - int(held_vec.sum()) + int(exit_vec.sum())
        entry_list = []
        candidate_vec = qualify_row_list[p_int]
        if slot_int > 0 and candidate_vec.size:
            # the engine's own call on the same symbol-sorted rows: DataFrame.sort_values('natr', ascending=False)
            order_vec = pd.Series(natr[p_int, candidate_vec]).sort_values(ascending=False).index.to_numpy()
            for asset_int in candidate_vec[order_vec]:
                if not member[p_int, asset_int] or position_vec[asset_int] != 0.0:
                    continue
                entry_list.append(asset_int)
                if len(entry_list) == slot_int:
                    break
        if not exit_vec.any() and not entry_list:
            return None
        weight_vec = np.full(asset_count_int, np.nan)
        weight_vec[exit_vec] = 0.0
        weight_vec[entry_list] = entry_weight_float
        return weight_vec

    return decide


def signal_mats(inputs: Dv2Inputs, config: Dv2Config = LIVE_CONFIG) -> dict:
    features = _features(inputs, config)
    close_mat = inputs.close_df.to_numpy(dtype=float)
    return {
        "qualify": qualify_mat(features, close_mat, inputs.complete_df.to_numpy(), config),
        "member": inputs.member_df.to_numpy(),
        "exit_signal": exit_mat(close_mat, inputs.high_df.to_numpy(dtype=float), config.exit_rule_str),
        "natr": features["natr_mat"],
    }


def simulate_config(inputs: Dv2Inputs, config: Dv2Config = LIVE_CONFIG, cost_model: CostModel = ENGINE_COST_MODEL,
                    capital_float: float = 100_000.0) -> WeightsResult:
    mats = signal_mats(inputs, config)
    empty_df = pd.DataFrame(columns=list(inputs.close_df.columns), dtype=float)
    return simulate(
        inputs.open_df, inputs.close_df, inputs.dividend_df, empty_df, start_date=inputs.backtest_start_str, capital_float=capital_float,
        share_unit_mode_str="adjusted", cost_model=cost_model, hold_nan_bool=True,
        decision_fn=dv2_decision_fn(mats["qualify"], mats["member"], mats["exit_signal"], mats["natr"], config.max_positions_int),
    )


# ---------------------------------------------------------------- MCPT replica on a point-in-time panel (S5, per-asset null)
@njit(cache=False)
def _dv2_book_daily(open_mat, close_mat, pointer_vec, candidate_vec, exit_signal_mat, max_positions_int):
    """Gross fractional-share ledger of the rule. At each session t, on the book at the close of T = t - 1: exits =
    held stocks with the exit signal; slots = n - held + exits; entries = the first `slots` stocks of row T's ranked
    candidate list (CSR: `candidate_vec[pointer_vec[T]:pointer_vec[T + 1]]`, best first) that are flat at T. Then a
    held stock without Open(t) or Close(t) is sold at its last close <= T, exits fill at Open(t), entries buy
    V_T / n / Close_T shares at Open(t) (cancelled without a bar), and the book is marked at Close(t).
    Returns (daily returns, holding sessions from entry open to exit open)."""
    row_count_int = close_mat.shape[0]
    share_vec = np.zeros(close_mat.shape[1])
    entry_row_vec = np.zeros(close_mat.shape[1], dtype=np.int64)
    held_vec = np.zeros(max_positions_int, dtype=np.int64)
    entry_vec = np.zeros(max_positions_int, dtype=np.int64)
    hold_vec = np.zeros(row_count_int * max_positions_int + 1, dtype=np.int64)
    held_int, hold_count_int = 0, 0
    entry_weight_float = 1.0 / max_positions_int
    cash_float, previous_total_float = 1.0, 1.0
    daily_vec = np.zeros(row_count_int)
    for t_int in range(1, row_count_int):
        p_int = t_int - 1
        exit_int = 0
        for k_int in range(held_int):
            if exit_signal_mat[p_int, held_vec[k_int]]:
                exit_int += 1
        slot_int = max_positions_int - held_int + exit_int
        entry_int = 0
        if slot_int > 0:
            for j_int in range(pointer_vec[p_int], pointer_vec[p_int + 1]):
                a_int = candidate_vec[j_int]
                if share_vec[a_int] > 0.0:
                    continue  # held at T (exiting or not): skipped without a slot
                entry_vec[entry_int] = a_int
                entry_int += 1
                if entry_int == slot_int:
                    break
        kept_int = 0
        for k_int in range(held_int):
            a_int = held_vec[k_int]
            if not (np.isfinite(open_mat[t_int, a_int]) and np.isfinite(close_mat[t_int, a_int])):
                q_int = p_int
                while not np.isfinite(close_mat[q_int, a_int]):
                    q_int -= 1
                cash_float += share_vec[a_int] * close_mat[q_int, a_int]
            elif exit_signal_mat[p_int, a_int]:
                cash_float += share_vec[a_int] * open_mat[t_int, a_int]
            else:
                held_vec[kept_int] = a_int
                kept_int += 1
                continue
            share_vec[a_int] = 0.0
            hold_vec[hold_count_int] = t_int - entry_row_vec[a_int]
            hold_count_int += 1
        held_int = kept_int
        for k_int in range(entry_int):
            a_int = entry_vec[k_int]
            if not (np.isfinite(open_mat[t_int, a_int]) and np.isfinite(close_mat[t_int, a_int])):
                continue  # cancelled: no bar to fill on
            share_vec[a_int] = previous_total_float * entry_weight_float / close_mat[p_int, a_int]
            cash_float -= share_vec[a_int] * open_mat[t_int, a_int]
            entry_row_vec[a_int] = t_int
            held_vec[held_int] = a_int
            held_int += 1
        total_float = cash_float
        for k_int in range(held_int):
            total_float += share_vec[held_vec[k_int]] * close_mat[t_int, held_vec[k_int]]
        daily_vec[t_int] = total_float / previous_total_float - 1.0
        previous_total_float = total_float
    return daily_vec, hold_vec[:hold_count_int]


def ranked_candidate_csr(candidate_mat: np.ndarray, rank_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(row pointers, asset ids): each row's candidates by rank descending (ties in asset order)."""
    row_vec, asset_vec = np.nonzero(candidate_mat)
    order_vec = np.lexsort((-rank_mat[row_vec, asset_vec], row_vec))
    pointer_vec = np.zeros(candidate_mat.shape[0] + 1, dtype=np.int64)
    pointer_vec[1:] = np.cumsum(np.bincount(row_vec, minlength=candidate_mat.shape[0]))
    return pointer_vec, asset_vec[order_vec].astype(np.int64)


def member_baseline_vec(close_mat: np.ndarray, member_mat: np.ndarray) -> np.ndarray:
    """Equal weight of the members at T with a close at T, held from the close of T+1 (alpha.scout.searches convention):
    baseline_t = mean over those stocks of Close_t / Close_(t-1) - 1, decided at row t - 2 (a missing return counts 0)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return_mat = np.nan_to_num(close_mat / np.vstack([np.full(close_mat.shape[1], np.nan), close_mat[:-1]]) - 1.0)
    weight_mat = (member_mat & np.isfinite(close_mat)).astype(float)
    weight_mat /= np.maximum(weight_mat.sum(axis=1, keepdims=True), 1.0)
    baseline_vec = np.zeros(close_mat.shape[0])
    baseline_vec[2:] = (weight_mat[:-2] * return_mat[2:]).sum(axis=1)
    return baseline_vec


def _panel_mats(panel) -> dict:
    field = lambda name_str: panel.field(name_str).to_numpy(dtype=float)
    open_mat, high_mat, low_mat, close_mat = field("Open"), field("High"), field("Low"), field("Close")
    return {"open": open_mat, "high": high_mat, "low": low_mat, "close": close_mat, "member": (panel.member_df == 1).to_numpy(),
            "complete": np.isfinite(open_mat) & np.isfinite(high_mat) & np.isfinite(low_mat) & np.isfinite(close_mat)}


def fast_daily_list_panel(panel, config_list: list[dict], base_config: Dv2Config = LIVE_CONFIG, return_hold_bool: bool = False):
    """(gross daily returns per configuration, equal-weight member baseline) from a Panel alone.

    Features are recomputed on the panel's bars; a candidate at T needs finite O/H/L/C, every feature, Close > SMA,
    the momentum bar, DV2 under the threshold and membership at T (the panel's exact flag, which equals the engine's
    as-of universe row on every session). Masks are cached per axis value and DV2 is ranked only where the other
    filters pass (speed; identical candidates). `return_hold_bool` adds each configuration's holding periods."""
    import dataclasses

    mats = _panel_mats(panel)
    close_df = panel.field("Close")
    base_cache_dict, csr_cache_dict, exit_cache_dict = {}, {}, {}
    daily_list, hold_list = [], []
    for config_dict in config_list:
        config = dataclasses.replace(base_config, **config_dict)
        base_key = (config.trend_sma_int, config.momentum_lookback_int, config.momentum_min_float, config.natr_length_int, config.dv2_length_int)
        if base_key not in base_cache_dict:
            with np.errstate(divide="ignore", invalid="ignore"):
                momentum_mat = _momentum_df(close_df, config.momentum_lookback_int).to_numpy(dtype=float)  # panel dtype (float32)
                sma_mat = close_df.rolling(config.trend_sma_int).mean().to_numpy(dtype=float)
                natr = natr_mat(mats["high"], mats["low"], mats["close"], config.natr_length_int)
                base_mat = (mats["complete"] & mats["member"] & ~np.isnan(natr) & ~np.isnan(momentum_mat) & (mats["close"] > sma_mat)
                            & (momentum_mat > config.momentum_min_float))
            dv2 = dv2_mat(mats["close"], mats["high"], mats["low"], config.dv2_length_int, base_mat)
            base_cache_dict[base_key] = (base_mat, dv2, natr)
        base_mat, dv2, natr = base_cache_dict[base_key]
        csr_key = (base_key, config.entry_dv2_max_float)
        if csr_key not in csr_cache_dict:
            with np.errstate(invalid="ignore"):
                csr_cache_dict[csr_key] = ranked_candidate_csr(base_mat & (dv2 < config.entry_dv2_max_float), natr)
        if config.exit_rule_str not in exit_cache_dict:
            exit_cache_dict[config.exit_rule_str] = exit_mat(mats["close"], mats["high"], config.exit_rule_str)
        pointer_vec, candidate_vec = csr_cache_dict[csr_key]
        daily_vec, hold_vec = _dv2_book_daily(mats["open"], mats["close"], pointer_vec, candidate_vec, exit_cache_dict[config.exit_rule_str],
                                              config.max_positions_int)
        daily_list.append(daily_vec)
        hold_list.append(hold_vec)
    baseline_vec = member_baseline_vec(mats["close"], mats["member"])
    if return_hold_bool:
        return daily_list, baseline_vec, hold_list
    return daily_list, baseline_vec


# ---------------------------------------------------------------- S3 (class E: the DV2 event inside the regime)
S3_HORIZON_TUPLE = (1, 2, 3, 5, 10, 20)


def s3_inputs(panel=None, variant_name_str: str = "dv2") -> dict:
    """Inputs for alpha.scout.stations.s3_edge.run_s3 on the variant's sealed panel ("S&P 500" or "Nasdaq 100"), from
    its own entry rule: regime = a complete bar, finite NATR and DV2, Close > SMA200 and the 126-session return above
    the variant's bar (0.05 live, 0.25 Nasdaq-100); event = DV2(126) under the variant's threshold at T (before the slot
    cap; S3 intersects with membership); indicator = -DV2 (higher = more oversold); liquidity = the 63-session
    turnover rank; horizon = the S3 horizon nearest the median holding period of the variant's rule on the same panel
    (gross replica, sessions from entry open to exit open; a tie goes to the shorter horizon, as in sector_ibs)."""
    from alpha.scout import features as scout_features
    from alpha.scout.panel import load_panel

    variant = VARIANT_DICT[variant_name_str]
    config = variant.config
    panel = panel if panel is not None else load_panel(variant.index_name_str)
    mats = _panel_mats(panel)
    features = feature_dict(panel.field("Close"), panel.field("High"), panel.field("Low"), config)
    with np.errstate(invalid="ignore"):
        regime_mat = (mats["complete"] & ~np.isnan(features["natr_mat"]) & ~np.isnan(features["dv2_mat"])
                      & (mats["close"] > features["sma_mat"]) & (features["momentum_mat"] > config.momentum_min_float))
        event_mat = features["dv2_mat"] < config.entry_dv2_max_float
    frame = lambda mat: pd.DataFrame(mat, index=panel.date_index, columns=panel.symbol_list)
    _, _, hold_list = fast_daily_list_panel(panel, [{}], base_config=config, return_hold_bool=True)
    hold_int = int(np.median(hold_list[0]))
    horizon_int = min(S3_HORIZON_TUPLE, key=lambda h: (abs(h - hold_int), h))
    return {"name_str": f"DV2 oversold ({variant.index_name_str} members, {variant_name_str} entry rule)", "panel": panel,
            "regime_mask_df": frame(regime_mat), "event_mask_df": frame(event_mat), "horizon_int": horizon_int,
            "indicator_df": frame(-features["dv2_mat"]), "liquidity_rank_df": scout_features.turnover_rank(63).compute_fn(panel),
            "expected_sign_int": 1}

def s3_result(input_dict: dict) -> dict:
    """Run S3 (class E) on `s3_inputs`, shaped for `PodPlan.s3_fn` and the card (alpha/scout/specs/sector_ibs.py)."""
    from alpha.scout.specs.sector_ibs import s3_result as shaped_s3_result

    return shaped_s3_result(input_dict)
