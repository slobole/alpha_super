"""Liquidity sizing of new entries in the MR capsule pods: a mixin on the wired 1/N pods.

The rules for signals, ranks, exits, gate and parking stay those of the 1/N pods; the rule for the dollars of a new
entry changes (docs/research/GROWTH_MR_PRODUCTS_20261101.md):

    ADV63_i,T   = mean of Turnover_i over the 63 rows ending at T, with at least 42 observations
    typical_T   = median of sqrt(ADV63_i,T) over the pod's own candidates at T; undefined on a day without candidates
    reference_T = median of typical over the 252 rows ending at T, with at least 60 observations, carried over gaps
    m_i,T       = clip(sqrt(ADV63_i,T) / reference_T, 0.5, 2), and 1 where that is undefined
    dollars_i   = min(m_i,T * NAV_T / 10, 0.99 * NAV_T - stocks kept at Close_T - dollars already placed that day)
    An entry under 0.5% of NAV_T is skipped; its slot stays used for the day, as an order that never filled.

Decided after Close_T, filled at Open_(T+1). m is a signal column built from windows ending at T, so the engine's
truncated-history audit can check it; run_daily skips that audit by default and the tests run it on the size columns.
Turnover is native dollar volume (split-adjusted close x split-adjusted volume).
The names actually held differ from the 1/N pod's, because larger entries fill the book sooner: fewer entries (DV2
4,942 to 4,630, HPI 3,002 to 2,534 over 2004-2026), and about a third of entry sessions buy a different set of names.

Known caveats (kept with the strategy, owner request; measured in the record above and in
docs/strategies/book-strategy-caveats.md):
- issue: m is not normalised, so real entries are larger than 1/N's (mean m 1.18 in DV2, 1.35 in HPI). About half of
  the book's gain is size, half liquidity; bias: n/a, but do not read the whole gain as liquidity information.
- issue: the liquidity part is t 1.7 and flat without 2020 and 2024-26; 2020 is about a third of the whole gain; the
  rule was picked from 13 on seen history (family p about 0.06-0.10); bias: optimistic; plan on half the gain.
- issue: most of the book's Max DD gain sits on the 2008 proxy for TAA (1.6 points, 0.6 on real TAA data), and the
  sized capsule alone falls further than 1/N in 2012-19, 2022 and 2023-26; bias: optimistic; impact: medium.
- issue: one name can reach about 24% of a pod against about 13% at 1/N; bias: n/a; impact: more single-name risk.
- issue: only the open route is backtested. Capacity above about $5M of book assumes moving stock orders to the close.
- issue: near the cap the dollars, and sometimes the names, depend on the account at Close_T, unlike 1/N: a small gap
  between live and backtest can change a capped entry or a skip. The nightly check therefore compares m / 10, which
  does not depend on the account; a skip on one side only still shows as a difference.
- issue: HPI's backtest reads Open_(T+1) to decide whether an exit frees its slot (G-033), and with sizing that also
  reaches the room left for entry dollars, while live assumes the exit fills. 0 such events in the real ledgers.
- issue: a candidate with fewer than 42 Turnover rows gets m = 1 by rule. Live also blocks the whole decision, exits
  included, when under 80% of members have usable Turnover at T; the backtest has no such check, so a day live blocks
  would trade there. No real session since 2004 falls under that bar in either pod's frame. Below the bar, faults on
  up to 20% of members still trade at m = 1, and a wrong scale passes, within the 5% to 20% bounds.
- issue: a small account rounds a 5% entry to zero shares at half the share price a 10% entry would, and BIL is sized
  as if those dollars were spent, until the next weekly sweep. Fund-scale pods are not affected.
- issue: negative cash, unfinanced as in the 1/N pods (G-023): DV2 55 sessions, minimum -10.9% of NAV; HPI 50 sessions,
  minimum -5.7% (1/N: 167 and 168 sessions, -9.6% and -8.5%).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from strategies.mr_capsule.parking import PARKING_SYMBOL_TUPLE

TURNOVER_FIELD_STR = "Turnover"
LIQUIDITY_SIZE_FIELD_STR = "liquidity_size_float"
ADV_WINDOW_INT = 63
ADV_MIN_OBSERVATION_INT = 42
REFERENCE_WINDOW_INT = 252
REFERENCE_MIN_OBSERVATION_INT = 60
SIZE_MIN_FLOAT = 0.5
SIZE_MAX_FLOAT = 2.0
STOCK_CAP_FLOAT = 0.99
MIN_ENTRY_FRACTION_FLOAT = 0.005
COVERAGE_MIN_FLOAT = 0.80


def liquidity_size_and_reference(turnover_df: pd.DataFrame, candidate_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """(m_i,T, reference_T). Both frames are session x symbol with the same labels; candidate_df holds booleans."""
    # *** CRITICAL*** every window ends at T: m_i,T uses Turnover through Close_T only, for a fill at Open_(T+1).
    score_mat = np.sqrt(turnover_df.astype(float).rolling(ADV_WINDOW_INT, min_periods=ADV_MIN_OBSERVATION_INT).mean().to_numpy())
    candidate_mat = candidate_df.to_numpy(dtype=bool)
    typical_vec = np.full(len(turnover_df), np.nan)
    candidate_row_vec = np.flatnonzero(candidate_mat.any(axis=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # a day whose candidates all lack ADV63 stays undefined
        typical_vec[candidate_row_vec] = np.nanmedian(np.where(candidate_mat, score_mat, np.nan)[candidate_row_vec], axis=1)
    reference_ser = pd.Series(typical_vec, index=turnover_df.index).rolling(REFERENCE_WINDOW_INT, min_periods=REFERENCE_MIN_OBSERVATION_INT).median().ffill()
    with np.errstate(divide="ignore", invalid="ignore"):
        size_mat = np.clip(score_mat / reference_ser.to_numpy()[:, None], SIZE_MIN_FLOAT, SIZE_MAX_FLOAT)
    size_mat = np.nan_to_num(size_mat, nan=1.0, posinf=1.0, neginf=1.0)
    return pd.DataFrame(size_mat, index=turnover_df.index, columns=turnover_df.columns), reference_ser


def member_at_each_session_df(universe_df: pd.DataFrame, session_index: pd.Index, stock_list: list[str]) -> pd.DataFrame:
    """PIT membership per session: the last universe row dated <= T, as both pods read it inside iterate."""
    sorted_universe_df = universe_df.sort_index().reindex(columns=stock_list).fillna(0)
    # *** CRITICAL*** method="ffill" takes the latest row on or before T, never a later row.
    return sorted_universe_df.reindex(session_index, method="ffill").fillna(0).eq(1)


class LiquiditySizedPodMixin:
    """Goes before the pod class. The pod keeps its own iterate; this changes the dollars of an entry and tells parking."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # (fill session, symbol) -> m / 10 of each placed entry, before the cap, for the nightly live-vs-backtest check.
        self.entry_weight_by_date_dict: dict[tuple[str, str], float] = {}
        self.skipped_entry_symbol_list: list[str] = []
        self.liquidity_reference_ser: pd.Series | None = None
        self._liquidity_close_row_ser: pd.Series | None = None
        self._exit_symbol_set: set[str] = set()
        self._placed_dollar_float = 0.0
        self._base_dollar_float = 0.0

    def _liquidity_candidate_df(self, signal_df: pd.DataFrame, stock_list: list[str]) -> pd.DataFrame:
        raise NotImplementedError

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        signal_df = super().compute_signals(pricing_data)
        stock_list = sorted({str(symbol_obj) for symbol_obj, _ in signal_df.columns if not str(symbol_obj).startswith("$")})
        if not any((symbol_str, TURNOVER_FIELD_STR) in signal_df.columns for symbol_str in stock_list if symbol_str not in PARKING_SYMBOL_TUPLE):
            # *** CRITICAL*** a frame without the field must stop the pod, not silently size every entry at m = 1.
            # One stock without it cannot be a candidate in either pod, so it needs no stop here; live coverage is checked below.
            raise ValueError(f"Liquidity sizing requires {TURNOVER_FIELD_STR}; the frame has none.")
        turnover_df = pd.DataFrame(
            {symbol_str: pd.to_numeric(signal_df[(symbol_str, TURNOVER_FIELD_STR)], errors="coerce") if (symbol_str, TURNOVER_FIELD_STR) in signal_df.columns
             else pd.Series(np.nan, index=signal_df.index) for symbol_str in stock_list},
            index=signal_df.index,
        )
        size_df, self.liquidity_reference_ser = liquidity_size_and_reference(turnover_df, self._liquidity_candidate_df(signal_df, stock_list))
        size_df.columns = pd.MultiIndex.from_tuples([(symbol_str, LIQUIDITY_SIZE_FIELD_STR) for symbol_str in stock_list])
        self._accounting_policy_dict["capsule_entry_sizing_str"] = (
            "1/10 x sqrt(ADV63) over the rolling typical candidate, between half and double; stocks stop at 99% of NAV; entries under 0.5% skipped"
        )
        output_df = pd.concat([signal_df, size_df], axis=1, copy=False)
        output_df.attrs = dict(signal_df.attrs)
        return output_df

    def validate_liquidity_sizing_ready(self, signal_df: pd.DataFrame, decision_date_ts: pd.Timestamp) -> float:
        """Live only: fail closed when the decision close cannot size honestly, else the share of members with usable Turnover.

        A carried reference and m = 1 would otherwise hide bad Turnover.
        """
        decision_date_ts = pd.Timestamp(decision_date_ts)
        date_str = decision_date_ts.date()
        reference_ser = self.liquidity_reference_ser
        reference_float = float(reference_ser.get(decision_date_ts, np.nan)) if reference_ser is not None else np.nan
        if not np.isfinite(reference_float) or reference_float <= 0.0:
            raise ValueError(f"Liquidity sizing has no usable reference at {date_str}.")
        stock_list = sorted({str(symbol_obj) for symbol_obj, _ in signal_df.columns if not str(symbol_obj).startswith("$") and str(symbol_obj) not in PARKING_SYMBOL_TUPLE})
        member_ser = member_at_each_session_df(self.universe_df, pd.DatetimeIndex([decision_date_ts]), stock_list).iloc[0]
        member_list = list(member_ser.index[member_ser.to_numpy(dtype=bool)])
        if not member_list:
            raise ValueError(f"Liquidity sizing found no members at {date_str}.")
        # *** CRITICAL*** rows through Close_T only.
        turnover_df = pd.DataFrame(
            {symbol_str: pd.to_numeric(signal_df[(symbol_str, TURNOVER_FIELD_STR)], errors="coerce") if (symbol_str, TURNOVER_FIELD_STR) in signal_df.columns
             else pd.Series(np.nan, index=signal_df.index) for symbol_str in member_list},
            index=signal_df.index,
        ).loc[:decision_date_ts].iloc[-ADV_WINDOW_INT:].astype(float)
        # A zero shrinks m and an infinity drops out of the rolling mean, so neither counts as an observation here.
        usable_df = turnover_df.where(np.isfinite(turnover_df) & (turnover_df > 0.0))
        usable_ser = (usable_df.notna().sum() >= ADV_MIN_OBSERVATION_INT) & usable_df.iloc[-1].notna()
        coverage_float = float(usable_ser.mean())
        if coverage_float < COVERAGE_MIN_FLOAT:
            raise ValueError(f"Liquidity sizing needs usable {TURNOVER_FIELD_STR} for {COVERAGE_MIN_FLOAT:.0%} of members at {date_str}; it has {coverage_float:.0%}.")
        return coverage_float

    def iterate(self, data, close, open_prices):
        self._liquidity_close_row_ser = close
        self._exit_symbol_set, self._placed_dollar_float, self._base_dollar_float = set(), 0.0, 0.0
        self.skipped_entry_symbol_list = []
        return super().iterate(data, close, open_prices)

    def order_target_value(self, asset, target, **kwargs):
        if float(target) == 0.0:
            self._exit_symbol_set.add(str(asset))
        return super().order_target_value(asset, target, **kwargs)

    def order_value(self, asset, value, **kwargs):
        base_float = float(value)
        self._base_dollar_float += base_float
        close_row_ser = self._liquidity_close_row_ser
        # *** CRITICAL*** m and the held value are read at Close_T; the fill is at Open_(T+1).
        size_float = float(close_row_ser.get((str(asset), LIQUIDITY_SIZE_FIELD_STR), 1.0))
        position_ser = self.get_positions()
        held_float = 0.0
        for symbol_obj, share_float in position_ser[position_ser > 0].items():
            symbol_str = str(symbol_obj)
            if symbol_str in self._exit_symbol_set or symbol_str in PARKING_SYMBOL_TUPLE:
                continue
            close_float = close_row_ser.get((symbol_str, "Close"), np.nan)
            held_float += float(share_float) * (float(close_float) if np.isfinite(close_float) else 0.0)
        total_float = float(self.previous_total_value)
        wanted_float = base_float * size_float
        # The stock cap is the parking logic's own 1% buffer, so a full book keeps room for gaps and costs.
        dollar_float = min(wanted_float, STOCK_CAP_FLOAT * total_float - held_float - self._placed_dollar_float)
        if dollar_float < MIN_ENTRY_FRACTION_FLOAT * total_float:
            self.skipped_entry_symbol_list.append(str(asset))
            return None
        self._placed_dollar_float += dollar_float
        self.entry_weight_by_date_dict[(pd.Timestamp(self.current_bar).date().isoformat(), str(asset))] = wanted_float / total_float
        return super().order_value(asset, dollar_float, **kwargs)

    def _place_parking_orders(self, data_df, close_row_ser, stock_value_after_orders_float, gate_open_bool):
        # *** CRITICAL*** the pod adds its 1/N dollars to the stock value; parking must see the true dollars, or cash goes negative unfinanced.
        return super()._place_parking_orders(
            data_df, close_row_ser, stock_value_after_orders_float + self._placed_dollar_float - self._base_dollar_float, gate_open_bool,
        )
