"""DV2 limit-entry study (scripts/research/scout_dv2_limit_entry_20261002/limit_book.py): fill rules on hand-made bars,
causality of the order prices, and parity of the market-on-open book with alpha.scout.universes.costed_book."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha.scout.panel import Panel
from alpha.scout.specs import dv2
from alpha.scout.universes import costed_book, rule_mats
from scripts.research.scout_dv2_limit_entry_20261002.limit_book import (
    buy_limit_fill,
    entry_limit_mat,
    event_fill_mats,
    exit_limit_mat,
    limit_book,
    sell_limit_fill,
    trade_through_margin_mat,
)


# ---------------------------------------------------------------- fill rules on one bar
def test_buy_limit_open_gap_through_fills_at_the_open_marketable():
    code_int, price_float = buy_limit_fill(9.50, 9.40, 10.00, 0.001)
    assert code_int == 1 and price_float == 9.50  # opened below the limit: filled at the (better) open, spread charged
    assert buy_limit_fill(10.00, 9.99, 10.00, 0.001) == (1, 10.00)  # open exactly at the limit is marketable


def test_buy_limit_touch_without_trade_through_does_not_fill():
    assert buy_limit_fill(10.20, 10.00, 10.00, 0.001)[0] == 0  # low touches the limit
    assert buy_limit_fill(10.20, 9.995, 10.00, 0.001)[0] == 0  # through by less than the margin (10 bp = 1 cent)
    code_int, price_float = buy_limit_fill(10.20, 9.99, 10.00, 0.001)
    assert code_int == 2 and price_float == 10.00  # traded through by the margin: passive fill at the limit


def test_buy_limit_no_fill_and_missing_inputs():
    assert buy_limit_fill(10.20, 10.05, 10.00, 0.001)[0] == 0  # never reached
    assert buy_limit_fill(10.20, 9.00, np.nan, 0.001)[0] == 0  # no order
    assert buy_limit_fill(np.nan, 9.00, 10.00, 0.001)[0] == 0  # no bar
    assert buy_limit_fill(10.20, 9.00, 10.00, np.nan)[0] == 0  # unknown margin: never assume a passive fill


def test_sell_limit_mirrors_the_buy_rule():
    assert sell_limit_fill(10.50, 10.60, 10.00, 0.001) == (1, 10.50)  # opened above: marketable at the open
    assert sell_limit_fill(9.80, 10.00, 10.00, 0.001)[0] == 0  # touch only
    assert sell_limit_fill(9.80, 10.005, 10.00, 0.001)[0] == 0  # through by less than the margin
    assert sell_limit_fill(9.80, 10.01, 10.00, 0.001) == (2, 10.00)  # traded through: at the limit
    assert sell_limit_fill(9.80, 9.95, 10.00, 0.001)[0] == 0  # never reached


# ---------------------------------------------------------------- order prices
def test_entry_limit_uses_close_and_natr_of_t_rounded_down_to_the_tick():
    close_mat = np.array([[20.0], [10.0]])  # adjusted
    nominal_mat = np.array([[40.0], [20.0]])  # nominal = 2 x adjusted
    natr_mat = np.array([[3.0], [2.5]])  # percent
    k0 = entry_limit_mat(close_mat, nominal_mat, natr_mat, 0.0)
    np.testing.assert_allclose(k0, close_mat)  # k = 0: the close itself (already on the tick)
    k1 = entry_limit_mat(close_mat, nominal_mat, natr_mat, 1.0)
    np.testing.assert_allclose(k1, [[40.0 * 0.97 / 2.0], [20.0 * 0.975 / 2.0]])  # 38.80 and 19.50 nominal, on the tick
    odd = entry_limit_mat(np.array([[10.0]]), np.array([[10.0]]), np.array([[3.33]]), 0.5)  # 10 x (1 - 0.01665) = 9.8335
    np.testing.assert_allclose(odd, [[9.83]])  # rounded DOWN
    assert np.isnan(entry_limit_mat(np.array([[10.0]]), np.array([[10.0]]), np.array([[np.nan]]), 0.5)).all()
    # float32 noise on a price that sits on the tick does not move a k = 0 limit a cent down
    np.testing.assert_allclose(entry_limit_mat(np.array([[np.float32(10.01)]]), np.array([[np.float32(10.01)]]), np.array([[2.0]]), 0.0),
                               [[float(np.float32(10.01))]], rtol=1e-6)


def test_exit_limit_is_the_close_rounded_up_and_sub_dollar_tick():
    np.testing.assert_allclose(exit_limit_mat(np.array([[5.0]]), np.array([[10.004]])), [[10.01 * 5.0 / 10.004]])  # 10.01 nominal, back in adjusted units
    np.testing.assert_allclose(exit_limit_mat(np.array([[0.5123]]), np.array([[0.5123]])), [[0.5123]])  # $0.0001 tick below $1 (not 0.52)
    np.testing.assert_allclose(entry_limit_mat(np.array([[0.8]]), np.array([[0.8]]), np.array([[1.37]]), 0.5), [[0.7945]])  # 0.79452 down


def test_trade_through_margin_is_a_tick_or_a_tenth_of_the_larger_spread():
    nominal_mat = np.array([[10.0], [2.0], [100.0]])
    model_a = np.array([[0.0005], [0.0400], [np.nan]])
    model_b = np.array([[0.0002], [0.0600], [0.0001]])
    margin_mat = trade_through_margin_mat(nominal_mat, [model_a, model_b])
    np.testing.assert_allclose(margin_mat[:, 0], [0.001, 0.006, 0.0001])  # tick 10 bp; 0.1 x 600 bp; tick 1 bp


@pytest.mark.parametrize("function_str", ["entry", "exit", "margin"])
def test_order_prices_are_prefix_invariant(function_str):
    """Row T of every order price depends on rows <= T only: changing every later row leaves it unchanged."""
    rng_obj = np.random.default_rng(1)
    close_mat = 20 + rng_obj.random((30, 4))
    nominal_mat = close_mat * 1.7
    natr_mat = 1 + rng_obj.random((30, 4))
    spread_mat = 0.001 * rng_obj.random((30, 4))

    def compute(c, n, a, s):
        if function_str == "entry":
            return entry_limit_mat(c, n, a, 0.5)
        if function_str == "exit":
            return exit_limit_mat(c, n)
        return trade_through_margin_mat(n, [s])

    full_mat = compute(close_mat, nominal_mat, natr_mat, spread_mat)
    changed = [m.copy() for m in (close_mat, nominal_mat, natr_mat, spread_mat)]
    for m in changed:
        m[15:] *= 3.0
    np.testing.assert_array_equal(compute(*changed)[:15], full_mat[:15])


def test_event_fills_read_only_the_next_session_bar():
    open_mat = np.array([[10.0], [9.0], [10.5], [9.8]])
    low_mat = np.array([[9.5], [8.9], [9.90], [9.7]])
    limit_mat = np.array([[9.5], [10.0], [9.95], [np.nan]])
    margin_mat = np.full((4, 1), 0.001)
    code_mat, price_mat = event_fill_mats(open_mat, low_mat, limit_mat, margin_mat)
    # row 0 works on row 1: open 9.0 <= 9.5 -> open fill; row 1 on row 2: open 10.5 > 10, low 9.90 <= 9.99 -> passive at 10.0
    # row 2 on row 3: open 9.8 <= 9.95 -> open; row 3 has no next bar and no limit -> none
    assert code_mat[:, 0].tolist() == [1, 2, 1, 0]
    np.testing.assert_allclose(price_mat[:3, 0], [9.0, 10.0, 9.8])


# ---------------------------------------------------------------- the book on hand-made bars
def _hand_mats(open_list, high_list, low_list, close_list, candidate_rows, exit_rows):
    """One asset; `candidate_rows` = decision rows T where it is a candidate; `exit_rows` = rows with the exit signal."""
    to = lambda v: np.array(v, dtype=float).reshape(-1, 1)
    row_count_int = len(close_list)
    candidate_mat = np.zeros((row_count_int, 1), dtype=bool)
    candidate_mat[list(candidate_rows), 0] = True
    pointer_vec, candidate_vec = dv2.ranked_candidate_csr(candidate_mat, np.ones((row_count_int, 1)))
    exit_mat = np.zeros((row_count_int, 1), dtype=bool)
    exit_mat[list(exit_rows), 0] = True
    mats = {"open": to(open_list), "close": to(close_list), "pointer_vec": pointer_vec, "candidate_vec": candidate_vec, "exit_signal": exit_mat}
    return mats, to(high_list), to(low_list)


def _run(mats, high_mat, low_mat, entry_limit, exit_str, slip=0.01, exit_limit=None, margin=0.001, max_attempt_int=5):
    date_index = pd.bdate_range("2020-01-01", periods=mats["close"].shape[0])
    shape_tuple = mats["close"].shape
    return limit_book(date_index, ["A"], mats, high_mat, low_mat, 1, str(date_index[0].date()), entry_limit, exit_str,
                      np.full(shape_tuple, margin), exit_limit, slip, np.ones(shape_tuple), 0.0, 0.0, 0.0, max_exit_attempt_int=max_attempt_int)


def test_book_unfilled_order_leaves_the_slot_empty_and_reorders_only_on_a_fresh_signal():
    # rows:        0      1      2      3      4      5
    close_list = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    open_list = [10.0, 10.2, 10.2, 10.2, 9.9, 10.0]
    high_list = [10.0, 10.3, 10.3, 10.3, 10.0, 10.0]
    low_list = [10.0, 10.1, 10.1, 10.1, 9.8, 10.0]
    mats, high_mat, low_mat = _hand_mats(open_list, high_list, low_list, close_list, candidate_rows=[0, 3], exit_rows=[])
    limit = np.full((6, 1), 10.0)  # k = 0 style limit at the close
    result = _run(mats, high_mat, low_mat, limit, "moo")
    log_df = result.log_df
    # decision 0 -> order on row 1: open 10.2 > 10, low 10.1 > 10 -> no fill; no candidate at rows 1-2 -> no order
    # decision 3 -> order on row 4: open 9.9 <= 10 -> filled at the open (marketable)
    assert log_df["kind_int"].tolist() == [0, 1]
    assert log_df["date"].tolist() == [result.daily_ser.index[1], result.daily_ser.index[4]]
    assert log_df["code_int"].tolist() == [0, 1]
    assert (result.daily_ser.iloc[1:4] == 0).all()  # the empty slot earns nothing
    entry_row = log_df.iloc[1]
    assert entry_row["value_float"] == pytest.approx(100_000 * 9.9 / 10.0)
    assert entry_row["spread_float"] == pytest.approx(entry_row["value_float"] * 0.01)  # marketable: spread charged


def test_book_passive_fill_pays_no_spread_and_moo_book_pays_it():
    close_list = [10.0, 10.0, 10.0]
    open_list = [10.0, 10.2, 10.0]
    high_list = [10.0, 10.3, 10.0]
    low_list = [10.0, 9.9, 10.0]
    mats, high_mat, low_mat = _hand_mats(open_list, high_list, low_list, close_list, candidate_rows=[0], exit_rows=[])
    passive = _run(mats, high_mat, low_mat, np.full((3, 1), 10.0), "moo").log_df.iloc[0]
    assert (passive["code_int"], passive["spread_float"]) == (2, 0.0)
    assert passive["value_float"] == pytest.approx(100_000.0)  # 10,000 shares at the 10.00 limit
    moo = _run(mats, high_mat, low_mat, None, "moo").log_df.iloc[0]
    assert moo["code_int"] == 1 and moo["value_float"] == pytest.approx(102_000.0) and moo["spread_float"] == pytest.approx(1_020.0)


def test_book_limit_exit_carries_then_forces_market_on_open():
    # entry on row 1 (MOO); exit signal at row 1's close; the sell limit (previous close) never trades through for 2
    # sessions (max 2 attempts here), so the position sells market-on-open on the 3rd session (row 4).
    close_list = [10.0, 11.0, 10.5, 10.4, 10.6, 10.6]
    open_list = [10.0, 10.0, 10.8, 10.4, 10.3, 10.6]
    high_list = [10.0, 11.0, 11.0, 10.5, 10.6, 10.6]  # row 2: limit 11.00 touched, not through; row 3: limit 10.50, high 10.5
    low_list = [10.0, 9.9, 10.4, 10.3, 10.2, 10.6]
    mats, high_mat, low_mat = _hand_mats(open_list, high_list, low_list, close_list, candidate_rows=[0], exit_rows=[1, 2, 3])
    exit_limit = np.array(close_list).reshape(-1, 1)
    result = _run(mats, high_mat, low_mat, None, "limit", exit_limit=exit_limit, max_attempt_int=2)
    exits = result.log_df[result.log_df["kind_int"] == -1]
    assert len(exits) == 1 and exits.iloc[0]["date"] == result.daily_ser.index[4] and exits.iloc[0]["code_int"] == 3
    assert exits.iloc[0]["value_float"] == pytest.approx(10_000 * 10.3)  # at the open of row 4
    # the same bars with a trade-through on row 3 (high 10.52 > 10.50 x 1.001) fill passively at the limit 10.50
    high_list[3] = 10.52
    mats, high_mat, low_mat = _hand_mats(open_list, high_list, low_list, close_list, candidate_rows=[0], exit_rows=[1, 2, 3])
    exits = _run(mats, high_mat, low_mat, None, "limit", exit_limit=exit_limit, max_attempt_int=2).log_df.query("kind_int == -1")
    assert exits.iloc[0]["date"] == result.daily_ser.index[3] and exits.iloc[0]["code_int"] == 2
    assert exits.iloc[0]["value_float"] == pytest.approx(10_000 * 10.5) and exits.iloc[0]["spread_float"] == 0.0


def test_book_resting_sell_keeps_its_slot():
    """With a limit exit, a candidate cannot use the slot of a position whose sell has not filled yet."""
    close_mat = np.array([[10.0, 10.0], [11.0, 10.0], [10.5, 10.0], [10.5, 10.0]])
    open_mat = np.array([[10.0, 10.0], [10.0, 10.0], [10.6, 10.0], [10.5, 10.0]])
    high_mat = np.array([[10.0, 10.0], [11.0, 10.0], [10.9, 10.0], [10.6, 10.0]])  # row 2: never reaches 11.00
    low_mat = high_mat - 0.5
    candidate_mat = np.zeros((4, 2), dtype=bool)
    candidate_mat[0, 0] = True
    candidate_mat[1, 1] = True  # B is a candidate at row 1, when A's exit is only committed
    pointer_vec, candidate_vec = dv2.ranked_candidate_csr(candidate_mat, np.ones((4, 2)))
    exit_signal = np.zeros((4, 2), dtype=bool)
    exit_signal[1, 0] = True
    mats = {"open": open_mat, "close": close_mat, "pointer_vec": pointer_vec, "candidate_vec": candidate_vec, "exit_signal": exit_signal}
    date_index = pd.bdate_range("2020-01-01", periods=4)
    run = lambda exit_str: limit_book(date_index, ["A", "B"], mats, high_mat, low_mat, 1, "2020-01-01", None, exit_str, np.full((4, 2), 0.001),
                                      close_mat.copy(), 0.0, np.ones((4, 2)), 0.0, 0.0, 0.0).log_df
    moo_log, limit_log = run("moo"), run("limit")
    assert ((moo_log["asset"] == "B") & (moo_log["kind_int"] == 1)).any()  # MOO exit frees the slot for B on row 2
    assert not ((limit_log["asset"] == "B") & (limit_log["kind_int"] == 1)).any()  # the resting sell keeps it


# ---------------------------------------------------------------- parity with costed_book
def _synthetic_panel(row_count_int: int = 700, column_count_int: int = 40, seed_int: int = 4) -> Panel:
    rng_obj = np.random.default_rng(seed_int)
    date_index = pd.bdate_range("2001-01-01", periods=row_count_int)
    close_mat = 30.0 * np.exp(np.cumsum(rng_obj.normal(0.0006, 0.02, (row_count_int, column_count_int)), axis=0))
    open_mat = close_mat * np.exp(rng_obj.normal(0, 0.01, close_mat.shape))
    high_mat = np.maximum(open_mat, close_mat) * (1.0 + rng_obj.uniform(0, 0.02, close_mat.shape))
    low_mat = np.minimum(open_mat, close_mat) * (1.0 - rng_obj.uniform(0, 0.02, close_mat.shape))
    close_mat[650:, 3] = np.nan  # a delisting
    columns = [f"S{i:02d}" for i in range(column_count_int)]
    frame = lambda m: pd.DataFrame(m.astype(np.float32), index=date_index, columns=columns)
    field_dict = {"Open": frame(open_mat), "High": frame(high_mat), "Low": frame(low_mat), "Close": frame(close_mat),
                  "Volume": frame(np.ones_like(close_mat)), "Turnover": frame(np.full_like(close_mat, 1e6)),
                  "Unadjusted Close": frame(close_mat * 2.0), "Dividend": frame(np.zeros_like(close_mat))}
    member_mat = (rng_obj.random(close_mat.shape) < 0.9).astype(np.int8)
    member_mat[: 260] = 0
    return Panel("synthetic", field_dict, pd.DataFrame(member_mat, index=date_index, columns=columns), "synthetic", True)


def _book_inputs(panel):
    mats = rule_mats(panel, dv2.LIVE_CONFIG)
    high_mat, low_mat = panel.field("High").to_numpy(dtype=float), panel.field("Low").to_numpy(dtype=float)
    scale_mat = np.nan_to_num(mats["close"] / panel.field("Unadjusted Close").to_numpy(dtype=float), nan=1.0, posinf=1.0)
    return mats, high_mat, low_mat, scale_mat


def test_moo_book_reproduces_costed_book_exactly():
    panel = _synthetic_panel()
    start_str = str(panel.date_index[260].date())
    mats, high_mat, low_mat, scale_mat = _book_inputs(panel)
    slip_mat = np.random.default_rng(3).uniform(0.0002, 0.003, mats["close"].shape)
    reference = costed_book(panel, dv2.LIVE_CONFIG, slip_mat, 0.005, 1.0, share_unit_str="nominal", start_date_str=start_str, mats=mats,
                            max_fee_fraction_float=0.01)
    mine = limit_book(panel.date_index, panel.symbol_list, mats, high_mat, low_mat, 10, start_str, None, "moo", np.full(slip_mat.shape, 0.001),
                      None, slip_mat, scale_mat, 0.005, 1.0, 0.01)
    assert (reference.fill_df["kind_int"] == 1).sum() > 50
    np.testing.assert_allclose(mine.daily_ser.to_numpy(), reference.daily_ser.to_numpy(), rtol=0, atol=1e-12)
    fills = mine.log_df[mine.log_df["kind_int"].isin([1, -1, -2])].reset_index(drop=True)
    pd.testing.assert_frame_equal(fills[["date", "asset", "kind_int", "value_float"]], reference.fill_df, check_dtype=False)


def test_a_limit_that_is_always_marketable_is_the_moo_book():
    panel = _synthetic_panel()
    start_str = str(panel.date_index[260].date())
    mats, high_mat, low_mat, scale_mat = _book_inputs(panel)
    shape_tuple = mats["close"].shape
    run = lambda entry_limit: limit_book(panel.date_index, panel.symbol_list, mats, high_mat, low_mat, 10, start_str, entry_limit, "moo",
                                         np.full(shape_tuple, 0.001), None, 0.001, scale_mat, 0.005, 1.0, 0.01).daily_ser.to_numpy()
    np.testing.assert_allclose(run(np.full(shape_tuple, 1e12)), run(None), rtol=0, atol=1e-12)


def test_limit_entries_fill_less_and_never_above_the_limit():
    panel = _synthetic_panel()
    start_str = str(panel.date_index[260].date())
    mats, high_mat, low_mat, scale_mat = _book_inputs(panel)
    natr = dv2.natr_mat(high_mat, low_mat, mats["close"], 14)
    limit = entry_limit_mat(mats["close"], panel.field("Unadjusted Close").to_numpy(dtype=float), natr, 0.5)
    result = limit_book(panel.date_index, panel.symbol_list, mats, high_mat, low_mat, 10, start_str, limit, "moo",
                        np.full(limit.shape, 0.001), None, 0.0, scale_mat, 0.0, 0.0, 0.0)
    log_df = result.log_df
    assert (log_df["kind_int"] == 0).sum() > 20 and (log_df["kind_int"] == 1).sum() > 20
    entries = log_df[log_df["kind_int"] == 1]
    row_vec = panel.date_index.get_indexer(entries["date"])
    column_vec = pd.Index(panel.symbol_list).get_indexer(entries["asset"])
    shares_vec = result.total_value_ser.to_numpy()[row_vec - 1] / 10 / mats["close"][row_vec - 1, column_vec]
    price_vec = entries["value_float"].to_numpy() / shares_vec
    assert np.all(price_vec <= limit[row_vec - 1, column_vec] * (1 + 1e-12))  # never pays above its limit
    assert np.all(price_vec <= mats["open"][row_vec, column_vec] * (1 + 1e-12))  # nor above the session's open
