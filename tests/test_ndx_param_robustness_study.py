"""Tests for the NDX parameter-robustness research replica (scripts/research/ndx_param_robustness_core.py)."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "research"))
import ndx_param_robustness_core as core  # noqa: E402


def make_universe(symbol_count_int: int = 12, seed_int: int = 7) -> dict:
    """Synthetic PIT universe: business days 1998-06..2001-12, random-walk OHLC, all members, quiet VXN."""
    rng = np.random.default_rng(seed_int)
    date_index = pd.bdate_range("1998-06-01", "2001-12-31")
    date_count_int = len(date_index)
    drift_vec = rng.normal(0.0005, 0.0004, symbol_count_int)
    return_arr = drift_vec[None, :] + rng.normal(0, 0.015, (date_count_int, symbol_count_int))
    close_arr = 50.0 * np.exp(np.cumsum(return_arr, axis=0))
    open_arr = close_arr * np.exp(rng.normal(0, 0.003, close_arr.shape))
    high_arr = np.maximum(open_arr, close_arr) * (1 + np.abs(rng.normal(0, 0.005, close_arr.shape)))
    low_arr = np.minimum(open_arr, close_arr) * (1 - np.abs(rng.normal(0, 0.005, close_arr.shape)))
    spy_ser = pd.Series(100.0 * np.exp(np.cumsum(np.full(date_count_int, 0.0004))), index=date_index)
    month_end_index = pd.DatetimeIndex(pd.Series(date_index, index=date_index.to_period("M")).groupby(level=0).max().to_numpy())
    return {
        "universe_str": "TEST",
        "date_index": date_index,
        "symbol_list": [f"S{i:02d}" for i in range(symbol_count_int)],
        "open_arr": open_arr,
        "high_arr": high_arr,
        "low_arr": low_arr,
        "close_arr": close_arr,
        "volume_arr": np.full(close_arr.shape, 1e6),
        "dividend_arr": np.zeros(close_arr.shape),
        "unadjusted_close_arr": close_arr.copy(),
        "member_arr": np.ones(close_arr.shape, dtype=np.int8),
        "spy_close_ser": spy_ser,
        "qqq_close_ser": spy_ser,
        "vxn_close_ser": pd.Series(20.0, index=date_index),
        "month_end_index": month_end_index,
    }


def selections(universe_dict: dict, cell: core.Cell) -> list[tuple]:
    target_list = core.build_target_list(core.FeatureBook(universe_dict), cell)
    return [(t["decision_pos"], tuple(sorted(t["symbol_idx_vec"].tolist())), tuple(np.round(np.sort(t["weight_vec"]), 12))) for t in target_list]


def test_scale_free_cells_ignore_a_per_stock_price_constant():
    base_dict = make_universe()
    scaled_dict = dict(base_dict)
    scale_vec = np.linspace(0.02, 50.0, len(base_dict["symbol_list"]))
    for field_str in ("open_arr", "high_arr", "low_arr", "close_arr"):
        scaled_dict[field_str] = base_dict[field_str] * scale_vec[None, :]
    for cell in (
        core.ANCHOR_CELL,
        dataclasses.replace(core.ANCHOR_CELL, numerator_str="B3612", denominator_str="NATR63"),
        dataclasses.replace(core.ANCHOR_CELL, numerator_str="ROC12-1", denominator_str="none", n_int=5, weight_str="IV"),
        dataclasses.replace(core.ANCHOR_CELL, buffer_int=2, offset_int=-3, stock_filter_int=50),
    ):
        base_list, scaled_list = selections(base_dict, cell), selections(scaled_dict, cell)
        assert [x[:2] for x in base_list] == [x[:2] for x in scaled_list]
        for a_tuple, b_tuple in zip(base_list, scaled_list):
            np.testing.assert_allclose(a_tuple[2], b_tuple[2], rtol=1e-9)
    # the dollar-ATR reference is not scale-free: its lists move with the constant
    b_cell = dataclasses.replace(core.B_CELL, n_int=3)
    assert [x[1] for x in selections(base_dict, b_cell)] != [x[1] for x in selections(scaled_dict, b_cell)]


def test_buffer_zero_is_plain_top_n_and_buffer_keeps_incumbents():
    universe_dict = make_universe()
    plain_list = selections(universe_dict, dataclasses.replace(core.ANCHOR_CELL, n_int=3))
    buffered_list = selections(universe_dict, dataclasses.replace(core.ANCHOR_CELL, n_int=3, buffer_int=4))
    assert len(plain_list) == len(buffered_list)
    # with a wide buffer the list changes less often than the plain top 3
    def changes(lst):
        return sum(a[1] != b[1] for a, b in zip(lst[:-1], lst[1:]))
    assert changes(buffered_list) <= changes(plain_list)
    assert all(len(x[1]) <= 3 for x in buffered_list)


def test_offset_schedule_shifts_decisions_and_roc_anchors():
    universe_dict = make_universe()
    base = core.build_schedule(universe_dict, 0)
    shifted = core.build_schedule(universe_dict, -4)
    np.testing.assert_array_equal(shifted["decision_pos_vec"], base["decision_pos_vec"] - 4)
    roc_arr = core.roc_table(universe_dict["close_arr"], shifted, "ROC3")
    row_int = 20
    pos_vec = shifted["decision_pos_vec"]
    expected_vec = universe_dict["close_arr"][pos_vec[row_int]] / universe_dict["close_arr"][pos_vec[row_int - 3]] - 1
    np.testing.assert_allclose(roc_arr[row_int], expected_vec)
    skip_arr = core.roc_table(universe_dict["close_arr"], base, "ROC12-1")
    pos_vec = base["decision_pos_vec"]
    expected_vec = universe_dict["close_arr"][pos_vec[row_int - 1]] / universe_dict["close_arr"][pos_vec[row_int - 12]] - 1
    np.testing.assert_allclose(skip_arr[row_int], expected_vec)


def test_simulator_sizing_fill_commission_and_dividend():
    date_index = pd.bdate_range("1999-12-29", "2000-01-10")
    close_arr = np.array([[10.0, 20.0]] * len(date_index))
    open_arr = close_arr.copy()
    open_arr[date_index.get_loc(pd.Timestamp("2000-01-04"))] = [11.0, 19.0]
    close_arr[date_index.get_loc(pd.Timestamp("2000-01-04")):] = [12.0, 18.0]
    dividend_arr = np.zeros_like(close_arr)
    dividend_arr[date_index.get_loc(pd.Timestamp("2000-01-05"))] = [0.4, 0.0]  # entitlement Jan 5, paid before Jan 6 open
    universe_dict = {"date_index": date_index, "open_arr": open_arr, "close_arr": close_arr, "dividend_arr": dividend_arr}
    decision_pos = date_index.get_loc(pd.Timestamp("2000-01-03"))
    target_list = [{"decision_pos": decision_pos, "execution_pos": decision_pos + 1,
                    "symbol_idx_vec": np.array([0, 1]), "weight_vec": np.array([0.5, 0.25])}]
    sim = core.simulate(universe_dict, target_list, slippage_float=0.001)
    # shares from the decision close and previous total value (100k): int(50000/10)=5000, int(25000/20)=1250
    cash_after_float = 100_000.0 - 5000 * 11.0 * 1.001 - 1250 * 19.0 * 1.001 - 25.0 - 6.25
    total_jan4_float = cash_after_float + 5000 * 12.0 + 1250 * 18.0
    assert sim["total_ser"].loc["2000-01-04"] == pytest.approx(total_jan4_float, rel=1e-12)
    total_jan6_float = total_jan4_float + 0.75 * 5000 * 0.4
    assert sim["total_ser"].loc["2000-01-06"] == pytest.approx(total_jan6_float, rel=1e-12)
    assert sim["commission_ser"].loc["2000-01-04"] == pytest.approx(31.25)


def test_relative_liquidity_filter_drops_the_least_traded_quarter():
    universe_dict = make_universe(symbol_count_int=8)
    universe_dict["volume_arr"] = np.tile(np.arange(1, 9, dtype=float) * 1e6, (len(universe_dict["date_index"]), 1))
    feature_obj = core.FeatureBook(universe_dict)
    pos = len(universe_dict["date_index"]) - 5
    dollar_vec = feature_obj.adv20_dollar()[pos]
    keep_vec = core.liquidity_pass_vec(feature_obj, pos, universe_dict["member_arr"][pos], "REL25")
    assert keep_vec.sum() == 6  # quantile(0.25) of 8 names sits between the 2nd and 3rd least traded
    assert set(np.flatnonzero(~keep_vec)) == set(np.argsort(dollar_vec)[:2])
    assert core.liquidity_pass_vec(feature_obj, pos, universe_dict["member_arr"][pos], "none").all()
