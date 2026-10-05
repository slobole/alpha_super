"""Contracts of the shelf-rebuild research code (scripts/research/shelf_rebuild_20260929, research only).

The results rely on: the vectorised pod model equals the house pod model, T-bill dilution equals a book that holds
T-bills as a pod, inverse-volatility weights use only history before each period, the budget objective takes the
largest feasible T-bill-free share, and the families have the declared size and weights.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

STUDY_DIR_PATH = Path(__file__).resolve().parents[1] / "scripts" / "research" / "shelf_rebuild_20260929"
sys.path.insert(0, str(STUDY_DIR_PATH))

import lib  # noqa: E402
import part_d  # noqa: E402
import part_g  # noqa: E402


def synthetic_frame(seed: int = 7, days: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2019-01-01", periods=days)
    frame = pd.DataFrame({"a": rng.normal(0.0005, 0.010, days), "b": rng.normal(0.0003, 0.004, days),
                          "c": rng.normal(0.0008, 0.020, days)}, index=index)
    frame[lib.TBILL] = 0.0001
    return frame


@pytest.mark.parametrize("policy", ["annual", "none"])
def test_pod_model_equals_house_book_return_ser(policy):
    frame = synthetic_frame()
    weights = {"a": 0.5, "b": 0.3, "c": 0.2}
    start = frame.index[10]
    mine = lib.book_returns(frame, lib.Book("x", tuple(weights), "EQ", weights, policy), start, frame.index[-1])
    house = lib.common.book_return_ser(frame.loc[start:, list(weights)], weights, policy)[0]
    np.testing.assert_allclose(mine.to_numpy(), house.to_numpy(), rtol=0, atol=1e-13)


def test_dilution_equals_book_holding_tbills_as_a_pod():
    frame = synthetic_frame()
    start = frame.index[0]
    book = lib.Book("ab", ("a", "b"), "EQ", {"a": 0.5, "b": 0.5})
    r_book = lib.book_returns(frame, book, start, frame.index[-1])
    nav = lib.dilute(r_book, frame[lib.TBILL], np.array([1.0, 0.6]))
    direct = lib.book_returns(frame, lib.Book("abt", ("a", "b", lib.TBILL), "EQ",
                                              {"a": 0.3, "b": 0.3, lib.TBILL: 0.4}), start, frame.index[-1])
    np.testing.assert_allclose(nav[0], np.cumprod(1.0 + r_book.to_numpy()), atol=1e-12)
    np.testing.assert_allclose(nav[1], np.cumprod(1.0 + direct.to_numpy()), atol=1e-12)


def test_iv_weights_use_only_history_before_each_period():
    frame = synthetic_frame()
    start = pd.Timestamp("2020-01-01")
    book = lib.Book("iv", ("a", "c"), "IV")
    log_base: list = []
    lib.book_returns(frame, book, start, frame.index[-1], weight_log=log_base)
    # Change every return from the first session of the second period on: the first two periods' weights
    # (set from history before their first session) must not move.
    second_period_start = log_base[1][0]
    changed = frame.copy()
    changed.loc[second_period_start:, "a"] *= 5.0  # one pod only: scaling both would leave IV ratios unchanged
    log_changed: list = []
    lib.book_returns(changed, book, start, frame.index[-1], weight_log=log_changed)
    assert log_changed[0][2] == pytest.approx(log_base[0][2])
    assert log_changed[1][2] == pytest.approx(log_base[1][2])
    assert log_changed[2][2] != pytest.approx(log_base[2][2])


def test_iv_first_period_falls_back_to_equal_without_history():
    frame = synthetic_frame()
    book = lib.Book("iv", ("a", "c"), "IV")
    log: list = []
    lib.book_returns(frame, book, frame.index[5], frame.index[-1], weight_log=log)
    assert log[0][2] == pytest.approx({"a": 0.5, "c": 0.5})


def test_cagr_at_budget_takes_the_largest_feasible_share():
    index = pd.bdate_range("2020-01-01", periods=300)
    book = pd.Series(0.002, index=index)
    book.iloc[100:130] = -0.01  # a ~26% drawdown, positive growth overall
    tbill = pd.Series(0.0, index=index)
    index_all = index.insert(0, index[0] - pd.Timedelta(days=3))
    obj, s = lib.cagr_at_budget(book, tbill, index_all, -0.10)
    nav = lib.dilute(book, tbill, np.array([s, s + 0.01]))
    dd = [lib.maxdd(np.diff(np.r_[1.0, path]) / np.r_[1.0, path][:-1]) for path in nav]
    assert dd[0] >= -0.10 > dd[1]
    assert 0.3 < s < 0.45
    assert obj > 0


def test_replace_pod_uses_tbill_returns():
    frame = synthetic_frame()
    out = lib.replace_pod(frame, "c")
    pd.testing.assert_series_equal(out["c"], frame[lib.TBILL], check_names=False)
    pd.testing.assert_series_equal(out["a"], frame["a"])


def test_pbo_is_a_probability():
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0003, 0.01, size=(800, 6))
    tbill = np.full(800, 0.0001)
    result = lib.pbo_cscv(returns, tbill, "excess_calmar", blocks=8)
    assert 0.0 <= result["pbo"] <= 1.0
    assert result["splits"] == 70


def test_slot_test_frame_keeps_the_replaced_pods_iv_weight():
    frame = synthetic_frame()
    book = lib.Book("iv", ("a", "c"), "IV")
    start = pd.Timestamp("2020-01-01")
    log_orig: list = []
    log_slot: list = []
    lib.book_returns(frame, book, start, frame.index[-1], weight_log=log_orig)
    lib.book_returns(lib.replace_pod(frame, "c"), book, start, frame.index[-1], weight_source=frame, weight_log=log_slot)
    assert [w for _, _, w in log_slot] == pytest.approx([w for _, _, w in log_orig])


def test_tie_band_and_tie_break_order():
    import family

    table = pd.DataFrame({"objective": [3.0, 2.9, 2.0, 1.0], "gates_pass": [True, True, True, False],
                          "slot_recent_pass": [False, True, False, True], "pods": [3, 2, 2, 2],
                          "shadow_share": 0.0, "pm_ready_share": 0.5, "trade_days_per_year": 12.0},
                         index=["top", "near", "far", "failed"])
    rng = np.random.default_rng(3)
    boot = np.column_stack([3.0 + rng.normal(0, 0.3, 1000), 2.9 + rng.normal(0, 0.3, 1000),
                            2.0 + rng.normal(0, 0.1, 1000), 1.0 + rng.normal(0, 0.1, 1000)])
    top, share = family.tie_band(table, boot, list(table.index), "gates_pass")
    assert top == "top" and "failed" not in share.index
    band = family.select(table, share, [("slot_recent_pass", False), ("pods", True), ("objective", False)])
    assert list(band.index) == ["near", "top"]  # "far" is beaten on > 90% of paths; "near" wins the tie-break


def test_pick_rung_takes_highest_cagr_within_budget_then_more_defensive():
    import part_m

    grid = pd.DataFrame({"d": [0.5, 0.4, 0.6, 0.0], "g": [0.5, 0.6, 0.4, 1.0], "t": [0.0, 0.0, 0.0, 0.0],
                         "long_cagr": [0.12, 0.12, 0.10, 0.20], "long_maxdd": [-0.09, -0.095, -0.07, -0.20]})
    assert part_m.pick_rung(grid, -0.10) == (0.5, 0.5, 0.0)  # CAGR tie -> more d
    assert part_m.pick_rung(grid, -0.25) == (0.0, 1.0, 0.0)


def test_cash_realism_uses_prior_cash_and_prior_rate():
    index = pd.bdate_range("2024-01-01", periods=4)
    path = pd.DataFrame({"cash_float": [100.0, 50.0, -20.0, 0.0], "total_value_float": [100.0] * 4}, index=index)
    rate = pd.Series([0.05, 0.10, 0.10, 0.10], index=index)
    add = lib.cash_realism_add(path, rate)
    days = [np.nan, 1, 1, 1]
    assert add.iloc[1] == pytest.approx(100.0 * (0.10 - 0.005) * days[1] / 360 / 100.0)
    assert add.iloc[3] == pytest.approx(-20.0 * (0.10 + 0.015) * days[3] / 360 / 100.0)


def test_families_have_declared_size_and_weights():
    d_books = part_d.family_books()
    g_books = part_g.family_books()
    assert len(d_books) == 83 and len({b.name for b in d_books}) == 83
    assert len(g_books) == 72 and len({b.name for b in g_books}) == 72
    assert all(b.pods[0] == "core5" for b in d_books)
    for book in g_books:
        assert sum(book.targets().values()) == pytest.approx(1.0)
        if book.tags["mr_option"] != "none":
            mr = sum(v for k, v in book.weights.items() if k in {"dv2", "dv2_adv", "hpi_vote", "etf_dv2"})
            assert mr == pytest.approx(0.36)
