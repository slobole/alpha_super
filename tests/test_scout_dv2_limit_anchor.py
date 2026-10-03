"""DV2 limit-anchor study (scripts/research/scout_dv2_limit_anchor_20261003/anchor_book.py): causality of every offset
measure, the open-anchored limit and its fills on hand-made bars, and a calibration that sees only its window."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha.scout.panel import Panel
from alpha.scout.specs import dv2
from alpha.scout.universes import rule_mats
from scripts.research.scout_dv2_limit_anchor_20261003.anchor_book import (
    ExcursionQuantile,
    book_fill_rate,
    calibrate,
    close_anchor_limit_mat,
    close_excursion_mat,
    close_return_std_mat,
    natr_fraction_mat,
    open_anchor_limit_mat,
    open_excursion_mat,
    rolling_mean_mat,
    truncate_mats,
)
from scripts.research.scout_dv2_limit_entry_20261002.limit_book import (
    entry_limit_mat,
    exit_limit_mat,
    limit_book,
)


def _random_bars(row_count_int: int = 120, column_count_int: int = 3, seed_int: int = 7):
    rng_obj = np.random.default_rng(seed_int)
    close_mat = 20.0 * np.exp(np.cumsum(rng_obj.normal(0, 0.02, (row_count_int, column_count_int)), axis=0))
    open_mat = close_mat * np.exp(rng_obj.normal(0, 0.01, close_mat.shape))
    low_mat = np.minimum(open_mat, close_mat) * (1.0 - rng_obj.uniform(0, 0.02, close_mat.shape))
    high_mat = np.maximum(open_mat, close_mat) * (1.0 + rng_obj.uniform(0, 0.02, close_mat.shape))
    return open_mat, high_mat, low_mat, close_mat


# ---------------------------------------------------------------- measures: values
def test_close_return_std_is_the_std_of_the_last_21_returns():
    _, _, _, close_mat = _random_bars()
    std_mat = close_return_std_mat(close_mat)
    returns_vec = close_mat[1:, 0] / close_mat[:-1, 0] - 1.0  # return of row t sits at index t - 1
    assert np.isnan(std_mat[20, 0]) and np.isfinite(std_mat[21, 0])  # 21 returns need rows 0 .. 21
    np.testing.assert_allclose(std_mat[60, 0], np.std(returns_vec[39:60], ddof=1))  # returns of rows 40 .. 60


def test_excursions_per_bar():
    open_mat = np.array([[10.0], [10.0], [9.0]])
    low_mat = np.array([[9.5], [9.8], [8.1]])
    close_mat = np.array([[10.0], [9.9], [8.5]])
    np.testing.assert_allclose(open_excursion_mat(open_mat, low_mat)[:, 0], [0.05, 0.02, 0.1])
    out_vec = close_excursion_mat(close_mat, low_mat)[:, 0]
    assert np.isnan(out_vec[0])
    np.testing.assert_allclose(out_vec[1:], [(10.0 - 9.8) / 10.0, (9.9 - 8.1) / 9.9])
    assert np.isnan(open_excursion_mat(np.array([[np.nan]]), np.array([[9.0]]))).all()


def test_excursion_quantile_matches_numpy_and_floors_at_zero():
    rng_obj = np.random.default_rng(3)
    value_mat = rng_obj.normal(0.01, 0.01, (100, 2))
    mask_mat = np.zeros((100, 2), dtype=bool)
    mask_mat[[10, 62, 63, 99], 0] = True
    mask_mat[80, 1] = True
    quantile = ExcursionQuantile(value_mat, mask_mat)
    offset_mat = quantile.offset_mat(0.40)  # the drop exceeded with probability 0.40 = the 0.60 quantile
    assert np.isnan(offset_mat[10, 0])  # fewer than 63 bars of history
    for r_int, c_int in ((62, 0), (63, 0), (99, 0), (80, 1)):
        expected_float = max(np.quantile(value_mat[r_int - 62: r_int + 1, c_int], 0.60), 0.0)
        assert offset_mat[r_int, c_int] == pytest.approx(expected_float)
    assert np.isnan(offset_mat[~mask_mat]).all()  # only the masked cells carry an offset
    negative = ExcursionQuantile(-np.abs(value_mat) - 0.001, mask_mat)
    assert (negative.offset_mat(0.40)[mask_mat & np.isfinite(negative.offset_mat(0.40))] == 0.0).all()
    gap_mat = value_mat.copy()
    gap_mat[50, 0] = np.nan  # a missing bar inside the window: no offset
    assert np.isnan(ExcursionQuantile(gap_mat, mask_mat).offset_mat(0.40)[63, 0])


# ---------------------------------------------------------------- measures: causality (row T reads rows <= T only)
@pytest.mark.parametrize("measure_str", ["natr", "std21", "open_excursion", "close_excursion", "dex21_mean", "quantile_open", "quantile_close"])
def test_every_offset_measure_is_prefix_invariant(measure_str):
    open_mat, high_mat, low_mat, close_mat = _random_bars()
    mask_mat = np.ones(close_mat.shape, dtype=bool)

    def compute(o, h, lo, c):
        if measure_str == "natr":
            return natr_fraction_mat(dv2.natr_mat(h, lo, c, 14))
        if measure_str == "std21":
            return close_return_std_mat(c)
        if measure_str == "open_excursion":
            return open_excursion_mat(o, lo)
        if measure_str == "close_excursion":
            return close_excursion_mat(c, lo)
        if measure_str == "dex21_mean":
            return rolling_mean_mat(open_excursion_mat(o, lo))
        if measure_str == "quantile_open":
            return ExcursionQuantile(open_excursion_mat(o, lo), mask_mat).offset_mat(0.4)
        return ExcursionQuantile(close_excursion_mat(c, lo), mask_mat).offset_mat(0.4)

    full_mat = compute(open_mat, high_mat, low_mat, close_mat)
    cut_int = 90
    changed = [m.copy() for m in (open_mat, high_mat, low_mat, close_mat)]
    for seed_int, m in enumerate(changed):  # a different draw per field, so ratios such as Low / Open change too
        m[cut_int:] *= np.random.default_rng(5 + seed_int).uniform(0.5, 1.5, m[cut_int:].shape)
    changed_mat = compute(*changed)
    np.testing.assert_array_equal(changed_mat[:cut_int], full_mat[:cut_int])
    assert not np.allclose(changed_mat[cut_int:], full_mat[cut_int:], equal_nan=True)  # the change is real


# ---------------------------------------------------------------- limit prices
def test_close_anchor_limit_equals_the_parent_formula():
    _, high_mat, low_mat, close_mat = _random_bars()
    nominal_mat = close_mat * 1.7
    natr_mat = dv2.natr_mat(high_mat, low_mat, close_mat, 14)
    np.testing.assert_allclose(close_anchor_limit_mat(close_mat, nominal_mat, 0.5 * natr_fraction_mat(natr_mat)),
                               entry_limit_mat(close_mat, nominal_mat, natr_mat, 0.5), equal_nan=True)


def test_open_anchor_limit_values_and_rounding():
    # adjusted Close_T 5 and nominal 10 (ratio 2); next open 4.9 adjusted = 9.80 nominal
    close_mat = np.array([[5.0], [5.0]])
    nominal_mat = np.array([[10.0], [10.0]])
    open_mat = np.array([[5.0], [4.9]])
    out_mat = open_anchor_limit_mat(open_mat, close_mat, nominal_mat, np.array([[0.01], [0.01]]))
    assert out_mat[0, 0] == pytest.approx(9.70 / 2.0)  # 9.80 x 0.99 = 9.702 -> 9.70 nominal, rounded DOWN
    assert np.isnan(out_mat[1, 0])  # no session after the last row: no order
    zero = open_anchor_limit_mat(np.array([[10.0], [9.50]]), np.array([[10.0], [10.0]]), np.array([[10.0], [10.0]]), np.zeros((2, 1)))
    assert zero[0, 0] == pytest.approx(9.49)  # offset 0: one tick below the open, never at it
    assert np.isnan(open_anchor_limit_mat(np.array([[10.0], [9.5]]), np.array([[10.0], [10.0]]), np.array([[10.0], [10.0]]),
                                          np.array([[np.nan], [0.01]]))[0, 0])


def test_open_anchor_limit_is_strictly_below_the_open():
    open_mat, _, _, close_mat = _random_bars()
    out_mat = open_anchor_limit_mat(open_mat, close_mat, close_mat * 3.1, np.full(close_mat.shape, 0.0003))
    assert np.all(out_mat[:-1] < open_mat[1:])


def test_open_anchor_row_t_reads_only_the_next_open_and_rows_up_to_t():
    open_mat, high_mat, low_mat, close_mat = _random_bars()
    nominal_mat = close_mat * 1.3
    offset_mat = rolling_mean_mat(open_excursion_mat(open_mat, low_mat))
    full_mat = open_anchor_limit_mat(open_mat, close_mat, nominal_mat, offset_mat)
    t_int = 80
    # everything of T+1 except its open, and every row after T+1: row T unchanged
    o, h, lo, c, n = (m.copy() for m in (open_mat, high_mat, low_mat, close_mat, nominal_mat))
    o[t_int + 2:] *= 1.3
    for m in (h, lo, c, n):
        m[t_int + 1:] *= 0.7
    changed_offset_mat = rolling_mean_mat(open_excursion_mat(o, lo))  # measures recomputed from the changed bars
    np.testing.assert_array_equal(open_anchor_limit_mat(o, c, n, changed_offset_mat)[: t_int + 1], full_mat[: t_int + 1])
    # the open of T+1 itself moves row T (it is the anchor)
    o2 = open_mat.copy()
    o2[t_int + 1] *= 0.9
    assert not np.allclose(open_anchor_limit_mat(o2, close_mat, nominal_mat, offset_mat)[t_int], full_mat[t_int])


# ---------------------------------------------------------------- open-anchored fills on hand-made bars
def _one_asset_book(open_list, low_list, close_list, entry_limit, slip_float=0.01):
    to = lambda v: np.array(v, dtype=float).reshape(-1, 1)
    row_count_int = len(close_list)
    candidate_mat = np.zeros((row_count_int, 1), dtype=bool)
    candidate_mat[0, 0] = True
    pointer_vec, candidate_vec = dv2.ranked_candidate_csr(candidate_mat, np.ones((row_count_int, 1)))
    mats = {"open": to(open_list), "close": to(close_list), "pointer_vec": pointer_vec, "candidate_vec": candidate_vec,
            "exit_signal": np.zeros((row_count_int, 1), dtype=bool)}
    high_mat = np.fmax(to(open_list), to(close_list)) + 0.1
    date_index = pd.bdate_range("2020-01-01", periods=row_count_int)
    shape_tuple = (row_count_int, 1)
    return limit_book(date_index, ["A"], mats, high_mat, to(low_list), 1, "2020-01-01", entry_limit, "moo", np.full(shape_tuple, 0.001), None,
                      slip_float, np.ones(shape_tuple), 0.0, 0.0, 0.0).log_df


def test_open_anchor_gap_down_never_fills_at_the_open():
    """Close_T 10.00, Open_(T+1) 9.50 (gap down), offset 2%: the close anchor (9.80) fills at the open and pays the spread;
    the open anchor (9.31, set after the open) needs a trade-through and fills passively at its limit."""
    close_list = [10.0, 9.6, 9.6]
    open_list = [10.0, 9.5, 9.6]
    offset_mat = np.full((3, 1), 0.02)
    nominal_mat = np.array(close_list).reshape(-1, 1)
    close_limit = close_anchor_limit_mat(nominal_mat, nominal_mat, offset_mat)
    open_limit = open_anchor_limit_mat(np.array(open_list).reshape(-1, 1), nominal_mat, nominal_mat, offset_mat)
    assert close_limit[0, 0] == pytest.approx(9.80) and open_limit[0, 0] == pytest.approx(9.31)
    through_low_list = [10.0, 9.30, 9.6]  # 9.30 <= 9.31 x (1 - 0.001)
    close_fill = _one_asset_book(open_list, through_low_list, close_list, close_limit).iloc[0]
    assert (close_fill["kind_int"], close_fill["code_int"]) == (1, 1)  # marketable at the open
    assert close_fill["value_float"] == pytest.approx(10_000 * 9.5) and close_fill["spread_float"] > 0
    open_fill = _one_asset_book(open_list, through_low_list, close_list, open_limit).iloc[0]
    assert (open_fill["kind_int"], open_fill["code_int"]) == (1, 2)  # passive, at the limit
    assert open_fill["value_float"] == pytest.approx(10_000 * 9.31) and open_fill["spread_float"] == 0.0
    touch_low_list = [10.0, 9.31, 9.6]  # touches the limit only
    assert _one_asset_book(open_list, touch_low_list, close_list, open_limit).iloc[0]["kind_int"] == 0


def test_open_anchor_with_an_open_at_the_close_anchor_level():
    """An open exactly where a close-anchored limit would sit: the open anchor still waits for a dip below the open."""
    close_list = [10.0, 10.0]
    open_list = [10.0, 9.80]
    nominal_mat = np.array(close_list).reshape(-1, 1)
    open_limit = open_anchor_limit_mat(np.array(open_list).reshape(-1, 1), nominal_mat, nominal_mat, np.full((2, 1), 0.0))
    assert open_limit[0, 0] == pytest.approx(9.79)
    assert _one_asset_book(open_list, [10.0, 9.80], close_list, open_limit).iloc[0]["kind_int"] == 0  # low = open: no dip
    entry = _one_asset_book(open_list, [10.0, 9.75], close_list, open_limit).iloc[0]
    assert (entry["code_int"], entry["value_float"]) == (2, pytest.approx(10_000 * 9.79))


# ---------------------------------------------------------------- calibration sees only its window
def _synthetic_panel(row_count_int: int = 700, column_count_int: int = 40, seed_int: int = 4) -> Panel:
    rng_obj = np.random.default_rng(seed_int)
    date_index = pd.bdate_range("2001-01-01", periods=row_count_int)
    close_mat = 30.0 * np.exp(np.cumsum(rng_obj.normal(0.0006, 0.02, (row_count_int, column_count_int)), axis=0))
    open_mat = close_mat * np.exp(rng_obj.normal(0, 0.01, close_mat.shape))
    high_mat = np.maximum(open_mat, close_mat) * (1.0 + rng_obj.uniform(0, 0.02, close_mat.shape))
    low_mat = np.minimum(open_mat, close_mat) * (1.0 - rng_obj.uniform(0, 0.02, close_mat.shape))
    columns = [f"S{i:02d}" for i in range(column_count_int)]
    frame = lambda m: pd.DataFrame(m, index=date_index, columns=columns)
    field_dict = {"Open": frame(open_mat), "High": frame(high_mat), "Low": frame(low_mat), "Close": frame(close_mat),
                  "Volume": frame(np.ones_like(close_mat)), "Turnover": frame(np.full_like(close_mat, 1e6)),
                  "Unadjusted Close": frame(close_mat), "Dividend": frame(np.zeros_like(close_mat))}
    member_mat = np.ones(close_mat.shape, dtype=np.int8)
    member_mat[:260] = 0
    return Panel("synthetic", field_dict, pd.DataFrame(member_mat, index=date_index, columns=columns), "synthetic", True)


def _calibration_inputs(panel: Panel):
    mats = rule_mats(panel, dv2.LIVE_CONFIG)
    field = lambda f: panel.field(f).to_numpy(dtype=float)
    return mats, field("High"), field("Low"), field("Unadjusted Close")


def test_truncate_mats_cuts_the_candidate_csr():
    mats, *_ = _calibration_inputs(_synthetic_panel())
    cut = truncate_mats(mats, 400)
    assert cut["close"].shape[0] == 400 and cut["pointer_vec"].size == 401
    assert cut["candidate_vec"].size == mats["pointer_vec"][400]


def test_calibration_uses_only_the_calibration_window():
    panel = _synthetic_panel()
    date_index = panel.date_index
    start_str, end_str = str(date_index[262].date()), str(date_index[520].date())
    mats, high_mat, low_mat, nominal_mat = _calibration_inputs(panel)
    margin_mat = np.full(mats["close"].shape, 0.001)

    def calibrated(mats, high_mat, low_mat, nominal_mat):
        exit_limit = exit_limit_mat(mats["close"], nominal_mat)
        offset_mat = rolling_mean_mat(open_excursion_mat(mats["open"], low_mat))
        rate_fn = lambda k: book_fill_rate(date_index, panel.symbol_list, mats, high_mat, low_mat,
                                           open_anchor_limit_mat(mats["open"], mats["close"], nominal_mat, k * offset_mat), margin_mat,
                                           exit_limit, 10, start_str, end_str)[0]
        return calibrate(rate_fn, 0.40, 0.0, 1.0, decreasing_bool=True, iteration_int=10), rate_fn

    reference, _ = calibrated(mats, high_mat, low_mat, nominal_mat)
    assert abs(reference["fill_rate_float"] - 0.40) < 0.05 and reference["param_float"] > 0
    # scramble every bar after the calibration end (the first scrambled open is the anchor of the last kept row's order)
    end_row_int = int(np.searchsorted(date_index.to_numpy(), np.datetime64(pd.Timestamp(end_str)), side="right"))
    rng_obj = np.random.default_rng(11)
    scrambled_panel = _synthetic_panel()
    for name_str in ("Open", "High", "Low", "Close", "Unadjusted Close"):
        values_mat = scrambled_panel.field(name_str).to_numpy(dtype=float).copy()
        values_mat[end_row_int:] *= rng_obj.uniform(0.6, 1.4, values_mat[end_row_int:].shape)
        scrambled_panel.field_dict[name_str] = pd.DataFrame(values_mat, index=date_index, columns=panel.symbol_list)
    scrambled_mats, scrambled_high, scrambled_low, scrambled_nominal = _calibration_inputs(scrambled_panel)
    assert not np.allclose(scrambled_mats["open"][end_row_int:], mats["open"][end_row_int:])
    scrambled, _ = calibrated(scrambled_mats, scrambled_high, scrambled_low, scrambled_nominal)
    assert scrambled == reference
    # and the same book over the whole sample does see the change (the invariance is not vacuous)
    full_rate_fn = lambda m, h, lo, n: book_fill_rate(date_index, panel.symbol_list, m, h, lo,
                                                      open_anchor_limit_mat(m["open"], m["close"], n, 0.3 * rolling_mean_mat(open_excursion_mat(m["open"], lo))),
                                                      margin_mat, exit_limit_mat(m["close"], n), 10, start_str, str(date_index[-1].date()))[1]
    assert full_rate_fn(mats, high_mat, low_mat, nominal_mat) != full_rate_fn(scrambled_mats, scrambled_high, scrambled_low, scrambled_nominal)


def test_calibrate_brackets_and_returns_the_closest_point():
    result = calibrate(lambda k: np.exp(-k), 0.25, 0.0, 1.0, decreasing_bool=True)  # needs the bound expanded past 1
    assert result["param_float"] == pytest.approx(np.log(4.0), abs=2e-3)
    increasing = calibrate(lambda p: p ** 2, 0.36, 0.02, 0.98, decreasing_bool=False)
    assert increasing["param_float"] == pytest.approx(0.6, abs=2e-3)
