"""Operators of the paper's Appendix A.1 with the PREREG section-4 conventions (research only).

Panels are (T sessions x S symbols) float64 arrays. Every time-series window is floor(d) sessions, includes t, and
needs floor(d) finite values (else NaN). Rolling statistics are computed directly per window (chunked sliding views,
no cumulative sums) so that download-date rescaling changes nothing beyond float rounding.

    ts_rank(x, d)      rank of today's value within the last d values (ties averaged) / d, in (0, 1]
    ts_argmax/argmin   days ago of the extreme, 0 = today (ties: the most recent occurrence)
    stddev / covariance / correlation   sample (ddof 1); NaN when any value is missing or an input is constant
    decay_linear(x, d) weights d, d-1, ..., 1 (today = d), rescaled to sum 1
    rank(x)            cross-sectional percentile rank among members with finite values, ties averaged, in (0, 1]
    scale(x, a)        x / sum |x| x a over members with finite values
    indneutralize(x,g) x minus the mean of x over members with finite values in the same group

Booleans are 1 / 0; comparisons, `&&`, `||` and the ternary propagate NaN. +-inf becomes NaN.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

WINDOW_BYTES_INT = 320_000_000  # chunk sliding views so that (T x c x d) doubles stay below this


def clean(x_arr: np.ndarray) -> np.ndarray:
    """+-inf -> NaN (in place when possible)."""
    x_arr = np.asarray(x_arr, dtype=np.float64)
    if x_arr.ndim == 0:
        return np.nan if not np.isfinite(x_arr) else x_arr
    bad_arr = ~np.isfinite(x_arr) & ~np.isnan(x_arr)
    if bad_arr.any():
        x_arr = x_arr.copy()
        x_arr[bad_arr] = np.nan
    return x_arr


TIE_REL_FLOAT = 1e-6
"""Tie tolerance = the data's precision. The cache's adjusted prices carry float32 precision (about 7 significant
digits), so two expressions that are equal in traded prices (h - l = 2 (o - c), or two stocks with the same traded
volume and different price factors) differ by up to ~1e-7 relative after back-adjustment, and by a different amount on
every download date. *** CRITICAL *** every tie-sensitive decision (comparisons, rank ties, ts_rank / ts_argmax ties,
constant windows) is therefore taken at a relative tolerance of 1e-6, scale-invariant by construction, and a sum or
difference below 1e-6 of its operands is zero. Real price and volume differences are orders of magnitude larger."""


def near_equal(a_arr, b_arr, rel_float: float = TIE_REL_FLOAT) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.abs(a_arr - b_arr) <= rel_float * np.maximum(np.abs(a_arr), np.abs(b_arr))


def snap_sum(out_arr: np.ndarray, a_arr, b_arr, rel_float: float = TIE_REL_FLOAT) -> np.ndarray:
    """a +/- b with a result below rel x max(|a|, |b|) set to exactly 0 (a zero in traded prices stays a zero)."""
    out_arr = np.asarray(out_arr, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        tiny_arr = np.abs(out_arr) <= rel_float * np.maximum(np.abs(np.asarray(a_arr, dtype=np.float64)), np.abs(np.asarray(b_arr, dtype=np.float64)))
    if np.ndim(out_arr) == 0:
        return np.float64(0.0) if bool(tiny_arr) and np.isfinite(out_arr) else out_arr
    return np.where(tiny_arr & np.isfinite(out_arr), 0.0, out_arr)


def floor_window(d_float: float) -> int:
    d_int = int(np.floor(float(d_float)))
    if d_int < 1:
        raise ValueError(f"window {d_float} floors below 1")
    return d_int


# ----------------------------------------------------------------------------------------------------------------------
# rolling machinery
# ----------------------------------------------------------------------------------------------------------------------
def _column_chunks(t_int: int, s_int: int, d_int: int, factor_int: int = 1):
    per_col_int = max(1, (t_int - d_int + 1) * d_int * 8 * factor_int)
    c_int = max(1, min(s_int, WINDOW_BYTES_INT // per_col_int))
    for start_int in range(0, s_int, c_int):
        yield slice(start_int, min(s_int, start_int + c_int))


def _rolling_apply(x_arr: np.ndarray, d_int: int, fn, factor_int: int = 1, *others: np.ndarray) -> np.ndarray:
    """fn(window_view, *other_views) -> (T-d+1, c) values; rows with any NaN in any window become NaN."""
    x_arr = np.asarray(x_arr, dtype=np.float64)
    t_int, s_int = x_arr.shape
    out_arr = np.full((t_int, s_int), np.nan)
    if t_int < d_int:
        return out_arr
    for col_slice in _column_chunks(t_int, s_int, d_int, factor_int):
        view_arr = np.lib.stride_tricks.sliding_window_view(x_arr[:, col_slice], d_int, axis=0)
        other_view_list = [np.lib.stride_tricks.sliding_window_view(o[:, col_slice], d_int, axis=0) for o in others]
        ok_arr = np.isfinite(view_arr).all(axis=-1)
        for other_view in other_view_list:
            ok_arr &= np.isfinite(other_view).all(axis=-1)
        with np.errstate(all="ignore"):
            value_arr = fn(view_arr, *other_view_list)
        value_arr = np.where(ok_arr, value_arr, np.nan)
        out_arr[d_int - 1 :, col_slice] = value_arr
    return clean(out_arr)


def _snap_window_sum(value_arr: np.ndarray, w_arr: np.ndarray) -> np.ndarray:
    """A window sum below the tie tolerance of the window's scale is zero (terms that cancel exactly in traded prices
    cancel exactly on every computation path)."""
    scale_arr = np.abs(w_arr).max(axis=-1)
    return np.where(np.abs(value_arr) <= TIE_REL_FLOAT * scale_arr, 0.0, value_arr)


def ts_sum(x_arr, d_float):
    return _rolling_apply(x_arr, floor_window(d_float), lambda w: _snap_window_sum(w.sum(axis=-1), w), 2)


def ts_product(x_arr, d_float):
    return _rolling_apply(x_arr, floor_window(d_float), lambda w: np.prod(w, axis=-1))


def ts_min(x_arr, d_float):
    return _rolling_apply(x_arr, floor_window(d_float), lambda w: w.min(axis=-1))


def ts_max(x_arr, d_float):
    return _rolling_apply(x_arr, floor_window(d_float), lambda w: w.max(axis=-1))


def ts_argmax(x_arr, d_float):
    # days ago of the maximum, 0 = today; values within the tie tolerance of the maximum tie and the most recent wins
    def fn(w):
        rev_arr = w[..., ::-1]
        max_arr = rev_arr.max(axis=-1, keepdims=True)
        near_arr = rev_arr >= max_arr - TIE_REL_FLOAT * np.abs(max_arr)
        return np.argmax(near_arr, axis=-1).astype(np.float64)

    return _rolling_apply(x_arr, floor_window(d_float), fn, 2)


def ts_argmin(x_arr, d_float):
    def fn(w):
        rev_arr = w[..., ::-1]
        min_arr = rev_arr.min(axis=-1, keepdims=True)
        near_arr = rev_arr <= min_arr + TIE_REL_FLOAT * np.abs(min_arr)
        return np.argmax(near_arr, axis=-1).astype(np.float64)

    return _rolling_apply(x_arr, floor_window(d_float), fn, 2)


def ts_rank(x_arr, d_float):
    d_int = floor_window(d_float)

    def fn(w):
        last_arr = w[..., -1:]
        equal_arr = near_equal(w, last_arr)
        less_arr = ((w < last_arr) & ~equal_arr).sum(axis=-1)
        return (less_arr + (equal_arr.sum(axis=-1) + 1.0) / 2.0) / d_int

    return _rolling_apply(x_arr, d_int, fn, 3)


def _constant_window(w_arr: np.ndarray) -> np.ndarray:
    """A window is constant when its spread is below the tie tolerance of its scale, so that a 1-ulp difference produced
    by a different summation order does not decide between NaN and a finite value."""
    spread_arr = w_arr.max(axis=-1) - w_arr.min(axis=-1)
    scale_arr = np.abs(w_arr).max(axis=-1)
    return spread_arr <= TIE_REL_FLOAT * scale_arr


def ts_stddev(x_arr, d_float):
    d_int = floor_window(d_float)
    if d_int < 2:
        return np.full(np.asarray(x_arr).shape, np.nan)

    def fn(w):
        dev_arr = w - w.mean(axis=-1, keepdims=True)
        var_arr = (dev_arr * dev_arr).sum(axis=-1) / (d_int - 1)
        return np.where(_constant_window(w), np.nan, np.sqrt(var_arr))  # constant window -> NaN (PREREG section 4)

    return _rolling_apply(x_arr, d_int, fn, 2)


def ts_covariance(x_arr, y_arr, d_float):
    d_int = floor_window(d_float)
    if d_int < 2:
        return np.full(np.asarray(x_arr).shape, np.nan)

    def fn(wx, wy):
        dx_arr = wx - wx.mean(axis=-1, keepdims=True)
        dy_arr = wy - wy.mean(axis=-1, keepdims=True)
        cov_arr = (dx_arr * dy_arr).sum(axis=-1) / (d_int - 1)
        return np.where(_constant_window(wx) | _constant_window(wy), np.nan, cov_arr)

    return _rolling_apply(x_arr, d_int, fn, 3, np.asarray(y_arr, dtype=np.float64))


def ts_correlation(x_arr, y_arr, d_float):
    d_int = floor_window(d_float)
    if d_int < 2:
        return np.full(np.asarray(x_arr).shape, np.nan)

    def fn(wx, wy):
        dx_arr = wx - wx.mean(axis=-1, keepdims=True)
        dy_arr = wy - wy.mean(axis=-1, keepdims=True)
        sxx_arr = (dx_arr * dx_arr).sum(axis=-1)
        syy_arr = (dy_arr * dy_arr).sum(axis=-1)
        sxy_arr = (dx_arr * dy_arr).sum(axis=-1)
        denominator_arr = np.sqrt(sxx_arr * syy_arr)
        constant_arr = _constant_window(wx) | _constant_window(wy) | ~(denominator_arr > 0)
        return np.where(constant_arr, np.nan, sxy_arr / np.where(constant_arr, 1.0, denominator_arr))

    return _rolling_apply(x_arr, d_int, fn, 3, np.asarray(y_arr, dtype=np.float64))


def decay_linear(x_arr, d_float):
    d_int = floor_window(d_float)
    weight_vec = np.arange(1, d_int + 1, dtype=np.float64)  # oldest -> 1, today -> d
    weight_vec = weight_vec / weight_vec.sum()
    return _rolling_apply(x_arr, d_int, lambda w: _snap_window_sum(w @ weight_vec, w), 2)


def delay(x_arr, d_float):
    d_int = int(np.floor(float(d_float)))
    x_arr = np.asarray(x_arr, dtype=np.float64)
    out_arr = np.full_like(x_arr, np.nan)
    if d_int == 0:
        return x_arr.copy()
    if d_int < x_arr.shape[0]:
        out_arr[d_int:] = x_arr[:-d_int]
    return out_arr


def delta(x_arr, d_float):
    x_arr = np.asarray(x_arr, dtype=np.float64)
    delayed_arr = delay(x_arr, d_float)
    return clean(snap_sum(x_arr - delayed_arr, x_arr, delayed_arr))  # a change that is zero in traded prices is zero


# ----------------------------------------------------------------------------------------------------------------------
# cross-sectional operators (members with finite values only; everything else NaN)
# ----------------------------------------------------------------------------------------------------------------------
def tolerant_ranks(x_arr: np.ndarray) -> np.ndarray:
    """Average ranks (1..n per row over finite cells) where adjacent sorted values within the tie tolerance form one
    tie group; NaN cells get NaN. Vectorized over rows."""
    t_int, s_int = x_arr.shape
    order_arr = np.argsort(x_arr, axis=1, kind="stable")  # NaN last
    sorted_arr = np.take_along_axis(x_arr, order_arr, axis=1)
    finite_arr = np.isfinite(sorted_arr)
    position_arr = np.arange(1, s_int + 1, dtype=np.float64)[None, :].repeat(t_int, axis=0)
    new_group_arr = np.ones((t_int, s_int), dtype=bool)
    with np.errstate(invalid="ignore"):
        new_group_arr[:, 1:] = ~near_equal(sorted_arr[:, 1:], sorted_arr[:, :-1]) | ~finite_arr[:, 1:] | ~finite_arr[:, :-1]
    start_arr = np.maximum.accumulate(np.where(new_group_arr, position_arr, 0.0), axis=1)
    end_flag_arr = np.ones((t_int, s_int), dtype=bool)
    end_flag_arr[:, :-1] = new_group_arr[:, 1:]
    end_arr = np.minimum.accumulate(np.where(end_flag_arr, position_arr, s_int + 1.0)[:, ::-1], axis=1)[:, ::-1]
    sorted_rank_arr = np.where(finite_arr, (start_arr + end_arr) / 2.0, np.nan)
    rank_arr = np.full((t_int, s_int), np.nan)
    np.put_along_axis(rank_arr, order_arr, sorted_rank_arr, axis=1)
    return rank_arr


def cs_rank(x_arr: np.ndarray, member_arr: np.ndarray) -> np.ndarray:
    """Percentile rank in (0, 1] with averaged ties (tie tolerance TIE_REL_FLOAT) among members with finite values."""
    x_arr = np.where(member_arr, np.asarray(x_arr, dtype=np.float64), np.nan)
    finite_arr = np.isfinite(x_arr)
    count_vec = finite_arr.sum(axis=1).astype(np.float64)
    out_arr = np.full(x_arr.shape, np.nan)
    row_vec = np.flatnonzero(count_vec > 0)
    if len(row_vec) == 0:
        return out_arr
    rank_arr = tolerant_ranks(x_arr[row_vec])
    out_arr[row_vec] = rank_arr / count_vec[row_vec, None]
    return out_arr


def cs_scale(x_arr: np.ndarray, member_arr: np.ndarray, a_float: float = 1.0) -> np.ndarray:
    x_arr = np.where(member_arr, np.asarray(x_arr, dtype=np.float64), np.nan)
    with np.errstate(all="ignore"):
        denominator_vec = np.nansum(np.abs(x_arr), axis=1)
        out_arr = x_arr * (a_float / denominator_vec)[:, None]
    return clean(out_arr)


def cs_indneutralize(x_arr: np.ndarray, member_arr: np.ndarray, group_vec: np.ndarray) -> np.ndarray:
    """x minus its group mean; groups are integer codes per symbol (0..G-1), computed per session over members with
    finite values."""
    x_arr = np.where(member_arr, np.asarray(x_arr, dtype=np.float64), np.nan)
    finite_arr = np.isfinite(x_arr)
    group_count_int = int(group_vec.max()) + 1
    onehot_arr = np.zeros((x_arr.shape[1], group_count_int))
    onehot_arr[np.arange(x_arr.shape[1]), group_vec] = 1.0
    sum_arr = np.where(finite_arr, x_arr, 0.0) @ onehot_arr
    count_arr = finite_arr.astype(np.float64) @ onehot_arr
    with np.errstate(all="ignore"):
        mean_arr = sum_arr / count_arr
    group_mean_arr = mean_arr[:, group_vec]
    # a demeaned value that is zero in traded prices (a single-member group, or members at the same price) is zero
    return clean(snap_sum(x_arr - group_mean_arr, x_arr, group_mean_arr))


# ----------------------------------------------------------------------------------------------------------------------
# element-wise
# ----------------------------------------------------------------------------------------------------------------------
def nan_compare(x_arr, y_arr, op_str: str) -> np.ndarray:
    """Comparison with the tie tolerance: two expressions equal in traded prices but 1e-7 apart after back-adjustment
    compare as equal on any download date."""
    x_arr = np.asarray(x_arr, dtype=np.float64)
    y_arr = np.asarray(y_arr, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        equal_arr = near_equal(x_arr, y_arr)
        if op_str == "<":
            value_arr = (x_arr < y_arr) & ~equal_arr
        elif op_str == ">":
            value_arr = (x_arr > y_arr) & ~equal_arr
        elif op_str == "==":
            value_arr = equal_arr
        else:
            raise ValueError(op_str)
        missing_arr = np.isnan(x_arr) | np.isnan(y_arr)
    return np.where(missing_arr, np.nan, value_arr.astype(np.float64))


def nan_logical(x_arr, y_arr, op_str: str) -> np.ndarray:
    x_arr = np.asarray(x_arr, dtype=np.float64)
    y_arr = np.asarray(y_arr, dtype=np.float64)
    missing_arr = np.isnan(x_arr) | np.isnan(y_arr)
    with np.errstate(invalid="ignore"):
        if op_str == "&&":
            value_arr = (x_arr != 0) & (y_arr != 0)
        elif op_str == "||":
            value_arr = (x_arr != 0) | (y_arr != 0)
        else:
            raise ValueError(op_str)
    return np.where(missing_arr, np.nan, value_arr.astype(np.float64))


def nan_where(cond_arr, a_arr, b_arr) -> np.ndarray:
    cond_arr = np.asarray(cond_arr, dtype=np.float64)
    a_arr = np.asarray(a_arr, dtype=np.float64)
    b_arr = np.asarray(b_arr, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        out_arr = np.where(cond_arr != 0, a_arr, b_arr)
    return clean(np.where(np.isnan(cond_arr), np.nan, out_arr))


def signed_power(x_arr, a_arr) -> np.ndarray:
    x_arr = np.asarray(x_arr, dtype=np.float64)
    with np.errstate(all="ignore"):
        return clean(np.sign(x_arr) * np.power(np.abs(x_arr), np.asarray(a_arr, dtype=np.float64)))


def power(x_arr, a_arr) -> np.ndarray:
    with np.errstate(all="ignore"):
        return clean(np.power(np.asarray(x_arr, dtype=np.float64), np.asarray(a_arr, dtype=np.float64)))


def log(x_arr) -> np.ndarray:
    with np.errstate(all="ignore"):
        return clean(np.log(np.asarray(x_arr, dtype=np.float64)))
