"""Monte Carlo permutation test of a whole search procedure (Masters, 2020).

Question answered: "Could this search procedure have found a winner this good
in data that has no temporal structure?" The real search is compared with the
same search re-run on histories whose dates were shuffled.

Null construction:
- `return_mat` is (dates × assets). A permutation reorders whole DATE ROWS, so
  every simulated day is a real historical cross-section: correlation between
  assets survives, serial structure (the thing a timing rule exploits) does not.
- Rows are shuffled only within `strata_vec` groups when given. With volatility
  strata, a calm day is swapped only with another calm day, which keeps the
  volatility level of each period (plain shuffling makes the null too easy).
- Panels whose NaN pattern changes over time (point-in-time membership, late
  listings) are REFUSED by `mcpt`. Shuffling their rows scatters NaN across every
  asset's history, breaks the rolling windows a search needs and makes the null
  too weak (measured in review: 10.7% false positives at p <= 0.05 on pure noise).
  They use `mcpt_live_spans` (P4b), built on `live_span_source_index_mat`: one
  global date permutation that every asset follows wherever the source date is
  one of its own dates in the same stratum (for example "member"), with its
  leftover dates filling the remaining slots at random. The NaN pattern stays in
  place, each asset keeps exactly its own set of returns per stratum, and assets
  that are live on the same dates share the same permuted cross-section.
  Membership masks are not permuted: they belong to the search, which receives
  them unchanged.
  *** CRITICAL*** Co-movement survives only where assets are live and in the
  same stratum on both dates (on the S&P 500 a stock is a member for about a
  third of its listed life), so the permuted market is calmer than the real one.
  A score that compares two Sharpe ratios (strategy minus baseline) is then
  biased: the calmer baseline gains Sharpe in the null (review: about 9% false
  passes at nominal 5% with market drift). Searches on these panels must score
  the Sharpe of the daily ACTIVE return (strategy minus baseline, same days), in
  which drift and common moves cancel (P4b amendment 2). Calibration (P4b, S&P-like
  synthetic panel): momentum 8.0% and reversal 2.0% false passes at p <= 0.05,
  power 67% and 40%; momentum-like families with 0.025 < p <= 0.05 are "marginal".
- Exogenous inputs (VIX, VXN, T5YIE, ...) must be passed as extra COLUMNS of
  `return_mat` so they move with their dates. Left outside, their link with
  same-day returns is broken and volatility-scaled rules face a null that is too
  easy.
- Stratification (`strata_vec`) keeps volatility regimes in place. It also keeps
  any volatility-timing edge in place, which weakens the test for TAA, trend and
  VXN-scaled pods (review: power 0.60 plain vs 0.33 stratified on a planted
  regime edge). Plain shuffling is the default; P2 decides.

`search_fn(return_mat) -> float` must run the FULL search, including parameter
selection, and return the selected result's score. It must rebuild any prices it
needs from the returns it is given.

*** CRITICAL*** The permuted histories exist only inside this test. Nothing
computed from them may be written back as a signal, feature or parameter.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from alpha.stats.permutation import permutation_p_value, shuffle_within_groups_index


@dataclass(frozen=True)
class McptResult:
    observed_score_float: float
    null_score_vec: np.ndarray
    p_value_float: float
    permutation_count_int: int
    stratified_bool: bool


def volatility_strata_vec(return_mat, window_int: int = 63, stratum_count_int: int = 3) -> np.ndarray:
    """Stratum label per date from the trailing realised volatility of the equal-weight panel return.

    *** CRITICAL*** Used only to build the permutation null, never as a trading
    input. The window is trailing ([t − window + 1, t]); the tercile cut points
    use the whole sample on purpose, because the null describes the whole sample.
    Dates before the first full window get the stratum of the first full window.
    """
    return_arr = np.asarray(return_mat, dtype=float)
    if return_arr.ndim == 1:
        return_arr = return_arr[:, None]
    date_count_int = return_arr.shape[0]
    if date_count_int < window_int + stratum_count_int:
        raise ValueError("Not enough dates for the volatility window.")

    finite_mask_mat = np.isfinite(return_arr)
    finite_count_vec = finite_mask_mat.sum(axis=1)
    finite_sum_vec = np.where(finite_mask_mat, return_arr, 0.0).sum(axis=1)
    # Equal-weight panel return; a date with no finite return counts as flat.
    panel_return_vec = np.divide(
        finite_sum_vec, finite_count_vec, out=np.zeros(date_count_int), where=finite_count_vec > 0
    )
    trailing_vol_vec = np.full(date_count_int, np.nan)
    for date_idx_int in range(window_int - 1, date_count_int):
        trailing_vol_vec[date_idx_int] = panel_return_vec[date_idx_int - window_int + 1 : date_idx_int + 1].std()
    trailing_vol_vec[: window_int - 1] = trailing_vol_vec[window_int - 1]

    cut_point_vec = np.quantile(trailing_vol_vec, np.linspace(0.0, 1.0, stratum_count_int + 1)[1:-1])
    return np.searchsorted(cut_point_vec, trailing_vol_vec, side="right")


def mcpt(
    search_fn: Callable[[np.ndarray], float],
    return_mat,
    permutation_count_int: int,
    random_seed_int: int,
    strata_vec=None,
    availability_mask_mat=None,
) -> McptResult:
    """Run the search on the real history and on `permutation_count_int` date-shuffled histories.

    `availability_mask_mat` (dates x assets, True where the asset is listed / a member) must be passed whenever
    missing data were filled (for example zeros before a listing); a mask that changes over time is refused just
    like a changing NaN pattern, because zero-filled spans would enter the null as fake flat days.
    """
    return_arr = np.asarray(return_mat, dtype=float)
    if permutation_count_int < 1:
        raise ValueError("permutation_count_int must be >= 1.")
    missing_mask_mat = np.isnan(return_arr).reshape(return_arr.shape[0], -1)
    if availability_mask_mat is not None:
        missing_mask_mat = missing_mask_mat | ~np.asarray(availability_mask_mat, dtype=bool).reshape(missing_mask_mat.shape)
    if missing_mask_mat.any() and not (missing_mask_mat == missing_mask_mat[0]).all():
        raise ValueError(
            "The panel's availability changes over time (point-in-time membership or late listings); row shuffling "
            "would scatter it. Do NOT cut to a common live span (that keeps only survivors and invents serial "
            "structure). Use mcpt_live_spans (the per-asset null) for this panel."
        )
    date_count_int = return_arr.shape[0]
    group_vec = np.zeros(date_count_int, dtype=int) if strata_vec is None else np.asarray(strata_vec)
    if group_vec.shape[0] != date_count_int:
        raise ValueError("strata_vec must have one label per date row.")

    observed_score_float = float(search_fn(return_arr))
    rng_obj = np.random.default_rng(int(random_seed_int))
    null_score_vec = np.empty(permutation_count_int)
    for permutation_idx_int in range(permutation_count_int):
        permuted_row_idx = shuffle_within_groups_index(group_vec, rng_obj)
        null_score_vec[permutation_idx_int] = float(search_fn(return_arr[permuted_row_idx]))

    return McptResult(
        observed_score_float=observed_score_float,
        null_score_vec=null_score_vec,
        p_value_float=permutation_p_value(observed_score_float, null_score_vec, "greater"),
        permutation_count_int=int(permutation_count_int),
        stratified_bool=strata_vec is not None,
    )


def live_span_source_index_mat(available_mask_mat, rng_obj: np.random.Generator, strata_mat=None) -> np.ndarray:
    """Source row for every (date, asset) cell: a global date permutation, followed per asset and stratum.

    A global permutation pi of the dates is drawn once. For asset j and stratum s, let T be its available rows in s.
    Target row t in T takes source pi(t) when pi(t) is also in T (the global cross-section is kept wherever the
    asset can follow it); the rows of T that pi did not use fill the other targets, in the order of one global
    random key, so the fill is also shared by assets with the same rows. So each
    (asset, stratum) is permuted by a bijection of its own rows (its return distribution is exactly preserved), and
    two assets live and in the same stratum on dates t and pi(t) both take date pi(t): the cross-section survives.
    Unavailable cells map to themselves. `strata_mat` (dates x assets, integer labels, e.g. the membership flag)
    keeps member-period returns among member dates; None = one stratum.
    """
    available_arr = np.asarray(available_mask_mat, dtype=bool)
    if available_arr.ndim != 2:
        raise ValueError("available_mask_mat must be (dates x assets).")
    if strata_mat is None:
        strata_arr = np.zeros(available_arr.shape, dtype=np.int64)
    else:
        raw_strata_arr = np.asarray(strata_mat)
        if raw_strata_arr.shape != available_arr.shape:
            raise ValueError("strata_mat must have the shape of available_mask_mat.")
        if raw_strata_arr.dtype.kind == "f" and not (np.isfinite(raw_strata_arr).all() and (raw_strata_arr == np.round(raw_strata_arr)).all()):
            raise ValueError("strata_mat must hold finite integer labels.")
        strata_arr = raw_strata_arr.astype(np.int64)
    date_count_int, asset_count_int = available_arr.shape
    global_source_vec = rng_obj.permutation(date_count_int)
    global_key_vec = rng_obj.random(date_count_int)
    source_mat = np.broadcast_to(np.arange(date_count_int)[:, None], available_arr.shape).copy()
    in_target_vec = np.zeros(date_count_int, dtype=bool)
    for asset_idx_int in range(asset_count_int):
        available_vec = available_arr[:, asset_idx_int]
        if not available_vec.any():
            continue
        for stratum_int in np.unique(strata_arr[available_vec, asset_idx_int]):
            target_vec = np.flatnonzero(available_vec & (strata_arr[:, asset_idx_int] == stratum_int))
            if target_vec.size < 2:
                continue
            in_target_vec[target_vec] = True
            candidate_vec = global_source_vec[target_vec]
            follow_mask = in_target_vec[candidate_vec]
            used_mask = np.zeros(date_count_int, dtype=bool)
            used_mask[candidate_vec[follow_mask]] = True
            leftover_vec = target_vec[~used_mask[target_vec]]
            candidate_vec[~follow_mask] = leftover_vec[np.argsort(global_key_vec[leftover_vec], kind="stable")]
            source_mat[target_vec, asset_idx_int] = candidate_vec
            in_target_vec[target_vec] = False
    return source_mat


def permute_live_spans(value_mat, source_index_mat) -> np.ndarray:
    value_arr = np.asarray(value_mat, dtype=float)
    return np.take_along_axis(value_arr, source_index_mat, axis=0)


def mcpt_live_spans(
    search_fn: Callable[[np.ndarray], float],
    return_mat,
    permutation_count_int: int,
    random_seed_int: int,
    strata_mat=None,
) -> McptResult:
    """MCPT for panels whose availability changes over time (NaN = not listed; P4b per-asset null).

    Each permuted history keeps every asset's NaN pattern and permutes its returns among its own listed dates with
    `live_span_source_index_mat` (within `strata_mat` labels when given, e.g. membership). `search_fn` receives a
    matrix with exactly the real NaN pattern, and any point-in-time membership mask it closes over stays aligned
    to the real dates. No volatility stratification: P2 chose plain shuffling.
    """
    return_arr = np.asarray(return_mat, dtype=float)
    if permutation_count_int < 1:
        raise ValueError("permutation_count_int must be >= 1.")
    if return_arr.ndim != 2:
        raise ValueError("return_mat must be (dates x assets).")
    available_mask_mat = np.isfinite(return_arr)
    observed_score_float = float(search_fn(return_arr))
    rng_obj = np.random.default_rng(int(random_seed_int))
    null_score_vec = np.empty(permutation_count_int)
    for permutation_idx_int in range(permutation_count_int):
        source_index_mat = live_span_source_index_mat(available_mask_mat, rng_obj, strata_mat)
        null_score_vec[permutation_idx_int] = float(search_fn(permute_live_spans(return_arr, source_index_mat)))
    return McptResult(
        observed_score_float=observed_score_float,
        null_score_vec=null_score_vec,
        p_value_float=permutation_p_value(observed_score_float, null_score_vec, "greater"),
        permutation_count_int=int(permutation_count_int),
        stratified_bool=False,
    )
