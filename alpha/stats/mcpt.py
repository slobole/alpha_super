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
  listings) are REFUSED. Shuffling their rows scatters NaN across every asset's
  history, breaks the rolling windows a search needs and makes the null too weak
  (measured in review: 10.7% false positives at p <= 0.05 on pure noise). A null
  that permutes returns inside each asset's live span belongs to P4.
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

from dataclasses import dataclass
from typing import Callable

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
            "structure). MCPT cannot run on this panel until the per-asset null (P4) exists; S5 cannot pass meanwhile."
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
