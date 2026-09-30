"""Parameter choice from a grid: the centre of the best plateau, never the raw peak (design D13).

Configurations are laid out on a regular grid (one axis per parameter, values in
registered order, flattened in C order, the order of
`Registration.grid_config_list`). For each configuration:

    neighbourhood median = median Sharpe of the configuration and its one-step
                           neighbours along every axis (edges have fewer neighbours)

The chosen configuration has the highest neighbourhood median; ties go to the
higher own Sharpe, then to the lower flat index. A lone lucky peak surrounded by
poor neighbours scores its neighbours' level, not its own. A configuration whose
own Sharpe is undefined (NaN, e.g. too little history) can never be chosen,
whatever its neighbours score.

    plateau ratio = neighbourhood median of the chosen configuration / peak Sharpe of the grid
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR_INT = 252


def neighbourhood_median_vec(sharpe_vec, grid_shape_tuple: tuple[int, ...]) -> np.ndarray:
    sharpe_grid = np.asarray(sharpe_vec, dtype=float).reshape(grid_shape_tuple)
    median_grid = np.empty_like(sharpe_grid)
    for flat_idx_int in range(sharpe_grid.size):
        index_tuple = np.unravel_index(flat_idx_int, grid_shape_tuple)
        neighbour_list = [sharpe_grid[index_tuple]]
        for axis_int, axis_size_int in enumerate(grid_shape_tuple):
            for step_int in (-1, 1):
                position_int = index_tuple[axis_int] + step_int
                if 0 <= position_int < axis_size_int:
                    neighbour_index_list = list(index_tuple)
                    neighbour_index_list[axis_int] = position_int
                    neighbour_list.append(sharpe_grid[tuple(neighbour_index_list)])
        median_grid[index_tuple] = np.nanmedian(neighbour_list)
    return median_grid.reshape(-1)


@dataclass(frozen=True)
class PlateauChoice:
    flat_index_int: int
    neighbourhood_median_float: float
    own_sharpe_float: float
    peak_sharpe_float: float
    plateau_ratio_float: float


def plateau_choice(sharpe_vec, grid_shape_tuple: tuple[int, ...]) -> PlateauChoice:
    sharpe_arr = np.asarray(sharpe_vec, dtype=float)
    if sharpe_arr.size != int(np.prod(grid_shape_tuple)):
        raise ValueError("sharpe_vec does not match the grid shape.")
    if not np.isfinite(sharpe_arr).any():
        raise ValueError("No configuration has a finite Sharpe.")
    median_arr = neighbourhood_median_vec(sharpe_arr, grid_shape_tuple)
    # A configuration whose neighbours are all undefined has no plateau to stand on: its "median" would be its own
    # Sharpe, which is exactly the lone peak the rule exists to avoid.
    finite_neighbour_count_vec = np.array(
        [np.isfinite(sharpe_arr[member_idx[1:]]).sum() for member_idx in _neighbour_index_list(grid_shape_tuple)]
    )
    has_plateau_vec = (finite_neighbour_count_vec > 0) | (sharpe_arr.size == 1)
    score_arr = np.where(np.isfinite(median_arr) & np.isfinite(sharpe_arr) & has_plateau_vec, median_arr, -np.inf)
    if not np.isfinite(score_arr).any():
        raise ValueError("No configuration has a finite Sharpe with a finite neighbour: there is no plateau to choose.")
    own_arr = np.where(np.isfinite(sharpe_arr), sharpe_arr, -np.inf)
    # lexsort: last key is primary. Highest median, then highest own Sharpe, then lowest index.
    order_arr = np.lexsort((np.arange(sharpe_arr.size), -own_arr, -score_arr))
    chosen_int = int(order_arr[0])
    peak_float = float(np.nanmax(sharpe_arr))
    return PlateauChoice(
        flat_index_int=chosen_int,
        neighbourhood_median_float=float(median_arr[chosen_int]),
        own_sharpe_float=float(sharpe_arr[chosen_int]),
        peak_sharpe_float=peak_float,
        plateau_ratio_float=float(median_arr[chosen_int] / peak_float) if peak_float > 0 else float("nan"),
    )


def _neighbour_index_list(grid_shape_tuple: tuple[int, ...]) -> list[np.ndarray]:
    neighbour_list = []
    for flat_idx_int in range(int(np.prod(grid_shape_tuple))):
        index_tuple = np.unravel_index(flat_idx_int, grid_shape_tuple)
        member_list = [flat_idx_int]
        for axis_int, axis_size_int in enumerate(grid_shape_tuple):
            for step_int in (-1, 1):
                position_int = index_tuple[axis_int] + step_int
                if 0 <= position_int < axis_size_int:
                    neighbour_index_list = list(index_tuple)
                    neighbour_index_list[axis_int] = position_int
                    member_list.append(int(np.ravel_multi_index(tuple(neighbour_index_list), grid_shape_tuple)))
        neighbour_list.append(np.array(member_list))
    return neighbour_list


def plateau_choice_index_mat(sharpe_mat, grid_shape_tuple: tuple[int, ...]) -> np.ndarray:
    """Vectorised `plateau_choice(...).flat_index_int` for every row of a draws x configurations matrix.

    Same rule and tie-breaks as `plateau_choice`; rows must be finite (used for null simulations).
    """
    sharpe_arr = np.asarray(sharpe_mat, dtype=float)
    if not np.all(np.isfinite(sharpe_arr)):
        raise ValueError("plateau_choice_index_mat needs finite rows; use plateau_choice for grids with NaN.")
    median_mat = np.column_stack(
        [np.median(sharpe_arr[:, member_idx], axis=1) for member_idx in _neighbour_index_list(grid_shape_tuple)]
    )
    best_score_vec = median_mat.max(axis=1, keepdims=True)
    own_on_best_mat = np.where(median_mat == best_score_vec, sharpe_arr, -np.inf)
    # argmax returns the first (lowest index) maximum, matching the scalar tie-break.
    return own_on_best_mat.argmax(axis=1)


def column_sharpe_vec(return_mat) -> np.ndarray:
    """Annualised Sharpe (zero risk-free rate, ddof = 1) of each column, NaN-aware."""
    return_arr = np.asarray(return_mat, dtype=float)
    mean_vec = np.nanmean(return_arr, axis=0)
    std_vec = np.nanstd(return_arr, axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe_vec = mean_vec / std_vec * np.sqrt(TRADING_DAYS_PER_YEAR_INT)
    return np.where(std_vec > 0, sharpe_vec, np.nan)


def make_plateau_selector(grid_shape_tuple: tuple[int, ...], min_observation_int: int = 252) -> Callable[[pd.DataFrame], object]:
    """A walk-forward selector: plateau choice on the training window (columns in grid order)."""

    def select_fn(train_return_df: pd.DataFrame) -> object:
        sharpe_vec = column_sharpe_vec(train_return_df.to_numpy())
        sharpe_vec = np.where(train_return_df.notna().sum().to_numpy() >= min_observation_int, sharpe_vec, np.nan)
        return train_return_df.columns[plateau_choice(sharpe_vec, grid_shape_tuple).flat_index_int]

    return select_fn
