"""Walk-forward re-selection over a grid of precomputed configuration returns.

Input is `config_return_df`: one column of daily net returns per parameter
configuration, each run over full history with fixed parameters. The walk-forward
does not re-run strategies; at every refit date it re-selects a column using
only earlier rows and then holds it until the next refit. Stitching the held
periods gives the out-of-sample (OOS) return series.

    walk-forward efficiency = Sharpe(OOS stitched) / mean(Sharpe_IS of each chosen config)

Refits happen on the first trading day of January (and of July for 6-month
refits). The first refit is the first such date with at least `train_years_int`
years of history before it.

Design sensitivity (Lotter, Financial Hacker): the same walk-forward under a
fixed grid of 8 designs: anchored (5-year minimum history) and rolling 3-, 5- and
8-year windows, each refitting every 6 or 12 months. Anchored designs with
different minimum histories were dropped from the grid because they make the
same selections and would count one piece of evidence several times. A real
edge survives most designs; an artefact of one design does not.

*** CRITICAL*** Look-ahead boundary: the selection for the test window starting
at date s sees only rows with index < s. The test window is [s, next refit).

Warm-up and gaps: a configuration needs `MIN_TRAIN_OBSERVATION_INT` finite
training returns to be selectable, so a long-lookback config with two finite
days cannot win on a meaningless Sharpe. A NaN inside a chosen configuration's
test window is an error: whether it means "flat" or "missing" must be decided by
the caller, not silently dropped.

Known optimism (stated, not modelled): switching configuration at a refit is
free here. In a live pod a switch costs roughly one extra rebalance of turnover
at each refit where the chosen configuration changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR_INT = 252


@dataclass(frozen=True)
class WalkForwardDesign:
    anchored_bool: bool
    train_years_int: int
    refit_months_int: int

    @property
    def label_str(self) -> str:
        kind_str = "anchored" if self.anchored_bool else "rolling"
        return f"{kind_str}_{self.train_years_int}y_refit{self.refit_months_int}m"


MIN_TRAIN_OBSERVATION_INT = 252
REGISTERED_DESIGN = WalkForwardDesign(anchored_bool=True, train_years_int=5, refit_months_int=12)

DESIGN_GRID_TUPLE = tuple(
    WalkForwardDesign(anchored_bool=anchored_bool, train_years_int=train_years_int, refit_months_int=refit_months_int)
    for anchored_bool, train_years_int in ((True, 5), (False, 3), (False, 5), (False, 8))
    for refit_months_int in (6, 12)
)


def annualized_sharpe_float(return_ser: pd.Series) -> float:
    """Annualised Sharpe with a zero risk-free rate; NaN when undefined."""
    clean_ser = return_ser.dropna()
    if clean_ser.size < 2:
        return float("nan")
    std_float = float(clean_ser.std(ddof=1))
    if std_float <= 0.0:
        return float("nan")
    return float(clean_ser.mean()) / std_float * np.sqrt(TRADING_DAYS_PER_YEAR_INT)


def select_max_sharpe(train_return_df: pd.DataFrame) -> object:
    """Simplest selector: the highest in-sample Sharpe among configs with enough finite history.

    Picks the peak, not the plateau: S4's plateau selector replaces it in P5.
    """
    eligible_df = train_return_df.loc[:, train_return_df.notna().sum() >= MIN_TRAIN_OBSERVATION_INT]
    sharpe_ser = eligible_df.apply(annualized_sharpe_float)
    if sharpe_ser.isna().all():
        raise ValueError(
            f"No configuration has {MIN_TRAIN_OBSERVATION_INT} finite training returns and a defined Sharpe."
        )
    return sharpe_ser.idxmax()


@dataclass(frozen=True)
class WalkForwardResult:
    design: WalkForwardDesign
    oos_return_ser: pd.Series
    refit_df: pd.DataFrame
    oos_sharpe_float: float
    mean_is_sharpe_float: float
    efficiency_float: float


def _refit_start_list(date_index: pd.DatetimeIndex, design: WalkForwardDesign) -> list[pd.Timestamp]:
    if 12 % design.refit_months_int != 0:
        raise ValueError("refit_months_int must divide 12 so refits stay on calendar anchors.")
    earliest_date = date_index[0] + pd.DateOffset(years=design.train_years_int)
    first_anchor_date = pd.Timestamp(year=earliest_date.year, month=1, day=1)
    while first_anchor_date < earliest_date:
        first_anchor_date = first_anchor_date + pd.DateOffset(months=design.refit_months_int)
    start_list: list[pd.Timestamp] = []
    refit_idx_int = 0
    while True:
        # Offsets from the first anchor (not cumulative), so month-end anchors do not drift.
        anchor_date = first_anchor_date + pd.DateOffset(months=design.refit_months_int * refit_idx_int)
        if anchor_date > date_index[-1]:
            break
        # First trading date on or after the calendar anchor.
        start_list.append(date_index[date_index.searchsorted(anchor_date)])
        refit_idx_int += 1
    return sorted(set(start_list))


def run_walk_forward(
    config_return_df: pd.DataFrame,
    select_config_fn: Callable[[pd.DataFrame], object],
    design: WalkForwardDesign = REGISTERED_DESIGN,
) -> WalkForwardResult:
    if not isinstance(config_return_df.index, pd.DatetimeIndex) or not config_return_df.index.is_monotonic_increasing:
        raise ValueError("config_return_df needs a sorted DatetimeIndex.")
    date_index = config_return_df.index
    start_list = _refit_start_list(date_index, design)
    if not start_list:
        raise ValueError("History is shorter than the training window.")

    oos_piece_list: list[pd.Series] = []
    refit_row_list: list[dict] = []
    for window_idx_int, test_start in enumerate(start_list):
        test_end = start_list[window_idx_int + 1] if window_idx_int + 1 < len(start_list) else None
        # *** CRITICAL*** training rows end strictly before the test window starts.
        train_mask = date_index < test_start
        if not design.anchored_bool:
            train_mask &= date_index >= test_start - pd.DateOffset(years=design.train_years_int)
        test_mask = date_index >= test_start
        if test_end is not None:
            test_mask &= date_index < test_end

        train_return_df = config_return_df.loc[train_mask]
        chosen_config = select_config_fn(train_return_df)
        test_return_ser = config_return_df.loc[test_mask, chosen_config]
        if test_return_ser.isna().any():
            raise ValueError(
                f"Config {chosen_config!r} has NaN returns in the test window starting {test_start.date()}; "
                "fill them explicitly (0.0 for flat) before the walk-forward."
            )
        oos_piece_list.append(test_return_ser)
        refit_row_list.append(
            {
                "test_start": test_start,
                "chosen_config": chosen_config,
                "is_sharpe_float": annualized_sharpe_float(train_return_df[chosen_config]),
            }
        )

    oos_return_ser = pd.concat(oos_piece_list)
    oos_return_ser.name = "oos_return_float"
    refit_df = pd.DataFrame(refit_row_list)
    oos_sharpe_float = annualized_sharpe_float(oos_return_ser)
    mean_is_sharpe_float = float(refit_df["is_sharpe_float"].mean())
    efficiency_float = oos_sharpe_float / mean_is_sharpe_float if mean_is_sharpe_float > 0.0 else float("nan")
    return WalkForwardResult(
        design=design,
        oos_return_ser=oos_return_ser,
        refit_df=refit_df,
        oos_sharpe_float=oos_sharpe_float,
        mean_is_sharpe_float=mean_is_sharpe_float,
        efficiency_float=efficiency_float,
    )


def design_sensitivity_df(
    config_return_df: pd.DataFrame,
    select_config_fn: Callable[[pd.DataFrame], object],
    design_tuple: tuple[WalkForwardDesign, ...] = DESIGN_GRID_TUPLE,
) -> pd.DataFrame:
    """One row per design: OOS Sharpe, efficiency and whether OOS Sharpe is positive.

    Designs whose training window is longer than the history are skipped and
    listed with NaN, so the share of positive designs is taken over runnable ones.
    """
    row_list = []
    for design in design_tuple:
        if not _refit_start_list(config_return_df.index, design):
            row_list.append(
                {"design_str": design.label_str, "oos_sharpe_float": np.nan, "efficiency_float": np.nan, "oos_positive_bool": np.nan}
            )
            continue
        result = run_walk_forward(config_return_df, select_config_fn, design)
        row_list.append(
            {
                "design_str": design.label_str,
                "oos_sharpe_float": result.oos_sharpe_float,
                "efficiency_float": result.efficiency_float,
                "oos_positive_bool": bool(result.oos_sharpe_float > 0.0),
            }
        )
    return pd.DataFrame(row_list)
