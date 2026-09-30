"""Run every candidate gate on one synthetic history (PROTOCOL.md, Part A)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha.scout.trials import effective_trial_count
from alpha.stats.mcpt import mcpt, volatility_strata_vec
from alpha.stats.pbo import probability_of_backtest_overfitting
from alpha.stats.psr_dsr import deflated_sharpe_ratio
from alpha.stats.selection import column_sharpe_vec, make_plateau_selector, plateau_choice
from alpha.stats.walk_forward import REGISTERED_DESIGN, design_sensitivity_df, run_walk_forward

from synth import GRID_SHAPE_TUPLE, config_return_mat

MCPT_PERMUTATION_COUNT_INT = 200
MCPT_PASS_FLOAT = 0.01
DSR_PASS_FLOAT = 0.95
WFA_EFFICIENCY_FLOAT = 0.5
WFA_DESIGN_PASS_INT = 6
PBO_PASS_FLOAT = 0.20
NAIVE_T_FLOAT = 2.0


def _selected_sharpe_of_search(return_mat: np.ndarray) -> float:
    """The whole search: build the 21 configurations from the (possibly permuted) returns, plateau-select."""
    config_mat = config_return_mat(return_mat[:, 0])
    sharpe_vec = column_sharpe_vec(config_mat)
    return float(sharpe_vec[plateau_choice(sharpe_vec, GRID_SHAPE_TUPLE).flat_index_int])


def evaluate_history(return_vec, seed_int: int) -> dict:
    return_arr = np.asarray(return_vec, dtype=float)
    config_mat = config_return_mat(return_arr)
    sharpe_vec = column_sharpe_vec(config_mat)
    choice = plateau_choice(sharpe_vec, GRID_SHAPE_TUPLE)
    chosen_vec = config_mat[:, choice.flat_index_int]
    years_float = chosen_vec.size / 252.0
    result_dict = {
        "chosen_index_int": choice.flat_index_int,
        "chosen_sharpe_float": choice.own_sharpe_float,
        "plateau_ratio_float": choice.plateau_ratio_float,
        "naive_t_float": choice.own_sharpe_float * np.sqrt(years_float),
    }

    # DSR: family = the 21 configurations.
    config_df = pd.DataFrame(config_mat, index=pd.bdate_range("2000-01-03", periods=config_mat.shape[0]))
    per_period_sharpe_vec = sharpe_vec / np.sqrt(252.0)
    n_eff_float = effective_trial_count(config_df.loc[:, config_df.std() > 0])
    dsr = deflated_sharpe_ratio(chosen_vec, float(np.var(per_period_sharpe_vec, ddof=1)), n_eff_float)
    result_dict.update({"dsr_float": dsr.deflated_sharpe_float, "n_eff_float": n_eff_float})

    # Walk-forward: registered design + 8-design sensitivity, plateau selector.
    selector_fn = make_plateau_selector(GRID_SHAPE_TUPLE)
    registered = run_walk_forward(config_df, selector_fn, REGISTERED_DESIGN)
    sensitivity_df = design_sensitivity_df(config_df, selector_fn)
    positive_count_int = int(sensitivity_df["oos_positive_bool"].fillna(False).astype(bool).sum())
    result_dict.update(
        {
            "wf_efficiency_float": registered.efficiency_float,
            "wf_oos_sharpe_float": registered.oos_sharpe_float,
            "wf_positive_designs_int": positive_count_int,
        }
    )

    # PBO with the same plateau selection.
    pbo = probability_of_backtest_overfitting(
        config_mat, 10, select_fn=lambda train_sharpe_vec: plateau_choice(train_sharpe_vec, GRID_SHAPE_TUPLE).flat_index_int
    )
    result_dict["pbo_float"] = pbo.pbo_float

    # MCPT: plain and volatility-stratified date shuffles of the underlying returns.
    plain = mcpt(_selected_sharpe_of_search, return_arr[:, None], MCPT_PERMUTATION_COUNT_INT, seed_int)
    strata_vec = volatility_strata_vec(return_arr[:, None], 63, 3)
    stratified = mcpt(_selected_sharpe_of_search, return_arr[:, None], MCPT_PERMUTATION_COUNT_INT, seed_int, strata_vec)
    result_dict.update({"mcpt_plain_p_float": plain.p_value_float, "mcpt_stratified_p_float": stratified.p_value_float})

    # Pass flags, exactly as frozen in PROTOCOL.md.
    result_dict.update(
        {
            "pass_mcpt_plain_bool": plain.p_value_float <= MCPT_PASS_FLOAT,
            "pass_mcpt_stratified_bool": stratified.p_value_float <= MCPT_PASS_FLOAT,
            "pass_dsr_bool": dsr.deflated_sharpe_float >= DSR_PASS_FLOAT,
            "pass_wf_bool": bool(
                np.isfinite(registered.efficiency_float)
                and registered.efficiency_float >= WFA_EFFICIENCY_FLOAT
                and registered.oos_sharpe_float > 0.0
                and positive_count_int >= WFA_DESIGN_PASS_INT
            ),
            "pass_pbo_bool": pbo.pbo_float <= PBO_PASS_FLOAT,
            "pass_naive_bool": result_dict["naive_t_float"] >= NAIVE_T_FLOAT,
        }
    )
    return result_dict
