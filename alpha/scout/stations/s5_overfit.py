"""Station S5: is the search manufacturing winners? (design section 9, S5, as amended by A5, A8 and A9).

- The GATE is the MCPT of the whole search (plateau selection included), p <= 0.05, one or more components:
  - ETF and timing families: plain date-row shuffle, score = Sharpe of the daily active return over the
    volatility-targeted equal-weight baseline (A9: the only score that held size on volatility-timed families);
  - point-in-time stock families: the per-asset null, score = Sharpe of the daily active return over the baseline
    (A8). A cross-sectional ranking component with 0.025 < p <= 0.05 passes as "marginal".
  The MCPT runs outside this module (it needs the family's fast search); its results are passed in.
- WARN: the correlation-aware DSR, an exact null p-value of the selected Sharpe given this grid's correlation and the
  family's prior trials (retro registrations: max(documented, 50)).
- Diagnostics: walk-forward (registered design and the 8-design strip) and PBO, both with plateau selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alpha.stats.pbo import probability_of_backtest_overfitting
from alpha.stats.psr_dsr import null_selected_sharpe_draws, null_selected_sharpe_p_value
from alpha.stats.selection import (
    make_plateau_selector,
    plateau_choice,
    plateau_choice_index_mat,
)
from alpha.stats.walk_forward import (
    REGISTERED_DESIGN,
    design_sensitivity_df,
    run_walk_forward,
)

MCPT_ALPHA_FLOAT, MARGINAL_FLOAT = 0.05, 0.025


@dataclass
class McptComponent:
    name_str: str
    null_str: str  # "date shuffle" or "per-asset"
    score_str: str
    observed_float: float
    null_score_vec: np.ndarray
    p_value_float: float
    ranking_family_bool: bool = False  # a cross-sectional ranking component (marginal band applies)
    note_str: str = ""

    @property
    def verdict_str(self) -> str:
        if self.p_value_float > MCPT_ALPHA_FLOAT:
            return "FAIL"
        if self.ranking_family_bool and self.p_value_float > MARGINAL_FLOAT:
            return "PASS (marginal)"
        return "PASS"


@dataclass
class S5Report:
    mcpt_list: list
    dsr_dict: dict = field(default_factory=dict)
    walk_forward_dict: dict = field(default_factory=dict)
    pbo_dict: dict = field(default_factory=dict)
    check_list: list = field(default_factory=list)


def run_s5(in_sample_grid_df: pd.DataFrame, grid_shape_tuple: tuple, chosen_label_str: str, live_label_str: str,
           mcpt_list: list[McptComponent], prior_trial_count_int: int) -> S5Report:
    report = S5Report(mcpt_list=mcpt_list)
    # One common window: every configuration has started (zero-filling the late starters would penalise them).
    filled_df = in_sample_grid_df.loc[in_sample_grid_df.apply(lambda s: s.first_valid_index()).max() :].fillna(0.0)

    # WARN: correlation-aware DSR, exact null p-value.
    per_period_sharpe_ser = filled_df.mean() / filled_df.std(ddof=1)
    correlation_mat = np.nan_to_num(filled_df.corr().to_numpy(), nan=0.0)
    np.fill_diagonal(correlation_mat, 1.0)
    null_draw_vec = null_selected_sharpe_draws(
        correlation_mat, len(filled_df), select_index_fn=lambda draw_mat: plateau_choice_index_mat(draw_mat, grid_shape_tuple),
        prior_independent_trial_count_int=prior_trial_count_int, draw_count_int=20000, random_seed_int=0,
    )
    for role_str, label_str in (("chosen", chosen_label_str), ("live", live_label_str)):
        p_float = null_selected_sharpe_p_value(float(per_period_sharpe_ser[label_str]), null_draw_vec)
        report.dsr_dict[role_str] = {"label_str": label_str, "p_value_float": p_float, "verdict_str": "PASS" if p_float <= 0.05 else "WARN"}
    report.dsr_dict["prior_trial_count_int"] = prior_trial_count_int
    report.dsr_dict["null_median_annual_sharpe_float"] = float(np.median(null_draw_vec) * np.sqrt(252.0))

    # Diagnostics: walk-forward and PBO, plateau selection.
    selector_fn = make_plateau_selector(grid_shape_tuple)
    registered = run_walk_forward(filled_df, selector_fn, REGISTERED_DESIGN)
    sensitivity_df = design_sensitivity_df(filled_df, selector_fn)
    runnable_df = sensitivity_df.dropna(subset=["oos_sharpe_float"])
    report.walk_forward_dict = {
        "oos_sharpe_float": registered.oos_sharpe_float,
        "efficiency_float": registered.efficiency_float,
        "oos_return_ser": registered.oos_return_ser,
        "refit_df": registered.refit_df,
        "design_df": sensitivity_df,
        "positive_design_share_float": float(runnable_df["oos_positive_bool"].astype(bool).mean()) if len(runnable_df) else float("nan"),
    }
    pbo = probability_of_backtest_overfitting(filled_df.to_numpy(), 10, select_fn=lambda sharpe_vec: plateau_choice(sharpe_vec, grid_shape_tuple).flat_index_int)
    report.pbo_dict = {"pbo_float": pbo.pbo_float, "logit_vec": pbo.logit_vec}

    gate_str = "PASS" if all(c.verdict_str.startswith("PASS") for c in mcpt_list) else "FAIL"
    if gate_str == "PASS" and any(c.verdict_str == "PASS (marginal)" for c in mcpt_list):
        gate_str = "PASS (marginal)"
    report.check_list = [
        ("MCPT of the whole search (gate)", gate_str, "; ".join(f"{c.name_str} p {c.p_value_float:.3f}" for c in mcpt_list)),
        ("correlation-aware DSR p <= 0.05, live configuration (warn)", report.dsr_dict["live"]["verdict_str"],
         f"p {report.dsr_dict['live']['p_value_float']:.3f} (plateau choice p {report.dsr_dict['chosen']['p_value_float']:.3f})"),
        ("walk-forward (diagnostic)", "INFO", (f"OOS Sharpe {registered.oos_sharpe_float:.2f}, efficiency {registered.efficiency_float:.2f}, "
                                               f"{report.walk_forward_dict['positive_design_share_float']:.0%} of designs positive")),
        ("PBO (diagnostic)", "INFO", f"{pbo.pbo_float:.2f}"),
    ]
    return report
