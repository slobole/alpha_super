"""Two-bucket capital templates for the fund product menu (no return inputs).

A product is one number - the return-engine share s of capital - applied to a
fixed template:

    book = s * ReturnBucket + (1 - s) * StabilizerBucket

- ReturnBucket: the line's return engines at equal capital. Inside an engine the
  capital is split between its primary and secondary sleeve (50/50 by default);
  the secondary is used only when both would hold at least the minimum pod
  weight, otherwise the whole engine sits in the primary.
- StabilizerBucket: fixed capital shares (CORE5 is the owner-designated anchor).
  A stabilizer below the minimum pod weight is folded into the anchor.

s is chosen so the book's ex-ante volatility (weekly covariance, annualised)
matches the product's target. Only covariance enters; returns never do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import construction


def template_weight_ser(
    spec_dict: dict,
    line_str: str,
    return_share_float: float,
    engine_share_override_dict: dict[str, float] | None = None,
    member_split_override_dict: dict[str, float] | None = None,
    stabilizer_override_dict: dict[str, float] | None = None,
    alias_swap_dict: dict[str, str] | None = None,
    disabled_secondary_set: set[str] | None = None,
) -> pd.Series:
    line_dict = spec_dict["lines"][line_str]
    minimum_float = float(spec_dict["minimum_pod_weight_float"])
    engine_list = list(line_dict["return_engines"])
    engine_share_dict = engine_share_override_dict or {e: 1.0 / len(engine_list) for e in engine_list}
    stabilizer_dict = stabilizer_override_dict or dict(line_dict["stabilizer_capital"])
    swap_dict = alias_swap_dict or {}
    disabled_set = disabled_secondary_set or set()
    weight_dict: dict[str, float] = {}

    def add(alias_str: str, weight_float: float) -> None:
        alias_str = swap_dict.get(alias_str, alias_str)
        weight_dict[alias_str] = weight_dict.get(alias_str, 0.0) + weight_float

    for engine_str in engine_list:
        engine_dict = spec_dict["engines"][engine_str]
        engine_capital_float = return_share_float * engine_share_dict[engine_str]
        if engine_capital_float <= 0:
            continue
        secondary_str = engine_dict.get("secondary_str")
        primary_split_float = (member_split_override_dict or {}).get(engine_str, 0.5)
        use_secondary_bool = (
            secondary_str is not None
            and engine_str not in disabled_set
            and engine_capital_float * min(primary_split_float, 1.0 - primary_split_float) >= minimum_float - 1e-12
        )
        if use_secondary_bool:
            add(engine_dict["primary_str"], engine_capital_float * primary_split_float)
            add(secondary_str, engine_capital_float * (1.0 - primary_split_float))
        else:
            add(engine_dict["primary_str"], engine_capital_float)

    stabilizer_capital_float = 1.0 - return_share_float
    if stabilizer_capital_float > 1e-12:
        anchor_str = line_dict["stabilizer_anchor_str"]
        total_float = sum(stabilizer_dict.values())
        for alias_str, share_float in stabilizer_dict.items():
            weight_float = stabilizer_capital_float * share_float / total_float
            add(anchor_str if (weight_float < minimum_float - 1e-12 and alias_str != anchor_str) else alias_str, weight_float)
    weight_ser = pd.Series(weight_dict)
    return weight_ser[weight_ser > 0]


def solve_share_for_target_volatility(
    spec_dict: dict,
    line_str: str,
    covariance_df: pd.DataFrame,
    target_volatility_float: float,
    **template_kwargs,
) -> tuple[float, pd.Series]:
    """Grid search over s in [0, 1] (step 0.001): the template whose ex-ante volatility is
    closest to the target. A grid, not a root finder, because the secondary-member and
    minimum-pod rules make the volatility curve step at a few points."""
    best_tuple = None
    for share_float in np.round(np.arange(0.0, 1.0000001, 0.001), 3):
        weight_ser = template_weight_ser(spec_dict, line_str, float(share_float), **template_kwargs)
        volatility_float = construction.book_volatility_float(covariance_df, weight_ser)
        gap_float = abs(volatility_float - target_volatility_float)
        if best_tuple is None or gap_float < best_tuple[0] - 1e-12:
            best_tuple = (gap_float, float(share_float), weight_ser)
    return best_tuple[1], best_tuple[2]


def equal_risk_return_bucket_ser(
    spec_dict: dict, line_str: str, covariance_df: pd.DataFrame, return_share_float: float
) -> pd.Series:
    """Alternative: return engines at equal RISK (Euler) instead of equal capital, same s."""
    line_dict = spec_dict["lines"][line_str]
    engine_list = list(line_dict["return_engines"])
    budget_dict: dict[str, float] = {}
    for engine_str in engine_list:
        engine_dict = spec_dict["engines"][engine_str]
        members_list = [engine_dict["primary_str"]] + ([engine_dict["secondary_str"]] if engine_dict.get("secondary_str") else [])
        for alias_str in members_list:
            budget_dict[alias_str] = 1.0 / len(engine_list) / len(members_list)
    budget_ser = pd.Series(budget_dict)
    bucket_ser = construction.risk_budget_weight_ser(covariance_df, budget_ser) * return_share_float
    stabilizer_ser = template_weight_ser(spec_dict, line_str, 0.0) * (1.0 - return_share_float)
    return bucket_ser.add(stabilizer_ser, fill_value=0.0)
