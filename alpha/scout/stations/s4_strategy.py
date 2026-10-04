"""Station S4: the strategy built in full reality (design section 9, S4).

On the in-sample period (up to the vault seal, 2022-12-30), with the parity engine's costs:
- the parameter surface over the registered grid, the plateau choice (centre of the best plateau, never the raw
  peak) and its plateau ratio (PASS >= 0.7, WARN 0.5-0.7, FAIL < 0.5); where the LIVE configuration sits;
- the luck band: decision offsets 0-15 sessions before month end (the design's 21 would drop months shorter than
  the offset; review 2026-10-02), min / median / max Sharpe; WARN if the median is below 70% of the best offset;
  the median is the number to plan with, the worst offset the planning case for drawdowns and the S8 monitor;
- one common in-sample window for every configuration (from the date all have started to the seal);
- cost stress: twice the costs plus 10 bp slippage per side (FAIL if net Sharpe <= 0) and the breakeven slippage
  per side (net Sharpe 0);
- the losing-streak test: the longest run of losing months against independent reorderings (WARN if p < 0.05);
- the metric set, turnover and exposure, the post-seal period (2023 on; seen, so contaminated), and the small-account
  run ($30K, whole shares) against an institutional one ($10M).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from alpha.scout.engines.weights import CostModel, WeightsResult
from alpha.scout.family import FamilyRunner
from alpha.scout.metrics import performance_dict, sharpe_float
from alpha.stats.selection import neighbourhood_median_vec, plateau_choice

SEAL_END_STR = "2022-12-30"
STRESS_EXTRA_SLIPPAGE_FLOAT = 0.0010
SMALL_CAPITAL_FLOAT, INSTITUTIONAL_CAPITAL_FLOAT = 30_000.0, 10_000_000.0


@dataclass
class S4Report:
    family_name_str: str
    grid_df: pd.DataFrame  # full-history daily returns, columns in grid order
    sharpe_ser: pd.Series  # in-sample Sharpe per configuration
    chosen_label_str: str
    live_label_str: str
    plateau_dict: dict
    live_dict: dict
    luck_dict: dict = field(default_factory=dict)  # per role ("chosen", "live")
    cost_dict: dict = field(default_factory=dict)
    streak_dict: dict = field(default_factory=dict)
    metric_dict: dict = field(default_factory=dict)
    account_dict: dict = field(default_factory=dict)
    check_list: list = field(default_factory=list)


def _in_sample(daily_ser: pd.Series, start_ts=None) -> pd.Series:
    """In-sample window: from the common start of the grid (every configuration trading) to the vault seal."""
    return daily_ser.loc[start_ts:SEAL_END_STR]


def _exposure_turnover(result: WeightsResult, close_df: pd.DataFrame | None) -> dict:
    trade_df = result.trade_df
    if trade_df.empty:
        return {"annual_turnover_float": 0.0}
    notional_ser = (trade_df["delta_float"].abs() * trade_df["price_float"]).groupby(pd.to_datetime(trade_df["date"])).sum()
    value_ser = result.total_value_ser.reindex(notional_ser.index).ffill()
    years_float = len(result.total_value_ser) / 252.0
    return {"annual_turnover_float": float((notional_ser / value_ser).sum() / years_float)}


def longest_losing_streak(month_return_vec: np.ndarray) -> int:
    longest_int, run_int = 0, 0
    for value_float in month_return_vec:
        run_int = run_int + 1 if value_float < 0 else 0
        longest_int = max(longest_int, run_int)
    return longest_int


def streak_test(daily_ser: pd.Series, draw_count_int: int = 5000, random_seed_int: int = 0) -> dict:
    month_vec = daily_ser.resample("ME").apply(lambda s: (1 + s).prod() - 1).to_numpy()
    observed_int = longest_losing_streak(month_vec)
    rng_obj = np.random.default_rng(random_seed_int)
    null_vec = np.array([longest_losing_streak(rng_obj.permutation(month_vec)) for _ in range(draw_count_int)])
    p_float = float((1 + np.sum(null_vec >= observed_int)) / (1 + draw_count_int))
    return {"longest_losing_months_int": observed_int, "independent_median_int": int(np.median(null_vec)), "p_float": p_float, "warn_bool": p_float < 0.05}


def breakeven_slippage(family: FamilyRunner, config_dict: dict, base_cost: CostModel, start_ts=None) -> float:
    """Slippage per side at which in-sample net Sharpe reaches zero (bisection, fees unchanged); inf if never."""
    def sharpe_at(slippage_float: float) -> float:
        return sharpe_float(_in_sample(family.run_config(config_dict, replace(base_cost, slippage_float=slippage_float)).daily_return_ser, start_ts))

    low_float, high_float = base_cost.slippage_float, 0.02
    if sharpe_at(high_float) > 0:
        return float("inf")
    if sharpe_at(low_float) <= 0:
        return low_float
    for _ in range(14):
        middle_float = 0.5 * (low_float + high_float)
        low_float, high_float = (middle_float, high_float) if sharpe_at(middle_float) > 0 else (low_float, middle_float)
    return 0.5 * (low_float + high_float)


def run_s4(family: FamilyRunner, tbill_daily_ser: pd.Series | None = None, cost_model: CostModel = CostModel()) -> S4Report:
    config_list = family.config_list()
    result_by_label_dict = {family.label_str(c): family.run_config(c, cost_model) for c in config_list}
    grid_df = pd.DataFrame({label_str: result.daily_return_ser for label_str, result in result_by_label_dict.items()})
    start_ts = grid_df.apply(lambda s: s.first_valid_index()).max()

    def in_sample(daily_ser: pd.Series) -> pd.Series:
        return _in_sample(daily_ser, start_ts)

    sharpe_ser = grid_df.loc[start_ts:SEAL_END_STR].apply(sharpe_float)
    choice = plateau_choice(sharpe_ser.to_numpy(), family.grid_shape_tuple)
    neighbourhood_vec = neighbourhood_median_vec(sharpe_ser.to_numpy(), family.grid_shape_tuple)
    chosen_label_str = sharpe_ser.index[choice.flat_index_int]
    live_label_str = family.label_str(family.live_config_dict)
    live_idx_int = list(sharpe_ser.index).index(live_label_str)
    plateau_ratio_float = choice.plateau_ratio_float
    plateau_dict = {
        "chosen_label_str": chosen_label_str,
        "chosen_sharpe_float": choice.own_sharpe_float,
        "neighbourhood_median_float": choice.neighbourhood_median_float,
        "peak_label_str": str(sharpe_ser.idxmax()),
        "peak_sharpe_float": float(sharpe_ser.max()),
        "plateau_ratio_float": plateau_ratio_float,
        "verdict_str": "PASS" if plateau_ratio_float >= 0.7 else ("WARN" if plateau_ratio_float >= 0.5 else "FAIL"),
        "grid_shape_tuple": family.grid_shape_tuple,
        "name_list": family.name_list,
        "common_start_str": str(start_ts.date()),
    }
    live_dict = {
        "label_str": live_label_str,
        "sharpe_float": float(sharpe_ser[live_label_str]),
        "neighbourhood_median_float": float(neighbourhood_vec[live_idx_int]),
        "rank_int": int((sharpe_ser > sharpe_ser[live_label_str]).sum() + 1),
        "grid_size_int": len(sharpe_ser),
        "is_chosen_bool": live_label_str == chosen_label_str,
    }
    report = S4Report(family.name_str, grid_df, sharpe_ser, chosen_label_str, live_label_str, plateau_dict, live_dict)

    role_config_dict = {"live": family.live_config_dict}
    if chosen_label_str != live_label_str:
        role_config_dict["chosen"] = config_list[choice.flat_index_int]
    for role_str, config_dict in role_config_dict.items():
        offset_sharpe_dict, offset_return_dict = {}, {}
        for offset_int in range(family.offset_count_int):
            daily_ser = family.run_config({**config_dict, "decision_offset_int": offset_int}, cost_model).daily_return_ser
            offset_sharpe_dict[offset_int] = sharpe_float(in_sample(daily_ser))
            offset_return_dict[offset_int] = daily_ser
        offset_sharpe_ser = pd.Series(offset_sharpe_dict)
        worst_offset_int = int(offset_sharpe_ser.idxmin())
        report.luck_dict[role_str] = {
            "sharpe_by_offset": offset_sharpe_dict,
            "min_float": float(offset_sharpe_ser.min()),
            "median_float": float(offset_sharpe_ser.median()),
            "max_float": float(offset_sharpe_ser.max()),
            "worst_offset_int": worst_offset_int,
            "worst_offset_return_ser": offset_return_dict[worst_offset_int],
            "warn_bool": bool(offset_sharpe_ser.median() < 0.7 * offset_sharpe_ser.max()),
        }

        stressed_cost = CostModel(
            slippage_float=2 * cost_model.slippage_float + STRESS_EXTRA_SLIPPAGE_FLOAT,
            fee_per_share_float=2 * cost_model.fee_per_share_float,
            min_fee_float=2 * cost_model.min_fee_float,
            dividend_withholding_float=cost_model.dividend_withholding_float,
        )
        base_result = result_by_label_dict[family.label_str(config_dict)]
        stressed_sharpe_float = sharpe_float(in_sample(family.run_config(config_dict, stressed_cost).daily_return_ser))
        gross_cost = CostModel(slippage_float=0.0, fee_per_share_float=0.0, min_fee_float=0.0, dividend_withholding_float=cost_model.dividend_withholding_float)
        report.cost_dict[role_str] = {
            "gross_sharpe_float": sharpe_float(in_sample(family.run_config(config_dict, gross_cost).daily_return_ser)),
            "net_sharpe_float": sharpe_float(in_sample(base_result.daily_return_ser)),
            "stressed_sharpe_float": stressed_sharpe_float,
            "breakeven_slippage_per_side_float": breakeven_slippage(family, config_dict, cost_model, start_ts),
            "fail_bool": not stressed_sharpe_float > 0,
            **_exposure_turnover(base_result, None),
        }
        in_sample_ser = in_sample(base_result.daily_return_ser)
        report.streak_dict[role_str] = streak_test(in_sample_ser)
        report.metric_dict[role_str] = {
            "in_sample": performance_dict(in_sample_ser, tbill_daily_ser),
            "post_seal": performance_dict(base_result.daily_return_ser.loc[SEAL_END_STR:].iloc[1:], tbill_daily_ser),
        }

    small_ser = in_sample(family.run_config(family.live_config_dict, cost_model, SMALL_CAPITAL_FLOAT).daily_return_ser)
    large_ser = in_sample(family.run_config(family.live_config_dict, cost_model, INSTITUTIONAL_CAPITAL_FLOAT).daily_return_ser)
    report.account_dict = {
        "small_capital_float": SMALL_CAPITAL_FLOAT,
        "small": performance_dict(small_ser),
        "institutional": performance_dict(large_ser),
    }
    report.check_list = [
        ("plateau ratio >= 0.7", plateau_dict["verdict_str"], f"{plateau_ratio_float:.2f}"),
        ("luck band median >= 70% of best offset (live)", "WARN" if report.luck_dict["live"]["warn_bool"] else "PASS",
         f"median {report.luck_dict['live']['median_float']:.2f} vs best {report.luck_dict['live']['max_float']:.2f}"),
        ("net Sharpe > 0 at twice the costs + 10 bp (live)", "FAIL" if report.cost_dict["live"]["fail_bool"] else "PASS",
         f"{report.cost_dict['live']['stressed_sharpe_float']:.2f}"),
        ("losing streak consistent with independence (live)", "WARN" if report.streak_dict["live"]["warn_bool"] else "PASS",
         f"{report.streak_dict['live']['longest_losing_months_int']} months, p {report.streak_dict['live']['p_float']:.2f}"),
    ]
    return report
