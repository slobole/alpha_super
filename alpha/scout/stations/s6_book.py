"""Station S6: value to the book (design section 9, S6).

- Spanning: monthly excess returns regressed on factor sets, Newey-West (lag 3) t of the intercept, alpha shown
  annualised, gross and net of costs (owner rule: always show factor alpha against QQQ, an ETF mix, and the mix
  plus trend). PASS if the net alpha t >= 2.0 in the fullest model.
  Fama-French factors are not stored offline; the ETF factors stand in for them (stated on the card).
- T-bill slot: the reference book with the candidate in its slot against the same book with T-bills in that slot;
  a paired stationary bootstrap (mean block 21 sessions) of the book Sharpe difference. PASS if
  P(candidate > T-bills) >= 0.80 and the point difference is > 0.
- Diversification: correlation with each book component and the market, returns in the crisis windows, and the
  candidate's mean return on the book's worst 5% of days.
- Capacity: per-trade weight / average daily dollar volume (63 sessions before the trade); the AUM at which the
  95th-percentile order stays at 1% of ADV, over the full history and over the last three years (today's volumes).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alpha.scout.metrics import sharpe_float
from alpha.stats.bootstrap import stationary_bootstrap_index_mat
from alpha.stats.newey_west import newey_west_mean_t_stat

CRISIS_WINDOW_DICT = {
    "GFC 2007-10 to 2009-03": ("2007-10-09", "2009-03-09"),
    "Euro / US downgrade 2011": ("2011-04-29", "2011-10-03"),
    "Q4 2018": ("2018-09-20", "2018-12-24"),
    "COVID crash 2020": ("2020-02-19", "2020-03-23"),
    "2022 bear": ("2022-01-03", "2022-10-12"),
}


def _monthly(daily_ser: pd.Series) -> pd.Series:
    """Compounded monthly return; a month with no data is NaN (not 0%: review 2026-10-02, a factor that did not exist
    yet entered the regression as a run of 0% months and inflated an alpha t from 1.2 to 2.9)."""
    return daily_ser.resample("ME").apply(lambda s: (1.0 + s.dropna()).prod() - 1.0 if s.notna().any() else np.nan)


def _ols_hac(y_vec: np.ndarray, x_mat: np.ndarray, lag_int: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and Newey-West standard errors (Bartlett)."""
    design_mat = np.column_stack([np.ones(len(y_vec)), x_mat])
    xtx_inv = np.linalg.inv(design_mat.T @ design_mat)
    beta_vec = xtx_inv @ design_mat.T @ y_vec
    score_mat = design_mat * (y_vec - design_mat @ beta_vec)[:, None]
    meat_mat = score_mat.T @ score_mat
    for lag_idx_int in range(1, lag_int + 1):
        weight_float = 1.0 - lag_idx_int / (lag_int + 1.0)
        cross_mat = score_mat[lag_idx_int:].T @ score_mat[:-lag_idx_int]
        meat_mat += weight_float * (cross_mat + cross_mat.T)
    covariance_mat = xtx_inv @ meat_mat @ xtx_inv
    return beta_vec, np.sqrt(np.diag(covariance_mat))


def spanning_table(net_daily_ser: pd.Series, gross_daily_ser: pd.Series, factor_daily_df: pd.DataFrame, tbill_daily_ser: pd.Series,
                   model_dict: dict[str, list[str]]) -> list[dict]:
    rf_month_ser = _monthly(tbill_daily_ser)
    factor_month_df = factor_daily_df.apply(_monthly).sub(rf_month_ser, axis=0)
    row_list = []
    for model_str, factor_list in model_dict.items():
        row_dict = {"model_str": model_str, "factor_list": factor_list}
        for kind_str, daily_ser in (("net", net_daily_ser), ("gross", gross_daily_ser)):
            y_ser = _monthly(daily_ser) - rf_month_ser
            frame = pd.concat([y_ser.rename("y"), factor_month_df[factor_list]], axis=1).dropna()
            beta_vec, se_vec = _ols_hac(frame["y"].to_numpy(), frame[factor_list].to_numpy())
            row_dict[f"{kind_str}_alpha_annual_float"] = float((1.0 + beta_vec[0]) ** 12 - 1.0)
            row_dict[f"{kind_str}_alpha_t_float"] = float(beta_vec[0] / se_vec[0])
            if kind_str == "net":
                row_dict["beta_dict"] = {name_str: float(b) for name_str, b in zip(factor_list, beta_vec[1:])}
                row_dict["months_int"] = len(frame)
        row_list.append(row_dict)
    return row_list


def tbill_slot_test(candidate_ser: pd.Series, book_weight_dict: dict[str, float], book_component_dict: dict[str, pd.Series],
                    slot_str: str, tbill_daily_ser: pd.Series, draw_count_int: int = 5000, random_seed_int: int = 0) -> dict:
    frame = pd.concat({name_str: ser for name_str, ser in book_component_dict.items()}, axis=1)
    frame[slot_str] = candidate_ser
    frame = frame.dropna()
    tbill_ser = tbill_daily_ser.reindex(frame.index).fillna(0.0)
    with_vec = sum(weight_float * frame[name_str] for name_str, weight_float in book_weight_dict.items()).to_numpy()
    without_vec = with_vec - book_weight_dict[slot_str] * (frame[slot_str].to_numpy() - tbill_ser.to_numpy())
    index_mat = stationary_bootstrap_index_mat(len(frame), draw_count_int, 21.0, len(frame), random_seed_int)

    def sharpe_rows(value_vec: np.ndarray) -> np.ndarray:
        sample_mat = value_vec[index_mat]
        return sample_mat.mean(axis=1) / sample_mat.std(axis=1, ddof=1) * np.sqrt(252.0)

    difference_vec = sharpe_rows(with_vec) - sharpe_rows(without_vec)
    point_float = sharpe_float(pd.Series(with_vec)) - sharpe_float(pd.Series(without_vec))
    probability_float = float(np.mean(difference_vec > 0))
    return {
        "book_str": " + ".join(f"{w:.0%} {n}" for n, w in book_weight_dict.items()),
        "start_str": str(frame.index[0].date()),
        "book_sharpe_with_float": sharpe_float(pd.Series(with_vec)),
        "book_sharpe_tbills_float": sharpe_float(pd.Series(without_vec)),
        "point_difference_float": point_float,
        "probability_better_float": probability_float,
        "verdict_str": "PASS" if probability_float >= 0.80 and point_float > 0 else "FAIL",
    }


def _window_return(daily_ser: pd.Series, start_str: str, end_str: str) -> float:
    """Compounded return over a window, NaN unless the series covers at least 90% of its business days."""
    window_ser = daily_ser.loc[start_str:end_str].dropna()
    if len(window_ser) < 0.9 * len(pd.bdate_range(start_str, end_str)) * 0.96:  # 0.96: market holidays
        return float("nan")
    return float((1 + window_ser).prod() - 1)


def diversification(candidate_ser: pd.Series, reference_dict: dict[str, pd.Series], book_ser: pd.Series) -> dict:
    """Pairwise: each reference is aligned with the candidate on its own overlap (a late-starting reference does not
    cut the candidate's history)."""
    crisis_list = []
    for window_str, (start_str, end_str) in CRISIS_WINDOW_DICT.items():
        row_dict = {"window_str": window_str, "candidate": _window_return(candidate_ser, start_str, end_str)}
        if not np.isfinite(row_dict["candidate"]):
            continue
        row_dict.update({name_str: _window_return(ser, start_str, end_str) for name_str, ser in reference_dict.items()})
        crisis_list.append(row_dict)
    book_frame = pd.concat({"candidate": candidate_ser, "book": book_ser}, axis=1).dropna()
    worst_mask = book_frame["book"] <= book_frame["book"].quantile(0.05)
    return {
        "correlation_dict": {name_str: float(pd.concat([candidate_ser, ser], axis=1).dropna().corr().iloc[0, 1]) for name_str, ser in reference_dict.items()},
        "correlation_start_dict": {name_str: str(pd.concat([candidate_ser, ser], axis=1).dropna().index[0].date()) for name_str, ser in reference_dict.items()},
        "crisis_list": crisis_list,
        "mean_on_book_worst_5pct_float": float(book_frame.loc[worst_mask, "candidate"].mean()),
        "book_mean_on_worst_5pct_float": float(book_frame.loc[worst_mask, "book"].mean()),
    }


def capacity(trade_df: pd.DataFrame, total_value_ser: pd.Series, dollar_volume_df: pd.DataFrame, participation_float: float = 0.01) -> dict:
    """AUM at which the 95th-percentile order is `participation_float` of ADV (63 sessions before the trade)."""
    adv_df = dollar_volume_df.rolling(63, min_periods=20).mean().shift(1)
    rebalance_df = trade_df[trade_df["kind_str"] == "rebalance"].copy()
    rebalance_df["date"] = pd.to_datetime(rebalance_df["date"])
    value_ser = total_value_ser.shift(1).reindex(rebalance_df["date"]).to_numpy()
    rebalance_df["weight_float"] = (rebalance_df["delta_float"].abs() * rebalance_df["price_float"]).to_numpy() / value_ser
    adv_vec = np.array([adv_df.at[d, a] if (d in adv_df.index and a in adv_df.columns) else np.nan for d, a in zip(rebalance_df["date"], rebalance_df["asset"])])
    rebalance_df["aum_limit_float"] = participation_float * adv_vec / rebalance_df["weight_float"].to_numpy()
    rebalance_df = rebalance_df[np.isfinite(rebalance_df["aum_limit_float"]) & (rebalance_df["weight_float"] > 1e-4)]
    recent_df = rebalance_df[rebalance_df["date"] >= rebalance_df["date"].max() - pd.DateOffset(years=3)]

    def limit(frame: pd.DataFrame) -> float:
        return float(np.quantile(frame["aum_limit_float"], 0.05)) if len(frame) else float("nan")

    # The binding asset is the one of the trade at the 5th percentile (the number quoted), not the single worst trade.
    binding_row = recent_df.sort_values("aum_limit_float").iloc[int(0.05 * (len(recent_df) - 1))] if len(recent_df) else None
    return {
        "participation_float": participation_float,
        "full_history_aum_float": limit(rebalance_df),
        "recent_3y_aum_float": limit(recent_df),
        "binding_asset_str": str(binding_row["asset"]) if binding_row is not None else "",
        "trade_count_int": len(rebalance_df),
    }


@dataclass
class S6Report:
    spanning_list: list
    slot_dict: dict
    diversification_dict: dict
    capacity_dict: dict
    check_list: list = field(default_factory=list)


def finish_s6(spanning_list, slot_dict, diversification_dict, capacity_dict) -> S6Report:
    fullest_row = spanning_list[-1]
    report = S6Report(spanning_list, slot_dict, diversification_dict, capacity_dict)
    report.check_list = [
        (f"net alpha t >= 2 ({fullest_row['model_str']})", "PASS" if fullest_row["net_alpha_t_float"] >= 2.0 else "FAIL",
         f"{fullest_row['net_alpha_annual_float']:+.1%} a year, t {fullest_row['net_alpha_t_float']:.2f}"),
        ("T-bill slot: P(candidate > T-bills) >= 0.80", slot_dict["verdict_str"],
         f"P {slot_dict['probability_better_float']:.2f}, book Sharpe {slot_dict['book_sharpe_with_float']:.2f} vs {slot_dict['book_sharpe_tbills_float']:.2f}"),
        ("capacity (1% of ADV, last 3 years)", "INFO", f"${capacity_dict['recent_3y_aum_float'] / 1e6:,.1f}M (binding: {capacity_dict['binding_asset_str']})"),
    ]
    return report


__all__ = ["capacity", "diversification", "finish_s6", "newey_west_mean_t_stat", "spanning_table", "tbill_slot_test"]
