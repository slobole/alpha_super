"""Robustness diagnostics (amendment A15, 2026-10-02): four printed checks from a concept review of a futures thesis
(github.com/Lucas-Joly-GH/trends-research, tests T23, T24, T30 and its Henriksson-Merton / Treynor-Mazuy battery).

None of them is a gate and none changes a grade (D20: a test gates only after a calibration shows it adds something).
Each returns a verdict word with a rule fixed here before any pod was run.

1. Contribution concentration (S4). Exact per-asset P&L from the weights engine (`asset_pnl_df`), as a share of the
   value at the previous close, so the contributions of a session add up to its return:
       c_i(t) = pnl_i(t) / V(t-1),   sum_i c_i(t) = r(t)
   Over the evaluation window: each asset's summed contribution, the share of the top 1 / 5 / 10 assets, the effective
   number of winners (sum c+)^2 / sum (c+)^2, and the Sharpe of the return with the top-k assets' contributions taken
   out (their capital assumed idle, an upper bound on the dependence: a real rule would have held something else).
   Verdict: CONCENTRATED if removing the top asset (an ETF pod, <= 20 traded assets) or the top 1% of names (a stock
   pod) leaves less than half the Sharpe; SPREAD otherwise; NO EDGE when the full Sharpe is <= 0.
2. Market-timing convexity (S6). Monthly excess returns on the market's excess return, Newey-West (lag 3):
       Henriksson-Merton  y = a + b x + g max(x, 0)   (down beta b, up beta b + g)
       Treynor-Mazuy      y = a + b x + c x^2
   Verdict per market: CONVEX if the HM g has t >= 2, CONCAVE if t <= -2, LINEAR otherwise; INSUFFICIENT with fewer than
   24 months or fewer than 6 down or 6 up months.
3. Component ablation (S4). Each registered component is switched off alone, then cumulatively in a registered order
   (overlays first, the core last). Per component: the Sharpe change and P(live Sharpe > ablated Sharpe) from a paired
   stationary bootstrap (mean block 21 sessions, 2,000 draws). Verdict: NO EVIDENCE if P < 0.50; otherwise, if P >= 0.80
   and the live Sharpe is higher, EARNS ITS PLACE when the difference is at least 0.05 and SMALL below that (two nearly
   identical series make a tiny difference "certain": materiality bar added after the first TAA smoke run, where
   removing the DTB3 hurdle cost 0.02 of Sharpe at P 0.86); UNCLEAR otherwise; NEVER BINDS when the switch leaves the
   returns unchanged. Idle cash: the runner credits it at the T-bill rate (the engine pays 0%, which would make every
   cash-heavy ablation look worse); `alt_run_fn` adds the 0%-cash Sharpe as a second column. The minimum spec is the deepest
   cumulative step whose Sharpe, and every step before it, stays at >= 80% of the live Sharpe.
4. Random-parameter percentile (S4/S5). N configurations drawn from a registered box of plausible values (wider than
   the S4 grid). Reported: the share of draws at or above the live Sharpe, the median and 10th percentile of the draws.
   Verdict: ROBUST TO VALUES if the median >= 70% of live and the 10th percentile > 0; DEPENDS ON VALUES if the median
   < 50% of live; PARTLY otherwise.

All Sharpe ratios are the S4 definition (zero risk-free rate, net of the house parity costs) over one fixed evaluation
window per pod, which starts after the longest warm-up in that pod's random box and ends at the vault seal.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alpha.scout.engines.weights import WeightsResult
from alpha.scout.metrics import performance_dict, sharpe_float
from alpha.scout.stations.s6_book import _monthly, _ols_hac
from alpha.stats.bootstrap import stationary_bootstrap_index_mat

ETF_POD_MAX_ASSET_INT = 20
STRIP_K_TUPLE = (1, 3, 5, 10)
CONCENTRATION_SHARE_FLOAT = 0.5
EARNS_PROBABILITY_FLOAT, NO_EVIDENCE_PROBABILITY_FLOAT, MATERIAL_SHARPE_FLOAT = 0.80, 0.50, 0.05
MIN_SPEC_SHARE_FLOAT = 0.80
ROBUST_MEDIAN_SHARE_FLOAT, DEPENDS_MEDIAN_SHARE_FLOAT = 0.70, 0.50
BOOTSTRAP_DRAW_INT, BOOTSTRAP_BLOCK_FLOAT = 2000, 21.0
MIN_TIMING_MONTH_INT, MIN_TIMING_SIDE_MONTH_INT = 24, 6  # fewer months, or fewer down / up months: INSUFFICIENT


def _window(daily_ser: pd.Series, start_str: str, end_str: str) -> pd.Series:
    return daily_ser.loc[start_str:end_str]


# ---------------------------------------------------------------- 1. contribution concentration
def contribution_dict(result: WeightsResult, start_str: str, end_str: str) -> dict:
    """Where the return came from (see the module docstring, item 1)."""
    if result.asset_pnl_df is None:
        raise ValueError("The engine result carries no per-asset P&L.")
    # V(t-1) recovered exactly from the engine's own return: V(t) / (1 + r(t)).
    previous_total_ser = result.total_value_ser / (1.0 + result.daily_return_ser)
    contribution_df = _window(result.asset_pnl_df.div(previous_total_ser, axis=0), start_str, end_str)
    daily_ser = _window(result.daily_return_ser, start_str, end_str)
    if not np.allclose(contribution_df.sum(axis=1).to_numpy(), daily_ser.to_numpy(), atol=1e-12):
        raise ValueError("Per-asset contributions do not add up to the daily return.")
    traded_list = [a for a in contribution_df.columns if contribution_df[a].abs().sum() > 0.0]
    if not traded_list:
        return {"window_str": f"{daily_ser.index[0].date()} to {daily_ser.index[-1].date()}", "pod_kind_str": "none", "traded_count_int": 0,
                "total_contribution_float": 0.0, "contribution_ser": pd.Series(dtype=float), "top_share_dict": {}, "effective_winner_float": float("nan"),
                "positive_asset_share_float": float("nan"), "full_sharpe_float": float("nan"), "strip_list": [], "key_k_int": 0, "year_list": [],
                "verdict_str": "NO TRADES"}
    contribution_ser = contribution_df[traded_list].sum().sort_values(ascending=False)
    total_float = float(contribution_ser.sum())
    positive_ser = contribution_ser[contribution_ser > 0]
    effective_winner_float = float(positive_ser.sum() ** 2 / (positive_ser ** 2).sum()) if len(positive_ser) else float("nan")
    full_sharpe_float = sharpe_float(daily_ser)

    def stripped(k_int: int) -> dict:
        top_list = list(contribution_ser.index[:k_int])
        remaining_ser = daily_ser - contribution_df[top_list].sum(axis=1)
        return {"k_int": k_int, "asset_list": top_list, "sharpe_float": sharpe_float(remaining_ser),
                "annual_mean_float": float(remaining_ser.mean() * 252.0)}

    etf_pod_bool = len(traded_list) <= ETF_POD_MAX_ASSET_INT
    key_k_int = 1 if etf_pod_bool else max(1, int(np.ceil(0.01 * len(traded_list))))
    # k = 1 (and the key k) always runs, even when it removes every traded asset (a one-asset book: nothing is left).
    strip_k_list = sorted({k for k in STRIP_K_TUPLE if k < len(traded_list)} | {key_k_int})
    strip_list = [stripped(k) for k in strip_k_list]
    key_dict = next(s for s in strip_list if s["k_int"] == key_k_int)
    # A NaN Sharpe left (no return or no variance remains) counts as concentrated.
    concentrated_bool = not (key_dict["sharpe_float"] >= CONCENTRATION_SHARE_FLOAT * full_sharpe_float)

    def top_share(k_int: int) -> float:
        return float(contribution_ser.iloc[:k_int].sum() / total_float) if total_float > 0 else float("nan")

    year_rows = []
    for year_int, year_df in contribution_df[traded_list].groupby(contribution_df.index.year):
        year_ser = year_df.sum()
        held_bool = bool(year_df.abs().to_numpy().sum() > 0.0)
        year_rows.append({"year_int": int(year_int), "return_float": float(year_ser.sum()), "top_asset_str": str(year_ser.idxmax()) if held_bool else "-",
                          "top_contribution_float": float(year_ser.max()) if held_bool else 0.0})
    return {
        "window_str": f"{daily_ser.index[0].date()} to {daily_ser.index[-1].date()}",
        "pod_kind_str": "ETF" if etf_pod_bool else "stock",
        "traded_count_int": len(traded_list),
        "total_contribution_float": total_float,
        "contribution_ser": contribution_ser,
        "top_share_dict": {k: top_share(k) for k in (1, 5, 10) if k <= len(traded_list)},
        "effective_winner_float": effective_winner_float,
        "positive_asset_share_float": float((contribution_ser > 0).mean()),
        "full_sharpe_float": full_sharpe_float,
        "strip_list": strip_list,
        "key_k_int": key_k_int,
        "year_list": year_rows,
        "verdict_str": "NO EDGE" if not full_sharpe_float > 0 else ("CONCENTRATED" if concentrated_bool else "SPREAD"),
    }


# ---------------------------------------------------------------- 2. market-timing convexity
def timing_dict(net_daily_ser: pd.Series, market_daily_dict: dict[str, pd.Series], tbill_daily_ser: pd.Series,
                start_str: str, end_str: str) -> list[dict]:
    """Henriksson-Merton and Treynor-Mazuy on monthly excess returns (module docstring, item 2)."""
    rf_month_ser = _monthly(tbill_daily_ser)
    pod_month_ser = _monthly(_window(net_daily_ser, start_str, end_str)) - rf_month_ser
    row_list = []
    for market_str, market_ser in market_daily_dict.items():
        frame = pd.concat({"y": pod_month_ser, "x": _monthly(market_ser) - rf_month_ser}, axis=1).dropna()
        down_count_int, up_count_int = int((frame["x"] < 0).sum()), int((frame["x"] >= 0).sum())
        if len(frame) < MIN_TIMING_MONTH_INT or min(down_count_int, up_count_int) < MIN_TIMING_SIDE_MONTH_INT:
            row_list.append({"market_str": market_str, "month_count_int": len(frame), "verdict_str": "INSUFFICIENT",
                             **{k: float("nan") for k in ("hm_alpha_annual_float", "hm_alpha_t_float", "down_beta_float", "up_beta_float", "hm_gamma_float",
                                                          "hm_gamma_t_float", "tm_c_float", "tm_c_t_float", "down_month_mean_float",
                                                          "market_down_month_mean_float", "up_month_mean_float", "market_up_month_mean_float")}})
            continue
        y_vec, x_vec = frame["y"].to_numpy(), frame["x"].to_numpy()
        hm_beta_vec, hm_se_vec = _ols_hac(y_vec, np.column_stack([x_vec, np.maximum(x_vec, 0.0)]))
        tm_beta_vec, tm_se_vec = _ols_hac(y_vec, np.column_stack([x_vec, x_vec ** 2]))
        gamma_t_float = float(hm_beta_vec[2] / hm_se_vec[2])
        down_ser, up_ser = frame.loc[frame["x"] < 0], frame.loc[frame["x"] >= 0]
        row_list.append({
            "market_str": market_str, "month_count_int": len(frame),
            "start_str": str(frame.index[0].date()), "end_str": str(frame.index[-1].date()),
            "hm_alpha_annual_float": float(hm_beta_vec[0] * 12.0), "hm_alpha_t_float": float(hm_beta_vec[0] / hm_se_vec[0]),
            "down_beta_float": float(hm_beta_vec[1]), "up_beta_float": float(hm_beta_vec[1] + hm_beta_vec[2]),
            "hm_gamma_float": float(hm_beta_vec[2]), "hm_gamma_t_float": gamma_t_float,
            "tm_c_float": float(tm_beta_vec[2]), "tm_c_t_float": float(tm_beta_vec[2] / tm_se_vec[2]),
            "down_month_mean_float": float(down_ser["y"].mean()), "market_down_month_mean_float": float(down_ser["x"].mean()),
            "up_month_mean_float": float(up_ser["y"].mean()), "market_up_month_mean_float": float(up_ser["x"].mean()),
            "verdict_str": "CONVEX" if gamma_t_float >= 2.0 else ("CONCAVE" if gamma_t_float <= -2.0 else "LINEAR"),
        })
    return row_list


# ---------------------------------------------------------------- 3. component ablation
@dataclass(frozen=True)
class AblationStep:
    name_str: str  # what is switched off
    override_dict: dict  # spec config fields that switch it off
    note_str: str = ""
    cumulative_bool: bool = True  # False: a single drop only (e.g. it would empty the book combined with the others)


def paired_sharpe_probability(live_ser: pd.Series, other_ser: pd.Series, random_seed_int: int = 0) -> float:
    """P(Sharpe(live) > Sharpe(other)) under a paired stationary bootstrap of the two daily return series."""
    frame = pd.concat({"a": live_ser, "b": other_ser}, axis=1).dropna()
    index_mat = stationary_bootstrap_index_mat(len(frame), BOOTSTRAP_DRAW_INT, BOOTSTRAP_BLOCK_FLOAT, len(frame), random_seed_int)

    def sharpe_rows(value_vec: np.ndarray) -> np.ndarray:
        sample_mat = value_vec[index_mat]
        sd_vec = sample_mat.std(axis=1, ddof=1)
        return np.where(sd_vec > 0, sample_mat.mean(axis=1) / np.where(sd_vec > 0, sd_vec, 1.0) * np.sqrt(252.0), 0.0)

    return float(np.mean(sharpe_rows(frame["a"].to_numpy()) > sharpe_rows(frame["b"].to_numpy())))


def _row(name_str: str, daily_ser: pd.Series, live_ser: pd.Series, live_sharpe_float: float) -> dict:
    metric_dict = performance_dict(daily_ser)
    sharpe_value_float = metric_dict.get("sharpe_float", float("nan"))
    return {"name_str": name_str, "sharpe_float": sharpe_value_float, "delta_float": live_sharpe_float - sharpe_value_float,
            "cagr_float": metric_dict.get("cagr_float", float("nan")), "max_drawdown_float": metric_dict.get("max_drawdown_float", float("nan")),
            "probability_live_better_float": paired_sharpe_probability(live_ser, daily_ser)}


def ablation_dict(run_fn: Callable[[dict], pd.Series], step_list: list[AblationStep], start_str: str, end_str: str,
                  alt_run_fn: Callable[[dict], pd.Series] | None = None) -> dict:
    """run_fn(override_dict) -> net daily returns of the live rule with those spec fields overridden ({} = live);
    alt_run_fn, if given, the same under another cash convention (its Sharpe is printed, not judged)."""
    live_ser = _window(run_fn({}), start_str, end_str)
    live_sharpe_float = sharpe_float(live_ser)

    def alt_sharpe(override_dict: dict) -> float:
        return sharpe_float(_window(alt_run_fn(override_dict), start_str, end_str)) if alt_run_fn is not None else float("nan")

    single_list = []
    for step in step_list:
        step_ser = _window(run_fn(step.override_dict), start_str, end_str)
        row = _row(step.name_str, step_ser, live_ser, live_sharpe_float)
        row["alt_sharpe_float"] = alt_sharpe(step.override_dict)
        probability_float = row["probability_live_better_float"]
        if step_ser.equals(live_ser):
            row["verdict_str"] = "NEVER BINDS"
        elif probability_float < NO_EVIDENCE_PROBABILITY_FLOAT:
            row["verdict_str"] = "NO EVIDENCE"
        elif probability_float >= EARNS_PROBABILITY_FLOAT and row["delta_float"] > 0:
            row["verdict_str"] = "EARNS ITS PLACE" if row["delta_float"] >= MATERIAL_SHARPE_FLOAT else "SMALL"
        else:
            row["verdict_str"] = "UNCLEAR"
        row["note_str"] = step.note_str
        single_list.append(row)
    cumulative_list, override_dict, label_list = [], {}, []
    for step in (s for s in step_list if s.cumulative_bool):
        override_dict, label_list = {**override_dict, **step.override_dict}, label_list + [step.name_str]
        cumulative_list.append({**_row(" + ".join(label_list), _window(run_fn(dict(override_dict)), start_str, end_str), live_ser, live_sharpe_float),
                                "alt_sharpe_float": alt_sharpe(dict(override_dict))})
    min_spec_int = 0  # number of cumulative steps kept
    for step_int, row in enumerate(cumulative_list, start=1):
        if not row["sharpe_float"] >= MIN_SPEC_SHARE_FLOAT * live_sharpe_float:
            break
        min_spec_int = step_int
    return {"window_str": f"{live_ser.index[0].date()} to {live_ser.index[-1].date()}", "live_sharpe_float": live_sharpe_float,
            "live_alt_sharpe_float": alt_sharpe({}),
            "live_metric": performance_dict(live_ser), "single_list": single_list, "cumulative_list": cumulative_list,
            "min_spec_step_int": min_spec_int,
            "min_spec_str": cumulative_list[min_spec_int - 1]["name_str"] if min_spec_int else "(none: every step loses more than 20%)"}


# ---------------------------------------------------------------- 4. random-parameter percentile
@dataclass
class RandomParameterResult:
    live_sharpe_float: float
    draw_list: list[dict] = field(default_factory=list)  # {"config": dict, "sharpe_float": float}

    @property
    def sharpe_vec(self) -> np.ndarray:
        return np.array([d["sharpe_float"] for d in self.draw_list], dtype=float)


def random_parameter_summary(result: RandomParameterResult) -> dict:
    sharpe_vec = result.sharpe_vec
    finite_vec = sharpe_vec[np.isfinite(sharpe_vec)]
    live_float = result.live_sharpe_float
    if finite_vec.size == 0:
        return {"draw_count_int": int(sharpe_vec.size), "failed_count_int": int(sharpe_vec.size), "live_sharpe_float": live_float,
                "share_at_or_above_live_float": float("nan"), "median_float": float("nan"), "p10_float": float("nan"),
                "p90_float": float("nan"), "positive_share_float": float("nan"), "verdict_str": "NO VALID DRAWS"}
    median_float, p10_float = float(np.median(finite_vec)), float(np.quantile(finite_vec, 0.10))
    if median_float >= ROBUST_MEDIAN_SHARE_FLOAT * live_float and p10_float > 0:
        verdict_str = "ROBUST TO VALUES"
    elif median_float < DEPENDS_MEDIAN_SHARE_FLOAT * live_float:
        verdict_str = "DEPENDS ON VALUES"
    else:
        verdict_str = "PARTLY"
    return {"draw_count_int": int(sharpe_vec.size), "failed_count_int": int(sharpe_vec.size - finite_vec.size),
            "live_sharpe_float": live_float, "share_at_or_above_live_float": float(np.mean(finite_vec >= live_float)),
            "median_float": median_float, "p10_float": p10_float, "p90_float": float(np.quantile(finite_vec, 0.90)),
            "positive_share_float": float(np.mean(finite_vec > 0)), "verdict_str": verdict_str}


# ---------------------------------------------------------------- multiplicity: Romano-Wolf step-down
def paired_sharpe_difference_draws(base_vec: np.ndarray, variant_mat: np.ndarray, draw_count_int: int = 2000,
                                   block_float: float = BOOTSTRAP_BLOCK_FLOAT, random_seed_int: int = 0,
                                   chunk_int: int = 250) -> tuple[np.ndarray, np.ndarray]:
    """Observed Sharpe(variant_k) - Sharpe(base) and its stationary-bootstrap draws (draws x K). Every draw resamples
    the same dates for the base and all variants, so the differences keep their joint dependence."""
    base_vec, variant_mat = np.asarray(base_vec, dtype=float), np.asarray(variant_mat, dtype=float)
    if variant_mat.ndim == 1:
        variant_mat = variant_mat[:, None]

    def sharpe_rows(path_mat: np.ndarray) -> np.ndarray:
        sd_vec = path_mat.std(axis=1, ddof=1)
        return np.where(sd_vec > 0, path_mat.mean(axis=1) / np.where(sd_vec > 0, sd_vec, 1.0) * np.sqrt(252.0), 0.0)

    def sharpe(value_vec: np.ndarray) -> float:
        sd_float = value_vec.std(ddof=1)
        return float(value_vec.mean() / sd_float * np.sqrt(252.0)) if sd_float > 0 else 0.0

    observed_vec = np.array([sharpe(variant_mat[:, k]) - sharpe(base_vec) for k in range(variant_mat.shape[1])])
    draw_list, done_int, chunk_seed_int = [], 0, random_seed_int
    while done_int < draw_count_int:
        size_int = min(chunk_int, draw_count_int - done_int)
        index_mat = stationary_bootstrap_index_mat(len(base_vec), size_int, block_float, len(base_vec), chunk_seed_int)
        base_sharpe_vec = sharpe_rows(base_vec[index_mat])
        draw_list.append(np.column_stack([sharpe_rows(variant_mat[:, k][index_mat]) - base_sharpe_vec for k in range(variant_mat.shape[1])]))
        done_int, chunk_seed_int = done_int + size_int, chunk_seed_int + 1
    return observed_vec, np.vstack(draw_list)


def romano_wolf_stepdown(observed_vec: np.ndarray, draw_mat: np.ndarray) -> np.ndarray:
    """Family-wise adjusted one-sided p-values for H0_k: difference_k <= 0 (Romano and Wolf 2005, studentised step-down).
    t_k = d_k / sd_k with sd_k the bootstrap standard deviation; the null of the largest remaining t is the bootstrap
    distribution of max over the remaining hypotheses of (d*_k - d_k) / sd_k; p-values are made monotone."""
    observed_vec, draw_mat = np.asarray(observed_vec, dtype=float), np.asarray(draw_mat, dtype=float)
    sd_vec = draw_mat.std(axis=0, ddof=1)
    safe_vec = np.where(sd_vec > 0, sd_vec, 1.0)
    t_vec = np.where(sd_vec > 0, observed_vec / safe_vec, 0.0)
    null_mat = np.where(sd_vec > 0, (draw_mat - observed_vec) / safe_vec, 0.0)
    adjusted_vec, running_float = np.empty(observed_vec.size), 0.0
    remaining_list = list(np.argsort(-t_vec))
    for k_int in list(remaining_list):
        max_null_vec = null_mat[:, remaining_list].max(axis=1)
        p_float = float((1 + np.sum(max_null_vec >= t_vec[k_int])) / (1 + draw_mat.shape[0]))
        running_float = max(running_float, p_float)
        adjusted_vec[k_int] = running_float if sd_vec[k_int] > 0 else 1.0
        remaining_list.remove(k_int)
    return adjusted_vec
