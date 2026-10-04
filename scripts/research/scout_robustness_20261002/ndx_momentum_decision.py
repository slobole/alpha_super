"""The momentum decision pack (owner request 2026-10-04: one ordered view of the momentum family, then a decision).

No new candidate and no re-selection. The four NDX finalists were fixed by earlier registered rules:
    L         the LIVE pod: ROC12 / dollar ATR20, Close > SMA100, top 10, SPY SMA200 regime, VXN scale
    NATR20    the same with the scale-free score ROC12 / (ATR20 / Close)
    E2        50/50 of the two (target weights averaged, one account)
    E2+cap    E2 with at most 4 of 10 names per current-label GICS sector in each book (the design of record, A15)
This run prints them side by side under one method and adds the controls a decision needs, registered in the Scout
ledger before any number is computed (ndx_momentum_decision_controls_20261004):

1. Selection ablation: the same gates and exposure with the stock selection removed.
       QQQ-EXP   QQQ (total return) held at L's own invested fraction of the day before; the idle part earns T-bills
       EW-ALL    every eligible member (member, Close > SMA100, finite score) at equal weight, same total weight as L
       RANDOM    200 books of 10 random eligible members; a held name that is still eligible is kept with L's own
                 average retention rate (so turnover and costs match), the free slots are refilled at random
   Verdict by the A15 thresholds (alpha/scout/stations/robustness.py): P = paired stationary-bootstrap
   P(Sharpe finalist > Sharpe control); EARNS ITS PLACE if P >= 0.80 and the gap >= 0.05; NO EVIDENCE if P < 0.50;
   UNCLEAR otherwise. RANDOM: the finalist's Sharpe percentile among the 200 books.
2. Factor alpha (the standing method of scripts/research/fund_menu_20260923/allocator_first_look.py): weekly excess
   returns, Newey-West 4 lags; QQQ alone; QQQ + the QQQ 200-day rule; ETF mix (QQQ, IEF, GLD, DBC, UUP); ETF mix + the
   QQQ 200-day rule (from UUP's first week, 2007); full window and halves.
3. In the book: 60% TAA 3x / 40% NDX leg (the live weights), monthly rebalance; 2012-11 on with the real TAA series and
   2008-03 on with the validated TQQQ/BTAL proxy before it.
4. Whole-share feasibility by pod size on the 2023+ decisions (the live engine floors to whole shares).
5. The 16-offset luck band, eras, crashes, correlations.

Idle cash is credited at the T-bill rate (the 0%-cash Sharpe is printed beside it). History through 2026-10 has been
examined; nothing here is an untouched holdout.

    uv run python scripts/research/scout_robustness_20261002/ndx_momentum_decision.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from alpha.scout.engines.weights import CostModel, simulate
from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH, Ledger
from alpha.scout.metrics import performance_dict, sharpe_float
from alpha.scout.registration import Registration, register, registration_rows
from alpha.scout.stations.robustness import (
    EARNS_PROBABILITY_FLOAT,
    MATERIAL_SHARPE_FLOAT,
    NO_EVIDENCE_PROBABILITY_FLOAT,
    paired_sharpe_probability,
)
from alpha.scout.stations.s4_strategy import SEAL_END_STR
from ndx_ranking_compare import CRISIS_DICT, window_return
from plans import PLAN_DICT
from run import PodRunner, cash_credited, market_inputs

OUTPUT_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "robustness" / "ndx_momentum_decision.json"
PROXY_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "research" / "portfolio" / "portfolio_refresh_20260927" / "sleeve_series_incl_2008.csv.gz"
START_STR = "2000-09-01"
SEEN_START_STR = "2023-01-01"
SEED_INT = 20261004
RANDOM_BOOK_INT = 200
ALL_ELIGIBLE_TOP_INT = 500  # more than the index ever holds: the top-N walk then returns every eligible name
CAP_DICT = {"sector_cap_int": 4, "sector_level_int": 1}
FINALIST_DICT = {
    "L live (dollar ATR)": [{}],
    "NATR20": [{"atr_unit_str": "percent"}],
    "E2 (50/50)": [{}, {"atr_unit_str": "percent"}],
    "E2 + sector cap": [dict(CAP_DICT), {"atr_unit_str": "percent", **CAP_DICT}],
}
DESIGN_STR = "E2 + sector cap"
LIVE_STR = "L live (dollar ATR)"
ERA_DICT = {"2000-09 to 2007": (START_STR, "2007-12-31"), "2008-2015": ("2008-01-01", "2015-12-31"),
            "2016-2022": ("2016-01-01", SEAL_END_STR), "2023 on (seen)": (SEEN_START_STR, None)}
POD_SIZE_TUPLE = (12_000, 25_000, 50_000, 100_000, 250_000, 1_000_000)
BOOK_WEIGHT_DICT = {"taa": 0.6, "ndx": 0.4}
NW_LAG_INT = 4
REGISTRATION = Registration(
    registration_id_str="ndx_momentum_decision_controls_20261004", family_id_str="equity_cross_sectional_momentum",
    hypothesis_str=("The top-10 momentum selection of the NDX pod (live L, NATR20, E2, E2 + sector cap) earns more than the "
                    "same gates without selection: QQQ held at the pod's own invested fraction, every eligible member at "
                    "equal weight, and 200 turnover-matched random-pick books."),
    mechanism_str=("Cross-sectional momentum: 12-month winners per unit of volatility keep beating the index average; the SPY "
                   "regime gate and the VXN scale are timing, shared by every control."),
    expected_sign_and_location_str=("Finalists above the random-book median and above exposure-matched QQQ over 2000-09 to "
                                    "2026-10; the gap larger before 2023 than after (QQQ's own run)."),
    hypothesis_class_str="X", universe_str="Nasdaq-100 point-in-time members", horizon_str="months",
    schedule_str="month-end decision", execution_str="next session's open",
    primary_metric_str=("Full-period net Sharpe (idle cash at T-bills): paired stationary-bootstrap P(finalist > control) and "
                        "the finalist's percentile among 200 random-pick books"),
    kill_criteria_str=("A15 thresholds, printed; no re-selection among the finalists (fixed by earlier registered rules). For the "
                       "design of record (E2 + sector cap): SELECTION EARNS if P >= 0.80 and the Sharpe gap >= 0.05 against BOTH "
                       "exposure-matched QQQ and equal-weight eligible, and its Sharpe is at or above the 90th percentile of the "
                       "random books; NO EVIDENCE if P < 0.50 against either control or it is below the random median; UNCLEAR "
                       "otherwise. Under NO EVIDENCE the decision record must say the momentum slot is index exposure plus "
                       "timing, and plan on the control's numbers."),
    source_str=("alpha/scout/specs/ndx_vxn.py; alpha/scout/stations/robustness.py (A15 thresholds); "
                "scripts/research/fund_menu_20260923/allocator_first_look.py (alpha method)"),
    parent_id_str="ndx_e2_sector_cap_20261004",
    universe_choice_str="The pod's own universe (Nasdaq-100 point-in-time), unchanged.",
)
_WORKER: dict = {}


# ---------------------------------------------------------------- books
def averaged_weight_df(inputs, override_list: list[dict], offset_int: int = 0) -> pd.DataFrame:
    from alpha.scout.specs import ndx_vxn

    frame_list = [ndx_vxn.rebalance_weight_df(inputs, dataclasses.replace(ndx_vxn.LIVE_CONFIG, **o, decision_offset_int=offset_int))
                  for o in override_list]
    index = frame_list[0].index
    for frame in frame_list[1:]:
        index = index.union(frame.index)
    columns = sorted(set().union(*[f.columns for f in frame_list]))
    return sum(f.reindex(index=index, columns=columns).fillna(0.0) for f in frame_list) / len(frame_list)


def simulate_weight_df(inputs, tbill_ser: pd.Series, weight_df: pd.DataFrame) -> dict:
    """One account on the given target weights: net daily returns (T-bill cash and 0% cash), invested fraction, turnover."""
    from alpha.scout.specs import ndx_vxn

    stock_list = list(weight_df.columns[(weight_df != 0).any(axis=0)])  # never-held names cannot trade: dropping them is exact
    weight_df = weight_df[stock_list]
    result = simulate(inputs.open_df[stock_list], inputs.close_df[stock_list], inputs.dividend_df[stock_list], weight_df,
                      start_date=ndx_vxn.TRADING_START_STR, capital_float=100_000.0, share_unit_mode_str="historical",
                      unadjusted_close_df=inputs.raw_close_df[stock_list], cost_model=CostModel())
    position_df = result.daily_position_df
    long_value_ser = (position_df * inputs.close_df[stock_list].reindex(position_df.index)).clip(lower=0.0).sum(axis=1)
    trade_df = result.trade_df.loc[result.trade_df["date"] >= START_STR]
    traded_float = float((trade_df["delta_float"].abs() * trade_df["price_float"]).sum())
    value_ser = result.total_value_ser.loc[START_STR:]
    return {"daily": cash_credited(result, inputs.close_df, tbill_ser).loc[START_STR:], "daily0": result.daily_return_ser.loc[START_STR:],
            "invested": (long_value_ser / result.total_value_ser), "names": float((position_df.loc[START_STR:] != 0).sum(axis=1).replace(0, np.nan).mean()),
            "turnover": traded_float / float(value_ser.mean()) / (len(value_ser) / 252.0)}


def _luck_init(inputs, tbill_ser, sector_cache: dict) -> None:
    from alpha.scout.specs import ndx_vxn

    _WORKER["inputs"], _WORKER["tbill"] = inputs, tbill_ser
    ndx_vxn._SECTOR_GROUP_CACHE.update(sector_cache)


def _luck_task(args) -> tuple:
    name_str, offset_int = args
    run = simulate_weight_df(_WORKER["inputs"], _WORKER["tbill"], averaged_weight_df(_WORKER["inputs"], FINALIST_DICT[name_str], offset_int))
    return name_str, offset_int, sharpe_float(run["daily"])


def _random_init(inputs, tbill_ser, eligible_df: pd.DataFrame, slot_weight_ser: pd.Series, retain_float: float) -> None:
    _WORKER.update(inputs=inputs, tbill=tbill_ser, eligible=eligible_df, slot=slot_weight_ser, retain=retain_float)


def random_weight_df(eligible_df: pd.DataFrame, slot_weight_ser: pd.Series, retain_float: float, draw_int: int, top_int: int = 10) -> pd.DataFrame:
    """10 random eligible names per decision; a held name that is still eligible is kept with probability retain_float."""
    rng_obj = np.random.default_rng([SEED_INT, draw_int])
    column_vec, held_list, row_dict = eligible_df.columns.to_numpy(), [], {}
    for execution_ts, flag_vec in zip(eligible_df.index, eligible_df.to_numpy()):
        name_list = list(column_vec[flag_vec])
        if not name_list:
            held_list, row_dict[execution_ts] = [], {}
            continue
        name_set = set(name_list)
        keep_list = [s for s in held_list if s in name_set and rng_obj.random() < retain_float]
        free_int = min(top_int, len(name_list)) - len(keep_list)
        pool_list = [s for s in name_list if s not in set(keep_list)]
        held_list = keep_list + (list(rng_obj.choice(pool_list, size=free_int, replace=False)) if free_int > 0 else [])
        row_dict[execution_ts] = dict.fromkeys(held_list, float(slot_weight_ser.loc[execution_ts]))
    return pd.DataFrame.from_dict(row_dict, orient="index").reindex(eligible_df.index).fillna(0.0)


def _random_task(draw_int: int) -> dict:
    weight_df = random_weight_df(_WORKER["eligible"], _WORKER["slot"], _WORKER["retain"], draw_int)
    run = simulate_weight_df(_WORKER["inputs"], _WORKER["tbill"], weight_df)
    return {"draw": draw_int, "daily": run["daily"], "turnover": run["turnover"]}


def retention_float(weight_df: pd.DataFrame, eligible_df: pd.DataFrame) -> float:
    """Share of the names held last month and still eligible that the rule holds again (pooled over months)."""
    held_df = weight_df.reindex(index=eligible_df.index, columns=eligible_df.columns).fillna(0.0) > 0
    still_df = held_df.shift(1, fill_value=False) & eligible_df
    return float((still_df & held_df).to_numpy().sum() / still_df.to_numpy().sum())


# ---------------------------------------------------------------- statistics
def stats_dict(daily_ser: pd.Series, tbill_ser: pd.Series) -> dict:
    out = {"full": performance_dict(daily_ser, tbill_ser), "to_2022": performance_dict(daily_ser.loc[:SEAL_END_STR], tbill_ser),
           "seen_2023_on": performance_dict(daily_ser.loc[SEEN_START_STR:], tbill_ser)}
    out["era_sharpe"] = {k: sharpe_float(daily_ser.loc[a:b]) for k, (a, b) in ERA_DICT.items()}
    out["crisis"] = {k: window_return(daily_ser, a, b) for k, (a, b) in CRISIS_DICT.items()}
    return out


def verdict_str(probability_float: float, gap_float: float) -> str:
    if probability_float < NO_EVIDENCE_PROBABILITY_FLOAT:
        return "NO EVIDENCE"
    if probability_float >= EARNS_PROBABILITY_FLOAT and gap_float > 0:
        return "EARNS ITS PLACE" if gap_float >= MATERIAL_SHARPE_FLOAT else "SMALL"
    return "UNCLEAR"


def weekly_ser(daily_ser: pd.Series) -> pd.Series:
    return (1.0 + daily_ser).resample("W-FRI").prod(min_count=1) - 1.0


def newey_west_ols(y_ser: pd.Series, x_df: pd.DataFrame, lag_int: int = NW_LAG_INT) -> dict:
    """OLS with Newey-West (Bartlett) standard errors; as scripts/research/fund_menu_20260923/allocator_first_look.py."""
    frame_df = pd.concat([y_ser.rename("y"), x_df], axis=1).dropna()
    y_arr = frame_df["y"].to_numpy()
    x_arr = np.column_stack([np.ones(len(frame_df)), frame_df[x_df.columns].to_numpy()])
    xtx_inv = np.linalg.inv(x_arr.T @ x_arr)
    beta_arr = xtx_inv @ x_arr.T @ y_arr
    resid_arr = y_arr - x_arr @ beta_arr
    score_arr = x_arr * resid_arr[:, None]
    s_mat = score_arr.T @ score_arr
    for lag in range(1, lag_int + 1):
        gamma_mat = score_arr[lag:].T @ score_arr[:-lag]
        s_mat += (1.0 - lag / (lag_int + 1.0)) * (gamma_mat + gamma_mat.T)
    se_arr = np.sqrt(np.diag(xtx_inv @ s_mat @ xtx_inv))
    out = {"alpha_ann": float(beta_arr[0] * 52.0), "alpha_t": float(beta_arr[0] / se_arr[0]), "r2": float(1.0 - resid_arr.var() / y_arr.var()),
           "weeks": int(len(frame_df))}
    out.update({f"b_{name}": float(b) for name, b in zip(x_df.columns, beta_arr[1:])})
    return out


def etf_return_df(date_index: pd.DatetimeIndex, tbill_ser: pd.Series) -> pd.DataFrame:
    """Daily total returns of the factor ETFs and the QQQ 200-day rule (QQQ above its 200-day average at the prior close,
    else IEF; T-bills before IEF exists). *** CRITICAL*** the close-t signal sets the holding for session t+1."""
    from data.norgate_loader import load_price_timeseries

    # date_index is the session calendar (no holiday rows): a 200-session average and close-to-close returns as traded.
    close_df = pd.DataFrame({s: load_price_timeseries(s, adjustment_str="TOTALRETURN", start_date_str="1998-01-01")["Close"]
                             for s in ("QQQ", "IEF", "GLD", "DBC", "UUP")}).reindex(date_index)
    return_df = close_df.pct_change(fill_method=None)
    in_qqq_ser = (close_df["QQQ"] > close_df["QQQ"].rolling(200).mean()).astype(float).shift(1)
    off_ser = return_df["IEF"].where(return_df["IEF"].notna(), tbill_ser.reindex(date_index))
    return_df["TREND200"] = in_qqq_ser * return_df["QQQ"] + (1.0 - in_qqq_ser) * off_ser
    # The same rule with T-bills as the off asset: the naive public benchmark for the whole window.
    return_df["QQQ_200D_TBILL"] = in_qqq_ser * return_df["QQQ"] + (1.0 - in_qqq_ser) * tbill_ser.reindex(date_index)
    return return_df


def alpha_dict(daily_ser: pd.Series, etf_df: pd.DataFrame, tbill_ser: pd.Series) -> dict:
    tbill_weekly_ser = weekly_ser(tbill_ser.reindex(etf_df.index).fillna(0.0))
    factor_weekly_df = pd.DataFrame({s: weekly_ser(etf_df[s]) for s in ("QQQ", "IEF", "GLD", "DBC", "UUP", "TREND200")}).sub(tbill_weekly_ser, axis=0)
    y_ser = (weekly_ser(daily_ser) - tbill_weekly_ser.reindex(weekly_ser(daily_ser).index)).iloc[1:-1]  # drop the partial end weeks
    half_int = len(y_ser) // 2
    window_dict = {"full": y_ser, "first_half": y_ser.iloc[:half_int], "second_half": y_ser.iloc[half_int:]}
    model_dict = {"QQQ": ["QQQ"], "QQQ+TREND200": ["QQQ", "TREND200"], "ETF mix": ["QQQ", "IEF", "GLD", "DBC", "UUP"],
                  "ETF mix+TREND200": ["QQQ", "IEF", "GLD", "DBC", "UUP", "TREND200"]}
    return {f"{window_str} | {model_str}": newey_west_ols(window_ser, factor_weekly_df[column_list].reindex(window_ser.index))
            for window_str, window_ser in window_dict.items() for model_str, column_list in model_dict.items()}


def book_ser(sleeve_df: pd.DataFrame, weight_dict: dict) -> pd.Series:
    """Fixed weights reset at each month's first session; the sleeves drift inside the month."""
    weight_ser = pd.Series(weight_dict)
    part_list = []
    for _, month_df in sleeve_df[list(weight_dict)].dropna().groupby(pd.Grouper(freq="MS")):
        if month_df.empty:
            continue
        value_ser = ((1.0 + month_df).cumprod() * weight_ser).sum(axis=1)
        return_ser = value_ser.pct_change()
        return_ser.iloc[0] = value_ser.iloc[0] - 1.0
        part_list.append(return_ser)
    return pd.concat(part_list)


def whole_share_dict(weight_df: pd.DataFrame, raw_close_df: pd.DataFrame, size_float: float) -> dict:
    """2023+ decisions at a pod of size_float: share of target positions below one share, and the share of the intended
    exposure that flooring to whole shares leaves in cash."""
    row_df = weight_df.loc[SEEN_START_STR:]
    row_df = row_df.loc[row_df.sum(axis=1) > 0, (row_df != 0).any(axis=0)]
    decision_close_df = raw_close_df.shift(1).reindex(index=row_df.index, columns=row_df.columns)  # raw close at T (execution T+1)
    held_df = row_df > 0
    share_df = (row_df * size_float / decision_close_df).where(held_df)
    bought_df = np.floor(share_df) * decision_close_df / size_float
    return {"sub_share_position_share": float((share_df < 1).to_numpy().sum() / held_df.to_numpy().sum()),
            "exposure_lost_share": float(((row_df.where(held_df) - bought_df).sum(axis=1) / row_df.sum(axis=1)).mean()),
            "names_mean": float(held_df.sum(axis=1).mean())}


def compact(perf_dict: dict) -> dict:
    return {k: perf_dict.get(k) for k in ("cagr_float", "volatility_float", "sharpe_float", "max_drawdown_float", "calmar_float", "excess_sharpe_float")}


# ---------------------------------------------------------------- main
def main() -> None:
    from alpha.scout.specs import ndx_vxn

    ledger = Ledger()
    if REGISTRATION.registration_id_str not in registration_rows(ledger):
        register(ledger, REGISTRATION)
        print("registered", REGISTRATION.registration_id_str, flush=True)
    market_dict, tbill_ser = market_inputs()
    inputs = ndx_vxn.load_inputs()
    symbol_list = [s for s in inputs.close_df.columns if s != ndx_vxn.REGIME_SYMBOL_STR]
    ndx_vxn.sector_group_dict(symbol_list, 1)  # seed the GICS cache once (Norgate) for the workers

    # Finalists (offset 0, in this process: the weights and the invested fraction are needed).
    weight_dict = {name_str: averaged_weight_df(inputs, override_list) for name_str, override_list in FINALIST_DICT.items()}
    run_dict = {name_str: simulate_weight_df(inputs, tbill_ser, weight_df) for name_str, weight_df in weight_dict.items()}
    daily_dict = {name_str: run["daily"] for name_str, run in run_dict.items()}
    date_index = daily_dict[LIVE_STR].index

    # Controls: every eligible name and L's own slot weight (0.1 x the VXN scale), read off an all-eligible run of L's rule.
    all_df = ndx_vxn.rebalance_weight_df(inputs, dataclasses.replace(ndx_vxn.LIVE_CONFIG, top_count_int=ALL_ELIGIBLE_TOP_INT))
    eligible_df = all_df > 0
    eligible_df = eligible_df.loc[:, eligible_df.any(axis=0)]
    slot_weight_ser = all_df.max(axis=1) * ALL_ELIGIBLE_TOP_INT / ndx_vxn.TOP_COUNT_INT
    count_ser = eligible_df.sum(axis=1)
    total_weight_ser = slot_weight_ser * count_ser.clip(upper=ndx_vxn.TOP_COUNT_INT)  # L's own total weight at each decision
    equal_weight_df = eligible_df.astype(float).mul(total_weight_ser / count_ser.where(count_ser > 0, np.nan), axis=0).fillna(0.0)
    live_weight_check_float = float((weight_dict[LIVE_STR].sum(axis=1) - total_weight_ser.reindex(weight_dict[LIVE_STR].index)).abs().max())
    retain_float = retention_float(weight_dict[LIVE_STR], eligible_df)
    equal_run = simulate_weight_df(inputs, tbill_ser, equal_weight_df)
    etf_df = etf_return_df(inputs.close_df.index, tbill_ser)
    qqq_ser = etf_df["QQQ"].reindex(date_index)
    tbill_day_ser = tbill_ser.reindex(date_index).fillna(0.0)
    invested_ser = run_dict[LIVE_STR]["invested"].shift(1).reindex(date_index).fillna(0.0).clip(0.0, 1.0)
    control_dict = {
        "QQQ-EXP (QQQ at L's exposure)": invested_ser * qqq_ser + (1.0 - invested_ser) * tbill_day_ser,
        "EW-ALL (all eligible, equal weight)": equal_run["daily"],
        "QQQ buy and hold": qqq_ser,
        "QQQ 200-day rule (else T-bills)": etf_df["QQQ_200D_TBILL"].reindex(date_index),
    }

    sector_cache = dict(ndx_vxn._SECTOR_GROUP_CACHE)
    with Pool(12, initializer=_luck_init, initargs=(inputs, tbill_ser, sector_cache)) as pool_obj:
        luck_list = pool_obj.map(_luck_task, [(name_str, o) for name_str in FINALIST_DICT for o in range(16)], chunksize=1)
    with Pool(12, initializer=_random_init, initargs=(inputs, tbill_ser, eligible_df, slot_weight_ser, retain_float)) as pool_obj:
        random_list = pool_obj.map(_random_task, range(RANDOM_BOOK_INT), chunksize=2)
    random_sharpe_vec = np.array([sharpe_float(r["daily"]) for r in random_list])
    random_cagr_vec = np.array([performance_dict(r["daily"], tbill_ser)["cagr_float"] for r in random_list])
    random_dd_vec = np.array([performance_dict(r["daily"], tbill_ser)["max_drawdown_float"] for r in random_list])
    random_seen_vec = np.array([sharpe_float(r["daily"].loc[SEEN_START_STR:]) for r in random_list])
    random_insample_vec = np.array([sharpe_float(r["daily"].loc[:SEAL_END_STR]) for r in random_list])

    out = {"window": {"start": START_STR, "end": str(date_index[-1].date())}, "finalists": {}, "controls": {}, "selection": {},
           "checks": {"live_total_weight_max_abs_diff": live_weight_check_float, "retention": retain_float,
                      "eligible_names_mean": float(count_ser[count_ser > 0].mean())}}
    for name_str, run in run_dict.items():
        luck_vec = np.array([s for n, _, s in luck_list if n == name_str])
        row = stats_dict(run["daily"], tbill_ser)
        row.update({"sharpe_zero_cash": sharpe_float(run["daily0"]), "cagr_zero_cash": performance_dict(run["daily0"])["cagr_float"],
                    "luck": {"min": float(luck_vec.min()), "median": float(np.median(luck_vec)), "max": float(luck_vec.max())},
                    "names": run["names"], "turnover": run["turnover"], "invested_mean": float(run["invested"].loc[START_STR:].mean()),
                    "alpha": alpha_dict(run["daily"], etf_df, tbill_ser),
                    "whole_share": {str(size): whole_share_dict(weight_dict[name_str], inputs.raw_close_df, float(size)) for size in POD_SIZE_TUPLE},
                    "random_percentile": {"full": float(np.mean(random_sharpe_vec < row["full"]["sharpe_float"])),
                                          "to_2022": float(np.mean(random_insample_vec < row["to_2022"]["sharpe_float"])),
                                          "seen_2023_on": float(np.mean(random_seen_vec < row["seen_2023_on"]["sharpe_float"]))}})
        row["vs_control"] = {}
        for control_str in ("QQQ-EXP (QQQ at L's exposure)", "EW-ALL (all eligible, equal weight)"):
            for window_str, (a, b) in {"full": (START_STR, None), "to_2022": (START_STR, SEAL_END_STR), "seen_2023_on": (SEEN_START_STR, None)}.items():
                p_float = paired_sharpe_probability(run["daily"].loc[a:b], control_dict[control_str].loc[a:b])
                gap_float = sharpe_float(run["daily"].loc[a:b]) - sharpe_float(control_dict[control_str].loc[a:b])
                row["vs_control"][f"{control_str} | {window_str}"] = {"p": p_float, "gap": gap_float, "verdict": verdict_str(p_float, gap_float)}
        out["finalists"][name_str] = row
    for control_str, control_ser in control_dict.items():
        out["controls"][control_str] = stats_dict(control_ser.dropna(), tbill_ser)
    out["controls"]["EW-ALL (all eligible, equal weight)"].update({"names": equal_run["names"], "turnover": equal_run["turnover"]})
    out["random"] = {"count": RANDOM_BOOK_INT, "retain": retain_float, "turnover_mean": float(np.mean([r["turnover"] for r in random_list])),
                     "sharpe": {str(q): float(np.percentile(random_sharpe_vec, q)) for q in (5, 25, 50, 75, 90, 95)},
                     "sharpe_to_2022": {str(q): float(np.percentile(random_insample_vec, q)) for q in (5, 50, 95)},
                     "sharpe_seen": {str(q): float(np.percentile(random_seen_vec, q)) for q in (5, 50, 95)},
                     "cagr": {str(q): float(np.percentile(random_cagr_vec, q)) for q in (5, 50, 95)},
                     "max_drawdown": {str(q): float(np.percentile(random_dd_vec, q)) for q in (5, 50, 95)}}
    design = out["finalists"][DESIGN_STR]
    p_qqq, p_ew = (design["vs_control"][f"{c} | full"] for c in ("QQQ-EXP (QQQ at L's exposure)", "EW-ALL (all eligible, equal weight)"))
    pct_float = design["random_percentile"]["full"]
    if p_qqq["verdict"] == "EARNS ITS PLACE" and p_ew["verdict"] == "EARNS ITS PLACE" and pct_float >= 0.90:
        out["selection"]["verdict"] = "SELECTION EARNS"
    elif min(p_qqq["p"], p_ew["p"]) < NO_EVIDENCE_PROBABILITY_FLOAT or pct_float < 0.50:
        out["selection"]["verdict"] = "NO EVIDENCE"
    else:
        out["selection"]["verdict"] = "UNCLEAR"
    out["selection"].update({"p_vs_qqq_exp": p_qqq, "p_vs_ew_all": p_ew, "random_percentile": pct_float})

    # Correlations and head-to-head among the finalists.
    daily_df = pd.DataFrame(daily_dict)
    month_df = (1.0 + daily_df).resample("ME").prod() - 1.0
    out["correlation"] = {"daily": daily_df.corr().round(3).to_dict(), "monthly": month_df.corr().round(3).to_dict()}
    out["head_to_head_p"] = {f"{a} > {b}": paired_sharpe_probability(daily_dict[a], daily_dict[b]) for a in FINALIST_DICT for b in FINALIST_DICT if a != b}

    # In the book: 60% TAA 3x / 40% NDX leg.
    taa_ser = PodRunner(PLAN_DICT["taa_3x"], tbill_ser=tbill_ser).daily({}).loc[PLAN_DICT["taa_3x"].eval_start_str:]
    proxy_ser = pd.read_csv(PROXY_PATH, index_col=0, parse_dates=True)["taa_btal_tqqq"].dropna()
    overlap_index = taa_ser.index.intersection(proxy_ser.index)
    long_taa_ser = pd.concat([proxy_ser.loc[proxy_ser.index < taa_ser.index[0]], taa_ser])
    out["book"] = {"weights": BOOK_WEIGHT_DICT, "taa_proxy_overlap": {"corr": float(taa_ser.loc[overlap_index].corr(proxy_ser.loc[overlap_index])),
                                                                     "mean_abs_diff_bp": float((taa_ser.loc[overlap_index] - proxy_ser.loc[overlap_index]).abs().mean() * 1e4)},
                   "real_taa_2012_on": {}, "with_proxy_2008_on": {}}
    leg_dict = {**daily_dict, "QQQ-EXP (QQQ at L's exposure)": control_dict["QQQ-EXP (QQQ at L's exposure)"]}
    book_dict = {}
    for label_str, taa_leg_ser in (("real_taa_2012_on", taa_ser), ("with_proxy_2008_on", long_taa_ser)):
        for name_str, leg_ser in leg_dict.items():
            series = book_ser(pd.DataFrame({"taa": taa_leg_ser, "ndx": leg_ser}), BOOK_WEIGHT_DICT)
            book_dict[(label_str, name_str)] = series
            out["book"][label_str][name_str] = {**compact(performance_dict(series, tbill_ser)), "start": str(series.index[0].date()),
                                                "seen_2023_on_sharpe": sharpe_float(series.loc[SEEN_START_STR:]),
                                                "to_2022_sharpe": sharpe_float(series.loc[:SEAL_END_STR])}
        out["book"][label_str]["TAA 3x alone"] = compact(performance_dict(taa_leg_ser.dropna(), tbill_ser))
        out["book"][label_str]["p_design_book_gt_live_book"] = paired_sharpe_probability(book_dict[(label_str, DESIGN_STR)], book_dict[(label_str, LIVE_STR)])
        out["book"][label_str]["corr_taa_monthly"] = {name_str: float(((1 + leg_ser).resample("ME").prod() - 1).corr((1 + taa_leg_ser).resample("ME").prod() - 1))
                                                      for name_str, leg_ser in leg_dict.items()}

    # Monthly curves for the chart: finalists, controls and the random band.
    def curve_list(series: pd.Series) -> list:
        return [round(float(x), 4) for x in (1.0 + series.fillna(0.0)).cumprod().resample("ME").last()]

    random_curve_df = pd.DataFrame({r["draw"]: (1.0 + r["daily"]).cumprod().resample("ME").last() for r in random_list})
    out["chart"] = {"month": [d.strftime("%Y-%m") for d in (1.0 + daily_dict[LIVE_STR]).cumprod().resample("ME").last().index],
                    "curve": {**{n: curve_list(s) for n, s in daily_dict.items()}, **{n: curve_list(s) for n, s in control_dict.items()}},
                    "random_band": {str(q): [round(float(x), 4) for x in random_curve_df.quantile(q / 100.0, axis=1)] for q in (5, 50, 95)}}
    OUTPUT_PATH.write_text(json.dumps(out, default=float), encoding="utf-8")

    # ---------------------------------------------------------------- print
    def pct(x):
        return f"{x:6.1%}"

    print(f"\nwindow {START_STR} to {date_index[-1].date()} | checks: live weight diff {live_weight_check_float:.1e}, retention {retain_float:.2f}, "
          f"eligible names {out['checks']['eligible_names_mean']:.0f}")
    print(f"{'':38s} {'CAGR':>6s} {'Vol':>6s} {'Sharpe':>6s} {'Sh 0%':>6s} {'MaxDD':>7s} {'Sh<=22':>6s} {'Sh 23+':>6s} {'luck min/med/max':>18s} {'rand pct':>8s} {'names':>5s} {'turn':>5s}")
    for name_str, row in out["finalists"].items():
        f = row["full"]
        print(f"{name_str:38s} {pct(f['cagr_float'])} {pct(f['volatility_float'])} {f['sharpe_float']:6.2f} {row['sharpe_zero_cash']:6.2f} {pct(f['max_drawdown_float']):>7s} "
              f"{row['to_2022']['sharpe_float']:6.2f} {row['seen_2023_on']['sharpe_float']:6.2f} {row['luck']['min']:6.2f}/{row['luck']['median']:.2f}/{row['luck']['max']:.2f} "
              f"{row['random_percentile']['full']:8.0%} {row['names']:5.1f} {row['turnover']:5.1f}")
    for control_str, row in out["controls"].items():
        f = row["full"]
        print(f"{control_str:38s} {pct(f['cagr_float'])} {pct(f['volatility_float'])} {f['sharpe_float']:6.2f} {'':6s} {pct(f['max_drawdown_float']):>7s} "
              f"{row['to_2022']['sharpe_float']:6.2f} {row['seen_2023_on']['sharpe_float']:6.2f}")
    r = out["random"]
    print(f"{'RANDOM 200 books (5/50/95 pct)':38s} {pct(r['cagr']['50'])} {'':6s} {r['sharpe']['5']:.2f}/{r['sharpe']['50']:.2f}/{r['sharpe']['95']:.2f} "
          f"DD {pct(r['max_drawdown']['50'])} | <=22 {r['sharpe_to_2022']['50']:.2f} | 23+ {r['sharpe_seen']['50']:.2f} | turnover {r['turnover_mean']:.1f}")
    print("\nselection vs controls (P finalist > control, gap, verdict):")
    for name_str, row in out["finalists"].items():
        print(f"  {name_str}")
        for key_str, v in row["vs_control"].items():
            print(f"     {key_str:52s} P {v['p']:.2f} gap {v['gap']:+.2f} {v['verdict']}")
    print("SELECTION VERDICT (design of record):", out["selection"])
    print("\nalpha (annual, t) weekly NW4:")
    for name_str, row in out["finalists"].items():
        print(f"  {name_str}")
        for key_str, v in row["alpha"].items():
            print(f"     {key_str:34s} alpha {v['alpha_ann']:6.1%} t {v['alpha_t']:5.2f} beta_QQQ {v.get('b_QQQ', float('nan')):.2f} R2 {v['r2']:.2f} weeks {v['weeks']}")
    print("\ncrises:")
    for name_str in list(out["finalists"]) + list(out["controls"]):
        row = out["finalists"].get(name_str) or out["controls"][name_str]
        print(f"  {name_str:38s}", [round(x * 100, 1) for x in row["crisis"].values()])
    print("\nbook 60/40:")
    for label_str in ("real_taa_2012_on", "with_proxy_2008_on"):
        print(" ", label_str, "| TAA proxy overlap", out["book"]["taa_proxy_overlap"])
        for name_str, v in out["book"][label_str].items():
            if isinstance(v, dict) and "sharpe_float" in v:
                print(f"     {name_str:38s} CAGR {pct(v['cagr_float'])} Vol {pct(v['volatility_float'])} Sharpe {v['sharpe_float']:.2f} MaxDD {pct(v['max_drawdown_float'])} "
                      f"<=22 {v.get('to_2022_sharpe', float('nan')):.2f} 23+ {v.get('seen_2023_on_sharpe', float('nan')):.2f}")
        print("     P(design book > live book)", out["book"][label_str]["p_design_book_gt_live_book"], "| corr with TAA (monthly)",
              {k: round(v, 2) for k, v in out["book"][label_str]["corr_taa_monthly"].items()})
    print("\nwhole shares (2023+ decisions): sub-share positions / exposure lost")
    for name_str, row in out["finalists"].items():
        print(f"  {name_str:24s}", " | ".join(f"${int(s) // 1000}K {v['sub_share_position_share']:.0%}/{v['exposure_lost_share']:.0%}" for s, v in row["whole_share"].items()))
    print("\ncorrelation (monthly):", out["correlation"]["monthly"])
    print("head to head P:", {k: round(v, 2) for k, v in out["head_to_head_p"].items()})


if __name__ == "__main__":
    main()
