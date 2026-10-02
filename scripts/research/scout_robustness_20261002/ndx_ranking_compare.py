"""NDX ranking comparison (owner request 2026-10-02): dollar ATR (live) vs NATR20 vs ROC alone, everything else the live
NDX VXN rule (VXN scaling, SPY SMA200 regime, stock SMA100 filter, top 10, ROC 12 month-ends).

Net of the house parity costs, idle cash credited at the T-bill rate (as A15). In sample = 2000-09-01 to the vault
seal 2022-12-30; 2023 on is shown separately and marked "seen" (the period has been looked at; not a vault test).

Robustness:
- luck band: the 16 rebalance offsets (decide k = 0..15 sessions before month end), in-sample Sharpe per ranking;
- random parameters: the same 200 draws (plans.ndx_sample, seed 20261002) for each ranking, so every draw is a paired
  comparison of the rankings on identical parameters;
- paired stationary bootstrap P(Sharpe A > Sharpe B), mean block 21 sessions;
- Sharpe by era and rolling 36-month Sharpe.

    uv run python scripts/research/scout_robustness_20261002/ndx_ranking_compare.py
"""

from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH
from alpha.scout.metrics import performance_dict, sharpe_float
from alpha.scout.stations.robustness import paired_sharpe_probability
from alpha.scout.stations.s4_strategy import SEAL_END_STR
from plans import DRAW_COUNT_INT, SEED_INT, RobustPlan, ndx_sample
from run import PodRunner, market_inputs

OUTPUT_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "robustness" / "ndx_ranking_compare.json"
START_STR = "2000-09-01"
RANKING_DICT = {"Dollar ATR (live)": "dollar", "NATR20": "percent", "ROC only": "none"}
CRISIS_DICT = {
    "Dot-com bear 2000-09 to 2002-10": ("2000-09-01", "2002-10-09"),
    "GFC 2007-10 to 2009-03": ("2007-10-09", "2009-03-09"),
    "Euro / US downgrade 2011": ("2011-04-29", "2011-10-03"),
    "Q4 2018": ("2018-09-20", "2018-12-24"),
    "COVID crash 2020": ("2020-02-19", "2020-03-23"),
    "2022 bear": ("2022-01-03", "2022-10-12"),
    "Tariff drop 2025 (seen)": ("2025-02-19", "2025-04-08"),
}
PLAN = RobustPlan("NDX ranking compare", "ndx", "ndx_vxn", START_STR)
_WORKER: dict = {}


def _init(live_inputs, tbill_ser) -> None:
    _WORKER["runner"] = PodRunner(PLAN, live_inputs, tbill_ser)


def _task(override_dict: dict) -> float:
    try:
        return sharpe_float(_WORKER["runner"].daily(override_dict).loc[START_STR:SEAL_END_STR])
    except Exception:  # noqa: BLE001 - a failed run is a NaN, counted in the output
        return float("nan")


def window_return(daily_ser: pd.Series, start_str: str, end_str: str) -> float:
    part_ser = daily_ser.loc[start_str:end_str]
    return float((1.0 + part_ser).prod() - 1.0) if len(part_ser) else float("nan")


def main() -> None:
    market_dict, tbill_ser = market_inputs()
    runner = PodRunner(PLAN, tbill_ser=tbill_ser)
    daily_dict, turnover_dict = {}, {}
    for name_str, unit_str in RANKING_DICT.items():
        result = runner.result({"atr_unit_str": unit_str})
        daily_dict[name_str] = runner.daily({"atr_unit_str": unit_str}, result).loc[START_STR:]
        trade_df = result.trade_df.loc[result.trade_df["date"] >= START_STR]
        traded_value_ser = (trade_df["delta_float"].abs() * trade_df["price_float"]).groupby(trade_df["date"].dt.year).sum()
        value_ser = result.total_value_ser.loc[START_STR:].groupby(result.total_value_ser.loc[START_STR:].index.year).mean()
        turnover_dict[name_str] = float((traded_value_ser / value_ser).loc[:2022].mean())
    qqq_ser = market_dict["QQQ"].reindex(daily_dict["NATR20"].index).fillna(0.0)
    daily_dict["QQQ (buy and hold)"] = qqq_ser

    stat_dict = {}
    for name_str, daily_ser in daily_dict.items():
        in_ser, seen_ser = daily_ser.loc[:SEAL_END_STR], daily_ser.loc["2023-01-01":]
        month_ser = in_ser.resample("ME").apply(lambda s: (1 + s).prod() - 1)
        qqq_month_ser = qqq_ser.loc[:SEAL_END_STR].resample("ME").apply(lambda s: (1 + s).prod() - 1)
        down_mask = qqq_month_ser < 0
        stat_dict[name_str] = {
            "in_sample": performance_dict(in_ser, tbill_ser), "seen_2023_on": performance_dict(seen_ser, tbill_ser),
            "worst_month_float": float(month_ser.min()), "best_month_float": float(month_ser.max()),
            "down_capture_float": float(month_ser[down_mask].mean() / qqq_month_ser[down_mask].mean()),
            "up_capture_float": float(month_ser[~down_mask].mean() / qqq_month_ser[~down_mask].mean()),
            "era_sharpe_dict": {era: sharpe_float(in_ser.loc[a:b]) for era, (a, b) in
                                {"2000-09 to 2007": (START_STR, "2007-12-31"), "2008-2015": ("2008-01-01", "2015-12-31"),
                                 "2016-2022": ("2016-01-01", SEAL_END_STR), "2023 on (seen)": ("2023-01-01", "2026-12-31")}.items()},
            "crisis_dict": {k: window_return(daily_ser, a, b) for k, (a, b) in CRISIS_DICT.items()},
            "turnover_float": turnover_dict.get(name_str, float("nan")),
        }

    with Pool(8, initializer=_init, initargs=(runner.inputs({}), tbill_ser)) as pool_obj:
        luck_dict = {name_str: pool_obj.map(_task, [{"atr_unit_str": u, "decision_offset_int": k} for k in range(16)])
                     for name_str, u in RANKING_DICT.items()}
        rng_obj = np.random.default_rng(SEED_INT)
        draw_list = [ndx_sample(rng_obj) for _ in range(DRAW_COUNT_INT)]
        random_dict = {name_str: pool_obj.map(_task, [{**d, "atr_unit_str": u} for d in draw_list], chunksize=6)
                       for name_str, u in RANKING_DICT.items()}

    name_list = list(RANKING_DICT)
    in_dict = {n: daily_dict[n].loc[:SEAL_END_STR] for n in name_list}
    paired_dict = {f"{a} > {b}": paired_sharpe_probability(in_dict[a], in_dict[b]) for i, a in enumerate(name_list) for b in name_list[i + 1:]}
    paired_dict.update({f"{b} > {a}": 1.0 - v for k, v in list(paired_dict.items()) for a, b in [k.split(" > ")]})
    draw_mat = np.array([random_dict[n] for n in name_list])
    draw_win_dict = {f"{a} > {b}": float(np.nanmean(draw_mat[i] > draw_mat[j])) for i, a in enumerate(name_list) for j, b in enumerate(name_list) if i != j}

    def curve(daily_ser: pd.Series) -> list:
        value_ser = (1.0 + daily_ser).cumprod().resample("ME").last()
        return [round(float(v), 4) for v in value_ser]

    def drawdown(daily_ser: pd.Series) -> list:
        value_ser = (1.0 + daily_ser).cumprod()
        return [round(float(v), 4) for v in (value_ser / value_ser.cummax() - 1.0).resample("ME").min()]

    def rolling(daily_ser: pd.Series) -> list:
        month_ser = daily_ser.resample("ME").apply(lambda s: (1 + s).prod() - 1)
        roll_ser = month_ser.rolling(36).mean() / month_ser.rolling(36).std() * np.sqrt(12)
        return [None if not np.isfinite(v) else round(float(v), 3) for v in roll_ser]

    month_index = (1.0 + daily_dict["NATR20"]).cumprod().resample("ME").last().index
    out_dict = {
        "month_list": [d.strftime("%Y-%m") for d in month_index],
        "curve": {n: curve(s) for n, s in daily_dict.items()}, "drawdown": {n: drawdown(s) for n, s in daily_dict.items()},
        "rolling_sharpe": {n: rolling(s) for n, s in daily_dict.items()},
        "stats": stat_dict, "luck": luck_dict, "random": random_dict, "paired_bootstrap": paired_dict, "draw_win": draw_win_dict,
    }
    OUTPUT_PATH.write_text(json.dumps(out_dict, default=float), encoding="utf-8")

    for n in daily_dict:
        s, t = stat_dict[n]["in_sample"], stat_dict[n]["seen_2023_on"]
        print(f"{n:20s} IS: Sharpe {s['sharpe_float']:.2f} exSh {s['excess_sharpe_float']:.2f} CAGR {s['cagr_float']:.1%} vol {s['volatility_float']:.1%} "
              f"Sortino {s['sortino_float']:.2f} MaxDD {s['max_drawdown_float']:.1%} Calmar {s['calmar_float']:.2f} UW {s['longest_underwater_days_int']} "
              f"skew {s['skew_float']:.2f} ES95 {s['expected_shortfall_95_float']:.2%} | 2023+: Sh {t['sharpe_float']:.2f} CAGR {t['cagr_float']:.1%} DD {t['max_drawdown_float']:.1%}")
        print(f"{'':20s} worst mo {stat_dict[n]['worst_month_float']:.1%} down cap {stat_dict[n]['down_capture_float']:.2f} up cap {stat_dict[n]['up_capture_float']:.2f} "
              f"turnover {stat_dict[n]['turnover_float']:.2f} eras {[round(v, 2) for v in stat_dict[n]['era_sharpe_dict'].values()]}")
        print(f"{'':20s} crises {[round(v * 100, 1) for v in stat_dict[n]['crisis_dict'].values()]}")
    for n in name_list:
        luck_vec, draw_vec = np.array(luck_dict[n]), np.array(random_dict[n])
        print(f"{n:20s} luck min/med/max {np.nanmin(luck_vec):.2f}/{np.nanmedian(luck_vec):.2f}/{np.nanmax(luck_vec):.2f}; "
              f"draws median {np.nanmedian(draw_vec):.2f} p10 {np.nanquantile(draw_vec, 0.1):.2f} p90 {np.nanquantile(draw_vec, 0.9):.2f} nan {int(np.isnan(draw_vec).sum())}")
    print("paired bootstrap", {k: round(v, 2) for k, v in paired_dict.items()})
    print("same-draw wins", {k: round(v, 2) for k, v in draw_win_dict.items()})


if __name__ == "__main__":
    main()
