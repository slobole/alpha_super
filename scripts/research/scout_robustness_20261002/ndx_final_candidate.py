"""Final NDX candidate C (registered with the bake-off before any run: ndx_rank_horizon_blend_candidate_20261003).

C = equal capital in nine sub-books, rankings {dollar ATR, NATR20, linear trend (ATR20, R2)} x horizons {9, 12, 15}
month-ends (linear trend n = 21 x months = 189, 252, 315 sessions), each top 10 with VXN scaling and the SPY SMA200
regime; the stock filter is the bake-off's selected filter (results/scout/robustness/ndx_filter_bakeoff.json). The
nine target-weight frames are averaged and simulated as one account.

Compared with the live pod (dollar ATR, ROC 12, Close > SMA100) and with NATR20 single (ROC 12, the same filter as C):
in sample (2000-09-01 to the seal), eras, 2023 on (seen), crashes, the 16-offset luck band, the paired bootstrap, and
200 random draws of the other parameters (top count, ATR window, SPY regime SMA, VXN reference; plans.ndx_sample,
seed 20261002) applied to C and to the live pod alike.
DECISION RULE (registered): C becomes the shadow candidate if its luck-band median >= the live pod's and it beats the live
pod on >= 70% of the paired draws. In-sample numbers are optimistic (C's parts were chosen after seeing the data).

    uv run python scripts/research/scout_robustness_20261002/ndx_final_candidate.py
    uv run python scripts/research/scout_robustness_20261002/ndx_final_candidate.py --live-filter
        (the conservative re-run: Close > SMA100 for C and NATR20, after the family-wide Romano-Wolf over all 38 filters
        tried, ndx_filter_family_rw.py, found no filter significant: best RW p 0.19)
"""

from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH
from alpha.scout.metrics import sharpe_float
from alpha.scout.stations.robustness import paired_sharpe_probability
from alpha.scout.stations.s4_strategy import SEAL_END_STR
from ndx_ensemble_cmma import _init, _task, summary
from ndx_filter_bakeoff import ERA_DICT
from plans import DRAW_COUNT_INT, SEED_INT, ndx_sample
from run import market_inputs

BAKEOFF_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "robustness" / "ndx_filter_bakeoff.json"
OUTPUT_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "robustness" / "ndx_final_candidate.json"
HORIZON_TUPLE = (9, 12, 15)


def candidate_list(filter_dict: dict, draw_dict: dict | None = None) -> list[dict]:
    draw_dict = dict(draw_dict or {})
    atr_window_int = draw_dict.get("atr_window_int", 20)
    out_list = []
    for horizon_int in HORIZON_TUPLE:
        out_list.append({**draw_dict, **filter_dict, "roc_month_int": horizon_int})  # dollar ATR
        out_list.append({**draw_dict, **filter_dict, "roc_month_int": horizon_int, "atr_unit_str": "percent"})  # NATR20
        out_list.append({**draw_dict, **filter_dict, "roc_month_int": horizon_int, "atr_unit_str": "linear_trend",
                         "lt_lookback_int": 21 * horizon_int, "lt_atr_int": atr_window_int})  # linear trend
    return out_list


def main() -> None:
    from alpha.scout.specs import ndx_vxn

    bakeoff_dict = json.loads(BAKEOFF_PATH.read_text(encoding="utf-8"))
    live_filter_bool = "--live-filter" in sys.argv[1:]
    filter_dict = {} if live_filter_bool else bakeoff_dict["selected_override"]
    filter_name_str = "F0 Close > SMA100 (live)" if live_filter_bool else bakeoff_dict["selected_filter"]
    output_path = OUTPUT_PATH.with_name("ndx_final_candidate_live_filter.json") if live_filter_bool else OUTPUT_PATH
    print("filter:", filter_name_str, filter_dict, flush=True)
    _, tbill_ser = market_inputs()
    inputs = ndx_vxn.load_inputs()
    rng_obj = np.random.default_rng(SEED_INT)
    draw_list = [{k: v for k, v in ndx_sample(rng_obj).items() if k not in ("stock_sma_int", "roc_month_int")} for _ in range(DRAW_COUNT_INT)]
    strategy_dict = {"Live (dollar ATR)": [{}], "NATR20 single": [{**filter_dict, "atr_unit_str": "percent"}], "C (9-book blend)": candidate_list(filter_dict)}
    task_list = [(k, v, o) for k, v in strategy_dict.items() for o in range(16)]
    task_list += [(f"draw {i}|C", candidate_list(filter_dict, d), 0) for i, d in enumerate(draw_list)]
    task_list += [(f"draw {i}|live", [dict(d)], 0) for i, d in enumerate(draw_list)]
    with Pool(12, initializer=_init, initargs=(inputs, tbill_ser)) as pool_obj:
        result_list = pool_obj.map(_task, task_list, chunksize=2)
    failed_list = [{k: v for k, v in r.items() if k != "daily"} for r in result_list if "error" in r]
    by_key = {(r["key"], r["offset"]): r for r in result_list if "error" not in r}

    live_in_ser = by_key[("Live (dollar ATR)", 0)]["daily"].loc[:SEAL_END_STR]
    natr_in_ser = by_key[("NATR20 single", 0)]["daily"].loc[:SEAL_END_STR]
    out = {}
    for k in strategy_dict:
        main_dict = by_key[(k, 0)]
        daily_ser = main_dict["daily"]
        luck_vec = np.array([sharpe_float(by_key[(k, o)]["daily"].loc[:SEAL_END_STR]) for o in range(16) if (k, o) in by_key])
        out[k] = {**summary(daily_ser, tbill_ser), "era": {e: sharpe_float(daily_ser.loc[a:b]) for e, (a, b) in ERA_DICT.items()},
                  "seen_sharpe": sharpe_float(daily_ser.loc["2023-01-01":]),
                  "luck": {"min": float(luck_vec.min()), "median": float(np.median(luck_vec)), "max": float(luck_vec.max())},
                  "p_vs_live": paired_sharpe_probability(daily_ser.loc[:SEAL_END_STR], live_in_ser),
                  "p_vs_natr": paired_sharpe_probability(daily_ser.loc[:SEAL_END_STR], natr_in_ser),
                  "correlation_with_live": float(daily_ser.loc[:SEAL_END_STR].corr(live_in_ser)),
                  "exposure": main_dict["exposure_float"], "names_held": main_dict["names_held_float"], "turnover": main_dict["turnover_float"]}
    c_draw = np.array([sharpe_float(by_key[(f"draw {i}|C", 0)]["daily"].loc[:SEAL_END_STR]) if (f"draw {i}|C", 0) in by_key else np.nan for i in range(DRAW_COUNT_INT)])
    live_draw = np.array([sharpe_float(by_key[(f"draw {i}|live", 0)]["daily"].loc[:SEAL_END_STR]) if (f"draw {i}|live", 0) in by_key else np.nan for i in range(DRAW_COUNT_INT)])
    c_seen = np.array([sharpe_float(by_key[(f"draw {i}|C", 0)]["daily"].loc["2023-01-01":]) if (f"draw {i}|C", 0) in by_key else np.nan for i in range(DRAW_COUNT_INT)])
    live_seen = np.array([sharpe_float(by_key[(f"draw {i}|live", 0)]["daily"].loc["2023-01-01":]) if (f"draw {i}|live", 0) in by_key else np.nan for i in range(DRAW_COUNT_INT)])
    win_float = float(np.nanmean(c_draw > live_draw))
    decision_dict = {"luck_median_ok": out["C (9-book blend)"]["luck"]["median"] >= out["Live (dollar ATR)"]["luck"]["median"],
                     "win_rate": win_float, "win_rate_ok": win_float >= 0.70}
    decision_dict["shadow_candidate_bool"] = decision_dict["luck_median_ok"] and decision_dict["win_rate_ok"]
    out_dict = {"filter": filter_name_str, "strategies": out, "draws": {"c": c_draw.tolist(), "live": live_draw.tolist(),
                "c_seen": c_seen.tolist(), "live_seen": live_seen.tolist()}, "decision": decision_dict, "failed": failed_list}
    output_path.write_text(json.dumps(out_dict, default=float), encoding="utf-8")

    print("failed", len(failed_list), [f.get("error") for f in failed_list[:3]])
    for k, r in out.items():
        s = r["in_sample"]
        print(f"{k:20s} Sh {s['sharpe_float']:.2f} CAGR {s['cagr_float']:.1%} vol {s['volatility_float']:.1%} DD {s['max_drawdown_float']:.1%} "
              f"Calmar {s['calmar_float']:.2f} | luck {r['luck']['min']:.2f}/{r['luck']['median']:.2f}/{r['luck']['max']:.2f} | eras "
              f"{'/'.join(f'{v:.2f}' for v in r['era'].values())} | 2023+ {r['seen_sharpe']:.2f} | P>live {r['p_vs_live']:.2f} P>NATR {r['p_vs_natr']:.2f} "
              f"| corr live {r['correlation_with_live']:.2f} | names {r['names_held']:.1f} expo {r['exposure']:.2f} turn {r['turnover']:.1f}")
        print(f"{'':20s} crises {[round(x * 100, 1) for x in r['crisis'].values()]}")
    print(f"draws: C > live {win_float:.2f}; median C {np.nanmedian(c_draw):.2f} vs live {np.nanmedian(live_draw):.2f}; "
          f"seen: C > live {float(np.nanmean(c_seen > live_seen)):.2f}, medians {np.nanmedian(c_seen):.2f} vs {np.nanmedian(live_seen):.2f}")
    print("decision:", decision_dict)


if __name__ == "__main__":
    main()
