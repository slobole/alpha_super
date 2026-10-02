"""Masters' linear trend as the NDX ranking (owner request 2026-10-03). Registered in the Scout ledger before any number
is computed. Everything else is the live NDX VXN rule (VXN scaling, SPY SMA200 regime, Close > SMA100, top 10).

    LT = OLS slope of ln C over n sessions x (n - 1) / ATR_ln(A) x R2        (alpha/scout/specs/ndx_vxn.py)

Configurations: n {126, 189, 252} sessions x A {20, 252}, with R2; plus n 252 / A 20 without R2 (isolates R2).
Primary (named before the run): n 252 / A 20, the analogue of the 12-month ROC over ATR20 of the NATR20 score.
Compared with the live pod (dollar ATR) and NATR20. Robustness:
- luck band: the 16 rebalance offsets for the primary, NATR20 and the live pod;
- random parameters: the same 200 draws as A15 (plans.ndx_sample, seed 20261002), mapped to n = 21 x ROC months and
  A = the drawn ATR window, so each draw is a paired comparison with NATR20 on identical parameters;
- the signal itself at month ends: Masters battery (S2) and the monthly rank IC with the next month's excess return
  over the member mean.
Net of the house parity costs, idle cash at the T-bill rate; in sample 2000-09-01 to the seal; 2023 on = seen.

    uv run python scripts/research/scout_robustness_20261002/ndx_linear_trend.py
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
from scipy import stats

from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH, Ledger
from alpha.scout.metrics import sharpe_float
from alpha.scout.registration import Registration, register, registration_rows
from alpha.scout.stations.robustness import paired_sharpe_probability
from alpha.scout.stations.s2_indicator import masters_battery
from alpha.scout.stations.s4_strategy import SEAL_END_STR
from ndx_ensemble_cmma import _init, _task, summary
from plans import DRAW_COUNT_INT, SEED_INT, ndx_sample
from run import market_inputs

OUTPUT_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "robustness" / "ndx_linear_trend.json"
START_STR = "2000-09-01"
LT_DICT = {"atr_unit_str": "linear_trend"}
PRIMARY = {**LT_DICT, "lt_lookback_int": 252, "lt_atr_int": 20}
GRID_LIST = [{**LT_DICT, "lt_lookback_int": n, "lt_atr_int": a} for n in (126, 189, 252) for a in (20, 252)] + [{**PRIMARY, "lt_rsq_bool": False}]
BASE_DICT = {"Live (dollar ATR)": {}, "NATR20": {"atr_unit_str": "percent"}, "LT 252/20 (primary)": PRIMARY}
REGISTRATION = Registration(
    registration_id_str="ndx_vxn_linear_trend_ranking_20261003", family_id_str="equity_cross_sectional_momentum",
    hypothesis_str=("NDX VXN: ranking members by Masters' linear trend (OLS slope of log price over n sessions, in ATR "
                    "units, times R2) picks smoother, more persistent winners than ROC over ATR, and beats the NATR20 "
                    "ranking on identical parameters more often than not."),
    mechanism_str="Cross-sectional momentum; a path-based, volatility-normalised trend estimate penalising choppy paths.",
    expected_sign_and_location_str="Higher Sharpe than NATR20 on most random parameter draws; similar or lower drawdown.",
    hypothesis_class_str="X", universe_str="Nasdaq-100 point-in-time members", horizon_str="months",
    schedule_str="month-end decision", execution_str="next session's open",
    param_grid_dict={"lt_lookback_int": (126, 189, 252), "lt_atr_int": (20, 252), "lt_rsq_bool": (True, False)},
    grid_justification_str="",
    primary_metric_str="In-sample net Sharpe; paired win rate vs NATR20 on 200 identical random draws; 16-offset luck band",
    kill_criteria_str="Not a promotion candidate; any change goes through S4-S6 and a forward shadow.",
    source_str="Masters, Statistically Sound Indicators (linear trend); alpha/scout/specs/ndx_vxn.py linear_trend_frame",
    parent_id_str="ndx_vxn_reaudition_20261002",
    universe_choice_str="The pod's own universe (Nasdaq-100 point-in-time), unchanged.",
)


def signal_quality(inputs) -> dict:
    """Masters battery and the monthly rank IC of each ranking score among members at month ends."""
    from alpha.scout.specs import ndx_vxn

    decision_index = ndx_vxn.decision_dates(inputs.close_df.index)
    decision_index = decision_index[(decision_index >= START_STR) & (decision_index <= SEAL_END_STR)]
    stock_list = [s for s in inputs.close_df.columns if s != ndx_vxn.REGIME_SYMBOL_STR]
    close_df = inputs.close_df[stock_list]
    member_df = (inputs.member_df[stock_list] == 1).reindex(decision_index)
    month_close_df = close_df.reindex(decision_index)
    forward_df = month_close_df.shift(-1) / month_close_df - 1.0
    forward_excess_df = forward_df.sub(forward_df.where(member_df).mean(axis=1), axis=0)
    # ROC over 12 decision month-ends, as the pod; NATR20 = ROC / (ATR20 / Close).
    roc_df = month_close_df / month_close_df.shift(12) - 1.0
    previous_close_df = close_df.shift(1)
    true_range_df = np.maximum(inputs.high_df[stock_list] - inputs.low_df[stock_list],
                               np.maximum((inputs.high_df[stock_list] - previous_close_df).abs(), (inputs.low_df[stock_list] - previous_close_df).abs()))
    natr_df = (true_range_df.rolling(20, min_periods=20).mean() / close_df).reindex(decision_index)
    lt_raw_df, lt_compressed_df = ndx_vxn.linear_trend_frame(inputs, 252, 20)
    score_dict = {"NATR20 score (ROC12 / NATR20)": roc_df / natr_df, "ROC12": roc_df,
                  "LT 252/20 (compressed)": lt_compressed_df[stock_list].reindex(decision_index)}
    out_dict = {}
    for name_str, score_df in score_dict.items():
        score_df = score_df.replace([np.inf, -np.inf], np.nan)
        eligible_df = member_df & score_df.notna() & forward_excess_df.notna()
        battery = masters_battery(score_df, eligible_df, forward_excess_df)
        ic_list = []
        for ts in decision_index[:-1]:
            mask = eligible_df.loc[ts]
            if mask.sum() >= 20:
                ic_list.append(stats.spearmanr(score_df.loc[ts, mask], forward_excess_df.loc[ts, mask]).statistic)
        ic_vec = np.array(ic_list, dtype=float)
        out_dict[name_str] = {"battery": battery, "ic_mean": float(np.nanmean(ic_vec)), "ic_t": float(np.nanmean(ic_vec) / np.nanstd(ic_vec, ddof=1) * np.sqrt(np.isfinite(ic_vec).sum())),
                              "ic_positive_share": float(np.nanmean(ic_vec > 0)), "months": int(np.isfinite(ic_vec).sum())}
    return out_dict


def main() -> None:
    from alpha.scout.specs import ndx_vxn

    ledger = Ledger()
    if REGISTRATION.registration_id_str not in registration_rows(ledger):
        register(ledger, REGISTRATION)
        print("registered", REGISTRATION.registration_id_str, flush=True)
    _, tbill_ser = market_inputs()
    inputs = ndx_vxn.load_inputs()
    rng_obj = np.random.default_rng(SEED_INT)
    draw_list = [ndx_sample(rng_obj) for _ in range(DRAW_COUNT_INT)]
    task_list = [(k, [v], o) for k, v in BASE_DICT.items() for o in range(16)]
    task_list += [(f"grid {i}", [g], 0) for i, g in enumerate(GRID_LIST)]
    task_list += [(f"draw {i} {name}", [{**d, **extra}], 0) for i, d in enumerate(draw_list)
                  for name, extra in (("natr", {"atr_unit_str": "percent"}),
                                      ("lt", {**LT_DICT, "lt_lookback_int": 21 * d["roc_month_int"], "lt_atr_int": d["atr_window_int"]}))]
    with Pool(8, initializer=_init, initargs=(inputs, tbill_ser)) as pool_obj:
        result_list = pool_obj.map(_task, task_list, chunksize=2)
    failed_list = [{k: v for k, v in r.items() if k != "daily"} for r in result_list if "error" in r]
    by_key = {(r["key"], r["offset"]): r for r in result_list if "error" not in r}

    natr_in_ser = by_key[("NATR20", 0)]["daily"].loc[:SEAL_END_STR]
    base_out = {}
    for key_str in BASE_DICT:
        luck_vec = np.array([sharpe_float(by_key[(key_str, o)]["daily"].loc[:SEAL_END_STR]) for o in range(16) if (key_str, o) in by_key])
        main_dict = by_key[(key_str, 0)]
        base_out[key_str] = {**summary(main_dict["daily"], tbill_ser), "turnover": main_dict["turnover_float"],
                             "luck": {"min": float(luck_vec.min()), "median": float(np.median(luck_vec)), "max": float(luck_vec.max())},
                             "p_vs_natr": paired_sharpe_probability(main_dict["daily"].loc[:SEAL_END_STR], natr_in_ser)}
    grid_out = []
    for i, g in enumerate(GRID_LIST):
        daily_ser = by_key[(f"grid {i}", 0)]["daily"]
        grid_out.append({**g, **summary(daily_ser, tbill_ser), "p_vs_natr": paired_sharpe_probability(daily_ser.loc[:SEAL_END_STR], natr_in_ser)})
    pair_list = []
    for i in range(DRAW_COUNT_INT):
        a, b = by_key.get((f"draw {i} natr", 0)), by_key.get((f"draw {i} lt", 0))
        if a is None or b is None:
            continue
        pair_list.append({"natr": sharpe_float(a["daily"].loc[:SEAL_END_STR]), "lt": sharpe_float(b["daily"].loc[:SEAL_END_STR]),
                          "natr_seen": sharpe_float(a["daily"].loc["2023-01-01":]), "lt_seen": sharpe_float(b["daily"].loc["2023-01-01":])})
    pair_df = pd.DataFrame(pair_list)
    quality_dict = signal_quality(inputs)

    def curve(daily_ser):
        return [round(float(v), 4) for v in (1.0 + daily_ser).cumprod().resample("ME").last()]

    out_dict = {"base": base_out, "grid": grid_out, "pairs": pair_list, "quality": quality_dict, "failed": failed_list,
                "curves": {k: curve(by_key[(k, 0)]["daily"]) for k in BASE_DICT}}
    OUTPUT_PATH.write_text(json.dumps(out_dict, default=float), encoding="utf-8")

    print("failed", len(failed_list), [f.get("error") for f in failed_list[:3]])
    for k, v in base_out.items():
        s, t = v["in_sample"], v["seen_2023_on"]
        print(f"{k:22s} Sh {s['sharpe_float']:.2f} CAGR {s['cagr_float']:.1%} vol {s['volatility_float']:.1%} DD {s['max_drawdown_float']:.1%} Calmar {s['calmar_float']:.2f} "
              f"| luck {v['luck']['min']:.2f}/{v['luck']['median']:.2f}/{v['luck']['max']:.2f} | P>NATR {v['p_vs_natr']:.2f} | turn {v['turnover']:.1f} "
              f"| 2023+ Sh {t['sharpe_float']:.2f} CAGR {t['cagr_float']:.1%} DD {t['max_drawdown_float']:.1%}")
        print(f"{'':22s} crises {[round(x * 100, 1) for x in v['crisis'].values()]}")
    for g in grid_out:
        s = g["in_sample"]
        print(f"LT n{g['lt_lookback_int']} A{g['lt_atr_int']} rsq {g.get('lt_rsq_bool', True)}: Sh {s['sharpe_float']:.2f} CAGR {s['cagr_float']:.1%} "
              f"DD {s['max_drawdown_float']:.1%} P>NATR {g['p_vs_natr']:.2f} 2023+ {g['seen_2023_on']['sharpe_float']:.2f}")
    print(f"draws: n {len(pair_df)}, LT > NATR {float((pair_df['lt'] > pair_df['natr']).mean()):.2f}, median LT {pair_df['lt'].median():.2f} vs NATR {pair_df['natr'].median():.2f}; "
          f"seen: LT > NATR {float((pair_df['lt_seen'] > pair_df['natr_seen']).mean()):.2f}, median {pair_df['lt_seen'].median():.2f} vs {pair_df['natr_seen'].median():.2f}")
    for k, q in quality_dict.items():
        b = q["battery"]
        mi = b.get("mutual_information", {})
        print(f"{k:30s} IC {q['ic_mean']:.3f} (t {q['ic_t']:.1f}, positive {q['ic_positive_share']:.0%}, {q['months']} mo) | range/IQR {b['range_over_iqr_float']:.1f} "
              f"tail {b['tail_share_float']:.2%} entropy {b['relative_entropy_float']:.2f} break {b['mean_break']['sup_wald_float']:.1f} MI {mi.get('bits_float', float('nan')):.4f} "
              f"(shuf max {mi.get('shuffled_max_float', float('nan')):.4f})")


if __name__ == "__main__":
    main()
