"""NDX stock trend-filter bake-off (owner request 2026-10-03), built to resist overfitting. Two Scout registrations are
written before any number is computed: this bake-off, and the final candidate C (ndx_final_candidate.py) whose filter
is chosen by this bake-off's rule.

Base: NATR20 VXN (live NDX VXN rule, NATR20 ranking, ROC 12, top 10). Ten filters with FIXED parameters (no tuning):
    F0 Close > SMA100 (live)          F5 Close > LowPass(100)        (Zorro Workshop 4a, Ehlers)
    F1 Close > SMA200                 F6 Close > LowPass(200)
    F2 Close / SMA200 > +5%           F7 LowPass(200) rising
    F3 Close / SMA200 > +10%          F8 CORE5 rule: SMA10 > adaptive AMA (CORE5 parameters)
    F4 CMMA(200, 252) > +10           F9 no filter
Evidence per filter (in sample = 2000-09-01 to the vault seal; 2023 on = seen; net, idle cash at T-bill):
    1. family-wise test: Romano-Wolf step-down adjusted p of Sharpe(Fk) - Sharpe(F0) > 0 over the nine alternatives
       (paired stationary bootstrap, mean block 21, 2,000 draws);
    2. parameters: the same 200 random draws of the other parameters (plans.ndx_sample without the SMA length, seed
       20261002) for every filter; the paired win rate against F0 on identical parameters;
    3. eras: Sharpe in 2000-09 to 2007, 2008-2015, 2016-2022, and 2023 on (seen);
    4. rebalance-day luck: the 16 decision offsets; the median is the planning number;
    5. mechanism: exposure, names held, turnover, the share of members the filter lets through and its monthly flip rate;
    6. selection process (diagnostic): each January from 2006, pick the filter with the best anchored trailing Sharpe
       and hold it for the year; does the stitched out-of-sample record beat F0?
DECISION RULE (fixed before the run): a filter replaces F0 only if ALL hold:
    RW-adjusted p <= 0.10; paired win rate >= 0.70; Sharpe above F0 in >= 2 of the 3 in-sample eras;
    luck-band median >= F0's; after 2022 (seen) no more than 0.10 below F0.
    Several pass -> the highest paired win rate (not the highest Sharpe). None -> keep F0.

    uv run python scripts/research/scout_robustness_20261002/ndx_filter_bakeoff.py
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

from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH, Ledger
from alpha.scout.metrics import sharpe_float
from alpha.scout.registration import Registration, register, registration_rows
from alpha.scout.stations.robustness import paired_sharpe_difference_draws, romano_wolf_stepdown
from alpha.scout.stations.s4_strategy import SEAL_END_STR
from ndx_ensemble_cmma import _init, _task, summary
from plans import DRAW_COUNT_INT, SEED_INT, ndx_sample
from run import market_inputs

OUTPUT_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "robustness" / "ndx_filter_bakeoff.json"
START_STR = "2000-09-01"
NATR_DICT = {"atr_unit_str": "percent"}
FILTER_DICT = {
    "F0 Close > SMA100 (live)": {},
    "F1 Close > SMA200": {"stock_sma_int": 200},
    "F2 Close/SMA200 > +5%": {"stock_sma_int": 200, "trend_threshold_float": 0.05},
    "F3 Close/SMA200 > +10%": {"stock_sma_int": 200, "trend_threshold_float": 0.10},
    "F4 CMMA(200) > +10": {"trend_filter_str": "cmma", "stock_sma_int": 200, "cmma_atr_int": 252, "cmma_threshold_float": 10.0},
    "F5 Close > LowPass(100)": {"trend_filter_str": "lowpass", "stock_sma_int": 100},
    "F6 Close > LowPass(200)": {"trend_filter_str": "lowpass", "stock_sma_int": 200},
    "F7 LowPass(200) rising": {"trend_filter_str": "lowpass_rising", "stock_sma_int": 200},
    "F8 CORE5 adaptive AMA": {"trend_filter_str": "adaptive_ama"},
    "F9 No filter": {"stock_trend_filter_bool": False},
}
BASE_KEY_STR = "F0 Close > SMA100 (live)"
ERA_DICT = {"2000-09 to 2007": (START_STR, "2007-12-31"), "2008-2015": ("2008-01-01", "2015-12-31"), "2016-2022": ("2016-01-01", SEAL_END_STR)}
RULE_DICT = {"rw_p_max": 0.10, "win_rate_min": 0.70, "eras_min": 2, "seen_tolerance": 0.10}
COMMON_DICT = {
    "family_id_str": "equity_cross_sectional_momentum", "hypothesis_class_str": "X", "universe_str": "Nasdaq-100 point-in-time members",
    "horizon_str": "months", "schedule_str": "month-end decision", "execution_str": "next session's open",
    "universe_choice_str": "The pod's own universe (Nasdaq-100 point-in-time), unchanged.",
}
BAKEOFF_REGISTRATION = Registration(
    registration_id_str="ndx_natr20_vxn_filter_bakeoff_20261003",
    hypothesis_str=("On the NATR20 VXN rule, one of nine pre-specified stock trend filters (fixed parameters: SMA200 levels, "
                    "CMMA(200), Zorro's second-order low-pass, the CORE5 adaptive AMA, none) beats the live Close > SMA100 "
                    "by a margin that survives a family-wise correction and holds across parameters, eras and rebalance days."),
    mechanism_str="Cross-sectional momentum; the stock trend filter removes names in a downtrend before ranking.",
    expected_sign_and_location_str="Most likely none passes (the earlier map was flat); a long filter (SMA200 family) is the prior favourite.",
    param_grid_dict={"trend_filter": tuple(FILTER_DICT)},
    primary_metric_str="Romano-Wolf adjusted p of the in-sample Sharpe difference vs F0; paired win rate on 200 random draws",
    kill_criteria_str=("Decision rule: a filter replaces F0 only if RW-adjusted p <= 0.10, paired win rate >= 0.70, Sharpe above F0 "
                       "in >= 2 of 3 in-sample eras, luck-band median >= F0's, and after 2022 no more than 0.10 below F0; several "
                       "-> highest win rate; none -> keep F0."),
    source_str="Zorro manual Workshop 4a (LowPass); CORE5 (alpha/scout/specs/core5.py); A15 follow-ups (SMA200, CMMA)",
    parent_id_str="ndx_natr20_vxn_plateau_filter_20261002", **COMMON_DICT,
)
CANDIDATE_REGISTRATION = Registration(
    registration_id_str="ndx_rank_horizon_blend_candidate_20261003",
    hypothesis_str=("An equal-capital blend of nine NDX sub-books - rankings {dollar ATR, NATR20, linear trend (ATR20, R2)} x "
                    "horizons {9, 12, 15} months (linear trend n = 189, 252, 315 sessions), each top 10 with VXN scaling and "
                    "the SPY SMA200 regime, stock filter = the bake-off result (F0 unless a filter passes) - is at least as "
                    "good as the live pod across rebalance days and parameters, with smaller crash losses."),
    mechanism_str="Ranking and horizon diversification: the three rankings won in different regimes (A15 follow-ups).",
    expected_sign_and_location_str="Luck-band median at or above the live pod's; crash losses between the parents'.",
    param_grid_dict={"ranking": ("dollar ATR", "NATR20", "linear trend"), "horizon_month_int": (9, 12, 15)},
    primary_metric_str="Luck-band median vs the live pod; paired win rate vs the live pod on 200 random draws",
    kill_criteria_str=("C becomes the shadow candidate beside the live pod if its luck-band median >= the live pod's and it beats "
                       "the live pod on >= 70% of 200 paired random draws. Its components were chosen after seeing in-sample "
                       "data, so in-sample numbers are optimistic; any promotion needs forward (shadow) evidence."),
    source_str="A15 follow-ups (ranking comparison, plateau ensemble, linear trend); ndx_final_candidate.py",
    parent_id_str="ndx_vxn_reaudition_20261002", **COMMON_DICT,
)


def filter_state_frames(inputs) -> dict[str, pd.DataFrame]:
    """Each filter's 1/0 state per stock and session (what the spec applies at a decision)."""
    from alpha.scout.specs import ndx_vxn

    close_df = inputs.close_df
    out_dict = {}
    for name_str, override_dict in FILTER_DICT.items():
        kind_str = override_dict.get("trend_filter_str", "sma")
        length_int = override_dict.get("stock_sma_int", 100)
        if not override_dict.get("stock_trend_filter_bool", True):
            out_dict[name_str] = close_df.notna().astype(float)
        elif kind_str == "sma":
            sma_df = close_df.rolling(length_int, min_periods=length_int).mean()
            threshold_float = override_dict.get("trend_threshold_float", 0.0)
            out_dict[name_str] = ((close_df / sma_df - 1.0) > threshold_float).astype(float).where(sma_df.notna())
        elif kind_str == "cmma":
            cmma_df = ndx_vxn.cmma_frame(inputs, length_int, override_dict["cmma_atr_int"])
            out_dict[name_str] = (cmma_df > override_dict["cmma_threshold_float"]).astype(float).where(cmma_df.notna())
        else:
            out_dict[name_str] = ndx_vxn.trend_state_frame(inputs, kind_str, length_int)
    return out_dict


def state_diagnostics(inputs, state_dict: dict) -> dict:
    from alpha.scout.specs import ndx_vxn

    decision_index = ndx_vxn.decision_dates(inputs.close_df.index)
    decision_index = decision_index[(decision_index >= START_STR) & (decision_index <= SEAL_END_STR)]
    out_dict = {}
    for name_str, state_df in state_dict.items():
        stock_list = [s for s in state_df.columns if s != ndx_vxn.REGIME_SYMBOL_STR]
        member_df = (inputs.member_df[stock_list] == 1).reindex(decision_index)
        month_df = state_df[stock_list].reindex(decision_index).where(member_df)
        both_df = month_df.notna() & month_df.shift(1).notna()
        flip_float = float((month_df - month_df.shift(1)).abs().where(both_df).stack().mean())
        out_dict[name_str] = {"share_on": float(month_df.stack().mean()), "flip_rate": flip_float}
    return out_dict


def walk_forward(in_dict: dict[str, pd.Series], seen_dict: dict[str, pd.Series]) -> dict:
    """Each January from 2006: the filter with the best anchored trailing Sharpe (from 2000-09) is held for the year."""
    full_dict = {k: pd.concat([in_dict[k], seen_dict[k]]).sort_index() for k in in_dict}
    frame = pd.DataFrame(full_dict).dropna()
    chosen_list, stitched_list = [], []
    for year_int in range(2006, int(frame.index[-1].year) + 1):
        train_df = frame.loc[: f"{year_int - 1}-12-31"]
        test_df = frame.loc[f"{year_int}-01-01": f"{year_int}-12-31"]
        if test_df.empty:
            continue
        best_str = max(train_df.columns, key=lambda c: sharpe_float(train_df[c]))
        chosen_list.append({"year": year_int, "filter": best_str})
        stitched_list.append(test_df[best_str])
    stitched_ser = pd.concat(stitched_list)
    base_ser = frame[BASE_KEY_STR].loc[stitched_ser.index]
    observed_vec, draw_mat = paired_sharpe_difference_draws(base_ser.loc[:SEAL_END_STR].to_numpy(), stitched_ser.loc[:SEAL_END_STR].to_numpy(),
                                                            random_seed_int=SEED_INT)
    return {"chosen": chosen_list,
            "oos_sharpe_2006_2022": sharpe_float(stitched_ser.loc[:SEAL_END_STR]), "f0_sharpe_2006_2022": sharpe_float(base_ser.loc[:SEAL_END_STR]),
            "p_oos_better": float(np.mean(draw_mat[:, 0] > 0)), "difference": float(observed_vec[0]),
            "oos_sharpe_2023_on": sharpe_float(stitched_ser.loc["2023-01-01":]), "f0_sharpe_2023_on": sharpe_float(base_ser.loc["2023-01-01":])}


def main() -> None:
    from alpha.scout.specs import ndx_vxn

    ledger = Ledger()
    for registration in (BAKEOFF_REGISTRATION, CANDIDATE_REGISTRATION):
        if registration.registration_id_str not in registration_rows(ledger):
            register(ledger, registration)
            print("registered", registration.registration_id_str, flush=True)
    _, tbill_ser = market_inputs()
    inputs = ndx_vxn.load_inputs()
    rng_obj = np.random.default_rng(SEED_INT)
    draw_list = [{k: v for k, v in ndx_sample(rng_obj).items() if k != "stock_sma_int"} for _ in range(DRAW_COUNT_INT)]
    task_list = [(name_str, [{**NATR_DICT, **f}], o) for name_str, f in FILTER_DICT.items() for o in range(16)]
    task_list += [(f"draw {i}|{name_str}", [{**d, **NATR_DICT, **f}], 0) for i, d in enumerate(draw_list) for name_str, f in FILTER_DICT.items()]
    with Pool(12, initializer=_init, initargs=(inputs, tbill_ser)) as pool_obj:
        result_list = pool_obj.map(_task, task_list, chunksize=4)
    failed_list = [{k: v for k, v in r.items() if k != "daily"} for r in result_list if "error" in r]
    by_key = {(r["key"], r["offset"]): r for r in result_list if "error" not in r}

    name_list = list(FILTER_DICT)
    in_dict = {n: by_key[(n, 0)]["daily"].loc[:SEAL_END_STR] for n in name_list}
    seen_dict = {n: by_key[(n, 0)]["daily"].loc["2023-01-01":] for n in name_list}
    alt_list = [n for n in name_list if n != BASE_KEY_STR]
    observed_vec, draw_mat = paired_sharpe_difference_draws(in_dict[BASE_KEY_STR].to_numpy(), np.column_stack([in_dict[n].to_numpy() for n in alt_list]),
                                                            random_seed_int=SEED_INT)
    rw_vec = romano_wolf_stepdown(observed_vec, draw_mat)
    rw_dict = dict(zip(alt_list, rw_vec))
    raw_p_dict = {n: float(np.mean(draw_mat[:, j] <= 0)) for j, n in enumerate(alt_list)}

    draw_sharpe = {n: np.array([sharpe_float(by_key[(f"draw {i}|{n}", 0)]["daily"].loc[:SEAL_END_STR]) if (f"draw {i}|{n}", 0) in by_key else np.nan
                                for i in range(DRAW_COUNT_INT)]) for n in name_list}
    base_luck_median = float(np.median([sharpe_float(by_key[(BASE_KEY_STR, o)]["daily"].loc[:SEAL_END_STR]) for o in range(16)]))
    stat_out = {}
    for n in name_list:
        luck_vec = np.array([sharpe_float(by_key[(n, o)]["daily"].loc[:SEAL_END_STR]) for o in range(16) if (n, o) in by_key])
        era_dict = {e: sharpe_float(in_dict[n].loc[a:b]) for e, (a, b) in ERA_DICT.items()}
        base_era_dict = {e: sharpe_float(in_dict[BASE_KEY_STR].loc[a:b]) for e, (a, b) in ERA_DICT.items()}
        win_float = float(np.nanmean(draw_sharpe[n] > draw_sharpe[BASE_KEY_STR])) if n != BASE_KEY_STR else float("nan")
        main_dict = by_key[(n, 0)]
        row = {**summary(main_dict["daily"], tbill_ser), "era": era_dict, "seen_sharpe": sharpe_float(seen_dict[n]),
               "luck": {"min": float(luck_vec.min()), "median": float(np.median(luck_vec)), "max": float(luck_vec.max())},
               "draw_median": float(np.nanmedian(draw_sharpe[n])), "win_rate_vs_f0": win_float,
               "rw_p": float(rw_dict.get(n, np.nan)), "raw_p": float(raw_p_dict.get(n, np.nan)),
               "exposure": main_dict["exposure_float"], "names_held": main_dict["names_held_float"], "turnover": main_dict["turnover_float"]}
        if n != BASE_KEY_STR:
            check_dict = {
                "rw_p": row["rw_p"] <= RULE_DICT["rw_p_max"],
                "win_rate": win_float >= RULE_DICT["win_rate_min"],
                "eras": sum(era_dict[e] > base_era_dict[e] for e in ERA_DICT) >= RULE_DICT["eras_min"],
                "luck_median": row["luck"]["median"] >= base_luck_median,
                "seen": row["seen_sharpe"] >= sharpe_float(seen_dict[BASE_KEY_STR]) - RULE_DICT["seen_tolerance"],
            }
            row["checks"] = check_dict
            row["passes_bool"] = all(check_dict.values())
        stat_out[n] = row
    passing_list = [n for n in alt_list if stat_out[n]["passes_bool"]]
    selected_str = max(passing_list, key=lambda n: stat_out[n]["win_rate_vs_f0"]) if passing_list else BASE_KEY_STR
    diagnostic_dict = state_diagnostics(inputs, filter_state_frames(inputs))
    walk_dict = walk_forward(in_dict, seen_dict)
    out_dict = {"filters": FILTER_DICT, "stats": stat_out, "diagnostics": diagnostic_dict, "walk_forward": walk_dict, "rule": RULE_DICT,
                "selected_filter": selected_str, "selected_override": FILTER_DICT[selected_str], "passing": passing_list, "failed": failed_list,
                "draw_sharpe": {n: [None if not np.isfinite(v) else round(float(v), 4) for v in draw_sharpe[n]] for n in name_list},
                "base_luck_median": base_luck_median}
    OUTPUT_PATH.write_text(json.dumps(out_dict, default=float), encoding="utf-8")

    print("failed", len(failed_list), [f.get("error") for f in failed_list[:3]])
    print(f"{'filter':26s} {'Sh':>5s} {'CAGR':>6s} {'MaxDD':>7s} {'luckMed':>7s} {'drawMed':>7s} {'win':>5s} {'RWp':>5s} {'rawp':>5s} "
          f"{'eras (00-07/08-15/16-22)':>24s} {'2023+':>6s} {'expo':>5s} {'on%':>5s} {'flip':>5s}  verdict")
    for n in name_list:
        r, d = stat_out[n], diagnostic_dict[n]
        s = r["in_sample"]
        era_str = "/".join(f"{v:.2f}" for v in r["era"].values())
        verdict_str = "base" if n == BASE_KEY_STR else ("PASS" if r["passes_bool"] else "fail: " + ",".join(k for k, v in r["checks"].items() if not v))
        print(f"{n:26s} {s['sharpe_float']:5.2f} {s['cagr_float']:6.1%} {s['max_drawdown_float']:7.1%} {r['luck']['median']:7.2f} {r['draw_median']:7.2f} "
              f"{r['win_rate_vs_f0']:5.2f} {r['rw_p']:5.2f} {r['raw_p']:5.2f} {era_str:>24s} {r['seen_sharpe']:6.2f} {r['exposure']:5.2f} "
              f"{d['share_on']:5.2f} {d['flip_rate']:5.2f}  {verdict_str}")
        print(f"{'':26s} crises {[round(x * 100, 1) for x in r['crisis'].values()]}")
    print("selected:", selected_str, "| passing:", passing_list)
    print("walk-forward:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in walk_dict.items() if k != "chosen"})
    print("walk-forward picks:", [c["filter"].split()[0] for c in walk_dict["chosen"]])


if __name__ == "__main__":
    main()
