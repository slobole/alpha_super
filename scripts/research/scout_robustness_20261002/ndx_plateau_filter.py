"""NDX parameter plateaus and stock trend-filter variants (owner questions 2026-10-02), on NATR20 VXN (the recommended
ranking; everything else the live NDX VXN rule). Registered in the Scout ledger before any number is computed.

Surfaces (in-sample Sharpe, 2000-09-01 to the vault seal, net, idle cash at T-bill, as A15):
    S1  ROC month-ends {3, 6, 9, 12, 15, 18} x stock trend filter {none, Close > SMA 20 / 50 / 100 / 150 / 200 / 250}, top 10
    S2  ROC month-ends {3, ..., 18} x top count {5, 10, 15, 20, 25}, Close > SMA100
Each surface gets the Scout plateau rule (alpha.stats.selection.plateau_choice: the centre of the best plateau, by the
median of each cell and its grid neighbours). The filter axis in S1 is ordered by length (none = 0 sessions).
Filter variants (ROC 12, top 10):
    crossovers  SMA(fast) > SMA(slow) for (10, 50), (21, 50), (21, 100), (50, 100), (50, 200)
    distance    Close / SMA100 - 1 > -5%, +5%, +10%; Close / SMA200 - 1 > +5%, +10%
Each variant: Sharpe, CAGR, Max DD and P(variant Sharpe > live-filter Sharpe), paired stationary bootstrap.

    uv run python scripts/research/scout_robustness_20261002/ndx_plateau_filter.py
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
from alpha.scout.metrics import performance_dict
from alpha.scout.registration import Registration, register, registration_rows
from alpha.scout.stations.robustness import paired_sharpe_probability
from alpha.scout.stations.s4_strategy import SEAL_END_STR
from alpha.stats.selection import neighbourhood_median_vec, plateau_choice
from plans import RobustPlan
from run import PodRunner, market_inputs

OUTPUT_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "robustness" / "ndx_plateau_filter.json"
START_STR = "2000-09-01"
BASE_DICT = {"atr_unit_str": "percent"}  # NATR20 VXN
ROC_TUPLE, SMA_TUPLE, TOP_TUPLE = (3, 6, 9, 12, 15, 18), (0, 20, 50, 100, 150, 200, 250), (5, 10, 15, 20, 25)
CROSS_TUPLE = ((10, 50), (21, 50), (21, 100), (50, 100), (50, 200))
DISTANCE_TUPLE = ((100, -0.05), (100, 0.05), (100, 0.10), (200, 0.05), (200, 0.10))
PLAN = RobustPlan("NDX NATR20 VXN plateau", "ndx", "ndx_natr20_vxn", START_STR)
REGISTRATION = Registration(
    registration_id_str="ndx_natr20_vxn_plateau_filter_20261002", family_id_str="equity_cross_sectional_momentum",
    hypothesis_str=("NDX NATR20 VXN momentum: the ROC length, the number of stocks held and the stock trend filter (length, "
                    "SMA crossover, or a distance threshold above the SMA) form a plateau rather than a peak, and the live "
                    "Close > SMA100 filter is not materially beaten by its variants."),
    mechanism_str="Cross-sectional momentum in large-cap growth, trend filter removes broken names (sensitivity study).",
    expected_sign_and_location_str="A broad plateau around ROC 9-15 and SMA 100-200; filter variants within noise of the live one.",
    hypothesis_class_str="X", universe_str="Nasdaq-100 point-in-time members", horizon_str="months",
    schedule_str="month-end decision", execution_str="next session's open",
    param_grid_dict={"roc_month_int": ROC_TUPLE, "stock_trend_filter": ("none", 20, 50, 100, 150, 200, 250, "cross 10/50", "cross 21/50",
                     "cross 21/100", "cross 50/100", "cross 50/200", "dist100 -5%", "dist100 +5%", "dist100 +10%", "dist200 +5%",
                     "dist200 +10%"), "top_count_int": TOP_TUPLE},
    grid_justification_str=("Sensitivity map, not a search: 42 + 30 surface cells (ROC x filter at top 10; ROC x top count at "
                            "SMA100) and 10 filter variants at the live ROC 12 / top 10, 77 distinct configurations. Nothing "
                            "is promoted from it; any change goes through S4-S6 and a forward shadow."),
    primary_metric_str="In-sample net Sharpe surface and plateau; paired bootstrap P of each filter variant vs the live filter",
    kill_criteria_str="Not a promotion candidate (sensitivity map).", source_str="alpha/scout/specs/ndx_vxn.py (NATR20 VXN, A15)",
    parent_id_str="ndx_natr20_vxn_reaudition_20261002",
    universe_choice_str="The pod's own universe (Nasdaq-100 point-in-time), unchanged.",
)
_WORKER: dict = {}


def _init(live_inputs, tbill_ser) -> None:
    _WORKER["runner"] = PodRunner(PLAN, live_inputs, tbill_ser)


def _task(override_dict: dict) -> dict:
    try:
        daily_ser = _WORKER["runner"].daily({**BASE_DICT, **override_dict}).loc[START_STR:SEAL_END_STR]
        return {"override": override_dict, "daily": daily_ser}
    except Exception as error_obj:  # noqa: BLE001 - listed as failed
        return {"override": override_dict, "error": repr(error_obj)[:200]}


def _filter_override(sma_int: int) -> dict:
    return {"stock_trend_filter_bool": False} if sma_int == 0 else {"stock_sma_int": sma_int}


def main() -> None:
    ledger = Ledger()
    if REGISTRATION.registration_id_str not in registration_rows(ledger):
        register(ledger, REGISTRATION)
        print("registered", REGISTRATION.registration_id_str, flush=True)
    market_dict, tbill_ser = market_inputs()
    runner = PodRunner(PLAN, tbill_ser=tbill_ser)
    s1_list = [{"roc_month_int": r, **_filter_override(m)} for r in ROC_TUPLE for m in SMA_TUPLE]
    s2_list = [{"roc_month_int": r, "top_count_int": t} for r in ROC_TUPLE for t in TOP_TUPLE]
    variant_list = ([{"trend_fast_sma_int": f, "stock_sma_int": s} for f, s in CROSS_TUPLE]
                    + [{"stock_sma_int": s, "trend_threshold_float": d} for s, d in DISTANCE_TUPLE])
    task_list = [{}] + s1_list + s2_list + variant_list
    with Pool(8, initializer=_init, initargs=(runner.inputs({}), tbill_ser)) as pool_obj:
        result_list = pool_obj.map(_task, task_list, chunksize=2)
    failed_list = [r for r in result_list if "error" in r]
    daily_by_key = {json.dumps(r["override"], sort_keys=True): r["daily"] for r in result_list if "daily" in r}

    def daily(override_dict: dict) -> pd.Series:
        return daily_by_key[json.dumps(override_dict, sort_keys=True)]

    live_ser = daily({})

    def row(override_dict: dict) -> dict:
        metric = performance_dict(daily(override_dict))
        return {"sharpe": metric["sharpe_float"], "cagr": metric["cagr_float"], "max_dd": metric["max_drawdown_float"],
                "vol": metric["volatility_float"]}

    def surface(config_list: list, shape_tuple: tuple) -> dict:
        sharpe_vec = np.array([row(c)["sharpe"] for c in config_list])
        choice = plateau_choice(sharpe_vec, shape_tuple)
        return {"sharpe": sharpe_vec.round(3).tolist(), "neighbourhood_median": np.round(neighbourhood_median_vec(sharpe_vec, shape_tuple), 3).tolist(),
                "cagr": [round(row(c)["cagr"], 4) for c in config_list], "max_dd": [round(row(c)["max_dd"], 4) for c in config_list],
                "chosen": choice.flat_index_int, "chosen_config": config_list[choice.flat_index_int],
                "plateau_ratio": choice.plateau_ratio_float, "peak": choice.peak_sharpe_float, "chosen_median": choice.neighbourhood_median_float}

    variant_name_list = [f"SMA{f} > SMA{s}" for f, s in CROSS_TUPLE] + [f"Close / SMA{s} - 1 > {d:+.0%}" for s, d in DISTANCE_TUPLE]
    filter_row_list = []
    # The S1 cells at ROC 12 carry roc_month_int explicitly: reuse those runs under the same key.
    for name_str, override_dict in ([("Close > SMA100 (live)", {}), ("no filter", {"roc_month_int": 12, **_filter_override(0)})]
                                    + [(f"Close > SMA{m}", {"roc_month_int": 12, **_filter_override(m)}) for m in (20, 50, 150, 200, 250)]
                                    + list(zip(variant_name_list, variant_list))):
        filter_row_list.append({"name": name_str, **row(override_dict),
                                "p_better_than_live": paired_sharpe_probability(daily(override_dict), live_ser) if override_dict else float("nan")})
    out_dict = {
        "roc": ROC_TUPLE, "sma": SMA_TUPLE, "top": TOP_TUPLE,
        "s1": surface(s1_list, (len(ROC_TUPLE), len(SMA_TUPLE))), "s2": surface(s2_list, (len(ROC_TUPLE), len(TOP_TUPLE))),
        "filters": filter_row_list, "failed": failed_list, "live": row({}),
    }
    OUTPUT_PATH.write_text(json.dumps(out_dict, default=float), encoding="utf-8")
    print("live", {k: round(v, 3) for k, v in row({}).items()}, "failed", len(failed_list))
    for key_str in ("s1", "s2"):
        s = out_dict[key_str]
        print(key_str, "chosen", s["chosen_config"], "median", round(s["chosen_median"], 3), "peak", round(s["peak"], 3), "ratio", round(s["plateau_ratio"], 2))
        print("  sharpe grid", np.array(s["sharpe"]).reshape(len(ROC_TUPLE), -1).round(2).tolist())
    for r in filter_row_list:
        print(f"  {r['name']:28s} Sharpe {r['sharpe']:.2f} CAGR {r['cagr']:.1%} DD {r['max_dd']:.1%} P>live {r['p_better_than_live']:.2f}")


if __name__ == "__main__":
    main()
