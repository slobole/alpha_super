"""A 40% sector cap on the NDX design of record E2 (50/50 dollar ATR + NATR20; owner request 2026-10-04). Risk
management, not an alpha search: registered in the Scout ledger before any number is computed.

Within each of E2's two books, the top-10 walk takes at most 4 names per GICS group (40%), the next-ranked name filling
the slot (alpha/scout/specs/ndx_vxn.py sector_cap_int). Two groupings:
    L1  GICS sector (Information Technology is 37-42% of Nasdaq-100 members, so the cap is near the index's own weight)
    L2  GICS industry group (semiconductors, hardware and software capped separately: the 2026 memory/semis theme)
Labels are Norgate's current GICS (no history kept): a mild look-ahead, as the existing sector-cap strategy documents.
Evidence: full period 2000-09 to 2026-09 and the two halves, crashes, the 2026 months, the 16-offset luck band, the
paired bootstrap against uncapped E2, and the largest single-group weight actually held.
DECISION RULE (fixed before the run): a cap is recommended only if its full-period Max DD is at least 2 points smaller
than E2's AND its full-period Sharpe and luck-band median are no more than 0.05 below E2's. Otherwise E2 stays uncapped.

    uv run python scripts/research/scout_robustness_20261002/ndx_sector_cap.py
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

from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH, Ledger
from alpha.scout.metrics import performance_dict, sharpe_float
from alpha.scout.registration import Registration, register, registration_rows
from alpha.scout.stations.robustness import paired_sharpe_probability
from alpha.scout.stations.s4_strategy import SEAL_END_STR
import ndx_ensemble_cmma
from ndx_ranking_compare import CRISIS_DICT, window_return
from run import market_inputs

OUTPUT_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "robustness" / "ndx_sector_cap.json"
START_STR = "2000-09-01"
BOOK_LIST = [{}, {"atr_unit_str": "percent"}]
VARIANT_DICT = {
    "E2 (no cap)": {},
    "E2 + sector cap 40% (GICS L1)": {"sector_cap_int": 4, "sector_level_int": 1},
    "E2 + industry-group cap 40% (GICS L2)": {"sector_cap_int": 4, "sector_level_int": 2},
}
REGISTRATION = Registration(
    registration_id_str="ndx_e2_sector_cap_20261004", family_id_str="equity_cross_sectional_momentum",
    hypothesis_str=("Capping each GICS group at 40% of the names in each book of E2 (dollar ATR + NATR20) lowers the "
                    "theme-concentration drawdowns (e.g. the 2026 memory/semiconductor run and crash) at little cost in Sharpe."),
    mechanism_str="Risk management: momentum loads on the leading theme; a group cap spreads the book across groups.",
    expected_sign_and_location_str="Smaller Max DD and smaller theme crashes; Sharpe roughly unchanged or slightly lower.",
    hypothesis_class_str="X", universe_str="Nasdaq-100 point-in-time members", horizon_str="months",
    schedule_str="month-end decision", execution_str="next session's open",
    param_grid_dict={"sector_level_int": (1, 2), "sector_cap_int": (4,)},
    primary_metric_str="Full-period Max DD and Sharpe vs uncapped E2; luck-band median",
    kill_criteria_str=("Recommend a cap only if full-period Max DD improves by >= 2 points and Sharpe and luck-band median are "
                       "no more than 0.05 below uncapped E2; otherwise E2 stays uncapped."),
    source_str="strategies/momentum/strategy_mo_atr_normalized_sector_cap.py; alpha/scout/specs/ndx_vxn.py sector_cap_int",
    parent_id_str="ndx_natr20_vxn_ensemble_cmma_20261002",
    universe_choice_str="The pod's own universe (Nasdaq-100 point-in-time), unchanged.",
)


def _init(inputs, tbill_ser, sector_cache: dict) -> None:
    from alpha.scout.specs import ndx_vxn

    ndx_ensemble_cmma._init(inputs, tbill_ser)
    ndx_vxn._SECTOR_GROUP_CACHE.update(sector_cache)


def _task(args):
    return ndx_ensemble_cmma._task(args)


def main() -> None:
    from alpha.scout.specs import ndx_vxn

    ledger = Ledger()
    if REGISTRATION.registration_id_str not in registration_rows(ledger):
        register(ledger, REGISTRATION)
        print("registered", REGISTRATION.registration_id_str, flush=True)
    market_dict, tbill_ser = market_inputs()
    inputs = ndx_vxn.load_inputs()
    symbol_list = [s for s in inputs.close_df.columns if s != ndx_vxn.REGIME_SYMBOL_STR]
    for level_int in (1, 2):
        ndx_vxn.sector_group_dict(symbol_list, level_int)  # seed the cache once (Norgate) for the workers
    task_list = [(name_str, [{**b, **v} for b in BOOK_LIST], o) for name_str, v in VARIANT_DICT.items() for o in range(16)]
    with Pool(12, initializer=_init, initargs=(inputs, tbill_ser, dict(ndx_vxn._SECTOR_GROUP_CACHE))) as pool_obj:
        result_list = pool_obj.map(_task, task_list, chunksize=1)
    failed_list = [{k: v for k, v in r.items() if k != "daily"} for r in result_list if "error" in r]
    by_key = {(r["key"], r["offset"]): r for r in result_list if "error" not in r}
    base_ser = by_key[("E2 (no cap)", 0)]["daily"].loc[START_STR:]

    def group_share(variant_dict: dict, level_int: int) -> tuple[float, float]:
        frame_list = [ndx_vxn.rebalance_weight_df(inputs, dataclasses.replace(ndx_vxn.LIVE_CONFIG, **b, **variant_dict)) for b in BOOK_LIST]
        index = frame_list[0].index.union(frame_list[1].index)
        columns = sorted(set(frame_list[0].columns) | set(frame_list[1].columns))
        weight_df = (sum(f.reindex(index=index, columns=columns).fillna(0.0) for f in frame_list) / 2).loc[START_STR:]
        weight_df = weight_df[weight_df.sum(axis=1) > 0]
        group_ser = pd.Series(ndx_vxn.sector_group_dict(columns, level_int))
        share_df = weight_df.T.groupby(group_ser).sum().T.div(weight_df.sum(axis=1), axis=0)
        largest_ser = share_df.max(axis=1)
        return float(largest_ser.mean()), float(largest_ser.max())

    out = {}
    for name_str, variant_dict in VARIANT_DICT.items():
        daily_ser = by_key[(name_str, 0)]["daily"].loc[START_STR:]
        luck_vec = np.array([sharpe_float(by_key[(name_str, o)]["daily"].loc[START_STR:]) for o in range(16) if (name_str, o) in by_key])
        month_ser = (1 + daily_ser).resample("ME").prod() - 1
        out[name_str] = {
            "full": performance_dict(daily_ser, tbill_ser), "to_2022": performance_dict(daily_ser.loc[:SEAL_END_STR], tbill_ser),
            "seen_2023_on": performance_dict(daily_ser.loc["2023-01-01":], tbill_ser),
            "crisis": {k: window_return(daily_ser, a, b) for k, (a, b) in CRISIS_DICT.items()},
            "months_2026": {d.strftime("%Y-%m"): float(x) for d, x in month_ser.loc["2026-01":"2026-09"].items()},
            "luck": {"min": float(luck_vec.min()), "median": float(np.median(luck_vec)), "max": float(luck_vec.max())},
            "p_vs_e2_full": paired_sharpe_probability(daily_ser, base_ser) if variant_dict else float("nan"),
            "largest_sector_share_L1": group_share(variant_dict, 1), "largest_group_share_L2": group_share(variant_dict, 2),
            "names": by_key[(name_str, 0)]["names_held_float"], "turnover": by_key[(name_str, 0)]["turnover_float"],
        }
    base = out["E2 (no cap)"]
    for name_str, row in out.items():
        if name_str == "E2 (no cap)":
            continue
        row["recommended_bool"] = bool(row["full"]["max_drawdown_float"] >= base["full"]["max_drawdown_float"] + 0.02
                                       and row["full"]["sharpe_float"] >= base["full"]["sharpe_float"] - 0.05
                                       and row["luck"]["median"] >= base["luck"]["median"] - 0.05)
    OUTPUT_PATH.write_text(json.dumps({"variants": out, "failed": failed_list}, default=float), encoding="utf-8")

    print("failed", len(failed_list), [f.get("error") for f in failed_list[:3]])
    for name_str, row in out.items():
        f, a, b = row["full"], row["to_2022"], row["seen_2023_on"]
        print(f"{name_str:40s} full Sh {f['sharpe_float']:.2f} CAGR {f['cagr_float']:.1%} vol {f['volatility_float']:.1%} DD {f['max_drawdown_float']:.1%} "
              f"Calmar {f['calmar_float']:.2f} | to 2022 Sh {a['sharpe_float']:.2f} DD {a['max_drawdown_float']:.1%} | 2023+ Sh {b['sharpe_float']:.2f} "
              f"CAGR {b['cagr_float']:.1%} DD {b['max_drawdown_float']:.1%} | luck {row['luck']['min']:.2f}/{row['luck']['median']:.2f}/{row['luck']['max']:.2f} "
              f"| P>E2 {row['p_vs_e2_full']:.2f} | largest L1 {row['largest_sector_share_L1'][0]:.0%} (max {row['largest_sector_share_L1'][1]:.0%}) "
              f"L2 {row['largest_group_share_L2'][0]:.0%} (max {row['largest_group_share_L2'][1]:.0%}) | names {row['names']:.1f} | rec {row.get('recommended_bool', '-')}")
        print(f"{'':40s} crises {[round(x * 100, 1) for x in row['crisis'].values()]} | 2026 {[round(x * 100, 1) for x in row['months_2026'].values()]}")


if __name__ == "__main__":
    main()
