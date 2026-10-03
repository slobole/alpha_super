"""Multiplicity over EVERY stock trend filter tried on NATR20 VXN (ROC 12, top 10) in A15's follow-ups, not only the
nine in the bake-off. The bake-off's two passing filters (Close / SMA200 > +10%, CMMA(200) > +10) were put on its list
because they had topped the earlier maps on the same history, so the bake-off's Romano-Wolf family (nine) undercounts
the search. This re-runs no new idea: it re-simulates the already-registered filters (registrations
ndx_natr20_vxn_plateau_filter_20261002, ndx_natr20_vxn_ensemble_cmma_20261002, ndx_natr20_vxn_filter_bakeoff_20261003)
and applies one Romano-Wolf step-down to all of them against Close > SMA100.

    uv run python scripts/research/scout_robustness_20261002/ndx_filter_family_rw.py
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
from alpha.scout.stations.robustness import paired_sharpe_difference_draws, romano_wolf_stepdown
from alpha.scout.stations.s4_strategy import SEAL_END_STR
from ndx_ensemble_cmma import _init, _task
from plans import SEED_INT
from run import market_inputs

OUTPUT_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "robustness" / "ndx_filter_family_rw.json"
NATR_DICT = {"atr_unit_str": "percent"}
FAMILY_DICT = {"F0 Close > SMA100 (live)": {}, "No filter": {"stock_trend_filter_bool": False}}
FAMILY_DICT.update({f"Close > SMA{m}": {"stock_sma_int": m} for m in (20, 50, 150, 200, 250)})
FAMILY_DICT.update({f"SMA{f} > SMA{s}": {"trend_fast_sma_int": f, "stock_sma_int": s} for f, s in ((10, 50), (21, 50), (21, 100), (50, 100), (50, 200))})
FAMILY_DICT.update({f"Close/SMA{s} > {d:+.0%}": {"stock_sma_int": s, "trend_threshold_float": d}
                    for s, d in ((100, -0.05), (100, 0.05), (100, 0.10), (200, 0.05), (200, 0.10))})
FAMILY_DICT.update({f"CMMA({lb},252) > {th:+.0f}": {"trend_filter_str": "cmma", "stock_sma_int": lb, "cmma_atr_int": 252, "cmma_threshold_float": th}
                    for lb in (50, 100, 200) for th in (-20.0, -10.0, 0.0, 10.0, 20.0)})
FAMILY_DICT.update({f"CMMA(100,63) > {th:+.0f}": {"trend_filter_str": "cmma", "stock_sma_int": 100, "cmma_atr_int": 63, "cmma_threshold_float": th}
                    for th in (-10.0, 0.0, 10.0)})
FAMILY_DICT.update({"Close > LowPass(100)": {"trend_filter_str": "lowpass", "stock_sma_int": 100},
                    "Close > LowPass(200)": {"trend_filter_str": "lowpass", "stock_sma_int": 200},
                    "LowPass(200) rising": {"trend_filter_str": "lowpass_rising", "stock_sma_int": 200},
                    "CORE5 adaptive AMA": {"trend_filter_str": "adaptive_ama"}})


def main() -> None:
    from alpha.scout.specs import ndx_vxn

    _, tbill_ser = market_inputs()
    inputs = ndx_vxn.load_inputs()
    task_list = [(name_str, [{**NATR_DICT, **override_dict}], 0) for name_str, override_dict in FAMILY_DICT.items()]
    with Pool(12, initializer=_init, initargs=(inputs, tbill_ser)) as pool_obj:
        result_list = pool_obj.map(_task, task_list, chunksize=1)
    failed_list = [{k: v for k, v in r.items() if k != "daily"} for r in result_list if "error" in r]
    daily_dict = {r["key"]: r["daily"] for r in result_list if "error" not in r}
    base_str = "F0 Close > SMA100 (live)"
    alt_list = [n for n in FAMILY_DICT if n != base_str and n in daily_dict]
    in_base_vec = daily_dict[base_str].loc[:SEAL_END_STR].to_numpy()
    in_mat = np.column_stack([daily_dict[n].loc[:SEAL_END_STR].to_numpy() for n in alt_list])
    observed_vec, draw_mat = paired_sharpe_difference_draws(in_base_vec, in_mat, draw_count_int=2000, random_seed_int=SEED_INT)
    rw_vec = romano_wolf_stepdown(observed_vec, draw_mat)
    sd_vec = draw_mat.std(axis=0, ddof=1)
    row_list = sorted(({"filter": n, "sharpe": sharpe_float(daily_dict[n].loc[:SEAL_END_STR]), "difference": float(observed_vec[j]),
                        "t": float(observed_vec[j] / sd_vec[j]) if sd_vec[j] > 0 else 0.0, "rw_p": float(rw_vec[j]),
                        "raw_p": float(np.mean(draw_mat[:, j] <= 0)), "seen_sharpe": sharpe_float(daily_dict[n].loc["2023-01-01":])}
                       for j, n in enumerate(alt_list)), key=lambda r: -r["t"])
    out_dict = {"family_size": len(alt_list), "base_sharpe": sharpe_float(daily_dict[base_str].loc[:SEAL_END_STR]),
                "base_seen_sharpe": sharpe_float(daily_dict[base_str].loc["2023-01-01":]), "rows": row_list, "failed": failed_list}
    OUTPUT_PATH.write_text(json.dumps(out_dict, default=float), encoding="utf-8")
    print("failed", len(failed_list), "| family of", len(alt_list), "alternatives vs", base_str, f"(Sharpe {out_dict['base_sharpe']:.2f}, 2023+ {out_dict['base_seen_sharpe']:.2f})")
    for r in row_list:
        print(f"{r['filter']:24s} Sh {r['sharpe']:.2f} diff {r['difference']:+.3f} t {r['t']:5.2f} raw p {r['raw_p']:.3f} RW p {r['rw_p']:.3f} 2023+ {r['seen_sharpe']:.2f}")


if __name__ == "__main__":
    main()
