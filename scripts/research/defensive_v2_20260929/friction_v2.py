"""A2 check 3: small-account friction of the defensive v2 picks and the champion (descriptive, after selection).

Same method as shelf_rebuild friction_runs.py: each pod re-run at pod capital = weight x account ($30K, $100K) on
the EXACT (2012-10-01) and RECENT (2023-08-21) windows, against a $1M run with the same start. Friction =
CAGR($1M) - CAGR(small). Weights are fixed here (IV picks use their LONG average weights).

Usage: python friction_v2.py [--workers 6]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "shelf_rebuild_20260929"))

import friction_runs as fr  # noqa: E402

OUT = HERE.parents[2] / "results" / "research" / "portfolio" / "defensive_v2_20260929" / "friction"
PRODUCTS = {
    "CORE5 + EOM + DISP [IV]": {"core5": 0.4285, "eom_flow": 0.2277, "disp": 0.3438},
    "BTAL_QQQ + DV2-IND [EQ]": {"btal_qqq": 0.5, "etf_dv2": 0.5},
    "CORE5 60 + BTAL_QQQ 40": {"core5": 0.6, "btal_qqq": 0.4},
    "CORE5 + EOM + DV2-IND [EQ]": {"core5": 1 / 3, "eom_flow": 1 / 3, "etf_dv2": 1 / 3},
    "CORE5 + BTAL_QQQ + EOM [IV]": {"core5": 0.4703, "btal_qqq": 0.2797, "eom_flow": 0.2500},  # A3 runner-up
    "CORE5 + BTAL_QQQ + DV2-IND [IV]": {"core5": 0.3839, "btal_qqq": 0.2276, "etf_dv2": 0.3885},  # owner candidate
    "CORE5 + BTAL_QQQ + EOM + DV2-IND [EQ]": {"core5": 0.25, "btal_qqq": 0.25, "eom_flow": 0.25, "etf_dv2": 0.25},
}


def main(arg_list: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(arg_list)
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = set()
    for pods in PRODUCTS.values():
        for alias, w in pods.items():
            for window in fr.WINDOW_DICT:
                jobs.add((alias, window, fr.REFERENCE_FLOAT))
                for account in fr.ACCOUNT_TUPLE:
                    jobs.add((alias, window, round(w * account, 2)))
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as pool:
        future_map = {pool.submit(fr.run_one, *job): job for job in sorted(jobs)}
        for future in as_completed(future_map):
            job = future_map[future]
            try:
                rows.append(future.result())
                print(json.dumps(rows[-1]), flush=True)
            except Exception as exc:  # noqa: BLE001 - a pod too small to trade is itself a finding
                rows.append({"alias": job[0], "window": job[1], "capital": job[2], "error": repr(exc)})
                print(f"FAILED {job}: {exc!r}", flush=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "friction_runs.csv", index=False, float_format="%.6g")
    ok = runs[runs.get("cagr").notna()] if "cagr" in runs else runs.iloc[0:0]
    reference = ok[ok["capital"] == fr.REFERENCE_FLOAT].set_index(["alias", "window"])["cagr"]
    summary = []
    for name, pods in PRODUCTS.items():
        for account in fr.ACCOUNT_TUPLE:
            for window in fr.WINDOW_DICT:
                row = {"product": name, "account": account, "window": window}
                weighted = 0.0
                for alias, w in pods.items():
                    hit = ok[(ok["alias"] == alias) & (ok["window"] == window) & (ok["capital"] == round(w * account, 2))]
                    small = float(hit["cagr"].iloc[0]) if len(hit) else float("nan")
                    friction = float(reference.get((alias, window), float("nan")) - small)
                    row[f"friction_{alias}"] = friction
                    row[f"pod_capital_{alias}"] = round(w * account, 2)
                    weighted += w * friction
                row["weighted_friction"] = weighted
                summary.append(row)
    pd.DataFrame(summary).to_csv(OUT / "friction_by_product.csv", index=False, float_format="%.6g")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
