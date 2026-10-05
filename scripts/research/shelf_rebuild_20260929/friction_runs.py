"""SPEC 10 + A1: small-account friction of the rung products (descriptive, after selection).

Every pod of every rung product is re-run at pod capital = weight x account for accounts of $30,000 and $100,000,
on two windows (EXACT start 2012-10-01; RECENT start 2023-08-21, today's share prices), each against a $1M run
with the same start. Friction = CAGR($1M) - CAGR(small), from the starting capital to 2026-08-19; whole-share
sizing and the $1 minimum commission are what the engine charges. T-bills (BIL, one position) are not re-run.

Usage: python friction_runs.py [--workers 6]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import importlib
import io
import json

import pandas as pd

import lib
from lib import TBILL
from run_sleeves import SLEEVE_DICT

OUT = lib.STUDY / "friction"
ACCOUNT_TUPLE = (30_000.0, 100_000.0)
REFERENCE_FLOAT = 1_000_000.0
WINDOW_DICT = {"EXACT": "2012-10-01", "RECENT": "2023-08-21"}


def run_one(alias: str, window: str, capital: float) -> dict:
    import_str, extra = SLEEVE_DICT[alias]
    module = importlib.import_module(import_str.split(":")[0])
    start = WINDOW_DICT[window]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        strategy = module.run_variant(show_display_bool=False, save_results_bool=False,
                                      output_dir_str=str(OUT / "scratch_unused"), backtest_start_date_str=start,
                                      capital_base_float=capital, end_date_str=lib.END.strftime("%Y-%m-%d"), **extra)
    nav = strategy.results["total_value"].astype(float)
    nav.index = pd.to_datetime(nav.index).normalize()
    days = (nav.index[-1] - pd.Timestamp(start)).days
    cagr = (nav.iloc[-1] / capital) ** (365.25 / days) - 1.0
    tx = strategy.get_transactions()
    return {"alias": alias, "window": window, "capital": capital, "cagr": float(cagr),
            "commission_share_of_start": float(tx["commission"].sum() / capital) if len(tx) else 0.0,
            "end_nav": float(nav.iloc[-1]), "first_date": str(nav.index[0].date())}


def main(arg_list: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(arg_list)
    OUT.mkdir(parents=True, exist_ok=True)
    products = pd.read_csv(lib.STUDY / "part_m" / "products.csv", index_col="product")
    weights = {name: json.loads(products.at[name, "pod_weights"]) for name in products.index}
    jobs = set()
    for pods in weights.values():
        for alias, w in pods.items():
            if alias == TBILL:
                continue
            for window in WINDOW_DICT:
                jobs.add((alias, window, REFERENCE_FLOAT))
                for account in ACCOUNT_TUPLE:
                    jobs.add((alias, window, round(w * account, 2)))
    lib.ledger("friction_runs_started", jobs=len(jobs))
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as pool:
        future_map = {pool.submit(run_one, *job): job for job in sorted(jobs)}
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

    reference = runs[runs["capital"] == REFERENCE_FLOAT].set_index(["alias", "window"])["cagr"]
    summary = []
    for name, pods in weights.items():
        for account in ACCOUNT_TUPLE:
            for window in WINDOW_DICT:
                row = {"product": name, "account": account, "window": window}
                weighted = 0.0
                for alias, w in pods.items():
                    if alias == TBILL:
                        continue
                    hit = runs[(runs["alias"] == alias) & (runs["window"] == window)
                               & (runs["capital"] == round(w * account, 2))]
                    small = float(hit["cagr"].iloc[0]) if len(hit) and "cagr" in hit and hit["cagr"].notna().any() else float("nan")
                    friction = float(reference.loc[(alias, window)] - small)
                    row[f"friction_{alias}"] = friction
                    row[f"pod_capital_{alias}"] = round(w * account, 2)
                    weighted += w * friction
                row["weighted_friction"] = weighted
                summary.append(row)
    pd.DataFrame(summary).to_csv(OUT / "friction_by_product.csv", index=False, float_format="%.6g")
    lib.ledger("friction_runs_finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
