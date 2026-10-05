"""defensive_delta audit, step 1b: fresh engine run of the defensive legs at HEAD on today's Norgate vintage.

Each sleeve is run exactly as shelf_rebuild_20260929/run_sleeves.py ran it ($1M, requested start 2000-01-03 with the
module default as fallback, end 2026-08-19) and its daily NAV is compared with the stored 2026-09-29 path that the
A6 / A6-d menus were built from. Code comes from this worktree (HEAD); nothing is written to the main checkout.

Usage: python dd_engine_rerun.py alias [alias ...] [--workers N]
Outputs: <audit>/defensive_delta/engine_rerun/<alias>__path.csv.gz, <alias>.json, <alias>.log
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import inspect
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

WT = Path(__file__).resolve().parents[4]
MAIN = Path(r"C:\Users\User\Documents\workspace\alpha_super")
STORED = MAIN / "results" / "research" / "portfolio" / "shelf_rebuild_20260929" / "sources"
OUT = WT / "results" / "research" / "portfolio" / "fund_products_20261005" / "audit" / "defensive_delta" / "engine_rerun"
START, END, CAPITAL = "2000-01-03", "2026-08-19", 1_000_000.0


def run_one(alias: str) -> dict:
    from scripts.research import run_ladder4_candidate_value_add_study as ladder_runner

    meta = json.loads((STORED / f"{alias}__metadata.json").read_text(encoding="utf-8"))
    import_str, extra = meta["strategy_import_str"], meta["extra_kwarg_dict"]
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with (OUT / f"{alias}.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        module = importlib.import_module(import_str.split(":", maxsplit=1)[0])
        assert str(Path(module.__file__).resolve()).startswith(str(WT)), module.__file__
        fn = getattr(module, "run_variant")
        kw = dict(show_display_bool=False, save_results_bool=False, output_dir_str=str(OUT / "scratch_unused"),
                  capital_base_float=CAPITAL, end_date_str=END, **extra)
        try:
            strat, start_used = fn(backtest_start_date_str=START, **kw), START
        except Exception as exc:  # noqa: BLE001  (same fallback as run_sleeves.py)
            default = inspect.signature(fn).parameters["backtest_start_date_str"].default
            print(f"requested start failed ({exc!r}); retrying with module default {default!r}")
            traceback.print_exc()
            strat, start_used = fn(backtest_start_date_str=default, **kw), str(default)
    new = ladder_runner.extract_source_result_df(strat)
    ladder_runner.write_csv_gzip(new, OUT / f"{alias}__path.csv.gz", index_bool=True, index_label_str="date")
    old = pd.read_csv(STORED / f"{alias}__path.csv.gz", index_col="date", parse_dates=True)
    same_index = bool(new.index.equals(old.index))
    idx = new.index.intersection(old.index)
    a, b = old.loc[idx, "total_value_float"].astype(float), new.loc[idx, "total_value_float"].astype(float)
    ra, rb = a.pct_change().fillna(0.0), b.pct_change().fillna(0.0)
    dret = (rb - ra).abs()
    rel = (b / a - 1.0).abs()
    first_diff = dret[dret > 1e-12].index[0].date().isoformat() if (dret > 1e-12).any() else None
    yrs = (idx[-1] - idx[0]).days / 365.25
    res = {"alias": alias, "module": meta["module_path_str"], "start_used": start_used, "stored_start_used": meta["requested_start_date_str"],
           "same_index": same_index, "n_old": int(len(old)), "n_new": int(len(new)),
           "max_abs_daily_return_diff": float(dret.max()), "days_return_diff_gt_1e-9": int((dret > 1e-9).sum()),
           "first_diff_date": first_diff, "max_rel_nav_diff": float(rel.max()),
           "final_nav_old": float(a.iloc[-1]), "final_nav_new": float(b.iloc[-1]),
           "cagr_old": float((a.iloc[-1] / a.iloc[0]) ** (1 / yrs) - 1), "cagr_new": float((b.iloc[-1] / b.iloc[0]) ** (1 / yrs) - 1),
           "max_abs_cash_diff": float((new.loc[idx, "cash_float"] - old.loc[idx, "cash_float"]).abs().max()),
           "runtime_s": round(time.time() - t0, 1)}
    # Same comparison restricted to the study's LONG window (2008-03-04 on).
    w = idx[idx >= pd.Timestamp("2008-03-04")]
    res["long_window_max_abs_daily_return_diff"] = float(dret.loc[w].max()) if len(w) else None
    res["long_window_cum_return_old"] = float(a.loc[w].iloc[-1] / a.loc[w].iloc[0] - 1) if len(w) else None
    res["long_window_cum_return_new"] = float(b.loc[w].iloc[-1] / b.loc[w].iloc[0] - 1) if len(w) else None
    (OUT / f"{alias}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("aliases", nargs="+")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    from scripts.research import run_ladder4_candidate_value_add_study as ladder_runner
    print("norgate vintage", json.dumps(ladder_runner.norgate_database_vintage_dict()), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, a): a for a in args.aliases}
        for f in as_completed(futs):
            a = futs[f]
            try:
                print("DONE", json.dumps(f.result()), flush=True)
            except Exception as exc:  # noqa: BLE001
                print("FAILED", a, repr(exc), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
