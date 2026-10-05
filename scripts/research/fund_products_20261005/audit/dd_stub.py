"""defensive_delta audit, extension check (NOT part of the frozen A6 window): the defensive legs run to 2026-10-02.

The A6 / A6-d menus end on 2026-08-19. This runs the five non-growth defensive legs at HEAD to 2026-10-02 (same call as
run_sleeves.py), checks that the new runs equal the stored paths up to 2026-08-19, appends the 31 new sessions to the
study's main frame (same fair-cash add, BIL as T-bills) and reports each growth-free slot on the extended window.
No bootstrap is re-drawn here. Nothing is written to the main checkout or the 2026-09-30 study folder.

Usage: python dd_stub.py [--workers 4]
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import inspect
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
import data.norgate_loader  # noqa: F401,E402

import numpy as np
import pandas as pd

WT = Path(__file__).resolve().parents[4]
MAIN = Path(r"C:\Users\User\Documents\workspace\alpha_super")
STORED = MAIN / "results" / "research" / "portfolio" / "shelf_rebuild_20260929" / "sources"
OUT = WT / "results" / "research" / "portfolio" / "fund_products_20261005" / "audit" / "defensive_delta" / "stub_20261002"
START, END2, CAPITAL = "2000-01-03", "2026-10-02", 1_000_000.0
LEGS = ["core5", "btal_qqq", "etf_dv2", "eom_flow", "downshock"]


def run_one(alias: str) -> str:
    from scripts.research import run_ladder4_candidate_value_add_study as ladder_runner

    meta = json.loads((STORED / f"{alias}__metadata.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{alias}.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        module = importlib.import_module(meta["strategy_import_str"].split(":", maxsplit=1)[0])
        fn = getattr(module, "run_variant")
        kw = dict(show_display_bool=False, save_results_bool=False, output_dir_str=str(OUT / "scratch_unused"),
                  capital_base_float=CAPITAL, end_date_str=END2, **meta["extra_kwarg_dict"])
        try:
            strat = fn(backtest_start_date_str=START, **kw)
        except Exception:  # noqa: BLE001
            strat = fn(backtest_start_date_str=inspect.signature(fn).parameters["backtest_start_date_str"].default, **kw)
    df = ladder_runner.extract_source_result_df(strat)
    ladder_runner.write_csv_gzip(df, OUT / f"{alias}__path.csv.gz", index_bool=True, index_label_str="date")
    return alias


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-runs", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    if not args.skip_runs:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for a in ex.map(run_one, LEGS):
                print("ran", a, round(time.time() - t0), "s", flush=True)

    sys.path.insert(0, str(WT / "scripts" / "research" / "fund_products_20260930"))
    import fp_lib as fp
    import a6
    from a6 import Lab
    lib = a6.lib
    lab = Lab()
    old_frame = lab.frame                                   # fair-cash main frame, ends 2026-08-19
    end_old = fp.END
    res: dict = {"end_old": str(end_old.date()), "end_new": END2, "legs": {}, "slots": {}}
    new_paths = {a: pd.read_csv(OUT / f"{a}__path.csv.gz", index_col="date", parse_dates=True) for a in LEGS}
    idx_new = new_paths["core5"].index
    stub_idx = idx_new[idx_new > end_old]
    full_idx = old_frame.index.union(stub_idx)
    rate = lib.dtb3_annual_rate(full_idx)
    res["dtb3_last_observation_used"] = "ffill of the last DTB3 row in 1_data/DTB3.csv (file ends 2026-09-28)"
    ext = pd.DataFrame(index=stub_idx, columns=LEGS + [fp.TBILL], dtype=float)
    for a in LEGS:
        p_new = new_paths[a]
        p_old = pd.read_csv(STORED / f"{a}__path.csv.gz", index_col="date", parse_dates=True)
        common = p_old.index
        d = (p_new.loc[common, "total_value_float"] / p_old["total_value_float"] - 1.0).abs().max()
        raw = p_new["total_value_float"].pct_change(fill_method=None)
        add = lib.cash_realism_add(p_new, rate.reindex(p_new.index))
        r = (raw + add).reindex(stub_idx)
        ext[a] = r
        stub_raw = float(np.prod(1 + raw.reindex(stub_idx).to_numpy()) - 1)
        res["legs"][a] = {"max_rel_nav_diff_vs_stored_to_old_end": float(d), "stub_sessions": int(len(stub_idx)),
                          "stub_return_engine_0pct_cash": stub_raw, "stub_return_fair_cash": float(np.prod(1 + r.to_numpy()) - 1)}
    bil = lib.common.load_total_return_close_ser("BIL", "2007-01-01", END2)
    ext[fp.TBILL] = bil.reindex(full_idx).pct_change(fill_method=None).reindex(stub_idx)
    res["legs"][fp.TBILL] = {"stub_return": float(np.prod(1 + ext[fp.TBILL].to_numpy()) - 1)}
    frame2 = pd.concat([old_frame[LEGS + [fp.TBILL]], ext])
    assert not frame2.loc[lab.start:].isna().any().any()

    stored = json.loads((fp.STUDY / "report" / "a6d.json").read_text(encoding="utf-8"))["defensive"]
    for slot in ("launch", "calm", "ds_upgrade_05", "ds_upgrade_10", "next", "calm_next", "target"):
        w = stored[slot]["weights"]
        bk = a6.Book(slot, tuple(w), "EQ", w)
        r_old = lib.book_returns(old_frame, bk, lab.start, end_old)
        r_new = lib.book_returns(frame2, bk, lab.start, pd.Timestamp(END2))
        assert np.allclose(r_old.to_numpy(), r_new.loc[:end_old].to_numpy(), atol=1e-14)

        def stats(r: pd.Series) -> dict:
            nav = np.r_[1.0, np.cumprod(1 + r.to_numpy())]
            rf = frame2[fp.TBILL].reindex(r.index).to_numpy()
            x = r.to_numpy() - rf
            return {"cagr": float(nav[-1] ** (252 / len(r)) - 1), "xs": float(x.mean() / x.std(ddof=1) * np.sqrt(252)),
                    "dd": float((nav / np.maximum.accumulate(nav) - 1).min())}

        stub = r_new.loc[r_new.index > end_old]
        nav = np.r_[1.0, np.cumprod(1 + r_new.to_numpy())]
        cur_dd = float(nav[-1] / nav.max() - 1)
        snav = np.r_[1.0, np.cumprod(1 + stub.to_numpy())]
        res["slots"][slot] = {"name": stored[slot]["name"], "to_old_end": stats(r_old), "to_new_end": stats(r_new),
                              "stub_return": float(snav[-1] - 1), "stub_max_dd": float((snav / np.maximum.accumulate(snav) - 1).min()),
                              "drawdown_at_new_end": cur_dd}
        print(slot, stored[slot]["name"], json.dumps(res["slots"][slot]), flush=True)
    for a, v in res["legs"].items():
        print("LEG", a, json.dumps(v), flush=True)
    (OUT / "stub_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
