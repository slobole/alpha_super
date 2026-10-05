"""Timing lens 12: end-date causality of the four new sleeves. Re-run each with end 2019-12-31 ($1M, module default start, worktree code)
and compare NAV / cash / invested value and the fills with the stored run (end 2026-10-02) on the common dates. Writes only under audit/review/timing_lens."""
import contextlib, importlib, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
WT = HERE.parents[4]
if str(WT) not in sys.path:
    sys.path.insert(0, str(WT))
OUT = WT / "results/research/portfolio/fund_products_20261005/audit/review/timing_lens/truncate"
SRC = WT / "results/research/portfolio/fund_products_20261005/sources"
CUT = "2019-12-31"
MODS = {"ndx_atr_cap": "strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled_sector_cap", "ndx_natr_cap": "strategies.momentum.strategy_mo_natr20_ndx_vxn_scaled_sector_cap",
        "dv2_g": "strategies.mr_capsule.strategy_mr_dv2_vix_gated_bil", "hpi_g": "strategies.mr_capsule.strategy_mr_hpi_vote_vix_gated_bil"}

def run(alias):
    from scripts.research import run_ladder4_candidate_value_add_study as lr
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    mod = importlib.import_module(MODS[alias])
    assert Path(mod.__file__).resolve().is_relative_to(WT), mod.__file__
    with (OUT / f"{alias}.log").open("w", encoding="utf-8") as lf, contextlib.redirect_stdout(lf), contextlib.redirect_stderr(lf):
        s = mod.run_variant(show_display_bool=False, save_results_bool=False, output_dir_str=str(OUT / "scratch_unused"), capital_base_float=1_000_000.0, end_date_str=CUT)
    res = lr.extract_source_result_df(s); tx = lr.extract_source_transaction_df(s, alias)
    res.to_csv(OUT / f"{alias}__path.csv.gz", index_label="date"); tx.to_csv(OUT / f"{alias}__transactions.csv.gz", index=False)
    old = pd.read_csv(SRC / f"{alias}__path.csv.gz", index_col="date", parse_dates=True)
    otx = pd.read_csv(SRC / f"{alias}__transactions.csv.gz", parse_dates=["date"])
    common = res.index.intersection(old.index)
    d = (res.loc[common] / old.loc[common] - 1).abs()
    rr, ro = res["total_value_float"].pct_change().loc[common], old["total_value_float"].pct_change().loc[common]
    first_bad = d[d["total_value_float"] > 1e-9].index.min()
    otx_c = otx[otx.date <= pd.Timestamp(CUT)]
    return (f"{alias}: truncated run ends {res.index[-1].date()} ({len(res)} rows; stored has {int((old.index <= pd.Timestamp(CUT)).sum())} rows to the cut); "
            f"max |NAV ratio - 1| {float(d['total_value_float'].max()):.3e}; max |daily return diff| {float((rr - ro).abs().max()):.3e}; first date NAV differs by > 1e-9: {first_bad}; "
            f"fills {len(tx)} vs stored-to-cut {len(otx_c)}; runtime {time.time() - t0:.0f}s")

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4, max_tasks_per_child=1) as ex:
        futs = {ex.submit(run, a): a for a in MODS}
        for f in as_completed(futs):
            try:
                print(f.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                print("FAILED", futs[f], repr(e), flush=True)
