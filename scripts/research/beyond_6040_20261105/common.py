"""Shared helpers for the Beyond 60/40 experiments (research-only)."""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from data import load
from engine import Params, run, fixed_mix, metrics, block_bootstrap_diff, DECADES, DEV_END, VAULT_START, TRADING_DAYS
OUT = HERE / "out"
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 500)

KEEP = ["cagr", "excess_cagr", "vol", "sharpe", "sharpe_raw", "max_dd", "calmar", "worst_12m", "worst_36m",
        "longest_underwater_years", "cvar5_monthly", "skew_monthly", "avg_exposure", "pct_fully_invested", "turnover_per_year"]

def dev(panel):
    """Development sample only: everything before the vault."""
    return panel.loc[:DEV_END]

def summarize(res, start="1970-01-01", end=DEV_END, blocks=True):
    m = metrics(res["ret"].loc[start:end], res["cash_daily"].loc[start:end], res["exposure"].loc[start:end], res["turnover"].loc[start:end])
    row = {k: m.get(k, np.nan) for k in KEEP}
    if blocks:
        for name, (s, e) in DECADES.items():
            if name.startswith("Full"):
                continue
            mb = metrics(res["ret"].loc[s:e], res["cash_daily"].loc[s:e])
            row[f"sh_{name}"] = mb.get("sharpe", np.nan)
            row[f"dd_{name}"] = mb.get("max_dd", np.nan)
    return row

def benchmarks(panel, cash="ff", cost=5.0):
    return {"60/40": fixed_mix(panel, {"r_stock": .6, "r_bond": .4}, cash=cash, cost_bps=cost),
            "EW3": fixed_mix(panel, {"r_stock": 1/3, "r_bond": 1/3, "r_gold": 1/3}, cash=cash, cost_bps=cost),
            "Stocks": fixed_mix(panel, {"r_stock": 1.0}, cash=cash, cost_bps=cost)}

def adoption_check(res_new, res_base, panel, start="1970-01-01", end=DEV_END):
    """Pre-registered adoption rule (SPEC.md): (a) worst block Sharpe up, (b) full Sharpe not down > 0.05,
    (c) bootstrap diff p05 > -0.05, (d) turnover <= 2x. Returns dict with flags."""
    a = summarize(res_new, start, end); b = summarize(res_base, start, end)
    blocks = [k for k in a if k.startswith("sh_") and not k.startswith("sh_ETF")]
    worst_a = min(a[k] for k in blocks); worst_b = min(b[k] for k in blocks)
    bs = block_bootstrap_diff(res_new["ret"].loc[start:end], res_base["ret"].loc[start:end], res_base["cash_daily"].loc[start:end], n_draws=1000)
    out = {"worst_block_sharpe_new": worst_a, "worst_block_sharpe_base": worst_b, "a_worst_block_up": worst_a > worst_b,
           "full_sharpe_new": a["sharpe"], "full_sharpe_base": b["sharpe"], "b_full_not_down": a["sharpe"] >= b["sharpe"] - 0.05,
           "boot_diff_median": bs["diff_median"], "boot_diff_p05": bs["diff_p05"], "boot_diff_p95": bs["diff_p95"],
           "c_boot_ok": bs["diff_p05"] > -0.05, "turnover_new": a["turnover_per_year"], "turnover_base": b["turnover_per_year"],
           "d_turnover_ok": a["turnover_per_year"] <= 2 * b["turnover_per_year"] + 1e-9,
           "max_dd_new": a["max_dd"], "max_dd_base": b["max_dd"], "dd_cut_25pct": a["max_dd"] >= b["max_dd"] * 0.75,
           "cagr_premium": b["cagr"] - a["cagr"]}
    out["adopt"] = out["a_worst_block_up"] and out["b_full_not_down"] and out["c_boot_ok"] and out["d_turnover_ok"]
    out["adopt_as_insurance"] = (not out["adopt"]) and out["dd_cut_25pct"] and out["cagr_premium"] <= 0.005 and out["c_boot_ok"] and out["d_turnover_ok"]
    return out

def save(df, name, sub):
    d = OUT / sub; d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / f"{name}.csv")
    return d / f"{name}.csv"

def save_json(obj, name, sub):
    d = OUT / sub; d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(obj, indent=2, default=float))
