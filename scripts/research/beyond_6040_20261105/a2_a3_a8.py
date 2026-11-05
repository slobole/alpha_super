"""Arms A2 (weighting), A3 (portfolio-vol estimator), A8 (leverage) of SPEC.md. Research-only.

Development sample only (dev(load()) ends 2022-12-31). Windows: full 1970-2022, ETF era 2005-2022.
Outputs: out/a2_a3_a8/*.csv + findings.md (written by hand after reading the tables).
"""
import time
from common import *

SUB = "a2_a3_a8"
WINDOWS = {"full": ("1970-01-01", DEV_END), "etf": ("2005-01-01", DEV_END)}
SHOW = ["cagr", "vol", "sharpe", "sharpe_raw", "max_dd", "worst_12m", "avg_exposure", "pct_fully_invested", "turnover_per_year"]

panel = dev(load())
assert panel.index[-1] <= pd.Timestamp(DEV_END)
dff = panel["dff"]

t0 = time.time()
B0 = run(panel, Params())
print(f"B0 run {time.time()-t0:.1f}s  sharpe full {summarize(B0)['sharpe']:.4f}")


def rows_for(results: dict, extra=None):
    """Metric rows (both windows) + adoption rows vs B0 (both windows)."""
    met, adopt = [], []
    for name, res in results.items():
        for win, (s, e) in WINDOWS.items():
            row = {"variant": name, "window": win, **summarize(res, s, e)}
            if extra is not None:
                row.update(extra(res, s, e))
            met.append(row)
            if res is not B0:
                ad = {"variant": name, "window": win, **adoption_check(res, B0, panel, s, e)}
                adopt.append(ad)
    met = pd.DataFrame(met).set_index(["variant", "window"])
    adopt = pd.DataFrame(adopt).set_index(["variant", "window"]) if adopt else pd.DataFrame()
    return met, adopt


def concentration(res, s, e):
    w = res["weights"].loc[s:e].dropna()
    w = w[w.sum(axis=1) > 0]
    wn = w.div(w.sum(axis=1), axis=0)   # weights as a fraction of the risky sleeve
    out = {"share_days_max_w_gt_0.6": float((wn.max(axis=1) > 0.6).mean()),
           "share_days_max_w_gt_0.5": float((wn.max(axis=1) > 0.5).mean()),
           "share_days_min_w_lt_0.05": float((wn.min(axis=1) < 0.05).mean())}
    for c in wn.columns:
        out[f"avg_w_{c.replace('r_', '')}"] = float(wn[c].mean())
    return out


def leverage_extra(res, s, e):
    ex = res["exposure"].loc[s:e].dropna()
    r = dff.reindex(ex.index).ffill()
    fin = (np.maximum(ex - 1.0, 0.0) * (r + 0.015)).mean()  # annual financing cost as a fraction of NAV
    d = ex.diff().abs()
    return {"share_days_exposure_gt_1.5": float((ex > 1.5).mean()),
            "share_days_exposure_gt_1.0": float((ex > 1.0 + 1e-9).mean()),
            "max_exposure_realised": float(ex.max()),
            "financing_cost_per_year": float(fin),
            "n_exposure_jumps_gt_0.3": int((d > 0.3).sum()),
            "max_exposure_jump": float(d.max())}


# ------------------------------------------------------------------ A2 weighting
print("\n=== A2 weighting ===")
a2 = {}
for wgt in ["ivol", "equal", "erc", "minvar", "ivol_ew"]:
    t0 = time.time()
    a2[wgt] = B0 if wgt == "ivol" else run(panel, Params(weighting=wgt))
    print(f"  {wgt:8s} {time.time()-t0:5.1f}s")
a2_met, a2_adopt = rows_for(a2, concentration)
save(a2_met, "a2_weighting", SUB); save(a2_adopt, "a2_adoption", SUB)
cols2 = SHOW + ["share_days_max_w_gt_0.6", "avg_w_stock", "avg_w_bond", "avg_w_gold"]
print(a2_met[cols2].round(3).to_string())
print(a2_met[[c for c in a2_met.columns if c.startswith("sh_")]].round(2).to_string())
print(a2_adopt[["worst_block_sharpe_new", "worst_block_sharpe_base", "full_sharpe_new", "full_sharpe_base",
                "boot_diff_median", "boot_diff_p05", "boot_diff_p95", "turnover_new", "turnover_base", "adopt", "adopt_as_insurance"]].round(3).to_string())

# ------------------------------------------------------------------ A3 estimator
print("\n=== A3 portfolio-vol estimator ===")
a3 = {"rolling63 (B0)": B0}
for w in [21, 126, 252]:
    a3[f"rolling{w}"] = run(panel, Params(port_vol_estimator="rolling", port_vol_window=w))
a3["ewma0.94"] = run(panel, Params(port_vol_estimator="ewma"))
a3["max(21,63)"] = run(panel, Params(port_vol_estimator="max2163"))
a3["none (ivol only, no overlay)"] = run(panel, Params(port_vol_estimator="none"))
a3_met, a3_adopt = rows_for(a3)
save(a3_met, "a3_estimator", SUB); save(a3_adopt, "a3_adoption", SUB)
print(a3_met[SHOW].round(3).to_string())
print(a3_met[[c for c in a3_met.columns if c.startswith("sh_")]].round(2).to_string())
print(a3_adopt[["worst_block_sharpe_new", "worst_block_sharpe_base", "full_sharpe_new", "boot_diff_median", "boot_diff_p05",
                "boot_diff_p95", "turnover_new", "max_dd_new", "max_dd_base", "adopt", "adopt_as_insurance"]].round(3).to_string())

# ------------------------------------------------------------------ A8 leverage
print("\n=== A8 leverage ===")
# sanity: max_exposure=1.0 must reproduce B0 exactly
x10 = run(panel, Params(max_exposure=1.0))
same = np.allclose(x10["ret"].values, B0["ret"].values, atol=1e-15) and abs(summarize(x10)["sharpe"] - summarize(B0)["sharpe"]) < 5e-4
print("sanity max_exposure=1.0 == B0:", same)
a8 = {}
for x in [1.0, 1.25, 1.5, 2.0]:
    a8[f"x{x} uncapped"] = B0 if x == 1.0 else run(panel, Params(max_exposure=x))
    a8[f"x{x} cap1.5/0.7"] = run(panel, Params(max_exposure=x, total_cap=1.5, per_asset_cap=0.7))
    a8[f"x{x} cap1.5/0.7 continuous"] = run(panel, Params(max_exposure=x, total_cap=1.5, per_asset_cap=0.7, continuous_overlay=True))
a8_met, a8_adopt = rows_for(a8, leverage_extra)
save(a8_met, "a8_leverage", SUB); save(a8_adopt, "a8_adoption", SUB)
cols8 = ["cagr", "vol", "sharpe", "max_dd", "worst_12m", "avg_exposure", "share_days_exposure_gt_1.5", "share_days_exposure_gt_1.0",
         "max_exposure_realised", "financing_cost_per_year", "turnover_per_year", "n_exposure_jumps_gt_0.3", "max_exposure_jump"]
print(a8_met[cols8].round(3).to_string())
print(a8_met[[c for c in a8_met.columns if c.startswith("sh_")]].round(2).to_string())
print(a8_adopt[["worst_block_sharpe_new", "worst_block_sharpe_base", "full_sharpe_new", "boot_diff_median", "boot_diff_p05",
                "boot_diff_p95", "turnover_new", "max_dd_new", "adopt"]].round(3).to_string())

# leverage path diagnostics for the 1.5x uncapped variant: realised vol vs target, and where the jumps come from
r15 = a8["x1.5 uncapped"]
ex = r15["exposure"].loc["1970":]
d = ex.diff()
jumps = d[d.abs() > 0.3]
diag = pd.DataFrame({"exposure_before": ex.shift(1).loc[jumps.index], "exposure_after": ex.loc[jumps.index], "jump": jumps})
save(diag, "a8_x1.5_exposure_jumps", SUB)
print("\nx1.5 uncapped realised vol full/etf:", round(summarize(r15)["vol"], 4), round(summarize(r15, "2005-01-01")["vol"], 4))
print("x1.5 uncapped exposure quantiles:", ex.quantile([.05, .25, .5, .75, .95]).round(3).to_dict())
print("jumps >0.3 (first 10):"); print(diag.head(10).round(3).to_string())
# rolling realised vol of the levered portfolio: how often above 10%?
rv = r15["ret"].loc["1970":].rolling(63).std() * np.sqrt(TRADING_DAYS)
print("x1.5 rolling-63 realised vol: median", round(rv.median(), 4), " share >10%:", round((rv > 0.10).mean(), 3), " share >12%:", round((rv > 0.12).mean(), 3))
rv0 = B0["ret"].loc["1970":].rolling(63).std() * np.sqrt(TRADING_DAYS)
print("B0   rolling-63 realised vol: median", round(rv0.median(), 4), " share >10%:", round((rv0 > 0.10).mean(), 3))
save(pd.DataFrame({"x1.5_rolling63_vol": rv, "B0_rolling63_vol": rv0, "x1.5_exposure": ex}), "a8_x1.5_realised_vol", SUB)
save_json({"sanity_x1_equals_B0": bool(same), "x1.5_vol_full": summarize(r15)["vol"], "x1.5_vol_etf": summarize(r15, "2005-01-01")["vol"]},
          "a8_sanity", SUB)
print("done")
