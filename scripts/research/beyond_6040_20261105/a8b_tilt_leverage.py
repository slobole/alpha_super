"""A8b: the tilted book uses ~60% of its risk budget; does scaling it back toward 8% (author's 'scaled' version) pay?"""
from common import *
panel = dev(load())
B0 = run(panel, Params())
T1 = run(panel, Params(tilt="sma10", tilt_strength=0.5, tilt_freq="M"))
arms = {
 "B0": B0, "T1 (tilt, no leverage)": T1,
 "T1 x1.25 capped, continuous": run(panel, Params(tilt="sma10", tilt_strength=0.5, tilt_freq="M", max_exposure=1.25, total_cap=1.5, per_asset_cap=0.7, continuous_overlay=True)),
 "T1 x1.5 capped, continuous": run(panel, Params(tilt="sma10", tilt_strength=0.5, tilt_freq="M", max_exposure=1.5, total_cap=1.5, per_asset_cap=0.7, continuous_overlay=True)),
 "T1 x1.5 capped, dead band": run(panel, Params(tilt="sma10", tilt_strength=0.5, tilt_freq="M", max_exposure=1.5, total_cap=1.5, per_asset_cap=0.7)),
 "T1 x2.0 capped 2.0/0.9, continuous": run(panel, Params(tilt="sma10", tilt_strength=0.5, tilt_freq="M", max_exposure=2.0, total_cap=2.0, per_asset_cap=0.9, continuous_overlay=True)),
 "T1 continuous overlay x1": run(panel, Params(tilt="sma10", tilt_strength=0.5, tilt_freq="M", continuous_overlay=True)),
 "B0 x1.5 capped, continuous": run(panel, Params(max_exposure=1.5, total_cap=1.5, per_asset_cap=0.7, continuous_overlay=True)),
}
rows = {k: summarize(r) for k, r in arms.items()}
for k, r in arms.items():
    e = r["exposure"].loc["1970":]; rows[k]["pct_days_exp_gt_1"] = float((e > 1.0).mean()); rows[k]["max_exposure_seen"] = e.max()
    fin = (np.maximum(e - 1, 0) * (panel["dff"].loc["1970":] + 0.015)).mean(); rows[k]["financing_cost_per_year"] = fin
df = pd.DataFrame(rows).T; save(df, "tilt_leverage", "a8")
print(df[["cagr","excess_cagr","vol","sharpe","max_dd","worst_12m","worst_36m","avg_exposure","pct_days_exp_gt_1","max_exposure_seen","financing_cost_per_year","turnover_per_year","sh_1970s","sh_1980s","sh_1990s","sh_2000s","sh_2010s","sh_2020-22","sh_ETF era 2005-22"]].round(3).to_string())
ad = pd.DataFrame({k: adoption_check(r, T1, panel) for k, r in arms.items() if k not in ("B0", "T1 (tilt, no leverage)")}).T
save(ad, "tilt_leverage_adoption_vs_T1", "a8"); print(); print(ad[["a_worst_block_up","b_full_not_down","boot_diff_median","boot_diff_p05","c_boot_ok","d_turnover_ok","max_dd_new","cagr_premium","adopt"]].round(3).to_string())
