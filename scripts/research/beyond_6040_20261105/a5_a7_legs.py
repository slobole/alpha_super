"""Arms A5 (equity leg), A6 (bond leg), A7 (inflation leg) of SPEC.md. Research-only.

Every arm is compared with B0 (author's rule) on the SAME window. Both results are trimmed to the
window before summarising so decade blocks are identical. Composite legs are daily-rebalanced
50/50 columns added to a copy of the panel; the engine then applies inverse-vol across the listed
assets. Where a leg starts late the window starts >= 1 year after the first valid date so the
63-day vol windows are warm (the engine holds cash while any asset's vol is NaN).
"""
from common import *

SUB = "a5_a7"
panel = dev(load())
panel = panel.copy()
panel["r_eq_5050"] = 0.5 * panel["r_stock"] + 0.5 * panel["r_world"]        # NaN before 1990-07
panel["r_bond_5050"] = 0.5 * panel["r_bond"] + 0.5 * panel["r_bond10"]
panel["r_infl_gold_dbc"] = 0.5 * panel["r_gold"] + 0.5 * panel["r_dbc"]      # NaN before 2006-02
panel["r_infl_gold_tip"] = 0.5 * panel["r_gold"] + 0.5 * panel["r_tip"]      # NaN before 2003-12

B0 = run(panel, Params())
B6040 = fixed_mix(panel, {"r_stock": .6, "r_bond": .4})


def trim(res, start, end):
    out = dict(res)
    for k in ("ret", "exposure", "turnover", "weights", "cash_daily"):
        out[k] = res[k].loc[start:end]
    return out


def avg_weights(res, start, end):
    w = res["weights"].loc[start:end]
    dollar = w.mean()
    s = w.sum(axis=1)
    alloc = w.div(s.where(s > 0), axis=0).mean()   # share of the risky sleeve (sums to 1)
    return dollar, alloc


def nan_safe_adoption(res_arm, res_b0, start, end):
    """adoption_check from common.py on the trimmed results; decade blocks that fall outside the
    window are NaN in summarize() and would poison min(); recompute the worst block with nanmin."""
    a_t, b_t = trim(res_arm, start, end), trim(res_b0, start, end)
    chk = adoption_check(a_t, b_t, panel, start, end)
    sa, sb = summarize(a_t, start, end), summarize(b_t, start, end)
    blocks = [k for k in sa if k.startswith("sh_") and not k.startswith("sh_ETF")]
    wa = np.nanmin([sa[k] for k in blocks]); wb = np.nanmin([sb[k] for k in blocks])
    chk["worst_block_sharpe_new"], chk["worst_block_sharpe_base"] = wa, wb
    chk["a_worst_block_up"] = bool(wa > wb)
    chk["adopt"] = bool(chk["a_worst_block_up"] and chk["b_full_not_down"] and chk["c_boot_ok"] and chk["d_turnover_ok"])
    chk["adopt_as_insurance"] = bool((not chk["adopt"]) and chk["dd_cut_25pct"] and chk["cagr_premium"] <= 0.005
                                     and chk["c_boot_ok"] and chk["d_turnover_ok"])
    chk["blocks_in_window"] = ",".join(k[3:] for k in blocks if np.isfinite(sa[k]))
    return chk


ARMS = {
    # name: (assets, start, end)
    "A5a eq 50/50 US/world": (("r_eq_5050", "r_bond", "r_gold"), "1991-07-01", DEV_END),
    "A5b eq 100% world ex-US": (("r_world", "r_bond", "r_gold"), "1991-07-01", DEV_END),
    "A5c eq VT": (("r_vt", "r_bond", "r_gold"), "2009-07-01", DEV_END),
    "A5d eq 50/50 US/EFA (EFA era)": (("r_eq_5050", "r_bond", "r_gold"), "2002-09-01", DEV_END),
    "A6a bond 10y, full": (("r_stock", "r_bond10", "r_gold"), "1970-01-01", DEV_END),
    "A6b bond 50/50 25y/10y, full": (("r_stock", "r_bond_5050", "r_gold"), "1970-01-01", DEV_END),
    "A6c 4 assets +10y, full": (("r_stock", "r_bond", "r_bond10", "r_gold"), "1970-01-01", DEV_END),
    "A6a bond 10y, ETF": (("r_stock", "r_bond10", "r_gold"), "2005-01-01", DEV_END),
    "A6b bond 50/50 25y/10y, ETF": (("r_stock", "r_bond_5050", "r_gold"), "2005-01-01", DEV_END),
    "A6c 4 assets +10y, ETF": (("r_stock", "r_bond", "r_bond10", "r_gold"), "2005-01-01", DEV_END),
    "A7a infl 50/50 gold/DBC": (("r_stock", "r_bond", "r_infl_gold_dbc"), "2007-02-01", DEV_END),
    "A7b infl gold + DBC separate": (("r_stock", "r_bond", "r_gold", "r_dbc"), "2007-02-01", DEV_END),
    "A7c infl 50/50 gold/TIP": (("r_stock", "r_bond", "r_infl_gold_tip"), "2007-02-01", DEV_END),
    "A7d infl gold + TIP separate": (("r_stock", "r_bond", "r_gold", "r_tip"), "2007-02-01", DEV_END),
}

summ_rows, adopt_rows, w_rows = {}, {}, []
for name, (assets, start, end) in ARMS.items():
    res = run(panel, Params(assets=assets))
    win = f"{start[:7]}..{end[:7]}"
    r_arm = summarize(trim(res, start, end), start, end)
    r_b0 = summarize(trim(B0, start, end), start, end)
    r_6040 = summarize(trim(B6040, start, end), start, end)
    # sanity: identical windows
    a_ret, b_ret = res["ret"].loc[start:end], B0["ret"].loc[start:end]
    assert a_ret.index.equals(b_ret.index), name
    first_live = res["exposure"].loc[start:end].dropna()
    first_live = first_live[first_live > 0].index[0]
    assert first_live == a_ret.index[0], f"{name}: arm not invested from window start ({first_live.date()})"
    for lab, row in ((name, r_arm), (f"B0 @ {win}", r_b0), (f"60/40 @ {win}", r_6040)):
        row = dict(row); row["window"] = win; row["n_days"] = len(a_ret); row["arm_group"] = name
        summ_rows[lab] = row
    chk = nan_safe_adoption(res, B0, start, end)
    chk["window"] = win
    adopt_rows[name] = chk
    d, al = avg_weights(res, start, end)
    d0, al0 = avg_weights(B0, start, end)
    for lab, dd_, aa_ in ((name, d, al), (f"B0 @ {win}", d0, al0)):
        for asset in dd_.index:
            w_rows.append({"arm": lab, "window": win, "asset": asset, "avg_dollar_weight": dd_[asset], "avg_sleeve_share": aa_[asset]})
    print(f"done {name}  sharpe arm {r_arm['sharpe']:.3f} vs B0 {r_b0['sharpe']:.3f}  adopt={chk['adopt']}")

summ = pd.DataFrame(summ_rows).T
adopt = pd.DataFrame(adopt_rows).T
weights = pd.DataFrame(w_rows)
save(summ, "a5_a7_summary", SUB)
save(adopt, "a5_a7_adoption", SUB)
save(weights, "a5_a7_avg_weights", SUB)

main_cols = ["window", "cagr", "excess_cagr", "vol", "sharpe", "max_dd", "worst_12m", "turnover_per_year", "avg_exposure"]
print("\n=== headline (same window per group) ===")
print(summ[main_cols].astype({c: float for c in main_cols[1:]}).round(3).to_string())
print("\n=== decade-block excess Sharpe ===")
print(summ[[c for c in summ.columns if c.startswith("sh_")]].astype(float).round(2).to_string())
print("\n=== decade-block max DD ===")
print(summ[[c for c in summ.columns if c.startswith("dd_")]].astype(float).round(3).to_string())
print("\n=== adoption rule ===")
ac = ["window", "blocks_in_window", "worst_block_sharpe_new", "worst_block_sharpe_base", "a_worst_block_up", "full_sharpe_new",
      "full_sharpe_base", "b_full_not_down", "boot_diff_median", "boot_diff_p05", "boot_diff_p95", "c_boot_ok",
      "turnover_new", "turnover_base", "d_turnover_ok", "max_dd_new", "max_dd_base", "cagr_premium", "adopt", "adopt_as_insurance"]
print(adopt[ac].to_string())
print("\n=== average weights (dollar / sleeve share) ===")
print(weights.round(3).to_string())

# --- calendar-year 2022 and 2008 returns (stress check for the inflation and bond legs)
yr_rows = {}
def yr(res, y):
    r = res["ret"].loc[f"{y}-01-01":f"{y}-12-31"]
    return float((1 + r).prod() - 1)
for name, (assets, start, end) in ARMS.items():
    res = run(panel, Params(assets=assets))
    yr_rows[name] = {"ret_2008": yr(res, 2008) if start <= "2008-01-01" else np.nan, "ret_2020": yr(res, 2020), "ret_2022": yr(res, 2022),
                     "max_dd_2022": float((lambda n: (n / n.cummax() - 1).min())((1 + res["ret"].loc["2022"]).cumprod()))}
yr_rows["B0"] = {"ret_2008": yr(B0, 2008), "ret_2020": yr(B0, 2020), "ret_2022": yr(B0, 2022),
                 "max_dd_2022": float((lambda n: (n / n.cummax() - 1).min())((1 + B0["ret"].loc["2022"]).cumprod()))}
yrs = pd.DataFrame(yr_rows).T
save(yrs, "a5_a7_stress_years", SUB)
print("\n=== calendar-year stress returns ===")
print(yrs.round(3).to_string())
