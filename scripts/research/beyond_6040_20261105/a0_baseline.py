from common import *
panel = dev(load())
B0 = run(panel, Params())
rows = {"B0 (author rule, FF cash)": summarize(B0)}
for nm, r in benchmarks(panel).items(): rows[nm] = summarize(r)
rows["B0 cash 0%"] = summarize(run(panel, Params(cash="zero")))
rows["60/40 cash n/a"] = rows["60/40"]
rows["B0 T-bill cash"] = summarize(run(panel, Params(cash="tbill")))
rows["B0 lag2"] = summarize(run(panel, Params(lag=2)))
rows["B0 cost 10bp"] = summarize(run(panel, Params(cost_bps=10)))
rows["B0 cost 0"] = summarize(run(panel, Params(cost_bps=0)))
df = pd.DataFrame(rows).T
save(df, "baseline_table", "a0")
print(df[[c for c in df.columns if not c.startswith(("sh_","dd_"))]].round(3).to_string())
print(); print(df[[c for c in df.columns if c.startswith("sh_")]].round(2).to_string())
print(); print(df[[c for c in df.columns if c.startswith("dd_")]].round(3).to_string())
# bootstrap B0 vs 60/40
bs = block_bootstrap_diff(B0["ret"].loc["1970":], benchmarks(panel)["60/40"]["ret"].loc["1970":], B0["cash_daily"].loc["1970":])
print("\nbootstrap excess-Sharpe diff B0 - 60/40 (1970-2022):", {k: round(v,3) for k,v in bs.items()})
bs2 = block_bootstrap_diff(B0["ret"].loc["2005":], benchmarks(panel)["60/40"]["ret"].loc["2005":], B0["cash_daily"].loc["2005":])
print("bootstrap excess-Sharpe diff B0 - 60/40 (2005-2022):", {k: round(v,3) for k,v in bs2.items()})
save_json({"full": bs, "etf": bs2}, "bootstrap_b0_vs_6040", "a0")
