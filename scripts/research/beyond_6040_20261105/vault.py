"""G6: open the vault once. 2023-01-01 -> latest, for the chosen variant T1, the baseline B0 and fixed benchmarks."""
from common import *
full = load()
cands = {"T1 (chosen)": Params(tilt="sma10", tilt_strength=0.5, tilt_freq="M"), "B0 (author rule)": Params(),
         "T1 x1.5 capped (growth dial)": Params(tilt="sma10", tilt_strength=0.5, tilt_freq="M", max_exposure=1.5, total_cap=1.5, per_asset_cap=0.7, continuous_overlay=True)}
res = {k: run(full, p) for k, p in cands.items()} | benchmarks(full)
rows = {}
for k, r in res.items():
    m = metrics(r["ret"].loc[VAULT_START:], r["cash_daily"].loc[VAULT_START:], r["exposure"].loc[VAULT_START:], r["turnover"].loc[VAULT_START:])
    rows[k] = {kk: m.get(kk) for kk in ["start", "end", "years", "cagr", "excess_cagr", "vol", "sharpe", "sharpe_raw", "max_dd", "worst_12m", "avg_exposure", "turnover_per_year"]}
    # calendar years
    y = (1 + r["ret"].loc[VAULT_START:]).resample("YE").prod() - 1
    for d, v in y.items(): rows[k][f"y{d.year}"] = v
df = pd.DataFrame(rows).T; save(df, "vault_results", "vault"); print(df.round(3).to_string())
# G6 bands: block-bootstrap paths of the vault length from the pre-vault daily excess returns
rng = np.random.default_rng(11)
cash = res["T1 (chosen)"]["cash_daily"]
out = {}
for k in ["T1 (chosen)", "B0 (author rule)"]:
    r = res[k]["ret"]; ex_dev = (r - cash).loc["1970":DEV_END].values; ex_v = (r - cash).loc[VAULT_START:]
    H = len(ex_v); T = len(ex_dev); sims = np.empty(4000); dds = np.empty(4000)
    for i in range(4000):
        idx = np.empty(H, dtype=int); pos = 0
        while pos < H:
            s0 = rng.integers(0, T); L = min(rng.geometric(1 / 63), H - pos); idx[pos:pos + L] = (s0 + np.arange(L)) % T; pos += L
        path = ex_dev[idx]; sims[i] = (1 + path).prod() - 1
        nav = np.cumprod(1 + r.loc["1970":DEV_END].values[idx]); dds[i] = (nav / np.maximum.accumulate(nav) - 1).min()
    realised = (1 + ex_v).prod() - 1
    nav_v = (1 + r.loc[VAULT_START:]).cumprod(); dd_v = (nav_v / nav_v.cummax() - 1).min()
    dev_blocks = [metrics(r.loc[s:e], cash.loc[s:e])["max_dd"] for k, (s, e) in DECADES.items() if not k.startswith(("ETF", "Full"))]  # decades only (audit: Full block was included)
    out[k] = {"realised_excess_return": realised, "band_p05": float(np.percentile(sims, 5)), "band_p50": float(np.percentile(sims, 50)), "band_p95": float(np.percentile(sims, 95)),
              "percentile_of_realised": float((sims < realised).mean()), "realised_max_dd": dd_v, "worst_dev_decade_dd": min(dev_blocks), "dd_limit_1.25x": 1.25 * min(dev_blocks),
              "G6a_inside_band": bool(np.percentile(sims, 5) <= realised <= np.percentile(sims, 95)), "G6b_dd_ok": bool(dd_v >= 1.25 * min(dev_blocks)),
              "sim_dd_p05": float(np.percentile(dds, 5)), "sim_dd_p50": float(np.percentile(dds, 50))}
save_json(out, "g6_vault_gate", "vault"); print(); print(json.dumps(out, indent=1))
pd.DataFrame({k: v["ret"] for k, v in res.items()}).to_parquet(OUT / "vault" / "series_daily_full.parquet")
pd.DataFrame({k: v["exposure"] for k, v in res.items()}).to_parquet(OUT / "vault" / "exposure_daily_full.parquet")
for k in ["T1 (chosen)", "B0 (author rule)"]:
    res[k]["weights"].to_parquet(OUT / "vault" / f"weights_{k.split()[0]}.parquet")
