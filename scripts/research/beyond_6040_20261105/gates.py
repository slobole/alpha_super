"""Pre-registered gates G1-G5 (SPEC.md) for B0 and the tilt candidate T1 on the development sample."""
from common import *
panel = dev(load())
bm = benchmarks(panel)
cands = {"B0 (author rule)": Params(), "T1 (B0 + SMA10 tilt k=0.5, monthly)": Params(tilt="sma10", tilt_strength=0.5, tilt_freq="M")}
b6040 = summarize(bm["60/40"]); b6040_zero = summarize(bm["60/40"])  # 60/40 holds no cash
blocks = ["sh_1970s", "sh_1980s", "sh_1990s", "sh_2000s", "sh_2010s", "sh_2020-22"]
out = {}
grid = pd.read_csv(OUT / "a1" / "plateau_grid.csv") if (OUT / "a1" / "plateau_grid.csv").exists() else None
tgrid = pd.read_csv(OUT / "a1" / "tilt_plateau_grid.csv")
wf = json.loads((OUT / "a13" / "walk_forward.json").read_text())
for name, p in cands.items():
    s = summarize(run(panel, p)); s0 = summarize(run(panel, Params(**{**p.__dict__, "cash": "zero"})))
    g = {}
    g["G1a full excess Sharpe >= 60/40 + 0.15"] = (s["sharpe"], b6040["sharpe"] + 0.15, s["sharpe"] >= b6040["sharpe"] + 0.15)
    wins = sum(s[b] >= b6040[b] for b in blocks); worst_gap = min(s[b] - b6040[b] for b in blocks)
    g["G1b >= 60/40 in >= 5 of 6 blocks"] = (wins, 5, wins >= 5)
    g["G1c never below 60/40 by > 0.10 in a block"] = (worst_gap, -0.10, worst_gap >= -0.10)
    g["G2 max DD <= 2/3 of 60/40 (full)"] = (s["max_dd"], b6040["max_dd"] * 2 / 3, s["max_dd"] >= b6040["max_dd"] * 2 / 3)
    g["G2 max DD <= 2/3 of 60/40 (ETF era)"] = (s["dd_ETF era 2005-22"], b6040["dd_ETF era 2005-22"] * 2 / 3, s["dd_ETF era 2005-22"] >= b6040["dd_ETF era 2005-22"] * 2 / 3)
    gd = grid if name.startswith("B0") else tgrid
    if gd is not None:
        if name.startswith("B0"):
            cell = gd[(gd.vol_window == 63) & (gd.target_vol == 0.08) & (gd.trigger_buffer == 0.005) & (gd.rebalance == "M") & (gd.band == 0.0)]
            nb = gd[(gd.vol_window.isin([42, 63, 126])) & (gd.target_vol.isin([0.07, 0.08, 0.09])) & (gd.rebalance.isin(["W", "M", "Q"]))]
        else:
            cell = gd[(gd.vol_window == 63) & (gd.target_vol == 0.08) & (gd.band == 0.0) & (gd.k == 0.5)]
            nb = gd[(gd.vol_window.isin([42, 63, 126])) & (gd.target_vol.isin([0.07, 0.08, 0.09])) & (gd.k == 0.5)]
        g["G3a chosen cell within 0.05 of neighbourhood median Sharpe"] = (float(cell.sharpe.iloc[0]), float(nb.sharpe.median()), abs(cell.sharpe.iloc[0] - nb.sharpe.median()) <= 0.05)
        g["G3b >= 80% of grid beats 60/40 Sharpe"] = (float((gd.sharpe > b6040["sharpe"]).mean()), 0.8, (gd.sharpe > b6040["sharpe"]).mean() >= 0.8)
        g["G3 grid Sharpe range"] = (float(gd.sharpe.min()), float(gd.sharpe.max()), None)
    key = "B0_fixed_sharpe_1980_2022" if name.startswith("B0") else "T1_fixed_sharpe_1980_2022"
    g["G4 walk-forward Sharpe >= fixed - 0.10 (1980-2022)"] = (wf["walk_forward_sharpe_1980_2022"], wf[key] - 0.10, wf["walk_forward_sharpe_1980_2022"] >= wf[key] - 0.10)
    g["G5 with 0% cash: Sharpe >= 60/40 raw + 0.15"] = (s0["sharpe"], b6040["sharpe_raw"] + 0.15, s0["sharpe"] >= b6040["sharpe_raw"] + 0.15)
    g["G5 with 0% cash: max DD <= 2/3 of 60/40"] = (s0["max_dd"], b6040["max_dd"] * 2 / 3, s0["max_dd"] >= b6040["max_dd"] * 2 / 3)
    out[name] = g
rows = []
for name, g in out.items():
    for k, (v, thr, ok) in g.items():
        rows.append({"candidate": name, "gate": k, "value": v, "threshold": thr, "pass": ok})
df = pd.DataFrame(rows); save(df, "gates_dev", "gates")
print(df.round(3).to_string())
