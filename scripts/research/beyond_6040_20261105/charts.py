"""Charts for the Beyond 60/40 report (PNG, light theme, English labels)."""
from common import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
C = OUT / "charts"; C.mkdir(exist_ok=True)
NAVY, RED, GREY, GOLD, GREEN = "#1f3a5f", "#c0392b", "#8a8f98", "#b8860b", "#2e7d32"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
full = pd.read_parquet(OUT / "vault" / "series_daily_full.parquet")
expo = pd.read_parquet(OUT / "vault" / "exposure_daily_full.parquet")
panel = load()
cash = panel["dff"] / 252
S = {"T1": full["T1 (chosen)"], "B0": full["B0 (author rule)"], "60/40": full["60/40"], "EW3": full["EW3"], "Stocks": full["Stocks"]}
col = {"T1": NAVY, "B0": GOLD, "60/40": RED, "EW3": GREY, "Stocks": "#444"}
# 1. equity + drawdown 1970-2026
fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
for k in ["Stocks", "60/40", "B0", "T1"]:
    nav = (1 + S[k].loc["1970":]).cumprod(); ax[0].plot(nav, color=col[k], lw=1.3 if k == "T1" else 1.0, label=k)
    dd = nav / nav.cummax() - 1; ax[1].plot(dd, color=col[k], lw=0.9)
ax[0].set_yscale("log"); ax[0].legend(loc="upper left", frameon=False, ncol=4); ax[0].set_title("Growth of $1, 1970-2026 (log). Shaded: vault 2023+", loc="left")
for a in ax: a.axvspan(pd.Timestamp("2023-01-01"), nav.index[-1], color="#eee", zorder=0)
ax[1].set_ylabel("drawdown"); ax[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
fig.tight_layout(); fig.savefig(C / "equity_drawdown.png"); plt.close(fig)
# 2. decade excess Sharpe bars
blocks = {k: v for k, v in DECADES.items() if not k.startswith("Full") and not k.startswith("ETF")} | {"2023-26 vault": ("2023-01-01", "2026-12-31")}
rows = {k: {b: metrics(S[k].loc[s:e], cash.loc[s:e])["sharpe"] for b, (s, e) in blocks.items()} for k in ["60/40", "B0", "T1"]}
df = pd.DataFrame(rows); df.to_csv(C / "decade_sharpe.csv")
fig, ax = plt.subplots(figsize=(11, 4)); x = np.arange(len(df)); w = 0.26
for i, k in enumerate(["60/40", "B0", "T1"]): ax.bar(x + (i - 1) * w, df[k], w, color=col[k], label=k)
ax.set_xticks(x); ax.set_xticklabels(df.index); ax.axhline(0, color="k", lw=0.6); ax.legend(frameon=False); ax.set_title("Sharpe over cash by period", loc="left")
fig.tight_layout(); fig.savefig(C / "decade_sharpe.png"); plt.close(fig)
# 3. plateau heatmaps: B0 grid (vol window x target, monthly, buffer 0.5, band 0) and T1 grid (vol window x target, k=0.5)
g = pd.read_csv(OUT / "a1" / "plateau_grid.csv"); tg = pd.read_csv(OUT / "a1" / "tilt_plateau_grid.csv")
gb = g[(g.rebalance == "M") & (g.trigger_buffer == 0.005) & (g.band == 0.0)].pivot(index="vol_window", columns="target_vol", values="sharpe")
tb = tg[(tg.k == 0.5) & (tg.band == 0.0)].pivot(index="vol_window", columns="target_vol", values="sharpe")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for a, m, t in [(ax[0], gb, "B0: Sharpe over cash by vol window x target (monthly)"), (ax[1], tb, "T1 (SMA10 tilt k=0.5): same grid")]:
    im = a.imshow(m.values, cmap="Blues", vmin=0.3, vmax=0.7, aspect="auto"); a.set_xticks(range(m.shape[1])); a.set_xticklabels([f"{c:.0%}" for c in m.columns]); a.set_yticks(range(m.shape[0])); a.set_yticklabels(m.index)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]): a.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center", fontsize=8, color="k")
    a.set_title(t, loc="left", fontsize=9); a.set_xlabel("vol target"); a.set_ylabel("vol window (days)")
fig.text(0.01, 0.01, f"60/40 Sharpe over cash 1970-2022: 0.45. B0 grid {g.sharpe.min():.2f}-{g.sharpe.max():.2f} over 450 cells; T1 grid {tg.sharpe.min():.2f}-{tg.sharpe.max():.2f} over 60 cells.", fontsize=8)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(C / "plateau.png"); plt.close(fig)
# 4. regimes: ann excess return by stock-bond correlation x stock trend regime
r = pd.read_csv(OUT / "a4" / "trend_by_regime.csv", index_col=0)
r = r.loc[[i for i in r.index if "ann excess" in i], ["B0", "tilt sma10 k=0.5 monthly"]].rename(columns={"tilt sma10 k=0.5 monthly": "T1"})
r.index = [i.replace(" ann excess ret", "") for i in r.index]
fig, ax = plt.subplots(figsize=(8, 3.6)); x = np.arange(len(r)); ax.bar(x - 0.18, r["B0"], 0.36, color=GOLD, label="B0"); ax.bar(x + 0.18, r["T1"], 0.36, color=NAVY, label="T1")
ax.set_xticks(x); ax.set_xticklabels(r.index, fontsize=8); ax.axhline(0, color="k", lw=0.6); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); ax.legend(frameon=False)
ax.set_title("Annual return over cash by regime, 1970-2022 (120-day stock-bond correlation x trailing 12m stock return)", loc="left", fontsize=9)
fig.tight_layout(); fig.savefig(C / "regimes.png"); plt.close(fig)
# 5. tilt strength plateau (SMA length x k)
tgrid = pd.read_csv(OUT / "a4" / "tilt_grid.csv"); tt = tgrid[tgrid.tilt.str.match(r"^sma[0-9]+m$") & (tgrid.freq == "M")].pivot(index="tilt", columns="k", values="sharpe")
dd = tgrid[tgrid.tilt.str.match(r"^sma[0-9]+m$") & (tgrid.freq == "M")].pivot(index="tilt", columns="k", values="max_dd")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for a, m, t, cm, vmin, vmax in [(ax[0], tt, "Sharpe over cash", "Blues", 0.4, 0.7), (ax[1], dd, "Max drawdown", "Reds_r", -0.25, -0.08)]:
    a.imshow(m.values, cmap=cm, vmin=vmin, vmax=vmax, aspect="auto"); a.set_xticks(range(m.shape[1])); a.set_xticklabels(m.columns); a.set_yticks(range(m.shape[0])); a.set_yticklabels(m.index)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]): a.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    a.set_title(f"Trend tilt: {t} by SMA length x tilt strength k", loc="left", fontsize=9); a.set_xlabel("k (0 = no tilt, 1 = fully out when below SMA)")
fig.tight_layout(); fig.savefig(C / "tilt_grid.png"); plt.close(fig)
# 6. vault panel 2023-2026
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for k in ["60/40", "B0", "T1"]:
    nav = (1 + S[k].loc["2023":]).cumprod(); ax[0].plot(nav, color=col[k], label=k)
ax[0].legend(frameon=False); ax[0].set_title("Vault 2023-01 to 2026-10: growth of $1 (opened once, after the design was frozen)", loc="left", fontsize=9)
ax[1].plot(expo["T1 (chosen)"].loc["2023":], color=NAVY, lw=0.8, label="T1 exposure"); ax[1].plot(expo["B0 (author rule)"].loc["2023":], color=GOLD, lw=0.8, label="B0 exposure"); ax[1].set_ylim(0, 1.05); ax[1].legend(frameon=False); ax[1].set_title("Risky exposure (rest in cash)", loc="left", fontsize=9)
fig.tight_layout(); fig.savefig(C / "vault.png"); plt.close(fig)
# 7. weights of T1 over time (stacked, monthly)
w = pd.read_parquet(OUT / "vault" / "weights_T1.parquet").loc["1970":].resample("ME").last().clip(lower=0)
fig, ax = plt.subplots(figsize=(11, 3.2)); ax.stackplot(w.index, w["r_stock"], w["r_bond"], w["r_gold"], 1 - w.sum(axis=1).clip(upper=1), colors=["#444", RED, GOLD, "#ddd"], labels=["stocks", "long bond", "gold", "cash"])
ax.legend(loc="upper left", ncol=4, frameon=False, fontsize=8); ax.set_ylim(0, 1); ax.set_title("T1 allocation over time (month-end)", loc="left", fontsize=9)
fig.tight_layout(); fig.savefig(C / "weights_t1.png"); plt.close(fig)
# 8. ILS panel from a10 tables
try:
    a10 = pd.read_csv(OUT / "a10" / "policies_1993_2022.csv", index_col=0); a10.to_csv(C / "ils_policies.csv"); print(a10.round(3).to_string())
except Exception as e: print("a10", e)
print("charts done")
