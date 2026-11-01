"""Equity and drawdown of today's capsule against the sized capsule, alone and in the Growth products. Usage: plot_compare.py research | fresh. Writes compare_<book>_<source>.png into the study's results folder."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import monthly_compare as mc  # noqa: E402

OUT = mc.MAIN / "results" / "scout" / "growth_mr_products_20261101"
TODAY_COLOR, SIZED_COLOR = "#7a7a7a", "#2a78d6"
INK, MUTED, GRID = "#1a1a1a", "#5f5f5f", "#e6e6e6"
BOOKS = (("Growth Plus", "Growth Plus: TAA 1N 50 / MR 30 / CORE5 20", "growth_plus"), ("Capsule", "MR alone: DV2 + HPI, 50 / 50, VIX gate, BIL", "mr_alone"))


def drawdown(r: pd.Series) -> pd.Series:
    nav = (1.0 + r).cumprod()
    return nav / nav.cummax().clip(lower=1.0) - 1.0


def plot(name: str, title: str, slug: str, series: dict, rf: pd.Series, source: str, last: pd.Timestamp) -> Path:
    today, sized = series[f"{name} today"], series[f"{name} sized"]
    st_t, st_s = mc.stats(today, rf), mc.stats(sized, rf)
    fig, (ax_eq, ax_dd) = plt.subplots(2, 1, figsize=(12.5, 8.2), sharex=True, gridspec_kw={"height_ratios": [2.6, 1.0], "hspace": 0.07})
    fig.patch.set_facecolor("white")
    for r, color, label, width in ((today, TODAY_COLOR, "Today: each stock the same size", 1.5), (sized, SIZED_COLOR, "Proposed: sized by liquidity", 2.0)):
        nav = 100_000.0 * (1.0 + r).cumprod()
        ax_eq.plot(nav.index, nav, color=color, lw=width, label=label, solid_capstyle="round")
        ax_eq.annotate(f"${nav.iloc[-1] / 1e6:.2f}M", (nav.index[-1], nav.iloc[-1]), xytext=(7, 0), textcoords="offset points", va="center", fontsize=11, color=INK)
        dd = drawdown(r) * 100.0
        if color == TODAY_COLOR:
            # The two falls nearly coincide: today's as a pale area, so the proposed line never hides it.
            ax_dd.fill_between(dd.index, dd, 0.0, color=TODAY_COLOR, alpha=0.35, lw=0.0)
        else:
            ax_dd.plot(dd.index, dd, color=color, lw=0.9, solid_capstyle="round")
    ax_eq.set_yscale("log")
    ticks = [t for t in (100_000, 200_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000, 50_000_000) if t <= 100_000.0 * (1.0 + sized).prod() * 1.4]
    ax_eq.set_yticks(ticks)
    ax_eq.set_yticklabels([f"${t / 1e6:g}M" if t >= 1e6 else f"${t / 1e3:g}K" for t in ticks])
    ax_eq.minorticks_off()
    ax_eq.set_ylabel("$100K grows to (log scale)", color=MUTED, fontsize=11)
    ax_dd.set_ylabel("Fall from peak, %", color=MUTED, fontsize=11)
    ax_dd.axhline(0.0, color=MUTED, lw=0.8)
    for ax in (ax_eq, ax_dd):
        ax.grid(True, axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, length=0, labelsize=11)
    ax_dd.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_eq.set_xlim(today.index[0], today.index[-1] + pd.Timedelta(days=330))
    ax_eq.legend(loc="upper left", frameon=False, fontsize=12, labelcolor=INK)
    fig.suptitle(title, x=0.075, y=0.975, ha="left", fontsize=16, fontweight="bold", color=INK)
    line = (f"CAGR {st_t['cagr'] * 100:.1f}% vs {st_s['cagr'] * 100:.1f}%     Sharpe {st_t['xs']:.2f} vs {st_s['xs']:.2f}     Max DD {st_t['dd'] * 100:.1f}% vs {st_s['dd'] * 100:.1f}%     Vol {st_t['vol'] * 100:.1f}% vs {st_s['vol'] * 100:.1f}%"
            f"     (today vs proposed, {today.index[0].date()} to {last.date()}, before fees)")
    fig.text(0.075, 0.93, line, fontsize=11, color=MUTED)
    path = OUT / f"compare_{slug}_{source}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    source = sys.argv[1]
    series, rf, ext, lab_end, last = mc.build(source)
    for name, title, slug in BOOKS:
        print(plot(name, title, slug, series, rf, source, last), flush=True)
    rows = {}
    for name in ("Capsule", "Growth", "Growth Plus", "Aggressive"):
        for label in ("today", "sized"):
            rows[f"{name} {label}"] = {**mc.stats(series[f"{name} {label}"], rf), "dd_2023": mc.stats(series[f"{name} {label}"].loc["2023":], rf)["dd"], "xs_2023": mc.stats(series[f"{name} {label}"].loc["2023":], rf)["xs"], "cagr_2023": mc.stats(series[f"{name} {label}"].loc["2023":], rf)["cagr"]}
    print(pd.DataFrame(rows).T.round(3).to_string())
