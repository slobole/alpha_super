"""Every table quoted in docs/research/GROWTH_SHELF_V2_20260926.md, regenerated from the study outputs.

Usage: python report_tables.py   (writes results/.../growth_shelf_v2_20260926/report_tables.md and prints it)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "portfolio" / "growth_shelf_v2_20260926"


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def money(x: float) -> str:
    return "-" if pd.isna(x) else f"${x / 1e6:g}M"


def main() -> int:
    lines = []
    validation = json.loads((OUT / "proxy" / "proxy_validation.json").read_text(encoding="utf-8"))
    lev = validation["levered"]
    lines += ["## A1 leveraged ETFs", "", "| Fund | drag/yr | daily corr (fit) | median |year diff| | OOS daily corr | OOS cum syn / real | GFC syn / real |",
              "|---|---|---|---|---|---|---|"]
    for fund, v in lev.items():
        lines.append(f"| {fund} | {pct(v['drag_per_year'], 2)} | {v['fit_daily_corr']:.4f} | {pct(v['fit_median_abs_year_diff'])} | "
                     f"{v.get('oos_daily_corr', float('nan')):.4f} | "
                     + (f"{pct(v['oos_cum_syn'])} / {pct(v['oos_cum_real'])}" if "oos_cum_syn" in v else "-") + " | "
                     + (f"{pct(v['gfc_syn'])} / {pct(v['gfc_real'])}" if "gfc_syn" in v else "-") + " |")
    btal = validation["btal"]
    lines += ["", "## A2 synthetic BTAL", "", "| Variant | daily corr | monthly corr |", "|---|---|---|"]
    for name, v in btal["variants"].items():
        lines.append(f"| {name} | {v['daily_corr']:.3f} | {v['monthly_corr']:.3f} |")
    for label in ("scaled", "unscaled"):
        v = btal[label]
        lines.append(f"\n{label}: k {v['k']:.3f}, drag {pct(v['drag_per_year'])}, beta {v['beta_syn']:.3f} vs real {v['beta_real']:.3f}, "
                     f"vol {pct(v['vol_syn'])} vs {pct(v['vol_real'])}, COVID {pct(v['covid_syn'])} vs {pct(v['covid_real'])}, "
                     f"2022 bear {pct(v['bear2022_syn'])} vs {pct(v['bear2022_real'])}, GFC {pct(v['gfc_syn'])}")
    years = sorted(set(btal["scaled"]["year_syn"]) & set(btal["scaled"]["year_real"]))
    lines += ["", "| Year | " + " | ".join(years) + " |", "|---|" + "---|" * len(years),
              "| synthetic (scaled) | " + " | ".join(pct(btal["scaled"]["year_syn"][y], 0) for y in years) + " |",
              "| BTAL | " + " | ".join(pct(btal["scaled"]["year_real"][y], 0) for y in years) + " |"]
    a3 = pd.read_csv(OUT / "proxy_runs" / "a3_strategy_validation.csv")
    lines += ["", "## A3 strategy-level validation (2012-10-02 -> 2026-08-19)", "",
              "| Sleeve | BTAL version | monthly corr | CAGR syn / real | CAGR diff | max DD syn / real | allocation overlap | pass |",
              "|---|---|---|---|---|---|---|---|"]
    for _, r in a3.iterrows():
        lines.append(f"| {r['alias']} | {r['mode']} | {r['monthly_corr']:.3f} | {pct(r['cagr_syn'])} / {pct(r['cagr_real'])} | "
                     f"{r['cagr_diff_pp']:+.1f} pp | {pct(r['maxdd_syn'])} / {pct(r['maxdd_real'])} | {pct(r['allocation_overlap_mean'])} | {r['pass']} |")
    growth = pd.read_csv(OUT / "growth_shelf_v2.csv", index_col=0)
    order = growth.assign(key=growth["long_calmar"].where(growth["gates_pass"], -1)).sort_values(["key", "long_calmar"], ascending=False).index
    lines += ["", "## C growth shelf (exact window 2012-10-02 -> 2026-08-19; long window from 2008-03-04 with the new proxy)", "",
              "| Book | CAGR | Sharpe | Max DD | Calmar | DD incl. 2008 | Calmar incl. 2008 | GFC | Sharpe 12-21 / 22-26 | Calmar +5bps | Worst year | Gates |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for book in order:
        r = growth.loc[book]
        lines.append(f"| {book} | {pct(r['cagr'])} | {r['sharpe']:.2f} | {pct(r['maxdd'])} | {r['calmar']:.2f} | {pct(r['long_maxdd'])} | "
                     f"{r['long_calmar']:.2f} | {pct(r['gfc'])} | {r['sharpe_2012_21']:.2f} / {r['sharpe_2022_26']:.2f} | {r['calmar_plus5']:.2f} | "
                     f"{pct(r['worst_year'])} | {'pass' if r['gates_pass'] else 'FAIL'} |")
    lines += ["", "| Book | Old proxy DD incl. 2008 / GFC | New unscaled DD / GFC | 2018 Q4 | COVID crash | 2022 bear | 2025 tariffs |",
              "|---|---|---|---|---|---|---|"]
    for book in order:
        r = growth.loc[book]
        lines.append(f"| {book} | {pct(r['long_maxdd_old'])} / {pct(r['gfc_old'])} | {pct(r['long_maxdd_new_unscaled'])} / {pct(r['gfc_new_unscaled'])} | "
                     f"{pct(r['q4_2018'])} | {pct(r['covid'])} | {pct(r['bear_2022'])} | {pct(r['tariffs_2025'])} |")
    lines += ["", "| Book | Pods | Trading days/yr | Turnover x NAV/yr | WIRED share | Daily MR | MOO today | MOC | Worked + blocks | First fail (fund route) | Cost at $25M (fund route) | Missing before live |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for book in order:
        r = growth.loc[book]
        lines.append(f"| {book} | {int(r['pods'])} | {r['trade_days_per_year']:.0f} | {r['turnover_x_nav']:.1f} | {pct(r['wired_share'], 0)} | "
                     f"{'yes' if r['daily_mr'] else 'no'} | {money(r['MOO_recommended'])} | {money(r['MOC_recommended'])} | {money(r['worked+blocks_recommended'])} | "
                     f"{r['worked+blocks_first_fail']} | {pct(r['worked+blocks_cost_at_25m'], 2)} | {r['needs']} |")
    fixed = pd.read_csv(OUT / "growth_shelf_v2_commission_fixed.csv", index_col=0)
    order_fixed = fixed.assign(key=fixed["long_calmar"].where(fixed["gates_pass"], -1)).sort_values(["key", "long_calmar"], ascending=False).index
    lines += ["", "## C growth shelf with commissions re-priced on real share counts (review H1)", "",
              "| Book | CAGR | Sharpe | Max DD | Calmar | DD incl. 2008 | Calmar incl. 2008 | GFC | Sharpe 12-21 / 22-26 | Gates |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for book in order_fixed:
        r = fixed.loc[book]
        lines.append(f"| {book} | {pct(r['cagr'])} | {r['sharpe']:.3f} | {pct(r['maxdd'])} | {r['calmar']:.2f} | {pct(r['long_maxdd'])} | "
                     f"{r['long_calmar']:.3f} | {pct(r['gfc'])} | {r['sharpe_2012_21']:.2f} / {r['sharpe_2022_26']:.3f} | "
                     f"{'pass' if r['gates_pass'] else 'FAIL'} |")
    lines += ["", "## Close-auction limit at $25M by pod (share of 20-day median dollar volume; limit P95 0.25%, P99 0.50%)", "",
              "| Book | DV2 P95 / P99 / orders > 0.50% | HPI P95 / P99 / orders > 0.50% |", "|---|---|---|"]
    for book in order:
        r = growth.loc[book]
        if pd.isna(r.get("moc25_dv2_p95", float("nan"))):
            continue
        lines.append(f"| {book} | {pct(r['moc25_dv2_p95'], 2)} / {pct(r['moc25_dv2_p99'], 2)} / {int(r['moc25_dv2_orders_above_hard'])} | "
                     f"{pct(r['moc25_hpi_vote_p95'], 2)} / {pct(r['moc25_hpi_vote_p99'], 2)} / {int(r['moc25_hpi_vote_orders_above_hard'])} |")
    defensive = pd.read_csv(OUT / "defensive_status.csv", index_col=0)
    lines += ["", "## B defensive status", "",
              "| Book | CAGR | Sharpe | Max DD | DD incl. 2008 new / old | GFC new / old | Calmar incl. 2008 | Worst year |",
              "|---|---|---|---|---|---|---|---|"]
    for book, r in defensive.iterrows():
        lines.append(f"| {book} | {pct(r['cagr'])} | {r['sharpe']:.2f} | {pct(r['maxdd'])} | {pct(r['long_maxdd'])} / {pct(r['long_maxdd_old'])} | "
                     f"{pct(r['gfc'])} / {pct(r['gfc_old'])} | {r['long_calmar']:.2f} | {pct(r['worst_year'])} |")
    text = "\n".join(lines) + "\n"
    (OUT / "report_tables.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
