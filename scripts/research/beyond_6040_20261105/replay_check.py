"""Replay check: house strategy_taa_beyond_6040_trend_tilt vs the fast research engine's T1 series.

Research-only. Run from the repo root:
    .venv\\Scripts\\python.exe scripts\\research\\beyond_6040_20261105\\replay_check.py

House: VTI/TLT/GLD CAPITALSPECIAL prices, Open_(T+1) fills, per-share commission + 1 bp slippage,
idle cash at 0%, dividends credited as cash. Research: SPY/TLT/GLD total return, Close-to-Close with a
one-session lag, 5 bp cost per unit traded, idle cash at Fed funds. Differences are expected to be small.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from strategies.taa_beyond_6040.strategy_taa_beyond_6040_trend_tilt import run_variant  # noqa: E402

OUT_DIR = HERE / "out" / "replay"
VAULT_DIR = HERE / "out" / "vault"
RESEARCH_COL = "T1 (chosen)"
ASSET_MAP = {"VTI": "r_stock", "TLT": "r_bond", "GLD": "r_gold"}
TRADING_DAYS = 252
PASS_CORR = 0.97
PASS_CAGR_GAP = 0.01


def stats(ret: pd.Series) -> dict:
    r = ret.dropna()
    nav = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    dd = nav / nav.cummax() - 1
    return {
        "cagr": nav.iloc[-1] ** (1 / years) - 1,
        "vol": r.std() * np.sqrt(TRADING_DAYS),
        "sharpe_raw": r.mean() / r.std() * np.sqrt(TRADING_DAYS),
        "max_dd": dd.min(),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    research_mtime = pd.Timestamp((VAULT_DIR / "series_daily_full.parquet").stat().st_mtime, unit="s")

    strategy = run_variant(show_display_bool=False, save_results_bool=False)
    house_ret = strategy.results["total_value"].astype(float).pct_change().dropna()
    house_ret.index = pd.to_datetime(house_ret.index)
    house_w = strategy.daily_target_weights.copy()
    house_w.index = pd.to_datetime(house_w.index)
    house_gross = house_w[list(ASSET_MAP)].sum(axis=1)

    research_ret = pd.read_parquet(VAULT_DIR / "series_daily_full.parquet")[RESEARCH_COL]
    research_w = pd.read_parquet(VAULT_DIR / "weights_T1.parquet")
    research_gross = pd.read_parquet(VAULT_DIR / "exposure_daily_full.parquet")[RESEARCH_COL]
    panel = pd.read_parquet(HERE / "out" / "panel.parquet")

    common = house_ret.index.intersection(research_ret.index)
    common = common[(common >= house_ret.index.min()) & (common <= min(house_ret.index.max(), research_ret.index.max()))]
    h = house_ret.loc[common]
    r = research_ret.loc[common]

    # --- headline
    hs, rs = stats(h), stats(r)
    corr = float(h.corr(r))
    diff = h - r
    te = float(diff.std() * np.sqrt(TRADING_DAYS))
    rows = []
    for k in ("cagr", "vol", "sharpe_raw", "max_dd"):
        rows.append({"metric": k, "house": hs[k], "research_T1": rs[k], "difference": hs[k] - rs[k]})
    rows.append({"metric": "corr_daily_returns", "house": corr, "research_T1": np.nan, "difference": np.nan})
    rows.append({"metric": "tracking_error_ann", "house": te, "research_T1": np.nan, "difference": np.nan})
    rows.append({"metric": "corr_gross_exposure", "house": float(house_gross.reindex(common).corr(research_gross.reindex(common))),
                 "research_T1": np.nan, "difference": np.nan})
    rows.append({"metric": "avg_gross_exposure", "house": float(house_gross.reindex(common).mean()),
                 "research_T1": float(research_gross.reindex(common).mean()),
                 "difference": float(house_gross.reindex(common).mean() - research_gross.reindex(common).mean())})
    rows.append({"metric": "n_days", "house": len(common), "research_T1": len(common), "difference": 0})
    headline = pd.DataFrame(rows).set_index("metric")

    # --- per year
    hy = (1 + h).groupby(h.index.year).prod() - 1
    ry = (1 + r).groupby(r.index.year).prod() - 1
    by_year = pd.DataFrame({"house": hy, "research_T1": ry, "difference": hy - ry,
                            "corr_daily": h.groupby(h.index.year).apply(lambda x: x.corr(r.loc[x.index]))})
    by_year.index.name = "year"

    # --- tilt state agreement on the first session of each month (weights held that day)
    first_days = pd.Series(common, index=common).groupby(common.to_period("M")).min()
    agree_rows = []
    for d in first_days:
        row = {"date": d}
        for house_sym, res_sym in ASSET_MAP.items():
            hw = house_w.loc[d, house_sym] if d in house_w.index else np.nan
            rw = research_w.loc[d, res_sym] if d in research_w.index else np.nan
            row[f"house_{house_sym}"] = hw
            row[f"research_{res_sym}"] = rw
            row[f"agree_{house_sym}"] = (hw > 1e-9) == (rw > 1e-9) if np.isfinite(hw) and np.isfinite(rw) else np.nan
        agree_rows.append(row)
    agree = pd.DataFrame(agree_rows).set_index("date")
    agree_cols = [f"agree_{s}" for s in ASSET_MAP]
    agree["agree_all"] = agree[agree_cols].all(axis=1)
    agree_share = {c: float(agree[c].mean()) for c in agree_cols + ["agree_all"]}
    disagree_months = agree.index[~agree["agree_all"]].strftime("%Y-%m").tolist()

    # --- diagnostics for the biggest source of difference
    spy_vs_vti = np.nan
    try:
        from strategies.taa_beyond_6040.strategy_taa_beyond_6040 import get_beyond_6040_data  # noqa: E402
        px = get_beyond_6040_data()
        vti = px[("VTI", "Close")].astype(float).pct_change().reindex(common)
        spy_vs_vti = float(vti.corr(panel["r_stock"].reindex(common)))
    except Exception:  # pragma: no cover - diagnostics only
        pass
    # Decompose the CAGR gap (house minus research), all in return points per year.
    years = len(common) / TRADING_DAYS
    cagr_gap = hs["cagr"] - rs["cagr"]
    ff_daily = panel["dff"].reindex(common).ffill() / TRADING_DAYS
    cash_yield_research = float(((1.0 - research_gross.reindex(common)) * ff_daily).sum() / years)
    nav = strategy.results["total_value"].astype(float)
    nav.index = pd.to_datetime(nav.index)
    mean_nav = float(nav.reindex(common).mean())
    gross_div = float(getattr(strategy, "dividend_cash_gross_total_float", 0.0))
    net_div = float(getattr(strategy, "dividend_cash_net_total_float", 0.0))
    withholding_drag = (gross_div - net_div) / mean_nav / years
    tx = strategy.get_transactions()
    commission_drag = float(tx["commission"].sum()) / mean_nav / years if len(tx) else 0.0
    monthly_diff = (1 + diff).groupby(diff.index.to_period("M")).prod() - (1 + 0 * diff).groupby(diff.index.to_period("M")).prod()
    hm = (1 + h).groupby(h.index.to_period("M")).prod() - 1
    rm = (1 + r).groupby(r.index.to_period("M")).prod() - 1
    tilt_months = [pd.Period(m, "M") for m in disagree_months]
    tilt_disagree_contrib = float((hm - rm).reindex(tilt_months).sum() / years) if tilt_months else 0.0
    explained = -cash_yield_research - withholding_drag - commission_drag + tilt_disagree_contrib
    residual = cagr_gap - explained
    diag = pd.DataFrame({
        "metric": ["cagr_gap_house_minus_research", "research_idle_cash_at_fed_funds", "house_dividend_withholding_25pct",
                   "house_commissions", "tilt_disagreement_months", "residual_timing_vti_vs_spy_other",
                   "dividend_ledger_status", "corr_VTI_vs_research_stock_leg", "months_tilt_disagree", "months_total",
                   "mean_abs_gross_exposure_gap", "research_parquet_mtime"],
        "value": [cagr_gap, -cash_yield_research, -withholding_drag, -commission_drag, tilt_disagree_contrib, residual,
                  str(strategy._accounting_policy_dict.get("dividend_data_status_str")), spy_vs_vti, len(disagree_months), len(agree),
                  float((house_gross.reindex(common) - research_gross.reindex(common)).abs().mean()), str(research_mtime)],
    }).set_index("metric")

    # --- save
    table = pd.concat([
        headline.assign(section="headline"),
        by_year.rename(columns={"corr_daily": "difference_corr_daily"}).assign(section="by_year").rename_axis("metric"),
        pd.DataFrame({"house": agree_share}).assign(section="tilt_agreement").rename_axis("metric"),
        diag.rename(columns={"value": "house"}).assign(section="diagnostics"),
    ])
    table.to_csv(OUT_DIR / "replay_check.csv")
    agree.to_csv(OUT_DIR / "tilt_agreement_by_month.csv")
    pd.DataFrame({"house": h, "research_T1": r}).to_csv(OUT_DIR / "daily_returns_common.csv")

    verdict = "PASS" if (corr >= PASS_CORR and abs(cagr_gap) <= PASS_CAGR_GAP) else "FAIL"
    worst_year = by_year["difference"].abs().idxmax()
    lines = [
        f"# Replay check: house trend-tilt strategy vs research T1 ({verdict})",
        "",
        f"Common window {common.min().date()} -> {common.max().date()}, {len(common)} sessions; "
        f"research parquet written {research_mtime:%Y-%m-%d %H:%M} UTC.",
        f"Correlation of daily returns {corr:.3f} (pass >= {PASS_CORR}); tracking error {te:.2%}/yr; exposure correlation "
        f"{headline.loc['corr_gross_exposure','house']:.3f}, average gross {headline.loc['avg_gross_exposure','house']:.3f} vs "
        f"{headline.loc['avg_gross_exposure','research_T1']:.3f}.",
        f"CAGR house {hs['cagr']:.2%} vs research {rs['cagr']:.2%} (gap {cagr_gap:+.2%}, pass within {PASS_CAGR_GAP:.0%}); "
        f"vol {hs['vol']:.2%} vs {rs['vol']:.2%}; max DD {hs['max_dd']:.2%} vs {rs['max_dd']:.2%}.",
        f"Tilt state agreement at month starts: all three assets {agree_share['agree_all']:.1%} "
        f"(" + ", ".join(f"{s} {agree_share[f'agree_{s}']:.1%}" for s in ASSET_MAP) + f"); "
        f"months with any disagreement ({len(disagree_months)}): {', '.join(disagree_months[:8])}{' ...' if len(disagree_months) > 8 else ''}.",
        f"Largest yearly gap {worst_year} ({by_year.loc[worst_year,'difference']:+.2%}); other years within "
        f"{by_year['difference'].drop(worst_year).abs().max():.2%}.",
        f"Gap decomposition (points/yr): idle cash at Fed funds in research, 0% in house {-cash_yield_research:+.2%}; "
        f"house 25% dividend withholding (research is gross total return) {-withholding_drag:+.2%}; commissions at the $100K default base {-commission_drag:+.2%} (near zero at fund scale); "
        f"tilt-disagreement months {tilt_disagree_contrib:+.2%}; residual (Open_(T+1) vs close fills, VTI vs SPY, "
        f"price-vs-total-return vol, 63- vs 40-session overlay warm-up) {residual:+.2%}.",
        f"Biggest source: {'idle-cash yield convention' if cash_yield_research >= max(withholding_drag, abs(residual)) else ('dividend withholding' if withholding_drag >= abs(residual) else 'timing/instrument residual')}. "
        f"Stock leg VTI vs research stock series daily correlation {spy_vs_vti:.3f}; dividend ledger {strategy._accounting_policy_dict.get('dividend_data_status_str')}.",
        "Files: replay_check.csv (headline, by year, tilt agreement, diagnostics), tilt_agreement_by_month.csv, daily_returns_common.csv.",
    ]
    (OUT_DIR / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print()
    print(headline.round(4).to_string())
    print()
    print(by_year.round(4).to_string())


if __name__ == "__main__":
    main()
