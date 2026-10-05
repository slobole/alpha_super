"""Write the A6 bench portfolio files (unlevered products only; the bench cannot run margin).

Usage: python write_yamls_a6.py   Writes portfolios/fund_*.yaml in this worktree and prints what loads.
"""

from __future__ import annotations

import json
from pathlib import Path

import fp_lib as fp

REPO = Path(__file__).resolve().parents[3]
IMP = {"taa3x": ("pod_taa_btal_tqqq", "strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash"),
       "taa3x_1n": ("pod_taa_btal_1n_tqqq", "strategies.taa_df.strategy_taa_df_btal_1n_fallback_tqqq_vix_cash"),
       "ndx_vxn": ("pod_ndx_vxn", "strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled:VxnScaledAtrNormalizedNdxStrategy"),
       "core5": ("pod_core5", "strategies.taa_beyond_6040.strategy_taa_adaptive_macro_core5"),
       "btal_qqq": ("pod_taa_btal_lin_qqq", "strategies.taa_df.strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash"),
       "dv2": ("pod_dv2", "strategies.dv2.strategy_mr_dv2:DVO2Strategy"),
       "hpi_vote": ("pod_hpi_vote", "strategies.hpi.strategy_mr_hpi_sp500_2_3_5_vote"),
       "hpi_ibs_rsi": ("pod_hpi_ibs_rsi", "strategies.hpi.strategy_mr_hpi_sp500_ibs_rsi_exit"),
       "eom_flow": ("pod_eom_flow", "strategies.taa_beyond_6040.strategy_taa_month_end_rebalancing_flow"),
       "etf_dv2": ("pod_dv2_industry_etf", "strategies.dv2.strategy_mr_dv2_industry_etf"),
       "downshock": ("pod_downshock_vox_iyr", "strategies.mean_reversion.strategy_mr_us_sector_etf_ibs_downshock_vox_iyr"),
       "tbill": ("pod_tbill_bil", "strategies.portfolio_controls.strategy_passive_bil")}
FILES = {  # file -> (a6.json section, slot, one-line description)
    "fund_defensive": ("defensive", "launch", "DEFENSIVE, launch"),
    "fund_defensive_next": ("defensive", "next", "DEFENSIVE, next step (conditional: DV2-IND forward test and pre-2010 rerun)"),
    "fund_defensive_calm": ("defensive", "calm", "DEFENSIVE, very defensive"),
    "fund_defensive_plus": ("defensive", "rich", "DEFENSIVE, more return"),
    "fund_defensive_target": ("defensive", "target", "DEFENSIVE, ideal target (conditional: EOM forward test and a tradable route; DV2-IND forward test)"),
    "fund_defensive_ds": ("defensive", "ds_upgrade_10", "DEFENSIVE, launch upgrade once downshock has a live route: 10% downshock instead of the cash"),
    "fund_growth": ("growth", "launch", "GROWTH, launch (no leverage)"),
    "fund_growth_plus": ("growth", "plus", "GROWTH, more return without leverage"),
    "fund_growth_mr": ("growth", "mr", "GROWTH, stock mean-reversion target (gated: live slippage <= 3-4 bps/side; HPI-RSI demoted to PM_READY on 2026-09-30, needs re-wiring)"),
}
COMMON = """# Source: fund products study, amendments A6 / A6-c / A6-d (2026-10-01), scripts/research/fund_products_20260930/a6.py, a6d.py.
# Research configuration only: not an allocation approval and not wired to LIVE. Simulation, before fees.
# Starts 2012-10-02 (TQQQ/BTAL real history); the study's 2008 proxy is not used by the bench.
"""


def round_weights(w: dict) -> dict:
    """Largest-remainder rounding to 1/1000 so the file sums to exactly 1."""
    keys = sorted(w, key=lambda k: -w[k])
    raw = {k: w[k] * 1000 for k in keys}
    base = {k: int(raw[k]) for k in keys}
    left = 1000 - sum(base.values())
    for k in sorted(keys, key=lambda k: -(raw[k] - base[k]))[:left]:
        base[k] += 1
    return {k: base[k] / 1000 for k in keys if base[k] > 0}


def main() -> int:
    A = json.loads((fp.STUDY / "report" / "a6.json").read_text(encoding="utf-8"))
    A["defensive"] = json.loads((fp.STUDY / "report" / "a6d.json").read_text(encoding="utf-8"))["defensive"]   # A6-d slots
    written = []
    for name, (sec, slot, desc) in FILES.items():
        row = A[sec].get(slot)
        if not row:
            print("skip", name, "(slot empty)")
            continue
        assert float(row.get("lever", 1.0)) == 1.0, name
        w = round_weights({k: v for k, v in row["weights"].items()})
        q, t = row["q"], row["tails"]
        stats = (f"# Study (LONG 2008-03..2026-08, gross): {q['cagr'] * 100:.1f}%/yr, max DD {q['dd'] * 100:.1f}%, excess Sharpe {q['xs']:.2f}, "
                 f"P(DD < -10%) {t['p10'] * 100:.1f}%, P(DD < -20%) {t['p20'] * 100:.1f}%.\n")
        mix = " / ".join(f"{IMP[k][0].replace('pod_', '')} {v * 100:.1f}%" for k, v in w.items())
        pods = "".join(f"- pod_id_str: {IMP[k][0]}\n  strategy_import_str: {IMP[k][1]}\n  weight_float: {v}\n" for k, v in w.items())
        txt = (f"# {desc}: {mix}.\n{stats}{COMMON}name_str: {name}\ncapital_base_float: 1000000.0\nbacktest_start_date_str: '2012-10-02'\n"
               f"end_date_str: null\nallocation_policy_str: fixed\nmax_workers_int: null\nrebalance:\n  frequency_str: annually\n  policy_str: fixed\n"
               f"save_pod_artifacts_bool: true\nregression_benchmark_symbol_str: $SPX\npods:\n{pods}")
        (REPO / "portfolios" / f"{name}.yaml").write_text(txt, encoding="utf-8")
        written.append(name)
    import alpha.engine.portfolio_manager as pm
    for name in written:
        try:
            pm.load_portfolio_manager_config(REPO / "portfolios" / f"{name}.yaml")
            print("LOADS  ", name)
        except Exception as exc:  # report, do not hide
            print("BLOCKED", name, str(exc)[:90])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
