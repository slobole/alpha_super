"""Audit helper: collect the house Bench capacity summaries (capacity_v2_1) for the fund-product legs. Read-only on MAIN."""
import json
from pathlib import Path
import pandas as pd

MAIN = Path(r"C:\Users\User\Documents\workspace\alpha_super")
OUT = Path(__file__).resolve().parents[4] / "results" / "research" / "portfolio" / "fund_products_20261005" / "audit" / "capacity"
NAMES = ["strategy_mo_atr_normalized_ndx_vxn_scaled", "strategy_mo_atr_normalized_ndx_vxn_scaled_sector_cap",
         "strategy_mo_natr20_ndx_vxn_scaled_sector_cap", "strategy_mr_dv2", "strategy_mr_hpi_sp500_2_3_5_vote",
         "strategy_taa_adaptive_macro_core5", "strategy_taa_df_btal_1n_fallback_tqqq_vix_cash",
         "strategy_taa_df_btal_fallback_tqqq_vix_cash", "strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash",
         "strategy_mr_dv2_vix_gated_bil", "strategy_mr_hpi_vote_vix_gated_bil", "strategy_mr_dv2_liquidity_floor_adv_rank",
         "strategy_mr_dv2_liquidity_floor", "strategy_mr_dv2_industry_etf"]
rows = []
for n in NAMES:
    d = MAIN / "results" / "research" / "strategy" / n / "capacity_analysis"
    runs = sorted(p for p in d.glob("*") if (p / "summary.json").exists()) if d.exists() else []
    if not runs:
        rows.append({"strategy": n, "run": "NONE"})
        continue
    run = runs[-1]
    s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    m = json.loads((run / "metadata.json").read_text(encoding="utf-8")) if (run / "metadata.json").exists() else {}
    ws = s.get("window_summary_dict") or {"single": s}
    for w, x in ws.items():
        a = x.get("model_assumption_dict", {})
        rows.append({"strategy": n, "run": run.name, "model": m.get("model_version_str"), "window": w,
                     "start": x.get("actual_start_date_str"), "end": x.get("actual_end_date_str"),
                     "policy": x.get("execution_policy_str"), "profile": x.get("impact_profile_str"),
                     "recommended": x.get("recommended_capacity_float"), "rec_censored": x.get("recommended_capacity_censored_bool"),
                     "optimal": x.get("optimal_capacity_float"), "outer": x.get("outer_capacity_float"),
                     "outer_censored": x.get("outer_capacity_censored_bool"), "break_even": x.get("break_even_capacity_bracket_str"),
                     "orders": x.get("total_order_count_int"), "soft": a.get("soft_order_adv_limit_float"),
                     "hard": a.get("hard_order_adv_limit_float"), "lambda_bps": a.get("central_lambda_1pct_adv_bps_float"),
                     "grid_max": max(x.get("aum_grid_list") or [0]), "n_runs": len(runs)})
df = pd.DataFrame(rows)
OUT.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT / "bench_capacity_summaries.csv", index=False)
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 50)
print(df.to_string())
