"""Run the two new strategy modules through the real engine and save path + transactions."""
import json, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUT = REPO / "results/research/dv2_deep_20260925/wired_check"
which = sys.argv[1]
t0 = time.time()
if which == "adv":
    from strategies.dv2.strategy_mr_dv2_liquidity_floor_adv_rank import run_variant
    st = run_variant(show_display_bool=False, save_results_bool=False, backtest_start_date_str="2000-01-03", capital_base_float=1_000_000.0, end_date_str="2026-08-19")
else:
    from strategies.dv2.strategy_mr_dv2_industry_etf import run_variant
    st = run_variant(show_display_bool=False, save_results_bool=False, capital_base_float=1_000_000.0, end_date_str="2026-08-19")
st.results[["total_value"]].to_csv(OUT / f"{which}__path.csv")
st.get_transactions().to_csv(OUT / f"{which}__transactions.csv", index=False)
(OUT / f"{which}__meta.json").write_text(json.dumps({"runtime_s": round(time.time() - t0, 1), "fills": int(len(st.get_transactions()))}))
print(which, "done", round(time.time() - t0, 1))
