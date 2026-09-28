"""Review probe (live-parity lens): PREVIEW of the NDX ATR VXN ranking on the latest local bar
(2026-09-25), treated as if it were a month-end. Not the 2026-09-30 decision.

Purpose: list the current top candidates with their 20-day ATR as % of price and flag names whose
risk-adjusted score is driven by an unusually low ATR (typical of a pending cash takeover). Such a
name is eligible live at T (still a current member) but a later backtest drops it from T once its
removal falls within 5 member rows of T (see ndx_trim_live_divergence_probe.py). Direct local Norgate,
read-only. Usage: uv run python <this file>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
OUT = REPO / "results/research/strategy_readiness_audit_20260928/review_live"

import strategies.momentum.strategy_mo_atr_normalized_ndx as atr_module  # noqa: E402
import strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled as vxn_module  # noqa: E402

END = "2026-09-25"


def main():
    real_fn = atr_module.get_monthly_decision_close_df

    def keep_partial_fn(price_close_df):
        decision_date_ser = pd.Series(price_close_df.index, index=price_close_df.index.to_period("M")).groupby(level=0).max()
        idx = pd.DatetimeIndex(decision_date_ser.to_numpy(), name="decision_date_ts")
        out = price_close_df.loc[idx].copy()
        out.index = idx
        return out

    atr_module.get_monthly_decision_close_df = keep_partial_fn
    try:
        cfg = vxn_module.DEFAULT_CONFIG.__class__(**{**vxn_module.DEFAULT_CONFIG.__dict__, "end_date_str": END})
        pricing_df, universe_df, schedule_df, vxn_df = vxn_module.get_vxn_scaled_atr_normalized_ndx_data(cfg)
        strat = vxn_module.VxnScaledAtrNormalizedNdxStrategy(
            name="preview", benchmarks=["SPY"], rebalance_schedule_df=schedule_df, vxn_scale_signal_df=vxn_df,
        )
        strat.universe_df = universe_df
        sig = strat.compute_signals(pricing_df.copy())
    finally:
        atr_module.get_monthly_decision_close_df = real_fn
    T = pd.Timestamp(END)
    strat.previous_bar = T
    ranked = strat.get_ranked_candidate_feature_df(close_row_ser=sig.loc[T])
    weights = strat.get_target_weight_ser(close_row_ser=sig.loc[T])
    rows = []
    for rank, sym in enumerate(ranked.index[:15], start=1):
        close = float(sig.loc[T, (sym, "Close")])
        unadj = float(sig.loc[T, (sym, "Unadjusted Close")])
        atr_nominal = float(sig.loc[T, (sym, "atr_20_ser")])  # ATR rebased to Close_T nominal units
        rows.append({
            "rank": rank, "symbol": sym, "selected": sym in weights.index,
            "score": float(ranked.loc[sym, "risk_adj_score_float"]),
            "roc_12m": float(sig.loc[T, (sym, "monthly_roc_12_ser")]),
            "atr20_pct_of_price": atr_nominal / unadj if unadj > 0 else np.nan,
            "unadjusted_close": unadj, "adjusted_close": close,
        })
    df = pd.DataFrame(rows)
    med_atr_pct = float(df["atr20_pct_of_price"].median())
    df["low_atr_flag"] = df["atr20_pct_of_price"] < 0.4 * med_atr_pct
    df.to_csv(OUT / "ndx_vxn_preview_20260925.csv", index=False)
    print(df.to_string())
    print(json.dumps({"vxn_scale": float(vxn_module.get_asof_vxn_scale_float(vxn_df, T)),
                      "selected": sorted(weights.index.tolist())}))


if __name__ == "__main__":
    main()
