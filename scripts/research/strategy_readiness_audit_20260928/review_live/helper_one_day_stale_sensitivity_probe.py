"""Review probe (live-parity lens): how often would a ONE-DAY-STALE helper value change a live
decision? No freshness check exists outside CORE5 (A-LIVE-13): a vendor lag on the month-end night is
either padded forward (ALLMARKETDAYS) or dropped by an inner join / as-of lookup, silently.

- TAA 3x VRP gate: vrp_gate(T) vs vrp_gate(T-1) at every month-end T (SPY rv20 vs $VIX).
- NDX VXN scale: |scale(T) - scale(T-1)| at every month-end T.
Direct local Norgate, read-only. Study code only. Usage: uv run python <this file>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
OUT = REPO / "results/research/strategy_readiness_audit_20260928/review_live"
OUT.mkdir(parents=True, exist_ok=True)

import strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled as vxn_module  # noqa: E402
import strategies.taa_df.strategy_taa_df_fallback_vix_cash_variant_utils as vix_utils  # noqa: E402

END = "2026-09-25"


def _month_end_rows(daily_df: pd.DataFrame, start_str: str) -> pd.DataFrame:
    grp = pd.Series(daily_df.index, index=daily_df.index).groupby(daily_df.index.to_period("M"))
    rows = []
    for _, date_ser in grp:
        if len(date_ser) < 2:
            continue
        T, Tm1 = date_ser.iloc[-1], date_ser.iloc[-2]
        if T < pd.Timestamp(start_str):
            continue
        rows.append((T, Tm1))
    return pd.DataFrame(rows, columns=["T", "T_minus_1"])


def main():
    spy = vix_utils.load_helper_close_ser("SPY", "2008-01-01", END)
    vix = vix_utils.load_helper_close_ser("$VIX", "2008-01-01", END)
    vrp = vix_utils.compute_daily_vrp_signal_df(spy, vix)
    me = _month_end_rows(vrp, "2012-09-01")
    gate_T = vrp.loc[me["T"], "vrp_gate"].to_numpy()
    gate_Tm1 = vrp.loc[me["T_minus_1"], "vrp_gate"].to_numpy()
    # Stale VIX only (SPY fresh): rv20 at T vs VIX at T-1.
    gate_stale_vix = (vrp.loc[me["T"], "rv20_ann_pct"].to_numpy() < vrp.loc[me["T_minus_1"], "vix_close"].to_numpy())
    taa = {
        "month_ends": int(len(me)),
        "gate_flips_if_both_helpers_one_day_stale": int((gate_T != gate_Tm1).sum()),
        "gate_flips_if_only_vix_one_day_stale": int((gate_T.astype(bool) != gate_stale_vix).sum()),
        "flip_dates_vix_only": [d.date().isoformat() for d, f in zip(me["T"], gate_T.astype(bool) != gate_stale_vix) if f],
    }
    vxn = vxn_module.load_vxn_close_ser("$VXN", "2008-01-01", END)
    scale_df = vxn_module.compute_vxn_scale_signal_df(vxn)
    me2 = _month_end_rows(scale_df, "2012-09-01")
    d = (scale_df.loc[me2["T"], "vxn_exposure_scale_float"].to_numpy()
         - scale_df.loc[me2["T_minus_1"], "vxn_exposure_scale_float"].to_numpy())
    ndx = {
        "month_ends": int(len(me2)),
        "months_scale_changes_if_vxn_one_day_stale": int((abs(d) > 1e-12).sum()),
        "max_abs_scale_change": float(abs(d).max()),
        "p90_abs_scale_change": float(pd.Series(abs(d)).quantile(0.9)),
    }
    out = {"taa3x_vrp_gate": taa, "ndx_vxn_scale": ndx}
    print(json.dumps(out, indent=1))
    (OUT / "helper_one_day_stale_sensitivity.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
