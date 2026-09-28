"""Tier B macro tradability (C1/C2 + small-account friction) for Compass, Compass QQQ and Tactical FI.

C1: every backtest fill's notional as a share of NAV_(t-1), rescaled to a constant pod size C in {30K, 1M, 10M},
    against the 20-session median of native Norgate Turnover (common/tradability.py). Last 3 years govern (AM-01);
    full history is a note. MOO guardrails 0.05% / 0.10% of ADV (alpha/engine/capacity_analysis.py:91-92).
C2: whole shares at USD 10K / 15K / 30K on nominal (Unadjusted) prices at each decision.
Friction at USD 30K: IBKR Fixed (USD 0.005/share, USD 1 minimum, 1% cap) and an approximate Tiered minimum
    (USD 0.35) per fill, as bp of NAV per year; the backtests charge 0 commission.

Outputs: OUT/tradability_<key>/..., OUT/tradability_tierb_macro.json
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import tb_common as tb

sys.path.insert(0, str(tb.HERE.parent / "common"))
from tradability import (  # noqa: E402
    load_turnover_ser,
    participation_table_df,
    summarize_participation_df,
    whole_share_table_df,
)
from data.norgate_loader import load_price_timeseries  # noqa: E402
import strategies.taa_beyond_6040.strategy_taa_tactical_fixed_income_ief_lqd as tfi  # noqa: E402

LAST3Y_START = "2023-09-25"
_UNADJ: dict[str, pd.Series] = {}


def unadj_close(asset: str, date_str: str) -> float:
    if asset not in _UNADJ:
        df = load_price_timeseries(asset, start_date_str="1998-01-01", end_date_str=tb.NORGATE_LAST_BAR_STR)
        _UNADJ[asset] = df["Unadjusted Close"].astype(float).dropna()
    s = _UNADJ[asset].loc[: pd.Timestamp(date_str)]
    return float(s.iloc[-1]) if len(s) else np.nan


def friction(part_df: pd.DataFrame, strategy_obj, capital: float, start: str | None = None) -> dict:
    tx = strategy_obj.get_transactions().copy()
    tx["bar"] = pd.to_datetime(tx["bar"])
    if start:
        tx = tx[tx["bar"] >= pd.Timestamp(start)]
        part_df = part_df[part_df["bar"] >= pd.Timestamp(start)]
    fixed, tiered = [], []
    for _, r in part_df.iterrows():
        notional = float(r["order_frac_of_nav"]) * capital
        px = unadj_close(str(r["asset"]), str(pd.Timestamp(r["bar"]).date()))
        shares = notional / px if px > 0 else 0.0
        if shares < 1:  # whole-share rounding would place no order
            continue
        fixed.append(min(max(1.0, 0.005 * shares), 0.01 * notional))
        tiered.append(min(max(0.35, 0.0035 * shares), 0.01 * notional))
    bars = pd.to_datetime(part_df["bar"])
    years = max((bars.max() - bars.min()).days / 365.25, 1e-9) if len(bars) else np.nan
    return {
        "fills": int(len(part_df)),
        "fills_per_year": float(len(part_df) / years),
        "ibkr_fixed_bp_per_year": float(sum(fixed) / years / capital * 1e4),
        "ibkr_tiered_min_bp_per_year": float(sum(tiered) / years / capital * 1e4),
    }


def moo_guardrail(part_df: pd.DataFrame) -> dict:
    out = {}
    for window, df in (("full", part_df), ("last3y", part_df[part_df["bar"] >= pd.Timestamp(LAST3Y_START)])):
        for cap in (30_000, 1_000_000, 10_000_000):
            col = df[f"part_{cap}"].replace([np.inf], np.nan)
            out[f"{window}_{cap}"] = {
                "share_orders_over_0p05pct_adv": float((col > 0.0005).mean()),
                "share_orders_over_0p10pct_adv": float((col > 0.0010).mean()),
                "max_pct_adv": float(col.max() * 100),
            }
    return out


def adv_table(assets: list[str]) -> dict:
    out = {}
    for a in assets:
        t = load_turnover_ser(a, end_date_str=tb.NORGATE_LAST_BAR_STR)
        out[a] = {
            "median_daily_turnover_last3y_usd": float(t[t.index >= pd.Timestamp(LAST3Y_START)].median()),
            "median_daily_turnover_full_usd": float(t.median()),
            "min_20d_median_last3y_usd": float(t.rolling(20).median()[t.index >= pd.Timestamp(LAST3Y_START)].min()),
            "moo_0p10pct_of_last3y_adv_usd": float(t[t.index >= pd.Timestamp(LAST3Y_START)].median() * 0.001),
            "first_date": str(t.index[0].date()),
        }
    return out


def analyze(key: str, strategy_obj, weight_by_decision: dict) -> dict:
    assets = sorted(set(strategy_obj.get_transactions()["asset"].astype(str)))
    turn = {a: load_turnover_ser(a, end_date_str=tb.NORGATE_LAST_BAR_STR) for a in assets}
    part = participation_table_df(strategy_obj, turn)
    d = tb.OUT / f"tradability_{key}"
    d.mkdir(exist_ok=True)
    part.to_csv(d / "participation_by_fill.csv", index=False)
    summ = summarize_participation_df(part, recent_start_str=LAST3Y_START)
    summ.to_csv(d / "participation_summary.csv", index=False)
    ws = whole_share_table_df(weight_by_decision, unadj_close)
    ws.to_csv(d / "whole_share.csv", index=False)
    ws_sum = ws.groupby("capital_usd").agg(
        max_name_weight_error=("weight_error", "max"),
        p95_name_weight_error=("weight_error", lambda s: float(s.quantile(0.95))),
        zero_share=("zero_share_bool", "sum"), n=("asset", "count"),
    ).reset_index()
    per_dec = ws.groupby(["capital_usd", "decision_date"])["weight_error"].sum().reset_index()
    ws_sum["mean_rounding_cash"] = per_dec.groupby("capital_usd")["weight_error"].mean().values
    ws_sum["max_rounding_cash"] = per_dec.groupby("capital_usd")["weight_error"].max().values
    out = {
        "assets": assets,
        "adv": adv_table(assets),
        "participation": summ.to_dict(orient="records"),
        "moo_guardrail": moo_guardrail(part),
        "whole_share": ws_sum.to_dict(orient="records"),
        "friction_30k_full": friction(part, strategy_obj, 30_000.0),
        "friction_30k_last3y": friction(part, strategy_obj, 30_000.0, LAST3Y_START),
        "friction_12k_last3y": friction(part, strategy_obj, 12_000.0, LAST3Y_START),
    }
    return out


def main() -> None:
    res = {}
    for variant in ("xlk", "qqq"):
        s = tb.run_compass_engine(variant, end_date_str=tb.NORGATE_LAST_BAR_STR)
        wdict = {}
        for rb, w in s.rebalance_weight_df.iterrows():
            dec = pd.Timestamp(rb) - pd.offsets.BDay(1)
            if dec >= pd.Timestamp("2016-01-01"):
                wdict[str(dec.date())] = {str(k): float(v) for k, v in w.items() if float(v) > 1e-12}
        res[f"compass_{variant}"] = analyze(f"compass_{variant}", s, wdict)
        print(variant, res[f"compass_{variant}"]["participation"][4:6], flush=True)
    s = tfi.run_variant(show_display_bool=False, save_results_bool=False)
    wdict = {}
    for rb, row in s.month_end_weight_df.iterrows():
        dec = pd.Timestamp(row["decision_date"])
        if dec >= pd.Timestamp("2016-01-01"):
            wdict[str(dec.date())] = {a: float(row[a]) for a in ("IEF", "LQD") if float(row[a]) > 1e-12}
    res["tactical_fi"] = analyze("tactical_fi", s, wdict)
    tb.write_json("tradability_tierb_macro.json", res)


if __name__ == "__main__":
    main()
