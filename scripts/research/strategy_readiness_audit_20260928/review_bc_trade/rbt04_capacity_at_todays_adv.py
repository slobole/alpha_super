"""Review BC-trade RBT04: capacity "as if trading from today" (owner rule).

AM-01 judges C1 on the last 3 years, but each order is still divided by the ADV20 AT ITS OWN FILL DATE (2023-2026).
The owner asked for capacity at CURRENT volumes. Here every historical order fraction (last 3y, and separately the
full history) is divided by today's ADV of that symbol:
  ADV_today = median native Turnover over the 252 sessions ending 2026-09-25 (and, as a check, the 20 sessions ending
  2026-09-25).
Same house square-root model and guardrails as RBT02. Symbols without a current bar (delisted) keep their fill-date
ADV. Output: p99 % ADV at 1M, share of orders above the MOO hard guardrail at 1M, extra drag at 30K/100K/1M, and the
capacity where the central extra drag reaches 0.25 pp/yr and where p99 reaches 5% of ADV.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rbt02_uniform_capacity_and_friction as R  # noqa: E402

_ADV: dict[str, tuple[float, float]] = {}


def adv_today(asset: str) -> tuple[float, float]:
    if asset not in _ADV:
        try:
            df = R.load_price_timeseries(asset, start_date_str="2025-06-01", end_date_str=str(R.END.date()))
            t = df["Turnover"].astype(float)
            t = t[t > 0]
            if len(t) == 0 or t.index.max() < R.END - pd.Timedelta(days=7):
                raise ValueError("no current bar")
            _ADV[asset] = (float(t.tail(252).median()), float(t.tail(20).median()))
        except Exception:  # noqa: BLE001
            _ADV[asset] = (np.nan, np.nan)
    return _ADV[asset]


def block(w: pd.DataFrame, lc: float, ls: float, hard: float, slip: float, years: float) -> dict:
    def drag(C, lam):
        p = w["order_frac_of_nav"] * C / w["adv"]
        extra = np.maximum(slip, lam * np.sqrt(p / 0.01)) - slip
        return float((w["order_frac_of_nav"] * extra).sum() / 1e4 / years * 100.0)

    def solve(lam):
        for c in np.exp(np.linspace(np.log(1e3), np.log(1e9), 400)):
            if drag(c, lam) >= 0.25:
                return round(float(c), -2)
        return None

    p1m = w["order_frac_of_nav"] * 1e6 / w["adv"]
    return {
        "p99_pct_adv_1m": round(100 * float(p1m.quantile(0.99)), 3),
        "share_over_hard_guardrail_1m": round(float((p1m > hard).mean()), 3),
        "extra_drag_central_pp_yr": {f"C{int(C)}": round(drag(C, lc), 3) for C in (30e3, 100e3, 1e6, 10e6)},
        "capacity_usd_extra_drag_025pp": solve(lc),
        "capacity_usd_extra_drag_025pp_stress": solve(ls),
        "capacity_usd_p99_5pct_adv": round(0.05 / float((w["order_frac_of_nav"] / w["adv"]).quantile(0.99)), -3),
    }


def main() -> None:
    out = {"adv_today_definition": "median native Turnover, 252 sessions to 2026-09-25", "pods": {}}
    for key, (path, fmt, (lc, ls, hard), slip, _fee, _c) in R.PODS.items():
        if key == "ndx_vxn":
            continue
        df = R.load(path, fmt)
        df = df[df["bar"] <= R.END].copy()
        if len(df) == 0:
            continue
        a252 = {a: adv_today(a)[0] for a in df["asset"].unique()}
        a20 = {a: adv_today(a)[1] for a in df["asset"].unique()}
        res = {"adv_today_252_usd": {a: round(v, 0) for a, v in a252.items()},
               "adv_today_20_usd": {a: round(v, 0) for a, v in a20.items()}}
        full_years = (R.END - df["bar"].min()).days / 365.25
        for name, sub, yrs in (("last3y_orders", df[df["bar"] >= R.L3Y], R.YEARS), ("full_history_orders", df, full_years)):
            if len(sub) == 0:
                res[name] = None
                continue
            for adv_name, amap in (("adv252", a252), ("adv20", a20)):
                w = sub.copy()
                w["adv"] = w["asset"].map(amap)
                w["adv"] = w["adv"].fillna(w["adv20_usd"])
                res[f"{name}__{adv_name}"] = block(w, lc, ls, hard, slip, yrs)
        out["pods"][key] = res
        print(key, json.dumps({k: v for k, v in res.items() if "__" in k and "adv252" in k}, default=float)[:700], flush=True)
    (R.OUT / "rbt04_capacity_at_todays_adv.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
