"""Optional B (SPEC_FROZEN.md): the stress gate x limit entry on Scout's DV2 S&P 500 limit book, 2004-2022 (sealed window).

Grid: entry {market-on-open in/out, limit k 0.5 in / limit out} x gate {off, on} x cost {gross, AR, pooled}.
The gate deletes a session's candidate list when the gate is closed at that close (exits unchanged).
Scout's book keeps idle cash at 0 and pays no dividends; an approximate T-bill sweep (idle share = 1 - positions / 10,
previous session) is reported next to it, because the gated book holds far more cash.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent / "scout_dv2_limit_entry_20261002"))
sys.path.insert(0, str(HERE))
import run_limit as rl  # noqa: E402
from limit_book import entry_limit_mat, exit_limit_mat, limit_book, trade_through_margin_mat  # noqa: E402
import components as cp  # noqa: E402

OUT = cp.OUT


def gated_mats(mats: dict, gate: np.ndarray) -> dict:
    ptr, cand = mats["pointer_vec"], mats["candidate_vec"]
    new_ptr, new = [0], []
    for t in range(len(ptr) - 1):
        if gate[t]:
            new.extend(cand[ptr[t]:ptr[t + 1]].tolist())
        new_ptr.append(len(new))
    out = dict(mats)
    out["pointer_vec"] = np.asarray(new_ptr, dtype=mats["pointer_vec"].dtype)
    out["candidate_vec"] = np.asarray(new, dtype=mats["candidate_vec"].dtype)
    return out


def idle_share(result, dates: pd.DatetimeIndex) -> pd.Series:
    log = result.log_df
    delta = pd.Series(np.where(log["kind_int"] == 1, 1, np.where(log["kind_int"].isin([-1, -2]), -1, 0)), index=log["date"])
    held = delta.groupby(level=0).sum().reindex(dates, fill_value=0).cumsum()
    return (1.0 - held / 10.0).clip(lower=0).shift(1).fillna(1.0)


def stats(r: pd.Series) -> dict:
    r = r.loc[rl.START_STR:rl.SEAL_END_STR]
    nav = (1 + r).cumprod()
    return {"sharpe": float(r.mean() / r.std() * np.sqrt(252)), "cagr": float(nav.iloc[-1] ** (252 / len(r)) - 1),
            "max_dd": float((nav / nav.cummax() - 1).min()),
            "eras": {e: float(r.loc[a:b].mean() / r.loc[a:b].std() * np.sqrt(252)) for e, a, b in rl.ERA_TUPLE}}


def main():
    superset = rl.load_superset_panel(rl.SUPERSET_NAME_STR, index_name_list=["S&P 500"])
    panel = rl.build_panel(superset, "S&P 500")
    inputs = rl.signal_inputs(panel)
    spread = rl.spread_dict_for(superset, panel)
    mats = inputs["mats"]
    dates = panel.date_index
    gate = cp.gate_on(pd.DatetimeIndex(dates))
    margin = trade_through_margin_mat(inputs["unadjusted"], [spread["ar"], spread["pooled"]])
    exit_lim = exit_limit_mat(mats["close"], inputs["unadjusted"])
    slip = {"ar": rl.fill_slippage_mat(spread["ar"], rl.FLOOR_SLIP_FLOAT), "pooled": rl.fill_slippage_mat(spread["pooled"], rl.FLOOR_SLIP_FLOAT)}
    rate = cp.rs.cash_rate(pd.DatetimeIndex(dates))
    rep = {"gate_open_share_2004_2022": float(gate[(dates >= rl.START_STR) & (dates <= rl.SEAL_END_STR)].mean())}
    daily = {}
    for gate_on in (False, True):
        m = gated_mats(mats, gate) if gate_on else mats
        for label, k, exit_str in (("moo", None, "moo"), ("limit_k0.5", 0.5, "limit")):
            elim = None if k is None else entry_limit_mat(mats["close"], inputs["unadjusted"], inputs["natr"], k)
            for cost in ("gross", "ar", "pooled"):
                s_, fee, mn, capf = (0.0, 0.0, 0.0, 0.0) if cost == "gross" else (slip[cost], rl.FEE_PER_SHARE_FLOAT, rl.MIN_FEE_FLOAT, rl.FEE_CAP_FLOAT)
                # same call as run_limit.run_pods: (..., entry_limit, exit_str, margin, exit_limit, slippage, share_scale, fee, min, cap)
                res = limit_book(dates, panel.symbol_list, m, inputs["high"], inputs["low"], rl.CONFIG.max_positions_int, rl.START_STR,
                                 elim, exit_str, margin, exit_lim, s_, inputs["scale"], fee, mn, capf)
                r0 = res.daily_ser.fillna(0.0)
                idle = idle_share(res, pd.DatetimeIndex(dates))
                key = f"{'gate' if gate_on else 'nogate'}|{label}|{cost}"
                daily[key] = (r0 + idle * rate.reindex(r0.index).fillna(0.0)).loc[rl.START_STR:rl.SEAL_END_STR]
                rep[key] = {"cash_zero": stats(r0), "cash_tbills_approx": stats(r0 + idle * rate.reindex(r0.index).fillna(0.0)),
                            "orders": rl.order_stats(res, dates)}
                print(key, "Sharpe cash0 %.2f  T-bills %.2f  CAGR %.3f  fill %.2f  trades/yr %.0f" % (
                    rep[key]["cash_zero"]["sharpe"], rep[key]["cash_tbills_approx"]["sharpe"], rep[key]["cash_tbills_approx"]["cagr"],
                    rep[key]["orders"]["fill_rate_float"], rep[key]["orders"]["trades_per_year_float"]), flush=True)
    pd.DataFrame(daily).to_parquet(OUT / "limit_axis_daily.parquet")
    (OUT / "limit_axis.json").write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
