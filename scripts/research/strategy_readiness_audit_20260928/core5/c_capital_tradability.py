"""A9 capital scaling, A10 determinism, A7 padded/zero-volume bars, C1 order size vs ADV, C2 whole shares at USD 30K.

Runs the unchanged engine at USD 30K, 100K (twice), 1M and 10M on the same real Norgate frame.
"""
import hashlib
import json
import pickle

import numpy as np
import pandas as pd

import core5_common as c

core5 = c.core5
df = c.load_pricing()
with (c.CACHE / "baseline_strategy.pkl").open("rb") as fh:
    base = pickle.load(fh)

out = {}


def eq_hash(s):
    return hashlib.sha256(np.ascontiguousarray(s.results["total_value"].to_numpy(dtype=float)).tobytes()).hexdigest()


# ---- A10 determinism ----
s_again = c.run_backtest(df, 100_000.0)
base_hash = hashlib.sha256(np.ascontiguousarray(base["results"]["total_value"].to_numpy(dtype=float)).tobytes()).hexdigest()
out["A10_determinism"] = {"hash_run1": base_hash, "hash_run2": eq_hash(s_again), "identical": base_hash == eq_hash(s_again),
                          "fills_identical": bool(base["tx"][["bar", "asset", "amount", "price", "commission"]].astype(str).equals(
                              s_again.get_transactions()[["bar", "asset", "amount", "price", "commission"]].astype(str)))}
print(out["A10_determinism"], flush=True)

# ---- A9 capital scaling + C1 inputs ----
runs = {100_000.0: s_again}
for cap in (30_000.0, 1_000_000.0, 10_000_000.0):
    runs[cap] = c.run_backtest(df, cap)
base_ret = runs[100_000.0].results["daily_returns"].astype(float)
scal = {}
for cap, s in runs.items():
    r = s.results["daily_returns"].astype(float)
    tx = s.get_transactions()
    scal[str(int(cap))] = {
        "metrics_full": c.metrics(r), "metrics_2012": c.metrics(r, "2012-10-02"),
        "metrics_last3y": c.metrics(r, "2023-09-25"),
        "max_abs_daily_ret_diff_vs_100k": float((r - base_ret).abs().max()),
        "corr_vs_100k": float(np.corrcoef(r.fillna(0), base_ret.fillna(0))[0, 1]),
        "commission_total": float(tx["commission"].sum()),
        "commission_pct_of_capital_per_year": float(100 * tx["commission"].sum() / cap / 19.06),
        "n_fills": int(len(tx)),
        "n_min_fee_fills": int((tx["commission"] <= 1.0 + 1e-12).sum()),
        "negative_cash_min_weight": float(s._accounting_policy_dict.get("minimum_cash_weight_float", np.nan)),
        "borrow_total": float(s.borrow_fee_total_float),
    }
out["A9_capital_scaling"] = scal
print(json.dumps(scal, indent=1, default=str), flush=True)

# ---- A7 padded / zero-volume bars in the run window ----
cal = core5.build_execution_calendar_idx(df, core5.DEFAULT_CONFIG, core5.DEFAULT_CONFIG.backtest_start_date_str)
pad = {}
fills = runs[100_000.0].get_transactions()
fills["bar"] = pd.to_datetime(fills["bar"])
for sym in c.TRADEABLES:
    v = df[(sym, "Volume")].reindex(cal)
    zero = v.fillna(0.0) <= 0.0
    fill_days = set(fills.loc[fills["asset"] == sym, "bar"])
    dec_on_zero = set(v.index[zero])
    # decision on T uses Close_T; the fill is at T+1
    pos = cal.get_indexer(sorted(fill_days))
    decision_days = set(cal[pos - 1]) if len(pos) else set()
    pad[sym] = {"zero_or_nan_volume_sessions_in_run": int(zero.sum()), "dates": [str(d.date()) for d in v.index[zero][:10]],
                "fills_on_zero_volume_bar": int(len(fill_days & dec_on_zero)),
                "decisions_on_zero_volume_bar": int(len(decision_days & dec_on_zero))}
xnys = c.xnys_sessions("2007-01-01", "2026-09-25")
pad["calendar_rows_not_xnys"] = int(len(cal.difference(xnys)))
pad["xnys_sessions_missing_from_calendar"] = int(len(xnys[(xnys >= cal[0]) & (xnys <= cal[-1])].difference(cal)))
out["A7_padding"] = pad
print(pad, flush=True)

# ---- C1 order notional / ADV (20-session median dollar volume through the decision session) ----
dv = {}
for sym in c.TRADEABLES:
    turn = df[(sym, "Turnover")].astype(float)
    alt = df[(sym, "Unadjusted Close")].astype(float) * df[(sym, "Volume")].astype(float) * (
        df[(sym, "Close")].astype(float) / df[(sym, "Unadjusted Close")].astype(float))  # adj volume back to raw shares
    dollar = turn.where(turn > 0, alt)
    dv[sym] = dollar.rolling(20, min_periods=15).median()
adv_rows = []
for cap, s in runs.items():
    tx = s.get_transactions().copy()
    tx["bar"] = pd.to_datetime(tx["bar"])
    tx["notional"] = (tx["amount"].astype(float) * tx["price"].astype(float)).abs()
    pos = df.index.get_indexer(tx["bar"])
    tx["decision"] = df.index[pos - 1]
    tx["adv"] = [float(dv[a].loc[d]) for a, d in zip(tx["asset"], tx["decision"])]
    tx["pct_adv"] = 100 * tx["notional"] / tx["adv"]
    for window, sub in (("full", tx), ("last3y", tx[tx["bar"] >= pd.Timestamp("2023-09-25")])):
        for asset in list(c.TRADEABLES) + ["ALL"]:
            x = sub if asset == "ALL" else sub[sub["asset"] == asset]
            if len(x) == 0:
                continue
            adv_rows.append({"capital": int(cap), "window": window, "asset": asset, "n_orders": int(len(x)),
                             "median_pct_adv": float(x["pct_adv"].median()), "p99_pct_adv": float(x["pct_adv"].quantile(0.99)),
                             "max_pct_adv": float(x["pct_adv"].max()),
                             "median_notional": float(x["notional"].median()),
                             "p99_notional": float(x["notional"].quantile(0.99))})
adv_df = pd.DataFrame(adv_rows)
adv_df.to_csv(c.OUT / "c1_order_vs_adv.csv", index=False)
print(adv_df[adv_df["asset"] == "ALL"].to_string(), flush=True)
out["C1_adv_all_assets"] = adv_df[adv_df["asset"] == "ALL"].to_dict("records")
# current ADV and turnover coverage
out["C1_current_adv_usd_20d_median"] = {sym: float(dv[sym].iloc[-1]) for sym in c.TRADEABLES}
out["C1_turnover_field_coverage"] = {
    sym: {"first_turnover": str(df[(sym, "Turnover")].dropna().loc[lambda x: x > 0].index.min().date()),
          "n_turnover_zero_or_nan_in_run": int((df[(sym, "Turnover")].reindex(cal).fillna(0) <= 0).sum())}
    for sym in c.TRADEABLES}

# ---- C2 whole shares at USD 30K: held weight vs the engine's target weight on every day ----
s30 = runs[30_000.0]
tx30 = s30.get_transactions().copy()
tx30["bar"] = pd.to_datetime(tx30["bar"])
held = tx30.pivot_table(index="bar", columns="asset", values="amount", aggfunc="sum").reindex(cal).fillna(0.0).cumsum()
close = pd.DataFrame({a: df[(a, "Close")].astype(float) for a in c.TRADEABLES}).reindex(cal)
nav = s30.results["total_value"].astype(float).reindex(cal)
wt = held.reindex(columns=list(c.TRADEABLES)).fillna(0.0) * close / nav.to_numpy()[:, None]
tgt = s30.daily_target_weights.reindex(columns=list(c.TRADEABLES)).copy()
# target decided at Close_(T-1) is the book the engine holds from Open_T; align to the holding day
tgt.index = df.index[df.index.get_indexer(tgt.index) + 1]
tgt = tgt.reindex(cal).ffill()
err = (wt - tgt).abs()
# Day-after-rebalance error isolates rounding from drift
reb_exec_days = cal[cal.get_indexer(s30.rebalance_target_weight_df.index) + 1]
err_reb = err.reindex(reb_exec_days)
c2 = {}
for a in c.TRADEABLES:
    c2[a] = {"median_err_pct_nav_on_rebalance_fill_day": float(100 * err_reb[a].median()),
             "p99_err_pct_nav_on_rebalance_fill_day": float(100 * err_reb[a].quantile(0.99)),
             "max_err_pct_nav_on_rebalance_fill_day": float(100 * err_reb[a].max()),
             "last3y_max_err_pct_nav_on_rebalance_fill_day": float(100 * err_reb[a][err_reb.index >= "2023-09-25"].max()),
             "current_price_raw": float(df[(a, "Unadjusted Close")].iloc[-1]),
             "one_share_pct_of_30k": float(100 * df[(a, "Unadjusted Close")].iloc[-1] / 30_000.0)}
short_val = (held["DBC"].clip(upper=0.0) * close["DBC"]) if "DBC" in held else 0.0
cash_w = (s30.results["cash"].astype(float).reindex(cal) + short_val) / nav
c2["cash_weight_30k"] = {"median_pct": float(100 * cash_w.median()), "p99_pct": float(100 * cash_w.quantile(0.99)),
                         "min_pct": float(100 * cash_w.min())}
out["C2_whole_shares_30k"] = c2
print(json.dumps(c2, indent=1), flush=True)
c.dump(out, "c_capital_tradability.json")
