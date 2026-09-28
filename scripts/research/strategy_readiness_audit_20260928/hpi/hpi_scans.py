"""A7/A8/A9/A10, B4 (gate, missing opens) and C1/C2/C4 scans from the full-history runs (hpi_full_runs.py).

Usage: uv run python hpi_scans.py <variant>
Writes results/research/strategy_readiness_audit_20260928/hpi/scans_<variant>.json (+ csv details).
"""

from __future__ import annotations

import pickle
import re
import sys

import numpy as np
import pandas as pd

import hpi_common as hc
from strategies.hpi import stateful_long as hpi_mod

LAST3_START = pd.Timestamp("2023-09-25")
CAPITALS = {"30k": 30_000.0, "1m": 1_000_000.0, "10m": 10_000_000.0}


def arm_dir(variant, arm):
    return hc.OUT / "full_runs" / f"{variant}_{arm}"


def load_arm(variant, arm):
    d = arm_dir(variant, arm)
    if not (d / "daily.csv.gz").exists():
        return None
    daily = pd.read_csv(d / "daily.csv.gz", index_col=0, parse_dates=True)
    tx = pd.read_csv(d / "transactions.csv.gz", parse_dates=["bar"])
    return {"daily": daily, "tx": tx, "stdout": (d / "stdout.txt").read_text(encoding="utf-8"),
            "summary": pd.read_json(d / "summary.json", typ="series").to_dict()}


def pct(x, q):
    return float(np.nanpercentile(x, q)) if len(x) else float("nan")


def main(variant: str) -> None:
    data = hc.load_full_inputs()
    pricing, universe = data["pricing_df"], data["universe"]
    base = load_arm(variant, "base")
    with (arm_dir(variant, "base") / "records.pkl").open("rb") as handle:
        records = pickle.load(handle)
    out: dict = {"variant": variant, "base_metrics": {k: base["summary"][k] for k in
                                                      ("cagr", "sharpe", "max_dd", "vol", "n_fills",
                                                       "commission_total", "dividend_net_total",
                                                       "dividend_gross_total", "final_total_value")}}
    daily, tx = base["daily"], base["tx"]
    nav = daily["total_value"]

    # ---------------------------------------------------------------- A10 determinism, A9 scaling, A8 hsu
    rep = load_arm(variant, "base_repeat")
    if rep is not None:
        out["A10_determinism"] = {
            "bit_identical_total_value": bool(np.array_equal(rep["daily"]["total_value"].to_numpy(),
                                                             nav.to_numpy())),
            "bit_identical_transactions": bool(rep["tx"].equals(tx))}
    ret = nav.pct_change().dropna()
    scale = {}
    for arm in ("cap30k", "cap1m", "cap10m", "hsu", "cap30k_hsu"):
        a = load_arm(variant, arm)
        if a is None:
            continue
        r2 = a["daily"]["total_value"].pct_change().dropna()
        m = hc.metrics(a["daily"]["total_value"])
        scale[arm] = {**{k: m[k] for k in ("cagr", "sharpe", "max_dd")},
                      "d_cagr_pp_vs_base": 100 * (m["cagr"] - base["summary"]["cagr"]),
                      "d_sharpe_vs_base": m["sharpe"] - base["summary"]["sharpe"],
                      "daily_ret_corr": float(ret.corr(r2)),
                      "max_abs_daily_ret_diff": float((ret - r2).abs().max()),
                      "commission_total": float(a["summary"]["commission_total"]),
                      "commission_bp_of_nav_per_yr": None,
                      "n_fills": int(a["summary"]["n_fills"])}
    out["A9_A8_arms"] = scale

    # ---------------------------------------------------------------- A8 negative cash and financing
    cash_frac = daily["cash"] / nav
    dtb3 = pd.read_csv(hc.REPO / "results/research/strategy_readiness_audit_20260928/taa/DTB3_audit_cache.csv",
                       index_col=0, parse_dates=True)["DTB3"].reindex(nav.index).ffill() / 100.0
    debit_rate = dtb3 + 0.015  # IBKR Pro benchmark + 1.5% (tier < USD 100k); bound, not a quote
    credit_rate = (dtb3 - 0.005).clip(lower=0.0)
    debit_cost = (-cash_frac.clip(upper=0.0)) * debit_rate / 252.0
    credit_income = cash_frac.clip(lower=0.0) * credit_rate / 252.0
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    adj_nav_debit = nav.iloc[0] * (1 + ret - debit_cost.reindex(ret.index)).cumprod()
    adj_nav_both = nav.iloc[0] * (1 + ret - debit_cost.reindex(ret.index) + credit_income.reindex(ret.index)).cumprod()
    m_debit, m_both = hc.metrics(adj_nav_debit), hc.metrics(adj_nav_both)
    out["A8_cash"] = {
        "engine_policy": base["summary"]["accounting_policy"],
        "min_cash_pct_nav": float(100 * cash_frac.min()), "min_cash_date": str(cash_frac.idxmin().date()),
        "pct_days_negative": float(100 * (cash_frac < 0).mean()),
        "pct_days_below_minus1pct": float(100 * (cash_frac < -0.01).mean()),
        "pct_days_below_minus2pct": float(100 * (cash_frac < -0.02).mean()),
        "mean_cash_pct_nav": float(100 * cash_frac.mean()),
        "mean_negative_cash_pct_nav_when_negative": float(100 * cash_frac[cash_frac < 0].mean()),
        "p1_cash_pct_nav": pct(100 * cash_frac, 1),
        "debit_financing_cost_bp_per_yr_(DTB3+1.5%)": float(1e4 * debit_cost.sum() / years),
        "credit_income_bp_per_yr_(DTB3-0.5%)": float(1e4 * credit_income.sum() / years),
        "cagr_pp_if_debit_charged": 100 * (m_debit["cagr"] - base["summary"]["cagr"]),
        "sharpe_delta_if_debit_charged": m_debit["sharpe"] - base["summary"]["sharpe"],
        "cagr_pp_if_debit_charged_and_credit_paid": 100 * (m_both["cagr"] - base["summary"]["cagr"]),
    }

    # ---------------------------------------------------------------- refill funding need (C4)
    rec_df = pd.DataFrame(records)
    tx["notional"] = tx["amount"].astype(float) * tx["price"].astype(float)
    buys = tx[tx["amount"] > 0].groupby("bar")["notional"].sum()
    sells = -tx[tx["amount"] < 0].groupby("bar")["notional"].sum()
    rec_df = rec_df.set_index("exec_date")
    rec_df["buy_notional"] = buys.reindex(rec_df.index).fillna(0.0)
    rec_df["sell_notional"] = sells.reindex(rec_df.index).fillna(0.0)
    rec_df["n_held"] = rec_df["positions"].map(len)
    rec_df["n_exits"] = rec_df["exits"].map(len)
    rec_df["n_entries"] = rec_df["entries"].map(len)
    rec_df["refill"] = (rec_df["n_exits"] > 0) & (rec_df["n_entries"] > 0) & (rec_df["n_held"] + rec_df["n_entries"] > 10)
    # buying power beyond settled cash needed at submission (sales have not executed yet)
    rec_df["need_pct_nav"] = 100 * (rec_df["buy_notional"] - rec_df["cash_before"].clip(lower=0)).clip(lower=0) / rec_df["prev_total_value"]
    rec_df["gross_buy_pct_nav"] = 100 * rec_df["buy_notional"] / rec_df["prev_total_value"]
    rf = rec_df[rec_df["refill"]]
    out["C4_refill_funding"] = {
        "n_refill_days": int(len(rf)), "n_refill_entries": int((rf["n_held"] + rf["n_entries"] - 10).clip(lower=0).sum()),
        "need_beyond_cash_pct_nav_median": pct(rf["need_pct_nav"], 50),
        "need_p95": pct(rf["need_pct_nav"], 95), "need_p99": pct(rf["need_pct_nav"], 99),
        "need_max": float(rf["need_pct_nav"].max()) if len(rf) else 0.0,
        "need_max_date": str(rf["need_pct_nav"].idxmax().date()) if len(rf) else None,
        "all_days_need_max": float(rec_df["need_pct_nav"].max()),
        "gross_buy_pct_nav_p99_all_days": pct(rec_df.loc[rec_df["buy_notional"] > 0, "gross_buy_pct_nav"], 99),
        "max_entries_one_day": int(rec_df["n_entries"].max()),
    }

    # ---------------------------------------------------------------- missing opens / cancellations / liquidations
    cancel = re.findall(r"Asset (\S+) has no tradable open on (\S+)", base["stdout"])
    removed = re.findall(r"Removed asset (\S+) has no open on (\S+) .*?from (\S+)\.", base["stdout"])
    held_missing = rec_df[rec_df["missing_open_held"].map(len) > 0]
    pend_missing = [(str(d.date()), s) for d, r in rec_df.iterrows() for s in r["missing_open_held"]
                    if s in set(r["pending_after"])]
    out["B4_missing_open"] = {
        "held_name_days_with_missing_open_T1": int(held_missing["missing_open_held"].map(len).sum()),
        "examples_held_missing": [(str(d.date()), r["missing_open_held"]) for d, r in held_missing.head(10).iterrows()],
        "pending_exit_with_missing_open_T1": pend_missing,
        "backtest_order_cancellations_missing_open": [(s, d) for s, d in cancel],
        "removed_asset_liquidations_at_last_close": removed,
    }

    # ---------------------------------------------------------------- A6 membership exits
    mem_exit = 0
    for r in records:
        members = None
        for s in r["exits"]:
            if members is None:
                members = hpi_mod.get_asof_universe_symbol_set(universe, r["signal_date"])
            if s not in members:
                mem_exit += 1
    out["A6_membership_exits"] = int(mem_exit)

    # ---------------------------------------------------------------- A7 zero-volume / padded fills
    vol = pricing.xs("Volume", axis=1, level=1)
    hi = pricing.xs("High", axis=1, level=1)
    fill_vol = [vol.at[b, a] if a in vol.columns else np.nan for a, b in zip(tx["asset"], tx["bar"])]
    fill_bar = [pd.notna(hi.at[b, a]) if a in hi.columns else False for a, b in zip(tx["asset"], tx["bar"])]
    tx["fill_volume"] = fill_vol
    tx["fill_has_bar"] = fill_bar
    liq = tx["order_id"] == -1
    out["A7"] = {"fills_on_zero_volume_bar": int((tx["fill_volume"] == 0).sum()),
                 "fills_without_bar_non_liquidation": int((~tx["fill_has_bar"] & ~liq).sum()),
                 "synthetic_liquidation_fills": int(liq.sum())}

    # ---------------------------------------------------------------- C1 order size vs 20-session median Turnover
    turnover = pricing.xs("Turnover", axis=1, level=1)
    tx = tx[tx["asset"].isin(turnover.columns)].copy()
    prev_nav = nav.shift(1)
    tx["frac_nav"] = tx["notional"].abs() / prev_nav.reindex(tx["bar"]).to_numpy()
    sessions = pricing.index
    tx["signal_date"] = [sessions[sessions.get_loc(b) - 1] for b in tx["bar"]]
    adv_cache = {}
    adv_list = []
    for a, sd in zip(tx["asset"], tx["signal_date"]):
        if a not in adv_cache:
            adv_cache[a] = turnover[a].dropna().rolling(20, min_periods=20).median()
        ser = adv_cache[a]
        pos = ser.index.searchsorted(sd, side="right") - 1
        adv_list.append(float(ser.iloc[pos]) if pos >= 0 else np.nan)
    tx["adv20_turnover"] = adv_list
    c1 = {}
    for label, cap in CAPITALS.items():
        ratio = 100 * tx["frac_nav"] * cap / tx["adv20_turnover"]
        last3 = ratio[tx["bar"] >= LAST3_START]
        c1[label] = {"median_pct_adv": pct(ratio.dropna(), 50), "p99_pct_adv": pct(ratio.dropna(), 99),
                     "max_pct_adv": float(ratio.max()), "share_orders_over_1pct": float((ratio > 1).mean()),
                     "share_orders_over_5pct": float((ratio > 5).mean()),
                     "last3y_median_pct_adv": pct(last3.dropna(), 50), "last3y_p99_pct_adv": pct(last3.dropna(), 99),
                     "last3y_max_pct_adv": float(last3.max()) if len(last3) else float("nan")}
    c1["n_orders"] = int(len(tx))
    c1["n_orders_missing_adv"] = int(tx["adv20_turnover"].isna().sum())
    c1["min_adv20_usd_last3y"] = float(tx.loc[tx["bar"] >= LAST3_START, "adv20_turnover"].min())
    c1["median_adv20_usd_last3y"] = float(tx.loc[tx["bar"] >= LAST3_START, "adv20_turnover"].median())
    c1["assumption"] = "order notional = (backtest order notional / previous NAV) x capital, i.e. NAV held at the stated capital"
    out["C1_adv"] = c1
    tx[["bar", "asset", "amount", "price", "notional", "frac_nav", "adv20_turnover"]].to_csv(
        hc.OUT / f"c1_orders_{variant}.csv.gz", index=False)

    # ---------------------------------------------------------------- C2 whole shares at USD 30K (raw prices)
    raw = pricing.xs("Unadjusted Close", axis=1, level=1)
    ent = pd.DataFrame([{"signal_date": r["signal_date"], "asset": s} for r in records for s in r["entries"]])
    ent["raw_close_T"] = [raw.at[d, a] if a in raw.columns else np.nan for d, a in zip(ent["signal_date"], ent["asset"])]
    c2 = {}
    for label, cap in (("30k", 30_000.0), ("100k", 100_000.0)):
        slot = cap / 10.0
        shares = np.floor(slot / ent["raw_close_T"])
        err = 100 * (slot - shares * ent["raw_close_T"]) / cap
        for tag, mask in (("all", ent["signal_date"] >= pd.Timestamp("2004-01-01")),
                          ("last3y", ent["signal_date"] >= LAST3_START)):
            e, sh = err[mask], shares[mask]
            c2[f"{label}_{tag}"] = {
                "n_entries": int(mask.sum()), "zero_share_entries": int((sh == 0).sum()),
                "zero_share_pct": float(100 * (sh == 0).mean()),
                "weight_err_pct_nav_median": pct(e, 50), "weight_err_p99": pct(e, 99),
                "entries_err_over_2pct_nav": int((e > 2).sum()), "entries_err_over_5pct_nav": int((e > 5).sum()),
                "zero_share_names": sorted(set(ent.loc[mask & (shares == 0), "asset"]))[:30],
                "one_share_names": sorted(set(ent.loc[mask & (shares == 1), "asset"]))[:30],
            }
    out["C2_whole_shares"] = c2

    # ---------------------------------------------------------------- B4 readiness gate over history
    s = hc.make_strategy(variant, universe)
    sig = s.compute_signals(pricing.copy())
    req = ["return_3d_ser", "hpi_value_ser", "ibs_value_ser", "rsi2_value_ser", "sma_200_price_ser", "Turnover"]
    if variant == "vote":
        req += [hpi_mod.RETURN_2D_FIELD_STR, hpi_mod.RETURN_5D_FIELD_STR, hpi_mod.HPI_2D_FIELD_STR,
                hpi_mod.HPI_5D_FIELD_STR]
    uni = universe.reindex(sig.index).fillna(0).astype(bool)
    ready = None
    for f in req:
        fr = sig.xs(f, axis=1, level=1).reindex(columns=uni.columns)
        ok = np.isfinite(fr.to_numpy(dtype=float))
        ready = ok if ready is None else (ready & ok)
    members = uni.to_numpy()
    n_mem = members.sum(axis=1)
    n_ready = (ready & members).sum(axis=1)
    need = np.maximum(400, np.ceil(0.8 * n_mem))
    gate = pd.DataFrame({"members": n_mem, "ready": n_ready, "need": need}, index=sig.index)
    gate = gate.loc["2004-01-01":]
    fail = gate[gate["ready"] < gate["need"]]
    traded_dates = {r["signal_date"] for r in records if r["exits"] or r["entries"]}
    out["B4_readiness_gate"] = {
        "n_sessions": int(len(gate)), "n_fail": int(len(fail)),
        "fail_first": str(fail.index[0].date()) if len(fail) else None,
        "fail_last": str(fail.index[-1].date()) if len(fail) else None,
        "fail_since_2006_10": int((fail.index >= "2006-10-02").sum()),
        "fail_days_with_backtest_orders": int(sum(d in traded_dates for d in fail.index)),
        "ready_ratio_min_since_2010": float((gate["ready"] / gate["members"]).loc["2010":].min()),
        "latest": {"date": str(gate.index[-1].date()), **{k: int(v) for k, v in gate.iloc[-1].items()}},
        "fail_by_year": {str(k): int(v) for k, v in fail.groupby(fail.index.year).size().items()},
    }
    gate.to_csv(hc.OUT / f"gate_{variant}.csv.gz")
    hc.dump_json(out, f"scans_{variant}.json")
    print(pd.Series(out).to_string()[:6000])


if __name__ == "__main__":
    main(sys.argv[1])
