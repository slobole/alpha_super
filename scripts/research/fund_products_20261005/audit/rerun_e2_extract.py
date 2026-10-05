"""Audit scratch (fund products 2026-10-05, task rerun_e2): compare the fresh E2 PortfolioManager run at HEAD with the
stored pickle, and write tidy inputs for the fund-product study.

Inputs
- NEW   : newest run under WT/results/research/portfolio/ndx_e2_sector_cap_5050/vanilla_backtest/ (run at HEAD).
- STORED: MAIN/results/research/portfolio/ndx_e2_sector_cap_5050/vanilla_backtest/2026-10-04_095907 (read only),
          the pickle read by scripts/research/mr_capsule_build_20261004/page_v2_data.py::load_e2_ret_ser.
- Optional: the $100M copy of the book (pm_100m) and the live NDX VXN rule alone (ndx_vxn_live_daily.csv).

Outputs (WT/results/research/portfolio/fund_products_20261005/audit/rerun_e2/)
- e2_book_daily.csv, e2_transactions.csv, e2_stats.json, e2_window_stats.csv, e2_positions_daily.csv

Conventions
- Return on date T = NAV_T / NAV_{T-1} - 1 (close to close). Window stats use the returns dated inside the window.
- CAGR = wealth ** (252 / n) - 1, vol = std(ddof=1) * sqrt(252), Sharpe = mean / std * sqrt(252) with rf = 0,
  Max DD on the in-window wealth path that starts at 1.0 before the first return.
- The book is the PortfolioManager aggregation: two pods run standalone ($500K each), the book re-weights their daily
  returns back to 50/50 on the first trading day of each month. No share-level trade and no cost is booked for that
  reset. Pod fills are therefore scaled into book terms by (book capital of the pod at the start of the day) /
  (standalone pod NAV at the prior close).
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
MAIN = Path(r"C:/Users/User/Documents/workspace/alpha_super")
OUT = REPO / "results/research/portfolio/fund_products_20261005/audit/rerun_e2"
STORED_PKL = MAIN / "results/research/portfolio/ndx_e2_sector_cap_5050/vanilla_backtest/2026-10-04_095907/ndx_e2_sector_cap_5050.pkl"
NEW_ROOT = REPO / "results/research/portfolio/ndx_e2_sector_cap_5050/vanilla_backtest"
BIG_ROOT = OUT / "pm_100m/research/portfolio/ndx_e2_sector_cap_5050_100m/vanilla_backtest"
LIVE_CSV = OUT / "ndx_vxn_live_daily.csv"

POD_SHORT = {
    "strategy_mo_atr_normalized_ndx_vxn_scaled_sector_cap": "atr_cap",
    "strategy_mo_natr20_ndx_vxn_scaled_sector_cap": "natr_cap",
}
WINDOWS = {
    "full 2000-01-03..latest": ("2000-01-03", None),
    "2008-03-04..2026-08-19": ("2008-03-04", "2026-08-19"),
    "2012-10-02..2026-08-19": ("2012-10-02", "2026-08-19"),
    "2026-08-20..latest": ("2026-08-20", None),
}


def newest_pickle(root: Path, name: str) -> Path:
    run_dirs = sorted(d for d in root.iterdir() if d.is_dir() and (d / f"{name}.pkl").exists())
    if not run_dirs:
        raise FileNotFoundError(f"No finished run under {root}")
    return run_dirs[-1] / f"{name}.pkl"


def load_port(path: Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def loader_ret(port) -> pd.Series:
    """Exactly page_v2_data.load_e2_ret_ser on a Portfolio object."""
    total_value = port.results["total_value"].astype(float)
    total_value.index = pd.to_datetime(total_value.index)
    return total_value.pct_change().dropna()


def stats(ret: pd.Series) -> dict:
    ret = ret.dropna()
    n = len(ret)
    if n < 2:
        return {"n_days": n}
    wealth = np.concatenate([[1.0], (1.0 + ret).cumprod().to_numpy()])
    return {
        "n_days": n,
        "first": ret.index[0].date().isoformat(),
        "last": ret.index[-1].date().isoformat(),
        "cagr": float(wealth[-1] ** (252.0 / n) - 1.0),
        "vol": float(ret.std(ddof=1) * np.sqrt(252.0)),
        "sharpe": float(ret.mean() / ret.std(ddof=1) * np.sqrt(252.0)),
        "max_dd": float((wealth / np.maximum.accumulate(wealth) - 1.0).min()),
        "total_return": float(wealth[-1] - 1.0),
    }


def window(ser: pd.Series | pd.DataFrame, a: str, b: str | None):
    return ser.loc[a:b] if b is not None else ser.loc[a:]


def book_frames(port) -> dict:
    """Daily book path, pod returns, cash and turnover in book terms, and the scaled fills."""
    nav = port.results["total_value"].astype(float)
    nav.index = pd.to_datetime(nav.index)
    idx = nav.index
    ret = nav.pct_change().fillna(0.0)
    nav_prev = nav.shift(1)

    daily = pd.DataFrame({"ret": ret, "nav": nav}, index=idx)
    cash_usd = pd.Series(0.0, index=idx)
    turnover_usd = pd.Series(0.0, index=idx)
    commission_usd = pd.Series(0.0, index=idx)
    tx_frames, pos_count = [], {}
    for strategy in port.strategies:
        short = POD_SHORT[strategy.name]
        res = strategy.results.loc[idx]
        pod_nav = res["total_value"].astype(float)
        pod_ret = port._daily_rets[strategy.name].astype(float)
        pod_book = port._pod_equities[strategy.name].astype(float)  # book capital of the pod at the close
        daily[f"{short}_ret"] = pod_ret
        daily[f"{short}_book_weight"] = pod_book / nav
        daily[f"{short}_cash_frac_of_pod"] = res["cash"].astype(float) / pod_nav
        cash_usd += pod_book * res["cash"].astype(float) / pod_nav

        # *** scale standalone fills into book dollars: capital the book gives the pod at the open of day T,
        # divided by the standalone pod NAV at the prior close (both known before the fills of day T).
        pod_book_start = pod_book / (1.0 + pod_ret)
        scale = (pod_book_start / pod_nav.shift(1)).rename("scale")
        tx = strategy._transactions.copy()
        tx["bar"] = pd.to_datetime(tx["bar"])
        tx["signed_notional_pod_usd"] = tx["amount"].astype(float) * tx["price"].astype(float)
        tx["scale_to_book"] = tx["bar"].map(scale)
        tx["signed_notional_book_usd"] = tx["signed_notional_pod_usd"] * tx["scale_to_book"]
        tx["book_nav_prev_usd"] = tx["bar"].map(nav_prev)
        tx["signed_notional_frac_nav"] = tx["signed_notional_book_usd"] / tx["book_nav_prev_usd"]
        tx["pod_nav_prev_usd"] = tx["bar"].map(pod_nav.shift(1))
        tx["commission_book_usd"] = tx["commission"].astype(float) * tx["scale_to_book"]
        tx["pod"] = short
        tx_frames.append(tx)
        turnover_usd = turnover_usd.add(tx.groupby("bar")["signed_notional_book_usd"].apply(lambda s: s.abs().sum()), fill_value=0.0)
        commission_usd = commission_usd.add(tx.groupby("bar")["commission_book_usd"].sum(), fill_value=0.0)

        weight_df = strategy.realized_weight_df.copy()
        weight_df.index = pd.to_datetime(weight_df.index)
        asset_weight_df = weight_df.drop(columns=["Cash"]).reindex(idx)
        pos_count[short] = (asset_weight_df.fillna(0.0).abs() > 1e-9).sum(axis=1)
        pos_count[f"{short}_held"] = asset_weight_df

    daily["cash_frac"] = cash_usd / nav
    daily["turnover_frac"] = (turnover_usd.reindex(idx).fillna(0.0) / nav_prev).fillna(0.0)
    daily["commission_frac"] = (commission_usd.reindex(idx).fillna(0.0) / nav_prev).fillna(0.0)
    # Inter-pod reset on book rebalance dates: capital moved between the pods (one way), never booked as a trade.
    drift_prev = port.drift_weight_df[next(iter(POD_SHORT))].shift(1)
    reset = pd.Series(0.0, index=idx)
    rebalance_idx = pd.DatetimeIndex(port._rebalance_date_index)
    reset.loc[rebalance_idx] = (drift_prev.loc[rebalance_idx] - 0.5).abs()
    daily["pod_reset_one_way_frac"] = reset
    daily["is_book_rebalance_day"] = idx.isin(rebalance_idx)

    held_atr = (pos_count["atr_cap_held"].fillna(0.0).abs() > 1e-9)
    held_natr = (pos_count["natr_cap_held"].fillna(0.0).abs() > 1e-9)
    all_symbols = sorted(set(held_atr.columns) | set(held_natr.columns))
    union = held_atr.reindex(columns=all_symbols, fill_value=False) | held_natr.reindex(columns=all_symbols, fill_value=False)
    both = held_atr.reindex(columns=all_symbols, fill_value=False) & held_natr.reindex(columns=all_symbols, fill_value=False)
    positions = pd.DataFrame({
        "atr_cap_positions": pos_count["atr_cap"], "natr_cap_positions": pos_count["natr_cap"],
        "distinct_names_book": union.sum(axis=1), "names_in_both_pods": both.sum(axis=1),
    }, index=idx)
    # Largest single-name weight in the book (both pods combined, book weights).
    w_atr = pos_count["atr_cap_held"].fillna(0.0).reindex(columns=all_symbols, fill_value=0.0).mul(daily["atr_cap_book_weight"], axis=0)
    w_natr = pos_count["natr_cap_held"].fillna(0.0).reindex(columns=all_symbols, fill_value=0.0).mul(daily["natr_cap_book_weight"], axis=0)
    positions["max_single_name_weight_book"] = (w_atr + w_natr).max(axis=1)

    tx_all = pd.concat(tx_frames, ignore_index=True).sort_values(["bar", "pod", "order_id"]).reset_index(drop=True)
    return {"daily": daily, "tx": tx_all, "positions": positions}


def compare_runs(new_port, stored_port) -> dict:
    new_ret, old_ret = loader_ret(new_port), loader_ret(stored_port)
    out = {
        "new_range": [new_ret.index[0].date().isoformat(), new_ret.index[-1].date().isoformat(), int(len(new_ret))],
        "stored_range": [old_ret.index[0].date().isoformat(), old_ret.index[-1].date().isoformat(), int(len(old_ret))],
        "same_index": bool(new_ret.index.equals(old_ret.index)),
    }
    common = new_ret.index.intersection(old_ret.index)
    diff = (new_ret.loc[common] - old_ret.loc[common]).abs()
    out.update({
        "common_days": int(len(common)),
        "max_abs_daily_diff": float(diff.max()),
        "max_abs_daily_diff_date": diff.idxmax().date().isoformat(),
        "days_diff_gt_1e-12": int((diff > 1e-12).sum()),
        "days_diff_gt_1e-6": int((diff > 1e-6).sum()),
        "first_diff_date_gt_1e-12": (diff[diff > 1e-12].index[0].date().isoformat() if (diff > 1e-12).any() else None),
        "correlation": float(np.corrcoef(new_ret.loc[common], old_ret.loc[common])[0, 1]),
        "final_nav_new": float(new_port.results["total_value"].iloc[-1]),
        "final_nav_stored": float(stored_port.results["total_value"].iloc[-1]),
    })
    out["final_nav_ratio_new_over_stored"] = out["final_nav_new"] / out["final_nav_stored"]
    pods = {}
    for new_s, old_s in zip(new_port.strategies, stored_port.strategies):
        a, b = new_s.results["total_value"].astype(float), old_s.results["total_value"].astype(float)
        c = a.index.intersection(b.index)
        rel = (a.loc[c] / b.loc[c] - 1.0).abs()
        pods[POD_SHORT[new_s.name]] = {
            "max_abs_rel_nav_diff": float(rel.max()),
            "first_nav_diff_date_gt_1e-12": (rel[rel > 1e-12].index[0].date().isoformat() if (rel > 1e-12).any() else None),
            "final_nav_new": float(a.iloc[-1]), "final_nav_stored": float(b.iloc[-1]),
            "transactions_new": int(len(new_s._transactions)), "transactions_stored": int(len(old_s._transactions)),
        }
    out["pods"] = pods
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    new_pkl = newest_pickle(NEW_ROOT, "ndx_e2_sector_cap_5050")
    new_port, stored_port = load_port(new_pkl), load_port(STORED_PKL)
    result: dict = {"new_pickle": str(new_pkl), "stored_pickle": str(STORED_PKL)}
    result["compare_new_vs_stored"] = compare_runs(new_port, stored_port)

    frames = book_frames(new_port)
    daily, tx, positions = frames["daily"], frames["tx"], frames["positions"]
    daily_out = daily[["ret", "nav", "cash_frac", "turnover_frac", "atr_cap_ret", "natr_cap_ret", "atr_cap_book_weight",
                       "natr_cap_book_weight", "atr_cap_cash_frac_of_pod", "natr_cap_cash_frac_of_pod", "commission_frac",
                       "pod_reset_one_way_frac", "is_book_rebalance_day"]]
    daily_out.to_csv(OUT / "e2_book_daily.csv", index_label="date", float_format="%.10g")
    tx_out = tx.rename(columns={"bar": "date", "asset": "symbol", "amount": "shares_ledger_units", "price": "fill_price"})[
        ["date", "symbol", "signed_notional_book_usd", "pod", "signed_notional_frac_nav", "signed_notional_pod_usd",
         "shares_ledger_units", "fill_price", "commission", "scale_to_book", "pod_nav_prev_usd", "book_nav_prev_usd"]]
    tx_out.to_csv(OUT / "e2_transactions.csv", index=False, float_format="%.10g")
    positions.to_csv(OUT / "e2_positions_daily.csv", index_label="date", float_format="%.6g")

    # ---- window stats: book, pods (standalone daily returns), pod correlation
    live_ret = None
    if LIVE_CSV.exists():
        live_nav = pd.read_csv(LIVE_CSV, index_col="date", parse_dates=True)["total_value"].astype(float)
        live_ret = live_nav.pct_change().dropna()
    big_daily = None
    big_pickles = sorted(BIG_ROOT.glob("*/ndx_e2_sector_cap_5050_100m.pkl")) if BIG_ROOT.exists() else []
    if big_pickles:
        big_port = load_port(big_pickles[-1])
        big_daily = book_frames(big_port)["daily"]
        big_daily[["ret", "nav", "cash_frac", "turnover_frac", "atr_cap_ret", "natr_cap_ret", "commission_frac"]].to_csv(
            OUT / "e2_book_daily_100m.csv", index_label="date", float_format="%.10g")
        result["big_pickle"] = str(big_pickles[-1])

    book_ret = daily["ret"].iloc[1:]
    rows = []
    for label, (a, b) in WINDOWS.items():
        series = {"E2 book ($1M)": book_ret, "atr_cap pod (standalone)": daily["atr_cap_ret"].iloc[1:],
                  "natr_cap pod (standalone)": daily["natr_cap_ret"].iloc[1:]}
        if big_daily is not None:
            series["E2 book ($100M)"] = big_daily["ret"].iloc[1:]
        if live_ret is not None:
            series["live NDX VXN rule alone ($1M)"] = live_ret
        for name, ser in series.items():
            rows.append({"window": label, "series": name, **stats(window(ser, a, b))})
        pod_df = window(daily[["atr_cap_ret", "natr_cap_ret"]].iloc[1:], a, b)
        row = {"window": label, "series": "corr(atr_cap, natr_cap) daily", "n_days": len(pod_df),
               "corr": float(pod_df.corr().iloc[0, 1])}
        monthly = (1.0 + pod_df).resample("ME").prod() - 1.0
        row["corr_monthly"] = float(monthly.corr().iloc[0, 1]) if len(monthly) > 2 else None
        rows.append(row)
        if live_ret is not None:
            both = pd.concat([window(book_ret, a, b), window(live_ret, a, b)], axis=1, join="inner").dropna()
            rows.append({"window": label, "series": "corr(E2 book, live NDX VXN rule) daily", "n_days": len(both),
                         "corr": float(both.corr().iloc[0, 1])})
    window_df = pd.DataFrame(rows)
    window_df.to_csv(OUT / "e2_window_stats.csv", index=False, float_format="%.6g")

    # ---- implementation facts measured on the $1M run
    years_float = len(daily) / 252.0
    cal_years_float = (daily.index[-1] - daily.index[0]).days / 365.25
    trade_days = daily.index[daily["turnover_frac"] > 0]
    invested_mask = daily["cash_frac"] < 0.99
    facts = {
        "start": daily.index[0].date().isoformat(), "end": daily.index[-1].date().isoformat(), "n_days": int(len(daily)),
        "first_fill_date": tx["bar"].min().date().isoformat(), "last_fill_date": tx["bar"].max().date().isoformat(),
        "cash_frac_mean": float(daily["cash_frac"].mean()), "cash_frac_median": float(daily["cash_frac"].median()),
        "cash_frac_p10": float(daily["cash_frac"].quantile(0.10)), "cash_frac_p90": float(daily["cash_frac"].quantile(0.90)),
        "cash_frac_min": float(daily["cash_frac"].min()), "cash_frac_max": float(daily["cash_frac"].max()),
        "days_cash_ge_99pct_share": float((daily["cash_frac"] >= 0.99).mean()),
        "days_cash_negative_share": float((daily["cash_frac"] < 0).mean()),
        "cash_frac_mean_when_invested": float(daily.loc[invested_mask, "cash_frac"].mean()),
        "cash_frac_median_when_invested": float(daily.loc[invested_mask, "cash_frac"].median()),
        "annual_turnover_two_sided_x_nav": float(daily["turnover_frac"].sum() / years_float),
        "annual_commission_frac_nav": float(daily["commission_frac"].sum() / years_float),
        "annual_slippage_frac_nav_at_2p5bps": float(daily["turnover_frac"].sum() / years_float * 0.00025),
        "trading_days_total": int(len(trade_days)), "trading_days_per_year": float(len(trade_days) / years_float),
        "years_252": years_float, "calendar_years": cal_years_float,
        "fills_total": int(len(tx)), "fills_per_trading_day_median": float(tx.groupby("bar").size().median()),
        "turnover_frac_on_trade_days_median": float(daily.loc[trade_days, "turnover_frac"].median()),
        "turnover_frac_on_trade_days_mean": float(daily.loc[trade_days, "turnover_frac"].mean()),
        "turnover_frac_on_trade_days_max": float(daily.loc[trade_days, "turnover_frac"].max()),
        "book_rebalance_dates": int(len(new_port._rebalance_date_index)),
        "fill_days_not_on_book_rebalance_day": int((~pd.DatetimeIndex(trade_days).isin(new_port._rebalance_date_index)).sum()),
        "pod_reset_one_way_frac_mean_per_rebalance": float(daily.loc[daily["is_book_rebalance_day"], "pod_reset_one_way_frac"].mean()),
        "pod_reset_one_way_frac_annual": float(daily["pod_reset_one_way_frac"].sum() / years_float),
        "pod_book_weight_min_max_atr": [float(daily["atr_cap_book_weight"].min()), float(daily["atr_cap_book_weight"].max())],
        "positions_when_invested": {
            col: {"median": float(positions.loc[invested_mask, col].median()), "mean": float(positions.loc[invested_mask, col].mean()),
                  "min": float(positions.loc[invested_mask, col].min()), "max": float(positions.loc[invested_mask, col].max())}
            for col in positions.columns},
        "standalone_pod_final_nav": {POD_SHORT[s.name]: float(s.results["total_value"].iloc[-1]) for s in new_port.strategies},
        "accounting_policy": {POD_SHORT[s.name]: {k: v for k, v in s._accounting_policy_dict.items()} for s in new_port.strategies},
        "cost_model": {POD_SHORT[s.name]: {"slippage": s._slippage, "commission_per_share": s._commission_per_share,
                                           "commission_minimum": s._commission_minimum} for s in new_port.strategies},
    }
    by_window_turnover = {}
    for label, (a, b) in WINDOWS.items():
        part = window(daily, a, b)
        yrs = len(part) / 252.0
        by_window_turnover[label] = {
            "annual_turnover_two_sided_x_nav": float(part["turnover_frac"].sum() / yrs),
            "trading_days_per_year": float((part["turnover_frac"] > 0).sum() / yrs),
            "cash_frac_mean": float(part["cash_frac"].mean()), "cash_frac_median": float(part["cash_frac"].median()),
            "days_cash_ge_99pct_share": float((part["cash_frac"] >= 0.99).mean()),
        }
    facts["by_window"] = by_window_turnover
    result["facts_1m"] = facts

    if big_daily is not None:
        both_invested = invested_mask & (big_daily["cash_frac"] < 0.99)
        gap = (daily["cash_frac"] - big_daily["cash_frac"])
        result["whole_share_1m_vs_100m"] = {
            "cash_frac_gap_mean_when_invested_all": float(gap[both_invested].mean()),
            "cash_frac_gap_mean_when_invested_2000_2004": float(gap[both_invested].loc["2000":"2004"].mean()),
            "cash_frac_gap_mean_when_invested_2020_on": float(gap[both_invested].loc["2020":].mean()),
            "max_abs_daily_ret_diff": float((daily["ret"] - big_daily["ret"]).abs().max()),
            "corr": float(np.corrcoef(daily["ret"].iloc[1:], big_daily["ret"].iloc[1:])[0, 1]),
            "annual_commission_frac_nav_1m": facts["annual_commission_frac_nav"],
            "annual_commission_frac_nav_100m": float(big_daily["commission_frac"].sum() / years_float),
            "commission_frac_2000_2004_annual_1m": float(daily["commission_frac"].loc["2000":"2004"].sum() / (len(daily.loc["2000":"2004"]) / 252.0)),
            "commission_frac_2000_2004_annual_100m": float(big_daily["commission_frac"].loc["2000":"2004"].sum() / (len(daily.loc["2000":"2004"]) / 252.0)),
            "nav_1m_start_usd_2000_2004_range": [float(daily["nav"].loc["2000":"2004"].min()), float(daily["nav"].loc["2000":"2004"].max())],
        }

    (OUT / "e2_stats.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(json.dumps({k: v for k, v in result.items() if k != "facts_1m"}, indent=2, default=str))
    print(json.dumps({k: v for k, v in facts.items() if k != "accounting_policy"}, indent=2, default=str))
    print(window_df.to_string())


if __name__ == "__main__":
    main()
