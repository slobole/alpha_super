"""Freshness audit (read-only, no backtests): do the stored sleeve series still reflect the strategy code at HEAD?

Parts
  A  file hashes: shelf-rebuild metadata / ledger hashes vs this worktree (HEAD 5c0d48d tree)
  B  MR capsule gate-switch formulas (pre-fef834a, HEAD, MAIN-uncommitted) on the real VIX / calendar; BIL / SPMO
     missing-open sessions (decides whether the uncommitted engine hook can ever fire in a backtest)
  C  portfolio_refresh_20260927/sleeve_series_incl_2008.csv.gz vs the shelf-rebuild 2026-09-29 runs
  D  MR capsule build-check runs vs the PortfolioManager pods (stock fills, NAV)

Reads MAIN results only; writes only under this worktree's results/research/portfolio/fund_products_20261005/audit/freshness.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

WT = Path(__file__).resolve().parents[4]
MAIN = Path(r"C:\Users\User\Documents\workspace\alpha_super")
OUT = WT / "results/research/portfolio/fund_products_20261005/audit/freshness"
SR = MAIN / "results/research/portfolio/shelf_rebuild_20260929"
END = pd.Timestamp("2026-08-19")
LONG_START = pd.Timestamp("2008-03-04")
EXACT_START = pd.Timestamp("2012-10-02")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def tree_hash(rel: str) -> str:
    d = hashlib.sha256()
    for p in sorted((WT / rel).rglob("*.py")):
        d.update(p.relative_to(WT).as_posix().encode("utf-8"))
        d.update(b"\0")
        d.update(sha256(p).encode("ascii"))
        d.update(b"\n")
    return d.hexdigest()


def nav_to_returns(path_df: pd.DataFrame) -> pd.Series:
    nav = path_df["total_value_float"].astype(float)
    invested = path_df["portfolio_value_float"].abs() > 1e-9
    first = nav.index.get_loc(invested[invested].index[0])
    return nav.iloc[max(first - 1, 0):].pct_change(fill_method=None).iloc[1:]


def read_path(folder: Path, alias: str) -> pd.DataFrame:
    return pd.read_csv(folder / f"{alias}__path.csv.gz", index_col="date", parse_dates=True)


def cagr(r: pd.Series) -> float:
    r = r.dropna()
    return float((1 + r).prod() ** (252.0 / len(r)) - 1) if len(r) else float("nan")


def maxdd(r: pd.Series) -> float:
    v = (1 + r.dropna()).cumprod()
    return float((v / v.cummax() - 1).min()) if len(v) else float("nan")


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if len(r) > 2 else float("nan")


# ─── A: hashes ────────────────────────────────────────────────────────────────


def part_a() -> dict:
    out: dict = {"module_hash": {}, "dependency_hash": {}}
    for meta_path in sorted((SR / "sources").glob("*__metadata.json")):
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        now = sha256(WT / m["module_path_str"])
        out["module_hash"][m["alias_str"]] = {"module": m["module_path_str"], "tier_at_run": m["tier_str"],
                                              "same_as_head": now == m["module_sha256_str"],
                                              "path_sha_ok": sha256(SR / "sources" / f"{m['alias_str']}__path.csv.gz") == m["path_sha256_str"],
                                              "first_invested": m["first_invested_date_str"], "end": m["end_date_str"],
                                              "cash_policy": m["positive_cash_rate_policy_str"],
                                              "mean_cash_w": m["mean_cash_nav_weight_float"],
                                              "neg_cash_days": m["negative_cash_day_count_int"],
                                              "min_cash_w": m["minimum_cash_nav_weight_float"]}
    recorded = None
    for line in (SR / "experiment_ledger.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        if d.get("event_str") == "sleeve_runs_started" and d.get("only_list") is None:
            recorded = d["shared_execution_dependency_hash_dict"]
            out["run_head"] = d["git_state_dict"]["head_commit_str"]
            out["run_dirty"] = d["git_state_dict"]["dirty_path_list"]
            out["run_started_utc"] = d["recorded_at_utc_str"]
            out["norgate_vintage"] = d["norgate_vintage_dict"]
    for key, old in recorded.items():
        now = tree_hash(key.split("::")[1]) if key.startswith("python_tree::") else sha256(WT / key)
        out["dependency_hash"][key] = {"same_as_head": now == old}
    return out


# ─── B: gate-switch formulas and parking-ETF missing opens ────────────────────


def part_b() -> dict:
    sys.path.insert(0, str(WT))
    from data.norgate_loader import load_price_timeseries
    from strategies.mr_capsule.vix_stress_gate import gate_state_at, load_vix_close_ser, stress_gate_open_ser
    import norgatedata

    # unpadded $SPX rows = the exchange sessions
    spx = norgatedata.price_timeseries("$SPX", start_date="1998-01-01", timeseriesformat="pandas-dataframe")
    cal = pd.DatetimeIndex(pd.to_datetime(spx.index))
    cal = cal[cal >= pd.Timestamp("2004-01-02")]
    vix = load_vix_close_ser(None)
    gate = stress_gate_open_ser(vix)
    out: dict = {"calendar_first": str(cal[0].date()), "calendar_last": str(cal[-1].date()), "sessions": int(len(cal)),
                 "vix_first": str(vix.index[0].date()), "vix_last": str(vix.index[-1].date()),
                 "sessions_without_vix_row": [str(d.date()) for d in cal.difference(vix.index)],
                 "vix_rows_not_sessions_since_2004": [str(d.date()) for d in vix.index[vix.index >= cal[0]].difference(cal)]}

    # decision closes T = cal[:-1]; execution at cal[1:]
    old_prev, n_diff_head_vs_old, n_diff_dirty_vs_head, switches = None, 0, 0, 0
    first_decision_old_true = None
    for i in range(len(cal) - 1):
        t = cal[i]
        g = gate_state_at(gate, t)
        old = old_prev is None or old_prev != g           # cc458bc / 4b6da58: state kept between iterate calls
        old_prev = g
        pos = int(gate.index.searchsorted(t, side="right")) - 1
        head = False if pos < 1 else bool(gate.iloc[pos - 1]) != bool(g)   # HEAD (fef834a)
        prev_pos = int(cal.searchsorted(t, side="left")) - 1
        # MAIN uncommitted: previous pricing session of the frame (the frame starts in 1998, so there is always one)
        dirty = gate_state_at(gate, cal[prev_pos]) != bool(g) if prev_pos >= 0 else gate_state_at(gate, t - pd.Timedelta(days=1)) != bool(g)
        if i == 0:
            first_decision_old_true = bool(old)
            # the first decision differs by construction (old: always re-target); recorded separately
            continue
        switches += int(head)
        n_diff_head_vs_old += int(head != old)
        n_diff_dirty_vs_head += int(dirty != head)
    out.update(gate_switch_count=int(switches), decisions_head_differs_from_pre_fef834a=int(n_diff_head_vs_old),
               decisions_uncommitted_differs_from_head=int(n_diff_dirty_vs_head),
               first_decision_old_formula_retargets=first_decision_old_true)

    for sym in ("BIL", "SPMO"):
        px = load_price_timeseries(sym, start_date_str="1998-01-01")
        px.index = pd.to_datetime(px.index)
        first = px["Close"].first_valid_index()
        sess = cal[cal >= first]
        opens = px["Open"].reindex(sess)
        out[sym] = {"first_bar": str(first.date()), "sessions_since_first_bar": int(len(sess)),
                    "sessions_missing_open_padded_loader": int(opens.isna().sum()),
                    "zero_volume_sessions": int((px["Volume"].reindex(sess).fillna(0) == 0).sum())}
    return out


# ─── C: refresh sleeve file (09-27) vs shelf rebuild (09-29) ──────────────────


def part_c() -> pd.DataFrame:
    old = pd.read_csv(MAIN / "results/research/portfolio/portfolio_refresh_20260927/sleeve_series_incl_2008.csv.gz",
                      index_col=0, parse_dates=True)
    meta_alias = [p.name.split("__")[0] for p in sorted((SR / "sources").glob("*__metadata.json"))]
    sleeve = pd.DataFrame({a: nav_to_returns(read_path(SR / "sources", a)) for a in meta_alias}).sort_index().loc[:END]
    index = sleeve.index
    long = sleeve.copy()
    for alias in ("taa3x", "taa3x_1n", "taa2x_1n", "btal_qqq"):
        proxy = nav_to_returns(read_path(SR / "proxy_runs" / "splice_scaled", alias)).reindex(index)
        early = index < EXACT_START
        long.loc[early, alias] = proxy[early]
    etf_research = pd.read_csv(MAIN / "results/research/dv2_deep_20260925/sources/etf_ind_adv50__path.csv.gz",
                               index_col="date", parse_dates=True)["total_value_float"]
    etf_first = sleeve["etf_dv2"].first_valid_index()
    mask = (index >= LONG_START) & (index < etf_first)
    long.loc[mask, "etf_dv2"] = etf_research.pct_change(fill_method=None).reindex(index)[mask]
    long.loc[LONG_START:END].to_csv(OUT / "shelf_rebuild_long_house_cash_returns.csv.gz", float_format="%.10g")

    pairs = [("taa_btal_tqqq", "taa3x"), ("taa_btal_1n_tqqq", "taa3x_1n"), ("taa_btal_lin_qqq", "btal_qqq"),
             ("core5", "core5"), ("ndx_vxn", "ndx_vxn"), ("ndx_atrfix", "ndx_vxn"), ("ndx_natr20", "ndx_natr20"),
             ("dv2", "dv2"), ("hpi_vote", "hpi_vote"), ("etf_ind_fix", "etf_dv2")]
    windows = {"2008-03-04..2026-08-19": (LONG_START, END), "2012-10-02..2026-08-19": (EXACT_START, END),
               "2023-08-21..2026-08-19": (pd.Timestamp("2023-08-21"), END)}
    rows = []
    for old_col, new_col in pairs:
        for wname, (lo, hi) in windows.items():
            both = pd.concat([old[old_col], long[new_col]], axis=1, keys=["old", "new"]).loc[lo:hi].dropna()
            diff = (both["new"] - both["old"]).abs()
            rows.append({"refresh_0927_col": old_col, "shelf_0929_alias": new_col, "window": wname, "n": len(both),
                         "daily_corr": float(both.corr().iloc[0, 1]), "days_diff_gt_1bp": int((diff > 1e-4).sum()),
                         "max_abs_daily_diff": float(diff.max()),
                         "cagr_old": cagr(both["old"]), "cagr_new": cagr(both["new"]),
                         "cagr_new_minus_old_pp": 100 * (cagr(both["new"]) - cagr(both["old"])),
                         "sharpe_old": sharpe(both["old"]), "sharpe_new": sharpe(both["new"]),
                         "maxdd_old": maxdd(both["old"]), "maxdd_new": maxdd(both["new"])})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "refresh0927_vs_shelf0929.csv", index=False, float_format="%.6g")
    return table


# ─── D: MR capsule build-check vs PortfolioManager pods ───────────────────────


def part_d() -> dict:
    sys.path.insert(0, str(WT))
    build = MAIN / "results/research/mr_capsule_build_20261004"
    pm = MAIN / "results/research/portfolio/mr_capsule_bil/vanilla_backtest/2026-10-04_115751"
    out: dict = {}
    for pod, pod_dir, pkl in (("dv2", "pod_mr_dv2_gated_bil", "strategy_mr_dv2_vix_gated_bil.pkl"),
                              ("hpi", "pod_mr_hpi_vote_gated_bil", "strategy_mr_hpi_vote_vix_gated_bil.pkl")):
        b_nav = pd.read_csv(build / f"{pod}_bil_nav.csv", index_col=0, parse_dates=True)["total_value"].astype(float)
        b_tx = pd.read_csv(build / f"{pod}_bil_transactions.csv")
        p_tx = pd.read_csv(pm / "pods" / pod_dir / "transactions.csv")
        date_col_b = "bar" if "bar" in b_tx.columns else b_tx.columns[0]
        date_col_p = "bar" if "bar" in p_tx.columns else p_tx.columns[0]

        def keyset(tx: pd.DataFrame, date_col: str, parking: bool) -> set:
            is_park = tx["asset"].isin(["BIL", "SPMO"])
            sel = tx[is_park] if parking else tx[~is_park]
            return set(zip(pd.to_datetime(sel[date_col]).dt.date, sel["asset"], np.sign(sel["amount"]).astype(int)))

        sb_, sp_ = keyset(b_tx, date_col_b, False), keyset(p_tx, date_col_p, False)
        pb_, pp_ = keyset(b_tx, date_col_b, True), keyset(p_tx, date_col_p, True)
        with (pm / "pods" / pod_dir / pkl).open("rb") as fh:
            strat = pickle.load(fh)
        p_nav = strat.results["total_value"].astype(float)
        p_nav.index = pd.to_datetime(p_nav.index)
        both = pd.concat([b_nav.pct_change(), p_nav.pct_change()], axis=1, keys=["build_100k", "pm_500k"]).dropna()
        out[pod] = {"build_first": str(b_nav.index[0].date()), "build_last": str(b_nav.index[-1].date()),
                    "pm_first": str(p_nav.index[0].date()), "pm_last": str(p_nav.index[-1].date()),
                    "stock_fills_build": len(sb_), "stock_fills_pm": len(sp_), "stock_fills_only_build": len(sb_ - sp_),
                    "stock_fills_only_pm": len(sp_ - sb_),
                    "parking_fills_build": len(pb_), "parking_fills_pm": len(pp_),
                    "parking_only_build": len(pb_ - pp_), "parking_only_pm": len(pp_ - pb_),
                    "first_parking_fill_build": str(min(k[0] for k in pb_)), "first_parking_fill_pm": str(min(k[0] for k in pp_)),
                    "daily_corr": float(both.corr().iloc[0, 1]), "cagr_build_100k": cagr(both["build_100k"]),
                    "cagr_pm_500k": cagr(both["pm_500k"]), "maxdd_build": maxdd(both["build_100k"]), "maxdd_pm": maxdd(both["pm_500k"]),
                    "max_abs_daily_diff": float((both["build_100k"] - both["pm_500k"]).abs().max()),
                    "only_build_sample": sorted(map(str, sb_ - sp_))[:6], "only_pm_sample": sorted(map(str, sp_ - sb_))[:6]}
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ["a", "b", "c", "d"]
    result: dict = {}
    if "a" in which:
        result["A"] = part_a()
    if "b" in which:
        result["B"] = part_b()
    if "c" in which:
        table = part_c()
        pd.set_option("display.width", 250)
        pd.set_option("display.max_columns", 30)
        print(table.round(5).to_string())
    if "d" in which:
        result["D"] = part_d()
    (OUT / f"freshness_checks_{'_'.join(which)}.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
