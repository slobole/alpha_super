"""Compare the audit re-runs with the stored shelf-rebuild sleeve series and report real-data statistics.

Three series per alias:
  stored  = MAIN/results/research/portfolio/shelf_rebuild_20260929/sources/<alias>__path.csv.gz (run 2026-09-29,
            commit f9ad358, end 2026-08-19)
  new     = WT/.../audit/rerun_taa_def/<alias>__path.csv.gz               (HEAD, end 2026-10-02)
  new819  = WT/.../audit/rerun_taa_def/end_20260819/<alias>__path.csv.gz  (HEAD, end 2026-08-19)

stored vs new819 isolates code / data-vintage changes at an unchanged end date; new819 vs new (cut at 2026-08-19)
isolates dependence on the end date (causality).

Conventions (shelf_rebuild lib.py / fund_menu common.py): r_t = V_t / V_(t-1) - 1 from the close before the first
invested day; Sharpe = mean / std * sqrt(252), rf 0; vol = std * sqrt(252); CAGR = (V_end / V_base) ** (365.25 /
calendar days) - 1; drawdown_t = V_t / max(V_base..V_t) - 1.

Read-only on MAIN. Writes only under the audit output folder.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WT_PATH = HERE.parents[3]
MAIN_PATH = Path(r"C:\Users\User\Documents\workspace\alpha_super")
STORED_DIR_PATH = MAIN_PATH / "results" / "research" / "portfolio" / "shelf_rebuild_20260929" / "sources"
STORED_STUDY_PATH = STORED_DIR_PATH.parent
NEW_DIR_PATH = WT_PATH / "results" / "research" / "portfolio" / "fund_products_20261005" / "audit" / "rerun_taa_def"
NEW819_DIR_PATH = NEW_DIR_PATH / "end_20260819"
OUT_DIR_PATH = NEW_DIR_PATH / "compare"

STORED_END_TS = pd.Timestamp("2026-08-19")
NEW_END_TS = pd.Timestamp("2026-10-02")
POST_START_TS = pd.Timestamp("2026-08-20")
LONG_START_TS = pd.Timestamp("2008-03-04")
EXACT_START_TS = pd.Timestamp("2012-10-02")
ENGINE_ALIAS_TUPLE = ("taa3x", "taa3x_1n", "core5", "btal_qqq", "ndx_vxn")
NAV_REL_TOL_FLOAT = 1e-10   # files carry 12 significant digits


def read_path(folder: Path, alias: str) -> pd.DataFrame:
    return pd.read_csv(folder / f"{alias}__path.csv.gz", index_col="date", parse_dates=True)


def read_tx(folder: Path, alias: str) -> pd.DataFrame:
    return pd.read_csv(folder / f"{alias}__transactions.csv.gz", parse_dates=["date"])


def read_meta(folder: Path, alias: str) -> dict:
    return json.loads((folder / f"{alias}__metadata.json").read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nav_to_returns(path_df: pd.DataFrame) -> pd.Series:
    """As shelf_rebuild lib.nav_to_returns: returns from the day before the first invested day."""
    nav = path_df["total_value_float"].astype(float)
    invested = path_df["portfolio_value_float"].abs() > 1e-9
    first = nav.index.get_loc(invested[invested].index[0])
    return nav.iloc[max(first - 1, 0):].pct_change(fill_method=None).iloc[1:]


def stats(return_ser: pd.Series, base_ts: pd.Timestamp) -> dict:
    r = return_ser.astype(float)
    nav = pd.concat([pd.Series([1.0], index=[base_ts]), (1.0 + r).cumprod()])
    days = (r.index[-1] - base_ts).days
    dd = nav / nav.cummax() - 1.0
    std = r.std()
    return {"first_return_date": r.index[0].date().isoformat(), "last_date": r.index[-1].date().isoformat(),
            "obs": int(len(r)), "total_return": float(nav.iloc[-1] - 1.0),
            "cagr": float(nav.iloc[-1] ** (365.25 / days) - 1.0),
            "vol": float(std * np.sqrt(252)), "sharpe_rf0": float(r.mean() / std * np.sqrt(252)) if std > 0 else np.nan,
            "maxdd": float(dd.min()), "maxdd_trough": dd.idxmin().date().isoformat()}


def window_stats(return_ser: pd.Series, lo: pd.Timestamp, hi: pd.Timestamp) -> dict:
    """Stats of the returns dated in [lo, hi]; the NAV base is the session before the first return."""
    full_index = return_ser.index
    r = return_ser.loc[lo:hi]
    pos = full_index.get_loc(r.index[0])
    # base = the prior session in the series (for the very first return: the prior calendar day's close stand-in)
    base_ts = full_index[pos - 1] if pos > 0 else r.index[0] - pd.Timedelta(days=1)
    return stats(r, base_ts)


def compare_paths(a_df: pd.DataFrame, b_df: pd.DataFrame, hi: pd.Timestamp) -> dict:
    """a = reference, b = candidate, both cut at hi."""
    a, b = a_df.loc[:hi], b_df.loc[:hi]
    out = {"a_first": a.index[0].date().isoformat(), "a_last": a.index[-1].date().isoformat(), "a_rows": int(len(a)),
           "b_first": b.index[0].date().isoformat(), "b_last": b.index[-1].date().isoformat(), "b_rows": int(len(b)),
           "same_index": bool(a.index.equals(b.index))}
    common = a.index.intersection(b.index)
    out["overlap_rows"] = int(len(common))
    na, nb = a.loc[common, "total_value_float"], b.loc[common, "total_value_float"]
    ra, rb = na.pct_change(fill_method=None).iloc[1:], nb.pct_change(fill_method=None).iloc[1:]
    rel = (nb / na - 1.0).abs()
    out["max_abs_nav_rel_diff"] = float(rel.max())
    out["max_abs_daily_return_diff"] = float((rb - ra).abs().max())
    out["max_abs_daily_return_diff_date"] = (rb - ra).abs().idxmax().date().isoformat()
    live = (ra != 0) | (rb != 0)
    out["return_corr"] = float(ra[live].corr(rb[live]))
    out["nav_ratio_at_hi"] = float(nb.loc[common[-1]] / na.loc[common[-1]])
    out["nav_a_at_hi"] = float(na.loc[common[-1]])
    out["nav_b_at_hi"] = float(nb.loc[common[-1]])
    diverged = rel[rel > NAV_REL_TOL_FLOAT]
    out["first_nav_divergence_date"] = diverged.index[0].date().isoformat() if len(diverged) else None
    for col in ("portfolio_value_float", "cash_float"):
        da = (b.loc[common, col] - a.loc[common, col]).abs() / na
        out[f"max_abs_{col}_diff_over_nav"] = float(da.max())
    out["identical_within_tol"] = bool(out["same_index"] and len(diverged) == 0
                                       and out["max_abs_portfolio_value_float_diff_over_nav"] <= NAV_REL_TOL_FLOAT
                                       and out["max_abs_cash_float_diff_over_nav"] <= NAV_REL_TOL_FLOAT)
    return out


def compare_tx(a_tx: pd.DataFrame, b_tx: pd.DataFrame, hi: pd.Timestamp) -> dict:
    a = a_tx[a_tx["date"] <= hi].reset_index(drop=True)
    b = b_tx[b_tx["date"] <= hi].reset_index(drop=True)
    out = {"a_count": int(len(a)), "b_count": int(len(b)), "b_count_total": int(len(b_tx)),
           "b_count_after_hi": int((b_tx["date"] > hi).sum())}
    cols = ["date", "asset_str", "amount_float", "fill_price_float", "signed_notional_float", "commission_float"]
    if len(a) == len(b):
        same_key = bool((a["date"].equals(b["date"])) and (a["asset_str"].equals(b["asset_str"])))
        out["same_date_asset_sequence"] = same_key
        if same_key and len(a):
            for col in cols[2:]:
                denom = a[col].abs().clip(lower=1e-12)
                out[f"max_rel_diff_{col}"] = float(((b[col] - a[col]).abs() / denom).max())
            out["identical_within_tol"] = bool(all(out[f"max_rel_diff_{c}"] <= 1e-9 for c in cols[2:]))
        else:
            out["identical_within_tol"] = bool(same_key)
    else:
        out["same_date_asset_sequence"] = False
        out["identical_within_tol"] = False
    if not out["identical_within_tol"]:
        merged = a[cols].merge(b[cols], on=["date", "asset_str"], how="outer", suffixes=("_a", "_b"), indicator=True)
        merged["amount_diff"] = merged["amount_float_b"].fillna(0) - merged["amount_float_a"].fillna(0)
        merged["price_rel_diff"] = merged["fill_price_float_b"] / merged["fill_price_float_a"] - 1.0
        bad = merged[(merged["_merge"] != "both") | (merged["amount_diff"].abs() > 1e-9)
                     | (merged["price_rel_diff"].abs() > 1e-9)].sort_values("date")
        out["first_tx_divergence_date"] = bad["date"].iloc[0].date().isoformat() if len(bad) else None
        out["diverging_tx_rows"] = int(len(bad))
        out["first_diverging_rows"] = json.loads(bad.head(12).to_json(orient="records", date_format="iso"))
    return out


def main() -> int:
    OUT_DIR_PATH.mkdir(parents=True, exist_ok=True)
    summary = {"stored_dir": str(STORED_DIR_PATH), "new_dir": str(NEW_DIR_PATH), "new819_dir": str(NEW819_DIR_PATH),
               "alias": {}}
    stat_rows, cmp_rows = [], []
    stored_sleeves = pd.read_csv(STORED_STUDY_PATH / "part_m" / "sleeves.csv", index_col="alias")

    for alias in ENGINE_ALIAS_TUPLE:
        stored, new, new819 = (read_path(STORED_DIR_PATH, alias), read_path(NEW_DIR_PATH, alias),
                               read_path(NEW819_DIR_PATH, alias))
        s_tx, n_tx, n819_tx = read_tx(STORED_DIR_PATH, alias), read_tx(NEW_DIR_PATH, alias), read_tx(NEW819_DIR_PATH, alias)
        s_meta, n_meta, n819_meta = (read_meta(STORED_DIR_PATH, alias), read_meta(NEW_DIR_PATH, alias),
                                     read_meta(NEW819_DIR_PATH, alias))
        entry = {
            "stored_vs_new": compare_paths(stored, new, STORED_END_TS),
            "stored_vs_new819": compare_paths(stored, new819, STORED_END_TS),
            "new819_vs_new_causality": compare_paths(new819, new, STORED_END_TS),
            "tx_stored_vs_new": compare_tx(s_tx, n_tx, STORED_END_TS),
            "tx_stored_vs_new819": compare_tx(s_tx, n819_tx, STORED_END_TS),
            "tx_new819_vs_new_causality": compare_tx(n819_tx, n_tx, STORED_END_TS),
            "bytes": {
                "stored_path_sha256": s_meta["path_sha256_str"],
                "stored_path_sha256_recomputed": sha256_file(STORED_DIR_PATH / f"{alias}__path.csv.gz"),
                "new819_path_sha256": n819_meta["path_sha256_str"],
                "new819_path_bytes_equal_stored": n819_meta["path_sha256_str"] == s_meta["path_sha256_str"],
                "stored_tx_sha256": s_meta["transaction_sha256_str"],
                "new819_tx_sha256": n819_meta["transaction_sha256_str"],
                "new819_tx_bytes_equal_stored": n819_meta["transaction_sha256_str"] == s_meta["transaction_sha256_str"],
            },
            "meta": {
                "stored_module_sha256": s_meta["module_sha256_str"], "new_module_sha256": n_meta["module_sha256_str"],
                "new_module_lf_sha256": n_meta.get("module_lf_sha256_str"),
                "stored_first_invested": s_meta["first_invested_date_str"],
                "new_first_invested": n_meta["first_invested_date_str"],
                "stored_tx_count": s_meta["transaction_count_int"], "new_tx_count": n_meta["transaction_count_int"],
                "new819_tx_count": n819_meta["transaction_count_int"],
                "new_requested_start": n_meta["requested_start_date_str"],
                "new_start_fallback_note": n_meta["start_fallback_note_str"],
                "new_git_head": n_meta["git_state_dict"]["head_commit_str"],
                "new_norgate_us_equities": n_meta["norgate_vintage_dict"]["database_last_update_by_name_dict"]["US Equities"],
                "new_vintage_changed_during_run": n_meta["norgate_vintage_changed_during_run_bool"],
                "new819_vintage_changed_during_run": n819_meta["norgate_vintage_changed_during_run_bool"],
            },
        }
        r_stored, r_new = nav_to_returns(stored), nav_to_returns(new)
        windows = {
            "stored_series_full_to_0819": (r_stored, r_stored.index[0], STORED_END_TS),
            "new_full_to_0819": (r_new, r_new.index[0], STORED_END_TS),
            "new_post_0820_1002": (r_new, POST_START_TS, NEW_END_TS),
            "new_full_to_1002": (r_new, r_new.index[0], NEW_END_TS),
            "new_exact_20121002_to_0819": (r_new, EXACT_START_TS, STORED_END_TS),
            "new_exact_20121002_to_1002": (r_new, EXACT_START_TS, NEW_END_TS),
        }
        entry["stats"] = {}
        for name, (ser, lo, hi) in windows.items():
            st = window_stats(ser, lo, hi)
            entry["stats"][name] = st
            stat_rows.append({"alias": alias, "window": name, **st})
        # cross-check of the metric code against the stored study table (EXACT window of the stored series)
        st_exact_stored = window_stats(r_stored, EXACT_START_TS, STORED_END_TS)
        entry["metric_code_check_vs_stored_sleeves_csv"] = {
            k: {"mine": st_exact_stored[m], "stored_table": float(stored_sleeves.at[alias, c])}
            for k, m, c in (("cagr", "cagr", "exact_cagr"), ("vol", "vol", "exact_vol"),
                            ("sharpe", "sharpe_rf0", "exact_sharpe"), ("maxdd", "maxdd", "exact_maxdd"))}
        summary["alias"][alias] = entry
        cmp_rows.append({"alias": alias,
                         **{f"sv819_{k}": v for k, v in entry["stored_vs_new819"].items()},
                         **{f"svnew_{k}": v for k, v in entry["stored_vs_new"].items()},
                         **{f"caus_{k}": v for k, v in entry["new819_vs_new_causality"].items()},
                         "tx_stored": entry["tx_stored_vs_new"]["a_count"],
                         "tx_new_to_0819": entry["tx_stored_vs_new"]["b_count"],
                         "tx_new_total": entry["tx_stored_vs_new"]["b_count_total"],
                         "tx_identical_stored_vs_new": entry["tx_stored_vs_new"]["identical_within_tol"],
                         "tx_identical_stored_vs_new819": entry["tx_stored_vs_new819"]["identical_within_tol"],
                         "tx_identical_causality": entry["tx_new819_vs_new_causality"]["identical_within_tol"]})

    # ── T-bills: BIL TOTALRETURN (the shelf rebuild's series; no stored path file exists) and the BIL engine pod ──
    tb_new, tb_819 = read_path(NEW_DIR_PATH, "tbill"), read_path(NEW819_DIR_PATH, "tbill")
    r_tb = tb_new["total_value_float"].pct_change(fill_method=None).iloc[1:]
    r_tb819 = tb_819["total_value_float"].pct_change(fill_method=None).iloc[1:]
    common = r_tb819.index.intersection(r_tb.index)
    bench = pd.read_csv(STORED_STUDY_PATH / "part_m" / "benchmarks.csv", index_col="book").loc["T-bills (BIL)"]
    tb_entry = {
        "first": tb_new.index[0].date().isoformat(), "last": tb_new.index[-1].date().isoformat(),
        "end1002_vs_end0819_max_abs_return_diff": float((r_tb.loc[common] - r_tb819.loc[common]).abs().max()),
        "end1002_vs_end0819_level_ratio_range": [float((tb_new["bil_totalreturn_close_float"].reindex(tb_819.index)
                                                        / tb_819["bil_totalreturn_close_float"]).min()),
                                                 float((tb_new["bil_totalreturn_close_float"].reindex(tb_819.index)
                                                        / tb_819["bil_totalreturn_close_float"]).max())],
        "stats": {"long_20080304_to_0819": window_stats(r_tb, LONG_START_TS, STORED_END_TS),
                  "exact_20121002_to_0819": window_stats(r_tb, EXACT_START_TS, STORED_END_TS),
                  "post_0820_1002": window_stats(r_tb, POST_START_TS, NEW_END_TS),
                  "full_to_1002": window_stats(r_tb, r_tb.index[0], NEW_END_TS),
                  "long_20080304_to_1002": window_stats(r_tb, LONG_START_TS, NEW_END_TS),
                  "exact_20121002_to_1002": window_stats(r_tb, EXACT_START_TS, NEW_END_TS)},
        "stored_benchmarks_csv": {k: float(bench[k]) for k in ("long_cagr", "long_vol", "long_sharpe", "long_maxdd",
                                                               "exact_cagr", "exact_vol", "exact_sharpe",
                                                               "exact_maxdd")},
    }
    summary["tbill"] = tb_entry
    for name, st in tb_entry["stats"].items():
        stat_rows.append({"alias": "tbill", "window": name, **st})

    pod_new, pod_819 = read_path(NEW_DIR_PATH, "bil_pod"), read_path(NEW819_DIR_PATH, "bil_pod")
    r_pod = nav_to_returns(pod_new)
    pod_entry = {
        "new819_vs_new_causality": compare_paths(pod_819, pod_new, STORED_END_TS),
        "tx_new819_vs_new_causality": compare_tx(read_tx(NEW819_DIR_PATH, "bil_pod"), read_tx(NEW_DIR_PATH, "bil_pod"),
                                                 STORED_END_TS),
        "meta": {k: read_meta(NEW_DIR_PATH, "bil_pod")[k] for k in (
            "engine_first_date_str", "first_invested_date_str", "end_date_str", "transaction_count_int",
            "requested_start_date_str", "start_fallback_note_str", "mean_cash_nav_weight_float")},
        "stats": {"full_to_0819": window_stats(r_pod, r_pod.index[0], STORED_END_TS),
                  "exact_20121002_to_0819": window_stats(r_pod, EXACT_START_TS, STORED_END_TS),
                  "post_0820_1002": window_stats(r_pod, POST_START_TS, NEW_END_TS),
                  "full_to_1002": window_stats(r_pod, r_pod.index[0], NEW_END_TS)},
        "corr_with_bil_totalreturn": float(r_pod.corr(r_tb.reindex(r_pod.index))),
    }
    summary["bil_pod"] = pod_entry
    for name, st in pod_entry["stats"].items():
        stat_rows.append({"alias": "bil_pod", "window": name, **st})

    (OUT_DIR_PATH / "compare_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(stat_rows).to_csv(OUT_DIR_PATH / "stats_by_window.csv", index=False, lineterminator="\n")
    pd.DataFrame(cmp_rows).to_csv(OUT_DIR_PATH / "comparison.csv", index=False, lineterminator="\n")
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
