"""CORE5 (strategy_taa_adaptive_macro_core5) leakage tests on REAL Norgate data.

Tests
  1. Decision-table future-action invariance: per asset, split case (CAPITALSPECIAL + TOTALRETURN signal both / k)
     and TR-only (dividend) case, k in (40, 0.1, 1.5); BIL CAPITALSPECIAL only.
  2. Engine-level invariance: full engine runs with rescaled data, compare transaction decisions and the engine's
     own daily target-weight table.
  3. Truncation (prefix): decision table from data ending at T equals the full-history table on all dates <= T.
  4. Padding provenance: ALLMARKETDAYS rows that are not observed (PaddingType.NONE) inside the run window;
     XNYS month-end coverage of the price index.
  5. E-02 accounting: historical_share_units_bool=True run; per-share fee recalculation; DBC borrow on raw units.
Outputs under results/research/leakage_hunt_20260927/def/.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from def_common import *  # noqa: F401,F403
from def_common import (BOOK_END, BOOK_START, FACTORS, OUT, cached, dump_json, harness, metric_rows,
                        per_share_fee_effect, rescale_frame, truncation_dates, xnys_sessions, month_end_sessions)

import numpy as np
import pandas as pd

from alpha.engine.backtest import run_daily
from strategies.taa_beyond_6040 import strategy_taa_adaptive_macro_core5 as m

CFG = m.DEFAULT_CONFIG
RISK = list(CFG.risk_asset_tuple)
TRADE = RISK + [CFG.reserve_asset_str]
NS = m.signal_namespace_str


def load_pricing() -> pd.DataFrame:
    return cached("core5_pricing", lambda: m.get_adaptive_macro_core5_data(CFG))


def decision_table(pricing_df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised twin of build_target_weight_ser (validated against it row by row on the baseline)."""
    s = m.AdaptiveMacroCore5Strategy()
    sig = s.compute_signals(pricing_df)
    ls = pd.DataFrame({a: sig[(NS(a), "long_state_ser")].astype(float) for a in RISK})
    sh = sig[(NS(CFG.commodity_asset_str), "short_state_ser")].astype(float)
    vol = sig[(NS(CFG.commodity_asset_str), "annualized_volatility_ser")].astype(float)
    ok = ls.notna().all(axis=1) & np.isfinite(vol)
    ls, sh, vol = ls[ok], sh[ok], vol[ok]
    w = ls * CFG.sleeve_weight_float
    short_w = -np.minimum(CFG.commodity_short_cap_float, CFG.commodity_short_vol_target_float / vol)
    short_w = short_w.where(sh == 1.0, 0.0)
    out = pd.DataFrame(index=ls.index)
    for a in RISK:
        out[f"w_{a}"] = w[a]
    out[f"w_{CFG.commodity_asset_str}"] = out[f"w_{CFG.commodity_asset_str}"] + short_w
    out[f"w_{CFG.reserve_asset_str}"] = 1.0 - w.sum(axis=1)
    out["w_Cash"] = short_w.abs()
    out["changed"] = sig.loc[ok.index[ok], (m.PORTFOLIO_NAMESPACE_STR, m.LONG_STATE_CHANGED_FIELD_STR)].astype(bool)
    out["month_end"] = sig.loc[ok.index[ok], (m.PORTFOLIO_NAMESPACE_STR, m.MONTH_END_REBALANCE_FIELD_STR)].astype(bool)
    return out


def decision_table_reference(pricing_df: pd.DataFrame) -> pd.DataFrame:
    """Per decision date: target weights implied by the Close_T signal row + rebalance flag + month-end flag."""
    s = m.AdaptiveMacroCore5Strategy()
    sig = s.compute_signals(pricing_df)
    rows = {}
    for ts in sig.index:
        row = sig.loc[ts]
        ls = pd.Series({a: row.get((NS(a), "long_state_ser"), np.nan) for a in RISK}, dtype=float)
        if ls.isna().any():
            continue
        sh = float(row.get((NS(CFG.commodity_asset_str), "short_state_ser"), np.nan))
        vol = float(row.get((NS(CFG.commodity_asset_str), "annualized_volatility_ser"), np.nan))
        if not np.isfinite(vol):
            continue
        w = m.build_target_weight_ser(ls, bool(sh), vol, CFG)
        rec = {f"w_{k}": float(v) for k, v in w.items()}
        rec["changed"] = bool(row[(m.PORTFOLIO_NAMESPACE_STR, m.LONG_STATE_CHANGED_FIELD_STR)])
        rec["month_end"] = bool(row[(m.PORTFOLIO_NAMESPACE_STR, m.MONTH_END_REBALANCE_FIELD_STR)])
        rows[ts] = rec
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def compare_tables(ref: pd.DataFrame, cand: pd.DataFrame, end_ts=None) -> dict:
    if end_ts is not None:
        ref = ref.loc[:end_ts]
        cand = cand.loc[:end_ts]
    wcols = [c for c in ref.columns if c.startswith("w_")]
    out = harness.compare_weight_frames(ref[wcols], cand[wcols], atol_float=1e-9)
    # DBC short size = 2.5%/vol63; pandas' rolling std (running sums) turns 1-ulp return differences from a
    # rescaled price into ~1e-8 weight noise.  A decision difference is a state/flag change or > 1e-6 in weight.
    strict_n = out["n_dates_different"]
    out["n_dates_different_strict_1e-9"] = strict_n
    out["n_dates_different"] = harness.compare_weight_frames(ref[wcols], cand[wcols], atol_float=1e-6)["n_dates_different"]
    common = ref.index.intersection(cand.index)
    for a in RISK:
        if f"w_{a}" in ref.columns:
            sgn = np.sign(ref.loc[common, f"w_{a}"].round(12)) != np.sign(cand.loc[common, f"w_{a}"].round(12))
            out[f"n_state_sign_diff_{a}"] = int(sgn.sum())
    for flag in ("changed", "month_end"):
        diff = ref.loc[common, flag] != cand.loc[common, flag]
        out[f"n_{flag}_flag_diff"] = int(diff.sum())
        out[f"first_{flag}_flag_diff"] = [d.date().isoformat() for d in common[diff.to_numpy()][:5]]
    out["passed"] = (out["n_dates_different"] == 0 and out["n_changed_flag_diff"] == 0
                     and out["n_month_end_flag_diff"] == 0 and out["reference_only_dates"] == 0
                     and out["candidate_only_dates"] == 0)
    return out


def run_engine(pricing_df: pd.DataFrame, start=None, historical_units=False, rate=None):
    cfg = CFG if rate is None else m.replace(CFG, annual_dbc_borrow_rate_float=rate)
    cal = m.build_execution_calendar_idx(pricing_df, cfg, start or cfg.backtest_start_date_str)
    s = m._build_strategy_obj(cfg, cal)
    if historical_units:
        s.historical_share_units_bool = True
    run_daily(s, pricing_df, calendar=cal, show_progress=False, show_signal_progress_bool=False,
              audit_override_bool=False)
    return s


def main():
    t0 = time.time()
    log = harness.ResultLog("core5")
    pricing = load_pricing()
    print("loaded", pricing.shape, pricing.index[0], pricing.index[-1], f"{time.time()-t0:.1f}s")

    base_tab = decision_table(pricing)
    ref_tab = cached("core5_base_table_reference", lambda: decision_table_reference(pricing))
    vres = compare_tables(ref_tab, base_tab.reindex(columns=ref_tab.columns))
    log.add("self_check", "vectorised_table_equals_build_target_weight_ser", vres["passed"], vres)
    base_tab.to_csv(OUT / "core5_decision_table_full.csv")
    print("decision table", base_tab.shape, f"{time.time()-t0:.1f}s")

    # ---- 1. decision-table invariance -------------------------------------------------------------------
    for a in RISK:
        for k in FACTORS:
            for case, ns_list in (("split", [a, NS(a)]), ("tr_dividend_only", [NS(a)])):
                cand = decision_table(rescale_frame(pricing, ns_list, k))
                res = compare_tables(base_tab, cand)
                log.add("invariance_decision_table", f"{a}_{case}_k{k}", res["passed"], res)
    for k in FACTORS:
        cand = decision_table(rescale_frame(pricing, [CFG.reserve_asset_str], k))
        res = compare_tables(base_tab, cand)
        log.add("invariance_decision_table", f"BIL_capitalspecial_k{k}", res["passed"], res)
    print(f"invariance tables done {time.time()-t0:.1f}s")

    # ---- 3. truncation ----------------------------------------------------------------------------------
    sessions = xnys_sessions()
    for name, T in truncation_dates(sessions).items():
        cand = decision_table(pricing.loc[:T])
        res = compare_tables(base_tab, cand, end_ts=T)
        res["row_T_full"] = base_tab.loc[T].to_dict() if T in base_tab.index else None
        res["row_T_trunc"] = cand.loc[T].to_dict() if T in cand.index else None
        log.add("truncation_prefix", name, res["passed"] and res["row_T_full"] == res["row_T_trunc"], res)
    # reference: the pre-fb81e86 helper (terminal row always month-end) would have flagged the cut-off
    old_flag = {}
    for name, T in truncation_dates(sessions).items():
        idx = pricing.loc[:T].index
        per = idx.to_period("M")
        nxt = pd.Series(per, index=idx).shift(-1)
        old = pd.Series(nxt.isna().to_numpy() | (per != nxt.to_numpy()), index=idx)
        old_flag[name] = {"old_helper_flag_T": bool(old.iloc[-1]), "new_helper_flag_T": bool(base_tab.loc[T, "month_end"])}
    dump_json(old_flag, "core5_old_vs_new_month_end_helper_at_cutoffs.json")
    print(f"truncation done {time.time()-t0:.1f}s")

    # ---- 4. padding provenance + XNYS month-end coverage -----------------------------------------------
    import norgatedata as nd
    run_start = pd.Timestamp(CFG.backtest_start_date_str)
    pad_rows = {}
    for a in TRADE:
        for adj_name, adj in (("CAPITALSPECIAL", nd.StockPriceAdjustmentType.CAPITALSPECIAL),
                              ("TOTALRETURN", nd.StockPriceAdjustmentType.TOTALRETURN)):
            if adj_name == "TOTALRETURN" and a not in RISK:
                continue
            obs = nd.price_timeseries(a, stock_price_adjustment_setting=adj, padding_setting=nd.PaddingType.NONE,
                                      start_date="1990-01-01", timeseriesformat="pandas-dataframe")
            col = (a, "Close") if adj_name == "CAPITALSPECIAL" else (NS(a), "Close")
            ser = pricing[col].dropna()
            padded = ser.index.difference(obs.index)
            pad_rows[f"{a}_{adj_name}"] = {
                "first_obs": obs.index[0].date().isoformat(),
                "n_padded_rows_total": int(len(padded)),
                "n_padded_rows_in_run": int((padded >= run_start).sum()),
                "n_padded_rows_in_signal_warmup_before_run": int(((padded < run_start) & (padded >= obs.index[0])).sum()),
                "padded_dates_examples": [d.date().isoformat() for d in padded[:10]],
            }
    idx = pricing.index
    not_xnys = idx.difference(sessions)
    me = month_end_sessions(sessions[(sessions >= run_start) & (sessions <= idx[-1])])
    pad_rows["index_rows_not_XNYS_sessions"] = [d.date().isoformat() for d in not_xnys[not_xnys >= "2000-01-01"]]
    pad_rows["xnys_month_ends_missing_from_index"] = [d.date().isoformat() for d in me.difference(idx)]
    xnys_in_range = sessions[(sessions >= run_start) & (sessions <= idx[-1])]
    pad_rows["xnys_sessions_missing_from_index"] = [d.date().isoformat() for d in xnys_in_range.difference(idx)]
    dump_json(pad_rows, "core5_padding_provenance.json")
    any_pad_run = sum(v["n_padded_rows_in_run"] for k, v in pad_rows.items() if isinstance(v, dict) and "n_padded_rows_in_run" in v)
    log.add("padding_provenance", "padded_rows_in_run_window", any_pad_run == 0, {"n": any_pad_run})
    log.add("calendar", "xnys_month_ends_in_index", len(pad_rows["xnys_month_ends_missing_from_index"]) == 0,
            {"missing": pad_rows["xnys_month_ends_missing_from_index"], "non_xnys_rows": pad_rows["index_rows_not_XNYS_sessions"]})
    print(f"padding done {time.time()-t0:.1f}s")

    # ---- 2 & 5. engine runs -----------------------------------------------------------------------------
    base = run_engine(pricing)
    base_tx = harness.transaction_decisions(base.get_transactions())
    print(f"base engine {time.time()-t0:.1f}s", base.summary.loc[["CAGR", "Sharpe Ratio"]] if "CAGR" in base.summary.index else "")
    base.daily_target_weights.to_csv(OUT / "core5_engine_daily_target_weights.csv")
    for a, k, case in (("DBC", 40.0, "split"), ("SPY", 0.1, "split"), ("BIL", 1.5, "split"), ("GLD", 40.0, "tr_dividend_only")):
        ns_list = [a] if a == "BIL" else ([a, NS(a)] if case == "split" else [NS(a)])
        s = run_engine(rescale_frame(pricing, ns_list, k))
        tx = harness.transaction_decisions(s.get_transactions())
        res = harness.compare_transaction_decisions(base_tx, tx)
        wres = harness.compare_weight_frames(base.daily_target_weights, s.daily_target_weights, atol_float=1e-9)
        res["daily_target_weight_compare"] = wres
        ok = res["n_only_reference"] == 0 and res["n_only_candidate"] == 0 and wres["n_dates_different"] == 0
        log.add("invariance_engine", f"{a}_{case}_k{k}", ok, res)
    print(f"engine invariance done {time.time()-t0:.1f}s")

    windows = {"full": (None, None), "book": (BOOK_START, BOOK_END)}
    rows = metric_rows("baseline_adjusted_units", base.results["daily_returns"], windows)
    hist = run_engine(pricing, historical_units=True)
    rows += metric_rows("historical_share_units", hist.results["daily_returns"], windows)
    htx = harness.transaction_decisions(hist.get_transactions())
    hres = harness.compare_transaction_decisions(base_tx, htx)
    log.add("historical_units_decisions", "same_date_asset_side", hres["n_only_reference"] == 0 and hres["n_only_candidate"] == 0, hres)
    fee = per_share_fee_effect(base.get_transactions(), pricing, CFG.commission_per_share_float, CFG.commission_minimum_float)
    fee_frame = fee.pop("_frame")
    fee_frame.to_csv(OUT / "core5_fee_units_by_fill.csv", index=False)
    fee["commission_total_model_run"] = float(base.get_transactions()["commission"].sum())
    fee["commission_total_historical_units_run"] = float(hist.get_transactions()["commission"].sum())

    # DBC borrow on raw units
    bf = base.borrow_fee_df.copy()
    bor = {}
    if len(bf):
        k_ser = pricing[("DBC", "Unadjusted Close")] / pricing[("DBC", "Close")]
        bf["k"] = bf["accrual_start_date_ts"].map(k_ser)
        bf["raw_close"] = bf["dbc_close_float"] * bf["k"]
        bf["raw_shares"] = bf["dbc_share_float"].abs() / bf["k"]
        bf["fee_raw"] = (bf["raw_shares"] * np.ceil(1.02 * bf["raw_close"]) * bf["annual_borrow_rate_float"]
                         * bf["calendar_day_count_int"] / 360.0)
        bf["fee_exact_102"] = (bf["dbc_share_float"].abs() * 1.02 * bf["dbc_close_float"] * bf["annual_borrow_rate_float"]
                               * bf["calendar_day_count_int"] / 360.0)
        bor = {"n_accrual_rows": int(len(bf)), "fee_model_total": float(bf["borrow_fee_float"].sum()),
               "fee_raw_units_total": float(bf["fee_raw"].sum()), "fee_no_ceil_total": float(bf["fee_exact_102"].sum()),
               "k_min": float(bf["k"].min()), "k_max": float(bf["k"].max())}
        bf.to_csv(OUT / "core5_dbc_borrow_raw_units.csv", index=False)
    # k ranges of traded assets
    kr = {a: {"k_min": float((pricing[(a, "Unadjusted Close")] / pricing[(a, "Close")]).loc[run_start:].min()),
              "k_max": float((pricing[(a, "Unadjusted Close")] / pricing[(a, "Close")]).loc[run_start:].max())} for a in TRADE}
    # borrow at 0 (bound of the entire borrow layer)
    nob = run_engine(pricing, rate=0.0)
    rows += metric_rows("borrow_rate_0_reference", nob.results["daily_returns"], windows)
    pd.DataFrame(rows).to_csv(OUT / "core5_metrics.csv", index=False)
    dump_json({"fee_units": fee, "dbc_borrow_units": bor, "k_range_since_run_start": kr,
               "historical_units_vs_baseline_decisions": hres}, "core5_accounting_units.json")
    log.save("def_core5")
    print(pd.DataFrame(rows).to_string())
    print(f"done {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
