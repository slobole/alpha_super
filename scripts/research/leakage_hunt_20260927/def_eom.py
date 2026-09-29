"""Month-End Rebalancing Flow (strategy_taa_month_end_rebalancing_flow) leakage tests on REAL Norgate data.

  1. Month-table / MOC-schedule invariance to future corporate actions (SPY split, TLT split, TR-only dividend
     factors on the SPY/IEF signal series), k in (40, 0.1, 1.5).
  2. Engine-level invariance (transactions: date, asset, side).
  3. Truncation: month table and decision rows from data ending at T equal the full-history rows, >= 6 T.
  4. Calendar hindsight: XNYS ad-hoc closures (not known at the measure date) vs a regular-holiday calendar.
  5. Accounting units: per-share commission on adjusted units (historical_share_units_bool=True converts only the
     commission for explicit share orders); TLT borrow collateral in raw units.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from def_common import (BOOK_END, BOOK_START, FACTORS, OUT, cached, dump_json, harness, metric_rows,
                        per_share_fee_effect, rescale_frame)

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from alpha.engine.backtest import run_daily
from strategies.taa_beyond_6040 import strategy_taa_month_end_rebalancing_flow as m

CFG = m.DEFAULT_CONFIG


def load_pricing():
    return cached("eom_pricing", lambda: m.get_month_end_flow_data(CFG))


def signal_tables(pricing_df):
    s = m.MonthEndRebalancingFlowStrategy(CFG)
    feat = s.compute_signals(pricing_df)
    ns = m.SIGNAL_NAMESPACE_STR
    dec = feat.loc[feat[(ns, "rebalance_bool")] == 1.0, [(ns, "SPY"), (ns, "TLT")]]
    dec.columns = ["SPY", "TLT"]
    return s.month_table_df.set_index("month_period"), dec


def compare_signal_tables(ref, cand, end_ts=None):
    (mt_r, dec_r), (mt_c, dec_c) = ref, cand
    if end_ts is not None:
        mt_r = mt_r[mt_r["measure_date"] <= end_ts]
        mt_c = mt_c[mt_c["measure_date"] <= end_ts]
        dec_r = dec_r.loc[:end_ts]
        dec_c = dec_c.loc[:end_ts]
    common = mt_r.index.intersection(mt_c.index)
    b_r = mt_r.loc[common, "bucket_ief_measure_causal_int"].fillna(-1)
    b_c = mt_c.loc[common, "bucket_ief_measure_causal_int"].fillna(-1)
    dcols = ["prev_eom_date", "measure_date", "final_fill_date", "early_fill_date", "exit_fill_date"]
    date_diff = (mt_r.loc[common, dcols] != mt_c.loc[common, dcols]).any(axis=1)
    p_diff = float((mt_r.loc[common, "pressure_ief_measure_bps_float"] - mt_c.loc[common, "pressure_ief_measure_bps_float"]).abs().max())
    wres = harness.compare_weight_frames(dec_r, dec_c, atol_float=1e-12)
    out = {"n_months_ref": int(len(mt_r)), "n_months_cand": int(len(mt_c)), "n_months_common": int(len(common)),
           "n_bucket_diff": int((b_r != b_c).sum()), "n_date_diff": int(date_diff.sum()),
           "max_abs_pressure_diff_bps": p_diff, "decision_rows": wres}
    out["passed"] = (out["n_bucket_diff"] == 0 and out["n_date_diff"] == 0 and len(mt_r) == len(mt_c)
                     and wres["n_dates_different"] == 0 and wres["reference_only_dates"] == 0
                     and wres["candidate_only_dates"] == 0)
    return out


def run_engine(pricing_df, historical_units=False):
    cal = pricing_df.index[pricing_df.index >= pd.Timestamp(CFG.backtest_start_date_str)]
    s = m.MonthEndRebalancingFlowStrategy(CFG)
    if historical_units:
        s.historical_share_units_bool = True
    run_daily(s, pricing_df, calendar=cal, show_progress=False, show_signal_progress_bool=False)
    return s


def main():
    t0 = time.time()
    log = harness.ResultLog("eom_flow")
    px = load_pricing()
    print("loaded", px.shape, px.index[0], px.index[-1], sorted({c[0] for c in px.columns}))
    base_sig = signal_tables(px)
    base_sig[0].to_csv(OUT / "eom_month_table_full.csv")

    # ---- 1. signal invariance ---------------------------------------------------------------------------
    cases = {"SPY_split": ["SPY", "FLOW_TR_SPY"], "TLT_split": ["TLT"], "SPY_tr_dividend_only": ["FLOW_TR_SPY"],
             "IEF_tr_factor": ["FLOW_TR_IEF"]}
    for cname, ns_list in cases.items():
        for k in FACTORS:
            res = compare_signal_tables(base_sig, signal_tables(rescale_frame(px, ns_list, k)))
            log.add("invariance_signal", f"{cname}_k{k}", res["passed"], res)

    # ---- 3. truncation ---------------------------------------------------------------------------------
    mt = base_sig[0]
    row = mt.loc["2021-05"]
    cutoffs = {
        "measure_date_2021-05": row["measure_date"], "final_fill_2021-05": row["final_fill_date"],
        "day_before_month_end_2021-05-27": pd.Timestamp("2021-05-27"), "month_end_holiday_2021-05-28": pd.Timestamp("2021-05-28"),
        "month_end_weekend_2020-05-29": pd.Timestamp("2020-05-29"), "mid_month_2013-10-22": pd.Timestamp("2013-10-22"),
        "month_end_weekend_2022-12-30": pd.Timestamp("2022-12-30"), "exit_fill_2023-01": mt.loc["2022-12", "exit_fill_date"],
        "mid_month_2008-10-15": pd.Timestamp("2008-10-15"),
    }
    for name, T in cutoffs.items():
        T = pd.Timestamp(T)
        res = compare_signal_tables(base_sig, signal_tables(px.loc[:T]), end_ts=T)
        log.add("truncation_prefix", name, res["passed"], res)
    print(f"truncation done {time.time()-t0:.1f}s")

    # ---- 4. calendar hindsight: ad-hoc closures ---------------------------------------------------------
    cal = xcals.get_calendar("XNYS", start="2002-07-01", end="2026-12-31")
    adhoc = pd.DatetimeIndex([d for d in cal.adhoc_holidays if pd.Timestamp("2002-07-01") <= d <= pd.Timestamp("2026-12-31")])
    adhoc = adhoc[adhoc.dayofweek < 5]
    orig_fn = m.exchange_session_idx

    def regular_only(end_date_ts):
        sess = orig_fn(end_date_ts)
        return sess.union(adhoc[adhoc <= sess[-1]])

    m.exchange_session_idx = regular_only
    try:
        mt_reg = m.build_month_table_df(pd.DataFrame({a: px[(f"FLOW_TR_{a}", "Close")] for a in m.SIGNAL_ASSET_TUPLE}).reindex(
            orig_fn(px.index[-1]).union(adhoc)).loc[:px.index[-1]])
    finally:
        m.exchange_session_idx = orig_fn
    mt_reg = mt_reg.set_index("month_period")
    dcols = ["measure_date", "final_fill_date", "early_fill_date", "exit_fill_date"]
    diff_rows = []
    for mp in mt.index.intersection(mt_reg.index):
        a, b = mt.loc[mp], mt_reg.loc[mp]
        if (a[dcols] != b[dcols]).any() or a["bucket_ief_measure_causal_int"] != b["bucket_ief_measure_causal_int"]:
            diff_rows.append({"month": mp, **{f"{c}_actual": a[c] for c in dcols}, **{f"{c}_regular": b[c] for c in dcols},
                              "bucket_actual": a["bucket_ief_measure_causal_int"], "bucket_regular": b["bucket_ief_measure_causal_int"]})
    cal_diff = pd.DataFrame(diff_rows)
    cal_diff.to_csv(OUT / "eom_adhoc_closure_schedule_diff.csv", index=False)
    log.add("calendar_hindsight", "adhoc_closures_change_schedule", len(cal_diff) == 0,
            {"adhoc_closures": [d.date().isoformat() for d in adhoc], "n_months_differ": int(len(cal_diff)),
             "months": cal_diff["month"].tolist() if len(cal_diff) else []})

    # ---- 2 & 5. engine -------------------------------------------------------------------------------
    base = run_engine(px)
    base_tx = harness.transaction_decisions(base.get_transactions())
    for cname, ns_list, k in (("TLT_split", ["TLT"], 40.0), ("SPY_split", ["SPY", "FLOW_TR_SPY"], 0.1),
                              ("IEF_tr_factor", ["FLOW_TR_IEF"], 1.5)):
        s = run_engine(rescale_frame(px, ns_list, k))
        res = harness.compare_transaction_decisions(base_tx, harness.transaction_decisions(s.get_transactions()))
        log.add("invariance_engine", f"{cname}_k{k}", res["n_only_reference"] == 0 and res["n_only_candidate"] == 0, res)
    print(f"engine invariance done {time.time()-t0:.1f}s")

    windows = {"full": (None, None), "book": (BOOK_START, BOOK_END)}
    rows = metric_rows("baseline_adjusted_units", base.results["daily_returns"], windows)
    hist = run_engine(px, historical_units=True)
    rows += metric_rows("commission_on_raw_units", hist.results["daily_returns"], windows)
    fee = per_share_fee_effect(base.get_transactions(), px, base._commission_per_share, base._commission_minimum)
    fee.pop("_frame")
    fee["commission_total_model_run"] = float(base.get_transactions()["commission"].sum())
    fee["commission_total_raw_units_run"] = float(hist.get_transactions()["commission"].sum())
    kr = {a: {"k_min": float((px[(a, "Unadjusted Close")] / px[(a, "Close")]).min()),
              "k_max": float((px[(a, "Unadjusted Close")] / px[(a, "Close")]).max())} for a in m.TRADED_ASSET_TUPLE}
    bf = base.borrow_fee_df.copy()
    bor = {}
    if len(bf):
        kser = px[("TLT", "Unadjusted Close")] / px[("TLT", "Close")]
        closes = px[("TLT", "Close")]
        bf["k"] = bf["accrual_start_date"].map(kser)
        bf["close"] = bf["accrual_start_date"].map(closes)
        bf["fee_raw"] = (bf["short_share_float"].abs() / bf["k"]) * np.ceil(1.02 * bf["close"] * bf["k"]) * 0.01 * bf["calendar_day_count_int"] / 360.0
        bor = {"fee_model_total": float(bf["borrow_fee_float"].sum()), "fee_raw_units_total": float(bf["fee_raw"].sum())}
    pd.DataFrame(rows).to_csv(OUT / "eom_metrics.csv", index=False)
    dump_json({"fee_units": fee, "k_range": kr, "tlt_borrow_units": bor}, "eom_accounting_units.json")
    log.save("def_eom_flow")
    print(pd.DataFrame(rows).to_string())
    print(f"done {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
