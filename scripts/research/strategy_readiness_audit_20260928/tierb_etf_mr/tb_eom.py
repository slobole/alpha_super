"""EOM flow (strategy_taa_month_end_rebalancing_flow): BC checks on REAL Norgate data (research-only).

  rowt      A3 row-T truncation: compute_signals(data <= T) vs compute_signals(full) for every feature row <= T,
            INCLUDING row T, plus the month table rows with measure_date <= T.  >= 12 cut-offs.
            A4 positive control: a planted one-session leak (pressure read at the FILL close [-6] while the recorded
            measure date stays [-7], with an as-of fallback so it never crashes) must be caught at row T.
  split     A2 future-split invariance: SPY (+ its TR namespace), TLT, IEF (TR) x k in {40, 0.1, 1.5}: month table,
            feature rows and engine decisions.  A4 control: pressure from price DIFFERENCES (scale-dependent).
  calendar  Ad-hoc closure hindsight (notice-time replay), early-close fills, observed-row provenance, causality of
            every decision row (decision = session before fill; measure <= decision; sizing close = decision close).
  account   Short dividend debit on TLT, borrow sensitivity (0 / 1% / 3%), idle-cash and short-proceeds interest
            bound (DTB3), unfinanced negative cash bound, raw-unit fees (historical_share_units_bool).

Usage: uv run python tb_eom.py <rowt|split|calendar|account>
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import tb_common as tc
from tb_common import eom_mod

NS = eom_mod.SIGNAL_NAMESPACE_STR
_ORIGINAL_BUILD_MONTH_TABLE = eom_mod.build_month_table_df


def _features(pricing: pd.DataFrame, cls=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    strat = (cls or eom_mod.MonthEndRebalancingFlowStrategy)(tc.config_for("eom"))
    out = strat.compute_signals(pricing.copy())
    return out.loc[:, [c for c in out.columns if c[0] == NS]], strat.month_table_df


# ------------------------------------------------------------------ A4 planted leak
def _leak_build_month_table_df(total_return_close_df: pd.DataFrame) -> pd.DataFrame:
    """Production code with ONE change: pressure uses the close of the FILL session [-6] (as-of fallback to the
    last available close when the data end earlier).  The recorded measure_date stays [-7]."""
    price_idx = total_return_close_df.index
    session_idx = eom_mod.exchange_session_idx(price_idx[-1])
    month_period_idx = session_idx.to_period("M")
    prior, rows = [], []
    for month_period in pd.period_range("2002-08", price_idx[-1].to_period("M"), freq="M"):
        msi = session_idx[month_period_idx == month_period]
        if len(msi) < 9:
            continue
        measure_ts = msi[-7]
        if measure_ts > price_idx[-1]:
            continue
        leak_ts = msi[-6]
        prev_ts = session_idx[month_period_idx == month_period - 1][-1]
        avail = total_return_close_df.loc[:leak_ts]
        leak_row = avail.iloc[-1]
        g_spy = float(leak_row["SPY"] / total_return_close_df.loc[prev_ts, "SPY"])
        g_ief = float(leak_row["IEF"] / total_return_close_df.loc[prev_ts, "IEF"])
        bw = 0.4 * g_ief / (0.6 * g_spy + 0.4 * g_ief)
        p = 10_000.0 * (0.4 - bw)
        b = eom_mod.causal_bucket_float(p, prior)
        nxt = session_idx[month_period_idx == month_period + 1]
        rows.append({"month_period": str(month_period), "prev_eom_date": prev_ts, "measure_date": measure_ts,
                     "final_fill_date": msi[-6], "early_fill_date": msi[-1], "exit_fill_date": nxt[4],
                     "pressure_ief_measure_bps_float": p, "bucket_ief_measure_causal_int": b,
                     "prior_month_count_int": len(prior)})
        prior.append(p)
    return pd.DataFrame(rows)


def _diff_pressure_build_month_table_df(total_return_close_df: pd.DataFrame) -> pd.DataFrame:
    """A2 positive control: growth from a DOLLAR difference (g = 1 + dP/100), which depends on the price scale."""
    tr = total_return_close_df.copy()
    synth = np.exp((tr.diff().fillna(0.0) / 100.0).cumsum())  # positive, but depends on the dollar scale
    return _ORIGINAL_BUILD_MONTH_TABLE(synth)


class _PatchedMonthTable(eom_mod.MonthEndRebalancingFlowStrategy):
    builder = None

    def compute_signals(self, pricing_data_df):
        original = eom_mod.build_month_table_df
        eom_mod.build_month_table_df = type(self).builder
        try:
            return super().compute_signals(pricing_data_df)
        finally:
            eom_mod.build_month_table_df = original


class LeakEOM(_PatchedMonthTable):
    builder = staticmethod(_leak_build_month_table_df)


class DiffPressureEOM(_PatchedMonthTable):
    builder = staticmethod(_diff_pressure_build_month_table_df)


def _cutoffs(pricing: pd.DataFrame) -> dict[str, str]:
    sess = eom_mod.exchange_session_idx(pricing.index[-1])
    per = sess.to_period("M")

    def month(m):
        return sess[per == pd.Period(m)]

    aug, sep = month("2026-08"), month("2026-09")
    oct12, mar20 = month("2012-10"), month("2020-03")
    return {
        "measure_minus7_2026-08": str(aug[-7].date()),
        "early_decision_minus2_2026-08": str(aug[-2].date()),
        "exit_decision_idx3_2026-09": str(sep[3].date()),
        "mid_month_2026-09-15": "2026-09-15",
        "month_end_weekend_2026-05-29": "2026-05-29",
        "month_end_before_holiday_2025-12-31": "2025-12-31",
        "first_session_2026-09-01": "2026-09-01",
        "last_completed_month_end_2026-08-31": "2026-08-31",
        "current_partial_month_2026-09-25": "2026-09-25",
        "half_day_2024-11-29": "2024-11-29",
        "sandy_measure_2012-10": str(oct12[-7].date()),
        "gap_day_decision_2020-03-23": "2020-03-23",
        "measure_minus7_2020-03": str(mar20[-7].date()),
        "measure_minus7_2008-09": str(month("2008-09")[-7].date()),
    }


def _compare_rows(full_f: pd.DataFrame, trunc_f: pd.DataFrame, cut_ts) -> dict:
    a = full_f.loc[:cut_ts]
    b = trunc_f.loc[:cut_ts]
    same_index = a.index.equals(b.index)
    arr_a, arr_b = a.to_numpy(float), b.reindex(a.index).to_numpy(float)
    diff_mask = ~((arr_a == arr_b) | (np.isnan(arr_a) & np.isnan(arr_b)))
    row_t_diff = int(diff_mask[-1].sum()) if len(diff_mask) else 0
    return {"same_index": same_index, "n_values": int(arr_a.size), "n_diff": int(diff_mask.sum()),
            "row_T_diff": row_t_diff, "row_T_rebalance_full": float(a.iloc[-1][(NS, "rebalance_bool")])}


def _compare_month_tables(full_mt: pd.DataFrame, trunc_mt: pd.DataFrame, cut_ts) -> dict:
    fm = full_mt[full_mt["measure_date"] <= cut_ts].reset_index(drop=True)
    tm = trunc_mt[trunc_mt["measure_date"] <= cut_ts].reset_index(drop=True)
    if len(fm) != len(tm):
        return {"rows_full": len(fm), "rows_trunc": len(tm), "n_diff": -1}
    cols = ["pressure_ief_measure_bps_float", "bucket_ief_measure_causal_int"]
    fa, ta = fm[cols].to_numpy(float), tm[cols].to_numpy(float)
    d = int((~((fa == ta) | (np.isnan(fa) & np.isnan(ta)))).sum())
    p_last = float(abs(fa[-1, 0] - ta[-1, 0])) if len(fa) else 0.0
    dates = ["measure_date", "final_fill_date", "early_fill_date", "exit_fill_date"]
    dd = int((fm[dates] != tm[dates]).to_numpy().sum())
    leak_active = bool(len(fm) and pd.Timestamp(fm.iloc[-1]["final_fill_date"]) > cut_ts)
    return {"rows": len(fm), "n_value_diff": d, "n_date_diff": dd, "last_row_abs_pressure_diff": p_last,
            "last_month": str(fm.iloc[-1]["month_period"]) if len(fm) else None,
            "cut_between_measure_and_fill": leak_active}


def run_rowt() -> None:
    pricing = tc.load_pricing("eom")
    cuts = _cutoffs(pricing)
    full_f, full_mt = _features(pricing)
    leak_f, leak_mt = _features(pricing, LeakEOM)
    rows = []
    for label, cut in cuts.items():
        cut_ts = pd.Timestamp(cut)
        tf, tmt = _features(pricing.loc[:cut_ts])
        rec = {"case": label, "cut": cut, **_compare_rows(full_f, tf, cut_ts),
               "month_table": _compare_month_tables(full_mt, tmt, cut_ts)}
        rec["passed"] = rec["n_diff"] == 0 and rec["same_index"] and rec["month_table"].get("n_value_diff") == 0 \
            and rec["month_table"].get("n_date_diff") == 0
        lf, lmt = _features(pricing.loc[:cut_ts], LeakEOM)
        lrec = _compare_rows(leak_f, lf, cut_ts)
        lmt_cmp = _compare_month_tables(leak_mt, lmt, cut_ts)
        rec["leak_control"] = {**lrec, "month_table": lmt_cmp,
                               "leak_reads_beyond_T": lmt_cmp.get("cut_between_measure_and_fill"),
                               "caught": bool(lrec["n_diff"] > 0 or lmt_cmp.get("n_value_diff", 1) != 0)}
        rows.append(rec)
        print(label, cut, "pass", rec["passed"], "leak caught", rec["leak_control"]["caught"], flush=True)
    # Where the leak is caught, is it caught AT row T / the latest month row?  (Leak affects only the month whose
    # measure date is the cut; earlier months have their fill close in the truncated data.)
    tc.write_json("eom_rowt.json", {"n_cases": len(rows), "n_pass": sum(r["passed"] for r in rows),
                                     "n_leak_caught": sum(r["leak_control"]["caught"] for r in rows),
                                     "n_leak_active_cuts": sum(bool(r["leak_control"]["leak_reads_beyond_T"]) for r in rows),
                                     "n_leak_caught_where_active": sum(r["leak_control"]["caught"] for r in rows
                                                                       if r["leak_control"]["leak_reads_beyond_T"]),
                                     "rows": rows})


def run_split() -> None:
    pricing = tc.load_pricing("eom")
    ns_map = {"SPY": ["SPY", "FLOW_TR_SPY"], "TLT": ["TLT"], "IEF": ["FLOW_TR_IEF"]}
    base_f, base_mt = _features(pricing)
    base_run = tc.run("eom", pricing)
    base_dec = base_run.decision_df.copy()
    base_m = tc.metric_pair(base_run)
    pc_f, pc_mt = _features(pricing, DiffPressureEOM)
    base_hsu = tc.run("eom", pricing, hsu=True)
    base_hsu_m = tc.metric_pair(base_hsu)
    rows = []
    for sym, nss in ns_map.items():
        for k in (40.0, 0.1, 1.5):
            cand = tc.rescale_symbol(pricing, nss, k)
            f, mt = _features(cand)
            n_feat_diff = int((f.to_numpy(float) != base_f.to_numpy(float)).sum())
            bucket_diff = _nan_ne(mt["bucket_ief_measure_causal_int"], base_mt["bucket_ief_measure_causal_int"])
            p_rel = float(np.nanmax(np.abs(mt["pressure_ief_measure_bps_float"].to_numpy() -
                                           base_mt["pressure_ief_measure_bps_float"].to_numpy())))
            r = tc.run("eom", cand)
            dec = r.decision_df.copy()
            same_dates = dec["fill_date"].equals(base_dec["fill_date"])
            # target weights implied by the share orders: q * Close_prev / NAV_prev (rounding differs with k)
            w_base = _implied_weights(base_dec, pricing)
            w_cand = _implied_weights(dec, cand)
            w_max = float(np.nanmax(np.abs(w_cand - w_base))) if same_dates else float("nan")
            m = tc.metric_pair(r)
            rec = {"symbol": sym, "k": k, "feature_cells_diff": n_feat_diff, "bucket_diff": bucket_diff,
                   "max_abs_pressure_diff_bps": p_rel, "engine_same_decision_dates": bool(same_dates),
                   "engine_max_abs_weight_diff": w_max,
                   "cagr_full_diff_pp": m["full"]["cagr_pct"] - base_m["full"]["cagr_pct"]}
            pf, pmt = _features(cand, DiffPressureEOM)
            rec["control_bucket_diff"] = _nan_ne(pmt["bucket_ief_measure_causal_int"], pc_mt["bucket_ief_measure_causal_int"])
            rh = tc.run("eom", cand, hsu=True)
            mh = tc.metric_pair(rh)
            rec["engine_raw_unit_fees_same_decision_dates"] = bool(rh.decision_df["fill_date"].equals(base_hsu.decision_df["fill_date"]))
            rec["cagr_full_diff_pp_raw_unit_fees"] = mh["full"]["cagr_pct"] - base_hsu_m["full"]["cagr_pct"]
            rec["passed"] = n_feat_diff == 0 and bucket_diff == 0 and same_dates
            rows.append(rec)
            print(rec, flush=True)
    tc.write_json("eom_split.json", {"n_cases": len(rows), "n_pass": sum(r["passed"] for r in rows),
                                      "n_control_caught": sum(r["control_bucket_diff"] > 0 for r in rows),
                                      "rows": rows})


def _nan_ne(a: pd.Series, b: pd.Series) -> int:
    x, y = a.to_numpy(float), b.to_numpy(float)
    return int((~((x == y) | (np.isnan(x) & np.isnan(y)))).sum())


def _implied_weights(dec: pd.DataFrame, pricing: pd.DataFrame) -> np.ndarray:
    out = []
    for _, row in dec.iterrows():
        d = pd.Timestamp(row["decision_date"])
        nav = float(row["sizing_nav_float"])
        out.append([float(row[a]) * float(pricing.loc[d, (a, "Close")]) / nav for a in ("SPY", "TLT")])
    return np.asarray(out)


# ------------------------------------------------------------------ calendar
ANNOUNCED = {  # public record (cited, not re-verified here): date the closure became known
    "2004-06-11": "2004-06-07", "2007-01-02": "2006-12-27", "2012-10-29": "2012-10-28",
    "2012-10-30": "2012-10-29", "2018-12-05": "2018-12-01", "2025-01-09": "2024-12-30",
}


def run_calendar() -> None:
    pricing = tc.load_pricing("eom")
    strat = tc.run("eom", pricing)
    mt, dec = strat.month_table_df.copy(), strat.decision_df.copy()
    sess = eom_mod.exchange_session_idx(pricing.index[-1])
    out: dict = {}
    # 1. causality of every decision row
    bad = []
    for _, row in dec.iterrows():
        fill, d = pd.Timestamp(row["fill_date"]), pd.Timestamp(row["decision_date"])
        if sess[sess.get_loc(fill) - 1] != d:
            bad.append(str(fill.date()))
    sched = strat.order_schedule_df.reset_index()
    late = int((sched["measure_date"] > sched["fill_date"].map(lambda x: sess[sess.get_loc(x) - 1])).sum())
    out["decision_rows"] = int(len(dec))
    out["decision_not_prior_session"] = bad
    out["measure_after_decision_rows"] = late
    # sizing uses the decision close: re-derive shares from decision close and compare
    mism = 0
    for _, row in dec.iterrows():
        d = pd.Timestamp(row["decision_date"])
        nav = float(row["sizing_nav_float"])
        sched_row = strat.order_schedule_df.loc[pd.Timestamp(row["fill_date"])]
        sched_row = sched_row.iloc[-1] if isinstance(sched_row, pd.DataFrame) else sched_row
        for a in ("SPY", "TLT"):
            q = int(np.trunc(float(sched_row[a]) * nav / float(pricing.loc[d, (a, "Close")])))
            mism += int(q != int(row[a]))
    out["sizing_from_decision_close_mismatch"] = mism
    # 2. observed rows vs XNYS
    run_idx = pricing.index[pricing.index >= pd.Timestamp("2003-01-02")]
    xnys = sess[(sess >= run_idx[0]) & (sess <= run_idx[-1])]
    out["rows_not_xnys"] = [str(x.date()) for x in run_idx.difference(xnys)]
    out["xnys_missing_rows"] = [str(x.date()) for x in xnys.difference(run_idx)]
    vol0 = {}
    for a in ("SPY", "TLT"):
        v = pricing.loc[run_idx, (a, "Volume")].astype(float)
        vol0[a] = [str(x.date()) for x in v.index[(v <= 0) | v.isna()]]
    out["zero_or_nan_volume_rows"] = vol0
    nan_close = {a: int(pricing.loc[run_idx, (a, "Close")].isna().sum()) for a in ("SPY", "TLT")}
    nan_close.update({ns: int(pricing.loc[run_idx, (ns, "Close")].isna().sum()) for ns in ("FLOW_TR_SPY", "FLOW_TR_IEF")})
    out["nan_close_rows_in_run"] = nan_close
    # 3. early-close fills
    import exchange_calendars as xcals

    cal = xcals.get_calendar("XNYS", start="2002-07-01", end="2026-12-31")
    early = set(pd.DatetimeIndex(cal.early_closes).tz_localize(None).normalize())
    fill_days = pd.to_datetime(dec["fill_date"])
    early_fills = sorted(str(x.date()) for x in fill_days if x in early)
    measure_early = sorted(str(pd.Timestamp(x).date()) for x in mt["measure_date"] if pd.Timestamp(x) in early)
    out["early_close_fill_days"] = early_fills
    out["early_close_measure_days"] = measure_early
    # 4. ad-hoc closure notice-time replay
    tr = pd.DataFrame({a: pricing[(f"FLOW_TR_{a}", "Close")] for a in ("SPY", "IEF")})
    base_mt = eom_mod.build_month_table_df(tr)
    closure_rows = []
    original_fn = eom_mod.exchange_session_idx
    for closure, announced in ANNOUNCED.items():
        cts = pd.Timestamp(closure)

        def patched(end_ts, _c=cts):
            s = original_fn(end_ts)
            return s.union(pd.DatetimeIndex([_c]))

        eom_mod.exchange_session_idx = patched
        try:
            alt = eom_mod.build_month_table_df(tr)
        except ValueError as exc:  # an endpoint would land on the closed day
            alt = None
            closure_rows.append({"closure": closure, "error": str(exc)})
        finally:
            eom_mod.exchange_session_idx = original_fn
        if alt is None:
            continue
        cols = ["measure_date", "final_fill_date", "early_fill_date", "exit_fill_date", "bucket_ief_measure_causal_int"]
        merged = base_mt.merge(alt, on="month_period", suffixes=("_hist", "_notice"))
        for _, r in merged.iterrows():
            changed = [c for c in cols if r[f"{c}_hist"] != r[f"{c}_notice"]]
            if not changed:
                continue
            first_decision = None
            for c in changed:
                if c.endswith("_date") and c != "measure_date":
                    for which in ("_hist", "_notice"):
                        f = pd.Timestamp(r[f"{c}{which}"])
                        dd = sess[sess.get_loc(f) - 1] if f in sess else original_fn(f)[original_fn(f).get_loc(f) - 1]
                        first_decision = dd if first_decision is None else min(first_decision, dd)
                elif c == "measure_date":
                    for which in ("_hist", "_notice"):
                        f = pd.Timestamp(r[f"{c}{which}"])
                        first_decision = f if first_decision is None else min(first_decision, f)
            closure_rows.append({
                "closure": closure, "announced": announced, "month": r["month_period"], "changed": changed,
                **{f"{c}_hist": str(r[f"{c}_hist"]) for c in changed}, **{f"{c}_notice": str(r[f"{c}_notice"]) for c in changed},
                "legs_hist": [eom_mod.target_weight_tuple(r["bucket_ief_measure_causal_int_hist"], l) for l in ("final", "early")],
                "legs_notice": [eom_mod.target_weight_tuple(r["bucket_ief_measure_causal_int_notice"], l) for l in ("final", "early")],
                "earliest_affected_decision": str(first_decision.date()) if first_decision is not None else None,
                "hindsight": bool(first_decision is not None and first_decision < pd.Timestamp(announced)),
            })
    out["adhoc_closure_diffs"] = closure_rows
    # Sandy P&L: hindsight final leg entered on 2012-10-22 MOC instead of 2012-10-24 MOC (TLT long 100%).
    tlt = pricing[("TLT", "Close")].astype(float)
    r_extra = float(tlt.loc["2012-10-24"] / tlt.loc["2012-10-22"] - 1.0)
    row = mt[mt["month_period"] == "2012-10"].iloc[0]
    legs = eom_mod.target_weight_tuple(row["bucket_ief_measure_causal_int"], "final")
    years = (pricing.index[-1] - pd.Timestamp("2003-01-02")).days / 365.25
    out["sandy"] = {"hist_measure": str(row["measure_date"].date()), "hist_final_fill": str(row["final_fill_date"].date()),
                    "final_leg_weights_spy_tlt": legs, "extra_tlt_return_2012_10_22_to_24": r_extra,
                    "nav_effect_once": r_extra * legs[1], "cagr_effect_pp": 100 * np.log1p(r_extra * legs[1]) / years}
    tc.write_json("eom_calendar.json", out)
    print({k: (v if not isinstance(v, list) or len(v) < 8 else f"{len(v)} items") for k, v in out.items()}, flush=True)


# ------------------------------------------------------------------ accounting
def run_account() -> None:
    pricing = tc.load_pricing("eom")
    base = tc.run("eom", pricing)
    out = {"base": tc.metric_pair(base)}
    # short dividend debit on TLT: the ledger rows
    ledger = pd.DataFrame(getattr(base, "_dividend_ledger_row_dict_list", []))
    if len(ledger):
        ledger.to_csv(tc.OUT / "eom_dividend_ledger.csv", index=False)
        short_rows = ledger[ledger["gross_dividend_cash_float"] < 0] if "gross_dividend_cash_float" in ledger else ledger.iloc[0:0]
        out["dividend_events"] = int(len(ledger))
        out["short_dividend_events"] = int(len(short_rows))
        out["short_dividend_gross_debit"] = float(short_rows["gross_dividend_cash_float"].sum()) if len(short_rows) else 0.0
        out["short_dividend_withholding_applied"] = float(short_rows["withholding_cash_float"].sum()) if len(short_rows) else 0.0
        out["ledger_columns"] = list(ledger.columns)
    # borrow sensitivity
    for rate in (0.0, 0.03):
        orig = eom_mod.ANNUAL_BORROW_RATE_FLOAT
        eom_mod.ANNUAL_BORROW_RATE_FLOAT = rate
        try:
            s = tc.run("eom", pricing)
        finally:
            eom_mod.ANNUAL_BORROW_RATE_FLOAT = orig
        out[f"borrow_{rate}"] = {**tc.metric_pair(s), "borrow_total": float(s.borrow_fee_total_float)}
    out["borrow_0.01_total"] = float(base.borrow_fee_total_float)
    # raw-unit fees
    h = tc.run("eom", pricing, hsu=True)
    out["hsu"] = {**tc.metric_pair(h), "commission_total": float(h.get_transactions()["commission"].sum()),
                  "base_commission_total": float(base.get_transactions()["commission"].sum())}
    k = (pricing[("SPY", "Unadjusted Close")] / pricing[("SPY", "Close")]).astype(float)
    kt = (pricing[("TLT", "Unadjusted Close")] / pricing[("TLT", "Close")]).astype(float)
    out["k_range"] = {"SPY": [float(k.min()), float(k.max())], "TLT": [float(kt.min()), float(kt.max())]}
    # cash: idle-cash / short proceeds interest at DTB3 and negative cash financing at DTB3 + 1.5% (bound)
    res = base.results.copy()
    res.index = pd.to_datetime(res.index)
    dtb3 = pd.read_csv(tc.REPO.parent / "1_data" / "DTB3.csv", parse_dates=["observation_date"], na_values=".")
    dtb3 = dtb3.set_index("observation_date")["DTB3"].astype(float).ffill() / 100.0
    rate = dtb3.reindex(res.index, method="ffill").shift(1).fillna(0.0)
    cash = res["cash"].astype(float)
    tv = res["total_value"].astype(float)
    idle = np.minimum(cash.clip(lower=0), tv)          # own equity sitting in cash
    proceeds = (cash - tv).clip(lower=0)                 # short-sale proceeds beyond own equity
    idle_int = idle / tv * rate / 252.0
    proceeds_int = proceeds / tv * rate / 252.0
    neg_cash_cost = (cash.clip(upper=0) / tv * (rate + 0.015) / 252.0)
    years = (res.index[-1] - res.index[0]).days / 365.25
    out["cash"] = {
        "min_cash_frac": float((cash / tv).min()), "days_negative": int((cash < 0).sum()), "days": int(len(cash)),
        "mean_idle_equity_cash_frac": float((idle / tv).mean()),
        "idle_equity_interest_at_dtb3_pp_per_yr": float(idle_int.sum() / years * 100),
        "short_proceeds_interest_at_dtb3_pp_per_yr": float(proceeds_int.sum() / years * 100),
        "negative_cash_financing_at_dtb3_plus_150bp_pp_per_yr": float(neg_cash_cost.sum() / years * 100),
        "last3y_idle_equity_interest_pp_per_yr": float(idle_int.loc["2023-09-25":].sum() / 3.0 * 100),
    }
    tc.write_json("eom_accounting.json", out)
    print(out, flush=True)


if __name__ == "__main__":
    {"rowt": run_rowt, "split": run_split, "calendar": run_calendar, "account": run_account}[sys.argv[1]]()
