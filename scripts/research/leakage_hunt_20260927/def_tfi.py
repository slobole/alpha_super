"""Tactical Fixed Income L14 (strategy_taa_tactical_fixed_income_ief_lqd) leakage tests.

  1. Hash/contract check of the module's own pipeline on today's data.
  2. ETF future-action invariance: rescale IEF / LQD (k = 40, 0.1, 1.5) -> identical month-end weight table (the
     signal is FRED-only) and identical engine transaction decisions.
  3. Truncation: decisions from FRED observations <= T and sessions <= T (+ the next XNYS session from the known
     calendar) equal full-history decisions, >= 6 cut-offs; plus a future-perturbation test (all FRED observations
     dated >= T replaced by noise).
  4. ALFRED vintage replay: at each month-end decision T recompute the whole rule from the vintage as it existed on
     T (optimistic: same-day vintage) and on the previous session (conservative); count flips vs the frozen
     contract; re-run the engine with real-time weights.  Also H.15 / Moody's publication-day evidence.
  5. Decision dates vs XNYS month-ends.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from def_common import (BOOK_END, BOOK_START, FACTORS, OUT, cached, dump_json, harness, metric_rows,
                        xnys_sessions, month_end_sessions)

import numpy as np
import pandas as pd

from strategies.taa_beyond_6040 import strategy_taa_tactical_fixed_income_ief_lqd as m
from strategies.taa_df.strategy_taa_df import load_execution_price_df

CFG = m.DEFAULT_CONFIG
SERIES = list(m.FRED_SERIES_ID_TUPLE)


def load_inputs():
    def _load():
        px = load_execution_price_df(tradeable_asset_list=CFG.tradeable_asset_tuple, benchmark_list=CFG.benchmark_tuple,
                                     start_date_str=CFG.price_start_date_str, end_date_str=CFG.end_date_str)
        common = px.loc[:, [(a, "Close") for a in CFG.tradeable_asset_tuple]].notna().all(axis=1)
        px = px.loc[px.index[common]].copy()
        return px
    px = cached("tfi_pricing", _load)
    yield_df, snaps = m.load_frozen_yield_panel(CFG)
    return px, yield_df, snaps


def weights_from(yield_df, session_index, last_month):
    sig, w = m.build_month_end_signal_and_weight_df(yield_df=yield_df, session_index=session_index,
                                                    last_complete_signal_month_str=last_month)
    return sig, w


def run_engine(px, sig, w, cash_ret, snaps):
    return m._run_strategy(config_obj=CFG, execution_price_df=px, signal_df=sig, rebalance_weight_df=w,
                           cash_return_ser=cash_ret, fred_snapshot_tuple=snaps, backtest_start_date_str="2002-08-01",
                           end_date_str=None, show_progress_bool=False)


def main():
    t0 = time.time()
    log = harness.ResultLog("tactical_fi")
    px, yield_df, snaps = load_inputs()
    sessions = pd.DatetimeIndex(px.index)
    # ---- 1. contract hash -----------------------------------------------------------------------------
    sig, w = weights_from(yield_df, sessions, CFG.last_complete_signal_month_str)
    contract_sha = m.canonical_dataframe_sha256_str(m.build_canonical_signal_contract_df(sig, w))
    norgate_sha = {s: m.canonical_dataframe_sha256_str(px[s]) for s in m.FROZEN_NORGATE_SHA256_BY_SYMBOL_DICT}
    log.add("contract", "signal_contract_sha_matches_frozen", contract_sha == m.FROZEN_SIGNAL_CONTRACT_SHA256_STR,
            {"sha": contract_sha, "n_decisions": int(len(sig))})
    log.add("contract", "norgate_sha_matches_frozen", norgate_sha == m.FROZEN_NORGATE_SHA256_BY_SYMBOL_DICT,
            {"actual": norgate_sha})
    sig.to_csv(OUT / "tfi_signal_table_current_vintage.csv")
    w.to_csv(OUT / "tfi_weight_table_current_vintage.csv")
    cash_ret = m.build_causal_cash_return_ser(sessions, snaps[SERIES.index("DGS3MO")].value_ser)

    # ---- 5. decision dates vs XNYS month-ends --------------------------------------------------------------
    xs = xnys_sessions("2002-01-01", "2026-12-31")
    me = month_end_sessions(xs[(xs >= sig.index[0]) & (xs <= pd.Timestamp("2026-07-31"))])
    miss = me.difference(sig.index)
    extra = sig.index.difference(me)
    log.add("calendar", "decision_dates_equal_xnys_month_ends", len(miss) == 0 and len(extra) == 0,
            {"xnys_not_decision": [d.date().isoformat() for d in miss], "decision_not_xnys": [d.date().isoformat() for d in extra]})
    not_xnys = sessions.difference(xs)
    log.add("calendar", "price_index_subset_of_xnys", len(not_xnys[not_xnys >= "2002-08-01"]) == 0,
            {"rows_not_xnys": [d.date().isoformat() for d in not_xnys[not_xnys >= "2002-08-01"]][:20]})

    # ---- 2. ETF invariance ----------------------------------------------------------------------------
    base = run_engine(px, sig, w, cash_ret, snaps)
    base_tx = harness.transaction_decisions(base.get_transactions())
    windows = {"full": (None, None), "book": (BOOK_START, BOOK_END)}
    metric_list = metric_rows("baseline_current_vintage", base.results["daily_returns"], windows)
    for a in CFG.tradeable_asset_tuple:
        for k in FACTORS:
            pxk = harness.rescale_symbol_history(px, a, k)
            sigk, wk = weights_from(yield_df, pd.DatetimeIndex(pxk.index), CFG.last_complete_signal_month_str)
            wres = harness.compare_weight_frames(w[["IEF", "LQD", "Cash"]], wk[["IEF", "LQD", "Cash"]])
            s = run_engine(pxk, sigk, wk, cash_ret, snaps)
            tres = harness.compare_transaction_decisions(base_tx, harness.transaction_decisions(s.get_transactions()))
            tres["weight_table"] = wres
            ok = wres["n_dates_different"] == 0 and tres["n_only_reference"] == 0 and tres["n_only_candidate"] == 0
            log.add("invariance", f"{a}_k{k}", ok, tres)
    print(f"invariance done {time.time()-t0:.1f}s")

    # ---- 3. truncation + future perturbation ------------------------------------------------------------
    cutoffs = {
        "mid_month_2013-10-22": "2013-10-22", "mid_month_2020-03-16": "2020-03-16",
        "month_end_weekend_2020-05-29": "2020-05-29", "month_end_holiday_2021-05-28": "2021-05-28",
        "day_before_month_end_2021-05-27": "2021-05-27", "month_end_plain_2008-09-30": "2008-09-30",
        "month_end_weekend_2022-12-30": "2022-12-30", "month_end_2008-12-31": "2008-12-31",
    }
    rng = np.random.default_rng(20260927)
    for name, T in cutoffs.items():
        T = pd.Timestamp(T)
        nxt = xs[xs > T][0]
        sess_T = pd.DatetimeIndex(list(sessions[sessions <= T]) + [nxt])
        is_me = T in me
        last_month = str(T.to_period("M")) if is_me else str(T.to_period("M") - 1)
        # (a) truncation: FRED obs dated <= T (obs T is not yet published at 17:15 on T; included deliberately)
        sig_t, w_t = weights_from(yield_df.loc[:T], sess_T, last_month)
        cols = ["observation_date", "term_spread_float", "credit_spread_float", "term_threshold_float",
                "credit_threshold_float", "term_state_float", "credit_state_float"]
        ref = sig.loc[:T, cols]
        cand = sig_t.loc[:T, cols]
        same = ref.index.equals(cand.index) and np.allclose(ref.iloc[:, 1:].to_numpy(float), cand.iloc[:, 1:].to_numpy(float), atol=1e-12) \
            and (ref["observation_date"] == cand["observation_date"]).all()
        log.add("truncation_prefix", name, bool(same), {"n_rows": int(len(ref)), "is_month_end": bool(is_me),
                                                        "last_row_full": ref.iloc[-1].astype(str).to_dict(),
                                                        "last_row_trunc": cand.iloc[-1].astype(str).to_dict()})
        # (b) future perturbation: every FRED observation dated >= T replaced by noise
        yp = yield_df.copy()
        fut = yp.index >= T
        yp.loc[fut] = rng.uniform(0.0, 15.0, size=(int(fut.sum()), yp.shape[1]))
        sig_p, _ = weights_from(yp, sessions, CFG.last_complete_signal_month_str)
        same_p = np.allclose(sig.loc[:T, cols[1:]].to_numpy(float), sig_p.loc[:T, cols[1:]].to_numpy(float), atol=1e-12)
        log.add("future_perturbation", name, bool(same_p), {"n_rows": int(len(sig.loc[:T]))})
    print(f"truncation done {time.time()-t0:.1f}s")

    # ---- 4. ALFRED vintages: see def_tfi_vintage.py (window-batched real-time replay) ----------------
    pd.DataFrame(metric_list).to_csv(OUT / "tfi_metrics.csv", index=False)
    log.save("def_tactical_fi")
    print(pd.DataFrame(metric_list).to_string())
    print(f"done {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
