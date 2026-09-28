"""A2 future-split invariance, A3 truncation invariance, A4 positive controls for CORE5 (real Norgate data).

Decision object = the engine's daily target-weight table (every decision date), the rebalance-date set, the
long/short states, and the engine fill ledger (date, asset, side). Share counts are adjusted-unit quantities
and legitimately scale with k; their effect is reported as a metric delta, not as a decision difference.

Usage: python a_invariance.py [split|trunc|pc|all]
"""
import pickle
import sys
import time
import json

import numpy as np
import pandas as pd

import core5_common as c

core5 = c.core5

MODE = sys.argv[1] if len(sys.argv) > 1 else "all"
df = c.load_pricing()
with (c.CACHE / "baseline_strategy.pkl").open("rb") as fh:
    base = pickle.load(fh)
base_daily = base["daily"]
base_rebal = base["rebal"]
base_tx = base["tx"]
base_ret = base["results"]["daily_returns"].astype(float)


def tx_keys(tx):
    t = tx.copy()
    t["side"] = np.sign(t["amount"].astype(float)).astype(int)
    return set(zip(pd.to_datetime(t["bar"]), t["asset"], t["side"]))


def compare_weights(a, b, atol=1e-9):
    idx = a.index.intersection(b.index)
    cols = a.columns.union(b.columns)
    x = a.reindex(index=idx, columns=cols).fillna(0.0)
    y = b.reindex(index=idx, columns=cols).fillna(0.0)
    d = (x - y).abs()
    bad = d.max(axis=1) > atol
    sign_x = np.sign(x.drop(columns=["Cash"], errors="ignore").round(12))
    sign_y = np.sign(y.drop(columns=["Cash"], errors="ignore").round(12))
    return {
        "n_dates": int(len(idx)),
        "n_dates_diff_gt_atol": int(bad.sum()),
        "max_abs_diff": float(d.to_numpy().max()) if d.size else 0.0,
        "n_sign_diffs": int((sign_x != sign_y).to_numpy().sum()),
        "first_diff_dates": [str(t.date()) for t in idx[bad][:5]],
        "only_a": int(len(a.index.difference(b.index))),
        "only_b": int(len(b.index.difference(a.index))),
    }


def signal_states(pdf):
    s = core5.AdaptiveMacroCore5Strategy()
    sig = s.compute_signals(pdf)
    cols = [
        (core5.signal_namespace_str(a), f)
        for a in core5.RISK_ASSET_TUPLE
        for f in ("long_state_ser", "short_state_ser")
    ]
    cols += [
        (core5.PORTFOLIO_NAMESPACE_STR, core5.MONTH_END_REBALANCE_FIELD_STR),
        (core5.PORTFOLIO_NAMESPACE_STR, core5.LONG_STATE_CHANGED_FIELD_STR),
    ]
    return sig, sig.loc[:, cols]


# ---------------- A2: future split invariance -----------------
if MODE in ("all", "split"):
    rows = []
    for sym in ("DBC", "BIL", "SPY", "GLD"):
        for k in (40.0, 0.1, 1.5):
            if sym == "GLD" and k != 40.0:
                continue
            t0 = time.time()
            pdf = c.rescale_symbol(df, sym, k)
            ns = core5.signal_namespace_str(sym)
            if (ns, "Close") in pdf.columns:
                pdf = c.rescale_symbol(pdf, ns, k)
            s = c.run_backtest(pdf, 100_000.0)
            r = s.results["daily_returns"].astype(float)
            wcmp = compare_weights(base_daily, s.daily_target_weights)
            rebal_same = set(base_rebal.index) == set(s.rebalance_target_weight_df.index)
            kb, kc = tx_keys(base_tx), tx_keys(s.get_transactions())
            mb, mc = c.metrics(base_ret), c.metrics(r)
            mb2, mc2 = c.metrics(base_ret, "2012-10-02"), c.metrics(r, "2012-10-02")
            row = {
                "symbol": sym, "k": k, "weights": wcmp, "rebalance_dates_identical": rebal_same,
                "fills_only_base": len(kb - kc), "fills_only_case": len(kc - kb),
                "d_cagr_pp_full": mc["cagr_pct"] - mb["cagr_pct"], "d_sharpe_full": mc["sharpe"] - mb["sharpe"],
                "d_maxdd_pp_full": mc["maxdd_pct"] - mb["maxdd_pct"],
                "d_cagr_pp_2012": mc2["cagr_pct"] - mb2["cagr_pct"], "d_sharpe_2012": mc2["sharpe"] - mb2["sharpe"],
                "commission_case": float(s.get_transactions()["commission"].sum()),
                "commission_base": float(base_tx["commission"].sum()),
                "borrow_case": float(s.borrow_fee_total_float),
                "elapsed_s": time.time() - t0,
            }
            row["decision_pass"] = bool(rebal_same and wcmp["n_sign_diffs"] == 0 and wcmp["max_abs_diff"] < 1e-6)
            print(json.dumps(row, default=str), flush=True)
            rows.append(row)
    c.dump(rows, "a2_split_invariance.json")

# ---------------- A3: truncation invariance -----------------
CUTOFFS = {
    "mid_month_2015-08-24": "2015-08-24",
    "mid_month_2020-03-16": "2020-03-16",
    "month_end_weekend_2019-08-30": "2019-08-30",
    "month_end_weekend_2020-05-29": "2020-05-29",
    "month_end_before_holiday_2016-12-30": "2016-12-30",
    "month_end_holiday_2021-05-28": "2021-05-28",
    "month_end_early_close_2025-11-28": "2025-11-28",
    "day_before_month_end_2021-05-27": "2021-05-27",
    "first_session_2017-01-03": "2017-01-03",
    "first_session_2024-07-01": "2024-07-01",
    "month_end_plain_2008-10-31": "2008-10-31",
    "last_completed_month_end_2026-08-31": "2026-08-31",
    "current_partial_month_2026-09-15": "2026-09-15",
    "current_partial_month_last_bar_2026-09-25": "2026-09-25",
}
if MODE in ("all", "trunc"):
    full_sig, full_states = signal_states(df)
    feat_cols = [
        col for col in full_sig.columns
        if col[0].startswith(core5.SIGNAL_NAMESPACE_PREFIX_STR) and col[1].endswith("_ser")
    ]
    rows = []
    for name, d in CUTOFFS.items():
        T = pd.Timestamp(d)
        assert T in df.index, name
        pre = df.loc[:T].copy()
        pre_sig, pre_states = signal_states(pre)
        a = full_sig.loc[T, feat_cols].astype(float).to_numpy()
        b = pre_sig.loc[T, feat_cols].astype(float).to_numpy()
        feat_equal = bool(np.array_equal(a, b, equal_nan=True))
        fa = full_sig.loc[:T, feat_cols].astype(float).to_numpy()
        fb = pre_sig.loc[:T, feat_cols].astype(float).to_numpy()
        prefix_equal = bool(np.array_equal(fa, fb, equal_nan=True))
        st_equal = bool(full_states.loc[:T].astype(float).equals(pre_states.astype(float)))
        me_key = (core5.PORTFOLIO_NAMESPACE_STR, core5.MONTH_END_REBALANCE_FIELD_STR)
        me_full = bool(full_states.loc[T, me_key])
        me_pre = bool(pre_states.loc[T, me_key])
        s = c.run_backtest(pre, 100_000.0)
        wcmp = compare_weights(base_daily.loc[:T], s.daily_target_weights)
        tv_full = base["results"]["total_value"].loc[:T].astype(float)
        tv_pre = s.results["total_value"].astype(float)
        eq_equal = bool(np.array_equal(tv_full.to_numpy(), tv_pre.reindex(tv_full.index).to_numpy()))
        txf = base_tx[pd.to_datetime(base_tx["bar"]) <= T].reset_index(drop=True)
        txp = s.get_transactions().reset_index(drop=True)
        cols = ["bar", "asset", "amount", "price", "commission"]
        tx_equal = bool(len(txf) == len(txp) and txf[cols].astype(str).equals(txp[cols].astype(str)))
        strat = core5.AdaptiveMacroCore5Strategy()
        strat.previous_bar = T
        row_pre = pre_sig.loc[T]
        tw_pre = strat._target_weight_ser(row_pre, strat._long_state_ser(row_pre))
        tw_full = strat._target_weight_ser(full_sig.loc[T], strat._long_state_ser(full_sig.loc[T]))
        dec_equal = bool(np.array_equal(tw_pre.to_numpy(), tw_full.reindex(tw_pre.index).to_numpy()))
        row = {
            "cutoff": name, "date": d, "feature_row_T_equal": feat_equal, "feature_prefix_equal": prefix_equal,
            "states_and_flags_prefix_equal": st_equal, "month_end_flag_full": me_full, "month_end_flag_prefix": me_pre,
            "engine_weights": wcmp, "engine_equity_bit_identical_to_T": eq_equal,
            "engine_fills_identical_to_T": tx_equal, "close_T_target_equal": dec_equal,
            "close_T_target": {k: round(float(v), 6) for k, v in tw_pre.items()},
        }
        row["pass"] = bool(
            feat_equal and prefix_equal and st_equal and me_full == me_pre and eq_equal and tx_equal
            and dec_equal and wcmp["max_abs_diff"] == 0.0
        )
        print(json.dumps(row, default=str), flush=True)
        rows.append(row)
    c.dump(rows, "a3_truncation_invariance.json")

# ---------------- A4: positive controls -----------------
if MODE in ("all", "pc"):
    orig_fn = core5.compute_adaptive_asset_signal_df
    rows = []

    def leak_fn(ser, config_obj=core5.DEFAULT_CONFIG):
        # Injected look-ahead: the signal close at T is Close_(T+1).
        return orig_fn(pd.Series(ser, copy=True).astype(float).shift(-1), config_obj)

    def scale_fn(ser, config_obj=core5.DEFAULT_CONFIG):
        out = orig_fn(ser, config_obj)
        # Injected scale dependence: an absolute 0.5 price-unit hysteresis on the long state.
        v = out["filtered_price_ser"].notna() & out["adaptive_moving_average_ser"].notna()
        out.loc[v, "long_state_ser"] = (
            out.loc[v, "filtered_price_ser"] > out.loc[v, "adaptive_moving_average_ser"] + 0.5
        ).astype(float)
        return out

    try:
        core5.compute_adaptive_asset_signal_df = leak_fn
        _, full_states = signal_states(df)
        n_fail = 0
        for name, d in list(CUTOFFS.items())[:8]:
            T = pd.Timestamp(d)
            _, pre_states = signal_states(df.loc[:T].copy())
            if not full_states.loc[:T].astype(float).equals(pre_states.astype(float)):
                n_fail += 1
        rows.append({"control": "signal reads Close_(T+1) (shift(-1))", "harness": "A3 truncation",
                     "cutoffs_flagged": n_fail, "of": 8, "caught": n_fail > 0})
        core5.compute_adaptive_asset_signal_df = scale_fn
        _, st0 = signal_states(df)
        spy40 = c.rescale_symbol(c.rescale_symbol(df, "SPY", 40.0), core5.signal_namespace_str("SPY"), 40.0)
        _, st1 = signal_states(spy40)
        nd = int((st0.astype(float).fillna(-1) != st1.astype(float).fillna(-1)).to_numpy().sum())
        rows.append({"control": "absolute 0.5-unit hysteresis in long state", "harness": "A2 split (SPY k=40)",
                     "state_cells_flagged": nd, "caught": nd > 0})
        # Same harness without the injection must be clean (control of the control).
        core5.compute_adaptive_asset_signal_df = orig_fn
        _, st0 = signal_states(df)
        _, st1 = signal_states(spy40)
        nd0 = int((st0.astype(float).fillna(-1) != st1.astype(float).fillna(-1)).to_numpy().sum())
        rows.append({"control": "no injection (negative control)", "harness": "A2 split (SPY k=40)",
                     "state_cells_flagged": nd0, "caught": nd0 > 0})
    finally:
        core5.compute_adaptive_asset_signal_df = orig_fn
    print(rows, flush=True)
    c.dump(rows, "a4_positive_controls.json")
