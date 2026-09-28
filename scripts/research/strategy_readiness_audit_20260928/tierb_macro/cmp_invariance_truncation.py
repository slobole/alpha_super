"""Compass / Compass QQQ: A2 future-split invariance, A3 row-T truncation, A4 positive controls, A7 padded bars.

A2 decision level: one signal symbol's TOTALRETURN history divided by k (k in 40, 0.1, 1.5) as if a k:1 split
   happened after the last bar; compare month-end target weights. float32 re-rounding (vendor-like) and float64.
A2 engine level: held ETFs' CAPITALSPECIAL OHLC and Dividend divided by k, Volume times k, Unadjusted Close and
   Turnover nominal; the engine is re-run on HEAD weights; compare position-switch trades and CAGR.
A3 row T: signal closes loaded from Norgate with end_date=T; T5YIE truncated two ways:
   (i)  published-by-Close_T model: observations dated < T;
   (ii) lenient: observations dated <= T+3 calendar days (models the fred_loader UTC as-of bug admitting later rows).
   The truncated run's last row (T) must equal the full-history DAILY feature row at T, and every complete month-end
   before T must keep its weights.
A4 planted leaks the same harness must catch:
   PC-split  growth = nominal SPY TR close > 150 (scale dependent)
   PC-close  SPY close at T replaced by Close_(T+1) (one-session price leak)
   PC-fred   T5YIE level read with allow_exact_matches=True (the pre-b7a0018 rule)

Outputs: OUT/cmp_invariance_truncation.json
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import tb_common as tb

cmp_mod = tb.cmp_mod
K_LIST = (40.0, 0.1, 1.5)
FEATURE_BOOL_COLS = ["growth_on_bool", "inflation_level_on_bool", "breakeven_up_bool", "asset_up_bool", "inflation_on_bool"]
FEATURE_FLOAT_COLS = ["t5yie_float", "t5yie_prior_float", "growth_sma_float", "spy_close_float"]
CUTOFF_LIST = [
    ("mid_month", "2012-03-15"),
    ("flip_2012_03_month_end", "2012-03-30"),
    ("flip_2012_11_month_end", "2012-11-30"),
    ("flip_2014_06_month_end", "2014-06-30"),
    ("flip_2018_05_month_end", "2018-05-31"),
    ("day_after_thanksgiving_month_end", "2019-11-29"),
    ("weekend_month_end_may31_sunday", "2020-05-29"),
    ("memorial_day_month_end", "2021-05-28"),
    ("weekend_month_end_apr30_saturday", "2022-04-29"),
    ("exact_tie_2023_04_28", "2023-04-28"),
    ("month_end_before_good_friday", "2024-03-28"),
    ("last_completed_month_end", "2026-08-31"),
    ("first_session_of_month", "2026-09-01"),
    ("current_partial_month_last_bar", "2026-09-25"),
]


# ---------------------------------------------------------------------------------------------------------------
def head_fn(sig, t5, cfg):
    return cmp_mod.compute_month_end_signal_and_weight_df(sig, t5, cfg)


def planted_split_fn(sig, t5, cfg):
    feat, w = cmp_mod.compute_month_end_signal_and_weight_df(sig, t5, cfg)
    rows = {}
    for d, r in feat.iterrows():
        _lab, ser = cmp_mod._regime_target_weight_ser(
            bool(r["spy_close_float"] > 150.0), bool(r["inflation_on_bool"]), cfg.tradeable_asset_tuple, cfg.goldilocks_asset_str
        )
        rows[d] = ser
    w2 = pd.DataFrame(rows).T.loc[:, list(cfg.tradeable_asset_tuple)]
    return feat, w2


def planted_close_fn(sig, t5, cfg):
    sig = sig.copy()
    sig["SPY"] = sig["SPY"].shift(-1).fillna(sig["SPY"])
    return cmp_mod.compute_month_end_signal_and_weight_df(sig, t5, cfg)


def planted_fred_fn(sig, t5, cfg):
    orig = cmp_mod.align_fred_to_session_ser

    def leaky(fred_value_ser, session_date_index, tolerance_day_int=7, include_same_date_bool=False):
        return orig(fred_value_ser, session_date_index, tolerance_day_int, True)

    cmp_mod.align_fred_to_session_ser = leaky
    try:
        return cmp_mod.compute_month_end_signal_and_weight_df(sig, t5, cfg)
    finally:
        cmp_mod.align_fred_to_session_ser = orig


FN_DICT = {"HEAD": head_fn, "PC_split": planted_split_fn, "PC_close": planted_close_fn, "PC_fred": planted_fred_fn}


def all_rows(fn, sig, t5, cfg):
    orig = cmp_mod.get_month_end_session_index
    cmp_mod.get_month_end_session_index = lambda idx: pd.DatetimeIndex(idx).tz_localize(None).normalize()
    try:
        return fn(sig, t5, cfg)
    finally:
        cmp_mod.get_month_end_session_index = orig


# ---------------------------------------------------------------------------------------------------------------
def weight_diff_dates(w1: pd.DataFrame, w2: pd.DataFrame) -> list[str]:
    idx = w1.index.intersection(w2.index)
    cols = [c for c in w1.columns if c in w2.columns]
    d = (w1.loc[idx, cols] - w2.loc[idx, cols]).abs().max(axis=1) > 1e-12
    out = [str(x.date()) for x in idx[d.to_numpy()]]
    if len(w1.index.symmetric_difference(w2.index)):
        out.append(f"index_mismatch:{len(w1.index.symmetric_difference(w2.index))}")
    return out


def a2_decision(variant_str: str) -> dict:
    cfg = tb.compass_config(variant_str)
    sig32 = tb.compass_signal_close_df(tb.NORGATE_LAST_BAR_STR)
    t5 = tb.ensure_frozen_t5yie()
    res = {}
    for fn_name in ("HEAD", "PC_split"):
        fn = FN_DICT[fn_name]
        _f0, w0 = fn(sig32, t5, cfg)
        cases = []
        for sym in ("SPY", "XLE", "XLU", "XLP", "XLV", "ALL"):
            for k in K_LIST:
                for prec in ("float32", "float64"):
                    s = sig32.copy()
                    cols = list(s.columns) if sym == "ALL" else [sym]
                    if prec == "float32":
                        s[cols] = (s[cols] / np.float32(k)).astype(np.float32)
                    else:
                        s = s.astype(np.float64)
                        s[cols] = s[cols] / k
                    _f1, w1 = fn(s, t5, cfg)
                    dd = weight_diff_dates(w0, w1)
                    cases.append({"symbol": sym, "k": k, "precision": prec, "n_decisions": int(len(w0)),
                                  "n_diff": len(dd), "diff_dates": dd[:8]})
        res[fn_name] = {
            "cases": cases,
            "pass": int(sum(c["n_diff"] == 0 for c in cases)),
            "total": len(cases),
        }
    return res


def rescale_exec(df: pd.DataFrame, sym: str, k: float) -> pd.DataFrame:
    df = df.copy()
    for fld in ("Open", "High", "Low", "Close", "Dividend"):
        df[(sym, fld)] = (df[(sym, fld)].astype(np.float64) / k)
    df[(sym, "Volume")] = df[(sym, "Volume")].astype(np.float64) * k
    return df


def switch_trades(strategy_obj) -> set:
    tx = strategy_obj.get_transactions()
    tx = tx[tx["order_id"] != -1] if "order_id" in tx.columns else tx
    out = set()
    pos: dict[str, float] = {}
    for _, r in tx.sort_values("bar").iterrows():
        a = str(r["asset"])
        before = pos.get(a, 0.0)
        after = before + float(r["amount"])
        pos[a] = after
        if (abs(before) < 1e-9) != (abs(after) < 1e-9):
            out.add((str(pd.Timestamp(r["bar"]).date()), a, "open" if abs(before) < 1e-9 else "close"))
    return out


def a2_engine(variant_str: str, symbols: tuple[str, ...]) -> dict:
    cfg = tb.compass_config(variant_str)
    _f, w = tb.month_end_weights(variant_str, tb.compass_signal_close_df(tb.MAIN_END_STR))
    base_exec = tb.compass_execution_price_df(variant_str, tb.MAIN_END_STR)
    s0 = tb.run_compass_engine(variant_str, month_end_weight_df=w, execution_price_df=base_exec)
    m0 = tb.metrics(tb.nav_ser(s0), tb.MAIN_START_STR, tb.MAIN_END_STR)
    sw0 = switch_trades(s0)
    cases = []
    for sym in symbols:
        for k in K_LIST:
            s1 = tb.run_compass_engine(variant_str, month_end_weight_df=w, execution_price_df=rescale_exec(base_exec, sym, k))
            m1 = tb.metrics(tb.nav_ser(s1), tb.MAIN_START_STR, tb.MAIN_END_STR)
            sw1 = switch_trades(s1)
            cases.append({
                "symbol": sym, "k": k,
                "switch_trades_equal": sw0 == sw1,
                "switch_trade_sym_diff": sorted(sw0 ^ sw1)[:6],
                "cagr_diff_pp": m1["cagr_pct"] - m0["cagr_pct"],
                "sharpe_diff": m1["sharpe"] - m0["sharpe"],
                "fills_base": int(len(s0.get_transactions())), "fills_rescaled": int(len(s1.get_transactions())),
            })
            print("engine", variant_str, sym, k, cases[-1]["switch_trades_equal"], round(cases[-1]["cagr_diff_pp"], 5), flush=True)
    return {"base": m0, "n_switch_trades": len(sw0), "cases": cases}


def a3_truncation(variant_str: str) -> dict:
    cfg = tb.compass_config(variant_str)
    t5_full = tb.ensure_frozen_t5yie()
    sig_full = tb.compass_signal_close_df(tb.NORGATE_LAST_BAR_STR)
    full_rows = {}
    full_me = {}
    for fn_name, fn in FN_DICT.items():
        if fn_name == "PC_split":
            continue
        full_rows[fn_name] = all_rows(fn, sig_full, t5_full, cfg)
        full_me[fn_name] = fn(sig_full, t5_full, cfg)
    cases = []
    for label, t_str in CUTOFF_LIST:
        T = pd.Timestamp(t_str)
        sig_T = cmp_mod.load_signal_close_df(cfg.signal_asset_tuple, "2002-01-01", t_str)
        load_equals_slice = bool(sig_T.astype(float).equals(sig_full.loc[:T].astype(float)))
        for fred_label, t5_T in (
            ("published_by_close_T", t5_full[t5_full.index < T]),
            ("lenient_le_T_plus_3", t5_full[t5_full.index <= T + pd.Timedelta(days=3)]),
        ):
            for fn_name, fn in FN_DICT.items():
                if fn_name == "PC_split":
                    continue
                try:
                    feat_T, w_T = fn(sig_T, t5_T, cfg)
                except Exception as exc:  # noqa: BLE001 - a loud failure counts as caught
                    cases.append({"label": label, "T": t_str, "fred": fred_label, "fn": fn_name,
                                  "row_T_equal": False, "raised": repr(exc)[:160]})
                    continue
                last_ts = feat_T.index[-1]
                feat_full, w_full = full_rows[fn_name]
                row_equal = last_ts == T
                if row_equal:
                    a = feat_T.loc[T]
                    b = feat_full.loc[T]
                    for c in FEATURE_BOOL_COLS:
                        row_equal &= bool(a[c]) == bool(b[c])
                    for c in FEATURE_FLOAT_COLS:
                        row_equal &= bool(np.isclose(float(a[c]), float(b[c]), rtol=0, atol=1e-9))
                    row_equal &= bool(np.allclose(w_T.loc[T].to_numpy(float), w_full.loc[T].to_numpy(float), atol=1e-12))
                decision_equal = bool(
                    last_ts == T and np.allclose(w_T.loc[T].to_numpy(float), w_full.loc[T].to_numpy(float), atol=1e-12)
                )
                _fm, wm_full = full_me[fn_name]
                prior = w_T.index[w_T.index < T.to_period("M").start_time]
                prior_diff = weight_diff_dates(w_T.loc[prior], wm_full.loc[wm_full.index.intersection(prior)])
                cases.append({
                    "label": label, "T": t_str, "fred": fred_label, "fn": fn_name,
                    "last_row": str(last_ts.date()), "row_T_equal": bool(row_equal),
                    "row_T_decision_equal": decision_equal,
                    "complete_months_before_T_diff": prior_diff[:5],
                    "norgate_truncated_load_equals_slice": load_equals_slice,
                })
        print("trunc", variant_str, label, flush=True)
    summary = {}
    for fn_name in ("HEAD", "PC_close", "PC_fred"):
        for fred_label in ("published_by_close_T", "lenient_le_T_plus_3"):
            sub = [c for c in cases if c["fn"] == fn_name and c["fred"] == fred_label]
            summary[f"{fn_name}|{fred_label}"] = {
                "cutoffs": len(sub),
                "row_T_equal": int(sum(c["row_T_equal"] for c in sub)),
                "row_T_decision_equal": int(sum(c.get("row_T_decision_equal", False) for c in sub)),
                "prior_months_clean": int(sum(len(c.get("complete_months_before_T_diff", ["x"])) == 0 for c in sub)),
            }
    return {"summary": summary, "cases": cases}


def a7_padding() -> dict:
    import exchange_calendars as xc

    sig = tb.compass_signal_close_df(tb.NORGATE_LAST_BAR_STR)
    cal = xc.get_calendar("XNYS", start="2002-01-01", end=tb.NORGATE_LAST_BAR_STR)
    sess = cal.sessions_in_range("2002-01-02", tb.NORGATE_LAST_BAR_STR).tz_localize(None)
    extra = sig.index.difference(sess)
    missing = sess.difference(sig.index)
    feat, _w = tb.month_end_weights("xlk", sig)
    zero_vol = {}
    from data.norgate_loader import load_price_timeseries

    for sym in tb.cmp_mod.SIGNAL_ASSET_TUPLE + ("XLK", "QQQ", "IEF"):
        df = load_price_timeseries(sym, start_date_str="2002-01-01", end_date_str=tb.NORGATE_LAST_BAR_STR)
        vol = df["Volume"].reindex(feat.index)
        zero_vol[sym] = [str(d.date()) for d in vol.index[(vol.fillna(0) <= 0).to_numpy()]]
    xnys_month_end = pd.Series(sess, index=sess).groupby(sess.to_period("M")).max()
    me_mismatch = sorted(set(feat.index.date) ^ set(pd.DatetimeIndex(xnys_month_end.values).date))
    return {
        "signal_index_extra_vs_xnys": [str(d.date()) for d in extra][:20],
        "signal_index_missing_vs_xnys": [str(d.date()) for d in missing][:20],
        "month_end_decisions_vs_xnys_month_ends_symmetric_diff": [str(d) for d in me_mismatch][:20],
        "zero_volume_at_month_end_decisions": zero_vol,
    }


def main() -> None:
    out = {}
    out["A7_padding"] = a7_padding()
    print("A7", out["A7_padding"]["month_end_decisions_vs_xnys_month_ends_symmetric_diff"], flush=True)
    for variant_str in ("xlk", "qqq"):
        out[f"A2_decision_{variant_str}"] = a2_decision(variant_str)
        print("A2", variant_str, {k: (v["pass"], v["total"]) for k, v in out[f"A2_decision_{variant_str}"].items()}, flush=True)
    out["A3_truncation_xlk"] = a3_truncation("xlk")
    out["A3_truncation_qqq"] = a3_truncation("qqq")
    tb.write_json("cmp_invariance_truncation_partial.json", out)
    out["A2_engine_xlk"] = a2_engine("xlk", ("XLK", "XLE", "IEF"))
    out["A2_engine_qqq"] = a2_engine("qqq", ("QQQ",))
    tb.write_json("cmp_invariance_truncation.json", out)


if __name__ == "__main__":
    main()
