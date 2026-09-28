"""Backtest-correctness checks for the TAA 2x / linearity PM_READY variants (protocol A2-A10). Research-only.

Modes (``uv run python taa2x_bc.py <mode> [variant ...]``):
  slices   Loader-faithfulness of the cache: for every symbol the variants load (TR signal, CAPITALSPECIAL execution,
           SPY/$VIX helpers) a fresh Norgate load ending at T equals the cached full load sliced at T (8 cut-offs).
  rowt     A3 ROW-T truncation: for EVERY month-end decision session T (last XNYS session of the month) the weight row
           computed from data ending at T equals the full-history row for that month; plus mid-month / first-session /
           partial-month cut-offs, where the partial-month row must never be mapped to a rebalance. A4 positive
           controls on 36 month-ends: signal close read at T+1, and the VIX gate reading SPY/$VIX at T+1.
  inv      A2 future-split invariance of month-end weights: every loaded symbol (defensive ETFs, fallback incl. QLD/SSO,
           SPY) rescaled at the loader boundary by k in {40, 0.1, 1.5}. A4 control: a level-sensitive score.
  dtb3     A5 DTB3 publication lag in XNYS SESSIONS (1 and 2 sessions) + smallest score-to-hurdle margin.
  helper   A5 helper staleness: one-session-stale $VIX, and stale SPY+$VIX, vs the month-end gate where the fallback
           sleeve is non-zero (what a live route would inherit, A-LIVE-13).
  runs     A7-A10: metrics (full / 2012+ / last 3y), determinism, capital scaling, negative cash, dividends, fills on
           zero-volume bars, adjusted-vs-raw share units (historical_share_units_bool) at 30K / 100K / 1M.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

import taa2x_common as tc

ALL = list(tc.VARIANT_DICT)


def _dump(name: str, obj) -> None:
    (tc.OUT / name).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _symbols(key: str) -> tuple[list[str], list[str]]:
    cfg = tc.config(key)
    return list(cfg.defensive_asset_list), list(cfg.tradeable_asset_list)


# ------------------------------------------------------------------------------------------------ slices
def mode_slices(keys) -> None:
    rows = []
    cut_list = ["2008-09-30", "2012-12-31", "2016-02-29", "2020-03-31", "2023-09-29", "2025-12-31", "2026-08-31", "2026-09-15"]
    pairs = set()
    for key in keys:
        cfg = tc.config(key)
        for s in cfg.defensive_asset_list:
            pairs.add((s, "TOTALRETURN", cfg.start_date_str))
        for s in cfg.tradeable_asset_list:
            pairs.add((s, "CAPITALSPECIAL", cfg.start_date_str))
        for s in ("SPY", "$VIX"):
            pairs.add((s, "CAPITALSPECIAL", cfg.start_date_str))
    for sym, adj, start in sorted(pairs):
        full = tc.cached_loader(sym, adjustment_str=adj, start_date_str=start)
        for cut in cut_list:
            fresh = tc._REAL_LOADER(sym, adjustment_str=adj, start_date_str=start, end_date_str=cut)
            sliced = full.loc[: pd.Timestamp(cut)]
            same = fresh.shape == sliced.shape and bool(np.allclose(fresh.to_numpy(dtype=float), sliced.to_numpy(dtype=float), equal_nan=True, rtol=0, atol=0)) and fresh.index.equals(sliced.index)
            rows.append({"symbol": sym, "adjustment": adj, "start": start, "cutoff": cut, "equal": same})
            if not same:
                print("SLICE DIFF", sym, adj, cut, fresh.shape, sliced.shape, flush=True)
    out = {"cases": len(rows), "equal": sum(r["equal"] for r in rows), "rows": rows}
    print("slices", out["equal"], "/", out["cases"], flush=True)
    _dump("loader_slices.json", out)


# ------------------------------------------------------------------------------------------------ row-T
def _leak_signal_patch(on: bool):
    real_std = tc.base_module.compute_month_end_weight_df
    real_lin = tc.linearity_1n_module.compute_daily_linearity_score_df

    def leaky_std(signal_close_df, cash_return_ser, config):
        return real_std(signal_close_df.shift(-1), cash_return_ser, config)  # *** CRITICAL*** planted leak

    def leaky_lin(signal_close_df, lookback_day_vec):
        return real_lin(signal_close_df=signal_close_df.shift(-1), lookback_day_vec=lookback_day_vec)  # planted leak

    if on:
        tc.base_module.compute_month_end_weight_df = leaky_std
        tc.linearity_1n_module.compute_daily_linearity_score_df = leaky_lin
    return real_std, real_lin


def _leak_gate_patch():
    real = tc.utils_module.compute_daily_vrp_signal_df

    def leaky(spy_close_ser, vix_close_ser, realized_vol_lookback_day_int=20):
        return real(spy_close_ser.shift(-1).dropna(), vix_close_ser.shift(-1).dropna(), realized_vol_lookback_day_int)  # leak

    tc.utils_module.compute_daily_vrp_signal_df = leaky
    return real


def _row_diff(a: pd.Series | None, b: pd.Series | None) -> float:
    if a is None or b is None:
        return float("inf")
    cols = a.index.union(b.index)
    return float((a.reindex(cols).fillna(0.0) - b.reindex(cols).fillna(0.0)).abs().max())


def _rowt_one(key: str, sessions: pd.DatetimeIndex, decision_list: list[pd.Timestamp], full_w: pd.DataFrame) -> list[dict]:
    rows = []
    for t in decision_list:
        w_t, rb_t, _, _ = tc.weight_frames(key, tc.config(key, end_date_str=t.date().isoformat()))
        label = (pd.Timestamp(t) + pd.offsets.MonthEnd(0)).normalize()
        a = full_w.loc[label] if label in full_w.index else None
        b = w_t.loc[label] if label in w_t.index else None
        rows.append({"decision_session": t.date().isoformat(), "label": label.date().isoformat(),
                     "max_abs_weight_diff": _row_diff(a, b), "rebalance_rows_after_T": int((rb_t.index > t).sum())})
    return rows


def _tolerate_empty_rebalance() -> None:
    """A truncated run ending at the FIRST decision has no next-month session, so the production mapper raises
    'No rebalance dates'. Study-only wrapper: return an empty table instead (decision rows are unaffected)."""
    real = tc.base_module.map_month_end_weights_to_rebalance_open_df

    def tolerant(month_end_weight_df, execution_index):
        try:
            return real(month_end_weight_df, execution_index)
        except RuntimeError:
            return pd.DataFrame(columns=month_end_weight_df.columns, index=pd.DatetimeIndex([], name="rebalance_date"))

    tc.base_module.map_month_end_weights_to_rebalance_open_df = tolerant
    tc.utils_module.map_month_end_weights_to_rebalance_open_df = tolerant
    tc.linearity_1n_module.map_month_end_weights_to_rebalance_open_df = tolerant


def mode_rowt(keys) -> None:
    tc.use_cached_loader(True)
    _tolerate_empty_rebalance()
    sessions = tc.xnys_sessions()
    out = {}
    for key in keys:
        t0 = time.time()
        full_w, full_rb, _, _ = tc.weight_frames(key, tc.config(key))
        decision_list = [tc.last_session_of_month(sessions, lab) for lab in full_w.index if lab <= pd.Timestamp("2026-08-31")]
        clean = _rowt_one(key, sessions, decision_list, full_w)
        n_ok = sum(r["max_abs_weight_diff"] <= 1e-12 for r in clean)
        # decision session -> rebalance row check: the backtest trades the row decided at T on the next session
        rb_check = []
        for t in decision_list:
            nxt = sessions[sessions > t][0]
            label = (t + pd.offsets.MonthEnd(0)).normalize()
            if nxt in full_rb.index:
                rb_check.append(_row_diff(full_rb.loc[nxt], full_w.loc[label]))
        # non-decision cut-offs: partial month rows must not become rebalance rows
        extra = {}
        for cut in ("2016-07-15", "2018-06-01", "2020-03-18", "2026-09-01", "2026-09-15", "2026-09-25"):
            w_t, rb_t, _, _ = tc.weight_frames(key, tc.config(key, end_date_str=cut))
            ct = pd.Timestamp(cut)
            completed = w_t.index[w_t.index.to_period("M") < ct.to_period("M")].intersection(full_w.index)
            extra[cut] = {"partial_month_row_present": bool((w_t.index.to_period("M") == ct.to_period("M")).any()),
                          "rebalance_rows_after_cutoff": int((rb_t.index > ct).sum()),
                          "last_rebalance": rb_t.index[-1].date().isoformat(),
                          "completed_rows_max_diff": float((w_t.loc[completed] - full_w.loc[completed]).abs().max().max())}
        # positive controls on 36 month-ends (every 6th decision, most recent included)
        pc_dates = decision_list[::6][-36:]
        real_std, real_lin = _leak_signal_patch(True)
        try:
            leak_full = tc.weight_frames(key, tc.config(key))[0]
            leak_rows = _rowt_one(key, sessions, pc_dates, leak_full)
        finally:
            tc.base_module.compute_month_end_weight_df = real_std
            tc.linearity_1n_module.compute_daily_linearity_score_df = real_lin
        real_gate = _leak_gate_patch()
        try:
            gleak_full = tc.weight_frames(key, tc.config(key))[0]
            gate_rows = _rowt_one(key, sessions, pc_dates, gleak_full)
        finally:
            tc.utils_module.compute_daily_vrp_signal_df = real_gate
        out[key] = {
            "first_decision": decision_list[0].date().isoformat(), "first_rebalance": full_rb.index[0].date().isoformat(),
            "decisions": len(clean), "row_t_exact": n_ok,
            "row_t_failures": [r for r in clean if r["max_abs_weight_diff"] > 1e-12][:10],
            "decisions_with_rebalance_rows_after_T_in_truncated_run": sum(r["rebalance_rows_after_T"] > 0 for r in clean),
            "rebalance_row_equals_decision_row": int(sum(d <= 1e-12 for d in rb_check)), "rebalance_rows_checked": len(rb_check),
            "non_decision_cutoffs": extra,
            "pc_signal_leak_caught": sum(r["max_abs_weight_diff"] > 1e-12 for r in leak_rows), "pc_cases": len(leak_rows),
            "pc_vix_gate_leak_caught": sum(r["max_abs_weight_diff"] > 1e-12 for r in gate_rows),
            "elapsed_s": round(time.time() - t0, 1),
        }
        print(key, json.dumps({k: v for k, v in out[key].items() if k not in ("non_decision_cutoffs", "row_t_failures")}), flush=True)
        pd.DataFrame(clean).to_csv(tc.OUT / f"rowt_{key}.csv", index=False)
    _dump(f"rowt_{'_'.join(keys)}.json", out)


# ------------------------------------------------------------------------------------------------ invariance
def mode_inv(keys) -> None:
    tc.use_cached_loader(True)
    out = {}
    for key in keys:
        cfg = tc.config(key)
        ref = tc.weight_frames(key, cfg)[0]
        symbols = list(dict.fromkeys(list(cfg.tradeable_asset_list) + ["SPY"]))
        rows = []
        for sym in symbols:
            for k in (40.0, 0.1, 1.5):
                def scaled(symbol_str, *args, _s=sym, _k=k, **kwargs):
                    df = tc.cached_loader(symbol_str, *args, **kwargs)
                    if symbol_str == _s:
                        for f in ("Open", "High", "Low", "Close", "Dividend"):
                            if f in df.columns:
                                df[f] = df[f] / _k
                        if "Volume" in df.columns:
                            df["Volume"] = df["Volume"] * _k
                    return df
                tc.base_module.load_price_timeseries = scaled
                tc.utils_module.load_price_timeseries = scaled
                try:
                    new = tc.weight_frames(key, cfg)[0]
                finally:
                    tc.use_cached_loader(True)
                common = ref.index.intersection(new.index)
                d = float((new.loc[common] - ref.loc[common]).abs().max().max())
                rows.append({"symbol": sym, "k": k, "max_weight_diff": d, "rows": len(common),
                             "pass": d <= 1e-12 and len(common) == len(ref)})
        # A4: level-sensitive score must be caught
        real_std = tc.base_module.compute_month_end_weight_df
        real_lin = tc.linearity_1n_module.compute_daily_linearity_score_df

        def level_std(signal_close_df, cash_return_ser, config):
            return real_std(signal_close_df.diff().cumsum().fillna(0.0) / 100.0 + 1.0, cash_return_ser, config)

        def level_lin(signal_close_df, lookback_day_vec):
            # additive (non-homogeneous) defect: a sign-only zero hurdle is blind to a pure rescale of the score
            return real_lin(signal_close_df=signal_close_df + 5.0, lookback_day_vec=lookback_day_vec)

        pc = []
        for sym in ("GLD", "TLT", "UUP", "DBC"):
            tc.base_module.compute_month_end_weight_df = level_std
            tc.linearity_1n_module.compute_daily_linearity_score_df = level_lin
            try:
                pref = tc.weight_frames(key, cfg)[0]

                def scaled(symbol_str, *args, _s=sym, **kwargs):
                    df = tc.cached_loader(symbol_str, *args, **kwargs)
                    if symbol_str == _s:
                        for f in ("Open", "High", "Low", "Close", "Dividend"):
                            if f in df.columns:
                                df[f] = df[f] / 40.0
                    return df
                tc.base_module.load_price_timeseries = scaled
                tc.utils_module.load_price_timeseries = scaled
                pnew = tc.weight_frames(key, cfg)[0]
            finally:
                tc.base_module.compute_month_end_weight_df = real_std
                tc.linearity_1n_module.compute_daily_linearity_score_df = real_lin
                tc.use_cached_loader(True)
            common = pref.index.intersection(pnew.index)
            d = float((pnew.loc[common] - pref.loc[common]).abs().max().max())
            pc.append({"symbol": sym, "k": 40.0, "planted_max_weight_diff": d, "caught": d > 1e-12})
        out[key] = {"cases": len(rows), "passed": sum(r["pass"] for r in rows), "symbols": symbols, "detail": rows,
                    "positive_control": pc}
        print(key, out[key]["passed"], "/", out[key]["cases"], "PC", [p["caught"] for p in pc], flush=True)
    _dump(f"invariance_{'_'.join(keys)}.json", out)


# ------------------------------------------------------------------------------------------------ DTB3
def mode_dtb3(keys) -> None:
    tc.use_cached_loader(True)
    sessions = tc.xnys_sessions()

    def redate(index, n):
        pos = np.clip(sessions.searchsorted(index, side="right") + (n - 1), 0, len(sessions) - 1)
        return pd.DatetimeIndex(sessions[pos])

    real_fn = tc.base_module.load_cash_return_ser_and_snapshot
    out = {}
    for key in keys:
        if tc.VARIANT_DICT[key]["kind"] != "standard":
            out[key] = {"applicable": False, "reason": "linearity rule uses a zero hurdle, no DTB3"}
            continue
        cfg = tc.config(key)
        ref = tc.weight_frames(key, cfg)[0]
        res = {"decisions": len(ref)}
        for n in (1, 2):
            def lagged(config, _n=n):
                ser, snap = real_fn(config)
                ser = ser.copy()
                ser.index = redate(ser.index, _n)
                return ser.groupby(level=0).last(), snap
            tc.base_module.load_cash_return_ser_and_snapshot = lagged
            try:
                lag = tc.weight_frames(key, cfg)[0]
            finally:
                tc.base_module.load_cash_return_ser_and_snapshot = real_fn
            common = ref.index.intersection(lag.index)
            diff = (ref.loc[common] - lag.loc[common]).abs().max(axis=1)
            res[f"session_lag_{n}"] = {"flips": int((diff > 1e-12).sum()),
                                       "flip_months": [d.date().isoformat() for d in common[diff > 1e-12]]}
        # score-to-hurdle margin with the backtest hurdle
        signal = tc.base_module.load_signal_close_df(cfg.defensive_asset_list, cfg.start_date_str, cfg.end_date_str)
        cash, _ = real_fn(cfg)
        mom = sum(signal.resample("ME").last().pct_change(m, fill_method=None) for m in cfg.momentum_lookback_month_vec) / 4.0
        hurdle = cash.resample("ME").last()
        margin = mom.sub(hurdle, axis=0).loc[ref.index].abs()
        res["min_abs_score_minus_hurdle"] = float(margin.min().min())
        res["min_margin_month"] = str(margin.min(axis=1).idxmin().date())
        res["largest_one_session_dtb3_monthly_hurdle_change"] = float(cash.diff().abs().max())
        out[key] = res
        print(key, json.dumps(res)[:600], flush=True)
    _dump(f"dtb3_{'_'.join(keys)}.json", out)


# ------------------------------------------------------------------------------------------------ helper staleness
def mode_helper(keys) -> None:
    tc.use_cached_loader(True)
    out = {}
    for key in keys:
        cfg = tc.config(key)
        w, _, daily, diag = tc.weight_frames(key, cfg)
        d = daily.copy()
        gate = d["rv20_ann_pct"] < d["vix_close"]
        stale_vix = d["rv20_ann_pct"] < d["vix_close"].shift(1)
        stale_both = d["rv20_ann_pct"].shift(1) < d["vix_close"].shift(1)
        me = pd.DataFrame({"gate": gate, "stale_vix": stale_vix, "stale_both": stale_both}).resample("ME").last()
        me = me.reindex(diag.index)
        base_fb = diag["base_fallback_weight"]
        live = base_fb > 1e-12
        res = {"decisions": int(len(diag)), "months_fallback_nonzero": int(live.sum())}
        for col in ("stale_vix", "stale_both"):
            flip = live & (me["gate"] != me[col])
            res[col] = {"flips": int(flip.sum()), "months": [x.date().isoformat() for x in diag.index[flip]],
                        "max_weight_moved": float(base_fb[flip].max()) if flip.any() else 0.0}
        out[key] = res
        print(key, json.dumps(res), flush=True)
    _dump(f"helper_stale_{'_'.join(keys)}.json", out)


# ------------------------------------------------------------------------------------------------ runs
def _zero_volume_fills(strategy) -> dict:
    tx = strategy.get_transactions().copy()
    tx["bar"] = pd.to_datetime(tx["bar"])
    out = {}
    for sym, g in tx.groupby("asset"):
        px = tc._REAL_LOADER(str(sym), start_date_str="2000-01-01", end_date_str=tc.END_DATE_STR)
        vol = px["Volume"].reindex(g["bar"]).fillna(0.0).to_numpy()
        z = vol <= 0
        out[str(sym)] = {"fills": int(len(g)), "zero_volume": int(z.sum()),
                         "dates": [d.date().isoformat() for d in g["bar"][z]][:10]}
    return out


def mode_runs(keys) -> None:
    out = {}
    for key in keys:
        res = {}
        a = tc.run(key)
        b = tc.run(key)
        nav = a.results["total_value"].astype(float)
        res["metrics_full"] = tc.metrics(nav)
        res["metrics_2012_10"] = tc.metrics(nav, start="2012-10-01")
        res["metrics_2019"] = tc.metrics(nav, start="2019-01-01")
        res["metrics_last3y"] = tc.metrics(nav, start="2023-09-25")
        res["determinism_bit_identical"] = bool(np.array_equal(nav.to_numpy(), b.results["total_value"].astype(float).to_numpy()))
        cash = a.results["cash"].astype(float) / nav
        res["min_cash_frac"] = float(cash.min())
        res["share_days_negative_cash"] = float((cash < 0).mean())
        res["mean_cash_frac"] = float(cash.mean())
        res["dividend_net_total"] = float(getattr(a, "dividend_cash_net_total_float", np.nan))
        tx = a.get_transactions()
        res["fills"] = int(len(tx))
        res["commission_total_100k"] = float(tx["commission"].sum())
        res["zero_volume_fills"] = _zero_volume_fills(a)
        tx.to_csv(tc.OUT / f"transactions_{key}_100k.csv", index=False)
        nav.to_csv(tc.OUT / f"nav_{key}_100k.csv")
        r = nav.pct_change().dropna()
        for cap in (30_000.0, 1_000_000.0):
            for hsu in (False, True):
                s = tc.run(key, capital=cap, hsu=hsu)
                v = s.results["total_value"].astype(float)
                rr = v.pct_change().dropna()
                res[f"cap_{int(cap)}_{'hsu' if hsu else 'engine'}"] = {
                    **tc.metrics(v), "last3y": tc.metrics(v, start="2023-09-25"),
                    "commission_total": float(s.get_transactions()["commission"].sum()),
                    "fills": int(len(s.get_transactions())),
                    "daily_ret_corr_vs_100k": float(rr.corr(r)), "daily_ret_max_abs_diff_vs_100k": float((rr - r).abs().max())}
        s = tc.run(key, hsu=True)
        res["cap_100000_hsu"] = {**tc.metrics(s.results["total_value"]), "commission_total": float(s.get_transactions()["commission"].sum())}
        out[key] = res
        print(key, json.dumps({k: v for k, v in res.items() if k != "zero_volume_fills"}, default=str)[:2500], flush=True)
    _dump(f"runs_{'_'.join(keys)}.json", out)


if __name__ == "__main__":
    t0 = time.time()
    mode = sys.argv[1]
    keys = sys.argv[2:] or ALL
    {"slices": mode_slices, "rowt": mode_rowt, "inv": mode_inv, "dtb3": mode_dtb3, "helper": mode_helper,
     "runs": mode_runs, "rowt_scores": lambda k: None}[mode](keys)
    print("elapsed_s", round(time.time() - t0, 1), flush=True)


# ------------------------------------------------------------------------------------------------ row-T on continuous scores
def _score_frames(key: str, config_obj) -> pd.DataFrame:
    """Continuous month-end decision inputs: momentum (or linearity) scores, hurdle, rv20 and $VIX at the month-end row."""
    if tc.VARIANT_DICT[key]["kind"] == "standard":
        t = tc.utils_module.get_standard_fallback_vix_cash_data(config=config_obj, base_data_loader_fn=tc.base_module.get_defense_first_data)
        score = t[1].copy()
        cash, _ = tc.base_module.load_cash_return_ser_and_snapshot(config_obj)
        score["hurdle"] = cash.resample("ME").last()
        daily_vrp = t[2]
    else:
        t = tc.utils_module.get_linearity_1n_fallback_vix_cash_data(
            config=config_obj, base_data_loader_fn=tc.linearity_1n_module.get_defense_first_linearity_1n_data)
        score = t[2].copy()
        daily_vrp = t[3]
    me = daily_vrp[["rv20_ann_pct", "vix_close"]].resample("ME").last()
    return score.join(me, how="left")


def mode_rowt_scores(keys) -> None:
    """Row-T on the CONTINUOUS inputs of each decision (every month-end score, hurdle, rv20, $VIX), so a planted
    one-session leak is visible on every tested row, not only when it flips a weight."""
    tc.use_cached_loader(True)
    _tolerate_empty_rebalance()
    sessions = tc.xnys_sessions()
    out = {}
    for key in keys:
        full = _score_frames(key, tc.config(key))
        labels = [lab for lab in full.dropna().index if lab <= pd.Timestamp("2026-08-31")]
        dates = [tc.last_session_of_month(sessions, lab) for lab in labels][::6][-36:]

        def one(full_df, patch=None):
            n_ok, rows = 0, []
            for t in dates:
                lab = (t + pd.offsets.MonthEnd(0)).normalize()
                tr = _score_frames(key, tc.config(key, end_date_str=t.date().isoformat()))
                a, b = full_df.loc[lab], tr.loc[lab] if lab in tr.index else None
                d = float((a - b).abs().max()) if b is not None else float("inf")
                rows.append(d)
            return rows

        clean = one(full)
        real_std, real_lin = _leak_signal_patch(True)
        try:
            sig_leak = one(_score_frames(key, tc.config(key)))
        finally:
            tc.base_module.compute_month_end_weight_df = real_std
            tc.linearity_1n_module.compute_daily_linearity_score_df = real_lin
        real_gate = _leak_gate_patch()
        try:
            gate_leak = one(_score_frames(key, tc.config(key)))
        finally:
            tc.utils_module.compute_daily_vrp_signal_df = real_gate
        out[key] = {"cases": len(dates), "clean_exact": sum(d <= 1e-12 for d in clean), "clean_max_diff": max(clean),
                    "signal_leak_caught": sum(d > 1e-12 for d in sig_leak), "gate_leak_caught": sum(d > 1e-12 for d in gate_leak)}
        print(key, json.dumps(out[key]), flush=True)
    _dump(f"rowt_scores_{'_'.join(keys)}.json", out)


if __name__ == "__main__" and sys.argv[1] == "rowt_scores":
    mode_rowt_scores(sys.argv[2:] or ALL)
