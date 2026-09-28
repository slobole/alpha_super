"""Tactical FI (L14 IEF/LQD): BC checks A2/A3/A4/A5/A8/A9/A10 on real data.

Parts
- baseline: frozen mode and ALFRED point-in-time (vintage T, previous session) through the module's run_variant;
  reproduce the handoff numbers; determinism; capital scaling; raw historical share units.
- accounting: 25% dividend withholding (house default) vs the module's 0%; cash-rate sensitivity (0%, DGS3MO-0.5pp,
  IBKR small-account rule at USD 30K); negative cash.
- A2 engine split invariance on IEF, LQD, $SPXTR with a planted positive control (sizing on nominal Unadjusted Close).
- A3 row-T truncation of the decision function with yields truncated by the module's own release model and leniently
  (T+3 calendar days), session index truncated at T plus the next XNYS session; planted one-session leak.
- A5 one-extra-session lag replay on all 289 frozen decisions (pre-2014 included); independent re-fetch of ALFRED
  vintages for a sample of decisions, compared with the hash-locked snapshot.

Outputs: OUT/tfi_checks.json
"""

from __future__ import annotations

import pickle
from dataclasses import replace

import numpy as np
import pandas as pd

import tb_common as tb
import strategies.taa_beyond_6040.strategy_taa_tactical_fixed_income_ief_lqd as tfi
from alpha.data.alfred_snapshot import fetch_alfred_vintage_dict

BOOK_START = "2012-10-02"
BOOK_END = "2026-08-19"
FULL_START = "2002-08-01"


def m_all(strategy_obj) -> dict:
    nav = tb.nav_ser(strategy_obj)
    return {
        "full": tb.metrics(nav, FULL_START, BOOK_END),
        "book": tb.metrics(nav, BOOK_START, BOOK_END),
        "since_2014_05": tb.metrics(nav, "2014-05-01", BOOK_END),
        "last3y": tb.metrics(nav, "2023-08-19", BOOK_END),
    }


def run(config_obj=None, **kw):
    return tfi.run_variant(show_display_bool=False, save_results_bool=False, config_obj=config_obj or tfi.DEFAULT_CONFIG, **kw)


def run_on_data(data_tuple, capital=100_000.0, strategy_class=tfi.TacticalYieldStrategy, raw_units=False,
                withholding=None, exec_df=None):
    execution_price_df, _y, signal_df, rebalance_weight_df, cash_return_ser, snaps = data_tuple
    if exec_df is not None:
        execution_price_df = exec_df
    cfg = replace(tfi.DEFAULT_CONFIG, capital_base_float=capital)
    orig_build = tfi._build_strategy_obj

    def build(config_obj, rw, cr, strategy_class_obj=tfi.TacticalYieldStrategy):
        s = strategy_class(
            name=tfi.strategy_name_for_config_str(config_obj), benchmarks=config_obj.benchmark_tuple,
            rebalance_weight_df=rw, cash_return_ser=cr, tradeable_asset_list=config_obj.tradeable_asset_tuple,
            capital_base=config_obj.capital_base_float, slippage=config_obj.slippage_per_side_float,
            commission_per_share=config_obj.commission_per_share_float,
            commission_minimum=config_obj.commission_minimum_float,
            # These checks document the cb29d4f (pre-BIL) ledger; pin it explicitly.
            cash_vehicle_str=tfi.CASH_VEHICLE_DGS3MO_ACCRUAL_STR,
            dividend_withholding_rate_float=0.0,
        )
        if raw_units:
            s.historical_share_units_bool = True
        if withholding is not None:
            s.configure_dividend_cash_ledger(enabled_bool=True, withholding_rate_float=withholding)
        return s

    tfi._build_strategy_obj = build
    try:
        return tfi._run_strategy(
            config_obj=cfg, execution_price_df=execution_price_df, signal_df=signal_df,
            rebalance_weight_df=rebalance_weight_df, cash_return_ser=cash_return_ser, fred_snapshot_tuple=snaps,
            backtest_start_date_str=FULL_START, end_date_str=None, show_progress_bool=False,
        )
    finally:
        tfi._build_strategy_obj = orig_build


class IbkrCashStrategy(tfi.TacticalYieldStrategy):
    """Cash interest as IBKR pays it (approximation): benchmark - 0.5pp, nothing on the first USD 10K,
    and for NAV < USD 100K the rate scaled by NAV/100K. DGS3MO stands in for the benchmark rate."""

    spread_float = 0.005
    small_account_bool = True

    def _accrue_positive_cash_interest_float(self) -> float:
        t = pd.Timestamp(self.current_bar)
        if t in self.cash_interest_processed_date_set:
            return 0.0
        base_ret = float(self.cash_return_ser.loc[t])
        prior = self.cash_return_ser.index[self.cash_return_ser.index.get_loc(t) - 1] if self.cash_return_ser.index.get_loc(t) > 0 else t
        days = max((t - prior).days, 1)
        annual = base_ret * 365.0 / days
        annual_eff = max(annual - self.spread_float, 0.0)
        nav = float(self.total_value)
        cash = max(float(self.cash), 0.0)
        if self.small_account_bool:
            cash = max(cash - 10_000.0, 0.0)
            annual_eff *= min(nav / 100_000.0, 1.0)
        interest = cash * annual_eff * days / 365.0
        self.cash += interest
        self.cash_interest_total_float += interest
        self.cash_interest_processed_date_set.add(t)
        return interest


class ZeroCashStrategy(tfi.TacticalYieldStrategy):
    def _accrue_positive_cash_interest_float(self) -> float:
        self.cash_interest_processed_date_set.add(pd.Timestamp(self.current_bar))
        return 0.0


class LargeAccountIbkrCash(IbkrCashStrategy):
    small_account_bool = False


def switch_trades(strategy_obj) -> set:
    tx = strategy_obj.get_transactions()
    out, pos = set(), {}
    for _, r in tx.sort_values("bar").iterrows():
        a = str(r["asset"])
        b = pos.get(a, 0.0)
        pos[a] = b + float(r["amount"])
        if (abs(b) < 1e-9) != (abs(pos[a]) < 1e-9):
            out.add((str(pd.Timestamp(r["bar"]).date()), a))
    return out


def rescale(df, sym, k):
    df = df.copy()
    for f in ("Open", "High", "Low", "Close", "Dividend"):
        if (sym, f) in df.columns:
            df[(sym, f)] = df[(sym, f)].astype(np.float64) / k
    if (sym, "Volume") in df.columns:
        df[(sym, "Volume")] = df[(sym, "Volume")].astype(np.float64) * k
    return df


class PlantedNominalSizing(tfi.TacticalYieldStrategy):
    """Positive control: a planted price-level rule in the execution path (halve a sleeve when its adjusted
    Close_T > 60), i.e. the classic split-dependent threshold the harness must catch."""

    def iterate(self, data_df, close_row_ser, open_price_ser):
        if close_row_ser is None or self.current_bar not in self.rebalance_weight_df.index:
            return super().iterate(data_df, close_row_ser, open_price_ser)
        saved = self.rebalance_weight_df
        row = saved.loc[self.current_bar].copy()
        for a in self.tradeable_asset_list:
            if float(close_row_ser[(a, "Close")]) > 60.0:
                row[a] = float(row[a]) * 0.5
        self.rebalance_weight_df = saved.copy()
        self.rebalance_weight_df.loc[self.current_bar] = row
        try:
            return super().iterate(data_df, close_row_ser, open_price_ser)
        finally:
            self.rebalance_weight_df = saved


def a2_engine(data_tuple) -> dict:
    base_exec = data_tuple[0]
    res = {}
    for label, cls in (("HEAD", tfi.TacticalYieldStrategy), ("PC_nominal_sizing", PlantedNominalSizing)):
        s0 = run_on_data(data_tuple, strategy_class=cls)
        m0 = tb.metrics(tb.nav_ser(s0), FULL_START, BOOK_END)
        sw0 = switch_trades(s0)
        cases = []
        for sym in ("IEF", "LQD", "$SPXTR"):
            for k in (40.0, 0.1, 1.5):
                s1 = run_on_data(data_tuple, strategy_class=cls, exec_df=rescale(base_exec, sym, k))
                m1 = tb.metrics(tb.nav_ser(s1), FULL_START, BOOK_END)
                cases.append({"symbol": sym, "k": k, "switch_trades_equal": switch_trades(s1) == sw0,
                              "cagr_diff_pp": m1["cagr_pct"] - m0["cagr_pct"], "sharpe_diff": m1["sharpe"] - m0["sharpe"]})
        res[label] = {"base": m0, "cases": cases,
                      "pass_abs_cagr_lt_0p01pp": int(sum(abs(c["cagr_diff_pp"]) < 0.01 for c in cases)), "total": len(cases)}
        print("A2", label, res[label]["pass_abs_cagr_lt_0p01pp"], flush=True)
    return res


def a3_truncation(yield_df, session_index) -> dict:
    import exchange_calendars as xc

    cal = xc.get_calendar("XNYS", start="2002-01-01", end="2027-12-31")
    full_sig, full_w = tfi.build_month_end_signal_and_weight_df(yield_df, session_index, "2026-07")
    full_w_by_d = full_w.reset_index().set_index("decision_date")

    def leaky_yields(y):
        return y.shift(-1).dropna(how="all")

    cut_list = ["2008-09-30", "2008-10-15", "2012-03-30", "2016-10-31", "2016-12-30", "2020-03-31", "2020-05-29",
                "2021-05-28", "2022-04-29", "2024-03-28", "2024-11-29", "2026-02-27", "2026-07-31", "2026-08-19"]
    cols = ["observation_date", "term_spread_float", "credit_spread_float"]
    cases = []
    for t_str in cut_list:
        T = pd.Timestamp(t_str)
        nxt = pd.Timestamp(cal.next_session(T)).tz_localize(None) if pd.Timestamp(cal.next_session(T)).tzinfo else pd.Timestamp(cal.next_session(T))
        sess_T = session_index[session_index <= T].append(pd.DatetimeIndex([nxt]))
        month_end_bool = bool(nxt.to_period("M") != T.to_period("M"))
        last_month = str(T.to_period("M") if month_end_bool else (T.to_period("M") - 1))
        prior_sess = session_index[session_index < T][-1]
        for fred_label, y_T in (("release_model_le_T_minus_1_session", yield_df[yield_df.index <= prior_sess]),
                                ("lenient_le_T_plus_3", yield_df[yield_df.index <= T + pd.Timedelta(days=3)])):
            for fn_label in ("HEAD", "PC_leak_obs_T_via_shift"):
                y_use = y_T if fn_label == "HEAD" else leaky_yields(y_T)
                y_full = yield_df if fn_label == "HEAD" else leaky_yields(yield_df)
                try:
                    sig_T, w_T = tfi.build_month_end_signal_and_weight_df(y_use, sess_T, last_month)
                    f_sig, f_w = (full_sig, full_w) if fn_label == "HEAD" else tfi.build_month_end_signal_and_weight_df(y_full, session_index, "2026-07")
                except Exception as exc:  # noqa: BLE001
                    cases.append({"T": t_str, "fred": fred_label, "fn": fn_label, "equal": False, "raised": repr(exc)[:160]})
                    continue
                f_sig = f_sig.loc[f_sig.index <= T]
                idx = sig_T.index
                eq = bool(idx.equals(f_sig.index))
                row_T_equal = None
                if eq:
                    a = sig_T[cols + ["term_threshold_float", "credit_threshold_float", "term_state_float", "credit_state_float"]]
                    b = f_sig.loc[idx, a.columns]
                    eq = bool((a["observation_date"] == b["observation_date"]).all()) and bool(
                        np.allclose(a.drop(columns="observation_date").to_numpy(float), b.drop(columns="observation_date").to_numpy(float), atol=1e-12)
                    )
                    if month_end_bool:
                        row_T_equal = bool(
                            (a.loc[T, "observation_date"] == b.loc[T, "observation_date"])
                            and np.allclose(a.loc[T].drop("observation_date").to_numpy(float), b.loc[T].drop("observation_date").to_numpy(float), atol=1e-12)
                        )
                cases.append({"T": t_str, "month_end": month_end_bool, "fred": fred_label, "fn": fn_label,
                              "n_decisions": int(len(idx)), "all_decisions_le_T_equal": eq, "row_T_equal": row_T_equal})
    summary = {}
    for fn_label in ("HEAD", "PC_leak_obs_T_via_shift"):
        for fred_label in ("release_model_le_T_minus_1_session", "lenient_le_T_plus_3"):
            sub = [c for c in cases if c["fn"] == fn_label and c["fred"] == fred_label]
            summary[f"{fn_label}|{fred_label}"] = {"cutoffs": len(sub), "all_equal": int(sum(c.get("all_decisions_le_T_equal", False) for c in sub))}
    return {"summary": summary, "cases": cases}


def a5_extra_session_lag(data_tuple) -> dict:
    execution_price_df, yield_df, signal_df, rebalance_weight_df, cash_return_ser, snaps = data_tuple
    session_index = pd.DatetimeIndex(execution_price_df.index)
    orig_prev = tfi.previous_session

    def lagged_select(decision_date_ts, yield_df_, session_index_):
        # one extra session: pretend the decision is made with information of session T-1
        valid = yield_df_.loc[:, list(tfi.FRED_SERIES_ID_TUPLE)].dropna()
        p2 = orig_prev(orig_prev(decision_date_ts, session_index_), session_index_)
        return pd.Timestamp(valid.index[valid.index <= p2][-1])

    orig_sel = tfi.select_publication_safe_observation_date
    tfi.select_publication_safe_observation_date = lagged_select
    try:
        sig_lag, w_lag = tfi.build_month_end_signal_and_weight_df(yield_df, session_index, "2026-07")
    finally:
        tfi.select_publication_safe_observation_date = orig_sel
    base_w = rebalance_weight_df.reset_index().set_index("decision_date")
    lag_w = w_lag.reset_index().set_index("decision_date")
    diff = (base_w[["IEF", "LQD"]] - lag_w.loc[base_w.index, ["IEF", "LQD"]]).abs().max(axis=1) > 0
    flips = [str(d.date()) for d in base_w.index[diff.to_numpy()]]
    lag_tuple = (execution_price_df, yield_df, sig_lag, w_lag, cash_return_ser, snaps)
    s_lag = run_on_data(lag_tuple)
    return {"n_decisions": int(len(base_w)), "flips": flips, "n_flips": len(flips),
            "pre_2014_flips": [f for f in flips if f < "2014-04-30"], "metrics_lagged": m_all(s_lag)}


def a5_alfred_spot_check() -> dict:
    manifest, snaps = tfi.load_alfred_point_in_time_snapshots(tfi.DEFAULT_CONFIG)
    sample = [pd.Timestamp(d) for d in ["2014-04-30", "2014-11-28", "2016-09-30", "2016-10-31", "2016-12-30", "2017-01-31",
                                        "2017-02-28", "2017-03-31", "2017-06-30", "2020-03-31", "2020-06-30", "2024-11-29",
                                        "2026-02-27", "2026-07-31"]]
    out = {"sample": [str(d.date()) for d in sample], "series": {}}
    cache_path = tb.CACHE / "tfi_alfred_spot.pkl"
    cached = pickle.loads(cache_path.read_bytes()) if cache_path.exists() else {}
    for sid in tfi.FRED_SERIES_ID_TUPLE:
        if sid not in cached:
            cached[sid], _rec = fetch_alfred_vintage_dict(sid, sample, batch_size_int=12, pause_seconds_float=1.0)
            cache_path.write_bytes(pickle.dumps(cached))
        rows = []
        for v in sample:
            fetched = cached[sid][v]
            stored = snaps[sid].value_ser_as_of(v)
            same_index = bool(fetched.index.equals(stored.index))
            max_abs = float((fetched - stored.reindex(fetched.index)).abs().max()) if same_index else np.nan
            rows.append({"vintage": str(v.date()), "fetched_last_obs": str(fetched.index[-1].date()),
                         "stored_last_obs": str(stored.index[-1].date()), "same_index": same_index, "max_abs_diff": max_abs})
        out["series"][sid] = rows
    out["all_equal"] = bool(all(r["same_index"] and r["max_abs_diff"] == 0 for rs in out["series"].values() for r in rs))
    return out


def main() -> None:
    out = {}
    # --- baseline & PIT through the module
    s_frozen = run()
    out["frozen"] = m_all(s_frozen)
    s_frozen2 = run()
    out["determinism_bit_identical"] = bool(tb.nav_ser(s_frozen).equals(tb.nav_ser(s_frozen2)))
    out["frozen_research_basis"] = s_frozen.research_metric_basis_dict
    for pol in ("decision_date", "previous_session"):
        s_p = run(fred_data_mode_str="alfred_point_in_time", alfred_vintage_policy_str=pol, stale_input_policy_str="block_and_hold")
        out[f"pit_{pol}"] = m_all(s_p)
        sig_p = s_p.month_end_signal_df
        out[f"pit_{pol}_blocked"] = [str(d.date()) for d in sig_p.index[sig_p["stale_input_blocked_bool"].to_numpy(bool)]]
        fz = s_frozen.month_end_signal_df
        common = sig_p.index[(~sig_p["stale_input_blocked_bool"].to_numpy(bool)) & (sig_p.index >= pd.Timestamp("2014-04-30"))]
        flip = (fz.loc[common, ["term_state_float", "credit_state_float"]] != sig_p.loc[common, ["term_state_float", "credit_state_float"]]).any(axis=1)
        out[f"pit_{pol}_usable_2014plus"] = int(len(common))
        out[f"pit_{pol}_flips_vs_frozen"] = [str(d.date()) for d in common[flip.to_numpy()]]
        out[f"pit_{pol}_age_counts"] = sig_p.loc[sig_p.index >= pd.Timestamp("2014-04-30"), "observation_age_sessions_int"].value_counts().to_dict()
    out["frozen_age_counts"] = s_frozen.month_end_signal_df["observation_age_sessions_int"].value_counts().to_dict()
    print("baseline", out["frozen"]["book"], flush=True)
    tb.write_json("tfi_checks_partial.json", out)

    data_tuple = tfi.get_tactical_yield_data(tfi.DEFAULT_CONFIG)
    # --- accounting
    acc = {}
    for cap in (30_000.0, 1_000_000.0, 10_000_000.0):
        acc[f"capital_{int(cap)}"] = m_all(run_on_data(data_tuple, capital=cap))
    acc["raw_share_units"] = m_all(run_on_data(data_tuple, raw_units=True))
    s_w = run_on_data(data_tuple, withholding=0.25)
    acc["withholding_25pct"] = m_all(s_w)
    acc["dividends_gross_total_100k_start"] = float(s_frozen.dividend_cash_gross_total_float)
    acc["zero_cash_rate"] = m_all(run_on_data(data_tuple, strategy_class=ZeroCashStrategy))
    acc["ibkr_cash_large_account_minus_50bp"] = m_all(run_on_data(data_tuple, strategy_class=LargeAccountIbkrCash, capital=1_000_000.0))
    acc["ibkr_cash_small_account_30k"] = m_all(run_on_data(data_tuple, strategy_class=IbkrCashStrategy, capital=30_000.0))
    acc["ibkr_cash_small_account_30k_plus_withholding"] = m_all(
        run_on_data(data_tuple, strategy_class=IbkrCashStrategy, capital=30_000.0, withholding=0.25)
    )
    res = s_frozen.results
    frac = res["cash"].astype(float) / res["total_value"].astype(float)
    acc["cash"] = {"negative_cash_days": int((res["cash"].astype(float) < 0).sum()), "days": int(len(res)),
                   "min_cash_frac": float(frac.min()), "mean_cash_frac": float(frac.mean()),
                   "mean_cash_frac_last3y": float(frac[frac.index >= pd.Timestamp("2023-08-19")].mean())}
    tx = s_frozen.get_transactions()
    yrs = (pd.Timestamp(BOOK_END) - pd.Timestamp(FULL_START)).days / 365.25
    acc["orders_per_year"] = float(len(tx) / yrs)
    acc["orders_per_year_last3y"] = float((pd.to_datetime(tx["bar"]) >= pd.Timestamp("2023-08-19")).sum() / 3.0)
    out["accounting"] = acc
    print("acc", {k: v["book"]["cagr_pct"] for k, v in acc.items() if isinstance(v, dict) and "book" in v}, flush=True)
    tb.write_json("tfi_checks_partial.json", out)

    out["A2_engine"] = a2_engine(data_tuple)
    out["A3_truncation"] = a3_truncation(data_tuple[1], pd.DatetimeIndex(data_tuple[0].index))
    print("A3", out["A3_truncation"]["summary"], flush=True)
    out["A5_extra_session_lag"] = a5_extra_session_lag(data_tuple)
    print("A5 lag", out["A5_extra_session_lag"]["n_flips"], flush=True)
    out["A5_alfred_spot_check"] = a5_alfred_spot_check()
    print("A5 spot", out["A5_alfred_spot_check"]["all_equal"], flush=True)
    tb.write_json("tfi_checks.json", out)


if __name__ == "__main__":
    main()
