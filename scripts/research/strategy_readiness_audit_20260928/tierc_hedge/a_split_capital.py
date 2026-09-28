"""A2 future-split invariance (full engine, real data), A4 split-harness positive control, A9 capital scaling and
A10 determinism for CTC, VIXM and Trinity.

A2: one symbol's full history rescaled as if a k:1 split happened after the last bar (k in 40, 0.1, 1.5):
    OHLC and Dividend / k, Volume * k, Unadjusted Close and Turnover nominal. For CTC the TOTALRETURN signal namespace
    of the same symbol is rescaled too (a split moves both series).
Decision objects compared with the baseline:
    target  : recorded daily target weights on rebalance days (CTC/VIXM: every day; TRIN: rebalance days)
    rebal   : the set of rebalance (order-cycle) dates
    fills   : the (date, asset, side) fill ledger
Share counts are adjusted-unit quantities (legacy engine mode) and legitimately scale with k; fees in adjusted units
are reported as a metric delta.

Usage: python a_split_capital.py [vixm|ctc|trin] [split|pc|capital|raw]
"""
from __future__ import annotations

import pickle
import sys
import time

import numpy as np
import pandas as pd

import tc_common as c

NAME = sys.argv[1]
PARTS = sys.argv[2:] or ["split", "pc", "capital"]

LOAD = {"vixm": c.load_vixm, "ctc": c.load_ctc_workaround, "trin": c.load_trin}[NAME]
RUN = {"vixm": c.run_vixm, "ctc": c.run_ctc, "trin": c.run_trin}[NAME]
SYMS = {"vixm": ("VIXM", "SHY"), "ctc": ("SPY", "TLT", "SHY", "USO"), "trin": ("VTI", "TLT", "GLD", "BIL")}[NAME]
KS = (40.0, 0.1, 1.5)

df = LOAD()


def snapshot(s) -> dict:
    tx = s.get_transactions().copy()
    tx["bar"] = pd.to_datetime(tx["bar"])
    daily = getattr(s, "daily_target_weights", None)
    rebal = getattr(s, "rebalance_target_weight_df", None)
    if rebal is None:  # Trinity: rebalance days = days with orders queued at Close_T -> fills next session
        rebal = None
    return {"tx": tx, "daily": daily, "rebal": rebal, "tv": s.results["total_value"].astype(float),
            "commission": float(tx["commission"].sum()), "cash": s.results["cash"].astype(float)}


def tx_keys(tx):
    return set(zip(tx["bar"], tx["asset"], np.sign(tx["amount"].astype(float)).astype(int)))


def weight_diff(a, b, atol=1e-9):
    if a is None or b is None:
        return None
    idx = a.index.intersection(b.index)
    cols = [x for x in a.columns if x in b.columns]
    d = (a.loc[idx, cols].astype(float).fillna(0) - b.loc[idx, cols].astype(float).fillna(0)).abs().max(axis=1)
    return {"n": int(len(idx)), "n_diff": int((d > atol).sum()), "max_abs": float(d.max()) if len(d) else 0.0,
            "only_a": int(len(a.index.difference(b.index))), "only_b": int(len(b.index.difference(a.index))),
            "first": [str(t.date()) for t in d[d > atol].index[:5]]}


def compare(base, other) -> dict:
    kb, ko = tx_keys(base["tx"]), tx_keys(other["tx"])
    out = {
        "fills_base": len(kb), "fills_other": len(ko), "fills_only_base": len(kb - ko), "fills_only_other": len(ko - kb),
        "first_fill_diffs": sorted([f"{d.date()} {a} {s}" for d, a, s in (kb ^ ko)])[:6],
        "target_daily": weight_diff(other["daily"], base["daily"]),
        "target_rebal": weight_diff(other["rebal"], base["rebal"]) if base["rebal"] is not None else None,
        "final_tv_rel_diff": float(other["tv"].iloc[-1] / base["tv"].iloc[-1] - 1.0),
        "cagr_base": c.metrics(base["tv"].pct_change().dropna())["cagr_pct"],
        "cagr_other": c.metrics(other["tv"].pct_change().dropna())["cagr_pct"],
        "commission_base": base["commission"], "commission_other": other["commission"],
    }
    out["cagr_delta_pp"] = out["cagr_other"] - out["cagr_base"]
    rb = {d for d, _, _ in kb}
    ro = {d for d, _, _ in ko}
    out["fill_dates_only_base"] = len(rb - ro)
    out["fill_dates_only_other"] = len(ro - rb)
    out["decision_identical"] = (
        len(kb ^ ko) == 0
        and (out["target_daily"] is None or NAME == "trin" or out["target_daily"]["n_diff"] == 0)
    )
    return out


t0 = time.time()
base_s = RUN(df)
base = snapshot(base_s)
print("baseline", round(time.time() - t0, 1), flush=True)
result = {"strategy": NAME}

if "capital" in PARTS:
    # A10 determinism
    rep = snapshot(RUN(df))
    result["A10_determinism_bit_identical"] = bool(np.array_equal(rep["tv"].to_numpy(), base["tv"].to_numpy()))
    # A9 capital scaling
    cap = {}
    for capital in (30_000.0, 1_000_000.0, 10_000_000.0):
        o = snapshot(RUN(df, capital=capital))
        r0 = base["tv"].pct_change().dropna()
        r1 = o["tv"].pct_change().dropna()
        cap[str(int(capital))] = {
            "cagr_pct": c.metrics(r1)["cagr_pct"], "sharpe": c.metrics(r1)["sharpe"],
            "cagr_last3y": c.metrics(r1, start="2023-09-25")["cagr_pct"],
            "corr_daily_vs_100k": float(np.corrcoef(r0.to_numpy(), r1.reindex(r0.index).to_numpy())[0, 1]),
            "commission_frac_of_capital": o["commission"] / capital,
            "min_cash_frac": float((o["cash"] / o["tv"]).min()),
        }
        print("capital", capital, cap[str(int(capital))], flush=True)
    cap["100000"] = {"cagr_pct": c.metrics(base["tv"].pct_change().dropna())["cagr_pct"],
                     "sharpe": c.metrics(base["tv"].pct_change().dropna())["sharpe"],
                     "cagr_last3y": c.metrics(base["tv"].pct_change().dropna(), start="2023-09-25")["cagr_pct"]}
    result["A9_capital"] = cap

if "split" in PARTS:
    rows = []
    for sym in SYMS:
        for k in KS:
            if NAME == "ctc" and sym == "USO" and k != 40.0:
                continue
            t = time.time()
            pdf = c.rescale_namespace(df, sym, k)
            if NAME == "ctc":
                pdf = c.rescale_namespace(pdf, c.ctc.signal_namespace_str(sym), k)
            o = snapshot(RUN(pdf))
            cmp = compare(base, o)
            cmp.update({"symbol": sym, "k": k, "secs": round(time.time() - t, 1)})
            rows.append(cmp)
            print(sym, k, cmp["decision_identical"], cmp["fills_only_base"], cmp["fills_only_other"],
                  cmp["cagr_delta_pp"], flush=True)
    result["A2_split"] = rows

if "raw" in PARTS and NAME == "trin":
    # Trinity in raw historical share units (engine opt-in): split invariance should be exact.
    def run_raw(pdf):
        return c.run_trin(pdf, historical_share_units_bool=True)
    raw_base = snapshot(run_raw(df))
    result["raw_vs_legacy_baseline"] = compare(base, raw_base)
    rows = []
    for sym, k in (("VTI", 40.0), ("BIL", 40.0), ("GLD", 0.1)):
        o = snapshot(run_raw(c.rescale_namespace(df, sym, k)))
        cmp = compare(raw_base, o)
        cmp.update({"symbol": sym, "k": k, "mode": "raw"})
        rows.append(cmp)
        print("raw", sym, k, cmp["decision_identical"], cmp["fills_only_base"], cmp["cagr_delta_pp"], flush=True)
    result["A2_split_raw"] = rows

if "pc" in PARTS:
    # A4 positive control for the split harness: a variant whose rule reads a nominal price level.
    if NAME == "vixm":
        thr = float(df[("VIXM", "Close")].median())

        class PCVixm(c.vixm.VixmBackwardationStrategy):
            def compute_signals(self, pricing_data_df):
                sig = super().compute_signals(pricing_data_df)
                key = (c.vixm.VIX_SIGNAL_NAMESPACE_STR, c.vixm.STATE_FIELD_STR)
                lvl = pricing_data_df[("VIXM", "Close")] < thr
                sig[key] = sig[key].where(sig[key].isna(), sig[key] * lvl.astype(float))
                return sig

        pc_run = lambda pdf: c.run_vixm(pdf, strategy_cls=PCVixm)  # noqa: E731
        sym = "VIXM"
    elif NAME == "ctc":
        thr = float(df[("SPY", "Close")].median())

        class PCCtc(c.ctc.CrisisTrendCoreStrategy):
            def compute_signals(self, pricing_data_df):
                sig = super().compute_signals(pricing_data_df)
                lvl = (pricing_data_df[("SPY", "Close")] < thr).astype(float)
                key = (c.ctc.signal_namespace_str("SPY"), c.ctc.DESIRED_WEIGHT_FIELD_STR)
                sig[key] = sig[key] * lvl
                return sig

        pc_run = lambda pdf: c.run_ctc(pdf, strategy_cls=PCCtc)  # noqa: E731
        sym = "SPY"
    else:
        thr = float(df[("VTI", "Close")].median())

        class PCTrin(c.trin.TrinityVolControlStrategy):
            def compute_signals(self, pricing_data_df):
                sig = super().compute_signals(pricing_data_df)
                lvl = (pricing_data_df[("VTI", "Close")] < thr)
                key = c.trin.MONTHLY_REBALANCE_FIELD_TUPLE
                sig[key] = sig[key].astype(bool) & lvl
                return sig

        def pc_run(pdf):
            cfg = c.trin.DEFAULT_CONFIG
            cal = c.trin._execution_calendar_index(pdf, cfg, None)
            s = c.trin._build_trinity_strategy(cfg, 100_000.0, PCTrin)
            c.run_daily(s, pdf, calendar=cal, show_progress=False, show_signal_progress_bool=False,
                        audit_override_bool=False)
            return s
        sym = "VTI"
    pc_base = snapshot(pc_run(df))
    pdf = c.rescale_namespace(df, sym, 40.0)
    if NAME == "ctc":
        pdf = c.rescale_namespace(pdf, c.ctc.signal_namespace_str(sym), 40.0)
    pc_k = snapshot(pc_run(pdf))
    cmpc = compare(pc_base, pc_k)
    result["A4_split_positive_control"] = {"symbol": sym, "k": 40.0, "caught": not cmpc["decision_identical"],
                                           "fills_only_base": cmpc["fills_only_base"],
                                           "fills_only_other": cmpc["fills_only_other"]}
    print("PC", result["A4_split_positive_control"], flush=True)

c.dump(result, f"{NAME}/a2_a9_a10_{'_'.join(PARTS)}.json")
print("done", round(time.time() - t0, 1))
