"""A2/A3/A4 for both HPI variants on REAL Norgate data.

Parts (usage: uv run python hpi_invariance.py <part> [variant] [window_index]):

  split <variant> <w>     Engine-level future-split invariance on window w. Reference run vs the same run with the
                          window's 5 most-traded names + real split names rescaled as if a k:1 split happened after
                          the last row (k in 40, 0.1, 1.5): OHLC and Dividend / k, Volume * k, Unadjusted Close and
                          Turnover kept nominal. Compared: per signal date the exit set and the ORDERED entry list
                          (rank) from iterate(), plus the engine (date, asset, side) fill set. Production accounting
                          (adjusted-unit whole shares) and historical_share_units (hsu) mode. Positive control PC2:
                          a ranking key rebuilt as Unadjusted Close x adjusted Volume (the E-01 bug) must FAIL.
  prefix <variant> <w>    Engine-level truncation: run to cut T vs run to window end; every iterate() record with
                          signal date < T and every fill <= T must be identical (exact values).
  rowT                    Feature-row truncation at >= 8 protocol cut-offs: compute_signals on the REAL loader output
                          load_exact_hpi_inputs(end_date_str=T) vs the full-history frame; row T of every feature of
                          every symbol, the candidate list (both variants) and the exit flags of every member must be
                          identical. Also checks loader(T) == loader(full).loc[:T]. Positive control PC1: a feature
                          reading IBS_(T+1) must be caught by rowT and is shown to be MISSED by the engine prefix test.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

import hpi_common as hc
from strategies.hpi import stateful_long as hpi_mod

WINDOWS = [("2008-06-02", "2009-06-30"), ("2020-01-02", "2020-12-31"), ("2025-01-02", "2026-09-25")]
SPLIT_NAMES = ("AAPL", "NVDA", "AMZN", "GOOGL", "TSLA", "AVGO", "NFLX", "CMG", "WMT", "LRCX")
FACTORS = (40.0, 0.1, 1.5)
CUTOFFS = {
    "crisis_2008_mid_month": "2008-10-10",
    "crisis_2020_mid_month": "2020-03-16",
    "month_end_weekend_before_good_friday": "2024-03-28",
    "month_end_on_weekend": "2025-05-30",
    "mid_month": "2025-06-13",
    "month_end_before_holiday": "2025-12-31",
    "last_completed_month_end": "2026-08-31",
    "first_session_of_month": "2026-09-01",
    "current_partial_month_prev": "2026-09-24",
    "current_partial_month_last_bar": "2026-09-25",
}
PRICE_FIELDS = ("Open", "High", "Low", "Close", "Dividend")


def rescale(pricing_df: pd.DataFrame, symbols, k: float) -> pd.DataFrame:
    out = pricing_df.copy()
    for s in symbols:
        for f in PRICE_FIELDS:
            if (s, f) in out.columns:
                out[(s, f)] = out[(s, f)] / k
        if (s, "Volume") in out.columns:
            out[(s, "Volume")] = out[(s, "Volume")] * k
    out.attrs.update(pricing_df.attrs)
    return out


class PC2UnadjTurnover(hc.RecordingHPIStrategy):
    """Positive control for A2: ranking key = Unadjusted Close x adjusted Volume (not split-invariant)."""

    def compute_signals(self, pricing_data_df):
        out = super().compute_signals(pricing_data_df)
        for s in out.columns.get_level_values(0).unique():
            if (s, "Unadjusted Close") in out.columns and (s, "Volume") in out.columns:
                out[(s, "Turnover")] = out[(s, "Unadjusted Close")] * out[(s, "Volume")]
        return out


class PC1IbsTomorrow(hc.RecordingHPIStrategy):
    """Positive control for A3: entry IBS reads IBS_(T+1) (one-bar look-ahead)."""

    def compute_signals(self, pricing_data_df):
        out = super().compute_signals(pricing_data_df)
        cols = [c for c in out.columns if c[1] == "ibs_value_ser"]
        out[cols] = out[cols].shift(-1)
        return out


def record_table(records) -> dict:
    return {r["signal_date"]: (tuple(r["exits"]), tuple(r["entries"]), tuple(r["entry_values"])) for r in records}


def fills(strategy) -> pd.DataFrame:
    tx = strategy.get_transactions().copy()
    if len(tx) == 0:
        return pd.DataFrame(columns=["date", "asset", "side", "notional"])
    tx["notional"] = tx["amount"].astype(float) * tx["price"].astype(float)
    tx["side"] = np.sign(tx["amount"].astype(float)).astype(int)
    g = tx.groupby(["bar", "asset"]).agg(side=("side", "first"), notional=("notional", "sum")).reset_index()
    return g.rename(columns={"bar": "date"})


def compare_records(ref: dict, cand: dict, before=None, value_rtol=None) -> dict:
    dates = sorted(set(ref) | set(cand))
    if before is not None:
        dates = [d for d in dates if d < pd.Timestamp(before)]
    bad, value_bad, max_rel = [], 0, 0.0
    for d in dates:
        a, b = ref.get(d), cand.get(d)
        if a is None or b is None or a[0] != b[0] or a[1] != b[1]:
            bad.append(d)
            continue
        if len(a[2]):
            rel = float(np.max(np.abs(np.array(a[2]) - np.array(b[2])) / np.abs(np.array(a[2]))))
            max_rel = max(max_rel, rel)
            if value_rtol is not None and rel > value_rtol:
                value_bad += 1
    return {"n_dates": len(dates), "n_decision_diff": len(bad), "first_diff": [str(d.date()) for d in bad[:5]],
            "examples": [{"date": str(d.date()), "ref": ref.get(d), "cand": cand.get(d)} for d in bad[:2]],
            "max_rel_entry_value_diff": max_rel, "n_entry_value_beyond_tol": value_bad,
            "n_entry_decisions": int(sum(len(v[1]) for v in ref.values())),
            "n_exit_decisions": int(sum(len(v[0]) for v in ref.values()))}


def compare_fills(ref: pd.DataFrame, cand: pd.DataFrame, end=None) -> dict:
    if end is not None:
        ref = ref[ref["date"] <= pd.Timestamp(end)]
        cand = cand[cand["date"] <= pd.Timestamp(end)]
    m = ref.merge(cand, on=["date", "asset", "side"], how="outer", indicator=True, suffixes=("_r", "_c"))
    both = m[m["_merge"] == "both"]
    rel = ((both["notional_c"] - both["notional_r"]).abs() / both["notional_r"].abs().clip(lower=1.0))
    return {"n_ref": int(len(ref)), "n_only_ref": int((m["_merge"] == "left_only").sum()),
            "n_only_cand": int((m["_merge"] == "right_only").sum()),
            "max_rel_notional_diff": float(rel.max()) if len(rel) else 0.0}


def part_split(variant: str, w: int) -> None:
    start, end = WINDOWS[w]
    data = hc.load_full_inputs()
    pricing = hc.subset_pricing(data["pricing_df"], data["universe"], start, end, extra=SPLIT_NAMES)
    rows, info = [], {"variant": variant, "window": [start, end],
                      "n_symbols": int(pricing.columns.get_level_values(0).nunique())}
    t0 = time.time()
    ref = {}
    for hsu in (False, True):
        s = hc.run(hc.make_strategy(variant, data["universe"], start=start, hsu=hsu), pricing, start, end)
        ref[hsu] = (record_table(s.records), fills(s))
    info["ref_runtime_s"] = round(time.time() - t0, 1)
    buys = ref[False][1][ref[False][1]["side"] > 0]
    top = buys["asset"].value_counts().index[:5].tolist()
    present = set(pricing.columns.get_level_values(0).astype(str))
    chosen = list(dict.fromkeys(top + [x for x in SPLIT_NAMES if x in present]))
    info["rescaled_symbols"] = chosen
    info["rescaled_symbols_traded_in_window"] = sorted(set(chosen) & set(ref[False][1]["asset"].astype(str)))
    for hsu in (False, True):
        for k in (FACTORS if not hsu else (40.0,)):
            cand_pricing = rescale(pricing, chosen, k)
            s = hc.run(hc.make_strategy(variant, data["universe"], start=start, hsu=hsu), cand_pricing, start, end)
            rc = compare_records(ref[hsu][0], record_table(s.records), value_rtol=1e-7 if hsu else None)
            fc = compare_fills(ref[hsu][1], fills(s))
            ok = rc["n_decision_diff"] == 0 and fc["n_only_ref"] == 0 and fc["n_only_cand"] == 0
            if hsu:
                ok = ok and rc["n_entry_value_beyond_tol"] == 0
            rows.append({"test": f"A2_split_{'hsu' if hsu else 'engine'}", "k": k, "passed": ok,
                         "records": rc, "fills": fc})
            print(rows[-1]["test"], k, ok, rc["n_decision_diff"], rc["max_rel_entry_value_diff"], fc)
    # PC2: must FAIL (difference detected) for k=40
    pc_ref = hc.run(hc.make_strategy(variant, data["universe"], start=start, cls=PC2UnadjTurnover), pricing, start, end)
    pc_cand = hc.run(hc.make_strategy(variant, data["universe"], start=start, cls=PC2UnadjTurnover),
                     rescale(pricing, chosen, 40.0), start, end)
    rc = compare_records(record_table(pc_ref.records), record_table(pc_cand.records))
    rows.append({"test": "A4_PC2_unadj_close_x_adj_volume_rank", "k": 40.0,
                 "caught": rc["n_decision_diff"] > 0, "records": rc})
    print("PC2 caught:", rc["n_decision_diff"] > 0, rc["n_decision_diff"])
    info["runtime_s"] = round(time.time() - t0, 1)
    hc.dump_json({"info": info, "rows": rows}, f"invariance/split_{variant}_w{w}.json")


def part_prefix(variant: str, w: int) -> None:
    start, end = WINDOWS[w]
    data = hc.load_full_inputs()
    pricing = hc.subset_pricing(data["pricing_df"], data["universe"], start, end)
    sessions = pricing.index[(pricing.index >= pd.Timestamp(start)) & (pricing.index <= pd.Timestamp(end))]
    full = hc.run(hc.make_strategy(variant, data["universe"], start=start), pricing, start, end)
    ref_rec, ref_fills = record_table(full.records), fills(full)
    rows = []
    for back in (40, 120):
        cut = sessions[-1 - back]
        s = hc.run(hc.make_strategy(variant, data["universe"], start=start), pricing.loc[:cut], start, cut)
        rc = compare_records(ref_rec, record_table(s.records), before=cut, value_rtol=0.0)
        fc = compare_fills(ref_fills, fills(s), end=cut)
        ok = (rc["n_decision_diff"] == 0 and rc["n_entry_value_beyond_tol"] == 0 and fc["n_only_ref"] == 0
              and fc["n_only_cand"] == 0 and fc["max_rel_notional_diff"] == 0.0)
        rows.append({"test": "A3_engine_prefix", "cut": str(cut.date()), "passed": ok, "records": rc, "fills": fc})
        print(rows[-1])
    # PC1 through the engine prefix test (expected to be MISSED: only the decision at the last bar differs)
    pc_full = hc.run(hc.make_strategy(variant, data["universe"], start=start, cls=PC1IbsTomorrow), pricing, start, end)
    cut = sessions[-1 - 40]
    pc_cut = hc.run(hc.make_strategy(variant, data["universe"], start=start, cls=PC1IbsTomorrow),
                    pricing.loc[:cut], start, cut)
    rc = compare_records(record_table(pc_full.records), record_table(pc_cut.records), before=cut, value_rtol=0.0)
    fc = compare_fills(fills(pc_full), fills(pc_cut), end=cut)
    rows.append({"test": "A4_PC1_ibs_tomorrow_via_engine_prefix", "cut": str(cut.date()),
                 "caught": bool(rc["n_decision_diff"] or fc["n_only_ref"] or fc["n_only_cand"]),
                 "records": rc, "fills": fc,
                 "note": "engine prefix compares decisions strictly before the cut; a one-bar leak only changes the "
                         "decision AT the cut, which the truncated engine run never makes"})
    print(rows[-1]["test"], rows[-1]["caught"])
    hc.dump_json({"variant": variant, "window": [start, end], "rows": rows}, f"invariance/prefix_{variant}_w{w}.json")


def _row_decisions(strategy, row: pd.Series, date: pd.Timestamp) -> dict:
    members = hpi_mod.get_asof_universe_symbol_set(strategy.universe_df, date)
    strategy.previous_bar = date
    cands = strategy.get_opportunity_list(row, members)
    ibs = row.xs("ibs_value_ser", level=1) if "ibs_value_ser" in row.index.get_level_values(1) else pd.Series()
    rsi = row.xs("rsi2_value_ser", level=1) if "rsi2_value_ser" in row.index.get_level_values(1) else pd.Series()
    exit_flags = {s for s in members if (pd.notna(ibs.get(s)) and float(ibs.get(s)) > 0.9)
                  or (pd.notna(rsi.get(s)) and float(rsi.get(s)) > 90.0)}
    return {"candidates": cands, "exit_flags": sorted(exit_flags)}


def part_rowT() -> None:
    data = hc.load_full_inputs()
    full_pricing, full_universe = data["pricing_df"], data["universe"]
    rows = []
    signal_full = {}
    for variant in hc.VARIANTS:
        t0 = time.time()
        s = hc.make_strategy(variant, full_universe)
        signal_full[variant] = s.compute_signals(full_pricing.copy())
        print(variant, "full compute_signals", round(time.time() - t0, 1), "s")
    pc_full = hc.make_strategy("vote", full_universe, cls=PC1IbsTomorrow).compute_signals(full_pricing.copy())
    for label, cut_str in CUTOFFS.items():
        cut = pd.Timestamp(cut_str)
        t0 = time.time()
        _, uni_t, pr_t = hpi_mod.load_exact_hpi_inputs("S&P 500", hc.BENCH, "1998-01-01", cut_str)
        load_s = time.time() - t0
        trunc_full = full_pricing.loc[:cut]
        cols = trunc_full.columns.intersection(pr_t.columns)
        loader_equal = bool(pr_t.index.equals(trunc_full.index) and set(pr_t.columns) <= set(trunc_full.columns)
                            and pr_t[cols].equals(trunc_full[cols]))
        extra_cols = sorted({c[0] for c in set(pr_t.columns) - set(trunc_full.columns)})
        uni_equal = bool(uni_t.equals(full_universe.loc[:cut, list(uni_t.columns)]))
        members_equal = (hpi_mod.get_asof_universe_symbol_set(uni_t, cut)
                         == hpi_mod.get_asof_universe_symbol_set(full_universe, cut))
        row = {"label": label, "cut": cut_str, "load_s": round(load_s, 1),
               "loader_prices_equal_on_common_columns": loader_equal, "symbol_set_diff": extra_cols[:10],
               "n_symbol_set_diff": len(extra_cols), "universe_equal": uni_equal,
               "members_at_T_equal": members_equal}
        for variant in hc.VARIANTS:
            s_t = hc.make_strategy(variant, uni_t)
            sig_t = s_t.compute_signals(pr_t.copy())
            feat_cols = [c for c in sig_t.columns if c[1] not in ("Open", "High", "Low", "Close", "Volume",
                                                                   "Turnover", "Unadjusted Close", "Dividend")]
            a = sig_t.loc[cut, feat_cols].astype(float)
            b = signal_full[variant].loc[cut, feat_cols].astype(float)
            both_nan = a.isna() & b.isna()
            diff = (a - b).abs()
            max_diff = float(diff[~both_nan].max()) if (~both_nan).any() else 0.0
            nan_mismatch = int((a.isna() != b.isna()).sum())
            d_t = _row_decisions(s_t, sig_t.loc[cut], cut)
            s_f = hc.make_strategy(variant, full_universe)
            d_f = _row_decisions(s_f, signal_full[variant].loc[cut], cut)
            row[variant] = {"n_feature_cells": int(len(feat_cols)), "max_abs_feature_diff": max_diff,
                            "nan_pattern_mismatch": nan_mismatch,
                            "candidates_equal": d_t["candidates"] == d_f["candidates"],
                            "n_candidates": len(d_f["candidates"]), "top_candidates": d_f["candidates"][:5],
                            "exit_flags_equal": d_t["exit_flags"] == d_f["exit_flags"],
                            "n_exit_flags": len(d_f["exit_flags"])}
            row[variant]["passed"] = (max_diff <= 1e-9 and nan_mismatch == 0 and row[variant]["candidates_equal"]
                                      and row[variant]["exit_flags_equal"])
        # PC1 on the same row
        pc_t = hc.make_strategy("vote", uni_t, cls=PC1IbsTomorrow).compute_signals(pr_t.copy())
        ibs_cols = [c for c in pc_t.columns if c[1] == "ibs_value_ser"]
        a, b = pc_t.loc[cut, ibs_cols].astype(float), pc_full.loc[cut, ibs_cols].astype(float)
        row["PC1_caught"] = bool(((a.isna() != b.isna()) | ((a - b).abs() > 1e-9)).any()) if cut != full_pricing.index[-1] else None
        rows.append(row)
        print(json.dumps(row, default=str)[:600])
        hc.dump_json(rows, "invariance/rowT_truncation.json")


if __name__ == "__main__":
    part = sys.argv[1]
    if part == "split":
        part_split(sys.argv[2], int(sys.argv[3]))
    elif part == "prefix":
        part_prefix(sys.argv[2], int(sys.argv[3]))
    elif part == "rowT":
        part_rowT()
