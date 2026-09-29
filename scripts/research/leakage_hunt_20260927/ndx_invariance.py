"""(b) Future-action invariance and (c) truncation/prefix tests on REAL Norgate NDX data, both modules.

(b) For each decision date T the frame is truncated at T, then one symbol's whole truncated history is re-based
    with harness.rescale_symbol_history (k in 40, 0.1, 1.5).  Selected set, top-20 rank order and target weights
    (VXN exposure included) at T must be identical.  Extra cases: SPY (regime) and a joint random re-base of every
    symbol.  Positive control: the LEGACY formula (ATR not rebased = Unadjusted Close replaced by adjusted Close,
    exactly the pre-fb81e86 score ROC12 / ATR_adjusted) must be caught by the same test.
(c) Decision at T from data ending at T, and from data ending mid-month after T, must equal the decision at T from
    the full history (to the latest Norgate date).  Run for every month-end 2000-2026 for both modules.

Usage: uv run python scripts/research/leakage_hunt_20260927/ndx_invariance.py [--quick]
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd

import ndx_common as nc
from harness import DEFAULT_FACTOR_TUPLE, ResultLog, rescale_symbol_history

INV_DATES = ["2000-03-31", "2003-06-30", "2008-10-31", "2013-06-28", "2014-06-30", "2020-08-31", "2022-07-29",
             "2024-06-28", "2026-08-31"]
SPLIT_NAMES = ["AAPL", "NVDA", "TSLA", "AMZN", "GOOGL"]


@contextmanager
def legacy_formula(key: str):
    """Pre-fix score: feed adjusted Close where the module reads Unadjusted Close (rebase factor == 1)."""
    import importlib
    # AtrNormalizedNdxStrategy.compute_signals resolves get_unadjusted_close_df in the base module namespace.
    target = nc.mod(key) if key == "natr_vxn" else importlib.import_module("strategies.momentum.strategy_mo_atr_normalized_ndx")

    def fake(pricing_data_df, symbol_list):
        return pd.DataFrame({s: pricing_data_df[(s, "Close")] for s in symbol_list},
                            index=pricing_data_df.index).astype(float)
    originals = {target: target.get_unadjusted_close_df}
    target.get_unadjusted_close_df = fake
    try:
        yield
    finally:
        for m, fn in originals.items():
            m.get_unadjusted_close_df = fn


def members_at(universe_df: pd.DataFrame, ts) -> list[str]:
    row = universe_df.loc[:ts].iloc[-1]
    return sorted(row[row == 1].index.astype(str))


def invariance(data: dict, log: ResultLog, key: str, legacy: bool, rng: np.random.Generator) -> list[dict]:
    pricing, universe, vxn = data["pricing"], data["universe"], data["vxn"]
    rows = []
    for date_str in INV_DATES:
        T = pd.Timestamp(date_str)
        frame = pricing.loc[:T]
        vxn_T = vxn.loc[:T]
        ref = nc.decide(key, frame, universe, vxn_T, T)
        active = [s for s in members_at(universe, T) if (s, "Close") in frame.columns and pd.notna(frame.loc[T, (s, "Close")])]
        random_member = str(rng.choice([s for s in active if s not in SPLIT_NAMES]))
        rank1 = ref["top_rank"][0] if ref["top_rank"] else None
        symbols = SPLIT_NAMES + [random_member, "SPY"] + ([rank1] if rank1 and rank1 not in SPLIT_NAMES else [])
        symbols = list(dict.fromkeys(symbols))
        case_rows = []
        for symbol in symbols:
            if (symbol, "Close") not in frame.columns or frame[(symbol, "Close")].notna().sum() == 0:
                case_rows.append({"date": date_str, "symbol": symbol, "factor": None, "note": "no data before T"})
                continue
            for k in DEFAULT_FACTOR_TUPLE:
                cand = nc.decide(key, rescale_symbol_history(frame, symbol, k), universe, vxn_T, T)
                ok, detail = nc.same_decision(ref, cand)
                case_rows.append({"date": date_str, "symbol": symbol, "factor": k, "passed": ok,
                                  "in_rank20": symbol in ref["top_rank"], **({} if ok else detail)})
                if not legacy:
                    log.add("future_action_invariance", f"{date_str}|{symbol}|k={k}", ok,
                            {"in_rank20": symbol in ref["top_rank"], **({} if ok else detail)})
        # joint random re-base of every stock symbol + SPY
        factor_map = {s: float(np.exp(rng.uniform(np.log(0.05), np.log(50.0))))
                      for s in frame.columns.get_level_values(0).unique() if s not in ("$SPXTR", "$SPX")}
        joint = frame.copy()  # vectorised equivalent of rescale_symbol_history applied to every symbol
        for field, op in (("Open", "div"), ("High", "div"), ("Low", "div"), ("Close", "div"), ("Dividend", "div"),
                          ("Volume", "mul")):
            cols = [(s, field) for s in factor_map if (s, field) in joint.columns]
            factors = np.array([factor_map[c[0]] for c in cols])
            joint[cols] = joint[cols].to_numpy() / factors if op == "div" else joint[cols].to_numpy() * factors
        cand = nc.decide(key, joint, universe, vxn_T, T)
        ok, detail = nc.same_decision(ref, cand)
        case_rows.append({"date": date_str, "symbol": "ALL(random log-uniform 0.05..50)", "factor": None,
                          "passed": ok, **({} if ok else detail)})
        if not legacy:
            log.add("future_action_invariance", f"{date_str}|ALL_random", ok, {} if ok else detail)
        n_tested = sum(1 for r in case_rows if r.get("passed") is not None)
        n_fail = sum(1 for r in case_rows if r.get("passed") is False)
        rows.append({"date": date_str, "reference": ref, "cases": case_rows, "n_cases": n_tested, "n_fail": n_fail})
        print(f"[{key}{' LEGACY' if legacy else ''}] {date_str}: exposure={ref['exposure']:.3f} "
              f"selected={len(ref['selected'])} eligible={ref['n_eligible']} cases={n_tested} fail={n_fail}")
    return rows


def prefix_tests(data: dict, log: ResultLog, key: str, quick: bool) -> dict:
    pricing, universe, vxn = data["pricing"], data["universe"], data["vxn"]
    full_strategy, full_signals = nc.signals(key, pricing, universe, vxn)
    month_ends = nc.month_end_decision_dates(pricing)
    month_ends = month_ends[(month_ends >= "2000-01-31") & (month_ends <= "2026-08-31")]
    if quick:
        month_ends = month_ends[::12]
    index = pricing.index
    n_fail, fails, n = 0, [], 0
    mid_fail, detect_fail = 0, []
    for T in month_ends:
        ref = nc.decision_at(full_strategy, full_signals, T)
        # data ending exactly at T
        cand = nc.decide(key, pricing.loc[:T], universe.loc[:T], vxn.loc[:T], T)
        ok, detail = nc.same_decision(ref, cand)
        n += 1
        if not ok:
            n_fail += 1
            fails.append({"T": str(T.date()), **detail})
        # data ending mid-month after T (10 sessions later): the partial month must not create a decision,
        # and the decision at T must be unchanged
        pos = index.get_loc(T)
        mid = index[min(pos + 10, len(index) - 1)]
        if mid.to_period("M") == T.to_period("M"):
            mid = index[min(pos + 1, len(index) - 1)]
        strategy_mid, signals_mid = nc.signals(key, pricing.loc[:mid], universe.loc[:mid], vxn.loc[:mid], T)
        cand_mid = nc.decision_at(strategy_mid, signals_mid, T)
        ok_mid, detail_mid = nc.same_decision(ref, cand_mid)
        last_decision = nc.month_end_decision_dates(pricing.loc[:mid])[-1]
        spurious = last_decision != T
        if (not ok_mid) or spurious:
            mid_fail += 1
            detect_fail.append({"T": str(T.date()), "mid": str(mid.date()), "last_decision": str(last_decision.date()),
                                **({} if ok_mid else detail_mid)})
    log.add("prefix_end_at_T", f"{len(month_ends)} month-ends", n_fail == 0, {"n": n, "n_fail": n_fail, "fails": fails[:5]})
    log.add("prefix_end_mid_month", f"{len(month_ends)} month-ends", mid_fail == 0,
            {"n": n, "n_fail": mid_fail, "fails": detect_fail[:5]})
    print(f"[{key}] prefix: n={n} fail_at_T={n_fail} fail_mid={mid_fail}")
    return {"n": n, "n_fail_end_at_T": n_fail, "n_fail_mid_month": mid_fail, "fails": fails, "mid_fails": detect_fail}


def main() -> None:
    quick = "--quick" in sys.argv
    data = nc.load_data("trimmed")
    summary = {}
    for key in ("atr_vxn", "natr_vxn"):
        rng = np.random.default_rng(20260927)
        log = ResultLog(f"ndx_{key}")
        inv_rows = invariance(data, log, key, legacy=False, rng=rng)
        rng = np.random.default_rng(20260927)
        with legacy_formula(key):
            legacy_rows = invariance(data, log, key, legacy=True, rng=rng)
        n_leg = sum(r["n_cases"] for r in legacy_rows)
        n_leg_fail = sum(r["n_fail"] for r in legacy_rows)
        dates_caught = [r["date"] for r in legacy_rows if r["n_fail"] > 0]
        # positive control: the legacy score must be caught on every date where the regime gate is open
        open_dates = [r["date"] for r in legacy_rows if r["reference"]["n_eligible"] > 0]
        control_expected = key == "atr_vxn"
        control_ok = (set(open_dates) <= set(dates_caught)) if control_expected else True
        log.add("positive_control_legacy_formula", "caught on every gate-open date" if control_expected
                else "NATR is scale-free even without rebase (no leak expected)", control_ok,
                {"legacy_cases": n_leg, "legacy_cases_changed": n_leg_fail, "dates_caught": dates_caught,
                 "gate_open_dates": open_dates})
        pre = prefix_tests(data, log, key, quick)
        (nc.OUT / f"ndx_{key}__invariance.json").write_text(json.dumps(log.rows, indent=2, default=str), encoding="utf-8")
        (nc.OUT / f"invariance_detail_{key}.json").write_text(json.dumps(
            {"invariance": inv_rows, "legacy_control": legacy_rows, "prefix": pre}, indent=2, default=str), encoding="utf-8")
        summary[key] = {
            "invariance_cases": sum(r["n_cases"] for r in inv_rows),
            "invariance_fail": sum(r["n_fail"] for r in inv_rows),
            "legacy_control_cases": n_leg, "legacy_control_changed": n_leg_fail, "legacy_dates_caught": dates_caught,
            "gate_open_dates": open_dates,
            "prefix_n": pre["n"], "prefix_fail_end_at_T": pre["n_fail_end_at_T"],
            "prefix_fail_mid_month": pre["n_fail_mid_month"],
            "tests_pass": sum(r["passed"] for r in log.rows), "tests_fail": sum(not r["passed"] for r in log.rows),
        }
    (nc.OUT / "invariance_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
