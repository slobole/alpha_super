"""A2 future-split invariance, A3 truncation invariance and A4 positive controls on REAL Norgate data (research-only).

  inv <fam> <w>     A2.  Window w: reference engine run (production accounting and historical-share units) vs the same
                    run with the ENTIRE history of >= 8 symbols rescaled by k in {40, 0.1, 1.5} (aud_common.
                    rescale_symbol_history).  Rescaled = 5 most-bought names of the reference run + names held at the
                    window end + well-known real split names present.  PASS = identical (fill date, asset, side) set
                    AND identical recorded order intents (decision date, asset, kind, weight to 1e-9).
                    Positive control (A4): the same test with the candidate ranking switched to the adjusted-units
                    'Volume' field (scale-dependent by construction) must FAIL.
  prefix <fam>      A3b. Engine run truncated at each cut-off T (>= 9 cut-offs) vs the reference run to 2026-09-25;
                    fills and order intents dated <= T must be identical (exact notionals).  Also run with the
                    universe the loader would have built from data ending at T (as-of trim) = live semantics.
                    Positive control (A4): a subclass reading two sessions ahead must FAIL (see QPILeak2).
  features <fam>    A3a. compute_signals(full history) vs compute_signals(history ending at T) for ALL symbols:
                    every feature value dated <= T must be identical (NaN == NaN), >= 9 cut-offs.
                    Positive control: a one-session look-ahead (IBS from Close_(T+1) / DV2_(T+1)) must FAIL.

Usage: uv run python aud_invariance.py <inv|prefix|features> <dv2|qpi> [window]
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

import aud_common as ac
import aud_data
from strategies.dv2.strategy_mr_dv2 import DVO2Strategy
from strategies.qpi.strategy_mr_qpi_ibs_rsi_exit import QPIIbsRsiExitStrategy

WINDOWS = [("2008-06-02", "2009-06-30"), ("2020-01-02", "2020-12-31"), ("2024-01-02", "2026-09-25")]
SPLIT_NAMES = ("AAPL", "NVDA", "AMZN", "GOOGL", "TSLA", "AVGO", "NFLX", "CMG", "WMT", "LRCX")
FACTORS = (40.0, 0.1, 1.5)
# A3 cut-offs (protocol A3): mid-month, month-end on a weekend, month-end before a holiday, first session of a
# month, last completed month-end, current partial month, plus stress / half-day sessions.
CUTOFFS = {
    "mid_month_2026-09-15": "2026-09-15",
    "month_end_weekend_2026-05-29": "2026-05-29",   # 31 May 2026 is a Sunday
    "month_end_weekend_2025-08-29": "2025-08-29",   # 31 Aug 2025 is a Sunday
    "month_end_before_holiday_2025-12-31": "2025-12-31",  # 1 Jan holiday
    "first_session_2026-09-01": "2026-09-01",
    "last_completed_month_end_2026-08-31": "2026-08-31",
    "current_partial_month_2026-09-24": "2026-09-24",
    "half_day_2024-11-29": "2024-11-29",
    "stress_mid_month_2025-04-08": "2025-04-08",
}


# ---------------------------------------------------------------- positive-control subclasses (A4)
class QPIVolumeRank(ac.RecQPI):
    """Scale-DEPENDENT ranking: adjusted-units Volume (x k under a future split) instead of nominal Turnover."""

    def get_opportunity_list(self, close_row_ser):
        base = super().get_opportunity_list(close_row_ser)
        if not base:
            return base
        vol = close_row_ser.unstack()["Volume"].reindex(base).astype(float)
        return vol.sort_values(ascending=False, kind="mergesort").index.tolist()


class DV2CloseRank(ac.RecDVO2):
    """Scale-DEPENDENT ranking: adjusted Close level instead of NATR (scale-free)."""

    def get_opportunities(self, close):
        base = super().get_opportunities(close)
        if not base:
            return base
        px = close.unstack()["Close"].reindex(base).astype(float)
        return px.sort_values(ascending=False, kind="mergesort").index.tolist()


class QPILeak(ac.RecQPI):
    """Look-ahead: entry IBS computed from Close_(T+lead)."""
    lead = 1

    def compute_signals(self, pricing_data_df):
        out = super().compute_signals(pricing_data_df)
        syms = [s for s in out.columns.get_level_values(0).unique() if (s, "ibs_value_ser") in out.columns]
        for s in syms:
            c_next = out[(s, "Close")].shift(-self.lead)
            rng = (out[(s, "High")] - out[(s, "Low")]).replace(0.0, np.nan)
            out[(s, "ibs_value_ser")] = (c_next - out[(s, "Low")]) / rng
        return out


class DV2Leak(ac.RecDVO2):
    """Look-ahead: DV2 filter reads the value `lead` sessions ahead."""
    lead = 1

    def compute_signals(self, pricing_data):
        out = super().compute_signals(pricing_data)
        syms = [s for s in out.columns.get_level_values(0).unique() if (s, "dv2") in out.columns]
        for s in syms:
            out[(s, "dv2")] = out[(s, "dv2")].shift(-self.lead)
        return out


class QPILeak2(QPILeak):
    """Two-session look-ahead.  The engine prefix harness is structurally blind to a ONE-session look-ahead: a run
    whose data end at T makes its last decision at T-1 (filled at T), and Close_T exists in the truncated data.
    One-session leaks are caught by A3a (feature rows at T) and by the B1 live replay (decision at T)."""
    lead = 2


class DV2Leak2(DV2Leak):
    lead = 2


class QPICentredSMA(QPIIbsRsiExitStrategy):
    def compute_signals(self, pricing_data_df):
        out = super().compute_signals(pricing_data_df)
        syms = [s for s in out.columns.get_level_values(0).unique() if (s, "sma_200_price_ser") in out.columns]
        for s in syms:
            out[(s, "sma_200_price_ser")] = out[(s, "Close")].rolling(200, center=True).mean()
        return out


class DV2CentredSMA(DVO2Strategy):
    def compute_signals(self, pricing_data):
        out = super().compute_signals(pricing_data)
        syms = [s for s in out.columns.get_level_values(0).unique() if (s, "sma_200") in out.columns]
        for s in syms:
            out[(s, "sma_200")] = out[(s, "Close")].rolling(200, center=True).mean()
        return out


def _build(fam, universe, hsu=False, cls=None, capital=100_000.0):
    strategy = ac.make(fam, universe, capital=capital, record=True, cls=cls)
    strategy.historical_share_units_bool = hsu
    return strategy


def _cmp_all(ref_strat, cand_strat, end_ts, rel_tol, order_end_ts=None):
    """Fills are compared through end_ts.  Order intents are compared through order_end_ts (default end_ts); for a
    run truncated at T the last decision it can make is dated T-1 (filled at T), so prefix tests pass T-1."""
    fills = ac.compare_decisions(ac.decisions(ref_strat), ac.decisions(cand_strat), end_ts, rel_tol=rel_tol)
    ref_o, cand_o = ac.order_decisions(ref_strat), ac.order_decisions(cand_strat)
    orders = ac.compare_decisions(ref_o, cand_o, order_end_ts if order_end_ts is not None else end_ts, rel_tol=1e-9,
                                  key=("date", "asset", "kind"), value_col="weight")
    ok = (fills["n_only_reference"] == 0 and fills["n_only_candidate"] == 0 and orders["n_only_reference"] == 0
          and orders["n_only_candidate"] == 0 and orders["n_value_beyond_tol"] == 0)
    return ok, {"fills": fills, "orders": orders}


def run_inv(fam: str, w: int) -> None:
    start, end = WINDOWS[w]
    data = aud_data.load()
    universe = data["universe_trimmed"]
    pricing = ac.subset_pricing(data["pricing_df"].loc[: pd.Timestamp(end)], universe, start, end, extra=SPLIT_NAMES)
    del data
    rows, info = [], {"window": [start, end], "n_symbols": int(pricing.columns.get_level_values(0).nunique())}
    t0 = time.time()
    ref = {hsu: ac.run(_build(fam, universe, hsu), pricing, start, end) for hsu in (False, True)}
    dec = ac.decisions(ref[False])
    buys = dec[dec["side"] > 0]
    top = buys["asset"].value_counts().index[:5].tolist()
    held_end = sorted(str(s) for s, q in ref[False].get_positions().items() if q > 0)
    present = set(pricing.columns.get_level_values(0).astype(str))
    chosen = list(dict.fromkeys(top + held_end[:3] + [s for s in SPLIT_NAMES if s in present]))
    info.update({"rescaled": chosen, "held_at_end": held_end,
                 "rescaled_traded": sorted(set(chosen) & set(dec["asset"].astype(str))),
                 "n_fills_ref": int(len(dec))})
    for hsu in (False, True):
        for k in FACTORS:
            cand_pricing = pricing
            for s in chosen:
                cand_pricing = ac.rescale_symbol_history(cand_pricing, s, k)
            cand = ac.run(_build(fam, universe, hsu), cand_pricing, start, end)
            ok, detail = _cmp_all(ref[hsu], cand, end, rel_tol=0.35 if not hsu else 1e-7)
            max_notional = detail["fills"]["max_rel_value_diff"]
            rows.append({"test": f"A2_invariance_{'hsu' if hsu else 'engine'}", "case": f"w{w}_k{k:g}", "passed": ok,
                         "max_rel_notional_diff": max_notional, "detail": detail})
            print(rows[-1]["test"], rows[-1]["case"], ok, max_notional, flush=True)
    # A4 positive control for the invariance harness
    pc_cls = QPIVolumeRank if fam == "qpi" else DV2CloseRank
    pc_ref = ac.run(_build(fam, universe, cls=pc_cls), pricing, start, end)
    cand_pricing = pricing
    for s in chosen:
        cand_pricing = ac.rescale_symbol_history(cand_pricing, s, 40.0)
    pc_cand = ac.run(_build(fam, universe, cls=pc_cls), cand_pricing, start, end)
    ok, detail = _cmp_all(pc_ref, pc_cand, end, rel_tol=0.35)
    rows.append({"test": "A4_positive_control_invariance", "case": f"w{w}_{pc_cls.__name__}_k40",
                 "passed_means_caught": (not ok), "detail": detail})
    print("A4 invariance control caught:", not ok, flush=True)
    info["runtime_s"] = round(time.time() - t0, 1)
    path = ac.OUT / "invariance" / f"{fam}_inv_w{w}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"info": info, "rows": rows}, indent=2, default=str), encoding="utf-8")


def run_prefix(fam: str) -> None:
    start, end = WINDOWS[2]
    data = aud_data.load()
    universe, untrimmed = data["universe_trimmed"], data["universe_untrimmed"]
    pricing = ac.subset_pricing(data["pricing_df"].loc[: pd.Timestamp(end)], universe, start, end)
    del data
    rows = []
    ref = ac.run(_build(fam, universe), pricing, start, end)
    leak_cls = QPILeak2 if fam == "qpi" else DV2Leak2
    leak_ref = ac.run(_build(fam, universe, cls=leak_cls), pricing, start, end)
    for label, cut in CUTOFFS.items():
        cut_ts = pd.Timestamp(cut)
        prev_ts = pricing.index[pricing.index.get_loc(cut_ts) - 1]
        cand = ac.run(_build(fam, universe), pricing.loc[:cut_ts], start, cut_ts)
        ok, detail = _cmp_all(ref, cand, cut_ts, rel_tol=1e-9, order_end_ts=prev_ts)
        ok = ok and detail["fills"]["n_value_beyond_tol"] == 0
        rows.append({"test": "A3b_prefix_engine", "case": label, "passed": ok, "detail": detail})
        print(rows[-1]["test"], label, ok, flush=True)
        asof_u = aud_data.asof_trimmed_universe(untrimmed, cut_ts)
        cand = ac.run(_build(fam, asof_u), pricing.loc[:cut_ts], start, cut_ts)
        ok, detail = _cmp_all(ref, cand, cut_ts, rel_tol=1e-9, order_end_ts=prev_ts)
        ok = ok and detail["fills"]["n_value_beyond_tol"] == 0
        rows.append({"test": "A3b_prefix_asof_universe", "case": label, "passed": ok, "detail": detail})
        print(rows[-1]["test"], label, ok, flush=True)
        leak_cand = ac.run(_build(fam, universe, cls=leak_cls), pricing.loc[:cut_ts], start, cut_ts)
        ok, detail = _cmp_all(leak_ref, leak_cand, cut_ts, rel_tol=1e-9, order_end_ts=prev_ts)
        rows.append({"test": "A4_positive_control_prefix", "case": f"{label}_{leak_cls.__name__}",
                     "passed_means_caught": (not ok), "detail": detail})
        print("A4 prefix control caught:", not ok, flush=True)
    path = ac.OUT / "invariance" / f"{fam}_prefix.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")


def _feature_cols(fam: str) -> list[str]:
    return (["p126d_return", "natr", "dv2", "sma_200"] if fam == "dv2"
            else ["three_day_return_ser", "qpi_value_ser", "sma_200_price_ser", "ibs_value_ser", "rsi2_value_ser"])


def run_features(fam: str) -> None:
    data = aud_data.load()
    pricing = data["pricing_df"]
    universe = data["universe_trimmed"]
    del data
    cls_ok = DVO2Strategy if fam == "dv2" else QPIIbsRsiExitStrategy
    cls_leak = DV2Leak if fam == "dv2" else QPILeak

    def signals(cls, frame):
        strategy = ac.make(fam, universe, cls=cls)
        return strategy.compute_signals(frame.copy())

    feats = _feature_cols(fam)
    t0 = time.time()
    full = signals(cls_ok, pricing)
    full_leak = signals(cls_leak, pricing)
    rows = []
    for label, cut in CUTOFFS.items():
        cut_ts = pd.Timestamp(cut)
        for cls, full_df, test in ((cls_ok, full, "A3a_feature_prefix"), (cls_leak, full_leak, "A4_positive_control_feature")):
            trunc = signals(cls, pricing.loc[:cut_ts])
            cols = [c for c in trunc.columns if c[1] in feats]
            a = full_df.loc[:cut_ts, cols].to_numpy(dtype=float)
            b = trunc.loc[:, cols].to_numpy(dtype=float)
            same_nan = np.isnan(a) == np.isnan(b)
            diff = np.where(np.isnan(a) | np.isnan(b), 0.0, np.abs(a - b))
            n_bad = int((~same_nan).sum() + (diff > 0).sum())
            # candidate list on the cut date itself (decision_t = cut)
            row_full, row_trunc = full_df.loc[cut_ts], trunc.loc[cut_ts]
            n_bad_row = int(((row_full[cols].astype(float) - row_trunc[cols].astype(float)).abs() > 0).sum()
                            + (row_full[cols].isna() != row_trunc[cols].isna()).sum())
            rec = {"test": test, "case": label, "n_values_compared": int(a.size), "n_values_different": n_bad,
                   "n_cut_row_values_different": n_bad_row, "max_abs_diff": float(diff.max()) if diff.size else 0.0}
            if test.startswith("A3a"):
                rec["passed"] = n_bad == 0
            else:
                rec["passed_means_caught"] = n_bad_row > 0
            rows.append(rec)
            print(rec, flush=True)
    path = ac.OUT / "invariance" / f"{fam}_features.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"runtime_s": round(time.time() - t0, 1), "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    mode, fam = sys.argv[1], sys.argv[2]
    if mode == "inv":
        run_inv(fam, int(sys.argv[3]))
    elif mode == "prefix":
        run_prefix(fam)
    else:
        run_features(fam)
