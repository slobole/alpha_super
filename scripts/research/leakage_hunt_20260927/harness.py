"""Shared leakage-hunt harness (study 2026-09-27; research-only, no strategy/engine edits).

Two tests are applied to every strategy in the books:

1. Future-action invariance.  A k:1 split (or a TOTALRETURN dividend factor) that happens AFTER the last
   loaded date D rescales that symbol's whole stored history, while nominal fields stay put:

       Open, High, Low, Close        -> x / k        (CAPITALSPECIAL and TOTALRETURN price fields)
       Dividend                      -> x / k        (Norgate CAPITALSPECIAL dividends are per adjusted share)
       Volume                        -> x * k
       Unadjusted Close, Turnover    -> unchanged    (nominal, vintage invariant)

   Decisions made on or before D must not change.  "Decision" means the selected symbol set and target weight
   (or dollar intent) at each decision date - not share counts, which legitimately scale by k in adjusted units.

2. Truncation (prefix).  Decisions computed on data that ends at T must equal the full-history decisions at T.
   Norgate returns one current vintage, so this only tests code causality (rolling windows, resample, month-end
   detection, full-sample normalisation); test 1 is what detects revised history.

Helpers here are deliberately small and pure so each strategy adapter can reuse them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "research" / "leakage_hunt_20260927"

PRICE_FIELD_TUPLE = ("Open", "High", "Low", "Close")
DIVIDED_FIELD_TUPLE = PRICE_FIELD_TUPLE + ("Dividend",)
MULTIPLIED_FIELD_TUPLE = ("Volume",)
NOMINAL_FIELD_TUPLE = ("Unadjusted Close", "Turnover")
# Factors cover a large forward split, a reverse split and an awkward non-integer ratio.
DEFAULT_FACTOR_TUPLE = (40.0, 0.1, 1.5)


def rescale_symbol_history(pricing_df: pd.DataFrame, symbol_str: str, factor_float: float) -> pd.DataFrame:
    """Return a copy of a (symbol, field) MultiIndex frame with one symbol's full history re-based as if a
    factor_float:1 split happened after the last row.  Nominal fields are left untouched."""
    out_df = pricing_df.copy()
    for field_str in DIVIDED_FIELD_TUPLE:
        key = (symbol_str, field_str)
        if key in out_df.columns:
            out_df[key] = out_df[key] / factor_float
    for field_str in MULTIPLIED_FIELD_TUPLE:
        key = (symbol_str, field_str)
        if key in out_df.columns:
            out_df[key] = out_df[key] * factor_float
    return out_df


def rescale_columns(frame_df: pd.DataFrame, factor_ser: pd.Series) -> pd.DataFrame:
    """Plain wide frame (dates x symbols) of prices: divide each listed column by its factor."""
    out_df = frame_df.copy()
    for symbol_str, factor_float in factor_ser.items():
        if symbol_str in out_df.columns:
            out_df[symbol_str] = out_df[symbol_str] / float(factor_float)
    return out_df


def truncate(frame, end_ts):
    return frame.loc[: pd.Timestamp(end_ts)].copy()


def compare_weight_frames(reference_df: pd.DataFrame, candidate_df: pd.DataFrame, atol_float: float = 1e-9) -> dict:
    """Compare two decision tables (dates x assets, weights) on their common dates."""
    common_idx = reference_df.index.intersection(candidate_df.index)
    columns = reference_df.columns.union(candidate_df.columns)
    ref = reference_df.reindex(index=common_idx, columns=columns).fillna(0.0).astype(float)
    cand = candidate_df.reindex(index=common_idx, columns=columns).fillna(0.0).astype(float)
    diff = (ref - cand).abs()
    bad_row = diff.max(axis=1) > atol_float
    return {
        "n_dates_compared": int(len(common_idx)),
        "n_dates_different": int(bad_row.sum()),
        "max_abs_weight_diff": float(diff.to_numpy().max()) if diff.size else 0.0,
        "first_different_dates": [d.date().isoformat() for d in common_idx[bad_row][:5]],
        "reference_only_dates": int(len(reference_df.index.difference(candidate_df.index))),
        "candidate_only_dates": int(len(candidate_df.index.difference(reference_df.index))),
    }


def compare_selection_sets(reference_dict: dict, candidate_dict: dict) -> dict:
    """reference/candidate: {date -> iterable of selected symbols} (ordered or not)."""
    common = sorted(set(reference_dict) & set(candidate_dict))
    bad = [d for d in common if set(reference_dict[d]) != set(candidate_dict[d])]
    return {
        "n_dates_compared": len(common),
        "n_dates_different": len(bad),
        "first_different_dates": [pd.Timestamp(d).date().isoformat() for d in bad[:5]],
        "examples": [
            {"date": pd.Timestamp(d).date().isoformat(),
             "reference_only": sorted(set(reference_dict[d]) - set(candidate_dict[d]))[:10],
             "candidate_only": sorted(set(candidate_dict[d]) - set(reference_dict[d]))[:10]}
            for d in bad[:3]
        ],
    }


def transaction_decisions(transactions_df: pd.DataFrame, date_col: str = "bar", asset_col: str = "asset",
                          amount_col: str = "amount", price_col: str = "price") -> pd.DataFrame:
    """Engine transactions -> one row per (date, asset) with side and dollar notional; share units dropped."""
    tx = transactions_df.copy()
    tx["notional"] = tx[amount_col].astype(float) * tx[price_col].astype(float)
    tx["side"] = np.sign(tx[amount_col].astype(float)).astype(int)
    grouped = tx.groupby([date_col, asset_col]).agg(side=("side", "first"), notional=("notional", "sum"))
    return grouped.reset_index().rename(columns={date_col: "date", asset_col: "asset"})


def compare_transaction_decisions(reference_tx: pd.DataFrame, candidate_tx: pd.DataFrame, end_ts=None,
                                  rel_tol_float: float = 0.02) -> dict:
    """Same (date, asset, side) set?  Notionals within rel_tol (whole-share rounding in scaled units legitimately
    moves notionals by < 1 share price, and fees differ; decisions must not)."""
    ref, cand = reference_tx.copy(), candidate_tx.copy()
    if end_ts is not None:
        ref = ref[ref["date"] <= pd.Timestamp(end_ts)]
        cand = cand[cand["date"] <= pd.Timestamp(end_ts)]
    key = ["date", "asset", "side"]
    merged = ref.merge(cand, on=key, how="outer", suffixes=("_ref", "_cand"), indicator=True)
    only_ref = merged[merged["_merge"] == "left_only"]
    only_cand = merged[merged["_merge"] == "right_only"]
    both = merged[merged["_merge"] == "both"].copy()
    both["rel_diff"] = (both["notional_cand"] - both["notional_ref"]).abs() / both["notional_ref"].abs().clip(lower=1.0)
    first_divergence = None
    if len(only_ref) or len(only_cand):
        first_divergence = pd.concat([only_ref["date"], only_cand["date"]]).min()
    return {
        "n_decisions_reference": int(len(ref)),
        "n_decisions_candidate": int(len(cand)),
        "n_only_reference": int(len(only_ref)),
        "n_only_candidate": int(len(only_cand)),
        "first_divergence_date": None if first_divergence is None else pd.Timestamp(first_divergence).date().isoformat(),
        "max_rel_notional_diff": float(both["rel_diff"].max()) if len(both) else 0.0,
        "n_notional_beyond_tol": int((both["rel_diff"] > rel_tol_float).sum()),
        "examples_only_reference": only_ref[key].head(5).astype(str).to_dict("records"),
        "examples_only_candidate": only_cand[key].head(5).astype(str).to_dict("records"),
    }


@dataclass
class ResultLog:
    strategy_str: str
    rows: list = field(default_factory=list)

    def add(self, test_str: str, case_str: str, passed_bool: bool, detail: dict | None = None) -> None:
        self.rows.append({"strategy": self.strategy_str, "test": test_str, "case": case_str,
                          "passed": bool(passed_bool), "detail": detail or {}})
        print(f"[{'PASS' if passed_bool else 'FAIL'}] {self.strategy_str} {test_str} {case_str} {detail or ''}")

    def save(self, name_str: str | None = None) -> Path:
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"{name_str or self.strategy_str}__invariance.json"
        path.write_text(json.dumps(self.rows, indent=2, default=str), encoding="utf-8")
        return path
