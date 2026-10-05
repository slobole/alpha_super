"""Evaluate a pre-declared family of books: metrics, gates, T-bill slot tests, bootstrap tie band, PBO.

Parts D and G (SPEC 6 and 7) differ only in the family, the objective, the gates and the tie-break order; the
mechanics live here so both parts are computed the same way.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import lib
from lib import BLOCK_DICT, EXACT_START, LONG_START, TBILL


def objective(kind: str, r: pd.Series, tb: pd.Series, index_all: pd.DatetimeIndex, budget: float) -> float:
    if kind == "excess_calmar":
        return lib.excess_calmar(r, tb, index_all)
    if kind == "cagr_at_budget":
        return lib.cagr_at_budget(r, tb, index_all, budget)[0]
    raise ValueError(kind)


def window_returns(r: pd.Series, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.Series:
    return lib.window(r, lo, hi)


def evaluate(books: list, data: dict, kind: str, budget: float, frame_key: str = "long",
             with_metrics: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per book and the LONG daily returns (sessions x books) used by the bootstrap and PBO."""
    frame = data[frame_key]
    index_all = data["index"]
    tb = frame[TBILL]
    spx = data["bench"]["SPXTR"]
    episodes = common_episodes(data) if with_metrics else []
    cofalls = lib.cofall_windows(data["bench"], LONG_START, lib.END) if with_metrics else []
    rows, series = [], {}
    recent_lo, recent_hi = BLOCK_DICT["RECENT"]
    for book in books:
        weight_log: list = []
        r_long = lib.book_returns(frame, book, LONG_START, weight_log=weight_log)
        avg_w = lib.average_weights(weight_log) if book.rule == "IV" else book.targets()
        obj = objective(kind, r_long, tb, index_all, budget)
        row = {"book": book.name, "family": book.family, "rule": book.rule, "policy": book.policy,
               "pods_list": "+".join(book.pods), "objective": obj,
               "avg_weights": json.dumps({k: round(v, 4) for k, v in avg_w.items()}), **book.tags}
        if kind == "cagr_at_budget":
            row["s_at_budget"] = lib.cagr_at_budget(r_long, tb, index_all, budget)[1]
        for block, (lo, hi) in BLOCK_DICT.items():
            row[f"xs_{block}"] = lib.excess_cagr(window_returns(r_long, lo, hi), tb, index_all)
        r_recent = window_returns(r_long, recent_lo, recent_hi)
        obj_recent = objective(kind, r_recent, tb, index_all, budget)
        slot_long, slot_recent = {}, {}
        for pod in book.pods:
            replaced = lib.book_returns(lib.replace_pod(frame, pod), book, LONG_START, weight_source=frame)
            slot_long[pod] = objective(kind, replaced, tb, index_all, budget)
            slot_recent[pod] = objective(kind, window_returns(replaced, recent_lo, recent_hi), tb, index_all, budget)
        row["slot_long_pass"] = all(v < obj for v in slot_long.values())
        row["slot_recent_pass"] = all(v < obj_recent for v in slot_recent.values())
        row["slot_long_fail_pods"] = ",".join(p for p, v in slot_long.items() if v >= obj)
        row["slot_recent_fail_pods"] = ",".join(p for p, v in slot_recent.items() if v >= obj_recent)
        row["slot_long_detail"] = json.dumps({p: round(v - obj, 5) for p, v in slot_long.items()})
        if with_metrics:
            r_exact = lib.book_returns(data["sleeve"], book, EXACT_START)
            row.update(lib.full_metrics(r_long, data, "long"))
            row.update(lib.full_metrics(r_exact, data, "exact"))
            row["exact_objective"] = objective(kind, r_exact, data["sleeve"][TBILL], index_all, budget)
            for name, (lo, hi) in lib.CRISIS_DICT.items():
                row[f"crisis_{name}"] = lib.common.window_return_float(r_long, lo, hi)
            for k, ep in enumerate(episodes):
                row[f"spx10_{k}_{ep['peak_date_str']}"] = lib.common.window_return_float(r_long, ep["peak_date_str"],
                                                                                         ep["trough_date_str"])
            for k, (lo, hi, _) in enumerate(cofalls):
                row[f"cofall_{k}_{lo.date()}"] = lib.common.window_return_float(r_long, lo.strftime("%Y-%m-%d"),
                                                                                hi.strftime("%Y-%m-%d"))
            row.update(lib.ops_fields(book, data, avg_w))
        rows.append(row)
        series[book.name] = r_long
    return pd.DataFrame(rows).set_index("book"), pd.DataFrame(series)


def common_episodes(data: dict) -> list[dict]:
    spx = data["bench"]["SPXTR"].loc[LONG_START:lib.END]
    return lib.common.equity_drawdown_episode_list(spx, -0.10)


def tie_band(table: pd.DataFrame, boot: np.ndarray, names: list[str], gate_col: str) -> tuple[str, pd.Series]:
    """Top gate-passing book and, for every passer, the share of paths on which the top book beats it."""
    passers = table.index[table[gate_col]]
    top = table.loc[passers, "objective"].idxmax()
    j_top = names.index(top)
    share = pd.Series({b: float(np.mean(boot[:, j_top] > boot[:, names.index(b)])) for b in passers})
    share[top] = 0.0
    return top, share


def select(table: pd.DataFrame, share: pd.Series, sort_spec: list[tuple[str, bool]]) -> pd.DataFrame:
    """The tie band (share < TIE_SHARE) ordered by the part's tie-break (column, ascending)."""
    band = table.loc[share.index[share < lib.TIE_SHARE]].copy()
    band["beaten_by_top_share"] = share.reindex(band.index)
    columns = [c for c, _ in sort_spec]
    ascending = [a for _, a in sort_spec]
    return band.sort_values(columns, ascending=ascending)
