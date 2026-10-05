"""Defensive shelf v2 (SPEC_FROZEN.md): robust MAIN and LOW-TOUCH lines under fair cash.

Every book = defensive part (1-4 pods, EQ or IV among themselves) scaled to 1 - S plus S of G3 (TAA3x 50 / NDX-VXN
50), S in {0, 10%, 20%}, pod-level annual reset. The defensive part and G3 reset on the same year-end closes, so
mixing the two component return series with an annual reset equals the pod-level book.

Usage: python defensive_v2.py
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "scripts" / "research" / "shelf_rebuild_20260929"))

import lib  # noqa: E402
from lib import BLOCK_DICT, END, EXACT_START, LONG_START, TBILL, Book  # noqa: E402

OUT = REPO / "results" / "research" / "portfolio" / "defensive_v2_20260929"
POOL = {"LOW": ["core5", "btal_qqq", "tactical_fi"],
        "MAIN": ["core5", "btal_qqq", "tactical_fi", "trinity", "eom_flow", "downshock", "disp", "etf_dv2"]}
MAX_PODS = {"LOW": 3, "MAIN": 4}
SLICES = (0.0, 0.10, 0.20)
NOT_LIVE = {"etf_dv2", "eom_flow", "trinity"}
G3 = Book("G3", ("taa3x", "ndx_vxn"), "EQ", {"taa3x": 0.5, "ndx_vxn": 0.5})
LABEL = {"core5": "CORE5", "btal_qqq": "BTAL_QQQ", "tactical_fi": "TFI", "trinity": "TRINITY", "eom_flow": "EOM",
         "downshock": "DOWNSHOCK", "disp": "DISP", "etf_dv2": "DV2-IND"}
DD_HIST, DD_LIMIT, BREACH_MAX = -0.08, -0.10, 0.10
CHAMPION = "CORE5 60 + BTAL_QQQ 40"


@dataclass
class DefBook:
    name: str
    pods: tuple[str, ...]
    rule: str            # "EQ", "IV" or "FIXED"
    weights: dict        # FIXED weights of the defensive part (empty otherwise)
    slice_: float        # share of G3

    def defensive(self) -> Book:
        if self.rule == "FIXED":
            return Book(self.name, self.pods, "EQ", dict(self.weights))
        return Book(self.name, self.pods, self.rule)

    @property
    def pod_count(self) -> int:
        return len(self.pods) + (2 if self.slice_ > 0 else 0)


def family(line: str) -> list[DefBook]:
    books = [DefBook(CHAMPION, ("core5", "btal_qqq"), "FIXED", {"core5": 0.6, "btal_qqq": 0.4}, 0.0)]
    for k in range(1, MAX_PODS[line] + 1):
        for pods in combinations(POOL[line], k):
            for rule in (("EQ",) if k == 1 else ("EQ", "IV")):
                for s in SLICES:
                    name = " + ".join(LABEL[p] for p in pods) + f" [{rule}]" + (f" + G3 {int(s * 100)}%" if s else "")
                    books.append(DefBook(name, pods, rule, {}, s))
    return books


def book_series(frame: pd.DataFrame, b: DefBook, replace: str | None = None, weight_source: pd.DataFrame | None = None
                ) -> pd.Series:
    """LONG daily returns; `replace` swaps one pod's (or "G3") returns for T-bills, weights unchanged (slot test)."""
    src = frame if weight_source is None else weight_source
    f = lib.replace_pod(frame, replace) if replace and replace != "G3" else frame
    d = lib.book_returns(f, b.defensive(), LONG_START, weight_source=src)
    if b.slice_ == 0:
        return d
    g = frame[TBILL].loc[LONG_START:END] if replace == "G3" else lib.book_returns(frame, G3, LONG_START)
    comp = pd.DataFrame({"D": d, "G": g})
    return lib.book_returns(comp, Book("mix", ("D", "G"), "EQ", {"D": 1 - b.slice_, "G": b.slice_}), LONG_START)


def excess_calmar(r: pd.Series, tb: pd.Series, idx) -> float:
    return lib.excess_calmar(r, tb, idx)


def gate_row(frame: pd.DataFrame, b: DefBook, idx) -> dict:
    tb = frame[TBILL]
    r = book_series(frame, b)
    obj = lib.excess_cagr(r, tb, idx)
    ecal = excess_calmar(r, tb, idx)
    blocks = {k: lib.excess_cagr(lib.window(r, lo, hi), tb, idx) for k, (lo, hi) in BLOCK_DICT.items()}
    slots = list(b.pods) + (["G3"] if b.slice_ > 0 else [])
    slot = {p: excess_calmar(book_series(frame, b, replace=p, weight_source=frame), tb, idx) for p in slots} \
        if len(slots) > 1 or b.slice_ > 0 else {}
    if len(slots) == 1:  # a single pod must beat T-bills outright
        slot = {slots[0]: 0.0}
    return {"objective": obj, "excess_calmar": ecal, "maxdd": lib.maxdd(r), **{f"xs_{k}": v for k, v in blocks.items()},
            "slot_pass": all(v < ecal for v in slot.values()), "slot_fail": ",".join(p for p, v in slot.items() if v >= ecal),
            "_r": r}


def fair(frame: pd.DataFrame, data: dict) -> pd.DataFrame:
    """Add the cash-realism increment (cash_long - long) to a house-cash frame."""
    add = (data["cash_long"] - data["long"]).fillna(0.0)
    out = frame.copy()
    cols = [c for c in add.columns if c in out.columns and c != TBILL]
    out[cols] = out[cols] + add[cols]
    return out


def boot_stats(R: np.ndarray, tb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per path: excess CAGR (by length) and max drawdown for every column."""
    idx = lib.boot_index(R.shape[0])
    obj = np.empty((idx.shape[0], R.shape[1]))
    dd = np.empty_like(obj)
    for k in range(idx.shape[0]):
        c, d = lib.path_stats(R[idx[k]])
        tbc = float(np.prod(1.0 + tb[idx[k]]) ** (252.0 / len(tb)) - 1.0)
        obj[k], dd[k] = c - tbc, d
    return obj, dd


def cscv(R: np.ndarray, tb: np.ndarray, picks: list[int], blocks: int = 16) -> dict:
    n = R.shape[0] - R.shape[0] % blocks
    R, tb = R[:n], tb[:n]
    parts = np.array_split(np.arange(n), blocks)
    logits, omega = [], {j: [] for j in picks}
    for combo in combinations(range(blocks), blocks // 2):
        ins = np.concatenate([parts[i] for i in combo])
        oos = np.concatenate([parts[i] for i in range(blocks) if i not in combo])
        vals = []
        for sel in (ins, oos):
            c, _ = lib.path_stats(R[sel])
            vals.append(c - (np.prod(1 + tb[sel]) ** (252.0 / len(sel)) - 1))
        best = int(np.argmax(vals[0]))
        for j in [best] + picks:
            below = np.sum(vals[1] < vals[1][j]) + 0.5 * (np.sum(vals[1] == vals[1][j]) - 1)
            w = (below + 1.0) / (R.shape[1] + 1)
            if j == best:
                logits.append(np.log(w / (1 - w)))
            if j in omega:
                omega[j].append(w)
    return {"pbo_argmax": float(np.mean(np.array(logits) <= 0)),
            "picks": {j: {"median_oos_omega": float(np.median(v)), "share_below_median": float(np.mean(np.array(v) <= 0.5))}
                      for j, v in omega.items()}}


def run_line(line: str, data: dict, frames: dict) -> dict:
    idx = data["index"]
    books = family(line)
    main = frames["main"]
    rows, series = [], {}
    for b in books:
        g = gate_row(main, b, idx)
        series[b.name] = g.pop("_r")
        rows.append({"book": b.name, "line": line, "pods": b.pod_count, "rule": b.rule, "slice": b.slice_,
                     "pods_list": "+".join(b.pods), **g})
    table = pd.DataFrame(rows).set_index("book")
    # R5: R1, R3, R4 under every sensitivity frame.
    for label, frame in frames.items():
        if label == "main":
            continue
        ok = []
        for b in books:
            g = gate_row(frame, b, idx)
            g.pop("_r")
            ok.append(g["maxdd"] >= DD_HIST and all(g[f"xs_{k}"] > 0 for k in BLOCK_DICT) and g["slot_pass"])
            table.loc[b.name, f"obj_{label}"] = g["objective"]
        table[f"r5_{label}"] = ok
    R = pd.DataFrame(series)
    tb = main[TBILL].reindex(R.index).to_numpy()
    obj_b, dd_b = boot_stats(R.to_numpy(), tb)
    names = list(R.columns)
    table["p_breach10"] = pd.Series((dd_b < DD_LIMIT).mean(axis=0), index=names)
    table["boot_dd_p50"] = pd.Series(np.median(dd_b, axis=0), index=names)
    table["r1"] = table["maxdd"] >= DD_HIST
    table["r2"] = table["p_breach10"] <= BREACH_MAX
    table["r3"] = (table[[f"xs_{k}" for k in BLOCK_DICT]] > 0).all(axis=1)
    table["r4"] = table["slot_pass"]
    table["r5"] = table[[c for c in table.columns if c.startswith("r5_")]].all(axis=1)
    table["gates_pass"] = table[["r1", "r2", "r3", "r4", "r5"]].all(axis=1)
    # ease fields
    years = (END - EXACT_START).days / 365.25
    not_live, tdays = [], []
    avg_w = {}
    for b in books:
        if b.rule == "IV":
            log: list = []
            lib.book_returns(main, b.defensive(), LONG_START, weight_log=log)
            dw = lib.average_weights(log)
        else:
            dw = b.defensive().targets()
        avg_w[b.name] = {p: v * (1 - b.slice_) for p, v in dw.items()}
        not_live.append(sum(w for p, w in dw.items() if p in NOT_LIVE) * (1 - b.slice_))
        dates = set()
        for p in list(b.pods) + (["taa3x", "ndx_vxn"] if b.slice_ else []):
            dates |= lib.trade_dates(data, p)
        tdays.append(len(dates) / years)
    table["not_live_share"] = not_live
    table["trade_days_per_year"] = tdays
    table["avg_weights"] = [json.dumps({k: round(v, 4) for k, v in avg_w[b.name].items()}) for b in books]
    passers = table[table["gates_pass"]]
    result = {"line": line, "family_size": len(books), "gate_passers": int(len(passers))}
    if len(passers):
        top = passers["objective"].idxmax()
        jt = names.index(top)
        share = pd.Series({b: float(np.mean(obj_b[:, jt] > obj_b[:, names.index(b)])) for b in passers.index})
        share[top] = 0.0
        band = passers.loc[share.index[share < 0.90]].copy()
        band["beaten_by_top"] = share
        band = band.sort_values(["pods", "not_live_share", "p_breach10", "trade_days_per_year", "objective"],
                                ascending=[True, True, True, True, False])
        pick = band.index[0]
        jc, jp = names.index(CHAMPION), names.index(pick)
        beats = float(np.mean(obj_b[:, jp] > obj_b[:, jc]))
        champion_holds = not (beats >= 0.80 and table.at[pick, "p_breach10"] <= table.at[CHAMPION, "p_breach10"] + 0.02)
        result.update({"top": top, "tie_band": list(band.index), "pick": pick, "pick_beats_champion_share": beats,
                       "recommendation": CHAMPION if champion_holds else pick, "champion_holds": champion_holds})
        # sensitivity: does the rule's pick change?
        sens_pick = {}
        for label in frames:
            if label == "main":
                continue
            col = f"obj_{label}"
            p2 = passers.copy()
            p2["objective"] = table.loc[passers.index, col]
            t2 = p2["objective"].idxmax()
            j2 = names.index(t2)
            sh2 = pd.Series({b: float(np.mean(obj_b[:, j2] > obj_b[:, names.index(b)])) for b in p2.index})
            sh2[t2] = 0.0
            b2 = p2.loc[sh2.index[sh2 < 0.90]].sort_values(["pods", "not_live_share", "p_breach10",
                                                             "trade_days_per_year", "objective"],
                                                            ascending=[True, True, True, True, False])
            sens_pick[label] = b2.index[0]
        result["pick_under_sensitivities"] = sens_pick
        result["cscv"] = cscv(R.to_numpy(), tb, [jp, jc])
        table["in_tie_band"] = table.index.isin(band.index)
        table["beaten_by_top"] = share.reindex(table.index)
    table["beats_champion_share"] = pd.Series([float(np.mean(obj_b[:, names.index(b)] > obj_b[:, names.index(CHAMPION)]))
                                               for b in names], index=names)
    table.to_csv(OUT / f"{line.lower()}_books.csv", float_format="%.6g")
    R.to_csv(OUT / f"{line.lower()}_long_returns.csv.gz", float_format="%.10g", compression="gzip")
    return result


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    spec_hash = hashlib.sha256((HERE / "SPEC_FROZEN.md").read_bytes()).hexdigest()
    with (OUT / "experiment_ledger.jsonl").open("a", encoding="utf-8") as h:
        h.write(json.dumps({"event_str": "defensive_v2_started", "spec_sha256_str": spec_hash}) + "\n")
    data = lib.load_inputs()
    swap_tfi = data["long"].copy()
    swap_tfi["tactical_fi"] = data["tfi_frozen"]
    frames = {"main": data["cash_long"], "proxy_unscaled": fair(data["long_unscaled"], data),
              "plus_5bps": fair(data["stressed_long"], data), "tfi_frozen": fair(swap_tfi, data),
              "etf_idle_2008_09": fair(data["long_etf_cash"], data)}
    out = {line: run_line(line, data, frames) for line in ("LOW", "MAIN")}
    (OUT / "selection.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    with (OUT / "experiment_ledger.jsonl").open("a", encoding="utf-8") as h:
        h.write(json.dumps({"event_str": "defensive_v2_finished", "spec_sha256_str": spec_hash,
                            "picks": {k: v.get("pick") for k, v in out.items()}}) + "\n")
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
