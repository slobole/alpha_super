"""Client barbell study (SPEC_FROZEN.md in this folder): a risky sleeve and a defensive core in one client account.

Account model (SPEC 3). Within a between-sleeve period that starts at account value V0,
    V_t = V0 * (s * GA_t + (1 - s) * GD_t),   G = growth of the sleeve since the period start.
Policy "annual": the sleeves are reset to s / 1 - s after the last close of each calendar year, paying 5 bps per side
on the dollars moved (2 x c x |V_R - s V|); the cost lands in the first return of the next year.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
GA_DIR = HERE.parent / "growth_aggressive_20260930"
if str(GA_DIR) not in sys.path:
    sys.path.insert(0, str(GA_DIR))

import ga_lib as ga  # noqa: E402
from ga_lib import END, EXACT_START, LONG_START, TBILL, Book, lib  # noqa: E402,F401

STUDY = ga.WT_REPO / "results" / "research" / "portfolio" / "barbell_20260930"
S_CLIENT = 1.0 / 3.0
TRANSFER_BPS = 0.0005
SEEDS = [20260929 + k for k in range(10)]
B_GRID = (-0.10, -0.125, -0.15, -0.175, -0.20)
TAA_CAP = 0.25
MAX_BREACH_ABS = 0.10
TIE_SHARE, CHAMP_SHARE = 0.90, 0.80
TAA_PODS = ("taa3x", "taa3x_1n", "taa2x_1n")
TAA_DF_PODS = TAA_PODS + ("btal_qqq",)   # all four live in strategies.taa_df (verified in run_sleeves.SLEEVE_DICT)

TAA_LABEL = {"taa3x": "TAA3x", "taa3x_1n": "TAA3x-1N", "taa2x_1n": "TAA2x-1N"}
NDX_LABEL = {"ndx_vxn": "NDX-VXN", "ndx_atr": "NDX-ATR"}
SAT_LT = {"none": (), "def2": ("core5", "btal_qqq"), "tbill": (TBILL,)}
SAT_MAIN = {"dv2": ("dv2",), "pair_dv2": ("dv2", "hpi_vote"), "hpi": ("hpi_vote",)}
SHARES = (0.18, 0.36)
WIRED_NOW = {"taa3x", "taa3x_1n", "ndx_vxn", "ndx_atr", "btal_qqq", "dv2", "hpi_vote", TBILL}


def ledger(event_str: str, **fields) -> None:
    STUDY.mkdir(parents=True, exist_ok=True)
    record = {"event_str": event_str, "recorded_at_utc_str": datetime.now(timezone.utc).isoformat(),
              "spec_sha256_str": hashlib.sha256((HERE / "SPEC_FROZEN.md").read_bytes()).hexdigest(), **fields}
    with (STUDY / "experiment_ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


# ─── families (SPEC 4) ───────────────────────────────────────────────────────


@dataclass
class Sleeve:
    name: str
    weights: dict
    line: str            # "LT" or "MAIN"
    tags: dict = field(default_factory=dict)

    def book(self) -> Book:
        return Book(self.name, tuple(self.weights), "EQ", dict(self.weights), "annual", "R")


def risky_sleeves() -> list[Sleeve]:
    cores = []
    for taa, ndx, ratio in product(TAA_LABEL, NDX_LABEL, (0.5, 0.6, 0.7)):
        cores.append((f"{TAA_LABEL[taa]} + {NDX_LABEL[ndx]} {int(round(ratio * 100))}:{int(round((1 - ratio) * 100))}",
                      {taa: ratio, ndx: 1.0 - ratio}))
    for taa in TAA_LABEL:
        cores.append((f"{TAA_LABEL[taa]} only", {taa: 1.0}))
    for ndx in NDX_LABEL:
        cores.append((f"{NDX_LABEL[ndx]} only", {ndx: 1.0}))
    out = []
    options = [("none", 0.0)] + [(k, sh) for k in ("def2", "tbill") for sh in SHARES]
    options_main = [(k, sh) for k in SAT_MAIN for sh in SHARES]
    for core_name, core_w in cores:
        for sat, share in options + options_main:
            pods = SAT_LT.get(sat) or SAT_MAIN.get(sat) or ()
            w = {p: v * (1.0 - share) for p, v in core_w.items()}
            for p in pods:
                w[p] = w.get(p, 0.0) + share / len(pods)
            name = core_name if sat == "none" else f"{core_name} | {sat}@{int(round(share * 100))}"
            line = "LT" if sat in SAT_LT else "MAIN"
            out.append(Sleeve(name, w, line, {"core": core_name, "satellite": sat, "share": share}))
    return out


DEF_CORES = {
    "D0": (("core5", "btal_qqq"), "EQ", {"core5": 0.6, "btal_qqq": 0.4}, "FUNDED", "CORE5 60 / BTAL_QQQ 40"),
    "D1": (("core5", "btal_qqq"), "EQ", {"core5": 0.5, "btal_qqq": 0.5}, "FUNDED", "CORE5 + BTAL_QQQ [EQ]"),
    "D2": (("core5", "btal_qqq"), "IV", {}, "FUNDED", "CORE5 + BTAL_QQQ [IV]"),
    "D6": (("core5",), "EQ", {"core5": 1.0}, "FUNDED", "CORE5 alone"),
    "D7": (("btal_qqq",), "EQ", {"btal_qqq": 1.0}, "FUNDED", "BTAL_QQQ alone"),
    "D8": ((TBILL,), "EQ", {TBILL: 1.0}, "FUNDED", "T-bills (BIL)"),
    "D9": (("btal_qqq", TBILL), "EQ", {"btal_qqq": 0.5, TBILL: 0.5}, "FUNDED", "BTAL_QQQ 50 / BIL 50"),
    "D3": (("core5", "btal_qqq", "etf_dv2"), "IV", {}, "TARGET", "CORE5 + BTAL_QQQ + DV2-IND [IV]"),
    "D4": (("core5", "btal_qqq", "eom_flow"), "IV", {}, "TARGET", "CORE5 + BTAL_QQQ + EOM [IV]"),
    "D5": (("core5", "btal_qqq", "eom_flow", "etf_dv2"), "EQ",
           {"core5": 0.25, "btal_qqq": 0.25, "eom_flow": 0.25, "etf_dv2": 0.25}, "TARGET", "four-pod core [EQ]"),
}
FUNDED = [k for k, v in DEF_CORES.items() if v[3] == "FUNDED"]
TARGET = [k for k, v in DEF_CORES.items() if v[3] == "TARGET"]
NOW_CORES = ["D7", "D8", "D9"]


def def_book(key: str) -> Book:
    pods, rule, w, _, label = DEF_CORES[key]
    return Book(key, pods, rule, dict(w), "annual", "D")


def def_avg_weights(frame: pd.DataFrame, key: str, start: pd.Timestamp) -> dict:
    b = def_book(key)
    if b.rule == "EQ":
        return b.targets()
    log: list = []
    lib.book_returns(frame, b, start, weight_log=log)
    return lib.average_weights(log)


GROWTH_V = "TAA3x-1N + NDX-VXN 60:40 | def2@36"
AGGR_V = "TAA3x-1N + NDX-VXN 70:30 | def2@18"
G3 = "TAA3x + NDX-VXN 50:50"
CH = (AGGR_V, "D0")
CH_G = (GROWTH_V, "D0")


def client_name(r: str, d: str, s: float = S_CLIENT) -> str:
    return f"{r} || {d}" + ("" if abs(s - S_CLIENT) < 1e-9 else f" @s{s:.2f}")


# ─── account model (SPEC 3) ──────────────────────────────────────────────────


def period_labels(index: pd.DatetimeIndex, policy: str, phase: int = 0) -> np.ndarray:
    """Between-sleeve reset periods. 'annual' resets after the last close of December (phase 0) or of month `phase`."""
    if policy == "annual":
        if phase == 0:
            return index.year.to_numpy()
        # Reset after the last close of calendar month `phase`: label = the year of (month - phase).
        return (index.to_period("M") - phase).year.to_numpy()
    if policy == "quarterly":
        return (index.year * 4 + (index.month - 1) // 3).to_numpy()
    if policy == "drift":
        return np.zeros(len(index), dtype=int)
    raise ValueError(policy)


def account_returns(rA: np.ndarray, rD: np.ndarray, s, periods: np.ndarray, cost: float = TRANSFER_BPS) -> np.ndarray:
    """Account daily returns (N x K) for sleeve returns rA (N x K) and rD (N x K or N), share s (scalar or K)."""
    rA = np.asarray(rA, dtype=float)
    if rA.ndim == 1:
        rA = rA[:, None]
    n, k = rA.shape
    rD = np.asarray(rD, dtype=float)
    if rD.ndim == 1:
        rD = np.repeat(rD[:, None], k, axis=1)
    s = np.broadcast_to(np.asarray(s, dtype=float), (k,))
    bounds = np.r_[np.flatnonzero(np.r_[True, periods[1:] != periods[:-1]]), n]
    out = np.empty((n, k))
    value = np.ones(k)
    prev_end = np.ones(k)   # account value at the previous close (before any transfer cost)
    for idx_p, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        gA = np.cumprod(1.0 + rA[a:b], axis=0)
        gD = np.cumprod(1.0 + rD[a:b], axis=0)
        level = value * (s * gA + (1.0 - s) * gD)
        prev = np.vstack([prev_end[None, :], level[:-1]])
        out[a:b] = level / prev - 1.0
        end = level[-1]
        share_end = value * s * gA[-1] / end
        moved = np.abs(share_end - s) * end
        prev_end = end
        value = end - 2.0 * cost * moved      # *** CRITICAL*** the transfer cost lands in the next period's first return
    return out


def look_through(r_w: dict, d_w: dict, s: float) -> dict:
    out: dict = {}
    for p, v in r_w.items():
        out[p] = out.get(p, 0.0) + s * v
    for p, v in d_w.items():
        out[p] = out.get(p, 0.0) + (1.0 - s) * v
    return out


# ─── streaming bootstrap with the between-sleeve reset on each path (SPEC 7) ──


def boot_accounts(RA: np.ndarray, RD: np.ndarray, iA: np.ndarray, jD: np.ndarray, s, idx: np.ndarray,
                  reset_every: int = 252, cost: float = TRANSFER_BPS, fy: int = 252, dtype=np.float64) -> dict:
    """Per (path, combo): gross/net CAGR, max DD, first-year max DD. Combos are (iA[c], jD[c]) pairs of sleeve columns.

    Rows of RA and RD are resampled jointly (same indices). A 'year' = 252 sessions for fees and the reset.
    """
    reps, n = idx.shape
    C = len(iA)
    RA = np.asarray(RA, dtype=dtype)
    RD = np.asarray(RD, dtype=dtype)
    s = np.broadcast_to(np.asarray(s, dtype=dtype), (C,))
    gA = np.ones((reps, RA.shape[1]), dtype=dtype)
    gD = np.ones((reps, RD.shape[1]), dtype=dtype)
    V0 = np.ones((reps, C), dtype=dtype)
    prev = np.ones((reps, C), dtype=dtype)
    peak = np.ones((reps, C), dtype=dtype)
    dd = np.zeros((reps, C), dtype=dtype)
    pre = np.ones((reps, C), dtype=dtype)
    hwm = np.ones((reps, C), dtype=dtype)
    npeak = np.ones((reps, C), dtype=dtype)
    ndd = np.zeros((reps, C), dtype=dtype)
    net = np.ones((reps, C), dtype=dtype)
    fy_dd = None
    keep = dtype(1.0 - ga.MGMT / 252.0)
    perf = dtype(ga.PERF)
    for t in range(n):
        rows = idx[:, t]
        gA *= 1.0 + RA[rows]
        gD *= 1.0 + RD[rows]
        sA = gA[:, iA]
        level = V0 * (s * sA + (1.0 - s) * gD[:, jD])
        ret = level / prev
        np.maximum(peak, level, out=peak)
        np.minimum(dd, level / peak - 1.0, out=dd)
        pre *= ret
        pre *= keep
        acc = perf * np.maximum(pre - hwm, 0.0)
        np.subtract(pre, acc, out=net)
        np.maximum(npeak, net, out=npeak)
        np.minimum(ndd, net / npeak - 1.0, out=ndd)
        prev = level
        if t + 1 == fy:
            fy_dd = dd.copy()
        if (t + 1) % 252 == 0:
            paid = acc > 0.0
            pre -= acc
            hwm = np.where(paid, pre, hwm)
        if reset_every and (t + 1) % reset_every == 0:
            share_end = V0 * s * sA / level
            moved = np.abs(share_end - s) * level
            V0 = level - 2.0 * cost * moved
            prev = level
            gA[:] = 1.0
            gD[:] = 1.0
    years = n / 252.0
    f = lambda x: np.asarray(x, dtype=np.float64)  # noqa: E731
    return {"gross_cagr": f(level) ** (1.0 / years) - 1.0, "net_cagr": f(net) ** (1.0 / years) - 1.0,
            "gross_dd": f(dd), "net_dd": f(ndd), "fy_dd": f(fy_dd if fy_dd is not None else dd)}


def boot_index(n: int, seed: int, block: float = 63.0) -> np.ndarray:
    return lib.evaluation.stationary_bootstrap_index_mat(n, ga.BOOT_REPS, block, seed)
