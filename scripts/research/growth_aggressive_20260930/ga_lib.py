"""Shared pieces of the GROWTH / AGGRESSIVE study (SPEC_FROZEN.md): inputs, family, 2/20 fee path, bootstrap.

Inputs come read-only from the main checkout's shelf-rebuild study (`lib.load_inputs`); outputs go to this worktree's
`results/research/portfolio/growth_aggressive_20260930`.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WT_REPO = HERE.parents[2]
MAIN_REPO = Path(r"C:\Users\User\Documents\workspace\alpha_super")
SR_DIR = MAIN_REPO / "scripts" / "research" / "shelf_rebuild_20260929"
if str(SR_DIR) not in sys.path:
    sys.path.insert(0, str(SR_DIR))

import lib  # noqa: E402  (shelf rebuild: inputs, book model, metrics)
from lib import BLOCK_DICT, END, EXACT_START, LONG_START, TBILL, Book  # noqa: E402,F401

STUDY = WT_REPO / "results" / "research" / "portfolio" / "growth_aggressive_20260930"
MGMT, PERF = 0.02, 0.20
BOOT_REPS, BOOT_SEED = 2000, 20260929
RUNGS = {"GROWTH": (-0.17, -0.20), "AGGRESSIVE": (-0.22, -0.25)}
MAX_BREACH, TIE_SHARE, CHAMP_SHARE, COMPASS_SHARE = 0.15, 0.90, 0.80, 0.90
HPI_GAP = 0.026

TAA_LEGS = {"taa3x": "TAA3x", "taa3x_1n": "TAA3x-1N", "taa2x_1n": "TAA2x-1N"}
NDX_LEGS = {"ndx_vxn": "NDX-VXN", "ndx_atr": "NDX-ATR", "ndx_natr20": "NDX-NATR20"}
DV2_VARIANTS = ("dv2", "dv2_adv", "dv2_floor")
SATELLITES: dict[str, tuple[str, ...]] = {}
for _v in DV2_VARIANTS:
    SATELLITES[_v] = (_v,)
    SATELLITES[f"pair_{_v}"] = (_v, "hpi_vote")
    SATELLITES[f"capsule_{_v}"] = (_v, "hpi_vote", "etf_dv2")
SATELLITES.update({"hpi": ("hpi_vote",), "etf": ("etf_dv2",), "hpi_etf": ("hpi_vote", "etf_dv2"),
                   "def3": ("core5", "btal_qqq", "etf_dv2"), "def2": ("core5", "btal_qqq"), "tbill": (TBILL,)})
SHARES = (0.18, 0.36)
LOW_TOUCH_SATELLITES = {"none", "def2", "tbill"}
G3_NAME = "TAA3x + NDX-VXN"


def ledger(event_str: str, **fields) -> None:
    STUDY.mkdir(parents=True, exist_ok=True)
    record = {"event_str": event_str, "recorded_at_utc_str": datetime.now(timezone.utc).isoformat(),
              "spec_sha256_str": hashlib.sha256((HERE / "SPEC_FROZEN.md").read_bytes()).hexdigest(), **fields}
    with (STUDY / "experiment_ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


# ─── family (SPEC 4) ─────────────────────────────────────────────────────────


def family_books() -> list[Book]:
    books = []
    for taa, ndx, compass in product(TAA_LEGS, NDX_LEGS, (False, True)):
        legs = [taa, ndx] + (["compass_qqq"] if compass else [])
        core = " + ".join([TAA_LEGS[taa], NDX_LEGS[ndx]] + (["COMPASS-QQQ"] if compass else []))
        options = [("none", 0.0)] + [(s, sh) for s in SATELLITES for sh in SHARES]
        for sat, share in options:
            weights = {leg: (1.0 - share) / len(legs) for leg in legs}
            if sat != "none":
                pods = SATELLITES[sat]
                for p in pods:
                    weights[p] = weights.get(p, 0.0) + share / len(pods)
            name = core if sat == "none" else f"{core} | {sat}@{int(round(share * 100))}"
            tags = {"taa_leg": taa, "ndx_leg": ndx, "compass": compass, "satellite": sat, "share": share,
                    "low_touch": sat in LOW_TOUCH_SATELLITES, "twin": core.replace(" + COMPASS-QQQ", "")
                    + ("" if sat == "none" else f" | {sat}@{int(round(share * 100))}")}
            books.append(Book(name, tuple(weights), "EQ", weights, "annual", "GA", tags))
    return books


# ─── frames (SPEC 2, 8) ──────────────────────────────────────────────────────


def frames(data: dict) -> dict[str, tuple[pd.DataFrame, pd.Timestamp]]:
    """Every selection frame: name -> (daily sleeve returns, window start)."""
    cash_add = data["cash_long"] - data["long"]           # the fair-cash add of the main frame, per sleeve
    cash_add = cash_add.fillna(0.0)

    def with_add(frame: pd.DataFrame) -> pd.DataFrame:
        # NaN (no return yet) stays NaN: NaN + 0 = NaN.
        return frame + cash_add.reindex_like(frame).fillna(0.0)

    hpi = data["cash_long"].copy()
    live = hpi["hpi_vote"].notna()
    hpi.loc[live, "hpi_vote"] = hpi.loc[live, "hpi_vote"] - HPI_GAP / 252.0
    return {"main": (data["cash_long"], LONG_START),
            "s1_house_cash": (data["long"], LONG_START),
            "s2_proxy_unscaled": (with_add(data["long_unscaled"]), LONG_START),
            "s3_plus_5bps": (with_add(data["stressed_long"]), LONG_START),
            "s4_etf_idle_pre2010": (with_add(data["long_etf_cash"]), LONG_START),
            "s5_hpi_live_gap": (hpi, LONG_START),
            "s6_exact": (data["cash_exact"], EXACT_START),
            "s7_block126": (data["cash_long"], LONG_START),
            "s8_block21": (data["cash_long"], LONG_START)}


BLOCK_BY_FRAME = {"s7_block126": 126.0, "s8_block21": 21.0}


# ─── 2/20 fee path (SPEC 3) ──────────────────────────────────────────────────


def year_end_flags(index: pd.DatetimeIndex) -> np.ndarray:
    y = index.year.to_numpy()
    return np.r_[y[1:] != y[:-1], True]


def fee_nav(R: np.ndarray, year_end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(investor net NAV, pre-fee NAV) paths, N x B, for gross daily returns R (N x B); start NAV 1, HWM 1.

    Management fee 2%/252 of the pre-fee NAV each session; 20% of the gain over the HWM accrued each session and
    paid at a year end (HWM resets to the post-payment NAV only when a fee was paid).
    """
    R = np.atleast_2d(R.T).T if R.ndim == 1 else R
    n, b = R.shape
    pre, hwm = np.ones(b), np.ones(b)
    net_out, pre_out = np.empty((n, b)), np.empty((n, b))
    keep = 1.0 - MGMT / 252.0
    for t in range(n):
        pre *= (1.0 + R[t]) * keep
        acc = PERF * np.maximum(pre - hwm, 0.0)
        net_out[t] = pre - acc
        pre_out[t] = pre
        if year_end[t]:
            paid = acc > 0.0
            pre = pre - acc
            hwm = np.where(paid, pre, hwm)
    return net_out, pre_out


def dd_of_nav(nav: np.ndarray) -> np.ndarray:
    nav1 = np.vstack([np.ones((1, nav.shape[1])), nav])
    return (nav1 / np.maximum.accumulate(nav1, axis=0) - 1.0).min(axis=0)


def cagr_of_nav(nav_end: np.ndarray, days: int) -> np.ndarray:
    return nav_end ** (365.25 / days) - 1.0


def window_stats(R: pd.DataFrame, index_all: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Gross and net CAGR / max DD / Sharpe for each column of R (one window, new investor at its start)."""
    base = lib.base_date(index_all, R.iloc[:, 0])
    days = (R.index[-1] - base).days
    arr = R.to_numpy(dtype=float)
    gross_nav = np.cumprod(1.0 + arr, axis=0)
    net_nav, _ = fee_nav(arr, year_end_flags(R.index))
    net_r = np.vstack([net_nav[:1] - 1.0, net_nav[1:] / net_nav[:-1] - 1.0])
    return {"gross_cagr": cagr_of_nav(gross_nav[-1], days), "net_cagr": cagr_of_nav(net_nav[-1], days),
            "gross_dd": dd_of_nav(gross_nav), "net_dd": dd_of_nav(net_nav),
            "gross_sharpe": arr.mean(axis=0) / arr.std(axis=0, ddof=1) * np.sqrt(252),
            "net_sharpe": net_r.mean(axis=0) / net_r.std(axis=0, ddof=1) * np.sqrt(252)}


def net_return_series(r: pd.Series) -> pd.Series:
    net_nav, _ = fee_nav(r.to_numpy(dtype=float)[:, None], year_end_flags(r.index))
    nav = pd.Series(net_nav[:, 0], index=r.index)
    return nav.pct_change().fillna(nav.iloc[0] - 1.0)


def fee_breakdown(r: pd.Series) -> pd.DataFrame:
    """Per calendar year: gross return, investor net return, management and performance fee (as NAV fractions)."""
    rows = []
    arr = r.to_numpy(dtype=float)
    years = r.index.year.to_numpy()
    pre, hwm, start, mgmt_year, gross_year = 1.0, 1.0, 1.0, 0.0, 1.0
    avg_acc = []
    keep = MGMT / 252.0
    for t, g in enumerate(arr):
        pre *= 1.0 + g
        gross_year *= 1.0 + g
        fee = pre * keep
        pre -= fee
        mgmt_year += fee
        acc = PERF * max(pre - hwm, 0.0)
        avg_acc.append(pre - acc)
        if t == len(arr) - 1 or years[t + 1] != years[t]:
            rows.append({"year": int(years[t]), "gross": gross_year - 1.0, "net": (pre - acc) / start - 1.0,
                         "mgmt_fee": mgmt_year, "perf_fee": acc, "start_nav": start,
                         "avg_nav": float(np.mean(avg_acc)), "sessions": len(avg_acc)})
            pre -= acc
            if acc > 0:
                hwm = pre
            start, mgmt_year, gross_year, avg_acc = pre, 0.0, 1.0, []
    return pd.DataFrame(rows)


# ─── bootstrap (SPEC 5 R2, 6) ────────────────────────────────────────────────


def boot_index(n: int, block: float = 63.0) -> np.ndarray:
    return lib.evaluation.stationary_bootstrap_index_mat(n, BOOT_REPS, block, BOOT_SEED)


def bootstrap_paths(R: np.ndarray, idx: np.ndarray) -> dict[str, np.ndarray]:
    """Stream the resampled paths day by day: gross / net CAGR and max DD per (path, book). A year = 252 sessions."""
    reps, n = idx.shape
    b = R.shape[1]
    gross = np.ones((reps, b))
    gpeak = np.ones((reps, b))
    gdd = np.zeros((reps, b))
    pre = np.ones((reps, b))
    hwm = np.ones((reps, b))
    npeak = np.ones((reps, b))
    ndd = np.zeros((reps, b))
    net = np.ones((reps, b))
    keep = 1.0 - MGMT / 252.0
    acc = np.zeros((reps, b))
    tmp = np.empty((reps, b))
    for t in range(n):
        r = R[idx[:, t]]
        np.add(r, 1.0, out=tmp)
        gross *= tmp
        np.maximum(gpeak, gross, out=gpeak)
        np.minimum(gdd, gross / gpeak - 1.0, out=gdd)
        pre *= tmp
        pre *= keep
        np.subtract(pre, hwm, out=acc)
        np.maximum(acc, 0.0, out=acc)
        acc *= PERF
        np.subtract(pre, acc, out=net)
        np.maximum(npeak, net, out=npeak)
        np.minimum(ndd, net / npeak - 1.0, out=ndd)
        if (t + 1) % 252 == 0:
            paid = acc > 0.0
            pre -= acc
            hwm = np.where(paid, pre, hwm)
    years = n / 252.0
    return {"gross_cagr": gross ** (1.0 / years) - 1.0, "net_cagr": net ** (1.0 / years) - 1.0,
            "gross_dd": gdd, "net_dd": ndd}
