"""Fund products, final pass (SPEC_FROZEN.md, 2026-10-05): inputs with the two new capsules, books, frames, the Lab.

The existing sleeves come unchanged from the shelf-rebuild frames (``lib.load_inputs`` of the main checkout). The four
new sleeves (momentum capsule pods ``ndx_atr_cap`` / ``ndx_natr_cap``, MR capsule pods ``dv2_g`` / ``hpi_g``) are fresh
runs from this worktree (``build_sources.py``) and are added to every frame with the same formulas as the old ones.

Import order matters: the shelf-rebuild ``lib.py`` puts the main checkout first on ``sys.path``, and that checkout
holds another session's uncommitted edits. This module loads the worktree's ``data`` / ``strategies`` packages first
and puts the worktree back in front afterwards. Run with PYTHONDONTWRITEBYTECODE=1.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WT_REPO = HERE.parents[2]
OLD_DIR = HERE.parent / "fund_products_20260930"
GA_DIR = HERE.parent / "growth_aggressive_20260930"
sys.path.insert(0, str(WT_REPO))
import data.norgate_loader  # noqa: E402,F401  (worktree copy, before lib.py can put the main checkout in front)
import strategies.mr_capsule.vix_stress_gate as vix_gate  # noqa: E402  (pure gate arithmetic, worktree copy)

for _p in (OLD_DIR, GA_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ga_lib as ga  # noqa: E402  (fee model, bootstrap streaming, frames)
from ga_lib import BLOCK_DICT, END, EXACT_START, LONG_START, TBILL, Book, lib  # noqa: E402,F401

sys.path = [str(WT_REPO)] + [p for p in sys.path if Path(p).resolve() not in (ga.MAIN_REPO.resolve(), WT_REPO.resolve())]

STUDY = WT_REPO / "results" / "research" / "portfolio" / "fund_products_20261005"
SOURCES = STUDY / "sources"
OUT = STUDY / "report"
SEED0 = 20260929                      # the earlier studies' bootstrap seed: old books reproduce their old tails
BOOT_REPS, BOOT_BLOCK = 2000, 63.0
CUT = pd.Timestamp("2017-06-30")      # halves
SPREAD = 0.015                        # margin spread over DTB3
LIMITS = (-0.10, -0.15, -0.17, -0.20, -0.22, -0.25, -0.27, -0.30, -0.35)
RUNGS = {"GROWTH": (-0.17, -0.20, 0.15), "GROWTH PLUS": (-0.22, -0.25, 0.15), "AGGRESSIVE": (-0.27, -0.30, 0.15)}
RUNG_ORDER = ("GROWTH", "GROWTH PLUS", "AGGRESSIVE")
MAX_BIL = 0.30

NEW_ALIAS_LIST = ["ndx_atr_cap", "ndx_natr_cap", "dv2_g", "hpi_g"]
CASH_RUN = {"dv2_g": "dv2_g_cash", "hpi_g": "hpi_g_cash"}      # parking disabled (idle cash 0%)
PARKING_SYMBOL = "BIL"
QQQ = "qqq_tr"                        # QQQ total return, a passive column for slot test T4
DEBT = "debt"                         # a pod that grows at DTB3 + spread (margin loan), used with a negative weight
NEW_OPS = {
    "ndx_atr_cap": dict(route="monthly MOO", instruments="~10 Nasdaq-100 stocks", daily=False, moc=False, short=False, levered=False, fred=False),
    "ndx_natr_cap": dict(route="monthly MOO", instruments="~10 Nasdaq-100 stocks", daily=False, moc=False, short=False, levered=False, fred=False),
    "dv2_g": dict(route="daily MOO (margin)", instruments="S&P 500 stocks + BIL", daily=True, moc=False, short=False, levered=False, fred=False),
    "hpi_g": dict(route="daily MOO (margin)", instruments="S&P 500 stocks + BIL", daily=True, moc=False, short=False, levered=False, fred=False),
}

# ─── capsules and books (SPEC 2-4) ───────────────────────────────────────────

MOM = {"ndx_atr_cap": 0.5, "ndx_natr_cap": 0.5}
MR = {"dv2_g": 0.5, "hpi_g": 0.5}
DEF = {"core5": 0.6, "btal_qqq": 0.4}
CAPSULE_OF = {"taa3x": "TAA", "taa3x_1n": "TAA", "ndx_atr_cap": "MOM", "ndx_natr_cap": "MOM", "ndx_vxn": "MOM",
              "dv2_g": "MR", "hpi_g": "MR", "dv2": "MR", "hpi_vote": "MR", "core5": "DEF", "btal_qqq": "DEF"}
CLUSTER_OF = {"TAA": "Nasdaq pair", "MOM": "Nasdaq pair", "MR": "MR", "DEF": "DEF"}


def blend(*parts: tuple[float, dict]) -> dict:
    """Weighted sum of weight dicts (each normalised first); the result sums to 1."""
    out: dict = {}
    for share, w in parts:
        tot = sum(w.values())
        for k, v in w.items():
            out[k] = out.get(k, 0.0) + share * v / tot
    s = sum(out.values())
    return {k: v / s for k, v in out.items() if v > 1e-12}


def three(taa: str, t: float, m: float, r: float) -> dict:
    """TAA share t on one TAA variant, momentum capsule m, MR capsule r."""
    parts = [(x, w) for x, w in ((t, {taa: 1.0}), (m, MOM), (r, MR)) if x > 1e-12]
    return blend(*parts)


def with_cash(w: dict, c: float) -> dict:
    return blend((1 - c, w), (c, {TBILL: 1.0})) if c > 0 else blend((1.0, w))


# Owner decision 2026-10-05, after results (amendment O1): the flagship moves from equal capital (1/3 each, the
# registered default, kept below as challenger S0) to the dial point TAA 40 / momentum 30 / MR 30.
G1 = (0.40, 0.30, 0.30)
PRODUCTS = {
    "GR1": three("taa3x", *G1),
    "GR2": three("taa3x_1n", *G1),          # owner decision 2026-10-05 (amendment O2): the same 40 / 30 / 30 with the 1N variant
    "GR3": three("taa3x_1n", 0.60, 0.20, 0.20),   # owner decision 2026-10-05 (amendment O6); was 1/2, 1/4, 1/4
}
PRODUCT_SHAPE = {"GR1": ("taa3x", *G1), "GR2": ("taa3x_1n", *G1), "GR3": ("taa3x_1n", 0.60, 0.20, 0.20)}
PRODUCT_LABEL = {"GR1": "Growth", "GR2": "Growth Plus", "GR3": "Aggressive"}
TARGET_RUNG = {"GR1": "GROWTH", "GR2": "GROWTH PLUS", "GR3": "AGGRESSIVE"}
INCUMBENT = {"taa3x_1n": 0.384, "ndx_vxn": 0.256, "core5": 0.18, "btal_qqq": 0.18}
OLD_PLUS = {"taa3x_1n": 0.574, "ndx_vxn": 0.246, "core5": 0.09, "btal_qqq": 0.09}
# Owner decision 2026-10-05 (amendment O3): the monthly books are two pods, TAA 3x 1N and CORE5. INCUMBENT and OLD_PLUS
# (the 2026-10-01 four-pod books) stay as reference rows. The internal keys "S9 incumbent launch" and "old growth plus"
# now hold the NEW Monthly and Monthly Plus; the old books are S13_OLD and OLD_PLUS_KEY.
MONTHLY = {"taa3x_1n": 0.60, "core5": 0.40}      # owner decision 2026-10-05 (O8): one monthly book, 60 / 40; was 50 / 50 plus a 65 / 35 "plus"
MONTHLY_PLUS = {"taa3x_1n": 0.65, "core5": 0.35}
S13_OLD, OLD_PLUS_KEY = "S13 old monthly (2026-10-01)", "old monthly plus (2026-10-01)"
S9, S10 = "S9 incumbent launch", "S10 old G3"
S0 = "S0 equal capital (registered default)"
CHALLENGERS = {
    S0: three("taa3x", 1 / 3, 1 / 3, 1 / 3),
    "S1 no momentum": three("taa3x", 0.5, 0.0, 0.5),
    "S2 no MR": three("taa3x", 0.5, 0.5, 0.0),
    "S3 no TAA": blend((0.5, MOM), (0.5, MR)),
    "S4 core + satellites": three("taa3x", 0.50, 0.25, 0.25),
    "S5 MR tilt": three("taa3x", 0.50, 0.15, 0.35),
    "S6 equal pods": three("taa3x", 0.20, 0.40, 0.40),
    "S8 GR1 75 / DEF 25": blend((0.75, PRODUCTS["GR1"]), (0.25, DEF)),
    S9: MONTHLY,
    S13_OLD: INCUMBENT,
    S10: {"taa3x": 0.5, "ndx_vxn": 0.5},
    "S11 cluster parity": three("taa3x", 0.25, 0.25, 0.50),
    "S12 inverse vol (fixed)": three("taa3x", 0.31, 0.33, 0.36),
}
S7 = "S7 inverse vol (walk-forward)"
RUNG_EXEMPT = {S9, S10}                # tested as they are (SPEC 4, check 0)
SLOT_TESTS = {
    "T1 MOM -> BIL": blend((G1[0], {"taa3x": 1.0}), (G1[1], {TBILL: 1.0}), (G1[2], MR)),
    "T2 MR -> BIL (GR1-L)": blend((G1[0], {"taa3x": 1.0}), (G1[1], MOM), (G1[2], {TBILL: 1.0})),
    "T3 TAA -> BIL": blend((G1[0], {TBILL: 1.0}), (G1[1], MOM), (G1[2], MR)),
    "T4 MOM -> QQQ": blend((G1[0], {"taa3x": 1.0}), (G1[1], {QQQ: 1.0}), (G1[2], MR)),
}
GR1_L = "T2 MR -> BIL (GR1-L)"
DIAL_SHARES = (1 / 3, 0.40, 0.50, 0.60, 2 / 3)
STAND_INS = {
    "GR1 with live NDX rule": blend((G1[0], {"taa3x": 1.0}), (G1[1], {"ndx_vxn": 1.0}), (G1[2], MR)),
    "GR1 with ungated DV2 + HPI": blend((G1[0], {"taa3x": 1.0}), (G1[1], MOM), (G1[2], {"dv2": 0.5, "hpi_vote": 0.5})),
    "GR1 all wired today": blend((G1[0], {"taa3x": 1.0}), (G1[1], {"ndx_vxn": 1.0}), (G1[2], {"dv2": 0.5, "hpi_vote": 0.5})),
}


def ledger(event_str: str, **fields) -> None:
    STUDY.mkdir(parents=True, exist_ok=True)
    rec = {"event_str": event_str, "recorded_at_utc_str": datetime.now(timezone.utc).isoformat(),
           "spec_sha256_str": hashlib.sha256((HERE / "SPEC_FROZEN.md").read_bytes()).hexdigest(), **fields}
    with (STUDY / "experiment_ledger.jsonl").open("a", encoding="utf-8") as h:
        h.write(json.dumps(rec, default=str) + "\n")


def r6(x):
    """JSON-safe copy with floats rounded to 6 places."""
    if isinstance(x, dict):
        return {str(k): r6(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [r6(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), 6)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


# ─── inputs (SPEC 1) ─────────────────────────────────────────────────────────


def load_inputs() -> dict:
    """The shelf-rebuild frames plus the new sleeves, each added with the old sleeves' formulas.

    house (data["long"], data["sleeve"]): the engine run as is (idle cash 0%; the MR pods hold BIL as a position).
    fair cash (data["cash_long"], ["cash_exact"]): + lib.cash_realism_add on the run's own cash column (for the MR pods
        that is their residual cash beside BIL).
    +5 bps (data["stressed_long"], ["stressed_exact"]): - 5 bps x |traded notional| / prior NAV, every fill.
    data["drag_ex_bil"]: the same drag for the MR pods without their BIL fills (the "stock fills only" frame).
    data["mr_cash_run"], ["mr_cash_add"]: the MR pods with parking disabled (idle cash 0%) and their fair-cash add.
    data["full"]: every new sleeve's house return on its whole run (before 2008 and after END), for the appendices.
    QQQ total return and the margin-debt pod are added as passive columns.
    """
    data = lib.load_inputs()
    index = data["index"]
    rate = lib.dtb3_annual_rate(index)
    old_cols = list(data["cash_long"].columns)
    data["path"], data["full"], data["mr_cash_run"], data["mr_cash_add"], data["drag_ex_bil"], data["drag"] = {}, {}, {}, {}, {}, {}
    for alias in NEW_ALIAS_LIST + list(CASH_RUN.values()):
        path = lib.read_path(SOURCES, alias)
        full = lib.nav_to_returns(path)
        r = full.reindex(index)
        if r.loc[LONG_START:END].isna().any():
            raise ValueError(f"{alias}: missing sessions inside the LONG window")
        data["full"][alias] = full
        data["path"][alias] = path
        add = lib.cash_realism_add(path, rate).reindex(index).fillna(0.0)
        if alias in CASH_RUN.values():
            data["mr_cash_run"][alias], data["mr_cash_add"][alias] = r, add
            continue
        tx = lib.read_tx(SOURCES, alias)
        nav = path["total_value_float"]
        drag = lib.evaluation.extra_slippage_cost_ser(tx, nav, 0.0005).reindex(index).fillna(0.0)
        data["drag"][alias] = drag
        data["drag_ex_bil"][alias] = lib.evaluation.extra_slippage_cost_ser(
            tx[tx["asset_str"] != PARKING_SYMBOL], nav, 0.0005).reindex(index).fillna(0.0)
        for key in ("sleeve", "long", "long_unscaled", "long_etf_cash"):
            data[key][alias] = r
        for key in ("stressed_long", "stressed_exact"):
            data[key][alias] = r - drag          # NaN before the first return stays NaN
        for key in ("cash_long", "cash_exact"):
            data[key][alias] = r + add
        data["tx"][alias], data["nav"][alias] = tx, nav
        data["meta"][alias] = json.loads((SOURCES / f"{alias}__metadata.json").read_text(encoding="utf-8"))
    qqq = data["bench"]["QQQ"].reindex(index)
    for key in ("sleeve", "long", "long_unscaled", "long_etf_cash", "stressed_long", "stressed_exact", "cash_long", "cash_exact"):
        data[key][QQQ] = qqq                     # passive total return: no cash add, no cost drag
        assert list(data[key].columns[:len(old_cols)]) == old_cols
    data["dtb3"] = rate
    return data


def frames(data: dict) -> dict[str, tuple[pd.DataFrame, pd.Timestamp]]:
    """ga.frames (main, house cash, unscaled proxy, +5 bps, EXACT, ...) plus this study's extra frames:

    s3b_plus_5bps_ex_bil  +5 bps on stock and risk-ETF fills only (the MR pods' BIL fills are not charged)
    s9_mr_fair_cash       main, with the MR pods' parking-off runs plus the fair-cash add (the symmetric treatment)
    s10_mr_cash_0         main, with the MR pods' parking-off runs as they are (their idle cash at 0%)
    Every frame also carries the margin-debt pod at three spreads (debt = 1.5%, debt_050, debt_250).
    """
    out = ga.frames(data)
    main = data["cash_long"]
    ex = out["s3_plus_5bps"][0].copy()
    for alias in CASH_RUN:
        live = ex[alias].notna()
        ex.loc[live, alias] = ex.loc[live, alias] + (data["drag"][alias] - data["drag_ex_bil"][alias])[live]
    fair, zero = main.copy(), main.copy()
    for alias, cash_alias in CASH_RUN.items():
        zero[alias] = data["mr_cash_run"][cash_alias]
        fair[alias] = data["mr_cash_run"][cash_alias] + data["mr_cash_add"][cash_alias]
    out["s3b_plus_5bps_ex_bil"] = (ex, LONG_START)
    out["s9_mr_fair_cash"] = (fair, LONG_START)
    out["s10_mr_cash_0"] = (zero, LONG_START)
    index = main.index
    days = pd.Series(index, index=index).diff().dt.days
    for key, (fr, start) in list(out.items()):
        fr = fr.copy()
        for name, spread in ((DEBT, SPREAD), (f"{DEBT}_050", 0.005), (f"{DEBT}_250", 0.025)):
            # *** CRITICAL*** DTB3 is the prior observation (lib.dtb3_annual_rate shifts by one); ACT/360.
            fr[name] = ((data["dtb3"] + spread) * days / 360.0).reindex(fr.index)
        out[key] = (fr, start)
    return out


def levered(w: dict, L: float, debt: str = DEBT) -> dict:
    """Margin as a pod with a negative weight: L x w on the pods, -(L - 1) on the debt pod (weights sum to 1)."""
    out = {k: L * v for k, v in w.items()}
    if abs(L - 1.0) > 1e-12:
        out[debt] = -(L - 1.0)
    return out


# ─── book model with other reset policies (SPEC 5.5) ─────────────────────────


def period_ids(index: pd.DatetimeIndex, policy: str) -> np.ndarray:
    """Reset-period label per session. "annual" = lib's rule (first session of each calendar year);
    "annual-m" resets at the first session of calendar month m; "quarterly"; "monthly"; "none"."""
    if policy in ("annual", "none"):
        return lib.period_ids(index, policy)
    if policy == "monthly":
        return (index.year * 12 + index.month).to_numpy()
    if policy == "quarterly":
        return (index.year * 4 + (index.month - 1) // 3).to_numpy()
    if policy.startswith("annual-"):
        m = int(policy.split("-")[1])
        return (index.year + (index.month >= m)).to_numpy()
    raise ValueError(policy)


def book_returns(frame: pd.DataFrame, w: dict, start: pd.Timestamp, end: pd.Timestamp = END, policy: str = "annual",
                 reset_cost: float = 0.0, weight_path: list | None = None) -> pd.Series:
    """lib.book_returns for fixed weights with any reset policy (identical to it for "annual"; checked in study.py).

    reset_cost charges that fraction of the capital moved between pods at each reset (sum of |drifted - target|
    weights), on the reset session. weight_path, if given, receives (index, prior-close weight matrix).
    """
    cols = list(w)
    window = frame.loc[start:end, cols]
    if window.isna().any().any():
        missing = window.isna().sum()
        raise ValueError(f"missing returns in window {missing[missing > 0].to_dict()}")
    ret = window.to_numpy(dtype=float)
    fixed_w = np.array([w[c] for c in cols], dtype=float)
    if abs(fixed_w.sum() - 1.0) > 1e-9:
        raise ValueError(f"weights sum to {fixed_w.sum()}")
    periods = period_ids(window.index, policy)
    bounds = np.r_[np.flatnonzero(np.r_[True, periods[1:] != periods[:-1]]), len(ret)]
    out, prior_w = np.empty(len(ret)), np.empty_like(ret)
    value, drift = 1.0, fixed_w.copy()
    for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        start_value = value
        if reset_cost and k > 0:
            start_value = value * (1.0 - reset_cost * float(np.abs(drift - fixed_w).sum()))
        pod_val = np.cumprod(1.0 + ret[a:b], axis=0) * fixed_w       # pods compound, reset at the period start
        level = start_value * pod_val.sum(axis=1)
        out[a:b] = level / np.r_[value, level[:-1]] - 1.0            # the reset session's return carries the cost
        prior_w[a:b] = np.vstack([fixed_w, pod_val[:-1] / pod_val[:-1].sum(axis=1, keepdims=True)])
        drift = pod_val[-1] / pod_val[-1].sum()
        value = level[-1]
    if weight_path is not None:
        weight_path.append((window.index, cols, prior_w))
    return pd.Series(out, index=window.index)


def xsharpe(r: np.ndarray, rf: np.ndarray) -> float:
    x = np.asarray(r, dtype=float) - np.asarray(rf, dtype=float)
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252))


def stats(r: pd.Series, rf: pd.Series) -> dict:
    """CAGR (252 sessions a year), volatility, excess Sharpe over BIL, house Sharpe (rf 0), max DD."""
    x, f = r.to_numpy(dtype=float), rf.reindex(r.index).to_numpy(dtype=float)
    nav = np.r_[1.0, np.cumprod(1.0 + x)]
    return {"cagr": float(nav[-1] ** (252 / len(x)) - 1), "vol": float(x.std(ddof=1) * np.sqrt(252)), "xs": xsharpe(x, f),
            "sharpe0": float(x.mean() / x.std(ddof=1) * np.sqrt(252)),
            "dd": float((nav / np.maximum.accumulate(nav) - 1.0).min()), "n": int(len(x))}


def monthly_xs(r: pd.Series, rf: pd.Series) -> float:
    """Excess Sharpe on calendar-month returns, x sqrt(12)."""
    m = (1.0 + r).groupby([r.index.year, r.index.month]).prod() - 1.0
    f = (1.0 + rf.reindex(r.index)).groupby([r.index.year, r.index.month]).prod() - 1.0
    x = (m - f).to_numpy()
    return float(x.mean() / x.std(ddof=1) * np.sqrt(12))


def gross_dd(R: np.ndarray, idx_t: np.ndarray) -> np.ndarray:
    """Max drawdown per (path, book): the streaming arithmetic of ga.bootstrap_paths, gross only. idx_t is
    day-major (days x paths)."""
    n, reps = idx_t.shape
    b = R.shape[1]
    gross, peak, dd, tmp = np.ones((reps, b)), np.ones((reps, b)), np.zeros((reps, b)), np.empty((reps, b))
    for t in range(n):
        np.add(R[idx_t[t]], 1.0, out=tmp)
        gross *= tmp
        np.maximum(peak, gross, out=peak)
        np.minimum(dd, gross / peak - 1.0, out=dd)
    return dd


class Lab:
    """Books on the study's frames: returns, quick statistics, bootstrap tails (cached), rungs, challenge test."""

    def __init__(self) -> None:
        self.data = load_inputs()
        self.frames = frames(self.data)
        self.frame, self.start = self.frames["main"]
        self.rf = self.frame[TBILL]
        self.n = len(self.frame.loc[self.start:END])
        self.cands: dict[str, dict] = {}
        self._ret: dict = {}
        self._idx: dict = {}
        self._disk_path = OUT / "tail_cache.json"
        self._disk = json.loads(self._disk_path.read_text(encoding="utf-8")) if self._disk_path.exists() else {}
        self._fp: dict[str, str] = {}

    # ── series ──
    def ret(self, w: dict, frame_key: str = "main", policy: str = "annual", frame: pd.DataFrame | None = None) -> pd.Series:
        """Daily returns of a fixed-weight book (annual reset by default). `frame` overrides the frame's data
        (same window), uncached: used for the edge-decay scenarios."""
        if frame is not None:
            return book_returns(frame, w, self.frames[frame_key][1], policy=policy)
        key = (tuple(sorted((k, round(v, 12)) for k, v in w.items())), frame_key, policy)
        if key not in self._ret:
            fr, s = self.frames[frame_key]
            self._ret[key] = book_returns(fr, w, s, policy=policy)
        return self._ret[key]

    def quick(self, r: pd.Series) -> dict:
        x = r.to_numpy()
        out = stats(r, self.rf)
        cr = {k: float(lib.common.window_return_float(r, lo, hi)) for k, (lo, hi) in lib.CRISIS_DICT.items()}
        yrs = (1.0 + r).groupby(r.index.year).prod() - 1.0
        down = x[x < 0]
        out.update({"sortino": float(x.mean() / down.std(ddof=1) * np.sqrt(252)), "xs_monthly": monthly_xs(r, self.rf),
                    "crises": cr, "worst_crisis": float(min(cr.values())), "worst_year": float(yrs.min()),
                    "worst_year_label": int(yrs.idxmin()), "best_year": float(yrs.max())})
        return out

    def add(self, name: str, w: dict, **meta) -> str:
        if name not in self.cands:
            self.cands[name] = {"w": w, "q": self.quick(self.ret(w)), **meta}
        return name

    def add_series(self, name: str, series_fn, **meta) -> str:
        """A book that is not a fixed-weight mix: series_fn(frame_key) returns its daily returns on that frame."""
        if name not in self.cands:
            self.cands[name] = {"w": {}, "series_fn": series_fn, **meta}
            self.cands[name]["q"] = self.quick(self.r(name))
        return name

    def r(self, n: str, frame_key: str = "main") -> pd.Series:
        c = self.cands[n]
        if "series_fn" in c:
            key = ("series", n, frame_key)
            if key not in self._ret:
                self._ret[key] = c["series_fn"](frame_key)
            return self._ret[key]
        return self.ret(c["w"], frame_key)

    # ── bootstrap (stationary, 2,000 paths per seed; seeds SEED0 + 0..9) ──
    def idx(self, s: int, block: float = BOOT_BLOCK, n: int | None = None) -> np.ndarray:
        """Day-major (days x paths) bootstrap index of seed s for a window of n sessions (default: LONG)."""
        n = self.n if n is None else n
        if (s, block, n) not in self._idx:
            mat = lib.evaluation.stationary_bootstrap_index_mat(n, BOOT_REPS, block, SEED0 + s)
            self._idx[(s, block, n)] = np.ascontiguousarray(mat.T.astype(np.int32))
        return self._idx[(s, block, n)]

    def tail_matrix(self, R: np.ndarray, limits=LIMITS, block: float = BOOT_BLOCK, seeds=range(10), horizon: int | None = None) -> dict:
        """P(max DD < limit) per column of R, per seed: {limit: array (seeds x books)}. horizon = the first h
        sessions of each path only."""
        assert not np.isnan(R).any()
        acc = {L: [] for L in limits}
        for s in seeds:
            idx_t = self.idx(s, block, len(R))
            dd = gross_dd(R, idx_t if horizon is None else idx_t[:horizon])
            for L in limits:
                acc[L].append((dd < L).mean(axis=0))
        return {L: np.array(v) for L, v in acc.items()}

    def _fingerprint(self, frame_key: str) -> str:
        if frame_key not in self._fp:
            fr, s = self.frames[frame_key]
            block = fr.loc[s:END].fillna(0.0)
            h = hashlib.sha256(np.ascontiguousarray(block.to_numpy(dtype=float)).tobytes())
            h.update(("|".join(block.columns) + f"|{s}|{END}|{BOOT_REPS}|{BOOT_BLOCK}|{SEED0}|{LIMITS}").encode())
            self._fp[frame_key] = h.hexdigest()[:16]
        return self._fp[frame_key]

    def _key(self, n: str, frame_key: str) -> str:
        c = self.cands[n]
        if "series_fn" in c:
            digest = hashlib.sha256(np.ascontiguousarray(self.r(n, frame_key).to_numpy(dtype=float)).tobytes()).hexdigest()[:16]
            return json.dumps([self._fingerprint(frame_key), "series", n, digest, frame_key])
        return json.dumps([self._fingerprint(frame_key), sorted((k, round(v, 12)) for k, v in c["w"].items()), frame_key])

    def run_tails(self, names: list[str], frame_key: str = "main") -> None:
        todo = [n for n in dict.fromkeys(names) if len(self._disk.get(self._key(n, frame_key), [])) < 10]
        if not todo:
            return
        R = np.column_stack([self.r(n, frame_key).to_numpy() for n in todo])
        tm = self.tail_matrix(R)
        for j, n in enumerate(todo):
            self._disk[self._key(n, frame_key)] = [{str(L): float(tm[L][s, j]) for L in LIMITS} for s in range(10)]
        OUT.mkdir(parents=True, exist_ok=True)
        self._disk_path.write_text(json.dumps(self._disk), encoding="utf-8")
        print(f"  tails {frame_key}: {len(todo)} books", flush=True)

    def tails(self, n: str, frame_key: str = "main") -> dict:
        self.run_tails([n], frame_key)
        a = self._disk[self._key(n, frame_key)]
        out = {}
        for L in LIMITS:
            vals = [d[str(L)] for d in a]
            key = f"p{int(round(-L * 100))}"
            out[key], out[key + "_max"] = float(np.mean(vals)), float(np.max(vals))
        return out

    # ── rungs (SPEC 3: historical max DD and the breach cap, mean and worst seed, MAIN and +5 bps) ──
    def rung_detail(self, n: str, rung: str) -> dict:
        build, hard, cap = RUNGS[rung]
        key = f"p{int(round(-hard * 100))}"
        out = {"rung": rung, "pass": True}
        for frame_key in ("main", "s3_plus_5bps"):
            dd = stats(self.r(n, frame_key), self.rf)["dd"]
            t = self.tails(n, frame_key)
            ok = bool(dd >= build and t[key] <= cap and t[key + "_max"] <= cap)
            out[frame_key] = {"dd": dd, "breach_mean": t[key], "breach_worst_seed": t[key + "_max"], "pass": ok}
            out["pass"] = out["pass"] and ok
        return out

    def rung_ok(self, n: str, rung: str) -> bool:
        return self.rung_detail(n, rung)["pass"]

    def strictest_rung(self, n: str) -> str | None:
        for rung in RUNG_ORDER:
            if self.rung_ok(n, rung):
                return rung
        return None

    # ── comparisons (SPEC 4) ──
    def paired(self, ra: pd.Series, rb: pd.Series, rf: pd.Series | None = None, seeds=range(10)) -> dict:
        """Paired stationary bootstrap of two books over the same resampled rows (seeds pooled): share of paths where
        a's excess Sharpe (and CAGR) is above b's, and the 5 / 50 / 95% points of the gap a - b."""
        rf = self.rf if rf is None else rf
        A = pd.concat([ra, rb, rf.reindex(ra.index)], axis=1).dropna().to_numpy()
        gaps_xs, gaps_cg = [], []
        for s in seeds:
            idx_t = self.idx(s, BOOT_BLOCK, len(A))
            for k in range(idx_t.shape[1]):
                smp = A[idx_t[:, k]]
                xa, xb = smp[:, 0] - smp[:, 2], smp[:, 1] - smp[:, 2]
                gaps_xs.append(xa.mean() / xa.std(ddof=1) - xb.mean() / xb.std(ddof=1))
                gaps_cg.append(np.log1p(smp[:, 0]).sum() - np.log1p(smp[:, 1]).sum())
        gx = np.array(gaps_xs) * np.sqrt(252)
        gc = np.expm1(np.array(gaps_cg) * 252 / len(A))        # annualised relative wealth gap, ~ CAGR gap
        pt = lambda v: [float(np.percentile(v, q)) for q in (5, 50, 95)]
        return {"share_xs": float((gx > 0).mean()), "share_cagr": float((gc > 0).mean()), "gap_xs_p5_50_95": pt(gx),
                "gap_cagr_p5_50_95": pt(gc), "paths": int(len(gx))}

    def frame_stats(self, n: str, frame_key: str) -> dict:
        r = self.r(n, frame_key)
        return stats(r, self.frames[frame_key][0][TBILL])

    def plus10(self, n: str) -> pd.Series:
        r0 = self.r(n)
        r5 = self.r(n, "s3_plus_5bps").reindex(r0.index)
        return r0 + 2.0 * (r5 - r0)

    def halves(self, n: str) -> tuple[float, float]:
        r = self.r(n)
        rf = self.rf.reindex(r.index)
        a, b = r.index <= CUT, r.index > CUT
        return xsharpe(r[a].to_numpy(), rf[a].to_numpy()), xsharpe(r[b].to_numpy(), rf[b].to_numpy())

    def challenge(self, ch: str, de: str, breach_key: str = "p20", rung: str | None = "GROWTH") -> dict:
        """SPEC 4: checks 0-6 of challenger `ch` against default `de` (excess Sharpe is the paired metric)."""
        pr = self.paired(self.r(ch), self.r(de))
        ex_c, ex_d = self.frame_stats(ch, "s6_exact"), self.frame_stats(de, "s6_exact")
        f5_c, f5_d = self.frame_stats(ch, "s3_plus_5bps"), self.frame_stats(de, "s3_plus_5bps")
        h_c, h_d = self.halves(ch), self.halves(de)
        t_c, t_d = self.tails(ch)[breach_key], self.tails(de)[breach_key]
        rung_pass = None if rung is None else self.rung_ok(ch, rung)
        checks = {"c0_rung": True if rung is None or ch in RUNG_EXEMPT else bool(rung_pass),
                  "c1_share_ge_80": pr["share_xs"] >= 0.80, "c2_breach_no_worse": t_c <= t_d,
                  "c3_exact_xs_higher": ex_c["xs"] > ex_d["xs"], "c4_plus5_xs_not_lower": f5_c["xs"] >= f5_d["xs"],
                  "c5_h1_higher": h_c[0] > h_d[0], "c6_h2_higher": h_c[1] > h_d[1]}
        rc, rd_ = self.r(ch), self.r(de)
        blocks = {k: xsharpe(lib.window(rc, lo, hi).to_numpy(), self.rf.reindex(lib.window(rc, lo, hi).index).to_numpy())
                  - xsharpe(lib.window(rd_, lo, hi).to_numpy(), self.rf.reindex(lib.window(rd_, lo, hi).index).to_numpy())
                  for k, (lo, hi) in BLOCK_DICT.items() if k in ("A", "B", "C")}
        return {"challenger": ch, "default": de, **pr, "borderline": bool(0.75 <= pr["share_xs"] <= 0.85),
                "rung_pass": rung_pass, "breach": [t_c, t_d], "breach_key": breach_key,
                "exact_xs": [ex_c["xs"], ex_d["xs"]], "plus5_xs": [f5_c["xs"], f5_d["xs"]], "halves_xs": [list(h_c), list(h_d)],
                "xs": [self.cands[ch]["q"]["xs"], self.cands[de]["q"]["xs"]], "cagr": [self.cands[ch]["q"]["cagr"], self.cands[de]["q"]["cagr"]],
                "xs_monthly": [self.cands[ch]["q"]["xs_monthly"], self.cands[de]["q"]["xs_monthly"]],
                "block_gap_xs": blocks, "checks": {k: bool(v) for k, v in checks.items()},
                "passed": bool(all(checks.values())),
                "passed_without_c2": bool(all(v for k, v in checks.items() if k != "c2_breach_no_worse"))}

    @staticmethod
    def reg_t(w: dict) -> float:
        """Worst-case Reg-T initial requirement as a share of equity: 3x-ETF pods 75%, every other pod 50%."""
        return float(sum(v * (0.75 if k in ("taa3x", "taa3x_1n") else 0.50) for k, v in w.items()
                         if k != TBILL and not k.startswith(DEBT)))
