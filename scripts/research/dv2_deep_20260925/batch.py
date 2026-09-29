"""Parallel runner + statistics for the DV2 deep study (research-only).

Every run is stored once: summary row (CSV) and daily NAV (one parquet column per run id), so later phases reuse
results and the trial log counts every configuration actually evaluated.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replica as rp  # noqa: E402

OUT = Path(__file__).resolve().parents[3] / "results/research/dv2_deep_20260925"
RUNS = OUT / "runs"
_PANELS: dict = {}


def rule_id(label: str, rule: rp.Rule, start: str, end: str) -> str:
    blob = json.dumps({"label": label, "rule": asdict(rule), "start": start, "end": end}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _worker(args):
    label, rule, start, end, rid = args
    if label not in _PANELS:
        _PANELS.clear()
        _PANELS[label] = rp.Panel(label)
    p = _PANELS[label]
    res = rp.run(p, rule, start=start, end=end)
    summ = rp.summarize(res)
    nav = pd.Series(res.nav, index=res.dates, name=rid)
    trades = res.trades.assign(run_id=rid)
    return rid, summ, nav, trades


def run_many(label: str, rules: dict[str, rp.Rule], start="2000-01-03", end="2026-08-19", workers=8, tag="default", keep_trades=False) -> pd.DataFrame:
    """rules: {name: Rule}. Returns summary DataFrame indexed by name (cached on disk by rule id)."""
    RUNS.mkdir(parents=True, exist_ok=True)
    summ_path = RUNS / "summary.csv"
    done = pd.read_csv(summ_path, index_col="run_id") if summ_path.exists() else pd.DataFrame()
    todo, name_by_id = [], {}
    for name, rule in rules.items():
        rid = rule_id(label, rule, start, end)
        name_by_id.setdefault(rid, []).append(name)
        if rid not in done.index and rid not in [t[4] for t in todo]:
            todo.append((label, rule, start, end, rid))
    new_rows, navs, trade_frames = [], [], []
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for rid, summ, nav, trades in ex.map(_worker, todo, chunksize=1):
                rule = next(t[1] for t in todo if t[4] == rid)
                new_rows.append({"run_id": rid, "label": label, "start": start, "end": end, "tag": tag, **{f"r_{k}": v for k, v in asdict(rule).items()}, **summ})
                navs.append(nav)
                if keep_trades:
                    trade_frames.append(trades)
        new_df = pd.DataFrame(new_rows).set_index("run_id")
        done = pd.concat([done, new_df])
        done.to_csv(summ_path)
        nav_df = pd.concat(navs, axis=1)
        nav_df.columns = [str(c) for c in nav_df.columns]
        nav_df.to_parquet(RUNS / f"nav_{tag}_{len(list(RUNS.glob('nav_*.parquet')))}.parquet")
        if trade_frames:
            pd.concat(trade_frames).to_parquet(RUNS / f"trades_{tag}_{len(list(RUNS.glob('trades_*.parquet')))}.parquet")
    rows = []
    for name, rule in rules.items():
        rid = rule_id(label, rule, start, end)
        rows.append(pd.Series(done.loc[rid]).rename(name).to_frame().T.assign(run_id=rid))
    return pd.concat(rows)


def load_nav(run_ids) -> pd.DataFrame:
    frames = [pd.read_parquet(f) for f in RUNS.glob("nav_*.parquet")]
    allnav = pd.concat(frames, axis=1)
    allnav = allnav.loc[:, ~allnav.columns.duplicated()]
    return allnav[list(run_ids)]


def trial_count() -> int:
    s = pd.read_csv(RUNS / "summary.csv")
    return int(len(s))


# ----------------------------------------------------------------------------- statistics
def sharpe(r: np.ndarray) -> float:
    return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(252))


def paired_bootstrap(ra: pd.Series, rb: pd.Series, n=2000, block=20, seed=11) -> dict:
    """Moving-block bootstrap of Sharpe(a) - Sharpe(b) on aligned daily returns."""
    df = pd.concat([ra, rb], axis=1).dropna()
    a, b = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy()
    T = len(a)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, T - block, size=(n, T // block + 1))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n, -1)[:, :T]
    A, B = a[idx], b[idx]
    d = A.mean(1) / A.std(1, ddof=1) * np.sqrt(252) - B.mean(1) / B.std(1, ddof=1) * np.sqrt(252)
    return {"dsharpe": sharpe(a) - sharpe(b), "p_le_0": float((d <= 0).mean()), "q05": float(np.quantile(d, 0.05)),
            "q95": float(np.quantile(d, 0.95)), "corr": float(np.corrcoef(a, b)[0, 1])}


def deflated_sharpe(r: np.ndarray, trial_sharpes_annual: np.ndarray, n_trials: int) -> dict:
    """Bailey & Lopez de Prado (2014). All Sharpe values in per-day units inside the formula.

        SR0 = sqrt(V[SR_trials]) * ((1 - g) * Z^-1(1 - 1/N) + g * Z^-1(1 - 1/(N e)))
        DSR = Z( (SR - SR0) * sqrt(T - 1) / sqrt(1 - skew * SR + (kurt - 1) / 4 * SR^2) )
    """
    g = 0.5772156649
    r = np.asarray(r)
    r = r[np.isfinite(r)]
    T = len(r)
    sr = r.mean() / r.std(ddof=1)
    v = np.var(np.asarray(trial_sharpes_annual) / np.sqrt(252), ddof=1)
    sr0 = np.sqrt(v) * ((1 - g) * norm.ppf(1 - 1 / n_trials) + g * norm.ppf(1 - 1 / (n_trials * np.e)))
    skew = float(pd.Series(r).skew())
    kurt = float(pd.Series(r).kurt()) + 3.0
    z = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return {"dsr": float(norm.cdf(z)), "sr0_annual": float(sr0 * np.sqrt(252)), "n_trials": int(n_trials), "skew": skew, "kurt": kurt}
