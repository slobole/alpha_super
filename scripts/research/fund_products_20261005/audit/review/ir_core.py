"""Independent recomputation (reviewer lens). Raw files only: pandas / numpy / norgatedata.

Does NOT import g_lib, study, lib, ga_lib, common or evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

WT = Path(r"C:\Users\User\Documents\workspace\alpha_super\.claude\worktrees\nervous-colden-784bbf")
MAIN = Path(r"C:\Users\User\Documents\workspace\alpha_super")
OLD_SRC = MAIN / "results" / "research" / "portfolio" / "shelf_rebuild_20260929" / "sources"
PROXY_SRC = MAIN / "results" / "research" / "portfolio" / "shelf_rebuild_20260929" / "proxy_runs" / "splice_scaled"
NEW_SRC = WT / "results" / "research" / "portfolio" / "fund_products_20261005" / "sources"
REPORT = WT / "results" / "research" / "portfolio" / "fund_products_20261005" / "report"
OUT = WT / "results" / "research" / "portfolio" / "fund_products_20261005" / "audit" / "review" / "independent_recompute"
DTB3 = MAIN.parent / "1_data" / "DTB3.csv"

LONG_START = pd.Timestamp("2008-03-04")
EXACT_START = pd.Timestamp("2012-10-02")
END = pd.Timestamp("2026-08-19")

PROXY_ALIASES = ("taa3x", "taa3x_1n", "btal_qqq")
OLD_ALIASES = ("ndx_vxn", "core5")
NEW_ALIASES = ("ndx_atr_cap", "ndx_natr_cap", "dv2_g", "hpi_g")

CRISES = {
    "gfc": ("2008-05-19", "2009-03-09"),
    "q4_2018": ("2018-09-20", "2018-12-24"),
    "covid": ("2020-02-19", "2020-03-23"),
    "bear_2022": ("2022-01-03", "2022-10-12"),
    "tariffs_2025": ("2025-02-19", "2025-04-08"),
}


def read_path(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index()
    return df.astype(float)


def read_tx(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p, parse_dates=["date"])
    return df


def dtb3_rate(index: pd.DatetimeIndex) -> pd.Series:
    """Annual DTB3 rate (decimal) of the last observation dated strictly before each session."""
    raw = pd.read_csv(DTB3, na_values=["."])
    raw["observation_date"] = pd.to_datetime(raw["observation_date"])
    s = raw.dropna().set_index("observation_date")["DTB3"].astype(float).sort_index() / 100.0
    obs_dates = s.index.values
    vals = s.values
    pos = np.searchsorted(obs_dates, index.values, side="left") - 1  # last obs with date < t
    out = np.where(pos >= 0, vals[np.clip(pos, 0, None)], np.nan)
    return pd.Series(out, index=index)


def pod_frames(path_df: pd.DataFrame, tx_df: pd.DataFrame) -> pd.DataFrame:
    """Per-session: raw return, fair-cash add, 5 bps drag. All on the path's own index."""
    nav = path_df["total_value_float"]
    cash = path_df["cash_float"]
    idx = path_df.index
    r = nav / nav.shift(1) - 1.0
    y = dtb3_rate(idx)
    days = pd.Series(idx, index=idx).diff().dt.days
    cash_prev = cash.shift(1)
    nav_prev = nav.shift(1)
    pos_cash = cash_prev.clip(lower=0.0)
    neg_cash = (-cash_prev).clip(lower=0.0)
    add = (pos_cash * (y - 0.005).clip(lower=0.0) - neg_cash * (y + 0.015)) * days / 360.0 / nav_prev
    traded = tx_df.assign(a=tx_df["signed_notional_float"].abs()).groupby("date")["a"].sum()
    drag = 0.0005 * traded.reindex(idx).fillna(0.0) / nav_prev
    return pd.DataFrame({"r": r, "add": add, "drag": drag})


def first_invested_pos(path_df: pd.DataFrame) -> int:
    inv = path_df["portfolio_value_float"].abs() > 0
    return int(np.argmax(inv.values))


def build_pods() -> dict[str, pd.DataFrame]:
    pods: dict[str, pd.DataFrame] = {}
    for a in OLD_ALIASES:
        p = read_path(OLD_SRC / f"{a}__path.csv.gz")
        t = read_tx(OLD_SRC / f"{a}__transactions.csv.gz")
        f = pod_frames(p, t)
        base = max(first_invested_pos(p) - 1, 0)
        f.iloc[: base + 1] = np.nan
        pods[a] = f
    for a in NEW_ALIASES:
        p = read_path(NEW_SRC / f"{a}__path.csv.gz")
        t = read_tx(NEW_SRC / f"{a}__transactions.csv.gz")
        f = pod_frames(p, t)
        base = max(first_invested_pos(p) - 1, 0)
        f.iloc[: base + 1] = np.nan
        pods[a] = f
    for a in PROXY_ALIASES:
        pr = read_path(PROXY_SRC / f"{a}__path.csv.gz")
        pt = read_tx(PROXY_SRC / f"{a}__transactions.csv.gz")
        rr = read_path(OLD_SRC / f"{a}__path.csv.gz")
        rt = read_tx(OLD_SRC / f"{a}__transactions.csv.gz")
        fp = pod_frames(pr, pt)
        fr = pod_frames(rr, rt)
        f = pd.concat([fp.loc[fp.index < EXACT_START], fr.loc[fr.index >= EXACT_START]]).sort_index()
        pods[a] = f
        pods[a + "__real"] = fr
        pods[a + "__proxyfull"] = fp
    return pods


def load_bil(index: pd.DatetimeIndex) -> pd.Series:
    import norgatedata as nd

    df = nd.price_timeseries(
        "BIL",
        stock_price_adjustment_setting=nd.StockPriceAdjustmentType.TOTALRETURN,
        padding_setting=nd.PaddingType.ALLMARKETDAYS,
        start_date="2007-01-01",
        end_date="2026-08-19",
        timeseriesformat="pandas-dataframe",
    )
    c = df["Close"].astype(float)
    c.index = pd.to_datetime(c.index).normalize()
    r = c.pct_change()
    return r.reindex(index)


def load_tr(symbol: str, index: pd.DatetimeIndex) -> pd.Series:
    import norgatedata as nd

    df = nd.price_timeseries(
        symbol,
        stock_price_adjustment_setting=nd.StockPriceAdjustmentType.TOTALRETURN,
        padding_setting=nd.PaddingType.ALLMARKETDAYS,
        start_date="2007-01-01",
        end_date="2026-08-19",
        timeseriesformat="pandas-dataframe",
    )
    c = df["Close"].astype(float)
    c.index = pd.to_datetime(c.index).normalize()
    return c.pct_change().reindex(index)


def book_returns(ret_df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Pods compound independently; reset to target weights at the first session of each calendar year."""
    cols = list(weights)
    w = np.array([weights[c] for c in cols], dtype=float)
    R = ret_df[cols].to_numpy(dtype=float)
    idx = ret_df.index
    years = idx.year.values
    out = np.empty(len(idx))
    pod_val = w.copy()
    total_prev = pod_val.sum()
    for i in range(len(idx)):
        if i == 0 or years[i] != years[i - 1]:
            pod_val = w * total_prev  # reset (transfer, no cost) before today's return
        pod_val = pod_val * (1.0 + R[i])
        total = pod_val.sum()
        out[i] = total / total_prev - 1.0
        total_prev = total
    return pd.Series(out, index=idx)


def maxdd(r: np.ndarray) -> float:
    nav = np.cumprod(1.0 + r)
    nav = np.concatenate([[1.0], nav])
    return float((nav / np.maximum.accumulate(nav) - 1.0).min())


def metrics(r: pd.Series, bil: pd.Series) -> dict:
    x = r.to_numpy(dtype=float)
    b = bil.reindex(r.index).to_numpy(dtype=float)
    n = len(x)
    wealth = float(np.prod(1.0 + x))
    ex = x - b
    return {
        "cagr": wealth ** (252.0 / n) - 1.0,
        "vol": float(np.std(x, ddof=1) * np.sqrt(252.0)),
        "xs": float(ex.mean() / np.std(ex, ddof=1) * np.sqrt(252.0)),
        "sharpe0": float(x.mean() / np.std(x, ddof=1) * np.sqrt(252.0)),
        "dd": maxdd(x),
        "n": n,
    }


def year_returns(r: pd.Series) -> dict[int, float]:
    return {int(y): float(np.prod(1.0 + g.values) - 1.0) for y, g in r.groupby(r.index.year)}


def crisis_returns(r: pd.Series) -> dict[str, dict[str, float]]:
    """Two conventions: incl = product of returns on start..end; excl = NAV(end)/NAV(start) - 1."""
    out = {}
    for k, (a, b) in CRISES.items():
        seg = r.loc[a:b]
        incl = float(np.prod(1.0 + seg.values) - 1.0)
        excl = float(np.prod(1.0 + seg.values[1:]) - 1.0)
        nav = np.cumprod(1.0 + seg.values)
        ddw = float((nav / np.maximum.accumulate(nav) - 1.0).min())
        out[k] = {"incl": incl, "excl": excl, "dd_in_window": ddw}
    return out


def sb_index(n: int, reps: int, block: float, seed: int) -> np.ndarray:
    """Own re-implementation of the Politis-Romano index draw, same RNG call order as the house function."""
    rng = np.random.default_rng(seed)
    p = 1.0 / block
    idx = np.empty((reps, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=reps)
    restart = rng.random((reps, n)) < p
    fresh = rng.integers(0, n, size=(reps, n))
    for j in range(1, n):
        cont = idx[:, j - 1] + 1
        cont[cont == n] = 0
        idx[:, j] = np.where(restart[:, j], fresh[:, j], cont)
    return idx


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    pods = build_pods()
    cal = read_path(OLD_SRC / "ndx_vxn__path.csv.gz").index
    idx = cal[(cal >= LONG_START) & (cal <= END)]
    main = pd.DataFrame({a: (f["r"] + f["add"]).reindex(idx) for a, f in pods.items()})
    plus5 = pd.DataFrame({a: (f["r"] + f["add"] - f["drag"]).reindex(idx) for a, f in pods.items()})
    house = pd.DataFrame({a: f["r"].reindex(idx) for a, f in pods.items()})
    bil = load_bil(idx)
    main["BIL"] = bil
    plus5["BIL"] = bil
    house["BIL"] = bil
    return main, plus5, bil, {"house": house, "pods": pods}


BOOKS = {
    "GR1": {"taa3x": 1 / 3, "ndx_atr_cap": 1 / 6, "ndx_natr_cap": 1 / 6, "dv2_g": 1 / 6, "hpi_g": 1 / 6},
    "GR2": {"taa3x_1n": 1 / 3, "ndx_atr_cap": 1 / 6, "ndx_natr_cap": 1 / 6, "dv2_g": 1 / 6, "hpi_g": 1 / 6},
    "GR3": {"taa3x_1n": 1 / 2, "ndx_atr_cap": 1 / 8, "ndx_natr_cap": 1 / 8, "dv2_g": 1 / 8, "hpi_g": 1 / 8},
    "S9 incumbent launch": {"taa3x_1n": 0.384, "ndx_vxn": 0.256, "core5": 0.18, "btal_qqq": 0.18},
    "T1 MOM -> BIL": {"taa3x": 1 / 3, "BIL": 1 / 3, "dv2_g": 1 / 6, "hpi_g": 1 / 6},
    "S1 no momentum": {"taa3x": 1 / 2, "dv2_g": 1 / 4, "hpi_g": 1 / 4},
}

if __name__ == "__main__":
    main, plus5, bil, extra = load_inputs()
    print(main.shape, main.index[0], main.index[-1])
    print(main.isna().sum())
    main.to_pickle(OUT / "main_frame.pkl")
    plus5.to_pickle(OUT / "plus5_frame.pkl")
    extra["house"].to_pickle(OUT / "house_frame.pkl")
