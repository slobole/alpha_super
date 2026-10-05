"""Inputs, book model, objectives and statistics for the shelf rebuild (SPEC_FROZEN.md).

Conventions (as fund_menu_20260923/common.py and QUANT_PHILOSOPHY.md):
- daily simple returns r_t = V_t / V_(t-1) - 1; Sharpe annualised with 252 sessions and a zero risk-free rate;
- CAGR in calendar time: (V_end / V_base) ** (365.25 / days) - 1, V_base = the close before the first return;
- drawdown_t = V_t / max(V_1..V_t) - 1;
- T-bills = BIL TOTALRETURN (SPEC 2.3); "excess" always means over that series on the same dates.

Book model (SPEC 4): pods compound independently; at a reset the pod values are set to the target weights. Within
a reset period that starts at book value V0,

    V_t = V0 * sum_i w_i * G_i,t,    G_i,t = prod_{u in period, u <= t} (1 + r_i,u)

which is exactly `common.book_return_ser` (checked in tests). The reset happens after the last close of a period,
so the first return that uses new weights is the first session of the next period.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FUND_MENU_DIR = REPO / "scripts" / "research" / "fund_menu_20260923"
for _path in (REPO, FUND_MENU_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import common  # noqa: E402  (fund menu helpers: metric_dict, book_return_ser, windows, DTB3 accrual)
import evaluation  # noqa: E402  (stationary bootstrap indices, +bps cost drag)

STUDY = REPO / "results" / "research" / "portfolio" / "shelf_rebuild_20260929"
SOURCE = STUDY / "sources"
PROXY = STUDY / "proxy_runs"
LEGACY_INVENTORY = REPO / "results" / "research" / "portfolio" / "fund_product_menu_20260923" / "inventory"
ETF_RESEARCH_DIR = REPO / "results" / "research" / "dv2_deep_20260925" / "sources"

END = pd.Timestamp("2026-08-19")
LONG_START = pd.Timestamp("2008-03-04")
EXACT_START = pd.Timestamp("2012-10-02")
BLOCK_DICT = {"A": (pd.Timestamp("2008-03-04"), pd.Timestamp("2012-10-01")),
              "B": (pd.Timestamp("2012-10-02"), pd.Timestamp("2021-12-31")),
              "C": (pd.Timestamp("2022-01-03"), pd.Timestamp("2026-08-19")),
              "RECENT": (pd.Timestamp("2023-08-21"), pd.Timestamp("2026-08-19"))}
CRISIS_DICT = {"gfc": ("2008-05-19", "2009-03-09"), "q4_2018": ("2018-09-20", "2018-12-24"),
               "covid": ("2020-02-19", "2020-03-23"), "bear_2022": ("2022-01-03", "2022-10-12"),
               "tariffs_2025": ("2025-02-19", "2025-04-08")}
PROXY_ALIAS_LIST = ["taa3x", "taa3x_1n", "taa2x_1n", "btal_qqq"]
TBILL = "tbill"
BOOT_REPS, BOOT_BLOCK, BOOT_SEED = 2000, 63.0, 20260929
TIE_SHARE = 0.90

# Ease-of-operation facts per sleeve (SPEC 10), from the module docs and the 2026-09-28 readiness audit.
# route: how orders go to market today; daily: a pod that can trade any session; moc/short/levered/fred: needs.
OPS_DICT = {
    "core5": dict(route="monthly MOO", instruments="ETFs", daily=False, moc=False, short=False, levered=False, fred=False),
    "btal_qqq": dict(route="monthly MOO", instruments="ETFs incl. BTAL", daily=False, moc=False, short=False, levered=False, fred=False),
    "tactical_fi": dict(route="monthly MOO", instruments="IEF/LQD/BIL", daily=False, moc=False, short=False, levered=False, fred=True),
    "trinity": dict(route="daily band check, MOO", instruments="VTI/GLD/TLT/BIL", daily=True, moc=False, short=False, levered=False, fred=False),
    "eom_flow": dict(route="month-end MOC", instruments="SPY/TLT", daily=False, moc=True, short=True, levered=False, fred=False),
    "downshock": dict(route="daily MOO", instruments="sector ETFs", daily=True, moc=False, short=False, levered=False, fred=False),
    "disp": dict(route="daily MOO", instruments="sector ETFs", daily=True, moc=False, short=False, levered=False, fred=False),
    "taa3x": dict(route="monthly MOO", instruments="ETFs incl. TQQQ, BTAL", daily=False, moc=False, short=False, levered=True, fred=False),
    "taa3x_1n": dict(route="monthly MOO", instruments="ETFs incl. TQQQ, BTAL", daily=False, moc=False, short=False, levered=True, fred=False),
    "taa2x_1n": dict(route="monthly MOO", instruments="ETFs incl. QLD, BTAL", daily=False, moc=False, short=False, levered=True, fred=False),
    "ndx_vxn": dict(route="monthly MOO", instruments="~10 Nasdaq-100 stocks", daily=False, moc=False, short=False, levered=False, fred=False),
    "ndx_atr": dict(route="monthly MOO", instruments="~10 Nasdaq-100 stocks", daily=False, moc=False, short=False, levered=False, fred=False),
    "ndx_natr20": dict(route="monthly MOO", instruments="~10 Nasdaq-100 stocks", daily=False, moc=False, short=False, levered=False, fred=False),
    "compass_qqq": dict(route="monthly MOO", instruments="sector ETFs + QQQ", daily=False, moc=False, short=False, levered=False, fred=True),
    "compass": dict(route="monthly MOO", instruments="sector ETFs", daily=False, moc=False, short=False, levered=False, fred=True),
    "dv2": dict(route="daily MOO", instruments="S&P 500 stocks", daily=True, moc=False, short=False, levered=False, fred=False),
    "dv2_adv": dict(route="daily MOO", instruments="S&P 500 stocks", daily=True, moc=False, short=False, levered=False, fred=False),
    "dv2_floor": dict(route="daily MOO", instruments="S&P 500 stocks", daily=True, moc=False, short=False, levered=False, fred=False),
    "hpi_vote": dict(route="daily MOO (margin)", instruments="S&P 500 stocks", daily=True, moc=False, short=False, levered=False, fred=False),
    "hpi_ibs_rsi": dict(route="daily MOO (margin)", instruments="S&P 500 stocks", daily=True, moc=False, short=False, levered=False, fred=False),
    "etf_dv2": dict(route="daily MOO", instruments="industry ETFs", daily=True, moc=False, short=False, levered=False, fred=False),
    "tbill": dict(route="hold BIL", instruments="BIL", daily=False, moc=False, short=False, levered=False, fred=False),
    # Inventory-only sleeves (they appear in existing portfolio YAMLs evaluated as references).
    "taa_lin_qqq": dict(route="monthly MOO", instruments="ETFs", daily=False, moc=False, short=False, levered=False, fred=False),
    "taa_1n_qld": dict(route="monthly MOO", instruments="ETFs incl. QLD", daily=False, moc=False, short=False, levered=True, fred=False),
    "taa_1n_sso": dict(route="monthly MOO", instruments="ETFs incl. SSO", daily=False, moc=False, short=False, levered=True, fred=False),
    "disp_xlc": dict(route="daily MOO", instruments="sector ETFs", daily=True, moc=False, short=False, levered=False, fred=False),
    "disp_xlc_sma": dict(route="daily MOO", instruments="sector ETFs", daily=True, moc=False, short=False, levered=False, fred=False),
}


def ledger(event_str: str, **fields) -> None:
    """Append an event with the current spec hash to the study's experiment ledger."""
    import hashlib
    from datetime import datetime, timezone

    record = {"event_str": event_str, "recorded_at_utc_str": datetime.now(timezone.utc).isoformat(),
              "spec_sha256_str": hashlib.sha256((HERE / "SPEC_FROZEN.md").read_bytes()).hexdigest(), **fields}
    with (STUDY / "experiment_ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


# ─── inputs ──────────────────────────────────────────────────────────────────


def load_metadata() -> dict[str, dict]:
    out = {}
    for path in sorted(SOURCE.glob("*__metadata.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        out[meta["alias_str"]] = meta
    return out


def nav_to_returns(path_df: pd.DataFrame) -> pd.Series:
    """Returns from the day before the first invested day (NaN before), as common.sleeve_nav_df."""
    nav = path_df["total_value_float"].astype(float)
    invested = path_df["portfolio_value_float"].abs() > 1e-9
    first = nav.index.get_loc(invested[invested].index[0])
    return nav.iloc[max(first - 1, 0):].pct_change(fill_method=None).iloc[1:]


def read_path(folder: Path, alias: str) -> pd.DataFrame:
    return pd.read_csv(folder / f"{alias}__path.csv.gz", index_col="date", parse_dates=True)


def read_tx(folder: Path, alias: str) -> pd.DataFrame:
    return pd.read_csv(folder / f"{alias}__transactions.csv.gz", parse_dates=["date"])


def cash_realism_add(path_df: pd.DataFrame, dtb3_rate: pd.Series) -> pd.Series:
    """SPEC 9 cash realism: positive idle cash earns max(DTB3 - 0.5%, 0), negative cash pays DTB3 + 1.5%.

    add_t = (max(cash_(t-1), 0) * max(y_t - 0.005, 0) - max(-cash_(t-1), 0) * (y_t + 0.015)) * days_t / 360 / NAV_(t-1)
    *** CRITICAL*** cash and NAV are the prior close's; y_t is the DTB3 observation dated before session t.
    """
    cash = path_df["cash_float"].shift(1)
    nav = path_df["total_value_float"].shift(1)
    idx = path_df.index
    days = pd.Series(idx, index=idx).diff().dt.days
    y = dtb3_rate.reindex(idx)
    earn = cash.clip(lower=0.0) * (y - 0.005).clip(lower=0.0)
    pay = (-cash).clip(lower=0.0) * (y + 0.015)
    return ((earn - pay) * days / 360.0 / nav).fillna(0.0)


def dtb3_annual_rate(index: pd.DatetimeIndex) -> pd.Series:
    """DTB3 as a decimal annual rate, prior observation only (*** CRITICAL*** shift(1): no same-day rate)."""
    frame = pd.read_csv(common.DTB3_CSV_PATH, parse_dates=["observation_date"], na_values=["."])
    rate = frame.set_index("observation_date")["DTB3"].astype(float).dropna().sort_index() / 100.0
    return rate.reindex(rate.index.union(index)).ffill().shift(1).reindex(index)


def load_inputs() -> dict:
    """Every frame the parts need. Frames share one session index; NaN before a sleeve's first return."""
    meta = load_metadata()
    paths = {a: read_path(SOURCE, a) for a in meta}
    sleeve = pd.DataFrame({a: nav_to_returns(p) for a, p in paths.items()}).sort_index().loc[:END]
    index = sleeve.index
    bil = common.load_total_return_close_ser("BIL", "2007-01-01", END.strftime("%Y-%m-%d"))
    sleeve[TBILL] = bil.reindex(index).pct_change(fill_method=None)
    bench = common.build_benchmark_return_df(index, index[0].date().isoformat(), END.date().isoformat())
    agg = common.load_total_return_close_ser("AGG", "2003-01-01", END.strftime("%Y-%m-%d"))
    bench["AGG"] = agg.reindex(index).pct_change(fill_method=None)
    bench["BIL"] = sleeve[TBILL]
    bench = bench.rename(columns={"TBILL": "DTB3"})

    def with_proxy(mode: str) -> pd.DataFrame:
        frame = sleeve.copy()
        for alias in PROXY_ALIAS_LIST:
            proxy = nav_to_returns(read_path(PROXY / mode, alias)).reindex(index)
            early = index < EXACT_START
            # *** CRITICAL*** the proxy fills only the dates before the real sleeve's LONG use starts (2012-10-02);
            # real returns from that date on stay untouched.
            frame.loc[early, alias] = proxy[early]
        return frame

    # Amendment A2: etf_dv2's engine run has no bars before 2009 and is idle until 2010-01-13 (module history
    # start), so before its first return the LONG frames use the DV2 deep study's research run of the same rules.
    etf_research = pd.read_csv(ETF_RESEARCH_DIR / "etf_ind_adv50__path.csv.gz", index_col="date",
                               parse_dates=True)["total_value_float"]
    etf_first = sleeve["etf_dv2"].first_valid_index()
    etf_fill_mask = (index >= LONG_START) & (index < etf_first)
    etf_fill = etf_research.pct_change(fill_method=None).reindex(index)

    def with_etf_fill(frame: pd.DataFrame, fill: pd.Series) -> pd.DataFrame:
        out = frame.copy()
        # *** CRITICAL*** only dates where the engine run has no return; the engine's returns stay untouched.
        out.loc[etf_fill_mask, "etf_dv2"] = fill[etf_fill_mask]
        return out

    long_scaled = with_etf_fill(with_proxy("splice_scaled"), etf_fill)
    long_unscaled = with_etf_fill(with_proxy("splice_unscaled"), etf_fill)
    long_etf_cash = with_etf_fill(with_proxy("splice_scaled"), pd.Series(0.0, index=index))

    tx = {a: read_tx(SOURCE, a) for a in meta}
    nav = {a: p["total_value_float"] for a, p in paths.items()}
    proxy_tx = {a: read_tx(PROXY / "splice_scaled", a) for a in PROXY_ALIAS_LIST}
    proxy_nav = {a: read_path(PROXY / "splice_scaled", a)["total_value_float"] for a in PROXY_ALIAS_LIST}
    etf_research_tx = pd.read_csv(ETF_RESEARCH_DIR / "etf_ind_adv50__transactions.csv.gz", parse_dates=["date"])

    def stressed(frame: pd.DataFrame) -> pd.DataFrame:
        """+5 bps per side on every traded dollar (SPEC 9), from each run's own fills and prior-day NAV."""
        out = frame.copy()
        for alias in meta:
            drag = evaluation.extra_slippage_cost_ser(tx[alias], nav[alias], 0.0005).reindex(index).fillna(0.0)
            live = out[alias].notna()
            out.loc[live, alias] = out.loc[live, alias] - drag[live]
        for alias in PROXY_ALIAS_LIST:
            drag = evaluation.extra_slippage_cost_ser(proxy_tx[alias], proxy_nav[alias], 0.0005).reindex(index).fillna(0.0)
            early = (index < EXACT_START) & out[alias].notna().to_numpy()
            out.loc[early, alias] = frame.loc[early, alias] - drag[early]
        drag = evaluation.extra_slippage_cost_ser(etf_research_tx, etf_research, 0.0005).reindex(index).fillna(0.0)
        fill_dates = etf_fill_mask & out["etf_dv2"].notna().to_numpy()
        out.loc[fill_dates, "etf_dv2"] = frame.loc[fill_dates, "etf_dv2"] - drag[fill_dates]
        return out

    rate = dtb3_annual_rate(index)

    def cash_real(frame: pd.DataFrame, is_long: bool) -> pd.DataFrame:
        out = frame.copy()
        for alias in meta:
            add = cash_realism_add(paths[alias], rate).reindex(index).fillna(0.0)
            if alias == "etf_dv2":
                add[etf_fill_mask] = 0.0  # A2: the research-run dates carry no cash column, so no add there
            live = out[alias].notna()
            out.loc[live, alias] = out.loc[live, alias] + add[live]
        if is_long:
            for alias in PROXY_ALIAS_LIST:
                add = cash_realism_add(read_path(PROXY / "splice_scaled", alias), rate).reindex(index).fillna(0.0)
                early = (index < EXACT_START) & out[alias].notna().to_numpy()
                out.loc[early, alias] = frame.loc[early, alias] + add[early]
        return out

    tfi_frozen = sleeve["tactical_fi_frozen"] if "tactical_fi_frozen" in sleeve else None
    return {"meta": meta, "sleeve": sleeve, "long": long_scaled, "long_unscaled": long_unscaled,
            "long_etf_cash": long_etf_cash, "bench": bench,
            "stressed_long": stressed(long_scaled), "stressed_exact": stressed(sleeve),
            "cash_long": cash_real(long_scaled, True), "cash_exact": cash_real(sleeve, False),
            "tx": tx, "nav": nav, "tfi_frozen": tfi_frozen, "index": index}


def tier_of(alias: str, meta: dict) -> str:
    if alias == TBILL:
        return "cash"
    return meta[alias]["tier_str"]


# ─── book model ──────────────────────────────────────────────────────────────


@dataclass
class Book:
    name: str
    pods: tuple[str, ...]
    rule: str = "EQ"                      # "EQ" (fixed targets) or "IV" (inverse vol at each reset)
    weights: dict[str, float] = field(default_factory=dict)   # EQ targets; empty = equal capital
    policy: str = "annual"                # "annual" or "none" (drift)
    family: str = ""
    tags: dict = field(default_factory=dict)

    def targets(self) -> dict[str, float]:
        if self.weights:
            return dict(self.weights)
        return {p: 1.0 / len(self.pods) for p in self.pods}


def period_ids(index: pd.DatetimeIndex, policy: str) -> np.ndarray:
    """Reset-period label per session: a new period starts at the first session of each calendar year."""
    if policy == "annual":
        return index.year.to_numpy()
    if policy == "none":
        return np.zeros(len(index), dtype=int)
    raise ValueError(f"unsupported policy {policy!r}")


def iv_weights(history: pd.DataFrame, pods: tuple[str, ...]) -> np.ndarray | None:
    """Inverse-volatility targets from the trailing 252 sessions (>= 60 valid returns per pod), else None."""
    tail = history[list(pods)].iloc[-252:]
    sigma = tail.std()
    if (tail.notna().sum() < 60).any() or (sigma <= 0).any() or sigma.isna().any():
        return None
    inv = 1.0 / sigma.to_numpy()
    return inv / inv.sum()


def book_returns(frame: pd.DataFrame, book: Book, start: pd.Timestamp, end: pd.Timestamp = END,
                 weight_source: pd.DataFrame | None = None, weight_log: list | None = None) -> pd.Series:
    """Daily book returns on [start, end] under the pod model (SPEC 4).

    For IV books the weights at each reset come from `weight_source` (default `frame`) using only returns up to
    and including the reset close. *** CRITICAL*** the weights for period p use history strictly before period p's
    first session, i.e. through the previous period's last close.
    """
    cols = list(book.pods)
    window = frame.loc[start:end, cols]
    if window.isna().any().any():
        missing = window.isna().sum()
        raise ValueError(f"{book.name}: missing returns in window {missing[missing > 0].to_dict()}")
    index = window.index
    ret = window.to_numpy(dtype=float)
    periods = period_ids(index, book.policy)
    source = frame if weight_source is None else weight_source
    target = book.targets()
    fixed_w = np.array([target[p] for p in cols], dtype=float)
    if book.rule == "EQ" and abs(fixed_w.sum() - 1.0) > 1e-9:
        raise ValueError(f"{book.name}: weights sum to {fixed_w.sum()}")
    out = np.empty(len(index))
    value = 1.0
    boundaries = np.flatnonzero(np.r_[True, periods[1:] != periods[:-1]])
    boundaries = np.r_[boundaries, len(index)]
    for k in range(len(boundaries) - 1):
        a, b = boundaries[k], boundaries[k + 1]
        if book.rule == "IV":
            history = source.loc[:index[a], cols].iloc[:-1]  # *** CRITICAL*** excludes the period's first session
            w = iv_weights(history, book.pods)
            if w is None:
                w = np.full(len(cols), 1.0 / len(cols))
        else:
            w = fixed_w
        if weight_log is not None:
            weight_log.append((index[a], b - a, dict(zip(cols, w))))
        growth = np.cumprod(1.0 + ret[a:b], axis=0)            # G_i,t within the period
        level = value * (growth @ w)                            # V_t
        prev = np.r_[value, level[:-1]]
        out[a:b] = level / prev - 1.0
        value = level[-1]
    return pd.Series(out, index=index, name=book.name)


def replace_pod(frame: pd.DataFrame, pod: str) -> pd.DataFrame:
    """The slot test's frame: the pod's returns become T-bill returns (weights and IV weights unchanged)."""
    out = frame.copy()
    out[pod] = frame[TBILL]
    return out


def dilute(book_r: pd.Series, tbill_r: pd.Series, s_grid: np.ndarray) -> np.ndarray:
    """NAV paths (len(s_grid) x N) of s * book + (1 - s) * T-bills with annual reset at the pod level.

    Both the book and the T-bill pod restart at their target shares after each year's last close, so within year y
    the mix is V_(y-1) * (s * B_y,t + (1 - s) * T_y,t), with B and T the growth since the reset.
    """
    index = book_r.index
    periods = period_ids(index, "annual")
    b = book_r.to_numpy(dtype=float)
    t = tbill_r.reindex(index).to_numpy(dtype=float)
    nav = np.empty((len(s_grid), len(index)))
    value = np.ones(len(s_grid))
    boundaries = np.r_[np.flatnonzero(np.r_[True, periods[1:] != periods[:-1]]), len(index)]
    for k in range(len(boundaries) - 1):
        lo, hi = boundaries[k], boundaries[k + 1]
        gb = np.cumprod(1.0 + b[lo:hi])
        gt = np.cumprod(1.0 + t[lo:hi])
        nav[:, lo:hi] = value[:, None] * (s_grid[:, None] * gb[None, :] + (1.0 - s_grid[:, None]) * gt[None, :])
        value = nav[:, hi - 1]
    return nav


# ─── metrics and objectives ──────────────────────────────────────────────────


def base_date(index_all: pd.DatetimeIndex, r: pd.Series) -> pd.Timestamp:
    """The close before the first return (the NAV base for calendar-time CAGR)."""
    position = index_all.get_loc(r.index[0])
    if position == 0:
        raise ValueError("A return series cannot start on the first session of the calendar (no NAV base).")
    return index_all[position - 1]


def cagr(r: pd.Series, base_ts: pd.Timestamp) -> float:
    growth = float(np.prod(1.0 + r.to_numpy(dtype=float)))
    return growth ** (365.25 / (r.index[-1] - base_ts).days) - 1.0


def maxdd(r: pd.Series | np.ndarray) -> float:
    v = np.cumprod(1.0 + np.asarray(r, dtype=float))
    v = np.r_[1.0, v]
    return float((v / np.maximum.accumulate(v) - 1.0).min())


def window(r: pd.Series, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.Series:
    return r.loc[(r.index >= lo) & (r.index <= hi)]


def excess_cagr(r: pd.Series, tb: pd.Series, index_all: pd.DatetimeIndex) -> float:
    base = base_date(index_all, r)
    return cagr(r, base) - cagr(tb.reindex(r.index), base)


def excess_calmar(r: pd.Series, tb: pd.Series, index_all: pd.DatetimeIndex) -> float:
    return excess_cagr(r, tb, index_all) / abs(maxdd(r))


S_GRID = np.round(np.arange(1.0, -0.0001, -0.01), 2)


def cagr_at_budget(r: pd.Series, tb: pd.Series, index_all: pd.DatetimeIndex, budget: float) -> tuple[float, float]:
    """(objective, s): CAGR of the largest-s mix with max drawdown >= budget (s on 1.00, 0.99, ..., 0.00)."""
    nav = dilute(r, tb, S_GRID)
    base = base_date(index_all, r)
    days = (r.index[-1] - base).days
    peak = np.maximum.accumulate(np.c_[np.ones(len(S_GRID)), nav], axis=1)
    dd = (np.c_[np.ones(len(S_GRID)), nav] / peak - 1.0).min(axis=1)
    feasible = np.flatnonzero(dd >= budget)
    k = feasible[0]  # S_GRID is descending, so the first feasible is the largest s
    return float(nav[k, -1] ** (365.25 / days) - 1.0), float(S_GRID[k])


def crisis_corr(r: pd.Series, spx: pd.Series) -> float:
    """Correlation with the S&P 500 on its worst 5% of days in the window."""
    s = spx.reindex(r.index)
    mask = s <= s.quantile(0.05)
    return float(r[mask].corr(s[mask]))


def cofall_windows(bench: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp, n: int = 6, length: int = 21) -> list[tuple]:
    """SPEC 3: the n worst non-overlapping 21-session windows where $SPXTR and AGG both lost, ranked by 60/40."""
    b = bench.loc[lo:hi, ["SPXTR", "AGG", "SIXTY_FORTY"]].dropna()
    grow = (1.0 + b).rolling(length).apply(np.prod, raw=True) - 1.0
    cand = grow[(grow["SPXTR"] < 0) & (grow["AGG"] < 0)].sort_values("SIXTY_FORTY")
    chosen: list[tuple] = []
    positions = {ts: i for i, ts in enumerate(b.index)}
    for end_ts, row in cand.iterrows():
        end_pos = positions[end_ts]
        start_pos = end_pos - length + 1
        if any(not (end_pos < s or start_pos > e) for s, e, *_ in chosen):
            continue
        chosen.append((start_pos, end_pos, b.index[start_pos - 1], end_ts, float(row["SIXTY_FORTY"])))
        if len(chosen) == n:
            break
    return [(start_ts, end_ts, ret) for _, _, start_ts, end_ts, ret in chosen]


def full_metrics(r: pd.Series, data: dict, prefix: str) -> dict:
    """SPEC 5 metrics for one window's book returns."""
    index_all = data["index"]
    bench = data["bench"]
    base = base_date(index_all, r)
    m = common.metric_dict(r, bench["SPXTR"], bench["BIL"], base)
    out = {"cagr": m["cagr_float"], "vol": m["volatility_float"], "sharpe": m["sharpe_rf0_float"],
           "sharpe_excess": m["sharpe_excess_float"], "maxdd": m["max_drawdown_float"],
           "dd_peak": m["max_dd_peak_date_str"], "dd_trough": m["max_dd_trough_date_str"],
           "dd_recovery": m["max_dd_recovery_date_str"], "underwater_days": m["longest_underwater_days_int"],
           "calmar": m["cagr_float"] / abs(m["max_drawdown_float"]),
           "excess_calmar": (m["cagr_float"] - m["tbill_cagr_float"]) / abs(m["max_drawdown_float"]),
           "tbill_cagr": m["tbill_cagr_float"], "cvar5_daily": m["es95_daily_float"], "cvar5_21d": m["es95_21d_float"],
           "worst_month": m["worst_month_float"], "worst_year": m["worst_year_float"],
           "worst_12m": float((common.nav_from_return_ser(r) / common.nav_from_return_ser(r).shift(252) - 1).min()),
           "pos_months": m["positive_month_share_float"], "beta": m["beta_spx_float"],
           "corr_spx": m["corr_spx_daily_float"], "crisis_corr": crisis_corr(r, bench["SPXTR"])}
    return {f"{prefix}_{k}": v for k, v in out.items()}


# ─── bootstrap and PBO ───────────────────────────────────────────────────────


def boot_index(n: int) -> np.ndarray:
    return evaluation.stationary_bootstrap_index_mat(n, BOOT_REPS, BOOT_BLOCK, BOOT_SEED)


def path_stats(sample: np.ndarray, periods_per_year: float = 252.0) -> tuple[np.ndarray, np.ndarray]:
    """(CAGR by length, max drawdown) per column of a (N x B) return sample."""
    nav = np.cumprod(1.0 + sample, axis=0)
    growth = nav[-1]
    cagr_arr = growth ** (periods_per_year / sample.shape[0]) - 1.0
    nav1 = np.vstack([np.ones((1, sample.shape[1])), nav])
    dd = (nav1 / np.maximum.accumulate(nav1, axis=0) - 1.0).min(axis=0)
    return cagr_arr, dd


def objective_from_stats(kind: str, cagr_arr: np.ndarray, dd: np.ndarray, tb_cagr: float, budget: float) -> np.ndarray:
    if kind == "excess_calmar":
        return (cagr_arr - tb_cagr) / np.abs(np.minimum(dd, -1e-9))
    if kind == "cagr_at_budget":
        s = np.minimum(1.0, abs(budget) / np.abs(np.minimum(dd, -1e-9)))
        return s * cagr_arr + (1.0 - s) * tb_cagr
    raise ValueError(kind)


def bootstrap_objective(R: np.ndarray, tb: np.ndarray, kind: str, budget: float = -0.20) -> np.ndarray:
    """Objective per bootstrap path (reps x B) from paired resampling of the rows of R (N x B) and tb (N)."""
    idx = boot_index(R.shape[0])
    out = np.empty((idx.shape[0], R.shape[1]))
    for k in range(idx.shape[0]):
        sample = R[idx[k]]
        c, d = path_stats(sample)
        tbc = float(np.prod(1.0 + tb[idx[k]]) ** (252.0 / len(tb)) - 1.0)
        out[k] = objective_from_stats(kind, c, d, tbc, budget)
    return out


def pbo_cscv(R: np.ndarray, tb: np.ndarray, kind: str, budget: float = -0.20, blocks: int = 16) -> dict:
    """Probability of backtest overfitting by combinatorially symmetric cross-validation (SPEC 9)."""
    n = R.shape[0] - R.shape[0] % blocks
    R, tb = R[:n], tb[:n]
    block_idx = np.array_split(np.arange(n), blocks)
    logits, oos_rank = [], []
    for combo in combinations(range(blocks), blocks // 2):
        is_idx = np.concatenate([block_idx[i] for i in combo])
        oos_idx = np.concatenate([block_idx[i] for i in range(blocks) if i not in combo])
        vals = []
        for sel in (is_idx, oos_idx):
            c, d = path_stats(R[sel])
            tbc = float(np.prod(1.0 + tb[sel]) ** (252.0 / len(sel)) - 1.0)
            vals.append(objective_from_stats(kind, c, d, tbc, budget))
        best = int(np.argmax(vals[0]))
        # Relative OOS rank of the IS-best book: rank = 1 + books below it (ties count half), omega = rank / (B + 1).
        below = np.sum(vals[1] < vals[1][best]) + 0.5 * (np.sum(vals[1] == vals[1][best]) - 1)
        omega = (below + 1.0) / (R.shape[1] + 1)
        oos_rank.append(omega)
        logits.append(np.log(omega / (1.0 - omega)))
    logits = np.array(logits)
    return {"pbo": float(np.mean(logits <= 0.0)), "median_oos_rank": float(np.median(oos_rank)),
            "splits": int(len(logits))}


# ─── ease of operation ───────────────────────────────────────────────────────


def trade_dates(data: dict, alias: str, lo: pd.Timestamp = EXACT_START, hi: pd.Timestamp = END) -> set:
    if alias == TBILL:
        return set()
    d = data["tx"][alias]["date"]
    return set(d[(d >= lo) & (d <= hi)])


def average_weights(weight_log: list) -> dict[str, float]:
    """Session-weighted average of the targets used in each reset period (for IV books)."""
    total = sum(n for _, n, _ in weight_log)
    out: dict[str, float] = {}
    for _, n, w in weight_log:
        for pod, value in w.items():
            out[pod] = out.get(pod, 0.0) + value * n / total
    return out


def ops_fields(book: Book, data: dict, weights: dict[str, float] | None = None) -> dict:
    """Ease fields; tier shares use `weights` (the IV average for IV books) or the fixed targets."""
    target = weights if weights is not None else book.targets()
    meta = data["meta"]
    years = (END - EXACT_START).days / 365.25
    dates = set()
    for p in book.pods:
        dates |= trade_dates(data, p)
    tier_share = {"wired": 0.0, "pm-ready": 0.0, "shadow": 0.0, "cash": 0.0}
    for p in book.pods:
        tier_share[tier_of(p, meta)] += target[p]
    return {"pods": len(book.pods), "trade_days_per_year": len(dates) / years,
            "wired_share": tier_share["wired"], "pm_ready_share": tier_share["pm-ready"],
            "shadow_share": tier_share["shadow"], "cash_share": tier_share["cash"],
            "any_daily": any(OPS_DICT[p]["daily"] for p in book.pods),
            "needs_moc": any(OPS_DICT[p]["moc"] for p in book.pods),
            "needs_short": any(OPS_DICT[p]["short"] for p in book.pods),
            "levered_etf": any(OPS_DICT[p]["levered"] for p in book.pods),
            "fred_dependency": any(OPS_DICT[p]["fred"] for p in book.pods)}
