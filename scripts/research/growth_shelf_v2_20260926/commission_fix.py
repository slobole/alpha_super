"""Research-only correction of the engine's per-share commission on split-adjusted share counts (review H1).

The engine charges max($1, $0.005 x shares) with shares sized on CAPITALSPECIAL (split-adjusted) prices. For a
ticker that later split forward, the adjusted price is lower than the price traded that day, so the share count and
the commission are inflated by the split factor R = Unadjusted Close / Close (e.g. TQQQ: 384 in 2010, 96 in 2013).
The correction re-prices each fill's commission on the real share count:

    real_commission = max(min(engine_commission, $1), engine_commission / R_t)
    add_back_t      = sum over fills on t of (engine_commission - real_commission) / NAV_{t-1}

Synthetic bars (proxy_runs.py) carry the split factor of the real fund's first bar, since they were spliced onto the
adjusted series at that bar. Nothing in the engine or in any sleeve file changes; the add-back series are written to
results/research/portfolio/growth_shelf_v2_20260926/commission_fix/ and applied only in shelf_books.py's
"commission-fixed" tables.

Usage: python commission_fix.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for path in (REPO, REPO / "scripts" / "research" / "fund_menu_20260923"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import common  # noqa: E402

OUT = REPO / "results" / "research" / "portfolio" / "growth_shelf_v2_20260926"
FIX_DIR = OUT / "commission_fix"
R1000_DIR = REPO / "results" / "research" / "mr_beyond_dv2_20260926" / "cache" / "r1000"
DV2_OUT = REPO / "results" / "research" / "dv2_deep_20260925"
SYNTHETIC_START = {"TQQQ": pd.Timestamp("2010-02-11"), "BTAL": pd.Timestamp("2011-09-13")}
INVENTORY_ALIASES = ["taa_btal_tqqq", "taa_btal_1n_tqqq", "taa_btal_lin_qqq", "ndx_vxn", "mosaic", "dv2", "hpi_vote",
                     "core5", "tactical_fi", "eom_flow", "sector_vox_iyr"]
PROXY_ALIASES = ["taa_btal_tqqq", "taa_btal_1n_tqqq", "taa_btal_lin_qqq"]
PROXY_MODES = ["splice_scaled", "splice_unscaled"]


def real_commission(engine_commission: np.ndarray, split_ratio: np.ndarray) -> np.ndarray:
    """Commission on the real share count; a fill at the engine's $1 minimum stays at the minimum."""
    engine_commission = np.asarray(engine_commission, dtype=float)
    ratio = np.where(np.isfinite(split_ratio) & (split_ratio > 0), split_ratio, 1.0)
    return np.maximum(np.minimum(engine_commission, 1.0), engine_commission / ratio)


def add_back_ser(tx: pd.DataFrame, nav: pd.Series, ratio: np.ndarray) -> pd.Series:
    """Daily return add-back: (engine - real commission) summed per fill date, over the prior close NAV."""
    saving = pd.Series(tx["commission_float"].to_numpy() - real_commission(tx["commission_float"].to_numpy(), ratio),
                       index=pd.DatetimeIndex(tx["date"]))
    daily = saving.groupby(level=0).sum()
    # *** CRITICAL*** divide by the prior close NAV so the add-back is in that day's return units.
    return (daily.reindex(nav.index).fillna(0.0) / nav.shift(1)).fillna(0.0)


class SplitRatio:
    """R(ticker, date) = Unadjusted Close / Close (CAPITALSPECIAL): the Russell 1000 cache first, Norgate otherwise."""

    def __init__(self) -> None:
        dates = pd.DatetimeIndex(np.load(R1000_DIR / "dates.npy"))
        symbols = np.load(R1000_DIR / "symbols.npy").astype(str)
        unadj = np.load(R1000_DIR / "Unadjusted Close.npy", mmap_mode="r")
        close = np.load(R1000_DIR / "Close.npy", mmap_mode="r")
        self._cache = {"dates": dates, "col": {s: i for i, s in enumerate(symbols)}, "unadj": unadj, "close": close}
        self._norgate: dict[str, pd.Series] = {}

    def series(self, ticker: str) -> pd.Series:
        if ticker in self._cache["col"]:
            i = self._cache["col"][ticker]
            with np.errstate(invalid="ignore", divide="ignore"):
                values = np.asarray(self._cache["unadj"][:, i], dtype=float) / np.asarray(self._cache["close"][:, i], dtype=float)
            return pd.Series(values, index=self._cache["dates"])
        if ticker not in self._norgate:
            from data.norgate_loader import load_price_timeseries

            frame = load_price_timeseries(ticker, start_date_str="1990-01-01")
            frame.index = pd.to_datetime(frame.index).normalize()
            self._norgate[ticker] = frame["Unadjusted Close"] / frame["Close"]
        return self._norgate[ticker]

    def at(self, tickers: pd.Series, dates: pd.Series) -> np.ndarray:
        out = np.full(len(tickers), np.nan)
        for ticker in pd.unique(tickers):
            mask = (tickers == ticker).to_numpy()
            ser = self.series(str(ticker)).dropna()
            fill_dates = pd.DatetimeIndex(dates[mask])
            if str(ticker) in SYNTHETIC_START:
                # Synthetic bars were spliced onto the adjusted series at the fund's first bar: use that bar's factor.
                first = SYNTHETIC_START[str(ticker)]
                fill_dates = pd.DatetimeIndex(np.where(fill_dates < first, first, fill_dates))
            # as-of lookup: the factor on the fill date (or the last one before it)
            base = ser.reindex(ser.index.union(fill_dates.unique())).ffill()
            out[mask] = base.reindex(fill_dates).to_numpy()
        return out


def normalise(tx: pd.DataFrame) -> pd.DataFrame:
    if "bar" in tx.columns:  # engine-native transaction frame (industry-ETF DV2 wired check)
        tx = tx.rename(columns={"bar": "date", "asset": "asset_str", "amount": "amount_float", "commission": "commission_float"})
    tx = tx.copy()
    tx["date"] = pd.to_datetime(tx["date"]).dt.normalize()
    return tx[["date", "asset_str", "amount_float", "commission_float"]]


def main() -> int:
    FIX_DIR.mkdir(parents=True, exist_ok=True)
    ratio = SplitRatio()
    paths = common.load_sleeve_path_dict()
    jobs = [(a, "inventory", common.SOURCE_DIR_PATH / f"{a}__transactions.csv.gz", paths[a]["total_value_float"]) for a in INVENTORY_ALIASES]
    jobs.append(("etf_ind", "engine", DV2_OUT / "wired_check" / "etf__transactions.csv",
                 pd.read_csv(DV2_OUT / "wired_check" / "etf__path.csv", index_col=0, parse_dates=True)["total_value"]))
    jobs.append(("etf_ind", "research", DV2_OUT / "sources" / "etf_ind_adv50__transactions.csv.gz",
                 pd.read_csv(DV2_OUT / "sources" / "etf_ind_adv50__path.csv.gz", index_col="date", parse_dates=True)["total_value_float"]))
    run_dir = OUT / "proxy_runs"
    for mode in PROXY_MODES:
        for alias in PROXY_ALIASES:
            nav = pd.read_csv(run_dir / mode / f"{alias}__path.csv.gz", index_col="date", parse_dates=True)["total_value_float"]
            jobs.append((alias, mode, run_dir / mode / f"{alias}__transactions.csv.gz", nav))
    summary = []
    for alias, source, tx_path, nav in jobs:
        tx = normalise(pd.read_csv(tx_path))
        nav = nav.copy()
        nav.index = pd.to_datetime(nav.index).normalize()
        split = ratio.at(tx["asset_str"], tx["date"])
        add = add_back_ser(tx, nav, split)
        if source in PROXY_MODES:
            # The run's first bar has no prior NAV; its fills are charged against the starting $1M.
            first_fill = tx["date"].min()
            if first_fill == nav.index[0]:
                first_mask = (tx["date"] == first_fill).to_numpy()
                saving = tx["commission_float"].to_numpy()[first_mask] - real_commission(tx["commission_float"].to_numpy()[first_mask], split[first_mask])
                add.iloc[0] = float(saving.sum()) / 1_000_000.0
        add.rename("add_back").to_csv(FIX_DIR / f"{alias}__{source}.csv.gz", float_format="%.10g")
        yearly = add.groupby(add.index.year).sum()
        summary.append({"alias": alias, "source": source, "fills": len(tx), "missing_ratio": int(np.isnan(split).sum()),
                        "engine_commission": float(tx["commission_float"].sum()),
                        "real_commission": float(real_commission(tx["commission_float"].to_numpy(), split).sum()),
                        **{f"y{y}": float(v) for y, v in yearly.items() if y in (2008, 2009, 2010, 2012, 2013, 2016, 2020, 2025)}})
    frame = pd.DataFrame(summary)
    frame.to_csv(FIX_DIR / "summary.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 250)
    print(frame.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
