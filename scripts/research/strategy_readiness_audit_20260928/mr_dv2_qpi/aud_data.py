"""Strategy readiness audit 2026-09-28, DV2 + QPI: load Norgate once and cache (research-only).

Both strategies load exactly the same inputs in production:
    symbols, universe = data.norgate_loader.build_index_constituent_matrix("S&P 500")   (trimmed: idx.iloc[:-5])
    pricing = load_raw_prices(symbols, ["$SPX"], "1998-01-01", None)                     (CAPITALSPECIAL, ALLMARKETDAYS)
(strategies/dv2/strategy_mr_dv2.py:278-279, strategies/qpi/strategy_mr_qpi_ibs_rsi_exit.py:475-481).

The cache additionally stores the UNTRIMMED membership matrix (same code without ``idx.iloc[:-5]``) so the audit can
rebuild the universe the loader would have produced with data ending at any historical date (live replay) and
measure the trim.

Cache: <scratchpad>/aud_cache/sp500_inputs.pkl
Usage: uv run python scripts/research/strategy_readiness_audit_20260928/mr_dv2_qpi/aud_data.py
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CACHE = Path(r"C:\Users\User\AppData\Local\Temp\claude\C--Users-User-Documents-workspace-alpha-super--claude-worktrees-interesting-shtern-a85aa1"
             r"\ea09c56d-df97-4d0a-a36e-f54bf271142c\scratchpad\aud_cache")
CACHE.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE / "sp500_inputs.pkl"


def _retry(fn, *args, tries: int = 5, **kwargs):
    for attempt in range(tries):
        try:
            return fn(*args, **kwargs)
        except ValueError:
            if attempt == tries - 1:
                raise
            time.sleep(5 + 5 * attempt)


def build_untrimmed_universe(indexname: str = "S&P 500") -> pd.DataFrame:
    import norgatedata

    symbols = _retry(norgatedata.watchlist_symbols, f"{indexname} Current & Past")
    frames = []
    for symbol in symbols:
        idx = _retry(norgatedata.index_constituent_timeseries, symbol, indexname, timeseriesformat="pandas-dataframe")
        if idx["Index Constituent"].sum() > 0:
            idx = idx.rename(columns={"Index Constituent": symbol})
            frames.append(idx.loc[idx[symbol] == 1])
    return pd.concat(frames, axis=1).fillna(0).astype(int).sort_index()


def asof_trimmed_universe(untrimmed_df: pd.DataFrame, asof_ts) -> pd.DataFrame:
    """Universe exactly as data/norgate_loader.build_index_constituent_matrix builds it from data ending at asof_ts:
    each symbol's member rows up to asof; if its last member row != asof (the $SPX last day), drop its last 5 member
    rows (data/norgate_loader.py:106-107)."""
    asof_ts = pd.Timestamp(asof_ts)
    frame = untrimmed_df.loc[:asof_ts]
    out = frame.copy()
    for symbol in frame.columns:
        rows = np.flatnonzero(frame[symbol].to_numpy() == 1)
        if len(rows) == 0:
            continue
        if frame.index[rows[-1]] != asof_ts:
            out.iloc[rows[-5:], out.columns.get_loc(symbol)] = 0
    out = out.loc[:, out.sum(axis=0) > 0]
    return out


def main() -> None:
    from data.norgate_loader import build_index_constituent_matrix, load_raw_prices

    t0 = time.time()
    symbols, universe_df = _retry(build_index_constituent_matrix, indexname="S&P 500")
    untrimmed_df = build_untrimmed_universe()
    last_ts = universe_df.index.max()
    replica = asof_trimmed_universe(untrimmed_df, untrimmed_df.index.max())
    same = universe_df.equals(replica.reindex(index=universe_df.index, columns=universe_df.columns).fillna(0).astype(int))
    print(f"replica(as-of last day) == production universe: {same}; prod {universe_df.shape} replica {replica.shape}")
    pricing_df = load_raw_prices(list(symbols), ["$SPX"], "1998-01-01", None)
    with CACHE_FILE.open("wb") as handle:
        pickle.dump({"symbols": list(symbols), "universe_trimmed": universe_df, "universe_untrimmed": untrimmed_df,
                     "pricing_df": pricing_df, "attrs": dict(pricing_df.attrs), "replica_equal": bool(same),
                     "loaded_at": pd.Timestamp.now().isoformat()}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"cached in {time.time() - t0:.0f}s; pricing {pricing_df.shape}, last bar {pricing_df.index[-1].date()}, "
          f"universe last row {last_ts.date()}")


def load() -> dict:
    with CACHE_FILE.open("rb") as handle:
        out = pickle.load(handle)
    out["pricing_df"].attrs.update(out.get("attrs", {}))
    return out


if __name__ == "__main__":
    main()
