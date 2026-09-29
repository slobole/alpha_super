"""MR capsule leakage hunt: load Norgate once and cache (research-only; no strategy/engine edits).

Caches (pickles) under the session scratchpad mr_cache/:
- dv2_inputs.pkl : symbols, trimmed universe (production loader), untrimmed universe (same code without
                   ``idx.iloc[:-5]``), pricing frame exactly as strategies.dv2.strategy_mr_dv2.run_variant loads it.
- hpi_inputs.pkl : symbols, universe, pricing frame exactly as strategies.hpi.stateful_long.load_exact_hpi_inputs.
- etf_inputs.pkl : industry-ETF DV2 pricing + history universe as strategy_mr_dv2_industry_etf._load.

Usage: uv run python scripts/research/leakage_hunt_20260927/mr_data.py [dv2|hpi|etf|all]
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CACHE = Path(r"C:\Users\User\AppData\Local\Temp\claude\C--Users-User-Documents-workspace-alpha-super"
             r"\2b26c675-1db0-4ceb-8ed6-fb7880170526\scratchpad\mr_cache")
CACHE.mkdir(parents=True, exist_ok=True)


def build_dv2_universes(indexname: str = "S&P 500"):
    """Replicates data/norgate_loader.build_index_constituent_matrix; returns (symbols, trimmed, untrimmed, removal_log)."""
    import norgatedata

    symbols = norgatedata.watchlist_symbols(f"{indexname} Current & Past")
    calendar = norgatedata.price_timeseries("$SPX", timeseriesformat="pandas-dataframe").index
    last_trading_day = calendar[-1]
    trimmed, untrimmed, removal_rows = [], [], []
    for symbol in symbols:
        idx = norgatedata.index_constituent_timeseries(symbol, indexname, timeseriesformat="pandas-dataframe")
        if idx["Index Constituent"].sum() > 0:
            idx = idx.rename(columns={"Index Constituent": symbol})
            idx = idx.loc[idx[symbol] == 1]
            untrimmed.append(idx.copy())
            if last_trading_day != idx.index[-1]:
                removal_rows.append({"symbol": symbol, "last_member_date": idx.index[-1],
                                     "trimmed_last_member_date": idx.index[-6] if len(idx) > 5 else pd.NaT})
                idx = idx.iloc[:-5]
            trimmed.append(idx)
    trimmed_df = pd.concat(trimmed, axis=1).fillna(0).astype(int).sort_index()
    untrimmed_df = pd.concat(untrimmed, axis=1).fillna(0).astype(int).sort_index()
    return list(symbols), trimmed_df, untrimmed_df, pd.DataFrame(removal_rows)


def cache_dv2() -> None:
    from data.norgate_loader import build_index_constituent_matrix
    from strategies.dv2.strategy_mr_dv2 import get_prices

    t0 = time.time()
    symbols, trimmed_df, untrimmed_df, removal_df = build_dv2_universes()
    prod_symbols, prod_universe_df = build_index_constituent_matrix(indexname="S&P 500")
    same_bool = prod_universe_df.equals(trimmed_df.reindex(index=prod_universe_df.index, columns=prod_universe_df.columns))
    print(f"replica trimmed == production universe: {same_bool}; shapes {prod_universe_df.shape} {trimmed_df.shape}")
    if not same_bool:
        raise RuntimeError("trimmed replica differs from production build_index_constituent_matrix")
    pricing_df = get_prices(list(prod_symbols), ["$SPX"], start_date="1998-01-01", end_date=None)
    with (CACHE / "dv2_inputs.pkl").open("wb") as handle:
        pickle.dump({"symbols": list(prod_symbols), "universe_trimmed": prod_universe_df,
                     "universe_untrimmed": untrimmed_df, "removal_df": removal_df, "pricing_df": pricing_df,
                     "attrs": dict(pricing_df.attrs)}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"dv2 cached in {time.time() - t0:.0f}s; pricing {pricing_df.shape}, last {pricing_df.index[-1].date()}")


def cache_hpi() -> None:
    from strategies.hpi.stateful_long import load_exact_hpi_inputs

    t0 = time.time()
    symbols, universe_df, pricing_df = load_exact_hpi_inputs("S&P 500", "$SPXTR", "1998-01-01", None)
    with (CACHE / "hpi_inputs.pkl").open("wb") as handle:
        pickle.dump({"symbols": symbols, "universe": universe_df, "pricing_df": pricing_df,
                     "attrs": dict(pricing_df.attrs)}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"hpi cached in {time.time() - t0:.0f}s; pricing {pricing_df.shape}, last {pricing_df.index[-1].date()}")


def cache_etf() -> None:
    from strategies.dv2 import strategy_mr_dv2_industry_etf as etf

    pricing_df, universe_df = etf._load(None)
    with (CACHE / "etf_inputs.pkl").open("wb") as handle:
        pickle.dump({"pricing_df": pricing_df, "universe": universe_df, "attrs": dict(pricing_df.attrs)}, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"etf cached; pricing {pricing_df.shape}")


def load(name: str) -> dict:
    with (CACHE / f"{name}_inputs.pkl").open("rb") as handle:
        out = pickle.load(handle)
    out["pricing_df"].attrs.update(out.get("attrs", {}))
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("etf", "all"):
        cache_etf()
    if which in ("dv2", "all"):
        cache_dv2()
    if which in ("hpi", "all"):
        cache_hpi()
