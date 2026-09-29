"""MR capsule leakage hunt: future-action invariance and prefix tests on REAL Norgate data (research-only).

For one strategy family and one window [start, end] (full pre-window history kept for warm-up, calendar = window):

  invariance/<mode>/k   reference run on data truncated at end  vs  the same run with 8-13 symbols' ENTIRE history
                        rescaled as if a k:1 split happened after end (harness.rescale_symbol_history: OHLC and
                        Dividend / k, Volume * k, Unadjusted Close and Turnover unchanged).  Symbols = the five most
                        traded names of the reference run + real future-split names present in the window.
                        mode 'engine' = production accounting (adjusted-unit whole shares, adjusted-unit fees);
                        mode 'hsu'    = historical_share_units_bool=True (raw whole shares, raw-equivalent fees).
                        PASS = same (date, asset, side) set; notional drift reported separately.
  prefix/T              run truncated at T vs the reference run (to end); decisions dated <= T must be identical.
  prefix_asof_universe  DV2 only: run to T with the universe the production loader would build from data ending at T
                        (``idx.iloc[:-5]`` applied as of T) vs the reference run (universe built from today's data).

Usage: uv run python mr_invariance.py <dv2|hpi|etf> <window_index>
"""

from __future__ import annotations

import json
import sys
import time

import pandas as pd

import harness
import mr_common as mc
import mr_data

WINDOWS = {
    "dv2": [("2008-06-02", "2009-06-30"), ("2020-01-02", "2020-12-31"), ("2024-01-02", "2024-12-31")],
    "hpi": [("2008-06-02", "2009-06-30"), ("2020-01-02", "2020-12-31"), ("2024-01-02", "2024-12-31")],
    "etf": [("2013-01-02", "2013-12-31"), ("2020-01-02", "2020-12-31"), ("2024-01-02", "2024-12-31")],
}
SPLIT_NAMES = ("AAPL", "NVDA", "AMZN", "GOOGL", "TSLA", "AVGO", "NFLX", "CMG", "WMT", "LRCX")
ETF_SPLIT_NAMES = ("SMH", "XBI", "IBB", "SOXX", "ITB")
FACTORS = harness.DEFAULT_FACTOR_TUPLE


def build(family: str, data: dict, universe=None, hsu: bool = False):
    if family == "dv2":
        strategy = mc.make_dv2(data["universe_trimmed"] if universe is None else universe)
    elif family == "hpi":
        strategy = mc.make_hpi(data["universe"] if universe is None else universe)
    else:
        strategy = mc.make_etf(data["universe"] if universe is None else universe)
    strategy.historical_share_units_bool = hsu
    return strategy


def decisions(strategy) -> pd.DataFrame:
    tx = strategy.get_transactions()
    if len(tx) == 0:
        return pd.DataFrame(columns=["date", "asset", "side", "notional"])
    return harness.transaction_decisions(tx)


def passed(cmp: dict) -> bool:
    return cmp["n_only_reference"] == 0 and cmp["n_only_candidate"] == 0


def main(family: str, window_index: int) -> None:
    start, end = WINDOWS[family][window_index]
    tag = f"{family}_w{window_index}_{start[:4]}"
    log = harness.ResultLog(family)
    data = mr_data.load("dv2" if family == "dv2" else family)
    full_pricing = data["pricing_df"]
    t0 = time.time()
    if family == "etf":
        pricing = full_pricing.loc[: pd.Timestamp(end)].copy()
        pricing.attrs.update(full_pricing.attrs)
        split_names = ETF_SPLIT_NAMES
    else:
        universe = data["universe_trimmed"] if family == "dv2" else data["universe"]
        split_names = SPLIT_NAMES
        pricing = mc.subset_pricing(full_pricing.loc[: pd.Timestamp(end)], universe, start, end, extra=split_names,
                                    bench=("$SPX",) if family == "dv2" else ("$SPXTR",))
    del full_pricing
    info = {"window": [start, end], "n_symbols_loaded": int(pricing.columns.get_level_values(0).nunique())}

    reference = {}
    for hsu in (False, True):
        strategy = mc.run(build(family, data, hsu=hsu), pricing, start, end)
        reference[hsu] = decisions(strategy)
        info[f"n_decisions_ref_{'hsu' if hsu else 'engine'}"] = int(len(reference[hsu]))
    ref = reference[False]
    buys = ref[ref["side"] > 0]
    top = buys["asset"].value_counts().index[:5].tolist()
    present = set(pricing.columns.get_level_values(0).astype(str))
    chosen = list(dict.fromkeys(top + [s for s in split_names if s in present]))
    info["rescaled_symbols"] = chosen
    info["rescaled_symbols_traded_in_window"] = sorted(set(chosen) & set(ref["asset"].astype(str)))

    for hsu in (False, True):
        mode = "hsu" if hsu else "engine"
        for k in FACTORS:
            cand_pricing = pricing
            for symbol in chosen:
                cand_pricing = harness.rescale_symbol_history(cand_pricing, symbol, k)
            cand_pricing.attrs.update(pricing.attrs)
            try:
                strategy = mc.run(build(family, data, hsu=hsu), cand_pricing, start, end)
                cmp = harness.compare_transaction_decisions(reference[hsu], decisions(strategy), end)
                cmp["rescaled_symbols"] = chosen
                log.add(f"invariance_{mode}", f"{tag}_k{k:g}", passed(cmp), cmp)
            except Exception as exc:
                log.add(f"invariance_{mode}", f"{tag}_k{k:g}", False, {"error": f"{type(exc).__name__}: {exc}"})

    sessions = pricing.index[(pricing.index >= pd.Timestamp(start)) & (pricing.index <= pd.Timestamp(end))]
    for back in (60, 120):
        cut = sessions[-1 - back]
        strategy = mc.run(build(family, data), pricing.loc[:cut], start, cut)
        cmp = harness.compare_transaction_decisions(ref, decisions(strategy), cut, rel_tol_float=1e-9)
        log.add("prefix", f"{tag}_T{cut.date()}", passed(cmp) and cmp["n_notional_beyond_tol"] == 0, cmp)

    if family == "dv2":
        for back in (60, 120):
            cut = sessions[-1 - back]
            asof_universe = mc.asof_trimmed_universe(data["universe_untrimmed"], cut)
            strategy = mc.run(build(family, data, universe=asof_universe), pricing.loc[:cut], start, cut)
            cmp = harness.compare_transaction_decisions(ref, decisions(strategy), cut, rel_tol_float=1e-9)
            log.add("prefix_asof_universe", f"{tag}_T{cut.date()}", passed(cmp), cmp)

    info["runtime_s"] = round(time.time() - t0, 1)
    out = mc.OUT / "invariance"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}.json").write_text(json.dumps({"info": info, "rows": log.rows}, indent=2, default=str),
                                      encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
