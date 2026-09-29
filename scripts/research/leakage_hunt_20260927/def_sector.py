"""Sector ETF mean-reversion pods of the defensive book / fund menu: leakage tests on REAL Norgate data.

  vox_iyr : strategies.mean_reversion.strategy_mr_us_sector_etf_ibs_downshock_vox_iyr
  kie_ihi : strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_asset_sma200

  1. Signal-table invariance (entry/exit booleans + ranking key order) for EVERY basket ETF, k in (40, 0.1, 1.5).
  2. Engine-level invariance on 4 ETFs (transactions: date, asset, side).
  3. Truncation: signal rows from data ending at T equal full-history rows, 8 cut-offs.
  4. Padding provenance: ALLMARKETDAYS rows that are not observed (PaddingType.NONE) in the run window.
  5. E-02: per-share commission on adjusted units -> commission on raw units run (historical_share_units_bool=True
     converts the commission of explicit share orders; these pods size fractional shares so rounding is moot).
Usage: python def_sector.py vox_iyr|kie_ihi
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from def_common import (BOOK_END, BOOK_START, FACTORS, OUT, cached, dump_json, harness, metric_rows,
                        per_share_fee_effect)

import numpy as np
import pandas as pd

from alpha.engine.backtest import run_daily

POD = sys.argv[1] if len(sys.argv) > 1 else "vox_iyr"

if POD == "vox_iyr":
    from strategies.mean_reversion import strategy_mr_us_sector_etf_ibs_downshock as base_m
    from strategies.mean_reversion import strategy_mr_us_sector_etf_ibs_downshock_vox_iyr as m
    CFG = m.DEFAULT_CONFIG
    RANK_FIELD = f"prior_natr_{CFG.atr_lookback_day_int}_ser"

    def load():
        return base_m.get_us_sector_etf_ibs_downshock_data(CFG)

    def signals(px):
        return base_m.compute_us_sector_etf_ibs_downshock_signal_df(px, CFG)

    def make_strategy():
        return m.UsSectorEtfIbsDownshockVoxIyrStrategy(name=m.STRATEGY_NAME_STR, benchmarks=[CFG.benchmark_symbol_str], config_obj=CFG)

    def calendar(px):
        return base_m.resolve_us_sector_etf_execution_calendar_idx(px, CFG)
    ENGINE_CASES = (("XLK", 40.0), ("VOX", 0.1), ("IYR", 1.5), ("XLE", 40.0))
else:
    from strategies.mean_reversion import strategy_mr_sector_dispersion_ibs as base_m
    from strategies.mean_reversion import strategy_mr_sector_dispersion_ibs_kie_ihi_asset_sma200 as m
    CFG = m.DEFAULT_CONFIG
    RANK_FIELD = None

    def load():
        return base_m.get_sector_dispersion_ibs_data(CFG)

    def signals(px):
        return m.compute_asset_sma200_filtered_signal_df(px, CFG)

    def make_strategy():
        return m.SectorDispersionIbsKieIhiAssetSma200Strategy(name=m.STRATEGY_NAME_STR, benchmarks=[CFG.benchmark_symbol_str], config_obj=CFG)

    def calendar(px):
        return base_m.resolve_full_basket_calendar_idx(px, CFG, required_close_history_observation_count_int=m.ASSET_SMA_LOOKBACK_DAY_INT)
    ENGINE_CASES = (("KIE", 40.0), ("IHI", 0.1), ("SOXX", 1.5), ("IBB", 40.0))

SYMS = list(CFG.symbol_tuple)


def sig_tables(px):
    sig = signals(px)
    ent = pd.DataFrame({s: sig[(s, "entry_signal_bool")].astype(bool) for s in SYMS})
    ext = pd.DataFrame({s: sig[(s, "exit_signal_bool")].astype(bool) for s in SYMS})
    rank = pd.DataFrame({s: sig[(s, RANK_FIELD)].astype(float) for s in SYMS}) if RANK_FIELD else None
    return ent, ext, rank


def rank_orders(ent, rank):
    """Per date: ordered tuple of entry candidates by descending ranking key (stable in basket order)."""
    out = {}
    for ts in ent.index[ent.any(axis=1)]:
        cands = [s for s in SYMS if ent.at[ts, s] and np.isfinite(rank.at[ts, s])]
        out[ts] = tuple(sorted(cands, key=lambda s: -rank.at[ts, s]))
    return out


def compare(ref, cand, end_ts=None):
    (e1, x1, r1), (e2, x2, r2) = ref, cand
    if end_ts is not None:
        e1, x1, e2, x2 = e1.loc[:end_ts], x1.loc[:end_ts], e2.loc[:end_ts], x2.loc[:end_ts]
        if r1 is not None:
            r1, r2 = r1.loc[:end_ts], r2.loc[:end_ts]
    idx_same = e1.index.equals(e2.index)
    ne = int((e1 != e2.reindex_like(e1)).to_numpy().sum())
    nx = int((x1 != x2.reindex_like(x1)).to_numpy().sum())
    out = {"n_rows": int(len(e1)), "index_equal": bool(idx_same), "n_entry_cells_diff": ne, "n_exit_cells_diff": nx,
           "n_entry_true": int(e1.to_numpy().sum()), "n_exit_true": int(x1.to_numpy().sum())}
    if r1 is not None:
        o1, o2 = rank_orders(e1, r1), rank_orders(e2, r2)
        bad = [d for d in o1 if o1[d] != o2.get(d)]
        out["n_rank_order_diff"] = len(bad)
        out["rank_order_examples"] = [(d.date().isoformat(), o1[d], o2.get(d)) for d in bad[:3]]
        out["max_rel_rank_key_diff"] = float(((r1 - r2) / r1).abs().max().max())
    out["passed"] = idx_same and ne == 0 and nx == 0 and out.get("n_rank_order_diff", 0) == 0
    return out


def run_engine(px, historical_units=False):
    s = make_strategy()
    if historical_units:
        s.historical_share_units_bool = True
    run_daily(s, px, calendar(px), show_progress=False, show_signal_progress_bool=False, audit_override_bool=False)
    return s


def main():
    t0 = time.time()
    log = harness.ResultLog(f"sector_{POD}")
    px = cached(f"sector_{POD}_pricing", load)
    print("loaded", px.shape, px.index[0], px.index[-1], f"{time.time()-t0:.1f}s")
    base_sig = sig_tables(px)

    for s in SYMS:
        for k in FACTORS:
            res = compare(base_sig, sig_tables(harness.rescale_symbol_history(px, s, k)))
            log.add("invariance_signal", f"{s}_k{k}", res["passed"], res)
    print(f"signal invariance done {time.time()-t0:.1f}s")

    for name, T in {"mid_month_2013-10-22": "2013-10-22", "mid_month_2020-03-16": "2020-03-16",
                    "month_end_weekend_2020-05-29": "2020-05-29", "month_end_holiday_2021-05-28": "2021-05-28",
                    "day_before_month_end_2021-05-27": "2021-05-27", "month_end_plain_2008-09-30": "2008-09-30",
                    "month_end_weekend_2022-12-30": "2022-12-30", "mid_month_2025-04-08": "2025-04-08"}.items():
        T = pd.Timestamp(T)
        res = compare(base_sig, sig_tables(px.loc[:T]), end_ts=T)
        log.add("truncation_prefix", name, res["passed"], res)

    # padding provenance
    import norgatedata as nd
    cal = calendar(px)
    pad = {}
    for s in SYMS:
        obs = nd.price_timeseries(s, stock_price_adjustment_setting=nd.StockPriceAdjustmentType.CAPITALSPECIAL,
                                  padding_setting=nd.PaddingType.NONE, start_date="1990-01-01", timeseriesformat="pandas-dataframe")
        ser = px[(s, "Close")].dropna()
        padded = ser.index.difference(obs.index)
        zero_range = px.loc[cal, (s, "High")] <= px.loc[cal, (s, "Low")]
        pad[s] = {"n_padded_rows_total": int(len(padded)), "n_padded_rows_in_run": int((padded >= cal[0]).sum()),
                  "padded_in_run_examples": [d.date().isoformat() for d in padded[padded >= cal[0]][:10]],
                  "n_zero_range_rows_in_run": int(zero_range.sum())}
    dump_json(pad, f"sector_{POD}_padding_provenance.json")
    npad = sum(v["n_padded_rows_in_run"] for v in pad.values())
    log.add("padding_provenance", "padded_rows_in_run_window", npad == 0, {"n": npad})
    print(f"padding done {time.time()-t0:.1f}s")

    base = run_engine(px)
    base_tx = harness.transaction_decisions(base.get_transactions())
    for s, k in ENGINE_CASES:
        c = run_engine(harness.rescale_symbol_history(px, s, k))
        res = harness.compare_transaction_decisions(base_tx, harness.transaction_decisions(c.get_transactions()))
        res["commission_total_ref"] = float(base.get_transactions()["commission"].sum())
        res["commission_total_cand"] = float(c.get_transactions()["commission"].sum())
        log.add("invariance_engine", f"{s}_k{k}", res["n_only_reference"] == 0 and res["n_only_candidate"] == 0, res)
    print(f"engine invariance done {time.time()-t0:.1f}s")

    windows = {"full": (None, None), "book": (BOOK_START, BOOK_END)}
    rows = metric_rows("baseline_adjusted_units", base.results["daily_returns"], windows)
    hist = run_engine(px, historical_units=True)
    rows += metric_rows("commission_on_raw_units", hist.results["daily_returns"], windows)
    htx = harness.transaction_decisions(hist.get_transactions())
    hres = harness.compare_transaction_decisions(base_tx, htx)
    log.add("historical_units_decisions", "same_date_asset_side", hres["n_only_reference"] == 0 and hres["n_only_candidate"] == 0, hres)
    fee = per_share_fee_effect(base.get_transactions(), px, base._commission_per_share, base._commission_minimum)
    fr = fee.pop("_frame")
    fr.to_csv(OUT / f"sector_{POD}_fee_units_by_fill.csv", index=False)
    fee["commission_total_model_run"] = float(base.get_transactions()["commission"].sum())
    fee["commission_total_raw_units_run"] = float(hist.get_transactions()["commission"].sum())
    fee["per_share_float"] = base._commission_per_share
    fee["minimum_float"] = base._commission_minimum
    kr = {s: {"k_min": float((px[(s, "Unadjusted Close")] / px[(s, "Close")]).loc[cal[0]:].min()),
              "k_max": float((px[(s, "Unadjusted Close")] / px[(s, "Close")]).loc[cal[0]:].max())} for s in SYMS}
    pd.DataFrame(rows).to_csv(OUT / f"sector_{POD}_metrics.csv", index=False)
    dump_json({"fee_units": fee, "k_range_in_run": kr, "run_start": cal[0].date().isoformat()}, f"sector_{POD}_accounting_units.json")
    log.save(f"def_sector_{POD}")
    print(pd.DataFrame(rows).to_string())
    print(f"done {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
