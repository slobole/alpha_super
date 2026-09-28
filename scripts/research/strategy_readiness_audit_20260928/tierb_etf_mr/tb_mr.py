"""Sector-ETF IBS pods (VOX/IYR downshock; KIE/IHI/XLC dispersion variants): BC + TR checks on REAL Norgate data.

  rowt <key>     A3 row-T truncation: compute_signals(data <= T) vs compute_signals(full) on every feature row <= T,
                 INCLUDING row T (IBS, ranges, ATR/NATR, SMA200, entry/exit booleans).  >= 12 cut-offs, 5 of them
                 decision dates with a real entry or exit at T.  Engine decision replay at 4 of them (data <= T plus
                 one synthetic T+1 row, recorded orders at decision T vs the full run).
                 A4: planted one-session leak (IBS from Close_(T+1), entry/exit booleans from T+1) must be caught.
  split <key>    A2 future-split invariance, >= 3 symbols incl. held ones, k in {40, 0.1, 1.5}; float32 (as loaded)
                 and float64 control; engine fill sets.  A4 control: a price-LEVEL rule (scale-dependent) must fail.
  ties <key>     float32 vs float64 decision flips on today's data, near-tie census, engine effect.
  padded <key>   PaddingType.NONE provenance: padded rows in the run, signals / decisions / fills on padded rows,
                 zero-volume observed rows.
  account <key>  raw-unit fees (historical_share_units_bool), dividends/withholding status, negative cash, idle cash.
  small <key>    USD 30K friction: raw whole shares + IBKR Fixed $0.005/share, $1 minimum (and Tiered) vs model.
  trade <key>    C1 participation vs native-Turnover ADV20 at 30K/1M/10M (full, last 3y), MOO guardrail share;
                 C2 whole-share weight error at 30K; C3 instrument inception / recent liquidity.

Usage: uv run python tb_mr.py <mode> <key>
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import tb_common as tc

PRICE_FIELDS = {"Open", "High", "Low", "Close", "Volume", "Turnover", "Unadjusted Close", "Dividend",
                "Delta", "Deliverable Quantity"}
SPLIT_SYMBOLS = {"vox": ("XLK", "XLE", "VOX", "IYR"), "xlc": ("KIE", "IHI", "SOXX", "XLC"),
                 "xlc200": ("KIE", "IHI", "SOXX", "XLC"), "kie200": ("KIE", "IHI", "SOXX", "IBB")}


def _cfg(key):
    return tc.config_for(key)


def _signals(key, pricing, cls=None):
    strat = tc.make_strategy(key, _cfg(key), cls)
    out = strat.compute_signals(pricing.copy())
    feat_cols = [c for c in out.columns if c[1] not in PRICE_FIELDS and c[0] in tc.symbols_for(key)]
    return out.loc[:, feat_cols]


def _thresholds(key):
    cfg = _cfg(key)
    if tc.MR_SPEC[key]["family"] == "downshock":
        return {"entry_ibs": cfg.entry_ibs_max_float, "exit_ibs": cfg.exit_ibs_min_float,
                "downshock": cfg.downshock_atr_max_float, "range_ratio": 1.0}
    return {"entry_ibs": cfg.entry_ibs_max_float, "exit_ibs": cfg.exit_ibs_min_float,
            "relative_range": cfg.min_relative_range_float}


# ------------------------------------------------------------------ leak / scale-dependent subclasses
def _leak_mixin(base_cls):
    class Leak(base_cls):
        def compute_signals(self, pricing_data_df):
            out = super().compute_signals(pricing_data_df)
            for s in self.config_obj.symbol_tuple:
                c1 = pd.to_numeric(out[(s, "Close")], errors="coerce").shift(-1)
                hi = pd.to_numeric(out[(s, "High")], errors="coerce")
                lo = pd.to_numeric(out[(s, "Low")], errors="coerce")
                out[(s, "ibs_value_ser")] = (c1 - lo) / (hi - lo).replace(0.0, np.nan)
                for f in ("entry_signal_bool", "exit_signal_bool"):
                    out[(s, f)] = out[(s, f)].shift(-1).fillna(False).astype(bool)
            return out
    Leak.__name__ = f"Leak{base_cls.__name__}"
    return Leak


def _level_mixin(base_cls, family):
    class Level(base_cls):
        if family == "downshock":
            def get_entry_candidate_list(self, close_row_ser):
                base = super().get_entry_candidate_list(close_row_ser)
                return sorted(base, key=lambda s: -float(close_row_ser[(s, "Close")]))
        else:
            def compute_signals(self, pricing_data_df):
                out = super().compute_signals(pricing_data_df)
                for s in self.config_obj.symbol_tuple:
                    gate = pd.to_numeric(out[(s, "Close")], errors="coerce") > 60.0
                    out[(s, "entry_signal_bool")] = out[(s, "entry_signal_bool")] & gate
                return out
    Level.__name__ = f"Level{base_cls.__name__}"
    return Level


def _record_mixin(base_cls):
    class Rec(base_cls):
        def iterate(self, data_df, close_row_ser, open_price_ser):
            if not hasattr(self, "order_log"):
                self.order_log = []
            n0 = len(self.get_orders())
            super().iterate(data_df, close_row_ser, open_price_ser)
            if close_row_ser is None:
                return
            for o in self.get_orders()[n0:]:
                self.order_log.append({"decision": pd.Timestamp(self.previous_bar), "asset": str(o.asset),
                                       "target": bool(o.target), "amount": float(o.amount)})
    Rec.__name__ = f"Rec{base_cls.__name__}"
    return Rec


# ------------------------------------------------------------------ A3 row-T
def _signal_cutoffs(key, pricing) -> dict[str, str]:
    base = {
        "mid_month_2026-09-15": "2026-09-15", "month_end_weekend_2026-05-29": "2026-05-29",
        "month_end_before_holiday_2025-12-31": "2025-12-31", "first_session_2026-09-01": "2026-09-01",
        "last_completed_month_end_2026-08-31": "2026-08-31", "current_partial_month_2026-09-24": "2026-09-24",
        "half_day_2024-11-29": "2024-11-29", "stress_2025-04-08": "2025-04-08",
    }
    tx = pd.read_parquet(tc.OUT / f"fills_{key}.parquet")
    tx["bar"] = pd.to_datetime(tx["bar"])
    idx = pricing.index
    dec = lambda b: idx[idx.get_loc(b) - 1]  # noqa: E731
    buys = tx[tx["amount"] > 0]["bar"].drop_duplicates().sort_values()
    sells = tx[tx["amount"] < 0]["bar"].drop_duplicates().sort_values()
    picks = {}
    for i, b in enumerate(list(buys.iloc[[len(buys) // 3, -8, -1]])):
        picks[f"entry_decision_{i}_{dec(b).date()}"] = str(dec(b).date())
    for i, b in enumerate(list(sells.iloc[[len(sells) // 2, -1]])):
        picks[f"exit_decision_{i}_{dec(b).date()}"] = str(dec(b).date())
    return {**base, **picks}


def run_rowt(key: str) -> None:
    pricing = tc.load_pricing(key)
    cuts = _signal_cutoffs(key, pricing)
    base_cls = tc.MR_SPEC[key]["cls"]
    leak_cls = _leak_mixin(base_cls)
    full = _signals(key, pricing)
    full_leak = _signals(key, pricing, leak_cls)
    rows = []
    for label, cut in cuts.items():
        cut_ts = pd.Timestamp(cut)
        rec = {"case": label, "cut": cut}
        for tag, cls, fref in (("prod", None, full), ("leak", leak_cls, full_leak)):
            tr = _signals(key, pricing.loc[:cut_ts], cls)
            a = fref.loc[:cut_ts]
            b = tr.reindex(index=a.index, columns=a.columns)
            av, bv = a.to_numpy(float), b.to_numpy(float)
            dm = ~((av == bv) | (np.isnan(av) & np.isnan(bv)))
            bool_cols = [i for i, c in enumerate(a.columns) if c[1].endswith("_bool")]
            rec[tag] = {"n_values": int(av.size), "n_diff": int(dm.sum()), "row_T_diff": int(dm[-1].sum()),
                        "row_T_decision_bool_diff": int(dm[-1, bool_cols].sum()),
                        "row_T_true_entries_full": int(np.nansum(av[-1, [i for i, c in enumerate(a.columns) if c[1] == "entry_signal_bool"]])),
                        "row_T_true_exits_full": int(np.nansum(av[-1, [i for i, c in enumerate(a.columns) if c[1] == "exit_signal_bool"]]))}
        rec["passed"] = rec["prod"]["n_diff"] == 0
        rec["leak_caught_row_T"] = rec["leak"]["row_T_diff"] > 0
        rec["leak_caught_decision_row_T"] = rec["leak"]["row_T_decision_bool_diff"] > 0
        rows.append(rec)
        print(key, label, "pass", rec["passed"], "leak", rec["leak_caught_row_T"], rec["leak_caught_decision_row_T"], flush=True)
    # engine decision replay at the 5 signal cut-offs (orders recorded at decision T)
    rec_cls, rec_leak = _record_mixin(base_cls), _record_mixin(leak_cls)
    cal_full = tc.calendar_for(key, pricing)
    full_run = tc.run(key, pricing, cls=rec_cls, calendar=cal_full)
    full_leak_run = tc.run(key, pricing, cls=rec_leak, calendar=cal_full)
    replay = []
    for label, cut in list(cuts.items())[-5:]:
        cut_ts = pd.Timestamp(cut)
        nxt = pricing.index[pricing.index.get_loc(cut_ts) + 1]
        frame = pricing.loc[:cut_ts]
        synth = frame.iloc[[-1]].copy()
        synth.index = pd.DatetimeIndex([nxt])
        frame2 = pd.concat([frame, synth])
        frame2.attrs.update(pricing.attrs)
        cal = cal_full[cal_full <= nxt]
        out = {"case": label, "cut": cut}
        for tag, cls, ref in (("prod", rec_cls, full_run), ("leak", rec_leak, full_leak_run)):
            r = tc.run(key, frame2, cls=cls, calendar=cal)
            fo = pd.DataFrame(ref.order_log)
            to = pd.DataFrame(r.order_log)
            fo = fo[fo["decision"] <= cut_ts].reset_index(drop=True)
            to = to[to["decision"] <= cut_ts].reset_index(drop=True)
            same = fo.equals(to)
            at_t_full = fo[fo["decision"] == cut_ts]
            at_t_tr = to[to["decision"] == cut_ts]
            out[tag] = {"orders_full": int(len(fo)), "orders_trunc": int(len(to)), "identical": bool(same),
                        "orders_at_T_full": at_t_full[["asset", "target", "amount"]].to_dict("records"),
                        "orders_at_T_trunc": at_t_tr[["asset", "target", "amount"]].to_dict("records")}
        out["passed"] = out["prod"]["identical"]
        out["leak_caught"] = not out["leak"]["identical"]
        replay.append(out)
        print(key, "replay", label, out["passed"], "leak caught", out["leak_caught"], flush=True)
    tc.write_json(f"{key}_rowt.json", {
        "n_cases": len(rows), "n_pass": sum(r["passed"] for r in rows),
        "n_leak_caught_row_T": sum(r["leak_caught_row_T"] for r in rows),
        "n_leak_caught_decision_row_T": sum(r["leak_caught_decision_row_T"] for r in rows),
        "replay_n": len(replay), "replay_pass": sum(r["passed"] for r in replay),
        "replay_leak_caught": sum(r["leak_caught"] for r in replay), "rows": rows, "replay": replay})


# ------------------------------------------------------------------ A2 split
def _decision_cell_diff(a: pd.DataFrame, b: pd.DataFrame, start) -> list[tuple]:
    cols = [c for c in a.columns if c[1] in ("entry_signal_bool", "exit_signal_bool")]
    x = a.loc[start:, cols].astype(bool)
    y = b.loc[start:, cols].astype(bool)
    diff = x.ne(y)
    return [(str(i.date()), c[0], c[1]) for i, row in diff.iterrows() for c in cols if row[c]]


def _distance_to_threshold(key, sig: pd.DataFrame, cells, close=None) -> list[dict]:
    th = _thresholds(key)
    out = []
    for d, s, f in cells:
        row = sig.loc[pd.Timestamp(d)]
        ibs = float(row[(s, "ibs_value_ser")])
        cand = [abs(ibs - th["entry_ibs"]), abs(ibs - th["exit_ibs"])]
        for fld, thv in (("downshock_atr_ser", th.get("downshock")), ("range_ratio_ser", th.get("range_ratio")),
                         ("relative_range_ser", th.get("relative_range"))):
            if thv is not None and (s, fld) in row.index:
                cand.append(abs(float(row[(s, fld)]) - thv))
        if (s, "asset_sma_200_ser") in row.index and close is not None:
            sma = float(row[(s, "asset_sma_200_ser")])
            cand.append(abs(float(close.loc[pd.Timestamp(d), (s, "Close")]) / sma - 1.0))
        out.append({"date": d, "symbol": s, "field": f, "min_abs_distance": float(np.nanmin(cand))})
    return out


def run_split(key: str, only_symbol: str | None = None) -> None:
    pricing = tc.load_pricing(key)
    start = tc.calendar_for(key, pricing)[0]
    syms = SPLIT_SYMBOLS[key] if only_symbol is None else (only_symbol,)
    held = pd.read_parquet(tc.OUT / f"fills_{key}.parquet")["asset"].value_counts()
    base_cls = tc.MR_SPEC[key]["cls"]
    level_cls = _level_mixin(base_cls, tc.MR_SPEC[key]["family"])
    p64 = tc.upcast64(pricing)
    ref32 = _signals(key, pricing)
    ref64 = _signals(key, p64)
    # Engine comparisons use raw-unit commissions (historical_share_units_bool=True): with fractional share orders
    # this converts only the per-share fee, so a future split cannot move fees (the E-02 adjusted-unit fee effect is
    # measured separately in `account`).
    ref_run = tc.run(key, p64, hsu=True)
    ref_fills = tc.fills(ref_run)
    lvl_ref = tc.fills(tc.run(key, p64, cls=level_cls, hsu=True))
    rows = []
    for s in syms:
        for k in (40.0, 0.1, 1.5):
            c32 = tc.rescale_symbol(pricing, [s], k)
            c64 = tc.rescale_symbol(p64, [s], k)
            d32 = _decision_cell_diff(ref32, _signals(key, c32), start)
            sig64 = _signals(key, c64)
            d64 = _decision_cell_diff(ref64, sig64, start)
            dist64 = _distance_to_threshold(key, ref64, d64, p64)
            run64 = tc.run(key, c64, hsu=True)
            cmp = tc.compare_fill_sets(ref_fills, tc.fills(run64), rel_tol=1e-6)
            lvl = tc.compare_fill_sets(lvl_ref, tc.fills(tc.run(key, c64, cls=level_cls, hsu=True)), rel_tol=1e-6)
            rec = {"symbol": s, "k": k, "fills_in_ref": int(held.get(s, 0)),
                   "f32_decision_cells_diff": len(d32), "f64_decision_cells_diff": len(d64),
                   "f64_diff_max_distance_to_threshold": max([x["min_abs_distance"] for x in dist64], default=0.0),
                   "f64_diff_cells": dist64[:6],
                   "engine_f64": cmp,
                   "engine_f64_pass": cmp["n_only_ref"] == 0 and cmp["n_only_cand"] == 0 and cmp["n_beyond_tol"] == 0,
                   "control_caught": bool(lvl["n_only_ref"] or lvl["n_only_cand"] or lvl["n_beyond_tol"])}
            rec["passed_f64_signal"] = len(d64) == 0 or rec["f64_diff_max_distance_to_threshold"] < 1e-12
            rows.append(rec)
            print(key, s, k, "f32", len(d32), "f64", len(d64), "eng", rec["engine_f64_pass"], "ctrl", rec["control_caught"], flush=True)
    suffix = "" if only_symbol is None else f"_{only_symbol}"
    tc.write_json(f"{key}_split{suffix}.json", {"n_cases": len(rows),
                                         "n_f64_signal_pass_or_exact_tie": sum(r["passed_f64_signal"] for r in rows),
                                         "n_engine_f64_pass": sum(r["engine_f64_pass"] for r in rows),
                                         "n_control_caught": sum(r["control_caught"] for r in rows), "rows": rows})


# ------------------------------------------------------------------ ties
def run_ties(key: str) -> None:
    pricing = tc.load_pricing(key)
    start = tc.calendar_for(key, pricing)[0]
    s32 = _signals(key, pricing)
    s64 = _signals(key, tc.upcast64(pricing))
    flips = _decision_cell_diff(s32, s64, start)
    th = _thresholds(key)
    ibs = s64.loc[start:, [c for c in s64.columns if c[1] == "ibs_value_ser"]]
    near = {}
    for name, v in (("entry_ibs", th["entry_ibs"]), ("exit_ibs", th["exit_ibs"])):
        dist = (ibs - v).abs()
        near[name] = {"exact_tie_f64": int((dist == 0).sum().sum()), "within_1e-6": int((dist < 1e-6).sum().sum()),
                      "within_1e-5": int((dist < 1e-5).sum().sum())}
    r32 = tc.run(key, pricing)
    r64 = tc.run(key, tc.upcast64(pricing))
    cmp = tc.compare_fill_sets(tc.fills(r32), tc.fills(r64), rel_tol=1e-6)
    out = {"decision_cells_f32_vs_f64": len(flips), "examples": flips[:10], "near_ties": near,
           "engine_fills_f32_vs_f64": cmp, "metrics_f32": tc.metric_pair(r32), "metrics_f64": tc.metric_pair(r64)}
    tc.write_json(f"{key}_ties.json", out)
    print(key, out["decision_cells_f32_vs_f64"], near, out["metrics_f32"]["full"], out["metrics_f64"]["full"], flush=True)


# ------------------------------------------------------------------ padded bars
def run_padded(key: str) -> None:
    from data.norgate_loader import norgatedata

    pricing = tc.load_pricing(key)
    cal = tc.calendar_for(key, pricing)
    sig = _signals(key, pricing)
    tx = pd.read_parquet(tc.OUT / f"fills_{key}.parquet")
    tx["bar"] = pd.to_datetime(tx["bar"])
    out = {"symbols": {}}
    for s in tc.symbols_for(key):
        obs = norgatedata.price_timeseries(s, stock_price_adjustment_setting=norgatedata.StockPriceAdjustmentType.CAPITALSPECIAL,
                                           padding_setting=norgatedata.PaddingType.NONE, start_date="1998-01-01",
                                           end_date=tc.END_STR, timeseriesformat="pandas-dataframe")
        obs_idx = pd.DatetimeIndex(obs.index)
        first_obs = obs_idx[0]
        rows_run = pricing.index[(pricing.index >= cal[0])]
        present = pd.to_numeric(pricing.loc[rows_run, (s, "Close")], errors="coerce").notna()
        padded = rows_run[present.to_numpy() & ~rows_run.isin(obs_idx)]
        pre_idx = pricing.index[(pricing.index < cal[0]) & (pricing.index >= first_obs)]
        padded_warm = pre_idx[~pre_idx.isin(obs_idx)]
        vol = pd.to_numeric(obs["Volume"], errors="coerce")
        zero_vol_run = obs_idx[(vol <= 0).to_numpy() & obs_idx.isin(rows_run)]
        ent = sig.loc[padded, (s, "entry_signal_bool")].astype(bool).sum() if len(padded) else 0
        ex = sig.loc[padded, (s, "exit_signal_bool")].astype(bool).sum() if len(padded) else 0
        fills_pad = tx[(tx["asset"] == s) & tx["bar"].isin(padded)]
        dec_pad = tx[(tx["asset"] == s) & tx["bar"].map(lambda b: pricing.index[pricing.index.get_loc(b) - 1]).isin(padded)]
        out["symbols"][s] = {"first_observed": str(first_obs.date()), "padded_rows_in_run": [str(x.date()) for x in padded],
                             "padded_rows_in_warmup": int(len(padded_warm)),
                             "zero_volume_observed_rows_in_run": [str(x.date()) for x in zero_vol_run][:20],
                             "n_zero_volume_observed_rows_in_run": int(len(zero_vol_run)),
                             "entry_signals_on_padded": int(ent), "exit_signals_on_padded": int(ex),
                             "fills_on_padded_bar": fills_pad[["bar", "amount", "price"]].astype(str).to_dict("records"),
                             "decisions_on_padded_bar": int(len(dec_pad))}
    out["totals"] = {"padded_rows_in_run": sum(len(v["padded_rows_in_run"]) for v in out["symbols"].values()),
                     "fills_on_padded": sum(len(v["fills_on_padded_bar"]) for v in out["symbols"].values()),
                     "decisions_on_padded": sum(v["decisions_on_padded_bar"] for v in out["symbols"].values())}
    tc.write_json(f"{key}_padded.json", out)
    print(key, out["totals"], flush=True)


# ------------------------------------------------------------------ accounting
def run_account(key: str) -> None:
    pricing = tc.load_pricing(key)
    base = tc.run(key, pricing)
    h = tc.run(key, pricing, hsu=True)
    res = base.results.copy()
    res.index = pd.to_datetime(res.index)
    cash, tv = res["cash"].astype(float), res["total_value"].astype(float)
    dtb3 = pd.read_csv(tc.REPO.parent / "1_data" / "DTB3.csv", parse_dates=["observation_date"], na_values=".")
    dtb3 = dtb3.set_index("observation_date")["DTB3"].astype(float).ffill() / 100.0
    rate = dtb3.reindex(res.index, method="ffill").shift(1).fillna(0.0)
    years = (res.index[-1] - res.index[0]).days / 365.25
    pos_int = (cash.clip(lower=0) / tv * rate / 252.0)
    neg_cost = (cash.clip(upper=0) / tv * (rate + 0.015) / 252.0)
    policy = {k: v for k, v in base._accounting_policy_dict.items() if "dividend" in k}
    k_ratio = {s: [float(x) for x in (pricing[(s, "Unadjusted Close")] / pricing[(s, "Close")]).astype(float)
                                     .loc[res.index[0]:].agg(["min", "max"])] for s in tc.symbols_for(key)}
    out = {"base": tc.metric_pair(base), "hsu_raw_unit_fees": tc.metric_pair(h),
           "commission_adjusted_units": float(base.get_transactions()["commission"].sum()),
           "commission_raw_units": float(h.get_transactions()["commission"].sum()),
           "k_unadj_over_adj": k_ratio, "dividend_policy": policy,
           "cash": {"min_cash_frac": float((cash / tv).min()), "days_negative": int((cash < 0).sum()),
                    "days": int(len(cash)), "mean_cash_frac": float((cash / tv).mean()),
                    "idle_cash_interest_at_dtb3_pp_per_yr": float(pos_int.sum() / years * 100),
                    "last3y_idle_cash_interest_pp_per_yr": float(pos_int.loc["2023-09-25":].sum() / 3.0 * 100),
                    "negative_cash_financing_pp_per_yr": float(neg_cost.sum() / years * 100)},
           "gross_exposure_max": float((res["portfolio_value"].astype(float) / tv).max())}
    tc.write_json(f"{key}_accounting.json", out)
    print(key, out, flush=True)


# ------------------------------------------------------------------ small account
def _whole_share_mixin(base_cls):
    class Whole(base_cls):
        def _entry_target_share_float(self, symbol_str, close_row_ser):
            adj = float(close_row_ser[(symbol_str, "Close")])
            raw = float(close_row_ser[(symbol_str, "Unadjusted Close")])
            target_value = float(self.previous_total_value) * self.target_weight_float
            raw_shares = int(np.floor(target_value / raw))
            self.zero_share_skips = getattr(self, "zero_share_skips", 0) + int(raw_shares == 0)
            return raw_shares * raw / adj
    Whole.__name__ = f"Whole{base_cls.__name__}"
    return Whole


def run_small(key: str) -> None:
    pricing = tc.load_pricing(key)
    base_cls = tc.MR_SPEC[key]["cls"]
    whole = _whole_share_mixin(base_cls)
    out = {"model_100k": tc.metric_pair(tc.run(key, pricing))}
    for label, cap, per_share, minimum in (("fixed_30k", 30_000.0, 0.005, 1.0), ("tiered_30k", 30_000.0, 0.0035, 0.35),
                                           ("fixed_100k", 100_000.0, 0.005, 1.0)):
        cfg = tc.config_for(key, capital_base_float=cap)
        strat = tc.make_strategy(key, cfg, whole)
        strat._commission_per_share = per_share
        strat._commission_minimum = minimum
        r = tc.run(key, pricing, cfg=cfg, hsu=True, strategy=strat)
        tx = r.get_transactions()
        out[label] = {**tc.metric_pair(r), "commission_total": float(tx["commission"].sum()),
                      "n_fills": int(len(tx)), "zero_share_skips": int(getattr(r, "zero_share_skips", 0)),
                      "orders_at_minimum_fee_share": float((tx["commission"] <= minimum + 1e-9).mean())}
    r10 = tc.run(key, pricing, cfg=tc.config_for(key, capital_base_float=10_000_000.0))
    out["model_10m"] = tc.metric_pair(r10)
    tc.write_json(f"{key}_small.json", out)
    print(key, {k: v["full"]["cagr_pct"] if isinstance(v, dict) and "full" in v else v for k, v in out.items()}, flush=True)


# ------------------------------------------------------------------ tradability
def run_trade(key: str) -> None:
    sys.path.insert(0, str(tc.HERE.parent / "common"))
    from tradability import load_turnover_ser, participation_table_df, summarize_participation_df

    pricing = tc.load_pricing(key)
    strat = tc.run(key, pricing)
    syms = tc.symbols_for(key)
    turnover = {s: load_turnover_ser(s, end_date_str=tc.END_STR) for s in syms}
    part = participation_table_df(strat, turnover)
    part.to_csv(tc.OUT / f"{key}_participation.csv", index=False)
    summ = summarize_participation_df(part, recent_start_str=tc.LAST3Y_START_STR)
    guard = {}
    soft, hard = (0.0025, 0.005) if key == "eom" else (0.0005, 0.001)
    for window, frame in (("full", part), ("last3y", part[part["bar"] >= pd.Timestamp(tc.LAST3Y_START_STR)])):
        for cap in (30_000, 1_000_000, 10_000_000):
            v = frame[f"part_{cap}"].replace([np.inf], np.nan)
            guard[f"{window}_{cap}"] = {"share_over_soft": float((v > soft).mean()), "share_over_hard": float((v > hard).mean())}
    # per-symbol recent liquidity and inception
    inst = {}
    for s in syms:
        t = turnover[s]
        t_obs = t[t > 0]
        inst[s] = {"first_turnover_date": str(t_obs.index[0].date()),
                   "median_turnover_last1y_usd": float(t.loc["2025-09-25":].median()),
                   "median_turnover_last3y_usd": float(t.loc[tc.LAST3Y_START_STR:].median()),
                   "min_20d_median_turnover_last3y_usd": float(t.loc[tc.LAST3Y_START_STR:].rolling(20).median().min()),
                   "last_unadjusted_close": float(pricing[(s, "Unadjusted Close")].dropna().iloc[-1])}
    # C2 whole shares at 30K (per-name weight error bound = price / NAV; zero-share names)
    if key == "eom":
        w = {"SPY": 1.0, "TLT": 1.0}
    else:
        w = {s: strat.target_weight_float for s in syms}
    c2 = {}
    for s in syms:
        px = pricing[(s, "Unadjusted Close")].astype(float).loc[tc.LAST3Y_START_STR:]
        c2[s] = {"target_weight": w[s], "max_one_share_pct_nav_30k": float(px.max() / 30_000 * 100),
                 "median_weight_error_pct_nav_30k": float(((w[s] * 30_000) % px).median() / 30_000 * 100)}
    out = {"participation_summary": summ.to_dict("records"), "guardrail": guard, "instruments": inst, "whole_share_30k": c2,
           "soft_hard_guardrail": [soft, hard]}
    tc.write_json(f"{key}_trade.json", out)
    print(key, summ.to_string(), flush=True)


# ------------------------------------------------------------------ float32 re-rounding after a future split (engine)
def run_tieflip(key: str) -> None:
    """Realistic vendor re-adjustment: the float32 history re-rounded by a future k:1 split.  Engine run with
    raw-unit fees (so only DECISION flips can move the result) vs the float32 reference."""
    import json

    pricing = tc.load_pricing(key)
    path = tc.OUT / f"{key}_split.json"
    if path.exists():
        split = json.loads(path.read_text(encoding="utf-8"))
    else:
        split = {"rows": [r for p in sorted(tc.OUT.glob(f"{key}_split_*.json"))
                          for r in json.loads(p.read_text(encoding="utf-8"))["rows"]]}
    ref = tc.run(key, pricing, hsu=True)
    ref_f, ref_m = tc.fills(ref), tc.metric_pair(ref)
    rows = []
    for r in split["rows"]:
        if r["f32_decision_cells_diff"] == 0:
            continue
        cand = tc.run(key, tc.rescale_symbol(pricing, [r["symbol"]], r["k"]), hsu=True)
        cmp = tc.compare_fill_sets(ref_f, tc.fills(cand), rel_tol=1e-6)
        m = tc.metric_pair(cand)
        rows.append({"symbol": r["symbol"], "k": r["k"], "f32_decision_cells_diff": r["f32_decision_cells_diff"],
                     "fills_only_ref": cmp["n_only_ref"], "fills_only_cand": cmp["n_only_cand"],
                     "d_cagr_full_pp": round(m["full"]["cagr_pct"] - ref_m["full"]["cagr_pct"], 4),
                     "d_sharpe_full": round(m["full"]["sharpe"] - ref_m["full"]["sharpe"], 4),
                     "d_cagr_last3y_pp": round(m["last3y"]["cagr_pct"] - ref_m["last3y"]["cagr_pct"], 4)})
        print(key, rows[-1], flush=True)
    worst = max(rows, key=lambda x: abs(x["d_cagr_full_pp"]), default=None)
    tc.write_json(f"{key}_tieflip.json", {"reference_hsu": ref_m, "rows": rows, "worst": worst})


if __name__ == "__main__":
    mode, key = sys.argv[1], sys.argv[2]
    if mode == "split" and len(sys.argv) > 3:
        run_split(key, sys.argv[3])
        sys.exit(0)
    fn = {"rowt": run_rowt, "split": run_split, "ties": run_ties, "padded": run_padded, "account": run_account,
          "small": run_small, "trade": run_trade, "tieflip": run_tieflip}[mode]
    fn(key)
