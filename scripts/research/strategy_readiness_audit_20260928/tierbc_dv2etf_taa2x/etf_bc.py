"""Industry-ETF DV2 backtest-correctness checks on REAL Norgate data (protocol A1-A10). Research-only.

Modes (``uv run python etf_bc.py <mode>``):
  runs      A9/A10/A8: baseline (production defaults), rerun (determinism), capital 30K/1M/10M, historical-share
            units (raw whole shares + raw-share fees), dividend/commission/cash ledger stats.
  rowt      A3 row-T truncation with the REAL loader: _load(end=T) -> history gate + compute_signals -> every feature
            dated <= T equals the full-history value, and the ranked entry list AT T (get_opportunities) equals the
            full-history list at T. >= 20 cut-offs (protocol set + entry/exit decision dates). A4 positive controls:
            one-session look-ahead in DV2 and in ADV63 (shift -1) must be caught on the cut-off row.
  prefix    A3b engine prefix: run truncated at T vs full run; fills <= T and order intents <= T-1 identical.
  inv       A2 future-split invariance: >= 6 ETFs (top traded + real splitters + reverse splitters) rescaled by
            k in {40, 0.1, 1.5}; production accounting and historical-share units. A4 control: the legacy E-01
            liquidity formula (Unadjusted Close x split-adjusted Volume) must be caught.
  diag      ADV-gate mechanics: invalid-window days, eligibility first dates, one-extra-session gate lag, candidate
            dropna causes, fills/decisions on padded (zero-Turnover) bars.
  splice    Pre-2012 research splice: corrected engine run from 2000 (history from 1998) vs the legacy splice source.
  hindsight Same frozen rules (native ADV63 > $50M) on the pre-declared 60-ETF scan and its groups.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

import etf_common as ec
from strategies.dv2 import strategy_mr_dv2_industry_etf as etf_module
from strategies.dv2.strategy_mr_dv2_liquidity_floor import get_asof_universe_symbol_list

FEATURES = ["p126d_return", "natr", "dv2", "sma_200", "adv_63", "raw_price"]


def _dump(name: str, obj) -> None:
    (ec.OUT / name).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


# ------------------------------------------------------------------------------------------------ runs
def mode_runs() -> None:
    pricing, universe = ec.load_cached()
    out = {}
    base = ec.run(ec.make(universe), pricing)
    rerun = ec.run(ec.make(universe), pricing)
    nav = base.results["total_value"].astype(float)
    out["baseline"] = ec.metrics(nav)
    out["baseline_to_2026_08_19"] = ec.metrics(nav, end="2026-08-19")
    out["baseline_last3y"] = ec.metrics(nav, start="2023-09-25")
    out["determinism_bit_identical"] = bool(np.array_equal(nav.to_numpy(), rerun.results["total_value"].astype(float).to_numpy()))
    tx = base.get_transactions()
    out["fills"] = int(len(tx))
    out["commission_total"] = float(tx["commission"].sum())
    out["dividend_net_total"] = float(getattr(base, "dividend_cash_net_total_float", np.nan))
    cash_frac = base.results["cash"].astype(float) / nav
    out["min_cash_frac_of_nav"] = float(cash_frac.min())
    out["share_days_negative_cash"] = float((cash_frac < 0).mean())
    out["mean_cash_frac_of_nav"] = float(cash_frac.mean())
    out["accounting_policy"] = {k: str(v) for k, v in getattr(base, "_accounting_policy_dict", {}).items()}
    base.get_transactions().to_csv(ec.OUT / "baseline_transactions.csv", index=False)
    nav.to_csv(ec.OUT / "baseline_nav.csv")
    ret100 = nav.pct_change().dropna()
    for cap in (30_000.0, 1_000_000.0, 10_000_000.0):
        for hsu in (False, True):
            s = ec.run(ec.make(universe, capital=cap, hsu=hsu), pricing)
            v = s.results["total_value"].astype(float)
            r = v.pct_change().dropna()
            key = f"cap_{int(cap)}_{'hsu' if hsu else 'engine'}"
            out[key] = {**ec.metrics(v), "last3y": ec.metrics(v, start="2023-09-25"),
                        "fills": int(len(s.get_transactions())),
                        "commission_bp_of_avg_nav_per_yr": float(s.get_transactions()["commission"].sum() / v.mean()
                                                                  / ((v.index[-1] - v.index[0]).days / 365.25) * 1e4),
                        "daily_ret_corr_vs_100k_engine": float(r.corr(ret100)),
                        "zero_share_cancels": int(sum(len(x["orders"]) for x in s.decision_log) - len(s.get_transactions()) // 1)}
            if hsu and cap == 1_000_000.0:
                s.get_transactions().to_csv(ec.OUT / "hsu_1m_transactions.csv", index=False)
            print(key, json.dumps({k: v for k, v in out[key].items() if k != "last3y"}), flush=True)
    s = ec.run(ec.make(universe, hsu=True), pricing)
    out["cap_100000_hsu"] = {**ec.metrics(s.results["total_value"]), "fills": int(len(s.get_transactions())),
                             "commission_total": float(s.get_transactions()["commission"].sum())}
    print(json.dumps({k: v for k, v in out.items() if k != "accounting_policy"}, default=str)[:3000], flush=True)
    _dump("runs.json", out)


# ------------------------------------------------------------------------------------------------ row-T
class DV2Leak(ec.RecEtf):
    """Positive control: the DV2 entry filter reads DV2_(T+1)."""

    def compute_signals(self, pricing_data):
        out = super().compute_signals(pricing_data)
        for s in out.columns.get_level_values(0).unique():
            if (s, "dv2") in out.columns:
                out[(s, "dv2")] = out[(s, "dv2")].shift(-1)  # *** CRITICAL*** planted look-ahead
        return out


class AdvLeak(ec.RecEtf):
    """Positive control: the $50M liquidity gate reads ADV63_(T+1)."""

    def compute_signals(self, pricing_data):
        out = super().compute_signals(pricing_data)
        for s in out.columns.get_level_values(0).unique():
            if (s, "adv_63") in out.columns:
                out[(s, "adv_63")] = out[(s, "adv_63")].shift(-1)  # *** CRITICAL*** planted look-ahead
        return out


def _entry_list_at(cls, signal_df: pd.DataFrame, universe: pd.DataFrame, t: pd.Timestamp) -> list[str]:
    strategy = ec.make(universe, cls=cls)
    strategy.previous_bar = t
    return strategy.get_opportunities(signal_df.loc[t])


def _cutoffs(pricing: pd.DataFrame) -> dict[str, str]:
    tx = pd.read_csv(ec.OUT / "baseline_transactions.csv", parse_dates=["bar"])
    idx = pricing.index
    decision_of = {b: idx[idx.get_loc(b) - 1] for b in tx["bar"].unique()}
    buys = sorted({decision_of[b] for b in tx.loc[tx["amount"] > 0, "bar"].unique()})
    sells = sorted({decision_of[b] for b in tx.loc[tx["amount"] < 0, "bar"].unique()})
    rng = np.random.default_rng(20260928)
    pick_buy = sorted(rng.choice(np.array(buys, dtype="datetime64[ns]"), 10, replace=False))
    pick_sell = sorted(rng.choice(np.array(sells, dtype="datetime64[ns]"), 4, replace=False))
    cut = {
        "mid_month_2026-09-15": "2026-09-15",
        "month_end_weekend_2026-05-29": "2026-05-29",
        "month_end_weekend_2025-08-29": "2025-08-29",
        "month_end_before_holiday_2025-12-31": "2025-12-31",
        "first_session_2026-09-01": "2026-09-01",
        "last_completed_month_end_2026-08-31": "2026-08-31",
        "current_partial_month_2026-09-25": "2026-09-25",
        "half_day_2024-11-29": "2024-11-29",
        "stress_2020-03-16": "2020-03-16",
    }
    for d in pick_buy:
        cut[f"entry_decision_{pd.Timestamp(d).date()}"] = pd.Timestamp(d).date().isoformat()
    for d in pick_sell:
        cut[f"exit_decision_{pd.Timestamp(d).date()}"] = pd.Timestamp(d).date().isoformat()
    return cut


def mode_rowt() -> None:
    pricing, universe = ec.load_cached()
    full_sig = {cls.__name__: ec.make(universe, cls=cls).compute_signals(pricing.copy()) for cls in (ec.RecEtf, DV2Leak, AdvLeak)}
    rows = []
    for label, cut in _cutoffs(pricing).items():
        t = pd.Timestamp(cut)
        t_pricing, t_universe = ec.load_full(end_date_str=cut)  # REAL loader truncation (and a rebuilt history gate)
        uni_equal = bool(t_universe.equals(universe.loc[:t].astype(t_universe.dtypes.iloc[0])))
        price_equal = bool(np.allclose(t_pricing.to_numpy(dtype=float), pricing.loc[:t, t_pricing.columns].to_numpy(dtype=float), equal_nan=True, rtol=0, atol=0))
        rec = {"case": label, "cutoff": cut, "loader_rows_equal_full_prefix": price_equal, "history_gate_equal": uni_equal}
        for cls in (ec.RecEtf, DV2Leak, AdvLeak):
            trunc_sig = ec.make(t_universe, cls=cls).compute_signals(t_pricing.copy())
            cols = [c for c in trunc_sig.columns if c[1] in FEATURES]
            a = full_sig[cls.__name__].loc[:t, cols].to_numpy(dtype=float)
            b = trunc_sig.loc[:, cols].to_numpy(dtype=float)
            n_bad = int((np.isnan(a) != np.isnan(b)).sum() + (np.where(np.isnan(a) | np.isnan(b), 0, np.abs(a - b)) > 0).sum())
            ra, rb = a[-1], b[-1]
            n_bad_row = int((np.isnan(ra) != np.isnan(rb)).sum() + (np.where(np.isnan(ra) | np.isnan(rb), 0, np.abs(ra - rb)) > 0).sum())
            full_list = _entry_list_at(cls, full_sig[cls.__name__], universe, t)
            trunc_list = _entry_list_at(cls, trunc_sig, t_universe, t)
            rec[cls.__name__] = {"feature_values_compared": int(a.size), "feature_values_different": n_bad,
                                 "cutoff_row_values_different": n_bad_row, "entry_list_full": full_list,
                                 "entry_list_truncated": trunc_list, "entry_list_equal": full_list == trunc_list}
        rec["passed"] = (rec["RecEtf"]["feature_values_different"] == 0 and rec["RecEtf"]["entry_list_equal"]
                         and price_equal and uni_equal)
        rec["dv2_leak_caught"] = rec["DV2Leak"]["cutoff_row_values_different"] > 0 or not rec["DV2Leak"]["entry_list_equal"]
        rec["adv_leak_caught"] = rec["AdvLeak"]["cutoff_row_values_different"] > 0 or not rec["AdvLeak"]["entry_list_equal"]
        rec["dv2_leak_changes_entry_list"] = not rec["DV2Leak"]["entry_list_equal"]
        rows.append(rec)
        print(label, rec["passed"], rec["dv2_leak_caught"], rec["adv_leak_caught"], rec["dv2_leak_changes_entry_list"],
              rec["RecEtf"]["entry_list_full"], flush=True)
    summary = {"cases": len(rows), "passed": sum(r["passed"] for r in rows),
               "dv2_leak_caught": sum(r["dv2_leak_caught"] for r in rows),
               "adv_leak_caught": sum(r["adv_leak_caught"] for r in rows),
               "dv2_leak_changes_entry_list": sum(r["dv2_leak_changes_entry_list"] for r in rows),
               "cases_with_nonempty_entry_list": sum(bool(r["RecEtf"]["entry_list_full"]) for r in rows)}
    print(summary, flush=True)
    _dump("rowt.json", {"summary": summary, "rows": rows})


# ------------------------------------------------------------------------------------------------ prefix
def mode_prefix() -> None:
    pricing, universe = ec.load_cached()
    start = "2024-01-02"
    ref = ec.run(ec.make(universe), pricing, start=start)
    rows = []
    for cut in ("2024-11-29", "2025-04-08", "2025-08-29", "2025-12-31", "2026-05-29", "2026-08-31", "2026-09-01",
                "2026-09-15", "2026-09-24"):
        t = pd.Timestamp(cut)
        prev_t = pricing.index[pricing.index.get_loc(t) - 1]
        t_pricing, t_universe = ec.load_full(end_date_str=cut)
        cand = ec.run(ec.make(t_universe), t_pricing, start=start, end=cut)
        f = ec.compare(ec.fills(ref), ec.fills(cand), t, rel_tol=1e-9)
        o = ec.compare(ec.order_intents(ref), ec.order_intents(cand), prev_t, rel_tol=1e-9, key=("date", "asset", "kind"), value_col="weight")
        ok = all(x["n_only_reference"] == 0 and x["n_only_candidate"] == 0 and x["n_value_beyond_tol"] == 0 for x in (f, o))
        rows.append({"cutoff": cut, "passed": ok, "fills": f, "orders": o})
        print(cut, ok, f["n_reference"], o["n_reference"], flush=True)
    _dump("prefix.json", {"passed": sum(r["passed"] for r in rows), "cases": len(rows), "rows": rows})


# ------------------------------------------------------------------------------------------------ invariance
class LegacyAdv(ec.RecEtf):
    """A4 control: E-01 legacy liquidity (Unadjusted Close x split-adjusted Volume), scale-dependent by construction."""

    def compute_signals(self, pricing_data):
        out = super().compute_signals(pricing_data)
        for s in out.columns.get_level_values(0).unique():
            if (s, "adv_63") in out.columns:
                legacy = out[(s, "Unadjusted Close")].astype(float) * out[(s, "Volume")].astype(float)
                out[(s, "adv_63")] = legacy.where(legacy > 0).rolling(63, min_periods=63).mean()
        return out


def mode_inv() -> None:
    pricing, universe = ec.load_cached()
    ref = {hsu: ec.run(ec.make(universe, hsu=hsu), pricing) for hsu in (False, True)}
    buys = ec.fills(ref[False])
    top = buys[buys["side"] > 0]["asset"].value_counts().index[:4].tolist()
    held_end = sorted(str(s) for s, q in ref[False].get_positions().items() if q > 0)
    splitters = ["XBI", "IBB", "SOXX", "IGV", "OIH", "XOP"]  # 3:1, 3:1, 3:1, 5:1, 1:20 reverse, 1:4 reverse
    chosen = list(dict.fromkeys(top + held_end + splitters))
    info = {"rescaled": chosen, "held_at_end": held_end, "top_traded": top,
            "rescaled_traded": sorted(set(chosen) & set(buys["asset"].astype(str)))}
    rows = []
    for hsu in (False, True):
        for k in (40.0, 0.1, 1.5):
            cand_pricing = pricing
            for s in chosen:
                cand_pricing = ec.rescale_symbol_history(cand_pricing, s, k)
            cand = ec.run(ec.make(universe, hsu=hsu), cand_pricing)
            f = ec.compare(ec.fills(ref[hsu]), ec.fills(cand), rel_tol=0.35 if not hsu else 1e-7)
            o = ec.compare(ec.order_intents(ref[hsu]), ec.order_intents(cand), rel_tol=1e-9, key=("date", "asset", "kind"), value_col="weight")
            ok = all(x["n_only_reference"] == 0 and x["n_only_candidate"] == 0 for x in (f, o)) and o["n_value_beyond_tol"] == 0
            nav_diff = float((cand.results["total_value"].astype(float) / ref[hsu].results["total_value"].astype(float) - 1).abs().max())
            rows.append({"accounting": "hsu" if hsu else "engine", "k": k, "passed": ok, "fills": f, "orders": o,
                         "max_rel_nav_diff": nav_diff})
            print("inv", "hsu" if hsu else "engine", k, ok, f["max_rel_value_diff"], nav_diff, flush=True)
    pc_ref = ec.run(ec.make(universe, cls=LegacyAdv), pricing)
    pc_rows = []
    for k in (40.0, 0.1):
        cand_pricing = pricing
        for s in chosen:
            cand_pricing = ec.rescale_symbol_history(cand_pricing, s, k)
        pc_cand = ec.run(ec.make(universe, cls=LegacyAdv), cand_pricing)
        o = ec.compare(ec.order_intents(pc_ref), ec.order_intents(pc_cand), rel_tol=1e-9, key=("date", "asset", "kind"), value_col="weight")
        caught = o["n_only_reference"] + o["n_only_candidate"] > 0
        pc_rows.append({"k": k, "caught": caught, "orders": o})
        print("PC legacy ADV k", k, "caught", caught, o["n_only_reference"], o["n_only_candidate"], flush=True)
    _dump("invariance.json", {"info": info, "rows": rows, "positive_control_legacy_adv": pc_rows,
                              "passed": sum(r["passed"] for r in rows), "cases": len(rows)})


# ------------------------------------------------------------------------------------------------ diag
def mode_diag() -> None:
    pricing, universe = ec.load_cached()
    sig = ec.make(universe).compute_signals(pricing.copy())
    start = pd.Timestamp(ec.START)
    out = {"per_symbol": {}}
    total_flip = 0
    for s in etf_module.INDUSTRY_ETF_SYMBOL_TUPLE:
        turn = pricing[(s, "Turnover")].astype(float)
        vol = pricing[(s, "Volume")].astype(float)
        close = pricing[(s, "Close")]
        live = close.notna()
        zero_turn = live & ~(turn > 0)
        adv = sig[(s, "adv_63")]
        gate = adv > etf_module.MIN_ADV_DOLLAR_FLOAT
        gate_lag = adv.shift(1) > etf_module.MIN_ADV_DOLLAR_FLOAT
        rule = (sig[(s, "dv2")] < 10) & (close > sig[(s, "sma_200")]) & (sig[(s, "p126d_return")] > 0.05) & (universe[s] == 1)
        in_bt = sig.index >= start
        flip = in_bt & rule & (gate != gate_lag)
        total_flip += int(flip.sum())
        adv_check = turn.where(turn > 0).rolling(63, min_periods=63).mean()
        out["per_symbol"][s] = {
            "first_bar": str(close.first_valid_index().date()),
            "zero_or_nan_turnover_sessions_2012plus": int((zero_turn & in_bt).sum()),
            "zero_turnover_dates_2012plus": [d.date().isoformat() for d in turn.index[zero_turn & in_bt]][:10],
            "zero_volume_sessions_2012plus": int((live & ~(vol > 0) & in_bt).sum()),
            "sessions_adv_invalid_after_backtest_start": int((adv.isna() & in_bt & live).sum()),
            "first_adv_gate_pass": str(adv.index[gate.to_numpy()][0].date()) if gate.any() else None,
            "share_2012plus_sessions_gate_pass": float(gate[in_bt].mean()),
            "adv63_formula_max_abs_diff": float((adv - adv_check).abs().max()),
            "gate_lag1_flips_on_rule_days": int(flip.sum()),
            "median_adv63_last1y_musd": float(adv.loc["2025-09-25":].median() / 1e6),
            "median_adv63_last3y_musd": float(adv.loc["2023-09-25":].median() / 1e6),
        }
    out["gate_lag1_flips_total_on_rule_days"] = total_flip
    # candidate dropna causes on member-days (production get_opportunities drops ANY NaN column)
    cause = {"rows": 0, "dropped": 0, "dropped_only_non_feature_nan": 0}
    for t in sig.index[sig.index >= start]:
        row = sig.loc[t].unstack()
        row = row[~row.index.astype(str).str.startswith("$")]
        members = get_asof_universe_symbol_list(universe, t)
        row = row[row.index.isin(members)]
        cause["rows"] += len(row)
        nan_mask = row.isna()
        dropped = nan_mask.any(axis=1)
        cause["dropped"] += int(dropped.sum())
        feat_nan = nan_mask[[c for c in FEATURES + ["Close", "High", "Low"] if c in nan_mask.columns]].any(axis=1)
        cause["dropped_only_non_feature_nan"] += int((dropped & ~feat_nan).sum())
    out["candidate_dropna"] = cause
    # fills on padded / zero-turnover bars
    tx = pd.read_csv(ec.OUT / "baseline_transactions.csv", parse_dates=["bar"])
    pad = 0
    for _, r in tx.iterrows():
        if not (float(pricing.loc[r["bar"], (r["asset"], "Turnover")]) > 0):
            pad += 1
    out["fills_on_zero_turnover_bar"] = pad
    idx = pricing.index
    dec = [idx[idx.get_loc(b) - 1] for b in tx["bar"]]
    out["decisions_on_zero_turnover_bar"] = int(sum(not (float(pricing.loc[d, (a, "Turnover")]) > 0) for d, a in zip(dec, tx["asset"])))
    print(json.dumps({k: v for k, v in out.items() if k != "per_symbol"}, default=str), flush=True)
    _dump("diag.json", out)


# ------------------------------------------------------------------------------------------------ splice
def mode_splice() -> None:
    pricing, universe = ec.load_full(end_date_str=ec.STUDY_END, history_start_str="1998-01-01")
    s = ec.run(ec.make(universe), pricing, start="2000-01-03")
    nav = s.results["total_value"].astype(float)
    legacy = pd.read_csv(ec.REPO.parents[2] / "results/research/dv2_deep_20260925/sources/etf_ind_adv50__path.csv.gz",
                         index_col="date", parse_dates=True)["total_value_float"].astype(float)
    cut = pd.Timestamp("2012-01-03")
    r_new, r_old = nav.loc[:cut].pct_change().dropna(), legacy.loc[:cut].pct_change().dropna()
    common = r_new.index.intersection(r_old.index)
    base_nav = pd.read_csv(ec.OUT / "baseline_nav.csv", index_col=0, parse_dates=True).iloc[:, 0]
    post = nav.loc["2012-01-04":].pct_change().dropna()
    base_r = base_nav.pct_change().dropna()
    cpost = post.index.intersection(base_r.index)
    out = {"pre2012_corrected_engine": ec.metrics(nav, end=cut), "pre2012_legacy_splice": ec.metrics(legacy, end=cut),
           "pre2012_daily_corr": float(r_new.loc[common].corr(r_old.loc[common])),
           "pre2012_fills": int((pd.to_datetime(s.get_transactions()["bar"]) <= cut).sum()),
           "post2012_2000start_vs_2009start_max_abs_daily_ret_diff": float((post.loc[cpost] - base_r.loc[cpost]).abs().max()),
           "post2012_2000start_vs_2009start_corr": float(post.loc[cpost].corr(base_r.loc[cpost])),
           "full_2000_2026": ec.metrics(nav)}
    print(json.dumps(out, default=str), flush=True)
    _dump("splice.json", out)


# ------------------------------------------------------------------------------------------------ hindsight
GROUPS = {
    "sectors": "XLB XLE XLF XLI XLK XLP XLU XLV XLY XLRE XLC".split(),
    "industries": list(etf_module.INDUSTRY_ETF_SYMBOL_TUPLE),
    "countries": "EWJ EWG EWU EWC EWA EWZ EWH EWT EWY EWW EWS EWP EWQ EWI EWL EWN FXI INDA EZA EEM EFA".split(),
    "broad": "SPY QQQ IWM DIA MDY IWD IWF IWN IWO".split(),
}


def mode_hindsight() -> None:
    all_syms = [s for g in GROUPS.values() for s in g]
    pricing = etf_module.get_prices(all_syms, ["$SPX"], start_date=etf_module.DEFAULT_HISTORY_START_DATE_STR, end_date=ec.STUDY_END)
    universe_all = etf_module.build_history_universe_df(pricing)
    out = {}
    for name, syms in [("all_60", all_syms)] + list(GROUPS.items()):
        u = universe_all.copy()
        u.loc[:, [c for c in u.columns if c not in syms]] = 0
        s = ec.run(ec.make(u), pricing)
        nav = s.results["total_value"].astype(float)
        out[name] = {**ec.metrics(nav), "last3y": ec.metrics(nav, start="2023-09-25"), "fills": int(len(s.get_transactions())),
                     "symbols_loaded": int(sum(c in pricing.columns.get_level_values(0) for c in syms))}
        print(name, json.dumps(out[name]), flush=True)
    _dump("hindsight_60etf.json", out)


if __name__ == "__main__":
    t0 = time.time()
    {"runs": mode_runs, "rowt": mode_rowt, "prefix": mode_prefix, "inv": mode_inv, "diag": mode_diag,
     "splice": mode_splice, "hindsight": mode_hindsight, "extra": lambda: None}[sys.argv[1]]()
    print("elapsed_s", round(time.time() - t0, 1), flush=True)


# ------------------------------------------------------------------------------------------------ extra: gate lag + idle cash
class AdvLag1(ec.RecEtf):
    """Sensitivity: the $50M gate reads ADV63_(T-1) (one extra session of lag on the only non-price-like input)."""

    def compute_signals(self, pricing_data):
        out = super().compute_signals(pricing_data)
        for s in out.columns.get_level_values(0).unique():
            if (s, "adv_63") in out.columns:
                out[(s, "adv_63")] = out[(s, "adv_63")].shift(1)
        return out


def mode_extra() -> None:
    pricing, universe = ec.load_cached()
    base_nav = pd.read_csv(ec.OUT / "baseline_nav.csv", index_col=0, parse_dates=True).iloc[:, 0].astype(float)
    lag = ec.run(ec.make(universe, cls=AdvLag1), pricing)
    lag_nav = lag.results["total_value"].astype(float)
    base = ec.run(ec.make(universe), pricing)
    cash_frac = (base.results["cash"].astype(float) / base.results["total_value"].astype(float))
    dtb3 = pd.read_csv(ec.OUT.parent / "DTB3_tierbc_cache.csv", index_col=0, parse_dates=True).iloc[:, 0]
    dtb3 = pd.to_numeric(dtb3, errors="coerce").reindex(cash_frac.index).ffill() / 100.0
    extra_daily = cash_frac.clip(lower=0) * dtb3 / 252.0
    base_ret = base_nav.pct_change().fillna(0.0)
    with_int = (1 + base_ret + extra_daily.reindex(base_ret.index).fillna(0.0)).cumprod() * base_nav.iloc[0]
    out = {"baseline": ec.metrics(base_nav), "adv_gate_lag1": ec.metrics(lag_nav),
           "adv_gate_lag1_delta_cagr_pp": 100 * (ec.metrics(lag_nav)["cagr"] - ec.metrics(base_nav)["cagr"]),
           "adv_gate_lag1_fills": int(len(lag.get_transactions())),
           "mean_cash_frac": float(cash_frac.mean()),
           "idle_cash_at_dtb3_bound": ec.metrics(with_int),
           "idle_cash_at_dtb3_delta_cagr_pp": 100 * (ec.metrics(with_int)["cagr"] - ec.metrics(base_nav)["cagr"]),
           "mean_gross_exposure": float(1 - cash_frac.mean())}
    print(json.dumps(out), flush=True)
    _dump("extra_advlag_idlecash.json", out)


if __name__ == "__main__" and sys.argv[1] == "extra":
    mode_extra()
