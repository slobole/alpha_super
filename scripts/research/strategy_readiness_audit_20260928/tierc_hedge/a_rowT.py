"""A3 row-T truncation invariance + A4 planted one-session leak, for CTC, VIXM and Trinity (real Norgate data).

For every cut-off T:
  prefix = the strategy's REAL loader called with end_date_str=T (loader cut-offs) or the full frame sliced to T
           (slice cut-offs, many more dates)
  full   = the full-history frame (last bar 2026-09-25)
  decision_T(prefix) must equal decision_T(full) for every price-derived feature / decision object at row T.

Calendar-only fields are compared separately against the XNYS calendar (they depend on knowing that T is the last
session of its month, which a truncated frame cannot see; the backtest precomputes them on the full frame):
  CTC   month_end_decision_bool
  TRIN  base_weight_ser (attached to the session before next month's first session) and monthly_rebalance_bool

Positive control (A4): the same harness is run with a leaked signal that reads Close_(T+1) (a shift(-1) of the
signal input). The harness must report a mismatch at (almost) every cut-off.

Usage: python a_rowT.py [ctc|vixm|trin|all]
"""
from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np
import pandas as pd

import tc_common as c

MODE = sys.argv[1] if len(sys.argv) > 1 else "all"
XNYS = c.xnys_sessions("2000-01-01", "2026-12-31")


def xnys_month_end_bool(ts: pd.Timestamp) -> bool:
    pos = XNYS.get_loc(ts)
    return XNYS[pos + 1].to_period("M") != ts.to_period("M")


def pick_cutoffs(index: pd.DatetimeIndex, extra: list[str]) -> list[pd.Timestamp]:
    idx = pd.DatetimeIndex(index)
    recent = idx[idx >= pd.Timestamp("2024-09-01")]
    per = recent.to_period("M")
    s = pd.Series(recent, index=recent)
    month_ends = s.groupby(per).max().tolist()
    firsts = s.groupby(per).min().tolist()
    mids = [s[per == p].iloc[len(s[per == p]) // 2] for p in sorted(set(per))]
    out = set(month_ends[:-1]) | set(firsts[-12:]) | set(mids[-12:]) | {idx[-1]}
    for e in extra:
        t = pd.Timestamp(e)
        if t in idx:
            out.add(t)
    return sorted(out)


SPECIAL = ["2025-05-30", "2025-08-29", "2024-12-31", "2025-12-31", "2025-11-26", "2025-07-03", "2024-11-29",
           "2008-10-31", "2008-09-15", "2020-03-16", "2020-03-31", "2022-06-30", "2024-08-02", "2024-08-05",
           "2025-04-04", "2025-04-08", "2025-04-30"]


def compare_rows(a: pd.Series, b: pd.Series, atol=1e-12, rtol=1e-10) -> tuple[bool, float, list]:
    a = a.astype(float)
    b = b.astype(float)
    both_nan = a.isna() & b.isna()
    diff = (a - b).abs()
    tol = atol + rtol * b.abs()
    bad = ~both_nan & ~(diff <= tol)
    return (not bool(bad.any())), float(diff[~both_nan].max() if (~both_nan).any() else 0.0), list(a.index[bad][:6])


# ---------------------------------------------------------------- CTC
def ctc_features(pricing_df: pd.DataFrame, leak: bool = False) -> pd.DataFrame:
    s = c.ctc.CrisisTrendCoreStrategy()
    if leak:
        pdf = pricing_df.copy()
        for a in c.ctc.TRADEABLE_ASSET_TUPLE:
            key = (c.ctc.signal_namespace_str(a), "Close")
            # planted leak: read Close_(T+1) where it exists, else Close_T (no NaN, so nothing fails loud)
            pdf[key] = pdf[key].shift(-1).fillna(pdf[key])
        sig = s.compute_signals(pdf)
    else:
        sig = s.compute_signals(pricing_df)
    feat_cols = [col for col in sig.columns if col[0].startswith(c.ctc.SIGNAL_NAMESPACE_PREFIX_STR)
                 and col[1] != "Close"]
    feat_cols += [(c.ctc.PORTFOLIO_NAMESPACE_STR, c.ctc.RAW_POD_RETURN_FIELD_STR),
                  (c.ctc.PORTFOLIO_NAMESPACE_STR, c.ctc.EXANTE_VOLATILITY_FIELD_STR)]
    out = sig.loc[:, feat_cols].astype(float)
    out[("CAL", "month_end")] = sig[(c.ctc.PORTFOLIO_NAMESPACE_STR, c.ctc.MONTH_END_FIELD_STR)].astype(float)
    return out


def ctc_prefix_loader(T: pd.Timestamp) -> pd.DataFrame:
    cfg = replace(c.ctc.DEFAULT_CONFIG, end_date_str=T.date().isoformat())
    df = c.ctc.get_crisis_trend_core_data(cfg)
    out = df.loc[df.index >= pd.Timestamp(c.CTC_WORKAROUND_START_STR)].copy()
    out.attrs.update(df.attrs)
    return out


# ---------------------------------------------------------------- VIXM
def vixm_features(pricing_df: pd.DataFrame, leak: bool = False) -> pd.DataFrame:
    s = c.vixm.VixmBackwardationStrategy()
    pdf = pricing_df
    if leak:
        pdf = pricing_df.copy()
        for f in ("vix_close_float", "vix3m_close_float"):
            key = (c.vixm.VIX_SIGNAL_NAMESPACE_STR, f)
            pdf[key] = pdf[key].shift(-1).fillna(pdf[key])
    sig = s.compute_signals(pdf)
    return sig.loc[:, [(c.vixm.VIX_SIGNAL_NAMESPACE_STR, c.vixm.STATE_FIELD_STR),
                       (c.vixm.VIX_SIGNAL_NAMESPACE_STR, c.vixm.TERM_RATIO_FIELD_STR)]].astype(float)


def vixm_prefix_loader(T: pd.Timestamp) -> pd.DataFrame:
    return c.vixm.get_vixm_backwardation_data(replace(c.vixm.DEFAULT_CONFIG, end_date_str=T.date().isoformat()))


# ---------------------------------------------------------------- TRINITY
def trin_decision(pricing_df: pd.DataFrame, T: pd.Timestamp, leak: bool = False) -> pd.Series:
    """Decision objects at Close_T. Features are computed on the frame passed in (a prefix ending at T, or the full
    history), then row T / rows <= T are selected; the base-weight month is chosen with the XNYS calendar."""
    pdf = pricing_df
    risk = list(c.trin.RISK_ASSET_TUPLE)
    close_df = pdf.loc[:, [(a, "Close") for a in risk]].astype(float)
    if leak:
        close_df = close_df.shift(-1).fillna(close_df)
    ret_df, vol_df, me_w = c.b6040.compute_month_end_inverse_vol_weight_df(close_df, 63)
    # base weights in effect at Close_T: T's own month if T is the XNYS month-end session, else the prior month.
    month_label = T.to_period("M") if xnys_month_end_bool(T) else (T.to_period("M") - 1)
    lab = month_label.to_timestamp(how="end").normalize()
    me_w.index = me_w.index.normalize()
    if lab not in me_w.index:
        return pd.Series(dtype=float)
    base_w = me_w.loc[lab, risk].astype(float)
    base_w.index = risk
    rr = ret_df.loc[:T, risk].astype(float).iloc[-63:]
    base_ret = rr.mul(base_w, axis=1).sum(axis=1)
    m = c.b6040.compute_gross_exposure_float(base_ret, 63, 0.08, 0.085)
    tgt = c.trin.build_target_weight_ser(base_w, m, risk, "BIL")
    out = pd.concat([tgt.add_prefix("w_"), pd.Series({"m": m}), vol_df.loc[T].add_prefix("vol_"),
                     ret_df.loc[T].add_prefix("ret_")])
    return out.astype(float)


def trin_engine_base_weight_row(full_sig: pd.DataFrame, T: pd.Timestamp) -> pd.Series:
    return pd.Series({a: float(full_sig.loc[T, (a, "base_weight_ser")]) for a in c.trin.RISK_ASSET_TUPLE})


def run_ctc():
    full = c.load_ctc_workaround()
    F = ctc_features(full)
    FL = ctc_features(full, leak=True)
    cuts = pick_cutoffs(full.index, SPECIAL)
    loader_cuts = set(cuts[::5]) | {full.index[-1], pd.Timestamp("2025-05-30"), pd.Timestamp("2024-12-31")}
    rows = []
    for T in cuts:
        src = "slice"
        if T in loader_cuts:
            P = ctc_features(ctc_prefix_loader(T))
            src = "loader"
        else:
            P = ctc_features(full.loc[:T])
        cols = [col for col in F.columns if col[0] != "CAL"]
        ok, mx, bad = compare_rows(P.loc[T, cols], F.loc[T, cols])
        # leak control: leaked prefix vs leaked full
        if T != full.index[-1]:
            PL = ctc_features(full.loc[:T], leak=True)
            okL, _, _ = compare_rows(PL.loc[T, cols], FL.loc[T, cols])
        else:
            okL = None
        cal_prefix = bool(P.loc[T, ("CAL", "month_end")])
        cal_full = bool(F.loc[T, ("CAL", "month_end")])
        rows.append({"T": str(T.date()), "src": src, "row_T_equal": ok, "max_abs_diff": mx,
                     "bad_cols": [str(b) for b in bad], "leak_caught": (None if okL is None else (not okL)),
                     "month_end_flag_prefix": cal_prefix, "month_end_flag_full": cal_full,
                     "xnys_month_end": xnys_month_end_bool(T)})
    return rows


def run_vixm():
    full = c.load_vixm()
    F = vixm_features(full)
    FL = vixm_features(full, leak=True)
    state = F.iloc[:, 0]
    changes = list(state.index[state.diff().abs() > 0])
    cuts = pick_cutoffs(full.index, SPECIAL + [str(t.date()) for t in changes[-30:]])
    loader_cuts = set(cuts[::5]) | {full.index[-1]}
    rows = []
    for T in cuts:
        src = "slice"
        if T in loader_cuts:
            P = vixm_features(vixm_prefix_loader(T))
            src = "loader"
        else:
            P = vixm_features(full.loc[:T])
        ok, mx, bad = compare_rows(P.loc[T], F.loc[T])
        PL = vixm_features(full.loc[:T], leak=True)
        okL, _, _ = compare_rows(PL.loc[T], FL.loc[T])
        rows.append({"T": str(T.date()), "src": src, "row_T_equal": ok, "max_abs_diff": mx,
                     "bad_cols": [str(b) for b in bad], "leak_caught": (not okL) if T != full.index[-1] else None,
                     "state_full": float(F.loc[T].iloc[0]), "state_leak_full": float(FL.loc[T].iloc[0])})
    return rows


def run_trin():
    full = c.load_trin()
    s = c.trin.TrinityVolControlStrategy(name="x", benchmarks=["$SPX"])
    full_sig = s.compute_signals(full)
    idx = full.index[full.index >= pd.Timestamp("2007-09-01")]
    cuts = pick_cutoffs(idx, SPECIAL)
    # daily strategy: add 40 more sessions spread over 2008-2026 including engine rebalance days
    import pickle
    base = pickle.load(open(c.CACHE / "trin_baseline.pkl", "rb"))
    tx_days = sorted(set(pd.to_datetime(base["tx"]["bar"])))
    prev = {d: idx[idx.get_loc(d) - 1] for d in tx_days if d in idx and idx.get_loc(d) > 0}
    rng = np.random.default_rng(20260928)
    extra = list(rng.choice(np.array(sorted(prev.values()), dtype="datetime64[ns]"), size=40, replace=False))
    cuts = sorted(set(cuts) | {pd.Timestamp(x) for x in extra})
    loader_cuts = set(cuts[::6]) | {full.index[-1]}
    rows = []
    for T in cuts:
        src = "slice"
        if T in loader_cuts:
            cfg = replace(c.trin.DEFAULT_CONFIG, end_date_str=T.date().isoformat())
            P_df = c.b6040.get_beyond_6040_data(config=cfg)
            src = "loader"
        else:
            P_df = full.loc[:T]
        dP = trin_decision(P_df, T)
        dF = trin_decision(full, T)  # features on the full history, row T selected
        ok, mx, bad = compare_rows(dP, dF)
        # engine-equivalence: the base weights that the engine reads at row T (full compute_signals) must equal
        # the calendar-based weights the decision uses.
        bw_engine = trin_engine_base_weight_row(full_sig, T)
        bw_dec = dF[[f"w_{a}" for a in c.trin.RISK_ASSET_TUPLE]] / dF["m"]
        bw_dec.index = list(c.trin.RISK_ASSET_TUPLE)
        ok_bw, mx_bw, _ = compare_rows(bw_dec, bw_engine, atol=1e-9)
        # engine-equivalence for the exposure: rebuild m from the full compute_signals return_ser rows <= T
        rr = pd.DataFrame({a: full_sig.loc[:T, (a, "return_ser")] for a in c.trin.RISK_ASSET_TUPLE}).iloc[-63:]
        m_eng = c.b6040.compute_gross_exposure_float(rr.mul(bw_engine, axis=1).sum(axis=1), 63, 0.08, 0.085)
        # leak control
        if T != full.index[-1]:
            dPL = trin_decision(full.loc[:T], T, leak=True)
            dFL = trin_decision(full, T, leak=True)
            okL, _, _ = compare_rows(dPL, dFL) if len(dPL) else (False, 0, [])
            leak_caught = not okL
        else:
            leak_caught = None
        # prefix-frame monthly flag / base weights as a truncated compute_signals would see them
        sigP = s.compute_signals(P_df)
        rows.append({"T": str(T.date()), "src": src, "row_T_equal": ok, "max_abs_diff": mx, "bad": bad,
                     "engine_base_weight_equal": ok_bw, "engine_m_equal": bool(abs(m_eng - dF["m"]) < 1e-12),
                     "m": float(dF["m"]), "leak_caught": leak_caught,
                     "xnys_month_end": xnys_month_end_bool(T),
                     "monthly_flag_full": bool(full_sig.loc[T, c.trin.MONTHLY_REBALANCE_FIELD_TUPLE]),
                     "monthly_flag_prefix": bool(sigP.loc[T, c.trin.MONTHLY_REBALANCE_FIELD_TUPLE]),
                     "prefix_base_weight_equal_full": bool(np.allclose(
                         [float(sigP.loc[T, (a, "base_weight_ser")]) for a in c.trin.RISK_ASSET_TUPLE],
                         bw_engine.to_numpy(), atol=1e-12, equal_nan=True))})
    return rows


def summarize(rows):
    df = pd.DataFrame(rows)
    out = {"n_cutoffs": int(len(df)), "n_loader_cutoffs": int((df["src"] == "loader").sum()),
           "row_T_equal": int(df["row_T_equal"].sum())}
    lc = df["leak_caught"].dropna()
    out["leak_caught"] = f"{int(lc.sum())}/{len(lc)}"
    for col in ("engine_base_weight_equal", "engine_m_equal"):
        if col in df:
            out[col] = f"{int(df[col].sum())}/{len(df)}"
    if "month_end_flag_prefix" in df:
        out["ctc_month_end_flag_prefix_true"] = int(df["month_end_flag_prefix"].sum())
        out["ctc_month_end_flag_full_equals_xnys"] = int((df["month_end_flag_full"] == df["xnys_month_end"]).sum())
    if "monthly_flag_prefix" in df:
        me = df[df["xnys_month_end"]]
        out["trin_month_end_cutoffs"] = int(len(me))
        out["trin_monthly_flag_full_true_at_month_end"] = int(me["monthly_flag_full"].sum())
        out["trin_monthly_flag_prefix_true_at_month_end"] = int(me["monthly_flag_prefix"].sum())
        out["trin_prefix_base_weight_equal_full_at_month_end"] = int(me["prefix_base_weight_equal_full"].sum())
    return out


if __name__ == "__main__":
    for name, fn in (("ctc", run_ctc), ("vixm", run_vixm), ("trin", run_trin)):
        if MODE not in ("all", name):
            continue
        rows = fn()
        summ = summarize(rows)
        print(name, summ, flush=True)
        c.dump({"summary": summ, "rows": rows}, f"{name}/a3_rowT.json")
