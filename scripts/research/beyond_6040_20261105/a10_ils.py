"""A10 — the Beyond 60/40 portfolio seen by an ILS-based investor. Research-only.

Policies (all on the B0 author rule, USD):
  USD ref        B0 in USD, cash = Fed funds
  ILS unhedged   NAV_ILS = NAV_USD x USDILS ; r = (1+r_usd)(1+fx) - 1
  ILS hedged     full NAV sold forward daily ; r = (1+r_usd)(1+fx) - 1 - fx + (il - dff)/252
  ILS 50%        static half hedge (h = 0.5 in the same formula)
  Active binary  monthly h in {0,1}: hedge when E[-USD move] + carry > 0 (expanding OLS on a trend signal)
  Active band    as above, h = 0.5 when |expected value| < 0.5 x residual sd
General formula: r_ILS = (1+r_usd)(1+fx) - 1 + h * ( -fx + (il - dff)/252 ),  fx = USDILS_t/USDILS_{t-1} - 1.
Cash benchmark: Fed funds for the USD investor, il_rate for the ILS investor (nominal, no CPI).
"""
from common import *

SUB = "a10"
START, END = "1993-02-01", DEV_END
BLOCKS = {"1993-99": ("1993-02-01", "1999-12-31"), "2000s": ("2000-01-01", "2009-12-31"),
          "2010s": ("2010-01-01", "2019-12-31"), "2020-22": ("2020-01-01", "2022-12-31")}
WARMUP_M = 60      # months of regression warm-up
MIN_Z_M = 24       # months before an expanding z-score is trusted

panel = dev(load())
B0 = run(panel, Params())
r_usd = B0["ret"].loc[START:END]
dff_d = B0["cash_daily"].loc[START:END]                     # Fed funds / 252
S = panel["usdils"].loc[START:END]
fx = (S / S.shift(1) - 1.0).fillna(0.0)
il_d = panel["il_rate"].loc[START:END] / TRADING_DAYS
carry_d = il_d - dff_d                                       # forward carry received per day when selling USD forward
assert 2.0 < S.mean() < 5.5, "USDILS must be shekels per dollar"


def ils_return(h: pd.Series | float) -> pd.Series:
    h = pd.Series(h, index=r_usd.index) if np.isscalar(h) else h.reindex(r_usd.index).ffill().fillna(0.0)
    return (1 + r_usd) * (1 + fx) - 1 + h * (-fx + carry_d)


# ------------------------------------------------------------ active trend hedge (author's rule)
def active_hedge(band: bool) -> tuple[pd.Series, pd.DataFrame]:
    """Monthly decision at month-end t from information through t, applied to every session of month t+1."""
    Sm = panel["usdils"].loc["1991-01-01":END].resample("ME").last()
    logS = np.log(Sm)
    sig = pd.DataFrame({k: logS - logS.shift(k) for k in (1, 3, 6, 12)})
    z = (sig - sig.expanding(MIN_Z_M).mean()) / sig.expanding(MIN_Z_M).std()
    signal = z.mean(axis=1)                                   # average z, info through month-end t
    nxt = (Sm.shift(-1) / Sm - 1.0)                           # realised USD/ILS move over month t+1
    il_m = panel["il_rate"].resample("ME").last().reindex(Sm.index)
    ff_m = panel["dff"].resample("ME").last().reindex(Sm.index)
    carry_m = (il_m - ff_m) / 12.0                            # carry received per month when hedging (short USD)
    rows = []
    hist_x, hist_y = [], []
    for i, t in enumerate(Sm.index):
        x_t = signal.iloc[i]
        # training pairs: (signal_s, nxt_s) with s+1 <= t, i.e. completed months only
        if i >= 1 and np.isfinite(signal.iloc[i - 1]) and np.isfinite(nxt.iloc[i - 1]):
            hist_x.append(signal.iloc[i - 1]); hist_y.append(nxt.iloc[i - 1])
        if len(hist_x) < WARMUP_M or not np.isfinite(x_t) or not np.isfinite(carry_m.iloc[i]):
            continue
        X = np.c_[np.ones(len(hist_x)), np.array(hist_x)]
        y = np.array(hist_y)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid_sd = float(np.std(y - X @ beta, ddof=2))
        pred = float(beta[0] + beta[1] * x_t)                 # predicted USD/ILS move next month (USD up = positive)
        ev_hedge = -pred + float(carry_m.iloc[i])             # expected value of selling USD forward for one month
        if band and abs(ev_hedge) < 0.5 * resid_sd:
            h = 0.5
        else:
            h = 1.0 if ev_hedge > 0 else 0.0
        rows.append({"month_end": t, "signal": x_t, "alpha": beta[0], "beta": beta[1], "pred_usd_move": pred,
                     "carry_m": float(carry_m.iloc[i]), "ev_hedge": ev_hedge, "resid_sd": resid_sd, "h": h,
                     "realised_usd_move": nxt.iloc[i]})
    dec = pd.DataFrame(rows).set_index("month_end")
    # apply h decided at month-end t to the sessions of month t+1
    h_daily = pd.Series(np.nan, index=r_usd.index)
    for t, h in dec["h"].items():
        nxt_sessions = r_usd.index[(r_usd.index > t) & (r_usd.index <= t + pd.offsets.MonthEnd(1))]
        h_daily.loc[nxt_sessions] = h
    return h_daily, dec


h_bin, dec_bin = active_hedge(band=False)
h_band, dec_band = active_hedge(band=True)
ACTIVE_START = h_bin.first_valid_index()

policies = {
    "USD ref (B0)": (r_usd, dff_d, pd.Series(np.nan, index=r_usd.index)),
    "ILS unhedged": (ils_return(0.0), il_d, pd.Series(0.0, index=r_usd.index)),
    "ILS full hedge": (ils_return(1.0), il_d, pd.Series(1.0, index=r_usd.index)),
    "ILS 50% hedge": (ils_return(0.5), il_d, pd.Series(0.5, index=r_usd.index)),
    "ILS active binary": (ils_return(h_bin), il_d, h_bin),
    "ILS active 0.5-band": (ils_return(h_band), il_d, h_band),
}
fx_leg = {"USD ref (B0)": pd.Series(0.0, index=r_usd.index), "ILS unhedged": fx, "ILS full hedge": carry_d,
          "ILS 50% hedge": 0.5 * fx + 0.5 * carry_d,
          "ILS active binary": (1 - h_bin) * fx + h_bin * carry_d,
          "ILS active 0.5-band": (1 - h_band) * fx + h_band * carry_d}
KEEP10 = ["start", "end", "cagr", "vol", "sharpe", "sharpe_raw", "max_dd", "worst_12m", "worst_36m", "longest_underwater_years",
          "cvar5_monthly", "pct_positive_years"]


def table(s, e, restrict_active=True):
    rows = {}
    for name, (ret, cash, h) in policies.items():
        s_eff = s
        if name.startswith("ILS active") and restrict_active:
            s_eff = max(pd.Timestamp(s), ACTIVE_START)
        rr = ret.loc[s_eff:e].dropna()
        if len(rr) < 60:
            continue
        m = metrics(rr, cash.loc[s_eff:e])
        row = {k: m.get(k, np.nan) for k in KEEP10}
        row["cash_cagr"] = float((1 + cash.loc[rr.index]).prod() ** (TRADING_DAYS / len(rr)) - 1)
        row["avg_hedge"] = float(h.loc[rr.index].mean()) if h.loc[rr.index].notna().any() else np.nan
        leg = fx_leg[name].loc[rr.index]
        row["fx_leg_ann"] = float(leg.mean() * TRADING_DAYS)                  # arithmetic, per unit notional
        row["fx_leg_total"] = float((1 + leg).prod() - 1)
        rows[name] = row
    return pd.DataFrame(rows).T


full = table(START, END)
etf = table("2005-01-01", END)
common = table(ACTIVE_START, END)
save(full, "policies_1993_2022", SUB); save(etf, "policies_2005_2022", SUB); save(common, "policies_common_window", SUB)

dec_rows = []
for b, (s, e) in BLOCKS.items():
    t = table(s, e)
    t.insert(0, "block", b)
    dec_rows.append(t)
decades = pd.concat(dec_rows)
save(decades, "policies_by_decade", SUB)

# standalone FX facts
fx_year = pd.DataFrame({"usdils_end": S.resample("YE").last(), "usd_vs_ils_pct": S.resample("YE").last().pct_change(),
                        "il_rate_avg": panel["il_rate"].loc[START:END].resample("YE").mean(),
                        "ff_avg": panel["dff"].loc[START:END].resample("YE").mean()})
fx_year["carry_received_if_hedged"] = fx_year["il_rate_avg"] - fx_year["ff_avg"]
fx_year["hedge_bin_avg"] = h_bin.resample("YE").mean()
save(fx_year, "fx_by_year", SUB)
save(dec_bin, "active_decisions_binary", SUB); save(dec_band, "active_decisions_band", SUB)

# drawdown episodes of the unhedged ILS investor
def dd_episodes(ret: pd.Series, top=4):
    nav = (1 + ret).cumprod(); peak = nav.cummax(); dd = nav / peak - 1
    eps = []
    in_dd = False
    for i, (d, v) in enumerate(dd.items()):
        if v < 0 and not in_dd:
            in_dd, start_i = True, i - 1
        elif v >= 0 and in_dd:
            in_dd = False
            seg = dd.iloc[start_i:i + 1]
            eps.append((seg.index[0], seg.idxmin(), d, seg.min()))
    if in_dd:
        seg = dd.iloc[start_i:]
        eps.append((seg.index[0], seg.idxmin(), pd.NaT, seg.min()))
    eps = sorted(eps, key=lambda x: x[3])[:top]
    out = []
    for pk, tr, rec, depth in eps:
        out.append({"peak": pk.date(), "trough": tr.date(), "recovery": rec.date() if pd.notna(rec) else "not by 2022-12",
                    "depth": depth, "usdils_at_peak": S.loc[pk], "usdils_at_trough": S.loc[tr],
                    "usd_move_peak_to_trough": S.loc[tr] / S.loc[pk] - 1,
                    "usd_strategy_same_span": (1 + r_usd.loc[pk:tr]).prod() - 1})
    return pd.DataFrame(out)

dd_un = dd_episodes(policies["ILS unhedged"][0]); dd_un.insert(0, "policy", "ILS unhedged")
dd_he = dd_episodes(policies["ILS full hedge"][0]); dd_he.insert(0, "policy", "ILS full hedge")
dd_us = dd_episodes(r_usd); dd_us.insert(0, "policy", "USD ref")
dd_all = pd.concat([dd_un, dd_he, dd_us], ignore_index=True)
save(dd_all, "drawdown_episodes", SUB)

# the two shekel-appreciation runs
runs = {}
for nm, (s, e) in {"2002-06 -> 2008-07": ("2002-06-28", "2008-07-31"), "2012-01 -> 2022-01": ("2012-01-02", "2022-01-31"),
                   "1993-02 -> 2002-06": ("1993-02-01", "2002-06-28")}.items():
    seg = {k: (1 + v[0].loc[s:e]).prod() - 1 for k, v in policies.items() if not k.startswith("ILS active") or pd.Timestamp(s) >= ACTIVE_START}
    seg["usdils_move"] = S.loc[s:e].iloc[-1] / S.loc[s:e].iloc[0] - 1
    seg["il_minus_ff_avg"] = float((panel["il_rate"] - panel["dff"]).loc[s:e].mean())
    runs[nm] = seg
runs = pd.DataFrame(runs).T
save(runs, "shekel_runs", SUB)

# hit rate of the active rule's predictions
hit = pd.DataFrame({
    "binary": {"months": len(dec_bin), "share_hedged": dec_bin["h"].mean(),
               "sign_hit_rate": float((np.sign(dec_bin["pred_usd_move"]) == np.sign(dec_bin["realised_usd_move"])).mean()),
               "corr_pred_realised": float(dec_bin[["pred_usd_move", "realised_usd_move"]].corr().iloc[0, 1]),
               "avg_beta": dec_bin["beta"].mean(), "share_months_carry_positive": float((dec_bin["carry_m"] > 0).mean())},
    "band": {"months": len(dec_band), "share_hedged": dec_band["h"].mean(), "share_half": float((dec_band["h"] == 0.5).mean())}}).T
save(hit, "active_hit_rate", SUB)

pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
print("ACTIVE_START", ACTIVE_START.date())
for nm, t in {"1993-02 -> 2022-12": full, "2005 -> 2022": etf, f"common {ACTIVE_START.date()} -> 2022": common}.items():
    print(f"\n== {nm}"); print(t.drop(columns=["start", "end"]).astype(float).round(4).to_string())
print("\n== by decade"); print(decades.drop(columns=["start", "end"]).to_string())
print("\n== drawdown episodes"); print(dd_all.to_string())
print("\n== shekel runs"); print(runs.astype(float).round(4).to_string())
print("\n== fx by year"); print(fx_year.round(4).to_string())
print("\n== active hit"); print(hit.to_string())

# ------------------------------------------------------------ diagnostics: why 50% has the lowest DD, why the band degenerates
m = pd.DataFrame({"r_usd": r_usd, "fx": fx}).resample("ME").apply(lambda x: (1 + x).prod() - 1)
bad = m[m["r_usd"] < m["r_usd"].quantile(0.1)]
ratio = dec_bin["ev_hedge"].abs() / dec_bin["resid_sd"]
diag = {"monthly_corr_rusd_fx": float(m.corr().iloc[0, 1]),
        "worst_decile_strategy_months": {"avg_r_usd": float(bad["r_usd"].mean()), "avg_fx": float(bad["fx"].mean()),
                                         "share_usd_up": float((bad["fx"] > 0).mean())},
        "band_abs_ev_over_resid_sd_quantiles": ratio.quantile([.5, .9, .99, 1.0]).round(3).to_dict(),
        "bootstrap_hedged_minus_unhedged_1993": block_bootstrap_diff(policies["ILS full hedge"][0], policies["ILS unhedged"][0], il_d, n_draws=1000),
        "bootstrap_hedged_minus_unhedged_2005": block_bootstrap_diff(policies["ILS full hedge"][0].loc["2005":], policies["ILS unhedged"][0].loc["2005":], il_d.loc["2005":], n_draws=1000),
        "bootstrap_half_minus_unhedged_1993": block_bootstrap_diff(policies["ILS 50% hedge"][0], policies["ILS unhedged"][0], il_d, n_draws=1000)}
save_json(diag, "diagnostics", SUB)
print("\n== diagnostics"); print(json.dumps(diag, indent=1, default=float))
