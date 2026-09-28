"""Phase 4: pre-declared improvement candidates C1..C7 and the promotion rules - see SPEC_FROZEN.md."""

from __future__ import annotations

import itertools
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

import common as cm

TRANCHE_OFFSETS = (0, -5, -10, -15)
ENSEMBLE = list(itertools.product((1.8, 2.0, 2.2), (40, 60, 80), (40, 60, 80)))
N_TRIALS_EXTERNAL = {"article_grid": 15, "allocate_smartly_variants": 4, "repo_module": 1,
                     "pakal_study_2026": 44}  # 12 parameter + 24 maps + 5 mechanism + 3 defensive sleeves


def run(w, lag=1, slippage=cm.SLIPPAGE, capital=100_000.0, cash_rate=None, scale=None):
    ew = cm.map_to_execution(w, cm.sessions(), lag=lag)
    sc = None
    if scale is not None:
        sc = cm.map_to_execution(scale.to_frame("s"), cm.sessions(), lag=lag)["s"]
    return cm.run_replica(ew, slippage=slippage, capital=capital, cash_rate_ser=cash_rate, scale_ser=sc)


def ensemble_weights(sig, t5, offset=0):
    frames = []
    for th, be, sl in ENSEMBLE:
        w, _ = cm.compass_weights(sig, t5, cm.Params(threshold=th, be_lookback=be, slope_lookback=sl),
                                  offset=offset)
        frames.append(w.reindex(columns=["XLE", "XLK", "XLU", "XLP", "IEF"]).fillna(0.0))
    common_idx = frames[0].index
    for f in frames[1:]:
        common_idx = common_idx.intersection(f.index)
    return sum(f.loc[common_idx] for f in frames) / len(frames)


def tranche_nav(weight_fn, **kw):
    navs = [run(weight_fn(off), capital=100_000.0 / len(TRANCHE_OFFSETS), **kw) for off in TRANCHE_OFFSETS]
    df = pd.concat(navs, axis=1).dropna()
    return df.sum(axis=1)


def vol_scale(sig, w, target=0.15, lookback=63):
    r = sig.pct_change(fill_method=None)
    out = {}
    for d, row in w.iterrows():
        cols = row[row > 0].index
        hist = r.loc[:d, cols].iloc[-lookback:]
        pr = (hist * row[cols]).sum(axis=1)
        vol = pr.std(ddof=1) * np.sqrt(252)
        out[d] = min(1.0, target / vol) if vol > 0 else 1.0
    return pd.Series(out)


def build(sig, t5):
    base_w, base_reg = cm.compass_weights(sig, t5)
    dtb3 = cm.fred_series("DTB3")
    c = {}
    c["BASE"] = lambda **kw: run(base_w, **kw)
    ens_w = ensemble_weights(sig, t5)
    c["C1 ensemble 27"] = lambda **kw: run(ens_w, **kw)
    qqq_map = {**cm.REGIME_WEIGHTS, (True, False): {"QQQ": 1.0}}
    c["C2 QQQ for XLK"] = lambda **kw: run(cm.weights_from_regimes(base_reg, qqq_map), **kw)
    stag_w = cm.weights_from_regimes(base_reg).reindex(columns=["XLE", "XLK", "XLU", "XLP", "IEF", "DBC"]).fillna(0)
    stag_mask = (base_reg["growth"] == False) & (base_reg["infl"] == True) & (base_reg.index >= "2006-03-01")  # noqa: E712
    stag_w.loc[stag_mask, "XLU"] = 0.5
    stag_w.loc[stag_mask, "DBC"] = 0.5
    c["C3 stagflation XLU+DBC"] = lambda **kw: run(stag_w, **kw)

    def base_off(off):
        return cm.compass_weights(sig, t5, cm.Params(), offset=off)[0]

    c["C4 tranching x4"] = lambda **kw: tranche_nav(base_off, **kw)
    hyst_w, _ = cm.compass_weights(sig, t5, cm.Params(hysteresis=(2.1, 1.9)))
    c["C5 hysteresis 2.1/1.9"] = lambda **kw: run(hyst_w, **kw)
    sc = vol_scale(sig, base_w)
    c["C6 vol target 15%"] = lambda **kw: run(base_w, cash_rate=dtb3, scale=sc, **kw)
    ens_cache = {}

    def ens_off(off):
        if off not in ens_cache:
            ens_cache[off] = ensemble_weights(sig, t5, offset=off)
        return ens_cache[off]

    c["C7 ensemble + tranching"] = lambda **kw: tranche_nav(ens_off, **kw)
    return c, sc


def main():
    sig = cm.signal_close_df()
    t5 = cm.fred_series("T5YIE")
    t5 = t5[t5.index <= pd.Timestamp(cm.MAIN_END_STR)]
    cands, sc = build(sig, t5)
    navs = {k: f() for k, f in cands.items()}
    navs_stress = {k: f(slippage=0.0015) for k, f in cands.items()}
    grid = pd.read_csv(cm.OUT / "p3_grid.csv")
    other = pd.read_csv(cm.OUT / "p3_other.csv")
    n_trials_study = len(grid) + len(other) + 24 + len(cands) - 1
    n_trials = n_trials_study + sum(N_TRIALS_EXTERNAL.values())
    sr_std = float(grid["ALL_sharpe"].std())
    rows = []
    base = navs["BASE"]
    for k, nav in navs.items():
        both = pd.concat([nav.rename("c"), base.rename("b")], axis=1).dropna()
        rc, rb = both["c"].pct_change().dropna(), both["b"].pct_change().dropna()
        pt_c, pt_b = cm.period_table(both["c"]), cm.period_table(both["b"])
        bs = cm.paired_sharpe_bootstrap(rc, rb) if k != "BASE" else {"dsharpe": 0, "p_le_0": np.nan,
                                                                      "p05": np.nan, "p95": np.nan}
        st = pd.concat([navs_stress[k].rename("c"), navs_stress["BASE"].rename("b")], axis=1).dropna()
        d_stress = cm.metrics(st["c"])["sharpe"] - cm.metrics(st["b"])["sharpe"]
        r_all = nav.pct_change().dropna()
        dsr = cm.deflated_sharpe(cm.metrics(nav)["sharpe"], len(r_all), n_trials, float(skew(r_all)),
                                 float(kurtosis(r_all, fisher=False)), sr_std)
        row = {"candidate": k, "cagr": pt_c["ALL"]["cagr"], "sharpe": pt_c["ALL"]["sharpe"],
               "maxdd": pt_c["ALL"]["maxdd"], "vol": pt_c["ALL"]["vol"],
               "dSharpe": pt_c["ALL"]["sharpe"] - pt_b["ALL"]["sharpe"], "boot_p_le_0": bs["p_le_0"],
               "boot_p05": bs["p05"], "boot_p95": bs["p95"],
               "dSharpe_P1": pt_c["P1"]["sharpe"] - pt_b["P1"]["sharpe"],
               "dSharpe_P2": pt_c["P2"]["sharpe"] - pt_b["P2"]["sharpe"],
               "dSharpe_P3": pt_c["P3"]["sharpe"] - pt_b["P3"]["sharpe"],
               "dMaxDD_pp": pt_c["ALL"]["maxdd"] - pt_b["ALL"]["maxdd"],
               "dSharpe_stress_10bps": d_stress, "DSR": dsr, "window": f"{both.index[0].date()}..{both.index[-1].date()}"}
        better = (row["boot_p_le_0"] < 0.05 and min(row["dSharpe_P1"], row["dSharpe_P2"], row["dSharpe_P3"]) > 0
                  and d_stress > 0 and dsr >= 0.95)
        noninf = (row["dSharpe"] >= -0.05 and row["boot_p05"] >= -0.15 and row["dMaxDD_pp"] >= -3
                  and min(row["dSharpe_P1"], row["dSharpe_P2"], row["dSharpe_P3"]) >= -0.10)
        row["passes_better_ex_holdout"] = bool(better) if k != "BASE" else None
        row["non_inferior"] = bool(noninf) if k in ("C1 ensemble 27", "C4 tranching x4", "C7 ensemble + tranching") else None
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(cm.OUT / "p4_candidates.csv", index=False)
    pd.DataFrame(navs).to_pickle(cm.OUT / "p4_navs.pkl")
    print(out.round(3).to_string(index=False))
    meta = {"n_trials_study": n_trials_study, "n_trials_total": n_trials, "external": N_TRIALS_EXTERNAL,
            "grid_sharpe_std": sr_std, "c6_mean_scale": float(sc.mean()), "c6_share_scaled": float((sc < 1).mean())}
    (cm.OUT / "p4_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
