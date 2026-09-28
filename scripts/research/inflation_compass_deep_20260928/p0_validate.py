"""Phase 0: validate the vectorized signal and the fast replica against the fixed module and the real engine."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common as cm

import strategies.taa_df.strategy_taa_inflation_compass as compass


def main():
    sig = cm.signal_close_df()
    t5 = cm.fred_series("T5YIE")
    t5 = t5[t5.index <= pd.Timestamp(cm.MAIN_END_STR)]

    # 1) signal equality vs the module (module's own data path, same Norgate cache)
    cfg = compass.DEFAULT_CONFIG
    sig_mod = sig.loc["2002-01-01":cm.MAIN_END_STR, list(cfg.signal_asset_tuple)]
    _, w_mod = compass.compute_month_end_signal_and_weight_df(sig_mod, t5, cfg)
    w_rep, reg = cm.compass_weights(sig.loc[:cm.MAIN_END_STR], t5)
    common_idx = w_mod.index.intersection(w_rep.index)
    wm = w_mod.loc[common_idx].reindex(columns=["XLE", "XLK", "XLU", "XLP", "IEF"]).fillna(0)
    wr = w_rep.loc[common_idx].reindex(columns=["XLE", "XLK", "XLU", "XLP", "IEF"]).fillna(0)
    mism = [str(d.date()) for d in common_idx if not np.allclose(wm.loc[d], wr.loc[d])]
    print(f"signal: module {len(w_mod)} decisions, replica {len(w_rep)}, common {len(common_idx)}, "
          f"mismatches {len(mism)} {mism[:10]}")

    # 2) replica NAV vs real engine, same window as the audit
    strat = compass.run_variant(show_display_bool=False, save_results_bool=False,
                                backtest_start_date_str=cm.MAIN_START_STR, end_date_str=cm.MAIN_END_STR)
    eng = strat.results["total_value"].astype(float)
    ew = cm.map_to_execution(w_rep, cm.sessions(), lag=1)
    rep = cm.run_replica(ew, start=cm.MAIN_START_STR, end=cm.MAIN_END_STR)
    both = pd.concat([eng.rename("eng"), rep.rename("rep")], axis=1).dropna()
    re, rr = both["eng"].pct_change().dropna(), both["rep"].pct_change().dropna()
    me, mr = cm.metrics(both["eng"]), cm.metrics(both["rep"])
    out = {
        "signal_mismatch_dates": mism,
        "engine": me, "replica": mr,
        "cagr_diff_pp": mr["cagr"] - me["cagr"], "sharpe_diff": mr["sharpe"] - me["sharpe"],
        "daily_corr": float(np.corrcoef(re, rr)[0, 1]),
        "max_abs_daily_diff": float((re - rr).abs().max()),
        "n_days": int(len(both)),
        "t5yie_download_status": strat._data_adjustment_policy_dict["fred_series_provenance_dict"][
            "download_status_str"],
    }
    print(json.dumps(out, indent=1, default=str))
    (cm.OUT / "p0_validation.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
