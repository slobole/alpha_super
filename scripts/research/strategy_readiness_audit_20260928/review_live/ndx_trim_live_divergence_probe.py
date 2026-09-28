"""Review probe (live-parity lens): months where LIVE NDX selection differs from the BACKTEST
because of the 5-member-row tail trim of ex-members.

The loader (data/norgate_loader.py:106-107) and the snapshot exporter
(scripts/export_norgate_snapshot.py:134-139) drop the last 5 member rows of every symbol whose
constituent series does not end on today's $SPX date. In the backtest (and in the primary replay,
which cached one universe built today) a name that LEFT the index within 5 sessions after a
month-end T is therefore a NON-member at T. Live at T, the same name was still a current member
(its series ended on T, so no trim fired) and was eligible. The primary replay cannot see this:
both sides used the same trimmed universe.

For every month-end decision date T of the backtest schedule and both NDX variants, compare
get_target_weight_ser with the trimmed universe (backtest) vs the untrimmed universe (what live saw).
Study code only; direct local Norgate, read-only. Usage: uv run python <this file>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
OUT = REPO / "results/research/strategy_readiness_audit_20260928/review_live"
OUT.mkdir(parents=True, exist_ok=True)

import norgatedata  # noqa: E402

import data.norgate_loader as loader_module  # noqa: E402
import strategies.momentum.strategy_mo_atr_normalized_ndx as atr_module  # noqa: E402
import strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled as vxn_module  # noqa: E402

END_DATE_STR = "2026-09-25"


def _untrimmed_universe(indexname: str = "Nasdaq 100"):
    symbols = norgatedata.watchlist_symbols(f"{indexname} Current & Past")
    frame_list = []
    for symbol in symbols:
        idx = norgatedata.index_constituent_timeseries(symbol, indexname, timeseriesformat="pandas-dataframe")
        if idx["Index Constituent"].sum() > 0:
            idx = idx.rename(columns={"Index Constituent": symbol})
            frame_list.append(idx.loc[idx[symbol] == 1])
    return symbols, pd.concat(frame_list, axis=1).fillna(0).astype(int).sort_index()


def main():
    trimmed_tuple = loader_module.build_index_constituent_matrix("Nasdaq 100")
    untrimmed_tuple = _untrimmed_universe("Nasdaq 100")
    result = {}
    event_rows = []
    for key, module_obj, loader_name, strat_cls in (
        ("vxn", vxn_module, "get_vxn_scaled_atr_normalized_ndx_data", vxn_module.VxnScaledAtrNormalizedNdxStrategy),
        ("atr", atr_module, "get_atr_normalized_ndx_data", atr_module.AtrNormalizedNdxStrategy),
    ):
        cfg = module_obj.DEFAULT_CONFIG.__class__(**{**module_obj.DEFAULT_CONFIG.__dict__, "end_date_str": END_DATE_STR})
        out = {}
        for label, uni in (("trimmed", trimmed_tuple), ("untrimmed", untrimmed_tuple)):
            atr_module.build_index_constituent_matrix = (lambda indexname="Nasdaq 100", _u=uni: (list(_u[0]), _u[1].copy()))
            data_tuple = getattr(module_obj, loader_name)(cfg)
            out[label] = data_tuple
        pricing_df, uni_trim_df, schedule_df = out["trimmed"][0], out["trimmed"][1], out["trimmed"][2]
        uni_raw_df = out["untrimmed"][1]
        kwargs = dict(name="review", benchmarks=["SPY"], rebalance_schedule_df=schedule_df, regime_symbol_str="SPY")
        if key == "vxn":
            kwargs["vxn_scale_signal_df"] = out["trimmed"][3]
        strat = strat_cls(**kwargs)
        # Signals are computed on the union of priced symbols; both universes load the same symbols
        # except names that are only ever members in the trimmed tail (none load differently here).
        common_cols = pricing_df.columns.intersection(out["untrimmed"][0].columns)
        signal_df = strat.compute_signals(out["untrimmed"][0].loc[:, common_cols].copy())
        n_diff = 0
        n_dates = 0
        for _, row in schedule_df.iterrows():
            T = pd.Timestamp(row["decision_date_ts"])
            if T not in signal_df.index:
                continue
            n_dates += 1
            strat.previous_bar = T
            strat.universe_df = uni_trim_df
            w_bt = strat.get_target_weight_ser(close_row_ser=signal_df.loc[T])
            strat.universe_df = uni_raw_df
            w_live = strat.get_target_weight_ser(close_row_ser=signal_df.loc[T])
            if set(w_bt.index) != set(w_live.index) or (len(w_bt) and (w_bt.sort_index() - w_live.reindex(w_bt.index).fillna(0)).abs().max() > 1e-12):
                n_diff += 1
                event_rows.append({
                    "variant": key, "decision_date": T.date().isoformat(),
                    "backtest_set": ",".join(sorted(w_bt.index)), "live_set": ",".join(sorted(w_live.index)),
                    "live_only": ",".join(sorted(set(w_live.index) - set(w_bt.index))),
                    "backtest_only": ",".join(sorted(set(w_bt.index) - set(w_live.index))),
                })
        # Membership rows that differ at month-end decision dates, whether or not selected.
        member_diff_count = 0
        for _, row in schedule_df.iterrows():
            T = pd.Timestamp(row["decision_date_ts"])
            a = atr_module.get_asof_universe_membership_ser(uni_trim_df, T)
            b = atr_module.get_asof_universe_membership_ser(uni_raw_df.reindex(columns=uni_trim_df.columns).fillna(0), T)
            member_diff_count += int(((a == 0) & (b == 1)).sum())
        result[key] = {"decision_dates": n_dates, "decisions_with_different_selection": n_diff,
                       "member_rows_live_only_at_decision_dates": member_diff_count}
        print(key, json.dumps(result[key]), flush=True)
    pd.DataFrame(event_rows).to_csv(OUT / "ndx_trim_live_divergence_events.csv", index=False)
    (OUT / "ndx_trim_live_divergence_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
