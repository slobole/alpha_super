"""Compass A5: T5YIE publication timing, ALFRED vintages, session-lag replays (XLK and QQQ share the signal).

Questions
1. For every month-end decision T with an ALFRED vintage, is the observation HEAD uses (latest dated < T) in the
   vintage dated T? Is the observation dated T itself already in vintage T (same-day publication)?
2. Revisions: is the value HEAD uses equal to the value published in vintage T?
3. Replays against HEAD decisions:
   A  vintage T, evening rule: level = latest obs <= T in vintage T (what a decider after FRED's ~17:03 ET
      update would see), anchor = obs dated <= session T-60 in vintage T
   B  vintage T, strict rule: level = latest obs < T in vintage T (HEAD's rule on vintage data)
   C  vintage of session T-1 (conservative time-of-day bound)
   D  full history, one extra SESSION of lag on the level (anchor unchanged)      [A5 extra-session lag]
   E  full history, one extra session of lag on level and anchor
   F  pre-fix rule (same-date level, allow_exact_matches=True)
4. Engine impact (main window 2003-05-01 .. 2026-08-19) of D, E, F and of the ALFRED-consistent evening rule.

Outputs: OUT/cmp_macro_timing.json, OUT/cmp_t5yie_alfred_by_decision.csv
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

import tb_common as tb
from alpha.data.alfred_snapshot import fetch_alfred_vintage_dict

cmp_mod = tb.cmp_mod


def fetch_t5yie_vintages(vintage_list: list[pd.Timestamp]) -> dict[pd.Timestamp, pd.Series]:
    path = tb.CACHE / "alfred_t5yie_vintages.pkl"
    cached: dict = pickle.loads(path.read_bytes()) if path.exists() else {}
    missing = [v for v in vintage_list if v not in cached]
    if missing:
        got, records = fetch_alfred_vintage_dict("T5YIE", missing, batch_size_int=12, pause_seconds_float=1.0)
        cached.update(got)
        path.write_bytes(pickle.dumps(cached))
        rec_path = tb.CACHE / "alfred_t5yie_request_records.pkl"
        old = pickle.loads(rec_path.read_bytes()) if rec_path.exists() else []
        rec_path.write_bytes(pickle.dumps(old + records))
    return {v: cached[v] for v in vintage_list}


def first_vintage_probe() -> str:
    for d in ["2014-01-24", "2014-01-27", "2014-01-28", "2014-01-29", "2014-01-30", "2014-01-31"]:
        try:
            fetch_alfred_vintage_dict("T5YIE", [pd.Timestamp(d)], pause_seconds_float=0.5)
            return d
        except Exception:  # noqa: BLE001
            continue
    return "not found in probe"


def inflation_on(level, anchor, asset_up, threshold=2.0) -> bool:
    return bool(level > threshold and (level > anchor or asset_up))


def regime(growth_on: bool, infl_on: bool) -> str:
    if growth_on and infl_on:
        return "XLE"
    if growth_on:
        return "GOLD"  # XLK or QQQ
    if infl_on:
        return "XLU"
    return "XLP_IEF"


class _LaggedAlign:
    """Wrap align_fred_to_session_ser: shift the published (and optionally the dated) output by one session."""

    def __init__(self, lag_published: int, lag_dated: int, force_same_date_level: bool = False):
        self.orig = cmp_mod.align_fred_to_session_ser
        self.lag_published = lag_published
        self.lag_dated = lag_dated
        self.force_same_date_level = force_same_date_level

    def __call__(self, fred_value_ser, session_date_index, tolerance_day_int=7, include_same_date_bool=False):
        if self.force_same_date_level and not include_same_date_bool:
            v, a = self.orig(fred_value_ser, session_date_index, tolerance_day_int, True)
            return v, a
        v, a = self.orig(fred_value_ser, session_date_index, tolerance_day_int, include_same_date_bool)
        lag = self.lag_dated if include_same_date_bool else self.lag_published
        if lag:
            v = v.shift(lag)
            a = a.shift(lag)
        return v, a


def weights_with_align(align_obj, variant_str="xlk", end_date_str=tb.MAIN_END_STR):
    cfg = tb.compass_config(variant_str)
    orig = cmp_mod.align_fred_to_session_ser
    cmp_mod.align_fred_to_session_ser = align_obj
    try:
        feat, w = cmp_mod.compute_month_end_signal_and_weight_df(
            tb.compass_signal_close_df(end_date_str), tb.ensure_frozen_t5yie(), cfg
        )
    finally:
        cmp_mod.align_fred_to_session_ser = orig
    return feat, w


def flips(base_w: pd.DataFrame, other_w: pd.DataFrame) -> list[str]:
    idx = base_w.index.intersection(other_w.index)
    diff = (base_w.loc[idx] - other_w.loc[idx]).abs().max(axis=1) > 1e-12
    return [str(d.date()) for d in idx[diff.to_numpy()]]


def main() -> None:
    out: dict = {}
    t5 = tb.ensure_frozen_t5yie()
    sig = tb.compass_signal_close_df(tb.NORGATE_LAST_BAR_STR)
    sessions = pd.DatetimeIndex(sig.index)
    cfg = tb.compass_config("xlk")
    head_feat, head_w = cmp_mod.compute_month_end_signal_and_weight_df(sig, t5, cfg)
    # complete months only (the last row 2026-09-25 is the partial month, reported separately)
    out["head_decisions"] = int(len(head_w))
    out["head_first_decision"] = str(head_w.index[0].date())
    out["head_last_row_is_partial_month"] = str(head_w.index[-1].date())

    out["first_alfred_vintage_probe"] = first_vintage_probe()
    decision_list = [d for d in head_feat.index if d >= pd.Timestamp("2014-01-31")]
    prev_list = [sessions[sessions.get_loc(d) - 1] for d in decision_list]
    vint = fetch_t5yie_vintages(sorted(set(decision_list) | set(prev_list)))

    rows = []
    for d, p in zip(decision_list, prev_list):
        vt = vint[d]
        vp = vint[p]
        feat = head_feat.loc[d]
        pos = sessions.get_loc(d)
        anchor_session = sessions[pos - cfg.breakeven_lookback_session_int]
        head_obs_date = t5.index[t5.index < d][-1]
        head_level = float(t5.loc[head_obs_date])
        # vintage T
        vt_le = vt[vt.index <= d]
        vt_lt = vt[vt.index < d]
        vt_anchor = float(vt[vt.index <= anchor_session].iloc[-1])
        vp_le = vp[vp.index <= p]
        vp_anchor = float(vp[vp.index <= anchor_session].iloc[-1])
        asset_up = bool(feat["asset_up_bool"])
        growth = bool(feat["growth_on_bool"])
        rows.append(
            {
                "decision": d,
                "vintage_T_last_obs": vt.index[-1],
                "same_day_obs_in_vintage_T": bool(vt.index[-1] == d),
                "head_obs_date": head_obs_date,
                "head_obs_in_vintage_T": bool(head_obs_date in vt.index),
                "head_level": head_level,
                "vintage_value_of_head_obs": float(vt.loc[head_obs_date]) if head_obs_date in vt.index else np.nan,
                "head_anchor": float(feat["t5yie_prior_float"]),
                "vintage_T_anchor": vt_anchor,
                "prev_vintage_last_obs": vp.index[-1],
                "asset_up": asset_up,
                "growth_on": growth,
                "head_regime": regime(growth, bool(feat["inflation_on_bool"])),
                "A_evening_regime": regime(growth, inflation_on(float(vt_le.iloc[-1]), vt_anchor, asset_up)),
                "A_level": float(vt_le.iloc[-1]),
                "B_strict_regime": regime(growth, inflation_on(float(vt_lt.iloc[-1]), vt_anchor, asset_up)),
                "C_prev_vintage_regime": regime(growth, inflation_on(float(vp_le.iloc[-1]), vp_anchor, asset_up)),
                "margin_level_minus_2": head_level - 2.0,
                "margin_level_minus_anchor": head_level - float(feat["t5yie_prior_float"]),
            }
        )
    tab = pd.DataFrame(rows)
    tab.to_csv(tb.OUT / "cmp_t5yie_alfred_by_decision.csv", index=False)
    complete = tab[tab["decision"] <= pd.Timestamp("2026-08-31")]
    same_day = complete[complete["same_day_obs_in_vintage_T"]]
    out["alfred"] = {
        "decisions_with_vintage": int(len(complete)),
        "head_obs_present_in_vintage_T": int(complete["head_obs_in_vintage_T"].sum()),
        "same_day_obs_in_vintage_T": int(complete["same_day_obs_in_vintage_T"].sum()),
        "first_same_day_decision": str(same_day["decision"].min().date()) if len(same_day) else None,
        "last_not_same_day_decision": str(
            complete.loc[~complete["same_day_obs_in_vintage_T"], "decision"].max().date()
        ),
        "not_same_day_after_first_same_day": [
            str(x.date())
            for x in complete.loc[
                (~complete["same_day_obs_in_vintage_T"]) & (complete["decision"] > same_day["decision"].min()),
                "decision",
            ]
        ],
        "revised_head_values": int(
            (complete["vintage_value_of_head_obs"] - complete["head_level"]).abs().gt(1e-9).sum()
        ),
        "anchor_differs_from_vintage": int((complete["vintage_T_anchor"] - complete["head_anchor"]).abs().gt(1e-9).sum()),
        "B_strict_vs_head_flips": [str(x.date()) for x in complete.loc[complete["B_strict_regime"] != complete["head_regime"], "decision"]],
        "A_evening_vs_head_flips": [str(x.date()) for x in complete.loc[complete["A_evening_regime"] != complete["head_regime"], "decision"]],
        "C_prev_vintage_vs_head_flips": [str(x.date()) for x in complete.loc[complete["C_prev_vintage_regime"] != complete["head_regime"], "decision"]],
        "min_abs_margin_level_minus_2": float(complete["margin_level_minus_2"].abs().min()),
        "n_exact_ties_level_vs_anchor": int(complete["margin_level_minus_anchor"].abs().lt(1e-9).sum()),
        "exact_tie_dates": [str(x.date()) for x in complete.loc[complete["margin_level_minus_anchor"].abs().lt(1e-9), "decision"]],
    }
    partial = tab[tab["decision"] > pd.Timestamp("2026-08-31")]
    out["partial_month_row_2026_09_25"] = partial.astype(str).to_dict(orient="records")

    # --- full-history replays on the session index (main window weights)
    base_feat, base_w = cmp_mod.compute_month_end_signal_and_weight_df(
        tb.compass_signal_close_df(tb.MAIN_END_STR), t5, cfg
    )
    variants = {
        "D_level_lag_plus1_session": _LaggedAlign(1, 0),
        "E_level_and_anchor_lag_plus1_session": _LaggedAlign(1, 1),
        "F_prefix_same_date_level": _LaggedAlign(0, 0, force_same_date_level=True),
    }
    replay: dict = {}
    w_by_name = {"HEAD": base_w}
    for name, align_obj in variants.items():
        _f, w = weights_with_align(align_obj)
        w_by_name[name] = w
        replay[name] = {"flips_vs_head": flips(base_w, w)}
    # ALFRED-consistent evening rule: HEAD before the first same-day vintage decision, same-date level after it
    first_same_day = pd.Timestamp(out["alfred"]["first_same_day_decision"])
    hybrid_w = base_w.copy()
    f_w = w_by_name["F_prefix_same_date_level"]
    late = hybrid_w.index >= first_same_day
    hybrid_w.loc[late] = f_w.loc[hybrid_w.index[late]]
    w_by_name["G_alfred_consistent_evening"] = hybrid_w
    replay["G_alfred_consistent_evening"] = {"flips_vs_head": flips(base_w, hybrid_w)}

    for variant_str in ("xlk", "qqq"):
        replay_metrics = {}
        for name, w in w_by_name.items():
            if variant_str == "qqq":
                w = w.rename(columns={"XLK": "QQQ"})
            s = tb.run_compass_engine(variant_str, month_end_weight_df=w, end_date_str=tb.MAIN_END_STR)
            replay_metrics[name] = tb.metrics(tb.nav_ser(s), tb.MAIN_START_STR, tb.MAIN_END_STR)
        out[f"engine_{variant_str}"] = replay_metrics
        print(variant_str, {k: round(v["cagr_pct"], 3) for k, v in replay_metrics.items()}, flush=True)
    out["replay_flips"] = replay
    tb.write_json("cmp_macro_timing.json", out)
    print(out["alfred"], flush=True)


if __name__ == "__main__":
    main()
