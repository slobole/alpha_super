"""EXPLORATORY (owner question 2026-10-04): do the liquidity-floor DV2 variants also lose their edge in calm markets?

Replica (engine-parity DV2 replica, engine costs), 2004-01-02 .. 2026-09-24, idle cash swept at the T-bill rate.
Three arms isolate the floor from the ranking:
    DV2         the wired rule (NATR14 rank), as in the MR capsule
    DV2-LF      strategies/dv2/strategy_mr_dv2_liquidity_floor.py: raw price > 5 and ADV63 > PIT member median
    DV2-LF-ADV  strategies/dv2/strategy_mr_dv2_liquidity_floor_adv_rank.py: the same floor, rank ADV63 descending
ADV63 = 63-session mean of NATIVE Norgate Turnover (all 63 finite > 0); the floor median is over PIT members with a
finite ADV63 (the modules' rule). The floor is applied here explicitly: loc_lib.dv2_masks does not implement it
(the first version of this script missed that; review 2026-10-04).
Gate: the MR capsule gate (VIX > expanding mean since 1990, open >= 15 sessions).
Per arm:
- trades split by the gate state at the entry decision; excess over SPY measured OPEN-to-OPEN over the same holding
  (aligned with the trade), both with beta 1 and with the arm's fitted beta; standard errors by a bootstrap over
  entry months (2,000 draws);
- standalone ungated vs gated by block, Sharpe in excess of the T-bill rate;
- the book (TAA 0.5 + NDX 0.25 + slot 0.25) 2008-03-04 .. 2026-08-19, with a paired block bootstrap
  P(Sharpe A > Sharpe B) for the key pairs.
Not pre-registered: the gate was selected on DV2 over 2000-2026 and the ADV rank on the DV2 deep-research grid, so
DV2-G is favoured in sample. Writes results/research/mr_capsule_build_20261004/adv_rank_calm.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mr_capsule_20261003"))
sys.path.insert(0, str(HERE.parent / "mr_stress_regime_20261002"))
import components as cp  # noqa: E402
import evaluate as ev  # noqa: E402
import run_stress as st  # noqa: E402

from data.norgate_loader import CAPITALSPECIAL_ADJUSTMENT_STR, load_price_timeseries  # noqa: E402

ll, rp, npc, tbc = cp.ll, cp.rp, cp.npc, cp.tbc
OUT = cp.REPO / "results/research/mr_capsule_build_20261004"
START, END = "2004-01-02", "2026-09-24"
BLOCKS = {"2004-09": ("2004-01-02", "2009-12-31"), "2010-19": ("2010-01-01", "2019-12-31"), "2020-26": ("2020-01-01", END)}
BOOK = ("2008-03-04", "2026-08-19")
DRAW_INT = 2000


def native_adv63(p) -> np.ndarray:
    turnover_arr = np.asarray(rp.load_arrays(p.label_str)["Turnover"], dtype=float)
    valid_arr = np.where(np.isfinite(turnover_arr) & (turnover_arr > 0.0), turnover_arr, np.nan)
    # *** CRITICAL*** trailing window ending at t; min_periods=63 needs all 63 sessions valid (module rule)
    return pd.DataFrame(valid_arr).rolling(63, min_periods=63).mean().to_numpy()


def liquidity_floor_mask(p, adv_arr: np.ndarray) -> np.ndarray:
    """1[raw price > 5] * 1[ADV63 > median ADV63 of that day's PIT members with a finite ADV63] (module rule)."""
    member_arr = np.asarray(p.member, dtype=bool)
    pool_arr = member_arr & np.isfinite(adv_arr)
    median_arr = np.full(len(p.dates), np.nan)
    for row_int in range(len(p.dates)):
        value_arr = adv_arr[row_int][pool_arr[row_int]]
        if value_arr.size:
            median_arr[row_int] = np.median(value_arr)
    with np.errstate(invalid="ignore"):
        return np.isfinite(adv_arr) & (np.asarray(p.RAW, dtype=float) > 5.0) & (adv_arr > median_arr[:, None])


def spy_open_ser(dates: pd.DatetimeIndex) -> pd.Series:
    spy_df = load_price_timeseries("SPY", adjustment_str=CAPITALSPECIAL_ADJUSTMENT_STR, start_date_str="2003-01-01")
    spy_df.index = pd.to_datetime(spy_df.index)
    return spy_df["Open"].astype(float).reindex(dates)


def trade_split(p, res, gate_arr, spy_open) -> dict:
    position = {d: k for k, d in enumerate(p.dates)}
    trade_df = res.trades[["entry_date", "exit_date", "ret"]].copy()
    entry_int = trade_df["entry_date"].map(position).to_numpy()
    exit_int = trade_df["exit_date"].map(position).to_numpy()
    open_arr = spy_open.to_numpy()
    # *** CRITICAL*** market leg aligned with the trade: Open(entry fill) -> Open(exit fill)
    trade_df["mkt"] = open_arr[exit_int] / open_arr[entry_int] - 1.0
    trade_df["stress"] = gate_arr[entry_int - 1]  # gate at the entry decision close
    trade_df["month"] = pd.to_datetime(trade_df["entry_date"]).dt.to_period("M").astype(str)
    trade_df = trade_df.dropna(subset=["mkt", "ret"])
    beta_float = float(np.polyfit(trade_df["mkt"], trade_df["ret"], 1)[0])
    trade_df["excess"] = trade_df["ret"] - trade_df["mkt"]
    trade_df["alpha"] = trade_df["ret"] - beta_float * trade_df["mkt"]
    out = {"beta_fitted": beta_float}
    month_list = sorted(trade_df["month"].unique())
    month_index = {m: k for k, m in enumerate(month_list)}
    trade_df["m"] = trade_df["month"].map(month_index)
    rng = np.random.default_rng(7)
    draw_arr = rng.integers(0, len(month_list), size=(DRAW_INT, len(month_list)))
    for field_str in ("excess", "alpha"):
        per_month = {}
        for flag_bool, label_str in ((False, "calm"), (True, "stress")):
            part = trade_df[trade_df["stress"] == flag_bool]
            sums = np.bincount(part["m"], weights=part[field_str], minlength=len(month_list))
            counts = np.bincount(part["m"], minlength=len(month_list)).astype(float)
            per_month[label_str] = (sums, counts)
            boot = sums[draw_arr].sum(axis=1) / np.maximum(counts[draw_arr].sum(axis=1), 1.0)
            out[f"{label_str}_{field_str}_bps"] = float(part[field_str].mean() * 1e4)
            out[f"{label_str}_{field_str}_se_bps"] = float(boot.std() * 1e4)
            out[f"{label_str}_trades"] = int(len(part))
        calm_s, calm_c = per_month["calm"]
        stress_s, stress_c = per_month["stress"]
        diff = (calm_s[draw_arr].sum(1) / np.maximum(calm_c[draw_arr].sum(1), 1)) - (stress_s[draw_arr].sum(1) / np.maximum(stress_c[draw_arr].sum(1), 1))
        out[f"calm_minus_stress_{field_str}_bps"] = out[f"calm_{field_str}_bps"] - out[f"stress_{field_str}_bps"]
        out[f"calm_minus_stress_{field_str}_ci90_bps"] = [float(np.percentile(diff, 5) * 1e4), float(np.percentile(diff, 95) * 1e4)]
    return out


def excess_stats(ret: pd.Series, rate: pd.Series) -> dict:
    base = st.stats(ret)
    excess = (ret - rate.reindex(ret.index).fillna(0.0)).dropna()
    base["sharpe_excess_tbill"] = float(excess.mean() / excess.std() * np.sqrt(252))
    return base


def main() -> None:
    p = rp.Panel("sp500")
    adv_arr = native_adv63(p)
    p._cache["adv63"] = adv_arr
    floor_arr = liquidity_floor_mask(p, adv_arr)
    gate_arr = cp.gate_on(p.dates)
    rate = st.cash_rate(p.dates)
    spy_open = spy_open_ser(p.dates)
    taa, ndx = tbc.load_taa_ser(), npc.load_l_ret_ser("engine")
    arms = {"DV2": (rp.Rule(), False), "DV2-LF": (rp.Rule(), True), "DV2-LF-ADV": (rp.Rule(rank="adv"), True)}
    report, book_ret = {}, {}
    for name, (rule, floor_bool) in arms.items():
        entry, score, exit_ = ll.dv2_masks(p, rule)
        if floor_bool:
            entry = entry & floor_arr
        block = {}
        for gated_bool in (False, True):
            tag = "gated" if gated_bool else "ungated"
            res = ll.run(p, ll.Spec(f"{name}-{tag}", entry & gate_arr[:, None] if gated_bool else entry, score, exit_), START, END)
            ret = st.swept(res, rate).loc[START:END]
            book = tbc.book_window_return_ser({"taa": taa, "L": ndx, "X": ret}, npc.CANDIDATE_WEIGHT_DICT, *BOOK)
            book_ret[f"{name}-{tag}"] = book
            block[tag] = {"standalone": excess_stats(ret, rate),
                          "blocks": {k: excess_stats(ret.loc[a:b], rate) for k, (a, b) in BLOCKS.items()},
                          "book": excess_stats(book, rate), "trades": int(len(res.trades))}
            if not gated_bool:
                block["trade_split_ungated"] = trade_split(p, res, gate_arr, spy_open)
            s, b = block[tag]["standalone"], block[tag]["book"]
            print(f"{name:<11}{tag:<8} trades {len(res.trades):>6}  Sh {s['sharpe']:.3f} (xs {s['sharpe_excess_tbill']:.3f})  "
                  f"CAGR {s['cagr'] * 100:5.1f}%  DD {s['max_dd'] * 100:5.1f}%  | book Sh {b['sharpe']:.3f}", flush=True)
        report[name] = block
    pairs = [("DV2-gated", "DV2-LF-ADV-gated"), ("DV2-gated", "DV2-LF-ADV-ungated"), ("DV2-gated", "DV2-LF-gated"),
             ("DV2-LF-ADV-gated", "DV2-LF-ADV-ungated"), ("DV2-gated", "DV2-ungated")]
    report["book_bootstrap_p_a_beats_b"] = {f"{a} > {b}": ev.bootstrap_p(book_ret[a], book_ret[b]) for a, b in pairs}
    (OUT / "adv_rank_calm.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    for name in arms:
        print(name, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in report[name]["trade_split_ungated"].items()}))
        print("   blocks ungated", {k: round(v["sharpe_excess_tbill"], 2) for k, v in report[name]["ungated"]["blocks"].items()},
              "gated", {k: round(v["sharpe_excess_tbill"], 2) for k, v in report[name]["gated"]["blocks"].items()})
    print(json.dumps(report["book_bootstrap_p_a_beats_b"], indent=1))


if __name__ == "__main__":
    main()
