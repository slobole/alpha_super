"""DV2 limit-entry study (registration dv2_limit_entry_20261002): do passive limit orders rescue DV2's small-cap edge?

The DV2 signal is FROZEN (alpha/scout/specs/dv2.py LIVE_CONFIG); the execution model is in limit_book.py. Every result
uses the size-ladder superset panel (data to 2022-12-30; alpha.scout.universes).

    PYTHONUTF8=1 uv run python scripts/research/scout_dv2_limit_entry_20261002/run_limit.py universe [name ...]
    PYTHONUTF8=1 uv run python scripts/research/scout_dv2_limit_entry_20261002/run_limit.py fillstress [name ...]
    PYTHONUTF8=1 uv run python scripts/research/scout_dv2_limit_entry_20261002/run_limit.py mcpt name [workers] [pooled|ar] [permutations]

Per universe (window 2004-01-01 .. 2022-12-30; membership zeroed before the start, as the size ladder):
  pods   the registered grid, entry {moo, k = 0, 0.25, 0.5, 1.0} x exit {moo, limit}, each under three cost cases on the
         SAME fills: gross (no spread, no fee), AR (per-stock Abdi-Ranaldo half-spread) and pooled (ADV-bucket pooled
         half-spread), both tick-floored and floored at 2.5 bp, charged on marketable fills only; engine commissions on
         nominal shares ($0.005/share, $1 minimum, 1% cap) on every fill except in the gross case. No dividends.
  events the adverse-selection table: every DV2 event (the rule's candidates at T, before slots), its limit order worked
         on T+1 alone (no book), and the h3 forward excess over the same-date regime-eligible members (the S3 baseline)
         of filled vs unfilled vs all events: (a) entry at Open_(T+1) for every event (the S3 measure), (b) entry at the
         fill price for filled events, (c) anchored at Close_T (pure selection, no price effect). Date-level means, Newey-
         West t (lag 2), eras.
  cap    alpha.scout.stations.s6_book.capacity on the pooled-case fills (5th-percentile fill at 1% of its 63-session ADV).
"""

from __future__ import annotations

import dataclasses
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from limit_book import (
    entry_limit_mat,
    event_fill_mats,
    exit_limit_mat,
    limit_book,
    trade_through_margin_mat,
)
from register import ENTRY_TUPLE, EXIT_TUPLE, UNIVERSE_TUPLE

from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH
from alpha.scout.metrics import performance_dict, sharpe_float
from alpha.scout.specs import dv2
from alpha.scout.universes import (
    SEAL_END_STR,
    SUPERSET_ROOT_PATH,
    fill_slippage_mat,
    half_spread_mat,
    load_superset_panel,
    rule_mats,
    tick_half_spread_mat,
    universe_panel,
)
from alpha.stats.newey_west import newey_west_mean_t_stat

OUT_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "dv2_limit_entry"
SUPERSET_NAME_STR = "size_ladder"
LOAD_INDEX_LIST = ["S&P 500", "S&P 100", "Russell 1000", "Russell 2000", "Russell Micro Cap", "S&P SmallCap 600"]
LOWER_HALF_NAME_STR = "Russell 2000 lower half"
POOLED_FILE_STR = "pooled_half_spread_b20_w63_m500_monotone.npy"  # built by the size ladder (run_ladder.pooled_spread_superset)
CONFIG = dv2.LIVE_CONFIG
START_STR = "2004-01-01"
HORIZON_INT = 3
FLOOR_SLIP_FLOAT = 0.00025
FEE_PER_SHARE_FLOAT, MIN_FEE_FLOAT, FEE_CAP_FLOAT = 0.005, 1.0, 0.01
ERA_TUPLE = (("2004-2007", "2004-01-01", "2007-12-31"), ("2008-2015", "2008-01-01", "2015-12-31"), ("2016-2022", "2016-01-01", SEAL_END_STR))
log_started_float = time.time()


def log(text_str: str) -> None:
    print(f"[{time.time() - log_started_float:7.0f}s] {text_str}", flush=True)


def slug(name_str: str) -> str:
    return name_str.replace(" ", "_").replace("&", "and")


def config_label(entry, exit_str: str) -> str:
    return f"entry_{entry if entry == 'moo' else f'k{entry:g}'}__exit_{exit_str}"


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")


# ---------------------------------------------------------------- universes and inputs
def build_panel(superset, name_str: str):
    """The universe's sealed Panel. Russell 2000 lower half = Russell 2000 members that are Russell Micro Cap members on
    the same session (the size ladder's "Russell 2000 in Micro Cap" bucket), on its ever-members' columns.

    *** CRITICAL*** membership is the exact flag of the session itself (both flags dated T); zeroed before the start."""
    if name_str != LOWER_HALF_NAME_STR:
        return universe_panel(superset, name_str, member_from_str=START_STR)
    row_mask = np.asarray(superset.date_index <= pd.Timestamp(SEAL_END_STR))
    both_mat = (np.asarray(superset.member_dict["Russell 2000"][row_mask]) == 1) & (np.asarray(superset.member_dict["Russell Micro Cap"][row_mask]) == 1)
    symbol_list = [s for s, keep_bool in zip(superset.symbol_list, both_mat.any(axis=0)) if keep_bool]
    del both_mat
    panel = universe_panel(superset, "Russell 2000", member_from_str=START_STR, symbol_list=symbol_list)
    micro_df = universe_panel(superset, "Russell Micro Cap", member_from_str=START_STR, symbol_list=symbol_list, field_tuple=()).member_df
    member_df = ((panel.member_df == 1) & (micro_df == 1)).astype(np.int8)
    return dataclasses.replace(panel, name_str=f"{LOWER_HALF_NAME_STR} (superset {superset.snapshot_id_str})", member_df=member_df,
                               snapshot_id_str=f"{superset.snapshot_id_str}:{LOWER_HALF_NAME_STR}")


def spread_dict_for(superset, panel) -> dict:
    """{"ar": per-stock Abdi-Ranaldo, "pooled": ADV-bucket pooled} half-spreads at T on the panel's axes, each floored at
    half a nominal tick (as run_ladder.spread_dict_for). Both estimators read bars <= T."""
    pooled_path = SUPERSET_ROOT_PATH / SUPERSET_NAME_STR / superset.snapshot_id_str / POOLED_FILE_STR
    if not pooled_path.exists():
        raise FileNotFoundError(f"{pooled_path} is missing: run the size ladder's universe step first (it builds the pooled spread).")
    position_vec = pd.Index(superset.symbol_list).get_indexer(panel.symbol_list)
    row_vec = superset.date_index.get_indexer(panel.date_index)
    tick_half = tick_half_spread_mat(panel.field("Unadjusted Close").to_numpy(dtype=float))
    pooled_mat = np.load(pooled_path, mmap_mode="r")
    return {"ar": np.fmax(half_spread_mat(*(panel.field(f).to_numpy(dtype=float) for f in ("High", "Low", "Close"))), tick_half),
            "pooled": np.fmax(np.asarray(pooled_mat[row_vec][:, position_vec], dtype=float), tick_half)}


def signal_inputs(panel) -> dict:
    """rule_mats (candidates, exit) plus the regime / event masks and NATR, recomputed with rule_mats' own expressions."""
    mats = rule_mats(panel, CONFIG)
    field = lambda f: panel.field(f).to_numpy(dtype=float)
    high_mat, low_mat = field("High"), field("Low")
    complete_mat = np.isfinite(mats["open"]) & np.isfinite(high_mat) & np.isfinite(low_mat) & np.isfinite(mats["close"])
    with np.errstate(divide="ignore", invalid="ignore"):
        momentum_mat = dv2._momentum_df(panel.field("Close"), CONFIG.momentum_lookback_int).to_numpy(dtype=float)
        sma_mat = panel.field("Close").rolling(CONFIG.trend_sma_int).mean().to_numpy(dtype=float)
        natr = dv2.natr_mat(high_mat, low_mat, mats["close"], CONFIG.natr_length_int)
        base_mat = (complete_mat & mats["member"] & ~np.isnan(natr) & ~np.isnan(momentum_mat) & (mats["close"] > sma_mat)
                    & (momentum_mat > CONFIG.momentum_min_float))
    dv2_values = dv2.dv2_mat(mats["close"], high_mat, low_mat, CONFIG.dv2_length_int, base_mat)
    with np.errstate(invalid="ignore"):
        event_mat = base_mat & (dv2_values < CONFIG.entry_dv2_max_float)
    if int(event_mat.sum()) != len(mats["candidate_vec"]):
        raise AssertionError("event mask does not match rule_mats' candidates")
    unadjusted_mat = field("Unadjusted Close")
    with np.errstate(divide="ignore", invalid="ignore"):
        scale_mat = np.nan_to_num(mats["close"] / unadjusted_mat, nan=1.0, posinf=1.0)  # adjusted -> nominal shares
    return {"mats": mats, "high": high_mat, "low": low_mat, "natr": natr, "base": base_mat, "event": event_mat,
            "unadjusted": unadjusted_mat, "scale": scale_mat}


# ---------------------------------------------------------------- pods
def window(ser: pd.Series) -> pd.Series:
    return ser.loc[START_STR:SEAL_END_STR]


def pod_metrics(result, baseline_ser: pd.Series) -> dict:
    daily_ser = window(result.daily_ser)
    performance = performance_dict(daily_ser)
    out_dict = {"sharpe_float": performance["sharpe_float"], "cagr_float": performance["cagr_float"],
                "max_drawdown_float": performance["max_drawdown_float"], "volatility_float": performance["volatility_float"],
                "active_sharpe_float": sharpe_float(daily_ser - baseline_ser.loc[daily_ser.index]),
                "era_sharpe_dict": {era_str: sharpe_float(daily_ser.loc[a:b]) for era_str, a, b in ERA_TUPLE},
                "ruin_date_str": str(result.ruin_date.date()) if result.ruin_date is not None else None}
    log_df = result.log_df[result.log_df["date"] >= pd.Timestamp(START_STR)]
    entry_value_float = float(log_df.loc[log_df["kind_int"] == 1, "value_float"].sum())
    fill_df = log_df[log_df["kind_int"].isin([1, -1, -2])]
    if entry_value_float > 0:
        out_dict["cost_per_round_trip_bp_float"] = float((fill_df["spread_float"].sum() + fill_df["fee_float"].sum()) / entry_value_float * 1e4)
        out_dict["spread_per_round_trip_bp_float"] = float(fill_df["spread_float"].sum() / entry_value_float * 1e4)
        out_dict["fee_per_round_trip_bp_float"] = float(fill_df["fee_float"].sum() / entry_value_float * 1e4)
    return out_dict


def order_stats(result, date_index: pd.DatetimeIndex) -> dict:
    """Fill statistics (identical in every cost case: fills do not depend on costs)."""
    log_df = result.log_df[result.log_df["date"] >= pd.Timestamp(START_STR)]
    years_float = len(window(result.daily_ser)) / 252.0
    entry_df = log_df[log_df["kind_int"] == 1]
    exit_df = log_df[log_df["kind_int"] == -1]
    order_count_int = int(log_df["kind_int"].isin([0, 1]).sum())
    position_delta_ser = pd.Series(np.where(log_df["kind_int"] == 1, 1, np.where(log_df["kind_int"].isin([-1, -2]), -1, 0)), index=log_df["date"])
    held_ser = position_delta_ser.groupby(level=0).sum().reindex(date_index, fill_value=0).cumsum().loc[START_STR:SEAL_END_STR]
    return {"orders_int": order_count_int, "entries_int": len(entry_df),
            "fill_rate_float": len(entry_df) / order_count_int if order_count_int else float("nan"),
            "entry_open_share_float": float((entry_df["code_int"] == 1).mean()) if len(entry_df) else float("nan"),
            "entry_passive_share_float": float((entry_df["code_int"] == 2).mean()) if len(entry_df) else float("nan"),
            "trades_per_year_float": len(entry_df) / years_float,
            "exit_open_share_float": float((exit_df["code_int"] == 1).mean()) if len(exit_df) else float("nan"),
            "exit_passive_share_float": float((exit_df["code_int"] == 2).mean()) if len(exit_df) else float("nan"),
            "exit_forced_share_float": float((exit_df["code_int"] == 3).mean()) if len(exit_df) else float("nan"),
            "delistings_int": int((log_df["kind_int"] == -2).sum()), "cancelled_int": int((log_df["kind_int"] == 9).sum()),
            "mean_positions_float": float(held_ser.mean())}


def capacity_for(result, turnover_df: pd.DataFrame) -> dict:
    from alpha.scout.stations.s6_book import capacity

    fill_df = result.log_df[(result.log_df["date"] >= pd.Timestamp(START_STR)) & result.log_df["kind_int"].isin([1, -1])]
    trade_df = pd.DataFrame({"kind_str": "rebalance", "date": fill_df["date"], "asset": fill_df["asset"], "delta_float": 1.0,
                             "price_float": fill_df["value_float"]})
    out_dict = capacity(trade_df, result.total_value_ser, turnover_df)
    entry_df = fill_df[fill_df["kind_int"] == 1]
    out_dict["entry_passive_share_float"] = float((entry_df["code_int"] == 2).mean()) if len(entry_df) else float("nan")
    return out_dict


def run_pods(panel, inputs: dict, spread_dict: dict) -> tuple[dict, pd.DataFrame, dict]:
    mats = inputs["mats"]
    margin_mat = trade_through_margin_mat(inputs["unadjusted"], [spread_dict["ar"], spread_dict["pooled"]])
    exit_limit = exit_limit_mat(mats["close"], inputs["unadjusted"])
    slip_dict = {"ar": fill_slippage_mat(spread_dict["ar"], FLOOR_SLIP_FLOAT), "pooled": fill_slippage_mat(spread_dict["pooled"], FLOOR_SLIP_FLOAT)}
    baseline_ser = pd.Series(dv2.member_baseline_vec(mats["close"], mats["member"]), index=panel.date_index)
    turnover_df = panel.field("Turnover").astype(float)
    out_dict, daily_dict, pooled_result_dict = {}, {}, {}
    for entry in ENTRY_TUPLE:
        entry_limit = None if entry == "moo" else entry_limit_mat(mats["close"], inputs["unadjusted"], inputs["natr"], float(entry))
        for exit_str in EXIT_TUPLE:
            label_str = config_label(entry, exit_str)
            run = lambda slip, fee, minimum, cap, entry_limit=entry_limit, exit_str=exit_str: limit_book(
                panel.date_index, panel.symbol_list, mats, inputs["high"], inputs["low"], CONFIG.max_positions_int, START_STR, entry_limit,
                exit_str, margin_mat, exit_limit, slip, inputs["scale"], fee, minimum, cap)
            result_dict = {"gross": run(0.0, 0.0, 0.0, 0.0),
                           "ar": run(slip_dict["ar"], FEE_PER_SHARE_FLOAT, MIN_FEE_FLOAT, FEE_CAP_FLOAT),
                           "pooled": run(slip_dict["pooled"], FEE_PER_SHARE_FLOAT, MIN_FEE_FLOAT, FEE_CAP_FLOAT)}
            row_dict = {"entry": entry, "exit_str": exit_str, "orders": order_stats(result_dict["pooled"], panel.date_index)}
            for case_str, result in result_dict.items():
                row_dict[case_str] = pod_metrics(result, baseline_ser)
                daily_dict[f"{label_str}|{case_str}"] = window(result.daily_ser)
            row_dict["capacity_pooled"] = capacity_for(result_dict["pooled"], turnover_df)
            out_dict[label_str] = row_dict
            pooled_result_dict[label_str] = result_dict["pooled"]
            log(f"  {label_str:28s} fill {row_dict['orders']['fill_rate_float']:.2f} trades/yr {row_dict['orders']['trades_per_year_float']:6.0f} "
                f"gross {row_dict['gross']['sharpe_float']:5.2f} AR {row_dict['ar']['sharpe_float']:5.2f} pooled {row_dict['pooled']['sharpe_float']:5.2f} "
                f"cost AR {row_dict['ar'].get('cost_per_round_trip_bp_float', float('nan')):5.0f} pooled {row_dict['pooled'].get('cost_per_round_trip_bp_float', float('nan')):5.0f} bp")
    daily_df = pd.DataFrame(daily_dict).assign(baseline=window(baseline_ser))
    return out_dict, daily_df, pooled_result_dict


# ---------------------------------------------------------------- the adverse-selection table
def date_mean_stats(value_mat: np.ndarray, use_mat: np.ndarray, date_index: pd.DatetimeIndex) -> dict:
    """S3-style: mean of the events' values per date, then over dates; Newey-West t with lag h - 1; eras."""
    count_vec = use_mat.sum(axis=1)
    keep_vec = count_vec > 0
    date_vec = np.where(use_mat, value_mat, 0.0).sum(axis=1)[keep_vec] / count_vec[keep_vec]
    date_ser = pd.Series(date_vec, index=date_index[keep_vec])
    out_dict = {"events_int": int(count_vec.sum()), "dates_int": int(keep_vec.sum())}
    if date_ser.size < 30:
        return {**out_dict, "mean_bp_float": float(date_ser.mean() * 1e4) if date_ser.size else float("nan"), "nw_t_float": float("nan")}
    nw = newey_west_mean_t_stat(date_vec, HORIZON_INT - 1)
    out_dict.update({"mean_bp_float": nw.mean_float * 1e4, "nw_t_float": nw.t_stat_float,
                     "era_bp_dict": {era_str: float(date_ser.loc[a:b].mean() * 1e4) if date_ser.loc[a:b].size else float("nan")
                                     for era_str, a, b in ERA_TUPLE}})
    return out_dict


def event_table(panel, inputs: dict, spread_dict: dict) -> dict:
    mats = inputs["mats"]
    open_mat, close_mat = mats["open"], mats["close"]
    in_window_vec = np.asarray((panel.date_index >= pd.Timestamp(START_STR)) & (panel.date_index <= pd.Timestamp(SEAL_END_STR)))
    with np.errstate(divide="ignore", invalid="ignore"):
        # *** CRITICAL*** forward labels: Close_(T+3) over Open_(T+1) (or Close_T, or the fill price); the panel ends at
        # the seal, so a label that would need a vault bar is NaN. Never used as a feature.
        close_h_mat = np.full(close_mat.shape, np.nan)
        close_h_mat[:-HORIZON_INT] = close_mat[HORIZON_INT:]
        open_next_mat = np.full(open_mat.shape, np.nan)
        open_next_mat[:-1] = open_mat[1:]
        r_moo_mat = close_h_mat / open_next_mat - 1.0
        r_close_mat = close_h_mat / close_mat - 1.0
    eligible_mat = inputs["base"] & np.isfinite(r_moo_mat) & np.isfinite(r_close_mat) & in_window_vec[:, None]
    count_vec = eligible_mat.sum(axis=1)
    with np.errstate(invalid="ignore"):
        baseline_moo_vec = np.where(eligible_mat, r_moo_mat, 0.0).sum(axis=1) / np.maximum(count_vec, 1)
        baseline_close_vec = np.where(eligible_mat, r_close_mat, 0.0).sum(axis=1) / np.maximum(count_vec, 1)
    excess_moo_mat = r_moo_mat - baseline_moo_vec[:, None]
    excess_close_mat = r_close_mat - baseline_close_vec[:, None]
    event_mat = inputs["event"] & eligible_mat
    half_spread_mat_ = np.fmax(np.nan_to_num(spread_dict["pooled"], nan=FLOOR_SLIP_FLOAT), FLOOR_SLIP_FLOAT)  # charged on T+1 fills
    margin_mat = trade_through_margin_mat(inputs["unadjusted"], [spread_dict["ar"], spread_dict["pooled"]])
    out_dict = {"all_events": {"moo_entry": date_mean_stats(excess_moo_mat, event_mat, panel.date_index),
                               "close_anchor": date_mean_stats(excess_close_mat, event_mat, panel.date_index),
                               "moo_entry_net_pooled": date_mean_stats(excess_moo_mat - 2.0 * half_spread_mat_, event_mat, panel.date_index)}}
    for entry in ENTRY_TUPLE:
        if entry == "moo":
            continue
        limit = entry_limit_mat(close_mat, inputs["unadjusted"], inputs["natr"], float(entry))
        code_mat, price_mat = event_fill_mats(open_mat, inputs["low"], limit, margin_mat)
        filled_mat = event_mat & (code_mat > 0)
        unfilled_mat = event_mat & (code_mat == 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            excess_fill_mat = close_h_mat / price_mat - 1.0 - baseline_moo_vec[:, None]
            saving_mat = 1.0 - price_mat / open_next_mat  # vs buying the same event at the open
        net_fill_mat = excess_fill_mat - np.where(code_mat == 1, half_spread_mat_, 0.0) - half_spread_mat_
        out_dict[f"k{entry:g}"] = {
            "fill_share_float": float(filled_mat.sum() / max(event_mat.sum(), 1)),
            "open_fill_share_of_fills_float": float((event_mat & (code_mat == 1)).sum() / max(filled_mat.sum(), 1)),
            "filled_moo_entry": date_mean_stats(excess_moo_mat, filled_mat, panel.date_index),
            "unfilled_moo_entry": date_mean_stats(excess_moo_mat, unfilled_mat, panel.date_index),
            "filled_from_fill_price": date_mean_stats(excess_fill_mat, filled_mat, panel.date_index),
            "filled_from_fill_price_net_pooled": date_mean_stats(net_fill_mat, filled_mat, panel.date_index),
            "filled_close_anchor": date_mean_stats(excess_close_mat, filled_mat, panel.date_index),
            "unfilled_close_anchor": date_mean_stats(excess_close_mat, unfilled_mat, panel.date_index),
            "filled_saving_vs_open": date_mean_stats(saving_mat, filled_mat, panel.date_index),
        }
    return out_dict


# ---------------------------------------------------------------- driver
def run_universe(superset, name_str: str) -> dict:
    panel = build_panel(superset, name_str)
    member_count_ser = panel.member_df.loc[START_STR:].sum(axis=1)
    log(f"{name_str}: {len(panel.symbol_list)} symbols, members median {int(member_count_ser.median())} (min {int(member_count_ser.min())})")
    spread_dict = spread_dict_for(superset, panel)
    inputs = signal_inputs(panel)
    out_dict = {"universe_str": name_str, "start_str": START_STR, "end_str": SEAL_END_STR, "symbol_count_int": len(panel.symbol_list),
                "members_median_int": int(member_count_ser.median()), "members_min_int": int(member_count_ser.min()),
                "snapshot_id_str": panel.snapshot_id_str, "events_int": int(inputs["event"][np.asarray(panel.date_index >= START_STR)].sum())}
    out_dict["events"] = event_table(panel, inputs, spread_dict)
    for key_str, row in out_dict["events"].items():
        if key_str == "all_events":
            log(f"  events all: {row['moo_entry']['mean_bp_float']:+.1f} bp t {row['moo_entry']['nw_t_float']:.2f}")
        else:
            log(f"  events {key_str}: fill {row['fill_share_float']:.2f}; filled {row['filled_moo_entry']['mean_bp_float']:+.1f} bp "
                f"vs unfilled {row['unfilled_moo_entry']['mean_bp_float']:+.1f} (at open); filled from fill {row['filled_from_fill_price']['mean_bp_float']:+.1f}; "
                f"close-anchored {row['filled_close_anchor']['mean_bp_float']:+.1f} vs {row['unfilled_close_anchor']['mean_bp_float']:+.1f}")
    gc.collect()
    pods, daily_df, _ = run_pods(panel, inputs, spread_dict)
    out_dict["pods"] = pods
    write_json(OUT_PATH / "universes" / f"{slug(name_str)}.json", out_dict)
    daily_df.to_csv(OUT_PATH / "universes" / f"{slug(name_str)}_daily.csv")
    return out_dict


# ---------------------------------------------------------------- fill-model stress (diagnostic, not a trial axis)
STRESS_SPREAD_FRACTION_TUPLE = (0.1, 0.5, 1.0)  # 0.1 = the registered trade-through margin


def run_fill_stress(superset, name_str: str) -> dict:
    """The registered grid's limit cells re-run with a stricter trade-through margin: the price must trade through the limit
    by max(one tick, f x the larger half-spread) for f in STRESS_SPREAD_FRACTION_TUPLE. f = 1 roughly asks the MID to reach
    the limit, i.e. no passive fill on a print that only hit the bid. A diagnostic of fill optimism, not a selection axis."""
    panel = build_panel(superset, name_str)
    spread_dict = spread_dict_for(superset, panel)
    inputs = signal_inputs(panel)
    mats = inputs["mats"]
    exit_limit = exit_limit_mat(mats["close"], inputs["unadjusted"])
    slip_dict = {k: fill_slippage_mat(spread_dict[k], FLOOR_SLIP_FLOAT) for k in ("ar", "pooled")}
    out_dict = {}
    for fraction_float in STRESS_SPREAD_FRACTION_TUPLE:
        margin_mat = trade_through_margin_mat(inputs["unadjusted"], [spread_dict["ar"], spread_dict["pooled"]], spread_fraction_float=fraction_float)
        for entry in ENTRY_TUPLE:
            entry_limit = None if entry == "moo" else entry_limit_mat(mats["close"], inputs["unadjusted"], inputs["natr"], float(entry))
            for exit_str in EXIT_TUPLE:
                if entry == "moo" and exit_str == "moo":
                    continue
                row_dict = {}
                for case_str in ("gross", "ar", "pooled"):
                    slip, fee, minimum, cap = (0.0, 0.0, 0.0, 0.0) if case_str == "gross" else (slip_dict[case_str], FEE_PER_SHARE_FLOAT, MIN_FEE_FLOAT, FEE_CAP_FLOAT)
                    result = limit_book(panel.date_index, panel.symbol_list, mats, inputs["high"], inputs["low"], CONFIG.max_positions_int, START_STR,
                                        entry_limit, exit_str, margin_mat, exit_limit, slip, inputs["scale"], fee, minimum, cap)
                    row_dict[f"sharpe_{case_str}"] = sharpe_float(window(result.daily_ser))
                    if case_str == "pooled":
                        row_dict.update({k: v for k, v in order_stats(result, panel.date_index).items()
                                         if k in ("fill_rate_float", "entry_passive_share_float", "exit_passive_share_float", "trades_per_year_float")})
                out_dict[f"f{fraction_float:g}|{config_label(entry, exit_str)}"] = row_dict
                log(f"  {name_str} f {fraction_float:g} {config_label(entry, exit_str):28s} fill {row_dict['fill_rate_float']:.2f} "
                    f"gross {row_dict['sharpe_gross']:5.2f} AR {row_dict['sharpe_ar']:5.2f} pooled {row_dict['sharpe_pooled']:5.2f}")
    write_json(OUT_PATH / "fill_stress" / f"{slug(name_str)}.json", out_dict)
    return out_dict


# ---------------------------------------------------------------- MCPT (per-asset null over the whole entry x exit grid)
def mcpt_score(panel, spread_dict: dict, cost_str: str) -> tuple[float, int, list]:
    """Plateau choice (alpha.stats.selection) over the registered entry x exit grid of the net active Sharpe (daily net
    return minus the equal-weight members, from the start) under one half-spread model. Returns (the chosen cell's own
    Sharpe, its flat index, every cell's Sharpe)."""
    from alpha.stats.selection import plateau_choice

    inputs = signal_inputs(panel)
    mats = inputs["mats"]
    margin_mat = trade_through_margin_mat(inputs["unadjusted"], [spread_dict["ar"], spread_dict["pooled"]])
    exit_limit = exit_limit_mat(mats["close"], inputs["unadjusted"])
    slip_mat = fill_slippage_mat(spread_dict[cost_str], FLOOR_SLIP_FLOAT)
    baseline_vec = dv2.member_baseline_vec(mats["close"], mats["member"])
    keep_vec = np.asarray((panel.date_index >= pd.Timestamp(START_STR)) & (panel.date_index <= pd.Timestamp(SEAL_END_STR)))
    sharpe_list = []
    for entry in ENTRY_TUPLE:
        entry_limit = None if entry == "moo" else entry_limit_mat(mats["close"], inputs["unadjusted"], inputs["natr"], float(entry))
        for exit_str in EXIT_TUPLE:
            result = limit_book(panel.date_index, panel.symbol_list, mats, inputs["high"], inputs["low"], CONFIG.max_positions_int, START_STR,
                                entry_limit, exit_str, margin_mat, exit_limit, slip_mat, inputs["scale"], FEE_PER_SHARE_FLOAT, MIN_FEE_FLOAT,
                                FEE_CAP_FLOAT)
            sharpe_list.append(sharpe_float(pd.Series((result.daily_ser.to_numpy() - baseline_vec)[keep_vec])))
    choice = plateau_choice(np.nan_to_num(np.array(sharpe_list), nan=-9.0), (len(ENTRY_TUPLE), len(EXIT_TUPLE)))
    return choice.own_sharpe_float, choice.flat_index_int, sharpe_list


def null_spread_dict(permuted, pooled_mat: np.ndarray) -> dict:
    """Spreads on a null draw: the pooled spread stays on its real dates (liquidity is structural, as Turnover and Volume
    in alpha.scout.null); the per-stock Abdi-Ranaldo spread is a price-shape estimator, so it is recomputed on the draw."""
    tick_half = tick_half_spread_mat(permuted.field("Unadjusted Close").to_numpy(dtype=float))
    return {"ar": np.fmax(half_spread_mat(*(permuted.field(f).to_numpy(dtype=float) for f in ("High", "Low", "Close"))), tick_half),
            "pooled": pooled_mat}


def _mcpt_chunk(args) -> np.ndarray:
    seed_int, count_int, name_str, cost_str, snapshot_id_str = args
    from alpha.scout.null import permuted_panel

    superset = load_superset_panel(SUPERSET_NAME_STR, snapshot_id_str, index_name_list=LOAD_INDEX_LIST)
    panel = build_panel(superset, name_str)  # real membership: the null's strata
    pooled_mat = spread_dict_for(superset, panel)["pooled"]
    rng_obj, cache_dict, out_list = np.random.default_rng(seed_int), {}, []
    for _ in range(count_int):
        permuted = permuted_panel(panel, rng_obj, cache_dict)
        out_list.append(mcpt_score(permuted, null_spread_dict(permuted, pooled_mat), cost_str)[0])
        del permuted
        gc.collect()
    return np.array(out_list)


def run_mcpt(superset, name_str: str, worker_count_int: int, cost_str: str = "pooled", permutation_count_int: int = 1000) -> dict:
    from multiprocessing import Pool

    panel = build_panel(superset, name_str)
    observed_float, chosen_int, observed_sharpe_list = mcpt_score(panel, spread_dict_for(superset, panel), cost_str)
    del panel
    gc.collect()
    log(f"MCPT {name_str} ({cost_str}): observed plateau cell {chosen_int}, active Sharpe {observed_float:.3f}; {permutation_count_int} permutations on {worker_count_int} workers")
    chunk_int = permutation_count_int // worker_count_int + 1
    with Pool(worker_count_int) as pool_obj:
        null_vec = np.concatenate(pool_obj.map(_mcpt_chunk, [(9_300 + i, chunk_int, name_str, cost_str, superset.snapshot_id_str)
                                                             for i in range(worker_count_int)]))[:permutation_count_int]
    p_float = float((1 + np.sum(null_vec >= observed_float)) / (1 + null_vec.size))
    grid_list = [config_label(e, x) for e in ENTRY_TUPLE for x in EXIT_TUPLE]
    out_dict = {"universe_str": name_str, "cost_str": cost_str, "observed_active_sharpe_float": observed_float, "chosen_config_str": grid_list[chosen_int],
                "observed_grid_active_sharpe_dict": dict(zip(grid_list, observed_sharpe_list)), "p_value_float": p_float,
                "permutation_count_int": int(null_vec.size), "null_95_float": float(np.quantile(null_vec, 0.95)), "null_mean_float": float(null_vec.mean()),
                "configurations_in_search_int": len(grid_list), "worker_count_int": worker_count_int, "snapshot_id_str": superset.snapshot_id_str,
                "score_str": f"plateau choice over the {len(grid_list)}-cell entry x exit grid of the net ({cost_str}) active Sharpe over EW members"}
    write_json(OUT_PATH / "mcpt" / f"{slug(name_str)}_{cost_str}.json", out_dict)
    np.save(OUT_PATH / "mcpt" / f"{slug(name_str)}_{cost_str}_null.npy", null_vec)
    log(f"MCPT {name_str} ({cost_str}): p {p_float:.4f} (null 95% {out_dict['null_95_float']:.3f}, mean {out_dict['null_mean_float']:.3f})")
    return out_dict


def main() -> None:
    command_str = sys.argv[1]
    superset = load_superset_panel(SUPERSET_NAME_STR, index_name_list=LOAD_INDEX_LIST)
    if command_str == "universe":
        for name_str in sys.argv[2:] or list(UNIVERSE_TUPLE):
            run_universe(superset, name_str)
            gc.collect()
    elif command_str == "fillstress":
        for name_str in sys.argv[2:] or list(UNIVERSE_TUPLE):
            run_fill_stress(superset, name_str)
            gc.collect()
    elif command_str == "mcpt":
        run_mcpt(superset, sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3, sys.argv[4] if len(sys.argv) > 4 else "pooled",
                 int(sys.argv[5]) if len(sys.argv) > 5 else 1000)
    else:
        raise SystemExit(f"unknown command {command_str!r}")


if __name__ == "__main__":
    main()
