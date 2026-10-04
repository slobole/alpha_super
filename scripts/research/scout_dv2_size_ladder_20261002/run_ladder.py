"""DV2 size ladder (registration dv2_size_ladder_20261002): where does DV2's alpha come from?

The rule is FROZEN at the WIRED live configuration (alpha/scout/specs/dv2.py LIVE_CONFIG). Every result uses the sealed
superset panel (data to 2022-12-30) built by build_superset.py.

    PYTHONUTF8=1 uv run python scripts/research/scout_dv2_size_ladder_20261002/run_ladder.py validate
    PYTHONUTF8=1 uv run python scripts/research/scout_dv2_size_ladder_20261002/run_ladder.py universe [name ...]
    PYTHONUTF8=1 uv run python scripts/research/scout_dv2_size_ladder_20261002/run_ladder.py buckets
    PYTHONUTF8=1 uv run python scripts/research/scout_dv2_size_ladder_20261002/run_ladder.py mcpt name [workers]

Per universe (window: 2004-01-01, or the first full year of the index's membership, to 2022-12-30):
  S3    the P7 class-E event study (alpha.scout.stations.s3_edge.run_s3) at h3: regime = complete bar, finite NATR14 and
        DV2(126), Close > SMA200, 126-session return > 0.05, member; event = DV2 < 10 (dv2.s3_inputs, re-built here so
        the features are computed once). Plus a liquidity-aware cost coverage: date-mean excess / date-mean of the
        events' round trip 2 x max(2.5 bp, causal Abdi-Ranaldo half-spread at T).
  pod   the frozen rule on the fast panel replica with costs (alpha.scout.universes.costed_book): gross; (a) engine costs
        (2.5 bp, $0.005/share, $1 minimum) on nominal shares, and on the engine's adjusted share units; (b) 2x costs + 10 bp
        (15 bp, $0.01/share, $2 minimum, nominal); (c) liquidity-aware: max(2.5 bp, half-spread of the session before the
        fill) per side, engine fees on nominal shares. No dividends (the replica's convention; about -0.01 to -0.04
        Sharpe vs the engine). Active Sharpe = Sharpe of the daily return minus the equal-weight members' return
        (dv2.member_baseline_vec).
        Both spread models are floored at half a one-cent tick over the nominal price ($0.005 / Unadjusted Close at T).
        Membership is zeroed before the start, so the first decision is the close of the first session on or after the
        start (the engine's is the close before it: one session later, in every universe alike).
        Nominal-share fees are capped at 1% of the trade value (2% under stress), as IBKR Fixed pricing. A book that falls to
        1% of its capital is ruined: that day is floored at -100% and it stops (ruin_date_str).
        (d) liquidity-pooled: as (c) with the ADV-bucket pooled Abdi-Ranaldo half-spread (alpha.scout.universes.
        pooled_half_spread_mat: 20 ADV63 buckets across every ladder member, 63 sessions of products), because the
        per-stock estimator is dominated by volatility noise for liquid stocks.
  cap   alpha.scout.stations.s6_book.capacity on the engine-cost fills: AUM at which the 5th-percentile fill is 1% of its
        63-session ADV (native Turnover, shifted one session), last three in-sample years.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from register import UNIVERSE_TUPLE

from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH
from alpha.scout.metrics import performance_dict, sharpe_float
from alpha.scout.specs import dv2
from alpha.scout.universes import (
    SEAL_END_STR,
    adv63_mat,
    adv_tercile_mat,
    costed_book,
    fill_slippage_mat,
    half_spread_mat,
    load_superset_panel,
    rule_mats,
    tick_half_spread_mat,
    universe_panel,
)
from alpha.stats.newey_west import newey_west_mean_t_stat

OUT_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "dv2_size_ladder"
SUPERSET_NAME_STR = "size_ladder"
CONFIG = dv2.LIVE_CONFIG
HORIZON_INT = 3
FLOOR_SLIP_FLOAT = 0.00025
COST_CASE_DICT = {  # name -> (slippage float or a spread name, fee per share, minimum fee, fee cap as a fraction of value, share units)
    "gross": (0.0, 0.0, 0.0, 0.0, "nominal"),
    "engine": (0.00025, 0.005, 1.0, 0.01, "nominal"),
    "engine_adjusted_units": (0.00025, 0.005, 1.0, 0.0, "adjusted"),  # the engine's own fee units, no cap (P7 parity)
    "stress_2x_plus_10bp": (2 * 0.00025 + 0.0010, 0.010, 2.0, 0.02, "nominal"),
    "liquidity_aware": ("liquidity_aware", 0.005, 1.0, 0.01, "nominal"),
    "liquidity_pooled": ("liquidity_pooled", 0.005, 1.0, 0.01, "nominal"),
}
log_started_float = time.time()


def log(text_str: str) -> None:
    print(f"[{time.time() - log_started_float:7.0f}s] {text_str}", flush=True)


def slug(name_str: str) -> str:
    return name_str.replace(" ", "_").replace("&", "and")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")


def universe_start_str(superset, index_name_str: str) -> str:
    """2004-01-01, or 1 January of the first full year of the index's membership in the panel."""
    member_mat = superset.member_dict[index_name_str]
    first_ts = superset.date_index[int(np.flatnonzero(np.asarray(member_mat).any(axis=1))[0])]
    first_session_of_year_ts = superset.date_index[superset.date_index.year == first_ts.year][0]
    start_year_int = first_ts.year if first_ts == first_session_of_year_ts else first_ts.year + 1
    return f"{max(2004, start_year_int)}-01-01"


def member_from(member_df: pd.DataFrame, start_str: str) -> pd.DataFrame:
    """Membership zeroed before the start (no entry and no baseline member before it)."""
    member_mat = member_df.to_numpy().copy()
    member_mat[np.asarray(member_df.index < pd.Timestamp(start_str))] = 0
    return pd.DataFrame(member_mat, index=member_df.index, columns=member_df.columns)


# ---------------------------------------------------------------- S3
def s3_masks(panel) -> dict:
    """dv2.s3_inputs' regime / event / indicator on any panel (LIVE entry rule), features computed once."""
    mats = dv2._panel_mats(panel)
    features = dv2.feature_dict(panel.field("Close"), panel.field("High"), panel.field("Low"), CONFIG)
    with np.errstate(invalid="ignore"):
        regime_mat = (mats["complete"] & ~np.isnan(features["natr_mat"]) & ~np.isnan(features["dv2_mat"])
                      & (mats["close"] > features["sma_mat"]) & (features["momentum_mat"] > CONFIG.momentum_min_float))
        event_mat = features["dv2_mat"] < CONFIG.entry_dv2_max_float
    return {"regime": regime_mat, "event": event_mat, "indicator": -features["dv2_mat"], "mats": mats}


def pooled_spread_superset(superset) -> np.ndarray:
    """The ADV-bucket pooled Abdi-Ranaldo half-spread on the whole superset (eligible = member of any ladder index),
    cached next to the superset panel."""
    from alpha.scout.universes import SUPERSET_ROOT_PATH, pooled_half_spread_mat

    # The file name carries the estimator's parameters (pooled_half_spread_mat defaults), so a change cannot reuse it.
    path = SUPERSET_ROOT_PATH / SUPERSET_NAME_STR / superset.snapshot_id_str / "pooled_half_spread_b20_w63_m500_monotone.npy"
    if not path.exists():
        eligible_mat = np.zeros(superset.field_dict["Close"].shape, dtype=bool)
        for member_mat in superset.member_dict.values():
            eligible_mat |= np.asarray(member_mat) == 1
        field = lambda f: np.asarray(superset.field_dict[f], dtype=float)
        out_mat, bucket_mat = pooled_half_spread_mat(field("High"), field("Low"), field("Close"), adv63_mat(field("Turnover")), eligible_mat)
        np.save(path, out_mat.astype(np.float32))
        pd.DataFrame(bucket_mat[:, 1:] * 1e4, index=superset.date_index, columns=[f"b{i:02d}" for i in range(1, bucket_mat.shape[1])]).to_csv(
            OUT_PATH / "pooled_half_spread_bp_by_adv_bucket.csv")
    return np.load(path, mmap_mode="r")


def spread_dict_for(superset, panel) -> dict:
    """{"liquidity_aware": per-stock Abdi-Ranaldo, "liquidity_pooled": ADV-bucket pooled} half-spreads on the panel's axes,
    each floored at half a one-cent tick over the nominal price at T ($0.005 / Unadjusted Close): a stock quoted at $2
    cannot have a half-spread below 25 bp, whatever a daily-bar estimator says."""
    position_vec = pd.Index(superset.symbol_list).get_indexer(panel.symbol_list)
    row_vec = superset.date_index.get_indexer(panel.date_index)
    tick_half = tick_half_spread_mat(panel.field("Unadjusted Close").to_numpy(dtype=float))
    return {"liquidity_aware": np.fmax(half_spread_mat(*(panel.field(f).to_numpy(dtype=float) for f in ("High", "Low", "Close"))), tick_half),
            "liquidity_pooled": np.fmax(np.asarray(pooled_spread_superset(superset)[row_vec][:, position_vec], dtype=float), tick_half)}


def run_universe_s3(panel, spread_dict: dict, masks: dict) -> dict:
    from alpha.scout import features as scout_features
    from alpha.scout.stations.s3_edge import run_s3

    frame = lambda mat: pd.DataFrame(mat, index=panel.date_index, columns=panel.symbol_list)
    report = run_s3(name_str=f"DV2 oversold ({panel.name_str})", panel=panel, regime_mask_df=frame(masks["regime"]),
                    event_mask_df=frame(masks["event"]), horizon_int=HORIZON_INT, indicator_df=frame(masks["indicator"]),
                    liquidity_rank_df=scout_features.turnover_rank(63).compute_fn(panel), expected_sign_int=1)
    event_mat = masks["event"] & masks["regime"] & (panel.member_df.to_numpy() == 1)
    liquidity = {k: liquidity_coverage(panel, event_mat, hs, report.headline_dict["date_mean_excess_float"]) for k, hs in spread_dict.items()}
    return {"headline": report.headline_dict, "eras": report.table_dict["eras"], "years": report.table_dict["years"],
            "liquidity_terciles": report.table_dict.get("liquidity_terciles"), "verdict_str": report.verdict_str,
            "check_list": [list(row) for row in report.check_list], "liquidity_cost": liquidity}


def liquidity_coverage(panel, event_mat: np.ndarray, half_spread: np.ndarray, date_mean_excess_float: float) -> dict:
    """Round trip 2 x max(2.5 bp, half-spread at T) per event, averaged per date then over dates (as the excess)."""
    with np.errstate(invalid="ignore"):
        forward_ok_mat = np.isfinite(panel.field("Close").shift(-HORIZON_INT).to_numpy(dtype=float) / panel.field("Open").shift(-1).to_numpy(dtype=float))
    use_mat = event_mat & forward_ok_mat
    cost_mat = np.where(use_mat, 2.0 * np.fmax(np.nan_to_num(half_spread, nan=FLOOR_SLIP_FLOAT), FLOOR_SLIP_FLOAT), 0.0)
    count_vec = use_mat.sum(axis=1)
    date_cost_vec = cost_mat.sum(axis=1)[count_vec > 0] / count_vec[count_vec > 0]
    hs_vec = half_spread[use_mat]
    adv_vec = adv63_mat(panel.field("Turnover").to_numpy(dtype=float))[use_mat]
    return {"date_mean_round_trip_float": float(date_cost_vec.mean()),
            "coverage_float": float(date_mean_excess_float / date_cost_vec.mean()),
            "event_median_half_spread_float": float(np.nanmedian(hs_vec)),
            "event_median_adv63_float": float(np.nanmedian(adv_vec)),
            "event_adv63_missing_share_float": float(np.mean(~np.isfinite(adv_vec)))}


# ---------------------------------------------------------------- pods
def pod_metrics(result, baseline_ser: pd.Series, start_str: str) -> dict:
    daily_ser = result.daily_ser.loc[start_str:SEAL_END_STR]
    performance = performance_dict(daily_ser)
    entry_df = result.fill_df[result.fill_df["kind_int"] == 1]
    years_float = len(daily_ser) / 252.0
    return {"cagr_float": performance["cagr_float"], "sharpe_float": performance["sharpe_float"],
            "max_drawdown_float": performance["max_drawdown_float"], "volatility_float": performance["volatility_float"],
            "active_sharpe_float": sharpe_float(daily_ser - baseline_ser.loc[daily_ser.index]),
            "trades_per_year_float": len(entry_df[entry_df["date"] >= pd.Timestamp(start_str)]) / years_float,
            "year_return_dict": performance["year_return_dict"]}


def run_universe_pods(panel, spread_dict: dict, start_str: str) -> tuple[dict, dict]:
    from alpha.scout.stations.s6_book import capacity

    mats = rule_mats(panel, CONFIG)
    baseline_ser = pd.Series(dv2.member_baseline_vec(mats["close"], mats["member"]), index=panel.date_index)
    slip_mat_dict = {k: fill_slippage_mat(hs, FLOOR_SLIP_FLOAT) for k, hs in spread_dict.items()}
    out_dict, result_dict = {}, {}
    for case_str, (slip, fee_float, min_fee_float, cap_float, unit_str) in COST_CASE_DICT.items():
        result = costed_book(panel, CONFIG, slip_mat_dict[slip] if isinstance(slip, str) else slip, fee_float, min_fee_float,
                             share_unit_str=unit_str, start_date_str=start_str, mats=mats, max_fee_fraction_float=cap_float)
        out_dict[case_str] = pod_metrics(result, baseline_ser, start_str)
        out_dict[case_str]["ruin_date_str"] = str(result.ruin_date.date()) if result.ruin_date is not None else None
        result_dict[case_str] = result
    engine = result_dict["engine"]
    fill_df = engine.fill_df[engine.fill_df["date"] >= pd.Timestamp(start_str)]
    trade_df = pd.DataFrame({"kind_str": "rebalance", "date": fill_df["date"], "asset": fill_df["asset"], "delta_float": 1.0,
                             "price_float": fill_df["value_float"]})
    out_dict["capacity"] = capacity(trade_df, engine.total_value_ser, panel.field("Turnover").astype(float))
    entry_row_vec = panel.date_index.get_indexer(fill_df.loc[fill_df["kind_int"] == 1, "date"])
    entry_asset_vec = pd.Index(panel.symbol_list).get_indexer(fill_df.loc[fill_df["kind_int"] == 1, "asset"])
    out_dict["liquidity_slippage"] = {}
    for key_str, slip_mat in slip_mat_dict.items():
        charged_vec = slip_mat[entry_row_vec, entry_asset_vec]
        out_dict["liquidity_slippage"][key_str] = {"entry_median_bp_float": float(np.median(charged_vec) * 1e4),
                                                   "entry_mean_bp_float": float(np.mean(charged_vec) * 1e4),
                                                   "entry_floor_share_float": float(np.mean(charged_vec <= FLOOR_SLIP_FLOAT))}
    _, _, hold_list = dv2.fast_daily_list_panel(panel, [{}], base_config=CONFIG, return_hold_bool=True)
    out_dict["median_hold_sessions_int"] = int(np.median(hold_list[0])) if len(hold_list[0]) else None
    daily_frame = pd.DataFrame({k: r.daily_ser for k, r in result_dict.items()}).assign(baseline=baseline_ser).loc[start_str:SEAL_END_STR]
    return out_dict, {"daily": daily_frame, "fills": engine.fill_df}


def run_universe(superset, index_name_str: str) -> dict:
    start_str = universe_start_str(superset, index_name_str)
    panel = universe_panel(superset, index_name_str, member_from_str=start_str)
    log(f"{index_name_str}: {len(panel.symbol_list)} symbols, start {start_str}")
    spread_dict = spread_dict_for(superset, panel)
    half_spread = spread_dict["liquidity_aware"]
    masks = s3_masks(panel)
    member_count_ser = panel.member_df.loc[start_str:].sum(axis=1)
    out_dict = {"index_name_str": index_name_str, "start_str": start_str, "end_str": SEAL_END_STR, "symbol_count_int": len(panel.symbol_list),
                "members_median_int": int(member_count_ser.median()), "members_min_int": int(member_count_ser.min()),
                "members_max_int": int(member_count_ser.max()), "snapshot_id_str": panel.snapshot_id_str,
                "member_half_spread_median_bp_by_year": {
                    int(y): float(np.nanmedian(half_spread[(panel.date_index.year == y) & (panel.date_index >= start_str)][
                        (panel.member_df.to_numpy() == 1)[(panel.date_index.year == y) & (panel.date_index >= start_str)]]) * 1e4)
                    for y in sorted(set(panel.date_index[panel.date_index >= start_str].year))}}
    out_dict["s3"] = run_universe_s3(panel, spread_dict, masks)
    del masks
    gc.collect()
    log(f"{index_name_str}: S3 done: {out_dict['s3']['headline']['date_mean_excess_float'] * 1e4:+.1f} bp t {out_dict['s3']['headline']['nw_t_float']:.2f}")
    pods, series = run_universe_pods(panel, spread_dict, start_str)
    out_dict["pod"] = pods
    log(f"{index_name_str}: pods done: net Sharpe {pods['engine']['sharpe_float']:.2f}, stress {pods['stress_2x_plus_10bp']['sharpe_float']:.2f}, "
        f"liquidity {pods['liquidity_aware']['sharpe_float']:.2f}, pooled {pods['liquidity_pooled']['sharpe_float']:.2f}")
    folder_path = OUT_PATH / "universes"
    write_json(folder_path / f"{slug(index_name_str)}.json", out_dict)
    series["daily"].to_csv(folder_path / f"{slug(index_name_str)}_daily.csv")
    series["fills"].to_csv(folder_path / f"{slug(index_name_str)}_fills.csv", index=False)
    return out_dict


# ---------------------------------------------------------------- size buckets inside the broad universes
def lite_s3(forward_mat: np.ndarray, regime_mat: np.ndarray, event_mat: np.ndarray, bucket_mat: np.ndarray, date_index: pd.DatetimeIndex,
            spread_dict: dict, adv_mat: np.ndarray, baseline_mat: np.ndarray | None = None, placebo_bool: bool = True) -> dict:
    """S3's headline estimators (s3_edge: date-level mean of event excess, Newey-West lag h-1, shift placebo, positive
    years, eras) for the events inside one bucket. Excess over the same-date eligible members of the same bucket
    (`baseline_mat` None) or of `baseline_mat`'s eligible set (a whole-universe baseline)."""
    from alpha.scout.stations.s3_edge import ERA_TUPLE, _shift_placebo_p

    eligible_mat = regime_mat & bucket_mat
    reference_mat = eligible_mat if baseline_mat is None else baseline_mat
    with np.errstate(invalid="ignore"):
        reference_forward_mat = np.where(reference_mat, forward_mat, np.nan)
        count_vec = np.isfinite(reference_forward_mat).sum(axis=1)
        baseline_vec = np.where(count_vec > 0, np.nansum(reference_forward_mat, axis=1) / np.maximum(count_vec, 1), np.nan)
    excess_mat = forward_mat - baseline_vec[:, None]
    use_mat = event_mat & eligible_mat & np.isfinite(excess_mat)
    event_count_vec = use_mat.sum(axis=1)
    date_mask_vec = event_count_vec > 0
    date_vec = np.where(use_mat, excess_mat, 0.0).sum(axis=1)[date_mask_vec] / event_count_vec[date_mask_vec]
    date_ser = pd.Series(date_vec, index=date_index[date_mask_vec])
    out_dict = {"events_int": int(event_count_vec.sum()), "event_dates_int": int(date_mask_vec.sum())}
    if date_ser.size < 30:
        return {**out_dict, "date_mean_excess_float": float(date_ser.mean()) if date_ser.size else float("nan"), "nw_t_float": float("nan")}
    nw = newey_west_mean_t_stat(date_vec, HORIZON_INT - 1)
    year_ser = date_ser.groupby(date_ser.index.year).mean()
    cost_dict = {}
    for key_str, half_spread in spread_dict.items():
        cost_mat = np.where(use_mat, 2.0 * np.fmax(np.nan_to_num(half_spread, nan=FLOOR_SLIP_FLOAT), FLOOR_SLIP_FLOAT), 0.0)
        date_cost_float = float((cost_mat.sum(axis=1)[date_mask_vec] / event_count_vec[date_mask_vec]).mean())
        cost_dict[key_str] = {"round_trip_float": date_cost_float, "coverage_float": nw.mean_float / date_cost_float,
                              "event_median_half_spread_bp_float": float(np.nanmedian(half_spread[use_mat]) * 1e4)}
    out_dict.update({
        "date_mean_excess_float": nw.mean_float, "nw_t_float": nw.t_stat_float,
        "positive_year_share_float": float((year_ser > 0).mean()),
        "eras": {era_str: float(date_ser.loc[a:b].mean()) if date_ser.loc[a:b].size else float("nan") for era_str, a, b in ERA_TUPLE},
        "liquidity_cost": cost_dict,
        "event_median_adv63_musd_float": float(np.nanmedian(adv_mat[use_mat]) / 1e6),
    })
    if placebo_bool:
        frame = lambda mat: pd.DataFrame(mat)
        out_dict["placebo_p_float"] = _shift_placebo_p(frame(excess_mat), frame(event_mat & eligible_mat), frame(eligible_mat), 1)
    return out_dict


def bucket_study(superset, universe_label_str: str, union_index_list: list[str], bucket_rule_list: list, adv_index_list: list[str]) -> dict:
    """One rule on one broad universe: S3 per membership bucket and per ADV63 tercile (among `adv_index_list` members)."""
    start_str = "2004-01-01"
    union_symbol_set = set()
    for name_str in union_index_list:
        union_symbol_set |= set(superset.ever_member_symbol_list(name_str))
    symbol_list = sorted(union_symbol_set)
    # The panel's membership = the union of the indexes; each index's own flag on the same columns.
    flag_dict = {name: universe_panel(superset, name, member_from_str=start_str, symbol_list=symbol_list, field_tuple=()).member_df.to_numpy() == 1
                 for name in superset.member_dict}
    panel = universe_panel(superset, union_index_list[0], member_from_str=start_str, symbol_list=symbol_list)
    union_mat = np.zeros_like(flag_dict[union_index_list[0]])
    for name_str in union_index_list:
        union_mat |= flag_dict[name_str]
    log(f"buckets {universe_label_str}: {len(symbol_list)} symbols")
    masks = s3_masks(panel)
    with np.errstate(invalid="ignore", divide="ignore"):
        forward_mat = panel.field("Close").shift(-HORIZON_INT).to_numpy(dtype=float) / panel.field("Open").shift(-1).to_numpy(dtype=float) - 1.0
    regime_mat = masks["regime"] & union_mat
    event_mat = masks["event"]
    spread_dict = spread_dict_for(superset, panel)
    adv_mat = adv63_mat(panel.field("Turnover").to_numpy(dtype=float))
    date_index = panel.date_index
    del masks
    gc.collect()

    out_dict = {"universe_str": universe_label_str, "union_index_list": union_index_list, "start_str": start_str, "horizon_int": HORIZON_INT}
    out_dict["whole_universe"] = lite_s3(forward_mat, regime_mat, event_mat, union_mat, date_index, spread_dict, adv_mat)
    log(f"buckets {universe_label_str}: whole {out_dict['whole_universe']['date_mean_excess_float'] * 1e4:+.1f} bp t {out_dict['whole_universe']['nw_t_float']:.2f}")
    membership_rows = []
    for label_str, rule_fn in bucket_rule_list:
        bucket_mat = rule_fn(flag_dict) & union_mat
        row = {"bucket_str": label_str, "within_bucket": lite_s3(forward_mat, regime_mat, event_mat, bucket_mat, date_index, spread_dict, adv_mat),
               "vs_whole_universe": lite_s3(forward_mat, regime_mat, event_mat, bucket_mat, date_index, spread_dict, adv_mat,
                                            baseline_mat=regime_mat, placebo_bool=False),
               "members_median_int": int(np.median(bucket_mat[np.asarray(date_index >= start_str)].sum(axis=1)))}
        membership_rows.append(row)
        log(f"  {label_str}: {row['within_bucket']['date_mean_excess_float'] * 1e4:+.1f} bp t {row['within_bucket']['nw_t_float']:.2f} "
            f"({row['within_bucket']['events_int']} events)")
    out_dict["membership_buckets"] = membership_rows
    adv_universe_mat = np.zeros_like(union_mat)
    for name_str in adv_index_list:
        adv_universe_mat |= flag_dict[name_str]
    tercile_mat = adv_tercile_mat(adv_mat, adv_universe_mat)
    adv_rows = []
    for tercile_int, label_str in ((1, "low ADV63"), (2, "middle ADV63"), (3, "high ADV63")):
        bucket_mat = (tercile_mat == tercile_int) & adv_universe_mat
        row = {"bucket_str": label_str, "within_bucket": lite_s3(forward_mat, regime_mat, event_mat, bucket_mat, date_index, spread_dict, adv_mat),
               "vs_whole_universe": lite_s3(forward_mat, regime_mat & adv_universe_mat, event_mat, bucket_mat, date_index, spread_dict, adv_mat,
                                            baseline_mat=regime_mat & adv_universe_mat, placebo_bool=False)}
        adv_rows.append(row)
        log(f"  {label_str}: {row['within_bucket']['date_mean_excess_float'] * 1e4:+.1f} bp t {row['within_bucket']['nw_t_float']:.2f}")
    unranked_share_float = float((event_mat & regime_mat & adv_universe_mat & (tercile_mat == 0)).sum() / max(1, (event_mat & regime_mat & adv_universe_mat).sum()))
    out_dict["adv_terciles"] = {"ranked_among_str": " | ".join(adv_index_list), "event_unranked_share_float": unranked_share_float, "rows": adv_rows}
    write_json(OUT_PATH / "buckets" / f"{slug(universe_label_str)}.json", out_dict)
    return out_dict


def run_buckets(superset) -> None:
    russell_rule_list = [
        ("Russell Top 200 (mega)", lambda f: f["Russell Top 200"]),
        ("Russell 1000 ex Top 200 (mid)", lambda f: f["Russell 1000"] & ~f["Russell Top 200"]),
        ("Russell 2000 ex Micro Cap (small)", lambda f: f["Russell 2000"] & ~f["Russell Micro Cap"]),
        ("Russell 2000 in Micro Cap (small-micro)", lambda f: f["Russell 2000"] & f["Russell Micro Cap"]),
        ("Micro Cap ex Russell 2000 (micro)", lambda f: f["Russell Micro Cap"] & ~f["Russell 2000"] & ~f["Russell 1000"]),
    ]
    bucket_study(superset, "Russell 3000 + Micro Cap", ["Russell 3000", "Russell Micro Cap"], russell_rule_list, ["Russell 3000"])
    gc.collect()
    sp_rule_list = [
        ("S&P 100 (mega)", lambda f: f["S&P 100"]),
        ("S&P 500 ex S&P 100 (large)", lambda f: f["S&P 500"] & ~f["S&P 100"]),
        ("S&P MidCap 400 (mid)", lambda f: f["S&P MidCap 400"]),
        ("S&P SmallCap 600 (small)", lambda f: f["S&P SmallCap 600"]),
    ]
    bucket_study(superset, "S&P Composite 1500", ["S&P Composite 1500"], sp_rule_list, ["S&P Composite 1500"])


# ---------------------------------------------------------------- validation against P7 (S&P 500)
def run_validate(superset) -> None:
    import pickle

    from alpha.scout.panel import load_panel
    from alpha.scout.stations.s3_edge import run_s3

    out_dict = {}
    scout_panel = load_panel("S&P 500")
    mine = universe_panel(superset, "S&P 500", member_from_str="2004-01-01")
    masked_scout = type(scout_panel)(scout_panel.name_str, scout_panel.field_dict, member_from(scout_panel.member_df, "2004-01-01"),
                                     scout_panel.snapshot_id_str, True)
    gross_scout = pd.Series(dv2.fast_daily_list_panel(masked_scout, [{}])[0][0], index=scout_panel.date_index).loc["2004-01-01":]
    gross_mine = pd.Series(dv2.fast_daily_list_panel(mine, [{}])[0][0], index=mine.date_index).loc["2004-01-01":]
    joined = pd.concat([gross_scout, gross_mine], axis=1, keys=["scout", "mine"]).dropna()
    member_scout = scout_panel.member_df.loc["2004-01-01":]
    member_mine = mine.member_df.loc["2004-01-01":].reindex(columns=member_scout.columns, fill_value=0)
    out_dict["gross_replica_scout_panel_vs_superset"] = {
        "sharpe_scout_float": sharpe_float(joined["scout"]), "sharpe_superset_float": sharpe_float(joined["mine"]),
        "correlation_float": float(joined.corr().iloc[0, 1]), "max_abs_daily_diff_float": float((joined["scout"] - joined["mine"]).abs().max()),
        "identical_days_share_float": float(((joined["scout"] - joined["mine"]).abs() < 1e-9).mean()),
        "member_cells_scout_int": int(member_scout.to_numpy().sum()), "member_cells_superset_int": int(member_mine.to_numpy().sum()),
        "member_cells_differ_int": int((member_scout.to_numpy() != member_mine.reindex(index=member_scout.index, fill_value=0).to_numpy()).sum()),
    }
    with open(MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "reaudition" / "DV2" / "bundle.pkl", "rb") as file_obj:
        bundle = pickle.load(file_obj)
    s4 = bundle["s4"]
    p7_net_ser = s4.grid_df[s4.live_label_str].loc["2004-01-01":SEAL_END_STR]
    for case_str in ("engine_adjusted_units", "engine", "stress_2x_plus_10bp"):
        slip, fee_float, min_fee_float, cap_float, unit_str = COST_CASE_DICT[case_str]
        for label_str, panel in (("scout_panel", masked_scout), ("superset", mine)):
            result = costed_book(panel, CONFIG, slip, fee_float, min_fee_float, share_unit_str=unit_str, start_date_str="2004-01-01",
                                 max_fee_fraction_float=cap_float)
            ser = result.daily_ser.loc["2004-01-01":SEAL_END_STR]
            both = pd.concat([ser, p7_net_ser], axis=1, keys=["replica", "p7"]).dropna()
            out_dict[f"{case_str}_{label_str}"] = {"sharpe_float": sharpe_float(ser), "cagr_float": performance_dict(ser)["cagr_float"],
                                                   "correlation_with_p7_engine_net_float": float(both.corr().iloc[0, 1])}
    out_dict["p7_engine_net"] = {"sharpe_float": sharpe_float(p7_net_ser), "cagr_float": performance_dict(p7_net_ser)["cagr_float"],
                                 "stress_sharpe_float": s4.cost_dict["live"]["stressed_sharpe_float"],
                                 "gross_sharpe_float": s4.cost_dict["live"]["gross_sharpe_float"]}
    # S3: P7's own call on the Scout panel (full sealed window) must give +9.1 bp, t 3.96, 120,965 events.
    p7_s3 = run_s3(**dv2.s3_inputs(panel=scout_panel)).headline_dict
    out_dict["s3_p7_call_scout_panel"] = {k: p7_s3[k] for k in ("events_int", "date_mean_excess_float", "nw_t_float", "placebo_p_float", "cost_coverage_float")}
    write_json(OUT_PATH / "validate_sp500.json", out_dict)
    log(json.dumps(out_dict, indent=1, default=str))


# ---------------------------------------------------------------- MCPT (per-asset null, frozen rule = one configuration)
_STATE: dict = {}


def _mcpt_score(panel, start_str: str) -> float:
    scored = type(panel)(panel.name_str, panel.field_dict, member_from(panel.member_df, start_str), panel.snapshot_id_str, True)
    daily_list, baseline_vec = dv2.fast_daily_list_panel(scored, [{}], base_config=CONFIG)
    keep_vec = np.asarray(panel.date_index >= pd.Timestamp(start_str))
    return sharpe_float(pd.Series((daily_list[0] - baseline_vec)[keep_vec]))


def _mcpt_chunk(args) -> np.ndarray:
    seed_int, count_int, index_name_str, start_str, snapshot_id_str = args
    from alpha.scout.null import permuted_panel

    superset = load_superset_panel(SUPERSET_NAME_STR, snapshot_id_str, index_name_list=[index_name_str])
    panel = universe_panel(superset, index_name_str)  # real membership: the null's strata
    rng_obj, cache_dict = np.random.default_rng(seed_int), {}
    out_list = []
    for _ in range(count_int):
        out_list.append(_mcpt_score(permuted_panel(panel, rng_obj, cache_dict), start_str))
    return np.array(out_list)


def run_mcpt(superset, index_name_str: str, worker_count_int: int, permutation_count_int: int = 1000) -> dict:
    from multiprocessing import Pool

    start_str = universe_start_str(superset, index_name_str)
    panel = universe_panel(superset, index_name_str)
    observed_float = _mcpt_score(panel, start_str)
    del panel
    gc.collect()
    log(f"MCPT {index_name_str}: observed active Sharpe {observed_float:.3f}; {permutation_count_int} permutations on {worker_count_int} workers")
    chunk_int = permutation_count_int // worker_count_int + 1
    with Pool(worker_count_int) as pool_obj:
        null_vec = np.concatenate(pool_obj.map(_mcpt_chunk, [(9_100 + i, chunk_int, index_name_str, start_str, superset.snapshot_id_str)
                                                             for i in range(worker_count_int)]))[:permutation_count_int]
    p_float = float((1 + np.sum(null_vec >= observed_float)) / (1 + null_vec.size))
    out_dict = {"index_name_str": index_name_str, "start_str": start_str, "observed_active_sharpe_float": observed_float,
                "p_value_float": p_float, "permutation_count_int": int(null_vec.size), "null_95_float": float(np.quantile(null_vec, 0.95)),
                "null_mean_float": float(null_vec.mean()), "worker_count_int": worker_count_int, "snapshot_id_str": superset.snapshot_id_str,
                "score_str": "Sharpe of daily (LIVE_CONFIG gross replica - equal-weight members), from the universe start; one configuration (frozen rule)"}
    write_json(OUT_PATH / "mcpt" / f"{slug(index_name_str)}.json", out_dict)
    np.save(OUT_PATH / "mcpt" / f"{slug(index_name_str)}_null.npy", null_vec)
    log(f"MCPT {index_name_str}: p {p_float:.4f} (null 95% {out_dict['null_95_float']:.3f})")
    return out_dict


def main() -> None:
    command_str = sys.argv[1]
    superset = load_superset_panel(SUPERSET_NAME_STR)
    if command_str == "validate":
        run_validate(superset)
    elif command_str == "universe":
        for name_str in sys.argv[2:] or [n for n, _ in UNIVERSE_TUPLE]:
            run_universe(superset, name_str)
            gc.collect()
    elif command_str == "buckets":
        run_buckets(superset)
    elif command_str == "mcpt":
        run_mcpt(superset, sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4)
    else:
        raise SystemExit(f"unknown command {command_str!r}")


if __name__ == "__main__":
    main()
