"""DV2 limit-anchor study (registration dv2_limit_anchor_20261003): where should the entry limit sit?

Every entry limit (anchor x offset measure) is calibrated to the same realized fill rates (25 / 40 / 60%) on 2004-2012,
then run unchanged on 2004-2022 and compared with the parent study's Close - k x NATR14 at the same fill target. The DV2
signal is FROZEN (alpha/scout/specs/dv2.py LIVE_CONFIG); the book, fill rules and costs are the parent's limit_book.py.

    PYTHONUTF8=1 uv run python scripts/research/scout_dv2_limit_anchor_20261003/run_anchor.py universe [name ...]
    PYTHONUTF8=1 uv run python scripts/research/scout_dv2_limit_anchor_20261003/run_anchor.py moo [name ...]
    PYTHONUTF8=1 uv run python scripts/research/scout_dv2_limit_anchor_20261003/run_anchor.py band [name ...]
    PYTHONUTF8=1 uv run python scripts/research/scout_dv2_limit_anchor_20261003/run_anchor.py mcpt [workers] [permutations]

universe  per universe: the correlation of NATR14 with the other measures on signal days; then for each anchor x measure
          x fill target the calibrated parameter, the limit-exit book under gross / AR / pooled costs, the fill stress
          (f = 0.5, 1.0; f = 0.1 is the main run), the event-level fill share and post-fill-day continuation (Close_(T+1) ->
          Close_(T+3) excess over same-date eligible members, filled vs unfilled), the fill overlap with the baseline, and
          the paired stationary-bootstrap Sharpe difference against close x natr14 at the same target.
moo       the market-on-open exit for the best two S&P 500 entries (mean of AR and pooled net Sharpe, limit exit) and for
          the baseline at the same targets, in every universe, with each universe's own calibrated parameters.
band      diagnostic: every limit-exit cell with its calibrated parameter x 0.9 .. 1.1 (path jitter of the slot book).
mcpt      the per-asset panel MCPT (A8 null) over the registered S&P 500 entry grid (anchor x measure x target, limit exit),
          plateau choice of the net active Sharpe; the calibrated parameters are held fixed on the null draws (calibration
          targets a fill rate, not a return).
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT_PATH))
sys.path.insert(0, str(REPO_ROOT_PATH / "scripts" / "research" / "scout_dv2_limit_entry_20261002"))

import run_limit as parent

from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH
from alpha.scout.metrics import sharpe_float
from alpha.scout.universes import (
    SEAL_END_STR,
    fill_slippage_mat,
    load_superset_panel,
)
from alpha.stats.bootstrap import stationary_bootstrap_index_mat
from scripts.research.scout_dv2_limit_anchor_20261003.anchor_book import (
    ExcursionQuantile,
    book_fill_rate,
    calibrate,
    close_anchor_limit_mat,
    close_excursion_mat,
    close_return_std_mat,
    natr_fraction_mat,
    open_anchor_limit_mat,
    open_excursion_mat,
    rolling_mean_mat,
)
from scripts.research.scout_dv2_limit_anchor_20261003.register import (
    ANCHOR_TUPLE,
    BASELINE_TUPLE,
    CALIBRATION_END_STR,
    CALIBRATION_START_STR,
    FILL_TARGET_TUPLE,
    MAIN_UNIVERSE_STR,
    MEASURE_TUPLE,
    UNIVERSE_TUPLE,
)
from scripts.research.scout_dv2_limit_entry_20261002.limit_book import (
    event_fill_mats,
    exit_limit_mat,
    limit_book,
    trade_through_margin_mat,
)

OUT_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "dv2_limit_anchor"
START_STR = parent.START_STR
CONFIG = parent.CONFIG
STRESS_FRACTION_TUPLE = (0.1, 0.5, 1.0)  # 0.1 = the registered trade-through margin
BOOTSTRAP_PATH_INT, BOOTSTRAP_BLOCK_FLOAT, BOOTSTRAP_SEED_INT = 2000, 20.0, 20261003
CORRELATION_MIN_EVENTS_INT = 5
DIAGNOSTIC_PROBABILITY_TUPLE = FILL_TARGET_TUPLE  # the uncalibrated p = target (diagnostic only)
log = parent.log


def cell_label(anchor_str: str, measure_str: str, target_float: float, exit_str: str = "limit") -> str:
    return f"{anchor_str}|{measure_str}|f{target_float:.2f}|{exit_str}"


# ---------------------------------------------------------------- inputs
def prepare(superset, name_str: str) -> dict:
    """Panel, spreads, signal inputs, the offset measures and the shared order inputs of one universe."""
    panel = parent.build_panel(superset, name_str)
    spread_dict = parent.spread_dict_for(superset, panel)
    inputs = parent.signal_inputs(panel)
    mats = inputs["mats"]
    open_mat, close_mat, low_mat = mats["open"], mats["close"], inputs["low"]
    open_excursion = open_excursion_mat(open_mat, low_mat)
    close_excursion = close_excursion_mat(close_mat, low_mat)
    measure_dict = {"natr14": natr_fraction_mat(inputs["natr"]), "std21": close_return_std_mat(close_mat),
                    "dex21_mean": rolling_mean_mat(open_excursion)}
    quantile_dict = {"open": ExcursionQuantile(open_excursion, inputs["event"]), "close": ExcursionQuantile(close_excursion, inputs["event"])}
    return {"panel": panel, "spread": spread_dict, "inputs": inputs, "measures": measure_dict, "quantiles": quantile_dict,
            "close_excursion_mean": rolling_mean_mat(close_excursion),
            "margins": {f: trade_through_margin_mat(inputs["unadjusted"], [spread_dict["ar"], spread_dict["pooled"]], spread_fraction_float=f)
                        for f in STRESS_FRACTION_TUPLE},
            "exit_limit": exit_limit_mat(close_mat, inputs["unadjusted"]),
            "slip": {k: fill_slippage_mat(spread_dict[k], parent.FLOOR_SLIP_FLOAT) for k in ("ar", "pooled")},
            "baseline": pd.Series(parent.dv2.member_baseline_vec(close_mat, mats["member"]), index=panel.date_index)}


def limit_for(prep: dict, anchor_str: str, measure_str: str, param_float: float) -> np.ndarray:
    """Row-T entry limit of one variant. k-measures: offset = k x measure; dex63_quantile: offset = the stock's own
    (1 - p) quantile of its last 63 excursions (open excursion for the open anchor, close excursion for the close anchor)."""
    inputs = prep["inputs"]
    if measure_str == "dex63_quantile":
        offset_mat = prep["quantiles"][anchor_str].offset_mat(param_float)
    else:
        offset_mat = param_float * prep["measures"][measure_str]
    if anchor_str == "close":
        return close_anchor_limit_mat(inputs["mats"]["close"], inputs["unadjusted"], offset_mat)
    return open_anchor_limit_mat(inputs["mats"]["open"], inputs["mats"]["close"], inputs["unadjusted"], offset_mat)


def run_book(prep: dict, entry_limit, exit_str: str, case_str: str, fraction_float: float = 0.1):
    inputs, panel = prep["inputs"], prep["panel"]
    if case_str == "gross":
        slip, fee, minimum, cap = 0.0, 0.0, 0.0, 0.0
    else:
        slip, fee, minimum, cap = prep["slip"][case_str], parent.FEE_PER_SHARE_FLOAT, parent.MIN_FEE_FLOAT, parent.FEE_CAP_FLOAT
    return limit_book(panel.date_index, panel.symbol_list, inputs["mats"], inputs["high"], inputs["low"], CONFIG.max_positions_int, START_STR,
                      entry_limit, exit_str, prep["margins"][fraction_float], prep["exit_limit"], slip, inputs["scale"], fee, minimum, cap)


# ---------------------------------------------------------------- calibration
def calibrate_cell(prep: dict, anchor_str: str, measure_str: str, target_float: float) -> dict:
    """The parameter whose limit-exit book fill rate on CALIBRATION_START_STR .. CALIBRATION_END_STR is closest to the
    target. k in [0, 1, expanding] for the k-measures (fill falls with k); p in [0.02, 0.98] for the quantile (fill rises
    with p)."""
    inputs, panel = prep["inputs"], prep["panel"]

    def fill_rate(param_float: float) -> float:
        return book_fill_rate(panel.date_index, panel.symbol_list, inputs["mats"], inputs["high"], inputs["low"],
                              limit_for(prep, anchor_str, measure_str, param_float), prep["margins"][0.1], prep["exit_limit"],
                              CONFIG.max_positions_int, CALIBRATION_START_STR, CALIBRATION_END_STR)[0]

    if measure_str == "dex63_quantile":
        return calibrate(fill_rate, target_float, 0.02, 0.98, decreasing_bool=False)
    return calibrate(fill_rate, target_float, 0.0, 1.0, decreasing_bool=True)


# ---------------------------------------------------------------- event-level tables
def event_labels(prep: dict) -> dict:
    """The parent's adverse-selection labels: eligible events in the window and the post-fill-day excess
    Close_(T+1) -> Close_(T+3) over the same-date regime-eligible members.

    *** CRITICAL*** forward labels, never features; a label needing a bar after the seal is NaN (event dropped)."""
    panel, inputs = prep["panel"], prep["inputs"]
    open_mat, close_mat = inputs["mats"]["open"], inputs["mats"]["close"]
    h_int = parent.HORIZON_INT
    in_window_vec = np.asarray((panel.date_index >= pd.Timestamp(START_STR)) & (panel.date_index <= pd.Timestamp(SEAL_END_STR)))
    shift = lambda m, n: np.vstack([m[n:], np.full((n, m.shape[1]), np.nan)])
    with np.errstate(divide="ignore", invalid="ignore"):
        close_h_mat = shift(close_mat, h_int)
        r_moo_mat = close_h_mat / shift(open_mat, 1) - 1.0
        r_close_mat = close_h_mat / close_mat - 1.0
        r_post_mat = close_h_mat / shift(close_mat, 1) - 1.0
    eligible_mat = inputs["base"] & np.isfinite(r_moo_mat) & np.isfinite(r_close_mat) & np.isfinite(r_post_mat) & in_window_vec[:, None]
    count_vec = np.maximum(eligible_mat.sum(axis=1), 1)
    baseline_post_vec = np.where(eligible_mat, r_post_mat, 0.0).sum(axis=1) / count_vec
    return {"event": inputs["event"] & eligible_mat, "excess_post": r_post_mat - baseline_post_vec[:, None],
            "event_in_window": inputs["event"] & in_window_vec[:, None]}


def event_fill_table(prep: dict, labels: dict, entry_limit: np.ndarray, baseline_fill_mat: np.ndarray | None) -> tuple[dict, np.ndarray]:
    """Event-level fill share (every DV2 event's order worked alone on T+1), post-fill-day continuation filled vs unfilled,
    and the overlap of the filled set with the baseline's at the same target."""
    inputs, panel = prep["inputs"], prep["panel"]
    code_mat, _ = event_fill_mats(inputs["mats"]["open"], inputs["low"], entry_limit, prep["margins"][0.1])
    window_event_mat = labels["event_in_window"]
    filled_all_mat = window_event_mat & (code_mat > 0)
    filled_mat = labels["event"] & (code_mat > 0)
    unfilled_mat = labels["event"] & (code_mat == 0)
    post_filled = parent.date_mean_stats(labels["excess_post"], filled_mat, panel.date_index)
    post_unfilled = parent.date_mean_stats(labels["excess_post"], unfilled_mat, panel.date_index)
    out_dict = {"event_fill_share_float": float(filled_all_mat.sum() / max(window_event_mat.sum(), 1)),
                "event_open_fill_share_of_fills_float": float((window_event_mat & (code_mat == 1)).sum() / max(filled_all_mat.sum(), 1)),
                "post_fill_day_filled": post_filled, "post_fill_day_unfilled": post_unfilled,
                "post_fill_day_gap_bp_float": post_filled["mean_bp_float"] - post_unfilled["mean_bp_float"]}
    if baseline_fill_mat is not None:
        union_int = int((filled_all_mat | baseline_fill_mat).sum())
        out_dict["jaccard_with_baseline_float"] = float((filled_all_mat & baseline_fill_mat).sum() / union_int) if union_int else float("nan")
    return out_dict, filled_all_mat


def correlation_table(prep: dict) -> dict:
    """Cross-sectional correlation of NATR14 with each other measure on signal days (events in the window): the mean over
    dates with >= 5 events of the date's Pearson and Spearman correlation, and the pooled Pearson."""
    panel = prep["panel"]
    event_mat = prep["inputs"]["event"] & np.asarray(panel.date_index >= pd.Timestamp(START_STR))[:, None]
    other_dict = {"std21": prep["measures"]["std21"], "dex21_mean (open)": prep["measures"]["dex21_mean"],
                  "dex21_mean (close analog)": prep["close_excursion_mean"],
                  "dex63_quantile p=0.40 (open)": prep["quantiles"]["open"].offset_mat(0.40),
                  "dex63_quantile p=0.40 (close)": prep["quantiles"]["close"].offset_mat(0.40)}
    natr_mat = prep["measures"]["natr14"]
    row_vec, column_vec = np.nonzero(event_mat)
    out_dict = {}
    for name_str, other_mat in other_dict.items():
        frame = pd.DataFrame({"row": row_vec, "natr": natr_mat[row_vec, column_vec], "other": other_mat[row_vec, column_vec]}).dropna()
        pearson_list, spearman_list = [], []
        for _, group in frame.groupby("row"):
            if len(group) >= CORRELATION_MIN_EVENTS_INT and group["natr"].std() > 0 and group["other"].std() > 0:
                pearson_list.append(group["natr"].corr(group["other"]))
                spearman_list.append(group["natr"].corr(group["other"], method="spearman"))
        out_dict[name_str] = {"mean_daily_pearson_float": float(np.nanmean(pearson_list)), "mean_daily_spearman_float": float(np.nanmean(spearman_list)),
                              "pooled_pearson_float": float(frame["natr"].corr(frame["other"])), "dates_int": len(pearson_list),
                              "events_int": len(frame)}
        log(f"  corr NATR14 vs {name_str:30s} daily Pearson {out_dict[name_str]['mean_daily_pearson_float']:.2f} "
            f"Spearman {out_dict[name_str]['mean_daily_spearman_float']:.2f} pooled {out_dict[name_str]['pooled_pearson_float']:.2f}")
    return out_dict


# ---------------------------------------------------------------- paired bootstrap
def paired_sharpe_difference(variant_ser: pd.Series, baseline_ser: pd.Series, path_count_int: int = BOOTSTRAP_PATH_INT,
                             block_float: float = BOOTSTRAP_BLOCK_FLOAT, seed_int: int = BOOTSTRAP_SEED_INT) -> dict:
    """Sharpe(variant) - Sharpe(baseline) on the same days, with a stationary-bootstrap 95% CI resampling the paired days
    together (mean block `block_float` sessions) and the bootstrap share of draws <= 0."""
    frame = pd.concat([variant_ser, baseline_ser], axis=1).dropna()
    value_mat = frame.to_numpy()
    index_mat = stationary_bootstrap_index_mat(len(frame), path_count_int, block_float, len(frame), seed_int)
    sharpe_diff_vec = np.zeros(path_count_int)
    for column_int, sign_float in ((0, 1.0), (1, -1.0)):
        path_mat = value_mat[:, column_int][index_mat]
        sharpe_diff_vec += sign_float * path_mat.mean(axis=1) / path_mat.std(axis=1, ddof=1) * np.sqrt(252.0)
    observed_float = sharpe_float(frame.iloc[:, 0]) - sharpe_float(frame.iloc[:, 1])
    return {"difference_float": observed_float, "ci_low_float": float(np.quantile(sharpe_diff_vec, 0.025)),
            "ci_high_float": float(np.quantile(sharpe_diff_vec, 0.975)), "p_not_better_float": float((sharpe_diff_vec <= 0).mean()),
            "daily_correlation_float": float(frame.corr().iloc[0, 1])}


# ---------------------------------------------------------------- one universe
def window(ser: pd.Series) -> pd.Series:
    return ser.loc[START_STR:SEAL_END_STR]


def run_cell(prep: dict, labels: dict, anchor_str: str, measure_str: str, target_float: float, param_float: float, exit_str: str,
             baseline_fill_mat: np.ndarray | None) -> tuple[dict, dict, np.ndarray]:
    entry_limit = limit_for(prep, anchor_str, measure_str, param_float)
    row_dict = {"anchor_str": anchor_str, "measure_str": measure_str, "fill_target_float": target_float, "exit_str": exit_str,
                "param_float": param_float}
    daily_dict = {}
    result_dict = {case_str: run_book(prep, entry_limit, exit_str, case_str) for case_str in ("gross", "ar", "pooled")}
    row_dict["orders"] = parent.order_stats(result_dict["pooled"], prep["panel"].date_index)
    for case_str, result in result_dict.items():
        row_dict[case_str] = parent.pod_metrics(result, prep["baseline"])
        daily_dict[case_str] = window(result.daily_ser)
    calibration_log_df = result_dict["pooled"].log_df
    calibration_log_df = calibration_log_df[(calibration_log_df["date"] >= pd.Timestamp(CALIBRATION_START_STR))
                                            & (calibration_log_df["date"] <= pd.Timestamp(CALIBRATION_END_STR))]
    order_int = int(calibration_log_df["kind_int"].isin([0, 1]).sum())
    row_dict["fill_rate_2004_2012_float"] = float((calibration_log_df["kind_int"] == 1).sum() / order_int) if order_int else float("nan")
    full_log_df = result_dict["pooled"].log_df
    late_log_df = full_log_df[full_log_df["date"] > pd.Timestamp(CALIBRATION_END_STR)]
    late_order_int = int(late_log_df["kind_int"].isin([0, 1]).sum())
    row_dict["fill_rate_2013_2022_float"] = float((late_log_df["kind_int"] == 1).sum() / late_order_int) if late_order_int else float("nan")
    stress_dict = {}
    for fraction_float in STRESS_FRACTION_TUPLE[1:]:
        stress_row = {}
        for case_str in ("gross", "ar", "pooled"):
            result = run_book(prep, entry_limit, exit_str, case_str, fraction_float)
            stress_row[f"sharpe_{case_str}"] = sharpe_float(window(result.daily_ser))
            if case_str == "pooled":
                stress_row["fill_rate_float"] = parent.order_stats(result, prep["panel"].date_index)["fill_rate_float"]
        stress_dict[f"f{fraction_float:g}"] = stress_row
    row_dict["fill_stress"] = stress_dict
    filled_mat = None
    if exit_str == "limit":
        row_dict["events"], filled_mat = event_fill_table(prep, labels, entry_limit, baseline_fill_mat)
    events = row_dict.get("events", {})
    log(f"  {cell_label(anchor_str, measure_str, target_float, exit_str):34s} param {param_float:7.4f} fill {row_dict['orders']['fill_rate_float']:.2f} "
        f"(cal {row_dict['fill_rate_2004_2012_float']:.2f}) tr/yr {row_dict['orders']['trades_per_year_float']:4.0f} "
        f"gross {row_dict['gross']['sharpe_float']:5.2f} AR {row_dict['ar']['sharpe_float']:5.2f} pooled {row_dict['pooled']['sharpe_float']:5.2f} "
        f"cost {row_dict['ar'].get('cost_per_round_trip_bp_float', np.nan):4.0f}/{row_dict['pooled'].get('cost_per_round_trip_bp_float', np.nan):3.0f} bp "
        f"f1 {stress_dict['f1']['sharpe_ar']:5.2f}/{stress_dict['f1']['sharpe_pooled']:5.2f} "
        f"post gap {events.get('post_fill_day_gap_bp_float', np.nan):+5.1f} bp")
    return row_dict, daily_dict, filled_mat


def run_universe(superset, name_str: str) -> dict:
    prep = prepare(superset, name_str)
    panel = prep["panel"]
    log(f"{name_str}: {len(panel.symbol_list)} symbols, events {int(prep['inputs']['event'][np.asarray(panel.date_index >= START_STR)].sum())}")
    out_dict = {"universe_str": name_str, "start_str": START_STR, "end_str": SEAL_END_STR, "snapshot_id_str": panel.snapshot_id_str,
                "calibration_window_str": f"{CALIBRATION_START_STR}..{CALIBRATION_END_STR}", "correlation": correlation_table(prep)}
    labels = event_labels(prep)
    # reference: today's market-on-open entry and exit (the parent's baseline; not a trial of this study)
    reference_dict = {}
    for case_str in ("gross", "ar", "pooled"):
        reference_dict[case_str] = parent.pod_metrics(run_book(prep, None, "moo", case_str), prep["baseline"])
    out_dict["reference_moo_moo"] = reference_dict
    log(f"  reference moo/moo: AR {reference_dict['ar']['sharpe_float']:.2f} pooled {reference_dict['pooled']['sharpe_float']:.2f}")
    # uncalibrated p = target for the quantile measure (diagnostic: does p set the per-order fill probability?)
    diagnostic_dict = {}
    for anchor_str in ANCHOR_TUPLE:
        for probability_float in DIAGNOSTIC_PROBABILITY_TUPLE:
            code_mat, _ = event_fill_mats(prep["inputs"]["mats"]["open"], prep["inputs"]["low"],
                                          limit_for(prep, anchor_str, "dex63_quantile", probability_float), prep["margins"][0.1])
            event_mat = labels["event_in_window"]
            diagnostic_dict[f"{anchor_str}|p{probability_float:.2f}"] = float((event_mat & (code_mat > 0)).sum() / max(event_mat.sum(), 1))
    out_dict["uncalibrated_quantile_event_fill_share"] = diagnostic_dict
    log(f"  uncalibrated quantile event fill shares: {diagnostic_dict}")
    cells, daily_dict, calibration_dict = {}, {}, {}
    for target_float in FILL_TARGET_TUPLE:
        baseline_fill_mat = None
        ordered_list = [BASELINE_TUPLE] + [(a, m) for a in ANCHOR_TUPLE for m in MEASURE_TUPLE if (a, m) != BASELINE_TUPLE]
        for anchor_str, measure_str in ordered_list:
            label_str = cell_label(anchor_str, measure_str, target_float)
            calibration = calibrate_cell(prep, anchor_str, measure_str, target_float)
            calibration_dict[label_str] = calibration
            row_dict, cell_daily_dict, filled_mat = run_cell(prep, labels, anchor_str, measure_str, target_float, calibration["param_float"],
                                                             "limit", baseline_fill_mat)
            row_dict["calibration"] = calibration
            if (anchor_str, measure_str) == BASELINE_TUPLE:
                baseline_fill_mat = filled_mat
            cells[label_str] = row_dict
            for case_str, ser in cell_daily_dict.items():
                daily_dict[f"{label_str}|{case_str}"] = ser
            gc.collect()
    # paired bootstrap against the baseline at the same target, per cost model
    for label_str, row_dict in cells.items():
        anchor_str, measure_str, target_float = row_dict["anchor_str"], row_dict["measure_str"], row_dict["fill_target_float"]
        if (anchor_str, measure_str) == BASELINE_TUPLE:
            continue
        base_label_str = cell_label(*BASELINE_TUPLE, target_float)
        row_dict["vs_baseline"] = {case_str: paired_sharpe_difference(daily_dict[f"{label_str}|{case_str}"], daily_dict[f"{base_label_str}|{case_str}"])
                                   for case_str in ("ar", "pooled")}
        comparison = row_dict["vs_baseline"]
        log(f"  vs baseline {label_str:34s} AR {comparison['ar']['difference_float']:+.2f} [{comparison['ar']['ci_low_float']:+.2f}, "
            f"{comparison['ar']['ci_high_float']:+.2f}] pooled {comparison['pooled']['difference_float']:+.2f} "
            f"[{comparison['pooled']['ci_low_float']:+.2f}, {comparison['pooled']['ci_high_float']:+.2f}]")
    out_dict["cells"] = cells
    parent.write_json(OUT_PATH / "universes" / f"{parent.slug(name_str)}.json", out_dict)
    pd.DataFrame(daily_dict).assign(baseline=window(prep["baseline"])).to_csv(OUT_PATH / "universes" / f"{parent.slug(name_str)}_daily.csv")
    return out_dict


# ---------------------------------------------------------------- market-on-open exit for the best two entries
def best_two_entries() -> list[tuple[str, str, float]]:
    """The two S&P 500 limit-exit entries with the highest mean of AR and pooled net Sharpe (any anchor x measure x target)."""
    main_dict = json.loads((OUT_PATH / "universes" / f"{parent.slug(MAIN_UNIVERSE_STR)}.json").read_text(encoding="utf-8"))
    scored_list = sorted(main_dict["cells"].values(), key=lambda r: -(r["ar"]["sharpe_float"] + r["pooled"]["sharpe_float"]) / 2.0)
    return [(r["anchor_str"], r["measure_str"], r["fill_target_float"]) for r in scored_list[:2]]


def run_moo_exits(superset, name_str: str, entry_list: list[tuple[str, str, float]]) -> dict:
    prep = prepare(superset, name_str)
    universe_dict = json.loads((OUT_PATH / "universes" / f"{parent.slug(name_str)}.json").read_text(encoding="utf-8"))
    labels = event_labels(prep)
    wanted_list = list(dict.fromkeys(entry_list + [(BASELINE_TUPLE[0], BASELINE_TUPLE[1], t) for _, _, t in entry_list]))
    out_dict, daily_dict = {"universe_str": name_str, "best_two_entries": [cell_label(*e) for e in entry_list]}, {}
    for anchor_str, measure_str, target_float in wanted_list:
        param_float = universe_dict["cells"][cell_label(anchor_str, measure_str, target_float)]["param_float"]
        row_dict, cell_daily_dict, _ = run_cell(prep, labels, anchor_str, measure_str, target_float, param_float, "moo", None)
        out_dict[cell_label(anchor_str, measure_str, target_float, "moo")] = row_dict
        for case_str, ser in cell_daily_dict.items():
            daily_dict[(anchor_str, measure_str, target_float, case_str)] = ser
    for anchor_str, measure_str, target_float in entry_list:
        if (anchor_str, measure_str) == BASELINE_TUPLE:
            continue
        comparison = {case_str: paired_sharpe_difference(daily_dict[(anchor_str, measure_str, target_float, case_str)],
                                                         daily_dict[(*BASELINE_TUPLE, target_float, case_str)]) for case_str in ("ar", "pooled")}
        out_dict[cell_label(anchor_str, measure_str, target_float, "moo")]["vs_baseline"] = comparison
        log(f"  moo exit vs baseline {cell_label(anchor_str, measure_str, target_float, 'moo')}: AR {comparison['ar']['difference_float']:+.2f} "
            f"[{comparison['ar']['ci_low_float']:+.2f}, {comparison['ar']['ci_high_float']:+.2f}] pooled {comparison['pooled']['difference_float']:+.2f} "
            f"[{comparison['pooled']['ci_low_float']:+.2f}, {comparison['pooled']['ci_high_float']:+.2f}]")
    parent.write_json(OUT_PATH / "moo_exit" / f"{parent.slug(name_str)}.json", out_dict)
    return out_dict


# ---------------------------------------------------------------- parameter-jitter band (diagnostic, not new trials)
BAND_FACTOR_TUPLE = (0.9, 0.95, 1.0, 1.05, 1.1)


def run_band(superset, name_str: str) -> dict:
    """Each registered limit-exit cell re-run with its calibrated parameter x 0.9 .. 1.1 (fill rate moves by a few points):
    the slot book is path dependent, so neighbouring parameters at nearly the same fill rate differ by noise. The band
    mean Sharpe, and the band-mean difference against the baseline's band, separate a real ordering from that jitter."""
    prep = prepare(superset, name_str)
    universe_dict = json.loads((OUT_PATH / "universes" / f"{parent.slug(name_str)}.json").read_text(encoding="utf-8"))
    out_dict = {"universe_str": name_str, "factor_list": list(BAND_FACTOR_TUPLE), "cells": {}}
    for label_str, row in universe_dict["cells"].items():
        band_list = []
        for factor_float in BAND_FACTOR_TUPLE:
            param_float = row["param_float"] * factor_float
            entry_limit = limit_for(prep, row["anchor_str"], row["measure_str"], param_float)
            result_dict = {case_str: run_book(prep, entry_limit, "limit", case_str) for case_str in ("ar", "pooled")}
            band_list.append({"factor_float": factor_float, "param_float": param_float,
                              "fill_rate_float": parent.order_stats(result_dict["pooled"], prep["panel"].date_index)["fill_rate_float"],
                              "sharpe_ar_float": sharpe_float(window(result_dict["ar"].daily_ser)),
                              "sharpe_pooled_float": sharpe_float(window(result_dict["pooled"].daily_ser))})
        band_df = pd.DataFrame(band_list)
        out_dict["cells"][label_str] = {"band": band_list, "mean_sharpe_ar_float": float(band_df["sharpe_ar_float"].mean()),
                                        "mean_sharpe_pooled_float": float(band_df["sharpe_pooled_float"].mean()),
                                        "sd_sharpe_pooled_float": float(band_df["sharpe_pooled_float"].std(ddof=1)),
                                        "fill_min_float": float(band_df["fill_rate_float"].min()), "fill_max_float": float(band_df["fill_rate_float"].max())}
        cell = out_dict["cells"][label_str]
        log(f"  band {label_str:34s} fill {cell['fill_min_float']:.2f}-{cell['fill_max_float']:.2f} mean AR {cell['mean_sharpe_ar_float']:.2f} "
            f"pooled {cell['mean_sharpe_pooled_float']:.2f} (sd {cell['sd_sharpe_pooled_float']:.3f})")
    for label_str, cell in out_dict["cells"].items():
        row = universe_dict["cells"][label_str]
        base = out_dict["cells"][cell_label(*BASELINE_TUPLE, row["fill_target_float"])]
        cell["band_difference_ar_float"] = cell["mean_sharpe_ar_float"] - base["mean_sharpe_ar_float"]
        cell["band_difference_pooled_float"] = cell["mean_sharpe_pooled_float"] - base["mean_sharpe_pooled_float"]
    parent.write_json(OUT_PATH / "band" / f"{parent.slug(name_str)}.json", out_dict)
    return out_dict


# ---------------------------------------------------------------- MCPT over the registered S&P 500 entry grid
def grid_entries() -> list[tuple[str, str, float]]:
    return [(a, m, t) for a in ANCHOR_TUPLE for m in MEASURE_TUPLE for t in FILL_TARGET_TUPLE]


def mcpt_score(prep: dict, param_dict: dict, cost_str: str) -> tuple[float, int, list]:
    """Plateau choice over the anchor x measure x target grid (limit exit) of the net active Sharpe (daily net return minus
    the equal-weight members) under one half-spread model, with the calibrated parameters held fixed."""
    from alpha.stats.selection import plateau_choice

    panel = prep["panel"]
    baseline_vec = prep["baseline"].to_numpy()
    keep_vec = np.asarray((panel.date_index >= pd.Timestamp(START_STR)) & (panel.date_index <= pd.Timestamp(SEAL_END_STR)))
    sharpe_list = []
    for anchor_str, measure_str, target_float in grid_entries():
        result = run_book(prep, limit_for(prep, anchor_str, measure_str, param_dict[cell_label(anchor_str, measure_str, target_float)]),
                          "limit", cost_str)
        sharpe_list.append(sharpe_float(pd.Series((result.daily_ser.to_numpy() - baseline_vec)[keep_vec])))
    choice = plateau_choice(np.nan_to_num(np.array(sharpe_list), nan=-9.0), (len(ANCHOR_TUPLE), len(MEASURE_TUPLE), len(FILL_TARGET_TUPLE)))
    return choice.own_sharpe_float, choice.flat_index_int, sharpe_list


def prepare_permuted(permuted, pooled_mat: np.ndarray) -> dict:
    """prepare() on a null draw: the pooled spread stays on its real dates; Abdi-Ranaldo and every measure are recomputed."""
    spread_dict = parent.null_spread_dict(permuted, pooled_mat)
    inputs = parent.signal_inputs(permuted)
    mats = inputs["mats"]
    open_excursion = open_excursion_mat(mats["open"], inputs["low"])
    close_excursion = close_excursion_mat(mats["close"], inputs["low"])
    return {"panel": permuted, "spread": spread_dict, "inputs": inputs,
            "measures": {"natr14": natr_fraction_mat(inputs["natr"]), "std21": close_return_std_mat(mats["close"]),
                         "dex21_mean": rolling_mean_mat(open_excursion)},
            "quantiles": {"open": ExcursionQuantile(open_excursion, inputs["event"]), "close": ExcursionQuantile(close_excursion, inputs["event"])},
            "margins": {0.1: trade_through_margin_mat(inputs["unadjusted"], [spread_dict["ar"], spread_dict["pooled"]])},
            "exit_limit": exit_limit_mat(mats["close"], inputs["unadjusted"]),
            "slip": {k: fill_slippage_mat(spread_dict[k], parent.FLOOR_SLIP_FLOAT) for k in ("ar", "pooled")},
            "baseline": pd.Series(parent.dv2.member_baseline_vec(mats["close"], mats["member"]), index=permuted.date_index)}


def _mcpt_chunk(args) -> np.ndarray:
    seed_int, count_int, cost_str, snapshot_id_str, param_dict = args
    from alpha.scout.null import permuted_panel

    superset = load_superset_panel(parent.SUPERSET_NAME_STR, snapshot_id_str, index_name_list=parent.LOAD_INDEX_LIST)
    panel = parent.build_panel(superset, MAIN_UNIVERSE_STR)
    pooled_mat = parent.spread_dict_for(superset, panel)["pooled"]
    rng_obj, cache_dict, out_list = np.random.default_rng(seed_int), {}, []
    for _ in range(count_int):
        permuted = permuted_panel(panel, rng_obj, cache_dict)
        out_list.append(mcpt_score(prepare_permuted(permuted, pooled_mat), param_dict, cost_str)[0])
        del permuted
        gc.collect()
    return np.array(out_list)


def run_mcpt(superset, worker_count_int: int, permutation_count_int: int, cost_str: str) -> dict:
    from multiprocessing import Pool

    main_dict = json.loads((OUT_PATH / "universes" / f"{parent.slug(MAIN_UNIVERSE_STR)}.json").read_text(encoding="utf-8"))
    param_dict = {label_str: row["param_float"] for label_str, row in main_dict["cells"].items()}
    prep = prepare(superset, MAIN_UNIVERSE_STR)
    observed_float, chosen_int, observed_list = mcpt_score(prep, param_dict, cost_str)
    del prep
    gc.collect()
    grid_list = [cell_label(*e) for e in grid_entries()]
    log(f"MCPT {MAIN_UNIVERSE_STR} ({cost_str}): plateau cell {grid_list[chosen_int]}, active Sharpe {observed_float:.3f}; "
        f"{permutation_count_int} permutations on {worker_count_int} workers")
    chunk_int = permutation_count_int // worker_count_int + 1
    with Pool(worker_count_int) as pool_obj:
        null_vec = np.concatenate(pool_obj.map(_mcpt_chunk, [(10_300 + i, chunk_int, cost_str, superset.snapshot_id_str, param_dict)
                                                             for i in range(worker_count_int)]))[:permutation_count_int]
    p_float = float((1 + np.sum(null_vec >= observed_float)) / (1 + null_vec.size))
    out_dict = {"universe_str": MAIN_UNIVERSE_STR, "cost_str": cost_str, "observed_active_sharpe_float": observed_float,
                "chosen_config_str": grid_list[chosen_int], "observed_grid_active_sharpe_dict": dict(zip(grid_list, observed_list)),
                "p_value_float": p_float, "permutation_count_int": int(null_vec.size), "null_95_float": float(np.quantile(null_vec, 0.95)),
                "null_mean_float": float(null_vec.mean()), "configurations_in_search_int": len(grid_list), "worker_count_int": worker_count_int,
                "score_str": "plateau choice over anchor x measure x target (limit exit) of the net active Sharpe over EW members; "
                             "calibrated parameters fixed on the null"}
    parent.write_json(OUT_PATH / "mcpt" / f"{parent.slug(MAIN_UNIVERSE_STR)}_{cost_str}.json", out_dict)
    np.save(OUT_PATH / "mcpt" / f"{parent.slug(MAIN_UNIVERSE_STR)}_{cost_str}_null.npy", null_vec)
    log(f"MCPT {MAIN_UNIVERSE_STR} ({cost_str}): p {p_float:.4f} (null 95% {out_dict['null_95_float']:.3f}, mean {out_dict['null_mean_float']:.3f})")
    return out_dict


def main() -> None:
    command_str = sys.argv[1]
    superset = load_superset_panel(parent.SUPERSET_NAME_STR, index_name_list=parent.LOAD_INDEX_LIST)
    if command_str == "universe":
        for name_str in sys.argv[2:] or list(UNIVERSE_TUPLE):
            run_universe(superset, name_str)
            gc.collect()
    elif command_str == "moo":
        entry_list = best_two_entries()
        log(f"best two S&P 500 entries: {entry_list}")
        for name_str in sys.argv[2:] or list(UNIVERSE_TUPLE):
            run_moo_exits(superset, name_str, entry_list)
            gc.collect()
    elif command_str == "band":
        for name_str in sys.argv[2:] or list(UNIVERSE_TUPLE):
            run_band(superset, name_str)
            gc.collect()
    elif command_str == "mcpt":
        worker_count_int = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        permutation_count_int = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
        cost_str = sys.argv[4] if len(sys.argv) > 4 else "pooled"
        run_mcpt(superset, worker_count_int, permutation_count_int, cost_str)
    else:
        raise SystemExit(f"unknown command {command_str!r}")


if __name__ == "__main__":
    main()
