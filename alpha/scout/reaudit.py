"""Re-audition runner (design section 10): one pod through S3-S6 and a research card, the P5 procedure made reusable.

A `PodPlan` names the pod, its family factory, its MCPT kind and options, its adoption date and prior trials, and
the LIVE pod whose slot it would take in the reference book (the live 60/40 book of TAA 3x and NDX VXN). In sample =
up to the vault seal (2022-12-30). MCPT components (A8, A9):
    "taa"  plain date shuffle of [TR returns of the traded ETFs, SPY return, VIX, DTB3]; score SD
    "ndx"  stock selection: per-asset null on the Nasdaq-100 panel, active return over overlay x EW members;
           timing overlay: plain date shuffle of [EW members return, SPY return, VXN]; score SD
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from multiprocessing import Pool

import numpy as np
import pandas as pd

from alpha.scout import searches
from alpha.scout.card import grade_str, render_card
from alpha.scout.engines.weights import CostModel
from alpha.scout.family import FamilyRunner
from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH
from alpha.scout.metrics import performance_dict, sharpe_float
from alpha.scout.stations.s4_strategy import SEAL_END_STR, run_s4
from alpha.scout.stations.s5_overfit import McptComponent, run_s5
from alpha.scout.stations.s6_book import (
    capacity,
    diversification,
    finish_s6,
    spanning_table,
    tbill_slot_test,
)
from alpha.stats.mcpt import mcpt
from alpha.stats.psr_dsr import minimum_track_record_length, sharpe_moments
from alpha.stats.selection import plateau_choice

OUTPUT_DIR_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "reaudition"
CARD_DIR_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "cards"
PERMUTATION_COUNT_INT, WORKER_COUNT_INT = 1000, 14
BOOK_WEIGHT_DICT = {"TAA 3x": 0.6, "NDX VXN": 0.4}  # the live account, about $18K and $12K
REGIME_SMA_TUPLE, VXN_REFERENCE_TUPLE = (100, 150, 200, 250), (18.0, 22.0, 26.0)


@dataclass
class PodPlan:
    name_str: str
    family_fn: Callable  # inputs -> FamilyRunner
    inputs_fn: Callable  # () -> spec inputs
    mcpt_kind_str: str  # "taa" or "ndx"
    adoption_date_str: str
    prior_trial_count_int: int
    slot_str: str  # the LIVE pod whose slot it takes in the reference book
    s3_fn: Callable | None = None  # () -> (class_str, note_str, result_dict)
    option_dict: dict = field(default_factory=dict)  # MCPT options: ndx_config (NdxConfig) / taa options


# ---------------------------------------------------------------- SD score (A9)
def vol_targeted_vec(daily_vec: np.ndarray) -> np.ndarray:
    realized_vec = pd.Series(daily_vec).rolling(20).std().shift(1).to_numpy() * np.sqrt(252.0)
    scale_vec = np.where(np.isfinite(realized_vec) & (realized_vec > 0), np.minimum(1.0, 0.10 / realized_vec), 0.0)
    return scale_vec * daily_vec


def sd_score(config_daily_list: list[np.ndarray], baseline_vec: np.ndarray, grid_shape_tuple: tuple, warm_int: int) -> float:
    targeted_vec = vol_targeted_vec(baseline_vec)
    sharpe_vec = np.array([sharpe_float(pd.Series((v - targeted_vec)[warm_int:])) for v in config_daily_list])
    return plateau_choice(np.nan_to_num(sharpe_vec, nan=-9.0), grid_shape_tuple).own_sharpe_float


def _p_value(observed_float: float, null_vec: np.ndarray) -> float:
    return float((1 + np.sum(null_vec >= observed_float)) / (1 + null_vec.size))


# ---------------------------------------------------------------- TAA MCPT
_STATE: dict = {}


def _taa_search(matrix: np.ndarray) -> float:
    state = _STATE
    daily_list = searches.taa_config_daily_list(matrix, state["date_index"], state["config_list"], **state["taa_option_dict"])
    return sd_score(daily_list, matrix[:, : state["asset_count_int"]].mean(axis=1), state["grid_shape_tuple"], 260)


def _taa_chunk(args) -> np.ndarray:
    matrix, seed_int, count_int, state = args
    _STATE.update(state)
    return mcpt(_taa_search, matrix, count_int, seed_int).null_score_vec


def taa_mcpt(plan: PodPlan, family: FamilyRunner, inputs) -> list[McptComponent]:
    from data.norgate_loader import load_price_timeseries

    taa_option_dict = dict(plan.option_dict.get("taa", {}))
    asset_tuple = taa_option_dict.get("asset_tuple", searches.TAA_ASSET_TUPLE)
    extra_close_dict = {s: load_price_timeseries(s, adjustment_str="TOTALRETURN", start_date_str="2006-01-01")["Close"]
                        for s in asset_tuple if s not in inputs.total_return_close_df.columns}
    close_df = inputs.total_return_close_df.assign(**extra_close_dict)
    # Start once every traded ETF has a price (zero-filled pre-listing days would be fake flat days in the null).
    first_ts = max(close_df[s].first_valid_index() for s in asset_tuple) + pd.Timedelta(days=1)
    date_index = inputs.open_df.index[(inputs.open_df.index >= first_ts) & (inputs.open_df.index <= SEAL_END_STR)]
    matrix = searches.taa_matrix(date_index, close_df, inputs.spy_close_ser, inputs.vix_close_ser, inputs.dtb3_ser, asset_tuple=asset_tuple)
    state = {"date_index": date_index, "config_list": family.config_list(), "grid_shape_tuple": family.grid_shape_tuple,
             "taa_option_dict": taa_option_dict, "asset_count_int": len(asset_tuple)}
    _STATE.update(state)
    observed_float = _taa_search(matrix)
    chunk_int = PERMUTATION_COUNT_INT // WORKER_COUNT_INT + 1
    with Pool(WORKER_COUNT_INT) as pool_obj:
        null_vec = np.concatenate(pool_obj.map(_taa_chunk, [(matrix, 7_000 + i, chunk_int, state) for i in range(WORKER_COUNT_INT)]))[:PERMUTATION_COUNT_INT]
    return [McptComponent("whole strategy", "date shuffle", f"SD: active Sharpe over vol-targeted EW of {len(asset_tuple)} ETFs", observed_float,
                          null_vec, _p_value(observed_float, null_vec), note_str="ETF timing family (A9 calibration: 2.0-7.0% false passes)")]


# ---------------------------------------------------------------- NDX MCPT
def overlay_scale_ser(spy_close_ser: pd.Series, vxn_close_ser: pd.Series, date_index, regime_sma_int: int, reference_float: float | None) -> pd.Series:
    """SPY regime (on -> 1) times the VXN scale clip(reference / VXN, 0.25, 1); reference None = no VXN scaling."""
    spy_ser = spy_close_ser.reindex(date_index).ffill()
    on_vec = (spy_ser > spy_ser.rolling(regime_sma_int, min_periods=regime_sma_int).mean()).to_numpy()
    if reference_float is None:
        return pd.Series(np.where(on_vec, 1.0, 0.0), index=date_index)
    vxn_ser = vxn_close_ser.reindex(vxn_close_ser.index.union(date_index)).ffill().reindex(date_index)
    return pd.Series(np.where(on_vec, np.clip(reference_float / vxn_ser, 0.25, 1.0), 0.0), index=date_index)


def _ndx_selection_score(panel) -> float:
    state = _STATE
    f = lambda n: panel.field(n).to_numpy(dtype=float)
    daily_list, baseline_vec = searches.ndx_selection_daily(
        f("Open"), f("High"), f("Low"), f("Close"), f("Unadjusted Close"), (panel.member_df == 1).to_numpy(),
        panel.date_index, state["overlay_ser"], state["config_list"], state["atr_unit_str"],
    )
    sharpe_vec = np.array([sharpe_float(pd.Series((v - baseline_vec)[520:])) for v in daily_list])
    return plateau_choice(np.nan_to_num(sharpe_vec, nan=-9.0), state["grid_shape_tuple"]).own_sharpe_float


def _ndx_chunk(args) -> np.ndarray:
    seed_int, count_int, state = args
    from alpha.scout.null import permuted_panel
    from alpha.scout.panel import load_panel

    _STATE.update(state)
    panel = load_panel("Nasdaq 100")
    rng_obj, cache_dict = np.random.default_rng(seed_int), {}
    return np.array([_ndx_selection_score(permuted_panel(panel, rng_obj, cache_dict)) for _ in range(count_int)])


def _overlay_search(matrix: np.ndarray) -> float:
    state = _STATE
    date_index = state["overlay_date_index"]
    index_vec, spy_vec, vxn_vec = matrix[:, 0], matrix[:, 1], matrix[:, 2]
    spy_price_ser = pd.Series(np.cumprod(1.0 + spy_vec), index=date_index)
    vxn_ser = pd.Series(vxn_vec, index=date_index)
    position_ser = pd.Series(np.arange(len(date_index)), index=date_index)
    decision_row_vec = position_ser.groupby(date_index.to_period("M")).max().to_numpy()[:-1]
    reference_tuple = state["reference_tuple"]
    daily_list = []
    for sma_int in REGIME_SMA_TUPLE:
        for reference_float in reference_tuple:
            scale_vec = overlay_scale_ser(spy_price_ser, vxn_ser, date_index, sma_int, reference_float).to_numpy()[decision_row_vec]
            daily_list.append(searches._hold_daily(scale_vec[:, None], decision_row_vec, index_vec[:, None]))
    return sd_score(daily_list, index_vec, (len(REGIME_SMA_TUPLE), len(reference_tuple)), 260)


def _overlay_chunk(args) -> np.ndarray:
    matrix, seed_int, count_int, state = args
    _STATE.update(state)
    return mcpt(_overlay_search, matrix, count_int, seed_int).null_score_vec


def ndx_mcpt(plan: PodPlan, family: FamilyRunner, inputs) -> list[McptComponent]:
    from alpha.scout.panel import load_panel

    config = plan.option_dict["ndx_config"]
    reference_live = config.vxn_reference_float if config.vxn_scaled_bool else None
    panel = load_panel("Nasdaq 100")
    state = {
        "overlay_ser": overlay_scale_ser(inputs.close_df["SPY"], inputs.vxn_close_ser, panel.date_index, config.regime_sma_int, reference_live),
        "config_list": family.config_list(), "grid_shape_tuple": family.grid_shape_tuple, "atr_unit_str": config.atr_unit_str,
        "reference_tuple": VXN_REFERENCE_TUPLE if config.vxn_scaled_bool else (None,),
    }
    _STATE.update(state)
    observed_float = _ndx_selection_score(panel)
    chunk_int = PERMUTATION_COUNT_INT // WORKER_COUNT_INT + 1
    with Pool(WORKER_COUNT_INT) as pool_obj:
        null_vec = np.concatenate(pool_obj.map(_ndx_chunk, [(8_000 + i, chunk_int, state) for i in range(WORKER_COUNT_INT)]))[:PERMUTATION_COUNT_INT]
    selection = McptComponent("stock selection", "per-asset", "active Sharpe over overlay x EW members", observed_float, null_vec,
                              _p_value(observed_float, null_vec), ranking_family_bool=True,
                              note_str="cross-sectional momentum (A8 calibration: 8.0% false passes; 0.025 < p <= 0.05 is marginal)")

    close_df = panel.field("Close")
    member_return_ser = (close_df / close_df.shift(1) - 1.0).where((panel.member_df == 1).shift(1, fill_value=False)).mean(axis=1).fillna(0.0)
    date_index = panel.date_index[panel.date_index >= "2000-01-03"]
    spy_return_ser = inputs.close_df["SPY"].reindex(date_index).ffill().pct_change().fillna(0.0)
    vxn_ser = inputs.vxn_close_ser.reindex(inputs.vxn_close_ser.index.union(date_index)).ffill().reindex(date_index).bfill()
    matrix = np.column_stack([member_return_ser.reindex(date_index).to_numpy(), spy_return_ser.to_numpy(), vxn_ser.to_numpy()])
    state["overlay_date_index"] = date_index
    _STATE.update(state)
    observed_float = _overlay_search(matrix)
    with Pool(WORKER_COUNT_INT) as pool_obj:
        null_vec = np.concatenate(pool_obj.map(_overlay_chunk, [(matrix, 9_000 + i, chunk_int, state) for i in range(WORKER_COUNT_INT)]))[:PERMUTATION_COUNT_INT]
    name_str = "timing overlay (SPY regime x VXN scale)" if config.vxn_scaled_bool else "timing overlay (SPY regime)"
    overlay = McptComponent(name_str, "date shuffle", "SD: active Sharpe over vol-targeted EW members", observed_float, null_vec,
                            _p_value(observed_float, null_vec), note_str="timing family (A9 calibration)")
    return [selection, overlay]


# ---------------------------------------------------------------- S6 inputs
def factor_daily_df(date_index: pd.DatetimeIndex, tbill_ser: pd.Series) -> pd.DataFrame:
    from data.norgate_loader import load_price_timeseries

    def tr(symbol_str):
        return load_price_timeseries(symbol_str, adjustment_str="TOTALRETURN", start_date_str="1998-01-01")["Close"]

    frame = pd.DataFrame({s: tr(s) for s in ("SPY", "QQQ", "IEF", "GLD")}).reindex(date_index).ffill().pct_change()
    trend_close_df = pd.DataFrame({s: tr(s) for s in ("SPY", "EFA", "EEM", "IEF", "TLT", "GLD", "DBC", "UUP")})
    month_close_df = trend_close_df.resample("ME").last()
    # 12-1 time-series momentum, signal labelled at month-end m (closes m-1 and m-12), applied from the next session.
    signal_df = np.sign(month_close_df.shift(1) / month_close_df.shift(12) - 1.0)
    daily_signal_df = signal_df.reindex(trend_close_df.index.union(date_index)).ffill().reindex(date_index).shift(1)
    daily_excess_df = trend_close_df.reindex(date_index).ffill().pct_change().sub(tbill_ser.reindex(date_index), axis=0)
    frame["TREND"] = (daily_signal_df * daily_excess_df).mean(axis=1) + tbill_ser.reindex(date_index)
    return frame


def dollar_volume_df(symbol_list: list[str]) -> pd.DataFrame:
    from data.norgate_loader import load_price_timeseries

    column_dict = {}
    for symbol_str in symbol_list:
        try:
            frame = load_price_timeseries(symbol_str, adjustment_str="CAPITALSPECIAL", start_date_str="1999-01-01")
        except Exception:  # noqa: BLE001, S112 - a delisted symbol without data has no capacity estimate
            continue
        column_dict[symbol_str] = frame["Turnover"] if "Turnover" in frame else frame["Close"] * frame["Volume"]
    return pd.DataFrame(column_dict)


# ---------------------------------------------------------------- one pod
def reaudit(plan: PodPlan, live_net_dict: dict[str, pd.Series], factor_df: pd.DataFrame, tbill_ser: pd.Series) -> dict:
    """Run S3-S6 for one pod, save its bundle and card, return the bundle."""
    inputs = plan.inputs_fn()
    family = plan.family_fn(inputs)
    s4 = run_s4(family, tbill_ser)
    mcpt_list = taa_mcpt(plan, family, inputs) if plan.mcpt_kind_str == "taa" else ndx_mcpt(plan, family, inputs)
    s5 = run_s5(s4.grid_df.loc[:SEAL_END_STR], family.grid_shape_tuple, s4.chosen_label_str, s4.live_label_str, mcpt_list, plan.prior_trial_count_int)

    net_ser = s4.grid_df[s4.live_label_str].loc[:SEAL_END_STR]
    gross_cost = CostModel(slippage_float=0.0, fee_per_share_float=0.0, min_fee_float=0.0)
    gross_ser = family.run_config(family.live_config_dict, gross_cost).daily_return_ser.loc[:SEAL_END_STR]
    other_str = next(p for p in BOOK_WEIGHT_DICT if p != plan.slot_str)
    spanning_list = spanning_table(net_ser, gross_ser, factor_df.assign(OTHER_POD=live_net_dict[other_str]), tbill_ser, {
        "QQQ": ["QQQ"],
        "ETF mix (SPY QQQ IEF GLD)": ["SPY", "QQQ", "IEF", "GLD"],
        "ETF mix + trend": ["SPY", "QQQ", "IEF", "GLD", "TREND"],
        f"ETF mix + trend + {other_str}": ["SPY", "QQQ", "IEF", "GLD", "TREND", "OTHER_POD"],
    })
    book_weight_dict = {(plan.name_str if p == plan.slot_str else p): w for p, w in BOOK_WEIGHT_DICT.items()}
    slot_dict = tbill_slot_test(net_ser, book_weight_dict, {other_str: live_net_dict[other_str].loc[:SEAL_END_STR]}, plan.name_str, tbill_ser)
    book_df = pd.DataFrame(live_net_dict).loc[:SEAL_END_STR].dropna()
    book_ser = sum(BOOK_WEIGHT_DICT[p] * book_df[p] for p in BOOK_WEIGHT_DICT)
    reference_dict = {**{p: s for p, s in live_net_dict.items() if p != plan.name_str}, "SPY": factor_df["SPY"], "QQQ": factor_df["QQQ"]}
    diversification_dict = diversification(net_ser, reference_dict, book_ser)
    live_result = family.run_config(family.live_config_dict)
    capacity_dict = capacity(live_result.trade_df, live_result.total_value_ser, dollar_volume_df(sorted(live_result.trade_df["asset"].unique())))
    s6 = finish_s6(spanning_list, slot_dict, diversification_dict, capacity_dict)

    moments = sharpe_moments(net_ser.loc[net_ser.ne(0).idxmax():].to_numpy())
    adoption_ser = s4.grid_df[s4.live_label_str].loc[plan.adoption_date_str:]
    bundle = {
        "pod_str": plan.name_str,
        "family": {"grid": family.param_grid_dict, "live": family.live_config_dict, "family_id_str": family.family_id_str},
        "prior_trial_count_int": plan.prior_trial_count_int, "s4": s4, "s5": s5, "s6": s6,
        "post_adoption": {
            "adoption_date_str": plan.adoption_date_str, "sessions_int": len(adoption_ser),
            "performance": performance_dict(adoption_ser, tbill_ser) if len(adoption_ser) > 20 else {},
            "min_track_record_months_float": float(minimum_track_record_length(moments.sharpe_float, moments.skewness_float, moments.kurtosis_float) / 21.0),
        },
    }
    if plan.s3_fn is not None:
        class_str, note_str, result_dict = plan.s3_fn()
        bundle["s3"] = {"class_str": class_str, "note_str": note_str, "result": result_dict}
    pod_dir_path = OUTPUT_DIR_PATH / plan.name_str.replace(" ", "_").replace("/", "-")
    pod_dir_path.mkdir(parents=True, exist_ok=True)
    with (pod_dir_path / "bundle.pkl").open("wb") as file_obj:
        pickle.dump(bundle, file_obj)
    summary_dict = {
        "grade": grade_str(bundle), "s4": s4.check_list, "s5": s5.check_list, "s6": s6.check_list,
        "mcpt": [{"name": c.name_str, "p": c.p_value_float, "observed": c.observed_float, "null_95": float(np.quantile(c.null_score_vec, 0.95))} for c in mcpt_list],
        "spanning": spanning_list, "slot": slot_dict, "capacity": capacity_dict, "plateau": s4.plateau_dict, "live": s4.live_dict,
        "luck": {k: v for k, v in s4.luck_dict["live"].items() if k != "worst_offset_return_ser"},
    }
    (pod_dir_path / "summary.json").write_text(json.dumps(summary_dict, indent=2, default=str), encoding="utf-8")
    CARD_DIR_PATH.mkdir(parents=True, exist_ok=True)
    (CARD_DIR_PATH / f"{pod_dir_path.name}_reaudition.html").write_text(render_card(bundle), encoding="utf-8")
    return bundle
