"""NDX plateau ensemble and a Masters CMMA trend filter (owner requests 2026-10-02). Registered in the Scout ledger
before any number is computed.

Base: NATR20 VXN (the live NDX VXN rule with the NATR20 ranking). Everything net of the house parity costs, idle cash
at the T-bill rate; in sample = 2000-09-01 to the vault seal; 2023 on is shown as "seen".

Part A, ensembles (capital split equally; the target weights are averaged, so orders net out as one account would):
    E1  plateau ensemble: ROC {9, 12, 15} month-ends x Close > SMA {100, 150, 200}, top 10 each, 1/9 each
    E2  ranking blend: 1/2 dollar ATR (the live pod) + 1/2 NATR20, both ROC 12 / SMA100 / top 10
  compared with the live pod (dollar ATR), NATR20 single, the plateau single (ROC 12 / SMA150) and the average of the
  nine singles inside E1 (what picking one plateau cell at random earns). Robustness: the 16 rebalance offsets.
Part B, Masters CMMA (alpha/scout/specs/ndx_vxn.py `cmma_frame`):
    B1  indicator quality (Masters battery, S2) on Nasdaq-100 members at month ends: raw distance Close / SMA100 - 1
        against CMMA(L, 252) for L = 50, 100, 200; plus threshold drift per 5-year block (the share of members a
        fixed threshold selects).
    B2  CMMA as the stock filter (NATR20 VXN, ROC 12, top 10): L {50, 100, 200} x threshold {-20, -10, 0, +10, +20}
        with A = 252, and L = 100, A = 63 at {-10, 0, +10}: 18 configurations.

    uv run python scripts/research/scout_robustness_20261002/ndx_ensemble_cmma.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from alpha.scout.engines.weights import CostModel, simulate
from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH, Ledger
from alpha.scout.metrics import performance_dict, sharpe_float
from alpha.scout.registration import Registration, register, registration_rows
from alpha.scout.stations.robustness import paired_sharpe_probability
from alpha.scout.stations.s2_indicator import BLOCK_TUPLE, masters_battery
from alpha.scout.stations.s4_strategy import SEAL_END_STR
from ndx_ranking_compare import CRISIS_DICT, window_return
from run import cash_credited, market_inputs

OUTPUT_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "robustness" / "ndx_ensemble_cmma.json"
START_STR = "2000-09-01"
NATR_DICT = {"atr_unit_str": "percent"}
E1_LIST = [{**NATR_DICT, "roc_month_int": r, "stock_sma_int": m} for r in (9, 12, 15) for m in (100, 150, 200)]
STRATEGY_DICT = {
    "Live (dollar ATR)": [{}],
    "NATR20 single": [NATR_DICT],
    "Plateau single (ROC12 SMA150)": [{**NATR_DICT, "stock_sma_int": 150}],
    "E1 plateau ensemble (9)": E1_LIST,
    "E2 ranking blend (ATR+NATR)": [{}, NATR_DICT],
}
CMMA_LIST = ([{"stock_sma_int": lb, "cmma_atr_int": 252, "cmma_threshold_float": th} for lb in (50, 100, 200) for th in (-20.0, -10.0, 0.0, 10.0, 20.0)]
             + [{"stock_sma_int": 100, "cmma_atr_int": 63, "cmma_threshold_float": th} for th in (-10.0, 0.0, 10.0)])
REGISTRATION = Registration(
    registration_id_str="ndx_natr20_vxn_ensemble_cmma_20261002", family_id_str="equity_cross_sectional_momentum",
    hypothesis_str=("NDX NATR20 VXN: (a) holding the 3x3 plateau of ROC and SMA filter as an equal ensemble keeps the "
                    "plateau's Sharpe with a narrower luck band than any single cell, and a 50/50 blend of the dollar-ATR and "
                    "NATR20 rankings combines the crash behaviour of the first with the long-run edge of the second; (b) "
                    "Masters' CMMA (volatility-normalised, bounded) is a more stationary stock trend measure than the raw "
                    "distance to the SMA, and as a filter it is at least as good as Close > SMA100."),
    mechanism_str="Cross-sectional momentum; parameter and ranking diversification; volatility-normalised trend state.",
    expected_sign_and_location_str="Ensemble Sharpe near the plateau median with lower dispersion; CMMA filter within noise of the live filter.",
    hypothesis_class_str="X", universe_str="Nasdaq-100 point-in-time members", horizon_str="months",
    schedule_str="month-end decision", execution_str="next session's open",
    param_grid_dict={"ensemble": ("E1 plateau 3x3", "E2 ATR+NATR blend"), "cmma_lookback_int": (50, 100, 200),
                     "cmma_threshold_float": (-20.0, -10.0, 0.0, 10.0, 20.0), "cmma_atr_int": (63, 252)},
    grid_justification_str=("2 ensembles plus 18 CMMA filter configurations (15 at A = 252, 3 at A = 63); the grid "
                            "product overstates the trials. A comparison, not a search: nothing is promoted from it."),
    primary_metric_str="In-sample net Sharpe and its 16-offset luck band; paired bootstrap P vs NATR20 single; Masters S2 battery",
    kill_criteria_str="Not a promotion candidate; any change goes through S4-S6 and a forward shadow.",
    source_str="Masters, Statistically Sound Indicators (CMMA); Carver (parameter diversification); alpha/scout/specs/ndx_vxn.py",
    parent_id_str="ndx_natr20_vxn_reaudition_20261002",
    universe_choice_str="The pod's own universe (Nasdaq-100 point-in-time), unchanged.",
)
_WORKER: dict = {}


def _init(inputs, tbill_ser) -> None:
    _WORKER["inputs"], _WORKER["tbill"] = inputs, tbill_ser


def run_strategy(override_list: list[dict], offset_int: int = 0) -> dict:
    """Average the members' target weights (equal capital), simulate once, credit idle cash."""
    from alpha.scout.specs import ndx_vxn

    inputs = _WORKER["inputs"]
    frame_list = [ndx_vxn.rebalance_weight_df(inputs, dataclasses.replace(ndx_vxn.LIVE_CONFIG, **o, decision_offset_int=offset_int))
                  for o in override_list]
    index = frame_list[0].index
    for frame in frame_list[1:]:
        index = index.union(frame.index)
    columns = sorted(set().union(*[f.columns for f in frame_list]))
    # A member with no row yet (its warm-up) holds nothing: 0 before its first decision. Every later month has a row.
    weight_df = sum(f.reindex(index=index, columns=columns).fillna(0.0) for f in frame_list) / len(frame_list)
    stock_list = list(weight_df.columns)
    result = simulate(inputs.open_df[stock_list], inputs.close_df[stock_list], inputs.dividend_df[stock_list], weight_df,
                      start_date=ndx_vxn.TRADING_START_STR, capital_float=100_000.0, share_unit_mode_str="historical",
                      unadjusted_close_df=inputs.raw_close_df[stock_list], cost_model=CostModel())
    daily_ser = cash_credited(result, inputs.close_df, _WORKER["tbill"]).loc[START_STR:]
    held_ser = (result.daily_position_df.loc[START_STR:SEAL_END_STR] != 0).sum(axis=1)
    trade_df = result.trade_df.loc[(result.trade_df["date"] >= START_STR) & (result.trade_df["date"] <= SEAL_END_STR)]
    traded_float = float((trade_df["delta_float"].abs() * trade_df["price_float"]).sum())
    value_float = float(result.total_value_ser.loc[START_STR:SEAL_END_STR].mean())
    years_float = len(result.total_value_ser.loc[START_STR:SEAL_END_STR]) / 252.0
    position_df = result.daily_position_df.loc[START_STR:SEAL_END_STR]
    long_value_ser = (position_df * inputs.close_df[stock_list].reindex(position_df.index)).clip(lower=0.0).sum(axis=1)
    exposure_float = float((long_value_ser / result.total_value_ser.loc[position_df.index]).mean())
    return {"daily": daily_ser, "names_held_float": float(held_ser[held_ser > 0].mean()), "turnover_float": traded_float / value_float / years_float,
            "exposure_float": exposure_float}


def _task(args) -> dict:
    key_str, override_list, offset_int = args
    try:
        return {"key": key_str, "offset": offset_int, **run_strategy(override_list, offset_int)}
    except Exception as error_obj:  # noqa: BLE001 - listed as failed
        return {"key": key_str, "offset": offset_int, "error": repr(error_obj)[:200]}


def indicator_quality(inputs) -> dict:
    """Masters battery and threshold drift on members at month ends (B1)."""
    from alpha.scout.specs import ndx_vxn

    decision_index = ndx_vxn.decision_dates(inputs.close_df.index)
    decision_index = decision_index[(decision_index >= START_STR) & (decision_index <= SEAL_END_STR)]
    stock_list = [s for s in inputs.close_df.columns if s != ndx_vxn.REGIME_SYMBOL_STR]
    close_df = inputs.close_df[stock_list]
    member_df = (inputs.member_df[stock_list] == 1).reindex(decision_index)
    month_close_df = close_df.reindex(decision_index)
    forward_df = month_close_df.shift(-1) / month_close_df - 1.0  # decision close to the next decision close
    forward_excess_df = forward_df.sub(forward_df.where(member_df).mean(axis=1), axis=0)
    indicator_dict = {"Close / SMA100 - 1 (raw)": (close_df / close_df.rolling(100, min_periods=100).mean() - 1.0).reindex(decision_index)}
    for lookback_int in (50, 100, 200):
        indicator_dict[f"CMMA({lookback_int}, 252)"] = ndx_vxn.cmma_frame(inputs, lookback_int, 252)[stock_list].reindex(decision_index)
    out_dict = {}
    for name_str, indicator_df in indicator_dict.items():
        eligible_df = member_df & indicator_df.notna()
        battery = masters_battery(indicator_df, eligible_df, forward_excess_df.where(forward_excess_df.notna()))
        block_list = []
        for label_str, start_str, end_str in BLOCK_TUPLE:
            value_vec = indicator_df.where(eligible_df).loc[start_str:end_str].stack().to_numpy()
            if value_vec.size == 0:
                continue
            raw_bool = name_str.startswith("Close")
            block_list.append({"block": label_str, "q05": float(np.quantile(value_vec, 0.05)), "median": float(np.median(value_vec)),
                               "q95": float(np.quantile(value_vec, 0.95)),
                               "share_above_zero": float(np.mean(value_vec > 0)),
                               "share_above_strict": float(np.mean(value_vec > (0.10 if raw_bool else 10.0))),
                               "share_below_loose": float(np.mean(value_vec < (-0.10 if raw_bool else -10.0)))})
        out_dict[name_str] = {"battery": battery, "blocks": block_list}
    return out_dict


def summary(daily_ser: pd.Series, tbill_ser: pd.Series) -> dict:
    in_ser = daily_ser.loc[:SEAL_END_STR]
    return {"in_sample": performance_dict(in_ser, tbill_ser), "seen_2023_on": performance_dict(daily_ser.loc["2023-01-01":], tbill_ser),
            "crisis": {k: window_return(daily_ser, a, b) for k, (a, b) in CRISIS_DICT.items()}}


def main() -> None:
    from alpha.scout.specs import ndx_vxn

    ledger = Ledger()
    if REGISTRATION.registration_id_str not in registration_rows(ledger):
        register(ledger, REGISTRATION)
        print("registered", REGISTRATION.registration_id_str, flush=True)
    _, tbill_ser = market_inputs()
    inputs = ndx_vxn.load_inputs()
    _init(inputs, tbill_ser)
    task_list = [(k, v, o) for k, v in STRATEGY_DICT.items() for o in range(16)]
    task_list += [(f"single {i}", [o], 0) for i, o in enumerate(E1_LIST)]
    task_list += [(f"cmma {i}", [{**NATR_DICT, "trend_filter_str": "cmma", **c}], 0) for i, c in enumerate(CMMA_LIST)]
    with Pool(8, initializer=_init, initargs=(inputs, tbill_ser)) as pool_obj:
        result_list = pool_obj.map(_task, task_list, chunksize=1)
    failed_list = [r for r in result_list if "error" in r]
    by_key = {(r["key"], r["offset"]): r for r in result_list if "error" not in r}
    natr_in_ser = by_key[("NATR20 single", 0)]["daily"].loc[:SEAL_END_STR]

    strategy_out = {}
    for key_str in STRATEGY_DICT:
        main_dict = by_key[(key_str, 0)]
        luck_vec = np.array([sharpe_float(by_key[(key_str, o)]["daily"].loc[:SEAL_END_STR]) for o in range(16) if (key_str, o) in by_key])
        strategy_out[key_str] = {**summary(main_dict["daily"], tbill_ser), "names_held": main_dict["names_held_float"],
                                 "turnover": main_dict["turnover_float"],
                                 "luck": {"min": float(luck_vec.min()), "median": float(np.median(luck_vec)), "max": float(luck_vec.max()),
                                          "all": luck_vec.round(3).tolist()},
                                 "p_vs_natr_single": paired_sharpe_probability(main_dict["daily"].loc[:SEAL_END_STR], natr_in_ser)}
    single_sharpe_list = [sharpe_float(by_key[(f"single {i}", 0)]["daily"].loc[:SEAL_END_STR]) for i in range(len(E1_LIST))]
    cmma_out = []
    for i, c in enumerate(CMMA_LIST):
        daily_ser = by_key[(f"cmma {i}", 0)]["daily"]
        cmma_out.append({**c, **{k: v for k, v in summary(daily_ser, tbill_ser).items() if k != "crisis"},
                         "crisis": summary(daily_ser, tbill_ser)["crisis"],
                         "p_vs_natr_single": paired_sharpe_probability(daily_ser.loc[:SEAL_END_STR], natr_in_ser)})
    quality_dict = indicator_quality(inputs)

    def curve(daily_ser):
        return [round(float(v), 4) for v in (1.0 + daily_ser).cumprod().resample("ME").last()]

    out_dict = {"strategies": strategy_out, "singles_in_e1": single_sharpe_list, "cmma": cmma_out, "quality": quality_dict,
                "failed": failed_list, "month_list": [d.strftime("%Y-%m") for d in (1.0 + natr_in_ser).cumprod().resample("ME").last().index],
                "curves": {k: curve(by_key[(k, 0)]["daily"]) for k in STRATEGY_DICT}}
    OUTPUT_PATH.write_text(json.dumps(out_dict, default=float), encoding="utf-8")

    print("failed", len(failed_list), [f["error"] for f in failed_list[:3]])
    for k, v in strategy_out.items():
        s, t = v["in_sample"], v["seen_2023_on"]
        print(f"{k:32s} Sh {s['sharpe_float']:.2f} CAGR {s['cagr_float']:.1%} vol {s['volatility_float']:.1%} DD {s['max_drawdown_float']:.1%} "
              f"Calmar {s['calmar_float']:.2f} | luck {v['luck']['min']:.2f}/{v['luck']['median']:.2f}/{v['luck']['max']:.2f} | P>NATR {v['p_vs_natr_single']:.2f} "
              f"| names {v['names_held']:.1f} turn {v['turnover']:.1f} | 2023+ Sh {t['sharpe_float']:.2f} DD {t['max_drawdown_float']:.1%}")
        print(f"{'':32s} crises {[round(x * 100, 1) for x in v['crisis'].values()]}")
    print("E1 singles", np.round(single_sharpe_list, 2), "mean", round(float(np.mean(single_sharpe_list)), 3))
    for c in cmma_out:
        s = c["in_sample"]
        print(f"CMMA L{c['stock_sma_int']} A{c['cmma_atr_int']} >{c['cmma_threshold_float']:+.0f}: Sh {s['sharpe_float']:.2f} CAGR {s['cagr_float']:.1%} "
              f"DD {s['max_drawdown_float']:.1%} P>NATR {c['p_vs_natr_single']:.2f} 2023+ {c['seen_2023_on']['sharpe_float']:.2f}")
    for name_str, q in quality_dict.items():
        b = q["battery"]
        mi = b.get("mutual_information", {})
        print(f"{name_str:26s} range/IQR {b['range_over_iqr_float']:.1f} tail {b['tail_share_float']:.2%} entropy {b['relative_entropy_float']:.2f} "
              f"break {b['mean_break']['sup_wald_float']:.1f} MI {mi.get('bits_float', float('nan')):.4f} (shuf max {mi.get('shuffled_max_float', float('nan')):.4f})")
        print(f"{'':26s} share>0 by block {[round(x['share_above_zero'], 2) for x in q['blocks']]}  strict {[round(x['share_above_strict'], 2) for x in q['blocks']]}"
              f"  median {[round(x['median'], 3) for x in q['blocks']]}")


if __name__ == "__main__":
    main()
