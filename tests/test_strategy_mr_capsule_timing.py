"""ExecutionTimingAnalysis on the real capsule hooks: it must run, and its default cell must be the Bench run."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from test_strategy_mr_capsule_pods import _synthetic_market

from alpha.engine.backtest import run_daily
from alpha.engine.execution_timing import ExecutionTimingAnalysis
from strategies.mr_capsule import dv2_vix_gated, dv2_vix_gated_liq, hpi_vote_vix_gated, hpi_vote_vix_gated_liq

CASE_LIST = [
    pytest.param(dv2_vix_gated, "load_pricing_data", "dv2", False, id="dv2_one_over_n"),
    pytest.param(dv2_vix_gated_liq, "load_pricing_data", "dv2", True, id="dv2_sized"),
    pytest.param(hpi_vote_vix_gated, "load_hpi_capsule_pricing_data", "hpi", False, id="hpi_one_over_n"),
    pytest.param(hpi_vote_vix_gated_liq, "load_hpi_capsule_pricing_data", "hpi", True, id="hpi_sized"),
]


def _market(pod_str: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    name_int, session_int, seed_int = (40, 700, 31) if pod_str == "dv2" else (160, 1900, 32)
    pricing_df = _synthetic_market(name_int, session_int, seed_int, with_parking_bool=True, drift_float=0.0010, market_vol_float=0.012)
    rng = np.random.default_rng(seed_int)
    stock_list = [f"S{i:03d}" for i in range(name_int)]
    for symbol_str in stock_list:
        pricing_df[(symbol_str, "Turnover")] = pricing_df[(symbol_str, "Turnover")] * float(np.exp(rng.normal(0.0, 1.5)))
    benchmark_df = pricing_df["S000"][["Open", "High", "Low", "Close", "Volume"]]
    pricing_df = pd.concat([pricing_df, pd.concat({"$SPX": benchmark_df, "$SPXTR": benchmark_df}, axis=1)], axis=1)
    pricing_df.attrs["norgate_adjustment_by_symbol_dict"] = {
        symbol_str: ("TOTALRETURN" if symbol_str.startswith("$") else "CAPITALSPECIAL") for symbol_str in pricing_df.columns.get_level_values(0).unique()}
    # VIX rises through its expanding mean except for two falls, so the gate opens and closes.
    vix_index = pd.bdate_range("1990-01-02", pricing_df.index[-1])
    vix_vec = np.linspace(10.0, 40.0, len(vix_index))
    vix_vec[-400:-340] = 5.0
    vix_vec[-120:-90] = 5.0
    return pricing_df, pd.DataFrame(1, index=pricing_df.index, columns=stock_list), pd.Series(vix_vec, index=vix_index)


def _analysis(monkeypatch, pod_module, loader_str, pod_str, sized_bool, entry_tuple, exit_tuple, session_int):
    pricing_df, universe_df, vix_close_ser = _market(pod_str)
    monkeypatch.setattr(pod_module, loader_str, lambda *_: (pricing_df, universe_df))
    monkeypatch.setattr(pod_module, "load_vix_close_ser", lambda *_: vix_close_ser)
    stem_str = f"build_{pod_str}_capsule_{'liquidity_' if sized_bool else ''}execution_timing_analysis_inputs"
    kwarg_dict = {"strategy_name_str": "timing"} if sized_bool else {"strategy_name_str": "timing", "parking_enabled_bool": True, "spmo_parking_enabled_bool": False}
    input_dict = getattr(pod_module, stem_str)(**kwarg_dict)
    input_dict.update(calendar_idx=pricing_df.index[-session_int:], entry_timing_str_tuple=entry_tuple, exit_timing_str_tuple=exit_tuple)
    return ExecutionTimingAnalysis(**input_dict, save_output_bool=False), input_dict, pricing_df, universe_df, vix_close_ser


def _trades(strategy_obj) -> pd.DataFrame:
    return strategy_obj.get_transactions()[["bar", "asset", "amount", "price"]].reset_index(drop=True)


@pytest.mark.parametrize("pod_module,loader_str,pod_str,sized_bool", CASE_LIST)
def test_the_default_cell_is_the_bench_run_trade_for_trade(monkeypatch, pod_module, loader_str, pod_str, sized_bool):
    analysis_obj, input_dict, pricing_df, universe_df, vix_close_ser = _analysis(monkeypatch, pod_module, loader_str, pod_str, sized_bool, ("next_open",), ("next_open",), 260)
    timing_obj = analysis_obj.run().strategy_map[("next_open", "next_open")]
    # The plain builder, as the Bench run uses it: the engine's own current_bar is already the execution session.
    bench_obj = getattr(pod_module, f"build_{pod_str}_capsule_{'liquidity_' if sized_bool else ''}strategy")(
        strategy_name_str="bench", parking_enabled_bool=True, spmo_parking_enabled_bool=False, universe_df=universe_df, vix_close_ser=vix_close_ser)
    run_daily(bench_obj, pricing_df, input_dict["calendar_idx"], show_progress=False, show_signal_progress_bool=False)
    bench_df = _trades(bench_obj)
    assert (bench_df["asset"] == "BIL").sum() > 5 and (bench_df["asset"] != "BIL").sum() > 5
    # Timing fills a day's exits first and Bench in placement order; the fills of a day are the same.
    key_list = ["bar", "asset", "amount"]
    pd.testing.assert_frame_equal(_trades(timing_obj).sort_values(key_list).reset_index(drop=True), bench_df.sort_values(key_list).reset_index(drop=True))
    np.testing.assert_allclose(timing_obj.results["total_value"].astype(float), bench_obj.results["total_value"].astype(float))


@pytest.mark.parametrize("pod_module,loader_str,pod_str,sized_bool", CASE_LIST)
def test_the_weekly_bil_sweep_still_fires_on_a_decision_at_the_close(monkeypatch, pod_module, loader_str, pod_str, sized_bool):
    # Signal-bar mode sets current_bar to the decision session; without the next session no week ever ends for the parking rule.
    analysis_obj, *_ = _analysis(monkeypatch, pod_module, loader_str, pod_str, sized_bool, ("next_open",), ("next_open",), 60)
    trade_df = _trades(analysis_obj.run().strategy_map[("next_open", "next_open")])
    bil_bar_ser = pd.to_datetime(trade_df.loc[trade_df["asset"] == "BIL", "bar"])
    assert bil_bar_ser.dt.isocalendar().week.nunique() > 5


@pytest.mark.parametrize("pod_module,loader_str,pod_str,sized_bool", [CASE_LIST[1], CASE_LIST[3]])
def test_every_cell_of_the_matrix_runs(monkeypatch, pod_module, loader_str, pod_str, sized_bool):
    timing_tuple = ("same_close_moc", "next_open", "next_close")
    analysis_obj, *_ = _analysis(monkeypatch, pod_module, loader_str, pod_str, sized_bool, timing_tuple, timing_tuple, 120)
    result_obj = analysis_obj.run()
    assert set(result_obj.strategy_map) == {(entry_str, exit_str) for entry_str in timing_tuple for exit_str in timing_tuple}
    assert len(result_obj.metric_df) == 9 and result_obj.metric_df["Ann. Return [%]"].notna().all()
