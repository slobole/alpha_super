"""Independent review (live-parity / failure-modes lens) of the 2026-09-28 readiness audit.

Synthetic data and fakes only: no Norgate, no broker, no network. Tests whose name contains
``documents_risk`` pin a behaviour that the review reports as a risk; they pass on the current
code and should be updated when the mitigation lands.
"""
from __future__ import annotations

import io
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import alpha.data.fred_loader as fred_loader_module
from alpha.live.execution_engine import build_broker_order_request_list_from_vplan, build_vplan
from alpha.live.models import BrokerSnapshot, DecisionPlan, LivePriceSnapshot, LiveRelease
from alpha.live.reconcile import reconcile_account_state
from strategies.momentum.strategy_mo_atr_normalized_ndx import (
    AtrNormalizedNdxConfig,
    compute_atr_normalized_signal_tables,
)
from strategies.taa_df.strategy_taa_df import compute_month_end_weight_df
from strategies.taa_df.strategy_taa_df_btal_fallback_tqqq import DEFAULT_CONFIG as TAA_BASE_CONFIG

ET = ZoneInfo("America/New_York")


def _et(year, month, day, hour=0, minute=0, second=0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=ET)


def _release() -> LiveRelease:
    return LiveRelease(
        release_id_str="review.release",
        user_id_str="review_user",
        pod_id_str="pod_review",
        account_route_str="U000REVIEW",
        strategy_import_str="strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash",
        mode_str="live",
        session_calendar_id_str="XNYS",
        signal_clock_str="month_end_snapshot_ready",
        execution_policy_str="next_month_first_open",
        data_profile_str="norgate_eod_etf_plus_vix_helper",
        params_dict={},
        risk_profile_str="review",
        enabled_bool=True,
        source_path_str="review.yaml",
        pod_budget_fraction_float=1.0,
        auto_submit_enabled_bool=False,
    )


def _plan(weight_map_dict, base_position_map_dict) -> DecisionPlan:
    return DecisionPlan(
        release_id_str="review.release",
        user_id_str="review_user",
        pod_id_str="pod_review",
        account_route_str="U000REVIEW",
        signal_timestamp_ts=_et(2026, 9, 30, 16, 0),
        submission_timestamp_ts=_et(2026, 10, 1, 9, 23, 30),
        target_execution_timestamp_ts=_et(2026, 10, 1, 9, 30),
        execution_policy_str="next_month_first_open",
        decision_base_position_map=dict(base_position_map_dict),
        snapshot_metadata_dict={"norgate_data_profile_str": "norgate_eod_etf_plus_vix_helper"},
        strategy_state_dict={},
        decision_book_type_str="full_target_weight_book",
        full_target_weight_map_dict=dict(weight_map_dict),
        cash_reserve_weight_float=max(0.0, 1.0 - sum(weight_map_dict.values())),
        preserve_untouched_positions_bool=False,
        rebalance_omitted_assets_to_zero_bool=True,
        decision_plan_id_int=11,
    )


def _broker(net_liq_float, position_map_dict) -> BrokerSnapshot:
    return BrokerSnapshot(
        account_route_str="U000REVIEW",
        snapshot_timestamp_ts=_et(2026, 10, 1, 9, 23, 30),
        cash_float=0.0,
        total_value_float=net_liq_float,
        position_amount_map=dict(position_map_dict),
        net_liq_float=net_liq_float,
    )


def _prices(price_map_dict) -> LivePriceSnapshot:
    return LivePriceSnapshot(
        account_route_str="U000REVIEW",
        snapshot_timestamp_ts=_et(2026, 10, 1, 9, 23, 31),
        price_source_str="ib_async.reqMktData.225.auctionPrice",
        asset_reference_price_map=dict(price_map_dict),
    )


# ---------------------------------------------------------------------------------------------
# R-01: a broker read that misses positions is caught only by a WARN-only reconciliation
# ---------------------------------------------------------------------------------------------


def test_documents_risk_empty_broker_position_read_rebuys_full_target_and_reconcile_only_warns():
    # Model (pod state) holds the full TQQQ book; the pre-open broker read returns no positions
    # (API sync gap, wrong account filter, etc.) while NetLiq is correct.
    plan_obj = _plan({"TQQQ": 1.0}, {"TQQQ": 1000.0})
    broker_obj = _broker(100_000.0, {})
    reconciliation_obj = reconcile_account_state(
        model_position_map=plan_obj.decision_base_position_map,
        model_cash_float=0.0,
        broker_snapshot_obj=broker_obj,
    )
    # The mismatch IS detected ...
    assert reconciliation_obj.passed_bool is False
    assert "TQQQ" in reconciliation_obj.mismatch_dict
    # ... but for non-CORE5 pods runner.py:2925-2955 only logs a warning and continues, so the
    # VPlan buys the whole target again on top of the real 1,000 shares (2x exposure).
    vplan_obj = build_vplan(_release(), plan_obj, broker_obj, _prices({"TQQQ": 100.0}))
    request_list = build_broker_order_request_list_from_vplan(vplan_obj)
    assert [(r.asset_str, r.amount_float) for r in request_list] == [("TQQQ", 1000.0)]


# ---------------------------------------------------------------------------------------------
# R-02: no sanity band between the auction indicative price and the decision close
# ---------------------------------------------------------------------------------------------


def test_documents_risk_outlier_auction_indicative_price_is_accepted_for_sizing():
    # BTAL closed at 20.00; a thin pre-open book shows an indicative of 40.00 (or 10.00).
    # Any positive finite price is accepted: the order is off by the price ratio.
    plan_obj = _plan({"BTAL": 0.5, "GLD": 0.5}, {"BTAL": 2500.0, "GLD": 200.0})
    broker_obj = _broker(100_000.0, {"BTAL": 2500.0, "GLD": 200.0})
    high_vplan = build_vplan(_release(), plan_obj, broker_obj, _prices({"BTAL": 40.0, "GLD": 250.0}))
    low_vplan = build_vplan(_release(), plan_obj, broker_obj, _prices({"BTAL": 10.0, "GLD": 250.0}))
    assert high_vplan.order_delta_map["BTAL"] == -1250.0  # sells half the sleeve
    assert low_vplan.order_delta_map["BTAL"] == 2500.0  # doubles the sleeve


# ---------------------------------------------------------------------------------------------
# R-03: live sizes on the auction price, the backtest on Close_T (A-LIVE-06 sharpened)
# ---------------------------------------------------------------------------------------------


def test_documents_risk_live_share_count_differs_from_taa_backtest_prior_close_rule():
    # strategies/taa_df/strategy_taa_df.py:411-415 and :497 size q = int(V_close * w / Close_T).
    nav_close_float, close_float, auction_float = 100_000.0, 100.0, 92.0
    backtest_target_int = int(nav_close_float * 1.0 / close_float)
    held_float = float(backtest_target_int)
    vplan_obj = build_vplan(
        _release(),
        _plan({"TQQQ": 1.0}, {"TQQQ": held_float}),
        _broker(nav_close_float, {"TQQQ": held_float}),  # NetLiq still marked at Close_T pre-open
        _prices({"TQQQ": auction_float}),
    )
    # Backtest: target equals holding -> delta 0 -> no order (strategy_taa_df.py:498-500).
    assert backtest_target_int - held_float == 0
    # Live: buys 86 more shares (8.6% extra exposure after an 8% down gap).
    assert vplan_obj.order_delta_map == {"TQQQ": 86.0}


# ---------------------------------------------------------------------------------------------
# R-04: a missing or lagging month-end bar is used silently by the TAA and NDX signal code
# ---------------------------------------------------------------------------------------------


def _taa_signal_df(last_day_nan_symbol_str: str | None) -> pd.DataFrame:
    date_index = pd.bdate_range("2023-01-02", "2024-06-28")
    rng = np.random.default_rng(7)
    data_dict = {}
    for i, symbol_str in enumerate(TAA_BASE_CONFIG.defensive_asset_list):
        data_dict[symbol_str] = 50.0 * np.exp(np.cumsum(rng.normal(0.0002 * (i - 2), 0.01, len(date_index))))
    signal_df = pd.DataFrame(data_dict, index=date_index)
    if last_day_nan_symbol_str is not None:
        signal_df.loc[date_index[-1], last_day_nan_symbol_str] = np.nan
    return signal_df


def test_documents_risk_taa_month_end_signal_silently_uses_prior_close_when_bar_missing():
    cash_ser = pd.Series(0.004, index=pd.bdate_range("2023-01-02", "2024-06-28"), name="cash_return")
    full_df = _taa_signal_df(None)
    lagged_df = _taa_signal_df("GLD")
    _, full_weight_df = compute_month_end_weight_df(full_df, cash_ser, TAA_BASE_CONFIG)
    _, lagged_weight_df = compute_month_end_weight_df(lagged_df, cash_ser, TAA_BASE_CONFIG)
    # No error, same month-end label: resample("ME").last() quietly takes GLD's 2024-06-27 close.
    assert lagged_weight_df.index[-1] == pd.Timestamp("2024-06-30")
    assert lagged_df.resample("ME").last().loc["2024-06-30", "GLD"] == lagged_df["GLD"].iloc[-2]
    assert full_weight_df.index.equals(lagged_weight_df.index)


def _ndx_panel(nan_symbol_str: str | None):
    date_index = pd.bdate_range("2023-01-02", "2024-06-28")
    rng = np.random.default_rng(11)
    symbol_list = [f"S{i:02d}" for i in range(12)]
    close_df = pd.DataFrame(
        {s: 100.0 * np.exp(np.cumsum(rng.normal(0.0008, 0.015, len(date_index)))) for s in symbol_list},
        index=date_index,
    )
    high_df, low_df = close_df * 1.01, close_df * 0.99
    regime_ser = pd.Series(np.linspace(100.0, 160.0, len(date_index)), index=date_index)
    if nan_symbol_str is not None:
        for frame_df in (close_df, high_df, low_df):
            frame_df.loc[date_index[-1], nan_symbol_str] = np.nan
    return close_df, high_df, low_df, regime_ser


def test_documents_risk_ndx_member_without_month_end_bar_is_silently_dropped_from_ranking():
    close_df, high_df, low_df, regime_ser = _ndx_panel("S03")
    tables = compute_atr_normalized_signal_tables(
        price_close_df=close_df,
        price_high_df=high_df,
        price_low_df=low_df,
        regime_close_ser=regime_ser,
        config=AtrNormalizedNdxConfig(),
        price_unadjusted_close_df=close_df,
    )
    monthly_decision_close_df, risk_adj_score_df = tables[0], tables[6]
    # The month-end decision still happens on 2024-06-28 ...
    assert monthly_decision_close_df.index[-1] == pd.Timestamp("2024-06-28")
    # ... and S03 simply has no score (excluded from the candidate set without any error).
    assert np.isnan(risk_adj_score_df.loc["2024-06-28", "S03"])
    assert np.isfinite(risk_adj_score_df.loc["2024-06-28"].drop("S03")).all()


# ---------------------------------------------------------------------------------------------
# R-05: the FRED "<= as_of" filter uses the UTC date of as_of
# ---------------------------------------------------------------------------------------------


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False


def test_documents_risk_fred_as_of_filter_uses_utc_date(monkeypatch, tmp_path):
    csv_bytes = b"observation_date,DTB3\n2026-08-28,4.00\n2026-08-31,4.01\n2026-09-01,4.02\n"
    monkeypatch.setattr(fred_loader_module, "urlopen", lambda url, timeout=None: _Resp(csv_bytes))
    snapshot_obj = fred_loader_module.load_daily_fred_series_snapshot(
        series_id_str="DTB3",
        cache_csv_path_str=str(tmp_path / "DTB3.csv"),
        as_of_ts=_et(2026, 8, 31, 20, 0),  # = 2026-09-01 00:00 UTC
        mode_str="live",
    )
    # A replay at T 20:00 New York admits the observation dated T+1 (harmless live, where FRED has
    # not published it yet, but it means a replay does not reproduce live's information set).
    assert snapshot_obj.latest_observation_date_ts == pd.Timestamp("2026-09-01")
