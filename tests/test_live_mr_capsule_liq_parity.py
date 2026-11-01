"""Real indicators and real builders on a synthetic market: the backtest's orders, the live decisions and the nightly reference agree. Only data access is replaced."""

from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime
from zoneinfo import ZoneInfo

import exchange_calendars
import numpy as np
import pandas as pd
import pytest

from alpha.engine.backtest import run_daily
from alpha.live import backtest_check, mr_capsule_adapter, reference_compare, strategy_host
from alpha.live.models import PodState
from alpha.live.release_manifest import parse_release_manifest
from strategies.mr_capsule import dv2_vix_gated, dv2_vix_gated_liq, hpi_vote_vix_gated, hpi_vote_vix_gated_liq, vix_stress_gate

MARKET_ZONE_OBJ = ZoneInfo("America/New_York")
# Names, sessions, first trading position: HPI needs many names and long history for candidates at all.
CASE_DICT = {"dv2_vix_gated": (40, 900, 300), "hpi_vote_vix_gated": (160, 2200, 1800)}


def _market(pod_str: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, int]:
    name_int, session_int, start_int = CASE_DICT[pod_str]
    all_session_idx = exchange_calendars.get_calendar("XNYS", start="1990-01-02", end="2026-10-14").sessions.tz_localize(None)
    session_idx = all_session_idx[-session_int:]
    rng = np.random.default_rng(20261101)
    stock_list = [f"S{symbol_int:03d}" for symbol_int in range(name_int)]
    market_vec = rng.normal(0.0, 0.012, session_int)
    column_dict = {}
    for symbol_str in [*stock_list, "BIL", "SPMO", "$SPX", "$SPXTR"]:
        close_vec = (91.0 * np.cumprod(1 + rng.normal(0.00008, 0.0002, session_int)) if symbol_str == "BIL"
                     else 50.0 * np.cumprod(1 + market_vec + rng.normal(0.0010, 0.018, session_int)))
        open_vec = close_vec * (1 + rng.normal(0, 0.004, session_int))
        volume_vec = rng.integers(1_000_000, 5_000_000, session_int).astype(float)
        column_dict.update({
            (symbol_str, "Open"): open_vec, (symbol_str, "Close"): close_vec, (symbol_str, "Volume"): volume_vec,
            (symbol_str, "High"): np.maximum(open_vec, close_vec) * (1 + np.abs(rng.normal(0, 0.006, session_int))),
            (symbol_str, "Low"): np.minimum(open_vec, close_vec) * (1 - np.abs(rng.normal(0, 0.006, session_int))),
            (symbol_str, "Turnover"): close_vec * volume_vec * float(np.exp(rng.normal(0.0, 1.5))),
            (symbol_str, "Unadjusted Close"): close_vec, (symbol_str, "Dividend"): np.zeros(session_int),
        })
    pricing_df = pd.DataFrame(column_dict, index=session_idx)
    pricing_df.attrs["norgate_adjustment_by_symbol_dict"] = {
        symbol_str: ("TOTALRETURN" if symbol_str.startswith("$") else "CAPITALSPECIAL") for symbol_str in pricing_df.columns.get_level_values(0).unique()}
    # VIX rises through its expanding mean, so the gate stays open.
    return pricing_df, pd.DataFrame(1, index=session_idx, columns=stock_list), pd.Series(np.linspace(10.0, 40.0, len(all_session_idx)), index=all_session_idx), start_int


def _install_market(monkeypatch, pricing_df, universe_df, vix_close_ser) -> None:
    def load_pricing(end_date_str=None):
        return (pricing_df.loc[:end_date_str].copy() if end_date_str else pricing_df.copy()), universe_df

    def load_vix(end_date_str=None):
        return vix_close_ser.loc[:end_date_str] if end_date_str else vix_close_ser

    for module_obj, loader_str in ((dv2_vix_gated, "load_pricing_data"), (dv2_vix_gated_liq, "load_pricing_data"),
                                   (hpi_vote_vix_gated, "load_hpi_capsule_pricing_data"), (hpi_vote_vix_gated_liq, "load_hpi_capsule_pricing_data")):
        monkeypatch.setattr(module_obj, loader_str, load_pricing)
    for module_obj in (dv2_vix_gated_liq, hpi_vote_vix_gated_liq, vix_stress_gate):
        monkeypatch.setattr(module_obj, "load_vix_close_ser", load_vix)
    # Synthetic history has no five years of HPI features for every member; the rule itself is covered in the 1/N parity tests.
    monkeypatch.setattr(strategy_host, "_validate_hpi_live_feature_readiness", lambda **_: None)
    monkeypatch.setattr(reference_compare, "use_norgate_data_profile", lambda _profile_str: nullcontext())


def _backtest_with_account_record(pod_str, pricing_df, universe_df, vix_close_ser, calendar_idx):
    """The real builder and engine; records the account each decision saw and the orders it placed."""
    builder_fn = dv2_vix_gated_liq.build_dv2_capsule_liquidity_strategy if pod_str == "dv2_vix_gated" else hpi_vote_vix_gated_liq.build_hpi_capsule_liquidity_strategy
    strategy_obj = builder_fn(strategy_name_str="parity", parking_enabled_bool=True, spmo_parking_enabled_bool=False, universe_df=universe_df, vix_close_ser=vix_close_ser, capital_base_float=100_000.0)
    record_dict = {}
    pod_class = type(strategy_obj)

    class RecordingPod(pod_class):
        def iterate(self, data, close, open_prices):
            account_dict = {
                "position_map_dict": {str(asset_obj): float(share_float) for asset_obj, share_float in self.get_positions().items() if float(share_float) != 0.0},
                "cash_float": float(self.cash), "nav_float": float(self.previous_total_value), "strategy_state_dict": strategy_host._extract_strategy_state_dict(self),
            }
            decision_ts = pd.Timestamp(self.previous_bar)
            super().iterate(data, close, open_prices)
            record_dict[decision_ts] = {**account_dict, "order_list": [strategy_host._build_order_shape_dict(order_obj) for order_obj in self.get_orders()]}

    strategy_obj.__class__ = RecordingPod
    run_daily(strategy_obj, pricing_df.loc[:calendar_idx[-1]], calendar_idx, show_progress=False, show_signal_progress_bool=False)
    return strategy_obj, record_dict


def _entry_dollar_dict(order_list):
    return {order_dict["asset_str"]: order_dict["amount_float"] for order_dict in order_list if order_dict["unit_str"] == "value" and not order_dict["target_bool"]}


@pytest.mark.parametrize("pod_str", list(CASE_DICT))
def test_live_decisions_and_the_nightly_reference_match_the_backtest(monkeypatch, tmp_path, pod_str):
    pricing_df, universe_df, vix_close_ser, start_int = _market(pod_str)
    _install_market(monkeypatch, pricing_df, universe_df, vix_close_ser)
    release_obj = parse_release_manifest(f"docs/live/release_templates/pod_mr_{pod_str}_liq_bil_daily_moo.yaml.example")
    calendar_idx = pricing_df.index[start_int:-1]
    strategy_obj, record_dict = _backtest_with_account_record(pod_str, pricing_df, universe_df, vix_close_ser, calendar_idx)

    weight_ser = pd.Series(strategy_obj.entry_weight_by_date_dict)
    assert len(weight_ser) > 40 and weight_ser.nunique() > 40  # real sizes, not an equal slot
    assert np.isfinite(strategy_obj.liquidity_reference_ser.iloc[-1])

    entry_date_list = [date_ts for date_ts in sorted(record_dict) if _entry_dollar_dict(record_dict[date_ts]["order_list"])]
    capped_date_list = [date_ts for date_ts in entry_date_list
                        if any(dollar_float < strategy_obj.entry_weight_by_date_dict[(pricing_df.index[pricing_df.index.get_loc(date_ts) + 1].date().isoformat(), asset_str)] * record_dict[date_ts]["nav_float"] - 1e-6
                               for asset_str, dollar_float in _entry_dollar_dict(record_dict[date_ts]["order_list"]).items())]
    assert len(capped_date_list) >= 2  # the account-dependent branch of the rule must be in the sample
    sample_list = sorted({*entry_date_list[:5], *capped_date_list[:3], *entry_date_list[-3:], *sorted(record_dict)[-2:]})

    metadata_dict = {}
    monkeypatch.setattr(mr_capsule_adapter, "build_data_source_metadata_dict", lambda *_: dict(metadata_dict))
    monkeypatch.setattr(strategy_host, "build_data_source_metadata_dict", lambda *_: dict(metadata_dict))
    decision_dict = {}
    for date_ts in sample_list:
        account_dict = record_dict[date_ts]
        metadata_dict.update({"norgate_snapshot_date_str": date_ts.date().isoformat(), "norgate_data_profile_str": release_obj.data_profile_str, "norgate_manifest_hash_str": "parity"})
        state_obj = PodState(
            pod_id_str=release_obj.pod_id_str, user_id_str=release_obj.user_id_str, account_route_str=release_obj.account_route_str, position_amount_map=dict(account_dict["position_map_dict"]),
            cash_float=account_dict["cash_float"], total_value_float=account_dict["nav_float"],
            strategy_state_dict={**account_dict["strategy_state_dict"], "mr_capsule_strategy_import_str": release_obj.strategy_import_str},
            updated_timestamp_ts=datetime(date_ts.year, date_ts.month, date_ts.day, 16, 30, tzinfo=MARKET_ZONE_OBJ), snapshot_stage_str="eod", snapshot_source_str="broker",
        )
        decision_obj = strategy_host.build_decision_plan_for_release(release_obj, datetime(date_ts.year, date_ts.month, date_ts.day, 18, tzinfo=MARKET_ZONE_OBJ), state_obj)
        decision_dict[date_ts] = decision_obj
        order_list = account_dict["order_list"]
        nav_float = float(decision_obj.snapshot_metadata_dict["decision_nav_float"])
        live_dollar_dict = {asset_str: weight_float * nav_float for asset_str, weight_float in decision_obj.entry_target_weight_map_dict.items()}
        assert live_dollar_dict == pytest.approx(_entry_dollar_dict(order_list), abs=1e-6 * nav_float), date_ts
        assert set(decision_obj.exit_asset_set) == {order_dict["asset_str"] for order_dict in order_list if order_dict["target_bool"] and order_dict["amount_float"] == 0.0}, date_ts
        assert {asset_str: float(share_float) for asset_str, share_float in decision_obj.target_share_map_dict.items()} == {
            order_dict["asset_str"]: order_dict["amount_float"] for order_dict in order_list if order_dict["target_bool"] and order_dict["unit_str"] == "shares" and order_dict["amount_float"] > 0.0}, date_ts
        assert decision_obj.snapshot_metadata_dict["liquidity_reference_float"] == strategy_obj.liquidity_reference_ser.loc[date_ts], date_ts

    # The nightly check's real reference path: the entry point, its maps and the compare.
    support_dict = reference_compare.inspect_auto_reference_support_dict(release_obj)
    assert support_dict["supported_bool"], support_dict
    maps_dict = backtest_check.run_reference_maps_dict(release_obj, start_str=calendar_idx[0].date().isoformat(), end_str=calendar_idx[-1].date().isoformat(), capital_float=100_000.0, output_dir_path=tmp_path / "reference")
    buy_set = {key_tuple for key_tuple, trade_dict in maps_dict["transaction_map_dict"].items() if trade_dict["quantity_float"] > 0 and key_tuple[1] != "BIL"}
    assert buy_set and set(maps_dict["entry_weight_by_date_dict"]) == buy_set
    status_list = []
    for date_ts, decision_obj in decision_dict.items():
        if date_ts == calendar_idx[-1]:
            continue  # its fill session lies past the reference's end
        session_str = pricing_df.index[pricing_df.index.get_loc(date_ts) + 1].date().isoformat()
        comparison_dict = backtest_check.compare_decision_dict(
            {"decision_book_type_str": "incremental_entry_exit_book", "exit_asset_list": sorted(decision_obj.exit_asset_set),
             "entry_target_weight_map_dict": dict(decision_obj.snapshot_metadata_dict["liquidity_entry_wanted_weight_dict"])},
            reference_weight_map_dict={},
            reference_trade_map_dict={asset_str: float(trade_dict["quantity_float"]) for (trade_date_str, asset_str), trade_dict in maps_dict["transaction_map_dict"].items() if trade_date_str == session_str},
            reference_entry_weight_map_dict={asset_str: weight_float for (trade_date_str, asset_str), weight_float in maps_dict["entry_weight_by_date_dict"].items() if trade_date_str == session_str},
            skip_set=backtest_check.PARKING_ASSET_SET,  # the tighter 1/N tolerance on purpose: the same data must give the same m
        )
        status_list.append(comparison_dict["status_str"])
    assert status_list and set(status_list) == {"same"}, status_list
