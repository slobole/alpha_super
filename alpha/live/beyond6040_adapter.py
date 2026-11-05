"""Beyond 60/40 trend-tilt (T1) decision adapter: frozen Close_T shares and committed daily state.

The live host runs the strategy's own `compute_signals` over observed prices through Close_T and reads row T, as the
backtest's `iterate` does before its next-open fill. Shares come from the observed end-of-day account:
    NAV_T = cash + sum(held shares_i * Close_T_i),   target_i = trunc(NAV_T * w_i / Close_T_i)
No later quote or NAV changes them. Every session is sized this way (the backtest re-sizes at every open); orders go out
only where target_i differs from held_i. A signal that is not ready at Close_T (a missing base weight or tilt) blocks the
decision rather than guessing a book.
"""
from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime

import numpy as np
import pandas as pd
import exchange_calendars as exchange_calendar_module

from alpha.live import scheduler_utils
from alpha.live.models import DecisionPlan, LiveRelease, PodState
from data import norgate_snapshot_store as snapshot_module
from data.beyond6040_snapshot_contract import BEYOND6040_PROFILE_STR


BEYOND6040_STRATEGY_IMPORT_STR = "strategies.taa_beyond_6040.strategy_taa_beyond_6040_trend_tilt"
BEYOND6040_CONTRACT_STR = "beyond6040_close_target_shares_v1"
BEYOND6040_ASSET_TUPLE = ("VTI", "TLT", "GLD", "BIL")
BEYOND6040_RISK_ASSET_TUPLE = BEYOND6040_ASSET_TUPLE[:-1]
BEYOND6040_RESERVE_ASSET_STR = BEYOND6040_ASSET_TUPLE[-1]
BEYOND6040_STATE_VERSION_INT = 1
_WEIGHT_TOLERANCE_FLOAT = 1e-12


def validate_beyond6040_release(release_obj: LiveRelease) -> None:
    required_field_dict = {
        "data_profile_str": BEYOND6040_PROFILE_STR,
        "session_calendar_id_str": "XNYS",
        "signal_clock_str": "eod_snapshot_ready",
        "execution_policy_str": "next_open_moo",
        "pod_budget_fraction_float": 1.0,
    }
    for field_str, expected_value_obj in required_field_dict.items():
        if getattr(release_obj, field_str) != expected_value_obj:
            raise ValueError(f"Beyond 60/40 requires {field_str}={expected_value_obj!r}.")
    if set(release_obj.params_dict) - {"capital_base_float"}:
        raise ValueError("Beyond 60/40 uses the frozen default strategy parameters; overrides require separate parity qualification.")


def is_beyond6040_decision_bool(decision_plan_obj: DecisionPlan | None) -> bool:
    return decision_plan_obj is not None and decision_plan_obj.snapshot_metadata_dict.get("sizing_contract_str") == BEYOND6040_CONTRACT_STR


def _whole_position_map_dict(position_map_dict: dict[str, float]) -> dict[str, float]:
    result_dict = {}
    for asset_str, amount_float in position_map_dict.items():
        amount_float = float(amount_float)
        if not math.isfinite(amount_float) or amount_float != math.trunc(amount_float):
            raise ValueError("Beyond 60/40 requires finite whole-share account positions.")
        if amount_float == 0.0:
            continue
        if asset_str not in BEYOND6040_ASSET_TUPLE or amount_float < 0.0:
            raise ValueError(f"Beyond 60/40 found an unsupported account position: {asset_str}.")
        result_dict[str(asset_str)] = amount_float
    return result_dict


def require_beyond6040_position_match(expected_position_dict: dict[str, float], actual_position_dict: dict[str, float]) -> None:
    if _whole_position_map_dict(expected_position_dict) != _whole_position_map_dict(actual_position_dict):
        raise ValueError("Beyond 60/40 account positions changed after the frozen decision; reconcile before proceeding.")


def _validate_target_weight_dict(target_weight_dict: dict[str, float], reserve_tradable_bool: bool) -> None:
    if set(target_weight_dict) != set(BEYOND6040_ASSET_TUPLE) | {"Cash"}:
        raise ValueError("Beyond 60/40 requires the complete target weight book.")
    if any(not math.isfinite(float(weight_float)) for weight_float in target_weight_dict.values()):
        raise ValueError("Beyond 60/40 target weights must be finite.")
    if any(float(weight_float) < -_WEIGHT_TOLERANCE_FLOAT for weight_float in target_weight_dict.values()):
        raise ValueError("Beyond 60/40 is long only; no target weight may be negative.")
    risky_sum_float = 0.0
    for asset_str in BEYOND6040_RISK_ASSET_TUPLE:
        weight_float = float(target_weight_dict[asset_str])
        if weight_float > 1.0 + _WEIGHT_TOLERANCE_FLOAT:
            raise ValueError("Beyond 60/40 risky weights must lie in [0, 1].")
        risky_sum_float += weight_float
    if risky_sum_float > 1.0 + _WEIGHT_TOLERANCE_FLOAT:
        raise ValueError("Beyond 60/40 risky weights must sum to at most 1.")
    residual_float = 1.0 - risky_sum_float
    expected_reserve_float = residual_float if reserve_tradable_bool else 0.0
    if abs(float(target_weight_dict[BEYOND6040_RESERVE_ASSET_STR]) - expected_reserve_float) > _WEIGHT_TOLERANCE_FLOAT:
        raise ValueError("Beyond 60/40 BIL must carry the residual once it trades, and nothing before.")
    if abs(float(target_weight_dict["Cash"]) - (residual_float - expected_reserve_float)) > _WEIGHT_TOLERANCE_FLOAT:
        raise ValueError("Beyond 60/40 cash must equal the residual not parked in BIL.")


def _build_strategy_obj():
    from strategies.taa_beyond_6040 import strategy_taa_beyond_6040_trend_tilt as tilt_module

    config_obj = tilt_module.DEFAULT_CONFIG
    return tilt_module.Beyond6040TrendTiltStrategy(
        name=tilt_module.STRATEGY_NAME_STR,
        trend_sma_lookback_int=config_obj.trend_sma_lookback_int,
        tilt_strength_float=config_obj.tilt_strength_float,
        trend_price_field_str=config_obj.trend_price_field_str,
        reserve_asset_str=config_obj.reserve_asset_str,
        benchmarks=config_obj.benchmark_list,
        asset_list=config_obj.asset_list,
        asset_vol_lookback_int=config_obj.asset_vol_lookback_int,
        portfolio_vol_lookback_int=config_obj.portfolio_vol_lookback_int,
        target_portfolio_vol_float=config_obj.target_portfolio_vol_float,
        trigger_portfolio_vol_float=config_obj.trigger_portfolio_vol_float,
        capital_base=config_obj.capital_base_float,
        slippage=config_obj.slippage_float,
        commission_per_share=config_obj.commission_per_share_float,
        commission_minimum=config_obj.commission_minimum_float,
    )


def _month_end_label_ts(decision_ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(decision_ts).to_period("M").end_time.normalize()


def _monthly_book_at_close(strategy_obj, pricing_data_df: pd.DataFrame, signal_df: pd.DataFrame, decision_ts: pd.Timestamp, month_end_bool: bool) -> tuple[pd.Series, pd.Series]:
    """Base weights and tilt multipliers in force after Close_T, exactly as the engine reads them from row T.

    *** CRITICAL *** The engine attaches a month-end decision to the last session of its month and reads it there before
    the next month's first open. On a history cut at T that schedule needs the next month's first session, which does not
    exist yet, so for T = last XNYS session of its month the month-end rows are read directly (as compute_signals forms
    them); a month-end row the engine would drop (any NaN) leaves the older schedule in force, as its ffill does. Every
    other row of the cut history equals the full history's.
    """
    from strategies.taa_beyond_6040 import strategy_taa_beyond_6040 as base_module
    from strategies.taa_beyond_6040 import strategy_taa_beyond_6040_trend_tilt as tilt_module

    asset_list = list(strategy_obj.asset_list)
    row_ser = signal_df.loc[decision_ts]
    base_weight_ser = strategy_obj._current_base_weight_ser(row_ser).astype(float)
    tilt_multiplier_ser = strategy_obj._current_tilt_multiplier_ser(row_ser).astype(float)
    if not month_end_bool:
        return base_weight_ser, tilt_multiplier_ser
    history_df = pricing_data_df.loc[:decision_ts]
    label_ts = _month_end_label_ts(decision_ts)
    _, _, month_end_weight_df = base_module.compute_month_end_inverse_vol_weight_df(
        price_close_df=history_df.loc[:, [(asset_str, "Close") for asset_str in asset_list]].astype(float),
        asset_vol_lookback_int=strategy_obj.asset_vol_lookback_int,
    )
    if label_ts in month_end_weight_df.index:
        base_weight_ser = month_end_weight_df.loc[label_ts].reindex(asset_list).astype(float)
    _, _, month_end_multiplier_df = tilt_module.compute_month_end_tilt_multiplier_df(
        trend_price_df=history_df.loc[:, [(asset_str, strategy_obj.trend_price_field_str) for asset_str in asset_list]].astype(float),
        trend_sma_lookback_int=strategy_obj.trend_sma_lookback_int,
        tilt_strength_float=strategy_obj.tilt_strength_float,
    )
    if label_ts in month_end_multiplier_df.index:
        tilt_multiplier_ser = month_end_multiplier_df.loc[label_ts].reindex(asset_list).astype(float)
    return base_weight_ser, tilt_multiplier_ser


def _weight_book_at_close(strategy_obj, pricing_data_df: pd.DataFrame, signal_df: pd.DataFrame, decision_ts: pd.Timestamp) -> dict:
    """The complete target weight book after Close_T: exposure x base x tilt per risk asset, residual to BIL or cash."""
    from strategies.taa_beyond_6040 import strategy_taa_beyond_6040_trend_tilt as tilt_module

    month_end_bool = scheduler_utils.is_last_session_of_month_bool(decision_ts, "XNYS")
    base_weight_ser, tilt_multiplier_ser = _monthly_book_at_close(strategy_obj, pricing_data_df, signal_df, decision_ts, month_end_bool)
    if base_weight_ser.isna().any() or tilt_multiplier_ser.isna().any():
        raise ValueError(f"Beyond 60/40 signal is not ready at Close_{decision_ts.date().isoformat()}; no decision.")
    # *** CRITICAL *** The overlay reads the base-weight sleeve returns through T only (the engine's data_df ends at
    # previous_bar). Row T of the cut history equals the full history's: it uses the weights in force at T-1.
    gross_exposure_float = strategy_obj._sleeve_gross_exposure_float(signal_df.loc[:decision_ts])
    risky_weight_ser = (base_weight_ser * tilt_multiplier_ser * gross_exposure_float).astype(float)
    if (risky_weight_ser < -_WEIGHT_TOLERANCE_FLOAT).any() or float(risky_weight_ser.sum()) > 1.0 + _WEIGHT_TOLERANCE_FLOAT:
        raise ValueError("Beyond 60/40 tilted weights must lie in [0, 1] and sum to at most 1.")
    reserve_tradable_bool = strategy_obj._reserve_tradable_bool(signal_df.loc[decision_ts])
    residual_float = float(1.0 - risky_weight_ser.sum())
    reserve_weight_float = residual_float if reserve_tradable_bool else 0.0
    target_weight_dict = {asset_str: float(risky_weight_ser[asset_str]) for asset_str in BEYOND6040_RISK_ASSET_TUPLE}
    target_weight_dict[BEYOND6040_RESERVE_ASSET_STR] = reserve_weight_float
    target_weight_dict["Cash"] = float(residual_float - reserve_weight_float)
    _validate_target_weight_dict(target_weight_dict, reserve_tradable_bool)
    return {
        "target_weight_dict": target_weight_dict,
        "base_weight_dict": {asset_str: float(base_weight_ser[asset_str]) for asset_str in BEYOND6040_RISK_ASSET_TUPLE},
        "tilt_multiplier_dict": {asset_str: float(tilt_multiplier_ser[asset_str]) for asset_str in BEYOND6040_RISK_ASSET_TUPLE},
        "gross_exposure_float": float(gross_exposure_float),
        "reserve_tradable_bool": bool(reserve_tradable_bool),
        "month_end_bool": bool(month_end_bool),
        "sleeve_label_str": tilt_module.SLEEVE_LABEL_STR,
    }


def _last_monthly_change_date_str(strategy_obj, signal_df: pd.DataFrame, decision_ts: pd.Timestamp, book_dict: dict) -> str:
    """The last session at or before T whose month-end decision changed the base x tilt book (the monthly rebalance)."""
    asset_list = list(strategy_obj.asset_list)
    pair_df = pd.DataFrame({
        asset_str: signal_df[(asset_str, "base_weight_ser")].astype(float) * signal_df[(asset_str, "tilt_multiplier_ser")].astype(float)
        for asset_str in asset_list
    }).loc[:decision_ts]
    pair_df.loc[decision_ts] = [book_dict["base_weight_dict"][asset_str] * book_dict["tilt_multiplier_dict"][asset_str] for asset_str in asset_list]
    valid_df = pair_df.dropna(how="any")
    if valid_df.empty:
        raise ValueError("Beyond 60/40 has no month-end decision in its history.")
    changed_ser = valid_df.ne(valid_df.shift()).any(axis=1)
    changed_ser.iloc[0] = True
    return pd.Timestamp(valid_df.index[changed_ser.to_numpy()][-1]).date().isoformat()


def _replay_beyond6040_state_dict(strategy_obj, pricing_data_df: pd.DataFrame, signal_df: pd.DataFrame, cached_state_dict: dict | None, signal_date_ts: pd.Timestamp, previous_session_ts: pd.Timestamp) -> tuple[dict, dict, dict]:
    """Signal memory from prices only; never invent account cash, positions or fills.

    The rule has no path dependence: the book at T is a function of observed prices through T. The replay recomputes it,
    recomputes yesterday's book for the data-revision check, and checks that the decision history has every exchange
    session.
    """
    exchange_calendar_obj = exchange_calendar_module.get_calendar(
        "XNYS", start=pricing_data_df.index[0], end=pricing_data_df.index[-1] + pd.Timedelta(days=10))
    expected_session_idx = exchange_calendar_obj.sessions_in_range(pricing_data_df.index[0], pricing_data_df.index[-1])
    if not pd.DatetimeIndex(pricing_data_df.index).equals(pd.DatetimeIndex(expected_session_idx)):
        raise ValueError("Beyond 60/40 replay requires every exchange session in its price history.")
    book_dict = _weight_book_at_close(strategy_obj, pricing_data_df, signal_df, signal_date_ts)
    last_rebalance_date_str = _last_monthly_change_date_str(strategy_obj, signal_df, signal_date_ts, book_dict)
    revision_warning_bool = False
    if cached_state_dict and cached_state_dict.get("last_signal_date_str") == previous_session_ts.date().isoformat():
        try:
            previous_book_dict = _weight_book_at_close(strategy_obj, pricing_data_df, signal_df, previous_session_ts)
            cached_weight_dict = cached_state_dict.get("last_target_weight_map_dict") or {}
            revision_warning_bool = set(cached_weight_dict) != set(previous_book_dict["target_weight_dict"]) or any(
                abs(float(cached_weight_dict[asset_str]) - float(weight_float)) > _WEIGHT_TOLERANCE_FLOAT
                for asset_str, weight_float in previous_book_dict["target_weight_dict"].items()
            )
        except (ValueError, TypeError, KeyError):
            revision_warning_bool = True
    replay_state_dict = {
        "beyond6040_state_version_int": BEYOND6040_STATE_VERSION_INT,
        "last_signal_date_str": signal_date_ts.date().isoformat(),
        "last_target_weight_map_dict": dict(book_dict["target_weight_dict"]),
        "last_rebalance_date_str": last_rebalance_date_str,
    }
    return replay_state_dict, book_dict, {
        "beyond6040_replay_start_date_str": pd.Timestamp(pricing_data_df.index[0]).date().isoformat(),
        "beyond6040_replay_session_count_int": int(len(pricing_data_df.index)),
        "beyond6040_data_revision_warning_bool": bool(revision_warning_bool),
    }


def build_beyond6040_decision_from_prices(
    release_obj: LiveRelease,
    as_of_ts: datetime,
    pod_state_obj: PodState,
    pricing_data_df: pd.DataFrame,
    snapshot_metadata_dict: dict[str, object],
) -> DecisionPlan:
    validate_beyond6040_release(release_obj)
    if release_obj.mode_str == "live" and not release_obj.enabled_bool:
        raise ValueError("Beyond 60/40 disabled LIVE templates cannot build operational decisions.")
    signal_date_ts = scheduler_utils.get_latest_completed_session_label_ts(as_of_ts, "XNYS")
    if signal_date_ts is None:
        raise ValueError("Beyond 60/40 has no completed signal session.")
    signal_date_str = signal_date_ts.date().isoformat()
    previous_session_ts = scheduler_utils.get_exchange_calendar_obj("XNYS").previous_session(signal_date_ts)
    if (
        snapshot_metadata_dict.get("norgate_snapshot_date_str") != signal_date_str
        or snapshot_metadata_dict.get("norgate_data_profile_str") != BEYOND6040_PROFILE_STR
        or not snapshot_metadata_dict.get("norgate_manifest_hash_str")
    ):
        raise ValueError("Beyond 60/40 requires the exact decision-session snapshot and manifest identity.")
    # *** CRITICAL*** Truncate before computing features. Nothing after Close_T may enter the signal or the NAV mark.
    pricing_data_df = pricing_data_df.loc[:signal_date_ts].copy()
    if (
        pricing_data_df.index.has_duplicates
        or not pricing_data_df.index.is_monotonic_increasing
        or signal_date_ts not in pricing_data_df.index
        or previous_session_ts not in pricing_data_df.index
    ):
        raise ValueError("Beyond 60/40 requires unique ordered prices for T and its prior exchange session.")
    if pod_state_obj is None or (
        pod_state_obj.pod_id_str != release_obj.pod_id_str
        or pod_state_obj.account_route_str != release_obj.account_route_str
        or pod_state_obj.user_id_str != release_obj.user_id_str
    ):
        raise ValueError("Beyond 60/40 requires its own saved EOD account state.")
    state_timestamp_ts = scheduler_utils.to_market_timestamp_ts(pod_state_obj.updated_timestamp_ts, "XNYS")
    close_timestamp_ts = scheduler_utils.get_session_close_timestamp_ts(signal_date_ts, "XNYS")
    expected_source_str = "virtual_broker" if release_obj.mode_str == "incubation" else "broker"
    if (
        pod_state_obj.snapshot_stage_str != "eod"
        or pod_state_obj.snapshot_source_str != expected_source_str
        or state_timestamp_ts.date() != signal_date_ts.date()
        or state_timestamp_ts < close_timestamp_ts
        or state_timestamp_ts > scheduler_utils.to_market_timestamp_ts(as_of_ts, "XNYS")
    ):
        raise ValueError("Beyond 60/40 requires a trusted EOD account snapshot from the signal session; bootstrap/default or stale NAV is insufficient.")
    position_map_dict = _whole_position_map_dict(pod_state_obj.position_amount_map)
    prior_state_dict = dict(pod_state_obj.strategy_state_dict)

    strategy_obj = _build_strategy_obj()
    signal_df = strategy_obj.compute_signals(pricing_data_df)
    replay_state_dict, book_dict, replay_metadata_dict = _replay_beyond6040_state_dict(
        strategy_obj, pricing_data_df, signal_df, prior_state_dict, signal_date_ts, previous_session_ts,
    )
    target_weight_dict = dict(replay_state_dict["last_target_weight_map_dict"])
    reserve_tradable_bool = bool(book_dict["reserve_tradable_bool"])
    close_row_ser = signal_df.loc[signal_date_ts]
    close_price_dict: dict[str, float | None] = {}
    for asset_str in BEYOND6040_ASSET_TUPLE:
        close_float = float(close_row_ser.get((asset_str, "Close"), np.nan))
        if asset_str == BEYOND6040_RESERVE_ASSET_STR and not reserve_tradable_bool:
            close_price_dict[asset_str] = None
            continue
        if not math.isfinite(close_float) or close_float <= 0.0:
            raise ValueError(f"Beyond 60/40 Close_T price for {asset_str} must be finite and positive.")
        close_price_dict[asset_str] = close_float
    if any(close_price_dict.get(asset_str) is None for asset_str in position_map_dict):
        raise ValueError("Beyond 60/40 holds an asset without an observed Close_T; the NAV mark is impossible.")
    # NAV_Close_T = EOD cash + sum(shares_i * Close_T_i). Long only: nothing is subtracted twice.
    close_nav_float = float(pod_state_obj.cash_float) + sum(
        amount_float * close_price_dict[asset_str] for asset_str, amount_float in position_map_dict.items()
    )
    if not math.isfinite(close_nav_float) or close_nav_float <= 0.0:
        raise ValueError("Beyond 60/40 Close_T NAV must be finite and positive.")

    prior_receipt_dict = prior_state_dict.get("beyond6040_execution_receipt_dict", {})
    receipt_invalid_bool = False
    receipt_matches_bool = False
    # Only a pod that has never decided initializes (Flow Vote rule). After a decision, an empty receipt means its first
    # fill did not complete; today's sizing from NAV catches it up, a partly filled first event included.
    initialization_bool = not prior_receipt_dict and not prior_state_dict.get("last_signal_date_str")
    if initialization_bool and position_map_dict:
        # *** CRITICAL*** As the Flow Vote: shares held before any decision have no known origin; never size over them.
        raise ValueError("Beyond 60/40 found holdings without a decision; a reviewed start is required.")
    if prior_receipt_dict:
        try:
            receipt_share_dict = prior_receipt_dict["last_applied_target_share_map_dict"]
            if not isinstance(receipt_share_dict, dict):
                raise TypeError("receipt share map must be a mapping")
            receipt_matches_bool = (
                set(receipt_share_dict) == set(BEYOND6040_ASSET_TUPLE)
                and _whole_position_map_dict(receipt_share_dict) == position_map_dict
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            receipt_invalid_bool = True
    # *** CRITICAL *** Every session is sized from NAV_T and Close_T, as the backtest re-sizes at every open; there is no
    # hold state. The receipt records the last fully applied book; it never suppresses today's sizing.
    rebalance_bool = True
    catch_up_bool = (not initialization_bool) and (not receipt_matches_bool)
    # *** CRITICAL*** q_i = trunc(NAV_Close_T * w_i / Close_T_i). Next-open prices cannot resize q_i.
    target_share_dict = {}
    for asset_str in BEYOND6040_ASSET_TUPLE:
        weight_float = float(target_weight_dict[asset_str])
        if close_price_dict[asset_str] is None:
            if weight_float != 0.0:
                raise ValueError("Beyond 60/40 cannot size a weight without an observed Close_T.")
            target_share_dict[asset_str] = 0.0
            continue
        target_share_dict[asset_str] = float(int(close_nav_float * weight_float / close_price_dict[asset_str]))
    _validate_target_weight_dict(target_weight_dict, reserve_tradable_bool)
    no_order_bool = all(target_share_dict[asset_str] == position_map_dict.get(asset_str, 0.0) for asset_str in BEYOND6040_ASSET_TUPLE)
    next_state_dict = {
        **replay_state_dict,
        "beyond6040_execution_receipt_dict": prior_receipt_dict if isinstance(prior_receipt_dict, dict) else {},
    }
    candidate_receipt_dict = {
        "last_applied_rebalance_date_str": replay_state_dict["last_rebalance_date_str"],
        "last_applied_signal_date_str": signal_date_str,
        "last_applied_target_weight_map_dict": dict(target_weight_dict),
        "last_applied_target_share_map_dict": dict(target_share_dict),
    }
    warning_list = []
    if prior_state_dict and prior_state_dict.get("last_signal_date_str") != previous_session_ts.date().isoformat():
        warning_list.append("beyond6040_strategy_state_cache_gap")
    if replay_metadata_dict["beyond6040_data_revision_warning_bool"]:
        warning_list.append("beyond6040_strategy_state_cache_revised")
    if receipt_invalid_bool:
        warning_list.append("beyond6040_execution_receipt_invalid")
    metadata_dict = dict(snapshot_metadata_dict)
    metadata_dict.update({
        "strategy_family_str": "beyond6040_trend_tilt",
        "sizing_contract_str": BEYOND6040_CONTRACT_STR,
        "fixed_target_share_map_dict": target_share_dict,
        "sizing_close_price_map_dict": close_price_dict,
        "sizing_close_nav_float": close_nav_float,
        "sizing_account_timestamp_str": pod_state_obj.updated_timestamp_ts.isoformat(),
        "sizing_account_cash_float": float(pod_state_obj.cash_float),
        "base_strategy_state_dict": prior_state_dict,
        **replay_metadata_dict,
        "beyond6040_state_basis_str": "price_history_replay_through_close_t",
        "beyond6040_base_weight_map_dict": book_dict["base_weight_dict"],
        "beyond6040_tilt_multiplier_map_dict": book_dict["tilt_multiplier_dict"],
        "beyond6040_gross_exposure_float": book_dict["gross_exposure_float"],
        "beyond6040_reserve_tradable_bool": reserve_tradable_bool,
        "beyond6040_warning_bool": bool(warning_list),
        "beyond6040_warning_code_list": warning_list,
        "beyond6040_catch_up_bool": bool(catch_up_bool),
        # Proposed application only. The daily finalizer commits this receipt after refreshed actual positions equal the
        # complete frozen share book.
        "beyond6040_candidate_execution_receipt_dict": candidate_receipt_dict,
        "rebalance_bool": bool(rebalance_bool),
        "no_order_bool": bool(no_order_bool),
        "initialization_bool": bool(initialization_bool),
        "month_end_bool": bool(book_dict["month_end_bool"]),
        "borrow_accounting_str": "observed_account_cash_only_no_synthetic_research_fee",
    })
    return DecisionPlan(
        release_id_str=release_obj.release_id_str, user_id_str=release_obj.user_id_str,
        pod_id_str=release_obj.pod_id_str, account_route_str=release_obj.account_route_str,
        signal_timestamp_ts=scheduler_utils.build_signal_timestamp_ts(signal_date_ts, release_obj),
        submission_timestamp_ts=scheduler_utils.build_submission_timestamp_ts(signal_date_ts, release_obj),
        target_execution_timestamp_ts=scheduler_utils.build_target_execution_timestamp_ts(signal_date_ts, release_obj),
        execution_policy_str=release_obj.execution_policy_str,
        decision_base_position_map=position_map_dict,
        snapshot_metadata_dict=metadata_dict, strategy_state_dict=next_state_dict,
        decision_book_type_str="full_target_weight_book",
        full_target_weight_map_dict={asset_str: target_weight_dict[asset_str] for asset_str in BEYOND6040_ASSET_TUPLE},
        cash_reserve_weight_float=float(target_weight_dict["Cash"]),
        preserve_untouched_positions_bool=not rebalance_bool,
        rebalance_omitted_assets_to_zero_bool=rebalance_bool,
    )


def _load_beyond6040_decision_prices(release_obj: LiveRelease, as_of_ts: datetime) -> tuple[pd.DataFrame, dict]:
    from strategies.taa_beyond_6040 import strategy_taa_beyond_6040_trend_tilt as tilt_module

    validate_beyond6040_release(release_obj)
    if not snapshot_module.is_snapshot_mode_enabled_bool():
        raise ValueError("Beyond 60/40 operational decisions require the qualified local snapshot profile.")
    signal_date_ts = scheduler_utils.get_latest_completed_session_label_ts(as_of_ts, "XNYS")
    if signal_date_ts is None:
        raise ValueError("Beyond 60/40 has no completed signal session.")
    initial_manifest_obj = snapshot_module.load_valid_snapshot_manifest(
        BEYOND6040_PROFILE_STR, minimum_snapshot_date_str=signal_date_ts.date().isoformat(),
    )
    pricing_data_df = tilt_module.get_trend_tilt_data(replace(tilt_module.DEFAULT_CONFIG, end_date_str=signal_date_ts.date().isoformat()))
    metadata_dict = snapshot_module.build_data_source_metadata_dict(BEYOND6040_PROFILE_STR)
    if metadata_dict.get("norgate_manifest_hash_str") != initial_manifest_obj.manifest_hash_str:
        raise ValueError("Beyond 60/40 snapshot changed while preparing decision metadata.")
    return pricing_data_df, metadata_dict


def build_beyond6040_decision_plan(release_obj: LiveRelease, as_of_ts: datetime, pod_state_obj: PodState | None) -> DecisionPlan:
    pricing_data_df, metadata_dict = _load_beyond6040_decision_prices(release_obj, as_of_ts)
    return build_beyond6040_decision_from_prices(release_obj, as_of_ts, pod_state_obj, pricing_data_df, metadata_dict)


def validated_beyond6040_target_share_dict(decision_plan_obj: DecisionPlan, release_obj: LiveRelease, actual_position_dict: dict[str, float]) -> dict[str, float]:
    validate_beyond6040_release(release_obj)
    if not is_beyond6040_decision_bool(decision_plan_obj):
        raise ValueError("Beyond 60/40 decision lacks its frozen share contract.")
    if (decision_plan_obj.release_id_str, decision_plan_obj.pod_id_str, decision_plan_obj.account_route_str) != (
        release_obj.release_id_str, release_obj.pod_id_str, release_obj.account_route_str,
    ):
        raise ValueError("Beyond 60/40 decision identity differs from its release.")
    require_beyond6040_position_match(decision_plan_obj.decision_base_position_map, actual_position_dict)
    metadata_dict = decision_plan_obj.snapshot_metadata_dict
    target_share_dict = dict(metadata_dict["fixed_target_share_map_dict"])
    close_price_dict = dict(metadata_dict["sizing_close_price_map_dict"])
    close_nav_float = float(metadata_dict["sizing_close_nav_float"])
    if set(target_share_dict) != set(BEYOND6040_ASSET_TUPLE) or set(close_price_dict) != set(BEYOND6040_ASSET_TUPLE):
        raise ValueError("Beyond 60/40 frozen share/price book is incomplete.")
    reserve_tradable_bool = close_price_dict[BEYOND6040_RESERVE_ASSET_STR] is not None
    _validate_target_weight_dict({
        **{asset_str: decision_plan_obj.full_target_weight_map_dict.get(asset_str, 0.0) for asset_str in BEYOND6040_ASSET_TUPLE},
        "Cash": decision_plan_obj.cash_reserve_weight_float,
    }, reserve_tradable_bool)
    _whole_position_map_dict(target_share_dict)
    if not math.isfinite(close_nav_float) or close_nav_float <= 0.0:
        raise ValueError("Beyond 60/40 frozen NAV is invalid.")
    for asset_str in BEYOND6040_ASSET_TUPLE:
        weight_float = float(decision_plan_obj.full_target_weight_map_dict.get(asset_str, 0.0))
        if close_price_dict[asset_str] is None:
            if asset_str != BEYOND6040_RESERVE_ASSET_STR or weight_float != 0.0 or target_share_dict[asset_str] != 0.0:
                raise ValueError("Beyond 60/40 frozen sizing close is invalid.")
            continue
        close_float = float(close_price_dict[asset_str])
        if not math.isfinite(close_float) or close_float <= 0.0:
            raise ValueError("Beyond 60/40 frozen sizing close is invalid.")
        expected_share_float = (
            float(int(close_nav_float * weight_float / close_float))
            if metadata_dict["rebalance_bool"] else float(actual_position_dict.get(asset_str, 0.0))
        )
        if target_share_dict[asset_str] != expected_share_float:
            raise ValueError("Beyond 60/40 frozen shares disagree with their Close_T sizing evidence.")
    return {asset_str: float(amount_float) for asset_str, amount_float in target_share_dict.items()}
