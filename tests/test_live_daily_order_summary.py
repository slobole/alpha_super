"""Sparse completed-order snapshots cannot erase verified daily reporting."""
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
import json

import pytest

from alpha.live.daily_reporting import normalize_daily_order_summary_list
from alpha.live.models import BrokerOrderFill, BrokerOrderRecord, BrokerOrderRequest


SUBMIT_TS = datetime(2026, 10, 21, 13, 22, tzinfo=UTC)
FILL_TS = SUBMIT_TS + timedelta(minutes=8)
AS_OF_TS = FILL_TS + timedelta(minutes=1)


def summary_case(amount_float=3.0, order_id_str="broker-1", request_key_str="vplan:1:SPY:3"):
    request_obj = BrokerOrderRequest("release", "core5", "TEST", "vplan:1", request_key_str,
        "SPY", "MOO", "entry", "shares", amount_float, False, None, 770.0, 14000.0,
        decision_plan_id_int=1, vplan_id_int=2)
    known_obj = BrokerOrderRecord(order_id_str, 1, 2, "TEST", "SPY", request_key_str,
        "MOO", "shares", amount_float, 0.0, "Submitted", SUBMIT_TS,
        raw_payload_dict={"original_str": "initial_order"}, remaining_amount_float=abs(amount_float),
        last_status_timestamp_ts=SUBMIT_TS, submission_key_str="vplan:1")
    sparse_obj = replace(known_obj, amount_float=0.0, filled_amount_float=0.0,
        remaining_amount_float=0.0, avg_fill_price_float=None, status_str="Filled",
        raw_payload_dict={"order_id_int": 0})
    fill_obj = BrokerOrderFill(order_id_str, 1, 2, "TEST", "SPY", amount_float, 771.14,
        FILL_TS, raw_payload_dict={"exec_id_str": "execution-1"})
    ack_dict = {"broker_order_id_str": order_id_str, "decision_plan_id_int": 1,
        "vplan_id_int": 2, "account_route_str": "TEST", "asset_str": "SPY",
        "order_request_key_str": request_key_str, "broker_order_type_str": "MOO",
        "local_submit_ack_bool": True, "broker_response_ack_bool": True,
        "ack_status_str": "broker_acked", "response_timestamp_str": SUBMIT_TS.isoformat()}
    return request_obj, known_obj, sparse_obj, fill_obj, ack_dict


def known_row(known_obj):
    known_dict = asdict(known_obj)
    known_dict["submitted_timestamp_str"] = known_dict.pop("submitted_timestamp_ts").isoformat()
    status_ts = known_dict.pop("last_status_timestamp_ts")
    known_dict["last_status_timestamp_str"] = status_ts.isoformat() if status_ts else None
    known_dict["raw_payload_json_str"] = json.dumps(known_dict.pop("raw_payload_dict"))
    return known_dict


def normalize_case(*, amount_float=3.0, fill_list=None, known_obj=None, ack_list=None):
    request_obj, initial_obj, sparse_obj, fill_obj, ack_dict = summary_case(amount_float)
    return normalize_daily_order_summary_list([sparse_obj], [fill_obj] if fill_list is None else fill_list,
        [known_row(initial_obj if known_obj is None else known_obj)], [request_obj],
        [ack_dict] if ack_list is None else ack_list, as_of_ts=AS_OF_TS)[0]


@pytest.mark.parametrize("amount_float", [3.0, -3.0])
def test_sparse_daily_summary_uses_canonical_request_and_identified_executions(amount_float):
    result_obj = normalize_case(amount_float=amount_float)
    assert result_obj.amount_float == amount_float
    assert result_obj.filled_amount_float == abs(amount_float)
    assert result_obj.remaining_amount_float == 0.0
    assert result_obj.avg_fill_price_float == 771.14
    assert result_obj.status_str == "Filled"
    assert result_obj.last_status_timestamp_ts == FILL_TS
    assert result_obj.raw_payload_dict["order_summary_source_str"] == "canonical_request_and_executions"


def test_split_execution_summary_uses_quantity_weighted_price():
    _, _, _, fill_obj, _ = summary_case()
    fill_list = [replace(fill_obj, fill_amount_float=1.0, fill_price_float=770.0),
        replace(fill_obj, fill_amount_float=2.0, fill_price_float=772.0,
            fill_timestamp_ts=FILL_TS + timedelta(seconds=1), raw_payload_dict={"exec_id_str": "execution-2"})]
    result_obj = normalize_case(fill_list=fill_list)
    assert result_obj.filled_amount_float == 3.0
    assert result_obj.avg_fill_price_float == pytest.approx((770.0 + 2 * 772.0) / 3)
    assert result_obj.last_status_timestamp_ts == FILL_TS + timedelta(seconds=1)


def test_known_complete_summary_survives_sparse_refresh_without_executions():
    _, known_obj, _, _, _ = summary_case()
    known_obj = replace(known_obj, filled_amount_float=3.0, remaining_amount_float=0.0,
        avg_fill_price_float=771.14, status_str="Filled", last_status_timestamp_ts=FILL_TS)
    assert normalize_case(known_obj=known_obj, fill_list=[]) == known_obj


@pytest.mark.parametrize("missing_field_str", ["requested", "filled", "remaining", "price"])
def test_partial_completed_refresh_preserves_each_known_field(missing_field_str):
    request_obj, initial_obj, _, _, _ = summary_case()
    known_obj = replace(initial_obj, filled_amount_float=3.0, remaining_amount_float=0.0,
        avg_fill_price_float=771.14, status_str="Filled", last_status_timestamp_ts=FILL_TS)
    mutation_dict = {"requested": {"amount_float": 0.0}, "filled": {"filled_amount_float": 0.0},
        "remaining": {"remaining_amount_float": None}, "price": {"avg_fill_price_float": None}}[missing_field_str]
    record_obj = replace(known_obj, **mutation_dict)
    result_obj, = normalize_daily_order_summary_list([record_obj], [], [known_row(known_obj)],
        [request_obj], [], as_of_ts=AS_OF_TS)
    assert result_obj == known_obj


@pytest.mark.parametrize("remaining_float", [0.0, None])
def test_native_filled_quantities_without_average_keep_legacy_reporting(remaining_float):
    request_obj, known_obj, _, fill_obj, _ = summary_case()
    record_obj = replace(known_obj, filled_amount_float=3.0, status_str="Filled",
        remaining_amount_float=remaining_float)
    legacy_fill_obj = replace(fill_obj, raw_payload_dict={})
    result_obj, = normalize_daily_order_summary_list([record_obj], [legacy_fill_obj], [],
        [request_obj], [], as_of_ts=AS_OF_TS)
    assert result_obj is record_obj


@pytest.mark.parametrize("evidence_str,gap_str", [("no_ack", "no_broker_ack"), ("negative_ack", "no_broker_ack"),
    ("partial", "incomplete_executions"), ("no_execution_id", "incomplete_executions")])
def test_incomplete_proof_keeps_the_saved_summary_and_names_the_gap(evidence_str, gap_str):
    request_obj, known_obj, sparse_obj, fill_obj, ack_dict = summary_case()
    fill_list, ack_list = [fill_obj], [ack_dict]
    if evidence_str == "no_ack":
        ack_list = []
    elif evidence_str == "negative_ack":
        ack_list = [{**ack_dict, "broker_response_ack_bool": False}]
    elif evidence_str == "partial":
        fill_list = [replace(fill_obj, fill_amount_float=2.0)]
    else:
        fill_list = [replace(fill_obj, raw_payload_dict={})]
    initial_dict = known_row(known_obj)
    for _attempt_int in range(2):
        result_obj, = normalize_daily_order_summary_list([sparse_obj], fill_list,
            [initial_dict], [request_obj], ack_list, as_of_ts=AS_OF_TS)
        # The saved (sent) summary stays as it is; the refresh cannot zero it.
        assert (result_obj.amount_float, result_obj.filled_amount_float, result_obj.remaining_amount_float,
            result_obj.avg_fill_price_float, result_obj.status_str) == (3.0, 0.0, 3.0, None, "Submitted")
        assert result_obj.raw_payload_dict == {"original_str": "initial_order",
            "order_summary_source_str": "unproved_sparse_refresh", "order_summary_proof_gap_str": gap_str}
        assert initial_dict == known_row(known_obj)


@pytest.mark.parametrize("saved_kwargs_dict", [
    # remaining NULL on a Filled row (column added by migration), a partial
    # without a price, a Filled row short of its request: all "invalid" for the
    # repair path, all kept as saved when the refresh brings no proof.
    {"filled_amount_float": 3.0, "remaining_amount_float": None, "avg_fill_price_float": 771.14, "status_str": "Filled"},
    {"filled_amount_float": 1.0, "remaining_amount_float": 2.0, "avg_fill_price_float": None, "status_str": "Submitted"},
    {"filled_amount_float": 2.0, "remaining_amount_float": 1.0, "avg_fill_price_float": 771.14, "status_str": "Filled"},
])
def test_invalid_saved_row_is_kept_when_proof_is_missing(saved_kwargs_dict):
    request_obj, known_obj, sparse_obj, fill_obj, _ack_dict = summary_case()
    saved_obj = replace(known_obj, **saved_kwargs_dict)
    result_obj, = normalize_daily_order_summary_list([sparse_obj], [fill_obj],
        [known_row(saved_obj)], [request_obj], [], as_of_ts=AS_OF_TS)
    assert (result_obj.amount_float, result_obj.filled_amount_float, result_obj.remaining_amount_float,
        result_obj.avg_fill_price_float, result_obj.status_str) == (3.0, saved_obj.filled_amount_float,
        saved_obj.remaining_amount_float, saved_obj.avg_fill_price_float, saved_obj.status_str)
    assert result_obj.raw_payload_dict["order_summary_proof_gap_str"] == "no_broker_ack"


def test_unproven_sparse_record_without_known_summary_is_written_as_is_and_names_the_gap():
    request_obj, _, sparse_obj, fill_obj, _ = summary_case()
    result_obj, = normalize_daily_order_summary_list([sparse_obj], [fill_obj], [], [request_obj], [], as_of_ts=AS_OF_TS)
    assert (result_obj.amount_float, result_obj.filled_amount_float, result_obj.status_str) == (0.0, 0.0, "Filled")
    assert result_obj.raw_payload_dict["order_summary_proof_gap_str"] == "no_broker_ack"


@pytest.mark.parametrize("mutation_dict", [
    {"status_str": "Cancelled"}, {"status_str": "Submitted"},
    {"remaining_amount_float": 1.0},
    {"amount_float": 3.0, "filled_amount_float": 3.0, "avg_fill_price_float": 771.14},
])
def test_other_broker_records_are_unchanged(mutation_dict):
    request_obj, known_obj, sparse_obj, fill_obj, ack_dict = summary_case()
    record_obj = replace(sparse_obj, **mutation_dict)
    result_obj, = normalize_daily_order_summary_list([record_obj], [fill_obj],
        [known_row(known_obj)], [request_obj], [ack_dict], as_of_ts=AS_OF_TS)
    assert result_obj is record_obj


@pytest.mark.parametrize("mutation_dict", [
    {"account_route_str": "OTHER"}, {"asset_str": "QQQ"},
    {"decision_plan_id_int": 99}, {"vplan_id_int": 99},
    {"fill_amount_float": -3.0}, {"fill_amount_float": 4.0},
    {"fill_amount_float": 0.0}, {"fill_amount_float": float("nan")},
    {"fill_price_float": 0.0}, {"fill_price_float": float("inf")},
    {"fill_timestamp_ts": SUBMIT_TS - timedelta(seconds=1)},
    {"fill_timestamp_ts": FILL_TS.replace(tzinfo=None)},
])
def test_conflicting_execution_evidence_is_rejected(mutation_dict):
    _, _, _, fill_obj, _ = summary_case()
    with pytest.raises(ValueError):
        normalize_case(fill_list=[replace(fill_obj, **mutation_dict)])


def test_duplicate_execution_id_never_counts_twice():
    _, _, _, fill_obj, _ = summary_case()
    with pytest.raises(ValueError, match="execution"):
        normalize_case(fill_list=[fill_obj, fill_obj])


@pytest.mark.parametrize("source_str", ["known", "request", "ack"])
def test_conflicting_order_identity_cannot_replace_existing_summary(source_str):
    request_obj, known_obj, sparse_obj, fill_obj, ack_dict = summary_case()
    known_dict = known_row(known_obj)
    if source_str == "known":
        known_dict["account_route_str"] = "OTHER"
    elif source_str == "request":
        request_obj = replace(request_obj, account_route_str="OTHER")
    else:
        ack_dict["account_route_str"] = "OTHER"
    with pytest.raises(ValueError, match="identity"):
        normalize_daily_order_summary_list([sparse_obj], [fill_obj], [known_dict], [request_obj], [ack_dict], as_of_ts=AS_OF_TS)


def test_same_symbol_opposing_legs_use_exact_order_id_and_request_key():
    buy_tuple = summary_case()
    sell_tuple = summary_case(-2.0, "broker-2", "vplan:1:SPY:4")
    sell_fill_obj = replace(sell_tuple[3], raw_payload_dict={"exec_id_str": "execution-2"})
    result_list = normalize_daily_order_summary_list([buy_tuple[2], sell_tuple[2]],
        [buy_tuple[3], sell_fill_obj], [known_row(buy_tuple[1]), known_row(sell_tuple[1])],
        [buy_tuple[0], sell_tuple[0]], [buy_tuple[4], sell_tuple[4]], as_of_ts=AS_OF_TS)
    assert [(record_obj.amount_float, record_obj.filled_amount_float) for record_obj in result_list] == [(3.0, 3.0), (-2.0, 2.0)]


def test_unknown_canonical_request_never_infers_requested_amount_from_fills():
    _, known_obj, sparse_obj, fill_obj, ack_dict = summary_case()
    result_obj, = normalize_daily_order_summary_list([sparse_obj], [fill_obj],
        [known_row(known_obj)], [], [ack_dict], as_of_ts=AS_OF_TS)
    # The saved request quantity stays; nothing is inferred from the fills.
    assert (result_obj.amount_float, result_obj.filled_amount_float, result_obj.status_str) == (3.0, 0.0, "Submitted")
    assert result_obj.raw_payload_dict["order_summary_proof_gap_str"] == "no_canonical_request"


def test_already_sparse_known_summary_can_be_repaired_with_complete_proof():
    request_obj, _, sparse_obj, fill_obj, ack_dict = summary_case()
    result_obj, = normalize_daily_order_summary_list([sparse_obj], [fill_obj],
        [known_row(sparse_obj)], [request_obj], [ack_dict], as_of_ts=AS_OF_TS)
    assert result_obj.amount_float == result_obj.filled_amount_float == 3.0


def test_duplicate_execution_id_across_orders_is_conflicting_evidence():
    _, _, _, fill_obj, _ = summary_case()
    with pytest.raises(ValueError, match="execution"):
        normalize_case(fill_list=[fill_obj, replace(fill_obj, broker_order_id_str="other-order")])


@pytest.mark.parametrize("amount_float", [0.0, 3.0])
@pytest.mark.parametrize("filled_float", [0.0, 3.0])
@pytest.mark.parametrize("remaining_float", [0.0, None])
def test_completed_default_status_is_repaired_when_requested_quantity_is_supplied(amount_float, filled_float, remaining_float):
    request_obj, known_obj, sparse_obj, fill_obj, ack_dict = summary_case()
    record_obj = replace(sparse_obj, amount_float=amount_float, filled_amount_float=filled_float,
        remaining_amount_float=remaining_float)
    result_obj, = normalize_daily_order_summary_list([record_obj], [fill_obj],
        [known_row(known_obj)], [request_obj], [ack_dict], as_of_ts=AS_OF_TS)
    assert result_obj.amount_float == result_obj.filled_amount_float == 3.0
    assert result_obj.avg_fill_price_float == 771.14


@pytest.mark.parametrize("amount_float", [2.0, -3.0, float("inf"), float("nan")])
def test_conflicting_requested_quantity_is_not_silently_replaced(amount_float):
    request_obj, known_obj, sparse_obj, fill_obj, ack_dict = summary_case()
    with pytest.raises(ValueError, match="requested quantity"):
        normalize_daily_order_summary_list([replace(sparse_obj, amount_float=amount_float)], [fill_obj],
            [known_row(known_obj)], [request_obj], [ack_dict], as_of_ts=AS_OF_TS)


@pytest.mark.parametrize("mutation_dict", [
    {"response_timestamp_str": None}, {"response_timestamp_str": "invalid"},
    {"response_timestamp_str": (SUBMIT_TS - timedelta(seconds=1)).isoformat()},
    {"response_timestamp_str": (AS_OF_TS + timedelta(seconds=1)).isoformat()},
    {"response_timestamp_str": SUBMIT_TS.replace(tzinfo=None).isoformat()},
    {"broker_response_ack_bool": 2}, {"broker_response_ack_bool": "false"},
    {"local_submit_ack_bool": 2},
])
def test_invalid_ack_cannot_prove_empty_filled_summary(mutation_dict):
    _, _, _, _, ack_dict = summary_case()
    result_obj = normalize_case(ack_list=[{**ack_dict, **mutation_dict}])
    assert (result_obj.filled_amount_float, result_obj.avg_fill_price_float, result_obj.status_str) == (0.0, None, "Submitted")
    assert result_obj.raw_payload_dict["order_summary_proof_gap_str"] == "no_broker_ack"


def test_future_execution_cannot_repair_this_observation():
    _, _, _, fill_obj, _ = summary_case()
    with pytest.raises(ValueError, match="timestamps"):
        normalize_case(fill_list=[replace(fill_obj, fill_timestamp_ts=AS_OF_TS + timedelta(seconds=1))])


def test_prior_native_complete_signed_sale_is_preserved_without_ack_or_executions():
    request_obj, known_obj, sparse_obj, _, _ = summary_case(-3.0)
    known_obj = replace(known_obj, filled_amount_float=-3.0, remaining_amount_float=0.0,
        avg_fill_price_float=771.14, status_str="Filled", last_status_timestamp_ts=FILL_TS)
    result_obj, = normalize_daily_order_summary_list([sparse_obj], [],
        [known_row(known_obj)], [request_obj], [], as_of_ts=AS_OF_TS)
    assert result_obj == known_obj


def test_true_partial_cancelled_sale_is_unchanged():
    request_obj, known_obj, sparse_obj, _, ack_dict = summary_case(-3.0)
    partial_obj = replace(sparse_obj, status_str="Cancelled", amount_float=-3.0,
        filled_amount_float=-1.0, remaining_amount_float=2.0, avg_fill_price_float=771.14)
    result_obj, = normalize_daily_order_summary_list([partial_obj], [],
        [known_row(known_obj)], [request_obj], [ack_dict], as_of_ts=AS_OF_TS)
    assert result_obj is partial_obj
