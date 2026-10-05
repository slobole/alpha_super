"""Opening-auction dispatch evidence for CORE5 and capsules only."""
from datetime import timedelta

from alpha.live.models import SubmitBatchResult


def moo_dispatch_deadline_ts(vplan_obj):
    """XNYS 09:30 open minus two minutes; do not alter the broker's OPG TIF."""
    return vplan_obj.target_execution_timestamp_ts - timedelta(minutes=2)


def is_transient_broker_error_bool(exception_obj):
    if isinstance(exception_obj, (TimeoutError, ConnectionError)):
        return True
    if getattr(exception_obj, "errorCode", None) in {502, 504, 1100, 1101, 1102, 2110}:
        return True
    return isinstance(exception_obj, RuntimeError) and any(text_str in str(exception_obj).lower()
        for text_str in ("not connected", "connection lost", "socket disconnected", "timed out"))


class DispatchFailure(RuntimeError):
    """Preserve progress without treating a failed placeOrder as proof of no send."""

    def __init__(self, exception_obj, request_list, attempted_key_list, partial_result_obj):
        super().__init__(str(exception_obj) or type(exception_obj).__name__)
        self.error_type_str = type(exception_obj).__name__
        self.transient_bool = is_transient_broker_error_bool(exception_obj)
        self.attempted_key_list = list(attempted_key_list)
        self.never_dispatched_request_list = [request_obj for request_obj in request_list
            if request_obj.order_request_key_str not in attempted_key_list]
        self.partial_result_obj = partial_result_obj


def partial_submit_result_obj(progress_dict):
    return SubmitBatchResult(
        broker_order_record_list=list(progress_dict.get("record_list", [])),
        broker_order_event_list=list(progress_dict.get("event_list", [])),
        broker_order_fill_list=list(progress_dict.get("fill_list", [])),
        submit_ack_status_str="missing_critical",
    )
