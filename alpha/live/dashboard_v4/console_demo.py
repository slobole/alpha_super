"""Synthetic console lines for the demo: an owned temporary operator log.

Lines are rendered by the serve's own operator-line renderer, so the demo
exercises the real Console reader. Nothing here reads workstation logs.
"""

import atexit
from datetime import timedelta
from pathlib import Path
import shutil
import tempfile
import threading

from alpha.live.logging_utils import render_operator_message_str


DEMO_ACCOUNT_TUPLE = ("DU1234561", "DU1234562", "DU1234563", "DU1234564")


def _line_str(level_str, action_str, timestamp_ts, field_dict=None):
    return render_operator_message_str(level_str, action_str, timestamp_ts, field_dict)


def build_demo_console_line_list(pod_id_list, *, as_of_ts):
    """About six hours of a calm day, one missing-ACK Pod, and the shared Norgate noise."""
    line_list = []
    start_ts = as_of_ts - timedelta(hours=6)
    for index_int, pod_id_str in enumerate(pod_id_list):
        account_str = DEMO_ACCOUNT_TUPLE[index_int % len(DEMO_ACCOUNT_TUPLE)]
        base_dict = {"pod": pod_id_str, "account": account_str}
        line_list.append((start_ts + timedelta(seconds=index_int), _line_str("INFO", "service.start", start_ts + timedelta(seconds=index_int),
            {**base_dict, "mode": "live", "idle_max_sleep": "3600s"})))
        for hour_int in range(1, 4):
            wait_ts = start_ts + timedelta(hours=hour_int, seconds=index_int)
            line_list.append((wait_ts, _line_str("INFO", "cycle.wait", wait_ts, {**base_dict, "decision": "completed",
                "vplan": "completed", "reason": "waiting for the next session", "next": (wait_ts + timedelta(hours=1)).strftime("%H:%M UTC")})))
        open_ts = as_of_ts.replace(hour=13, minute=20, second=0, microsecond=0) + timedelta(seconds=index_int)
        if index_int < 2:
            line_list.extend([
                (open_ts, _line_str("INFO", "broker_connect.start", open_ts, {**base_dict, "client_id": 31 + index_int})),
                (open_ts + timedelta(seconds=2), _line_str("INFO", "broker_connect.ok", open_ts + timedelta(seconds=2), base_dict)),
                (open_ts + timedelta(seconds=90), _line_str("INFO", "build_vplan.ok", open_ts + timedelta(seconds=90),
                    {**base_dict, "vplan": 2, "orders": 3})),
                (open_ts + timedelta(minutes=3, seconds=30), _line_str("INFO", "submit_vplan.start", open_ts + timedelta(minutes=3, seconds=30),
                    {**base_dict, "vplan": 2, "orders": 3})),
            ])
        if index_int == 0:
            fill_ts = as_of_ts.replace(hour=13, minute=30, second=2, microsecond=0)
            line_list.extend([
                (fill_ts, _line_str("INFO", "fill.ok", fill_ts, {**base_dict, "vplan": 2, "filled": "3/3"})),
                (fill_ts + timedelta(minutes=6), _line_str("INFO", "reconcile.ok", fill_ts + timedelta(minutes=6),
                    {**base_dict, "vplan": 2, "diffs": 0})),
            ])
        if index_int == 1:
            ack_ts = as_of_ts.replace(hour=13, minute=24, second=40, microsecond=0)
            line_list.append((ack_ts, _line_str("WARN", "submit_vplan.ack_missing", ack_ts,
                {**base_dict, "vplan": 2, "missing": "1/3", "reason": "broker did not acknowledge all submitted orders"})))
            for retry_int in range(8):
                retry_ts = ack_ts + timedelta(minutes=2, seconds=60 * retry_int)
                line_list.append((retry_ts, _line_str("ERROR", "post_execution_reconcile.fail", retry_ts,
                    {**base_dict, "vplan": 2, "reason": "missing broker ACK for MSFT; check the broker connection", "retry_in": "60s"})))
    for second_int in range(0, 6 * 3600, 20):
        noise_ts = start_ts + timedelta(seconds=second_int)
        line_list.append((noise_ts, _line_str("INFO", "norgate.sync.skipped", noise_ts,
            {"status": "direct", "dates": "{}", "reason": "direct_norgate_mode"})))
    line_list.sort(key=lambda item_tuple: item_tuple[0])
    return [line_str for _, line_str in line_list]


def attach_demo_console(provider_obj, *, as_of_ts):
    """Give the demo provider its own operator log and Pod list; removed at exit."""
    temp_dir_str = tempfile.mkdtemp(prefix="dashboard_v4_console_demo_")
    atexit.register(shutil.rmtree, temp_dir_str, True)
    log_path_obj = Path(temp_dir_str) / "live_operator.log"
    pod_list = [{"pod_id_str": row_dict["pod_id_str"], "mode_str": "live",
        "name_str": row_dict.get("strategy_name_str") or row_dict["pod_id_str"]} for row_dict in provider_obj.row_list]
    line_list = build_demo_console_line_list([item_dict["pod_id_str"] for item_dict in pod_list], as_of_ts=as_of_ts)
    log_path_obj.write_text("\n".join(line_list) + "\n", encoding="utf-8")
    provider_obj.console_log_path_str = str(log_path_obj)
    provider_obj.get_console_pod_list = lambda: [dict(item_dict) for item_dict in pod_list]
    return log_path_obj


def start_demo_console_appender(provider_obj, now_fn, *, interval_seconds_float=4.0):
    """Demo only: append one synthetic line every few seconds, like a running serve."""
    stop_event_obj = threading.Event()
    pod_id_list = [item_dict["pod_id_str"] for item_dict in provider_obj.get_console_pod_list()]
    log_path_obj = Path(provider_obj.console_log_path_str)

    def append_loop():
        index_int = 0
        while not stop_event_obj.wait(interval_seconds_float):
            now_ts = now_fn()
            pod_id_str = pod_id_list[index_int % len(pod_id_list)]
            field_dict = {"pod": pod_id_str, "account": DEMO_ACCOUNT_TUPLE[index_int % len(DEMO_ACCOUNT_TUPLE)]}
            if index_int % 9 == 4:
                line_str = _line_str("WARN", "reconcile.wait", now_ts, {**field_dict, "reason": "broker positions are still settling"})
            elif index_int % 3 == 0:
                line_str = _line_str("INFO", "norgate.sync.skipped", now_ts, {"status": "direct", "reason": "direct_norgate_mode"})
            else:
                line_str = _line_str("INFO", "fill.none", now_ts, {**field_dict, "vplan": 2})
            with log_path_obj.open("a", encoding="utf-8") as log_file_obj:
                log_file_obj.write(line_str + "\n")
            index_int += 1

    if pod_id_list:
        threading.Thread(target=append_loop, name="dashboard-v4-console-demo", daemon=True).start()
    return stop_event_obj
