"""Read-only IBKR probe for one daily pod (CORE5 or MR capsule) before go-live.

Runs the same broker reads and pre-submit checks the daily serve runs, against
the pod's own account, and prints PASS/FAIL per step. It never transmits an
order: the margin/funding/borrow checks use IBKR what-if previews
(whatIf=True), and the broker adapter is wrapped so only read calls exist.

Usage, from the repo root, with a client ID no other process uses:
  .\\.venv\\Scripts\\python.exe scripts\\probe_daily_pod_ibkr.py --release <pod YAML> --client-id 90
Run it once during market hours and once around 09:20 New York time (the
serve's checks run between 09:23:30 and 09:28). Exit 0 = every step passed.

Owner decision 2026-10-11: the pre-submit checks stay. If a check fails only
because IBKR does not return a field it reads, remove that check instead of
working around it.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

REPO_ROOT_PATH = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from alpha.live.core5_adapter import CORE5_ASSET_TUPLE, CORE5_STRATEGY_IMPORT_STR  # noqa: E402
from alpha.live.models import BrokerOrderRequest  # noqa: E402
from alpha.live.mr_capsule_adapter import MR_CAPSULE_STRATEGY_IMPORT_TUPLE  # noqa: E402
from alpha.live.order_clerk import IBKRGatewayBrokerAdapter  # noqa: E402
from alpha.live.release_manifest import parse_release_manifest  # noqa: E402


DEFAULT_PROBE_CLIENT_ID_INT = 90
# Capsule funding compares buy notional with BuyingPower; one share at this
# placeholder price only proves the field is read, not that a real batch fits.
PLACEHOLDER_PRICE_FLOAT = 100.0
CAPSULE_SAMPLE_ASSET_TUPLE = ("AAPL", "BIL")
NEW_YORK_TIMEZONE_OBJ = ZoneInfo("America/New_York")


class ReadOnlyBroker:
    """Forwards only the read calls this probe needs; anything else raises."""

    ALLOWED_METHOD_TUPLE = (
        "get_visible_account_route_set", "get_core5_account_snapshot", "get_eod_account_snapshot",
        "get_capsule_account_snapshot", "get_daily_execution_snapshot",
        "get_core5_funding_evidence", "get_capsule_funding_evidence",
    )

    def __init__(self, adapter_obj):
        object.__setattr__(self, "_adapter_obj", adapter_obj)

    def __getattr__(self, name_str):
        if name_str not in self.ALLOWED_METHOD_TUPLE:
            raise AttributeError(f"The probe is read-only; {name_str} is not allowed.")
        return getattr(object.__getattribute__(self, "_adapter_obj"), name_str)


def _sample_request(release_obj, asset_str, amount_float, index_int):
    return BrokerOrderRequest(
        release_id_str=release_obj.release_id_str, pod_id_str=release_obj.pod_id_str,
        account_route_str=release_obj.account_route_str, submission_key_str="probe",
        order_request_key_str=f"probe:{asset_str}:{index_int}", asset_str=asset_str,
        broker_order_type_str="MOO", order_class_str="probe", unit_str="shares",
        amount_float=float(amount_float), target_bool=False, trade_id_int=None,
        sizing_reference_price_float=PLACEHOLDER_PRICE_FLOAT, portfolio_value_float=0.0)


def core5_sample_request_list(release_obj, position_dict):
    """One-share buys of every CORE5 asset except DBC, plus a DBC sell that grows
    the short by one share so the shortable-shares check also runs."""
    request_list = []
    for index_int, asset_str in enumerate(CORE5_ASSET_TUPLE):
        if asset_str == "DBC":
            amount_float = -(max(float(position_dict.get("DBC", 0.0)), 0.0) + 1.0)
        else:
            amount_float = 1.0
        request_list.append(_sample_request(release_obj, asset_str, amount_float, index_int))
    return request_list


def capsule_sample_request_list(release_obj):
    return [_sample_request(release_obj, asset_str, 1.0, index_int)
            for index_int, asset_str in enumerate(CAPSULE_SAMPLE_ASSET_TUPLE)]


def _plain(value_obj):
    if is_dataclass(value_obj):
        return _plain(asdict(value_obj))
    if isinstance(value_obj, dict):
        return {str(key_obj): _plain(item_obj) for key_obj, item_obj in value_obj.items()}
    if isinstance(value_obj, (list, tuple, set)):
        return [_plain(item_obj) for item_obj in value_obj]
    if isinstance(value_obj, datetime):
        return value_obj.isoformat()
    return value_obj


def _snapshot_summary_str(snapshot_obj):
    return (f"cash {snapshot_obj.cash_float:,.2f}, NAV {snapshot_obj.total_value_float:,.2f}, "
            f"positions {dict(sorted(snapshot_obj.position_amount_map.items()))}, "
            f"open orders {len(snapshot_obj.open_order_id_list)}")


def _funding_summary_str(evidence_dict):
    preview_str = ", ".join(f"{item_dict['asset_str']} {item_dict['status_str']}"
        for item_dict in evidence_dict.get("preview_list") or [])
    fact_list = [f"type {evidence_dict.get('trading_type_str')}"]
    for key_str, label_str in (("available_funds_float", "AvailableFunds"), ("excess_liquidity_float", "ExcessLiquidity"),
            ("buying_power_float", "BuyingPower"), ("dbc_shortable_shares_float", "DBC shortable")):
        if evidence_dict.get(key_str) is not None:
            fact_list.append(f"{label_str} {evidence_dict[key_str]:,.0f}")
    return "; ".join(fact_list) + (f"; previews: {preview_str}" if preview_str else "")


def run_probe(release_obj, broker_obj):
    """Run every step; one failure never stops the others. Returns step dicts."""
    account_str = release_obj.account_route_str
    core5_bool = release_obj.strategy_import_str == CORE5_STRATEGY_IMPORT_STR
    step_list = []
    context_dict = {}

    def run_step(label_str, step_fn):
        try:
            detail_str, raw_obj = step_fn()
            step_list.append({"label_str": label_str, "status_str": "PASS", "detail_str": detail_str, "raw": _plain(raw_obj)})
        except Exception as exception_obj:  # Every IBKR or check failure is the finding itself.
            step_list.append({"label_str": label_str, "status_str": "FAIL",
                "detail_str": f"{type(exception_obj).__name__}: {exception_obj}", "raw": None})

    def visible_step():
        account_set = broker_obj.get_visible_account_route_set()
        if account_set is None or account_str not in account_set:
            raise ValueError(f"account {account_str} is not visible on this IBKR session")
        return f"account {account_str} visible", sorted(account_set)

    run_step("IBKR session sees the account", visible_step)
    if core5_bool:
        def account_step():
            snapshot_obj = broker_obj.get_core5_account_snapshot(account_str, include_portfolio_valuation_bool=True)
            context_dict["position_dict"] = dict(snapshot_obj.position_amount_map)
            return _snapshot_summary_str(snapshot_obj), snapshot_obj
        run_step("Account read (16:10 and before sending)", account_step)
    else:
        def eod_step():
            snapshot_obj = broker_obj.get_eod_account_snapshot(account_str)
            return _snapshot_summary_str(snapshot_obj), snapshot_obj
        run_step("Account read at 16:10", eod_step)

        def presubmit_step():
            snapshot_obj = broker_obj.get_capsule_account_snapshot(account_str)
            return _snapshot_summary_str(snapshot_obj), snapshot_obj
        run_step("Account read before sending", presubmit_step)

    def daily_step():
        daily_obj = broker_obj.get_daily_execution_snapshot(account_str)
        if not daily_obj.complete_bool:
            raise ValueError("the daily account refresh reported incomplete data")
        return (_snapshot_summary_str(daily_obj.broker_snapshot_obj)
                + f"; open order rows {len(daily_obj.open_order_row_list)}"), daily_obj
    run_step("Daily close read (holdings and all-client open orders)", daily_step)

    if core5_bool:
        def core5_funding_step():
            if "position_dict" not in context_dict:
                raise ValueError("skipped: needs the account read above to pass first")
            request_list = core5_sample_request_list(release_obj, context_dict["position_dict"])
            evidence_dict = broker_obj.get_core5_funding_evidence(account_str, request_list, context_dict["position_dict"])
            return _funding_summary_str(evidence_dict), evidence_dict
        run_step("Pre-submit margin, funding and DBC borrow check (what-if only)", core5_funding_step)
    else:
        def capsule_funding_step():
            evidence_dict = broker_obj.get_capsule_funding_evidence(account_str, capsule_sample_request_list(release_obj))
            return _funding_summary_str(evidence_dict), evidence_dict
        run_step("Pre-submit buying power and margin check (what-if only)", capsule_funding_step)
    return step_list


def main(argv_list=None, *, adapter_factory_fn=IBKRGatewayBrokerAdapter):
    parser_obj = argparse.ArgumentParser(description="Read-only IBKR probe for a CORE5 or MR capsule pod.")
    parser_obj.add_argument("--release", required=True, help="Path to the pod's release YAML.")
    parser_obj.add_argument("--client-id", type=int, default=DEFAULT_PROBE_CLIENT_ID_INT,
        help="IBKR API client ID for this probe; no other process may use it.")
    parser_obj.add_argument("--port", type=int, help="Override the release's IBKR port.")
    parser_obj.add_argument("--json", action="store_true", help="Print the full result as JSON.")
    args_obj = parser_obj.parse_args(argv_list)

    release_obj = parse_release_manifest(str(Path(args_obj.release).resolve()))
    if (release_obj.strategy_import_str != CORE5_STRATEGY_IMPORT_STR
            and release_obj.strategy_import_str not in MR_CAPSULE_STRATEGY_IMPORT_TUPLE):
        print("This probe is for CORE5 and MR capsule releases only.", file=sys.stderr)
        return 2
    if args_obj.client_id <= 0 or args_obj.client_id == release_obj.broker_client_id_int:
        print("Use a positive client ID that differs from the pod's own serve.", file=sys.stderr)
        return 2
    port_int = args_obj.port if args_obj.port is not None else release_obj.broker_port_int
    broker_obj = ReadOnlyBroker(adapter_factory_fn(
        host_str=release_obj.broker_host_str, port_int=port_int,
        client_id_int=args_obj.client_id, timeout_seconds_float=release_obj.broker_timeout_seconds_float))
    started_ts = datetime.now(UTC)
    step_list = run_probe(release_obj, broker_obj)
    passed_bool = all(step_dict["status_str"] == "PASS" for step_dict in step_list)
    if args_obj.json:
        print(json.dumps({"release_id_str": release_obj.release_id_str, "pod_id_str": release_obj.pod_id_str,
            "account_route_str": release_obj.account_route_str, "mode_str": release_obj.mode_str,
            "port_int": port_int, "client_id_int": args_obj.client_id, "started_at_str": started_ts.isoformat(),
            "passed_bool": passed_bool, "step_list": step_list}, indent=1, default=str))
    else:
        print(f"Probe {release_obj.pod_id_str} ({release_obj.mode_str}) account {release_obj.account_route_str} "
              f"via {release_obj.broker_host_str}:{port_int} client {args_obj.client_id} at "
              f"{started_ts.astimezone(NEW_YORK_TIMEZONE_OBJ):%Y-%m-%d %H:%M} New York")
        print("Read-only: what-if previews only; no order is transmitted.")
        for step_dict in step_list:
            print(f"{step_dict['status_str']}  {step_dict['label_str']}: {step_dict['detail_str']}")
        print("RESULT: all checks passed." if passed_bool else "RESULT: at least one check failed; send this output for review.")
    return 0 if passed_bool else 1


if __name__ == "__main__":
    raise SystemExit(main())
