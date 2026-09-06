"""Read a pinned saved reference summary; never run or certify a LIVE replay.

Legacy summaries lack historical account identity, frozen code/data hashes and
a full mark/timing contract. Today's route/release cannot fill those proof gaps.
"""

from datetime import date, datetime
import hashlib
import json
import math
from pathlib import Path
import re
from zoneinfo import ZoneInfo

from alpha.live.dashboard_v3.operator_tools import redact_diagnostic_value


METADATA_FIELD_TUPLE = (
    "release_id_str", "actual_equity_source_str", "actual_equity_basis_str",
    "actual_equity_timestamp_str", "reference_accounting_contract_version_str",
    "reference_dividend_cash_ledger_mode_str",
)


def saved_comparison_dict(account_dict, *, from_date_str, to_date_str):
    """Selected dates bound the effective account interval; no clipped totals.

    Only explicitly attributable, exact-period artifacts expose recorded values.
    Even those values are diagnostics, not validated P&L/returns/fill slippage.
    Source bytes are bounded, hashed and parsed once; pickle files are never read.
    """
    requested_from_str = max(from_date_str, account_dict["effective_from"])
    requested_to_str = min(to_date_str, account_dict.get("effective_to") or "9999-12-31")
    result_dict = {
        "status_str": "unavailable", "source_hash_str": None, "artifact_folder_str": None,
        "from_date_str": requested_from_str, "to_date_str": requested_to_str,
        "artifact_from_date_str": None, "artifact_to_date_str": None,
        "attribution_str": "Unproven", "interval_str": "Unproven", "valuation_str": "Not established",
        "metadata_dict": {}, "recorded_value_list": [], "issue_list": [],
        "replay_str": "Not proven — saved diagnostic, not a certified replay",
    }
    source_path_str = account_dict.get("reference_summary_path")
    if not source_path_str:
        result_dict["issue_list"].append("No saved comparison is pinned for this account period. Configure reference_summary_path in the operator registry; this view never runs a comparison.")
        return result_dict
    try:
        source_path_obj = Path(source_path_str)
        with source_path_obj.open("rb") as source_file_obj:
            summary_bytes = source_file_obj.read(1_000_001)
        if len(summary_bytes) > 1_000_000:
            raise ValueError("Oversized summary.")
        summary_dict = json.loads(summary_bytes.decode("utf-8"))
        if not isinstance(summary_dict, dict):
            raise ValueError("Summary must be an object.")
        result_dict["source_hash_str"] = hashlib.sha256(summary_bytes).hexdigest()
        account_alias_list = [summary_dict[key_str] for key_str in ("account_route_str", "account_id_str", "account_str") if summary_dict.get(key_str) not in (None, "")]
        mode_alias_list = [summary_dict[key_str] for key_str in ("mode_str", "env_mode_str", "session_mode_str") if summary_dict.get(key_str) not in (None, "")]
        identity_bool = (summary_dict.get("pod_id_str") == account_dict["pod_id"] and summary_dict.get("mode_str") == "live"
                         and bool(account_alias_list) and all(value_obj == account_dict["account_route"] for value_obj in account_alias_list))
        if summary_dict.get("pod_id_str") != account_dict["pod_id"] or summary_dict.get("pod_str") not in (None, "", account_dict["pod_id"]) or summary_dict.get("mode_str") != "live" or any(value_obj != "live" for value_obj in mode_alias_list) or any(value_obj != account_dict["account_route"] for value_obj in account_alias_list):
            result_dict["issue_list"].append("Saved strategy, account or LIVE identity does not match this account period. Artifact facts are withheld.")
            return result_dict
        result_dict["artifact_folder_str"] = redact_diagnostic_value(source_path_obj.parent.name[:80])
        result_dict["status_str"] = "unverified"
        result_dict["attribution_str"] = "Explicit saved strategy + account + LIVE" if identity_bool else "Historical account identity missing"
        if not identity_bool:
            result_dict["issue_list"].append("The saved summary has no historical account identity. Today's routing or the pinned path does not prove whose money this artifact represents; numeric facts are withheld.")
        artifact_from_str, artifact_to_str = summary_dict.get("deployment_start_date_str"), summary_dict.get("target_session_date_str")
        if any(not isinstance(value_obj, str) or date.fromisoformat(value_obj).isoformat() != value_obj for value_obj in (artifact_from_str, artifact_to_str)) or artifact_from_str > artifact_to_str:
            raise ValueError("Invalid saved interval.")
        result_dict.update(artifact_from_date_str=artifact_from_str, artifact_to_date_str=artifact_to_str)
        exact_interval_bool = (artifact_from_str, artifact_to_str) == (requested_from_str, requested_to_str)
        result_dict["interval_str"] = "Exact effective account dates" if exact_interval_bool else "Different dates — no selected-period numeric comparison"
        if not exact_interval_bool:
            result_dict["issue_list"].append("Saved aggregates cannot be clipped to the selected period. Choose the recorded dates or pin the correct historical artifact.")
        for key_str in METADATA_FIELD_TUPLE:
            value_obj = summary_dict.get(key_str)
            if not isinstance(value_obj, str):
                continue
            # Credential redaction alone does not remove paths embedded in a
            # supposedly descriptive source/basis field. Never expose those.
            if "/" in value_obj or "\\" in value_obj or re.match(r"^[A-Za-z]:", value_obj):
                result_dict["metadata_dict"][key_str] = "Path-like metadata withheld"
            else:
                result_dict["metadata_dict"][key_str] = redact_diagnostic_value(value_obj[:500])
        # *** CRITICAL *** retrospective diagnostics only: matching date labels
        # do not establish equal marks, causal timing, capital, or code/data vintage.
        result_dict["issue_list"].append("Code/data snapshot hashes and a complete decision/fill/valuation timing contract are not proven by this legacy summary. Matching release IDs would not prove replay; a different current release would not invalidate history.")
        result_dict["issue_list"].append("Trade aggregates describe the target execution session, not every trade in the selected period. Missing price components may be stored as zero; zero is not proof of zero slippage. No statistical tracking error or investment P&L is inferred.")
        if summary_dict.get("reference_dividend_cash_ledger_mode_str") != "enabled":
            result_dict["issue_list"].append("Reference dividend cash accounting is disabled or unproven; do not equate this simulation ledger with official broker NAV/TWR.")
        if identity_bool and exact_interval_bool:
            actual_timestamp_str = summary_dict.get("actual_equity_timestamp_str")
            try:
                actual_ts = datetime.fromisoformat(actual_timestamp_str.replace("Z", "+00:00"))
                if actual_ts.tzinfo is None or actual_ts.astimezone(ZoneInfo("America/New_York")).date().isoformat() != artifact_to_str:
                    raise ValueError("Missing or different mark date.")
                if summary_dict.get("actual_equity_basis_str") != "eod_broker_netliq" or summary_dict.get("actual_equity_source_str") != "pod_state_history.eod":
                    raise ValueError("Selected-session EOD source not established.")
            except (ValueError, TypeError, AttributeError):
                result_dict["valuation_str"] = "Actual selected-session EOD evidence missing or mismatched"
                result_dict["issue_list"].append("Actual valuation time, date or EOD source is missing, ambiguous or mismatched. Recorded values are withheld; same-time valuation is not established.")
                return result_dict
            result_dict["valuation_str"] = "Actual saved EOD record; reference mark alignment still unproven"
            for field_str, label_str in (
                ("deployment_initial_cash_float", "Recorded reference starting budget"),
                ("actual_equity_float", "Saved actual value — see source and timestamp"),
                ("backtest_equity_float", "Saved reference value — exact mark time unproven"),
            ):
                value_obj = summary_dict.get(field_str)
                try:
                    valid_bool = type(value_obj) in {float, int} and math.isfinite(value_obj) and value_obj >= 0
                except OverflowError:
                    valid_bool = False
                if valid_bool:
                    result_dict["recorded_value_list"].append({"label_str": label_str, "value_float": value_obj})
            if len(result_dict["recorded_value_list"]) != 3:
                result_dict["issue_list"].append("Some recorded values are missing or invalid. They are omitted, never treated as zero.")
    except (OSError, ValueError, TypeError):
        result_dict["status_str"] = "unavailable"
        result_dict["recorded_value_list"] = []
        result_dict["issue_list"].append("Pinned comparison evidence could not be read or validated. No other artifact was substituted.")
    return result_dict
