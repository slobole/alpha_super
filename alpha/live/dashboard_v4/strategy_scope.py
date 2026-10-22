"""Measured Pod capital for operator performance; full client ownership is retained."""

from copy import deepcopy


def strategy_workspace_dict(workspace_dict):
    """Use the same accounting on dated strategies, including their cash/history.

    *** CRITICAL *** Reporting only: preserve each Pod's original measured
    start/end and every incomplete-scope gate. Current status, positions and
    enabled flags cannot remove capital or rebase an observed return period.
    """
    result_dict = deepcopy(workspace_dict)
    client_dict = result_dict["client_dict"]
    client_dict["accounts"] = [account_dict for account_dict in client_dict["accounts"]
        if account_dict.get("role", "strategy") == "strategy"]
    if client_dict["accounts"]:
        client_dict["mandate_start_date"] = min(account_dict["effective_from"] for account_dict in client_dict["accounts"])
    if "valuation_account_list" in result_dict:
        result_dict["valuation_account_list"] = [account_dict for account_dict in result_dict["valuation_account_list"]
            if account_dict.get("role", "strategy") == "strategy"]
    return result_dict
