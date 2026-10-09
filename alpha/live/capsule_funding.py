"""A capsule funding rejection is permanent for that decision cycle."""
import json

from alpha.live.mr_capsule_adapter import MR_CAPSULE_STRATEGY_IMPORT_TUPLE


def persist_capsule_buy_drop(state_store_obj, release_obj, vplan_obj):
    """Commit before claim; a losing preflight cannot change a claimed batch."""
    if release_obj.strategy_import_str not in MR_CAPSULE_STRATEGY_IMPORT_TUPLE:
        raise ValueError("Funding buy suppression is limited to MR capsule.")
    with state_store_obj._connect() as connection_obj:
        connection_obj.execute("BEGIN IMMEDIATE")
        row_obj = connection_obj.execute("""SELECT d.snapshot_metadata_json_str, v.status_str
            FROM decision_plan d JOIN vplan v ON v.decision_plan_id_int=d.decision_plan_id_int
            WHERE v.vplan_id_int=? AND d.decision_plan_id_int=? AND d.release_id_str=?
              AND d.pod_id_str=? AND d.account_route_str=?""",
            (vplan_obj.vplan_id_int, vplan_obj.decision_plan_id_int, release_obj.release_id_str,
                release_obj.pod_id_str, release_obj.account_route_str)).fetchone()
        if row_obj is None:
            raise ValueError("Capsule funding cycle identity differs.")
        if row_obj["status_str"] != "ready":
            return False
        metadata_dict = json.loads(row_obj["snapshot_metadata_json_str"])
        metadata_dict["funding_buys_dropped_bool"] = True
        connection_obj.execute("UPDATE decision_plan SET snapshot_metadata_json_str=? WHERE decision_plan_id_int=?",
            (json.dumps(metadata_dict, sort_keys=True), vplan_obj.decision_plan_id_int))
    return True
