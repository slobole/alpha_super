"""Compare actual-host replay weights with the freshly captured engine orders."""
import gzip
import json
from pathlib import Path
import pickle

import pandas as pd

from build_signoff_report import ROOT_PATH, STRATEGY_SPEC_LIST, hash_file


def main() -> None:
    input_manifest_list = []
    row_list = []
    for variant_str, family_str, _, key_str in STRATEGY_SPEC_LIST:
        if family_str not in {"taa", "ndx"}:
            continue
        replay_path = ROOT_PATH / "taa_ndx_native" / f"{family_str}_actual_host_replay.json"
        trace_path = ROOT_PATH / "baselines" / key_str / "trace.pkl.gz"
        for input_path in (replay_path, trace_path):
            input_manifest_list.append({"path": str(input_path), "bytes": input_path.stat().st_size, "sha256": hash_file(input_path)})
        with gzip.open(trace_path, "rb") as handle_obj:
            trace_dict = {pd.Timestamp(row_dict["decision_date"]): row_dict for row_dict in pickle.load(handle_obj)}
        replay_dict = json.loads(replay_path.read_text(encoding="utf-8"))["variant_result_dict"][variant_str]
        for replay_row_dict in replay_dict["row_list"]:
            decision_ts = pd.Timestamp(replay_row_dict["decision_date_str"])
            engine_row_dict = trace_dict[decision_ts]
            weight_dict = {}
            for order_dict in engine_row_dict["orders"]:
                if order_dict["amount_float"] == 0:
                    continue
                if not order_dict["target_bool"] or order_dict["unit_str"] != "percent":
                    raise ValueError(f"Unsupported monthly engine intent: {order_dict}")
                weight_dict[order_dict["asset_str"]] = float(order_dict["amount_float"])
            actual_dict = replay_row_dict["actual_weight_dict"]
            symbol_set = set(weight_dict) | set(actual_dict)
            max_difference_float = max((abs(weight_dict.get(symbol_str, 0)-actual_dict.get(symbol_str, 0)) for symbol_str in symbol_set), default=0.)
            row_list.append({"variant": variant_str, "decision_date": str(decision_ts.date()),
                "execution_date": str(pd.Timestamp(engine_row_dict["execution_date"]).date()),
                "engine_weight_map": weight_dict, "host_weight_map": actual_dict,
                "maximum_weight_difference": max_difference_float,
                "passed": max_difference_float <= 1e-12})
    for source_dict in input_manifest_list:
        if hash_file(Path(source_dict["path"])) != source_dict["sha256"]:
            raise ValueError("Monthly bridge input changed during comparison.")
    result_dict = {"status": "pass" if row_list and all(row_dict["passed"] for row_dict in row_list) else "fail",
        "source": {"path": str(Path(__file__).resolve()), "sha256": hash_file(Path(__file__))},
        "input_manifest": input_manifest_list, "rows": row_list,
        "attempted": len(row_list), "passed": sum(row_dict["passed"] for row_dict in row_list),
        "limitation": "Weights only; does not establish live physical shares, broker fills or historical data vintages."}
    (ROOT_PATH / "monthly_canonical_trace_bridge.json").write_text(json.dumps(result_dict, indent=2), encoding="utf-8")
    print({"status": result_dict["status"], "attempted": result_dict["attempted"], "passed": result_dict["passed"]})
    if result_dict["status"] != "pass":
        print([row_dict for row_dict in row_list if not row_dict["passed"]][:5])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
