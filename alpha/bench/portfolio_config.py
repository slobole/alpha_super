"""Portfolio configuration editing and saved-run identity for BENCH.

Edits preserve fields outside the form, including pinned pickle paths. The
review token binds the exact proposed config to the original file revision.
"""

from __future__ import annotations

import copy
import difflib
import hashlib
import json
import math
import os
import tempfile
import threading
from pathlib import Path

import yaml

from alpha.bench import portfolio_builder


MUTATION_LOCK = threading.RLock()


def fingerprint_str(config_dict: dict) -> str:
    payload_str = json.dumps(config_dict, sort_keys=True, default=str, allow_nan=False)
    return hashlib.sha256(payload_str.encode("utf-8")).hexdigest()


def resolve_config_path(rel_path_str: str) -> Path:
    if not rel_path_str.startswith("portfolios/"):
        raise ValueError("Unknown portfolio config.")
    filename_str = rel_path_str.removeprefix("portfolios/")
    config_path = portfolio_builder.resolve_write_path(filename_str)
    if config_path.name != filename_str or not config_path.is_file():
        raise ValueError("Unknown portfolio config.")
    return config_path


def delete_source_tuple(rel_path_str: str) -> tuple[Path, str]:
    config_path = resolve_config_path(rel_path_str)
    return config_path, hashlib.sha256(config_path.read_bytes()).hexdigest()


def read_config_tuple(rel_path_str: str) -> tuple[Path, dict, str]:
    config_path = resolve_config_path(rel_path_str)
    try:
        config_dict = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exception_obj:
        raise ValueError(f"Invalid portfolio YAML: {config_path.name}") from exception_obj
    if not isinstance(config_dict, dict):
        raise ValueError("The portfolio config must be a mapping.")
    return config_path, config_dict, fingerprint_str(config_dict)


def is_manager_bool(config_dict: dict) -> bool:
    return any(key_str in config_dict for key_str in (
        "name_str", "capital_base_float", "allocation_policy_str"
    ))


def rebalance_dict_from_form(form_obj, manager_bool: bool = True):
    frequency_str = form_obj.get("rebalance_frequency", "").strip()
    if not frequency_str:
        return None
    from alpha.engine.portfolio_manager import _coerce_rebalance_config

    policy_str = form_obj.get("rebalance_policy", "fixed")
    if not manager_bool and policy_str != "fixed":
        raise ValueError("Combine-pickles portfolios support fixed target weights only.")
    rebalance_dict = {"frequency_str": frequency_str, "policy_str": policy_str}
    if policy_str == "inverse_volatility":
        rebalance_dict["lookback_day_int"] = int(form_obj.get("rebalance_lookback", "60"))
    rebalance_obj = _coerce_rebalance_config(rebalance_dict)
    return rebalance_config_dict(rebalance_obj) if manager_bool else frequency_str


def rebalance_config_dict(rebalance_obj) -> dict:
    config_dict = {"frequency_str": rebalance_obj.frequency_str, "policy_str": rebalance_obj.policy_str}
    if rebalance_obj.policy_str == "inverse_volatility":
        config_dict["lookback_day_int"] = rebalance_obj.lookback_day_int
    return config_dict


def rebalance_label_str(rebalance_obj) -> str:
    if not rebalance_obj:
        return "No periodic rebalance"
    if isinstance(rebalance_obj, str):
        return f"{rebalance_obj.title()} · Fixed target weights"
    policy_str = rebalance_obj.get("policy_str", "fixed")
    policy_label_str = {
        "fixed": "Fixed target weights", "equal": "Equal weight",
        "inverse_volatility": "Inverse volatility", "risk_parity": "Inverse volatility",
    }.get(policy_str, policy_str)
    suffix_str = (
        f" · {rebalance_obj.get('lookback_day_int', 60)} days"
        if policy_str in {"inverse_volatility", "risk_parity"} else ""
    )
    return f"{rebalance_obj.get('frequency_str', '').title()} · {policy_label_str}{suffix_str}"


def editor_context_dict(config_dict: dict) -> dict:
    manager_bool = is_manager_bool(config_dict)
    rebalance_obj = config_dict.get("rebalance")
    if manager_bool and rebalance_obj:
        from alpha.engine.portfolio_manager import _coerce_rebalance_config
        rebalance_obj = rebalance_config_dict(_coerce_rebalance_config(rebalance_obj))
    elif isinstance(rebalance_obj, str):
        rebalance_obj = rebalance_obj.strip().lower()
    rebalance_dict = rebalance_obj if isinstance(rebalance_obj, dict) else {}
    return {
        "manager_bool": manager_bool,
        "name_str": config_dict.get("name_str" if manager_bool else "name", "Portfolio"),
        "capital_obj": config_dict.get("capital_base_float" if manager_bool else "capital"),
        "benchmark_str": config_dict.get("regression_benchmark_symbol_str" if manager_bool else "benchmark") or "",
        "start_str": config_dict.get("backtest_start_date_str", ""),
        "end_str": config_dict.get("end_date_str") or "",
        "allocation_str": config_dict.get("allocation_policy_str", "fixed").strip().lower(),
        "rebalance_frequency_str": rebalance_dict.get("frequency_str", "") if isinstance(rebalance_obj, dict) else rebalance_obj or "",
        "rebalance_policy_str": rebalance_dict.get("policy_str", "fixed"),
        "rebalance_lookback_int": rebalance_dict.get("lookback_day_int") or 60,
        "pod_list": [
            dict(pod_dict, weight_float=1.0 / len(config_dict["pods"]))
            if manager_bool and config_dict.get("allocation_policy_str", "fixed").strip().lower() == "equal"
            else pod_dict for pod_dict in config_dict.get("pods", [])
        ],
    }


def validate_name(config_dict: dict, config_path: Path) -> None:
    manager_bool = is_manager_bool(config_dict)
    name_str = config_dict.get("name_str" if manager_bool else "name", "")
    if not isinstance(name_str, str) or not name_str.strip() or any(
        char_str in name_str for char_str in '/\\:*?"<>|'
    ) or name_str in {".", ".."} or name_str.endswith((".", " ")):
        raise ValueError("Use a non-empty portfolio name without path or reserved filename characters.")
    for other_path in portfolio_builder.PORTFOLIOS_ROOT_PATH.glob("*.yaml"):
        if other_path.resolve() == config_path.resolve():
            continue
        try:
            other_dict = yaml.safe_load(other_path.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            other_dict = {}
        owned_name_set = {other_path.stem.casefold()}
        if isinstance(other_dict, dict):
            owned_name_set.add(str(other_dict.get("name_str", other_dict.get("name", ""))).casefold())
        owned_name_set.update(str(alias_str).casefold() for alias_str in lineage_dict_for(other_path).get("previous_name_list", []))
        if name_str.casefold() in owned_name_set or config_path.stem.casefold() in owned_name_set:
            raise ValueError("Another portfolio owns this name or filename, including historical names. Choose a unique name and filename to keep run histories separate.")


def validate_config(config_dict: dict, config_path: Path) -> None:
    validate_name(config_dict, config_path)
    manager_bool = is_manager_bool(config_dict)
    capital_obj = config_dict.get("capital_base_float" if manager_bool else "capital")
    if capital_obj is not None and (not math.isfinite(float(capital_obj)) or float(capital_obj) <= 0):
        raise ValueError("Capital must be finite and positive.")
    for pod_dict in config_dict.get("pods", []):
        weight_float = float(pod_dict.get("weight_float" if manager_bool else "weight", 0))
        if not math.isfinite(weight_float) or weight_float < 0:
            raise ValueError("Weights must be finite and non-negative.")
    if manager_bool:
        from alpha.engine.portfolio_manager import build_portfolio_manager_config
        build_portfolio_manager_config(config_dict)
    else:
        from strategies.run_portfolio import validate_portfolio_config
        validate_portfolio_config(config_dict, config_path)


def edited_config_dict(original_dict: dict, form_obj) -> dict:
    config_dict = copy.deepcopy(original_dict)
    manager_bool = is_manager_bool(config_dict)
    config_dict["name_str" if manager_bool else "name"] = form_obj.get("name", "").strip()
    capital_str = form_obj.get("capital", "").strip()
    config_dict["capital_base_float" if manager_bool else "capital"] = float(capital_str) if capital_str else None
    config_dict["regression_benchmark_symbol_str" if manager_bool else "benchmark"] = form_obj.get("benchmark", "").strip() or None
    config_dict["rebalance"] = rebalance_dict_from_form(form_obj, manager_bool)
    if manager_bool:
        config_dict["backtest_start_date_str"] = form_obj.get("start", "").strip()
        config_dict["end_date_str"] = form_obj.get("end", "").strip() or None
        config_dict["allocation_policy_str"] = form_obj.get("allocation", "fixed")
    pod_list = []
    # Original row IDs preserve every non-form field on retained pods.
    index_list = form_obj.getlist("row_id")
    if len(index_list) != len(set(index_list)) or set(index_list) != {str(index_int) for index_int in range(len(original_dict["pods"]))}:
        raise ValueError("The submitted pod rows do not match the saved config. Reload it.")
    for index_str in index_list:
        if index_str in form_obj.getlist("remove_pod"):
            continue
        index_int = int(index_str)
        pod_dict = copy.deepcopy(original_dict["pods"][index_int])
        pod_dict["weight_float" if manager_bool else "weight"] = float(form_obj.get(f"weight_{index_int}", "0"))
        pod_list.append(pod_dict)
    new_strategy_str = form_obj.get("add_strategy", "").strip()
    if new_strategy_str:
        weight_float = float(form_obj.get("add_weight", "0"))
        if manager_bool:
            pod_id_str = form_obj.get("add_pod_id", "").strip()
            pod_list.append({"pod_id_str": pod_id_str, "strategy_import_str": new_strategy_str, "weight_float": weight_float})
        else:
            pod_list.append({"strategy": new_strategy_str, "weight": weight_float})
    config_dict["pods"] = pod_list
    if manager_bool and config_dict["allocation_policy_str"] == "equal":
        for pod_dict in pod_list:
            pod_dict.pop("weight_float", None)
    for key_str in list(config_dict):
        if key_str not in original_dict and config_dict[key_str] is None:
            del config_dict[key_str]
    return config_dict


def yaml_text_str(config_dict: dict) -> str:
    return yaml.safe_dump(config_dict, sort_keys=False, allow_unicode=True)


def diff_text_str(original_dict: dict, proposed_dict: dict) -> str:
    return "".join(difflib.unified_diff(
        yaml_text_str(original_dict).splitlines(keepends=True),
        yaml_text_str(proposed_dict).splitlines(keepends=True),
        fromfile="Saved config", tofile="Proposed config",
    )) or "No configuration changes."


def assert_revision(rel_path_str: str, revision_str: str) -> tuple[Path, dict]:
    config_path, config_dict, current_revision_str = read_config_tuple(rel_path_str)
    if revision_str != current_revision_str:
        raise ValueError("This config changed after you opened it. Reload before saving or deleting.")
    return config_path, config_dict


def write_reviewed_config(payload_dict: dict) -> Path:
    with MUTATION_LOCK:
        source_path, original_dict = assert_revision(payload_dict["source"], payload_dict["revision"])
        config_dict = payload_dict["config"]
        clone_bool = payload_dict["mode"] == "clone"
        target_path = portfolio_builder.resolve_write_path(payload_dict["filename"]) if clone_bool else source_path
        validate_config(config_dict, target_path)
        if clone_bool:
            # Exclusive creation prevents accidental overwrite, including races.
            with target_path.open("x", encoding="utf-8") as file_obj:
                file_obj.write(yaml_text_str(config_dict))
            lineage_path = target_path.parent / ".bench" / f"{target_path.stem}.json"
            lineage_path.parent.mkdir(exist_ok=True)
            lineage_path.write_text(json.dumps({"parent_path_str": payload_dict["source"], "parent_revision_str": payload_dict["revision"]}), encoding="utf-8")
        else:
            old_name_str = original_dict.get("name_str", original_dict.get("name", source_path.stem))
            new_name_str = config_dict.get("name_str", config_dict.get("name", source_path.stem))
            if old_name_str != new_name_str:
                lineage_dict = lineage_dict_for(source_path)
                lineage_dict["previous_name_list"] = sorted(set([*lineage_dict.get("previous_name_list", []), old_name_str]))
                lineage_path = source_path.parent / ".bench" / f"{source_path.stem}.json"
                lineage_path.parent.mkdir(exist_ok=True)
                lineage_path.write_text(json.dumps(lineage_dict), encoding="utf-8")
            descriptor_int, temporary_str = tempfile.mkstemp(dir=source_path.parent, suffix=".tmp")
            temporary_path = Path(temporary_str)
            try:
                with os.fdopen(descriptor_int, "w", encoding="utf-8") as file_obj:
                    file_obj.write(yaml_text_str(config_dict))
                os.replace(temporary_path, source_path)
            finally:
                temporary_path.unlink(missing_ok=True)
        return target_path


def lineage_dict_for(config_path: Path) -> dict:
    lineage_path = config_path.parent / ".bench" / f"{config_path.stem}.json"
    try:
        lineage_dict = json.loads(lineage_path.read_text(encoding="utf-8"))
        return lineage_dict if isinstance(lineage_dict, dict) else {}
    except (OSError, ValueError, AttributeError):
        return {}


def parent_path_str(config_path: Path) -> str | None:
    return lineage_dict_for(config_path).get("parent_path_str")


def run_config_status_str(config_dict: dict, run_obj) -> str:
    if run_obj is None:
        return "Never run"
    saved_dict = run_obj.metadata_dict.get("source_config_dict")
    if not isinstance(saved_dict, dict):
        return "Config unverified"
    return "Current config" if fingerprint_str(saved_dict) == fingerprint_str(config_dict) else "Config changed"
