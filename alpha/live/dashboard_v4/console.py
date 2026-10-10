"""Console page scope: which Pods, which source, and the page's own wording."""

import re

import yaml

from alpha.live.dashboard_v3.operator_tools import strategy_display_name_str
from alpha.live.logging_utils import resolve_operator_log_path_str
from alpha.live.release_manifest import load_release_list


CONSOLE_MODE_TUPLE = ("live", "paper")
POD_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,200}")
SOURCE_NOTE_STR = ("The operator lines each serve prints, from the shared operator log. "
    "Tick summaries and Python error traces appear only in the serve window.")


def resolve_console_log_path_str(provider_obj):
    """The serve's operator log beside its event log; the demo owns a fixture file."""
    demo_path_str = getattr(provider_obj, "console_log_path_str", None)
    if demo_path_str:
        return demo_path_str
    return resolve_operator_log_path_str(provider_obj.event_log_path_str)


def load_console_pod_list(provider_obj):
    """Enabled LIVE Pods first, then enabled PAPER Pods. Never INCUBATION."""
    if hasattr(provider_obj, "get_console_pod_list"):
        return provider_obj.get_console_pod_list()
    try:
        release_list = load_release_list(provider_obj.app_obj().releases_root_path_str)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, yaml.YAMLError):
        return []
    pod_list, seen_set = [], set()
    for mode_str in CONSOLE_MODE_TUPLE:
        for release_obj in sorted(release_list, key=lambda item_obj: item_obj.pod_id_str):
            pod_id_str = release_obj.pod_id_str
            if (release_obj.mode_str != mode_str or not release_obj.enabled_bool or pod_id_str in seen_set
                    or pod_id_str == "all" or not POD_ID_RE.fullmatch(pod_id_str)):
                continue
            seen_set.add(pod_id_str)
            pod_list.append({"pod_id_str": pod_id_str, "mode_str": mode_str,
                "name_str": strategy_display_name_str({"pod_id_str": pod_id_str,
                    "strategy_import_str": release_obj.strategy_import_str})})
    return pod_list


def build_console_page_dict(pod_list, selected_pod_str, url_for_fn):
    option_list = [{"pod_id_str": "all", "label_str": "All pods", "mode_str": "",
        "selected_bool": selected_pod_str == "all", "url_str": url_for_fn("console", pod="all")}]
    option_list.extend({"pod_id_str": item_dict["pod_id_str"], "label_str": item_dict["name_str"],
        "mode_str": item_dict["mode_str"], "selected_bool": item_dict["pod_id_str"] == selected_pod_str,
        "url_str": url_for_fn("console", pod=item_dict["pod_id_str"])} for item_dict in pod_list)
    selected_dict = next(item_dict for item_dict in option_list if item_dict["selected_bool"])
    return {
        "selected_pod_str": selected_pod_str, "title_str": selected_dict["label_str"],
        "mode_str": selected_dict["mode_str"], "option_list": option_list,
        "pod_url_str": url_for_fn("pod", pod_id_str=selected_pod_str) if selected_dict["mode_str"] == "live" else "",
        "tail_url_str": url_for_fn("console_tail", pod_id_str=selected_pod_str),
        "download_url_str": url_for_fn("console_download", pod_id_str=selected_pod_str),
        "source_note_str": SOURCE_NOTE_STR,
    }
