"""Cross-mode verdict resolution for the top bar."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alpha.live.dashboard_v3.data import (
    _effective_severity_str,
    count_pods_needing_action_int,
    count_pods_waiting_int,
)


@dataclass
class TopBarVerdict:
    severity_str: str
    title_str: str
    subtitle_str: str

    def as_dict(self) -> dict[str, str]:
        return {
            "severity_str": self.severity_str,
            "title_str": self.title_str,
            "subtitle_str": self.subtitle_str,
        }


def resolve_top_bar_verdict(summary_dict: dict[str, Any]) -> TopBarVerdict:
    pod_row_dict_list = summary_dict.get("pod_row_dict_list") or []
    if not pod_row_dict_list:
        return TopBarVerdict(
            severity_str="gray",
            title_str="No PODs enabled",
            subtitle_str="No live, paper, or incubation pods loaded.",
        )
    red_count_int = count_pods_needing_action_int(pod_row_dict_list)
    yellow_count_int = count_pods_waiting_int(pod_row_dict_list)
    gray_count_int = sum(
        1
        for row_dict in pod_row_dict_list
        if _effective_severity_str(row_dict) == "gray"
    )
    if red_count_int > 0:
        return TopBarVerdict(
            severity_str="red",
            title_str=f"{red_count_int} pod(s) need action",
            subtitle_str="Open the attention queue and inspect the first red item.",
        )
    if yellow_count_int > 0:
        return TopBarVerdict(
            severity_str="yellow",
            title_str=f"{yellow_count_int} pod(s) waiting",
            subtitle_str="No red blocker — timing or review is pending.",
        )
    if gray_count_int > 0:
        return TopBarVerdict(
            severity_str="gray",
            title_str=f"{gray_count_int} pod(s) missing state",
            subtitle_str="Setup or stale data needs inspection before trusting the view.",
        )
    return TopBarVerdict(
        severity_str="green",
        title_str="All clear",
        subtitle_str="Enabled pods are idle, complete, or healthy.",
    )
