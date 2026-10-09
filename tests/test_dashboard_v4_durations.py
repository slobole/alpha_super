"""Countdowns and ages read as durations, not clock times."""

import pytest

from alpha.live.dashboard_v4.durations import human_duration_str


@pytest.mark.parametrize("seconds_float,expected_str", [
    (-5, "under 1 min"), (0, "under 1 min"), (59.9, "under 1 min"), (60, "1 min"), (42 * 60 + 59, "42 min"),
    (3600, "1 h"), (3600 + 5 * 60, "1 h 05 min"), (6 * 3600 + 25 * 60 + 30, "6 h 25 min"),
    (24 * 3600, "1 d 0 h"), (2 * 86400 + 4 * 3600 + 59 * 60, "2 d 4 h"),
])
def test_human_duration(seconds_float, expected_str):
    assert human_duration_str(seconds_float) == expected_str
