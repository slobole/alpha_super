"""One readable duration style for countdowns and ages (refreshed every 15 s)."""


def human_duration_str(seconds_float):
    """'under 1 min', '42 min', '6 h 05 min', '4 h', '2 d 4 h'."""
    minutes_int = max(0, int(seconds_float)) // 60
    if minutes_int < 1:
        return "under 1 min"
    if minutes_int < 60:
        return f"{minutes_int} min"
    hours_int, minutes_int = divmod(minutes_int, 60)
    if hours_int < 24:
        return f"{hours_int} h" if minutes_int == 0 else f"{hours_int} h {minutes_int:02} min"
    return f"{hours_int // 24} d {hours_int % 24} h"
