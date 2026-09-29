"""Frozen cell definitions and grids (PREREG section 4)."""

from __future__ import annotations

import dataclasses

MAX_MOVE_FLOAT = 0.03  # pin window: the largest |close-to-close| move allowed
HOLD_FLOOR_FLOAT = 0.97
VOLUME_SHOCK_MULT_FLOAT = 5.0
VOLUME_MEDIAN_WINDOW_INT = 60
QUEUE_MAX_AGE_INT = 60
TIME_STOP_INT = 252


@dataclasses.dataclass(frozen=True)
class VCell:
    jump_float: float = 0.15
    theta_float: float = 0.005  # median |close-to-close| over the pin window
    window_int: int = 5
    slots_int: int = 10
    break_float: float | None = 0.95
    terminal_factor_float: float = 1.0
    half_str: str = "ALL"  # ALL | R1000 (events of Russell 1000 members) | R2000 (events of Russell 2000-only members)

    @property
    def key_str(self) -> str:
        break_str = "noBreak" if self.break_float is None else f"brk{self.break_float * 100:g}"
        term_str = "" if self.terminal_factor_float == 1.0 else f"|term{self.terminal_factor_float:g}"
        half_str = "" if self.half_str == "ALL" else f"|{self.half_str}"
        return f"V|J{self.jump_float * 100:g}|th{self.theta_float * 100:g}|W{self.window_int}|K{self.slots_int}|{break_str}{term_str}{half_str}"


V0_CELL = VCell()
P_JUMP_TUPLE = (0.10, 0.15)
P_THETA_TUPLE = (0.003, 0.005, 0.008)
S_WINDOW_TUPLE = (3, 5, 10)
S_SLOTS_TUPLE = (5, 10, 20)
HALF_TUPLE = ("R1000", "R2000")
SENSITIVITY_DICT = {
    "terminal_x0.995": dataclasses.replace(V0_CELL, terminal_factor_float=0.995),
    "terminal_x0.99": dataclasses.replace(V0_CELL, terminal_factor_float=0.99),
    "no_break_stop": dataclasses.replace(V0_CELL, break_float=None),
    "break_stop_10pct": dataclasses.replace(V0_CELL, break_float=0.90),
}


def stage_dict() -> dict[str, list[tuple[tuple, VCell]]]:
    return {
        "P_event_pin": [((jump_float, theta_float), dataclasses.replace(V0_CELL, jump_float=jump_float, theta_float=theta_float)) for jump_float in P_JUMP_TUPLE for theta_float in P_THETA_TUPLE],
        "S_speed_slots": [((window_int, slots_int), dataclasses.replace(V0_CELL, window_int=window_int, slots_int=slots_int)) for window_int in S_WINDOW_TUPLE for slots_int in S_SLOTS_TUPLE],
    }


def grid_cells() -> list[VCell]:
    seen_dict: dict[str, VCell] = {}
    for cell_list in stage_dict().values():
        for _, cell in cell_list:
            seen_dict.setdefault(cell.key_str, cell)
    return list(seen_dict.values())


def all_primary_cells() -> list[VCell]:
    return grid_cells() + list(SENSITIVITY_DICT.values())


def half_cells(half_str: str) -> list[VCell]:
    return [dataclasses.replace(cell, half_str=half_str) for cell in grid_cells()]
