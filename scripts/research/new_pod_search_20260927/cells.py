"""Frozen cell definitions and grids (PREREG section 4)."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class MCell:
    """Family M: jump J on a volume shock, pin window W with pin_vol <= theta, K slots, break stop, 252-session time stop."""

    jump_float: float = 0.15
    theta_float: float = 0.010
    window_int: int = 5
    slots_int: int = 20
    break_float: float | None = 0.95  # break stop level as a fraction of pin_ref; None = no break stop
    terminal_factor_float: float = 1.0  # terminal proceeds x factor (sensitivity)
    time_stop_int: int = 252

    @property
    def key_str(self) -> str:
        break_str = "noBreak" if self.break_float is None else f"brk{self.break_float * 100:g}"
        term_str = "" if self.terminal_factor_float == 1.0 else f"|term{self.terminal_factor_float:g}"
        return f"M|J{self.jump_float * 100:g}|th{self.theta_float * 100:g}|W{self.window_int}|K{self.slots_int}|{break_str}{term_str}"


@dataclasses.dataclass(frozen=True)
class SCell:
    """Family S: same-calendar-month seasonality score, top N, GATED or HEDGED form."""

    horizon_str: str = "SE_1_10"  # SE_1 | SE_2_5 | SE_1_10 | SE_1_20
    n_int: int = 20
    form_str: str = "GATED"  # GATED | HEDGED
    offset_int: int = 0  # rebalance-day offset (timing-luck diagnostic only)

    @property
    def key_str(self) -> str:
        return f"S|{self.horizon_str}|N{self.n_int}|{self.form_str}|k{self.offset_int:+d}"


# horizon -> (year offsets k in 1..; minimum valid count)
HORIZON_DICT = {
    "SE_1": ((1,), 1),
    "SE_2_5": ((2, 3, 4, 5), 3),
    "SE_1_10": (tuple(range(1, 11)), 5),
    "SE_1_20": (tuple(range(1, 21)), 10),
}
HORIZON_TUPLE = ("SE_1", "SE_2_5", "SE_1_10", "SE_1_20")
S_N_TUPLE = (10, 20, 50)
S_FORM_TUPLE = ("GATED", "HEDGED")
M1_JUMP_TUPLE = (0.10, 0.15, 0.20)
M1_THETA_TUPLE = (0.006, 0.010, 0.015)
M2_WINDOW_TUPLE = (5, 10)
M2_SLOTS_TUPLE = (10, 20)
OFFSET_TUPLE = tuple(range(-10, 11))

M0_CELL = MCell()
M_SENSITIVITY_DICT = {
    "no_break_stop": dataclasses.replace(M0_CELL, break_float=None),
    "break_stop_10pct": dataclasses.replace(M0_CELL, break_float=0.90),
    "terminal_x0.99": dataclasses.replace(M0_CELL, terminal_factor_float=0.99),
}
S_ANCHOR_DICT = {form_str: SCell("SE_1_10", 20, form_str) for form_str in S_FORM_TUPLE}


def family_m_stage_dict() -> dict[str, list[tuple[tuple, MCell]]]:
    return {
        "M1_event_pin": [((jump_float, theta_float), dataclasses.replace(M0_CELL, jump_float=jump_float, theta_float=theta_float)) for jump_float in M1_JUMP_TUPLE for theta_float in M1_THETA_TUPLE],
        "M2_window_slots": [((window_int, slots_int), dataclasses.replace(M0_CELL, window_int=window_int, slots_int=slots_int)) for window_int in M2_WINDOW_TUPLE for slots_int in M2_SLOTS_TUPLE],
    }


def family_m_cells(include_sensitivities_bool: bool = True) -> list[MCell]:
    seen_dict: dict[str, MCell] = {}
    for cell_list in family_m_stage_dict().values():
        for _, cell in cell_list:
            seen_dict.setdefault(cell.key_str, cell)
    if include_sensitivities_bool:
        for cell in M_SENSITIVITY_DICT.values():
            seen_dict.setdefault(cell.key_str, cell)
    return list(seen_dict.values())


def family_s_stage_dict() -> dict[str, list[tuple[tuple, SCell]]]:
    """One stage per form: rows = horizons (ordered), cols = N (ordered)."""
    return {
        f"S_{form_str}": [((horizon_str, n_int), SCell(horizon_str, n_int, form_str)) for horizon_str in HORIZON_TUPLE for n_int in S_N_TUPLE]
        for form_str in S_FORM_TUPLE
    }


def family_s_cells() -> list[SCell]:
    return [cell for cell_list in family_s_stage_dict().values() for _, cell in cell_list]
