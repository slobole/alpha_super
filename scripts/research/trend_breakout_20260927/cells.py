"""Frozen cell definitions and grids (PREREG section 4). Nothing else is added without an amendment."""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class StopSpec:
    type_str: str = "none"  # none | CH (chandelier, k x ATR20 below the highest close since entry) | PT (percent trailing)
    level_float: float = 0.0

    @property
    def label_str(self) -> str:
        if self.type_str == "none":
            return "none"
        if self.type_str == "CH":
            return f"CH-{self.level_float:g}"
        if self.type_str == "PT":
            return f"PT-{self.level_float * 100:g}"
        raise ValueError(self.type_str)

    def stop_level_vec(self, hwm_vec: np.ndarray, atr_vec: np.ndarray) -> np.ndarray:
        """Stop level in the same adjusted units as the close: HWM - k x ATR20, or (1 - x) x HWM."""
        if self.type_str == "CH":
            return hwm_vec - self.level_float * atr_vec
        if self.type_str == "PT":
            return (1.0 - self.level_float) * hwm_vec
        return np.full_like(hwm_vec, -np.inf)

    def fires_vec(self, close_vec: np.ndarray, hwm_vec: np.ndarray, atr_vec: np.ndarray) -> np.ndarray:
        """Fires at close t when Close_t <= stop level_t; an undefined ATR or HWM never fires."""
        if self.type_str == "none":
            return np.zeros(len(close_vec), dtype=bool)
        level_vec = self.stop_level_vec(hwm_vec, atr_vec)
        with np.errstate(invalid="ignore"):
            return np.isfinite(level_vec) & np.isfinite(close_vec) & (close_vec <= level_vec)


@dataclasses.dataclass(frozen=True)
class ACell:
    """Family A: the live pod L plus a per-position stop and a freed-slot policy."""

    stop: StopSpec = StopSpec()
    policy_str: str = "CASH"  # CASH | REFILL
    offset_int: int = 0  # rebalance-day offset (timing-luck diagnostic only)

    @property
    def key_str(self) -> str:
        return f"A|{self.stop.label_str}|{self.policy_str}|k{self.offset_int:+d}"


@dataclasses.dataclass(frozen=True)
class BCell:
    """Family B: daily N-day closing-high breakouts with a chandelier exit, K slots."""

    n_int: int = 100
    k_float: float = 5.0
    slots_int: int = 20
    rank_str: str = "R1"  # R1 = ROC252 / NATR20 desc | R2 = NATR20 asc
    vxn_bool: bool = False  # S-a sensitivity: slot budget scaled by the VXN scale
    regime_exit_bool: bool = False  # S-b sensitivity: sell everything when SPY <= SMA200

    @property
    def key_str(self) -> str:
        return (
            f"B|N{self.n_int}|k{self.k_float:g}|K{self.slots_int}|{self.rank_str}"
            f"|{'VXN' if self.vxn_bool else 'noVXN'}|{'RX' if self.regime_exit_bool else 'noRX'}"
        )

    @property
    def stop(self) -> StopSpec:
        return StopSpec("CH", float(self.k_float))


@dataclasses.dataclass(frozen=True)
class CCell:
    """Family C: residual momentum score in the L shell."""

    score_str: str = "RES12-1"  # RES12-1 | RES12-0 | TOT12-1
    window_int: int = 36

    @property
    def key_str(self) -> str:
        return f"C|{self.score_str}|W{self.window_int}"

    @property
    def numerator_key_str(self) -> str:
        return f"{self.score_str}_W{self.window_int}"


# ----------------------------------------------------------------------------------------------------------------------
# grids
# ----------------------------------------------------------------------------------------------------------------------
A_LEVEL_DICT = {"CH": (2.0, 3.0, 4.0, 5.0), "PT": (0.10, 0.15, 0.20, 0.25)}
A_POLICY_TUPLE = ("CASH", "REFILL")
A_ROW_TUPLE = tuple((type_str, policy_str) for type_str in ("CH", "PT") for policy_str in A_POLICY_TUPLE)
L_REFERENCE_CELL = ACell(StopSpec(), "CASH", 0)
OFFSET_TUPLE = tuple(range(-10, 11))

B0_CELL = BCell()
B1_N_TUPLE = (50, 100, 250)
B1_K_TUPLE = (3.0, 5.0, 8.0)
B2_SLOT_TUPLE = (10, 20, 30)
B2_RANK_TUPLE = ("R1", "R2")
B_SENSITIVITY_DICT = {
    "S-a_vxn": dataclasses.replace(B0_CELL, vxn_bool=True),
    "S-b_regime_exit": dataclasses.replace(B0_CELL, regime_exit_bool=True),
}

C0_CELL = CCell("RES12-1", 36)
C_CELL_TUPLE = (C0_CELL, CCell("RES12-0", 36), CCell("RES12-1", 24), CCell("TOT12-1", 36))
C0_NEIGHBOURHOOD_TUPLE = (C0_CELL, CCell("RES12-0", 36), CCell("RES12-1", 24))


def family_a_row_cells(type_str: str, policy_str: str, offset_int: int = 0) -> list[ACell]:
    return [ACell(StopSpec(type_str, level_float), policy_str, offset_int) for level_float in A_LEVEL_DICT[type_str]]


def family_a_cells() -> list[ACell]:
    return [cell for type_str, policy_str in A_ROW_TUPLE for cell in family_a_row_cells(type_str, policy_str)]


def family_b_stage_dict() -> dict[str, list[tuple[tuple, BCell]]]:
    """Each stage: list of ((row_label, col_label), cell). Rows/cols are the heatmap axes (ordered)."""
    return {
        "B1_entry_exit": [((n_int, k_float), dataclasses.replace(B0_CELL, n_int=n_int, k_float=k_float)) for n_int in B1_N_TUPLE for k_float in B1_K_TUPLE],
        "B2_slots_rank": [((rank_str, slots_int), dataclasses.replace(B0_CELL, slots_int=slots_int, rank_str=rank_str)) for rank_str in B2_RANK_TUPLE for slots_int in B2_SLOT_TUPLE],
    }


def family_b_cells() -> list[BCell]:
    seen_dict: dict[str, BCell] = {}
    for cell_list in family_b_stage_dict().values():
        for _, cell in cell_list:
            seen_dict.setdefault(cell.key_str, cell)
    for cell in B_SENSITIVITY_DICT.values():
        seen_dict.setdefault(cell.key_str, cell)
    return list(seen_dict.values())


def family_c_cells() -> list[CCell]:
    return list(C_CELL_TUPLE)
