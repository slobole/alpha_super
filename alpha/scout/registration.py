"""Station S0: registration, written before any result is seen.

A registration freezes the hypothesis, the mechanism, the search space and the
kill criteria. It is hashed and written to the ledger. It is never edited: a
change after seeing a result is a new registration with `parent_id_str`, and its
trials count toward the same family.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import date

from alpha.scout.families import validate_family_id
from alpha.scout.ledger import Ledger, canonical_json_str

HYPOTHESIS_CLASS_TUPLE = ("E", "X", "W")  # event, cross-sectional rebalance, weights / TAA
MAX_GRID_SIZE_WITHOUT_JUSTIFICATION_INT = 200


@dataclass(frozen=True)
class Registration:
    registration_id_str: str
    family_id_str: str
    hypothesis_str: str
    mechanism_str: str
    expected_sign_and_location_str: str
    hypothesis_class_str: str
    universe_str: str
    horizon_str: str
    schedule_str: str
    execution_str: str
    param_grid_dict: dict = field(default_factory=dict)
    primary_metric_str: str = ""
    kill_criteria_str: str = ""
    source_str: str = ""
    source_published_date_str: str | None = None
    retro_bool: bool = False
    prior_trials_int: int | None = None
    parent_id_str: str | None = None
    grid_justification_str: str = ""

    @property
    def grid_size_int(self) -> int:
        if not self.param_grid_dict:
            return 1
        return math.prod(len(value_tuple) for value_tuple in self.param_grid_dict.values())

    def grid_config_list(self) -> list[dict]:
        """Every configuration in the registered grid, in a stable order."""
        name_list = sorted(self.param_grid_dict)
        value_list_list = [list(self.param_grid_dict[name_str]) for name_str in name_list]
        return [dict(zip(name_list, combo_tuple)) for combo_tuple in itertools.product(*value_list_list)]

    def payload_dict(self) -> dict:
        payload_dict = asdict(self)
        payload_dict["param_grid_dict"] = {
            name_str: list(value_tuple) for name_str, value_tuple in sorted(self.param_grid_dict.items())
        }
        return payload_dict

    def hash_str(self) -> str:
        return hashlib.sha256(canonical_json_str(self.payload_dict()).encode("utf-8")).hexdigest()

    def validate(self) -> None:
        if not re.fullmatch(r"[a-z0-9_]+", self.registration_id_str):
            raise ValueError("registration_id_str must be a non-empty slug of lowercase ASCII letters, digits and underscores.")
        validate_family_id(self.family_id_str)
        if self.hypothesis_class_str not in HYPOTHESIS_CLASS_TUPLE:
            raise ValueError(f"hypothesis_class_str must be one of {HYPOTHESIS_CLASS_TUPLE}.")
        required_text_dict = {
            "hypothesis_str": self.hypothesis_str,
            "mechanism_str": self.mechanism_str,
            "expected_sign_and_location_str": self.expected_sign_and_location_str,
            "universe_str": self.universe_str,
            "primary_metric_str": self.primary_metric_str,
            "kill_criteria_str": self.kill_criteria_str,
            "source_str": self.source_str,
        }
        missing_list = [name_str for name_str, text_str in required_text_dict.items() if not text_str.strip()]
        if missing_list:
            raise ValueError(f"Registration is missing required text: {missing_list}.")
        for name_str, value_tuple in self.param_grid_dict.items():
            if len(value_tuple) == 0:
                raise ValueError(f"Parameter {name_str!r} has no values.")
        if self.grid_size_int > MAX_GRID_SIZE_WITHOUT_JUSTIFICATION_INT and not self.grid_justification_str.strip():
            raise ValueError(
                f"Grid has {self.grid_size_int} configurations; more than "
                f"{MAX_GRID_SIZE_WITHOUT_JUSTIFICATION_INT} needs grid_justification_str."
            )
        if self.retro_bool and self.prior_trials_int is None:
            raise ValueError("A retro registration must state prior_trials_int.")
        if self.prior_trials_int is not None and self.prior_trials_int < 0:
            raise ValueError("prior_trials_int must be >= 0.")
        if self.source_published_date_str is not None:
            date.fromisoformat(self.source_published_date_str)


def registration_rows(ledger: Ledger) -> dict[str, dict]:
    return {row_dict["registration_id_str"]: row_dict for row_dict in ledger.rows("registration")}


def register(ledger: Ledger, registration: Registration) -> dict:
    """Validate and write a registration. Refuses duplicates and orphan or cross-family parents."""
    registration.validate()

    def precondition(existing_ledger: Ledger) -> None:
        existing_dict = registration_rows(existing_ledger)
        if registration.registration_id_str in existing_dict:
            raise ValueError(
                f"Registration {registration.registration_id_str!r} already exists. Registrations are never edited; "
                "register a new id with parent_id_str instead."
            )
        if registration.parent_id_str is not None:
            parent_row_dict = existing_dict.get(registration.parent_id_str)
            if parent_row_dict is None:
                raise ValueError(f"Parent registration {registration.parent_id_str!r} does not exist.")
            if parent_row_dict["family_id_str"] != registration.family_id_str:
                raise ValueError("A child registration must stay in its parent's family.")

    payload_dict = registration.payload_dict()
    payload_dict["registration_hash_str"] = registration.hash_str()
    payload_dict["grid_size_int"] = registration.grid_size_int
    return ledger.append("registration", payload_dict, precondition_fn=precondition)
