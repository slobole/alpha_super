"""S8 provider for re-audited pods: the live configuration of a re-audition family, re-run on the newest data.

The four pods whose S5 MCPT survives the program-wide BY-FDR (2026-10-16 report: EOM, DV2, DV2 Nasdaq-100, DV2 industry
ETF) are tracked forward from 2026-10-17 (design A21). Each is re-run through the same family runner its re-audition
card used (alpha/scout/family.py), at the family's live configuration and the default cost model (`CostModel()`, as in
S4), so the forward returns and the expected process come from one code path.

*** CRITICAL*** the shadow_start row freezes the configuration. If the spec's live configuration later changes, the
provider refuses (a changed rule is a new registration), rather than silently tracking a different rule.

Research only: no strategy module, no live path and no order is touched. DV2 and EOM load their data from Norgate
directly (not from a sealed panel), as their re-auditions did.
"""

from __future__ import annotations

import json

import pandas as pd

from alpha.scout import family as family_module

# registration id -> family builder name in alpha.scout.family
FAMILY_BUILDER_DICT: dict[str, str] = {
    "eom_reaudition_20261002": "eom_family",
    "dv2_reaudition_20261002": "dv2_family",
    "dv2_ndx_reaudition_20261002": "dv2_ndx_family",
    "dv2_industry_etf_reaudition_20261002": "dv2_industry_family",
}


def _jsonable(config_dict: dict) -> dict:
    """The ledger form of a configuration (tuples become lists, keys sorted)."""
    return json.loads(json.dumps(config_dict, sort_keys=True))


def build_family(registration_id_str: str):
    return getattr(family_module, FAMILY_BUILDER_DICT[registration_id_str])()


def live_config_dict(registration_id_str: str, family=None) -> dict:
    family = family or build_family(registration_id_str)
    return _jsonable(family.live_config_dict)


def live_daily_return_ser(registration_id_str: str, family=None) -> pd.Series:
    family = family or build_family(registration_id_str)
    return family.run_config(family.live_config_dict).daily_return_ser


def shadow_daily_return_ser(shadow_row_dict: dict) -> pd.Series:
    registration_id_str = shadow_row_dict["registration_id_str"]
    family = build_family(registration_id_str)
    frozen_dict = _jsonable(shadow_row_dict["frozen_config_dict"]["live_config_dict"])
    current_dict = live_config_dict(registration_id_str, family)
    if frozen_dict != current_dict:
        raise RuntimeError(f"{registration_id_str}: the spec's live configuration {current_dict} differs from the frozen "
                           f"{frozen_dict}; a changed rule is a new registration and a new shadow.")
    return live_daily_return_ser(registration_id_str, family).loc[shadow_row_dict["start_date_str"]:]
