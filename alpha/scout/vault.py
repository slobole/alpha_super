"""Station S7: the vault, opened once per family and only through this module (design sections 4.2 and S7, P8).

The seal itself lives in the data layer (alpha/scout/panel.py): a panel handed to research code ends on 2022-12-30
unless the caller passes the id of a `vault_opening` row for that family from the verified ledger chain. This module is
the only code that writes such a row, and it refuses to unless:

1. the registration exists and its latest `grade` station verdict is CANDIDATE (S0-S6 passed). A WATCHLIST or
   REJECTED family never opens the vault: it is tracked forward by S8 instead, and its vault stays clean;
2. no vault_opening row exists for the family yet (a second opening is refused, whatever the first one found).

The opening row is written BEFORE any vault data is read, with the frozen configuration and the contamination verdict:
the vault test is CONTAMINATED when the idea's source was published on or after the seal, when the family is on the
seen-after-2022 list (design 4.2, as of 2026-10-16), or when the family has a retro registration (a study run on full
history). A contaminated result is a diagnostic only; promotion then waits for S8 forward evidence.

Verdict (clean vault): PASS if the frozen configuration's net return beats T-bills over the vault period AND its vault
Sharpe is at or above the 5th percentile of the in-sample stationary-bootstrap Sharpe for windows of the same length.
A clean FAIL is final for the family.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from alpha.scout.ledger import Ledger
from alpha.scout.panel import VAULT_SEAL_STR
from alpha.scout.registration import registration_rows
from alpha.stats.bootstrap import stationary_bootstrap_path_mat

BOOTSTRAP_PATH_COUNT_INT = 10_000
BOOTSTRAP_BLOCK_LENGTH_FLOAT = 21.0
PERCENTILE_FLOAT = 5.0
# Families the owner (or a study for the owner) has already examined on post-2022 data, with the evidence.
SEEN_AFTER_SEAL_DICT: dict[str, str] = {
    "us_equity_short_term_reversal": "DV2 / HPI / QPI studies and re-auditions on full history (2026-09/10)",
    "etf_short_term_reversal": "DV2 industry ETF, sector IBS, dispersion IBS (2026-09/10)",
    "equity_cross_sectional_momentum": "NDX pods live since 2026-04; momentum decision studies (2026-10)",
    "tactical_asset_allocation": "TAA 3x live since 2026-04; TAA variants",
    "time_series_trend_and_breakout": "CORE5, trend/breakout and megacap studies (2026-09/10)",
    "calendar_and_flow": "EOM flow (2026-09), bond calendar flow (2026-10-16), FX hours (2026-10-16)",
    "formulaic_alpha_mining": "Alpha101 (2026-09-28), RangerZ (2026-10-12)",
    "index_and_listing_events": "IPO / ATH and post-split tests (2026-09-28), S&P inclusion fade (2026-09-26)",
    "macro_regime_allocation": "Inflation Compass and TFI re-auditions and fixes (2026-09/10)",
    "optimized_low_risk_allocation": "Trinity re-audition (2026-10-02)",
    "volatility_premium_and_hedges": "VIXM / BTAL / crisis pods (2026-09)",
    "merger_arbitrage": "merger arbitrage study (2026-09-27)",
    "index_drawdown_dip_buying": "NDX dip ladder 1988-2026 (2026-10-16)",
}


class VaultRefusal(RuntimeError):
    """The vault cannot be opened for this registration (wrong grade, or the family already opened it)."""


def latest_grade_str(ledger: Ledger, registration_id_str: str) -> str | None:
    grade_str = None
    for row_dict in ledger.rows("station_verdict"):
        if row_dict.get("registration_id_str") == registration_id_str and row_dict.get("station_str") == "grade":
            grade_str = row_dict["verdict_str"]
    return grade_str


def contamination(ledger: Ledger, registration_id_str: str) -> dict:
    registration_dict = registration_rows(ledger)
    reg_dict = registration_dict[registration_id_str]
    family_id_str = reg_dict["family_id_str"]
    reason_list = []
    published_str = reg_dict.get("source_published_date_str")
    if published_str and date.fromisoformat(published_str) >= date.fromisoformat(VAULT_SEAL_STR):
        reason_list.append(f"source published {published_str}, on or after the seal {VAULT_SEAL_STR}")
    if family_id_str in SEEN_AFTER_SEAL_DICT:
        reason_list.append(f"family seen after the seal: {SEEN_AFTER_SEAL_DICT[family_id_str]}")
    retro_id_list = [rid for rid, row in registration_dict.items() if row["family_id_str"] == family_id_str and row.get("retro_bool")]
    if retro_id_list:
        reason_list.append(f"family has retro registrations (full-history studies): {', '.join(retro_id_list[:5])}")
    shadow_id_list = [row["registration_id_str"] for row in ledger.rows("shadow_start") if row.get("family_id_str") == family_id_str]
    if shadow_id_list:
        reason_list.append(f"family has shadows (S8 unseals the panel for warm-up): {', '.join(shadow_id_list[:5])}")
    return {"family_id_str": family_id_str, "contaminated_bool": bool(reason_list), "reason_list": reason_list}


def open_vault(ledger: Ledger, registration_id_str: str, frozen_config_dict: dict, note_str: str = "") -> dict:
    """Write the family's one vault_opening row (before any vault data is read) and return it; its row_id_int is the
    token for alpha.scout.panel.load_panel(..., vault_opening_row_id_int=..., family_id_str=...)."""
    ledger.verify()
    if registration_id_str not in registration_rows(ledger):
        raise VaultRefusal(f"Registration {registration_id_str!r} does not exist.")
    grade_str = latest_grade_str(ledger, registration_id_str)
    if grade_str != "CANDIDATE":  # exact: "CANDIDATE (S3 pending)" has not finished S0-S6
        raise VaultRefusal(f"{registration_id_str} is graded {grade_str!r}; only a CANDIDATE (S0-S6 passed) opens the vault.")
    contamination_dict = contamination(ledger, registration_id_str)
    family_id_str = contamination_dict["family_id_str"]

    def precondition(existing_ledger: Ledger) -> None:
        for row_dict in existing_ledger.rows("vault_opening"):
            if row_dict.get("family_id_str") == family_id_str:
                raise VaultRefusal(f"Family {family_id_str!r} already opened the vault (row {row_dict['row_id_int']}); a second opening is refused.")

    return ledger.append("vault_opening", {
        "registration_id_str": registration_id_str, "family_id_str": family_id_str, "seal_str": VAULT_SEAL_STR,
        "frozen_config_dict": frozen_config_dict, "grade_at_opening_str": grade_str,
        "contaminated_bool": contamination_dict["contaminated_bool"], "contamination_reason_list": contamination_dict["reason_list"],
        "criteria_str": (f"PASS if net return > T-bills over the vault and vault Sharpe >= the {PERCENTILE_FLOAT:.0f}th percentile of "
                         "the in-sample stationary-bootstrap Sharpe for windows of the same length; a clean FAIL is final."),
        "note_str": note_str,
    }, precondition_fn=precondition)


def _sharpe_vec(path_mat: np.ndarray) -> np.ndarray:
    std_vec = path_mat.std(axis=1, ddof=1)
    return np.where(std_vec > 0, path_mat.mean(axis=1) / np.where(std_vec > 0, std_vec, 1.0) * np.sqrt(252.0), 0.0)


def vault_verdict(vault_daily_ser: pd.Series, in_sample_daily_ser: pd.Series, tbill_daily_ser: pd.Series, contaminated_bool: bool,
                  random_seed_int: int = 0) -> dict:
    vault_ser = vault_daily_ser.loc[VAULT_SEAL_STR:].dropna()
    if len(vault_ser) < 21:
        raise ValueError("The vault period has fewer than 21 sessions.")
    in_sample_ser = in_sample_daily_ser.loc[: pd.Timestamp(VAULT_SEAL_STR) - pd.Timedelta(days=1)].dropna()
    # Warm-up zeros before the first position would dilute the bootstrap and lower the floor.
    in_sample_vec = in_sample_ser.loc[in_sample_ser.ne(0).idxmax():].to_numpy(dtype=float)
    tbill_vec = tbill_daily_ser.reindex(vault_ser.index).fillna(0.0).to_numpy(dtype=float)
    vault_vec = vault_ser.to_numpy(dtype=float)
    vault_return_float = float(np.prod(1.0 + vault_vec) - 1.0)
    tbill_return_float = float(np.prod(1.0 + tbill_vec) - 1.0)
    vault_sharpe_float = float(_sharpe_vec(vault_vec[None, :])[0])
    path_mat = stationary_bootstrap_path_mat(in_sample_vec, BOOTSTRAP_PATH_COUNT_INT, BOOTSTRAP_BLOCK_LENGTH_FLOAT, len(vault_vec), random_seed_int)
    floor_float = float(np.percentile(_sharpe_vec(path_mat), PERCENTILE_FLOAT))
    beats_tbill_bool = vault_return_float > tbill_return_float
    above_floor_bool = vault_sharpe_float >= floor_float
    clean_verdict_str = "PASS" if (beats_tbill_bool and above_floor_bool) else "FAIL"
    return {
        "verdict_str": "CONTAMINATED" if contaminated_bool else clean_verdict_str,
        "verdict_if_clean_str": clean_verdict_str, "contaminated_bool": bool(contaminated_bool),
        "start_str": str(vault_ser.index[0].date()), "end_str": str(vault_ser.index[-1].date()), "sessions_int": int(len(vault_vec)),
        "vault_return_float": vault_return_float, "tbill_return_float": tbill_return_float, "beats_tbill_bool": bool(beats_tbill_bool),
        "vault_sharpe_float": vault_sharpe_float, "in_sample_sharpe_float": float(_sharpe_vec(in_sample_vec[None, :])[0]),
        "bootstrap_floor_sharpe_float": floor_float, "above_floor_bool": bool(above_floor_bool),
    }


def record_vault_verdict(ledger: Ledger, registration_id_str: str, vault_opening_row_id_int: int, verdict_dict: dict) -> dict:
    """One S7 verdict per family, tied to the family's vault_opening row (which must be this registration's)."""
    from alpha.scout.fdr_report import record_station_verdict

    opening_dict = next((row for row in ledger.rows("vault_opening") if row["row_id_int"] == vault_opening_row_id_int), None)
    if opening_dict is None or opening_dict.get("registration_id_str") != registration_id_str:
        raise VaultRefusal(f"Row {vault_opening_row_id_int} is not {registration_id_str}'s vault opening.")
    if any(row.get("station_str") == "S7" and row.get("registration_id_str") in
           {r["registration_id_str"] for r in ledger.rows("vault_opening") if r.get("family_id_str") == opening_dict["family_id_str"]}
           for row in ledger.rows("station_verdict")):
        raise VaultRefusal(f"Family {opening_dict['family_id_str']!r} already has its vault verdict.")

    detail_str = (f"{verdict_dict['start_str']}..{verdict_dict['end_str']}: return {verdict_dict['vault_return_float']:+.2%} vs T-bills "
                  f"{verdict_dict['tbill_return_float']:+.2%}; Sharpe {verdict_dict['vault_sharpe_float']:.2f} vs floor "
                  f"{verdict_dict['bootstrap_floor_sharpe_float']:.2f}")
    verdict_str = verdict_dict["verdict_str"]
    row_verdict_str = "INFO" if verdict_str == "CONTAMINATED" else verdict_str
    return record_station_verdict(ledger, registration_id_str, "S7",
                                  [("vault test", row_verdict_str, f"{verdict_str}: {detail_str} (opening row {vault_opening_row_id_int})")])
