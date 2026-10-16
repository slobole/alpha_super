"""Station S8: shadow tracking with rules written before the shadow starts (design S8, P8).

`start_shadow` writes one `shadow_start` row per registration, BEFORE any forward data exists:
- the frozen configuration, and the start date: after both the registration and the day the row is written (only
  data that did not exist yet is clean by construction, design 4.2);
- a shadow_start row unseals its family's panel (the rule needs the bars before the start as warm-up), so S7 counts it
  as contamination, and a CANDIDATE whose family has not opened the vault is refused (S7 comes first); the providers
  return sessions from the start only;
- the expected process: a daily return series that ends before the start (the vault-inclusive series after a vault
  opening, otherwise the in-sample worst-offset series of S4), saved under results/scout/shadow/<registration>/ with
  its SHA-256 in the row;
- MinTRL against the registered benchmark Sharpe (months, stated honestly even when it is years).

`shadow_report` re-runs the frozen configuration on the latest data through a registered provider (a function of the
shadow_start row that returns the daily net returns), keeps the sessions from the start date only, and judges them
with the live pod monitor (alpha.scout.pod_health.evaluate_pod: the Cold Blood Index with thresholds calibrated on the
monitoring procedure, and the monthly CUSUM). Status: NO_DATA / TOO_EARLY / GREEN / AMBER / RED / STALE. A RED card
does not retire anything: that stays the owner's decision.

The shadow is simulated: no orders exist, so the design's parity check (frozen spec vs live decisions) applies only
once a shadow is also traded on paper. Each run rebuilds the decisions from the frozen spec on the newest panel snapshot,
so rebuild the panel first (`python -m alpha.scout panel "S&P 500"`).
"""

from __future__ import annotations

import hashlib
import importlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH, Ledger
from alpha.scout.registration import registration_rows
from alpha.stats.psr_dsr import minimum_track_record_length, sharpe_moments

SHADOW_ROOT_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "shadow"
SESSIONS_PER_MONTH_FLOAT = 21.0
# registration id -> "module:function"; the function takes the shadow_start row and returns daily net returns.
SHADOW_PROVIDER_DICT: dict[str, str] = {
    "dividend_month_premium_20261016": "alpha.scout.specs.dividend_month:shadow_daily_return_ser",
    "core5_equity_vehicle_v2_retro_20261011": "alpha.scout.specs.core5_vehicle:shadow_daily_return_ser",
    # The four BY-FDR survivors of 2026-10-16 (design A21): live configuration of the re-audition family.
    "eom_reaudition_20261002": "alpha.scout.specs.reaudition_shadow:shadow_daily_return_ser",
    "dv2_reaudition_20261002": "alpha.scout.specs.reaudition_shadow:shadow_daily_return_ser",
    "dv2_ndx_reaudition_20261002": "alpha.scout.specs.reaudition_shadow:shadow_daily_return_ser",
    "dv2_industry_etf_reaudition_20261002": "alpha.scout.specs.reaudition_shadow:shadow_daily_return_ser",
}


class ShadowRefusal(RuntimeError):
    """The shadow cannot start (unknown registration, a start before the registration, or a second start)."""


def _sha256_str(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shadow_rows(ledger: Ledger) -> dict[str, dict]:
    return {row_dict["registration_id_str"]: row_dict for row_dict in ledger.rows("shadow_start")}


def start_shadow(ledger: Ledger, registration_id_str: str, frozen_config_dict: dict, expected_daily_ser: pd.Series,
                 expected_source_str: str, start_date_str: str | None = None, benchmark_annual_sharpe_float: float = 0.0,
                 note_str: str = "", root_path: Path = SHADOW_ROOT_PATH) -> dict:
    from alpha.scout.vault import latest_grade_str

    ledger.verify()
    reg_dict = registration_rows(ledger).get(registration_id_str)
    if reg_dict is None:
        raise ShadowRefusal(f"Registration {registration_id_str!r} does not exist.")
    if registration_id_str in shadow_rows(ledger):
        raise ShadowRefusal(f"{registration_id_str} already has a shadow; a changed rule is a new registration.")
    family_id_str = reg_dict["family_id_str"]
    # *** CRITICAL*** a shadow_start row unseals the family's panel (warm-up bars), which contaminates its vault. A
    # CANDIDATE of a family whose vault is still closed must open it through S7 first.
    vault_opened_bool = any(row.get("family_id_str") == family_id_str for row in ledger.rows("vault_opening"))
    if latest_grade_str(ledger, registration_id_str) == "CANDIDATE" and not vault_opened_bool:
        raise ShadowRefusal(f"{registration_id_str} is a CANDIDATE whose family has not opened the vault: run S7 first.")
    registered_date = datetime.fromisoformat(reg_dict["utc_ts_str"]).date()
    earliest_date = max(registered_date, datetime.now(timezone.utc).date())  # forward = data that did not exist yet
    start_date = pd.Timestamp(start_date_str) if start_date_str else pd.Timestamp(earliest_date) + pd.Timedelta(days=1)
    if start_date.date() <= earliest_date:
        raise ShadowRefusal(f"The shadow must start after both the registration and today ({earliest_date}); asked {start_date.date()}.")
    expected_ser = expected_daily_ser.dropna()
    if expected_ser.empty or expected_ser.index[-1] >= start_date:
        raise ShadowRefusal("The expected process must end before the shadow start.")

    folder_path = Path(root_path) / registration_id_str
    folder_path.mkdir(parents=True, exist_ok=True)
    expected_path = folder_path / "expected.parquet"
    if expected_path.exists():
        raise ShadowRefusal(f"{expected_path} already exists; it belongs to an earlier shadow attempt and is not overwritten.")
    expected_ser.rename("expected_return_float").to_frame().to_parquet(expected_path)
    moments = sharpe_moments(expected_ser.to_numpy(dtype=float))
    min_trl_sessions_float = minimum_track_record_length(
        moments.sharpe_float, moments.skewness_float, moments.kurtosis_float, benchmark_annual_sharpe_float / np.sqrt(252.0))

    def precondition(existing_ledger: Ledger) -> dict:
        if registration_id_str in shadow_rows(existing_ledger):
            raise ShadowRefusal(f"{registration_id_str} already has a shadow; a changed rule is a new registration.")
        return {"family_id_str": reg_dict["family_id_str"]}

    return ledger.append("shadow_start", {
        "registration_id_str": registration_id_str, "frozen_config_dict": frozen_config_dict,
        "start_date_str": str(start_date.date()), "expected_source_str": expected_source_str,
        "expected_path_str": str(expected_path.relative_to(MAIN_CHECKOUT_ROOT_PATH)) if expected_path.is_relative_to(MAIN_CHECKOUT_ROOT_PATH) else str(expected_path),
        "expected_sha256_str": _sha256_str(expected_path), "expected_start_str": str(expected_ser.index[0].date()),
        "expected_end_str": str(expected_ser.index[-1].date()), "expected_annual_sharpe_float": float(moments.sharpe_float * np.sqrt(252.0)),
        "benchmark_annual_sharpe_float": float(benchmark_annual_sharpe_float),
        "min_trl_months_float": float(min_trl_sessions_float / SESSIONS_PER_MONTH_FLOAT) if np.isfinite(min_trl_sessions_float) else None,
        "kill_rule_str": "card RED when the calibrated Cold Blood Index (5% a year false RED) or the monthly CUSUM alarms",
        "note_str": note_str,
    }, precondition_fn=precondition)


def load_expected(row_dict: dict) -> pd.Series:
    path = Path(row_dict["expected_path_str"])
    path = path if path.is_absolute() else MAIN_CHECKOUT_ROOT_PATH / path
    if _sha256_str(path) != row_dict["expected_sha256_str"]:
        raise RuntimeError(f"The expected process {path} changed after the shadow started (SHA-256 mismatch).")
    return pd.read_parquet(path)["expected_return_float"]


def shadow_report(row_dict: dict, realized_daily_ser: pd.Series, as_of_date: date) -> dict:
    from alpha.scout.pod_health import ExpectedProcess, LivePodSpec, evaluate_pod

    start_date_str = row_dict["start_date_str"]
    expected = ExpectedProcess(load_expected(row_dict), row_dict["expected_source_str"], row_dict["expected_sha256_str"])
    spec = LivePodSpec(pod_id_str=row_dict["registration_id_str"], label_str=row_dict["registration_id_str"],
                       account_route_str="shadow (simulated)", monitoring_start_date_str=start_date_str,
                       expected_loader=lambda *_: expected)
    forward_ser = realized_daily_ser.loc[start_date_str:].dropna()  # *** CRITICAL*** nothing before the start is reported
    report_dict = evaluate_pod(spec, expected, forward_ser, as_of_date)
    months_float = len(forward_ser) / SESSIONS_PER_MONTH_FLOAT
    report_dict["shadow"] = {
        "start_date_str": start_date_str, "forward_sessions_int": int(len(forward_ser)), "forward_months_float": months_float,
        "min_trl_months_float": row_dict.get("min_trl_months_float"),
        "progress_float": (months_float / row_dict["min_trl_months_float"]) if row_dict.get("min_trl_months_float") else None,
        "expected_annual_sharpe_float": row_dict["expected_annual_sharpe_float"],
    }
    return report_dict


def provider_for(registration_id_str: str):
    module_str, function_str = SHADOW_PROVIDER_DICT[registration_id_str].split(":")
    return getattr(importlib.import_module(module_str), function_str)


def render_markdown(report_list: list[dict], as_of_str: str) -> str:
    line_list = [f"# Scout shadows (S8), {as_of_str}", "",
                 "Simulated forward tracking from each registration onward. RED = the calibrated Cold Blood Index or the "
                 "monthly CUSUM alarmed; retiring stays the owner's decision.", "",
                 "| Registration | Start | Status | Forward months | MinTRL months | Expected Sharpe | Cumulative | CBI now |",
                 "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for report_dict in report_list:
        shadow_dict, live_dict = report_dict["shadow"], report_dict.get("live") or {}
        min_trl = shadow_dict["min_trl_months_float"]
        line_list.append(
            f"| {report_dict['pod_id_str']} | {shadow_dict['start_date_str']} | {report_dict['status_str']} | "
            f"{shadow_dict['forward_months_float']:.1f} | {('%.0f' % min_trl) if min_trl else 'n/a'} | "
            f"{shadow_dict['expected_annual_sharpe_float']:.2f} | "
            f"{('%+.2f%%' % (100 * live_dict['cumulative_return_float'])) if live_dict else 'n/a'} | "
            f"{('%.3f' % live_dict['current_cbi_float']) if live_dict.get('current_cbi_float') is not None else 'n/a'} |")
    error_list = [r for r in report_list if r.get("error_str")]
    if error_list:
        line_list += ["", "## Errors", ""] + [f"- {r['pod_id_str']}: {r['error_str']}" for r in error_list]
    return "\n".join(line_list) + "\n"


def run_shadow_monitor(ledger: Ledger | None = None, as_of_date: date | None = None, output_dir_path: Path | None = None) -> tuple[Path, list[dict]]:
    ledger = ledger or Ledger()
    as_of_date = as_of_date or date.today()
    report_list = []
    for registration_id_str, row_dict in shadow_rows(ledger).items():
        try:
            realized_ser = provider_for(registration_id_str)(row_dict)
            report_list.append(shadow_report(row_dict, realized_ser, as_of_date))
        except Exception as error:  # one broken provider must not hide the other shadows; the row says ERROR
            report_list.append({
                "pod_id_str": registration_id_str, "status_str": "ERROR", "error_str": f"{type(error).__name__}: {error}",
                "live": {}, "shadow": {"start_date_str": row_dict["start_date_str"], "forward_sessions_int": 0, "forward_months_float": 0.0,
                                       "min_trl_months_float": row_dict.get("min_trl_months_float"), "progress_float": None,
                                       "expected_annual_sharpe_float": row_dict["expected_annual_sharpe_float"]},
            })
    output_dir_path = Path(output_dir_path or SHADOW_ROOT_PATH / "reports" / as_of_date.isoformat())
    output_dir_path.mkdir(parents=True, exist_ok=True)
    (output_dir_path / "shadow.md").write_text(render_markdown(report_list, as_of_date.isoformat()), encoding="utf-8")
    (output_dir_path / "shadow.json").write_text(json.dumps(report_list, indent=2, default=str), encoding="utf-8")
    return output_dir_path, report_list
