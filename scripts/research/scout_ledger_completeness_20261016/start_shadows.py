"""S8 for the four BY-FDR survivors of 2026-10-16 (design A21, gap 5): EOM, DV2, DV2 Nasdaq-100, DV2 industry ETF.

The only clean evidence left for these pods is time that has not happened yet. Each gets a `shadow_start` row today,
starting 2026-10-17, with the frozen live configuration, the expected process and MinTRL:

- **Provider:** alpha.scout.specs.reaudition_shadow, the re-audition family's live configuration on the newest Norgate
  data (refuses if the spec's live configuration changes later).
- **Expected process:** the same provider's full daily history before the start, from the first non-zero day. All of
  it was seen (the re-audition used it), so it is a yardstick, not evidence. Checked here against the card's saved S4
  series (results/scout/reaudition/<pod>/bundle.pkl); the run stops if they disagree.
- **Benchmark Sharpe 0** for MinTRL, as for the first shadow (row 119): MinTRL then answers "how long until the forward
  record alone shows a positive Sharpe at 95%".
- **EOM is graded CANDIDATE,** so S7 runs first (shadow.start_shadow refuses otherwise). Its family, calendar_and_flow,
  is on the seen-after-2022 list and has retro registrations, so the vault test is CONTAMINATED by construction: the
  opening row and an INFO verdict are written, and the result is a diagnostic only (design section 10: a re-audition
  has no clean vault). This uses the family's one vault opening; any later calendar_and_flow vault would have been
  contaminated for the same reasons.

    PYTHONUTF8=1 .venv/Scripts/python.exe scripts/research/scout_ledger_completeness_20261016/start_shadows.py [--dry-run]
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_PATH))

import numpy as np
import pandas as pd

from alpha.scout.ledger import Ledger
from alpha.scout.metrics import tbill_daily_ser
from alpha.scout.shadow import shadow_rows, start_shadow
from alpha.scout.specs import reaudition_shadow
from alpha.scout.vault import latest_grade_str, open_vault, record_vault_verdict, vault_verdict

START_DATE_STR = "2026-10-17"  # the day after today (2026-10-16); the first session is Monday 2026-10-19
BUNDLE_DIR_DICT = {
    "eom_reaudition_20261002": "EOM",
    "dv2_reaudition_20261002": "DV2",
    "dv2_ndx_reaudition_20261002": "DV2_Nasdaq_100",
    "dv2_industry_etf_reaudition_20261002": "DV2_industry_ETF",
}
# Same rule = near-identical path. Norgate revises a few historical bars after the card ran (2026-10-02): EOM moves by
# ~1e-5 on some days, DV2 S&P 500 differs on 9 sessions of March 2019 (one revised stock, Sharpe 0.975 vs 0.976). A
# changed rule moves many days and the Sharpe; these thresholds catch that and pass revisions.
DIFF_DAY_FLOAT = 1e-4  # a day "differs" above this absolute daily return gap
MAX_DIFF_DAY_SHARE_FLOAT = 0.005
MIN_CORR_FLOAT = 0.9999
MAX_SHARPE_GAP_FLOAT = 0.01


def card_check(registration_id_str: str, daily_ser: pd.Series) -> str:
    bundle_path = ROOT_PATH / "results" / "scout" / "reaudition" / BUNDLE_DIR_DICT[registration_id_str] / "bundle.pkl"
    with bundle_path.open("rb") as file_obj:
        card_ser = pickle.load(file_obj)["s4"].luck_dict["live"]["worst_offset_return_ser"]
    common_index = card_ser.index.intersection(daily_ser.index)
    card_vec, new_vec = card_ser.reindex(common_index).to_numpy(), daily_ser.reindex(common_index).to_numpy()
    abs_diff_vec = np.abs(card_vec - new_vec)
    diff_day_share_float = float(np.mean(abs_diff_vec > DIFF_DAY_FLOAT))
    corr_float = float(np.corrcoef(card_vec, new_vec)[0, 1])
    card_sharpe_float, new_sharpe_float = sharpe_float(card_ser), sharpe_float(daily_ser.reindex(common_index))
    detail_str = (f"{len(common_index)}/{len(card_ser)} sessions to {card_ser.index[-1].date()}, {int(np.sum(abs_diff_vec > DIFF_DAY_FLOAT))} days "
                  f"differ > {DIFF_DAY_FLOAT:.0e} (max {np.nanmax(abs_diff_vec):.1e}), corr {corr_float:.6f}, Sharpe {card_sharpe_float:.3f} vs {new_sharpe_float:.3f}")
    if (len(common_index) < 0.99 * len(card_ser) or diff_day_share_float > MAX_DIFF_DAY_SHARE_FLOAT or corr_float < MIN_CORR_FLOAT
            or abs(card_sharpe_float - new_sharpe_float) > MAX_SHARPE_GAP_FLOAT):
        raise SystemExit(f"{registration_id_str}: provider does not reproduce the card ({detail_str}). Not starting.")
    return f"matches the card ({detail_str})"


def sharpe_float(daily_ser: pd.Series) -> float:
    return float(daily_ser.mean() / daily_ser.std(ddof=1) * np.sqrt(252.0))


def main() -> None:
    dry_run_bool = "--dry-run" in sys.argv
    ledger = Ledger()
    existing_dict = shadow_rows(ledger)
    tbill_ser = None
    for registration_id_str in BUNDLE_DIR_DICT:
        if registration_id_str in existing_dict:
            print(f"skip (shadow exists)  {registration_id_str}")
            continue
        family = reaudition_shadow.build_family(registration_id_str)
        config_dict = reaudition_shadow.live_config_dict(registration_id_str, family)
        daily_ser = reaudition_shadow.live_daily_return_ser(registration_id_str, family)
        print(f"{registration_id_str}: live {config_dict}; {card_check(registration_id_str, daily_ser)}; data to {daily_ser.index[-1].date()}")
        expected_ser = daily_ser.loc[daily_ser.ne(0).idxmax(): pd.Timestamp(START_DATE_STR) - pd.Timedelta(days=1)]
        print(f"  expected process {expected_ser.index[0].date()}..{expected_ser.index[-1].date()}, Sharpe {sharpe_float(expected_ser):.2f}")

        family_id_str = family.family_id_str
        vault_opened_bool = any(row.get("family_id_str") == family_id_str for row in ledger.rows("vault_opening"))
        if latest_grade_str(ledger, registration_id_str) == "CANDIDATE" and not vault_opened_bool:
            if tbill_ser is None:
                from alpha.scout.specs import taa_3x

                tbill_ser = tbill_daily_ser(taa_3x.load_inputs().dtb3_ser, daily_ser.index)
            verdict_dict = vault_verdict(daily_ser, daily_ser, tbill_ser.reindex(daily_ser.index).fillna(0.0), contaminated_bool=True)
            print(f"  S7 (CANDIDATE, family vault closed): {verdict_dict['verdict_str']} (if clean: {verdict_dict['verdict_if_clean_str']}); "
                  f"{verdict_dict['start_str']}..{verdict_dict['end_str']} return {verdict_dict['vault_return_float']:+.2%} vs T-bills "
                  f"{verdict_dict['tbill_return_float']:+.2%}, Sharpe {verdict_dict['vault_sharpe_float']:.2f} vs floor "
                  f"{verdict_dict['bootstrap_floor_sharpe_float']:.2f}")
            if not dry_run_bool:
                opening_dict = open_vault(ledger, registration_id_str, {"live_config_dict": config_dict},
                                          note_str="Re-audition (design section 10): no clean vault; opened only because S8 needs S7 first. Diagnostic.")
                if not opening_dict["contaminated_bool"]:
                    raise SystemExit("Expected a contaminated vault for a re-audition family; stopping before the verdict.")
                record_vault_verdict(ledger, registration_id_str, opening_dict["row_id_int"], verdict_dict)
                print(f"  vault_opening row {opening_dict['row_id_int']} and S7 verdict written")
        if dry_run_bool:
            continue
        row_dict = start_shadow(
            ledger, registration_id_str,
            {"live_config_dict": config_dict, "family_builder_str": f"alpha.scout.family:{reaudition_shadow.FAMILY_BUILDER_DICT[registration_id_str]}",
             "cost_model_str": "CostModel() (S4 default)", "capital_float": 100_000.0},
            expected_ser,
            f"alpha.scout.specs.reaudition_shadow.live_daily_return_ser('{registration_id_str}'), weights engine, $100K, "
            f"{expected_ser.index[0].date()}..{expected_ser.index[-1].date()} (seen; equals the card's S4 series)",
            start_date_str=START_DATE_STR, benchmark_annual_sharpe_float=0.0,
            note_str="BY-FDR survivor of 2026-10-16 (q 0.0275). Simulated shadow, research only; no paper or live route.",
        )
        print(f"  shadow_start row {row_dict['row_id_int']}: start {row_dict['start_date_str']}, expected Sharpe "
              f"{row_dict['expected_annual_sharpe_float']:.2f}, MinTRL {row_dict['min_trl_months_float']:.0f} months")
    print(f"ledger chain OK: {ledger.verify()} rows")


if __name__ == "__main__":
    main()
