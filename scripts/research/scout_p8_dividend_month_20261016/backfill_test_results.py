"""P8: write the saved S5 MCPT p-values of the 27 re-auditions (P5-P7b) to the ledger as retro test_result rows, so the
program-wide BY report has them. One row per MCPT component, plus one PRIMARY gate row per registration whose p is the
largest component p (the gate needs every component to pass: an intersection-union test). Idempotent: rows that already
exist (same registration and test id, retro) are skipped.

    PYTHONUTF8=1 .venv/Scripts/python.exe scripts/research/scout_p8_dividend_month_20261016/backfill_test_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from alpha.scout.fdr_report import record_test_result
from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH, Ledger

REAUDITION_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "reaudition"
FOLDER_TO_REGISTRATION_DICT = {
    "TAA_3x": "taa_3x_reaudition_20261002", "NDX_VXN": "ndx_vxn_reaudition_20261002",
    "TAA_3x_1-N": "taa_3x_1n_reaudition_20261002", "TAA_linearity_1-N_QQQ": "taa_lin_1n_qqq_reaudition_20261002",
    "TAA_2x_1-N_QLD": "taa_2x_1n_qld_reaudition_20261002", "TAA_no-BTAL_2x_1-N_QLD": "taa_nobtal_2x_1n_qld_reaudition_20261002",
    "TAA_no-BTAL_2x_1-N_SSO": "taa_nobtal_2x_1n_sso_reaudition_20261002", "NDX_ATR": "ndx_atr_reaudition_20261002",
    "NDX_NATR20": "ndx_natr20_reaudition_20261002", "NDX_NATR20_VXN": "ndx_natr20_vxn_reaudition_20261002",
    "Compass": "compass_reaudition_20261002", "Compass_QQQ": "compass_qqq_reaudition_20261002",
    "CORE5": "core5_reaudition_20261002", "TFI": "tfi_reaudition_20261002", "Trinity": "trinity_reaudition_20261002",
    "EOM": "eom_reaudition_20261002", "Sector_IBS_VOX_IYR": "sector_ibs_vox_iyr_reaudition_20261002",
    "Dispersion_IBS_KIE_IHI_XLC": "dispersion_ibs_kie_ihi_xlc_reaudition_20261002",
    "Dispersion_IBS_KIE_IHI_XLC_SMA200": "dispersion_ibs_kie_ihi_xlc_sma200_reaudition_20261002",
    "Dispersion_IBS_KIE_IHI_SMA200": "dispersion_ibs_kie_ihi_sma200_reaudition_20261002",
    "BTC-Gold_trend": "btc_gold_trend_20261002", "BTC-Gold_static": "btc_gold_static_20261002",
    "DV2": "dv2_reaudition_20261002", "DV2_Nasdaq_100": "dv2_ndx_reaudition_20261002",
    "HPI_2-3-5_vote": "hpi_vote_reaudition_20261002", "HPI_IBS_RSI_exit": "hpi_ibs_rsi_reaudition_20261002",
    "DV2_industry_ETF": "dv2_industry_etf_reaudition_20261002",
}


def main() -> None:
    ledger = Ledger()
    done_set = {(row["registration_id_str"], row["test_id_str"]) for row in ledger.rows("test_result") if row.get("retro_bool")}
    for folder_str, registration_id_str in FOLDER_TO_REGISTRATION_DICT.items():
        summary_path = REAUDITION_PATH / folder_str / "summary.json"
        summary_dict = json.loads(summary_path.read_text(encoding="utf-8"))
        source_str = str(summary_path.relative_to(MAIN_CHECKOUT_ROOT_PATH))
        component_list = summary_dict["mcpt"]
        for component_dict in component_list:
            if (registration_id_str, "s5_mcpt_" + component_dict["name"].replace(" ", "_")) in done_set:
                continue
            record_test_result(ledger, registration_id_str, "s5_mcpt_" + component_dict["name"].replace(" ", "_"), "S5", component_dict["p"],
                               {"observed_float": component_dict["observed"], "null_95_float": component_dict["null_95"]},
                               primary_bool=False, source_str=source_str, retro_bool=True)
        gate_p_float = max(component_dict["p"] for component_dict in component_list)
        if (registration_id_str, "s5_mcpt_gate") in done_set:
            print(f"skip {registration_id_str}")
            continue
        record_test_result(ledger, registration_id_str, "s5_mcpt_gate", "S5", gate_p_float,
                           {"component_count_int": len(component_list), "grade_str": summary_dict.get("grade", "see card")},
                           primary_bool=True, source_str=source_str, retro_bool=True)
        print(f"{registration_id_str}: gate p {gate_p_float:.3f} ({len(component_list)} components, {summary_dict.get('grade', 'see card')})")
    print(f"ledger chain OK: {ledger.verify()} rows")


if __name__ == "__main__":
    main()
