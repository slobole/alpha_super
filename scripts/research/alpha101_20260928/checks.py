"""C_BIL reproduction of the PREREG section-0 table and the tests_and_checks.json ledger (research only)."""

from __future__ import annotations

import datetime as dt

import numpy as np

from alpha101_20260928 import common

SECTION0_TABLE_DICT = {
    "C_BIL": {"G-P1": 0.728, "G-P2": 1.334, "G-P3": 1.437, "G-FULL": (1.365, -0.117), "G-LONG": (1.226, -0.143)},
    "G3": {"G-P1": 0.689, "G-P2": 1.303, "G-P3": 1.257, "G-FULL": (1.288, -0.141), "G-LONG": (1.167, -0.153)},
}


def update_checks(section_str: str, payload_obj) -> dict:
    checks_dict = common.read_json("tests_and_checks.json", {}) or {}
    checks_dict[section_str] = payload_obj
    checks_dict["updated_at"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M +03:00")
    common.write_json("tests_and_checks.json", checks_dict)
    return checks_dict


def control_series() -> dict:
    taa_ser = common.load_taa_ser()
    bil_ser = common.load_bil_ret_ser()
    spy_ser = common.load_spy_tr_ret_ser()
    return {"taa": taa_ser, "bil": bil_ser, "spy": spy_ser, "L_engine": common.load_l_ret_ser("engine"), "L_stress": common.load_l_ret_ser("stress"), "dv2": common.load_dv2_ser()}


def check_cbil() -> dict:
    series_dict = control_series()
    controls = common.control_books(series_dict["taa"], series_dict["L_engine"], series_dict["bil"], series_dict["spy"])
    report_dict = {"computed": {}, "section0": SECTION0_TABLE_DICT, "max_abs_sharpe_diff": 0.0, "max_abs_dd_diff_pp": 0.0}
    for control_str, table_dict in SECTION0_TABLE_DICT.items():
        report_dict["computed"][control_str] = {}
        for block_str, expected_obj in table_dict.items():
            metric = controls[control_str][block_str]
            expected_sharpe = expected_obj[0] if isinstance(expected_obj, tuple) else expected_obj
            report_dict["computed"][control_str][block_str] = {"sharpe": metric["sharpe"], "max_dd": metric["max_dd"], "cagr": metric["cagr"]}
            report_dict["max_abs_sharpe_diff"] = max(report_dict["max_abs_sharpe_diff"], abs(metric["sharpe"] - expected_sharpe))
            if isinstance(expected_obj, tuple):
                report_dict["max_abs_dd_diff_pp"] = max(report_dict["max_abs_dd_diff_pp"], abs(metric["max_dd"] - expected_obj[1]) * 100)
    report_dict["passed_bool"] = bool(report_dict["max_abs_sharpe_diff"] <= 0.0006 and report_dict["max_abs_dd_diff_pp"] <= 0.06)
    report_dict["dv2_slot"] = common.candidate_book_blocks(series_dict["taa"], series_dict["L_engine"], series_dict["dv2"])
    report_dict["C_BIL_stress"] = common.candidate_book_blocks(series_dict["taa"], series_dict["L_stress"], series_dict["bil"])
    update_checks("c_bil_reproduction", report_dict)
    common.log_progress(f"C_BIL reproduction: max |Sharpe diff| {report_dict['max_abs_sharpe_diff']:.5f}, max |DD diff| {report_dict['max_abs_dd_diff_pp']:.3f} pp, passed {report_dict['passed_bool']}")
    return report_dict


def record_unit_tests(passed_int: int, failed_int: int, skipped_int: int, output_str: str) -> None:
    update_checks("unit_tests", {"passed_int": passed_int, "failed_int": failed_int, "skipped_int": skipped_int, "tail": output_str[-1500:]})


def as_float(value_obj) -> float:
    return float(value_obj) if value_obj is not None and np.isfinite(value_obj) else float("nan")
