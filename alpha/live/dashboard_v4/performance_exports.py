"""In-memory, selected-period exports of canonical saved reporting facts."""

import csv
from decimal import Decimal, ROUND_HALF_UP
from io import StringIO
import math

from alpha.live.investor_report import build_investor_snapshot_dict, render_investor_pdf_bytes


def _text_str(value_obj):
    if value_obj is None:
        return ""
    if not isinstance(value_obj, str):
        raise ValueError("Invalid report text")
    # CSV quoting protects delimiters, not spreadsheet formula execution.
    # Only text gets this guard; genuine negative numeric facts remain numeric.
    stripped_str = value_obj.lstrip(" \t\r\n\ufeff")
    if stripped_str.startswith(("=", "+", "-", "@")) or value_obj.startswith(("\t", "\r", "\n")):
        return "'" + value_obj
    return value_obj


def _number_str(value_obj, *, percent_bool=False):
    if value_obj is None:
        return ""
    if type(value_obj) not in {int, float} or not math.isfinite(value_obj):
        raise ValueError("Invalid report number")
    # Presentation only: money has 2 decimals and return percentages have 6,
    # rounded half up. The canonical report and its hash retain full precision.
    number_decimal = Decimal(str(value_obj)) * (100 if percent_bool else 1)
    number_decimal = number_decimal.quantize(Decimal(".000001") if percent_bool else Decimal(".01"), rounding=ROUND_HALF_UP)
    if number_decimal == 0:
        number_decimal = abs(number_decimal)
    return format(number_decimal, ".6f" if percent_bool else ".2f")


def _flag_str(value_obj):
    if value_obj is None:
        return ""
    if type(value_obj) is not bool:
        raise ValueError("Invalid report flag")
    return "Yes" if value_obj else "No"


def export_performance_csv_str(report_dict, *, level_str="portfolio"):
    """Export summary facts for the selected tab; blanks mean unavailable.

    One portfolio row or one row per measured Pod period. Preserve membership
    dates and canonical withholding. No account IDs, raw source fields, paths,
    or operational diagnostics enter this explicit column allowlist.
    """
    if level_str not in {"portfolio", "pods"}:
        raise ValueError("Unknown performance level")
    common_header_list = ["Selected from", "Selected to", "Currency", "Source", "Report status",
        "Demo", "Report hash", "Generated at"]
    common_value_list = [_text_str(report_dict.get(field_str)) for field_str in (
        "requested_from_date_str", "requested_to_date_str", "base_currency_str")]
    common_value_list += ["Saved IBKR account facts", _text_str(report_dict.get("status_str")),
        _flag_str(report_dict.get("is_demo_bool")), _text_str(report_dict.get("report_hash_str")),
        _text_str(report_dict.get("generated_at_str"))]
    header_list = ["Name", "Measured from", "Measured to", "Start value", "End value", "Profit / loss", "Return TWR (%)"]
    if level_str == "portfolio":
        header_list += ["Capital movements", "Linking adjustments", "Membership movements"]
        source_list = [report_dict]
    else:
        header_list += ["Max drawdown (%)", "Period-end drawdown (%)"]
        source_list = report_dict["strategy_list"]
    header_list += ["Return method", "Coverage complete", "Flow coverage complete"] + common_header_list
    output_obj = StringIO(newline="")
    writer_obj = csv.writer(output_obj, lineterminator="\r\n")
    writer_obj.writerow(header_list)
    for source_dict in source_list:
        portfolio_bool = level_str == "portfolio"
        value_list = [_text_str("Portfolio" if portfolio_bool else source_dict.get("display_name_str")),
            _text_str(source_dict.get("opening_date_str" if portfolio_bool else "from_date_str")),
            _text_str(source_dict.get("closing_date_str" if portfolio_bool else "to_date_str"))]
        value_list += [_number_str(source_dict.get(field_str)) for field_str in ("opening_nav_float", "closing_nav_float", "pnl_float")]
        value_list.append(_number_str(source_dict.get("twr_float"), percent_bool=True))
        if portfolio_bool:
            value_list += [_number_str(source_dict.get(field_str)) for field_str in (
                "capital_movement_float", "linking_adjustment_float", "scope_movement_float")]
        else:
            performance_dict = source_dict.get("performance_dict") or {}
            value_list += [_number_str(performance_dict.get(field_str), percent_bool=True)
                for field_str in ("max_drawdown_float", "current_drawdown_float")]
        value_list += [_text_str(source_dict.get("twr_method_str")),
            _flag_str(source_dict.get("coverage_complete_bool")), _flag_str(source_dict.get("flows_complete_bool"))]
        writer_obj.writerow(value_list + common_value_list)
    return output_obj.getvalue()


def export_performance_pdf_bytes(report_dict):
    """Preserve the existing public-field allowlist and DRAFT/FINAL policy."""
    return render_investor_pdf_bytes(build_investor_snapshot_dict(report_dict))
