"""Explicit synthetic close composition for the local visual preview only."""

from dataclasses import replace
from decimal import Decimal
import hashlib
import xml.etree.ElementTree as ElementTree

from alpha.live.dashboard_v4.pod_holdings import parse_close_holdings_dict


def attach_demo_holdings(provider_obj):
    """Exercise the real parser with invented, clearly labelled demo XML.

    This is not a broker export or production-format acceptance evidence.
    Synthetic holdings divide the synthetic invested total; no production marks
    or quantities are read, and no real-account source can be reached here.
    """
    def get_close_holdings_dict(nav_row_obj, *, query_name_str, as_of_ts):
        account_row_list = [row_dict for row_dict in provider_obj.row_list
            if row_dict["account_route_str"] == nav_row_obj.account_route_str]
        if len(account_row_list) != 1:
            return {"available_bool": False, "reason_str": "Demo holdings unavailable"}
        account_row_dict = account_row_list[0]
        cash_decimal = Decimal(str(account_row_dict["eod_snapshot_dict"]["cash_float"]))
        invested_decimal = nav_row_obj.closing_nav_decimal - cash_decimal
        daily_bool = account_row_dict["pod_id_str"] in {"demo_1_0", "demo_1_1"}
        symbol_tuple = ("ADBE", "AMD", "CRM", "CSCO", "LOW", "MRK", "NKE", "QCOM", "TGT", "UNH") if daily_bool else ("DBC", "GLD", "TQQQ", "UUP")
        share_tuple = (9, 31, 17, 88, 19, 40, 58, 28, 39, 9) if daily_bool else (212, 13, 58, 142)
        fraction_tuple = (Decimal(".1"),) * 10 if daily_bool else tuple(map(Decimal, (".45", ".3", ".15", ".1")))
        root_obj = ElementTree.Element("FlexQueryResponse", queryName=query_name_str)
        statements_obj = ElementTree.SubElement(root_obj, "FlexStatements", count="1")
        statement_obj = ElementTree.SubElement(statements_obj, "FlexStatement", accountId=nav_row_obj.account_route_str)
        ElementTree.SubElement(statement_obj, "AccountInformation", accountId=nav_row_obj.account_route_str, currency="USD")
        ElementTree.SubElement(statement_obj, "ChangeInNAV", **nav_row_obj.attribute_dict)
        positions_obj = ElementTree.SubElement(statement_obj, "OpenPositions")
        allocated_decimal = Decimal(0)
        for index_int, (symbol_str, shares_int, fraction_decimal) in enumerate(zip(symbol_tuple, share_tuple, fraction_tuple)):
            value_decimal = (invested_decimal * fraction_decimal).quantize(Decimal(".01")) if index_int < len(symbol_tuple) - 1 else invested_decimal - allocated_decimal
            allocated_decimal += value_decimal
            ElementTree.SubElement(positions_obj, "OpenPosition", accountId=nav_row_obj.account_route_str,
                reportDate=nav_row_obj.market_date_str.replace("-", ""), symbol=symbol_str, conid=str(1000 + index_int),
                currency="USD", assetCategory="STK", levelOfDetail="SUMMARY", multiplier="1", fxRateToBase="1",
                position=str(shares_int), markPrice=str(value_decimal / shares_int), positionValue=str(value_decimal))
        raw_xml_str = ElementTree.tostring(root_obj, encoding="unicode")
        synthetic_nav_obj = replace(nav_row_obj, source_checksum_str=hashlib.sha256(raw_xml_str.encode()).hexdigest())
        return parse_close_holdings_dict(raw_xml_str, synthetic_nav_obj, query_name_str=query_name_str, as_of_ts=as_of_ts)

    provider_obj.get_close_holdings_dict = get_close_holdings_dict
