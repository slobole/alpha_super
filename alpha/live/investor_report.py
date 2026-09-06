"""In-memory investor PDF, derived only from the common financial projection.

This is a report renderer, not a second accounting engine. The public snapshot
is an allowlist: no broker account IDs, operator paths, commands, raw logs,
query identifiers, access tokens or unscoped sync details enter the document.
"""

from datetime import UTC, datetime
from io import BytesIO
import json
from xml.sax.saxutils import escape

from alpha.live.client_reporting import content_hash_str


PUBLIC_REPORT_FIELD_TUPLE = (
    "client_name_str", "base_currency_str", "fee_basis_str", "requested_from_date_str",
    "requested_to_date_str", "opening_date_str", "closing_date_str", "opening_nav_float",
    "closing_nav_float", "capital_movement_float", "linking_adjustment_float",
    "scope_movement_float", "pnl_float", "twr_float", "twr_method_str",
    "coverage_complete_bool", "flows_complete_bool", "generated_at_str", "method_version_str",
    "report_hash_str", "scope_hash_str", "limitations_list",
    "is_demo_bool", "client_twr_configured_bool", "twr_method_id_str",
)
PUBLIC_STRATEGY_FIELD_TUPLE = (
    "display_name_str", "from_date_str", "to_date_str", "opening_nav_float", "closing_nav_float",
    "pnl_float", "twr_float", "coverage_complete_bool", "flows_complete_bool",
)
PUBLIC_BENCHMARK_FIELD_TUPLE = (
    "status_str", "label_str", "method_str", "symbol_str", "adjustment_str", "currency_str",
    "is_demo_bool", "from_date_str", "to_date_str", "baseline_date_str", "end_price_date_str",
    "return_float", "difference_pp_float", "manifest_hash_str", "price_hash_str", "snapshot_date_str",
    "reason_str", "basis_str",
)


def build_investor_snapshot_dict(report_dict):
    snapshot_dict = {key_str: report_dict[key_str] for key_str in PUBLIC_REPORT_FIELD_TUPLE}
    snapshot_dict["strategy_list"] = [
        {**{key_str: strategy_dict[key_str] for key_str in PUBLIC_STRATEGY_FIELD_TUPLE},
         "benchmark_dict": {key_str: strategy_dict["benchmark_dict"].get(key_str) for key_str in PUBLIC_BENCHMARK_FIELD_TUPLE}}
        for strategy_dict in report_dict["strategy_list"]
    ]
    snapshot_dict["source_checksum_list"] = sorted({source_dict["checksum_str"] for source_dict in report_dict["source_list"]})
    # Owner-approved policy: finality covers the verified facts displayed, not
    # an unavailable unconfigured consolidated TWR. An explicitly configured
    # client method must have a valid result before that report can be FINAL.
    final_evidence_bool = (
        report_dict["coverage_complete_bool"] is True
        and report_dict["flows_complete_bool"] is True
        and (not report_dict["client_twr_configured_bool"] or report_dict["twr_float"] is not None)
        and bool(report_dict["strategy_list"])
        and all(
            strategy_dict["coverage_complete_bool"] is True
            and strategy_dict["flows_complete_bool"] is True
            and strategy_dict["twr_float"] is not None
            for strategy_dict in report_dict["strategy_list"]
        )
    )
    snapshot_dict["document_status_str"] = (
        "demonstration" if report_dict["is_demo_bool"] else "final" if final_evidence_bool else "draft"
    )
    snapshot_dict = json.loads(json.dumps(snapshot_dict, allow_nan=False))
    snapshot_dict["renderer_version_str"] = "investor_pdf_v6"
    # Issued-document identity includes the printed issuance time. The separate
    # accounting report_hash stays stable across refreshes of unchanged facts.
    # This is a hash of the issued snapshot, not a hash of PDF bytes.
    snapshot_dict["document_hash_str"] = content_hash_str(snapshot_dict)
    return snapshot_dict


def _money_str(value_obj):
    return "Not available" if value_obj is None else f"{'-' if value_obj < 0 else ''}${abs(value_obj):,.2f}"


def _return_str(value_obj):
    return "Not available" if value_obj is None else f"{value_obj * 100:,.2f}%"


def render_investor_pdf_bytes(snapshot_dict):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether

    ink_color = colors.HexColor("#16181d")
    muted_color = colors.HexColor("#64748b")
    rule_color = colors.HexColor("#e2e8f0")
    body_style = ParagraphStyle("Body", fontName="Helvetica", fontSize=9, leading=14, textColor=ink_color, spaceAfter=9)
    title_style = ParagraphStyle("Title", parent=body_style, fontSize=27, leading=32, spaceAfter=14)
    heading_style = ParagraphStyle("Heading", parent=body_style, fontSize=14, leading=19, spaceBefore=14, spaceAfter=12, keepWithNext=True)
    muted_style = ParagraphStyle("Muted", parent=body_style, fontSize=8, leading=12, textColor=muted_color, spaceAfter=4)
    number_style = ParagraphStyle("Number", parent=body_style, alignment=TA_RIGHT)
    # All configuration text is escaped before reaching ReportLab's XML markup.
    def paragraph_obj(text_obj, style_obj=body_style):
        return Paragraph(escape(str(text_obj)), style_obj)

    output_obj = BytesIO()
    document_obj = SimpleDocTemplate(
        output_obj, pagesize=A4, leftMargin=46, rightMargin=46, topMargin=52, bottomMargin=48,
        title="Investment report", author="Alpha", pageCompression=1,
    )
    story_list = [
        paragraph_obj("ALPHA / INVESTMENT REPORT", muted_style),
        Spacer(1, 20), paragraph_obj(snapshot_dict["client_name_str"], title_style),
        paragraph_obj(f"{snapshot_dict['requested_from_date_str']} to {snapshot_dict['requested_to_date_str']} | {snapshot_dict['base_currency_str']}"),
    ]
    if snapshot_dict["is_demo_bool"]:
        story_list.append(paragraph_obj("DEMONSTRATION ONLY - synthetic numbers, not actual performance", heading_style))
    if snapshot_dict["document_status_str"] == "draft":
        story_list.extend([
            paragraph_obj("DRAFT - financial evidence incomplete", heading_style),
            paragraph_obj("Some account data is missing. This draft is not ready to issue."),
        ])
    if snapshot_dict["twr_float"] is None:
        story_list.append(paragraph_obj("Returns are shown separately for each account; a combined return is not available.", muted_style))
    story_list.append(paragraph_obj("Period at a glance", heading_style))
    kpi_row_list = [["Starting value", "Ending value", "Profit / loss", "Return (TWR)"], [
        _money_str(snapshot_dict["opening_nav_float"]), _money_str(snapshot_dict["closing_nav_float"]),
        _money_str(snapshot_dict["pnl_float"]), "Not reported" if snapshot_dict["twr_float"] is None else _return_str(snapshot_dict["twr_float"]),
    ]]
    kpi_table_obj = Table([[paragraph_obj(value_str, muted_style if row_int == 0 else body_style) for value_str in row_list] for row_int, row_list in enumerate(kpi_row_list)], colWidths=[125.8] * 4)
    kpi_table_obj.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), .5, rule_color), ("INNERGRID", (0, 0), (-1, -1), .3, rule_color),
        ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story_list.append(kpi_table_obj)
    if snapshot_dict["client_twr_configured_bool"]:
        source_str = "synthetic data" if snapshot_dict["is_demo_bool"] else "IBKR data"
        story_list.append(paragraph_obj(f"Client TWR calculated from {source_str} using an end-of-day cash-flow convention.", muted_style))
    story_list.extend([Spacer(1, 10), paragraph_obj("How the balance changed", heading_style)])
    bridge_row_list = [
        ("Starting value", "opening_nav_float"), ("Net transfers and owner payments", "capital_movement_float"),
        ("Broker adjustments", "linking_adjustment_float"),
        ("Capital from strategies added / removed", "scope_movement_float"),
        ("Profit / loss", "pnl_float"), ("Ending value", "closing_nav_float"),
    ]
    bridge_table_obj = Table([[paragraph_obj(label_str), paragraph_obj(_money_str(snapshot_dict[key_str]), number_style)] for label_str, key_str in bridge_row_list], colWidths=[343.2, 160])
    bridge_table_obj.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), .3, rule_color), ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5), ("LINEABOVE", (0, -1), (-1, -1), .8, ink_color),
    ]))
    story_list.extend([
        bridge_table_obj, Spacer(1, 15),
        paragraph_obj("Profit / loss includes changes in investments still held. Money added or withdrawn is not investment profit.", muted_style),
        paragraph_obj(f"Account values cover the start of {snapshot_dict['opening_date_str'] or 'unavailable'} to the end of {snapshot_dict['closing_date_str'] or 'unavailable'}.", muted_style),
        PageBreak(), paragraph_obj("Strategy results", title_style),
        paragraph_obj("Results cover the dates shown for each strategy, including strategies that have since ended."),
    ])
    strategy_row_list = [[paragraph_obj(label_str, muted_style) for label_str in ("Strategy / dates", "Ending value", "Profit / loss", "Account return")]]
    for strategy_dict in snapshot_dict["strategy_list"]:
        strategy_row_list.append([
            [paragraph_obj(strategy_dict["display_name_str"]), paragraph_obj(f"{strategy_dict['from_date_str']} to {strategy_dict['to_date_str']}", muted_style)],
            paragraph_obj(_money_str(strategy_dict["closing_nav_float"]), number_style),
            paragraph_obj(_money_str(strategy_dict["pnl_float"]), number_style),
            paragraph_obj(_return_str(strategy_dict["twr_float"]), number_style),
        ])
    strategy_table_obj = Table(strategy_row_list, colWidths=[203.2, 100, 100, 100], repeatRows=1, hAlign="LEFT")
    strategy_table_obj.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), .3, rule_color), ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6), ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story_list.append(strategy_table_obj)
    benchmark_strategy_list = [strategy_dict for strategy_dict in snapshot_dict["strategy_list"] if strategy_dict["benchmark_dict"].get("status_str") == "ready"]
    if benchmark_strategy_list:
        story_list.append(paragraph_obj("How each strategy compared with the market", heading_style))
        comparison_row_list = [[paragraph_obj(label_str, muted_style) for label_str in ("Strategy / dates", "Benchmark", "Market return", "Gap (pct. points)")]]
        for strategy_dict in benchmark_strategy_list:
            benchmark_dict = strategy_dict["benchmark_dict"]
            comparison_row_list.append([
                [paragraph_obj(strategy_dict["display_name_str"]), paragraph_obj(f"{benchmark_dict['from_date_str']} to {benchmark_dict['to_date_str']}", muted_style),
                 paragraph_obj(f"Benchmark closes: {benchmark_dict['baseline_date_str']} to {benchmark_dict['end_price_date_str']}", muted_style)],
                paragraph_obj(benchmark_dict["label_str"] + (" (synthetic)" if benchmark_dict["is_demo_bool"] else "")),
                paragraph_obj(_return_str(benchmark_dict["return_float"]), number_style),
                paragraph_obj(f"{benchmark_dict['difference_pp_float']:+.2f}", number_style),
            ])
        comparison_table_obj = Table(comparison_row_list, colWidths=[193.2, 120, 90, 100], repeatRows=1, hAlign="LEFT")
        comparison_table_obj.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), .3, rule_color),
            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        story_list.extend([comparison_table_obj, paragraph_obj("The gap is account return minus market return, in percentage points. Comparisons use matching account dates and the market closing dates shown. Market returns include dividends, before investor fees and taxes.", muted_style)])
        for strategy_dict in snapshot_dict["strategy_list"]:
            if strategy_dict["benchmark_dict"].get("status_str") != "ready":
                story_list.append(paragraph_obj(strategy_dict["display_name_str"] + ": benchmark comparison unavailable for the full account interval.", muted_style))
    story_list.append(KeepTogether([
        Spacer(1, 12),
        paragraph_obj("Example data only - not actual IBKR results." if snapshot_dict["is_demo_bool"]
                      else "Account data is taken from IBKR statements.", muted_style),
        paragraph_obj("Prepared: " + datetime.fromisoformat(snapshot_dict["generated_at_str"]).astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC"), muted_style),
        paragraph_obj("Report ID: " + snapshot_dict["document_hash_str"], muted_style),
    ]))

    def page_frame_fn(canvas_obj, document_obj):
        canvas_obj.saveState()
        canvas_obj.setStrokeColor(rule_color)
        canvas_obj.line(46, 36, A4[0] - 46, 36)
        canvas_obj.setFont("Helvetica", 8)
        canvas_obj.setFillColor(muted_color)
        canvas_obj.drawString(46, 23, "ALPHA | " + snapshot_dict["document_status_str"].upper())
        canvas_obj.drawRightString(A4[0] - 46, 23, f"Page {document_obj.page}")
        canvas_obj.restoreState()

    document_obj.build(story_list, onFirstPage=page_frame_fn, onLaterPages=page_frame_fn)
    return output_obj.getvalue()
