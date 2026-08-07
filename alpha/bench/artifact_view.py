"""Present saved analyzer reports inside the BENCH workspace.

The analyzer artifacts remain the source of truth.  This module does not
recalculate a return, risk statistic, capacity estimate, or stress result; it
only extracts the already-rendered report sections so BENCH can present them
without an iframe.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from html.parser import HTMLParser

from markupsafe import Markup

from alpha.bench import runs


_DIV_TAG_RE = re.compile(r"</?div\b[^>]*>", re.IGNORECASE)
_MAIN_RE = re.compile(r"<main\b[^>]*>(.*)</main\s*>", re.IGNORECASE | re.DOTALL)
_BODY_RE = re.compile(r"<body\b[^>]*>(.*)</body\s*>", re.IGNORECASE | re.DOTALL)
_PLATE_START_RE = re.compile(
    r'<div\b(?=[^>]*\bclass=["\'][^"\']*\bplate\b[^"\']*["\'])'
    r'(?=[^>]*\bid=["\'](plate-\d+)["\'])[^>]*>',
    re.IGNORECASE,
)
_TITLE_RE = re.compile(r"<h1\b[^>]*>.*?</h1\s*>", re.IGNORECASE | re.DOTALL)
_H2_RE = re.compile(r"<h2\b[^>]*>(.*?)</h2\s*>", re.IGNORECASE | re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_REPORT_SHELL_OPEN_RE = re.compile(
    r'^\s*<div\b[^>]*\bclass=["\'][^"\']*\breport-shell\b[^"\']*["\'][^>]*>',
    re.IGNORECASE,
)
_REPORT_HEADER_BLOCK_RE = re.compile(
    r'<header\b(?=[^>]*\bclass=["\'][^"\']*\breport-header\b[^"\']*["\'])[^>]*>'
    r'.*?</header\s*>',
    re.IGNORECASE | re.DOTALL,
)
_PLATE_INDEX_BLOCK_RE = re.compile(
    r'<(?P<tag>nav|div)\b(?=[^>]*\bclass=["\'][^"\']*\bplate-index\b[^"\']*["\'])[^>]*>'
    r'.*?</(?P=tag)\s*>',
    re.IGNORECASE | re.DOTALL,
)
_SPEC_MASTHEAD_START_RE = re.compile(
    r'<div\b(?=[^>]*\bclass=["\'][^"\']*\bspec-masthead\b[^"\']*["\'])[^>]*>',
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ArtifactTab:
    key_str: str
    label_str: str
    html_markup: Markup


@dataclass(frozen=True)
class ArtifactView:
    tab_tuple: tuple[ArtifactTab, ...]
    selected_tab: ArtifactTab


def build_artifact_view(
    analysis_str: str,
    run_obj: runs.RunEntry | None,
    requested_tab_str: str | None,
) -> ArtifactView | None:
    """Return trusted, display-only sections from a saved report artifact."""
    if run_obj is None or not run_obj.has_report_bool:
        return None

    report_path = (runs.RESULTS_ROOT_PATH / run_obj.report_artifact_str).resolve()
    try:
        report_path.relative_to(runs.RESULTS_ROOT_PATH.resolve())
        report_html_str = report_path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None

    main_match_obj = _MAIN_RE.search(report_html_str) or _BODY_RE.search(report_html_str)
    if main_match_obj is None:
        return None
    main_html_str = main_match_obj.group(1)

    if analysis_str == "vanilla":
        tab_tuple = _vanilla_tab_tuple(main_html_str, run_obj)
    else:
        tab_tuple = (
            ArtifactTab(
                key_str="report",
                label_str="Full report",
                html_markup=_sanitize_saved_html(_strip_report_heading(main_html_str)),
            ),
        )

    if not any(str(tab_obj.html_markup).strip() for tab_obj in tab_tuple):
        return None

    selected_tab = next(
        (tab_obj for tab_obj in tab_tuple if tab_obj.key_str == requested_tab_str),
        tab_tuple[0],
    )
    return ArtifactView(tab_tuple=tab_tuple, selected_tab=selected_tab)


_ALLOWED_TAG_SET = {
    "a", "b", "br", "button", "circle", "code", "details", "div", "em", "g",
    "h2", "h3", "h4", "img", "li", "line", "ol", "p", "path", "polygon",
    "polyline", "rect", "section", "small", "span", "strong", "summary", "svg",
    "table", "tbody", "td", "text", "th", "thead", "tr", "ul",
}
_VOID_TAG_SET = {"br", "img", "line", "circle", "rect", "path", "polygon", "polyline"}
_SUPPRESSED_TAG_SET = {"embed", "form", "iframe", "object", "script", "style"}
_GLOBAL_ATTR_SET = {"class", "id", "role", "title", "aria-label", "aria-expanded"}
_TAG_ATTR_DICT = {
    "a": {"href", "rel", "target"},
    "button": {"type", "data-help"},
    "img": {"src", "alt"},
    "svg": {"viewbox", "width", "height"},
    "td": {"colspan", "rowspan", "style"},
    "th": {"colspan", "rowspan", "style"},
    "circle": {"cx", "cy", "r", "fill", "stroke", "stroke-width"},
    "line": {"x1", "x2", "y1", "y2", "fill", "stroke", "stroke-width"},
    "path": {"d", "fill", "stroke", "stroke-width"},
    "polygon": {"points", "fill", "stroke", "stroke-width"},
    "polyline": {"points", "fill", "stroke", "stroke-width"},
    "rect": {"x", "y", "width", "height", "rx", "fill", "stroke", "stroke-width"},
    "text": {"x", "y", "dx", "dy", "fill", "font-size", "text-anchor", "transform"},
}
_TOKEN_ATTR_RE = re.compile(r"^[A-Za-z0-9_ .:-]+$")
_SVG_VALUE_RE = re.compile(r"^[A-Za-z0-9#.,() +\-]+$")
_DATA_IMAGE_RE = re.compile(
    r"^data:image/(?:png|jpeg|jpg|webp|gif);base64,[A-Za-z0-9+/=\s]+$", re.IGNORECASE
)
_SAFE_STYLE_DECLARATION_RE = re.compile(
    r"^(?:background(?:-color)?|color|text-align|white-space)\s*:\s*"
    r"(?:#[0-9a-fA-F]{3,8}|rgba?\([0-9., %]+\)|[A-Za-z -]+)$"
)


class _SavedReportSanitizer(HTMLParser):
    """Strict allow-list sanitizer for static, locally generated reports."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.output_str_list: list[str] = []
        self.open_tag_list: list[str] = []
        self.suppressed_depth_int = 0

    def handle_starttag(self, tag_str: str, attr_pair_list: list[tuple[str, str | None]]) -> None:
        tag_str = tag_str.lower()
        if tag_str in _SUPPRESSED_TAG_SET:
            self.suppressed_depth_int += 1
            return
        if self.suppressed_depth_int or tag_str not in _ALLOWED_TAG_SET:
            return
        attr_str = "".join(
            f' {name_str}="{html.escape(value_str, quote=True)}"'
            for name_str, value_str in self._safe_attr_pair_list(tag_str, attr_pair_list)
        )
        self.output_str_list.append(f"<{tag_str}{attr_str}>")
        if tag_str not in _VOID_TAG_SET:
            self.open_tag_list.append(tag_str)

    def handle_startendtag(self, tag_str: str, attr_pair_list: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag_str, attr_pair_list)

    def handle_endtag(self, tag_str: str) -> None:
        tag_str = tag_str.lower()
        if tag_str in _SUPPRESSED_TAG_SET:
            self.suppressed_depth_int = max(0, self.suppressed_depth_int - 1)
            return
        if self.suppressed_depth_int or tag_str not in self.open_tag_list:
            return
        while self.open_tag_list:
            open_tag_str = self.open_tag_list.pop()
            self.output_str_list.append(f"</{open_tag_str}>")
            if open_tag_str == tag_str:
                break

    def handle_data(self, data_str: str) -> None:
        if not self.suppressed_depth_int:
            self.output_str_list.append(html.escape(data_str))

    def markup(self) -> Markup:
        while self.open_tag_list:
            self.output_str_list.append(f"</{self.open_tag_list.pop()}>")
        return Markup("".join(self.output_str_list))

    def _safe_attr_pair_list(
        self, tag_str: str, attr_pair_list: list[tuple[str, str | None]]
    ) -> list[tuple[str, str]]:
        safe_pair_list: list[tuple[str, str]] = []
        allowed_attr_set = _GLOBAL_ATTR_SET | _TAG_ATTR_DICT.get(tag_str, set())
        for raw_name_str, raw_value_str in attr_pair_list:
            name_str = raw_name_str.lower()
            value_str = raw_value_str or ""
            if name_str not in allowed_attr_set:
                continue
            if name_str in {"class", "id", "role", "aria-expanded"}:
                if _TOKEN_ATTR_RE.fullmatch(value_str):
                    safe_pair_list.append((name_str, value_str))
            elif name_str in {"title", "aria-label", "alt", "data-help"}:
                # Text is attribute-escaped when emitted. Punctuation is part of
                # the saved explanation, not executable markup.
                safe_pair_list.append((name_str, value_str[:4000]))
            elif name_str == "src" and _DATA_IMAGE_RE.fullmatch(value_str):
                safe_pair_list.append((name_str, value_str))
            elif name_str == "href" and value_str.startswith(("https://", "http://")):
                safe_pair_list.append((name_str, value_str))
            elif name_str == "target" and value_str == "_blank":
                safe_pair_list.append((name_str, value_str))
            elif name_str == "rel":
                safe_pair_list.append((name_str, "noopener noreferrer"))
            elif name_str == "type" and value_str == "button":
                safe_pair_list.append((name_str, value_str))
            elif name_str == "style":
                safe_style_str = ";".join(
                    declaration_str.strip()
                    for declaration_str in value_str.split(";")
                    if _SAFE_STYLE_DECLARATION_RE.fullmatch(declaration_str.strip())
                )
                if safe_style_str:
                    safe_pair_list.append((name_str, safe_style_str))
            elif _SVG_VALUE_RE.fullmatch(value_str):
                output_name_str = "viewBox" if name_str == "viewbox" else name_str
                safe_pair_list.append((output_name_str, value_str))
        return safe_pair_list


def _sanitize_saved_html(html_str: str) -> Markup:
    sanitizer_obj = _SavedReportSanitizer()
    sanitizer_obj.feed(html_str)
    sanitizer_obj.close()
    return sanitizer_obj.markup()


def _strip_report_heading(html_str: str) -> str:
    """The BENCH workspace already owns the report title and metadata."""
    return _TITLE_RE.sub("", html_str, count=1).strip()


def _balanced_div_end(html_str: str, start_int: int) -> int | None:
    depth_int = 0
    for match_obj in _DIV_TAG_RE.finditer(html_str, start_int):
        tag_str = match_obj.group(0)
        if tag_str.startswith("</"):
            depth_int -= 1
            if depth_int == 0:
                return match_obj.end()
        else:
            depth_int += 1
    return None


def _strip_pre_plate_chrome(html_str: str) -> str:
    """Keep saved headline evidence but remove report-owned navigation chrome."""
    output_html_str = _REPORT_SHELL_OPEN_RE.sub("", html_str, count=1)
    output_html_str = _REPORT_HEADER_BLOCK_RE.sub("", output_html_str, count=1)
    output_html_str = _PLATE_INDEX_BLOCK_RE.sub("", output_html_str, count=1)
    masthead_match_obj = _SPEC_MASTHEAD_START_RE.search(output_html_str)
    if masthead_match_obj is not None:
        masthead_end_int = _balanced_div_end(output_html_str, masthead_match_obj.start())
        if masthead_end_int is not None:
            output_html_str = (
                output_html_str[: masthead_match_obj.start()]
                + output_html_str[masthead_end_int:]
            )
    return output_html_str


def _plate_html_dict(main_html_str: str) -> dict[str, str]:
    plate_html_dict: dict[str, str] = {}
    for match_obj in _PLATE_START_RE.finditer(main_html_str):
        end_int = _balanced_div_end(main_html_str, match_obj.start())
        if end_int is not None:
            plate_html_dict[match_obj.group(1).lower()] = main_html_str[
                match_obj.start() : end_int
            ]
    return plate_html_dict


def _vanilla_tab_tuple(
    main_html_str: str, run_obj: runs.RunEntry
) -> tuple[ArtifactTab, ...]:
    plate_html_dict = _plate_html_dict(main_html_str)
    if not plate_html_dict:
        return (
            ArtifactTab(
                key_str="report",
                label_str="Full report",
                html_markup=_sanitize_saved_html(_strip_report_heading(main_html_str)),
            ),
        )

    grouped_html_dict: dict[str, list[str]] = {
        "overview": [],
        "statistics": [],
        "composition": [],
        "trades": [],
        "audit": [],
    }
    first_plate_match_obj = _PLATE_START_RE.search(main_html_str)
    if first_plate_match_obj is not None:
        pre_plate_html_str = main_html_str[: first_plate_match_obj.start()]
        # BENCH owns the page title, run metadata and tab navigation. Preserve
        # only the saved pre-plate evidence, especially the benchmark/delta
        # headline table, and remove the now-duplicate report chrome.
        pre_plate_html_str = _strip_pre_plate_chrome(pre_plate_html_str)
        grouped_html_dict["overview"].append(pre_plate_html_str)
    for plate_html_str in plate_html_dict.values():
        heading_str = _plate_heading_str(plate_html_str)
        group_key_str = _vanilla_group_key_str(heading_str)
        grouped_html_dict[group_key_str].append(plate_html_str)
        if group_key_str == "statistics":
            # The saved statistics section owns turnover and cost attribution,
            # so it is repeated in Trades & Costs rather than silently omitted.
            grouped_html_dict["trades"].append(plate_html_str)

    grouped_html_dict["audit"].append(_vanilla_audit_html_str(run_obj))
    tab_label_tuple = (
        ("overview", "Overview"),
        ("statistics", "Statistics"),
        ("composition", "Composition"),
        ("trades", "Trades & Costs"),
        ("audit", "Audit"),
    )
    return tuple(
        ArtifactTab(
            key_str=key_str,
            label_str=label_str,
            html_markup=_sanitize_saved_html("".join(grouped_html_dict[key_str])),
        )
        for key_str, label_str in tab_label_tuple
        if grouped_html_dict[key_str]
    )


def _plate_heading_str(plate_html_str: str) -> str:
    heading_match_obj = _H2_RE.search(plate_html_str)
    if heading_match_obj is None:
        return ""
    heading_html_str = heading_match_obj.group(1)
    return html.unescape(_HTML_TAG_RE.sub("", heading_html_str)).strip().lower()


def _vanilla_group_key_str(heading_str: str) -> str:
    if any(token_str in heading_str for token_str in ("audit", "provenance")):
        return "audit"
    if any(token_str in heading_str for token_str in ("open trades", "closed trades")):
        return "trades"
    if any(
        token_str in heading_str
        for token_str in ("composition", "portfolio weights", "deployed capital")
    ):
        return "composition"
    if any(
        token_str in heading_str
        for token_str in ("statistics", "performance summary", "conditional beta")
    ):
        return "statistics"
    # Time-path, benchmark-relative, monthly and unknown future plates remain
    # visible in Overview. Unknown headings are never discarded.
    return "overview"


def _vanilla_audit_html_str(run_obj: runs.RunEntry) -> str:
    item_tuple = (
        ("Saved at", run_obj.metadata_dict.get("saved_at")),
        ("Strategy", run_obj.metadata_dict.get("strategy_name") or run_obj.run_name_str),
        ("Class", run_obj.metadata_dict.get("class_name")),
        ("Module", run_obj.metadata_dict.get("class_module")),
        ("Source file", run_obj.metadata_dict.get("class_file")),
        ("Pickle", run_obj.metadata_dict.get("pickle_path")),
        ("Capital base", run_obj.metadata_dict.get("capital_base")),
        ("Accounting policy", run_obj.metadata_dict.get("accounting_policy")),
        ("Data adjustment", run_obj.metadata_dict.get("data_adjustment_policy")),
        ("Benchmarks", run_obj.metadata_dict.get("benchmarks")),
        ("Benchmark symbols", run_obj.metadata_dict.get("benchmark_data_symbol_map")),
        ("Run parameters", run_obj.parameter_dict),
    )
    row_html_str = "".join(
        "<tr><th>{}</th><td>{}</td></tr>".format(
            html.escape(label_str),
            html.escape(_display_value_str(value_obj)),
        )
        for label_str, value_obj in item_tuple
    )
    return (
        '<section class="panel audit-panel"><h2>Audit &amp; provenance</h2>'
        "<p>This view repeats the metadata saved beside the backtest. It does not prove "
        "that the current source still matches the historical artifact; use a fresh BENCH "
        "run for current-code evidence.</p><div class=\"table-wrap\"><table><tbody>"
        f"{row_html_str}</tbody></table></div></section>"
        '<section class="panel"><h2>Interpretation limits</h2><ul>'
        "<li>Research artifact only; this page makes no LIVE-readiness claim.</li>"
        "<li>Execution, accounting, adjustment and benchmark assumptions are those recorded above.</li>"
        "<li>Open artifact remains the exact saved source document for independent review.</li>"
        "</ul></section>"
    )


def _display_value_str(value_obj: object) -> str:
    if value_obj is None or value_obj == "":
        return "Not recorded"
    if isinstance(value_obj, dict):
        return "; ".join(f"{key_str}: {_display_value_str(item_obj)}" for key_str, item_obj in value_obj.items())
    if isinstance(value_obj, (list, tuple)):
        return ", ".join(_display_value_str(item_obj) for item_obj in value_obj)
    return str(value_obj)
