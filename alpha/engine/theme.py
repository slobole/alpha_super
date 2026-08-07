from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib import colors as mcolors


SEABORN_DEEP_COLOR_LIST: list[str] = [
    '#4c72b0',
    '#dd8452',
    '#55a868',
    '#c44e52',
    '#8172b3',
    '#937860',
    '#da8bc3',
    '#8c8c8c',
    '#ccb974',
    '#64b5cd',
]


SIGNATURE_FONT_STACK_LIST: list[str] = [
    'Atlassian Sans',
    'Segoe UI',
    'Arial',
    'DejaVu Sans',
    'sans-serif',
]
SIGNATURE_FONT_STACK_STR: str = '"Atlassian Sans", "Segoe UI", Arial, "DejaVu Sans", sans-serif'
_ATLASSIAN_FONT_CDN_BASE_URL_STR: str = 'https://ds-cdn.prod-east.frontend.public.atl-paas.net'


SIGNATURE_PALETTE_DICT: dict[str, object] = {
    'ink': '#172b4d',
    'page': '#ffffff',
    'panel': '#ffffff',
    'neutral': '#f7f8fa',
    'grid': '#dfe1e6',
    'border': '#dfe1e6',
    'axes_border': '#7a869a',
    'muted': '#6b778c',
    'strategy': '#357de8',
    'strategy_dark': '#1d5fdb',
    'benchmark': '#f28c28',
    'benchmark_dark': '#c66a13',
    'profit': '#22a06b',
    'profit_dark': '#216e4e',
    'loss': '#c9372c',
    'loss_dark': '#ae2e24',
    # Navigation accent and the third state between profit and loss. Kept
    # separate from profit/loss on purpose: a link and an active tab are not
    # "good news", and a SKIP is not a FAIL. Before these existed the console
    # spent --color-ink on both, which is why every interactive affordance read
    # with the same weight as body text.
    'accent': '#357de8',
    'accent_dark': '#1d5fdb',
    'warning': '#b45309',
    'warning_dark': '#8a3f07',
    # Tinted fields for badges and selected rows: the same semantic as the hue
    # above, at a weight that can sit behind text. Kept as explicit palette
    # entries rather than an alpha of the base colour, because a translucent
    # badge over a striped or hovered table row shifts with whatever is beneath
    # it and stops meaning one fixed state.
    'accent_wash': '#e8f0fe',
    'profit_wash': '#e6f4ea',
    'warning_wash': '#fef7e0',
    'loss_wash': '#fce8e6',
    # Area-fill weights, as alpha over the line colour they belong to. A
    # palette whose series lines are dark must dial the equity band down; one
    # with a mid-tone accent can afford more.
    #
    # *** UI*** The drawdown weight is raised here too, not only for desk. The
    # outlines are now hairlines because the area carries the shape, and
    # leaving the baseline at 0.07 would pair a faint fill with a thin line and
    # render the panel *less* readable than before. Drawing drawdown as an area
    # is correct for any palette, so there is one behaviour, not two.
    'equity_fill_alpha_float': 0.24,
    'drawdown_fill_alpha_float': 0.38,
    'benchmark_drawdown_fill_alpha_float': 0.14,
    'vertical_line': '#8590a2',
    'zero_line': '#626f86',
    'bar_edge': '#dfe1e6',
    'legend_face': '#ffffff',
    'legend_edge': '#dfe1e6',
    'label_face': '#ffffff',
    'overlay_cycle': list(SEABORN_DEEP_COLOR_LIST),
    'mean_line': '#357de8',
    'shadow_rgba': 'rgba(9, 30, 66, 0.04)',
    'font_family_str': 'sans-serif',
    'font_stack_list': list(SIGNATURE_FONT_STACK_LIST),
    'font_stack_str': SIGNATURE_FONT_STACK_STR,
    'prose_font_stack_str': SIGNATURE_FONT_STACK_STR,
    # Program output and identifiers — logs, commands, paths. A distinct role
    # from 'figure': a variant whose figure face is a sans still needs code to
    # be monospace, or a command line stops being copy-readable.
    'code_font_stack_str': '"Cascadia Mono", Consolas, "DejaVu Sans Mono", monospace',
    # Populated just below, once SIGNATURE_ASSET_COLOR_DICT exists.
    'asset_color_dict': {},
    'layout_str': 'dashboard',
    'axis_style_str': 'dashboard',
    # Empty means fills are distinguished by colour alone. A populated cycle
    # lets a monochrome variant separate areas by texture instead.
    'hatch_cycle_list': [],
}

SIGNATURE_ASSET_COLOR_DICT: dict[str, str] = {
    # Baseline (dashboard) asset hues. Variants override this wholesale via the
    # 'asset_color_dict' palette key — see _build_cycle_asset_color_dict.
    'TLT': '#4f6bed',
    'GLD': '#d9a441',
    'DBC': '#36b37e',
    'UUP': '#00a3bf',
    'SPY': SIGNATURE_PALETTE_DICT['benchmark'],
    'SSO': SIGNATURE_PALETTE_DICT['benchmark'],
    'QQQ': '#6554c0',
    'QLD': '#7a5af8',
    'TQQQ': '#8f63ff',
    'UPRO': '#9b8afb',
    'BTAL': '#c251c0',
    'SH': '#6b778c',
    'CASH': '#b3bac5',
    'DEFAULT': '#7a869a',
}

SIGNATURE_PALETTE_DICT['asset_color_dict'] = dict(SIGNATURE_ASSET_COLOR_DICT)


_BASE_VARIANT_NAME_STR: str = 'current'

# Candidate signature variants under evaluation. Each entry lists only the keys
# that differ from SIGNATURE_PALETTE_DICT; the resolver merges them onto the
# base and rejects unknown keys, so a typo fails loud instead of silently
# introducing an unused colour.
# Desk: the institutional trading-desk console the investor mockups describe.
# Where journal and swiss both spend their whole colour budget on one accent,
# this one accepts that a control panel has *states* a research note does not —
# a link, an active tab, a PASS, a SKIP, a FAIL — and gives each exactly one
# hue. Nothing else on the page is allowed colour: the equity curve is ink, the
# benchmark is grey, and every container is a hairline on white.
#
# The rule that keeps this from drifting into decoration: a colour here always
# encodes machine-readable state. If a value cannot be PASS/SKIP/FAIL or
# navigable, it renders in ink.
# Prose is a grotesque, figures are mono. The split is the point: an identifier
# like a module path or a price is a figure and must align in a column; a
# heading is language and should not. IBM Plex Sans and IBM Plex Mono are
# siblings from one design, so the split reads as deliberate rather than as two
# unrelated fonts colliding.
#
# These are vendored under alpha/assets/fonts (SIL OFL) and shipped with the
# artifact — see that directory's README for why. The short version: a report
# must look the same on the machine that wrote it and the machine that reads
# it, and neither a CDN nor an OS font can promise that.
#
# The OS names after Plex are the offline fallback if the woff2 fails to load.
#
# *** UI*** Two faces were tried in this stack and removed after MEASURING, not
# after guessing. Do not re-add either without re-measuring:
#
#   * "Inter" — not installed on the target machine. It sat second in the stack
#     and never applied; it only misled anyone reading the theme.
#   * "Segoe UI Variable" (Text / Display / bare) — installed at the OS level
#     and enumerable through GDI, but NOT addressable from Chromium under any
#     name form. Measured against a deliberately bogus family, every variant
#     produced identical text metrics, i.e. it silently fell through.
#
# A font belongs in a stack only if it resolves in the browser that renders the
# console. Verify with a canvas width measurement against a nonsense family
# name — checking the OS font list is not sufficient.
_DESK_FIGURE_FONT_STACK_LIST: list[str] = [
    'IBM Plex Mono', 'Cascadia Mono', 'Consolas', 'DejaVu Sans Mono', 'monospace',
]
_DESK_FIGURE_FONT_STACK_STR: str = (
    '"IBM Plex Mono", "Cascadia Mono", Consolas, "DejaVu Sans Mono", monospace'
)
_DESK_PROSE_FONT_STACK_STR: str = (
    '"IBM Plex Sans", "Segoe UI", system-ui, -apple-system, '
    '"Helvetica Neue", Arial, sans-serif'
)


# Vendored faces, in the order they are declared. Each entry is
# (family, weight-or-range, filename).
VENDORED_FONT_FACE_TUPLE: tuple[tuple[str, str, str], ...] = (
    # Sans is a variable font: one file covers the whole weight axis, so it is
    # declared with a range rather than once per weight.
    ('IBM Plex Sans', '100 700', 'IBMPlexSans-latin.woff2'),
    ('IBM Plex Mono', '400', 'IBMPlexMono-400-latin.woff2'),
    ('IBM Plex Mono', '500', 'IBMPlexMono-500-latin.woff2'),
)
VENDORED_FONT_DIR_PATH = Path(__file__).resolve().parents[1] / 'assets' / 'fonts'

# TrueType copies of the same faces, for matplotlib.
#
# *** UI*** These exist because matplotlib cannot read WOFF2. Without them the
# charts silently fell back to whatever mono the OS had while the surrounding
# HTML rendered in Plex — two fonts inside one document, which is worse than
# having picked neither. They are build-time assets: matplotlib rasterises to
# PNG, so unlike the woff2 files these are never embedded in an artifact.
_MATPLOTLIB_FONT_FILE_TUPLE: tuple[str, ...] = (
    'IBMPlexSans-Regular.ttf',
    'IBMPlexMono-Regular.ttf',
)


def _register_vendored_matplotlib_fonts() -> None:
    """Make the vendored faces resolvable by name in matplotlib.

    Missing files are skipped rather than raised on: a chart drawn in a
    fallback face is still correct evidence, and a research run must not die
    because an asset is absent.
    """
    from matplotlib import font_manager

    for font_file_name_str in _MATPLOTLIB_FONT_FILE_TUPLE:
        font_path = VENDORED_FONT_DIR_PATH / font_file_name_str
        if not font_path.is_file():
            continue
        try:
            font_manager.fontManager.addfont(str(font_path))
        except (OSError, RuntimeError):
            continue


_register_vendored_matplotlib_fonts()


def build_font_face_css(url_builder_fn) -> str:
    """Emit @font-face rules for the vendored faces.

    ``url_builder_fn`` maps a filename to whatever the consuming surface needs
    as the src URL — a served path for the console, a base64 data URI for a
    saved report. The declarations are otherwise identical, so the two surfaces
    cannot drift in weight, style or family name.

    *** UI*** font-display is 'block', not 'swap'. These pages are dense
    numeric tables; a swap would render them once in a fallback metric and then
    reflow every column the moment the real face arrives. The files are local
    and tens of kilobytes, so the block period is imperceptible.
    """
    rule_str_list: list[str] = []
    for family_str, weight_str, file_name_str in VENDORED_FONT_FACE_TUPLE:
        rule_str_list.append(
            "@font-face{{font-family:'{0}';font-style:normal;font-weight:{1};"
            "font-display:block;src:url({2}) format('woff2');}}".format(
                family_str, weight_str, url_builder_fn(file_name_str)
            )
        )
    return '\n'.join(rule_str_list)


def build_embedded_font_face_css() -> str:
    """@font-face rules with the woff2 inlined as base64 data URIs.

    This is what makes a saved report self-contained: it carries its own type
    and renders identically with no network and no particular OS. Missing files
    degrade to the OS fallback in the stack rather than raising — a report that
    renders in the wrong face is still readable evidence, one that fails to
    build is not.
    """
    import base64

    def data_uri_str(file_name_str: str) -> str:
        font_path = VENDORED_FONT_DIR_PATH / file_name_str
        try:
            encoded_str = base64.b64encode(font_path.read_bytes()).decode('ascii')
        except OSError:
            return ''
        return f'data:font/woff2;base64,{encoded_str}'

    return build_font_face_css(data_uri_str)


_DESK_PALETTE_DICT: dict[str, object] = {
    # Cool near-black on pure white. Not #000: full black on white vibrates
    # under projection, which is where this design gets used.
    'ink': '#16181d',
    'page': '#ffffff',
    'panel': '#ffffff',
    'neutral': '#f8f9fa',
    'grid': '#eceef1',
    'border': '#e0e3e8',
    'axes_border': '#16181d',
    'muted': '#5f6470',
    # The equity curve is ink and the benchmark is grey — the same restraint
    # every monochrome variant here keeps. Colour is reserved for state.
    'strategy': '#16181d',
    'strategy_dark': '#000000',
    'benchmark': '#9aa0a6',
    'benchmark_dark': '#71767d',
    'profit': '#1e8e3e',
    'profit_dark': '#137333',
    'loss': '#d93025',
    'loss_dark': '#b3261e',
    'accent': '#1a73e8',
    'accent_dark': '#1558d6',
    'warning': '#e37400',
    'warning_dark': '#b45309',
    'accent_wash': '#e8f0fe',
    'profit_wash': '#e6f4ea',
    'warning_wash': '#fef7e0',
    'loss_wash': '#fce8e6',
    # *** UI*** The two panels want opposite treatments, and it is worth being
    # explicit about why.
    #
    # Equity: the fill is a *band between two curves*, so it must stay faint —
    # at the baseline weight it composited over an ink line into solid grey and
    # hid the benchmark running through it.
    #
    # Drawdown: the fill IS the mark. Drawdown is an area quantity — depth
    # below zero — and a daily series over a decade is far too dense to read as
    # a line; at a faint weight the panel collapsed into a black scribble. The
    # area carries the shape and the line becomes a hairline edge on it.
    'equity_fill_alpha_float': 0.09,
    'drawdown_fill_alpha_float': 0.45,
    'benchmark_drawdown_fill_alpha_float': 0.16,
    'vertical_line': '#9aa0a6',
    'zero_line': '#16181d',
    'bar_edge': '#16181d',
    'legend_face': '#ffffff',
    'legend_edge': '#e0e3e8',
    'label_face': '#ffffff',
    # Muted earth-and-slate ramp for composition stacks, lifted from the
    # mockups' weight charts. Desaturated enough that a seven-sleeve stack
    # never competes with a PASS/FAIL badge sitting next to it.
    'overlay_cycle': [
        '#3f5a63', '#a8845c', '#6b7f6a', '#c4b39a',
        '#4a6670', '#8b6f4e', '#9aa5a0', '#d8cdb8',
    ],
    # *** UI*** Deliberately empty. Hatch textures alias badly through a
    # projector and this variant exists to be presented, so sleeves separate by
    # value alone and the ramp above is spaced to allow it.
    'hatch_cycle_list': [],
    'mean_line': '#16181d',
    'shadow_rgba': 'rgba(22, 24, 29, 0.06)',
    'font_family_str': 'monospace',
    'font_stack_list': list(_DESK_FIGURE_FONT_STACK_LIST),
    'font_stack_str': _DESK_FIGURE_FONT_STACK_STR,
    'prose_font_stack_str': _DESK_PROSE_FONT_STACK_STR,
    'code_font_stack_str': _DESK_FIGURE_FONT_STACK_STR,
    'axis_style_str': 'minimal',
}


# One house look, plus the legacy baseline.
#
# The journal, journal_spec, swiss and blueprint variants were a design search:
# four candidate identities kept side by side so they could be compared. That
# search is settled — desk is the house style — and keeping the losers alive
# had a real cost. Bench shipped a menu offering four looks for a
# single-operator console, and the console and its embedded reports could
# disagree about which one was active.
#
# desk carries the *spec* layout: numbered plates with a ruled masthead. That
# is not cosmetic — alpha/bench/artifact_view.py locates report sections by
# their `plate-NN` ids, so the Bench vanilla overview grid depends on plates
# existing. A layout without them silently degrades that page to one long
# unstructured column.
_VARIANT_OVERRIDE_DICT: dict[str, dict[str, object]] = {
    _BASE_VARIANT_NAME_STR: {},
    'desk': {**_DESK_PALETTE_DICT, 'layout_str': 'spec'},
}

SIGNATURE_VARIANT_NAME_LIST: list[str] = list(_VARIANT_OVERRIDE_DICT)


def _copy_palette_dict(palette_dict: dict[str, object]) -> dict[str, object]:
    """Copy a palette, duplicating its mutable containers.

    *** CRITICAL*** Lists and dicts are copied, not shared. The variant context
    mutates SIGNATURE_PALETTE_DICT in place, so a shared container would let a
    variant's edits leak into the saved baseline and survive the restore.
    """
    copied_palette_dict: dict[str, object] = {}
    for key_str, value_obj in palette_dict.items():
        if isinstance(value_obj, list):
            copied_palette_dict[key_str] = list(value_obj)
        elif isinstance(value_obj, dict):
            copied_palette_dict[key_str] = dict(value_obj)
        else:
            copied_palette_dict[key_str] = value_obj
    return copied_palette_dict


def resolve_variant_palette_dict(variant_name_str: str = _BASE_VARIANT_NAME_STR) -> dict[str, object]:
    """Return the full palette for a named variant.

    The base variant resolves to SIGNATURE_PALETTE_DICT unchanged, so every
    existing call path keeps its current appearance byte for byte.
    """
    if variant_name_str not in _VARIANT_OVERRIDE_DICT:
        raise ValueError(
            f'Unknown signature variant {variant_name_str!r}. '
            f'Expected one of {SIGNATURE_VARIANT_NAME_LIST}.'
        )

    resolved_palette_dict = _copy_palette_dict(SIGNATURE_PALETTE_DICT)
    override_dict = _VARIANT_OVERRIDE_DICT[variant_name_str]

    unknown_key_list = sorted(set(override_dict) - set(resolved_palette_dict))
    if unknown_key_list:
        raise KeyError(
            f'Signature variant {variant_name_str!r} defines unknown palette keys: '
            f'{unknown_key_list}.'
        )

    for key_str, value_obj in override_dict.items():
        if isinstance(value_obj, list):
            resolved_palette_dict[key_str] = list(value_obj)
        elif isinstance(value_obj, dict):
            resolved_palette_dict[key_str] = dict(value_obj)
        else:
            resolved_palette_dict[key_str] = value_obj

    return resolved_palette_dict


@contextmanager
def signature_variant_context(variant_name_str: str) -> Iterator[dict[str, object]]:
    """Temporarily activate a named signature variant.

    *** CRITICAL*** SIGNATURE_PALETTE_DICT is mutated in place rather than
    rebound. plot.py and report.py bind the dict object at import time, so
    rebinding the module-level name here would leave those modules pointing at
    the old palette and the preview would silently show the wrong theme. The
    original contents are always restored on exit.
    """
    restore_palette_dict = _copy_palette_dict(SIGNATURE_PALETTE_DICT)
    resolved_palette_dict = resolve_variant_palette_dict(variant_name_str)

    SIGNATURE_PALETTE_DICT.clear()
    SIGNATURE_PALETTE_DICT.update(resolved_palette_dict)
    try:
        yield resolved_palette_dict
    finally:
        SIGNATURE_PALETTE_DICT.clear()
        SIGNATURE_PALETTE_DICT.update(restore_palette_dict)


def get_signature_palette_dict() -> dict[str, object]:
    return _copy_palette_dict(SIGNATURE_PALETTE_DICT)


def build_analyzer_report_css() -> str:
    """Signature CSS plus a mapping for the analyzers' own class names.

    Capacity and risk each grew a private design system — panels, tiles, cards,
    verdict rows — before the signature existed. Rewriting their markup would
    be a large change to working research code for no analytical gain, so this
    maps those class names onto the active palette instead: the pages inherit
    the theme, and their builders keep emitting exactly what they emit today.
    """
    palette_dict = SIGNATURE_PALETTE_DICT
    return build_report_css() + f'''
/* Shared containers */
main, .wrap {{
    max-width: 1000px;
    margin: 0 auto;
}}
.panel, .tile, .verdict-panel, .chart-panel {{
    background: var(--color-panel);
    border: 1px solid var(--color-border);
    border-radius: 0;
    box-shadow: none;
    padding: 14px 16px;
    margin: 14px 0;
}}
.panel > h2:first-child, .verdict-panel > .verdict-title {{
    margin-top: 0;
}}
/* Section headings across the analyzers get the same ruled caption bar the
   strategy report gives a plate. Without this they fell through to the bare
   h2 rule -- 0.7rem, muted, the eyebrow style -- so the analyzers read as a
   different product from the report they sit beside. No plate counter: these
   sections are not numbered plates, and stamping a number on them would
   invent an ordering the page does not have. */
.panel > h2, .card > h2, .stress-section > h2, .section > h2, .verdict-panel > .verdict-title {{
    background: var(--color-neutral);
    border: none;
    border-bottom: 1px solid var(--color-ink);
    border-radius: 0;
    font-family: var(--font-figure);
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--color-ink);
    padding: 9px 13px;
    margin: 0 0 14px;
}}
/* Pull the caption bar out to the container's edge so it rules the full width
   of its panel, the way a plate caption does.
   *** UI*** The offsets must mirror the container's own padding. Only .panel
   and .verdict-panel are padded here -- this stylesheet strips .card and
   .stress-section to zero -- so applying the same pull to those pushed the bar
   clean outside its box. */
.panel > h2, .verdict-panel > .verdict-title {{
    margin: -14px -16px 14px;
}}
.read-first {{
    border-left: 3px solid var(--color-ink);
}}
/* Metric tiles reuse the headline grammar */
.cards, .tile-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 14px 24px;
    margin: 16px 0 20px;
    padding: 16px 0 18px;
    border-top: 1.5px solid var(--color-ink);
    border-bottom: 1px solid var(--color-border);
}}
.card, .tile {{
    background: none;
    border: none;
    padding: 0;
    margin: 0;
}}
.card b, .tile-value {{
    font-family: var(--font-figure);
    display: block;
    font-size: 1.35rem;
    font-weight: 500;
    font-variant-numeric: tabular-nums;
    margin-top: 6px;
}}
.tile-label {{
    font-family: var(--font-figure);
    font-size: 0.62rem;
    letter-spacing: 0.11em;
    text-transform: uppercase;
    color: var(--color-muted);
}}
.muted, .meta, .tile-sub, .verdict-text {{
    color: var(--color-muted);
}}
/* Warnings keep their weight: these are the caveats that must not fade out. */
.status-banner {{
    background: {blend_hex_color_str(str(palette_dict['page']), str(palette_dict['loss']), 0.12)};
    border: 1px solid var(--color-loss);
    border-radius: 0;
    color: var(--color-loss-dark);
    padding: 12px 14px;
    font-weight: 700;
}}
.caveat {{
    background: var(--color-neutral);
    border: 1px solid var(--color-border);
    border-left: 3px solid var(--color-loss);
    border-radius: 0;
    color: var(--color-ink);
    padding: 12px 14px;
}}
.formula {{
    font-family: var(--font-figure);
    background: var(--color-neutral);
    border: 1px solid var(--color-border);
    border-radius: 0;
    padding: 12px;
}}
code {{
    font-family: var(--font-figure);
    background: var(--color-neutral);
    border-radius: 0;
    padding: 1px 4px;
}}
.verdict-title, .verdict-label {{
    font-family: var(--font-figure);
    color: var(--color-ink);
}}
.verdict-value {{
    font-family: var(--font-figure);
    font-variant-numeric: tabular-nums;
    color: var(--color-ink);
}}
/* The row's four spans carried no layout at all, so label, value and
   conclusion rendered as one run-on string. Give them columns. */
.verdict-row {{
    display: grid;
    grid-template-columns: 10px 150px minmax(150px, 1fr) 2fr;
    align-items: baseline;
    gap: 0 16px;
    padding: 9px 0;
    border-bottom: 1px solid var(--color-border);
}}
.verdict-row:last-of-type {{
    border-bottom: none;
}}
.verdict-dot {{
    width: 7px;
    height: 7px;
    border-radius: 50%;
    align-self: center;
    background: var(--color-muted);
}}
/* The caution band must not read as the no-signal band. --color-benchmark-dark
   is a near-neutral grey in the earth palette and was indistinguishable from
   the default dot, so the middle step comes from the palette's tan instead:
   green, tan, umber is a visible three-step progression. */
.verdict-dot.v-green {{ background: var(--color-profit-dark); }}
.verdict-dot.v-amber {{ background: {palette_dict['overlay_cycle'][3]}; }}
.verdict-dot.v-red {{ background: var(--color-loss-dark); }}
.verdict-dot.v-na {{ background: var(--color-border); }}
.verdict-text {{
    font-family: var(--font-prose);
    line-height: 1.5;
}}
@media (max-width: 780px) {{
    .verdict-row {{
        grid-template-columns: 10px 1fr;
    }}
    .verdict-value, .verdict-text {{
        grid-column: 2;
    }}
}}
/* Method and limits: kept in the artifact and out of the reading path. The
   caveats are house doctrine, so they are collapsed rather than dropped. */
details.method-note {{
    margin: 34px 0 0;
    padding-top: 16px;
    border-top: 1px solid var(--color-border);
}}
details.method-note > summary {{
    font-family: var(--font-figure);
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--color-muted);
    cursor: pointer;
}}
details.method-note p {{
    font-family: var(--font-prose);
    font-size: 0.8rem;
    line-height: 1.6;
    color: var(--color-muted);
    max-width: 68ch;
}}
/* Plain-language horizon readout: prose measure, figures in figure type. */
.horizon-readout {{
    margin: 4px 0 22px;
}}
.horizon-readout-row {{
    display: grid;
    grid-template-columns: 120px minmax(0, 1fr);
    gap: 0 20px;
    padding: 11px 0;
    border-bottom: 1px solid var(--color-border);
}}
.horizon-readout-term {{
    font-family: var(--font-figure);
    font-size: 0.72rem;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--color-muted);
    padding-top: 2px;
}}
.horizon-readout-body {{
    font-family: var(--font-prose);
    font-size: 0.95rem;
    line-height: 1.65;
    max-width: 62ch;
}}
.horizon-readout-body b {{
    font-family: var(--font-figure);
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
}}
.horizon-readout-note {{
    font-family: var(--font-prose);
    font-size: 0.78rem;
    line-height: 1.55;
    color: var(--color-muted);
    max-width: 62ch;
    padding-top: 12px;
}}
@media (max-width: 780px) {{
    .horizon-readout-row {{
        grid-template-columns: 1fr;
        gap: 6px 0;
    }}
}}
.table-wrap {{
    overflow-x: auto;
}}
a {{
    color: var(--color-ink);
}}
'''


def build_bench_theme_css(variant_name_str: str = 'desk') -> str:
    """Emit a signature variant's palette for the Bench control panel.

    Bench's stylesheet is written directly against the report's own token names
    (``--color-ink``, ``--font-figure``, …), so this is the whole mapping: the
    console and the reports it embeds read from one palette rather than two
    parallel ones that drift. Emitted after bench.css so a variant switch wins
    over the stylesheet's built-in defaults.
    """
    palette_dict = resolve_variant_palette_dict(variant_name_str)
    return _build_palette_root_css(
        palette_dict,
        figure_font_stack_str=str(palette_dict['font_stack_str']),
        prose_font_stack_str=str(palette_dict['prose_font_stack_str']),
    ).lstrip('\n')


def build_report_font_head_html() -> str:
    """Return font preload tags for the active variant's report HTML.

    Only the Atlassian-derived baseline uses a CDN font. Variants whose stacks
    are entirely locally installed emit nothing: preloading a face they never
    render would be a pointless external request and would break the property
    that those reports display identically with no network access.
    """
    active_font_stack_str = str(SIGNATURE_PALETTE_DICT['font_stack_str'])
    active_prose_stack_str = str(SIGNATURE_PALETTE_DICT['prose_font_stack_str'])
    if 'Atlassian Sans' not in active_font_stack_str + active_prose_stack_str:
        return ''
    return (
        f'<link rel="preconnect" href="{_ATLASSIAN_FONT_CDN_BASE_URL_STR}" crossorigin>\n'
        f'<link rel="preload" href="{_ATLASSIAN_FONT_CDN_BASE_URL_STR}/assets/fonts/atlassian-sans/v3/AtlassianSans-latin.woff2" '
        'as="font" type="font/woff2" crossorigin>\n'
        f'<link rel="preload stylesheet" href="{_ATLASSIAN_FONT_CDN_BASE_URL_STR}/assets/font-rules/v5/atlassian-fonts.css" '
        'as="style" crossorigin>'
    )


def build_signature_rcparams(to_web_bool: bool) -> dict[str, object]:
    base_style_dict = dict(plt.style.library.get('seaborn-v0_8-whitegrid', {}))
    font_family_str = str(SIGNATURE_PALETTE_DICT['font_family_str'])
    font_stack_list = list(SIGNATURE_PALETTE_DICT['font_stack_list'])
    override_style_dict = {
        'axes.prop_cycle': cycler(color=list(SIGNATURE_PALETTE_DICT['overlay_cycle'])),
        'figure.facecolor': SIGNATURE_PALETTE_DICT['page'],
        'axes.facecolor': SIGNATURE_PALETTE_DICT['panel'],
        'axes.edgecolor': SIGNATURE_PALETTE_DICT['axes_border'],
        'axes.linewidth': 0.85,
        'axes.labelcolor': SIGNATURE_PALETTE_DICT['ink'],
        'axes.titlecolor': SIGNATURE_PALETTE_DICT['ink'],
        'grid.color': SIGNATURE_PALETTE_DICT['grid'],
        'grid.alpha': 1.0,
        'grid.linewidth': 0.75,
        'xtick.color': SIGNATURE_PALETTE_DICT['ink'],
        'ytick.color': SIGNATURE_PALETTE_DICT['ink'],
        'text.color': SIGNATURE_PALETTE_DICT['ink'],
        'font.family': font_family_str,
        f'font.{font_family_str}': font_stack_list,
        'font.size': 9.5 if to_web_bool else 10.0,
        'axes.titlesize': 10.5 if to_web_bool else 11.0,
        'axes.labelsize': 9.0 if to_web_bool else 9.5,
        'legend.framealpha': 1.0,
        'legend.fontsize': 8.0 if to_web_bool else 8.5,
        'legend.facecolor': SIGNATURE_PALETTE_DICT['legend_face'],
        'legend.edgecolor': SIGNATURE_PALETTE_DICT['legend_edge'],
        'xtick.major.size': 3.5,
        'ytick.major.size': 3.5,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'savefig.facecolor': SIGNATURE_PALETTE_DICT['page'],
        'savefig.edgecolor': SIGNATURE_PALETTE_DICT['page'],
    }
    base_style_dict.update(override_style_dict)
    return base_style_dict


def build_plot_color_dict(colors=None) -> dict[str, object]:
    if colors is None:
        strategy_color_str = SIGNATURE_PALETTE_DICT['strategy']
        benchmark_color_str = SIGNATURE_PALETTE_DICT['benchmark']
    else:
        strategy_color_str, benchmark_color_str = colors

    return {
        'strategy': strategy_color_str,
        'benchmark': benchmark_color_str,
        'additional_cycle': [
            SEABORN_DEEP_COLOR_LIST[0],
            SEABORN_DEEP_COLOR_LIST[1],
            SEABORN_DEEP_COLOR_LIST[4],
            SEABORN_DEEP_COLOR_LIST[5],
            SEABORN_DEEP_COLOR_LIST[6],
            SEABORN_DEEP_COLOR_LIST[7],
            SEABORN_DEEP_COLOR_LIST[8],
            SEABORN_DEEP_COLOR_LIST[9],
        ],
    }


def apply_signature_axis_style(axis_obj, vertical_line_iterable: Iterable[object] = ()) -> None:
    if str(SIGNATURE_PALETTE_DICT['axis_style_str']) == 'minimal':
        _apply_minimal_axis_style(axis_obj, vertical_line_iterable)
        return

    signature_palette_dict = SIGNATURE_PALETTE_DICT

    axis_obj.spines['top'].set_visible(False)
    for side_name_str in ('right', 'left', 'bottom'):
        axis_obj.spines[side_name_str].set_visible(True)
        axis_obj.spines[side_name_str].set_color(signature_palette_dict['axes_border'])
        axis_obj.spines[side_name_str].set_linewidth(0.85)

    axis_obj.tick_params(axis='x', labelsize=8, colors=signature_palette_dict['ink'], pad=4)
    axis_obj.tick_params(axis='y', labelsize=8, colors=signature_palette_dict['ink'], pad=4)
    axis_obj.grid(
        axis='y',
        which='major',
        linestyle='-',
        linewidth=0.75,
        color=signature_palette_dict['grid'],
        alpha=1.0,
    )
    axis_obj.xaxis.grid(False, which='major')
    axis_obj.xaxis.grid(False, which='minor')
    axis_obj.set_axisbelow(True)
    axis_obj.yaxis.tick_right()
    axis_obj.yaxis.set_label_position('left')

    for vertical_line_obj in vertical_line_iterable:
        axis_obj.axvline(
            vertical_line_obj,
            color=signature_palette_dict['vertical_line'],
            linestyle='--',
            linewidth=0.9,
            alpha=0.9,
            zorder=1,
        )


SIGNATURE_TIME_AXIS_ROTATION_FLOAT: float = 90.0
_SHORT_WINDOW_MAX_DAY_COUNT_INT = 62
_MEDIUM_WINDOW_MAX_DAY_COUNT_INT = 366
_INTERMEDIATE_WINDOW_MAX_DAY_COUNT_INT = 3 * 366
_SHORT_WINDOW_MAX_TICK_COUNT_INT = 8
_MAX_YEAR_TICK_COUNT_INT = 40


def build_signature_time_axis_spec(bar_date_idx) -> tuple[object, object, float]:
    """Build the one date-axis convention every time panel shares.

    *** CRITICAL*** Tick labels are always vertical, at every span. Rotation
    used to vary with the plotted window, so a drawdown panel and the return
    panel directly beneath it printed the same years in two different
    orientations within a single figure. One convention, everywhere.

    Year spacing widens only when annual ticks would exceed what fits, so
    panels covering the same window always agree on their tick dates.
    """
    import matplotlib.dates as mdates
    import numpy as np
    import pandas as pd
    from matplotlib.ticker import FixedLocator

    normalized_bar_date_idx = pd.DatetimeIndex(bar_date_idx).sort_values().unique()
    rotation_float = SIGNATURE_TIME_AXIS_ROTATION_FLOAT

    if len(normalized_bar_date_idx) == 0:
        return mdates.YearLocator(), mdates.DateFormatter('%Y'), rotation_float

    if len(normalized_bar_date_idx) == 1:
        single_tick_float = float(mdates.date2num(normalized_bar_date_idx[0].to_pydatetime()))
        return FixedLocator([single_tick_float]), mdates.DateFormatter('%Y-%m-%d'), rotation_float

    span_day_count_int = int((normalized_bar_date_idx[-1] - normalized_bar_date_idx[0]).days)

    if span_day_count_int <= _SHORT_WINDOW_MAX_DAY_COUNT_INT:
        tick_count_int = min(_SHORT_WINDOW_MAX_TICK_COUNT_INT, len(normalized_bar_date_idx))
        tick_position_vec = np.linspace(
            0.0, float(len(normalized_bar_date_idx) - 1), tick_count_int
        )
        tick_index_vec = np.unique(np.round(tick_position_vec).astype(int))
        # *** CRITICAL*** Sample short-window tick labels from the actual
        # observed bar dates so crisis plots show real tradable dates rather
        # than synthetic calendar interpolation.
        tick_date_idx = normalized_bar_date_idx[tick_index_vec]
        tick_location_vec = mdates.date2num(tick_date_idx.to_pydatetime())
        return FixedLocator(tick_location_vec), mdates.DateFormatter('%Y-%m-%d'), rotation_float

    if span_day_count_int <= _MEDIUM_WINDOW_MAX_DAY_COUNT_INT:
        return mdates.MonthLocator(interval=1), mdates.DateFormatter('%Y-%m'), rotation_float

    if span_day_count_int <= _INTERMEDIATE_WINDOW_MAX_DAY_COUNT_INT:
        return mdates.MonthLocator(interval=3), mdates.DateFormatter('%Y-%m'), rotation_float

    year_count_int = max(1, int(round(span_day_count_int / 365.25)))
    year_interval_int = 1
    for candidate_interval_int in (1, 2, 5, 10):
        year_interval_int = candidate_interval_int
        if year_count_int / candidate_interval_int <= _MAX_YEAR_TICK_COUNT_INT:
            break
    return (
        mdates.YearLocator(base=year_interval_int),
        mdates.DateFormatter('%Y'),
        rotation_float,
    )


def apply_signature_time_axis(axis_obj, bar_date_idx) -> None:
    """Apply the shared date-axis convention to an axis."""
    locator_obj, formatter_obj, rotation_float = build_signature_time_axis_spec(bar_date_idx)
    axis_obj.xaxis.set_major_locator(locator_obj)
    axis_obj.xaxis.set_major_formatter(formatter_obj)
    axis_obj.tick_params(axis='x', labelbottom=True, rotation=rotation_float)


def _apply_minimal_axis_style(axis_obj, vertical_line_iterable: Iterable[object] = ()) -> None:
    """Strip the chart frame down to a baseline and hairline value rules.

    Data-ink discipline: the box around the plot encodes nothing, so it goes.
    What remains is a single bottom baseline plus the faintest horizontal rules
    needed to read a value off the axis. Series stay directly labelled rather
    than relying on a legend box.
    """
    signature_palette_dict = SIGNATURE_PALETTE_DICT

    for side_name_str in ('top', 'left', 'right'):
        axis_obj.spines[side_name_str].set_visible(False)
    axis_obj.spines['bottom'].set_visible(True)
    axis_obj.spines['bottom'].set_color(signature_palette_dict['axes_border'])
    axis_obj.spines['bottom'].set_linewidth(0.6)

    axis_obj.tick_params(
        axis='x', labelsize=7.5, colors=signature_palette_dict['muted'], pad=5, length=2.5, width=0.6
    )
    axis_obj.tick_params(
        axis='y', labelsize=7.5, colors=signature_palette_dict['muted'], pad=5, length=0.0
    )
    axis_obj.grid(
        axis='y',
        which='major',
        linestyle='-',
        linewidth=0.5,
        color=signature_palette_dict['grid'],
        alpha=1.0,
    )
    axis_obj.xaxis.grid(False, which='major')
    axis_obj.xaxis.grid(False, which='minor')
    axis_obj.set_axisbelow(True)
    axis_obj.yaxis.tick_right()
    axis_obj.yaxis.set_label_position('left')

    for vertical_line_obj in vertical_line_iterable:
        axis_obj.axvline(
            vertical_line_obj,
            color=signature_palette_dict['vertical_line'],
            linestyle='-',
            linewidth=0.6,
            alpha=0.8,
            zorder=1,
        )


def blend_hex_color_str(
        start_color_str: str,
        end_color_str: str,
        weight_float: float,
) -> str:
    bounded_weight_float = max(0.0, min(1.0, float(weight_float)))
    start_rgb_tuple = mcolors.to_rgb(start_color_str)
    end_rgb_tuple = mcolors.to_rgb(end_color_str)

    channel_value_list: list[int] = []
    for channel_idx_int in range(3):
        blended_channel_float = (
            (1.0 - bounded_weight_float) * start_rgb_tuple[channel_idx_int]
            + bounded_weight_float * end_rgb_tuple[channel_idx_int]
        )
        channel_value_list.append(int(round(blended_channel_float * 255.0)))

    return '#{0:02x}{1:02x}{2:02x}'.format(*channel_value_list)


def _build_cycle_asset_color_dict(
        overlay_cycle_list: list[str],
        light_color_str: str,
        page_color_str: str,
) -> dict[str, str]:
    """Assign each named asset a hue from a variant's own overlay cycle.

    Desk keeps muted composition hues rather than a grey ramp, so its weight
    stacks stay readable at seven sleeves without borrowing the saturated
    baseline asset colours — which would put a bright blue TLT next to a PASS
    badge and break the rule that colour means state.

    Ordering follows the baseline dict so an asset keeps the same hue across
    runs instead of shifting when the set of held names changes.
    """
    cycle_asset_name_list = [
        asset_name_str for asset_name_str in SIGNATURE_ASSET_COLOR_DICT
        if asset_name_str not in ('CASH', 'DEFAULT')
    ]
    cycle_asset_color_dict: dict[str, str] = {}
    for asset_idx_int, asset_name_str in enumerate(cycle_asset_name_list):
        cycle_asset_color_dict[asset_name_str] = overlay_cycle_list[
            asset_idx_int % len(overlay_cycle_list)
        ]
    # Cash is not an exposure, so it reads as near-empty; unknown names sit mid-ramp.
    cycle_asset_color_dict['CASH'] = blend_hex_color_str(light_color_str, page_color_str, 0.62)
    cycle_asset_color_dict['DEFAULT'] = light_color_str
    return cycle_asset_color_dict


_VARIANT_OVERRIDE_DICT['desk']['asset_color_dict'] = _build_cycle_asset_color_dict(
    overlay_cycle_list=list(_DESK_PALETTE_DICT['overlay_cycle']),
    light_color_str=str(_DESK_PALETTE_DICT['benchmark']),
    page_color_str=str(_DESK_PALETTE_DICT['page']),
)


def _build_palette_root_css(
        palette_dict: dict[str, object],
        figure_font_stack_str: str,
        prose_font_stack_str: str,
) -> str:
    """Emit one palette as the ``:root`` custom-property block.

    Shared by the document report layout and the Bench console so a colour only
    ever has to be named once. Callers pass the font stacks explicitly because
    the dashboard layout deliberately sets figure and prose type to the same
    stack while the document layout does not.
    """
    return f'''
:root {{
    --color-ink: {palette_dict["ink"]};
    --color-page: {palette_dict["page"]};
    --color-panel: {palette_dict["panel"]};
    --color-neutral: {palette_dict["neutral"]};
    --color-grid: {palette_dict["grid"]};
    --color-border: {palette_dict["border"]};
    --color-muted: {palette_dict["muted"]};
    --color-strategy: {palette_dict["strategy"]};
    --color-strategy-dark: {palette_dict["strategy_dark"]};
    --color-benchmark: {palette_dict["benchmark"]};
    --color-benchmark-dark: {palette_dict["benchmark_dark"]};
    --color-profit: {palette_dict["profit"]};
    --color-profit-dark: {palette_dict["profit_dark"]};
    --color-loss: {palette_dict["loss"]};
    --color-loss-dark: {palette_dict["loss_dark"]};
    --color-accent: {palette_dict["accent"]};
    --color-accent-dark: {palette_dict["accent_dark"]};
    --color-warning: {palette_dict["warning"]};
    --color-warning-dark: {palette_dict["warning_dark"]};
    --color-accent-wash: {palette_dict["accent_wash"]};
    --color-profit-wash: {palette_dict["profit_wash"]};
    --color-warning-wash: {palette_dict["warning_wash"]};
    --color-loss-wash: {palette_dict["loss_wash"]};
    --color-shadow: {palette_dict["shadow_rgba"]};
    --font-figure: {figure_font_stack_str};
    --font-prose: {prose_font_stack_str};
    --font-code: {palette_dict["code_font_stack_str"]};
}}'''


def _build_document_report_css() -> str:
    """Single-column research-note grammar over the standard report classes.

    Deliberately reuses every class name emitted by report.py so the same markup
    renders as a note rather than a dashboard with no change to the report
    builders. Containers carry no weight of their own: cards lose their border,
    radius, fill and shadow, and structure comes from typography, whitespace and
    hairline rules instead.
    """
    signature_palette_dict = SIGNATURE_PALETTE_DICT
    figure_font_stack_str = str(signature_palette_dict['font_stack_str'])
    prose_font_stack_str = str(signature_palette_dict['prose_font_stack_str'])
    return _build_palette_root_css(
        signature_palette_dict,
        figure_font_stack_str=figure_font_stack_str,
        prose_font_stack_str=prose_font_stack_str,
    ) + f'''
/* *** UI*** Buttons, inputs and selects do not inherit font-family from the
   page — they fall back to a UA font. Without this the report's own tooltip
   buttons rendered in Arial inside a document set entirely in Plex. */
button, input, select, textarea {{
    font: inherit;
}}
body {{
    font-family: var(--font-prose);
    margin: 0;
    padding: 56px 24px 96px;
    background: var(--color-page);
    color: var(--color-ink);
    line-height: 1.62;
    font-size: 15.5px;
}}
.report-shell {{
    max-width: 880px;
    margin: 0 auto;
}}
.report-header {{
    margin-bottom: 40px;
    padding-bottom: 22px;
    border-bottom: 1.5px solid var(--color-ink);
}}
.report-eyebrow {{
    margin: 0 0 10px;
    font-family: var(--font-figure);
    color: var(--color-muted);
    font-size: 0.66rem;
    font-weight: 400;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}}
h1 {{
    font-family: var(--font-prose);
    font-size: 1.85rem;
    font-weight: 600;
    line-height: 1.2;
    letter-spacing: -0.012em;
    margin: 0 0 8px;
    color: var(--color-ink);
}}
h2 {{
    font-family: var(--font-figure);
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--color-muted);
    margin: 52px 0 20px;
    padding: 0 0 7px;
    background: none;
    border: none;
    border-bottom: 1px solid var(--color-border);
    border-radius: 0;
}}
h3 {{
    font-family: var(--font-figure);
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--color-ink);
    margin: 0 0 12px;
}}
.meta {{
    font-family: var(--font-figure);
    color: var(--color-muted);
    font-size: 0.74rem;
    line-height: 1.55;
    margin: 0 0 3px;
}}
p {{
    color: var(--color-ink);
    margin-top: 0;
    margin-bottom: 1.1em;
    max-width: 68ch;
}}
/* The headline result is stated as a sentence. A row of stat tiles reports
   numbers without ever saying what they mean; a lede has to make a claim. */
.lede {{
    font-size: 1.16rem;
    line-height: 1.62;
    max-width: 60ch;
    margin: 0 0 1.5em;
}}
.fig {{
    font-family: var(--font-figure);
    font-size: 0.9em;
    font-weight: 500;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
}}
.fig.neg {{
    color: var(--color-loss-dark);
}}
/* Headline metrics span the full measure in equal columns rather than packing
   to the left, so the row reads as one balanced band under the masthead. */
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 14px 24px;
    margin: 0 0 12px;
    padding: 16px 0 18px;
    border-top: 1.5px solid var(--color-ink);
    border-bottom: 1px solid var(--color-border);
}}
.kpi-card {{
    background: none;
    border: none;
    border-radius: 0;
    padding: 0;
    box-shadow: none;
    min-width: 118px;
}}
.kpi-label {{
    font-family: var(--font-figure);
    color: var(--color-muted);
    font-size: 0.62rem;
    font-weight: 400;
    letter-spacing: 0.11em;
    text-transform: uppercase;
}}
.kpi-value {{
    font-family: var(--font-figure);
    margin-top: 7px;
    font-size: 1.42rem;
    font-weight: 500;
    line-height: 1.05;
    font-variant-numeric: tabular-nums;
    color: var(--color-ink);
}}
.kpi-value.pos {{
    color: var(--color-ink);
}}
.kpi-value.neg {{
    color: var(--color-loss-dark);
}}
.kpi-note {{
    font-family: var(--font-prose);
    margin-top: 5px;
    color: var(--color-muted);
    font-size: 0.76rem;
    font-style: italic;
}}
/* Containers carry no visual weight; whitespace is the separator. */
.card, .chart-panel {{
    background: none;
    border: none;
    border-radius: 0;
    padding: 0;
    box-shadow: none;
    margin-bottom: 34px;
}}
.card-primary {{
    padding-top: 0;
}}
.card-grid, .crisis-chart-grid, .chart-grid {{
    display: block;
    margin-bottom: 0;
}}
.card-grid > .card, .crisis-chart-grid > .card {{
    margin-bottom: 34px;
}}
.section-stack, .summary-section-stack {{
    display: block;
}}
/* Scientific table rules: horizontal only, figures in tabular monospace. */
table {{
    font-family: var(--font-figure);
    border-collapse: collapse;
    width: 100%;
    font-size: 0.92rem;
    font-variant-numeric: tabular-nums;
    margin-bottom: 0;
}}
th {{
    background: none;
    border: none;
    border-bottom: 1px solid var(--color-ink);
    padding: 0 10px 7px;
    text-align: right;
    font-weight: 600;
    font-size: 0.66rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--color-muted);
    white-space: nowrap;
}}
td {{
    background: none;
    border: none;
    border-bottom: 1px solid var(--color-border);
    padding: 7px 10px;
    text-align: right;
}}
th:first-child, td:first-child, td.metric {{
    text-align: left;
}}
td.metric {{
    background: none;
    font-weight: 400;
    color: var(--color-muted);
    white-space: nowrap;
}}
tbody tr:last-child td {{
    border-bottom: 1px solid var(--color-ink);
}}
/* A row measured over a different window than the rows above it. The rule
   separates it so it is not read as one more step in the same sequence. */
tr.full-sample-row td {{
    border-top: 1px solid var(--color-ink);
}}
td.pos {{
    color: var(--color-profit-dark);
    font-weight: 500;
}}
td.neg {{
    color: var(--color-loss-dark);
    font-weight: 500;
}}
.heatmap td {{
    text-align: center;
    font-size: 0.72rem;
    padding: 6px 4px;
    border-bottom: none;
}}
.heatmap th {{
    text-align: center;
    padding: 0 4px 7px;
}}
.heatmap tbody tr:last-child td {{
    border-bottom: none;
}}
.heatmap {{
    table-layout: fixed;
}}
.heatmap .divider-left {{
    border-left: 1px solid var(--color-border);
}}
.metric-help {{
    appearance: none;
    display: none;
}}
.metric-context {{
    font-family: var(--font-prose);
    margin-top: 3px;
    color: var(--color-muted);
    font-size: 0.72rem;
    font-style: italic;
    font-weight: 400;
    white-space: normal;
}}
.regression-model-note {{
    margin-bottom: 10px;
    color: var(--color-muted);
    font-size: 0.8rem;
    font-style: italic;
}}
.summary-details {{
    border: none;
    border-top: 1px solid var(--color-border);
    border-radius: 0;
    background: none;
    padding: 12px 0 0;
}}
.summary-details summary {{
    font-family: var(--font-figure);
    color: var(--color-muted);
    cursor: pointer;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.11em;
    text-transform: uppercase;
}}
.summary-details[open] summary {{
    margin-bottom: 12px;
}}
.chart-wrap {{
    margin: 0;
}}
.chart-wrap img, .chart-panel img {{
    max-width: 100%;
    width: 100%;
    display: block;
    border: none;
    border-radius: 0;
    background: none;
    box-shadow: none;
}}
.stats-table {{
    width: 100%;
    min-width: 0;
}}
.scroll {{
    overflow-x: auto;
    width: 100%;
    border: none;
    border-radius: 0;
    background: none;
    padding: 0;
    margin-bottom: 0;
}}
strong {{
    color: var(--color-ink);
    font-weight: 600;
}}
@media (max-width: 760px) {{
    body {{
        padding: 32px 18px 64px;
        font-size: 15px;
    }}
    .kpi-grid {{
        gap: 18px 32px;
    }}
}}
'''


def _build_spec_layout_css() -> str:
    """Datasheet layout: ruled field masthead, numbered plates, tight rhythm.

    Plate numbers come from a CSS counter on the plate's own heading, so any
    section wrapped in ``.plate`` is indexed automatically — the report reuses
    its existing ``<h2>`` sections as plates with no per-section renumbering.
    """
    return '''
body {
    font-size: 14.5px;
    padding-top: 40px;
}
.report-shell {
    max-width: 1000px;
    counter-reset: plate;
}
.report-header {
    margin-bottom: 0;
    padding-bottom: 14px;
    border-bottom: 2px solid var(--color-ink);
}
h1 {
    font-family: var(--font-figure);
    font-size: 1.06rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
}
.report-shell > p {
    max-width: 70ch;
}
/* Every section is a numbered plate framed like a datasheet figure, never a
   table or chart floating in whitespace. The plate's own <h2> becomes the
   ruled caption bar, and a CSS counter stamps the plate number onto it. */
.plate {
    counter-increment: plate;
    border: 1px solid var(--color-border);
    background: var(--color-panel);
    padding: 0 13px 13px;
    margin-bottom: 16px;
}
/* Plate headings carry real weight: they are the page's main separator, so
   they read at a glance rather than as fine print. */
.plate > h2 {
    margin: 0 -13px 14px;
    padding: 9px 13px;
    background: var(--color-neutral);
    border: none;
    border-bottom: 1px solid var(--color-ink);
    border-radius: 0;
    font-family: var(--font-figure);
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--color-ink);
}
.plate > h2::before {
    content: "Plate " counter(plate, decimal-leading-zero) " \\2014 ";
}
.plate > h2 ~ h2 {
    counter-increment: none;
}
.plate .chart-wrap img {
    border-radius: 0;
}
/* Index of plates: gives a long sheet a visible shape without hiding any of
   it behind tabs, and prints as plain text. */
.plate-index {
    margin: 18px 0 22px;
    padding: 12px 14px;
    border: 1px solid var(--color-border);
    background: var(--color-neutral);
}
.plate-index ol {
    margin: 0;
    padding-left: 22px;
    columns: 2;
    column-gap: 28px;
    font-family: var(--font-figure);
    font-size: 0.72rem;
    line-height: 1.85;
}
.plate-index a {
    color: var(--color-ink);
    text-decoration: none;
    border-bottom: 1px solid var(--color-border);
}
.plate-index a:hover {
    border-bottom-color: var(--color-ink);
}
@media (max-width: 720px) {
    .plate-index ol {
        columns: 1;
    }
}
.spec-masthead {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(168px, 1fr));
    border-bottom: 1px solid var(--color-ink);
}
.spec-field {
    padding: 8px 12px;
    border-right: 1px solid var(--color-border);
}
.spec-field:last-child {
    border-right: none;
}
.spec-field-label {
    font-family: var(--font-figure);
    font-size: 0.55rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--color-muted);
}
.spec-field-value {
    font-family: var(--font-figure);
    font-size: 0.74rem;
    color: var(--color-ink);
    margin-top: 3px;
}
.card {
    margin-bottom: 22px;
}
table {
    /* The spec layout re-declares the table size after the document
     * stylesheet, so this is the value that actually applies there.
     */
    font-size: 0.86rem;
}
td {
    padding: 5px 10px;
}
'''


def build_report_css() -> str:
    # Every report carries its own type. Prepended here rather than in each
    # layout branch so no layout can ship without it.
    return build_embedded_font_face_css() + '\n' + _build_report_body_css()


def _build_report_body_css() -> str:
    layout_str = str(SIGNATURE_PALETTE_DICT['layout_str'])

    if layout_str in ('document', 'spec'):
        report_css_str = _build_document_report_css()
        if layout_str == 'spec':
            report_css_str += _build_spec_layout_css()
        return report_css_str

    if str(SIGNATURE_PALETTE_DICT['layout_str']) == 'document':
        return _build_document_report_css()

    signature_palette_dict = SIGNATURE_PALETTE_DICT
    report_font_stack_str = str(signature_palette_dict['font_stack_str'])
    return f'''
:root {{
    --color-ink: {signature_palette_dict["ink"]};
    --color-page: {signature_palette_dict["page"]};
    --color-panel: {signature_palette_dict["panel"]};
    --color-neutral: {signature_palette_dict["neutral"]};
    --color-grid: {signature_palette_dict["grid"]};
    --color-border: {signature_palette_dict["border"]};
    --color-muted: {signature_palette_dict["muted"]};
    --color-strategy: {signature_palette_dict["strategy"]};
    --color-strategy-dark: {signature_palette_dict["strategy_dark"]};
    --color-benchmark: {signature_palette_dict["benchmark"]};
    --color-benchmark-dark: {signature_palette_dict["benchmark_dark"]};
    --color-profit: {signature_palette_dict["profit"]};
    --color-profit-dark: {signature_palette_dict["profit_dark"]};
    --color-loss: {signature_palette_dict["loss"]};
    --color-loss-dark: {signature_palette_dict["loss_dark"]};
    --color-accent: {signature_palette_dict["accent"]};
    --color-accent-dark: {signature_palette_dict["accent_dark"]};
    --color-warning: {signature_palette_dict["warning"]};
    --color-warning-dark: {signature_palette_dict["warning_dark"]};
    --color-accent-wash: {signature_palette_dict["accent_wash"]};
    --color-profit-wash: {signature_palette_dict["profit_wash"]};
    --color-warning-wash: {signature_palette_dict["warning_wash"]};
    --color-loss-wash: {signature_palette_dict["loss_wash"]};
    --color-shadow: {signature_palette_dict["shadow_rgba"]};
    /* The analyzer appendix sets figure and prose type on tiles, tables and
       verdict rows. Without these the whole declaration is void and the text
       silently falls back to the body font in every non-document layout. */
    --font-figure: {report_font_stack_str};
    --font-prose: {report_font_stack_str};
    --font-code: {signature_palette_dict["code_font_stack_str"]};
}}
body {{
    font-family: {report_font_stack_str};
    margin: 0;
    padding: 18px 20px 32px;
    background: var(--color-page);
    color: var(--color-ink);
    line-height: 1.45;
}}
.report-shell {{
    max-width: 1480px;
    margin: 0 auto;
}}
.report-header {{
    margin-bottom: 14px;
}}
.report-eyebrow {{
    margin: 0 0 6px;
    color: var(--color-muted);
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}}
h1, h2, h3 {{
    font-family: {report_font_stack_str};
    color: var(--color-ink);
}}
h1 {{
    font-size: 1.72rem;
    margin: 0 0 4px;
    font-weight: 700;
}}
h2 {{
    font-size: 1.18rem;
    line-height: 1.25;
    font-weight: 750;
    margin: 0 0 16px;
    padding: 8px 10px;
    background: var(--color-neutral);
    border-left: 4px solid var(--color-strategy-dark);
    border-radius: 3px;
}}
h3 {{
    color: var(--color-strategy-dark);
    font-size: 0.94rem;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 9px;
}}
.meta {{
    color: var(--color-muted);
    font-size: 0.88rem;
    margin-bottom: 0;
}}
p {{
    color: var(--color-ink);
    margin-top: 0;
}}
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
    gap: 12px;
    margin: 0 0 14px;
}}
.kpi-card {{
    background: var(--color-panel);
    border: 1px solid var(--color-border);
    border-radius: 4px;
    padding: 14px 16px;
    box-shadow: none;
}}
.kpi-label {{
    color: var(--color-ink);
    font-size: 0.92rem;
    font-weight: 600;
}}
.kpi-value {{
    margin-top: 6px;
    font-size: 1.8rem;
    font-weight: 700;
    line-height: 1.1;
}}
.kpi-value.pos {{
    color: var(--color-profit-dark);
}}
.kpi-value.neg {{
    color: var(--color-loss-dark);
}}
.kpi-note {{
    margin-top: 4px;
    color: var(--color-muted);
    font-size: 0.84rem;
}}
.card {{
    background: var(--color-panel);
    border: 1px solid var(--color-border);
    border-radius: 4px;
    padding: 16px 18px;
    box-shadow: none;
    margin-bottom: 12px;
}}
.card-primary {{
    padding-top: 18px;
}}
.card-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
    gap: 12px;
    margin-bottom: 12px;
    align-items: start;
}}
.card-grid > .card {{
    margin-bottom: 0;
}}
.crisis-chart-grid {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
    margin-bottom: 12px;
    align-items: start;
}}
.crisis-chart-grid > .card {{
    margin-bottom: 0;
}}
.crisis-chart-grid .card {{
    padding: 12px 14px;
}}
.crisis-chart-grid .chart-wrap {{
    margin-top: 6px;
}}
.section-stack {{
    display: flex;
    flex-direction: column;
    gap: 12px;
}}
.section-stack > .card {{
    margin-bottom: 0;
}}
table {{
    border-collapse: collapse;
    font-size: 0.85em;
    width: 100%;
    margin-bottom: 0;
}}
th {{
    background: var(--color-neutral);
    padding: 8px 10px;
    text-align: left;
    border: 1px solid var(--color-border);
    font-weight: 600;
}}
td {{
    padding: 7px 10px;
    border: 1px solid var(--color-border);
    background: var(--color-panel);
}}
td.metric {{
    font-weight: 500;
    background: var(--color-panel);
    white-space: nowrap;
}}
/* A row measured over a different window than the rows above it. The rule
   separates it so it is not read as one more step in the same sequence. */
tr.full-sample-row td {{
    border-top: 2px solid var(--color-ink);
}}
.metric-help {{
    appearance: none;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    box-sizing: border-box;
    width: 15px;
    height: 15px;
    margin-left: 3px;
    padding: 0;
    border: 1px solid var(--color-muted);
    border-radius: 50%;
    background: transparent;
    color: var(--color-muted);
    font-family: inherit;
    font-size: 10px;
    font-weight: 700;
    line-height: 1;
    cursor: help;
    vertical-align: middle;
}}
.metric-help:focus {{
    outline: 2px solid var(--color-strategy);
    outline-offset: 2px;
}}
html:not(.metric-tooltip-js-enabled) .metric-help:hover::after,
html:not(.metric-tooltip-js-enabled) .metric-help:focus::after {{
    content: attr(data-help);
    position: fixed;
    right: 16px;
    bottom: 16px;
    z-index: 1000;
    width: max-content;
    max-width: min(320px, calc(100vw - 32px));
    padding: 9px 11px;
    border: 1px solid var(--color-border);
    border-radius: 4px;
    background: var(--color-ink);
    color: var(--color-panel);
    box-shadow: 0 4px 12px var(--color-shadow);
    font-size: 12px;
    font-weight: 400;
    line-height: 1.4;
    text-align: left;
    white-space: normal;
}}
.metric-help-tooltip {{
    position: fixed;
    z-index: 1000;
    max-width: 320px;
    padding: 9px 11px;
    border: 1px solid var(--color-border);
    border-radius: 4px;
    background: var(--color-ink);
    color: var(--color-panel);
    box-shadow: 0 4px 12px var(--color-shadow);
    font-size: 12px;
    font-weight: 400;
    line-height: 1.4;
    white-space: normal;
}}
.metric-help-tooltip[hidden] {{
    display: none;
}}
.metric-context {{
    margin-top: 3px;
    color: var(--color-muted);
    font-size: 0.78em;
    font-weight: 400;
    white-space: normal;
}}
.regression-model-note {{
    margin-bottom: 8px;
    color: var(--color-muted);
    font-size: 0.82em;
}}
.summary-section-stack {{
    display: flex;
    flex-direction: column;
    gap: 18px;
}}
.summary-section h3 {{
    margin-bottom: 8px;
}}
.summary-details {{
    border: 1px solid var(--color-border);
    border-radius: 4px;
    background: var(--color-neutral);
    padding: 10px 12px;
}}
.summary-details summary {{
    color: var(--color-ink);
    cursor: pointer;
    font-size: 0.95em;
    font-weight: 600;
}}
.summary-details[open] summary {{
    margin-bottom: 10px;
}}
td.pos {{
    color: var(--color-profit-dark);
    font-weight: 600;
}}
td.neg {{
    color: var(--color-loss-dark);
    font-weight: 600;
}}
.heatmap td {{
    text-align: center;
    font-size: 0.92em;
    min-width: 64px;
    padding: 7px 9px;
}}
.heatmap th {{
    text-align: center;
    font-size: 0.88em;
    padding: 7px 9px;
}}
.heatmap .divider-left {{
    border-left: 3px solid var(--color-muted);
}}
.heatmap {{
    table-layout: fixed;
}}
.card-monthly-returns .scroll {{
    overflow-x: visible;
}}
.chart-wrap {{
    margin: 0;
}}
.chart-wrap img,
.chart-panel img {{
    max-width: 100%;
    width: 100%;
    display: block;
    border: none;
    border-radius: 4px;
    background: var(--color-panel);
    box-shadow: none;
}}
.chart-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 12px;
    margin-top: 12px;
    margin-bottom: 12px;
}}
.chart-panel {{
    background: var(--color-panel);
    border: 1px solid var(--color-border);
    border-radius: 4px;
    padding: 12px;
}}
.stats-table {{
    width: auto;
    min-width: 420px;
}}
.scroll {{
    overflow-x: auto;
    width: 100%;
    border: none;
    border-radius: 0;
    background: transparent;
    padding: 0;
    margin-bottom: 0;
}}
strong {{
    color: var(--color-ink);
}}
@media (max-width: 960px) {{
    body {{
        padding: 16px 14px 24px;
    }}
    .card {{
        padding: 16px;
    }}
    .card-grid {{
        grid-template-columns: 1fr;
    }}
    .crisis-chart-grid {{
        grid-template-columns: 1fr;
    }}
}}
'''
