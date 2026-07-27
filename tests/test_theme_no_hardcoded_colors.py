"""Drift guard: presentation modules must not hardcode their own colours.

Every colour belongs in ``alpha/engine/theme.py``. When a chart or report
builder inlines its own hex literal, that mark stops following the active
signature variant — which is exactly how a monochrome report ends up with a
blue sleeve in its weight stack, and how a second, silently diverging palette
took root in the Bench stylesheet.

This test fails on any new hex literal outside the theme so the identity does
not decay one convenient exception at a time.
"""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

REPO_ROOT_PATH = Path(__file__).resolve().parents[1]

# Presentation modules that must source every colour from the theme.
_GUARDED_MODULE_PATH_TUPLE = (
    Path('alpha/engine/report.py'),
    Path('alpha/engine/plot.py'),
    Path('alpha/engine/signature.py'),
    Path('alpha/engine/stress_test.py'),
    Path('alpha/engine/execution_timing.py'),
    Path('alpha/engine/capacity_analysis.py'),
    Path('alpha/engine/risk_analysis.py'),
)

# Modules still carrying their own palette. Each entry is a debt to clear, not
# a permanent exemption: the guard covers the module the moment it is
# converted, and the list is expected to shrink to empty.
#
# alpha/live/reference_compare.py is live-path code and is being handled
# separately from the research analyzers.
_PENDING_CONVERSION_MODULE_PATH_TUPLE = (
    Path('alpha/live/reference_compare.py'),
)

_HEX_COLOR_PATTERN = re.compile(r'#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?\b')

# Colours that are structural rather than thematic, with the reason they stay.
_ALLOWED_HEX_COLOR_DICT = {
    '#101418': 'weight-stack edge; folded into the theme when stacks are restyled',
}


def _hex_literal_finding_list(module_path: Path) -> list[str]:
    finding_list: list[str] = []
    source_text_str = (REPO_ROOT_PATH / module_path).read_text(encoding='utf-8')
    for line_number_int, line_str in enumerate(source_text_str.splitlines(), start=1):
        stripped_line_str = line_str.strip()
        # Comments and docstring prose may name a colour when explaining one.
        if stripped_line_str.startswith('#'):
            continue
        for hex_color_str in _HEX_COLOR_PATTERN.findall(line_str):
            if hex_color_str.lower() in _ALLOWED_HEX_COLOR_DICT:
                continue
            finding_list.append(f'{module_path}:{line_number_int}: {hex_color_str}  |  {stripped_line_str[:90]}')
    return finding_list


class ThemeColorOwnershipTests(unittest.TestCase):
    def test_presentation_modules_have_no_hardcoded_hex_colors(self):
        finding_list: list[str] = []
        for module_path in _GUARDED_MODULE_PATH_TUPLE:
            finding_list.extend(_hex_literal_finding_list(module_path))

        self.assertEqual(
            finding_list,
            [],
            'Hardcoded colours found outside alpha/engine/theme.py. Move them into the '
            'palette so they follow the active signature variant:\n  '
            + '\n  '.join(finding_list),
        )

    def test_bench_stylesheet_tokens_come_from_the_theme(self):
        """Bench must not reintroduce its own palette under :root."""
        from alpha.engine.theme import build_bench_theme_css

        bench_theme_css_str = build_bench_theme_css()
        self.assertIn('--accent:', bench_theme_css_str)
        self.assertIn('--text:', bench_theme_css_str)
        # The generated block must actually carry the journal ink, not the
        # stylesheet's original blue accent.
        self.assertNotIn('#0c8ce0', bench_theme_css_str)

    def test_every_referenced_css_variable_is_defined(self):
        """An undefined custom property silently voids its whole declaration.

        ``border-top: 2px solid var(--color-text)`` where ``--color-text`` does
        not exist is not a wrong colour — the browser drops the rule entirely
        and the border never renders. Nothing else in the suite notices, because
        the CSS text still contains exactly what was written.
        """
        from alpha.engine.theme import (
            build_analyzer_report_css,
            build_bench_theme_css,
            build_report_css,
            signature_variant_context,
        )

        builder_dict = {
            'build_report_css': build_report_css,
            'build_analyzer_report_css': build_analyzer_report_css,
            'build_bench_theme_css': build_bench_theme_css,
        }
        # The stylesheets branch on the active variant, so a rule can exist in
        # one layout and be missing from another. Check every variant.
        finding_list: list[str] = []
        for variant_name_str in ('current', 'journal', 'journal_spec'):
            for builder_name_str, builder_callable in builder_dict.items():
                with signature_variant_context(variant_name_str):
                    stylesheet_str = builder_callable()
                defined_name_set = set(re.findall(r'(--[\w-]+)\s*:', stylesheet_str))
                referenced_name_set = set(re.findall(r'var\(\s*(--[\w-]+)', stylesheet_str))
                for undefined_name_str in sorted(referenced_name_set - defined_name_set):
                    finding_list.append(
                        f'{builder_name_str} [{variant_name_str}]: {undefined_name_str}'
                    )

        self.assertEqual(
            finding_list,
            [],
            'CSS variables referenced but never defined in the same stylesheet. '
            'Every declaration using them is dropped at render time:\n  '
            + '\n  '.join(finding_list),
        )

    def test_no_theme_placeholder_is_left_unrendered(self):
        """A plain string holding an f-string placeholder emits it verbatim.

        These modules were converted from hex literals to palette lookups by
        bulk edit, which left ``"stroke=\\"{SIGNATURE_PALETTE_DICT['grid']}\\""``
        on strings that were never marked ``f``. The result reaches the page as
        a literal brace expression, and since it is not a valid colour the SVG
        element falls back to a stroke of ``none`` -- an invisible gridline, not
        a wrong one. No hex literal is involved, so the colour guard is blind
        to it and the source reads as correct.
        """
        placeholder_pattern = re.compile(r'\{\s*(?:SIGNATURE_PALETTE_DICT|blend_hex_color_str)')
        finding_list: list[str] = []
        for module_path in _GUARDED_MODULE_PATH_TUPLE + _PENDING_CONVERSION_MODULE_PATH_TUPLE:
            source_text_str = (REPO_ROOT_PATH / module_path).read_text(encoding='utf-8')
            for node in ast.walk(ast.parse(source_text_str)):
                # A JoinedStr (f-string) renders its placeholders; a bare
                # Constant cannot, so any placeholder inside one is dead text.
                if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                    continue
                if placeholder_pattern.search(node.value):
                    finding_list.append(
                        f'{module_path}:{node.lineno}: {node.value.strip()[:80]}'
                    )

        self.assertEqual(
            finding_list,
            [],
            'Theme placeholders inside non-f-strings. They are emitted literally '
            'and render as an invalid value:\n  ' + '\n  '.join(finding_list),
        )

    def test_guarded_modules_exist(self):
        for module_path in _GUARDED_MODULE_PATH_TUPLE:
            self.assertTrue((REPO_ROOT_PATH / module_path).exists(), module_path)

    def test_pending_modules_are_tracked_and_shrinking(self):
        """Keep the unconverted analyzers visible instead of silently excluded.

        A module that still hardcodes colours belongs on the pending list; one
        that no longer does belongs in the guard. This fails either way round,
        so the debt cannot quietly grow or be forgotten once paid.
        """
        for module_path in _PENDING_CONVERSION_MODULE_PATH_TUPLE:
            self.assertTrue((REPO_ROOT_PATH / module_path).exists(), module_path)
            self.assertGreater(
                len(_hex_literal_finding_list(module_path)),
                0,
                f'{module_path} no longer hardcodes colours — move it into '
                '_GUARDED_MODULE_PATH_TUPLE.',
            )


if __name__ == '__main__':
    unittest.main()


class AnalyzerMatchesVanillaTests(unittest.TestCase):
    """The analyzers must render as the same product as the strategy report."""

    _ANALYZER_MODULE_NAME_TUPLE = (
        'alpha.engine.risk_analysis',
        'alpha.engine.capacity_analysis',
        'alpha.engine.execution_timing',
        'alpha.engine.stress_test',
    )

    def test_every_analyzer_renders_inside_the_active_variant(self):
        """A report built outside the context uses the baseline dashboard palette.

        stress_test was the one analyzer missing this, so it alone came out in
        the old blue palette and Atlassian Sans while the rest of the book had
        moved to the journal signature. The palette is read at render time, so
        the omission is invisible until you look at the page.
        """
        import importlib

        finding_list: list[str] = []
        for module_name_str in self._ANALYZER_MODULE_NAME_TUPLE:
            source_text_str = Path(
                importlib.import_module(module_name_str).__file__
            ).read_text(encoding='utf-8')
            if 'signature_variant_context(_ACTIVE_REPORT_VARIANT_STR)' not in source_text_str:
                finding_list.append(module_name_str)
        self.assertEqual(finding_list, [], f'Analyzers rendering outside the variant: {finding_list}')

    def test_analyzers_do_not_load_the_bare_report_stylesheet(self):
        """build_analyzer_report_css maps .panel/.card/.muted onto the theme.

        Loading build_report_css directly skips that mapping, so a report using
        the analyzer class vocabulary gets none of it.
        """
        import importlib
        import re

        finding_list: list[str] = []
        for module_name_str in self._ANALYZER_MODULE_NAME_TUPLE:
            source_text_str = Path(
                importlib.import_module(module_name_str).__file__
            ).read_text(encoding='utf-8')
            if re.search(r'(?<!analyzer_)build_report_css\(\)', source_text_str):
                finding_list.append(module_name_str)
        self.assertEqual(finding_list, [], f'Analyzers on the bare stylesheet: {finding_list}')

    def test_analyzer_section_headings_match_the_plate_heading(self):
        """Same size, weight and colour as the strategy report's plate caption."""
        import re

        from alpha.engine.theme import (
            build_analyzer_report_css,
            build_report_css,
            signature_variant_context,
        )

        with signature_variant_context('journal_spec'):
            analyzer_css_str = build_analyzer_report_css()
            report_css_str = build_report_css()

        def declaration_dict(css_str: str, selector_str: str) -> dict[str, str]:
            match_obj = re.search(re.escape(selector_str) + r'[^{]*\{([^}]*)\}', css_str)
            assert match_obj is not None, selector_str
            return {
                key_str.strip(): value_str.strip()
                for key_str, _s, value_str in (
                    declaration_str.partition(':')
                    for declaration_str in match_obj.group(1).split(';')
                )
                if key_str.strip()
            }

        plate_dict = declaration_dict(report_css_str, '.plate > h2')
        analyzer_dict = declaration_dict(analyzer_css_str, '.panel > h2, .card > h2')
        for property_name_str in ('font-size', 'font-weight', 'color', 'text-transform'):
            self.assertEqual(
                analyzer_dict.get(property_name_str),
                plate_dict.get(property_name_str),
                f'{property_name_str} differs between the analyzers and the report',
            )
