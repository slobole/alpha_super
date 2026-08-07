"""Guards for the signature theme variant seam.

The variant machinery is a presentation-layer pilot. These tests pin the
property that makes it safe to land: resolving or activating the base variant
must leave the shipping appearance bit-for-bit unchanged.
"""

from __future__ import annotations

import pytest

from alpha.engine.theme import (
    SIGNATURE_PALETTE_DICT,
    SIGNATURE_VARIANT_NAME_LIST,
    build_report_css,
    build_signature_rcparams,
    resolve_variant_palette_dict,
    signature_variant_context,
)


def test_base_variant_resolves_to_shipping_palette():
    assert resolve_variant_palette_dict('current') == SIGNATURE_PALETTE_DICT


def test_base_variant_context_is_a_no_op():
    before_palette_dict = dict(SIGNATURE_PALETTE_DICT)
    before_css_str = build_report_css()
    before_rcparams_dict = build_signature_rcparams(to_web_bool=True)

    with signature_variant_context('current'):
        assert dict(SIGNATURE_PALETTE_DICT) == before_palette_dict
        assert build_report_css() == before_css_str
        assert str(build_signature_rcparams(to_web_bool=True)) == str(before_rcparams_dict)


def test_variant_context_restores_palette_on_exit():
    before_palette_dict = dict(SIGNATURE_PALETTE_DICT)

    with signature_variant_context('desk'):
        assert SIGNATURE_PALETTE_DICT['strategy'] != before_palette_dict['strategy']

    assert dict(SIGNATURE_PALETTE_DICT) == before_palette_dict


def test_variant_context_restores_palette_after_exception():
    before_palette_dict = dict(SIGNATURE_PALETTE_DICT)

    with pytest.raises(RuntimeError):
        with signature_variant_context('desk'):
            raise RuntimeError('boom')

    assert dict(SIGNATURE_PALETTE_DICT) == before_palette_dict


@pytest.mark.parametrize('variant_name_str', SIGNATURE_VARIANT_NAME_LIST)
def test_every_variant_is_complete_and_renderable(variant_name_str):
    resolved_palette_dict = resolve_variant_palette_dict(variant_name_str)
    assert set(resolved_palette_dict) == set(SIGNATURE_PALETTE_DICT)

    with signature_variant_context(variant_name_str):
        rcparams_dict = build_signature_rcparams(to_web_bool=True)
        font_family_str = str(SIGNATURE_PALETTE_DICT['font_family_str'])
        assert rcparams_dict['font.family'] == font_family_str
        assert rcparams_dict[f'font.{font_family_str}'] == list(
            SIGNATURE_PALETTE_DICT['font_stack_list']
        )
        # The CSS must carry this variant's own accent, not a leaked one.
        assert str(SIGNATURE_PALETTE_DICT['strategy']) in build_report_css()


def test_unknown_variant_fails_loud():
    with pytest.raises(ValueError, match='Unknown signature variant'):
        resolve_variant_palette_dict('does-not-exist')
