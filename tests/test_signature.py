"""Guards for the signature marks (title block, sparklines, small multiples)."""

from __future__ import annotations

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from alpha.engine.signature import (
    apply_figure_stamp,
    build_metric_delta_table_html,
    build_metric_deviation_html,
    build_metric_dumbbell_html,
    build_metric_scale_html,
    build_sparkline_img_html,
    build_title_block_html,
    compute_conditional_beta_dict,
    compute_holding_period_length_list,
    detect_composition_mode_str,
    render_composition_data_uri_str,
    render_relative_performance_data_uri_str,
    render_small_multiples_data_uri_str,
    render_sparkline_data_uri_str,
)
from alpha.engine.theme import signature_variant_context


@pytest.fixture
def value_ser():
    return pd.Series(np.linspace(0.0, 0.2, 60) + np.sin(np.linspace(0, 8, 60)) * 0.03)


def test_sparkline_renders_png_data_uri(value_ser):
    data_uri_str = render_sparkline_data_uri_str(value_ser)
    assert data_uri_str.startswith('data:image/png;base64,')
    assert len(data_uri_str) > 200


@pytest.mark.parametrize('short_ser', [pd.Series(dtype=float), pd.Series([1.0])])
def test_sparkline_declines_degenerate_series(short_ser):
    assert render_sparkline_data_uri_str(short_ser) == ''
    assert build_sparkline_img_html(short_ser) == ''


def test_sparkline_img_is_inline_and_sized(value_ser):
    img_html_str = build_sparkline_img_html(value_ser, width_px_int=58, height_px_int=13)
    assert 'width="58"' in img_html_str
    assert 'height="13"' in img_html_str
    assert 'display:inline-block' in img_html_str


def test_small_multiples_renders_every_panel(value_ser):
    panel_ser_dict = {str(2000 + offset_int): value_ser * (1 + offset_int) for offset_int in range(6)}
    data_uri_str = render_small_multiples_data_uri_str(panel_ser_dict, column_count_int=3)
    assert data_uri_str.startswith('data:image/png;base64,')


def test_small_multiples_requires_panels():
    with pytest.raises(ValueError, match='at least one panel'):
        render_small_multiples_data_uri_str({})


def test_small_multiples_shares_the_vertical_scale(value_ser):
    """A shared scale is the whole point; per-panel autoscaling would mislead."""
    quiet_ser = value_ser * 0.05
    loud_ser = value_ser * 5.0

    shared_uri_str = render_small_multiples_data_uri_str(
        {'quiet': quiet_ser, 'loud': loud_ser}, column_count_int=2, share_ylim_bool=True
    )
    unshared_uri_str = render_small_multiples_data_uri_str(
        {'quiet': quiet_ser, 'loud': loud_ser}, column_count_int=2, share_ylim_bool=False
    )
    # Under a shared scale the quiet panel is visibly flat, so the two renders
    # must differ; identical output would mean sharing was not applied.
    assert shared_uri_str != unshared_uri_str


def test_title_block_lists_every_field():
    title_block_html_str = build_title_block_html(
        [('Run id', '20260723T101500Z'), ('Costs', '5bp one-way')]
    )
    assert 'Run id' in title_block_html_str
    assert '20260723T101500Z' in title_block_html_str
    assert '5bp one-way' in title_block_html_str


def test_title_block_escapes_field_values():
    title_block_html_str = build_title_block_html([('Note', '<script>x</script>')])
    assert '<script>' not in title_block_html_str
    assert '&lt;script&gt;' in title_block_html_str


def test_title_block_requires_fields():
    with pytest.raises(ValueError, match='at least one field'):
        build_title_block_html([])


def test_figure_stamp_writes_text():
    figure_obj = plt.figure()
    apply_figure_stamp(figure_obj, 'run 42 · norgate 2026-06-30')
    assert any('run 42' in text_obj.get_text() for text_obj in figure_obj.texts)
    plt.close(figure_obj)


def _metric_spec_dict(**override_dict):
    base_spec_dict = {
        'label_str': 'CAGR', 'value_float': 0.06, 'display_str': '6.0%',
        'benchmark_float': 0.04, 'domain_min_float': 0.0, 'domain_max_float': 0.15,
    }
    base_spec_dict.update(override_dict)
    return base_spec_dict


def test_metric_scale_positions_value_and_benchmark():
    scale_html_str = build_metric_scale_html([_metric_spec_dict()])
    # 0.06 across a 0.00-0.15 domain is 40%; the benchmark rule sits at 26.67%.
    assert 'width:40.00%' in scale_html_str
    assert 'left:26.67%' in scale_html_str


def test_metric_scale_clamps_out_of_domain_readings():
    """An out-of-domain value must pin to the end of the track, not overflow it."""
    scale_html_str = build_metric_scale_html(
        [_metric_spec_dict(value_float=0.9, benchmark_float=-0.5)]
    )
    assert 'width:100.00%' in scale_html_str
    assert 'left:0.00%' in scale_html_str


def test_metric_scale_marks_adverse_metrics_differently():
    plain_html_str = build_metric_scale_html([_metric_spec_dict()])
    adverse_html_str = build_metric_scale_html([_metric_spec_dict(is_adverse_bool=True)])
    assert 'var(--color-strategy)' in plain_html_str
    assert 'var(--color-loss)' in adverse_html_str


def test_metric_scale_rejects_degenerate_domain():
    with pytest.raises(ValueError, match='non-positive scale domain'):
        build_metric_scale_html(
            [_metric_spec_dict(domain_min_float=0.1, domain_max_float=0.1)]
        )


def test_metric_scale_omits_benchmark_rule_when_absent():
    benchmark_rule_marker_str = 'background:var(--color-ink)'

    with_benchmark_html_str = build_metric_scale_html([_metric_spec_dict()])
    without_benchmark_html_str = build_metric_scale_html(
        [_metric_spec_dict(benchmark_float=None)]
    )

    assert benchmark_rule_marker_str in with_benchmark_html_str
    assert benchmark_rule_marker_str not in without_benchmark_html_str
    assert 'width:40.00%' in without_benchmark_html_str


@pytest.mark.parametrize('builder_fn', [build_metric_scale_html, build_metric_delta_table_html])
def test_metric_builders_require_metrics(builder_fn):
    with pytest.raises(ValueError, match='at least one metric'):
        builder_fn([])


def test_dumbbell_places_both_dots():
    dumbbell_html_str = build_metric_dumbbell_html([_metric_spec_dict()])
    assert 'left:40.00%' in dumbbell_html_str
    assert 'left:26.67%' in dumbbell_html_str


def test_dumbbell_omits_benchmark_dot_when_absent():
    dumbbell_html_str = build_metric_dumbbell_html([_metric_spec_dict(benchmark_float=None)])
    assert 'left:40.00%' in dumbbell_html_str
    assert 'left:26.67%' not in dumbbell_html_str


@pytest.mark.parametrize(
    ('higher_is_better_bool', 'expected_side_str', 'unexpected_side_str'),
    [(True, 'left', 'right'), (False, 'right', 'left')],
)
def test_deviation_direction_follows_metric_polarity(
    higher_is_better_bool, expected_side_str, unexpected_side_str
):
    """A higher reading is only 'better' when the metric says higher is better."""
    deviation_html_str = build_metric_deviation_html(
        [_metric_spec_dict(higher_is_better_bool=higher_is_better_bool)]
    )
    # 0.06 vs 0.04 over a 0.15 domain is a 13.33% deviation. Favourable bars
    # grow rightwards from the centre rule, adverse ones leftwards. Matched
    # together with the width so the centre rule's own 'left:50%' cannot pass.
    assert f'{expected_side_str}:50%;width:13.33%' in deviation_html_str
    assert f'{unexpected_side_str}:50%;width:13.33%' not in deviation_html_str


def test_deviation_saturates_instead_of_escaping_its_axis():
    deviation_html_str = build_metric_deviation_html(
        [_metric_spec_dict(value_float=0.15, benchmark_float=-5.0)]
    )
    assert 'width:50.00%' in deviation_html_str


def test_delta_table_colours_by_favourability():
    better_html_str = build_metric_delta_table_html(
        [_metric_spec_dict(higher_is_better_bool=True, delta_display_str='+2.0pp')]
    )
    worse_html_str = build_metric_delta_table_html(
        [_metric_spec_dict(higher_is_better_bool=False, delta_display_str='+2.0pp')]
    )
    assert 'var(--color-profit-dark)' in better_html_str
    assert 'var(--color-loss-dark)' in worse_html_str


def _rotation_holding_df(bar_count_int=180, name_count_int=25, slot_count_int=5):
    random_generator = np.random.default_rng(7)
    bar_date_idx = pd.bdate_range('2024-01-01', periods=bar_count_int)
    name_list = [f'N{idx:02d}' for idx in range(name_count_int)]
    holding_matrix = np.zeros((bar_count_int, name_count_int))
    for bar_idx_int in range(bar_count_int):
        held_idx_vec = random_generator.choice(name_count_int, slot_count_int, replace=False)
        holding_matrix[bar_idx_int, held_idx_vec] = 1.0 / slot_count_int
    return pd.DataFrame(holding_matrix, index=bar_date_idx, columns=name_list)


def test_detects_sleeve_book():
    sleeve_df = _rotation_holding_df(name_count_int=4, slot_count_int=4)
    assert detect_composition_mode_str(sleeve_df) == 'sleeve'


def test_detects_rotation_book():
    rotation_df = _rotation_holding_df(name_count_int=40, slot_count_int=5)
    assert detect_composition_mode_str(rotation_df) == 'rotation'


def test_detection_uses_distinct_names_not_concurrency():
    """A book holding few names at once but many over time is still a rotation."""
    rotation_df = _rotation_holding_df(name_count_int=40, slot_count_int=3)
    assert int(rotation_df.gt(0.0).sum(axis=1).max()) == 3
    assert detect_composition_mode_str(rotation_df) == 'rotation'


@pytest.mark.parametrize('mode_str', ['sleeve', 'rotation'])
def test_composition_renders_each_mode(mode_str):
    data_uri_str, resolved_mode_str = render_composition_data_uri_str(
        _rotation_holding_df(name_count_int=40, slot_count_int=5),
        slot_capacity_int=5,
        composition_mode_str=mode_str,
    )
    assert data_uri_str.startswith('data:image/png;base64,')
    assert resolved_mode_str == mode_str


def test_composition_auto_detects_when_mode_omitted():
    _, resolved_mode_str = render_composition_data_uri_str(
        _rotation_holding_df(name_count_int=4, slot_count_int=4)
    )
    assert resolved_mode_str == 'sleeve'


def test_composition_rejects_unknown_mode():
    with pytest.raises(ValueError, match='Unknown composition mode'):
        render_composition_data_uri_str(
            _rotation_holding_df(), composition_mode_str='barcode'
        )


def test_composition_rejects_empty_book():
    with pytest.raises(ValueError, match='no held positions'):
        render_composition_data_uri_str(_rotation_holding_df() * 0.0)
    with pytest.raises(ValueError, match='at least one name column'):
        render_composition_data_uri_str(
            pd.DataFrame(index=pd.bdate_range('2024-01-01', periods=3))
        )


def test_holding_periods_split_re_entries_into_separate_spells():
    """Two separate holds must not merge into one long span."""
    bar_date_idx = pd.bdate_range('2024-01-01', periods=10)
    holding_df = pd.DataFrame({'A': [1, 1, 0, 0, 1, 1, 1, 0, 0, 0]}, index=bar_date_idx, dtype=float)
    assert sorted(compute_holding_period_length_list(holding_df)) == [2, 3]


def test_holding_periods_count_a_spell_open_at_the_end():
    bar_date_idx = pd.bdate_range('2024-01-01', periods=5)
    holding_df = pd.DataFrame({'A': [0, 0, 1, 1, 1]}, index=bar_date_idx, dtype=float)
    assert compute_holding_period_length_list(holding_df) == [3]


def _conditional_return_pair(down_beta_float, up_beta_float, observation_count_int=900):
    random_generator = np.random.default_rng(11)
    benchmark_return_vec = random_generator.normal(0.0004, 0.011, observation_count_int)
    conditional_beta_vec = np.where(benchmark_return_vec < 0.0, down_beta_float, up_beta_float)
    strategy_return_vec = conditional_beta_vec * benchmark_return_vec
    return pd.Series(strategy_return_vec), pd.Series(benchmark_return_vec)


def test_conditional_beta_recovers_the_generating_slopes():
    strategy_ser, benchmark_ser = _conditional_return_pair(0.30, 0.70)
    conditional_metric_dict = compute_conditional_beta_dict(strategy_ser, benchmark_ser)

    assert conditional_metric_dict['down_beta_float'] == pytest.approx(0.30, abs=0.02)
    assert conditional_metric_dict['up_beta_float'] == pytest.approx(0.70, abs=0.02)
    assert conditional_metric_dict['beta_asymmetry_float'] == pytest.approx(0.40, abs=0.04)


def test_conditional_beta_is_symmetric_for_a_constant_beta_process():
    """A constant-beta book must report ~zero asymmetry, not a spurious edge."""
    strategy_ser, benchmark_ser = _conditional_return_pair(0.50, 0.50)
    conditional_metric_dict = compute_conditional_beta_dict(strategy_ser, benchmark_ser)
    assert conditional_metric_dict['beta_asymmetry_float'] == pytest.approx(0.0, abs=0.02)


def test_conditional_beta_conditions_on_the_benchmark_not_the_strategy():
    """Selecting on the strategy's own sign would flatter it by construction.

    Here the strategy is pure noise, independent of the benchmark. Conditioning
    correctly (on the benchmark) must therefore find no asymmetry.
    """
    random_generator = np.random.default_rng(5)
    benchmark_ser = pd.Series(random_generator.normal(0.0004, 0.011, 1500))
    strategy_ser = pd.Series(random_generator.normal(0.0003, 0.009, 1500))

    conditional_metric_dict = compute_conditional_beta_dict(strategy_ser, benchmark_ser)
    assert conditional_metric_dict['beta_asymmetry_float'] == pytest.approx(0.0, abs=0.12)


def test_conditional_beta_splits_every_observation():
    strategy_ser, benchmark_ser = _conditional_return_pair(0.4, 0.6, observation_count_int=750)
    conditional_metric_dict = compute_conditional_beta_dict(strategy_ser, benchmark_ser)
    total_day_count_float = (
        conditional_metric_dict['down_day_count_float']
        + conditional_metric_dict['up_day_count_float']
    )
    assert total_day_count_float == 750.0


def test_conditional_beta_requires_enough_observations():
    with pytest.raises(ValueError, match='at least three overlapping'):
        compute_conditional_beta_dict(pd.Series([0.01, -0.01]), pd.Series([0.02, -0.02]))


def test_conditional_beta_fails_loud_when_a_regime_is_empty():
    """An all-up sample cannot yield a down-beta; say so instead of guessing."""
    benchmark_ser = pd.Series(np.linspace(0.001, 0.02, 40))
    strategy_ser = benchmark_ser * 0.5
    with pytest.raises(ValueError, match='down-market observations'):
        compute_conditional_beta_dict(strategy_ser, benchmark_ser)


def _total_value_ser(daily_return_float, bar_count_int=400, start_date_str='2020-01-01'):
    bar_date_idx = pd.bdate_range(start_date_str, periods=bar_count_int)
    return pd.Series(
        10_000.0 * np.cumprod(np.full(bar_count_int, 1.0 + daily_return_float)),
        index=bar_date_idx,
    )


def test_relative_performance_renders():
    data_uri_str = render_relative_performance_data_uri_str(
        _total_value_ser(0.0006), _total_value_ser(0.0003)
    )
    assert data_uri_str.startswith('data:image/png;base64,')


def test_relative_performance_handles_a_flat_ratio():
    """Identical series give a constant ratio of 1.0 — must render, not crash
    on a degenerate log-scale range."""
    strategy_ser = _total_value_ser(0.0005)
    data_uri_str = render_relative_performance_data_uri_str(strategy_ser, strategy_ser.copy())
    assert data_uri_str.startswith('data:image/png;base64,')


def test_relative_performance_aligns_on_common_dates():
    """Different calendars must intersect, not fabricate relative performance
    from the non-overlapping stub."""
    strategy_ser = _total_value_ser(0.0006, start_date_str='2020-01-01')
    benchmark_ser = _total_value_ser(0.0003, start_date_str='2020-06-01')
    data_uri_str = render_relative_performance_data_uri_str(strategy_ser, benchmark_ser)
    assert data_uri_str.startswith('data:image/png;base64,')


def test_relative_performance_requires_overlap():
    strategy_ser = _total_value_ser(0.0006, bar_count_int=50, start_date_str='2020-01-01')
    benchmark_ser = _total_value_ser(0.0003, bar_count_int=50, start_date_str='2024-01-01')
    with pytest.raises(ValueError, match='overlapping total-value'):
        render_relative_performance_data_uri_str(strategy_ser, benchmark_ser)


def test_relative_performance_rejects_nonpositive_values():
    strategy_ser = _total_value_ser(0.0006)
    broken_benchmark_ser = _total_value_ser(0.0003)
    broken_benchmark_ser.iloc[10] = 0.0
    with pytest.raises(ValueError, match='strictly positive'):
        render_relative_performance_data_uri_str(strategy_ser, broken_benchmark_ser)


def test_marks_follow_the_active_variant(value_ser):
    """Sparkline colour must come from whichever theme is active."""
    with signature_variant_context('journal'):
        journal_uri_str = render_sparkline_data_uri_str(value_ser)
    with signature_variant_context('current'):
        current_uri_str = render_sparkline_data_uri_str(value_ser)

    assert journal_uri_str != current_uri_str


def _decode_small_multiples_png_bytes(data_uri_str: str) -> bytes:
    import base64

    return base64.b64decode(data_uri_str.split(",", 1)[1])


class TestSmallMultiplesOverlay:
    """The reference line must be drawn, and must not be scaled off the panel."""

    def test_overlay_series_widen_the_shared_range(self):
        """*** CRITICAL*** regression: a benchmark that fell further than the
        strategy is exactly the case the comparison exists for. Scaling to the
        panel series alone clips it, and clipping is silent -- the grid would
        understate how much the strategy avoided.
        """
        panel_ser_dict = {"crisis": pd.Series([0.0, -0.05, -0.10])}
        overlay_ser_dict = {"crisis": pd.Series([0.0, -0.30, -0.44])}

        without_overlay_uri_str = render_small_multiples_data_uri_str(
            panel_ser_dict, column_count_int=1,
            value_formatter_fn=lambda v: f"{v * 100:.0f}%",
        )
        with_overlay_uri_str = render_small_multiples_data_uri_str(
            panel_ser_dict, column_count_int=1,
            value_formatter_fn=lambda v: f"{v * 100:.0f}%",
            overlay_ser_dict=overlay_ser_dict,
        )
        # Different y range and an extra line means different pixels.
        assert _decode_small_multiples_png_bytes(
            without_overlay_uri_str
        ) != _decode_small_multiples_png_bytes(with_overlay_uri_str)

    def test_a_panel_without_an_overlay_still_renders(self):
        """Overlays are keyed by panel name; a missing key is not an error."""
        data_uri_str = render_small_multiples_data_uri_str(
            {"a": pd.Series([0.0, 0.1]), "b": pd.Series([0.0, -0.1])},
            column_count_int=2,
            overlay_ser_dict={"a": pd.Series([0.0, 0.05])},
        )
        assert data_uri_str.startswith("data:image/png;base64,")

    def test_omitting_the_overlay_keeps_the_previous_output(self):
        """The parameter is additive: existing callers must be unaffected."""
        panel_ser_dict = {"a": pd.Series([0.0, 0.1, 0.05])}
        assert render_small_multiples_data_uri_str(
            panel_ser_dict, column_count_int=1
        ) == render_small_multiples_data_uri_str(
            panel_ser_dict, column_count_int=1, overlay_ser_dict=None
        )
