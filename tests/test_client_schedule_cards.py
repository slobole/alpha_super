"""Independent saved schedules, never regrouped clocks or operational changes."""

from copy import deepcopy

from flask import render_template
import pytest

from alpha.live.dashboard_v3.app import create_app
from alpha.live.dashboard_v3.client_operations import build_client_operations_dict
from alpha.live.dashboard_v3.schedule import TradingWindow, build_trading_window_list
from test_client_operations import AS_OF_TS, fixture_tuple
from test_dashboard_operator_access import ForbiddenProvider


def render_schedule_str(operations_dict):
    app_obj = create_app(ForbiddenProvider(), read_only_bool=True)
    with app_obj.test_request_context():
        return render_template('_client_operations_summary.html', operations_dict=operations_dict, view_str='overview')


def test_each_strategy_retains_its_independent_schedule_and_exact_timestamps():
    registry_dict, _, _, summary_dict = fixture_tuple()
    original_dict = deepcopy(summary_dict)
    result_dict = build_client_operations_dict(registry_dict['clients'][0], summary_dict, as_of_ts=AS_OF_TS)
    for strategy_dict in result_dict['strategy_list']:
        expected_list = [window_obj.as_dict() for window_obj in build_trading_window_list(
            {'pod_row_dict_list': [strategy_dict['evidence_dict']]}, mode_str='live', now_dt=AS_OF_TS)]
        assert strategy_dict['trading_window_list'] == expected_list
        assert expected_list[0]['pod_id_str_list'] == [strategy_dict['pod_id_str']]
    assert summary_dict == original_dict
    html_str = render_schedule_str(result_dict)
    assert html_str.count('client-schedule-card') == 2
    assert html_str.count('<time datetime=') == 6
    assert 'live pods share this window' not in html_str


def test_shared_clocks_do_not_replace_each_pods_status_or_action(monkeypatch):
    registry_dict, _, _, summary_dict = fixture_tuple()
    first_pod_str, second_pod_str = [account_dict['pod_id'] for account_dict in registry_dict['clients'][0]['accounts']]

    def independent_window_list(summary_dict, **kwargs_dict):
        pod_list = [row_dict['pod_id_str'] for row_dict in summary_dict['pod_row_dict_list']]
        first_bool = pod_list == [first_pod_str]
        return [TradingWindow(has_data_bool=True, pod_id_str_list=pod_list,
            status_label_str='Submit pending' if first_bool else 'No action' if len(pod_list) == 1 else 'Shared window',
            severity_str='yellow' if first_bool else 'gray', action_required_bool=first_bool,
            action_str='submit_vplan' if first_bool else None,
            signal_timestamp_str='2026-09-30T20:00:00+00:00',
            submission_timestamp_str='2026-10-01T13:23:30+00:00',
            target_timestamp_str='2026-10-01T13:30:00+00:00')]

    monkeypatch.setattr('alpha.live.dashboard_v3.client_operations.build_trading_window_list', independent_window_list)
    result_dict = build_client_operations_dict(registry_dict['clients'][0], summary_dict, as_of_ts=AS_OF_TS)
    first_dict, second_dict = result_dict['strategy_list']
    assert result_dict['trading_window_list'][0]['status_label_str'] == 'Shared window'
    assert first_dict['trading_window_list'][0]['action_str'] == 'submit_vplan'
    assert second_dict['trading_window_list'][0]['pod_id_str_list'] == [second_pod_str]
    assert not second_dict['trading_window_list'][0]['action_required_bool']
    html_str = render_schedule_str(result_dict)
    assert 'Submit pending' in html_str and 'Shared window' not in html_str


@pytest.mark.parametrize('failure_str', ['identity', 'calendar'])
def test_unknown_strategy_keeps_its_card_without_hiding_healthy_peer(failure_str):
    registry_dict, _, _, summary_dict = fixture_tuple()
    row_dict = summary_dict['pod_row_dict_list'][0]
    row_dict['account_route_str' if failure_str == 'identity' else 'session_calendar_id_str'] = 'UNKNOWN'
    result_dict = build_client_operations_dict(registry_dict['clients'][0], summary_dict, as_of_ts=AS_OF_TS)
    unknown_dict, healthy_dict = result_dict['strategy_list']
    assert unknown_dict['trading_window_list'][0]['status_label_str'] == 'Cannot verify'
    assert unknown_dict['trading_window_list'][0]['target_timestamp_str'] is None
    assert healthy_dict['trading_window_list'][0]['has_data_bool']
    html_str = render_schedule_str(result_dict)
    assert html_str.count('client-schedule-card') == 2
    assert html_str.count('<time datetime=') == 3
    assert html_str.count('<time >—</time>') == 3


def test_retired_strategy_is_not_given_a_current_schedule():
    registry_dict, _, _, summary_dict = fixture_tuple()
    retired_dict = registry_dict['clients'][0]['accounts'][0]
    retired_dict['effective_to'] = '2026-09-04'
    result_dict = build_client_operations_dict(registry_dict['clients'][0], summary_dict, as_of_ts=AS_OF_TS)
    html_str = render_schedule_str(result_dict)
    assert html_str.count('client-schedule-card') == 1
    assert f'data-pod="{retired_dict["pod_id"]}"' not in html_str
