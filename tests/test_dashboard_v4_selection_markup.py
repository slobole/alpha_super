"""Copy selection must stay within the same rendered Pod, cycle and field."""

from html.parser import HTMLParser

import pytest

from alpha.live.dashboard_v4.demo import create_demo_app


class SelectionMarkupParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scope_str = ""
        self.key_list = []

    def handle_starttag(self, tag_str, attribute_list):
        attribute_dict = dict(attribute_list)
        if attribute_dict.get("id") == "overview-shell":
            self.scope_str = attribute_dict.get("data-selection-scope", "")
        if "data-selection-key" in attribute_dict:
            self.key_list.append(attribute_dict["data-selection-key"])


def _markup_obj(client_obj, route_str):
    response_obj = client_obj.get(route_str)
    assert response_obj.status_code == 200
    parser_obj = SelectionMarkupParser()
    parser_obj.feed(response_obj.get_data(as_text=True))
    return parser_obj


@pytest.mark.parametrize("route_str", ["/", "/pods/demo_1_0?tab=orders", "/pods/demo_1_1?tab=plan",
    "/pods/demo_1_1?cycle=vplan:1&tab=events", "/pods/demo_1_2?tab=fills", "/system"])
def test_copy_regions_are_unique_and_stable_across_refresh(route_str):
    client_obj = create_demo_app().test_client()
    initial_obj = _markup_obj(client_obj, route_str)
    path_str, separator_str, query_str = route_str.partition("?")
    refresh_str = "/overview/refresh" if path_str == "/" else path_str + "/refresh"
    refreshed_obj = _markup_obj(client_obj, refresh_str + (separator_str + query_str if separator_str else ""))
    assert initial_obj.scope_str and initial_obj.scope_str == refreshed_obj.scope_str
    assert initial_obj.key_list and all(initial_obj.key_list)
    assert len(initial_obj.key_list) == len(set(initial_obj.key_list))
    assert initial_obj.key_list == refreshed_obj.key_list


def test_different_pods_cycles_tabs_and_periods_cannot_share_selection_scope():
    client_obj = create_demo_app().test_client()
    route_list = ["/?period=All", "/?period=3M", "/pods/demo_1_0?cycle=vplan:1&tab=orders",
        "/pods/demo_1_0?cycle=vplan:2&tab=orders", "/pods/demo_1_0?cycle=vplan:1&tab=fills",
        "/pods/demo_1_1?cycle=vplan:1&tab=orders"]
    scope_list = [_markup_obj(client_obj, route_str).scope_str for route_str in route_list]
    assert len(scope_list) == len(set(scope_list))
