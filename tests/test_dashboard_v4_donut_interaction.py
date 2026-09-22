"""Donut inspection reuses the saved allocation labels, values and identity keys."""

from html.parser import HTMLParser
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from alpha.live.dashboard_v4.finance import _allocation_dict
from alpha.live.dashboard_v4.pod_allocation import build_pod_allocation_dict


class DonutParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.path_list = []
        self.svg_dict = {}
        self.readout_dict = {}
        self.inspector_dict = {}
        self.tag_list = []

    def handle_starttag(self, tag_str, attribute_list):
        attribute_dict = dict(attribute_list)
        self.tag_list.append(tag_str)
        if tag_str == "path":
            self.path_list.append(attribute_dict)
        elif tag_str == "svg":
            self.svg_dict = attribute_dict
        if "data-donut-value" in attribute_dict:
            self.readout_dict = attribute_dict
        if "data-donut-inspector" in attribute_dict:
            self.inspector_dict = attribute_dict


def _markup_obj(allocation_dict, *, pod_bool=False):
    environment_obj = Environment(loader=FileSystemLoader(Path(__file__).resolve().parents[1] / "alpha/live/dashboard_v4/templates"),
        autoescape=select_autoescape())
    markup_str = environment_obj.get_template("_donut.html").render(allocation_dict=allocation_dict, donut_interactive_bool=pod_bool)
    parser_obj = DonutParser()
    parser_obj.feed(markup_str)
    return parser_obj


def test_overview_every_saved_slice_including_cash_is_keyboard_inspectable():
    allocation_dict = _allocation_dict({"total_float": 100, "date_str": "2026-09-18", "cash_complete_bool": True, "cash_weight_float": .2,
        "item_list": [{"label_str": "Long strategy name", "value_float": 100, "weight_float": 1,
            "cash_weight_float": .2, "invested_weight_float": .8}]})
    parser_obj = _markup_obj(allocation_dict)
    assert parser_obj.svg_dict["viewbox"] == "0 0 300 250" and parser_obj.svg_dict["role"] == "group"
    assert parser_obj.inspector_dict["data-donut-id"] == "overview-allocation"
    assert parser_obj.inspector_dict["data-donut-date"] == "2026-09-18"
    assert len(parser_obj.path_list) == 3
    assert [path_dict["data-donut-readout"] for path_dict in parser_obj.path_list] == [
        "Cash 20.0%", "Long strategy name invested 80.0%", "Long strategy name cash 20.0%"]
    for path_dict, slice_dict in zip(parser_obj.path_list, allocation_dict["slice_list"]):
        assert path_dict["tabindex"] == "0" and path_dict["role"] == "img"
        assert path_dict["aria-label"] == slice_dict["label_str"]
        assert path_dict["data-donut-key"] == slice_dict["label_str"]
        assert path_dict["fill"] == slice_dict["color_str"] and path_dict["d"] == slice_dict["path_str"]
        assert "$" not in path_dict["data-donut-readout"]  # No reconstructed dollar valuation.
        assert "data-allocation-key" not in path_dict
    assert parser_obj.readout_dict["role"] == "status" and parser_obj.readout_dict["aria-atomic"] == "true"


def test_pod_readout_reuses_saved_values_and_keeps_table_identity_keys():
    allocation_dict = build_pod_allocation_dict({"available_bool": True, "close_date_str": "2026-09-18", "nav_float": 1250,
        "cash_float": 250, "position_list": [{"symbol_str": "ABC", "shares_float": 10, "value_float": 1000}]}, color_str="#123456")
    parser_obj = _markup_obj(allocation_dict, pod_bool=True)
    assert parser_obj.svg_dict["viewbox"] == "6 15 220 220"
    assert "data-donut-id" not in parser_obj.inspector_dict
    assert all("data-donut-key" not in path_dict for path_dict in parser_obj.path_list)
    assert [path_dict["data-donut-readout"] for path_dict in parser_obj.path_list] == ["ABC · 80.0% · $1,000.00", "Cash · 20.0% · $250.00"]
    assert [path_dict["data-allocation-key"] for path_dict in parser_obj.path_list] == ["position:ABC", "cash"]
    assert [path_dict["fill"] for path_dict in parser_obj.path_list] == ["#123456", "#dfe3e9"]


def test_missing_cash_split_does_not_gain_a_cash_slice_or_invent_a_value():
    allocation_dict = _allocation_dict({"total_float": 100, "cash_complete_bool": False,
        "item_list": [{"label_str": "First strategy", "value_float": 100, "weight_float": 1}]})
    parser_obj = _markup_obj(allocation_dict)
    assert len(parser_obj.path_list) == 1
    assert parser_obj.path_list[0]["data-donut-readout"] == "First strategy value 100.0%"
    assert parser_obj.svg_dict["aria-label"] == "Allocation · Cash —"


def test_readout_labels_are_escaped_as_text_not_markup():
    label_str = '<img src=x onerror="alert(1)"> & Cash'
    parser_obj = _markup_obj({"cash_percent_str": "0.0%", "label_list": [], "slice_list": [
        {"label_str": label_str, "color_str": "#123456", "path_str": "M 0,0 Z"}]})
    assert parser_obj.path_list[0]["data-donut-readout"] == label_str
    assert "img" not in parser_obj.tag_list and "script" not in parser_obj.tag_list
