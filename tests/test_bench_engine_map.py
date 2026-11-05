"""Tests for Bench's Strategy map (``/map``, ``alpha.bench.engine_map``).

The page draws in the browser from one data file, so the server side is small:
the page and its nav item, the data route (no-store, plain 404 when missing),
the refresh button's job (CSRF-gated, the documented command), and a CLI that
imports without running a single engine. No backtest and no build runs here.
"""

from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path

import pytest

from alpha.bench import engine_map
from alpha.bench.app import create_app
from alpha.bench.engine_map import pages


REPO_ROOT_PATH = Path(__file__).resolve().parents[1]
ENGINE_MAP_CSS_PATH = REPO_ROOT_PATH / "alpha" / "bench" / "static" / "engine_map.css"


class RecordingJobManager:
    def __init__(self, job_list=None) -> None:
        self.call_list: list[tuple[str, str, str, list[str]]] = []
        self.job_list = list(job_list or [])

    def submit(self, label_str, target_str, kind_str, command_list):
        self.call_list.append((label_str, target_str, kind_str, list(command_list)))

    def active_count(self) -> int:
        return 0

    def list_jobs(self):
        return list(self.job_list)

    def get_job(self, job_id_str):
        return None


@pytest.fixture()
def map_paths(monkeypatch, tmp_path):
    """Point the map at an empty folder: nothing the operator built is read or touched."""
    output_path = tmp_path / "engine_map"
    monkeypatch.setattr(engine_map, "DATA_PATH", output_path / "data.json")
    monkeypatch.setattr(engine_map, "BUILD_SUMMARY_PATH", output_path / "build_summary.json")
    return output_path


@pytest.fixture()
def map_client(map_paths):
    job_manager = RecordingJobManager()
    app = create_app(job_manager_obj=job_manager)
    return app.test_client(), job_manager, app.config["bench_token_str"], map_paths


def _write_built(output_path: Path) -> None:
    output_path.mkdir(parents=True)
    (output_path / "data.json").write_text(json.dumps({"asOf": "2026-10-30", "engines": []}), encoding="utf-8")
    (output_path / "build_summary.json").write_text(
        json.dumps({"as_of": "2026-10-30", "built": "2026-11-05T00:09:50", "warnings": []}), encoding="utf-8"
    )


def test_map_page_renders_with_its_nav_item_active(map_client):
    client, _job_manager, _token_str, output_path = map_client
    _write_built(output_path)
    response_obj = client.get("/map")
    html_str = response_obj.get_data(as_text=True)

    assert response_obj.status_code == 200
    assert "<span>Map</span>" in html_str
    assert re.search(r'<a href="/map" class="active">', html_str)
    assert 'data-src="/map/data.json"' in html_str
    assert "engine_map.js" in html_str and "engine_map.css" in html_str
    assert "Strategy map" in html_str and "Research control panel" in html_str
    assert "Last close Oct 30, 2026, updated Nov 5, 00:09" in html_str
    assert 'action="/api/map/refresh"' in html_str


def test_candidate_stage_has_its_own_count(map_client):
    client, _job_manager, _token_str, output_path = map_client
    _write_built(output_path)
    html_str = client.get("/map").get_data(as_text=True)
    assert 'id="n-cand"' in html_str
    assert ">soon<" not in html_str


def test_map_nav_item_sits_right_after_strategies_on_every_page(map_client):
    client, *_ = map_client
    html_str = client.get("/").get_data(as_text=True)
    labels = re.findall(r"<span>(Strategies|Map|Compare|Studies|Portfolios|Jobs)</span>", html_str)
    assert labels[:3] == ["Strategies", "Map", "Compare"]
    # the map's stylesheet is the map page's alone
    assert "engine_map.css" not in html_str


def test_map_page_without_data_says_so_and_offers_refresh(map_client):
    client, *_ = map_client
    html_str = client.get("/map").get_data(as_text=True)

    assert "The map has no data yet." in html_str
    assert 'action="/api/map/refresh"' in html_str
    assert "engine_map.js" not in html_str


def test_map_data_route_is_plain_404_when_missing(map_client):
    client, *_ = map_client
    response_obj = client.get("/map/data.json")

    assert response_obj.status_code == 404
    assert response_obj.mimetype == "text/plain"
    assert "no data yet" in response_obj.get_data(as_text=True)


def test_map_data_route_serves_the_built_file_uncached(map_client):
    client, _job_manager, _token_str, output_path = map_client
    _write_built(output_path)
    response_obj = client.get("/map/data.json")

    assert response_obj.status_code == 200
    assert response_obj.mimetype == "application/json"
    assert response_obj.headers["Cache-Control"] == "no-store"
    assert response_obj.get_json()["asOf"] == "2026-10-30"


def test_refresh_needs_the_csrf_token(map_client):
    client, job_manager, _token_str, _output_path = map_client
    assert client.post("/api/map/refresh", data={}).status_code == 403
    assert client.post("/api/map/refresh", data={"csrf_token": "wrong"}).status_code == 403
    assert job_manager.call_list == []


def test_refresh_starts_the_documented_command_as_a_job(map_client):
    client, job_manager, token_str, _output_path = map_client
    response_obj = client.post("/api/map/refresh", data={"csrf_token": token_str})

    assert response_obj.status_code == 302
    assert response_obj.headers["Location"].endswith("/jobs")
    assert len(job_manager.call_list) == 1
    label_str, target_str, kind_str, command_list = job_manager.call_list[0]
    assert (target_str, kind_str) == ("engine_map", "map")
    assert command_list == [sys.executable, "-X", "utf8", "-m", "alpha.bench.engine_map", "refresh"]


def test_running_refresh_replaces_the_button_with_a_jobs_link(map_paths):
    class _Job:
        kind_str = "map"
        is_active_bool = True

    client = create_app(job_manager_obj=RecordingJobManager([_Job()])).test_client()
    html_str = client.get("/map").get_data(as_text=True)
    assert "Refresh running" in html_str
    assert 'action="/api/map/refresh"' not in html_str


def test_engine_map_package_imports_without_running_anything():
    """The package, its CLI and its routes import without reading data or running an engine.

    build.py is a script (importing it is the build), so it is deliberately not imported here.
    """
    cli_module = importlib.import_module("alpha.bench.engine_map.__main__")
    assert engine_map.OUTPUT_DIR_PATH == REPO_ROOT_PATH / "results" / "bench" / "engine_map"
    assert engine_map.WORK_DIR_PATH.parent == engine_map.OUTPUT_DIR_PATH
    assert engine_map.MAP_BASE_PATH.exists()
    assert set(json.loads(engine_map.MAP_BASE_PATH.read_text(encoding="utf-8"))) >= {"taa3x", "ndx_vxn", "core5"}
    with pytest.raises(SystemExit):
        cli_module.main(["nonsense"])
    assert "norgatedata" not in sys.modules


def test_refresh_module_lists_every_engine_without_starting_one():
    refresh_module = importlib.import_module("alpha.bench.engine_map.refresh")
    assert refresh_module.WORK == engine_map.WORK_DIR_PATH
    assert {"taa3x", "ndx_vxn", "core5", "eom_vote"} <= set(refresh_module.RUNS)
    assert "norgatedata" not in sys.modules


def test_seed_copies_inputs_and_sessions_only(monkeypatch, tmp_path):
    cli_module = importlib.import_module("alpha.bench.engine_map.__main__")
    work_path = tmp_path / "work"
    monkeypatch.setattr(cli_module, "WORK_DIR_PATH", work_path)
    monkeypatch.setattr(cli_module, "DAILY_DIR_PATH", work_path / "daily")
    source_path = tmp_path / "old"
    (source_path / "daily").mkdir(parents=True)
    (source_path / "records.json").write_text("{}", encoding="utf-8")
    (source_path / "published_data.json").write_text("{}", encoding="utf-8")
    (source_path / "notes.md").write_text("x", encoding="utf-8")
    (source_path / "daily" / "core5.parquet").write_bytes(b"x")

    assert cli_module.main(["seed", "--from", str(source_path)]) == 0
    assert sorted(p.relative_to(work_path).as_posix() for p in work_path.rglob("*") if p.is_file()) == [
        "daily/core5.parquet", "records.json",
    ]


def test_map_stylesheet_is_scoped_and_owns_no_palette():
    """Every rule sits under .emap, and every colour is a Bench token, as bench.css's own guard requires of it."""
    css_str = ENGINE_MAP_CSS_PATH.read_text(encoding="utf-8")
    assert re.findall(r"#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?\b", re.sub(r"#(tip|scrim|drawer|corrbox)\b", "", css_str)) == []
    assert "prefers-color-scheme" not in css_str
    body_str = re.sub(r"/\*.*?\*/", "", css_str, flags=re.S)
    selector_list = []
    for selector_group_str in re.findall(r"([^{}]+)\{[^{}]*\}", re.sub(r"@media[^{]+\{|@supports[^{]+\{", "", body_str)):
        # split on top-level commas only: :is(a, b) is one selector
        selector_list.extend(part.strip() for part in re.split(r",(?![^()]*\))", selector_group_str))
    unscoped_list = [s for s in selector_list if s and ".emap" not in s]
    assert unscoped_list == []


def test_date_words_match_the_page():
    assert pages._date_words_str("2026-10-30", with_time_bool=False) == "Oct 30, 2026"
    assert pages._date_words_str("2026-11-05T00:09:50", with_time_bool=True) == "Nov 5, 00:09"
    assert pages._date_words_str(None, with_time_bool=True) is None
