"""Smoke tests for Dashboard V3 — Phase 0 only verifies that the Flask app
boots and that the placeholder routes respond. Real route coverage is added
alongside the templates in Phase 1.
"""

from __future__ import annotations

import pytest

from alpha.live.dashboard_v3.app import (
    DASHBOARD_V3_VERSION_STR,
    create_app,
)


@pytest.fixture(name="test_client_obj")
def fixture_test_client_obj():
    flask_app_obj = create_app()
    flask_app_obj.config["TESTING"] = True
    with flask_app_obj.test_client() as test_client_obj:
        yield test_client_obj


def test_healthz_route_returns_version_marker(test_client_obj) -> None:
    response_obj = test_client_obj.get("/healthz")
    assert response_obj.status_code == 200
    response_text_str = response_obj.get_data(as_text=True)
    assert "dashboard_v3 ok" in response_text_str
    assert DASHBOARD_V3_VERSION_STR in response_text_str


def test_index_route_returns_phase_0_placeholder(test_client_obj) -> None:
    response_obj = test_client_obj.get("/")
    assert response_obj.status_code == 200
    response_text_str = response_obj.get_data(as_text=True)
    assert "Phase 0" in response_text_str


def test_unknown_route_returns_404(test_client_obj) -> None:
    response_obj = test_client_obj.get("/this-route-does-not-exist")
    assert response_obj.status_code == 404
