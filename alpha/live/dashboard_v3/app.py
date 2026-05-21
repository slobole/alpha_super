"""Flask application factory for Dashboard V3.

Phase 0 only wires the smallest possible app: one health-check route that
returns plain text. Subsequent phases will add the real routes, templates,
and bindings to ``alpha.live.dashboard.*`` data builders.
"""

from __future__ import annotations

from flask import Flask


DASHBOARD_V3_VERSION_STR = "0.0.1-phase-0"


def create_app() -> Flask:
    flask_app_obj = Flask(__name__)

    @flask_app_obj.route("/healthz")
    def healthz_route_fn() -> tuple[str, int]:
        return (f"dashboard_v3 ok {DASHBOARD_V3_VERSION_STR}", 200)

    @flask_app_obj.route("/")
    def index_route_fn() -> tuple[str, int]:
        return (
            "Dashboard V3 skeleton — Phase 0. "
            "Phase 1 will replace this with the live operator console.",
            200,
        )

    return flask_app_obj
