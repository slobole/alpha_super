"""Dashboard V3 — Flask-based operator console.

Single-operator dashboard for the multi-pod live trading book. Replaces the
React-based dashboard_v2 with server-rendered Jinja templates + HTMX. Designed
to be served from a VPS behind Tailscale (no public exposure).

Phase 0: skeleton + smoke test only. Real routes and data binding land in
Phase 1.
"""

from alpha.live.dashboard_v3.app import create_app

__all__ = ["create_app"]
