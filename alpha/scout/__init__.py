"""Scout: the research pipeline in front of the real engine.

Design: docs/plans/SCOUT_DESIGN.md. Scout never trades and never produces a final
number; final numbers come from the real engine after an identity gate.

Import rule (enforced by tests/test_scout_import_boundary.py): `alpha.live` must
never import `alpha.scout`.
"""
