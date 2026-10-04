"""Shared research statistics for Scout, Pakal studies and the pod-health report.

Every function here is post-run analysis: it reads return series that already
exist and never feeds signal, sizing, or order logic.

Import rule (enforced by tests/test_scout_import_boundary.py): modules in this
package depend only on the standard library, numpy, pandas and scipy. They never
import `alpha.scout`, `alpha.engine` or `alpha.live`, so any of those layers can
use them without pulling in research or reporting code.
"""
