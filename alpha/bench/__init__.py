"""Bench — a local research control panel for the alpha_super quant stack.

Bench is a thin, read-mostly web UI that sits on top of the existing research
CLIs. It does not implement any quant logic of its own. It only:

  * lists the strategies that live under ``strategies/`` and flags which ones
    are *wired* (live / supported) via ``SUPPORTED_STRATEGY_IMPORT_TUPLE``,
  * surfaces the most recent analyzer runs from the ``results/`` tree,
  * launches the existing ``run_strategy_analysis.py`` / ``run_portfolio.py``
    scripts as tracked background jobs when you click a button.

Because every heavy operation is delegated to a script that already exists,
the backend stays light: discovery + a job runner + a results reader.

Run it with::

    uv run python -m alpha.bench            # http://127.0.0.1:8765
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
