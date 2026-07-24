"""Render a real saved strategy run through the existing report under a variant.

Throwaway verification tool. Loads a completed strategy pickle and rebuilds its
HTML report with a signature variant active, so we can see the real report — all
of its sections, on real data — in the candidate look before changing report.py.

report.py freezes ``_CSS`` and the font head at import time, so this patches
those two module globals inside the variant context. That import-time binding is
exactly what the production change will replace with render-time resolution.

    uv run python scripts/dev/preview_real_report.py
    uv run python scripts/dev/preview_real_report.py --variant journal --pickle <path>
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from alpha.engine import report as report_module
from alpha.engine.strategy import Strategy
from alpha.engine.theme import signature_variant_context

_DEFAULT_PICKLE_PATH_STR = (
    'results/research/portfolio/current_multipod_all/vanilla_backtest/'
    '2026-05-08_174952/pods/pod_dv2/strategy_mr_dv2.pkl'
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', default=report_module._ACTIVE_REPORT_VARIANT_STR)
    parser.add_argument('--pickle', default=_DEFAULT_PICKLE_PATH_STR)
    parser.add_argument('--out', default='results/_theme_preview')
    parsed_args = parser.parse_args()

    pickle_path_obj = (REPO_ROOT_PATH / parsed_args.pickle).resolve()
    strategy_obj = Strategy.read_pickle(str(pickle_path_obj))

    # _build_html resolves CSS/fonts at render time, so activating the variant
    # here is all that is needed — no module-constant patching.
    with signature_variant_context(parsed_args.variant):
        chart_buffer = io.BytesIO()
        strategy_obj.plot(save_to=chart_buffer)
        plt.close('all')
        chart_buffer.seek(0)
        chart_b64_str = base64.b64encode(chart_buffer.read()).decode('ascii')

        report_html_str = report_module._build_html(strategy_obj, chart_b64_str)

    output_dir_path = REPO_ROOT_PATH / parsed_args.out
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_path_obj = output_dir_path / f'real_report_{parsed_args.variant}.html'
    output_path_obj.write_text(report_html_str, encoding='utf-8')
    print(f'wrote {output_path_obj}')


if __name__ == '__main__':
    main()
